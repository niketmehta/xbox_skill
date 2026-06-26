import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from config import Config
import pytz
import re

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

_yf = None


def _get_yfinance():
    global _yf
    if _yf is None:
        import yfinance as yf
        _yf = yf
    return _yf


class MarketDataProvider:
    """
    Provides real-time and historical market data from multiple sources.
    Supports daily (swing/week) and weekly (position/month) timeframes.
    Also provides market-regime (VIX) and relative-strength data.

    Includes an in-memory quote cache to avoid Yahoo Finance rate limits.
    """

    # Cache TTLs (seconds)
    QUOTE_CACHE_TTL = 120        # 2 minutes for individual quotes
    MOVERS_CACHE_TTL = 180       # 3 minutes for market movers
    VIX_CACHE_TTL = 300          # 5 minutes for VIX / market regime
    FUNDAMENTALS_CACHE_TTL = 600 # 10 minutes for fundamentals
    HISTORY_CACHE_TTL = 900      # 15 minutes for daily / weekly candles

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.eastern_tz = pytz.timezone('US/Eastern')
        self._http = requests.Session()
        self._http.headers.update({
            'User-Agent': 'Mozilla/5.0 trading-agent/1.0'
        })

        # ── Caches ───────────────────────────────────────────
        # Quote cache:  symbol -> {data: dict, ts: datetime}
        self._quote_cache: Dict[str, Dict] = {}
        # Market movers cache
        self._movers_cache: Optional[List[Dict]] = None
        self._movers_cache_time: Optional[datetime] = None
        # VIX / market regime cache
        self._regime_cache: Optional[Dict] = None
        self._regime_cache_time: Optional[datetime] = None
        # Fundamentals cache
        self._fundamentals_cache: Dict[str, Dict] = {}
        # SPY data for relative-strength calculations
        self._spy_cache: Dict[str, pd.DataFrame] = {}
        self._spy_cache_time: Dict[str, datetime] = {}
        # Historical candle cache: (source, symbol, period, interval) -> {data, ts}
        self._history_cache: Dict[Tuple[str, str, str, str], Dict] = {}
        self._yahoo_blocked_until: Optional[datetime] = None
        self._alpaca_api = None

    def _cache_valid(self, ts: Optional[datetime], ttl: float) -> bool:
        """Check if a cached entry is still valid."""
        if ts is None:
            return False
        return (datetime.now() - ts).total_seconds() < ttl

    def _is_rate_limit_error(self, e):
        msg = str(e)
        patterns = [
            r"429", r"rate limit", r"too many requests",
            r"temporarily blocked", r"try again later"
        ]
        return any(re.search(p, msg, re.IGNORECASE) for p in patterns)

    def _prefer_yahoo(self) -> bool:
        if not self.config.USE_YAHOO_FINANCE:
            return False
        if self.config.MARKET_DATA_PRIMARY in ('stooq', 'alpaca', 'alpha_vantage'):
            return False
        if self._yahoo_blocked_until and datetime.now() < self._yahoo_blocked_until:
            return False
        return True

    def _prefer_alpaca(self) -> bool:
        return (
            self.config.USE_ALPACA_MARKET_DATA
            and self.config.MARKET_DATA_PRIMARY == 'alpaca'
            and bool(self.config.ALPACA_API_KEY)
            and bool(self.config.ALPACA_SECRET_KEY)
        )

    def _mark_yahoo_rate_limited(self, reason: str):
        cooldown = max(self.config.YAHOO_COOLDOWN_SECONDS, 60)
        self._yahoo_blocked_until = datetime.now() + timedelta(seconds=cooldown)
        self.logger.warning(
            "Yahoo rate-limited/unavailable; using fallback for %s seconds. Reason: %s",
            cooldown,
            reason,
        )

    def _history_cache_get(self, source: str, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        key = (source, symbol.upper(), period, interval)
        cached = self._history_cache.get(key)
        if cached and self._cache_valid(cached.get('ts'), self.HISTORY_CACHE_TTL):
            return cached.get('data')
        return None

    def _history_cache_set(self, source: str, symbol: str, period: str, interval: str, data: pd.DataFrame):
        key = (source, symbol.upper(), period, interval)
        self._history_cache[key] = {'data': data, 'ts': datetime.now()}

    def _stooq_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().lower().replace('-', '.')
        if symbol.startswith('^'):
            return symbol[1:]
        if symbol in ('spy', 'qqq', 'iwm', 'dia') or re.match(r'^[a-z.]+$', symbol):
            return f"{symbol}.us"
        return symbol

    def _period_start(self, period: str) -> datetime:
        period = (period or '3mo').lower()
        amount = int(re.sub(r'\D', '', period) or 1)
        if period.endswith('d'):
            return datetime.now() - timedelta(days=amount + 5)
        if period.endswith('mo'):
            return datetime.now() - timedelta(days=amount * 31 + 10)
        if period.endswith('y'):
            return datetime.now() - timedelta(days=amount * 366 + 10)
        return datetime.now() - timedelta(days=120)

    def _get_alpaca_api(self):
        if self._alpaca_api is not None:
            return self._alpaca_api
        if tradeapi is None:
            self.logger.warning("alpaca-trade-api is not installed")
            return None
        if not self.config.ALPACA_API_KEY or not self.config.ALPACA_SECRET_KEY:
            return None

        self._alpaca_api = tradeapi.REST(
            key_id=self.config.ALPACA_API_KEY,
            secret_key=self.config.ALPACA_SECRET_KEY,
            base_url=self.config.ALPACA_BASE_URL,
            api_version='v2',
        )
        return self._alpaca_api

    def _alpaca_timeframe(self, interval: str):
        if tradeapi is None:
            return None
        interval = (interval or '1d').lower()
        if interval in ('1wk', '1w', 'wk'):
            return tradeapi.TimeFrame.Week
        if interval in ('1d', 'd', 'day'):
            return tradeapi.TimeFrame.Day
        match = re.match(r'^(\d+)m$', interval)
        if match:
            return tradeapi.TimeFrame(int(match.group(1)), tradeapi.TimeFrameUnit.Minute)
        if interval in ('1h', '60m'):
            return tradeapi.TimeFrame.Hour
        return tradeapi.TimeFrame.Day

    def _alpaca_symbol(self, symbol: str) -> str:
        return symbol.strip().upper().replace('-', '.')

    def _fetch_alpaca_history(self, symbol: str, period: str = '3mo', interval: str = '1d') -> pd.DataFrame:
        cached = self._history_cache_get('alpaca', symbol, period, interval)
        if cached is not None:
            return cached.copy()
        if symbol.startswith('^'):
            return pd.DataFrame()

        api = self._get_alpaca_api()
        timeframe = self._alpaca_timeframe(interval)
        if api is None or timeframe is None:
            return pd.DataFrame()

        start = self._period_start(period).strftime('%Y-%m-%dT%H:%M:%SZ')
        alpaca_symbol = self._alpaca_symbol(symbol)
        try:
            kwargs = {
                'start': start,
                'adjustment': 'all',
            }
            if self.config.ALPACA_DATA_FEED:
                kwargs['feed'] = self.config.ALPACA_DATA_FEED
            bars = api.get_bars(alpaca_symbol, timeframe, **kwargs)
            data = bars.df
            if data is None or data.empty:
                return pd.DataFrame()

            if 'symbol' in data.columns:
                data = data[data['symbol'] == alpaca_symbol]
            rename = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }
            data = data.rename(columns=rename)
            keep = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in data.columns]
            data = data[keep].copy()
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            if getattr(data.index, 'tz', None) is not None:
                data.index = data.index.tz_convert(None)
            data = data.sort_index()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            data['Volume'] = data['Volume'].fillna(0)
            if not data.empty:
                self._history_cache_set('alpaca', symbol, period, interval, data)
            return data.copy()
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_stooq_history(self, symbol: str, period: str = '3mo', interval: str = '1d') -> pd.DataFrame:
        cached = self._history_cache_get('stooq', symbol, period, interval)
        if cached is not None:
            return cached.copy()
        if not self.config.STOOQ_API_KEY:
            self.logger.debug("Stooq fallback skipped because STOOQ_API_KEY is not configured")
            return pd.DataFrame()

        start = self._period_start(period).strftime('%Y%m%d')
        end = datetime.now().strftime('%Y%m%d')
        stooq_interval = 'w' if interval in ('1wk', '1w', 'wk') else 'd'
        stooq_symbol = self._stooq_symbol(symbol)
        url = 'https://stooq.com/q/d/l/'
        params = {
            's': stooq_symbol,
            'd1': start,
            'd2': end,
            'i': stooq_interval,
            'apikey': self.config.STOOQ_API_KEY,
        }

        try:
            response = self._http.get(url, params=params, timeout=15)
            response.raise_for_status()
            text = response.text.strip()
            if not text or text.lower().startswith('no data'):
                self.logger.warning(f"No Stooq data returned for {symbol}")
                return pd.DataFrame()

            from io import StringIO
            data = pd.read_csv(StringIO(text))
            if data.empty or 'Date' not in data.columns:
                return pd.DataFrame()
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date').sort_index()
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            data['Volume'] = data['Volume'].fillna(0)
            self._history_cache_set('stooq', symbol, period, interval, data)
            return data.copy()
        except Exception as e:
            self.logger.error(f"Error fetching Stooq data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_yahoo_history(self, symbol: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
        cached = self._history_cache_get('yahoo', symbol, period, interval)
        if cached is not None:
            return cached.copy()
        ticker = _get_yfinance().Ticker(symbol)
        data = ticker.history(period=period, interval=interval, prepost=prepost)
        if not data.empty:
            self._history_cache_set('yahoo', symbol, period, interval, data)
        return data

    def _get_history(self, symbol: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
        if self._prefer_alpaca():
            data = self._fetch_alpaca_history(symbol, period=period, interval=interval)
            if not data.empty:
                return data

        if self.config.MARKET_DATA_PRIMARY == 'stooq' and self.config.USE_STOOQ_FALLBACK:
            data = self._fetch_stooq_history(symbol, period=period, interval=interval)
            if not data.empty:
                return data

        if self._prefer_yahoo():
            try:
                data = self._fetch_yahoo_history(symbol, period, interval, prepost=prepost)
                if not data.empty:
                    return data
            except Exception as e:
                if self._is_rate_limit_error(e):
                    self._mark_yahoo_rate_limited(str(e))
                else:
                    self.logger.error(f"Error getting Yahoo history for {symbol}: {e}")

        if self.config.USE_ALPACA_MARKET_DATA:
            data = self._fetch_alpaca_history(symbol, period=period, interval=interval)
            if not data.empty:
                return data

        if self.config.USE_STOOQ_FALLBACK:
            return self._fetch_stooq_history(symbol, period=period, interval=interval)

        return pd.DataFrame()

    # ── Real-time quote (cached) ─────────────────────────────────────
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol (with 2-minute cache)."""
        # Check cache first
        cached = self._quote_cache.get(symbol)
        if cached and self._cache_valid(cached.get('_cache_ts'), self.QUOTE_CACHE_TTL):
            return cached

        if self._prefer_alpaca():
            quote = self._get_alpaca_quote(symbol)
            if quote:
                self._quote_cache[symbol] = quote
                return quote

        if self._prefer_yahoo():
            try:
                ticker = _get_yfinance().Ticker(symbol)
                info = ticker.info

                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                previous_close = info.get('previousClose', 0)

                quote = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'change': current_price - previous_close if current_price and previous_close else 0,
                    'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'timestamp': datetime.now(),
                    '_cache_ts': datetime.now(),
                    'source': 'yahoo',
                }

                self._quote_cache[symbol] = quote
                return quote

            except Exception as e:
                if self._is_rate_limit_error(e):
                    self.logger.error(f"RATE LIMIT: quote for {symbol}: {e}")
                    self._mark_yahoo_rate_limited(str(e))
                    if cached:
                        self.logger.info(f"Returning stale cache for {symbol}")
                        return cached
                else:
                    self.logger.error(f"Error getting quote for {symbol}: {e}")

        if self.config.USE_ALPACA_MARKET_DATA:
            quote = self._get_alpaca_quote(symbol)
            if quote:
                self._quote_cache[symbol] = quote
                return quote

        if self.config.USE_STOOQ_FALLBACK:
            quote = self._get_stooq_quote(symbol)
            if quote:
                self._quote_cache[symbol] = quote
                return quote

        return {}

    def _get_alpaca_quote(self, symbol: str) -> Dict:
        """Get the latest available quote-like snapshot from Alpaca bars."""
        data = self._fetch_alpaca_history(symbol, period='10d', interval='1d')
        return self._quote_from_history(symbol, data, source='alpaca') if not data.empty else {}

    def _get_stooq_quote(self, symbol: str) -> Dict:
        """Get a delayed quote from Stooq daily candles."""
        try:
            data = self._fetch_stooq_history(symbol, period='10d', interval='1d')
            if data.empty:
                return {}
            last = data.iloc[-1]
            prev = data.iloc[-2] if len(data) >= 2 else last
            current_price = float(last['Close'])
            previous_close = float(prev['Close'])
            volume = int(last.get('Volume', 0) or 0)
            return {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': current_price - previous_close if previous_close else 0,
                'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                'volume': volume,
                'avg_volume': int(data['Volume'].tail(20).mean()) if 'Volume' in data.columns and not data.empty else 0,
                'market_cap': 0,
                'bid': 0,
                'ask': 0,
                'timestamp': datetime.now(),
                '_cache_ts': datetime.now(),
                'source': 'stooq',
            }
        except Exception as e:
            self.logger.error(f"Error getting Stooq quote for {symbol}: {e}")
            return {}

    def _get_alpha_vantage_quote(self, symbol: str) -> Dict:
        """Optional Alpha Vantage quote fallback, limited by free API quotas."""
        if not self.config.ALPHA_VANTAGE_API_KEY:
            return {}
        try:
            response = self._http.get(
                'https://www.alphavantage.co/query',
                params={
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.config.ALPHA_VANTAGE_API_KEY,
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            quote = payload.get('Global Quote', {})
            if not quote:
                return {}
            current_price = float(quote.get('05. price') or 0)
            previous_close = float(quote.get('08. previous close') or 0)
            volume = int(float(quote.get('06. volume') or 0))
            return {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': current_price - previous_close if previous_close else 0,
                'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                'volume': volume,
                'avg_volume': 0,
                'market_cap': 0,
                'bid': 0,
                'ask': 0,
                'timestamp': datetime.now(),
                '_cache_ts': datetime.now(),
                'source': 'alpha_vantage',
            }
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage quote for {symbol}: {e}")
            return {}

    def get_real_time_quote_alpha(self, symbol: str) -> Dict:
        return self._get_alpha_vantage_quote(symbol)

    # ── Daily data (for WEEK / swing trading) ───────────────────────
    def get_daily_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get daily candles for swing-trade / weekly analysis."""
        try:
            data = self._get_history(symbol, period=period, interval="1d")

            if data.empty:
                self.logger.warning(f"No daily data returned for {symbol}")
                return pd.DataFrame()

            data = self._add_technical_indicators(data)
            return data

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: daily data for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting daily data for {symbol}: {e}")
            return pd.DataFrame()

    # ── Weekly data (for MONTH / position trading) ──────────────────
    def get_weekly_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get weekly candles for position-trade / monthly analysis."""
        try:
            data = self._get_history(symbol, period=period, interval="1wk")

            if data.empty:
                self.logger.warning(f"No weekly data returned for {symbol}")
                return pd.DataFrame()

            data = self._add_technical_indicators(data)
            return data

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: weekly data for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting weekly data for {symbol}: {e}")
            return pd.DataFrame()

    # ── Intraday data (kept for chart display, not used for signals) ─
    def get_intraday_data(self, symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """Get intraday data (for chart display only, not for trading signals)."""
        try:
            data = self._get_history(symbol, period=period, interval=interval, prepost=True)
            if data.empty:
                return pd.DataFrame()
            data = self._add_technical_indicators(data)
            return data
        except Exception as e:
            self.logger.error(f"Error getting intraday data for {symbol}: {e}")
            return pd.DataFrame()

    # ── Market regime (VIX) — cached for 5 minutes ─────────────────
    def get_market_regime(self) -> Dict:
        """Fetch VIX and classify market regime (cached 5 min).

        Returns:
            dict with keys: vix, regime ('LOW_VOL', 'NORMAL', 'HIGH_VOL', 'EXTREME_FEAR'),
            position_multiplier (0.0–1.0)
        """
        if (self._regime_cache is not None and
                self._cache_valid(self._regime_cache_time, self.VIX_CACHE_TTL)):
            return self._regime_cache

        default = {'vix': 20, 'regime': 'NORMAL', 'position_multiplier': 0.8}
        try:
            vix_data = self._get_history('^VIX', period='5d', interval='1d')
            if vix_data.empty:
                return self._regime_cache or default

            vix = float(vix_data['Close'].iloc[-1])

            if vix < self.config.VIX_LOW:
                regime = 'LOW_VOL'
                mult = 1.0
            elif vix < self.config.VIX_MODERATE:
                regime = 'NORMAL'
                mult = 0.8
            elif vix < self.config.VIX_HIGH:
                regime = 'HIGH_VOL'
                mult = 0.5
            else:
                regime = 'EXTREME_FEAR'
                mult = 0.25

            result = {'vix': vix, 'regime': regime, 'position_multiplier': mult}
            self._regime_cache = result
            self._regime_cache_time = datetime.now()
            return result

        except Exception as e:
            self.logger.error(f"Error fetching VIX: {e}")
            return self._regime_cache or default

    # ── Relative strength vs SPY ────────────────────────────────────
    def get_relative_strength(self, symbol: str, period: str = '3mo') -> Dict:
        """Compare a stock's recent performance to SPY.

        Returns:
            dict with keys: stock_return, spy_return, relative_strength, outperforming (bool)
        """
        try:
            # Get or cache SPY data (refresh every 30 min)
            cache_key = period
            now = datetime.now()
            if (cache_key not in self._spy_cache or
                    (now - self._spy_cache_time.get(cache_key, datetime.min)).total_seconds() > 1800):
                self._spy_cache[cache_key] = self._get_history('SPY', period=period, interval='1d')
                self._spy_cache_time[cache_key] = now

            spy_data = self._spy_cache[cache_key]
            if spy_data.empty:
                return {}

            stock_data = self._get_history(symbol, period=period, interval='1d')
            if stock_data.empty or len(stock_data) < 5:
                return {}

            stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100
            rs = stock_return - spy_return

            return {
                'stock_return': float(stock_return),
                'spy_return': float(spy_return),
                'relative_strength': float(rs),
                'outperforming': rs > 0
            }

        except Exception as e:
            self.logger.error(f"Error computing relative strength for {symbol}: {e}")
            return {}

    # ── Earnings calendar ───────────────────────────────────────────
    def get_next_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Return the next earnings date for a symbol, or None."""
        if not self._prefer_yahoo():
            return None
        try:
            ticker = _get_yfinance().Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # yfinance returns a DataFrame with 'Earnings Date' rows
                if isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
                    dates = cal.loc['Earnings Date']
                    if hasattr(dates, '__len__') and len(dates) > 0:
                        return pd.Timestamp(dates.iloc[0]).to_pydatetime()
                    return pd.Timestamp(dates).to_pydatetime()
            # Alternative: check .earnings_dates
            if hasattr(ticker, 'earnings_dates') and ticker.earnings_dates is not None:
                upcoming = ticker.earnings_dates
                if not upcoming.empty:
                    future_dates = upcoming.index[upcoming.index >= pd.Timestamp.now()]
                    if len(future_dates) > 0:
                        return future_dates[0].to_pydatetime()
            return None
        except Exception as e:
            self.logger.debug(f"Could not get earnings date for {symbol}: {e}")
            return None

    # ── Market movers (cached, batch-downloaded) ───────────────────
    MOVERS_SYMBOLS = [
        'SNDK', 'WDC', 'STX', 'MU', 'MRVL', 'SMCI', 'ARM', 'DELL',
        'PSTG', 'NTAP', 'AVGO', 'AMD', 'NVDA', 'QCOM', 'ADI', 'LRCX',
        'KLAC', 'AMAT', 'TSM', 'ASML', 'SPY', 'QQQ', 'AAPL', 'TSLA',
        'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NFLX', 'BABA', 'DIS',
        'V', 'JPM', 'BAC', 'XOM', 'JNJ', 'PG', 'KO', 'PFE', 'T', 'VZ',
        'WMT', 'HD', 'UNH', 'CVX', 'LLY', 'ABBV', 'CRM', 'ORCL', 'ACN',
        'TMO', 'COST', 'ABT', 'DHR', 'TXN', 'NEE', 'LIN', 'NKE',
        'CMCSA', 'INTC', 'COP', 'HON', 'UPS', 'LOW', 'IBM', 'CSCO',
        'NOW', 'PANW', 'CRWD', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'RBLX'
    ]

    def get_market_movers(self, count: int = 25) -> List[Dict]:
        """Get top market movers — cached for 3 minutes, batch-downloaded."""
        # Return cache if fresh
        if (self._movers_cache is not None and
                self._cache_valid(self._movers_cache_time, self.MOVERS_CACHE_TTL)):
            return self._movers_cache[:count]

        try:
            scan_count = min(len(self.MOVERS_SYMBOLS), max(int(count or 25), 100))
            symbols = list(dict.fromkeys(self.MOVERS_SYMBOLS))[:scan_count]
            if self._prefer_yahoo():
                try:
                    # Use one batch HTTP request instead of N calls
                    batch = _get_yfinance().download(
                        symbols,
                        period='2d',
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        threads=True,
                    )
                except Exception as e:
                    if self._is_rate_limit_error(e):
                        self._mark_yahoo_rate_limited(str(e))
                    else:
                        self.logger.error(f"Error getting Yahoo market movers: {e}")
                    batch = pd.DataFrame()
            else:
                batch = pd.DataFrame()

            movers = []
            if not batch.empty:
                for sym in symbols:
                    try:
                        if len(symbols) == 1:
                            df = batch
                        else:
                            df = batch[sym] if sym in batch.columns.get_level_values(0) else None
                        quote = self._quote_from_history(sym, df, source='yahoo_batch')
                        if quote:
                            self._quote_cache[sym] = quote
                            movers.append(quote)
                    except Exception:
                        continue

            if not movers and self.config.USE_ALPACA_MARKET_DATA:
                for sym in symbols:
                    quote = self._get_alpaca_quote(sym)
                    if quote:
                        self._quote_cache[sym] = quote
                        movers.append(quote)

            if not movers and self.config.USE_STOOQ_FALLBACK:
                for sym in symbols:
                    quote = self._get_stooq_quote(sym)
                    if quote:
                        self._quote_cache[sym] = quote
                        movers.append(quote)

            movers.sort(key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            self._movers_cache = movers
            self._movers_cache_time = datetime.now()
            return movers[:count]

        except Exception as e:
            self.logger.error(f"Error getting market movers: {e}")
            # Return stale cache if available
            if self._movers_cache is not None:
                return self._movers_cache[:count]
            return []

    def _quote_from_history(self, symbol: str, df: Optional[pd.DataFrame], source: str) -> Dict:
        if df is None or df.empty or len(df) < 2:
            return {}
        current_price = float(df['Close'].iloc[-1])
        previous_close = float(df['Close'].iloc[-2])
        volume = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns and not pd.isna(df['Volume'].iloc[-1]) else 0
        if previous_close == 0:
            return {}
        change = current_price - previous_close
        change_pct = (change / previous_close) * 100
        return {
            'symbol': symbol,
            'current_price': current_price,
            'previous_close': previous_close,
            'change': change,
            'change_percent': change_pct,
            'volume': volume,
            'avg_volume': 0,
            'market_cap': 0,
            'bid': 0,
            'ask': 0,
            'timestamp': datetime.now(),
            '_cache_ts': datetime.now(),
            'source': source,
        }

    # ── Technical indicators ────────────────────────────────────────
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to price data."""
        try:
            if data.empty:
                return data

            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # ── Moving Averages ────────────────────────────
            data['SMA_20'] = close.rolling(window=20).mean()
            data['SMA_50'] = close.rolling(window=50).mean()
            data['SMA_200'] = close.rolling(window=200).mean()  # NEW: key long-term level

            data['EMA_12'] = close.ewm(span=12).mean()
            data['EMA_26'] = close.ewm(span=26).mean()

            # ── MACD ───────────────────────────────────────
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

            # ── RSI ────────────────────────────────────────
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rs = rs.fillna(0).replace([np.inf, -np.inf], 0)
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(50)

            # Fill NaN before Bollinger Bands
            data = data.ffill().bfill()

            # ── Bollinger Bands ────────────────────────────
            data['BB_Middle'] = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

            # ── Volume indicators ──────────────────────────
            data['Volume_SMA'] = volume.rolling(window=20).mean()
            data['Volume_Ratio'] = volume / data['Volume_SMA']

            # ── OBV (On-Balance Volume) ── NEW ─────────────
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            data['OBV'] = obv
            data['OBV_SMA'] = data['OBV'].rolling(window=20).mean()

            # ── Stochastic Oscillator ── NEW ───────────────
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            denom = high_14 - low_14
            denom = denom.replace(0, np.nan)
            data['Stoch_K'] = ((close - low_14) / denom * 100).fillna(50)
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean().fillna(50)

            # ── ATR (Average True Range) ── NEW ────────────
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['ATR'] = true_range.rolling(window=14).mean()
            data['ATR_pct'] = (data['ATR'] / close * 100)  # ATR as % of price

            # ── ADX (Average Directional Index) ────────────
            data = self._add_adx(data)

            # ── 52-Week High / Low Proximity ── NEW ────────
            if len(data) >= 52:
                data['High_52w'] = high.rolling(window=min(252, len(data))).max()
                data['Low_52w'] = low.rolling(window=min(252, len(data))).min()
                range_52w = data['High_52w'] - data['Low_52w']
                range_52w = range_52w.replace(0, np.nan)
                data['Pct_from_52w_high'] = ((close - data['High_52w']) / data['High_52w'] * 100).fillna(0)
                data['Pct_from_52w_low'] = ((close - data['Low_52w']) / data['Low_52w'] * 100).fillna(0)
            else:
                data['High_52w'] = high.max()
                data['Low_52w'] = low.min()
                data['Pct_from_52w_high'] = 0
                data['Pct_from_52w_low'] = 0

            return data

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data

    def _add_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX for trend-strength filtering."""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']

            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = true_range.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            dx = dx.fillna(0).replace([np.inf, -np.inf], 0)
            data['ADX'] = dx.rolling(window=period).mean().fillna(0)
            data['Plus_DI'] = plus_di.fillna(0)
            data['Minus_DI'] = minus_di.fillna(0)
        except Exception as e:
            self.logger.error(f"Error computing ADX: {e}")
            data['ADX'] = 0
            data['Plus_DI'] = 0
            data['Minus_DI'] = 0
        return data

    # ── Fundamentals (cached 10 min) ────────────────────────────────
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data for a stock (cached 10 min)."""
        cached = self._fundamentals_cache.get(symbol)
        if cached and self._cache_valid(cached.get('_cache_ts'), self.FUNDAMENTALS_CACHE_TTL):
            return cached

        if not self._prefer_yahoo():
            return {
                'pe_ratio': 0,
                'forward_pe': 0,
                'peg_ratio': 0,
                'price_to_book': 0,
                'debt_to_equity': 0,
                'return_on_equity': 0,
                'profit_margin': 0,
                'revenue_growth': 0,
                'earnings_growth': 0,
                'beta': 0,
                'dividend_yield': 0,
                'market_cap': 0,
                'enterprise_value': 0,
                'shares_outstanding': 0,
                'float_shares': 0,
                'short_ratio': 0,
                'short_percent_of_float': 0,
                'current_ratio': 0,
                'operating_margins': 0,
                'sector': 'Unknown',
                'industry': 'Unknown',
                '_cache_ts': datetime.now(),
            }

        try:
            ticker = _get_yfinance().Ticker(symbol)
            info = ticker.info

            fundamentals = {
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'peg_ratio': info.get('pegRatio', 0) or 0,
                'price_to_book': info.get('priceToBook', 0) or 0,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'return_on_equity': info.get('returnOnEquity', 0) or 0,
                'profit_margin': info.get('profitMargins', 0) or 0,
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'earnings_growth': info.get('earningsGrowth', 0) or 0,
                'beta': info.get('beta', 0) or 0,
                'dividend_yield': info.get('dividendYield', 0) or 0,
                'market_cap': info.get('marketCap', 0) or 0,
                'enterprise_value': info.get('enterpriseValue', 0) or 0,
                'shares_outstanding': info.get('sharesOutstanding', 0) or 0,
                'float_shares': info.get('floatShares', 0) or 0,
                'short_ratio': info.get('shortRatio', 0) or 0,
                'short_percent_of_float': info.get('shortPercentOfFloat', 0) or 0,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'operating_margins': info.get('operatingMargins', 0) or 0,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                '_cache_ts': datetime.now(),
            }
            self._fundamentals_cache[symbol] = fundamentals
            return fundamentals

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: fundamentals for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {}

    # ── Time helpers ────────────────────────────────────────────────
    def _get_eastern_time(self) -> datetime:
        utc_now = datetime.now(pytz.UTC)
        return utc_now.astimezone(self.eastern_tz)

    def get_eastern_time_string(self) -> str:
        return self._get_eastern_time().strftime("%H:%M:%S")

    def is_market_open(self) -> bool:
        now = self._get_eastern_time()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        market_open = datetime.strptime(self.config.MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(self.config.MARKET_CLOSE, "%H:%M").time()
        return market_open <= current_time <= market_close

    def is_extended_hours(self) -> bool:
        now = self._get_eastern_time()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        extended_start = datetime.strptime(self.config.EXTENDED_HOURS_START, "%H:%M").time()
        extended_end = datetime.strptime(self.config.EXTENDED_HOURS_END, "%H:%M").time()
        return extended_start <= current_time <= extended_end
