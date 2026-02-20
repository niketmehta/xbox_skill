import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging
from config import Config
import pytz
import re


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

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.eastern_tz = pytz.timezone('US/Eastern')

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

    # ── Real-time quote (cached) ─────────────────────────────────────
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol (with 2-minute cache)."""
        # Check cache first
        cached = self._quote_cache.get(symbol)
        if cached and self._cache_valid(cached.get('_cache_ts'), self.QUOTE_CACHE_TTL):
            return cached

        try:
            ticker = yf.Ticker(symbol)
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
            }

            self._quote_cache[symbol] = quote
            return quote

        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: quote for {symbol}: {e}")
                # Return stale cache if available during rate-limit
                if cached:
                    self.logger.info(f"Returning stale cache for {symbol}")
                    return cached
            else:
                self.logger.error(f"Error getting quote for {symbol}: {e}")
            return {}

    # ── Daily data (for WEEK / swing trading) ───────────────────────
    def get_daily_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get daily candles for swing-trade / weekly analysis."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d")

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
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1wk")

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
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, prepost=True)
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
            vix_ticker = yf.Ticker('^VIX')
            vix_data = vix_ticker.history(period='5d', interval='1d')
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
                spy = yf.Ticker('SPY')
                self._spy_cache[cache_key] = spy.history(period=period, interval='1d')
                self._spy_cache_time[cache_key] = now

            spy_data = self._spy_cache[cache_key]
            if spy_data.empty:
                return {}

            stock = yf.Ticker(symbol)
            stock_data = stock.history(period=period, interval='1d')
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
        try:
            ticker = yf.Ticker(symbol)
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
        'SPY', 'QQQ', 'AAPL', 'TSLA', 'AMD', 'NVDA', 'MSFT', 'GOOGL',
        'AMZN', 'META', 'NFLX', 'BABA', 'DIS', 'V', 'JPM', 'BAC',
        'XOM', 'JNJ', 'PG', 'KO', 'PFE', 'T', 'VZ', 'WMT', 'HD',
        'UNH', 'CVX', 'LLY', 'ABBV', 'CRM', 'ORCL', 'ACN', 'TMO',
        'COST', 'ABT', 'DHR', 'TXN', 'NEE', 'PMT', 'LIN', 'NKE',
        'CMCSA', 'INTC', 'COP', 'QCOM', 'HON', 'UPS', 'LOW', 'IBM'
    ]

    def get_market_movers(self, count: int = 25) -> List[Dict]:
        """Get top market movers — cached for 3 minutes, batch-downloaded."""
        # Return cache if fresh
        if (self._movers_cache is not None and
                self._cache_valid(self._movers_cache_time, self.MOVERS_CACHE_TTL)):
            return self._movers_cache[:count]

        try:
            symbols = self.MOVERS_SYMBOLS[:count]
            # Use yf.download for a single batch HTTP request instead of N calls
            batch = yf.download(symbols, period='2d', interval='1d',
                                group_by='ticker', progress=False, threads=True)

            movers = []
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = batch
                    else:
                        df = batch[sym] if sym in batch.columns.get_level_values(0) else None
                    if df is None or df.empty or len(df) < 2:
                        continue

                    current_price = float(df['Close'].iloc[-1])
                    previous_close = float(df['Close'].iloc[-2])
                    volume = int(df['Volume'].iloc[-1]) if not pd.isna(df['Volume'].iloc[-1]) else 0

                    if previous_close == 0:
                        continue

                    change = current_price - previous_close
                    change_pct = (change / previous_close) * 100

                    quote = {
                        'symbol': sym,
                        'current_price': current_price,
                        'previous_close': previous_close,
                        'change': change,
                        'change_percent': change_pct,
                        'volume': volume,
                        'avg_volume': 0,
                        'market_cap': 0,
                        'bid': 0, 'ask': 0,
                        'timestamp': datetime.now(),
                        '_cache_ts': datetime.now(),
                    }
                    # Also populate the quote cache so other calls benefit
                    self._quote_cache[sym] = quote
                    if abs(change_pct) > 0:
                        movers.append(quote)
                except Exception:
                    continue

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

        try:
            ticker = yf.Ticker(symbol)
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
