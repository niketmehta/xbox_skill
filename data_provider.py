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
    Supports intraday (day-trade), daily (swing/week), and weekly (position/month) timeframes.
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def _is_rate_limit_error(self, e):
        msg = str(e)
        patterns = [
            r"429", r"rate limit", r"too many requests",
            r"temporarily blocked", r"try again later"
        ]
        return any(re.search(p, msg, re.IGNORECASE) for p in patterns)

    # ── Real-time quote ─────────────────────────────────────────────
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol"""
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
                'timestamp': datetime.now()
            }
            
            return quote
            
        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: quote for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting quote for {symbol}: {e}")
            return {}
    
    # ── Intraday data (for DAY horizon) ─────────────────────────────
    def get_intraday_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """Get intraday data for day-trading technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval, prepost=True)
            
            if data.empty:
                self.logger.warning(f"No intraday data returned for {symbol}")
                return pd.DataFrame()
            
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: intraday data for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    # ── Daily data (for WEEK horizon / swing trading) ───────────────
    def get_daily_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get daily candles for swing-trade / weekly analysis.
        
        Args:
            symbol: Ticker symbol
            period: yfinance period string (default '3mo' = 3 months)
        """
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
    
    # ── Weekly data (for MONTH horizon / position trading) ──────────
    def get_weekly_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get weekly candles for position-trade / monthly analysis.
        
        Args:
            symbol: Ticker symbol
            period: yfinance period string (default '1y' = 1 year)
        """
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
    
    # ── Extended-hours data ─────────────────────────────────────────
    def get_extended_hours_data(self, symbol: str) -> Dict:
        """Get pre-market and after-hours data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m", prepost=True)
            
            if data.empty:
                return {}
            
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            premarket_data = data[data.index.time < market_open.time()]
            regular_data = data[(data.index.time >= market_open.time()) & 
                              (data.index.time <= market_close.time())]
            afterhours_data = data[data.index.time > market_close.time()]
            
            extended_hours = {
                'premarket': {
                    'high': premarket_data['High'].max() if not premarket_data.empty else 0,
                    'low': premarket_data['Low'].min() if not premarket_data.empty else 0,
                    'volume': premarket_data['Volume'].sum() if not premarket_data.empty else 0,
                    'last_price': premarket_data['Close'].iloc[-1] if not premarket_data.empty else 0
                },
                'regular': {
                    'high': regular_data['High'].max() if not regular_data.empty else 0,
                    'low': regular_data['Low'].min() if not regular_data.empty else 0,
                    'volume': regular_data['Volume'].sum() if not regular_data.empty else 0,
                    'last_price': regular_data['Close'].iloc[-1] if not regular_data.empty else 0
                },
                'afterhours': {
                    'high': afterhours_data['High'].max() if not afterhours_data.empty else 0,
                    'low': afterhours_data['Low'].min() if not afterhours_data.empty else 0,
                    'volume': afterhours_data['Volume'].sum() if not afterhours_data.empty else 0,
                    'last_price': afterhours_data['Close'].iloc[-1] if not afterhours_data.empty else 0
                }
            }
            
            return extended_hours
            
        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: extended hours for {symbol}: {e}")
            else:
                self.logger.error(f"Error getting extended hours data for {symbol}: {e}")
            return {}
    
    # ── Market movers ───────────────────────────────────────────────
    def get_market_movers(self, count: int = 50) -> List[Dict]:
        """Get top market movers (gainers and losers)"""
        try:
            symbols = [
                'SPY', 'QQQ', 'AAPL', 'TSLA', 'AMD', 'NVDA', 'MSFT', 'GOOGL',
                'AMZN', 'META', 'NFLX', 'BABA', 'DIS', 'V', 'JPM', 'BAC',
                'XOM', 'JNJ', 'PG', 'KO', 'PFE', 'T', 'VZ', 'WMT', 'HD',
                'UNH', 'CVX', 'LLY', 'ABBV', 'CRM', 'ORCL', 'ACN', 'TMO',
                'COST', 'ABT', 'DHR', 'TXN', 'NEE', 'PMT', 'LIN', 'NKE',
                'CMCSA', 'INTC', 'COP', 'QCOM', 'HON', 'UPS', 'LOW', 'IBM'
            ]
            
            movers = []
            for symbol in symbols[:count]:
                quote = self.get_real_time_quote(symbol)
                if quote and quote.get('change_percent'):
                    movers.append(quote)
            
            movers.sort(key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            return movers
            
        except Exception as e:
            if self._is_rate_limit_error(e):
                self.logger.error(f"RATE LIMIT: market movers: {e}")
            else:
                self.logger.error(f"Error getting market movers: {e}")
            return []
    
    # ── Technical indicators ────────────────────────────────────────
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            if data.empty:
                return data
            
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rs = rs.fillna(0).replace([np.inf, -np.inf], 0)
            data['RSI'] = 100 - (100 / (1 + rs))
            data['RSI'] = data['RSI'].fillna(50)
            
            # Fill NaN values
            data = data.ffill().bfill()
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # ADX (Average Directional Index) – trend strength
            data = self._add_adx(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _add_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX for trend-strength filtering"""
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
    
    # ── Fundamentals ────────────────────────────────────────────────
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """Get basic fundamental data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0)
            }
            
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
