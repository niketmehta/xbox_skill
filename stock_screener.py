import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_provider import MarketDataProvider
from config import Config

class StockScreener:
    """
    Automatic stock screener for swing and position trading opportunities.
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Stock universes for screening
        self.stock_universes = {
            'sp500': self._get_sp500_symbols(),
            'nasdaq100': self._get_nasdaq100_symbols(),
            'nyse': self._get_nyse_symbols(),
            'small_cap': self._get_small_cap_symbols(),
            'high_volume': self._get_high_volume_stocks(),
            'popular_trading': self._get_popular_trading_stocks(),
            'high_volatility': self._get_high_volatility_stocks()
        }
    
    def screen_stocks(self, max_stocks: int = 80, screen_type: str = 'swing') -> List[str]:
        """
        Screen stocks for swing/position trading opportunities.
        
        Args:
            max_stocks: Maximum number of stocks to return (default: 80)
            screen_type: Type of screening ('swing', 'breakout', 'momentum',
                         'high_volume', 'high_volatility', 'value')
        """
        
        self.logger.info(f"Starting automatic stock screening: {screen_type}")
        
        if screen_type == 'swing':
            return self._screen_swing_stocks(max_stocks)
        elif screen_type == 'breakout':
            return self._screen_breakout_stocks(max_stocks)
        elif screen_type == 'momentum':
            return self._screen_momentum_stocks(max_stocks)
        elif screen_type == 'high_volume':
            return self._screen_high_volume_stocks(max_stocks)
        elif screen_type == 'high_volatility':
            return self._screen_high_volatility_stocks(max_stocks)
        elif screen_type == 'value':
            return self._screen_value_stocks(max_stocks)
        else:
            return self._screen_swing_stocks(max_stocks)
    
    def _screen_swing_stocks(self, max_stocks: int) -> List[str]:
        """Screen for optimal swing/position trading stocks."""
        
        # Combine multiple stock universes
        candidate_symbols = list(set(
            self.stock_universes['popular_trading'] +
            self.stock_universes['high_volume'][:40] +
            self.stock_universes['nyse'][:50] +
            self.stock_universes['sp500'][:60] +
            self.stock_universes['nasdaq100'][:40] +
            self._get_current_market_movers(30)
        ))
        
        screened_stocks = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._evaluate_swing_stock, symbol): symbol 
                for symbol in candidate_symbols[:100]
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=10)
                    if result and result['suitable']:
                        screened_stocks.append({
                            'symbol': symbol,
                            'score': result['score'],
                            'metrics': result['metrics']
                        })
                except Exception as e:
                    self.logger.warning(f"Error screening {symbol}: {e}")
        
        screened_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected_symbols = [stock['symbol'] for stock in screened_stocks[:max_stocks]]
        self.logger.info(f"Selected {len(selected_symbols)} stocks for swing trading")
        return selected_symbols
    
    def _evaluate_volatility_stock(self, symbol: str) -> Optional[Dict]:
        """Evaluate if a stock has high volatility suitable for aggressive trading"""
        
        try:
            # Get recent data
            ticker = yf.Ticker(symbol)
            
            # Get more data for volatility calculation
            data = ticker.history(period="30d", interval="1d")
            
            if len(data) < 20:
                return None
            
            # Calculate volatility metrics
            daily_returns = data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Calculate ATR for intraday volatility
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            atr_percent = (atr / data['Close'].iloc[-1]) * 100
            
            # High volatility criteria
            high_volatility = (
                volatility > 40 or  # High annualized volatility
                atr_percent > 5 or   # High intraday range
                abs(daily_returns.iloc[-1]) > 0.05  # Recent large move
            )
            
            return {
                'high_volatility': high_volatility,
                'volatility': volatility,
                'atr_percent': atr_percent,
                'current_price': data['Close'].iloc[-1]
            }
            
        except Exception as e:
            self.logger.warning(f"Error evaluating volatility for {symbol}: {e}")
            return None
    
    def _evaluate_swing_stock(self, symbol: str) -> Optional[Dict]:
        """Evaluate if a stock is suitable for swing/position trading."""
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="60d", interval="1d")
            if data.empty or len(data) < 20:
                return None
            
            quote = self.data_provider.get_real_time_quote(symbol)
            if not quote:
                return None
            
            metrics = self._calculate_screening_metrics(data, quote)
            score = self._calculate_swing_score(metrics)
            
            suitable = score > 55  # slightly lower threshold for wider net
            
            return {
                'suitable': suitable,
                'score': score,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating {symbol}: {e}")
            return None
    
    def _calculate_screening_metrics(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Calculate metrics for stock screening"""
        
        current_price = quote.get('current_price', data.iloc[-1]['Close'])
        
        # Price metrics
        price_20d_high = data['High'].tail(20).max()
        price_20d_low = data['Low'].tail(20).min()
        price_range = (price_20d_high - price_20d_low) / price_20d_low * 100
        
        # Volume metrics
        avg_volume_20d = data['Volume'].tail(20).mean()
        current_volume = quote.get('volume', data.iloc[-1]['Volume'])
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1
        
        # Volatility metrics
        returns = data['Close'].pct_change().dropna()
        daily_volatility = returns.std() * 100
        
        # Price momentum
        price_5d_change = (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100 if len(data) > 5 else 0
        price_20d_change = (current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21] * 100 if len(data) > 20 else 0
        
        # Market cap (if available)
        market_cap = quote.get('market_cap', 0)
        
        # Price level
        price_level = current_price
        
        # Average True Range (ATR) - simplified
        high_low = data['High'] - data['Low']
        atr = high_low.tail(14).mean() / current_price * 100 if current_price > 0 else 0
        
        return {
            'current_price': current_price,
            'price_range_20d': price_range,
            'avg_volume_20d': avg_volume_20d,
            'volume_ratio': volume_ratio,
            'daily_volatility': daily_volatility,
            'price_5d_change': price_5d_change,
            'price_20d_change': price_20d_change,
            'market_cap': market_cap,
            'atr_percent': atr
        }
    
    def _calculate_swing_score(self, metrics: Dict) -> float:
        """Calculate swing-trading suitability score (0-100)."""
        
        score = 0
        
        # Volume criteria (25 points max)
        if metrics['avg_volume_20d'] > 1000000:
            score += 18
        elif metrics['avg_volume_20d'] > 500000:
            score += 12
        elif metrics['avg_volume_20d'] > 100000:
            score += 8
        
        if metrics['volume_ratio'] > 1.5:
            score += 7
        elif metrics['volume_ratio'] > 1.2:
            score += 4
        
        # Volatility criteria (25 points max) — moderate vol is best for swing
        if 1.5 <= metrics['daily_volatility'] <= 4:
            score += 25
        elif 1 <= metrics['daily_volatility'] <= 6:
            score += 20
        elif 0.5 <= metrics['daily_volatility'] <= 8:
            score += 12
        
        # Price criteria (15 points max)
        if 15 <= metrics['current_price'] <= 500:
            score += 15
        elif 5 <= metrics['current_price'] <= 1000:
            score += 10
        elif metrics['current_price'] >= 1:
            score += 5
        
        # Market cap criteria (20 points max) — favour quality names
        if metrics['market_cap'] > 10e9:
            score += 20
        elif metrics['market_cap'] > 2e9:
            score += 15
        elif metrics['market_cap'] > 500e6:
            score += 10
        elif metrics['market_cap'] > 100e6:
            score += 5
        
        # Momentum criteria (15 points max)
        if 2 < metrics['price_20d_change'] < 20:  # uptrend but not parabolic
            score += 10
        elif 0 < metrics['price_20d_change'] < 30:
            score += 5
        
        if abs(metrics['price_5d_change']) > 1:
            score += 5
        
        return min(score, 100)
    
    def _screen_breakout_stocks(self, max_stocks: int) -> List[str]:
        """Screen for breakout opportunities"""
        
        candidate_symbols = self.stock_universes['sp500'][:200]
        breakout_stocks = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self._check_breakout_pattern, symbol): symbol 
                for symbol in candidate_symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=10)
                    if result and result['is_breakout']:
                        breakout_stocks.append({
                            'symbol': symbol,
                            'strength': result['strength']
                        })
                except Exception:
                    continue
        
        # Sort by breakout strength
        breakout_stocks.sort(key=lambda x: x['strength'], reverse=True)
        
        return [stock['symbol'] for stock in breakout_stocks[:max_stocks]]
    
    def _check_breakout_pattern(self, symbol: str) -> Optional[Dict]:
        """Check for breakout patterns"""
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="60d", interval="1d")
            
            if len(data) < 30:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            # Check for resistance breakout
            resistance_20d = data['High'].tail(20).max()
            resistance_50d = data['High'].tail(50).max()
            
            # Volume confirmation
            avg_volume = data['Volume'].tail(20).mean()
            recent_volume = data['Volume'].iloc[-1]
            
            # Breakout conditions
            resistance_breakout = current_price > resistance_20d * 1.02  # 2% above resistance
            volume_confirmation = recent_volume > avg_volume * 1.5
            
            if resistance_breakout and volume_confirmation:
                # Calculate strength
                price_strength = (current_price - resistance_20d) / resistance_20d * 100
                volume_strength = recent_volume / avg_volume
                
                strength = (price_strength * 0.6) + (volume_strength * 0.4)
                
                return {
                    'is_breakout': True,
                    'strength': strength
                }
            
            return {'is_breakout': False, 'strength': 0}
            
        except Exception:
            return None
    
    def _screen_momentum_stocks(self, max_stocks: int) -> List[str]:
        """Screen for momentum stocks"""
        
        # Get recent market movers
        movers = self._get_current_market_movers(100)
        
        momentum_stocks = []
        
        for symbol in movers:
            quote = self.data_provider.get_real_time_quote(symbol)
            if quote and abs(quote.get('change_percent', 0)) > 3:  # > 3% move
                momentum_stocks.append({
                    'symbol': symbol,
                    'momentum': abs(quote['change_percent'])
                })
        
        # Sort by momentum
        momentum_stocks.sort(key=lambda x: x['momentum'], reverse=True)
        
        return [stock['symbol'] for stock in momentum_stocks[:max_stocks]]
    
    def _screen_high_volume_stocks(self, max_stocks: int) -> List[str]:
        """Screen for high volume stocks"""
        return self.stock_universes['high_volume'][:max_stocks]
    
    def _screen_high_volatility_stocks(self, max_stocks: int) -> List[str]:
        """Screen for high volatility stocks"""
        try:
            # Combine high volatility universe with additional screening
            candidate_symbols = list(set(
                self.stock_universes['high_volatility'] +
                self.stock_universes['small_cap'][:30] +
                self._get_current_market_movers(20)
            ))
            
            high_volatility_stocks = []
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {
                    executor.submit(self._evaluate_volatility_stock, symbol): symbol 
                    for symbol in candidate_symbols[:80]
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=10)
                        if result and result['high_volatility']:
                            high_volatility_stocks.append({
                                'symbol': symbol,
                                'volatility': result['volatility']
                            })
                    except Exception as e:
                        self.logger.warning(f"Error screening {symbol} for volatility: {e}")
            
            # Sort by volatility and return top stocks
            high_volatility_stocks.sort(key=lambda x: x['volatility'], reverse=True)
            
            selected_symbols = [stock['symbol'] for stock in high_volatility_stocks[:max_stocks]]
            
            self.logger.info(f"Selected {len(selected_symbols)} high volatility stocks")
            
            return selected_symbols
            
        except Exception as e:
            self.logger.error(f"Error in high volatility screening: {e}")
            return self.stock_universes['high_volatility'][:max_stocks]
    
    def _get_current_market_movers(self, count: int) -> List[str]:
        """Get current market movers"""
        try:
            movers = self.data_provider.get_market_movers(count)
            return [mover['symbol'] for mover in movers if mover.get('symbol')]
        except Exception as e:
            self.logger.error(f"Error getting market movers: {e}")
            return []
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        try:
            # Common S&P 500 symbols (subset for efficiency)
            sp500_symbols = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'META', 'NVDA',
                'BRK-B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX',
                'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'PEP', 'TMO',
                'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'ACN', 'VZ', 'ADBE', 'NEE',
                'CRM', 'NFLX', 'TXN', 'CMCSA', 'RTX', 'NKE', 'QCOM', 'T', 'BMY',
                'UPS', 'PM', 'LOW', 'ORCL', 'HON', 'IBM', 'AMGN', 'SBUX', 'CAT',
                'GE', 'MDLZ', 'AMT', 'BA', 'AXP', 'BLK', 'DE', 'MMM', 'AMD'
            ]
            return sp500_symbols
        except Exception:
            return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols"""
        nasdaq100_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'META', 'NVDA',
            'AVGO', 'ORCL', 'CRM', 'NFLX', 'ADBE', 'TXN', 'QCOM', 'COST',
            'PEP', 'AMD', 'INTC', 'CMCSA', 'TMUS', 'HON', 'AMGN', 'SBUX',
            'INTU', 'ISRG', 'BKNG', 'ADP', 'GILD', 'ADI', 'MDLZ', 'MU',
            'PYPL', 'LRCX', 'REGN', 'ATVI', 'FISV', 'CSX', 'MRNA', 'KLAC'
        ]
        return nasdaq100_symbols
    
    def _get_nyse_symbols(self) -> List[str]:
        """Get NYSE symbols"""
        nyse_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'META', 'NVDA',
            'BRK-B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX',
            'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'PEP', 'TMO',
            'COST', 'WMT', 'DIS', 'ABT', 'DHR', 'ACN', 'VZ', 'ADBE', 'NEE',
            'CRM', 'NFLX', 'TXN', 'CMCSA', 'RTX', 'NKE', 'QCOM', 'T', 'BMY',
            'UPS', 'PM', 'LOW', 'ORCL', 'HON', 'IBM', 'AMGN', 'SBUX', 'CAT',
            'GE', 'MDLZ', 'AMT', 'BA', 'AXP', 'BLK', 'DE', 'MMM', 'AMD'
        ]
        return nyse_symbols
    
    def _get_small_cap_symbols(self) -> List[str]:
        """Get small cap symbols"""
        small_cap_symbols = [
            'PLTR', 'RBLX', 'HOOD', 'COIN', 'RIVN', 'LCID', 'NIO', 'XPEV',
            'LI', 'BYND', 'SNAP', 'PINS', 'TWTR', 'UBER', 'LYFT', 'DASH',
            'ABNB', 'SNOW', 'CRWD', 'NET', 'ZM', 'DOCU', 'TEAM', 'OKTA',
            'MDB', 'TWLO', 'SQ', 'PYPL', 'SHOP', 'ROKU', 'PTON', 'ZM',
            'AMC', 'GME', 'BBBY', 'EXPR', 'NAKD', 'SNDL', 'TLRY', 'ACB',
            'CGC', 'HEXO', 'APHA', 'CRON', 'OGI', 'VFF', 'GTBIF', 'TCNNF',
            'JUSHF', 'MMNFF', 'CURLF', 'GRWG', 'HYFM', 'SMG', 'IIPR'
        ]
        return small_cap_symbols
    
    def _get_high_volatility_stocks(self) -> List[str]:
        """Get high volatility stocks"""
        high_volatility_stocks = [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'AMD', 'NVDA', 'SQQQ', 'TQQQ',
            'IWM', 'EEM', 'XLF', 'GLD', 'SLV', 'VXX', 'UVXY', 'SPXL',
            'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'BABA', 'NIO',
            'PLTR', 'AMC', 'GME', 'PTON', 'ZOOM', 'ROKU', 'SQ', 'DOCU'
        ]
        return high_volatility_stocks
    
    def _get_high_volume_stocks(self) -> List[str]:
        """Get high volume stocks"""
        high_volume_stocks = [
            'SPY', 'QQQ', 'AAPL', 'TSLA', 'AMD', 'NVDA', 'SQQQ', 'TQQQ',
            'IWM', 'EEM', 'XLF', 'GLD', 'SLV', 'VXX', 'UVXY', 'SPXL',
            'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'BABA', 'NIO',
            'PLTR', 'AMC', 'GME', 'PTON', 'ZOOM', 'ROKU', 'SQ', 'DOCU'
        ]
        return high_volume_stocks
    
    def _get_popular_trading_stocks(self) -> List[str]:
        """Get popular liquid trading stocks."""
        popular_stocks = [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'AMD', 'NVDA', 'MSFT',
            'AMZN', 'GOOGL', 'META', 'NFLX', 'BABA', 'DIS', 'V', 'JPM',
            'BAC', 'XOM', 'CVX', 'JNJ', 'PG', 'KO', 'PFE', 'T', 'VZ',
            'WMT', 'HD', 'UNH', 'LLY', 'ABBV', 'CRM', 'ORCL', 'ACN',
            'TMO', 'COST', 'ABT', 'DHR', 'TXN', 'NEE', 'PMT', 'LIN',
            'NKE', 'CMCSA', 'INTC', 'COP', 'QCOM', 'HON', 'UPS', 'LOW'
        ]
        return popular_stocks
    
    def _screen_value_stocks(self, max_stocks: int) -> List[str]:
        """Screen for value stocks with reasonable P/E and solid fundamentals."""
        
        candidate_symbols = list(set(
            self.stock_universes['sp500'][:60] +
            self.stock_universes['nyse'][:40]
        ))
        
        value_stocks = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self._evaluate_value_stock, symbol): symbol
                for symbol in candidate_symbols[:80]
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=10)
                    if result and result['suitable']:
                        value_stocks.append({
                            'symbol': symbol,
                            'score': result['score']
                        })
                except Exception as e:
                    self.logger.warning(f"Error screening value stock {symbol}: {e}")
        
        value_stocks.sort(key=lambda x: x['score'], reverse=True)
        selected = [s['symbol'] for s in value_stocks[:max_stocks]]
        self.logger.info(f"Selected {len(selected)} value stocks")
        return selected

    def _evaluate_value_stock(self, symbol: str) -> Optional[Dict]:
        """Evaluate a stock for value characteristics."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            pe = info.get('trailingPE') or info.get('forwardPE')
            pb = info.get('priceToBook')
            div_yield = info.get('dividendYield', 0) or 0
            profit_margin = info.get('profitMargins', 0) or 0
            
            if pe is None or pe <= 0:
                return None
            
            score = 0
            # Reasonable P/E (max 30 pts)
            if 5 < pe < 15:
                score += 30
            elif 15 <= pe < 25:
                score += 20
            elif 25 <= pe < 35:
                score += 10
            
            # Low P/B (max 20 pts)
            if pb is not None:
                if pb < 1.5:
                    score += 20
                elif pb < 3:
                    score += 12
                elif pb < 5:
                    score += 5
            
            # Dividend yield (max 20 pts)
            if div_yield > 0.04:
                score += 20
            elif div_yield > 0.02:
                score += 12
            elif div_yield > 0.01:
                score += 5
            
            # Profit margin (max 20 pts)
            if profit_margin > 0.20:
                score += 20
            elif profit_margin > 0.10:
                score += 12
            elif profit_margin > 0.05:
                score += 5
            
            return {'suitable': score >= 40, 'score': score}
        except Exception:
            return None

    def get_smart_watchlist(self, size: int = 50) -> List[str]:
        """Get a smart watchlist combining multiple screening methods."""
        
        try:
            swing_stocks = self._screen_swing_stocks(size // 2)
            momentum_stocks = self._screen_momentum_stocks(size // 4)
            breakout_stocks = self._screen_breakout_stocks(size // 4)
            value_stocks = self._screen_value_stocks(size // 6)
            sp500_stocks = self.stock_universes['sp500'][:size // 6]
            
            # Combine and deduplicate
            combined_stocks = list(set(
                swing_stocks + momentum_stocks + breakout_stocks +
                value_stocks + sp500_stocks
            ))
            
            # If not enough, pad with high-quality names
            if len(combined_stocks) < size:
                for stock in self.stock_universes['nasdaq100']:
                    if stock not in combined_stocks and len(combined_stocks) < size:
                        combined_stocks.append(stock)
            
            self.logger.info(f"Generated smart watchlist with {len(combined_stocks)} stocks")
            return combined_stocks[:size]
            
        except Exception as e:
            self.logger.error(f"Error generating smart watchlist: {e}")
            return self._get_popular_trading_stocks()[:size]