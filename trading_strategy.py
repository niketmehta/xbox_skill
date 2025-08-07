import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from data_provider import MarketDataProvider
from config import Config

class TradingStrategy:
    """
    Day trading strategies focused on maximizing profits and limiting losses
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    def analyze_stock(self, symbol: str) -> Dict:
        """Comprehensive analysis of a stock for day trading opportunities"""
        try:
            # Get real-time quote
            quote = self.data_provider.get_real_time_quote(symbol)
            if not quote:
                self.logger.warning(f"No quote data available for {symbol}")
                return {}
            
            # Get intraday data for technical analysis
            intraday_data = self.data_provider.get_intraday_data(symbol, period="1d", interval="5m")
            if intraday_data.empty:
                self.logger.warning(f"No intraday data available for {symbol}")
                return {}
            
            # Get extended hours data
            extended_hours = self.data_provider.get_extended_hours_data(symbol)
            
            # Run multiple strategy analyses
            momentum_signal = self._momentum_strategy(intraday_data, quote)
            reversal_signal = self._mean_reversion_strategy(intraday_data, quote)
            breakout_signal = self._breakout_strategy(intraday_data, quote)
            volume_signal = self._volume_analysis(intraday_data, quote)
            
            # Combine signals with weighted scoring
            combined_score = self._combine_signals(
                momentum_signal, reversal_signal, breakout_signal, volume_signal
            )
            
            # Risk assessment
            risk_score = self._assess_risk(intraday_data, quote, extended_hours)
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': quote.get('current_price', 0),
                'signals': {
                    'momentum': momentum_signal,
                    'reversal': reversal_signal,
                    'breakout': breakout_signal,
                    'volume': volume_signal,
                    'combined_score': combined_score
                },
                'risk_score': risk_score,
                'extended_hours': extended_hours,
                'recommendation': self._generate_recommendation(combined_score, risk_score, quote)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {}
    
    def _momentum_strategy(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Momentum-based trading signals"""
        if data.empty or len(data) < 50:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])
        
        # Ensure we have valid price data
        if not current_price or current_price <= 0:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Invalid price data'}
        
        # Check if price is above key moving averages (handle NaN values)
        sma20 = latest.get('SMA_20', 0)
        sma50 = latest.get('SMA_50', 0)
        
        # Handle NaN values
        if pd.isna(sma20):
            sma20 = current_price
        if pd.isna(sma50):
            sma50 = current_price
            
        above_sma20 = current_price > sma20
        above_sma50 = current_price > sma50
        
        # MACD analysis
        macd_bullish = (latest.get('MACD', 0) > latest.get('MACD_Signal', 0) and
                       latest.get('MACD_Histogram', 0) > 0)
        
        # Price momentum (5-period rate of change)
        if len(data) >= 5:
            price_momentum = (current_price - data.iloc[-5]['Close']) / data.iloc[-5]['Close'] * 100
        else:
            price_momentum = 0
        
        # Volume momentum
        avg_volume = data['Volume'].tail(20).mean()
        current_volume = quote.get('volume', latest['Volume'])
        volume_spike = current_volume > avg_volume * self.config.VOLUME_SPIKE_THRESHOLD
        
        # Score calculation
        score = 0
        reasons = []
        
        if above_sma20 and above_sma50:
            score += 2
            reasons.append("Above key moving averages")
        elif above_sma20:
            score += 1
            reasons.append("Above SMA20")
        
        if macd_bullish:
            score += 2
            reasons.append("MACD bullish crossover")
        
        if price_momentum > 2:
            score += 2
            reasons.append(f"Strong momentum ({price_momentum:.1f}%)")
        elif price_momentum > 0.5:
            score += 1
            reasons.append(f"Positive momentum ({price_momentum:.1f}%)")
        
        if volume_spike:
            score += 1
            reasons.append("Volume spike detected")
        
        # Determine signal
        if score >= 4:
            signal = 'BUY'
        elif score <= -2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': abs(score) / 7 * 100,  # Normalize to 0-100
            'score': score,
            'reasons': reasons,
            'momentum': price_momentum
        }
    
    def _mean_reversion_strategy(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Mean reversion strategy using RSI and Bollinger Bands"""
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])
        
        # RSI analysis
        rsi = latest.get('RSI', 50)
        # Handle NaN values
        if pd.isna(rsi):
            rsi = 50
        rsi_oversold = rsi < self.config.RSI_OVERSOLD
        rsi_overbought = rsi > self.config.RSI_OVERBOUGHT
        
        # Bollinger Bands analysis
        bb_upper = latest.get('BB_Upper', 0)
        bb_lower = latest.get('BB_Lower', 0)
        bb_middle = latest.get('BB_Middle', 0)
        
        bb_position = 0
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Distance from moving average
        sma20 = latest.get('SMA_20', 0)
        if sma20:
            distance_from_sma = (current_price - sma20) / sma20 * 100
        else:
            distance_from_sma = 0
        
        score = 0
        reasons = []
        
        # RSI signals
        if rsi_oversold:
            score += 3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi_overbought:
            score -= 3
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # Bollinger Bands signals
        if bb_position < 0.2:  # Near lower band
            score += 2
            reasons.append("Near Bollinger lower band")
        elif bb_position > 0.8:  # Near upper band
            score -= 2
            reasons.append("Near Bollinger upper band")
        
        # Distance from SMA
        if distance_from_sma < -3:  # More than 3% below SMA
            score += 1
            reasons.append("Significantly below SMA20")
        elif distance_from_sma > 3:  # More than 3% above SMA
            score -= 1
            reasons.append("Significantly above SMA20")
        
        # Determine signal
        if score >= 3:
            signal = 'BUY'
        elif score <= -3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': abs(score) / 6 * 100,  # Normalize to 0-100
            'score': score,
            'reasons': reasons,
            'rsi': rsi,
            'bb_position': bb_position
        }
    
    def _breakout_strategy(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Breakout strategy based on support/resistance levels"""
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        current_price = quote.get('current_price', data.iloc[-1]['Close'])
        
        # Calculate support and resistance levels
        high_20 = data['High'].tail(20).max()
        low_20 = data['Low'].tail(20).min()
        
        # Intraday high/low
        today_high = data['High'].tail(78).max()  # Assuming 5-min intervals for 6.5 hours
        today_low = data['Low'].tail(78).min()
        
        # Volume analysis for breakout confirmation
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(3).mean()
        volume_confirmation = recent_volume > avg_volume * 1.5
        
        score = 0
        reasons = []
        
        # Check for breakouts
        resistance_breakout = current_price > high_20 * 1.001  # 0.1% buffer
        support_breakdown = current_price < low_20 * 0.999    # 0.1% buffer
        
        if resistance_breakout and volume_confirmation:
            score += 4
            reasons.append("Resistance breakout with volume")
        elif resistance_breakout:
            score += 2
            reasons.append("Resistance breakout")
        
        if support_breakdown and volume_confirmation:
            score -= 4
            reasons.append("Support breakdown with volume")
        elif support_breakdown:
            score -= 2
            reasons.append("Support breakdown")
        
        # Check proximity to levels
        resistance_proximity = abs(current_price - high_20) / high_20
        support_proximity = abs(current_price - low_20) / low_20
        
        if resistance_proximity < 0.005:  # Within 0.5%
            score += 1
            reasons.append("Near resistance level")
        
        if support_proximity < 0.005:  # Within 0.5%
            score += 1
            reasons.append("Near support level")
        
        # Determine signal
        if score >= 3:
            signal = 'BUY'
        elif score <= -3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': abs(score) / 5 * 100,
            'score': score,
            'reasons': reasons,
            'resistance': high_20,
            'support': low_20,
            'volume_confirmation': volume_confirmation
        }
    
    def _volume_analysis(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Volume-based analysis for trade confirmation"""
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        current_volume = quote.get('volume', data.iloc[-1]['Volume'])
        avg_volume_20 = data['Volume'].tail(20).mean()
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # Price-volume relationship
        price_change = quote.get('change_percent', 0)
        
        score = 0
        reasons = []
        
        # High volume analysis
        if volume_ratio > 3:
            score += 3
            reasons.append(f"Extremely high volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 2:
            score += 2
            reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 1.5:
            score += 1
            reasons.append(f"Above average volume ({volume_ratio:.1f}x avg)")
        
        # Price-volume confirmation
        if price_change > 1 and volume_ratio > 1.5:
            score += 1
            reasons.append("Volume confirms price move up")
        elif price_change < -1 and volume_ratio > 1.5:
            score -= 1
            reasons.append("Volume confirms price move down")
        
        # Low volume warning
        if volume_ratio < 0.5:
            score -= 1
            reasons.append("Low volume - lack of interest")
        
        signal = 'HOLD'
        if score >= 2:
            signal = 'BUY' if price_change >= 0 else 'SELL'
        
        return {
            'signal': signal,
            'strength': abs(score) / 4 * 100,
            'score': score,
            'reasons': reasons,
            'volume_ratio': volume_ratio
        }
    
    def _combine_signals(self, momentum: Dict, reversal: Dict, breakout: Dict, volume: Dict) -> Dict:
        """Combine multiple strategy signals with weighted scoring"""
        
        # Weights for different strategies
        weights = {
            'momentum': 0.3,
            'reversal': 0.25,
            'breakout': 0.3,
            'volume': 0.15
        }
        
        # Convert signals to numeric scores
        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        
        signals = {
            'momentum': signal_map.get(momentum.get('signal', 'HOLD'), 0),
            'reversal': signal_map.get(reversal.get('signal', 'HOLD'), 0),
            'breakout': signal_map.get(breakout.get('signal', 'HOLD'), 0),
            'volume': signal_map.get(volume.get('signal', 'HOLD'), 0)
        }
        
        # Calculate weighted score
        weighted_score = sum(signals[strategy] * weights[strategy] * 
                           (strategy_data.get('strength', 0) / 100)
                           for strategy, strategy_data in 
                           [('momentum', momentum), ('reversal', reversal), 
                            ('breakout', breakout), ('volume', volume)])
        
        # Determine final signal
        if weighted_score > 0.3:
            final_signal = 'BUY'
        elif weighted_score < -0.3:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        return {
            'signal': final_signal,
            'score': weighted_score,
            'strength': abs(weighted_score) * 100,
            'individual_signals': signals,
            'strategy_details': {
                'momentum': momentum,
                'reversal': reversal,
                'breakout': breakout,
                'volume': volume
            }
        }
    
    def _assess_risk(self, data: pd.DataFrame, quote: Dict, extended_hours: Dict) -> Dict:
        """Assess risk factors for the trade"""
        risk_factors = []
        risk_score = 0  # Lower is better
        
        current_price = quote.get('current_price', 0)
        
        # Volatility assessment
        if not data.empty and len(data) >= 20:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            if volatility > 50:
                risk_score += 3
                risk_factors.append(f"High volatility ({volatility:.1f}%)")
            elif volatility > 30:
                risk_score += 2
                risk_factors.append(f"Moderate volatility ({volatility:.1f}%)")
        
        # Gap risk from extended hours
        if extended_hours:
            premarket_price = extended_hours.get('premarket', {}).get('last_price', 0)
            if premarket_price and current_price:
                gap_percent = abs(premarket_price - current_price) / current_price * 100
                if gap_percent > 5:
                    risk_score += 2
                    risk_factors.append(f"Large gap from premarket ({gap_percent:.1f}%)")
        
        # Volume risk
        volume = quote.get('volume', 0)
        avg_volume = quote.get('avg_volume', 0)
        if avg_volume and volume < avg_volume * 0.5:
            risk_score += 1
            risk_factors.append("Low volume - liquidity risk")
        
        # Time of day risk
        now = datetime.now()
        if now.hour < 10 or now.hour > 15:  # First/last hour
            risk_score += 1
            risk_factors.append("High volatility time period")
        
        # Market cap risk
        market_cap = quote.get('market_cap', 0)
        if market_cap and market_cap < 1e9:  # Less than $1B
            risk_score += 2
            risk_factors.append("Small cap stock - higher risk")
        
        return {
            'score': risk_score,
            'level': 'HIGH' if risk_score >= 5 else 'MEDIUM' if risk_score >= 3 else 'LOW',
            'factors': risk_factors
        }
    
    def _generate_recommendation(self, combined_signal: Dict, risk_assessment: Dict, quote: Dict) -> Dict:
        """Generate final trading recommendation with position sizing"""
        
        signal = combined_signal.get('signal', 'HOLD')
        signal_strength = combined_signal.get('strength', 0)
        risk_score = risk_assessment.get('score', 0)
        current_price = quote.get('current_price', 0)
        
        # Adjust position size based on signal strength and risk
        base_position_size = self.config.MAX_POSITION_SIZE
        
        # Reduce size for high risk
        if risk_score >= 5:
            position_multiplier = 0.3
        elif risk_score >= 3:
            position_multiplier = 0.5
        else:
            position_multiplier = 0.8
        
        # Adjust for signal strength
        strength_multiplier = signal_strength / 100
        
        recommended_size = base_position_size * position_multiplier * strength_multiplier
        
        # Calculate stop loss and take profit levels
        stop_loss_price = 0
        take_profit_price = 0
        
        if signal == 'BUY' and current_price:
            stop_loss_price = current_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
            take_profit_price = current_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
        elif signal == 'SELL' and current_price:
            stop_loss_price = current_price * (1 + self.config.STOP_LOSS_PERCENTAGE)
            take_profit_price = current_price * (1 - self.config.TAKE_PROFIT_PERCENTAGE)
        
        return {
            'action': signal,
            'confidence': signal_strength,
            'position_size': recommended_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_level': risk_assessment.get('level', 'MEDIUM'),
            'max_risk_amount': recommended_size * self.config.STOP_LOSS_PERCENTAGE,
            'reasoning': combined_signal.get('strategy_details', {}),
            'timestamp': datetime.now()
        }