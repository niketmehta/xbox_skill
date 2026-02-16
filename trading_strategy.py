import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from data_provider import MarketDataProvider
from config import Config


class TradingStrategy:
    """
    Multi-timeframe trading strategies that produce recommendations for
    DAY (intraday), WEEK (swing), and MONTH (position) horizons.
    
    Each horizon fetches the appropriate data resolution, runs the same
    core signal generators, and then applies horizon-specific thresholds
    and position-sizing rules.
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
    
    # ── Public entry points ─────────────────────────────────────────

    def analyze_stock(self, symbol: str, horizon: str = 'DAY') -> Dict:
        """Comprehensive analysis of a stock for a given time horizon.
        
        Args:
            symbol: Ticker symbol
            horizon: 'DAY', 'WEEK', or 'MONTH'
        """
        try:
            horizon = horizon.upper()
            quote = self.data_provider.get_real_time_quote(symbol)
            if not quote:
                return {}
            
            # Fetch horizon-appropriate data
            data = self._get_data_for_horizon(symbol, horizon)
            if data is None or data.empty:
                return {}
            
            # Extended hours (only relevant for DAY)
            extended_hours = {}
            if horizon == 'DAY':
                extended_hours = self.data_provider.get_extended_hours_data(symbol)
            
            # Run strategy analyses
            momentum_signal = self._momentum_strategy(data, quote, horizon)
            reversal_signal = self._mean_reversion_strategy(data, quote, horizon)
            breakout_signal = self._breakout_strategy(data, quote, horizon)
            volume_signal = self._volume_analysis(data, quote, horizon)
            
            # Combine
            combined_score = self._combine_signals(
                momentum_signal, reversal_signal, breakout_signal, volume_signal, horizon
            )
            
            # Risk
            risk_score = self._assess_risk(data, quote, extended_hours, horizon)
            
            # Recommendation
            recommendation = self._generate_recommendation(
                combined_score, risk_score, quote, horizon
            )
            
            return {
                'symbol': symbol,
                'horizon': horizon,
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
                'recommendation': recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} ({horizon}): {e}")
            return {}
    
    def analyze_stock_all_horizons(self, symbol: str) -> Dict:
        """Run analysis for all three horizons and return combined result."""
        results = {}
        for h in ('DAY', 'WEEK', 'MONTH'):
            analysis = self.analyze_stock(symbol, horizon=h)
            if analysis:
                results[h] = analysis
        return results
    
    # ── Data fetching per horizon ───────────────────────────────────

    def _get_data_for_horizon(self, symbol: str, horizon: str) -> Optional[pd.DataFrame]:
        if horizon == 'WEEK':
            return self.data_provider.get_daily_data(symbol, period='3mo')
        elif horizon == 'MONTH':
            return self.data_provider.get_weekly_data(symbol, period='1y')
        else:  # DAY
            return self.data_provider.get_intraday_data(symbol, period='5d', interval='5m')
    
    # ── Momentum strategy ───────────────────────────────────────────

    def _momentum_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'DAY') -> Dict:
        min_bars = 50 if horizon == 'DAY' else 20
        if data.empty or len(data) < min_bars:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])
        
        above_sma20 = current_price > latest.get('SMA_20', 0)
        above_sma50 = current_price > latest.get('SMA_50', 0)
        
        macd_bullish = (latest.get('MACD', 0) > latest.get('MACD_Signal', 0) and
                       latest.get('MACD_Histogram', 0) > 0)
        macd_bearish = (latest.get('MACD', 0) < latest.get('MACD_Signal', 0) and
                       latest.get('MACD_Histogram', 0) < 0)
        
        lookback = 5 if horizon == 'DAY' else (3 if horizon == 'WEEK' else 4)
        if len(data) >= lookback:
            price_momentum = (current_price - data.iloc[-lookback]['Close']) / data.iloc[-lookback]['Close'] * 100
        else:
            price_momentum = 0
        
        avg_volume = data['Volume'].tail(20).mean()
        current_volume = quote.get('volume', latest['Volume'])
        volume_spike = current_volume > avg_volume * self.config.VOLUME_SPIKE_THRESHOLD
        
        # ADX trend-strength filter
        adx = latest.get('ADX', 0)
        strong_trend = adx > 25
        
        score = 0
        reasons = []
        
        if above_sma20 and above_sma50:
            score += 2
            reasons.append("Above key moving averages")
        elif above_sma20:
            score += 1
            reasons.append("Above SMA20")
        elif not above_sma20 and not above_sma50:
            score -= 1
            reasons.append("Below key moving averages")
        
        if macd_bullish:
            score += 2
            reasons.append("MACD bullish crossover")
        elif macd_bearish:
            score -= 2
            reasons.append("MACD bearish crossover")
        
        # Momentum thresholds scale with horizon
        mom_strong = 2.0 if horizon == 'DAY' else (4.0 if horizon == 'WEEK' else 6.0)
        mom_mild = 0.5 if horizon == 'DAY' else (1.5 if horizon == 'WEEK' else 2.5)
        
        if price_momentum > mom_strong:
            score += 2
            reasons.append(f"Strong momentum ({price_momentum:.1f}%)")
        elif price_momentum > mom_mild:
            score += 1
            reasons.append(f"Positive momentum ({price_momentum:.1f}%)")
        elif price_momentum < -mom_strong:
            score -= 2
            reasons.append(f"Strong negative momentum ({price_momentum:.1f}%)")
        elif price_momentum < -mom_mild:
            score -= 1
            reasons.append(f"Negative momentum ({price_momentum:.1f}%)")
        
        if volume_spike:
            score += 1
            reasons.append("Volume spike detected")
        
        # ADX confirmation bonus
        if strong_trend and abs(score) >= 2:
            score += 1 if score > 0 else -1
            reasons.append(f"Strong trend (ADX {adx:.0f})")
        
        # Downtrend detection
        if len(data) >= 10:
            recent_highs = data['High'].tail(10)
            if (recent_highs.iloc[-1] < recent_highs.iloc[-2] < recent_highs.iloc[-3]
                    and price_momentum < -1):
                score -= 1
                reasons.append("Downtrend detected (lower highs)")
            
            if current_price < recent_highs.min() and price_momentum < -2:
                score -= 2
                reasons.append("Strong downtrend")
        
        # Tighter thresholds than before
        if score >= 3:
            signal = 'BUY'
        elif score <= -2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': min(abs(score) / 8 * 100, 100),
            'score': score,
            'reasons': reasons,
            'momentum': price_momentum
        }
    
    # ── Mean-reversion strategy ─────────────────────────────────────

    def _mean_reversion_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'DAY') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])
        
        rsi = latest.get('RSI', 50)
        rsi_oversold = rsi < self.config.RSI_OVERSOLD
        rsi_overbought = rsi > self.config.RSI_OVERBOUGHT
        
        bb_upper = latest.get('BB_Upper', 0)
        bb_lower = latest.get('BB_Lower', 0)
        
        bb_position = 0
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        sma20 = latest.get('SMA_20', 0)
        distance_from_sma = (current_price - sma20) / sma20 * 100 if sma20 else 0
        
        score = 0
        reasons = []
        
        if rsi_oversold:
            score += 3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi_overbought:
            score -= 3
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        if bb_position < 0.15:
            score += 2
            reasons.append("Near Bollinger lower band")
        elif bb_position > 0.85:
            score -= 2
            reasons.append("Near Bollinger upper band")
        
        dist_thresh = 3 if horizon == 'DAY' else (4 if horizon == 'WEEK' else 6)
        if distance_from_sma < -dist_thresh:
            score += 1
            reasons.append(f"Significantly below SMA20 ({distance_from_sma:.1f}%)")
        elif distance_from_sma > dist_thresh:
            score -= 1
            reasons.append(f"Significantly above SMA20 ({distance_from_sma:.1f}%)")
        
        # Tighter thresholds
        if score >= 2:
            signal = 'BUY'
        elif score <= -2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': min(abs(score) / 6 * 100, 100),
            'score': score,
            'reasons': reasons,
            'rsi': rsi,
            'bb_position': bb_position
        }
    
    # ── Breakout strategy ───────────────────────────────────────────

    def _breakout_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'DAY') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        current_price = quote.get('current_price', data.iloc[-1]['Close'])
        
        lookback = 20 if horizon == 'DAY' else (30 if horizon == 'WEEK' else 40)
        lookback = min(lookback, len(data))
        
        high_n = data['High'].tail(lookback).max()
        low_n = data['Low'].tail(lookback).min()
        
        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(3).mean()
        volume_confirmation = recent_volume > avg_volume * 1.5
        
        score = 0
        reasons = []
        
        resistance_breakout = current_price > high_n * 1.002
        support_breakdown = current_price < low_n * 0.998
        
        if resistance_breakout and volume_confirmation:
            score += 4
            reasons.append("Resistance breakout with volume")
        elif resistance_breakout:
            score += 2
            reasons.append("Resistance breakout (no vol confirm)")
        
        if support_breakdown and volume_confirmation:
            score -= 4
            reasons.append("Support breakdown with volume")
        elif support_breakdown:
            score -= 2
            reasons.append("Support breakdown (no vol confirm)")
        
        # Tighter thresholds
        if score >= 2:
            signal = 'BUY'
        elif score <= -2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'strength': min(abs(score) / 5 * 100, 100),
            'score': score,
            'reasons': reasons,
            'resistance': high_n,
            'support': low_n,
            'volume_confirmation': volume_confirmation
        }
    
    # ── Volume analysis ─────────────────────────────────────────────

    def _volume_analysis(self, data: pd.DataFrame, quote: Dict, horizon: str = 'DAY') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
        
        current_volume = quote.get('volume', data.iloc[-1]['Volume'])
        avg_volume_20 = data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        price_change = quote.get('change_percent', 0)
        
        score = 0
        reasons = []
        
        if volume_ratio > 3:
            score += 3
            reasons.append(f"Extremely high volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 2:
            score += 2
            reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 1.5:
            score += 1
            reasons.append(f"Above average volume ({volume_ratio:.1f}x avg)")
        
        if price_change > 1 and volume_ratio > 1.5:
            score += 1
            reasons.append("Volume confirms price move up")
        elif price_change < -1 and volume_ratio > 1.5:
            score -= 1
            reasons.append("Volume confirms price move down")
        
        if volume_ratio < 0.5:
            score -= 1
            reasons.append("Low volume - lack of interest")
        
        signal = 'HOLD'
        if score >= 2:
            signal = 'BUY' if price_change >= 0 else 'SELL'
        
        return {
            'signal': signal,
            'strength': min(abs(score) / 4 * 100, 100),
            'score': score,
            'reasons': reasons,
            'volume_ratio': volume_ratio
        }
    
    # ── Signal combiner ─────────────────────────────────────────────

    def _combine_signals(self, momentum: Dict, reversal: Dict,
                         breakout: Dict, volume: Dict, horizon: str = 'DAY') -> Dict:
        # Horizon-specific weights
        if horizon == 'MONTH':
            weights = {'momentum': 0.35, 'reversal': 0.20, 'breakout': 0.30, 'volume': 0.15}
        elif horizon == 'WEEK':
            weights = {'momentum': 0.30, 'reversal': 0.25, 'breakout': 0.30, 'volume': 0.15}
        else:
            weights = {'momentum': 0.30, 'reversal': 0.25, 'breakout': 0.30, 'volume': 0.15}
        
        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        
        signals = {
            'momentum': signal_map.get(momentum.get('signal', 'HOLD'), 0),
            'reversal': signal_map.get(reversal.get('signal', 'HOLD'), 0),
            'breakout': signal_map.get(breakout.get('signal', 'HOLD'), 0),
            'volume': signal_map.get(volume.get('signal', 'HOLD'), 0)
        }
        
        weighted_score = sum(
            signals[strategy] * weights[strategy] * (strategy_data.get('strength', 0) / 100)
            for strategy, strategy_data in
            [('momentum', momentum), ('reversal', reversal),
             ('breakout', breakout), ('volume', volume)]
        )
        
        # ── Tighter thresholds (restored from overly-loose values) ──
        # Require stronger agreement before signalling BUY/SELL
        threshold = 0.20  # was 0.1
        
        if weighted_score > threshold:
            final_signal = 'BUY'
        elif weighted_score < -threshold:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Require at least 2 strategies to agree for a signal
        buy_count = sum(1 for v in signals.values() if v > 0)
        sell_count = sum(1 for v in signals.values() if v < 0)
        
        if final_signal == 'BUY' and buy_count < 2:
            final_signal = 'HOLD'
        if final_signal == 'SELL' and sell_count < 2:
            final_signal = 'HOLD'
        
        if abs(weighted_score) > 0.15:
            self.logger.info(f"Signal generated ({horizon}): {final_signal} (score: {weighted_score:.3f})")
        
        return {
            'signal': final_signal,
            'score': weighted_score,
            'strength': min(abs(weighted_score) * 100, 100),
            'individual_signals': signals,
            'strategy_details': {
                'momentum': momentum,
                'reversal': reversal,
                'breakout': breakout,
                'volume': volume
            }
        }
    
    # ── Risk assessment ─────────────────────────────────────────────

    def _assess_risk(self, data: pd.DataFrame, quote: Dict,
                     extended_hours: Dict, horizon: str = 'DAY') -> Dict:
        risk_factors = []
        risk_score = 0
        
        current_price = quote.get('current_price', 0)
        
        if not data.empty and len(data) >= 20:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            vol_high = 50 if horizon == 'DAY' else (40 if horizon == 'WEEK' else 35)
            vol_mod = 30 if horizon == 'DAY' else (25 if horizon == 'WEEK' else 20)
            
            if volatility > vol_high:
                risk_score += 3
                risk_factors.append(f"High volatility ({volatility:.1f}%)")
            elif volatility > vol_mod:
                risk_score += 2
                risk_factors.append(f"Moderate volatility ({volatility:.1f}%)")
        
        if extended_hours and horizon == 'DAY':
            premarket_price = extended_hours.get('premarket', {}).get('last_price', 0)
            if premarket_price and current_price:
                gap_percent = abs(premarket_price - current_price) / current_price * 100
                if gap_percent > 5:
                    risk_score += 2
                    risk_factors.append(f"Large gap from premarket ({gap_percent:.1f}%)")
        
        volume = quote.get('volume', 0)
        avg_volume = quote.get('avg_volume', 0)
        if avg_volume and volume < avg_volume * 0.5:
            risk_score += 1
            risk_factors.append("Low volume - liquidity risk")
        
        if horizon == 'DAY':
            now = datetime.now()
            if now.hour < 10 or now.hour > 15:
                risk_score += 1
                risk_factors.append("High volatility time period")
        
        market_cap = quote.get('market_cap', 0)
        if market_cap and market_cap < 1e9:
            risk_score += 2
            risk_factors.append("Small cap stock - higher risk")
        
        return {
            'score': risk_score,
            'level': 'HIGH' if risk_score >= 5 else 'MEDIUM' if risk_score >= 3 else 'LOW',
            'factors': risk_factors
        }
    
    # ── Recommendation generator ────────────────────────────────────

    def _generate_recommendation(self, combined_signal: Dict, risk_assessment: Dict,
                                  quote: Dict, horizon: str = 'DAY') -> Dict:
        signal = combined_signal.get('signal', 'HOLD')
        signal_strength = combined_signal.get('strength', 0)
        risk_score = risk_assessment.get('score', 0)
        current_price = quote.get('current_price', 0)
        
        hp = self.config.get_horizon_params(horizon)
        stop_loss_pct = hp['stop_loss_pct']
        take_profit_pct = hp['take_profit_pct']
        
        base_position_size = self.config.MAX_POSITION_SIZE
        
        if risk_score >= 5:
            position_multiplier = 0.3
        elif risk_score >= 3:
            position_multiplier = 0.5
        else:
            position_multiplier = 0.8
        
        strength_multiplier = signal_strength / 100
        recommended_size = base_position_size * position_multiplier * strength_multiplier
        
        stop_loss_price = 0
        take_profit_price = 0
        
        if signal == 'BUY' and current_price:
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)
        elif signal == 'SELL' and current_price:
            stop_loss_price = current_price * (1 + stop_loss_pct)
            take_profit_price = current_price * (1 - take_profit_pct)
        
        return {
            'action': signal,
            'confidence': signal_strength,
            'position_size': recommended_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_level': risk_assessment.get('level', 'MEDIUM'),
            'max_risk_amount': recommended_size * stop_loss_pct,
            'horizon': horizon,
            'reasoning': combined_signal.get('strategy_details', {}),
            'timestamp': datetime.now()
        }
