"""
Trading strategy engine with 6 signal legs:
  1. Momentum          – SMA crossovers, MACD, price momentum, ADX
  2. Mean Reversion    – RSI, Bollinger Bands, Stochastic, distance from SMA
  3. Breakout          – N-period high/low break with volume confirmation
  4. Volume / OBV      – Volume ratio + On-Balance Volume trend
  5. Fundamentals      – P/E, earnings growth, profit margin, revenue growth  (NEW)
  6. Relative Strength – Performance vs SPY                                   (NEW)

Market-regime overlay (VIX) modulates position sizing.
ATR-based dynamic stop losses adapt to each stock's volatility.
Earnings-awareness penalises risk when a report is imminent.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from data_provider import MarketDataProvider
from config import Config


class TradingStrategy:
    """
    Swing / position trading strategies for WEEK and MONTH horizons.
    """

    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        # Cache market regime for 30 min
        self._regime_cache: Optional[Dict] = None
        self._regime_cache_time: datetime = datetime.min

    # ── Public entry points ─────────────────────────────────────────

    def analyze_stock(self, symbol: str, horizon: str = 'WEEK') -> Dict:
        """Comprehensive analysis of a stock for a given time horizon.

        Args:
            symbol: Ticker symbol
            horizon: 'WEEK' or 'MONTH'
        """
        try:
            horizon = horizon.upper()
            if horizon not in ('WEEK', 'MONTH'):
                horizon = 'WEEK'

            quote = self.data_provider.get_real_time_quote(symbol)
            if not quote:
                return {}

            # Fetch horizon-appropriate data
            data = self._get_data_for_horizon(symbol, horizon)
            if data is None or data.empty:
                return {}

            # Market regime
            regime = self._get_market_regime()

            # Run the 6 strategy legs
            momentum_signal = self._momentum_strategy(data, quote, horizon)
            reversal_signal = self._mean_reversion_strategy(data, quote, horizon)
            breakout_signal = self._breakout_strategy(data, quote, horizon)
            volume_signal = self._volume_obv_strategy(data, quote, horizon)
            fundamental_signal = self._fundamentals_strategy(symbol)
            rs_signal = self._relative_strength_strategy(symbol, horizon)

            # Combine all 6
            combined_score = self._combine_signals(
                momentum_signal, reversal_signal, breakout_signal,
                volume_signal, fundamental_signal, rs_signal, horizon
            )

            # Risk assessment (includes earnings awareness)
            risk_score = self._assess_risk(data, quote, symbol, horizon)

            # Recommendation with ATR-based stops
            recommendation = self._generate_recommendation(
                combined_score, risk_score, quote, data, horizon, regime
            )

            return {
                'symbol': symbol,
                'horizon': horizon,
                'timestamp': datetime.now(),
                'current_price': quote.get('current_price', 0),
                'market_regime': regime,
                'signals': {
                    'momentum': momentum_signal,
                    'reversal': reversal_signal,
                    'breakout': breakout_signal,
                    'volume': volume_signal,
                    'fundamentals': fundamental_signal,
                    'relative_strength': rs_signal,
                    'combined_score': combined_score
                },
                'risk_score': risk_score,
                'recommendation': recommendation
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} ({horizon}): {e}")
            return {}

    def analyze_stock_all_horizons(self, symbol: str) -> Dict:
        """Run analysis for WEEK and MONTH horizons."""
        results = {}
        for h in ('WEEK', 'MONTH'):
            analysis = self.analyze_stock(symbol, horizon=h)
            if analysis:
                results[h] = analysis
        return results

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_market_regime(self) -> Dict:
        """Get cached market regime."""
        now = datetime.now()
        if (self._regime_cache is None or
                (now - self._regime_cache_time).total_seconds() > 1800):
            self._regime_cache = self.data_provider.get_market_regime()
            self._regime_cache_time = now
        return self._regime_cache

    def _get_data_for_horizon(self, symbol: str, horizon: str) -> Optional[pd.DataFrame]:
        if horizon == 'MONTH':
            return self.data_provider.get_weekly_data(symbol, period='1y')
        else:  # WEEK
            return self.data_provider.get_daily_data(symbol, period='3mo')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 1: Momentum
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _momentum_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'WEEK') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['Insufficient data']}

        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])

        above_sma20 = current_price > latest.get('SMA_20', 0)
        above_sma50 = current_price > latest.get('SMA_50', 0)
        above_sma200 = current_price > latest.get('SMA_200', 0) if latest.get('SMA_200', 0) > 0 else None

        macd_bullish = (latest.get('MACD', 0) > latest.get('MACD_Signal', 0) and
                       latest.get('MACD_Histogram', 0) > 0)
        macd_bearish = (latest.get('MACD', 0) < latest.get('MACD_Signal', 0) and
                       latest.get('MACD_Histogram', 0) < 0)

        lookback = 5 if horizon == 'WEEK' else 4
        if len(data) >= lookback:
            price_momentum = (current_price - data.iloc[-lookback]['Close']) / data.iloc[-lookback]['Close'] * 100
        else:
            price_momentum = 0

        adx = latest.get('ADX', 0)
        strong_trend = adx > 25

        score = 0
        reasons = []

        # SMA alignment (trend structure)
        if above_sma200 is not None:
            if above_sma20 and above_sma50 and above_sma200:
                score += 3
                reasons.append("Above SMA20/50/200 — strong uptrend")
            elif above_sma20 and above_sma50:
                score += 2
                reasons.append("Above SMA20/50")
            elif above_sma20:
                score += 1
                reasons.append("Above SMA20")
            elif not above_sma20 and not above_sma50 and not above_sma200:
                score -= 2
                reasons.append("Below all key moving averages")
            elif not above_sma20 and not above_sma50:
                score -= 1
                reasons.append("Below SMA20 and SMA50")
        else:
            if above_sma20 and above_sma50:
                score += 2
                reasons.append("Above SMA20/50")
            elif not above_sma20 and not above_sma50:
                score -= 1
                reasons.append("Below key moving averages")

        # MACD
        if macd_bullish:
            score += 2
            reasons.append("MACD bullish crossover")
        elif macd_bearish:
            score -= 2
            reasons.append("MACD bearish crossover")

        # Momentum thresholds by horizon
        mom_strong = 4.0 if horizon == 'WEEK' else 6.0
        mom_mild = 1.5 if horizon == 'WEEK' else 2.5

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

        # ADX confirmation
        if strong_trend and abs(score) >= 2:
            score += 1 if score > 0 else -1
            reasons.append(f"Strong trend confirmed (ADX {adx:.0f})")

        # Downtrend detection
        if len(data) >= 10:
            recent_highs = data['High'].tail(10)
            if (recent_highs.iloc[-1] < recent_highs.iloc[-2] < recent_highs.iloc[-3]
                    and price_momentum < -1):
                score -= 1
                reasons.append("Downtrend detected (lower highs)")

        if score >= 3:
            signal = 'BUY'
        elif score <= -2:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'strength': min(abs(score) / 9 * 100, 100),
            'score': score,
            'reasons': reasons,
            'momentum': price_momentum
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 2: Mean Reversion (+ Stochastic)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _mean_reversion_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'WEEK') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['Insufficient data']}

        latest = data.iloc[-1]
        current_price = quote.get('current_price', latest['Close'])

        rsi = latest.get('RSI', 50)
        stoch_k = latest.get('Stoch_K', 50)
        stoch_d = latest.get('Stoch_D', 50)

        bb_upper = latest.get('BB_Upper', 0)
        bb_lower = latest.get('BB_Lower', 0)
        bb_position = 0
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)

        sma20 = latest.get('SMA_20', 0)
        distance_from_sma = (current_price - sma20) / sma20 * 100 if sma20 else 0

        score = 0
        reasons = []

        # RSI
        if rsi < self.config.RSI_OVERSOLD:
            score += 3
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > self.config.RSI_OVERBOUGHT:
            score -= 3
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # Stochastic confirmation
        if stoch_k < 20 and stoch_d < 20:
            score += 2
            reasons.append(f"Stochastic oversold (K={stoch_k:.0f}, D={stoch_d:.0f})")
        elif stoch_k > 80 and stoch_d > 80:
            score -= 2
            reasons.append(f"Stochastic overbought (K={stoch_k:.0f}, D={stoch_d:.0f})")
        # Stochastic crossover
        if stoch_k > stoch_d and stoch_k < 30:
            score += 1
            reasons.append("Stochastic bullish crossover in oversold zone")
        elif stoch_k < stoch_d and stoch_k > 70:
            score -= 1
            reasons.append("Stochastic bearish crossover in overbought zone")

        # Bollinger Bands
        if bb_position < 0.10:
            score += 2
            reasons.append("At Bollinger lower band")
        elif bb_position > 0.90:
            score -= 2
            reasons.append("At Bollinger upper band")

        # Distance from SMA
        dist_thresh = 4 if horizon == 'WEEK' else 6
        if distance_from_sma < -dist_thresh:
            score += 1
            reasons.append(f"Significantly below SMA20 ({distance_from_sma:.1f}%)")
        elif distance_from_sma > dist_thresh:
            score -= 1
            reasons.append(f"Significantly above SMA20 ({distance_from_sma:.1f}%)")

        if score >= 3:
            signal = 'BUY'
        elif score <= -3:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'strength': min(abs(score) / 8 * 100, 100),
            'score': score,
            'reasons': reasons,
            'rsi': rsi,
            'stochastic': {'k': stoch_k, 'd': stoch_d},
            'bb_position': bb_position
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 3: Breakout
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _breakout_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'WEEK') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['Insufficient data']}

        current_price = quote.get('current_price', data.iloc[-1]['Close'])

        lookback = 30 if horizon == 'WEEK' else 40
        lookback = min(lookback, len(data))

        high_n = data['High'].tail(lookback).max()
        low_n = data['Low'].tail(lookback).min()

        avg_volume = data['Volume'].tail(20).mean()
        recent_volume = data['Volume'].tail(3).mean()
        volume_confirmation = recent_volume > avg_volume * 1.5

        # 52-week proximity
        pct_from_high = data.iloc[-1].get('Pct_from_52w_high', 0)

        score = 0
        reasons = []

        resistance_breakout = current_price > high_n * 1.002
        support_breakdown = current_price < low_n * 0.998

        if resistance_breakout and volume_confirmation:
            score += 4
            reasons.append("Resistance breakout with volume confirmation")
        elif resistance_breakout:
            score += 2
            reasons.append("Resistance breakout (weak volume)")

        if support_breakdown and volume_confirmation:
            score -= 4
            reasons.append("Support breakdown with volume")
        elif support_breakdown:
            score -= 2
            reasons.append("Support breakdown (weak volume)")

        # Near 52-week high is bullish for momentum breakout
        if pct_from_high is not None and pct_from_high > -3:
            score += 1
            reasons.append(f"Near 52-week high ({pct_from_high:.1f}%)")

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
            'volume_confirmation': volume_confirmation,
            'pct_from_52w_high': pct_from_high
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 4: Volume + OBV
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _volume_obv_strategy(self, data: pd.DataFrame, quote: Dict, horizon: str = 'WEEK') -> Dict:
        if data.empty or len(data) < 20:
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['Insufficient data']}

        latest = data.iloc[-1]
        current_volume = quote.get('volume', latest['Volume'])
        avg_volume_20 = data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        price_change = quote.get('change_percent', 0)

        # OBV trend
        obv = latest.get('OBV', 0)
        obv_sma = latest.get('OBV_SMA', 0)
        obv_bullish = obv > obv_sma
        obv_bearish = obv < obv_sma

        score = 0
        reasons = []

        # Volume ratio
        if volume_ratio > 3:
            score += 3
            reasons.append(f"Extremely high volume ({volume_ratio:.1f}× avg)")
        elif volume_ratio > 2:
            score += 2
            reasons.append(f"High volume ({volume_ratio:.1f}× avg)")
        elif volume_ratio > 1.5:
            score += 1
            reasons.append(f"Above average volume ({volume_ratio:.1f}× avg)")

        # Price + volume agreement
        if price_change > 1 and volume_ratio > 1.5:
            score += 1
            reasons.append("Volume confirms price move up")
        elif price_change < -1 and volume_ratio > 1.5:
            score -= 1
            reasons.append("Volume confirms price move down")

        # Low volume warning
        if volume_ratio < 0.5:
            score -= 1
            reasons.append("Low volume — lack of interest")

        # OBV trend
        if obv_bullish:
            score += 1
            reasons.append("OBV trending up (accumulation)")
        elif obv_bearish:
            score -= 1
            reasons.append("OBV trending down (distribution)")

        # OBV divergence (price up but OBV down = bearish divergence)
        if price_change > 0 and obv_bearish:
            score -= 1
            reasons.append("Bearish OBV divergence (price up, volume out)")
        elif price_change < 0 and obv_bullish:
            score += 1
            reasons.append("Bullish OBV divergence (price down, volume in)")

        signal = 'HOLD'
        if score >= 2:
            signal = 'BUY' if price_change >= 0 else 'SELL'
        elif score <= -2:
            signal = 'SELL' if price_change <= 0 else 'HOLD'

        return {
            'signal': signal,
            'strength': min(abs(score) / 6 * 100, 100),
            'score': score,
            'reasons': reasons,
            'volume_ratio': volume_ratio,
            'obv_bullish': obv_bullish
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 5: Fundamentals  ★ NEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fundamentals_strategy(self, symbol: str) -> Dict:
        """Score a stock based on fundamental quality."""
        try:
            fund = self.data_provider.get_stock_fundamentals(symbol)
            if not fund:
                return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['No fundamental data']}

            score = 0
            reasons = []

            # P/E ratio
            pe = fund.get('pe_ratio', 0)
            if 0 < pe < 15:
                score += 2
                reasons.append(f"Attractive P/E ({pe:.1f})")
            elif 0 < pe < 25:
                score += 1
                reasons.append(f"Reasonable P/E ({pe:.1f})")
            elif pe > self.config.MAX_PE_RATIO:
                score -= 2
                reasons.append(f"Very high P/E ({pe:.1f})")
            elif pe > 40:
                score -= 1
                reasons.append(f"Elevated P/E ({pe:.1f})")

            # Earnings growth
            eg = fund.get('earnings_growth', 0) or 0
            if eg > 0.20:
                score += 2
                reasons.append(f"Strong earnings growth ({eg*100:.0f}%)")
            elif eg > 0.05:
                score += 1
                reasons.append(f"Positive earnings growth ({eg*100:.0f}%)")
            elif eg < -0.10:
                score -= 1
                reasons.append(f"Earnings declining ({eg*100:.0f}%)")

            # Revenue growth
            rg = fund.get('revenue_growth', 0) or 0
            if rg > 0.15:
                score += 1
                reasons.append(f"Strong revenue growth ({rg*100:.0f}%)")
            elif rg < -0.05:
                score -= 1
                reasons.append(f"Revenue declining ({rg*100:.0f}%)")

            # Profit margins
            pm = fund.get('profit_margin', 0) or 0
            if pm > 0.20:
                score += 1
                reasons.append(f"High profit margin ({pm*100:.0f}%)")
            elif pm < 0:
                score -= 1
                reasons.append(f"Unprofitable (margin {pm*100:.0f}%)")

            # ROE
            roe = fund.get('return_on_equity', 0) or 0
            if roe > 0.15:
                score += 1
                reasons.append(f"Strong ROE ({roe*100:.0f}%)")
            elif roe < 0:
                score -= 1
                reasons.append(f"Negative ROE ({roe*100:.0f}%)")

            # Debt-to-equity
            de = fund.get('debt_to_equity', 0) or 0
            if 0 < de < 50:
                score += 1
                reasons.append(f"Low debt (D/E {de:.0f})")
            elif de > 200:
                score -= 1
                reasons.append(f"High debt (D/E {de:.0f})")

            # Market cap filter
            mc = fund.get('market_cap', 0) or 0
            if mc < self.config.MIN_MARKET_CAP and mc > 0:
                score -= 1
                reasons.append(f"Small cap (${mc/1e6:.0f}M)")

            if score >= 3:
                signal = 'BUY'
            elif score <= -2:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'signal': signal,
                'strength': min(abs(score) / 7 * 100, 100),
                'score': score,
                'reasons': reasons,
                'pe_ratio': pe,
                'earnings_growth': eg,
                'profit_margin': pm,
                'sector': fund.get('sector', 'Unknown')
            }

        except Exception as e:
            self.logger.error(f"Error in fundamentals strategy for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': [f'Error: {e}']}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STRATEGY 6: Relative Strength vs SPY  ★ NEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _relative_strength_strategy(self, symbol: str, horizon: str = 'WEEK') -> Dict:
        """Compare stock performance to SPY."""
        try:
            period = '3mo' if horizon == 'WEEK' else '6mo'
            rs = self.data_provider.get_relative_strength(symbol, period=period)
            if not rs:
                return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': ['No RS data']}

            relative = rs.get('relative_strength', 0)
            stock_ret = rs.get('stock_return', 0)
            spy_ret = rs.get('spy_return', 0)

            score = 0
            reasons = []

            if relative > 10:
                score += 3
                reasons.append(f"Strong outperformance vs SPY (+{relative:.1f}%)")
            elif relative > 5:
                score += 2
                reasons.append(f"Outperforming SPY (+{relative:.1f}%)")
            elif relative > 0:
                score += 1
                reasons.append(f"Slightly outperforming SPY (+{relative:.1f}%)")
            elif relative < -10:
                score -= 2
                reasons.append(f"Significantly underperforming SPY ({relative:.1f}%)")
            elif relative < -5:
                score -= 1
                reasons.append(f"Underperforming SPY ({relative:.1f}%)")

            if score >= 2:
                signal = 'BUY'
            elif score <= -1:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'signal': signal,
                'strength': min(abs(score) / 3 * 100, 100),
                'score': score,
                'reasons': reasons,
                'stock_return': stock_ret,
                'spy_return': spy_ret,
                'relative_strength': relative
            }

        except Exception as e:
            self.logger.error(f"Error in RS strategy for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'score': 0, 'reasons': [f'Error: {e}']}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Signal combiner (6 legs)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _combine_signals(self, momentum: Dict, reversal: Dict, breakout: Dict,
                         volume: Dict, fundamentals: Dict, relative_strength: Dict,
                         horizon: str = 'WEEK') -> Dict:
        """Combine 6 strategy signals into one recommendation."""

        # Weights differ by horizon
        if horizon == 'MONTH':
            # Position trading: fundamentals & RS weigh more
            weights = {
                'momentum': 0.15, 'reversal': 0.10, 'breakout': 0.15,
                'volume': 0.10, 'fundamentals': 0.25, 'relative_strength': 0.25
            }
        else:  # WEEK
            # Swing trading: technicals still dominate, but fundamentals matter
            weights = {
                'momentum': 0.20, 'reversal': 0.15, 'breakout': 0.20,
                'volume': 0.10, 'fundamentals': 0.15, 'relative_strength': 0.20
            }

        signal_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}

        strategies = {
            'momentum': momentum,
            'reversal': reversal,
            'breakout': breakout,
            'volume': volume,
            'fundamentals': fundamentals,
            'relative_strength': relative_strength
        }

        signals = {
            name: signal_map.get(strat.get('signal', 'HOLD'), 0)
            for name, strat in strategies.items()
        }

        weighted_score = sum(
            signals[name] * weights[name] * (strategies[name].get('strength', 0) / 100)
            for name in strategies
        )

        # Threshold
        threshold = 0.18

        if weighted_score > threshold:
            final_signal = 'BUY'
        elif weighted_score < -threshold:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        # Require at least 3 strategies to agree (stricter with 6 legs)
        buy_count = sum(1 for v in signals.values() if v > 0)
        sell_count = sum(1 for v in signals.values() if v < 0)

        if final_signal == 'BUY' and buy_count < 3:
            final_signal = 'HOLD'
        if final_signal == 'SELL' and sell_count < 3:
            final_signal = 'HOLD'

        if abs(weighted_score) > 0.12:
            self.logger.info(
                f"Signal ({horizon}): {final_signal} (score: {weighted_score:.3f}, "
                f"buy_agree: {buy_count}, sell_agree: {sell_count})"
            )

        return {
            'signal': final_signal,
            'score': weighted_score,
            'strength': min(abs(weighted_score) * 100, 100),
            'individual_signals': signals,
            'buy_agreement': buy_count,
            'sell_agreement': sell_count,
            'strategy_details': strategies
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Risk assessment (with earnings awareness)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _assess_risk(self, data: pd.DataFrame, quote: Dict,
                     symbol: str, horizon: str = 'WEEK') -> Dict:
        risk_factors = []
        risk_score = 0

        current_price = quote.get('current_price', 0)

        # Volatility
        if not data.empty and len(data) >= 20:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100

            vol_high = 40 if horizon == 'WEEK' else 35
            vol_mod = 25 if horizon == 'WEEK' else 20

            if volatility > vol_high:
                risk_score += 3
                risk_factors.append(f"High volatility ({volatility:.1f}%)")
            elif volatility > vol_mod:
                risk_score += 2
                risk_factors.append(f"Moderate volatility ({volatility:.1f}%)")

        # Low volume / liquidity
        volume = quote.get('volume', 0)
        avg_volume = quote.get('avg_volume', 0)
        if avg_volume and volume < avg_volume * 0.5:
            risk_score += 1
            risk_factors.append("Low volume — liquidity risk")

        # Market cap
        market_cap = quote.get('market_cap', 0)
        if market_cap and market_cap < 1e9:
            risk_score += 2
            risk_factors.append("Small cap stock — higher risk")
        elif market_cap and market_cap < 5e9:
            risk_score += 1
            risk_factors.append("Mid-small cap")

        # ── Earnings proximity ── NEW ──────────────────────
        try:
            earnings_date = self.data_provider.get_next_earnings_date(symbol)
            if earnings_date:
                days_to_earnings = (earnings_date - datetime.now()).days
                if 0 <= days_to_earnings <= 3:
                    risk_score += 3
                    risk_factors.append(f"Earnings in {days_to_earnings} days — HIGH RISK")
                elif 0 <= days_to_earnings <= 7:
                    risk_score += 2
                    risk_factors.append(f"Earnings in {days_to_earnings} days")
                elif 0 <= days_to_earnings <= 14:
                    risk_score += 1
                    risk_factors.append(f"Earnings in {days_to_earnings} days")
        except Exception:
            pass  # Earnings data not critical

        # Market regime penalty
        regime = self._get_market_regime()
        if regime.get('regime') == 'HIGH_VOL':
            risk_score += 1
            risk_factors.append(f"High VIX ({regime.get('vix', 0):.1f})")
        elif regime.get('regime') == 'EXTREME_FEAR':
            risk_score += 2
            risk_factors.append(f"Extreme fear (VIX {regime.get('vix', 0):.1f})")

        return {
            'score': risk_score,
            'level': 'HIGH' if risk_score >= 5 else 'MEDIUM' if risk_score >= 3 else 'LOW',
            'factors': risk_factors
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recommendation with ATR-based stops
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_recommendation(self, combined_signal: Dict, risk_assessment: Dict,
                                  quote: Dict, data: pd.DataFrame,
                                  horizon: str = 'WEEK',
                                  regime: Optional[Dict] = None) -> Dict:
        signal = combined_signal.get('signal', 'HOLD')
        signal_strength = combined_signal.get('strength', 0)
        risk_score = risk_assessment.get('score', 0)
        current_price = quote.get('current_price', 0)

        hp = self.config.get_horizon_params(horizon)

        # ── ATR-based dynamic stops ────────────────────────
        atr = 0
        if not data.empty:
            atr = float(data.iloc[-1].get('ATR', 0))

        if atr > 0 and current_price > 0:
            # Use ATR multipliers from config
            stop_loss_distance = atr * hp['atr_sl_mult']
            take_profit_distance = atr * hp['atr_tp_mult']
            # Clamp to percentage-based limits as floor/ceiling
            min_sl = current_price * hp['stop_loss_pct']
            min_tp = current_price * hp['take_profit_pct']
            stop_loss_distance = max(stop_loss_distance, min_sl)
            take_profit_distance = max(take_profit_distance, min_tp)
        else:
            stop_loss_distance = current_price * hp['stop_loss_pct']
            take_profit_distance = current_price * hp['take_profit_pct']

        # ── Position sizing ────────────────────────────────
        base_position_size = self.config.MAX_POSITION_SIZE

        # Risk-based multiplier
        if risk_score >= 5:
            risk_mult = 0.25
        elif risk_score >= 3:
            risk_mult = 0.5
        else:
            risk_mult = 0.8

        # Signal strength multiplier
        strength_mult = signal_strength / 100

        # Market regime multiplier
        regime_mult = 1.0
        if regime:
            regime_mult = regime.get('position_multiplier', 1.0)

        recommended_size = base_position_size * risk_mult * strength_mult * regime_mult

        # Stop / take-profit prices
        stop_loss_price = 0
        take_profit_price = 0

        if signal == 'BUY' and current_price:
            stop_loss_price = current_price - stop_loss_distance
            take_profit_price = current_price + take_profit_distance
        elif signal == 'SELL' and current_price:
            stop_loss_price = current_price + stop_loss_distance
            take_profit_price = current_price - take_profit_distance

        return {
            'action': signal,
            'confidence': signal_strength,
            'position_size': recommended_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_level': risk_assessment.get('level', 'MEDIUM'),
            'max_risk_amount': recommended_size * hp['stop_loss_pct'],
            'horizon': horizon,
            'atr': atr,
            'atr_stop_distance': stop_loss_distance if atr > 0 else 0,
            'regime_multiplier': regime_mult,
            'reasoning': combined_signal.get('strategy_details', {}),
            'timestamp': datetime.now()
        }
