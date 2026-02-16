import logging
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from config import Config
from data_provider import MarketDataProvider
from trading_strategy import TradingStrategy
from portfolio_manager import PortfolioManager
from stock_screener import StockScreener
from notifications import NotificationService


class TradingAgent:
    """
    Main trading agent that orchestrates multi-timeframe market analysis,
    strategy execution, risk management, and SMS notifications.
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = self._setup_logging()
        
        # Components
        self.data_provider = MarketDataProvider()
        self.trading_strategy = TradingStrategy(self.data_provider)
        self.portfolio_manager = PortfolioManager(self.data_provider)
        self.stock_screener = StockScreener(self.data_provider)
        self.notifications = NotificationService()
        
        # Agent state
        self.is_running = False
        self.is_market_hours = False
        self.last_scan_time = datetime.min
        self.scan_interval = 120  # seconds between scans
        self.last_after_hours_scan_time = datetime.min
        self.after_hours_scan_interval = 120
        
        # Watchlist
        self.watchlist = self._get_default_watchlist()
        self.auto_watchlist_enabled = True
        self.last_watchlist_update = datetime.min
        self.recommendations: Dict[str, Dict] = {}  # symbol -> {DAY, WEEK, MONTH}
        
        self.logger.info("Trading Agent initialised (multi-timeframe mode)")
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_agent.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _get_default_watchlist(self) -> List[str]:
        return [
            'SPY', 'QQQ', 'IWM',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'AMD', 'NFLX', 'META', 'BABA',
            'JPM', 'BAC', 'XOM', 'CVX',
            'DIS', 'V', 'UNH', 'JNJ', 'PG'
        ]
    
    # ── Agent lifecycle ─────────────────────────────────────────────

    def start(self):
        self.is_running = True
        self.logger.info("Starting Trading Agent...")
        
        schedule.every().day.at("06:00").do(self._daily_startup)
        schedule.every().day.at("16:00").do(self._daily_shutdown)
        schedule.every().hour.do(self._hourly_portfolio_snapshot)
        
        try:
            while self.is_running:
                self._check_market_status()
                if self.is_market_hours:
                    self._trading_loop()
                else:
                    self._after_hours_monitoring()
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            self.logger.info("Trading Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Trading Agent error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        self.is_running = False
        self.logger.info("Stopping Trading Agent...")
        if self.data_provider.is_market_open():
            self.portfolio_manager.liquidate_day_positions("Agent shutdown")
        self.portfolio_manager.save_portfolio_snapshot()
        self.logger.info("Trading Agent stopped")
    
    def _check_market_status(self):
        self.is_market_hours = (
            self.data_provider.is_market_open() or
            self.data_provider.is_extended_hours()
        )
    
    # ── Main trading loop ───────────────────────────────────────────

    def _trading_loop(self):
        now = datetime.now()
        if (now - self.last_scan_time).total_seconds() < self.scan_interval:
            return
        self.last_scan_time = now
        
        try:
            self.portfolio_manager.update_positions()
            self.portfolio_manager.check_stop_loss_take_profit()
            
            if self.portfolio_manager.should_liquidate_for_risk():
                self.portfolio_manager.liquidate_all_positions("Risk management")
                self.notifications.notify_risk_alert(
                    "All positions liquidated due to risk limits"
                )
                return
            
            if self.portfolio_manager.is_end_of_day():
                # Only liquidate DAY positions; keep WEEK/MONTH
                self.portfolio_manager.liquidate_day_positions("End of day")
                return
            
            if self.auto_watchlist_enabled:
                self._check_watchlist_update()
            
            self._scan_watchlist()
            self._execute_trades()
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
    
    # ── Multi-horizon scanning ──────────────────────────────────────

    def _scan_watchlist(self):
        self.logger.info("Scanning watchlist for opportunities (multi-timeframe)...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for symbol in self.watchlist:
                future = executor.submit(self._analyze_all_horizons, symbol)
                futures[future] = symbol
            
            for future in futures:
                symbol = futures[future]
                try:
                    all_horizons = future.result(timeout=30)
                    if all_horizons:
                        self.recommendations[symbol] = all_horizons
                        
                        # Notify on strong signals
                        for horizon, analysis in all_horizons.items():
                            rec = analysis.get('recommendation', {})
                            action = rec.get('action', 'HOLD')
                            confidence = rec.get('confidence', 0)
                            hp = self.config.get_horizon_params(horizon)
                            
                            if action in ('BUY', 'SELL') and confidence >= hp['min_confidence']:
                                self.logger.info(
                                    f"{symbol} [{horizon}]: {action} @ {confidence:.1f}% confidence"
                                )
                                self.notifications.notify_signal(
                                    symbol, action, confidence,
                                    analysis.get('current_price', 0), horizon
                                )
                except Exception as e:
                    self.logger.error(f"Error analysing {symbol}: {e}")
    
    def _analyze_all_horizons(self, symbol: str) -> Dict:
        """Analyse a symbol across DAY, WEEK, MONTH horizons."""
        results = {}
        for h in ('DAY', 'WEEK', 'MONTH'):
            try:
                analysis = self.trading_strategy.analyze_stock(symbol, horizon=h)
                if analysis:
                    results[h] = analysis
            except Exception as e:
                self.logger.error(f"Error analysing {symbol} [{h}]: {e}")
        return results
    
    # ── Trade execution ─────────────────────────────────────────────

    def _execute_trades(self):
        for symbol, horizons in self.recommendations.items():
            for horizon, analysis in horizons.items():
                try:
                    rec = analysis.get('recommendation', {})
                    action = rec.get('action', 'HOLD')
                    confidence = rec.get('confidence', 0)
                    position_size = rec.get('position_size', 0)
                    stop_loss = rec.get('stop_loss', 0)
                    take_profit = rec.get('take_profit', 0)
                    
                    if symbol in self.portfolio_manager.positions:
                        continue
                    
                    hp = self.config.get_horizon_params(horizon)
                    if confidence < hp['min_confidence']:
                        continue
                    
                    current_price = analysis.get('current_price', 0)
                    if not current_price:
                        continue
                    
                    quantity = int(position_size / current_price)
                    
                    if action == 'BUY' and quantity > 0:
                        success = self.portfolio_manager.open_position(
                            symbol=symbol, quantity=quantity,
                            entry_price=current_price, position_type='LONG',
                            stop_loss=stop_loss, take_profit=take_profit,
                            horizon=horizon
                        )
                        if success:
                            self.logger.info(
                                f"Executed BUY: {symbol} x{quantity} "
                                f"@ ${current_price:.2f} [{horizon}]"
                            )
                            self.notifications.notify_trade_opened(
                                symbol, 'BUY', quantity, current_price, horizon
                            )
                    
                    # Once a position is opened for any horizon, skip the rest
                    if symbol in self.portfolio_manager.positions:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error executing trade for {symbol} [{horizon}]: {e}")
    
    # ── After hours ─────────────────────────────────────────────────

    def _after_hours_monitoring(self):
        if not self.data_provider.is_extended_hours():
            return
        now = datetime.now()
        if (now - self.last_after_hours_scan_time).total_seconds() < self.after_hours_scan_interval:
            return
        self.last_after_hours_scan_time = now
        
        try:
            movers = self.data_provider.get_market_movers(20)
            for mover in movers[:5]:
                symbol = mover.get('symbol')
                change_percent = mover.get('change_percent', 0)
                if abs(change_percent) > 3:
                    self.data_provider.get_extended_hours_data(symbol)
                    self.logger.info(f"Extended hours activity: {symbol} {change_percent:+.1f}%")
                    if symbol not in self.watchlist:
                        self.watchlist.append(symbol)
        except Exception as e:
            self.logger.error(f"Error in after hours monitoring: {e}")
    
    # ── Daily routines ──────────────────────────────────────────────

    def _daily_startup(self):
        self.logger.info("Daily startup routine...")
        self.portfolio_manager.reset_daily_metrics()
        self.recommendations.clear()
        
        if self.auto_watchlist_enabled:
            self._update_smart_watchlist()
        else:
            try:
                movers = self.data_provider.get_market_movers(50)
                top_movers = [m['symbol'] for m in movers[:20]]
                self.watchlist = list(set(self._get_default_watchlist() + top_movers))
                self.logger.info(f"Updated watchlist with {len(self.watchlist)} symbols")
            except Exception as e:
                self.logger.error(f"Error updating watchlist: {e}")
    
    def _daily_shutdown(self):
        self.logger.info("Daily shutdown routine...")
        # Only liquidate DAY positions
        self.portfolio_manager.liquidate_day_positions("End of trading day")
        self.portfolio_manager.save_portfolio_snapshot()
        
        summary = self.portfolio_manager.get_portfolio_summary()
        metrics = self.portfolio_manager.get_performance_metrics()
        
        self.logger.info(f"Daily P&L: ${summary.get('daily_pnl', 0):.2f}")
        self.logger.info(f"Total Portfolio Value: ${summary.get('total_portfolio_value', 0):.2f}")
        
        if metrics:
            self.logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
    
    def _hourly_portfolio_snapshot(self):
        if self.is_market_hours:
            self.portfolio_manager.save_portfolio_snapshot()
    
    # ── Status & analysis ───────────────────────────────────────────

    def get_status(self) -> Dict:
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        performance_metrics = self.portfolio_manager.get_performance_metrics()
        
        recent_recommendations = {}
        for symbol, horizons in self.recommendations.items():
            for horizon, analysis in horizons.items():
                rec = analysis.get('recommendation', {})
                action = rec.get('action', 'HOLD')
                if action in ('BUY', 'SELL'):
                    recent_recommendations[f"{symbol}_{horizon}"] = {
                        'symbol': symbol,
                        'horizon': horizon,
                        'action': action,
                        'confidence': rec.get('confidence'),
                        'price': analysis.get('current_price')
                    }
        
        return {
            'agent_status': {
                'is_running': self.is_running,
                'is_market_hours': self.is_market_hours,
                'last_scan': self.last_scan_time.isoformat(),
                'watchlist_size': len(self.watchlist)
            },
            'portfolio': portfolio_summary,
            'performance': performance_metrics,
            'recent_recommendations': recent_recommendations,
            'market_status': {
                'is_market_open': self.data_provider.is_market_open(),
                'is_extended_hours': self.data_provider.is_extended_hours(),
                'current_time_est': self.data_provider.get_eastern_time_string()
            }
        }
    
    def analyze_symbol(self, symbol: str, horizon: str = 'DAY') -> Dict:
        try:
            return self.trading_strategy.analyze_stock(symbol.upper(), horizon=horizon)
        except Exception as e:
            self.logger.error(f"Error analysing {symbol}: {e}")
            return {}
    
    def analyze_symbol_all_horizons(self, symbol: str) -> Dict:
        """Return analysis for all horizons."""
        try:
            return self.trading_strategy.analyze_stock_all_horizons(symbol.upper())
        except Exception as e:
            self.logger.error(f"Error analysing {symbol} (all horizons): {e}")
            return {}
    
    # ── Watchlist management ────────────────────────────────────────

    def add_to_watchlist(self, symbol: str):
        if symbol not in self.watchlist:
            self.watchlist.append(symbol.upper())
            self.logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbol: str):
        if symbol in self.watchlist:
            self.watchlist.remove(symbol.upper())
            self.logger.info(f"Removed {symbol} from watchlist")
    
    def force_liquidate(self):
        self.portfolio_manager.liquidate_all_positions("Manual liquidation")
        self.notifications.notify_risk_alert("Manual liquidation executed")
        self.logger.info("Manual liquidation completed")
    
    def _update_smart_watchlist(self):
        try:
            self.logger.info("Updating smart watchlist...")
            smart_watchlist = self.stock_screener.get_smart_watchlist(size=60)
            if smart_watchlist:
                self.watchlist = smart_watchlist
                self.last_watchlist_update = datetime.now()
                self.logger.info(f"Smart watchlist updated with {len(self.watchlist)} stocks")
            else:
                self.logger.warning("Smart watchlist failed, keeping current")
        except Exception as e:
            self.logger.error(f"Error updating smart watchlist: {e}")
            self.watchlist = self._get_default_watchlist()
    
    def _check_watchlist_update(self):
        now = datetime.now()
        hours_since_update = (now - self.last_watchlist_update).total_seconds() / 3600
        if hours_since_update >= 2:
            self.logger.info("Periodic watchlist update due")
            try:
                momentum_stocks = self.stock_screener.screen_stocks(max_stocks=15, screen_type='momentum')
                breakout_stocks = self.stock_screener.screen_stocks(max_stocks=10, screen_type='breakout')
                new_stocks = [s for s in momentum_stocks + breakout_stocks if s not in self.watchlist]
                if new_stocks:
                    self.watchlist.extend(new_stocks[:15])
                    self.watchlist = self.watchlist[:80]
                    self.logger.info(f"Added {len(new_stocks[:15])} new stocks to watchlist")
                self.last_watchlist_update = now
            except Exception as e:
                self.logger.error(f"Error in periodic watchlist update: {e}")
    
    def enable_auto_watchlist(self, enabled: bool = True):
        self.auto_watchlist_enabled = enabled
        if enabled:
            self.logger.info("Automatic watchlist generation enabled")
            self._update_smart_watchlist()
        else:
            self.logger.info("Automatic watchlist generation disabled")
    
    def get_screening_options(self) -> List[str]:
        return ['day_trading', 'breakout', 'momentum', 'high_volume', 'high_volatility']
    
    def screen_stocks_manual(self, screen_type: str = 'day_trading', max_stocks: int = 50) -> List[str]:
        try:
            return self.stock_screener.screen_stocks(max_stocks=max_stocks, screen_type=screen_type)
        except Exception as e:
            self.logger.error(f"Error in manual stock screening: {e}")
            return []


def main():
    agent = TradingAgent()
    try:
        agent.start()
    except KeyboardInterrupt:
        print("\nShutting down trading agent...")
        agent.stop()

if __name__ == "__main__":
    main()
