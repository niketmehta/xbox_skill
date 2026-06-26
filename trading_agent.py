"""
Trading agent orchestrator for swing (WEEK) and position (MONTH) trading.
No day-trading logic — positions are held overnight.
"""

import logging
import time
import schedule
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

from config import Config
from market_calendar import MarketCalendar
from data_provider import MarketDataProvider
from trading_strategy import TradingStrategy
from portfolio_manager import PortfolioManager
from stock_screener import StockScreener
from notifications import NotificationService
from recommendation_engine import TopRecommendationEngine, format_top_recommendations_message
from trade_simulator import TradeSimulationEngine


class TradingAgent:
    """
    Main trading agent that orchestrates multi-timeframe market analysis
    (WEEK and MONTH horizons), strategy execution, risk management,
    and WhatsApp notifications.
    """

    def __init__(self):
        self.config = Config()
        self.logger = self._setup_logging()

        # Components
        self.market_calendar = MarketCalendar()
        self.data_provider = MarketDataProvider()
        self.trading_strategy = TradingStrategy(self.data_provider)
        self.notifications = NotificationService()
        self.portfolio_manager = PortfolioManager(self.data_provider, notifications=self.notifications)
        self.stock_screener = StockScreener(self.data_provider)
        self.recommendation_engine = TopRecommendationEngine(
            self.trading_strategy,
            portfolio_manager=self.portfolio_manager
        )
        self.trade_simulator = TradeSimulationEngine(
            data_provider=self.data_provider,
            notifications=self.notifications,
        )

        # Agent state
        self.is_running = False
        self.is_market_hours = False
        self.last_scan_time = datetime.min
        self.scan_interval = 300  # 5 minutes between scans (swing trading is less urgent)

        # Watchlist
        self.manual_watchlist = []
        self.watchlist = self._get_default_watchlist()
        self.auto_watchlist_enabled = True
        self.last_watchlist_update = datetime.min
        self.recommendations: Dict[str, Dict] = {}  # symbol -> {WEEK, MONTH}

        self.logger.info("Trading Agent initialised (swing + position mode)")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(
                    'trading_agent.log',
                    maxBytes=1_000_000,
                    backupCount=3,
                    encoding='utf-8',
                ),
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
            'DIS', 'V', 'UNH', 'JNJ', 'PG',
            'LLY', 'AVGO', 'CRM', 'ORCL', 'COST'
        ]

    def _unique_symbols(self, symbols: List[str]) -> List[str]:
        seen = set()
        unique = []
        for raw in symbols or []:
            symbol = str(raw or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            unique.append(symbol)
        return unique

    def _set_watchlist(self, symbols: List[str], limit: int = 80):
        """Keep manual symbols at the front so they are always analysed first."""
        ordered = self._unique_symbols(self.manual_watchlist + (symbols or []))
        if limit and len(ordered) > limit:
            manual = self._unique_symbols(self.manual_watchlist)
            manual_set = set(manual)
            rest = [symbol for symbol in ordered if symbol not in manual_set]
            ordered = manual + rest[: max(0, limit - len(manual))]
        self.watchlist = ordered

    def build_recommendation_universe(self, universe_size: Optional[int] = None) -> List[str]:
        """Build a wider, current universe for scheduled top-pick digests."""
        manual = self._unique_symbols(self.manual_watchlist)
        requested_size = max(10, min(int(universe_size or self.config.TOP_RECOMMENDATIONS_UNIVERSE_SIZE), 120))
        size = max(requested_size, min(len(manual), 120))
        if not self.config.TOP_RECOMMENDATIONS_SMART_UNIVERSE:
            return self._unique_symbols(manual + self.watchlist)[:size]

        symbols = list(manual)

        try:
            movers = self.data_provider.get_market_movers(min(size, 80))
            symbols.extend(mover.get("symbol") for mover in movers)
        except Exception as e:
            self.logger.warning("Could not add market movers to recommendation universe: %s", e)

        try:
            symbols.extend(
                self.stock_screener.screen_stocks(
                    max_stocks=min(25, max(size // 4, 10)),
                    screen_type="momentum",
                )
            )
        except Exception as e:
            self.logger.warning("Could not add momentum stocks to recommendation universe: %s", e)

        try:
            symbols.extend(
                self.stock_screener.screen_stocks(
                    max_stocks=min(25, max(size // 4, 10)),
                    screen_type="breakout",
                )
            )
        except Exception as e:
            self.logger.warning("Could not add breakout stocks to recommendation universe: %s", e)

        try:
            symbols.extend(self.stock_screener.get_smart_watchlist(size=size))
        except Exception as e:
            self.logger.warning("Could not add smart watchlist to recommendation universe: %s", e)

        symbols.extend(self._get_default_watchlist())
        symbols.extend(self.watchlist)
        symbols.extend(getattr(self.portfolio_manager, "positions", {}).keys())

        universe = self._unique_symbols(symbols)
        if len(universe) > size:
            manual_set = set(manual)
            rest = [symbol for symbol in universe if symbol not in manual_set]
            universe = manual + rest[: max(0, size - len(manual))]
        if universe:
            self._set_watchlist(universe)
            self.last_watchlist_update = datetime.now()
            self.logger.info("Recommendation universe prepared with %s symbols", len(universe))
        return universe or self._unique_symbols(manual + self.watchlist)[:size]

    def _schedule_weekdays(self, run_time: str, job_func, *args, **kwargs):
        for weekday in (
            schedule.every().monday,
            schedule.every().tuesday,
            schedule.every().wednesday,
            schedule.every().thursday,
            schedule.every().friday,
        ):
            weekday.at(run_time).do(job_func, *args, **kwargs)

    # ── Agent lifecycle ─────────────────────────────────────────────

    def start(self):
        self.is_running = True
        self.logger.info("Starting Trading Agent (swing/position mode)...")

        self._schedule_weekdays("06:00", self._daily_startup)
        self._schedule_weekdays("16:30", self._daily_review)
        schedule.every().hour.do(self._hourly_portfolio_snapshot)
        if self.config.TOP_RECOMMENDATIONS_ENABLED:
            self._schedule_weekdays(
                self.config.TOP_RECOMMENDATIONS_TIME,
                self._send_scheduled_top_recommendations
            )
        if self.config.SIMULATION_ENABLED:
            self._schedule_weekdays(
                self.config.SIMULATION_OPEN_TIME,
                self._capture_scheduled_open_simulation
            )
            self._schedule_weekdays(
                self.config.SIMULATION_MIDDAY_TIME,
                self._send_scheduled_midday_simulation_summary
            )
            self._schedule_weekdays(
                self.config.SIMULATION_EOD_TIME,
                self._send_scheduled_simulation_summary
            )

        try:
            while self.is_running:
                self._check_market_status()
                if self.is_market_hours:
                    self._trading_loop()
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
        # No forced liquidation — swing/position trades stay open
        self.portfolio_manager.save_portfolio_snapshot()
        self.logger.info("Trading Agent stopped (positions remain open)")

    def _check_market_status(self):
        self.is_market_hours = self.data_provider.is_market_open()

    def _should_run_market_job(self, job_name: str) -> bool:
        session = self.market_calendar.get_session()
        if session.get("is_trading_day"):
            return True

        self.logger.info(
            "Skipping %s: market is closed on %s (%s via %s)",
            job_name,
            session.get("date", "today"),
            session.get("reason", "no market session"),
            session.get("source", "unknown"),
        )
        return False

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

            if self.auto_watchlist_enabled:
                self._check_watchlist_update()

            self._scan_watchlist()
            self._execute_trades()

        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")

    # ── Multi-horizon scanning (WEEK + MONTH) ───────────────────────

    def _scan_watchlist(self):
        self.logger.info("Scanning watchlist (WEEK + MONTH horizons)...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for symbol in self.watchlist:
                future = executor.submit(self._analyze_all_horizons, symbol)
                futures[future] = symbol

            for future in futures:
                symbol = futures[future]
                try:
                    all_horizons = future.result(timeout=60)
                    if all_horizons:
                        self.recommendations[symbol] = all_horizons

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
        """Analyse a symbol across WEEK and MONTH horizons."""
        results = {}
        for h in ('WEEK', 'MONTH'):
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

                    # Skip rest of horizons once a position is opened for this symbol
                    if symbol in self.portfolio_manager.positions:
                        break

                except Exception as e:
                    self.logger.error(f"Error executing trade for {symbol} [{horizon}]: {e}")

    # ── Daily routines ──────────────────────────────────────────────

    def _daily_startup(self):
        if not self._should_run_market_job("daily startup"):
            return

        self.logger.info("Daily startup routine...")
        self.portfolio_manager.reset_daily_metrics()
        self.recommendations.clear()

        if self.auto_watchlist_enabled:
            self._update_smart_watchlist()
        else:
            try:
                movers = self.data_provider.get_market_movers(50)
                top_movers = [m['symbol'] for m in movers[:20]]
                self._set_watchlist(self._get_default_watchlist() + top_movers)
                self.logger.info(f"Updated watchlist with {len(self.watchlist)} symbols")
            except Exception as e:
                self.logger.error(f"Error updating watchlist: {e}")

    def _daily_review(self):
        """End-of-day review — NO forced liquidation for swing/position trades."""
        if not self._should_run_market_job("daily review"):
            return

        self.logger.info("Daily review (no forced liquidation)...")
        self.portfolio_manager.save_portfolio_snapshot()

        summary = self.portfolio_manager.get_portfolio_summary()
        metrics = self.portfolio_manager.get_performance_metrics()

        self.logger.info(f"Daily P&L: ${summary.get('daily_pnl', 0):.2f}")
        self.logger.info(f"Total Portfolio Value: ${summary.get('total_portfolio_value', 0):.2f}")
        self.logger.info(f"Open positions: {summary.get('num_positions', 0)}")

        if metrics:
            self.logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")

    def _hourly_portfolio_snapshot(self):
        if self.is_market_hours:
            self.portfolio_manager.save_portfolio_snapshot()

    def _send_scheduled_top_recommendations(self):
        if not self._should_run_market_job("scheduled trading council digest"):
            return

        horizon = self.config.TOP_RECOMMENDATIONS_HORIZON
        self.logger.info("Sending scheduled trading council digest [%s]", horizon)
        result = self.send_top_recommendations_whatsapp(
            horizon=horizon,
            limit=5,
            universe_size=self.config.TOP_RECOMMENDATIONS_UNIVERSE_SIZE,
        )
        if not result.get('delivery', {}).get('sent'):
            self.logger.warning(
                "Scheduled trading council digest was not delivered: %s",
                result.get("delivery", {}).get("error"),
            )

    def _capture_scheduled_open_simulation(self):
        if not self._should_run_market_job("scheduled open simulation capture"):
            return

        self.logger.info("Capturing simulated open entries for top recommendations")
        result = self.trade_simulator.capture_open_trades(
            top_n=self.config.SIMULATION_TOP_N
        )
        self.logger.info(
            "Simulated open capture: %s/%s trades",
            result.get("captured", 0),
            result.get("requested", 0),
        )
        if result.get("errors"):
            self.logger.warning("Open simulation errors: %s", result.get("errors"))

    def _send_scheduled_simulation_summary(self):
        if not self._should_run_market_job("scheduled end-of-day simulation summary"):
            return

        self.logger.info("Sending simulated end-of-day P&L summary")
        result = self.trade_simulator.send_eod_summary_whatsapp(label="EOD")
        if not result.get("delivery", {}).get("sent"):
            self.logger.warning(
                "Simulated end-of-day P&L summary was not delivered: %s",
                result.get("delivery", {}).get("error"),
            )

    def _send_scheduled_midday_simulation_summary(self):
        if not self._should_run_market_job("scheduled midday simulation summary"):
            return

        self.logger.info("Sending simulated midday P&L summary")
        result = self.trade_simulator.send_eod_summary_whatsapp(label="MIDDAY")
        if not result.get("delivery", {}).get("sent"):
            self.logger.warning(
                "Simulated midday P&L summary was not delivered: %s",
                result.get("delivery", {}).get("error"),
            )

    # ── Status & analysis ───────────────────────────────────────────

    def get_status(self) -> Dict:
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        performance_metrics = self.portfolio_manager.get_performance_metrics()

        # Market regime
        regime = self.trading_strategy._get_market_regime()

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
                'watchlist_size': len(self.watchlist),
                'mode': 'Swing + Position'
            },
            'market_regime': regime,
            'portfolio': portfolio_summary,
            'performance': performance_metrics,
            'recent_recommendations': recent_recommendations,
            'market_status': {
                'is_market_open': self.data_provider.is_market_open(),
                'is_extended_hours': self.data_provider.is_extended_hours(),
                'current_time_est': self.data_provider.get_eastern_time_string()
            }
        }

    def analyze_symbol(self, symbol: str, horizon: str = 'WEEK') -> Dict:
        try:
            return self.trading_strategy.analyze_stock(symbol.upper(), horizon=horizon)
        except Exception as e:
            self.logger.error(f"Error analysing {symbol}: {e}")
            return {}

    def analyze_symbol_all_horizons(self, symbol: str) -> Dict:
        """Return analysis for WEEK and MONTH horizons."""
        try:
            return self.trading_strategy.analyze_stock_all_horizons(symbol.upper())
        except Exception as e:
            self.logger.error(f"Error analysing {symbol} (all horizons): {e}")
            return {}

    # ── Watchlist management ────────────────────────────────────────

    def add_to_watchlist(self, symbol: str):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return False
        was_new = symbol not in self.watchlist and symbol not in self.manual_watchlist
        if symbol not in self.manual_watchlist:
            self.manual_watchlist.append(symbol)
        self._set_watchlist([symbol] + self.watchlist)
        self.logger.info("Added %s to manual watchlist", symbol)
        return was_new

    def remove_from_watchlist(self, symbol: str):
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return False
        removed = symbol in self.watchlist or symbol in self.manual_watchlist
        self.manual_watchlist = [item for item in self.manual_watchlist if item != symbol]
        self.watchlist = [item for item in self.watchlist if item != symbol]
        if removed:
            self.logger.info("Removed %s from watchlist", symbol)
        return removed

    def force_liquidate(self):
        self.portfolio_manager.liquidate_all_positions("Manual liquidation")
        self.notifications.notify_risk_alert("Manual liquidation executed")
        self.logger.info("Manual liquidation completed")

    def _update_smart_watchlist(self):
        try:
            self.logger.info("Updating smart watchlist...")
            smart_watchlist = self.stock_screener.get_smart_watchlist(size=60)
            if smart_watchlist:
                self._set_watchlist(smart_watchlist)
                self.last_watchlist_update = datetime.now()
                self.logger.info(f"Smart watchlist updated with {len(self.watchlist)} stocks")
            else:
                self.logger.warning("Smart watchlist failed, keeping current")
        except Exception as e:
            self.logger.error(f"Error updating smart watchlist: {e}")
            self._set_watchlist(self._get_default_watchlist())

    def _check_watchlist_update(self):
        now = datetime.now()
        hours_since_update = (now - self.last_watchlist_update).total_seconds() / 3600
        if hours_since_update >= 4:  # Less frequent for swing trading
            self.logger.info("Periodic watchlist update due")
            try:
                momentum_stocks = self.stock_screener.screen_stocks(max_stocks=15, screen_type='momentum')
                breakout_stocks = self.stock_screener.screen_stocks(max_stocks=10, screen_type='breakout')
                new_stocks = [s for s in momentum_stocks + breakout_stocks if s not in self.watchlist]
                if new_stocks:
                    self._set_watchlist(self.watchlist + new_stocks[:15])
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
        return ['swing', 'breakout', 'momentum', 'high_volume', 'high_volatility', 'value']

    def screen_stocks_manual(self, screen_type: str = 'swing', max_stocks: int = 50) -> List[str]:
        try:
            return self.stock_screener.screen_stocks(max_stocks=max_stocks, screen_type=screen_type)
        except Exception as e:
            self.logger.error(f"Error in manual stock screening: {e}")
            return []

    def get_top_recommendations(
        self,
        horizon: str = 'WEEK',
        limit: int = 5,
        universe_size: int = 50,
    ) -> Dict:
        """Run the multi-agent council and return challenged top BUY candidates."""
        symbols = self.build_recommendation_universe(universe_size)
        return self.recommendation_engine.get_top_recommendations(
            symbols=symbols,
            horizon=horizon,
            limit=limit,
            universe_size=len(symbols),
        )

    def send_top_recommendations_whatsapp(
        self,
        horizon: str = 'WEEK',
        limit: int = 5,
        universe_size: int = 50,
        target: Optional[str] = None,
    ) -> Dict:
        """Generate top recommendations and deliver the digest via OpenClaw WhatsApp."""
        result = self.get_top_recommendations(
            horizon=horizon,
            limit=limit,
            universe_size=universe_size,
        )
        body = format_top_recommendations_message(result)
        sent = self.notifications.send_openclaw_whatsapp(body, target=target)
        result['delivery'] = {
            'channel': 'openclaw_whatsapp',
            'sent': sent,
            'target': target or self.notifications.get_openclaw_target(),
            'message': body,
            'error': self.notifications.get_last_error(),
        }
        return result


def main():
    agent = TradingAgent()
    try:
        agent.start()
    except KeyboardInterrupt:
        print("\nShutting down trading agent...")
        agent.stop()

if __name__ == "__main__":
    main()
