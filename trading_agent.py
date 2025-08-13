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

class TradingAgent:
    """
    Main trading agent that orchestrates market analysis, strategy execution, and risk management
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_provider = MarketDataProvider()
        self.trading_strategy = TradingStrategy(self.data_provider)
        self.portfolio_manager = PortfolioManager(self.data_provider)
        self.stock_screener = StockScreener(self.data_provider)
        
        # Agent state
        self.is_running = False
        self.is_market_hours = False
        self.last_scan_time = datetime.min
        self.scan_interval = 120  # seconds between scans
        # Add after-hours scan interval and last scan time
        self.last_after_hours_scan_time = datetime.min
        self.after_hours_scan_interval = 120  # seconds between after-hours scans
        
        # Watchlist for monitoring
        self.watchlist = self._get_default_watchlist()
        self.auto_watchlist_enabled = True  # Enable automatic stock picking
        self.last_watchlist_update = datetime.min
        self.recommendations = {}
        
        self.logger.info("Trading Agent initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
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
        """Get default watchlist of stocks to monitor"""
        return [
            # High-volume, liquid stocks good for day trading
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Large cap
            'NVDA', 'AMD', 'NFLX', 'META', 'BABA',  # Growth stocks
            'JPM', 'BAC', 'XOM', 'CVX',  # Value stocks
            'DIS', 'V', 'UNH', 'JNJ', 'PG'  # Blue chips
        ]
    
    def start(self):
        """Start the trading agent"""
        self.is_running = True
        self.logger.info("Starting Trading Agent...")
        
        # Schedule daily tasks
        schedule.every().day.at("06:00").do(self._daily_startup)
        schedule.every().day.at("16:00").do(self._daily_shutdown)
        schedule.every().hour.do(self._hourly_portfolio_snapshot)
        
        # Main trading loop
        try:
            while self.is_running:
                self._check_market_status()
                
                if self.is_market_hours:
                    self._trading_loop()
                else:
                    self._after_hours_monitoring()
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep before next iteration
                time.sleep(10)
                
        except KeyboardInterrupt:
            self.logger.info("Trading Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Trading Agent error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading agent"""
        self.is_running = False
        self.logger.info("Stopping Trading Agent...")
        
        # Liquidate all positions if market is open
        if self.data_provider.is_market_open():
            self.portfolio_manager.liquidate_all_positions("Agent shutdown")
        
        # Save final portfolio snapshot
        self.portfolio_manager.save_portfolio_snapshot()
        
        self.logger.info("Trading Agent stopped")
    
    def _check_market_status(self):
        """Check if market is currently open"""
        self.is_market_hours = (
            self.data_provider.is_market_open() or 
            self.data_provider.is_extended_hours()
        )
    
    def _trading_loop(self):
        """Main trading loop for market hours"""
        now = datetime.now()
        
        # Check if it's time for a new scan (use total_seconds)
        if (now - self.last_scan_time).total_seconds() < self.scan_interval:
            return
        
        self.last_scan_time = now
        
        try:
            # Update portfolio positions
            self.portfolio_manager.update_positions()
            
            # Check stop loss and take profit triggers
            self.portfolio_manager.check_stop_loss_take_profit()
            
            # Check if we should liquidate due to risk or end of day
            if self.portfolio_manager.should_liquidate_for_risk():
                self.portfolio_manager.liquidate_all_positions("Risk management")
                return
            
            if self.portfolio_manager.is_end_of_day():
                self.portfolio_manager.liquidate_all_positions("End of day")
                return
            
            # Update smart watchlist periodically
            if self.auto_watchlist_enabled:
                self._check_watchlist_update()
            
            # Scan watchlist for new opportunities
            self._scan_watchlist()
            
            # Execute trades based on recommendations
            self._execute_trades()
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
    
    def _scan_watchlist(self):
        """Scan watchlist for trading opportunities"""
        self.logger.info("Scanning watchlist for opportunities...")
        
        # Use threading to analyze multiple stocks concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for symbol in self.watchlist:
                future = executor.submit(self.trading_strategy.analyze_stock, symbol)
                futures[future] = symbol
            
            # Collect results
            for future in futures:
                symbol = futures[future]
                try:
                    analysis = future.result(timeout=10)  # 10 second timeout
                    if analysis:
                        self.recommendations[symbol] = analysis
                        
                        recommendation = analysis.get('recommendation', {})
                        action = recommendation.get('action', 'HOLD')
                        confidence = recommendation.get('confidence', 0)
                        
                        if action in ['BUY', 'SELL'] and confidence > 60:
                            self.logger.info(
                                f"{symbol}: {action} signal with {confidence:.1f}% confidence"
                            )
                            
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
    
    def _execute_trades(self):
        """Execute trades based on recommendations"""
        for symbol, analysis in self.recommendations.items():
            try:
                recommendation = analysis.get('recommendation', {})
                action = recommendation.get('action', 'HOLD')
                confidence = recommendation.get('confidence', 0)
                position_size = recommendation.get('position_size', 0)
                stop_loss = recommendation.get('stop_loss', 0)
                take_profit = recommendation.get('take_profit', 0)
                
                # Skip if already have position in this symbol
                if symbol in self.portfolio_manager.positions:
                    continue
                
                # Only execute high-confidence signals
                if confidence < 70:
                    continue
                
                current_price = analysis.get('current_price', 0)
                if not current_price:
                    continue
                
                # Calculate quantity based on position size
                quantity = int(position_size / current_price)
                
                if action == 'BUY' and quantity > 0:
                    success = self.portfolio_manager.open_position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=current_price,
                        position_type='LONG',
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if success:
                        self.logger.info(
                            f"Executed BUY: {symbol} x{quantity} @ ${current_price:.2f}"
                        )
                
                elif action == 'SELL' and quantity > 0:
                    # For simplicity, we'll implement short selling later
                    # For now, only close existing long positions
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def _after_hours_monitoring(self):
        """Monitor extended hours for gap opportunities"""
        if not self.data_provider.is_extended_hours():
            return
        
        now = datetime.now()
        # Throttle after-hours monitoring
        if (now - self.last_after_hours_scan_time).total_seconds() < self.after_hours_scan_interval:
            return
        self.last_after_hours_scan_time = now
        
        # Get extended hours movers
        try:
            movers = self.data_provider.get_market_movers(20)
            
            for mover in movers[:5]:  # Top 5 movers
                symbol = mover.get('symbol')
                change_percent = mover.get('change_percent', 0)
                
                if abs(change_percent) > 3:  # Significant move
                    extended_data = self.data_provider.get_extended_hours_data(symbol)
                    
                    self.logger.info(
                        f"Extended hours activity: {symbol} {change_percent:+.1f}%"
                    )
                    
                    # Add to watchlist for next trading day if not already there
                    if symbol not in self.watchlist:
                        self.watchlist.append(symbol)
                        
        except Exception as e:
            self.logger.error(f"Error in after hours monitoring: {e}")
    
    def _daily_startup(self):
        """Daily startup routine"""
        self.logger.info("Daily startup routine...")
        
        # Reset daily metrics
        self.portfolio_manager.reset_daily_metrics()
        
        # Clear old recommendations
        self.recommendations.clear()
        
        # Update watchlist automatically if enabled
        if self.auto_watchlist_enabled:
            self._update_smart_watchlist()
        else:
            # Update watchlist with current market movers
            try:
                movers = self.data_provider.get_market_movers(50)
                top_movers = [m['symbol'] for m in movers[:20]]
                
                # Merge with default watchlist
                self.watchlist = list(set(self._get_default_watchlist() + top_movers))
                
                self.logger.info(f"Updated watchlist with {len(self.watchlist)} symbols")
                
            except Exception as e:
                self.logger.error(f"Error updating watchlist: {e}")
    
    def _daily_shutdown(self):
        """Daily shutdown routine"""
        self.logger.info("Daily shutdown routine...")
        
        # Liquidate all positions
        self.portfolio_manager.liquidate_all_positions("End of trading day")
        
        # Save portfolio snapshot
        self.portfolio_manager.save_portfolio_snapshot()
        
        # Log daily performance
        summary = self.portfolio_manager.get_portfolio_summary()
        metrics = self.portfolio_manager.get_performance_metrics()
        
        self.logger.info(f"Daily P&L: ${summary.get('daily_pnl', 0):.2f}")
        self.logger.info(f"Total Portfolio Value: ${summary.get('total_portfolio_value', 0):.2f}")
        self.logger.info(f"Daily Trades: {summary.get('daily_trades', 0)}")
        
        if metrics:
            self.logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            self.logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    def _hourly_portfolio_snapshot(self):
        """Save hourly portfolio snapshot"""
        if self.is_market_hours:
            self.portfolio_manager.save_portfolio_snapshot()
    
    def get_status(self) -> Dict:
        """Get current status of the trading agent"""
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        performance_metrics = self.portfolio_manager.get_performance_metrics()
        
        # Get recent recommendations
        recent_recommendations = {}
        for symbol, analysis in self.recommendations.items():
            recommendation = analysis.get('recommendation', {})
            action = recommendation.get('action', 'HOLD')
            if action in ['BUY', 'SELL']:
                recent_recommendations[symbol] = {
                    'action': action,
                    'confidence': recommendation.get('confidence'),
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
    
    def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol.upper())
            self.logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol.upper())
            self.logger.info(f"Removed {symbol} from watchlist")
    
    def force_liquidate(self):
        """Force liquidation of all positions"""
        self.portfolio_manager.liquidate_all_positions("Manual liquidation")
        self.logger.info("Manual liquidation completed")
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Get detailed analysis for a specific symbol"""
        try:
            return self.trading_strategy.analyze_stock(symbol.upper())
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {}
    
    def _update_smart_watchlist(self):
        """Update watchlist using intelligent stock screening"""
        try:
            self.logger.info("Updating smart watchlist...")
            
            # Get smart watchlist from screener with increased size
            smart_watchlist = self.stock_screener.get_smart_watchlist(size=60)
            
            if smart_watchlist:
                self.watchlist = smart_watchlist
                self.last_watchlist_update = datetime.now()
                self.logger.info(f"Smart watchlist updated with {len(self.watchlist)} stocks: {self.watchlist[:10]}...")
            else:
                self.logger.warning("Smart watchlist generation failed, keeping current watchlist")
                
        except Exception as e:
            self.logger.error(f"Error updating smart watchlist: {e}")
            # Fallback to default watchlist
            self.watchlist = self._get_default_watchlist()
    
    def _check_watchlist_update(self):
        """Check if watchlist needs updating during trading"""
        now = datetime.now()
        
        # Update watchlist every 2 hours during market hours
        hours_since_update = (now - self.last_watchlist_update).total_seconds() / 3600
        
        if hours_since_update >= 2:
            self.logger.info("Periodic watchlist update due")
            
            # Get current momentum stocks for quick updates
            try:
                momentum_stocks = self.stock_screener.screen_stocks(max_stocks=15, screen_type='momentum')
                breakout_stocks = self.stock_screener.screen_stocks(max_stocks=10, screen_type='breakout')
                
                # Add new opportunities to existing watchlist
                new_stocks = []
                for stock in momentum_stocks + breakout_stocks:
                    if stock not in self.watchlist:
                        new_stocks.append(stock)
                
                if new_stocks:
                    # Add new stocks, but limit total watchlist size
                    self.watchlist.extend(new_stocks[:15])
                    self.watchlist = self.watchlist[:80]  # Keep max 80 stocks
                    
                    self.logger.info(f"Added {len(new_stocks[:10])} new stocks to watchlist: {new_stocks[:10]}")
                
                self.last_watchlist_update = now
                
            except Exception as e:
                self.logger.error(f"Error in periodic watchlist update: {e}")
    
    def enable_auto_watchlist(self, enabled: bool = True):
        """Enable or disable automatic watchlist generation"""
        self.auto_watchlist_enabled = enabled
        if enabled:
            self.logger.info("Automatic watchlist generation enabled")
            self._update_smart_watchlist()
        else:
            self.logger.info("Automatic watchlist generation disabled")
    
    def get_screening_options(self) -> List[str]:
        """Get available screening options"""
        return ['day_trading', 'breakout', 'momentum', 'high_volume', 'high_volatility']
    
    def screen_stocks_manual(self, screen_type: str = 'day_trading', max_stocks: int = 50) -> List[str]:
        """Manually screen stocks with specific criteria"""
        try:
            return self.stock_screener.screen_stocks(max_stocks=max_stocks, screen_type=screen_type)
        except Exception as e:
            self.logger.error(f"Error in manual stock screening: {e}")
            return []

def main():
    """Main entry point"""
    agent = TradingAgent()
    
    try:
        agent.start()
    except KeyboardInterrupt:
        print("\nShutting down trading agent...")
        agent.stop()

if __name__ == "__main__":
    main()