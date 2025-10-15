import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import logging
import json
import sqlite3
from pathlib import Path
from config import Config
from data_provider import MarketDataProvider
from broker_integration import AlpacaBroker

class Position:
    """Represents a trading position"""
    
    def __init__(self, symbol: str, quantity: int, entry_price: float, 
                 entry_time: datetime, position_type: str = 'LONG',
                 stop_loss: float = 0, take_profit: float = 0):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_type = position_type  # LONG or SHORT
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.current_price = entry_price
        self.unrealized_pnl = 0
        self.is_open = True
        
    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price
        
        if self.position_type == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def get_percentage_gain_loss(self) -> float:
        """Calculate percentage gain or loss for the position"""
        if self.entry_price == 0:
            return 0.0
        
        if self.position_type == 'LONG':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    def get_market_value(self) -> float:
        """Get current market value of position"""
        return self.current_price * abs(self.quantity)
    
    def should_stop_loss(self) -> bool:
        """Check if position should be closed due to stop loss"""
        if not self.stop_loss:
            return False
            
        if self.position_type == 'LONG':
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Check if position should be closed due to take profit"""
        if not self.take_profit:
            return False
            
        if self.position_type == 'LONG':
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time.isoformat(),
            'position_type': self.position_type,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'percentage_gain_loss': self.get_percentage_gain_loss(),
            'market_value': self.get_market_value(),
            'is_open': self.is_open
        }

class PortfolioManager:
    """
    Manages trading portfolio with risk controls and automatic liquidation
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state - Initialize first
        self.positions: Dict[str, Position] = {}
        self.cash_balance = self.config.MAX_PORTFOLIO_VALUE
        self.daily_pnl = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.cash_balance
        
        # Risk tracking
        self.daily_trades = 0
        self.losing_streak = 0
        self.winning_streak = 0
        
        # Database for persistence
        self.db_path = Path("trading_data.db")
        self._init_database()
        
        # Initialize broker for actual trading
        self.broker = AlpacaBroker()
        if self.broker.is_connected():
            self.logger.info("Connected to Alpaca broker for live trading")
            # Sync with broker account
            self._sync_with_broker()
        else:
            self.logger.warning("Broker not connected - running in simulation mode")
        
        # Load existing positions if any
        self._load_positions()
    
    def _sync_with_broker(self):
        """Sync portfolio state with broker account"""
        try:
            if not self.broker.is_connected():
                return
            
            # Get account info
            account_info = self.broker.get_account_info()
            if account_info:
                self.cash_balance = account_info.get('cash', self.cash_balance)
                self.logger.info(f"Synced with broker - Cash: ${self.cash_balance:.2f}")
            
            # Get broker positions and sync
            broker_positions = self.broker.get_positions()
            for broker_pos in broker_positions:
                symbol = broker_pos['symbol']
                
                # Create position object from broker data
                position = Position(
                    symbol=symbol,
                    quantity=broker_pos['quantity'],
                    entry_price=broker_pos['avg_entry_price'],
                    entry_time=datetime.now(),  # Approximate, since we don't have exact time
                    position_type='LONG' if broker_pos['quantity'] > 0 else 'SHORT',
                    stop_loss=0,  # Will be set by strategy
                    take_profit=0  # Will be set by strategy
                )
                
                position.current_price = broker_pos['current_price']
                position.unrealized_pnl = broker_pos['unrealized_pl']
                
                self.positions[symbol] = position
                self.logger.info(f"Synced position: {symbol} - {broker_pos['quantity']} shares")
                
        except Exception as e:
            self.logger.error(f"Error syncing with broker: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
    def _init_database(self):
        """Initialize SQLite database for trade tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    position_type TEXT NOT NULL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    stop_loss REAL,
                    take_profit REAL,
                    reason TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash_balance REAL NOT NULL,
                    total_value REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    num_positions INTEGER NOT NULL
                )
            """)
    
    def _save_position(self, position: Position):
        """Save position to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades 
                (symbol, quantity, entry_price, entry_time, position_type, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol, position.quantity, position.entry_price,
                position.entry_time.isoformat(), position.position_type,
                position.stop_loss, position.take_profit
            ))
    
    def _update_position_exit(self, position: Position, exit_price: float, reason: str):
        """Update position with exit information"""
        pnl = position.unrealized_pnl
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED', reason = ?
                WHERE symbol = ? AND status = 'OPEN'
            """, (
                exit_price, datetime.now().isoformat(), pnl, reason, position.symbol
            ))
    
    def _load_positions(self):
        """Load open positions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, quantity, entry_price, entry_time, position_type, 
                           stop_loss, take_profit
                    FROM trades 
                    WHERE status = 'OPEN'
                """)
                
                for row in cursor.fetchall():
                    symbol, quantity, entry_price, entry_time, position_type, stop_loss, take_profit = row
                    
                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        entry_time=datetime.fromisoformat(entry_time),
                        position_type=position_type,
                        stop_loss=stop_loss or 0,
                        take_profit=take_profit or 0
                    )
                    
                    self.positions[symbol] = position
                    self.logger.info(f"Loaded position: {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    def can_open_position(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        
        # Check if already have position in this symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        # Check maximum number of positions
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False, f"Maximum positions ({self.config.MAX_POSITIONS}) reached"
        
        # Check position size limits
        if position_size > self.config.MAX_POSITION_SIZE:
            return False, f"Position size exceeds maximum ({self.config.MAX_POSITION_SIZE})"
        
        # Check available cash
        if position_size > self.cash_balance:
            return False, f"Insufficient cash (have: {self.cash_balance:.2f}, need: {position_size:.2f})"
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
        
        # Check if market is open or extended hours
        if not (self.data_provider.is_market_open() or self.data_provider.is_extended_hours()):
            return False, "Market is closed"
        
        return True, "OK"
    
    def open_position(self, symbol: str, quantity: int, entry_price: float,
                     position_type: str = 'LONG', stop_loss: float = 0,
                     take_profit: float = 0) -> bool:
        """Open a new position"""
        
        position_size = abs(quantity) * entry_price
        can_open, reason = self.can_open_position(symbol, position_size)
        
        if not can_open:
            self.logger.warning(f"Cannot open position in {symbol}: {reason}")
            return False
        
        try:
            # Execute trade through broker if connected
            if self.broker.is_connected():
                side = 'buy' if position_type == 'LONG' else 'sell'
                
                # Place bracket order with stop loss and take profit
                if stop_loss and take_profit:
                    order_result = self.broker.place_bracket_order(
                        symbol=symbol,
                        quantity=abs(quantity),
                        side=side,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                else:
                    # Place simple market order
                    order_result = self.broker.place_order(
                        symbol=symbol,
                        quantity=abs(quantity),
                        side=side,
                        order_type='market'
                    )
                
                if not order_result.get('success', False):
                    self.logger.error(f"Broker order failed: {order_result.get('error', 'Unknown error')}")
                    return False
                
                # Use filled price if available, otherwise use entry_price
                actual_entry_price = order_result.get('filled_avg_price', entry_price)
                actual_quantity = order_result.get('filled_qty', quantity)
                
                if actual_quantity == 0:
                    self.logger.warning(f"Order not filled for {symbol}")
                    return False
                
                self.logger.info(f"Broker order executed: {order_result.get('order_id')}")
                
            else:
                # Simulation mode
                actual_entry_price = entry_price
                actual_quantity = quantity
                self.logger.info(f"Simulation mode - no actual trade executed")
            
            # Create position object
            position = Position(
                symbol=symbol,
                quantity=actual_quantity,
                entry_price=actual_entry_price,
                entry_time=datetime.now(),
                position_type=position_type,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions[symbol] = position
            
            # Update cash balance (only in simulation mode or if broker not connected)
            if not self.broker.is_connected():
                self.cash_balance -= abs(actual_quantity) * actual_entry_price
            else:
                # Sync with broker to get updated cash balance
                self._sync_with_broker()
            
            self.daily_trades += 1
            
            # Save to database
            self._save_position(position)
            
            self.logger.info(f"Opened {position_type} position: {symbol} x{actual_quantity} @ ${actual_entry_price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening position in {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> bool:
        """Close an existing position"""
        
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return False
        
        try:
            position = self.positions[symbol]
            
            # Execute trade through broker if connected
            if self.broker.is_connected():
                close_result = self.broker.close_position(symbol)
                
                if not close_result.get('success', False):
                    self.logger.error(f"Broker close failed: {close_result.get('error', 'Unknown error')}")
                    return False
                
                self.logger.info(f"Broker position closed: {close_result.get('order_id')}")
                
                # Sync with broker to get updated positions and cash
                self._sync_with_broker()
                
                # Get actual exit price from broker if available
                # For now, use the provided exit_price
                actual_exit_price = exit_price
                
            else:
                # Simulation mode
                actual_exit_price = exit_price
                self.logger.info(f"Simulation mode - no actual trade executed")
            
            position.update_price(actual_exit_price)
            
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            position_value = position.get_market_value()
            
            # Update portfolio (only in simulation mode)
            if not self.broker.is_connected():
                self.cash_balance += position_value
            
            self.daily_pnl += realized_pnl
            self.total_pnl += realized_pnl
            
            # Update streaks
            if realized_pnl > 0:
                self.winning_streak += 1
                self.losing_streak = 0
            else:
                self.losing_streak += 1
                self.winning_streak = 0
            
            # Update database
            self._update_position_exit(position, actual_exit_price, reason)
            
            # Remove from active positions
            del self.positions[symbol]
            
            self.logger.info(f"Closed position: {symbol} @ ${actual_exit_price:.2f}, P&L: ${realized_pnl:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position in {symbol}: {e}")
            return False
    
    def update_positions(self):
        """Update all positions with current market prices"""
        for symbol, position in self.positions.items():
            try:
                quote = self.data_provider.get_real_time_quote(symbol)
                if quote and quote.get('current_price'):
                    position.update_price(quote['current_price'])
                    
            except Exception as e:
                self.logger.error(f"Error updating position {symbol}: {e}")
    
    def check_stop_loss_take_profit(self):
        """Check all positions for stop loss and take profit triggers"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if position.should_stop_loss():
                positions_to_close.append((symbol, position.current_price, "Stop Loss"))
            elif position.should_take_profit():
                positions_to_close.append((symbol, position.current_price, "Take Profit"))
        
        # Close triggered positions
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason)
    
    def liquidate_all_positions(self, reason: str = "End of day liquidation"):
        """Close all open positions"""
        
        # Use broker's close all positions if connected
        if self.broker.is_connected():
            try:
                result = self.broker.close_all_positions()
                if result.get('success', False):
                    self.logger.info(f"Broker liquidated all positions: {reason}")
                    
                    # Update P&L for all positions before clearing
                    for symbol, position in self.positions.items():
                        realized_pnl = position.unrealized_pnl
                        self.daily_pnl += realized_pnl
                        self.total_pnl += realized_pnl
                        
                        # Update database
                        self._update_position_exit(position, position.current_price, reason)
                    
                    # Clear positions and sync with broker
                    self.positions.clear()
                    self._sync_with_broker()
                    
                    self.logger.info(f"All positions liquidated via broker: {reason}")
                    return
                else:
                    self.logger.warning(f"Broker liquidation failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"Error with broker liquidation: {e}")
        
        # Fallback to individual position closing
        positions_to_close = list(self.positions.keys())
        
        for symbol in positions_to_close:
            position = self.positions.get(symbol)
            if position:
                self.close_position(symbol, position.current_price, reason)
        
        self.logger.info(f"Liquidated all positions: {reason}")
    
    def should_liquidate_for_risk(self) -> bool:
        """Check if we should liquidate due to risk limits"""
        
        # Daily loss limit
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            return True
        
        # Losing streak (optional additional risk control)
        if self.losing_streak >= 5:  # 5 consecutive losses
            return True
        
        # Drawdown limit
        current_value = self.get_total_portfolio_value()
        if current_value < self.peak_portfolio_value * 0.9:  # 10% drawdown
            return True
        
        return False
    
    def is_end_of_day(self) -> bool:
        """Check if it's time for end-of-day liquidation"""
        now = datetime.now()
        
        # Market close time (4:00 PM ET)
        market_close = time(16, 0)
        
        # Liquidate 15 minutes before market close
        liquidation_time = time(15, 45)
        
        return now.time() >= liquidation_time
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        total_position_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash_balance + total_position_value
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        self.update_positions()
        
        total_value = self.get_total_portfolio_value()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update peak value for drawdown calculation
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        
        drawdown = (self.peak_portfolio_value - total_value) / self.peak_portfolio_value * 100
        
        position_summaries = []
        for symbol, position in self.positions.items():
            position_summaries.append(position.to_dict())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cash_balance': self.cash_balance,
            'total_portfolio_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'num_positions': len(self.positions),
            'daily_trades': self.daily_trades,
            'winning_streak': self.winning_streak,
            'losing_streak': self.losing_streak,
            'drawdown_percent': drawdown,
            'positions': position_summaries,
            'risk_metrics': {
                'max_daily_loss_limit': self.config.MAX_DAILY_LOSS,
                'max_positions_limit': self.config.MAX_POSITIONS,
                'current_risk_level': 'HIGH' if self.should_liquidate_for_risk() else 'NORMAL'
            }
        }
    
    def save_portfolio_snapshot(self):
        """Save current portfolio state to database"""
        summary = self.get_portfolio_summary()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_history 
                (timestamp, cash_balance, total_value, daily_pnl, total_pnl, num_positions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary['timestamp'],
                summary['cash_balance'],
                summary['total_portfolio_value'],
                summary['daily_pnl'],
                summary['total_pnl'],
                summary['num_positions']
            ))
    
    def get_trading_history(self, days: int = 30) -> pd.DataFrame:
        """Get trading history for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM trades 
                    WHERE entry_time >= date('now', '-{} days')
                    ORDER BY entry_time DESC
                """.format(days)
                
                return pd.read_sql_query(query, conn)
                
        except Exception as e:
            self.logger.error(f"Error getting trading history: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            df = self.get_trading_history()
            
            if df.empty:
                return {}
            
            closed_trades = df[df['status'] == 'CLOSED']
            
            if closed_trades.empty:
                return {}
            
            # Basic metrics
            total_trades = len(closed_trades)
            winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
            losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = closed_trades['pnl'].sum()
            avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
            
            # Sharpe ratio (simplified)
            returns = closed_trades['pnl'] / self.config.MAX_POSITION_SIZE
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'current_streak': self.winning_streak if self.winning_streak > 0 else -self.losing_streak
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics (call at start of each trading day)"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.logger.info("Reset daily metrics for new trading day")