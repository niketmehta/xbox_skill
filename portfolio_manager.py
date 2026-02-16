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
    """Represents a trading position with a time-horizon tag."""
    
    def __init__(self, symbol: str, quantity: int, entry_price: float, 
                 entry_time: datetime, position_type: str = 'LONG',
                 stop_loss: float = 0, take_profit: float = 0,
                 horizon: str = 'DAY'):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_type = position_type  # LONG or SHORT
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.horizon = horizon.upper()  # DAY, WEEK, MONTH
        self.current_price = entry_price
        self.unrealized_pnl = 0
        self.is_open = True
        
    def update_price(self, current_price: float):
        self.current_price = current_price
        if self.position_type == 'LONG':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def get_percentage_gain_loss(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.position_type == 'LONG':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    def get_market_value(self) -> float:
        return self.current_price * abs(self.quantity)
    
    def should_stop_loss(self) -> bool:
        if not self.stop_loss:
            return False
        if self.position_type == 'LONG':
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        if not self.take_profit:
            return False
        if self.position_type == 'LONG':
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit
    
    def holding_days(self) -> int:
        """Number of calendar days the position has been held."""
        return (datetime.now() - self.entry_time).days
    
    def is_expired(self) -> bool:
        """Has the position exceeded its intended holding window?"""
        days = self.holding_days()
        if self.horizon == 'DAY':
            return False  # handled by EOD liquidation
        elif self.horizon == 'WEEK':
            return days >= 7
        elif self.horizon == 'MONTH':
            return days >= 30
        return False
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time.isoformat(),
            'position_type': self.position_type,
            'horizon': self.horizon,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'percentage_gain_loss': self.get_percentage_gain_loss(),
            'market_value': self.get_market_value(),
            'holding_days': self.holding_days(),
            'is_open': self.is_open
        }


class PortfolioManager:
    """
    Manages trading portfolio with risk controls, multi-horizon holding,
    and automatic liquidation.
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
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
        
        # Database
        self.db_path = Path("trading_data.db")
        self._init_database()
        
        # Broker
        self.broker = AlpacaBroker()
        if self.broker.is_connected():
            self.logger.info("Connected to Alpaca broker for live trading")
            self._sync_with_broker()
        else:
            self.logger.warning("Broker not connected – simulation mode")
        
        self._load_positions()
    
    # ── Broker sync ─────────────────────────────────────────────────

    def _sync_with_broker(self):
        try:
            if not self.broker.is_connected():
                return
            account_info = self.broker.get_account_info()
            if account_info:
                self.cash_balance = account_info.get('cash', self.cash_balance)
                self.logger.info(f"Synced with broker – Cash: ${self.cash_balance:.2f}")
            
            broker_positions = self.broker.get_positions()
            for bp in broker_positions:
                symbol = bp['symbol']
                position = Position(
                    symbol=symbol,
                    quantity=bp['quantity'],
                    entry_price=bp['avg_entry_price'],
                    entry_time=datetime.now(),
                    position_type='LONG' if bp['quantity'] > 0 else 'SHORT',
                    horizon='DAY'  # default; will be overridden if in DB
                )
                position.current_price = bp['current_price']
                position.unrealized_pnl = bp['unrealized_pl']
                self.positions[symbol] = position
                self.logger.info(f"Synced position: {symbol} – {bp['quantity']} shares")
        except Exception as e:
            self.logger.error(f"Error syncing with broker: {e}")
    
    # ── Database ────────────────────────────────────────────────────

    def _init_database(self):
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
                    horizon TEXT DEFAULT 'DAY',
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
            # Ensure horizon column exists (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN horizon TEXT DEFAULT 'DAY'")
            except sqlite3.OperationalError:
                pass  # column already exists
    
    def _save_position(self, position: Position):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades 
                (symbol, quantity, entry_price, entry_time, position_type,
                 horizon, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol, position.quantity, position.entry_price,
                position.entry_time.isoformat(), position.position_type,
                position.horizon, position.stop_loss, position.take_profit
            ))
    
    def _update_position_exit(self, position: Position, exit_price: float, reason: str):
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
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, quantity, entry_price, entry_time, position_type,
                           stop_loss, take_profit, horizon
                    FROM trades WHERE status = 'OPEN'
                """)
                for row in cursor.fetchall():
                    symbol, quantity, entry_price, entry_time, position_type, \
                        stop_loss, take_profit, horizon = row
                    position = Position(
                        symbol=symbol, quantity=quantity, entry_price=entry_price,
                        entry_time=datetime.fromisoformat(entry_time),
                        position_type=position_type,
                        stop_loss=stop_loss or 0, take_profit=take_profit or 0,
                        horizon=horizon or 'DAY'
                    )
                    self.positions[symbol] = position
                    self.logger.info(f"Loaded position: {symbol} ({horizon})")
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    # ── Position management ─────────────────────────────────────────

    def can_open_position(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False, f"Maximum positions ({self.config.MAX_POSITIONS}) reached"
        if position_size > self.config.MAX_POSITION_SIZE:
            return False, f"Position size exceeds maximum ({self.config.MAX_POSITION_SIZE})"
        if position_size > self.cash_balance:
            return False, f"Insufficient cash (have: {self.cash_balance:.2f}, need: {position_size:.2f})"
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"
        if not (self.data_provider.is_market_open() or self.data_provider.is_extended_hours()):
            return False, "Market is closed"
        return True, "OK"
    
    def open_position(self, symbol: str, quantity: int, entry_price: float,
                     position_type: str = 'LONG', stop_loss: float = 0,
                     take_profit: float = 0, horizon: str = 'DAY') -> bool:
        position_size = abs(quantity) * entry_price
        can_open, reason = self.can_open_position(symbol, position_size)
        if not can_open:
            self.logger.warning(f"Cannot open position in {symbol}: {reason}")
            return False
        
        try:
            if self.broker.is_connected():
                side = 'buy' if position_type == 'LONG' else 'sell'
                if stop_loss and take_profit:
                    order_result = self.broker.place_bracket_order(
                        symbol=symbol, quantity=abs(quantity), side=side,
                        stop_loss=stop_loss, take_profit=take_profit
                    )
                else:
                    order_result = self.broker.place_order(
                        symbol=symbol, quantity=abs(quantity), side=side,
                        order_type='market'
                    )
                if not order_result.get('success', False):
                    self.logger.error(f"Broker order failed: {order_result.get('error')}")
                    return False
                actual_entry_price = order_result.get('filled_avg_price', entry_price)
                actual_quantity = order_result.get('filled_qty', quantity)
                if actual_quantity == 0:
                    self.logger.warning(f"Order not filled for {symbol}")
                    return False
            else:
                actual_entry_price = entry_price
                actual_quantity = quantity
            
            position = Position(
                symbol=symbol, quantity=actual_quantity,
                entry_price=actual_entry_price, entry_time=datetime.now(),
                position_type=position_type,
                stop_loss=stop_loss, take_profit=take_profit,
                horizon=horizon
            )
            self.positions[symbol] = position
            
            if not self.broker.is_connected():
                self.cash_balance -= abs(actual_quantity) * actual_entry_price
            else:
                self._sync_with_broker()
            
            self.daily_trades += 1
            self._save_position(position)
            self.logger.info(
                f"Opened {position_type} position: {symbol} x{actual_quantity} "
                f"@ ${actual_entry_price:.2f} [{horizon}]"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error opening position in {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> bool:
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return False
        try:
            position = self.positions[symbol]
            if self.broker.is_connected():
                close_result = self.broker.close_position(symbol)
                if not close_result.get('success', False):
                    self.logger.error(f"Broker close failed: {close_result.get('error')}")
                    return False
                self._sync_with_broker()
                actual_exit_price = exit_price
            else:
                actual_exit_price = exit_price
            
            position.update_price(actual_exit_price)
            realized_pnl = position.unrealized_pnl
            position_value = position.get_market_value()
            
            if not self.broker.is_connected():
                self.cash_balance += position_value
            
            self.daily_pnl += realized_pnl
            self.total_pnl += realized_pnl
            
            if realized_pnl > 0:
                self.winning_streak += 1
                self.losing_streak = 0
            else:
                self.losing_streak += 1
                self.winning_streak = 0
            
            self._update_position_exit(position, actual_exit_price, reason)
            del self.positions[symbol]
            self.logger.info(f"Closed position: {symbol} @ ${actual_exit_price:.2f}, P&L: ${realized_pnl:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error closing position in {symbol}: {e}")
            return False
    
    def update_positions(self):
        for symbol, position in self.positions.items():
            try:
                quote = self.data_provider.get_real_time_quote(symbol)
                if quote and quote.get('current_price'):
                    position.update_price(quote['current_price'])
            except Exception as e:
                self.logger.error(f"Error updating position {symbol}: {e}")
    
    def check_stop_loss_take_profit(self):
        positions_to_close = []
        for symbol, position in self.positions.items():
            if position.should_stop_loss():
                positions_to_close.append((symbol, position.current_price, "Stop Loss"))
            elif position.should_take_profit():
                positions_to_close.append((symbol, position.current_price, "Take Profit"))
            elif position.is_expired():
                positions_to_close.append(
                    (symbol, position.current_price,
                     f"Horizon expired ({position.horizon})")
                )
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason)
    
    def liquidate_day_positions(self, reason: str = "End of day liquidation"):
        """Close only DAY-horizon positions (swing/position trades kept open)."""
        positions_to_close = [
            s for s, p in self.positions.items() if p.horizon == 'DAY'
        ]
        for symbol in positions_to_close:
            pos = self.positions.get(symbol)
            if pos:
                self.close_position(symbol, pos.current_price, reason)
        self.logger.info(f"Liquidated {len(positions_to_close)} DAY positions: {reason}")
    
    def liquidate_all_positions(self, reason: str = "End of day liquidation"):
        if self.broker.is_connected():
            try:
                result = self.broker.close_all_positions()
                if result.get('success', False):
                    for symbol, position in self.positions.items():
                        realized_pnl = position.unrealized_pnl
                        self.daily_pnl += realized_pnl
                        self.total_pnl += realized_pnl
                        self._update_position_exit(position, position.current_price, reason)
                    self.positions.clear()
                    self._sync_with_broker()
                    self.logger.info(f"All positions liquidated via broker: {reason}")
                    return
            except Exception as e:
                self.logger.error(f"Error with broker liquidation: {e}")
        
        positions_to_close = list(self.positions.keys())
        for symbol in positions_to_close:
            position = self.positions.get(symbol)
            if position:
                self.close_position(symbol, position.current_price, reason)
        self.logger.info(f"Liquidated all positions: {reason}")
    
    def should_liquidate_for_risk(self) -> bool:
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            return True
        if self.losing_streak >= 5:
            return True
        current_value = self.get_total_portfolio_value()
        if current_value < self.peak_portfolio_value * 0.9:
            return True
        return False
    
    def is_end_of_day(self) -> bool:
        now = datetime.now()
        liquidation_time = time(15, 45)
        return now.time() >= liquidation_time
    
    # ── Portfolio value & summaries ─────────────────────────────────

    def get_total_portfolio_value(self) -> float:
        total_position_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.cash_balance + total_position_value
    
    def get_portfolio_summary(self) -> Dict:
        self.update_positions()
        total_value = self.get_total_portfolio_value()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        
        drawdown = (self.peak_portfolio_value - total_value) / self.peak_portfolio_value * 100
        
        position_summaries = [pos.to_dict() for pos in self.positions.values()]
        
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
    
    # ── Returns by time window ──────────────────────────────────────

    def get_returns_by_window(self) -> Dict:
        """Calculate realised returns for 1-day, 1-week, and 1-month windows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now()
                windows = {
                    '1d': (now - timedelta(days=1)).isoformat(),
                    '1w': (now - timedelta(days=7)).isoformat(),
                    '1m': (now - timedelta(days=30)).isoformat(),
                }
                result = {}
                for label, since in windows.items():
                    row = conn.execute("""
                        SELECT COALESCE(SUM(pnl), 0) as total_pnl,
                               COUNT(*) as trade_count,
                               COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as wins
                        FROM trades
                        WHERE status = 'CLOSED' AND exit_time >= ?
                    """, (since,)).fetchone()
                    
                    total_pnl, trade_count, wins = row
                    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
                    
                    # Return % relative to portfolio value
                    pv = self.config.MAX_PORTFOLIO_VALUE or 1
                    return_pct = (total_pnl / pv) * 100
                    
                    result[label] = {
                        'pnl': total_pnl,
                        'return_pct': return_pct,
                        'trade_count': trade_count,
                        'win_rate': win_rate
                    }
                
                # Unrealised P&L by horizon
                for horizon in ('DAY', 'WEEK', 'MONTH'):
                    unrealised = sum(
                        p.unrealized_pnl for p in self.positions.values()
                        if p.horizon == horizon
                    )
                    key = {'DAY': '1d', 'WEEK': '1w', 'MONTH': '1m'}[horizon]
                    result[key]['unrealized_pnl'] = unrealised
                
                return result
        except Exception as e:
            self.logger.error(f"Error calculating returns by window: {e}")
            return {
                '1d': {'pnl': 0, 'return_pct': 0, 'trade_count': 0, 'win_rate': 0, 'unrealized_pnl': 0},
                '1w': {'pnl': 0, 'return_pct': 0, 'trade_count': 0, 'win_rate': 0, 'unrealized_pnl': 0},
                '1m': {'pnl': 0, 'return_pct': 0, 'trade_count': 0, 'win_rate': 0, 'unrealized_pnl': 0}
            }
    
    # ── Snapshots & history ─────────────────────────────────────────

    def save_portfolio_snapshot(self):
        summary = self.get_portfolio_summary()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_history 
                (timestamp, cash_balance, total_value, daily_pnl, total_pnl, num_positions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                summary['timestamp'], summary['cash_balance'],
                summary['total_portfolio_value'], summary['daily_pnl'],
                summary['total_pnl'], summary['num_positions']
            ))
    
    def get_trading_history(self, days: int = 30) -> pd.DataFrame:
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
        try:
            df = self.get_trading_history()
            if df.empty:
                return {}
            closed_trades = df[df['status'] == 'CLOSED']
            if closed_trades.empty:
                return {}
            
            total_trades = len(closed_trades)
            winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
            losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            total_pnl = closed_trades['pnl'].sum()
            avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) \
                if losing_trades > 0 and avg_loss != 0 else float('inf')
            
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
        self.daily_pnl = 0
        self.daily_trades = 0
        self.logger.info("Reset daily metrics for new trading day")
