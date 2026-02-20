import alpaca_trade_api as tradeapi
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import Config

class AlpacaBroker:
    """
    Alpaca broker integration for executing trades
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize Alpaca API
            self.api = tradeapi.REST(
                key_id=self.config.ALPACA_API_KEY,
                secret_key=self.config.ALPACA_SECRET_KEY,
                base_url=self.config.ALPACA_BASE_URL,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.logger.info(f"Connected to Alpaca - Account: {account.id}")
            self.logger.info(f"Paper Trading: {account.pattern_day_trader}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.api = None
    
    def is_connected(self) -> bool:
        """Check if broker connection is active"""
        return self.api is not None
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.api:
            return {}
        
        try:
            account = self.api.get_account()
            
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'day_trade_count': getattr(account, 'day_trade_count', 0),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'trade_suspended_by_user': getattr(account, 'trade_suspended_by_user', False),
                'trading_blocked': getattr(account, 'trading_blocked', False),
                'account_blocked': getattr(account, 'account_blocked', False)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.api:
            return []
        
        try:
            positions = self.api.list_positions()
            
            position_list = []
            for pos in positions:
                position_data = {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'side': pos.side,
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price) if pos.current_price else 0,
                    'market_value': float(pos.market_value) if pos.market_value else 0,
                    'unrealized_pl': float(pos.unrealized_pl) if pos.unrealized_pl else 0,
                    'unrealized_plpc': float(pos.unrealized_plpc) if pos.unrealized_plpc else 0,
                    'cost_basis': float(pos.cost_basis),
                    'asset_id': pos.asset_id,
                    'exchange': pos.exchange
                }
                position_list.append(position_data)
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def place_order(self, symbol: str, quantity: int, side: str, 
                   order_type: str = 'market', time_in_force: str = 'gtc',
                   limit_price: float = None, stop_price: float = None,
                   extended_hours: bool = False) -> Dict:
        """
        Place an order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
        """
        if not self.api:
            return {'error': 'Not connected to broker'}
        
        try:
            order_data = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if limit_price and order_type in ['limit', 'stop_limit']:
                order_data['limit_price'] = round(limit_price, 2)
            
            if stop_price and order_type in ['stop', 'stop_limit']:
                order_data['stop_price'] = round(stop_price, 2)

            # Extended-hours flag only valid for limit orders with 'day' TIF
            if extended_hours and order_type == 'limit' and time_in_force == 'day':
                order_data['extended_hours'] = True
            
            order = self.api.submit_order(**order_data)
            
            self.logger.info(f"Order placed: {side} {quantity} {symbol} @ {order_type}")
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side,
                'order_type': order.order_type,
                'status': order.status,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'submitted_at': order.submitted_at,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'error': str(e), 'success': False}
    
    def place_bracket_order(self, symbol: str, quantity: int, side: str,
                          limit_price: float = None, stop_loss: float = None,
                          take_profit: float = None) -> Dict:
        """
        Place a bracket order with stop loss and take profit.
        Prices are rounded to 2 decimal places (Alpaca rejects sub-penny).
        """
        if not self.api:
            return {'error': 'Not connected to broker'}
        
        try:
            order_data = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': 'market',
                'time_in_force': 'day'   # bracket orders require 'day'
            }
            
            # Add bracket orders if specified (round to cents)
            if stop_loss or take_profit:
                order_class = 'bracket'
                order_data['order_class'] = order_class
                
                if stop_loss:
                    order_data['stop_loss'] = {'stop_price': round(stop_loss, 2)}
                
                if take_profit:
                    order_data['take_profit'] = {'limit_price': round(take_profit, 2)}
            
            order = self.api.submit_order(**order_data)
            
            self.logger.info(f"Bracket order placed: {side} {quantity} {symbol}")
            
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side,
                'status': order.status,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return {'error': str(e), 'success': False}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.api:
            return False
        
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_orders(self, status: str = 'all', limit: int = 50) -> List[Dict]:
        """Get orders"""
        if not self.api:
            return []
        
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            
            order_list = []
            for order in orders:
                order_data = {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'quantity': int(order.qty),
                    'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                    'side': order.side,
                    'order_type': order.order_type,
                    'status': order.status,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'time_in_force': order.time_in_force
                }
                order_list.append(order_data)
            
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def close_position(self, symbol: str, quantity: int = None) -> Dict:
        """Close a position (sell all or specified quantity).
        Uses GTC time-in-force so orders are accepted even outside market hours.
        """
        if not self.api:
            return {'error': 'Not connected to broker'}
        
        try:
            if quantity:
                # Close partial position
                result = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                # Close entire position via Alpaca position API
                try:
                    result = self.api.close_position(symbol)
                except Exception as close_err:
                    # Fallback: get qty from Alpaca positions and submit GTC sell
                    self.logger.warning(
                        f"close_position API failed for {symbol}: {close_err}. "
                        f"Falling back to GTC market sell."
                    )
                    positions = self.api.list_positions()
                    qty = None
                    for p in positions:
                        if p.symbol == symbol:
                            qty = int(p.qty)
                            break
                    if not qty:
                        raise close_err  # re-raise original
                    result = self.api.submit_order(
                        symbol=symbol, qty=qty, side='sell',
                        type='market', time_in_force='gtc'
                    )
            
            self.logger.info(f"Position closed: {symbol}")
            
            return {
                'symbol': symbol,
                'quantity': quantity or 'all',
                'success': True,
                'order_id': result.id if hasattr(result, 'id') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return {'error': str(e), 'success': False}
    
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        if not self.api:
            return {'error': 'Not connected to broker'}
        
        try:
            result = self.api.close_all_positions()
            self.logger.info("All positions closed")
            
            return {
                'message': 'All positions closed',
                'success': True,
                'orders': len(result) if result else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {'error': str(e), 'success': False}
    
    def get_asset_info(self, symbol: str) -> Dict:
        """Get asset information"""
        if not self.api:
            return {}
        
        try:
            asset = self.api.get_asset(symbol)
            
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'asset_class': asset.asset_class,
                'status': asset.status,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
            }
            
        except Exception as e:
            self.logger.error(f"Error getting asset info for {symbol}: {e}")
            return {}
    
    def is_market_open(self) -> bool:
        """Check if market is open according to Alpaca"""
        if not self.api:
            return False
        
        try:
            clock = self.api.get_clock()
            return clock.is_open
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def get_market_calendar(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get market calendar"""
        if not self.api:
            return []
        
        try:
            calendar = self.api.get_calendar(start=start_date, end=end_date)
            
            calendar_list = []
            for day in calendar:
                calendar_data = {
                    'date': day.date.isoformat(),
                    'open': day.open.isoformat(),
                    'close': day.close.isoformat()
                }
                calendar_list.append(calendar_data)
            
            return calendar_list
            
        except Exception as e:
            self.logger.error(f"Error getting market calendar: {e}")
            return []