import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import yfinance as yf
from data_provider import MarketDataProvider
from trading_strategy import TradingStrategy
from portfolio_manager import PortfolioManager

class Backtester:
    """
    Backtesting engine for validating trading strategies
    """
    
    def __init__(self):
        self.data_provider = MarketDataProvider()
        self.trading_strategy = TradingStrategy(self.data_provider)
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str,
                    initial_capital: float = 10000) -> Dict:
        """
        Run backtest on given symbols and date range
        
        Args:
            symbols: List of symbols to test
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_capital: Starting capital amount
        """
        
        self.logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Initialize backtest state
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'trades': [],
            'daily_values': [],
            'daily_returns': []
        }
        
        # Get historical data for all symbols
        historical_data = self._get_historical_data(symbols, start_date, end_date)
        
        if not historical_data:
            return {'error': 'No historical data available'}
        
        # Run day-by-day simulation
        trading_days = self._get_trading_days(start_date, end_date)
        
        for day in trading_days:
            self._simulate_trading_day(day, symbols, historical_data, portfolio)
        
        # Calculate performance metrics
        performance = self._calculate_performance(portfolio, initial_capital)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_value': portfolio['cash'] + sum(pos['market_value'] for pos in portfolio['positions'].values()),
            'total_trades': len(portfolio['trades']),
            'performance': performance,
            'trades': portfolio['trades'],
            'daily_values': portfolio['daily_values']
        }
    
    def _get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols"""
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if not data.empty:
                    # Add technical indicators
                    data = self.data_provider._add_technical_indicators(data)
                    historical_data[symbol] = data
                    self.logger.info(f"Loaded {len(data)} days of data for {symbol}")
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
        
        return historical_data
    
    def _get_trading_days(self, start_date: str, end_date: str) -> List[datetime]:
        """Get list of trading days between dates"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current = start
        
        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def _simulate_trading_day(self, day: datetime, symbols: List[str], 
                            historical_data: Dict[str, pd.DataFrame], portfolio: Dict):
        """Simulate one trading day"""
        
        day_str = day.strftime('%Y-%m-%d')
        
        # Update position values
        self._update_position_values(day, historical_data, portfolio)
        
        # Check for stop loss and take profit triggers
        self._check_exit_triggers(day, historical_data, portfolio)
        
        # Look for new trading opportunities
        for symbol in symbols:
            if symbol not in historical_data:
                continue
                
            data = historical_data[symbol]
            
            # Get data up to current day
            current_data = data[data.index.date <= day.date()]
            
            if len(current_data) < 50:  # Need enough data for indicators
                continue
            
            # Simulate real-time quote
            latest = current_data.iloc[-1]
            quote = {
                'symbol': symbol,
                'current_price': latest['Close'],
                'previous_close': current_data.iloc[-2]['Close'] if len(current_data) > 1 else latest['Close'],
                'volume': latest['Volume'],
                'timestamp': day
            }
            
            # Calculate change
            quote['change'] = quote['current_price'] - quote['previous_close']
            quote['change_percent'] = (quote['change'] / quote['previous_close'] * 100) if quote['previous_close'] else 0
            
            # Get trading signal
            try:
                signals = self._get_trading_signals(current_data, quote)
                
                if signals and signals.get('recommendation'):
                    self._execute_backtest_trade(day, symbol, signals, portfolio, quote['current_price'])
                    
            except Exception as e:
                self.logger.error(f"Error getting signals for {symbol} on {day_str}: {e}")
        
        # Record daily portfolio value
        total_value = portfolio['cash']
        for pos in portfolio['positions'].values():
            total_value += pos['market_value']
        
        portfolio['daily_values'].append({
            'date': day_str,
            'total_value': total_value,
            'cash': portfolio['cash'],
            'positions_value': total_value - portfolio['cash']
        })
    
    def _get_trading_signals(self, data: pd.DataFrame, quote: Dict) -> Dict:
        """Get trading signals for backtest"""
        try:
            # Use the same strategy logic as live trading
            momentum_signal = self.trading_strategy._momentum_strategy(data, quote)
            reversal_signal = self.trading_strategy._mean_reversion_strategy(data, quote)
            breakout_signal = self.trading_strategy._breakout_strategy(data, quote)
            volume_signal = self.trading_strategy._volume_analysis(data, quote)
            
            # Combine signals
            combined_score = self.trading_strategy._combine_signals(
                momentum_signal, reversal_signal, breakout_signal, volume_signal
            )
            
            # Risk assessment
            risk_score = self.trading_strategy._assess_risk(data, quote, {})
            
            # Generate recommendation
            recommendation = self.trading_strategy._generate_recommendation(combined_score, risk_score, quote)
            
            return {
                'signals': {
                    'momentum': momentum_signal,
                    'reversal': reversal_signal,
                    'breakout': breakout_signal,
                    'volume': volume_signal,
                    'combined_score': combined_score
                },
                'risk_score': risk_score,
                'recommendation': recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading signals: {e}")
            return {}
    
    def _execute_backtest_trade(self, day: datetime, symbol: str, signals: Dict, 
                              portfolio: Dict, current_price: float):
        """Execute a trade in backtest"""
        
        recommendation = signals.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 0)
        
        # Only execute high-confidence signals
        if confidence < 70:
            return
        
        # Check if already have position
        if symbol in portfolio['positions']:
            return
        
        # Check available cash
        position_size = recommendation.get('position_size', 1000)
        if position_size > portfolio['cash']:
            position_size = portfolio['cash'] * 0.8  # Use 80% of available cash
        
        if action == 'BUY' and position_size >= current_price:
            quantity = int(position_size / current_price)
            total_cost = quantity * current_price
            
            if total_cost <= portfolio['cash']:
                # Execute buy
                portfolio['cash'] -= total_cost
                portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_date': day,
                    'current_price': current_price,
                    'market_value': total_cost,
                    'stop_loss': recommendation.get('stop_loss', 0),
                    'take_profit': recommendation.get('take_profit', 0),
                    'unrealized_pnl': 0
                }
                
                # Record trade
                portfolio['trades'].append({
                    'date': day.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'value': total_cost,
                    'confidence': confidence
                })
                
                self.logger.info(f"Backtest BUY: {symbol} x{quantity} @ ${current_price:.2f}")
    
    def _update_position_values(self, day: datetime, historical_data: Dict[str, pd.DataFrame], portfolio: Dict):
        """Update position values with current prices"""
        
        for symbol, position in portfolio['positions'].items():
            if symbol in historical_data:
                data = historical_data[symbol]
                current_data = data[data.index.date <= day.date()]
                
                if not current_data.empty:
                    current_price = current_data.iloc[-1]['Close']
                    position['current_price'] = current_price
                    position['market_value'] = position['quantity'] * current_price
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
    
    def _check_exit_triggers(self, day: datetime, historical_data: Dict[str, pd.DataFrame], portfolio: Dict):
        """Check for stop loss and take profit triggers"""
        
        positions_to_close = []
        
        for symbol, position in portfolio['positions'].items():
            current_price = position['current_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            exit_reason = None
            
            # Check stop loss
            if stop_loss and current_price <= stop_loss:
                exit_reason = 'Stop Loss'
            
            # Check take profit
            elif take_profit and current_price >= take_profit:
                exit_reason = 'Take Profit'
            
            # Check end of day (simplified - exit all positions at end of day)
            elif day.hour >= 15:  # 3 PM or later
                exit_reason = 'End of Day'
            
            if exit_reason:
                positions_to_close.append((symbol, exit_reason))
        
        # Execute exits
        for symbol, reason in positions_to_close:
            self._close_backtest_position(day, symbol, reason, portfolio)
    
    def _close_backtest_position(self, day: datetime, symbol: str, reason: str, portfolio: Dict):
        """Close a position in backtest"""
        
        if symbol not in portfolio['positions']:
            return
        
        position = portfolio['positions'][symbol]
        exit_price = position['current_price']
        quantity = position['quantity']
        proceeds = quantity * exit_price
        
        # Calculate P&L
        realized_pnl = position['unrealized_pnl']
        
        # Update cash
        portfolio['cash'] += proceeds
        
        # Record trade
        portfolio['trades'].append({
            'date': day.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': exit_price,
            'value': proceeds,
            'pnl': realized_pnl,
            'reason': reason,
            'hold_days': (day - position['entry_date']).days
        })
        
        # Remove position
        del portfolio['positions'][symbol]
        
        self.logger.info(f"Backtest SELL: {symbol} x{quantity} @ ${exit_price:.2f}, P&L: ${realized_pnl:.2f}, Reason: {reason}")
    
    def _calculate_performance(self, portfolio: Dict, initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        
        trades = portfolio['trades']
        daily_values = portfolio['daily_values']
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Basic trade statistics
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl' in t]
        
        total_trades = len(buy_trades)
        winning_trades = len([t for t in sell_trades if t['pnl'] > 0])
        losing_trades = len([t for t in sell_trades if t['pnl'] <= 0])
        
        win_rate = (winning_trades / len(sell_trades) * 100) if sell_trades else 0
        
        # P&L statistics
        total_pnl = sum(t['pnl'] for t in sell_trades)
        wins = [t['pnl'] for t in sell_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in sell_trades if t['pnl'] <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Calculate profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Portfolio performance
        final_value = portfolio['cash'] + sum(pos['market_value'] for pos in portfolio['positions'].values())
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate daily returns and volatility
        if len(daily_values) > 1:
            values = [dv['total_value'] for dv in daily_values]
            daily_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            volatility = np.std(daily_returns) if daily_returns else 0
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            sharpe_ratio = (avg_daily_return * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            max_drawdown_percent = max_drawdown * 100
        else:
            avg_daily_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown_percent = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_percent': max_drawdown_percent,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility
        }

def run_sample_backtest():
    """Run a sample backtest for demonstration"""
    backtester = Backtester()
    
    # Test on popular stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    # Test last 3 months
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"Running backtest from {start_date} to {end_date}")
    
    results = backtester.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000
    )
    
    print("\nBacktest Results:")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['performance']['total_return_percent']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['performance']['win_rate']:.1f}%")
    print(f"Profit Factor: {results['performance']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown_percent']:.2f}%")
    
    return results

if __name__ == "__main__":
    run_sample_backtest()