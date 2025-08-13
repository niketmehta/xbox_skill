from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime
import threading
import logging

from config import Config
from trading_agent import TradingAgent

# JSON encoder to handle numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder
CORS(app)

# Global trading agent instance
trading_agent = None
agent_thread = None

def start_trading_agent():
    """Start the trading agent in a separate thread"""
    global trading_agent
    if trading_agent:
        trading_agent.start()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current status of the trading agent"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        status = trading_agent.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio summary"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        portfolio = trading_agent.portfolio_manager.get_portfolio_summary()
        return jsonify(portfolio)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        performance = trading_agent.portfolio_manager.get_performance_metrics()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist')
def get_watchlist():
    """Get current watchlist"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        return jsonify({'watchlist': trading_agent.watchlist})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """Add symbol to watchlist"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        trading_agent.add_to_watchlist(symbol)
        return jsonify({'message': f'Added {symbol} to watchlist'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    """Remove symbol from watchlist"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        trading_agent.remove_from_watchlist(symbol)
        return jsonify({'message': f'Removed {symbol} from watchlist'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/recommendations')
def get_watchlist_recommendations():
    """Get recommendations for all watchlist symbols"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        recommendations = []
        for symbol in trading_agent.watchlist:
            try:
                analysis = trading_agent.analyze_symbol(symbol)
                if analysis and 'recommendation' in analysis:
                    recommendation = analysis['recommendation']
                    recommendations.append({
                        'symbol': symbol,
                        'current_price': analysis.get('current_price', 0),
                        'action': recommendation.get('action', 'HOLD'),
                        'confidence': recommendation.get('confidence', 0),
                        'position_size': recommendation.get('position_size', 0),
                        'stop_loss': recommendation.get('stop_loss', 0),
                        'take_profit': recommendation.get('take_profit', 0),
                        'timestamp': analysis.get('timestamp', datetime.now())
                    })
            except Exception as e:
                app.logger.error(f"Error analyzing {symbol}: {e}")
                # Add placeholder for failed analysis
                recommendations.append({
                    'symbol': symbol,
                    'current_price': 0,
                    'action': 'ERROR',
                    'confidence': 0,
                    'position_size': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'timestamp': datetime.now()
                })
        
        # Convert numpy types to JSON serializable types
        recommendations = convert_numpy_types(recommendations)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        app.logger.error(f"Error getting watchlist recommendations: {e}")
        return jsonify({'error': str(e)}), 500

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    return obj

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """Get detailed analysis for a symbol"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        analysis = trading_agent.analyze_symbol(symbol.upper())
        # Convert numpy types to JSON serializable types
        analysis = convert_numpy_types(analysis)
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error analyzing {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        positions = []
        for symbol, position in trading_agent.portfolio_manager.positions.items():
            positions.append(position.to_dict())
        return jsonify({'positions': positions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/liquidate', methods=['POST'])
def liquidate_positions():
    """Force liquidation of all positions"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        trading_agent.force_liquidate()
        return jsonify({'message': 'All positions liquidated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-history')
def get_trading_history():
    """Get trading history"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        days = request.args.get('days', 30, type=int)
        history = trading_agent.portfolio_manager.get_trading_history(days)
        
        # Convert DataFrame to dict for JSON serialization
        if not history.empty:
            history_dict = history.to_dict('records')
            return jsonify({'history': history_dict})
        else:
            return jsonify({'history': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-movers')
def get_market_movers():
    """Get current market movers"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        count = request.args.get('count', 20, type=int)
        movers = trading_agent.data_provider.get_market_movers(count)
        return jsonify({'movers': movers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-agent', methods=['POST'])
def start_agent():
    """Start the trading agent"""
    global trading_agent, agent_thread
    
    if trading_agent and trading_agent.is_running:
        return jsonify({'message': 'Trading agent is already running'})
    
    try:
        # Initialize trading agent if not exists
        if not trading_agent:
            trading_agent = TradingAgent()
        
        # Start agent in separate thread
        if not agent_thread or not agent_thread.is_alive():
            agent_thread = threading.Thread(target=start_trading_agent, daemon=True)
            agent_thread.start()
        
        return jsonify({'message': 'Trading agent started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-agent', methods=['POST'])
def stop_agent():
    """Stop the trading agent"""
    global trading_agent
    
    if not trading_agent:
        return jsonify({'message': 'Trading agent is not running'})
    
    try:
        trading_agent.stop()
        return jsonify({'message': 'Trading agent stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    try:
        config = Config()
        config_dict = {
            'MAX_PORTFOLIO_VALUE': config.MAX_PORTFOLIO_VALUE,
            'MAX_POSITION_SIZE': config.MAX_POSITION_SIZE,
            'STOP_LOSS_PERCENTAGE': config.STOP_LOSS_PERCENTAGE,
            'TAKE_PROFIT_PERCENTAGE': config.TAKE_PROFIT_PERCENTAGE,
            'MAX_DAILY_LOSS': config.MAX_DAILY_LOSS,
            'MAX_POSITIONS': config.MAX_POSITIONS,
            'RSI_OVERSOLD': config.RSI_OVERSOLD,
            'RSI_OVERBOUGHT': config.RSI_OVERBOUGHT,
            'VOLUME_SPIKE_THRESHOLD': config.VOLUME_SPIKE_THRESHOLD
        }
        return jsonify(config_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/broker/account')
def get_broker_account():
    """Get broker account information"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        if hasattr(trading_agent.portfolio_manager, 'broker'):
            account_info = trading_agent.portfolio_manager.broker.get_account_info()
            broker_connected = trading_agent.portfolio_manager.broker.is_connected()
            return jsonify({
                'connected': broker_connected,
                'account_info': account_info
            })
        else:
            return jsonify({'connected': False, 'account_info': {}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/broker/orders')
def get_broker_orders():
    """Get broker orders"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        if hasattr(trading_agent.portfolio_manager, 'broker'):
            orders = trading_agent.portfolio_manager.broker.get_orders(limit=50)
            return jsonify({'orders': orders})
        else:
            return jsonify({'orders': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-watchlist/enable', methods=['POST'])
def enable_auto_watchlist():
    """Enable automatic watchlist generation"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled', True)
        
        trading_agent.enable_auto_watchlist(enabled)
        
        return jsonify({
            'message': f"Auto watchlist {'enabled' if enabled else 'disabled'}",
            'enabled': enabled
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screen-stocks')
def screen_stocks():
    """Screen stocks with specific criteria"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        screen_type = request.args.get('type', 'day_trading')
        max_stocks = request.args.get('max_stocks', 50, type=int)
        
        screened_stocks = trading_agent.screen_stocks_manual(screen_type, max_stocks)
        
        return jsonify({
            'screen_type': screen_type,
            'stocks': screened_stocks,
            'count': len(screened_stocks)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screening-options')
def get_screening_options():
    """Get available screening options"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        options = trading_agent.get_screening_options()
        return jsonify({'options': options})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-watchlist/status')
def get_auto_watchlist_status():
    """Get auto watchlist status"""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    
    try:
        return jsonify({
            'enabled': trading_agent.auto_watchlist_enabled,
            'last_update': trading_agent.last_watchlist_update.isoformat() if trading_agent.last_watchlist_update != trading_agent.last_watchlist_update.min else None,
            'watchlist_size': len(trading_agent.watchlist)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trading agent
    trading_agent = TradingAgent()
    
    # Start Flask app
    config = Config()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )