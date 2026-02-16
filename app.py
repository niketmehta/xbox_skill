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
    global trading_agent
    if trading_agent:
        trading_agent.start()

# ── Helpers ─────────────────────────────────────────────────────

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
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

# ── Dashboard ───────────────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# ── Agent status & control ──────────────────────────────────────

@app.route('/api/status')
def get_status():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        status = trading_agent.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start-agent', methods=['POST'])
def start_agent():
    global trading_agent, agent_thread
    if trading_agent and trading_agent.is_running:
        return jsonify({'message': 'Trading agent is already running'})
    try:
        if not trading_agent:
            trading_agent = TradingAgent()
        if not agent_thread or not agent_thread.is_alive():
            agent_thread = threading.Thread(target=start_trading_agent, daemon=True)
            agent_thread.start()
        return jsonify({'message': 'Trading agent started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-agent', methods=['POST'])
def stop_agent():
    global trading_agent
    if not trading_agent:
        return jsonify({'message': 'Trading agent is not running'})
    try:
        trading_agent.stop()
        return jsonify({'message': 'Trading agent stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Portfolio ───────────────────────────────────────────────────

@app.route('/api/portfolio')
def get_portfolio():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        portfolio = trading_agent.portfolio_manager.get_portfolio_summary()
        return jsonify(portfolio)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        performance = trading_agent.portfolio_manager.get_performance_metrics()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        positions = [pos.to_dict() for pos in trading_agent.portfolio_manager.positions.values()]
        return jsonify({'positions': positions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/liquidate', methods=['POST'])
def liquidate_positions():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        trading_agent.force_liquidate()
        return jsonify({'message': 'All positions liquidated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-history')
def get_trading_history():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        days = request.args.get('days', 30, type=int)
        history = trading_agent.portfolio_manager.get_trading_history(days)
        if not history.empty:
            return jsonify({'history': history.to_dict('records')})
        else:
            return jsonify({'history': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Returns by window (NEW) ────────────────────────────────────

@app.route('/api/returns')
def get_returns():
    """Get realised + unrealised returns for 1D / 1W / 1M windows."""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        returns = trading_agent.portfolio_manager.get_returns_by_window()
        returns = convert_numpy_types(returns)
        return jsonify(returns)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Watchlist ───────────────────────────────────────────────────

@app.route('/api/watchlist')
def get_watchlist():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        return jsonify({'watchlist': trading_agent.watchlist})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
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
    """Get recommendations for all watchlist symbols (supports horizon query param)."""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        horizon = request.args.get('horizon', 'DAY').upper()
        if horizon not in ('DAY', 'WEEK', 'MONTH'):
            horizon = 'DAY'
        
        recommendations = []
        for symbol in trading_agent.watchlist:
            try:
                analysis = trading_agent.analyze_symbol(symbol, horizon=horizon)
                if analysis and 'recommendation' in analysis:
                    rec = analysis['recommendation']
                    recommendations.append({
                        'symbol': symbol,
                        'current_price': analysis.get('current_price', 0),
                        'action': rec.get('action', 'HOLD'),
                        'confidence': rec.get('confidence', 0),
                        'position_size': rec.get('position_size', 0),
                        'stop_loss': rec.get('stop_loss', 0),
                        'take_profit': rec.get('take_profit', 0),
                        'horizon': horizon,
                        'timestamp': analysis.get('timestamp', datetime.now())
                    })
            except Exception as e:
                app.logger.error(f"Error analysing {symbol}: {e}")
                recommendations.append({
                    'symbol': symbol,
                    'current_price': 0,
                    'action': 'ERROR',
                    'confidence': 0,
                    'position_size': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'horizon': horizon,
                    'timestamp': datetime.now()
                })
        
        recommendations = convert_numpy_types(recommendations)
        return jsonify({'recommendations': recommendations, 'horizon': horizon})
    except Exception as e:
        app.logger.error(f"Error getting watchlist recommendations: {e}")
        return jsonify({'error': str(e)}), 500

# ── Analysis ────────────────────────────────────────────────────

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        horizon = request.args.get('horizon', 'DAY').upper()
        analysis = trading_agent.analyze_symbol(symbol.upper(), horizon=horizon)
        analysis = convert_numpy_types(analysis)
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error analysing {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<symbol>/all')
def analyze_symbol_all(symbol):
    """Analyse a symbol across all three horizons."""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        analysis = trading_agent.analyze_symbol_all_horizons(symbol.upper())
        analysis = convert_numpy_types(analysis)
        return jsonify(analysis)
    except Exception as e:
        app.logger.error(f"Error analysing {symbol} (all horizons): {e}")
        return jsonify({'error': str(e)}), 500

# ── Market movers ───────────────────────────────────────────────

@app.route('/api/market-movers')
def get_market_movers():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        count = request.args.get('count', 20, type=int)
        movers = trading_agent.data_provider.get_market_movers(count)
        return jsonify({'movers': movers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Config ──────────────────────────────────────────────────────

@app.route('/api/config')
def get_config():
    try:
        config = Config()
        config_dict = {
            'MAX_PORTFOLIO_VALUE': config.MAX_PORTFOLIO_VALUE,
            'MAX_POSITION_SIZE': config.MAX_POSITION_SIZE,
            'DAY_STOP_LOSS_PCT': config.DAY_STOP_LOSS_PCT,
            'DAY_TAKE_PROFIT_PCT': config.DAY_TAKE_PROFIT_PCT,
            'WEEK_STOP_LOSS_PCT': config.WEEK_STOP_LOSS_PCT,
            'WEEK_TAKE_PROFIT_PCT': config.WEEK_TAKE_PROFIT_PCT,
            'MONTH_STOP_LOSS_PCT': config.MONTH_STOP_LOSS_PCT,
            'MONTH_TAKE_PROFIT_PCT': config.MONTH_TAKE_PROFIT_PCT,
            'MAX_DAILY_LOSS': config.MAX_DAILY_LOSS,
            'MAX_POSITIONS': config.MAX_POSITIONS,
            'RSI_OVERSOLD': config.RSI_OVERSOLD,
            'RSI_OVERBOUGHT': config.RSI_OVERBOUGHT,
            'VOLUME_SPIKE_THRESHOLD': config.VOLUME_SPIKE_THRESHOLD
        }
        return jsonify(config_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Broker ──────────────────────────────────────────────────────

@app.route('/api/broker/account')
def get_broker_account():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        if hasattr(trading_agent.portfolio_manager, 'broker'):
            account_info = trading_agent.portfolio_manager.broker.get_account_info()
            broker_connected = trading_agent.portfolio_manager.broker.is_connected()
            return jsonify({'connected': broker_connected, 'account_info': account_info})
        else:
            return jsonify({'connected': False, 'account_info': {}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/broker/orders')
def get_broker_orders():
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

# ── Auto watchlist / screening ──────────────────────────────────

@app.route('/api/auto-watchlist/enable', methods=['POST'])
def enable_auto_watchlist():
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
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        screen_type = request.args.get('type', 'day_trading')
        max_stocks = request.args.get('max_stocks', 25, type=int)
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
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        options = trading_agent.get_screening_options()
        return jsonify({'options': options})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto-watchlist/status')
def get_auto_watchlist_status():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        return jsonify({
            'enabled': trading_agent.auto_watchlist_enabled,
            'last_update': trading_agent.last_watchlist_update.isoformat()
                if trading_agent.last_watchlist_update != trading_agent.last_watchlist_update.min else None,
            'watchlist_size': len(trading_agent.watchlist)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Notifications (NEW) ────────────────────────────────────────

@app.route('/api/notifications/config', methods=['GET'])
def get_notification_config():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        ns = trading_agent.notifications
        return jsonify({
            'enabled': ns.is_enabled(),
            'phone_number': ns.get_phone_number()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/config', methods=['POST'])
def update_notification_config():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        data = request.get_json() or {}
        ns = trading_agent.notifications
        
        if 'enabled' in data:
            ns.enable(data['enabled'])
        if 'phone_number' in data:
            ns.set_phone_number(data['phone_number'])
        
        return jsonify({
            'message': 'Notification settings updated',
            'enabled': ns.is_enabled(),
            'phone_number': ns.get_phone_number()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/test', methods=['POST'])
def test_notification():
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        success = trading_agent.notifications.send_test()
        if success:
            return jsonify({'message': 'Test SMS sent successfully'})
        else:
            return jsonify({'error': 'Failed to send test SMS. Check Twilio configuration.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Error handlers ──────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ── Main ────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trading_agent = TradingAgent()
    config = Config()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
