from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import json
import math
import numpy as np
from datetime import datetime
import threading
import time as _time
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

def _safe_float(v):
    """Return 0 for NaN / Inf so JSON stays valid."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0
    return v

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types (NaN/Inf → 0)."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        return _safe_float(v)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        return _safe_float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):
        return _safe_float(obj.item()) if isinstance(obj.item(), float) else obj.item()
    return obj

# ── Simple in-memory recommendations cache ─────────────────────
_rec_cache = {}          # {horizon: {'data': [...], 'ts': float}}
_REC_CACHE_TTL = 60      # seconds – serve cached data if < 60 s old
_REC_MAX_SYMBOLS = 25    # analyse at most this many symbols per request

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

# ── Manual Buy / Sell ──────────────────────────────────────

@app.route('/api/trade', methods=['POST'])
def manual_trade():
    """Place a manual buy or sell order via Alpaca.

    Body JSON:
        symbol   (str, required) – e.g. "AAPL"
        side     (str, required) – "buy" or "sell"
        quantity (int, optional) – number of shares (default: auto-size)
        order_type (str, optional) – "market" (default) or "limit"
        limit_price (float, optional) – required when order_type is "limit"
        horizon  (str, optional) – "WEEK" or "MONTH" (default "WEEK")
    """
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500

    data = request.get_json() or {}
    symbol = (data.get('symbol') or '').upper()
    side = (data.get('side') or '').lower()
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400
    if side not in ('buy', 'sell'):
        return jsonify({'error': 'Side must be "buy" or "sell"'}), 400

    horizon = (data.get('horizon') or 'WEEK').upper()
    if horizon not in ('WEEK', 'MONTH'):
        horizon = 'WEEK'
    order_type = data.get('order_type', 'market')
    limit_price = data.get('limit_price')

    try:
        pm = trading_agent.portfolio_manager
        broker = pm.broker

        if not broker.is_connected():
            return jsonify({'error': 'Broker not connected'}), 503

        # ── SELL path: close existing position ──────────────
        if side == 'sell':
            if symbol not in pm.positions:
                # Try Alpaca-only close
                result = broker.close_position(symbol)
                if result.get('success'):
                    trading_agent.notifications.notify_trade_closed(
                        symbol, 0, 0, 0, 'Manual sell')
                    return jsonify({'message': f'Closed Alpaca position for {symbol}', **result})
                return jsonify({'error': f'No open position in {symbol}'}), 400

            pos = pm.positions[symbol]
            quote = trading_agent.data_provider.get_real_time_quote(symbol)
            exit_price = quote.get('current_price', pos.current_price) if quote else pos.current_price
            pos.update_price(exit_price)
            pnl = pos.unrealized_pnl
            success = pm.close_position(symbol, exit_price, 'Manual sell')
            if success:
                trading_agent.notifications.notify_trade_closed(
                    symbol, pos.quantity, exit_price, pnl, 'Manual sell')
                return jsonify({
                    'message': f'Sold {symbol}',
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'success': True
                })
            return jsonify({'error': f'Failed to close position for {symbol}'}), 500

        # ── BUY path ────────────────────────────────────────
        quote = trading_agent.data_provider.get_real_time_quote(symbol)
        if not quote or not quote.get('current_price'):
            return jsonify({'error': f'Cannot get price for {symbol}'}), 400

        current_price = quote['current_price']

        # Auto-size if quantity not given
        quantity = data.get('quantity')
        if not quantity:
            max_pos = pm.config.MAX_POSITION_SIZE
            quantity = max(1, int(max_pos / current_price))

        quantity = int(quantity)
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be > 0'}), 400

        # Pre-check risk limits (return specific reason)
        position_size = quantity * current_price
        can_open, reason = pm.can_open_position(symbol, position_size)
        if not can_open:
            return jsonify({'error': reason}), 400

        # Get stop-loss / take-profit from strategy analysis
        stop_loss = 0
        take_profit = 0
        try:
            analysis = trading_agent.trading_strategy.analyze_stock(symbol, horizon=horizon)
            if analysis and 'recommendation' in analysis:
                rec = analysis['recommendation']
                stop_loss = rec.get('stop_loss', 0)
                take_profit = rec.get('take_profit', 0)
        except Exception:
            pass  # proceed without SL/TP

        # Check if market is open to inform user about order timing
        market_open = broker.is_market_open()

        success = pm.open_position(
            symbol=symbol, quantity=quantity,
            entry_price=current_price, position_type='LONG',
            stop_loss=stop_loss, take_profit=take_profit,
            horizon=horizon
        )

        if success:
            if market_open:
                msg = f'Bought {quantity} x {symbol} @ ${current_price:.2f}'
            else:
                msg = (f'Order for {quantity} x {symbol} queued @ ~${current_price:.2f}. '
                       f'Market is closed — it will fill at next market open.')
            trading_agent.notifications.notify_trade_opened(
                symbol, 'BUY', quantity, current_price, horizon)
            return jsonify({
                'message': msg,
                'symbol': symbol,
                'quantity': quantity,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'horizon': horizon,
                'market_open': market_open,
                'success': True
            })
        else:
            return jsonify({'error': f'Order for {symbol} was rejected by the broker. Check logs for details.'}), 400

    except Exception as e:
        app.logger.error(f"Manual trade error: {e}")
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
    """Get realised + unrealised returns for 1W / 1M windows."""
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
    """Get recommendations for watchlist symbols (cached, limited to fastest N)."""
    if not trading_agent:
        return jsonify({'error': 'Trading agent not initialized'}), 500
    try:
        horizon = request.args.get('horizon', 'WEEK').upper()
        if horizon not in ('WEEK', 'MONTH'):
            horizon = 'WEEK'

        # ── Return cached data if fresh enough ─────────────────
        cached = _rec_cache.get(horizon)
        if cached and (_time.time() - cached['ts']) < _REC_CACHE_TTL:
            return jsonify({'recommendations': cached['data'],
                            'horizon': horizon, 'cached': True})

        # ── Build fresh recommendations (limit count) ──────────
        symbols = list(trading_agent.watchlist)[:_REC_MAX_SYMBOLS]
        recommendations = []
        for symbol in symbols:
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

        # ── Store in cache ─────────────────────────────────────
        _rec_cache[horizon] = {'data': recommendations, 'ts': _time.time()}

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
        horizon = request.args.get('horizon', 'WEEK').upper()
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
        screen_type = request.args.get('type', 'swing')
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
