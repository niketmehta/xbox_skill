import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (set these in .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    ALPACA_DATA_FEED = os.getenv('ALPACA_DATA_FEED', 'iex')
    STOOQ_API_KEY = os.getenv('STOOQ_API_KEY', '')
    
    # OpenClaw WhatsApp delivery
    OPENCLAW_ENABLED = os.getenv('OPENCLAW_ENABLED', 'false').lower() == 'true'
    OPENCLAW_CLI = os.getenv('OPENCLAW_CLI', 'openclaw')
    OPENCLAW_CHANNEL = os.getenv('OPENCLAW_CHANNEL', 'whatsapp')
    OPENCLAW_ACCOUNT = os.getenv('OPENCLAW_ACCOUNT', '')
    OPENCLAW_WHATSAPP_TARGET = os.getenv('OPENCLAW_WHATSAPP_TARGET', '')
    OPENCLAW_TIMEOUT_SECONDS = int(os.getenv('OPENCLAW_TIMEOUT_SECONDS', 45))
    OPENCLAW_HANDSHAKE_TIMEOUT_MS = int(os.getenv('OPENCLAW_HANDSHAKE_TIMEOUT_MS', 120000))
    OPENCLAW_SEND_ATTEMPTS = int(os.getenv('OPENCLAW_SEND_ATTEMPTS', 4))
    OPENCLAW_AUTO_RESTART = os.getenv('OPENCLAW_AUTO_RESTART', 'true').lower() == 'true'

    # Scheduled council digest
    TOP_RECOMMENDATIONS_ENABLED = os.getenv('TOP_RECOMMENDATIONS_ENABLED', 'false').lower() == 'true'
    TOP_RECOMMENDATIONS_TIME = os.getenv('TOP_RECOMMENDATIONS_TIME', '06:15')
    TOP_RECOMMENDATIONS_HORIZON = os.getenv('TOP_RECOMMENDATIONS_HORIZON', 'WEEK')
    TOP_RECOMMENDATIONS_UNIVERSE_SIZE = int(os.getenv('TOP_RECOMMENDATIONS_UNIVERSE_SIZE', 100))
    TOP_RECOMMENDATIONS_SMART_UNIVERSE = os.getenv('TOP_RECOMMENDATIONS_SMART_UNIVERSE', 'true').lower() == 'true'
    TOP_RECOMMENDATIONS_MIN_CONFIDENCE = float(os.getenv('TOP_RECOMMENDATIONS_MIN_CONFIDENCE', 20.0))
    TOP_RECOMMENDATIONS_HISTORY_LOOKBACK_DAYS = int(os.getenv('TOP_RECOMMENDATIONS_HISTORY_LOOKBACK_DAYS', 30))
    TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES = int(os.getenv('TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES', 2))
    TOP_RECOMMENDATIONS_MAX_PER_SECTOR = int(os.getenv('TOP_RECOMMENDATIONS_MAX_PER_SECTOR', 2))
    COUNCIL_RAG_ENABLED = os.getenv('COUNCIL_RAG_ENABLED', 'true').lower() == 'true'
    COUNCIL_RAG_LOOKBACK_DAYS = int(os.getenv('COUNCIL_RAG_LOOKBACK_DAYS', 45))
    COUNCIL_RAG_MAX_LESSONS = int(os.getenv('COUNCIL_RAG_MAX_LESSONS', 6))
    COUNCIL_RAG_SYMBOL_WEIGHT = float(os.getenv('COUNCIL_RAG_SYMBOL_WEIGHT', 1.0))
    COUNCIL_RAG_SECTOR_WEIGHT = float(os.getenv('COUNCIL_RAG_SECTOR_WEIGHT', 0.45))
    COUNCIL_RAG_MARKET_WEIGHT = float(os.getenv('COUNCIL_RAG_MARKET_WEIGHT', 0.20))
    COUNCIL_RAG_MAX_SCORE_ADJUSTMENT = float(os.getenv('COUNCIL_RAG_MAX_SCORE_ADJUSTMENT', 14.0))
    COUNCIL_RAG_MISSED_MOVER_MIN_CHANGE_PCT = float(os.getenv('COUNCIL_RAG_MISSED_MOVER_MIN_CHANGE_PCT', 3.0))
    COUNCIL_RAG_MISSED_MOVER_WEIGHT = float(os.getenv('COUNCIL_RAG_MISSED_MOVER_WEIGHT', 0.60))
    HELD_MOMENTUM_ALERT_LIMIT = int(os.getenv('HELD_MOMENTUM_ALERT_LIMIT', 3))
    INTRADAY_BREAKOUT_ALERT_LIMIT = int(os.getenv('INTRADAY_BREAKOUT_ALERT_LIMIT', 5))
    INTRADAY_BREAKOUT_MIN_CHANGE_PCT = float(os.getenv('INTRADAY_BREAKOUT_MIN_CHANGE_PCT', 2.0))
    INTRADAY_BREAKOUT_MIN_VOLUME_RATIO = float(os.getenv('INTRADAY_BREAKOUT_MIN_VOLUME_RATIO', 1.1))

    # Simulated open-entry / end-of-day P&L summary
    SIMULATION_ENABLED = os.getenv('SIMULATION_ENABLED', 'true').lower() == 'true'
    SIMULATION_TOP_N = int(os.getenv('SIMULATION_TOP_N', 5))
    SIMULATION_NOTIONAL_PER_PICK = float(os.getenv('SIMULATION_NOTIONAL_PER_PICK', 1000))
    SIMULATION_OPEN_TIME = os.getenv('SIMULATION_OPEN_TIME', '06:35')
    SIMULATION_MIDDAY_TIME = os.getenv('SIMULATION_MIDDAY_TIME', '09:00')
    SIMULATION_EOD_TIME = os.getenv('SIMULATION_EOD_TIME', '13:10')
    SIMULATION_OPEN_WHATSAPP_ENABLED = os.getenv('SIMULATION_OPEN_WHATSAPP_ENABLED', 'true').lower() == 'true'
    SIMULATION_BACKFILL_ON_SUMMARY = os.getenv('SIMULATION_BACKFILL_ON_SUMMARY', 'true').lower() == 'true'
    
    # Trading Parameters
    MAX_PORTFOLIO_VALUE = float(os.getenv('MAX_PORTFOLIO_VALUE', 10000))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 5000))
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 500))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))
    
    # ── Time-Horizon Settings (WEEK = swing, MONTH = position) ────
    # No more DAY horizon — we focus on swing and position trades.
    
    # WEEK (swing trade, held ~3-10 trading days)
    WEEK_STOP_LOSS_PCT = float(os.getenv('WEEK_STOP_LOSS_PCT', 0.04))     # 4%
    WEEK_TAKE_PROFIT_PCT = float(os.getenv('WEEK_TAKE_PROFIT_PCT', 0.08)) # 8%
    WEEK_MIN_CONFIDENCE = float(os.getenv('WEEK_MIN_CONFIDENCE', 60))      # 60%
    WEEK_ATR_SL_MULT = float(os.getenv('WEEK_ATR_SL_MULT', 2.0))          # Stop = 2× ATR
    WEEK_ATR_TP_MULT = float(os.getenv('WEEK_ATR_TP_MULT', 3.0))          # Profit = 3× ATR
    
    # MONTH (position trade, held ~15-45 trading days)
    MONTH_STOP_LOSS_PCT = float(os.getenv('MONTH_STOP_LOSS_PCT', 0.07))   # 7%
    MONTH_TAKE_PROFIT_PCT = float(os.getenv('MONTH_TAKE_PROFIT_PCT', 0.15)) # 15%
    MONTH_MIN_CONFIDENCE = float(os.getenv('MONTH_MIN_CONFIDENCE', 55))    # 55%
    MONTH_ATR_SL_MULT = float(os.getenv('MONTH_ATR_SL_MULT', 3.0))        # Stop = 3× ATR
    MONTH_ATR_TP_MULT = float(os.getenv('MONTH_ATR_TP_MULT', 5.0))        # Profit = 5× ATR
    
    # Legacy aliases (for any old code paths)
    STOP_LOSS_PERCENTAGE = WEEK_STOP_LOSS_PCT
    TAKE_PROFIT_PERCENTAGE = WEEK_TAKE_PROFIT_PCT
    
    # ── Market Regime / VIX Thresholds ────────────────────────────
    VIX_LOW = float(os.getenv('VIX_LOW', 15))       # Low vol → full position sizes
    VIX_MODERATE = float(os.getenv('VIX_MODERATE', 25))  # Moderate → reduce sizes
    VIX_HIGH = float(os.getenv('VIX_HIGH', 35))      # High fear → very selective
    
    # ── Fundamental Filters ───────────────────────────────────────
    MIN_MARKET_CAP = float(os.getenv('MIN_MARKET_CAP', 500e6))   # $500M minimum
    MAX_PE_RATIO = float(os.getenv('MAX_PE_RATIO', 60))          # Avoid extreme P/E
    MIN_PROFIT_MARGIN = float(os.getenv('MIN_PROFIT_MARGIN', 0.0))  # 0% = allow unprofitable growth
    
    # Trading Hours (Eastern Time) — still needed for market-open detection
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    EXTENDED_HOURS_START = "04:00"
    EXTENDED_HOURS_END = "20:00"
    
    # Data Sources
    USE_YAHOO_FINANCE = True
    USE_ALPHA_VANTAGE = True
    USE_ALPACA_MARKET_DATA = os.getenv('USE_ALPACA_MARKET_DATA', 'true').lower() == 'true'
    MARKET_DATA_PRIMARY = os.getenv('MARKET_DATA_PRIMARY', 'yahoo').lower()
    USE_STOOQ_FALLBACK = os.getenv('USE_STOOQ_FALLBACK', 'true').lower() == 'true'
    YAHOO_COOLDOWN_SECONDS = int(os.getenv('YAHOO_COOLDOWN_SECONDS', 900))
    
    # Strategy Parameters
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE_THRESHOLD = 2.0
    
    # Flask Settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5001
    FLASK_DEBUG = True
    
    # ── Helpers ─────────────────────────────────────────────────────
    @classmethod
    def get_horizon_params(cls, horizon: str) -> dict:
        """Return stop-loss %, take-profit %, min confidence, and ATR multipliers."""
        horizon = horizon.upper()
        if horizon == 'MONTH':
            return {
                'stop_loss_pct': cls.MONTH_STOP_LOSS_PCT,
                'take_profit_pct': cls.MONTH_TAKE_PROFIT_PCT,
                'min_confidence': cls.MONTH_MIN_CONFIDENCE,
                'atr_sl_mult': cls.MONTH_ATR_SL_MULT,
                'atr_tp_mult': cls.MONTH_ATR_TP_MULT,
            }
        else:  # WEEK (default)
            return {
                'stop_loss_pct': cls.WEEK_STOP_LOSS_PCT,
                'take_profit_pct': cls.WEEK_TAKE_PROFIT_PCT,
                'min_confidence': cls.WEEK_MIN_CONFIDENCE,
                'atr_sl_mult': cls.WEEK_ATR_SL_MULT,
                'atr_tp_mult': cls.WEEK_ATR_TP_MULT,
            }
