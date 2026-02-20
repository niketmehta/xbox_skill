import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (set these in .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Twilio SMS Notifications
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
    TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '')
    NOTIFICATION_PHONE_NUMBER = os.getenv('NOTIFICATION_PHONE_NUMBER', '')
    NOTIFICATIONS_ENABLED = os.getenv('NOTIFICATIONS_ENABLED', 'false').lower() == 'true'
    
    # Trading Parameters
    MAX_PORTFOLIO_VALUE = float(os.getenv('MAX_PORTFOLIO_VALUE', 10000))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 1000))
    
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
