import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (set these in .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Paper trading by default
    
    # Twilio SMS Notifications
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
    TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '')  # Your Twilio phone number
    NOTIFICATION_PHONE_NUMBER = os.getenv('NOTIFICATION_PHONE_NUMBER', '')  # Your personal phone number
    NOTIFICATIONS_ENABLED = os.getenv('NOTIFICATIONS_ENABLED', 'false').lower() == 'true'
    
    # Trading Parameters
    MAX_PORTFOLIO_VALUE = float(os.getenv('MAX_PORTFOLIO_VALUE', 10000))  # Maximum portfolio value
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 1000))       # Maximum per position
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 500))             # Maximum daily loss
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))                   # Maximum concurrent positions
    
    # ── Time-Horizon Specific Settings ──────────────────────────────
    # Each horizon has its own stop-loss, take-profit, and confidence threshold.
    
    # DAY (intraday, liquidated at EOD)
    DAY_STOP_LOSS_PCT = float(os.getenv('DAY_STOP_LOSS_PCT', 0.02))       # 2%
    DAY_TAKE_PROFIT_PCT = float(os.getenv('DAY_TAKE_PROFIT_PCT', 0.04))   # 4%
    DAY_MIN_CONFIDENCE = float(os.getenv('DAY_MIN_CONFIDENCE', 70))        # 70%
    
    # WEEK (swing trade, held up to ~5 trading days)
    WEEK_STOP_LOSS_PCT = float(os.getenv('WEEK_STOP_LOSS_PCT', 0.04))     # 4%
    WEEK_TAKE_PROFIT_PCT = float(os.getenv('WEEK_TAKE_PROFIT_PCT', 0.08)) # 8%
    WEEK_MIN_CONFIDENCE = float(os.getenv('WEEK_MIN_CONFIDENCE', 65))      # 65%
    
    # MONTH (position trade, held up to ~22 trading days)
    MONTH_STOP_LOSS_PCT = float(os.getenv('MONTH_STOP_LOSS_PCT', 0.07))   # 7%
    MONTH_TAKE_PROFIT_PCT = float(os.getenv('MONTH_TAKE_PROFIT_PCT', 0.15)) # 15%
    MONTH_MIN_CONFIDENCE = float(os.getenv('MONTH_MIN_CONFIDENCE', 60))    # 60%
    
    # Legacy aliases (used by some old code paths)
    STOP_LOSS_PERCENTAGE = DAY_STOP_LOSS_PCT
    TAKE_PROFIT_PERCENTAGE = DAY_TAKE_PROFIT_PCT
    
    # Trading Hours (Eastern Time)
    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    EXTENDED_HOURS_START = "04:00"  # Pre-market
    EXTENDED_HOURS_END = "20:00"    # After-hours
    
    # Data Sources
    USE_YAHOO_FINANCE = True
    USE_ALPHA_VANTAGE = True
    
    # Strategy Parameters
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE_THRESHOLD = 2.0  # Volume spike threshold (2x average)
    
    # Flask Settings
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5001
    FLASK_DEBUG = True
    
    # ── Helpers ─────────────────────────────────────────────────────
    @classmethod
    def get_horizon_params(cls, horizon: str) -> dict:
        """Return stop-loss %, take-profit %, and min confidence for a horizon.
        
        Args:
            horizon: One of 'DAY', 'WEEK', 'MONTH'
        """
        horizon = horizon.upper()
        if horizon == 'WEEK':
            return {
                'stop_loss_pct': cls.WEEK_STOP_LOSS_PCT,
                'take_profit_pct': cls.WEEK_TAKE_PROFIT_PCT,
                'min_confidence': cls.WEEK_MIN_CONFIDENCE,
            }
        elif horizon == 'MONTH':
            return {
                'stop_loss_pct': cls.MONTH_STOP_LOSS_PCT,
                'take_profit_pct': cls.MONTH_TAKE_PROFIT_PCT,
                'min_confidence': cls.MONTH_MIN_CONFIDENCE,
            }
        else:  # DAY (default)
            return {
                'stop_loss_pct': cls.DAY_STOP_LOSS_PCT,
                'take_profit_pct': cls.DAY_TAKE_PROFIT_PCT,
                'min_confidence': cls.DAY_MIN_CONFIDENCE,
            }
