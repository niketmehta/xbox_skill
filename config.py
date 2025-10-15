import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys (set these in .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Paper trading by default
    
    # Trading Parameters
    MAX_PORTFOLIO_VALUE = float(os.getenv('MAX_PORTFOLIO_VALUE', 10000))  # Maximum portfolio value
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 1000))       # Maximum per position
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.02)) # 2% stop loss
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.05)) # 5% take profit
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 500))             # Maximum daily loss
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 5))                   # Maximum concurrent positions
    
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
    FLASK_DEBUG = False