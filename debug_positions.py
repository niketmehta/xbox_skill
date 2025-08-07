#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from broker_integration import AlpacaBroker
from portfolio_manager import PortfolioManager
from data_provider import MarketDataProvider

def test_broker_connection():
    """Test Alpaca broker connection"""
    print("=== Testing Alpaca Broker Connection ===")
    
    broker = AlpacaBroker()
    
    # Check if connected
    if broker.is_connected():
        print("✅ Broker connected successfully")
        
        # Get account info
        account_info = broker.get_account_info()
        print(f"Account Info: {account_info}")
        
        # Get positions
        positions = broker.get_positions()
        print(f"Broker Positions: {positions}")
        
        if not positions:
            print("⚠️  No positions found in Alpaca account")
            print("   This is why the dashboard shows empty positions")
            print("   You need to have open positions in your Alpaca account")
        
        return True
    else:
        print("❌ Broker not connected")
        print("Please check your ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        return False

def test_portfolio_manager():
    """Test portfolio manager position syncing"""
    print("\n=== Testing Portfolio Manager ===")
    
    data_provider = MarketDataProvider()
    portfolio_manager = PortfolioManager(data_provider)
    
    # Check if broker is connected
    if portfolio_manager.broker.is_connected():
        print("✅ Portfolio manager broker connected")
        
        # Sync with broker
        try:
            portfolio_manager._sync_with_broker()
            print("✅ Broker sync completed")
        except Exception as e:
            print(f"❌ Error during broker sync: {e}")
        
        # Check positions
        print(f"Portfolio Manager Positions: {len(portfolio_manager.positions)}")
        for symbol, position in portfolio_manager.positions.items():
            print(f"  {symbol}: {position.to_dict()}")
        
        return True
    else:
        print("❌ Portfolio manager broker not connected")
        return False

def test_api_endpoint():
    """Test the API endpoint directly"""
    print("\n=== Testing API Endpoint ===")
    
    try:
        # Import app components
        from app import trading_agent
        
        if trading_agent:
            print("✅ Trading agent exists")
            
            # Check portfolio manager positions
            positions = []
            for symbol, position in trading_agent.portfolio_manager.positions.items():
                positions.append(position.to_dict())
            
            print(f"API Positions: {positions}")
            return True
        else:
            print("❌ Trading agent not initialized")
            print("   The Flask app needs to initialize the trading agent")
            return False
    except Exception as e:
        print(f"❌ Error testing API endpoint: {e}")
        return False

def test_flask_app():
    """Test if Flask app is running and accessible"""
    print("\n=== Testing Flask App ===")
    
    import requests
    try:
        response = requests.get('http://localhost:5001/api/positions', timeout=5)
        print(f"Flask Response Status: {response.status_code}")
        print(f"Flask Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Flask app not running on localhost:5001")
        print("   Start the app with: python3 app.py")
        return False
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")
        return False

if __name__ == "__main__":
    print("Debugging Position Issues")
    print("=" * 50)
    
    # Test broker connection
    broker_ok = test_broker_connection()
    
    # Test portfolio manager
    portfolio_ok = test_portfolio_manager()
    
    # Test API endpoint
    api_ok = test_api_endpoint()
    
    # Test Flask app
    flask_ok = test_flask_app()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Broker Connection: {'✅' if broker_ok else '❌'}")
    print(f"Portfolio Manager: {'✅' if portfolio_ok else '❌'}")
    print(f"API Endpoint: {'✅' if api_ok else '❌'}")
    print(f"Flask App: {'✅' if flask_ok else '❌'}")
    
    print("\n🔧 Solutions:")
    if not broker_ok:
        print("1. Create a .env file with your Alpaca API keys:")
        print("   ALPACA_API_KEY=your_api_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_key_here")
    
    if broker_ok and not portfolio_ok:
        print("2. Check if you have positions in your Alpaca account")
        print("   The dashboard shows empty because there are no positions")
    
    if not flask_ok:
        print("3. Start the Flask app:")
        print("   python3 app.py")
    
    if broker_ok and not api_ok:
        print("4. The trading agent needs to be initialized in the Flask app")