#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from broker_integration import AlpacaBroker

def add_test_position():
    """Add a small test position to Alpaca for debugging"""
    print("=== Adding Test Position ===")
    
    broker = AlpacaBroker()
    
    if not broker.is_connected():
        print("❌ Broker not connected")
        return False
    
    print("✅ Broker connected")
    
    # Check current positions
    positions = broker.get_positions()
    print(f"Current positions: {len(positions)}")
    
    if positions:
        print("✅ You already have positions!")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} shares")
        return True
    
    # Check account status
    account_info = broker.get_account_info()
    buying_power = account_info.get('buying_power', 0)
    print(f"Buying Power: ${buying_power:,.2f}")
    
    if buying_power < 100:
        print("❌ Insufficient buying power")
        return False
    
    # Try to add a small test position (1 share of SPY)
    try:
        print("📈 Adding 1 share of SPY...")
        
        order_result = broker.place_order(
            symbol='SPY',
            quantity=1,
            side='buy',
            order_type='market'
        )
        
        print(f"Order Result: {order_result}")
        
        if order_result.get('success'):
            print("✅ Test position added successfully!")
            print("   You should now see positions in the dashboard")
            return True
        else:
            error_msg = order_result.get('error', 'Unknown error')
            print(f"❌ Failed to add position: {error_msg}")
            
            if "market closed" in error_msg.lower():
                print("   Market is currently closed. Try during market hours.")
            elif "insufficient" in error_msg.lower():
                print("   Insufficient funds or buying power.")
            else:
                print("   This might be due to account restrictions or market conditions.")
            
            return False
            
    except Exception as e:
        print(f"❌ Error adding position: {e}")
        return False

def check_market_status():
    """Check if market is open"""
    print("\n=== Checking Market Status ===")
    
    broker = AlpacaBroker()
    
    if not broker.is_connected():
        print("❌ Broker not connected")
        return False
    
    is_open = broker.is_market_open()
    print(f"Market Open: {'✅' if is_open else '❌'}")
    
    if not is_open:
        print("⚠️  Market is currently closed")
        print("   You can still add positions, but they won't execute until market opens")
    
    return is_open

if __name__ == "__main__":
    print("Adding Test Position for Dashboard Debugging")
    print("=" * 50)
    
    # Check market status
    market_ok = check_market_status()
    
    # Try to add test position
    position_ok = add_test_position()
    
    if position_ok:
        print("\n🎉 Success! Now test the dashboard:")
        print("1. Start the Flask app: python3 app.py")
        print("2. Open http://localhost:5001 in your browser")
        print("3. Check the 'Current Positions' section")
    else:
        print("\n💡 Alternative solutions:")
        print("1. Add positions manually through Alpaca dashboard")
        print("2. Wait for market to open and try again")
        print("3. Check your account permissions and restrictions")