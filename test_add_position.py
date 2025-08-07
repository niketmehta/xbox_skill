#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append('.')

from broker_integration import AlpacaBroker

def test_add_position():
    """Test adding a small position to Alpaca for debugging"""
    print("=== Testing Position Addition ===")
    
    broker = AlpacaBroker()
    
    if not broker.is_connected():
        print("❌ Broker not connected")
        return False
    
    print("✅ Broker connected")
    
    # Check current positions
    positions = broker.get_positions()
    print(f"Current positions: {positions}")
    
    # Try to add a small test position (1 share of SPY)
    try:
        # First check if we can get asset info
        asset_info = broker.get_asset_info('SPY')
        print(f"SPY Asset Info: {asset_info}")
        
        # Try to place a small market order
        order_result = broker.place_order(
            symbol='SPY',
            quantity=1,
            side='buy',
            order_type='market'
        )
        
        print(f"Order Result: {order_result}")
        
        if order_result.get('success'):
            print("✅ Test position added successfully")
            return True
        else:
            print(f"❌ Failed to add position: {order_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error adding position: {e}")
        return False

def check_account_status():
    """Check account status and buying power"""
    print("\n=== Checking Account Status ===")
    
    broker = AlpacaBroker()
    
    if not broker.is_connected():
        print("❌ Broker not connected")
        return False
    
    account_info = broker.get_account_info()
    print(f"Account Info: {account_info}")
    
    buying_power = account_info.get('buying_power', 0)
    cash = account_info.get('cash', 0)
    
    print(f"Buying Power: ${buying_power:,.2f}")
    print(f"Cash: ${cash:,.2f}")
    
    if buying_power > 100:
        print("✅ Sufficient buying power for test position")
        return True
    else:
        print("❌ Insufficient buying power")
        return False

if __name__ == "__main__":
    print("Testing Position Addition for Debugging")
    print("=" * 50)
    
    # Check account status first
    account_ok = check_account_status()
    
    if account_ok:
        # Try to add a test position
        position_ok = test_add_position()
        
        if position_ok:
            print("\n✅ Test position added successfully!")
            print("   You should now see positions in the dashboard")
        else:
            print("\n❌ Failed to add test position")
            print("   This might be due to:")
            print("   - Market being closed")
            print("   - Insufficient funds")
            print("   - Account restrictions")
    else:
        print("\n❌ Cannot add test position due to account issues")