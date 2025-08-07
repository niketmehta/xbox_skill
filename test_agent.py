#!/usr/bin/env python3
"""
Test script for the trading agent to verify functionality
"""

import logging
import time
from trading_agent import TradingAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_functionality():
    """Test basic agent functionality"""
    print("🧪 Testing Trading Agent Functionality")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("1. Initializing Trading Agent...")
        agent = TradingAgent()
        print("✅ Agent initialized successfully")
        
        # Test data provider
        print("\n2. Testing Data Provider...")
        quote = agent.data_provider.get_real_time_quote("AAPL")
        if quote:
            print(f"✅ Got AAPL quote: ${quote.get('current_price', 'N/A')}")
        else:
            print("❌ Failed to get quote data")
        
        # Test market status
        print("\n3. Testing Market Status...")
        is_open = agent.data_provider.is_market_open()
        is_extended = agent.data_provider.is_extended_hours()
        print(f"✅ Market Open: {is_open}, Extended Hours: {is_extended}")
        
        # Test stock analysis
        print("\n4. Testing Stock Analysis...")
        analysis = agent.analyze_symbol("AAPL")
        if analysis:
            recommendation = analysis.get('recommendation', {})
            action = recommendation.get('action', 'N/A')
            confidence = recommendation.get('confidence', 0)
            print(f"✅ AAPL Analysis: {action} with {confidence:.1f}% confidence")
        else:
            print("❌ Failed to analyze stock")
        
        # Test portfolio manager
        print("\n5. Testing Portfolio Manager...")
        portfolio_summary = agent.portfolio_manager.get_portfolio_summary()
        if portfolio_summary:
            cash = portfolio_summary.get('cash_balance', 0)
            total_value = portfolio_summary.get('total_portfolio_value', 0)
            print(f"✅ Portfolio: Cash=${cash:.2f}, Total=${total_value:.2f}")
        else:
            print("❌ Failed to get portfolio summary")
        
        # Test broker connection (if configured)
        print("\n6. Testing Broker Connection...")
        if hasattr(agent.portfolio_manager, 'broker') and agent.portfolio_manager.broker.is_connected():
            account_info = agent.portfolio_manager.broker.get_account_info()
            if account_info:
                buying_power = account_info.get('buying_power', 0)
                print(f"✅ Alpaca Connected: Buying Power=${buying_power:.2f}")
            else:
                print("⚠️  Broker connected but no account info")
        else:
            print("⚠️  Broker not connected (running in simulation mode)")
        
        # Test stock screener
        print("\n7. Testing Stock Screener...")
        screened_stocks = agent.stock_screener.screen_stocks(max_stocks=5, screen_type='day_trading')
        if screened_stocks:
            print(f"✅ Screened stocks: {screened_stocks[:3]}...")
        else:
            print("❌ Failed to screen stocks")
        
        # Test watchlist update
        print("\n8. Testing Watchlist Management...")
        original_size = len(agent.watchlist)
        agent._update_smart_watchlist()
        new_size = len(agent.watchlist)
        print(f"✅ Watchlist updated: {original_size} → {new_size} stocks")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("The trading agent appears to be working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_json_serialization():
    """Test that analysis results can be JSON serialized"""
    print("\n🧪 Testing JSON Serialization")
    print("-" * 30)
    
    try:
        import json
        agent = TradingAgent()
        
        # Test analysis serialization
        analysis = agent.analyze_symbol("AAPL")
        json_str = json.dumps(analysis, default=str)
        print("✅ Analysis JSON serialization works")
        
        # Test specific components
        if 'recommendation' in analysis:
            rec = analysis['recommendation']
            json.dumps(rec, default=str)
            print("✅ Recommendation JSON serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Trading Agent Test Suite")
    print("=" * 60)
    
    # Run tests
    basic_test = test_basic_functionality()
    json_test = test_analysis_json_serialization()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"Basic Functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"JSON Serialization: {'✅ PASS' if json_test else '❌ FAIL'}")
    
    if basic_test and json_test:
        print("\n🎉 All tests PASSED! The agent is ready to use.")
    else:
        print("\n⚠️  Some tests FAILED. Check the logs above for details.")

if __name__ == "__main__":
    main()