#!/usr/bin/env python3
"""
Trading Agent Startup Script
This script helps initialize and run the trading agent with proper error handling
"""

import os
import sys
import logging
import time
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_agent.log'),
            logging.StreamHandler()
        ]
    )

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'requests', 'python-dotenv',
        'schedule', 'matplotlib', 'seaborn', 'flask', 'flask-cors',
        'ta', 'scikit-learn', 'alpaca-trade-api'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def check_config():
    """Check if configuration is properly set up"""
    print("🔍 Checking configuration...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found, creating from template...")
        # Copy from .env.example if it exists
        example_file = Path('.env.example')
        if example_file.exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from template")
        else:
            print("❌ No .env.example file found")
            return False
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['ALPHA_VANTAGE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
    
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if alpaca_key and alpaca_secret:
        print("✅ Alpaca credentials configured")
    else:
        print("⚠️  Alpaca credentials not configured (will run in simulation mode)")
    
    print("✅ Configuration check completed")
    return True

def run_tests():
    """Run basic functionality tests"""
    print("🧪 Running functionality tests...")
    
    try:
        # Import and run test
        from test_agent import test_basic_functionality, test_analysis_json_serialization
        
        basic_test = test_basic_functionality()
        json_test = test_analysis_json_serialization()
        
        if basic_test and json_test:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_web_interface():
    """Start the web interface"""
    print("🌐 Starting web interface...")
    
    try:
        from app import app
        print("✅ Web interface starting at http://localhost:5000")
        print("📊 Open your browser to access the dashboard")
        print("🛑 Press Ctrl+C to stop")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped by user")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")

def start_command_line():
    """Start the agent in command line mode"""
    print("💻 Starting agent in command line mode...")
    
    try:
        from trading_agent import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")

def main():
    """Main startup function"""
    print("🤖 Trading Agent Startup")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check configuration
    if not check_config():
        print("⚠️  Configuration issues detected, but continuing...")
    
    # Run tests
    print("\n" + "=" * 50)
    if not run_tests():
        print("⚠️  Some tests failed, but you can still try to run the agent")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Choose mode
    print("\n" + "=" * 50)
    print("🚀 Choose how to run the trading agent:")
    print("1. Web Interface (Recommended)")
    print("2. Command Line")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            start_web_interface()
            break
        elif choice == '2':
            start_command_line()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()