#!/usr/bin/env python3
"""
Simple test for web interface
"""

import requests
import time
import threading
from app import app

def test_web_interface():
    """Test the web interface functionality"""
    print("🌐 Testing Web Interface")
    print("-" * 30)
    
    try:
        # Start Flask app in a thread
        def run_app():
            app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
        
        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()
        
        # Wait for app to start
        time.sleep(3)
        
        # Test basic endpoints
        base_url = "http://127.0.0.1:5001"
        
        # Test homepage
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Homepage loads successfully")
        else:
            print(f"❌ Homepage failed: {response.status_code}")
        
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status API works")
        else:
            print(f"❌ Status API failed: {response.status_code}")
        
        # Test analyze endpoint (this was causing JSON errors)
        response = requests.get(f"{base_url}/api/analyze/AAPL", timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'recommendation' in data:
                print("✅ Analysis API works (JSON serialization fixed)")
            else:
                print("⚠️  Analysis API returns data but no recommendation")
        else:
            print(f"❌ Analysis API failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

if __name__ == "__main__":
    test_web_interface()