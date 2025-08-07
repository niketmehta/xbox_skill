# Troubleshooting Guide

## 🔧 Common Issues and Solutions

### 1. "Error analyzing symbol: Object of type int64 is not JSON serializable"
**Fixed** ✅ - Updated `app.py` with numpy type conversion in the analyze endpoint.

### 2. "ModuleNotFoundError: No module named 'schedule'"
**Solution**: Install requirements
```bash
pip3 install --break-system-packages -r requirements.txt
```

### 3. "No data found, symbol may be delisted"
**This is normal** - Some symbols like ZOOM, SQ may be delisted or renamed. The agent handles this gracefully.

### 4. "externally-managed-environment" pip error
**Solution**: Use the --break-system-packages flag or create a virtual environment
```bash
pip3 install --break-system-packages -r requirements.txt
```

### 5. Agent not finding good trading opportunities
**Possible causes**:
- Market is closed or low volatility period
- Screening criteria too strict
- Limited API data access

**Solutions**:
- Check if market is open
- Try different screening types (momentum, breakout)
- Verify API keys are working

### 6. Broker connection issues
**Check**:
- Alpaca API keys are correct in `.env`
- Network connectivity
- Alpaca service status

### 7. Analysis shows low confidence
**This is normal** - The agent is conservative and only trades high-confidence signals (>70%).

## 🚀 Quick Start Checklist

1. ✅ Install packages: `pip3 install --break-system-packages -r requirements.txt`
2. ✅ Configure `.env` file with API keys
3. ✅ Run test: `python3 test_agent.py`
4. ✅ Start dashboard: `python3 app.py`
5. ✅ Open browser: http://localhost:5000

## 🔒 Safety Notes

- **Paper Trading**: System defaults to paper trading (safe)
- **Risk Controls**: Multiple safety mechanisms active
- **Auto Liquidation**: All positions closed at 3:45 PM ET
- **Conservative**: Only high-confidence trades (>70%)

The system is designed to be safe and conservative by default!