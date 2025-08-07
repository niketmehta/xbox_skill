# 🤖 Day Trading Agent

An automated day trading system that analyzes real-time market data, recommends stocks to buy and sell, and manages positions with strict risk controls. The agent is designed for same-day liquidation to avoid overnight exposure.

## 🌟 Features

### Core Trading Capabilities
- **Real-time Market Data**: Integration with Yahoo Finance and Alpha Vantage APIs
- **Automatic Stock Picking**: AI-powered stock screening based on day trading criteria
- **Multi-Strategy Analysis**: Momentum, mean reversion, breakout, and volume-based strategies
- **Extended Hours Monitoring**: Pre-market and after-hours analysis for gap opportunities
- **Automatic Risk Management**: Stop-loss and take-profit mechanisms
- **End-of-Day Liquidation**: Automatic position closing before market close
- **Portfolio Management**: Position tracking, P&L calculation, and performance metrics

### Web Dashboard
- **Real-time Monitoring**: Live portfolio and position updates
- **Automatic Stock Screening**: Multiple screening criteria (day trading, momentum, breakout, high volume)
- **Smart Watchlist**: Auto-generated and periodically updated watchlists
- **Market Analysis**: Interactive stock analysis with technical indicators
- **Risk Metrics**: Portfolio drawdown, win rate, and profit factor tracking
- **Watchlist Management**: Both automatic and manual stock selection
- **Manual Controls**: Start/stop agent, force liquidation, and configuration

### Risk Controls
- **Position Sizing**: Automatic position sizing based on risk tolerance
- **Stop Loss**: 2% default stop loss on all positions
- **Take Profit**: 5% default take profit target
- **Daily Loss Limit**: Maximum daily loss protection
- **Maximum Positions**: Limit on concurrent open positions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Internet connection for market data

### Installation

1. **Clone and Navigate**
   ```bash
   cd /path/to/trading-agent
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (Alpha Vantage key already configured)
   ```

4. **Run the Web Dashboard**
   ```bash
   python app.py
   ```

5. **Access Dashboard**
   Open http://localhost:5000 in your browser

### Command Line Usage

Run the trading agent directly:
```bash
python trading_agent.py
```

## 📊 Configuration

### Environment Variables (.env)

```bash
# Alpha Vantage API (already configured)
ALPHA_VANTAGE_API_KEY=QLKXLIUQUBNYQEJU

# Optional: Alpaca for live trading (paper trading by default)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Trading Parameters
MAX_PORTFOLIO_VALUE=10000      # Maximum portfolio value
MAX_POSITION_SIZE=1000         # Maximum per position
STOP_LOSS_PERCENTAGE=0.02      # 2% stop loss
TAKE_PROFIT_PERCENTAGE=0.05    # 5% take profit
MAX_DAILY_LOSS=500            # Maximum daily loss
MAX_POSITIONS=5               # Maximum concurrent positions
```

### Trading Hours
- **Market Hours**: 9:30 AM - 4:00 PM ET
- **Extended Hours**: 4:00 AM - 8:00 PM ET
- **Auto Liquidation**: 15 minutes before market close (3:45 PM ET)

## 🎯 Trading Strategies

### 1. Momentum Strategy
- Moving average crossovers (SMA 20/50)
- MACD bullish/bearish signals
- Price momentum analysis
- Volume confirmation

### 2. Mean Reversion Strategy
- RSI oversold/overbought levels (30/70)
- Bollinger Bands positioning
- Distance from moving averages
- Support/resistance bounces

### 3. Breakout Strategy
- 20-period high/low breakouts
- Volume confirmation required
- Support/resistance level analysis
- Momentum continuation signals

### 4. Volume Analysis
- Volume spike detection (2x average)
- Price-volume relationship
- Institutional activity indicators
- Liquidity assessment

## 📈 Risk Management

### Position Risk Controls
- **Maximum Loss per Trade**: 2% of position value
- **Position Sizing**: Dynamic based on volatility and confidence
- **Correlation Limits**: Avoid overexposure to similar assets
- **Liquidity Requirements**: Minimum volume thresholds

### Portfolio Risk Controls
- **Daily Loss Limit**: Stop trading if daily loss exceeds limit
- **Drawdown Protection**: Reduce position sizes during drawdowns
- **Concentration Limits**: Maximum positions per sector/asset class
- **Market Condition Adaptation**: Adjust strategies based on volatility

## 🖥️ Web Dashboard

### Main Features
- **Agent Status**: Real-time monitoring of agent state
- **Portfolio Summary**: Cash, positions, P&L, and total value
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio
- **Current Positions**: Live position tracking with P&L
- **Watchlist**: Monitor and analyze specific stocks
- **Market Movers**: Top gainers and losers

### API Endpoints
- `GET /api/status` - Agent and market status
- `GET /api/portfolio` - Portfolio summary
- `GET /api/positions` - Current positions
- `GET /api/watchlist` - Watchlist management
- `POST /api/start-agent` - Start trading agent
- `POST /api/stop-agent` - Stop trading agent
- `POST /api/liquidate` - Force liquidation

## 📁 Project Structure

```
trading-agent/
├── app.py                 # Flask web application
├── trading_agent.py       # Main trading agent
├── data_provider.py       # Market data integration
├── trading_strategy.py    # Trading strategies and analysis
├── portfolio_manager.py   # Portfolio and risk management
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── templates/
│   └── dashboard.html    # Web dashboard template
└── README.md            # This file
```

## 🔧 Customization

### Adding New Strategies
1. Extend `TradingStrategy` class in `trading_strategy.py`
2. Implement strategy logic with entry/exit signals
3. Add to strategy combination in `_combine_signals()`

### Custom Risk Rules
1. Modify `PortfolioManager` class in `portfolio_manager.py`
2. Add new risk checks in `should_liquidate_for_risk()`
3. Implement custom position sizing logic

### Data Sources
1. Add new providers in `MarketDataProvider` class
2. Implement data normalization methods
3. Update configuration for new API keys

## ⚠️ Important Disclaimers

### Risk Warning
- **This is for educational purposes only**
- **Day trading involves significant financial risk**
- **Past performance does not guarantee future results**
- **Only trade with money you can afford to lose**

### Paper Trading Recommended
- Start with paper trading to test strategies
- Validate performance over extended periods
- Understand all risk controls before live trading
- Never trade without proper risk management

### Legal Compliance
- Ensure compliance with local trading regulations
- Understand tax implications of day trading
- Consider professional financial advice
- Review broker terms and conditions

## 📊 Performance Monitoring

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Typical holding period

### Database Tracking
All trades and portfolio snapshots are stored in SQLite database (`trading_data.db`) for analysis and backtesting.

## 🛠️ Troubleshooting

### Common Issues
1. **API Rate Limits**: Reduce scan frequency if hitting limits
2. **Market Data Delays**: Yahoo Finance data may have delays
3. **WebSocket Connections**: Restart agent if connection issues
4. **Database Locks**: Ensure only one agent instance running

### Logging
Check `trading_agent.log` for detailed execution logs and error messages.

## 🔄 Updates and Maintenance

### Regular Tasks
- Monitor API key usage and limits
- Review and adjust risk parameters
- Update watchlists based on market conditions
- Backup trading database regularly

### Performance Reviews
- Weekly strategy performance analysis
- Monthly risk parameter optimization
- Quarterly system maintenance and updates

## 📞 Support

For issues or questions:
1. Check the logs for error messages
2. Verify API keys and network connectivity
3. Review configuration parameters
4. Test with paper trading first

---

**Remember**: This system is a tool to assist with trading decisions. Always maintain oversight and never rely solely on automated systems for financial decisions.