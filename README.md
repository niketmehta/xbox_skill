# 🤖 Day Trading Agent

An automated day trading system that analyzes real-time market data, recommends stocks to buy and sell, and manages positions with strict risk controls. The agent is designed for same-day liquidation to avoid overnight exposure.

## 🌟 Features

### Core Trading Capabilities
- **Market Data**: Alpaca-first market data, with optional Yahoo/Stooq fallback configuration
- **Automatic Stock Picking**: Multi-agent council for Top N recommendations
- **Multi-Strategy Analysis**: Momentum, mean reversion, breakout, and volume-based strategies
- **Automatic Risk Management**: Stop-loss and take-profit mechanisms
- **WhatsApp Digests**: OpenClaw messages for picks, simulated open entries, and EOD P/L
- **Portfolio Management**: Position tracking, P&L calculation, and performance metrics

### Web Dashboard
- **Real-time Monitoring**: Live portfolio and position updates
- **Automatic Stock Screening**: Multiple screening criteria (swing, momentum, breakout, high volume)
- **Smart Watchlist**: Auto-generated and periodically updated watchlists
- **Market Analysis**: Interactive stock analysis with technical indicators
- **Risk Metrics**: Portfolio drawdown, win rate, and profit factor tracking
- **Watchlist Management**: Both automatic and manual stock selection
- **Manual Controls**: Start/stop agent, send digests, force liquidation, and configuration

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
MAX_POSITION_SIZE=5000         # Maximum per position
STOP_LOSS_PERCENTAGE=0.02      # 2% stop loss
TAKE_PROFIT_PERCENTAGE=0.05    # 5% take profit
MAX_DAILY_LOSS=500            # Maximum daily loss
MAX_POSITIONS=5               # Maximum concurrent positions

# Optional: OpenClaw WhatsApp delivery
OPENCLAW_ENABLED=false
OPENCLAW_CLI=openclaw
OPENCLAW_CHANNEL=whatsapp
OPENCLAW_ACCOUNT=
OPENCLAW_WHATSAPP_TARGET=+1234567890
OPENCLAW_TIMEOUT_SECONDS=45
OPENCLAW_HANDSHAKE_TIMEOUT_MS=120000
OPENCLAW_SEND_ATTEMPTS=4
OPENCLAW_AUTO_RESTART=true

# Optional scheduled WhatsApp digest
TOP_RECOMMENDATIONS_ENABLED=false
TOP_RECOMMENDATIONS_TIME=05:35
TOP_RECOMMENDATIONS_HORIZON=WEEK
TOP_RECOMMENDATIONS_UNIVERSE_SIZE=100
TOP_RECOMMENDATIONS_SMART_UNIVERSE=true
TOP_RECOMMENDATIONS_MIN_CONFIDENCE=20
TOP_RECOMMENDATIONS_HISTORY_LOOKBACK_DAYS=30
TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES=2
TOP_RECOMMENDATIONS_MAX_PER_SECTOR=2
COUNCIL_RAG_ENABLED=true
COUNCIL_RAG_LOOKBACK_DAYS=45
COUNCIL_RAG_MAX_LESSONS=6
COUNCIL_RAG_SYMBOL_WEIGHT=1.0
COUNCIL_RAG_SECTOR_WEIGHT=0.45
COUNCIL_RAG_MARKET_WEIGHT=0.20
COUNCIL_RAG_MAX_SCORE_ADJUSTMENT=14
COUNCIL_RAG_MISSED_MOVER_MIN_CHANGE_PCT=3
COUNCIL_RAG_MISSED_MOVER_WEIGHT=0.60
HELD_MOMENTUM_ALERT_LIMIT=3
INTRADAY_BREAKOUT_ALERT_LIMIT=5
INTRADAY_BREAKOUT_MIN_CHANGE_PCT=2.0
INTRADAY_BREAKOUT_MIN_VOLUME_RATIO=1.1
SIMULATION_ENABLED=true
SIMULATION_TOP_N=5
SIMULATION_NOTIONAL_PER_PICK=1000
SIMULATION_OPEN_TIME=06:35
SIMULATION_MIDDAY_TIME=09:00
SIMULATION_EOD_TIME=13:10
SIMULATION_OPEN_WHATSAPP_ENABLED=true
SIMULATION_BACKFILL_ON_SUMMARY=true
```

### OpenClaw Trading Council
- `GET /api/recommendations/top5?horizon=WEEK&limit=5` runs the multi-agent council and returns challenged BUY candidates with buy zone, exit target, stop loss, risk/reward, objections, and a WhatsApp-ready digest.
- `POST /api/recommendations/top5/send-whatsapp` regenerates the council digest and sends it through OpenClaw WhatsApp when `OPENCLAW_ENABLED=true`.
- The council uses momentum, breakout, mean-reversion, volume, fundamentals, relative strength, macro risk, skeptic, and arbiter agents. Picks are saved to `trading_data.db` for audit.
- Scheduled digests use a smart universe by default: current movers, momentum/breakout screens, the smart watchlist, default liquid names, and currently held positions.
- The smart universe now uses an internal broad sector-discovery map plus live movers/screens, so emerging themes such as storage, memory, software, health care, energy, industrials, financials, and consumer names are evaluated without maintaining a user focus list.
- Broad and leveraged ETFs are filtered automatically inside the council, keeping the output focused on individual equities without a user-maintained exclude list.
- Recommendation ranking applies recent simulated open-entry performance feedback, cooling down repeated simulated losers and modestly rewarding symbols whose recent picks have worked.
- The council has a local SQLite RAG memory: EOD simulation results write daily lessons by symbol, sector, and market; the next recommendation run retrieves those lessons and adjusts scoring before ranking.
- EOD learning also scans automatic market movers and records missed-opportunity lessons for strong positive movers that were not recommended, so the council learns from stocks it failed to surface.
- Top picks are sector-balanced by default with `TOP_RECOMMENDATIONS_MAX_PER_SECTOR=2`, so one crowded theme does not consume every recommendation slot.
- Recommendation payloads save the screened universe and a lightweight candidate snapshot for later missed-pick audits.
- Digests include supplemental "held momentum review" and "intraday breakout watch" sections so strong existing positions or fast movers can surface even when the conservative council does not mark them as fresh top BUY picks.
- Scheduled digest/simulation jobs run on weekdays and also check the Alpaca US equities calendar at runtime, so weekends and market holidays are skipped cleanly.
- MIDDAY/EOD simulation summaries backfill open-entry rows from the latest recommendation run when the open-capture task was missed, and include all captured same-day recommendation runs so earlier picks are not hidden.
- Backfill or inspect learning with `py -3 scripts\rebuild_council_memory.py` and `py -3 scripts\analyze_recommendation_history.py`.

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
1. **API Rate Limits**: Reduce scan frequency or use Alpaca-first data if hitting limits
2. **Market Data Delays**: Alpaca IEX and other free feeds may be delayed or partial
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
