# Hybrid Grid + Trend Following Trading Bot

**Small Account Edition for IC Markets Australia**

A comprehensive Python trading bot designed for small accounts ($1,000+) with strict risk management, supporting both grid trading in ranging markets and trend following for directional moves.

## Features

### Core Strategies

- **Grid Trading Module**: Automated buy/sell pairs in ranging markets
  - Configurable grid levels (10-25 recommended)
  - Auto-calculated grid spacing and position sizes
  - Automatic profit taking at each grid level
  - Break-out detection with position closure

- **Trend Following Module**: Capture directional moves
  - SMA crossover signals (9/20 period default)
  - Higher highs/lows pattern detection
  - SMA rejection/support signals
  - Signal strength calculation

### Risk Management (Non-Negotiable)

- **Position Sizing**: 1% risk per trade rule
- **Daily Loss Limit**: $50 maximum (5% of $1,000)
- **Drawdown Protection**: Trading paused at 10% drawdown
- **Stop-Loss Enforcement**: Every order must have SL/TP
- **Margin Validation**: Pre-trade margin checks with buffer
- **Spread Monitoring**: Trade rejection on wide spreads

### Monitoring & Debugging

- **Comprehensive Logging**: DEBUG, INFO, WARNING, ERROR, CRITICAL levels
- **Real-time Dashboard**: CLI-based monitoring with Rich library
- **Trade History**: Full audit trail of all trades
- **Performance Metrics**: Win rate, profit factor, drawdown tracking

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 (developed and tested)

### Setup

1. Clone or download the repository:
```bash
cd C:\Users\Comp\GoldEA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.json` to customize:

```json
{
  "trading": {
    "pairs": ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD"],
    "min_lot_size": 0.01,
    "max_lot_size": 0.10
  },
  "risk_management": {
    "account_balance": 1000,
    "risk_percent_per_trade": 1.0,
    "daily_loss_limit": 50,
    "max_drawdown_percent": 10
  },
  "grid_strategy": {
    "enabled": true,
    "grid_lines": 15,
    "profit_per_grid_pips": 15
  },
  "trend_strategy": {
    "enabled": true,
    "sma_period_short": 9,
    "sma_period_long": 20,
    "take_profit_pips": 75,
    "stop_loss_pips": 50
  },
  "dry_run": {
    "enabled": true
  }
}
```

## Usage

### Starting the Bot

```bash
# Run in dry-run mode (default, recommended for testing)
python -m src.bot

# Or use the main script
python main.py
```

### Interactive Commands (Example Session)

```python
import asyncio
from src.bot import TradingBot

async def main():
    # Create bot in dry-run mode
    bot = TradingBot(dry_run=True)
    await bot.start()
    
    # Setup a grid for EUR/USD
    await bot.setup_grid(
        pair="EUR/USD",
        upper_limit=1.0900,
        lower_limit=1.0800,
        num_grids=10
    )
    
    # Analyze and get signals
    analysis = await bot.analyze_pair("EUR/USD")
    
    # Execute signals
    for signal in analysis['signals']:
        await bot.execute_signal(signal)
    
    # Check status
    status = bot.get_status()
    print(f"Balance: ${status['account']['balance']:.2f}")
    
    # Stop bot
    await bot.stop()

asyncio.run(main())
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_risk_management.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
GoldEA/
├── config/
│   └── settings.json       # Main configuration file
├── src/
│   ├── __init__.py
│   ├── api_client.py       # Broker API client (dry-run included)
│   ├── bot.py              # Main bot orchestrator
│   ├── config.py           # Configuration management
│   ├── dashboard.py        # CLI monitoring dashboard
│   ├── grid_trader.py      # Grid trading strategy
│   ├── logger.py           # Logging framework
│   ├── models.py           # Data models
│   ├── risk_manager.py     # Risk management module
│   └── trend_follower.py   # Trend following strategy
├── tests/
│   ├── __init__.py
│   ├── test_entry_signals.py
│   ├── test_position_sizing.py
│   └── test_risk_management.py
├── logs/                   # Log files (created on first run)
├── main.py                 # Entry point script
├── requirements.txt
└── README.md
```

## Risk Management Details

### Position Sizing Formula

```
Lot Size = Risk Amount / (Stop Loss Pips × Pip Value)

Example:
- Account: $1,000
- Risk: 1% = $10
- Stop Loss: 30 pips
- EUR/USD pip value: $10 per lot
- Lot Size = $10 / (30 × $10) = 0.033 → 0.03 lots
```

### Daily Loss Limit Enforcement

- Trading halted when daily P&L reaches -$50
- Warning triggered at -$35 (70% of limit)
- Automatic reset at 00:00 UTC

### Leverage Constraint

- IC Markets Australia max leverage: 1:30
- All margin calculations account for this limit

## Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Development, detailed calculations |
| INFO | Normal operation, trade execution |
| WARNING | Approaching limits, high spreads |
| ERROR | Failed operations, rejected orders |
| CRITICAL | Emergency stops, connection lost |

## Dashboard

The real-time CLI dashboard shows:

```
═══════════════════════════════════════════════════════════
          TRADING BOT LIVE MONITOR
═══════════════════════════════════════════════════════════
Account Status:
  Balance: $1,008.75
  Equity: $1,008.75
  Margin Used: $75.00
  Free Margin: $425.00
  Drawdown Today: 0.5%

Current Market:
  Pair: EUR/USD
  Bid/Ask: 1.08101/1.08103
  Spread: 2.0 pips

Grid Status:
  Active Grids: EUR/USD (10 levels)
  
Trend Status:
  SMA9: 1.08420
  SMA20: 1.08390
  Direction: BULLISH
```

## Deployment Recommendations

1. **Test in dry-run mode** for 1-2 weeks before live trading
2. **Start with minimum lot size** (0.01) until confident
3. **Monitor logs daily** for WARNING/ERROR messages
4. **Review dashboard weekly** for performance assessment
5. **Never remove stop-loss** from any trade
6. **Pause trading** if drawdown exceeds 10%
7. **Backup logs** weekly for analysis

## Supported Currency Pairs

Major pairs only (for lower spreads and better liquidity):

- EUR/USD
- USD/JPY
- GBP/USD
- AUD/USD
- USD/CAD

## API Integration

Currently supports dry-run mode with simulated execution. For live trading, implement the `BrokerAPIClient` interface for your broker (IC Markets cTrader, MT4/MT5, etc.).

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation as needed
4. Use type hints throughout

## License

Private use only. Not licensed for redistribution.

## Disclaimer

**IMPORTANT**: Trading forex involves significant risk of loss. This bot is provided for educational purposes. Past performance does not guarantee future results. Always use proper risk management and never trade with money you cannot afford to lose.

---

**Version**: 1.0.0  
**Created**: 2026-01-30  
**Author**: Trading Bot Developer
