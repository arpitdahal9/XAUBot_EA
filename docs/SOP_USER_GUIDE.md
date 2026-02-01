# Standard Operating Procedure (SOP)
# Trading Bot User Guide

**Version:** 2.0  
**Last Updated:** February 2026  
**Author:** Trading Bot Developer

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [How to Change the Trading Asset](#2-how-to-change-the-trading-asset)
3. [How to Backtest with Custom Dates](#3-how-to-backtest-with-custom-dates)
4. [How to Run the Bot (Dry Run Mode)](#4-how-to-run-the-bot-dry-run-mode)
5. [How to Connect to Real Broker](#5-how-to-connect-to-real-broker)
6. [Configuration Reference](#6-configuration-reference)
7. [Enhanced TPU Strategy (NEW)](#7-enhanced-tpu-strategy-new)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick Start

### Prerequisites
- Windows 10/11
- Python 3.10+
- MetaTrader 5 (for real data)

### First Time Setup
```powershell
cd C:\Users\Comp\GoldEA
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Activate Environment (Every Time)
```powershell
cd C:\Users\Comp\GoldEA
.\venv\Scripts\activate
```

---

## 2. How to Change the Trading Asset

### Method 1: Edit Configuration File (Permanent)

**File:** `config\settings.json`

```json
{
  "trading": {
    "pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
  }
}
```

**Change to your preferred pairs:**
```json
{
  "trading": {
    "pairs": ["GBP/USD"]
  }
}
```

**Supported Pairs:**
| Pair | Description |
|------|-------------|
| EUR/USD | Euro / US Dollar |
| GBP/USD | British Pound / US Dollar |
| USD/JPY | US Dollar / Japanese Yen |
| AUD/USD | Australian Dollar / US Dollar |
| USD/CAD | US Dollar / Canadian Dollar |

---

### Method 2: Edit Test Script (Temporary)

**File:** `test_my_strategy.py`

Find this line at the top:
```python
PAIR = "EUR/USD"
```

Change to:
```python
PAIR = "GBP/USD"
```

Then run:
```powershell
python test_my_strategy.py
```

---

### Method 3: Command Line (One-Time Test)

```powershell
python -c "
from src.backtester import Backtester
bt = Backtester(initial_balance=1000)
bt.load_data_from_mt5('GBP/USD', timeframe='1h', bars=5000)
result = bt.run('GBP/USD', strategy='trend')
print(result.summary())
"
```

**Replace `GBP/USD` with your desired pair.**

---

## 3. How to Backtest with Custom Dates

### Method 1: Using the Test Script

**File:** `test_my_strategy.py`

Change these settings:
```python
# Currency pair
PAIR = "EUR/USD"

# Timeframe options: "15m", "1h", "4h", "1d"
TIMEFRAME = "1h"

# Number of candles (more = longer history)
CANDLES = 10000
```

**Candle Count Reference:**
| Timeframe | Candles | Approximate Period |
|-----------|---------|-------------------|
| 15m | 50,000 | ~1 year |
| 1h | 10,000 | ~1 year |
| 1h | 50,000 | ~5 years |
| 4h | 10,000 | ~4 years |
| 1d | 2,000 | ~8 years |

---

### Method 2: Specific Date Range (Advanced)

Create a new file `my_custom_backtest.py`:

```python
from datetime import datetime
from src.backtester import Backtester

# === EDIT THESE SETTINGS ===
PAIR = "EUR/USD"
START_DATE = datetime(2023, 1, 1)   # Year, Month, Day
END_DATE = datetime(2024, 12, 31)   # Year, Month, Day
TIMEFRAME = "1h"
STARTING_BALANCE = 1000
# ===========================

bt = Backtester(initial_balance=STARTING_BALANCE)

print(f"Loading {PAIR} data from {START_DATE.date()} to {END_DATE.date()}...")

import MetaTrader5 as mt5
mt5.initialize()
mt5.symbol_select(PAIR.replace("/", ""), True)

import time
time.sleep(1)

rates = mt5.copy_rates_range(
    PAIR.replace("/", ""),
    mt5.TIMEFRAME_H1 if TIMEFRAME == "1h" else mt5.TIMEFRAME_M15,
    START_DATE,
    END_DATE
)
mt5.shutdown()

if rates is not None and len(rates) > 0:
    print(f"Loaded {len(rates)} candles")
    
    # Manually add candles to backtester
    from src.models import Candle
    candles = []
    for rate in rates:
        candles.append(Candle(
            pair=PAIR,
            timeframe=TIMEFRAME,
            timestamp=datetime.fromtimestamp(rate['time']),
            open=rate['open'],
            high=rate['high'],
            low=rate['low'],
            close=rate['close'],
            volume=rate['tick_volume']
        ))
    
    bt._candles[PAIR] = candles
    
    print("Running backtest...")
    result = bt.run(PAIR, strategy='trend')
    print(result.summary())
else:
    print("ERROR: No data available for this date range")
    print("Make sure MT5 is running and the chart is open")
```

Run it:
```powershell
python my_custom_backtest.py
```

---

### Method 3: Quick Date Range Command

```powershell
python -c "
from datetime import datetime
from src.backtester import Backtester
import MetaTrader5 as mt5
from src.models import Candle

# SETTINGS - EDIT THESE
PAIR = 'EUR/USD'
START = datetime(2024, 1, 1)
END = datetime(2024, 12, 31)

mt5.initialize()
mt5.symbol_select(PAIR.replace('/', ''), True)
import time; time.sleep(1)

rates = mt5.copy_rates_range(PAIR.replace('/', ''), mt5.TIMEFRAME_H1, START, END)
mt5.shutdown()

if rates is not None:
    print(f'Loaded {len(rates)} candles from {START.date()} to {END.date()}')
    bt = Backtester(1000)
    candles = [Candle(PAIR, '1h', datetime.fromtimestamp(r['time']), r['open'], r['high'], r['low'], r['close'], r['tick_volume']) for r in rates]
    bt._candles[PAIR] = candles
    result = bt.run(PAIR, 'trend')
    print(result.summary())
else:
    print('No data - open chart in MT5 first')
"
```

---

## 4. How to Run the Bot (Dry Run Mode)

Dry run mode simulates trading without real money.

```powershell
python main.py
```

**Available Commands:**
| Command | Description |
|---------|-------------|
| `status` | Show account balance and equity |
| `analyze` | Find trading signals for all pairs |
| `grid` | Set up a new grid trading range |
| `positions` | Show open positions |
| `close` | Close a position |
| `quit` | Exit the bot |

---

## 5. How to Connect to Real Broker

### Step 1: Install MT5 Package
```powershell
pip install MetaTrader5
```

### Step 2: Open MT5 and Login
1. Download MetaTrader 5 from IC Markets
2. Install and open MT5
3. Login to your account

### Step 3: Edit Configuration
**File:** `config\settings.json`

Change:
```json
{
  "dry_run": {
    "enabled": false
  }
}
```

### Step 4: Run with Real Connection
```python
from src.mt5_client import MT5Client
from src.bot import TradingBot

# Connect to MT5
mt5_client = MT5Client()
await mt5_client.connect()

# Create bot with real client
bot = TradingBot(dry_run=False)
bot.api_client = mt5_client

await bot.start()
```

**⚠️ WARNING:** Only do this after extensive backtesting!

---

## 6. Configuration Reference

### Main Settings File: `config\settings.json`

#### Basic Trading Settings
| Setting | Location | Options |
|---------|----------|---------|
| Trading pairs | `trading.pairs` | ["EUR/USD", "GBP/USD", etc.] |
| Lot size range | `trading.min_lot_size`, `trading.max_lot_size` | 0.01 - 0.10 |
| SMA periods | `trend_strategy.sma_period_short/long` | e.g., 9, 20 |
| Stop loss | `trend_strategy.stop_loss_pips` | e.g., 50 |
| Take profit | `trend_strategy.take_profit_pips` | e.g., 75 |
| Grid lines | `grid_strategy.grid_lines` | 10 - 25 |
| Dry run mode | `dry_run.enabled` | true / false |

---

## 6.1 Risk Management System (Advanced)

The bot uses a professional-grade risk management system:

### Position Sizing (Section 1)
```json
"risk_percent_per_trade": 0.75,    // Default: 0.75% of equity
"risk_percent_min": 0.5,           // Minimum allowed
"risk_percent_max": 1.0,           // Maximum allowed
"margin_safety_threshold": 30      // Skip trade if margin > 30% of free margin
```

**Formula:**
```
LotSize = (Equity × Risk%) / (StopDistance × PipValue)
```

### Partial Exit System (Section 2) - Model A: Balanced Asymmetry
```json
"partial_exits": {
  "enabled": true,
  "tp1_r_multiple": 2.0,      // Take Profit 1 at +2R
  "tp1_close_percent": 50,    // Close 50% at TP1
  "tp2_r_multiple": 3.0,      // Take Profit 2 at +3R
  "tp2_close_percent": 30,    // Close 30% at TP2
  "runner_percent": 20        // 20% trails for big moves
}
```

| Target | R-Multiple | Action |
|--------|------------|--------|
| TP1 | +2R | Close 50%, move SL to +0.2R |
| TP2 | +3R | Close 30% |
| Runner | Trailing | Let 20% run for 6R-10R moves |

### Stop Loss & Break-Even (Section 3)
```json
"initial_stop_r": 1.0,         // Initial SL at -1R
"breakeven_trigger_r": 1.2,    // At +1.2R, move SL to breakeven
"breakeven_offset_r": 0.1,     // SL moves to -0.1R (small buffer)
"post_tp1_stop_r": 0.2         // After TP1, SL locks +0.2R profit
```

### ATR Trailing for Runner (Section 4)
```json
"trailing": {
  "enabled": true,
  "atr_multiplier_forex": 1.8,    // k=1.8 for forex
  "atr_multiplier_gold": 2.2,     // k=2.2 for XAU/USD
  "atr_period": 14
}
```

**Formula:**
```
TrailingStop = HighestCloseSinceTP1 - (k × ATR)
```

### Portfolio Limits (Section 5)
```json
"max_total_risk_percent": 2.0,    // Max 2% total open risk
"max_concurrent_trades": 2,        // Max 2 trades at once
"avoid_correlated_trades": true    // Don't open EUR/USD + GBP/USD same direction
```

### Daily/Weekly Drawdown Protection (Section 6)
```json
"daily_loss_limit_r": 2.0,    // Stop trading after -2R daily
"weekly_loss_limit_r": 6.0    // Stop trading after -6R weekly
```

### Spread & Volatility Filters (Section 7)
```json
"spread_filter": {
  "enabled": true,
  "max_spread_multiplier": 1.8    // Skip if spread > 1.8x average
},
"volatility_filter": {
  "enabled": true,
  "atr_spike_multiplier": 2.5     // Skip if ATR > 2.5x median
}
```

---

## 8. Troubleshooting

### Problem: "No data returned for symbol"

**Solution:**
1. Open MetaTrader 5
2. Press Ctrl+M to open Market Watch
3. Right-click → Show All
4. Double-click EURUSD to open chart
5. Wait 30 seconds
6. Try again

---

### Problem: "MT5 initialization failed"

**Solution:**
1. Make sure MT5 is installed
2. Make sure MT5 is running
3. Make sure you are logged in
4. Try running as Administrator

---

### Problem: "Module not found"

**Solution:**
```powershell
cd C:\Users\Comp\GoldEA
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

### Problem: Unicode/Encoding errors

**Solution:**
```powershell
$env:PYTHONIOENCODING='utf-8'
python your_script.py
```

---

## 7. Enhanced TPU Strategy (NEW)

The bot now includes an **Enhanced Entry System** based on Arpit Dahal's TPU Masterclass Strategy. This dramatically reduces false signals by requiring multiple confirmations before entering a trade.

### 5-Step Entry Confirmation Process

1. **Top-Down Analysis (TDA)**
   - Analyzes 4 timeframes: Weekly → Daily → H4 → H1
   - Requires 3+ timeframes to agree on direction
   - PERFECT alignment (4/4) = 95% confidence
   - GOOD alignment (3/4) = 80% confidence
   - WEAK alignment (<3/4) = NO TRADE

2. **Fibonacci Entry (88.6%)**
   - Calculates swing high/low on recent candles
   - Waits for price to retrace to 88.6% level
   - Entry within 20 pips of the level

3. **Divergence Confirmation**
   - Bullish: Price makes lower low, RSI makes higher low
   - Bearish: Price makes higher high, RSI makes lower high
   - Reduces false signals by ~50%

4. **Pattern Recognition**
   - Double Bottom → Bullish reversal
   - Double Top → Bearish reversal
   - Head & Shoulders → Major reversal

5. **Precise Execution**
   - Stop Loss at Fibonacci 100% level (1R)
   - TP1 at +2R (close 50%)
   - TP2 at +3R (close 30%)
   - Runner trails remaining 20%

### Configuration

Edit `config/settings.json`:

```json
{
  "tda": {
    "enabled": true,
    "min_alignment_quality": "GOOD"
  },
  "fibonacci": {
    "enabled": true,
    "entry_at_level": 0.886,
    "proximity_tolerance_pips": 20
  },
  "divergence": {
    "enabled": true,
    "rsi_period": 14,
    "require_confirmation": false
  },
  "patterns": {
    "enabled": true,
    "require_pattern": false
  }
}
```

### Expected Improvements

| Metric | Before (SMA Only) | After (TPU Strategy) |
|--------|-------------------|----------------------|
| Win Rate | ~45% | 55%+ |
| Risk:Reward | 1.0-1.5 | 2.0+ |
| False Signals | 55% | ~35% |
| Selectivity | 5 trades/day | 1-2 trades/day |

### Important Notes

1. **Multi-Timeframe Data Required**: For best results, the enhanced strategy needs real multi-timeframe data (W/D/H4/H1). In backtests using only H1 data, it will fall back to SMA crossovers.

2. **Quality Over Quantity**: The TPU strategy is more selective - fewer trades but higher quality setups.

3. **Fallback Behavior**: If enhanced analysis can't find a valid setup, the system falls back to SMA crossover signals.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                 TRADING BOT QUICK REFERENCE             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ACTIVATE ENVIRONMENT:                                  │
│    cd C:\Users\Comp\GoldEA                              │
│    .\venv\Scripts\activate                              │
│                                                         │
│  RUN BACKTEST:                                          │
│    python test_my_strategy.py                           │
│                                                         │
│  RUN BOT (DRY RUN):                                     │
│    python main.py                                       │
│                                                         │
│  CHANGE PAIR:                                           │
│    Edit: test_my_strategy.py                            │
│    Line: PAIR = "GBP/USD"                               │
│                                                         │
│  CHANGE TIMEFRAME:                                      │
│    Edit: test_my_strategy.py                            │
│    Line: TIMEFRAME = "4h"                               │
│    Options: "15m", "1h", "4h", "1d"                     │
│                                                         │
│  VIEW RESULTS:                                          │
│    Open: backtest_trades.csv (in Excel)                 │
│                                                         │
│  GOOD RESULTS TO LOOK FOR:                              │
│    Profit Factor > 1.5                                  │
│    Win Rate > 50%                                       │
│    Max Drawdown < 20%                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Support

If you encounter issues:
1. Check the `logs\trading_bot.log` file
2. Make sure MT5 is running with the chart open
3. Verify your virtual environment is activated

---

*End of SOP Document*
