"""
XAUUSD Backtest Script
Tests the trading bot on Gold (XAU/USD)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

print("=" * 60)
print("XAUUSD BACKTEST (5+ Years)")
print("=" * 60)

from src.backtester import Backtester
from src.config import get_config

config = get_config()

# Create backtester with gold-specific spread
backtester = Backtester(
    initial_balance=config.risk_management.account_balance,
    spread_pips=3.0,  # Gold typically has wider spread (30 cents)
    slippage_pips=1.0
)

print(f"\nBacktest Settings:")
print(f"  Initial Balance: {config.risk_management.account_balance} USD")
print(f"  Risk per Trade: {config.risk_management.risk_percent_per_trade}%")
print(f"  Strategy: TPU Enhanced (TDA + Fib + Divergence)")
print(f"  Exit Mode: Model A (Partial Exits)")
print(f"    TP1: +2R closes 50%")
print(f"    TP2: +3R closes 30%")
print(f"    Runner: 20% with ATR trailing")

# Load data from MT5
print("\nLoading XAUUSD data from MT5...")
symbol = "XAUUSD"

try:
    candle_count = backtester.load_data_from_mt5(
        pair=symbol,
        timeframe="1h",
        start_date=datetime(2020, 1, 1),
        end_date=datetime.now()
    )
    print(f"Loaded {candle_count} H1 candles")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Trying XAUUSDm symbol...")
    symbol = "XAUUSDm"
    try:
        candle_count = backtester.load_data_from_mt5(
            pair=symbol,
            timeframe="1h",
            start_date=datetime(2020, 1, 1),
            end_date=datetime.now()
        )
        print(f"Loaded {candle_count} H1 candles")
    except Exception as e2:
        print(f"Failed to load XAUUSD data: {e2}")
        sys.exit(1)

# Run backtest
print("\nRunning backtest (this may take a few minutes)...")
result = backtester.run(pair=symbol, strategy="trend")

# Print results
print("\n" + result.summary())

# Export trades
csv_path = "backtest_xauusd_trades.csv"
backtester.export_trades_csv(csv_path)
print(f"\nTrades exported to: {csv_path}")
