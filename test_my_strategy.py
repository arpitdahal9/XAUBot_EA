"""
TEST YOUR STRATEGY
==================
Edit the settings below and run this script to test different configurations.

Usage:
    python test_my_strategy.py
"""

from src.backtester import Backtester

# ============================================
# EDIT THESE SETTINGS TO TEST DIFFERENT VALUES
# ============================================

# Your starting capital
INITIAL_BALANCE = 1000

# Currency pair to test
PAIR = "EUR/USD"

# Timeframe: "15m", "1h", "4h", "1d"
TIMEFRAME = "1h"

# How many candles to test (more = longer history)
CANDLES = 5000

# Strategy: "trend" or "grid"
STRATEGY = "trend"

# For Grid strategy only:
GRID_UPPER = 1.12
GRID_LOWER = 1.02

# Spread and commission (IC Markets typical)
SPREAD_PIPS = 1.2
COMMISSION_PER_LOT = 7.0

# ============================================
# DON'T EDIT BELOW THIS LINE
# ============================================

def main():
    print("=" * 50)
    print("  STRATEGY TEST")
    print("=" * 50)
    print(f"  Pair: {PAIR}")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Strategy: {STRATEGY}")
    print(f"  Initial Balance: ${INITIAL_BALANCE}")
    print("=" * 50)
    
    # Create backtester
    bt = Backtester(
        initial_balance=INITIAL_BALANCE,
        spread_pips=SPREAD_PIPS,
        commission_per_lot=COMMISSION_PER_LOT
    )
    
    # Load data
    print("\nLoading data...")
    try:
        # Try MT5 first
        count = bt.load_data_from_mt5(PAIR, timeframe=TIMEFRAME, bars=CANDLES)
        print(f"Loaded {count:,} candles from MT5")
    except Exception as e:
        print(f"MT5 not available: {e}")
        print("Using sample data instead...")
        bt.generate_sample_data(PAIR, days=60, timeframe=TIMEFRAME)
    
    # Run backtest
    print("\nRunning backtest...")
    if STRATEGY == "grid":
        result = bt.run(PAIR, strategy="grid", grid_upper=GRID_UPPER, grid_lower=GRID_LOWER)
    else:
        result = bt.run(PAIR, strategy="trend")
    
    # Show results
    print(result.summary())
    
    # Verdict
    print("\n" + "=" * 50)
    print("  VERDICT")
    print("=" * 50)
    
    if result.profit_factor >= 1.5:
        print("  [OK] PROFITABLE - Good to use!")
    elif result.profit_factor >= 1.0:
        print("  [--] BREAKEVEN - Needs optimization")
    else:
        print("  [X] LOSING - Do NOT use these settings")
    
    if result.max_drawdown_pct > 30:
        print("  [!] HIGH RISK - Reduce position size")
    
    # Save trades
    bt.export_trades_csv("test_trades.csv")
    print("\nTrade details saved to: test_trades.csv")
    print("\nEdit the settings at the top of this file and run again!")

if __name__ == "__main__":
    main()
