"""
Example: Run Backtests

Test your trading strategies on historical data before going live.

Usage:
    python examples/run_backtest.py
    python examples/run_backtest.py --strategy trend --days 60
    python examples/run_backtest.py --data mydata.csv --strategy grid
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import Backtester, BacktestResult


def run_sample_backtest():
    """Run backtest with sample data."""
    print("\n" + "=" * 60)
    print("  BACKTESTING TRADING STRATEGIES")
    print("=" * 60)
    
    # Create backtester with $1000 starting balance
    backtester = Backtester(
        initial_balance=1000.0,
        spread_pips=1.5,      # Realistic IC Markets spread
        slippage_pips=0.5,    # Realistic slippage
        commission_per_lot=7.0  # IC Markets commission
    )
    
    # Generate 30 days of sample data
    print("\n1. Generating sample data...")
    backtester.generate_sample_data(
        pair="EUR/USD",
        days=30,
        timeframe="15m"
    )
    
    # Test Trend Following Strategy
    print("\n2. Testing TREND FOLLOWING strategy...")
    trend_result = backtester.run(
        pair="EUR/USD",
        strategy="trend"
    )
    
    print(trend_result.summary())
    
    # Reset and test Grid Strategy
    print("\n3. Testing GRID strategy...")
    backtester_grid = Backtester(initial_balance=1000.0)
    backtester_grid.generate_sample_data("EUR/USD", days=30)
    
    grid_result = backtester_grid.run(
        pair="EUR/USD",
        strategy="grid",
        grid_upper=1.0900,
        grid_lower=1.0800
    )
    
    print(grid_result.summary())
    
    # Compare results
    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Trend':>15} {'Grid':>15}")
    print("-" * 55)
    print(f"{'Net Profit':<25} ${trend_result.net_profit:>14.2f} ${grid_result.net_profit:>14.2f}")
    print(f"{'Win Rate':<25} {trend_result.win_rate:>14.1f}% {grid_result.win_rate:>14.1f}%")
    print(f"{'Total Trades':<25} {trend_result.total_trades:>15} {grid_result.total_trades:>15}")
    print(f"{'Profit Factor':<25} {trend_result.profit_factor:>15.2f} {grid_result.profit_factor:>15.2f}")
    print(f"{'Max Drawdown':<25} ${trend_result.max_drawdown:>14.2f} ${grid_result.max_drawdown:>14.2f}")
    print(f"{'Sharpe Ratio':<25} {trend_result.sharpe_ratio:>15.2f} {grid_result.sharpe_ratio:>15.2f}")
    
    return trend_result, grid_result


def run_backtest_with_mt5_data():
    """Run backtest with real historical data from MT5."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MT5 not available. Install with: pip install MetaTrader5")
        return None
    
    print("\n" + "=" * 60)
    print("  BACKTESTING WITH MT5 DATA")
    print("=" * 60)
    
    backtester = Backtester(initial_balance=1000.0)
    
    print("\n1. Loading data from MT5...")
    try:
        candle_count = backtester.load_data_from_mt5(
            pair="EUR/USD",
            timeframe="15m",
            bars=5000  # About 2 months of 15m data
        )
        print(f"   Loaded {candle_count} candles")
    except Exception as e:
        print(f"   Failed: {e}")
        return None
    
    print("\n2. Running backtest...")
    result = backtester.run(pair="EUR/USD", strategy="trend")
    
    print(result.summary())
    
    # Export trades
    backtester.export_trades_csv("backtest_trades.csv")
    print("\nTrade list exported to backtest_trades.csv")
    
    return result


def interactive_backtest():
    """Interactive backtest configuration."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE BACKTEST")
    print("=" * 60)
    
    # Get parameters
    print("\nConfigure backtest parameters:")
    
    balance = input("  Initial balance [$1000]: ").strip()
    balance = float(balance) if balance else 1000.0
    
    pair = input("  Currency pair [EUR/USD]: ").strip()
    pair = pair.upper() if pair else "EUR/USD"
    if "/" not in pair and len(pair) == 6:
        pair = pair[:3] + "/" + pair[3:]
    
    strategy = input("  Strategy (trend/grid/both) [trend]: ").strip().lower()
    strategy = strategy if strategy in ("trend", "grid", "both") else "trend"
    
    days = input("  Days of data [30]: ").strip()
    days = int(days) if days else 30
    
    # Create and run backtest
    backtester = Backtester(initial_balance=balance)
    
    # Try MT5 first, fall back to sample data
    try:
        import MetaTrader5 as mt5
        use_mt5 = input("\n  Use real MT5 data? (yes/no) [no]: ").strip().lower()
        if use_mt5 == "yes":
            backtester.load_data_from_mt5(pair, bars=days * 96)  # ~96 15m candles per day
        else:
            raise ImportError()
    except:
        print(f"\n  Generating {days} days of sample data...")
        backtester.generate_sample_data(pair, days=days)
    
    print(f"\n  Running {strategy} strategy backtest...")
    
    if strategy == "grid":
        upper = input("  Grid upper limit: ").strip()
        lower = input("  Grid lower limit: ").strip()
        result = backtester.run(
            pair, 
            strategy=strategy,
            grid_upper=float(upper) if upper else None,
            grid_lower=float(lower) if lower else None
        )
    else:
        result = backtester.run(pair, strategy=strategy)
    
    print(result.summary())
    
    # Export option
    export = input("\nExport trades to CSV? (yes/no): ").strip().lower()
    if export == "yes":
        filename = input("  Filename [backtest_trades.csv]: ").strip()
        filename = filename if filename else "backtest_trades.csv"
        backtester.export_trades_csv(filename)
        print(f"  Exported to {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--mt5", action="store_true", help="Use MT5 data")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_backtest()
    elif args.mt5:
        run_backtest_with_mt5_data()
    else:
        run_sample_backtest()
    
    print("\n✓ Backtest complete!")
