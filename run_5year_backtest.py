"""
5-Year Backtest: January 2020 to Present
Using real EUR/USD data from MetaTrader 5
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
import time

def check_mt5_connection():
    """Check if MT5 is running and has data."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.")
        print("Run: pip install MetaTrader5")
        return False
    
    if not mt5.initialize():
        print("ERROR: Cannot connect to MT5!")
        print("")
        print("Please make sure:")
        print("  1. MetaTrader 5 is INSTALLED on your computer")
        print("  2. MetaTrader 5 is RUNNING (open it now)")
        print("  3. You are LOGGED IN to your IC Markets account")
        print("")
        return False
    
    # Try to get EURUSD data
    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)
    time.sleep(1)  # Wait for data
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    
    if rates is None or len(rates) == 0:
        print("ERROR: No data available for EURUSD")
        print("")
        print("Please do this in MetaTrader 5:")
        print("  1. Click View -> Market Watch (or press Ctrl+M)")
        print("  2. Right-click in Market Watch -> Show All")
        print("  3. Find EURUSD and double-click to open chart")
        print("  4. Wait 30 seconds for data to download")
        print("  5. Run this script again")
        print("")
        mt5.shutdown()
        return False
    
    print(f"MT5 connected! Found {len(rates)} bars of data.")
    mt5.shutdown()
    return True

def main():
    print("=" * 60)
    print("  5-YEAR BACKTEST: EUR/USD")
    print("  Period: January 2020 - Now")
    print("=" * 60)
    
    # Check MT5 first
    print("\nChecking MT5 connection...")
    if not check_mt5_connection():
        print("\nCannot proceed without MT5 data.")
        print("Fix the issues above and try again.")
        return
    
    from src.backtester import Backtester
    from src.config import get_config
    
    # Load balance from config
    config = get_config()
    balance = config.risk_management.account_balance
    
    # Create backtester with realistic IC Markets conditions
    backtester = Backtester(
        initial_balance=balance,
        spread_pips=1.2,        # IC Markets typical spread
        slippage_pips=0.3,      # Low slippage
        commission_per_lot=7.0  # IC Markets Raw Spread commission
    )
    
    print(f"      Account Balance: ${balance:,.2f}")
    
    # Define exact date range: Jan 2020 to now
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    print("\n[1/3] Loading historical data from MT5...")
    print(f"      Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("      This may take a moment...")
    
    try:
        # Load data for exact date range (not arbitrary bar count)
        candle_count = backtester.load_data_from_mt5(
            pair="EUR/USD",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"      Loaded {candle_count:,} candles")
        
        # Verify date range
        if candle_count > 0:
            years = (end_date - start_date).days / 365.25
            print(f"      Period: ~{years:.1f} years")
        
    except Exception as e:
        print(f"      ERROR: {e}")
        print("\n      Using sample data instead...")
        backtester.generate_sample_data("EUR/USD", days=365*5, timeframe="1h")
        candle_count = 365 * 5 * 24
        print(f"      Generated {candle_count:,} sample candles (5 years)")
    
    print("\n[2/3] Running TREND FOLLOWING backtest...")
    print("      Strategy: SMA 9/20 Crossover")
    print(f"      Risk: {config.risk_management.risk_percent_per_trade}% per trade")
    print(f"      Stop Loss: {config.trend_strategy.stop_loss_pips} pips")
    
    # Check if partial exits enabled
    import json
    with open("config/settings.json", "r") as f:
        raw_cfg = json.load(f)
        partial = raw_cfg.get("risk_management", {}).get("partial_exits", {})
        if partial.get("enabled", False):
            print("      Exit Mode: MODEL A (Partial Exits)")
            print(f"        TP1: +{partial.get('tp1_r_multiple', 2.0)}R closes {partial.get('tp1_close_percent', 50)}%")
            print(f"        TP2: +{partial.get('tp2_r_multiple', 3.0)}R closes {partial.get('tp2_close_percent', 30)}%")
            print("        Runner: ATR trailing on remaining 20%")
        else:
            print(f"      Take Profit: {config.trend_strategy.take_profit_pips} pips (fixed)")
    
    result = backtester.run(
        pair="EUR/USD",
        strategy="trend"
    )
    
    print("\n[3/3] Results:")
    print(result.summary())
    
    # Export trades for analysis
    export_file = "backtest_5year_trades.csv"
    backtester.export_trades_csv(export_file)
    print(f"\nTrade details exported to: {export_file}")
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    
    if result.total_trades > 0:
        # Calculate actual duration from results
        actual_days = (result.end_date - result.start_date).days
        actual_months = actual_days / 30.44
        actual_years = actual_days / 365.25
        
        return_pct = (result.net_profit / result.initial_balance) * 100
        monthly_return = return_pct / actual_months if actual_months > 0 else 0
        yearly_return = return_pct / actual_years if actual_years > 0 else 0
        
        print(f"\n  Actual Period:       {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"  Duration:            {actual_years:.1f} years ({actual_months:.0f} months)")
        print(f"\n  Total Return:        {return_pct:+.2f}%")
        print(f"  Avg Monthly Return:  {monthly_return:+.2f}%")
        print(f"  Avg Yearly Return:   {yearly_return:+.2f}%")
        print(f"\n  Risk Assessment:")
        
        if result.profit_factor >= 1.5:
            print("    Profit Factor:     GOOD (>1.5)")
        elif result.profit_factor >= 1.0:
            print("    Profit Factor:     MARGINAL (1.0-1.5)")
        else:
            print("    Profit Factor:     POOR (<1.0)")
        
        if result.max_drawdown_pct <= 20:
            print("    Max Drawdown:      ACCEPTABLE (<20%)")
        elif result.max_drawdown_pct <= 30:
            print("    Max Drawdown:      HIGH (20-30%)")
        else:
            print("    Max Drawdown:      DANGEROUS (>30%)")
        
        if result.win_rate >= 50:
            print(f"    Win Rate:          GOOD ({result.win_rate:.1f}%)")
        else:
            print(f"    Win Rate:          NEEDS WORK ({result.win_rate:.1f}%)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
