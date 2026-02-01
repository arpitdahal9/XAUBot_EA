"""
Example: Connect to IC Markets MT5

This script shows how to connect your IC Markets MT5 account
to the trading bot.

Prerequisites:
1. Install MetaTrader 5 terminal on your PC
2. Log into your IC Markets account in MT5
3. Run: pip install MetaTrader5

Usage:
    python examples/connect_ic_markets.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if MT5 is available
try:
    import MetaTrader5 as mt5
    print("✓ MetaTrader5 package found")
except ImportError:
    print("✗ MetaTrader5 package not installed")
    print("  Install with: pip install MetaTrader5")
    print("  Note: Only works on Windows")
    sys.exit(1)


def test_mt5_connection():
    """Test basic MT5 connection."""
    print("\n" + "=" * 50)
    print("Testing MetaTrader 5 Connection")
    print("=" * 50)
    
    # Initialize MT5
    print("\n1. Initializing MT5...")
    if not mt5.initialize():
        print(f"   ✗ Failed: {mt5.last_error()}")
        print("\n   Make sure:")
        print("   - MT5 terminal is installed")
        print("   - MT5 terminal is running")
        print("   - You are logged into your account")
        return False
    
    print("   ✓ MT5 initialized")
    
    # Get account info
    print("\n2. Getting account info...")
    account = mt5.account_info()
    if account is None:
        print(f"   ✗ Failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print(f"   ✓ Connected to account")
    print(f"\n   Account Details:")
    print(f"   ├─ Login:    {account.login}")
    print(f"   ├─ Server:   {account.server}")
    print(f"   ├─ Name:     {account.name}")
    print(f"   ├─ Balance:  ${account.balance:.2f}")
    print(f"   ├─ Equity:   ${account.equity:.2f}")
    print(f"   ├─ Leverage: 1:{account.leverage}")
    print(f"   └─ Currency: {account.currency}")
    
    # Test market data
    print("\n3. Testing market data...")
    symbol = "EURUSD"
    
    # Enable symbol
    if not mt5.symbol_select(symbol, True):
        print(f"   ✗ Symbol {symbol} not available")
    else:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = (tick.ask - tick.bid) * 10000
            print(f"   ✓ {symbol}:")
            print(f"   ├─ Bid: {tick.bid:.5f}")
            print(f"   ├─ Ask: {tick.ask:.5f}")
            print(f"   └─ Spread: {spread:.1f} pips")
    
    # Check trading permissions
    print("\n4. Checking trading permissions...")
    terminal_info = mt5.terminal_info()
    if terminal_info:
        if terminal_info.trade_allowed:
            print("   ✓ Trading is allowed")
        else:
            print("   ✗ Trading is NOT allowed")
            print("   Enable in MT5: Tools > Options > Expert Advisors")
            print("   Check 'Allow algorithmic trading'")
    
    # Cleanup
    mt5.shutdown()
    
    print("\n" + "=" * 50)
    print("Connection test complete!")
    print("=" * 50)
    
    return True


async def run_bot_with_mt5():
    """Example of running the bot with real MT5 connection."""
    from src.mt5_client import MT5Client
    from src.api_client import APIClientWithRetry
    from src.bot import TradingBot
    from src.config import get_config
    
    print("\n" + "=" * 50)
    print("Starting Bot with MT5 Connection")
    print("=" * 50)
    
    # Create MT5 client
    mt5_client = MT5Client()
    
    # Connect (uses already logged-in MT5 terminal)
    await mt5_client.connect()
    
    # Wrap with retry logic
    api_client = APIClientWithRetry(mt5_client)
    
    # Create bot
    bot = TradingBot(dry_run=False)  # Real trading!
    bot.api_client = api_client
    
    # Start bot
    await bot.start()
    
    # Get status
    status = bot.get_status()
    print(f"\nBot Status:")
    print(f"  Balance: ${status['account']['balance']:.2f}")
    print(f"  Equity: ${status['account']['equity']:.2f}")
    
    # Analyze EUR/USD
    print("\nAnalyzing EUR/USD...")
    analysis = await bot.analyze_pair("EUR/USD")
    
    if analysis.get('signals'):
        print(f"  Found {len(analysis['signals'])} signals")
        for signal in analysis['signals'][:3]:
            print(f"    - {signal.signal_type.value} @ {signal.price:.5f}")
    
    # Stop bot
    await bot.stop()
    
    print("\nBot stopped successfully")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("IC Markets MT5 Connection Example")
    print("=" * 50)
    
    # First test basic connection
    if test_mt5_connection():
        print("\n\nWould you like to run the bot with real MT5? (yes/no)")
        response = input("> ").strip().lower()
        
        if response == "yes":
            print("\n⚠️  WARNING: This will use your REAL account!")
            print("Make sure you understand the risks.")
            confirm = input("Type 'I understand' to continue: ").strip()
            
            if confirm == "I understand":
                asyncio.run(run_bot_with_mt5())
            else:
                print("Cancelled.")
        else:
            print("\nTo run with MT5, modify main.py or use run_live.py")
