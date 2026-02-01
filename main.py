#!/usr/bin/env python3
"""
Hybrid Grid + Trend Following Trading Bot
Small Account Edition

Entry point script for the trading bot.

Usage:
    python main.py [--live] [--config PATH]
    
Options:
    --live      Run in live trading mode (default is dry-run)
    --config    Path to configuration file (default: config/settings.json)

Examples:
    # Run in dry-run mode (recommended for testing)
    python main.py
    
    # Run with custom config
    python main.py --config my_config.json
    
    # Run in live mode (use with caution!)
    python main.py --live
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent))

from src.bot import TradingBot, main as bot_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Grid + Trend Following Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run in dry-run mode
  python main.py --config my.json   # Use custom config
  python main.py --live             # Live trading (caution!)
        """
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (default is dry-run for safety)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config/settings.json)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗                ║
║   ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗               ║
║   ███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║               ║
║   ██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║               ║
║   ██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝               ║
║   ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝               ║
║                                                               ║
║       Grid + Trend Following Trading Bot                      ║
║       Small Account Edition ($1,000+)                         ║
║       IC Markets Australia (1:30 Leverage)                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_warning():
    """Print risk warning."""
    warning = """
┌───────────────────────────────────────────────────────────────┐
│                     ⚠️  RISK WARNING ⚠️                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Trading forex involves significant risk of loss.             │
│  Never trade with money you cannot afford to lose.            │
│                                                               │
│  This bot enforces strict risk management:                    │
│    • 1% risk per trade                                        │
│    • $50 daily loss limit                                     │
│    • 10% maximum drawdown                                     │
│    • Mandatory stop-loss on all trades                        │
│                                                               │
│  ALWAYS test in dry-run mode before live trading!             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
    """
    print(warning)


async def interactive_session(bot: TradingBot):
    """
    Run an interactive trading session.
    
    This provides a simple command-line interface for manual operation.
    """
    print("\n" + "=" * 60)
    print("  INTERACTIVE TRADING SESSION")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  status     - Show current bot status")
    print("  analyze    - Analyze all configured pairs")
    print("  grid       - Setup a new grid")
    print("  positions  - Show open positions")
    print("  close      - Close a position")
    print("  help       - Show this help")
    print("  quit       - Stop bot and exit")
    print()
    
    while bot._is_running:
        try:
            command = input("\n[Bot] > ").strip().lower()
            
            if command == "quit" or command == "exit":
                print("Shutting down...")
                await bot.stop()
                break
                
            elif command == "status":
                status = bot.get_status()
                print(f"\n  Running: {status['is_running']}")
                print(f"  Mode: {'Dry Run' if status['dry_run'] else 'LIVE'}")
                print(f"  Balance: ${status['account']['balance']:.2f}")
                print(f"  Equity: ${status['account']['equity']:.2f}")
                print(f"  Open Positions: {status['open_positions']}")
                print(f"  Daily Trades: {status['daily_stats']['trades']}")
                print(f"  Daily P&L: ${status['daily_stats']['pnl']:.2f}")
                
            elif command == "analyze":
                print("\nAnalyzing pairs...")
                for pair in bot.config.trading.pairs:
                    analysis = await bot.analyze_pair(pair)
                    signals = analysis.get('signals', [])
                    trend = analysis.get('trend_state', {})
                    
                    direction = "BULLISH" if trend.get('is_bullish') else "BEARISH" if trend.get('is_bearish') else "NEUTRAL"
                    print(f"\n  {pair}:")
                    print(f"    Trend: {direction}")
                    print(f"    Signals: {len(signals)}")
                    if signals:
                        for s in signals[:3]:  # Show first 3
                            print(f"      - {s.signal_type.value} @ {s.price:.5f}")
                
            elif command == "grid":
                pair = input("  Pair (e.g., EUR/USD): ").strip().upper()
                if "/" not in pair:
                    pair = pair[:3] + "/" + pair[3:]
                
                try:
                    upper = float(input("  Upper limit: "))
                    lower = float(input("  Lower limit: "))
                    grids = input("  Number of grids (Enter for default): ").strip()
                    grids = int(grids) if grids else None
                    
                    success = await bot.setup_grid(pair, upper, lower, grids)
                    if success:
                        print(f"\n  ✓ Grid set up for {pair}")
                    else:
                        print(f"\n  ✗ Failed to setup grid")
                except ValueError:
                    print("  Invalid input")
                
            elif command == "positions":
                if not bot._open_orders:
                    print("\n  No open positions")
                else:
                    print(f"\n  Open Positions ({len(bot._open_orders)}):")
                    for order in bot._open_orders:
                        print(f"    #{order.order_id}: {order.side.value} {order.lot_size} {order.pair}")
                
            elif command == "close":
                if not bot._open_orders:
                    print("\n  No positions to close")
                else:
                    print("\n  Open orders:")
                    for i, order in enumerate(bot._open_orders):
                        print(f"    {i+1}. #{order.order_id}: {order.side.value} {order.pair}")
                    
                    choice = input("  Enter order ID to close: ").strip()
                    order = await bot.close_position(choice)
                    if order:
                        print(f"\n  ✓ Position closed. P&L: ${order.pnl:+.2f}")
                    else:
                        print("\n  ✗ Failed to close position")
                
            elif command == "help":
                print("\n  status     - Show current bot status")
                print("  analyze    - Analyze all configured pairs")
                print("  grid       - Setup a new grid")
                print("  positions  - Show open positions")
                print("  close      - Close a position")
                print("  quit       - Stop bot and exit")
                
            else:
                print(f"  Unknown command: {command}")
                print("  Type 'help' for available commands")
                
        except asyncio.CancelledError:
            break
        except KeyboardInterrupt:
            print("\n\nInterrupted. Shutting down...")
            await bot.stop()
            break
        except Exception as e:
            print(f"  Error: {e}")


async def run_bot(args):
    """Run the trading bot."""
    print_banner()
    
    if not args.live:
        print("\n  Mode: DRY RUN (Safe testing mode)")
    else:
        print_warning()
        confirm = input("\n  Are you sure you want to run in LIVE mode? (yes/no): ")
        if confirm.lower() != "yes":
            print("  Cancelled. Running in dry-run mode instead.")
            args.live = False
    
    # Create and start bot
    bot = TradingBot(
        config_path=args.config,
        dry_run=not args.live
    )
    
    try:
        await bot.start()
        
        # Run interactive session
        await interactive_session(bot)
        
    except Exception as e:
        print(f"\n  Error: {e}")
        await bot.stop()
        raise
    finally:
        if bot._is_running:
            await bot.stop()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        asyncio.run(run_bot(args))
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
