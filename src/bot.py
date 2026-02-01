"""
Trading Bot Orchestrator

Main entry point for the Hybrid Grid + Trend Following Trading Bot.
Coordinates all modules and manages the trading workflow.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .config import BotConfig, load_config, set_config, get_config
from .logger import setup_logger, TradingBotLogger, log_daily_summary
from .risk_manager import RiskManager
from .grid_trader import GridTrader, GridState
from .trend_follower import TrendFollower, TrendState
from .api_client import (
    BrokerAPIClient, DryRunAPIClient, APIClientWithRetry,
    create_api_client, check_api_connection
)
from .dashboard import TradingDashboard, create_simple_dashboard
from .models import (
    Order, OrderSide, OrderType, OrderStatus, MarketData,
    AccountState, Candle, Signal, SignalType, TradingModule
)


class TradingBot:
    """
    Main Trading Bot Orchestrator.
    
    Coordinates:
    - Grid Trading Module
    - Trend Following Module
    - Risk Management
    - API Client
    - Dashboard/Monitoring
    
    Workflow:
    1. Initialize all modules with configuration
    2. Connect to broker API (or dry run)
    3. Fetch market data and account state
    4. Generate signals from both strategies
    5. Validate signals through risk management
    6. Execute approved signals
    7. Update dashboard and logs
    8. Monitor positions for exits
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        dry_run: bool = True
    ):
        """
        Initialize the trading bot.
        
        Args:
            config_path: Path to configuration file. Uses default if None.
            dry_run: If True, use simulated API client.
        """
        # Load configuration
        self.config = load_config(config_path)
        set_config(self.config)
        
        # Validate configuration
        warnings = self.config.validate_for_trading()
        if warnings:
            for warning in warnings:
                print(f"⚠️  CONFIG WARNING: {warning}")
        
        # Setup logging
        self.logger = setup_logger(
            name="TradingBot",
            log_level=self.config.logging.level,
            output_dir=self.config.logging.output_dir,
            output_file=self.config.logging.output_file,
            max_file_size_mb=self.config.logging.max_log_file_size_mb,
            backup_count=self.config.logging.backup_log_count,
            console_output=self.config.logging.console_output,
            colored_output=self.config.logging.colored_output
        )
        
        # Initialize modules
        self.risk_manager = RiskManager(
            config=self.config.risk_management,
            logger=self.logger
        )
        
        self.grid_trader = GridTrader(
            config=self.config.grid_strategy,
            risk_manager=self.risk_manager,
            logger=self.logger
        )
        
        self.trend_follower = TrendFollower(
            config=self.config.trend_strategy,
            risk_manager=self.risk_manager,
            logger=self.logger
        )
        
        # API client
        self.dry_run = dry_run or self.config.dry_run.enabled
        self.api_client: Optional[BrokerAPIClient] = None
        
        # Dashboard
        self.dashboard = TradingDashboard()
        
        # State tracking
        self._is_running = False
        self._account_state: Optional[AccountState] = None
        self._market_data: Dict[str, MarketData] = {}
        self._open_orders: List[Order] = []
        self._pending_signals: List[Signal] = []
        
        self.logger.info("=" * 60)
        self.logger.info("TRADING BOT INITIALIZED")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        self.logger.info(f"Pairs: {', '.join(self.config.trading.pairs)}")
        self.logger.info(f"Grid Strategy: {'Enabled' if self.config.grid_strategy.enabled else 'Disabled'}")
        self.logger.info(f"Trend Strategy: {'Enabled' if self.config.trend_strategy.enabled else 'Disabled'}")
        self.logger.info("=" * 60)

    async def start(self) -> None:
        """Start the trading bot."""
        self.logger.info("Starting trading bot...")
        
        # Create API client
        self.api_client = create_api_client(dry_run=self.dry_run)
        
        # Connect to broker
        try:
            connected = await self.api_client.connect()
            if not connected:
                self.logger.critical("Failed to connect to broker")
                return
        except Exception as e:
            self.logger.critical(f"Connection error: {e}")
            return
        
        self._is_running = True
        
        # Initial data fetch
        await self._update_account_state()
        await self._update_market_data()
        
        # Print startup info
        print(create_simple_dashboard())
        
        self.logger.info("Bot started successfully. Waiting for commands...")

    async def stop(self) -> None:
        """Stop the trading bot."""
        self.logger.info("Stopping trading bot...")
        self._is_running = False
        
        # Close any open positions if configured
        # (Optional: add force close on shutdown)
        
        # Disconnect from API
        if self.api_client:
            await self.api_client.disconnect()
        
        # Log daily summary
        stats = self.risk_manager.get_daily_stats()
        log_daily_summary(
            self.logger,
            trades_count=stats.total_trades,
            wins=stats.winning_trades,
            losses=stats.losing_trades,
            total_pnl=stats.net_profit,
            drawdown_pct=stats.max_drawdown_pct
        )
        
        self.logger.info("Bot stopped")

    async def _update_account_state(self) -> None:
        """Fetch and update current account state."""
        try:
            self._account_state = await self.api_client.get_account_info()
            self.dashboard.update_account_state(self._account_state)
            
            self.logger.account_state(
                balance=self._account_state.balance,
                equity=self._account_state.equity,
                margin_used=self._account_state.margin_used,
                free_margin=self._account_state.free_margin,
                daily_pnl=self._account_state.equity - self.config.risk_management.account_balance,
                drawdown_pct=((self.config.risk_management.account_balance - self._account_state.equity) /
                              self.config.risk_management.account_balance * 100)
            )
        except Exception as e:
            self.logger.error(f"Failed to update account state: {e}")

    async def _update_market_data(self) -> None:
        """Fetch and update market data for all pairs."""
        for pair in self.config.trading.pairs:
            try:
                data = await self.api_client.get_market_data(pair)
                self._market_data[pair] = data
                self.dashboard.update_market_data(pair, data)
                
                self.logger.market_data(
                    pair=pair,
                    bid=data.bid,
                    ask=data.ask,
                    spread_pips=data.spread_pips
                )
            except Exception as e:
                self.logger.error(f"Failed to get market data for {pair}: {e}")

    async def _fetch_candles(self, pair: str, count: int = 30) -> List[Candle]:
        """Fetch historical candles for a pair."""
        try:
            candles = await self.api_client.get_candles(
                pair=pair,
                timeframe=self.config.trend_strategy.timeframe,
                count=count
            )
            return candles
        except Exception as e:
            self.logger.error(f"Failed to fetch candles for {pair}: {e}")
            return []

    async def setup_grid(
        self,
        pair: str,
        upper_limit: float,
        lower_limit: float,
        num_grids: Optional[int] = None
    ) -> bool:
        """
        Set up a grid for a trading pair.
        
        Args:
            pair: Currency pair (e.g., "EUR/USD").
            upper_limit: Upper price boundary.
            lower_limit: Lower price boundary.
            num_grids: Number of grid levels (uses config default if None).
            
        Returns:
            True if grid setup successful.
        """
        if not self._account_state:
            await self._update_account_state()
        
        try:
            grid_config = self.grid_trader.setup_grid(
                pair=pair,
                upper_limit=upper_limit,
                lower_limit=lower_limit,
                account_balance=self._account_state.balance,
                num_grids=num_grids
            )
            
            self.logger.info(f"Grid set up for {pair}")
            self.logger.info(f"  Range: {lower_limit:.5f} - {upper_limit:.5f}")
            self.logger.info(f"  Grids: {grid_config.num_grids}")
            self.logger.info(f"  Spacing: {grid_config.grid_spacing_pips:.1f} pips")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup grid: {e}")
            return False

    async def analyze_pair(self, pair: str) -> Dict:
        """
        Analyze a pair and generate signals.
        
        Args:
            pair: Currency pair to analyze.
            
        Returns:
            Analysis results including signals.
        """
        if not self._account_state:
            await self._update_account_state()
        
        if pair not in self._market_data:
            await self._update_market_data()
        
        market_data = self._market_data.get(pair)
        if not market_data:
            return {"error": f"No market data for {pair}"}
        
        results = {
            "pair": pair,
            "timestamp": datetime.now().isoformat(),
            "market_data": {
                "bid": market_data.bid,
                "ask": market_data.ask,
                "spread_pips": market_data.spread_pips
            },
            "signals": [],
            "grid_state": None,
            "trend_state": None
        }
        
        # Fetch candles for trend analysis
        candles = await self._fetch_candles(pair)
        for candle in candles:
            self.trend_follower.add_candle(candle)
        
        # Generate grid signals
        if self.config.grid_strategy.enabled:
            grid_signals = self.grid_trader.generate_grid_signals(
                pair=pair,
                market_data=market_data,
                account_state=self._account_state
            )
            results["signals"].extend(grid_signals)
            
            grid_state = self.grid_trader.get_grid_state(pair)
            if grid_state:
                results["grid_state"] = {
                    "active": grid_state.is_active,
                    "buy_orders": grid_state.active_buy_orders,
                    "sell_orders": grid_state.active_sell_orders,
                    "profit": grid_state.total_profit
                }
        
        # Generate trend signals
        if self.config.trend_strategy.enabled:
            trend_signals = self.trend_follower.generate_signals(
                pair=pair,
                market_data=market_data,
                account_state=self._account_state
            )
            results["signals"].extend(trend_signals)
            
            trend_state = self.trend_follower.get_state(pair)
            results["trend_state"] = {
                "sma_short": trend_state.sma_short,
                "sma_long": trend_state.sma_long,
                "is_bullish": trend_state.is_bullish,
                "is_bearish": trend_state.is_bearish,
                "signal_strength": trend_state.signal_strength
            }
            self.dashboard.update_trend_state(pair, trend_state)
        
        return results

    async def execute_signal(self, signal: Signal) -> Optional[Order]:
        """
        Execute a trading signal after risk validation.
        
        Args:
            signal: Signal to execute.
            
        Returns:
            Executed order or None if rejected.
        """
        if not self._account_state:
            await self._update_account_state()
        
        market_data = self._market_data.get(signal.pair)
        if not market_data:
            self.logger.error(f"No market data for {signal.pair}")
            return None
        
        # Create order from signal
        order = Order(
            order_id="",  # Will be assigned by API
            pair=signal.pair,
            side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            lot_size=signal.metadata.get("lot_size", 0.01),
            entry_price=signal.price,
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            module=signal.module
        )
        
        # Validate through risk manager
        risk_result = self.risk_manager.validate_order(
            order=order,
            account_state=self._account_state,
            market_data=market_data
        )
        
        if not risk_result.passed:
            self.logger.warning(f"Signal rejected by risk manager: {risk_result.message}")
            return None
        
        # Execute order
        try:
            executed_order = await self.api_client.place_order(order)
            
            # Track the order
            self._open_orders.append(executed_order)
            self.dashboard.update_positions(self._open_orders)
            
            # Register with appropriate module
            if signal.module == TradingModule.GRID:
                grid_level = signal.metadata.get("grid_level", 0)
                self.grid_trader.register_order(signal.pair, grid_level, executed_order)
            elif signal.module == TradingModule.TREND:
                self.trend_follower.register_position(executed_order)
            
            self.logger.info(
                f"Order executed: {executed_order.side.value} {executed_order.lot_size} "
                f"{executed_order.pair} @ {executed_order.fill_price:.5f}"
            )
            
            return executed_order
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return None

    async def close_position(self, order_id: str) -> Optional[Order]:
        """
        Close an open position.
        
        Args:
            order_id: ID of order to close.
            
        Returns:
            Closed order or None if failed.
        """
        try:
            closed_order = await self.api_client.close_order(order_id)
            
            # Remove from open orders
            self._open_orders = [o for o in self._open_orders if o.order_id != order_id]
            self.dashboard.update_positions(self._open_orders)
            self.dashboard.add_trade_to_history(closed_order)
            
            # Record with risk manager
            self.risk_manager.record_trade(closed_order)
            
            # Update modules
            if closed_order.module == TradingModule.GRID:
                self.grid_trader.close_order(
                    pair=closed_order.pair,
                    order_id=order_id,
                    close_price=closed_order.close_price,
                    pnl=closed_order.pnl
                )
            elif closed_order.module == TradingModule.TREND:
                self.trend_follower.close_position(closed_order.pair)
            
            self.logger.info(
                f"Position closed: {closed_order.pair} P&L: ${closed_order.pnl:+.2f}"
            )
            
            return closed_order
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return None

    async def check_health(self) -> bool:
        """
        Perform health check on all components.
        
        Returns:
            True if all components healthy.
        """
        # Check API connection
        api_healthy = await check_api_connection(self.api_client, self.logger)
        if not api_healthy:
            return False
        
        # Check risk limits
        if self._account_state:
            can_trade, reason = self.risk_manager.can_trade(self._account_state)
            if not can_trade:
                self.logger.warning(f"Trading restricted: {reason}")
                return False
        
        return True

    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            "is_running": self._is_running,
            "dry_run": self.dry_run,
            "account": {
                "balance": self._account_state.balance if self._account_state else 0,
                "equity": self._account_state.equity if self._account_state else 0,
                "margin_used": self._account_state.margin_used if self._account_state else 0,
            },
            "open_positions": len(self._open_orders),
            "active_grids": self.grid_trader.get_active_grids(),
            "daily_stats": {
                "trades": self.risk_manager.get_daily_stats().total_trades,
                "pnl": self.risk_manager.get_daily_stats().net_profit,
                "win_rate": self.risk_manager.get_daily_stats().win_rate
            }
        }


async def main():
    """Main entry point for the trading bot."""
    print("=" * 60)
    print("  HYBRID GRID + TREND FOLLOWING TRADING BOT")
    print("  Small Account Edition")
    print("=" * 60)
    print()
    
    # Create bot instance
    bot = TradingBot(dry_run=True)
    
    # Handle shutdown signals
    def shutdown_handler(sig, frame):
        print("\nShutdown signal received...")
        asyncio.create_task(bot.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start the bot
    await bot.start()
    
    # Example: Setup a grid and analyze
    print("\nSetting up example grid for EUR/USD...")
    await bot.setup_grid(
        pair="EUR/USD",
        upper_limit=1.0900,
        lower_limit=1.0800,
        num_grids=10
    )
    
    print("\nAnalyzing EUR/USD...")
    analysis = await bot.analyze_pair("EUR/USD")
    print(f"Analysis complete. Found {len(analysis.get('signals', []))} signals.")
    
    # Show status
    status = bot.get_status()
    print(f"\nBot Status:")
    print(f"  Running: {status['is_running']}")
    print(f"  Mode: {'Dry Run' if status['dry_run'] else 'Live'}")
    print(f"  Balance: ${status['account']['balance']:.2f}")
    print(f"  Open Positions: {status['open_positions']}")
    print(f"  Active Grids: {status['active_grids']}")
    
    print("\nBot is ready. Use Ctrl+C to stop.")
    
    # Keep running
    while bot._is_running:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
