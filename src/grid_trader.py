"""
Grid Trading Module

Automates buy/sell pairs in ranging markets using a grid strategy.
Places orders at predefined price levels and captures profits from
price oscillations within the grid range.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

from .models import (
    Order, OrderType, OrderSide, OrderStatus, MarketData,
    GridLevel, GridConfig, Signal, SignalType, TradingModule, AccountState
)
from .config import GridStrategyConfig, get_config
from .risk_manager import RiskManager, PositionSizeResult
from .logger import TradingBotLogger, get_logger, TradeLogEntry


@dataclass
class GridState:
    """Current state of the grid trading system."""
    is_active: bool = False
    current_price: float = 0.0
    active_buy_orders: int = 0
    active_sell_orders: int = 0
    total_profit: float = 0.0
    grid_config: Optional[GridConfig] = None


class GridTrader:
    """
    Grid Trading Module for ranging market conditions.
    
    Strategy:
    1. Define upper and lower price limits
    2. Divide range into N grid levels
    3. Place buy orders at grid levels below current price
    4. Each buy order has paired sell order at +profit_pips above
    5. When price bounces within grid, capture each movement
    6. If price breaks limits, close all orders and wait
    
    This module generates signals - actual order execution is handled
    by the order manager/API client.
    """

    def __init__(
        self,
        config: Optional[GridStrategyConfig] = None,
        risk_manager: Optional[RiskManager] = None,
        logger: Optional[TradingBotLogger] = None
    ):
        """
        Initialize the Grid Trader.
        
        Args:
            config: Grid strategy configuration.
            risk_manager: Risk manager instance for position sizing.
            logger: Logger instance.
        """
        self.config = config or get_config().grid_strategy
        self.risk_manager = risk_manager or RiskManager()
        self.logger = logger or get_logger("GridTrader")
        
        # Grid state
        self._grid_configs: Dict[str, GridConfig] = {}
        self._active_orders: Dict[str, List[Order]] = {}
        self._state = GridState()
        
        self.logger.info("[GRID] Grid Trader initialized")
        self.logger.info(f"  Grid Lines: {self.config.grid_lines}")
        self.logger.info(f"  Profit per Grid: {self.config.profit_per_grid_pips} pips")
        self.logger.info(f"  Stop Loss: {self.config.stop_loss_pips} pips")

    def setup_grid(
        self,
        pair: str,
        upper_limit: float,
        lower_limit: float,
        account_balance: float,
        num_grids: Optional[int] = None
    ) -> GridConfig:
        """
        Set up a new grid for a trading pair.
        
        Args:
            pair: Currency pair (e.g., "EUR/USD").
            upper_limit: Upper price limit of the grid.
            lower_limit: Lower price limit of the grid.
            account_balance: Current account balance for position sizing.
            num_grids: Number of grid lines (uses config default if None).
            
        Returns:
            Configured GridConfig object.
        """
        num_grids = num_grids or self.config.grid_lines
        
        # Validate limits
        if upper_limit <= lower_limit:
            raise ValueError("Upper limit must be greater than lower limit")
        
        # Calculate grid spacing
        grid_spacing = (upper_limit - lower_limit) / num_grids
        grid_spacing_pips = grid_spacing * (100 if "JPY" in pair else 10000)
        
        self.logger.info(f"[GRID] Setting up grid for {pair}")
        self.logger.debug(f"  Upper Limit: {upper_limit:.5f}")
        self.logger.debug(f"  Lower Limit: {lower_limit:.5f}")
        self.logger.debug(f"  Number of Grids: {num_grids}")
        self.logger.debug(f"  Grid Spacing: {grid_spacing_pips:.1f} pips")
        
        # Calculate investment per grid
        # Reserve some capital for margin requirements
        usable_capital = account_balance * 0.8  # Use 80% of capital
        investment_per_grid = usable_capital / num_grids
        
        self.logger.debug(f"  Investment per Grid: ${investment_per_grid:.2f}")
        
        # Create grid configuration
        grid_config = GridConfig(
            pair=pair,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            num_grids=num_grids,
            investment_per_grid=investment_per_grid,
            profit_per_grid_pips=self.config.profit_per_grid_pips,
            stop_loss_pips=self.config.stop_loss_pips
        )
        
        # Initialize grid levels
        grid_config.initialize_levels()
        
        self.logger.info(f"[GRID] Grid created with {len(grid_config.levels)} levels")
        for level in grid_config.levels:
            self.logger.debug(f"  Level {level.level_number}: {level.price:.5f}")
        
        # Store configuration
        self._grid_configs[pair] = grid_config
        self._active_orders[pair] = []
        
        return grid_config

    def calculate_grid_lot_size(
        self,
        pair: str,
        account_balance: float
    ) -> PositionSizeResult:
        """
        Calculate lot size for grid orders.
        
        Uses risk manager to ensure proper position sizing.
        
        Args:
            pair: Currency pair.
            account_balance: Current account balance.
            
        Returns:
            PositionSizeResult with calculated lot size.
        """
        return self.risk_manager.calculate_position_size(
            pair=pair,
            stop_loss_pips=self.config.stop_loss_pips,
            account_equity=account_balance,
            free_margin=account_balance,
            risk_percent=get_config().risk_management.risk_percent_per_trade
        )

    def generate_grid_signals(
        self,
        pair: str,
        market_data: MarketData,
        account_state: AccountState
    ) -> List[Signal]:
        """
        Generate trading signals based on current price and grid levels.
        
        Args:
            pair: Currency pair.
            market_data: Current market data.
            account_state: Current account state.
            
        Returns:
            List of signals to execute.
        """
        signals = []
        
        if pair not in self._grid_configs:
            self.logger.warning(f"[GRID] No grid configured for {pair}")
            return signals
        
        grid_config = self._grid_configs[pair]
        current_price = market_data.mid_price
        
        self.logger.debug(f"[GRID] Analyzing {pair} at {current_price:.5f}")
        
        # Check if price is within grid range
        if current_price > grid_config.upper_limit:
            self.logger.warning(f"[GRID] Price {current_price:.5f} ABOVE upper limit {grid_config.upper_limit:.5f}")
            signals.extend(self._generate_close_signals(pair, "UPPER_BREAK"))
            return signals
        
        if current_price < grid_config.lower_limit:
            self.logger.warning(f"[GRID] Price {current_price:.5f} BELOW lower limit {grid_config.lower_limit:.5f}")
            signals.extend(self._generate_close_signals(pair, "LOWER_BREAK"))
            return signals
        
        # Calculate lot size for new orders
        lot_result = self.calculate_grid_lot_size(pair, account_state.balance)
        if not lot_result.is_valid:
            self.logger.error(f"[GRID] Cannot calculate lot size: {lot_result.message}")
            return signals
        
        lot_size = lot_result.lot_size
        
        # Check which grid levels need orders
        for level in grid_config.levels:
            if not level.is_active:
                continue
            
            # Place buy order at levels below current price
            if level.price < current_price and level.buy_order is None:
                # Check if we should place a buy order at this level
                distance_pips = (current_price - level.price) * (100 if "JPY" in pair else 10000)
                
                # Only place order if price is within reasonable distance
                if distance_pips <= self.config.profit_per_grid_pips * 2:
                    signal = self._create_buy_signal(
                        pair=pair,
                        level=level,
                        lot_size=lot_size,
                        market_data=market_data
                    )
                    if signal:
                        signals.append(signal)
                        self.logger.info(
                            f"[GRID] BUY SIGNAL at level {level.level_number}: "
                            f"{pair} @ {level.price:.5f}"
                        )
            
            # Check if existing buy order should be paired with sell (take profit)
            if level.buy_order is not None and level.buy_order.is_open:
                if level.sell_order is None:
                    # Price has moved up, check if we should take profit
                    entry_price = level.buy_order.fill_price or level.buy_order.entry_price
                    profit_target = entry_price + (
                        self.config.profit_per_grid_pips / 
                        (100 if "JPY" in pair else 10000)
                    )
                    
                    if current_price >= profit_target:
                        signal = self._create_close_signal(
                            pair=pair,
                            level=level,
                            market_data=market_data,
                            reason="TAKE_PROFIT"
                        )
                        if signal:
                            signals.append(signal)
                            self.logger.info(
                                f"[GRID] CLOSE SIGNAL at level {level.level_number}: "
                                f"Take profit @ {current_price:.5f}"
                            )
        
        return signals

    def _create_buy_signal(
        self,
        pair: str,
        level: GridLevel,
        lot_size: float,
        market_data: MarketData
    ) -> Optional[Signal]:
        """Create a buy signal for a grid level."""
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            return None
        
        # Calculate stop loss and take profit
        pip_multiplier = 100 if "JPY" in pair else 10000
        stop_loss = level.price - (self.config.stop_loss_pips / pip_multiplier)
        take_profit = level.price + (self.config.profit_per_grid_pips / pip_multiplier)
        
        return Signal(
            signal_type=SignalType.BUY,
            pair=pair,
            price=level.price,
            timestamp=datetime.now(),
            module=TradingModule.GRID,
            confidence=1.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"Grid level {level.level_number} buy",
            metadata={
                "grid_level": level.level_number,
                "lot_size": lot_size,
                "grid_spacing_pips": grid_config.grid_spacing_pips
            }
        )

    def _create_close_signal(
        self,
        pair: str,
        level: GridLevel,
        market_data: MarketData,
        reason: str
    ) -> Optional[Signal]:
        """Create a close signal for an existing position."""
        if level.buy_order is None:
            return None
        
        return Signal(
            signal_type=SignalType.CLOSE,
            pair=pair,
            price=market_data.mid_price,
            timestamp=datetime.now(),
            module=TradingModule.GRID,
            confidence=1.0,
            reason=reason,
            metadata={
                "grid_level": level.level_number,
                "order_id": level.buy_order.order_id
            }
        )

    def _generate_close_signals(self, pair: str, reason: str) -> List[Signal]:
        """Generate signals to close all positions for a pair."""
        signals = []
        
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            return signals
        
        for level in grid_config.levels:
            if level.buy_order and level.buy_order.is_open:
                signal = Signal(
                    signal_type=SignalType.CLOSE,
                    pair=pair,
                    price=0,  # Market close
                    timestamp=datetime.now(),
                    module=TradingModule.GRID,
                    confidence=1.0,
                    reason=f"Grid break: {reason}",
                    metadata={
                        "grid_level": level.level_number,
                        "order_id": level.buy_order.order_id,
                        "close_all": True
                    }
                )
                signals.append(signal)
        
        self.logger.warning(f"[GRID] Generated {len(signals)} close signals due to {reason}")
        return signals

    def register_order(self, pair: str, level_number: int, order: Order) -> None:
        """
        Register an executed order with a grid level.
        
        Args:
            pair: Currency pair.
            level_number: Grid level number.
            order: Executed order.
        """
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            self.logger.error(f"[GRID] Cannot register order: No grid for {pair}")
            return
        
        for level in grid_config.levels:
            if level.level_number == level_number:
                if order.side == OrderSide.BUY:
                    level.buy_order = order
                else:
                    level.sell_order = order
                
                self.logger.info(
                    f"[GRID] Order registered at level {level_number}: "
                    f"{order.side.value} #{order.order_id}"
                )
                break
        
        if pair not in self._active_orders:
            self._active_orders[pair] = []
        self._active_orders[pair].append(order)

    def close_order(
        self,
        pair: str,
        order_id: str,
        close_price: float,
        pnl: float
    ) -> None:
        """
        Handle order closure.
        
        Args:
            pair: Currency pair.
            order_id: Order ID.
            close_price: Close price.
            pnl: Profit/loss.
        """
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            return
        
        for level in grid_config.levels:
            if level.buy_order and level.buy_order.order_id == order_id:
                level.buy_order.close_price = close_price
                level.buy_order.close_time = datetime.now()
                level.buy_order.pnl = pnl
                level.buy_order.status = OrderStatus.FILLED
                level.buy_order = None  # Clear for next trade
                
                self.logger.info(
                    f"[GRID] Order closed at level {level.level_number}: "
                    f"P&L ${pnl:+.2f}"
                )
                self._state.total_profit += pnl
                break

    def get_grid_state(self, pair: str) -> Optional[GridState]:
        """
        Get current state of a grid.
        
        Args:
            pair: Currency pair.
            
        Returns:
            GridState or None if no grid configured.
        """
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            return None
        
        active_buys = sum(
            1 for level in grid_config.levels 
            if level.buy_order and level.buy_order.is_open
        )
        active_sells = sum(
            1 for level in grid_config.levels 
            if level.sell_order and level.sell_order.is_open
        )
        
        return GridState(
            is_active=True,
            current_price=self._state.current_price,
            active_buy_orders=active_buys,
            active_sell_orders=active_sells,
            total_profit=self._state.total_profit,
            grid_config=grid_config
        )

    def update_price(self, pair: str, price: float) -> None:
        """Update current price for state tracking."""
        self._state.current_price = price

    def deactivate_grid(self, pair: str) -> None:
        """
        Deactivate a grid and mark it for cleanup.
        
        Args:
            pair: Currency pair.
        """
        if pair in self._grid_configs:
            self.logger.info(f"[GRID] Deactivating grid for {pair}")
            del self._grid_configs[pair]
        
        if pair in self._active_orders:
            del self._active_orders[pair]

    def get_active_grids(self) -> List[str]:
        """Get list of pairs with active grids."""
        return list(self._grid_configs.keys())

    def get_grid_summary(self, pair: str) -> Dict:
        """
        Get a summary of the grid configuration and state.
        
        Args:
            pair: Currency pair.
            
        Returns:
            Dictionary with grid summary.
        """
        grid_config = self._grid_configs.get(pair)
        if not grid_config:
            return {}
        
        state = self.get_grid_state(pair)
        
        return {
            "pair": pair,
            "upper_limit": grid_config.upper_limit,
            "lower_limit": grid_config.lower_limit,
            "num_grids": grid_config.num_grids,
            "grid_spacing_pips": grid_config.grid_spacing_pips,
            "profit_per_grid_pips": grid_config.profit_per_grid_pips,
            "stop_loss_pips": grid_config.stop_loss_pips,
            "active_buy_orders": state.active_buy_orders if state else 0,
            "active_sell_orders": state.active_sell_orders if state else 0,
            "total_profit": state.total_profit if state else 0,
            "levels": [
                {
                    "level": level.level_number,
                    "price": level.price,
                    "has_buy_order": level.buy_order is not None,
                    "has_sell_order": level.sell_order is not None,
                    "is_active": level.is_active
                }
                for level in grid_config.levels
            ]
        }
