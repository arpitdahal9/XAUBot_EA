"""
API Client for Broker Connection

Handles all communication with the broker API including:
- Account information retrieval
- Market data fetching
- Order placement and management
- Connection health monitoring
- Dry-run mode for testing

Includes exponential backoff retry logic and comprehensive error handling.
"""

import asyncio
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

from .models import (
    Order, OrderType, OrderSide, OrderStatus, MarketData,
    AccountState, Candle, Signal, SignalType
)
from .config import APIConfig, DryRunConfig, get_config
from .logger import TradingBotLogger, get_logger, TradeLogEntry


@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ConnectionError(Exception):
    """Raised when API connection fails."""
    pass


class OrderError(Exception):
    """Raised when order placement/modification fails."""
    pass


class InsufficientMarginError(OrderError):
    """Raised when margin is insufficient for order."""
    pass


class BrokerAPIClient(ABC):
    """
    Abstract base class for broker API clients.
    
    Defines the interface that all broker implementations must follow.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the broker."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountState:
        """Get current account information."""
        pass

    @abstractmethod
    async def get_market_data(self, pair: str) -> MarketData:
        """Get current market data for a pair."""
        pass

    @abstractmethod
    async def get_candles(
        self, pair: str, timeframe: str, count: int
    ) -> List[Candle]:
        """Get historical candle data."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place a new order."""
        pass

    @abstractmethod
    async def modify_order(
        self, order_id: str, stop_loss: float, take_profit: float
    ) -> Order:
        """Modify an existing order."""
        pass

    @abstractmethod
    async def close_order(self, order_id: str) -> Order:
        """Close an existing order."""
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        pass


class DryRunAPIClient(BrokerAPIClient):
    """
    Simulated API client for testing without real trading.
    
    Simulates:
    - Market data with realistic spreads and movements
    - Order execution with simulated slippage
    - Account balance and equity calculations
    - Position tracking
    
    Use this for:
    - Strategy testing
    - System validation
    - Risk-free development
    """

    def __init__(
        self,
        config: Optional[DryRunConfig] = None,
        api_config: Optional[APIConfig] = None,
        logger: Optional[TradingBotLogger] = None,
        initial_balance: float = 1000.0
    ):
        """
        Initialize the dry run client.
        
        Args:
            config: Dry run configuration.
            api_config: API configuration for timeouts/retries.
            logger: Logger instance.
            initial_balance: Starting account balance.
        """
        self.config = config or get_config().dry_run
        self.api_config = api_config or get_config().api
        self.logger = logger or get_logger("DryRunAPI")
        
        # Account state
        self._balance = initial_balance
        self._equity = initial_balance
        self._margin_used = 0.0
        self._leverage = 30
        
        # Market simulation
        self._base_prices: Dict[str, float] = {
            "EUR/USD": 1.0850,
            "USD/JPY": 150.50,
            "GBP/USD": 1.2650,
            "AUD/USD": 0.6550,
            "USD/CAD": 1.3550
        }
        self._price_volatility = 0.0005  # 5 pips typical movement
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        
        # Connection state
        self._connected = False
        
        self.logger.info("[DRY_RUN] Dry Run API Client initialized")
        self.logger.info(f"  Initial Balance: ${initial_balance:,.2f}")
        self.logger.info(f"  Simulated Spread: {self.config.simulated_spread_pips} pips")
        self.logger.info(f"  Simulated Slippage: {self.config.simulated_slippage_pips} pips")

    async def connect(self) -> bool:
        """Simulate connection establishment."""
        self.logger.info("[DRY_RUN] Connecting to simulated broker...")
        await asyncio.sleep(0.5)  # Simulate connection delay
        self._connected = True
        self.logger.info("[DRY_RUN] Connected successfully (DRY RUN MODE)")
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self.logger.info("[DRY_RUN] Disconnecting from simulated broker...")
        self._connected = False

    async def get_account_info(self) -> AccountState:
        """Get simulated account information."""
        self._check_connection()
        
        # Update equity based on open positions
        unrealized_pnl = self._calculate_unrealized_pnl()
        self._equity = self._balance + unrealized_pnl
        
        # Calculate margin used
        self._margin_used = sum(
            (order.lot_size * 100000) / self._leverage
            for order in self._orders.values()
            if order.is_open
        )
        
        free_margin = self._equity - self._margin_used
        margin_level = (self._equity / self._margin_used * 100) if self._margin_used > 0 else float('inf')
        
        return AccountState(
            balance=self._balance,
            equity=self._equity,
            margin_used=self._margin_used,
            free_margin=max(0, free_margin),
            margin_level=margin_level,
            leverage=self._leverage
        )

    async def get_market_data(self, pair: str) -> MarketData:
        """Get simulated market data with realistic movement."""
        self._check_connection()
        
        if pair not in self._base_prices:
            raise ValueError(f"Unsupported pair: {pair}")
        
        # Simulate price movement (random walk)
        change = random.gauss(0, self._price_volatility)
        self._base_prices[pair] += change
        
        base_price = self._base_prices[pair]
        
        # Calculate spread
        pip_size = 0.01 if "JPY" in pair else 0.0001
        spread = self.config.simulated_spread_pips * pip_size
        
        bid = base_price
        ask = base_price + spread
        
        return MarketData(
            pair=pair,
            bid=bid,
            ask=ask,
            timestamp=datetime.now()
        )

    async def get_candles(
        self, pair: str, timeframe: str, count: int
    ) -> List[Candle]:
        """Generate simulated historical candles."""
        self._check_connection()
        
        if pair not in self._base_prices:
            raise ValueError(f"Unsupported pair: {pair}")
        
        candles = []
        current_price = self._base_prices[pair]
        
        # Generate candles going backwards in time
        for i in range(count):
            # Random OHLC generation
            volatility = self._price_volatility * 2
            open_price = current_price + random.gauss(0, volatility)
            close_price = current_price + random.gauss(0, volatility)
            high_price = max(open_price, close_price) + abs(random.gauss(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(random.gauss(0, volatility * 0.5))
            
            candle = Candle(
                pair=pair,
                timeframe=timeframe,
                timestamp=datetime.now(),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.randint(100, 1000)
            )
            candles.insert(0, candle)
            
            # Move price for next candle
            current_price = open_price
        
        return candles

    async def place_order(self, order: Order) -> Order:
        """
        Simulate order placement.
        
        Includes:
        - Slippage simulation
        - Margin validation
        - Fill confirmation
        """
        self._check_connection()
        start_time = time.time()
        
        self.logger.info(
            f"[DRY_RUN] Placing order: {order.side.value} {order.lot_size} {order.pair}"
        )
        
        # Simulate execution delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Calculate margin required
        margin_required = (order.lot_size * 100000) / self._leverage
        available_margin = self._equity - self._margin_used
        
        if margin_required > available_margin:
            latency = (time.time() - start_time) * 1000
            self.logger.error(
                f"[DRY_RUN] Order rejected: Insufficient margin. "
                f"Required: ${margin_required:.2f}, Available: ${available_margin:.2f}"
            )
            raise InsufficientMarginError(
                f"Insufficient margin. Required: ${margin_required:.2f}, Available: ${available_margin:.2f}"
            )
        
        # Apply slippage
        pip_size = 0.01 if "JPY" in order.pair else 0.0001
        slippage = self.config.simulated_slippage_pips * pip_size
        
        if order.side == OrderSide.BUY:
            fill_price = order.entry_price + slippage
        else:
            fill_price = order.entry_price - slippage
        
        # Update order with fill information
        order.fill_price = fill_price
        order.fill_time = datetime.now()
        order.status = OrderStatus.FILLED
        order.slippage_pips = self.config.simulated_slippage_pips
        
        # Generate order ID
        self._order_counter += 1
        order.order_id = f"DRY{self._order_counter:06d}"
        
        # Store order
        self._orders[order.order_id] = order
        
        latency = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"[DRY_RUN] Order filled: #{order.order_id} {order.side.value} "
            f"{order.lot_size} {order.pair} @ {fill_price:.5f} "
            f"(Slippage: {order.slippage_pips:.1f} pips, Latency: {latency:.0f}ms)"
        )
        
        # Log trade entry
        trade_entry = TradeLogEntry(
            order_id=order.order_id,
            action="OPEN",
            trade_type=order.side.value,
            pair=order.pair,
            lot_size=order.lot_size,
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            slippage_pips=order.slippage_pips,
            module="DRY_RUN"
        )
        self.logger.info(f"[DRY_RUN] {trade_entry.format_message()}")
        
        return order

    async def modify_order(
        self, order_id: str, stop_loss: float, take_profit: float
    ) -> Order:
        """Modify an existing order's SL/TP."""
        self._check_connection()
        
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        
        order = self._orders[order_id]
        old_sl = order.stop_loss
        old_tp = order.take_profit
        
        order.stop_loss = stop_loss
        order.take_profit = take_profit
        
        self.logger.info(
            f"[DRY_RUN] Order modified: #{order_id} "
            f"SL: {old_sl:.5f} → {stop_loss:.5f}, "
            f"TP: {old_tp:.5f} → {take_profit:.5f}"
        )
        
        return order

    async def close_order(self, order_id: str) -> Order:
        """Close an existing order."""
        self._check_connection()
        
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        
        order = self._orders[order_id]
        
        if not order.is_open:
            raise OrderError(f"Order already closed: {order_id}")
        
        # Get current market price
        market_data = await self.get_market_data(order.pair)
        
        if order.side == OrderSide.BUY:
            close_price = market_data.bid  # Close buy at bid
        else:
            close_price = market_data.ask  # Close sell at ask
        
        # Calculate P&L
        pip_size = 0.01 if "JPY" in order.pair else 0.0001
        entry = order.fill_price or order.entry_price
        
        if order.side == OrderSide.BUY:
            pips = (close_price - entry) / pip_size
        else:
            pips = (entry - close_price) / pip_size
        
        pip_value = self._get_pip_value(order.pair, order.lot_size)
        pnl = pips * pip_value
        
        # Update order
        order.close_price = close_price
        order.close_time = datetime.now()
        order.pnl = pnl
        order.status = OrderStatus.FILLED
        
        # Update account balance
        self._balance += pnl
        
        self.logger.info(
            f"[DRY_RUN] Order closed: #{order_id} {order.side.value} "
            f"{order.lot_size} {order.pair} @ {close_price:.5f}, "
            f"P&L: ${pnl:+.2f} ({pips:+.1f} pips)"
        )
        
        return order

    async def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        self._check_connection()
        return [order for order in self._orders.values() if order.is_open]

    def _check_connection(self) -> None:
        """Verify connection is established."""
        if not self._connected:
            raise ConnectionError("Not connected to broker")

    def _get_pip_value(self, pair: str, lot_size: float) -> float:
        """
        Get pip value for a currency pair.
        
        Args:
            pair: Currency pair
            lot_size: Position size in lots
            
        Returns:
            Pip value in USD for the given lot size
        """
        # Pip values per standard lot (USD account)
        pip_values = {
            "EUR/USD": 10.0, "EURUSD": 10.0,
            "GBP/USD": 10.0, "GBPUSD": 10.0,
            "AUD/USD": 10.0, "AUDUSD": 10.0,
            "NZD/USD": 10.0, "NZDUSD": 10.0,
            "USD/CHF": 10.0, "USDCHF": 10.0,
            "USD/CAD": 7.40, "USDCAD": 7.40,
            "USD/JPY": 6.67, "USDJPY": 6.67,
            "XAU/USD": 10.0, "XAUUSD": 10.0,
            "GOLD": 10.0,
        }
        pair_clean = pair.upper().replace("/", "")
        base_value = pip_values.get(pair, pip_values.get(pair_clean, 10.0))
        return base_value * lot_size

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L from open positions."""
        total_pnl = 0.0
        
        for order in self._orders.values():
            if not order.is_open:
                continue
            
            # Get current price (use stored base price)
            current_price = self._base_prices.get(order.pair, 0)
            if current_price == 0:
                continue
            
            entry = order.fill_price or order.entry_price
            pip_size = 0.01 if "JPY" in order.pair else 0.0001
            
            if order.side == OrderSide.BUY:
                pips = (current_price - entry) / pip_size
            else:
                pips = (entry - current_price) / pip_size
            
            pip_value = self._get_pip_value(order.pair, order.lot_size)
            total_pnl += pips * pip_value
        
        return total_pnl

    def set_market_price(self, pair: str, price: float) -> None:
        """
        Manually set market price (for testing).
        
        Args:
            pair: Currency pair.
            price: New price.
        """
        if pair in self._base_prices:
            self._base_prices[pair] = price
            self.logger.debug(f"[DRY_RUN] Price set: {pair} = {price:.5f}")


class APIClientWithRetry:
    """
    Wrapper that adds retry logic with exponential backoff.
    
    Handles transient failures gracefully by:
    - Retrying failed requests
    - Exponential backoff between retries
    - Logging all retry attempts
    """

    def __init__(
        self,
        client: BrokerAPIClient,
        config: Optional[APIConfig] = None,
        logger: Optional[TradingBotLogger] = None
    ):
        """
        Initialize the retry wrapper.
        
        Args:
            client: Underlying API client.
            config: API configuration.
            logger: Logger instance.
        """
        self.client = client
        self.config = config or get_config().api
        self.logger = logger or get_logger("APIRetry")

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            operation: Operation name for logging.
            func: Async function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.
            
        Returns:
            Result of the function.
            
        Raises:
            Last exception if all retries fail.
        """
        last_exception = None
        
        for attempt in range(1, self.config.retry_max_attempts + 1):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
                latency = (time.time() - start_time) * 1000
                
                self.logger.debug(
                    f"[API] {operation} completed (Latency: {latency:.0f}ms)"
                )
                return result
                
            except asyncio.TimeoutError:
                last_exception = ConnectionError(f"Timeout after {self.config.timeout_seconds}s")
                self.logger.error(
                    f"[API] {operation} timeout (attempt {attempt}/{self.config.retry_max_attempts})"
                )
                
            except ConnectionError as e:
                last_exception = e
                self.logger.error(
                    f"[API] {operation} connection error: {e} "
                    f"(attempt {attempt}/{self.config.retry_max_attempts})"
                )
                
            except Exception as e:
                last_exception = e
                self.logger.error(
                    f"[API] {operation} failed: {type(e).__name__}: {e} "
                    f"(attempt {attempt}/{self.config.retry_max_attempts})"
                )
            
            # Don't retry on last attempt
            if attempt < self.config.retry_max_attempts:
                # Exponential backoff
                delay = (self.config.retry_backoff_multiplier ** attempt)
                self.logger.warning(f"[API] Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All retries failed
        self.logger.critical(
            f"[API] {operation} failed after {self.config.retry_max_attempts} attempts"
        )
        raise last_exception

    async def connect(self) -> bool:
        """Connect with retry."""
        return await self._execute_with_retry(
            "Connect",
            self.client.connect
        )

    async def disconnect(self) -> None:
        """Disconnect (no retry needed)."""
        await self.client.disconnect()

    async def get_account_info(self) -> AccountState:
        """Get account info with retry."""
        return await self._execute_with_retry(
            "GetAccountInfo",
            self.client.get_account_info
        )

    async def get_market_data(self, pair: str) -> MarketData:
        """Get market data with retry."""
        return await self._execute_with_retry(
            f"GetMarketData({pair})",
            self.client.get_market_data,
            pair
        )

    async def get_candles(
        self, pair: str, timeframe: str, count: int
    ) -> List[Candle]:
        """Get candles with retry."""
        return await self._execute_with_retry(
            f"GetCandles({pair}, {timeframe}, {count})",
            self.client.get_candles,
            pair, timeframe, count
        )

    async def place_order(self, order: Order) -> Order:
        """Place order with retry."""
        return await self._execute_with_retry(
            f"PlaceOrder({order.pair})",
            self.client.place_order,
            order
        )

    async def modify_order(
        self, order_id: str, stop_loss: float, take_profit: float
    ) -> Order:
        """Modify order with retry."""
        return await self._execute_with_retry(
            f"ModifyOrder({order_id})",
            self.client.modify_order,
            order_id, stop_loss, take_profit
        )

    async def close_order(self, order_id: str) -> Order:
        """Close order with retry."""
        return await self._execute_with_retry(
            f"CloseOrder({order_id})",
            self.client.close_order,
            order_id
        )

    async def get_open_orders(self) -> List[Order]:
        """Get open orders with retry."""
        return await self._execute_with_retry(
            "GetOpenOrders",
            self.client.get_open_orders
        )


async def check_api_connection(client: BrokerAPIClient, logger: TradingBotLogger) -> bool:
    """
    Perform health check on API connection.
    
    Args:
        client: API client to check.
        logger: Logger instance.
        
    Returns:
        True if connection is healthy, False otherwise.
    """
    config = get_config().api
    max_retries = config.retry_max_attempts
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            start_time = time.time()
            account_info = await client.get_account_info()
            latency = (time.time() - start_time) * 1000
            
            logger.debug(f"[API] Connection: HEALTHY (Latency: {latency:.0f}ms)")
            logger.debug(f"  Balance: ${account_info.balance:.2f}")
            logger.debug(f"  Margin Level: {account_info.margin_level:.1f}%")
            return True
            
        except ConnectionError as e:
            retry_count += 1
            logger.error(
                f"[API] Connection Failed: {type(e).__name__} "
                f"(retry {retry_count}/{max_retries})"
            )
            
            if retry_count >= max_retries:
                logger.critical("[API] Connection Lost: Max retries exceeded")
                return False
            
            await asyncio.sleep(config.retry_backoff_multiplier ** retry_count)
    
    return False


def create_api_client(dry_run: bool = True) -> BrokerAPIClient:
    """
    Factory function to create appropriate API client.
    
    Args:
        dry_run: If True, create dry run client. Otherwise, create live client.
        
    Returns:
        Configured API client.
    """
    config = get_config()
    logger = get_logger("APIClient")
    
    if dry_run or config.dry_run.enabled:
        logger.info("[API] Creating Dry Run API Client")
        client = DryRunAPIClient(
            config=config.dry_run,
            api_config=config.api,
            logger=logger,
            initial_balance=config.risk_management.account_balance
        )
    else:
        # In a real implementation, this would create a live API client
        # for the specific broker (e.g., IC Markets)
        logger.warning("[API] Live trading not implemented - falling back to dry run")
        client = DryRunAPIClient(
            config=config.dry_run,
            api_config=config.api,
            logger=logger,
            initial_balance=config.risk_management.account_balance
        )
    
    # Wrap with retry logic
    return APIClientWithRetry(client, config.api, logger)
