"""
MetaTrader 5 API Client for IC Markets

This module provides a real broker connection using the official
MetaTrader 5 Python package. Works with IC Markets MT5 accounts.

Prerequisites:
1. Install MetaTrader 5 terminal on your machine (or VPS)
2. Log into your IC Markets MT5 account in the terminal
3. Enable "Algo Trading" in MT5 settings
4. pip install MetaTrader5

Usage:
    from src.mt5_client import MT5Client
    
    client = MT5Client()
    await client.connect(
        login=12345678,
        password="your_password",
        server="ICMarketsSC-Demo"  # or "ICMarketsSC-Live"
    )
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

# MetaTrader5 package - install with: pip install MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from .api_client import BrokerAPIClient, ConnectionError, OrderError, InsufficientMarginError
from .models import (
    Order, OrderType, OrderSide, OrderStatus, MarketData,
    AccountState, Candle
)
from .logger import TradingBotLogger, get_logger


class MT5Client(BrokerAPIClient):
    """
    MetaTrader 5 API Client for IC Markets.
    
    Connects to MT5 terminal running on the same machine.
    Supports both demo and live accounts.
    
    IC Markets MT5 Servers:
    - Demo: ICMarketsSC-Demo
    - Live: ICMarketsSC-Live01, ICMarketsSC-Live02, etc.
    """

    # Symbol mapping (our format -> MT5 format)
    SYMBOL_MAP = {
        "EUR/USD": "EURUSD",
        "USD/JPY": "USDJPY",
        "GBP/USD": "GBPUSD",
        "AUD/USD": "AUDUSD",
        "USD/CAD": "USDCAD",
    }

    # Reverse mapping
    SYMBOL_MAP_REVERSE = {v: k for k, v in SYMBOL_MAP.items()}

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": mt5.TIMEFRAME_M1 if mt5 else 1,
        "5m": mt5.TIMEFRAME_M5 if mt5 else 5,
        "15m": mt5.TIMEFRAME_M15 if mt5 else 15,
        "30m": mt5.TIMEFRAME_M30 if mt5 else 30,
        "1h": mt5.TIMEFRAME_H1 if mt5 else 60,
        "4h": mt5.TIMEFRAME_H4 if mt5 else 240,
        "1d": mt5.TIMEFRAME_D1 if mt5 else 1440,
    }

    def __init__(self, logger: Optional[TradingBotLogger] = None):
        """Initialize MT5 client."""
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
        
        self.logger = logger or get_logger("MT5Client")
        self._connected = False
        self._login = None
        self._server = None

    async def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None
    ) -> bool:
        """
        Connect to MetaTrader 5 terminal.
        
        Args:
            login: MT5 account number (optional if already logged in terminal)
            password: MT5 account password
            server: MT5 server name (e.g., "ICMarketsSC-Demo")
            path: Path to MT5 terminal (optional, auto-detected)
            
        Returns:
            True if connected successfully.
        """
        self.logger.info("[MT5] Initializing MetaTrader 5 connection...")
        
        # Initialize MT5
        init_params = {}
        if path:
            init_params["path"] = path
        if login:
            init_params["login"] = login
        if password:
            init_params["password"] = password
        if server:
            init_params["server"] = server
        
        if init_params:
            initialized = mt5.initialize(**init_params)
        else:
            initialized = mt5.initialize()
        
        if not initialized:
            error = mt5.last_error()
            self.logger.error(f"[MT5] Initialization failed: {error}")
            raise ConnectionError(f"MT5 initialization failed: {error}")
        
        # Get account info to verify connection
        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            self.logger.error(f"[MT5] Failed to get account info: {error}")
            mt5.shutdown()
            raise ConnectionError(f"Failed to get account info: {error}")
        
        self._connected = True
        self._login = account_info.login
        self._server = account_info.server
        
        self.logger.info(f"[MT5] Connected successfully!")
        self.logger.info(f"  Account: {account_info.login}")
        self.logger.info(f"  Server: {account_info.server}")
        self.logger.info(f"  Balance: ${account_info.balance:.2f}")
        self.logger.info(f"  Leverage: 1:{account_info.leverage}")
        
        return True

    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            self.logger.info("[MT5] Disconnected from MetaTrader 5")

    async def get_account_info(self) -> AccountState:
        """Get current account information."""
        self._check_connection()
        
        info = mt5.account_info()
        if info is None:
            raise ConnectionError(f"Failed to get account info: {mt5.last_error()}")
        
        return AccountState(
            balance=info.balance,
            equity=info.equity,
            margin_used=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin_level else float('inf'),
            leverage=info.leverage,
            currency=info.currency
        )

    async def get_market_data(self, pair: str) -> MarketData:
        """Get current market data for a pair."""
        self._check_connection()
        
        symbol = self.SYMBOL_MAP.get(pair, pair.replace("/", ""))
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            # Try to enable the symbol first
            if not mt5.symbol_select(symbol, True):
                raise ConnectionError(f"Symbol {symbol} not available: {mt5.last_error()}")
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise ConnectionError(f"Failed to get tick for {symbol}: {mt5.last_error()}")
        
        return MarketData(
            pair=pair,
            bid=tick.bid,
            ask=tick.ask,
            timestamp=datetime.fromtimestamp(tick.time),
            volume=tick.volume if hasattr(tick, 'volume') else None
        )

    async def get_candles(
        self, pair: str, timeframe: str, count: int
    ) -> List[Candle]:
        """Get historical candle data."""
        self._check_connection()
        
        symbol = self.SYMBOL_MAP.get(pair, pair.replace("/", ""))
        tf = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None:
            raise ConnectionError(f"Failed to get candles: {mt5.last_error()}")
        
        candles = []
        for rate in rates:
            candles.append(Candle(
                pair=pair,
                timeframe=timeframe,
                timestamp=datetime.fromtimestamp(rate['time']),
                open=rate['open'],
                high=rate['high'],
                low=rate['low'],
                close=rate['close'],
                volume=rate['tick_volume']
            ))
        
        return candles

    async def place_order(self, order: Order) -> Order:
        """Place a new order."""
        self._check_connection()
        
        symbol = self.SYMBOL_MAP.get(order.pair, order.pair.replace("/", ""))
        
        # Get symbol info for proper lot sizing
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise OrderError(f"Symbol {symbol} not found")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise OrderError(f"Failed to select symbol {symbol}")
        
        # Determine order type
        if order.side == OrderSide.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": order.lot_size,
            "type": order_type,
            "price": price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,  # Max slippage in points
            "magic": 123456,  # EA identifier
            "comment": f"TradingBot {order.module.value if order.module else 'MANUAL'}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        self.logger.info(f"[MT5] Placing order: {order.side.value} {order.lot_size} {symbol}")
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            raise OrderError(f"Order send failed: {mt5.last_error()}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order rejected: {result.comment} (code: {result.retcode})"
            self.logger.error(f"[MT5] {error_msg}")
            
            if result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                raise InsufficientMarginError(error_msg)
            raise OrderError(error_msg)
        
        # Update order with result
        order.order_id = str(result.order)
        order.fill_price = result.price
        order.fill_time = datetime.now()
        order.status = OrderStatus.FILLED
        
        self.logger.info(
            f"[MT5] Order filled: #{result.order} @ {result.price:.5f}"
        )
        
        return order

    async def modify_order(
        self, order_id: str, stop_loss: float, take_profit: float
    ) -> Order:
        """Modify an existing order's SL/TP."""
        self._check_connection()
        
        # Get position info
        position = mt5.positions_get(ticket=int(order_id))
        if not position:
            raise OrderError(f"Position {order_id} not found")
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(order_id),
            "symbol": position.symbol,
            "sl": stop_loss,
            "tp": take_profit,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise OrderError(f"Modify failed: {result.comment}")
        
        # Return updated order
        order = Order(
            order_id=order_id,
            pair=self.SYMBOL_MAP_REVERSE.get(position.symbol, position.symbol),
            side=OrderSide.BUY if position.type == mt5.POSITION_TYPE_BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            lot_size=position.volume,
            entry_price=position.price_open,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.FILLED
        )
        
        return order

    async def close_order(self, order_id: str) -> Order:
        """Close an existing position."""
        self._check_connection()
        
        # Get position info
        position = mt5.positions_get(ticket=int(order_id))
        if not position:
            raise OrderError(f"Position {order_id} not found")
        
        position = position[0]
        symbol = position.symbol
        
        # Determine close order type (opposite of position)
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": int(order_id),
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "TradingBot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise OrderError(f"Close failed: {result.comment}")
        
        # Create closed order
        order = Order(
            order_id=order_id,
            pair=self.SYMBOL_MAP_REVERSE.get(symbol, symbol),
            side=OrderSide.BUY if position.type == mt5.POSITION_TYPE_BUY else OrderSide.SELL,
            order_type=OrderType.MARKET,
            lot_size=position.volume,
            entry_price=position.price_open,
            stop_loss=position.sl,
            take_profit=position.tp,
            close_price=result.price,
            close_time=datetime.now(),
            pnl=position.profit,
            status=OrderStatus.FILLED
        )
        
        self.logger.info(f"[MT5] Position closed: #{order_id} P&L: ${position.profit:.2f}")
        
        return order

    async def get_open_orders(self) -> List[Order]:
        """Get all open positions."""
        self._check_connection()
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        orders = []
        for pos in positions:
            order = Order(
                order_id=str(pos.ticket),
                pair=self.SYMBOL_MAP_REVERSE.get(pos.symbol, pos.symbol),
                side=OrderSide.BUY if pos.type == mt5.POSITION_TYPE_BUY else OrderSide.SELL,
                order_type=OrderType.MARKET,
                lot_size=pos.volume,
                entry_price=pos.price_open,
                stop_loss=pos.sl,
                take_profit=pos.tp,
                fill_price=pos.price_open,
                fill_time=datetime.fromtimestamp(pos.time),
                status=OrderStatus.FILLED
            )
            order.metadata["current_price"] = pos.price_current
            order.metadata["profit"] = pos.profit
            orders.append(order)
        
        return orders

    def _check_connection(self) -> None:
        """Verify connection is established."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")


# Factory function to create MT5 client
def create_mt5_client(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: str = "ICMarketsSC-Demo"
) -> MT5Client:
    """
    Create and configure MT5 client.
    
    Args:
        login: MT5 account number
        password: MT5 password
        server: MT5 server (default: IC Markets Demo)
        
    Returns:
        Configured MT5Client instance.
    """
    if not MT5_AVAILABLE:
        raise ImportError(
            "MetaTrader5 package not installed.\n"
            "Install with: pip install MetaTrader5\n"
            "Note: Only works on Windows with MT5 terminal installed."
        )
    
    return MT5Client()
