"""
Data Models for Trading Bot

Defines all data structures used throughout the trading bot,
including orders, positions, market data, and account state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class TradingModule(Enum):
    """Trading module identifiers."""
    GRID = "GRID"
    TREND = "TREND"
    RISK_MGMT = "RISK_MGMT"


class TradeState(Enum):
    """
    Trade management state machine states.
    Implements Model A: Balanced Asymmetry partial exit strategy.
    """
    # Entry phase
    PENDING_ENTRY = "PENDING_ENTRY"       # Order placed, waiting for fill
    ENTRY_FILLED = "ENTRY_FILLED"         # Entry filled, SL at -1R
    
    # Breakeven phase
    BREAKEVEN_TRIGGERED = "BREAKEVEN_TRIGGERED"  # +1.2R reached, SL moved to -0.1R
    
    # Partial exit phases
    TP1_HIT = "TP1_HIT"                   # +2R reached, 50% closed, SL at +0.2R
    TP2_HIT = "TP2_HIT"                   # +3R reached, 30% closed, runner trailing
    
    # Exit states
    STOPPED_OUT = "STOPPED_OUT"           # Hit stop loss
    RUNNER_STOPPED = "RUNNER_STOPPED"     # Runner hit trailing stop
    FULLY_CLOSED = "FULLY_CLOSED"         # All positions closed
    CANCELLED = "CANCELLED"               # Trade cancelled before fill


class CorrelationGroup(Enum):
    """Currency correlation groups to avoid simultaneous exposure."""
    USD_LONG = "USD_LONG"     # Trades that profit when USD strengthens
    USD_SHORT = "USD_SHORT"   # Trades that profit when USD weakens
    JPY_LONG = "JPY_LONG"     # Trades that profit when JPY strengthens
    JPY_SHORT = "JPY_SHORT"   # Trades that profit when JPY weakens
    GOLD = "GOLD"             # Gold trades


@dataclass
class MarketData:
    """Real-time market data for a trading pair."""
    pair: str
    bid: float
    ask: float
    timestamp: datetime
    volume: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

    @property
    def spread(self) -> float:
        """Calculate spread in price units."""
        return self.ask - self.bid

    @property
    def spread_pips(self) -> float:
        """
        Calculate spread in pips.
        Assumes 4 decimal places for most pairs, 2 for JPY pairs.
        """
        if "JPY" in self.pair:
            return (self.ask - self.bid) * 100
        return (self.ask - self.bid) * 10000

    @property
    def mid_price(self) -> float:
        """Calculate mid-point price."""
        return (self.bid + self.ask) / 2


@dataclass
class Candle:
    """OHLC candle data."""
    pair: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """Calculate candle range (high - low)."""
        return self.high - self.low


@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    slippage_pips: float = 0.0
    module: TradingModule = TradingModule.GRID
    grid_level: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if order is currently open."""
        return self.status in (OrderStatus.OPEN, OrderStatus.FILLED)

    @property
    def is_closed(self) -> bool:
        """Check if order has been closed."""
        return self.close_time is not None

    @property
    def unrealized_pnl(self) -> Optional[float]:
        """Calculate unrealized P&L based on current price in metadata."""
        if not self.is_open or "current_price" not in self.metadata:
            return None
        
        current = self.metadata["current_price"]
        entry = self.fill_price or self.entry_price
        
        if self.side == OrderSide.BUY:
            pips = (current - entry) * (100 if "JPY" in self.pair else 10000)
        else:
            pips = (entry - current) * (100 if "JPY" in self.pair else 10000)
        
        pip_value = 0.10 if "JPY" not in self.pair else 0.01
        return pips * pip_value * self.lot_size * 100

    @property
    def duration(self) -> Optional[float]:
        """Calculate trade duration in seconds."""
        if self.fill_time and self.close_time:
            return (self.close_time - self.fill_time).total_seconds()
        return None


@dataclass
class Position:
    """Aggregated position for a trading pair."""
    pair: str
    orders: List[Order] = field(default_factory=list)
    
    @property
    def total_lots(self) -> float:
        """Calculate total lot size across all orders."""
        buy_lots = sum(o.lot_size for o in self.orders if o.side == OrderSide.BUY and o.is_open)
        sell_lots = sum(o.lot_size for o in self.orders if o.side == OrderSide.SELL and o.is_open)
        return buy_lots - sell_lots

    @property
    def avg_entry_price(self) -> Optional[float]:
        """Calculate average entry price."""
        open_orders = [o for o in self.orders if o.is_open]
        if not open_orders:
            return None
        
        total_value = sum((o.fill_price or o.entry_price) * o.lot_size for o in open_orders)
        total_lots = sum(o.lot_size for o in open_orders)
        return total_value / total_lots if total_lots > 0 else None

    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(o.unrealized_pnl or 0 for o in self.orders if o.is_open)


@dataclass
class AccountState:
    """Current account state."""
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_level: float  # Percentage
    leverage: int
    currency: str = "USD"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def margin_available_pct(self) -> float:
        """Calculate available margin as percentage."""
        if self.equity == 0:
            return 0
        return (self.free_margin / self.equity) * 100


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: datetime
    starting_balance: float
    ending_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    commissions: float = 0.0
    swaps: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def net_profit(self) -> float:
        """Calculate net profit."""
        return self.gross_profit - self.gross_loss - self.commissions - abs(self.swaps)

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0
        return self.gross_profit / self.gross_loss

    @property
    def avg_win(self) -> float:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades

    @property
    def avg_loss(self) -> float:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades


@dataclass
class GridLevel:
    """Individual grid level for grid trading."""
    level_number: int
    price: float
    buy_order: Optional[Order] = None
    sell_order: Optional[Order] = None
    is_active: bool = True

    @property
    def has_open_position(self) -> bool:
        """Check if this grid level has an open position."""
        return (
            (self.buy_order is not None and self.buy_order.is_open) or
            (self.sell_order is not None and self.sell_order.is_open)
        )


@dataclass
class GridConfig:
    """Grid trading configuration for a specific setup."""
    pair: str
    upper_limit: float
    lower_limit: float
    num_grids: int
    investment_per_grid: float
    profit_per_grid_pips: float
    stop_loss_pips: float
    levels: List[GridLevel] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def grid_spacing(self) -> float:
        """Calculate grid spacing in price units."""
        return (self.upper_limit - self.lower_limit) / self.num_grids

    @property
    def grid_spacing_pips(self) -> float:
        """Calculate grid spacing in pips."""
        if "JPY" in self.pair:
            return self.grid_spacing * 100
        return self.grid_spacing * 10000

    def initialize_levels(self) -> None:
        """Initialize grid levels based on configuration."""
        self.levels = []
        for i in range(self.num_grids + 1):
            price = self.lower_limit + (i * self.grid_spacing)
            self.levels.append(GridLevel(level_number=i, price=price))


@dataclass
class Signal:
    """Trading signal generated by strategy modules."""
    signal_type: SignalType
    pair: str
    price: float
    timestamp: datetime
    module: TradingModule
    confidence: float = 1.0  # 0.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)

    @property
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type == SignalType.CLOSE


@dataclass
class ManagedTrade:
    """
    A trade managed by the state machine with partial exits.
    Implements Model A: Balanced Asymmetry (50/30/20 split).
    """
    trade_id: str
    pair: str
    side: OrderSide
    entry_price: float
    initial_stop_loss: float
    initial_lot_size: float
    
    # R-value (risk per trade in price units)
    r_value: float
    
    # State tracking
    state: TradeState = TradeState.PENDING_ENTRY
    current_stop_loss: float = 0.0
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = 0.0
    highest_close_since_tp1: float = 0.0
    
    # Position tracking
    remaining_lot_size: float = 0.0
    tp1_lot_size: float = 0.0  # 50%
    tp2_lot_size: float = 0.0  # 30%
    runner_lot_size: float = 0.0  # 20%
    
    # Partial exit prices
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    breakeven_trigger_price: float = 0.0
    
    # Realized P&L from partial exits
    realized_pnl: float = 0.0
    tp1_pnl: float = 0.0
    tp2_pnl: float = 0.0
    runner_pnl: float = 0.0
    
    # ATR for trailing
    current_atr: float = 0.0
    trailing_stop: float = 0.0
    
    # Timestamps
    entry_time: datetime = field(default_factory=datetime.now)
    tp1_time: Optional[datetime] = None
    tp2_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Risk tracking
    risk_amount_usd: float = 0.0  # Dollar amount risked
    
    # Metadata
    module: TradingModule = TradingModule.TREND
    correlation_group: Optional[CorrelationGroup] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Initialize basic fields after creation.
        
        NOTE: Lot sizing and TP price calculation is done by TradeManager.create_trade()
        which is the SINGLE SOURCE OF TRUTH for these values. This avoids config desync.
        """
        # Initialize basic tracking fields only
        if self.current_stop_loss == 0.0:
            self.current_stop_loss = self.initial_stop_loss
        if self.remaining_lot_size == 0.0:
            self.remaining_lot_size = self.initial_lot_size
        
        # Lot sizes (tp1_lot_size, tp2_lot_size, runner_lot_size) are set by TradeManager
        # TP prices (tp1_price, tp2_price, breakeven_trigger_price) are set by TradeManager
        # This ensures config values (tp1_r, tp2_r, breakeven_r) are always used correctly
    
    @property
    def current_r_multiple(self) -> float:
        """Calculate current R-multiple based on price."""
        current_price = self.metadata.get("current_price", self.entry_price)
        if self.r_value == 0:
            return 0.0
        
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) / self.r_value
        else:
            return (self.entry_price - current_price) / self.r_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L on remaining position."""
        current_price = self.metadata.get("current_price", self.entry_price)
        pip_multiplier = 100 if "JPY" in self.pair else 10000
        
        if self.side == OrderSide.BUY:
            pips = (current_price - self.entry_price) * pip_multiplier
        else:
            pips = (self.entry_price - current_price) * pip_multiplier
        
        # Get pip value based on pair
        pip_value = self._get_pip_value_per_lot() * self.remaining_lot_size
        return pips * pip_value
    
    def _get_pip_value_per_lot(self) -> float:
        """Get pip value per standard lot based on currency pair."""
        # Pip values for USD account (per standard lot)
        pip_values = {
            "EUR/USD": 10.0, "EURUSD": 10.0,
            "GBP/USD": 10.0, "GBPUSD": 10.0,
            "AUD/USD": 10.0, "AUDUSD": 10.0,
            "NZD/USD": 10.0, "NZDUSD": 10.0,
            "USD/CHF": 10.0, "USDCHF": 10.0,
            "USD/CAD": 7.40, "USDCAD": 7.40,  # ~10/1.35
            "USD/JPY": 6.67, "USDJPY": 6.67,  # ~1000/150
            "XAU/USD": 10.0, "XAUUSD": 10.0,  # Gold: $10 per pip per lot
            "GOLD": 10.0,
        }
        pair_clean = self.pair.upper().replace("/", "")
        return pip_values.get(self.pair, pip_values.get(pair_clean, 10.0))
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_active(self) -> bool:
        """Check if trade is still active."""
        return self.state not in (
            TradeState.STOPPED_OUT,
            TradeState.RUNNER_STOPPED,
            TradeState.FULLY_CLOSED,
            TradeState.CANCELLED
        )
    
    @property
    def is_in_profit(self) -> bool:
        """Check if trade is currently in profit."""
        return self.current_r_multiple > 0
    
    def get_breakeven_stop(self, offset_r: float = 0.1) -> float:
        """Calculate breakeven stop with offset."""
        if self.side == OrderSide.BUY:
            return self.entry_price - (offset_r * self.r_value)
        else:
            return self.entry_price + (offset_r * self.r_value)
    
    def get_post_tp1_stop(self, lock_r: float = 0.2) -> float:
        """Calculate stop loss after TP1 (locks in profit)."""
        if self.side == OrderSide.BUY:
            return self.entry_price + (lock_r * self.r_value)
        else:
            return self.entry_price - (lock_r * self.r_value)
    
    def calculate_trailing_stop(self, atr: float, k: float) -> float:
        """
        Calculate ATR-based trailing stop for runner.
        
        Args:
            atr: Current ATR value
            k: ATR multiplier (1.8 for forex, 2.2 for gold)
        
        Returns:
            New trailing stop price
        """
        if self.side == OrderSide.BUY:
            return self.highest_close_since_tp1 - (k * atr)
        else:
            return self.highest_close_since_tp1 + (k * atr)


@dataclass
class RiskMetrics:
    """Current risk metrics for portfolio management."""
    total_open_risk_percent: float = 0.0
    total_open_risk_usd: float = 0.0
    open_trade_count: int = 0
    daily_realized_pnl: float = 0.0
    daily_r_pnl: float = 0.0
    weekly_realized_pnl: float = 0.0
    weekly_r_pnl: float = 0.0
    active_correlation_groups: List[CorrelationGroup] = field(default_factory=list)
    
    # Drawdown tracking
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_percent: float = 0.0
    
    # Trading status
    can_trade: bool = True
    block_reason: str = ""
    daily_limit_hit: bool = False
    weekly_limit_hit: bool = False
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SpreadVolatilityData:
    """Spread and volatility filter data."""
    current_spread_pips: float = 0.0
    average_spread_pips: float = 0.0
    spread_ratio: float = 0.0
    
    current_atr: float = 0.0
    median_atr: float = 0.0
    atr_ratio: float = 0.0
    
    spread_ok: bool = True
    volatility_ok: bool = True
    
    @property
    def can_trade(self) -> bool:
        """Check if conditions allow trading."""
        return self.spread_ok and self.volatility_ok
    
    @property
    def block_reason(self) -> str:
        """Get reason if trading is blocked."""
        reasons = []
        if not self.spread_ok:
            reasons.append(f"Spread too high ({self.spread_ratio:.1f}x avg)")
        if not self.volatility_ok:
            reasons.append(f"ATR spike ({self.atr_ratio:.1f}x median)")
        return ", ".join(reasons) if reasons else ""
