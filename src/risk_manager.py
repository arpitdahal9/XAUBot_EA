"""
Risk Management Module - NON-NEGOTIABLE

This module enforces strict risk management rules for the trading bot:
- Position sizing based on account risk (0.75% default, adjustable 0.5%-1%)
- Margin safety (skip trade if margin > 30% of free margin)
- Daily/Weekly loss limits (2R daily, 6R weekly)
- Drawdown protection
- Spread filter (1.8x average spread)
- Volatility filter (ATR spike detection)
- Stop-loss and take-profit enforcement

All trades MUST pass through this module before execution.

Position Size Formula:
    LotSize = (Equity × Risk%) / (StopDistance × PipValue)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
from collections import deque
import statistics

from .models import (
    Order, OrderSide, AccountState, DailyStats, MarketData,
    Candle, SpreadVolatilityData
)
from .config import RiskManagementConfig, get_config
from .logger import TradingBotLogger, get_logger


@dataclass
class RiskCheckResult:
    """Result of a risk validation check."""
    passed: bool
    message: str
    warning: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    lot_size: float
    risk_amount: float
    margin_required: float
    is_valid: bool
    message: str
    r_value: float = 0.0  # Risk in price units (entry - SL)
    margin_percent_of_free: float = 0.0  # Margin as % of free margin
    reduced_for_margin: bool = False  # True if lot was reduced due to margin


class RiskManager:
    """
    Risk Management Module for the Trading Bot.
    
    Enforces all risk management rules including:
    - Position sizing (0.75% risk rule, adjustable 0.5%-1%)
    - Margin safety (30% threshold)
    - Daily/Weekly loss limits (2R/6R)
    - Maximum drawdown protection
    - Spread filter (1.8x average)
    - Volatility filter (ATR spike)
    - Stop-loss/take-profit requirements
    
    This module has veto power over ALL trades.
    """

    # Pip values for different currency pairs (per standard lot)
    PIP_VALUES = {
        "EUR/USD": 10.0,
        "GBP/USD": 10.0,
        "AUD/USD": 10.0,
        "USD/CAD": 10.0 / 1.35,  # Approximate, depends on USD/CAD rate
        "USD/JPY": 1000 / 150,   # Approximate, depends on USD/JPY rate
        "XAU/USD": 10.0,         # Gold
    }

    def __init__(
        self,
        config: Optional[RiskManagementConfig] = None,
        logger: Optional[TradingBotLogger] = None
    ):
        """
        Initialize the Risk Manager.
        
        Args:
            config: Risk management configuration. Uses global config if None.
            logger: Logger instance. Creates new one if None.
        """
        self.config = config or get_config().risk_management
        self.logger = logger or get_logger("RiskManager")
        
        # Daily/Weekly tracking
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_r_loss: float = 0.0  # R-based tracking
        self._weekly_r_loss: float = 0.0
        self._daily_trades: List[Order] = []
        self._current_date: date = datetime.now().date()
        self._current_week: int = datetime.now().isocalendar()[1]
        self._starting_balance: float = self.config.account_balance
        self._peak_equity: float = self.config.account_balance
        
        # Spread/Volatility tracking (rolling windows)
        self._spread_history: Dict[str, deque] = {}  # Per-pair spread history
        self._atr_history: Dict[str, deque] = {}     # Per-pair ATR history
        self._lookback_candles = 50  # For spread/volatility average
        
        # Load new config options with defaults
        self._risk_percent = getattr(self.config, 'risk_percent_per_trade', 0.75)
        self._risk_min = getattr(self.config, 'risk_percent_min', 0.5)
        self._risk_max = getattr(self.config, 'risk_percent_max', 1.0)
        self._margin_safety_threshold = getattr(self.config, 'margin_safety_threshold', 30)
        self._daily_loss_limit_r = getattr(self.config, 'daily_loss_limit_r', 2.0)
        self._weekly_loss_limit_r = getattr(self.config, 'weekly_loss_limit_r', 6.0)
        
        # Spread filter settings
        spread_config = getattr(self.config, 'spread_filter', {})
        if isinstance(spread_config, dict):
            self._spread_filter_enabled = spread_config.get('enabled', True)
            self._max_spread_multiplier = spread_config.get('max_spread_multiplier', 1.8)
        else:
            self._spread_filter_enabled = True
            self._max_spread_multiplier = 1.8
        
        # Volatility filter settings
        vol_config = getattr(self.config, 'volatility_filter', {})
        if isinstance(vol_config, dict):
            self._volatility_filter_enabled = vol_config.get('enabled', True)
            self._atr_spike_multiplier = vol_config.get('atr_spike_multiplier', 2.5)
        else:
            self._volatility_filter_enabled = True
            self._atr_spike_multiplier = 2.5
        
        self.logger.info("[RISK_MGMT] Risk Manager initialized (Enhanced)")
        self.logger.info(f"  Account Balance: ${self.config.account_balance:.2f}")
        self.logger.info(f"  Risk Per Trade: {self._risk_percent}% (range: {self._risk_min}%-{self._risk_max}%)")
        self.logger.info(f"  Margin Safety: {self._margin_safety_threshold}% of free margin")
        self.logger.info(f"  Daily Loss Limit: {self._daily_loss_limit_r}R")
        self.logger.info(f"  Weekly Loss Limit: {self._weekly_loss_limit_r}R")
        self.logger.info(f"  Spread Filter: {'ON' if self._spread_filter_enabled else 'OFF'} ({self._max_spread_multiplier}x)")
        self.logger.info(f"  Volatility Filter: {'ON' if self._volatility_filter_enabled else 'OFF'} ({self._atr_spike_multiplier}x)")

    def _check_daily_reset(self) -> None:
        """Reset daily tracking if new day (UTC)."""
        today = datetime.utcnow().date()
        current_week = datetime.utcnow().isocalendar()[1]
        
        if today != self._current_date:
            self.logger.info(f"[RISK_MGMT] New trading day - resetting daily limits")
            self.logger.info(f"[RISK_MGMT]   Previous day R-loss: {self._daily_r_loss:.2f}R")
            self._daily_pnl = 0.0
            self._daily_r_loss = 0.0
            self._daily_trades = []
            self._current_date = today
        
        if current_week != self._current_week:
            self.logger.info(f"[RISK_MGMT] New trading week - resetting weekly limits")
            self.logger.info(f"[RISK_MGMT]   Previous week R-loss: {self._weekly_r_loss:.2f}R")
            self._weekly_pnl = 0.0
            self._weekly_r_loss = 0.0
            self._current_week = current_week

    def get_pip_value(self, pair: str, lot_size: float = 1.0) -> float:
        """
        Get pip value for a currency pair.
        
        Args:
            pair: Currency pair (e.g., "EUR/USD").
            lot_size: Lot size (1.0 = standard lot).
            
        Returns:
            Pip value in USD.
        """
        base_pip_value = self.PIP_VALUES.get(pair, 10.0)
        # For micro lots (0.01), pip value is $0.10 for EUR/USD
        return base_pip_value * lot_size

    def calculate_position_size(
        self,
        pair: str,
        stop_loss_pips: float,
        account_equity: Optional[float] = None,
        free_margin: Optional[float] = None,
        risk_percent: Optional[float] = None,
        entry_price: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate appropriate position size based on risk parameters.
        
        Formula: LotSize = (Equity × Risk%) / (StopDistance × PipValue)
        
        Includes margin safety check: if margin > 30% of free margin, reduce or skip.
        
        Args:
            pair: Currency pair to trade.
            stop_loss_pips: Stop loss distance in pips.
            account_equity: Account equity (uses config balance if None).
            free_margin: Available margin (uses equity if None).
            risk_percent: Risk percentage (uses config if None).
            entry_price: Entry price for R-value calculation.
            
        Returns:
            PositionSizeResult with calculated lot size and validation.
        """
        equity = account_equity or self.config.account_balance
        free_mrg = free_margin or equity
        risk_pct = risk_percent or self._risk_percent
        
        # Clamp risk percent to allowed range
        risk_pct = max(self._risk_min, min(self._risk_max, risk_pct))
        
        self.logger.debug("[RISK_MGMT] Position size calculation initiated")
        self.logger.debug(f"  Account Equity: ${equity:.2f}")
        self.logger.debug(f"  Free Margin: ${free_mrg:.2f}")
        self.logger.debug(f"  Risk Percentage: {risk_pct}%")
        self.logger.debug(f"  Stop Loss Pips: {stop_loss_pips}")
        
        # Calculate risk amount
        risk_amount = equity * (risk_pct / 100)
        self.logger.debug(f"  Risk Amount: ${risk_amount:.2f}")
        
        # Get pip value for 1 standard lot
        pip_value_per_lot = self.PIP_VALUES.get(pair, 10.0)
        self.logger.debug(f"  Pip Value (per lot): ${pip_value_per_lot:.2f}")
        
        # Calculate lot size
        if stop_loss_pips <= 0:
            return PositionSizeResult(
                lot_size=0,
                risk_amount=risk_amount,
                margin_required=0,
                is_valid=False,
                message="Invalid stop loss: must be greater than 0"
            )
        
        # Formula: LotSize = (Equity × Risk%) / (StopDistance × PipValue)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        self.logger.debug(f"  Calculated Lot Size: {lot_size:.4f}")
        
        # Round down to nearest 0.01 (micro lot)
        rounded_lot = round(lot_size, 2)
        # Ensure we don't round up
        if rounded_lot > lot_size:
            rounded_lot -= 0.01
        rounded_lot = max(0.01, rounded_lot)  # Minimum 0.01 lot
        
        self.logger.debug(f"  Rounded Lot Size: {rounded_lot:.2f}")
        
        # Calculate margin required (assuming 1:30 leverage)
        contract_size = 100000  # Standard lot = 100,000 units
        leverage = 30
        margin_required = (rounded_lot * contract_size) / leverage
        self.logger.debug(f"  Margin Required: ${margin_required:.2f}")
        
        # Calculate R-value (risk per trade in price units)
        r_value = 0.0
        if entry_price and stop_loss_pips > 0:
            pip_divisor = 100 if "JPY" in pair else 10000
            r_value = stop_loss_pips / pip_divisor
        
        # MARGIN SAFETY CHECK (Section 1)
        # If margin > 30% of free margin, reduce position or skip
        margin_percent = (margin_required / free_mrg * 100) if free_mrg > 0 else 100
        reduced_for_margin = False
        
        if margin_percent > self._margin_safety_threshold:
            self.logger.warning(
                f"[RISK_MGMT] Margin safety triggered: {margin_percent:.1f}% > {self._margin_safety_threshold}%"
            )
            
            # Calculate maximum lot size that fits within margin safety
            max_margin = free_mrg * (self._margin_safety_threshold / 100)
            max_lot_by_margin = (max_margin * leverage) / contract_size
            max_lot_by_margin = max(0.01, round(max_lot_by_margin, 2))
            
            if max_lot_by_margin < 0.01:
                return PositionSizeResult(
                    lot_size=0,
                    risk_amount=risk_amount,
                    margin_required=margin_required,
                    is_valid=False,
                    message=f"SKIP TRADE: Margin {margin_percent:.1f}% exceeds safety threshold",
                    r_value=r_value,
                    margin_percent_of_free=margin_percent,
                    reduced_for_margin=True
                )
            
            # Reduce lot size to fit margin
            rounded_lot = min(rounded_lot, max_lot_by_margin)
            margin_required = (rounded_lot * contract_size) / leverage
            margin_percent = (margin_required / free_mrg * 100) if free_mrg > 0 else 0
            reduced_for_margin = True
            
            self.logger.warning(
                f"[RISK_MGMT] Reduced lot size to {rounded_lot:.2f} (margin now {margin_percent:.1f}%)"
            )
        
        # Validate against config limits
        config = get_config().trading
        if rounded_lot < config.min_lot_size:
            return PositionSizeResult(
                lot_size=config.min_lot_size,
                risk_amount=risk_amount,
                margin_required=margin_required,
                is_valid=True,
                message=f"Adjusted to minimum lot size: {config.min_lot_size}",
                r_value=r_value,
                margin_percent_of_free=margin_percent,
                reduced_for_margin=reduced_for_margin
            )
        
        if rounded_lot > config.max_lot_size:
            rounded_lot = config.max_lot_size
            margin_required = (rounded_lot * contract_size) / leverage
            margin_percent = (margin_required / free_mrg * 100) if free_mrg > 0 else 0
            
            return PositionSizeResult(
                lot_size=config.max_lot_size,
                risk_amount=risk_amount,
                margin_required=margin_required,
                is_valid=True,
                message=f"Capped at maximum lot size: {config.max_lot_size}",
                r_value=r_value,
                margin_percent_of_free=margin_percent,
                reduced_for_margin=reduced_for_margin
            )
        
        self.logger.info(
            f"[RISK_MGMT] Position Size: {rounded_lot:.2f} lots for {pair} "
            f"({stop_loss_pips:.1f} pip SL, ${risk_amount:.2f} risk, {margin_percent:.1f}% margin)"
        )
        
        return PositionSizeResult(
            lot_size=rounded_lot,
            risk_amount=risk_amount,
            margin_required=margin_required,
            is_valid=True,
            message="Position size validated",
            r_value=r_value,
            margin_percent_of_free=margin_percent,
            reduced_for_margin=reduced_for_margin
        )

    def validate_order(
        self,
        order: Order,
        account_state: AccountState,
        market_data: MarketData
    ) -> RiskCheckResult:
        """
        Validate an order against all risk management rules.
        
        This is the main entry point for risk validation.
        An order MUST pass ALL checks to be approved.
        
        Args:
            order: Order to validate.
            account_state: Current account state.
            market_data: Current market data for the pair.
            
        Returns:
            RiskCheckResult indicating pass/fail with details.
        """
        self._check_daily_reset()
        
        checks = [
            self._check_daily_loss_limit(account_state),
            self._check_drawdown(account_state),
            self._check_stop_loss(order),
            self._check_take_profit(order),
            self._check_margin(order, account_state),
            self._check_spread(market_data),
            self._check_position_size(order),
        ]
        
        # All checks must pass
        for check in checks:
            if not check.passed:
                self.logger.error(f"[RISK_MGMT] Order REJECTED: {check.message}")
                return check
            if check.warning:
                self.logger.warning(f"[RISK_MGMT] {check.message}")
        
        self.logger.info(f"[RISK_MGMT] Order APPROVED: {order.pair} {order.side.value} {order.lot_size} lots")
        return RiskCheckResult(passed=True, message="All risk checks passed")

    def _check_daily_loss_limit(self, account_state: AccountState) -> RiskCheckResult:
        """Check if daily loss limit has been reached."""
        self._check_daily_reset()
        
        # Calculate current daily P&L from account state
        daily_pnl = account_state.equity - self._starting_balance
        self._daily_pnl = daily_pnl
        
        remaining = self.config.daily_loss_limit + daily_pnl  # daily_pnl is negative for losses
        
        if daily_pnl <= -self.config.daily_loss_limit:
            return RiskCheckResult(
                passed=False,
                message=f"DAILY LOSS LIMIT EXCEEDED: ${daily_pnl:.2f} (limit: ${self.config.daily_loss_limit})",
                details={"daily_pnl": daily_pnl, "limit": self.config.daily_loss_limit}
            )
        
        # Warning at 70% of limit
        warning_threshold = self.config.daily_loss_limit * 0.7
        if daily_pnl <= -warning_threshold:
            return RiskCheckResult(
                passed=True,
                warning=True,
                message=f"Daily loss at 70% threshold: ${daily_pnl:.2f} (${remaining:.2f} remaining)",
                details={"daily_pnl": daily_pnl, "remaining": remaining}
            )
        
        return RiskCheckResult(passed=True, message="Daily loss limit OK")

    def _check_drawdown(self, account_state: AccountState) -> RiskCheckResult:
        """Check if maximum drawdown has been reached."""
        # Update peak equity
        if account_state.equity > self._peak_equity:
            self._peak_equity = account_state.equity
        
        # Calculate current drawdown
        drawdown = self._peak_equity - account_state.equity
        drawdown_pct = (drawdown / self._peak_equity) * 100 if self._peak_equity > 0 else 0
        
        max_allowed = self.config.max_drawdown_percent
        
        if drawdown_pct >= max_allowed:
            return RiskCheckResult(
                passed=False,
                message=f"MAX DRAWDOWN EXCEEDED: {drawdown_pct:.1f}% (limit: {max_allowed}%)",
                details={"drawdown_pct": drawdown_pct, "drawdown_amount": drawdown}
            )
        
        # Alert when equity drops below 10% drawdown threshold
        alert_threshold = self.config.account_balance * (1 - max_allowed / 100)
        if account_state.equity <= alert_threshold:
            return RiskCheckResult(
                passed=True,
                warning=True,
                message=f"Equity at drawdown threshold: ${account_state.equity:.2f}",
                details={"equity": account_state.equity, "threshold": alert_threshold}
            )
        
        return RiskCheckResult(passed=True, message="Drawdown OK")

    def _check_stop_loss(self, order: Order) -> RiskCheckResult:
        """Verify order has valid stop loss."""
        if order.stop_loss == 0:
            return RiskCheckResult(
                passed=False,
                message="REJECTED: Order has no stop loss. ALL orders must have stop loss.",
                details={"order_id": order.order_id}
            )
        
        # Calculate stop loss distance in pips
        if "JPY" in order.pair:
            sl_pips = abs(order.entry_price - order.stop_loss) * 100
        else:
            sl_pips = abs(order.entry_price - order.stop_loss) * 10000
        
        if sl_pips < self.config.min_stop_loss_pips:
            return RiskCheckResult(
                passed=False,
                message=f"Stop loss too tight: {sl_pips:.1f} pips (minimum: {self.config.min_stop_loss_pips})",
                details={"sl_pips": sl_pips, "minimum": self.config.min_stop_loss_pips}
            )
        
        # Verify stop loss is on correct side
        if order.side == OrderSide.BUY and order.stop_loss >= order.entry_price:
            return RiskCheckResult(
                passed=False,
                message="BUY order stop loss must be BELOW entry price",
                details={"entry": order.entry_price, "stop_loss": order.stop_loss}
            )
        
        if order.side == OrderSide.SELL and order.stop_loss <= order.entry_price:
            return RiskCheckResult(
                passed=False,
                message="SELL order stop loss must be ABOVE entry price",
                details={"entry": order.entry_price, "stop_loss": order.stop_loss}
            )
        
        return RiskCheckResult(passed=True, message="Stop loss OK")

    def _check_take_profit(self, order: Order) -> RiskCheckResult:
        """Verify order has valid take profit (or None if partial exits enabled)."""
        # Check if partial exits are enabled - if so, TP=0/None is allowed
        # TradeManager handles exits via TP1/TP2/Runner
        partial_exits_enabled = self._check_partial_exits_enabled()
        
        if order.take_profit == 0 or order.take_profit is None:
            if partial_exits_enabled:
                # No broker TP needed - TradeManager handles all exits
                self.logger.debug("[RISK_MGMT] No TP set - partial exits mode (TradeManager controls)")
                return RiskCheckResult(passed=True, message="No TP (partial exits mode)")
            else:
                return RiskCheckResult(
                    passed=False,
                    message="REJECTED: Order has no take profit. ALL orders must have take profit.",
                    details={"order_id": order.order_id}
                )
        
        # Calculate take profit distance in pips
        if "JPY" in order.pair:
            tp_pips = abs(order.take_profit - order.entry_price) * 100
        else:
            tp_pips = abs(order.take_profit - order.entry_price) * 10000
        
        if tp_pips < self.config.min_take_profit_pips:
            return RiskCheckResult(
                passed=False,
                message=f"Take profit too tight: {tp_pips:.1f} pips (minimum: {self.config.min_take_profit_pips})",
                details={"tp_pips": tp_pips, "minimum": self.config.min_take_profit_pips}
            )
        
        # Verify take profit is on correct side
        if order.side == OrderSide.BUY and order.take_profit <= order.entry_price:
            return RiskCheckResult(
                passed=False,
                message="BUY order take profit must be ABOVE entry price",
                details={"entry": order.entry_price, "take_profit": order.take_profit}
            )
        
        if order.side == OrderSide.SELL and order.take_profit >= order.entry_price:
            return RiskCheckResult(
                passed=False,
                message="SELL order take profit must be BELOW entry price",
                details={"entry": order.entry_price, "take_profit": order.take_profit}
            )
        
        return RiskCheckResult(passed=True, message="Take profit OK")
    
    def _check_partial_exits_enabled(self) -> bool:
        """Check if partial exits are enabled in config."""
        try:
            import json
            with open("config/settings.json", "r") as f:
                raw_config = json.load(f)
                return raw_config.get("risk_management", {}).get("partial_exits", {}).get("enabled", False)
        except:
            return False

    def _check_margin(self, order: Order, account_state: AccountState) -> RiskCheckResult:
        """Verify sufficient margin for the order."""
        # Calculate margin required (1:30 leverage)
        contract_size = 100000
        leverage = 30
        margin_required = (order.lot_size * contract_size) / leverage
        
        # Check against available margin with buffer
        buffer_pct = self.config.margin_buffer_percent / 100
        required_with_buffer = margin_required * (1 + buffer_pct)
        
        if margin_required > account_state.free_margin:
            return RiskCheckResult(
                passed=False,
                message=f"Insufficient margin. Required: ${margin_required:.2f}, Available: ${account_state.free_margin:.2f}",
                details={
                    "margin_required": margin_required,
                    "free_margin": account_state.free_margin
                }
            )
        
        if required_with_buffer > account_state.free_margin:
            return RiskCheckResult(
                passed=True,
                warning=True,
                message=f"Margin buffer warning. Required with buffer: ${required_with_buffer:.2f}, Available: ${account_state.free_margin:.2f}",
                details={
                    "margin_required": margin_required,
                    "margin_with_buffer": required_with_buffer,
                    "free_margin": account_state.free_margin
                }
            )
        
        return RiskCheckResult(
            passed=True,
            message="Margin OK",
            details={"margin_required": margin_required, "free_margin": account_state.free_margin}
        )

    def _check_spread(self, market_data: MarketData) -> RiskCheckResult:
        """Check if current spread is acceptable."""
        spread_pips = market_data.spread_pips
        max_spread = self.config.max_spread_allowed
        
        if spread_pips > max_spread:
            return RiskCheckResult(
                passed=False,
                message=f"Spread too wide: {spread_pips:.1f} pips (max: {max_spread} pips)",
                details={"spread_pips": spread_pips, "max_spread": max_spread}
            )
        
        # Warning if spread is above normal (80% of max)
        if spread_pips > max_spread * 0.8:
            return RiskCheckResult(
                passed=True,
                warning=True,
                message=f"Spread wider than normal: {spread_pips:.1f} pips",
                details={"spread_pips": spread_pips}
            )
        
        return RiskCheckResult(passed=True, message="Spread OK")

    def _check_position_size(self, order: Order) -> RiskCheckResult:
        """Validate position size is within limits."""
        config = get_config().trading
        
        if order.lot_size < config.min_lot_size:
            return RiskCheckResult(
                passed=False,
                message=f"Lot size {order.lot_size} below minimum {config.min_lot_size}",
                details={"lot_size": order.lot_size, "min": config.min_lot_size}
            )
        
        if order.lot_size > config.max_lot_size:
            return RiskCheckResult(
                passed=False,
                message=f"Lot size {order.lot_size} above maximum {config.max_lot_size}",
                details={"lot_size": order.lot_size, "max": config.max_lot_size}
            )
        
        return RiskCheckResult(passed=True, message="Position size OK")

    def record_trade(self, order: Order) -> None:
        """
        Record a completed trade for daily tracking.
        
        Args:
            order: Completed order with P&L.
        """
        self._check_daily_reset()
        self._daily_trades.append(order)
        
        if order.pnl is not None:
            self._daily_pnl += order.pnl
            
            if order.pnl < 0:
                self.logger.info(f"[RISK_MGMT] Trade closed with loss: ${order.pnl:.2f}")
            else:
                self.logger.info(f"[RISK_MGMT] Trade closed with profit: ${order.pnl:.2f}")
            
            self.logger.info(f"[RISK_MGMT] Daily P&L: ${self._daily_pnl:.2f}")

    def get_daily_stats(self) -> DailyStats:
        """
        Get current daily statistics.
        
        Returns:
            DailyStats with today's trading performance.
        """
        self._check_daily_reset()
        
        wins = [t for t in self._daily_trades if t.pnl and t.pnl > 0]
        losses = [t for t in self._daily_trades if t.pnl and t.pnl < 0]
        
        return DailyStats(
            date=datetime.combine(self._current_date, datetime.min.time()),
            starting_balance=self._starting_balance,
            ending_balance=self._starting_balance + self._daily_pnl,
            total_trades=len(self._daily_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            gross_profit=sum(t.pnl for t in wins),
            gross_loss=abs(sum(t.pnl for t in losses)),
            max_drawdown=self._peak_equity - (self._starting_balance + self._daily_pnl),
            max_drawdown_pct=((self._peak_equity - (self._starting_balance + self._daily_pnl)) / 
                              self._peak_equity * 100) if self._peak_equity > 0 else 0
        )

    def can_trade(self, account_state: AccountState) -> Tuple[bool, str]:
        """
        Quick check if trading is allowed.
        
        Args:
            account_state: Current account state.
            
        Returns:
            Tuple of (can_trade, reason).
        """
        self._check_daily_reset()
        
        # Check daily loss limit
        daily_pnl = account_state.equity - self._starting_balance
        if daily_pnl <= -self.config.daily_loss_limit:
            return False, f"Daily loss limit reached: ${daily_pnl:.2f}"
        
        # Check drawdown
        if account_state.equity > self._peak_equity:
            self._peak_equity = account_state.equity
        
        drawdown_pct = ((self._peak_equity - account_state.equity) / 
                        self._peak_equity * 100) if self._peak_equity > 0 else 0
        
        if drawdown_pct >= self.config.max_drawdown_percent:
            return False, f"Max drawdown exceeded: {drawdown_pct:.1f}%"
        
        # Check equity threshold
        equity_threshold = self.config.account_balance * 0.9  # 10% below starting
        if account_state.equity < equity_threshold:
            return False, f"Equity below threshold: ${account_state.equity:.2f}"
        
        return True, "Trading allowed"

    def get_remaining_daily_risk(self) -> float:
        """
        Get remaining daily risk allowance.
        
        Returns:
            Remaining amount in USD that can be risked today.
        """
        self._check_daily_reset()
        return max(0, self.config.daily_loss_limit + self._daily_pnl)
    
    def get_remaining_daily_r(self) -> float:
        """
        Get remaining daily R allowance.
        
        Returns:
            Remaining R that can be lost today.
        """
        self._check_daily_reset()
        return max(0, self._daily_loss_limit_r - self._daily_r_loss)
    
    def get_remaining_weekly_r(self) -> float:
        """
        Get remaining weekly R allowance.
        
        Returns:
            Remaining R that can be lost this week.
        """
        self._check_daily_reset()
        return max(0, self._weekly_loss_limit_r - self._weekly_r_loss)
    
    def record_r_result(self, r_result: float) -> None:
        """
        Record R result from a trade.
        
        Args:
            r_result: R-multiple result (negative for loss, positive for win)
        """
        self._check_daily_reset()
        
        if r_result < 0:
            self._daily_r_loss += abs(r_result)
            self._weekly_r_loss += abs(r_result)
            self.logger.info(f"[RISK_MGMT] R-Loss recorded: {r_result:.2f}R")
            self.logger.info(f"[RISK_MGMT]   Daily: {self._daily_r_loss:.2f}R / {self._daily_loss_limit_r}R")
            self.logger.info(f"[RISK_MGMT]   Weekly: {self._weekly_r_loss:.2f}R / {self._weekly_loss_limit_r}R")
    
    def check_r_limits(self) -> Tuple[bool, str]:
        """
        Check if daily/weekly R limits allow trading.
        
        Returns:
            Tuple of (can_trade, reason).
        """
        self._check_daily_reset()
        
        if self._daily_r_loss >= self._daily_loss_limit_r:
            return False, f"Daily R limit reached: {self._daily_r_loss:.1f}R >= {self._daily_loss_limit_r}R"
        
        if self._weekly_r_loss >= self._weekly_loss_limit_r:
            return False, f"Weekly R limit reached: {self._weekly_r_loss:.1f}R >= {self._weekly_loss_limit_r}R"
        
        return True, "R limits OK"
    
    # ========== SPREAD & VOLATILITY FILTERS (Section 7) ==========
    
    def update_spread_history(self, pair: str, spread_pips: float) -> None:
        """
        Update spread history for a pair.
        
        Args:
            pair: Currency pair
            spread_pips: Current spread in pips
        """
        if pair not in self._spread_history:
            self._spread_history[pair] = deque(maxlen=self._lookback_candles)
        self._spread_history[pair].append(spread_pips)
    
    def update_atr_history(self, pair: str, atr: float) -> None:
        """
        Update ATR history for a pair.
        
        Args:
            pair: Currency pair
            atr: Current ATR value
        """
        if pair not in self._atr_history:
            self._atr_history[pair] = deque(maxlen=self._lookback_candles)
        self._atr_history[pair].append(atr)
    
    def check_spread_volatility(
        self,
        pair: str,
        current_spread_pips: float,
        current_atr: float
    ) -> SpreadVolatilityData:
        """
        Check spread and volatility filters.
        
        Args:
            pair: Currency pair
            current_spread_pips: Current spread in pips
            current_atr: Current ATR value
            
        Returns:
            SpreadVolatilityData with filter results
        """
        result = SpreadVolatilityData(
            current_spread_pips=current_spread_pips,
            current_atr=current_atr,
            spread_ok=True,
            volatility_ok=True
        )
        
        # Update histories
        self.update_spread_history(pair, current_spread_pips)
        self.update_atr_history(pair, current_atr)
        
        # Check spread filter
        if self._spread_filter_enabled:
            spread_hist = self._spread_history.get(pair, [])
            if len(spread_hist) >= 10:  # Need some history
                avg_spread = statistics.mean(spread_hist)
                result.average_spread_pips = avg_spread
                result.spread_ratio = current_spread_pips / avg_spread if avg_spread > 0 else 1.0
                
                if result.spread_ratio > self._max_spread_multiplier:
                    result.spread_ok = False
                    self.logger.warning(
                        f"[RISK_MGMT] Spread filter: {pair} spread {current_spread_pips:.1f} pips "
                        f"is {result.spread_ratio:.1f}x average ({avg_spread:.1f})"
                    )
        
        # Check volatility filter
        if self._volatility_filter_enabled:
            atr_hist = self._atr_history.get(pair, [])
            if len(atr_hist) >= 10:  # Need some history
                median_atr = statistics.median(atr_hist)
                result.median_atr = median_atr
                result.atr_ratio = current_atr / median_atr if median_atr > 0 else 1.0
                
                if result.atr_ratio > self._atr_spike_multiplier:
                    result.volatility_ok = False
                    self.logger.warning(
                        f"[RISK_MGMT] Volatility filter: {pair} ATR {current_atr:.5f} "
                        f"is {result.atr_ratio:.1f}x median ({median_atr:.5f})"
                    )
        
        if not result.can_trade:
            self.logger.warning(f"[RISK_MGMT] SKIP TRADE: {result.block_reason}")
        
        return result
    
    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """
        Calculate Average True Range from candle data.
        
        Args:
            candles: List of Candle objects (most recent last)
            period: ATR period (default 14)
            
        Returns:
            ATR value
        """
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            candle = candles[i]
            prev_candle = candles[i - 1]
            
            tr = max(
                candle.high - candle.low,  # Current high-low
                abs(candle.high - prev_candle.close),  # High - prev close
                abs(candle.low - prev_candle.close)   # Low - prev close
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return statistics.mean(true_ranges) if true_ranges else 0.0
        
        # Use simple average for ATR (could use EMA for more smoothing)
        return statistics.mean(true_ranges[-period:])
    
    def get_risk_status(self) -> dict:
        """
        Get current risk management status.
        
        Returns:
            Dict with current risk metrics and limits.
        """
        self._check_daily_reset()
        
        return {
            "daily_r_loss": self._daily_r_loss,
            "daily_r_limit": self._daily_loss_limit_r,
            "daily_r_remaining": self.get_remaining_daily_r(),
            "weekly_r_loss": self._weekly_r_loss,
            "weekly_r_limit": self._weekly_loss_limit_r,
            "weekly_r_remaining": self.get_remaining_weekly_r(),
            "daily_limit_hit": self._daily_r_loss >= self._daily_loss_limit_r,
            "weekly_limit_hit": self._weekly_r_loss >= self._weekly_loss_limit_r,
            "can_trade": self._daily_r_loss < self._daily_loss_limit_r and 
                        self._weekly_r_loss < self._weekly_loss_limit_r,
            "risk_percent": self._risk_percent,
            "margin_safety_threshold": self._margin_safety_threshold
        }