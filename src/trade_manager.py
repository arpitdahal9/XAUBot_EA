"""
Trade Manager - State Machine for Partial Exit Strategy

Implements Model A: Balanced Asymmetry
- TP1 at +2R: Close 50%
- TP2 at +3R: Close 30%
- Runner: Trail with ATR until stopped (targeting 6R-10R)

State transitions:
ENTRY_FILLED -> BREAKEVEN_TRIGGERED (+1.2R) -> TP1_HIT (+2R) -> TP2_HIT (+3R) -> RUNNER_STOPPED
              -> STOPPED_OUT (at any point)
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .models import (
    ManagedTrade, TradeState, OrderSide, TradingModule,
    CorrelationGroup, RiskMetrics, SpreadVolatilityData, Candle
)
from .logger import get_logger

logger = get_logger(__name__)


class TradeManager:
    """
    Manages active trades using a state machine approach.
    Handles partial exits, trailing stops, and risk tracking.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the trade manager.
        
        Args:
            config: Risk management configuration dict
        """
        self.config = config
        self.active_trades: Dict[str, ManagedTrade] = {}
        self.closed_trades: List[ManagedTrade] = []
        self.risk_metrics = RiskMetrics()
        
        # Load partial exit settings
        partial_config = config.get("partial_exits", {})
        self.partial_exits_enabled = partial_config.get("enabled", True)
        self.tp1_r = partial_config.get("tp1_r_multiple", 2.0)
        self.tp1_percent = partial_config.get("tp1_close_percent", 50)
        self.tp2_r = partial_config.get("tp2_r_multiple", 3.0)
        self.tp2_percent = partial_config.get("tp2_close_percent", 30)
        self.runner_percent = partial_config.get("runner_percent", 20)
        
        # Breakeven settings
        self.breakeven_trigger_r = config.get("breakeven_trigger_r", 1.2)
        self.breakeven_offset_r = config.get("breakeven_offset_r", 0.1)
        self.post_tp1_stop_r = config.get("post_tp1_stop_r", 0.2)
        
        # Trailing settings
        trailing_config = config.get("trailing", {})
        self.trailing_enabled = trailing_config.get("enabled", True)
        self.atr_mult_forex = trailing_config.get("atr_multiplier_forex", 1.8)
        self.atr_mult_gold = trailing_config.get("atr_multiplier_gold", 2.2)
        self.atr_period = trailing_config.get("atr_period", 14)
        
        # Portfolio limits
        self.max_total_risk = config.get("max_total_risk_percent", 2.0)
        self.max_concurrent = config.get("max_concurrent_trades", 2)
        self.avoid_correlation = config.get("avoid_correlated_trades", True)
        
        # Drawdown limits
        self.daily_loss_limit_r = config.get("daily_loss_limit_r", 2.0)
        self.weekly_loss_limit_r = config.get("weekly_loss_limit_r", 6.0)
        
        # Tracking
        self._daily_r_loss = 0.0
        self._weekly_r_loss = 0.0
        self._last_daily_reset = datetime.now().date()
        self._last_weekly_reset = datetime.now().isocalendar()[1]
        
        logger.info("[TRADE_MGR] Trade Manager initialized")
        logger.info(f"[TRADE_MGR] Partial exits: TP1={self.tp1_r}R ({self.tp1_percent}%), "
                   f"TP2={self.tp2_r}R ({self.tp2_percent}%), Runner={self.runner_percent}%")
    
    def create_trade(
        self,
        trade_id: str,
        pair: str,
        side: OrderSide,
        entry_price: float,
        stop_loss_price: float,
        lot_size: float,
        risk_amount_usd: float,
        module: TradingModule = TradingModule.TREND,
        timestamp: Optional[datetime] = None
    ) -> Optional[ManagedTrade]:
        """
        Create a new managed trade.
        
        Args:
            trade_id: Unique identifier
            pair: Currency pair
            side: BUY or SELL
            entry_price: Entry price
            stop_loss_price: Initial stop loss
            lot_size: Position size
            risk_amount_usd: Dollar amount at risk
            module: Trading module that created this trade
            timestamp: Optional timestamp for backtesting
            
        Returns:
            ManagedTrade if created, None if blocked by risk limits
        """
        # Check daily/weekly limits
        self._check_reset_periods(timestamp)
        
        if self._daily_r_loss >= self.daily_loss_limit_r:
            logger.warning(f"[TRADE_MGR] BLOCKED: Daily loss limit reached ({self._daily_r_loss:.1f}R)")
            return None
        
        if self._weekly_r_loss >= self.weekly_loss_limit_r:
            logger.warning(f"[TRADE_MGR] BLOCKED: Weekly loss limit reached ({self._weekly_r_loss:.1f}R)")
            return None
        
        # Check max concurrent trades
        if len(self.active_trades) >= self.max_concurrent:
            logger.warning(f"[TRADE_MGR] BLOCKED: Max concurrent trades ({self.max_concurrent}) reached")
            return None
        
        # Calculate R-value (risk per trade in price units)
        if side == OrderSide.BUY:
            r_value = entry_price - stop_loss_price
        else:
            r_value = stop_loss_price - entry_price
        
        if r_value <= 0:
            logger.error(f"[TRADE_MGR] Invalid R-value: {r_value}")
            return None
        
        # Determine correlation group
        correlation_group = self._get_correlation_group(pair, side)
        
        # Check correlation
        if self.avoid_correlation and correlation_group:
            for existing_trade in self.active_trades.values():
                if existing_trade.correlation_group == correlation_group:
                    logger.warning(f"[TRADE_MGR] BLOCKED: Correlated trade already open "
                                 f"({correlation_group.value})")
                    return None
        
        # Create the managed trade
        trade = ManagedTrade(
            trade_id=trade_id,
            pair=pair,
            side=side,
            entry_price=entry_price,
            initial_stop_loss=stop_loss_price,
            initial_lot_size=lot_size,
            r_value=r_value,
            state=TradeState.ENTRY_FILLED,
            current_stop_loss=stop_loss_price,
            risk_amount_usd=risk_amount_usd,
            module=module,
            correlation_group=correlation_group,
            entry_time=timestamp or datetime.now()  # FIX: Use passed timestamp
        )
        
        # Calculate partial lot sizes with proper minimum handling
        # FIX: Don't round to 0 - use floor with minimum step
        if self.partial_exits_enabled:
            min_lot_step = 0.01  # Broker minimum lot step
            
            # Calculate raw partial sizes
            raw_tp1 = lot_size * (self.tp1_percent / 100)
            raw_tp2 = lot_size * (self.tp2_percent / 100)
            raw_runner = lot_size * (self.runner_percent / 100)
            
            # For small lot sizes (0.01), we need special handling
            # Priority: TP1 > Runner > TP2 (ensure profitable exits first)
            if lot_size <= min_lot_step:
                # Single micro lot - all goes to runner (let it trail)
                trade.tp1_lot_size = 0.0
                trade.tp2_lot_size = 0.0
                trade.runner_lot_size = lot_size
                logger.info(f"[TRADE_MGR] Small lot ({lot_size}) - entire position is runner")
            elif lot_size <= min_lot_step * 2:
                # 0.02 lots: TP1 gets 0.01, runner gets 0.01, TP2 = 0
                trade.tp1_lot_size = min_lot_step
                trade.tp2_lot_size = 0.0
                trade.runner_lot_size = lot_size - min_lot_step
                logger.info(f"[TRADE_MGR] Small lot ({lot_size}) - TP1={trade.tp1_lot_size}, Runner={trade.runner_lot_size}")
            elif lot_size <= min_lot_step * 3:
                # 0.03 lots: TP1=0.01, TP2=0.01, Runner=0.01
                trade.tp1_lot_size = min_lot_step
                trade.tp2_lot_size = min_lot_step
                trade.runner_lot_size = lot_size - (2 * min_lot_step)
                logger.info(f"[TRADE_MGR] Small lot ({lot_size}) - TP1={trade.tp1_lot_size}, TP2={trade.tp2_lot_size}, Runner={trade.runner_lot_size}")
            else:
                # Normal lot sizes - use floor to ensure valid lots
                import math
                trade.tp1_lot_size = math.floor(raw_tp1 / min_lot_step) * min_lot_step
                trade.tp2_lot_size = math.floor(raw_tp2 / min_lot_step) * min_lot_step
                
                # Ensure minimums
                trade.tp1_lot_size = max(min_lot_step, trade.tp1_lot_size)
                trade.tp2_lot_size = max(min_lot_step, trade.tp2_lot_size)
                
                # Runner gets the rest
                trade.runner_lot_size = lot_size - trade.tp1_lot_size - trade.tp2_lot_size
                
                # If runner is too small, merge into TP2
                if trade.runner_lot_size < min_lot_step:
                    trade.tp2_lot_size += trade.runner_lot_size
                    trade.runner_lot_size = 0.0
                    # Round TP2 to valid lot
                    trade.tp2_lot_size = round(trade.tp2_lot_size, 2)
        
        # Calculate TP prices using config R-multiples (SINGLE SOURCE OF TRUTH)
        if side == OrderSide.BUY:
            trade.tp1_price = entry_price + (self.tp1_r * r_value)
            trade.tp2_price = entry_price + (self.tp2_r * r_value)
            trade.breakeven_trigger_price = entry_price + (self.breakeven_trigger_r * r_value)
        else:  # SELL
            trade.tp1_price = entry_price - (self.tp1_r * r_value)
            trade.tp2_price = entry_price - (self.tp2_r * r_value)
            trade.breakeven_trigger_price = entry_price - (self.breakeven_trigger_r * r_value)
        
        self.active_trades[trade_id] = trade
        self._update_risk_metrics()
        
        logger.info(f"[TRADE_MGR] NEW TRADE: {trade_id}")
        logger.info(f"[TRADE_MGR]   {side.value} {lot_size} {pair} @ {entry_price}")
        logger.info(f"[TRADE_MGR]   SL: {stop_loss_price} (1R = {r_value:.5f})")
        logger.info(f"[TRADE_MGR]   TP1: {trade.tp1_price:.5f} (+{self.tp1_r}R, {trade.tp1_lot_size} lots)")
        logger.info(f"[TRADE_MGR]   TP2: {trade.tp2_price:.5f} (+{self.tp2_r}R, {trade.tp2_lot_size} lots)")
        logger.info(f"[TRADE_MGR]   Runner: {trade.runner_lot_size} lots (trailing)")
        
        return trade
    
    def update_trade(
        self,
        trade_id: str,
        current_price: float,
        current_close: float,
        current_atr: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> List[dict]:
        """
        Update a trade based on current price.
        Returns list of actions to execute.
        
        Args:
            trade_id: Trade to update
            current_price: Current market price
            current_close: Current candle close (for trailing)
            current_atr: Current ATR value (for trailing)
            timestamp: Current timestamp (for backtesting)
            
        Returns:
            List of action dicts: [{"action": "CLOSE_PARTIAL", "lots": 0.05, "reason": "TP1"}]
        """
        if trade_id not in self.active_trades:
            return []
        
        trade = self.active_trades[trade_id]
        trade.metadata["current_price"] = current_price
        trade.metadata["current_timestamp"] = timestamp or datetime.now()
        
        if current_atr:
            trade.current_atr = current_atr
        
        # Track high/low
        if trade.side == OrderSide.BUY:
            trade.highest_price_since_entry = max(trade.highest_price_since_entry, current_price)
            trade.lowest_price_since_entry = min(trade.lowest_price_since_entry or current_price, current_price)
        else:
            trade.lowest_price_since_entry = min(trade.lowest_price_since_entry or current_price, current_price)
            trade.highest_price_since_entry = max(trade.highest_price_since_entry, current_price)
        
        actions = []
        r_multiple = trade.current_r_multiple
        
        # State machine transitions
        if trade.state == TradeState.ENTRY_FILLED:
            actions.extend(self._handle_entry_filled(trade, current_price, r_multiple))
        
        elif trade.state == TradeState.BREAKEVEN_TRIGGERED:
            actions.extend(self._handle_breakeven(trade, current_price, r_multiple))
        
        elif trade.state == TradeState.TP1_HIT:
            actions.extend(self._handle_tp1_hit(trade, current_price, current_close, r_multiple))
        
        elif trade.state == TradeState.TP2_HIT:
            actions.extend(self._handle_tp2_hit(trade, current_price, current_close))
        
        # Check stop loss at all states
        if trade.is_active:
            stop_action = self._check_stop_loss(trade, current_price)
            if stop_action:
                actions.append(stop_action)
        
        return actions
    
    def _get_pip_value(self, pair: str, lot_size: float) -> float:
        """Get pip value for a currency pair using symbol spec."""
        from .symbol_spec import get_symbol_spec
        spec = get_symbol_spec(pair)
        return spec.pip_value_per_lot * lot_size
    
    def _get_pip_size(self, pair: str) -> float:
        """Get pip size for a currency pair using symbol spec."""
        from .symbol_spec import get_symbol_spec
        spec = get_symbol_spec(pair)
        return spec.pip_size
    
    def _handle_entry_filled(self, trade: ManagedTrade, price: float, r_mult: float) -> List[dict]:
        """Handle ENTRY_FILLED state."""
        actions = []
        
        # Check for breakeven trigger (+1.2R)
        if r_mult >= self.breakeven_trigger_r:
            old_sl = trade.current_stop_loss
            trade.current_stop_loss = trade.get_breakeven_stop(self.breakeven_offset_r)
            trade.state = TradeState.BREAKEVEN_TRIGGERED
            
            logger.info(f"[TRADE_MGR] {trade.trade_id}: BREAKEVEN TRIGGERED at {r_mult:.2f}R")
            logger.info(f"[TRADE_MGR]   SL moved: {old_sl:.5f} -> {trade.current_stop_loss:.5f}")
            
            actions.append({
                "action": "MODIFY_SL",
                "trade_id": trade.trade_id,
                "new_sl": trade.current_stop_loss,
                "reason": f"Breakeven at +{r_mult:.2f}R"
            })
        
        return actions
    
    def _handle_breakeven(self, trade: ManagedTrade, price: float, r_mult: float) -> List[dict]:
        """Handle BREAKEVEN_TRIGGERED state."""
        actions = []
        
        # Check for TP1 (+2R)
        if r_mult >= self.tp1_r:
            # Close TP1 portion
            close_lots = trade.tp1_lot_size
            
            # Skip if no lots to close (small position)
            if close_lots <= 0:
                # For small positions, skip TP1 and go straight to runner mode
                trade.state = TradeState.TP1_HIT  # Move to next state anyway
                trade.tp1_time = trade.metadata.get("current_timestamp", datetime.now())
                trade.highest_close_since_tp1 = price
                logger.info(f"[TRADE_MGR] {trade.trade_id}: TP1 SKIPPED (no lots), moving to runner")
                return actions
            
            # Calculate P&L for this partial using symbol spec
            pip_size = self._get_pip_size(trade.pair)
            if trade.side == OrderSide.BUY:
                pips = (price - trade.entry_price) / pip_size
            else:
                pips = (trade.entry_price - price) / pip_size
            
            pip_value = self._get_pip_value(trade.pair, close_lots)
            pnl = pips * pip_value
            trade.tp1_pnl = pnl
            trade.realized_pnl += pnl
            trade.remaining_lot_size -= close_lots
            trade.tp1_time = trade.metadata.get("current_timestamp", datetime.now())
            
            # Move stop to +0.2R
            old_sl = trade.current_stop_loss
            trade.current_stop_loss = trade.get_post_tp1_stop(self.post_tp1_stop_r)
            trade.state = TradeState.TP1_HIT
            trade.highest_close_since_tp1 = price
            
            logger.info(f"[TRADE_MGR] {trade.trade_id}: TP1 HIT at +{r_mult:.2f}R")
            logger.info(f"[TRADE_MGR]   Closed {close_lots} lots (+${pnl:.2f})")
            logger.info(f"[TRADE_MGR]   SL moved: {old_sl:.5f} -> {trade.current_stop_loss:.5f}")
            logger.info(f"[TRADE_MGR]   Remaining: {trade.remaining_lot_size} lots")
            
            actions.append({
                "action": "CLOSE_PARTIAL",
                "trade_id": trade.trade_id,
                "lots": close_lots,
                "price": price,
                "pnl": pnl,
                "reason": f"TP1 at +{self.tp1_r}R"
            })
            
            actions.append({
                "action": "MODIFY_SL",
                "trade_id": trade.trade_id,
                "new_sl": trade.current_stop_loss,
                "reason": f"Lock profit after TP1"
            })
        
        return actions
    
    def _handle_tp1_hit(
        self,
        trade: ManagedTrade,
        price: float,
        close_price: float,
        r_mult: float
    ) -> List[dict]:
        """Handle TP1_HIT state - waiting for TP2."""
        actions = []
        
        # Track highest close for trailing
        if trade.side == OrderSide.BUY:
            trade.highest_close_since_tp1 = max(trade.highest_close_since_tp1, close_price)
        else:
            # For sells, track lowest close
            if trade.highest_close_since_tp1 == 0:
                trade.highest_close_since_tp1 = close_price
            trade.highest_close_since_tp1 = min(trade.highest_close_since_tp1, close_price)
        
        # Check for TP2 (+3R)
        if r_mult >= self.tp2_r:
            # Close TP2 portion
            close_lots = trade.tp2_lot_size
            
            # Skip if no lots to close
            if close_lots <= 0:
                trade.state = TradeState.TP2_HIT
                trade.tp2_time = trade.metadata.get("current_timestamp", datetime.now())
                logger.info(f"[TRADE_MGR] {trade.trade_id}: TP2 SKIPPED (no lots), runner continues")
                return actions
            
            # Calculate P&L using symbol spec
            pip_size = self._get_pip_size(trade.pair)
            if trade.side == OrderSide.BUY:
                pips = (price - trade.entry_price) / pip_size
            else:
                pips = (trade.entry_price - price) / pip_size
            
            pip_value = self._get_pip_value(trade.pair, close_lots)
            pnl = pips * pip_value
            trade.tp2_pnl = pnl
            trade.realized_pnl += pnl
            trade.remaining_lot_size -= close_lots
            trade.tp2_time = trade.metadata.get("current_timestamp", datetime.now())
            trade.state = TradeState.TP2_HIT
            
            logger.info(f"[TRADE_MGR] {trade.trade_id}: TP2 HIT at +{r_mult:.2f}R")
            logger.info(f"[TRADE_MGR]   Closed {close_lots} lots (+${pnl:.2f})")
            logger.info(f"[TRADE_MGR]   Runner active: {trade.remaining_lot_size} lots")
            
            actions.append({
                "action": "CLOSE_PARTIAL",
                "trade_id": trade.trade_id,
                "lots": close_lots,
                "price": price,
                "pnl": pnl,
                "reason": f"TP2 at +{self.tp2_r}R"
            })
        
        return actions
    
    def _handle_tp2_hit(
        self,
        trade: ManagedTrade,
        price: float,
        close_price: float
    ) -> List[dict]:
        """Handle TP2_HIT state - runner with ATR trailing."""
        actions = []
        
        if not self.trailing_enabled or trade.current_atr == 0:
            return actions
        
        # Track highest/lowest close
        if trade.side == OrderSide.BUY:
            trade.highest_close_since_tp1 = max(trade.highest_close_since_tp1, close_price)
        else:
            trade.highest_close_since_tp1 = min(trade.highest_close_since_tp1, close_price)
        
        # Calculate trailing stop
        k = self.atr_mult_gold if "XAU" in trade.pair or "GOLD" in trade.pair else self.atr_mult_forex
        new_trailing = trade.calculate_trailing_stop(trade.current_atr, k)
        
        # Only move stop if it tightens (locks in more profit)
        should_update = False
        if trade.side == OrderSide.BUY:
            if new_trailing > trade.current_stop_loss:
                should_update = True
        else:
            if new_trailing < trade.current_stop_loss:
                should_update = True
        
        if should_update:
            old_sl = trade.current_stop_loss
            trade.current_stop_loss = new_trailing
            trade.trailing_stop = new_trailing
            
            logger.debug(f"[TRADE_MGR] {trade.trade_id}: Trailing SL: {old_sl:.5f} -> {new_trailing:.5f}")
            
            actions.append({
                "action": "MODIFY_SL",
                "trade_id": trade.trade_id,
                "new_sl": new_trailing,
                "reason": f"ATR trailing (k={k})"
            })
        
        return actions
    
    def _check_stop_loss(self, trade: ManagedTrade, current_price: float) -> Optional[dict]:
        """Check if stop loss has been hit."""
        stopped = False
        
        if trade.side == OrderSide.BUY:
            if current_price <= trade.current_stop_loss:
                stopped = True
        else:
            if current_price >= trade.current_stop_loss:
                stopped = True
        
        if stopped:
            # Calculate final P&L using symbol spec
            pip_size = self._get_pip_size(trade.pair)
            if trade.side == OrderSide.BUY:
                pips = (current_price - trade.entry_price) / pip_size
            else:
                pips = (trade.entry_price - current_price) / pip_size
            
            pip_value = self._get_pip_value(trade.pair, trade.remaining_lot_size)
            final_pnl = pips * pip_value
            
            if trade.state == TradeState.TP2_HIT:
                trade.runner_pnl = final_pnl
                trade.state = TradeState.RUNNER_STOPPED
            else:
                trade.state = TradeState.STOPPED_OUT
            
            trade.realized_pnl += final_pnl
            trade.exit_time = trade.metadata.get("current_timestamp", datetime.now())
            
            # Track R-loss for daily/weekly limits
            r_result = trade.current_r_multiple
            if r_result < 0:
                self._daily_r_loss += abs(r_result)
                self._weekly_r_loss += abs(r_result)
            
            logger.info(f"[TRADE_MGR] {trade.trade_id}: STOPPED at {current_price:.5f}")
            logger.info(f"[TRADE_MGR]   State: {trade.state.value}")
            logger.info(f"[TRADE_MGR]   Final P&L: ${trade.realized_pnl:.2f}")
            logger.info(f"[TRADE_MGR]   R-Result: {r_result:+.2f}R")
            
            # Move to closed trades
            self.closed_trades.append(trade)
            del self.active_trades[trade.trade_id]
            self._update_risk_metrics()
            
            return {
                "action": "CLOSE_ALL",
                "trade_id": trade.trade_id,
                "lots": trade.remaining_lot_size,
                "price": current_price,
                "pnl": final_pnl,
                "total_pnl": trade.realized_pnl,
                "reason": trade.state.value
            }
        
        return None
    
    def close_trade(self, trade_id: str, exit_price: float, reason: str = "Manual", timestamp: Optional[datetime] = None) -> Optional[dict]:
        """Manually close an entire trade."""
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        
        # Calculate final P&L using symbol spec
        pip_size = self._get_pip_size(trade.pair)
        if trade.side == OrderSide.BUY:
            pips = (exit_price - trade.entry_price) / pip_size
        else:
            pips = (trade.entry_price - exit_price) / pip_size
        
        pip_value = self._get_pip_value(trade.pair, trade.remaining_lot_size)
        final_pnl = pips * pip_value
        trade.realized_pnl += final_pnl
        trade.state = TradeState.FULLY_CLOSED
        trade.exit_time = timestamp or trade.metadata.get("current_timestamp", datetime.now())
        
        logger.info(f"[TRADE_MGR] {trade_id}: CLOSED ({reason})")
        logger.info(f"[TRADE_MGR]   Exit: {exit_price}, P&L: ${trade.realized_pnl:.2f}")
        
        self.closed_trades.append(trade)
        del self.active_trades[trade_id]
        self._update_risk_metrics()
        
        return {
            "action": "CLOSE_ALL",
            "trade_id": trade_id,
            "lots": trade.remaining_lot_size,
            "price": exit_price,
            "pnl": final_pnl,
            "total_pnl": trade.realized_pnl,
            "reason": reason
        }
    
    def _get_correlation_group(self, pair: str, side: OrderSide) -> Optional[CorrelationGroup]:
        """Determine correlation group for a trade."""
        pair_upper = pair.upper().replace("/", "")
        
        # USD pairs
        if pair_upper in ["EURUSD", "GBPUSD", "AUDUSD"]:
            return CorrelationGroup.USD_SHORT if side == OrderSide.BUY else CorrelationGroup.USD_LONG
        elif pair_upper == "USDJPY":
            return CorrelationGroup.USD_LONG if side == OrderSide.BUY else CorrelationGroup.USD_SHORT
        elif pair_upper == "USDCAD":
            return CorrelationGroup.USD_LONG if side == OrderSide.BUY else CorrelationGroup.USD_SHORT
        elif "XAU" in pair_upper or "GOLD" in pair_upper:
            return CorrelationGroup.GOLD
        
        return None
    
    def _check_reset_periods(self, timestamp: Optional[datetime] = None):
        """
        Check and reset daily/weekly loss counters.
        
        Args:
            timestamp: Optional timestamp for backtesting (uses current time if None)
        """
        check_time = timestamp or datetime.now()
        today = check_time.date()
        current_week = check_time.isocalendar()[1]
        
        if today != self._last_daily_reset:
            if self._daily_r_loss > 0:
                logger.info(f"[TRADE_MGR] Daily reset ({today}) - Previous loss: {self._daily_r_loss:.2f}R")
            self._daily_r_loss = 0.0
            self._last_daily_reset = today
        
        if current_week != self._last_weekly_reset:
            if self._weekly_r_loss > 0:
                logger.info(f"[TRADE_MGR] Weekly reset (week {current_week}) - Previous loss: {self._weekly_r_loss:.2f}R")
            self._weekly_r_loss = 0.0
            self._last_weekly_reset = current_week
    
    def _update_risk_metrics(self):
        """Update portfolio risk metrics."""
        self.risk_metrics.open_trade_count = len(self.active_trades)
        self.risk_metrics.total_open_risk_usd = sum(
            t.risk_amount_usd for t in self.active_trades.values()
        )
        self.risk_metrics.daily_r_pnl = -self._daily_r_loss
        self.risk_metrics.weekly_r_pnl = -self._weekly_r_loss
        self.risk_metrics.daily_limit_hit = self._daily_r_loss >= self.daily_loss_limit_r
        self.risk_metrics.weekly_limit_hit = self._weekly_r_loss >= self.weekly_loss_limit_r
        self.risk_metrics.active_correlation_groups = [
            t.correlation_group for t in self.active_trades.values() if t.correlation_group
        ]
        self.risk_metrics.can_trade = not (
            self.risk_metrics.daily_limit_hit or
            self.risk_metrics.weekly_limit_hit or
            self.risk_metrics.open_trade_count >= self.max_concurrent
        )
        self.risk_metrics.last_updated = datetime.now()
        
        if not self.risk_metrics.can_trade:
            reasons = []
            if self.risk_metrics.daily_limit_hit:
                reasons.append(f"Daily limit ({self._daily_r_loss:.1f}R)")
            if self.risk_metrics.weekly_limit_hit:
                reasons.append(f"Weekly limit ({self._weekly_r_loss:.1f}R)")
            if self.risk_metrics.open_trade_count >= self.max_concurrent:
                reasons.append(f"Max trades ({self.max_concurrent})")
            self.risk_metrics.block_reason = ", ".join(reasons)
    
    def get_status(self) -> dict:
        """Get current trade manager status."""
        self._check_reset_periods()
        self._update_risk_metrics()
        
        return {
            "active_trades": len(self.active_trades),
            "closed_trades_today": len([
                t for t in self.closed_trades
                if t.exit_time and t.exit_time.date() == datetime.now().date()
            ]),
            "daily_r_pnl": -self._daily_r_loss,
            "weekly_r_pnl": -self._weekly_r_loss,
            "daily_limit": f"{self._daily_r_loss:.2f}R / {self.daily_loss_limit_r}R",
            "weekly_limit": f"{self._weekly_r_loss:.2f}R / {self.weekly_loss_limit_r}R",
            "can_trade": self.risk_metrics.can_trade,
            "block_reason": self.risk_metrics.block_reason,
            "correlation_exposure": [g.value for g in self.risk_metrics.active_correlation_groups]
        }
    
    def get_trade_summary(self, trade_id: str) -> Optional[dict]:
        """Get summary of a specific trade."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            # Check closed trades
            for t in self.closed_trades:
                if t.trade_id == trade_id:
                    trade = t
                    break
        
        if not trade:
            return None
        
        return {
            "trade_id": trade.trade_id,
            "pair": trade.pair,
            "side": trade.side.value,
            "state": trade.state.value,
            "entry_price": trade.entry_price,
            "current_stop": trade.current_stop_loss,
            "r_value": trade.r_value,
            "current_r": trade.current_r_multiple,
            "initial_lots": trade.initial_lot_size,
            "remaining_lots": trade.remaining_lot_size,
            "realized_pnl": trade.realized_pnl,
            "tp1_pnl": trade.tp1_pnl,
            "tp2_pnl": trade.tp2_pnl,
            "runner_pnl": trade.runner_pnl,
            "entry_time": trade.entry_time.isoformat(),
            "tp1_price": trade.tp1_price,
            "tp2_price": trade.tp2_price
        }
