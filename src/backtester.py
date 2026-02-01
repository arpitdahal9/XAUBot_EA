"""
Backtesting Engine for Trading Strategies.

Provides historical simulation capabilities for:
- Grid Trading Strategy
- Trend Following Strategy
- Combined strategies

Features:
- Multiple data sources (MT5, CSV, generated)
- Realistic spread/slippage simulation
- Comprehensive performance metrics
- Model A partial exit tracking
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .models import (
    OrderSide, OrderStatus, SignalType, TradingModule,
    MarketData, AccountState, Candle, Signal, Order,
    ManagedTrade, TradeState
)
from .config import get_config, TradingConfig
from .logger import get_logger
from .risk_manager import RiskManager
from .trend_follower import TrendFollower
from .grid_trader import GridTrader
from .trade_manager import TradeManager


class RowType(Enum):
    """Type of CSV row for proper PnL accounting."""
    ENTRY = "ENTRY"
    TP1_PARTIAL = "TP1_PARTIAL"
    TP2_PARTIAL = "TP2_PARTIAL"
    FINAL = "FINAL"  # Final close (runner stop, SL, or manual)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade event."""
    entry_time: datetime
    exit_time: Optional[datetime]
    pair: str
    side: OrderSide
    lot_size: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: str = ""
    module: TradingModule = TradingModule.GRID
    
    # === MODEL A TRACKING ===
    trade_id: str = ""
    row_type: RowType = RowType.FINAL
    is_partial: bool = False
    r_multiple: float = 0.0
    trade_state: str = ""
    
    # Event PnL (just this row's contribution)
    pnl_event: float = 0.0
    # Total trade PnL (only meaningful on FINAL row)
    pnl_total_trade: float = 0.0
    
    # Lot tracking
    initial_lots: float = 0.0
    tp1_closed_lots: float = 0.0
    tp2_closed_lots: float = 0.0
    runner_closed_lots: float = 0.0
    
    # State transitions
    sl_moved_to_be: bool = False
    sl_moved_to_profit: bool = False
    be_trigger_time: Optional[datetime] = None
    tp1_hit_time: Optional[datetime] = None
    tp2_hit_time: Optional[datetime] = None
    
    # R-values at each exit
    r_at_exit: float = 0.0
    tp1_r: float = 0.0
    tp2_r: float = 0.0
    
    # PnL breakdown
    tp1_pnl: float = 0.0
    tp2_pnl: float = 0.0
    runner_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    pair: str
    strategy: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float
    net_profit: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    sharpe_ratio: float
    sortino_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    return_percent: float
    avg_trade_duration: float
    
    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            "=" * 64,
            "  BACKTEST RESULTS",
            "=" * 64,
            f"  Pair: {self.pair}",
            f"  Strategy: {self.strategy}",
            f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "-" * 64,
            "  PERFORMANCE",
            f"    Initial Balance:    ${self.initial_balance:>12,.2f}",
            f"    Final Balance:      ${self.final_balance:>12,.2f}",
            f"    Net Profit:         ${self.net_profit:>+12,.2f}",
            f"    Return:             {self.return_percent:>12.2f}%",
            "-" * 64,
            "  TRADES",
            f"    Total Trades:       {self.total_trades:>12}",
            f"    Winning Trades:     {self.winning_trades:>12}",
            f"    Losing Trades:      {self.losing_trades:>12}",
            f"    Win Rate:           {self.win_rate:>12.1f}%",
            f"    Profit Factor:      {self.profit_factor:>12.2f}",
            "-" * 64,
            "  AVERAGES",
            f"    Avg Win:            ${self.avg_win:>+12,.2f}",
            f"    Avg Loss:           ${self.avg_loss:>+12,.2f}",
            f"    Avg Trade:          ${self.avg_trade:>+12,.2f}",
            f"    Avg Duration:       {self.avg_trade_duration:>12.1f} min",
            "-" * 64,
            "  RISK",
            f"    Max Drawdown:       ${self.max_drawdown:>12,.2f}",
            f"    Max Drawdown %:     {self.max_drawdown_pct:>12.2f}%",
            f"    Sharpe Ratio:       {self.sharpe_ratio:>12.2f}",
            f"    Sortino Ratio:      {self.sortino_ratio:>12.2f}",
            "-" * 64,
            "  STREAKS",
            f"    Max Consecutive Wins:  {self.max_consecutive_wins:>9}",
            f"    Max Consecutive Losses:{self.max_consecutive_losses:>9}",
            "=" * 64,
        ]
        return "\n".join(lines)


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    Supports Model A partial exits with proper PnL tracking.
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        commission_per_lot: float = 7.0
    ):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting account balance
            spread_pips: Simulated spread in pips
            slippage_pips: Simulated slippage in pips
            commission_per_lot: Commission per lot traded
        """
        self.logger = get_logger(__name__)
        self.config = get_config()
        
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        
        self._balance = initial_balance
        self._equity = initial_balance
        self._candles: Dict[str, List[Candle]] = {}
        self._open_positions: List[BacktestTrade] = []
        self._closed_trades: List[BacktestTrade] = []
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._trade_counter = 0
        
        # Track managed trades separately
        self._managed_trades: Dict[str, ManagedTrade] = {}
        
        # Multi-timeframe candle storage for TDA
        self._candles_weekly: Dict[str, List[Candle]] = {}
        self._candles_daily: Dict[str, List[Candle]] = {}
        self._candles_h4: Dict[str, List[Candle]] = {}
    
    def _resample_candles_to_timeframe(
        self,
        candles: List[Candle],
        target_tf: str
    ) -> List[Candle]:
        """
        Resample H1 candles to higher timeframe (H4, D, W).
        
        This creates REAL multi-timeframe data for TDA.
        """
        if not candles:
            return []
        
        pair = candles[0].pair
        
        # Determine how many H1 candles make one higher TF candle
        tf_multipliers = {
            "4h": 4,    # 4 H1 candles = 1 H4
            "1d": 24,   # 24 H1 candles = 1 D1
            "1w": 120,  # ~120 H1 candles = 1 W1 (5 trading days)
        }
        
        multiplier = tf_multipliers.get(target_tf, 4)
        resampled = []
        
        i = 0
        while i + multiplier <= len(candles):
            chunk = candles[i:i + multiplier]
            
            # Aggregate OHLCV
            open_price = chunk[0].open
            high_price = max(c.high for c in chunk)
            low_price = min(c.low for c in chunk)
            close_price = chunk[-1].close
            volume = sum(c.volume for c in chunk)
            timestamp = chunk[0].timestamp
            
            resampled_candle = Candle(
                pair=pair,
                timeframe=target_tf,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            resampled.append(resampled_candle)
            i += multiplier
        
        return resampled
    
    def _prepare_multi_timeframe_data(self, pair: str, h1_candles: List[Candle]) -> None:
        """
        Prepare all timeframe data from H1 candles for TDA.
        
        This is called ONCE before the backtest loop starts.
        """
        self.logger.info(f"[BACKTEST] Preparing multi-timeframe data for TDA...")
        
        # Resample H1 to higher timeframes
        self._candles_h4[pair] = self._resample_candles_to_timeframe(h1_candles, "4h")
        self._candles_daily[pair] = self._resample_candles_to_timeframe(h1_candles, "1d")
        self._candles_weekly[pair] = self._resample_candles_to_timeframe(h1_candles, "1w")
        
        self.logger.info(f"  H1 candles: {len(h1_candles)}")
        self.logger.info(f"  H4 candles: {len(self._candles_h4[pair])}")
        self.logger.info(f"  Daily candles: {len(self._candles_daily[pair])}")
        self.logger.info(f"  Weekly candles: {len(self._candles_weekly[pair])}")
    
    def load_data_from_mt5(
        self,
        pair: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bars: Optional[int] = None
    ) -> int:
        """Load historical data from MetaTrader 5."""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            raise ImportError("MetaTrader5 package not installed")
        
        if not mt5.initialize():
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        
        # Convert pair format
        symbol = pair.replace("/", "")
        
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            mt5.shutdown()
            raise ValueError(f"Symbol {symbol} not found")
        
        # Map timeframe
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe.lower(), mt5.TIMEFRAME_H1)
        
        # Get data
        if bars:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        elif start_date and end_date:
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
        else:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 10000)
        
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol}")
        
        # Convert to candles
        candles = []
        for rate in rates:
            candle = Candle(
                pair=pair,
                timeframe=timeframe,
                timestamp=datetime.fromtimestamp(rate['time']),
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=float(rate['tick_volume'])
            )
            candles.append(candle)
        
        self._candles[pair] = candles
        self.logger.info(f"Loaded {len(candles)} candles for {pair}")
        return len(candles)
    
    def generate_sample_data(
        self,
        pair: str,
        days: int = 30,
        timeframe: str = "1h",
        base_price: float = 1.1000,
        volatility: float = 0.0005
    ) -> int:
        """Generate sample price data for testing."""
        import random
        
        candles = []
        price = base_price
        
        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe.lower(), 60)
        candles_per_day = int(24 * 60 / minutes)
        
        start_time = datetime.now() - timedelta(days=days)
        
        for i in range(days * candles_per_day):
            timestamp = start_time + timedelta(minutes=i * minutes)
            
            change = random.gauss(0, volatility)
            open_price = price
            close_price = price + change
            high_price = max(open_price, close_price) + random.uniform(0, volatility)
            low_price = min(open_price, close_price) - random.uniform(0, volatility)
            
            candle = Candle(
                pair=pair,
                timeframe=timeframe,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(100, 1000)
            )
            candles.append(candle)
            price = close_price
        
        self._candles[pair] = candles
        self.logger.info(f"Generated {len(candles)} sample candles for {pair}")
        return len(candles)
    
    def run(
        self,
        pair: str,
        strategy: str = "trend"
    ) -> BacktestResult:
        """
        Run backtest on loaded data.
        
        Args:
            pair: Currency pair to test
            strategy: Strategy to use ("grid", "trend", or "both")
            
        Returns:
            BacktestResult with performance metrics
        """
        if pair not in self._candles:
            raise ValueError(f"No data loaded for {pair}")
        
        candles = self._candles[pair]
        if len(candles) < 50:
            raise ValueError("Insufficient data for backtest (need at least 50 candles)")
        
        # Reset state
        self._balance = self.initial_balance
        self._equity = self.initial_balance
        self._open_positions = []
        self._closed_trades = []
        self._equity_curve = []
        self._managed_trades = {}
        self._trade_counter = 0
        
        # Get pip size from symbol spec
        from .symbol_spec import get_symbol_spec
        symbol_spec = get_symbol_spec(pair)
        pip_size = symbol_spec.pip_size
        self.logger.info(f"[BACKTEST] Using symbol spec for {pair}: pip_size={pip_size}, pip_value=${symbol_spec.pip_value_per_lot}/lot")
        
        # Initialize risk manager
        risk_manager = RiskManager(self.config.risk_management)
        
        # Check if partial exits are enabled
        import json
        try:
            with open("config/settings.json", "r") as f:
                raw_config = json.load(f)
                partial_exits_config = raw_config.get("risk_management", {}).get("partial_exits", {})
        except:
            partial_exits_config = {}
        
        use_partial_exits = partial_exits_config.get("enabled", False)
        
        # Initialize TradeManager if partial exits enabled
        trade_manager = None
        if use_partial_exits:
            rm_config = self.config.risk_management
            trade_mgr_config = {
                "partial_exits": partial_exits_config,
                "breakeven_trigger_r": getattr(rm_config, 'breakeven_trigger_r', 1.2),
                "breakeven_offset_r": getattr(rm_config, 'breakeven_offset_r', 0.1),
                "post_tp1_stop_r": getattr(rm_config, 'post_tp1_stop_r', 0.2),
                "trailing": getattr(rm_config, 'trailing', {}),
                "max_total_risk_percent": getattr(rm_config, 'max_total_risk_percent', 2.0),
                "max_concurrent_trades": getattr(rm_config, 'max_concurrent_trades', 2),
                "avoid_correlated_pairs": getattr(rm_config, 'avoid_correlated_pairs', True),
                "daily_loss_limit_r": getattr(rm_config, 'daily_loss_limit_r', 2.0),
                "weekly_loss_limit_r": getattr(rm_config, 'weekly_loss_limit_r', 6.0),
            }
            trade_manager = TradeManager(trade_mgr_config)
            self.logger.info("[BACKTEST] TradeManager enabled for Model A partial exits")
        
        # Initialize strategy
        trend_follower = None
        grid_trader = None
        
        if strategy in ("trend", "both"):
            trend_follower = TrendFollower(self.config.trend_strategy, risk_manager)
            
            # FIX #1: Prepare REAL multi-timeframe data for TDA
            self._prepare_multi_timeframe_data(pair, candles)
        
        if strategy in ("grid", "both"):
            grid_trader = GridTrader(self.config.grid_strategy, risk_manager)
            grid_trader.initialize_grid(pair, candles[0].close)
        
        # Trading state
        trading_halted = False
        halt_reason = ""
        
        # Track current position in higher timeframe data
        h4_idx = 0
        daily_idx = 0
        weekly_idx = 0
        
        # Run through candles
        for i, candle in enumerate(candles):
            # Create market data
            spread = self.spread_pips * pip_size
            market_data = MarketData(
                pair=pair,
                bid=candle.close - spread/2,
                ask=candle.close + spread/2,
                timestamp=candle.timestamp
            )
            
            # Calculate ATR for trailing
            current_atr = self._calculate_atr(candles[max(0, i-20):i+1])
            
            # Update managed trades
            if trade_manager and not trading_halted:
                for trade_id in list(trade_manager.active_trades.keys()):
                    actions = trade_manager.update_trade(
                        trade_id=trade_id,
                        current_price=candle.close,
                        current_close=candle.close,
                        current_atr=current_atr,
                        timestamp=candle.timestamp
                    )
                    self._execute_trade_actions(actions, candle, pip_size, trade_manager)
            
            # Check legacy positions (non-managed)
            if not trade_manager:
                self._check_positions(candle, pip_size)
            
            # Check drawdown limits
            current_dd_pct = ((self.initial_balance - self._equity) / self.initial_balance) * 100
            max_dd = getattr(self.config.risk_management, 'max_drawdown_percent', 25)
            if current_dd_pct >= max_dd:
                if not trading_halted:
                    trading_halted = True
                    halt_reason = f"Max drawdown {current_dd_pct:.1f}%"
                    self.logger.warning(f"[BACKTEST] HALTED: {halt_reason}")
            
            # Create account state
            account_state = AccountState(
                balance=self._balance,
                equity=self._equity,
                margin_used=0,
                free_margin=self._equity,
                margin_level=100,
                leverage=30
            )
            
            # Generate signals
            signals = []
            
            if not trading_halted:
                if trend_follower:
                    trend_follower.add_candle(candle)
                    
                    # FIX #1: Inject multi-timeframe data for TDA
                    # Calculate which higher TF candles are available at this point
                    h4_available = [c for c in self._candles_h4.get(pair, []) if c.timestamp <= candle.timestamp]
                    daily_available = [c for c in self._candles_daily.get(pair, []) if c.timestamp <= candle.timestamp]
                    weekly_available = [c for c in self._candles_weekly.get(pair, []) if c.timestamp <= candle.timestamp]
                    
                    # Get rolling windows (last N candles of each timeframe)
                    h1_window = candles[max(0, i-100):i+1]
                    h4_window = h4_available[-50:] if h4_available else h1_window
                    daily_window = daily_available[-30:] if daily_available else h1_window
                    weekly_window = weekly_available[-52:] if weekly_available else h1_window
                    
                    # Inject into TrendFollower for TDA
                    trend_follower.set_multi_timeframe_data(
                        pair=pair,
                        weekly=weekly_window,
                        daily=daily_window,
                        h4=h4_window,
                        h1=h1_window
                    )
                    
                    if trend_follower.is_ready(pair):
                        trend_signals = trend_follower.generate_signals(pair, market_data, account_state)
                        signals.extend(trend_signals)
                
                if grid_trader:
                    grid_trader.update_price(pair, candle.close)
                    grid_signals = grid_trader.generate_grid_signals(pair, market_data, account_state)
                    signals.extend(grid_signals)
            
            # Execute signals
            for signal in signals:
                if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                    if trade_manager:
                        self._open_managed_position(signal, candle, pip_size, trade_manager, risk_manager)
                    else:
                        self._open_position(signal, candle, pip_size)
            
            # Update equity
            self._update_equity(candle, trade_manager)
            self._equity_curve.append((candle.timestamp, self._equity))
        
        # Close remaining positions
        if trade_manager:
            for trade_id in list(trade_manager.active_trades.keys()):
                result = trade_manager.close_trade(
                    trade_id, candles[-1].close, "End of backtest",
                    timestamp=candles[-1].timestamp
                )
                if result:
                    self._record_managed_close(result, candles[-1], pip_size, trade_manager)
        
        for pos in self._open_positions[:]:
            self._close_position(pos, candles[-1], "End of backtest")
        
        return self._calculate_results(pair, candles)
    
    def _open_managed_position(
        self,
        signal: Signal,
        candle: Candle,
        pip_size: float,
        trade_manager: TradeManager,
        risk_manager: RiskManager
    ) -> None:
        """Open a managed position using TradeManager."""
        slippage = self.slippage_pips * pip_size
        if signal.signal_type == SignalType.BUY:
            entry_price = candle.close + slippage
            side = OrderSide.BUY
        else:
            entry_price = candle.close - slippage
            side = OrderSide.SELL
        
        lot_size = signal.metadata.get("lot_size", 0.01)
        r_value = signal.metadata.get("r_value", abs(entry_price - signal.stop_loss))
        
        # Calculate risk amount
        risk_amount_usd = lot_size * r_value * self._get_pip_value(signal.pair, lot_size) / pip_size
        
        self._trade_counter += 1
        trade_id = f"BT_{self._trade_counter}"
        
        # Create managed trade (TradeManager sets lot allocation + TP prices)
        managed_trade = trade_manager.create_trade(
            trade_id=trade_id,
            pair=signal.pair,
            side=side,
            entry_price=entry_price,
            stop_loss_price=signal.stop_loss,
            lot_size=lot_size,
            risk_amount_usd=risk_amount_usd,
            module=signal.module,
            timestamp=candle.timestamp
        )
        
        if not managed_trade:
            self.logger.info(f"[BACKTEST] Trade blocked by TradeManager (R-limits or correlation)")
            return
        
        # Track in open positions (for equity calculation)
        trade = BacktestTrade(
            entry_time=candle.timestamp,
            exit_time=None,
            pair=signal.pair,
            side=side,
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=None,
            stop_loss=signal.stop_loss or 0,
            take_profit=0,
            module=signal.module,
            trade_id=trade_id,
            row_type=RowType.ENTRY,
            is_partial=False,
            initial_lots=lot_size
        )
        self._open_positions.append(trade)
        
        self.logger.info(f"[BACKTEST] ENTRY: {trade_id} {side.value} {lot_size} {signal.pair} @ {entry_price:.5f}")
    
    def _execute_trade_actions(
        self,
        actions: List[dict],
        candle: Candle,
        pip_size: float,
        trade_manager: TradeManager
    ) -> None:
        """Execute actions from TradeManager with proper PnL tracking."""
        for action in actions:
            action_type = action.get("action")
            trade_id = action.get("trade_id")
            
            if action_type == "CLOSE_PARTIAL":
                lots = action.get("lots", 0)
                price = action.get("price", candle.close)
                pnl = action.get("pnl", 0)
                reason = action.get("reason", "Partial close")
                
                self.logger.info(f"[BACKTEST] {reason}: {trade_id} closed {lots} lots @ {price:.5f} (${pnl:+.2f})")
                
                managed_trade = trade_manager.active_trades.get(trade_id)
                if managed_trade:
                    is_tp1 = "TP1" in reason
                    is_tp2 = "TP2" in reason
                    
                    # Create partial exit row
                    partial_trade = BacktestTrade(
                        entry_time=managed_trade.entry_time,
                        exit_time=candle.timestamp,
                        pair=managed_trade.pair,
                        side=managed_trade.side,
                        lot_size=lots,
                        entry_price=managed_trade.entry_price,
                        exit_price=price,
                        stop_loss=managed_trade.current_stop_loss,
                        take_profit=0,
                        pnl=pnl,  # Event PnL
                        pnl_event=pnl,  # Same for partials
                        pnl_total_trade=0,  # Not final row
                        exit_reason=reason,
                        module=managed_trade.module,
                        trade_id=trade_id,
                        row_type=RowType.TP1_PARTIAL if is_tp1 else RowType.TP2_PARTIAL,
                        is_partial=True,
                        r_multiple=managed_trade.current_r_multiple,
                        trade_state=managed_trade.state.value,
                        initial_lots=managed_trade.initial_lot_size,
                        tp1_closed_lots=lots if is_tp1 else 0,
                        tp2_closed_lots=lots if is_tp2 else 0,
                        sl_moved_to_be=True,
                        sl_moved_to_profit=is_tp1 or is_tp2,
                        tp1_hit_time=candle.timestamp if is_tp1 else None,
                        tp2_hit_time=candle.timestamp if is_tp2 else None,
                        tp1_r=managed_trade.current_r_multiple if is_tp1 else 0,
                        tp2_r=managed_trade.current_r_multiple if is_tp2 else 0,
                        tp1_pnl=pnl if is_tp1 else 0,
                        tp2_pnl=pnl if is_tp2 else 0
                    )
                    self._closed_trades.append(partial_trade)
                    self._balance += pnl
                    
                    # Update main position record
                    for pos in self._open_positions:
                        if pos.trade_id == trade_id:
                            if is_tp1:
                                pos.tp1_closed_lots = lots
                                pos.tp1_pnl = pnl
                                pos.tp1_hit_time = candle.timestamp
                            elif is_tp2:
                                pos.tp2_closed_lots = lots
                                pos.tp2_pnl = pnl
                                pos.tp2_hit_time = candle.timestamp
                            pos.sl_moved_to_be = True
                            pos.sl_moved_to_profit = True
                            break
            
            elif action_type == "CLOSE_ALL":
                lots = action.get("lots", 0)
                price = action.get("price", candle.close)
                pnl = action.get("pnl", 0)  # This is just the runner/final piece PnL
                total_pnl = action.get("total_pnl", pnl)  # Total including partials
                reason = action.get("reason", "Closed")
                
                self.logger.info(f"[BACKTEST] {reason}: {trade_id} closed ALL @ {price:.5f} (final: ${total_pnl:+.2f})")
                
                # Get managed trade for full info
                managed_trade = None
                for t in trade_manager.closed_trades:
                    if t.trade_id == trade_id:
                        managed_trade = t
                        break
                
                # Update and close position
                for pos in self._open_positions[:]:
                    if pos.trade_id == trade_id:
                        pos.exit_time = candle.timestamp
                        pos.exit_price = price
                        pos.exit_reason = reason
                        pos.trade_state = reason
                        pos.row_type = RowType.FINAL
                        pos.is_partial = False
                        
                        # PnL tracking - FIXED
                        pos.pnl_event = pnl  # Just this row's PnL (runner/final piece)
                        pos.pnl_total_trade = total_pnl  # Total trade PnL
                        pos.pnl = pnl  # For backward compatibility, use event pnl
                        
                        pos.runner_closed_lots = lots
                        pos.runner_pnl = pnl
                        pos.r_at_exit = managed_trade.current_r_multiple if managed_trade else 0
                        
                        if managed_trade:
                            pos.tp1_pnl = managed_trade.tp1_pnl
                            pos.tp2_pnl = managed_trade.tp2_pnl
                            pos.tp1_hit_time = managed_trade.tp1_time
                            pos.tp2_hit_time = managed_trade.tp2_time
                        
                        self._open_positions.remove(pos)
                        self._closed_trades.append(pos)
                        self._balance += pnl  # Only add final piece
                        break
            
            elif action_type == "MODIFY_SL":
                new_sl = action.get("new_sl")
                reason = action.get("reason", "SL modified")
                
                self.logger.info(f"[BACKTEST] {reason}: {trade_id} SL -> {new_sl:.5f}")
                
                for pos in self._open_positions:
                    if pos.trade_id == trade_id:
                        pos.stop_loss = new_sl
                        if "breakeven" in reason.lower() or "Breakeven" in reason:
                            pos.sl_moved_to_be = True
                            pos.be_trigger_time = candle.timestamp
                        if "profit" in reason.lower() or "TP1" in reason:
                            pos.sl_moved_to_profit = True
                        break
    
    def _record_managed_close(
        self,
        result: dict,
        candle: Candle,
        pip_size: float,
        trade_manager: TradeManager
    ) -> None:
        """Record a managed trade close from TradeManager."""
        # This is called for end-of-backtest closes
        self._execute_trade_actions([result], candle, pip_size, trade_manager)
    
    def _open_position(self, signal: Signal, candle: Candle, pip_size: float) -> None:
        """Open a legacy (non-managed) position."""
        slippage = self.slippage_pips * pip_size
        if signal.signal_type == SignalType.BUY:
            entry_price = candle.close + slippage
            side = OrderSide.BUY
        else:
            entry_price = candle.close - slippage
            side = OrderSide.SELL
        
        self._trade_counter += 1
        trade = BacktestTrade(
            entry_time=candle.timestamp,
            exit_time=None,
            pair=signal.pair,
            side=side,
            lot_size=signal.metadata.get("lot_size", 0.01),
            entry_price=entry_price,
            exit_price=None,
            stop_loss=signal.stop_loss or 0,
            take_profit=signal.take_profit or 0,
            module=signal.module,
            trade_id=f"BT_{self._trade_counter}",
            row_type=RowType.FINAL
        )
        self._open_positions.append(trade)
    
    def _check_positions(self, candle: Candle, pip_size: float) -> None:
        """Check legacy positions for SL/TP hits."""
        for pos in self._open_positions[:]:
            if pos.stop_loss > 0:
                if pos.side == OrderSide.BUY and candle.low <= pos.stop_loss:
                    self._close_position(pos, candle, "Stop Loss", pos.stop_loss)
                    continue
                elif pos.side == OrderSide.SELL and candle.high >= pos.stop_loss:
                    self._close_position(pos, candle, "Stop Loss", pos.stop_loss)
                    continue
            
            if pos.take_profit > 0:
                if pos.side == OrderSide.BUY and candle.high >= pos.take_profit:
                    self._close_position(pos, candle, "Take Profit", pos.take_profit)
                    continue
                elif pos.side == OrderSide.SELL and candle.low <= pos.take_profit:
                    self._close_position(pos, candle, "Take Profit", pos.take_profit)
                    continue
    
    def _close_position(
        self,
        pos: BacktestTrade,
        candle: Candle,
        reason: str,
        exit_price: Optional[float] = None
    ) -> None:
        """Close a position."""
        if exit_price is None:
            exit_price = candle.close
        
        from .symbol_spec import get_symbol_spec
        spec = get_symbol_spec(pos.pair)
        
        if pos.side == OrderSide.BUY:
            pnl_pips = (exit_price - pos.entry_price) / spec.pip_size
        else:
            pnl_pips = (pos.entry_price - exit_price) / spec.pip_size
        
        pip_value = spec.pip_value_per_lot * pos.lot_size
        pnl = pnl_pips * pip_value
        pnl -= self.commission_per_lot * pos.lot_size
        
        pos.exit_time = candle.timestamp
        pos.exit_price = exit_price
        pos.pnl = pnl
        pos.pnl_event = pnl
        pos.pnl_total_trade = pnl
        pos.pnl_pips = pnl_pips
        pos.exit_reason = reason
        pos.row_type = RowType.FINAL
        
        self._open_positions.remove(pos)
        self._closed_trades.append(pos)
        self._balance += pnl
    
    def _update_equity(self, candle: Candle, trade_manager: Optional[TradeManager] = None) -> None:
        """Update current equity with unrealized P&L."""
        from .symbol_spec import get_symbol_spec
        spec = get_symbol_spec(candle.pair)
        unrealized = 0.0
        
        if trade_manager:
            for trade in trade_manager.active_trades.values():
                if trade.side == OrderSide.BUY:
                    pips = (candle.close - trade.entry_price) / spec.pip_size
                else:
                    pips = (trade.entry_price - candle.close) / spec.pip_size
                pip_value = spec.pip_value_per_lot * trade.remaining_lot_size
                unrealized += pips * pip_value
        else:
            for pos in self._open_positions:
                if pos.side == OrderSide.BUY:
                    pips = (candle.close - pos.entry_price) / spec.pip_size
                else:
                    pips = (pos.entry_price - candle.close) / spec.pip_size
                unrealized += pips * spec.pip_value_per_lot * pos.lot_size
        
        self._equity = self._balance + unrealized
    
    def _get_pip_value(self, pair: str, lot_size: float = 1.0) -> float:
        """Get pip value for a currency pair using symbol spec."""
        from .symbol_spec import get_symbol_spec
        spec = get_symbol_spec(pair)
        return spec.pip_value_per_lot * lot_size
    
    def _calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(candles) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        if not true_ranges:
            return 0.0
        
        return sum(true_ranges[-period:]) / min(period, len(true_ranges))
    
    def _calculate_results(self, pair: str, candles: List[Candle]) -> BacktestResult:
        """Calculate backtest performance metrics."""
        # Only count FINAL rows for trade metrics
        final_trades = [t for t in self._closed_trades if t.row_type == RowType.FINAL]
        
        if not final_trades:
            return BacktestResult(
                pair=pair, strategy="", start_date=candles[0].timestamp,
                end_date=candles[-1].timestamp, initial_balance=self.initial_balance,
                final_balance=self._balance, total_trades=0, winning_trades=0,
                losing_trades=0, gross_profit=0, gross_loss=0, net_profit=0,
                max_drawdown=0, max_drawdown_pct=0, win_rate=0, profit_factor=0,
                avg_win=0, avg_loss=0, avg_trade=0, sharpe_ratio=0, sortino_ratio=0,
                max_consecutive_wins=0, max_consecutive_losses=0, return_percent=0,
                avg_trade_duration=0
            )
        
        # Use pnl_total_trade for final rows
        winning = [t for t in final_trades if t.pnl_total_trade > 0]
        losing = [t for t in final_trades if t.pnl_total_trade <= 0]
        
        gross_profit = sum(t.pnl_total_trade for t in winning)
        gross_loss = abs(sum(t.pnl_total_trade for t in losing))
        net_profit = gross_profit - gross_loss
        
        # Calculate max drawdown from equity curve
        max_dd = 0
        peak = self.initial_balance
        for _, equity in self._equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        max_dd_pct = (max_dd / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Win rate
        win_rate = (len(winning) / len(final_trades)) * 100 if final_trades else 0
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Averages
        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = -gross_loss / len(losing) if losing else 0
        avg_trade = net_profit / len(final_trades) if final_trades else 0
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_total_trade for t in final_trades]
        if len(returns) > 1:
            import statistics
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Sortino ratio
        neg_returns = [r for r in returns if r < 0]
        if neg_returns and len(neg_returns) > 1:
            import statistics
            downside_std = statistics.stdev(neg_returns)
            mean_return = statistics.mean(returns)
            sortino = (mean_return / downside_std) * (252 ** 0.5) if downside_std > 0 else 0
        else:
            sortino = sharpe * 2  # Approximate if no negative returns
        
        # Consecutive wins/losses
        max_wins = max_losses = current_wins = current_losses = 0
        for t in final_trades:
            if t.pnl_total_trade > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Average duration
        durations = []
        for t in final_trades:
            if t.exit_time and t.entry_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(duration)
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return BacktestResult(
            pair=pair,
            strategy="trend",
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            initial_balance=self.initial_balance,
            final_balance=self._balance,
            total_trades=len(final_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            return_percent=(net_profit / self.initial_balance) * 100,
            avg_trade_duration=avg_duration
        )
    
    def export_trades_csv(self, filepath: str) -> None:
        """
        Export trades to CSV with proper Model A tracking.
        
        PnL columns explained:
        - pnl_event: PnL for THIS row only (partial or final piece)
        - pnl_total_trade: Total trade PnL (only on FINAL rows)
        - pnl: Legacy field (same as pnl_event)
        
        Use row_type to filter:
        - FINAL rows for trade-level stats
        - All rows for detailed event analysis
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                # Identifiers
                'trade_id', 'row_type', 'is_partial',
                
                # Timing
                'entry_time', 'exit_time',
                
                # Trade info
                'pair', 'side', 'entry_price', 'exit_price', 'stop_loss',
                
                # Lots
                'initial_lots', 'lot_size', 'tp1_closed_lots', 'tp2_closed_lots', 'runner_closed_lots',
                
                # Exit info
                'exit_reason', 'trade_state',
                
                # PnL - CLEAR SEPARATION
                'pnl_event', 'pnl_total_trade', 'pnl_pips',
                'tp1_pnl', 'tp2_pnl', 'runner_pnl',
                
                # R-multiples
                'r_at_exit', 'tp1_r', 'tp2_r',
                
                # State transitions
                'sl_moved_to_be', 'sl_moved_to_profit',
                'be_trigger_time', 'tp1_hit_time', 'tp2_hit_time',
                
                # Module
                'module'
            ])
            
            for t in self._closed_trades:
                writer.writerow([
                    t.trade_id,
                    t.row_type.value,
                    'Yes' if t.is_partial else 'No',
                    
                    t.entry_time.isoformat() if t.entry_time else '',
                    t.exit_time.isoformat() if t.exit_time else '',
                    
                    t.pair,
                    t.side.value,
                    f"{t.entry_price:.5f}",
                    f"{t.exit_price:.5f}" if t.exit_price else '',
                    f"{t.stop_loss:.5f}" if t.stop_loss else '',
                    
                    f"{t.initial_lots:.2f}",
                    f"{t.lot_size:.2f}",
                    f"{t.tp1_closed_lots:.2f}",
                    f"{t.tp2_closed_lots:.2f}",
                    f"{t.runner_closed_lots:.2f}",
                    
                    t.exit_reason,
                    t.trade_state,
                    
                    f"{t.pnl_event:.2f}",
                    f"{t.pnl_total_trade:.2f}",
                    f"{t.pnl_pips:.1f}",
                    f"{t.tp1_pnl:.2f}",
                    f"{t.tp2_pnl:.2f}",
                    f"{t.runner_pnl:.2f}",
                    
                    f"{t.r_at_exit:.2f}",
                    f"{t.tp1_r:.2f}",
                    f"{t.tp2_r:.2f}",
                    
                    'Yes' if t.sl_moved_to_be else 'No',
                    'Yes' if t.sl_moved_to_profit else 'No',
                    t.be_trigger_time.isoformat() if t.be_trigger_time else '',
                    t.tp1_hit_time.isoformat() if t.tp1_hit_time else '',
                    t.tp2_hit_time.isoformat() if t.tp2_hit_time else '',
                    
                    t.module.value
                ])
        
        self.logger.info(f"Exported {len(self._closed_trades)} trades to {filepath}")
