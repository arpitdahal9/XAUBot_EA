"""
Trend Following Module - Enhanced with Arpit's TPU Strategy

5-Step Entry Confirmation Process:
1. Top-Down Analysis - Establish macro trend bias (Weekly → Daily → H4 → H1)
2. Fibonacci Retracement - Find optimal entry at 88.6%
3. Divergence Detection - Confirm with RSI divergence
4. Pattern Recognition - Validate with Double Top/Bottom
5. Execute Precisely - Use Fibonacci-based targets

This dramatically reduces false signals compared to simple SMA crossovers.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from .models import (
    Order, OrderSide, OrderStatus, MarketData, Candle,
    Signal, SignalType, TradingModule, AccountState
)
from .config import TrendStrategyConfig, get_config
from .risk_manager import RiskManager, PositionSizeResult
from .logger import TradingBotLogger, get_logger

# New imports for enhanced strategy
from .fibonacci import FibonacciCalculator, FibDirection, FibonacciLevels
from .divergence import DivergenceDetector, DivergenceType, DivergenceStrength, RSICalculator
from .top_down_analyzer import TopDownAnalyzer, TrendBias, AlignmentQuality
from .pattern_recognizer import PatternRecognizer, PatternType, PatternSignal

# ML Filter imports
from .ml_filter import MLTradeFilter, MLPrediction
from .news_filter import NewsFilter


@dataclass
class TrendState:
    """Current state of trend analysis."""
    sma_short: float = 0.0
    sma_long: float = 0.0
    is_bullish: bool = False
    is_bearish: bool = False
    signal_strength: float = 0.0
    consecutive_higher_highs: int = 0
    consecutive_lower_lows: int = 0
    last_crossover: Optional[str] = None
    last_crossover_time: Optional[datetime] = None
    
    # Enhanced analysis state
    tda_alignment: Optional[str] = None
    tda_bias: Optional[str] = None
    fib_entry_zone: bool = False
    divergence_type: Optional[str] = None
    pattern_detected: Optional[str] = None


@dataclass
class EnhancedSignalResult:
    """Result of enhanced signal analysis."""
    can_trade: bool
    direction: Optional[str]  # "BUY" or "SELL"
    confidence: float
    
    # Analysis components
    tda_alignment: str
    tda_confidence: float
    at_fib_entry: bool
    fib_distance_pips: float
    has_divergence: bool
    divergence_type: Optional[str]
    has_pattern: bool
    pattern_type: Optional[str]
    
    # Trade parameters
    entry_price: float
    stop_loss: float
    r_value: float
    
    reason: str
    
    # ML Filter results
    ml_probability: float = 0.0
    ml_risk_multiplier: float = 1.0
    ml_passed_filter: bool = True
    
    # Raw analysis objects for ML feature extraction
    tda_result: Any = None
    fib_levels: Any = None
    divergence_result: Any = None
    pattern_result: Any = None


class SimpleMovingAverage:
    """
    Simple Moving Average calculator with efficient updates.
    """

    def __init__(self, period: int):
        self.period = period
        self._prices: Deque[float] = deque(maxlen=period)
        self._sum: float = 0.0

    def add(self, price: float) -> Optional[float]:
        if len(self._prices) == self.period:
            self._sum -= self._prices[0]
        
        self._prices.append(price)
        self._sum += price
        
        if len(self._prices) == self.period:
            return self._sum / self.period
        return None

    @property
    def value(self) -> Optional[float]:
        if len(self._prices) == self.period:
            return self._sum / self.period
        return None

    @property
    def is_ready(self) -> bool:
        return len(self._prices) == self.period

    def clear(self) -> None:
        self._prices.clear()
        self._sum = 0.0


class TrendFollower:
    """
    Enhanced Trend Following Module with Arpit's TPU Strategy.
    
    Entry Conditions (ALL must be met for high-probability trade):
    1. TDA shows GOOD or PERFECT alignment (3-4 timeframes agree)
    2. Price is at or near 88.6% Fibonacci retracement
    3. Divergence confirms the reversal (RSI divergence)
    4. Optional: Pattern confirms (Double Top/Bottom)
    
    This reduces false signals by ~50% compared to simple SMA crossovers.
    """

    def __init__(
        self,
        config: Optional[TrendStrategyConfig] = None,
        risk_manager: Optional[RiskManager] = None,
        logger: Optional[TradingBotLogger] = None
    ):
        self.config = config or get_config().trend_strategy
        self.risk_manager = risk_manager or RiskManager()
        self.logger = logger or get_logger("TrendFollower")
        
        # Load enhanced strategy config
        self._load_enhanced_config()
        
        # Initialize enhanced analysis components
        self.fib_calculator = FibonacciCalculator()
        self.divergence_detector = DivergenceDetector(
            rsi_period=self._rsi_period,
            lookback=self._divergence_lookback
        )
        self.rsi_calculator = RSICalculator(period=self._rsi_period)
        self.tda_analyzer = TopDownAnalyzer(
            short_sma_period=self.config.sma_period_short,
            long_sma_period=self.config.sma_period_long
        )
        self.pattern_recognizer = PatternRecognizer(
            tolerance_pips=self._pattern_tolerance_pips
        )
        
        # Initialize ML Filter
        self._init_ml_filter()
        
        # Initialize News Filter (backtest only)
        self._init_news_filter()
        
        # Moving averages per pair (fallback)
        self._sma_short: Dict[str, SimpleMovingAverage] = {}
        self._sma_long: Dict[str, SimpleMovingAverage] = {}
        
        # Price history for analysis
        self._candle_history: Dict[str, Deque[Candle]] = {}
        self._max_history = 200  # More history for TDA
        
        # Multi-timeframe candle storage
        self._candles_weekly: Dict[str, List] = {}
        self._candles_daily: Dict[str, List] = {}
        self._candles_h4: Dict[str, List] = {}
        self._candles_h1: Dict[str, List] = {}
        
        # State tracking
        self._states: Dict[str, TrendState] = {}
        self._active_positions: Dict[str, Order] = {}
        self._previous_sma_short: Dict[str, float] = {}
        self._previous_sma_long: Dict[str, float] = {}
        
        self._log_initialization()
    
    def _load_enhanced_config(self):
        """Load enhanced strategy configuration."""
        try:
            import json
            with open("config/settings.json", "r") as f:
                raw_config = json.load(f)
            
            # Partial exits
            partial_config = raw_config.get("risk_management", {}).get("partial_exits", {})
            self._use_partial_exits = partial_config.get("enabled", False)
            
            # TDA config
            tda_config = raw_config.get("tda", {})
            self._tda_enabled = tda_config.get("enabled", True)
            self._tda_min_alignment = tda_config.get("min_alignment_quality", "GOOD")
            
            # Fibonacci config
            fib_config = raw_config.get("fibonacci", {})
            self._fib_enabled = fib_config.get("enabled", True)
            self._fib_entry_level = fib_config.get("entry_at_level", 0.886)
            self._fib_tolerance_pips = fib_config.get("proximity_tolerance_pips", 20)
            self._fib_lookback = fib_config.get("lookback_candles", 20)
            
            # Divergence config
            div_config = raw_config.get("divergence", {})
            self._divergence_enabled = div_config.get("enabled", True)
            self._rsi_period = div_config.get("rsi_period", 14)
            self._divergence_lookback = div_config.get("lookback_candles", 20)
            self._require_divergence = div_config.get("require_confirmation", True)
            
            # Pattern config
            pattern_config = raw_config.get("patterns", {})
            self._patterns_enabled = pattern_config.get("enabled", True)
            self._pattern_tolerance_pips = pattern_config.get("tolerance_pips", 15)
            self._require_pattern = pattern_config.get("require_pattern", False)
            
            # FIX #4: Confidence gate (minimum threshold to take a trade)
            self._min_confidence = 0.70  # Default: 70% minimum (lowered from 75%)
            signal_config = raw_config.get("signal_quality", {})
            self._min_confidence = signal_config.get("min_confidence", 0.70)
            self._use_tpu_lite = signal_config.get("use_tpu_lite", True)  # TPU-lite mode (replaces SMA fallback)
            
            # NEW: OR-gating mode (Divergence OR Pattern instead of AND)
            self._use_or_gating = signal_config.get("use_or_gating", True)
            
            # NEW: Multi-fib levels for trend regime
            self._fib_trend_levels = [0.618, 0.786, 0.886]  # Trend regime
            self._fib_reversal_level = 0.886  # Reversal regime (strict)
            
        except Exception as e:
            self.logger.warning(f"[TREND] Using default config: {e}")
            self._use_partial_exits = False
            self._tda_enabled = True
            self._tda_min_alignment = "GOOD"
            self._fib_enabled = True
            self._fib_entry_level = 0.886
            self._fib_tolerance_pips = 20
            self._fib_lookback = 20
            self._divergence_enabled = True
            self._rsi_period = 14
            self._divergence_lookback = 20
            self._require_divergence = True
            self._patterns_enabled = True
            self._pattern_tolerance_pips = 15
            self._require_pattern = False
            # FIX #4: Confidence gate defaults
            self._min_confidence = 0.70  # Lowered from 0.75
            self._use_tpu_lite = True  # Replaces SMA fallback
            self._use_or_gating = True  # Divergence OR Pattern
            self._fib_trend_levels = [0.618, 0.786, 0.886]
            self._fib_reversal_level = 0.886
            
            # ML Filter defaults
            self._ml_enabled = False
            self._ml_model_path = ""
            self._ml_probability_threshold = 0.58
            self._ml_base_risk = 0.75
            self._ml_min_risk_mult = 0.5
            self._ml_max_risk_mult = 1.5
            
            # News filter defaults
            self._news_filter_backtest = False
            self._news_avoidance_minutes = 30
    
    def _init_ml_filter(self):
        """Initialize ML Trade Filter."""
        try:
            import json
            with open("config/settings.json", "r") as f:
                raw_config = json.load(f)
            
            ml_config = raw_config.get("ml_filter", {})
            self._ml_enabled = ml_config.get("enabled", False)
            self._ml_model_path = ml_config.get("model_path", "")
            self._ml_probability_threshold = ml_config.get("probability_threshold", 0.58)
            self._ml_base_risk = ml_config.get("base_risk_percent", 0.75)
            self._ml_min_risk_mult = ml_config.get("min_risk_multiplier", 0.5)
            self._ml_max_risk_mult = ml_config.get("max_risk_multiplier", 1.5)
            
            if self._ml_enabled:
                import os
                model_path = self._ml_model_path if os.path.exists(self._ml_model_path) else None
                
                self.ml_filter = MLTradeFilter(
                    model_path=model_path,
                    probability_threshold=self._ml_probability_threshold,
                    base_risk_percent=self._ml_base_risk,
                    min_risk_multiplier=self._ml_min_risk_mult,
                    max_risk_multiplier=self._ml_max_risk_mult,
                    enabled=True
                )
                self.logger.info("[TREND] ML Filter: ENABLED")
            else:
                self.ml_filter = None
                self.logger.info("[TREND] ML Filter: DISABLED")
                
        except Exception as e:
            self.logger.warning(f"[TREND] ML Filter init failed: {e}")
            self.ml_filter = None
            self._ml_enabled = False
    
    def _init_news_filter(self):
        """Initialize News Filter for backtesting."""
        try:
            import json
            with open("config/settings.json", "r") as f:
                raw_config = json.load(f)
            
            news_config = raw_config.get("news_filter", {})
            self._news_filter_backtest = news_config.get("enabled_in_backtest", False)
            self._news_avoidance_minutes = news_config.get("avoidance_minutes", 30)
            
            if self._news_filter_backtest:
                self.news_filter = NewsFilter(
                    avoidance_minutes=self._news_avoidance_minutes
                )
                self.logger.info("[TREND] News Filter (Backtest): ENABLED")
            else:
                self.news_filter = None
                self.logger.info("[TREND] News Filter (Backtest): DISABLED")
                
        except Exception as e:
            self.logger.warning(f"[TREND] News Filter init failed: {e}")
            self.news_filter = None
            self._news_filter_backtest = False
    
    def _log_initialization(self):
        """Log initialization details."""
        self.logger.info("[TREND] Enhanced Trend Follower initialized")
        self.logger.info(f"  Short SMA: {self.config.sma_period_short} periods")
        self.logger.info(f"  Long SMA: {self.config.sma_period_long} periods")
        self.logger.info(f"  Timeframe: {self.config.timeframe}")
        
        self.logger.info(f"  [ENHANCED] Top-Down Analysis: {'ON' if self._tda_enabled else 'OFF'}")
        self.logger.info(f"  [ENHANCED] Fibonacci Entry: {'ON' if self._fib_enabled else 'OFF'}")
        self.logger.info(f"  [ENHANCED] Divergence: {'ON' if self._divergence_enabled else 'OFF'}")
        self.logger.info(f"  [ENHANCED] Patterns: {'ON' if self._patterns_enabled else 'OFF'}")
        self.logger.info(f"  [QUALITY] Min Confidence: {self._min_confidence:.0%}")
        self.logger.info(f"  [QUALITY] Gating: {'OR (Div OR Pat)' if self._use_or_gating else 'AND (Div AND Pat)'}")
        self.logger.info(f"  [QUALITY] Fib Levels: {self._fib_trend_levels} (trend) / {self._fib_reversal_level} (reversal)")
        self.logger.info(f"  [QUALITY] TPU-Lite: {'ON' if self._use_tpu_lite else 'OFF'}")
        
        if self._use_partial_exits:
            self.logger.info(f"  Exit Mode: PARTIAL (TradeManager controls TP1/TP2/Runner)")
        else:
            self.logger.info(f"  Take Profit: {self.config.take_profit_pips} pips")
        self.logger.info(f"  Stop Loss: {self.config.stop_loss_pips} pips")

    def _initialize_pair(self, pair: str) -> None:
        """Initialize tracking for a new pair."""
        if pair not in self._sma_short:
            self._sma_short[pair] = SimpleMovingAverage(self.config.sma_period_short)
            self._sma_long[pair] = SimpleMovingAverage(self.config.sma_period_long)
            self._candle_history[pair] = deque(maxlen=self._max_history)
            self._states[pair] = TrendState()
            self._previous_sma_short[pair] = 0.0
            self._previous_sma_long[pair] = 0.0
            
            # Multi-timeframe storage
            self._candles_weekly[pair] = []
            self._candles_daily[pair] = []
            self._candles_h4[pair] = []
            self._candles_h1[pair] = []
            
            self.logger.debug(f"[TREND] Initialized tracking for {pair}")

    def add_candle(self, candle: Candle) -> None:
        """
        Add a new candle and update all analysis components.
        """
        pair = candle.pair
        self._initialize_pair(pair)
        
        # Store previous SMA values
        if self._sma_short[pair].is_ready:
            self._previous_sma_short[pair] = self._sma_short[pair].value
        if self._sma_long[pair].is_ready:
            self._previous_sma_long[pair] = self._sma_long[pair].value
        
        # Update SMAs
        sma_short = self._sma_short[pair].add(candle.close)
        sma_long = self._sma_long[pair].add(candle.close)
        
        # Add to history
        self._candle_history[pair].append(candle)
        
        # Update state
        state = self._states[pair]
        if sma_short is not None:
            state.sma_short = sma_short
        if sma_long is not None:
            state.sma_long = sma_long
        
        if sma_short is not None and sma_long is not None:
            state.is_bullish = sma_short > sma_long
            state.is_bearish = sma_short < sma_long
        
        self.logger.debug(f"[TREND] {pair} candle added: Close={candle.close:.5f}")

    def set_multi_timeframe_data(
        self,
        pair: str,
        weekly: List = None,
        daily: List = None,
        h4: List = None,
        h1: List = None
    ) -> None:
        """
        Set multi-timeframe candle data for TDA.
        
        In live trading, this would be called periodically to update
        the higher timeframe data.
        """
        self._initialize_pair(pair)
        
        if weekly is not None:
            self._candles_weekly[pair] = weekly
        if daily is not None:
            self._candles_daily[pair] = daily
        if h4 is not None:
            self._candles_h4[pair] = h4
        if h1 is not None:
            self._candles_h1[pair] = h1

    def _run_enhanced_analysis(
        self,
        pair: str,
        market_data: MarketData
    ) -> EnhancedSignalResult:
        """
        Run the enhanced 5-step analysis process.
        
        Returns comprehensive analysis result.
        """
        current_price = market_data.mid_price
        candles = list(self._candle_history[pair])
        
        # Default result
        result = EnhancedSignalResult(
            can_trade=False,
            direction=None,
            confidence=0.0,
            tda_alignment="UNKNOWN",
            tda_confidence=0.0,
            at_fib_entry=False,
            fib_distance_pips=999,
            has_divergence=False,
            divergence_type=None,
            has_pattern=False,
            pattern_type=None,
            entry_price=current_price,
            stop_loss=0,
            r_value=0,
            reason="Insufficient data"
        )
        
        if len(candles) < self._fib_lookback + 5:
            return result
        
        # ===== STEP 1: Top-Down Analysis =====
        tda_result = None
        if self._tda_enabled:
            weekly = self._candles_weekly.get(pair, candles)
            daily = self._candles_daily.get(pair, candles)
            h4 = self._candles_h4.get(pair, candles)
            h1 = candles[-100:] if len(candles) >= 100 else candles
            
            tda_result = self.tda_analyzer.run_tda(weekly, daily, h4, h1)
            result.tda_alignment = tda_result.alignment.value
            result.tda_confidence = tda_result.confidence
            
            self.logger.debug(
                f"[TREND] TDA: {tda_result.alignment.value} "
                f"(B:{tda_result.bullish_count} N:{tda_result.neutral_count} S:{tda_result.bearish_count})"
            )
            
            # Check minimum alignment
            if tda_result.alignment == AlignmentQuality.WEAK:
                result.reason = "TDA alignment WEAK - no trade"
                return result
        
        # ===== STEP 2: Fibonacci Retracement (Multi-Level Support) =====
        fib_levels = None
        fib_level_used = None
        if self._fib_enabled:
            fib_levels = self.fib_calculator.calculate_levels(candles, self._fib_lookback)
            
            if fib_levels:
                # Determine if we're in trend or reversal regime based on TDA
                is_trend_regime = tda_result and tda_result.alignment == AlignmentQuality.PERFECT
                
                # Check multiple fib levels in trend regime, strict 88.6% in reversal
                fib_levels_to_check = self._fib_trend_levels if is_trend_regime else [self._fib_reversal_level]
                
                in_zone = False
                best_distance = 999
                for fib_level in fib_levels_to_check:
                    level_price = fib_levels.get_level_price(fib_level)
                    if level_price > 0:
                        pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
                        distance = abs(current_price - level_price) * pip_mult
                        if distance < self._fib_tolerance_pips and distance < best_distance:
                            in_zone = True
                            best_distance = distance
                            fib_level_used = fib_level
                
                result.at_fib_entry = in_zone
                result.fib_distance_pips = best_distance if in_zone else 999
                
                regime_str = "TREND" if is_trend_regime else "REVERSAL"
                self.logger.debug(
                    f"[TREND] Fib ({regime_str}): {'IN ZONE' if in_zone else 'OUT'} "
                    f"(levels={fib_levels_to_check}, best_dist={best_distance:.1f} pips)"
                )
                
                if not in_zone:
                    result.reason = f"Price not at Fib entry zone (dist={best_distance:.1f} pips)"
                    return result
        
        # ===== STEP 3: Divergence Detection =====
        divergence = None
        if self._divergence_enabled:
            divergence = self.divergence_detector.detect(candles)
            
            if divergence:
                result.has_divergence = True
                result.divergence_type = divergence.type.value
                self.logger.info(
                    f"[TREND] Divergence: {divergence.type.value} "
                    f"({divergence.strength.value}, conf={divergence.confidence:.2f})"
                )
        
        # ===== STEP 4: Pattern Recognition =====
        pattern = None
        if self._patterns_enabled:
            patterns = self.pattern_recognizer.detect_all_patterns(candles)
            pattern = self.pattern_recognizer.get_strongest_pattern(patterns)
            
            if pattern:
                result.has_pattern = True
                result.pattern_type = pattern.pattern_type.value
                self.logger.info(
                    f"[TREND] Pattern: {pattern.pattern_type.value} "
                    f"(conf={pattern.confidence:.2f}, confirmed={pattern.is_confirmed})"
                )
        
        # ===== NEW: OR-Gating Check =====
        # Require: TDA >= GOOD + Fib zone + (Divergence OR Pattern)
        if self._use_or_gating:
            has_confirmation = result.has_divergence or result.has_pattern
            if not has_confirmation:
                result.reason = "No confirmation (need Divergence OR Pattern)"
                return result
        else:
            # Old AND logic: require both if enabled
            if self._require_divergence and not result.has_divergence:
                result.reason = "No divergence confirmation"
                return result
            if self._require_pattern and not result.has_pattern:
                result.reason = "No pattern confirmation"
                return result
        
        # ===== STEP 5: Determine Trade Direction =====
        direction = self._determine_direction(tda_result, fib_levels, divergence, pattern)
        
        if direction is None:
            result.reason = "Conflicting signals - no clear direction"
            return result
        
        # Calculate entry parameters
        pip_multiplier = 100 if "JPY" in pair else 10000
        
        if fib_levels:
            entry_price = fib_levels.entry_price
            stop_loss = fib_levels.stop_loss
            r_value = fib_levels.r_value
        else:
            # Fallback to fixed SL
            r_value = self.config.stop_loss_pips / pip_multiplier
            if direction == "BUY":
                entry_price = market_data.ask
                stop_loss = entry_price - r_value
            else:
                entry_price = market_data.bid
                stop_loss = entry_price + r_value
        
        # Calculate confidence
        confidence = self._calculate_setup_confidence(
            tda_result, fib_levels, divergence, pattern, result.fib_distance_pips
        )
        
        result.can_trade = True
        result.direction = direction
        result.confidence = confidence
        result.entry_price = entry_price
        result.stop_loss = stop_loss
        result.r_value = r_value
        result.reason = self._generate_signal_reason(
            tda_result, fib_levels, divergence, pattern
        )
        
        # Store raw analysis objects for ML feature extraction
        result.tda_result = tda_result
        result.fib_levels = fib_levels
        result.divergence_result = divergence
        result.pattern_result = pattern
        
        return result
    
    def _determine_direction(
        self,
        tda_result,
        fib_levels: Optional[FibonacciLevels],
        divergence,
        pattern
    ) -> Optional[str]:
        """
        Determine trade direction from all analysis components.
        
        Returns "BUY", "SELL", or None if conflicting.
        """
        bullish_signals = 0
        bearish_signals = 0
        
        # TDA bias
        if tda_result:
            if tda_result.overall_bias == TrendBias.BULLISH:
                bullish_signals += 2  # Higher weight for TDA
            elif tda_result.overall_bias == TrendBias.BEARISH:
                bearish_signals += 2
        
        # Fibonacci direction
        if fib_levels:
            if fib_levels.direction == FibDirection.BULLISH:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Divergence
        if divergence:
            if divergence.type in [DivergenceType.BULLISH, DivergenceType.HIDDEN_BULLISH]:
                bullish_signals += 1
            elif divergence.type in [DivergenceType.BEARISH, DivergenceType.HIDDEN_BEARISH]:
                bearish_signals += 1
        
        # Pattern
        if pattern:
            if pattern.signal == PatternSignal.BULLISH:
                bullish_signals += 1
            elif pattern.signal == PatternSignal.BEARISH:
                bearish_signals += 1
        
        # Require clear majority
        if bullish_signals >= 3 and bearish_signals <= 1:
            return "BUY"
        elif bearish_signals >= 3 and bullish_signals <= 1:
            return "SELL"
        
        return None
    
    def _calculate_setup_confidence(
        self,
        tda_result,
        fib_levels: Optional[FibonacciLevels],
        divergence,
        pattern,
        fib_distance: float
    ) -> float:
        """Calculate overall setup confidence (0-1)."""
        confidence = 0.0
        
        # TDA contribution (max 40%)
        if tda_result:
            if tda_result.alignment == AlignmentQuality.PERFECT:
                confidence += 0.40
            elif tda_result.alignment == AlignmentQuality.GOOD:
                confidence += 0.30
        
        # Fibonacci proximity (max 25%)
        if fib_levels:
            if fib_distance <= 5:
                confidence += 0.25
            elif fib_distance <= 10:
                confidence += 0.20
            elif fib_distance <= 15:
                confidence += 0.15
            elif fib_distance <= 20:
                confidence += 0.10
        
        # Divergence (max 20%)
        if divergence:
            if divergence.strength == DivergenceStrength.CONFIRMED:
                confidence += 0.20
            elif divergence.strength == DivergenceStrength.MODERATE:
                confidence += 0.15
            else:
                confidence += 0.10
        
        # Pattern (max 15%)
        if pattern:
            if pattern.is_confirmed:
                confidence += 0.15
            else:
                confidence += 0.10
        
        return min(1.0, confidence)
    
    def _generate_signal_reason(
        self,
        tda_result,
        fib_levels: Optional[FibonacciLevels],
        divergence,
        pattern
    ) -> str:
        """Generate descriptive reason for the signal."""
        parts = []
        
        if tda_result:
            parts.append(f"TDA:{tda_result.alignment.value}")
        
        if fib_levels:
            parts.append(f"Fib:88.6%")
        
        if divergence:
            parts.append(f"Div:{divergence.type.value}")
        
        if pattern:
            parts.append(f"Pat:{pattern.pattern_type.value}")
        
        return " + ".join(parts) if parts else "SMA crossover"

    def detect_crossover(self, pair: str) -> Optional[str]:
        """
        Detect SMA crossover signal (fallback method).
        """
        self._initialize_pair(pair)
        
        sma_short = self._sma_short[pair]
        sma_long = self._sma_long[pair]
        
        if not sma_short.is_ready or not sma_long.is_ready:
            return None
        
        current_short = sma_short.value
        current_long = sma_long.value
        prev_short = self._previous_sma_short.get(pair, 0)
        prev_long = self._previous_sma_long.get(pair, 0)
        
        if prev_short == 0 or prev_long == 0:
            return None
        
        # Bullish crossover
        if prev_short < prev_long and current_short > current_long:
            self.logger.info(
                f"[TREND] BULLISH CROSSOVER detected on {pair}: "
                f"SMA{self.config.sma_period_short} ({current_short:.5f}) "
                f"crossed above SMA{self.config.sma_period_long} ({current_long:.5f})"
            )
            state = self._states[pair]
            state.last_crossover = "BULLISH"
            state.last_crossover_time = datetime.now()
            return "BULLISH"
        
        # Bearish crossover
        if prev_short > prev_long and current_short < current_long:
            self.logger.info(
                f"[TREND] BEARISH CROSSOVER detected on {pair}: "
                f"SMA{self.config.sma_period_short} ({current_short:.5f}) "
                f"crossed below SMA{self.config.sma_period_long} ({current_long:.5f})"
            )
            state = self._states[pair]
            state.last_crossover = "BEARISH"
            state.last_crossover_time = datetime.now()
            return "BEARISH"
        
        return None

    def calculate_signal_strength(self, pair: str) -> float:
        """Calculate overall signal strength (0 to 1)."""
        self._initialize_pair(pair)
        
        state = self._states[pair]
        strength = 0.0
        
        if state.sma_short > 0 and state.sma_long > 0:
            sma_diff_pct = abs(state.sma_short - state.sma_long) / state.sma_long * 100
            sma_strength = min(1.0, sma_diff_pct / 0.5)
            strength += sma_strength * 0.4
        
        if state.consecutive_higher_highs >= 3 or state.consecutive_lower_lows >= 3:
            strength += 0.3
        
        if state.last_crossover_time:
            time_since = (datetime.now() - state.last_crossover_time).total_seconds()
            candle_seconds = self._timeframe_to_seconds(self.config.timeframe)
            if time_since < candle_seconds * 5:
                recency_factor = 1 - (time_since / (candle_seconds * 5))
                strength += recency_factor * 0.3
        
        state.signal_strength = strength
        return strength

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        multipliers = {'m': 60, 'h': 3600, 'd': 86400}
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60)

    def generate_signals(
        self,
        pair: str,
        market_data: MarketData,
        account_state: AccountState
    ) -> List[Signal]:
        """
        Generate trading signals using enhanced analysis.
        
        Uses the 5-step confirmation process when enabled:
        1. TDA alignment check
        2. Fibonacci entry zone
        3. Divergence confirmation
        4. Pattern validation
        5. Execute with Fib-based levels
        
        Falls back to SMA crossover if enhanced analysis is disabled.
        """
        signals = []
        self._initialize_pair(pair)
        
        state = self._states[pair]
        
        # Try enhanced analysis first
        enhanced = self._run_enhanced_analysis(pair, market_data)
        
        if enhanced.can_trade:
            # FIX #4: Apply confidence gate
            if enhanced.confidence < self._min_confidence:
                self.logger.debug(
                    f"[TREND] Signal REJECTED: Confidence {enhanced.confidence:.1%} < {self._min_confidence:.1%} threshold"
                )
                # Don't generate signal - confidence too low
            else:
                # ===== ML FILTER GATE =====
                ml_passed = True
                ml_probability = 0.5
                ml_risk_multiplier = 1.0
                ml_adjusted_risk = self._ml_base_risk if hasattr(self, '_ml_base_risk') else 0.75
                
                if self._ml_enabled and self.ml_filter is not None:
                    # Calculate current ATR for features
                    candles = list(self._candle_history[pair])
                    current_atr = self._calculate_atr(candles) if len(candles) >= 14 else 0.001
                    
                    # Run ML evaluation
                    ml_passed, ml_adjusted_risk, ml_prediction = self.ml_filter.evaluate_signal(
                        tda_result=enhanced.tda_result,
                        fib_levels=enhanced.fib_levels,
                        divergence_result=enhanced.divergence_result,
                        pattern_result=enhanced.pattern_result,
                        trade_direction=enhanced.direction,
                        entry_price=enhanced.entry_price,
                        stop_loss=enhanced.stop_loss,
                        current_atr=current_atr,
                        current_spread=market_data.spread if hasattr(market_data, 'spread') else 0,
                        timestamp=market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.now(),
                        pair=pair,
                        tpu_confidence=enhanced.confidence
                    )
                    
                    ml_probability = ml_prediction.probability_win
                    ml_risk_multiplier = ml_prediction.risk_multiplier
                    
                    if not ml_passed:
                        self.logger.info(
                            f"[TREND] ML FILTER REJECTED: {pair} {enhanced.direction} - "
                            f"P(win)={ml_probability:.1%} < {self._ml_probability_threshold:.1%}"
                        )
                        # Store ML results in enhanced for logging
                        enhanced.ml_probability = ml_probability
                        enhanced.ml_risk_multiplier = ml_risk_multiplier
                        enhanced.ml_passed_filter = False
                        # Don't generate signal - ML says skip
                        return signals
                    
                    self.logger.info(
                        f"[TREND] ML FILTER PASSED: P(win)={ml_probability:.1%}, "
                        f"Risk={ml_adjusted_risk:.2f}%"
                    )
                
                # ===== NEWS FILTER (Backtest only) =====
                if self._news_filter_backtest and self.news_filter is not None:
                    timestamp = market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.now()
                    if self.news_filter.should_skip_trade(timestamp):
                        self.logger.info(f"[TREND] NEWS FILTER: Skipping trade near major news event")
                        return signals
                
                # Calculate position size (with ML-adjusted risk if applicable)
                lot_result = self.risk_manager.calculate_position_size(
                    pair=pair,
                    stop_loss_pips=abs(enhanced.entry_price - enhanced.stop_loss) * (100 if "JPY" in pair else 10000),
                    account_equity=account_state.equity,
                    free_margin=account_state.free_margin,
                    risk_percent=ml_adjusted_risk if self._ml_enabled else None
                )
                
                if lot_result.is_valid:
                    signal = self._create_enhanced_signal(
                        pair=pair,
                        direction=enhanced.direction,
                        entry_price=enhanced.entry_price,
                        stop_loss=enhanced.stop_loss,
                        r_value=enhanced.r_value,
                        lot_size=lot_result.lot_size,
                        confidence=enhanced.confidence,
                        reason=enhanced.reason,
                        ml_probability=ml_probability,
                        ml_risk_multiplier=ml_risk_multiplier
                    )
                    signals.append(signal)
                    
                    # Update state
                    state.tda_alignment = enhanced.tda_alignment
                    state.divergence_type = enhanced.divergence_type
                    state.pattern_detected = enhanced.pattern_type
                    state.fib_entry_zone = enhanced.at_fib_entry
                    
                    return signals
        
        # ===== TPU-Lite Mode =====
        # Lighter version: TDA + Fib only (no divergence/pattern required)
        # Has lower confidence requirement (0.55 minimum)
        if self._use_tpu_lite and not enhanced.can_trade:
            tpu_lite_signal = self._run_tpu_lite_analysis(pair, market_data, account_state)
            if tpu_lite_signal:
                signals.append(tpu_lite_signal)
                return signals
        
        # No more SMA fallback - TPU-lite replaces it
        return signals
        
        # Check for exit signals
        if pair in self._active_positions:
            exit_signal = self._check_exit_conditions(pair, market_data)
            if exit_signal:
                signals.append(exit_signal)
        
        return signals
    
    def _calculate_atr(self, candles: List, period: int = 14) -> float:
        """Calculate Average True Range from candles."""
        if len(candles) < period + 1:
            return 0.001
        
        true_ranges = []
        for i in range(1, min(period + 1, len(candles))):
            high = candles[-i].high
            low = candles[-i].low
            prev_close = candles[-(i+1)].close
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.001
    
    def _run_tpu_lite_analysis(
        self,
        pair: str,
        market_data: MarketData,
        account_state: AccountState
    ) -> Optional[Signal]:
        """
        TPU-Lite Mode: Lighter requirements than full TPU.
        
        Requirements:
        - TDA >= GOOD (still required for direction)
        - In Fib zone (0.5-0.886, wider range)
        - NO divergence/pattern required
        - Lower confidence threshold (0.55)
        
        This replaces SMA fallback with smarter logic.
        """
        current_price = market_data.mid_price
        candles = list(self._candle_history[pair])
        
        if len(candles) < self._fib_lookback + 5:
            return None
        
        # Step 1: TDA check (still required)
        tda_result = None
        if self._tda_enabled:
            weekly = self._candles_weekly.get(pair, candles)
            daily = self._candles_daily.get(pair, candles)
            h4 = self._candles_h4.get(pair, candles)
            h1 = candles[-100:] if len(candles) >= 100 else candles
            
            tda_result = self.tda_analyzer.run_tda(weekly, daily, h4, h1)
            
            if tda_result.alignment == AlignmentQuality.WEAK:
                return None  # TDA still required even in lite mode
        
        # Step 2: Fib check (wider tolerance for lite mode)
        fib_levels = self.fib_calculator.calculate_levels(candles, self._fib_lookback)
        if not fib_levels:
            return None
        
        # TPU-lite uses wider fib levels: 0.5, 0.618, 0.786, 0.886
        tpu_lite_fib_levels = [0.5, 0.618, 0.786, 0.886]
        tpu_lite_tolerance = self._fib_tolerance_pips * 1.5  # 50% wider tolerance
        
        in_zone = False
        best_distance = 999
        fib_level_used = None
        for fib_level in tpu_lite_fib_levels:
            level_price = fib_levels.get_level_price(fib_level)
            if level_price > 0:
                pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
                distance = abs(current_price - level_price) * pip_mult
                if distance < tpu_lite_tolerance and distance < best_distance:
                    in_zone = True
                    best_distance = distance
                    fib_level_used = fib_level
        
        if not in_zone:
            return None
        
        # Determine direction from TDA
        if tda_result and tda_result.trade_direction:
            tda_direction = tda_result.trade_direction.value.upper()
        else:
            tda_direction = None
        
        # Fib direction
        fib_direction = "BUY" if fib_levels.direction == FibDirection.BULLISH else "SELL"
        
        # CRITICAL: Ensure TDA and Fib directions align
        # If they conflict, skip the trade (ambiguous setup)
        if tda_direction and tda_direction != fib_direction:
            self.logger.debug(
                f"[TREND] TPU-LITE: Direction conflict TDA={tda_direction} vs Fib={fib_direction} - skipping"
            )
            return None
        
        direction = fib_direction  # Use Fib direction (guaranteed to match SL)
        
        # Calculate entry parameters
        entry_price = current_price
        stop_loss = fib_levels.stop_loss
        r_value = fib_levels.r_value
        
        # VALIDATION: Ensure SL is on correct side
        if direction == "BUY" and stop_loss >= entry_price:
            self.logger.debug(f"[TREND] TPU-LITE: Invalid SL {stop_loss} >= entry {entry_price} for BUY - skipping")
            return None
        if direction == "SELL" and stop_loss <= entry_price:
            self.logger.debug(f"[TREND] TPU-LITE: Invalid SL {stop_loss} <= entry {entry_price} for SELL - skipping")
            return None
        
        # TPU-Lite confidence: 55-70% range
        confidence = 0.55 + (1.0 - best_distance / tpu_lite_tolerance) * 0.15
        confidence = min(0.70, max(0.55, confidence))
        
        # ML filter (3-zone)
        ml_passed = True
        ml_probability = 0.55  # Default for lite mode
        ml_risk_multiplier = 0.5  # Lite mode uses reduced risk
        ml_adjusted_risk = self._ml_base_risk * 0.5 if hasattr(self, '_ml_base_risk') else 0.375
        
        if self._ml_enabled and self.ml_filter is not None:
            current_atr = self._calculate_atr(candles) if len(candles) >= 14 else 0.001
            
            ml_passed, ml_adjusted_risk, ml_prediction = self.ml_filter.evaluate_signal(
                tda_result=tda_result,
                fib_levels=fib_levels,
                divergence_result=None,
                pattern_result=None,
                trade_direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                current_atr=current_atr,
                current_spread=market_data.spread if hasattr(market_data, 'spread') else 0,
                timestamp=market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.now(),
                pair=pair,
                tpu_confidence=confidence
            )
            
            ml_probability = ml_prediction.probability_win
            ml_risk_multiplier = ml_prediction.risk_multiplier
            
            if not ml_passed:
                self.logger.debug(
                    f"[TREND] TPU-LITE ML REJECTED: {pair} - P(win)={ml_probability:.1%}"
                )
                return None
        
        # Calculate position size (at reduced risk for lite mode)
        pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
        lot_result = self.risk_manager.calculate_position_size(
            pair=pair,
            stop_loss_pips=abs(entry_price - stop_loss) * pip_mult,
            account_equity=account_state.equity,
            free_margin=account_state.free_margin,
            risk_percent=ml_adjusted_risk
        )
        
        if not lot_result.is_valid:
            return None
        
        reason = f"TPU-LITE: TDA {tda_result.alignment.value if tda_result else 'N/A'}, Fib {fib_level_used:.1%} zone"
        
        self.logger.info(
            f"[TREND] TPU-LITE {direction}: {pair} @ {entry_price:.5f}"
        )
        self.logger.info(
            f"  Fib {fib_level_used:.1%}, dist={best_distance:.1f} pips, conf={confidence:.1%}"
        )
        
        return self._create_enhanced_signal(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            r_value=r_value,
            lot_size=lot_result.lot_size,
            confidence=confidence,
            reason=reason,
            ml_probability=ml_probability,
            ml_risk_multiplier=ml_risk_multiplier
        )

    def _create_enhanced_signal(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        r_value: float,
        lot_size: float,
        confidence: float,
        reason: str,
        ml_probability: float = 0.5,
        ml_risk_multiplier: float = 1.0
    ) -> Signal:
        """Create signal from enhanced analysis."""
        side = SignalType.BUY if direction == "BUY" else SignalType.SELL
        
        # Take profit: None if partial exits enabled
        if self._use_partial_exits:
            take_profit = None
        else:
            if direction == "BUY":
                take_profit = entry_price + (r_value * 2)  # 2R target
            else:
                take_profit = entry_price - (r_value * 2)
        
        self.logger.info(
            f"[TREND] ENHANCED {direction} SIGNAL: {pair} @ {entry_price:.5f}"
        )
        self.logger.info(
            f"  SL: {stop_loss:.5f}, "
            f"TP: {'MANAGED' if take_profit is None else f'{take_profit:.5f}'}"
        )
        self.logger.info(
            f"  R-value: {r_value:.5f}, Confidence: {confidence:.1%}"
        )
        if self._ml_enabled:
            self.logger.info(
                f"  ML: P(win)={ml_probability:.1%}, Risk Mult={ml_risk_multiplier:.2f}x"
            )
        self.logger.info(f"  Reason: {reason}")
        
        return Signal(
            signal_type=side,
            pair=pair,
            price=entry_price,
            timestamp=datetime.now(),
            module=TradingModule.TREND,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            metadata={
                "lot_size": lot_size,
                "r_value": r_value,
                "enhanced_analysis": True,
                "sma_short": self._states[pair].sma_short,
                "sma_long": self._states[pair].sma_long,
                "ml_probability": ml_probability,
                "ml_risk_multiplier": ml_risk_multiplier,
            }
        )

    def _create_entry_signal(
        self,
        pair: str,
        side: SignalType,
        price: float,
        lot_size: float,
        strength: float,
        reason: str,
        use_partial_exits: bool = False
    ) -> Signal:
        """Create entry signal (fallback SMA method)."""
        pip_multiplier = 100 if "JPY" in pair else 10000
        
        if side == SignalType.BUY:
            stop_loss = price - (self.config.stop_loss_pips / pip_multiplier)
        else:
            stop_loss = price + (self.config.stop_loss_pips / pip_multiplier)
        
        r_value = self.config.stop_loss_pips / pip_multiplier
        
        if use_partial_exits:
            take_profit = None
            self.logger.info(
                f"[TREND] {side.value} SIGNAL: {pair} @ {price:.5f}, "
                f"SL: {stop_loss:.5f}, TP: MANAGED (partial exits), "
                f"R-value: {r_value:.5f}, Strength: {strength:.1%}"
            )
        else:
            if side == SignalType.BUY:
                take_profit = price + (self.config.take_profit_pips / pip_multiplier)
            else:
                take_profit = price - (self.config.take_profit_pips / pip_multiplier)
            self.logger.info(
                f"[TREND] {side.value} SIGNAL: {pair} @ {price:.5f}, "
                f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}, "
                f"Strength: {strength:.1%}"
            )
        
        return Signal(
            signal_type=side,
            pair=pair,
            price=price,
            timestamp=datetime.now(),
            module=TradingModule.TREND,
            confidence=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            metadata={
                "lot_size": lot_size,
                "r_value": r_value,
                "sma_short": self._states[pair].sma_short,
                "sma_long": self._states[pair].sma_long,
            }
        )

    def _check_exit_conditions(
        self,
        pair: str,
        market_data: MarketData
    ) -> Optional[Signal]:
        """Check if existing position should be closed."""
        position = self._active_positions.get(pair)
        if not position:
            return None
        
        crossover = self.detect_crossover(pair)
        
        should_exit = False
        reason = ""
        
        if position.side == OrderSide.BUY:
            if crossover == "BEARISH":
                should_exit = True
                reason = "Bearish crossover exit"
        else:
            if crossover == "BULLISH":
                should_exit = True
                reason = "Bullish crossover exit"
        
        if should_exit:
            current_price = market_data.mid_price
            self.logger.info(
                f"[TREND] EXIT SIGNAL: {pair} @ {current_price:.5f} - {reason}"
            )
            return Signal(
                signal_type=SignalType.CLOSE,
                pair=pair,
                price=current_price,
                timestamp=datetime.now(),
                module=TradingModule.TREND,
                confidence=1.0,
                reason=reason,
                metadata={"order_id": position.order_id}
            )
        
        return None

    def register_position(self, order: Order) -> None:
        """Register an opened position."""
        self._active_positions[order.pair] = order
        self.logger.info(
            f"[TREND] Position registered: {order.side.value} {order.pair} #{order.order_id}"
        )

    def close_position(self, pair: str) -> None:
        """Remove a closed position from tracking."""
        if pair in self._active_positions:
            del self._active_positions[pair]
            self.logger.info(f"[TREND] Position closed for {pair}")

    def get_state(self, pair: str) -> TrendState:
        """Get current trend state for a pair."""
        self._initialize_pair(pair)
        return self._states[pair]

    def get_sma_values(self, pair: str) -> Tuple[Optional[float], Optional[float]]:
        """Get current SMA values for a pair."""
        self._initialize_pair(pair)
        return (
            self._sma_short[pair].value,
            self._sma_long[pair].value
        )

    def is_ready(self, pair: str) -> bool:
        """Check if enough data to generate signals."""
        self._initialize_pair(pair)
        return (
            self._sma_short[pair].is_ready and 
            self._sma_long[pair].is_ready and
            len(self._candle_history[pair]) >= 4
        )

    def clear_pair(self, pair: str) -> None:
        """Clear all data for a pair."""
        if pair in self._sma_short:
            self._sma_short[pair].clear()
        if pair in self._sma_long:
            self._sma_long[pair].clear()
        if pair in self._candle_history:
            self._candle_history[pair].clear()
        if pair in self._states:
            self._states[pair] = TrendState()
        if pair in self._active_positions:
            del self._active_positions[pair]
