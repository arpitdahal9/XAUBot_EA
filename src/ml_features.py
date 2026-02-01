"""
ML Feature Extraction from TPU Signals

Extracts structured features from existing TPU components:
- Top-Down Analysis bias scores
- Fibonacci entry quality
- Divergence strength
- Pattern confidence
- Volatility context

NO new indicators added - uses only existing TPU outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import logging


@dataclass
class MLFeatures:
    """
    Feature vector for ML trade quality prediction.
    
    All features come from existing TPU modules.
    """
    # Timestamp and identifiers
    timestamp: datetime = None
    pair: str = ""
    direction: str = ""  # "BUY" or "SELL"
    
    # === TOP-DOWN ANALYSIS FEATURES ===
    tda_alignment_score: float = 0.0  # 0=WEAK, 0.5=GOOD, 1.0=PERFECT
    weekly_bias: float = 0.0   # -1=BEARISH, 0=NEUTRAL, 1=BULLISH
    daily_bias: float = 0.0
    h4_bias: float = 0.0
    h1_bias: float = 0.0
    bias_agreement_count: int = 0  # How many TFs agree with trade direction
    
    # === FIBONACCI FEATURES ===
    fib_level_distance: float = 0.0  # Distance from 88.6% in pips
    fib_level_distance_normalized: float = 0.0  # Distance / tolerance
    swing_size_atr_ratio: float = 0.0  # Swing size relative to ATR
    fib_direction_match: int = 0  # 1 if fib direction matches trade, 0 otherwise
    
    # === DIVERGENCE FEATURES ===
    has_divergence: int = 0  # 1 if divergence present
    divergence_type: int = 0  # 0=none, 1=bullish, -1=bearish
    divergence_strength: float = 0.0  # 0-1 scale
    divergence_matches_trade: int = 0  # 1 if divergence supports trade direction
    
    # === PATTERN FEATURES ===
    has_pattern: int = 0
    pattern_type: int = 0  # Encoded pattern type
    pattern_confidence: float = 0.0
    pattern_confirmed: int = 0  # 1 if neckline broken
    pattern_matches_trade: int = 0  # 1 if pattern supports trade direction
    
    # === VOLATILITY/CONTEXT FEATURES ===
    atr_value: float = 0.0
    atr_percentile: float = 0.0  # Percentile of current ATR vs history
    spread_pips: float = 0.0
    spread_percentile: float = 0.0  # Percentile of current spread
    
    # === TRADE GEOMETRY ===
    stop_distance_pips: float = 0.0
    stop_distance_atr_ratio: float = 0.0  # SL distance / ATR
    expected_r_to_tp1: float = 2.0  # Usually 2R
    entry_price: float = 0.0
    stop_loss: float = 0.0
    
    # === TIME FEATURES ===
    hour_of_day: int = 0
    day_of_week: int = 0
    is_london_session: int = 0
    is_ny_session: int = 0
    is_overlap_session: int = 0
    
    # === COMPOSITE SCORES ===
    tpu_confidence: float = 0.0  # Overall TPU confidence from TrendFollower
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML model input."""
        return {
            # TDA features
            'tda_alignment_score': self.tda_alignment_score,
            'weekly_bias': self.weekly_bias,
            'daily_bias': self.daily_bias,
            'h4_bias': self.h4_bias,
            'h1_bias': self.h1_bias,
            'bias_agreement_count': self.bias_agreement_count,
            
            # Fibonacci features
            'fib_level_distance': self.fib_level_distance,
            'fib_level_distance_normalized': self.fib_level_distance_normalized,
            'swing_size_atr_ratio': self.swing_size_atr_ratio,
            'fib_direction_match': self.fib_direction_match,
            
            # Divergence features
            'has_divergence': self.has_divergence,
            'divergence_type': self.divergence_type,
            'divergence_strength': self.divergence_strength,
            'divergence_matches_trade': self.divergence_matches_trade,
            
            # Pattern features
            'has_pattern': self.has_pattern,
            'pattern_type': self.pattern_type,
            'pattern_confidence': self.pattern_confidence,
            'pattern_confirmed': self.pattern_confirmed,
            'pattern_matches_trade': self.pattern_matches_trade,
            
            # Volatility features
            'atr_percentile': self.atr_percentile,
            'spread_percentile': self.spread_percentile,
            
            # Trade geometry
            'stop_distance_atr_ratio': self.stop_distance_atr_ratio,
            
            # Time features
            'hour_of_day': self.hour_of_day,
            'day_of_week': self.day_of_week,
            'is_london_session': self.is_london_session,
            'is_ny_session': self.is_ny_session,
            'is_overlap_session': self.is_overlap_session,
            
            # Composite
            'tpu_confidence': self.tpu_confidence,
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array(list(self.to_dict().values()), dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return [
            'tda_alignment_score', 'weekly_bias', 'daily_bias', 'h4_bias', 'h1_bias',
            'bias_agreement_count', 'fib_level_distance', 'fib_level_distance_normalized',
            'swing_size_atr_ratio', 'fib_direction_match', 'has_divergence',
            'divergence_type', 'divergence_strength', 'divergence_matches_trade',
            'has_pattern', 'pattern_type', 'pattern_confidence', 'pattern_confirmed',
            'pattern_matches_trade', 'atr_percentile', 'spread_percentile',
            'stop_distance_atr_ratio', 'hour_of_day', 'day_of_week',
            'is_london_session', 'is_ny_session', 'is_overlap_session', 'tpu_confidence'
        ]


class MLFeatureExtractor:
    """
    Extract ML features from TPU signal components.
    
    Uses ONLY existing TPU module outputs - no new indicators.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical data for percentile calculations
        self._atr_history: List[float] = []
        self._spread_history: List[float] = []
        self._max_history = 500  # Rolling window size
    
    def extract_features(
        self,
        # TPU Analysis Results
        tda_result: Any,
        fib_levels: Any,
        divergence_result: Any,
        pattern_result: Any,
        # Trade Parameters
        trade_direction: str,  # "BUY" or "SELL"
        entry_price: float,
        stop_loss: float,
        # Market Context
        current_atr: float,
        current_spread: float,
        timestamp: datetime,
        pair: str,
        # TPU confidence
        tpu_confidence: float
    ) -> MLFeatures:
        """
        Extract all features from TPU signal components.
        
        Args:
            tda_result: Result from TopDownAnalyzer.run_tda()
            fib_levels: Result from FibonacciCalculator.calculate_levels()
            divergence_result: Result from DivergenceDetector.detect()
            pattern_result: Result from PatternRecognizer.detect_all_patterns()
            trade_direction: "BUY" or "SELL"
            entry_price: Trade entry price
            stop_loss: Stop loss price
            current_atr: Current ATR value
            current_spread: Current spread in pips
            timestamp: Signal timestamp
            pair: Currency pair
            tpu_confidence: Overall TPU confidence score
            
        Returns:
            MLFeatures with all extracted features
        """
        features = MLFeatures(
            timestamp=timestamp,
            pair=pair,
            direction=trade_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tpu_confidence=tpu_confidence
        )
        
        # Extract TDA features
        self._extract_tda_features(features, tda_result, trade_direction)
        
        # Extract Fibonacci features
        self._extract_fib_features(features, fib_levels, trade_direction, current_atr, entry_price)
        
        # Extract Divergence features
        self._extract_divergence_features(features, divergence_result, trade_direction)
        
        # Extract Pattern features
        self._extract_pattern_features(features, pattern_result, trade_direction)
        
        # Extract Volatility/Context features
        self._extract_volatility_features(features, current_atr, current_spread)
        
        # Extract Trade Geometry features
        self._extract_geometry_features(features, entry_price, stop_loss, current_atr, pair)
        
        # Extract Time features
        self._extract_time_features(features, timestamp)
        
        return features
    
    def _extract_tda_features(
        self,
        features: MLFeatures,
        tda_result: Any,
        trade_direction: str
    ) -> None:
        """Extract Top-Down Analysis features."""
        if tda_result is None:
            return
        
        # Alignment score: WEAK=0, GOOD=0.5, PERFECT=1.0
        alignment_map = {"WEAK": 0.0, "GOOD": 0.5, "PERFECT": 1.0}
        features.tda_alignment_score = alignment_map.get(
            getattr(tda_result, 'alignment', type('', (), {'value': 'WEAK'})()).value, 
            0.0
        )
        
        # Bias encoding: BULLISH=1, NEUTRAL=0, BEARISH=-1
        def bias_to_num(bias):
            if bias is None:
                return 0
            bias_val = getattr(bias, 'bias', None)
            if bias_val is None:
                return 0
            bias_str = getattr(bias_val, 'value', str(bias_val))
            return 1 if 'BULL' in bias_str.upper() else (-1 if 'BEAR' in bias_str.upper() else 0)
        
        features.weekly_bias = bias_to_num(getattr(tda_result, 'weekly_bias', None))
        features.daily_bias = bias_to_num(getattr(tda_result, 'daily_bias', None))
        features.h4_bias = bias_to_num(getattr(tda_result, 'h4_bias', None))
        features.h1_bias = bias_to_num(getattr(tda_result, 'h1_bias', None))
        
        # Count how many biases agree with trade direction
        trade_bias = 1 if trade_direction == "BUY" else -1
        biases = [features.weekly_bias, features.daily_bias, features.h4_bias, features.h1_bias]
        features.bias_agreement_count = sum(1 for b in biases if b == trade_bias)
    
    def _extract_fib_features(
        self,
        features: MLFeatures,
        fib_levels: Any,
        trade_direction: str,
        current_atr: float,
        entry_price: float
    ) -> None:
        """Extract Fibonacci features."""
        if fib_levels is None:
            return
        
        # Distance from 88.6% level
        fib_886 = getattr(fib_levels, 'level_886', 0)
        if fib_886 > 0:
            features.fib_level_distance = abs(entry_price - fib_886) * 10000  # In pips
            features.fib_level_distance_normalized = features.fib_level_distance / 20  # Normalized by tolerance
        
        # Swing size relative to ATR
        swing_range = getattr(fib_levels, 'range_size', 0)
        if current_atr > 0 and swing_range > 0:
            features.swing_size_atr_ratio = swing_range / current_atr
        
        # Does fib direction match trade?
        fib_dir = getattr(fib_levels, 'direction', None)
        if fib_dir:
            fib_dir_str = getattr(fib_dir, 'value', str(fib_dir)).upper()
            if trade_direction == "BUY" and "BULL" in fib_dir_str:
                features.fib_direction_match = 1
            elif trade_direction == "SELL" and "BEAR" in fib_dir_str:
                features.fib_direction_match = 1
    
    def _extract_divergence_features(
        self,
        features: MLFeatures,
        divergence_result: Any,
        trade_direction: str
    ) -> None:
        """Extract Divergence features."""
        if divergence_result is None:
            return
        
        features.has_divergence = 1
        
        # Divergence type encoding
        div_type = getattr(divergence_result, 'type', None)
        if div_type:
            div_str = getattr(div_type, 'value', str(div_type)).upper()
            if 'BULL' in div_str:
                features.divergence_type = 1
            elif 'BEAR' in div_str:
                features.divergence_type = -1
        
        # Divergence strength
        features.divergence_strength = getattr(divergence_result, 'confidence', 0.5)
        
        # Does divergence support trade direction?
        if trade_direction == "BUY" and features.divergence_type == 1:
            features.divergence_matches_trade = 1
        elif trade_direction == "SELL" and features.divergence_type == -1:
            features.divergence_matches_trade = 1
    
    def _extract_pattern_features(
        self,
        features: MLFeatures,
        pattern_result: Any,
        trade_direction: str
    ) -> None:
        """Extract Pattern features."""
        if pattern_result is None:
            return
        
        features.has_pattern = 1
        
        # Pattern type encoding
        pattern_type = getattr(pattern_result, 'pattern_type', None)
        if pattern_type:
            pat_str = getattr(pattern_type, 'value', str(pattern_type)).upper()
            # Encode pattern types
            pattern_encoding = {
                'DOUBLE_BOTTOM': 1,
                'DOUBLE_TOP': 2,
                'HEAD_AND_SHOULDERS': 3,
                'INVERSE_HEAD_AND_SHOULDERS': 4,
            }
            for key, val in pattern_encoding.items():
                if key in pat_str:
                    features.pattern_type = val
                    break
        
        features.pattern_confidence = getattr(pattern_result, 'confidence', 0.5)
        features.pattern_confirmed = 1 if getattr(pattern_result, 'is_confirmed', False) else 0
        
        # Does pattern support trade direction?
        signal = getattr(pattern_result, 'signal', None)
        if signal:
            sig_str = getattr(signal, 'value', str(signal)).upper()
            if trade_direction == "BUY" and "BULL" in sig_str:
                features.pattern_matches_trade = 1
            elif trade_direction == "SELL" and "BEAR" in sig_str:
                features.pattern_matches_trade = 1
    
    def _extract_volatility_features(
        self,
        features: MLFeatures,
        current_atr: float,
        current_spread: float
    ) -> None:
        """Extract volatility and spread context features."""
        features.atr_value = current_atr
        features.spread_pips = current_spread
        
        # Update history
        self._atr_history.append(current_atr)
        self._spread_history.append(current_spread)
        
        # Keep history bounded
        if len(self._atr_history) > self._max_history:
            self._atr_history = self._atr_history[-self._max_history:]
        if len(self._spread_history) > self._max_history:
            self._spread_history = self._spread_history[-self._max_history:]
        
        # Calculate percentiles
        if len(self._atr_history) >= 20:
            features.atr_percentile = np.percentile(
                self._atr_history, 
                (np.array(self._atr_history) <= current_atr).mean() * 100
            ) / 100
        
        if len(self._spread_history) >= 20:
            features.spread_percentile = np.percentile(
                self._spread_history,
                (np.array(self._spread_history) <= current_spread).mean() * 100
            ) / 100
    
    def _extract_geometry_features(
        self,
        features: MLFeatures,
        entry_price: float,
        stop_loss: float,
        current_atr: float,
        pair: str
    ) -> None:
        """Extract trade geometry features."""
        pip_multiplier = 100 if "JPY" in pair else 10000
        
        features.stop_distance_pips = abs(entry_price - stop_loss) * pip_multiplier
        
        if current_atr > 0:
            features.stop_distance_atr_ratio = abs(entry_price - stop_loss) / current_atr
        
        features.expected_r_to_tp1 = 2.0  # Model A TP1 is always at +2R
    
    def _extract_time_features(
        self,
        features: MLFeatures,
        timestamp: datetime
    ) -> None:
        """Extract time-based features."""
        features.hour_of_day = timestamp.hour
        features.day_of_week = timestamp.weekday()
        
        # Trading sessions (UTC times)
        hour = timestamp.hour
        
        # London: 8:00-16:00 UTC
        features.is_london_session = 1 if 8 <= hour < 16 else 0
        
        # New York: 13:00-21:00 UTC
        features.is_ny_session = 1 if 13 <= hour < 21 else 0
        
        # Overlap: 13:00-16:00 UTC (highest liquidity)
        features.is_overlap_session = 1 if 13 <= hour < 16 else 0
    
    def reset_history(self) -> None:
        """Reset historical data for new backtest."""
        self._atr_history = []
        self._spread_history = []
