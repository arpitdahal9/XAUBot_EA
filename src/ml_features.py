"""
ML Feature Extraction from TPU Signals
FIX: Percentile calculation now uses rank percentile (count <= value / total)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import logging


@dataclass
class MLFeatures:
    timestamp: datetime = None
    pair: str = ""
    direction: str = ""
    tda_alignment_score: float = 0.0
    weekly_bias: float = 0.0
    daily_bias: float = 0.0
    h4_bias: float = 0.0
    h1_bias: float = 0.0
    bias_agreement_count: int = 0
    tda_score: float = 0.0
    fib_level_distance: float = 0.0
    fib_level_distance_normalized: float = 0.0
    swing_size_atr_ratio: float = 0.0
    fib_direction_match: int = 0
    fib_score: float = 0.0
    has_divergence: int = 0
    divergence_type: int = 0
    divergence_strength: float = 0.0
    divergence_matches_trade: int = 0
    divergence_score: float = 0.0
    has_pattern: int = 0
    pattern_type: int = 0
    pattern_confidence: float = 0.0
    pattern_confirmed: int = 0
    pattern_matches_trade: int = 0
    pattern_score: float = 0.0
    atr_value: float = 0.0
    atr_percentile: float = 0.0
    spread_pips: float = 0.0
    spread_percentile: float = 0.0
    volatility_regime: int = 0
    spread_regime: int = 0
    distance_to_swing_high: float = 0.0
    distance_to_swing_low: float = 0.0
    distance_to_structure: float = 0.0
    stop_distance_pips: float = 0.0
    stop_distance_atr_ratio: float = 0.0
    expected_r_to_tp1: float = 2.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0
    is_london_session: int = 0
    is_ny_session: int = 0
    is_overlap_session: int = 0
    time_since_last_signal: float = 0.0
    tpu_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tda_alignment_score': self.tda_alignment_score,
            'weekly_bias': self.weekly_bias,
            'daily_bias': self.daily_bias,
            'h4_bias': self.h4_bias,
            'h1_bias': self.h1_bias,
            'bias_agreement_count': self.bias_agreement_count,
            'tda_score': self.tda_score,
            'fib_level_distance': self.fib_level_distance,
            'fib_level_distance_normalized': self.fib_level_distance_normalized,
            'swing_size_atr_ratio': self.swing_size_atr_ratio,
            'fib_direction_match': self.fib_direction_match,
            'fib_score': self.fib_score,
            'has_divergence': self.has_divergence,
            'divergence_type': self.divergence_type,
            'divergence_strength': self.divergence_strength,
            'divergence_matches_trade': self.divergence_matches_trade,
            'divergence_score': self.divergence_score,
            'has_pattern': self.has_pattern,
            'pattern_type': self.pattern_type,
            'pattern_confidence': self.pattern_confidence,
            'pattern_confirmed': self.pattern_confirmed,
            'pattern_matches_trade': self.pattern_matches_trade,
            'pattern_score': self.pattern_score,
            'atr_percentile': self.atr_percentile,
            'spread_percentile': self.spread_percentile,
            'volatility_regime': self.volatility_regime,
            'spread_regime': self.spread_regime,
            'distance_to_structure': self.distance_to_structure,
            'stop_distance_atr_ratio': self.stop_distance_atr_ratio,
            'hour_of_day': self.hour_of_day,
            'day_of_week': self.day_of_week,
            'is_london_session': self.is_london_session,
            'is_ny_session': self.is_ny_session,
            'is_overlap_session': self.is_overlap_session,
            'time_since_last_signal': self.time_since_last_signal,
            'tpu_confidence': self.tpu_confidence,
        }

    def to_array(self) -> np.ndarray:
        return np.array(list(self.to_dict().values()), dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return list(MLFeatures().to_dict().keys())


class MLFeatureExtractor:
    def __init__(self, max_history: int = 500):
        self.logger = logging.getLogger(__name__)
        self._atr_history: List[float] = []
        self._spread_history: List[float] = []
        self._max_history = max_history
        self._last_signal_time: Dict[str, datetime] = {}
        self._last_swing_high: Dict[str, float] = {}
        self._last_swing_low: Dict[str, float] = {}

    def extract_features(self, tda_result: Any, fib_levels: Any, divergence_result: Any,
                         pattern_result: Any, trade_direction: str, entry_price: float,
                         stop_loss: float, current_atr: float, current_spread: float,
                         timestamp: datetime, pair: str, tpu_confidence: float) -> MLFeatures:
        features = MLFeatures(timestamp=timestamp, pair=pair, direction=trade_direction,
                              entry_price=entry_price, stop_loss=stop_loss, tpu_confidence=tpu_confidence)
        self._extract_tda_features(features, tda_result, trade_direction)
        self._extract_fib_features(features, fib_levels, trade_direction, current_atr, entry_price, pair)
        self._extract_divergence_features(features, divergence_result, trade_direction)
        self._extract_pattern_features(features, pattern_result, trade_direction)
        self._extract_volatility_features(features, current_atr, current_spread)
        self._extract_structure_features(features, entry_price, pair)
        self._extract_geometry_features(features, entry_price, stop_loss, current_atr, pair)
        self._extract_time_features(features, timestamp, pair)
        return features

    def _extract_tda_features(self, features, tda_result, trade_direction):
        if tda_result is None:
            return
        alignment_map = {"WEAK": 0.0, "GOOD": 0.5, "PERFECT": 1.0}
        alignment_val = getattr(tda_result, 'alignment', None)
        if alignment_val:
            alignment_str = getattr(alignment_val, 'value', str(alignment_val))
            features.tda_alignment_score = alignment_map.get(alignment_str, 0.0)
        features.tda_score = features.tda_alignment_score
        def bias_to_num(bias_obj):
            if bias_obj is None:
                return 0
            bias_val = getattr(bias_obj, 'bias', bias_obj)
            bias_str = getattr(bias_val, 'value', str(bias_val))
            return 1 if 'BULL' in bias_str.upper() else (-1 if 'BEAR' in bias_str.upper() else 0)
        features.weekly_bias = bias_to_num(getattr(tda_result, 'weekly_bias', None))
        features.daily_bias = bias_to_num(getattr(tda_result, 'daily_bias', None))
        features.h4_bias = bias_to_num(getattr(tda_result, 'h4_bias', None))
        features.h1_bias = bias_to_num(getattr(tda_result, 'h1_bias', None))
        trade_bias = 1 if trade_direction == "BUY" else -1
        biases = [features.weekly_bias, features.daily_bias, features.h4_bias, features.h1_bias]
        features.bias_agreement_count = sum(1 for b in biases if b == trade_bias)

    def _extract_fib_features(self, features, fib_levels, trade_direction, current_atr, entry_price, pair):
        if fib_levels is None:
            return
        pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
        fib_886 = getattr(fib_levels, 'level_886', 0)
        if fib_886 > 0:
            features.fib_level_distance = abs(entry_price - fib_886) * pip_mult
            features.fib_level_distance_normalized = min(features.fib_level_distance / 20, 2.0)
        swing_range = getattr(fib_levels, 'range_size', 0)
        if current_atr > 0 and swing_range > 0:
            features.swing_size_atr_ratio = swing_range / current_atr
        swing_high = getattr(fib_levels, 'swing_high', 0)
        swing_low = getattr(fib_levels, 'swing_low', 0)
        if swing_high > 0:
            self._last_swing_high[pair] = swing_high
        if swing_low > 0:
            self._last_swing_low[pair] = swing_low
        fib_dir = getattr(fib_levels, 'direction', None)
        if fib_dir:
            fib_dir_str = getattr(fib_dir, 'value', str(fib_dir)).upper()
            if (trade_direction == "BUY" and "BULL" in fib_dir_str) or (trade_direction == "SELL" and "BEAR" in fib_dir_str):
                features.fib_direction_match = 1
        fib_proximity_score = max(0, 1 - features.fib_level_distance_normalized)
        features.fib_score = fib_proximity_score * (0.8 + 0.2 * features.fib_direction_match)

    def _extract_divergence_features(self, features, divergence_result, trade_direction):
        if divergence_result is None:
            return
        features.has_divergence = 1
        div_type = getattr(divergence_result, 'type', None)
        if div_type:
            div_str = getattr(div_type, 'value', str(div_type)).upper()
            features.divergence_type = 1 if 'BULL' in div_str else (-1 if 'BEAR' in div_str else 0)
        features.divergence_strength = getattr(divergence_result, 'confidence', 0.5)
        if (trade_direction == "BUY" and features.divergence_type == 1) or (trade_direction == "SELL" and features.divergence_type == -1):
            features.divergence_matches_trade = 1
        features.divergence_score = features.divergence_strength * (0.7 + 0.3 * features.divergence_matches_trade)

    def _extract_pattern_features(self, features, pattern_result, trade_direction):
        if pattern_result is None:
            return
        features.has_pattern = 1
        pattern_type = getattr(pattern_result, 'pattern_type', None)
        if pattern_type:
            pat_str = getattr(pattern_type, 'value', str(pattern_type)).upper()
            encoding = {'DOUBLE_BOTTOM': 1, 'DOUBLE_TOP': 2, 'HEAD_AND_SHOULDERS': 3, 'INVERSE_HEAD_AND_SHOULDERS': 4}
            for key, val in encoding.items():
                if key in pat_str:
                    features.pattern_type = val
                    break
        features.pattern_confidence = getattr(pattern_result, 'confidence', 0.5)
        features.pattern_confirmed = 1 if getattr(pattern_result, 'is_confirmed', False) else 0
        signal = getattr(pattern_result, 'signal', None)
        if signal:
            sig_str = getattr(signal, 'value', str(signal)).upper()
            if (trade_direction == "BUY" and "BULL" in sig_str) or (trade_direction == "SELL" and "BEAR" in sig_str):
                features.pattern_matches_trade = 1
        confirmation_bonus = 0.3 if features.pattern_confirmed else 0
        features.pattern_score = min(1.0, features.pattern_confidence * (0.6 + 0.4 * features.pattern_matches_trade) + confirmation_bonus)

    def _extract_volatility_features(self, features, current_atr, current_spread):
        features.atr_value = current_atr
        features.spread_pips = current_spread
        if current_atr > 0:
            self._atr_history.append(current_atr)
        if current_spread >= 0:
            self._spread_history.append(current_spread)
        if len(self._atr_history) > self._max_history:
            self._atr_history = self._atr_history[-self._max_history:]
        if len(self._spread_history) > self._max_history:
            self._spread_history = self._spread_history[-self._max_history:]
        # FIXED: Rank percentile = count(hist <= val) / len(hist)
        if len(self._atr_history) >= 20:
            count_le = sum(1 for v in self._atr_history if v <= current_atr)
            features.atr_percentile = np.clip(count_le / len(self._atr_history), 0.0, 1.0)
        if len(self._spread_history) >= 20:
            count_le = sum(1 for v in self._spread_history if v <= current_spread)
            features.spread_percentile = np.clip(count_le / len(self._spread_history), 0.0, 1.0)
        features.volatility_regime = 0 if features.atr_percentile < 0.33 else (1 if features.atr_percentile < 0.67 else 2)
        features.spread_regime = 0 if features.spread_percentile < 0.33 else (1 if features.spread_percentile < 0.67 else 2)

    def _extract_structure_features(self, features, entry_price, pair):
        pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
        swing_high = self._last_swing_high.get(pair, 0)
        swing_low = self._last_swing_low.get(pair, 0)
        if swing_high > 0:
            features.distance_to_swing_high = abs(entry_price - swing_high) * pip_mult
        if swing_low > 0:
            features.distance_to_swing_low = abs(entry_price - swing_low) * pip_mult
        if swing_high > 0 and swing_low > 0:
            features.distance_to_structure = min(features.distance_to_swing_high, features.distance_to_swing_low)
        elif swing_high > 0:
            features.distance_to_structure = features.distance_to_swing_high
        elif swing_low > 0:
            features.distance_to_structure = features.distance_to_swing_low

    def _extract_geometry_features(self, features, entry_price, stop_loss, current_atr, pair):
        pip_mult = 100 if "JPY" in pair else (10 if "XAU" in pair else 10000)
        features.stop_distance_pips = abs(entry_price - stop_loss) * pip_mult
        if current_atr > 0:
            features.stop_distance_atr_ratio = abs(entry_price - stop_loss) / current_atr
        features.expected_r_to_tp1 = 2.0

    def _extract_time_features(self, features, timestamp, pair):
        features.hour_of_day = timestamp.hour
        features.day_of_week = timestamp.weekday()
        hour = timestamp.hour
        features.is_london_session = 1 if 8 <= hour < 16 else 0
        features.is_ny_session = 1 if 13 <= hour < 21 else 0
        features.is_overlap_session = 1 if 13 <= hour < 16 else 0
        last_time = self._last_signal_time.get(pair)
        if last_time:
            delta = timestamp - last_time
            features.time_since_last_signal = delta.total_seconds() / 3600.0
        else:
            features.time_since_last_signal = 999.0
        self._last_signal_time[pair] = timestamp

    def reset_history(self):
        self._atr_history = []
        self._spread_history = []
        self._last_signal_time = {}
        self._last_swing_high = {}
        self._last_swing_low = {}
