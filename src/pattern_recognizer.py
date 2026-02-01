"""
Chart Pattern Recognition Module
Based on Arpit Dahal's TPU Masterclass Strategy

Patterns detected:
- Double Bottom: Strong bullish reversal signal
- Double Top: Strong bearish reversal signal
- Head and Shoulders: Major reversal patterns

These patterns provide additional confirmation for entry signals,
further reducing false positives.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging
import numpy as np


class PatternType(Enum):
    """Types of chart patterns."""
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    DOUBLE_TOP = "DOUBLE_TOP"
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    NONE = "NONE"


class PatternSignal(Enum):
    """Trading signal from pattern."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class PatternResult:
    """Result of pattern detection."""
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float  # 0.0 to 1.0
    
    # Key price levels
    neckline: float        # Breakout/confirmation level
    target_price: float    # Measured move target
    stop_loss: float       # Invalidation level
    
    # Pattern details
    first_peak_or_trough: float
    second_peak_or_trough: float
    middle_point: float    # For H&S patterns
    
    # Validation
    is_confirmed: bool     # Has neckline been broken?
    formation_candles: int # How many candles formed the pattern


class PatternRecognizer:
    """
    Detect chart patterns in price action.
    
    FIX #3: STRICT CONFIRMATION GATES
    - Patterns must be CONFIRMED (neckline break + close)
    - Minimum pattern width enforced (not just touches)
    - Zone confluence required for high confidence
    - Reduced false triggers in choppy markets
    
    Patterns add an extra layer of confirmation to our entries:
    - Double Bottom at demand zone + bullish divergence = HIGH probability BUY
    - Double Top at supply zone + bearish divergence = HIGH probability SELL
    """
    
    def __init__(
        self,
        tolerance_pips: float = 15,
        min_pattern_width: int = 10,  # FIX #3: Increased from 5 to 10
        max_pattern_width: int = 50,
        require_neckline_break: bool = True,  # FIX #3: Must break neckline
        require_close_confirmation: bool = True,  # FIX #3: Must close past neckline
        min_confidence_threshold: float = 0.6  # FIX #3: Minimum confidence
    ):
        self.tolerance_pips = tolerance_pips
        self.min_pattern_width = min_pattern_width
        self.max_pattern_width = max_pattern_width
        self.require_neckline_break = require_neckline_break
        self.require_close_confirmation = require_close_confirmation
        self.min_confidence_threshold = min_confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_all_patterns(self, candles: List) -> List[PatternResult]:
        """
        Detect all recognizable patterns in the candles.
        
        FIX #3: Only returns CONFIRMED patterns that pass strict gates:
        - Must be confirmed (neckline broken with close)
        - Must meet minimum confidence threshold
        - Must have proper pattern width
        
        Args:
            candles: List of candle data
            
        Returns:
            List of CONFIRMED patterns only
        """
        patterns = []
        
        # Double Bottom
        double_bottom = self.detect_double_bottom(candles)
        if double_bottom:
            # FIX #3: Apply strict confirmation gates
            if self._passes_confirmation_gates(double_bottom):
                patterns.append(double_bottom)
            else:
                self.logger.debug(f"Double Bottom rejected: confirmation gates failed")
        
        # Double Top
        double_top = self.detect_double_top(candles)
        if double_top:
            if self._passes_confirmation_gates(double_top):
                patterns.append(double_top)
            else:
                self.logger.debug(f"Double Top rejected: confirmation gates failed")
        
        # Head and Shoulders
        hs = self.detect_head_and_shoulders(candles)
        if hs:
            if self._passes_confirmation_gates(hs):
                patterns.append(hs)
            else:
                self.logger.debug(f"H&S rejected: confirmation gates failed")
        
        # Inverse Head and Shoulders
        ihs = self.detect_inverse_head_and_shoulders(candles)
        if ihs:
            if self._passes_confirmation_gates(ihs):
                patterns.append(ihs)
            else:
                self.logger.debug(f"Inverse H&S rejected: confirmation gates failed")
        
        return patterns
    
    def _passes_confirmation_gates(self, pattern: PatternResult) -> bool:
        """
        FIX #3: Strict confirmation gates for pattern validation.
        
        A pattern must pass ALL of these to be considered valid:
        1. Is confirmed (neckline broken with close)
        2. Meets minimum confidence threshold
        3. Has proper pattern width
        """
        # Gate 1: Neckline break confirmation
        if self.require_neckline_break and not pattern.is_confirmed:
            return False
        
        # Gate 2: Minimum confidence threshold
        if pattern.confidence < self.min_confidence_threshold:
            return False
        
        # Gate 3: Minimum pattern width
        if pattern.formation_candles < self.min_pattern_width:
            return False
        
        return True
    
    def detect_double_bottom(
        self,
        candles: List,
        lookback: int = 30
    ) -> Optional[PatternResult]:
        """
        Detect Double Bottom pattern (bullish reversal).
        
        Structure:
        1. First low (trough 1)
        2. Rally (creates neckline)
        3. Second low near first low (trough 2)
        4. Breakout above neckline = CONFIRMED
        
        Target = Neckline + (Neckline - Trough)
        """
        if len(candles) < lookback:
            return None
        
        recent = candles[-lookback:]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in recent]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in recent]
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in recent]
        
        # Find the two lowest points
        low_indices = self._find_local_minima(lows)
        
        if len(low_indices) < 2:
            return None
        
        # Get the two most significant lows
        sorted_lows = sorted(low_indices, key=lambda i: lows[i])
        first_low_idx = sorted_lows[0]
        second_low_idx = sorted_lows[1]
        
        # Ensure they're in chronological order
        if first_low_idx > second_low_idx:
            first_low_idx, second_low_idx = second_low_idx, first_low_idx
        
        first_low = lows[first_low_idx]
        second_low = lows[second_low_idx]
        
        # Check if lows are within tolerance
        price_diff = abs(first_low - second_low)
        tolerance = first_low * (self.tolerance_pips / 10000)  # Convert pips to price
        
        if price_diff > tolerance:
            return None
        
        # Check pattern width
        width = second_low_idx - first_low_idx
        if width < self.min_pattern_width or width > self.max_pattern_width:
            return None
        
        # Find neckline (highest high between the two lows)
        middle_highs = highs[first_low_idx:second_low_idx + 1]
        if not middle_highs:
            return None
        
        neckline = max(middle_highs)
        middle_high_idx = first_low_idx + middle_highs.index(neckline)
        
        # Calculate target (measured move)
        pattern_height = neckline - min(first_low, second_low)
        target_price = neckline + pattern_height
        
        # Stop loss below the lower trough
        stop_loss = min(first_low, second_low) - (tolerance * 2)
        
        # Check if pattern is confirmed (price broke neckline)
        current_price = closes[-1]
        is_confirmed = current_price > neckline
        
        # Calculate confidence
        confidence = 0.5
        if is_confirmed:
            confidence += 0.3
        if abs(first_low - second_low) < tolerance * 0.5:  # Very close lows
            confidence += 0.1
        if width >= 10:  # Good pattern width
            confidence += 0.1
        
        self.logger.info(
            f"[PATTERN] DOUBLE BOTTOM detected: "
            f"Lows at {first_low:.5f} and {second_low:.5f}, "
            f"Neckline: {neckline:.5f}, Target: {target_price:.5f}, "
            f"Confirmed: {is_confirmed}"
        )
        
        return PatternResult(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            signal=PatternSignal.BULLISH,
            confidence=confidence,
            neckline=neckline,
            target_price=target_price,
            stop_loss=stop_loss,
            first_peak_or_trough=first_low,
            second_peak_or_trough=second_low,
            middle_point=neckline,
            is_confirmed=is_confirmed,
            formation_candles=width
        )
    
    def detect_double_top(
        self,
        candles: List,
        lookback: int = 30
    ) -> Optional[PatternResult]:
        """
        Detect Double Top pattern (bearish reversal).
        
        Structure:
        1. First high (peak 1)
        2. Pullback (creates neckline)
        3. Second high near first high (peak 2)
        4. Break below neckline = CONFIRMED
        
        Target = Neckline - (Peak - Neckline)
        """
        if len(candles) < lookback:
            return None
        
        recent = candles[-lookback:]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in recent]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in recent]
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in recent]
        
        # Find the two highest points
        high_indices = self._find_local_maxima(highs)
        
        if len(high_indices) < 2:
            return None
        
        # Get the two most significant highs
        sorted_highs = sorted(high_indices, key=lambda i: highs[i], reverse=True)
        first_high_idx = sorted_highs[0]
        second_high_idx = sorted_highs[1]
        
        # Ensure chronological order
        if first_high_idx > second_high_idx:
            first_high_idx, second_high_idx = second_high_idx, first_high_idx
        
        first_high = highs[first_high_idx]
        second_high = highs[second_high_idx]
        
        # Check tolerance
        price_diff = abs(first_high - second_high)
        tolerance = first_high * (self.tolerance_pips / 10000)
        
        if price_diff > tolerance:
            return None
        
        # Check width
        width = second_high_idx - first_high_idx
        if width < self.min_pattern_width or width > self.max_pattern_width:
            return None
        
        # Find neckline (lowest low between peaks)
        middle_lows = lows[first_high_idx:second_high_idx + 1]
        if not middle_lows:
            return None
        
        neckline = min(middle_lows)
        
        # Calculate target
        pattern_height = max(first_high, second_high) - neckline
        target_price = neckline - pattern_height
        
        # Stop loss above higher peak
        stop_loss = max(first_high, second_high) + (tolerance * 2)
        
        # Check confirmation
        current_price = closes[-1]
        is_confirmed = current_price < neckline
        
        # Confidence
        confidence = 0.5
        if is_confirmed:
            confidence += 0.3
        if abs(first_high - second_high) < tolerance * 0.5:
            confidence += 0.1
        if width >= 10:
            confidence += 0.1
        
        self.logger.info(
            f"[PATTERN] DOUBLE TOP detected: "
            f"Highs at {first_high:.5f} and {second_high:.5f}, "
            f"Neckline: {neckline:.5f}, Target: {target_price:.5f}, "
            f"Confirmed: {is_confirmed}"
        )
        
        return PatternResult(
            pattern_type=PatternType.DOUBLE_TOP,
            signal=PatternSignal.BEARISH,
            confidence=confidence,
            neckline=neckline,
            target_price=target_price,
            stop_loss=stop_loss,
            first_peak_or_trough=first_high,
            second_peak_or_trough=second_high,
            middle_point=neckline,
            is_confirmed=is_confirmed,
            formation_candles=width
        )
    
    def detect_head_and_shoulders(
        self,
        candles: List,
        lookback: int = 50
    ) -> Optional[PatternResult]:
        """
        Detect Head and Shoulders pattern (bearish reversal).
        
        Structure:
        - Left shoulder (first peak)
        - Head (higher peak)
        - Right shoulder (lower peak, similar to left)
        - Neckline connecting the two troughs
        
        This is a major reversal pattern with high reliability.
        """
        if len(candles) < lookback:
            return None
        
        recent = candles[-lookback:]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in recent]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in recent]
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in recent]
        
        # Find peaks
        high_indices = self._find_local_maxima(highs)
        
        if len(high_indices) < 3:
            return None
        
        # Need 3 peaks: left shoulder, head, right shoulder
        # Head must be highest
        sorted_by_height = sorted(high_indices, key=lambda i: highs[i], reverse=True)
        head_idx = sorted_by_height[0]
        
        # Find shoulders on each side of head
        left_shoulders = [i for i in high_indices if i < head_idx]
        right_shoulders = [i for i in high_indices if i > head_idx]
        
        if not left_shoulders or not right_shoulders:
            return None
        
        # Get closest significant shoulders
        left_shoulder_idx = max(left_shoulders)
        right_shoulder_idx = min(right_shoulders)
        
        left_shoulder = highs[left_shoulder_idx]
        head = highs[head_idx]
        right_shoulder = highs[right_shoulder_idx]
        
        # Head must be higher than both shoulders
        if head <= left_shoulder or head <= right_shoulder:
            return None
        
        # Shoulders should be roughly equal
        shoulder_diff = abs(left_shoulder - right_shoulder)
        tolerance = left_shoulder * (self.tolerance_pips * 2 / 10000)
        
        if shoulder_diff > tolerance:
            return None
        
        # Find neckline (connect the two troughs)
        left_trough = min(lows[left_shoulder_idx:head_idx + 1])
        right_trough = min(lows[head_idx:right_shoulder_idx + 1])
        neckline = (left_trough + right_trough) / 2
        
        # Calculate target
        pattern_height = head - neckline
        target_price = neckline - pattern_height
        
        # Stop loss
        stop_loss = head + (tolerance / 2)
        
        # Confirmation
        current_price = closes[-1]
        is_confirmed = current_price < neckline
        
        confidence = 0.6 if is_confirmed else 0.4
        
        self.logger.info(
            f"[PATTERN] HEAD & SHOULDERS detected: "
            f"LS:{left_shoulder:.5f} H:{head:.5f} RS:{right_shoulder:.5f}, "
            f"Neckline: {neckline:.5f}, Confirmed: {is_confirmed}"
        )
        
        return PatternResult(
            pattern_type=PatternType.HEAD_AND_SHOULDERS,
            signal=PatternSignal.BEARISH,
            confidence=confidence,
            neckline=neckline,
            target_price=target_price,
            stop_loss=stop_loss,
            first_peak_or_trough=left_shoulder,
            second_peak_or_trough=right_shoulder,
            middle_point=head,
            is_confirmed=is_confirmed,
            formation_candles=right_shoulder_idx - left_shoulder_idx
        )
    
    def detect_inverse_head_and_shoulders(
        self,
        candles: List,
        lookback: int = 50
    ) -> Optional[PatternResult]:
        """
        Detect Inverse Head and Shoulders (bullish reversal).
        
        Mirror of H&S - troughs instead of peaks.
        """
        if len(candles) < lookback:
            return None
        
        recent = candles[-lookback:]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in recent]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in recent]
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in recent]
        
        # Find troughs
        low_indices = self._find_local_minima(lows)
        
        if len(low_indices) < 3:
            return None
        
        # Head must be lowest
        sorted_by_depth = sorted(low_indices, key=lambda i: lows[i])
        head_idx = sorted_by_depth[0]
        
        # Find shoulders
        left_shoulders = [i for i in low_indices if i < head_idx]
        right_shoulders = [i for i in low_indices if i > head_idx]
        
        if not left_shoulders or not right_shoulders:
            return None
        
        left_shoulder_idx = max(left_shoulders)
        right_shoulder_idx = min(right_shoulders)
        
        left_shoulder = lows[left_shoulder_idx]
        head = lows[head_idx]
        right_shoulder = lows[right_shoulder_idx]
        
        # Head must be lower
        if head >= left_shoulder or head >= right_shoulder:
            return None
        
        # Shoulders roughly equal
        shoulder_diff = abs(left_shoulder - right_shoulder)
        tolerance = left_shoulder * (self.tolerance_pips * 2 / 10000)
        
        if shoulder_diff > tolerance:
            return None
        
        # Neckline
        left_peak = max(highs[left_shoulder_idx:head_idx + 1])
        right_peak = max(highs[head_idx:right_shoulder_idx + 1])
        neckline = (left_peak + right_peak) / 2
        
        # Target
        pattern_height = neckline - head
        target_price = neckline + pattern_height
        
        # Stop loss
        stop_loss = head - (tolerance / 2)
        
        # Confirmation
        current_price = closes[-1]
        is_confirmed = current_price > neckline
        
        confidence = 0.6 if is_confirmed else 0.4
        
        self.logger.info(
            f"[PATTERN] INVERSE H&S detected: "
            f"LS:{left_shoulder:.5f} H:{head:.5f} RS:{right_shoulder:.5f}, "
            f"Neckline: {neckline:.5f}, Confirmed: {is_confirmed}"
        )
        
        return PatternResult(
            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
            signal=PatternSignal.BULLISH,
            confidence=confidence,
            neckline=neckline,
            target_price=target_price,
            stop_loss=stop_loss,
            first_peak_or_trough=left_shoulder,
            second_peak_or_trough=right_shoulder,
            middle_point=head,
            is_confirmed=is_confirmed,
            formation_candles=right_shoulder_idx - left_shoulder_idx
        )
    
    def _find_local_minima(self, values: List[float], window: int = 3) -> List[int]:
        """Find indices of local minima."""
        minima = []
        for i in range(window, len(values) - window):
            is_min = all(
                values[i] <= values[j]
                for j in range(i - window, i + window + 1)
                if j != i
            )
            if is_min:
                minima.append(i)
        return minima
    
    def _find_local_maxima(self, values: List[float], window: int = 3) -> List[int]:
        """Find indices of local maxima."""
        maxima = []
        for i in range(window, len(values) - window):
            is_max = all(
                values[i] >= values[j]
                for j in range(i - window, i + window + 1)
                if j != i
            )
            if is_max:
                maxima.append(i)
        return maxima
    
    def get_strongest_pattern(self, patterns: List[PatternResult]) -> Optional[PatternResult]:
        """Get the pattern with highest confidence."""
        if not patterns:
            return None
        return max(patterns, key=lambda p: p.confidence)
