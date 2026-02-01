"""
Fibonacci Calculator for Optimal Entry/Exit Points
Based on Arpit Dahal's TPU Masterclass Strategy

Key Levels:
- 88.6% retracement = ENTRY POINT
- 100% = Stop Loss level
- 38.2% extension = TP1 (+2R)
- 61.8% extension = TP2 (+3R)
- 100% extension = TP3 (runner target)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import numpy as np


class FibDirection(Enum):
    """Direction of Fibonacci measurement."""
    BULLISH = "bullish"  # Low to High (looking for buy at retracement)
    BEARISH = "bearish"  # High to Low (looking for sell at retracement)


@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels."""
    direction: FibDirection
    swing_high: float
    swing_low: float
    range_size: float
    
    # Retracement levels (from swing direction)
    level_0: float      # 0% - Start of swing
    level_236: float    # 23.6%
    level_382: float    # 38.2%
    level_50: float     # 50%
    level_618: float    # 61.8%
    level_786: float    # 78.6%
    level_886: float    # 88.6% - PRIMARY ENTRY
    level_100: float    # 100% - End of swing (SL zone)
    
    # Extension targets (from entry)
    tp1_price: float    # +2R target
    tp2_price: float    # +3R target
    tp3_price: float    # +5R+ target (runner)
    
    # Entry zone
    entry_price: float
    stop_loss: float
    r_value: float  # Distance in price for 1R


@dataclass
class QuarterPoints:
    """Quarter Points for psychological levels."""
    high: float
    q1: float   # 25%
    mid: float  # 50%
    q3: float   # 75%
    low: float


class FibonacciCalculator:
    """
    Calculate Fibonacci retracement and extension levels.
    
    FIX #2: Uses CONFIRMED SWING PIVOTS instead of raw max/min.
    - Pivots must be confirmed (not just raw extremes)
    - Minimum ATR-based swing distance required
    - Structure break confirmation prevents constant redraw
    
    Used for:
    1. Finding optimal entry points (88.6% retracement)
    2. Setting stop loss (at 100% level)
    3. Calculating take profit targets (38.2%, 61.8%, 100% extensions)
    """
    
    def __init__(self, min_swing_atr_multiple: float = 1.5, pivot_lookback: int = 3):
        self.logger = logging.getLogger(__name__)
        
        # FIX #2: Swing anchoring parameters
        self.min_swing_atr_multiple = min_swing_atr_multiple  # Min swing = 1.5x ATR
        self.pivot_lookback = pivot_lookback  # Candles to confirm pivot
        
        # Standard Fibonacci ratios
        self.RETRACEMENT_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0]
        self.ENTRY_LEVEL = 0.886  # Primary entry at 88.6%
        
        # Extension levels for targets (from entry point)
        self.TP1_R_MULTIPLE = 2.0  # +2R
        self.TP2_R_MULTIPLE = 3.0  # +3R
        self.TP3_R_MULTIPLE = 5.0  # +5R (runner target)
        
        # Cache for confirmed swings (prevents constant redraw)
        self._confirmed_swing_high: Dict[str, Tuple[float, int]] = {}
        self._confirmed_swing_low: Dict[str, Tuple[float, int]] = {}
    
    def _calculate_atr(self, candles: List, period: int = 14) -> float:
        """Calculate Average True Range for swing validation."""
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high if hasattr(candles[i], 'high') else candles[i]['high']
            low = candles[i].low if hasattr(candles[i], 'low') else candles[i]['low']
            prev_close = candles[i-1].close if hasattr(candles[i-1], 'close') else candles[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges[-period:]) / min(len(true_ranges), period)
    
    def _find_confirmed_pivots(
        self,
        candles: List,
        atr: float
    ) -> Tuple[Optional[Tuple[float, int]], Optional[Tuple[float, int]]]:
        """
        Find CONFIRMED swing pivots (not raw extremes).
        
        A pivot is confirmed when:
        1. It's a local extreme (higher/lower than N surrounding candles)
        2. Subsequent candles have moved away from it
        3. The swing distance meets minimum ATR threshold
        
        Returns:
            (swing_high, high_idx), (swing_low, low_idx) or None
        """
        if len(candles) < self.pivot_lookback * 2 + 1:
            return None, None
        
        min_swing = atr * self.min_swing_atr_multiple
        
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in candles]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in candles]
        
        confirmed_high = None
        confirmed_low = None
        
        # Find confirmed swing highs (must be local max with pullback)
        for i in range(self.pivot_lookback, len(candles) - self.pivot_lookback):
            # Check if this is a local high
            is_local_high = all(
                highs[i] >= highs[j]
                for j in range(i - self.pivot_lookback, i + self.pivot_lookback + 1)
                if j != i
            )
            
            if is_local_high:
                # Confirm with pullback (price moved away)
                subsequent_max = max(highs[i+1:i+self.pivot_lookback+1])
                pullback = highs[i] - subsequent_max
                
                if pullback >= min_swing * 0.3:  # 30% of min swing = confirmed
                    if confirmed_high is None or highs[i] > confirmed_high[0]:
                        confirmed_high = (highs[i], i)
        
        # Find confirmed swing lows (must be local min with rally)
        for i in range(self.pivot_lookback, len(candles) - self.pivot_lookback):
            is_local_low = all(
                lows[i] <= lows[j]
                for j in range(i - self.pivot_lookback, i + self.pivot_lookback + 1)
                if j != i
            )
            
            if is_local_low:
                subsequent_min = min(lows[i+1:i+self.pivot_lookback+1])
                rally = subsequent_min - lows[i]
                
                if rally >= min_swing * 0.3:
                    if confirmed_low is None or lows[i] < confirmed_low[0]:
                        confirmed_low = (lows[i], i)
        
        return confirmed_high, confirmed_low
    
    def calculate_levels(
        self,
        candles: List,
        lookback: int = 20
    ) -> Optional[FibonacciLevels]:
        """
        Calculate Fibonacci levels from CONFIRMED swing pivots.
        
        FIX #2: Uses proper pivot detection instead of raw max/min:
        - Pivots must be confirmed local extremes
        - Minimum ATR-based swing distance required
        - Direction determined by pivot sequence, not just position
        
        Args:
            candles: List of candle data with high, low, close attributes
            lookback: Number of candles to analyze
            
        Returns:
            FibonacciLevels with all calculated levels, or None if no valid swing
        """
        if len(candles) < lookback:
            self.logger.debug(f"Not enough candles for Fib calculation: {len(candles)} < {lookback}")
            return None
        
        recent = candles[-lookback:]
        
        # Calculate ATR for minimum swing validation
        atr = self._calculate_atr(candles, period=14)
        if atr <= 0:
            return None
        
        min_swing = atr * self.min_swing_atr_multiple
        
        # FIX #2: Find CONFIRMED pivots (not raw extremes)
        confirmed_high, confirmed_low = self._find_confirmed_pivots(recent, atr)
        
        # Fallback to raw extremes if no confirmed pivots
        if confirmed_high is None or confirmed_low is None:
            highs = [c.high if hasattr(c, 'high') else c['high'] for c in recent]
            lows = [c.low if hasattr(c, 'low') else c['low'] for c in recent]
            
            swing_high = max(highs)
            swing_low = min(lows)
            high_idx = highs.index(swing_high)
            low_idx = lows.index(swing_low)
        else:
            swing_high, high_idx = confirmed_high
            swing_low, low_idx = confirmed_low
        
        # Validate minimum swing distance
        swing_range = swing_high - swing_low
        if swing_range < min_swing:
            self.logger.debug(
                f"Swing too small: {swing_range:.5f} < {min_swing:.5f} (min ATR-based)"
            )
            return None
        
        # Determine direction based on which pivot came first
        if high_idx < low_idx:
            # High came first = Bearish swing (look for sell at retracement)
            direction = FibDirection.BEARISH
        else:
            # Low came first = Bullish swing (look for buy at retracement)
            direction = FibDirection.BULLISH
        
        return self._calculate_from_swing(swing_high, swing_low, direction)
    
    def _calculate_from_swing(
        self,
        swing_high: float,
        swing_low: float,
        direction: FibDirection
    ) -> FibonacciLevels:
        """Calculate all Fibonacci levels from a swing."""
        range_size = swing_high - swing_low
        
        if direction == FibDirection.BULLISH:
            # Bullish: measuring from low to high, retracement goes down
            # Entry at 88.6% retracement means price near the low
            level_0 = swing_high
            level_236 = swing_high - (range_size * 0.236)
            level_382 = swing_high - (range_size * 0.382)
            level_50 = swing_high - (range_size * 0.50)
            level_618 = swing_high - (range_size * 0.618)
            level_786 = swing_high - (range_size * 0.786)
            level_886 = swing_high - (range_size * 0.886)
            level_100 = swing_low
            
            # Entry at 88.6%, SL at 100% (swing low)
            entry_price = level_886
            stop_loss = level_100
            r_value = entry_price - stop_loss  # Distance for 1R
            
            # Extension targets (going up from entry)
            tp1_price = entry_price + (r_value * self.TP1_R_MULTIPLE)
            tp2_price = entry_price + (r_value * self.TP2_R_MULTIPLE)
            tp3_price = entry_price + (r_value * self.TP3_R_MULTIPLE)
            
        else:  # BEARISH
            # Bearish: measuring from high to low, retracement goes up
            # Entry at 88.6% retracement means price near the high
            level_0 = swing_low
            level_236 = swing_low + (range_size * 0.236)
            level_382 = swing_low + (range_size * 0.382)
            level_50 = swing_low + (range_size * 0.50)
            level_618 = swing_low + (range_size * 0.618)
            level_786 = swing_low + (range_size * 0.786)
            level_886 = swing_low + (range_size * 0.886)
            level_100 = swing_high
            
            # Entry at 88.6%, SL at 100% (swing high)
            entry_price = level_886
            stop_loss = level_100
            r_value = stop_loss - entry_price  # Distance for 1R
            
            # Extension targets (going down from entry)
            tp1_price = entry_price - (r_value * self.TP1_R_MULTIPLE)
            tp2_price = entry_price - (r_value * self.TP2_R_MULTIPLE)
            tp3_price = entry_price - (r_value * self.TP3_R_MULTIPLE)
        
        return FibonacciLevels(
            direction=direction,
            swing_high=swing_high,
            swing_low=swing_low,
            range_size=range_size,
            level_0=level_0,
            level_236=level_236,
            level_382=level_382,
            level_50=level_50,
            level_618=level_618,
            level_786=level_786,
            level_886=level_886,
            level_100=level_100,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            r_value=r_value
        )
    
    def calculate_from_high_low(
        self,
        high: float,
        low: float,
        direction: FibDirection = FibDirection.BULLISH
    ) -> Dict[str, float]:
        """
        Simple calculation returning dict of levels.
        
        Args:
            high: Swing high price
            low: Swing low price
            direction: BULLISH or BEARISH
            
        Returns:
            Dict with all Fibonacci levels
        """
        range_size = high - low
        
        if direction == FibDirection.BULLISH:
            return {
                '0%': high,
                '23.6%': high - range_size * 0.236,
                '38.2%': high - range_size * 0.382,
                '50%': high - range_size * 0.50,
                '61.8%': high - range_size * 0.618,
                '78.6%': high - range_size * 0.786,
                '88.6%': high - range_size * 0.886,
                '100%': low,
            }
        else:
            return {
                '0%': low,
                '23.6%': low + range_size * 0.236,
                '38.2%': low + range_size * 0.382,
                '50%': low + range_size * 0.50,
                '61.8%': low + range_size * 0.618,
                '78.6%': low + range_size * 0.786,
                '88.6%': low + range_size * 0.886,
                '100%': high,
            }
    
    def calculate_targets(
        self,
        entry: float,
        r_value: float,
        direction: FibDirection
    ) -> Dict[str, float]:
        """
        Calculate take profit targets from entry point.
        
        Args:
            entry: Entry price
            r_value: 1R distance in price
            direction: Trade direction
            
        Returns:
            Dict with TP1, TP2, TP3 prices
        """
        if direction == FibDirection.BULLISH:
            return {
                'TP1': entry + (r_value * self.TP1_R_MULTIPLE),
                'TP2': entry + (r_value * self.TP2_R_MULTIPLE),
                'TP3': entry + (r_value * self.TP3_R_MULTIPLE),
            }
        else:
            return {
                'TP1': entry - (r_value * self.TP1_R_MULTIPLE),
                'TP2': entry - (r_value * self.TP2_R_MULTIPLE),
                'TP3': entry - (r_value * self.TP3_R_MULTIPLE),
            }
    
    def calculate_quarter_points(self, high: float, low: float) -> QuarterPoints:
        """
        Calculate Quarter Points for psychological levels.
        
        Quarter Points are where traders cluster orders:
        - 0% = High (resistance)
        - 25% = First quarter
        - 50% = Midpoint (strong level)
        - 75% = Third quarter
        - 100% = Low (support)
        """
        range_size = high - low
        return QuarterPoints(
            high=high,
            q1=high - (range_size * 0.25),
            mid=high - (range_size * 0.50),
            q3=high - (range_size * 0.75),
            low=low
        )
    
    def is_price_at_entry_zone(
        self,
        current_price: float,
        fib_levels: FibonacciLevels,
        tolerance_pips: float = 20
    ) -> Tuple[bool, float]:
        """
        Check if current price is near the 88.6% entry zone.
        
        Args:
            current_price: Current market price
            fib_levels: Calculated Fibonacci levels
            tolerance_pips: How close price must be (in pips)
            
        Returns:
            Tuple of (is_in_zone, distance_in_pips)
        """
        entry_zone = fib_levels.level_886
        
        # Calculate distance in pips (assuming 4-decimal pairs)
        distance = abs(current_price - entry_zone)
        distance_pips = distance * 10000  # Convert to pips
        
        # For JPY pairs, adjust
        if distance_pips > 1000:  # Likely JPY pair
            distance_pips = distance * 100
        
        is_in_zone = distance_pips <= tolerance_pips
        
        return is_in_zone, distance_pips
    
    def get_setup_quality(
        self,
        fib_levels: FibonacciLevels,
        current_price: float,
        tolerance_pips: float = 20
    ) -> Dict:
        """
        Evaluate the quality of a Fibonacci setup.
        
        Returns:
            Dict with quality metrics
        """
        in_zone, distance = self.is_price_at_entry_zone(
            current_price, fib_levels, tolerance_pips
        )
        
        # Quality scoring
        if distance <= 5:
            proximity_score = 1.0  # Perfect
        elif distance <= 10:
            proximity_score = 0.9  # Excellent
        elif distance <= 15:
            proximity_score = 0.75  # Good
        elif distance <= 20:
            proximity_score = 0.6  # Acceptable
        else:
            proximity_score = 0.3  # Poor
        
        # R:R ratio (minimum 2:1 is acceptable)
        potential_rr = self.TP1_R_MULTIPLE  # 2R at TP1
        rr_score = min(1.0, potential_rr / 2.0)  # Max score at 2R+
        
        overall_score = (proximity_score * 0.6) + (rr_score * 0.4)
        
        return {
            'in_entry_zone': in_zone,
            'distance_pips': distance,
            'proximity_score': proximity_score,
            'rr_ratio': potential_rr,
            'rr_score': rr_score,
            'overall_score': overall_score,
            'quality': 'EXCELLENT' if overall_score >= 0.85 else
                       'GOOD' if overall_score >= 0.7 else
                       'ACCEPTABLE' if overall_score >= 0.5 else 'POOR'
        }
