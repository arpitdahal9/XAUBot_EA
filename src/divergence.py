"""
RSI Indicator and Divergence Detection
Based on Arpit Dahal's TPU Masterclass Strategy

Divergence is a HIGH-PROBABILITY confirmation signal:
- Bullish Divergence: Price makes LOWER low, RSI makes HIGHER low → Reversal UP
- Bearish Divergence: Price makes HIGHER high, RSI makes LOWER high → Reversal DOWN

This filters out 50%+ of false signals from other indicators.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging
import numpy as np


class DivergenceType(Enum):
    """Type of divergence detected."""
    BULLISH = "BULLISH"      # Price lower low, RSI higher low
    BEARISH = "BEARISH"      # Price higher high, RSI lower high
    HIDDEN_BULLISH = "HIDDEN_BULLISH"  # Price higher low, RSI lower low (trend continuation)
    HIDDEN_BEARISH = "HIDDEN_BEARISH"  # Price lower high, RSI higher high (trend continuation)
    NONE = "NONE"


class DivergenceStrength(Enum):
    """Strength of the divergence signal."""
    CONFIRMED = "CONFIRMED"    # Clear divergence with good separation
    MODERATE = "MODERATE"      # Divergence present but weak
    POTENTIAL = "POTENTIAL"    # Possible divergence forming
    NONE = "NONE"


@dataclass
class DivergenceResult:
    """Result of divergence analysis."""
    type: DivergenceType
    strength: DivergenceStrength
    price_point_1: float       # First price extreme
    price_point_2: float       # Second price extreme
    rsi_point_1: float         # RSI at first extreme
    rsi_point_2: float         # RSI at second extreme
    candles_apart: int         # Distance between the two points
    confidence: float          # 0.0 to 1.0


@dataclass
class RSIResult:
    """RSI calculation result."""
    current_value: float
    values: List[float]
    is_oversold: bool          # RSI < 30
    is_overbought: bool        # RSI > 70
    zone: str                  # "OVERSOLD", "NEUTRAL", "OVERBOUGHT"


class RSICalculator:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum and overbought/oversold conditions.
    - RSI < 30 = Oversold (potential buy)
    - RSI > 70 = Overbought (potential sell)
    - RSI 30-70 = Neutral
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, prices: List[float]) -> Optional[RSIResult]:
        """
        Calculate RSI from a list of closing prices.
        
        Args:
            prices: List of closing prices (oldest first)
            
        Returns:
            RSIResult with current RSI and full series
        """
        if len(prices) < self.period + 1:
            self.logger.warning(f"Not enough prices for RSI: {len(prices)} < {self.period + 1}")
            return None
        
        prices_array = np.array(prices, dtype=float)
        rsi_values = self._calculate_rsi_series(prices_array)
        
        current_rsi = rsi_values[-1]
        
        return RSIResult(
            current_value=current_rsi,
            values=rsi_values.tolist(),
            is_oversold=current_rsi < 30,
            is_overbought=current_rsi > 70,
            zone="OVERSOLD" if current_rsi < 30 else 
                 "OVERBOUGHT" if current_rsi > 70 else "NEUTRAL"
        )
    
    def _calculate_rsi_series(self, prices: np.ndarray) -> np.ndarray:
        """Calculate RSI for entire price series."""
        deltas = np.diff(prices)
        
        # Initialize
        rsi = np.zeros(len(prices))
        rsi[:self.period] = 50  # Default value for insufficient data
        
        # Calculate initial averages
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        if avg_loss == 0:
            rsi[self.period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[self.period] = 100 - (100 / (1 + rs))
        
        # Calculate RSI using Wilder's smoothing
        for i in range(self.period + 1, len(prices)):
            gain = gains[i - 1] if i - 1 < len(gains) else 0
            loss = losses[i - 1] if i - 1 < len(losses) else 0
            
            avg_gain = (avg_gain * (self.period - 1) + gain) / self.period
            avg_loss = (avg_loss * (self.period - 1) + loss) / self.period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_single(self, prices: List[float]) -> float:
        """Calculate just the current RSI value."""
        result = self.calculate(prices)
        return result.current_value if result else 50.0


class DivergenceDetector:
    """
    Detect bullish and bearish divergence between price and RSI.
    
    Divergence occurs when price and momentum move in opposite directions,
    signaling a potential reversal.
    
    Types:
    - Regular Bullish: Price makes lower low, RSI makes higher low → BUY signal
    - Regular Bearish: Price makes higher high, RSI makes lower high → SELL signal
    - Hidden Bullish: Price makes higher low, RSI makes lower low → Trend continuation (BUY)
    - Hidden Bearish: Price makes lower high, RSI makes higher high → Trend continuation (SELL)
    """
    
    def __init__(self, rsi_period: int = 14, lookback: int = 20, min_candles_apart: int = 5):
        self.rsi_calculator = RSICalculator(period=rsi_period)
        self.lookback = lookback
        self.min_candles_apart = min_candles_apart
        self.logger = logging.getLogger(__name__)
    
    def detect(self, candles: List) -> Optional[DivergenceResult]:
        """
        Detect divergence in recent candles.
        
        Args:
            candles: List of candle data with high, low, close attributes
            
        Returns:
            DivergenceResult if divergence found, None otherwise
        """
        if len(candles) < self.lookback + self.rsi_calculator.period:
            return None
        
        # Extract prices
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in candles]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in candles]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in candles]
        
        # Calculate RSI
        rsi_result = self.rsi_calculator.calculate(closes)
        if not rsi_result:
            return None
        
        rsi_values = rsi_result.values
        
        # Look for divergence in recent candles
        recent_highs = highs[-self.lookback:]
        recent_lows = lows[-self.lookback:]
        recent_rsi = rsi_values[-self.lookback:]
        
        # Check for bullish divergence (at lows)
        bullish = self._detect_bullish_divergence(recent_lows, recent_rsi)
        if bullish:
            return bullish
        
        # Check for bearish divergence (at highs)
        bearish = self._detect_bearish_divergence(recent_highs, recent_rsi)
        if bearish:
            return bearish
        
        return None
    
    def _detect_bullish_divergence(
        self,
        price_lows: List[float],
        rsi_values: List[float]
    ) -> Optional[DivergenceResult]:
        """
        Detect bullish divergence: Price lower low + RSI higher low.
        
        This signals that downward momentum is weakening despite price
        making new lows - a potential reversal UP.
        """
        # Find significant lows
        low_points = self._find_local_extremes(price_lows, find_lows=True)
        
        if len(low_points) < 2:
            return None
        
        # Get the two most recent significant lows
        point1_idx, point1_price = low_points[-2]
        point2_idx, point2_price = low_points[-1]
        
        # Must be at least min_candles_apart
        if point2_idx - point1_idx < self.min_candles_apart:
            return None
        
        # Get RSI at those points
        rsi1 = rsi_values[point1_idx]
        rsi2 = rsi_values[point2_idx]
        
        # Check for bullish divergence:
        # Price makes LOWER low (point2 < point1)
        # RSI makes HIGHER low (rsi2 > rsi1)
        price_lower = point2_price < point1_price
        rsi_higher = rsi2 > rsi1
        
        if price_lower and rsi_higher:
            # Calculate strength based on separation
            price_diff_pct = abs(point2_price - point1_price) / point1_price * 100
            rsi_diff = rsi2 - rsi1
            
            if rsi_diff >= 5 and price_diff_pct >= 0.1:
                strength = DivergenceStrength.CONFIRMED
                confidence = min(0.95, 0.7 + (rsi_diff / 50) + (price_diff_pct / 10))
            elif rsi_diff >= 3:
                strength = DivergenceStrength.MODERATE
                confidence = 0.6 + (rsi_diff / 50)
            else:
                strength = DivergenceStrength.POTENTIAL
                confidence = 0.4
            
            self.logger.info(
                f"[DIVERGENCE] BULLISH detected: Price {point1_price:.5f}->{point2_price:.5f} (lower), "
                f"RSI {rsi1:.1f}->{rsi2:.1f} (higher), Strength: {strength.value}"
            )
            
            return DivergenceResult(
                type=DivergenceType.BULLISH,
                strength=strength,
                price_point_1=point1_price,
                price_point_2=point2_price,
                rsi_point_1=rsi1,
                rsi_point_2=rsi2,
                candles_apart=point2_idx - point1_idx,
                confidence=confidence
            )
        
        # Check for hidden bullish (trend continuation)
        # Price makes HIGHER low, RSI makes LOWER low
        price_higher = point2_price > point1_price
        rsi_lower = rsi2 < rsi1
        
        if price_higher and rsi_lower:
            rsi_diff = abs(rsi2 - rsi1)
            if rsi_diff >= 3:
                return DivergenceResult(
                    type=DivergenceType.HIDDEN_BULLISH,
                    strength=DivergenceStrength.MODERATE,
                    price_point_1=point1_price,
                    price_point_2=point2_price,
                    rsi_point_1=rsi1,
                    rsi_point_2=rsi2,
                    candles_apart=point2_idx - point1_idx,
                    confidence=0.55
                )
        
        return None
    
    def _detect_bearish_divergence(
        self,
        price_highs: List[float],
        rsi_values: List[float]
    ) -> Optional[DivergenceResult]:
        """
        Detect bearish divergence: Price higher high + RSI lower high.
        
        This signals that upward momentum is weakening despite price
        making new highs - a potential reversal DOWN.
        """
        # Find significant highs
        high_points = self._find_local_extremes(price_highs, find_lows=False)
        
        if len(high_points) < 2:
            return None
        
        # Get the two most recent significant highs
        point1_idx, point1_price = high_points[-2]
        point2_idx, point2_price = high_points[-1]
        
        # Must be at least min_candles_apart
        if point2_idx - point1_idx < self.min_candles_apart:
            return None
        
        # Get RSI at those points
        rsi1 = rsi_values[point1_idx]
        rsi2 = rsi_values[point2_idx]
        
        # Check for bearish divergence:
        # Price makes HIGHER high (point2 > point1)
        # RSI makes LOWER high (rsi2 < rsi1)
        price_higher = point2_price > point1_price
        rsi_lower = rsi2 < rsi1
        
        if price_higher and rsi_lower:
            # Calculate strength
            price_diff_pct = abs(point2_price - point1_price) / point1_price * 100
            rsi_diff = rsi1 - rsi2
            
            if rsi_diff >= 5 and price_diff_pct >= 0.1:
                strength = DivergenceStrength.CONFIRMED
                confidence = min(0.95, 0.7 + (rsi_diff / 50) + (price_diff_pct / 10))
            elif rsi_diff >= 3:
                strength = DivergenceStrength.MODERATE
                confidence = 0.6 + (rsi_diff / 50)
            else:
                strength = DivergenceStrength.POTENTIAL
                confidence = 0.4
            
            self.logger.info(
                f"[DIVERGENCE] BEARISH detected: Price {point1_price:.5f}->{point2_price:.5f} (higher), "
                f"RSI {rsi1:.1f}->{rsi2:.1f} (lower), Strength: {strength.value}"
            )
            
            return DivergenceResult(
                type=DivergenceType.BEARISH,
                strength=strength,
                price_point_1=point1_price,
                price_point_2=point2_price,
                rsi_point_1=rsi1,
                rsi_point_2=rsi2,
                candles_apart=point2_idx - point1_idx,
                confidence=confidence
            )
        
        # Check for hidden bearish
        price_lower = point2_price < point1_price
        rsi_higher = rsi2 > rsi1
        
        if price_lower and rsi_higher:
            rsi_diff = abs(rsi2 - rsi1)
            if rsi_diff >= 3:
                return DivergenceResult(
                    type=DivergenceType.HIDDEN_BEARISH,
                    strength=DivergenceStrength.MODERATE,
                    price_point_1=point1_price,
                    price_point_2=point2_price,
                    rsi_point_1=rsi1,
                    rsi_point_2=rsi2,
                    candles_apart=point2_idx - point1_idx,
                    confidence=0.55
                )
        
        return None
    
    def _find_local_extremes(
        self,
        values: List[float],
        find_lows: bool = True,
        window: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Find local minima or maxima in a series.
        
        Args:
            values: List of values to search
            find_lows: True for minima, False for maxima
            window: Size of comparison window
            
        Returns:
            List of (index, value) tuples for extremes
        """
        extremes = []
        
        for i in range(window, len(values) - window):
            is_extreme = True
            current = values[i]
            
            # Check surrounding values
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                    
                if find_lows:
                    if values[j] < current:
                        is_extreme = False
                        break
                else:
                    if values[j] > current:
                        is_extreme = False
                        break
            
            if is_extreme:
                extremes.append((i, current))
        
        return extremes
    
    def detect_bullish_simple(
        self,
        price_lows: List[float],
        rsi_values: List[float]
    ) -> Optional[dict]:
        """
        Simple bullish divergence detection (for compatibility).
        
        Args:
            price_lows: Recent price lows
            rsi_values: RSI values at those points
            
        Returns:
            Dict with type and strength if divergence found
        """
        if len(price_lows) < 2 or len(rsi_values) < 2:
            return None
        
        # Price makes lower low, RSI makes higher low
        if price_lows[-1] < price_lows[-2] and rsi_values[-1] > rsi_values[-2]:
            return {
                "type": "BULLISH",
                "strength": "CONFIRMED",
                "price_change": price_lows[-1] - price_lows[-2],
                "rsi_change": rsi_values[-1] - rsi_values[-2]
            }
        return None
    
    def detect_bearish_simple(
        self,
        price_highs: List[float],
        rsi_values: List[float]
    ) -> Optional[dict]:
        """
        Simple bearish divergence detection (for compatibility).
        
        Args:
            price_highs: Recent price highs
            rsi_values: RSI values at those points
            
        Returns:
            Dict with type and strength if divergence found
        """
        if len(price_highs) < 2 or len(rsi_values) < 2:
            return None
        
        # Price makes higher high, RSI makes lower high
        if price_highs[-1] > price_highs[-2] and rsi_values[-1] < rsi_values[-2]:
            return {
                "type": "BEARISH",
                "strength": "CONFIRMED",
                "price_change": price_highs[-1] - price_highs[-2],
                "rsi_change": rsi_values[-1] - rsi_values[-2]
            }
        return None
