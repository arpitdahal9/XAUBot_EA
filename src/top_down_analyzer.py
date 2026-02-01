"""
Top-Down Analysis (TDA) Module
Based on Arpit Dahal's TPU Masterclass Strategy

Multi-Timeframe Analysis:
- Weekly (W) = Macro trend direction
- Daily (D) = Intermediate trend + supply/demand zones
- 4-Hour (H4) = Setup refinement
- 1-Hour (H1) = Entry timing

CRITICAL RULE: Never trade against the weekly bias!

Alignment Quality:
- PERFECT: All 4 timeframes agree (95% confidence)
- GOOD: 3/4 timeframes agree (80% confidence)
- WEAK: Less than 3 agree (50% confidence - NO TRADE)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import numpy as np


class TrendBias(Enum):
    """Market trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class AlignmentQuality(Enum):
    """Quality of multi-timeframe alignment."""
    PERFECT = "PERFECT"    # 4/4 timeframes agree
    GOOD = "GOOD"          # 3/4 timeframes agree
    WEAK = "WEAK"          # Less than 3 agree


@dataclass
class TimeframeBias:
    """Bias analysis for a single timeframe."""
    timeframe: str
    bias: TrendBias
    confidence: float
    sma_short: float
    sma_long: float
    price_above_sma: bool
    trend_strength: float  # 0.0 to 1.0
    higher_highs: bool
    higher_lows: bool


@dataclass
class TDAResult:
    """Result of Top-Down Analysis."""
    weekly_bias: TimeframeBias
    daily_bias: TimeframeBias
    h4_bias: TimeframeBias
    h1_bias: TimeframeBias
    
    alignment: AlignmentQuality
    overall_bias: TrendBias
    confidence: float
    
    bullish_count: int
    bearish_count: int
    neutral_count: int
    
    can_trade: bool
    trade_direction: Optional[TrendBias]
    reason: str


@dataclass
class SupplyDemandZone:
    """Supply or demand zone."""
    zone_type: str  # "SUPPLY" or "DEMAND"
    high: float
    low: float
    strength: float  # How many times tested
    timeframe: str


class TopDownAnalyzer:
    """
    Analyze multiple timeframes to establish trade bias.
    
    The goal is to ensure we only trade in the direction of the
    higher timeframe trend, dramatically reducing false signals.
    
    Process:
    1. Analyze Weekly for macro trend
    2. Analyze Daily for intermediate trend + zones
    3. Analyze H4 for setup
    4. Analyze H1 for entry
    5. Only trade if 3+ timeframes agree
    """
    
    def __init__(
        self,
        short_sma_period: int = 9,
        long_sma_period: int = 20,
        trend_confirmation_candles: int = 3
    ):
        self.short_sma_period = short_sma_period
        self.long_sma_period = long_sma_period
        self.trend_confirmation_candles = trend_confirmation_candles
        self.logger = logging.getLogger(__name__)
    
    def run_tda(
        self,
        weekly_candles: List,
        daily_candles: List,
        h4_candles: List,
        h1_candles: List
    ) -> TDAResult:
        """
        Run full Top-Down Analysis on all timeframes.
        
        Args:
            weekly_candles: Weekly candles (52+ recommended)
            daily_candles: Daily candles (20+ recommended)
            h4_candles: 4-hour candles (50+ recommended)
            h1_candles: 1-hour candles (100+ recommended)
            
        Returns:
            TDAResult with alignment and trade recommendation
        """
        # Analyze each timeframe
        weekly = self._analyze_timeframe(weekly_candles, "W")
        daily = self._analyze_timeframe(daily_candles, "D")
        h4 = self._analyze_timeframe(h4_candles, "H4")
        h1 = self._analyze_timeframe(h1_candles, "H1")
        
        # Count biases
        biases = [weekly.bias, daily.bias, h4.bias, h1.bias]
        bullish_count = sum(1 for b in biases if b == TrendBias.BULLISH)
        bearish_count = sum(1 for b in biases if b == TrendBias.BEARISH)
        neutral_count = sum(1 for b in biases if b == TrendBias.NEUTRAL)
        
        # Determine alignment quality
        if bullish_count == 4 or bearish_count == 4:
            alignment = AlignmentQuality.PERFECT
            confidence = 0.95
        elif bullish_count >= 3 or bearish_count >= 3:
            alignment = AlignmentQuality.GOOD
            confidence = 0.80
        else:
            alignment = AlignmentQuality.WEAK
            confidence = 0.50
        
        # Determine overall bias (weighted toward higher timeframes)
        # Weekly has highest weight
        if weekly.bias != TrendBias.NEUTRAL:
            overall_bias = weekly.bias
        elif daily.bias != TrendBias.NEUTRAL:
            overall_bias = daily.bias
        elif bullish_count > bearish_count:
            overall_bias = TrendBias.BULLISH
        elif bearish_count > bullish_count:
            overall_bias = TrendBias.BEARISH
        else:
            overall_bias = TrendBias.NEUTRAL
        
        # Determine if we can trade
        can_trade = alignment in [AlignmentQuality.PERFECT, AlignmentQuality.GOOD]
        trade_direction = overall_bias if can_trade and overall_bias != TrendBias.NEUTRAL else None
        
        # Generate reason
        if alignment == AlignmentQuality.PERFECT:
            reason = f"PERFECT alignment: All 4 timeframes {overall_bias.value}"
        elif alignment == AlignmentQuality.GOOD:
            reason = f"GOOD alignment: 3/4 timeframes favor {overall_bias.value}"
        else:
            reason = f"WEAK alignment: Mixed signals (B:{bullish_count} N:{neutral_count} S:{bearish_count})"
        
        self.logger.info(f"[TDA] {reason}")
        self.logger.info(f"[TDA] W:{weekly.bias.value} D:{daily.bias.value} H4:{h4.bias.value} H1:{h1.bias.value}")
        
        return TDAResult(
            weekly_bias=weekly,
            daily_bias=daily,
            h4_bias=h4,
            h1_bias=h1,
            alignment=alignment,
            overall_bias=overall_bias,
            confidence=confidence,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            can_trade=can_trade,
            trade_direction=trade_direction,
            reason=reason
        )
    
    def _analyze_timeframe(self, candles: List, timeframe: str) -> TimeframeBias:
        """
        Analyze a single timeframe for trend bias.
        
        Uses multiple methods:
        1. SMA crossover (9/20)
        2. Price position relative to SMAs
        3. Higher highs/lows (trend structure)
        """
        if len(candles) < self.long_sma_period + 5:
            return TimeframeBias(
                timeframe=timeframe,
                bias=TrendBias.NEUTRAL,
                confidence=0.3,
                sma_short=0,
                sma_long=0,
                price_above_sma=False,
                trend_strength=0,
                higher_highs=False,
                higher_lows=False
            )
        
        # Extract prices
        closes = [c.close if hasattr(c, 'close') else c['close'] for c in candles]
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in candles]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in candles]
        
        # Calculate SMAs
        sma_short = self._calculate_sma(closes, self.short_sma_period)
        sma_long = self._calculate_sma(closes, self.long_sma_period)
        current_price = closes[-1]
        
        # Check SMA relationship
        sma_bullish = sma_short > sma_long
        price_above_short = current_price > sma_short
        price_above_long = current_price > sma_long
        
        # Check trend structure (higher highs, higher lows)
        recent_highs = highs[-self.trend_confirmation_candles:]
        recent_lows = lows[-self.trend_confirmation_candles:]
        prev_highs = highs[-(self.trend_confirmation_candles * 2):-self.trend_confirmation_candles]
        prev_lows = lows[-(self.trend_confirmation_candles * 2):-self.trend_confirmation_candles]
        
        higher_highs = max(recent_highs) > max(prev_highs) if prev_highs else False
        higher_lows = min(recent_lows) > min(prev_lows) if prev_lows else False
        lower_highs = max(recent_highs) < max(prev_highs) if prev_highs else False
        lower_lows = min(recent_lows) < min(prev_lows) if prev_lows else False
        
        # Calculate trend strength (0-1)
        bullish_signals = sum([
            sma_bullish,
            price_above_short,
            price_above_long,
            higher_highs,
            higher_lows
        ])
        bearish_signals = sum([
            not sma_bullish,
            not price_above_short,
            not price_above_long,
            lower_highs,
            lower_lows
        ])
        
        # Determine bias
        if bullish_signals >= 4:
            bias = TrendBias.BULLISH
            trend_strength = bullish_signals / 5.0
        elif bearish_signals >= 4:
            bias = TrendBias.BEARISH
            trend_strength = bearish_signals / 5.0
        elif bullish_signals >= 3:
            bias = TrendBias.BULLISH
            trend_strength = bullish_signals / 5.0
        elif bearish_signals >= 3:
            bias = TrendBias.BEARISH
            trend_strength = bearish_signals / 5.0
        else:
            bias = TrendBias.NEUTRAL
            trend_strength = 0.5
        
        return TimeframeBias(
            timeframe=timeframe,
            bias=bias,
            confidence=trend_strength,
            sma_short=sma_short,
            sma_long=sma_long,
            price_above_sma=price_above_long,
            trend_strength=trend_strength,
            higher_highs=higher_highs,
            higher_lows=higher_lows
        )
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period
    
    def get_bias(self, candles: List) -> TrendBias:
        """
        Get simple bias from candles (for compatibility).
        
        Args:
            candles: List of candles
            
        Returns:
            TrendBias (BULLISH, BEARISH, NEUTRAL)
        """
        result = self._analyze_timeframe(candles, "")
        return result.bias
    
    def find_supply_demand_zones(
        self,
        candles: List,
        timeframe: str,
        zone_threshold: float = 0.002  # 0.2% price range
    ) -> List[SupplyDemandZone]:
        """
        Identify supply and demand zones from price action.
        
        Supply zones: Areas where price repeatedly fell from (resistance)
        Demand zones: Areas where price repeatedly bounced from (support)
        """
        zones = []
        
        if len(candles) < 20:
            return zones
        
        highs = [c.high if hasattr(c, 'high') else c['high'] for c in candles]
        lows = [c.low if hasattr(c, 'low') else c['low'] for c in candles]
        
        # Find swing highs (potential supply zones)
        for i in range(5, len(candles) - 5):
            is_swing_high = all(
                highs[i] > highs[j] for j in range(i-3, i+4) if j != i
            )
            if is_swing_high:
                zone_high = highs[i]
                zone_low = zone_high * (1 - zone_threshold)
                
                # Count how many times price touched this zone
                touches = sum(
                    1 for h in highs if zone_low <= h <= zone_high
                )
                
                if touches >= 2:
                    zones.append(SupplyDemandZone(
                        zone_type="SUPPLY",
                        high=zone_high,
                        low=zone_low,
                        strength=touches,
                        timeframe=timeframe
                    ))
        
        # Find swing lows (potential demand zones)
        for i in range(5, len(candles) - 5):
            is_swing_low = all(
                lows[i] < lows[j] for j in range(i-3, i+4) if j != i
            )
            if is_swing_low:
                zone_low = lows[i]
                zone_high = zone_low * (1 + zone_threshold)
                
                # Count touches
                touches = sum(
                    1 for l in lows if zone_low <= l <= zone_high
                )
                
                if touches >= 2:
                    zones.append(SupplyDemandZone(
                        zone_type="DEMAND",
                        high=zone_high,
                        low=zone_low,
                        strength=touches,
                        timeframe=timeframe
                    ))
        
        # Sort by strength
        zones.sort(key=lambda z: z.strength, reverse=True)
        
        return zones[:10]  # Return top 10 zones
    
    def is_price_at_zone(
        self,
        current_price: float,
        zones: List[SupplyDemandZone],
        zone_type: str = "DEMAND"
    ) -> Tuple[bool, Optional[SupplyDemandZone]]:
        """
        Check if current price is at a supply/demand zone.
        
        Args:
            current_price: Current market price
            zones: List of identified zones
            zone_type: "SUPPLY" or "DEMAND"
            
        Returns:
            Tuple of (is_at_zone, zone_if_found)
        """
        for zone in zones:
            if zone.zone_type != zone_type:
                continue
            
            if zone.low <= current_price <= zone.high:
                return True, zone
        
        return False, None
