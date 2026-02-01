"""
Unit tests for Top-Down Analysis (TDA).
"""

import pytest
import sys
sys.path.insert(0, '.')

from src.top_down_analyzer import TopDownAnalyzer, TrendBias, AlignmentQuality


class MockCandle:
    """Mock candle for testing."""
    def __init__(self, high, low, close):
        self.high = high
        self.low = low
        self.close = close


class TestTopDownAnalyzer:
    """Tests for Top-Down Analysis."""
    
    def setup_method(self):
        self.tda = TopDownAnalyzer(
            short_sma_period=9,
            long_sma_period=20
        )
    
    def _create_bullish_candles(self, count=50):
        """Create candles showing bullish trend."""
        candles = []
        base = 1.0500
        for i in range(count):
            price = base + (i * 0.001)  # Trending up
            candles.append(MockCandle(
                high=price + 0.0010,
                low=price - 0.0005,
                close=price + 0.0005
            ))
        return candles
    
    def _create_bearish_candles(self, count=50):
        """Create candles showing bearish trend."""
        candles = []
        base = 1.1000
        for i in range(count):
            price = base - (i * 0.001)  # Trending down
            candles.append(MockCandle(
                high=price + 0.0010,
                low=price - 0.0005,
                close=price - 0.0005
            ))
        return candles
    
    def _create_neutral_candles(self, count=50):
        """Create ranging candles."""
        candles = []
        base = 1.0750
        for i in range(count):
            offset = 0.001 if i % 2 == 0 else -0.001
            price = base + offset
            candles.append(MockCandle(
                high=price + 0.0005,
                low=price - 0.0005,
                close=price
            ))
        return candles
    
    def test_perfect_bullish_alignment(self):
        """Test perfect alignment with all timeframes bullish."""
        weekly = self._create_bullish_candles(52)
        daily = self._create_bullish_candles(30)
        h4 = self._create_bullish_candles(50)
        h1 = self._create_bullish_candles(100)
        
        result = self.tda.run_tda(weekly, daily, h4, h1)
        
        assert result.alignment == AlignmentQuality.PERFECT
        assert result.overall_bias == TrendBias.BULLISH
        assert result.confidence >= 0.90
        assert result.can_trade == True
        assert result.bullish_count == 4
    
    def test_perfect_bearish_alignment(self):
        """Test perfect alignment with all timeframes bearish."""
        weekly = self._create_bearish_candles(52)
        daily = self._create_bearish_candles(30)
        h4 = self._create_bearish_candles(50)
        h1 = self._create_bearish_candles(100)
        
        result = self.tda.run_tda(weekly, daily, h4, h1)
        
        assert result.alignment == AlignmentQuality.PERFECT
        assert result.overall_bias == TrendBias.BEARISH
        assert result.can_trade == True
        assert result.bearish_count == 4
    
    def test_good_alignment(self):
        """Test good alignment with 3/4 timeframes agreeing."""
        weekly = self._create_bullish_candles(52)
        daily = self._create_bullish_candles(30)
        h4 = self._create_bullish_candles(50)
        h1 = self._create_neutral_candles(100)  # H1 neutral
        
        result = self.tda.run_tda(weekly, daily, h4, h1)
        
        assert result.alignment == AlignmentQuality.GOOD
        assert result.confidence >= 0.75
        assert result.can_trade == True
        assert result.bullish_count == 3
    
    def test_weak_alignment(self):
        """Test weak alignment with mixed signals."""
        weekly = self._create_bullish_candles(52)
        daily = self._create_bearish_candles(30)  # Mixed
        h4 = self._create_neutral_candles(50)
        h1 = self._create_bearish_candles(100)
        
        result = self.tda.run_tda(weekly, daily, h4, h1)
        
        # With mixed signals, should be WEAK
        assert result.alignment == AlignmentQuality.WEAK
        assert result.can_trade == False
    
    def test_single_timeframe_bias(self):
        """Test getting bias from single timeframe."""
        bullish_candles = self._create_bullish_candles(50)
        
        bias = self.tda.get_bias(bullish_candles)
        
        assert bias == TrendBias.BULLISH
    
    def test_supply_demand_zones(self):
        """Test supply/demand zone identification."""
        # Create candles with clear swing points
        candles = []
        base = 1.0700
        
        # Up move
        for i in range(10):
            candles.append(MockCandle(
                high=base + i * 0.002,
                low=base + i * 0.002 - 0.001,
                close=base + i * 0.002 - 0.0005
            ))
        
        # Down move
        for i in range(10):
            candles.append(MockCandle(
                high=base + 0.018 - i * 0.002,
                low=base + 0.018 - i * 0.002 - 0.001,
                close=base + 0.018 - i * 0.002 - 0.0005
            ))
        
        # Another up move to same high
        for i in range(10):
            candles.append(MockCandle(
                high=base + i * 0.002,
                low=base + i * 0.002 - 0.001,
                close=base + i * 0.002 - 0.0005
            ))
        
        zones = self.tda.find_supply_demand_zones(candles, "H1")
        
        # Should find some zones
        assert isinstance(zones, list)
    
    def test_price_at_zone(self):
        """Test checking if price is at a zone."""
        from src.top_down_analyzer import SupplyDemandZone
        
        zones = [
            SupplyDemandZone(
                zone_type="DEMAND",
                high=1.0750,
                low=1.0720,
                strength=3,
                timeframe="H4"
            ),
            SupplyDemandZone(
                zone_type="SUPPLY",
                high=1.0900,
                low=1.0880,
                strength=2,
                timeframe="H4"
            )
        ]
        
        # Price at demand zone
        at_zone, zone = self.tda.is_price_at_zone(1.0735, zones, "DEMAND")
        assert at_zone == True
        assert zone.zone_type == "DEMAND"
        
        # Price not at any zone
        at_zone, zone = self.tda.is_price_at_zone(1.0800, zones, "DEMAND")
        assert at_zone == False
        assert zone is None


class TestTDAWeeklyBiasPriority:
    """Test that weekly bias has priority."""
    
    def setup_method(self):
        self.tda = TopDownAnalyzer()
    
    def test_weekly_bias_dominates(self):
        """Weekly bias should dominate overall bias."""
        # Create strong weekly bullish but mixed lower timeframes
        weekly = [MockCandle(1.0 + i * 0.01, 0.99 + i * 0.01, 1.0 + i * 0.01) for i in range(52)]
        daily = [MockCandle(1.5 - i * 0.01, 1.49 - i * 0.01, 1.49 - i * 0.01) for i in range(30)]
        h4 = [MockCandle(1.3 - i * 0.005, 1.29 - i * 0.005, 1.29 - i * 0.005) for i in range(50)]
        h1 = [MockCandle(1.2 - i * 0.002, 1.19 - i * 0.002, 1.19 - i * 0.002) for i in range(100)]
        
        result = self.tda.run_tda(weekly, daily, h4, h1)
        
        # Even with weak alignment, overall bias should follow weekly
        assert result.overall_bias == TrendBias.BULLISH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
