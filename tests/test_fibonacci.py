"""
Unit tests for Fibonacci Calculator.
"""

import pytest
import sys
sys.path.insert(0, '.')

from src.fibonacci import FibonacciCalculator, FibDirection


class TestFibonacciCalculator:
    """Tests for FibonacciCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calc = FibonacciCalculator()
    
    def test_calculate_from_high_low_bullish(self):
        """Test Fibonacci calculation for bullish setup."""
        high = 1.1000
        low = 1.0500
        
        levels = self.calc.calculate_from_high_low(high, low, FibDirection.BULLISH)
        
        # Verify key levels
        assert levels['0%'] == high
        assert levels['100%'] == low
        assert levels['50%'] == pytest.approx(1.0750, abs=0.0001)
        assert levels['61.8%'] == pytest.approx(1.0691, abs=0.0001)
        assert levels['88.6%'] == pytest.approx(1.0557, abs=0.0001)
    
    def test_calculate_from_high_low_bearish(self):
        """Test Fibonacci calculation for bearish setup."""
        high = 1.1000
        low = 1.0500
        
        levels = self.calc.calculate_from_high_low(high, low, FibDirection.BEARISH)
        
        # Bearish: retracement goes up from low
        assert levels['0%'] == low
        assert levels['100%'] == high
        assert levels['50%'] == pytest.approx(1.0750, abs=0.0001)
        assert levels['88.6%'] == pytest.approx(1.0943, abs=0.0001)
    
    def test_calculate_targets(self):
        """Test take profit target calculation."""
        entry = 1.0557
        r_value = 0.0057  # Entry to SL distance
        
        targets = self.calc.calculate_targets(entry, r_value, FibDirection.BULLISH)
        
        # TP1 = +2R, TP2 = +3R, TP3 = +5R
        assert targets['TP1'] == pytest.approx(entry + (r_value * 2), abs=0.0001)
        assert targets['TP2'] == pytest.approx(entry + (r_value * 3), abs=0.0001)
        assert targets['TP3'] == pytest.approx(entry + (r_value * 5), abs=0.0001)
    
    def test_quarter_points(self):
        """Test Quarter Points calculation."""
        high = 1.1000
        low = 1.0500
        
        qp = self.calc.calculate_quarter_points(high, low)
        
        assert qp.high == high
        assert qp.low == low
        assert qp.mid == pytest.approx(1.0750, abs=0.0001)  # 50%
        assert qp.q1 == pytest.approx(1.0875, abs=0.0001)   # 25%
        assert qp.q3 == pytest.approx(1.0625, abs=0.0001)   # 75%
    
    def test_is_price_at_entry_zone(self):
        """Test entry zone detection."""
        # Create mock candles
        class MockCandle:
            def __init__(self, h, l, c):
                self.high = h
                self.low = l
                self.close = c
        
        # Simulate a swing from 1.10 to 1.05
        candles = [
            MockCandle(1.1000, 1.0950, 1.0980),  # High
            MockCandle(1.0980, 1.0900, 1.0850),
            MockCandle(1.0850, 1.0700, 1.0600),
            MockCandle(1.0600, 1.0500, 1.0500),  # Low
        ]
        
        # Create Fib levels manually for testing
        levels = self.calc._calculate_from_swing(1.1000, 1.0500, FibDirection.BULLISH)
        
        # Test at 88.6% level (should be ~1.0557)
        in_zone, distance = self.calc.is_price_at_entry_zone(1.0557, levels, tolerance_pips=20)
        assert in_zone == True
        assert distance < 5
        
        # Test far from entry zone
        in_zone, distance = self.calc.is_price_at_entry_zone(1.0800, levels, tolerance_pips=20)
        assert in_zone == False
    
    def test_setup_quality(self):
        """Test setup quality evaluation."""
        levels = self.calc._calculate_from_swing(1.1000, 1.0500, FibDirection.BULLISH)
        
        # Perfect entry at 88.6%
        quality = self.calc.get_setup_quality(levels, 1.0557, tolerance_pips=20)
        assert quality['in_entry_zone'] == True
        assert quality['proximity_score'] >= 0.9
        assert quality['quality'] in ['EXCELLENT', 'GOOD']


class TestFibonacciEdgeCases:
    """Edge case tests for Fibonacci."""
    
    def setup_method(self):
        self.calc = FibonacciCalculator()
    
    def test_small_range(self):
        """Test with very small price range."""
        high = 1.1005
        low = 1.1000
        
        levels = self.calc.calculate_from_high_low(high, low, FibDirection.BULLISH)
        
        assert levels['0%'] == high
        assert levels['100%'] == low
        # Small range should still calculate correctly
        assert levels['50%'] == pytest.approx(1.10025, abs=0.00001)
    
    def test_jpy_pair(self):
        """Test with JPY pair prices."""
        high = 150.00
        low = 145.00
        
        levels = self.calc.calculate_from_high_low(high, low, FibDirection.BULLISH)
        
        assert levels['0%'] == high
        assert levels['100%'] == low
        assert levels['88.6%'] == pytest.approx(145.57, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
