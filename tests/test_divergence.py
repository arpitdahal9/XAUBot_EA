"""
Unit tests for RSI and Divergence Detection.
"""

import pytest
import sys
sys.path.insert(0, '.')

from src.divergence import RSICalculator, DivergenceDetector, DivergenceType


class TestRSICalculator:
    """Tests for RSI calculation."""
    
    def setup_method(self):
        self.rsi = RSICalculator(period=14)
    
    def test_rsi_calculation_basic(self):
        """Test basic RSI calculation."""
        # Generate trending up prices
        prices = [100 + i * 0.5 for i in range(20)]
        
        result = self.rsi.calculate(prices)
        
        assert result is not None
        assert result.current_value > 50  # Uptrend should have RSI > 50
        assert len(result.values) == len(prices)
    
    def test_rsi_overbought(self):
        """Test RSI overbought detection."""
        # Strong uptrend prices
        prices = [100 + i * 2 for i in range(20)]
        
        result = self.rsi.calculate(prices)
        
        assert result.is_overbought == True
        assert result.zone == "OVERBOUGHT"
    
    def test_rsi_oversold(self):
        """Test RSI oversold detection."""
        # Strong downtrend prices
        prices = [100 - i * 2 for i in range(20)]
        
        result = self.rsi.calculate(prices)
        
        assert result.is_oversold == True
        assert result.zone == "OVERSOLD"
    
    def test_rsi_neutral(self):
        """Test RSI neutral zone."""
        # Ranging prices
        prices = [100 + (0.5 if i % 2 == 0 else -0.5) for i in range(20)]
        
        result = self.rsi.calculate(prices)
        
        assert 30 <= result.current_value <= 70
        assert result.zone == "NEUTRAL"
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = [100, 101, 102]  # Only 3 prices
        
        result = self.rsi.calculate(prices)
        
        assert result is None


class TestDivergenceDetector:
    """Tests for divergence detection."""
    
    def setup_method(self):
        self.detector = DivergenceDetector(rsi_period=14, lookback=20)
    
    def test_bullish_divergence_simple(self):
        """Test simple bullish divergence detection."""
        price_lows = [100, 95, 90]  # Lower lows
        rsi_values = [30, 32, 35]   # Higher lows
        
        result = self.detector.detect_bullish_simple(price_lows, rsi_values)
        
        assert result is not None
        assert result['type'] == 'BULLISH'
    
    def test_bearish_divergence_simple(self):
        """Test simple bearish divergence detection."""
        price_highs = [100, 105, 110]  # Higher highs
        rsi_values = [70, 68, 65]      # Lower highs
        
        result = self.detector.detect_bearish_simple(price_highs, rsi_values)
        
        assert result is not None
        assert result['type'] == 'BEARISH'
    
    def test_no_divergence(self):
        """Test when no divergence exists."""
        # Price and RSI moving in same direction
        price_lows = [100, 95, 90]
        rsi_values = [50, 45, 40]  # Both making lower lows
        
        result = self.detector.detect_bullish_simple(price_lows, rsi_values)
        
        assert result is None
    
    def test_divergence_with_candles(self):
        """Test divergence detection with candle data."""
        class MockCandle:
            def __init__(self, h, l, c):
                self.high = h
                self.low = l
                self.close = c
        
        # Create candles with price making lower lows
        candles = []
        base_price = 1.1000
        
        # First swing low
        for i in range(10):
            candles.append(MockCandle(
                base_price - i * 0.001,
                base_price - i * 0.001 - 0.0005,
                base_price - i * 0.001 - 0.0003
            ))
        
        # Rally
        for i in range(5):
            candles.append(MockCandle(
                base_price - 0.01 + i * 0.002,
                base_price - 0.01 + i * 0.002 - 0.0005,
                base_price - 0.01 + i * 0.002 - 0.0003
            ))
        
        # Second swing low (lower than first)
        for i in range(10):
            candles.append(MockCandle(
                base_price - 0.015 - i * 0.001,
                base_price - 0.015 - i * 0.001 - 0.0005,
                base_price - 0.015 - i * 0.001 - 0.0003
            ))
        
        # May or may not detect divergence depending on RSI
        # This tests that the detector doesn't crash
        result = self.detector.detect(candles)
        
        # Just verify it returns a result or None without error
        assert result is None or result.type in [
            DivergenceType.BULLISH,
            DivergenceType.BEARISH,
            DivergenceType.HIDDEN_BULLISH,
            DivergenceType.HIDDEN_BEARISH
        ]


class TestDivergenceStrength:
    """Test divergence strength calculation."""
    
    def setup_method(self):
        self.detector = DivergenceDetector()
    
    def test_strong_divergence(self):
        """Test detection of strong divergence."""
        # Large difference between price and RSI directions
        price_lows = [100, 90]  # -10% move
        rsi_values = [20, 35]   # +15 RSI points (strong)
        
        result = self.detector.detect_bullish_simple(price_lows, rsi_values)
        
        assert result is not None
        # Strong divergence should be confirmed
        assert result['strength'] == 'CONFIRMED'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
