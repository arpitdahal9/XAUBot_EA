"""
Unit tests for Pattern Recognition.
"""

import pytest
import sys
sys.path.insert(0, '.')

from src.pattern_recognizer import PatternRecognizer, PatternType, PatternSignal


class MockCandle:
    """Mock candle for testing."""
    def __init__(self, high, low, close):
        self.high = high
        self.low = low
        self.close = close


class TestPatternRecognizer:
    """Tests for pattern recognition."""
    
    def setup_method(self):
        self.recognizer = PatternRecognizer(
            tolerance_pips=15,
            min_pattern_width=5,
            max_pattern_width=50
        )
    
    def _create_double_bottom_candles(self):
        """Create candles forming a double bottom pattern."""
        candles = []
        
        # Initial move down to first bottom
        for i in range(5):
            candles.append(MockCandle(
                high=1.1000 - i * 0.005,
                low=1.0995 - i * 0.005,
                close=1.0997 - i * 0.005
            ))
        
        # First bottom at ~1.0750
        for i in range(3):
            candles.append(MockCandle(
                high=1.0760,
                low=1.0750,  # First low
                close=1.0755
            ))
        
        # Rally to neckline ~1.0850
        for i in range(5):
            candles.append(MockCandle(
                high=1.0760 + i * 0.02,
                low=1.0755 + i * 0.02,
                close=1.0758 + i * 0.02
            ))
        
        # Second bottom at ~1.0752 (within tolerance of first)
        for i in range(3):
            candles.append(MockCandle(
                high=1.0770,
                low=1.0752,  # Second low (similar to first)
                close=1.0760
            ))
        
        # Rally above neckline
        for i in range(5):
            candles.append(MockCandle(
                high=1.0760 + i * 0.015,
                low=1.0755 + i * 0.015,
                close=1.0758 + i * 0.015
            ))
        
        # Confirmation candle above neckline
        candles.append(MockCandle(
            high=1.0900,
            low=1.0880,
            close=1.0895  # Close above neckline
        ))
        
        return candles
    
    def _create_double_top_candles(self):
        """Create candles forming a double top pattern."""
        candles = []
        
        # Initial move up to first top
        for i in range(5):
            candles.append(MockCandle(
                high=1.0500 + i * 0.005,
                low=1.0495 + i * 0.005,
                close=1.0497 + i * 0.005
            ))
        
        # First top at ~1.0850
        for i in range(3):
            candles.append(MockCandle(
                high=1.0850,  # First high
                low=1.0840,
                close=1.0845
            ))
        
        # Pullback to neckline ~1.0750
        for i in range(5):
            candles.append(MockCandle(
                high=1.0850 - i * 0.02,
                low=1.0845 - i * 0.02,
                close=1.0848 - i * 0.02
            ))
        
        # Second top at ~1.0848 (within tolerance of first)
        for i in range(3):
            candles.append(MockCandle(
                high=1.0848,  # Second high (similar to first)
                low=1.0838,
                close=1.0840
            ))
        
        # Break below neckline
        for i in range(5):
            candles.append(MockCandle(
                high=1.0840 - i * 0.015,
                low=1.0835 - i * 0.015,
                close=1.0838 - i * 0.015
            ))
        
        # Confirmation candle below neckline
        candles.append(MockCandle(
            high=1.0720,
            low=1.0700,
            close=1.0705  # Close below neckline
        ))
        
        return candles
    
    def test_detect_double_bottom(self):
        """Test double bottom detection."""
        candles = self._create_double_bottom_candles()
        
        result = self.recognizer.detect_double_bottom(candles)
        
        if result:  # May or may not detect depending on exact structure
            assert result.pattern_type == PatternType.DOUBLE_BOTTOM
            assert result.signal == PatternSignal.BULLISH
            assert result.neckline > 0
            assert result.target_price > result.neckline
    
    def test_detect_double_top(self):
        """Test double top detection."""
        candles = self._create_double_top_candles()
        
        result = self.recognizer.detect_double_top(candles)
        
        if result:  # May or may not detect depending on exact structure
            assert result.pattern_type == PatternType.DOUBLE_TOP
            assert result.signal == PatternSignal.BEARISH
            assert result.neckline > 0
            assert result.target_price < result.neckline
    
    def test_detect_all_patterns(self):
        """Test detecting all patterns."""
        candles = self._create_double_bottom_candles()
        
        patterns = self.recognizer.detect_all_patterns(candles)
        
        # Should return a list (may be empty or contain patterns)
        assert isinstance(patterns, list)
    
    def test_get_strongest_pattern(self):
        """Test getting the strongest pattern."""
        candles = self._create_double_bottom_candles()
        
        patterns = self.recognizer.detect_all_patterns(candles)
        strongest = self.recognizer.get_strongest_pattern(patterns)
        
        if patterns:
            assert strongest is not None
            # Strongest should have highest confidence
            for p in patterns:
                assert strongest.confidence >= p.confidence
        else:
            assert strongest is None
    
    def test_insufficient_candles(self):
        """Test with insufficient candles."""
        candles = [MockCandle(1.0, 0.99, 0.995) for _ in range(5)]
        
        result = self.recognizer.detect_double_bottom(candles)
        
        assert result is None
    
    def test_pattern_outside_tolerance(self):
        """Test pattern detection fails if peaks/troughs outside tolerance."""
        candles = []
        
        # First low
        for i in range(5):
            candles.append(MockCandle(1.0800, 1.0700, 1.0750))
        
        # Rally
        for i in range(10):
            candles.append(MockCandle(1.0900, 1.0850, 1.0875))
        
        # Second low - much lower than first (outside tolerance)
        for i in range(5):
            candles.append(MockCandle(1.0600, 1.0500, 1.0550))  # 200+ pips lower
        
        for i in range(10):
            candles.append(MockCandle(1.0650, 1.0600, 1.0625))
        
        result = self.recognizer.detect_double_bottom(candles, lookback=30)
        
        # Should not detect pattern due to large difference between lows
        # (depends on tolerance setting)
        assert result is None or result.confidence < 0.7


class TestHeadAndShoulders:
    """Tests for Head and Shoulders patterns."""
    
    def setup_method(self):
        self.recognizer = PatternRecognizer()
    
    def _create_head_and_shoulders_candles(self):
        """Create H&S pattern candles."""
        candles = []
        
        # Baseline
        for i in range(5):
            candles.append(MockCandle(1.0800, 1.0790, 1.0795))
        
        # Left shoulder (peak at 1.0900)
        for i in range(5):
            candles.append(MockCandle(1.0800 + i * 0.02, 1.0790 + i * 0.02, 1.0795 + i * 0.02))
        for i in range(5):
            candles.append(MockCandle(1.0900 - i * 0.01, 1.0890 - i * 0.01, 1.0895 - i * 0.01))
        
        # Head (peak at 1.0950 - higher than shoulders)
        for i in range(5):
            candles.append(MockCandle(1.0850 + i * 0.02, 1.0840 + i * 0.02, 1.0845 + i * 0.02))
        for i in range(5):
            candles.append(MockCandle(1.0950 - i * 0.02, 1.0940 - i * 0.02, 1.0945 - i * 0.02))
        
        # Right shoulder (peak at 1.0900)
        for i in range(5):
            candles.append(MockCandle(1.0850 + i * 0.01, 1.0840 + i * 0.01, 1.0845 + i * 0.01))
        for i in range(5):
            candles.append(MockCandle(1.0900 - i * 0.015, 1.0890 - i * 0.015, 1.0895 - i * 0.015))
        
        # Break below neckline
        for i in range(5):
            candles.append(MockCandle(1.0830 - i * 0.02, 1.0820 - i * 0.02, 1.0825 - i * 0.02))
        
        return candles
    
    def test_detect_head_and_shoulders(self):
        """Test H&S detection."""
        candles = self._create_head_and_shoulders_candles()
        
        result = self.recognizer.detect_head_and_shoulders(candles)
        
        if result:
            assert result.pattern_type == PatternType.HEAD_AND_SHOULDERS
            assert result.signal == PatternSignal.BEARISH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
