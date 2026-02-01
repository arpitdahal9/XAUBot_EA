"""
Unit Tests for Position Sizing

Tests the risk management module's position sizing calculations
to ensure proper lot sizes based on account risk parameters.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk_manager import RiskManager, RiskCheckResult, PositionSizeResult
from src.models import Order, OrderSide, OrderType, OrderStatus, AccountState, MarketData
from src.config import RiskManagementConfig, BotConfig, TradingConfig
from datetime import datetime


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def risk_config():
    """Create risk management config for testing."""
    return RiskManagementConfig(
        account_balance=1000,
        risk_percent_per_trade=1.0,
        daily_loss_limit=50,
        max_drawdown_percent=10,
        max_spread_allowed=3.0,
        min_stop_loss_pips=20,
        min_take_profit_pips=20,
        margin_buffer_percent=20
    )


@pytest.fixture
def risk_manager(risk_config, mock_logger):
    """Create risk manager for testing."""
    return RiskManager(config=risk_config, logger=mock_logger)


class TestPositionSizing:
    """Tests for position size calculation."""

    def test_position_size_1percent_risk_30pip_sl(self, risk_manager):
        """
        Test: 1% risk with 30 pip stop = correct lot size.
        
        Expected calculation:
        - Account: $1,000
        - Risk: 1% = $10
        - Stop Loss: 30 pips
        - Pip Value: $10 per lot for EUR/USD
        - Lot Size = $10 / (30 × $10) = 0.033 → rounds to 0.03
        """
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        assert result.lot_size == 0.03  # $10 / (30 × $10) = 0.033 → 0.03
        assert result.risk_amount == 10.0

    def test_position_size_2percent_risk(self, risk_manager):
        """Test: 2% risk calculates larger position."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=2.0
        )
        
        assert result.is_valid is True
        assert result.lot_size == 0.06  # $20 / (30 × $10) = 0.066 → 0.06
        assert result.risk_amount == 20.0

    def test_position_size_tight_stop_loss(self, risk_manager):
        """Test: Tighter stop loss results in larger position."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=20,
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        assert result.lot_size == 0.05  # $10 / (20 × $10) = 0.05

    def test_position_size_wide_stop_loss(self, risk_manager):
        """Test: Wider stop loss results in smaller position."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=50,
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        assert result.lot_size == 0.02  # $10 / (50 × $10) = 0.02

    def test_position_size_minimum_lot(self, risk_manager):
        """Test: Very wide stop doesn't go below minimum lot."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=200,  # Very wide stop
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        assert result.lot_size >= 0.01  # Minimum lot size

    def test_position_size_invalid_stop_loss(self, risk_manager):
        """Test: Zero or negative stop loss returns invalid."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=0,
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is False
        assert "Invalid stop loss" in result.message

    def test_position_size_jpy_pair(self, risk_manager):
        """Test: JPY pairs use different pip value."""
        result = risk_manager.calculate_position_size(
            pair="USD/JPY",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        # JPY pairs have ~$6.67 pip value vs $10 for others
        assert result.lot_size > 0

    def test_position_size_small_account(self, risk_manager):
        """Test: Small account still gets valid lot size."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=500,  # Smaller account
            risk_percent=1.0
        )
        
        assert result.is_valid is True
        assert result.lot_size == 0.01  # $5 / (30 × $10) = 0.016 → 0.01
        assert result.risk_amount == 5.0


class TestPositionSizeLogging:
    """Tests for position sizing logging."""

    def test_calculation_logged_at_debug(self, risk_manager, mock_logger):
        """Test: All calculation steps logged at DEBUG level."""
        risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=1.0
        )
        
        # Verify debug logs were called
        assert mock_logger.debug.called
        
        # Check for expected debug messages
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("Account Balance" in str(call) for call in debug_calls)
        assert any("Risk Percentage" in str(call) for call in debug_calls)
        assert any("Stop Loss Pips" in str(call) for call in debug_calls)

    def test_result_logged_at_info(self, risk_manager, mock_logger):
        """Test: Final result logged at INFO level."""
        risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=1.0
        )
        
        # Verify info log was called with position size
        assert mock_logger.info.called


class TestMarginCalculation:
    """Tests for margin requirement calculations."""

    def test_margin_calculated_correctly(self, risk_manager):
        """Test: Margin required is calculated correctly for 1:30 leverage."""
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30,
            account_balance=1000,
            risk_percent=1.0
        )
        
        # For 0.03 lots with 1:30 leverage:
        # Margin = (0.03 × 100,000) / 30 = $100
        expected_margin = (result.lot_size * 100000) / 30
        assert abs(result.margin_required - expected_margin) < 0.01

    def test_larger_position_needs_more_margin(self, risk_manager):
        """Test: Larger positions require more margin."""
        small_result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=50,
            account_balance=1000,
            risk_percent=1.0
        )
        
        large_result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=20,
            account_balance=1000,
            risk_percent=1.0
        )
        
        # Tighter stop = larger position = more margin
        assert large_result.margin_required > small_result.margin_required


class TestRiskPercentageEdgeCases:
    """Tests for edge cases in risk percentage."""

    def test_half_percent_risk(self, risk_config, mock_logger):
        """Test: 0.5% risk is allowed and calculates correctly."""
        config = RiskManagementConfig(
            account_balance=1000,
            risk_percent_per_trade=0.5,  # Conservative risk
            daily_loss_limit=50,
            max_drawdown_percent=10,
            max_spread_allowed=3.0,
            min_stop_loss_pips=20,
            min_take_profit_pips=20,
            margin_buffer_percent=20
        )
        risk_manager = RiskManager(config=config, logger=mock_logger)
        
        result = risk_manager.calculate_position_size(
            pair="EUR/USD",
            stop_loss_pips=30
        )
        
        assert result.is_valid is True
        assert result.risk_amount == 5.0  # 0.5% of $1000

    def test_max_risk_percentage(self, risk_config, mock_logger):
        """Test: 2% max risk is enforced by config validation."""
        # This should be caught by Pydantic validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            RiskManagementConfig(
                account_balance=1000,
                risk_percent_per_trade=5.0,  # Too high
                daily_loss_limit=50,
                max_drawdown_percent=10,
                max_spread_allowed=3.0,
                min_stop_loss_pips=20,
                min_take_profit_pips=20,
                margin_buffer_percent=20
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
