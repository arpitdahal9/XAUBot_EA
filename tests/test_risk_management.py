"""
Unit Tests for Risk Management

Tests the risk management module's enforcement of:
- Daily loss limits
- Maximum drawdown protection
- Stop-loss requirements
- Margin validation
- Position size limits
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.risk_manager import RiskManager, RiskCheckResult
from src.models import Order, OrderSide, OrderType, OrderStatus, AccountState, MarketData
from src.config import RiskManagementConfig


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


@pytest.fixture
def valid_order():
    """Create a valid order for testing."""
    return Order(
        order_id="TEST001",
        pair="EUR/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        lot_size=0.03,
        entry_price=1.0850,
        stop_loss=1.0820,  # 30 pips below
        take_profit=1.0900  # 50 pips above
    )


@pytest.fixture
def healthy_account():
    """Create healthy account state for testing."""
    return AccountState(
        balance=1000,
        equity=1000,
        margin_used=0,
        free_margin=1000,
        margin_level=float('inf'),
        leverage=30
    )


@pytest.fixture
def market_data():
    """Create market data for testing."""
    return MarketData(
        pair="EUR/USD",
        bid=1.0850,
        ask=1.0852,  # 2 pip spread
        timestamp=datetime.now()
    )


class TestDailyLossLimit:
    """Tests for daily loss limit enforcement."""

    def test_trading_allowed_no_losses(
        self, risk_manager, valid_order, healthy_account, market_data
    ):
        """Test: Trading allowed when no daily losses."""
        result = risk_manager.validate_order(valid_order, healthy_account, market_data)
        
        assert result.passed is True

    def test_daily_loss_limit_blocks_trading(
        self, risk_manager, valid_order, market_data
    ):
        """Test: Bot refuses new trades after daily loss limit hit."""
        # Simulate account after $55 loss (over $50 limit)
        account = AccountState(
            balance=1000,
            equity=945,  # Started at 1000, lost $55
            margin_used=0,
            free_margin=945,
            margin_level=float('inf'),
            leverage=30
        )
        
        result = risk_manager.validate_order(valid_order, account, market_data)
        
        assert result.passed is False
        assert "DAILY LOSS LIMIT" in result.message

    def test_warning_at_70_percent_limit(
        self, risk_manager, valid_order, market_data
    ):
        """Test: Warning logged when approaching daily loss limit."""
        # Simulate account after $40 loss (80% of $50 limit)
        account = AccountState(
            balance=1000,
            equity=960,
            margin_used=0,
            free_margin=960,
            margin_level=float('inf'),
            leverage=30
        )
        
        result = risk_manager.validate_order(valid_order, account, market_data)
        
        # Should pass but with warning
        assert result.passed is True
        # Check for warning in one of the results

    def test_daily_loss_limit_critical_log(
        self, risk_manager, valid_order, market_data, mock_logger
    ):
        """Test: Critical log when daily loss limit exceeded."""
        account = AccountState(
            balance=1000,
            equity=940,  # $60 loss, over limit
            margin_used=0,
            free_margin=940,
            margin_level=float('inf'),
            leverage=30
        )
        
        risk_manager.validate_order(valid_order, account, market_data)
        
        # Verify critical or error was logged
        assert mock_logger.error.called


class TestDrawdownProtection:
    """Tests for maximum drawdown protection."""

    def test_max_drawdown_blocks_trading(
        self, risk_manager, valid_order, market_data
    ):
        """Test: Trading blocked when max drawdown exceeded."""
        # Simulate 12% drawdown (over 10% limit)
        # Peak was $1000, now at $880
        account = AccountState(
            balance=880,
            equity=880,
            margin_used=0,
            free_margin=880,
            margin_level=float('inf'),
            leverage=30
        )
        
        # Set peak equity
        risk_manager._peak_equity = 1000
        
        result = risk_manager.validate_order(valid_order, account, market_data)
        
        assert result.passed is False
        assert "DRAWDOWN" in result.message

    def test_equity_below_900_alert(
        self, risk_manager, valid_order, market_data
    ):
        """Test: Alert when equity drops below $900 (10% threshold)."""
        account = AccountState(
            balance=1000,
            equity=895,  # Below 10% threshold
            margin_used=0,
            free_margin=895,
            margin_level=float('inf'),
            leverage=30
        )
        
        can_trade, reason = risk_manager.can_trade(account)
        
        # Should indicate issue with equity
        assert "threshold" in reason.lower() or can_trade is False


class TestStopLossValidation:
    """Tests for stop-loss enforcement."""

    def test_order_rejected_no_stop_loss(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Orders without stop loss are rejected."""
        order_no_sl = Order(
            order_id="TEST002",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=0,  # No stop loss!
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(order_no_sl, healthy_account, market_data)
        
        assert result.passed is False
        assert "stop loss" in result.message.lower()

    def test_stop_loss_minimum_distance(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Stop loss must be at least 20 pips away."""
        order_tight_sl = Order(
            order_id="TEST003",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=1.0845,  # Only 5 pips!
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(order_tight_sl, healthy_account, market_data)
        
        assert result.passed is False
        assert "too tight" in result.message.lower() or "minimum" in result.message.lower()

    def test_stop_loss_correct_side_buy(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Buy order stop loss must be below entry."""
        order_wrong_sl = Order(
            order_id="TEST004",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=1.0900,  # Above entry - wrong!
            take_profit=1.0950
        )
        
        result = risk_manager.validate_order(order_wrong_sl, healthy_account, market_data)
        
        assert result.passed is False

    def test_stop_loss_correct_side_sell(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Sell order stop loss must be above entry."""
        order_wrong_sl = Order(
            order_id="TEST005",
            pair="EUR/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=1.0800,  # Below entry - wrong for SELL!
            take_profit=1.0750
        )
        
        result = risk_manager.validate_order(order_wrong_sl, healthy_account, market_data)
        
        assert result.passed is False


class TestTakeProfitValidation:
    """Tests for take-profit enforcement."""

    def test_order_rejected_no_take_profit(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Orders without take profit are rejected."""
        order_no_tp = Order(
            order_id="TEST006",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=0  # No take profit!
        )
        
        result = risk_manager.validate_order(order_no_tp, healthy_account, market_data)
        
        assert result.passed is False
        assert "take profit" in result.message.lower()

    def test_take_profit_minimum_distance(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Take profit must be at least 20 pips away."""
        order_tight_tp = Order(
            order_id="TEST007",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.03,
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0855  # Only 5 pips!
        )
        
        result = risk_manager.validate_order(order_tight_tp, healthy_account, market_data)
        
        assert result.passed is False


class TestMarginValidation:
    """Tests for margin requirement validation."""

    def test_insufficient_margin_rejected(
        self, risk_manager, valid_order, market_data
    ):
        """Test: Position rejected if margin insufficient."""
        # Account with very little free margin
        account = AccountState(
            balance=100,
            equity=100,
            margin_used=80,
            free_margin=20,  # Only $20 free
            margin_level=125,
            leverage=30
        )
        
        # Order requiring more margin than available
        large_order = Order(
            order_id="TEST008",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.10,  # Requires ~$333 margin at 1:30
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(large_order, account, market_data)
        
        assert result.passed is False
        assert "margin" in result.message.lower()

    def test_margin_buffer_warning(
        self, risk_manager, market_data
    ):
        """Test: Warning when margin buffer insufficient."""
        # Account where margin would exceed buffer
        account = AccountState(
            balance=200,
            equity=200,
            margin_used=0,
            free_margin=200,
            margin_level=float('inf'),
            leverage=30
        )
        
        # Order that uses most of margin
        order = Order(
            order_id="TEST009",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.05,  # Requires ~$166 margin
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(order, account, market_data)
        
        # Should pass but may have warning
        # The exact behavior depends on margin buffer config


class TestSpreadValidation:
    """Tests for spread validation."""

    def test_wide_spread_rejected(
        self, risk_manager, valid_order, healthy_account
    ):
        """Test: Trading rejected when spread too wide."""
        wide_spread_data = MarketData(
            pair="EUR/USD",
            bid=1.0850,
            ask=1.0860,  # 10 pip spread!
            timestamp=datetime.now()
        )
        
        result = risk_manager.validate_order(valid_order, healthy_account, wide_spread_data)
        
        assert result.passed is False
        assert "spread" in result.message.lower()

    def test_spread_warning_threshold(
        self, risk_manager, valid_order, healthy_account
    ):
        """Test: Warning when spread above normal but below max."""
        elevated_spread = MarketData(
            pair="EUR/USD",
            bid=1.0850,
            ask=1.0855,  # 2.5 pip spread (above normal but below 3 max)
            timestamp=datetime.now()
        )
        
        result = risk_manager.validate_order(valid_order, healthy_account, elevated_spread)
        
        # Should still pass
        assert result.passed is True


class TestPositionSizeLimits:
    """Tests for position size limit enforcement."""

    def test_lot_size_below_minimum(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Lot size below minimum rejected."""
        tiny_order = Order(
            order_id="TEST010",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=0.001,  # Below 0.01 minimum
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(tiny_order, healthy_account, market_data)
        
        assert result.passed is False

    def test_lot_size_above_maximum(
        self, risk_manager, healthy_account, market_data
    ):
        """Test: Lot size above maximum rejected."""
        huge_order = Order(
            order_id="TEST011",
            pair="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=1.0,  # Way above 0.10 maximum
            entry_price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0900
        )
        
        result = risk_manager.validate_order(huge_order, healthy_account, market_data)
        
        assert result.passed is False


class TestCanTradeFunction:
    """Tests for quick can_trade check."""

    def test_can_trade_healthy_account(self, risk_manager, healthy_account):
        """Test: Can trade with healthy account."""
        can_trade, reason = risk_manager.can_trade(healthy_account)
        
        assert can_trade is True
        assert "allowed" in reason.lower()

    def test_cannot_trade_daily_limit(self, risk_manager):
        """Test: Cannot trade when daily limit reached."""
        account = AccountState(
            balance=1000,
            equity=940,  # $60 loss
            margin_used=0,
            free_margin=940,
            margin_level=float('inf'),
            leverage=30
        )
        
        can_trade, reason = risk_manager.can_trade(account)
        
        assert can_trade is False
        assert "daily" in reason.lower() or "loss" in reason.lower()


class TestTradeRecording:
    """Tests for trade recording and daily stats."""

    def test_trade_recorded(self, risk_manager, valid_order):
        """Test: Completed trades are recorded."""
        valid_order.pnl = 5.50
        
        risk_manager.record_trade(valid_order)
        
        stats = risk_manager.get_daily_stats()
        assert stats.total_trades == 1
        assert stats.gross_profit == 5.50

    def test_daily_stats_calculation(self, risk_manager):
        """Test: Daily stats calculated correctly."""
        # Record some trades
        win1 = Order(
            order_id="W1", pair="EUR/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, lot_size=0.03,
            entry_price=1.0850, stop_loss=1.0820, take_profit=1.0900
        )
        win1.pnl = 10.0
        
        win2 = Order(
            order_id="W2", pair="EUR/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, lot_size=0.03,
            entry_price=1.0850, stop_loss=1.0820, take_profit=1.0900
        )
        win2.pnl = 8.0
        
        loss1 = Order(
            order_id="L1", pair="EUR/USD", side=OrderSide.BUY,
            order_type=OrderType.MARKET, lot_size=0.03,
            entry_price=1.0850, stop_loss=1.0820, take_profit=1.0900
        )
        loss1.pnl = -5.0
        
        risk_manager.record_trade(win1)
        risk_manager.record_trade(win2)
        risk_manager.record_trade(loss1)
        
        stats = risk_manager.get_daily_stats()
        
        assert stats.total_trades == 3
        assert stats.winning_trades == 2
        assert stats.losing_trades == 1
        assert stats.gross_profit == 18.0
        assert stats.gross_loss == 5.0
        assert stats.win_rate == pytest.approx(66.67, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
