"""
Comprehensive Logging Framework for Trading Bot

Provides structured logging with multiple levels, file rotation,
colored console output, and trade-specific logging utilities.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


# Custom log level for TRADE events (between INFO and WARNING)
TRADE_LEVEL = 25
logging.addLevelName(TRADE_LEVEL, "TRADE")


@dataclass
class TradeLogEntry:
    """Structured trade log entry for consistent formatting."""
    order_id: str
    action: str  # OPEN, CLOSE, MODIFY
    trade_type: str  # BUY, SELL
    pair: str
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    pnl: Optional[float] = None
    slippage_pips: Optional[float] = None
    module: str = "TRADE"
    extra: Dict[str, Any] = field(default_factory=dict)

    def format_message(self) -> str:
        """Format trade entry as log message."""
        if self.action == "OPEN":
            msg = (
                f"ORDER OPENED: {self.trade_type} {self.lot_size} {self.pair} "
                f"at {self.entry_price:.5f}, SL: {self.stop_loss:.5f}, TP: {self.take_profit:.5f}"
            )
        elif self.action == "CLOSE":
            pnl_str = f"${self.pnl:+.2f}" if self.pnl is not None else "N/A"
            msg = (
                f"ORDER CLOSED: {self.trade_type} {self.lot_size} {self.pair} "
                f"at {self.entry_price:.5f}, P&L: {pnl_str}"
            )
        elif self.action == "MODIFY":
            msg = (
                f"ORDER MODIFIED: #{self.order_id} {self.pair} "
                f"New SL: {self.stop_loss:.5f}, New TP: {self.take_profit:.5f}"
            )
        else:
            msg = f"ORDER {self.action}: #{self.order_id} {self.pair}"
        
        if self.slippage_pips is not None and self.slippage_pips > 0:
            msg += f" (Slippage: {self.slippage_pips:.1f} pips)"
        
        return msg


class TradingBotLogger(logging.Logger):
    """
    Extended logger with trading-specific methods.
    
    Provides convenience methods for logging trades, account state,
    and market data with consistent formatting.
    """

    def trade(self, entry: TradeLogEntry) -> None:
        """Log a trade event with structured data."""
        if self.isEnabledFor(TRADE_LEVEL):
            msg = f"[{entry.module}] Order #{entry.order_id} - {entry.format_message()}"
            self.log(TRADE_LEVEL, msg)

    def account_state(
        self,
        balance: float,
        equity: float,
        margin_used: float,
        free_margin: float,
        daily_pnl: float,
        drawdown_pct: float
    ) -> None:
        """Log current account state."""
        self.info(
            f"[ACCOUNT] Balance: ${balance:.2f} | Equity: ${equity:.2f} | "
            f"Margin Used: ${margin_used:.2f} | Free: ${free_margin:.2f} | "
            f"Daily P&L: ${daily_pnl:+.2f} | Drawdown: {drawdown_pct:.1f}%"
        )

    def market_data(
        self,
        pair: str,
        bid: float,
        ask: float,
        spread_pips: float
    ) -> None:
        """Log market data update."""
        self.debug(
            f"[MARKET] {pair} Bid: {bid:.5f} | Ask: {ask:.5f} | Spread: {spread_pips:.1f} pips"
        )

    def signal(
        self,
        signal_type: str,
        module: str,
        pair: str,
        price: float,
        details: str = ""
    ) -> None:
        """Log trading signal detection."""
        self.info(f"[{module}] {signal_type} SIGNAL: {pair} at {price:.5f} - {details}")

    def risk_warning(self, message: str, current_value: float, limit_value: float) -> None:
        """Log risk management warning."""
        self.warning(
            f"[RISK] {message} - Current: {current_value:.2f}, Limit: {limit_value:.2f}"
        )

    def api_status(self, status: str, latency_ms: Optional[float] = None) -> None:
        """Log API connection status."""
        latency_str = f" (Latency: {latency_ms:.0f}ms)" if latency_ms else ""
        self.debug(f"[API] Connection Status: {status}{latency_str}")


class LogFormatter(logging.Formatter):
    """Custom formatter with consistent timestamp and level formatting."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with milliseconds precision."""
        record.asctime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return super().format(record)


def setup_logger(
    name: str = "TradingBot",
    log_level: str = "INFO",
    output_dir: str = "./logs",
    output_file: str = "trading_bot.log",
    max_file_size_mb: int = 50,
    backup_count: int = 5,
    console_output: bool = True,
    colored_output: bool = True
) -> TradingBotLogger:
    """
    Set up and configure the trading bot logger.
    
    Args:
        name: Logger name.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        output_dir: Directory for log files.
        output_file: Log file name.
        max_file_size_mb: Maximum log file size before rotation.
        backup_count: Number of backup log files to keep.
        console_output: Enable console output.
        colored_output: Enable colored console output (requires colorlog).
        
    Returns:
        Configured TradingBotLogger instance.
    """
    # Register custom logger class
    logging.setLoggerClass(TradingBotLogger)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Log format
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    
    # Create logs directory
    log_path = Path(output_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path / output_file,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # File captures all levels
    file_handler.setFormatter(LogFormatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        if colored_output and COLORLOG_AVAILABLE:
            # Colored console output
            color_format = (
                "%(log_color)s[%(levelname)s]%(reset)s "
                "%(cyan)s%(asctime)s%(reset)s - %(message)s"
            )
            console_handler.setFormatter(colorlog.ColoredFormatter(
                color_format,
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'white',
                    'INFO': 'green',
                    'TRADE': 'blue',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
        else:
            console_handler.setFormatter(LogFormatter(log_format))
        
        logger.addHandler(console_handler)
    
    # Log initialization
    logger.info("=" * 60)
    logger.info("Trading Bot Logger Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_path / output_file}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str = "TradingBot") -> TradingBotLogger:
    """
    Get existing logger or create new one with default settings.
    
    Args:
        name: Logger name.
        
    Returns:
        TradingBotLogger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Example logging functions for debugging (as specified)
def log_position_calculation(
    logger: TradingBotLogger,
    account_balance: float,
    risk_percentage: float,
    stop_loss_pips: float,
    pip_value: float,
    calculated_lot: float,
    rounded_lot: float
) -> None:
    """Log detailed position size calculation for debugging."""
    logger.debug("Position size calculation initiated")
    logger.debug(f"  Account Balance: ${account_balance:.2f}")
    logger.debug(f"  Risk Percentage: {risk_percentage}%")
    logger.debug(f"  Stop Loss Pips: {stop_loss_pips}")
    logger.debug(f"  Pip Value: ${pip_value:.4f}")
    logger.debug(f"  Calculated Lot Size: {calculated_lot:.4f}")
    logger.debug(f"  Rounded Lot Size: {rounded_lot:.2f}")


def log_order_fill_validation(
    logger: TradingBotLogger,
    order_id: str,
    expected_price: float,
    actual_price: float,
    stop_loss: float,
    take_profit: float
) -> bool:
    """
    Validate and log order fill details.
    
    Returns:
        True if validation passed, False otherwise.
    """
    slippage = abs(actual_price - expected_price) * 10000  # Convert to pips
    
    logger.info(f"ORDER FILLED: Order #{order_id}")
    logger.info(f"  Expected Price: {expected_price:.5f}")
    logger.info(f"  Actual Price: {actual_price:.5f}")
    logger.info(f"  Slippage: {slippage:.1f} pips")
    logger.info(f"  Stop Loss: {stop_loss:.5f}")
    logger.info(f"  Take Profit: {take_profit:.5f}")
    
    # Validation checks
    if slippage > 5:
        logger.warning(f"High slippage detected: {slippage:.1f} pips")
        return False
    
    if stop_loss == 0 or take_profit == 0:
        logger.error(f"CRITICAL: Order missing SL or TP. SL: {stop_loss}, TP: {take_profit}")
        return False
    
    logger.debug("Order fill validation passed")
    return True


def log_daily_summary(
    logger: TradingBotLogger,
    trades_count: int,
    wins: int,
    losses: int,
    total_pnl: float,
    drawdown_pct: float
) -> None:
    """Log end-of-day trading summary."""
    win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
    
    logger.info("=" * 60)
    logger.info("DAILY SUMMARY")
    logger.info(f"  Total Trades: {trades_count}")
    logger.info(f"  Wins: {wins} | Losses: {losses}")
    logger.info(f"  Win Rate: {win_rate:.1f}%")
    logger.info(f"  Total P&L: ${total_pnl:+.2f}")
    logger.info(f"  Max Drawdown: {drawdown_pct:.1f}%")
    logger.info("=" * 60)
