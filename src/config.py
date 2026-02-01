"""
Configuration Management Module

Loads, validates, and provides access to all bot configuration settings.
Uses Pydantic for type validation and settings management.
"""

import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class BrokerConfig(BaseModel):
    """Broker connection settings."""
    name: str = "IC Markets Australia"
    account_type: str = "Standard"
    max_leverage: int = Field(default=30, ge=1, le=500)
    base_currency: str = "USD"
    api_endpoint: str = "https://api.icmarkets.com"
    demo_endpoint: str = "https://demo-api.icmarkets.com"


class TradingConfig(BaseModel):
    """General trading settings."""
    pairs: List[str] = ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD"]
    trading_mode: str = "MANUAL_TRIGGER"
    min_lot_size: float = Field(default=0.01, ge=0.01)
    max_lot_size: float = Field(default=0.10, le=10.0)
    default_timeframe: str = "15m"

    @field_validator('pairs')
    @classmethod
    def validate_pairs(cls, v: List[str]) -> List[str]:
        """Ensure only supported major pairs are used."""
        supported_pairs = {"EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD"}
        for pair in v:
            if pair not in supported_pairs:
                raise ValueError(f"Unsupported pair: {pair}. Only major pairs allowed.")
        return v


class GridStrategyConfig(BaseModel):
    """Grid trading strategy settings."""
    enabled: bool = True
    grid_lines: int = Field(default=15, ge=10, le=25)
    profit_per_grid_pips: float = Field(default=15, ge=10, le=30)
    stop_loss_pips: float = Field(default=30, ge=20)
    upper_limit: float = 0  # Set dynamically per trade
    lower_limit: float = 0  # Set dynamically per trade


class TrendStrategyConfig(BaseModel):
    """Trend following strategy settings."""
    enabled: bool = True
    sma_period_short: int = Field(default=9, ge=5, le=20)
    sma_period_long: int = Field(default=20, ge=15, le=50)
    timeframe: str = "15m"
    take_profit_pips: float = Field(default=75, ge=50, le=150)
    stop_loss_pips: float = Field(default=50, ge=30, le=100)
    trailing_stop_trigger_pips: float = Field(default=30, ge=20)
    trailing_stop_offset_pips: float = Field(default=10, ge=5)


class RiskManagementConfig(BaseModel):
    """Risk management settings - NON-NEGOTIABLE."""
    account_balance: float = Field(default=1000, ge=100)
    risk_percent_per_trade: float = Field(default=1.0, ge=0.5, le=2.0)
    daily_loss_limit: float = Field(default=50, ge=10)
    max_drawdown_percent: float = Field(default=10, ge=5, le=20)
    max_spread_allowed: float = Field(default=3.0, ge=1.0, le=5.0)
    min_stop_loss_pips: float = Field(default=20, ge=15)
    min_take_profit_pips: float = Field(default=20, ge=15)
    margin_buffer_percent: float = Field(default=20, ge=10, le=50)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    output_dir: str = "./logs"
    output_file: str = "trading_bot.log"
    max_log_file_size_mb: int = Field(default=50, ge=10)
    backup_log_count: int = Field(default=5, ge=1)
    console_output: bool = True
    colored_output: bool = True

    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class APIConfig(BaseModel):
    """API connection settings."""
    timeout_seconds: int = Field(default=10, ge=5, le=60)
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_backoff_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    health_check_interval_seconds: int = Field(default=30, ge=10)


class DashboardConfig(BaseModel):
    """Dashboard display settings."""
    refresh_interval_seconds: int = Field(default=5, ge=1, le=30)
    show_debug_panel: bool = False


class DryRunConfig(BaseModel):
    """Dry run / simulation settings."""
    enabled: bool = True
    simulated_spread_pips: float = Field(default=1.5, ge=0.5)
    simulated_slippage_pips: float = Field(default=0.5, ge=0)


class BotConfig(BaseModel):
    """
    Master configuration for the Trading Bot.
    Aggregates all sub-configurations.
    """
    broker: BrokerConfig = BrokerConfig()
    trading: TradingConfig = TradingConfig()
    grid_strategy: GridStrategyConfig = GridStrategyConfig()
    trend_strategy: TrendStrategyConfig = TrendStrategyConfig()
    risk_management: RiskManagementConfig = RiskManagementConfig()
    logging: LoggingConfig = LoggingConfig()
    api: APIConfig = APIConfig()
    dashboard: DashboardConfig = DashboardConfig()
    dry_run: DryRunConfig = DryRunConfig()

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "BotConfig":
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            BotConfig instance with loaded settings.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)

    def to_json_file(self, file_path: str | Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save the JSON configuration file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

    def validate_for_trading(self) -> List[str]:
        """
        Perform trading-specific validation.
        
        Returns:
            List of warning messages (empty if all OK).
        """
        warnings = []
        
        # Check leverage constraint for IC Markets Australia
        if self.broker.max_leverage > 30:
            warnings.append(
                f"Leverage {self.broker.max_leverage}x exceeds IC Markets AU limit of 30x"
            )
        
        # Check account size vs daily loss limit
        max_daily_loss_pct = (self.risk_management.daily_loss_limit / 
                              self.risk_management.account_balance * 100)
        if max_daily_loss_pct > 5:
            warnings.append(
                f"Daily loss limit is {max_daily_loss_pct:.1f}% of account (recommended max 5%)"
            )
        
        # Check lot size constraints
        if self.trading.max_lot_size > 0.10 and self.risk_management.account_balance <= 1000:
            warnings.append(
                "Max lot size > 0.10 may be too aggressive for $1000 account"
            )
        
        return warnings


def load_config(config_path: Optional[str] = None) -> BotConfig:
    """
    Load bot configuration from file or use defaults.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Validated BotConfig instance.
    """
    if config_path:
        return BotConfig.from_json_file(config_path)
    
    # Default config path
    default_path = Path(__file__).parent.parent / "config" / "settings.json"
    if default_path.exists():
        return BotConfig.from_json_file(default_path)
    
    # Return default configuration
    return BotConfig()


# Singleton instance for global access
_config: Optional[BotConfig] = None


def get_config() -> BotConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: BotConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
