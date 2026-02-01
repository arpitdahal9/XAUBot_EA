"""
Symbol Specification Model

Provides accurate pip values, tick sizes, and PnL calculations for different instruments.
Critical for correct risk sizing and ML label accuracy.

Supports:
- Forex majors (EURUSD, GBPUSD, etc.)
- JPY pairs (USDJPY, etc.)
- Gold (XAUUSD)
- Custom symbols via config
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SymbolSpec:
    """Specification for a trading symbol."""
    symbol: str
    tick_size: float          # Minimum price movement (e.g., 0.00001 for EURUSD)
    tick_value_per_lot: float # Value in USD of one tick for 1 standard lot
    contract_size: float      # Units per lot (e.g., 100000 for forex)
    pip_size: float           # Size of one pip (e.g., 0.0001 for EURUSD)
    pip_value_per_lot: float  # Value in USD of one pip for 1 standard lot
    digits: int               # Number of decimal places
    min_lot: float = 0.01
    lot_step: float = 0.01
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips."""
        return price_diff / self.pip_size
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price difference."""
        return pips * self.pip_size
    
    def calculate_pnl(self, entry_price: float, exit_price: float, lot_size: float, is_buy: bool) -> float:
        """Calculate PnL in USD for a trade."""
        price_diff = exit_price - entry_price if is_buy else entry_price - exit_price
        pips = self.price_to_pips(price_diff)
        return pips * self.pip_value_per_lot * lot_size
    
    def calculate_lot_size(self, risk_usd: float, stop_distance_pips: float) -> float:
        """Calculate lot size for given risk and stop distance."""
        if stop_distance_pips <= 0:
            return 0.0
        risk_per_pip = risk_usd / stop_distance_pips
        lot_size = risk_per_pip / self.pip_value_per_lot
        # Round to lot step
        lot_size = max(self.min_lot, round(lot_size / self.lot_step) * self.lot_step)
        return lot_size


# Default symbol specifications (USD account)
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    # Forex Majors
    "EUR/USD": SymbolSpec(
        symbol="EUR/USD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    "GBP/USD": SymbolSpec(
        symbol="GBP/USD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    "GBPUSD": SymbolSpec(
        symbol="GBPUSD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    "AUD/USD": SymbolSpec(
        symbol="AUD/USD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    "AUDUSD": SymbolSpec(
        symbol="AUDUSD",
        tick_size=0.00001,
        tick_value_per_lot=1.0,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=10.0,
        digits=5
    ),
    # JPY Pairs (different pip size)
    "USD/JPY": SymbolSpec(
        symbol="USD/JPY",
        tick_size=0.001,
        tick_value_per_lot=0.67,  # Approx for 150 USDJPY
        contract_size=100000,
        pip_size=0.01,
        pip_value_per_lot=6.7,   # Approx for 150 USDJPY
        digits=3
    ),
    "USDJPY": SymbolSpec(
        symbol="USDJPY",
        tick_size=0.001,
        tick_value_per_lot=0.67,
        contract_size=100000,
        pip_size=0.01,
        pip_value_per_lot=6.7,
        digits=3
    ),
    # GBP/JPY (JPY cross - pip size 0.01)
    "GBP/JPY": SymbolSpec(
        symbol="GBP/JPY",
        tick_size=0.001,
        tick_value_per_lot=0.67,
        contract_size=100000,
        pip_size=0.01,
        pip_value_per_lot=6.7,
        digits=3
    ),
    "GBPJPY": SymbolSpec(
        symbol="GBPJPY",
        tick_size=0.001,
        tick_value_per_lot=0.67,
        contract_size=100000,
        pip_size=0.01,
        pip_value_per_lot=6.7,
        digits=3
    ),
    # USD/CAD
    "USD/CAD": SymbolSpec(
        symbol="USD/CAD",
        tick_size=0.00001,
        tick_value_per_lot=0.74,  # Approx for 1.35 USDCAD
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=7.4,
        digits=5
    ),
    "USDCAD": SymbolSpec(
        symbol="USDCAD",
        tick_size=0.00001,
        tick_value_per_lot=0.74,
        contract_size=100000,
        pip_size=0.0001,
        pip_value_per_lot=7.4,
        digits=5
    ),
    # GOLD (XAUUSD) - CRITICAL: Different from forex
    "XAU/USD": SymbolSpec(
        symbol="XAU/USD",
        tick_size=0.01,           # Most brokers use 0.01 tick
        tick_value_per_lot=1.0,   #  per tick per lot (100 oz)
        contract_size=100,        # 100 troy ounces per lot
        pip_size=0.1,             # Gold pip = .10
        pip_value_per_lot=10.0,   #  per pip per lot
        digits=2
    ),
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        tick_size=0.01,
        tick_value_per_lot=1.0,
        contract_size=100,
        pip_size=0.1,
        pip_value_per_lot=10.0,
        digits=2
    ),
}


class SymbolSpecRegistry:
    """Registry for symbol specifications."""
    
    def __init__(self):
        self._specs = DEFAULT_SPECS.copy()
        self._warned: set = set()
    
    def get_spec(self, symbol: str) -> SymbolSpec:
        """
        Get specification for a symbol.
        
        Falls back to forex defaults with warning if symbol not found.
        """
        # Normalize symbol name
        normalized = symbol.upper().replace("/", "")
        
        # Try exact match first
        if symbol in self._specs:
            return self._specs[symbol]
        
        # Try normalized
        for key, spec in self._specs.items():
            if key.upper().replace("/", "") == normalized:
                return spec
        
        # Fallback with warning
        if symbol not in self._warned:
            logger.warning(
                f"Symbol spec not found for '{symbol}', using forex defaults. "
                f"PnL and lot sizing may be inaccurate!"
            )
            self._warned.add(symbol)
        
        # Return forex default
        return SymbolSpec(
            symbol=symbol,
            tick_size=0.00001,
            tick_value_per_lot=1.0,
            contract_size=100000,
            pip_size=0.0001,
            pip_value_per_lot=10.0,
            digits=5
        )
    
    def register_spec(self, spec: SymbolSpec) -> None:
        """Register a custom symbol specification."""
        self._specs[spec.symbol] = spec
        logger.info(f"Registered symbol spec: {spec.symbol}")
    
    def load_from_config(self, config: dict) -> None:
        """Load custom symbol specs from config dictionary."""
        symbols = config.get("symbol_specs", {})
        for sym_name, sym_data in symbols.items():
            try:
                spec = SymbolSpec(
                    symbol=sym_name,
                    tick_size=sym_data.get("tick_size", 0.00001),
                    tick_value_per_lot=sym_data.get("tick_value_per_lot", 1.0),
                    contract_size=sym_data.get("contract_size", 100000),
                    pip_size=sym_data.get("pip_size", 0.0001),
                    pip_value_per_lot=sym_data.get("pip_value_per_lot", 10.0),
                    digits=sym_data.get("digits", 5),
                    min_lot=sym_data.get("min_lot", 0.01),
                    lot_step=sym_data.get("lot_step", 0.01)
                )
                self.register_spec(spec)
            except Exception as e:
                logger.error(f"Failed to load symbol spec {sym_name}: {e}")


# Global registry instance
_registry = SymbolSpecRegistry()


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """Get symbol specification from global registry."""
    return _registry.get_spec(symbol)


def register_symbol_spec(spec: SymbolSpec) -> None:
    """Register symbol specification in global registry."""
    _registry.register_spec(spec)


def load_symbol_specs_from_config(config: dict) -> None:
    """Load symbol specs from config into global registry."""
    _registry.load_from_config(config)
