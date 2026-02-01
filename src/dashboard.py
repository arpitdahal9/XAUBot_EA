"""
Real-Time Monitoring Dashboard

Provides a comprehensive CLI dashboard for monitoring the trading bot.
Displays account status, market data, positions, and trade history.
Uses the Rich library for beautiful terminal output.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .models import (
    Order, OrderSide, OrderStatus, MarketData, AccountState,
    DailyStats, GridConfig, Signal
)
from .grid_trader import GridState
from .trend_follower import TrendState
from .config import get_config


class TradingDashboard:
    """
    Real-time CLI dashboard for the trading bot.
    
    Displays:
    - Account status (balance, equity, margin, drawdown)
    - Current market data (prices, spreads)
    - Grid trading status
    - Trend following status
    - Open positions table
    - Recent trade history
    - Live P&L updates
    """

    def __init__(self):
        """Initialize the dashboard."""
        self.console = Console()
        self.config = get_config()
        
        # State storage
        self._account_state: Optional[AccountState] = None
        self._market_data: Dict[str, MarketData] = {}
        self._open_positions: List[Order] = []
        self._trade_history: List[Order] = []
        self._grid_states: Dict[str, GridState] = {}
        self._trend_states: Dict[str, TrendState] = {}
        self._daily_stats: Optional[DailyStats] = None
        self._last_update: datetime = datetime.now()
        self._is_running: bool = False
        self._alerts: List[str] = []

    def update_account_state(self, state: AccountState) -> None:
        """Update account state data."""
        self._account_state = state
        self._last_update = datetime.now()

    def update_market_data(self, pair: str, data: MarketData) -> None:
        """Update market data for a pair."""
        self._market_data[pair] = data
        self._last_update = datetime.now()

    def update_positions(self, positions: List[Order]) -> None:
        """Update open positions list."""
        self._open_positions = positions
        self._last_update = datetime.now()

    def add_trade_to_history(self, trade: Order) -> None:
        """Add a completed trade to history."""
        self._trade_history.insert(0, trade)
        # Keep last 20 trades
        self._trade_history = self._trade_history[:20]
        self._last_update = datetime.now()

    def update_grid_state(self, pair: str, state: GridState) -> None:
        """Update grid trading state."""
        self._grid_states[pair] = state
        self._last_update = datetime.now()

    def update_trend_state(self, pair: str, state: TrendState) -> None:
        """Update trend following state."""
        self._trend_states[pair] = state
        self._last_update = datetime.now()

    def update_daily_stats(self, stats: DailyStats) -> None:
        """Update daily statistics."""
        self._daily_stats = stats
        self._last_update = datetime.now()

    def add_alert(self, message: str) -> None:
        """Add an alert message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._alerts.insert(0, f"[{timestamp}] {message}")
        # Keep last 5 alerts
        self._alerts = self._alerts[:5]

    def _create_header(self) -> Panel:
        """Create the dashboard header."""
        title = Text("TRADING BOT LIVE MONITOR", style="bold white on blue", justify="center")
        subtitle = Text(
            f"Last Update: {self._last_update.strftime('%Y-%m-%d %H:%M:%S')}",
            style="dim",
            justify="center"
        )
        
        header_text = Text()
        header_text.append_text(title)
        header_text.append("\n")
        header_text.append_text(subtitle)
        
        return Panel(
            header_text,
            box=box.DOUBLE,
            border_style="blue"
        )

    def _create_account_panel(self) -> Panel:
        """Create account status panel."""
        if not self._account_state:
            return Panel("Waiting for account data...", title="Account Status")
        
        state = self._account_state
        config = self.config.risk_management
        
        # Calculate drawdown
        drawdown = config.account_balance - state.equity
        drawdown_pct = (drawdown / config.account_balance * 100) if config.account_balance > 0 else 0
        
        # Daily loss remaining
        daily_loss_used = config.daily_loss_limit - (config.account_balance - state.balance)
        
        # Color coding
        equity_color = "green" if state.equity >= config.account_balance else "red"
        drawdown_color = "green" if drawdown_pct < 5 else "yellow" if drawdown_pct < 10 else "red"
        margin_color = "green" if state.margin_level > 200 else "yellow" if state.margin_level > 100 else "red"
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Account Balance:", f"${state.balance:,.2f}")
        table.add_row("Equity:", Text(f"${state.equity:,.2f}", style=equity_color))
        table.add_row("Margin Used:", f"${state.margin_used:,.2f}")
        table.add_row("Free Margin:", f"${state.free_margin:,.2f}")
        table.add_row("Margin Level:", Text(f"{state.margin_level:.1f}%", style=margin_color))
        table.add_row("", "")
        table.add_row("Drawdown Today:", Text(f"{drawdown_pct:.1f}% (${drawdown:,.2f})", style=drawdown_color))
        table.add_row("Daily Loss Limit:", f"${config.daily_loss_limit:.2f}")
        table.add_row("Remaining:", f"${max(0, daily_loss_used):,.2f}")
        
        return Panel(
            table,
            title="[bold]Account Status[/bold]",
            border_style="green" if state.equity >= config.account_balance else "red"
        )

    def _create_market_panel(self) -> Panel:
        """Create current market data panel."""
        if not self._market_data:
            return Panel("Waiting for market data...", title="Current Market")
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Pair", style="bold cyan")
        table.add_column("Bid", justify="right")
        table.add_column("Ask", justify="right")
        table.add_column("Spread", justify="right")
        table.add_column("Updated", justify="right", style="dim")
        
        for pair, data in self._market_data.items():
            spread_color = "green" if data.spread_pips < 2 else "yellow" if data.spread_pips < 3 else "red"
            time_str = data.timestamp.strftime("%H:%M:%S")
            
            table.add_row(
                pair,
                f"{data.bid:.5f}",
                f"{data.ask:.5f}",
                Text(f"{data.spread_pips:.1f} pips", style=spread_color),
                time_str
            )
        
        return Panel(
            table,
            title="[bold]Current Market[/bold]",
            border_style="cyan"
        )

    def _create_grid_panel(self) -> Panel:
        """Create grid trading status panel."""
        if not self._grid_states:
            return Panel("No active grids", title="Grid Trading Status")
        
        content = Text()
        
        for pair, state in self._grid_states.items():
            if not state.grid_config:
                continue
            
            gc = state.grid_config
            content.append(f"Pair: ", style="bold")
            content.append(f"{pair}\n", style="cyan")
            content.append(f"Range: {gc.lower_limit:.5f} - {gc.upper_limit:.5f}\n")
            content.append(f"Grid Lines: {gc.num_grids} ({gc.grid_spacing_pips:.1f} pips each)\n")
            content.append(f"Active Orders: ", style="bold")
            content.append(f"Buy: {state.active_buy_orders} | Sell: {state.active_sell_orders}\n")
            content.append(f"Grid P&L: ", style="bold")
            pnl_color = "green" if state.total_profit >= 0 else "red"
            content.append(f"${state.total_profit:+.2f}\n", style=pnl_color)
        
        return Panel(
            content,
            title="[bold]Grid Trading Status[/bold]",
            border_style="magenta"
        )

    def _create_trend_panel(self) -> Panel:
        """Create trend following status panel."""
        if not self._trend_states:
            return Panel("No trend data", title="Trend Following Status")
        
        content = Text()
        
        for pair, state in self._trend_states.items():
            content.append(f"Pair: ", style="bold")
            content.append(f"{pair}\n", style="cyan")
            
            if state.sma_short > 0 and state.sma_long > 0:
                content.append(f"SMA9: {state.sma_short:.5f}\n")
                content.append(f"SMA20: {state.sma_long:.5f}\n")
                
                if state.is_bullish:
                    content.append("Status: ", style="bold")
                    content.append("BULLISH (9 above 20)\n", style="green bold")
                elif state.is_bearish:
                    content.append("Status: ", style="bold")
                    content.append("BEARISH (9 below 20)\n", style="red bold")
                else:
                    content.append("Status: ", style="bold")
                    content.append("NEUTRAL\n", style="yellow")
                
                content.append(f"Signal Strength: {state.signal_strength:.0%}\n")
                
                if state.last_crossover:
                    content.append(f"Last Crossover: {state.last_crossover}\n")
        
        return Panel(
            content,
            title="[bold]Trend Following Status[/bold]",
            border_style="yellow"
        )

    def _create_positions_panel(self) -> Panel:
        """Create open positions table panel."""
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Order #", style="dim")
        table.add_column("Type", justify="center")
        table.add_column("Entry", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")
        table.add_column("Pips", justify="right")
        table.add_column("P&L", justify="right")
        
        if not self._open_positions:
            table.add_row("", "", "No open positions", "", "", "", "")
        else:
            total_pnl = 0.0
            for order in self._open_positions:
                # Calculate pips and P&L
                current_price = order.metadata.get("current_price", order.entry_price)
                entry = order.fill_price or order.entry_price
                
                if order.side == OrderSide.BUY:
                    pips = (current_price - entry) * (100 if "JPY" in order.pair else 10000)
                else:
                    pips = (entry - current_price) * (100 if "JPY" in order.pair else 10000)
                
                pip_value = 0.10 * order.lot_size * 100
                pnl = pips * pip_value
                total_pnl += pnl
                
                pips_color = "green" if pips >= 0 else "red"
                type_color = "green" if order.side == OrderSide.BUY else "red"
                
                table.add_row(
                    order.order_id[:8],
                    Text(order.side.value, style=type_color),
                    f"{entry:.5f}",
                    f"{order.stop_loss:.5f}",
                    f"{order.take_profit:.5f}",
                    Text(f"{pips:+.1f}", style=pips_color),
                    Text(f"${pnl:+.2f}", style=pips_color)
                )
            
            # Add total row
            total_color = "green bold" if total_pnl >= 0 else "red bold"
            table.add_row("", "", "", "", "Total:", "", Text(f"${total_pnl:+.2f}", style=total_color))
        
        return Panel(
            table,
            title="[bold]Active Positions[/bold]",
            border_style="blue"
        )

    def _create_history_panel(self) -> Panel:
        """Create recent trade history panel."""
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("Time", style="dim")
        table.add_column("Type")
        table.add_column("Pair")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("Pips", justify="right")
        table.add_column("Profit", justify="right")
        
        if not self._trade_history:
            table.add_row("", "", "No trade history", "", "", "", "")
        else:
            for trade in self._trade_history[:10]:  # Show last 10 trades
                if not trade.close_time:
                    continue
                
                entry = trade.fill_price or trade.entry_price
                exit_price = trade.close_price or entry
                
                if trade.side == OrderSide.BUY:
                    pips = (exit_price - entry) * (100 if "JPY" in trade.pair else 10000)
                else:
                    pips = (entry - exit_price) * (100 if "JPY" in trade.pair else 10000)
                
                pnl = trade.pnl or 0
                pips_color = "green" if pips >= 0 else "red"
                type_color = "green" if trade.side == OrderSide.BUY else "red"
                
                table.add_row(
                    trade.close_time.strftime("%H:%M"),
                    Text(trade.side.value, style=type_color),
                    trade.pair,
                    f"{entry:.5f}",
                    f"{exit_price:.5f}",
                    Text(f"{pips:+.0f}", style=pips_color),
                    Text(f"${pnl:+.2f}", style=pips_color)
                )
        
        # Add summary
        if self._daily_stats and self._daily_stats.total_trades > 0:
            stats = self._daily_stats
            summary = Text(f"\nWin Rate: {stats.win_rate:.0f}% | ")
            summary.append(f"Avg Win: ${stats.avg_win:.2f} | ", style="green")
            summary.append(f"Avg Loss: ${stats.avg_loss:.2f} | ", style="red")
            summary.append(f"Profit Factor: {stats.profit_factor:.2f}")
            table.add_row("", "", "", "", "", "", "")
            table.add_row(summary, "", "", "", "", "", "")
        
        return Panel(
            table,
            title="[bold]Recent Trades (Last 24h)[/bold]",
            border_style="white"
        )

    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        if not self._alerts:
            content = Text("No alerts", style="dim")
        else:
            content = Text()
            for alert in self._alerts:
                content.append("⚠ ", style="yellow bold")
                content.append(alert + "\n")
        
        return Panel(
            content,
            title="[bold]Alerts[/bold]",
            border_style="yellow"
        )

    def create_layout(self) -> Layout:
        """Create the full dashboard layout."""
        layout = Layout()
        
        # Main structure
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=8)
        )
        
        # Body split into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Left side: Account and Market
        layout["left"].split_column(
            Layout(name="account", size=14),
            Layout(name="market")
        )
        
        # Right side: Strategies and Positions
        layout["right"].split_column(
            Layout(name="strategies"),
            Layout(name="positions")
        )
        
        # Strategies split
        layout["strategies"].split_row(
            Layout(name="grid"),
            Layout(name="trend")
        )
        
        # Assign panels
        layout["header"].update(self._create_header())
        layout["account"].update(self._create_account_panel())
        layout["market"].update(self._create_market_panel())
        layout["grid"].update(self._create_grid_panel())
        layout["trend"].update(self._create_trend_panel())
        layout["positions"].update(self._create_positions_panel())
        layout["footer"].update(self._create_history_panel())
        
        return layout

    def render_once(self) -> None:
        """Render the dashboard once."""
        layout = self.create_layout()
        self.console.clear()
        self.console.print(layout)

    async def run_live(self, refresh_interval: float = 5.0) -> None:
        """
        Run the dashboard with live updates.
        
        Args:
            refresh_interval: Seconds between updates.
        """
        self._is_running = True
        
        with Live(
            self.create_layout(),
            refresh_per_second=1 / refresh_interval,
            console=self.console,
            screen=True
        ) as live:
            while self._is_running:
                live.update(self.create_layout())
                await asyncio.sleep(refresh_interval)

    def stop(self) -> None:
        """Stop the live dashboard."""
        self._is_running = False


def create_simple_dashboard() -> str:
    """
    Create a simple text-based dashboard string.
    
    Returns:
        Formatted dashboard string for logging or simple display.
    """
    config = get_config()
    
    lines = [
        "═" * 60,
        "          TRADING BOT LIVE MONITOR",
        "═" * 60,
        "",
        "Account Status:",
        f"  Account Balance: ${config.risk_management.account_balance:,.2f}",
        f"  Daily Loss Limit: ${config.risk_management.daily_loss_limit:.2f}",
        f"  Max Drawdown: {config.risk_management.max_drawdown_percent}%",
        "",
        "Trading Configuration:",
        f"  Pairs: {', '.join(config.trading.pairs)}",
        f"  Mode: {config.trading.trading_mode}",
        f"  Lot Range: {config.trading.min_lot_size} - {config.trading.max_lot_size}",
        "",
        "Grid Strategy:",
        f"  Enabled: {'Yes' if config.grid_strategy.enabled else 'No'}",
        f"  Grid Lines: {config.grid_strategy.grid_lines}",
        f"  Profit/Grid: {config.grid_strategy.profit_per_grid_pips} pips",
        "",
        "Trend Strategy:",
        f"  Enabled: {'Yes' if config.trend_strategy.enabled else 'No'}",
        f"  SMA Periods: {config.trend_strategy.sma_period_short}/{config.trend_strategy.sma_period_long}",
        f"  TP/SL: {config.trend_strategy.take_profit_pips}/{config.trend_strategy.stop_loss_pips} pips",
        "",
        "═" * 60
    ]
    
    return "\n".join(lines)
