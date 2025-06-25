"""
Backtesting Framework for Statistical Arbitrage Strategies

This module provides a comprehensive framework for simulating trading strategies
using historical data to evaluate the performance of generated alpha signals.
It handles order execution, transaction costs, slippage, and performance metrics.

Classes:
    - Backtester: Main backtesting engine
    - Portfolio: Portfolio state management
    - Order: Trade order representation
    - Trade: Executed trade representation
    - PerformanceAnalyzer: Performance metrics calculation
    - TransactionCostModel: Transaction cost modeling
    - RiskManager: Risk management and position sizing

Features:
    - Event-driven simulation
    - Realistic transaction costs and slippage
    - Market impact modeling
    - Comprehensive performance metrics
    - Risk management integration
    - Position sizing algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle

# Statistical and financial libraries
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """
    Represents a trading order.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"


@dataclass
class Trade:
    """
    Represents an executed trade.
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: pd.Timestamp
    trade_id: str
    order_id: str
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    
    @property
    def notional(self) -> float:
        """Calculate notional value of the trade."""
        return abs(self.quantity * self.price)
    
    @property
    def net_cash_flow(self) -> float:
        """Calculate net cash flow including costs."""
        base_flow = -self.quantity * self.price if self.side == OrderSide.BUY else self.quantity * self.price
        return base_flow - self.commission


@dataclass
class Position:
    """
    Represents a position in a security.
    """
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    
    def update_market_data(self, price: float):
        """Update position with new market price."""
        self.last_price = price
        self.market_value = self.quantity * price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity
    
    def add_trade(self, trade: Trade):
        """Add a trade to the position."""
        if trade.side == OrderSide.BUY:
            new_quantity = self.quantity + trade.quantity
            if new_quantity != 0:
                self.avg_price = (self.quantity * self.avg_price + trade.quantity * trade.price) / new_quantity
            self.quantity = new_quantity
        else:  # SELL
            if self.quantity > 0:
                # Calculate realized PnL for the sold portion
                sold_quantity = min(trade.quantity, self.quantity)
                self.realized_pnl += (trade.price - self.avg_price) * sold_quantity
            self.quantity -= trade.quantity
            
            # Update average price if position flips to short
            if self.quantity < 0:
                remaining_short = abs(self.quantity)
                self.avg_price = trade.price  # Reset to current trade price for short position


class TransactionCostModel:
    """
    Models transaction costs including commissions, slippage, and market impact.
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,  # 0.1% commission
                 fixed_commission: float = 0.0,
                 bid_ask_spread: float = 0.001,  # 0.1% spread
                 market_impact_model: str = 'square_root',
                 liquidity_factor: float = 1e6):
        """
        Initialize transaction cost model.
        
        Args:
            commission_rate: Percentage commission rate
            fixed_commission: Fixed commission per trade
            bid_ask_spread: Bid-ask spread as percentage of price
            market_impact_model: Model for market impact ('linear', 'square_root', 'log')
            liquidity_factor: Factor for market impact calculation
        """
        self.commission_rate = commission_rate
        self.fixed_commission = fixed_commission
        self.bid_ask_spread = bid_ask_spread
        self.market_impact_model = market_impact_model
        self.liquidity_factor = liquidity_factor
    
    def calculate_commission(self, notional_value: float) -> float:
        """Calculate commission for a trade."""
        return self.fixed_commission + (notional_value * self.commission_rate)
    
    def calculate_slippage(self, order: Order, market_price: float) -> float:
        """Calculate slippage for an order."""
        if order.order_type == OrderType.MARKET:
            # Market orders face bid-ask spread
            return market_price * self.bid_ask_spread / 2
        else:
            # Limit orders may have less slippage
            return 0.0
    
    def calculate_market_impact(self, order: Order, market_price: float, 
                              avg_volume: float) -> float:
        """Calculate market impact for an order."""
        if avg_volume <= 0:
            return 0.0
        
        participation_rate = abs(order.quantity) / avg_volume
        
        if self.market_impact_model == 'linear':
            impact = participation_rate * market_price * 0.01
        elif self.market_impact_model == 'square_root':
            impact = np.sqrt(participation_rate) * market_price * 0.01
        elif self.market_impact_model == 'log':
            impact = np.log(1 + participation_rate) * market_price * 0.01
        else:
            impact = 0.0
        
        return impact
    
    def calculate_total_costs(self, order: Order, market_price: float, 
                            avg_volume: float) -> Tuple[float, float, float]:
        """
        Calculate total transaction costs.
        
        Returns:
            Tuple of (commission, slippage, market_impact)
        """
        notional = abs(order.quantity * market_price)
        commission = self.calculate_commission(notional)
        slippage = self.calculate_slippage(order, market_price)
        market_impact = self.calculate_market_impact(order, market_price, avg_volume)
        
        return commission, slippage, market_impact


class RiskManager:
    """
    Risk management system for position sizing and risk controls.
    """
    
    def __init__(self,
                 max_position_size: float = 0.05,  # 5% of portfolio
                 max_sector_exposure: float = 0.20,  # 20% per sector
                 max_leverage: float = 1.0,  # No leverage
                 stop_loss_pct: float = 0.05,  # 5% stop loss
                 volatility_target: float = 0.10):  # 10% annual volatility target
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_sector_exposure: Maximum sector exposure
            max_leverage: Maximum leverage allowed
            stop_loss_pct: Stop loss percentage
            volatility_target: Target portfolio volatility
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.volatility_target = volatility_target
    
    def calculate_position_size(self, signal_strength: float, portfolio_value: float,
                              price: float, volatility: float = None) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal_strength: Alpha signal strength (-1 to 1)
            portfolio_value: Current portfolio value
            price: Current price of the security
            volatility: Security volatility for volatility targeting
            
        Returns:
            Position size in shares
        """
        # Base position size from signal strength
        base_size = abs(signal_strength) * self.max_position_size * portfolio_value
        
        # Adjust for volatility if provided
        if volatility and volatility > 0:
            vol_adjustment = self.volatility_target / volatility
            base_size *= min(vol_adjustment, 2.0)  # Cap adjustment at 2x
        
        # Convert to shares
        shares = base_size / price
        
        return shares * np.sign(signal_strength)
    
    def check_risk_limits(self, order: Order, portfolio, market_data: pd.DataFrame) -> bool:
        """
        Check if an order violates risk limits.
        
        Args:
            order: Order to check
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            True if order passes risk checks
        """
        # Check maximum position size
        current_price = market_data.loc[market_data.index[-1], order.symbol]
        new_notional = abs(order.quantity * current_price)
        
        if new_notional > self.max_position_size * portfolio.total_value:
            logger.warning(f"Order {order.order_id} exceeds max position size")
            return False
        
        # Check leverage
        total_leverage = portfolio.calculate_leverage()
        if total_leverage > self.max_leverage:
            logger.warning(f"Portfolio leverage {total_leverage:.2f} exceeds limit")
            return False
        
        return True


class Portfolio:
    """
    Portfolio state management.
    """
    
    def __init__(self, initial_cash: float = 1000000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Initial cash amount
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.daily_returns: List[float] = []
        self.daily_values: List[float] = [initial_cash]
        self.timestamps: List[pd.Timestamp] = []
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.total_value - self.initial_cash
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    def calculate_leverage(self) -> float:
        """Calculate portfolio leverage."""
        gross_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return gross_exposure / self.total_value if self.total_value > 0 else 0.0
    
    def update_market_data(self, market_data: pd.Series, timestamp: pd.Timestamp):
        """Update portfolio with new market data."""
        for symbol, position in self.positions.items():
            if symbol in market_data.index:
                position.update_market_data(market_data[symbol])
        
        # Track daily values and returns
        current_value = self.total_value
        self.daily_values.append(current_value)
        self.timestamps.append(timestamp)
        
        if len(self.daily_values) > 1:
            daily_return = (current_value - self.daily_values[-2]) / self.daily_values[-2]
            self.daily_returns.append(daily_return)
    
    def add_trade(self, trade: Trade):
        """Add a trade to the portfolio."""
        self.trades.append(trade)
        
        # Update position
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(trade.symbol)
        
        self.positions[trade.symbol].add_trade(trade)
        
        # Update cash
        self.cash += trade.net_cash_flow
        
        # Track costs
        self.total_commission += trade.commission
        self.total_slippage += trade.slippage
        self.total_market_impact += trade.market_impact
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'leverage': self.calculate_leverage(),
            'num_positions': len([p for p in self.positions.values() if p.quantity != 0]),
            'total_trades': len(self.trades),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_market_impact': self.total_market_impact
        }


class PerformanceAnalyzer:
    """
    Performance metrics calculation and analysis.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: Portfolio object
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio.daily_returns) == 0:
            return {}
        
        returns = np.array(portfolio.daily_returns)
        values = np.array(portfolio.daily_values)
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        winning_trades = [t for t in portfolio.trades if t.net_cash_flow > 0]
        losing_trades = [t for t in portfolio.trades if t.net_cash_flow < 0]
        
        win_rate = len(winning_trades) / len(portfolio.trades) if portfolio.trades else 0
        gross_profit = sum(t.net_cash_flow for t in winning_trades)
        gross_loss = abs(sum(t.net_cash_flow for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trade statistics
        avg_trade_pnl = np.mean([t.net_cash_flow for t in portfolio.trades]) if portfolio.trades else 0
        
        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_trades': len(portfolio.trades),
            'total_commission': portfolio.total_commission,
            'total_slippage': portfolio.total_slippage,
            'final_portfolio_value': portfolio.total_value
        }
    
    def plot_performance(self, portfolio: Portfolio, save_path: Optional[str] = None):
        """
        Create performance visualization plots.
        
        Args:
            portfolio: Portfolio object
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio.timestamps, portfolio.daily_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Daily returns
        if portfolio.daily_returns:
            axes[0, 1].hist(portfolio.daily_returns, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Distribution of Daily Returns')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Cumulative returns
        if portfolio.daily_returns:
            cumulative_returns = np.cumprod(1 + np.array(portfolio.daily_returns))
            axes[1, 0].plot(portfolio.timestamps[1:], cumulative_returns)
            axes[1, 0].set_title('Cumulative Returns')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Cumulative Return')
            axes[1, 0].grid(True)
        
        # Drawdown
        if portfolio.daily_returns:
            returns = np.array(portfolio.daily_returns)
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            axes[1, 1].fill_between(portfolio.timestamps[1:], drawdowns, 0, alpha=0.7, color='red')
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class Backtester:
    """
    Main backtesting engine for statistical arbitrage strategies.
    """
    
    def __init__(self,
                 initial_cash: float = 1000000.0,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 risk_manager: Optional[RiskManager] = None,
                 performance_analyzer: Optional[PerformanceAnalyzer] = None):
        """
        Initialize the backtester.
        
        Args:
            initial_cash: Initial portfolio cash
            transaction_cost_model: Transaction cost model
            risk_manager: Risk management system
            performance_analyzer: Performance analysis system
        """
        self.initial_cash = initial_cash
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.risk_manager = risk_manager or RiskManager()
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()
        
        # State
        self.portfolio = Portfolio(initial_cash)
        self.current_timestamp = None
        self.market_data = None
        self.volume_data = None
        
        # Strategy function
        self.strategy_function: Optional[Callable] = None
        
    def set_strategy(self, strategy_function: Callable):
        """
        Set the trading strategy function.
        
        Args:
            strategy_function: Function that takes market data and returns signals
        """
        self.strategy_function = strategy_function
    
    def run_backtest(self, 
                    market_data: pd.DataFrame,
                    volume_data: Optional[pd.DataFrame] = None,
                    signal_data: Optional[pd.DataFrame] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    rebalance_frequency: str = 'daily') -> Dict[str, Any]:
        """
        Run the backtest simulation.
        
        Args:
            market_data: DataFrame with OHLCV data (MultiIndex: (date, symbol))
            volume_data: DataFrame with volume data
            signal_data: DataFrame with pre-computed signals
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary with backtest results
        """
        # Validate inputs
        if self.strategy_function is None and signal_data is None:
            raise ValueError("Either strategy_function or signal_data must be provided")
        
        # Filter data by date range
        if start_date:
            market_data = market_data[market_data.index.get_level_values(0) >= start_date]
        if end_date:
            market_data = market_data[market_data.index.get_level_values(0) <= end_date]
        
        self.market_data = market_data
        self.volume_data = volume_data
        
        # Get unique dates and symbols
        dates = market_data.index.get_level_values(0).unique().sort_values()
        symbols = market_data.index.get_level_values(1).unique()
        
        logger.info(f"Running backtest from {dates[0]} to {dates[-1]} with {len(symbols)} symbols")
        
        # Main simulation loop
        for i, date in enumerate(dates):
            self.current_timestamp = pd.Timestamp(date)
            
            # Get current market data
            current_data = market_data.loc[date]
            if isinstance(current_data, pd.Series):
                current_data = current_data.to_frame().T
            
            # Update portfolio with market data
            if 'close' in current_data.columns:
                close_prices = current_data['close']
                self.portfolio.update_market_data(close_prices, self.current_timestamp)
            
            # Generate signals (daily or at rebalance frequency)
            should_rebalance = self._should_rebalance(i, rebalance_frequency)
            
            if should_rebalance:
                if signal_data is not None:
                    # Use pre-computed signals
                    if date in signal_data.index:
                        signals = signal_data.loc[date]
                else:
                    # Generate signals using strategy function
                    historical_data = self._get_historical_data(date, lookback_days=252)
                    signals = self.strategy_function(historical_data)
                
                # Execute trades based on signals
                if 'signals' in locals() and signals is not None:
                    self._execute_signals(signals, current_data)
            
            # Log progress
            if i % 50 == 0 or i == len(dates) - 1:
                logger.info(f"Processed {i+1}/{len(dates)} days. Portfolio value: ${self.portfolio.total_value:,.2f}")
        
        # Calculate final performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(self.portfolio)
        
        # Prepare results
        results = {
            'portfolio': self.portfolio,
            'performance_metrics': performance_metrics,
            'trades': self.portfolio.trades,
            'positions': self.portfolio.positions,
            'daily_values': self.portfolio.daily_values,
            'daily_returns': self.portfolio.daily_returns,
            'timestamps': self.portfolio.timestamps
        }
        
        logger.info("Backtest completed successfully")
        logger.info(f"Final portfolio value: ${self.portfolio.total_value:,.2f}")
        logger.info(f"Total return: {performance_metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
        
        return results
    
    def _should_rebalance(self, day_index: int, frequency: str) -> bool:
        """Determine if portfolio should be rebalanced."""
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return day_index % 5 == 0  # Every 5 business days
        elif frequency == 'monthly':
            return day_index % 21 == 0  # Every ~21 business days
        else:
            return True
    
    def _get_historical_data(self, current_date: pd.Timestamp, 
                           lookback_days: int = 252) -> pd.DataFrame:
        """Get historical data for strategy calculation."""
        start_date = current_date - pd.Timedelta(days=lookback_days)
        
        # Filter market data
        historical = self.market_data[
            (self.market_data.index.get_level_values(0) >= start_date) &
            (self.market_data.index.get_level_values(0) < current_date)
        ]
        
        return historical
    
    def _execute_signals(self, signals: pd.Series, market_data: pd.DataFrame):
        """Execute trades based on signals."""
        if 'close' not in market_data.columns:
            logger.warning("No close prices available for trade execution")
            return
        
        close_prices = market_data['close']
        volume_data = market_data.get('volume', pd.Series(index=close_prices.index, data=1e6))
        
        for symbol in signals.index:
            if symbol not in close_prices.index:
                continue
            
            signal_strength = signals[symbol]
            if abs(signal_strength) < 0.01:  # Skip weak signals
                continue
            
            current_price = close_prices[symbol]
            current_volume = volume_data.get(symbol, 1e6)
            
            # Calculate target position size
            target_size = self.risk_manager.calculate_position_size(
                signal_strength, self.portfolio.total_value, current_price
            )
            
            # Calculate order quantity
            current_position = self.portfolio.get_position(symbol).quantity
            order_quantity = target_size - current_position
            
            if abs(order_quantity) < 1:  # Skip small orders
                continue
            
            # Create order
            order_side = OrderSide.BUY if order_quantity > 0 else OrderSide.SELL
            order = Order(
                symbol=symbol,
                side=order_side,
                quantity=abs(order_quantity),
                order_type=OrderType.MARKET,
                timestamp=self.current_timestamp
            )
            
            # Risk check
            if not self.risk_manager.check_risk_limits(order, self.portfolio, self.market_data):
                continue
            
            # Execute order
            self._execute_order(order, current_price, current_volume)
    
    def _execute_order(self, order: Order, market_price: float, avg_volume: float):
        """Execute a single order."""
        # Calculate transaction costs
        commission, slippage, market_impact = self.transaction_cost_model.calculate_total_costs(
            order, market_price, avg_volume
        )
        
        # Calculate execution price
        execution_price = market_price
        if order.side == OrderSide.BUY:
            execution_price += slippage + market_impact
        else:
            execution_price -= slippage + market_impact
        
        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity if order.side == OrderSide.BUY else -order.quantity,
            price=execution_price,
            timestamp=order.timestamp,
            trade_id=f"T_{order.order_id}",
            order_id=order.order_id,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact
        )
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            required_cash = trade.quantity * execution_price + commission
            if required_cash > self.portfolio.cash:
                logger.warning(f"Insufficient cash for order {order.order_id}")
                return
        
        # Add trade to portfolio
        self.portfolio.add_trade(trade)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.commission = commission
        
        self.portfolio.orders.append(order)
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save backtest results to file."""
        # Create summary for JSON serialization
        summary = {
            'performance_metrics': results['performance_metrics'],
            'portfolio_summary': results['portfolio'].get_portfolio_summary(),
            'num_trades': len(results['trades']),
            'backtest_period': {
                'start': results['timestamps'][0].isoformat() if results['timestamps'] else None,
                'end': results['timestamps'][-1].isoformat() if results['timestamps'] else None
            }
        }
        
        # Save as JSON
        json_path = f"{filepath}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save full results as pickle
        pickle_path = f"{filepath}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {json_path} and {pickle_path}")


# Example strategy function
def mean_reversion_strategy(historical_data: pd.DataFrame, 
                          lookback_window: int = 20,
                          entry_threshold: float = 2.0) -> pd.Series:
    """
    Example mean reversion strategy.
    
    Args:
        historical_data: Historical market data
        lookback_window: Lookback window for mean calculation
        entry_threshold: Z-score threshold for entry signals
        
    Returns:
        Series of signals for each symbol
    """
    if historical_data.empty:
        return pd.Series()
    
    # Get close prices
    close_data = historical_data['close'].unstack(level=1)
    
    # Calculate rolling mean and std
    rolling_mean = close_data.rolling(window=lookback_window).mean()
    rolling_std = close_data.rolling(window=lookback_window).std()
    
    # Calculate z-scores
    z_scores = (close_data - rolling_mean) / rolling_std
    
    # Generate signals (latest z-scores)
    latest_z_scores = z_scores.iloc[-1]
    
    # Create signals: -1 for oversold (buy), +1 for overbought (sell)
    signals = pd.Series(0.0, index=latest_z_scores.index)
    signals[latest_z_scores < -entry_threshold] = 0.5  # Buy signal
    signals[latest_z_scores > entry_threshold] = -0.5  # Sell signal
    
    return signals.dropna()


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    transaction_costs = TransactionCostModel(
        commission_rate=0.001,
        bid_ask_spread=0.0005,
        market_impact_model='square_root'
    )
    
    risk_manager = RiskManager(
        max_position_size=0.02,
        max_leverage=1.0,
        volatility_target=0.15
    )
    
    performance_analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Initialize backtester
    backtester = Backtester(
        initial_cash=1000000,
        transaction_cost_model=transaction_costs,
        risk_manager=risk_manager,
        performance_analyzer=performance_analyzer
    )
    
    # Set strategy
    backtester.set_strategy(mean_reversion_strategy)
    
    print("Backtester initialized successfully!")
    print("Ready to run backtests with comprehensive performance analysis.")