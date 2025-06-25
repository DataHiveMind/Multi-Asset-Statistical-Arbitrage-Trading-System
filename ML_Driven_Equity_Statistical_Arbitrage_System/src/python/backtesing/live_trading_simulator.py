# Live trading simulation with real market conditions
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None  # None for market orders
    order_type: str = "market"  # 'market' or 'limit'
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    commission: float = 0.0

class LiveTradingSimulator:
    """
    Production-grade trading simulator with realistic market conditions
    """
    
    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.positions = {}  # {symbol: quantity}
        self.portfolio_value_history = []
        self.orders_history = []
        self.trades_history = []
        self.performance_metrics = {}
        
        # Market data cache
        self.market_data_cache = {}
        self.last_update = None
        
        # Risk limits
        self.max_position_size = 0.05  # 5% max position
        self.max_portfolio_leverage = 2.0
        self.stop_loss_threshold = -0.02  # 2% stop loss
        
        self.logger = logging.getLogger(__name__)
    
    async def update_market_data(self, symbols: List[str]):
        """Update real-time market data"""
        try:
            # In production, this would connect to a real data feed
            data = yf.download(symbols, period="1d", interval="1m")
            if len(symbols) == 1:
                data = {symbols[0]: data}
            else:
                for symbol in symbols:
                    self.market_data_cache[symbol] = data['Close'][symbol]
            
            self.last_update = datetime.now()
            self.logger.info(f"Updated market data for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        if symbol in self.market_data_cache:
            return float(self.market_data_cache[symbol].iloc[-1])
        return None
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        cash = self.current_capital
        positions_value = 0
        
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                current_price = self.get_current_price(symbol)
                if current_price:
                    positions_value += quantity * current_price
        
        return cash + positions_value
    
    def check_risk_limits(self, order: Order) -> bool:
        """Check if order violates risk limits"""
        current_price = self.get_current_price(order.symbol)
        if not current_price:
            return False
        
        # Calculate position value
        order_value = abs(order.quantity * current_price)
        portfolio_value = self.calculate_portfolio_value()
        
        # Check position size limit
        if order_value > portfolio_value * self.max_position_size:
            self.logger.warning(f"Order exceeds position size limit: {order.symbol}")
            return False
        
        # Check leverage limit
        total_exposure = sum(abs(qty * self.get_current_price(sym)) 
                           for sym, qty in self.positions.items() 
                           if self.get_current_price(sym))
        total_exposure += order_value
        
        if total_exposure > portfolio_value * self.max_portfolio_leverage:
            self.logger.warning(f"Order exceeds leverage limit")
            return False
        
        return True
    
    async def place_order(self, order: Order) -> bool:
        """Place and execute order with realistic market simulation"""
        order.timestamp = datetime.now()
        
        # Risk checks
        if not self.check_risk_limits(order):
            order.status = OrderStatus.REJECTED
            self.orders_history.append(order)
            return False
        
        # Get current market price
        current_price = self.get_current_price(order.symbol)
        if not current_price:
            order.status = OrderStatus.REJECTED
            self.orders_history.append(order)
            return False
        
        # Simulate market impact and slippage
        slippage = self.calculate_slippage(order, current_price)
        fill_price = current_price + slippage
        
        # Calculate commission
        commission = abs(order.quantity * fill_price * self.commission_rate)
        
        # Execute trade
        if order.side == 'buy':
            required_capital = order.quantity * fill_price + commission
            if required_capital <= self.current_capital:
                self.current_capital -= required_capital
                self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
            else:
                order.status = OrderStatus.REJECTED
                self.orders_history.append(order)
                return False
        else:  # sell
            current_position = self.positions.get(order.symbol, 0)
            if abs(order.quantity) <= current_position:
                self.current_capital += order.quantity * fill_price - commission
                self.positions[order.symbol] = current_position - order.quantity
            else:
                order.status = OrderStatus.REJECTED
                self.orders_history.append(order)
                return False
        
        # Record successful trade
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.commission = commission
        
        self.orders_history.append(order)
        self.trades_history.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'portfolio_value': self.calculate_portfolio_value()
        })
        
        self.logger.info(f"Order filled: {order.symbol} {order.side} {order.quantity} @ {fill_price:.4f}")
        return True
    
    def calculate_slippage(self, order: Order, current_price: float) -> float:
        """Calculate realistic slippage based on order size and market conditions"""
        # Simple slippage model - in production, this would be more sophisticated
        base_slippage = 0.0005  # 5 basis points
        size_impact = abs(order.quantity) / 10000  # Simplified size impact
        
        slippage = base_slippage + size_impact
        
        # Apply slippage direction
        if order.side == 'buy':
            return slippage * current_price
        else:
            return -slippage * current_price
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.trades_history:
            return {}
        
        df = pd.DataFrame(self.trades_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Portfolio value over time
        portfolio_values = df['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trading metrics
        total_trades = len(self.trades_history)
        total_commission = sum(trade['commission'] for trade in self.trades_history)
        
        # Win rate (simplified)
        winning_trades = sum(1 for i in range(1, len(portfolio_values)) 
                           if portfolio_values.iloc[i] > portfolio_values.iloc[i-1])
        win_rate = winning_trades / (len(portfolio_values) - 1) if len(portfolio_values) > 1 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'win_rate': win_rate,
            'current_portfolio_value': portfolio_values.iloc[-1],
            'profit_loss': portfolio_values.iloc[-1] - self.initial_capital
        }
        
        return self.performance_metrics
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = f"""
=== LIVE TRADING SIMULATION PERFORMANCE REPORT ===

Portfolio Summary:
- Initial Capital: ${self.initial_capital:,.2f}
- Current Value: ${metrics.get('current_portfolio_value', 0):,.2f}
- Total P&L: ${metrics.get('profit_loss', 0):,.2f}
- Total Return: {metrics.get('total_return', 0):.2%}

Risk-Adjusted Returns:
- Annualized Return: {metrics.get('annualized_return', 0):.2%}
- Volatility: {metrics.get('volatility', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}

Trading Activity:
- Total Trades: {metrics.get('total_trades', 0)}
- Total Commission: ${metrics.get('total_commission', 0):,.2f}
- Win Rate: {metrics.get('win_rate', 0):.2%}

Current Positions:
"""
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                current_price = self.get_current_price(symbol)
                if current_price:
                    position_value = quantity * current_price
                    report += f"- {symbol}: {quantity:,.0f} shares @ ${current_price:.2f} = ${position_value:,.2f}\n"
        
        return report

# Usage example for resume demonstration
async def demonstrate_live_trading():
    """Demonstration of live trading capabilities"""
    simulator = LiveTradingSimulator(initial_capital=1000000)
    
    # Symbols for demonstration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Update market data
    await simulator.update_market_data(symbols)
    
    # Generate some sample trades based on signals
    orders = [
        Order('AAPL', 'buy', 100),
        Order('MSFT', 'buy', 150),
        Order('GOOGL', 'buy', 50),
        # Add some sells later
        Order('AAPL', 'sell', 50),
    ]
    
    # Execute orders
    for order in orders:
        await simulator.place_order(order)
        await asyncio.sleep(1)  # Simulate time between trades
    
    # Generate performance report
    report = simulator.generate_performance_report()
    print(report)
    
    return simulator.performance_metrics

if __name__ == "__main__":
    # Run demonstration
    metrics = asyncio.run(demonstrate_live_trading())
