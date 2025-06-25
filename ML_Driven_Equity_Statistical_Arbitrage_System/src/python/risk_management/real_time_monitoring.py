# Real-time compliance and risk monitoring system
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
import asyncio

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"

class RiskMetric(Enum):
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    LEVERAGE = "leverage"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    BETA = "beta"
    CORRELATION = "correlation"

@dataclass
class RiskAlert:
    timestamp: datetime
    metric: RiskMetric
    level: AlertLevel
    current_value: float
    limit_value: float
    breach_percentage: float
    message: str
    recommended_action: str

class RealTimeRiskMonitor:
    """
    Real-time risk monitoring and compliance system for institutional trading
    """
    
    def __init__(self, risk_limits: Dict[RiskMetric, float]):
        self.risk_limits = risk_limits
        self.current_metrics = {}
        self.alert_history = []
        self.position_history = []
        self.compliance_status = {}
        
        # Risk calculation parameters
        self.var_confidence_levels = [0.95, 0.99]
        self.lookback_period = 252  # 1 year
        self.monitoring_frequency = timedelta(minutes=1)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize compliance rules
        self.setup_compliance_rules()
    
    def setup_compliance_rules(self):
        """Setup regulatory and internal compliance rules"""
        self.compliance_rules = {
            'position_limits': {
                'max_single_position': 0.05,  # 5% max in any single position
                'max_sector_exposure': 0.20,   # 20% max in any sector
                'max_country_exposure': 0.30,  # 30% max in any country
            },
            'leverage_limits': {
                'gross_leverage': 3.0,         # 3x gross leverage limit
                'net_leverage': 2.0,           # 2x net leverage limit
            },
            'liquidity_requirements': {
                'min_liquidity_buffer': 0.05,  # 5% cash buffer
                'max_illiquid_positions': 0.15, # 15% max in illiquid assets
            },
            'var_limits': {
                'daily_var_95': 0.02,          # 2% daily VaR at 95% confidence
                'daily_var_99': 0.03,          # 3% daily VaR at 99% confidence
            },
            'correlation_limits': {
                'max_portfolio_correlation': 0.80,  # Max correlation to benchmark
                'max_position_correlation': 0.60,   # Max correlation between positions
            }
        }
    
    async def calculate_portfolio_var(self, 
                                    returns: pd.DataFrame, 
                                    weights: pd.Series, 
                                    confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk using historical simulation"""
        try:
            # Ensure data alignment
            common_assets = returns.columns.intersection(weights.index)
            returns_aligned = returns[common_assets].fillna(0)
            weights_aligned = weights[common_assets]
            
            # Calculate portfolio returns
            portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
            
            # Historical VaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            return abs(var)  # Return positive value
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    async def calculate_expected_shortfall(self, 
                                         returns: pd.DataFrame, 
                                         weights: pd.Series, 
                                         confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            common_assets = returns.columns.intersection(weights.index)
            returns_aligned = returns[common_assets].fillna(0)
            weights_aligned = weights[common_assets]
            
            portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # Expected Shortfall is the mean of returns below VaR
            tail_returns = portfolio_returns[portfolio_returns <= var]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var
            
            return abs(expected_shortfall)
            
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    def calculate_leverage_metrics(self, positions: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate various leverage metrics"""
        portfolio_value = sum(abs(pos * prices.get(symbol, 0)) for symbol, pos in positions.items())
        
        long_exposure = sum(pos * prices.get(symbol, 0) for symbol, pos in positions.items() if pos > 0)
        short_exposure = sum(abs(pos * prices.get(symbol, 0)) for symbol, pos in positions.items() if pos < 0)
        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        
        return {
            'gross_leverage': gross_exposure / portfolio_value if portfolio_value > 0 else 0,
            'net_leverage': abs(net_exposure) / portfolio_value if portfolio_value > 0 else 0,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure
        }
    
    def calculate_concentration_risk(self, positions: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        position_values = {symbol: abs(pos * prices.get(symbol, 0)) 
                          for symbol, pos in positions.items()}
        total_value = sum(position_values.values())
        
        if total_value == 0:
            return {'max_position_weight': 0, 'top_5_concentration': 0, 'herfindahl_index': 0}
        
        # Position weights
        weights = {symbol: value / total_value for symbol, value in position_values.items()}
        
        # Concentration metrics
        max_position_weight = max(weights.values()) if weights else 0
        
        # Top 5 concentration
        top_5_weights = sorted(weights.values(), reverse=True)[:5]
        top_5_concentration = sum(top_5_weights)
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(w**2 for w in weights.values())
        
        return {
            'max_position_weight': max_position_weight,
            'top_5_concentration': top_5_concentration,
            'herfindahl_index': herfindahl_index
        }
    
    async def check_risk_limits(self, 
                              positions: Dict[str, float], 
                              prices: Dict[str, float],
                              returns_data: pd.DataFrame) -> List[RiskAlert]:
        """Comprehensive risk limit checking"""
        alerts = []
        
        # Calculate current weights
        position_values = {symbol: pos * prices.get(symbol, 0) for symbol, pos in positions.items()}
        total_value = sum(abs(value) for value in position_values.values())
        weights = pd.Series({symbol: value / total_value for symbol, value in position_values.items()}) if total_value > 0 else pd.Series()
        
        # VaR checks
        for confidence_level in self.var_confidence_levels:
            var_metric = RiskMetric.VAR_95 if confidence_level == 0.95 else RiskMetric.VAR_99
            
            if var_metric in self.risk_limits:
                current_var = await self.calculate_portfolio_var(returns_data, weights, confidence_level)
                limit = self.risk_limits[var_metric]
                
                if current_var > limit:
                    breach_percentage = ((current_var - limit) / limit) * 100
                    alert = RiskAlert(
                        timestamp=datetime.now(),
                        metric=var_metric,
                        level=AlertLevel.CRITICAL if breach_percentage > 50 else AlertLevel.WARNING,
                        current_value=current_var,
                        limit_value=limit,
                        breach_percentage=breach_percentage,
                        message=f"VaR {confidence_level*100:.0f}% breach: {current_var:.2%} > {limit:.2%}",
                        recommended_action="Reduce position sizes or hedge exposure"
                    )
                    alerts.append(alert)
        
        # Leverage checks
        leverage_metrics = self.calculate_leverage_metrics(positions, prices)
        
        if RiskMetric.LEVERAGE in self.risk_limits:
            gross_leverage = leverage_metrics['gross_leverage']
            limit = self.risk_limits[RiskMetric.LEVERAGE]
            
            if gross_leverage > limit:
                breach_percentage = ((gross_leverage - limit) / limit) * 100
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    metric=RiskMetric.LEVERAGE,
                    level=AlertLevel.BREACH if breach_percentage > 20 else AlertLevel.WARNING,
                    current_value=gross_leverage,
                    limit_value=limit,
                    breach_percentage=breach_percentage,
                    message=f"Leverage breach: {gross_leverage:.2f}x > {limit:.2f}x",
                    recommended_action="Reduce position sizes immediately"
                )
                alerts.append(alert)
        
        # Concentration checks
        concentration_metrics = self.calculate_concentration_risk(positions, prices)
        
        if RiskMetric.CONCENTRATION in self.risk_limits:
            max_concentration = concentration_metrics['max_position_weight']
            limit = self.risk_limits[RiskMetric.CONCENTRATION]
            
            if max_concentration > limit:
                breach_percentage = ((max_concentration - limit) / limit) * 100
                alert = RiskAlert(
                    timestamp=datetime.now(),
                    metric=RiskMetric.CONCENTRATION,
                    level=AlertLevel.WARNING,
                    current_value=max_concentration,
                    limit_value=limit,
                    breach_percentage=breach_percentage,
                    message=f"Concentration risk: {max_concentration:.2%} > {limit:.2%}",
                    recommended_action="Diversify position or reduce concentration"
                )
                alerts.append(alert)
        
        return alerts
    
    async def real_time_monitoring_loop(self, 
                                      get_positions_func,
                                      get_prices_func,
                                      get_returns_func):
        """Main real-time monitoring loop"""
        while True:
            try:
                # Get current data
                positions = await get_positions_func()
                prices = await get_prices_func()
                returns_data = await get_returns_func()
                
                # Check risk limits
                alerts = await self.check_risk_limits(positions, prices, returns_data)
                
                # Process alerts
                for alert in alerts:
                    self.alert_history.append(alert)
                    await self.handle_alert(alert)
                
                # Update metrics
                await self.update_current_metrics(positions, prices, returns_data)
                
                # Wait before next check
                await asyncio.sleep(self.monitoring_frequency.total_seconds())
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def handle_alert(self, alert: RiskAlert):
        """Handle risk alerts with appropriate actions"""
        # Log alert
        self.logger.warning(f"Risk Alert: {alert.message}")
        
        # Critical alerts require immediate action
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.BREACH]:
            await self.emergency_risk_reduction(alert)
        
        # Send notifications (in production, this would integrate with alerting systems)
        await self.send_alert_notification(alert)
    
    async def emergency_risk_reduction(self, alert: RiskAlert):
        """Emergency risk reduction procedures"""
        self.logger.critical(f"Emergency risk reduction triggered: {alert.message}")
        
        # In production, this would trigger automatic position reduction
        # For now, we log the recommended actions
        actions = {
            RiskMetric.VAR_95: "Reduce position sizes by 20%",
            RiskMetric.VAR_99: "Reduce position sizes by 30%",
            RiskMetric.LEVERAGE: "Close 50% of most leveraged positions",
            RiskMetric.CONCENTRATION: "Reduce largest position by 30%"
        }
        
        action = actions.get(alert.metric, "Manual intervention required")
        self.logger.critical(f"Recommended action: {action}")
    
    async def send_alert_notification(self, alert: RiskAlert):
        """Send alert notifications to relevant parties"""
        # In production, this would integrate with email, SMS, or messaging systems
        notification_message = f"""
Risk Alert: {alert.level.value.upper()}
Metric: {alert.metric.value}
Current Value: {alert.current_value:.4f}
Limit: {alert.limit_value:.4f}
Breach: {alert.breach_percentage:.1f}%
Recommended Action: {alert.recommended_action}
Timestamp: {alert.timestamp}
"""
        
        # For demonstration, we'll just log
        self.logger.info(f"Alert notification sent: {notification_message}")
    
    async def update_current_metrics(self, positions, prices, returns_data):
        """Update current risk metrics"""
        # Calculate and store current metrics for dashboard/reporting
        position_values = {symbol: pos * prices.get(symbol, 0) for symbol, pos in positions.items()}
        total_value = sum(abs(value) for value in position_values.values())
        weights = pd.Series({symbol: value / total_value for symbol, value in position_values.items()}) if total_value > 0 else pd.Series()
        
        self.current_metrics = {
            'timestamp': datetime.now(),
            'portfolio_value': total_value,
            'var_95': await self.calculate_portfolio_var(returns_data, weights, 0.95),
            'var_99': await self.calculate_portfolio_var(returns_data, weights, 0.99),
            'leverage': self.calculate_leverage_metrics(positions, prices),
            'concentration': self.calculate_concentration_risk(positions, prices)
        }
    
    def generate_risk_dashboard_data(self) -> Dict:
        """Generate data for risk dashboard"""
        return {
            'current_metrics': self.current_metrics,
            'risk_limits': {metric.value: limit for metric, limit in self.risk_limits.items()},
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'metric': alert.metric.value,
                    'level': alert.level.value,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'limit_value': alert.limit_value
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ],
            'compliance_status': self.compliance_status
        }

# Example usage for demonstration
async def demonstrate_risk_monitoring():
    """Demonstrate real-time risk monitoring capabilities"""
    
    # Setup risk limits
    risk_limits = {
        RiskMetric.VAR_95: 0.02,  # 2% daily VaR
        RiskMetric.VAR_99: 0.03,  # 3% daily VaR  
        RiskMetric.LEVERAGE: 2.0,  # 2x leverage limit
        RiskMetric.CONCENTRATION: 0.10,  # 10% max position
    }
    
    # Initialize monitor
    monitor = RealTimeRiskMonitor(risk_limits)
    
    # Mock data functions
    async def get_positions():
        return {'AAPL': 1000, 'MSFT': 1500, 'GOOGL': 500}
    
    async def get_prices():
        return {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
    
    async def get_returns():
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        return pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'MSFT': np.random.normal(0.0008, 0.018, 252),
            'GOOGL': np.random.normal(0.0012, 0.025, 252)
        }, index=dates)
    
    # Test risk checking
    positions = await get_positions()
    prices = await get_prices()
    returns_data = await get_returns()
    
    alerts = await monitor.check_risk_limits(positions, prices, returns_data)
    
    print("=== RISK MONITORING DEMONSTRATION ===")
    print(f"Generated {len(alerts)} alerts")
    
    for alert in alerts:
        print(f"Alert: {alert.message}")
        print(f"Action: {alert.recommended_action}")
    
    # Generate dashboard data
    dashboard_data = monitor.generate_risk_dashboard_data()
    print(f"\nDashboard metrics updated: {list(dashboard_data.keys())}")
    
    return monitor

if __name__ == "__main__":
    asyncio.run(demonstrate_risk_monitoring())
