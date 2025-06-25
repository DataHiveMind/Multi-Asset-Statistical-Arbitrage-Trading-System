# Advanced risk management with modern portfolio theory and risk parity
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """
    Advanced risk management system with multiple risk models and optimization techniques
    """
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
        self.risk_models = {}
        
    def calculate_var_cvar(self, returns: pd.Series, method: str = 'historical') -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk
        """
        if method == 'historical':
            var = returns.quantile(self.confidence_level)
            cvar = returns[returns <= var].mean()
            
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            var = norm.ppf(self.confidence_level, mu, sigma)
            cvar = mu - sigma * norm.pdf(norm.ppf(self.confidence_level)) / self.confidence_level
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            random_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
            var = np.percentile(random_returns, self.confidence_level * 100)
            cvar = random_returns[random_returns <= var].mean()
            
        return var, cvar
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'current_drawdown': drawdown.iloc[-1]
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate the duration of maximum drawdown"""
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for dd in is_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def calculate_risk_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive risk metrics for multiple assets
        """
        risk_metrics = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            # Basic metrics
            volatility = asset_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (asset_returns.mean() * 252) / volatility
            
            # Risk metrics
            var_hist, cvar_hist = self.calculate_var_cvar(asset_returns, 'historical')
            var_param, cvar_param = self.calculate_var_cvar(asset_returns, 'parametric')
            
            # Drawdown metrics
            dd_metrics = self.calculate_maximum_drawdown(asset_returns)
            
            # Tail risk metrics
            skewness = asset_returns.skew()
            kurtosis = asset_returns.kurtosis()
            
            risk_metrics[asset] = {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'var_historical': var_hist,
                'cvar_historical': cvar_hist,
                'var_parametric': var_param,
                'cvar_parametric': cvar_param,
                'max_drawdown': dd_metrics['max_drawdown'],
                'max_dd_duration': dd_metrics['max_drawdown_duration'],
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        
        return pd.DataFrame(risk_metrics).T

class ModernPortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple objective functions
    """
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.n_assets = len(returns.columns)
        
    def optimize_mean_variance(self, target_return: Optional[float] = None) -> Dict[str, any]:
        """
        Classic Markowitz mean-variance optimization
        """
        # Decision variables
        weights = cp.Variable(self.n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only (can be modified for long-short)
        ]
        
        # Target return constraint (if specified)
        if target_return is not None:
            portfolio_return = self.mean_returns.values @ weights
            constraints.append(portfolio_return >= target_return)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == cp.OPTIMAL:
            optimal_weights = pd.Series(weights.value, index=self.returns.columns)
            portfolio_return = (optimal_weights @ self.mean_returns)
            portfolio_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights)
            sharpe_ratio = portfolio_return / portfolio_vol
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def optimize_risk_parity(self) -> Dict[str, any]:
        """
        Risk parity optimization - equal risk contribution from each asset
        """
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = self.cov_matrix @ weights / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize the sum of squared deviations from equal risk contribution
            target_contrib = portfolio_vol / self.n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (long-only)
        bounds = [(0.01, 0.5) for _ in range(self.n_assets)]  # Min 1%, max 50%
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=self.returns.columns)
            portfolio_return = optimal_weights @ self.mean_returns
            portfolio_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights)
            sharpe_ratio = portfolio_return / portfolio_vol
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': result.message}
    
    def optimize_max_diversification(self) -> Dict[str, any]:
        """
        Maximum diversification portfolio
        """
        # Diversification ratio = (w^T * sigma) / sqrt(w^T * Cov * w)
        # where sigma is vector of individual volatilities
        individual_vols = np.sqrt(np.diag(self.cov_matrix))
        
        def negative_diversification_ratio(weights):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            weighted_avg_vol = weights @ individual_vols
            return -weighted_avg_vol / portfolio_vol  # Negative for maximization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        # Initial guess
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            negative_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=self.returns.columns)
            portfolio_return = optimal_weights @ self.mean_returns
            portfolio_vol = np.sqrt(optimal_weights @ self.cov_matrix @ optimal_weights)
            sharpe_ratio = portfolio_return / portfolio_vol
            diversification_ratio = -result.fun
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': result.message}

class DynamicRiskManager:
    """
    Dynamic risk management with regime detection and adaptive position sizing
    """
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.risk_regimes = {}
        
    def detect_volatility_regime(self, returns: pd.Series) -> pd.Series:
        """
        Detect volatility regimes using rolling statistics
        """
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        vol_percentile = rolling_vol.rolling(window=self.lookback_window).rank(pct=True)
        
        # Define regimes: Low (0-33%), Medium (33-67%), High (67-100%)
        regimes = pd.cut(vol_percentile, bins=[0, 0.33, 0.67, 1.0], 
                        labels=['Low_Vol', 'Medium_Vol', 'High_Vol'])
        
        return regimes
    
    def adaptive_position_sizing(self, 
                               signal_strength: float, 
                               current_vol: float, 
                               target_vol: float = 0.10,
                               max_position: float = 0.05) -> float:
        """
        Calculate position size based on volatility targeting and signal strength
        """
        # Volatility adjustment
        vol_adjustment = target_vol / current_vol
        
        # Signal-based sizing
        base_position = signal_strength * max_position
        
        # Combine adjustments
        adjusted_position = base_position * vol_adjustment
        
        # Apply position limits
        return np.clip(adjusted_position, -max_position, max_position)
