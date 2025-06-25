# Advanced performance attribution for institutional-grade analysis
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class PerformanceAttributionEngine:
    """
    Institutional-grade performance attribution and risk decomposition
    """
    
    def __init__(self, benchmark_returns: pd.Series, factor_returns: pd.DataFrame = None):
        self.benchmark_returns = benchmark_returns
        self.factor_returns = factor_returns  # Fama-French factors, etc.
        self.attribution_results = {}
        
    def brinson_attribution(self, 
                          portfolio_weights: pd.DataFrame,
                          benchmark_weights: pd.DataFrame,
                          asset_returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Brinson performance attribution analysis
        """
        # Ensure all data is aligned
        common_assets = portfolio_weights.columns.intersection(
            benchmark_weights.columns).intersection(asset_returns.columns)
        
        pw = portfolio_weights[common_assets]
        bw = benchmark_weights[common_assets]
        returns = asset_returns[common_assets]
        
        # Calculate attribution components
        allocation_effect = ((pw - bw) * returns.mean()).sum(axis=1)
        selection_effect = (bw * (returns - returns.mean())).sum(axis=1)
        interaction_effect = ((pw - bw) * (returns - returns.mean())).sum(axis=1)
        
        total_active_return = allocation_effect + selection_effect + interaction_effect
        
        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_active_return': total_active_return
        }
    
    def factor_attribution(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        Factor-based performance attribution using regression analysis
        """
        if self.factor_returns is None:
            # Create synthetic factors for demonstration
            market_factor = self.benchmark_returns
            size_factor = pd.Series(np.random.normal(0, 0.02, len(portfolio_returns)), 
                                  index=portfolio_returns.index)
            value_factor = pd.Series(np.random.normal(0, 0.015, len(portfolio_returns)), 
                                   index=portfolio_returns.index)
            momentum_factor = pd.Series(np.random.normal(0, 0.01, len(portfolio_returns)), 
                                      index=portfolio_returns.index)
            
            factors = pd.DataFrame({
                'Market': market_factor,
                'Size': size_factor,
                'Value': value_factor,
                'Momentum': momentum_factor
            })
        else:
            factors = self.factor_returns
        
        # Align data
        common_dates = portfolio_returns.index.intersection(factors.index)
        y = portfolio_returns[common_dates]
        X = factors.loc[common_dates]
        
        # Run regression
        reg = LinearRegression().fit(X, y)
        
        # Calculate factor contributions
        factor_exposures = pd.Series(reg.coef_, index=X.columns)
        factor_returns_avg = X.mean()
        factor_contributions = factor_exposures * factor_returns_avg
        
        # Calculate attribution
        total_return = y.mean()
        alpha = reg.intercept_
        factor_return = factor_contributions.sum()
        
        attribution = {
            'total_return': total_return,
            'alpha': alpha,
            'factor_return': factor_return,
            'factor_contributions': factor_contributions.to_dict(),
            'factor_exposures': factor_exposures.to_dict(),
            'r_squared': reg.score(X, y)
        }
        
        return attribution
    
    def risk_decomposition(self, 
                          portfolio_returns: pd.Series,
                          asset_returns: pd.DataFrame,
                          portfolio_weights: pd.DataFrame) -> Dict[str, any]:
        """
        Decompose portfolio risk into systematic and idiosyncratic components
        """
        # Calculate portfolio variance
        portfolio_var = portfolio_returns.var()
        
        # Asset covariance matrix
        cov_matrix = asset_returns.cov()
        
        # Average portfolio weights
        avg_weights = portfolio_weights.mean()
        
        # Portfolio variance decomposition
        systematic_var = np.dot(avg_weights, np.dot(cov_matrix, avg_weights))
        
        # Principal component analysis for systematic risk factors
        pca = PCA()
        pca.fit(asset_returns.fillna(0))
        
        # Calculate component contributions
        pc_loadings = pd.DataFrame(
            pca.components_[:5].T,  # First 5 components
            columns=[f'PC{i+1}' for i in range(5)],
            index=asset_returns.columns
        )
        
        # Risk contributions by component
        risk_contributions = {}
        for i, pc in enumerate(pc_loadings.columns):
            component_exposure = (avg_weights * pc_loadings[pc]).sum()
            component_var = pca.explained_variance_[i]
            risk_contributions[pc] = component_exposure**2 * component_var
        
        # Marginal contribution to risk (MCTR)
        marginal_contributions = np.dot(cov_matrix, avg_weights) / np.sqrt(systematic_var)
        component_contributions = avg_weights * marginal_contributions
        
        return {
            'portfolio_variance': portfolio_var,
            'systematic_variance': systematic_var,
            'idiosyncratic_variance': portfolio_var - systematic_var,
            'systematic_risk_ratio': systematic_var / portfolio_var,
            'pc_risk_contributions': risk_contributions,
            'marginal_contributions': pd.Series(marginal_contributions, index=asset_returns.columns),
            'component_contributions': pd.Series(component_contributions, index=asset_returns.columns),
            'explained_variance_ratio': pca.explained_variance_ratio_[:5]
        }
    
    def tracking_error_analysis(self, 
                               portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Detailed tracking error analysis
        """
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns
        
        # Align returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns[common_dates]
        bench_ret = benchmark_returns[common_dates]
        
        # Active returns
        active_returns = port_ret - bench_ret
        
        # Tracking error metrics
        tracking_error = active_returns.std() * np.sqrt(252)  # Annualized
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        # Up/Down capture ratios
        up_market = bench_ret > 0
        down_market = bench_ret < 0
        
        up_capture = (port_ret[up_market].mean() / bench_ret[up_market].mean()) if bench_ret[up_market].mean() != 0 else 1
        down_capture = (port_ret[down_market].mean() / bench_ret[down_market].mean()) if bench_ret[down_market].mean() != 0 else 1
        
        # Beta and correlation
        beta = np.cov(port_ret, bench_ret)[0, 1] / np.var(bench_ret)
        correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'active_return': active_returns.mean() * 252,  # Annualized
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'beta': beta,
            'correlation': correlation,
            'hit_rate': (active_returns > 0).mean()  # Percentage of periods beating benchmark
        }
    
    def style_analysis(self, 
                      portfolio_returns: pd.Series,
                      style_indices: pd.DataFrame) -> Dict[str, any]:
        """
        Returns-based style analysis (RBSA) following Sharpe (1992)
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(style_indices.index)
        y = portfolio_returns[common_dates]
        X = style_indices.loc[common_dates]
        
        # Constrained optimization for style weights
        def objective(weights):
            predicted_returns = X @ weights
            return np.sum((y - predicted_returns)**2)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, 1) for _ in range(len(X.columns))]
        
        # Initial guess
        x0 = np.array([1/len(X.columns)] * len(X.columns))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            style_weights = pd.Series(result.x, index=X.columns)
            
            # Calculate R-squared
            predicted_returns = X @ style_weights
            ss_res = np.sum((y - predicted_returns)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Selection and timing metrics
            selection_return = y.mean() - predicted_returns.mean()
            
            return {
                'style_weights': style_weights.to_dict(),
                'r_squared': r_squared,
                'selection_return': selection_return * 252,  # Annualized
                'tracking_error': np.sqrt(ss_res / len(y)) * np.sqrt(252),
                'successful_optimization': True
            }
        else:
            return {'successful_optimization': False, 'error': result.message}
    
    def generate_attribution_report(self, 
                                  portfolio_returns: pd.Series,
                                  asset_returns: pd.DataFrame = None,
                                  portfolio_weights: pd.DataFrame = None) -> str:
        """
        Generate comprehensive attribution report
        """
        report = "=== PERFORMANCE ATTRIBUTION ANALYSIS ===\n\n"
        
        # Basic performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + portfolio_returns.mean())**252 - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        report += f"Portfolio Performance Summary:\n"
        report += f"- Total Return: {total_return:.2%}\n"
        report += f"- Annualized Return: {annualized_return:.2%}\n"
        report += f"- Volatility: {volatility:.2%}\n"
        report += f"- Sharpe Ratio: {sharpe_ratio:.3f}\n\n"
        
        # Factor attribution
        factor_attr = self.factor_attribution(portfolio_returns)
        report += f"Factor Attribution:\n"
        report += f"- Alpha: {factor_attr['alpha']*252:.2%}\n"
        report += f"- Factor Return: {factor_attr['factor_return']*252:.2%}\n"
        report += f"- R-squared: {factor_attr['r_squared']:.3f}\n"
        
        report += f"\nFactor Contributions:\n"
        for factor, contrib in factor_attr['factor_contributions'].items():
            report += f"- {factor}: {contrib*252:.2%}\n"
        
        # Tracking error analysis
        te_analysis = self.tracking_error_analysis(portfolio_returns)
        report += f"\nTracking Error Analysis:\n"
        report += f"- Tracking Error: {te_analysis['tracking_error']:.2%}\n"
        report += f"- Information Ratio: {te_analysis['information_ratio']:.3f}\n"
        report += f"- Beta: {te_analysis['beta']:.3f}\n"
        report += f"- Up Capture: {te_analysis['up_capture_ratio']:.3f}\n"
        report += f"- Down Capture: {te_analysis['down_capture_ratio']:.3f}\n"
        
        # Risk decomposition (if weights provided)
        if asset_returns is not None and portfolio_weights is not None:
            risk_decomp = self.risk_decomposition(portfolio_returns, asset_returns, portfolio_weights)
            report += f"\nRisk Decomposition:\n"
            report += f"- Systematic Risk Ratio: {risk_decomp['systematic_risk_ratio']:.2%}\n"
            report += f"- Idiosyncratic Risk Ratio: {1-risk_decomp['systematic_risk_ratio']:.2%}\n"
        
        return report

# Example usage for resume demonstration
def demonstrate_attribution_analysis():
    """Demonstrate advanced attribution capabilities"""
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # Sample portfolio and benchmark returns
    np.random.seed(42)
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
    portfolio_returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
    
    # Initialize attribution engine
    attr_engine = PerformanceAttributionEngine(benchmark_returns)
    
    # Generate comprehensive report
    report = attr_engine.generate_attribution_report(portfolio_returns)
    print(report)
    
    return attr_engine

if __name__ == "__main__":
    demonstrate_attribution_analysis()
