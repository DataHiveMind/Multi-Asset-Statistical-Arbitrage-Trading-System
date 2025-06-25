"""
Feature Engineering for Alpha Signal Generation

This module implements functions and classes for preprocessing raw market data
and extracting relevant features for machine learning models in statistical
arbitrage strategies.

Classes:
    - FeaturePipeline: Main pipeline for feature engineering
    - TechnicalIndicators: Collection of technical analysis indicators
    - StatisticalFeatures: Statistical transformations and rolling metrics
    - DataPreprocessor: Data cleaning and preprocessing utilities

Functions:
    - Various technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Statistical transformations (normalization, standardization, etc.)
    - Rolling statistics and time-based features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Collection of technical analysis indicators commonly used in trading.
    """
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series (typically close prices)
            window: Lookback period for RSI calculation
            
        Returns:
            RSI values (0-100)
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.
        
        Returns:
            DataFrame with MACD line, signal line, and histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Returns:
            DataFrame with upper band, lower band, and middle band (SMA)
        """
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': upper_band - lower_band,
            'position': (data - lower_band) / (upper_band - lower_band)
        })
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Returns:
            DataFrame with %K and %D lines
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range - measure of volatility.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R oscillator.
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    @staticmethod
    def momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """
        Price momentum indicator.
        """
        return data.diff(window)
    
    @staticmethod
    def rate_of_change(data: pd.Series, window: int = 10) -> pd.Series:
        """
        Rate of Change (ROC) indicator.
        """
        return (data.diff(window) / data.shift(window)) * 100


class StatisticalFeatures:
    """
    Collection of statistical transformations and rolling metrics.
    """
    
    @staticmethod
    def rolling_statistics(data: pd.Series, window: int) -> pd.DataFrame:
        """
        Calculate comprehensive rolling statistics.
        """
        rolling = data.rolling(window=window)
        
        return pd.DataFrame({
            'mean': rolling.mean(),
            'std': rolling.std(),
            'var': rolling.var(),
            'min': rolling.min(),
            'max': rolling.max(),
            'median': rolling.median(),
            'skew': rolling.skew(),
            'kurt': rolling.kurt(),
            'quantile_25': rolling.quantile(0.25),
            'quantile_75': rolling.quantile(0.75)
        })
    
    @staticmethod
    def z_score(data: pd.Series, window: int) -> pd.Series:
        """
        Rolling z-score normalization.
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        return (data - rolling_mean) / rolling_std
    
    @staticmethod
    def percentile_rank(data: pd.Series, window: int) -> pd.Series:
        """
        Rolling percentile rank.
        """
        def rank_last(x):
            return stats.percentileofscore(x[:-1], x[-1]) / 100.0 if len(x) > 1 else 0.5
        
        return data.rolling(window=window).apply(rank_last, raw=False)
    
    @staticmethod
    def returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate returns.
        """
        return data.pct_change(periods)
    
    @staticmethod
    def log_returns(data: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate log returns.
        """
        return np.log(data / data.shift(periods))
    
    @staticmethod
    def volatility(data: pd.Series, window: int, annualize: bool = False) -> pd.Series:
        """
        Rolling volatility calculation.
        """
        returns = StatisticalFeatures.returns(data)
        vol = returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)  # Assuming daily data
        
        return vol
    
    @staticmethod
    def correlation_features(data1: pd.Series, data2: pd.Series, window: int) -> pd.DataFrame:
        """
        Rolling correlation and related metrics between two series.
        """
        correlation = data1.rolling(window=window).corr(data2)
        
        # Calculate beta (slope of regression)
        def rolling_beta(x, y):
            if len(x) < 2 or x.std() == 0:
                return np.nan
            return np.cov(x, y)[0, 1] / np.var(x)
        
        beta = data1.rolling(window=window).apply(
            lambda x: rolling_beta(x.values, data2.loc[x.index].values), raw=False
        )
        
        return pd.DataFrame({
            'correlation': correlation,
            'beta': beta
        })
    
    @staticmethod
    def entropy(data: pd.Series, window: int, bins: int = 10) -> pd.Series:
        """
        Rolling Shannon entropy.
        """
        def shannon_entropy(x):
            hist, _ = np.histogram(x, bins=bins)
            hist = hist[hist > 0]
            prob = hist / hist.sum()
            return -np.sum(prob * np.log2(prob))
        
        return data.rolling(window=window).apply(shannon_entropy, raw=True)
    
    @staticmethod
    def hurst_exponent(data: pd.Series, window: int) -> pd.Series:
        """
        Rolling Hurst exponent for measuring long-term memory.
        """
        def hurst(ts):
            if len(ts) < 10:
                return np.nan
            
            lags = range(2, min(len(ts)//2, 20))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            if len(tau) < 2:
                return np.nan
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return data.rolling(window=window).apply(hurst, raw=True)


class DataPreprocessor:
    """
    Data cleaning and preprocessing utilities.
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.outlier_bounds = {}
    
    def handle_missing_data(self, data: pd.DataFrame, method: str = 'forward_fill',
                          **kwargs) -> pd.DataFrame:
        """
        Handle missing data using various strategies.
        
        Args:
            data: Input DataFrame
            method: 'forward_fill', 'backward_fill', 'mean', 'median', 'knn', 'drop'
        """
        data_clean = data.copy()
        
        if method == 'forward_fill':
            data_clean = data_clean.fillna(method='ffill')
        elif method == 'backward_fill':
            data_clean = data_clean.fillna(method='bfill')
        elif method == 'mean':
            data_clean = data_clean.fillna(data_clean.mean())
        elif method == 'median':
            data_clean = data_clean.fillna(data_clean.median())
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
            data_clean = pd.DataFrame(
                imputer.fit_transform(data_clean),
                index=data_clean.index,
                columns=data_clean.columns
            )
        elif method == 'drop':
            data_clean = data_clean.dropna()
        else:
            raise ValueError(f"Unknown missing data method: {method}")
        
        return data_clean
    
    def detect_outliers(self, data: pd.Series, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers using various methods.
        
        Args:
            data: Input series
            method: 'iqr', 'zscore', 'modified_zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            return pd.Series(z_scores > threshold, index=data.dropna().index)
        
        elif method == 'modified_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def handle_outliers(self, data: pd.Series, method: str = 'clip',
                       detection_method: str = 'iqr', **kwargs) -> pd.Series:
        """
        Handle outliers using various strategies.
        
        Args:
            data: Input series
            method: 'clip', 'remove', 'transform'
            detection_method: Method for detecting outliers
        """
        outliers = self.detect_outliers(data, detection_method, **kwargs)
        
        if method == 'clip':
            if detection_method == 'iqr':
                threshold = kwargs.get('threshold', 1.5)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return data.clip(lower=lower_bound, upper=upper_bound)
            else:
                # For other methods, use percentile clipping
                lower_percentile = kwargs.get('lower_percentile', 1)
                upper_percentile = kwargs.get('upper_percentile', 99)
                lower_bound = data.quantile(lower_percentile / 100)
                upper_bound = data.quantile(upper_percentile / 100)
                return data.clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'remove':
            return data[~outliers]
        
        elif method == 'transform':
            # Winsorization
            lower_percentile = kwargs.get('lower_percentile', 5)
            upper_percentile = kwargs.get('upper_percentile', 95)
            return data.clip(
                lower=data.quantile(lower_percentile / 100),
                upper=data.quantile(upper_percentile / 100)
            )
        
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Normalize data using various scaling methods.
        
        Args:
            data: Input DataFrame
            method: 'standard', 'minmax', 'robust'
            fit: Whether to fit the scaler or use existing one
        """
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if fit or method not in self.scalers:
            self.scalers[method] = scaler_class()
            scaled_data = self.scalers[method].fit_transform(data)
        else:
            scaled_data = self.scalers[method].transform(data)
        
        return pd.DataFrame(
            scaled_data,
            index=data.index,
            columns=data.columns
        )


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Main feature engineering pipeline that combines all transformations.
    """
    
    def __init__(self, 
                 technical_indicators: Optional[Dict[str, Dict]] = None,
                 statistical_features: Optional[Dict[str, Dict]] = None,
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 feature_selection: Optional[List[str]] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            technical_indicators: Dict of technical indicators to compute
            statistical_features: Dict of statistical features to compute
            preprocessing_config: Configuration for data preprocessing
            feature_selection: List of features to select
        """
        self.technical_indicators = technical_indicators or {}
        self.statistical_features = statistical_features or {}
        self.preprocessing_config = preprocessing_config or {}
        self.feature_selection = feature_selection
        
        self.preprocessor = DataPreprocessor()
        self.tech_indicators = TechnicalIndicators()
        self.stat_features = StatisticalFeatures()
        
        self.fitted_features = []
        self.is_fitted = False
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> None:
        """Validate that required OHLCV columns are present."""
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the feature set."""
        features = data.copy()
        
        for indicator, params in self.technical_indicators.items():
            try:
                if indicator == 'sma':
                    for window in params.get('windows', [10, 20, 50]):
                        features[f'sma_{window}'] = self.tech_indicators.sma(data['close'], window)
                
                elif indicator == 'ema':
                    for window in params.get('windows', [10, 20, 50]):
                        features[f'ema_{window}'] = self.tech_indicators.ema(data['close'], window)
                
                elif indicator == 'rsi':
                    window = params.get('window', 14)
                    features[f'rsi_{window}'] = self.tech_indicators.rsi(data['close'], window)
                
                elif indicator == 'macd':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    macd_data = self.tech_indicators.macd(data['close'], fast, slow, signal)
                    for col in macd_data.columns:
                        features[f'macd_{col}'] = macd_data[col]
                
                elif indicator == 'bollinger_bands':
                    window = params.get('window', 20)
                    num_std = params.get('num_std', 2)
                    bb_data = self.tech_indicators.bollinger_bands(data['close'], window, num_std)
                    for col in bb_data.columns:
                        features[f'bb_{col}'] = bb_data[col]
                
                elif indicator == 'stochastic':
                    k_window = params.get('k_window', 14)
                    d_window = params.get('d_window', 3)
                    stoch_data = self.tech_indicators.stochastic(
                        data['high'], data['low'], data['close'], k_window, d_window
                    )
                    for col in stoch_data.columns:
                        features[f'stoch_{col}'] = stoch_data[col]
                
                elif indicator == 'atr':
                    window = params.get('window', 14)
                    features[f'atr_{window}'] = self.tech_indicators.atr(
                        data['high'], data['low'], data['close'], window
                    )
                
                elif indicator == 'williams_r':
                    window = params.get('window', 14)
                    features[f'williams_r_{window}'] = self.tech_indicators.williams_r(
                        data['high'], data['low'], data['close'], window
                    )
                
                elif indicator == 'momentum':
                    for window in params.get('windows', [10, 20]):
                        features[f'momentum_{window}'] = self.tech_indicators.momentum(data['close'], window)
                
                elif indicator == 'roc':
                    for window in params.get('windows', [10, 20]):
                        features[f'roc_{window}'] = self.tech_indicators.rate_of_change(data['close'], window)
                
            except Exception as e:
                logger.warning(f"Failed to compute {indicator}: {e}")
        
        return features
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features to the feature set."""
        features = data.copy()
        
        for feature_type, params in self.statistical_features.items():
            try:
                if feature_type == 'rolling_stats':
                    for window in params.get('windows', [10, 20, 50]):
                        for col in params.get('columns', ['close']):
                            if col in data.columns:
                                stats_data = self.stat_features.rolling_statistics(data[col], window)
                                for stat_col in stats_data.columns:
                                    features[f'{col}_{stat_col}_{window}'] = stats_data[stat_col]
                
                elif feature_type == 'returns':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for period in params.get('periods', [1, 5, 10]):
                                features[f'{col}_return_{period}'] = self.stat_features.returns(data[col], period)
                                features[f'{col}_log_return_{period}'] = self.stat_features.log_returns(data[col], period)
                
                elif feature_type == 'volatility':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for window in params.get('windows', [10, 20, 50]):
                                features[f'{col}_volatility_{window}'] = self.stat_features.volatility(
                                    data[col], window, params.get('annualize', False)
                                )
                
                elif feature_type == 'z_score':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for window in params.get('windows', [20, 50]):
                                features[f'{col}_zscore_{window}'] = self.stat_features.z_score(data[col], window)
                
                elif feature_type == 'percentile_rank':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for window in params.get('windows', [20, 50]):
                                features[f'{col}_pctrank_{window}'] = self.stat_features.percentile_rank(data[col], window)
                
                elif feature_type == 'entropy':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for window in params.get('windows', [20, 50]):
                                features[f'{col}_entropy_{window}'] = self.stat_features.entropy(
                                    data[col], window, params.get('bins', 10)
                                )
                
                elif feature_type == 'hurst':
                    for col in params.get('columns', ['close']):
                        if col in data.columns:
                            for window in params.get('windows', [50, 100]):
                                features[f'{col}_hurst_{window}'] = self.stat_features.hurst_exponent(data[col], window)
                
            except Exception as e:
                logger.warning(f"Failed to compute {feature_type}: {e}")
        
        return features
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        features = data.copy()
        
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['year'] = data.index.year
            
            # Market session indicators
            features['is_market_open'] = (data.index.hour >= 9) & (data.index.hour < 16)
            features['is_weekend'] = data.index.dayofweek >= 5
        
        return features
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps."""
        processed_data = data.copy()
        
        # Handle missing data
        missing_method = self.preprocessing_config.get('missing_data_method', 'forward_fill')
        processed_data = self.preprocessor.handle_missing_data(processed_data, missing_method)
        
        # Handle outliers
        outlier_config = self.preprocessing_config.get('outlier_handling', {})
        if outlier_config:
            for col in processed_data.select_dtypes(include=[np.number]).columns:
                processed_data[col] = self.preprocessor.handle_outliers(
                    processed_data[col], **outlier_config
                )
        
        # Normalize data
        normalization_method = self.preprocessing_config.get('normalization_method')
        if normalization_method:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = self.preprocessor.normalize_data(
                processed_data[numeric_cols], normalization_method, fit=not self.is_fitted
            )
        
        return processed_data
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature pipeline."""
        # Validate input data
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Store fitted features for consistency
        temp_features = self.transform(X)
        self.fitted_features = temp_features.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data using the fitted pipeline."""
        # Start with the original data
        features = X.copy()
        
        # Add technical indicators
        if self.technical_indicators:
            try:
                self._validate_ohlcv_data(features)
                features = self._add_technical_indicators(features)
            except ValueError as e:
                logger.warning(f"Skipping technical indicators: {e}")
        
        # Add statistical features
        if self.statistical_features:
            features = self._add_statistical_features(features)
        
        # Add time-based features
        features = self._add_time_features(features)
        
        # Apply preprocessing
        if self.preprocessing_config:
            features = self._preprocess_data(features)
        
        # Feature selection
        if self.feature_selection:
            available_features = [col for col in self.feature_selection if col in features.columns]
            features = features[available_features]
        elif self.is_fitted and self.fitted_features:
            # Use fitted features for consistency
            available_features = [col for col in self.fitted_features if col in features.columns]
            features = features[available_features]
        
        return features
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the pipeline and transform the data."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the generated features."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        return self.fitted_features


# Utility functions for common feature engineering tasks
def create_lagged_features(data: pd.DataFrame, lags: List[int], 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Args:
        data: Input DataFrame
        lags: List of lag periods
        columns: Columns to create lags for (default: all numeric columns)
    """
    features = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        for lag in lags:
            features[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    return features


def create_interaction_features(data: pd.DataFrame, 
                               column_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between column pairs.
    """
    features = data.copy()
    
    for col1, col2 in column_pairs:
        if col1 in data.columns and col2 in data.columns:
            features[f'{col1}_{col2}_multiply'] = data[col1] * data[col2]
            features[f'{col1}_{col2}_divide'] = data[col1] / (data[col2] + 1e-8)
            features[f'{col1}_{col2}_add'] = data[col1] + data[col2]
            features[f'{col1}_{col2}_subtract'] = data[col1] - data[col2]
    
    return features


def create_rolling_rank_features(data: pd.DataFrame, windows: List[int],
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create rolling rank features.
    """
    features = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        for window in windows:
            features[f'{col}_rank_{window}'] = data[col].rolling(window).rank(pct=True)
    
    return features


# Example configuration for common use cases
DEFAULT_TECHNICAL_CONFIG = {
    'sma': {'windows': [10, 20, 50, 200]},
    'ema': {'windows': [12, 26, 50]},
    'rsi': {'window': 14},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger_bands': {'window': 20, 'num_std': 2},
    'atr': {'window': 14},
    'momentum': {'windows': [10, 20]},
    'roc': {'windows': [10, 20]}
}

DEFAULT_STATISTICAL_CONFIG = {
    'returns': {'columns': ['close'], 'periods': [1, 5, 10, 20]},
    'volatility': {'columns': ['close'], 'windows': [10, 20, 50], 'annualize': True},
    'rolling_stats': {'columns': ['close', 'volume'], 'windows': [10, 20, 50]},
    'z_score': {'columns': ['close'], 'windows': [20, 50]},
    'percentile_rank': {'columns': ['close'], 'windows': [20, 50]}
}

DEFAULT_PREPROCESSING_CONFIG = {
    'missing_data_method': 'forward_fill',
    'outlier_handling': {
        'method': 'clip',
        'detection_method': 'iqr',
        'threshold': 2.0
    },
    'normalization_method': 'standard'
}