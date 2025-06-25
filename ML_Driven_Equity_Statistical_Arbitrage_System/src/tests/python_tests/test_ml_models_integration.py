"""
Integration Tests for ML Models

This module contains integration tests that test the models with realistic
data scenarios and end-to-end workflows.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from alpha_signals.ml_models import (
    XGBoostPredictor, RandomForestPredictor, LinearRegressor,
    LightGBMPredictor, EnsembleModel, create_model, TENSORFLOW_AVAILABLE
)

if TENSORFLOW_AVAILABLE:
    from alpha_signals.ml_models import LSTMModel, TransformerModel


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=1000, n_features=20, noise=0.1, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def time_series_data():
    """Generate synthetic time series data."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Create correlated time series
    X = np.random.randn(n_samples, n_features)
    for i in range(1, n_samples):
        X[i] = 0.8 * X[i-1] + 0.2 * np.random.randn(n_features)
    
    # Create target with temporal dependency
    y = np.zeros(n_samples)
    for i in range(5, n_samples):
        y[i] = 0.5 * np.sum(X[i-5:i, :3]) + 0.1 * np.random.randn()
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


@pytest.fixture
def financial_like_data():
    """Generate financial market-like data."""
    np.random.seed(42)
    n_samples = 2000
    
    # Simulate price returns
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create features: technical indicators
    features = []
    for i in range(20, n_samples):
        # Moving averages
        ma_5 = np.mean(prices[i-5:i])
        ma_20 = np.mean(prices[i-20:i])
        
        # Volatility
        vol = np.std(returns[i-20:i])
        
        # Price ratios
        price_ratio = prices[i] / ma_20
        ma_ratio = ma_5 / ma_20
        
        # Momentum
        momentum = (prices[i] - prices[i-20]) / prices[i-20]
        
        features.append([
            returns[i-1], returns[i-2], returns[i-3],  # Lagged returns
            ma_5, ma_20, vol, price_ratio, ma_ratio, momentum,
            prices[i-1] / prices[i-2] - 1  # Price change
        ])
    
    X = np.array(features)
    y = returns[20:]  # Predict next return
    
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


class TestModelIntegration:
    """Integration tests for individual models."""
    
    def test_xgboost_full_workflow(self, regression_data):
        """Test complete XGBoost workflow."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Initialize and train model
        model = XGBoostPredictor(max_depth=3, n_estimators=50)
        metrics = model.train(X_train, y_train)
        
        # Verify training metrics
        assert 'train_mse' in metrics
        assert 'train_r2' in metrics
        assert metrics['train_r2'] > 0.5  # Should achieve reasonable fit
        
        # Test prediction
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        
        # Test evaluation
        eval_metrics = model.evaluate(X_test, y_test)
        assert eval_metrics['r2'] > 0.3  # Should generalize reasonably
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        assert all(v >= 0 for v in importance.values())
    
    def test_random_forest_full_workflow(self, regression_data):
        """Test complete Random Forest workflow."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = RandomForestPredictor(n_estimators=50, max_depth=5)
        metrics = model.train(X_train, y_train)
        
        assert metrics['train_r2'] > 0.8  # RF should fit training data well
        
        predictions = model.predict(X_test)
        eval_metrics = model.evaluate(X_test, y_test)
        assert eval_metrics['r2'] > 0.3
    
    def test_linear_models_comparison(self, regression_data):
        """Test and compare different linear models."""
        X_train, X_test, y_train, y_test = regression_data
        
        models = [
            LinearRegressor(regularization='ridge', alpha=1.0),
            LinearRegressor(regularization='lasso', alpha=1.0),
            LinearRegressor(regularization='elastic', alpha=1.0)
        ]
        
        results = {}
        for model in models:
            model.train(X_train, y_train)
            eval_metrics = model.evaluate(X_test, y_test)
            results[model.regularization] = eval_metrics['r2']
        
        # All models should achieve reasonable performance
        for reg_type, r2 in results.items():
            assert r2 > 0.1, f"{reg_type} model performed poorly: R² = {r2}"
    
    def test_lightgbm_full_workflow(self, regression_data):
        """Test complete LightGBM workflow."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = LightGBMPredictor(num_leaves=15, learning_rate=0.1)
        metrics = model.train(X_train, y_train, num_boost_round=50)
        
        assert 'train_mse' in metrics
        assert 'train_r2' in metrics
        
        predictions = model.predict(X_test)
        eval_metrics = model.evaluate(X_test, y_test)
        assert eval_metrics['r2'] > 0.3
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.slow
    def test_lstm_time_series_workflow(self, time_series_data):
        """Test LSTM with time series data."""
        X_train, X_test, y_train, y_test = time_series_data
        
        model = LSTMModel(sequence_length=20, units=16, layers=1)
        metrics = model.train(X_train, y_train, epochs=5, batch_size=32)
        
        assert 'final_loss' in metrics
        assert 'epochs_trained' in metrics
        assert model.is_trained
        
        # Predict (noting that LSTM needs sequence length)
        if len(X_test) > model.sequence_length:
            predictions = model.predict(X_test)
            assert len(predictions) > 0
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.slow
    def test_transformer_time_series_workflow(self, time_series_data):
        """Test Transformer with time series data."""
        X_train, X_test, y_train, y_test = time_series_data
        
        model = TransformerModel(
            sequence_length=15, d_model=16, num_heads=2, num_layers=1
        )
        metrics = model.train(X_train, y_train, epochs=3, batch_size=32)
        
        assert 'final_loss' in metrics
        assert model.is_trained
        
        if len(X_test) > model.sequence_length:
            predictions = model.predict(X_test)
            assert len(predictions) > 0


class TestEnsembleIntegration:
    """Integration tests for ensemble models."""
    
    def test_heterogeneous_ensemble(self, regression_data):
        """Test ensemble with different model types."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Create diverse models
        models = [
            XGBoostPredictor(n_estimators=30, max_depth=3),
            RandomForestPredictor(n_estimators=30, max_depth=5),
            LinearRegressor(regularization='ridge', alpha=1.0),
            LightGBMPredictor(num_leaves=10)
        ]
        
        # Test equal weighting
        ensemble = EnsembleModel(models)
        metrics = ensemble.train(X_train, y_train)
        
        # Verify all models trained
        assert len(metrics) == len(models)
        for model in models:
            assert model.is_trained
        
        # Test ensemble prediction
        predictions = ensemble.predict(X_test)
        eval_metrics = ensemble.evaluate(X_test, y_test)
        
        # Ensemble should perform reasonably
        assert eval_metrics['r2'] > 0.2
        
        # Test custom weighting
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_ensemble = EnsembleModel(models, weights=weights)
        weighted_ensemble.is_trained = True  # Models already trained
        
        weighted_predictions = weighted_ensemble.predict(X_test)
        assert len(weighted_predictions) == len(predictions)
        
        # Weighted predictions should be different
        assert not np.allclose(predictions, weighted_predictions)
    
    def test_ensemble_vs_individual_models(self, regression_data):
        """Test that ensemble can outperform individual models."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Train individual models
        individual_models = [
            XGBoostPredictor(n_estimators=20),
            RandomForestPredictor(n_estimators=20),
            LinearRegressor(regularization='ridge')
        ]
        
        individual_scores = []
        for model in individual_models:
            if isinstance(model, LightGBMPredictor):
                model.train(X_train, y_train, num_boost_round=20)
            else:
                model.train(X_train, y_train)
            eval_metrics = model.evaluate(X_test, y_test)
            individual_scores.append(eval_metrics['r2'])
        
        # Train ensemble
        ensemble = EnsembleModel(individual_models)
        ensemble.is_trained = True  # Models already trained
        ensemble_metrics = ensemble.evaluate(X_test, y_test)
        
        # Ensemble should perform at least as well as average individual model
        avg_individual_score = np.mean(individual_scores)
        assert ensemble_metrics['r2'] >= avg_individual_score * 0.9


class TestFinancialDataScenarios:
    """Test models with financial market-like data."""
    
    def test_models_with_financial_data(self, financial_like_data):
        """Test all models with financial-like features."""
        X_train, X_test, y_train, y_test = financial_like_data
        
        # Scale features (important for financial data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models_to_test = [
            ('XGBoost', XGBoostPredictor(n_estimators=50)),
            ('RandomForest', RandomForestPredictor(n_estimators=50)),
            ('Ridge', LinearRegressor(regularization='ridge', alpha=0.1)),
            ('LightGBM', LightGBMPredictor())
        ]
        
        results = {}
        for name, model in models_to_test:
            if isinstance(model, LightGBMPredictor):
                model.train(X_train_scaled, y_train, num_boost_round=50)
            else:
                model.train(X_train_scaled, y_train)
            
            eval_metrics = model.evaluate(X_test_scaled, y_test)
            results[name] = eval_metrics
            
            # Financial prediction is challenging, but models should at least fit
            assert eval_metrics['mse'] > 0
            assert not np.isnan(eval_metrics['r2'])
        
        # Print results for debugging
        for name, metrics in results.items():
            print(f"{name}: R² = {metrics['r2']:.4f}, MSE = {metrics['mse']:.6f}")
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.slow
    def test_lstm_with_financial_data(self, financial_like_data):
        """Test LSTM specifically with financial time series."""
        X_train, X_test, y_train, y_test = financial_like_data
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        model = LSTMModel(sequence_length=30, units=32, layers=2, dropout=0.2)
        
        # Train with early stopping
        metrics = model.train(
            X_train_scaled, y_train_scaled, 
            epochs=10, batch_size=64, validation_split=0.2
        )
        
        assert model.is_trained
        assert 'final_loss' in metrics
        
        # Test prediction
        if len(X_test_scaled) > model.sequence_length:
            predictions_scaled = model.predict(X_test_scaled)
            # Transform back to original scale
            predictions = scaler_y.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            
            assert len(predictions) > 0
            assert not np.any(np.isnan(predictions))


class TestModelRobustness:
    """Test model robustness and edge cases."""
    
    def test_models_with_noisy_data(self, regression_data):
        """Test model performance with very noisy data."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Add significant noise
        noise_scale = np.std(y_train)
        y_train_noisy = y_train + np.random.normal(0, noise_scale, len(y_train))
        
        model = XGBoostPredictor(n_estimators=100, max_depth=3)
        model.train(X_train, y_train_noisy)
        
        # Model should still make reasonable predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert not np.any(np.isnan(predictions))
        assert np.all(np.isfinite(predictions))
    
    def test_models_with_few_samples(self):
        """Test models with very few training samples."""
        X_train = np.random.random((20, 5))  # Very few samples
        y_train = np.random.random(20)
        X_test = np.random.random((5, 5))
        
        # Simple models should still work
        model = LinearRegressor(regularization='ridge', alpha=10.0)  # High regularization
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_models_with_many_features(self):
        """Test models with high-dimensional data."""
        # More features than samples
        X_train = np.random.random((50, 100))
        y_train = np.random.random(50)
        X_test = np.random.random((10, 100))
        
        # Test regularized models
        models = [
            LinearRegressor(regularization='lasso', alpha=1.0),
            LinearRegressor(regularization='elastic', alpha=1.0),
            RandomForestPredictor(n_estimators=20, max_features='sqrt')
        ]
        
        for model in models:
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) == len(X_test)


class TestModelFactory:
    """Test the model factory functionality."""
    
    def test_factory_creates_all_models(self):
        """Test that factory can create all model types."""
        model_configs = [
            ('xgboost', {'n_estimators': 10}),
            ('random_forest', {'n_estimators': 10}),
            ('linear', {'regularization': 'ridge'}),
            ('lightgbm', {'num_leaves': 5})
        ]
        
        if TENSORFLOW_AVAILABLE:
            model_configs.extend([
                ('lstm', {'sequence_length': 10, 'units': 8}),
                ('transformer', {'sequence_length': 10, 'd_model': 8})
            ])
        
        for model_type, config in model_configs:
            model = create_model(model_type, **config)
            assert model is not None
            assert hasattr(model, 'train')
            assert hasattr(model, 'predict')
    
    def test_factory_with_training(self, regression_data):
        """Test factory-created models can train and predict."""
        X_train, X_test, y_train, y_test = regression_data
        
        model_types = ['xgboost', 'random_forest', 'linear', 'lightgbm']
        
        for model_type in model_types:
            model = create_model(model_type)
            
            if model_type == 'lightgbm':
                model.train(X_train, y_train, num_boost_round=10)
            else:
                model.train(X_train, y_train)
            
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
