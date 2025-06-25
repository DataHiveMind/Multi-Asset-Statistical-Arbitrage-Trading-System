"""
Unit Tests for Machine Learning Models

This module contains comprehensive unit tests for all machine learning models
defined in src/python/alpha_signals/ml_models.py.

Test Coverage:
- BaseModel abstract class functionality
- All concrete model implementations (LSTM, XGBoost, RandomForest, Linear, LightGBM, Transformer)
- Model training, prediction, and evaluation
- Error handling and edge cases
- Model persistence (save/load)
- Feature importance extraction
- Ensemble model functionality
"""

import unittest
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any

# Import the models to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from alpha_signals.ml_models import (
    BaseModel, LSTMModel, XGBoostPredictor, RandomForestPredictor,
    LinearRegressor, LightGBMPredictor, TransformerModel, EnsembleModel,
    create_model, TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
)

# Suppress warnings during testing
warnings.filterwarnings('ignore')


class TestBaseModel(unittest.TestCase):
    """Test cases for the abstract BaseModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class ConcreteModel(BaseModel):
            def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
                self.is_trained = True
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                return {'test_metric': 0.5}
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                if not self.is_trained:
                    raise ValueError("Model must be trained before prediction")
                return np.random.random(X.shape[0])
        
        self.model = ConcreteModel("TestModel")
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
        self.y_test = np.random.random(20)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "TestModel")
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.feature_names)
        self.assertEqual(self.model.training_metrics, {})
    
    def test_train_sets_flags(self):
        """Test that training sets appropriate flags."""
        self.assertFalse(self.model.is_trained)
        metrics = self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_metric', metrics)
    
    def test_predict_requires_training(self):
        """Test that prediction requires training."""
        with self.assertRaises(ValueError) as context:
            self.model.predict(self.X_test)
        self.assertIn("must be trained", str(context.exception))
    
    def test_evaluate_functionality(self):
        """Test model evaluation functionality."""
        # Train model first
        self.model.train(self.X_train, self.y_train)
        
        # Test evaluation
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        expected_metrics = ['mse', 'mae', 'r2', 'rmse']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_evaluate_requires_training(self):
        """Test that evaluation requires training."""
        with self.assertRaises(ValueError) as context:
            self.model.evaluate(self.X_test, self.y_test)
        self.assertIn("must be trained", str(context.exception))
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Test saving
            self.model.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new model instance and load
            new_model = self.model.__class__("LoadedModel")
            self.assertFalse(new_model.is_trained)
            
            new_model.load_model(temp_path)
            self.assertTrue(new_model.is_trained)
            self.assertEqual(new_model.name, "TestModel")
            self.assertIsNotNone(new_model.feature_names)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_untrained_model_raises_error(self):
        """Test that saving untrained model raises error."""
        with self.assertRaises(ValueError) as context:
            self.model.save_model("test.pkl")
        self.assertIn("Cannot save untrained model", str(context.exception))


class TestXGBoostPredictor(unittest.TestCase):
    """Test cases for XGBoostPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = XGBoostPredictor()
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_initialization(self):
        """Test XGBoost model initialization."""
        self.assertEqual(self.model.name, "XGBoost")
        self.assertIn('objective', self.model.xgb_params)
        self.assertIn('max_depth', self.model.xgb_params)
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        custom_model = XGBoostPredictor(max_depth=10, n_estimators=200)
        self.assertEqual(custom_model.xgb_params['max_depth'], 10)
        self.assertEqual(custom_model.xgb_params['n_estimators'], 200)
    
    def test_train_and_predict(self):
        """Test training and prediction."""
        # Train model
        metrics = self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        self.assertIn('train_mse', metrics)
        self.assertIn('train_r2', metrics)
        
        # Test prediction
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, (self.X_test.shape[0],))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train(self.X_train, self.y_train)
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), self.X_train.shape[1])
        for key, value in importance.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, (int, float))
    
    def test_feature_importance_requires_training(self):
        """Test that feature importance requires training."""
        with self.assertRaises(ValueError):
            self.model.get_feature_importance()
    
    @patch('alpha_signals.ml_models.GridSearchCV')
    def test_hyperparameter_tuning(self, mock_grid_search):
        """Test hyperparameter tuning functionality."""
        # Mock the grid search
        mock_search_instance = MagicMock()
        mock_search_instance.best_params_ = {'max_depth': 5, 'learning_rate': 0.05}
        mock_grid_search.return_value = mock_search_instance
        
        # Train with hyperparameter tuning
        self.model.train(self.X_train, self.y_train, tune_hyperparameters=True)
        
        # Verify grid search was called
        mock_grid_search.assert_called_once()
        mock_search_instance.fit.assert_called_once()


class TestRandomForestPredictor(unittest.TestCase):
    """Test cases for RandomForestPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RandomForestPredictor()
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_initialization(self):
        """Test Random Forest model initialization."""
        self.assertEqual(self.model.name, "RandomForest")
        self.assertIn('n_estimators', self.model.rf_params)
        self.assertIn('random_state', self.model.rf_params)
    
    def test_train_and_predict(self):
        """Test training and prediction."""
        metrics = self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        self.assertIn('train_mse', metrics)
        self.assertIn('train_r2', metrics)
        
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, (self.X_test.shape[0],))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train(self.X_train, self.y_train)
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), self.X_train.shape[1])


class TestLinearRegressor(unittest.TestCase):
    """Test cases for LinearRegressor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_ridge_initialization(self):
        """Test Ridge regression initialization."""
        model = LinearRegressor(regularization='ridge', alpha=1.0)
        self.assertEqual(model.regularization, 'ridge')
        self.assertEqual(model.alpha, 1.0)
    
    def test_lasso_initialization(self):
        """Test Lasso regression initialization."""
        model = LinearRegressor(regularization='lasso', alpha=0.5)
        self.assertEqual(model.regularization, 'lasso')
        self.assertEqual(model.alpha, 0.5)
    
    def test_elastic_initialization(self):
        """Test Elastic Net regression initialization."""
        model = LinearRegressor(regularization='elastic', alpha=0.1)
        self.assertEqual(model.regularization, 'elastic')
        self.assertEqual(model.alpha, 0.1)
    
    def test_invalid_regularization(self):
        """Test invalid regularization parameter."""
        with self.assertRaises(ValueError):
            LinearRegressor(regularization='invalid')
    
    def test_train_and_predict(self):
        """Test training and prediction for all regularization types."""
        for reg_type in ['ridge', 'lasso', 'elastic']:
            with self.subTest(regularization=reg_type):
                model = LinearRegressor(regularization=reg_type)
                
                metrics = model.train(self.X_train, self.y_train)
                self.assertTrue(model.is_trained)
                self.assertIn('train_mse', metrics)
                
                predictions = model.predict(self.X_test)
                self.assertEqual(predictions.shape, (self.X_test.shape[0],))
    
    def test_get_coefficients(self):
        """Test coefficient extraction."""
        model = LinearRegressor(regularization='ridge')
        model.train(self.X_train, self.y_train)
        
        coefficients = model.get_coefficients()
        self.assertIsInstance(coefficients, dict)
        self.assertEqual(len(coefficients), self.X_train.shape[1])
    
    def test_coefficients_require_training(self):
        """Test that getting coefficients requires training."""
        model = LinearRegressor()
        with self.assertRaises(ValueError):
            model.get_coefficients()


class TestLightGBMPredictor(unittest.TestCase):
    """Test cases for LightGBMPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = LightGBMPredictor()
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_initialization(self):
        """Test LightGBM model initialization."""
        self.assertEqual(self.model.name, "LightGBM")
        self.assertIn('objective', self.model.lgb_params)
        self.assertIn('metric', self.model.lgb_params)
    
    def test_train_and_predict(self):
        """Test training and prediction."""
        metrics = self.model.train(self.X_train, self.y_train, num_boost_round=10)
        self.assertTrue(self.model.is_trained)
        self.assertIn('train_mse', metrics)
        
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, (self.X_test.shape[0],))
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.model.train(self.X_train, self.y_train, num_boost_round=10)
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), self.X_train.shape[1])


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestLSTMModel(unittest.TestCase):
    """Test cases for LSTMModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = LSTMModel(sequence_length=10, units=8, layers=1)
        # Create sequential data for time series
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_initialization(self):
        """Test LSTM model initialization."""
        self.assertEqual(self.model.name, "LSTM")
        self.assertEqual(self.model.sequence_length, 10)
        self.assertEqual(self.model.units, 8)
        self.assertEqual(self.model.layers, 1)
    
    def test_initialization_without_tensorflow(self):
        """Test LSTM initialization fails without TensorFlow."""
        with patch('alpha_signals.ml_models.TENSORFLOW_AVAILABLE', False):
            with self.assertRaises(ImportError):
                LSTMModel()
    
    def test_sequence_preparation(self):
        """Test sequence preparation for LSTM."""
        X = np.random.random((50, 3))
        y = np.random.random(50)
        
        X_seq, y_seq = self.model._prepare_sequences(X, y)
        
        expected_length = len(X) - self.model.sequence_length
        self.assertEqual(len(X_seq), expected_length)
        self.assertEqual(len(y_seq), expected_length)
        self.assertEqual(X_seq.shape, (expected_length, self.model.sequence_length, 3))
    
    def test_train_and_predict(self):
        """Test LSTM training and prediction."""
        # Use more data for LSTM to work properly
        X_train = np.random.random((200, 5))
        y_train = np.random.random(200)
        X_test = np.random.random((50, 5))
        
        metrics = self.model.train(X_train, y_train, epochs=2, batch_size=16)
        self.assertTrue(self.model.is_trained)
        self.assertIn('final_loss', metrics)
        
        predictions = self.model.predict(X_test)
        expected_length = len(X_test) - self.model.sequence_length
        self.assertEqual(len(predictions), expected_length)
    
    def test_predict_insufficient_data(self):
        """Test prediction with insufficient data."""
        self.model.is_trained = True
        self.model.scaler_X = MagicMock()
        self.model.scaler_X.transform.return_value = np.random.random((5, 3))
        
        with self.assertRaises(ValueError) as context:
            self.model.predict(np.random.random((5, 3)))
        self.assertIn("at least", str(context.exception))


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestTransformerModel(unittest.TestCase):
    """Test cases for TransformerModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TransformerModel(sequence_length=10, d_model=16, num_heads=2, num_layers=1)
        self.X_train = np.random.random((200, 5))
        self.y_train = np.random.random(200)
        self.X_test = np.random.random((50, 5))
    
    def test_initialization(self):
        """Test Transformer model initialization."""
        self.assertEqual(self.model.name, "Transformer")
        self.assertEqual(self.model.sequence_length, 10)
        self.assertEqual(self.model.d_model, 16)
        self.assertEqual(self.model.num_heads, 2)
    
    def test_train_and_predict(self):
        """Test Transformer training and prediction."""
        metrics = self.model.train(self.X_train, self.y_train, epochs=1, batch_size=16)
        self.assertTrue(self.model.is_trained)
        self.assertIn('final_loss', metrics)
        
        predictions = self.model.predict(self.X_test)
        expected_length = len(self.X_test) - self.model.sequence_length
        self.assertEqual(len(predictions), expected_length)


class TestEnsembleModel(unittest.TestCase):
    """Test cases for EnsembleModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model1 = XGBoostPredictor(name="XGB1")
        self.model2 = RandomForestPredictor(name="RF1")
        self.ensemble = EnsembleModel([self.model1, self.model2])
        
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_initialization(self):
        """Test ensemble model initialization."""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertEqual(len(self.ensemble.weights), 2)
        self.assertEqual(self.ensemble.weights, [0.5, 0.5])
    
    def test_custom_weights(self):
        """Test ensemble with custom weights."""
        weights = [0.7, 0.3]
        ensemble = EnsembleModel([self.model1, self.model2], weights=weights)
        self.assertEqual(ensemble.weights, weights)
    
    def test_mismatched_weights(self):
        """Test error on mismatched weights and models."""
        with self.assertRaises(ValueError):
            EnsembleModel([self.model1, self.model2], weights=[0.5])
    
    def test_train_and_predict(self):
        """Test ensemble training and prediction."""
        # Train ensemble
        metrics = self.ensemble.train(self.X_train, self.y_train)
        self.assertTrue(self.ensemble.is_trained)
        self.assertIsInstance(metrics, dict)
        self.assertTrue(len(metrics) >= 2)  # Should have metrics from both models
        
        # Test prediction
        predictions = self.ensemble.predict(self.X_test)
        self.assertEqual(predictions.shape, (self.X_test.shape[0],))


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function."""
    
    def test_create_xgboost(self):
        """Test creating XGBoost model via factory."""
        model = create_model('xgboost', max_depth=5)
        self.assertIsInstance(model, XGBoostPredictor)
        self.assertEqual(model.xgb_params['max_depth'], 5)
    
    def test_create_random_forest(self):
        """Test creating Random Forest model via factory."""
        model = create_model('random_forest', n_estimators=50)
        self.assertIsInstance(model, RandomForestPredictor)
        self.assertEqual(model.rf_params['n_estimators'], 50)
    
    def test_create_linear(self):
        """Test creating Linear model via factory."""
        model = create_model('linear', regularization='lasso')
        self.assertIsInstance(model, LinearRegressor)
        self.assertEqual(model.regularization, 'lasso')
    
    def test_create_lightgbm(self):
        """Test creating LightGBM model via factory."""
        model = create_model('lightgbm', num_leaves=20)
        self.assertIsInstance(model, LightGBMPredictor)
        self.assertEqual(model.lgb_params['num_leaves'], 20)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_lstm(self):
        """Test creating LSTM model via factory."""
        model = create_model('lstm', sequence_length=30)
        self.assertIsInstance(model, LSTMModel)
        self.assertEqual(model.sequence_length, 30)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_transformer(self):
        """Test creating Transformer model via factory."""
        model = create_model('transformer', d_model=32)
        self.assertIsInstance(model, TransformerModel)
        self.assertEqual(model.d_model, 32)
    
    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with self.assertRaises(ValueError) as context:
            create_model('invalid_model')
        self.assertIn("Unknown model type", str(context.exception))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = XGBoostPredictor()
    
    def test_empty_data(self):
        """Test handling of empty data."""
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        with self.assertRaises((ValueError, IndexError)):
            self.model.train(X_empty, y_empty)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched X and y dimensions."""
        X = np.random.random((100, 5))
        y = np.random.random(50)  # Wrong size
        
        with self.assertRaises((ValueError, IndexError)):
            self.model.train(X, y)
    
    def test_predict_wrong_features(self):
        """Test prediction with wrong number of features."""
        X_train = np.random.random((100, 5))
        y_train = np.random.random(100)
        X_test_wrong = np.random.random((20, 3))  # Wrong number of features
        
        self.model.train(X_train, y_train)
        
        with self.assertRaises((ValueError, IndexError)):
            self.model.predict(X_test_wrong)
    
    def test_nan_in_data(self):
        """Test handling of NaN values in data."""
        X_train = np.random.random((100, 5))
        X_train[0, 0] = np.nan
        y_train = np.random.random(100)
        
        # Some models might handle NaN, others might not
        # The specific behavior depends on the underlying library
        try:
            self.model.train(X_train, y_train)
        except (ValueError, TypeError):
            pass  # Expected for some models
    
    def test_inf_in_data(self):
        """Test handling of infinite values in data."""
        X_train = np.random.random((100, 5))
        X_train[0, 0] = np.inf
        y_train = np.random.random(100)
        
        try:
            self.model.train(X_train, y_train)
        except (ValueError, TypeError):
            pass  # Expected for some models


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = [
            XGBoostPredictor(),
            RandomForestPredictor(),
            LinearRegressor(),
            LightGBMPredictor()
        ]
        
        self.X_train = np.random.random((100, 5))
        self.y_train = np.random.random(100)
        self.X_test = np.random.random((20, 5))
    
    def test_save_load_all_models(self):
        """Test saving and loading for all model types."""
        for model in self.models:
            with self.subTest(model_type=model.__class__.__name__):
                # Train model
                if isinstance(model, LightGBMPredictor):
                    model.train(self.X_train, self.y_train, num_boost_round=5)
                else:
                    model.train(self.X_train, self.y_train)
                
                # Get predictions before saving
                predictions_before = model.predict(self.X_test)
                
                # Save and load
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                    temp_path = f.name
                
                try:
                    model.save_model(temp_path)
                    
                    # Create new instance and load
                    new_model = model.__class__()
                    new_model.load_model(temp_path)
                    
                    # Test that loaded model produces same predictions
                    predictions_after = new_model.predict(self.X_test)
                    np.testing.assert_array_almost_equal(
                        predictions_before, predictions_after, decimal=5
                    )
                    
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
