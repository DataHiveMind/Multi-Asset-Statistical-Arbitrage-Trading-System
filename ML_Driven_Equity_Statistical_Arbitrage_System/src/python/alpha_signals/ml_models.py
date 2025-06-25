"""
Machine Learning Models for Alpha Signal Generation

This module contains implementations of various machine learning and deep learning models
used for generating alpha signals in statistical arbitrage strategies.

Classes:
    - BaseModel: Abstract base class for all models
    - LSTMModel: Long Short-Term Memory neural network model
    - XGBoostPredictor: XGBoost gradient boosting model
    - RandomForestPredictor: Random Forest ensemble model
    - LinearRegressor: Linear regression model with regularization
    - LightGBMPredictor: LightGBM gradient boosting model
    - TransformerModel: Transformer-based time series model
"""

import abc
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# Deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Deep learning models will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some deep learning models will be disabled.")


class BaseModel(abc.ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the common interface that all models should implement.
    """
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        
    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the model on the provided data."""
        pass
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the provided data."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.name = model_data['name']
        self.feature_names = model_data.get('feature_names')
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True


class LSTMModel(BaseModel):
    """
    Long Short-Term Memory neural network model for time series prediction.
    """
    
    def __init__(self, sequence_length: int = 60, units: int = 50, dropout: float = 0.2, 
                 layers: int = 2, name: str = "LSTM"):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.layers = layers
        self.scaler_X = None
        self.scaler_y = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.units, return_sequences=True if self.layers > 1 else False,
                      input_shape=input_shape))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.layers):
            return_sequences = i < self.layers - 1
            model.add(LSTM(units=self.units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare sequences for LSTM training/prediction."""
        X_seq = []
        if y is not None:
            y_seq = []
            
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict[str, float]:
        """Train the LSTM model."""
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale the data
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_scaled)
        
        # Build model
        self.model = self._build_model((self.sequence_length, X.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Store training metrics
        self.training_metrics = {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        
        if len(X_seq) == 0:
            raise ValueError(f"Input data must have at least {self.sequence_length} samples")
        
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred


class XGBoostPredictor(BaseModel):
    """
    XGBoost gradient boosting model for alpha signal prediction.
    """
    
    def __init__(self, name: str = "XGBoost", **xgb_params):
        super().__init__(name)
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.xgb_params.update(xgb_params)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              tune_hyperparameters: bool = False, **kwargs) -> Dict[str, float]:
        """Train the XGBoost model."""
        
        if tune_hyperparameters:
            self._tune_hyperparameters(X, y)
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X, y)
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = {
            'train_mse': mean_squared_error(y, y_pred),
            'train_r2': r2_score(y, y_pred)
        }
        
        return self.training_metrics
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Tune hyperparameters using grid search."""
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        base_model = xgb.XGBRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        self.xgb_params.update(grid_search.best_params_)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class RandomForestPredictor(BaseModel):
    """
    Random Forest ensemble model for alpha signal prediction.
    """
    
    def __init__(self, name: str = "RandomForest", **rf_params):
        super().__init__(name)
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        self.rf_params.update(rf_params)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the Random Forest model."""
        self.model = RandomForestRegressor(**self.rf_params)
        self.model.fit(X, y)
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = {
            'train_mse': mean_squared_error(y, y_pred),
            'train_r2': r2_score(y, y_pred)
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained Random Forest model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LinearRegressor(BaseModel):
    """
    Linear regression model with regularization options.
    """
    
    def __init__(self, regularization: str = 'ridge', alpha: float = 1.0, name: str = "Linear"):
        super().__init__(name)
        self.regularization = regularization
        self.alpha = alpha
        
        if regularization == 'ridge':
            self.model_class = Ridge
        elif regularization == 'lasso':
            self.model_class = Lasso
        elif regularization == 'elastic':
            self.model_class = ElasticNet
        else:
            raise ValueError("Regularization must be 'ridge', 'lasso', or 'elastic'")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the linear regression model."""
        self.model = self.model_class(alpha=self.alpha)
        self.model.fit(X, y)
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = {
            'train_mse': mean_squared_error(y, y_pred),
            'train_r2': r2_score(y, y_pred)
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained linear model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        return dict(zip(self.feature_names, self.model.coef_))


class LightGBMPredictor(BaseModel):
    """
    LightGBM gradient boosting model for alpha signal prediction.
    """
    
    def __init__(self, name: str = "LightGBM", **lgb_params):
        super().__init__(name)
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.lgb_params.update(lgb_params)
    
    def train(self, X: np.ndarray, y: np.ndarray, num_boost_round: int = 100, **kwargs) -> Dict[str, float]:
        """Train the LightGBM model."""
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = {
            'train_mse': mean_squared_error(y, y_pred),
            'train_r2': r2_score(y, y_pred)
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained LightGBM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importance()
        return dict(zip(self.feature_names, importance))


class TransformerModel(BaseModel):
    """
    Transformer-based model for time series prediction.
    """
    
    def __init__(self, sequence_length: int = 60, d_model: int = 64, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1, name: str = "Transformer"):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler_X = None
        self.scaler_y = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Transformer model")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build the Transformer model architecture."""
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = inputs
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model
            )(x, x)
            
            attention = Dropout(self.dropout)(attention)
            x = LayerNormalization()(x + attention)
            
            # Feed-forward network
            ff = Dense(self.d_model * 4, activation='relu')(x)
            ff = Dense(self.d_model)(ff)
            ff = Dropout(self.dropout)(ff)
            x = LayerNormalization()(x + ff)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(self.d_model, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Prepare sequences for Transformer training/prediction."""
        X_seq = []
        if y is not None:
            y_seq = []
            
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict[str, float]:
        """Train the Transformer model."""
        from sklearn.preprocessing import StandardScaler
        
        # Scale the data
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_scaled)
        
        # Build model
        self.model = self._build_model((self.sequence_length, X.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Store training metrics
        self.training_metrics = {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained Transformer model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        
        if len(X_seq) == 0:
            raise ValueError(f"Input data must have at least {self.sequence_length} samples")
        
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred


# Model factory function
def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ('lstm', 'xgboost', 'random_forest', 
                   'linear', 'lightgbm', 'transformer')
        **kwargs: Additional parameters for the model
    
    Returns:
        Initialized model instance
    """
    model_mapping = {
        'lstm': LSTMModel,
        'xgboost': XGBoostPredictor,
        'random_forest': RandomForestPredictor,
        'linear': LinearRegressor,
        'lightgbm': LightGBMPredictor,
        'transformer': TransformerModel
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_mapping.keys())}")
    
    return model_mapping[model_type](**kwargs)


# Ensemble model class
class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from multiple models.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None, name: str = "Ensemble"):
        super().__init__(name)
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train all models in the ensemble."""
        all_metrics = {}
        
        for i, model in enumerate(self.models):
            metrics = model.train(X, y, **kwargs)
            all_metrics[f"{model.name}_{i}"] = metrics
        
        self.is_trained = True
        self.training_metrics = all_metrics
        
        return all_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions using weighted average."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights).reshape(-1, 1)
        
        return np.sum(predictions * weights, axis=0)