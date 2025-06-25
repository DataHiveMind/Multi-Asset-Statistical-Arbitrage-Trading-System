# MLOps pipeline with model versioning and monitoring
import mlflow
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
from datetime import datetime
import logging

class MLOpsModelPipeline:
    """
    Production MLOps pipeline for model training, validation, and deployment
    """
    
    def __init__(self, experiment_name: str = "statistical_arbitrage_models"):
        mlflow.set_experiment(experiment_name)
        self.logger = logging.getLogger(__name__)
        
    def train_and_validate_model(self, 
                                features: pd.DataFrame, 
                                target: pd.Series,
                                model_params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Train model with proper time series cross-validation
        """
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(features):
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
                
                # Train model
                model = RandomForestRegressor(**model_params)
                model.fit(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                
                cv_scores.append({'mse': mse, 'mae': mae})
            
            # Calculate average scores
            avg_mse = np.mean([score['mse'] for score in cv_scores])
            avg_mae = np.mean([score['mae'] for score in cv_scores])
            
            # Log metrics
            mlflow.log_metric("cv_mse", avg_mse)
            mlflow.log_metric("cv_mae", avg_mae)
            
            # Train final model on all data
            final_model = RandomForestRegressor(**model_params)
            final_model.fit(features, target)
            
            # Log model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_artifact("feature_importance.csv")
            
            metrics = {
                'cv_mse': avg_mse,
                'cv_mae': avg_mae,
                'feature_importance': feature_importance
            }
            
            return final_model, metrics
    
    def model_drift_detection(self, 
                             model: Any, 
                             new_data: pd.DataFrame, 
                             reference_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect model drift using statistical tests
        """
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        for column in new_data.columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test for distribution drift
                ks_stat, p_value = ks_2samp(
                    reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                
                drift_results[f'{column}_drift_pvalue'] = p_value
                drift_results[f'{column}_drift_detected'] = p_value < 0.05
        
        return drift_results
    
    def deploy_model_to_staging(self, model_uri: str, model_name: str):
        """
        Deploy model to staging environment
        """
        client = mlflow.tracking.MlflowClient()
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        self.logger.info(f"Model {model_name} v{model_version.version} deployed to staging")
        
        return model_version
