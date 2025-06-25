"""
Alpha Signals Package

This package contains modules for generating trading signals using machine learning
and feature engineering techniques for statistical arbitrage strategies.

Modules:
    - ml_models: Machine learning models for signal generation
    - feature_engineering: Feature engineering utilities and transformations
"""

from . import ml_models, feature_engineering

# Make key components easily accessible at package level
__all__ = ['ml_models', 'feature_engineering']