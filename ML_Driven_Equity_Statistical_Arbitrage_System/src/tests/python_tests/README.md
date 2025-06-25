# ML Models Testing Suite

This directory contains comprehensive unit and integration tests for the machine learning models in the statistical arbitrage system.

## Test Structure

```
src/tests/python_tests/
├── test_ml_models.py              # Unit tests for individual model classes
├── test_ml_models_integration.py  # Integration tests with realistic data
└── conftest.py                    # Test configuration and utilities
```

## Test Coverage

### Unit Tests (`test_ml_models.py`)
- **BaseModel**: Abstract class functionality, save/load, evaluation
- **XGBoostPredictor**: Training, prediction, feature importance, hyperparameter tuning
- **RandomForestPredictor**: Training, prediction, feature importance
- **LinearRegressor**: All regularization types (Ridge, Lasso, Elastic Net), coefficients
- **LightGBMPredictor**: Training, prediction, feature importance
- **LSTMModel**: Sequence preparation, training, prediction (requires TensorFlow)
- **TransformerModel**: Time series modeling (requires TensorFlow)
- **EnsembleModel**: Multiple model combination, weighted voting
- **Model Factory**: Dynamic model creation
- **Edge Cases**: Empty data, mismatched dimensions, NaN/Inf handling

### Integration Tests (`test_ml_models_integration.py`)
- **Full Workflows**: End-to-end training and prediction pipelines
- **Financial Data**: Tests with market-like time series features
- **Ensemble Performance**: Multi-model combination effectiveness
- **Robustness**: Noisy data, few samples, high dimensions
- **Time Series**: LSTM and Transformer with temporal dependencies

## Running Tests

### Option 1: Using the Python test runner
```bash
python run_tests.py --type all --verbose
python run_tests.py --type unit
python run_tests.py --type integration --coverage
```

### Option 2: Using pytest directly
```bash
# Install dependencies first
pip install -r requirements.txt

# Run all tests
pytest src/tests/python_tests/ -v

# Run only unit tests
pytest src/tests/python_tests/test_ml_models.py -v

# Run with coverage
pytest src/tests/python_tests/ --cov=src/python/alpha_signals --cov-report=html

# Skip slow tests
pytest src/tests/python_tests/ -m "not slow"
```

### Option 3: Windows batch file
```cmd
run_tests.bat
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Marker definitions (slow, integration, tensorflow, pytorch)
- Warning filters
- Default options

### Environment Setup
- Automatic PYTHONPATH configuration
- TensorFlow logging suppression
- Warning filtering
- Random seed management

## Test Data

Tests use synthetic data generators:
- **Regression data**: sklearn.make_regression for general ML testing
- **Time series data**: Correlated sequences with temporal dependencies
- **Financial data**: Market-like features (returns, moving averages, volatility)

## Dependencies

### Core Testing
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities

### ML Libraries (tested)
- `scikit-learn`: Base estimators and metrics
- `xgboost`: Gradient boosting
- `lightgbm`: Alternative gradient boosting
- `tensorflow`: Deep learning (optional)
- `torch`: PyTorch (optional)

### Data Processing
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `sklearn.preprocessing`: Data scaling

## Test Markers

Use pytest markers to run specific test subsets:

```bash
# Run only TensorFlow tests
pytest -m tensorflow

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run unit tests only
pytest -m unit
```

## Mock and Fixtures

Tests include:
- **Data fixtures**: Reusable synthetic datasets
- **Model mocking**: For testing error conditions
- **TensorFlow mocking**: When TF is not available
- **File system mocking**: For save/load testing

## Performance Testing

Integration tests include basic performance validation:
- Model training time monitoring  
- Memory usage validation
- Prediction latency checks
- Convergence verification

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Fast unit tests (< 30 seconds)
- Slower integration tests (< 5 minutes)
- Optional deep learning tests (requires GPU)
- Coverage reporting integration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **TensorFlow Warnings**: Set `TF_CPP_MIN_LOG_LEVEL=2`
3. **Memory Issues**: Reduce test data size or skip slow tests
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Windows-Specific
- Use `run_tests.bat` for environment setup
- PowerShell may require execution policy changes
- Path separators handled automatically

### Linux/Mac
- Use `python run_tests.py` or direct pytest
- Virtual environment recommended
- May need `export PYTHONPATH=.`

## Contributing

When adding new models or features:
1. Add unit tests to `test_ml_models.py`
2. Add integration tests to `test_ml_models_integration.py`
3. Update this README
4. Ensure all tests pass before committing
5. Add appropriate pytest markers
6. Mock external dependencies

## Test Results

Successful test runs should show:
- All unit tests passing (> 50 tests)
- Integration tests passing (> 20 tests)
- High code coverage (> 80%)
- No memory leaks or warnings
- Consistent performance across runs
