@echo off
REM Test runner for Windows
REM This script sets up the environment and runs the ML model tests

echo Setting up environment...
set PYTHONPATH=%cd%
set TF_CPP_MIN_LOG_LEVEL=2

echo Installing dependencies...
pip install pytest pytest-cov pytest-mock numpy pandas scikit-learn xgboost lightgbm

echo Running unit tests...
python -m pytest src/tests/python_tests/test_ml_models.py -v -x

echo.
echo Running integration tests (this may take longer)...
python -m pytest src/tests/python_tests/test_ml_models_integration.py -v -x -m "not slow"

echo.
echo Test run complete!
pause
