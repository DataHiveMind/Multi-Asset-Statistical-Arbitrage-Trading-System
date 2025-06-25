"""
Test Configuration and Utilities

This module provides configuration and utility functions for running tests.
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Set environment variables for testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)

# Test data directory
TEST_DATA_DIR = PROJECT_ROOT / 'data' / 'test'
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Test constants
RANDOM_SEED = 42
TEST_SAMPLE_SIZE = 1000
TEST_FEATURES = 10

print(f"Test configuration loaded. Project root: {PROJECT_ROOT}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
