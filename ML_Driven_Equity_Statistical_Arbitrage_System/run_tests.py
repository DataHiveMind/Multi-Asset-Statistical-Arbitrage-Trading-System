#!/usr/bin/env python3
"""
Test Runner for ML Models

This script runs all tests for the machine learning models with proper
environment setup and reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Set up the test environment."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    
    return project_root

def run_tests(test_type='all', verbose=False, coverage=False):
    """Run the specified tests."""
    project_root = setup_environment()
    test_dir = project_root / 'src' / 'tests' / 'python_tests'
    
    # Base pytest command
    cmd = ['python', '-m', 'pytest']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src/python/alpha_signals', '--cov-report=html', '--cov-report=term'])
    
    # Add test files based on type
    if test_type == 'unit':
        cmd.append(str(test_dir / 'test_ml_models.py'))
    elif test_type == 'integration':
        cmd.append(str(test_dir / 'test_ml_models_integration.py'))
    elif test_type == 'all':
        cmd.append(str(test_dir))
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add markers to skip slow tests by default
    if test_type != 'integration':
        cmd.extend(['-m', 'not slow'])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run ML model tests')
    parser.add_argument(
        '--type', '-t', 
        choices=['unit', 'integration', 'all'], 
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Run tests in verbose mode'
    )
    parser.add_argument(
        '--coverage', '-c', 
        action='store_true',
        help='Run tests with coverage reporting'
    )
    parser.add_argument(
        '--install-deps', '-i', 
        action='store_true',
        help='Install test dependencies before running'
    )
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            'pip', 'install', '-r', 'requirements.txt'
        ])
        subprocess.run([
            'pip', 'install', 'pytest', 'pytest-cov', 'pytest-mock'
        ])
    
    return run_tests(args.type, args.verbose, args.coverage)

if __name__ == '__main__':
    sys.exit(main())
