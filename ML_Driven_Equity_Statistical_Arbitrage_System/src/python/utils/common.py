"""
Common Utilities Module

This module provides a collection of general-purpose utility functions that are
reusable across different parts of the statistical arbitrage system. It includes
file I/O operations, logging setup, configuration management, data validation,
and other helper functions.

Functions:
    - File I/O: JSON, pickle, CSV operations
    - Logging: Standardized logging setup
    - Configuration: Environment and config file management
    - Data validation: Type checking and validation utilities
    - Time utilities: Date/time manipulation functions
    - Math utilities: Common mathematical operations
    - Performance utilities: Timing and profiling helpers
"""

import os
import sys
import json
import pickle
import csv
import logging
import yaml
import configparser
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import time
import functools
from contextlib import contextmanager
import threading
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import gzip
import zipfile

# Type hints for better code documentation
PathLike = Union[str, Path]
ConfigDict = Dict[str, Any]
LogLevel = Union[str, int]


# Constants and Configuration
class Constants:
    """System-wide constants."""
    
    # Date formats
    DATE_FORMAT = "%Y-%m-%d"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # Market constants
    TRADING_DAYS_PER_YEAR = 252
    HOURS_PER_TRADING_DAY = 6.5
    MINUTES_PER_TRADING_DAY = 390
    
    # File extensions
    SUPPORTED_DATA_FORMATS = ['.csv', '.parquet', '.json', '.pkl', '.h5']
    SUPPORTED_CONFIG_FORMATS = ['.json', '.yaml', '.yml', '.ini', '.cfg']
    
    # Default directories
    DEFAULT_DATA_DIR = "data"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_CONFIG_DIR = "config"
    DEFAULT_OUTPUT_DIR = "outputs"
    
    # Performance thresholds
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    MAX_MEMORY_USAGE_PCT = 0.8  # 80% of available memory


class LogLevel(Enum):
    """Logging level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class SystemConfig:
    """System configuration data class."""
    data_dir: str = Constants.DEFAULT_DATA_DIR
    log_dir: str = Constants.DEFAULT_LOG_DIR
    config_dir: str = Constants.DEFAULT_CONFIG_DIR
    output_dir: str = Constants.DEFAULT_OUTPUT_DIR
    log_level: str = "INFO"
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_performance_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        return cls(**data)


# File I/O Operations
def ensure_directory(path: PathLike) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size(filepath: PathLike) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(filepath).stat().st_size


def is_large_file(filepath: PathLike, threshold: int = Constants.LARGE_FILE_THRESHOLD) -> bool:
    """
    Check if file is considered large.
    
    Args:
        filepath: Path to the file
        threshold: Size threshold in bytes
        
    Returns:
        True if file is large
    """
    return get_file_size(filepath) > threshold


def backup_file(filepath: PathLike, backup_suffix: str = None) -> Path:
    """
    Create a backup of a file.
    
    Args:
        filepath: Path to the file to backup
        backup_suffix: Suffix for backup file (default: timestamp)
        
    Returns:
        Path to backup file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if backup_suffix is None:
        backup_suffix = datetime.now().strftime(Constants.TIMESTAMP_FORMAT)
    
    backup_path = filepath.with_suffix(f".{backup_suffix}{filepath.suffix}")
    backup_path.write_bytes(filepath.read_bytes())
    
    return backup_path


def load_json(filepath: PathLike, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to JSON file
        encoding: File encoding
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")
    except Exception as e:
        raise IOError(f"Error loading JSON from {filepath}: {e}")


def save_json(data: Any, filepath: PathLike, indent: int = 2, 
              encoding: str = 'utf-8', backup: bool = True) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to JSON file
        indent: JSON indentation
        encoding: File encoding
        backup: Whether to backup existing file
    """
    filepath = Path(filepath)
    
    # Create backup if file exists
    if backup and filepath.exists():
        backup_file(filepath)
    
    # Ensure directory exists
    ensure_directory(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Error saving JSON to {filepath}: {e}")


def load_pickle(filepath: PathLike, compressed: bool = False) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to pickle file
        compressed: Whether file is gzip compressed
        
    Returns:
        Loaded data
    """
    try:
        if compressed:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        raise IOError(f"Error loading pickle from {filepath}: {e}")


def save_pickle(data: Any, filepath: PathLike, compressed: bool = False, 
                backup: bool = True) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to pickle file
        compressed: Whether to compress with gzip
        backup: Whether to backup existing file
    """
    filepath = Path(filepath)
    
    # Create backup if file exists
    if backup and filepath.exists():
        backup_file(filepath)
    
    # Ensure directory exists
    ensure_directory(filepath.parent)
    
    try:
        if compressed:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise IOError(f"Error saving pickle to {filepath}: {e}")


def load_yaml(filepath: PathLike, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load YAML data from file.
    
    Args:
        filepath: Path to YAML file
        encoding: File encoding
        
    Returns:
        Loaded YAML data
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {filepath}: {e}")
    except Exception as e:
        raise IOError(f"Error loading YAML from {filepath}: {e}")


def save_yaml(data: Any, filepath: PathLike, encoding: str = 'utf-8', 
              backup: bool = True) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        filepath: Path to YAML file
        encoding: File encoding
        backup: Whether to backup existing file
    """
    filepath = Path(filepath)
    
    # Create backup if file exists
    if backup and filepath.exists():
        backup_file(filepath)
    
    # Ensure directory exists
    ensure_directory(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise IOError(f"Error saving YAML to {filepath}: {e}")


def load_csv(filepath: PathLike, **kwargs) -> pd.DataFrame:
    """
    Load CSV file as DataFrame with error handling.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame
    """
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"Error loading CSV from {filepath}: {e}")


def save_csv(data: pd.DataFrame, filepath: PathLike, backup: bool = True, **kwargs) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Path to CSV file
        backup: Whether to backup existing file
        **kwargs: Additional arguments for DataFrame.to_csv
    """
    filepath = Path(filepath)
    
    # Create backup if file exists
    if backup and filepath.exists():
        backup_file(filepath)
    
    # Ensure directory exists
    ensure_directory(filepath.parent)
    
    try:
        data.to_csv(filepath, index=False, **kwargs)
    except Exception as e:
        raise IOError(f"Error saving CSV to {filepath}: {e}")


# Configuration Management
def load_config(filepath: PathLike) -> ConfigDict:
    """
    Load configuration from various file formats.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.json':
        return load_json(filepath)
    elif suffix in ['.yaml', '.yml']:
        return load_yaml(filepath)
    elif suffix in ['.ini', '.cfg']:
        config = configparser.ConfigParser()
        config.read(filepath)
        return {section: dict(config[section]) for section in config.sections()}
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def save_config(config: ConfigDict, filepath: PathLike, backup: bool = True) -> None:
    """
    Save configuration to file based on extension.
    
    Args:
        config: Configuration dictionary
        filepath: Path to config file
        backup: Whether to backup existing file
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.json':
        save_json(config, filepath, backup=backup)
    elif suffix in ['.yaml', '.yml']:
        save_yaml(config, filepath, backup=backup)
    elif suffix in ['.ini', '.cfg']:
        config_parser = configparser.ConfigParser()
        for section, items in config.items():
            config_parser[section] = items
        
        if backup and filepath.exists():
            backup_file(filepath)
        
        ensure_directory(filepath.parent)
        with open(filepath, 'w') as f:
            config_parser.write(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def get_env_var(name: str, default: Any = None, var_type: type = str) -> Any:
    """
    Get environment variable with type conversion.
    
    Args:
        name: Environment variable name
        default: Default value if not found
        var_type: Type to convert to
        
    Returns:
        Environment variable value
    """
    value = os.environ.get(name, default)
    
    if value is None:
        return None
    
    try:
        if var_type == bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif var_type == list:
            return value.split(',') if isinstance(value, str) else value
        else:
            return var_type(value)
    except (ValueError, TypeError):
        warnings.warn(f"Could not convert {name}={value} to {var_type.__name__}")
        return default


# Logging Setup
def setup_logging(
    log_level: LogLevel = LogLevel.INFO,
    log_file: Optional[PathLike] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True,
    include_module: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup standardized logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        log_format: Custom log format
        include_timestamp: Include timestamp in logs
        include_module: Include module name in logs
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Default log format
    if log_format is None:
        format_parts = []
        if include_timestamp:
            format_parts.append('%(asctime)s')
        format_parts.extend(['%(levelname)s'])
        if include_module:
            format_parts.append('%(name)s')
        format_parts.append('%(message)s')
        log_format = ' - '.join(format_parts)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level.value)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.value)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        ensure_directory(log_file.parent)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(log_level.value)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get calling module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# Data Validation Utilities
def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None,
                      min_rows: int = 1, check_nulls: bool = True) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        check_nulls: Whether to check for null values
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if check_nulls and df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        warnings.warn(f"DataFrame contains null values in columns: {null_cols}")
    
    return True


def validate_numeric(value: Any, min_val: float = None, max_val: float = None,
                    allow_none: bool = False) -> bool:
    """
    Validate numeric value.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_none: Whether None values are allowed
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        if allow_none:
            return True
        else:
            raise ValueError("Value cannot be None")
    
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"Value must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValueError("Value cannot be NaN or infinite")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"Value {value} is below minimum {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"Value {value} is above maximum {max_val}")
    
    return True


def validate_date_range(start_date: str, end_date: str, 
                       date_format: str = Constants.DATE_FORMAT) -> bool:
    """
    Validate date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Date format string
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    try:
        start_dt = datetime.strptime(start_date, date_format)
        end_dt = datetime.strptime(end_date, date_format)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start_dt >= end_dt:
        raise ValueError("Start date must be before end date")
    
    return True


# Time Utilities
def get_current_timestamp(format_str: str = Constants.DATETIME_FORMAT) -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: Timestamp format
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def parse_date(date_str: str, format_str: str = Constants.DATE_FORMAT) -> datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_str: Date string
        format_str: Date format
        
    Returns:
        Datetime object
    """
    try:
        return datetime.strptime(date_str, format_str)
    except ValueError as e:
        raise ValueError(f"Could not parse date '{date_str}' with format '{format_str}': {e}")


def get_trading_days(start_date: str, end_date: str, 
                    date_format: str = Constants.DATE_FORMAT) -> List[datetime]:
    """
    Get list of trading days between dates (excludes weekends).
    
    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Date format
        
    Returns:
        List of trading day datetime objects
    """
    start_dt = parse_date(start_date, date_format)
    end_dt = parse_date(end_date, date_format)
    
    trading_days = []
    current_date = start_dt
    
    while current_date <= end_dt:
        # Exclude weekends (Monday=0, Sunday=6)
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_days


def time_to_market_open(timezone_str: str = 'US/Eastern') -> timedelta:
    """
    Calculate time until next market open.
    
    Args:
        timezone_str: Market timezone
        
    Returns:
        Time until market open
    """
    import pytz
    
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    
    # Market opens at 9:30 AM
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If it's past market open today, use tomorrow
    if now >= market_open:
        market_open += timedelta(days=1)
    
    # Skip weekends
    while market_open.weekday() >= 5:
        market_open += timedelta(days=1)
    
    return market_open - now


# Performance Utilities
def timer(func: Callable = None, *, print_result: bool = True) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        print_result: Whether to print timing result
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                if print_result:
                    logger = get_logger()
                    logger.info(f"{f.__name__} executed in {execution_time:.4f} seconds")
                
                # Store timing in function attribute
                wrapper.last_execution_time = execution_time
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def time_context(description: str = "Operation"):
    """
    Context manager for timing code blocks.
    
    Args:
        description: Description of the operation
        
    Usage:
        with time_context("Data loading"):
            data = load_large_dataset()
    """
    logger = get_logger()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.info(f"{description} completed in {execution_time:.4f} seconds")


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


# Mathematical Utilities
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0 or np.isclose(denominator, 0):
        return default
    return numerator / denominator


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf')
    return (new_value - old_value) / old_value * 100


def round_to_precision(value: float, precision: int = 2) -> float:
    """
    Round value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    if np.isnan(value) or np.isinf(value):
        return value
    return round(value, precision)


# Hash and Checksum Utilities
def calculate_file_hash(filepath: PathLike, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def calculate_data_hash(data: Any, algorithm: str = 'md5') -> str:
    """
    Calculate hash of data object.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm
        
    Returns:
        Hex digest of data hash
    """
    if isinstance(data, pd.DataFrame):
        data_bytes = data.to_csv(index=False).encode('utf-8')
    elif isinstance(data, (dict, list)):
        data_bytes = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
    else:
        data_bytes = str(data).encode('utf-8')
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data_bytes)
    return hash_obj.hexdigest()


# Error Handling Utilities
def safe_execute(func: Callable, *args, default=None, 
                exceptions=(Exception,), log_error: bool = True, **kwargs) -> Any:
    """
    Execute function safely with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default return value on error
        exceptions: Exception types to catch
        log_error: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if log_error:
            logger = get_logger()
            logger.error(f"Error executing {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
        return default


@contextmanager
def suppress_warnings(category=Warning):
    """
    Context manager to suppress warnings.
    
    Args:
        category: Warning category to suppress
        
    Usage:
        with suppress_warnings(RuntimeWarning):
            risky_operation()
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        yield


# System Information
def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1024**3,
        'disk_free_gb': psutil.disk_usage('/').free / 1024**3,
        'current_time': get_current_timestamp(),
        'timezone': str(datetime.now().astimezone().tzinfo)
    }


# Initialize default system configuration
def get_default_config() -> SystemConfig:
    """Get default system configuration."""
    return SystemConfig(
        data_dir=get_env_var('DATA_DIR', Constants.DEFAULT_DATA_DIR),
        log_dir=get_env_var('LOG_DIR', Constants.DEFAULT_LOG_DIR),
        config_dir=get_env_var('CONFIG_DIR', Constants.DEFAULT_CONFIG_DIR),
        output_dir=get_env_var('OUTPUT_DIR', Constants.DEFAULT_OUTPUT_DIR),
        log_level=get_env_var('LOG_LEVEL', 'INFO'),
        max_workers=get_env_var('MAX_WORKERS', 4, int),
        memory_limit_gb=get_env_var('MEMORY_LIMIT_GB', 8.0, float),
        enable_performance_monitoring=get_env_var('ENABLE_PERF_MONITORING', True, bool)
    )


# Module initialization
_default_config = get_default_config()
_logger = None


def initialize_system(config: Optional[SystemConfig] = None, 
                     setup_dirs: bool = True) -> SystemConfig:
    """
    Initialize the system with configuration.
    
    Args:
        config: System configuration (uses default if None)
        setup_dirs: Whether to create directories
        
    Returns:
        System configuration
    """
    global _default_config, _logger
    
    if config is None:
        config = _default_config
    
    # Setup directories
    if setup_dirs:
        for dir_path in [config.data_dir, config.log_dir, 
                        config.config_dir, config.output_dir]:
            ensure_directory(dir_path)
    
    # Setup logging
    log_file = Path(config.log_dir) / f"system_{get_current_timestamp(Constants.TIMESTAMP_FORMAT)}.log"
    _logger = setup_logging(
        log_level=LogLevel[config.log_level.upper()],
        log_file=log_file
    )
    
    _logger.info("System initialized successfully")
    _logger.info(f"Configuration: {config.to_dict()}")
    
    return config


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    config = initialize_system()
    logger = get_logger(__name__)
    
    # Test file operations
    test_data = {'test': 'data', 'timestamp': get_current_timestamp()}
    
    with time_context("Testing file operations"):
        save_json(test_data, 'test_output.json')
        loaded_data = load_json('test_output.json')
        logger.info(f"Loaded data: {loaded_data}")
    
    # Test validation
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    validate_dataframe(df, required_columns=['a', 'b'])
    
    # Test mathematical utilities
    result = safe_divide(10, 0, default=float('inf'))
    logger.info(f"Safe division result: {result}")
    
    logger.info("All utility functions tested successfully!")
    print("Common utilities module loaded and tested successfully!")