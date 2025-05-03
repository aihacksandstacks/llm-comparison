"""
Logging module for the LLM Comparison Tool.
Provides a consistent interface for logging throughout the application.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# Try to import the configuration module
try:
    from src.shared.config import LOG_LEVEL
except ImportError:
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Map string log levels to logging constants
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Set up log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create a timestamped log file
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"llm_comparison_{TIMESTAMP}.log"

# Configure the root logger
logging.basicConfig(
    level=LOG_LEVEL_MAP.get(LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    Args:
        name: Module name or identifier. If None, returns the root logger.
        
    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    return logger

def set_log_level(level: Union[str, int]) -> None:
    """
    Set the log level for the root logger.
    
    Args:
        level: Log level as string ('DEBUG', 'INFO', etc.) or logging constant.
    """
    if isinstance(level, str):
        level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(level)
        
def log_function_call(logger: logging.Logger, func_name: str, args: tuple = None, kwargs: dict = None) -> None:
    """
    Log a function call with its arguments for debugging.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
    """
    args_str = str(args) if args else "()"
    kwargs_str = str(kwargs) if kwargs else "{}"
    logger.debug(f"Function call: {func_name} - args: {args_str}, kwargs: {kwargs_str}")

def log_db_operation(logger: logging.Logger, operation: str, details: dict) -> None:
    """
    Log a database operation for debugging.
    
    Args:
        logger: Logger instance
        operation: Type of operation (e.g., 'query', 'insert', 'update')
        details: Details about the operation
    """
    logger.debug(f"Database {operation}: {details}")

def log_api_request(logger: logging.Logger, api: str, endpoint: str, method: str, params: dict = None) -> None:
    """
    Log an API request for debugging.
    
    Args:
        logger: Logger instance
        api: API name or provider
        endpoint: Endpoint being called
        method: HTTP method used
        params: Request parameters
    """
    params_str = str(params) if params else "{}"
    logger.debug(f"API Request: {api} - {method} {endpoint} - params: {params_str}")

def log_api_response(logger: logging.Logger, api: str, endpoint: str, status_code: int, response_data: dict = None) -> None:
    """
    Log an API response for debugging.
    
    Args:
        logger: Logger instance
        api: API name or provider
        endpoint: Endpoint being called
        status_code: HTTP status code received
        response_data: Response data (truncated if too large)
    """
    # Truncate response data if it's too large
    if response_data and isinstance(response_data, dict) and len(str(response_data)) > 1000:
        response_data = "Response too large to log fully"
    
    logger.debug(f"API Response: {api} - {endpoint} - status: {status_code} - data: {response_data}")

# Export logger for convenience
logger = get_logger("llm_comparison") 