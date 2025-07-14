"""
Logging utilities for pandas_agent

This module provides specialized logging functionality for the pandas_agent,
ensuring consistent log formatting and proper integration with the A2A system.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class PandasAgentLogger:
    """
    Specialized logger for pandas_agent
    
    Provides structured logging with consistent formatting
    and integration with the CherryAI logging system.
    """
    
    def __init__(self, name: str = "pandas_agent", level: int = logging.INFO):
        """
        Initialize the pandas agent logger
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_logger(level)
    
    def _setup_logger(self, level: int):
        """
        Setup logger with appropriate formatting
        
        Args:
            level: Logging level
        """
        self.logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [pandas_agent] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def log_query(self, query: str, user_id: Optional[str] = None):
        """
        Log user query with metadata
        
        Args:
            query: User query
            user_id: Optional user identifier
        """
        metadata = f"[User: {user_id}] " if user_id else ""
        self.info(f"{metadata}Query: {query}")
    
    def log_result(self, result_type: str, execution_time: float):
        """
        Log analysis result
        
        Args:
            result_type: Type of result (e.g., 'visualization', 'analysis')
            execution_time: Execution time in seconds
        """
        self.info(f"Result generated: {result_type} (execution time: {execution_time:.2f}s)")
    
    def log_error_with_context(self, error: Exception, context: dict):
        """
        Log error with additional context
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        self.error(f"Error: {str(error)} | Context: {context_str}")


# Global logger instance
_global_logger = None


def get_logger() -> PandasAgentLogger:
    """
    Get global pandas agent logger instance
    
    Returns:
        Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = PandasAgentLogger()
    return _global_logger 