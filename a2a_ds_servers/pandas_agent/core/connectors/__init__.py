"""
Data connectors for pandas_agent

This module provides various data source connectors including:
- File-based connectors (CSV, Excel, JSON, Parquet)
- SQL database connectors (PostgreSQL, MySQL, SQLite, etc.)
- Future: API connectors, NoSQL connectors
"""

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from core.connectors.base_connector import DataSourceConnector, FileConnector
from core.connectors.sql_connector import SQLConnector, SQLiteConnector

__all__ = [
    "DataSourceConnector",
    "FileConnector", 
    "SQLConnector",
    "SQLiteConnector",
] 