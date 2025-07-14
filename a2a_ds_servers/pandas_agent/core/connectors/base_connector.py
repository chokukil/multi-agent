"""
Base connector for data sources

This module provides the abstract base class for all data source connectors
in the pandas_agent system.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from helpers.logger import get_logger


class DataSourceConnector(ABC):
    """
    Abstract base class for data source connectors
    
    All data source connectors must inherit from this class and implement
    the required methods for connecting to and reading from data sources.
    """
    
    def __init__(self, 
                 name: str,
                 connection_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data source connector
        
        Args:
            name: Name of the data source
            connection_config: Configuration for connection
        """
        self.name = name
        self.connection_config = connection_config or {}
        self.logger = get_logger()
        self._connection = None
        self._is_connected = False
        
        self.logger.info(f"DataSourceConnector '{name}' initialized")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """
        Close connection to the data source
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the data source
        
        Returns:
            Dictionary with connection test results
        """
        pass
    
    @abstractmethod
    async def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information from the data source
        
        Returns:
            Dictionary with schema information (tables, columns, etc.)
        """
        pass
    
    @abstractmethod
    async def read_data(self, 
                       query: Optional[str] = None,
                       table_name: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Read data from the data source
        
        Args:
            query: SQL query or equivalent for the data source
            table_name: Name of table/collection to read
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame with the loaded data
        """
        pass
    
    @abstractmethod
    async def get_data_preview(self, 
                              table_name: str,
                              limit: int = 100) -> pd.DataFrame:
        """
        Get a preview of data from a table
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with preview data
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connector is connected"""
        return self._is_connected
    
    @property
    def connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "connected": self._is_connected,
            "config": {k: "***" if "password" in k.lower() or "key" in k.lower() 
                      else v for k, v in self.connection_config.items()}
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', connected={self._is_connected})"


class FileConnector(DataSourceConnector):
    """
    Base class for file-based data sources (CSV, Excel, JSON, etc.)
    """
    
    def __init__(self, 
                 name: str,
                 file_path: str,
                 file_type: str = "auto",
                 **kwargs):
        """
        Initialize file connector
        
        Args:
            name: Name of the data source
            file_path: Path to the file
            file_type: Type of file (csv, excel, json, parquet, auto)
            **kwargs: Additional file reading parameters
        """
        super().__init__(name, {"file_path": file_path, "file_type": file_type, **kwargs})
        self.file_path = file_path
        self.file_type = file_type
        self.read_params = kwargs
    
    async def connect(self) -> bool:
        """File connectors don't need persistent connections"""
        try:
            import os
            if os.path.exists(self.file_path):
                self._is_connected = True
                self.logger.info(f"File connector '{self.name}' connected to {self.file_path}")
                return True
            else:
                self.logger.error(f"File not found: {self.file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to file: {e}")
            return False
    
    async def disconnect(self):
        """File connectors don't need to disconnect"""
        self._is_connected = False
        self.logger.info(f"File connector '{self.name}' disconnected")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test file accessibility"""
        try:
            import os
            
            result = {
                "success": False,
                "file_exists": False,
                "file_size": 0,
                "readable": False,
                "error": None
            }
            
            if os.path.exists(self.file_path):
                result["file_exists"] = True
                result["file_size"] = os.path.getsize(self.file_path)
                
                if os.access(self.file_path, os.R_OK):
                    result["readable"] = True
                    result["success"] = True
                else:
                    result["error"] = "File is not readable"
            else:
                result["error"] = "File does not exist"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_exists": False,
                "file_size": 0,
                "readable": False
            }
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get file schema information"""
        try:
            # Read a small sample to get schema
            sample_df = await self.get_data_preview("", limit=5)
            
            return {
                "columns": list(sample_df.columns),
                "dtypes": sample_df.dtypes.to_dict(),
                "shape": sample_df.shape,
                "file_info": {
                    "path": self.file_path,
                    "type": self.file_type
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {e}")
            return {"error": str(e)}
    
    async def read_data(self, 
                       query: Optional[str] = None,
                       table_name: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """Read data from file"""
        try:
            # Determine file type if auto
            file_type = self.file_type
            if file_type == "auto":
                file_type = self._detect_file_type()
            
            # Merge parameters
            read_params = {**self.read_params, **kwargs}
            
            # Read based on file type
            if file_type == "csv":
                df = pd.read_csv(self.file_path, **read_params)
            elif file_type in ["excel", "xlsx", "xls"]:
                df = pd.read_excel(self.file_path, **read_params)
            elif file_type == "json":
                df = pd.read_json(self.file_path, **read_params)
            elif file_type == "parquet":
                df = pd.read_parquet(self.file_path, **read_params)
            elif file_type == "feather":
                df = pd.read_feather(self.file_path, **read_params)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Successfully read {len(df)} rows from {self.file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading data from file: {e}")
            raise
    
    async def get_data_preview(self, 
                              table_name: str,
                              limit: int = 100) -> pd.DataFrame:
        """Get preview of file data"""
        try:
            # For files, we'll read with nrows parameter
            if self.file_type in ["csv"]:
                read_params = {**self.read_params, "nrows": limit}
            else:
                read_params = self.read_params
            
            df = await self.read_data(**read_params)
            
            # Limit rows if the file reader doesn't support nrows
            if len(df) > limit:
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data preview: {e}")
            raise
    
    def _detect_file_type(self) -> str:
        """Auto-detect file type from extension"""
        import os
        
        _, ext = os.path.splitext(self.file_path.lower())
        
        type_mapping = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.parquet': 'parquet',
            '.feather': 'feather'
        }
        
        return type_mapping.get(ext, 'csv') 