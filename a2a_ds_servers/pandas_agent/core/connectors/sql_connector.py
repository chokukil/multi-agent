"""
SQL Database connector for pandas_agent

This module provides SQL database connectivity using SQLAlchemy,
supporting multiple database types (PostgreSQL, MySQL, SQLite, etc.)
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from urllib.parse import quote_plus

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from core.connectors.base_connector import DataSourceConnector


class SQLConnector(DataSourceConnector):
    """
    SQL Database connector using SQLAlchemy
    
    Supports multiple database types:
    - PostgreSQL
    - MySQL
    - SQLite
    - SQL Server
    - Oracle
    """
    
    def __init__(self, 
                 name: str,
                 connection_string: Optional[str] = None,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 database: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 db_type: str = "postgresql",
                 **engine_kwargs):
        """
        Initialize SQL connector
        
        Args:
            name: Name of the connection
            connection_string: Full connection string (if provided, other params ignored)
            host: Database host
            port: Database port
            database: Database name
            username: Username
            password: Password
            db_type: Database type (postgresql, mysql, sqlite, mssql, oracle)
            **engine_kwargs: Additional SQLAlchemy engine parameters
        """
        
        # Build connection config
        config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "database": database,
            "username": username,
            "connection_string": connection_string,
            **engine_kwargs
        }
        
        super().__init__(name, config)
        
        self.db_type = db_type
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.engine_kwargs = engine_kwargs
        
        # Connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = self._build_connection_string()
        
        self.engine = None
        self._tables_cache = None
    
    def _build_connection_string(self) -> str:
        """Build connection string from components"""
        
        if not all([self.host, self.database, self.username]):
            raise ValueError("host, database, and username are required")
        
        # URL-encode password to handle special characters
        encoded_password = quote_plus(self.password) if self.password else ""
        
        # Default ports for different databases
        default_ports = {
            "postgresql": 5432,
            "mysql": 3306,
            "mssql": 1433,
            "oracle": 1521,
            "sqlite": None
        }
        
        port = self.port or default_ports.get(self.db_type)
        
        if self.db_type == "postgresql":
            driver = "postgresql+psycopg2"
        elif self.db_type == "mysql":
            driver = "mysql+pymysql"
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.db_type == "mssql":
            driver = "mssql+pyodbc"
        elif self.db_type == "oracle":
            driver = "oracle+cx_oracle"
        else:
            driver = self.db_type
        
        if port:
            return f"{driver}://{self.username}:{encoded_password}@{self.host}:{port}/{self.database}"
        else:
            return f"{driver}://{self.username}:{encoded_password}@{self.host}/{self.database}"
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                **self.engine_kwargs
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._is_connected = True
            self.logger.info(f"SQL connector '{self.name}' connected to {self.db_type} database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to SQL database: {e}")
            self._is_connected = False
            return False
    
    async def disconnect(self):
        """Close database connection"""
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
            
            self._is_connected = False
            self._tables_cache = None
            self.logger.info(f"SQL connector '{self.name}' disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from SQL database: {e}")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test database connection"""
        try:
            if not self.engine:
                await self.connect()
            
            with self.engine.connect() as conn:
                # Get database info
                result = conn.execute(text("SELECT version()"))
                version_info = result.fetchone()
                
                return {
                    "success": True,
                    "database_type": self.db_type,
                    "version": str(version_info[0]) if version_info else "Unknown",
                    "connection_string": self.connection_string.replace(self.password or "", "***"),
                    "engine_info": str(self.engine.url)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "database_type": self.db_type
            }
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            if not self.engine:
                await self.connect()
            
            inspector = inspect(self.engine)
            
            # Get tables
            tables = inspector.get_table_names()
            
            # Get schema info for each table
            schema_info = {
                "database": self.database,
                "tables": {},
                "total_tables": len(tables)
            }
            
            for table in tables[:20]:  # Limit to first 20 tables
                try:
                    columns = inspector.get_columns(table)
                    schema_info["tables"][table] = {
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col.get("nullable", True),
                                "default": col.get("default")
                            }
                            for col in columns
                        ],
                        "column_count": len(columns)
                    }
                except Exception as e:
                    self.logger.warning(f"Error getting schema for table {table}: {e}")
                    continue
            
            self._tables_cache = tables
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {e}")
            return {"error": str(e)}
    
    async def read_data(self, 
                       query: Optional[str] = None,
                       table_name: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """Read data from database"""
        try:
            if not self.engine:
                await self.connect()
            
            if query:
                # Execute custom query
                df = pd.read_sql(query, self.engine, **kwargs)
                self.logger.info(f"Executed custom query, returned {len(df)} rows")
                
            elif table_name:
                # Read entire table
                df = pd.read_sql_table(table_name, self.engine, **kwargs)
                self.logger.info(f"Read table '{table_name}', returned {len(df)} rows")
                
            else:
                raise ValueError("Either query or table_name must be provided")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading data from SQL database: {e}")
            raise
    
    async def get_data_preview(self, 
                              table_name: str,
                              limit: int = 100) -> pd.DataFrame:
        """Get preview of table data"""
        try:
            if not self.engine:
                await self.connect()
            
            # Create LIMIT query
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            
            # For some databases, use different syntax
            if self.db_type in ["mssql", "oracle"]:
                query = f"SELECT TOP {limit} * FROM {table_name}"
            
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"Retrieved preview of table '{table_name}': {len(df)} rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data preview: {e}")
            raise
    
    async def get_table_list(self) -> List[str]:
        """Get list of available tables"""
        try:
            if self._tables_cache:
                return self._tables_cache
            
            if not self.engine:
                await self.connect()
            
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            self._tables_cache = tables
            return tables
            
        except Exception as e:
            self.logger.error(f"Error getting table list: {e}")
            return []
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a query and return results with metadata"""
        try:
            if not self.engine:
                await self.connect()
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # Check if query returns data
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    return {
                        "success": True,
                        "data": df,
                        "row_count": len(df),
                        "column_count": len(df.columns) if not df.empty else 0,
                        "columns": list(df.columns) if not df.empty else []
                    }
                else:
                    # Non-SELECT query (INSERT, UPDATE, DELETE, etc.)
                    return {
                        "success": True,
                        "affected_rows": result.rowcount,
                        "message": "Query executed successfully"
                    }
                    
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        try:
            if not self.engine:
                await self.connect()
            
            inspector = inspect(self.engine)
            
            # Get columns
            columns = inspector.get_columns(table_name)
            
            # Get row count
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.fetchone()[0]
            
            # Get primary keys
            try:
                pk_constraint = inspector.get_pk_constraint(table_name)
                primary_keys = pk_constraint.get("constrained_columns", [])
            except:
                primary_keys = []
            
            # Get foreign keys
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
            except:
                foreign_keys = []
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "default": col.get("default"),
                        "is_primary_key": col["name"] in primary_keys
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys
            }
            
        except Exception as e:
            self.logger.error(f"Error getting table info for {table_name}: {e}")
            return {"error": str(e), "table_name": table_name}


class SQLiteConnector(SQLConnector):
    """Specialized SQLite connector"""
    
    def __init__(self, name: str, database_path: str, **kwargs):
        """
        Initialize SQLite connector
        
        Args:
            name: Connection name
            database_path: Path to SQLite database file
            **kwargs: Additional engine parameters
        """
        super().__init__(
            name=name,
            connection_string=f"sqlite:///{database_path}",
            db_type="sqlite",
            **kwargs
        )
        self.database_path = database_path 