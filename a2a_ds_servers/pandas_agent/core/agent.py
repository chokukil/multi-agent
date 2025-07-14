"""
Core pandas_agent implementation

This module contains the main PandasAgent class that provides
natural language interface to pandas data analysis with LLM integration.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union
import json
import time
import traceback
from datetime import datetime

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from helpers.logger import get_logger
from core.llm import LLMEngine


class PandasAgent:
    """
    Main pandas_agent class for LLM-powered natural language data analysis
    
    Provides intelligent data analysis capabilities including:
    - LLM-powered natural language query processing
    - Automated pandas code generation and execution
    - Intelligent result interpretation
    - Smart visualization suggestions
    - Context-aware insights generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pandas agent with LLM integration
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger()
        
        # Initialize LLM Engine
        self.llm_engine = LLMEngine(
            model=self.config.get('llm_model', 'gpt-4o-mini')
        )
        
        # Data storage
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.query_history: List[Dict[str, Any]] = []
        
        # Analysis context
        self.current_context: Dict[str, Any] = {
            "last_query": None,
            "last_results": None,
            "active_dataframe": None
        }
        
        self.logger.info("PandasAgent initialized with LLM integration")
    
    def load_dataframe(self, df: pd.DataFrame, name: str = "main") -> str:
        """
        Load a dataframe for analysis
        
        Args:
            df: Pandas DataFrame to load
            name: Name to assign to the dataframe
            
        Returns:
            DataFrame identifier
        """
        self.dataframes[name] = df
        self.current_context["active_dataframe"] = name
        
        # Log dataframe info
        info = self._get_dataframe_info(df)
        self.logger.info(f"Loaded dataframe '{name}': {info['shape'][0]} rows × {info['shape'][1]} columns")
        
        return name
    
    def load_from_file(self, file_path: str, name: str = "main") -> str:
        """
        Load dataframe from file
        
        Args:
            file_path: Path to the file
            name: Name to assign to the dataframe
            
        Returns:
            DataFrame identifier
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            return self.load_dataframe(df, name)
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    async def chat(self, query: str, df_name: str = "main") -> Dict[str, Any]:
        """
        Process natural language query about the data using LLM
        
        Args:
            query: Natural language query
            df_name: Name of the dataframe to analyze
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.log_query(query)
            self.current_context["last_query"] = query
            
            # Check if dataframe exists
            if df_name not in self.dataframes:
                return {
                    "error": f"Dataframe '{df_name}' not found. Please load data first.",
                    "available_dataframes": list(self.dataframes.keys()),
                    "suggestions": ["Use load_dataframe() or load_from_file() to load data first"]
                }
            
            df = self.dataframes[df_name]
            df_info = self._get_dataframe_info(df)
            
            # Phase 1: LLM-powered intent analysis
            self.logger.info("🧠 Analyzing query intent with LLM...")
            intent_analysis = await self.llm_engine.analyze_query_intent(query)
            
            # Phase 2: Generate pandas code using LLM
            self.logger.info("🔧 Generating pandas code with LLM...")
            code_generation = await self.llm_engine.generate_pandas_code(query, df_info)
            
            # Phase 3: Execute the generated code
            self.logger.info("⚡ Executing generated code...")
            execution_result = await self._execute_generated_code(
                code_generation.get('code', ''), 
                df, 
                df_name
            )
            
            # Phase 4: LLM-powered result interpretation
            self.logger.info("🎯 Interpreting results with LLM...")
            interpretation = await self.llm_engine.interpret_results(
                query, 
                execution_result.get('result'), 
                execution_result.get('execution_time', 0)
            )
            
            # Phase 5: Visualization suggestions
            visualization_suggestions = None
            if intent_analysis.get('visualization_needed') or 'plot' in query.lower():
                self.logger.info("📊 Generating visualization suggestions...")
                visualization_suggestions = await self.llm_engine.suggest_visualizations(
                    query, df_info, execution_result.get('result')
                )
            
            # Phase 6: Compile comprehensive response
            execution_time = time.time() - start_time
            
            result = {
                "status": "success",
                "query": query,
                "dataframe": df_name,
                "execution_time": execution_time,
                "intent_analysis": intent_analysis,
                "code_generation": code_generation,
                "execution_result": execution_result,
                "interpretation": interpretation,
                "visualization_suggestions": visualization_suggestions,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "agent_version": "1.0.0",
                    "llm_model": self.llm_engine.model,
                    "dataframe_info": df_info
                }
            }
            
            # Store in history and context
            self._update_history_and_context(result)
            
            # Log completion
            self.logger.log_result("llm_analysis", execution_time)
            
            return result
            
        except Exception as e:
            error_context = {
                "query": query, 
                "df_name": df_name, 
                "error_type": type(e).__name__
            }
            self.logger.log_error_with_context(e, error_context)
            
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "traceback": traceback.format_exc(),
                "suggestions": self._get_error_recovery_suggestions(e, query)
            }
    
    async def _execute_generated_code(self, code: str, df: pd.DataFrame, df_name: str) -> Dict[str, Any]:
        """
        Execute LLM-generated pandas code safely
        
        Args:
            code: Generated pandas code
            df: DataFrame to operate on
            df_name: DataFrame name for context
            
        Returns:
            Execution results
        """
        start_time = time.time()
        
        try:
            if not code or code.strip() == "":
                return {
                    "result": "No code generated",
                    "execution_time": 0,
                    "success": False,
                    "error": "Empty code"
                }
            
            # Prepare safe execution environment
            safe_globals = {
                'pd': pd,
                'df': df,
                'np': pd.np if hasattr(pd, 'np') else None,
                '__builtins__': {}  # Restrict built-ins for safety
            }
            
            # Execute the code
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            # Get the result (look for 'result' variable)
            result = local_vars.get('result', None)
            
            # If no 'result' variable, try to get the last expression
            if result is None and len(local_vars) > 0:
                result = list(local_vars.values())[-1]
            
            execution_time = time.time() - start_time
            
            return {
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "code_executed": code,
                "local_variables": list(local_vars.keys())
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error(f"Code execution failed: {e}")
            
            return {
                "result": None,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "code_executed": code
            }
    
    def _get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive dataframe information
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame information dictionary
        """
        try:
            # Basic info
            info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample": {}
            }
            
            # Add sample data (first few rows)
            try:
                sample_df = df.head(3)
                info["sample"] = sample_df.to_dict('records')
            except:
                info["sample"] = {"error": "Could not generate sample"}
            
            # Add statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                info["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting dataframe info: {e}")
            return {
                "shape": (0, 0),
                "columns": [],
                "dtypes": {},
                "error": str(e)
            }
    
    def _update_history_and_context(self, result: Dict[str, Any]):
        """Update query history and current context"""
        
        # Update history
        history_entry = {
            "query": result["query"],
            "timestamp": result["metadata"]["timestamp"],
            "execution_time": result["execution_time"],
            "dataframe": result["dataframe"],
            "success": result["status"] == "success",
            "intent": result.get("intent_analysis", {}).get("primary_intent", "unknown")
        }
        
        self.query_history.append(history_entry)
        
        # Keep only last 50 queries
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
        
        # Update context
        self.current_context.update({
            "last_query": result["query"],
            "last_results": result,
            "last_intent": result.get("intent_analysis", {}).get("primary_intent"),
            "last_execution_time": result["execution_time"]
        })
    
    def _get_error_recovery_suggestions(self, error: Exception, query: str) -> List[str]:
        """
        Generate error recovery suggestions
        
        Args:
            error: The exception that occurred
            query: The original query
            
        Returns:
            List of suggestions
        """
        error_type = type(error).__name__
        suggestions = []
        
        if "KeyError" in error_type:
            suggestions.extend([
                "Check if the column names in your query match the actual column names",
                f"Available columns: {list(self.dataframes.get(self.current_context.get('active_dataframe', 'main'), pd.DataFrame()).columns)}"
            ])
        elif "SyntaxError" in error_type:
            suggestions.extend([
                "The generated code has syntax errors. Try rephrasing your query",
                "Use simpler language and be more specific about what you want"
            ])
        elif "ValueError" in error_type:
            suggestions.extend([
                "Check if the data types are compatible with the requested operation",
                "Ensure the dataframe contains the expected data format"
            ])
        else:
            suggestions.extend([
                "Try rephrasing your query with different words",
                "Be more specific about what analysis you want",
                "Check if your dataframe contains the expected data"
            ])
        
        return suggestions
    
    # Existing methods (updated with better error handling)
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get query history"""
        return self.query_history
    
    def clear_history(self):
        """Clear query history and cache"""
        self.query_history.clear()
        self.llm_engine.clear_cache()
        self.current_context = {
            "last_query": None,
            "last_results": None,
            "active_dataframe": None
        }
        self.logger.info("Query history and cache cleared")
    
    def get_available_dataframes(self) -> List[str]:
        """Get list of available dataframes"""
        return list(self.dataframes.keys())
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "dataframes_loaded": len(self.dataframes),
            "queries_processed": len(self.query_history),
            "llm_cache_stats": self.llm_engine.get_cache_stats(),
            "current_context": self.current_context,
            "last_queries": self.query_history[-5:] if self.query_history else []
        }
    
    def get_dataframe_summary(self, df_name: str = "main") -> Dict[str, Any]:
        """
        Get comprehensive summary of a dataframe
        
        Args:
            df_name: Name of the dataframe
            
        Returns:
            Dataframe summary
        """
        if df_name not in self.dataframes:
            return {"error": f"Dataframe '{df_name}' not found"}
        
        df = self.dataframes[df_name]
        return self._get_dataframe_info(df) 