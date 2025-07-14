"""
Smart DataFrame implementation for pandas_agent

This module provides an intelligent wrapper around pandas DataFrame
that enables natural language interaction and context-aware analysis.
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
import json
import uuid
from datetime import datetime
from functools import cached_property

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from helpers.logger import get_logger
from core.llm import LLMEngine


class SmartDataFrame:
    """
    Intelligent DataFrame wrapper with natural language capabilities
    
    Enhances pandas DataFrame with:
    - Natural language query interface
    - Context-aware analysis
    - Automatic insight generation
    - Smart caching and optimization
    - Interactive exploration capabilities
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 custom_head: Optional[pd.DataFrame] = None,
                 llm_engine: Optional[LLMEngine] = None):
        """
        Initialize Smart DataFrame
        
        Args:
            df: Pandas DataFrame to wrap
            name: Optional name for the dataframe
            description: Optional description of the data
            custom_head: Optional custom head to show for privacy
            llm_engine: Optional LLM engine instance
        """
        self._original_df = df.copy()
        self._df = df
        self._name = name or f"smartdf_{uuid.uuid4().hex[:8]}"
        self._description = description
        self._custom_head = custom_head
        
        # Initialize LLM engine
        self.llm_engine = llm_engine or LLMEngine()
        self.logger = get_logger()
        
        # Analysis context and history
        self._analysis_context = {
            "queries": [],
            "insights": [],
            "cached_results": {},
            "data_profile": None
        }
        
        # Generate initial data profile
        self._generate_data_profile()
        
        self.logger.info(f"SmartDataFrame '{self._name}' initialized with shape {df.shape}")
    
    def _generate_data_profile(self):
        """Generate comprehensive data profile for intelligent analysis"""
        try:
            profile = {
                "basic_info": {
                    "shape": self._df.shape,
                    "columns": list(self._df.columns),
                    "dtypes": self._df.dtypes.to_dict(),
                    "memory_usage": self._df.memory_usage(deep=True).sum(),
                    "index_type": str(type(self._df.index).__name__)
                },
                "data_quality": {
                    "null_counts": self._df.isnull().sum().to_dict(),
                    "null_percentages": (self._df.isnull().sum() / len(self._df) * 100).round(2).to_dict(),
                    "duplicate_rows": self._df.duplicated().sum(),
                    "unique_counts": {col: self._df[col].nunique() for col in self._df.columns}
                },
                "statistical_summary": {},
                "data_types": {
                    "numeric_columns": list(self._df.select_dtypes(include=['number']).columns),
                    "categorical_columns": list(self._df.select_dtypes(include=['object', 'category']).columns),
                    "datetime_columns": list(self._df.select_dtypes(include=['datetime']).columns),
                    "boolean_columns": list(self._df.select_dtypes(include=['bool']).columns)
                },
                "generated_at": datetime.now().isoformat()
            }
            
            # Add statistical summary for numeric columns
            numeric_cols = profile["data_types"]["numeric_columns"]
            if numeric_cols:
                profile["statistical_summary"] = self._df[numeric_cols].describe().to_dict()
            
            # Add categorical insights
            categorical_cols = profile["data_types"]["categorical_columns"]
            if categorical_cols:
                profile["categorical_insights"] = {}
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    try:
                        value_counts = self._df[col].value_counts().head(10)
                        profile["categorical_insights"][col] = {
                            "top_values": value_counts.to_dict(),
                            "unique_count": self._df[col].nunique(),
                            "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None
                        }
                    except:
                        pass
            
            self._analysis_context["data_profile"] = profile
            
        except Exception as e:
            self.logger.error(f"Error generating data profile: {e}")
            self._analysis_context["data_profile"] = {"error": str(e)}
    
    async def chat(self, query: str) -> Dict[str, Any]:
        """
        Natural language chat interface for data analysis
        
        Args:
            query: Natural language query about the data
            
        Returns:
            Comprehensive analysis results
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self._analysis_context["cached_results"]:
                cached_result = self._analysis_context["cached_results"][cache_key]
                cached_result["from_cache"] = True
                return cached_result
            
            # Phase 1: Intent analysis with context
            intent_analysis = await self.llm_engine.analyze_query_intent(query)
            
            # Phase 2: Enhanced pandas code generation using data profile
            code_generation = await self.llm_engine.generate_pandas_code(
                query, 
                self._analysis_context["data_profile"]["basic_info"]
            )
            
            # Phase 3: Safe code execution
            execution_result = await self._execute_code_safely(code_generation.get('code', ''))
            
            # Phase 4: Context-aware interpretation
            interpretation = await self.llm_engine.interpret_results(
                query, 
                execution_result.get('result'), 
                execution_result.get('execution_time', 0)
            )
            
            # Phase 5: Generate insights and suggestions
            insights = await self._generate_contextual_insights(
                query, execution_result, intent_analysis
            )
            
            # Phase 6: Visualization recommendations
            viz_suggestions = None
            if intent_analysis.get('visualization_needed', False) or 'plot' in query.lower():
                viz_suggestions = await self.llm_engine.suggest_visualizations(
                    query, 
                    self._analysis_context["data_profile"]["basic_info"],
                    execution_result.get('result')
                )
            
            # Compile comprehensive result
            result = {
                "query": query,
                "timestamp": start_time.isoformat(),
                "dataframe_name": self._name,
                "intent_analysis": intent_analysis,
                "code_generation": code_generation,
                "execution_result": execution_result,
                "interpretation": interpretation,
                "contextual_insights": insights,
                "visualization_suggestions": viz_suggestions,
                "data_context": {
                    "shape": self._df.shape,
                    "columns_analyzed": list(self._df.columns),
                    "data_types": self._analysis_context["data_profile"]["data_types"]
                },
                "performance": {
                    "total_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "cached": False
                }
            }
            
            # Cache the result
            self._analysis_context["cached_results"][cache_key] = result
            
            # Update analysis history
            self._analysis_context["queries"].append({
                "query": query,
                "timestamp": start_time.isoformat(),
                "intent": intent_analysis.get('primary_intent', 'unknown'),
                "success": execution_result.get('success', False)
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"SmartDataFrame chat error: {e}")
            return {
                "error": str(e),
                "query": query,
                "timestamp": start_time.isoformat(),
                "dataframe_name": self._name,
                "suggestions": self._get_error_suggestions(e, query)
            }
    
    async def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """
        Execute generated code safely with SmartDataFrame context
        
        Args:
            code: Generated pandas code
            
        Returns:
            Execution results
        """
        start_time = datetime.now()
        
        try:
            if not code or code.strip() == "":
                return {
                    "result": None,
                    "success": False,
                    "error": "No code generated",
                    "execution_time": 0
                }
            
            # Prepare execution environment with SmartDataFrame context
            safe_globals = {
                'pd': pd,
                'df': self._df,
                'original_df': self._original_df,
                'numpy': pd.np if hasattr(pd, 'np') else None,
                'np': pd.np if hasattr(pd, 'np') else None,
                '__builtins__': {}
            }
            
            # Execute code
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            # Extract result
            result = local_vars.get('result', None)
            if result is None and len(local_vars) > 0:
                # Get the last assigned variable
                result = list(local_vars.values())[-1]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": result,
                "success": True,
                "execution_time": execution_time,
                "code_executed": code,
                "variables_created": list(local_vars.keys())
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": None,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time,
                "code_executed": code
            }
    
    async def _generate_contextual_insights(self, 
                                          query: str, 
                                          execution_result: Dict[str, Any],
                                          intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate contextual insights based on analysis results and data profile
        
        Args:
            query: Original query
            execution_result: Code execution results
            intent_analysis: Intent analysis results
            
        Returns:
            Contextual insights
        """
        try:
            # Basic insights from data profile
            profile = self._analysis_context["data_profile"]
            basic_insights = []
            
            # Data size insights
            rows, cols = profile["basic_info"]["shape"]
            basic_insights.append(f"Dataset contains {rows:,} rows and {cols} columns")
            
            # Data quality insights
            total_nulls = sum(profile["data_quality"]["null_counts"].values())
            if total_nulls > 0:
                null_pct = (total_nulls / (rows * cols)) * 100
                basic_insights.append(f"Data has {null_pct:.1f}% missing values")
            
            # Data type insights
            dtypes = profile["data_types"]
            basic_insights.append(
                f"Data types: {len(dtypes['numeric_columns'])} numeric, "
                f"{len(dtypes['categorical_columns'])} categorical columns"
            )
            
            # Analysis-specific insights
            analysis_insights = []
            if execution_result.get('success'):
                result = execution_result.get('result')
                if result is not None:
                    if hasattr(result, 'shape'):
                        analysis_insights.append(f"Analysis returned a {type(result).__name__} with shape {result.shape}")
                    elif isinstance(result, (int, float)):
                        analysis_insights.append(f"Analysis returned a numeric value: {result}")
                    elif isinstance(result, str):
                        analysis_insights.append("Analysis returned a text result")
            
            # Intent-based insights
            intent_insights = []
            intent = intent_analysis.get('primary_intent', 'unknown')
            if intent == 'visualization':
                intent_insights.append("Consider using the visualization suggestions below")
            elif intent == 'summary':
                intent_insights.append("Summary analysis provides overview of data characteristics")
            elif intent == 'correlation':
                intent_insights.append("Correlation analysis helps understand relationships between variables")
            
            return {
                "basic_insights": basic_insights,
                "analysis_insights": analysis_insights,
                "intent_insights": intent_insights,
                "data_quality_score": self._calculate_data_quality_score(),
                "recommendations": self._generate_recommendations(intent_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating contextual insights: {e}")
            return {
                "basic_insights": ["Error generating insights"],
                "error": str(e)
            }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate data quality score (0-1)"""
        try:
            profile = self._analysis_context["data_profile"]
            
            # Completeness score (no missing values = 1.0)
            total_cells = profile["basic_info"]["shape"][0] * profile["basic_info"]["shape"][1]
            total_nulls = sum(profile["data_quality"]["null_counts"].values())
            completeness = 1.0 - (total_nulls / total_cells) if total_cells > 0 else 0.0
            
            # Uniqueness score (no duplicates = 1.0)
            total_rows = profile["basic_info"]["shape"][0]
            duplicate_rows = profile["data_quality"]["duplicate_rows"]
            uniqueness = 1.0 - (duplicate_rows / total_rows) if total_rows > 0 else 0.0
            
            # Combined score (weighted average)
            return round((completeness * 0.7 + uniqueness * 0.3), 2)
            
        except:
            return 0.5  # Default middle score if calculation fails
    
    def _generate_recommendations(self, intent_analysis: Dict[str, Any]) -> List[str]:
        """Generate analysis recommendations based on intent and data profile"""
        recommendations = []
        
        try:
            profile = self._analysis_context["data_profile"]
            intent = intent_analysis.get('primary_intent', 'unknown')
            
            # Data quality recommendations
            total_nulls = sum(profile["data_quality"]["null_counts"].values())
            if total_nulls > 0:
                recommendations.append("Consider handling missing values before analysis")
            
            if profile["data_quality"]["duplicate_rows"] > 0:
                recommendations.append("Check for and handle duplicate rows")
            
            # Intent-specific recommendations
            if intent == 'visualization':
                numeric_cols = len(profile["data_types"]["numeric_columns"])
                if numeric_cols > 0:
                    recommendations.append("Try histogram or scatter plots for numeric data")
                
                categorical_cols = len(profile["data_types"]["categorical_columns"])
                if categorical_cols > 0:
                    recommendations.append("Consider bar charts for categorical data")
            
            elif intent == 'correlation':
                numeric_cols = len(profile["data_types"]["numeric_columns"])
                if numeric_cols < 2:
                    recommendations.append("Correlation analysis requires at least 2 numeric columns")
                else:
                    recommendations.append("Use correlation matrix or heatmap for better insights")
            
            elif intent == 'summary':
                recommendations.append("Explore specific columns for detailed insights")
                recommendations.append("Consider groupby operations for categorical analysis")
            
            # General recommendations
            if len(profile["basic_info"]["columns"]) > 20:
                recommendations.append("Consider focusing on specific columns for better analysis")
            
            return recommendations
            
        except:
            return ["Explore the data with basic operations like describe() or info()"]
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        
        # Include dataframe shape and query in cache key
        cache_data = f"{query}_{self._df.shape}_{hash(tuple(self._df.columns))}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_error_suggestions(self, error: Exception, query: str) -> List[str]:
        """Generate error-specific suggestions"""
        error_type = type(error).__name__
        suggestions = []
        
        if "KeyError" in error_type:
            suggestions.extend([
                "Check column names in your query",
                f"Available columns: {list(self._df.columns)}"
            ])
        elif "ValueError" in error_type:
            suggestions.extend([
                "Check data types compatibility",
                "Ensure the operation is valid for the data"
            ])
        else:
            suggestions.extend([
                "Try rephrasing your question",
                "Be more specific about what you want to analyze"
            ])
        
        return suggestions
    
    # Properties and utility methods
    
    @property
    def name(self) -> str:
        """Get dataframe name"""
        return self._name
    
    @property
    def description(self) -> Optional[str]:
        """Get dataframe description"""
        return self._description
    
    @property
    def shape(self) -> tuple:
        """Get dataframe shape"""
        return self._df.shape
    
    @property
    def columns(self) -> pd.Index:
        """Get dataframe columns"""
        return self._df.columns
    
    @property
    def dtypes(self) -> pd.Series:
        """Get dataframe dtypes"""
        return self._df.dtypes
    
    @cached_property
    def head_df(self) -> pd.DataFrame:
        """Get head of dataframe (or custom head)"""
        if self._custom_head is not None:
            return self._custom_head
        return self._df.head()
    
    def get_data_profile(self) -> Dict[str, Any]:
        """Get comprehensive data profile"""
        return self._analysis_context["data_profile"]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self._analysis_context["queries"]
    
    def clear_cache(self):
        """Clear analysis cache"""
        self._analysis_context["cached_results"].clear()
        self.logger.info(f"Cache cleared for SmartDataFrame '{self._name}'")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cached_queries": len(self._analysis_context["cached_results"]),
            "total_queries": len(self._analysis_context["queries"]),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = len(self._analysis_context["queries"])
        if total == 0:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd track actual cache hits
        cached = len(self._analysis_context["cached_results"])
        return min(cached / total, 1.0)
    
    # DataFrame delegation methods
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying DataFrame"""
        return getattr(self._df, name)
    
    def __getitem__(self, key):
        """Delegate indexing to underlying DataFrame"""
        return self._df[key]
    
    def __len__(self):
        """Get length of DataFrame"""
        return len(self._df)
    
    def __repr__(self):
        """String representation"""
        return f"SmartDataFrame(name='{self._name}', shape={self._df.shape})" 