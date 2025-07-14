"""
LLM integration engine for pandas_agent

This module provides LLM-powered natural language processing capabilities
for pandas data analysis following CherryAI's LLM First principles.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from core.llm_factory import create_llm_instance
except ImportError:
    create_llm_instance = None

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from helpers.logger import get_logger


class LLMEngine:
    """
    LLM Engine for pandas_agent
    
    Provides natural language processing capabilities for:
    - Query intent analysis
    - Pandas code generation
    - Result interpretation
    - Visualization suggestions
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize LLM Engine
        
        Args:
            model: LLM model to use
        """
        self.model = model
        self.logger = get_logger()
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai_client()
        
        # Initialize LangChain LLM (fallback)
        self.langchain_llm = self._initialize_langchain_llm()
        
        # Query processing cache
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"LLM Engine initialized with model: {model}")
    
    def _initialize_openai_client(self) -> Optional[AsyncOpenAI]:
        """Initialize OpenAI client following CherryAI pattern"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                client = AsyncOpenAI(api_key=api_key)
                self.logger.info("✅ OpenAI client initialized")
                return client
            else:
                self.logger.warning("⚠️ OPENAI_API_KEY not found")
                return None
        except Exception as e:
            self.logger.error(f"❌ OpenAI client initialization failed: {e}")
            return None
    
    def _initialize_langchain_llm(self) -> Optional[Any]:
        """Initialize LangChain LLM as fallback"""
        try:
            if create_llm_instance:
                llm = create_llm_instance()
                self.logger.info("✅ LangChain LLM initialized")
                return llm
            else:
                self.logger.warning("⚠️ LangChain LLM factory not available")
                return None
        except Exception as e:
            self.logger.error(f"❌ LangChain LLM initialization failed: {e}")
            return None
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query intent using LLM
        
        Args:
            query: Natural language query
            
        Returns:
            Query analysis results
        """
        # Check cache first
        cache_key = f"intent_{hash(query)}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        prompt = f"""
Analyze the following data analysis query and determine the user's intent.

Query: "{query}"

Analyze and respond with JSON in this exact format:
{{
    "primary_intent": "one of: summary, exploration, visualization, statistics, filtering, aggregation, comparison, correlation, missing_data, general",
    "confidence": "number between 0.0 and 1.0",
    "data_requirements": ["list of data types/columns likely needed"],
    "suggested_analysis": ["list of specific analysis steps"],
    "visualization_needed": true/false,
    "complexity_level": "one of: simple, medium, complex",
    "pandas_operations": ["list of pandas operations that might be needed"],
    "output_format": "one of: dataframe, plot, summary, statistics, insights"
}}

Focus on understanding what the user wants to achieve with their data.
"""
        
        try:
            result = await self._call_llm_async(prompt)
            analysis = self._parse_json_response(result)
            
            # Cache the result
            self.query_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Query intent analysis failed: {e}")
            return self._create_fallback_intent(query)
    
    async def generate_pandas_code(self, query: str, df_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate pandas code for the query
        
        Args:
            query: Natural language query
            df_info: DataFrame information (columns, dtypes, shape, etc.)
            
        Returns:
            Generated code and explanation
        """
        columns_info = df_info.get('columns', [])
        dtypes_info = df_info.get('dtypes', {})
        shape_info = df_info.get('shape', (0, 0))
        sample_data = df_info.get('sample', {})
        
        prompt = f"""
Generate pandas code to answer the following query about a dataset.

Query: "{query}"

Dataset Information:
- Shape: {shape_info[0]} rows, {shape_info[1]} columns
- Columns: {columns_info}
- Data Types: {dtypes_info}
- Sample Data: {sample_data}

Generate Python pandas code that:
1. Directly answers the user's question
2. Uses only the available columns
3. Handles missing data appropriately
4. Is efficient and follows pandas best practices
5. Returns meaningful results

Respond with JSON in this exact format:
{{
    "code": "# Main pandas code\\nresult = df.some_operation()",
    "explanation": "Clear explanation of what the code does",
    "expected_output": "Description of what the result will be",
    "visualization_code": "# Optional matplotlib/seaborn code if visualization is needed",
    "error_handling": "# Additional code for error handling if needed",
    "assumptions": ["List any assumptions made about the data"]
}}

The variable 'df' represents the DataFrame. Generate practical, working code.
"""
        
        try:
            result = await self._call_llm_async(prompt)
            code_analysis = self._parse_json_response(result)
            
            return code_analysis
            
        except Exception as e:
            self.logger.error(f"Pandas code generation failed: {e}")
            return self._create_fallback_code(query)
    
    async def interpret_results(self, query: str, result_data: Any, execution_time: float) -> Dict[str, Any]:
        """
        Interpret analysis results using LLM
        
        Args:
            query: Original query
            result_data: Analysis results
            execution_time: Execution time in seconds
            
        Returns:
            Interpreted insights
        """
        # Convert result to string representation
        if hasattr(result_data, 'to_string'):
            result_str = result_data.to_string()[:1000]  # Limit size
        elif hasattr(result_data, 'to_dict'):
            result_str = str(result_data.to_dict())[:1000]
        else:
            result_str = str(result_data)[:1000]
        
        prompt = f"""
Interpret the following data analysis results and provide insights.

Original Query: "{query}"
Analysis Results: {result_str}
Execution Time: {execution_time:.2f} seconds

Provide insights in JSON format:
{{
    "key_findings": ["List of 3-5 key insights from the results"],
    "statistical_summary": "Brief statistical interpretation",
    "business_implications": ["Potential business or practical implications"],
    "data_quality_notes": ["Any observations about data quality"],
    "recommendations": ["Suggested next analysis steps"],
    "confidence_level": "Assessment of result reliability (high/medium/low)",
    "interpretation": "Plain English explanation of what the results mean"
}}

Focus on making the technical results accessible and actionable.
"""
        
        try:
            result = await self._call_llm_async(prompt)
            interpretation = self._parse_json_response(result)
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Result interpretation failed: {e}")
            return self._create_fallback_interpretation(query, result_data)
    
    async def suggest_visualizations(self, query: str, df_info: Dict[str, Any], analysis_result: Any) -> Dict[str, Any]:
        """
        Suggest appropriate visualizations
        
        Args:
            query: Original query
            df_info: DataFrame information
            analysis_result: Analysis results
            
        Returns:
            Visualization suggestions
        """
        columns_info = df_info.get('columns', [])
        dtypes_info = df_info.get('dtypes', {})
        
        prompt = f"""
Suggest appropriate visualizations for the following data analysis.

Query: "{query}"
Available Columns: {columns_info}
Data Types: {dtypes_info}
Analysis Result Type: {type(analysis_result).__name__}

Suggest visualizations in JSON format:
{{
    "recommended_plots": [
        {{
            "plot_type": "histogram/scatter/bar/line/box/heatmap/etc",
            "reason": "Why this plot is suitable",
            "matplotlib_code": "plt.figure()\\n# matplotlib code",
            "seaborn_code": "sns.plot()\\n# seaborn alternative",
            "columns_needed": ["list of columns for this plot"],
            "difficulty": "easy/medium/hard"
        }}
    ],
    "plot_priority": "Which plot to show first",
    "styling_suggestions": "Color schemes, themes, etc.",
    "interactive_options": "Suggestions for interactive plots (plotly, etc.)"
}}

Focus on plots that best answer the user's question and are appropriate for the data types.
"""
        
        try:
            result = await self._call_llm_async(prompt)
            visualization_suggestions = self._parse_json_response(result)
            
            return visualization_suggestions
            
        except Exception as e:
            self.logger.error(f"Visualization suggestions failed: {e}")
            return self._create_fallback_visualizations()
    
    async def _call_llm_async(self, prompt: str) -> str:
        """Call LLM asynchronously with fallback options"""
        
        # Try OpenAI first
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                    timeout=60
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"OpenAI call failed: {e}, trying fallback")
        
        # Try LangChain LLM
        if self.langchain_llm:
            try:
                if hasattr(self.langchain_llm, 'ainvoke'):
                    response = await self.langchain_llm.ainvoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                elif hasattr(self.langchain_llm, 'acall'):
                    return await self.langchain_llm.acall(prompt)
                else:
                    # Fallback to sync call
                    return await asyncio.get_event_loop().run_in_executor(
                        None, self._call_llm_sync, prompt
                    )
            except Exception as e:
                self.logger.warning(f"LangChain LLM call failed: {e}")
        
        # Ultimate fallback
        raise RuntimeError("No LLM available for processing")
    
    def _call_llm_sync(self, prompt: str) -> str:
        """Synchronous LLM call (fallback)"""
        if hasattr(self.langchain_llm, 'invoke'):
            response = self.langchain_llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        elif hasattr(self.langchain_llm, 'call'):
            return self.langchain_llm.call(prompt)
        else:
            return self.langchain_llm(prompt)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                # No JSON found, create a simple response
                return {"content": response, "parsed": False}
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}")
            return {"content": response, "parsed": False, "error": str(e)}
    
    def _create_fallback_intent(self, query: str) -> Dict[str, Any]:
        """Create fallback intent analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['plot', 'chart', 'visualize', 'graph']):
            primary_intent = "visualization"
        elif any(word in query_lower for word in ['describe', 'summary', 'overview']):
            primary_intent = "summary"
        elif any(word in query_lower for word in ['correlation', 'relationship']):
            primary_intent = "correlation"
        elif any(word in query_lower for word in ['null', 'missing', 'nan']):
            primary_intent = "missing_data"
        else:
            primary_intent = "general"
        
        return {
            "primary_intent": primary_intent,
            "confidence": 0.6,
            "data_requirements": ["any"],
            "suggested_analysis": ["basic exploration"],
            "visualization_needed": "plot" in query_lower,
            "complexity_level": "medium",
            "pandas_operations": ["info", "describe"],
            "output_format": "summary"
        }
    
    def _create_fallback_code(self, query: str) -> Dict[str, Any]:
        """Create fallback pandas code"""
        return {
            "code": "# Basic data exploration\nresult = df.describe()",
            "explanation": "Generating basic statistical summary of the dataset",
            "expected_output": "Statistical summary with count, mean, std, min, max, etc.",
            "visualization_code": "# df.hist(figsize=(12, 8))",
            "error_handling": "# Check if df is not empty before processing",
            "assumptions": ["Dataset contains numeric columns for statistical summary"]
        }
    
    def _create_fallback_interpretation(self, query: str, result_data: Any) -> Dict[str, Any]:
        """Create fallback result interpretation"""
        return {
            "key_findings": ["Analysis completed successfully"],
            "statistical_summary": "Statistical analysis performed on the dataset",
            "business_implications": ["Results provide insights into data patterns"],
            "data_quality_notes": ["Data processed without major issues"],
            "recommendations": ["Consider further analysis based on initial findings"],
            "confidence_level": "medium",
            "interpretation": "The analysis has been completed and results are available for review"
        }
    
    def _create_fallback_visualizations(self) -> Dict[str, Any]:
        """Create fallback visualization suggestions"""
        return {
            "recommended_plots": [
                {
                    "plot_type": "histogram",
                    "reason": "Good for showing distribution of numeric data",
                    "matplotlib_code": "df.hist(figsize=(12, 8))",
                    "seaborn_code": "sns.histplot(data=df)",
                    "columns_needed": ["numeric columns"],
                    "difficulty": "easy"
                }
            ],
            "plot_priority": "histogram",
            "styling_suggestions": "Use default color scheme",
            "interactive_options": "Consider plotly for interactive exploration"
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.query_cache),
            "has_openai": self.openai_client is not None,
            "has_langchain": self.langchain_llm is not None,
            "model": self.model
        } 