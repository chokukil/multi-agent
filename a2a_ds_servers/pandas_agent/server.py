"""
CherryAI pandas_agent A2A Server (Enhanced)

This module provides a fully-featured A2A-compliant server wrapper for the pandas_agent,
enabling comprehensive natural language data analysis through the Agent-to-Agent protocol.

Features:
- A2A SDK 0.2.9 compliant
- SSE streaming support
- LLM-powered natural language processing
- SmartDataFrame integration
- Automatic visualization generation
- Multi-datasource connectivity
- Intelligent caching
"""

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core.agent import PandasAgent
from core.smart_dataframe import SmartDataFrame
from core.visualization import AutoVisualizationEngine
from core.connectors import FileConnector, SQLConnector
from helpers.logger import get_logger
from helpers.cache import get_cache_manager
from helpers.df_info import DataFrameInfo


class PandasAgentExecutor(AgentExecutor):
    """
    Enhanced A2A Agent Executor for pandas_agent
    
    Provides comprehensive natural language interface to pandas data analysis
    following LLM First principles and A2A protocol standards.
    """
    
    def __init__(self):
        self.agent = None
        self.smart_dataframes: Dict[str, SmartDataFrame] = {}
        self.connectors: Dict[str, Any] = {}
        self.viz_engine = AutoVisualizationEngine()
        self.cache_manager = get_cache_manager(
            max_size_mb=50,
            persistent=True,
            cache_dir="a2a_ds_servers/pandas_agent/.cache"
        )
        self.logger = get_logger()
        
        self.logger.info("Enhanced PandasAgentExecutor initialized")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        Execute comprehensive pandas_agent analysis request
        
        Args:
            context: A2A request context containing user message
            task_updater: A2A task updater for streaming responses
        """
        try:
            # Initialize task
            await task_updater.update_status(
                TaskState.working,
                message="🚀 Starting comprehensive data analysis..."
            )
            
            # Extract and parse user query
            user_query = self._extract_query_from_context(context)
            self.logger.info(f"Processing query: {user_query}")
            
            # Check cache first
            cache_key = f"query:{user_query}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                await task_updater.update_status(
                    TaskState.working,
                    message="📦 Found cached results, preparing response..."
                )
                
                await self._send_cached_response(cached_result, task_updater)
                
                await task_updater.update_status(
                    TaskState.completed,
                    message="✅ Analysis completed from cache"
                )
                return
            
            # Initialize pandas agent if not exists
            if self.agent is None:
                await task_updater.update_status(
                    TaskState.working,
                    message="🧠 Initializing intelligent pandas agent..."
                )
                self.agent = PandasAgent()
            
            # Parse query for data source information
            await task_updater.update_status(
                TaskState.working,
                message="🔍 Analyzing query and detecting data requirements..."
            )
            
            query_analysis = await self._analyze_query_requirements(user_query)
            
            # Handle data loading if needed
            dataframe_name = "main"
            if query_analysis.get("needs_data_loading", False):
                await task_updater.update_status(
                    TaskState.working,
                    message="📁 Loading data from specified source..."
                )
                
                dataframe_name = await self._handle_data_loading(query_analysis, task_updater)
                
                if not dataframe_name:
                    await task_updater.update_status(
                        TaskState.failed,
                        message="❌ Failed to load data source"
                    )
                    return
            
            # Check if we have data to work with
            if not self.smart_dataframes and dataframe_name not in self.agent.dataframes:
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 No data loaded. Creating sample data for demonstration..."
                )
                
                # Create sample data for demonstration
                import pandas as pd
                import numpy as np
                
                sample_data = pd.DataFrame({
                    'A': np.random.randn(100),
                    'B': np.random.randint(1, 5, 100),
                    'C': np.random.choice(['X', 'Y', 'Z'], 100),
                    'D': pd.date_range('2023-01-01', periods=100)
                })
                
                self.agent.load_dataframe(sample_data, "sample")
                dataframe_name = "sample"
                
                await task_updater.update_status(
                    TaskState.working,
                    message="✅ Sample data created for analysis"
                )
            
            # Create or get SmartDataFrame
            if dataframe_name not in self.smart_dataframes:
                df = self.agent.dataframes.get(dataframe_name)
                if df is not None:
                    smart_df = SmartDataFrame(
                        df, 
                        name=dataframe_name,
                        description=f"Data loaded for query: {user_query[:100]}..."
                    )
                    self.smart_dataframes[dataframe_name] = smart_df
            
            # Process the query with SmartDataFrame
            await task_updater.update_status(
                TaskState.working,
                message="🤖 Processing query with AI-powered analysis..."
            )
            
            if dataframe_name in self.smart_dataframes:
                smart_df = self.smart_dataframes[dataframe_name]
                analysis_result = await smart_df.chat(user_query)
            else:
                # Fallback to regular PandasAgent
                analysis_result = await self.agent.chat(user_query, dataframe_name)
            
            # Generate visualizations if needed
            visualization_result = None
            if (analysis_result.get("visualization_suggestions") or 
                "plot" in user_query.lower() or "chart" in user_query.lower()):
                
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 Generating intelligent visualizations..."
                )
                
                visualization_result = await self._generate_visualizations(
                    user_query, dataframe_name, analysis_result
                )
            
            # Prepare comprehensive response
            await task_updater.update_status(
                TaskState.working,
                message="📝 Preparing comprehensive analysis report..."
            )
            
            comprehensive_result = await self._prepare_comprehensive_response(
                user_query,
                analysis_result,
                visualization_result,
                dataframe_name
            )
            
            # Cache the result
            self.cache_manager.set(
                cache_key,
                comprehensive_result,
                ttl_seconds=3600,  # 1 hour
                tags=["query_result", dataframe_name]
            )
            
            # Send the response
            await self._send_comprehensive_response(comprehensive_result, task_updater)
            
            # Complete the task
            await task_updater.update_status(
                TaskState.completed,
                message="🎉 Comprehensive analysis completed successfully!"
            )
            
        except Exception as e:
            error_msg = f"Error in pandas_agent execution: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            await task_updater.update_status(
                TaskState.failed,
                message=f"❌ {error_msg}"
            )
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        Cancel pandas_agent execution
        
        Args:
            context: A2A request context
            task_updater: A2A task updater
        """
        self.logger.info("Cancelling pandas_agent execution")
        await task_updater.update_status(
            TaskState.cancelled,
            message="🛑 Pandas agent execution cancelled"
        )
    
    def _extract_query_from_context(self, context: RequestContext) -> str:
        """
        Extract user query from A2A request context
        
        Args:
            context: A2A request context
            
        Returns:
            User query string
        """
        try:
            if hasattr(context, 'message') and hasattr(context.message, 'parts'):
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        return part.root.text
                    elif hasattr(part, 'text'):
                        return part.text
            
            # Fallback
            return "Analyze the data"
            
        except Exception as e:
            self.logger.error(f"Error extracting query: {e}")
            return "Analyze the data"
    
    async def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine data loading requirements
        
        Args:
            query: User query
            
        Returns:
            Analysis of query requirements
        """
        analysis = {
            "needs_data_loading": False,
            "data_source_type": None,
            "data_source_path": None,
            "query_intent": "general_analysis"
        }
        
        # Simple keyword-based detection (could be enhanced with LLM)
        query_lower = query.lower()
        
        # Check for file references
        if any(ext in query_lower for ext in ['.csv', '.xlsx', '.json', '.parquet']):
            analysis["needs_data_loading"] = True
            analysis["data_source_type"] = "file"
            
            # Try to extract file path
            import re
            file_pattern = r'([^\s]+\.(csv|xlsx|xls|json|parquet))'
            match = re.search(file_pattern, query, re.IGNORECASE)
            if match:
                analysis["data_source_path"] = match.group(1)
        
        # Check for SQL/database references
        elif any(word in query_lower for word in ['database', 'table', 'sql', 'select']):
            analysis["needs_data_loading"] = True
            analysis["data_source_type"] = "sql"
        
        # Determine query intent
        if any(word in query_lower for word in ['plot', 'chart', 'visualize', 'graph']):
            analysis["query_intent"] = "visualization"
        elif any(word in query_lower for word in ['correlation', 'relationship']):
            analysis["query_intent"] = "correlation"
        elif any(word in query_lower for word in ['summary', 'describe', 'overview']):
            analysis["query_intent"] = "summary"
        
        return analysis
    
    async def _handle_data_loading(self, 
                                 query_analysis: Dict[str, Any],
                                 task_updater: TaskUpdater) -> Optional[str]:
        """
        Handle data loading based on query analysis
        
        Args:
            query_analysis: Analysis of query requirements
            task_updater: Task updater for status updates
            
        Returns:
            Name of loaded dataframe or None if failed
        """
        try:
            data_source_type = query_analysis.get("data_source_type")
            data_source_path = query_analysis.get("data_source_path")
            
            if data_source_type == "file" and data_source_path:
                # Create file connector
                connector = FileConnector(
                    name="query_file",
                    file_path=data_source_path,
                    file_type="auto"
                )
                
                # Test connection
                if await connector.connect():
                    # Load data
                    df = await connector.read_data()
                    dataframe_name = f"loaded_{datetime.now().strftime('%H%M%S')}"
                    self.agent.load_dataframe(df, dataframe_name)
                    
                    await task_updater.update_status(
                        TaskState.working,
                        message=f"✅ Loaded {len(df)} rows from {data_source_path}"
                    )
                    
                    return dataframe_name
                else:
                    await task_updater.update_status(
                        TaskState.working,
                        message=f"❌ Failed to connect to {data_source_path}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            await task_updater.update_status(
                TaskState.working,
                message=f"❌ Error loading data: {str(e)}"
            )
            return None
    
    async def _generate_visualizations(self,
                                     query: str,
                                     dataframe_name: str,
                                     analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate visualizations using the visualization engine
        
        Args:
            query: Original query
            dataframe_name: Name of dataframe
            analysis_result: Analysis results
            
        Returns:
            Visualization results or None
        """
        try:
            if dataframe_name in self.agent.dataframes:
                df = self.agent.dataframes[dataframe_name]
                
                # Determine intent from analysis
                intent = "exploration"
                if analysis_result.get("intent_analysis"):
                    intent = analysis_result["intent_analysis"].get("primary_intent", "exploration")
                
                # Generate visualizations
                viz_result = self.viz_engine.generate_auto_visualizations(
                    df=df,
                    query=query,
                    intent=intent,
                    max_charts=3
                )
                
                return viz_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return None
    
    async def _prepare_comprehensive_response(self,
                                            query: str,
                                            analysis_result: Dict[str, Any],
                                            visualization_result: Optional[Dict[str, Any]],
                                            dataframe_name: str) -> Dict[str, Any]:
        """
        Prepare comprehensive response with all analysis components
        
        Args:
            query: Original query
            analysis_result: Main analysis results
            visualization_result: Visualization results
            dataframe_name: Name of analyzed dataframe
            
        Returns:
            Comprehensive response dictionary
        """
        
        # Get dataframe info
        df_info = None
        if dataframe_name in self.agent.dataframes:
            df = self.agent.dataframes[dataframe_name]
            df_analyzer = DataFrameInfo(df, dataframe_name)
            df_info = df_analyzer.get_basic_info()
        
        # Prepare response
        response = {
            "query": query,
            "dataframe_analyzed": dataframe_name,
            "timestamp": datetime.now().isoformat(),
            "analysis_result": analysis_result,
            "dataframe_info": df_info,
            "cache_stats": self.cache_manager.get_stats(),
            "agent_stats": {
                "smart_dataframes": len(self.smart_dataframes),
                "connectors": len(self.connectors),
                "total_queries": len(self.agent.query_history) if self.agent else 0
            }
        }
        
        # Add visualizations if available
        if visualization_result:
            response["visualizations"] = visualization_result
        
        # Add performance metrics
        if hasattr(analysis_result, 'get') and 'execution_time' in analysis_result:
            response["performance"] = {
                "total_execution_time": analysis_result.get('execution_time', 0),
                "cache_hit": False
            }
        
        return response
    
    async def _send_comprehensive_response(self,
                                         response: Dict[str, Any],
                                         task_updater: TaskUpdater):
        """
        Send comprehensive response as A2A artifacts
        
        Args:
            response: Comprehensive response
            task_updater: Task updater
        """
        
        # Main analysis artifact
        analysis_artifact = [TextPart(text=json.dumps(response, indent=2, default=str))]
        
        await task_updater.add_artifact(
            parts=analysis_artifact,
            name="comprehensive_analysis",
            metadata={
                "content_type": "application/json",
                "query": response["query"],
                "agent_type": "pandas_agent_enhanced",
                "features": ["llm_analysis", "smart_dataframe", "auto_visualization", "caching"]
            }
        )
        
        # Summary artifact for quick reading
        summary = self._generate_response_summary(response)
        summary_artifact = [TextPart(text=summary)]
        
        await task_updater.add_artifact(
            parts=summary_artifact,
            name="analysis_summary",
            metadata={
                "content_type": "text/plain",
                "description": "Human-readable summary of analysis results"
            }
        )
        
        # Visualization artifacts if available
        if "visualizations" in response and response["visualizations"].get("charts"):
            for i, chart in enumerate(response["visualizations"]["charts"]):
                if chart.get("image_base64"):
                    chart_artifact = [TextPart(text=json.dumps(chart, indent=2))]
                    
                    await task_updater.add_artifact(
                        parts=chart_artifact,
                        name=f"visualization_{i+1}",
                        metadata={
                            "content_type": "application/json",
                            "chart_type": chart.get("type", "unknown"),
                            "description": chart.get("description", "Generated visualization")
                        }
                    )
    
    async def _send_cached_response(self,
                                   cached_result: Dict[str, Any],
                                   task_updater: TaskUpdater):
        """Send cached response"""
        
        # Mark as cached
        cached_result["from_cache"] = True
        cached_result["cache_timestamp"] = datetime.now().isoformat()
        
        await self._send_comprehensive_response(cached_result, task_updater)
    
    def _generate_response_summary(self, response: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of response
        
        Args:
            response: Comprehensive response
            
        Returns:
            Summary text
        """
        summary = f"""
🤖 Pandas Agent Analysis Results
================================

Query: {response['query']}
Analyzed Dataset: {response['dataframe_analyzed']}
Timestamp: {response['timestamp']}

"""
        
        # Add dataframe info
        if response.get('dataframe_info'):
            df_info = response['dataframe_info']
            summary += f"""
📊 Dataset Information:
- Shape: {df_info['shape'][0]:,} rows × {df_info['shape'][1]} columns
- Data Types: {df_info['data_types']['counts']}
- Memory Usage: {df_info['memory_usage'] / 1024 / 1024:.2f} MB

"""
        
        # Add analysis insights
        if response.get('analysis_result'):
            analysis = response['analysis_result']
            
            if analysis.get('interpretation'):
                interpretation = analysis['interpretation']
                if interpretation.get('key_findings'):
                    summary += "🔍 Key Findings:\n"
                    for finding in interpretation['key_findings'][:3]:
                        summary += f"- {finding}\n"
                    summary += "\n"
        
        # Add visualization info
        if response.get('visualizations'):
            viz = response['visualizations']
            if viz.get('charts'):
                summary += f"📈 Generated {len(viz['charts'])} visualizations\n\n"
        
        # Add performance info
        if response.get('performance'):
            perf = response['performance']
            summary += f"⚡ Execution Time: {perf.get('total_execution_time', 0):.2f}s\n"
        
        if response.get('cache_stats'):
            cache = response['cache_stats']
            summary += f"💾 Cache: {cache.get('hit_rate', 0):.1%} hit rate\n"
        
        return summary


# A2A Server Configuration
def create_agent_card() -> AgentCard:
    """Create agent card for pandas_agent"""
    return AgentCard(
        name="pandas_agent_enhanced",
        description="Comprehensive AI-powered pandas data analysis agent with natural language interface, smart dataframes, auto-visualization, and multi-datasource connectivity",
        version="2.0.0",
        url="http://localhost:8210",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="data_analysis",
                name="data_analysis",
                description="Perform comprehensive data analysis on pandas dataframes",
                tags=["analysis", "pandas"]
            ),
            AgentSkill(
                id="data_visualization",
                name="data_visualization", 
                description="Generate automatic visualizations based on data characteristics",
                tags=["visualization", "charts"]
            ),
            AgentSkill(
                id="data_loading",
                name="data_loading",
                description="Load data from multiple sources (CSV, Excel, SQL, JSON)",
                tags=["loading", "data"]
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            cancellation=True
        )
    )

def create_pandas_agent_server(port: int = 8210) -> A2AStarletteApplication:
    """
    Create enhanced A2A Starlette application for pandas_agent
    
    Args:
        port: Server port (default: 8210)
        
    Returns:
        A2A Starlette application
    """
    task_store = InMemoryTaskStore()
    executor = PandasAgentExecutor()
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    
    return A2AStarletteApplication(
        agent_card=create_agent_card(),
        http_handler=request_handler,
    )


# Main entry point for running the server
if __name__ == "__main__":
    import uvicorn
    
    app = create_pandas_agent_server()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8210,
        log_level="info"
    )