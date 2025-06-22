"""
MCP (Model-Context-Protocol) Tool Setup

This module contains functions and classes for initializing and wrapping MCP tools
to be compatible with the LangChain framework.
"""

import logging
import asyncio
import streamlit as st

# Conditional import for MCP packages
try:
    from fastmcp import Client as FastMCPClient
    from fastmcp.client.transports import SSETransport, PythonStdioTransport
    import aiohttp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Global MCP Client instance
mcp_client = None

# MCP Client for multiple servers
if MCP_AVAILABLE:
    class MultiServerMCPClient:
        def __init__(self, server_configs):
            self.server_configs = server_configs
            self.client = None
            
        async def __aenter__(self):
            try:
                # Use fastmcp's multi-server client capabilities
                self.client = FastMCPClient(self.server_configs)
                await self.client.__aenter__()
                logging.info(f"✅ Connected to {len(self.server_configs)} MCP servers using fastmcp")
                return self
            except Exception as e:
                logging.error(f"❌ Failed to connect to MCP servers with fastmcp: {e}")
                # Ensure graceful failure
                self.client = None
                return self
        
        async def __aexit__(self, *args):
            if self.client:
                await self.client.__aexit__(*args)
        
        async def get_tools(self):
            if not self.client:
                return []
            
            all_tools = []
            try:
                # fastmcp client returns a list of all tools from all servers
                mcp_tools = await self.client.list_tools()
                
                for tool in mcp_tools:
                    # The tool name is already prefixed by the client (e.g., "server_name_tool_name")
                    # We need to store the client instance for later calls
                    tool._client = self.client
                    all_tools.append(tool)
                    
                logging.info(f"✅ Loaded {len(all_tools)} tools from all connected servers")
            except Exception as e:
                logging.error(f"❌ Failed to get tools from fastmcp client: {e}")
            
            return all_tools

    def create_mcp_tool_wrapper(mcp_tool):
        """Wrap MCP tool for LangChain compatibility using fastmcp.Client"""
        from langchain_core.tools import BaseTool
        from pydantic.v1 import BaseModel as BaseModelV1, Field
        from typing import Any, Dict, Union

        class _FlexInput(BaseModelV1):
            root: Union[str, Dict[str, Any]] = Field(default="", description="Tool input")

        def sync_run(args: Union[str, Dict[str, Any]]) -> str:
            """Synchronous wrapper for MCP tool execution"""
            try:
                loop = asyncio.get_event_loop()
                
                async def async_run():
                    try:
                        if isinstance(args, str):
                            tool_args = {"input": args}
                        else:
                            tool_args = args
                        
                        client = getattr(mcp_tool, '_client', None)
                        if client:
                            # fastmcp client uses prefixed names
                            response = await client.call_tool(mcp_tool.name, tool_args)
                            
                            if response.isError:
                                return f"Error: {response.error}"
                            else:
                                # fastmcp returns content in a list
                                if response.content:
                                    return '\n'.join(str(item.text) for item in response.content if hasattr(item, 'text'))
                                else:
                                    return "Tool executed successfully with no content."
                        else:
                            return f"Error: No fastmcp client available for {mcp_tool.name}"
                            
                    except Exception as e:
                        logging.error(f"fastmcp tool execution error: {e}", exc_info=True)
                        return f"Error executing fastmcp tool: {e}"
                
                return loop.run_until_complete(async_run())
                
            except Exception as e:
                logging.error(f"Sync wrapper error for fastmcp: {e}")
                return f"Tool execution failed: {e}"

        # 🆕 향상된 도구 설명 생성
        def generate_enhanced_description(tool_name: str, original_desc: str) -> str:
            """도구 이름과 원래 설명을 기반으로 향상된 설명 생성"""
            
            # 도구 카테고리별 향상된 설명 매핑
            description_enhancements = {
                # 통계 분석 도구들
                "statistical_analysis_tools": "🔢 Advanced Statistical Analysis: Perform hypothesis testing, correlation analysis, ANOVA, regression, and distribution fitting. Use for statistical validation, significance testing, and relationship discovery.",
                "data_science_tools": "📊 Comprehensive Data Science Toolkit: Access full data science pipeline including advanced analytics, feature engineering, model building, and statistical computations.",
                
                # 데이터 전처리 도구들
                "data_preprocessing_tools": "🧹 Data Preprocessing Suite: Clean, transform, and prepare data. Handle missing values, outliers, normalization, encoding, and data quality issues.",
                "anomaly_detection": "🚨 Anomaly Detection Engine: Identify outliers, anomalies, and unusual patterns in data using statistical methods, isolation forests, and machine learning approaches.",
                
                # 머신러닝 도구들
                "advanced_ml_tools": "🤖 Advanced Machine Learning: Build sophisticated ML models, perform hyperparameter tuning, feature selection, model evaluation, and ensemble methods.",
                "ml_specialist": "🎯 ML Specialist Tools: End-to-end machine learning workflow including model training, validation, interpretation, and deployment preparation.",
                
                # 시각화 및 분석
                "timeseries_analysis": "📈 Time Series Analysis: Analyze temporal data, detect trends/seasonality, forecast future values, and identify change points in time series.",
                "visualization": "📊 Data Visualization: Create professional charts, plots, dashboards, and interactive visualizations to communicate insights effectively.",
                
                # 파일 및 보고서 관리
                "file_management": "📁 File Operations: Read, write, copy, move files and manage directories. Handle various file formats (CSV, JSON, Excel, etc.).",
                "report_writing_tools": "📝 Report Generation: Create professional reports, documentation, and presentations with automated formatting and template support."
            }
            
            # 도구 이름에서 카테고리 추출
            tool_base_name = tool_name.lower()
            for category, enhanced_desc in description_enhancements.items():
                if category in tool_base_name:
                    return enhanced_desc
            
            # 개별 도구명 기반 설명 매핑
            specific_tools = {
                "correlation_analysis": "🔗 Correlation Analysis: Calculate Pearson, Spearman, and Kendall correlations between variables",
                "hypothesis_testing": "🧪 Hypothesis Testing: Perform t-tests, chi-square tests, and other statistical hypothesis tests",
                "data_cleaning": "🧽 Data Cleaning: Handle missing values, duplicates, and data quality issues",
                "feature_engineering": "⚙️ Feature Engineering: Create and select informative features for machine learning",
                "model_evaluation": "📏 Model Evaluation: Assess model performance with comprehensive metrics and validation",
                "outlier_detection": "🎯 Outlier Detection: Identify statistical outliers and anomalies in datasets",
                "trend_analysis": "📊 Trend Analysis: Analyze trends, patterns, and temporal changes in data",
                "forecasting": "🔮 Forecasting: Predict future values using time series and ML models"
            }
            
            for tool_pattern, desc in specific_tools.items():
                if tool_pattern in tool_base_name:
                    return desc
            
            # 원래 설명이 유용하면 그대로 사용, 아니면 기본 설명 생성
            if original_desc and original_desc.strip() and original_desc != "No description":
                return f"🔧 {original_desc}"
            else:
                return f"🔧 MCP Tool: {tool_name.replace('_', ' ').title()} - Specialized data science operation"

        # 도구 이름과 설명 향상
        tool_name = getattr(mcp_tool, "name", "unknown_tool")
        original_description = getattr(mcp_tool, "description", "")
        enhanced_description = generate_enhanced_description(tool_name, original_description)
        
        # Create LangChain tool
        langchain_tool = BaseTool(
            name=tool_name,
            description=enhanced_description,
            func=sync_run,
            args_schema=_FlexInput,
            handle_tool_error=True,
        )
        
        return langchain_tool

    async def initialize_mcp_tools(tool_config):
        """Initialize MCP tools from configuration with better error handling"""
        if not MCP_AVAILABLE:
            logging.info("ℹ️ MCP not available, using basic tools only")
            return []
            
        if not tool_config:
            logging.info("ℹ️ No MCP tool configuration provided")
            return []
        
        try:
            connections = tool_config.get("mcpServers", tool_config)
            working_connections = {}
            
            logging.info(f"🔍 Testing {len(connections)} MCP server connections...")
            
            # Test connections first
            for server_name, server_config in connections.items():
                logging.info(f"🔌 Testing connection to MCP server: {server_name}")
                logging.debug(f"   Server config: {server_config}")
                
                if "url" in server_config and server_config.get("transport") == "sse":
                    try:
                        # Test connection to SSE endpoint
                        async with aiohttp.ClientSession() as session:
                            async with session.get(server_config["url"], timeout=aiohttp.ClientTimeout(total=3)) as response:
                                if response.status == 200:
                                    working_connections[server_name] = server_config
                                    logging.info(f"✅ MCP server {server_name} is running at {server_config['url']}")
                                else:
                                    logging.warning(f"⚠️ MCP server {server_name} returned status {response.status}")
                    except asyncio.TimeoutError:
                        logging.warning(f"⏰ MCP server '{server_name}' connection timeout at {server_config.get('url', 'unknown URL')}")
                    except Exception as e:
                        logging.warning(f"❌ MCP server '{server_name}' is not accessible: {e}")
                else:
                    logging.warning(f"⚠️ Invalid or missing configuration for MCP server '{server_name}': {server_config}")
            
            if not working_connections:
                logging.warning("⚠️ No working MCP servers found. Using basic tools only.")
                logging.info("💡 To enable MCP tools:")
                logging.info("   1. Check if MCP servers are running")
                logging.info("   2. Verify server URLs in configuration")
                logging.info("   3. Ensure network connectivity")
                return []
            
            logging.info(f"📡 Connecting to {len(working_connections)} working MCP servers...")
            
            # Initialize MCP client with only working connections
            client = MultiServerMCPClient(working_connections)
            
            try:
                raw_tools = await client.get_tools()
                logging.info(f"🔧 Retrieved {len(raw_tools)} raw tools from MCP servers")
                
                # Debug: List all retrieved tools
                for i, tool in enumerate(raw_tools):
                    tool_name = getattr(tool, 'name', f'unknown_tool_{i}')
                    tool_desc = getattr(tool, 'description', 'No description')
                    logging.debug(f"   Tool {i+1}: {tool_name} - {tool_desc[:50]}...")
                
                # Wrap tools properly for LangChain
                tools = []
                successful_wraps = 0
                failed_wraps = 0
                
                for tool in raw_tools:
                    try:
                        wrapped_tool = create_mcp_tool_wrapper(tool)
                        tools.append(wrapped_tool)
                        successful_wraps += 1
                        
                        tool_name = getattr(tool, 'name', 'unknown')
                        logging.debug(f"✅ Successfully wrapped MCP tool: {tool_name}")
                        logging.debug(f"   Enhanced description: {wrapped_tool.description[:100]}...")
                        
                    except Exception as e:
                        failed_wraps += 1
                        tool_name = getattr(tool, 'name', 'unknown')
                        logging.error(f"❌ Failed to wrap tool {tool_name}: {e}")
                        logging.debug(f"   Tool attributes: {dir(tool)}")
                
                # Update the global mcp_client instance
                global mcp_client
                mcp_client = client
                
                logging.info(f"✅ Successfully initialized {successful_wraps} MCP tools from {len(working_connections)} servers")
                if failed_wraps > 0:
                    logging.warning(f"⚠️ Failed to wrap {failed_wraps} tools")
                
                # Debug: List final tool set
                logging.info("🔧 Final tool set:")
                for i, tool in enumerate(tools):
                    logging.info(f"   {i+1}. {tool.name}: {tool.description[:80]}...")
                
                return tools
                
            except Exception as e:
                logging.error(f"❌ Failed to get tools from MCP client: {e}")
                logging.debug(f"   Client state: {client}")
                logging.debug(f"   Working connections: {working_connections}")
                return []
            
        except Exception as e:
            logging.error(f"❌ Critical MCP initialization error: {e}")
            import traceback
            logging.debug(f"   Full traceback: {traceback.format_exc()}")
            return []
else:
    async def initialize_mcp_tools(tool_config):
        logging.info("ℹ️ MCP not available, using basic tools only")
        return [] 