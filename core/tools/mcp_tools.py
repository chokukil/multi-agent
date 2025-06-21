# File: core/tools/mcp_tools.py
# Location: ./core/tools/mcp_tools.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
from langchain_core.tools import Tool

# MCP imports - optional
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP adapters not available. Install langchain-mcp-adapters to use MCP tools.")
    
    # Dummy class for compatibility
    class MultiServerMCPClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get_tools(self):
            return []

from pydantic import BaseModel, Field, RootModel
from typing import Union

async def check_mcp_server_availability(server_configs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """
    MCP 서버의 현재 동작 상태를 비동기적으로 확인
    
    Args:
        server_configs: 서버 이름과 설정을 담은 딕셔너리
        
    Returns:
        서버 이름과 가용성 상태를 담은 딕셔너리
    """
    if not MCP_AVAILABLE:
        logging.warning("MCP not available, all servers marked as unavailable")
        return {name: False for name in server_configs.keys()}
    
    import aiohttp
    availability = {}
    
    async def check_single_server(server_name: str, server_config: Dict[str, Any]) -> Tuple[str, bool]:
        """단일 서버 상태 확인 - 개선된 타임아웃과 에러 처리"""
        try:
            if server_config.get("transport") == "sse" and "url" in server_config:
                # 더 긴 타임아웃 설정 (MCP 서버 시작 시간 고려)
                timeout = aiohttp.ClientTimeout(total=10, connect=5)
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        server_config["url"], 
                        timeout=timeout
                    ) as response:
                        is_available = response.status == 200
                        if is_available:
                            logging.info(f"✅ MCP server '{server_name}' is available at {server_config['url']}")
                        else:
                            logging.warning(f"⚠️ MCP server '{server_name}' returned status {response.status} at {server_config['url']}")
                        return server_name, is_available
            else:
                logging.warning(f"⚠️ MCP server '{server_name}' has unsupported transport or missing URL")
                return server_name, False
        except asyncio.TimeoutError:
            logging.warning(f"⏰ MCP server '{server_name}' timed out (may still be starting)")
            return server_name, False
        except aiohttp.ClientConnectorError as e:
            if "Connection refused" in str(e):
                logging.info(f"💤 MCP server '{server_name}' not running (connection refused)")
            else:
                logging.warning(f"🔌 MCP server '{server_name}' connection error: {e}")
            return server_name, False
        except Exception as e:
            logging.warning(f"❌ MCP server '{server_name}' check failed: {e}")
            return server_name, False
    
    # 모든 서버를 병렬로 확인
    tasks = [check_single_server(name, config) for name, config in server_configs.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Error checking MCP server: {result}")
        elif isinstance(result, tuple) and len(result) == 2:
            server_name, is_available = result
            availability[server_name] = is_available
    
    return availability

def create_mcp_tool_wrapper(mcp_tool) -> Tool:
    """Create a proper LangChain tool wrapper for MCP tools"""
    
    # Flexible input schema
    class FlexInput(RootModel[Union[str, Dict[str, Any]]]):
        """RootModel accepting str or dict as payload."""
        pass
    
    def sync_run(root: Union[str, Dict[str, Any]]):
        """Run remote MCP tool with flexible payload"""
        payload = root
        
        # Attempt JSON parse for string payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                pass  # leave as raw string if not JSON
        
        try:
            # Prefer .invoke (StructuredTool & Runnable)
            if hasattr(mcp_tool, "invoke"):
                try:
                    return mcp_tool.invoke(payload)
                except Exception as e_inv:
                    # StructuredTool blocks sync invoke; fallback to async
                    if "sync invocation" in str(e_inv) or "StructuredTool" in str(e_inv):
                        try:
                            return asyncio.run(mcp_tool.ainvoke(payload))
                        except RuntimeError:
                            # Already inside running loop
                            loop = asyncio.get_event_loop()
                            return loop.run_until_complete(mcp_tool.ainvoke(payload))
                    else:
                        raise
            
            # Next, if the object itself is callable
            if callable(mcp_tool):
                return mcp_tool(payload)
            
            # Finally, fall back to .run if available
            if hasattr(mcp_tool, "run"):
                return mcp_tool.run(payload)
            
            raise RuntimeError("Unsupported MCP tool interface")
            
        except Exception as e:
            logging.error(f"Error executing tool {getattr(mcp_tool, 'name', 'unknown')}: {e}")
            return f"❌ MCP tool error: {e}"
    
    tool_name = getattr(mcp_tool, "name", "unknown_tool")
    tool_description = getattr(mcp_tool, "description", "No description")
    
    langchain_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=sync_run,
        args_schema=FlexInput,
        handle_tool_error=True,
    )
    
    return langchain_tool

async def initialize_mcp_tools(tool_config: Dict) -> List[Tool]:
    """Initialize MCP tools from configuration with better error handling"""
    if not MCP_AVAILABLE:
        logging.warning("MCP not available, skipping tool initialization")
        return []
    
    if not tool_config:
        return []
    
    try:
        connections = tool_config.get("mcpServers", tool_config)
        
        # 서버 가용성 확인
        availability = await check_mcp_server_availability(connections)
        working_connections = {
            name: config for name, config in connections.items() 
            if availability.get(name, False)
        }
        
        if not working_connections:
            logging.warning("No working MCP servers found")
            return []
        
        # Initialize MCP client with only working connections
        client = MultiServerMCPClient(working_connections)
        
        try:
            raw_tools = await client.get_tools()
            
            # Wrap tools properly for LangChain
            tools = []
            for tool in raw_tools:
                try:
                    wrapped_tool = create_mcp_tool_wrapper(tool)
                    tools.append(wrapped_tool)
                except Exception as e:
                    logging.error(f"Failed to wrap tool {getattr(tool, 'name', 'unknown')}: {e}")
            
            logging.info(f"✅ Initialized {len(tools)} MCP tools from {len(working_connections)} servers")
            return tools
            
        except Exception as e:
            logging.error(f"Failed to get tools from MCP client: {e}")
            return []
        
    except Exception as e:
        logging.error(f"MCP initialization error: {e}")
        return []

async def test_mcp_server_availability() -> Dict[str, bool]:
    """Test which MCP servers are available - 모든 실제 MCP 서버 포함"""
    # mcp_config.py와 동일한 포트 매핑 사용
    mcp_servers = {
        # 실제 구현된 MCP 서버들 (mcp_config.py 포트와 일치)
        "file_management": {"url": "http://localhost:8006/sse", "transport": "sse"},
        "data_science_tools": {"url": "http://localhost:8007/sse", "transport": "sse"},
        "semiconductor_yield_analysis": {"url": "http://localhost:8008/sse", "transport": "sse"},
        "process_control_charts": {"url": "http://localhost:8009/sse", "transport": "sse"},
        "semiconductor_equipment_analysis": {"url": "http://localhost:8010/sse", "transport": "sse"},
        "defect_pattern_analysis": {"url": "http://localhost:8011/sse", "transport": "sse"},
        "process_optimization": {"url": "http://localhost:8012/sse", "transport": "sse"},
        "timeseries_analysis": {"url": "http://localhost:8013/sse", "transport": "sse"},
        "anomaly_detection": {"url": "http://localhost:8014/sse", "transport": "sse"},
        "advanced_ml_tools": {"url": "http://localhost:8016/sse", "transport": "sse"},
        "data_preprocessing_tools": {"url": "http://localhost:8017/sse", "transport": "sse"},
        "statistical_analysis_tools": {"url": "http://localhost:8018/sse", "transport": "sse"},
        "report_writing_tools": {"url": "http://localhost:8019/sse", "transport": "sse"},
        "semiconductor_process_tools": {"url": "http://localhost:8020/sse", "transport": "sse"}
    }
    
    return await check_mcp_server_availability(mcp_servers)

def get_role_mcp_tools(role_name: str, available_servers: Dict[str, bool]) -> Tuple[List[str], Dict]:
    """Get appropriate MCP tools for a specific role - 새로운 역할명 지원"""
    base_tools = ["python_repl_ast"]  # All roles get Python tool
    mcp_configs = {}
    
    # 새로운 역할명을 기존 매핑으로 변환 (호환성)
    role_name_mapping = {
        "Data_Validator": "Data_Preprocessor",
        "Preprocessing_Expert": "Data_Preprocessor", 
        "EDA_Analyst": "EDA_Specialist",
        "Visualization_Expert": "Visualization_Expert",
        "ML_Specialist": "ML_Engineer",
        "Statistical_Analyst": "Statistical_Analyst",
        "Report_Generator": "Report_Writer"
    }
    
    # 역할명 매핑 적용
    mapped_role = role_name_mapping.get(role_name, role_name)
    
    # Role to MCP tool mapping (확장된 서버 리스트)
    role_mcp_mapping = {
        "EDA_Specialist": ["data_science_tools", "statistical_analysis_tools"],
        "Visualization_Expert": ["data_science_tools", "statistical_analysis_tools"],
        "ML_Engineer": ["data_science_tools", "advanced_ml_tools", "statistical_analysis_tools"],
        "Data_Preprocessor": ["data_science_tools", "data_preprocessing_tools", "file_management"],
        "Statistical_Analyst": ["statistical_analysis_tools", "data_science_tools", "timeseries_analysis"],
        "Report_Writer": ["report_writing_tools", "file_management", "data_science_tools"]
    }
    
    if mapped_role in role_mcp_mapping:
        required_servers = role_mcp_mapping[mapped_role]
        available_count = 0
        
        for server_name in required_servers:
            if available_servers.get(server_name, False):
                tool_name = f"mcp:supervisor_tools:{server_name}"
                base_tools.append(tool_name)
                
                # 서버 포트 매핑 (mcp_config.py와 일치)
                port_mapping = {
                    "file_management": 8006, "data_science_tools": 8007,
                    "semiconductor_yield_analysis": 8008, "process_control_charts": 8009,
                    "semiconductor_equipment_analysis": 8010, "defect_pattern_analysis": 8011,
                    "process_optimization": 8012, "timeseries_analysis": 8013,
                    "anomaly_detection": 8014, "advanced_ml_tools": 8016,
                    "data_preprocessing_tools": 8017, "statistical_analysis_tools": 8018,
                    "report_writing_tools": 8019, "semiconductor_process_tools": 8020
                }
                
                port = port_mapping.get(server_name, 8000)
                
                mcp_configs[tool_name] = {
                    "config_name": "supervisor_tools",
                    "server_name": server_name,
                    "server_config": {
                        "url": f"http://localhost:{port}/sse",
                        "transport": "sse"
                    }
                }
                available_count += 1
                logging.info(f"✅ Added MCP tool '{server_name}' for {role_name}")
            else:
                logging.info(f"💤 MCP server '{server_name}' not available for {role_name}")
        
        logging.info(f"🔧 {role_name} configured with {available_count}/{len(required_servers)} MCP servers")
    else:
        logging.warning(f"⚠️ No MCP mapping found for role '{role_name}' (mapped: '{mapped_role}')")
    
    return base_tools, {"mcp_configs": mcp_configs}

def get_available_mcp_tools_info(config_name: str = None) -> Dict[str, Any]:
    """현재 사용 가능한 MCP 도구 정보를 반환합니다."""
    from core.utils.config import get_mcp_config
    
    if config_name:
        config = get_mcp_config(config_name)
        if not config:
            return {"available": False, "tools": [], "error": "Configuration not found"}
        
        # 비동기 함수를 동기적으로 실행
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        servers = config.get("mcpServers", {})
        availability = loop.run_until_complete(check_mcp_server_availability(servers))
        
        available_tools = []
        for server_name, is_available in availability.items():
            if is_available:
                available_tools.append({
                    "server_name": server_name,
                    "config": servers[server_name],
                    "status": "available"
                })
            else:
                available_tools.append({
                    "server_name": server_name,
                    "config": servers[server_name],
                    "status": "unavailable"
                })
        
        return {
            "available": any(availability.values()),
            "tools": available_tools,
            "total_servers": len(servers),
            "available_servers": sum(availability.values())
        }
    
    return {"available": False, "tools": [], "error": "No configuration specified"}