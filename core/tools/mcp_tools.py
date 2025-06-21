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
        # Check if MCP servers are already running
        import aiohttp
        
        connections = tool_config.get("mcpServers", tool_config)
        working_connections = {}
        
        # Test connections first
        for server_name, server_config in connections.items():
            if "url" in server_config and server_config.get("transport") == "sse":
                try:
                    # Test connection to SSE endpoint
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            server_config["url"], 
                            timeout=aiohttp.ClientTimeout(total=3)
                        ) as response:
                            if response.status == 200:
                                working_connections[server_name] = server_config
                                logging.info(f"✅ MCP server {server_name} is running")
                            else:
                                logging.warning(f"⚠️ MCP server {server_name} returned status {response.status}")
                except Exception as e:
                    logging.warning(f"❌ MCP server '{server_name}' is not running: {e}")
        
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
    """Test which MCP servers are available"""
    import aiohttp
    
    mcp_servers = {
        "task_manager": {"url": "http://localhost:8001/sse", "transport": "sse"},
        "self_critic": {"url": "http://localhost:8002/sse", "transport": "sse"},
        "memory_kv": {"url": "http://localhost:8003/sse", "transport": "sse"},
        "result_ranker": {"url": "http://localhost:8004/sse", "transport": "sse"},
        "logger": {"url": "http://localhost:8005/sse", "transport": "sse"},
        "file_management": {"url": "http://localhost:8006/sse", "transport": "sse"},
        "data_science_tools": {"url": "http://localhost:8007/sse", "transport": "sse"}
    }
    
    availability = {}
    
    for server_name, server_config in mcp_servers.items():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    server_config["url"], 
                    timeout=aiohttp.ClientTimeout(total=1)
                ) as response:
                    availability[server_name] = response.status == 200
        except Exception:
            availability[server_name] = False
    
    return availability

def get_role_mcp_tools(role_name: str, available_servers: Dict[str, bool]) -> Tuple[List[str], Dict]:
    """Get appropriate MCP tools for a specific role"""
    base_tools = ["python_repl_ast"]  # All roles get Python tool
    mcp_configs = {}
    
    # Role to MCP tool mapping
    role_mcp_mapping = {
        "EDA_Specialist": ["data_science_tools"],
        "Visualization_Expert": ["data_science_tools"],
        "ML_Engineer": ["data_science_tools", "result_ranker"],
        "Data_Preprocessor": ["data_science_tools", "file_management"],
        "Statistical_Analyst": ["data_science_tools", "result_ranker"],
        "Report_Writer": ["logger", "file_management"]
    }
    
    if role_name in role_mcp_mapping:
        for server_name in role_mcp_mapping[role_name]:
            if available_servers.get(server_name, False):
                tool_name = f"mcp:supervisor_tools:{server_name}"
                base_tools.append(tool_name)
                mcp_configs[tool_name] = {
                    "config_name": "supervisor_tools",
                    "server_name": server_name,
                    "server_config": {
                        "url": f"http://localhost:{8001 + list(available_servers.keys()).index(server_name)}/sse",
                        "transport": "sse"
                    }
                }
    
    return base_tools, {"mcp_configs": mcp_configs}