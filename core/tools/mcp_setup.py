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

        # Create LangChain tool
        langchain_tool = BaseTool(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
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
                if "url" in server_config and server_config.get("transport") == "sse":
                    try:
                        # Test connection to SSE endpoint
                        async with aiohttp.ClientSession() as session:
                            async with session.get(server_config["url"], timeout=aiohttp.ClientTimeout(total=3)) as response:
                                if response.status == 200:
                                    working_connections[server_name] = server_config
                                    logging.info(f"✅ MCP server {server_name} is running")
                                else:
                                    logging.warning(f"⚠️ MCP server {server_name} returned status {response.status}")
                    except asyncio.TimeoutError:
                        logging.warning(f"⏰ MCP server '{server_name}' connection timeout")
                    except Exception as e:
                        logging.warning(f"❌ MCP server '{server_name}' is not accessible: {e}")
                else:
                    logging.warning(f"⚠️ Invalid configuration for MCP server '{server_name}'")
            
            if not working_connections:
                logging.warning("⚠️ No working MCP servers found. Using basic tools only.")
                return []
            
            logging.info(f"📡 Connecting to {len(working_connections)} working MCP servers...")
            
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
                        logging.debug(f"✅ Wrapped MCP tool: {tool.name}")
                    except Exception as e:
                        logging.error(f"❌ Failed to wrap tool {getattr(tool, 'name', 'unknown')}: {e}")
                
                # Update the global mcp_client instance
                global mcp_client
                mcp_client = client
                
                logging.info(f"✅ Successfully initialized {len(tools)} MCP tools from {len(working_connections)} servers")
                return tools
                
            except Exception as e:
                logging.error(f"❌ Failed to get tools from MCP client: {e}")
                return []
            
        except Exception as e:
            logging.error(f"❌ Critical MCP initialization error: {e}")
            return []
else:
    async def initialize_mcp_tools(tool_config):
        logging.info("ℹ️ MCP not available, using basic tools only")
        return [] 