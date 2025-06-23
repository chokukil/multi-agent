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

from pydantic import BaseModel, Field, RootModel, create_model
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
    
    # --- Logging ---
    logging.info(f"🕵️  [MCP PROBE] Starting availability check for {len(server_configs)} MCP server(s)...")
    
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
            # --- Logging ---
            logging.warning(f"⏰ [MCP PROBE] Server '{server_name}' timed out. This is common if the server is still starting up (race condition).")
            return server_name, False
        except aiohttp.ClientConnectorError as e:
            if "Connection refused" in str(e):
                # --- Logging ---
                logging.warning(f"🔌 [MCP PROBE] Server '{server_name}' refused connection. It's likely not running or still initializing (race condition).")
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
    
    # --- Logging ---
    available_servers = [name for name, is_on in availability.items() if is_on]
    logging.info(f"✅ [MCP PROBE] Check complete. Found {len(available_servers)} available server(s): {available_servers}")
    
    return availability

def create_mcp_tool_wrapper(mcp_tool) -> Tool:
    """Create a proper LangChain tool wrapper for MCP tools, preserving the original schema."""
    
    # 💡 1. 원본 도구의 스키마를 그대로 사용
    args_schema = getattr(mcp_tool, "args_schema", None)
    tool_name = getattr(mcp_tool, "name", "unknown_tool")
    tool_description = getattr(mcp_tool, "description", "No description")

    # 💡 2. 스키마가 없으면 경고 후, 최소한의 폴백 스키마 적용
    if not args_schema or not issubclass(args_schema, BaseModel):
        logging.warning(f"⚠️ Tool '{tool_name}' is missing a valid Pydantic args_schema. Falling back to generic input.")
        class GenericInput(BaseModel):
            input: Union[str, Dict[str, Any]] = Field(..., description="The input for the tool. Can be a string or a JSON object.")
        args_schema = GenericInput

    def sync_run(tool_input: BaseModel):
        """Run remote MCP tool with a Pydantic model as input."""
        
        # Pydantic 모델을 딕셔너리로 변환하여 payload 생성
        payload = tool_input.model_dump()

        # 만약 스키마가 'input' 필드 하나만 가진다면, 내용물만 전달
        if len(payload.keys()) == 1 and 'input' in payload:
            payload = payload['input']

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
            logging.error(f"Error executing tool {tool_name}: {e}")
            return f"❌ MCP tool error: {e}"

    langchain_tool = Tool(
        name=tool_name,
        description=tool_description,
        func=sync_run,
        args_schema=args_schema, # 💡 원본 스키마 또는 폴백 스키마 사용
        handle_tool_error=True,
    )
    
    return langchain_tool

async def initialize_mcp_tools(tool_config: Dict) -> List[Tool]:
    """Initialize MCP tools from configuration with better error handling"""
    # --- Logging ---
    logging.info("🛠️  [MCP INIT] Starting MCP tool initialization process...")
    
    if not MCP_AVAILABLE:
        logging.warning("MCP not available, skipping tool initialization")
        return []
    
    if not tool_config:
        # --- Logging ---
        logging.warning("⚠️ [MCP INIT] No tool configuration provided. Skipping initialization.")
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
            # --- Logging ---
            logging.error("❌ [MCP INIT] CRITICAL: No working MCP servers found.")
            logging.error("    -> HYPOTHESIS: This is likely due to a race condition where the main app started before the MCP servers were ready.")
            logging.error("    -> To confirm, restart the system and check the logs for '[MCP PROBE]' messages.")
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
            
            # --- Logging ---
            logging.info(f"✅ [MCP INIT] Successfully initialized {len(tools)} MCP tools from {len(working_connections)} servers.")
            return tools
            
        except Exception as e:
            # --- Logging ---
            logging.error(f"❌ [MCP INIT] Failed to get tools from MCP client even though servers seemed available: {e}")
            return []
        
    except Exception as e:
        # --- Logging ---
        logging.error(f"❌ [MCP INIT] An unexpected error occurred during MCP tool initialization: {e}", exc_info=True)
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
        "EDA_Specialist": ["statistical_analysis_tools", "data_preprocessing_tools", "data_science_tools"],
        "Visualization_Expert": ["data_science_tools", "statistical_analysis_tools"],
        "ML_Engineer": ["advanced_ml_tools", "data_science_tools", "statistical_analysis_tools"],
        "Data_Preprocessor": ["data_preprocessing_tools", "data_science_tools", "file_management"],
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

def create_enhanced_agent_prompt(executor_name: str, tool_names: List[str]) -> str:
    """
    에이전트별로 균형잡힌 도구 선택 가이드라인을 제공하는 프롬프트 생성
    
    Args:
        executor_name: 에이전트 이름
        tool_names: 사용 가능한 도구 목록
        
    Returns:
        균형잡힌 도구 선택 프롬프트
    """
    
    # MCP 도구 분류
    mcp_tools = [tool for tool in tool_names if any(mcp_server in tool for mcp_server in [
        'statistical_analysis', 'data_preprocessing', 'data_science', 'file_management',
        'timeseries_analysis', 'anomaly_detection', 'ml_specialist', 'report_writing'
    ])]
    
    python_tools = [tool for tool in tool_names if 'python' in tool.lower()]
    
    # 에이전트별 권장 도구 매핑 (강제가 아닌 권장)
    agent_recommendations = {
        'EDA_Analyst': ['data_science_tools', 'statistical_analysis_tools', 'python_repl_ast'],
        'Statistical_Analyst': ['statistical_analysis_tools', 'data_science_tools', 'python_repl_ast'],
        'Data_Preprocessor': ['data_preprocessing_tools', 'anomaly_detection', 'python_repl_ast'],
        'Visualization_Expert': ['python_repl_ast', 'data_science_tools'],
        'ML_Engineer': ['ml_specialist', 'statistical_analysis_tools', 'python_repl_ast'],
        'Report_Writer': ['report_writing_tools', 'file_management', 'python_repl_ast'],
        'Time_Series_Analyst': ['timeseries_analysis', 'statistical_analysis_tools', 'python_repl_ast']
    }
    
    # 에이전트별 권장 도구
    recommended_tools = agent_recommendations.get(executor_name, tool_names)
    
    enhanced_prompt = f"""
🔧 **INTELLIGENT TOOL SELECTION GUIDELINES:**

Your available tools: {', '.join(tool_names)}

**🎯 RECOMMENDED TOOLS FOR YOUR ROLE ({executor_name}):**
{', '.join(recommended_tools)}

**📋 SMART TOOL SELECTION PRINCIPLES:**

**Choose the RIGHT tool for the task:**
- **Specialized MCP tools** are great for standard operations with built-in validation
- **Python tools** excel at custom logic, complex transformations, and unique visualizations
- **File management tools** for file operations and data I/O
- **Consider task complexity, customization needs, and available tool capabilities**

**🎯 TASK-BASED RECOMMENDATIONS:**

📊 **For Statistical Analysis:**
- Standard stats (mean, correlation, t-tests) → `statistical_analysis_tools` OR `python_repl_ast`
- Custom statistical methods → `python_repl_ast`
- Quick exploratory stats → Either tool works well

🧹 **For Data Preprocessing:**
- Standard cleaning operations → `data_preprocessing_tools` OR `python_repl_ast`
- Complex custom transformations → `python_repl_ast`
- Missing value handling → Either tool works well

🤖 **For Machine Learning:**
- Standard ML workflows → `ml_specialist` OR `python_repl_ast`
- Custom model architectures → `python_repl_ast`
- Model evaluation → Either tool works well

📈 **For Visualization:**
- Standard charts → `data_science_tools` OR `python_repl_ast`
- Custom interactive plots → `python_repl_ast`
- Quick data exploration → Either tool works well

📝 **For File Operations:**
- File management → `file_management` OR `python_repl_ast`
- Complex file processing → `python_repl_ast`

**⚡ DECISION FRAMEWORK:**
1. **Identify the task type and complexity**
2. **Consider if you need custom logic or standard operations**
3. **Choose the tool that best fits your specific needs**
4. **MCP tools provide structure, Python provides flexibility**

**💡 BEST PRACTICES:**
- Try MCP tools first for standard operations (they're often faster)
- Use Python when you need custom logic or the MCP tool doesn't fit
- Combine tools when beneficial (e.g., MCP for data prep, Python for custom viz)
- Don't hesitate to switch tools if one doesn't work as expected

**Remember: Choose the tool that makes the most sense for your specific task!**
    """
    
    return enhanced_prompt.strip()