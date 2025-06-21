# File: app.py
# Location: ./app.py

"""
🍒 Cherry AI - Data Science Multi-Agent System
"""

import os
import sys
import asyncio
import logging
import platform
from pathlib import Path
from datetime import datetime
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Apply nest_asyncio for Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Reload config to ensure env variables are loaded
from core.utils.config import reload_config
reload_config()

# MCP initialization
try:
    from mcp.client import ClientSession
    from mcp.client.sse import sse_client
    from mcp.types import CallToolRequest, ListToolsRequest
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available. Install mcp package for MCP tool support.")

# MCP Client for multiple servers
if MCP_AVAILABLE:
    class MultiServerMCPClient:
        def __init__(self, server_configs):
            self.server_configs = server_configs
            self.clients = {}
            self.sessions = {}
            
        async def __aenter__(self):
            for server_name, config in self.server_configs.items():
                if config.get("transport") == "sse" and "url" in config:
                    try:
                        # Create SSE client and session
                        client = sse_client(config["url"])
                        session = await client.__aenter__()
                        
                        self.clients[server_name] = client
                        self.sessions[server_name] = session
                        logging.info(f"✅ Connected to MCP server: {server_name}")
                    except Exception as e:
                        logging.error(f"❌ Failed to connect to {server_name}: {e}")
            return self
        
        async def __aexit__(self, *args):
            for server_name, client in self.clients.items():
                try:
                    await client.__aexit__(*args)
                except Exception as e:
                    logging.error(f"Error disconnecting from {server_name}: {e}")
        
        async def get_tools(self):
            all_tools = []
            for server_name, session in self.sessions.items():
                try:
                    response = await session.call(ListToolsRequest())
                    server_tools = response.tools
                    
                    # Add server context to tool names
                    for tool in server_tools:
                        tool.name = f"{server_name}_{tool.name}"
                        tool._server_name = server_name
                        tool._session = session
                    
                    all_tools.extend(server_tools)
                    logging.info(f"✅ Loaded {len(server_tools)} tools from {server_name}")
                except Exception as e:
                    logging.error(f"❌ Failed to get tools from {server_name}: {e}")
            
            return all_tools

    def create_mcp_tool_wrapper(mcp_tool):
        """Wrap MCP tool for LangChain compatibility"""
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field
        from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1
        from typing import Any, Dict, Type, Union
        
        # Use the appropriate BaseModel version
        try:
            from pydantic.v1 import BaseModel as PydanticModel, Field as PydanticField
        except ImportError:
            from pydantic import BaseModel as PydanticModel, Field as PydanticField
        
        class _FlexInput(PydanticModel):
            root: Union[str, Dict[str, Any]] = PydanticField(default="", description="Tool input")

        def sync_run(args: Union[str, Dict[str, Any]]) -> str:
            """Synchronous wrapper for MCP tool execution"""
            try:
                import asyncio
                
                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                async def async_run():
                    try:
                        # Prepare arguments
                        if isinstance(args, str):
                            tool_args = {"input": args}
                        elif isinstance(args, dict):
                            tool_args = args
                        else:
                            tool_args = {"input": str(args)}
                        
                        # Call MCP tool
                        session = getattr(mcp_tool, '_session', None)
                        if session:
                            request = CallToolRequest(
                                name=mcp_tool.name.split('_', 1)[-1],  # Remove server prefix
                                arguments=tool_args
                            )
                            response = await session.call(request)
                            
                            if response.isError:
                                return f"Error: {response.error}"
                            else:
                                # Handle different response types
                                if hasattr(response, 'content') and response.content:
                                    if isinstance(response.content, list):
                                        return '\n'.join(str(item) for item in response.content)
                                    else:
                                        return str(response.content)
                                else:
                                    return str(response)
                        else:
                            return f"Error: No session available for {mcp_tool.name}"
                            
                    except Exception as e:
                        logging.error(f"MCP tool execution error: {e}")
                        return f"Error executing MCP tool: {e}"
                
                return loop.run_until_complete(async_run())
                
            except Exception as e:
                logging.error(f"Sync wrapper error: {e}")
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
            logging.warning("MCP not available, skipping tool initialization")
            return []
            
        if not tool_config:
            return []
        
        try:
            # Check if MCP servers are already running
            import aiohttp
            import asyncio
            
            connections = tool_config.get("mcpServers", tool_config)
            working_connections = {}
            
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
                
                st.session_state.mcp_client = client
                
                logging.info(f"✅ Initialized {len(tools)} MCP tools from {len(working_connections)} servers")
                return tools
                
            except Exception as e:
                logging.error(f"Failed to get tools from MCP client: {e}")
                return []
            
        except Exception as e:
            logging.error(f"MCP initialization error: {e}")
            return []
else:
    async def initialize_mcp_tools(tool_config):
        logging.warning("MCP not available")
        return []

# Initialize Langfuse globally (following multi_agent_supervisor.py pattern)
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler  # 2.60.8에서는 callback 모듈에서 import
    LANGFUSE_AVAILABLE = True
    
    # Initialize global langfuse object after environment variables are loaded
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        logging.info("✅ Global Langfuse initialized successfully")
    else:
        langfuse = None
        logging.warning("⚠️ Langfuse environment variables not found")
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse = None
    logging.warning("Langfuse not available. Install langfuse for advanced tracing.")
except Exception as e:
    LANGFUSE_AVAILABLE = False
    langfuse = None
    logging.warning(f"⚠️ Langfuse initialization failed: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import core modules
from core import (
    # Data Management
    UnifiedDataManager, data_manager,
    
    # Debug System
    DebugManager, debug_manager,
    
    # Data Lineage
    data_lineage_tracker,
    
    # LLM Factory
    create_llm_instance,
    
    # Tools
    create_enhanced_python_tool,
    
    # Plan-Execute Components
    PlanExecuteState,
    planner_node,
    router_node,
    route_to_executor,
    TASK_EXECUTOR_MAPPING,
    create_executor_node,
    replanner_node,
    should_continue,
    final_responder_node,
    
    # Utilities
    log_event
)

# Import UI modules
from ui import (
    render_data_upload_section,
    render_executor_creation_form,
    render_saved_systems,
    render_quick_templates,
    render_system_settings,
    render_mcp_config_section,
    render_template_management_section,
    save_multi_agent_config,
    render_chat_interface,
    render_system_status,
    visualize_plan_execute_structure,
    render_bottom_tabs,
    # New real-time and artifact components
    render_real_time_dashboard,
    render_artifact_interface,
    apply_dashboard_styles,
    apply_artifact_styles
)

# Page config
st.set_page_config(
    page_title="Cherry AI - Data Science Multi-Agent System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍒"
)

# Initialize session state
def initialize_session_state():
    """초기 세션 상태 설정"""
    if "executors" not in st.session_state:
        st.session_state.executors = {}
    if "plan_execute_graph" not in st.session_state:
        st.session_state.plan_execute_graph = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        # .env 파일의 EMP_NO를 사용하여 user_id 설정
        st.session_state.user_id = os.getenv("EMP_NO", "default-user")
    if "event_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state.event_loop = loop
        asyncio.set_event_loop(loop)
    if "graph_initialized" not in st.session_state:
        st.session_state.graph_initialized = False
    if "current_system_name" not in st.session_state:
        st.session_state.current_system_name = ""
    if "timeout_seconds" not in st.session_state:
        st.session_state.timeout_seconds = 180
    if "recursion_limit" not in st.session_state:
        st.session_state.recursion_limit = 30
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = []
    
    # Real-time dashboard states
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = []
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}
    
    # Artifact management states  
    if "selected_artifact_id" not in st.session_state:
        st.session_state.selected_artifact_id = None
    if "execution_result" not in st.session_state:
        st.session_state.execution_result = None
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []

# Initialize
initialize_session_state()

# Custom CSS and styles
st.markdown("""
<style>
    .executor-card {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        background-color: #e3f2fd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .streaming-content {
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #2196F3;
        background-color: #f0f7ff;
    }
    .plan-step {
        padding: 8px;
        margin: 4px 0;
        background-color: #f5f5f5;
        border-radius: 4px;
    }
    .plan-step.completed {
        background-color: #e8f5e9;
        border-left: 3px solid #4CAF50;
    }
    .plan-step.active {
        background-color: #fff3e0;
        border-left: 3px solid #FF9800;
    }
    .tool-activity {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 10px;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Apply additional styles
apply_dashboard_styles()
apply_artifact_styles()

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Configuration")
    
    # Pre-configured Systems
    render_quick_templates()
    
    st.markdown("---")
    
    # CSV 파일 업로드
    render_data_upload_section()
    
    st.markdown("---")
    
    # System Management
    st.markdown("### 🧹 System Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("✅ History cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['event_loop']:
                    del st.session_state[key]
            initialize_session_state()
            st.success("✅ System reset!")
            st.rerun()
    
    # MCP Configuration Section
    render_mcp_config_section()
    
    st.markdown("---")

# Main area
st.title("🍒 Cherry AI - Data Science Multi-Agent System")
st.markdown("### Plan-Execute Pattern with SSOT, Data Lineage Tracking & MCP Tools")

# System status
render_system_status()

# Visualize system structure
st.markdown("### 🏗️ System Architecture")
viz_result = visualize_plan_execute_structure()
if viz_result is not None:
    st.plotly_chart(viz_result, use_container_width=True)

# System info
if not st.session_state.executors:
    st.info("""
    👋 **Welcome to the Enhanced Plan-Execute Multi-Agent System!**
    
    ### 🚀 New Architecture:
    - **Plan**: Analyze request and create execution plan
    - **Execute**: Route tasks to specialized executors
    - **Re-plan**: Evaluate progress and adapt
    - **Finalize**: Generate comprehensive response with data validation
    
    ### 📚 Quick Start:
    1. Click "🔬 Data Science Team" for pre-configured executors
    2. Upload CSV data in the sidebar
    3. Click "Create Plan-Execute System" to initialize
    4. Start chatting!
    
    ### 🎯 Key Features:
    - **Plan-Execute Pattern**: Structured, efficient task execution
    - **SSOT**: All executors access identical data
    - **Data Lineage**: Track all data transformations
    - **MCP Tools**: Model Context Protocol tool integration
    - **Real-time Streaming**: Live tool execution monitoring
    - **Hallucination Prevention**: Validate data usage
    """)

# Visualize executors
if st.session_state.executors:
    st.markdown("### 📋 Registered Executors")
    
    cols = st.columns(min(len(st.session_state.executors), 3))
    for i, (name, config) in enumerate(st.session_state.executors.items()):
        with cols[i % 3]:
            # 기본 도구 수집
            tools = config.get("tools", [])
            
            # MCP 도구 정보 수집
            mcp_config = config.get("mcp_config", {})
            mcp_tools = []
            
            if mcp_config:
                # mcp_configs에서 MCP 도구 추출 (multi_agent_supervisor.py 패턴)
                mcp_configs = mcp_config.get("mcp_configs", {})
                for tool_name, tool_config in mcp_configs.items():
                    server_name = tool_config.get("server_name", "unknown")
                    mcp_tools.append(server_name)
                
                # selected_tools에서도 확인 (호환성)
                if mcp_config.get("selected_tools"):
                    mcp_tools.extend(mcp_config["selected_tools"])
                
                # tools 리스트에서 mcp: 로 시작하는 도구들도 확인
                for tool in tools:
                    if tool.startswith("mcp:"):
                        # mcp:supervisor_tools:data_science_tools 형태에서 마지막 부분 추출
                        parts = tool.split(":")
                        if len(parts) >= 3:
                            mcp_tools.append(parts[-1])
            
            # 도구 목록 생성
            all_tools = []
            
            # 기본 도구 정리 (python_repl_ast -> Python)
            for tool in tools:
                if tool == "python_repl_ast":
                    all_tools.append("🐍 Python")
                elif tool.startswith("mcp:"):
                    # MCP 도구는 별도 처리하므로 제외
                    continue
                else:
                    all_tools.append(tool)
            
            # MCP 도구 추가 (개선된 매핑)
            for mcp_tool in set(mcp_tools):  # 중복 제거
                if mcp_tool == "data_science_tools":
                    all_tools.append("📊 Data Analysis")
                elif mcp_tool == "file_management":
                    all_tools.append("📁 File Manager")
                elif mcp_tool == "statistical_analysis_tools":
                    all_tools.append("📈 Statistical Analysis")
                elif mcp_tool == "data_preprocessing_tools":
                    all_tools.append("🔧 Data Preprocessing")
                elif mcp_tool == "advanced_ml_tools":
                    all_tools.append("🤖 Advanced ML")
                elif mcp_tool == "anomaly_detection":
                    all_tools.append("🚨 Anomaly Detection")
                elif mcp_tool == "timeseries_analysis":
                    all_tools.append("📅 Time Series")
                elif mcp_tool == "report_writing_tools":
                    all_tools.append("📄 Report Writing")
                else:
                    all_tools.append(f"🔧 {mcp_tool}")
            
            tools_str = ", ".join(all_tools) if all_tools else "No tools"
            
            # Get prompt preview
            prompt = config.get("prompt", "No prompt defined")
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            # 상태 정보 생성 (개선된 MCP 도구 카운팅)
            status_info = ""
            if mcp_tools:
                unique_mcp_tools = len(set(mcp_tools))
                status_info = f"<p><small>💡 {unique_mcp_tools}개 MCP 도구 활성화</small></p>"
            else:
                # MCP 도구가 없는 경우에도 상태 표시
                status_info = "<p><small>🔧 기본 도구만 사용</small></p>"
            
            # 에이전트 카드 표시
            st.markdown(f"""
            <div class='executor-card'>
                <h4>🤖 {name}</h4>
                <p><b>Role:</b> {prompt_preview}</p>
                <p><b>Tools:</b> {tools_str}</p>
                {status_info}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Create system button
if st.session_state.executors and not st.session_state.graph_initialized:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Plan-Execute System", type="primary", use_container_width=True):
            with st.spinner("🔧 Building Plan-Execute system with MCP tools..."):
                # Build the graph
                async def build_plan_execute_system():
                    """Build the Plan-Execute graph with MCP tools integration"""
                    
                    logging.info("Building Plan-Execute system with MCP tools")
                    
                    # Create graph
                    workflow = StateGraph(PlanExecuteState)
                    
                    # Add nodes
                    workflow.add_node("planner", planner_node)
                    workflow.add_node("router", router_node)
                    workflow.add_node("replanner", replanner_node)
                    workflow.add_node("final_responder", final_responder_node)
                    
                    # Add executor nodes with MCP tools
                    llm = create_llm_instance(
                        temperature=0.1,
                        session_id=st.session_state.get('thread_id', 'default-session'),
                        user_id=st.session_state.get('user_id', 'default-user')
                    )
                    
                    for executor_name, executor_config in st.session_state.executors.items():
                        # Create tools
                        tools = []
                        
                        # Add Python tool if configured
                        if "python_repl_ast" in executor_config.get("tools", []):
                            tools.append(create_enhanced_python_tool())
                        
                        # Add MCP tools if configured (multi_agent_supervisor.py 패턴 적용)
                        mcp_config = executor_config.get("mcp_config", {})
                        if mcp_config and mcp_config.get("mcp_configs"):
                            try:
                                # multi_agent_supervisor.py 방식으로 MCP 도구 초기화
                                # mcp_configs에서 서버 설정을 추출하여 mcpServers 형태로 변환
                                mcp_server_configs = {}
                                
                                for tool_name, tool_config in mcp_config["mcp_configs"].items():
                                    server_name = tool_config["server_name"] 
                                    server_config = tool_config["server_config"]
                                    mcp_server_configs[server_name] = server_config
                                
                                # initialize_mcp_tools에 올바른 형태로 전달
                                mcp_tool_config = {"mcpServers": mcp_server_configs}
                                mcp_tools = await initialize_mcp_tools(mcp_tool_config)
                                tools.extend(mcp_tools)
                                
                                logging.info(f"✅ Added {len(mcp_tools)} MCP tools to {executor_name}")
                                
                                # 서버별 상세 정보 로깅
                                server_names = list(mcp_server_configs.keys())
                                logging.info(f"   MCP servers for {executor_name}: {server_names}")
                                
                            except Exception as e:
                                logging.error(f"❌ Failed to initialize MCP tools for {executor_name}: {e}")
                                logging.error(f"   mcp_config: {mcp_config}")
                        else:
                            logging.info(f"💤 No MCP tools configured for {executor_name}")
                            # mcp_config 구조 확인을 위한 디버깅 로그
                            if mcp_config:
                                logging.info(f"   mcp_config keys: {list(mcp_config.keys())}")
                                if "mcp_configs" in mcp_config:
                                    logging.info(f"   mcp_configs: {list(mcp_config['mcp_configs'].keys())}")
                                else:
                                    logging.info(f"   No 'mcp_configs' key found in mcp_config")
                        
                        # Create agent with all tools
                        agent = create_react_agent(
                            model=llm,
                            tools=tools,
                            prompt=executor_config["prompt"]
                        )
                        
                        # Add node
                        workflow.add_node(
                            executor_name,
                            create_executor_node(agent, executor_name)
                        )
                    
                    # Add edges
                    workflow.add_edge(START, "planner")
                    workflow.add_edge("planner", "router")
                    
                    # Router to executors
                    executor_mapping = {name: name for name in st.session_state.executors}
                    
                    def route_function(state):
                        next_action = state.get("next_action", "")
                        if next_action in executor_mapping:
                            return next_action
                        # Try to map from task type
                        if state.get("plan") and state.get("current_step", 0) < len(state["plan"]):
                            task_type = state["plan"][state["current_step"]].get("type", "")
                            return TASK_EXECUTOR_MAPPING.get(task_type, list(executor_mapping.keys())[0])
                        return list(executor_mapping.keys())[0]
                    
                    workflow.add_conditional_edges(
                        "router",
                        route_function,
                        executor_mapping
                    )
                    
                    # Executors to replanner
                    for executor_name in st.session_state.executors:
                        workflow.add_edge(executor_name, "replanner")
                    
                    # Replanner routing
                    workflow.add_conditional_edges(
                        "replanner",
                        should_continue,
                        {
                            "continue": "router",
                            "finalize": "final_responder"
                        }
                    )
                    
                    # Final responder to END
                    workflow.add_edge("final_responder", END)
                    
                    # Compile
                    checkpointer = MemorySaver()
                    graph = workflow.compile(checkpointer=checkpointer)
                    
                    return graph
                
                try:
                    st.session_state.plan_execute_graph = st.session_state.event_loop.run_until_complete(
                        build_plan_execute_system()
                    )
                    st.session_state.graph_initialized = True
                    st.success("✅ Plan-Execute System with MCP tools created successfully!")
                    log_event("system_created", {
                        "executors": list(st.session_state.executors.keys()),
                        "timestamp": datetime.now().isoformat(),
                        "mcp_enabled": any(ex.get("mcp_config") for ex in st.session_state.executors.values())
                    })
                    logging.info("Plan-Execute system with MCP tools created")
                except Exception as e:
                    st.error(f"❌ Failed to create system: {e}")
                    logging.error(f"System creation failed: {e}")

# Save system
if st.session_state.graph_initialized and st.session_state.executors:
    st.markdown("### 💾 Save System")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        system_name = st.text_input(
            "System Name",
            value=st.session_state.current_system_name,
            placeholder="Enter a name for this system"
        )
    
    with col2:
        st.write("")  # Spacer
        if st.button("💾 Save", type="secondary", use_container_width=True):
            if system_name:
                config = {
                    "executors": st.session_state.executors,
                    "description": f"Plan-Execute system with {len(st.session_state.executors)} executors"
                }
                
                try:
                    save_multi_agent_config(system_name, config)
                    st.session_state.current_system_name = system_name
                    st.success(f"✅ Saved '{system_name}' successfully!")
                    log_event("system_saved", {
                        "name": system_name,
                        "executors_count": len(st.session_state.executors)
                    })
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
            else:
                st.error("Please enter a name")

st.markdown("---")

# Main Interface Tabs
if st.session_state.graph_initialized:
    # Create main tabs for enhanced interface
    tab1, tab2, tab3 = st.tabs(["💬 Chat & Analytics", "🚀 Real-time Dashboard", "🎨 Artifact Canvas"])
    
    with tab1:
        # Traditional chat interface with enhanced analytics
        render_chat_interface()
    
    with tab2:
        # Real-time process visualization dashboard
        render_real_time_dashboard()
    
    with tab3:
        # Canvas/Artifact style interface
        render_artifact_interface()

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.executors:
            st.info("⚠️ **System not initialized**\n\nClick 'Create Plan-Execute System' above to start!")
        else:
            st.info("⚠️ **No executors configured**\n\nUse the sidebar to add executors or load a template!")

st.markdown("---")

# Bottom tabs
render_bottom_tabs()

# Footer
st.markdown("---")
st.caption("🍒Cherry AI - Data Science Multi-Agent System with MCP Integration")

if __name__ == "__main__":
    logging.info("Application started")