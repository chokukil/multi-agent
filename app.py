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

# MCP initialization - Refactored to core.tools.mcp_setup
from core.tools.mcp_setup import initialize_mcp_tools

# We check this early to provide immediate feedback in the UI if it's unavailable.
try:
    from core.tools.mcp_setup import mcp_client, MCP_AVAILABLE
    if not MCP_AVAILABLE:
        logging.warning("⚠️ MCP libraries not found. MCP features will be disabled. Please run install_mcp_dependencies.bat.")
except (ImportError, AttributeError) as e:
    mcp_client = None
    MCP_AVAILABLE = False
    logging.warning(f"⚠️ Failed to import or access MCP client: {e}. MCP tools will be disabled. Please run install_mcp_dependencies.bat.")

# Initialize Langfuse globally (following multi_agent_supervisor.py pattern)
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
    
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
    # Artifact components only
    render_artifact_interface,
    apply_dashboard_styles,
    apply_artifact_styles
)

# Page config
st.set_page_config(
    page_title="Cherry AI - Data Science Multi-Agent System",
    layout="wide",
    initial_sidebar_state="collapsed" if st.session_state.get("left_sidebar_collapsed", False) else "expanded",
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
        
        try:
            from core.artifact_system import artifact_manager
            artifact_manager.clear_all_artifacts()
            logging.info(f"🧹 Cleared artifacts for new session: {st.session_state.thread_id}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to clear artifacts: {e}")
    
    if "user_id" not in st.session_state:
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
    
    if "selected_artifact_id" not in st.session_state:
        st.session_state.selected_artifact_id = None
    if "execution_result" not in st.session_state:
        st.session_state.execution_result = None
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    if "show_artifact_sidebar" not in st.session_state:
        st.session_state.show_artifact_sidebar = False
    if "left_sidebar_collapsed" not in st.session_state:
        st.session_state.left_sidebar_collapsed = False

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

apply_artifact_styles()

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Configuration")
    
    render_quick_templates()
    
    st.markdown("---")
    
    st.markdown("### 🧹 System Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.history = []
            try:
                from core.artifact_system import artifact_manager
                artifact_manager.clear_all_artifacts()
                logging.info("🧹 Cleared artifacts with history")
            except Exception as e:
                logging.warning(f"⚠️ Failed to clear artifacts: {e}")
            st.success("✅ History and artifacts cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['event_loop']:
                    del st.session_state[key]
            initialize_session_state()
            st.success("✅ System reset!")
            st.rerun()
    
    render_mcp_config_section()
    
    st.markdown("---")

# Main area
st.title("🍒 Cherry AI - Data Science Multi-Agent System")
st.markdown("### Plan-Execute Pattern with SSOT, Data Lineage Tracking & MCP Tools")

render_system_status()

if not MCP_AVAILABLE:
    st.warning(
        "**MCP Subsystem Not Available:** The MCP client could not be loaded. "
        "Any agent or tool relying on MCP will be disabled or may not function correctly. "
        "Please check the console logs for import errors and ensure dependencies from `install_mcp_dependencies.bat` are installed."
    )

st.markdown("### 🏗️ System Architecture")
viz_result = visualize_plan_execute_structure()
if viz_result is not None:
    st.plotly_chart(viz_result, use_container_width=True)

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
    """)

if st.session_state.executors:
    st.markdown("### 📋 Registered Executors")
    
    cols = st.columns(min(len(st.session_state.executors), 3))
    for i, (name, config) in enumerate(st.session_state.executors.items()):
        with cols[i % 3]:
            tools = config.get("tools", [])
            mcp_config = config.get("mcp_config", {})
            mcp_tools_list = []
            
            if mcp_config:
                mcp_configs = mcp_config.get("mcp_configs", {})
                for tool_name, tool_config in mcp_configs.items():
                    server_name = tool_config.get("server_name", "unknown")
                    mcp_tools_list.append(server_name)
                
                if mcp_config.get("selected_tools"):
                    mcp_tools_list.extend(mcp_config["selected_tools"])
                
                for tool in tools:
                    if tool.startswith("mcp:"):
                        parts = tool.split(":")
                        if len(parts) >= 3:
                            mcp_tools_list.append(parts[-1])
            
            all_tools = []
            
            for tool in tools:
                if tool == "python_repl_ast":
                    all_tools.append("🐍 Python")
                elif not tool.startswith("mcp:"):
                    all_tools.append(tool)
            
            for mcp_tool in set(mcp_tools_list):
                if mcp_tool == "data_science_tools":
                    all_tools.append("📊 Data Analysis")
                elif mcp_tool == "file_management":
                    all_tools.append("📁 File Manager")
                elif mcp_tool == "statistical_analysis_tools":
                    all_tools.append("📈 Statistical Analysis")
                else:
                    all_tools.append(f"🔧 {mcp_tool}")
            
            tools_str = ", ".join(all_tools) if all_tools else "No tools"
            prompt = config.get("prompt", "No prompt defined")
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            status_info = f"<p><small>💡 {len(set(mcp_tools_list))} MCP tools enabled</small></p>" if mcp_tools_list else "<p><small>🔧 Basic tools only</small></p>"
            
            st.markdown(f"""
            <div class='executor-card'>
                <h4>🤖 {name}</h4>
                <p><b>Role:</b> {prompt_preview}</p>
                <p><b>Tools:</b> {tools_str}</p>
                {status_info}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

if st.session_state.executors and not st.session_state.graph_initialized:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Plan-Execute System", type="primary", use_container_width=True):
            st.session_state.left_sidebar_collapsed = True
            
            with st.spinner("🔧 Building Plan-Execute system with MCP tools..."):
                async def build_plan_execute_system():
                    logging.info("Building Plan-Execute system with MCP tools")
                    
                    workflow = StateGraph(PlanExecuteState)
                    
                    # Add nodes
                    workflow.add_node("planner", planner_node)
                    workflow.add_node("router", router_node)
                    workflow.add_node("replanner", replanner_node)
                    workflow.add_node("final_responder", final_responder_node)
                    
                    llm = create_llm_instance(
                        temperature=0.1,
                        session_id=st.session_state.get('thread_id', 'default-session'),
                        user_id=st.session_state.get('user_id', 'default-user')
                    )
                    
                    for executor_name, executor_config in st.session_state.executors.items():
                        tools = []
                        if "python_repl_ast" in executor_config.get("tools", []):
                            tools.append(create_enhanced_python_tool())
                        
                        mcp_config = executor_config.get("mcp_config", {})
                        if mcp_config and mcp_config.get("mcp_configs"):
                            try:
                                mcp_server_configs = {}
                                for tool_name, tool_config in mcp_config["mcp_configs"].items():
                                    server_name = tool_config["server_name"] 
                                    server_config = tool_config["server_config"]
                                    mcp_server_configs[server_name] = server_config
                                
                                mcp_tool_config = {"mcpServers": mcp_server_configs}
                                mcp_tools = await initialize_mcp_tools(mcp_tool_config)
                                tools.extend(mcp_tools)
                                logging.info(f"✅ Added {len(mcp_tools)} MCP tools to {executor_name}")
                                
                            except Exception as e:
                                logging.error(f"❌ Failed to initialize MCP tools for {executor_name}: {e}")
                        
                        agent = create_react_agent(
                            model=llm,
                            tools=tools,
                            prompt=executor_config["prompt"]
                        )
                        
                        workflow.add_node(
                            executor_name,
                            create_executor_node(agent, executor_name)
                        )
                    
                    # Add edges
                    workflow.add_edge(START, "planner")
                    workflow.add_edge("planner", "router")
                    
                    executor_mapping = {name: name for name in st.session_state.executors}
                    
                    def route_function(state):
                        next_action = state.get("next_action", "")
                        if next_action in executor_mapping:
                            return next_action
                        if state.get("plan") and state.get("current_step", 0) < len(state["plan"]):
                            task_type = state["plan"][state["current_step"]].get("type", "")
                            return TASK_EXECUTOR_MAPPING.get(task_type, list(executor_mapping.keys())[0])
                        return list(executor_mapping.keys())[0]
                    
                    workflow.add_conditional_edges(
                        "router",
                        route_function,
                        executor_mapping
                    )
                    
                    for executor_name in st.session_state.executors:
                        workflow.add_edge(executor_name, "replanner")
                    
                    workflow.add_conditional_edges(
                        "replanner",
                        should_continue,
                        {
                            "continue": "router",
                            "finalize": "final_responder"
                        }
                    )
                    
                    workflow.add_edge("final_responder", END)
                    
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
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create system: {e}")
                    logging.error(f"System creation failed: {e}")

if st.session_state.executors:
    st.markdown("### 📊 Data Upload")
    render_data_upload_section()
    st.markdown("---")

st.markdown("---")

if st.session_state.graph_initialized:
    if st.session_state.get("left_sidebar_collapsed", False):
        st.markdown("""
        <style>
        .stSidebar {
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out;
        }
        .main .block-container {
            padding-left: 1rem;
            max-width: none;
        }
        </style>
        """, unsafe_allow_html=True)
    
    main_col, artifact_col = st.columns([1, 1])
    
    with main_col:
        render_chat_interface()
    
    with artifact_col:
        st.markdown("### 🎨 Artifacts")
        render_artifact_interface()

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.executors:
            st.info("⚠️ **System not initialized**\n\nClick 'Create Plan-Execute System' above to start!")
        else:
            st.info("⚠️ **No executors configured**\n\nUse the sidebar to add executors or load a template!")

st.markdown("---")
st.caption("🍒Cherry AI - Data Science Multi-Agent System with MCP Integration")

if __name__ == "__main__":
    logging.info("Application started")