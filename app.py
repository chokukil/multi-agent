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
    save_multi_agent_config,
    render_chat_interface,
    render_system_status,
    visualize_plan_execute_structure,
    render_bottom_tabs
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

# Initialize
initialize_session_state()

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Configuration")
    
    # Saved systems
    render_saved_systems()
    
    # Current system status
    st.markdown("### 📊 Current System")
    
    if st.session_state.executors:
        st.info(f"🤖 **Executors**: {len(st.session_state.executors)}")
        
        # List executors
        with st.expander("View Executors", expanded=False):
            for name in st.session_state.executors:
                st.write(f"- {name}")
        
        if st.session_state.graph_initialized:
            st.success("✅ **Status**: System Ready!")
        else:
            st.warning("⚠️ **Status**: Not initialized")
    else:
        st.info("📋 No executors created yet")
    
    st.markdown("---")
    
    # Data upload section
    render_data_upload_section()
    
    st.markdown("---")
    
    # Quick templates
    render_quick_templates()
    
    st.markdown("---")
    
    # System settings
    render_system_settings()
    
    st.markdown("---")
    
    # Executor creation
    render_executor_creation_form()
    
    st.markdown("---")
    
    # System management
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

# Main area
st.title("🍒 Cherry AI - Data Science Multi-Agent System")
st.markdown("### Plan-Execute Pattern with SSOT and Data Lineage Tracking")

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
    - **Hallucination Prevention**: Validate data usage
    """)

# Visualize executors
if st.session_state.executors:
    st.markdown("### 📋 Registered Executors")
    
    cols = st.columns(min(len(st.session_state.executors), 3))
    for i, (name, config) in enumerate(st.session_state.executors.items()):
        with cols[i % 3]:
            tools = config.get("tools", [])
            tools_str = ", ".join(tools) if tools else "No tools"
            
            # Get prompt preview
            prompt = config.get("prompt", "No prompt defined")
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            st.markdown(f"""
            <div class='executor-card'>
                <h4>🤖 {name}</h4>
                <p><b>Role:</b> {prompt_preview}</p>
                <p><b>Tools:</b> {tools_str}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Create system button
if st.session_state.executors and not st.session_state.graph_initialized:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Plan-Execute System", type="primary", use_container_width=True):
            with st.spinner("🔧 Building Plan-Execute system..."):
                # Build the graph
                async def build_plan_execute_system():
                    """Build the Plan-Execute graph"""
                    logging.info("Building Plan-Execute system")
                    
                    # Create graph
                    workflow = StateGraph(PlanExecuteState)
                    
                    # Add nodes
                    workflow.add_node("planner", planner_node)
                    workflow.add_node("router", router_node)
                    workflow.add_node("replanner", replanner_node)
                    workflow.add_node("final_responder", final_responder_node)
                    
                    # Add executor nodes
                    llm = create_llm_instance(temperature=0.1)
                    
                    for executor_name, executor_config in st.session_state.executors.items():
                        # Create tools
                        tools = []
                        if "python_repl_ast" in executor_config.get("tools", []):
                            tools.append(create_enhanced_python_tool())
                        
                        # Create agent
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
                    st.success("✅ Plan-Execute System created successfully!")
                    log_event("system_created", {
                        "executors": list(st.session_state.executors.keys()),
                        "timestamp": datetime.now().isoformat()
                    })
                    logging.info("Plan-Execute system created")
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

# Chat interface
if st.session_state.graph_initialized:
    render_chat_interface()
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
st.caption("🍒Cherry AI - Data Science Multi-Agent System")

if __name__ == "__main__":
    logging.info("Application started")