# pages/3_⚙️_Agent_Management.py
import asyncio
import platform
import subprocess
import time

import httpx
import streamlit as st
from core.utils.logging import setup_logging

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Agent Management",
    page_icon="⚙️",
    layout="wide",
)

# --- Initial Setup ---
setup_logging()

st.title("⚙️ Agent Process Management")
st.markdown(
    """
    This dashboard provides an overview of all registered A2A agents and allows you
    to control their processes. You can start or stop all agents collectively
    using the buttons below. The status of each agent is checked via its `/health`
    endpoint.
    """
)

# ------------------------------------------------------------------------------
# Agent Definitions
# ------------------------------------------------------------------------------
# Import from config.py for centralized management
try:
    from config import AGENT_SERVERS
    
    # Convert config format to agent management format
    AGENTS = []
    for key, config in AGENT_SERVERS.items():
        AGENTS.append({
            "name": key,
            "display_name": config.get("name", key.replace("_", " ").title()),
            "url": config["url"],
            "health_endpoint": f"{config['url']}/.well-known/agent.json",
            "description": config.get("description", "A2A Data Science Agent"),
            "port": config["port"]
        })
        
except ImportError:
    # Fallback if config.py is not available
    AGENTS = [
        {
            "name": "data_loader",
            "display_name": "Data Loader Agent",
            "url": "http://localhost:8000",
            "health_endpoint": "http://localhost:8000/.well-known/agent.json",
            "description": "Data loading and processing with file operations",
            "port": 8000
        },
        {
            "name": "pandas_analyst",
            "display_name": "Pandas Data Analyst",
            "url": "http://localhost:8001",
            "health_endpoint": "http://localhost:8001/.well-known/agent.json",
            "description": "Advanced pandas data analysis with interactive visualizations",
            "port": 8001
        },
        {
            "name": "sql_analyst",
            "display_name": "SQL Data Analyst",
            "url": "http://localhost:8002",
            "health_endpoint": "http://localhost:8002/.well-known/agent.json",
            "description": "SQL database analysis with query generation",
            "port": 8002
        },
        {
            "name": "eda_tools",
            "display_name": "EDA Tools Analyst",
            "url": "http://localhost:8003",
            "health_endpoint": "http://localhost:8003/.well-known/agent.json",
            "description": "Comprehensive exploratory data analysis and statistical insights",
            "port": 8003
        },
        {
            "name": "data_visualization",
            "display_name": "Data Visualization Analyst",
            "url": "http://localhost:8004",
            "health_endpoint": "http://localhost:8004/.well-known/agent.json",
            "description": "Interactive chart and dashboard creation with Plotly",
            "port": 8004
        },
        {
            "name": "orchestrator",
            "display_name": "Data Science Orchestrator",
            "url": "http://localhost:8100",
            "health_endpoint": "http://localhost:8100/.well-known/agent.json",
            "description": "Central management and orchestration of all data science agents",
            "port": 8100
        }
    ]

# Initialize session state for agent statuses
if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {agent["name"]: "Checking..." for agent in AGENTS}

# ------------------------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------------------------

async def check_agent_health(agent: dict):
    """Asynchronously checks the health of a single agent."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(agent["health_endpoint"])
            if response.status_code == 200:
                # For A2A, we just check if we can get the agent card
                return "🟢 Running"
            else:
                return f"🔴 Stopped ({response.status_code})"
    except httpx.RequestError:
        return "🔴 Stopped"

async def update_all_agent_statuses():
    """Gathers health check results for all agents concurrently."""
    tasks = [check_agent_health(agent) for agent in AGENTS]
    results = await asyncio.gather(*tasks)
    
    # Create a new dictionary to trigger a UI update
    new_statuses = {}
    for agent, status in zip(AGENTS, results):
        new_statuses[agent["name"]] = status
    st.session_state.agent_statuses = new_statuses
    st.rerun()

def run_system_script(script_name: str):
    """Runs a system script (.sh or .bat) based on the OS."""
    is_windows = platform.system() == "Windows"
    script_file = f"{script_name}.bat" if is_windows else f"./{script_name}.sh"
    
    st.info(f"Executing `{script_file}`...")
    try:
        # For .sh, it must be executable. For .bat, this is not needed.
        if not is_windows:
            subprocess.run(["chmod", "+x", script_file], check=True)
        
        # We use Popen to run in the background without blocking the UI
        process = subprocess.Popen(script_file, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # We don't wait for it to finish, just launch it
        st.toast(f"Successfully launched `{script_file}`!")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        st.error(f"Failed to execute `{script_file}`: {e}")
        st.code(e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "No stderr output.")

# ------------------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------------------

# --- Control Buttons ---
st.subheader("System Control")
col1, col2, col3 = st.columns([1, 1, 5])
with col1:
    if st.button("🚀 Start All Agents", use_container_width=True):
        run_system_script("system_start")
        time.sleep(3) # Give agents time to start before next refresh
        st.session_state.agent_statuses = {agent["name"]: "Checking..." for agent in AGENTS}

with col2:
    if st.button("🛑 Stop All Agents", use_container_width=True):
        run_system_script("system_stop")
        time.sleep(2) # Give OS time to stop processes
        st.session_state.agent_statuses = {agent["name"]: "Stopped" for agent in AGENTS}

with col3:
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state.agent_statuses = {agent["name"]: "Checking..." for agent in AGENTS}


# --- Agent Status Table ---
st.subheader("Agent Status")
status_placeholder = st.empty()

# Display current statuses from session state
with status_placeholder.container():
    for agent in AGENTS:
        status = st.session_state.agent_statuses.get(agent["name"], "Unknown")
        display_name = agent.get("display_name", agent["name"])
        
        with st.expander(f"{display_name} - {status}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Service:** `{agent['name']}`")
                st.markdown(f"**URL:** `{agent['url']}`")
                st.markdown(f"**Port:** `{agent['port']}`")
            
            with col2:
                st.markdown(f"**Description:** {agent['description']}")
                
                # Individual agent control buttons
                col_start, col_stop = st.columns(2)
                with col_start:
                    if st.button(f"🚀 Start", key=f"start_{agent['name']}", use_container_width=True):
                        st.info(f"Starting {display_name}...")
                        # You can add individual start logic here
                        
                with col_stop:
                    if st.button(f"🛑 Stop", key=f"stop_{agent['name']}", use_container_width=True):
                        st.info(f"Stopping {display_name}...")
                        # You can add individual stop logic here

# --- Logic to run the async status update ---
# This runs automatically when the page loads or refresh is clicked
if any("Checking..." in status for status in st.session_state.agent_statuses.values()):
    try:
        # This is the correct way to run an async function from the top-level script in Streamlit
        asyncio.run(update_all_agent_statuses())
    except Exception as e:
        # This can happen if another asyncio loop is running, e.g. in some versions of Streamlit/Jupyter
        # Let's try to get the existing loop instead.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(update_all_agent_statuses())
        except RuntimeError:
             st.warning(f"Could not run status update automatically. Please use the Refresh button. Error: {e}") 