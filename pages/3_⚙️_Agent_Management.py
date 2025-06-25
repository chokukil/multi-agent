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
# In a real-world app, this might come from a config file or agent_registry.
AGENTS = [
    {
        "name": "pandas_data_analyst",
        "url": "http://localhost:10001",
        "health_endpoint": "http://localhost:10001/.well-known/agent.json", # A2A standard endpoint
        "description": "The main orchestrator agent for data analysis tasks.",
    },
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
        
        with st.expander(f"{agent['name']} - {status}", expanded=True):
            st.markdown(f"**URL:** `{agent['url']}`")
            st.markdown(f"**Description:** {agent['description']}")

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