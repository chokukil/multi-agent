# apps/dashboard.py

import streamlit as st
import pandas as pd
from python_a2a.client import A2AClient
from streamlit_autorefresh import st_autorefresh
from config import REGISTRY_URL
import subprocess
import sys

st.set_page_config(layout="wide", page_title="A2A Agent Control Tower")

# --- Initialize A2A Client for Registry ---
if 'registry_client' not in st.session_state:
    st.session_state.registry_client = A2AClient(REGISTRY_URL)

client = st.session_state.registry_client

# --- Auto-refresh component ---
st_autorefresh(interval=5000, limit=100, key="dashboard_refresh")

# --- Helper Functions ---
def get_system_status():
    try:
        response = client.ask("get_system_status")
        return response.get("agents", [])
    except Exception as e:
        st.error(f"Failed to connect to Registry Server: {e}")
        return []

def get_logs(agent_name="__all__", limit=50):
    try:
        response = client.ask("get_agent_logs", agent_name=agent_name, limit=limit)
        return response.get("logs", [])
    except Exception as e:
        # Don't show error for logs, just return empty
        return []

# --- Main Dashboard UI ---
st.title("🗼 A2A Agent Control Tower")
st.markdown("---")

# --- Key Metrics ---
agents_data = get_system_status()
online_agents = [agent for agent in agents_data if agent.get("status") == "online"]
offline_agents = len(agents_data) - len(online_agents)

col1, col2, col3 = st.columns(3)
col1.metric("Total Agents", len(agents_data))
col2.metric("Online", len(online_agents), delta_color="off")
col3.metric("Offline", offline_agents, delta_color="inverse")

st.markdown("---")

# --- Agent Status Table ---
st.subheader("Agent Status")
if agents_data:
    df = pd.DataFrame(agents_data)
    # Simple status indicator
    df['Status'] = df['status'].apply(lambda s: "🟢 Online" if s == "online" else "🔴 Offline")
    df_display = df[['Status', 'name', 'url', 'last_heartbeat']]
    df_display = df_display.rename(columns={'name': 'Agent Name', 'url': 'URL', 'last_heartbeat': 'Last Heartbeat (UTC)'})
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No agents registered. Is the Registry Server running?")

st.markdown("---")

# --- System-wide Logs ---
st.subheader("Real-time System Logs")
log_container = st.container(height=400)
logs_data = get_logs()

if logs_data:
    for log in logs_data:
        log_color = "green" if "success" in log['type'] else "red" if "error" in log['type'] else "blue"
        log_container.markdown(
            f"`{log['timestamp']}` | **{log['agent']}** | <span style='color:{log_color};'>{log['type']}</span> | `{log['task_id']}` | {log['content']}",
            unsafe_allow_html=True
        )
else:
    log_container.info("No system logs yet.")

# --- System Control (Example) ---
st.sidebar.title("System Control")
if st.sidebar.button("Restart All Agents (run_system.py)", type="primary"):
    # This is a placeholder for a more robust control mechanism.
    # In a real scenario, this would trigger an API call to a process manager.
    try:
        # A simple implementation: find and restart the run_system.py process
        # This is NOT robust for production.
        subprocess.run(["pkill", "-f", "run_system.py"], check=True)
        subprocess.Popen([sys.executable, "run_system.py"])
        st.sidebar.success("Restart signal sent to run_system.py!")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to restart system: {e}") 