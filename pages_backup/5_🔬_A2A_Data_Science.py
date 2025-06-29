# pages/5_🔬_A2A_Data_Science.py
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from core.utils.logging import setup_logging

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="A2A Data Science",
    page_icon="🔬",
    layout="wide",
)

# --- Initial Setup ---
setup_logging()

st.title("🔬 A2A Data Science Agents")
st.markdown("""
**Direct interaction with A2A Data Science Agents**

이 페이지에서는 A2A 프로토콜을 통해 데이터 사이언스 에이전트들과 직접 소통할 수 있습니다. 
각 에이전트는 특화된 기능을 제공하며, 실시간 스트리밍으로 분석 과정을 확인할 수 있습니다.
""")

# ------------------------------------------------------------------------------
# Agent Configuration
# ------------------------------------------------------------------------------
try:
    from config import AGENT_SERVERS
    
    # Filter out orchestrator for direct interaction
    AVAILABLE_AGENTS = {k: v for k, v in AGENT_SERVERS.items() if k != "orchestrator"}
    
except ImportError:
    AVAILABLE_AGENTS = {
        "data_loader": {
            "name": "Data Loader Agent",
            "url": "http://localhost:8000",
            "description": "Data loading and processing with file operations",
            "port": 8000
        },
        "pandas_analyst": {
            "name": "Pandas Data Analyst",
            "url": "http://localhost:8001", 
            "description": "Advanced pandas data analysis with interactive visualizations",
            "port": 8001
        },
        "sql_analyst": {
            "name": "SQL Data Analyst",
            "url": "http://localhost:8002",
            "description": "SQL database analysis with query generation", 
            "port": 8002
        },
        "eda_tools": {
            "name": "EDA Tools Analyst",
            "url": "http://localhost:8003",
            "description": "Comprehensive exploratory data analysis and statistical insights",
            "port": 8003
        },
        "data_visualization": {
            "name": "Data Visualization Analyst",
            "url": "http://localhost:8004",
            "description": "Interactive chart and dashboard creation with Plotly",
            "port": 8004
        }
    }

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

async def check_agent_status(agent_url: str) -> bool:
    """Check if an agent is running."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{agent_url}/.well-known/agent.json")
            return response.status_code == 200
    except:
        return False

async def send_a2a_request(agent_url: str, message: str) -> dict:
    """Send A2A request to an agent."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": {
                    "task_id": f"task_{int(time.time())}",
                    "context_id": f"ctx_{int(time.time())}",
                    "message": {
                        "parts": [{"text": message}]
                    }
                },
                "id": 1
            }
            
            response = await client.post(f"{agent_url}/a2a", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def load_sample_datasets():
    """Load available sample datasets."""
    data_dir = Path("a2a_ds_servers/artifacts/data/shared_dataframes")
    datasets = {}
    
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                datasets[csv_file.stem] = {
                    "path": str(csv_file),
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "preview": df.head(3).to_dict('records')
                }
            except Exception as e:
                st.warning(f"Could not load {csv_file.name}: {e}")
    
    return datasets

# ------------------------------------------------------------------------------
# Session State Management
# ------------------------------------------------------------------------------
if "agent_responses" not in st.session_state:
    st.session_state.agent_responses = {}

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None

if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {}

# ------------------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------------------

# --- Agent Selection ---
st.subheader("🤖 Select Agent")

col1, col2 = st.columns([2, 1])

with col1:
    agent_options = [f"{config['name']} ({key})" for key, config in AVAILABLE_AGENTS.items()]
    selected_option = st.selectbox(
        "Choose an A2A Data Science Agent:",
        ["None"] + agent_options,
        help="Select an agent to interact with"
    )
    
    if selected_option != "None":
        agent_key = selected_option.split("(")[-1].rstrip(")")
        st.session_state.selected_agent = agent_key
        agent_config = AVAILABLE_AGENTS[agent_key]
        
        st.info(f"**Selected:** {agent_config['name']}")
        st.markdown(f"**Description:** {agent_config['description']}")
        st.markdown(f"**Endpoint:** `{agent_config['url']}`")

with col2:
    if st.button("🔄 Check All Agents", use_container_width=True):
        with st.spinner("Checking agent status..."):
            for key, config in AVAILABLE_AGENTS.items():
                status = asyncio.run(check_agent_status(config['url']))
                st.session_state.agent_statuses[key] = "🟢 Online" if status else "🔴 Offline"
        st.success("Status updated!")

# --- Agent Status Display ---
if st.session_state.agent_statuses:
    st.subheader("📊 Agent Status")
    
    status_cols = st.columns(len(AVAILABLE_AGENTS))
    for i, (key, config) in enumerate(AVAILABLE_AGENTS.items()):
        with status_cols[i]:
            status = st.session_state.agent_statuses.get(key, "❓ Unknown")
            st.metric(
                label=config['name'],
                value=status,
                delta=f"Port {config['port']}"
            )

st.divider()

# --- Data Selection ---
st.subheader("📁 Sample Datasets")

datasets = load_sample_datasets()
if datasets:
    dataset_names = list(datasets.keys())
    selected_dataset = st.selectbox(
        "Choose a sample dataset (optional):",
        ["None"] + dataset_names,
        help="Select a dataset to reference in your request"
    )
    
    if selected_dataset != "None":
        dataset_info = datasets[selected_dataset]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Shape:** {dataset_info['shape'][0]} rows × {dataset_info['shape'][1]} columns")
            st.markdown(f"**Columns:** {', '.join(dataset_info['columns'][:5])}{'...' if len(dataset_info['columns']) > 5 else ''}")
        
        with col2:
            st.markdown("**Preview:**")
            st.dataframe(pd.DataFrame(dataset_info['preview']), use_container_width=True)
        
        # Add dataset reference to query
        dataset_reference = f"\n\nDataset: {selected_dataset}.csv (Shape: {dataset_info['shape']}, Columns: {', '.join(dataset_info['columns'])})"
else:
    st.info("No sample datasets found. You can upload your own data or reference external files.")

st.divider()

# --- Query Interface ---
st.subheader("💬 Agent Interaction")

if st.session_state.selected_agent:
    agent_key = st.session_state.selected_agent
    agent_config = AVAILABLE_AGENTS[agent_key]
    
    # Query input
    query_examples = {
        "data_loader": "Load the titanic.csv file and show basic information about the dataset",
        "pandas_analyst": "Analyze the sales data and create a comprehensive report with visualizations",
        "sql_analyst": "Generate SQL queries to analyze customer data and show insights",
        "eda_tools": "Perform exploratory data analysis on the dataset and identify key patterns",
        "data_visualization": "Create interactive visualizations showing the relationship between variables"
    }
    
    example_query = query_examples.get(agent_key, "Analyze the data and provide insights")
    
    user_query = st.text_area(
        f"Enter your request for {agent_config['name']}:",
        placeholder=example_query,
        height=100,
        help="Describe what you want the agent to do with your data"
    )
    
    # Add dataset reference if selected
    if 'selected_dataset' in locals() and selected_dataset != "None":
        user_query += dataset_reference
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Send Request", use_container_width=True, disabled=not user_query.strip()):
            if user_query.strip():
                with st.spinner(f"Sending request to {agent_config['name']}..."):
                    response = asyncio.run(send_a2a_request(agent_config['url'], user_query))
                    
                    # Store response
                    if agent_key not in st.session_state.agent_responses:
                        st.session_state.agent_responses[agent_key] = []
                    
                    st.session_state.agent_responses[agent_key].append({
                        "query": user_query,
                        "response": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.success("Request sent!")
                    st.rerun()
    
    with col2:
        if st.button("🧪 Quick Test", use_container_width=True):
            test_query = f"Hello! Please introduce yourself and explain what you can do."
            
            with st.spinner("Running quick test..."):
                response = asyncio.run(send_a2a_request(agent_config['url'], test_query))
                
                if agent_key not in st.session_state.agent_responses:
                    st.session_state.agent_responses[agent_key] = []
                
                st.session_state.agent_responses[agent_key].append({
                    "query": test_query,
                    "response": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success("Test completed!")
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            if agent_key in st.session_state.agent_responses:
                st.session_state.agent_responses[agent_key] = []
            st.success("History cleared!")
            st.rerun()

else:
    st.info("👆 Please select an agent above to start interacting.")

st.divider()

# --- Response Display ---
st.subheader("📋 Agent Responses")

if st.session_state.selected_agent and st.session_state.selected_agent in st.session_state.agent_responses:
    agent_key = st.session_state.selected_agent
    responses = st.session_state.agent_responses[agent_key]
    
    if responses:
        # Display responses in reverse chronological order
        for i, interaction in enumerate(reversed(responses)):
            with st.expander(f"🕐 {interaction['timestamp']} - Query {len(responses)-i}", expanded=(i==0)):
                st.markdown("**Query:**")
                st.code(interaction['query'], language="text")
                
                st.markdown("**Response:**")
                response = interaction['response']
                
                if 'error' in response:
                    st.error(f"❌ Error: {response['error']}")
                else:
                    if 'result' in response:
                        result = response['result']
                        if isinstance(result, dict):
                            # Pretty print JSON response
                            st.json(result)
                        else:
                            st.markdown(str(result))
                    else:
                        # Display raw response
                        st.json(response)
                
                # Export option
                if st.button(f"💾 Export Response {len(responses)-i}", key=f"export_{agent_key}_{i}"):
                    export_data = {
                        "agent": st.session_state.selected_agent,
                        "timestamp": interaction['timestamp'],
                        "query": interaction['query'],
                        "response": interaction['response']
                    }
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"a2a_response_{agent_key}_{interaction['timestamp'].replace(':', '-')}.json",
                        mime="application/json",
                        key=f"download_{agent_key}_{i}"
                    )
    else:
        st.info("No responses yet. Send a request to see results here.")

elif st.session_state.selected_agent:
    st.info("No interaction history for this agent yet.")

else:
    st.info("Select an agent and send a request to see responses here.")

# --- Footer ---
st.divider()
st.markdown("""
### 💡 Tips for Using A2A Data Science Agents

- **Data Loader Agent**: Best for file operations, data loading, and basic preprocessing
- **Pandas Data Analyst**: Ideal for comprehensive data analysis with pandas and visualizations  
- **SQL Data Analyst**: Perfect for database queries and SQL-based analysis
- **EDA Tools Agent**: Specialized in exploratory data analysis and statistical insights
- **Data Visualization Agent**: Focused on creating interactive charts and dashboards

**Pro Tip**: Start with the EDA Tools Agent for initial data exploration, then use specialized agents for specific tasks!
""") 