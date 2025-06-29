# File: pages/2_📊_EDA_Copilot.py

import streamlit as st
from ui.sidebar_components import render_sidebar
from core.data_manager import data_manager
import pandas as pd
import asyncio
import nest_asyncio
import platform
import logging

from core.plan_execute.a2a_executor import A2AExecutor
from core.callbacks.progress_stream import progress_stream_manager
from ui.artifact_manager import render_artifact
from core.utils.logging import setup_logging

# Apply nest_asyncio to allow running asyncio event loops within Streamlit
def setup_environment():
    """Sets up the environment for asyncio."""
    # Call the central logging setup
    setup_logging()

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    # The basicConfig call is now handled by setup_logging()
    # logging.basicConfig(level=logging.INFO)

async def run_and_render_eda_skill(skill_name: str, df_key: str, params: dict = None):
    """
    Constructs a plan for a single EDA skill, executes it, and renders the artifact.
    """
    plan = [{
        "step": 1,
        "agent_name": "EDA",
        "skill_name": skill_name,
        "params": {"dataframe_key": df_key, **(params or {})},
    }]
    
    execution_state = {"plan": plan}
    
    queue = asyncio.Queue()
    progress_stream_manager.register_queue(queue)
    
    executor = A2AExecutor()
    executor_task = asyncio.create_task(executor.execute(execution_state))

    result_container = st.container()
    result_container.empty()

    is_done = False
    with st.spinner(f"Running `{skill_name}` on `{df_key}`... Please wait."):
        while not is_done:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=120.0) # Increased timeout for long tasks
                event_type = update.get("event_type")
                data = update.get("data", {})

                if event_type == "agent_end":
                    with result_container:
                        st.subheader(f"Output from `{data.get('agent_name')}`")
                        exp = st.expander("✨ Artifact", expanded=True)
                        render_artifact(data.get('output_type'), data.get('output'), exp)
                
                elif event_type == "agent_error":
                    with result_container:
                        st.error(f"An error occurred: {data.get('error_message')}")

                queue.task_done()

            except asyncio.TimeoutError:
                if executor_task.done():
                    is_done = True
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                is_done = True
    
    progress_stream_manager.unregister_queue(queue)
    final_state = await executor_task
    if final_state and final_state.get("error"):
         st.error(f"Final execution failed: {final_state['error']}")
    else:
         st.success("🎉 Analysis complete!")


def render_eda_copilot():
    """Renders the EDA Copilot page."""
    st.title("📊 EDA Copilot")
    st.markdown("A guided tool for Exploratory Data Analysis. Upload your dataset and let the AI assistant generate key insights.")

    # Render sidebar for agent configuration, etc.
    render_sidebar()

    # File Uploader
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "excel"])

    if uploaded_file is not None:
        try:
            # Use a unique key for the dataframe in the data manager
            df_key = f"eda_copilot_{uploaded_file.name}"
            
            # Check if this file is already loaded to prevent infinite refresh
            if df_key not in data_manager.list_dataframes():
                # Load data into a DataFrame
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                data_manager.add_dataframe(df_key, df)
                st.success(f"Successfully loaded `{uploaded_file.name}`. The DataFrame is now available as `{df_key}`.")
            else:
                st.info(f"`{uploaded_file.name}` is already loaded as `{df_key}`.")
            
            # Get the dataframe for display (whether newly loaded or existing)
            df = data_manager.get_dataframe(df_key)
            if df is not None:
                st.dataframe(df.head())

                st.subheader("Automated EDA Actions")
                
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Generate Profile Report", use_container_width=True, key="profile_report"):
                        asyncio.run(run_and_render_eda_skill("generate_profile_report", df_key))

                with col2:
                    if st.button("Analyze Missing Values", use_container_width=True, key="missing_values"):
                        asyncio.run(run_and_render_eda_skill("get_missing_values_summary", df_key))

                with col3:
                    if st.button("Run Correlation Analysis", use_container_width=True, key="correlation_analysis"):
                        asyncio.run(run_and_render_eda_skill("get_correlation_matrix_plot", df_key))
            else:
                st.error(f"Failed to retrieve dataframe `{df_key}` from DataManager.")

        except Exception as e:
            st.error(f"Failed to load or process the file: {e}")

    else:
        st.info("Please upload a file to begin.")

# --- Main App Execution ---
if __name__ == "__main__":
    st.set_page_config(page_title="EDA Copilot", layout="wide", page_icon="📊")
    setup_environment()
    render_eda_copilot() 