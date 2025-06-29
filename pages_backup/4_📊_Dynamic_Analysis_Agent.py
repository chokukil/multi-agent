import streamlit as st
import pandas as pd
import requests
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Analysis Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dynamic Analysis Agent")
st.caption("Powered by LangGraph and ADK")

# --- ADK Server Configuration ---
ADK_HOST = "localhost"
ADK_PORT = 8000
AGENT_NAME = "pandas_data_analyst_agent"
ADK_INVOKE_URL = f"http://{ADK_HOST}:{ADK_PORT}/apps/{AGENT_NAME}/invoke"


def run_analysis_agent(user_instructions: str, data_raw: dict):
    """
    Calls the ADK backend to run the analysis agent.
    """
    payload = {
        "input": {
            "user_instructions": user_instructions,
            "data_raw": data_raw,
        },
        "config": {"configurable": {"thread_id": "streamlit-thread"}}
    }
    
    try:
        response = requests.post(ADK_INVOKE_URL, json=payload, timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the agent backend: {e}")
        return None

# --- UI Components ---
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. **Upload your data** using the file uploader.
    2. **Write your instructions** for the analysis in the text area.
    3. **Click 'Run Analysis'** and wait for the results.
    """)

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
instructions = st.text_area("What would you like to do with this data?", height=150, placeholder="e.g., 'Show me the distribution of values.' or 'Create a scatter plot of column A vs column B.'")

if st.button("Run Analysis", use_container_width=True):
    if uploaded_file is None:
        st.warning("Please upload a data file.")
    elif not instructions.strip():
        st.warning("Please provide instructions for the analysis.")
    else:
        # Load data based on file type
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            data_dict = df.to_dict(orient='list')

            with st.spinner("The agent is thinking... Please wait."):
                start_time = time.time()
                result = run_analysis_agent(instructions, data_dict)
                end_time = time.time()
                
                st.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")

            if result and 'output' in result:
                output = result['output']
                
                st.divider()
                st.header("Analysis Results")

                # Display wrangled data if available
                if output.get("data_wrangled"):
                    st.subheader("Processed Data")
                    st.dataframe(pd.DataFrame(output["data_wrangled"]))

                # Display Plotly chart if available
                if output.get("plotly_graph"):
                    st.subheader("Generated Visualization")
                    try:
                        st.plotly_chart(output["plotly_graph"], use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to render Plotly chart: {e}")
                        st.json(output["plotly_graph"])

                # Display workflow summary
                if output.get("messages"):
                    summary = ""
                    agents = [msg.get('name') for msg in output["messages"] if msg.get('name')]
                    if agents:
                        agent_labels = [f"- **Agent {i+1}:** {role}" for i, role in enumerate(agents)]
                        summary += f"### Workflow Summary\nThis workflow contained {len(agents)} agents:\n\n" + "\n".join(agent_labels) + "\n\n"

                    st.subheader("Agent Workflow")
                    st.markdown(summary)

                # Display generated code
                if output.get("data_wrangler_function"):
                    st.subheader("Generated Wrangling Code")
                    st.code(output["data_wrangler_function"], language="python")
                
                if output.get("data_visualization_function"):
                    st.subheader("Generated Visualization Code")
                    st.code(output["data_visualization_function"], language="python")
                    
            else:
                st.error("Failed to get a valid response from the agent.")
                st.json(result)

        except Exception as e:
            st.error(f"An error occurred: {e}") 