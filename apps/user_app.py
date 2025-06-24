# apps/user_app.py

import streamlit as st
import pandas as pd
import json
import uuid
import plotly.graph_objects as go
from python_a2a.client import A2AClient
from config import AGENT_SERVERS

st.set_page_config(layout="wide")

# --- Initialize Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "a2a_client" not in st.session_state:
    st.session_state.a2a_client = A2AClient(AGENT_SERVERS["pandas_analyst"]["url"])

# --- Sidebar ---
with st.sidebar:
    st.title("📊 A2A Data Analyst")
    st.write("---")
    
    agent_choice = st.selectbox("Choose an Agent", ["Pandas Data Analyst", "SQL Data Analyst (coming soon)"])
    
    uploaded_file = st.file_uploader("Upload your data (CSV)", type="csv")
    
    st.write("---")
    st.info(f"Session ID: {st.session_state.session_id}")

# --- Main Chat Interface ---
st.title(f"Chat with {agent_choice}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle different content types
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        elif isinstance(message["content"], go.Figure):
            st.plotly_chart(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

def render_final_results(result_data):
    """Renders the final structured results in tabs."""
    summary = result_data.get("summary", "No summary provided.")
    wrangled_data_json = result_data.get("wrangled_data")
    plot_json = result_data.get("plot")
    wrangler_code = result_data.get("wrangler_code", "# No code generated")
    viz_code = result_data.get("visualization_code", "# No code generated")

    summary_tab, data_tab, code_tab, plot_tab = st.tabs([
        "📊 Analysis Summary", "📄 Wrangled Data", "💻 Generated Code", "📈 Chart"
    ])

    with summary_tab:
        st.markdown(summary)

    with data_tab:
        if wrangled_data_json:
            df = pd.read_json(wrangled_data_json, orient='split')
            st.dataframe(df)
        else:
            st.info("No wrangled data available.")

    with code_tab:
        st.subheader("Data Wrangling Function")
        st.code(wrangler_code, language='python')
        st.subheader("Data Visualization Function")
        st.code(viz_code, language='python')

    with plot_tab:
        if plot_json:
            fig = go.Figure(json.loads(plot_json))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chart was generated for this analysis.")

# Handle user input
if prompt := st.chat_input("What would you like to analyze?"):
    if uploaded_file is None:
        st.warning("Please upload a data file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Start analysis stream
        with st.chat_message("assistant"):
            status_container = st.status("Connecting to agent...", expanded=True)
            final_result_data = None
            
            try:
                client = st.session_state.a2a_client
                df = pd.read_csv(uploaded_file)
                data_raw_json = df.to_json(orient='split')

                # Use the streaming skill
                stream = client.stream(
                    "run_pandas_analysis_stream",
                    user_instructions=prompt,
                    data_raw_json=data_raw_json,
                    session_id=st.session_state.session_id
                )

                for chunk in stream:
                    response = json.loads(chunk)
                    event_type = response.get("event")
                    node_name = response.get("node_name", "graph")
                    
                    status_container.update(label=f"Running: `{node_name}`", state="running")
                    
                    if event_type == "on_chain_stream" and response.get("output"):
                        content = response["output"]
                        if isinstance(content, list) and content:
                            msg = content[0].get('content', '')
                            if "Plan:" in msg:
                                status_container.write(f"**Plan:**\n```markdown\n{msg}\n```")

                    elif event_type == "final_result":
                        final_result_data = response.get("output", {})
                        status_container.update(label="Analysis complete!", state="complete", expanded=False)
                        break

                    elif response.get("status") == "error":
                        st.error(f"An error occurred: {response.get('output')}")
                        status_container.update(label="Error!", state="error", expanded=True)
                        break
            
            except Exception as e:
                st.error(f"Failed to process request: {e}")
                status_container.update(label="Connection Error!", state="error", expanded=True)

            if final_result_data:
                render_final_results(final_result_data)
                # Add results to session state if needed for history
                # st.session_state.messages.append({"role": "assistant", "content": ...}) 