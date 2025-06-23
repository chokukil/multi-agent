# File: pages/1_💬_Agent_Chat.py

import streamlit as st
import asyncio
import logging
import platform
from datetime import datetime
from dotenv import load_dotenv
import nest_asyncio

from langchain_core.messages import HumanMessage

from core.plan_execute.planner import planner_node
from core.plan_execute.a2a_executor import A2AExecutor
from core.callbacks.progress_stream import progress_stream_manager
from ui.artifact_manager import render_artifact
from ui.sidebar_components import render_sidebar

# --- Environment Setup ---
def setup_environment():
    """Sets up the environment for the Streamlit app."""
    # Apply nest_asyncio for environments where it's needed
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    nest_asyncio.apply()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# --- Session State Management ---
def initialize_session_state():
    """Initializes session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- UI Rendering ---
def render_chat_history():
    """Renders the chat history from session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict) and "plan_summary" in content:
                st.markdown(content["plan_summary"])
            elif isinstance(content, dict) and "artifact" in content:
                artifact = content["artifact"]
                agent_name = artifact.get('agent_name', 'Unknown Agent')
                exp = st.expander(f"✨ **{agent_name}**로부터 아티팩트 도착", expanded=True)
                if exp:
                    render_artifact(artifact.get('output_type'), artifact.get('output'), exp)
            else:
                st.markdown(str(content))

async def execute_and_render(execution_state: dict):
    """Asynchronously executes the plan and renders updates to the UI."""
    queue = asyncio.Queue()
    progress_stream_manager.register_queue(queue)
    
    # Instantiate the executor and run it
    executor = A2AExecutor()
    executor_task = asyncio.create_task(executor.execute(execution_state))

    active_statuses = {}
    is_done = False

    while not is_done:
        try:
            update = await asyncio.wait_for(queue.get(), timeout=30.0)
            event_type = update.get("event_type")
            data = update.get("data", {})
            step_num = data.get("step")

            if event_type == "agent_start":
                status = st.status(f"⏳ **Step {step_num}:** `{data['agent_name']}` 실행 중...", expanded=True)
                active_statuses[step_num] = status
            
            elif event_type == "agent_end":
                if step_num in active_statuses:
                    active_statuses[step_num].update(label=f"✅ **Step {step_num}:** `{data['agent_name']}` 완료!", state="complete", expanded=False)
                
                artifact_message = {"role": "assistant", "content": {"artifact": data}}
                st.session_state.messages.append(artifact_message)
                
                with st.chat_message("assistant"):
                    agent_name = data.get('agent_name', 'Unknown Agent')
                    exp = st.expander(f"✨ **{agent_name}**로부터 아티팩트 도착", expanded=True)
                    if exp:
                        render_artifact(data.get('output_type'), data.get('output'), exp)
                
            elif event_type == "agent_error":
                if step_num in active_statuses:
                    active_statuses[step_num].update(label=f"❌ **Step {step_num}:** `{data['agent_name']}` 오류 발생", state="error", expanded=True)
                    with active_statuses[step_num]:
                        st.error(data.get("error_message", "알 수 없는 오류"))
            
            queue.task_done()

        except asyncio.TimeoutError:
            if executor_task.done():
                is_done = True
        except Exception:
            is_done = True
    
    progress_stream_manager.unregister_queue()
    final_state = await executor_task
    if final_state and final_state.get("error"):
         st.error(f"최종 실행 실패: {final_state['error']}")
    else:
         st.success("🎉 모든 분석 단계가 성공적으로 완료되었습니다!")

def process_user_query(prompt: str):
    """Processes the user query through the new A2A-based plan-and-execute flow."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": st.session_state.session_id,
        }

        # 1. Planner Execution
        with st.status("🧠 **Thinking...** Analyzing request and building a plan.", expanded=True) as status:
            try:
                plan_state = planner_node(initial_state)
                if not plan_state.get("plan"):
                    status.update(label="Planning Failed", state="error", expanded=False)
                    st.error("Could not create a plan for this request. Please try rephrasing.")
                    st.session_state.messages.append({"role": "assistant", "content": "I was unable to create a plan for this request."})
                    return

                plan_summary = "📋 **Execution Plan**\n\n"
                for step in plan_state["plan"]:
                    plan_summary += f"**{step['step']}. `{step['agent_name']}`** �� `{step['skill_name']}`\n"
                
                status.update(label="✅ Plan Created!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": {"plan_summary": plan_summary}})
                st.markdown(plan_summary)
            except Exception as e:
                status.update(label="Error during planning!", state="error")
                st.error(f"An error occurred during the planning phase: {e}")
                logging.error(f"Planning error: {e}", exc_info=True)
                return

        # 2. Executor Execution
        try:
            # The new executor class expects a Pydantic model, but for now we can pass the dict
            # Let's ensure the core logic works first.
            asyncio.run(execute_and_render(plan_state))
        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
            logging.error(f"Execution error: {e}", exc_info=True)

def main_chat_interface():
    """Main function to render the chat interface."""
    st.title("💬 Agent Chat")
    st.markdown("Direct the AI agent team to perform complex, multi-step data analysis tasks.")
    
    initialize_session_state()
    render_chat_history()
    
    if prompt := st.chat_input("What would you like to analyze today?"):
        process_user_query(prompt)

# --- Main App Execution ---
if __name__ == "__main__":
    st.set_page_config(page_title="Agent Chat", layout="wide", page_icon="💬")
    setup_environment()
    render_sidebar()
    main_chat_interface() 