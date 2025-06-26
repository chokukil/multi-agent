# File: pages/1_💬_Agent_Chat.py

# Python 경로 설정을 맨 위로 이동
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from core.utils.logging import setup_logging
from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from ui.message_translator import MessageRenderer

# --- Initial Setup ---
setup_logging()

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
    if "message_renderer" not in st.session_state:
        st.session_state.message_renderer = MessageRenderer()
    if "thinking_stream" not in st.session_state:
        st.session_state.thinking_stream = None

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
                
                # 🆕 아름다운 결과 표시 사용
                beautiful_results = BeautifulResults()
                beautiful_results.display_analysis_result(artifact, agent_name)
                
            elif isinstance(content, dict) and "a2a_message" in content:
                # 🆕 A2A 메시지 친화적 렌더링
                st.session_state.message_renderer.render_a2a_message(content["a2a_message"])
            else:
                st.markdown(str(content))

async def execute_and_render(execution_state: dict):
    """Asynchronously executes the plan and renders updates to the UI."""
    queue = asyncio.Queue()
    await progress_stream_manager.register_queue(queue)
    
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
                
                # 🆕 A2A 메시지 처리 개선
                if 'a2a_response' in data:
                    # A2A 응답을 사용자 친화적으로 변환
                    a2a_message = {"role": "assistant", "content": {"a2a_message": data['a2a_response']}}
                    st.session_state.messages.append(a2a_message)
                    
                    with st.chat_message("assistant"):
                        st.session_state.message_renderer.render_a2a_message(data['a2a_response'])
                else:
                    # 기존 아티팩트 처리
                    artifact_message = {"role": "assistant", "content": {"artifact": data}}
                    st.session_state.messages.append(artifact_message)
                    
                    with st.chat_message("assistant"):
                        agent_name = data.get('agent_name', 'Unknown Agent')
                        beautiful_results = BeautifulResults()
                        beautiful_results.display_analysis_result(data, agent_name)
                
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
    
    await progress_stream_manager.unregister_queue(queue)
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

        # 🆕 1. 사고 과정 스트리밍 시작
        thinking_container = st.container()
        thinking_stream = ThinkingStream(thinking_container)
        st.session_state.thinking_stream = thinking_stream
        
        thinking_stream.start_thinking("요청을 분석하고 있습니다...")
        thinking_stream.add_thought("사용자의 요청을 이해하고 적절한 분석 방법을 찾고 있습니다.", "analysis")
        
        # 1. Planner Execution
        with st.status("🧠 **계획 수립 중...** 최적의 분석 전략을 설계하고 있습니다.", expanded=True) as status:
            try:
                thinking_stream.add_thought("데이터 분석에 필요한 단계들을 계획하고 있습니다.", "planning")
                
                plan_state = planner_node(initial_state)
                if not plan_state.get("plan"):
                    thinking_stream.add_thought("계획 수립에 실패했습니다. 다시 시도해주세요.", "error")
                    status.update(label="계획 수립 실패", state="error", expanded=False)
                    st.error("요청에 대한 계획을 수립할 수 없습니다. 다시 표현해 주세요.")
                    st.session_state.messages.append({"role": "assistant", "content": "요청에 대한 계획을 수립할 수 없었습니다."})
                    return

                thinking_stream.add_thought("완벽한 분석 계획이 완성되었습니다!", "success")
                thinking_stream.finish_thinking("계획 수립 완료! 이제 실행을 시작합니다.")
                
                # 🆕 아름다운 계획 시각화
                plan_viz = PlanVisualization()
                plan_viz.display_plan(plan_state["plan"], "🎯 데이터 분석 실행 계획")
                
                status.update(label="✅ 계획 완성!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": {"plan_summary": "계획이 성공적으로 수립되었습니다."}})
                
            except Exception as e:
                thinking_stream.add_thought(f"계획 수립 중 문제가 발생했습니다: {str(e)[:100]}", "error")
                status.update(label="계획 수립 오류!", state="error")
                st.error(f"계획 수립 중 오류가 발생했습니다: {e}")
                logging.error(f"Planning error: {e}", exc_info=True)
                return

        # 2. Executor Execution
        try:
            # The UI needs to run the async function to start the process
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