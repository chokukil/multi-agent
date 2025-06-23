# File: ui/chat_interface.py
# Location: ./ui/chat_interface.py

import streamlit as st
import asyncio
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage

from core.plan_execute.planner import planner_node
from core.plan_execute.a2a_executor import a2a_executor_node
from core.callbacks.progress_stream import progress_stream_manager
from ui.artifact_manager import render_artifact

def initialize_session_state():
    """세션 상태 변수를 초기화합니다."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_chat_history():
    """세션 상태에서 채팅 기록을 렌더링합니다."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict) and "plan_summary" in content:
                st.markdown(content["plan_summary"])
            elif isinstance(content, dict) and "artifact" in content:
                artifact = content["artifact"]
                agent_name = artifact.get('agent_name', 'Unknown Agent')
                exp = st.expander(f"✨ **{agent_name}**로부터 아티팩트 도착", expanded=True)
                render_artifact(artifact.get('output_type'), artifact.get('output'), exp)
            else:
                st.markdown(str(content))

def process_user_query(prompt: str):
    """새로운 A2A 기반 계획-실행 흐름을 통해 사용자 쿼리를 처리합니다."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": st.session_state.session_id,
        }

        # 1. 플래너 실행
        with st.status("🧠 **Thinking...** 분석 계획을 수립하고 있습니다.", expanded=True) as status:
            try:
                plan_state = planner_node(initial_state)
                if not plan_state.get("plan"):
                    status.update(label="계획 수립 실패", state="error", expanded=False)
                    st.error("요청을 처리할 계획을 세우지 못했습니다. 다른 방식으로 다시 요청해 주세요.")
                    st.session_state.messages.append({"role": "assistant", "content": "이 요청에 대한 계획을 세울 수 없었습니다."})
                    return

                plan_summary = "📋 **Execution Plan**\n\n"
                for step in plan_state["plan"]:
                    plan_summary += f"**{step['step']}. `{step['agent_name']}`** 👉 `{step['skill_name']}`\n"
                
                status.update(label="✅ 계획 수립 완료!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": {"plan_summary": plan_summary}})
                st.markdown(plan_summary)
            except Exception as e:
                status.update(label="계획 수립 중 오류 발생!", state="error")
                st.error(f"계획 단계에서 오류가 발생했습니다: {e}")
                logging.error(f"Planning error: {e}", exc_info=True)
                return

        # 2. 실행기 실행
        try:
            # 비동기 함수 실행
            asyncio.run(execute_and_render(plan_state))
        except Exception as e:
            st.error(f"실행 중 오류가 발생했습니다: {e}")
            logging.error(f"Execution error: {e}", exc_info=True)

async def execute_and_render(execution_state: dict):
    """실행기를 비동기적으로 실행하고 UI에 업데이트를 렌더링합니다."""
    queue = asyncio.Queue()
    progress_stream_manager.register_queue(queue)
    
    executor_task = asyncio.create_task(a2a_executor_node(execution_state))

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
                break
    
    progress_stream_manager.unregister_queue()
    final_state = await executor_task
    if final_state.get("error"):
         st.error(f"최종 실행 실패: {final_state['error']}")
    else:
         st.success("🎉 모든 분석 단계가 성공적으로 완료되었습니다!")

def render_chat_interface():
    st.title("🍒 CherryAI: A2A-Powered Data Science Team")
    
    initialize_session_state()
    render_chat_history()
    
    if prompt := st.chat_input("오늘 무엇을 분석해드릴까요?"):
        process_user_query(prompt)

# This allows the app to be run directly
if __name__ == "__main__":
    render_chat_interface()

def render_system_status():
    """시스템 상태 표시"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        executors_count = len(st.session_state.get("executors", {}))
        st.metric("Executors", executors_count)
    
    with col2:
        messages_count = len(st.session_state.get("history", []))
        st.metric("Messages", messages_count)
    
    with col3:
        status = "✅ Ready" if st.session_state.get("graph_initialized") else "⚠️ Not Initialized"
        st.metric("System", status)
    
    with col4:
        has_data = "✅ Loaded" if data_manager.is_data_loaded() else "❌ Empty"
        st.metric("Data", has_data)