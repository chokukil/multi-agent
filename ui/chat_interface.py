# File: ui/chat_interface.py
# Location: ./ui/chat_interface.py

import streamlit as st
import asyncio
import time
import logging
from typing import Dict, Tuple
from core import data_manager

MAX_HISTORY_LENGTH = 20  # 최대 대화 기록 수 (사용자 입력 + 어시스턴트 응답 = 1턴)

def render_chat_interface():
    """채팅 인터페이스 렌더링"""
    st.markdown("### 💬 Enhanced Multi-Agent Chat with Plan-Execute Pattern")
    
    # 데이터 상태 표시
    if data_manager.is_data_loaded():
        st.info(f"📊 **Data Ready**: {data_manager.get_status_message()}")
    else:
        st.warning("📂 **No Data**: Upload CSV in sidebar for data analysis")
    
    # 대화 기록 출력
    print_chat_history()
    
    # 사용자 입력 처리
    user_query = st.session_state.pop("user_query", None) or st.chat_input("💬 질문을 입력하세요")
    
    if user_query:
        # 사용자 메시지 표시
        st.chat_message("user", avatar="🧑").write(user_query)
        st.session_state.history.append({"role": "user", "content": user_query})
        
        # 어시스턴트 응답 처리
        with st.chat_message("assistant", avatar="🤖"):
            # 상태 컨테이너들
            plan_container = st.container()
            progress_container = st.container() 
            tool_container = st.container()
            response_container = st.container()
            
            # 초기 상태 표시
            with plan_container:
                plan_placeholder = st.empty()
                plan_placeholder.info("🎯 Creating execution plan...")
            
            with progress_container:
                progress_placeholder = st.empty()
            
            with tool_container:
                tool_expander = st.expander("🔧 Execution Details", expanded=True)
                with tool_expander:
                    tool_placeholder = st.empty()
                    tool_placeholder.markdown("*Waiting for execution...*")
            
            with response_container:
                st.markdown("### 📝 Final Response:")
                text_placeholder = st.empty()
                text_placeholder.markdown("*Generating response...*")
            
            # 쿼리 처리
            resp, plan_info, progress_info, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query_with_plan_execute(
                        user_query,
                        plan_placeholder,
                        progress_placeholder,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.get("timeout_seconds", 180)
                    )
                )
            )
            
            # 완료 후 정리
            if final_tool:
                tool_container.empty()
                with tool_container:
                    with st.expander("🔧 Execution Details (Completed)", expanded=False):
                        st.markdown(final_tool)

        # 응답을 대화 기록에 추가
        if "error" in resp:
            assistant_response = {
                "role": "assistant",
                "content": resp.get("error", "An unknown error occurred."),
                "error": True
            }
        else:
            assistant_response = {
                "role": "assistant",
                "content": final_text,
                "plan": plan_info,
                "tool_output": final_tool
            }
        st.session_state.history.append(assistant_response)
        
        # 대화 기록 길이 제한 (user + assistant = 2 messages per turn)
        while len(st.session_state.history) > MAX_HISTORY_LENGTH * 2:
            st.session_state.history.pop(0)

        st.rerun()

def print_chat_history():
    """대화 기록 출력"""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑").write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
                
                # 에러 메시지가 아닌 경우에만 계획과 도구 출력 표시
                if not message.get("error"):
                    # 계획 정보가 있으면 표시
                    if "plan" in message and message["plan"]:
                        with st.expander("📋 Execution Plan", expanded=False):
                            st.json(message["plan"])
                    
                    # 도구 출력이 있으면 표시
                    if "tool_output" in message and message["tool_output"]:
                        with st.expander("🔧 Execution Details", expanded=False):
                            st.write(message["tool_output"])

async def process_query_with_plan_execute(
    query: str,
    plan_placeholder,
    progress_placeholder,
    text_placeholder,
    tool_placeholder,
    timeout_seconds: int = 180
) -> Tuple[Dict, Dict, str, str, str]:
    """Plan-Execute 패턴으로 쿼리 처리"""
    from core import astream_graph, get_streaming_callback
    
    logging.info(f"Processing query with Plan-Execute pattern: {query}")
    
    start_time = time.time()
    plan_info = {}
    progress_info = ""
    
    try:
        if st.session_state.plan_execute_graph:
            # 스트리밍 콜백 설정
            streaming_callback, accumulated_text, accumulated_tool = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            
            # Plan 표시 콜백
            def plan_callback(msg):
                if msg.get("node") == "planner" and "plan" in msg.get("content", {}):
                    plan_info.update(msg["content"]["plan"])
                    plan_placeholder.json(plan_info)
                    
            # Progress 표시 콜백
            def progress_callback(msg):
                if msg.get("node") == "replanner":
                    content = msg.get("content", {})
                    if "progress" in content:
                        progress_placeholder.progress(
                            content["progress"]["completed"] / content["progress"]["total"],
                            text=content["progress"]["text"]
                        )
            
            # 통합 콜백
            def combined_callback(msg):
                streaming_callback(msg)
                plan_callback(msg)
                progress_callback(msg)
            
            # 그래프 실행
            from langchain_core.messages import HumanMessage
            from langchain_core.runnables import RunnableConfig
            
            config = RunnableConfig(
                recursion_limit=30,
                configurable={"thread_id": st.session_state.thread_id}
            )
            
            response = await asyncio.wait_for(
                astream_graph(
                    st.session_state.plan_execute_graph,
                    {"messages": [HumanMessage(content=query)]},
                    callback=combined_callback,
                    config=config
                ),
                timeout=timeout_seconds
            )
            
            final_text = "".join(accumulated_text)
            final_tool = "".join(accumulated_tool)
            
            # 최종 응답이 없다면 response에서 추출
            if not final_text and response:
                if isinstance(response, dict) and "messages" in response:
                    for msg in reversed(response["messages"]):
                        if hasattr(msg, "name") and msg.name == "Final_Responder":
                            final_text = msg.content
                            break
            
            duration = time.time() - start_time
            logging.info(f"Query processed in {duration:.2f}s")
            
            return response, plan_info, progress_info, final_text, final_tool
            
        else:
            error_msg = "🚫 Plan-Execute system not initialized"
            return {"error": error_msg}, {}, "", error_msg, ""
            
    except asyncio.TimeoutError:
        error_msg = f"⏱️ Request timed out after {timeout_seconds} seconds"
        return {"error": error_msg}, plan_info, progress_info, error_msg, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logging.error(f"Query processing error: {e}")
        return {"error": error_msg}, plan_info, progress_info, error_msg, ""

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