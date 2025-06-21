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
    """채팅 인터페이스 렌더링 - 실시간 도구 활동 표시 포함"""
    st.markdown("### 💬 Enhanced Multi-Agent Chat with Real-time Tool Monitoring")
    
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
            # 응답 컨테이너
            response_container = st.container()
            
            # 실시간 도구 활동 모니터링 섹션
            tool_activity_container = st.container()
            
            with response_container:
                st.markdown("### 📝 Final Response:")
                text_placeholder = st.empty()
                text_placeholder.markdown("*분석을 시작합니다...*")
            
            with tool_activity_container:
                tool_expander = st.expander("🔬 실행 과정 (Tool Activity)", expanded=True)
                with tool_expander:
                    tool_activity_placeholder = st.empty()
                    tool_activity_placeholder.markdown("*도구 실행을 기다리는 중...*")
            
            # 쿼리 처리
            resp, final_text, tool_activity_content = (
                st.session_state.event_loop.run_until_complete(
                    process_query_with_enhanced_streaming(
                        user_query,
                        text_placeholder,
                        tool_activity_placeholder,
                        st.session_state.get("timeout_seconds", 180)
                    )
                )
            )
            
            # 완료 후 정리
            if tool_activity_content:
                with tool_activity_container:
                    with st.expander("🔬 실행 과정 (Tool Activity) - 완료", expanded=False):
                        st.markdown(tool_activity_content)

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
                "tool_activity": tool_activity_content
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
                
                # 에러 메시지가 아닌 경우에만 도구 활동 표시
                if not message.get("error") and message.get("tool_activity"):
                    with st.expander("🔬 실행 과정 (Tool Activity)", expanded=False):
                        st.markdown(message["tool_activity"])

async def process_query_with_enhanced_streaming(
    query: str,
    text_placeholder,
    tool_activity_placeholder,
    timeout_seconds: int = 180
) -> Tuple[Dict, str, str]:
    """향상된 스트리밍으로 쿼리 처리"""
    from core.utils.streaming import astream_graph, get_plan_execute_streaming_callback
    
    logging.info(f"Processing query with enhanced streaming: {query}")
    
    start_time = time.time()
    
    try:
        if st.session_state.plan_execute_graph:
            # 새로운 스트리밍 콜백 설정
            streaming_callback = get_plan_execute_streaming_callback(tool_activity_placeholder)
            
            # 최종 응답 수집을 위한 변수
            final_text_parts = []
            
            def response_callback(msg):
                """최종 응답 수집 콜백"""
                node = msg.get("node", "")
                content = msg.get("content")
                
                if node == "final_responder" and hasattr(content, "content"):
                    final_text_parts.append(content.content)
                    text_placeholder.markdown("".join(final_text_parts))
                elif node == "final_responder" and isinstance(content, str):
                    final_text_parts.append(content)
                    text_placeholder.markdown("".join(final_text_parts))
            
            # 통합 콜백
            def combined_callback(msg):
                streaming_callback(msg)
                response_callback(msg)
            
            # 그래프 실행
            from langchain_core.messages import HumanMessage
            from langchain_core.runnables import RunnableConfig
            
            # 디버깅: 세션 정보 확인
            session_id = st.session_state.get('thread_id', 'default-session')
            user_id = st.session_state.get('user_id', 'default-user')
            logging.info(f"🔍 Chat Interface - Session ID: {session_id}")
            logging.info(f"🔍 Chat Interface - User ID: {user_id}")
            logging.info(f"🔍 Chat Interface - st.session_state keys: {list(st.session_state.keys())}")
            
            config = RunnableConfig(
                recursion_limit=st.session_state.get("recursion_limit", 30),
                configurable={"thread_id": st.session_state.thread_id}
            )
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "session_id": session_id,
                "user_id": user_id
            }
            logging.info(f"🔍 Chat Interface - Initial state: {initial_state}")
            
            response = await asyncio.wait_for(
                astream_graph(
                    st.session_state.plan_execute_graph,
                    initial_state,
                    callback=combined_callback,
                    config=config
                ),
                timeout=timeout_seconds
            )
            
            # 최종 텍스트 추출
            final_text = "".join(final_text_parts)
            
            # 최종 응답이 없다면 response에서 추출
            if not final_text and response:
                if isinstance(response, dict) and "messages" in response:
                    for msg in reversed(response["messages"]):
                        if hasattr(msg, "name") and msg.name == "Final_Responder":
                            final_text = msg.content
                            break
                        elif hasattr(msg, "content") and msg.content:
                            # 마지막 AI 메시지를 최종 응답으로 사용
                            if hasattr(msg, "type") and msg.type == "ai":
                                final_text = msg.content
                                break
            
            # 도구 활동 내용 가져오기 (실제로는 streaming_callback에서 관리됨)
            tool_activity_content = "실행이 완료되었습니다."
            
            duration = time.time() - start_time
            logging.info(f"Query processed in {duration:.2f}s")
            
            if not final_text:
                final_text = "분석이 완료되었습니다. 자세한 내용은 실행 과정을 확인해주세요."
            
            text_placeholder.markdown(final_text)
            
            return response, final_text, tool_activity_content
            
        else:
            error_msg = "🚫 Plan-Execute system not initialized"
            return {"error": error_msg}, error_msg, ""
            
    except asyncio.TimeoutError:
        error_msg = f"⏱️ Request timed out after {timeout_seconds} seconds"
        text_placeholder.markdown(error_msg)
        return {"error": error_msg}, error_msg, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logging.error(f"Query processing error: {e}")
        text_placeholder.markdown(error_msg)
        return {"error": error_msg}, error_msg, ""

async def process_query_with_plan_execute(
    query: str,
    plan_placeholder,
    progress_placeholder,
    text_placeholder,
    tool_placeholder,
    timeout_seconds: int = 180
) -> Tuple[Dict, Dict, str, str, str]:
    """Plan-Execute 패턴으로 쿼리 처리 (레거시 호환성)"""
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
            
            # 디버깅: 세션 정보 확인
            session_id = st.session_state.get('thread_id', 'default-session')
            user_id = st.session_state.get('user_id', 'default-user')
            logging.info(f"🔍 Legacy Chat Interface - Session ID: {session_id}")
            logging.info(f"🔍 Legacy Chat Interface - User ID: {user_id}")
            
            config = RunnableConfig(
                recursion_limit=30,
                configurable={"thread_id": st.session_state.thread_id}
            )
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "session_id": session_id,
                "user_id": user_id
            }
            logging.info(f"🔍 Legacy Chat Interface - Initial state: {initial_state}")
            
            response = await asyncio.wait_for(
                astream_graph(
                    st.session_state.plan_execute_graph,
                    initial_state,
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