# File: ui/chat_interface.py
# Location: ./ui/chat_interface.py

import streamlit as st
import asyncio
import time
import logging
from typing import Dict, Tuple
from langchain_core.messages import HumanMessage
from core import data_manager
from datetime import datetime

# Core modules
from core import data_manager

# 새로운 콜백 및 스트리밍 함수 임포트
from core.callbacks.chat_stream import ChatStreamCallback
from core.callbacks.progress_stream import ProgressStreamCallback
from core.callbacks.artifact_stream import ArtifactStreamCallback
from core.utils.streaming import astream_graph_with_callbacks
from core.streaming.typed_chat_stream import TypedChatStreamCallback
from core.utils.streaming import create_timeout_aware_callback
from core.execution.timeout_manager import TimeoutManager

# Query Router는 이제 그래프 내부에서 처리됨

MAX_HISTORY_LENGTH = 20  # 최대 대화 기록 수 (사용자 입력 + 어시스턴트 응답 = 1턴)

def render_chat_interface():
    """메인 채팅 인터페이스를 렌더링합니다."""
    st.title("📄 Enhanced Multi-Agent Chat")
    
    # 메시지 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 데이터 업로더
    if not data_manager.is_data_loaded():
        uploaded_file = st.file_uploader("먼저 데이터 파일을 업로드하세요.", type=["csv", "xlsx", "json"])
        if uploaded_file:
            data_manager.load_data(uploaded_file)
            st.success(f"✅ '{uploaded_file.name}'이 성공적으로 로드되었습니다.")
            st.rerun()
    
    # 채팅 기록 출력
    print_chat_history()

    # 사용자 입력
    if prompt := st.chat_input("분석할 내용을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # UI 플레이스홀더 설정
            progress_placeholder = st.empty()
            response_placeholder = st.empty()
            
            # 동기 작업 실행
            try:
                process_query_with_timeout_and_streaming(
                    prompt, progress_placeholder, response_placeholder
                )
            except Exception as e:
                logging.error(f"Error processing query: {e}", exc_info=True)
                st.error(f"오류가 발생했습니다: {e}")
                # 에러 메시지도 히스토리에 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 오류가 발생했습니다: {e}",
                    "timestamp": datetime.now().isoformat()
                })

def print_chat_history():
    """세션의 채팅 기록을 출력합니다."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

def process_query_with_timeout_and_streaming(
    query: str, progress_placeholder, response_placeholder, timeout_seconds: int = 180
):
    """쿼리 처리 with 개선된 타임아웃 및 스트리밍 - 동기 방식"""
    
    # 🆕 타임아웃 매니저 초기화
    timeout_manager = TimeoutManager()
    
    # 쿼리 복잡도 분석
    complexity_info = timeout_manager.analyze_query_complexity(query)
    
    # 동적 타임아웃 계산
    timeout_seconds = timeout_manager.calculate_timeout(
        complexity=complexity_info['complexity'],
        agent_type='EDA_Analyst'  # 주요 에이전트
    )
    
    progress_placeholder.info(f"📊 Query Complexity: {complexity_info['complexity'].value} | ⏱️ Timeout: {timeout_seconds}s")
    
    # 워크플로우 확인
    if "plan_execute_graph" not in st.session_state or not st.session_state.plan_execute_graph:
        response_placeholder.error("🚫 Plan-Execute 시스템이 초기화되지 않았습니다.")
        return
    
    # 기존의 안정적인 콜백 사용
    chat_callback = ChatStreamCallback(response_placeholder)
    progress_callback = ProgressStreamCallback(progress_placeholder)
    artifact_callback = ArtifactStreamCallback()
    
    callbacks = [chat_callback, progress_callback, artifact_callback]
    
    try:
        # 세션 ID 설정
        session_id = st.session_state.get('thread_id', 'default-session')
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "session_id": st.session_state.get('session_id', session_id),
            "thread_id": session_id,
            "user_id": st.session_state.get('user_id', 'default-user'),
            "execution_history": [],
            "plan": [],
            "current_step": 0,
            "step_results": {},
            "task_completed": False
        }
        
        # 🆕 동기 방식으로 직접 그래프 실행
        import time
        start_time = time.time()
        
        try:
            from langchain_core.runnables import RunnableConfig
            
            # 설정 구성
            config = RunnableConfig(
                recursion_limit=st.session_state.get("recursion_limit", 30),
                configurable={"thread_id": session_id}
            )
            
            # 🆕 직접 동기 스트리밍 (타임아웃 체크 포함)
            for chunk in st.session_state.plan_execute_graph.stream(
                initial_state,
                config=config,
                stream_mode="values"
            ):
                # 타임아웃 체크
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    progress_placeholder.warning(f"⏰ 분석이 {timeout_seconds}초 후 타임아웃되었습니다.")
                    response_placeholder.error("⏰ 시간 초과로 분석이 중단되었습니다. 더 간단한 요청을 시도해보세요.")
                    break
                
                # 콜백 실행
                chunk_msg = {"content": chunk, "node": chunk.get("next_action", "unknown")}
                for callback in callbacks:
                    try:
                        if callable(callback):
                            callback(chunk_msg)
                    except Exception as cb_error:
                        logging.error(f"Callback error: {cb_error}")
                
                # 완료 조건 체크
                if chunk.get("task_completed") or chunk.get("next_action") == "final_responder":
                    break
                    
        except Exception as stream_error:
            logging.error(f"Stream error: {stream_error}")
            response_placeholder.error(f"❌ 스트리밍 오류: {str(stream_error)}")
        
        # 최종 응답 저장
        final_response = chat_callback.get_final_response()
        if final_response:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # 백업 응답
            fallback_response = "".join(chat_callback.buffer)
            if fallback_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_response,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "✅ 분석이 완료되었습니다. 위의 결과를 확인해주세요.",
                    "timestamp": datetime.now().isoformat()
                })
        
    except Exception as e:
        logging.error(f"Query processing error: {e}", exc_info=True)
        response_placeholder.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ 죄송합니다. 오류가 발생했습니다: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

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