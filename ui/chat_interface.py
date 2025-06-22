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
        if isinstance(resp, dict) and "error" in resp:
            # 오류 타입에 따른 메시지 분류
            error_msg = resp.get("error", "An unknown error occurred.")
            
            if "timeout" in error_msg.lower():
                assistant_response = {
                    "role": "assistant",
                    "content": f"""## ⏰ 처리 시간 초과

요청 처리가 제한 시간({st.session_state.get("timeout_seconds", 180)}초)을 초과했습니다.

### 💡 해결 방법:
- 더 간단한 분석으로 나누어 요청해 보세요
- 사이드바에서 timeout 설정을 늘려보세요
- 데이터 크기가 큰 경우 샘플링을 고려해 보세요

### 🔧 시스템 상태:
- MCP 서버: {'✅ 연결됨' if st.session_state.get('mcp_client') else '❌ 연결 안됨'}
- Plan-Execute: {'✅ 초기화됨' if st.session_state.get('plan_execute_graph') else '❌ 초기화 안됨'}
""",
                    "error": True,
                    "error_type": "timeout"
                }
            elif "mcp" in error_msg.lower():
                assistant_response = {
                    "role": "assistant", 
                    "content": f"""## 🔧 MCP 도구 연결 오류

MCP (Model Context Protocol) 도구 연결에 문제가 발생했습니다.

### 💡 해결 방법:
1. MCP 서버가 실행 중인지 확인
2. `system_start.bat`를 다시 실행
3. 기본 도구만으로 분석 진행 가능

### 📊 현재 사용 가능한 기능:
- ✅ 기본 데이터 분석 (Python)
- ✅ 시각화 (matplotlib, plotly)
- ✅ 통계 분석 (pandas, numpy)
- ❌ 고급 MCP 도구들

오류 상세: {error_msg}
""",
                    "error": True,
                    "error_type": "mcp"
                }
            else:
                assistant_response = {
                    "role": "assistant",
                    "content": f"""## ❌ 처리 중 오류 발생

분석 처리 중 예상치 못한 오류가 발생했습니다.

### 🔍 오류 정보:
```
{error_msg}
```

### 💡 해결 방법:
1. 요청을 다시 작성해 보세요
2. 더 구체적인 질문으로 나누어 요청해 보세요
3. 데이터가 올바르게 업로드되었는지 확인해 보세요

시스템이 여전히 작동 중이므로 다른 요청을 시도할 수 있습니다.
""",
                    "error": True,
                    "error_type": "general"
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
                
                # 🆕 Final Responder 응답 처리 강화
                if node == "final_responder":
                    if hasattr(content, "content"):
                        final_response = content.content
                    elif isinstance(content, str):
                        final_response = content
                    else:
                        final_response = str(content)
                    
                    final_text_parts.append(final_response)
                    text_placeholder.markdown(final_response)
                    logging.info(f"✅ Final response displayed: {len(final_response)} characters")
            
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
            
            # 최종 텍스트 추출 - 개선된 로직
            final_text = "".join(final_text_parts)
            
            # --- 🛟 UI 최종 안전망 ---
            # 만약 모든 과정이 끝났는데도 final_text가 비어있다면, 
            # UI 단에서 최소한의 응답을 보장합니다.
            if not final_text.strip():
                logging.warning("No final text was generated from the graph. Displaying UI fallback message.")
                final_text = """
### ✅ 분석 프로세스 완료

시스템이 모든 분석 단계를 완료했습니다. 
하지만 최종 요약 보고서를 생성하는 데 문제가 발생한 것으로 보입니다.

**생성된 결과는 우측의 '아티팩트' 패널 또는 위의 '실행 과정'에서 직접 확인하실 수 있습니다.**

만약 결과가 만족스럽지 않다면, 질문을 조금 더 구체적으로 변경하여 다시 시도해 보세요.
"""
                text_placeholder.markdown(final_text)

            tool_activity_content = tool_activity_placeholder.markdown_content if hasattr(tool_activity_placeholder, 'markdown_content') else ""
            
            duration = time.time() - start_time
            logging.info(f"Query processed in {duration:.2f}s")
            
            return response, final_text, tool_activity_content
            
        else:
            error_msg = "🚫 Plan-Execute system not initialized"
            return {"error": error_msg}, error_msg, ""
            
    except asyncio.TimeoutError:
        error_msg = f"⏰ Query timed out after {timeout_seconds} seconds"
        logging.error(error_msg)
        return {"error": error_msg}, error_msg, ""
        
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        logging.error(f"Query processing error: {e}", exc_info=True)
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