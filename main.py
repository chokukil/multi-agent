#!/usr/bin/env python3
"""
🍒 CherryAI - 세계 최초 A2A + MCP 통합 플랫폼

실제 동작하는 하이브리드 AI 협업 시스템
- 11개 A2A 에이전트 + 7개 MCP 도구
- 실시간 스트리밍 처리
- ChatGPT/Claude 스타일 UI/UX
- 완전 모듈화된 아키텍처
"""

import streamlit as st
import asyncio
import uuid
from typing import Dict, Any, List, Optional
import logging

# 핵심 모듈들 임포트
from core.app_components.main_app_controller import (
    initialize_app_controller, 
    get_app_controller,
    SystemStatus
)
from core.app_components.realtime_streaming_handler import (
    get_streaming_handler,
    process_query_with_streaming
)
from core.app_components.file_upload_processor import (
    get_file_upload_processor,
    process_and_prepare_files_for_a2a
)
from core.app_components.system_status_monitor import (
    get_system_status_monitor,
    sync_system_health_check
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_cherry_theme():
    """CherryAI 테마 적용"""
    st.markdown("""
    <style>
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* 헤더 컨테이너 */
    .cherry-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* 상태 표시기 */
    .status-indicator {
        background: rgba(0,255,0,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00ff00;
    }
    
    /* 채팅 메시지 - 사용자 */
    .user-message {
        background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* 채팅 메시지 - AI */
    .ai-message {
        background: linear-gradient(135deg, #da3633 0%, #a21e1e 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* 파일 업로드 영역 */
    .file-upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* 시스템 통계 */
    .system-stats {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """헤더 렌더링"""
    st.markdown("""
    <div class="cherry-header">
        <h1 style="color: white; margin-bottom: 0.5rem;">🍒 CherryAI</h1>
        <h3 style="color: white; margin-bottom: 0.5rem;">세계 최초 A2A + MCP 통합 플랫폼</h3>
        <p style="color: white; opacity: 0.9; margin: 0;">
            🌟 11개 A2A 에이전트 + 7개 MCP 도구 | 실시간 스트리밍 | LLM First 아키텍처
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_system_status(controller):
    """시스템 상태 표시"""
    system_health = controller.system_health
    stats = controller.get_system_stats()
    
    # 상태에 따른 색상 결정
    if system_health.status == SystemStatus.READY:
        status_color = "#00ff00"
        status_text = "🟢 시스템 준비 완료"
    elif system_health.status == SystemStatus.RUNNING:
        status_color = "#00ff00"
        status_text = "🟢 시스템 실행 중"
    elif system_health.status == SystemStatus.ERROR:
        status_color = "#ff0000"
        status_text = "🔴 시스템 오류"
    else:
        status_color = "#ffa500"
        status_text = "🟡 시스템 초기화 중"
    
    st.markdown(f"""
    <div class="status-indicator" style="border-left-color: {status_color};">
        <strong>{status_text}</strong> | 
        🤖 A2A 에이전트: {stats['a2a_agents']} | 
        🔧 MCP 도구: {stats['mcp_tools']} |
        📊 브로커: {stats['broker_status']} |
        🎭 오케스트레이터: {stats['orchestrator_status']}
    </div>
    """, unsafe_allow_html=True)

def handle_user_query(user_input: str, controller, uploaded_files: List[Any] = None) -> str:
    """사용자 쿼리 처리 - 실시간 스트리밍 포함"""
    
    try:
        # 1. 사용자 메시지 추가
        controller.add_message("user", user_input)
        
        # 2. 파일 처리 (있는 경우)
        file_context = None
        if uploaded_files:
            file_processor = get_file_upload_processor()
            processed_files = file_processor.process_uploaded_files(uploaded_files)
            
            if processed_files:
                file_ids = [pf.metadata.file_id for pf in processed_files if pf.a2a_ready]
                file_context = file_processor.prepare_for_a2a_system(file_ids)
                
                # 파일 정보를 쿼리에 추가
                file_info = f"\n\n[업로드된 파일: {len(processed_files)}개 - {', '.join([pf.metadata.filename for pf in processed_files])}]"
                user_input += file_info
        
        # 3. A2A 시스템으로 쿼리 전송
        if not controller.unified_broker:
            return "❌ 시스템이 초기화되지 않았습니다. 페이지를 새로고침해주세요."
        
        # 브로커 세션 생성
        session = controller.get_current_session()
        if not session.broker_session_id:
            # 비동기 세션 생성을 동기적으로 처리
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        controller.unified_broker.create_session(user_input, session.session_id)
                    )
                    session.broker_session_id = future.result(timeout=10)
            except RuntimeError:
                session.broker_session_id = asyncio.run(
                    controller.unified_broker.create_session(user_input, session.session_id)
                )
        
        # 4. 실시간 스트리밍으로 응답 생성
        streaming_handler = get_streaming_handler()
        
        # 브로커에서 스트리밍 생성기 생성
        async def create_broker_stream():
            async for event in controller.unified_broker.orchestrate_multi_agent_query(
                session.broker_session_id,
                user_input,
                capabilities=None
            ):
                yield event
        
        # 스트리밍 처리 (UI 컨테이너는 None으로 설정, 별도 처리)
        response = process_query_with_streaming(
            user_input,
            create_broker_stream(),
            ui_container=None
        )
        
        # 5. 응답 메시지 추가
        controller.add_message("assistant", response)
        controller.stats['successful_queries'] += 1
        
        return response
        
    except Exception as e:
        error_message = controller.handle_error(e, "사용자 쿼리 처리")
        controller.add_message("assistant", error_message)
        return error_message

def render_chat_interface(controller):
    """채팅 인터페이스 렌더링"""
    
    # 현재 세션의 메시지들 표시
    session = controller.get_current_session()
    
    # 메시지 히스토리 표시
    if session.messages:
        for message in session.messages[-10:]:  # 최근 10개만 표시
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 사용자:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-message">
                    <strong>🍒 CherryAI:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
    
    # 채팅 입력
    user_input = st.chat_input("CherryAI에게 질문하세요... (A2A + MCP 하이브리드 시스템)")
    
    return user_input

def render_file_upload_section():
    """파일 업로드 섹션 렌더링"""
    
    with st.expander("📁 파일 업로드", expanded=False):
        uploaded_files = st.file_uploader(
            "CSV, Excel, JSON 파일을 업로드하세요",
            type=['csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True,
            help="업로드된 파일은 자동으로 A2A 시스템에 전달되어 분석됩니다"
        )
        
        if uploaded_files:
            file_processor = get_file_upload_processor()
            
            # 파일 처리 및 미리보기
            processed_files = file_processor.process_uploaded_files(uploaded_files)
            
            if processed_files:
                st.success(f"✅ {len(processed_files)}개 파일 처리 완료")
                
                # 간단한 파일 정보 표시
                for pf in processed_files:
                    if pf.a2a_ready:
                        st.info(f"📄 {pf.metadata.filename} ({pf.metadata.rows:,}행, {pf.metadata.columns}열)")
                    else:
                        st.error(f"❌ {pf.metadata.filename} 처리 실패: {pf.metadata.error_message}")
        
        return uploaded_files

def render_sidebar_status(controller):
    """사이드바 상태 표시"""
    
    with st.sidebar:
        st.markdown("### 📊 시스템 현황")
        
        # 시스템 통계
        stats = controller.get_system_stats()
        
        st.metric("시스템 상태", stats['system_status'])
        st.metric("세션 ID", stats['session_id'])
        st.metric("총 메시지", stats['total_messages'])
        st.metric("성공률", f"{stats['success_rate']}%")
        
        # 실시간 상태 확인 버튼
        if st.button("🔄 상태 새로고침"):
            with st.spinner("시스템 상태 확인 중..."):
                system_overview = sync_system_health_check()
                st.success(f"✅ 상태 업데이트 완료 ({system_overview.overall_health:.1f}%)")
        
        # 간단한 에이전트 목록
        st.markdown("#### 🤖 A2A 에이전트")
        agents = [
            "🎭 Orchestrator", "🧹 DataCleaning", "📁 DataLoader",
            "📊 DataVisualization", "🔧 DataWrangling", "🔍 EDA",
            "⚙️ FeatureEngineering", "🤖 H2O_Modeling",
            "📈 MLflow", "🗄️ SQLDatabase", "🐼 Pandas"
        ]
        for agent in agents:
            st.markdown(f"- {agent}")
        
        st.markdown("#### 🔧 MCP 도구")
        tools = [
            "🌐 Playwright", "📁 FileManager", "🗄️ Database",
            "🌍 API Gateway", "📈 Analyzer", "📊 Charts", "🤖 LLM"
        ]
        for tool in tools:
            st.markdown(f"- {tool}")

def main():
    """메인 애플리케이션"""
    
    try:
        # 페이지 설정
        st.set_page_config(
            page_title="🍒 CherryAI",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 테마 적용
        apply_cherry_theme()
        
        # 헤더 렌더링
        render_header()
        
        # 앱 컨트롤러 초기화
        controller = initialize_app_controller()
        
        # 시스템 상태가 초기화되지 않은 경우
        if not st.session_state.get('system_initialized', False):
            st.warning("⏳ 시스템 초기화 중... 잠시만 기다려주세요.")
            st.stop()
        
        # 시스템 상태 표시
        render_system_status(controller)
        
        # 메인 레이아웃
        main_col, sidebar_col = st.columns([3, 1])
        
        with main_col:
            # 파일 업로드 섹션
            uploaded_files = render_file_upload_section()
            
            # 채팅 인터페이스
            st.markdown("### 💬 AI 협업 채팅")
            user_input = render_chat_interface(controller)
            
            # 사용자 입력 처리
            if user_input:
                with st.spinner("🚀 A2A + MCP 하이브리드 시스템이 처리 중..."):
                    # 실시간 처리 시작
                    response = handle_user_query(user_input, controller, uploaded_files)
                
                # 처리 완료 후 UI 새로고침
                st.rerun()
        
        with sidebar_col:
            # 사이드바 상태 표시
            render_sidebar_status(controller)
    
    except Exception as e:
        # 전역 에러 처리
        logger.error(f"🚨 메인 애플리케이션 오류: {e}")
        
        st.error(f"""
        🚨 **시스템 오류가 발생했습니다**
        
        오류: `{str(e)}`
        
        **해결 방법:**
        1. 페이지 새로고침 (F5)
        2. A2A 서버 상태 확인
        3. 시스템 재시작
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 페이지 새로고침", type="primary"):
                st.rerun()
        with col2:
            if st.button("🔧 시스템 재시작", type="secondary"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    # CherryAI 시스템 시작
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        🍒 <strong>CherryAI</strong> - 세계 최초 A2A + MCP 통합 플랫폼<br>
        실시간 스트리밍 | 하이브리드 아키텍처 | LLM First 철학 ✅
    </div>
    """, unsafe_allow_html=True)
    
    main()
