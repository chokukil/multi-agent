#!/usr/bin/env python3
"""
🎨 CherryAI Main UI Controller

모든 UI 렌더링 로직을 담당하는 컨트롤러
main.py에서 UI 관련 코드를 분리하여 테스트 가능하고 유지보수가 쉬운 구조 구현

Key Features:
- UI 컴포넌트 렌더링
- 사용자 상호작용 처리
- 실시간 업데이트 관리
- 반응형 레이아웃
- 에러 처리 및 사용자 피드백

Architecture:
- Presentation Layer: UI 렌더링
- Interaction Layer: 사용자 입력 처리
- Feedback Layer: 상태 및 피드백 표시
"""

import streamlit as st
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from datetime import datetime
import json

# 프로젝트 임포트
from core.app_components.main_app_controller import (
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

logger = logging.getLogger(__name__)

class CherryAIUIController:
    """
    🎨 CherryAI UI 컨트롤러
    
    모든 UI 렌더링과 사용자 상호작용을 관리
    """
    
    def __init__(self, 
                 app_engine = None,
                 config_manager = None,
                 session_manager = None):
        """
        UI 컨트롤러 초기화
        
        Args:
            app_engine: 비즈니스 로직 엔진
            config_manager: 설정 관리자
            session_manager: 세션 관리자
        """
        self.app_engine = app_engine
        self.config_manager = config_manager
        self.session_manager = session_manager
        
        # UI 상태 관리
        self.ui_state = {
            "last_update": datetime.now(),
            "active_components": set(),
            "error_count": 0,
            "user_interactions": 0
        }
        
        logger.info("🎨 CherryAI UI Controller 초기화 완료")

    def apply_cherry_theme(self):
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
        
        /* 스트리밍 텍스트 애니메이션 */
        .streaming-text {
            background: linear-gradient(90deg, #f0f0f0, #e0e0e0, #f0f0f0);
            background-size: 200% 100%;
            animation: shimmer 2s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        /* 성공 메시지 */
        .success-message {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* 경고 메시지 */
        .warning-message {
            background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* 에러 메시지 */
        .error-message {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self) -> None:
        """헤더 렌더링"""
        st.markdown("""
        <div class="cherry-header">
            <h1 style="color: white; margin: 0;">🍒 CherryAI</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                세계 최초 A2A + MCP 통합 플랫폼 ✨
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        self.ui_state["active_components"].add("header")

    def render_system_status(self, controller) -> SystemStatus:
        """시스템 상태 렌더링"""
        try:
            status = controller.get_system_status()
            
            # 상태에 따른 색상 및 아이콘
            if status.overall_health >= 90:
                status_color = "#28a745"  # 녹색
                status_icon = "🟢"
                status_text = "최적"
            elif status.overall_health >= 70:
                status_color = "#ffc107"  # 노란색  
                status_icon = "🟡"
                status_text = "양호"
            else:
                status_color = "#dc3545"  # 빨간색
                status_icon = "🔴"
                status_text = "주의"
            
            # 상태 표시
            st.markdown(f"""
            <div class="status-indicator" style="border-left-color: {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2em;">{status_icon}</span>
                        <strong>시스템 상태: {status_text}</strong>
                        <span style="color: #666; margin-left: 1rem;">
                            ({status.overall_health:.1f}% | A2A: {status.a2a_agents_count}개 | MCP: {status.mcp_tools_count}개)
                        </span>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">
                        마지막 업데이트: {status.last_check.strftime('%H:%M:%S')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 상세 정보 (확장 가능)
            with st.expander("🔍 상세 시스템 정보"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("A2A 에이전트", status.a2a_agents_count, "활성화")
                    
                with col2:
                    st.metric("MCP 도구", status.mcp_tools_count, "연결됨")
                    
                with col3:
                    st.metric("전체 상태", f"{status.overall_health:.1f}%", 
                             f"+{status.overall_health-80:.1f}%" if status.overall_health > 80 else None)
            
            self.ui_state["active_components"].add("system_status")
            return status
            
        except Exception as e:
            logger.error(f"시스템 상태 렌더링 오류: {e}")
            st.error("⚠️ 시스템 상태를 가져올 수 없습니다.")
            return None

    def render_chat_interface(self, controller) -> Optional[str]:
        """채팅 인터페이스 렌더링"""
        
        # 채팅 히스토리 표시
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # 채팅 컨테이너
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-message">
                        <strong>🤖 CherryAI:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # 사용자 입력
        user_input = st.chat_input("💬 CherryAI에게 질문하거나 작업을 요청하세요...")
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            self.ui_state["user_interactions"] += 1
            
        self.ui_state["active_components"].add("chat_interface")
        return user_input

    def render_file_upload_section(self) -> List[Any]:
        """파일 업로드 섹션 렌더링"""
        st.markdown("### 📁 파일 업로드")
        
        # 지원 파일 형식 안내
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong>📋 지원 형식:</strong> CSV, Excel (.xlsx, .xls), JSON<br>
            <strong>💡 팁:</strong> 데이터 분석을 위한 구조화된 파일을 업로드하세요
        </div>
        """, unsafe_allow_html=True)
        
        # 파일 업로더
        uploaded_files = st.file_uploader(
            "파일 선택 (복수 선택 가능)",
            type=['csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True,
            help="pandas가 처리할 수 있는 데이터 파일을 선택하세요"
        )
        
        # 업로드된 파일 정보 표시
        if uploaded_files:
            st.markdown("#### 📊 업로드된 파일")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # KB
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0;">
                    📄 <strong>{file.name}</strong> ({file_size:.1f} KB)
                </div>
                """, unsafe_allow_html=True)
        
        self.ui_state["active_components"].add("file_upload")
        return uploaded_files or []

    def render_sidebar_status(self, controller) -> None:
        """사이드바 상태 렌더링"""
        
        st.markdown("### 🔧 시스템 제어")
        
        # 실시간 상태 확인 버튼
        if st.button("🔄 상태 새로고침"):
            with st.spinner("시스템 상태 확인 중..."):
                system_overview = sync_system_health_check()
                st.success(f"✅ 상태 업데이트 완료 ({system_overview.overall_health:.1f}%)")
        
        # A2A 에이전트 목록
        st.markdown("#### 🤖 A2A 에이전트")
        agents = [
            "🎭 Orchestrator", "🧹 DataCleaning", "📁 DataLoader",
            "📊 DataVisualization", "🔧 DataWrangling", "🔍 EDA",
            "⚙️ FeatureEngineering", "🤖 H2O_Modeling",
            "📈 MLflow", "🗄️ SQLDatabase", "🐼 Pandas"
        ]
        
        for agent in agents:
            st.markdown(f"- {agent}")
        
        # MCP 도구 목록
        st.markdown("#### 🔧 MCP 도구")
        tools = [
            # Playwright removed for enterprise/intranet compatibility
            "📁 FileManager", "🗄️ Database",
            "🌍 API Gateway", "📈 Analyzer", "📊 Charts", "🤖 LLM"
        ]
        
        for tool in tools:
            st.markdown(f"- {tool}")
        
        # 실시간 메트릭
        if st.checkbox("📊 실시간 메트릭 표시"):
            st.markdown("#### 📈 UI 상태")
            st.json({
                "활성 컴포넌트": len(self.ui_state["active_components"]),
                "사용자 상호작용": self.ui_state["user_interactions"],
                "오류 횟수": self.ui_state["error_count"],
                "마지막 업데이트": self.ui_state["last_update"].strftime('%H:%M:%S')
            })
        
        self.ui_state["active_components"].add("sidebar")

    def display_streaming_response(self, response_text: str, placeholder = None) -> None:
        """실시간 스트리밍 응답 표시 (LLM First)"""
        if placeholder is None:
            placeholder = st.empty()
        
        # 실시간 SSE 스트리밍 - 블로킹 제거
        placeholder.markdown(f"""
        <div class="ai-message">
            <strong>🤖 CherryAI:</strong><br>
            {response_text}
        </div>
        """, unsafe_allow_html=True)

    def show_success_message(self, message: str) -> None:
        """성공 메시지 표시"""
        st.markdown(f"""
        <div class="success-message">
            ✅ {message}
        </div>
        """, unsafe_allow_html=True)

    def show_warning_message(self, message: str) -> None:
        """경고 메시지 표시"""
        st.markdown(f"""
        <div class="warning-message">
            ⚠️ {message}
        </div>
        """, unsafe_allow_html=True)

    def show_error_message(self, message: str) -> None:
        """에러 메시지 표시"""
        self.ui_state["error_count"] += 1
        
        st.markdown(f"""
        <div class="error-message">
            🚨 {message}
        </div>
        """, unsafe_allow_html=True)

    def handle_user_query(self, user_input: str, controller, uploaded_files: List[Any] = None) -> str:
        """사용자 쿼리 처리 및 UI 업데이트"""
        try:
            # 응답 플레이스홀더 생성
            response_placeholder = st.empty()
            response_placeholder.markdown("""
            <div class="streaming-text" style="padding: 1rem; border-radius: 10px;">
                🤖 CherryAI가 생각하고 있습니다...
            </div>
            """, unsafe_allow_html=True)
            
            # 파일 처리 (업로드된 파일이 있는 경우)
            processed_files = []
            if uploaded_files:
                file_processor = get_file_upload_processor()
                processed_files = process_and_prepare_files_for_a2a(uploaded_files)
                
                if processed_files:
                    self.show_success_message(f"📁 {len(processed_files)}개 파일 처리 완료")
            
            # 비동기 처리를 위한 래퍼
            async def create_broker_stream():
                streaming_handler = get_streaming_handler()
                
                # A2A 메시지 브로커로 실시간 스트리밍 처리
                async for chunk in process_query_with_streaming(
                    user_input, 
                    controller, 
                    processed_files
                ):
                    # 실시간 업데이트
                    if chunk.strip():
                        self.display_streaming_response(chunk, response_placeholder)
                        yield chunk
            
            # 실제 스트리밍 처리
            full_response = ""
            for chunk in asyncio.run(create_broker_stream().__anext__()):
                full_response += chunk
            
            # 채팅 히스토리에 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now()
            })
            
            self.ui_state["last_update"] = datetime.now()
            return full_response
            
        except Exception as e:
            error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
            self.show_error_message(error_msg)
            logger.error(f"사용자 쿼리 처리 오류: {e}")
            return error_msg

    def render_layout(self, controller) -> Tuple[str, List[Any]]:
        """전체 레이아웃 렌더링"""
        # 시스템 상태 확인
        if not st.session_state.get('system_initialized', False):
            st.warning("⏳ 시스템 초기화 중... 잠시만 기다려주세요.")
            st.stop()
        
        # 시스템 상태 표시
        self.render_system_status(controller)
        
        # 메인 레이아웃
        main_col, sidebar_col = st.columns([3, 1])
        
        user_input = None
        uploaded_files = []
        
        with main_col:
            # 파일 업로드 섹션
            uploaded_files = self.render_file_upload_section()
            
            # 채팅 인터페이스
            st.markdown("### 💬 AI 협업 채팅")
            user_input = self.render_chat_interface(controller)
        
        with sidebar_col:
            # 사이드바 상태 표시
            self.render_sidebar_status(controller)
        
        return user_input, uploaded_files

    def handle_global_error(self, error: Exception) -> None:
        """전역 에러 처리"""
        logger.error(f"🚨 전역 UI 오류: {error}")
        
        st.error(f"""
        🚨 **시스템 오류가 발생했습니다**
        
        오류: `{str(error)}`
        
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

    def get_ui_metrics(self) -> Dict[str, Any]:
        """UI 메트릭 조회"""
        return {
            "active_components": list(self.ui_state["active_components"]),
            "user_interactions": self.ui_state["user_interactions"],
            "error_count": self.ui_state["error_count"],
            "last_update": self.ui_state["last_update"].isoformat(),
            "chat_history_length": len(st.session_state.get("chat_history", [])),
            "session_state_keys": list(st.session_state.keys())
        }

# 전역 인스턴스 관리
_global_ui_controller: Optional[CherryAIUIController] = None

def get_ui_controller() -> CherryAIUIController:
    """전역 UI 컨트롤러 인스턴스 조회"""
    global _global_ui_controller
    if _global_ui_controller is None:
        _global_ui_controller = CherryAIUIController()
    return _global_ui_controller

def initialize_ui_controller(**kwargs) -> CherryAIUIController:
    """UI 컨트롤러 초기화"""
    global _global_ui_controller
    _global_ui_controller = CherryAIUIController(**kwargs)
    return _global_ui_controller 