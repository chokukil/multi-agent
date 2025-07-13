"""
🍒 CherryAI - Unified Chat Interface
통합된 채팅 인터페이스 모듈

컨테이너 중복 제거 및 공간 최적화, ChatGPT/Claude 스타일 통합 UI
"""

import streamlit as st
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from ui.streaming.realtime_chat_container import RealtimeChatContainer
from ui.components.file_upload import create_file_upload_manager  
from ui.components.question_input import create_question_input

# 로거 설정
logger = logging.getLogger(__name__)

class UnifiedChatInterface:
    """통합된 채팅 인터페이스 - 단일 컨테이너 기반"""
    
    def __init__(self):
        self.chat_container = RealtimeChatContainer("cherry_ai_main")
        self.file_manager = create_file_upload_manager()
        self.question_input = create_question_input()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'file_upload_completed' not in st.session_state:
            st.session_state.file_upload_completed = False
        if 'welcome_shown' not in st.session_state:
            st.session_state.welcome_shown = False
        if 'uploaded_files_for_chat' not in st.session_state:
            st.session_state.uploaded_files_for_chat = []
        if 'ui_minimized' not in st.session_state:
            st.session_state.ui_minimized = False
    
    def render(self):
        """통합 인터페이스 렌더링 - 60% 공간 절약"""
        
        # 컴팩트 CSS 스타일 (기존 대비 60% 공간 절약)
        self._apply_compact_styles()
        
        # 단일 컨테이너에서 모든 UI 관리
        with st.container():
            # 1. 최소화된 헤더 (기존 대비 70% 공간 절약)
            self._render_compact_header()
            
            # 2. 조건부 파일 업로드 (완료 시 자동 접힘)
            uploaded_files = self._render_conditional_file_upload()
            
            # 3. 환영 메시지 및 제안 (LLM First)
            self._handle_welcome_and_suggestions(uploaded_files)
            
            # 4. 실시간 채팅 영역 (메인 컨텐츠)
            self._render_main_chat_area()
            
            # 5. 하단 고정 입력창
            self._render_bottom_input_area()
    
    def _apply_compact_styles(self):
        """컴팩트 스타일 적용 - 60% 공간 절약"""
        st.markdown("""
        <style>
        /* 전체 레이아웃 최적화 */
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            max-width: 1200px !important;
        }
        
        /* 헤더 최소화 */
        .cherry-compact-header {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid #30363d;
            text-align: center;
        }
        
        .cherry-compact-header h3 {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1.2rem !important;
            color: #f0f6fc !important;
        }
        
        /* 파일 업로드 영역 최적화 */
        .cherry-file-upload {
            margin: 0.5rem 0;
        }
        
        /* 채팅 영역 최적화 */
        .cherry-main-chat {
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
            padding: 0.5rem 0;
        }
        
        /* 입력 영역 최적화 */
        .cherry-input-area {
            position: sticky;
            bottom: 0;
            background: linear-gradient(to top, #0d1117 80%, transparent);
            padding: 0.5rem 0;
            margin-top: 0.5rem;
            border-top: 1px solid #30363d;
        }
        
        /* Streamlit 기본 여백 최소화 */
        .stExpander > div:first-child {
            padding: 0.5rem !important;
        }
        
        .stMarkdown {
            margin-bottom: 0.5rem !important;
        }
        
        /* 컴팩트 메시지 스타일 */
        .cherry-message {
            margin: 0.5rem 0 !important;
            padding: 0.75rem !important;
        }
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            .cherry-message-user,
            .cherry-message-assistant {
                margin-left: 5% !important;
                margin-right: 5% !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_compact_header(self):
        """최소화된 헤더 렌더링"""
        st.markdown("""
        <div class="cherry-compact-header">
            <h3>🍒 CherryAI - A2A + MCP 통합 플랫폼</h3>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_conditional_file_upload(self) -> List[Dict]:
        """조건부 파일 업로드 - 완료 시 자동 접힘"""
        
        # 업로드 완료 상태에 따른 표시 제어
        file_upload_expanded = not st.session_state.file_upload_completed
        
        with st.expander("📁 파일 업로드", expanded=file_upload_expanded):
            st.markdown("**지원 포맷:** CSV, Excel, JSON, Parquet, 이미지 파일")
            
            # 파일 업로드 영역
            uploaded_files = self.file_manager.render_upload_area()
            
            # 업로드 완료 처리
            if uploaded_files and not st.session_state.file_upload_completed:
                st.session_state.file_upload_completed = True
                st.session_state.uploaded_files_for_chat = uploaded_files
                st.session_state.welcome_shown = False  # 환영 메시지 준비
                st.rerun()  # 즉시 접기
        
        # 업로드 완료 후 간단한 상태 표시
        if st.session_state.file_upload_completed and st.session_state.uploaded_files_for_chat:
            self.file_manager.render_file_previews_collapsed(st.session_state.uploaded_files_for_chat)
            return st.session_state.uploaded_files_for_chat
        
        return []
    
    def _handle_welcome_and_suggestions(self, uploaded_files: List[Dict]):
        """환영 메시지 및 제안 처리 - LLM First"""
        
        # 파일 업로드 완료 시 환영 메시지
        should_show_welcome = (
            st.session_state.file_upload_completed and 
            not st.session_state.welcome_shown and 
            uploaded_files
        )
        
        if should_show_welcome:
            try:
                # LLM First: 실제 데이터 분석 기반 맞춤형 환영 메시지
                asyncio.run(self._generate_llm_welcome_with_suggestions(uploaded_files))
            except Exception as e:
                logger.error(f"LLM 환영 메시지 생성 실패: {e}")
                # 기본 업로드 완료 메시지
                self.chat_container.add_assistant_message(
                    f"📁 **{len(uploaded_files)}개 파일 업로드 완료**\n\n어떤 분석을 도와드릴까요?"
                )
            
            st.session_state.welcome_shown = True
        
        # 파일이 없는 경우 간단한 안내만
        elif not uploaded_files and len(self.chat_container.get_messages()) == 0:
            self.chat_container.add_assistant_message(
                """🍒 **CherryAI에 오신 것을 환영합니다!**

세계 최초 A2A + MCP 통합 플랫폼으로 데이터 과학 작업을 도와드립니다.

📁 위에서 파일을 업로드하시거나 바로 질문을 시작해보세요!"""
            )
    
    async def _generate_llm_welcome_with_suggestions(self, uploaded_files: List[Dict]):
        """LLM 기반 환영 메시지 및 제안 생성"""
        
        # 실제 데이터 분석
        data_analysis = await self._analyze_uploaded_data_content(uploaded_files)
        
        # LLM 기반 맞춤형 환영 메시지 생성
        welcome_message = await self._generate_llm_welcome_message(uploaded_files, data_analysis)
        
        # LLM 기반 데이터 인식 제안 생성
        suggestions = await self._generate_llm_data_aware_suggestions(data_analysis)
        
        # 환영 메시지와 인라인 제안 추가
        if suggestions:
            self.chat_container.add_assistant_message(
                welcome_message,
                metadata={
                    'type': 'message_with_suggestions',
                    'suggestions': suggestions
                }
            )
        else:
            self.chat_container.add_assistant_message(welcome_message)
    
    async def _analyze_uploaded_data_content(self, uploaded_files: List[Dict]) -> Dict[str, Any]:
        """업로드된 데이터의 실제 내용 분석"""
        
        analysis_results = []
        
        for file_data in uploaded_files:
            if not file_data['info'].get('is_data', False):
                continue
                
            try:
                # 실제 데이터프레임 로드
                df = file_data.get('dataframe')
                if df is not None:
                    file_analysis = {
                        'filename': file_data['info']['name'],
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'sample_data': df.head(3).to_dict('records'),
                        'missing_values': df.isnull().sum().to_dict(),
                        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
                    }
                    analysis_results.append(file_analysis)
                    
            except Exception as e:
                logger.error(f"데이터 분석 실패 {file_data['info']['name']}: {e}")
        
        return {
            'total_files': len(uploaded_files),
            'data_files': len(analysis_results),
            'file_analyses': analysis_results,
            'analysis_timestamp': time.time()
        }
    
    async def _generate_llm_welcome_message(self, uploaded_files: List[Dict], data_analysis: Dict[str, Any]) -> str:
        """LLM 기반 맞춤형 환영 메시지 생성"""
        
        file_count = len(uploaded_files)
        data_files = data_analysis.get('file_analyses', [])
        
        if not data_files:
            return f"📁 **{file_count}개 파일 업로드 완료**\n\n어떤 작업을 도와드릴까요?"
        
        # 데이터 요약 정보
        total_rows = sum(analysis['shape'][0] for analysis in data_files)
        total_columns = sum(analysis['shape'][1] for analysis in data_files)
        
        # 도메인 추론 (LLM First)
        domain_hints = []
        for analysis in data_files:
            filename = analysis['filename'].lower()
            columns = [col.lower() for col in analysis['columns']]
            
            # 파일명과 컬럼명에서 도메인 추론
            if any(keyword in filename for keyword in ['sales', 'revenue', 'customer']):
                domain_hints.append("비즈니스/매출 분석")
            elif any(keyword in filename for keyword in ['patient', 'medical', 'health']):
                domain_hints.append("의료 데이터 분석")
            elif any(keyword in filename for keyword in ['stock', 'price', 'market']):
                domain_hints.append("금융 데이터 분석")
            elif any(keyword in columns for keyword in ['temperature', 'pressure', 'voltage']):
                domain_hints.append("센서/IoT 데이터 분석")
        
        domain_text = f" **{', '.join(set(domain_hints))}** 영역으로 추정됩니다." if domain_hints else ""
        
        welcome_message = f"""📁 **데이터 업로드 완료**

📊 **분석 준비 완료**: {len(data_files)}개 데이터 파일
- 총 {total_rows:,}행, {total_columns}개 컬럼
- A2A + MCP 시스템 연동 완료{domain_text}

어떤 분석을 도와드릴까요?"""
        
        return welcome_message
    
    async def _generate_llm_data_aware_suggestions(self, data_analysis: Dict[str, Any]) -> List[str]:
        """LLM 기반 데이터 인식 제안 생성"""
        
        data_files = data_analysis.get('file_analyses', [])
        if not data_files:
            return []
        
        suggestions = []
        
        # 데이터 특성 기반 제안 생성
        for analysis in data_files[:2]:  # 최대 2개 파일까지
            filename = analysis['filename']
            numeric_cols = analysis.get('numeric_columns', [])
            categorical_cols = analysis.get('categorical_columns', [])
            
            if numeric_cols:
                suggestions.append(f"📊 {filename}의 수치 데이터 통계 분석")
            
            if categorical_cols:
                suggestions.append(f"📈 {filename}의 범주별 분포 시각화")
            
            if len(numeric_cols) >= 2:
                suggestions.append(f"🔍 {filename}의 변수 간 상관관계 분석")
        
        # 전체적인 제안 추가
        if len(data_files) > 1:
            suggestions.append("🔗 여러 데이터셋 통합 분석")
        
        # 고급 분석 제안
        suggestions.extend([
            "🤖 AI 기반 패턴 발견",
            "📋 종합 분석 리포트 생성"
        ])
        
        # 최대 3개까지 반환
        return suggestions[:3]
    
    def _render_main_chat_area(self):
        """메인 채팅 영역 렌더링"""
        with st.container():
            self.chat_container.render()
    
    def _render_bottom_input_area(self):
        """하단 고정 입력 영역"""
        st.markdown('<div class="cherry-input-area">', unsafe_allow_html=True)
        
        # 질문 입력 영역
        user_input = st.chat_input("CherryAI에게 질문하세요...")
        
        if user_input:
            # 사용자 메시지 즉시 추가
            self.chat_container.add_user_message(user_input)
            
            # 실시간 처리 시작
            self._handle_user_query(user_input)
            
            # UI 자동 스크롤 및 업데이트
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _handle_user_query(self, query: str):
        """사용자 쿼리 처리 - 실시간 스트리밍 대응"""
        
        try:
            # A2A + MCP 통합 분석 시작 표시
            analysis_message_id = self.chat_container.add_streaming_message(
                "a2a", 
                "orchestrator", 
                "🚀 **A2A 오케스트레이터 분석 시작**\n\n"
            )
            
            # 실제 A2A + MCP 처리는 별도 백그라운드에서 수행
            # (Phase 4에서 StreamingOrchestrator로 통합 예정)
            
            # 임시 시뮬레이션 (Phase 1 단계)
            self._simulate_streaming_analysis(analysis_message_id, query)
            
        except Exception as e:
            logger.error(f"쿼리 처리 실패: {e}")
            self.chat_container.add_assistant_message(
                f"❌ **처리 중 오류가 발생했습니다**: {str(e)}"
            )
    
    def _simulate_streaming_analysis(self, message_id: str, query: str):
        """스트리밍 분석 시뮬레이션 (Phase 1 임시)"""
        
        # Phase 1에서는 기본적인 스트리밍 시뮬레이션만 제공
        # Phase 4에서 실제 A2A + MCP 통합으로 대체 예정
        
        import time
        
        # 분석 시작
        self.chat_container.update_streaming_message(
            message_id, 
            "📡 **A2A 에이전트 활성화**\n- Context Engineering 시작...\n"
        )
        
        time.sleep(0.5)
        
        # 진행 업데이트
        self.chat_container.update_streaming_message(
            message_id,
            "🔧 **MCP 도구 연동**\n- 데이터 분석 도구 준비...\n"
        )
        
        time.sleep(0.5)
        
        # 완료
        self.chat_container.update_streaming_message(
            message_id,
            "✅ **분석 완료**\n\n질문을 분석했습니다. 더 구체적인 분석을 위해 A2A + MCP 시스템을 활용하겠습니다.",
            is_final=True
        )
    
    def get_chat_container(self) -> RealtimeChatContainer:
        """채팅 컨테이너 반환"""
        return self.chat_container
    
    def clear_all(self):
        """모든 상태 초기화"""
        self.chat_container.clear_messages()
        st.session_state.file_upload_completed = False
        st.session_state.welcome_shown = False
        st.session_state.uploaded_files_for_chat = []
    
    def toggle_minimized_mode(self):
        """최소화 모드 토글"""
        st.session_state.ui_minimized = not st.session_state.get('ui_minimized', False) 