#!/usr/bin/env python3
"""
🍒 CherryAI - 모듈화된 메인 애플리케이션

세계 최초 A2A + MCP 통합 플랫폼 - 완전 모듈화 버전
- 11개 A2A 에이전트 + 7개 MCP 도구
- 실시간 스트리밍 처리
- 클린 아키텍처 적용
- LLM First 원칙 준수
- 의존성 주입 패턴

Architecture:
- main_modular.py: 진입점 (50라인 이하)
- ui/main_ui_controller.py: UI 컨트롤러
- core/main_app_engine.py: 비즈니스 엔진
- core/shared_knowledge_bank.py: 지식 뱅크
- core/llm_first_engine.py: LLM First 엔진
"""

import streamlit as st
import asyncio
import logging
from typing import Optional

# 모듈화된 컴포넌트들
from ui.main_ui_controller import (
    CherryAIUIController,
    get_ui_controller,
    initialize_ui_controller
)
from core.main_app_engine import (
    CherryAIMainEngine,
    get_main_engine,
    initialize_and_start_engine
)
from core.shared_knowledge_bank import (
    initialize_shared_knowledge_bank
)
from core.llm_first_engine import (
    initialize_llm_first_engine
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CherryAIApplication:
    """
    🍒 CherryAI 애플리케이션 
    
    모듈화된 컴포넌트들을 조합한 메인 애플리케이션
    """
    
    def __init__(self):
        self.ui_controller: Optional[CherryAIUIController] = None
        self.app_engine: Optional[CherryAIMainEngine] = None
        self.initialized = False

    async def initialize(self) -> bool:
        """애플리케이션 초기화"""
        try:
            logger.info("🚀 CherryAI 애플리케이션 초기화 시작")
            
            # 1. 핵심 엔진들 초기화
            initialize_shared_knowledge_bank(
                persist_directory="./chroma_knowledge_bank",
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            initialize_llm_first_engine(
                enable_learning=True
            )
            
            # 2. 메인 엔진 초기화
            self.app_engine = await initialize_and_start_engine()
            
            # 3. UI 컨트롤러 초기화
            self.ui_controller = initialize_ui_controller(
                app_engine=self.app_engine,
                config_manager=None,  # 추후 구현
                session_manager=None  # 추후 구현
            )
            
            self.initialized = True
            logger.info("✅ CherryAI 애플리케이션 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 애플리케이션 초기화 실패: {e}")
            return False

    def run(self):
        """애플리케이션 실행"""
        try:
            # Streamlit 페이지 설정
            st.set_page_config(
                page_title="🍒 CherryAI",
                page_icon="🍒",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # 초기화 확인
            if not self.initialized:
                if not st.session_state.get('initialization_attempted', False):
                    st.session_state.initialization_attempted = True
                    success = asyncio.run(self.initialize())
                    if success:
                        st.session_state.system_initialized = True
                    else:
                        st.error("❌ 시스템 초기화에 실패했습니다.")
                        return
                else:
                    if not st.session_state.get('system_initialized', False):
                        st.warning("⏳ 시스템 초기화 중... 잠시만 기다려주세요.")
                        st.stop()
            
            # UI 컨트롤러가 없으면 기본 초기화
            if self.ui_controller is None:
                self.ui_controller = get_ui_controller()
            
            # 앱 엔진이 없으면 기본 초기화  
            if self.app_engine is None:
                self.app_engine = get_main_engine()
            
            # 테마 적용
            self.ui_controller.apply_cherry_theme()
            
            # 헤더 렌더링
            self.ui_controller.render_header()
            
            # 메인 레이아웃 렌더링
            user_input, uploaded_files = self.ui_controller.render_layout(self.app_engine)
            
            # 사용자 입력 처리
            if user_input:
                with st.spinner("🚀 A2A + MCP 하이브리드 시스템이 처리 중..."):
                    # 실시간 스트리밍 처리
                    response = self.ui_controller.handle_user_query(
                        user_input, 
                        self.app_engine, 
                        uploaded_files
                    )
                
                # UI 새로고침
                st.rerun()
            
            # 푸터
            self._render_footer()
            
        except Exception as e:
            # 전역 에러 처리
            logger.error(f"🚨 애플리케이션 실행 오류: {e}")
            
            if self.ui_controller:
                self.ui_controller.handle_global_error(e)
            else:
                st.error(f"""
                🚨 **심각한 시스템 오류가 발생했습니다**
                
                오류: `{str(e)}`
                
                시스템을 재시작해주세요.
                """)

    def _render_footer(self):
        """푸터 렌더링"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
            🍒 <strong>CherryAI</strong> - 세계 최초 A2A + MCP 통합 플랫폼<br>
            실시간 스트리밍 | 하이브리드 아키텍처 | LLM First 철학 | 모듈화 설계 ✅
        </div>
        """, unsafe_allow_html=True)

def main():
    """메인 함수 - 애플리케이션 진입점"""
    app = CherryAIApplication()
    app.run()

if __name__ == "__main__":
    main() 