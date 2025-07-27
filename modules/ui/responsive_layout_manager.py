"""
Responsive Layout Manager - 종합적인 반응형 디자인

모바일 우선 설계:
- 점진적 향상 및 터치 친화적 컨트롤
- 접근성 기능: 키보드 탐색, 스크린 리더 호환성, 고대비 모드
- 다양한 화면 크기에 대한 적응형 레이아웃
- 최적화된 모바일 파일 업로드 경험
- 점진적 로딩, 오프라인 기능 표시기, 대역폭 적응형 콘텐츠 전달
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, Callable
from ..models import UIContext, ScreenSize

logger = logging.getLogger(__name__)


class ResponsiveLayoutManager:
    """종합적인 반응형 레이아웃 관리자"""
    
    def __init__(self):
        """Responsive Layout Manager 초기화"""
        self.breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1200,
            'large': 1440
        }
        
        self.ui_context = self._detect_ui_context()
        self._inject_responsive_styles()
        
        logger.info("Responsive Layout Manager initialized")
    
    def _inject_responsive_styles(self):
        """반응형 CSS 스타일 주입"""
        st.markdown("""
        <style>
        /* Cherry AI Responsive Design System */
        
        /* Base responsive container */
        .responsive-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* Mobile-first breakpoints */
        @media (max-width: 768px) {
            .responsive-container {
                padding: 0 0.5rem;
            }
            
            .mobile-stack {
                flex-direction: column !important;
            }
            
            .mobile-full-width {
                width: 100% !important;
            }
            
            .mobile-hide {
                display: none !important;
            }
        }
        
        @media (min-width: 769px) and (max-width: 1024px) {
            .tablet-stack {
                flex-direction: column !important;
            }
        }
        
        /* Touch-friendly controls */
        @media (hover: none) and (pointer: coarse) {
            button, .stButton > button {
                min-height: 44px !important;
                min-width: 44px !important;
                padding: 0.75rem 1rem !important;
                font-size: 16px !important;
            }
            
            .stSelectbox > div > div {
                min-height: 44px !important;
            }
            
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                font-size: 16px !important;
                padding: 0.75rem !important;
            }
        }
        
        /* Accessibility improvements */
        .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        
        /* Focus indicators */
        button:focus,
        input:focus,
        textarea:focus,
        select:focus,
        .stButton > button:focus {
            outline: 2px solid #007bff !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.25) !important;
        }
        
        /* High contrast mode */
        @media (prefers-contrast: high) {
            .cherry-ai-container,
            .header-section,
            .chat-section,
            .input-section {
                border: 2px solid currentColor !important;
                background: Canvas !important;
                color: CanvasText !important;
            }
            
            button, .stButton > button {
                border: 2px solid currentColor !important;
                background: ButtonFace !important;
                color: ButtonText !important;
            }
        }
        
        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .responsive-container {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        }
        
        /* Print styles */
        @media print {
            .no-print {
                display: none !important;
            }
            
            .print-break {
                page-break-after: always;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def adapt_layout_to_screen(self, screen_size: ScreenSize) -> None:
        """화면 크기에 따른 레이아웃 적응"""
        try:
            if screen_size == ScreenSize.MOBILE:
                self._apply_mobile_layout()
            elif screen_size == ScreenSize.TABLET:
                self._apply_tablet_layout()
            elif screen_size == ScreenSize.DESKTOP:
                self._apply_desktop_layout()
            else:
                self._apply_large_layout()
                
        except Exception as e:
            logger.error(f"Layout adaptation error: {str(e)}")
    
    def _apply_mobile_layout(self):
        """모바일 레이아웃 적용"""
        st.markdown("""
        <style>
        .main .block-container {
            padding: 0.5rem !important;
            max-width: 100% !important;
        }
        
        .stColumns {
            flex-direction: column !important;
        }
        
        .stColumn {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
        
        .chat-section {
            min-height: 50vh !important;
            max-height: 60vh !important;
        }
        
        .file-upload-area {
            padding: 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_tablet_layout(self):
        """태블릿 레이아웃 적용"""
        st.markdown("""
        <style>
        .main .block-container {
            padding: 1rem !important;
            max-width: 95% !important;
        }
        
        .chat-section {
            min-height: 55vh !important;
            max-height: 65vh !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_desktop_layout(self):
        """데스크톱 레이아웃 적용"""
        st.markdown("""
        <style>
        .main .block-container {
            padding: 1rem 2rem !important;
            max-width: 1200px !important;
        }
        
        .chat-section {
            min-height: 60vh !important;
            max-height: 70vh !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_large_layout(self):
        """대형 화면 레이아웃 적용"""
        st.markdown("""
        <style>
        .main .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px !important;
        }
        
        .chat-section {
            min-height: 65vh !important;
            max-height: 75vh !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _detect_ui_context(self) -> UIContext:
        """UI 컨텍스트 감지"""
        # 실제 구현에서는 JavaScript를 통해 화면 크기 감지
        # 현재는 기본값 사용
        return UIContext(
            screen_size=ScreenSize.DESKTOP,
            device_type="desktop",
            viewport_width=1200,
            viewport_height=800,
            is_mobile=False,
            theme_preference="auto"
        )
    
    def render_accessibility_controls(self):
        """접근성 컨트롤 렌더링"""
        with st.sidebar:
            st.markdown("### ♿ 접근성 설정")
            
            # 폰트 크기 조정
            font_size = st.selectbox(
                "폰트 크기",
                ["작게", "보통", "크게", "매우 크게"],
                index=1,
                key="font_size_setting"
            )
            
            # 고대비 모드
            high_contrast = st.checkbox(
                "고대비 모드",
                key="high_contrast_mode"
            )
            
            # 애니메이션 감소
            reduce_motion = st.checkbox(
                "애니메이션 감소",
                key="reduce_motion"
            )
            
            # 키보드 탐색 도움말
            if st.button("키보드 단축키 보기"):
                self._show_keyboard_shortcuts()
            
            # 설정 적용
            self._apply_accessibility_settings(
                font_size, high_contrast, reduce_motion
            )
    
    def _apply_accessibility_settings(self, 
                                    font_size: str,
                                    high_contrast: bool,
                                    reduce_motion: bool):
        """접근성 설정 적용"""
        
        # 폰트 크기 매핑
        font_sizes = {
            "작게": "0.9rem",
            "보통": "1rem",
            "크게": "1.2rem",
            "매우 크게": "1.4rem"
        }
        
        selected_font_size = font_sizes.get(font_size, "1rem")
        
        # 동적 스타일 적용
        accessibility_styles = f"""
        <style>
        .accessibility-font {{
            font-size: {selected_font_size} !important;
        }}
        
        {'/* High contrast mode */' if high_contrast else ''}
        {'''
        .high-contrast {
            background: #000000 !important;
            color: #ffffff !important;
        }
        
        .high-contrast button {
            background: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #ffffff !important;
        }
        ''' if high_contrast else ''}
        
        {'/* Reduced motion */' if reduce_motion else ''}
        {'''
        .reduce-motion * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        ''' if reduce_motion else ''}
        </style>
        """
        
        st.markdown(accessibility_styles, unsafe_allow_html=True)
    
    def _show_keyboard_shortcuts(self):
        """키보드 단축키 표시"""
        st.info("""
        **키보드 단축키:**
        
        - **Tab**: 다음 요소로 이동
        - **Shift + Tab**: 이전 요소로 이동
        - **Enter**: 버튼 클릭 또는 선택
        - **Space**: 체크박스 토글
        - **Esc**: 모달 닫기
        - **Ctrl + /**: 이 도움말 표시
        """)
    
    def render_progressive_loading(self, content_loader: Callable):
        """점진적 로딩 렌더링"""
        try:
            # 로딩 상태 표시
            loading_placeholder = st.empty()
            
            with loading_placeholder:
                st.markdown("""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 2rem;
                    color: #666;
                ">
                    <div style="
                        width: 40px;
                        height: 40px;
                        border: 4px solid #f3f3f3;
                        border-top: 4px solid #667eea;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-right: 1rem;
                    "></div>
                    <span>콘텐츠를 로딩 중...</span>
                </div>
                
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
            
            # 콘텐츠 로드
            content = content_loader()
            
            # 로딩 완료 후 콘텐츠 표시
            loading_placeholder.empty()
            return content
            
        except Exception as e:
            logger.error(f"Progressive loading error: {str(e)}")
            st.error("콘텐츠 로딩 중 오류가 발생했습니다.")
    
    def render_offline_indicator(self):
        """오프라인 상태 표시기"""
        # JavaScript를 통한 네트워크 상태 감지 (실제 구현 필요)
        st.markdown("""
        <div id="offline-indicator" style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #f44336;
            color: white;
            text-align: center;
            padding: 0.5rem;
            z-index: 9999;
            display: none;
        ">
            ⚠️ 인터넷 연결이 끊어졌습니다. 일부 기능이 제한될 수 있습니다.
        </div>
        
        <script>
        window.addEventListener('online', function() {
            document.getElementById('offline-indicator').style.display = 'none';
        });
        
        window.addEventListener('offline', function() {
            document.getElementById('offline-indicator').style.display = 'block';
        });
        </script>
        """, unsafe_allow_html=True)
    
    def create_responsive_columns(self, 
                                 desktop_ratios: list,
                                 tablet_ratios: list = None,
                                 mobile_stack: bool = True):
        """반응형 컬럼 생성"""
        
        # 기본값 설정
        if tablet_ratios is None:
            tablet_ratios = desktop_ratios
        
        # 화면 크기에 따른 컬럼 설정
        if self.ui_context.screen_size == ScreenSize.MOBILE and mobile_stack:
            # 모바일에서는 세로 스택
            columns = []
            for _ in desktop_ratios:
                columns.append(st.container())
            return columns
        elif self.ui_context.screen_size == ScreenSize.TABLET:
            return st.columns(tablet_ratios)
        else:
            return st.columns(desktop_ratios)
    
    def render_bandwidth_adaptive_content(self, 
                                        high_quality_content: Callable,
                                        low_quality_content: Callable):
        """대역폭 적응형 콘텐츠 렌더링"""
        
        # 사용자 선택 옵션 제공
        quality_preference = st.radio(
            "콘텐츠 품질 설정",
            ["자동", "고품질", "저품질"],
            index=0,
            horizontal=True,
            help="네트워크 상황에 따라 콘텐츠 품질을 조정합니다."
        )
        
        if quality_preference == "고품질":
            return high_quality_content()
        elif quality_preference == "저품질":
            return low_quality_content()
        else:
            # 자동 모드 - 간단한 휴리스틱 사용
            # 실제 구현에서는 네트워크 속도 감지 필요
            try:
                return high_quality_content()
            except Exception:
                st.warning("네트워크 상태가 좋지 않아 저품질 모드로 전환합니다.")
                return low_quality_content()
    
    def ensure_browser_compatibility(self):
        """브라우저 호환성 보장"""
        
        st.markdown("""
        <script>
        // 기본 브라우저 호환성 체크
        (function() {
            var isCompatible = true;
            var warnings = [];
            
            // ES6 지원 체크
            try {
                eval('const test = () => {};');
            } catch (e) {
                isCompatible = false;
                warnings.push('ES6 지원이 필요합니다.');
            }
            
            // Flexbox 지원 체크
            var testEl = document.createElement('div');
            testEl.style.display = 'flex';
            if (testEl.style.display !== 'flex') {
                warnings.push('Flexbox 지원이 필요합니다.');
            }
            
            // 경고 표시
            if (warnings.length > 0) {
                var warningDiv = document.createElement('div');
                warningDiv.innerHTML = '⚠️ 브라우저 호환성 경고: ' + warnings.join(', ');
                warningDiv.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background: #ff9800;
                    color: white;
                    text-align: center;
                    padding: 0.5rem;
                    z-index: 9999;
                `;
                document.body.appendChild(warningDiv);
            }
        })();
        </script>
        """, unsafe_allow_html=True)
    
    def get_responsive_metrics(self) -> Dict[str, Any]:
        """반응형 메트릭 반환"""
        return {
            'screen_size': self.ui_context.screen_size.value,
            'viewport_width': self.ui_context.viewport_width,
            'viewport_height': self.ui_context.viewport_height,
            'is_mobile': self.ui_context.is_mobile,
            'device_type': self.ui_context.device_type,
            'theme_preference': self.ui_context.theme_preference
        }