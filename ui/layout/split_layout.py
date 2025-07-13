"""
드래그 조절 가능한 Split 레이아웃 컴포넌트
"""
import streamlit as st
import streamlit.components.v1 as components
from typing import Callable, Optional, Any

class SplitLayout:
    """드래그 조절 가능한 Split 레이아웃"""
    
    def __init__(
        self,
        default_ratio: float = 0.3,
        min_ratio: float = 0.2,
        max_ratio: float = 0.6,
        session_key: str = "split_ratio"
    ):
        self.default_ratio = default_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.session_key = session_key
        
        # 세션 상태 초기화
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = default_ratio
    
    def _get_split_css(self, ratio: float) -> str:
        """Split 레이아웃용 CSS 생성"""
        return f"""
        <style>
        .split-container {{
            display: flex;
            height: 100vh;
            width: 100%;
            position: relative;
        }}
        
        .split-left {{
            width: {ratio * 100}%;
            height: 100%;
            overflow-y: auto;
            padding: 1rem;
            border-right: 1px solid #333;
            position: relative;
        }}
        
        .split-right {{
            width: {(1 - ratio) * 100}%;
            height: 100%;
            overflow-y: auto;
            padding: 1rem;
            position: relative;
        }}
        
        .split-divider {{
            width: 8px;
            height: 100%;
            background: #333;
            cursor: col-resize;
            position: absolute;
            left: {ratio * 100}%;
            top: 0;
            z-index: 1000;
            transition: background-color 0.2s;
        }}
        
        .split-divider:hover {{
            background: #555;
        }}
        
        .split-divider::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 3px;
            height: 30px;
            background: #666;
            border-radius: 2px;
        }}
        
        /* 드래그 중 스타일 */
        .split-divider.dragging {{
            background: #007acc;
        }}
        
        .split-container.dragging {{
            user-select: none;
        }}
        
        /* 자동 스크롤 스타일 */
        .auto-scroll {{
            scroll-behavior: smooth;
        }}
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {{
            .split-container {{
                flex-direction: column;
            }}
            
            .split-left {{
                width: 100% !important;
                height: 40%;
                border-right: none;
                border-bottom: 1px solid #333;
            }}
            
            .split-right {{
                width: 100% !important;
                height: 60%;
            }}
            
            .split-divider {{
                display: none;
            }}
        }}
        </style>
        """
    
    def _get_split_js(self) -> str:
        """Split 레이아웃용 JavaScript 생성"""
        return f"""
        <script>
        (function() {{
            let isDragging = false;
            let startX = 0;
            let startRatio = {st.session_state[self.session_key]};
            
            function initializeSplitLayout() {{
                const container = document.querySelector('.split-container');
                const divider = document.querySelector('.split-divider');
                
                if (!container || !divider) {{
                    setTimeout(initializeSplitLayout, 100);
                    return;
                }}
                
                divider.addEventListener('mousedown', handleMouseDown);
                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
                
                // 터치 이벤트 지원
                divider.addEventListener('touchstart', handleTouchStart);
                document.addEventListener('touchmove', handleTouchMove);
                document.addEventListener('touchend', handleTouchEnd);
            }}
            
            function handleMouseDown(e) {{
                isDragging = true;
                startX = e.clientX;
                startRatio = {st.session_state[self.session_key]};
                
                document.body.classList.add('dragging');
                document.querySelector('.split-divider').classList.add('dragging');
                e.preventDefault();
            }}
            
            function handleMouseMove(e) {{
                if (!isDragging) return;
                
                const container = document.querySelector('.split-container');
                const deltaX = e.clientX - startX;
                const containerWidth = container.offsetWidth;
                const deltaRatio = deltaX / containerWidth;
                
                let newRatio = startRatio + deltaRatio;
                newRatio = Math.max({self.min_ratio}, Math.min({self.max_ratio}, newRatio));
                
                updateLayout(newRatio);
                e.preventDefault();
            }}
            
            function handleMouseUp() {{
                if (!isDragging) return;
                
                isDragging = false;
                document.body.classList.remove('dragging');
                document.querySelector('.split-divider').classList.remove('dragging');
                
                // Streamlit에 새로운 비율 전송
                const newRatio = parseFloat(document.querySelector('.split-divider').style.left) / 100;
                window.parent.postMessage({{
                    type: 'split_ratio_change',
                    ratio: newRatio
                }}, '*');
            }}
            
            function handleTouchStart(e) {{
                const touch = e.touches[0];
                handleMouseDown({{ clientX: touch.clientX, preventDefault: () => e.preventDefault() }});
            }}
            
            function handleTouchMove(e) {{
                const touch = e.touches[0];
                handleMouseMove({{ clientX: touch.clientX, preventDefault: () => e.preventDefault() }});
            }}
            
            function handleTouchEnd(e) {{
                handleMouseUp();
            }}
            
            function updateLayout(ratio) {{
                const leftPanel = document.querySelector('.split-left');
                const rightPanel = document.querySelector('.split-right');
                const divider = document.querySelector('.split-divider');
                
                if (leftPanel && rightPanel && divider) {{
                    leftPanel.style.width = (ratio * 100) + '%';
                    rightPanel.style.width = ((1 - ratio) * 100) + '%';
                    divider.style.left = (ratio * 100) + '%';
                }}
            }}
            
            // 초기화
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', initializeSplitLayout);
            }} else {{
                initializeSplitLayout();
            }}
        }})();
        </script>
        """
    
    def render(self, left_content: Callable, right_content: Callable) -> None:
        """Split 레이아웃 렌더링"""
        ratio = st.session_state[self.session_key]
        
        # CSS 및 JS 적용
        st.markdown(self._get_split_css(ratio), unsafe_allow_html=True)
        components.html(self._get_split_js(), height=0)
        
        # 비율 조절을 위한 슬라이더 (개발용)
        if st.session_state.get('debug_mode', False):
            st.sidebar.subheader("🔧 Layout Controls")
            new_ratio = st.sidebar.slider(
                "Split Ratio",
                min_value=self.min_ratio,
                max_value=self.max_ratio,
                value=ratio,
                step=0.05,
                key=f"{self.session_key}_slider"
            )
            if new_ratio != ratio:
                st.session_state[self.session_key] = new_ratio
                st.rerun()
        
        # Split 컨테이너 시작
        st.markdown('<div class="split-container">', unsafe_allow_html=True)
        
        # 좌측 패널
        st.markdown('<div class="split-left">', unsafe_allow_html=True)
        left_content()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 구분선
        st.markdown('<div class="split-divider"></div>', unsafe_allow_html=True)
        
        # 우측 패널
        st.markdown('<div class="split-right">', unsafe_allow_html=True)
        right_content()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Split 컨테이너 끝
        st.markdown('</div>', unsafe_allow_html=True)
    
    def update_ratio(self, new_ratio: float) -> None:
        """비율 업데이트"""
        new_ratio = max(self.min_ratio, min(self.max_ratio, new_ratio))
        st.session_state[self.session_key] = new_ratio


def create_split_layout(
    default_ratio: float = 0.3,
    min_ratio: float = 0.2,
    max_ratio: float = 0.6,
    session_key: str = "split_ratio"
) -> SplitLayout:
    """Split 레이아웃 인스턴스 생성"""
    return SplitLayout(
        default_ratio=default_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        session_key=session_key
    ) 