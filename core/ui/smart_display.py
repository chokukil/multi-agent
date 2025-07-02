"""
🎨 Smart Display - Streamlit Magic 기반 지능형 콘텐츠 렌더링
A2A SDK 0.2.9 표준을 준수하면서 타입별 자동 렌더링 제공
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Optional
from datetime import datetime
import uuid


# 전역 인스턴스를 클래스 정의 앞으로 이동
smart_display = None

class SmartDisplayManager:
    """Streamlit Magic을 활용한 지능형 콘텐츠 표시 관리자"""
    
    def __init__(self):
        self.chart_counter = 0
        self.unique_session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # 전역 인스턴스 설정
        global smart_display
        if smart_display is None:
            smart_display = self
    
    def smart_display_content(self, content: Any, container: Optional[st.container] = None) -> None:
        """Streamlit Magic을 활용한 지능형 콘텐츠 표시"""
        
        if container is None:
            container = st.container()
        
        with container:
            # 1. 기본 Streamlit Magic 활용
            if self._is_simple_content(content):
                st.write(content)
                return
            
            # 2. 코드 블록 감지 및 예쁜 표시
            if self._is_code_content(content):
                self._render_code_block(content)
                return
            
            # 3. 마크다운 감지 및 렌더링
            if self._is_markdown_content(content):
                self._render_markdown(content)
                return
            
            # 4. DataFrame 자동 감지
            if self._is_dataframe_content(content):
                self._render_dataframe(content)
                return
            
            # 5. Plotly 차트 감지
            if self._is_plotly_chart(content):
                self._render_plotly_chart(content)
                return
            
            # 6. 이미지 감지
            if self._is_image_content(content):
                self._render_image(content)
                return
            
            # 7. JSON/Dict 구조화된 데이터
            if self._is_structured_data(content):
                self._render_structured_data(content)
                return
            
            # 8. Fallback to Streamlit Magic
            st.write(content)
    
    def _is_simple_content(self, content: Any) -> bool:
        """단순한 콘텐츠인지 확인"""
        return isinstance(content, (int, float, bool, type(None)))
    
    def _is_code_content(self, content: Any) -> bool:
        """코드 블록인지 감지"""
        if not isinstance(content, str):
            return False
        
        code_indicators = [
            'import ', 'from ', 'def ', 'class ', 'print(', 'return ',
            'if __name__', 'async def', 'await ', 'lambda ', '#!/usr/bin'
        ]
        
        return any(indicator in content.lower() for indicator in code_indicators)
    
    def _is_markdown_content(self, content: Any) -> bool:
        """마크다운 콘텐츠인지 감지"""
        if not isinstance(content, str):
            return False
        
        markdown_indicators = ['#', '**', '*', '`', '>', '- ', '+ ', '1. ', '[', '](']
        return any(indicator in content for indicator in markdown_indicators)
    
    def _is_dataframe_content(self, content: Any) -> bool:
        """DataFrame 또는 테이블 데이터인지 확인"""
        return (
            isinstance(content, pd.DataFrame) or
            hasattr(content, 'to_dataframe') or
            (isinstance(content, dict) and 'columns' in content and 'data' in content)
        )
    
    def _is_plotly_chart(self, content: Any) -> bool:
        """Plotly 차트인지 확인"""
        return (
            isinstance(content, go.Figure) or
            (isinstance(content, dict) and 'data' in content and 'layout' in content)
        )
    
    def _is_image_content(self, content: Any) -> bool:
        """이미지 콘텐츠인지 확인"""
        return (
            isinstance(content, dict) and content.get('type') == 'image'
        )
    
    def _is_structured_data(self, content: Any) -> bool:
        """구조화된 데이터인지 확인"""
        return isinstance(content, (dict, list)) and not self._is_plotly_chart(content)
    
    def _render_code_block(self, content: str) -> None:
        """코드 블록을 예쁘게 렌더링"""
        language = self._detect_language(content)
        title = self._extract_code_title(content)
        
        if title:
            st.markdown(f"**{title}**")
        
        st.code(content, language=language, line_numbers=True)
    
    def _render_markdown(self, content: str) -> None:
        """마크다운을 예쁘게 렌더링"""
        # 커스텀 CSS 스타일 적용
        st.markdown("""
        <style>
        .smart-markdown {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
        }
        .smart-markdown h1, .smart-markdown h2, .smart-markdown h3 {
            color: #3498db;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 20px;
        }
        .smart-markdown code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            color: #e74c3c;
            font-family: 'Courier New', monospace;
        }
        .smart-markdown pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }
        .smart-markdown blockquote {
            border-left: 4px solid #3498db;
            margin: 10px 0;
            padding-left: 20px;
            color: #7f8c8d;
            font-style: italic;
            background: #f8f9fa;
            padding: 10px 20px;
            border-radius: 0 8px 8px 0;
        }
        .smart-markdown ul, .smart-markdown ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        .smart-markdown li {
            margin: 5px 0;
        }
        .smart-markdown table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        .smart-markdown th, .smart-markdown td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .smart-markdown th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 마크다운 내용을 스타일과 함께 렌더링
        st.markdown(content, unsafe_allow_html=True)
    
    def _render_dataframe(self, content: Any) -> None:
        """DataFrame을 예쁘게 렌더링"""
        if isinstance(content, pd.DataFrame):
            df = content
        elif hasattr(content, 'to_dataframe'):
            df = content.to_dataframe()
        elif isinstance(content, dict):
            df = pd.DataFrame(content['data'], columns=content['columns'])
        else:
            df = pd.DataFrame(content)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("행 수", len(df))
        with col2:
            st.metric("열 수", len(df.columns))
        with col3:
            st.metric("메모리 사용량", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        st.dataframe(df, use_container_width=True, height=400)
    
    def _render_plotly_chart(self, content: Any) -> None:
        """Plotly 차트를 고유 키와 함께 렌더링"""
        self.chart_counter += 1
        chart_key = f"chart_{self.unique_session_id}_{self.chart_counter}"
        
        if isinstance(content, dict):
            fig = go.Figure(data=content.get('data', []), layout=content.get('layout', {}))
        else:
            fig = content
        
        st.plotly_chart(fig, key=chart_key, use_container_width=True)
    
    def _render_image(self, content: Any) -> None:
        """이미지를 예쁘게 렌더링"""
        if isinstance(content, dict) and content.get('type') == 'image':
            st.image(content['data'], caption=content.get('caption', ''), use_column_width=True)
        else:
            st.image(content, use_column_width=True)
    
    def _render_structured_data(self, content: Any) -> None:
        """구조화된 데이터를 예쁘게 렌더링"""
        if isinstance(content, dict):
            important_keys = ['type', 'name', 'description', 'status', 'result']
            
            for key in important_keys:
                if key in content:
                    st.markdown(f"**{key.title()}**: {content[key]}")
            
            remaining_data = {k: v for k, v in content.items() if k not in important_keys}
            if remaining_data:
                with st.expander("📋 상세 데이터", expanded=False):
                    st.json(remaining_data)
        else:
            st.json(content)
    
    def _detect_language(self, content: str) -> str:
        """코드 언어 자동 감지"""
        if 'import ' in content or 'def ' in content or 'print(' in content:
            return 'python'
        elif 'function ' in content or 'const ' in content or 'let ' in content:
            return 'javascript'
        elif 'SELECT ' in content.upper() or 'FROM ' in content.upper():
            return 'sql'
        elif '<' in content and '>' in content and 'html' in content.lower():
            return 'html'
        elif '{' in content and '}' in content and 'css' in content.lower():
            return 'css'
        else:
            return 'text'
    
    def _extract_code_title(self, content: str) -> Optional[str]:
        """코드에서 제목 추출"""
        lines = content.split('\n')
        for line in lines[:3]:
            if line.strip().startswith('#') and not line.strip().startswith('#!/'):
                return line.strip().lstrip('#').strip()
            elif '"""' in line or "'''" in line:
                return line.strip().strip('"""').strip("'''").strip()
        return None


class AccumulativeStreamContainer:
    """누적형 스트리밍 컨테이너"""
    
    def __init__(self, container_key: str = None):
        self.container_key = container_key or f"stream_{uuid.uuid4().hex[:8]}"
        self.accumulated_content = []
        self.container = st.empty()
        
        # SmartDisplayManager 인스턴스 확보
        global smart_display
        if smart_display is None:
            smart_display = SmartDisplayManager()
        
    def add_chunk(self, chunk: str, content_type: str = "text") -> None:
        """새로운 청크를 누적하여 표시 (호환성을 위해 content_type 매개변수 추가)"""
        if chunk:
            # 아티팩트인 경우 특별 처리
            if content_type == "artifact" and isinstance(chunk, dict):
                # 아티팩트를 Smart Display로 직접 렌더링
                artifact_name = chunk.get('name', 'Artifact')
                self.accumulated_content.append(f"\n\n### 📦 {artifact_name}\n")
                
                with self.container.container():
                    # 기존 누적 내용 표시
                    full_content = "".join(self.accumulated_content)
                    smart_display.smart_display_content(full_content)
                    
                    # 아티팩트를 별도 컨테이너에서 렌더링 (중복 방지)
                    with st.expander(f"📄 {artifact_name}", expanded=True):
                        smart_display.smart_display_content(chunk)
            else:
                # 일반 텍스트 처리
                if isinstance(chunk, str):
                    self.accumulated_content.append(chunk)
                else:
                    self.accumulated_content.append(str(chunk))
                
                full_content = "".join(self.accumulated_content)
                
                with self.container.container():
                    smart_display.smart_display_content(full_content)
    
    def clear(self) -> None:
        """컨테이너 내용 초기화"""
        self.accumulated_content = []
        self.container.empty()
    
    def finalize(self) -> str:
        """최종 누적 내용 반환"""
        return "".join(self.accumulated_content)


# 전역 인스턴스 초기화
if smart_display is None:
    smart_display = SmartDisplayManager()
