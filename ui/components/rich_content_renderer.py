#!/usr/bin/env python3
"""
🎨 CherryAI 리치 콘텐츠 렌더링 시스템

차트, 코드, 테이블, 이미지 등 다양한 콘텐츠를 완벽하게 렌더링하는 고급 시스템

Key Features:
- Plotly/Altair 차트 인터랙티브 렌더링
- 코드 블록 문법 하이라이팅 (Python, SQL, R, JavaScript)
- 데이터프레임 정렬/필터 가능한 테이블
- 이미지 표시 및 다운로드
- 수식 렌더링 (LaTeX)
- JSON/XML 포맷팅
- 복사/다운로드 기능
- 반응형 레이아웃

Architecture:
- Content Detection: 콘텐츠 타입 자동 감지
- Rendering Engine: 타입별 최적화된 렌더링
- Interactive Elements: 사용자 인터랙션 지원
- Export Functions: 다양한 형식으로 내보내기
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import json
import re
import base64
import io
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# 코드 하이라이팅을 위한 임포트
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """콘텐츠 타입"""
    TEXT = "text"
    CODE = "code"
    DATAFRAME = "dataframe"
    CHART_PLOTLY = "chart_plotly"
    CHART_ALTAIR = "chart_altair"
    IMAGE = "image"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    ERROR = "error"

class RenderingMode(Enum):
    """렌더링 모드"""
    INLINE = "inline"  # 인라인 표시
    EXPANDABLE = "expandable"  # 확장 가능
    MODAL = "modal"  # 모달 창
    DOWNLOAD = "download"  # 다운로드만

@dataclass
class RichContent:
    """리치 콘텐츠 데이터 클래스"""
    id: str
    content_type: ContentType
    data: Any
    metadata: Dict[str, Any]
    rendering_mode: RenderingMode = RenderingMode.INLINE
    title: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())

class RichContentRenderer:
    """
    🎨 리치 콘텐츠 렌더링 엔진
    
    다양한 타입의 콘텐츠를 최적화된 방식으로 렌더링
    """
    
    def __init__(self):
        """렌더러 초기화"""
        self.supported_languages = [
            'python', 'sql', 'r', 'javascript', 'typescript', 
            'bash', 'json', 'yaml', 'xml', 'html', 'css', 'markdown'
        ]
        
        # 차트 설정
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'cherry_chart',
                'height': 500,
                'width': 700,
                'scale': 1
            }
        }
        
        # 테이블 설정
        self.table_config = {
            'height': 400,
            'use_container_width': True
        }
        
        logger.info("🎨 리치 콘텐츠 렌더러 초기화 완료")
    
    def apply_rich_content_styles(self):
        """리치 콘텐츠 스타일 적용"""
        st.markdown("""
        <style>
        /* 리치 콘텐츠 컨테이너 */
        .rich-content-container {
            margin: 16px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }
        
        /* 콘텐츠 헤더 */
        .content-header {
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 12px 16px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .content-title {
            font-weight: 600;
            color: #2d3748;
            margin: 0;
            font-size: 14px;
        }
        
        .content-actions {
            display: flex;
            gap: 8px;
        }
        
        .action-btn {
            background: none;
            border: 1px solid #cbd5e0;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .action-btn:hover {
            background: #f7fafc;
            border-color: #a0aec0;
        }
        
        /* 코드 블록 스타일 */
        .code-container {
            position: relative;
            background: #f8f9fa;
        }
        
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #e9ecef;
            border-bottom: 1px solid #dee2e6;
            font-size: 12px;
            color: #6c757d;
        }
        
        .code-language {
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .code-copy-btn {
            background: none;
            border: none;
            color: #6c757d;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        .code-copy-btn:hover {
            background: #f8f9fa;
            color: #495057;
        }
        
        .code-content {
            padding: 16px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        
        /* 테이블 스타일 */
        .table-container {
            overflow-x: auto;
            max-height: 400px;
        }
        
        .table-info {
            padding: 8px 12px;
            background: #f8f9fa;
            border-bottom: 1px solid #e2e8f0;
            font-size: 12px;
            color: #6c757d;
        }
        
        /* 차트 컨테이너 */
        .chart-container {
            padding: 16px;
            background: white;
        }
        
        .chart-info {
            margin-bottom: 12px;
            font-size: 12px;
            color: #6c757d;
            text-align: center;
        }
        
        /* 이미지 컨테이너 */
        .image-container {
            text-align: center;
            padding: 16px;
            background: white;
        }
        
        .image-wrapper {
            display: inline-block;
            max-width: 100%;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* JSON/XML 포맷팅 */
        .json-container, .xml-container {
            background: #f8f9fa;
            padding: 16px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        /* 에러 표시 */
        .error-container {
            background: #fed7d7;
            border: 1px solid #feb2b2;
            color: #c53030;
            padding: 16px;
        }
        
        .error-title {
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .error-message {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
        }
        
        /* 확장 가능한 콘텐츠 */
        .expandable-content {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .expandable-content:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .expand-indicator {
            margin-left: 8px;
            font-size: 12px;
            transition: transform 0.3s ease;
        }
        
        .expanded .expand-indicator {
            transform: rotate(90deg);
        }
        
        /* 반응형 디자인 */
        @media (max-width: 768px) {
            .rich-content-container {
                margin: 12px 0;
            }
            
            .content-header {
                padding: 8px 12px;
            }
            
            .content-actions {
                gap: 4px;
            }
            
            .action-btn {
                padding: 2px 6px;
                font-size: 11px;
            }
            
            .code-content {
                padding: 12px;
                font-size: 13px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def detect_content_type(self, content: Any) -> ContentType:
        """콘텐츠 타입 자동 감지"""
        try:
            # DataFrame 체크
            if isinstance(content, pd.DataFrame):
                return ContentType.DATAFRAME
            
            # Plotly 차트 체크
            if hasattr(content, '__module__') and 'plotly' in str(content.__module__):
                return ContentType.CHART_PLOTLY
            
            # Altair 차트 체크
            if hasattr(content, '__module__') and 'altair' in str(content.__module__):
                return ContentType.CHART_ALTAIR
            
            # 문자열 콘텐츠 분석
            if isinstance(content, str):
                content_lower = content.strip().lower()
                
                # JSON 체크
                if content_lower.startswith(('{', '[')):
                    try:
                        json.loads(content)
                        return ContentType.JSON
                    except json.JSONDecodeError:
                        pass
                
                # XML 체크
                if content_lower.startswith('<?xml') or re.match(r'<\w+.*>', content_lower):
                    return ContentType.XML
                
                # LaTeX 체크
                if content.strip().startswith('$$') or '\\begin{' in content:
                    return ContentType.LATEX
                
                # HTML 체크
                if content_lower.startswith('<!doctype html') or '<html' in content_lower:
                    return ContentType.HTML
                
                # 코드 블록 체크 (```로 감싸진 경우)
                if content.strip().startswith('```') and content.strip().endswith('```'):
                    return ContentType.CODE
                
                # Markdown 체크 (헤더, 리스트 등이 있는 경우)
                if any(line.strip().startswith(('#', '*', '-', '1.')) for line in content.split('\n')):
                    return ContentType.MARKDOWN
            
            # 이미지 바이트 체크
            if isinstance(content, bytes):
                # 이미지 시그니처 체크
                if content.startswith(b'\xff\xd8\xff') or content.startswith(b'\x89PNG'):
                    return ContentType.IMAGE
            
            # 기본적으로 텍스트로 처리
            return ContentType.TEXT
            
        except Exception as e:
            logger.error(f"콘텐츠 타입 감지 실패: {e}")
            return ContentType.ERROR
    
    def render_content(self, content: RichContent) -> None:
        """콘텐츠 렌더링 메인 함수"""
        try:
            # 스타일 적용
            self.apply_rich_content_styles()
            
            # 컨테이너 시작
            container_class = "rich-content-container"
            if content.rendering_mode == RenderingMode.EXPANDABLE:
                container_class += " expandable-content"
            
            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            
            # 헤더 렌더링
            self._render_content_header(content)
            
            # 콘텐츠 타입별 렌더링
            if content.content_type == ContentType.DATAFRAME:
                self._render_dataframe(content)
            elif content.content_type == ContentType.CHART_PLOTLY:
                self._render_plotly_chart(content)
            elif content.content_type == ContentType.CHART_ALTAIR:
                self._render_altair_chart(content)
            elif content.content_type == ContentType.CODE:
                self._render_code_block(content)
            elif content.content_type == ContentType.IMAGE:
                self._render_image(content)
            elif content.content_type == ContentType.JSON:
                self._render_json(content)
            elif content.content_type == ContentType.XML:
                self._render_xml(content)
            elif content.content_type == ContentType.MARKDOWN:
                self._render_markdown(content)
            elif content.content_type == ContentType.LATEX:
                self._render_latex(content)
            elif content.content_type == ContentType.HTML:
                self._render_html(content)
            elif content.content_type == ContentType.ERROR:
                self._render_error(content)
            else:
                self._render_text(content)
            
            # 컨테이너 종료
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"콘텐츠 렌더링 실패: {e}")
            self._render_error(RichContent(
                id=content.id,
                content_type=ContentType.ERROR,
                data=str(e),
                metadata={"error_type": "rendering_error"}
            ))
    
    def _render_content_header(self, content: RichContent) -> None:
        """콘텐츠 헤더 렌더링"""
        title = content.title or f"{content.content_type.value.title()} Content"
        
        # 액션 버튼들
        actions_html = f"""
        <div class="content-actions">
            <button class="action-btn" onclick="copyContent('{content.id}')">📋 복사</button>
            <button class="action-btn" onclick="downloadContent('{content.id}')">⬇️ 다운로드</button>
            <button class="action-btn" onclick="expandContent('{content.id}')">🔍 확대</button>
        </div>
        """
        
        st.markdown(f"""
        <div class="content-header">
            <h4 class="content-title">{title}</h4>
            {actions_html}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_dataframe(self, content: RichContent) -> None:
        """데이터프레임 렌더링"""
        df = content.data
        
        # 테이블 정보
        info_html = f"""
        <div class="table-info">
            📊 {df.shape[0]:,} 행 × {df.shape[1]:,} 열 | 메모리: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
        
        # 데이터프레임 표시 (Streamlit의 고급 기능 사용)
        st.dataframe(
            df,
            use_container_width=True,
            height=self.table_config['height']
        )
        
        # 기본 통계 정보 (숫자 컬럼이 있는 경우)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            with st.expander("📈 기본 통계 정보"):
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    def _render_plotly_chart(self, content: RichContent) -> None:
        """Plotly 차트 렌더링"""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # 차트 정보
        chart_info = content.metadata.get('chart_info', '')
        if chart_info:
            st.markdown(f'<div class="chart-info">{chart_info}</div>', unsafe_allow_html=True)
        
        # 차트 표시
        st.plotly_chart(
            content.data,
            use_container_width=True,
            config=self.chart_config
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_altair_chart(self, content: RichContent) -> None:
        """Altair 차트 렌더링"""
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # 차트 정보
        chart_info = content.metadata.get('chart_info', '')
        if chart_info:
            st.markdown(f'<div class="chart-info">{chart_info}</div>', unsafe_allow_html=True)
        
        # 차트 표시
        st.altair_chart(content.data, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_code_block(self, content: RichContent) -> None:
        """코드 블록 렌더링"""
        code_text = content.data
        language = content.metadata.get('language', 'text')
        
        # 코드 블록에서 언어와 코드 분리
        if code_text.strip().startswith('```'):
            lines = code_text.strip().split('\n')
            if len(lines) > 1:
                first_line = lines[0][3:].strip()
                if first_line and first_line in self.supported_languages:
                    language = first_line
                    code_text = '\n'.join(lines[1:-1])
        
        st.markdown('<div class="code-container">', unsafe_allow_html=True)
        
        # 코드 헤더
        st.markdown(f"""
        <div class="code-header">
            <span class="code-language">{language}</span>
            <button class="code-copy-btn" onclick="copyCode('{content.id}')">📋 복사</button>
        </div>
        """, unsafe_allow_html=True)
        
        # 코드 내용
        if PYGMENTS_AVAILABLE and language in self.supported_languages:
            try:
                lexer = get_lexer_by_name(language)
                formatter = HtmlFormatter(style='github', noclasses=True)
                highlighted_code = highlight(code_text, lexer, formatter)
                st.markdown(f'<div class="code-content">{highlighted_code}</div>', unsafe_allow_html=True)
            except Exception:
                # 하이라이팅 실패 시 기본 코드 표시
                st.code(code_text, language=language)
        else:
            st.code(code_text, language=language)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_image(self, content: RichContent) -> None:
        """이미지 렌더링"""
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        
        # 이미지 표시
        if isinstance(content.data, bytes):
            st.image(content.data, use_column_width=True)
        elif isinstance(content.data, str):
            # Base64 또는 URL
            st.image(content.data, use_column_width=True)
        
        # 이미지 정보
        image_info = content.metadata.get('image_info', '')
        if image_info:
            st.caption(image_info)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_json(self, content: RichContent) -> None:
        """JSON 렌더링"""
        try:
            if isinstance(content.data, str):
                json_obj = json.loads(content.data)
            else:
                json_obj = content.data
            
            formatted_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
            
            st.markdown('<div class="json-container">', unsafe_allow_html=True)
            st.markdown(f'<pre>{formatted_json}</pre>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except json.JSONDecodeError as e:
            self._render_error(RichContent(
                id=content.id,
                content_type=ContentType.ERROR,
                data=f"JSON 파싱 오류: {str(e)}",
                metadata={"error_type": "json_parse_error"}
            ))
    
    def _render_xml(self, content: RichContent) -> None:
        """XML 렌더링 - LLM First 원칙 적용"""
        st.markdown('<div class="xml-container">', unsafe_allow_html=True)
        
        # XML 콘텐츠 처리
        xml_content = content.data
        if isinstance(xml_content, str):
            # XML이 실제 렌더링 가능한 형태인지 LLM이 판단한 결과라면 그대로 표시
            # 코드로 표시해야 하는 경우만 이스케이프
            if xml_content.strip().startswith('<?xml') or xml_content.startswith('<'):
                # 실제 XML 문서나 코드 예제의 경우 이스케이프하여 코드로 표시
                xml_content = xml_content.replace('<', '&lt;').replace('>', '&gt;')
                st.markdown(f'<pre><code>{xml_content}</code></pre>', unsafe_allow_html=True)
            else:
                # LLM이 생성한 형식화된 콘텐츠는 그대로 렌더링
                st.markdown(xml_content, unsafe_allow_html=True)
        else:
            st.markdown(f'<pre><code>{xml_content}</code></pre>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_markdown(self, content: RichContent) -> None:
        """Markdown 렌더링"""
        st.markdown(content.data)
    
    def _render_latex(self, content: RichContent) -> None:
        """LaTeX 수식 렌더링"""
        st.latex(content.data)
    
    def _render_html(self, content: RichContent) -> None:
        """HTML 렌더링"""
        st.markdown(content.data, unsafe_allow_html=True)
    
    def _render_text(self, content: RichContent) -> None:
        """일반 텍스트 렌더링"""
        st.text(content.data)
    
    def _render_error(self, content: RichContent) -> None:
        """에러 메시지 렌더링"""
        error_message = content.data
        error_type = content.metadata.get('error_type', 'unknown_error')
        
        st.markdown(f"""
        <div class="error-container">
            <div class="error-title">⚠️ 오류 발생 ({error_type})</div>
            <div class="error-message">{error_message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_content(self, data: Any, title: str = None, **kwargs) -> RichContent:
        """콘텐츠 객체 생성"""
        content_type = self.detect_content_type(data)
        
        return RichContent(
            id=str(uuid.uuid4()),
            content_type=content_type,
            data=data,
            metadata=kwargs,
            title=title
        )
    
    def render_auto(self, data: Any, title: str = None, **kwargs) -> None:
        """자동 감지 및 렌더링"""
        content = self.create_content(data, title, **kwargs)
        self.render_content(content)

# JavaScript 함수들 (클라이언트 사이드)
def inject_javascript_functions():
    """JavaScript 함수들을 주입"""
    st.markdown("""
    <script>
    function copyContent(contentId) {
        console.log('Copy content:', contentId);
        // 실제 복사 로직 구현
    }
    
    function downloadContent(contentId) {
        console.log('Download content:', contentId);
        // 실제 다운로드 로직 구현
    }
    
    function expandContent(contentId) {
        console.log('Expand content:', contentId);
        // 실제 확대 로직 구현
    }
    
    function copyCode(contentId) {
        const codeElement = document.querySelector(`[data-content-id="${contentId}"] code`);
        if (codeElement) {
            navigator.clipboard.writeText(codeElement.textContent);
            console.log('Code copied to clipboard');
        }
    }
    </script>
    """, unsafe_allow_html=True)

# 전역 인스턴스 관리
_rich_content_renderer_instance = None

def get_rich_content_renderer() -> RichContentRenderer:
    """리치 콘텐츠 렌더러 싱글톤 인스턴스 반환"""
    global _rich_content_renderer_instance
    if _rich_content_renderer_instance is None:
        _rich_content_renderer_instance = RichContentRenderer()
    return _rich_content_renderer_instance

def initialize_rich_content_renderer() -> RichContentRenderer:
    """리치 콘텐츠 렌더러 초기화"""
    global _rich_content_renderer_instance
    _rich_content_renderer_instance = RichContentRenderer()
    inject_javascript_functions()
    return _rich_content_renderer_instance 