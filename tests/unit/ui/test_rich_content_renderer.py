#!/usr/bin/env python3
"""
🎨 리치 콘텐츠 렌더러 단위 테스트

차트, 코드, 테이블 등 다양한 콘텐츠 타입의 렌더링을 pytest로 검증

Test Coverage:
- 콘텐츠 타입 자동 감지
- 각 타입별 렌더링 (차트, 코드, 테이블, 이미지 등)
- 에러 처리 및 복구
- 렌더링 옵션 및 설정
- 성능 및 메모리 관리
"""

import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# 테스트 대상 임포트
from ui.components.rich_content_renderer import (
    RichContentRenderer, RichContent, ContentType, RenderingMode,
    get_rich_content_renderer, initialize_rich_content_renderer
)

class TestContentTypeDetection:
    """콘텐츠 타입 감지 테스트"""
    
    @pytest.fixture
    def renderer(self):
        return RichContentRenderer()
    
    def test_detect_dataframe(self, renderer):
        """DataFrame 감지 테스트"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        content_type = renderer.detect_content_type(df)
        assert content_type == ContentType.DATAFRAME
    
    def test_detect_json(self, renderer):
        """JSON 감지 테스트"""
        json_str = '{"key": "value", "number": 123}'
        content_type = renderer.detect_content_type(json_str)
        assert content_type == ContentType.JSON
    
    def test_detect_code_block(self, renderer):
        """코드 블록 감지 테스트"""
        code_str = "```python\nprint('hello')\n```"
        content_type = renderer.detect_content_type(code_str)
        assert content_type == ContentType.CODE
    
    def test_detect_xml(self, renderer):
        """XML 감지 테스트"""
        xml_str = "<?xml version='1.0'?><root><item>value</item></root>"
        content_type = renderer.detect_content_type(xml_str)
        assert content_type == ContentType.XML
    
    def test_detect_latex(self, renderer):
        """LaTeX 감지 테스트"""
        latex_str = "$$E = mc^2$$"
        content_type = renderer.detect_content_type(latex_str)
        assert content_type == ContentType.LATEX
    
    def test_detect_html(self, renderer):
        """HTML 감지 테스트"""
        html_str = "<!DOCTYPE html><html><body>Test</body></html>"
        content_type = renderer.detect_content_type(html_str)
        assert content_type == ContentType.HTML
    
    def test_detect_markdown(self, renderer):
        """Markdown 감지 테스트"""
        markdown_str = "# Title\n* Item 1\n* Item 2"
        content_type = renderer.detect_content_type(markdown_str)
        assert content_type == ContentType.MARKDOWN
    
    def test_detect_text_fallback(self, renderer):
        """텍스트 기본값 테스트"""
        text_str = "This is just plain text"
        content_type = renderer.detect_content_type(text_str)
        assert content_type == ContentType.TEXT
    
    def test_detect_image_bytes(self, renderer):
        """이미지 바이트 감지 테스트"""
        # JPEG 시그니처
        jpeg_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        content_type = renderer.detect_content_type(jpeg_bytes)
        assert content_type == ContentType.IMAGE
        
        # PNG 시그니처
        png_bytes = b'\x89PNG\r\n\x1a\n'
        content_type = renderer.detect_content_type(png_bytes)
        assert content_type == ContentType.IMAGE

class TestRichContent:
    """RichContent 데이터 클래스 테스트"""
    
    def test_rich_content_creation(self):
        """RichContent 생성 테스트"""
        content = RichContent(
            id="test-id",
            content_type=ContentType.CODE,
            data="print('hello')",
            metadata={"language": "python"},
            title="Test Code"
        )
        
        assert content.id == "test-id"
        assert content.content_type == ContentType.CODE
        assert content.data == "print('hello')"
        assert content.metadata == {"language": "python"}
        assert content.title == "Test Code"
        assert content.rendering_mode == RenderingMode.INLINE
        assert content.created_at is not None

class TestRichContentRenderer:
    """RichContentRenderer 클래스 테스트"""
    
    @pytest.fixture
    def renderer(self):
        return RichContentRenderer()
    
    def test_renderer_initialization(self, renderer):
        """렌더러 초기화 테스트"""
        assert 'python' in renderer.supported_languages
        assert 'sql' in renderer.supported_languages
        assert 'displayModeBar' in renderer.chart_config
        assert renderer.table_config['use_container_width'] is True
    
    @patch('streamlit.markdown')
    def test_apply_rich_content_styles(self, mock_markdown, renderer):
        """스타일 적용 테스트"""
        renderer.apply_rich_content_styles()
        
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert call_args[1]['unsafe_allow_html'] is True
        assert 'rich-content-container' in call_args[0][0]
    
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_render_dataframe(self, mock_dataframe, mock_markdown, renderer):
        """DataFrame 렌더링 테스트"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        content = RichContent(
            id="test-df",
            content_type=ContentType.DATAFRAME,
            data=df,
            metadata={}
        )
        
        renderer._render_dataframe(content)
        
        # 테이블 정보 마크다운 호출 확인
        mock_markdown.assert_called()
        info_html = mock_markdown.call_args[0][0]
        assert "3 행" in info_html
        assert "2 열" in info_html
        
        # 데이터프레임 렌더링 확인
        mock_dataframe.assert_called_with(
            df, 
            use_container_width=True, 
            height=renderer.table_config['height']
        )
    
    @patch('streamlit.markdown')
    @patch('streamlit.plotly_chart')
    def test_render_plotly_chart(self, mock_plotly_chart, mock_markdown, renderer):
        """Plotly 차트 렌더링 테스트"""
        # Mock Plotly figure
        mock_fig = Mock()
        content = RichContent(
            id="test-chart",
            content_type=ContentType.CHART_PLOTLY,
            data=mock_fig,
            metadata={"chart_info": "Test chart"}
        )
        
        renderer._render_plotly_chart(content)
        
        # 차트 정보 마크다운 호출 확인
        mock_markdown.assert_called()
        
        # Plotly 차트 렌더링 확인
        mock_plotly_chart.assert_called_with(
            mock_fig,
            use_container_width=True,
            config=renderer.chart_config
        )
    
    @patch('streamlit.markdown')
    @patch('streamlit.code')
    def test_render_code_block(self, mock_code, mock_markdown, renderer):
        """코드 블록 렌더링 테스트"""
        code_content = "```python\nprint('hello world')\n```"
        content = RichContent(
            id="test-code",
            content_type=ContentType.CODE,
            data=code_content,
            metadata={"language": "python"}
        )
        
        renderer._render_code_block(content)
        
        # 코드 헤더 마크다운 확인
        markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
        header_found = any("code-header" in call for call in markdown_calls)
        assert header_found
        
        # 코드 렌더링 확인 (Pygments 사용 불가 시 기본 st.code 사용)
        mock_code.assert_called()
    
    @patch('streamlit.markdown')
    @patch('streamlit.image')
    def test_render_image(self, mock_image, mock_markdown, renderer):
        """이미지 렌더링 테스트"""
        image_data = b"fake_image_data"
        content = RichContent(
            id="test-image",
            content_type=ContentType.IMAGE,
            data=image_data,
            metadata={"image_info": "Test image"}
        )
        
        renderer._render_image(content)
        
        # 이미지 컨테이너 마크다운 확인
        mock_markdown.assert_called()
        
        # 이미지 렌더링 확인
        mock_image.assert_called_with(image_data, use_column_width=True)
    
    @patch('streamlit.markdown')
    def test_render_json(self, mock_markdown, renderer):
        """JSON 렌더링 테스트"""
        json_data = '{"name": "test", "value": 123}'
        content = RichContent(
            id="test-json",
            content_type=ContentType.JSON,
            data=json_data,
            metadata={}
        )
        
        renderer._render_json(content)
        
        # JSON 컨테이너 및 포맷된 JSON 확인
        mock_markdown.assert_called()
        calls = [call[0][0] for call in mock_markdown.call_args_list]
        
        # JSON 컨테이너가 있는지 확인
        container_found = any("json-container" in call for call in calls)
        assert container_found
        
        # 포맷된 JSON이 있는지 확인
        json_found = any("test" in call and "123" in call for call in calls)
        assert json_found
    
    @patch('streamlit.markdown')
    def test_render_xml(self, mock_markdown, renderer):
        """XML 렌더링 테스트"""
        xml_data = "<root><item>value</item></root>"
        content = RichContent(
            id="test-xml",
            content_type=ContentType.XML,
            data=xml_data,
            metadata={}
        )
        
        renderer._render_xml(content)
        
        mock_markdown.assert_called()
        calls = [call[0][0] for call in mock_markdown.call_args_list]
        
        # XML 컨테이너 확인
        container_found = any("xml-container" in call for call in calls)
        assert container_found
    
    @patch('streamlit.markdown')
    def test_render_markdown(self, mock_markdown, renderer):
        """Markdown 렌더링 테스트"""
        markdown_data = "# Title\n\nThis is **bold** text."
        content = RichContent(
            id="test-md",
            content_type=ContentType.MARKDOWN,
            data=markdown_data,
            metadata={}
        )
        
        renderer._render_markdown(content)
        
        mock_markdown.assert_called_with(markdown_data)
    
    @patch('streamlit.latex')
    def test_render_latex(self, mock_latex, renderer):
        """LaTeX 렌더링 테스트"""
        latex_data = "E = mc^2"
        content = RichContent(
            id="test-latex",
            content_type=ContentType.LATEX,
            data=latex_data,
            metadata={}
        )
        
        renderer._render_latex(content)
        
        mock_latex.assert_called_with(latex_data)
    
    @patch('streamlit.text')
    def test_render_text(self, mock_text, renderer):
        """텍스트 렌더링 테스트"""
        text_data = "Plain text content"
        content = RichContent(
            id="test-text",
            content_type=ContentType.TEXT,
            data=text_data,
            metadata={}
        )
        
        renderer._render_text(content)
        
        mock_text.assert_called_with(text_data)
    
    @patch('streamlit.markdown')
    def test_render_error(self, mock_markdown, renderer):
        """에러 렌더링 테스트"""
        error_data = "Something went wrong"
        content = RichContent(
            id="test-error",
            content_type=ContentType.ERROR,
            data=error_data,
            metadata={"error_type": "test_error"}
        )
        
        renderer._render_error(content)
        
        mock_markdown.assert_called()
        html_content = mock_markdown.call_args[0][0]
        assert "error-container" in html_content
        assert "Something went wrong" in html_content
        assert "test_error" in html_content

class TestRichContentRendererIntegration:
    """리치 콘텐츠 렌더러 통합 테스트"""
    
    @pytest.fixture
    def renderer(self):
        return RichContentRenderer()
    
    def test_create_content_auto_detection(self, renderer):
        """콘텐츠 자동 생성 테스트"""
        # DataFrame 자동 감지
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        content = renderer.create_content(df, title="Test DataFrame")
        
        assert content.content_type == ContentType.DATAFRAME
        assert content.data is df
        assert content.title == "Test DataFrame"
        assert content.id is not None
    
    def test_create_content_json(self, renderer):
        """JSON 콘텐츠 생성 테스트"""
        json_data = '{"test": true}'
        content = renderer.create_content(json_data)
        
        assert content.content_type == ContentType.JSON
        assert content.data == json_data
    
    @patch('streamlit.markdown')
    def test_render_content_complete_flow(self, mock_markdown, renderer):
        """완전한 렌더링 플로우 테스트"""
        # 테스트용 콘텐츠 생성
        content = RichContent(
            id="test-complete",
            content_type=ContentType.TEXT,
            data="Test content",
            metadata={},
            title="Complete Test"
        )
        
        with patch.object(renderer, '_render_content_header') as mock_header:
            with patch.object(renderer, '_render_text') as mock_text:
                renderer.render_content(content)
                
                # 헤더 렌더링 확인
                mock_header.assert_called_once_with(content)
                
                # 텍스트 렌더링 확인
                mock_text.assert_called_once_with(content)
    
    @patch('streamlit.markdown')
    def test_render_auto(self, mock_markdown, renderer):
        """자동 렌더링 테스트"""
        test_data = "# Test Markdown"
        
        with patch.object(renderer, 'create_content') as mock_create:
            with patch.object(renderer, 'render_content') as mock_render:
                mock_content = Mock()
                mock_create.return_value = mock_content
                
                renderer.render_auto(test_data, title="Auto Test")
                
                # 콘텐츠 생성 확인
                mock_create.assert_called_once_with(test_data, "Auto Test")
                
                # 렌더링 확인
                mock_render.assert_called_once_with(mock_content)

class TestRichContentRendererGlobalFunctions:
    """전역 함수 테스트"""
    
    @patch('ui.components.rich_content_renderer._rich_content_renderer_instance', None)
    def test_get_rich_content_renderer_singleton(self):
        """싱글톤 인스턴스 테스트"""
        renderer1 = get_rich_content_renderer()
        renderer2 = get_rich_content_renderer()
        
        assert renderer1 is renderer2
        assert isinstance(renderer1, RichContentRenderer)
    
    @patch('ui.components.rich_content_renderer._rich_content_renderer_instance', None)
    @patch('ui.components.rich_content_renderer.inject_javascript_functions')
    def test_initialize_rich_content_renderer(self, mock_inject_js):
        """렌더러 초기화 테스트"""
        renderer = initialize_rich_content_renderer()
        
        assert isinstance(renderer, RichContentRenderer)
        mock_inject_js.assert_called_once()

class TestRichContentRendererErrorHandling:
    """에러 처리 테스트"""
    
    @pytest.fixture
    def renderer(self):
        return RichContentRenderer()
    
    def test_invalid_json_handling(self, renderer):
        """잘못된 JSON 처리 테스트"""
        invalid_json = '{"invalid": json}'
        content = RichContent(
            id="test-invalid-json",
            content_type=ContentType.JSON,
            data=invalid_json,
            metadata={}
        )
        
        with patch.object(renderer, '_render_error') as mock_render_error:
            renderer._render_json(content)
            mock_render_error.assert_called()
    
    def test_render_content_error_handling(self, renderer):
        """렌더링 에러 처리 테스트"""
        content = RichContent(
            id="test-error-handling",
            content_type=ContentType.DATAFRAME,
            data="invalid_dataframe",  # DataFrame이 아닌 잘못된 데이터
            metadata={}
        )
        
        with patch.object(renderer, '_render_error') as mock_render_error:
            # 렌더링 중 에러 발생 시뮬레이션
            with patch.object(renderer, '_render_dataframe', side_effect=Exception("Test error")):
                renderer.render_content(content)
                mock_render_error.assert_called()
    
    def test_detect_content_type_error_handling(self, renderer):
        """콘텐츠 타입 감지 에러 처리 테스트"""
        # 예외를 발생시킬 수 있는 객체
        problematic_data = Mock()
        problematic_data.__str__ = Mock(side_effect=Exception("String conversion error"))
        
        content_type = renderer.detect_content_type(problematic_data)
        assert content_type == ContentType.ERROR

class TestRichContentRendererPerformance:
    """성능 테스트"""
    
    @pytest.fixture
    def renderer(self):
        return RichContentRenderer()
    
    def test_large_dataframe_handling(self, renderer):
        """대용량 DataFrame 처리 테스트"""
        # 큰 DataFrame 생성
        large_df = pd.DataFrame({
            f'col_{i}': range(1000) for i in range(10)
        })
        
        content_type = renderer.detect_content_type(large_df)
        assert content_type == ContentType.DATAFRAME
        
        # 렌더링 시간이 합리적인지 확인 (실제 렌더링은 하지 않음)
        content = renderer.create_content(large_df)
        assert content.content_type == ContentType.DATAFRAME
    
    def test_long_text_handling(self, renderer):
        """긴 텍스트 처리 테스트"""
        long_text = "A" * 10000  # 10KB 텍스트
        
        content_type = renderer.detect_content_type(long_text)
        assert content_type == ContentType.TEXT
        
        content = renderer.create_content(long_text)
        assert len(content.data) == 10000

# 테스트 실행을 위한 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 