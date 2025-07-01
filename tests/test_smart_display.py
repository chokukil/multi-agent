"""
Smart Display Manager 단위 테스트
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
from unittest.mock import Mock, patch

# 테스트 대상 import
try:
    from core.ui.smart_display import SmartDisplayManager, AccumulativeStreamContainer, render_plotly_chart_safe
    SMART_UI_AVAILABLE = True
except ImportError:
    SMART_UI_AVAILABLE = False
    pytestmark = pytest.mark.skip("Smart UI components not available")


class TestSmartDisplayManager:
    """SmartDisplayManager 단위 테스트"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        if SMART_UI_AVAILABLE:
            self.smart_display = SmartDisplayManager()
    
    def test_init(self):
        """초기화 테스트"""
        assert self.smart_display.chart_counter == 0
        assert isinstance(self.smart_display.unique_session_id, str)
        assert len(self.smart_display.unique_session_id) > 0
    
    def test_is_simple_content(self):
        """단순 콘텐츠 감지 테스트"""
        # 단순 타입들
        assert self.smart_display._is_simple_content(42)
        assert self.smart_display._is_simple_content(3.14)
        assert self.smart_display._is_simple_content(True)
        assert self.smart_display._is_simple_content(None)
        
        # 복잡한 타입들
        assert not self.smart_display._is_simple_content("text")
        assert not self.smart_display._is_simple_content([1, 2, 3])
        assert not self.smart_display._is_simple_content({"key": "value"})
    
    def test_is_code_content(self):
        """코드 콘텐츠 감지 테스트"""
        # Python 코드
        python_code = """
import pandas as pd
def hello():
    print("Hello World")
    return True
        """
        assert self.smart_display._is_code_content(python_code)
        
        # 일반 텍스트
        assert not self.smart_display._is_code_content("This is just text")
        assert not self.smart_display._is_code_content(123)
    
    def test_is_markdown_content(self):
        """마크다운 콘텐츠 감지 테스트"""
        # 마크다운 텍스트
        markdown_text = """
# 제목
**굵은 글씨**
- 리스트 항목
> 인용문
        """
        assert self.smart_display._is_markdown_content(markdown_text)
        
        # 일반 텍스트
        assert not self.smart_display._is_markdown_content("Plain text without markdown")
        assert not self.smart_display._is_markdown_content(123)
    
    def test_is_dataframe_content(self):
        """DataFrame 콘텐츠 감지 테스트"""
        # 실제 DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert self.smart_display._is_dataframe_content(df)
        
        # DataFrame 형태의 dict
        df_dict = {'columns': ['A', 'B'], 'data': [[1, 4], [2, 5], [3, 6]]}
        assert self.smart_display._is_dataframe_content(df_dict)
        
        # 일반 dict
        assert not self.smart_display._is_dataframe_content({'key': 'value'})
    
    def test_detect_language(self):
        """코드 언어 감지 테스트"""
        # Python
        python_code = "import pandas as pd\ndef func(): pass"
        assert self.smart_display._detect_language(python_code) == 'python'
        
        # JavaScript
        js_code = "function test() { const x = 10; }"
        assert self.smart_display._detect_language(js_code) == 'javascript'
        
        # SQL
        sql_code = "SELECT * FROM table WHERE id = 1"
        assert self.smart_display._detect_language(sql_code) == 'sql'
        
        # 일반 텍스트
        plain_text = "This is just plain text"
        assert self.smart_display._detect_language(plain_text) == 'text'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
