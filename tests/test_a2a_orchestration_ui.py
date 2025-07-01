"""
A2A Orchestration UI 단위 테스트
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# 테스트 대상 import
try:
    from core.ui.a2a_orchestration_ui import A2AOrchestrationDashboard, ComplexityLevel
    A2A_UI_AVAILABLE = True
except ImportError:
    A2A_UI_AVAILABLE = False
    unittest.skip("A2A Orchestration UI components not available")


class TestA2AOrchestrationDashboard(unittest.TestCase):
    """A2A Orchestration Dashboard 테스트"""
    
    def setUp(self):
        """테스트 셋업"""
        self.dashboard = A2AOrchestrationDashboard()
    
    def test_init(self):
        """초기화 테스트"""
        self.assertIsInstance(self.dashboard, A2AOrchestrationDashboard)
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.plotly_chart')
    def test_render_complexity_analyzer(self, mock_plotly_chart, mock_columns, mock_markdown):
        """복잡도 분석 렌더링 테스트"""
        user_input = "데이터를 분석하고 시각화해주세요"
        complexity_result = {
            'level': 'complex',
            'score': 0.8,
            'matched_patterns': ['데이터 분석', '시각화']
        }
        
        # Mock columns with context manager support
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_col3 = Mock()
        
        # Context manager support
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # 테스트 실행
        self.dashboard.render_complexity_analyzer(user_input, complexity_result)
        
        # 검증
        mock_columns.assert_called_once()
        mock_markdown.assert_called()
        mock_plotly_chart.assert_called()


class TestComplexityLevel(unittest.TestCase):
    """ComplexityLevel Enum 테스트"""
    
    def test_complexity_levels(self):
        """복잡도 레벨 테스트"""
        self.assertEqual(ComplexityLevel.SIMPLE.value, "simple")
        self.assertEqual(ComplexityLevel.SINGLE_AGENT.value, "single_agent")
        self.assertEqual(ComplexityLevel.COMPLEX.value, "complex")


if __name__ == '__main__':
    unittest.main()
