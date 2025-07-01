"""
Enhanced UI 통합 테스트
"""

import unittest
import time
from unittest.mock import Mock, patch

# 테스트 대상 import
try:
    from core.ui.smart_display import SmartDisplayManager, AccumulativeStreamContainer
    from core.ui.a2a_orchestration_ui import A2AOrchestrationDashboard
    ENHANCED_UI_AVAILABLE = True
except ImportError:
    ENHANCED_UI_AVAILABLE = False


class TestEnhancedUIIntegration(unittest.TestCase):
    """Enhanced UI 통합 테스트"""
    
    def setUp(self):
        """테스트 셋업"""
        if not ENHANCED_UI_AVAILABLE:
            self.skipTest("Enhanced UI components not available")
        
        self.smart_display = SmartDisplayManager()
        self.dashboard = A2AOrchestrationDashboard()
    
    def test_smart_display_and_dashboard_integration(self):
        """Smart Display와 Dashboard의 통합 테스트"""
        # 복잡도 분석 결과
        complexity_result = {
            'level': 'complex',
            'score': 0.85,
            'matched_patterns': ['데이터 분석', '머신러닝', '시각화']
        }
        
        # Mock 설정
        with patch('streamlit.write') as mock_write, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart') as mock_plotly:
            
            # Mock columns 설정
            mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
            for col in [mock_col1, mock_col2, mock_col3]:
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
            
            # 테스트 실행
            self.dashboard.render_complexity_analyzer("테스트 입력", complexity_result)
            
            # 검증
            mock_columns.assert_called()
            mock_markdown.assert_called()
            mock_plotly.assert_called()
    
    def test_artifact_rendering_workflow(self):
        """아티팩트 렌더링 워크플로우 테스트"""
        test_artifacts = [
            {"type": "text", "content": "일반 텍스트"},
            {"type": "code", "content": "def hello():\n    print('Hello')"},
            {"type": "markdown", "content": "# 제목\n**볼드 텍스트**"},
            {"type": "json", "content": {"key": "value", "number": 42}}
        ]
        
        with patch('streamlit.write') as mock_write, \
             patch('streamlit.code') as mock_code, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.json') as mock_json:
            
            for artifact in test_artifacts:
                self.smart_display.smart_display_content(artifact["content"])
            
            # 각 타입에 맞는 렌더링이 호출되었는지 확인
            self.assertTrue(mock_write.called or mock_code.called or 
                          mock_markdown.called or mock_json.called)
    
    def test_complexity_analysis_integration(self):
        """복잡도 분석 통합 테스트"""
        test_cases = [
            {"input": "안녕하세요", "expected_level": "simple"},
            {"input": "데이터를 분석해주세요", "expected_level": "single_agent"},
            {"input": "데이터 분석하고 시각화하고 보고서 작성해주세요", "expected_level": "complex"}
        ]
        
        for case in test_cases:
            complexity_result = {
                'level': case["expected_level"],
                'score': 0.5,
                'matched_patterns': []
            }
            
            with patch('streamlit.markdown'), \
                 patch('streamlit.columns') as mock_columns, \
                 patch('streamlit.plotly_chart'):
                
                # Mock columns 설정
                mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
                for col in [mock_col1, mock_col2, mock_col3]:
                    col.__enter__ = Mock(return_value=col)
                    col.__exit__ = Mock(return_value=None)
                mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
                
                # 오류 없이 실행되어야 함
                self.dashboard.render_complexity_analyzer(case["input"], complexity_result)
    
    @patch('streamlit.container')
    def test_streaming_container_workflow(self, mock_container):
        """스트리밍 컨테이너 워크플로우 테스트"""
        # Mock container 설정
        mock_container_instance = Mock()
        mock_container_instance.__enter__ = Mock(return_value=mock_container_instance)
        mock_container_instance.__exit__ = Mock(return_value=None)
        mock_container.return_value = mock_container_instance
        
        # Mock st.empty() 설정
        with patch('streamlit.empty') as mock_empty:
            mock_empty_instance = Mock()
            mock_empty_instance.container.return_value = mock_container_instance
            mock_empty.return_value = mock_empty_instance
            
            # 스트리밍 컨테이너 테스트
            stream_container = AccumulativeStreamContainer()
            
            # 청크 추가 테스트
            chunks = ["첫 번째 ", "두 번째 ", "세 번째 청크"]
            for chunk in chunks:
                stream_container.add_chunk(chunk)
            
            # 최종 내용 확인
            final_content = stream_container.finalize()
            expected_content = "".join(chunks)
            self.assertEqual(final_content, expected_content)


class TestEnhancedUIErrorHandling(unittest.TestCase):
    """Enhanced UI 오류 처리 테스트"""
    
    def test_smart_display_error_handling(self):
        """Smart Display 오류 처리 테스트"""
        if not ENHANCED_UI_AVAILABLE:
            self.skipTest("Enhanced UI components not available")
        
        smart_display = SmartDisplayManager()
        
        # 잘못된 데이터 타입에 대한 처리
        with patch('streamlit.write') as mock_write:
            # None 값 처리
            smart_display.smart_display_content(None)
            mock_write.assert_called_with(None)
            
            # 빈 문자열 처리
            smart_display.smart_display_content("")
            mock_write.assert_called_with("")
    
    def test_dashboard_error_handling(self):
        """Dashboard 오류 처리 테스트"""
        if not ENHANCED_UI_AVAILABLE:
            self.skipTest("Enhanced UI components not available")
        
        dashboard = A2AOrchestrationDashboard()
        
        # 빈 복잡도 결과 처리
        empty_complexity_result = {}
        
        with patch('streamlit.markdown'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.plotly_chart'):
            
            # Mock columns 설정
            mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
            for col in [mock_col1, mock_col2, mock_col3]:
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
            
            # 오류 없이 처리되어야 함
            try:
                dashboard.render_complexity_analyzer("test", empty_complexity_result)
            except Exception as e:
                self.fail(f"Dashboard should handle empty complexity result: {e}")


class TestEnhancedUIPerformance(unittest.TestCase):
    """Enhanced UI 성능 테스트"""
    
    def test_smart_display_performance(self):
        """Smart Display 성능 테스트"""
        if not ENHANCED_UI_AVAILABLE:
            self.skipTest("Enhanced UI components not available")
        
        smart_display = SmartDisplayManager()
        
        # 대량 데이터 처리 시뮬레이션 (마크다운 형태로)
        large_text = "# 제목\n" + "테스트 " * 1000
        
        with patch('streamlit.markdown') as mock_markdown:
            import time
            start_time = time.time()
            
            smart_display.smart_display_content(large_text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 1초 이내에 처리되어야 함
            self.assertLess(processing_time, 1.0)
            mock_markdown.assert_called_once()
    
    def test_streaming_container_performance(self):
        """스트리밍 컨테이너 성능 테스트"""
        if not ENHANCED_UI_AVAILABLE:
            self.skipTest("Enhanced UI components not available")
        
        with patch('streamlit.empty') as mock_empty, \
             patch('streamlit.container') as mock_container:
            
            # Mock 설정
            mock_empty_instance = Mock()
            mock_container_instance = Mock()
            mock_container_instance.__enter__ = Mock(return_value=mock_container_instance)
            mock_container_instance.__exit__ = Mock(return_value=None)
            mock_empty_instance.container.return_value = mock_container_instance
            mock_empty.return_value = mock_empty_instance
            mock_container.return_value = mock_container_instance
            
            stream_container = AccumulativeStreamContainer()
            
            # 대량 청크 처리 성능 테스트
            start_time = time.time()
            
            for i in range(100):
                stream_container.add_chunk(f"청크 {i} ")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 2초 이내에 처리되어야 함
            self.assertLess(processing_time, 2.0)


if __name__ == '__main__':
    unittest.main()
