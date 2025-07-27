"""
아티팩트 렌더링 테스트

Task 4.2.1: 아티팩트 렌더링 테스트 - 모든 아티팩트 타입 렌더링 검증
시각적 회귀 테스트 및 브라우저 호환성 테스트

테스트 시나리오:
1. Plotly 차트 렌더링 테스트
2. DataFrame 테이블 렌더링 테스트
3. 이미지 렌더링 테스트
4. 코드 블록 렌더링 테스트
5. 텍스트/마크다운 렌더링 테스트
6. 다중 아티팩트 동시 렌더링 테스트
7. 반응형 레이아웃 테스트
8. 다운로드 기능 테스트
9. 에러 상태 렌더링 테스트
10. 성능 측정 테스트
"""

import unittest
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import base64
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sys
import os
from unittest.mock import MagicMock, patch

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.ui.real_time_artifact_renderer import RealTimeArtifactRenderer, ArtifactType
from modules.ui.artifact_display_integration import ArtifactDisplayIntegration
from modules.artifacts.a2a_artifact_extractor import ArtifactInfo

class TestArtifactRendering(unittest.TestCase):
    """아티팩트 렌더링 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.renderer = RealTimeArtifactRenderer()
        self.display_integration = ArtifactDisplayIntegration()
        
        # 테스트 아티팩트 데이터
        self.test_artifacts = self._create_test_artifacts()
    
    def _create_test_artifacts(self) -> List[ArtifactInfo]:
        """테스트용 아티팩트 생성"""
        
        artifacts = []
        
        # 1. Plotly 차트 아티팩트
        plotly_data = {
            "data": [
                {
                    "x": ["Jan", "Feb", "Mar", "Apr", "May"],
                    "y": [20, 25, 30, 35, 40],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Sales"
                },
                {
                    "x": ["Jan", "Feb", "Mar", "Apr", "May"],
                    "y": [15, 20, 25, 30, 35],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Costs"
                }
            ],
            "layout": {
                "title": "Monthly Sales vs Costs",
                "xaxis": {"title": "Month"},
                "yaxis": {"title": "Amount ($K)"},
                "hovermode": "x unified"
            }
        }
        
        artifacts.append(ArtifactInfo(
            artifact_id="chart_001",
            type=ArtifactType.PLOTLY_CHART,
            title="월별 매출 대비 비용",
            data=plotly_data,
            agent_id="visualization_agent",
            created_at=datetime.now(),
            metadata={
                "chart_type": "line",
                "interactive": True,
                "data_points": 10
            }
        ))
        
        # 2. DataFrame 아티팩트
        df_data = {
            "columns": ["Product", "Q1", "Q2", "Q3", "Q4", "Total"],
            "data": [
                ["Product A", 100, 120, 110, 130, 460],
                ["Product B", 80, 90, 95, 100, 365],
                ["Product C", 60, 70, 75, 80, 285],
                ["Product D", 40, 45, 50, 55, 190]
            ],
            "index": [0, 1, 2, 3]
        }
        
        artifacts.append(ArtifactInfo(
            artifact_id="table_001",
            type=ArtifactType.DATAFRAME,
            title="분기별 제품 판매 현황",
            data=df_data,
            agent_id="analysis_agent",
            created_at=datetime.now(),
            metadata={
                "rows": 4,
                "columns": 6,
                "sortable": True,
                "filterable": True
            }
        ))
        
        # 3. 이미지 아티팩트 (Base64 인코딩된 1x1 투명 PNG)
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        artifacts.append(ArtifactInfo(
            artifact_id="image_001",
            type=ArtifactType.IMAGE,
            title="분석 결과 차트",
            data=image_b64,
            agent_id="visualization_agent",
            created_at=datetime.now(),
            metadata={
                "format": "PNG",
                "width": 800,
                "height": 600,
                "file_size": "125KB"
            }
        ))
        
        # 4. 코드 아티팩트
        code_content = """
import pandas as pd
import plotly.express as px

def analyze_sales_data(df):
    \"\"\"매출 데이터 분석 함수\"\"\"
    
    # 기본 통계
    summary = df.describe()
    
    # 트렌드 분석
    trend = df['sales'].rolling(window=3).mean()
    
    # 시각화
    fig = px.line(df, x='date', y='sales', 
                  title='Sales Trend Analysis')
    
    return {
        'summary': summary,
        'trend': trend,
        'chart': fig
    }

# 사용 예시
if __name__ == "__main__":
    data = pd.read_csv('sales_data.csv')
    results = analyze_sales_data(data)
    print("Analysis completed successfully!")
"""
        
        artifacts.append(ArtifactInfo(
            artifact_id="code_001",
            type=ArtifactType.CODE,
            title="매출 분석 스크립트",
            data=code_content,
            agent_id="code_agent",
            created_at=datetime.now(),
            metadata={
                "language": "python",
                "lines": 28,
                "functions": ["analyze_sales_data"],
                "imports": ["pandas", "plotly.express"]
            }
        ))
        
        # 5. 텍스트/마크다운 아티팩트
        markdown_content = """
# 데이터 분석 보고서

## 📊 분석 개요

이번 분석에서는 **2024년 Q1-Q4 매출 데이터**를 종합적으로 검토하였습니다.

### 주요 발견사항

1. **매출 증가 트렌드**: 전년 대비 15% 증가
2. **지역별 성과**: 북미 지역이 가장 높은 성장률 기록
3. **제품별 분석**: Product A가 전체 매출의 35% 차지

### 상세 분석 결과

| 지표 | Q1 | Q2 | Q3 | Q4 | 증감률 |
|------|----|----|----|----|--------|
| 매출 | $2.1M | $2.3M | $2.5M | $2.7M | +28.6% |
| 비용 | $1.5M | $1.6M | $1.7M | $1.8M | +20.0% |
| 순이익 | $0.6M | $0.7M | $0.8M | $0.9M | +50.0% |

### 🎯 권장사항

- **마케팅 투자 확대**: 성과가 좋은 북미 지역에 추가 투자
- **제품 포트폴리오 최적화**: Product A의 생산 능력 확장
- **비용 관리**: 운영 효율성 개선을 통한 마진 증대

### 📈 다음 단계

1. 상세 시장 조사 실시
2. 투자 계획 수립
3. 실행 로드맵 작성

> **참고**: 이 분석은 내부 데이터를 기반으로 하며, 외부 시장 요인은 별도 고려가 필요합니다.
"""
        
        artifacts.append(ArtifactInfo(
            artifact_id="text_001",
            type=ArtifactType.TEXT,
            title="Q4 매출 분석 보고서",
            data=markdown_content,
            agent_id="report_agent",
            created_at=datetime.now(),
            metadata={
                "format": "markdown",
                "word_count": 180,
                "sections": 5,
                "tables": 1,
                "lists": 2
            }
        ))
        
        return artifacts
    
    @patch('streamlit.plotly_chart')
    def test_plotly_chart_rendering(self, mock_plotly_chart):
        """Plotly 차트 렌더링 테스트"""
        
        chart_artifact = self.test_artifacts[0]  # Plotly 차트
        
        # 렌더링 실행
        success = self.renderer.render_artifact(chart_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_plotly_chart.assert_called_once()
        
        # 호출 인수 검증
        args, kwargs = mock_plotly_chart.call_args
        self.assertIn('data', args[0])
        self.assertIn('layout', args[0])
        self.assertEqual(kwargs.get('use_container_width'), True)
    
    @patch('streamlit.dataframe')
    def test_dataframe_rendering(self, mock_dataframe):
        """DataFrame 테이블 렌더링 테스트"""
        
        table_artifact = self.test_artifacts[1]  # DataFrame
        
        # 렌더링 실행
        success = self.renderer.render_artifact(table_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_dataframe.assert_called_once()
        
        # DataFrame 구조 검증
        args, kwargs = mock_dataframe.call_args
        df = args[0]
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 6)  # Product, Q1, Q2, Q3, Q4, Total
        self.assertEqual(len(df), 4)  # 4 products
        self.assertEqual(kwargs.get('use_container_width'), True)
    
    @patch('streamlit.image')
    def test_image_rendering(self, mock_image):
        """이미지 렌더링 테스트"""
        
        image_artifact = self.test_artifacts[2]  # Image
        
        # 렌더링 실행
        success = self.renderer.render_artifact(image_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_image.assert_called_once()
        
        # 이미지 인수 검증
        args, kwargs = mock_image.call_args
        self.assertIsNotNone(args[0])  # 이미지 데이터
        self.assertEqual(kwargs.get('use_column_width'), True)
    
    @patch('streamlit.code')
    def test_code_rendering(self, mock_code):
        """코드 블록 렌더링 테스트"""
        
        code_artifact = self.test_artifacts[3]  # Code
        
        # 렌더링 실행
        success = self.renderer.render_artifact(code_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_code.assert_called_once()
        
        # 코드 인수 검증
        args, kwargs = mock_code.call_args
        self.assertIn("import pandas", args[0])
        self.assertEqual(kwargs.get('language'), 'python')
    
    @patch('streamlit.markdown')
    def test_text_rendering(self, mock_markdown):
        """텍스트/마크다운 렌더링 테스트"""
        
        text_artifact = self.test_artifacts[4]  # Text
        
        # 렌더링 실행
        success = self.renderer.render_artifact(text_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_markdown.assert_called_once()
        
        # 마크다운 내용 검증
        args, kwargs = mock_markdown.call_args
        markdown_content = args[0]
        
        self.assertIn("# 데이터 분석 보고서", markdown_content)
        self.assertIn("📊 분석 개요", markdown_content)
        self.assertIn("주요 발견사항", markdown_content)
    
    @patch('streamlit.container')
    @patch('streamlit.plotly_chart')
    @patch('streamlit.dataframe')
    def test_multiple_artifacts_rendering(self, mock_dataframe, mock_plotly_chart, mock_container):
        """다중 아티팩트 동시 렌더링 테스트"""
        
        # 여러 아티팩트 동시 렌더링
        artifacts_to_render = self.test_artifacts[:3]  # 차트, 테이블, 이미지
        
        results = []
        for artifact in artifacts_to_render:
            success = self.renderer.render_artifact(artifact)
            results.append(success)
        
        # 모든 아티팩트 렌더링 성공 확인
        self.assertTrue(all(results))
        
        # 각 렌더링 함수 호출 확인
        mock_plotly_chart.assert_called_once()
        mock_dataframe.assert_called_once()
    
    def test_artifact_validation_before_rendering(self):
        """렌더링 전 아티팩트 검증 테스트"""
        
        # 잘못된 아티팩트 생성
        invalid_artifact = ArtifactInfo(
            artifact_id="invalid_001",
            type=ArtifactType.PLOTLY_CHART,
            title="Invalid Chart",
            data={"invalid": "structure"},  # 잘못된 구조
            agent_id="test_agent",
            created_at=datetime.now(),
            metadata={}
        )
        
        # 렌더링 시도
        success = self.renderer.render_artifact(invalid_artifact)
        
        # 실패해야 함
        self.assertFalse(success)
    
    @patch('streamlit.download_button')
    def test_download_functionality(self, mock_download_button):
        """다운로드 기능 테스트"""
        
        chart_artifact = self.test_artifacts[0]  # Plotly 차트
        
        # 다운로드 버튼 렌더링
        success = self.renderer.render_download_button(chart_artifact)
        
        # 검증
        self.assertTrue(success)
        mock_download_button.assert_called_once()
        
        # 다운로드 인수 검증
        args, kwargs = mock_download_button.call_args
        self.assertIn("label", kwargs)
        self.assertIn("data", kwargs)
        self.assertIn("file_name", kwargs)
        self.assertIn("mime", kwargs)
    
    def test_rendering_performance(self):
        """렌더링 성능 테스트"""
        
        chart_artifact = self.test_artifacts[0]
        
        # 성능 측정
        start_time = time.time()
        
        with patch('streamlit.plotly_chart'):
            success = self.renderer.render_artifact(chart_artifact)
        
        end_time = time.time()
        rendering_time = end_time - start_time
        
        # 성능 기준: 100ms 이내
        self.assertTrue(success)
        self.assertLess(rendering_time, 0.1)  # 100ms
    
    def test_large_data_rendering(self):
        """대용량 데이터 렌더링 테스트"""
        
        # 대용량 DataFrame 생성
        large_df_data = {
            "columns": [f"col_{i}" for i in range(20)],
            "data": [[f"value_{i}_{j}" for j in range(20)] for i in range(1000)],
            "index": list(range(1000))
        }
        
        large_artifact = ArtifactInfo(
            artifact_id="large_table",
            type=ArtifactType.DATAFRAME,
            title="Large Dataset",
            data=large_df_data,
            agent_id="analysis_agent",
            created_at=datetime.now(),
            metadata={"rows": 1000, "columns": 20}
        )
        
        # 렌더링 성능 측정
        start_time = time.time()
        
        with patch('streamlit.dataframe'):
            success = self.renderer.render_artifact(large_artifact)
        
        end_time = time.time()
        rendering_time = end_time - start_time
        
        # 대용량 데이터도 1초 이내 처리
        self.assertTrue(success)
        self.assertLess(rendering_time, 1.0)
    
    @patch('streamlit.error')
    def test_error_state_rendering(self, mock_error):
        """에러 상태 렌더링 테스트"""
        
        # None 데이터로 에러 상황 시뮬레이션
        error_artifact = ArtifactInfo(
            artifact_id="error_001",
            type=ArtifactType.PLOTLY_CHART,
            title="Error Chart",
            data=None,
            agent_id="test_agent",
            created_at=datetime.now(),
            metadata={}
        )
        
        # 에러 렌더링
        success = self.renderer.render_artifact(error_artifact)
        
        # 실패하고 에러 메시지 표시
        self.assertFalse(success)
        mock_error.assert_called_once()
    
    def test_responsive_layout(self):
        """반응형 레이아웃 테스트"""
        
        chart_artifact = self.test_artifacts[0]
        
        # 다양한 컨테이너 크기에서 테스트
        container_widths = [300, 600, 900, 1200]
        
        for width in container_widths:
            with patch('streamlit.plotly_chart') as mock_chart:
                # 컨테이너 크기 설정 시뮬레이션
                with patch.object(self.renderer, '_get_container_width', return_value=width):
                    success = self.renderer.render_artifact(chart_artifact)
                
                self.assertTrue(success)
                
                # use_container_width 옵션 확인
                args, kwargs = mock_chart.call_args
                self.assertEqual(kwargs.get('use_container_width'), True)

class TestArtifactDisplayIntegration(unittest.TestCase):
    """아티팩트 표시 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.display_integration = ArtifactDisplayIntegration()
        self.test_artifacts = [
            ArtifactInfo(
                artifact_id="integration_test",
                type=ArtifactType.PLOTLY_CHART,
                title="Integration Test Chart",
                data={"data": [], "layout": {"title": "Test"}},
                agent_id="test_agent",
                created_at=datetime.now(),
                metadata={}
            )
        ]
    
    @patch('streamlit.container')
    @patch('streamlit.columns')
    def test_artifact_layout_integration(self, mock_columns, mock_container):
        """아티팩트 레이아웃 통합 테스트"""
        
        # 컨테이너 및 컬럼 모킹
        mock_container.return_value.__enter__ = MagicMock()
        mock_container.return_value.__exit__ = MagicMock()
        mock_columns.return_value = [MagicMock(), MagicMock()]
        
        # 레이아웃 생성
        success = self.display_integration.create_artifact_layout(
            self.test_artifacts,
            layout_type="grid"
        )
        
        # 검증
        self.assertTrue(success)
        mock_container.assert_called()
        mock_columns.assert_called()
    
    @patch('streamlit.empty')
    def test_real_time_update(self, mock_empty):
        """실시간 업데이트 테스트"""
        
        # 빈 플레이스홀더 모킹
        placeholder = MagicMock()
        mock_empty.return_value = placeholder
        
        # 실시간 업데이트 시뮬레이션
        success = self.display_integration.update_artifacts_realtime(
            self.test_artifacts,
            placeholder
        )
        
        # 검증
        self.assertTrue(success)
        mock_empty.assert_called_once()
    
    def test_artifact_filtering(self):
        """아티팩트 필터링 테스트"""
        
        # 다양한 타입의 아티팩트 생성
        mixed_artifacts = [
            ArtifactInfo(
                artifact_id="chart1",
                type=ArtifactType.PLOTLY_CHART,
                title="Chart 1",
                data={},
                agent_id="agent1",
                created_at=datetime.now(),
                metadata={}
            ),
            ArtifactInfo(
                artifact_id="table1",
                type=ArtifactType.DATAFRAME,
                title="Table 1",
                data={},
                agent_id="agent1",
                created_at=datetime.now(),
                metadata={}
            ),
            ArtifactInfo(
                artifact_id="chart2",
                type=ArtifactType.PLOTLY_CHART,
                title="Chart 2",
                data={},
                agent_id="agent2",
                created_at=datetime.now(),
                metadata={}
            )
        ]
        
        # 차트만 필터링
        chart_artifacts = self.display_integration.filter_artifacts(
            mixed_artifacts,
            artifact_type=ArtifactType.PLOTLY_CHART
        )
        
        # 검증
        self.assertEqual(len(chart_artifacts), 2)
        for artifact in chart_artifacts:
            self.assertEqual(artifact.type, ArtifactType.PLOTLY_CHART)
        
        # 에이전트별 필터링
        agent1_artifacts = self.display_integration.filter_artifacts(
            mixed_artifacts,
            agent_id="agent1"
        )
        
        # 검증
        self.assertEqual(len(agent1_artifacts), 2)
        for artifact in agent1_artifacts:
            self.assertEqual(artifact.agent_id, "agent1")
    
    def test_artifact_sorting(self):
        """아티팩트 정렬 테스트"""
        
        # 시간이 다른 아티팩트들 생성
        base_time = datetime.now()
        time_ordered_artifacts = [
            ArtifactInfo(
                artifact_id=f"artifact_{i}",
                type=ArtifactType.TEXT,
                title=f"Artifact {i}",
                data="content",
                agent_id="test_agent",
                created_at=base_time.replace(minute=i),
                metadata={}
            )
            for i in [3, 1, 4, 2]  # 무작위 순서
        ]
        
        # 시간순 정렬
        sorted_artifacts = self.display_integration.sort_artifacts(
            time_ordered_artifacts,
            sort_by="created_at",
            ascending=True
        )
        
        # 검증
        self.assertEqual(len(sorted_artifacts), 4)
        
        # 시간 순서 확인
        for i in range(len(sorted_artifacts) - 1):
            self.assertLessEqual(
                sorted_artifacts[i].created_at,
                sorted_artifacts[i + 1].created_at
            )

class TestVisualRegressionPrevention(unittest.TestCase):
    """시각적 회귀 방지 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.renderer = RealTimeArtifactRenderer()
    
    def test_consistent_chart_styling(self):
        """일관된 차트 스타일링 테스트"""
        
        # 동일한 구조의 차트 데이터
        chart_data = {
            "data": [{
                "x": [1, 2, 3],
                "y": [1, 4, 2],
                "type": "scatter"
            }],
            "layout": {"title": "Test Chart"}
        }
        
        artifact = ArtifactInfo(
            artifact_id="style_test",
            type=ArtifactType.PLOTLY_CHART,
            title="Style Test",
            data=chart_data,
            agent_id="test_agent",
            created_at=datetime.now(),
            metadata={}
        )
        
        # 스타일 적용 테스트
        with patch('streamlit.plotly_chart') as mock_chart:
            success = self.renderer.render_artifact(artifact)
            
            # 성공 확인
            self.assertTrue(success)
            
            # 차트 구성 확인
            args, kwargs = mock_chart.call_args
            chart_config = args[0]
            
            # 기본 스타일 요소 확인
            self.assertIn("data", chart_config)
            self.assertIn("layout", chart_config)
    
    def test_table_formatting_consistency(self):
        """테이블 포맷팅 일관성 테스트"""
        
        table_data = {
            "columns": ["A", "B", "C"],
            "data": [[1, 2, 3], [4, 5, 6]],
            "index": [0, 1]
        }
        
        artifact = ArtifactInfo(
            artifact_id="table_format_test",
            type=ArtifactType.DATAFRAME,
            title="Table Format Test",
            data=table_data,
            agent_id="test_agent",
            created_at=datetime.now(),
            metadata={}
        )
        
        with patch('streamlit.dataframe') as mock_dataframe:
            success = self.renderer.render_artifact(artifact)
            
            self.assertTrue(success)
            
            # DataFrame 포맷 확인
            args, kwargs = mock_dataframe.call_args
            df = args[0]
            
            # 컬럼명과 데이터 일치 확인
            self.assertEqual(list(df.columns), ["A", "B", "C"])
            self.assertEqual(len(df), 2)

if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)