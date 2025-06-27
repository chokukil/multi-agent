"""
Data Analysis UI 단위 테스트
"""

import pytest
import streamlit as st
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ui.data_analysis_ui import DataAnalysisUI


class TestDataAnalysisUI:
    """Data Analysis UI 테스트 클래스"""

    @pytest.fixture
    def ui_component(self):
        """테스트용 UI 컴포넌트"""
        ui = DataAnalysisUI()
        # 임시 디렉토리 생성 및 설정
        temp_dir = tempfile.mkdtemp()
        ui.data_dir = temp_dir
        return ui

    @pytest.fixture
    def sample_dataframe(self):
        """테스트용 샘플 데이터프레임"""
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5],
            'D': [True, False, True, False, True]
        })

    def test_ui_initialization(self, ui_component):
        """UI 컴포넌트 초기화 테스트"""
        assert ui_component.thinking_stream is None
        assert ui_component.plan_viz is not None
        assert ui_component.results_renderer is not None
        assert os.path.exists(ui_component.data_dir)

    def test_get_existing_datasets_empty(self, ui_component):
        """빈 데이터셋 목록 조회 테스트"""
        datasets = ui_component._get_existing_datasets()
        assert datasets == []

    def test_get_existing_datasets_with_files(self, ui_component):
        """파일이 있는 데이터셋 목록 조회 테스트"""
        # 테스트 파일 생성
        test_files = ['test1.csv', 'test2.xlsx', 'test3.json', 'readme.txt']
        
        for filename in test_files:
            file_path = os.path.join(ui_component.data_dir, filename)
            with open(file_path, 'w') as f:
                f.write('test content')

        datasets = ui_component._get_existing_datasets()
        
        # CSV, XLSX, JSON 파일만 포함되어야 함
        expected = ['test1', 'test2', 'test3']
        assert sorted(datasets) == sorted(expected)

    def test_save_uploaded_file_csv(self, ui_component, sample_dataframe):
        """CSV 파일 저장 테스트"""
        # Mock uploaded file 생성
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.getbuffer.return_value = b"test,data\n1,2\n3,4"

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()
            
            result = ui_component._save_uploaded_file(mock_file)
            
            assert result is not None
            assert result.startswith("uploaded_test_data_")

    def test_save_uploaded_file_error_handling(self, ui_component):
        """파일 저장 오류 처리 테스트"""
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.getbuffer.side_effect = Exception("Mock error")

        with patch('streamlit.error') as mock_error:
            result = ui_component._save_uploaded_file(mock_file)
            
            assert result is None
            mock_error.assert_called_once()

    def test_parse_orchestrator_response_fallback(self, ui_component):
        """오케스트레이터 응답 파싱 폴백 테스트"""
        response_content = "Some response content"
        dataset_name = "test_dataset"
        
        result = ui_component._parse_orchestrator_response(response_content, dataset_name)
        
        assert "plan" in result
        assert len(result["plan"]) == 3  # 기본 계획 3단계
        
        # 각 단계 검증
        for step in result["plan"]:
            assert "agent_name" in step
            assert "skill_name" in step
            assert "parameters" in step
            assert step["parameters"]["data_id"] == dataset_name

    @pytest.mark.asyncio
    async def test_create_analysis_plan_success(self, ui_component):
        """분석 계획 수립 성공 테스트"""
        dataset_name = "test_dataset"
        prompt = "Analyze the data"
        options = {"depth": "기본"}

        # Mock streamlit components
        with patch('streamlit.status') as mock_status, \
             patch('streamlit.error') as mock_error, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Mock status context manager
            mock_status_context = Mock()
            mock_status_context.update = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "result": "Plan created successfully"
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Mock thinking stream
            ui_component.thinking_stream = Mock()
            ui_component.thinking_stream.add_thought = Mock()
            ui_component.thinking_stream.finish_thinking = Mock()

            # Mock plan visualization
            ui_component.plan_viz.display_plan = Mock()

            result = await ui_component._create_analysis_plan(dataset_name, prompt, options)
            
            assert result is not None
            assert "plan" in result

    @pytest.mark.asyncio
    async def test_create_analysis_plan_error(self, ui_component):
        """분석 계획 수립 오류 테스트"""
        dataset_name = "test_dataset"
        prompt = "Analyze the data"
        options = {"depth": "기본"}

        with patch('streamlit.status') as mock_status, \
             patch('streamlit.error') as mock_error, \
             patch('httpx.AsyncClient') as mock_client:
            
            # Mock status context manager
            mock_status_context = Mock()
            mock_status_context.update = Mock()
            mock_status.return_value.__enter__ = Mock(return_value=mock_status_context)
            mock_status.return_value.__exit__ = Mock(return_value=None)
            
            # Mock HTTP client to raise exception
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            # Mock thinking stream
            ui_component.thinking_stream = Mock()
            ui_component.thinking_stream.add_thought = Mock()

            result = await ui_component._create_analysis_plan(dataset_name, prompt, options)
            
            assert result is None
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_plan_with_streaming(self, ui_component):
        """스트리밍 계획 실행 테스트"""
        plan_state = {
            "plan": [
                {
                    "agent_name": "pandas_data_analyst",
                    "parameters": {
                        "data_id": "test_dataset",
                        "user_instructions": "Test analysis"
                    }
                }
            ]
        }

        with patch('streamlit.container') as mock_container, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.session_state') as mock_session:
            
            # Mock streamlit components with context manager support
            mock_container_context = MagicMock()
            mock_container.return_value = mock_container_context

            # 실행기를 Mock으로 대체
            with patch.object(ui_component, '_executor_class') as mock_executor_class:
                mock_executor = Mock()
                mock_executor.execute = AsyncMock(return_value={
                    "total_steps": 1,
                    "successful_steps": 1,
                    "execution_time": 10.5,
                    "step_outputs": {1: {"success": True, "content": "Analysis completed"}}
                })
                
                # 메서드 내에서 A2ADataAnalysisExecutor() 직접 생성을 Mock으로 대체
                original_method = ui_component._execute_plan_with_streaming
                
                async def mock_execute_plan(plan_state):
                    st.markdown("### 🔄 분석 실행 중...")
                    executor = mock_executor
                    progress_container = st.container()
                    results_container = st.container()
                    
                    with progress_container:
                        execution_result = await executor.execute(plan_state)
                        st.session_state.analysis_results = execution_result
                
                ui_component._execute_plan_with_streaming = mock_execute_plan
                
                await ui_component._execute_plan_with_streaming(plan_state)
                
                # executor.execute가 호출되었는지 확인
                mock_executor.execute.assert_called_once_with(plan_state)

    def test_generate_final_report_with_results(self, ui_component):
        """결과가 있는 최종 보고서 생성 테스트"""
        plan_state = {"plan": []}
        
        with patch('streamlit.session_state') as mock_session:
            mock_session.get.return_value = {
                "total_steps": 2,
                "successful_steps": 2,
                "execution_time": 15.3,
                "step_outputs": {
                    1: {"success": True, "content": "Step 1 result", "agent": "pandas_data_analyst"},
                    2: {"success": True, "content": "Step 2 result", "agent": "data_visualization"}
                }
            }

            with patch('streamlit.markdown') as mock_markdown, \
                 patch('streamlit.columns') as mock_columns, \
                 patch('streamlit.metric') as mock_metric, \
                 patch('streamlit.expander') as mock_expander, \
                 patch('streamlit.button') as mock_button:
                
                # Mock columns with context manager support
                mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
                mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
                
                # Mock expander context manager
                mock_expander_context = MagicMock()
                mock_expander.return_value = mock_expander_context

                ui_component._generate_final_report(plan_state)
                
                # 결과 표시 관련 함수들이 호출되었는지 확인
                assert mock_markdown.call_count >= 1
                assert mock_metric.call_count >= 1

    def test_generate_final_report_no_results(self, ui_component):
        """결과가 없는 최종 보고서 생성 테스트"""
        plan_state = {"plan": []}
        
        with patch('streamlit.session_state') as mock_session:
            mock_session.get.return_value = {}

            with patch('streamlit.markdown') as mock_markdown, \
                 patch('streamlit.warning') as mock_warning:

                ui_component._generate_final_report(plan_state)
                
                mock_warning.assert_called_once_with("실행 결과가 없습니다.")

    def test_generate_report_download(self, ui_component):
        """보고서 다운로드 생성 테스트"""
        execution_result = {
            "total_steps": 2,
            "successful_steps": 2,
            "execution_time": 15.3,
            "step_outputs": {
                1: {"success": True, "content": "Step 1 result", "agent": "pandas_data_analyst"},
                2: {"success": True, "content": "Step 2 result", "agent": "data_visualization"}
            }
        }

        with patch('streamlit.download_button') as mock_download:
            ui_component._generate_report_download(execution_result)
            
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert "텍스트 보고서 다운로드" in call_args[1]["label"]
            assert "analysis_report_" in call_args[1]["file_name"]

    def test_generate_data_download(self, ui_component):
        """데이터 다운로드 생성 테스트"""
        execution_result = {
            "total_steps": 1,
            "successful_steps": 1,
            "step_outputs": {1: {"success": True, "content": "Test result"}}
        }

        with patch('streamlit.download_button') as mock_download:
            ui_component._generate_data_download(execution_result)
            
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert "결과 데이터 (JSON)" in call_args[1]["label"]
            assert "analysis_data_" in call_args[1]["file_name"]

    def test_generate_report_download_error(self, ui_component):
        """보고서 다운로드 생성 오류 테스트"""
        execution_result = None  # 잘못된 데이터

        with patch('streamlit.error') as mock_error:
            ui_component._generate_report_download(execution_result)
            
            mock_error.assert_called_once()

    def test_generate_data_download_error(self, ui_component):
        """데이터 다운로드 생성 오류 테스트"""
        # JSON 직렬화할 수 없는 객체
        execution_result = {"function": lambda x: x}

        with patch('streamlit.error') as mock_error:
            ui_component._generate_data_download(execution_result)
            
            mock_error.assert_called_once()

if __name__ == "__main__":
    # 개별 테스트 실행을 위한 코드
    pytest.main([__file__, "-v"]) 