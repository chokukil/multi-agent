#!/usr/bin/env python3
"""
🧪 Enhanced SessionDataManager Integration Tests

UserFileTracker와 통합된 SessionDataManager 단위 테스트
A2A SDK 0.2.9 호환성과 파일 관리 통합 기능을 pytest로 검증
"""

import pytest
import pandas as pd
import os
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.session_data_manager import SessionDataManager, FileMetadata, SessionMetadata
from core.data_manager import DataManager

class TestSessionDataManagerIntegration:
    """통합된 SessionDataManager 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """테스트용 샘플 DataFrame"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 92.0, 78.5, 96.5, 88.0],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """테스트용 SessionDataManager 인스턴스"""
        # 필요한 디렉토리 먼저 생성
        ai_ds_team_dir = temp_dir / "ai_ds_team" / "data"
        ai_ds_team_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 경로로 초기화
        with patch('core.session_data_manager.AI_DS_TEAM_DATA_DIR', ai_ds_team_dir):
            manager = SessionDataManager()
            
            # 임시 경로 설정
            manager._metadata_dir = temp_dir / "sessions_metadata"
            manager._metadata_dir.mkdir(exist_ok=True)
            
            return manager
    
    def test_initialization_with_user_file_tracker(self, session_manager):
        """UserFileTracker 통합 초기화 테스트"""
        # Then
        assert hasattr(session_manager, 'user_file_tracker')
        assert hasattr(session_manager, 'data_manager')
        assert isinstance(session_manager.data_manager, DataManager)
        assert session_manager._current_session_id is None
        assert session_manager._session_metadata == {}
    
    def test_create_session_with_data_integration(self, session_manager, sample_dataframe):
        """UserFileTracker 통합 세션 생성 테스트"""
        # Given
        data_id = "test_integration.csv"
        user_instructions = "통합 테스트용 데이터"
        
        # When
        session_id = session_manager.create_session_with_data(
            data_id=data_id,
            data=sample_dataframe,
            user_instructions=user_instructions
        )
        
        # Then
        assert session_id is not None
        assert session_id in session_manager._session_metadata
        assert session_manager._current_session_id == session_id
        
        # 세션 메타데이터 검증
        session_meta = session_manager._session_metadata[session_id]
        assert len(session_meta.uploaded_files) == 1
        assert session_meta.uploaded_files[0].data_id == data_id
        assert session_meta.active_file == data_id
        
        # DataManager에도 등록되었는지 확인
        assert data_id in session_manager.data_manager.list_dataframes()
    
    @patch('core.session_data_manager.USER_FILE_TRACKER_AVAILABLE', True)
    def test_create_session_with_user_file_tracker_success(self, session_manager, sample_dataframe):
        """UserFileTracker 등록 성공 시나리오 테스트"""
        # Given
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_tracker.register_uploaded_file.return_value = True
            
            data_id = "test_with_tracker.csv"
            user_instructions = "UserFileTracker 통합 테스트"
            
            # When
            session_id = session_manager.create_session_with_data(
                data_id=data_id,
                data=sample_dataframe,
                user_instructions=user_instructions
            )
            
            # Then
            assert session_id is not None
            mock_tracker.register_uploaded_file.assert_called_once_with(
                file_id=data_id,
                original_name=data_id,
                session_id=session_id,
                data=sample_dataframe,
                user_context=user_instructions
            )
    
    @patch('core.session_data_manager.USER_FILE_TRACKER_AVAILABLE', True)
    def test_create_session_with_user_file_tracker_failure(self, session_manager, sample_dataframe):
        """UserFileTracker 등록 실패 시나리오 테스트"""
        # Given
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_tracker.register_uploaded_file.return_value = False
            
            data_id = "test_tracker_fail.csv"
            user_instructions = "UserFileTracker 실패 테스트"
            
            # When
            session_id = session_manager.create_session_with_data(
                data_id=data_id,
                data=sample_dataframe,
                user_instructions=user_instructions
            )
            
            # Then - 세션은 여전히 생성되어야 함
            assert session_id is not None
            assert session_id in session_manager._session_metadata
            mock_tracker.register_uploaded_file.assert_called_once()
    
    def test_smart_file_selection_with_user_file_tracker(self, session_manager, sample_dataframe):
        """UserFileTracker 통합 스마트 파일 선택 테스트"""
        # Given
        session_id = session_manager.create_session_with_data(
            data_id="test_smart.csv",
            data=sample_dataframe,
            user_instructions="스마트 선택 테스트"
        )
        
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_tracker.get_file_for_a2a_request.return_value = ("/path/to/smart_file.csv", "UserFileTracker 선택")
            
            # When
            file_name, reason = session_manager.smart_file_selection(
                user_request="데이터를 분석해줘",
                session_id=session_id
            )
            
            # Then
            assert file_name == "smart_file.csv"
            assert "UserFileTracker" in reason
            mock_tracker.get_file_for_a2a_request.assert_called_once_with(
                user_request="데이터를 분석해줘",
                session_id=session_id
            )
    
    def test_smart_file_selection_fallback(self, session_manager, sample_dataframe):
        """UserFileTracker 실패 시 fallback 테스트"""
        # Given
        session_id = session_manager.create_session_with_data(
            data_id="fallback_test.csv",
            data=sample_dataframe,
            user_instructions="fallback 테스트"
        )
        
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_tracker.get_file_for_a2a_request.return_value = (None, "파일 없음")
            
            # When
            file_name, reason = session_manager.smart_file_selection(
                user_request="fallback_test.csv 파일을 분석해줘",
                session_id=session_id
            )
            
            # Then
            assert file_name == "fallback_test.csv"
            assert "파일명 패턴 일치" in reason
    
    def test_get_file_for_a2a_agent(self, session_manager, sample_dataframe):
        """A2A 에이전트용 파일 경로 반환 테스트"""
        # Given
        session_id = session_manager.create_session_with_data(
            data_id="a2a_test.csv",
            data=sample_dataframe,
            user_instructions="A2A 에이전트 테스트"
        )
        
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            expected_path = "/shared/path/a2a_test.csv"
            mock_tracker.get_file_for_a2a_request.return_value = (expected_path, "A2A 전용 선택")
            
            # When
            file_path, reason = session_manager.get_file_for_a2a_agent(
                user_request="a2a_test.csv를 분석해줘",
                session_id=session_id,
                agent_name="EDA Agent"
            )
            
            # Then
            assert file_path == expected_path
            assert reason == "A2A 전용 선택"
            mock_tracker.get_file_for_a2a_request.assert_called_once_with(
                user_request="a2a_test.csv를 분석해줘",
                session_id=session_id,
                agent_name="EDA Agent"
            )
    
    def test_get_file_for_a2a_agent_without_tracker(self, session_manager, sample_dataframe):
        """UserFileTracker 없이 A2A 에이전트용 파일 선택 테스트"""
        # Given
        session_manager.user_file_tracker = None
        
        session_id = session_manager.create_session_with_data(
            data_id="no_tracker_test.csv",
            data=sample_dataframe,
            user_instructions="tracker 없는 테스트"
        )
        
        # When
        file_path, reason = session_manager.get_file_for_a2a_agent(
            user_request="no_tracker_test.csv를 분석해줘",
            session_id=session_id
        )
        
        # Then
        assert file_path is None or "no_tracker_test" in file_path
        assert "A2A 에이전트용 파일을 찾을 수 없음" in reason or "파일명 패턴 일치" in reason
    
    def test_prepare_ai_ds_team_environment_with_shared_info(self, session_manager, sample_dataframe):
        """A2A 공유 정보 포함 환경 준비 테스트"""
        # Given
        session_id = session_manager.create_session_with_data(
            data_id="env_test.csv",
            data=sample_dataframe,
            user_instructions="환경 준비 테스트"
        )
        
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_files_info = [
                {
                    'file_id': 'env_test.csv',
                    'original_name': 'env_test.csv',
                    'shared_path': '/shared/env_test.csv',
                    'session_path': '/session/env_test.csv'
                }
            ]
            mock_tracker.get_session_files_info.return_value = mock_files_info
            
            # When
            env_info = session_manager.prepare_ai_ds_team_environment(session_id)
            
            # Then
            assert env_info['session_id'] == session_id
            assert 'shared_data_info' in env_info
            
            shared_info = env_info['shared_data_info']
            assert 'available_files' in shared_info
            assert 'shared_path' in shared_info
            assert shared_info['available_files'] == mock_files_info
            assert shared_info['shared_path'] == "a2a_ds_servers/artifacts/data/shared_dataframes"
    
    def test_domain_extraction(self, session_manager):
        """도메인 추출 기능 테스트"""
        # Test cases
        test_cases = [
            ("반도체 이온주입 데이터 분석", "semiconductor"),
            ("금융 투자 데이터 검토", "finance"),
            ("의료 환자 데이터 분석", "medical"),
            ("판매 고객 데이터 분석", "retail"),
            ("제조 생산 품질 분석", "manufacturing"),
            ("일반적인 데이터 분석", None)
        ]
        
        for user_request, expected_domain in test_cases:
            # When
            result = session_manager.extract_domain_from_request(user_request)
            
            # Then
            assert result == expected_domain, f"Failed for: {user_request}"
    
    def test_filename_pattern_extraction(self, session_manager):
        """파일명 패턴 추출 기능 테스트"""
        # Test cases
        test_cases = [
            ("data.csv 파일을 분석해줘", "data.csv"),
            ("분석용 dataset.xlsx 사용", "dataset.xlsx"),
            ("ion_implant 관련 분석", "ion_implant"),
            ("titanic 데이터셋으로 작업", "titanic"),
            ("일반적인 분석 요청", None)
        ]
        
        for user_request, expected_pattern in test_cases:
            # When
            result = session_manager.extract_filename_pattern(user_request)
            
            # Then
            assert result == expected_pattern, f"Failed for: {user_request}"
    
    def test_session_metadata_persistence(self, session_manager, sample_dataframe, temp_dir):
        """세션 메타데이터 영속성 테스트"""
        # Given
        data_id = "persistence_test.csv"
        user_instructions = "영속성 테스트"
        
        # When - 세션 생성
        session_id = session_manager.create_session_with_data(
            data_id=data_id,
            data=sample_dataframe,
            user_instructions=user_instructions
        )
        
        # 메타데이터 파일이 생성되었는지 확인
        metadata_file = session_manager._metadata_dir / f"session_{session_id}.json"
        assert metadata_file.exists()
        
        # 새로운 SessionDataManager 인스턴스로 로드 테스트
        new_manager = SessionDataManager()
        new_manager._metadata_dir = session_manager._metadata_dir
        new_manager._load_existing_sessions()
        
        # Then
        assert session_id in new_manager._session_metadata
        session_meta = new_manager._session_metadata[session_id]
        assert len(session_meta.uploaded_files) == 1
        assert session_meta.uploaded_files[0].data_id == data_id
    
    def test_multiple_sessions_management(self, session_manager, sample_dataframe):
        """다중 세션 관리 테스트"""
        # Given
        sessions_data = [
            ("session1_data.csv", "첫 번째 세션"),
            ("session2_data.xlsx", "두 번째 세션"),
            ("session3_data.json", "세 번째 세션")
        ]
        
        created_sessions = []
        
        # When - 다중 세션 생성
        for data_id, instructions in sessions_data:
            session_id = session_manager.create_session_with_data(
                data_id=data_id,
                data=sample_dataframe,
                user_instructions=instructions
            )
            created_sessions.append(session_id)
        
        # Then
        assert len(session_manager._session_metadata) == 3
        assert session_manager._current_session_id == created_sessions[-1]  # 마지막 생성된 세션이 활성
        
        # 각 세션의 독립성 확인
        for i, session_id in enumerate(created_sessions):
            session_meta = session_manager._session_metadata[session_id]
            expected_data_id = sessions_data[i][0]
            assert session_meta.active_file == expected_data_id
            assert len(session_meta.uploaded_files) == 1
    
    def test_session_context_retrieval(self, session_manager, sample_dataframe):
        """세션 컨텍스트 조회 테스트"""
        # Given
        data_id = "context_test.csv"
        user_instructions = "컨텍스트 테스트용 데이터"
        
        session_id = session_manager.create_session_with_data(
            data_id=data_id,
            data=sample_dataframe,
            user_instructions=user_instructions
        )
        
        # When
        context = session_manager.get_session_context(session_id)
        
        # Then
        assert context is not None
        assert context['user_instructions'] == user_instructions
        assert context['data_id'] == data_id
        assert 'created_at' in context
        assert 'file_path' in context
    
    def test_active_file_management(self, session_manager, sample_dataframe):
        """활성 파일 관리 테스트"""
        # Given
        session_id = session_manager.create_session_with_data(
            data_id="first_file.csv",
            data=sample_dataframe,
            user_instructions="첫 번째 파일"
        )
        
        # 같은 세션에 두 번째 파일 추가
        session_manager.create_session_with_data(
            data_id="second_file.csv",
            data=sample_dataframe,
            user_instructions="두 번째 파일",
            session_id=session_id
        )
        
        # When - 활성 파일 정보 조회
        active_file, reason = session_manager.get_active_file_info(session_id)
        
        # Then
        assert active_file == "second_file.csv"  # 마지막 파일이 활성
        assert "세션의 활성 파일" in reason
        
        # When - 활성 파일 변경
        session_manager.update_active_file("first_file.csv", session_id)
        active_file, reason = session_manager.get_active_file_info(session_id)
        
        # Then
        assert active_file == "first_file.csv"

class TestSessionDataManagerErrorHandling:
    """SessionDataManager 에러 처리 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """테스트용 SessionDataManager"""
        # 필요한 디렉토리 먼저 생성
        ai_ds_team_dir = temp_dir / "ai_ds_team" / "data"
        ai_ds_team_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('core.session_data_manager.AI_DS_TEAM_DATA_DIR', ai_ds_team_dir):
            manager = SessionDataManager()
            manager._metadata_dir = temp_dir / "sessions_metadata"
            manager._metadata_dir.mkdir(exist_ok=True)
            return manager
    
    def test_invalid_session_id_handling(self, session_manager):
        """잘못된 세션 ID 처리 테스트"""
        # When & Then
        assert session_manager.get_active_file_info("invalid_session") == (None, "세션을 찾을 수 없음")
        assert session_manager.get_session_files("invalid_session") == []
        assert session_manager.get_session_context("invalid_session") is None
    
    def test_empty_dataframe_handling(self, session_manager):
        """빈 DataFrame 처리 테스트"""
        # Given
        empty_df = pd.DataFrame()
        
        # When
        session_id = session_manager.create_session_with_data(
            data_id="empty.csv",
            data=empty_df,
            user_instructions="빈 데이터프레임 테스트"
        )
        
        # Then
        assert session_id is not None
        session_meta = session_manager._session_metadata[session_id]
        assert session_meta.uploaded_files[0].data_id == "empty.csv"
    
    @patch('core.session_data_manager.USER_FILE_TRACKER_AVAILABLE', True)
    def test_user_file_tracker_exception_handling(self, session_manager):
        """UserFileTracker 예외 처리 테스트"""
        # Given
        with patch.object(session_manager, 'user_file_tracker') as mock_tracker:
            mock_tracker.register_uploaded_file.side_effect = Exception("Tracker error")
            
            sample_df = pd.DataFrame({'test': [1, 2, 3]})
            
            # When - 예외가 발생해도 세션은 생성되어야 함
            session_id = session_manager.create_session_with_data(
                data_id="exception_test.csv",
                data=sample_df,
                user_instructions="예외 처리 테스트"
            )
            
            # Then
            assert session_id is not None
            assert session_id in session_manager._session_metadata

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 