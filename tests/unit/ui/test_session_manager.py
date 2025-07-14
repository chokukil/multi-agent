#!/usr/bin/env python3
"""
📚 세션 관리자 단위 테스트

세션 생성, 저장, 불러오기, 삭제 등 모든 세션 관리 기능을 pytest로 검증

Test Coverage:
- 세션 생성 및 메타데이터 관리
- 세션 저장/불러오기/삭제
- 세션 검색 및 필터링
- 태그 및 즐겨찾기 관리
- 백업 및 복구 기능
- UI 컴포넌트 렌더링
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# 테스트 대상 임포트
from ui.components.session_manager import (
    SessionManager, SessionManagerUI, SessionMetadata, SessionData,
    SessionStatus, SessionType, get_session_manager, get_session_manager_ui,
    initialize_session_manager
)

class TestSessionMetadata:
    """SessionMetadata 데이터 클래스 테스트"""
    
    def test_session_metadata_creation(self):
        """세션 메타데이터 생성 테스트"""
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        metadata = SessionMetadata(
            id=session_id,
            name="테스트 세션",
            description="테스트용 세션입니다",
            session_type=SessionType.CHAT,
            status=SessionStatus.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
            last_accessed=created_at,
            message_count=5,
            file_count=2,
            tags=["test", "demo"],
            is_favorite=True,
            size_bytes=1024
        )
        
        assert metadata.id == session_id
        assert metadata.name == "테스트 세션"
        assert metadata.session_type == SessionType.CHAT
        assert metadata.status == SessionStatus.ACTIVE
        assert metadata.message_count == 5
        assert metadata.file_count == 2
        assert metadata.tags == ["test", "demo"]
        assert metadata.is_favorite is True
    
    def test_session_metadata_to_dict(self):
        """메타데이터 딕셔너리 변환 테스트"""
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        metadata = SessionMetadata(
            id="test-id",
            name="테스트",
            description="설명",
            session_type=SessionType.DATA_ANALYSIS,
            status=SessionStatus.ACTIVE,
            created_at=created_at,
            updated_at=created_at,
            last_accessed=created_at,
            message_count=3,
            file_count=1,
            tags=["tag1"],
            is_favorite=False,
            size_bytes=512
        )
        
        result = metadata.to_dict()
        
        assert result["id"] == "test-id"
        assert result["session_type"] == "data_analysis"
        assert result["status"] == "active"
        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["message_count"] == 3
        assert result["is_favorite"] is False
    
    def test_session_metadata_from_dict(self):
        """딕셔너리에서 메타데이터 생성 테스트"""
        data = {
            "id": "test-id",
            "name": "테스트",
            "description": "설명",
            "session_type": "file_processing",
            "status": "archived",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:30:00",
            "last_accessed": "2024-01-01T13:00:00",
            "message_count": 10,
            "file_count": 5,
            "tags": ["important"],
            "is_favorite": True,
            "size_bytes": 2048,
            "version": "1.0"
        }
        
        metadata = SessionMetadata.from_dict(data)
        
        assert metadata.id == "test-id"
        assert metadata.session_type == SessionType.FILE_PROCESSING
        assert metadata.status == SessionStatus.ARCHIVED
        assert metadata.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert metadata.is_favorite is True

class TestSessionData:
    """SessionData 데이터 클래스 테스트"""
    
    def test_session_data_creation(self):
        """세션 데이터 생성 테스트"""
        metadata = SessionMetadata(
            id="test-id",
            name="테스트",
            description="",
            session_type=SessionType.CHAT,
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_accessed=datetime.now(),
            message_count=0,
            file_count=0,
            tags=[],
            is_favorite=False,
            size_bytes=0
        )
        
        session_data = SessionData(
            metadata=metadata,
            messages=[{"role": "user", "content": "안녕"}],
            files=[{"name": "test.txt", "path": "/path/to/test.txt"}],
            variables={"var1": "value1"},
            custom_data={"custom": "data"}
        )
        
        assert session_data.metadata == metadata
        assert len(session_data.messages) == 1
        assert len(session_data.files) == 1
        assert session_data.variables["var1"] == "value1"
        assert session_data.custom_data["custom"] == "data"

class TestSessionManager:
    """SessionManager 클래스 테스트"""
    
    @pytest.fixture
    def temp_sessions_dir(self):
        """임시 세션 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """테스트용 세션 관리자"""
        return SessionManager(sessions_dir=temp_sessions_dir)
    
    def test_session_manager_initialization(self, session_manager, temp_sessions_dir):
        """세션 관리자 초기화 테스트"""
        assert session_manager.sessions_dir == Path(temp_sessions_dir)
        assert session_manager.backup_dir.exists()
        assert session_manager.current_session_id is None
        assert isinstance(session_manager._session_cache, dict)
        assert isinstance(session_manager._metadata_cache, dict)
    
    def test_create_session(self, session_manager):
        """세션 생성 테스트"""
        session_id = session_manager.create_session(
            name="테스트 세션",
            session_type=SessionType.DATA_ANALYSIS,
            description="테스트용",
            tags=["test"]
        )
        
        assert session_id is not None
        uuid.UUID(session_id)  # 유효한 UUID 확인
        
        # 세션 파일 생성 확인
        session_file = session_manager.sessions_dir / f"session_{session_id}.json"
        assert session_file.exists()
        
        # 캐시 업데이트 확인
        assert session_id in session_manager._session_cache
        assert session_id in session_manager._metadata_cache
        
        # 현재 세션 설정 확인
        assert session_manager.current_session_id == session_id
    
    def test_create_session_auto_name(self, session_manager):
        """자동 이름 생성 세션 테스트"""
        session_id = session_manager.create_session()
        
        metadata = session_manager._metadata_cache[session_id]
        assert "대화" in metadata.name  # 기본 이름 패턴 확인
    
    def test_load_session(self, session_manager):
        """세션 불러오기 테스트"""
        # 세션 생성
        session_id = session_manager.create_session(name="로드 테스트")
        
        # 캐시 클리어
        session_manager._session_cache.clear()
        session_manager._metadata_cache.clear()
        
        # 세션 로드
        session_data = session_manager.load_session(session_id)
        
        assert session_data is not None
        assert session_data.metadata.id == session_id
        assert session_data.metadata.name == "로드 테스트"
        
        # 접근 시간 업데이트 확인
        assert session_data.metadata.last_accessed is not None
    
    def test_load_nonexistent_session(self, session_manager):
        """존재하지 않는 세션 로드 테스트"""
        fake_id = str(uuid.uuid4())
        session_data = session_manager.load_session(fake_id)
        assert session_data is None
    
    def test_save_session(self, session_manager):
        """세션 저장 테스트"""
        # 세션 생성
        session_id = session_manager.create_session(name="저장 테스트")
        session_data = session_manager.load_session(session_id)
        
        # 세션 데이터 수정
        session_data.messages.append({"role": "user", "content": "새 메시지"})
        session_data.files.append({"name": "new.txt", "path": "/new.txt"})
        
        # 저장
        result = session_manager.save_session(session_data)
        assert result is True
        
        # 메타데이터 업데이트 확인
        assert session_data.metadata.message_count == 1
        assert session_data.metadata.file_count == 1
        assert session_data.metadata.updated_at is not None
    
    def test_delete_session_soft(self, session_manager):
        """세션 소프트 삭제 테스트"""
        session_id = session_manager.create_session(name="삭제 테스트")
        
        # 소프트 삭제
        result = session_manager.delete_session(session_id, permanent=False)
        assert result is True
        
        # 파일은 존재하지만 상태가 삭제로 변경됨
        session_file = session_manager.sessions_dir / f"session_{session_id}.json"
        assert session_file.exists()
        
        # 메타데이터에서 삭제 상태 확인
        metadata = session_manager._metadata_cache[session_id]
        assert metadata.status == SessionStatus.DELETED
    
    def test_delete_session_permanent(self, session_manager):
        """세션 영구 삭제 테스트"""
        session_id = session_manager.create_session(name="영구 삭제 테스트")
        
        # 영구 삭제
        result = session_manager.delete_session(session_id, permanent=True)
        assert result is True
        
        # 파일 삭제 확인
        session_file = session_manager.sessions_dir / f"session_{session_id}.json"
        assert not session_file.exists()
        
        # 캐시에서 제거 확인
        assert session_id not in session_manager._session_cache
        assert session_id not in session_manager._metadata_cache
    
    def test_restore_session(self, session_manager):
        """세션 복구 테스트"""
        session_id = session_manager.create_session(name="복구 테스트")
        
        # 삭제 후 복구
        session_manager.delete_session(session_id, permanent=False)
        result = session_manager.restore_session(session_id)
        assert result is True
        
        # 상태 확인
        metadata = session_manager._metadata_cache[session_id]
        assert metadata.status == SessionStatus.ACTIVE
    
    def test_get_sessions_list(self, session_manager):
        """세션 목록 조회 테스트"""
        # 여러 세션 생성
        session1 = session_manager.create_session(name="세션1", session_type=SessionType.CHAT)
        session2 = session_manager.create_session(name="세션2", session_type=SessionType.DATA_ANALYSIS)
        session3 = session_manager.create_session(name="세션3", session_type=SessionType.CHAT)
        
        # 하나 삭제
        session_manager.delete_session(session3, permanent=False)
        
        # 전체 목록 (삭제된 것 제외)
        sessions = session_manager.get_sessions_list(include_deleted=False)
        assert len(sessions) == 2
        
        # 삭제된 것 포함
        sessions_all = session_manager.get_sessions_list(include_deleted=True)
        assert len(sessions_all) == 3
        
        # 타입별 필터링
        chat_sessions = session_manager.get_sessions_list(session_type=SessionType.CHAT)
        assert len([s for s in chat_sessions if s.session_type == SessionType.CHAT]) >= 1
        
        # 제한 개수
        limited_sessions = session_manager.get_sessions_list(limit=1)
        assert len(limited_sessions) == 1
    
    def test_search_sessions(self, session_manager):
        """세션 검색 테스트"""
        # 검색 가능한 세션들 생성
        session1 = session_manager.create_session(name="파이썬 학습", description="파이썬 기초 학습")
        session2 = session_manager.create_session(name="데이터 분석", description="pandas 사용법")
        session3 = session_manager.create_session(name="머신러닝", description="scikit-learn 예제")
        
        # 이름으로 검색
        results = session_manager.search_sessions("파이썬")
        assert len(results) >= 1
        assert any(metadata.name == "파이썬 학습" for metadata, score in results)
        
        # 설명으로 검색
        results = session_manager.search_sessions("pandas")
        assert len(results) >= 1
        assert any(metadata.description == "pandas 사용법" for metadata, score in results)
    
    def test_add_remove_tag(self, session_manager):
        """태그 추가/제거 테스트"""
        session_id = session_manager.create_session(name="태그 테스트")
        
        # 태그 추가
        result = session_manager.add_tag(session_id, "중요")
        assert result is True
        
        metadata = session_manager._metadata_cache[session_id]
        assert "중요" in metadata.tags
        
        # 중복 태그 추가 (무시됨)
        session_manager.add_tag(session_id, "중요")
        assert metadata.tags.count("중요") == 1
        
        # 태그 제거
        result = session_manager.remove_tag(session_id, "중요")
        assert result is True
        assert "중요" not in metadata.tags
    
    def test_set_favorite(self, session_manager):
        """즐겨찾기 설정 테스트"""
        session_id = session_manager.create_session(name="즐겨찾기 테스트")
        
        # 즐겨찾기 설정
        result = session_manager.set_favorite(session_id, True)
        assert result is True
        
        metadata = session_manager._metadata_cache[session_id]
        assert metadata.is_favorite is True
        
        # 즐겨찾기 해제
        result = session_manager.set_favorite(session_id, False)
        assert result is True
        assert metadata.is_favorite is False
    
    def test_backup_session(self, session_manager):
        """세션 백업 테스트"""
        session_id = session_manager.create_session(name="백업 테스트")
        
        result = session_manager.backup_session(session_id)
        assert result is True
        
        # 백업 파일 존재 확인
        backup_files = list(session_manager.backup_dir.glob(f"backup_{session_id}_*.json"))
        assert len(backup_files) >= 1
    
    def test_get_session_statistics(self, session_manager):
        """세션 통계 테스트"""
        # 여러 세션 생성
        session1 = session_manager.create_session(name="통계1", session_type=SessionType.CHAT)
        session2 = session_manager.create_session(name="통계2", session_type=SessionType.DATA_ANALYSIS)
        session_manager.set_favorite(session1, True)
        session_manager.delete_session(session2, permanent=False)
        
        stats = session_manager.get_session_statistics()
        
        assert stats["total_sessions"] >= 2
        assert stats["active_sessions"] >= 1
        assert stats["deleted_sessions"] >= 1
        assert stats["favorite_sessions"] >= 1
        assert "session_types" in stats
        assert stats["session_types"]["chat"] >= 1

class TestSessionManagerUI:
    """SessionManagerUI 클래스 테스트"""
    
    @pytest.fixture
    def session_manager(self):
        return Mock(spec=SessionManager)
    
    @pytest.fixture
    def session_ui(self, session_manager):
        return SessionManagerUI(session_manager)
    
    def test_session_ui_initialization(self, session_ui, session_manager):
        """세션 UI 초기화 테스트"""
        assert session_ui.session_manager == session_manager
    
    @patch('streamlit.sidebar')
    @patch('streamlit.header')
    @patch('streamlit.button')
    @patch('streamlit.text_input')
    @patch('streamlit.subheader')
    def test_render_session_sidebar(self, mock_subheader, mock_text_input, 
                                  mock_button, mock_header, mock_sidebar, 
                                  session_ui, session_manager):
        """세션 사이드바 렌더링 테스트"""
        # Mock 설정
        mock_button.return_value = False
        mock_text_input.return_value = ""
        session_manager.get_sessions_list.return_value = []
        session_manager.get_session_statistics.return_value = {
            "total_sessions": 5,
            "active_sessions": 4,
            "total_messages": 100,
            "total_files": 20
        }
        
        session_ui.render_session_sidebar()
        
        # 기본 UI 요소 렌더링 확인
        mock_header.assert_called()
        mock_text_input.assert_called()
        mock_subheader.assert_called()

class TestSessionManagerGlobalFunctions:
    """전역 함수 테스트"""
    
    @patch('ui.components.session_manager._session_manager_instance', None)
    @patch('ui.components.session_manager._session_manager_ui_instance', None)
    def test_get_session_manager_singleton(self):
        """싱글톤 인스턴스 테스트"""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, SessionManager)
    
    @patch('ui.components.session_manager._session_manager_instance', None)
    @patch('ui.components.session_manager._session_manager_ui_instance', None)
    def test_get_session_manager_ui_singleton(self):
        """UI 싱글톤 인스턴스 테스트"""
        ui1 = get_session_manager_ui()
        ui2 = get_session_manager_ui()
        
        assert ui1 is ui2
        assert isinstance(ui1, SessionManagerUI)
    
    @patch('ui.components.session_manager._session_manager_instance', None)
    @patch('ui.components.session_manager._session_manager_ui_instance', None)
    def test_initialize_session_manager(self):
        """세션 관리자 초기화 테스트"""
        manager, ui = initialize_session_manager()
        
        assert isinstance(manager, SessionManager)
        assert isinstance(ui, SessionManagerUI)
        assert ui.session_manager is manager

class TestSessionManagerErrorHandling:
    """에러 처리 테스트"""
    
    def test_invalid_session_type(self):
        """잘못된 세션 타입 테스트"""
        with pytest.raises(ValueError):
            SessionType("invalid_type")
    
    def test_invalid_session_status(self):
        """잘못된 세션 상태 테스트"""
        with pytest.raises(ValueError):
            SessionStatus("invalid_status")
    
    @pytest.fixture
    def broken_session_manager(self):
        """깨진 세션 관리자 (권한 없는 디렉토리)"""
        return SessionManager(sessions_dir="/root/no_permission")
    
    def test_permission_error_handling(self, broken_session_manager):
        """권한 오류 처리 테스트"""
        # 세션 생성 시도 (실패해야 함)
        result = broken_session_manager.create_session(name="실패 테스트")
        # 실제로는 예외가 발생하거나 None을 반환해야 함
        # 구현에 따라 적절히 수정 필요

class TestSessionManagerPerformance:
    """성능 테스트"""
    
    @pytest.fixture
    def temp_sessions_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_many_sessions_handling(self, temp_sessions_dir):
        """많은 세션 처리 테스트"""
        session_manager = SessionManager(sessions_dir=temp_sessions_dir)
        
        # 100개 세션 생성
        session_ids = []
        for i in range(100):
            session_id = session_manager.create_session(name=f"세션 {i}")
            session_ids.append(session_id)
        
        # 목록 조회 성능 확인
        sessions = session_manager.get_sessions_list()
        assert len(sessions) == 100
        
        # 검색 성능 확인
        results = session_manager.search_sessions("세션")
        assert len(results) > 0

# 테스트 실행을 위한 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 