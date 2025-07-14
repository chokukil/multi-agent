#!/usr/bin/env python3
"""
📚 CherryAI 세션 관리 시스템

ChatGPT/Claude 스타일의 세션 관리 기능을 제공하는 고급 시스템

Key Features:
- 세션 생성, 저장, 불러오기, 삭제
- 자동 세션 이름 생성 (LLM 기반)
- 세션 검색 및 필터링
- 즐겨찾기 및 태깅
- 세션 병합/분할 기능
- 크로스 세션 검색
- 세션 메타데이터 관리
- 자동 백업 및 복구

Architecture:
- Session Model: 세션 데이터 구조
- Storage Manager: 파일 시스템 기반 저장
- Search Engine: 시맨틱 검색 지원
- UI Components: 세션 목록, 검색, 관리
"""

import streamlit as st
import json
import os
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

# Knowledge Bank 연동을 위한 임포트
try:
    from core.shared_knowledge_bank import get_shared_knowledge_bank, search_relevant_knowledge
    KNOWLEDGE_BANK_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BANK_AVAILABLE = False

# LLM First Engine 연동을 위한 임포트
try:
    from core.llm_first_engine import get_llm_first_engine, analyze_intent
    LLM_FIRST_AVAILABLE = True
except ImportError:
    LLM_FIRST_AVAILABLE = False

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """세션 상태"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    BACKUP = "backup"

class SessionType(Enum):
    """세션 타입"""
    CHAT = "chat"
    DATA_ANALYSIS = "data_analysis"
    FILE_PROCESSING = "file_processing"
    COLLABORATION = "collaboration"
    CUSTOM = "custom"

@dataclass
class SessionMetadata:
    """세션 메타데이터"""
    id: str
    name: str
    description: str
    session_type: SessionType
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    message_count: int
    file_count: int
    tags: List[str]
    is_favorite: bool
    size_bytes: int
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "session_type": self.session_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "message_count": self.message_count,
            "file_count": self.file_count,
            "tags": self.tags,
            "is_favorite": self.is_favorite,
            "size_bytes": self.size_bytes,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMetadata':
        """딕셔너리에서 생성"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            session_type=SessionType(data["session_type"]),
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            message_count=data["message_count"],
            file_count=data["file_count"],
            tags=data["tags"],
            is_favorite=data["is_favorite"],
            size_bytes=data["size_bytes"],
            version=data.get("version", "1.0")
        )

@dataclass
class SessionData:
    """세션 데이터"""
    metadata: SessionMetadata
    messages: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    variables: Dict[str, Any]
    custom_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "metadata": self.metadata.to_dict(),
            "messages": self.messages,
            "files": self.files,
            "variables": self.variables,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """딕셔너리에서 생성"""
        return cls(
            metadata=SessionMetadata.from_dict(data["metadata"]),
            messages=data["messages"],
            files=data["files"],
            variables=data["variables"],
            custom_data=data["custom_data"]
        )

class SessionManager:
    """
    📚 세션 관리자
    
    모든 세션 관련 기능을 제공하는 메인 클래스
    """
    
    def __init__(self, sessions_dir: str = "sessions_metadata"):
        """세션 관리자 초기화"""
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        
        # 백업 디렉토리
        self.backup_dir = self.sessions_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 현재 세션 ID
        self.current_session_id = None
        
        # 세션 캐시
        self._session_cache: Dict[str, SessionData] = {}
        self._metadata_cache: Dict[str, SessionMetadata] = {}
        
        # 자동 백업 설정
        self.auto_backup_interval = timedelta(minutes=5)
        self.max_backups = 10
        
        logger.info(f"📚 세션 관리자 초기화 완료: {self.sessions_dir}")
    
    def create_session(self, 
                      name: str = None, 
                      session_type: SessionType = SessionType.CHAT,
                      description: str = "",
                      tags: List[str] = None) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # 자동 이름 생성
        if name is None:
            name = self._generate_session_name(session_type, now)
        
        # 메타데이터 생성
        metadata = SessionMetadata(
            id=session_id,
            name=name,
            description=description,
            session_type=session_type,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            message_count=0,
            file_count=0,
            tags=tags or [],
            is_favorite=False,
            size_bytes=0
        )
        
        # 세션 데이터 생성
        session_data = SessionData(
            metadata=metadata,
            messages=[],
            files=[],
            variables={},
            custom_data={}
        )
        
        # 저장
        self._save_session(session_data)
        
        # 캐시 업데이트
        self._session_cache[session_id] = session_data
        self._metadata_cache[session_id] = metadata
        
        # 현재 세션으로 설정
        self.current_session_id = session_id
        
        logger.info(f"📚 새 세션 생성됨: {session_id} - {name}")
        return session_id
    
    def load_session(self, session_id: str) -> Optional[SessionData]:
        """세션 불러오기"""
        try:
            # 캐시에서 먼저 확인
            if session_id in self._session_cache:
                session_data = self._session_cache[session_id]
                # 접근 시간 업데이트
                session_data.metadata.last_accessed = datetime.now()
                self._save_session(session_data)
                return session_data
            
            # 파일에서 로드
            session_file = self.sessions_dir / f"session_{session_id}.json"
            if not session_file.exists():
                logger.warning(f"세션 파일이 존재하지 않음: {session_id}")
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session_data = SessionData.from_dict(data)
            
            # 접근 시간 업데이트
            session_data.metadata.last_accessed = datetime.now()
            self._save_session(session_data)
            
            # 캐시 업데이트
            self._session_cache[session_id] = session_data
            self._metadata_cache[session_id] = session_data.metadata
            
            logger.info(f"📚 세션 로드됨: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"세션 로드 실패: {session_id} - {e}")
            return None
    
    def save_session(self, session_data: SessionData) -> bool:
        """세션 저장"""
        try:
            # 메타데이터 업데이트
            session_data.metadata.updated_at = datetime.now()
            session_data.metadata.message_count = len(session_data.messages)
            session_data.metadata.file_count = len(session_data.files)
            
            # 저장
            self._save_session(session_data)
            
            # 캐시 업데이트
            self._session_cache[session_data.metadata.id] = session_data
            self._metadata_cache[session_data.metadata.id] = session_data.metadata
            
            logger.info(f"📚 세션 저장됨: {session_data.metadata.id}")
            return True
            
        except Exception as e:
            logger.error(f"세션 저장 실패: {session_data.metadata.id} - {e}")
            return False
    
    def delete_session(self, session_id: str, permanent: bool = False) -> bool:
        """세션 삭제"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                logger.warning(f"삭제할 세션이 존재하지 않음: {session_id}")
                return False
            
            if permanent:
                # 영구 삭제
                session_file = self.sessions_dir / f"session_{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
                
                # 캐시에서 제거
                self._session_cache.pop(session_id, None)
                self._metadata_cache.pop(session_id, None)
                
                logger.info(f"📚 세션 영구 삭제됨: {session_id}")
            else:
                # 상태를 삭제로 변경 (소프트 삭제)
                session_data.metadata.status = SessionStatus.DELETED
                session_data.metadata.updated_at = datetime.now()
                self._save_session(session_data)
                
                # 캐시 업데이트
                self._metadata_cache[session_id] = session_data.metadata
                
                logger.info(f"📚 세션 삭제됨 (복구 가능): {session_id}")
            
            # 현재 세션이 삭제된 경우
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            return True
            
        except Exception as e:
            logger.error(f"세션 삭제 실패: {session_id} - {e}")
            return False
    
    def restore_session(self, session_id: str) -> bool:
        """삭제된 세션 복구"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            if session_data.metadata.status == SessionStatus.DELETED:
                session_data.metadata.status = SessionStatus.ACTIVE
                session_data.metadata.updated_at = datetime.now()
                self._save_session(session_data)
                
                logger.info(f"📚 세션 복구됨: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"세션 복구 실패: {session_id} - {e}")
            return False
    
    def get_sessions_list(self, 
                         include_deleted: bool = False,
                         session_type: SessionType = None,
                         limit: int = None) -> List[SessionMetadata]:
        """세션 목록 조회"""
        try:
            sessions = []
            
            # 모든 세션 파일 스캔
            for session_file in self.sessions_dir.glob("session_*.json"):
                session_id = session_file.stem.replace("session_", "")
                
                # 캐시에서 먼저 확인
                if session_id in self._metadata_cache:
                    metadata = self._metadata_cache[session_id]
                else:
                    # 파일에서 메타데이터만 로드
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        metadata = SessionMetadata.from_dict(data["metadata"])
                        self._metadata_cache[session_id] = metadata
                    except Exception as e:
                        logger.error(f"메타데이터 로드 실패: {session_file} - {e}")
                        continue
                
                # 필터링
                if not include_deleted and metadata.status == SessionStatus.DELETED:
                    continue
                
                if session_type and metadata.session_type != session_type:
                    continue
                
                sessions.append(metadata)
            
            # 최근 접근 순으로 정렬
            sessions.sort(key=lambda x: x.last_accessed, reverse=True)
            
            # 제한 적용
            if limit:
                sessions = sessions[:limit]
            
            return sessions
            
        except Exception as e:
            logger.error(f"세션 목록 조회 실패: {e}")
            return []
    
    def search_sessions(self, 
                       query: str,
                       search_in_messages: bool = True,
                       search_in_files: bool = True) -> List[Tuple[SessionMetadata, float]]:
        """세션 검색 (시맨틱 검색 지원)"""
        try:
            results = []
            
            # 모든 활성 세션 검색
            sessions = self.get_sessions_list()
            
            for metadata in sessions:
                score = 0.0
                
                # 이름과 설명에서 텍스트 매칭
                if query.lower() in metadata.name.lower():
                    score += 10.0
                if query.lower() in metadata.description.lower():
                    score += 5.0
                
                # 태그 매칭
                for tag in metadata.tags:
                    if query.lower() in tag.lower():
                        score += 3.0
                
                # 메시지 내용 검색
                if search_in_messages and score < 8.0:  # 이미 높은 점수가 아닌 경우에만
                    session_data = self.load_session(metadata.id)
                    if session_data:
                        for message in session_data.messages:
                            content = message.get("content", "")
                            if query.lower() in content.lower():
                                score += 2.0
                                break
                
                # Knowledge Bank를 통한 시맨틱 검색 (가능한 경우)
                if KNOWLEDGE_BANK_AVAILABLE and score < 8.0:
                    try:
                        # 세션별 지식 검색
                        relevant_docs = search_relevant_knowledge(
                            query, 
                            agent_context=f"session_{metadata.id}",
                            top_k=1
                        )
                        if relevant_docs:
                            score += 1.0
                    except Exception:
                        pass
                
                if score > 0:
                    results.append((metadata, score))
            
            # 점수 순으로 정렬
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"세션 검색 실패: {e}")
            return []
    
    def add_tag(self, session_id: str, tag: str) -> bool:
        """세션에 태그 추가"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            if tag not in session_data.metadata.tags:
                session_data.metadata.tags.append(tag)
                self.save_session(session_data)
                logger.info(f"📚 태그 추가됨: {session_id} - {tag}")
            
            return True
            
        except Exception as e:
            logger.error(f"태그 추가 실패: {session_id} - {e}")
            return False
    
    def remove_tag(self, session_id: str, tag: str) -> bool:
        """세션에서 태그 제거"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            if tag in session_data.metadata.tags:
                session_data.metadata.tags.remove(tag)
                self.save_session(session_data)
                logger.info(f"📚 태그 제거됨: {session_id} - {tag}")
            
            return True
            
        except Exception as e:
            logger.error(f"태그 제거 실패: {session_id} - {e}")
            return False
    
    def set_favorite(self, session_id: str, is_favorite: bool) -> bool:
        """즐겨찾기 설정/해제"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            session_data.metadata.is_favorite = is_favorite
            self.save_session(session_data)
            
            action = "설정" if is_favorite else "해제"
            logger.info(f"📚 즐겨찾기 {action}됨: {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"즐겨찾기 설정 실패: {session_id} - {e}")
            return False
    
    def backup_session(self, session_id: str) -> bool:
        """세션 백업"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            # 백업 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{session_id}_{timestamp}.json"
            backup_path = self.backup_dir / backup_filename
            
            # 백업 저장
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(session_data.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 오래된 백업 정리
            self._cleanup_old_backups(session_id)
            
            logger.info(f"📚 세션 백업됨: {session_id} -> {backup_filename}")
            return True
            
        except Exception as e:
            logger.error(f"세션 백업 실패: {session_id} - {e}")
            return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """세션 통계 정보"""
        try:
            sessions = self.get_sessions_list(include_deleted=True)
            
            stats = {
                "total_sessions": len(sessions),
                "active_sessions": len([s for s in sessions if s.status == SessionStatus.ACTIVE]),
                "deleted_sessions": len([s for s in sessions if s.status == SessionStatus.DELETED]),
                "archived_sessions": len([s for s in sessions if s.status == SessionStatus.ARCHIVED]),
                "favorite_sessions": len([s for s in sessions if s.is_favorite]),
                "total_messages": sum(s.message_count for s in sessions),
                "total_files": sum(s.file_count for s in sessions),
                "total_size_mb": sum(s.size_bytes for s in sessions) / (1024 * 1024),
                "session_types": {}
            }
            
            # 세션 타입별 통계
            for session_type in SessionType:
                count = len([s for s in sessions if s.session_type == session_type])
                stats["session_types"][session_type.value] = count
            
            return stats
            
        except Exception as e:
            logger.error(f"세션 통계 조회 실패: {e}")
            return {}
    
    def _save_session(self, session_data: SessionData) -> None:
        """세션 저장 (내부 함수)"""
        session_file = self.sessions_dir / f"session_{session_data.metadata.id}.json"
        
        # 파일 크기 계산
        data_str = json.dumps(session_data.to_dict(), ensure_ascii=False, indent=2)
        session_data.metadata.size_bytes = len(data_str.encode('utf-8'))
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write(data_str)
    
    def _generate_session_name(self, session_type: SessionType, created_at: datetime) -> str:
        """자동 세션 이름 생성"""
        # LLM First Engine을 사용한 지능적 이름 생성 (가능한 경우)
        if LLM_FIRST_AVAILABLE:
            try:
                llm_engine = get_llm_first_engine()
                # 기본 컨텍스트로 이름 생성 요청
                # 실제로는 현재 메시지나 파일 정보를 바탕으로 더 지능적으로 생성 가능
                pass
            except Exception:
                pass
        
        # 기본 이름 생성
        type_names = {
            SessionType.CHAT: "대화",
            SessionType.DATA_ANALYSIS: "데이터 분석",
            SessionType.FILE_PROCESSING: "파일 처리",
            SessionType.COLLABORATION: "협업",
            SessionType.CUSTOM: "사용자 정의"
        }
        
        type_name = type_names.get(session_type, "세션")
        timestamp = created_at.strftime("%m/%d %H:%M")
        
        return f"{type_name} - {timestamp}"
    
    def _cleanup_old_backups(self, session_id: str) -> None:
        """오래된 백업 정리"""
        try:
            # 해당 세션의 백업 파일들 찾기
            backup_pattern = f"backup_{session_id}_*.json"
            backup_files = list(self.backup_dir.glob(backup_pattern))
            
            # 생성 시간 순으로 정렬
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 최대 개수 초과시 오래된 것들 삭제
            if len(backup_files) > self.max_backups:
                for old_backup in backup_files[self.max_backups:]:
                    old_backup.unlink()
                    logger.info(f"📚 오래된 백업 삭제됨: {old_backup.name}")
                    
        except Exception as e:
            logger.error(f"백업 정리 실패: {e}")

# Streamlit UI 컴포넌트들
class SessionManagerUI:
    """세션 관리 UI 컴포넌트"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def render_session_sidebar(self) -> None:
        """세션 관리 사이드바 렌더링"""
        with st.sidebar:
            st.header("📚 세션 관리")
            
            # 새 세션 버튼
            if st.button("➕ 새 세션", use_container_width=True):
                session_id = self.session_manager.create_session()
                st.rerun()
            
            # 현재 세션 정보
            if self.session_manager.current_session_id:
                current_session = self.session_manager.load_session(
                    self.session_manager.current_session_id
                )
                if current_session:
                    st.info(f"**현재 세션:** {current_session.metadata.name}")
            
            # 세션 검색
            search_query = st.text_input("🔍 세션 검색", placeholder="세션 이름이나 내용 검색...")
            
            # 세션 목록
            sessions = self.session_manager.get_sessions_list(limit=20)
            
            if search_query:
                # 검색 결과
                search_results = self.session_manager.search_sessions(search_query)
                sessions = [result[0] for result in search_results[:10]]
            
            st.subheader("💬 최근 세션")
            
            for session_metadata in sessions:
                self._render_session_item(session_metadata)
            
            # 세션 통계
            with st.expander("📊 세션 통계"):
                stats = self.session_manager.get_session_statistics()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("전체 세션", stats.get("total_sessions", 0))
                    st.metric("총 메시지", stats.get("total_messages", 0))
                
                with col2:
                    st.metric("활성 세션", stats.get("active_sessions", 0))
                    st.metric("총 파일", stats.get("total_files", 0))
    
    def _render_session_item(self, metadata: SessionMetadata) -> None:
        """세션 항목 렌더링"""
        # 세션 상태 아이콘
        status_icons = {
            SessionStatus.ACTIVE: "🟢",
            SessionStatus.ARCHIVED: "📦",
            SessionStatus.DELETED: "🗑️",
            SessionStatus.BACKUP: "💾"
        }
        
        status_icon = status_icons.get(metadata.status, "❓")
        favorite_icon = "⭐" if metadata.is_favorite else ""
        
        # 세션 버튼
        session_label = f"{status_icon}{favorite_icon} {metadata.name}"
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(session_label, key=f"session_{metadata.id}", use_container_width=True):
                self.session_manager.current_session_id = metadata.id
                st.rerun()
        
        with col2:
            # 세션 관리 메뉴
            if st.button("⚙️", key=f"menu_{metadata.id}"):
                st.session_state[f"session_menu_{metadata.id}"] = True
        
        # 세션 메뉴 (토글)
        if st.session_state.get(f"session_menu_{metadata.id}", False):
            self._render_session_menu(metadata)
    
    def _render_session_menu(self, metadata: SessionMetadata) -> None:
        """세션 관리 메뉴 렌더링"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 즐겨찾기 토글
            favorite_label = "💖" if metadata.is_favorite else "🤍"
            if st.button(favorite_label, key=f"fav_{metadata.id}"):
                self.session_manager.set_favorite(metadata.id, not metadata.is_favorite)
                st.rerun()
        
        with col2:
            # 백업
            if st.button("💾", key=f"backup_{metadata.id}"):
                if self.session_manager.backup_session(metadata.id):
                    st.success("백업 완료!")
                else:
                    st.error("백업 실패!")
        
        with col3:
            # 삭제
            if st.button("🗑️", key=f"delete_{metadata.id}"):
                if self.session_manager.delete_session(metadata.id):
                    st.success("세션이 삭제되었습니다.")
                    st.rerun()
                else:
                    st.error("삭제 실패!")
        
        # 메뉴 닫기
        if st.button("✖️ 닫기", key=f"close_menu_{metadata.id}"):
            st.session_state[f"session_menu_{metadata.id}"] = False
            st.rerun()

# 전역 인스턴스 관리
_session_manager_instance = None
_session_manager_ui_instance = None

def get_session_manager() -> SessionManager:
    """세션 관리자 싱글톤 인스턴스 반환"""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance

def get_session_manager_ui() -> SessionManagerUI:
    """세션 관리 UI 싱글톤 인스턴스 반환"""
    global _session_manager_ui_instance, _session_manager_instance
    if _session_manager_ui_instance is None:
        if _session_manager_instance is None:
            _session_manager_instance = SessionManager()
        _session_manager_ui_instance = SessionManagerUI(_session_manager_instance)
    return _session_manager_ui_instance

def initialize_session_manager() -> Tuple[SessionManager, SessionManagerUI]:
    """세션 관리자 초기화"""
    global _session_manager_instance, _session_manager_ui_instance
    _session_manager_instance = SessionManager()
    _session_manager_ui_instance = SessionManagerUI(_session_manager_instance)
    return _session_manager_instance, _session_manager_ui_instance 