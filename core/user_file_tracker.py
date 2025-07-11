#!/usr/bin/env python3
"""
🔍 User File Tracking System for CherryAI

A2A SDK 0.2.9 준수 사용자 파일 추적 및 관리 시스템
SessionDataManager와 연동하여 업로드된 파일을 A2A 에이전트가 정확히 사용하도록 함
"""

import os
import json
import shutil
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class UserFileInfo:
    """사용자 업로드 파일 정보"""
    file_id: str
    original_name: str
    session_id: str
    uploaded_at: datetime
    file_size: int
    file_type: str
    data_shape: Tuple[int, int]
    is_active: bool = True
    user_context: Optional[str] = None
    file_paths: Dict[str, str] = None  # 여러 경로에 저장된 파일 정보
    
    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = {}

@dataclass
class FileSelectionRequest:
    """파일 선택 요청 정보"""
    user_request: str
    session_id: str
    agent_name: str
    requested_at: datetime
    context: Dict[str, Any] = None

class UserFileTracker:
    """
    사용자 파일 추적 및 관리 시스템
    
    A2A SDK 0.2.9 호환 파일 추적 시스템으로 다음 기능 제공:
    - 업로드된 파일의 전체 생명주기 추적
    - 세션별 파일 관리
    - A2A 에이전트용 파일 선택 최적화
    - SessionDataManager와의 완벽한 연동
    """
    
    def __init__(self):
        # 경로 설정
        self.session_data_path = Path("ai_ds_team/data")
        self.shared_data_path = Path("a2a_ds_servers/artifacts/data/shared_dataframes")
        self.metadata_path = Path("core/file_tracking_metadata")
        
        # 경로 생성
        self.session_data_path.mkdir(parents=True, exist_ok=True)
        self.shared_data_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # 내부 상태
        self.tracked_files: Dict[str, UserFileInfo] = {}
        self.session_files: Dict[str, List[str]] = {}  # session_id -> file_ids
        self.current_session_id: Optional[str] = None
        
        # 기존 메타데이터 로드
        self._load_existing_metadata()
        
        logger.info("UserFileTracker initialized with A2A SDK 0.2.9 compatibility")
    
    def register_uploaded_file(self, 
                             file_id: str,
                             original_name: str,
                             session_id: str,
                             data: pd.DataFrame,
                             user_context: Optional[str] = None) -> bool:
        """
        사용자 업로드 파일 등록
        
        Args:
            file_id: 파일 고유 ID (보통 원본 파일명)
            original_name: 원본 파일명
            session_id: 세션 ID
            data: 판다스 데이터프레임
            user_context: 사용자 컨텍스트 정보
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            # 파일 정보 생성
            file_info = UserFileInfo(
                file_id=file_id,
                original_name=original_name,
                session_id=session_id,
                uploaded_at=datetime.now(),
                file_size=int(data.memory_usage(deep=True).sum()),
                file_type=Path(original_name).suffix.lower(),
                data_shape=data.shape,
                user_context=user_context
            )
            
            # 1. 세션 경로에 저장 (SessionDataManager 호환)
            session_dir = self.session_data_path / session_id
            session_dir.mkdir(exist_ok=True)
            session_file_path = session_dir / original_name
            self._save_dataframe(data, session_file_path)
            file_info.file_paths['session'] = str(session_file_path)
            
            # 2. 공유 경로에 저장 (A2A 에이전트 호환)
            shared_file_path = self.shared_data_path / f"{session_id}_{original_name}"
            self._save_dataframe(data, shared_file_path)
            file_info.file_paths['shared'] = str(shared_file_path)
            
            # 3. 메타데이터에 컨텍스트 저장
            context_file = session_dir / "file_context.json"
            context_data = {
                "file_id": file_id,
                "original_name": original_name,
                "uploaded_at": file_info.uploaded_at.isoformat(),
                "user_context": user_context,
                "data_shape": data.shape,
                "shared_file_path": str(shared_file_path)
            }
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, ensure_ascii=False, indent=2)
            
            # 4. 추적 정보 업데이트
            self.tracked_files[file_id] = file_info
            
            if session_id not in self.session_files:
                self.session_files[session_id] = []
            self.session_files[session_id].append(file_id)
            
            # 5. 현재 세션으로 설정
            self.current_session_id = session_id
            
            # 6. 메타데이터 저장
            self._save_metadata()
            
            logger.info(f"✅ File registered: {file_id} in session {session_id}")
            logger.info(f"   - Session path: {session_file_path}")
            logger.info(f"   - Shared path: {shared_file_path}")
            logger.info(f"   - Data shape: {data.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register file {file_id}: {e}")
            return False
    
    def get_file_for_a2a_request(self, 
                                user_request: str,
                                session_id: Optional[str] = None,
                                agent_name: Optional[str] = None) -> Tuple[Optional[str], str]:
        """
        A2A 요청에 대한 최적 파일 선택
        
        Args:
            user_request: 사용자 요청 텍스트
            session_id: 세션 ID (없으면 현재 세션)
            agent_name: 요청한 에이전트명
            
        Returns:
            Tuple[파일경로, 선택이유]: 선택된 파일 경로와 선택 이유
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.session_files:
            return None, "세션을 찾을 수 없음"
        
        session_file_ids = self.session_files[session_id]
        if not session_file_ids:
            return None, "세션에 업로드된 파일이 없음"
        
        # 요청 분석 및 파일 선택
        selection_request = FileSelectionRequest(
            user_request=user_request,
            session_id=session_id,
            agent_name=agent_name or "unknown",
            requested_at=datetime.now(),
            context={"source": "a2a_request"}
        )
        
        selected_file_id, reason = self._smart_file_selection(selection_request, session_file_ids)
        
        if selected_file_id:
            file_info = self.tracked_files[selected_file_id]
            # A2A 에이전트가 접근 가능한 공유 경로 반환
            shared_path = file_info.file_paths.get('shared')
            if shared_path and os.path.exists(shared_path):
                logger.info(f"🎯 Selected file for A2A: {shared_path} ({reason})")
                return shared_path, reason
            else:
                # 공유 경로가 없으면 세션 경로 반환
                session_path = file_info.file_paths.get('session')
                if session_path and os.path.exists(session_path):
                    logger.info(f"🎯 Selected file for A2A: {session_path} ({reason})")
                    return session_path, reason
        
        return None, "선택된 파일을 찾을 수 없음"
    
    def _smart_file_selection(self, 
                             request: FileSelectionRequest,
                             available_file_ids: List[str]) -> Tuple[Optional[str], str]:
        """스마트 파일 선택 로직"""
        
        # 1순위: 사용자가 명시적으로 파일명 언급
        mentioned_file = self._extract_mentioned_filename(request.user_request)
        if mentioned_file:
            for file_id in available_file_ids:
                file_info = self.tracked_files[file_id]
                if (mentioned_file.lower() in file_info.original_name.lower() or
                    mentioned_file.lower() in file_id.lower()):
                    return file_id, f"사용자 언급 파일: '{mentioned_file}'"
        
        # 2순위: 도메인별 최적화 (반도체, 금융, 의료 등)
        domain_file = self._find_domain_optimized_file(request.user_request, available_file_ids)
        if domain_file:
            return domain_file, "도메인 최적화 선택"
        
        # 3순위: 활성 파일 (가장 최근 업로드)
        active_files = [fid for fid in available_file_ids 
                       if self.tracked_files[fid].is_active]
        if active_files:
            latest_file = max(active_files, 
                            key=lambda fid: self.tracked_files[fid].uploaded_at)
            return latest_file, "가장 최근 업로드 파일"
        
        # 4순위: 첫 번째 사용 가능 파일
        if available_file_ids:
            return available_file_ids[0], "첫 번째 사용 가능 파일"
        
        return None, "사용 가능한 파일 없음"
    
    def _extract_mentioned_filename(self, user_request: str) -> Optional[str]:
        """사용자 요청에서 언급된 파일명 추출"""
        import re
        
        # 파일 확장자가 포함된 패턴 (우선순위 순서)
        patterns = [
            r'([a-zA-Z0-9_\-]+\.(?:csv|xlsx|xls|json|pkl))',  # 확장자 포함 파일명 최우선
            r'([a-zA-Z0-9_\-]*ion_implant[a-zA-Z0-9_\-]*)',  # ion_implant 포함
            r'([a-zA-Z0-9_\-]*titanic[a-zA-Z0-9_\-]*)',      # titanic 포함
            r'([a-zA-Z0-9_\-]+_dataset[a-zA-Z0-9_\-]*)',     # _dataset 포함 (더 구체적)
            r'([a-zA-Z0-9_\-]+dataset[a-zA-Z0-9_\-]*)',      # dataset 포함
            r'([a-zA-Z0-9_\-]+_data[a-zA-Z0-9_\-]*)'         # _data 포함
        ]
        
        # 가장 긴 매치를 찾기 위해 모든 패턴 확인
        all_matches = []
        
        for pattern in patterns:
            matches = re.findall(pattern, user_request, re.IGNORECASE)
            for match in matches:
                all_matches.append(match)
        
        if all_matches:
            # 가장 긴 매치 반환 (더 구체적인 것을 선호)
            return max(all_matches, key=len)
        
        return None
    
    def _find_domain_optimized_file(self, user_request: str, file_ids: List[str]) -> Optional[str]:
        """도메인 기반 최적화 파일 선택"""
        user_request_lower = user_request.lower()
        
        # 도메인별 키워드 매핑
        domain_keywords = {
            'semiconductor': ['반도체', 'ion', 'implant', 'wafer', 'fab', 'process'],
            'finance': ['financial', '금융', 'bank', 'stock', 'investment'],
            'medical': ['medical', '의료', 'patient', 'clinical', 'health'],
            'retail': ['sales', '판매', 'customer', 'product', 'marketing']
        }
        
        # 사용자 요청에서 도메인 감지
        detected_domain = None
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_request_lower for keyword in keywords):
                detected_domain = domain
                break
        
        if detected_domain:
            # 해당 도메인에 맞는 파일 검색
            domain_files = []
            for file_id in file_ids:
                file_info = self.tracked_files[file_id]
                file_name_lower = file_info.original_name.lower()
                
                if detected_domain == 'semiconductor' and 'ion' in file_name_lower:
                    domain_files.append(file_id)
                elif detected_domain == 'finance' and any(kw in file_name_lower for kw in ['financial', 'bank', 'stock']):
                    domain_files.append(file_id)
                # 추가 도메인 매칭 로직...
            
            if domain_files:
                # 가장 최근 도메인 파일 반환
                return max(domain_files, 
                         key=lambda fid: self.tracked_files[fid].uploaded_at)
        
        return None
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: Path):
        """데이터프레임을 적절한 형식으로 저장"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            df.to_csv(file_path, index=False)
        elif file_extension in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif file_extension == '.pkl':
            df.to_pickle(file_path)
        elif file_extension == '.json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            # 기본값: CSV로 저장
            df.to_csv(file_path.with_suffix('.csv'), index=False)
    
    def _load_existing_metadata(self):
        """기존 메타데이터 로드"""
        metadata_file = self.metadata_path / "file_tracking.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # UserFileInfo 객체 복원
                for file_id, file_data in data.get('tracked_files', {}).items():
                    # datetime 복원
                    file_data['uploaded_at'] = datetime.fromisoformat(file_data['uploaded_at'])
                    
                    # tuple 복원
                    file_data['data_shape'] = tuple(file_data['data_shape'])
                    
                    self.tracked_files[file_id] = UserFileInfo(**file_data)
                
                self.session_files = data.get('session_files', {})
                self.current_session_id = data.get('current_session_id')
                
                logger.info(f"Loaded {len(self.tracked_files)} tracked files")
                
            except Exception as e:
                logger.warning(f"Failed to load file tracking metadata: {e}")
    
    def _save_metadata(self):
        """메타데이터 저장"""
        metadata_file = self.metadata_path / "file_tracking.json"
        
        try:
            # 직렬화 가능한 형태로 변환
            tracked_files_data = {}
            for file_id, file_info in self.tracked_files.items():
                file_data = asdict(file_info)
                file_data['uploaded_at'] = file_info.uploaded_at.isoformat()
                tracked_files_data[file_id] = file_data
            
            data = {
                'tracked_files': tracked_files_data,
                'session_files': self.session_files,
                'current_session_id': self.current_session_id,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save file tracking metadata: {e}")
    
    def get_session_files_info(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """세션의 파일 정보 반환"""
        if session_id is None:
            session_id = self.current_session_id
        
        if not session_id or session_id not in self.session_files:
            return []
        
        files_info = []
        for file_id in self.session_files[session_id]:
            if file_id in self.tracked_files:
                file_info = self.tracked_files[file_id]
                files_info.append({
                    'file_id': file_info.file_id,
                    'original_name': file_info.original_name,
                    'uploaded_at': file_info.uploaded_at,
                    'file_size': file_info.file_size,
                    'data_shape': file_info.data_shape,
                    'is_active': file_info.is_active,
                    'shared_path': file_info.file_paths.get('shared'),
                    'session_path': file_info.file_paths.get('session')
                })
        
        return files_info
    
    def cleanup_old_files(self, hours_threshold: int = 48):
        """오래된 파일 정리"""
        current_time = datetime.now()
        files_to_remove = []
        
        for file_id, file_info in self.tracked_files.items():
            age = current_time - file_info.uploaded_at
            if age.total_seconds() / 3600 > hours_threshold:
                files_to_remove.append(file_id)
        
        for file_id in files_to_remove:
            try:
                file_info = self.tracked_files[file_id]
                
                # 파일 삭제
                for path in file_info.file_paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                
                # 메타데이터에서 제거
                del self.tracked_files[file_id]
                
                # 세션에서 제거
                for session_id, file_ids in self.session_files.items():
                    if file_id in file_ids:
                        file_ids.remove(file_id)
                
                logger.info(f"🗑️ Cleaned up old file: {file_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup file {file_id}: {e}")
        
        if files_to_remove:
            self._save_metadata()

# 전역 인스턴스
_user_file_tracker = None

def get_user_file_tracker() -> UserFileTracker:
    """전역 UserFileTracker 인스턴스 반환"""
    global _user_file_tracker
    if _user_file_tracker is None:
        _user_file_tracker = UserFileTracker()
    return _user_file_tracker 