# File: core/session_data_manager_enhanced.py
import pandas as pd
import json
import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass, asdict
from core.data_manager import DataManager

# UserFileTracker 통합
try:
    from core.user_file_tracker import get_user_file_tracker
    USER_FILE_TRACKER_AVAILABLE = True
except ImportError:
    USER_FILE_TRACKER_AVAILABLE = False
    logging.warning("UserFileTracker not available")

# AI DS Team data directory for session-based data
AI_DS_TEAM_DATA_DIR = Path("ai_ds_team/data")
AI_DS_TEAM_DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class FileMetadata:
    """파일 메타데이터 구조"""
    data_id: str
    uploaded_at: datetime
    file_size: int
    file_type: str
    user_context: Optional[str] = None
    domain: Optional[str] = None
    original_name: str = ""

@dataclass
class SessionMetadata:
    """세션 메타데이터 구조"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    uploaded_files: List[FileMetadata]
    active_file: Optional[str] = None
    auto_cleanup_at: datetime = None
    status: str = "active"  # "active", "warning", "expired"
    
    def __post_init__(self):
        if self.auto_cleanup_at is None:
            self.auto_cleanup_at = self.created_at + timedelta(hours=24)

class SessionDataManager:
    """
    세션 기반 데이터 관리자 (Enhanced with UserFileTracker)
    AI DS Team과 완벽하게 통합되어 세션별로 데이터와 컨텍스트를 관리합니다.
    UserFileTracker와 통합하여 A2A 에이전트들이 올바른 파일을 사용하도록 합니다.
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self._current_session_id: Optional[str] = None
        self._session_metadata: Dict[str, SessionMetadata] = {}
        
        # UserFileTracker 통합
        if USER_FILE_TRACKER_AVAILABLE:
            self.user_file_tracker = get_user_file_tracker()
        else:
            self.user_file_tracker = None
        
        # 세션 메타데이터 저장 디렉토리
        self._metadata_dir = Path("sessions_metadata")
        self._metadata_dir.mkdir(exist_ok=True)
        
        # 기본 데이터 폴더 초기화
        self._initialize_default_data()
        
        # 기존 세션 메타데이터 로드
        self._load_existing_sessions()
        
        logging.info("Enhanced SessionDataManager initialized with UserFileTracker integration")

    def _initialize_default_data(self):
        """기본 AI DS Team 데이터 폴더 초기화"""
        default_dir = AI_DS_TEAM_DATA_DIR / "default"
        default_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 샘플 데이터를 default 폴더로 이동
        sample_files = [
            "ai_ds_team/data/bike_sales_data.csv",
            "ai_ds_team/data/churn_data.csv", 
            "ai_ds_team/data/dirty_dataset.csv"
        ]
        
        for sample_file in sample_files:
            if os.path.exists(sample_file):
                filename = os.path.basename(sample_file)
                target_path = default_dir / filename
                if not target_path.exists():
                    try:
                        shutil.copy2(sample_file, target_path)
                        logging.info(f"Moved sample data to default folder: {filename}")
                    except Exception as e:
                        logging.warning(f"Failed to copy {filename}: {e}")

    def _load_existing_sessions(self):
        """기존 세션 메타데이터 로드"""
        for metadata_file in self._metadata_dir.glob("session_*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # datetime 객체 복원
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['last_activity'] = datetime.fromisoformat(data['last_activity'])
                data['auto_cleanup_at'] = datetime.fromisoformat(data['auto_cleanup_at'])
                
                # FileMetadata 객체 복원
                files = []
                for file_data in data['uploaded_files']:
                    file_data['uploaded_at'] = datetime.fromisoformat(file_data['uploaded_at'])
                    files.append(FileMetadata(**file_data))
                data['uploaded_files'] = files
                
                session_metadata = SessionMetadata(**data)
                self._session_metadata[session_metadata.session_id] = session_metadata
                
                logging.info(f"Loaded existing session: {session_metadata.session_id}")
                
            except Exception as e:
                logging.warning(f"Failed to load session metadata {metadata_file}: {e}")

    def _save_session_metadata(self, session_id: str):
        """세션 메타데이터를 파일에 저장"""
        if session_id not in self._session_metadata:
            return
        
        metadata = self._session_metadata[session_id]
        metadata_file = self._metadata_dir / f"session_{session_id}.json"
        
        # JSON 직렬화를 위한 데이터 변환
        data = asdict(metadata)
        data['created_at'] = metadata.created_at.isoformat()
        data['last_activity'] = metadata.last_activity.isoformat()
        data['auto_cleanup_at'] = metadata.auto_cleanup_at.isoformat()
        
        for file_data in data['uploaded_files']:
            file_data['uploaded_at'] = file_data['uploaded_at'].isoformat()
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save session metadata {session_id}: {e}")

    def extract_domain_from_request(self, user_request: str) -> Optional[str]:
        """사용자 요청에서 도메인 컨텍스트 추출"""
        domain_keywords = self._get_domain_configs_dynamically()
        
        user_request_lower = user_request.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_request_lower for keyword in keywords):
                return domain
        
        return None

    def extract_filename_pattern(self, user_request: str) -> Optional[str]:
        """사용자 요청에서 파일명 패턴 추출"""
        # 파일 확장자가 포함된 패턴 찾기
        import re
        
        # .csv, .xlsx 등이 포함된 파일명 패턴
        file_patterns = re.findall(r'(\w+\.(csv|xlsx|xls|json))', user_request.lower())
        if file_patterns:
            return file_patterns[0][0]
        
        # 특정 키워드 패턴
        keyword_patterns = [
            "ion_implant", "titanic", "churn", "sales", "bike", "employee"
        ]
        
        for pattern in keyword_patterns:
            if pattern in user_request.lower():
                return pattern
        
        return None

    def smart_file_selection(self, user_request: str, session_id: Optional[str] = None) -> Tuple[Optional[str], str]:
        """스마트 파일 선택 로직 (UserFileTracker 통합)"""
        if session_id is None:
            session_id = self._current_session_id
        
        # UserFileTracker를 우선 사용
        if self.user_file_tracker:
            file_path, reason = self.user_file_tracker.get_file_for_a2a_request(
                user_request=user_request,
                session_id=session_id
            )
            if file_path:
                # 파일 경로에서 파일명 추출
                file_name = os.path.basename(file_path)
                return file_name, f"UserFileTracker: {reason}"
        
        # Fallback: 기존 로직
        if not session_id or session_id not in self._session_metadata:
            return None, "세션을 찾을 수 없음"
        
        session_meta = self._session_metadata[session_id]
        
        if not session_meta.uploaded_files:
            return None, "업로드된 파일이 없음"
        
        # 1순위: 사용자가 명시적으로 파일명 언급
        filename_pattern = self.extract_filename_pattern(user_request)
        if filename_pattern:
            for file_meta in session_meta.uploaded_files:
                if filename_pattern in file_meta.data_id.lower():
                    return file_meta.data_id, f"파일명 패턴 일치: '{filename_pattern}'"
        
        # 2순위: 도메인 컨텍스트 매칭
        domain = self.extract_domain_from_request(user_request)
        if domain:
            for file_meta in session_meta.uploaded_files:
                if file_meta.domain == domain:
                    return file_meta.data_id, f"도메인 일치: {domain}"
        
        # 3순위: 가장 최근에 업로드된 파일
        latest_file = max(session_meta.uploaded_files, key=lambda f: f.uploaded_at)
        return latest_file.data_id, "가장 최근에 업로드된 파일"

    def create_session_with_data(self, data_id: str, data: pd.DataFrame, 
                                user_instructions: str, session_id: Optional[str] = None) -> str:
        """데이터와 함께 새로운 세션 생성 (Enhanced with UserFileTracker)"""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # 세션 디렉토리 생성
        session_dir = AI_DS_TEAM_DATA_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # 파일 크기 계산
        file_size = data.memory_usage(deep=True).sum()
        
        # 도메인 추출
        domain = self.extract_domain_from_request(user_instructions)
        
        # 파일 메타데이터 생성
        file_metadata = FileMetadata(
            data_id=data_id,
            uploaded_at=datetime.now(),
            file_size=int(file_size),
            file_type=Path(data_id).suffix.lower(),
            user_context=user_instructions,
            domain=domain,
            original_name=data_id
        )
        
        # 데이터 저장 (AI DS Team이 읽을 수 있는 형태로)
        if data_id.endswith('.xlsx') or data_id.endswith('.xls'):
            file_path = session_dir / data_id
            data.to_excel(file_path, index=False)
        elif data_id.endswith('.csv'):
            file_path = session_dir / data_id
            data.to_csv(file_path, index=False)
        else:
            # 기본적으로 CSV로 저장
            file_path = session_dir / f"{data_id}.csv"
            data.to_csv(file_path, index=False)
        
        # 🔥 UserFileTracker에 등록 (A2A 에이전트 호환성)
        if self.user_file_tracker:
            success = self.user_file_tracker.register_uploaded_file(
                file_id=data_id,
                original_name=data_id,
                session_id=session_id,
                data=data,
                user_context=user_instructions
            )
            if success:
                logging.info(f"✅ File registered in UserFileTracker: {data_id}")
            else:
                logging.warning(f"⚠️ Failed to register file in UserFileTracker: {data_id}")
        
        # 세션 메타데이터 생성 또는 업데이트
        if session_id in self._session_metadata:
            session_meta = self._session_metadata[session_id]
            session_meta.uploaded_files.append(file_metadata)
            session_meta.last_activity = datetime.now()
            session_meta.active_file = data_id  # 새로 업로드된 파일을 활성 파일로 설정
        else:
            session_meta = SessionMetadata(
                session_id=session_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                uploaded_files=[file_metadata],
                active_file=data_id
            )
            self._session_metadata[session_id] = session_meta
        
        # 컨텍스트 저장 (하위 호환성)
        context = {
            "user_instructions": user_instructions,
            "data_id": data_id,
            "data_shape": data.shape,
            "created_at": datetime.now().isoformat(),
            "file_path": str(file_path)
        }
        
        context_file = session_dir / "context.json"
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        
        # 메타데이터 저장
        self._save_session_metadata(session_id)
        
        # DataManager에도 등록
        self.data_manager.add_dataframe(data_id, data, f"Session {session_id}")
        
        # 현재 세션으로 설정
        self._current_session_id = session_id
        
        logging.info(f"Created session {session_id} with data {data_id}, domain: {domain}")
        return session_id

    def get_active_file_info(self, session_id: Optional[str] = None) -> Tuple[Optional[str], str]:
        """활성 파일 정보 반환"""
        if session_id is None:
            session_id = self._current_session_id
        
        if not session_id or session_id not in self._session_metadata:
            return None, "세션을 찾을 수 없음"
        
        session_meta = self._session_metadata[session_id]
        
        if session_meta.active_file:
            return session_meta.active_file, "세션의 활성 파일"
        elif session_meta.uploaded_files:
            # 활성 파일이 없으면 가장 최근 파일 반환
            latest_file = max(session_meta.uploaded_files, key=lambda f: f.uploaded_at)
            return latest_file.data_id, "가장 최근 업로드 파일"
        
        return None, "업로드된 파일이 없음"

    def get_session_files(self, session_id: Optional[str] = None) -> List[str]:
        """세션의 모든 파일 목록 반환"""
        if session_id is None:
            session_id = self._current_session_id
        
        if not session_id or session_id not in self._session_metadata:
            return []
        
        session_meta = self._session_metadata[session_id]
        return [f.data_id for f in session_meta.uploaded_files]

    def update_active_file(self, data_id: str, session_id: Optional[str] = None):
        """활성 파일 변경"""
        if session_id is None:
            session_id = self._current_session_id
        
        if session_id and session_id in self._session_metadata:
            session_meta = self._session_metadata[session_id]
            
            # 파일이 세션에 존재하는지 확인
            file_ids = [f.data_id for f in session_meta.uploaded_files]
            if data_id in file_ids:
                session_meta.active_file = data_id
                session_meta.last_activity = datetime.now()
                self._save_session_metadata(session_id)
                logging.info(f"Updated active file to {data_id} in session {session_id}")

    def check_session_age(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """세션 나이 및 상태 확인"""
        if session_id is None:
            session_id = self._current_session_id
        
        if not session_id or session_id not in self._session_metadata:
            return {"status": "not_found", "age_hours": 0}
        
        session_meta = self._session_metadata[session_id]
        now = datetime.now()
        age = now - session_meta.created_at
        age_hours = age.total_seconds() / 3600
        
        # 상태 결정
        if age_hours >= 24:
            status = "expired"
        elif age_hours >= 22:
            status = "warning"
        else:
            status = "active"
        
        # 세션 상태 업데이트
        if session_meta.status != status:
            session_meta.status = status
            self._save_session_metadata(session_id)
        
        return {
            "status": status,
            "age_hours": age_hours,
            "created_at": session_meta.created_at,
            "cleanup_at": session_meta.auto_cleanup_at,
            "hours_until_cleanup": max(0, (session_meta.auto_cleanup_at - now).total_seconds() / 3600)
        }

    def extend_session_lifetime(self, session_id: Optional[str] = None, hours: int = 24):
        """세션 생명주기 연장"""
        if session_id is None:
            session_id = self._current_session_id
        
        if session_id and session_id in self._session_metadata:
            session_meta = self._session_metadata[session_id]
            session_meta.auto_cleanup_at = datetime.now() + timedelta(hours=hours)
            session_meta.status = "active"
            session_meta.last_activity = datetime.now()
            self._save_session_metadata(session_id)
            logging.info(f"Extended session {session_id} lifetime by {hours} hours")

    # 기존 메서드들 (하위 호환성)
    def prepare_ai_ds_team_environment(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """AI DS Team 에이전트가 사용할 환경 준비 (UserFileTracker 통합)"""
        if session_id is None:
            session_id = self._current_session_id
        
        session_dir = AI_DS_TEAM_DATA_DIR / session_id if session_id else AI_DS_TEAM_DATA_DIR / "default"
        context = self.get_session_context(session_id) if session_id else {}
        
        # AI DS Team이 스캔할 수 있도록 현재 세션 데이터를 메인 data 폴더에 복사
        main_data_dir = Path("ai_ds_team/data")
        
        # 기존 파일들 정리 (default 제외)
        for item in main_data_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    logging.info(f"Cleaned up old file: {item.name}")
                except:
                    pass
        
        # 현재 세션 데이터를 메인 폴더에 복사
        if session_dir.exists():
            for file_path in session_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('.json'):
                    target_path = main_data_dir / file_path.name
                    try:
                        shutil.copy2(file_path, target_path)
                        logging.info(f"Copied {file_path.name} to main data directory")
                    except Exception as e:
                        logging.warning(f"Failed to copy {file_path.name}: {e}")
        
        # 🔥 A2A 공유 경로 정보도 포함
        shared_data_info = {}
        if self.user_file_tracker:
            files_info = self.user_file_tracker.get_session_files_info(session_id)
            if files_info:
                shared_data_info = {
                    "available_files": files_info,
                    "shared_path": "a2a_ds_servers/artifacts/data/shared_dataframes"
                }
        
        return {
            "session_id": session_id,
            "data_directory": str(session_dir),
            "context": context,
            "main_data_directory": str(main_data_dir),
            "shared_data_info": shared_data_info  # A2A 에이전트용 정보 추가
        }

    def get_current_session_id(self) -> Optional[str]:
        """현재 활성 세션 ID 반환"""
        return self._current_session_id

    def get_session_context(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """세션 컨텍스트 반환 (하위 호환성)"""
        if session_id is None:
            session_id = self._current_session_id
        
        if session_id and session_id in self._session_metadata:
            session_meta = self._session_metadata[session_id]
            if session_meta.uploaded_files:
                latest_file = max(session_meta.uploaded_files, key=lambda f: f.uploaded_at)
                return {
                    "user_instructions": latest_file.user_context or "",
                    "data_id": latest_file.data_id,
                    "data_shape": [0, 0],  # 실제 shape는 DataManager에서 가져와야 함
                    "created_at": session_meta.created_at.isoformat(),
                    "file_path": f"ai_ds_team/data/{session_id}/{latest_file.data_id}"
                }
        return None

    def get_file_for_a2a_agent(self, user_request: str, session_id: Optional[str] = None, agent_name: Optional[str] = None) -> Tuple[Optional[str], str]:
        """A2A 에이전트가 사용할 파일 경로 반환"""
        if self.user_file_tracker:
            return self.user_file_tracker.get_file_for_a2a_request(
                user_request=user_request,
                session_id=session_id,
                agent_name=agent_name
            )
        else:
            # Fallback: 기존 로직으로 파일 선택
            file_name, reason = self.smart_file_selection(user_request, session_id)
            if file_name:
                # shared_dataframes 경로에서 파일 찾기
                shared_path = Path("a2a_ds_servers/artifacts/data/shared_dataframes")
                potential_files = [
                    shared_path / file_name,
                    shared_path / f"{session_id or 'unknown'}_{file_name}",
                ]
                
                for file_path in potential_files:
                    if file_path.exists():
                        return str(file_path), reason
                
                # 세션 경로에서 찾기
                if session_id:
                    session_file_path = AI_DS_TEAM_DATA_DIR / session_id / file_name
                    if session_file_path.exists():
                        return str(session_file_path), reason
            
            return None, "A2A 에이전트용 파일을 찾을 수 없음"
