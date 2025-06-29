# File: core/data_manager.py
import pandas as pd
import numpy as np
import threading
import hashlib
import json
import pickle
import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

# Shared directory for cross-process data sharing
SHARED_DATA_DIR = Path("artifacts/data/shared_dataframes")
SHARED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# AI DS Team data directory for session-based data
AI_DS_TEAM_DATA_DIR = Path("ai_ds_team/data")
AI_DS_TEAM_DATA_DIR.mkdir(parents=True, exist_ok=True)

class DataManager:
    """
    통합 데이터 관리자 - Single Source of Truth (SSOT)
    모든 에이전트가 생성하고 사용하는 데이터프레임을 ID 기반으로 관리합니다.
    여러 데이터프레임을 동시에 메모리에 저장하고, 고유 ID를 통해 접근합니다.
    프로세스 간 공유를 위해 파일 기반 백업을 제공합니다.
    세션 기반 AI DS Team 데이터 폴더 관리를 지원합니다.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._data_store: Dict[str, Dict[str, Any]] = {}
        
        
        self._initialized = True
        
        # Load existing data from shared storage
        self._load_from_shared_storage()
        
        # Initialize default AI DS Team data
        
        logging.info("DataManager initialized for multi-dataframe management with .")

    def _get_shared_file_path(self, data_id: str) -> Path:
        """Get the shared file path for a given data ID."""
        safe_id = "".join(c for c in data_id if c.isalnum() or c in ('_', '-', '.'))
        return SHARED_DATA_DIR / f"{safe_id}.pkl"

    def _save_to_shared_storage(self, data_id: str):
        """Save a dataframe to shared storage for cross-process access."""
        try:
            if data_id not in self._data_store:
                return
            
            entry = self._data_store[data_id]
            shared_data = {
                'data': entry['data'],
                'source': entry['source'],
                'created_at': entry['created_at'].isoformat(),
                'metadata': entry['metadata']
            }
            
            file_path = self._get_shared_file_path(data_id)
            with open(file_path, 'wb') as f:
                pickle.dump(shared_data, f)
            
            logging.info(f"📁 Saved dataframe '{data_id}' to shared storage: {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to save dataframe '{data_id}' to shared storage: {e}")

    def _load_from_shared_storage(self):
        """Load all dataframes from shared storage."""
        try:
            for file_path in SHARED_DATA_DIR.glob("*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        shared_data = pickle.load(f)
                    
                    data_id = file_path.stem
                    if data_id not in self._data_store:
                        entry = {
                            "data": shared_data['data'],
                            "hash": self._compute_hash(shared_data['data']),
                            "source": shared_data.get('source', 'Loaded from shared storage'),
                            "created_at": datetime.fromisoformat(shared_data['created_at']),
                            "access_count": 0,
                            "metadata": shared_data.get('metadata', {})
                        }
                        self._data_store[data_id] = entry
                        logging.info(f"📥 Loaded dataframe '{data_id}' from shared storage")
                        
                except Exception as e:
                    logging.warning(f"Failed to load dataframe from {file_path}: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to load from shared storage: {e}")

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """데이터프레임의 해시 계산"""
        if data is None or not isinstance(data, pd.DataFrame):
            return ""
        hash_str = pd.util.hash_pandas_object(data, index=True).to_string()
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def add_dataframe(self, data_id: str, data: pd.DataFrame, source: str = "Unknown") -> Optional[str]:
        """
        주어진 ID로 새로운 데이터프레임을 등록합니다.
        """
        if data is None or not isinstance(data, pd.DataFrame):
            logging.warning(f"Attempted to add a None or invalid DataFrame with id: {data_id}.")
            return None

        with self._lock:
            if data_id in self._data_store:
                logging.warning(f"DataFrame with ID '{data_id}' already exists. Overwriting.")

            new_entry = {
                "data": data.copy(),
                "hash": self._compute_hash(data),
                "source": source,
                "created_at": datetime.now(),
                "access_count": 0,
                "metadata": self._extract_metadata(data)
            }
            
            self._data_store[data_id] = new_entry
            
            # Save to shared storage for cross-process access
            self._save_to_shared_storage(data_id)
            
            logging.info(f"DataFrame added/updated with ID: {data_id} from source: {source}, shape={data.shape}")
            return data_id

    def get_dataframe(self, data_id: str) -> Optional[pd.DataFrame]:
        """
        주어진 ID에 해당하는 데이터프레임의 복사본을 반환합니다.
        """
        with self._lock:
            # Try to load from memory first
            entry = self._data_store.get(data_id)
            
            # If not in memory, try to load from shared storage
            if not entry:
                shared_file = self._get_shared_file_path(data_id)
                if shared_file.exists():
                    logging.info(f"🔄 Loading dataframe '{data_id}' from shared storage")
                    self._load_from_shared_storage()
                    entry = self._data_store.get(data_id)
            
            if not entry:
                logging.error(f"DataFrame with ID '{data_id}' not found in memory or shared storage.")
                return None
            
            entry["access_count"] += 1
            return entry["data"].copy()

    def get_data_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """ID에 해당하는 데이터의 정보를 반환합니다."""
        with self._lock:
            entry = self._data_store.get(data_id)
            if not entry:
                logging.error(f"Info request failed: DataFrame with ID '{data_id}' not found.")
                return None

            info = {k: v for k, v in entry.items() if k != 'data'}
            info["data_id"] = data_id
            info["created_at"] = info["created_at"].isoformat()
            return info

    def list_dataframe_info(self) -> List[Dict[str, Any]]:
        """저장된 모든 데이터프레임의 요약 정보를 리스트로 반환합니다."""
        with self._lock:
            summary_list = []
            for data_id, entry in self._data_store.items():
                summary = {
                    "data_id": data_id,
                    "source": entry["source"],
                    "shape": entry["data"].shape,
                    "created_at": entry["created_at"].isoformat(),
                    "access_count": entry["access_count"]
                }
                summary_list.append(summary)
            return summary_list

    def delete_dataframe(self, data_id: str) -> bool:
        """Deletes a dataframe by its ID from both memory and shared storage."""
        with self._lock:
            deleted = False
            
            # Delete from memory
            if data_id in self._data_store:
                del self._data_store[data_id]
                logging.info(f"DataFrame with ID '{data_id}' has been deleted from memory.")
                deleted = True
            
            # Delete from shared storage
            shared_file = self._get_shared_file_path(data_id)
            if shared_file.exists():
                try:
                    shared_file.unlink()
                    logging.info(f"DataFrame file '{data_id}' has been deleted from shared storage: {shared_file}")
                    deleted = True
                except Exception as e:
                    logging.error(f"Failed to delete shared file for '{data_id}': {e}")
            
            if not deleted:
                logging.warning(f"Attempted to delete non-existent DataFrame with ID '{data_id}'.")
            
            return deleted

    def clear(self):
        """Clears all dataframes from the manager. Used for testing."""
        with self._lock:
            count = len(self._data_store)
            self._data_store.clear()
            logging.info(f"All {count} DataFrames have been cleared from DataManager.")

    def _extract_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """데이터프레임에서 메타데이터를 추출합니다."""
        return {
            "shape": data.shape,
            "row_count": len(data),
            "col_count": len(data.columns),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_count": int(data.isnull().sum().sum()),
        }

    def list_dataframes(self) -> List[str]:
        """저장된 모든 데이터프레임의 ID 목록을 반환합니다."""
        with self._lock:
            return list(self._data_store.keys())

    def create_session(self, session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """새로운 세션 생성"""
        import uuid
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        session_dir = AI_DS_TEAM_DATA_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # 세션 컨텍스트 저장
        if context:
            self._session_contexts[session_id] = context
            context_file = session_dir / "context.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, ensure_ascii=False, indent=2)
        
        self._current_session_id = session_id
        logging.info(f"Created new session: {session_id}")
        return session_id

    def set_current_session(self, session_id: str) -> bool:
        """현재 활성 세션 설정"""
        session_dir = AI_DS_TEAM_DATA_DIR / session_id
        if session_dir.exists():
            self._current_session_id = session_id
            
            # 세션 컨텍스트 로드
            context_file = session_dir / "context.json"
            if context_file.exists():
                try:
                    with open(context_file, 'r', encoding='utf-8') as f:
                        self._session_contexts[session_id] = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load session context: {e}")
            
            logging.info(f"Set current session to: {session_id}")
            return True
        else:
            logging.warning(f"Session not found: {session_id}")
            return False

    def get_current_session_id(self) -> Optional[str]:
        """현재 활성 세션 ID 반환"""
        return self._current_session_id

    def get_session_context(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """세션 컨텍스트 반환"""
        if session_id is None:
            session_id = self._current_session_id
        
        if session_id:
            return self._session_contexts.get(session_id)
        return None

    def add_data_to_session(self, data_id: str, data: pd.DataFrame, session_id: Optional[str] = None, 
                           source: str = "Unknown", context: Optional[Dict[str, Any]] = None) -> str:
        """세션별 데이터 추가"""
        if session_id is None:
            session_id = self._current_session_id
            
        if session_id is None:
            # 세션이 없으면 새로 생성
            session_id = self.create_session(context=context)
        
        # 일반 DataManager에 추가
        self.add_dataframe(data_id, data, source)
        
        # AI DS Team 세션 폴더에 파일 저장
        session_dir = AI_DS_TEAM_DATA_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # 파일 확장자 결정
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
        
        # 컨텍스트 업데이트
        if context:
            self._session_contexts[session_id] = context
            context_file = session_dir / "context.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(context, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Added data to session {session_id}: {data_id} -> {file_path}")
        return session_id

    def get_session_data_directory(self, session_id: Optional[str] = None) -> Path:
        """세션 데이터 디렉토리 경로 반환"""
        if session_id is None:
            session_id = self._current_session_id
            
        if session_id:
            return AI_DS_TEAM_DATA_DIR / session_id
        else:
            # 세션이 없으면 default 폴더 반환
            return AI_DS_TEAM_DATA_DIR / "default"

    def list_sessions(self) -> List[str]:
        """모든 세션 목록 반환"""
        sessions = []
        for item in AI_DS_TEAM_DATA_DIR.iterdir():
            if item.is_dir() and item.name != "default":
                sessions.append(item.name)
        return sessions

    def cleanup_session(self, session_id: str) -> bool:
        """세션 정리 (폴더 삭제)"""
        session_dir = AI_DS_TEAM_DATA_DIR / session_id
        if session_dir.exists() and session_dir.name != "default":
            try:
                shutil.rmtree(session_dir)
                if session_id in self._session_contexts:
                    del self._session_contexts[session_id]
                if self._current_session_id == session_id:
                    self._current_session_id = None
                logging.info(f"Cleaned up session: {session_id}")
                return True
            except Exception as e:
                logging.error(f"Failed to cleanup session {session_id}: {e}")
        return False


# --- 기존 하위 호환성 함수들 ---
def get_current_df() -> Optional[pd.DataFrame]:
    logging.warning("get_current_df is deprecated. Use DataManager().get_dataframe(data_id) instead.")
    return None

def load_data(file_path: str) -> pd.DataFrame:
    logging.warning("load_data is deprecated. Use specific loader agents instead.")
    raise NotImplementedError

def check_data_status() -> Dict[str, Any]:
    logging.warning("check_data_status is deprecated. Use DataManager().list_dataframe_info() instead.")
    dm = DataManager()
    return {"managed_dataframe_count": len(dm.list_dataframe_info())}

def show_data_info():
    logging.warning("show_data_info is deprecated.")
    dm = DataManager()
    print(dm.list_dataframe_info())

# Singleton instance for global access
data_manager = DataManager()