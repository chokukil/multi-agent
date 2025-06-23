# File: core/data_manager.py
import pandas as pd
import numpy as np
import threading
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

class DataManager:
    """
    통합 데이터 관리자 - Single Source of Truth (SSOT)
    모든 에이전트가 생성하고 사용하는 데이터프레임을 ID 기반으로 관리합니다.
    여러 데이터프레임을 동시에 메모리에 저장하고, 고유 ID를 통해 접근합니다.
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
        logging.info("DataManager initialized for multi-dataframe management.")

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
            logging.info(f"DataFrame added/updated with ID: {data_id} from source: {source}, shape={data.shape}")
            return data_id

    def get_dataframe(self, data_id: str) -> Optional[pd.DataFrame]:
        """
        주어진 ID에 해당하는 데이터프레임의 복사본을 반환합니다.
        """
        with self._lock:
            entry = self._data_store.get(data_id)
            if not entry:
                logging.error(f"DataFrame with ID '{data_id}' not found.")
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
        """Deletes a dataframe by its ID."""
        with self._lock:
            if data_id in self._data_store:
                del self._data_store[data_id]
                logging.info(f"DataFrame with ID '{data_id}' has been deleted.")
                return True
            logging.warning(f"Attempted to delete non-existent DataFrame with ID '{data_id}'.")
            return False

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
        """Returns a list of available dataframe IDs."""
        return list(self._data_store.keys())

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