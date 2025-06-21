# File: core/data_manager.py
# Location: ./core/data_manager.py

import pandas as pd
import numpy as np
import threading
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Callable
import logging

class UnifiedDataManager:
    """
    통합 데이터 관리자 - Single Source of Truth (SSOT)
    모든 에이전트가 동일한 데이터를 참조하도록 보장
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
        if self._initialized:
            return
            
        self._data: Optional[pd.DataFrame] = None
        self._data_hash: Optional[str] = None
        self._source_info: str = ""
        self._last_update: Optional[datetime] = None
        self._access_count: int = 0
        self._modification_history: List[Dict] = []
        self._change_listeners: List[Callable] = []
        self._metadata: Dict[str, Any] = {}
        self._initialized = True
        
        logging.info("UnifiedDataManager initialized")
    
    def _compute_hash(self, data: pd.DataFrame) -> str:
        """데이터프레임의 해시 계산"""
        # 데이터프레임의 주요 특성을 포함한 해시 생성
        hash_dict = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "head": data.head(10).to_dict() if len(data) > 0 else {},
            "null_counts": data.isnull().sum().to_dict()
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def set_data(self, data: pd.DataFrame, source: str = "Unknown") -> bool:
        """
        데이터 설정 (모든 에이전트가 이 데이터를 참조)
        
        Args:
            data: 설정할 데이터프레임
            source: 데이터 출처 정보
        
        Returns:
            성공 여부
        """
        with self._lock:
            try:
                if data is None or data.empty:
                    logging.warning("Attempted to set empty or None data")
                    return False
                
                # 이전 데이터 백업
                old_data = self._data
                old_hash = self._data_hash
                
                # 새 데이터 설정
                self._data = data.copy()  # 복사본 저장으로 외부 변경 방지
                self._data_hash = self._compute_hash(self._data)
                self._source_info = source
                self._last_update = datetime.now()
                
                # 메타데이터 업데이트
                self._update_metadata()
                
                # 변경 이력 기록
                self._modification_history.append({
                    "timestamp": self._last_update.isoformat(),
                    "action": "set_data",
                    "source": source,
                    "old_hash": old_hash,
                    "new_hash": self._data_hash,
                    "shape": self._data.shape
                })
                
                # 변경 리스너 호출
                self._notify_listeners("data_set", {
                    "source": source,
                    "shape": self._data.shape,
                    "columns": list(self._data.columns)
                })
                
                logging.info(f"Data set successfully: {source}, shape={self._data.shape}")
                return True
                
            except Exception as e:
                logging.error(f"Error setting data: {e}")
                return False
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        현재 데이터 반환 (복사본)
        
        Returns:
            데이터프레임 복사본 또는 None
        """
        with self._lock:
            self._access_count += 1
            
            if self._data is None:
                logging.warning("Data requested but none is loaded")
                return None
            
            # 복사본 반환으로 외부 수정 방지
            return self._data.copy()
    
    def is_data_loaded(self) -> bool:
        """데이터가 로드되어 있는지 확인"""
        with self._lock:
            return self._data is not None
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터 정보 반환"""
        with self._lock:
            if self._data is None:
                return {
                    "loaded": False,
                    "message": "No data loaded"
                }
            
            return {
                "loaded": True,
                "source": self._source_info,
                "shape": self._data.shape,
                "row_count": len(self._data),
                "col_count": len(self._data.columns),
                "columns": list(self._data.columns),
                "dtypes": {col: str(dtype) for col, dtype in self._data.dtypes.items()},
                "memory_mb": self._data.memory_usage(deep=True).sum() / 1024 / 1024,
                "last_update": self._last_update.isoformat() if self._last_update else None,
                "access_count": self._access_count,
                "data_hash": self._data_hash,
                "null_count": int(self._data.isnull().sum().sum()),
                "numeric_cols": list(self._data.select_dtypes(include=[np.number]).columns),
                "categorical_cols": list(self._data.select_dtypes(include=['object', 'category']).columns),
                **self._metadata
            }
    
    def validate_data_consistency(self) -> Tuple[bool, str]:
        """
        데이터 일관성 검증
        
        Returns:
            (일관성 여부, 메시지)
        """
        with self._lock:
            if self._data is None:
                return False, "No data loaded"
            
            try:
                # 현재 해시 재계산
                current_hash = self._compute_hash(self._data)
                
                if current_hash != self._data_hash:
                    return False, f"Data integrity compromised! Expected hash: {self._data_hash}, Current: {current_hash}"
                
                # 기본 검증
                checks = []
                
                # 1. 데이터 크기 검증
                if len(self._data) == 0:
                    checks.append("Warning: Data has 0 rows")
                
                # 2. 컬럼 검증
                if len(self._data.columns) == 0:
                    checks.append("Warning: Data has 0 columns")
                
                # 3. 전체 null 검증
                if self._data.isnull().all().all():
                    checks.append("Warning: All data values are null")
                
                if checks:
                    return True, f"Data consistent but with warnings: {'; '.join(checks)}"
                
                return True, f"Data consistency verified (hash: {self._data_hash[:8]}...)"
                
            except Exception as e:
                return False, f"Consistency check failed: {str(e)}"
    
    def get_status_message(self) -> str:
        """현재 상태 메시지 반환"""
        with self._lock:
            if self._data is None:
                return "No data loaded"
            
            return f"Data ready: {self._source_info} ({len(self._data):,} rows × {len(self._data.columns):,} cols)"
    
    def clear_data(self):
        """데이터 초기화"""
        with self._lock:
            old_source = self._source_info
            
            self._data = None
            self._data_hash = None
            self._source_info = ""
            self._last_update = None
            self._access_count = 0
            self._metadata = {}
            
            # 변경 이력 기록
            self._modification_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "clear_data",
                "old_source": old_source
            })
            
            # 리스너 알림
            self._notify_listeners("data_cleared", {"old_source": old_source})
            
            logging.info("Data cleared")
    
    def _update_metadata(self):
        """메타데이터 업데이트"""
        if self._data is None:
            return
        
        # 기본 통계
        numeric_data = self._data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            self._metadata["numeric_stats"] = {
                col: {
                    "mean": float(numeric_data[col].mean()),
                    "std": float(numeric_data[col].std()),
                    "min": float(numeric_data[col].min()),
                    "max": float(numeric_data[col].max())
                } for col in numeric_data.columns
            }
    
    def add_change_listener(self, callback: Callable):
        """변경 리스너 추가"""
        with self._lock:
            self._change_listeners.append(callback)
    
    def _notify_listeners(self, event: str, data: Dict):
        """리스너들에게 변경 알림"""
        for listener in self._change_listeners:
            try:
                listener(event, data)
            except Exception as e:
                logging.error(f"Error in change listener: {e}")
    
    def get_modification_history(self) -> List[Dict]:
        """수정 이력 반환"""
        with self._lock:
            return self._modification_history.copy()

# 전역 인스턴스
data_manager = UnifiedDataManager()

# 에이전트들이 사용할 헬퍼 함수들
def get_current_df() -> Optional[pd.DataFrame]:
    """현재 데이터프레임 반환 (에이전트용)"""
    return data_manager.get_data()

def load_data(file_path: str) -> pd.DataFrame:
    """데이터 로드 (SSOT에 자동 등록)"""
    try:
        df = pd.read_csv(file_path)
        data_manager.set_data(df, f"Loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def check_data_status() -> Dict[str, Any]:
    """데이터 상태 확인 (에이전트용)"""
    return data_manager.get_data_info()

def show_data_info():
    """데이터 정보 출력 (에이전트용)"""
    info = data_manager.get_data_info()
    if info["loaded"]:
        print(f"📊 Data Status: Loaded")
        print(f"   Source: {info['source']}")
        print(f"   Shape: {info['shape']}")
        print(f"   Columns: {', '.join(info['columns'][:5])}...")
        print(f"   Memory: {info['memory_mb']:.2f} MB")
    else:
        print("❌ No data loaded")
    return info

def create_unified_data_access_functions():
    """에이전트가 사용할 수 있는 통합 데이터 접근 함수들 생성"""
    return {
        "get_current_data": get_current_df,
        "load_data": load_data,
        "check_data_status": check_data_status,
        "show_data_info": show_data_info,
    }