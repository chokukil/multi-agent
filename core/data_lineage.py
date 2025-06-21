# File: core/data_lineage.py
# Location: ./core/data_lineage.py

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class DataLineageTracker:
    """데이터 계보 추적 및 검증 시스템"""
    
    def __init__(self):
        self.original_data = None
        self.original_hash = None
        self.original_metadata = None
        self.transformations = []
        
    def _compute_hash(self, data: Any) -> str:
        """데이터의 해시값 계산"""
        if isinstance(data, pd.DataFrame):
            # DataFrame의 경우 shape, columns, dtypes, 샘플 데이터로 해시 생성
            hash_dict = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "sample": data.head(10).to_dict() if len(data) > 0 else {}
            }
            hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        else:
            hash_str = str(data)
            
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def _extract_metadata(self, data: pd.DataFrame) -> Dict:
        """데이터프레임의 메타데이터 추출"""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_usage": data.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            "null_counts": data.isnull().sum().to_dict()
        }
    
    def set_original_data(self, data: pd.DataFrame) -> str:
        """원본 데이터 설정 및 해시 생성"""
        self.original_data = data.copy()
        self.original_hash = self._compute_hash(data)
        self.original_metadata = self._extract_metadata(data)
        
        self.transformations = [{
            "step": 0,
            "executor": "SSOT",
            "operation": "initial_load",
            "hash": self.original_hash,
            "metadata": self.original_metadata,
            "timestamp": datetime.now().isoformat(),
            "description": "원본 데이터 로드"
        }]
        
        return self.original_hash
    
    def track_transformation(self, executor_name: str, operation: str, 
                           current_data: pd.DataFrame, description: str) -> Dict:
        """데이터 변환 추적"""
        current_hash = self._compute_hash(current_data)
        current_metadata = self._extract_metadata(current_data)
        
        # 이전 상태와 비교
        prev_transform = self.transformations[-1] if self.transformations else None
        changes = self._detect_changes(prev_transform["metadata"] if prev_transform else {}, 
                                     current_metadata)
        
        transformation = {
            "step": len(self.transformations),
            "executor": executor_name,
            "operation": operation,
            "hash": current_hash,
            "metadata": current_metadata,
            "changes": changes,
            "timestamp": datetime.now().isoformat(),
            "description": description
        }
        
        self.transformations.append(transformation)
        return transformation
    
    def _detect_changes(self, before: Dict, after: Dict) -> Dict:
        """메타데이터 변경 사항 감지"""
        changes = {
            "rows_changed": after.get("shape", [0])[0] - before.get("shape", [0])[0],
            "cols_changed": after.get("shape", [0])[1] - before.get("shape", [0])[1],
            "columns_added": list(set(after.get("columns", [])) - set(before.get("columns", []))),
            "columns_removed": list(set(before.get("columns", [])) - set(after.get("columns", []))),
            "memory_change": after.get("memory_usage", 0) - before.get("memory_usage", 0)
        }
        return changes
    
    def validate_data_consistency(self, current_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """현재 데이터가 원본에서 파생되었는지 검증"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "lineage_intact": True
        }
        
        if self.original_data is None:
            validation_result["is_valid"] = False
            validation_result["errors"].append("원본 데이터가 설정되지 않음")
            return False, validation_result
        
        current_metadata = self._extract_metadata(current_data)
        
        # 1. 컬럼 손실 검증
        original_cols = set(self.original_metadata["columns"])
        current_cols = set(current_metadata["columns"])
        lost_cols = original_cols - current_cols
        
        if len(lost_cols) > len(original_cols) * 0.5:
            validation_result["warnings"].append(f"원본 컬럼의 50% 이상 손실: {lost_cols}")
        
        # 2. 데이터 크기 검증
        original_rows = self.original_metadata["shape"][0]
        current_rows = current_metadata["shape"][0]
        
        if current_rows > original_rows * 2:
            validation_result["warnings"].append(f"데이터 행이 2배 이상 증가: {original_rows} → {current_rows}")
        elif current_rows < original_rows * 0.1:
            validation_result["warnings"].append(f"데이터 행이 90% 이상 감소: {original_rows} → {current_rows}")
        
        # 3. 데이터 타입 호환성 검증
        for col in current_cols & original_cols:
            if current_metadata["dtypes"].get(col) != self.original_metadata["dtypes"].get(col):
                validation_result["warnings"].append(
                    f"컬럼 '{col}'의 데이터 타입 변경: "
                    f"{self.original_metadata['dtypes'].get(col)} → {current_metadata['dtypes'].get(col)}"
                )
        
        # 4. 변환 이력 연속성 검증
        if len(self.transformations) > 1:
            for i in range(1, len(self.transformations)):
                if self.transformations[i]["step"] != i:
                    validation_result["lineage_intact"] = False
                    validation_result["errors"].append(f"변환 이력 불연속: 단계 {i}")
        
        validation_result["is_valid"] = len(validation_result["errors"]) == 0
        
        return validation_result["is_valid"], validation_result
    
    def detect_suspicious_patterns(self) -> List[Dict]:
        """의심스러운 데이터 사용 패턴 감지"""
        suspicious_patterns = []
        
        if len(self.transformations) < 2:
            return suspicious_patterns
        
        for i in range(1, len(self.transformations)):
            current = self.transformations[i]
            previous = self.transformations[i-1]
            
            # 패턴 1: 갑작스러운 데이터 교체 (해시가 완전히 다르고 공통 컬럼이 거의 없음)
            if current["hash"] != previous["hash"]:
                prev_cols = set(previous["metadata"]["columns"])
                curr_cols = set(current["metadata"]["columns"])
                common_cols = prev_cols & curr_cols
                
                if len(common_cols) < len(prev_cols) * 0.2:
                    suspicious_patterns.append({
                        "type": "sudden_data_replacement",
                        "step": i,
                        "executor": current["executor"],
                        "description": "데이터가 갑작스럽게 교체됨 (공통 컬럼 20% 미만)"
                    })
            
            # 패턴 2: 비정상적인 행 증가
            prev_rows = previous["metadata"]["shape"][0]
            curr_rows = current["metadata"]["shape"][0]
            
            if curr_rows > prev_rows * 10:
                suspicious_patterns.append({
                    "type": "abnormal_row_increase",
                    "step": i,
                    "executor": current["executor"],
                    "description": f"비정상적인 행 증가: {prev_rows} → {curr_rows} (10배 이상)"
                })
            
            # 패턴 3: 메모리 사용량 급증
            prev_memory = previous["metadata"]["memory_usage"]
            curr_memory = current["metadata"]["memory_usage"]
            
            if curr_memory > prev_memory * 5:
                suspicious_patterns.append({
                    "type": "memory_explosion",
                    "step": i,
                    "executor": current["executor"],
                    "description": f"메모리 사용량 급증: {prev_memory:.2f}MB → {curr_memory:.2f}MB"
                })
        
        return suspicious_patterns
    
    def get_lineage_summary(self) -> Dict:
        """데이터 계보 요약 정보 반환"""
        if not self.transformations:
            return {"error": "No transformations tracked"}
        
        return {
            "original_data": {
                "hash": self.original_hash,
                "shape": self.original_metadata["shape"],
                "columns": len(self.original_metadata["columns"])
            },
            "total_transformations": len(self.transformations) - 1,  # 초기 로드 제외
            "executors_involved": list(set(t["executor"] for t in self.transformations[1:])),
            "final_data": {
                "hash": self.transformations[-1]["hash"],
                "shape": self.transformations[-1]["metadata"]["shape"],
                "columns": len(self.transformations[-1]["metadata"]["columns"])
            },
            "suspicious_patterns": self.detect_suspicious_patterns()
        }

# 전역 인스턴스
data_lineage_tracker = DataLineageTracker()