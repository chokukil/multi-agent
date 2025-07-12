"""
안전한 데이터 로딩 유틸리티

A2A 에이전트들의 변수 초기화 문제를 해결하기 위한 통합 데이터 로딩 시스템
"""

import os
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class SafeDataLoader:
    """안전한 데이터 로딩 클래스"""
    
    def __init__(self, data_path: str = "a2a_ds_servers/artifacts/data/shared_dataframes/"):
        self.data_path = data_path
        self.supported_formats = ['.csv', '.pkl', '.xlsx', '.xls', '.json']
        
    def load_data_safely(self, 
                        available_data: List[str] = None, 
                        preferred_file: str = None,
                        fallback_strategy: str = 'latest') -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """
        안전한 데이터 로딩 함수
        
        Args:
            available_data: 사용 가능한 데이터 파일 목록
            preferred_file: 우선 선택할 파일명
            fallback_strategy: 폴백 전략 ('latest', 'first', 'largest')
            
        Returns:
            Tuple[DataFrame, selected_file, error_message]
        """
        df = None
        selected_file = None
        error_msg = None
        
        try:
            # 1. 사용 가능한 데이터 목록 확인
            if available_data is None:
                available_data = self._scan_available_data()
            
            if not available_data:
                error_msg = "❌ 사용 가능한 데이터 파일이 없습니다."
                return df, selected_file, error_msg
            
            # 2. 파일 선택 로직
            selected_file = self._select_best_file(available_data, preferred_file, fallback_strategy)
            
            if not selected_file:
                error_msg = "❌ 적절한 데이터 파일을 선택할 수 없습니다."
                return df, selected_file, error_msg
            
            # 3. 데이터 로딩 시도
            df = self._load_file_safely(selected_file)
            
            if df is None:
                error_msg = f"❌ 데이터 로딩 실패: {selected_file}"
                return df, selected_file, error_msg
            
            logger.info(f"✅ 데이터 로딩 성공: {selected_file}, 형태: {df.shape}")
            return df, selected_file, None
            
        except Exception as e:
            error_msg = f"❌ 데이터 로딩 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return df, selected_file, error_msg
    
    def _scan_available_data(self) -> List[str]:
        """사용 가능한 데이터 파일 스캔"""
        available_data = []
        
        try:
            if os.path.exists(self.data_path):
                for file in os.listdir(self.data_path):
                    if any(file.endswith(fmt) for fmt in self.supported_formats):
                        available_data.append(file)
        except Exception as e:
            logger.warning(f"데이터 스캔 실패: {e}")
        
        return available_data
    
    def _select_best_file(self, 
                         available_data: List[str], 
                         preferred_file: str = None,
                         fallback_strategy: str = 'latest') -> Optional[str]:
        """최적 파일 선택 로직"""
        
        # 1. 우선 파일 확인
        if preferred_file and preferred_file in available_data:
            return preferred_file
        
        # 2. 특정 패턴 우선 (ion_implant 데이터 등)
        priority_patterns = ['ion_implant', 'main', 'primary', 'data']
        for pattern in priority_patterns:
            for file in available_data:
                if pattern.lower() in file.lower():
                    return file
        
        # 3. 폴백 전략 적용
        if fallback_strategy == 'latest':
            return self._get_latest_file(available_data)
        elif fallback_strategy == 'first':
            return available_data[0] if available_data else None
        elif fallback_strategy == 'largest':
            return self._get_largest_file(available_data)
        
        return available_data[0] if available_data else None
    
    def _get_latest_file(self, available_data: List[str]) -> Optional[str]:
        """가장 최근 파일 선택"""
        try:
            file_times = []
            for file in available_data:
                file_path = os.path.join(self.data_path, file)
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    file_times.append((file, mtime))
            
            if file_times:
                file_times.sort(key=lambda x: x[1], reverse=True)
                return file_times[0][0]
        except Exception as e:
            logger.warning(f"최근 파일 선택 실패: {e}")
        
        return available_data[0] if available_data else None
    
    def _get_largest_file(self, available_data: List[str]) -> Optional[str]:
        """가장 큰 파일 선택"""
        try:
            file_sizes = []
            for file in available_data:
                file_path = os.path.join(self.data_path, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    file_sizes.append((file, size))
            
            if file_sizes:
                file_sizes.sort(key=lambda x: x[1], reverse=True)
                return file_sizes[0][0]
        except Exception as e:
            logger.warning(f"큰 파일 선택 실패: {e}")
        
        return available_data[0] if available_data else None
    
    def _load_file_safely(self, filename: str) -> Optional[pd.DataFrame]:
        """안전한 파일 로딩"""
        if not filename:
            return None
        
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"파일 없음: {file_path}")
            return None
        
        try:
            if filename.endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.endswith('.pkl'):
                return pd.read_pickle(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            elif filename.endswith('.json'):
                return pd.read_json(file_path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"파일 로딩 오류 {filename}: {e}")
            return None
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, any]:
        """데이터프레임 검증"""
        if df is None:
            return {"valid": False, "error": "DataFrame이 None입니다."}
        
        if df.empty:
            return {"valid": False, "error": "DataFrame이 비어있습니다."}
        
        return {
            "valid": True,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }

# 전역 인스턴스
safe_data_loader = SafeDataLoader()

def load_data_safely(available_data: List[str] = None, 
                    preferred_file: str = None,
                    fallback_strategy: str = 'latest') -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """전역 안전한 데이터 로딩 함수"""
    return safe_data_loader.load_data_safely(available_data, preferred_file, fallback_strategy)

def create_safe_data_response(df: pd.DataFrame, 
                            selected_file: str, 
                            user_instructions: str,
                            agent_name: str) -> str:
    """안전한 데이터 응답 메시지 생성"""
    if df is None:
        return f"""## ❌ 데이터 없음

{agent_name} 작업을 수행하려면 먼저 데이터를 업로드해야 합니다.

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "data.xlsx 파일을 분석해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

**요청**: {user_instructions}
"""
    
    validation = safe_data_loader.validate_dataframe(df)
    
    if not validation["valid"]:
        return f"""## ❌ 데이터 검증 실패

선택된 파일: {selected_file}
오류: {validation["error"]}

**요청**: {user_instructions}
"""
    
    return f"""## ✅ 데이터 로딩 성공

**선택된 파일**: {selected_file}
**데이터 형태**: {validation["shape"]}
**컬럼 수**: {len(validation["columns"])}
**메모리 사용량**: {validation["memory_usage"]:,} bytes

**요청**: {user_instructions}
""" 