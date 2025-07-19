"""
Common Data Processor Module

모든 A2A 서버에서 사용하는 공통 데이터 처리 유틸리티
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
import aiofiles
import io
import csv

logger = logging.getLogger(__name__)


class CommonDataProcessor:
    """공통 데이터 처리기"""
    
    @staticmethod
    async def parse_csv_data(csv_data: str) -> pd.DataFrame:
        """
        CSV 문자열을 DataFrame으로 파싱
        
        Args:
            csv_data: CSV 형식의 문자열
            
        Returns:
            파싱된 DataFrame
        """
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ CSV 파싱 오류: {e}")
            raise
            
    @staticmethod
    async def parse_json_data(json_data: str) -> pd.DataFrame:
        """
        JSON 문자열을 DataFrame으로 파싱
        
        Args:
            json_data: JSON 형식의 문자열
            
        Returns:
            파싱된 DataFrame
        """
        try:
            data = json.loads(json_data)
            
            # JSON 배열인 경우
            if isinstance(data, list):
                df = pd.DataFrame(data)
            # JSON 객체인 경우
            elif isinstance(data, dict):
                # records 형식인지 확인
                if 'data' in data and isinstance(data['data'], list):
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"지원하지 않는 JSON 형식: {type(data)}")
                
            logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ JSON 파싱 오류: {e}")
            raise
            
    @staticmethod
    def generate_sample_data(
        rows: int = 100,
        columns: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        샘플 데이터 생성
        
        Args:
            rows: 행 수
            columns: 컬럼 리스트 (없으면 기본 컬럼 사용)
            seed: 랜덤 시드
            
        Returns:
            생성된 샘플 DataFrame
        """
        if seed:
            np.random.seed(seed)
            
        if not columns:
            columns = ['id', 'name', 'age', 'score', 'category', 'created_at']
            
        data = {}
        
        for col in columns:
            if col == 'id':
                data[col] = range(1, rows + 1)
            elif col == 'name':
                data[col] = [f"User_{i}" for i in range(1, rows + 1)]
            elif col == 'age':
                data[col] = np.random.randint(18, 65, size=rows)
            elif col == 'score':
                data[col] = np.random.uniform(0, 100, size=rows)
            elif col == 'category':
                categories = ['A', 'B', 'C', 'D']
                data[col] = np.random.choice(categories, size=rows)
            elif col == 'created_at':
                base = pd.Timestamp('2024-01-01')
                data[col] = [base + pd.Timedelta(days=x) for x in range(rows)]
            else:
                # 기본값으로 랜덤 float
                data[col] = np.random.random(size=rows)
                
        df = pd.DataFrame(data)
        logger.info(f"✅ 샘플 데이터 생성 완료: {df.shape}")
        return df
        
    @staticmethod
    def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        DataFrame 정보 추출
        
        Args:
            df: 정보를 추출할 DataFrame
            
        Returns:
            DataFrame 정보 딕셔너리
        """
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns}
        }
        
        # 수치형 컬럼 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
        return info
        
    @staticmethod
    async def save_dataframe(
        df: pd.DataFrame,
        filepath: Union[str, Path],
        format: str = "csv"
    ) -> str:
        """
        DataFrame을 파일로 저장
        
        Args:
            df: 저장할 DataFrame
            filepath: 저장 경로
            format: 파일 형식 ('csv', 'json', 'parquet')
            
        Returns:
            저장된 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient='records', indent=2)
            elif format == "parquet":
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
                
            logger.info(f"✅ DataFrame 저장 완료: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"❌ DataFrame 저장 오류: {e}")
            raise
            
    @staticmethod
    async def load_dataframe(
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        파일에서 DataFrame 로드
        
        Args:
            filepath: 파일 경로
            format: 파일 형식 (None이면 확장자로 추론)
            
        Returns:
            로드된 DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {filepath}")
            
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
            
        try:
            if format == "csv":
                df = pd.read_csv(filepath)
            elif format == "json":
                df = pd.read_json(filepath)
            elif format == "parquet":
                df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"지원하지 않는 형식: {format}")
                
            logger.info(f"✅ DataFrame 로드 완료: {filepath} - shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ DataFrame 로드 오류: {e}")
            raise
            
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        DataFrame 유효성 검증
        
        Args:
            df: 검증할 DataFrame
            required_columns: 필수 컬럼 리스트
            min_rows: 최소 행 수
            
        Returns:
            (유효 여부, 오류 메시지)
        """
        # DataFrame 타입 체크
        if not isinstance(df, pd.DataFrame):
            return False, "입력이 DataFrame이 아닙니다"
            
        # 빈 DataFrame 체크
        if df.empty:
            return False, "DataFrame이 비어있습니다"
            
        # 최소 행 수 체크
        if len(df) < min_rows:
            return False, f"최소 {min_rows}개 이상의 행이 필요합니다 (현재: {len(df)})"
            
        # 필수 컬럼 체크
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"필수 컬럼이 없습니다: {missing_cols}"
                
        return True, None
        
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼명 정리
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            컬럼명이 정리된 DataFrame
        """
        df = df.copy()
        
        # 공백 제거 및 언더스코어로 변경
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # 특수문자 제거
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # 소문자로 변환
        df.columns = df.columns.str.lower()
        
        logger.info(f"✅ 컬럼명 정리 완료: {df.columns.tolist()}")
        return df
        
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        각 컬럼의 실제 데이터 타입 감지
        
        Args:
            df: 분석할 DataFrame
            
        Returns:
            컬럼별 추천 데이터 타입
        """
        type_recommendations = {}
        
        for col in df.columns:
            series = df[col]
            
            # 숫자형 감지
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().sum() > len(series) * 0.9:
                    if (numeric_series % 1 == 0).all():
                        type_recommendations[col] = 'int'
                    else:
                        type_recommendations[col] = 'float'
                    continue
            except:
                pass
                
            # 날짜형 감지
            try:
                datetime_series = pd.to_datetime(series, errors='coerce')
                if datetime_series.notna().sum() > len(series) * 0.9:
                    type_recommendations[col] = 'datetime'
                    continue
            except:
                pass
                
            # 카테고리형 감지 (unique 값이 전체의 50% 미만)
            if series.nunique() < len(series) * 0.5:
                type_recommendations[col] = 'category'
            else:
                type_recommendations[col] = 'string'
                
        return type_recommendations
        
    @staticmethod
    def create_sample_csv_string(rows: int = 10) -> str:
        """
        샘플 CSV 문자열 생성
        
        Args:
            rows: 생성할 행 수
            
        Returns:
            CSV 형식의 문자열
        """
        df = CommonDataProcessor.generate_sample_data(rows=rows)
        return df.to_csv(index=False)
        
    @staticmethod
    def create_sample_json_string(rows: int = 10) -> str:
        """
        샘플 JSON 문자열 생성
        
        Args:
            rows: 생성할 행 수
            
        Returns:
            JSON 형식의 문자열
        """
        df = CommonDataProcessor.generate_sample_data(rows=rows)
        return df.to_json(orient='records', indent=2)