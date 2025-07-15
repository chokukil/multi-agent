"""
스마트 데이터프레임 (Smart DataFrame)

pandas_agent의 SmartDataFrame 패턴을 기준으로 한 지능형 DataFrame 래퍼
자동 프로파일링, 품질 검증, 메타데이터 관리 등의 지능형 기능 제공

핵심 원칙:
- pandas DataFrame을 확장하되 기존 기능 100% 보존
- LLM First: 메타데이터 생성과 인사이트를 LLM이 담당
- 성능 최적화: 캐싱과 지연 평가 적극 활용
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from functools import cached_property
import hashlib

from .unified_data_interface import DataProfile, QualityReport

logger = logging.getLogger(__name__)


class SmartDataFrame:
    """
    지능형 DataFrame 클래스
    
    pandas_agent의 SmartDataFrame 패턴을 기준으로 구현된
    자동 프로파일링, 품질 검증, 메타데이터 관리 기능을 제공하는 지능형 DataFrame
    """
    
    def __init__(self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """
        SmartDataFrame 초기화
        
        Args:
            df: 원본 pandas DataFrame
            metadata: 추가 메타데이터 정보
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        self._df = df.copy()  # 원본 보호를 위한 복사
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        
        # 캐시된 정보들
        self._profile: Optional[DataProfile] = None
        self._quality_report: Optional[QualityReport] = None
        self._schema_hash: Optional[str] = None
        self._sample_data: Optional[Dict[str, Any]] = None
        
        # 자동 초기화
        self._initialize_metadata()
        
        logger.info(f"✅ SmartDataFrame 초기화: {self.shape} 형태")
    
    def _initialize_metadata(self):
        """메타데이터 자동 초기화"""
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
        
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = self.created_at.isoformat()
        
        if "version" not in self.metadata:
            self.metadata["version"] = "1.0"
    
    @property
    def df(self) -> pd.DataFrame:
        """원본 DataFrame 접근 (읽기 전용)"""
        return self._df
    
    @property
    def shape(self) -> Tuple[int, int]:
        """DataFrame 형태"""
        return self._df.shape
    
    @property
    def columns(self) -> pd.Index:
        """컬럼 리스트"""
        return self._df.columns
    
    @property
    def dtypes(self) -> pd.Series:
        """데이터 타입들"""
        return self._df.dtypes
    
    @property
    def index(self) -> pd.Index:
        """인덱스"""
        return self._df.index
    
    def is_empty(self) -> bool:
        """빈 데이터 여부 확인"""
        return self._df.empty or self._df.shape[0] == 0 or self._df.shape[1] == 0
    
    def has_columns(self) -> bool:
        """컬럼이 있는지 확인"""
        return self._df.shape[1] > 0
    
    def has_data(self) -> bool:
        """데이터가 있는지 확인"""
        return not self.is_empty() and self.has_columns()
    
    @cached_property
    def schema_hash(self) -> str:
        """스키마 해시 (컬럼명 + 데이터타입 기반)"""
        if self._schema_hash is None:
            schema_str = str(list(self.columns)) + str(list(self.dtypes))
            self._schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
        return self._schema_hash
    
    async def auto_profile(self, force_refresh: bool = False) -> DataProfile:
        """
        자동 데이터 프로파일링
        
        Args:
            force_refresh: 캐시된 프로파일 무시하고 새로 생성
            
        Returns:
            DataProfile: 데이터 프로파일 정보
        """
        if self._profile is None or force_refresh:
            try:
                # 기본 프로파일 정보
                missing_values = self._df.isnull().sum().to_dict()
                memory_usage = self._df.memory_usage(deep=True).sum()
                
                # 샘플 데이터 생성
                sample_data = await self._generate_sample_data()
                
                # 컬럼별 통계 생성
                column_stats = await self._generate_column_stats()
                
                # 데이터 품질 점수 계산
                quality_score = await self._calculate_quality_score()
                
                self._profile = DataProfile(
                    shape=self.shape,
                    dtypes={col: str(dtype) for col, dtype in self.dtypes.items()},
                    missing_values=missing_values,
                    memory_usage=int(memory_usage),
                    encoding=self.metadata.get('encoding', 'unknown'),
                    file_size=self.metadata.get('file_size', 0),
                    sample_data=sample_data,
                    column_stats=column_stats,
                    data_quality_score=quality_score
                )
                
                logger.info(f"✅ 데이터 프로파일링 완료: 품질점수 {quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"❌ 데이터 프로파일링 실패: {e}")
                # 기본 프로파일 반환
                self._profile = DataProfile(
                    shape=self.shape,
                    dtypes={col: str(dtype) for col, dtype in self.dtypes.items()},
                    missing_values=self._df.isnull().sum().to_dict(),
                    memory_usage=int(self._df.memory_usage(deep=True).sum()),
                    encoding=self.metadata.get('encoding', 'unknown'),
                    file_size=self.metadata.get('file_size', 0)
                )
        
        return self._profile
    
    async def validate_quality(self, force_refresh: bool = False) -> QualityReport:
        """
        데이터 품질 검증
        
        Args:
            force_refresh: 캐시된 리포트 무시하고 새로 생성
            
        Returns:
            QualityReport: 데이터 품질 리포트
        """
        if self._quality_report is None or force_refresh:
            try:
                # 기본 품질 메트릭 계산
                completeness = await self._calculate_completeness()
                consistency = await self._calculate_consistency()
                validity = await self._calculate_validity()
                accuracy = await self._calculate_accuracy()
                uniqueness = await self._calculate_uniqueness()
                
                # 전체 점수 계산
                overall_score = (completeness + consistency + validity + accuracy + uniqueness) / 5
                
                # 이슈 및 권장사항 생성
                issues, recommendations = await self._analyze_quality_issues()
                
                # 통과/실패 체크 생성
                passed_checks, failed_checks = await self._run_quality_checks()
                
                self._quality_report = QualityReport(
                    overall_score=overall_score,
                    completeness=completeness,
                    consistency=consistency,
                    validity=validity,
                    accuracy=accuracy,
                    uniqueness=uniqueness,
                    issues=issues,
                    recommendations=recommendations,
                    passed_checks=passed_checks,
                    failed_checks=failed_checks
                )
                
                logger.info(f"✅ 품질 검증 완료: 전체 점수 {overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"❌ 품질 검증 실패: {e}")
                # 기본 품질 리포트 반환
                self._quality_report = QualityReport(
                    overall_score=0.5,
                    completeness=0.5,
                    consistency=0.5,
                    validity=0.5,
                    accuracy=0.5,
                    uniqueness=0.5,
                    issues=[f"품질 검증 중 오류 발생: {str(e)}"],
                    recommendations=["데이터를 다시 확인해보세요."],
                    passed_checks=[],
                    failed_checks=["quality_validation"]
                )
        
        return self._quality_report
    
    async def _generate_sample_data(self, sample_size: int = 5) -> Dict[str, Any]:
        """샘플 데이터 생성"""
        try:
            if self.is_empty():
                return {"error": "No data available"}
            
            # 처음 몇 행과 마지막 몇 행
            sample_rows = min(sample_size, len(self._df))
            
            sample = {
                "head": self._df.head(sample_rows).to_dict('records'),
                "tail": self._df.tail(sample_rows).to_dict('records'),
                "random_sample": self._df.sample(min(sample_rows, len(self._df))).to_dict('records')
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"❌ 샘플 데이터 생성 실패: {e}")
            return {"error": str(e)}
    
    async def _generate_column_stats(self) -> Dict[str, Any]:
        """컬럼별 통계 생성"""
        try:
            stats = {}
            
            for column in self.columns:
                col_stats = {
                    "dtype": str(self._df[column].dtype),
                    "null_count": int(self._df[column].isnull().sum()),
                    "null_percentage": float(self._df[column].isnull().sum() / len(self._df) * 100),
                    "unique_count": int(self._df[column].nunique()),
                    "unique_percentage": float(self._df[column].nunique() / len(self._df) * 100)
                }
                
                # 숫자형 컬럼 추가 통계
                if pd.api.types.is_numeric_dtype(self._df[column]):
                    col_stats.update({
                        "mean": float(self._df[column].mean()) if not self._df[column].isna().all() else None,
                        "std": float(self._df[column].std()) if not self._df[column].isna().all() else None,
                        "min": float(self._df[column].min()) if not self._df[column].isna().all() else None,
                        "max": float(self._df[column].max()) if not self._df[column].isna().all() else None,
                        "median": float(self._df[column].median()) if not self._df[column].isna().all() else None
                    })
                
                # 텍스트형 컬럼 추가 통계
                elif pd.api.types.is_string_dtype(self._df[column]) or self._df[column].dtype == 'object':
                    non_null_series = self._df[column].dropna()
                    if len(non_null_series) > 0:
                        col_stats.update({
                            "avg_length": float(non_null_series.astype(str).str.len().mean()),
                            "max_length": int(non_null_series.astype(str).str.len().max()),
                            "min_length": int(non_null_series.astype(str).str.len().min())
                        })
                
                stats[column] = col_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 컬럼 통계 생성 실패: {e}")
            return {}
    
    async def _calculate_quality_score(self) -> float:
        """전체 데이터 품질 점수 계산"""
        try:
            if self.is_empty():
                return 0.0
            
            # 기본 품질 지표들
            total_cells = self._df.shape[0] * self._df.shape[1]
            missing_cells = self._df.isnull().sum().sum()
            completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
            
            # 데이터 타입 일관성
            consistency = 1.0  # 기본적으로 pandas가 타입 관리
            
            # 기본 품질 점수
            quality_score = (completeness + consistency) / 2
            
            return min(max(quality_score, 0.0), 1.0)  # 0-1 범위 보장
            
        except Exception as e:
            logger.error(f"❌ 품질 점수 계산 실패: {e}")
            return 0.5
    
    async def _calculate_completeness(self) -> float:
        """완전성 계산"""
        try:
            if self.is_empty():
                return 0.0
            
            total_cells = self._df.shape[0] * self._df.shape[1]
            missing_cells = self._df.isnull().sum().sum()
            return (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0
            
        except Exception:
            return 0.5
    
    async def _calculate_consistency(self) -> float:
        """일관성 계산"""
        try:
            # 데이터 타입 일관성 확인
            type_consistency = 1.0  # pandas가 자동으로 타입 관리
            
            # 중복 데이터 비율 확인
            if not self.is_empty():
                duplicate_ratio = self._df.duplicated().sum() / len(self._df)
                duplicate_consistency = 1.0 - duplicate_ratio
            else:
                duplicate_consistency = 1.0
            
            return (type_consistency + duplicate_consistency) / 2
            
        except Exception:
            return 0.5
    
    async def _calculate_validity(self) -> float:
        """유효성 계산"""
        try:
            # 기본적으로 pandas가 유효한 데이터만 로딩했다고 가정
            if self.is_empty():
                return 0.0
            
            # 극단값 비율 확인 (숫자형 컬럼만)
            validity_scores = []
            
            for column in self.columns:
                if pd.api.types.is_numeric_dtype(self._df[column]):
                    col_data = self._df[column].dropna()
                    if len(col_data) > 0:
                        # IQR 방법으로 이상값 감지
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                        validity = 1.0 - (outliers / len(col_data))
                        validity_scores.append(validity)
            
            return np.mean(validity_scores) if validity_scores else 0.8
            
        except Exception:
            return 0.5
    
    async def _calculate_accuracy(self) -> float:
        """정확성 계산 (기본 휴리스틱)"""
        try:
            # 데이터 로딩이 성공했다면 기본적으로 정확하다고 가정
            if self.is_empty():
                return 0.0
            
            # 기본 정확성 점수
            base_accuracy = 0.8
            
            # 인코딩 문제가 있으면 감점
            encoding_penalty = 0.0
            for column in self.columns:
                if self._df[column].dtype == 'object':
                    sample_text = self._df[column].dropna().astype(str).head(100)
                    if any('�' in text for text in sample_text):
                        encoding_penalty += 0.1
            
            accuracy = max(base_accuracy - encoding_penalty, 0.0)
            return min(accuracy, 1.0)
            
        except Exception:
            return 0.5
    
    async def _calculate_uniqueness(self) -> float:
        """고유성 계산"""
        try:
            if self.is_empty():
                return 0.0
            
            # 전체 행 대비 고유 행 비율
            total_rows = len(self._df)
            unique_rows = len(self._df.drop_duplicates())
            
            return unique_rows / total_rows if total_rows > 0 else 0.0
            
        except Exception:
            return 0.5
    
    async def _analyze_quality_issues(self) -> Tuple[List[str], List[str]]:
        """품질 이슈 분석 및 권장사항 생성"""
        issues = []
        recommendations = []
        
        try:
            # 빈 데이터 체크
            if self.is_empty():
                issues.append("데이터가 비어있습니다")
                recommendations.append("유효한 데이터 파일을 업로드해주세요")
                return issues, recommendations
            
            # 결측값 체크
            missing_ratio = self._df.isnull().sum().sum() / (self._df.shape[0] * self._df.shape[1])
            if missing_ratio > 0.1:
                issues.append(f"결측값이 {missing_ratio:.1%}로 높습니다")
                recommendations.append("결측값 처리를 고려해보세요 (제거 또는 보간)")
            
            # 중복 데이터 체크
            duplicate_ratio = self._df.duplicated().sum() / len(self._df)
            if duplicate_ratio > 0.05:
                issues.append(f"중복 데이터가 {duplicate_ratio:.1%} 존재합니다")
                recommendations.append("중복 데이터 제거를 고려해보세요")
            
            # 메모리 사용량 체크
            memory_mb = self._df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_mb > 500:
                issues.append(f"메모리 사용량이 {memory_mb:.1f}MB로 높습니다")
                recommendations.append("큰 데이터셋의 경우 샘플링 또는 청크 단위 처리를 고려해보세요")
            
            # 기본 권장사항
            if not issues:
                recommendations.append("데이터 품질이 양호합니다")
            
        except Exception as e:
            issues.append(f"품질 분석 중 오류: {str(e)}")
            recommendations.append("데이터 구조를 다시 확인해보세요")
        
        return issues, recommendations
    
    async def _run_quality_checks(self) -> Tuple[List[str], List[str]]:
        """품질 체크 실행"""
        passed_checks = []
        failed_checks = []
        
        try:
            # 기본 구조 체크
            if not self.is_empty():
                passed_checks.append("non_empty_data")
            else:
                failed_checks.append("non_empty_data")
            
            # 컬럼 존재 체크
            if self.has_columns():
                passed_checks.append("has_columns")
            else:
                failed_checks.append("has_columns")
            
            # 데이터 타입 체크
            if not self._df.dtypes.isnull().any():
                passed_checks.append("valid_data_types")
            else:
                failed_checks.append("valid_data_types")
            
            # 기본 읽기 가능성 체크
            try:
                _ = str(self._df.head(1))
                passed_checks.append("readable_data")
            except:
                failed_checks.append("readable_data")
            
        except Exception as e:
            failed_checks.append(f"quality_checks_error: {str(e)}")
        
        return passed_checks, failed_checks
    
    # pandas DataFrame 인터페이스 위임
    def head(self, n: int = 5) -> pd.DataFrame:
        """처음 n행 반환"""
        return self._df.head(n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """마지막 n행 반환"""
        return self._df.tail(n)
    
    def info(self) -> None:
        """DataFrame 정보 출력"""
        return self._df.info()
    
    def describe(self) -> pd.DataFrame:
        """기술통계 요약"""
        return self._df.describe()
    
    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, **kwargs) -> pd.DataFrame:
        """랜덤 샘플링"""
        return self._df.sample(n=n, frac=frac, **kwargs)
    
    def __getitem__(self, key):
        """컬럼 또는 인덱스 접근"""
        return self._df[key]
    
    def __len__(self) -> int:
        """행 개수"""
        return len(self._df)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"SmartDataFrame(shape={self.shape}, quality_score={getattr(self._profile, 'data_quality_score', 'N/A')})"
    
    def __repr__(self) -> str:
        """객체 표현"""
        return self.__str__()
    
    def to_dict(self, orient: str = 'dict') -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return self._df.to_dict(orient=orient)
    
    def to_json(self, **kwargs) -> str:
        """JSON으로 변환"""
        return self._df.to_json(**kwargs)
    
    def to_csv(self, path_or_buf=None, **kwargs) -> Optional[str]:
        """CSV로 변환/저장"""
        return self._df.to_csv(path_or_buf=path_or_buf, **kwargs)
    
    def copy(self) -> 'SmartDataFrame':
        """SmartDataFrame 복사"""
        return SmartDataFrame(self._df.copy(), self.metadata.copy())
    
    async def get_summary(self) -> Dict[str, Any]:
        """전체 요약 정보 반환"""
        try:
            profile = await self.auto_profile()
            quality = await self.validate_quality()
            
            return {
                "basic_info": {
                    "shape": self.shape,
                    "columns": list(self.columns),
                    "dtypes": {col: str(dtype) for col, dtype in self.dtypes.items()},
                    "memory_usage_mb": profile.memory_usage / (1024 * 1024),
                    "is_empty": self.is_empty()
                },
                "data_profile": profile.__dict__,
                "quality_report": quality.__dict__,
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 요약 정보 생성 실패: {e}")
            return {
                "basic_info": {
                    "shape": self.shape,
                    "error": str(e)
                }
            } 