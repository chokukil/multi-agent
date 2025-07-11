"""
Automatic Data Profiling System

자동 데이터 프로파일링 시스템
- 업로드된 데이터의 자동 분석
- 데이터 품질 평가
- 구조 및 패턴 탐지
- 데이터 타입 추론
- 이상치 및 누락값 탐지
- 통계적 특성 분석
- 액션 가능한 인사이트 제공

Author: CherryAI Team
Date: 2024-12-30
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional imports for enhanced profiling
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Our imports
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    from core.user_file_tracker import get_user_file_tracker
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """데이터 품질 등급"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 80-89%
    FAIR = "fair"               # 60-79%
    POOR = "poor"               # 40-59%
    CRITICAL = "critical"       # 0-39%


class ColumnType(Enum):
    """컬럼 타입 분류"""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DataPattern(Enum):
    """데이터 패턴 유형"""
    TIME_SERIES = "time_series"
    HIERARCHICAL = "hierarchical"
    RELATIONAL = "relational"
    TRANSACTIONAL = "transactional"
    EXPERIMENTAL = "experimental"
    SURVEY = "survey"
    LOG_DATA = "log_data"
    MIXED = "mixed"


@dataclass
class ColumnProfile:
    """컬럼 프로파일 정보"""
    name: str
    dtype: str
    inferred_type: ColumnType
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # 통계 정보
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    
    # 분포 정보
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # 범주형 데이터 정보
    top_values: Optional[List[Tuple[Any, int]]] = None
    
    # 품질 지표
    quality_score: float = 0.0
    quality_issues: List[str] = None
    
    # 추천사항
    recommendations: List[str] = None


@dataclass 
class DataProfile:
    """전체 데이터 프로파일"""
    # 기본 정보
    dataset_name: str
    shape: Tuple[int, int]
    memory_usage: float  # MB
    dtypes_summary: Dict[str, int]
    
    # 품질 정보
    overall_quality: DataQuality
    quality_score: float
    
    # 컬럼 프로파일
    columns: List[ColumnProfile]
    
    # 데이터 패턴
    detected_patterns: List[DataPattern]
    
    # 상관관계 정보
    correlations: Optional[Dict[str, Any]] = None
    
    # 중복 및 누락값 정보
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    total_missing: int = 0
    missing_percentage: float = 0.0
    
    # 이상치 정보
    outliers_detected: Dict[str, int] = None
    
    # 인사이트 및 추천사항
    key_insights: List[str] = None
    data_quality_issues: List[str] = None
    recommendations: List[str] = None
    
    # 메타데이터
    profiling_timestamp: str = None
    profiling_duration: float = 0.0


class AutoDataProfiler:
    """
    자동 데이터 프로파일링 시스템
    
    업로드된 데이터를 종합적으로 분석하여 구조, 품질, 패턴을 파악
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enhanced_tracer = None
        self.user_file_tracker = None
        
        # 프로파일링 설정
        self.max_categorical_values = self.config.get('max_categorical_values', 50)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # Z-score
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # 시스템 초기화
        self._initialize_systems()
        
        logger.info("🔍 Auto Data Profiler 초기화 완료")
    
    def _initialize_systems(self):
        """핵심 시스템 초기화"""
        if not CORE_SYSTEMS_AVAILABLE:
            logger.warning("⚠️ Core systems not available")
            return
        
        try:
            self.enhanced_tracer = get_enhanced_tracer()
            self.user_file_tracker = get_user_file_tracker()
            logger.info("✅ Enhanced tracking systems activated")
        except Exception as e:
            logger.warning(f"⚠️ System initialization failed: {e}")
    
    def profile_data(
        self, 
        data: Union[pd.DataFrame, str, Dict], 
        dataset_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> DataProfile:
        """
        데이터 프로파일링 수행
        
        Args:
            data: 분석할 데이터 (DataFrame, 파일 경로, 또는 dict)
            dataset_name: 데이터셋 이름
            session_id: 세션 ID
            
        Returns:
            DataProfile: 종합 데이터 프로파일
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 데이터 프로파일링 시작: {dataset_name or 'Unknown'}")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "data_profiling_start",
                    {"dataset_name": dataset_name, "session_id": session_id},
                    "Starting automatic data profiling"
                )
            
            # 데이터 로드 및 전처리
            df = self._prepare_data(data)
            if df is None or df.empty:
                raise ValueError("Invalid or empty dataset")
            
            # 기본 정보 수집
            basic_info = self._analyze_basic_info(df, dataset_name)
            
            # 컬럼별 상세 분석
            column_profiles = self._analyze_columns(df)
            
            # 데이터 품질 평가
            quality_info = self._assess_data_quality(df, column_profiles)
            
            # 패턴 탐지
            patterns = self._detect_data_patterns(df, column_profiles)
            
            # 상관관계 분석
            correlations = self._analyze_correlations(df)
            
            # 이상치 탐지
            outliers = self._detect_outliers(df)
            
            # 인사이트 생성
            insights = self._generate_insights(df, column_profiles, quality_info, patterns)
            
            # 추천사항 생성
            recommendations = self._generate_recommendations(df, column_profiles, quality_info)
            
            # 프로파일 생성
            profile = DataProfile(
                dataset_name=dataset_name or f"Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                shape=df.shape,
                memory_usage=df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                dtypes_summary=df.dtypes.value_counts().to_dict(),
                overall_quality=quality_info['overall_quality'],
                quality_score=quality_info['quality_score'],
                columns=column_profiles,
                detected_patterns=patterns,
                correlations=correlations,
                duplicate_rows=df.duplicated().sum(),
                duplicate_percentage=(df.duplicated().sum() / len(df)) * 100,
                total_missing=df.isnull().sum().sum(),
                missing_percentage=(df.isnull().sum().sum() / df.size) * 100,
                outliers_detected=outliers,
                key_insights=insights,
                data_quality_issues=quality_info['issues'],
                recommendations=recommendations,
                profiling_timestamp=datetime.now().isoformat(),
                profiling_duration=(datetime.now() - start_time).total_seconds()
            )
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "data_profiling_complete",
                    {
                        "dataset_name": dataset_name,
                        "quality_score": profile.quality_score,
                        "shape": profile.shape,
                        "duration": profile.profiling_duration
                    },
                    "Data profiling completed successfully"
                )
            
            logger.info(f"✅ 데이터 프로파일링 완료 (소요시간: {profile.profiling_duration:.2f}초)")
            return profile
            
        except Exception as e:
            logger.error(f"❌ 데이터 프로파일링 실패: {e}")
            
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "data_profiling_error",
                    {"error": str(e), "dataset_name": dataset_name},
                    "Data profiling failed"
                )
            
            # 기본 프로파일 반환
            return self._create_error_profile(str(e), dataset_name)
    
    def _prepare_data(self, data: Union[pd.DataFrame, str, Dict]) -> pd.DataFrame:
        """데이터 준비 및 로드"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        elif isinstance(data, str):
            # 파일 경로인 경우
            try:
                if data.endswith('.csv'):
                    return pd.read_csv(data, nrows=10000)  # 최대 10k 행
                elif data.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(data, nrows=10000)
                elif data.endswith('.json'):
                    return pd.read_json(data, lines=True, nrows=10000)
                elif data.endswith('.parquet'):
                    return pd.read_parquet(data).head(10000)
                else:
                    # 기본적으로 CSV로 시도
                    return pd.read_csv(data, nrows=10000)
            except Exception as e:
                logger.warning(f"⚠️ 파일 로드 실패: {e}")
                return None
        
        elif isinstance(data, dict):
            # Dictionary를 DataFrame으로 변환
            try:
                return pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"⚠️ Dict to DataFrame 변환 실패: {e}")
                return None
        
        elif isinstance(data, list):
            # List를 DataFrame으로 변환
            try:
                return pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"⚠️ List to DataFrame 변환 실패: {e}")
                return None
        
        else:
            logger.warning(f"⚠️ 지원하지 않는 데이터 타입: {type(data)}")
            return None
    
    def _analyze_basic_info(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """기본 정보 분석"""
        return {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[ColumnProfile]:
        """컬럼별 상세 분석"""
        profiles = []
        
        for col in df.columns:
            try:
                profile = self._analyze_single_column(df[col], col)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"⚠️ 컬럼 {col} 분석 실패: {e}")
                # 기본 프로파일 생성
                profile = ColumnProfile(
                    name=col,
                    dtype=str(df[col].dtype),
                    inferred_type=ColumnType.UNKNOWN,
                    null_count=df[col].isnull().sum(),
                    null_percentage=(df[col].isnull().sum() / len(df)) * 100,
                    unique_count=df[col].nunique(),
                    unique_percentage=(df[col].nunique() / len(df)) * 100,
                    quality_score=0.5,
                    quality_issues=[f"분석 오류: {str(e)}"],
                    recommendations=["컬럼 데이터를 확인해주세요"]
                )
                profiles.append(profile)
        
        return profiles
    
    def _analyze_single_column(self, series: pd.Series, col_name: str) -> ColumnProfile:
        """단일 컬럼 분석"""
        # 기본 정보
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100
        unique_count = series.nunique()
        unique_percentage = (unique_count / len(series)) * 100
        
        # 타입 추론
        inferred_type = self._infer_column_type(series)
        
        # 통계 정보 (수치형인 경우)
        mean = median = std = skewness = kurtosis = None
        min_value = max_value = None
        
        if inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                mean = numeric_series.mean()
                median = numeric_series.median()
                std = numeric_series.std()
                min_value = numeric_series.min()
                max_value = numeric_series.max()
                
                if SCIPY_AVAILABLE:
                    skewness = stats.skew(numeric_series.dropna())
                    kurtosis = stats.kurtosis(numeric_series.dropna())
            except Exception as e:
                logger.warning(f"⚠️ 수치 통계 계산 실패 {col_name}: {e}")
        
        # Top values (범주형 또는 적은 unique 값인 경우)
        top_values = None
        if unique_count <= self.max_categorical_values:
            try:
                value_counts = series.value_counts().head(10)
                top_values = [(val, count) for val, count in value_counts.items()]
            except Exception as e:
                logger.warning(f"⚠️ Top values 계산 실패 {col_name}: {e}")
        
        # 품질 평가
        quality_score, quality_issues = self._assess_column_quality(series, inferred_type)
        
        # 추천사항
        recommendations = self._generate_column_recommendations(series, inferred_type, quality_issues)
        
        return ColumnProfile(
            name=col_name,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            mean=mean,
            median=median,
            std=std,
            min_value=min_value,
            max_value=max_value,
            skewness=skewness,
            kurtosis=kurtosis,
            top_values=top_values,
            quality_score=quality_score,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
    
    def _infer_column_type(self, series: pd.Series) -> ColumnType:
        """컬럼 타입 추론"""
        # 너무 많은 누락값이 있는 경우
        if series.isnull().sum() / len(series) > 0.9:
            return ColumnType.UNKNOWN
        
        # Boolean 타입 체크
        if series.dtype == 'bool' or set(series.dropna().unique()) <= {True, False, 0, 1, 'True', 'False', 'true', 'false'}:
            return ColumnType.BOOLEAN
        
        # 수치형 타입 체크
        if pd.api.types.is_numeric_dtype(series):
            # 정수형이고 unique 값이 적으면 discrete
            if pd.api.types.is_integer_dtype(series) and series.nunique() <= 20:
                return ColumnType.NUMERIC_DISCRETE
            else:
                return ColumnType.NUMERIC_CONTINUOUS
        
        # 날짜/시간 타입 체크
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        
        # 문자열 타입에서 날짜 추론 시도
        if series.dtype == 'object':
            try:
                # 샘플 데이터로 날짜 파싱 시도
                sample = series.dropna().head(100)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    return ColumnType.DATETIME
            except:
                pass
        
        # 범주형 vs 텍스트 구분
        if series.dtype == 'object' or series.dtype.name == 'category':
            unique_ratio = series.nunique() / len(series)
            
            # Unique 비율이 낮으면 범주형
            if unique_ratio < 0.5 and series.nunique() <= self.max_categorical_values:
                return ColumnType.CATEGORICAL
            else:
                # 텍스트 길이 확인
                try:
                    avg_length = series.astype(str).str.len().mean()
                    if avg_length > 50:  # 긴 텍스트
                        return ColumnType.TEXT
                    else:
                        return ColumnType.CATEGORICAL
                except:
                    return ColumnType.TEXT
        
        return ColumnType.UNKNOWN
    
    def _assess_column_quality(self, series: pd.Series, col_type: ColumnType) -> Tuple[float, List[str]]:
        """컬럼 품질 평가"""
        issues = []
        score = 100.0
        
        # 누락값 검사
        null_percentage = (series.isnull().sum() / len(series)) * 100
        if null_percentage > 50:
            issues.append(f"높은 누락값 비율: {null_percentage:.1f}%")
            score -= 30
        elif null_percentage > 20:
            issues.append(f"상당한 누락값: {null_percentage:.1f}%")
            score -= 15
        elif null_percentage > 5:
            score -= 5
        
        # 중복값 검사 (범주형이 아닌 경우)
        if col_type not in [ColumnType.CATEGORICAL, ColumnType.BOOLEAN]:
            duplicate_percentage = (series.duplicated().sum() / len(series)) * 100
            if duplicate_percentage > 80:
                issues.append(f"매우 높은 중복값 비율: {duplicate_percentage:.1f}%")
                score -= 20
            elif duplicate_percentage > 50:
                issues.append(f"높은 중복값 비율: {duplicate_percentage:.1f}%")
                score -= 10
        
        # 고유값 비율 검사
        unique_ratio = series.nunique() / len(series)
        if col_type == ColumnType.CATEGORICAL and unique_ratio > 0.8:
            issues.append("범주형 데이터에 너무 많은 고유값")
            score -= 15
        
        # 수치형 데이터 특별 검사
        if col_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                
                # 이상치 검사 (Z-score 방법)
                if len(numeric_series.dropna()) > 0:
                    z_scores = np.abs(stats.zscore(numeric_series.dropna())) if SCIPY_AVAILABLE else []
                    if len(z_scores) > 0:
                        outlier_percentage = (np.sum(z_scores > self.outlier_threshold) / len(z_scores)) * 100
                        if outlier_percentage > 10:
                            issues.append(f"높은 이상치 비율: {outlier_percentage:.1f}%")
                            score -= 10
                
                # 영 분산 검사
                if numeric_series.std() == 0:
                    issues.append("모든 값이 동일 (영 분산)")
                    score -= 25
            except Exception as e:
                logger.warning(f"⚠️ 수치형 품질 검사 실패: {e}")
        
        # 최종 점수 조정
        score = max(0, min(100, score))
        
        return score / 100.0, issues
    
    def _generate_column_recommendations(
        self, 
        series: pd.Series, 
        col_type: ColumnType, 
        issues: List[str]
    ) -> List[str]:
        """컬럼별 추천사항 생성"""
        recommendations = []
        
        # 누락값 처리
        null_percentage = (series.isnull().sum() / len(series)) * 100
        if null_percentage > 20:
            if col_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]:
                recommendations.append("수치형 누락값: 평균/중앙값 대체 또는 보간 고려")
            elif col_type == ColumnType.CATEGORICAL:
                recommendations.append("범주형 누락값: 최빈값 대체 또는 'Unknown' 카테고리 생성")
            else:
                recommendations.append("누락값 처리 전략 수립 필요")
        
        # 타입별 추천사항
        if col_type == ColumnType.NUMERIC_CONTINUOUS:
            if series.skew() > 2 if hasattr(series, 'skew') else False:
                recommendations.append("높은 왜도: 로그 변환 또는 Box-Cox 변환 고려")
        
        elif col_type == ColumnType.CATEGORICAL:
            if series.nunique() > 20:
                recommendations.append("범주가 많음: 범주 통합 또는 원-핫 인코딩 고려")
        
        elif col_type == ColumnType.TEXT:
            recommendations.append("텍스트 데이터: NLP 전처리 (토큰화, 정규화) 필요")
        
        elif col_type == ColumnType.DATETIME:
            recommendations.append("날짜 데이터: 시계열 분석 또는 시간 피처 추출 고려")
        
        elif col_type == ColumnType.UNKNOWN:
            recommendations.append("데이터 타입 명확화 및 전처리 필요")
        
        # 이상치 관련
        if "이상치" in " ".join(issues):
            recommendations.append("이상치 처리: 제거, 변환, 또는 별도 분석 고려")
        
        return recommendations
    
    def _assess_data_quality(self, df: pd.DataFrame, column_profiles: List[ColumnProfile]) -> Dict[str, Any]:
        """전체 데이터 품질 평가"""
        # 컬럼별 품질 점수 평균
        column_scores = [col.quality_score for col in column_profiles]
        avg_column_quality = np.mean(column_scores) if column_scores else 0.0
        
        # 전체 데이터 이슈들
        issues = []
        quality_penalty = 0
        
        # 중복 행 검사
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 10:
            issues.append(f"높은 중복 행 비율: {duplicate_percentage:.1f}%")
            quality_penalty += 0.1
        
        # 전체 누락값 비율
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 30:
            issues.append(f"높은 전체 누락값 비율: {missing_percentage:.1f}%")
            quality_penalty += 0.15
        elif missing_percentage > 10:
            issues.append(f"상당한 누락값: {missing_percentage:.1f}%")
            quality_penalty += 0.05
        
        # 데이터 일관성 검사
        type_consistency = self._check_type_consistency(df)
        if not type_consistency:
            issues.append("데이터 타입 일관성 문제 발견")
            quality_penalty += 0.1
        
        # 최종 품질 점수 계산
        final_score = max(0, avg_column_quality - quality_penalty)
        
        # 품질 등급 결정
        if final_score >= 0.9:
            quality_grade = DataQuality.EXCELLENT
        elif final_score >= 0.8:
            quality_grade = DataQuality.GOOD
        elif final_score >= 0.6:
            quality_grade = DataQuality.FAIR
        elif final_score >= 0.4:
            quality_grade = DataQuality.POOR
        else:
            quality_grade = DataQuality.CRITICAL
        
        return {
            'overall_quality': quality_grade,
            'quality_score': final_score,
            'issues': issues,
            'column_scores': column_scores
        }
    
    def _check_type_consistency(self, df: pd.DataFrame) -> bool:
        """데이터 타입 일관성 검사"""
        try:
            # 각 컬럼의 실제 값들이 예상 타입과 일치하는지 확인
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 문자열 컬럼에서 수치값 혼재 확인
                    sample = df[col].dropna().astype(str).head(100)
                    numeric_count = 0
                    for val in sample:
                        try:
                            float(val)
                            numeric_count += 1
                        except ValueError:
                            pass
                    
                    # 절반 이상이 숫자면 타입 불일치
                    if numeric_count / len(sample) > 0.5:
                        return False
            
            return True
        except Exception:
            return True  # 검사 실패 시 일관성 있다고 가정
    
    def _detect_data_patterns(self, df: pd.DataFrame, column_profiles: List[ColumnProfile]) -> List[DataPattern]:
        """데이터 패턴 탐지"""
        patterns = []
        
        # 시계열 패턴 탐지
        datetime_columns = [col.name for col in column_profiles if col.inferred_type == ColumnType.DATETIME]
        if datetime_columns:
            patterns.append(DataPattern.TIME_SERIES)
        
        # 계층적 패턴 탐지 (ID 관련 컬럼들)
        hierarchical_keywords = ['id', 'code', 'key', 'parent', 'child', 'level']
        hierarchical_cols = [col for col in df.columns if any(kw in col.lower() for kw in hierarchical_keywords)]
        if len(hierarchical_cols) >= 2:
            patterns.append(DataPattern.HIERARCHICAL)
        
        # 관계형 패턴 탐지 (외래키 같은 참조 관계)
        if len([col for col in column_profiles if col.inferred_type in [ColumnType.NUMERIC_DISCRETE, ColumnType.CATEGORICAL]]) >= 3:
            patterns.append(DataPattern.RELATIONAL)
        
        # 거래 데이터 패턴 탐지
        transaction_keywords = ['amount', 'price', 'cost', 'fee', 'payment', 'transaction', 'order']
        transaction_cols = [col for col in df.columns if any(kw in col.lower() for kw in transaction_keywords)]
        if transaction_cols and datetime_columns:
            patterns.append(DataPattern.TRANSACTIONAL)
        
        # 실험 데이터 패턴 탐지
        experiment_keywords = ['experiment', 'test', 'trial', 'group', 'treatment', 'control']
        experiment_cols = [col for col in df.columns if any(kw in col.lower() for kw in experiment_keywords)]
        if experiment_cols:
            patterns.append(DataPattern.EXPERIMENTAL)
        
        # 설문조사 패턴 탐지
        survey_keywords = ['score', 'rating', 'satisfaction', 'response', 'answer', 'question']
        survey_cols = [col for col in df.columns if any(kw in col.lower() for kw in survey_keywords)]
        if len(survey_cols) >= 3:
            patterns.append(DataPattern.SURVEY)
        
        # 로그 데이터 패턴 탐지
        log_keywords = ['log', 'event', 'timestamp', 'level', 'message', 'error']
        log_cols = [col for col in df.columns if any(kw in col.lower() for kw in log_keywords)]
        if log_cols and datetime_columns:
            patterns.append(DataPattern.LOG_DATA)
        
        # 패턴이 여러 개면 MIXED
        if len(patterns) > 2:
            patterns = [DataPattern.MIXED]
        elif not patterns:
            patterns = [DataPattern.MIXED]
        
        return patterns
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """상관관계 분석"""
        try:
            # 수치형 컬럼만 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
            
            corr_matrix = df[numeric_cols].corr()
            
            # 높은 상관관계 쌍 찾기
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= self.correlation_threshold:
                        high_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'matrix': corr_matrix.to_dict(),
                'high_correlations': high_correlations,
                'summary': {
                    'total_pairs': len(numeric_cols) * (len(numeric_cols) - 1) // 2,
                    'high_correlation_pairs': len(high_correlations)
                }
            }
        
        except Exception as e:
            logger.warning(f"⚠️ 상관관계 분석 실패: {e}")
            return None
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """이상치 탐지"""
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                if SCIPY_AVAILABLE:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_count = np.sum(z_scores > self.outlier_threshold)
                else:
                    # IQR 방법 사용
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                
                if outlier_count > 0:
                    outliers[col] = outlier_count
            
            except Exception as e:
                logger.warning(f"⚠️ {col} 이상치 탐지 실패: {e}")
        
        return outliers
    
    def _generate_insights(
        self, 
        df: pd.DataFrame, 
        column_profiles: List[ColumnProfile], 
        quality_info: Dict[str, Any], 
        patterns: List[DataPattern]
    ) -> List[str]:
        """주요 인사이트 생성"""
        insights = []
        
        # 데이터 크기 관련
        rows, cols = df.shape
        insights.append(f"데이터셋 크기: {rows:,}행 × {cols}열")
        
        # 품질 관련
        quality_grade = quality_info['overall_quality'].value
        insights.append(f"전체 데이터 품질: {quality_grade.upper()} ({quality_info['quality_score']:.1%})")
        
        # 컬럼 타입 분포
        type_counts = {}
        for profile in column_profiles:
            type_name = profile.inferred_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        top_type = max(type_counts.items(), key=lambda x: x[1])
        insights.append(f"주요 데이터 타입: {top_type[0]} ({top_type[1]}개 컬럼)")
        
        # 누락값 관련
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 5:
            insights.append(f"전체 누락값 비율: {missing_percentage:.1f}%")
        
        # 중복 행 관련
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 1:
            insights.append(f"중복 행 비율: {duplicate_percentage:.1f}%")
        
        # 메모리 사용량
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        insights.append(f"메모리 사용량: {memory_mb:.1f}MB")
        
        # 패턴 관련
        if patterns:
            pattern_names = [p.value for p in patterns]
            insights.append(f"탐지된 데이터 패턴: {', '.join(pattern_names)}")
        
        # 품질 이슈가 있는 컬럼
        poor_quality_cols = [col.name for col in column_profiles if col.quality_score < 0.7]
        if poor_quality_cols:
            insights.append(f"품질 주의 컬럼: {', '.join(poor_quality_cols[:3])}{'...' if len(poor_quality_cols) > 3 else ''}")
        
        return insights
    
    def _generate_recommendations(
        self, 
        df: pd.DataFrame, 
        column_profiles: List[ColumnProfile], 
        quality_info: Dict[str, Any]
    ) -> List[str]:
        """전체 추천사항 생성"""
        recommendations = []
        
        # 품질 기반 추천사항
        if quality_info['overall_quality'] in [DataQuality.POOR, DataQuality.CRITICAL]:
            recommendations.append("🚨 데이터 품질이 낮습니다. 전처리 및 정제 작업을 우선 수행하세요.")
        
        # 누락값 처리
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 20:
            recommendations.append("📋 높은 누락값 비율: 누락값 처리 전략을 수립하세요.")
        
        # 중복 처리
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 5:
            recommendations.append("🔄 중복 행 확인 및 제거를 고려하세요.")
        
        # 메모리 최적화
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 500:  # 500MB 이상
            recommendations.append("💾 메모리 사용량이 높습니다. 데이터 타입 최적화를 고려하세요.")
        
        # 컬럼별 추천사항 집계
        column_issues = {}
        for profile in column_profiles:
            for rec in profile.recommendations or []:
                issue_type = rec.split(':')[0] if ':' in rec else rec
                column_issues[issue_type] = column_issues.get(issue_type, 0) + 1
        
        # 공통 이슈들을 전체 추천사항으로
        if column_issues:
            top_issue = max(column_issues.items(), key=lambda x: x[1])
            if top_issue[1] >= 3:  # 3개 이상 컬럼에서 발생
                recommendations.append(f"⚠️ 공통 이슈 발견: {top_issue[0]} ({top_issue[1]}개 컬럼)")
        
        # 분석 방향 추천
        numeric_cols = len([col for col in column_profiles if col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]])
        categorical_cols = len([col for col in column_profiles if col.inferred_type == ColumnType.CATEGORICAL])
        datetime_cols = len([col for col in column_profiles if col.inferred_type == ColumnType.DATETIME])
        
        if numeric_cols >= 3:
            recommendations.append("📊 수치형 데이터가 풍부합니다. 통계 분석 및 상관관계 분석을 수행하세요.")
        
        if categorical_cols >= 3:
            recommendations.append("🏷️ 범주형 데이터가 많습니다. 교차 분석 및 세그먼트 분석을 고려하세요.")
        
        if datetime_cols >= 1:
            recommendations.append("📅 시간 데이터가 있습니다. 시계열 분석을 고려하세요.")
        
        # 기본 추천사항
        if not recommendations:
            recommendations.append("✅ 데이터 품질이 양호합니다. 분석을 진행하세요.")
        
        return recommendations
    
    def _create_error_profile(self, error_message: str, dataset_name: Optional[str]) -> DataProfile:
        """오류 발생 시 기본 프로파일 생성"""
        return DataProfile(
            dataset_name=dataset_name or "Error Dataset",
            shape=(0, 0),
            memory_usage=0.0,
            dtypes_summary={},
            overall_quality=DataQuality.CRITICAL,
            quality_score=0.0,
            columns=[],
            detected_patterns=[],
            key_insights=[f"프로파일링 오류 발생: {error_message}"],
            data_quality_issues=[error_message],
            recommendations=["데이터 형식을 확인하고 다시 시도해주세요."],
            profiling_timestamp=datetime.now().isoformat(),
            profiling_duration=0.0
        )
    
    def export_profile_report(self, profile: DataProfile, format: str = 'json') -> str:
        """프로파일 보고서 내보내기"""
        try:
            if format.lower() == 'json':
                return json.dumps(asdict(profile), indent=2, default=str)
            
            elif format.lower() == 'html':
                return self._generate_html_report(profile)
            
            elif format.lower() == 'markdown':
                return self._generate_markdown_report(profile)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            logger.error(f"❌ 프로파일 보고서 생성 실패: {e}")
            return f"보고서 생성 오류: {str(e)}"
    
    def _generate_html_report(self, profile: DataProfile) -> str:
        """HTML 보고서 생성"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {profile.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .quality-{profile.overall_quality.value} {{ 
                    background-color: {'#d4edda' if profile.overall_quality == DataQuality.EXCELLENT else 
                                     '#fff3cd' if profile.overall_quality == DataQuality.GOOD else
                                     '#f8d7da' if profile.overall_quality in [DataQuality.POOR, DataQuality.CRITICAL] else '#e2e3e5'};
                    padding: 10px; border-radius: 3px; 
                }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>데이터 프로파일 보고서</h1>
                <h2>{profile.dataset_name}</h2>
                <p>생성일시: {profile.profiling_timestamp}</p>
            </div>
            
            <div class="section quality-{profile.overall_quality.value}">
                <h3>전체 품질 평가</h3>
                <p><strong>품질 등급:</strong> {profile.overall_quality.value.upper()}</p>
                <p><strong>품질 점수:</strong> {profile.quality_score:.1%}</p>
            </div>
            
            <div class="section">
                <h3>기본 정보</h3>
                <ul>
                    <li>크기: {profile.shape[0]:,}행 × {profile.shape[1]}열</li>
                    <li>메모리 사용량: {profile.memory_usage:.1f}MB</li>
                    <li>중복 행: {profile.duplicate_percentage:.1f}%</li>
                    <li>누락값: {profile.missing_percentage:.1f}%</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>주요 인사이트</h3>
                <ul>
                    {''.join(f'<li>{insight}</li>' for insight in profile.key_insights or [])}
                </ul>
            </div>
            
            <div class="section">
                <h3>추천사항</h3>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in profile.recommendations or [])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_markdown_report(self, profile: DataProfile) -> str:
        """Markdown 보고서 생성"""
        md = f"""# 데이터 프로파일 보고서: {profile.dataset_name}

**생성일시:** {profile.profiling_timestamp}  
**프로파일링 소요시간:** {profile.profiling_duration:.2f}초

## 전체 품질 평가
- **품질 등급:** {profile.overall_quality.value.upper()}
- **품질 점수:** {profile.quality_score:.1%}

## 기본 정보
- **데이터 크기:** {profile.shape[0]:,}행 × {profile.shape[1]}열
- **메모리 사용량:** {profile.memory_usage:.1f}MB
- **중복 행 비율:** {profile.duplicate_percentage:.1f}%
- **전체 누락값 비율:** {profile.missing_percentage:.1f}%

## 주요 인사이트
{chr(10).join(f'- {insight}' for insight in profile.key_insights or [])}

## 데이터 품질 이슈
{chr(10).join(f'- {issue}' for issue in profile.data_quality_issues or [])}

## 추천사항
{chr(10).join(f'- {rec}' for rec in profile.recommendations or [])}

## 컬럼 정보
| 컬럼명 | 타입 | 누락값(%) | 고유값(%) | 품질점수 |
|--------|------|-----------|-----------|----------|
{chr(10).join(f'| {col.name} | {col.inferred_type.value} | {col.null_percentage:.1f}% | {col.unique_percentage:.1f}% | {col.quality_score:.1%} |' for col in profile.columns)}
"""
        return md


# 전역 인스턴스
_profiler_instance = None


def get_auto_data_profiler(config: Optional[Dict] = None) -> AutoDataProfiler:
    """Auto Data Profiler 인스턴스 반환"""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = AutoDataProfiler(config)
    return _profiler_instance


# 편의 함수들
def profile_dataset(
    data: Union[pd.DataFrame, str, Dict], 
    dataset_name: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[Dict] = None
) -> DataProfile:
    """데이터셋 프로파일링 편의 함수"""
    profiler = get_auto_data_profiler(config)
    return profiler.profile_data(data, dataset_name, session_id)


def quick_profile(data: Union[pd.DataFrame, str, Dict]) -> Dict[str, Any]:
    """빠른 프로파일링 (주요 정보만)"""
    try:
        profile = profile_dataset(data, "Quick Profile")
        return {
            'shape': profile.shape,
            'quality': profile.overall_quality.value,
            'quality_score': profile.quality_score,
            'missing_percentage': profile.missing_percentage,
            'insights': profile.key_insights[:3],  # 상위 3개만
            'recommendations': profile.recommendations[:3]  # 상위 3개만
        }
    except Exception as e:
        return {'error': str(e)}


# CLI 테스트 함수
def test_auto_data_profiler():
    """Auto Data Profiler 테스트"""
    print("🔍 Auto Data Profiler 테스트 시작\n")
    
    # 샘플 데이터 생성
    np.random.seed(42)
    
    # 다양한 타입의 데이터
    sample_data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 1000),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'is_active': np.random.choice([True, False], 1000, p=[0.8, 0.2]),
        'performance_score': np.random.uniform(1, 5, 1000),
        'comments': [f'This is comment {i}' if i % 10 != 0 else None for i in range(1000)]
    })
    
    # 의도적으로 품질 이슈 추가
    sample_data.loc[50:100, 'salary'] = None  # 누락값
    sample_data.loc[200:250, :] = sample_data.loc[200:250, :].copy()  # 중복값
    
    profiler = get_auto_data_profiler()
    
    print("📊 샘플 데이터 프로파일링...")
    profile = profiler.profile_data(sample_data, "Sample Employee Data", "test_session")
    
    print(f"\n✅ 프로파일링 완료!")
    print(f"📈 데이터 크기: {profile.shape[0]:,}행 × {profile.shape[1]}열")
    print(f"🏆 품질 등급: {profile.overall_quality.value.upper()} ({profile.quality_score:.1%})")
    print(f"💾 메모리 사용량: {profile.memory_usage:.1f}MB")
    print(f"⏱️ 프로파일링 시간: {profile.profiling_duration:.2f}초")
    
    print(f"\n🔍 주요 인사이트:")
    for insight in profile.key_insights[:5]:
        print(f"  • {insight}")
    
    print(f"\n💡 추천사항:")
    for rec in profile.recommendations[:3]:
        print(f"  • {rec}")
    
    print(f"\n📋 컬럼 품질 요약:")
    for col in profile.columns[:5]:  # 상위 5개 컬럼만
        print(f"  • {col.name}: {col.inferred_type.value} (품질: {col.quality_score:.1%})")
    
    # JSON 내보내기 테스트
    print(f"\n📄 JSON 보고서 생성 테스트...")
    json_report = profiler.export_profile_report(profile, 'json')
    print(f"JSON 보고서 크기: {len(json_report):,} 문자")
    
    # Quick profile 테스트
    print(f"\n⚡ 빠른 프로파일링 테스트...")
    quick_result = quick_profile(sample_data.head(100))
    print(f"빠른 프로파일링 결과: {quick_result}")
    
    print(f"\n✅ Auto Data Profiler 테스트 완료!")


if __name__ == "__main__":
    test_auto_data_profiler() 