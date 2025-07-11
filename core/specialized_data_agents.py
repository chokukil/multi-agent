"""
Specialized Data Analysis Agents

데이터 타입별 전문화 에이전트 시스템
정형/시계열/텍스트/이미지 데이터에 특화된 분석 기능 제공

Author: CherryAI Team
Date: 2024-12-30
"""

import logging
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from abc import ABC, abstractmethod

# Enhanced Tracking System
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """데이터 타입 분류"""
    STRUCTURED = "structured"       # 정형 데이터 (CSV, 테이블)
    TIME_SERIES = "time_series"     # 시계열 데이터
    TEXT = "text"                   # 텍스트 데이터
    IMAGE = "image"                 # 이미지 데이터
    MIXED = "mixed"                 # 혼합 데이터
    UNKNOWN = "unknown"             # 알 수 없는 타입


@dataclass
class DataAnalysisResult:
    """데이터 분석 결과"""
    analysis_type: str
    data_type: DataType
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any] = None
    visualizations: List[Dict] = None


@dataclass 
class DataTypeDetectionResult:
    """데이터 타입 탐지 결과"""
    detected_type: DataType
    confidence: float
    reasoning: str
    characteristics: Dict[str, Any]
    recommendations: List[str]


class BaseSpecializedAgent(ABC):
    """전문화 에이전트 기본 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enhanced_tracer = None
        
        # Enhanced Tracking 초기화
        if ENHANCED_TRACKING_AVAILABLE:
            try:
                self.enhanced_tracer = get_enhanced_tracer()
                logger.info(f"✅ {self.__class__.__name__} Enhanced Tracking 활성화")
            except Exception as e:
                logger.warning(f"⚠️ {self.__class__.__name__} Enhanced Tracking 초기화 실패: {e}")
    
    @abstractmethod
    async def analyze(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """데이터 분석 수행"""
        pass
    
    @abstractmethod
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """데이터 타입 탐지"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트 능력 반환"""
        pass


class StructuredDataAgent(BaseSpecializedAgent):
    """정형 데이터 전문 에이전트"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_type = DataType.STRUCTURED
    
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """정형 데이터 탐지"""
        confidence = 0.0
        characteristics = {}
        reasoning = ""
        
        if isinstance(data, pd.DataFrame):
            confidence = 0.9
            characteristics = {
                "rows": len(data),
                "columns": len(data.columns),
                "column_types": dict(data.dtypes.astype(str)),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "missing_values": data.isnull().sum().sum()
            }
            reasoning = f"pandas DataFrame with {len(data)} rows and {len(data.columns)} columns"
            
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            # 리스트/튜플 형태의 정형 데이터 체크
            first_item = data[0]
            if isinstance(first_item, (dict, list, tuple)):
                confidence = 0.7
                characteristics = {
                    "records": len(data),
                    "structure": type(first_item).__name__,
                    "sample_keys": list(first_item.keys()) if isinstance(first_item, dict) else "list_structure"
                }
                reasoning = f"Structured data as {type(first_item).__name__} with {len(data)} records"
        
        recommendations = []
        if confidence > 0.5:
            recommendations.extend([
                "탐색적 데이터 분석(EDA) 수행",
                "기술통계 분석",
                "상관관계 분석",
                "데이터 품질 검사"
            ])
        
        return DataTypeDetectionResult(
            detected_type=DataType.STRUCTURED if confidence > 0.5 else DataType.UNKNOWN,
            confidence=confidence,
            reasoning=reasoning,
            characteristics=characteristics,
            recommendations=recommendations
        )
    
    async def analyze(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """정형 데이터 분석"""
        try:
            logger.info(f"🔄 정형 데이터 분석 시작: {query[:50]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "structured_data_analysis",
                    {"query": query, "data_shape": getattr(data, 'shape', 'unknown')},
                    "Analyzing structured data"
                )
            
            # DataFrame으로 변환
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, (list, dict)):
                    data = pd.DataFrame(data)
                else:
                    raise ValueError("지원하지 않는 데이터 형식입니다")
            
            results = {}
            insights = []
            recommendations = []
            visualizations = []
            
            # 기본 데이터 정보
            results["basic_info"] = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": dict(data.dtypes.astype(str)),
                "missing_values": data.isnull().sum().to_dict(),
                "memory_usage": data.memory_usage(deep=True).sum()
            }
            
            # 기본 insights 항상 생성
            insights.append(f"데이터셋에 {data.shape[0]}개 행, {data.shape[1]}개 열이 있습니다")
            
            # 질의 분석 및 적절한 분석 수행
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ['요약', '기술통계', 'summary', 'describe']):
                results["descriptive_stats"] = data.describe().to_dict()
                insights.append("기술통계 분석을 수행했습니다")
                
            if any(keyword in query_lower for keyword in ['상관관계', 'correlation', '관계']):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = data[numeric_cols].corr()
                    results["correlation_matrix"] = corr_matrix.to_dict()
                    
                    # 강한 상관관계 찾기
                    strong_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                strong_corr.append({
                                    "variables": [corr_matrix.columns[i], corr_matrix.columns[j]],
                                    "correlation": corr_val
                                })
                    
                    if strong_corr:
                        insights.append(f"강한 상관관계({len(strong_corr)}개)가 발견되었습니다")
                        results["strong_correlations"] = strong_corr
                
            if any(keyword in query_lower for keyword in ['분포', 'distribution', '히스토그램']):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    distribution_stats = {}
                    for col in numeric_cols:
                        dist_data = data[col].dropna()
                        distribution_stats[col] = {
                            "mean": float(dist_data.mean()),
                            "std": float(dist_data.std()),
                            "skewness": float(dist_data.skew()),
                            "kurtosis": float(dist_data.kurtosis()),
                            "quartiles": dist_data.quantile([0.25, 0.5, 0.75]).to_dict()
                        }
                    results["distribution_analysis"] = distribution_stats
                    insights.append(f"{len(numeric_cols)}개 수치형 변수의 분포를 분석했습니다")
            
            if any(keyword in query_lower for keyword in ['이상값', 'outlier', '특이값']):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                outlier_analysis = {}
                
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    outlier_analysis[col] = {
                        "count": len(outliers),
                        "percentage": len(outliers) / len(data) * 100,
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    }
                
                results["outlier_analysis"] = outlier_analysis
                total_outliers = sum(stats["count"] for stats in outlier_analysis.values())
                insights.append(f"총 {total_outliers}개의 이상값이 발견되었습니다")
            
            # 결측값 분석
            missing_analysis = data.isnull().sum()
            if missing_analysis.sum() > 0:
                results["missing_value_analysis"] = missing_analysis.to_dict()
                missing_cols = missing_analysis[missing_analysis > 0]
                insights.append(f"{len(missing_cols)}개 열에 결측값이 있습니다")
                recommendations.append("결측값 처리 전략을 고려하세요")
            
            # 데이터 품질 점수 계산
            quality_score = self._calculate_data_quality_score(data)
            results["data_quality_score"] = quality_score
            
            if quality_score < 0.7:
                recommendations.append("데이터 품질 개선이 필요합니다")
            
            # 일반적인 추천사항
            if len(data) > 10000:
                recommendations.append("대용량 데이터이므로 샘플링을 고려하세요")
            
            if len(data.select_dtypes(include=['object']).columns) > 0:
                recommendations.append("범주형 변수에 대한 인코딩을 고려하세요")
                
            return DataAnalysisResult(
                analysis_type="structured_data_analysis",
                data_type=DataType.STRUCTURED,
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence=0.9,
                metadata={"query": query, "data_shape": data.shape}
            )
            
        except Exception as e:
            logger.error(f"❌ 정형 데이터 분석 실패: {e}")
            return DataAnalysisResult(
                analysis_type="structured_data_analysis",
                data_type=DataType.STRUCTURED,
                results={"error": str(e)},
                insights=[f"분석 중 오류 발생: {str(e)}"],
                recommendations=["데이터 형식을 확인하고 다시 시도하세요"],
                confidence=0.1
            )
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """데이터 품질 점수 계산 (0-1)"""
        scores = []
        
        # 결측값 점수 (결측값이 적을수록 높은 점수)
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        missing_score = max(0, 1 - missing_ratio * 2)  # 50% 이상 결측시 0점
        scores.append(missing_score)
        
        # 중복값 점수
        duplicate_ratio = data.duplicated().sum() / len(data)
        duplicate_score = max(0, 1 - duplicate_ratio * 2)
        scores.append(duplicate_score)
        
        # 데이터 타입 일관성 점수
        type_consistency = 1.0  # 기본적으로 pandas가 타입 추론을 하므로 높은 점수
        scores.append(type_consistency)
        
        return np.mean(scores)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트 능력 반환"""
        return {
            "name": "Structured Data Agent",
            "data_type": "structured",
            "capabilities": [
                "기술통계 분석",
                "상관관계 분석", 
                "분포 분석",
                "이상값 탐지",
                "결측값 분석",
                "데이터 품질 평가"
            ],
            "supported_formats": ["DataFrame", "CSV", "Excel", "JSON"],
            "analysis_types": ["descriptive", "exploratory", "quality_assessment"]
        }


class TimeSeriesDataAgent(BaseSpecializedAgent):
    """시계열 데이터 전문 에이전트"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_type = DataType.TIME_SERIES
    
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """시계열 데이터 탐지"""
        confidence = 0.0
        characteristics = {}
        reasoning = ""
        
        if isinstance(data, pd.DataFrame):
            # 날짜/시간 컬럼 탐지
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            date_like_cols = []
            
            # 문자열 컬럼에서 날짜 패턴 찾기
            for col in data.select_dtypes(include=['object']).columns:
                sample_values = data[col].dropna().head(10)
                date_pattern_count = 0
                for val in sample_values:
                    if self._is_date_like(str(val)):
                        date_pattern_count += 1
                
                if date_pattern_count >= 5:  # 샘플의 절반 이상이 날짜 패턴
                    date_like_cols.append(col)
            
            total_time_cols = len(datetime_cols) + len(date_like_cols)
            
            if total_time_cols > 0:
                confidence = min(0.9, 0.3 + total_time_cols * 0.3)
                characteristics = {
                    "datetime_columns": list(datetime_cols),
                    "date_like_columns": date_like_cols,
                    "total_records": len(data),
                    "potential_time_cols": total_time_cols
                }
                reasoning = f"Found {total_time_cols} time-related columns in DataFrame"
            
            # 시간 순서 데이터 패턴 체크
            if hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                confidence = max(confidence, 0.8)
                characteristics["indexed_by_time"] = True
                reasoning += "; Data is indexed by datetime"
        
        recommendations = []
        if confidence > 0.5:
            recommendations.extend([
                "시계열 분해 분석",
                "추세 및 계절성 분석",
                "자기상관 분석",
                "예측 모델링 고려"
            ])
        
        return DataTypeDetectionResult(
            detected_type=DataType.TIME_SERIES if confidence > 0.5 else DataType.UNKNOWN,
            confidence=confidence,
            reasoning=reasoning,
            characteristics=characteristics,
            recommendations=recommendations
        )
    
    def _is_date_like(self, value: str) -> bool:
        """문자열이 날짜 형식인지 확인"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',           # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',           # YYYY/MM/DD
            r'\d{2}-\d{2}-\d{4}',           # MM-DD-YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', # YYYY-MM-DD HH:MM
        ]
        
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    async def analyze(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """시계열 데이터 분석"""
        try:
            logger.info(f"🔄 시계열 데이터 분석 시작: {query[:50]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "time_series_analysis",
                    {"query": query, "data_type": str(type(data))},
                    "Analyzing time series data"
                )
            
            # DataFrame으로 변환 및 시간 인덱스 설정
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # 시간 컬럼 찾기 및 설정
            time_col = self._find_time_column(data)
            if time_col and not isinstance(data.index, pd.DatetimeIndex):
                data[time_col] = pd.to_datetime(data[time_col])
                data = data.set_index(time_col)
                data = data.sort_index()
            
            results = {}
            insights = []
            recommendations = []
            
            # 기본 시계열 정보
            results["basic_info"] = {
                "start_date": str(data.index.min()) if hasattr(data.index, 'min') else "unknown",
                "end_date": str(data.index.max()) if hasattr(data.index, 'max') else "unknown",
                "frequency": str(data.index.freq) if hasattr(data.index, 'freq') else "irregular",
                "total_periods": len(data),
                "missing_periods": data.isnull().sum().sum()
            }
            
            query_lower = query.lower()
            
            # 추세 분석
            if any(keyword in query_lower for keyword in ['추세', 'trend', '경향']):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                trend_analysis = {}
                
                for col in numeric_cols[:3]:  # 처음 3개 수치형 컬럼만 분석
                    series = data[col].dropna()
                    if len(series) > 10:
                        # 단순 선형 추세
                        x = np.arange(len(series))
                        z = np.polyfit(x, series.values, 1)
                        trend_slope = z[0]
                        
                        trend_analysis[col] = {
                            "slope": float(trend_slope),
                            "direction": "증가" if trend_slope > 0 else "감소" if trend_slope < 0 else "평평",
                            "magnitude": "강함" if abs(trend_slope) > series.std() else "약함"
                        }
                
                results["trend_analysis"] = trend_analysis
                insights.append(f"{len(trend_analysis)}개 변수의 추세를 분석했습니다")
            
            # 계절성 분석 (간단한 버전)
            if any(keyword in query_lower for keyword in ['계절성', 'seasonal', '주기']):
                if isinstance(data.index, pd.DatetimeIndex):
                    seasonal_analysis = {}
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_cols[:2]:  # 처음 2개 컬럼만
                        monthly_avg = data.groupby(data.index.month)[col].mean()
                        seasonal_strength = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
                        
                        seasonal_analysis[col] = {
                            "seasonal_strength": float(seasonal_strength),
                            "peak_month": int(monthly_avg.idxmax()),
                            "low_month": int(monthly_avg.idxmin()),
                            "has_seasonality": seasonal_strength > 0.1
                        }
                    
                    results["seasonal_analysis"] = seasonal_analysis
                    insights.append("월별 계절성 패턴을 분석했습니다")
            
            # 변동성 분석
            if any(keyword in query_lower for keyword in ['변동성', 'volatility', '변화']):
                volatility_analysis = {}
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols[:3]:
                    series = data[col].dropna()
                    if len(series) > 1:
                        # 롤링 표준편차로 변동성 측정
                        rolling_std = series.rolling(window=min(30, len(series)//4)).std()
                        
                        volatility_analysis[col] = {
                            "overall_volatility": float(series.std()),
                            "recent_volatility": float(rolling_std.iloc[-1]) if not rolling_std.empty else 0,
                            "volatility_trend": "증가" if rolling_std.iloc[-1] > rolling_std.mean() else "감소"
                        }
                
                results["volatility_analysis"] = volatility_analysis
                insights.append(f"{len(volatility_analysis)}개 변수의 변동성을 분석했습니다")
            
            # 상관관계 시계열 분석
            if any(keyword in query_lower for keyword in ['상관관계', 'correlation', '관계']):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    # 시차 상관관계 분석
                    lag_correlation = {}
                    base_col = numeric_cols[0]
                    
                    for col in numeric_cols[1:3]:  # 최대 2개 다른 변수와 비교
                        correlations = []
                        for lag in range(0, min(10, len(data)//10)):  # 최대 10개 시차
                            if lag == 0:
                                corr = data[base_col].corr(data[col])
                            else:
                                corr = data[base_col].corr(data[col].shift(lag))
                            
                            if not np.isnan(corr):
                                correlations.append({"lag": lag, "correlation": float(corr)})
                        
                        lag_correlation[f"{base_col}_vs_{col}"] = correlations
                    
                    results["lag_correlation"] = lag_correlation
                    insights.append("시차 상관관계를 분석했습니다")
            
            # 일반적인 추천사항
            if len(data) > 1000:
                recommendations.append("장기 시계열이므로 시계열 분해를 고려하세요")
            
            if isinstance(data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(data.index)
                if freq is None:
                    recommendations.append("불규칙한 시간 간격이므로 리샘플링을 고려하세요")
            
            recommendations.extend([
                "시계열 예측 모델 적용 고려",
                "이상 패턴 탐지 수행",
                "시계열 분해를 통한 성분 분석"
            ])
            
            return DataAnalysisResult(
                analysis_type="time_series_analysis",
                data_type=DataType.TIME_SERIES,
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence=0.8,
                metadata={"query": query, "data_periods": len(data)}
            )
            
        except Exception as e:
            logger.error(f"❌ 시계열 데이터 분석 실패: {e}")
            return DataAnalysisResult(
                analysis_type="time_series_analysis",
                data_type=DataType.TIME_SERIES,
                results={"error": str(e)},
                insights=[f"분석 중 오류 발생: {str(e)}"],
                recommendations=["시간 컬럼 형식을 확인하고 다시 시도하세요"],
                confidence=0.1
            )
    
    def _find_time_column(self, data: pd.DataFrame) -> Optional[str]:
        """시간 컬럼 찾기"""
        # datetime 타입 컬럼 우선
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return datetime_cols[0]
        
        # 이름으로 추정
        time_like_names = ['time', 'date', 'datetime', 'timestamp', '시간', '날짜']
        for col in data.columns:
            if any(name in col.lower() for name in time_like_names):
                return col
        
        return None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트 능력 반환"""
        return {
            "name": "Time Series Data Agent",
            "data_type": "time_series",
            "capabilities": [
                "추세 분석",
                "계절성 분석",
                "변동성 분석",
                "시차 상관관계 분석",
                "시계열 분해",
                "이상 패턴 탐지"
            ],
            "supported_formats": ["DateTime indexed DataFrame", "CSV with time columns"],
            "analysis_types": ["trend", "seasonal", "volatility", "forecasting_prep"]
        }


class TextDataAgent(BaseSpecializedAgent):
    """텍스트 데이터 전문 에이전트"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_type = DataType.TEXT
    
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """텍스트 데이터 탐지"""
        confidence = 0.0
        characteristics = {}
        reasoning = ""
        
        if isinstance(data, pd.DataFrame):
            text_cols = data.select_dtypes(include=['object']).columns
            text_analysis = {}
            
            for col in text_cols:
                sample_values = data[col].dropna().head(20)
                avg_length = np.mean([len(str(val)) for val in sample_values])
                
                # 텍스트 특성 분석
                if avg_length > 50:  # 평균 50자 이상이면 텍스트로 간주
                    text_analysis[col] = {
                        "avg_length": avg_length,
                        "max_length": max([len(str(val)) for val in sample_values]),
                        "contains_sentences": any('.' in str(val) for val in sample_values)
                    }
            
            if text_analysis:
                confidence = min(0.9, 0.4 + len(text_analysis) * 0.2)
                characteristics = {
                    "text_columns": list(text_analysis.keys()),
                    "text_analysis": text_analysis,
                    "total_records": len(data)
                }
                reasoning = f"Found {len(text_analysis)} text columns with substantial content"
        
        elif isinstance(data, (list, str)):
            if isinstance(data, str):
                # 이미지 파일 확장자가 있으면 텍스트가 아님
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
                if any(ext in data.lower() for ext in image_extensions):
                    confidence = 0.1  # 이미지 파일로 보임
                elif len(data) > 50:  # 충분히 긴 텍스트만 높은 신뢰도
                    confidence = 0.9
                else:
                    confidence = 0.6  # 짧은 텍스트는 낮은 신뢰도
                
                characteristics = {
                    "length": len(data),
                    "word_count": len(data.split()),
                    "contains_sentences": '.' in data
                }
                reasoning = "Single text string detected"
            elif isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, str) and len(first_item) > 20:
                    confidence = 0.8
                    avg_length = np.mean([len(str(item)) for item in data[:10]])
                    characteristics = {
                        "records": len(data),
                        "avg_length": avg_length,
                        "sample_item": str(first_item)[:100]
                    }
                    reasoning = f"List of {len(data)} text items detected"
        
        recommendations = []
        if confidence > 0.5:
            recommendations.extend([
                "텍스트 전처리 수행",
                "감정 분석",
                "키워드 추출",
                "토픽 모델링 고려"
            ])
        
        return DataTypeDetectionResult(
            detected_type=DataType.TEXT if confidence > 0.5 else DataType.UNKNOWN,
            confidence=confidence,
            reasoning=reasoning,
            characteristics=characteristics,
            recommendations=recommendations
        )
    
    async def analyze(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """텍스트 데이터 분석"""
        try:
            logger.info(f"🔄 텍스트 데이터 분석 시작: {query[:50]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "text_data_analysis",
                    {"query": query, "data_type": str(type(data))},
                    "Analyzing text data"
                )
            
            # 텍스트 데이터 준비
            text_data = self._prepare_text_data(data)
            
            results = {}
            insights = []
            recommendations = []
            
            # 기본 텍스트 통계
            results["basic_stats"] = self._compute_basic_text_stats(text_data)
            insights.append(f"총 {len(text_data)}개의 텍스트 항목을 분석했습니다")
            
            query_lower = query.lower()
            
            # 키워드 분석
            if any(keyword in query_lower for keyword in ['키워드', 'keyword', '단어', '용어']):
                keyword_analysis = self._analyze_keywords(text_data)
                results["keyword_analysis"] = keyword_analysis
                insights.append(f"상위 {len(keyword_analysis.get('top_words', []))}개 키워드를 추출했습니다")
            
            # 감정 분석 (간단한 버전)
            if any(keyword in query_lower for keyword in ['감정', 'sentiment', '긍정', '부정']):
                sentiment_analysis = self._analyze_sentiment(text_data)
                results["sentiment_analysis"] = sentiment_analysis
                insights.append("감정 분석을 수행했습니다")
            
            # 텍스트 길이 분석
            if any(keyword in query_lower for keyword in ['길이', 'length', '문장', '단어수']):
                length_analysis = self._analyze_text_length(text_data)
                results["length_analysis"] = length_analysis
                insights.append("텍스트 길이 분포를 분석했습니다")
            
            # 언어 특성 분석
            if any(keyword in query_lower for keyword in ['언어', 'language', '특성']):
                language_analysis = self._analyze_language_features(text_data)
                results["language_analysis"] = language_analysis
                insights.append("언어적 특성을 분석했습니다")
            
            # 일반적인 추천사항
            if len(text_data) > 1000:
                recommendations.append("대용량 텍스트이므로 샘플링 또는 배치 처리를 고려하세요")
            
            recommendations.extend([
                "전문적인 NLP 도구 활용 고려",
                "토픽 모델링을 통한 주제 분석",
                "텍스트 전처리 및 정제 수행"
            ])
            
            return DataAnalysisResult(
                analysis_type="text_data_analysis",
                data_type=DataType.TEXT,
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence=0.8,
                metadata={"query": query, "text_items": len(text_data)}
            )
            
        except Exception as e:
            logger.error(f"❌ 텍스트 데이터 분석 실패: {e}")
            return DataAnalysisResult(
                analysis_type="text_data_analysis",
                data_type=DataType.TEXT,
                results={"error": str(e)},
                insights=[f"분석 중 오류 발생: {str(e)}"],
                recommendations=["텍스트 데이터 형식을 확인하고 다시 시도하세요"],
                confidence=0.1
            )
    
    def _prepare_text_data(self, data: Any) -> List[str]:
        """텍스트 데이터 준비"""
        text_data = []
        
        if isinstance(data, str):
            text_data = [data]
        elif isinstance(data, list):
            text_data = [str(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            # 텍스트 컬럼 찾기
            text_cols = data.select_dtypes(include=['object']).columns
            for col in text_cols:
                sample_values = data[col].dropna().head(10)
                avg_length = np.mean([len(str(val)) for val in sample_values])
                if avg_length > 20:  # 긴 텍스트 컬럼 선택
                    text_data.extend(data[col].dropna().astype(str).tolist())
                    break
        
        return text_data
    
    def _compute_basic_text_stats(self, text_data: List[str]) -> Dict[str, Any]:
        """기본 텍스트 통계 계산"""
        lengths = [len(text) for text in text_data]
        word_counts = [len(text.split()) for text in text_data]
        
        return {
            "total_texts": len(text_data),
            "avg_char_length": np.mean(lengths),
            "avg_word_count": np.mean(word_counts),
            "max_char_length": max(lengths) if lengths else 0,
            "min_char_length": min(lengths) if lengths else 0,
            "total_characters": sum(lengths),
            "total_words": sum(word_counts)
        }
    
    def _analyze_keywords(self, text_data: List[str]) -> Dict[str, Any]:
        """키워드 분석"""
        from collections import Counter
        import re
        
        # 간단한 키워드 추출 (단어 빈도 기반)
        all_words = []
        for text in text_data:
            # 단순 단어 분리 (한글/영문)
            words = re.findall(r'\b\w+\b', text.lower())
            # 길이 2 이상인 단어만 선택
            words = [word for word in words if len(word) >= 2]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        return {
            "total_unique_words": len(word_freq),
            "top_words": [{"word": word, "count": count} for word, count in word_freq.most_common(20)],
            "vocabulary_size": len(word_freq)
        }
    
    def _analyze_sentiment(self, text_data: List[str]) -> Dict[str, Any]:
        """간단한 감정 분석"""
        # 기본적인 감정 키워드 기반 분석
        positive_words = ['좋다', '훌륭하다', '완벽하다', '최고', '만족', 'good', 'great', 'excellent', 'perfect', 'amazing']
        negative_words = ['나쁘다', '최악', '싫다', '불만', '화나다', 'bad', 'terrible', 'awful', 'hate', 'angry']
        
        sentiment_scores = []
        for text in text_data:
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment_scores.append(1)  # 긍정
            elif negative_count > positive_count:
                sentiment_scores.append(-1)  # 부정
            else:
                sentiment_scores.append(0)  # 중립
        
        positive_ratio = sum(1 for s in sentiment_scores if s > 0) / len(sentiment_scores)
        negative_ratio = sum(1 for s in sentiment_scores if s < 0) / len(sentiment_scores)
        neutral_ratio = sum(1 for s in sentiment_scores if s == 0) / len(sentiment_scores)
        
        return {
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
            "overall_sentiment": "긍정적" if positive_ratio > negative_ratio else "부정적" if negative_ratio > positive_ratio else "중립적"
        }
    
    def _analyze_text_length(self, text_data: List[str]) -> Dict[str, Any]:
        """텍스트 길이 분석"""
        char_lengths = [len(text) for text in text_data]
        word_lengths = [len(text.split()) for text in text_data]
        
        return {
            "char_length_stats": {
                "mean": np.mean(char_lengths),
                "std": np.std(char_lengths),
                "min": min(char_lengths),
                "max": max(char_lengths),
                "median": np.median(char_lengths)
            },
            "word_length_stats": {
                "mean": np.mean(word_lengths),
                "std": np.std(word_lengths),
                "min": min(word_lengths),
                "max": max(word_lengths),
                "median": np.median(word_lengths)
            }
        }
    
    def _analyze_language_features(self, text_data: List[str]) -> Dict[str, Any]:
        """언어적 특성 분석"""
        # 문장 부호 사용 빈도
        punctuation_count = 0
        question_count = 0
        exclamation_count = 0
        
        for text in text_data:
            punctuation_count += len(re.findall(r'[.,;:!?]', text))
            question_count += text.count('?')
            exclamation_count += text.count('!')
        
        return {
            "avg_punctuation_per_text": punctuation_count / len(text_data),
            "question_ratio": question_count / len(text_data),
            "exclamation_ratio": exclamation_count / len(text_data),
            "avg_sentence_length": np.mean([len(text.split('.')) for text in text_data])
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트 능력 반환"""
        return {
            "name": "Text Data Agent",
            "data_type": "text",
            "capabilities": [
                "키워드 추출",
                "감정 분석",
                "텍스트 길이 분석",
                "언어적 특성 분석",
                "기본 통계 분석"
            ],
            "supported_formats": ["String", "List of strings", "DataFrame with text columns"],
            "analysis_types": ["keyword", "sentiment", "linguistic", "statistical"]
        }


class ImageDataAgent(BaseSpecializedAgent):
    """이미지 데이터 전문 에이전트"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.data_type = DataType.IMAGE
    
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """이미지 데이터 탐지"""
        confidence = 0.0
        characteristics = {}
        reasoning = ""
        
        # 파일 경로 기반 탐지
        if isinstance(data, (str, Path)):
            file_path = Path(data)
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
            
            if file_path.suffix.lower() in image_extensions:
                confidence = 0.9
                characteristics = {
                    "file_path": str(file_path),
                    "file_extension": file_path.suffix.lower(),
                    "file_exists": file_path.exists()
                }
                reasoning = f"Image file detected: {file_path.suffix}"
        
        # DataFrame에서 이미지 경로 컬럼 탐지
        elif isinstance(data, pd.DataFrame):
            image_path_cols = []
            for col in data.columns:
                if 'image' in col.lower() or 'img' in col.lower() or 'photo' in col.lower():
                    sample_values = data[col].dropna().head(10)
                    image_files = sum(1 for val in sample_values 
                                    if isinstance(val, str) and any(ext in val.lower() 
                                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']))
                    
                    if image_files >= 3:  # 더 관대한 기준
                        image_path_cols.append(col)
            
            if image_path_cols:
                confidence = 0.8
                characteristics = {
                    "image_path_columns": image_path_cols,
                    "total_records": len(data)
                }
                reasoning = f"Found {len(image_path_cols)} columns with image file paths"
        
        # 리스트 형태의 이미지 경로들
        elif isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, str):
                image_files = sum(1 for item in data[:10] 
                                if any(ext in item.lower() 
                                for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']))
                
                if image_files >= 3:  # 더 관대한 기준
                    confidence = 0.7
                    characteristics = {
                        "image_files_count": len(data),
                        "sample_files": data[:5]
                    }
                    reasoning = f"List of {len(data)} image file paths detected"
        
        recommendations = []
        if confidence > 0.5:
            recommendations.extend([
                "이미지 메타데이터 분석",
                "이미지 크기 및 형식 분석",
                "색상 분포 분석",
                "컴퓨터 비전 분석 고려"
            ])
        
        return DataTypeDetectionResult(
            detected_type=DataType.IMAGE if confidence > 0.5 else DataType.UNKNOWN,
            confidence=confidence,
            reasoning=reasoning,
            characteristics=characteristics,
            recommendations=recommendations
        )
    
    async def analyze(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """이미지 데이터 분석"""
        try:
            logger.info(f"🔄 이미지 데이터 분석 시작: {query[:50]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "image_data_analysis",
                    {"query": query, "data_type": str(type(data))},
                    "Analyzing image data"
                )
            
            # 이미지 파일 경로 수집
            image_paths = self._collect_image_paths(data)
            
            if not image_paths:
                raise ValueError("이미지 파일을 찾을 수 없습니다")
            
            results = {}
            insights = []
            recommendations = []
            
            # 기본 이미지 정보 분석
            results["basic_info"] = self._analyze_basic_image_info(image_paths)
            insights.append(f"총 {len(image_paths)}개의 이미지를 분석했습니다")
            
            query_lower = query.lower()
            
            # 메타데이터 분석
            if any(keyword in query_lower for keyword in ['메타데이터', 'metadata', '정보', 'exif']):
                metadata_analysis = self._analyze_image_metadata(image_paths)
                results["metadata_analysis"] = metadata_analysis
                insights.append("이미지 메타데이터를 분석했습니다")
            
            # 크기 및 형식 분석
            if any(keyword in query_lower for keyword in ['크기', 'size', '해상도', 'resolution', '형식', 'format']):
                size_format_analysis = self._analyze_size_and_format(image_paths)
                results["size_format_analysis"] = size_format_analysis
                insights.append("이미지 크기와 형식을 분석했습니다")
            
            # 일반적인 추천사항
            if len(image_paths) > 100:
                recommendations.append("대량 이미지이므로 배치 처리를 고려하세요")
            
            recommendations.extend([
                "전문 이미지 분석 도구 활용 고려",
                "컴퓨터 비전 라이브러리 사용",
                "이미지 전처리 및 최적화 수행"
            ])
            
            return DataAnalysisResult(
                analysis_type="image_data_analysis",
                data_type=DataType.IMAGE,
                results=results,
                insights=insights,
                recommendations=recommendations,
                confidence=0.7,
                metadata={"query": query, "image_count": len(image_paths)}
            )
            
        except Exception as e:
            logger.error(f"❌ 이미지 데이터 분석 실패: {e}")
            return DataAnalysisResult(
                analysis_type="image_data_analysis",
                data_type=DataType.IMAGE,
                results={"error": str(e)},
                insights=[f"분석 중 오류 발생: {str(e)}"],
                recommendations=["이미지 파일 경로를 확인하고 다시 시도하세요"],
                confidence=0.1
            )
    
    def _collect_image_paths(self, data: Any) -> List[str]:
        """이미지 파일 경로 수집"""
        image_paths = []
        
        if isinstance(data, (str, Path)):
            if Path(data).exists():
                image_paths = [str(data)]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and Path(item).exists():
                    image_paths.append(item)
        elif isinstance(data, pd.DataFrame):
            for col in data.columns:
                if 'image' in col.lower() or 'img' in col.lower():
                    paths = data[col].dropna().tolist()
                    valid_paths = [path for path in paths if isinstance(path, str) and Path(path).exists()]
                    image_paths.extend(valid_paths)
                    break
        
        return image_paths
    
    def _analyze_basic_image_info(self, image_paths: List[str]) -> Dict[str, Any]:
        """기본 이미지 정보 분석"""
        file_sizes = []
        file_extensions = []
        
        for path in image_paths:
            try:
                file_path = Path(path)
                if file_path.exists():
                    file_sizes.append(file_path.stat().st_size)
                    file_extensions.append(file_path.suffix.lower())
            except Exception:
                continue
        
        from collections import Counter
        extension_counts = Counter(file_extensions)
        
        return {
            "total_images": len(image_paths),
            "valid_files": len(file_sizes),
            "total_size_bytes": sum(file_sizes),
            "avg_size_bytes": np.mean(file_sizes) if file_sizes else 0,
            "extension_distribution": dict(extension_counts),
            "size_range": {
                "min": min(file_sizes) if file_sizes else 0,
                "max": max(file_sizes) if file_sizes else 0
            }
        }
    
    def _analyze_image_metadata(self, image_paths: List[str]) -> Dict[str, Any]:
        """이미지 메타데이터 분석 (기본 정보)"""
        # 기본적인 파일 정보만 제공 (EXIF는 별도 라이브러리 필요)
        metadata_info = {
            "analyzed_images": 0,
            "creation_dates": [],
            "file_formats": [],
            "estimated_dimensions": "분석을 위해 PIL 라이브러리가 필요합니다"
        }
        
        for path in image_paths[:10]:  # 처음 10개만 분석
            try:
                file_path = Path(path)
                if file_path.exists():
                    stat = file_path.stat()
                    metadata_info["creation_dates"].append(stat.st_mtime)
                    metadata_info["file_formats"].append(file_path.suffix.lower())
                    metadata_info["analyzed_images"] += 1
            except Exception:
                continue
        
        return metadata_info
    
    def _analyze_size_and_format(self, image_paths: List[str]) -> Dict[str, Any]:
        """크기 및 형식 분석"""
        format_analysis = {
            "supported_formats": ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            "found_formats": [],
            "format_distribution": {},
            "size_analysis": "이미지 크기 분석을 위해 PIL 라이브러리가 필요합니다"
        }
        
        from collections import Counter
        formats = [Path(path).suffix.lower() for path in image_paths]
        format_counts = Counter(formats)
        
        format_analysis["found_formats"] = list(set(formats))
        format_analysis["format_distribution"] = dict(format_counts)
        
        return format_analysis
    
    def get_capabilities(self) -> Dict[str, Any]:
        """에이전트 능력 반환"""
        return {
            "name": "Image Data Agent",
            "data_type": "image",
            "capabilities": [
                "이미지 메타데이터 분석",
                "파일 크기 및 형식 분석",
                "기본 이미지 정보 추출",
                "이미지 경로 관리"
            ],
            "supported_formats": ["JPG", "JPEG", "PNG", "GIF", "BMP", "TIFF"],
            "analysis_types": ["metadata", "format", "basic_stats"],
            "limitations": ["고급 이미지 분석을 위해서는 OpenCV, PIL 등의 라이브러리가 필요합니다"]
        }


class DataTypeDetector:
    """데이터 타입 자동 탐지 시스템"""
    
    def __init__(self):
        self.agents = {
            DataType.STRUCTURED: StructuredDataAgent(),
            DataType.TIME_SERIES: TimeSeriesDataAgent(),
            DataType.TEXT: TextDataAgent(),
            DataType.IMAGE: ImageDataAgent()
        }
    
    def detect_data_type(self, data: Any) -> DataTypeDetectionResult:
        """데이터 타입 자동 탐지"""
        detection_results = []
        
        for data_type, agent in self.agents.items():
            result = agent.detect_data_type(data)
            detection_results.append((data_type, result))
        
        # 가장 높은 신뢰도의 결과 선택
        best_result = max(detection_results, key=lambda x: x[1].confidence)
        
        if best_result[1].confidence > 0.5:
            return best_result[1]
        else:
            return DataTypeDetectionResult(
                detected_type=DataType.UNKNOWN,
                confidence=0.0,
                reasoning="데이터 타입을 확실하게 탐지할 수 없습니다",
                characteristics={},
                recommendations=["데이터 형식을 확인하고 적절한 전처리를 수행하세요"]
            )
    
    async def analyze_with_best_agent(self, data: Any, query: str, context: Optional[Dict] = None) -> DataAnalysisResult:
        """최적의 에이전트로 데이터 분석"""
        detection_result = self.detect_data_type(data)
        
        if detection_result.detected_type == DataType.UNKNOWN:
            # 기본적으로 정형 데이터 에이전트 사용
            agent = self.agents[DataType.STRUCTURED]
        else:
            agent = self.agents[detection_result.detected_type]
        
        return await agent.analyze(data, query, context)


# 전역 인스턴스
_detector_instance = None


def get_data_type_detector() -> DataTypeDetector:
    """데이터 타입 탐지기 인스턴스 반환"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DataTypeDetector()
    return _detector_instance


# 개별 에이전트 인스턴스 반환 함수들
def get_structured_agent(config: Optional[Dict] = None) -> StructuredDataAgent:
    """정형 데이터 에이전트 반환"""
    return StructuredDataAgent(config)


def get_time_series_agent(config: Optional[Dict] = None) -> TimeSeriesDataAgent:
    """시계열 데이터 에이전트 반환"""
    return TimeSeriesDataAgent(config)


def get_text_agent(config: Optional[Dict] = None) -> TextDataAgent:
    """텍스트 데이터 에이전트 반환"""
    return TextDataAgent(config)


def get_image_agent(config: Optional[Dict] = None) -> ImageDataAgent:
    """이미지 데이터 에이전트 반환"""
    return ImageDataAgent(config)


# CLI 테스트 함수
async def test_specialized_agents():
    """전문화 에이전트 테스트"""
    print("🔄 전문화 에이전트 시스템 테스트 시작\n")
    
    # 테스트 데이터 생성
    structured_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    text_data = [
        "이것은 긍정적인 리뷰입니다. 정말 좋았어요!",
        "별로였습니다. 다시는 사용하지 않을 것 같아요.",
        "보통이었습니다. 그럭저럭 괜찮았어요."
    ]
    
    # 데이터 타입 탐지 테스트
    detector = get_data_type_detector()
    
    print("📊 정형 데이터 탐지:")
    structured_detection = detector.detect_data_type(structured_data)
    print(f"  탐지 결과: {structured_detection.detected_type.value}")
    print(f"  신뢰도: {structured_detection.confidence:.2f}")
    print(f"  이유: {structured_detection.reasoning}\n")
    
    print("📝 텍스트 데이터 탐지:")
    text_detection = detector.detect_data_type(text_data)
    print(f"  탐지 결과: {text_detection.detected_type.value}")
    print(f"  신뢰도: {text_detection.confidence:.2f}")
    print(f"  이유: {text_detection.reasoning}\n")
    
    # 분석 테스트
    print("🔍 정형 데이터 분석:")
    structured_result = await detector.analyze_with_best_agent(
        structured_data, 
        "데이터 요약과 상관관계를 분석해주세요"
    )
    print(f"  분석 결과: {len(structured_result.insights)}개 인사이트")
    for insight in structured_result.insights:
        print(f"    - {insight}")
    print()
    
    print("📝 텍스트 데이터 분석:")
    text_result = await detector.analyze_with_best_agent(
        text_data,
        "감정 분석과 키워드를 추출해주세요"
    )
    print(f"  분석 결과: {len(text_result.insights)}개 인사이트")
    for insight in text_result.insights:
        print(f"    - {insight}")
    
    print("\n✅ 전문화 에이전트 시스템 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_specialized_agents()) 