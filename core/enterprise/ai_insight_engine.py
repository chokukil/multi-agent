"""
AI-Based Insight Engine
Phase 4.3: 고급 분석 및 인사이트

핵심 기능:
- 자동 패턴 발견 및 분석
- 예측 분석 및 모델링
- 비즈니스 인사이트 생성
- 실시간 이상 감지
- 트렌드 분석 및 예측
- 자동 보고서 생성
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
import sqlite3
from pathlib import Path
import pickle
import warnings
from collections import defaultdict
import scipy.stats as stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """인사이트 유형"""
    PATTERN = "pattern"                    # 패턴 발견
    ANOMALY = "anomaly"                   # 이상 감지
    TREND = "trend"                       # 트렌드 분석
    CORRELATION = "correlation"            # 상관관계 분석
    PREDICTION = "prediction"             # 예측 분석
    CLUSTER = "cluster"                   # 클러스터 분석
    CLASSIFICATION = "classification"      # 분류 분석
    BUSINESS_RULE = "business_rule"       # 비즈니스 규칙
    OPTIMIZATION = "optimization"         # 최적화 제안

class SeverityLevel(Enum):
    """심각도 수준"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AnalysisMethod(Enum):
    """분석 방법"""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    TIME_SERIES = "time_series"
    GRAPH_ANALYSIS = "graph_analysis"
    RULE_BASED = "rule_based"

@dataclass
class Insight:
    """인사이트 정보"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    severity: SeverityLevel
    confidence_score: float  # 0.0 - 1.0
    method: AnalysisMethod
    data_points: List[Dict[str, Any]]
    recommendations: List[str]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    business_impact: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternAnalysis:
    """패턴 분석 결과"""
    pattern_id: str
    pattern_type: str
    frequency: int
    strength: float  # 패턴 강도
    variables: List[str]
    pattern_data: Dict[str, Any]
    statistical_significance: float

@dataclass
class AnomalyDetection:
    """이상 감지 결과"""
    anomaly_id: str
    detection_method: str
    anomaly_score: float
    affected_features: List[str]
    context: Dict[str, Any]
    timestamp: datetime

@dataclass
class TrendAnalysis:
    """트렌드 분석 결과"""
    trend_id: str
    direction: str  # increasing, decreasing, stable, cyclical
    strength: float
    seasonality: Optional[Dict[str, Any]]
    forecast_data: List[Dict[str, Any]]
    trend_equation: Optional[str]

class PatternDiscovery:
    """패턴 발견 엔진"""
    
    def __init__(self):
        self.discovered_patterns = {}
        
    async def discover_patterns(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """데이터에서 패턴 자동 발견"""
        patterns = []
        
        # 1. 통계적 패턴 발견
        statistical_patterns = await self._discover_statistical_patterns(df)
        patterns.extend(statistical_patterns)
        
        # 2. 시계열 패턴 발견
        if 'date' in df.columns or 'datetime' in df.columns or 'timestamp' in df.columns:
            time_patterns = await self._discover_time_patterns(df)
            patterns.extend(time_patterns)
        
        # 3. 클러스터링 패턴 발견
        cluster_patterns = await self._discover_cluster_patterns(df)
        patterns.extend(cluster_patterns)
        
        # 4. 상관관계 패턴 발견
        correlation_patterns = await self._discover_correlation_patterns(df)
        patterns.extend(correlation_patterns)
        
        logger.info(f"발견된 패턴 개수: {len(patterns)}")
        return patterns
    
    async def _discover_statistical_patterns(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """통계적 패턴 발견"""
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].dropna().empty:
                continue
                
            # 정규성 검정
            _, p_value = stats.normaltest(df[col].dropna())
            if p_value > 0.05:
                pattern = PatternAnalysis(
                    pattern_id=f"normal_dist_{col}",
                    pattern_type="normal_distribution",
                    frequency=len(df),
                    strength=1 - p_value,
                    variables=[col],
                    pattern_data={"p_value": p_value, "column": col},
                    statistical_significance=p_value
                )
                patterns.append(pattern)
            
            # 이상치 패턴
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                pattern = PatternAnalysis(
                    pattern_id=f"outliers_{col}",
                    pattern_type="outlier_pattern",
                    frequency=len(outliers),
                    strength=len(outliers) / len(df),
                    variables=[col],
                    pattern_data={
                        "outlier_count": len(outliers),
                        "outlier_percentage": len(outliers) / len(df) * 100,
                        "Q1": Q1, "Q3": Q3, "IQR": IQR
                    },
                    statistical_significance=0.05
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _discover_time_patterns(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """시계열 패턴 발견"""
        patterns = []
        
        # 날짜 컬럼 찾기
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        if not date_cols:
            return patterns
        
        date_col = date_cols[0]
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        df_time = df_time.sort_values(date_col)
        
        numeric_cols = df_time.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_time[col].dropna().empty:
                continue
                
            # 계절성 분석
            try:
                ts_data = df_time.set_index(date_col)[col].dropna()
                if len(ts_data) > 10:  # 최소 데이터 포인트
                    decomposition = seasonal_decompose(ts_data, period=min(7, len(ts_data)//2))
                    
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_data)
                    
                    if seasonal_strength > 0.1:  # 10% 이상의 계절성
                        pattern = PatternAnalysis(
                            pattern_id=f"seasonality_{col}",
                            pattern_type="seasonal_pattern",
                            frequency=7,  # 주간 패턴 가정
                            strength=seasonal_strength,
                            variables=[col, date_col],
                            pattern_data={
                                "seasonal_strength": seasonal_strength,
                                "trend_component": decomposition.trend.mean(),
                                "seasonal_component": decomposition.seasonal.mean()
                            },
                            statistical_significance=0.05
                        )
                        patterns.append(pattern)
            except Exception as e:
                logger.warning(f"시계열 분석 실패 {col}: {e}")
        
        return patterns
    
    async def _discover_cluster_patterns(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """클러스터링 패턴 발견"""
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return patterns
        
        # 데이터 전처리
        df_numeric = df[numeric_cols].dropna()
        if len(df_numeric) < 10:
            return patterns
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # K-means 클러스터링
        best_k = 2
        best_score = -1
        
        for k in range(2, min(10, len(df_numeric)//5)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        if best_score > 0.3:  # 적절한 클러스터링
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            pattern = PatternAnalysis(
                pattern_id=f"clusters_{best_k}",
                pattern_type="cluster_pattern",
                frequency=best_k,
                strength=best_score,
                variables=list(numeric_cols),
                pattern_data={
                    "n_clusters": best_k,
                    "silhouette_score": best_score,
                    "cluster_centers": kmeans.cluster_centers_.tolist()
                },
                statistical_significance=0.05
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _discover_correlation_patterns(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """상관관계 패턴 발견"""
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return patterns
        
        correlation_matrix = df[numeric_cols].corr()
        
        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 강한 상관관계
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    strong_correlations.append((col1, col2, corr_value))
        
        for col1, col2, corr_value in strong_correlations:
            pattern = PatternAnalysis(
                pattern_id=f"correlation_{col1}_{col2}",
                pattern_type="correlation_pattern",
                frequency=len(df),
                strength=abs(corr_value),
                variables=[col1, col2],
                pattern_data={
                    "correlation_coefficient": corr_value,
                    "relationship_type": "positive" if corr_value > 0 else "negative"
                },
                statistical_significance=0.05
            )
            patterns.append(pattern)
        
        return patterns

class AnomalyDetector:
    """이상 감지 엔진"""
    
    def __init__(self):
        self.detection_models = {}
        
    async def detect_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetection]:
        """데이터에서 이상 감지"""
        anomalies = []
        
        # 1. 통계적 이상 감지
        statistical_anomalies = await self._detect_statistical_anomalies(df)
        anomalies.extend(statistical_anomalies)
        
        # 2. 머신러닝 기반 이상 감지
        ml_anomalies = await self._detect_ml_anomalies(df)
        anomalies.extend(ml_anomalies)
        
        # 3. 시계열 이상 감지
        time_anomalies = await self._detect_time_series_anomalies(df)
        anomalies.extend(time_anomalies)
        
        logger.info(f"감지된 이상 개수: {len(anomalies)}")
        return anomalies
    
    async def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetection]:
        """통계적 이상 감지"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].dropna().empty:
                continue
                
            # Z-score 기반 이상 감지
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_indices = np.where(z_scores > 3)[0]
            
            for idx in outlier_indices:
                anomaly = AnomalyDetection(
                    anomaly_id=f"zscore_{col}_{idx}",
                    detection_method="z_score",
                    anomaly_score=z_scores[idx],
                    affected_features=[col],
                    context={
                        "value": df[col].iloc[idx],
                        "z_score": z_scores[idx],
                        "mean": df[col].mean(),
                        "std": df[col].std()
                    },
                    timestamp=datetime.now()
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_ml_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetection]:
        """머신러닝 기반 이상 감지"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return anomalies
        
        df_numeric = df[numeric_cols].dropna()
        if len(df_numeric) < 10:
            return anomalies
        
        # Isolation Forest 사용
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_predictions = isolation_forest.fit_predict(df_numeric)
        outlier_scores = isolation_forest.score_samples(df_numeric)
        
        outlier_indices = np.where(outlier_predictions == -1)[0]
        
        for idx in outlier_indices:
            anomaly = AnomalyDetection(
                anomaly_id=f"isolation_forest_{idx}",
                detection_method="isolation_forest",
                anomaly_score=abs(outlier_scores[idx]),
                affected_features=list(numeric_cols),
                context={
                    "row_index": idx,
                    "anomaly_score": outlier_scores[idx],
                    "data_point": df_numeric.iloc[idx].to_dict()
                },
                timestamp=datetime.now()
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_time_series_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetection]:
        """시계열 이상 감지"""
        anomalies = []
        
        # 날짜 컬럼 찾기
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        if not date_cols:
            return anomalies
        
        date_col = date_cols[0]
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        df_time = df_time.sort_values(date_col)
        
        numeric_cols = df_time.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_time[col].dropna().empty:
                continue
                
            ts_data = df_time.set_index(date_col)[col].dropna()
            if len(ts_data) < 10:
                continue
            
            # 이동평균 기반 이상 감지
            window_size = min(7, len(ts_data)//3)
            rolling_mean = ts_data.rolling(window=window_size).mean()
            rolling_std = ts_data.rolling(window=window_size).std()
            
            # 3 시그마 규칙
            upper_bound = rolling_mean + 3 * rolling_std
            lower_bound = rolling_mean - 3 * rolling_std
            
            anomaly_indices = ts_data[(ts_data > upper_bound) | (ts_data < lower_bound)].index
            
            for timestamp in anomaly_indices:
                anomaly = AnomalyDetection(
                    anomaly_id=f"time_series_{col}_{timestamp}",
                    detection_method="time_series_threshold",
                    anomaly_score=abs(ts_data[timestamp] - rolling_mean[timestamp]) / rolling_std[timestamp],
                    affected_features=[col],
                    context={
                        "timestamp": timestamp,
                        "value": ts_data[timestamp],
                        "expected_range": [lower_bound[timestamp], upper_bound[timestamp]],
                        "rolling_mean": rolling_mean[timestamp]
                    },
                    timestamp=datetime.now()
                )
                anomalies.append(anomaly)
        
        return anomalies

class TrendAnalyzer:
    """트렌드 분석 엔진"""
    
    def __init__(self):
        self.trend_models = {}
        
    async def analyze_trends(self, df: pd.DataFrame) -> List[TrendAnalysis]:
        """트렌드 분석 수행"""
        trends = []
        
        # 날짜 컬럼 찾기
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        if not date_cols:
            logger.warning("날짜 컬럼을 찾을 수 없어 트렌드 분석을 건너뜁니다.")
            return trends
        
        date_col = date_cols[0]
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        df_time = df_time.sort_values(date_col)
        
        numeric_cols = df_time.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_time[col].dropna().empty:
                continue
                
            trend = await self._analyze_single_trend(df_time, date_col, col)
            if trend:
                trends.append(trend)
        
        return trends
    
    async def _analyze_single_trend(self, df: pd.DataFrame, date_col: str, value_col: str) -> Optional[TrendAnalysis]:
        """단일 변수의 트렌드 분석"""
        ts_data = df.set_index(date_col)[value_col].dropna()
        
        if len(ts_data) < 5:
            return None
        
        # 선형 트렌드 분석
        x = np.arange(len(ts_data))
        y = ts_data.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 트렌드 방향 결정
        if abs(slope) < std_err:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = abs(r_value)
        else:
            direction = "decreasing"
            strength = abs(r_value)
        
        # 계절성 분석
        seasonality = None
        if len(ts_data) > 10:
            try:
                decomposition = seasonal_decompose(ts_data, period=min(7, len(ts_data)//2))
                seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_data)
                
                if seasonal_strength > 0.1:
                    seasonality = {
                        "strength": seasonal_strength,
                        "period": 7,
                        "component": decomposition.seasonal.mean()
                    }
            except Exception:
                pass
        
        # 예측 데이터 생성
        forecast_data = []
        future_x = np.arange(len(ts_data), len(ts_data) + 5)
        future_y = slope * future_x + intercept
        
        for i, pred_value in enumerate(future_y):
            forecast_data.append({
                "step": i + 1,
                "predicted_value": pred_value,
                "confidence_interval": [pred_value - 2*std_err, pred_value + 2*std_err]
            })
        
        return TrendAnalysis(
            trend_id=f"trend_{value_col}",
            direction=direction,
            strength=strength,
            seasonality=seasonality,
            forecast_data=forecast_data,
            trend_equation=f"y = {slope:.4f}x + {intercept:.4f}"
        )

class BusinessInsightGenerator:
    """비즈니스 인사이트 생성기"""
    
    def __init__(self):
        self.insight_rules = self._initialize_business_rules()
        
    def _initialize_business_rules(self) -> Dict[str, Callable]:
        """비즈니스 규칙 초기화"""
        return {
            "revenue_analysis": self._analyze_revenue_patterns,
            "customer_analysis": self._analyze_customer_patterns,
            "performance_analysis": self._analyze_performance_metrics,
            "risk_analysis": self._analyze_risk_factors,
            "efficiency_analysis": self._analyze_efficiency_metrics
        }
    
    async def generate_insights(self, df: pd.DataFrame, patterns: List[PatternAnalysis], 
                              anomalies: List[AnomalyDetection], trends: List[TrendAnalysis]) -> List[Insight]:
        """종합적인 비즈니스 인사이트 생성"""
        insights = []
        
        # 1. 패턴 기반 인사이트
        pattern_insights = await self._generate_pattern_insights(df, patterns)
        insights.extend(pattern_insights)
        
        # 2. 이상 기반 인사이트
        anomaly_insights = await self._generate_anomaly_insights(df, anomalies)
        insights.extend(anomaly_insights)
        
        # 3. 트렌드 기반 인사이트
        trend_insights = await self._generate_trend_insights(df, trends)
        insights.extend(trend_insights)
        
        # 4. 비즈니스 규칙 기반 인사이트
        business_insights = await self._generate_business_rule_insights(df)
        insights.extend(business_insights)
        
        # 인사이트 우선순위 정렬
        insights.sort(key=lambda x: (x.severity.value, -x.confidence_score))
        
        return insights
    
    async def _generate_pattern_insights(self, df: pd.DataFrame, patterns: List[PatternAnalysis]) -> List[Insight]:
        """패턴 기반 인사이트 생성"""
        insights = []
        
        for pattern in patterns:
            if pattern.pattern_type == "correlation_pattern":
                insight = Insight(
                    insight_id=f"insight_corr_{pattern.pattern_id}",
                    insight_type=InsightType.CORRELATION,
                    title=f"강한 상관관계 발견: {' vs '.join(pattern.variables)}",
                    description=f"{pattern.variables[0]}과 {pattern.variables[1]} 사이에 {pattern.strength:.2f}의 강한 상관관계가 발견되었습니다.",
                    severity=SeverityLevel.MEDIUM,
                    confidence_score=pattern.strength,
                    method=AnalysisMethod.STATISTICAL,
                    data_points=[pattern.pattern_data],
                    recommendations=[
                        f"{pattern.variables[0]}을 조정하여 {pattern.variables[1]}에 영향을 줄 수 있습니다.",
                        "이 관계를 활용한 예측 모델 개발을 고려해보세요."
                    ],
                    business_impact="변수 간 관계를 이해하여 더 나은 의사결정이 가능합니다."
                )
                insights.append(insight)
            
            elif pattern.pattern_type == "seasonal_pattern":
                insight = Insight(
                    insight_id=f"insight_seasonal_{pattern.pattern_id}",
                    insight_type=InsightType.TREND,
                    title=f"계절성 패턴 발견: {pattern.variables[0]}",
                    description=f"{pattern.variables[0]}에서 주기적인 패턴이 발견되었습니다 (강도: {pattern.strength:.2f})",
                    severity=SeverityLevel.MEDIUM,
                    confidence_score=pattern.strength,
                    method=AnalysisMethod.TIME_SERIES,
                    data_points=[pattern.pattern_data],
                    recommendations=[
                        "계절성을 고려한 예측 모델을 사용하세요.",
                        "주기적인 변동에 맞춰 리소스 계획을 수립하세요."
                    ],
                    business_impact="계절성을 고려한 계획으로 효율성을 개선할 수 있습니다."
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_anomaly_insights(self, df: pd.DataFrame, anomalies: List[AnomalyDetection]) -> List[Insight]:
        """이상 기반 인사이트 생성"""
        insights = []
        
        # 이상 빈도별 그룹화
        anomaly_counts = defaultdict(int)
        for anomaly in anomalies:
            for feature in anomaly.affected_features:
                anomaly_counts[feature] += 1
        
        for feature, count in anomaly_counts.items():
            if count > len(df) * 0.05:  # 5% 이상의 이상
                insight = Insight(
                    insight_id=f"insight_anomaly_{feature}",
                    insight_type=InsightType.ANOMALY,
                    title=f"높은 이상 빈도 감지: {feature}",
                    description=f"{feature}에서 {count}개의 이상값이 감지되었습니다 ({count/len(df)*100:.1f}%)",
                    severity=SeverityLevel.HIGH if count > len(df) * 0.1 else SeverityLevel.MEDIUM,
                    confidence_score=min(count / len(df) * 10, 1.0),
                    method=AnalysisMethod.MACHINE_LEARNING,
                    data_points=[{"feature": feature, "anomaly_count": count, "percentage": count/len(df)*100}],
                    recommendations=[
                        f"{feature}의 데이터 품질을 점검하세요.",
                        "이상값의 원인을 조사하고 데이터 수집 프로세스를 개선하세요.",
                        "이상값 처리 전략을 수립하세요."
                    ],
                    business_impact="데이터 품질 개선으로 분석 정확도를 향상시킬 수 있습니다."
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_trend_insights(self, df: pd.DataFrame, trends: List[TrendAnalysis]) -> List[Insight]:
        """트렌드 기반 인사이트 생성"""
        insights = []
        
        for trend in trends:
            if trend.strength > 0.7:  # 강한 트렌드
                insight = Insight(
                    insight_id=f"insight_trend_{trend.trend_id}",
                    insight_type=InsightType.TREND,
                    title=f"강한 {trend.direction} 트렌드 감지",
                    description=f"변수에서 {trend.direction} 방향의 강한 트렌드가 감지되었습니다 (강도: {trend.strength:.2f})",
                    severity=SeverityLevel.HIGH if trend.strength > 0.85 else SeverityLevel.MEDIUM,
                    confidence_score=trend.strength,
                    method=AnalysisMethod.TIME_SERIES,
                    data_points=[{
                        "trend_equation": trend.trend_equation,
                        "direction": trend.direction,
                        "strength": trend.strength,
                        "forecast": trend.forecast_data[:3]  # 첫 3개 예측
                    }],
                    recommendations=[
                        f"현재 {trend.direction} 트렌드를 고려한 계획을 수립하세요.",
                        "트렌드 지속성을 모니터링하고 적절한 조치를 취하세요.",
                        "예측 모델을 활용하여 미래 계획을 세우세요."
                    ],
                    business_impact="트렌드를 활용한 전략적 계획으로 경쟁 우위를 확보할 수 있습니다."
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_business_rule_insights(self, df: pd.DataFrame) -> List[Insight]:
        """비즈니스 규칙 기반 인사이트 생성"""
        insights = []
        
        # 매출/수익 관련 컬럼 찾기
        revenue_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'income', 'profit', '매출', '수익', '판매'])]
        
        for col in revenue_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                revenue_insights = await self._analyze_revenue_patterns(df, col)
                insights.extend(revenue_insights)
        
        # 고객 관련 컬럼 찾기
        customer_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['customer', 'client', 'user', '고객', '사용자'])]
        
        if customer_cols:
            customer_insights = await self._analyze_customer_patterns(df, customer_cols)
            insights.extend(customer_insights)
        
        return insights
    
    async def _analyze_revenue_patterns(self, df: pd.DataFrame, revenue_col: str) -> List[Insight]:
        """매출 패턴 분석"""
        insights = []
        
        if df[revenue_col].dropna().empty:
            return insights
        
        revenue_data = df[revenue_col].dropna()
        
        # 매출 분포 분석
        q25, q50, q75 = revenue_data.quantile([0.25, 0.5, 0.75])
        
        # 높은 수익 구간 분석
        high_revenue_threshold = q75 + 1.5 * (q75 - q25)
        high_revenue_count = len(revenue_data[revenue_data > high_revenue_threshold])
        
        if high_revenue_count > 0:
            insight = Insight(
                insight_id=f"insight_high_revenue_{revenue_col}",
                insight_type=InsightType.BUSINESS_RULE,
                title="고수익 기회 식별",
                description=f"{high_revenue_count}개의 고수익 케이스가 식별되었습니다 (임계값: {high_revenue_threshold:.2f})",
                severity=SeverityLevel.HIGH,
                confidence_score=0.8,
                method=AnalysisMethod.STATISTICAL,
                data_points=[{
                    "high_revenue_count": high_revenue_count,
                    "threshold": high_revenue_threshold,
                    "percentage": high_revenue_count / len(revenue_data) * 100
                }],
                recommendations=[
                    "고수익 케이스의 공통 특성을 분석하세요.",
                    "성공 패턴을 다른 케이스에 적용해보세요.",
                    "고수익 고객/제품에 집중하는 전략을 고려하세요."
                ],
                business_impact="고수익 패턴 복제로 전체 수익성을 개선할 수 있습니다."
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_customer_patterns(self, df: pd.DataFrame, customer_cols: List[str]) -> List[Insight]:
        """고객 패턴 분석"""
        insights = []
        
        # 고객 수 또는 고객 ID 기반 분석
        for col in customer_cols:
            if 'id' in col.lower():
                unique_customers = df[col].nunique()
                total_records = len(df)
                
                if unique_customers > 0:
                    avg_records_per_customer = total_records / unique_customers
                    
                    insight = Insight(
                        insight_id=f"insight_customer_activity_{col}",
                        insight_type=InsightType.BUSINESS_RULE,
                        title="고객 활동 패턴 분석",
                        description=f"고객당 평균 {avg_records_per_customer:.1f}개의 기록이 있습니다",
                        severity=SeverityLevel.MEDIUM,
                        confidence_score=0.7,
                        method=AnalysisMethod.STATISTICAL,
                        data_points=[{
                            "unique_customers": unique_customers,
                            "total_records": total_records,
                            "avg_records_per_customer": avg_records_per_customer
                        }],
                        recommendations=[
                            "고객 세분화 전략을 개발하세요.",
                            "활동이 많은 고객과 적은 고객을 구분하여 관리하세요.",
                            "고객 생애가치(CLV) 분석을 수행하세요."
                        ],
                        business_impact="고객별 맞춤 전략으로 고객 만족도와 수익성을 개선할 수 있습니다."
                    )
                    insights.append(insight)
        
        return insights
    
    async def _analyze_performance_metrics(self, df: pd.DataFrame) -> List[Insight]:
        """성능 지표 분석"""
        # 구체적인 성능 지표 분석 로직 구현
        return []
    
    async def _analyze_risk_factors(self, df: pd.DataFrame) -> List[Insight]:
        """위험 요소 분석"""
        # 구체적인 위험 분석 로직 구현
        return []
    
    async def _analyze_efficiency_metrics(self, df: pd.DataFrame) -> List[Insight]:
        """효율성 지표 분석"""
        # 구체적인 효율성 분석 로직 구현
        return []

class InsightDatabase:
    """인사이트 데이터베이스 관리"""
    
    def __init__(self, db_path: str = "core/enterprise/insights.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    method TEXT NOT NULL,
                    data_points TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    business_impact TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insight_performance (
                    insight_id TEXT,
                    execution_time_ms REAL,
                    dataset_size INTEGER,
                    accuracy_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (insight_id) REFERENCES insights (insight_id)
                )
            """)
    
    def save_insight(self, insight: Insight) -> bool:
        """인사이트 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO insights 
                    (insight_id, insight_type, title, description, severity, 
                     confidence_score, method, data_points, recommendations, 
                     business_impact, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.insight_id,
                    insight.insight_type.value,
                    insight.title,
                    insight.description,
                    insight.severity.value,
                    insight.confidence_score,
                    insight.method.value,
                    json.dumps(insight.data_points),
                    json.dumps(insight.recommendations),
                    insight.business_impact,
                    insight.created_at.isoformat(),
                    json.dumps(insight.metadata)
                ))
            return True
        except Exception as e:
            logger.error(f"인사이트 저장 실패: {e}")
            return False
    
    def get_insights(self, limit: int = 100) -> List[Insight]:
        """저장된 인사이트 조회"""
        insights = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM insights 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    insight = Insight(
                        insight_id=row[0],
                        insight_type=InsightType(row[1]),
                        title=row[2],
                        description=row[3],
                        severity=SeverityLevel(row[4]),
                        confidence_score=row[5],
                        method=AnalysisMethod(row[6]),
                        data_points=json.loads(row[7]),
                        recommendations=json.loads(row[8]),
                        business_impact=row[9],
                        created_at=datetime.fromisoformat(row[10]),
                        metadata=json.loads(row[11]) if row[11] else {}
                    )
                    insights.append(insight)
        except Exception as e:
            logger.error(f"인사이트 조회 실패: {e}")
        
        return insights

class AIInsightEngine:
    """AI 기반 인사이트 엔진 통합 클래스"""
    
    def __init__(self):
        self.pattern_discovery = PatternDiscovery()
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.insight_generator = BusinessInsightGenerator()
        self.database = InsightDatabase()
        
    async def analyze_data(self, df: pd.DataFrame, save_results: bool = True) -> Dict[str, Any]:
        """데이터 종합 분석 및 인사이트 생성"""
        start_time = time.time()
        
        logger.info(f"데이터 분석 시작: {df.shape}")
        
        try:
            # 1. 패턴 발견
            patterns = await self.pattern_discovery.discover_patterns(df)
            
            # 2. 이상 감지
            anomalies = await self.anomaly_detector.detect_anomalies(df)
            
            # 3. 트렌드 분석
            trends = await self.trend_analyzer.analyze_trends(df)
            
            # 4. 인사이트 생성
            insights = await self.insight_generator.generate_insights(df, patterns, anomalies, trends)
            
            # 5. 결과 저장
            if save_results:
                for insight in insights:
                    self.database.save_insight(insight)
            
            execution_time = (time.time() - start_time) * 1000
            
            result = {
                "status": "success",
                "execution_time_ms": execution_time,
                "dataset_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "analysis_results": {
                    "patterns_discovered": len(patterns),
                    "anomalies_detected": len(anomalies),
                    "trends_identified": len(trends),
                    "insights_generated": len(insights)
                },
                "patterns": [self._serialize_pattern(p) for p in patterns],
                "anomalies": [self._serialize_anomaly(a) for a in anomalies],
                "trends": [self._serialize_trend(t) for t in trends],
                "insights": [self._serialize_insight(i) for i in insights]
            }
            
            logger.info(f"분석 완료: {execution_time:.1f}ms, 인사이트 {len(insights)}개 생성")
            return result
            
        except Exception as e:
            logger.error(f"데이터 분석 실패: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            }
    
    def _serialize_pattern(self, pattern: PatternAnalysis) -> Dict[str, Any]:
        """패턴 분석 결과 직렬화"""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "frequency": pattern.frequency,
            "strength": pattern.strength,
            "variables": pattern.variables,
            "pattern_data": pattern.pattern_data,
            "statistical_significance": pattern.statistical_significance
        }
    
    def _serialize_anomaly(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """이상 감지 결과 직렬화"""
        return {
            "anomaly_id": anomaly.anomaly_id,
            "detection_method": anomaly.detection_method,
            "anomaly_score": anomaly.anomaly_score,
            "affected_features": anomaly.affected_features,
            "context": anomaly.context,
            "timestamp": anomaly.timestamp.isoformat()
        }
    
    def _serialize_trend(self, trend: TrendAnalysis) -> Dict[str, Any]:
        """트렌드 분석 결과 직렬화"""
        return {
            "trend_id": trend.trend_id,
            "direction": trend.direction,
            "strength": trend.strength,
            "seasonality": trend.seasonality,
            "forecast_data": trend.forecast_data,
            "trend_equation": trend.trend_equation
        }
    
    def _serialize_insight(self, insight: Insight) -> Dict[str, Any]:
        """인사이트 직렬화"""
        return {
            "insight_id": insight.insight_id,
            "insight_type": insight.insight_type.value,
            "title": insight.title,
            "description": insight.description,
            "severity": insight.severity.value,
            "confidence_score": insight.confidence_score,
            "method": insight.method.value,
            "data_points": insight.data_points,
            "recommendations": insight.recommendations,
            "business_impact": insight.business_impact,
            "created_at": insight.created_at.isoformat()
        }
    
    def get_insight_dashboard(self) -> Dict[str, Any]:
        """인사이트 대시보드 데이터 생성"""
        insights = self.database.get_insights(limit=50)
        
        # 심각도별 분류
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for insight in insights:
            severity_counts[insight.severity.value] += 1
            type_counts[insight.insight_type.value] += 1
        
        # 최근 중요 인사이트
        critical_insights = [i for i in insights if i.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        
        return {
            "total_insights": len(insights),
            "severity_distribution": dict(severity_counts),
            "type_distribution": dict(type_counts),
            "recent_critical_insights": [self._serialize_insight(i) for i in critical_insights[:5]],
            "average_confidence": np.mean([i.confidence_score for i in insights]) if insights else 0,
            "last_analysis": insights[0].created_at.isoformat() if insights else None
        }

# 전역 인스턴스
_ai_insight_engine = None

def get_ai_insight_engine() -> AIInsightEngine:
    """AI 인사이트 엔진 싱글톤 인스턴스 반환"""
    global _ai_insight_engine
    if _ai_insight_engine is None:
        _ai_insight_engine = AIInsightEngine()
    return _ai_insight_engine

# 편의 함수들
async def analyze_data_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """데이터 인사이트 분석 편의 함수"""
    engine = get_ai_insight_engine()
    return await engine.analyze_data(df)

def get_insight_dashboard() -> Dict[str, Any]:
    """인사이트 대시보드 편의 함수"""
    engine = get_ai_insight_engine()
    return engine.get_insight_dashboard()

async def test_ai_insight_engine():
    """AI 인사이트 엔진 테스트"""
    print("🧪 AI Insight Engine 테스트 시작")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    revenue = 1000 + 50 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 100, n_samples)
    customers = np.random.poisson(50, n_samples)
    
    # 이상값 추가
    revenue[100] = 5000  # 이상값
    revenue[200] = -500  # 이상값
    
    test_df = pd.DataFrame({
        'date': dates,
        'revenue': revenue,
        'customers': customers,
        'profit_margin': revenue * 0.2 + np.random.normal(0, 50, n_samples),
        'customer_id': [f'CUST_{i%100}' for i in range(n_samples)]
    })
    
    try:
        # 분석 실행
        result = await analyze_data_insights(test_df)
        
        print(f"✅ 분석 완료: {result['status']}")
        print(f"📊 실행 시간: {result['execution_time_ms']:.1f}ms")
        print(f"📈 패턴 발견: {result['analysis_results']['patterns_discovered']}개")
        print(f"🚨 이상 감지: {result['analysis_results']['anomalies_detected']}개")
        print(f"📊 트렌드: {result['analysis_results']['trends_identified']}개")
        print(f"💡 인사이트: {result['analysis_results']['insights_generated']}개")
        
        # 대시보드 테스트
        dashboard = get_insight_dashboard()
        print(f"📋 대시보드 인사이트 총계: {dashboard['total_insights']}개")
        
        print("✅ AI Insight Engine 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_ai_insight_engine()) 