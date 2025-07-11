"""
Intelligent Result Interpretation and Recommendation System

지능형 결과 해석 및 추천 시스템
- 다양한 에이전트 결과 통합 분석
- 지능형 인사이트 생성
- 실행 가능한 추천사항 제공
- 다음 단계 제안
- 도메인별 전문 해석
- 비즈니스 가치 추출

Author: CherryAI Team
Date: 2024-12-30
"""

import json
import logging
import re
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional imports for enhanced functionality
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Our imports
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    from core.auto_data_profiler import DataProfile, DataQuality
    from core.advanced_code_tracker import CodeExecution, ExecutionResult
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightType(Enum):
    """인사이트 유형"""
    DESCRIPTIVE = "descriptive"         # 기술 통계 인사이트
    DIAGNOSTIC = "diagnostic"           # 원인 분석 인사이트
    PREDICTIVE = "predictive"           # 예측적 인사이트
    PRESCRIPTIVE = "prescriptive"       # 처방적 인사이트
    COMPARATIVE = "comparative"         # 비교 분석 인사이트
    TREND = "trend"                     # 트렌드 분석 인사이트
    ANOMALY = "anomaly"                 # 이상 탐지 인사이트
    CORRELATION = "correlation"         # 상관관계 인사이트


class RecommendationType(Enum):
    """추천사항 유형"""
    DATA_QUALITY = "data_quality"       # 데이터 품질 개선
    ANALYSIS = "analysis"               # 추가 분석
    VISUALIZATION = "visualization"     # 시각화 개선
    MODELING = "modeling"               # 모델링 제안
    BUSINESS_ACTION = "business_action" # 비즈니스 액션
    TECHNICAL = "technical"             # 기술적 개선
    EXPLORATION = "exploration"         # 탐색적 분석
    VALIDATION = "validation"           # 검증 필요


class Priority(Enum):
    """우선순위"""
    CRITICAL = "critical"    # 즉시 조치 필요
    HIGH = "high"           # 높은 우선순위
    MEDIUM = "medium"       # 보통 우선순위
    LOW = "low"             # 낮은 우선순위
    INFO = "info"           # 정보성


@dataclass
class Insight:
    """인사이트 정보"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    evidence: List[str]
    confidence: float  # 0.0 ~ 1.0
    impact_score: float  # 0.0 ~ 1.0
    
    # 메타데이터
    data_sources: List[str] = None
    related_metrics: Dict[str, Any] = None
    generated_at: str = None
    
    # 비즈니스 컨텍스트
    business_implications: List[str] = None
    affected_stakeholders: List[str] = None


@dataclass
class Recommendation:
    """추천사항 정보"""
    recommendation_id: str
    recommendation_type: RecommendationType
    priority: Priority
    title: str
    description: str
    rationale: str
    
    # 실행 정보
    action_steps: List[str] = None
    estimated_effort: str = None  # "Low", "Medium", "High"
    expected_impact: str = None   # "Low", "Medium", "High"
    timeline: str = None          # "Immediate", "Short-term", "Long-term"
    
    # 리소스 정보
    required_skills: List[str] = None
    required_tools: List[str] = None
    prerequisites: List[str] = None
    
    # 메트릭
    success_metrics: List[str] = None
    risk_factors: List[str] = None


@dataclass
class InterpretationResult:
    """해석 결과"""
    session_id: str
    analysis_summary: str
    
    # 핵심 결과
    key_findings: List[str]
    insights: List[Insight]
    recommendations: List[Recommendation]
    
    # 다음 단계
    next_steps: List[str]
    alternative_approaches: List[str] = None
    
    # 메타데이터
    confidence_score: float = 0.0
    interpretation_timestamp: str = None
    data_sources_analyzed: List[str] = None
    
    # 비즈니스 가치
    business_value: str = None
    roi_potential: str = None


class DomainExpert:
    """도메인별 전문 해석기"""
    
    @staticmethod
    def interpret_statistical_results(results: Dict[str, Any]) -> List[Insight]:
        """통계 분석 결과 해석"""
        insights = []
        
        # 기술 통계 해석
        if "mean" in results and "std" in results:
            mean_val = results["mean"]
            std_val = results["std"]
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            
            if cv < 0.1:
                insights.append(Insight(
                    insight_id="low_variability",
                    insight_type=InsightType.DESCRIPTIVE,
                    title="낮은 변동성 탐지",
                    description=f"데이터의 변동계수가 {cv:.2%}로 매우 낮아 일관된 패턴을 보입니다.",
                    evidence=[f"평균: {mean_val:.2f}, 표준편차: {std_val:.2f}"],
                    confidence=0.9,
                    impact_score=0.6,
                    business_implications=["예측 가능한 패턴", "안정적인 프로세스"]
                ))
            elif cv > 0.5:
                insights.append(Insight(
                    insight_id="high_variability",
                    insight_type=InsightType.DIAGNOSTIC,
                    title="높은 변동성 발견",
                    description=f"데이터의 변동계수가 {cv:.2%}로 높아 불안정한 패턴을 보입니다.",
                    evidence=[f"평균: {mean_val:.2f}, 표준편차: {std_val:.2f}"],
                    confidence=0.9,
                    impact_score=0.8,
                    business_implications=["불확실성 증가", "리스크 관리 필요"]
                ))
        
        # 상관관계 해석
        if "correlations" in results:
            correlations = results["correlations"]
            if isinstance(correlations, dict):
                high_corrs = [(k, v) for k, v in correlations.items() if abs(v) > 0.7]
                if high_corrs:
                    insights.append(Insight(
                        insight_id="strong_correlations",
                        insight_type=InsightType.CORRELATION,
                        title="강한 상관관계 발견",
                        description=f"{len(high_corrs)}개의 변수 쌍에서 강한 상관관계가 발견되었습니다.",
                        evidence=[f"{k}: {v:.3f}" for k, v in high_corrs[:3]],
                        confidence=0.85,
                        impact_score=0.7,
                        business_implications=["변수 간 의존성", "다중공선성 가능성"]
                    ))
        
        return insights
    
    @staticmethod
    def interpret_data_quality_results(profile: DataProfile) -> List[Insight]:
        """데이터 품질 결과 해석"""
        insights = []
        
        # 전체 품질 평가
        if profile.overall_quality == DataQuality.EXCELLENT:
            insights.append(Insight(
                insight_id="excellent_quality",
                insight_type=InsightType.DESCRIPTIVE,
                title="우수한 데이터 품질",
                description="데이터 품질이 우수하여 신뢰할 수 있는 분석이 가능합니다.",
                evidence=[f"품질 점수: {profile.quality_score:.1%}"],
                confidence=0.95,
                impact_score=0.8,
                business_implications=["신뢰할 수 있는 의사결정", "분석 결과 신뢰성 높음"]
            ))
        elif profile.overall_quality in [DataQuality.POOR, DataQuality.CRITICAL]:
            insights.append(Insight(
                insight_id="poor_quality",
                insight_type=InsightType.DIAGNOSTIC,
                title="데이터 품질 문제",
                description="데이터 품질이 낮아 분석 결과의 신뢰성이 제한적입니다.",
                evidence=profile.data_quality_issues[:3],
                confidence=0.9,
                impact_score=0.9,
                business_implications=["분석 결과 신뢰성 저하", "데이터 정제 필요"]
            ))
        
        # 누락값 분석
        if profile.missing_percentage > 20:
            insights.append(Insight(
                insight_id="high_missing_data",
                insight_type=InsightType.DIAGNOSTIC,
                title="높은 누락값 비율",
                description=f"전체 데이터의 {profile.missing_percentage:.1f}%가 누락되어 있습니다.",
                evidence=[f"누락값: {profile.total_missing:,}개"],
                confidence=0.95,
                impact_score=0.7,
                business_implications=["분석 범위 제한", "데이터 수집 프로세스 검토 필요"]
            ))
        
        # 중복 데이터 분석
        if profile.duplicate_percentage > 10:
            insights.append(Insight(
                insight_id="high_duplicates",
                insight_type=InsightType.DIAGNOSTIC,
                title="높은 중복 데이터 비율",
                description=f"전체 데이터의 {profile.duplicate_percentage:.1f}%가 중복됩니다.",
                evidence=[f"중복 행: {profile.duplicate_rows:,}개"],
                confidence=0.9,
                impact_score=0.6,
                business_implications=["데이터 저장 비효율", "분석 결과 왜곡 가능성"]
            ))
        
        return insights
    
    @staticmethod
    def interpret_visualization_results(results: Dict[str, Any]) -> List[Insight]:
        """시각화 결과 해석"""
        insights = []
        
        # 분포 분석
        if "distribution_type" in results:
            dist_type = results["distribution_type"]
            if dist_type == "normal":
                insights.append(Insight(
                    insight_id="normal_distribution",
                    insight_type=InsightType.DESCRIPTIVE,
                    title="정규분포 패턴",
                    description="데이터가 정규분포를 따르는 것으로 보입니다.",
                    evidence=["정규성 검정 통과"],
                    confidence=0.8,
                    impact_score=0.6,
                    business_implications=["통계적 분석 적용 가능", "예측 모델 적합성 높음"]
                ))
            elif dist_type == "skewed":
                insights.append(Insight(
                    insight_id="skewed_distribution",
                    insight_type=InsightType.DIAGNOSTIC,
                    title="비대칭 분포",
                    description="데이터가 한쪽으로 치우친 분포를 보입니다.",
                    evidence=["왜도 검정 결과"],
                    confidence=0.85,
                    impact_score=0.7,
                    business_implications=["변환 필요", "이상치 확인 필요"]
                ))
        
        # 트렌드 분석
        if "trend" in results:
            trend = results["trend"]
            if trend == "increasing":
                insights.append(Insight(
                    insight_id="positive_trend",
                    insight_type=InsightType.TREND,
                    title="증가 추세",
                    description="시간에 따른 증가 추세가 관찰됩니다.",
                    evidence=["선형 회귀 기울기 양수"],
                    confidence=0.8,
                    impact_score=0.8,
                    business_implications=["성장 기회", "지속 가능성 검토"]
                ))
            elif trend == "decreasing":
                insights.append(Insight(
                    insight_id="negative_trend",
                    insight_type=InsightType.TREND,
                    title="감소 추세",
                    description="시간에 따른 감소 추세가 관찰됩니다.",
                    evidence=["선형 회귀 기울기 음수"],
                    confidence=0.8,
                    impact_score=0.9,
                    business_implications=["주의 필요", "개선 방안 모색"]
                ))
        
        return insights
    
    @staticmethod
    def interpret_ml_results(results: Dict[str, Any]) -> List[Insight]:
        """머신러닝 결과 해석"""
        insights = []
        
        # 모델 성능 해석
        if "accuracy" in results:
            accuracy = results["accuracy"]
            if accuracy > 0.9:
                insights.append(Insight(
                    insight_id="high_accuracy",
                    insight_type=InsightType.PREDICTIVE,
                    title="높은 모델 정확도",
                    description=f"모델 정확도가 {accuracy:.1%}로 매우 우수합니다.",
                    evidence=[f"정확도: {accuracy:.3f}"],
                    confidence=0.9,
                    impact_score=0.9,
                    business_implications=["신뢰할 수 있는 예측", "운영 환경 적용 가능"]
                ))
            elif accuracy < 0.7:
                insights.append(Insight(
                    insight_id="low_accuracy",
                    insight_type=InsightType.DIAGNOSTIC,
                    title="낮은 모델 정확도",
                    description=f"모델 정확도가 {accuracy:.1%}로 개선이 필요합니다.",
                    evidence=[f"정확도: {accuracy:.3f}"],
                    confidence=0.95,
                    impact_score=0.8,
                    business_implications=["모델 개선 필요", "추가 데이터 수집 고려"]
                ))
        
        # 특성 중요도 해석
        if "feature_importance" in results:
            importance = results["feature_importance"]
            if isinstance(importance, dict):
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                insights.append(Insight(
                    insight_id="key_features",
                    insight_type=InsightType.PRESCRIPTIVE,
                    title="주요 특성 식별",
                    description="모델 예측에 가장 중요한 특성들이 식별되었습니다.",
                    evidence=[f"{k}: {v:.3f}" for k, v in top_features],
                    confidence=0.85,
                    impact_score=0.8,
                    business_implications=["핵심 변수 집중", "리소스 효율적 배분"]
                ))
        
        return insights


class RecommendationEngine:
    """추천 엔진"""
    
    @staticmethod
    def generate_data_quality_recommendations(profile: DataProfile) -> List[Recommendation]:
        """데이터 품질 기반 추천사항"""
        recommendations = []
        
        # 누락값 처리 추천
        if profile.missing_percentage > 10:
            priority = Priority.HIGH if profile.missing_percentage > 30 else Priority.MEDIUM
            recommendations.append(Recommendation(
                recommendation_id="handle_missing_data",
                recommendation_type=RecommendationType.DATA_QUALITY,
                priority=priority,
                title="누락값 처리",
                description="높은 누락값 비율로 인한 데이터 품질 개선이 필요합니다.",
                rationale=f"전체 데이터의 {profile.missing_percentage:.1f}%가 누락되어 분석 정확도에 영향을 줍니다.",
                action_steps=[
                    "누락값 패턴 분석",
                    "적절한 대치 방법 선택 (평균, 중앙값, 모드 등)",
                    "누락값 처리 후 데이터 검증",
                    "처리 전후 분석 결과 비교"
                ],
                estimated_effort="Medium",
                expected_impact="High",
                timeline="Short-term",
                required_skills=["데이터 전처리", "통계학"],
                success_metrics=["누락값 비율 감소", "데이터 품질 점수 향상"]
            ))
        
        # 중복 데이터 제거 추천
        if profile.duplicate_percentage > 5:
            recommendations.append(Recommendation(
                recommendation_id="remove_duplicates",
                recommendation_type=RecommendationType.DATA_QUALITY,
                priority=Priority.MEDIUM,
                title="중복 데이터 제거",
                description="중복 데이터를 제거하여 분석 정확도를 향상시키세요.",
                rationale=f"{profile.duplicate_percentage:.1f}%의 중복 데이터가 분석 결과를 왜곡할 수 있습니다.",
                action_steps=[
                    "중복 기준 정의",
                    "중복 데이터 식별 및 검토",
                    "중복 제거 규칙 적용",
                    "데이터 무결성 검증"
                ],
                estimated_effort="Low",
                expected_impact="Medium",
                timeline="Immediate"
            ))
        
        # 데이터 타입 최적화 추천
        memory_usage_mb = profile.memory_usage
        if memory_usage_mb > 100:  # 100MB 이상
            recommendations.append(Recommendation(
                recommendation_id="optimize_data_types",
                recommendation_type=RecommendationType.TECHNICAL,
                priority=Priority.LOW,
                title="데이터 타입 최적화",
                description="메모리 사용량을 줄이기 위해 데이터 타입을 최적화하세요.",
                rationale=f"현재 {memory_usage_mb:.1f}MB의 메모리를 사용하고 있어 최적화가 필요합니다.",
                action_steps=[
                    "각 컬럼의 실제 데이터 범위 분석",
                    "적절한 데이터 타입 선택 (int32 vs int64 등)",
                    "범주형 데이터 category 타입 변환",
                    "메모리 사용량 비교 검증"
                ],
                estimated_effort="Low",
                expected_impact="Medium",
                timeline="Short-term",
                required_skills=["데이터 엔지니어링"]
            ))
        
        return recommendations
    
    @staticmethod
    def generate_analysis_recommendations(insights: List[Insight]) -> List[Recommendation]:
        """인사이트 기반 분석 추천사항"""
        recommendations = []
        
        # 상관관계 발견 시 추가 분석 추천
        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION]
        if correlation_insights:
            recommendations.append(Recommendation(
                recommendation_id="correlation_analysis",
                recommendation_type=RecommendationType.ANALYSIS,
                priority=Priority.MEDIUM,
                title="상관관계 심화 분석",
                description="발견된 상관관계에 대한 심화 분석을 수행하세요.",
                rationale="강한 상관관계가 발견되어 인과관계 분석이 필요합니다.",
                action_steps=[
                    "편상관계수 계산",
                    "시차 상관관계 분석",
                    "인과관계 검정 (Granger causality)",
                    "도메인 전문가 검토"
                ],
                estimated_effort="Medium",
                expected_impact="High",
                timeline="Short-term",
                required_skills=["통계학", "도메인 지식"]
            ))
        
        # 이상치 발견 시 추천
        anomaly_insights = [i for i in insights if i.insight_type == InsightType.ANOMALY]
        if anomaly_insights:
            recommendations.append(Recommendation(
                recommendation_id="anomaly_investigation",
                recommendation_type=RecommendationType.ANALYSIS,
                priority=Priority.HIGH,
                title="이상치 조사",
                description="발견된 이상치에 대한 상세 조사를 진행하세요.",
                rationale="이상치가 데이터 오류인지 실제 현상인지 확인이 필요합니다.",
                action_steps=[
                    "이상치 발생 원인 조사",
                    "데이터 수집 과정 검토",
                    "도메인 전문가 상담",
                    "이상치 처리 방안 결정"
                ],
                estimated_effort="High",
                expected_impact="High",
                timeline="Immediate",
                required_skills=["데이터 분석", "도메인 지식"],
                risk_factors=["이상치 오판 시 정보 손실"]
            ))
        
        # 예측 모델링 추천
        predictive_insights = [i for i in insights if i.insight_type == InsightType.PREDICTIVE]
        if predictive_insights or len(insights) >= 3:
            recommendations.append(Recommendation(
                recommendation_id="predictive_modeling",
                recommendation_type=RecommendationType.MODELING,
                priority=Priority.MEDIUM,
                title="예측 모델 구축",
                description="수집된 인사이트를 바탕으로 예측 모델을 구축하세요.",
                rationale="충분한 데이터와 패턴이 확인되어 예측 모델링이 가능합니다.",
                action_steps=[
                    "목표 변수 정의",
                    "특성 선택 및 엔지니어링",
                    "모델 알고리즘 선택",
                    "모델 훈련 및 평가",
                    "모델 해석 및 검증"
                ],
                estimated_effort="High",
                expected_impact="High",
                timeline="Long-term",
                required_skills=["머신러닝", "통계학", "프로그래밍"],
                required_tools=["Python/R", "ML 라이브러리"],
                success_metrics=["모델 정확도", "비즈니스 KPI 개선"]
            ))
        
        return recommendations
    
    @staticmethod
    def generate_visualization_recommendations(results: Dict[str, Any]) -> List[Recommendation]:
        """시각화 추천사항"""
        recommendations = []
        
        # 기본 시각화 추천
        recommendations.append(Recommendation(
            recommendation_id="comprehensive_visualization",
            recommendation_type=RecommendationType.VISUALIZATION,
            priority=Priority.MEDIUM,
            title="포괄적 시각화",
            description="데이터의 다양한 측면을 보여주는 시각화를 생성하세요.",
            rationale="시각화를 통해 데이터 패턴을 더 명확하게 이해할 수 있습니다.",
            action_steps=[
                "분포 히스토그램 생성",
                "상관관계 히트맵 작성",
                "시계열 트렌드 차트 (시간 데이터 있는 경우)",
                "박스플롯으로 이상치 확인",
                "산점도로 관계 탐색"
            ],
            estimated_effort="Low",
            expected_impact="Medium",
            timeline="Immediate",
            required_tools=["Matplotlib", "Seaborn", "Plotly"],
            success_metrics=["인사이트 발견 개수", "이해관계자 만족도"]
        ))
        
        # 대시보드 구축 추천
        if "multiple_metrics" in results:
            recommendations.append(Recommendation(
                recommendation_id="interactive_dashboard",
                recommendation_type=RecommendationType.VISUALIZATION,
                priority=Priority.LOW,
                title="인터랙티브 대시보드",
                description="실시간 모니터링을 위한 대시보드를 구축하세요.",
                rationale="여러 메트릭이 있어 통합적인 모니터링이 필요합니다.",
                action_steps=[
                    "핵심 KPI 정의",
                    "대시보드 레이아웃 설계",
                    "인터랙티브 기능 구현",
                    "자동 업데이트 설정",
                    "사용자 피드백 수집"
                ],
                estimated_effort="High",
                expected_impact="High",
                timeline="Long-term",
                required_skills=["웹 개발", "데이터 시각화"],
                required_tools=["Streamlit", "Dash", "Tableau"]
            ))
        
        return recommendations


class IntelligentResultInterpreter:
    """
    지능형 결과 해석 및 추천 시스템
    
    다양한 분석 결과를 통합하여 의미 있는 인사이트와 추천사항을 제공
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enhanced_tracer = None
        self.openai_client = None
        
        # 해석 히스토리
        self.interpretation_history: List[InterpretationResult] = []
        
        # 도메인 전문가들
        self.domain_expert = DomainExpert()
        self.recommendation_engine = RecommendationEngine()
        
        # 시스템 초기화
        self._initialize_systems()
        
        logger.info("🧠 Intelligent Result Interpreter 초기화 완료")
    
    def _initialize_systems(self):
        """핵심 시스템 초기화"""
        if CORE_SYSTEMS_AVAILABLE:
            try:
                self.enhanced_tracer = get_enhanced_tracer()
                logger.info("✅ Enhanced tracking activated")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced tracking initialization failed: {e}")
        
        if OPENAI_AVAILABLE:
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI initialization failed: {e}")
    
    def interpret_results(
        self,
        session_id: str,
        results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        data_profile: Optional[DataProfile] = None,
        code_executions: Optional[List[CodeExecution]] = None
    ) -> InterpretationResult:
        """종합적인 결과 해석"""
        try:
            logger.info(f"🔍 결과 해석 시작: {session_id}")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "result_interpretation_start",
                    {"session_id": session_id, "context": context},
                    "Starting intelligent result interpretation"
                )
            
            # 인사이트 수집
            all_insights = []
            
            # 통계 결과 해석
            if "statistics" in results:
                stat_insights = self.domain_expert.interpret_statistical_results(results["statistics"])
                all_insights.extend(stat_insights)
            
            # 데이터 품질 해석
            if data_profile:
                quality_insights = self.domain_expert.interpret_data_quality_results(data_profile)
                all_insights.extend(quality_insights)
            
            # 시각화 결과 해석
            if "visualization" in results:
                viz_insights = self.domain_expert.interpret_visualization_results(results["visualization"])
                all_insights.extend(viz_insights)
            
            # 머신러닝 결과 해석
            if "machine_learning" in results:
                ml_insights = self.domain_expert.interpret_ml_results(results["machine_learning"])
                all_insights.extend(ml_insights)
            
            # 코드 실행 결과 해석
            if code_executions:
                code_insights = self._interpret_code_execution_results(code_executions)
                all_insights.extend(code_insights)
            
            # 일반적인 결과 해석
            general_insights = self._interpret_general_results(results)
            all_insights.extend(general_insights)
            
            # 인사이트 우선순위 정렬
            all_insights.sort(key=lambda x: (x.impact_score * x.confidence), reverse=True)
            
            # 추천사항 생성
            recommendations = []
            
            if data_profile:
                quality_recs = self.recommendation_engine.generate_data_quality_recommendations(data_profile)
                recommendations.extend(quality_recs)
            
            analysis_recs = self.recommendation_engine.generate_analysis_recommendations(all_insights)
            recommendations.extend(analysis_recs)
            
            viz_recs = self.recommendation_engine.generate_visualization_recommendations(results)
            recommendations.extend(viz_recs)
            
            # 추천사항 우선순위 정렬
            priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3, Priority.INFO: 4}
            recommendations.sort(key=lambda x: priority_order[x.priority])
            
            # 핵심 발견사항 요약
            key_findings = self._generate_key_findings(all_insights, results)
            
            # 분석 요약 생성
            analysis_summary = self._generate_analysis_summary(key_findings, all_insights, recommendations)
            
            # 다음 단계 제안
            next_steps = self._generate_next_steps(all_insights, recommendations)
            
            # 신뢰도 점수 계산
            confidence_score = self._calculate_overall_confidence(all_insights, results)
            
            # 비즈니스 가치 평가
            business_value = self._assess_business_value(all_insights, recommendations)
            
            # 해석 결과 생성
            interpretation = InterpretationResult(
                session_id=session_id,
                analysis_summary=analysis_summary,
                key_findings=key_findings,
                insights=all_insights[:10],  # 상위 10개
                recommendations=recommendations[:8],  # 상위 8개
                next_steps=next_steps,
                confidence_score=confidence_score,
                interpretation_timestamp=datetime.now().isoformat(),
                data_sources_analyzed=list(results.keys()),
                business_value=business_value
            )
            
            # 히스토리에 추가
            self.interpretation_history.append(interpretation)
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "result_interpretation_complete",
                    {
                        "session_id": session_id,
                        "insights_count": len(all_insights),
                        "recommendations_count": len(recommendations),
                        "confidence_score": confidence_score
                    },
                    "Result interpretation completed successfully"
                )
            
            logger.info(f"✅ 결과 해석 완료: {len(all_insights)}개 인사이트, {len(recommendations)}개 추천사항")
            return interpretation
            
        except Exception as e:
            logger.error(f"❌ 결과 해석 실패: {e}")
            
            # 기본 해석 결과 반환
            return InterpretationResult(
                session_id=session_id,
                analysis_summary="결과 해석 중 오류가 발생했습니다.",
                key_findings=[f"해석 오류: {str(e)}"],
                insights=[],
                recommendations=[],
                next_steps=["시스템 상태를 확인하고 다시 시도해주세요."],
                confidence_score=0.0,
                interpretation_timestamp=datetime.now().isoformat()
            )
    
    def _interpret_code_execution_results(self, executions: List[CodeExecution]) -> List[Insight]:
        """코드 실행 결과 해석"""
        insights = []
        
        if not executions:
            return insights
        
        # 성공률 분석
        successful_executions = [e for e in executions if e.execution_result and e.execution_result.status.value == "success"]
        success_rate = len(successful_executions) / len(executions)
        
        if success_rate < 0.5:
            insights.append(Insight(
                insight_id="low_code_success_rate",
                insight_type=InsightType.DIAGNOSTIC,
                title="낮은 코드 실행 성공률",
                description=f"코드 실행 성공률이 {success_rate:.1%}로 낮습니다.",
                evidence=[f"총 {len(executions)}개 중 {len(successful_executions)}개 성공"],
                confidence=0.9,
                impact_score=0.7,
                business_implications=["코드 품질 개선 필요", "디버깅 프로세스 강화"]
            ))
        
        # 복잡도 분석
        if executions and executions[0].code_metrics:
            complexities = [e.code_metrics.complexity_score for e in executions if e.code_metrics]
            if complexities:
                avg_complexity = statistics.mean(complexities)
                if avg_complexity > 6:
                    insights.append(Insight(
                        insight_id="high_code_complexity",
                        insight_type=InsightType.DIAGNOSTIC,
                        title="높은 코드 복잡도",
                        description=f"평균 코드 복잡도가 {avg_complexity:.1f}로 높습니다.",
                        evidence=[f"복잡도 범위: {min(complexities):.1f} - {max(complexities):.1f}"],
                        confidence=0.8,
                        impact_score=0.6,
                        business_implications=["유지보수 어려움", "코드 리팩토링 필요"]
                    ))
        
        return insights
    
    def _interpret_general_results(self, results: Dict[str, Any]) -> List[Insight]:
        """일반적인 결과 해석"""
        insights = []
        
        # 데이터 크기 분석
        if "data_size" in results:
            size = results["data_size"]
            if isinstance(size, dict) and "rows" in size:
                rows = size["rows"]
                if rows < 100:
                    insights.append(Insight(
                        insight_id="small_dataset",
                        insight_type=InsightType.DIAGNOSTIC,
                        title="작은 데이터셋",
                        description=f"데이터셋 크기가 {rows}행으로 작습니다.",
                        evidence=[f"행 수: {rows}"],
                        confidence=0.95,
                        impact_score=0.6,
                        business_implications=["통계적 유의성 제한", "더 많은 데이터 수집 필요"]
                    ))
                elif rows > 100000:
                    insights.append(Insight(
                        insight_id="large_dataset",
                        insight_type=InsightType.DESCRIPTIVE,
                        title="대용량 데이터셋",
                        description=f"데이터셋 크기가 {rows:,}행으로 대용량입니다.",
                        evidence=[f"행 수: {rows:,}"],
                        confidence=0.95,
                        impact_score=0.7,
                        business_implications=["고급 분석 기법 적용 가능", "처리 성능 고려 필요"]
                    ))
        
        # 처리 시간 분석
        if "processing_time" in results:
            proc_time = results["processing_time"]
            if proc_time > 60:  # 1분 이상
                insights.append(Insight(
                    insight_id="long_processing_time",
                    insight_type=InsightType.DIAGNOSTIC,
                    title="긴 처리 시간",
                    description=f"분석 처리 시간이 {proc_time:.1f}초로 깁니다.",
                    evidence=[f"처리 시간: {proc_time:.1f}초"],
                    confidence=0.9,
                    impact_score=0.5,
                    business_implications=["성능 최적화 필요", "리소스 증설 고려"]
                ))
        
        return insights
    
    def _generate_key_findings(self, insights: List[Insight], results: Dict[str, Any]) -> List[str]:
        """핵심 발견사항 생성"""
        findings = []
        
        # 높은 영향도의 인사이트를 핵심 발견사항으로
        high_impact_insights = [i for i in insights if i.impact_score > 0.7]
        for insight in high_impact_insights[:5]:
            findings.append(f"{insight.title}: {insight.description}")
        
        # 결과 기반 발견사항 추가
        if "summary" in results:
            summary = results["summary"]
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        findings.append(f"{key}: {value}")
        
        # 최소 한 개의 발견사항 보장
        if not findings:
            findings.append("분석이 완료되었습니다. 상세 인사이트를 확인해주세요.")
        
        return findings[:7]  # 최대 7개
    
    def _generate_analysis_summary(
        self, 
        key_findings: List[str], 
        insights: List[Insight], 
        recommendations: List[Recommendation]
    ) -> str:
        """분석 요약 생성"""
        summary_parts = []
        
        # 핵심 발견사항 요약
        if key_findings:
            summary_parts.append(f"주요 발견사항: {len(key_findings)}개의 핵심 인사이트가 도출되었습니다.")
        
        # 인사이트 유형별 분류
        if insights:
            insight_types = {}
            for insight in insights:
                insight_type = insight.insight_type.value
                insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
            
            type_summary = ", ".join([f"{k} {v}개" for k, v in insight_types.items()])
            summary_parts.append(f"인사이트 구성: {type_summary}")
        
        # 추천사항 우선순위 요약
        if recommendations:
            priority_counts = {}
            for rec in recommendations:
                priority = rec.priority.value
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            high_priority = priority_counts.get("high", 0) + priority_counts.get("critical", 0)
            if high_priority > 0:
                summary_parts.append(f"높은 우선순위 액션 아이템: {high_priority}개")
        
        # 신뢰도 및 품질 언급
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if high_confidence_insights:
            summary_parts.append(f"높은 신뢰도 인사이트: {len(high_confidence_insights)}개")
        
        return " ".join(summary_parts) if summary_parts else "분석이 완료되었습니다."
    
    def _generate_next_steps(self, insights: List[Insight], recommendations: List[Recommendation]) -> List[str]:
        """다음 단계 제안"""
        next_steps = []
        
        # 긴급한 추천사항을 다음 단계로
        urgent_recs = [r for r in recommendations if r.priority in [Priority.CRITICAL, Priority.HIGH]]
        for rec in urgent_recs[:3]:
            if rec.action_steps:
                next_steps.append(f"{rec.title}: {rec.action_steps[0]}")
            else:
                next_steps.append(rec.title)
        
        # 추가 분석 제안
        predictive_insights = [i for i in insights if i.insight_type == InsightType.PREDICTIVE]
        if not predictive_insights and len(insights) >= 2:
            next_steps.append("예측 분석을 통한 미래 트렌드 파악")
        
        # 시각화 제안
        if not any("시각화" in step for step in next_steps):
            next_steps.append("핵심 인사이트를 위한 시각화 대시보드 구축")
        
        # 기본 다음 단계
        if not next_steps:
            next_steps = [
                "추가 데이터 수집 및 검증",
                "결과에 대한 도메인 전문가 검토",
                "비즈니스 의사결정 반영"
            ]
        
        return next_steps[:5]  # 최대 5개
    
    def _calculate_overall_confidence(self, insights: List[Insight], results: Dict[str, Any]) -> float:
        """전체 신뢰도 점수 계산"""
        if not insights:
            return 0.5  # 기본값
        
        # 인사이트들의 가중 평균 신뢰도
        weighted_confidence = sum(i.confidence * i.impact_score for i in insights)
        total_weight = sum(i.impact_score for i in insights)
        
        if total_weight == 0:
            return 0.5
        
        base_confidence = weighted_confidence / total_weight
        
        # 데이터 품질에 따른 조정
        data_quality_factor = 1.0
        if "data_quality_score" in results:
            quality_score = results["data_quality_score"]
            data_quality_factor = quality_score
        
        # 결과 일관성에 따른 조정
        consistency_factor = 1.0
        if len(insights) >= 3:
            # 서로 모순되는 인사이트가 있는지 확인 (간단한 휴리스틱)
            contradictory_count = 0
            for i, insight1 in enumerate(insights):
                for insight2 in insights[i+1:]:
                    if insight1.insight_type != insight2.insight_type:
                        if abs(insight1.impact_score - insight2.impact_score) > 0.5:
                            contradictory_count += 1
            
            if contradictory_count > 0:
                consistency_factor = max(0.7, 1.0 - (contradictory_count * 0.1))
        
        final_confidence = base_confidence * data_quality_factor * consistency_factor
        return max(0.0, min(1.0, final_confidence))
    
    def _assess_business_value(self, insights: List[Insight], recommendations: List[Recommendation]) -> str:
        """비즈니스 가치 평가"""
        # 높은 영향도 인사이트 수
        high_impact_count = len([i for i in insights if i.impact_score > 0.7])
        
        # 실행 가능한 추천사항 수
        actionable_recs = len([r for r in recommendations if r.action_steps])
        
        # 비즈니스 의미가 있는 인사이트 수
        business_insights = len([i for i in insights if i.business_implications])
        
        # 가치 평가
        if high_impact_count >= 3 and actionable_recs >= 2:
            return "높음 - 즉시 실행 가능한 고영향 인사이트 다수 발견"
        elif high_impact_count >= 2 or actionable_recs >= 3:
            return "중간 - 유의미한 인사이트와 실행 방안 확보"
        elif business_insights >= 1:
            return "낮음 - 일부 비즈니스 인사이트 확인"
        else:
            return "제한적 - 추가 분석 및 데이터 수집 필요"
    
    def get_interpretation_history(self, session_id: Optional[str] = None, limit: int = 10) -> List[InterpretationResult]:
        """해석 히스토리 조회"""
        if session_id:
            filtered_history = [h for h in self.interpretation_history if h.session_id == session_id]
            return filtered_history[-limit:]
        else:
            return self.interpretation_history[-limit:]
    
    def generate_comprehensive_report(self, interpretation: InterpretationResult) -> str:
        """종합 보고서 생성"""
        report = f"""# 데이터 분석 종합 보고서

**세션 ID**: {interpretation.session_id}
**분석 일시**: {interpretation.interpretation_timestamp}
**신뢰도 점수**: {interpretation.confidence_score:.1%}

## 📊 분석 요약
{interpretation.analysis_summary}

## 🔍 핵심 발견사항
{chr(10).join(f'- {finding}' for finding in interpretation.key_findings)}

## 💡 주요 인사이트
"""
        
        for i, insight in enumerate(interpretation.insights[:5], 1):
            report += f"""
### {i}. {insight.title}
- **유형**: {insight.insight_type.value}
- **설명**: {insight.description}
- **신뢰도**: {insight.confidence:.1%}
- **영향도**: {insight.impact_score:.1%}
"""
            if insight.evidence:
                report += f"- **근거**: {', '.join(insight.evidence[:3])}\n"
            if insight.business_implications:
                report += f"- **비즈니스 의미**: {', '.join(insight.business_implications[:2])}\n"
        
        report += f"""
## 🎯 추천사항
"""
        
        for i, rec in enumerate(interpretation.recommendations[:5], 1):
            report += f"""
### {i}. {rec.title} ({rec.priority.value.upper()})
- **설명**: {rec.description}
- **이유**: {rec.rationale}
- **예상 노력**: {rec.estimated_effort or 'N/A'}
- **예상 효과**: {rec.expected_impact or 'N/A'}
"""
            if rec.action_steps:
                report += f"- **실행 단계**: {', '.join(rec.action_steps[:3])}\n"
        
        report += f"""
## 🚀 다음 단계
{chr(10).join(f'- {step}' for step in interpretation.next_steps)}

## 💰 비즈니스 가치
{interpretation.business_value or '평가 중'}

---
*본 보고서는 AI 기반 지능형 분석 시스템에 의해 자동 생성되었습니다.*
"""
        
        return report


# 전역 인스턴스
_interpreter_instance = None


def get_intelligent_result_interpreter(config: Optional[Dict] = None) -> IntelligentResultInterpreter:
    """Intelligent Result Interpreter 인스턴스 반환"""
    global _interpreter_instance
    if _interpreter_instance is None:
        _interpreter_instance = IntelligentResultInterpreter(config)
    return _interpreter_instance


# 편의 함수들
def interpret_analysis_results(
    session_id: str,
    results: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    data_profile: Optional[DataProfile] = None
) -> InterpretationResult:
    """분석 결과 해석 편의 함수"""
    interpreter = get_intelligent_result_interpreter()
    return interpreter.interpret_results(session_id, results, context, data_profile)


def generate_insight_report(interpretation: InterpretationResult) -> str:
    """인사이트 보고서 생성 편의 함수"""
    interpreter = get_intelligent_result_interpreter()
    return interpreter.generate_comprehensive_report(interpretation)


# CLI 테스트 함수
def test_intelligent_result_interpreter():
    """Intelligent Result Interpreter 테스트"""
    print("🧠 Intelligent Result Interpreter 테스트 시작\n")
    
    interpreter = get_intelligent_result_interpreter()
    
    # 테스트 데이터 준비
    sample_results = {
        "statistics": {
            "mean": 75.5,
            "std": 12.3,
            "correlations": {
                "var1_var2": 0.85,
                "var1_var3": 0.23,
                "var2_var3": 0.78
            }
        },
        "data_size": {
            "rows": 1500,
            "columns": 8
        },
        "processing_time": 45.2,
        "visualization": {
            "distribution_type": "normal",
            "trend": "increasing"
        }
    }
    
    # 데이터 프로파일 시뮬레이션
    if PANDAS_AVAILABLE:
        from core.auto_data_profiler import DataProfile, DataQuality
        sample_profile = DataProfile(
            dataset_name="Sample Dataset",
            shape=(1500, 8),
            memory_usage=12.5,
            dtypes_summary={"float64": 6, "object": 2},
            overall_quality=DataQuality.GOOD,
            quality_score=0.82,
            columns=[],
            detected_patterns=[],
            missing_percentage=8.5,
            duplicate_percentage=2.1,
            key_insights=["Good data quality", "Manageable missing values"],
            recommendations=["Handle missing values", "Consider outlier analysis"]
        )
    else:
        sample_profile = None
    
    print("📊 샘플 분석 결과 해석:")
    
    # 결과 해석 실행
    interpretation = interpreter.interpret_results(
        session_id="test_session_001",
        results=sample_results,
        context={"domain": "business_analytics", "goal": "performance_analysis"},
        data_profile=sample_profile
    )
    
    print(f"✅ 해석 완료!")
    print(f"📈 신뢰도 점수: {interpretation.confidence_score:.1%}")
    print(f"🔍 인사이트 수: {len(interpretation.insights)}")
    print(f"🎯 추천사항 수: {len(interpretation.recommendations)}")
    
    print(f"\n💡 주요 인사이트:")
    for i, insight in enumerate(interpretation.insights[:3], 1):
        print(f"  {i}. {insight.title}")
        print(f"     → {insight.description}")
        print(f"     → 신뢰도: {insight.confidence:.1%}, 영향도: {insight.impact_score:.1%}")
    
    print(f"\n🎯 우선순위 추천사항:")
    for i, rec in enumerate(interpretation.recommendations[:3], 1):
        print(f"  {i}. [{rec.priority.value.upper()}] {rec.title}")
        print(f"     → {rec.description}")
        if rec.action_steps:
            print(f"     → 첫 번째 단계: {rec.action_steps[0]}")
    
    print(f"\n🚀 다음 단계:")
    for i, step in enumerate(interpretation.next_steps, 1):
        print(f"  {i}. {step}")
    
    # 보고서 생성 테스트
    print(f"\n📄 종합 보고서 생성 테스트:")
    report = interpreter.generate_comprehensive_report(interpretation)
    report_preview = report[:300] + "..." if len(report) > 300 else report
    print(f"보고서 미리보기:\n{report_preview}")
    print(f"전체 보고서 길이: {len(report):,} 문자")
    
    # 히스토리 테스트
    print(f"\n📋 해석 히스토리:")
    history = interpreter.get_interpretation_history(limit=3)
    for i, hist in enumerate(history, 1):
        print(f"  {i}. {hist.session_id} ({hist.interpretation_timestamp})")
        print(f"     → 인사이트: {len(hist.insights)}개, 추천사항: {len(hist.recommendations)}개")
    
    print(f"\n✅ Intelligent Result Interpreter 테스트 완료!")


if __name__ == "__main__":
    test_intelligent_result_interpreter() 