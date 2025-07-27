"""
멀티 에이전트 추천사항 생성 시스템

이 모듈은 분석 결과와 인사이트를 바탕으로 실행 가능한 추천사항을 생성하고,
우선순위 및 예상 임팩트를 평가하여 다음 단계 분석을 제안하는 시스템을 제공합니다.

주요 기능:
- 분석 결과 기반 액션 아이템 도출
- 우선순위 및 예상 임팩트 평가
- 다음 단계 분석 제안
- 실행 가능성 및 리소스 요구사항 평가
"""

import json
import logging
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .insight_generator import InsightAnalysis, Insight, InsightType, InsightPriority
from .result_integrator import IntegrationResult
from .agent_result_collector import AgentResult

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """추천사항 유형"""
    IMMEDIATE_ACTION = "immediate_action"        # 즉시 실행 필요
    DATA_COLLECTION = "data_collection"          # 추가 데이터 수집
    FURTHER_ANALYSIS = "further_analysis"        # 심화 분석
    PROCESS_IMPROVEMENT = "process_improvement"   # 프로세스 개선
    MONITORING = "monitoring"                     # 모니터링 설정
    VALIDATION = "validation"                     # 검증 및 확인
    OPTIMIZATION = "optimization"                 # 최적화
    REPORTING = "reporting"                       # 보고 및 공유

class Priority(Enum):
    """우선순위"""
    URGENT = "urgent"        # 긴급 (24시간 내)
    HIGH = "high"           # 높음 (1주일 내)
    MEDIUM = "medium"       # 보통 (1개월 내)
    LOW = "low"            # 낮음 (분기 내)
    FUTURE = "future"       # 향후 고려

class Feasibility(Enum):
    """실행 가능성"""
    EASY = "easy"           # 쉬움 (즉시 실행 가능)
    MODERATE = "moderate"   # 보통 (일부 리소스 필요)
    COMPLEX = "complex"     # 복잡 (상당한 리소스 필요)
    DIFFICULT = "difficult" # 어려움 (대규모 투자 필요)

@dataclass
class Recommendation:
    """개별 추천사항"""
    recommendation_id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    priority: Priority
    
    # 임팩트 및 실행성
    expected_impact: float  # 0.0 ~ 1.0
    feasibility: Feasibility
    estimated_effort: str  # "1일", "1주", "1개월" 등
    required_resources: List[str] = field(default_factory=list)
    
    # 관련 정보
    related_insights: List[str] = field(default_factory=list)  # Insight ID들
    supporting_evidence: List[str] = field(default_factory=list)
    
    # 실행 계획
    action_steps: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    
    # 메타데이터
    target_stakeholders: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    # 다음 단계
    follow_up_recommendations: List[str] = field(default_factory=list)

@dataclass
class RecommendationPlan:
    """종합 추천 계획"""
    session_id: str
    total_recommendations: int
    recommendations: List[Recommendation] = field(default_factory=list)
    
    # 분류별 통계
    by_type: Dict[RecommendationType, int] = field(default_factory=dict)
    by_priority: Dict[Priority, int] = field(default_factory=dict)
    by_feasibility: Dict[Feasibility, int] = field(default_factory=dict)
    
    # 실행 계획
    immediate_actions: List[Recommendation] = field(default_factory=list)
    short_term_plan: List[Recommendation] = field(default_factory=list)
    long_term_plan: List[Recommendation] = field(default_factory=list)
    
    # 요약 정보
    overall_impact_score: float = 0.0
    total_estimated_effort: str = ""
    key_success_factors: List[str] = field(default_factory=list)
    
    # 생성 정보
    generation_time: datetime = field(default_factory=datetime.now)
    processing_notes: List[str] = field(default_factory=list)

class RecommendationGenerator:
    """멀티 에이전트 추천사항 생성기"""
    
    def __init__(self):
        # 우선순위 가중치
        self.priority_weights = {
            InsightPriority.CRITICAL: 1.0,
            InsightPriority.HIGH: 0.8,
            InsightPriority.MEDIUM: 0.6,
            InsightPriority.LOW: 0.4,
            InsightPriority.INFORMATIONAL: 0.2
        }
        
        # 임팩트 계산 가중치
        self.impact_weights = {
            'insight_priority': 0.4,
            'insight_confidence': 0.3,
            'data_quality': 0.2,
            'business_relevance': 0.1
        }
        
        # 키워드 기반 추천 패턴
        self.recommendation_patterns = {
            'trend': {
                'type': RecommendationType.MONITORING,
                'actions': ['트렌드 지속성 모니터링', '트렌드 원인 분석', '예측 모델 구축']
            },
            'anomaly': {
                'type': RecommendationType.IMMEDIATE_ACTION,
                'actions': ['이상치 원인 조사', '데이터 품질 검증', '보정 조치 실행']
            },
            'correlation': {
                'type': RecommendationType.FURTHER_ANALYSIS,
                'actions': ['인과관계 분석', '추가 변수 조사', '실험 설계']
            },
            'pattern': {
                'type': RecommendationType.PROCESS_IMPROVEMENT,
                'actions': ['패턴 활용 방안 모색', '프로세스 최적화', '자동화 검토']
            },
            'quality': {
                'type': RecommendationType.VALIDATION,
                'actions': ['데이터 품질 개선', '검증 프로세스 강화', '품질 지표 설정']
            }
        }
    
    def generate_recommendations(self,
                               insight_analysis: InsightAnalysis,
                               integration_result: IntegrationResult,
                               agent_results: Dict[str, AgentResult] = None) -> RecommendationPlan:
        """종합 추천사항 생성"""
        
        logger.info(f"💡 추천사항 생성 시작 - 세션 {insight_analysis.session_id}, "
                   f"인사이트 {insight_analysis.total_insights}개")
        
        plan = RecommendationPlan(
            session_id=insight_analysis.session_id,
            total_recommendations=0
        )
        
        try:
            recommendations = []
            
            # 1. 인사이트 기반 추천사항 생성
            insight_recommendations = self._generate_insight_based_recommendations(
                insight_analysis.insights
            )
            recommendations.extend(insight_recommendations)
            
            # 2. 통합 품질 기반 추천사항
            quality_recommendations = self._generate_quality_based_recommendations(
                integration_result
            )
            recommendations.extend(quality_recommendations)
            
            # 3. 데이터 분석 개선 추천사항
            if agent_results:
                improvement_recommendations = self._generate_improvement_recommendations(
                    agent_results, integration_result
                )
                recommendations.extend(improvement_recommendations)
            
            # 4. 일반적인 분석 프로세스 추천사항
            process_recommendations = self._generate_process_recommendations(
                insight_analysis, integration_result
            )
            recommendations.extend(process_recommendations)
            
            # 5. 추천사항 우선순위 결정 및 정제
            prioritized_recommendations = self._prioritize_and_refine_recommendations(
                recommendations, insight_analysis
            )
            
            plan.recommendations = prioritized_recommendations
            plan.total_recommendations = len(prioritized_recommendations)
            
            # 6. 실행 계획 구성
            self._organize_execution_plan(plan)
            
            # 7. 분석 결과 집계
            self._aggregate_plan_analysis(plan)
            
            logger.info(f"✅ 추천사항 생성 완료 - {plan.total_recommendations}개 추천사항, "
                       f"종합 임팩트: {plan.overall_impact_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 추천사항 생성 중 오류: {e}")
            plan.processing_notes.append(f"생성 오류: {str(e)}")
        
        return plan
    
    def _generate_insight_based_recommendations(self, insights: List[Insight]) -> List[Recommendation]:
        """인사이트 기반 추천사항 생성"""
        
        recommendations = []
        
        for insight in insights:
            try:
                # 인사이트 유형에 따른 기본 추천사항
                base_recommendations = self._get_base_recommendations_for_insight(insight)
                
                for i, base_rec in enumerate(base_recommendations):
                    recommendation = Recommendation(
                        recommendation_id=f"insight_{insight.insight_id}_{i}",
                        title=base_rec['title'],
                        description=base_rec['description'],
                        recommendation_type=base_rec['type'],
                        priority=self._determine_priority_from_insight(insight),
                        expected_impact=self._calculate_expected_impact(insight),
                        feasibility=base_rec.get('feasibility', Feasibility.MODERATE),
                        estimated_effort=base_rec.get('effort', '1주'),
                        required_resources=base_rec.get('resources', []),
                        related_insights=[insight.insight_id],
                        supporting_evidence=insight.evidence.copy(),
                        action_steps=base_rec.get('actions', []),
                        success_metrics=base_rec.get('metrics', []),
                        potential_risks=base_rec.get('risks', []),
                        target_stakeholders=['데이터 분석팀', '의사결정자']
                    )
                    
                    recommendations.append(recommendation)
            
            except Exception as e:
                logger.warning(f"인사이트 {insight.insight_id} 추천사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _generate_quality_based_recommendations(self, integration_result: IntegrationResult) -> List[Recommendation]:
        """통합 품질 기반 추천사항 생성"""
        
        recommendations = []
        
        try:
            # 통합 품질이 낮은 경우
            if integration_result.integration_quality < 0.7:
                recommendation = Recommendation(
                    recommendation_id="quality_improvement",
                    title="분석 품질 개선",
                    description=f"현재 통합 품질이 {integration_result.integration_quality:.1%}로 개선이 필요합니다",
                    recommendation_type=RecommendationType.PROCESS_IMPROVEMENT,
                    priority=Priority.HIGH,
                    expected_impact=0.8,
                    feasibility=Feasibility.MODERATE,
                    estimated_effort="2주",
                    required_resources=['데이터 품질 전문가', '분석 도구 개선'],
                    action_steps=[
                        "데이터 품질 기준 재정의",
                        "에이전트 성능 최적화",
                        "검증 프로세스 강화"
                    ],
                    success_metrics=["통합 품질 점수 80% 이상 달성"],
                    potential_risks=["초기 분석 지연", "리소스 요구 증가"]
                )
                recommendations.append(recommendation)
            
            # 커버리지가 낮은 경우
            if integration_result.coverage_score < 0.8:
                recommendation = Recommendation(
                    recommendation_id="coverage_improvement",
                    title="분석 커버리지 확대",
                    description=f"분석 커버리지가 {integration_result.coverage_score:.1%}로 확대가 필요합니다",
                    recommendation_type=RecommendationType.DATA_COLLECTION,
                    priority=Priority.MEDIUM,
                    expected_impact=0.6,
                    feasibility=Feasibility.MODERATE,
                    estimated_effort="1주",
                    required_resources=['추가 데이터 소스', '에이전트 설정 조정'],
                    action_steps=[
                        "비활성 에이전트 원인 조사",
                        "데이터 소스 확장",
                        "분석 범위 재정의"
                    ],
                    success_metrics=["커버리지 90% 이상 달성"]
                )
                recommendations.append(recommendation)
            
            # 신뢰도가 낮은 경우
            if integration_result.overall_confidence < 0.6:
                recommendation = Recommendation(
                    recommendation_id="confidence_enhancement",
                    title="분석 신뢰도 향상",
                    description=f"분석 신뢰도가 {integration_result.overall_confidence:.1%}로 향상이 필요합니다",
                    recommendation_type=RecommendationType.VALIDATION,
                    priority=Priority.HIGH,
                    expected_impact=0.7,
                    feasibility=Feasibility.COMPLEX,
                    estimated_effort="3주",
                    required_resources=['데이터 검증 도구', '전문가 검토'],
                    action_steps=[
                        "데이터 검증 강화",
                        "분석 방법론 개선",
                        "전문가 리뷰 프로세스 도입"
                    ],
                    success_metrics=["신뢰도 80% 이상 달성"]
                )
                recommendations.append(recommendation)
        
        except Exception as e:
            logger.warning(f"품질 기반 추천사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _generate_improvement_recommendations(self,
                                           agent_results: Dict[str, AgentResult],
                                           integration_result: IntegrationResult) -> List[Recommendation]:
        """데이터 분석 개선 추천사항 생성"""
        
        recommendations = []
        
        try:
            # 에이전트 성능 분석
            performance_issues = self._analyze_agent_performance(agent_results)
            
            if performance_issues['slow_agents']:
                recommendation = Recommendation(
                    recommendation_id="performance_optimization",
                    title="느린 에이전트 성능 최적화",
                    description=f"{len(performance_issues['slow_agents'])}개 에이전트의 성능 개선 필요",
                    recommendation_type=RecommendationType.OPTIMIZATION,
                    priority=Priority.MEDIUM,
                    expected_impact=0.5,
                    feasibility=Feasibility.MODERATE,
                    estimated_effort="1주",
                    required_resources=['성능 분석 도구', '시스템 관리자'],
                    action_steps=[
                        "느린 에이전트 식별 및 분석",
                        "병목 지점 해결",
                        "리소스 할당 최적화"
                    ],
                    success_metrics=["평균 실행 시간 30% 단축"]
                )
                recommendations.append(recommendation)
            
            if performance_issues['error_prone_agents']:
                recommendation = Recommendation(
                    recommendation_id="error_reduction",
                    title="에러 발생 에이전트 안정화",
                    description=f"{len(performance_issues['error_prone_agents'])}개 에이전트의 안정성 개선 필요",
                    recommendation_type=RecommendationType.IMMEDIATE_ACTION,
                    priority=Priority.HIGH,
                    expected_impact=0.8,
                    feasibility=Feasibility.MODERATE,
                    estimated_effort="3일",
                    required_resources=['개발팀', '로그 분석 도구'],
                    action_steps=[
                        "에러 패턴 분석",
                        "에러 처리 로직 개선",
                        "모니터링 시스템 강화"
                    ],
                    success_metrics=["에러 발생률 5% 이하 달성"]
                )
                recommendations.append(recommendation)
            
            # 데이터 품질 개선
            if self._has_data_quality_issues(agent_results):
                recommendation = Recommendation(
                    recommendation_id="data_quality_enhancement",
                    title="데이터 품질 개선 프로그램",
                    description="일관되지 않은 데이터 품질로 인한 분석 신뢰도 저하 해결",
                    recommendation_type=RecommendationType.PROCESS_IMPROVEMENT,
                    priority=Priority.HIGH,
                    expected_impact=0.9,
                    feasibility=Feasibility.COMPLEX,
                    estimated_effort="1개월",
                    required_resources=['데이터 엔지니어', '품질 관리 도구'],
                    action_steps=[
                        "데이터 품질 기준 수립",
                        "자동 품질 검증 시스템 구축",
                        "데이터 정제 프로세스 개선"
                    ],
                    success_metrics=["데이터 품질 점수 95% 이상"]
                )
                recommendations.append(recommendation)
        
        except Exception as e:
            logger.warning(f"개선 추천사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _generate_process_recommendations(self,
                                        insight_analysis: InsightAnalysis,
                                        integration_result: IntegrationResult) -> List[Recommendation]:
        """일반적인 분석 프로세스 추천사항 생성"""
        
        recommendations = []
        
        try:
            # 정기적인 분석 자동화
            if insight_analysis.total_insights > 5:
                recommendation = Recommendation(
                    recommendation_id="analysis_automation",
                    title="정기 분석 자동화 시스템 구축",
                    description="반복적인 분석 작업의 자동화를 통한 효율성 향상",
                    recommendation_type=RecommendationType.OPTIMIZATION,
                    priority=Priority.MEDIUM,
                    expected_impact=0.7,
                    feasibility=Feasibility.COMPLEX,
                    estimated_effort="6주",
                    required_resources=['자동화 도구', '개발 리소스'],
                    action_steps=[
                        "반복 분석 패턴 식별",
                        "자동화 워크플로우 설계",
                        "스케줄링 시스템 구축"
                    ],
                    success_metrics=["분석 시간 50% 단축", "분석 주기 정규화"]
                )
                recommendations.append(recommendation)
            
            # 결과 모니터링 시스템
            recommendation = Recommendation(
                recommendation_id="monitoring_system",
                title="분석 결과 모니터링 대시보드 구축",
                description="분석 결과의 지속적인 모니터링과 알림 시스템 구축",
                recommendation_type=RecommendationType.MONITORING,
                priority=Priority.MEDIUM,
                expected_impact=0.6,
                feasibility=Feasibility.MODERATE,
                estimated_effort="2주",
                required_resources=['대시보드 도구', 'UI/UX 디자이너'],
                action_steps=[
                    "핵심 지표 정의",
                    "대시보드 설계 및 구현",
                    "알림 시스템 설정"
                ],
                success_metrics=["실시간 모니터링 가능", "이상 상황 자동 감지"]
            )
            recommendations.append(recommendation)
            
            # 결과 공유 및 협업
            recommendation = Recommendation(
                recommendation_id="collaboration_platform",
                title="분석 결과 공유 플랫폼 구축",
                description="팀 간 분석 결과 공유와 협업을 위한 플랫폼 구축",
                recommendation_type=RecommendationType.REPORTING,
                priority=Priority.LOW,
                expected_impact=0.4,
                feasibility=Feasibility.MODERATE,
                estimated_effort="3주",
                required_resources=['협업 도구', '문서화 시스템'],
                action_steps=[
                    "공유 요구사항 수집",
                    "플랫폼 설계",
                    "사용자 교육 프로그램 실시"
                ],
                success_metrics=["팀 간 정보 공유 증가", "의사결정 속도 향상"]
            )
            recommendations.append(recommendation)
        
        except Exception as e:
            logger.warning(f"프로세스 추천사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _prioritize_and_refine_recommendations(self,
                                             recommendations: List[Recommendation],
                                             insight_analysis: InsightAnalysis) -> List[Recommendation]:
        """추천사항 우선순위 결정 및 정제"""
        
        # 중복 제거
        unique_recommendations = self._remove_duplicate_recommendations(recommendations)
        
        # 우선순위 점수 계산
        for rec in unique_recommendations:
            rec.priority_score = self._calculate_priority_score(rec, insight_analysis)
        
        # 우선순위 순으로 정렬
        prioritized = sorted(unique_recommendations, 
                           key=lambda r: r.priority_score, 
                           reverse=True)
        
        # 상위 추천사항으로 제한 (최대 15개)
        return prioritized[:15]
    
    def _organize_execution_plan(self, plan: RecommendationPlan):
        """실행 계획 구성"""
        
        for rec in plan.recommendations:
            if rec.priority == Priority.URGENT:
                plan.immediate_actions.append(rec)
            elif rec.priority in [Priority.HIGH, Priority.MEDIUM]:
                plan.short_term_plan.append(rec)
            else:
                plan.long_term_plan.append(rec)
        
        # 시간 순서대로 정렬
        plan.immediate_actions.sort(key=lambda r: r.expected_impact, reverse=True)
        plan.short_term_plan.sort(key=lambda r: r.expected_impact, reverse=True)
        plan.long_term_plan.sort(key=lambda r: r.expected_impact, reverse=True)
    
    def _aggregate_plan_analysis(self, plan: RecommendationPlan):
        """계획 분석 결과 집계"""
        
        if not plan.recommendations:
            return
        
        # 유형별/우선순위별/실행성별 집계
        for rec in plan.recommendations:
            plan.by_type[rec.recommendation_type] = plan.by_type.get(rec.recommendation_type, 0) + 1
            plan.by_priority[rec.priority] = plan.by_priority.get(rec.priority, 0) + 1
            plan.by_feasibility[rec.feasibility] = plan.by_feasibility.get(rec.feasibility, 0) + 1
        
        # 전체 임팩트 점수
        impact_scores = [rec.expected_impact for rec in plan.recommendations]
        plan.overall_impact_score = sum(impact_scores) / len(impact_scores)
        
        # 총 예상 노력
        effort_mapping = {
            '1일': 1, '3일': 3, '1주': 7, '2주': 14, '3주': 21, '1개월': 30, '6주': 42
        }
        
        total_days = 0
        for rec in plan.recommendations:
            days = effort_mapping.get(rec.estimated_effort, 7)  # 기본 1주
            total_days += days
        
        if total_days <= 7:
            plan.total_estimated_effort = f"{total_days}일"
        elif total_days <= 30:
            plan.total_estimated_effort = f"{total_days//7}주"
        else:
            plan.total_estimated_effort = f"{total_days//30}개월"
        
        # 핵심 성공 요인
        plan.key_success_factors = self._identify_key_success_factors(plan.recommendations)
    
    def _get_base_recommendations_for_insight(self, insight: Insight) -> List[Dict[str, Any]]:
        """인사이트 유형별 기본 추천사항"""
        
        insight_type_lower = insight.insight_type.value.lower()
        
        # 키워드 매칭으로 패턴 찾기
        for pattern, config in self.recommendation_patterns.items():
            if pattern in insight_type_lower or pattern in insight.description.lower():
                return [{
                    'title': f"{insight.title}에 대한 {pattern} 대응",
                    'description': f"{insight.description}에 따른 권장 조치",
                    'type': config['type'],
                    'actions': config['actions'],
                    'feasibility': Feasibility.MODERATE,
                    'effort': '1주',
                    'resources': ['분석팀'],
                    'metrics': [f"{pattern} 관련 지표 개선"],
                    'risks': ['분석 결과 불확실성']
                }]
        
        # 기본 추천사항
        return [{
            'title': f"{insight.title} 후속 조치",
            'description': f"{insight.description}에 대한 추가 분석 및 조치",
            'type': RecommendationType.FURTHER_ANALYSIS,
            'actions': ['상세 분석 수행', '결과 검증', '대응 방안 수립'],
            'feasibility': Feasibility.MODERATE,
            'effort': '1주',
            'resources': ['분석팀'],
            'metrics': ['분석 완료', '검증 통과'],
            'risks': ['시간 지연']
        }]
    
    def _determine_priority_from_insight(self, insight: Insight) -> Priority:
        """인사이트에서 우선순위 결정"""
        
        if insight.priority == InsightPriority.CRITICAL:
            return Priority.URGENT
        elif insight.priority == InsightPriority.HIGH:
            return Priority.HIGH
        elif insight.priority == InsightPriority.MEDIUM:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def _calculate_expected_impact(self, insight: Insight) -> float:
        """예상 임팩트 계산"""
        
        # 기본 임팩트는 인사이트의 임팩트 점수
        base_impact = insight.impact_score
        
        # 우선순위에 따른 가중치
        priority_weight = self.priority_weights.get(insight.priority, 0.5)
        
        # 신뢰도에 따른 가중치
        confidence_weight = insight.confidence
        
        # 최종 임팩트 계산
        expected_impact = (
            base_impact * 0.5 +
            priority_weight * 0.3 +
            confidence_weight * 0.2
        )
        
        return min(1.0, expected_impact)
    
    def _analyze_agent_performance(self, agent_results: Dict[str, AgentResult]) -> Dict[str, List[str]]:
        """에이전트 성능 분석"""
        
        performance_issues = {
            'slow_agents': [],
            'error_prone_agents': [],
            'low_quality_agents': []
        }
        
        execution_times = [result.execution_duration for result in agent_results.values()]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        for agent_id, result in agent_results.items():
            # 느린 에이전트 (평균의 2배 이상)
            if result.execution_duration > avg_time * 2:
                performance_issues['slow_agents'].append(agent_id)
            
            # 에러 발생 에이전트
            if result.error_message:
                performance_issues['error_prone_agents'].append(agent_id)
            
            # 낮은 품질 (아티팩트 없거나 짧은 텍스트)
            if not result.artifacts and len(result.processed_text) < 100:
                performance_issues['low_quality_agents'].append(agent_id)
        
        return performance_issues
    
    def _has_data_quality_issues(self, agent_results: Dict[str, AgentResult]) -> bool:
        """데이터 품질 이슈 감지"""
        
        total_results = len(agent_results)
        quality_issues = 0
        
        for result in agent_results.values():
            if (result.error_message or 
                not result.processed_text or 
                len(result.processed_text) < 50):
                quality_issues += 1
        
        # 30% 이상이 품질 이슈가 있으면 문제로 판단
        return quality_issues / total_results > 0.3
    
    def _remove_duplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """중복 추천사항 제거"""
        
        unique_recommendations = []
        seen_titles = set()
        
        for rec in recommendations:
            # 제목 기반 중복 체크 (간단한 방식)
            title_key = rec.title.lower().strip()
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_priority_score(self, 
                                recommendation: Recommendation, 
                                insight_analysis: InsightAnalysis) -> float:
        """우선순위 점수 계산"""
        
        # 우선순위 기본 점수
        priority_scores = {
            Priority.URGENT: 5.0,
            Priority.HIGH: 4.0,
            Priority.MEDIUM: 3.0,
            Priority.LOW: 2.0,
            Priority.FUTURE: 1.0
        }
        
        base_score = priority_scores.get(recommendation.priority, 2.0)
        
        # 예상 임팩트 가중치
        impact_weight = recommendation.expected_impact
        
        # 실행 가능성 가중치
        feasibility_weights = {
            Feasibility.EASY: 1.0,
            Feasibility.MODERATE: 0.8,
            Feasibility.COMPLEX: 0.6,
            Feasibility.DIFFICULT: 0.4
        }
        
        feasibility_weight = feasibility_weights.get(recommendation.feasibility, 0.5)
        
        # 최종 점수 계산
        priority_score = (
            base_score * 0.4 +
            impact_weight * 5.0 * 0.4 +
            feasibility_weight * 5.0 * 0.2
        )
        
        return priority_score
    
    def _identify_key_success_factors(self, recommendations: List[Recommendation]) -> List[str]:
        """핵심 성공 요인 식별"""
        
        success_factors = []
        
        # 자주 언급되는 리소스
        all_resources = []
        for rec in recommendations:
            all_resources.extend(rec.required_resources)
        
        resource_counts = Counter(all_resources)
        common_resources = [resource for resource, count in resource_counts.most_common(3)]
        
        if common_resources:
            success_factors.append(f"핵심 리소스 확보: {', '.join(common_resources)}")
        
        # 높은 임팩트 추천사항들의 공통점
        high_impact_recs = [rec for rec in recommendations if rec.expected_impact > 0.7]
        
        if len(high_impact_recs) > len(recommendations) / 2:
            success_factors.append("높은 임팩트 활동에 집중")
        
        # 실행 가능성 확보
        difficult_recs = [rec for rec in recommendations if rec.feasibility == Feasibility.DIFFICULT]
        
        if difficult_recs:
            success_factors.append("복잡한 프로젝트 관리 역량")
        
        # 기본 성공 요인
        success_factors.extend([
            "체계적인 실행 계획",
            "지속적인 모니터링",
            "팀 간 원활한 소통"
        ])
        
        return success_factors[:5]  # 최대 5개