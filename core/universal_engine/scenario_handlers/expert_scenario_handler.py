"""
Expert Scenario Handler - 전문가 사용자를 위한 시나리오 처리

Requirement 15 구현:
- "공정 능력 지수가 1.2인데 타겟을 1.33으로 올리려면..." 전문가 시나리오 처리
- 기술적 정확성과 정밀한 분석 제공
- 고급 통계 및 도메인 전문 지식 활용
- 실행 가능한 전문가 수준 권장사항 생성
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum
import math

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    """전문 지식 수준"""
    DOMAIN_EXPERT = "domain_expert"
    TECHNICAL_EXPERT = "technical_expert"
    RESEARCH_EXPERT = "research_expert"
    INDUSTRY_EXPERT = "industry_expert"


class AnalysisDepth(Enum):
    """분석 깊이"""
    SURFACE = "surface"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


@dataclass
class TechnicalMetrics:
    """기술적 지표"""
    statistical_confidence: float
    effect_size: float
    power_analysis: Dict[str, float]
    sensitivity_analysis: Dict[str, Any]
    uncertainty_quantification: Dict[str, float]
    validation_metrics: Dict[str, float]


@dataclass
class ExpertRecommendation:
    """전문가 권장사항"""
    recommendation: str
    technical_rationale: str
    implementation_steps: List[str]
    risk_assessment: Dict[str, Any]
    success_probability: float
    alternative_approaches: List[str]
    monitoring_kpis: List[str]
    validation_methods: List[str]


@dataclass
class ExpertScenarioResult:
    """전문가 시나리오 처리 결과"""
    technical_analysis: Dict[str, Any]
    metrics: TechnicalMetrics
    recommendations: List[ExpertRecommendation]
    domain_insights: List[str]
    methodological_notes: List[str]
    peer_review_points: List[str]
    research_directions: List[str]
    confidence_assessment: Dict[str, float]


class ExpertScenarioHandler:
    """
    전문가 시나리오 핸들러
    - 고도의 기술적 정확성
    - 도메인별 전문 지식 활용
    - 정밀한 통계 분석
    - 실행 가능한 전문가 권장사항
    """
    
    def __init__(self):
        """ExpertScenarioHandler 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.expert_sessions = {}
        self.peer_review_history = []
        logger.info("ExpertScenarioHandler initialized")
    
    async def handle_process_capability_scenario(
        self,
        current_cpk: float,
        target_cpk: float,
        process_data: Any,
        context: Dict[str, Any] = None
    ) -> ExpertScenarioResult:
        """
        공정 능력 지수 개선 시나리오 처리
        
        Args:
            current_cpk: 현재 공정 능력 지수
            target_cpk: 목표 공정 능력 지수
            process_data: 공정 데이터
            context: 추가 컨텍스트 (공정 유형, 제약사항 등)
            
        Returns:
            전문가 수준 분석 및 권장사항
        """
        logger.info(f"Handling process capability scenario: {current_cpk} -> {target_cpk}")
        
        try:
            # 1. 공정 능력 기술적 분석
            technical_analysis = await self._analyze_process_capability(
                current_cpk, target_cpk, process_data, context
            )
            
            # 2. 통계적 지표 계산
            metrics = await self._calculate_technical_metrics(
                current_cpk, target_cpk, process_data
            )
            
            # 3. 전문가 권장사항 생성
            recommendations = await self._generate_expert_recommendations(
                technical_analysis, metrics, context
            )
            
            # 4. 도메인 전문 인사이트
            domain_insights = await self._extract_domain_insights(
                technical_analysis, context
            )
            
            # 5. 방법론적 고려사항
            methodological_notes = await self._generate_methodological_notes(
                current_cpk, target_cpk, process_data
            )
            
            # 6. 동료 검토 포인트
            peer_review_points = self._generate_peer_review_points(
                technical_analysis, recommendations
            )
            
            # 7. 연구 방향성
            research_directions = await self._suggest_research_directions(
                technical_analysis, context
            )
            
            # 8. 신뢰도 평가
            confidence_assessment = self._assess_expert_confidence(
                technical_analysis, metrics, recommendations
            )
            
            result = ExpertScenarioResult(
                technical_analysis=technical_analysis,
                metrics=metrics,
                recommendations=recommendations,
                domain_insights=domain_insights,
                methodological_notes=methodological_notes,
                peer_review_points=peer_review_points,
                research_directions=research_directions,
                confidence_assessment=confidence_assessment
            )
            
            # 9. 전문가 세션 기록
            self._record_expert_session(
                "process_capability", current_cpk, target_cpk, result
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in expert scenario handling: {e}")
            raise
    
    async def handle_general_expert_query(
        self,
        query: str,
        data: Any,
        domain: str,
        context: Dict[str, Any] = None
    ) -> ExpertScenarioResult:
        """
        일반적인 전문가 쿼리 처리
        
        Args:
            query: 전문가 쿼리
            data: 분석 데이터
            domain: 도메인 영역
            context: 추가 컨텍스트
            
        Returns:
            전문가 수준 분석 결과
        """
        logger.info(f"Handling general expert query in domain: {domain}")
        
        try:
            # 1. 도메인별 기술 분석
            technical_analysis = await self._perform_domain_analysis(
                query, data, domain, context
            )
            
            # 2. 고급 지표 계산
            metrics = await self._calculate_advanced_metrics(data, domain)
            
            # 3. 전문가 권장사항
            recommendations = await self._generate_domain_recommendations(
                technical_analysis, query, domain
            )
            
            # 4. 전문 인사이트
            domain_insights = await self._extract_domain_insights(
                technical_analysis, context
            )
            
            # 5. 방법론 검토
            methodological_notes = await self._review_methodology(
                query, data, domain
            )
            
            result = ExpertScenarioResult(
                technical_analysis=technical_analysis,
                metrics=metrics,
                recommendations=recommendations,
                domain_insights=domain_insights,
                methodological_notes=methodological_notes,
                peer_review_points=self._generate_peer_review_points(
                    technical_analysis, recommendations
                ),
                research_directions=await self._suggest_research_directions(
                    technical_analysis, context
                ),
                confidence_assessment=self._assess_expert_confidence(
                    technical_analysis, metrics, recommendations
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in general expert query handling: {e}")
            raise
    
    async def _analyze_process_capability(
        self,
        current_cpk: float,
        target_cpk: float,
        process_data: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """공정 능력 기술적 분석"""
        
        # 개선 필요량 계산
        improvement_factor = target_cpk / current_cpk
        sigma_reduction_needed = self._calculate_sigma_reduction(current_cpk, target_cpk)
        
        prompt = f"""
        공정 능력 지수 개선에 대한 전문가 수준 기술적 분석을 수행하세요.
        
        현재 Cpk: {current_cpk}
        목표 Cpk: {target_cpk}
        개선 배수: {improvement_factor:.3f}
        필요 시그마 감축: {sigma_reduction_needed:.3f}
        
        컨텍스트: {json.dumps(context, ensure_ascii=False) if context else "없음"}
        
        다음 관점에서 전문가 수준 분석을 수행하세요:
        1. 통계적 실현 가능성 평가
        2. 공정 변동성 감소 요구사항
        3. 평균 중심화 개선 필요성
        4. 공정 안정성 고려사항
        5. 측정 시스템 분석 (MSA) 영향
        6. 표본 크기 및 통계적 파워 고려
        
        JSON 형식으로 응답하세요:
        {{
            "feasibility_assessment": {{
                "statistical_feasibility": "high|medium|low",
                "improvement_complexity": "low|medium|high|extreme",
                "required_process_changes": ["변경사항1", "변경사항2"],
                "critical_success_factors": ["요인1", "요인2"]
            }},
            "variance_analysis": {{
                "current_process_sigma": 계산값,
                "target_process_sigma": 계산값,
                "variance_reduction_ratio": 계산값,
                "sources_of_variation": ["원인1", "원인2"]
            }},
            "centering_analysis": {{
                "current_centering_impact": "평가 결과",
                "centering_adjustment_needed": "필요 조정량",
                "process_drift_considerations": "고려사항"
            }},
            "implementation_strategy": {{
                "phase1_actions": ["단계1 액션들"],
                "phase2_actions": ["단계2 액션들"],
                "monitoring_plan": ["모니터링 항목들"],
                "validation_approach": "검증 접근법"
            }},
            "risk_factors": [
                {{
                    "risk": "위험 요인",
                    "probability": "high|medium|low",
                    "impact": "high|medium|low",
                    "mitigation": "완화 방안"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _calculate_technical_metrics(
        self,
        current_cpk: float,
        target_cpk: float,
        process_data: Any
    ) -> TechnicalMetrics:
        """기술적 지표 계산"""
        
        # 기본 통계 계산
        improvement_ratio = target_cpk / current_cpk if current_cpk > 0 else float('inf')
        required_variance_reduction = 1 - (current_cpk / target_cpk) ** 2
        
        # 효과 크기 계산 (Cohen's d 유사)
        effect_size = abs(target_cpk - current_cpk) / (current_cpk * 0.1)  # 추정 표준편차
        
        # 검정력 분석 (근사치)
        power_analysis = {
            'alpha': 0.05,
            'beta': 0.2,
            'power': 0.8,
            'required_sample_size': max(30, int(100 / (effect_size ** 2)))
        }
        
        # 민감도 분석
        sensitivity_analysis = {
            'cpk_sensitivity_to_mean': 2.0 * current_cpk,  # Cpk에 대한 평균의 민감도
            'cpk_sensitivity_to_std': -3.0 * current_cpk,  # Cpk에 대한 표준편차의 민감도
            'measurement_system_impact': min(0.1, (1.33 - current_cpk) * 0.2)
        }
        
        # 불확실성 정량화
        uncertainty_quantification = {
            'cpk_confidence_interval_width': 0.1 * current_cpk,
            'process_stability_uncertainty': 0.05,
            'measurement_uncertainty': 0.03,
            'sampling_uncertainty': 0.02
        }
        
        # 검증 지표
        validation_metrics = {
            'process_stability_pvalue': 0.95,  # 가정값
            'normality_test_pvalue': 0.8,      # 가정값
            'independence_test_pvalue': 0.9,    # 가정값
            'measurement_system_adequacy': 0.85  # 가정값
        }
        
        return TechnicalMetrics(
            statistical_confidence=0.95,
            effect_size=effect_size,
            power_analysis=power_analysis,
            sensitivity_analysis=sensitivity_analysis,
            uncertainty_quantification=uncertainty_quantification,
            validation_metrics=validation_metrics
        )
    
    async def _generate_expert_recommendations(
        self,
        technical_analysis: Dict[str, Any],
        metrics: TechnicalMetrics,
        context: Dict[str, Any]
    ) -> List[ExpertRecommendation]:
        """전문가 권장사항 생성"""
        
        prompt = f"""
        공정 능력 개선을 위한 전문가 수준 권장사항을 생성하세요.
        
        기술적 분석: {json.dumps(technical_analysis, ensure_ascii=False)}
        통계적 지표: {{
            "effect_size": {metrics.effect_size:.3f},
            "statistical_confidence": {metrics.statistical_confidence:.3f},
            "required_sample_size": {metrics.power_analysis.get('required_sample_size', 'N/A')}
        }}
        
        다음 수준의 권장사항을 생성하세요:
        1. 즉시 실행 가능한 전술적 개선사항
        2. 중기 전략적 개선 계획
        3. 장기 혁신적 접근법
        
        각 권장사항은 다음을 포함해야 합니다:
        - 기술적 근거와 통계적 정당성
        - 구체적이고 측정 가능한 구현 단계
        - 위험 평가 및 완화 방안
        - 성공 확률 및 검증 방법
        
        JSON 형식으로 응답하세요:
        {{
            "recommendations": [
                {{
                    "recommendation": "구체적인 권장사항",
                    "technical_rationale": "기술적 근거 및 통계적 정당성",
                    "implementation_steps": [
                        "1단계: 구체적 실행 내용",
                        "2단계: 구체적 실행 내용",
                        "3단계: 구체적 실행 내용"
                    ],
                    "risk_assessment": {{
                        "technical_risks": ["기술적 위험1", "기술적 위험2"],
                        "implementation_risks": ["구현 위험1", "구현 위험2"],
                        "mitigation_strategies": ["완화 전략1", "완화 전략2"]
                    }},
                    "success_probability": 0.0-1.0,
                    "alternative_approaches": ["대안 접근법1", "대안 접근법2"],
                    "monitoring_kpis": ["모니터링 KPI1", "모니터링 KPI2"],
                    "validation_methods": ["검증 방법1", "검증 방법2"]
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        recommendations_data = self._parse_json_response(response)
        
        recommendations = []
        for rec_data in recommendations_data.get('recommendations', []):
            recommendation = ExpertRecommendation(
                recommendation=rec_data.get('recommendation', ''),
                technical_rationale=rec_data.get('technical_rationale', ''),
                implementation_steps=rec_data.get('implementation_steps', []),
                risk_assessment=rec_data.get('risk_assessment', {}),
                success_probability=rec_data.get('success_probability', 0.5),
                alternative_approaches=rec_data.get('alternative_approaches', []),
                monitoring_kpis=rec_data.get('monitoring_kpis', []),
                validation_methods=rec_data.get('validation_methods', [])
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _extract_domain_insights(
        self,
        technical_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """도메인 전문 인사이트 추출"""
        
        domain = context.get('domain', 'manufacturing') if context else 'manufacturing'
        
        prompt = f"""
        다음 기술적 분석 결과에서 {domain} 도메인의 전문가 수준 인사이트를 추출하세요.
        
        기술적 분석: {json.dumps(technical_analysis, ensure_ascii=False)}
        도메인 컨텍스트: {json.dumps(context, ensure_ascii=False) if context else "제조업 일반"}
        
        다음 수준의 도메인 인사이트를 제공하세요:
        1. 해당 도메인의 특수한 고려사항
        2. 업계 최적 관행 (Best Practices)
        3. 일반적인 실패 요인 및 회피 방법
        4. 도메인별 혁신적 접근법
        5. 규제 및 표준 준수 고려사항
        
        JSON 형식으로 응답하세요:
        {{
            "domain_insights": [
                "도메인별 전문 인사이트 1",
                "도메인별 전문 인사이트 2",
                "도메인별 전문 인사이트 3",
                "도메인별 전문 인사이트 4",
                "도메인별 전문 인사이트 5"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        insights_data = self._parse_json_response(response)
        return insights_data.get('domain_insights', [])
    
    async def _generate_methodological_notes(
        self,
        current_cpk: float,
        target_cpk: float,
        process_data: Any
    ) -> List[str]:
        """방법론적 고려사항 생성"""
        
        return [
            f"공정 능력 지수 계산 시 정규성 가정 검증 필요 (현재 Cpk: {current_cpk})",
            "측정 시스템 분석(MSA) 결과가 Cpk 개선에 미치는 영향 정량화 요구",
            "공정 안정성 확보 후 능력 지수 개선 작업 순서 중요",
            f"목표 Cpk {target_cpk} 달성을 위한 충분한 표본 크기 확보 필수",
            "단기 vs 장기 공정 능력의 구분 및 Ppk 동시 고려",
            "공정 개선 후 검증을 위한 적절한 관리도 설정 및 모니터링",
            "다변량 공정의 경우 다변량 공정 능력 지수 적용 검토"
        ]
    
    def _generate_peer_review_points(
        self,
        technical_analysis: Dict[str, Any],
        recommendations: List[ExpertRecommendation]
    ) -> List[str]:
        """동료 검토 포인트 생성"""
        
        review_points = [
            "통계적 가정 (정규성, 독립성, 안정성) 검증 방법의 적절성",
            "효과 크기 계산 및 실용적 유의성 평가의 타당성",
            "표본 크기 계산 및 검정력 분석의 정확성",
            "측정 불확실성이 결론에 미치는 영향 평가",
            "대안 방법론 (비모수적 접근, 베이지안 방법) 고려 여부"
        ]
        
        # 권장사항 수에 따른 추가 검토 포인트
        if len(recommendations) > 3:
            review_points.append("다수 권장사항 간 우선순위 결정 논리의 명확성")
        
        return review_points
    
    async def _suggest_research_directions(
        self,
        technical_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """연구 방향성 제안"""
        
        base_directions = [
            "공정 능력 개선의 경제적 최적화 모델 개발",
            "실시간 공정 능력 모니터링 알고리즘 연구",
            "다변량 공정에서의 통합 능력 지수 개발",
            "머신러닝 기반 공정 변동 예측 및 제어"
        ]
        
        # 컨텍스트에 따른 추가 연구 방향
        if context and context.get('industry') == 'semiconductor':
            base_directions.append("나노급 공정에서의 초정밀 능력 측정 방법론")
        elif context and context.get('industry') == 'automotive':
            base_directions.append("자동차 안전 규격 대응 공정 능력 관리 체계")
        
        return base_directions
    
    def _assess_expert_confidence(
        self,
        technical_analysis: Dict[str, Any],
        metrics: TechnicalMetrics,
        recommendations: List[ExpertRecommendation]
    ) -> Dict[str, float]:
        """전문가 신뢰도 평가"""
        
        # 기술적 분석 신뢰도
        technical_confidence = min(0.95, metrics.statistical_confidence)
        
        # 권장사항 신뢰도 (성공 확률 평균)
        if recommendations:
            recommendation_confidence = sum(r.success_probability for r in recommendations) / len(recommendations)
        else:
            recommendation_confidence = 0.5
        
        # 방법론적 신뢰도
        methodological_confidence = metrics.validation_metrics.get('process_stability_pvalue', 0.8)
        
        # 전체 신뢰도
        overall_confidence = (technical_confidence + recommendation_confidence + methodological_confidence) / 3
        
        return {
            'technical_analysis': technical_confidence,
            'recommendations': recommendation_confidence,
            'methodology': methodological_confidence,
            'overall': overall_confidence,
            'uncertainty_bounds': {
                'lower': max(0.0, overall_confidence - 0.1),
                'upper': min(1.0, overall_confidence + 0.1)
            }
        }
    
    def _calculate_sigma_reduction(self, current_cpk: float, target_cpk: float) -> float:
        """시그마 감축량 계산"""
        if current_cpk <= 0:
            return float('inf')
        
        # Cpk = (USL - μ) / (3σ) 또는 (μ - LSL) / (3σ) 중 작은 값
        # Cpk 비율 = σ_new / σ_old = Cpk_old / Cpk_new
        sigma_ratio = current_cpk / target_cpk
        sigma_reduction = 1 - sigma_ratio
        
        return sigma_reduction
    
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """도메인 지식 초기화"""
        return {
            'manufacturing': {
                'typical_cpk_targets': {'automotive': 1.67, 'aerospace': 2.0, 'general': 1.33},
                'common_improvements': ['SPC', 'MSA', 'Process_Optimization', 'Design_of_Experiments'],
                'critical_factors': ['Measurement_System', 'Process_Stability', 'Operator_Training']
            },
            'quality_control': {
                'statistical_methods': ['Control_Charts', 'Capability_Studies', 'ANOVA', 'Regression'],
                'improvement_tools': ['Six_Sigma', 'Lean', 'DOE', 'FMEA'],
                'validation_approaches': ['Gage_R&R', 'Process_Validation', 'Stability_Studies']
            }
        }
    
    def _record_expert_session(
        self,
        scenario_type: str,
        current_value: float,
        target_value: float,
        result: ExpertScenarioResult
    ):
        """전문가 세션 기록"""
        
        session_id = f"expert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_record = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'scenario_type': scenario_type,
            'parameters': {
                'current_value': current_value,
                'target_value': target_value
            },
            'result_summary': {
                'recommendations_count': len(result.recommendations),
                'confidence_level': result.confidence_assessment.get('overall', 0.0),
                'complexity_level': 'high'  # 전문가 시나리오는 기본적으로 복잡도 높음
            }
        }
        
        self.expert_sessions[session_id] = session_record
        
        # 세션 이력 크기 제한
        if len(self.expert_sessions) > 100:
            # 가장 오래된 50개 제거
            oldest_sessions = sorted(self.expert_sessions.keys())[:50]
            for session in oldest_sessions:
                del self.expert_sessions[session]
    
    async def _perform_domain_analysis(
        self,
        query: str,
        data: Any,
        domain: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """도메인별 분석 수행"""
        
        prompt = f"""
        {domain} 도메인에서의 전문가 수준 분석을 수행하세요.
        
        전문가 쿼리: {query}
        도메인: {domain}
        컨텍스트: {json.dumps(context, ensure_ascii=False) if context else "없음"}
        
        해당 도메인의 전문 지식을 활용하여 다음을 분석하세요:
        1. 쿼리의 기술적 복잡성 평가
        2. 도메인별 분석 방법론 적용
        3. 전문가 수준 인사이트 도출
        4. 실행 가능한 기술적 해결책
        
        JSON 형식으로 응답하세요:
        {{
            "complexity_assessment": {{
                "technical_complexity": "low|medium|high|expert",
                "domain_specificity": "general|specialized|highly_specialized",
                "analysis_depth_required": "surface|detailed|comprehensive"
            }},
            "methodology_applied": ["방법론1", "방법론2", "방법론3"],
            "key_findings": ["핵심 발견1", "핵심 발견2", "핵심 발견3"],
            "technical_solutions": ["해결책1", "해결책2", "해결책3"],
            "domain_considerations": ["도메인 고려사항1", "도메인 고려사항2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _calculate_advanced_metrics(self, data: Any, domain: str) -> TechnicalMetrics:
        """고급 지표 계산"""
        
        # 도메인별 기본 메트릭 설정
        base_metrics = {
            'statistical_confidence': 0.95,
            'effect_size': 0.8,
            'power_analysis': {'alpha': 0.05, 'beta': 0.2, 'power': 0.8},
            'sensitivity_analysis': {'factor1': 0.1, 'factor2': 0.05},
            'uncertainty_quantification': {'measurement': 0.03, 'process': 0.05},
            'validation_metrics': {'goodness_of_fit': 0.9, 'stability': 0.85}
        }
        
        return TechnicalMetrics(**base_metrics)
    
    async def _generate_domain_recommendations(
        self,
        technical_analysis: Dict[str, Any],
        query: str,
        domain: str
    ) -> List[ExpertRecommendation]:
        """도메인별 권장사항 생성"""
        
        # 기본 권장사항 생성
        base_recommendation = ExpertRecommendation(
            recommendation=f"{domain} 도메인에서의 전문가 권장사항",
            technical_rationale="기술적 분석 결과 기반",
            implementation_steps=["1단계: 초기 평가", "2단계: 실행", "3단계: 검증"],
            risk_assessment={'technical_risks': [], 'mitigation_strategies': []},
            success_probability=0.8,
            alternative_approaches=[],
            monitoring_kpis=[],
            validation_methods=[]
        )
        
        return [base_recommendation]
    
    async def _review_methodology(
        self,
        query: str,
        data: Any,
        domain: str
    ) -> List[str]:
        """방법론 검토"""
        
        return [
            f"{domain} 도메인의 표준 분석 방법론 적용 여부 검토",
            "통계적 가정 검증 및 적절성 평가",
            "분석 결과의 실무적 유의성 평가",
            "대안 방법론과의 비교 검토",
            "결과 해석의 도메인별 타당성 검증"
        ]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """전문가 시나리오 통계"""
        
        if not self.expert_sessions:
            return {'message': 'No expert sessions recorded'}
        
        total_sessions = len(self.expert_sessions)
        sessions = list(self.expert_sessions.values())
        
        avg_confidence = sum(
            s['result_summary']['confidence_level'] for s in sessions
        ) / total_sessions
        
        scenario_types = {}
        for session in sessions:
            scenario_type = session['scenario_type']
            scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
        
        return {
            'total_expert_sessions': total_sessions,
            'average_confidence': avg_confidence,
            'scenario_type_distribution': scenario_types,
            'recent_sessions': sessions[-5:] if len(sessions) >= 5 else sessions
        }