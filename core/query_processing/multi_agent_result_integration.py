"""
Multi-Agent Result Integration

이 모듈은 A2A Agent Execution Orchestrator에서 수집된 다중 에이전트 결과를 
통합하고 분석하여 전체론적인 인사이트를 생성합니다.

주요 기능:
- 에이전트 결과 통합 및 정규화
- 교차 검증 및 일관성 분석
- 인사이트 추출 및 패턴 발견
- 결과 품질 평가
- 종합 분석 보고서 생성
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from core.llm_factory import create_llm_instance
from .a2a_agent_execution_orchestrator import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


class IntegrationStrategy(Enum):
    """통합 전략"""
    SEQUENTIAL = "sequential"      # 순차 통합
    HIERARCHICAL = "hierarchical"  # 계층적 통합
    CONSENSUS = "consensus"        # 합의 기반 통합
    WEIGHTED = "weighted"          # 가중치 기반 통합


class ResultType(Enum):
    """결과 타입"""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    STATISTICAL = "statistical"
    PREDICTION = "prediction"
    INSIGHT = "insight"
    RECOMMENDATION = "recommendation"
    REPORT = "report"


class QualityMetric(Enum):
    """품질 지표"""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"


@dataclass
class AgentResult:
    """에이전트 결과"""
    agent_name: str
    agent_type: str
    result_type: ResultType
    content: Dict[str, Any]
    confidence: float
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)


@dataclass
class CrossValidationResult:
    """교차 검증 결과"""
    consistency_score: float
    conflicting_findings: List[Dict[str, Any]]
    supporting_evidence: List[Dict[str, Any]]
    validation_notes: str
    confidence_adjustment: float


@dataclass
class IntegratedInsight:
    """통합 인사이트"""
    insight_type: str
    content: str
    confidence: float
    supporting_agents: List[str]
    evidence_strength: float
    actionable_items: List[str]
    priority: int


@dataclass
class IntegrationResult:
    """통합 결과"""
    integration_id: str
    strategy: IntegrationStrategy
    agent_results: List[AgentResult]
    cross_validation: CrossValidationResult
    integrated_insights: List[IntegratedInsight]
    quality_assessment: Dict[QualityMetric, float]
    synthesis_report: str
    recommendations: List[str]
    confidence_score: float
    integration_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentResultIntegrator:
    """다중 에이전트 결과 통합기"""
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.integration_history: List[IntegrationResult] = []
        logger.info("MultiAgentResultIntegrator initialized")
    
    async def integrate_results(
        self,
        execution_result: ExecutionResult,
        integration_strategy: IntegrationStrategy = IntegrationStrategy.HIERARCHICAL,
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        다중 에이전트 결과 통합
        
        Args:
            execution_result: A2A 에이전트 실행 결과
            integration_strategy: 통합 전략
            context: 통합 컨텍스트
            
        Returns:
            IntegrationResult: 통합 결과
        """
        start_time = time.time()
        integration_id = f"integration_{int(start_time)}"
        
        logger.info(f"🔄 Starting result integration: {integration_id}")
        
        try:
            # 1. 에이전트 결과 정규화
            normalized_results = await self._normalize_agent_results(execution_result)
            
            # 2. 교차 검증 수행
            cross_validation = await self._perform_cross_validation(normalized_results)
            
            # 3. 인사이트 추출
            insights = await self._extract_integrated_insights(
                normalized_results, cross_validation, integration_strategy
            )
            
            # 4. 품질 평가
            quality_assessment = await self._assess_integration_quality(
                normalized_results, insights, cross_validation
            )
            
            # 5. 종합 보고서 생성
            synthesis_report = await self._generate_synthesis_report(
                normalized_results, insights, cross_validation, quality_assessment
            )
            
            # 6. 추천사항 생성
            recommendations = await self._generate_recommendations(
                insights, quality_assessment, execution_result
            )
            
            # 7. 전체 신뢰도 계산
            confidence_score = self._calculate_overall_confidence(
                normalized_results, cross_validation, quality_assessment
            )
            
            integration_time = time.time() - start_time
            
            result = IntegrationResult(
                integration_id=integration_id,
                strategy=integration_strategy,
                agent_results=normalized_results,
                cross_validation=cross_validation,
                integrated_insights=insights,
                quality_assessment=quality_assessment,
                synthesis_report=synthesis_report,
                recommendations=recommendations,
                confidence_score=confidence_score,
                integration_time=integration_time,
                metadata=context or {}
            )
            
            self.integration_history.append(result)
            
            logger.info(f"✅ Integration completed: {integration_id} ({integration_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Integration failed: {integration_id} - {e}")
            raise
    
    async def _normalize_agent_results(self, execution_result: ExecutionResult) -> List[AgentResult]:
        """에이전트 결과 정규화"""
        normalized_results = []
        
        for task_result in execution_result.task_results:
            if task_result.get("status") == "completed" and task_result.get("result"):
                try:
                    # 결과 타입 추론
                    result_type = self._infer_result_type(task_result)
                    
                    # 에이전트 결과 객체 생성
                    agent_result = AgentResult(
                        agent_name=task_result["agent_name"],
                        agent_type=task_result["agent_type"],
                        result_type=result_type,
                        content=task_result["result"],
                        confidence=self._extract_confidence(task_result),
                        execution_time=task_result.get("execution_time", 0),
                        timestamp=datetime.now(),
                        metadata=task_result
                    )
                    
                    # 품질 점수 계산
                    agent_result.quality_scores = await self._calculate_quality_scores(agent_result)
                    
                    normalized_results.append(agent_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to normalize result from {task_result.get('agent_name')}: {e}")
        
        logger.info(f"📊 Normalized {len(normalized_results)} agent results")
        return normalized_results
    
    def _infer_result_type(self, task_result: Dict[str, Any]) -> ResultType:
        """결과 타입 추론"""
        agent_type = task_result.get("agent_type", "").lower()
        
        if "visualization" in agent_type:
            return ResultType.VISUALIZATION
        elif "eda" in agent_type or "analysis" in agent_type:
            return ResultType.DATA_ANALYSIS
        elif "ml" in agent_type or "model" in agent_type:
            return ResultType.PREDICTION
        elif "cleaning" in agent_type:
            return ResultType.DATA_ANALYSIS
        else:
            return ResultType.INSIGHT
    
    def _extract_confidence(self, task_result: Dict[str, Any]) -> float:
        """신뢰도 추출"""
        result = task_result.get("result", {})
        
        # 다양한 신뢰도 필드 확인
        for field in ["confidence", "confidence_score", "reliability", "certainty"]:
            if field in result:
                return float(result[field])
        
        # 기본 신뢰도 (성공 시 0.8)
        return 0.8 if task_result.get("status") == "completed" else 0.0
    
    async def _calculate_quality_scores(self, agent_result: AgentResult) -> Dict[QualityMetric, float]:
        """품질 점수 계산"""
        
        prompt = f"""다음 에이전트 결과의 품질을 평가해주세요:

**에이전트 정보**:
- 이름: {agent_result.agent_name}
- 타입: {agent_result.agent_type}
- 결과 타입: {agent_result.result_type.value}
- 신뢰도: {agent_result.confidence:.2f}

**결과 내용**:
{json.dumps(agent_result.content, ensure_ascii=False, indent=2)}

**품질 평가 기준**:
1. completeness (완성도): 결과가 얼마나 완전한가?
2. consistency (일관성): 내부적으로 일관성이 있는가?
3. accuracy (정확성): 결과가 정확해 보이는가?
4. relevance (관련성): 요청과 얼마나 관련이 있는가?
5. clarity (명확성): 결과가 명확하고 이해하기 쉬운가?
6. actionability (실행가능성): 실제로 활용 가능한가?

각 품질 지표를 0.0-1.0 사이의 점수로 평가하고 JSON 형식으로 응답해주세요:
{{
  "completeness": 0.8,
  "consistency": 0.9,
  "accuracy": 0.7,
  "relevance": 0.8,
  "clarity": 0.9,
  "actionability": 0.6
}}"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                scores_dict = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    scores_dict = json.loads(json_match.group(1))
                else:
                    scores_dict = {}
            
            # QualityMetric enum으로 변환
            quality_scores = {}
            for metric in QualityMetric:
                score = scores_dict.get(metric.value, 0.5)  # 기본값 0.5
                quality_scores[metric] = max(0.0, min(1.0, float(score)))
            
            return quality_scores
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            # 기본 점수 반환
            return {metric: 0.5 for metric in QualityMetric}
    
    async def _perform_cross_validation(self, agent_results: List[AgentResult]) -> CrossValidationResult:
        """교차 검증 수행"""
        
        if len(agent_results) < 2:
            return CrossValidationResult(
                consistency_score=1.0,
                conflicting_findings=[],
                supporting_evidence=[],
                validation_notes="단일 에이전트 결과로 교차 검증 불가",
                confidence_adjustment=0.0
            )
        
        # 결과 간 일관성 분석
        consistency_prompt = f"""다음 다중 에이전트 결과들의 일관성을 분석해주세요:

**에이전트 결과들**:
{json.dumps([{
    'agent': r.agent_name,
    'type': r.agent_type,
    'content': r.content,
    'confidence': r.confidence
} for r in agent_results], ensure_ascii=False, indent=2)}

**분석 요구사항**:
1. 결과 간 일관성 평가 (0.0-1.0)
2. 충돌하는 발견사항 식별
3. 서로 지지하는 증거 식별
4. 교차 검증 노트 작성
5. 신뢰도 조정 권고 (-0.2 ~ +0.2)

다음 JSON 형식으로 응답해주세요:
{{
  "consistency_score": 0.8,
  "conflicting_findings": [
    {{"description": "에이전트 A는 X라고 했지만 에이전트 B는 Y라고 함", "severity": "medium"}}
  ],
  "supporting_evidence": [
    {{"description": "에이전트 A와 B 모두 Z를 확인함", "strength": "high"}}
  ],
  "validation_notes": "전반적으로 일관성 있는 결과...",
  "confidence_adjustment": 0.1
}}"""

        try:
            response = await self.llm.ainvoke(consistency_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                validation_dict = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    validation_dict = json.loads(json_match.group(1))
                else:
                    validation_dict = {}
            
            return CrossValidationResult(
                consistency_score=validation_dict.get("consistency_score", 0.5),
                conflicting_findings=validation_dict.get("conflicting_findings", []),
                supporting_evidence=validation_dict.get("supporting_evidence", []),
                validation_notes=validation_dict.get("validation_notes", "교차 검증 완료"),
                confidence_adjustment=validation_dict.get("confidence_adjustment", 0.0)
            )
            
        except Exception as e:
            logger.warning(f"Cross validation failed: {e}")
            return CrossValidationResult(
                consistency_score=0.5,
                conflicting_findings=[],
                supporting_evidence=[],
                validation_notes=f"교차 검증 오류: {str(e)}",
                confidence_adjustment=0.0
            )
    
    async def _extract_integrated_insights(
        self,
        agent_results: List[AgentResult],
        cross_validation: CrossValidationResult,
        strategy: IntegrationStrategy
    ) -> List[IntegratedInsight]:
        """통합 인사이트 추출"""
        
        insights_prompt = f"""다음 다중 에이전트 분석 결과를 바탕으로 통합 인사이트를 추출해주세요:

**에이전트 결과들**:
{json.dumps([{
    'agent': r.agent_name,
    'type': r.agent_type,
    'content': r.content,
    'confidence': r.confidence,
    'quality_scores': {k.value: v for k, v in r.quality_scores.items()}
} for r in agent_results], ensure_ascii=False, indent=2)}

**교차 검증 결과**:
- 일관성 점수: {cross_validation.consistency_score:.2f}
- 지지 증거: {len(cross_validation.supporting_evidence)}개
- 충돌 발견: {len(cross_validation.conflicting_findings)}개

**통합 전략**: {strategy.value}

**인사이트 추출 요구사항**:
1. 각 에이전트 결과를 종합하여 핵심 인사이트 도출
2. 인사이트별 신뢰도 평가
3. 실행 가능한 항목 식별
4. 우선순위 설정

다음 JSON 형식으로 응답해주세요:
{{
  "insights": [
    {{
      "insight_type": "data_quality",
      "content": "데이터 품질이 전반적으로 양호하며...",
      "confidence": 0.9,
      "supporting_agents": ["DataCleaningAgent", "EDAAgent"],
      "evidence_strength": 0.8,
      "actionable_items": ["데이터 정리 프로세스 유지", "품질 모니터링 강화"],
      "priority": 1
    }}
  ]
}}"""

        try:
            response = await self.llm.ainvoke(insights_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                insights_dict = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    insights_dict = json.loads(json_match.group(1))
                else:
                    insights_dict = {"insights": []}
            
            # IntegratedInsight 객체로 변환
            insights = []
            for insight_data in insights_dict.get("insights", []):
                insight = IntegratedInsight(
                    insight_type=insight_data.get("insight_type", "general"),
                    content=insight_data.get("content", ""),
                    confidence=insight_data.get("confidence", 0.5),
                    supporting_agents=insight_data.get("supporting_agents", []),
                    evidence_strength=insight_data.get("evidence_strength", 0.5),
                    actionable_items=insight_data.get("actionable_items", []),
                    priority=insight_data.get("priority", 5)
                )
                insights.append(insight)
            
            # 우선순위별 정렬
            insights.sort(key=lambda x: x.priority)
            
            logger.info(f"📝 Extracted {len(insights)} integrated insights")
            return insights
            
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            return []
    
    async def _assess_integration_quality(
        self,
        agent_results: List[AgentResult],
        insights: List[IntegratedInsight],
        cross_validation: CrossValidationResult
    ) -> Dict[QualityMetric, float]:
        """통합 품질 평가"""
        
        quality_assessment = {}
        
        # 완성도: 에이전트 결과 수와 인사이트 수를 기반으로 평가
        completeness = min(1.0, len(agent_results) / 3.0) * 0.5 + min(1.0, len(insights) / 3.0) * 0.5
        quality_assessment[QualityMetric.COMPLETENESS] = completeness
        
        # 일관성: 교차 검증 결과 활용
        quality_assessment[QualityMetric.CONSISTENCY] = cross_validation.consistency_score
        
        # 정확성: 에이전트 결과들의 평균 신뢰도
        if agent_results:
            accuracy = sum(r.confidence for r in agent_results) / len(agent_results)
            quality_assessment[QualityMetric.ACCURACY] = accuracy
        else:
            quality_assessment[QualityMetric.ACCURACY] = 0.0
        
        # 관련성: 에이전트 결과들의 관련성 점수 평균
        relevance_scores = []
        for result in agent_results:
            if QualityMetric.RELEVANCE in result.quality_scores:
                relevance_scores.append(result.quality_scores[QualityMetric.RELEVANCE])
        
        if relevance_scores:
            quality_assessment[QualityMetric.RELEVANCE] = sum(relevance_scores) / len(relevance_scores)
        else:
            quality_assessment[QualityMetric.RELEVANCE] = 0.5
        
        # 명확성: 인사이트의 명확성 평가
        if insights:
            clarity = sum(i.confidence for i in insights) / len(insights)
            quality_assessment[QualityMetric.CLARITY] = clarity
        else:
            quality_assessment[QualityMetric.CLARITY] = 0.0
        
        # 실행가능성: 실행 가능한 항목의 비율
        total_actionable = sum(len(i.actionable_items) for i in insights)
        actionability = min(1.0, total_actionable / max(1, len(insights) * 2))
        quality_assessment[QualityMetric.ACTIONABILITY] = actionability
        
        return quality_assessment
    
    async def _generate_synthesis_report(
        self,
        agent_results: List[AgentResult],
        insights: List[IntegratedInsight],
        cross_validation: CrossValidationResult,
        quality_assessment: Dict[QualityMetric, float]
    ) -> str:
        """종합 보고서 생성"""
        
        report_prompt = f"""다음 정보를 바탕으로 종합 분석 보고서를 작성해주세요:

**에이전트 분석 결과 요약**:
- 총 {len(agent_results)}개 에이전트 결과
- 평균 신뢰도: {sum(r.confidence for r in agent_results) / len(agent_results):.2f}
- 주요 결과 타입: {', '.join(set(r.result_type.value for r in agent_results))}

**통합 인사이트**: {len(insights)}개
{chr(10).join(f"- {i.insight_type}: {i.content[:100]}..." for i in insights[:5])}

**교차 검증 결과**:
- 일관성: {cross_validation.consistency_score:.2f}
- 지지 증거: {len(cross_validation.supporting_evidence)}개
- 충돌 발견: {len(cross_validation.conflicting_findings)}개

**품질 평가**:
{chr(10).join(f"- {k.value}: {v:.2f}" for k, v in quality_assessment.items())}

**보고서 구성 요구사항**:
1. 실행 개요 (3-4문장)
2. 주요 발견사항 (5-8개 항목)
3. 품질 및 신뢰도 평가
4. 제한사항 및 고려사항
5. 결론 및 다음 단계

체계적이고 전문적인 한국어 보고서를 작성해주세요."""

        try:
            response = await self.llm.ainvoke(report_prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            return f"보고서 생성 오류: {str(e)}"
    
    async def _generate_recommendations(
        self,
        insights: List[IntegratedInsight],
        quality_assessment: Dict[QualityMetric, float],
        execution_result: ExecutionResult
    ) -> List[str]:
        """추천사항 생성"""
        
        recommendations_prompt = f"""다음 분석 결과를 바탕으로 구체적인 추천사항을 제안해주세요:

**통합 인사이트**:
{json.dumps([{
    'type': i.insight_type,
    'content': i.content,
    'confidence': i.confidence,
    'actionable_items': i.actionable_items,
    'priority': i.priority
} for i in insights], ensure_ascii=False, indent=2)}

**품질 평가**:
{json.dumps({k.value: v for k, v in quality_assessment.items()}, ensure_ascii=False, indent=2)}

**실행 결과**:
- 총 실행 시간: {execution_result.execution_time:.2f}초
- 완료 태스크: {execution_result.completed_tasks}/{execution_result.total_tasks}
- 전체 신뢰도: {execution_result.confidence_score:.2f}

**추천사항 유형**:
1. 즉시 실행 가능한 조치
2. 단기 개선 방안 (1-2주)
3. 중기 전략 (1-3개월)
4. 장기 계획 (3개월 이상)
5. 리스크 관리 방안

구체적이고 실행 가능한 추천사항을 리스트 형태로 제공해주세요."""

        try:
            response = await self.llm.ainvoke(recommendations_prompt)
            content = response.content.strip()
            
            # 리스트 형태로 파싱
            recommendations = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    recommendations.append(line[1:].strip())
                elif line and any(char.isdigit() for char in line[:3]):
                    # 번호가 있는 항목
                    recommendations.append(line.split('.', 1)[-1].strip())
            
            return recommendations if recommendations else [content]
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            return [f"추천사항 생성 오류: {str(e)}"]
    
    def _calculate_overall_confidence(
        self,
        agent_results: List[AgentResult],
        cross_validation: CrossValidationResult,
        quality_assessment: Dict[QualityMetric, float]
    ) -> float:
        """전체 신뢰도 계산"""
        
        if not agent_results:
            return 0.0
        
        # 기본 신뢰도: 에이전트 결과들의 평균 신뢰도
        base_confidence = sum(r.confidence for r in agent_results) / len(agent_results)
        
        # 교차 검증 조정
        consistency_adjustment = (cross_validation.consistency_score - 0.5) * 0.2
        validation_adjustment = cross_validation.confidence_adjustment
        
        # 품질 평가 조정
        quality_score = sum(quality_assessment.values()) / len(quality_assessment)
        quality_adjustment = (quality_score - 0.5) * 0.1
        
        # 에이전트 수 보정 (더 많은 에이전트 = 더 높은 신뢰도)
        agent_count_bonus = min(0.1, len(agent_results) * 0.02)
        
        final_confidence = base_confidence + consistency_adjustment + validation_adjustment + quality_adjustment + agent_count_bonus
        
        return max(0.0, min(1.0, final_confidence))
    
    async def get_integration_history(self) -> List[IntegrationResult]:
        """통합 이력 조회"""
        return self.integration_history.copy()
    
    async def get_integration_summary(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """통합 요약 조회"""
        for result in self.integration_history:
            if result.integration_id == integration_id:
                return {
                    "integration_id": result.integration_id,
                    "strategy": result.strategy.value,
                    "agent_count": len(result.agent_results),
                    "insight_count": len(result.integrated_insights),
                    "confidence_score": result.confidence_score,
                    "integration_time": result.integration_time,
                    "quality_scores": {k.value: v for k, v in result.quality_assessment.items()}
                }
        return None 