"""
A2A Result Integrator - A2A 결과 통합 시스템

요구사항에 따른 구현:
- 다중 에이전트 결과 일관성 검증
- 충돌 해결 및 결과 통합 알고리즘
- 에이전트별 기여도 계산 및 품질 평가
- LLM 기반 지능적 결과 통합
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum

from ...llm_factory import LLMFactory
from .a2a_communication_protocol import A2AResponse
from .a2a_agent_discovery import A2AAgentInfo

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """충돌 유형"""
    VALUE_MISMATCH = "value_mismatch"
    METHODOLOGY_DIFFERENCE = "methodology_difference"
    CONFIDENCE_CONFLICT = "confidence_conflict"
    FORMAT_INCONSISTENCY = "format_inconsistency"
    LOGICAL_CONTRADICTION = "logical_contradiction"


class IntegrationStrategy(Enum):
    """통합 전략"""
    CONSENSUS = "consensus"
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    EXPERT_PREFERENCE = "expert_preference"
    LLM_MEDIATED = "llm_mediated"


@dataclass
class ConflictResolution:
    """충돌 해결 정보"""
    conflict_type: ConflictType
    conflicting_agents: List[str]
    resolution_strategy: IntegrationStrategy
    resolved_value: Any
    confidence: float
    reasoning: str


@dataclass
class AgentContribution:
    """에이전트 기여도"""
    agent_id: str
    agent_name: str
    contribution_score: float
    quality_score: float
    uniqueness_score: float
    reliability_score: float
    key_insights: List[str]
    data_quality: Dict[str, float]


@dataclass
class IntegratedResult:
    """통합된 결과"""
    consolidated_data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence_scores: Dict[str, float]
    agent_contributions: List[AgentContribution]
    conflict_resolutions: List[ConflictResolution]
    integration_metadata: Dict[str, Any]
    quality_assessment: Dict[str, float]


class A2AResultIntegrator:
    """
    A2A 결과 통합 시스템
    - 다중 에이전트 결과의 지능적 통합
    - 충돌 감지 및 해결
    - 품질 평가 및 기여도 계산
    """
    
    def __init__(self):
        """A2AResultIntegrator 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.integration_history: List[Dict] = []
        logger.info("A2AResultIntegrator initialized")
    
    async def integrate_results(
        self,
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo],
        original_query: str,
        meta_analysis: Dict[str, Any]
    ) -> IntegratedResult:
        """
        다중 에이전트 결과 통합
        
        Args:
            responses: A2A 에이전트 응답들
            agents: 에이전트 정보들
            original_query: 원본 사용자 쿼리
            meta_analysis: 메타 분석 결과
            
        Returns:
            통합된 결과
        """
        logger.info(f"Integrating results from {len(responses)} agents")
        
        try:
            # 1. 응답 품질 평가
            quality_assessments = await self._assess_response_quality(
                responses, agents
            )
            
            # 2. 충돌 감지
            conflicts = await self._detect_conflicts(responses, agents)
            
            # 3. 충돌 해결
            conflict_resolutions = []
            if conflicts:
                conflict_resolutions = await self._resolve_conflicts(
                    conflicts, responses, agents, meta_analysis
                )
            
            # 4. 에이전트 기여도 계산
            agent_contributions = await self._calculate_agent_contributions(
                responses, agents, quality_assessments
            )
            
            # 5. 결과 통합
            consolidated_data = await self._consolidate_data(
                responses, conflict_resolutions, agent_contributions
            )
            
            # 6. 인사이트 및 추천사항 생성
            insights = await self._generate_consolidated_insights(
                consolidated_data, responses, original_query
            )
            
            recommendations = await self._generate_recommendations(
                consolidated_data, insights, meta_analysis
            )
            
            # 7. 신뢰도 계산
            confidence_scores = await self._calculate_confidence_scores(
                consolidated_data, agent_contributions, conflict_resolutions
            )
            
            # 8. 통합 메타데이터 생성
            integration_metadata = self._generate_integration_metadata(
                responses, agents, conflict_resolutions
            )
            
            # 9. 전체 품질 평가
            overall_quality = await self._assess_overall_quality(
                consolidated_data, agent_contributions, conflicts
            )
            
            integrated_result = IntegratedResult(
                consolidated_data=consolidated_data,
                insights=insights,
                recommendations=recommendations,
                confidence_scores=confidence_scores,
                agent_contributions=agent_contributions,
                conflict_resolutions=conflict_resolutions,
                integration_metadata=integration_metadata,
                quality_assessment=overall_quality
            )
            
            # 10. 통합 이력 저장
            self._record_integration_history(
                original_query, responses, integrated_result
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error in result integration: {e}")
            raise
    
    async def _assess_response_quality(
        self,
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo]
    ) -> List[Dict[str, float]]:
        """
        응답 품질 평가
        """
        quality_assessments = []
        
        for response, agent in zip(responses, agents):
            # 기본 품질 지표
            basic_quality = {
                "completeness": self._assess_completeness(response),
                "format_consistency": self._assess_format_consistency(response),
                "execution_time_score": self._assess_execution_time(response),
                "error_rate": 1.0 if response.status == "success" else 0.0
            }
            
            # LLM 기반 품질 평가
            llm_quality = await self._llm_assess_quality(response, agent)
            
            # 통합 품질 점수
            combined_quality = {**basic_quality, **llm_quality}
            quality_assessments.append(combined_quality)
        
        return quality_assessments
    
    def _assess_completeness(self, response: A2AResponse) -> float:
        """응답 완전성 평가"""
        if response.status != "success":
            return 0.0
        
        data = response.data
        if not data:
            return 0.0
        
        # 데이터 필드 수와 내용을 기반으로 완전성 평가
        field_count = len(data) if isinstance(data, dict) else 1
        content_score = min(1.0, field_count / 5)  # 5개 필드를 기준으로 정규화
        
        return content_score
    
    def _assess_format_consistency(self, response: A2AResponse) -> float:
        """형식 일관성 평가"""
        try:
            # JSON 직렬화 가능성 체크
            json.dumps(response.data)
            return 1.0
        except (TypeError, ValueError):
            return 0.5
    
    def _assess_execution_time(self, response: A2AResponse) -> float:
        """실행 시간 기반 점수"""
        # 30초를 기준으로 점수 계산 (빠를수록 높은 점수)
        if response.execution_time <= 0:
            return 1.0
        
        score = max(0.0, 1.0 - (response.execution_time / 30.0))
        return score
    
    async def _llm_assess_quality(
        self, 
        response: A2AResponse, 
        agent: A2AAgentInfo
    ) -> Dict[str, float]:
        """
        LLM 기반 응답 품질 평가
        """
        prompt = f"""
        A2A 에이전트의 응답 품질을 평가하세요.
        
        에이전트: {agent.name}
        에이전트 능력: {agent.capabilities}
        응답 상태: {response.status}
        응답 데이터: {json.dumps(response.data, ensure_ascii=False)[:1000]}
        실행 시간: {response.execution_time}초
        
        다음 기준으로 0.0-1.0 점수를 매기세요:
        
        JSON 형식으로 응답하세요:
        {{
            "relevance": 0.0-1.0,
            "accuracy": 0.0-1.0,
            "clarity": 0.0-1.0,
            "usefulness": 0.0-1.0,
            "technical_quality": 0.0-1.0
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response_text)
    
    async def _detect_conflicts(
        self,
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo]
    ) -> List[Dict[str, Any]]:
        """
        응답 간 충돌 감지
        """
        conflicts = []
        
        if len(responses) < 2:
            return conflicts
        
        # 성공한 응답들만 비교
        successful_responses = [
            (r, a) for r, a in zip(responses, agents) 
            if r.status == "success"
        ]
        
        if len(successful_responses) < 2:
            return conflicts
        
        # LLM을 사용한 충돌 감지
        comparison_data = []
        for response, agent in successful_responses:
            comparison_data.append({
                "agent_name": agent.name,
                "agent_id": agent.id,
                "capabilities": agent.capabilities,
                "response_data": response.data,
                "metadata": response.metadata
            })
        
        prompt = f"""
        다음 에이전트들의 응답을 비교하여 충돌을 감지하세요.
        
        응답들: {json.dumps(comparison_data, ensure_ascii=False)}
        
        다음 유형의 충돌을 찾아보세요:
        1. 값 불일치 (같은 항목에 대해 다른 값)
        2. 방법론 차이 (같은 목표에 대해 다른 접근법)
        3. 신뢰도 충돌 (같은 결론에 대해 다른 신뢰도)
        4. 형식 불일치 (같은 데이터를 다른 형식으로 표현)
        5. 논리적 모순 (서로 모순되는 결론)
        
        JSON 형식으로 응답하세요:
        {{
            "conflicts": [
                {{
                    "type": "value_mismatch|methodology_difference|confidence_conflict|format_inconsistency|logical_contradiction",
                    "description": "충돌 설명",
                    "involved_agents": ["agent_id1", "agent_id2"],
                    "severity": "low|medium|high",
                    "conflicting_data": {{
                        "agent_id1": "값1",
                        "agent_id2": "값2"
                    }}
                }}
            ]
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        conflict_data = self._parse_json_response(response_text)
        
        return conflict_data.get("conflicts", [])
    
    async def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo],
        meta_analysis: Dict[str, Any]
    ) -> List[ConflictResolution]:
        """
        충돌 해결
        """
        resolutions = []
        
        for conflict in conflicts:
            resolution = await self._resolve_single_conflict(
                conflict, responses, agents, meta_analysis
            )
            resolutions.append(resolution)
        
        return resolutions
    
    async def _resolve_single_conflict(
        self,
        conflict: Dict[str, Any],
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo],
        meta_analysis: Dict[str, Any]
    ) -> ConflictResolution:
        """
        개별 충돌 해결
        """
        # 관련 에이전트들의 응답 추출
        involved_agent_ids = conflict.get("involved_agents", [])
        involved_responses = []
        
        for agent_id in involved_agent_ids:
            for response, agent in zip(responses, agents):
                if agent.id == agent_id:
                    involved_responses.append((response, agent))
                    break
        
        prompt = f"""
        다음 충돌을 해결하세요.
        
        충돌 정보: {conflict}
        관련 에이전트 응답들: {json.dumps([
            {
                "agent": agent.name,
                "capabilities": agent.capabilities,
                "response": response.data
            }
            for response, agent in involved_responses
        ], ensure_ascii=False)}
        
        사용자 컨텍스트: {meta_analysis.get('user_profile', {})}
        
        최적의 해결 방법을 선택하고 결과를 제시하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "resolution_strategy": "consensus|weighted_average|highest_confidence|expert_preference|llm_mediated",
            "resolved_value": "해결된 값 또는 결론",
            "confidence": 0.0-1.0,
            "reasoning": "해결 과정의 상세한 설명",
            "weight_assignments": {{
                "agent_id1": 0.6,
                "agent_id2": 0.4
            }}
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        resolution_data = self._parse_json_response(response_text)
        
        return ConflictResolution(
            conflict_type=ConflictType(conflict.get("type", "value_mismatch")),
            conflicting_agents=involved_agent_ids,
            resolution_strategy=IntegrationStrategy(
                resolution_data.get("resolution_strategy", "llm_mediated")
            ),
            resolved_value=resolution_data.get("resolved_value"),
            confidence=resolution_data.get("confidence", 0.5),
            reasoning=resolution_data.get("reasoning", "")
        )
    
    async def _calculate_agent_contributions(
        self,
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo],
        quality_assessments: List[Dict[str, float]]
    ) -> List[AgentContribution]:
        """
        에이전트 기여도 계산
        """
        contributions = []
        
        for response, agent, quality in zip(responses, agents, quality_assessments):
            contribution = await self._calculate_single_contribution(
                response, agent, quality, responses
            )
            contributions.append(contribution)
        
        return contributions
    
    async def _calculate_single_contribution(
        self,
        response: A2AResponse,
        agent: A2AAgentInfo,
        quality: Dict[str, float],
        all_responses: List[A2AResponse]
    ) -> AgentContribution:
        """
        개별 에이전트 기여도 계산
        """
        # 기본 점수들
        quality_score = sum(quality.values()) / len(quality) if quality else 0.0
        reliability_score = 1.0 if response.status == "success" else 0.0
        
        # 고유성 점수 계산 (다른 에이전트들과 얼마나 다른 인사이트를 제공하는가)
        uniqueness_score = await self._calculate_uniqueness(response, all_responses)
        
        # LLM 기반 기여도 평가
        contribution_assessment = await self._llm_assess_contribution(
            response, agent, quality
        )
        
        contribution_score = (
            quality_score * 0.3 +
            reliability_score * 0.2 +
            uniqueness_score * 0.2 +
            contribution_assessment.get("overall_contribution", 0.5) * 0.3
        )
        
        return AgentContribution(
            agent_id=agent.id,
            agent_name=agent.name,
            contribution_score=contribution_score,
            quality_score=quality_score,
            uniqueness_score=uniqueness_score,
            reliability_score=reliability_score,
            key_insights=contribution_assessment.get("key_insights", []),
            data_quality=quality
        )
    
    async def _calculate_uniqueness(
        self, 
        target_response: A2AResponse, 
        all_responses: List[A2AResponse]
    ) -> float:
        """
        응답의 고유성 점수 계산
        """
        if len(all_responses) <= 1:
            return 1.0
        
        # 다른 성공한 응답들과 비교
        other_responses = [
            r for r in all_responses 
            if r.agent_id != target_response.agent_id and r.status == "success"
        ]
        
        if not other_responses:
            return 1.0
        
        # LLM을 사용한 고유성 평가
        prompt = f"""
        대상 응답이 다른 응답들과 비교했을 때 얼마나 고유한 정보를 제공하는지 평가하세요.
        
        대상 응답: {json.dumps(target_response.data, ensure_ascii=False)[:500]}
        
        다른 응답들: {json.dumps([
            r.data for r in other_responses
        ], ensure_ascii=False)[:1000]}
        
        0.0 (완전히 중복) ~ 1.0 (완전히 고유) 범위로 점수를 매기세요.
        
        JSON 형식으로 응답하세요:
        {{
            "uniqueness_score": 0.0-1.0,
            "unique_aspects": ["고유한 측면1", "고유한 측면2"]
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        uniqueness_data = self._parse_json_response(response_text)
        
        return uniqueness_data.get("uniqueness_score", 0.5)
    
    async def _llm_assess_contribution(
        self,
        response: A2AResponse,
        agent: A2AAgentInfo,
        quality: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        LLM 기반 기여도 평가
        """
        prompt = f"""
        에이전트의 기여도를 평가하세요.
        
        에이전트: {agent.name}
        능력: {agent.capabilities}
        응답: {json.dumps(response.data, ensure_ascii=False)[:800]}
        품질 점수: {quality}
        
        다음을 평가하세요:
        1. 전반적인 기여도
        2. 핵심 인사이트들
        3. 데이터 분석의 깊이
        4. 실용적 가치
        
        JSON 형식으로 응답하세요:
        {{
            "overall_contribution": 0.0-1.0,
            "key_insights": ["인사이트1", "인사이트2"],
            "analysis_depth": 0.0-1.0,
            "practical_value": 0.0-1.0
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response_text)
    
    async def _consolidate_data(
        self,
        responses: List[A2AResponse],
        conflict_resolutions: List[ConflictResolution],
        agent_contributions: List[AgentContribution]
    ) -> Dict[str, Any]:
        """
        데이터 통합
        """
        # 성공한 응답들의 데이터 수집
        successful_data = []
        for response in responses:
            if response.status == "success" and response.data:
                successful_data.append(response.data)
        
        if not successful_data:
            return {"error": "No successful responses to consolidate"}
        
        # LLM 기반 데이터 통합
        prompt = f"""
        다음 에이전트 응답들을 통합하여 일관된 결과를 생성하세요.
        
        응답 데이터들: {json.dumps(successful_data, ensure_ascii=False)}
        
        충돌 해결사항들: {json.dumps([
            {
                "conflict_type": res.conflict_type.value,
                "resolved_value": res.resolved_value,
                "reasoning": res.reasoning
            }
            for res in conflict_resolutions
        ], ensure_ascii=False)}
        
        에이전트 기여도들: {json.dumps([
            {
                "agent": contrib.agent_name,
                "contribution_score": contrib.contribution_score,
                "key_insights": contrib.key_insights
            }
            for contrib in agent_contributions
        ], ensure_ascii=False)}
        
        통합 원칙:
        1. 충돌이 해결된 값들을 우선 사용
        2. 기여도가 높은 에이전트의 결과에 더 높은 가중치
        3. 일관성 있는 데이터 구조 유지
        4. 중복 제거 및 정보 보완
        
        JSON 형식으로 통합된 결과를 제공하세요:
        {{
            "summary": "전체 요약",
            "key_findings": ["주요 발견사항1", "주요 발견사항2"],
            "data_analysis": {{
                "patterns": ["패턴1", "패턴2"],
                "statistics": {{}},
                "anomalies": ["이상점1", "이상점2"]
            }},
            "recommendations": ["권장사항1", "권장사항2"],
            "confidence_assessment": {{
                "overall_confidence": 0.0-1.0,
                "reliability_factors": ["요인1", "요인2"]
            }}
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response_text)
    
    async def _generate_consolidated_insights(
        self,
        consolidated_data: Dict[str, Any],
        responses: List[A2AResponse],
        original_query: str
    ) -> List[str]:
        """
        통합된 인사이트 생성
        """
        prompt = f"""
        통합된 데이터를 바탕으로 핵심 인사이트를 생성하세요.
        
        원본 쿼리: {original_query}
        통합된 데이터: {json.dumps(consolidated_data, ensure_ascii=False)}
        
        다음 기준으로 인사이트를 생성하세요:
        1. 사용자 질문에 직접적으로 답변하는 인사이트
        2. 데이터에서 발견된 중요한 패턴
        3. 예상치 못한 발견사항
        4. 실행 가능한 조치 방향
        
        JSON 형식으로 응답하세요:
        {{
            "insights": [
                "인사이트1 - 사용자 질문에 대한 직접적 답변",
                "인사이트2 - 중요한 데이터 패턴",
                "인사이트3 - 예상치 못한 발견",
                "인사이트4 - 실행 가능한 방향"
            ]
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        insights_data = self._parse_json_response(response_text)
        
        return insights_data.get("insights", [])
    
    async def _generate_recommendations(
        self,
        consolidated_data: Dict[str, Any],
        insights: List[str],
        meta_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        추천사항 생성
        """
        user_level = meta_analysis.get("user_profile", {}).get("expertise_level", "intermediate")
        
        prompt = f"""
        분석 결과와 인사이트를 바탕으로 사용자 수준에 맞는 추천사항을 생성하세요.
        
        통합된 데이터: {json.dumps(consolidated_data, ensure_ascii=False)[:800]}
        인사이트들: {insights}
        사용자 수준: {user_level}
        
        사용자 수준에 맞는 추천사항을 생성하세요:
        - 초보자: 구체적이고 단계별 가이드
        - 중급자: 선택 옵션과 고려사항 제시
        - 전문가: 심화 분석 방향과 기술적 권장사항
        
        JSON 형식으로 응답하세요:
        {{
            "recommendations": [
                "추천사항1 - 즉시 실행 가능한 조치",
                "추천사항2 - 단기 개선 방안",
                "추천사항3 - 중장기 전략",
                "추천사항4 - 추가 분석 방향"
            ]
        }}
        """
        
        response_text = await self.llm_client.agenerate(prompt)
        recommendations_data = self._parse_json_response(response_text)
        
        return recommendations_data.get("recommendations", [])
    
    async def _calculate_confidence_scores(
        self,
        consolidated_data: Dict[str, Any],
        agent_contributions: List[AgentContribution],
        conflict_resolutions: List[ConflictResolution]
    ) -> Dict[str, float]:
        """
        신뢰도 점수 계산
        """
        # 에이전트 기여도 기반 전체 신뢰도
        avg_contribution = sum(c.contribution_score for c in agent_contributions) / len(agent_contributions)
        
        # 충돌 해결 신뢰도
        conflict_confidence = 1.0
        if conflict_resolutions:
            conflict_confidence = sum(r.confidence for r in conflict_resolutions) / len(conflict_resolutions)
        
        # 데이터 품질 기반 신뢰도
        data_confidence = consolidated_data.get("confidence_assessment", {}).get("overall_confidence", 0.5)
        
        return {
            "overall_confidence": (avg_contribution + conflict_confidence + data_confidence) / 3,
            "agent_consensus": avg_contribution,
            "conflict_resolution": conflict_confidence,
            "data_quality": data_confidence
        }
    
    def _generate_integration_metadata(
        self,
        responses: List[A2AResponse],
        agents: List[A2AAgentInfo],
        conflict_resolutions: List[ConflictResolution]
    ) -> Dict[str, Any]:
        """
        통합 메타데이터 생성
        """
        return {
            "integration_timestamp": datetime.now().isoformat(),
            "participating_agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "status": response.status,
                    "execution_time": response.execution_time
                }
                for response, agent in zip(responses, agents)
            ],
            "conflicts_detected": len(conflict_resolutions),
            "conflict_types": [r.conflict_type.value for r in conflict_resolutions],
            "integration_strategy": "llm_mediated_multi_agent",
            "total_execution_time": sum(r.execution_time for r in responses)
        }
    
    async def _assess_overall_quality(
        self,
        consolidated_data: Dict[str, Any],
        agent_contributions: List[AgentContribution],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        전체 품질 평가
        """
        # 기본 품질 지표들
        completeness = 1.0 if consolidated_data and len(consolidated_data) > 1 else 0.5
        consistency = 1.0 - (len(conflicts) / max(len(agent_contributions), 1)) * 0.5
        reliability = sum(c.reliability_score for c in agent_contributions) / len(agent_contributions)
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "reliability": reliability,
            "overall_quality": (completeness + consistency + reliability) / 3
        }
    
    def _record_integration_history(
        self,
        original_query: str,
        responses: List[A2AResponse],
        integrated_result: IntegratedResult
    ) -> None:
        """
        통합 이력 기록
        """
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": original_query,
            "agent_count": len(responses),
            "success_count": sum(1 for r in responses if r.status == "success"),
            "conflicts_resolved": len(integrated_result.conflict_resolutions),
            "overall_confidence": integrated_result.confidence_scores.get("overall_confidence", 0.0),
            "quality_score": integrated_result.quality_assessment.get("overall_quality", 0.0)
        }
        
        self.integration_history.append(history_entry)
        
        # 이력 크기 제한
        if len(self.integration_history) > 100:
            self.integration_history = self.integration_history[-100:]
    
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
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        통합 통계 조회
        
        Returns:
            통합 통계 정보
        """
        if not self.integration_history:
            return {"message": "No integration history available"}
        
        total_integrations = len(self.integration_history)
        avg_confidence = sum(h["overall_confidence"] for h in self.integration_history) / total_integrations
        avg_quality = sum(h["quality_score"] for h in self.integration_history) / total_integrations
        total_conflicts = sum(h["conflicts_resolved"] for h in self.integration_history)
        
        return {
            "total_integrations": total_integrations,
            "average_confidence": avg_confidence,
            "average_quality": avg_quality,
            "total_conflicts_resolved": total_conflicts,
            "success_rate": sum(h["success_count"] for h in self.integration_history) / 
                           sum(h["agent_count"] for h in self.integration_history),
            "recent_performance": self.integration_history[-5:]  # 최근 5개
        }