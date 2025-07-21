"""
LLM 기반 동적 에이전트 선택 로직

요구사항에 따른 구현:
- 메타 분석 결과를 바탕으로 한 에이전트 선택
- 하드코딩 없는 순수 LLM 기반 선택 로직
- 에이전트 조합 최적화 및 실행 순서 결정
- 병렬 실행 가능 에이전트 식별
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from ...llm_factory import LLMFactory
from .a2a_agent_discovery import A2AAgentInfo, A2AAgentDiscoverySystem

logger = logging.getLogger(__name__)


@dataclass
class AgentSelectionCriteria:
    """에이전트 선택 기준"""
    required_capabilities: List[str]
    preferred_capabilities: List[str]
    data_types: List[str]
    complexity_level: str
    performance_requirements: Dict[str, any]
    exclusions: List[str]


@dataclass
class AgentSelectionResult:
    """에이전트 선택 결과"""
    selected_agents: List[A2AAgentInfo]
    execution_plan: Dict[str, any]
    parallel_groups: List[List[str]]
    dependencies: Dict[str, List[str]]
    estimated_duration: float
    confidence_score: float
    reasoning: str


class LLMBasedAgentSelector:
    """
    LLM 기반 동적 에이전트 선택기
    - 하드코딩된 규칙 없이 LLM이 최적 에이전트 조합 결정
    - 메타 분석 결과와 사용자 요구사항 기반 동적 선택
    - 실행 계획 최적화 및 병렬 처리 식별
    """
    
    def __init__(self, discovery_system: A2AAgentDiscoverySystem):
        """
        LLMBasedAgentSelector 초기화
        
        Args:
            discovery_system: A2A 에이전트 발견 시스템
        """
        self.discovery_system = discovery_system
        self.llm_client = LLMFactory.create_llm()
        self.selection_history = []
        logger.info("LLMBasedAgentSelector initialized")
    
    async def select_agents_for_query(
        self, 
        meta_analysis: Dict, 
        query: str, 
        data_info: Dict,
        user_preferences: Dict = None
    ) -> AgentSelectionResult:
        """
        쿼리와 메타 분석 결과를 바탕으로 최적 에이전트 선택
        
        Args:
            meta_analysis: Universal Engine의 메타 분석 결과
            query: 사용자 쿼리
            data_info: 데이터 정보
            user_preferences: 사용자 선호사항
            
        Returns:
            에이전트 선택 결과
        """
        logger.info("Selecting optimal agents based on meta-analysis")
        
        try:
            # 1. 사용 가능한 에이전트 조회
            available_agents = self.discovery_system.get_available_agents()
            
            if not available_agents:
                raise ValueError("No available agents found")
            
            # 2. 선택 기준 생성
            selection_criteria = await self._generate_selection_criteria(
                meta_analysis, query, data_info, user_preferences
            )
            
            # 3. LLM 기반 에이전트 선택
            selection_result = await self._llm_select_agents(
                available_agents, selection_criteria, meta_analysis
            )
            
            # 4. 실행 계획 최적화
            optimized_plan = await self._optimize_execution_plan(
                selection_result, available_agents
            )
            
            # 5. 선택 이력 저장
            self.selection_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'selected_agents': [agent.id for agent in selection_result.selected_agents],
                'confidence': selection_result.confidence_score
            })
            
            return AgentSelectionResult(
                selected_agents=selection_result.selected_agents,
                execution_plan=optimized_plan,
                parallel_groups=optimized_plan.get('parallel_groups', []),
                dependencies=optimized_plan.get('dependencies', {}),
                estimated_duration=optimized_plan.get('estimated_duration', 0.0),
                confidence_score=selection_result.confidence_score,
                reasoning=selection_result.reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            raise
    
    async def _generate_selection_criteria(
        self, 
        meta_analysis: Dict, 
        query: str, 
        data_info: Dict,
        user_preferences: Dict = None
    ) -> AgentSelectionCriteria:
        """
        메타 분석을 바탕으로 에이전트 선택 기준 생성
        """
        prompt = f"""
        메타 분석 결과와 사용자 요구사항을 바탕으로 A2A 에이전트 선택 기준을 생성하세요.
        
        메타 분석 결과: {meta_analysis}
        사용자 쿼리: {query}
        데이터 정보: {data_info}
        사용자 선호사항: {user_preferences or "없음"}
        
        하드코딩된 규칙이 아닌, 이 특정 상황에 맞는 선택 기준을 동적으로 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "required_capabilities": ["필수 능력1", "필수 능력2"],
            "preferred_capabilities": ["선호 능력1", "선호 능력2"],
            "data_types": ["처리할 데이터 유형1", "처리할 데이터 유형2"],
            "complexity_level": "low|medium|high|expert",
            "performance_requirements": {{
                "speed": "fast|medium|slow",
                "accuracy": "high|medium|low",
                "resource_usage": "low|medium|high"
            }},
            "exclusions": ["제외할 에이전트 유형1", "제외할 에이전트 유형2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        criteria_data = self._parse_json_response(response)
        
        return AgentSelectionCriteria(
            required_capabilities=criteria_data.get('required_capabilities', []),
            preferred_capabilities=criteria_data.get('preferred_capabilities', []),
            data_types=criteria_data.get('data_types', []),
            complexity_level=criteria_data.get('complexity_level', 'medium'),
            performance_requirements=criteria_data.get('performance_requirements', {}),
            exclusions=criteria_data.get('exclusions', [])
        )
    
    async def _llm_select_agents(
        self, 
        available_agents: Dict[str, A2AAgentInfo],
        criteria: AgentSelectionCriteria,
        meta_analysis: Dict
    ) -> AgentSelectionResult:
        """
        LLM을 사용한 에이전트 선택
        """
        # 에이전트 정보를 LLM이 이해할 수 있는 형태로 변환
        agent_descriptions = {}
        for agent_id, agent in available_agents.items():
            agent_descriptions[agent_id] = {
                'name': agent.name,
                'capabilities': agent.capabilities,
                'description': agent.description,
                'performance': {
                    'response_time': agent.response_time,
                    'success_rate': agent.success_count / max(agent.success_count + agent.error_count, 1)
                }
            }
        
        prompt = f"""
        사용 가능한 A2A 에이전트들 중에서 주어진 기준에 가장 적합한 에이전트들을 선택하세요.
        
        사용 가능한 에이전트들: {json.dumps(agent_descriptions, ensure_ascii=False, indent=2)}
        
        선택 기준: {asdict(criteria)}
        
        메타 분석 컨텍스트: {meta_analysis.get('domain_context', {})}
        사용자 수준: {meta_analysis.get('user_profile', {}).get('expertise_level', 'unknown')}
        
        다음 원칙에 따라 선택하세요:
        1. 하드코딩된 규칙 없이 순수 추론으로 선택
        2. 에이전트 조합의 시너지 효과 고려
        3. 실행 순서와 의존성 관계 파악
        4. 병렬 실행 가능성 식별
        
        JSON 형식으로 응답하세요:
        {{
            "selected_agents": [
                {{
                    "agent_id": "에이전트 ID",
                    "selection_reason": "선택 이유",
                    "role_in_workflow": "워크플로우에서의 역할",
                    "priority": "high|medium|low"
                }}
            ],
            "execution_strategy": {{
                "approach": "sequential|parallel|hybrid",
                "rationale": "실행 전략 선택 이유"
            }},
            "expected_outcome": "예상 결과",
            "confidence_score": 0.0-1.0,
            "reasoning": "전체 선택 과정의 상세한 추론"
        }}
        """
        
        response = await self.llm_client.agenerate(response)
        selection_data = self._parse_json_response(response)
        
        # 선택된 에이전트 객체들 조회
        selected_agents = []
        for agent_selection in selection_data.get('selected_agents', []):
            agent_id = agent_selection.get('agent_id')
            if agent_id in available_agents:
                selected_agents.append(available_agents[agent_id])
        
        return AgentSelectionResult(
            selected_agents=selected_agents,
            execution_plan=selection_data.get('execution_strategy', {}),
            parallel_groups=[],  # 다음 단계에서 최적화
            dependencies={},     # 다음 단계에서 최적화
            estimated_duration=0.0,  # 다음 단계에서 계산
            confidence_score=selection_data.get('confidence_score', 0.5),
            reasoning=selection_data.get('reasoning', '')
        )
    
    async def _optimize_execution_plan(
        self, 
        selection_result: AgentSelectionResult,
        available_agents: Dict[str, A2AAgentInfo]
    ) -> Dict:
        """
        실행 계획 최적화
        """
        if not selection_result.selected_agents:
            return {'parallel_groups': [], 'dependencies': {}, 'estimated_duration': 0.0}
        
        # 선택된 에이전트들의 상세 정보
        agent_details = []
        for agent in selection_result.selected_agents:
            agent_details.append({
                'id': agent.id,
                'name': agent.name,
                'capabilities': agent.capabilities,
                'avg_response_time': agent.response_time
            })
        
        prompt = f"""
        선택된 에이전트들의 실행 계획을 최적화하세요.
        
        선택된 에이전트들: {json.dumps(agent_details, ensure_ascii=False, indent=2)}
        실행 전략: {selection_result.execution_plan}
        
        다음을 결정하세요:
        1. 병렬 실행 가능한 에이전트 그룹
        2. 에이전트 간 의존성 관계
        3. 최적 실행 순서
        4. 예상 실행 시간
        
        JSON 형식으로 응답하세요:
        {{
            "parallel_groups": [
                ["agent_id1", "agent_id2"],
                ["agent_id3"]
            ],
            "dependencies": {{
                "agent_id3": ["agent_id1", "agent_id2"]
            }},
            "execution_order": [
                {{
                    "step": 1,
                    "agents": ["agent_id1", "agent_id2"],
                    "execution_type": "parallel",
                    "estimated_time": 30.0
                }},
                {{
                    "step": 2,
                    "agents": ["agent_id3"],
                    "execution_type": "sequential",
                    "estimated_time": 15.0
                }}
            ],
            "estimated_duration": 45.0,
            "optimization_notes": "최적화 과정 설명"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def select_fallback_agents(
        self, 
        failed_agents: List[str],
        original_criteria: AgentSelectionCriteria,
        available_agents: Dict[str, A2AAgentInfo]
    ) -> List[A2AAgentInfo]:
        """
        실패한 에이전트들의 대안 선택
        
        Args:
            failed_agents: 실패한 에이전트 ID 목록
            original_criteria: 원래 선택 기준
            available_agents: 사용 가능한 에이전트들
            
        Returns:
            대안 에이전트 목록
        """
        logger.info(f"Selecting fallback agents for failed: {failed_agents}")
        
        # 실패한 에이전트들 제외
        available_for_fallback = {
            agent_id: agent 
            for agent_id, agent in available_agents.items()
            if agent_id not in failed_agents
        }
        
        if not available_for_fallback:
            logger.warning("No agents available for fallback")
            return []
        
        prompt = f"""
        실패한 에이전트들을 대체할 수 있는 대안 에이전트들을 선택하세요.
        
        실패한 에이전트들: {failed_agents}
        원래 선택 기준: {asdict(original_criteria)}
        사용 가능한 대안 에이전트들: {json.dumps({
            agent_id: {
                'name': agent.name,
                'capabilities': agent.capabilities,
                'description': agent.description
            }
            for agent_id, agent in available_for_fallback.items()
        }, ensure_ascii=False)}
        
        다음을 고려하여 대안을 선택하세요:
        1. 실패한 에이전트의 역할을 대체할 수 있는 능력
        2. 기존 워크플로우와의 호환성
        3. 성능과 안정성
        
        JSON 형식으로 응답하세요:
        {{
            "fallback_agents": [
                {{
                    "agent_id": "대안 에이전트 ID",
                    "replaces": "대체하는 실패 에이전트 ID",
                    "rationale": "선택 이유"
                }}
            ],
            "workflow_adjustments": "워크플로우 조정 사항"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        fallback_data = self._parse_json_response(response)
        
        fallback_agents = []
        for fallback in fallback_data.get('fallback_agents', []):
            agent_id = fallback.get('agent_id')
            if agent_id in available_for_fallback:
                fallback_agents.append(available_for_fallback[agent_id])
        
        return fallback_agents
    
    def get_selection_statistics(self) -> Dict:
        """
        에이전트 선택 통계 조회
        
        Returns:
            선택 통계 정보
        """
        if not self.selection_history:
            return {'message': 'No selection history available'}
        
        # 최근 선택들 분석
        recent_selections = self.selection_history[-10:]  # 최근 10개
        
        agent_usage = {}
        confidence_scores = []
        
        for selection in recent_selections:
            confidence_scores.append(selection['confidence'])
            for agent_id in selection['selected_agents']:
                agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1
        
        return {
            'total_selections': len(self.selection_history),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'most_used_agents': sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            'agent_usage_distribution': agent_usage
        }
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        import json
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
            return {
                'raw_response': response,
                'parse_error': str(e)
            }