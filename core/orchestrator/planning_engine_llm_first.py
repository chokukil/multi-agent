"""
Planning Engine - 100% LLM First 지능형 분석 계획 수립
모든 의사결정을 LLM이 담당하는 순수 동적 시스템

Features:
- 100% LLM 기반 사용자 의도 분석
- LLM 기반 동적 에이전트 선택 및 우선순위 설정
- LLM 기반 실행 순서 최적화
- 하드코딩 제로 아키텍처
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import asyncio

from config.agents_config import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class UserIntent:
    """사용자 의도 분석 결과"""
    primary_goal: str  # 주요 목표
    data_type: str  # 데이터 유형
    analysis_type: List[str]  # 분석 종류
    complexity_level: str  # 복잡도 (low, medium, high)
    domain: Optional[str]  # 도메인 (semiconductor, finance, etc.)
    required_capabilities: List[str]  # 필요한 능력
    priority: int  # 우선순위 (1-5)

@dataclass
class AgentSelection:
    """에이전트 선택 결과"""
    agent_id: str
    confidence: float  # 선택 신뢰도 (0-1)
    reasoning: str  # 선택 이유
    expected_contribution: str  # 예상 기여도

@dataclass
class ExecutionSequence:
    """실행 순서 계획"""
    sequence: List[Dict[str, Any]]
    total_steps: int
    estimated_time: timedelta
    parallelizable_steps: List[int]  # 병렬 실행 가능한 단계

class PlanningEngineLLMFirst:
    """100% LLM First 지능형 분석 계획 수립"""
    
    def __init__(self):
        """순수 LLM 기반 초기화"""
        self.llm_client = None  # LLM 클라이언트는 필요시 초기화
        logger.info("🚀 PlanningEngineLLMFirst 초기화 - 100% LLM First 아키텍처")
    
    async def analyze_user_intent(self, query: str, data_context: Dict = None) -> UserIntent:
        """LLM 기반 사용자 의도 분석"""
        from core.universal_engine.llm_factory import LLMFactory
        
        if not self.llm_client:
            self.llm_client = LLMFactory.create_llm()
        
        prompt = f"""
        Analyze the user's intent from the following query:
        
        Query: "{query}"
        Data Context: {json.dumps(data_context, indent=2) if data_context else "None"}
        
        Extract and analyze:
        1. Primary goal of the query
        2. Type of data being analyzed
        3. Types of analysis requested (list all)
        4. Complexity level (low/medium/high)
        5. Domain (if specific domain is mentioned)
        6. Required capabilities to fulfill this request
        7. Priority level (1-5, where 5 is highest)
        
        Respond in JSON format:
        {{
            "primary_goal": "specific goal description",
            "data_type": "type of data",
            "analysis_types": ["type1", "type2", ...],
            "complexity_level": "low|medium|high",
            "domain": "domain name or null",
            "required_capabilities": ["capability1", "capability2", ...],
            "priority": 1-5
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 파싱
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # JSON 추출 시도
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("LLM response is not valid JSON")
            
            return UserIntent(
                primary_goal=data.get('primary_goal', 'General analysis'),
                data_type=data.get('data_type', 'unknown'),
                analysis_type=data.get('analysis_types', []),
                complexity_level=data.get('complexity_level', 'medium'),
                domain=data.get('domain'),
                required_capabilities=data.get('required_capabilities', []),
                priority=data.get('priority', 3)
            )
            
        except Exception as e:
            logger.error(f"사용자 의도 분석 실패: {e}")
            # 폴백 응답
            return UserIntent(
                primary_goal="Analyze the provided data",
                data_type="general",
                analysis_type=["general_analysis"],
                complexity_level="medium",
                domain=None,
                required_capabilities=["data_analysis"],
                priority=3
            )
    
    async def select_agents(self, intent: UserIntent, available_agents: List[AgentConfig]) -> List[AgentSelection]:
        """LLM 기반 동적 에이전트 선택"""
        from core.universal_engine.llm_factory import LLMFactory
        
        if not self.llm_client:
            self.llm_client = LLMFactory.create_llm()
        
        # 에이전트 정보 준비
        agents_info = []
        for agent in available_agents:
            agents_info.append({
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "required_inputs": agent.required_inputs,
                "supported_outputs": agent.supported_outputs
            })
        
        prompt = f"""
        Based on the user intent analysis, select the most appropriate agents for the task.
        
        User Intent:
        - Primary Goal: {intent.primary_goal}
        - Data Type: {intent.data_type}
        - Analysis Types: {intent.analysis_type}
        - Domain: {intent.domain}
        - Required Capabilities: {intent.required_capabilities}
        - Complexity: {intent.complexity_level}
        
        Available Agents:
        {json.dumps(agents_info, indent=2)}
        
        Select agents that best match the requirements. For each selected agent, provide:
        1. Agent ID
        2. Confidence score (0.0-1.0)
        3. Reasoning for selection
        4. Expected contribution to the goal
        
        Respond in JSON format:
        {{
            "selected_agents": [
                {{
                    "agent_id": "agent_id",
                    "confidence": 0.0-1.0,
                    "reasoning": "why this agent was selected",
                    "expected_contribution": "what this agent will contribute"
                }},
                ...
            ]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 파싱
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # JSON 추출 시도
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("LLM response is not valid JSON")
            
            selections = []
            for agent_data in data.get('selected_agents', []):
                selections.append(AgentSelection(
                    agent_id=agent_data['agent_id'],
                    confidence=float(agent_data['confidence']),
                    reasoning=agent_data['reasoning'],
                    expected_contribution=agent_data['expected_contribution']
                ))
            
            return selections
            
        except Exception as e:
            logger.error(f"에이전트 선택 실패: {e}")
            # 폴백: 첫 번째 에이전트 선택
            if available_agents:
                return [AgentSelection(
                    agent_id=available_agents[0].id,
                    confidence=0.5,
                    reasoning="Fallback selection",
                    expected_contribution="General analysis"
                )]
            return []
    
    async def create_execution_plan(self, intent: UserIntent, selected_agents: List[AgentSelection]) -> ExecutionSequence:
        """LLM 기반 실행 계획 수립"""
        from core.universal_engine.llm_factory import LLMFactory
        
        if not self.llm_client:
            self.llm_client = LLMFactory.create_llm()
        
        prompt = f"""
        Create an execution plan for the selected agents.
        
        User Intent:
        - Primary Goal: {intent.primary_goal}
        - Complexity: {intent.complexity_level}
        
        Selected Agents:
        {json.dumps([{
            'agent_id': agent.agent_id,
            'confidence': agent.confidence,
            'expected_contribution': agent.expected_contribution
        } for agent in selected_agents], indent=2)}
        
        Create an execution sequence that:
        1. Orders agents for optimal results
        2. Identifies which steps can run in parallel
        3. Estimates time for each step
        4. Calculates total execution time
        
        Consider:
        - Data dependencies (data must be loaded before analysis)
        - Logical flow (exploration before advanced analysis)
        - Parallel execution opportunities
        
        Respond in JSON format:
        {{
            "sequence": [
                {{
                    "step": 1,
                    "agent_id": "agent_id",
                    "task": "specific task description",
                    "estimated_time_seconds": 30,
                    "dependencies": []
                }},
                ...
            ],
            "parallelizable_steps": [2, 3],  // steps that can run in parallel
            "total_estimated_seconds": 180
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 파싱
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # JSON 추출 시도
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("LLM response is not valid JSON")
            
            sequence = data.get('sequence', [])
            parallelizable = data.get('parallelizable_steps', [])
            total_seconds = data.get('total_estimated_seconds', 180)
            
            return ExecutionSequence(
                sequence=sequence,
                total_steps=len(sequence),
                estimated_time=timedelta(seconds=total_seconds),
                parallelizable_steps=parallelizable
            )
            
        except Exception as e:
            logger.error(f"실행 계획 수립 실패: {e}")
            # 폴백: 순차 실행
            sequence = []
            for i, agent in enumerate(selected_agents):
                sequence.append({
                    'step': i + 1,
                    'agent_id': agent.agent_id,
                    'task': f"Execute {agent.agent_id}",
                    'estimated_time_seconds': 60,
                    'dependencies': [i] if i > 0 else []
                })
            
            return ExecutionSequence(
                sequence=sequence,
                total_steps=len(sequence),
                estimated_time=timedelta(seconds=60 * len(sequence)),
                parallelizable_steps=[]
            )
    
    async def create_analysis_plan(self, query: str, data_context: Dict = None) -> Tuple[ExecutionSequence, List[AgentSelection], UserIntent]:
        """통합 분석 계획 수립 - 100% LLM First"""
        logger.info(f"🧠 LLM First 분석 계획 수립: {query[:100]}...")
        
        # 1. 사용자 의도 분석
        intent = await self.analyze_user_intent(query, data_context)
        logger.info(f"📋 의도 분석 완료: {intent.primary_goal}")
        
        # 2. 사용 가능한 에이전트 가져오기
        available_agents = AgentConfig.get_all_agents()
        
        # 3. 에이전트 선택
        selected_agents = await self.select_agents(intent, available_agents)
        logger.info(f"🤖 {len(selected_agents)}개 에이전트 선택됨")
        
        # 4. 실행 계획 수립
        execution_plan = await self.create_execution_plan(intent, selected_agents)
        logger.info(f"📊 실행 계획 수립 완료: {execution_plan.total_steps}단계, 예상 시간: {execution_plan.estimated_time}")
        
        return execution_plan, selected_agents, intent