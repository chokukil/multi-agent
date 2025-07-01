#!/usr/bin/env python3
"""
A2A Orchestrator v7.0 - Universal Intelligent Orchestrator
완전한 통합 버전: 적응적 처리 + 동적 리플래닝 + 범용 LLM 시스템
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI DS Team 에이전트 포트 매핑
AGENT_PORTS = {
    "data_cleaning": 8306,
    "data_loader": 8307, 
    "data_visualization": 8308,
    "data_wrangling": 8309,
    "eda_tools": 8310,
    "feature_engineering": 8311,
    "h2o_modeling": 8312,
    "mlflow_tracking": 8313,
    "sql_database": 8314
}


class StreamingTaskUpdater(TaskUpdater):
    """실시간 스트리밍 업데이트 지원"""
    
    async def stream_update(self, content: str):
        """중간 결과 스트리밍"""
        await self.update_status(
            TaskState.working,
            message=self.new_agent_message(parts=[TextPart(text=content)])
        )
    
    async def stream_final_response(self, response: str):
        """최종 응답을 완료 상태로 전달"""
        await self.update_status(
            TaskState.completed,
            message=self.new_agent_message(parts=[TextPart(text=response)])
        )


class ExecutionMonitor:
    """실행 상태를 실시간으로 모니터링하고 리플래닝 필요성을 판단"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.execution_history = []
        self.original_plan = None
        self.current_context = {}
        
    async def should_replan(self, 
                          current_step: int,
                          agent_result: Dict,
                          remaining_steps: List[Dict],
                          user_intent: Dict) -> Dict:
        """리플래닝이 필요한지 판단"""
        
        # 1. 실패 기반 리플래닝
        if agent_result.get('status') == 'failed':
            return {
                'should_replan': True,
                'reason': 'agent_failure',
                'severity': 'high',
                'details': f"에이전트 실행 실패: {agent_result.get('error')}"
            }
        
        # 2. 품질 기반 리플래닝
        validation = agent_result.get('validation', {})
        if not validation.get('is_valid', True):
            return {
                'should_replan': True,
                'reason': 'quality_issue',
                'severity': 'medium',
                'details': f"결과 품질 부족: {validation.get('warnings')}"
            }
        
        # 3. 새로운 발견 기반 리플래닝
        new_insights = await self._extract_new_insights(agent_result)
        if new_insights.get('changes_direction'):
            return {
                'should_replan': True,
                'reason': 'new_discovery',
                'severity': 'medium',
                'details': new_insights.get('discovery')
            }
        
        # 4. 목표 달성도 기반 리플래닝
        achievement = await self._assess_goal_achievement(
            self.execution_history,
            user_intent
        )
        
        if achievement.get('percentage', 0) > 90:
            return {
                'should_replan': True,
                'reason': 'early_completion',
                'severity': 'low',
                'details': "목표가 조기에 달성되어 나머지 단계 생략 가능"
            }
        
        # 5. 효율성 기반 리플래닝
        if await self._found_better_path(current_step, remaining_steps):
            return {
                'should_replan': True,
                'reason': 'optimization',
                'severity': 'low',
                'details': "더 효율적인 경로 발견"
            }
        
        return {'should_replan': False}
    
    async def _extract_new_insights(self, agent_result: Dict) -> Dict:
        """결과에서 새로운 인사이트 추출"""
        if not self.openai_client:
            return {'changes_direction': False}
        
        insight_prompt = f"""
        다음 분석 결과를 검토하세요:
        {json.dumps(agent_result, ensure_ascii=False)[:1000]}
        
        이 결과가 다음을 포함하는지 판단하세요:
        1. 초기 가정과 다른 발견
        2. 분석 방향을 바꿔야 할 중요한 정보
        3. 추가 조사가 필요한 이상 패턴
        4. 다른 접근이 필요한 복잡성
        
        JSON 응답:
        {{
            "changes_direction": true/false,
            "discovery": "발견 내용",
            "implications": "이것이 의미하는 바",
            "recommended_action": "권장 조치"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": insight_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"인사이트 추출 실패: {e}")
            return {'changes_direction': False}
    
    async def _assess_goal_achievement(self, execution_history: List[Dict], 
                                     user_intent: Dict) -> Dict:
        """목표 달성도 평가"""
        if not self.openai_client or not execution_history:
            return {'percentage': 0}
        
        assessment_prompt = f"""
        사용자 목표: {user_intent.get('main_goal', '')}
        기대 결과: {json.dumps(user_intent.get('expected_outcomes', []), ensure_ascii=False)}
        
        현재까지 실행 결과 요약:
        {json.dumps([{
            'agent': h['agent'],
            'status': h['result'].get('status', 'unknown')
        } for h in execution_history[-5:]], ensure_ascii=False)}
        
        목표 달성도를 0-100%로 평가하세요.
        
        JSON 응답:
        {{
            "percentage": 0-100,
            "achieved_goals": ["달성된 목표들"],
            "remaining_goals": ["남은 목표들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"목표 달성도 평가 실패: {e}")
            return {'percentage': 0}
    
    async def _found_better_path(self, current_step: int, 
                               remaining_steps: List[Dict]) -> bool:
        """더 나은 경로가 있는지 확인"""
        # 간단한 휴리스틱: 남은 단계가 3개 이상이고 중복 가능성이 있으면 True
        if len(remaining_steps) >= 3:
            agents = [step.get('agent') for step in remaining_steps]
            if len(agents) != len(set(agents)):  # 중복 에이전트가 있으면
                return True
        return False


class ReplanningEngine:
    """동적으로 계획을 수정하는 리플래닝 엔진"""
    
    def __init__(self, openai_client, available_agents):
        self.openai_client = openai_client
        self.available_agents = available_agents
        self.replanning_history = []
        
    async def create_new_plan(self,
                            replan_reason: Dict,
                            current_state: Dict,
                            remaining_steps: List[Dict],
                            user_intent: Dict,
                            execution_history: List[Dict]) -> Dict:
        """현재 상황에 맞는 새로운 계획 생성"""
        
        strategy = self._determine_replanning_strategy(replan_reason)
        
        if strategy == 'recovery':
            return await self._create_recovery_plan(
                replan_reason,
                current_state,
                remaining_steps,
                user_intent
            )
        elif strategy == 'optimization':
            return await self._create_optimized_plan(
                current_state,
                remaining_steps,
                user_intent,
                execution_history
            )
        elif strategy == 'pivot':
            return await self._create_pivot_plan(
                replan_reason,
                current_state,
                user_intent,
                execution_history
            )
        elif strategy == 'completion':
            return await self._create_completion_plan(
                current_state,
                user_intent,
                execution_history
            )
        else:
            return {'steps': remaining_steps}
    
    def _determine_replanning_strategy(self, replan_reason: Dict) -> str:
        """리플래닝 전략 결정"""
        reason = replan_reason.get('reason', '')
        severity = replan_reason.get('severity', 'low')
        
        if reason == 'agent_failure' and severity == 'high':
            return 'recovery'
        elif reason == 'optimization':
            return 'optimization'
        elif reason == 'new_discovery':
            return 'pivot'
        elif reason == 'early_completion':
            return 'completion'
        else:
            return 'continue'
    
    async def _create_recovery_plan(self,
                                  replan_reason: Dict,
                                  current_state: Dict,
                                  remaining_steps: List[Dict],
                                  user_intent: Dict) -> Dict:
        """실패 복구를 위한 계획"""
        
        if not self.openai_client:
            return self._create_fallback_recovery_plan(replan_reason, remaining_steps)
        
        recovery_prompt = f"""
        실행 실패 상황:
        - 실패 이유: {replan_reason['details']}
        - 현재 상태: {json.dumps(current_state, ensure_ascii=False)[:500]}
        - 사용자 목표: {user_intent['main_goal']}
        
        사용 가능한 에이전트:
        {json.dumps(list(self.available_agents.keys()), ensure_ascii=False)}
        
        다음 중 하나를 선택하여 복구 계획을 수립하세요:
        1. 대체 에이전트로 같은 작업 시도
        2. 다른 접근 방법으로 우회
        3. 부분적 결과로 진행
        4. 추가 데이터 수집 후 재시도
        
        JSON 형식으로 새로운 계획을 작성하세요:
        {{
            "recovery_strategy": "선택한 전략",
            "reasoning": "이유",
            "steps": [
                {{
                    "agent": "에이전트명",
                    "purpose": "목적",
                    "comprehensive_instructions": "구체적 작업",
                    "fallback": "실패시 대안"
                }}
            ]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": recovery_prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
                timeout=60.0
            )
            
            recovery_plan = json.loads(response.choices[0].message.content)
            self._log_replanning('recovery', replan_reason, recovery_plan)
            return recovery_plan
            
        except Exception as e:
            logger.error(f"복구 계획 생성 실패: {e}")
            return self._create_fallback_recovery_plan(replan_reason, remaining_steps)
    
    def _create_fallback_recovery_plan(self, replan_reason: Dict, 
                                     remaining_steps: List[Dict]) -> Dict:
        """폴백 복구 계획"""
        # 실패한 에이전트를 건너뛰고 다음 단계로
        return {
            'recovery_strategy': 'skip_failed',
            'reasoning': '실패한 단계를 건너뛰고 진행',
            'steps': remaining_steps
        }
    
    async def _create_optimized_plan(self,
                                   current_state: Dict,
                                   remaining_steps: List[Dict],
                                   user_intent: Dict,
                                   execution_history: List[Dict]) -> Dict:
        """더 효율적인 경로로 최적화"""
        
        if not self.openai_client:
            return {'steps': remaining_steps}
        
        optimization_prompt = f"""
        현재까지의 실행 결과를 보고 더 효율적인 경로를 찾으세요.
        
        실행 이력 요약:
        {json.dumps([{
            'agent': h['agent'],
            'status': h['result'].get('status', 'unknown')
        } for h in execution_history[-5:]], ensure_ascii=False)}
        
        남은 단계:
        {json.dumps(remaining_steps, ensure_ascii=False)}
        
        사용자 목표:
        {user_intent.get('main_goal', '')}
        
        최적화 기준:
        1. 불필요한 단계 제거
        2. 병렬 실행 가능한 작업 식별
        3. 더 적합한 에이전트로 교체
        4. 중복 작업 통합
        
        JSON 형식으로 최적화된 계획을 작성하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": optimization_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            optimized_plan = json.loads(response.choices[0].message.content)
            self._log_replanning('optimization', {'reason': '효율성 개선'}, optimized_plan)
            return optimized_plan
            
        except Exception as e:
            logger.warning(f"최적화 실패: {e}")
            return {'steps': remaining_steps}
    
    async def _create_pivot_plan(self,
                               replan_reason: Dict,
                               current_state: Dict,
                               user_intent: Dict,
                               execution_history: List[Dict]) -> Dict:
        """새로운 발견에 따른 방향 전환"""
        
        if not self.openai_client:
            return {'steps': []}  # 안전하게 종료
        
        pivot_prompt = f"""
        새로운 발견: {replan_reason['details']}
        
        이 발견을 바탕으로 분석 방향을 조정하세요.
        사용자 목표: {user_intent.get('main_goal', '')}
        
        새로운 계획을 수립하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": pivot_prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
                timeout=60.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"방향 전환 실패: {e}")
            return {'steps': []}
    
    async def _create_completion_plan(self,
                                    current_state: Dict,
                                    user_intent: Dict,
                                    execution_history: List[Dict]) -> Dict:
        """조기 완료를 위한 마무리 계획"""
        return {
            'completion_strategy': 'early_finish',
            'reasoning': '충분한 결과를 얻어 조기 완료',
            'steps': []  # 추가 단계 없음
        }
    
    def _log_replanning(self, strategy: str, reason: Dict, new_plan: Dict):
        """리플래닝 이력 기록"""
        self.replanning_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'reason': reason,
            'new_steps': len(new_plan.get('steps', []))
        })


class UniversalIntelligentOrchestrator(AgentExecutor):
    """LLM 기반 범용 지능형 오케스트레이터 - 적응적 처리 + 동적 리플래닝"""
    
    def __init__(self):
        # OpenAI 클라이언트 초기화
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 Universal Intelligent Orchestrator v7.0 initialized")
            else:
                self.openai_client = None
                logger.info("📊 Standard Orchestrator (No LLM)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        self.available_agents = {}
        self.agent_capabilities = {}
        self.execution_monitor = None
        self.replanning_engine = None
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """🎯 메인 실행 함수 - 적응적 처리 + 리플래닝"""
        task_updater = StreamingTaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            logger.info(f"🎯 Universal Orchestrator Processing: {user_input}")
            
            if not user_input:
                user_input = "Please provide an analysis request."
            
            # 🎯 Step 1: 요청 복잡도 평가
            complexity = await self._assess_request_complexity(user_input)
            logger.info(f"📊 Request complexity: {complexity['level']}")
            
            # 🎯 Step 2: 복잡도에 따른 적응적 처리
            if complexity['level'] == 'simple':
                # 즉답 가능한 경우 - 리플래닝 불필요
                await self._handle_simple_request(user_input, task_updater)
                
            elif complexity['level'] == 'single_agent':
                # 단일 에이전트로 처리 - 간단한 리플래닝만
                await self._handle_single_agent_request_with_recovery(
                    user_input, 
                    complexity['recommended_agent'],
                    task_updater
                )
                
            else:  # complex
                # 복잡한 요청 - 전체 기능 활성화
                await self._handle_complex_request_with_full_features(
                    user_input, 
                    task_updater
                )
                
        except Exception as e:
            error_msg = f"Orchestrator execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=error_msg)])
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")
    
    async def _assess_request_complexity(self, user_input: str) -> Dict:
        """요청 복잡도를 판단하는 지능형 분류기"""
        
        if not self.openai_client:
            return {'level': 'complex', 'reasoning': '기본값'}
        
        assessment_prompt = f"""
        다음 사용자 요청의 복잡도를 평가하세요:
        "{user_input}"
        
        평가 기준:
        1. **simple**: 즉답 가능 (정의, 개념 설명, 간단한 사실 확인)
        2. **single_agent**: 한 에이전트로 충분 (단일 작업)
        3. **complex**: 여러 에이전트 협업 필요 (다단계 분석)
        
        판단 요소:
        - 요청된 작업의 수
        - 데이터 처리 필요 여부
        - 분석 깊이
        - 도메인 전문성 필요도
        
        JSON 응답:
        {{
            "level": "simple/single_agent/complex",
            "reasoning": "판단 근거",
            "recommended_agent": "single_agent인 경우 추천 에이전트",
            "key_requirements": ["핵심 요구사항들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"복잡도 평가 실패: {e}")
            return {'level': 'complex', 'reasoning': '평가 실패로 안전한 경로 선택'}
    
    async def _handle_simple_request(self, user_input: str, task_updater: StreamingTaskUpdater):
        """간단한 요청 즉답 처리"""
        await task_updater.stream_update("💬 간단한 질문으로 판단되어 즉시 답변드립니다...")
        
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": "당신은 도움이 되는 AI 어시스턴트입니다. 간결하고 정확하게 답변하세요."
                    }, {
                        "role": "user",
                        "content": user_input
                    }],
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=30.0
                )
                
                answer = response.choices[0].message.content
                await task_updater.update_status(
                    TaskState.completed,
                    message=task_updater.new_agent_message(parts=[TextPart(text=answer)])
                )
                
            except Exception as e:
                logger.error(f"Simple request handling failed: {e}")
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text=f"답변 생성 중 오류가 발생했습니다: {str(e)}")]
                    )
                )
        else:
            answer = "죄송합니다. LLM이 설정되지 않아 간단한 답변을 제공할 수 없습니다."
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=answer)])
            )
    
    async def _handle_single_agent_request_with_recovery(self, 
                                                        user_input: str,
                                                        agent_name: str,
                                                        task_updater: StreamingTaskUpdater):
        """단일 에이전트 처리 with 실패 복구"""
        await task_updater.stream_update(f"🤖 {agent_name} 에이전트로 처리 중...")
        
        # 에이전트 발견
        self.available_agents = await self._discover_agents()
        
        if agent_name not in self.available_agents:
            # 대체 에이전트 찾기
            alternative = await self._find_alternative_agent(user_input, self.available_agents)
            if alternative:
                agent_name = alternative
                await task_updater.stream_update(f"🔄 대체 에이전트 {agent_name} 사용")
            else:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text="적합한 에이전트를 찾을 수 없습니다.")]
                    )
                )
                return
        
        # 사용자 의도 추출
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        # 에이전트 능력 파악
        agent_capability = await self._discover_agent_capabilities(agent_name)
        
        # 정밀한 지시 생성
        instruction = await self._create_precise_agent_instruction(
            agent_name,
            user_intent,
            agent_capability,
            {}
        )
        
        # 실행 with 재시도
        max_retries = 2
        for attempt in range(max_retries):
            result = await self._execute_agent_with_comprehensive_instructions(
                agent_name,
                {'comprehensive_instructions': instruction},
                {},
                {}
            )
            
            # 검증
            validation = await self._validate_agent_response(agent_name, result)
            result['validation'] = validation
            
            if validation['is_valid']:
                # 성공 - 결과 생성
                final_response = await self._create_evidence_based_response(
                    user_input,
                    {agent_name: result},
                    user_intent
                )
                
                await task_updater.update_status(
                    TaskState.completed,
                    message=task_updater.new_agent_message(parts=[TextPart(text=final_response)])
                )
                return
            else:
                # 실패 - 재시도 또는 복구
                if attempt < max_retries - 1:
                    await task_updater.stream_update(
                        f"⚠️ 결과 검증 실패. 재시도 중... ({attempt + 2}/{max_retries})"
                    )
                    # 지시 개선
                    instruction = await self._improve_instruction_based_on_failure(
                        instruction,
                        validation['warnings'],
                        user_intent
                    )
                else:
                    # 최종 실패 - 대체 방안
                    await task_updater.stream_update("🔄 대체 방안으로 전환...")
                    await self._handle_with_alternative_approach(
                        user_input,
                        user_intent,
                        task_updater
                    )
    
    async def _handle_complex_request_with_full_features(self, 
                                                        user_input: str,
                                                        task_updater: StreamingTaskUpdater):
        """복잡한 요청 처리 - 모든 기능 활성화"""
        
        # 초기화
        self.execution_monitor = ExecutionMonitor(self.openai_client)
        self.replanning_engine = ReplanningEngine(self.openai_client, self.available_agents)
        
        # 🎯 Phase 1: 깊이 있는 요청 분석
        await task_updater.stream_update("🧠 요청을 깊이 분석하고 있습니다...")
        
        # 1.1 요청 깊이 분석
        request_analysis = await self._analyze_request_depth(user_input)
        
        # 1.2 사용자 의도 정밀 추출
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        # 1.3 적응적 컨텍스트 구축
        adaptive_context = await self._build_adaptive_context(user_input, request_analysis)
        
        # 1.4 요청 확장 (필요시)
        expanded_request = await self._expand_simple_requests(user_input, request_analysis)
        
        # 🎯 Phase 2: 에이전트 준비 및 능력 파악
        await task_updater.stream_update("🔍 AI DS Team 에이전트 능력을 파악하고 있습니다...")
        
        # 2.1 에이전트 발견
        self.available_agents = await self._discover_agents()
        
        if not self.available_agents:
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="❌ 사용 가능한 에이전트를 찾을 수 없습니다.")]
                )
            )
            return
        
        # 2.2 각 에이전트 능력 상세 파악
        for agent_name in self.available_agents:
            self.agent_capabilities[agent_name] = await self._discover_agent_capabilities(agent_name)
        
        await task_updater.stream_update(
            f"✅ {len(self.available_agents)}개 에이전트 준비 완료"
        )
        
        # 🎯 Phase 3: 초기 계획 수립
        await task_updater.stream_update("📋 최적의 실행 계획을 수립하고 있습니다...")
        
        initial_plan = await self._create_comprehensive_execution_plan(
            expanded_request,
            user_intent,
            self.available_agents,
            self.agent_capabilities,
            adaptive_context
        )
        
        if not initial_plan or not initial_plan.get('steps'):
            initial_plan = self._create_fallback_plan(self.available_agents)
        
        # 계획 표시
        plan_display = self._create_beautiful_plan_display(initial_plan, user_intent)
        await task_updater.stream_update(plan_display)
        
        # 계획 아티팩트 저장
        await self._save_plan_artifact(initial_plan, task_updater)
        
        # 🎯 Phase 4: 적응적 실행 with 리플래닝
        await task_updater.stream_update("🚀 실행을 시작합니다...")
        
        execution_result = await self._execute_with_adaptive_replanning(
            initial_plan,
            user_intent,
            adaptive_context,
            task_updater
        )
        
        # 🎯 Phase 5: 결과 종합 및 검증
        await task_updater.stream_update("🎨 분석 결과를 종합하고 있습니다...")
        
        # 5.1 콘텐츠 풍부도 평가
        content_assessment = await self._assess_content_richness(execution_result['results'])
        
        # 5.2 증거 기반 응답 생성
        final_response = await self._create_intelligent_final_response(
            user_input,
            user_intent,
            execution_result,
            content_assessment,
            adaptive_context
        )
        
        # 5.3 의도 매칭 검증
        if not await self._verify_response_matches_intent(final_response, user_intent):
            final_response = await self._regenerate_response_for_intent(
                user_input,
                execution_result['results'],
                user_intent
            )
        
        # 🎯 Phase 6: 최종 전달
        await task_updater.stream_update("🎉 분석이 완료되었습니다!")
        
        # 실행 요약 아티팩트
        await self._save_execution_summary(
            execution_result,
            content_assessment,
            task_updater
        )
        
        # 최종 응답 전달
        await task_updater.update_status(
            TaskState.completed,
            message=task_updater.new_agent_message(parts=[TextPart(text=final_response)])
        )
    
    async def _execute_with_adaptive_replanning(self,
                                              initial_plan: Dict,
                                              user_intent: Dict,
                                              context: Dict,
                                              task_updater: StreamingTaskUpdater) -> Dict:
        """적응적 리플래닝을 포함한 실행"""
        
        current_plan = initial_plan.copy()
        execution_history = []
        validated_results = {}
        replanning_count = 0
        max_replanning = 5
        
        step_index = 0
        
        while step_index < len(current_plan['steps']) and replanning_count < max_replanning:
            current_step = current_plan['steps'][step_index]
            agent_name = current_step.get('agent', 'unknown')
            
            # 진행 상황 업데이트
            await task_updater.stream_update(
                f"🔄 단계 {step_index + 1}/{len(current_plan['steps'])}: "
                f"{agent_name} 실행 중..."
            )
            
            # 정밀한 지시 생성 (이전 결과 반영)
            instruction = await self._create_contextual_instruction(
                agent_name,
                current_step,
                user_intent,
                self.agent_capabilities.get(agent_name, {}),
                validated_results,
                context
            )
            
            # 에이전트 실행
            result = await self._execute_agent_with_comprehensive_instructions(
                agent_name,
                {'comprehensive_instructions': instruction},
                context,
                validated_results
            )
            
            # 응답 검증
            validation = await self._validate_agent_response(agent_name, result)
            result['validation'] = validation
            
            # 실행 이력 기록
            execution_history.append({
                'step': step_index,
                'agent': agent_name,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            # 리플래닝 필요성 평가
            remaining_steps = current_plan['steps'][step_index + 1:]
            replan_decision = await self.execution_monitor.should_replan(
                step_index,
                result,
                remaining_steps,
                user_intent
            )
            
            if replan_decision['should_replan']:
                replanning_count += 1
                await task_updater.stream_update(
                    f"🔄 리플래닝 {replanning_count}/{max_replanning}: "
                    f"{replan_decision['reason']} - {replan_decision['details']}"
                )
                
                # 새로운 계획 생성
                new_plan = await self.replanning_engine.create_new_plan(
                    replan_decision,
                    {
                        'step_index': step_index,
                        'history': execution_history,
                        'current_results': validated_results
                    },
                    remaining_steps,
                    user_intent,
                    execution_history
                )
                
                if new_plan and new_plan.get('steps'):
                    # 계획 업데이트
                    current_plan['steps'] = (
                        current_plan['steps'][:step_index + 1] + 
                        new_plan['steps']
                    )
                    
                    await task_updater.stream_update(
                        f"✅ 리플래닝 완료: {len(new_plan['steps'])}개의 새로운 단계"
                    )
                    
                    # 조기 완료 체크
                    if replan_decision['reason'] == 'early_completion':
                        break
            
            # 결과 저장 (검증 통과한 것만)
            if validation['is_valid']:
                validated_results[agent_name] = result
            else:
                await task_updater.stream_update(
                    f"⚠️ {agent_name} 결과가 검증을 통과하지 못했습니다."
                )
            
            step_index += 1
            
            # 주기적 진행률 체크 (3단계마다)
            if step_index % 3 == 0:
                progress = await self._check_overall_progress(
                    execution_history,
                    user_intent,
                    current_plan
                )
                
                if progress.get('should_conclude'):
                    await task_updater.stream_update(
                        f"✅ 목표 달성도 {progress['achievement_percentage']}%로 조기 완료합니다."
                    )
                    break
        
        return {
            'results': validated_results,
            'history': execution_history,
            'final_plan': current_plan,
            'replanning_count': replanning_count,
            'completion_reason': 'normal' if step_index >= len(current_plan['steps']) else 'early'
        }
    
    async def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """A2A 표준 에이전트 발견"""
        available_agents = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for agent_name, port in AGENT_PORTS.items():
                try:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        available_agents[agent_name] = {
                            "name": agent_card.get("name", agent_name),
                            "url": f"http://localhost:{port}",
                            "port": port,
                            "description": agent_card.get("description", ""),
                            "status": "available"
                        }
                        logger.info(f"✅ {agent_name} agent discovered on port {port}")
                except Exception as e:
                    logger.warning(f"⚠️ {agent_name} agent on port {port} not available: {e}")
        
        logger.info(f"🔍 Total discovered agents: {len(available_agents)}")
        return available_agents
    
    async def _discover_agent_capabilities(self, agent_name: str) -> Dict:
        """에이전트의 실제 능력을 A2A 프로토콜로 조회"""
        
        agent_info = self.available_agents.get(agent_name, {})
        agent_url = agent_info.get('url')
        
        if not agent_url:
            return {}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{agent_url}/.well-known/agent.json")
                
                if response.status_code == 200:
                    agent_card = response.json()
                    
                    skills = agent_card.get('skills', [])
                    capabilities = {
                        'skills': [
                            {
                                'id': skill.get('id'),
                                'name': skill.get('name'),
                                'description': skill.get('description'),
                                'tags': skill.get('tags', [])
                            }
                            for skill in skills
                        ],
                        'input_modes': agent_card.get('defaultInputModes', []),
                        'output_modes': agent_card.get('defaultOutputModes', []),
                        'capabilities': agent_card.get('capabilities', {})
                    }
                    
                    return capabilities
                    
        except Exception as e:
            logger.warning(f"에이전트 능력 조회 실패 {agent_name}: {e}")
        
        return {}
    
    async def _analyze_request_depth(self, user_input: str) -> Dict:
        """요청의 깊이와 특성을 LLM이 자동 분석"""
        
        if not self.openai_client:
            return {
                "detail_level": 5,
                "has_role_description": False,
                "role_description": "",
                "explicit_requirements": ["기본 분석"],
                "implicit_needs": ["데이터 이해"],
                "suggested_response_depth": "moderate",
                "needs_clarification": []
            }
        
        analysis_prompt = f"""
        다음 사용자 요청을 분석하세요:
        "{user_input}"
        
        분석할 내용:
        1. 요청의 구체성 수준 (1-10)
        2. 역할이나 전문성 언급 여부
        3. 명시적 요구사항 vs 암시적 니즈
        4. 예상되는 답변의 깊이
        5. 추가 컨텍스트가 필요한지 여부
        
        JSON 응답:
        {{
            "detail_level": 1-10,
            "has_role_description": true/false,
            "role_description": "있다면 어떤 역할인지",
            "explicit_requirements": ["명시적으로 요청한 것들"],
            "implicit_needs": ["맥락상 필요할 것으로 보이는 것들"],
            "suggested_response_depth": "brief/moderate/comprehensive",
            "needs_clarification": ["명확히 할 필요가 있는 부분들"],
            "explicitly_wants_brief": true/false
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"📊 Request depth analysis: level {analysis.get('detail_level', 5)}/10")
            return analysis
            
        except Exception as e:
            logger.warning(f"Request depth analysis failed: {e}")
            return {
                "detail_level": 5,
                "has_role_description": False,
                "role_description": "",
                "explicit_requirements": ["기본 분석"],
                "implicit_needs": ["데이터 이해"],
                "suggested_response_depth": "moderate",
                "needs_clarification": []
            }
    
    async def _extract_user_intent_precisely(self, user_input: str) -> Dict:
        """사용자 의도를 정밀하게 추출"""
        
        if not self.openai_client:
            return {
                'main_goal': '데이터 분석',
                'action_type': 'analyze',
                'specific_requirements': [],
                'expected_outcomes': ['분석 결과']
            }
        
        intent_prompt = f"""
        사용자 입력을 정밀 분석하세요:
        "{user_input}"
        
        추출해야 할 정보:
        
        1. **action_type**: 사용자가 원하는 행동 유형
           - analyze: 분석하기
           - verify: 검증/확인하기
           - recommend: 추천하기
           - diagnose: 진단하기
           - predict: 예측하기
           - compare: 비교하기
           - explain: 설명하기
        
        2. **main_goal**: 한 문장으로 요약한 주요 목표
        
        3. **specific_requirements**: 구체적 요구사항 목록
           - 분석해야 할 변수
           - 확인해야 할 조건
           - 포함되어야 할 내용
        
        4. **expected_outcomes**: 기대하는 결과물
           - 수치적 결과
           - 시각화
           - 권장사항
           - 진단 결과
        
        5. **domain_context**: 도메인 특화 컨텍스트
           - 전문 용어
           - 업계 기준
           - 특별한 제약사항
        
        6. **priority_aspects**: 우선순위가 높은 측면들
        
        JSON 형식으로 정확히 응답하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": intent_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=60.0
            )
            
            intent = json.loads(response.choices[0].message.content)
            logger.info(f"🎯 사용자 의도 추출: {intent['action_type']} - {intent['main_goal']}")
            return intent
            
        except Exception as e:
            logger.warning(f"의도 추출 실패: {e}")
            return {
                'main_goal': user_input,
                'action_type': 'analyze',
                'specific_requirements': [],
                'expected_outcomes': ['분석 결과']
            }
    
    async def _build_adaptive_context(self, user_input: str, request_analysis: Dict) -> Dict:
        """요청 분석에 따라 적응적 컨텍스트 구축"""
        
        if not self.openai_client:
            return {
                "adopted_perspective": "일반 데이터 분석가",
                "analysis_approach": "표준 분석",
                "response_style": "professional",
                "focus_areas": ["기본 통계"],
                "depth_strategy": "moderate"
            }
        
        context_prompt = f"""
        사용자 요청: "{user_input}"
        요청 분석: {json.dumps(request_analysis, ensure_ascii=False)}
        
        이 정보를 바탕으로 적절한 분석 컨텍스트를 구축하세요:
        
        1. 역할이 명시되었다면: 그 역할의 관점 채택
        2. 역할이 없다면: 요청 내용에서 적절한 전문성 수준 추론
        3. 상세한 요청이면: 깊이 있는 분석 준비
        4. 간단한 요청이면: 핵심만 간결하게
        
        다음 형식으로 컨텍스트를 만드세요:
        {{
            "adopted_perspective": "채택할 관점",
            "analysis_approach": "분석 접근 방식",
            "response_style": "답변 스타일",
            "focus_areas": ["집중할 영역들"],
            "depth_strategy": "얼마나 깊이 들어갈지",
            "domain": "추론된 도메인",
            "expertise_level": "필요한 전문성 수준"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": context_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            context = json.loads(response.choices[0].message.content)
            logger.info(f"🎭 Adaptive context: {context.get('adopted_perspective', 'unknown')}")
            return context
            
        except Exception as e:
            logger.warning(f"Adaptive context building failed: {e}")
            return {
                "adopted_perspective": "일반 데이터 분석가",
                "analysis_approach": "표준 분석",
                "response_style": "professional",
                "focus_areas": ["기본 통계"],
                "depth_strategy": "moderate"
            }
    
    async def _expand_simple_requests(self, user_input: str, request_analysis: Dict) -> str:
        """간단한 요청을 지능적으로 확장 (필요한 경우만)"""
        
        if request_analysis['detail_level'] >= 7:
            return user_input
        
        if request_analysis['needs_clarification'] and not self.openai_client:
            return user_input
        
        if request_analysis['needs_clarification']:
            expansion_prompt = f"""
            사용자의 간단한 요청: "{user_input}"
            
            이 요청에서 사용자가 암묵적으로 알고 싶어할 수 있는 것들을 추론하세요.
            하지만 과도하게 확장하지 말고, 맥락상 합리적인 수준에서만 보완하세요.
            
            예시:
            - "데이터 분석해줘" → 기본 통계, 패턴, 이상치 정도
            - "불량 원인 찾아줘" → 데이터 기반 원인 추정, 가능성 순위
            
            추론된 상세 요구사항:
            """
            
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    timeout=60.0
                )
                
                expanded = response.choices[0].message.content
                logger.info(f"📈 Request expanded: {len(expanded)} chars")
                return expanded
                
            except Exception as e:
                logger.warning(f"Request expansion failed: {e}")
                return user_input
        
        return user_input
    
    async def _create_comprehensive_execution_plan(self, 
                                                 expanded_request: str,
                                                 user_intent: Dict,
                                                 available_agents: Dict,
                                                 agent_capabilities: Dict,
                                                 adaptive_context: Dict) -> Dict:
        """완전 LLM 기반 동적 계획 생성"""
        
        if not self.openai_client:
            return self._create_fallback_plan(available_agents)
        
        # 에이전트들의 상세 정보 구조화
        agents_details = {}
        for name, info in available_agents.items():
            agents_details[name] = {
                "description": info.get('description', ''),
                "capabilities": agent_capabilities.get(name, {}).get('skills', []),
                "typical_use_cases": self._infer_use_cases(name)
            }
        
        planning_prompt = f"""당신은 데이터 분석 워크플로우 설계 전문가입니다.
사용자의 요청을 분석하고, 가장 효과적인 에이전트 실행 계획을 수립해야 합니다.

## 📋 사용자 요청 분석
원본 요청: {expanded_request}
의도 분석: {json.dumps(user_intent, ensure_ascii=False, indent=2)}
컨텍스트: {json.dumps(adaptive_context, ensure_ascii=False, indent=2)}

## 🤖 사용 가능한 에이전트들
{json.dumps(agents_details, ensure_ascii=False, indent=2)}

## 🎯 계획 수립 지침
1. **요청 중심 접근**: 사용자가 원하는 결과에 집중하여 필요한 에이전트만 선택
2. **논리적 순서**: 데이터 흐름과 의존성을 고려한 순서 결정
3. **효율성 최적화**: 불필요한 단계 제거, 핵심 분석에 집중
4. **도메인 적응**: {adaptive_context.get('domain', '일반')} 도메인 특성 반영
5. **사용자 수준 고려**: {adaptive_context.get('expertise_level', '일반')} 수준에 맞는 분석 깊이

## 🚀 동적 에이전트 선택 기준
- 사용자 질문의 핵심 의도가 무엇인가?
- 어떤 종류의 분석이 실제로 필요한가?
- 각 에이전트가 제공할 수 있는 가치는 무엇인가?
- 최소한의 단계로 최대한의 인사이트를 얻으려면?

다음 JSON 형식으로 응답하세요:
{{
    "execution_strategy": "이 요청에 대한 전체적 분석 전략",
    "agent_selection_reasoning": "선택된 에이전트들과 그 이유",
    "steps": [
        {{
            "step_number": 1,
            "agent": "선택된_에이전트명",
            "purpose": "이 단계의 구체적 목적",
            "comprehensive_instructions": "에이전트에게 전달할 상세한 작업 지시 (도메인 컨텍스트 포함)",
            "expected_deliverables": {{
                "minimum": "최소한 필요한 결과",
                "standard": "표준적인 결과",
                "exceptional": "탁월한 결과"
            }},
            "success_criteria": "성공 판단 기준",
            "context_for_next": "다음 단계로 전달할 핵심 정보"
        }}
    ],
    "final_synthesis_strategy": "모든 결과를 어떻게 종합할 것인가",
    "potential_insights": "이 계획으로 얻을 수 있는 예상 인사이트들"
}}

중요: 템플릿이나 고정된 패턴을 사용하지 말고, 이 특정 요청에 최적화된 계획을 수립하세요."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 데이터 분석 워크플로우 최적화 전문가입니다. 각 요청의 고유한 특성을 파악하고, 가장 효율적이고 효과적인 분석 경로를 설계합니다."
                    },
                    {"role": "user", "content": planning_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=3000,
                timeout=90.0
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            # 계획 검증 및 보정
            validated_plan = self._validate_and_enhance_plan(plan, available_agents)
            
            return validated_plan
            
        except Exception as e:
            logger.warning(f"Dynamic planning failed: {e}")
            return self._create_fallback_plan(available_agents)
    
    def _validate_and_enhance_plan(self, plan: Dict, available_agents: Dict) -> Dict:
        """생성된 계획을 검증하고 보강"""
        try:
            if not plan.get('steps'):
                logger.warning("Plan has no steps, using fallback")
                return self._create_fallback_plan(available_agents)
            
            # 에이전트 존재 여부 확인 및 보정
            valid_steps = []
            for i, step in enumerate(plan.get('steps', [])):
                agent_name = step.get('agent', '')
                if agent_name in available_agents:
                    # 필수 필드 보완
                    enhanced_step = {
                        "step_number": step.get('step_number', i + 1),
                        "agent": agent_name,
                        "purpose": step.get('purpose', f'{agent_name} 분석 수행'),
                        "comprehensive_instructions": step.get(
                            'comprehensive_instructions', 
                            step.get('enriched_task', f'{agent_name} 작업 수행')
                        ),
                        "expected_deliverables": step.get('expected_deliverables', {
                            "minimum": "기본 분석 결과",
                            "standard": "상세 분석 결과",
                            "exceptional": "심층 인사이트"
                        }),
                        "success_criteria": step.get('success_criteria', '분석 완료'),
                        "context_for_next": step.get('context_for_next', ['분석 결과'])
                    }
                    valid_steps.append(enhanced_step)
                else:
                    logger.warning(f"Agent {agent_name} not available, skipping step")
            
            if not valid_steps:
                logger.warning("No valid steps after validation, using fallback")
                return self._create_fallback_plan(available_agents)
            
            # 향상된 계획 반환
            enhanced_plan = {
                "execution_strategy": plan.get('execution_strategy', '사용자 요청에 최적화된 분석'),
                "agent_selection_reasoning": plan.get('agent_selection_reasoning', '요청 기반 선택'),
                "steps": valid_steps,
                "final_synthesis_strategy": plan.get('final_synthesis_strategy', '결과 종합'),
                "potential_insights": plan.get('potential_insights', ['데이터 기반 인사이트'])
            }
            
            return enhanced_plan
            
        except Exception as e:
            logger.error(f"Plan validation failed: {e}")
            return self._create_fallback_plan(available_agents)
    
    async def _create_precise_agent_instruction(self, 
                                              agent_name: str,
                                              user_intent: Dict,
                                              agent_capabilities: Dict,
                                              context: Dict) -> str:
        """에이전트 능력에 맞춘 정밀한 지시 생성"""
        
        if not self.openai_client:
            return f"{agent_name}을 사용하여 {user_intent.get('main_goal', '분석')}을 수행하세요."
        
        instruction_prompt = f"""
        에이전트: {agent_name}
        에이전트 능력: {json.dumps(agent_capabilities, ensure_ascii=False)}
        
        사용자 의도:
        - 주요 목표: {user_intent.get('main_goal')}
        - 구체적 요구사항: {user_intent.get('specific_requirements')}
        - 기대 결과: {user_intent.get('expected_outcomes')}
        
        컨텍스트:
        - 도메인: {context.get('domain')}
        - 이전 결과: {context.get('previous_insights')}
        
        이 에이전트의 능력을 최대한 활용하여 사용자 의도를 달성할 수 있는
        매우 구체적이고 상세한 작업 지시를 작성하세요.
        
        포함해야 할 내용:
        1. 정확히 무엇을 분석/처리해야 하는지
        2. 어떤 방법론이나 기법을 사용해야 하는지
        3. 결과물의 형태와 포함되어야 할 정보
        4. 주의사항이나 제약사항
        5. 다음 단계를 위해 보존해야 할 정보
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0.3,
                max_tokens=1500,
                timeout=60.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"정밀 지시 생성 실패: {e}")
            return f"{agent_name}을 사용하여 {user_intent.get('main_goal', '분석')}을 수행하세요."
    
    async def _execute_agent_with_comprehensive_instructions(self, 
                                                           agent_name: str, 
                                                           step: Dict,
                                                           adaptive_context: Dict,
                                                           previous_results: Dict) -> Dict:
        """종합적 지시사항으로 에이전트 실행"""
        
        if agent_name not in self.available_agents:
            return {
                'status': 'failed',
                'error': f'Agent {agent_name} not available',
                'summary': f'에이전트 {agent_name}를 찾을 수 없습니다'
            }
        
        comprehensive_instructions = step.get(
            'comprehensive_instructions', 
            f'{agent_name}에 대한 분석을 수행하세요.'
        )
        
        logger.info(f"🔍 Sending to {agent_name}: {comprehensive_instructions[:200]}...")
        
        agent_url = self.available_agents[agent_name]['url']
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"universal_req_{agent_name}_{int(time.time())}",
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": comprehensive_instructions
                    }]
                }
            },
            "id": f"universal_req_{agent_name}_{int(time.time())}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    agent_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ {agent_name} response received")
                    return self._parse_agent_response(result, agent_name)
                else:
                    logger.warning(f"❌ {agent_name} HTTP error: {response.status_code}")
                    return {
                        'status': 'failed',
                        'error': f'HTTP {response.status_code}',
                        'summary': f'에이전트 호출 실패 (HTTP {response.status_code})'
                    }
                    
        except Exception as e:
            logger.error(f"Agent {agent_name} execution error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'summary': f'에이전트 실행 중 오류: {str(e)}'
            }
    
    def _parse_agent_response(self, response: Dict, agent_name: str) -> Dict:
        """에이전트 응답 파싱"""
        try:
            if 'result' in response:
                result = response['result']
                if isinstance(result, dict):
                    # A2A 표준 응답 처리
                    status = result.get('status', {})
                    if isinstance(status, dict) and status.get('state') == 'completed':
                        return {
                            'status': 'success',
                            'result': result,
                            'summary': f'{agent_name} 에이전트 작업 완료'
                        }
                    else:
                        return {
                            'status': 'partial',
                            'result': result,
                            'summary': f'{agent_name} 에이전트 부분 완료'
                        }
                else:
                    return {
                        'status': 'success',
                        'result': result,
                        'summary': f'{agent_name} 에이전트 작업 완료'
                    }
            else:
                return {
                    'status': 'failed',
                    'error': 'No result in response',
                    'summary': f'{agent_name} 에이전트 응답 오류'
                }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'summary': f'{agent_name} 에이전트 응답 파싱 오류'
            }
    
    async def _validate_agent_response(self, agent_name: str, response: Dict) -> Dict:
        """에이전트 응답의 신뢰성 검증"""
        
        validation_result = {
            'is_valid': True,
            'has_data': False,
            'data_sources': [],
            'confidence_score': 0,
            'warnings': []
        }
        
        try:
            # 응답에서 실제 데이터 추출
            if 'result' in response and isinstance(response['result'], dict):
                result = response['result']
                
                # 아티팩트 확인
                if 'artifacts' in result:
                    for artifact in result['artifacts']:
                        if isinstance(artifact, dict):
                            content_type = artifact.get('metadata', {}).get('content_type', '')
                            if content_type in ['application/json', 'text/csv', 'application/vnd.plotly.v1+json']:
                                validation_result['has_data'] = True
                                validation_result['data_sources'].append({
                                    'name': artifact.get('name'),
                                    'type': content_type
                                })
                
                # 메시지 내용 확인
                if 'history' in result:
                    for msg in result['history']:
                        if msg.get('role') == 'agent' and 'parts' in msg:
                            for part in msg['parts']:
                                if part.get('kind') == 'text':
                                    text = part.get('text', '')
                                    # 수치 데이터 존재 여부
                                    numbers = re.findall(r'\d+\.?\d*', text)
                                    if numbers:
                                        validation_result['confidence_score'] += 30
                                    
                                    # 분석 키워드 존재 여부
                                    keywords = ['평균', '표준편차', '상관관계', '분포', '패턴', '추세']
                                    for keyword in keywords:
                                        if keyword in text:
                                            validation_result['confidence_score'] += 10
            
            # 검증 결과 판단
            validation_result['confidence_score'] = min(validation_result['confidence_score'], 100)
            
            if not validation_result['has_data'] and validation_result['confidence_score'] < 50:
                validation_result['warnings'].append('구체적인 데이터나 분석 결과가 부족함')
                validation_result['is_valid'] = False
                
        except Exception as e:
            logger.warning(f"응답 검증 실패: {e}")
            validation_result['warnings'].append(f'검증 오류: {str(e)}')
        
        return validation_result
    
    async def _check_overall_progress(self,
                                    execution_history: List[Dict],
                                    user_intent: Dict,
                                    current_plan: Dict) -> Dict:
        """전체 진행 상황을 평가하고 조기 완료 가능성 판단"""
        
        if not self.openai_client:
            return {'should_conclude': False}
        
        # 실행 이력 요약
        history_summary = []
        for h in execution_history[-5:]:  # 최근 5개만
            history_summary.append({
                'agent': h['agent'],
                'status': h['result'].get('status', 'unknown'),
                'has_valid_data': h['result'].get('validation', {}).get('is_valid', False)
            })
        
        progress_prompt = f"""
        사용자 목표: {user_intent['main_goal']}
        필요한 결과: {json.dumps(user_intent['expected_outcomes'], ensure_ascii=False)}
        
        현재까지 실행 결과:
        {json.dumps(history_summary, ensure_ascii=False)}
        
        다음을 평가하세요:
        1. 사용자 목표 달성도 (0-100%)
        2. 추가 실행이 의미있는 개선을 가져올지
        3. 현재 결과로 충분한 답변이 가능한지
        
        JSON 응답:
        {{
            "achievement_percentage": 0-100,
            "key_goals_met": ["달성된 목표들"],
            "missing_elements": ["부족한 부분들"],
            "should_conclude": true/false,
            "reasoning": "판단 근거"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": progress_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"진행률 평가 실패: {e}")
            return {'should_conclude': False}
    
    async def _create_contextual_instruction(self,
                                           agent_name: str,
                                           step: Dict,
                                           user_intent: Dict,
                                           agent_capability: Dict,
                                           previous_results: Dict,
                                           context: Dict) -> str:
        """컨텍스트를 반영한 정밀한 지시 생성"""
        
        # 이전 결과에서 핵심 인사이트 추출
        previous_insights = self._extract_key_insights(previous_results)
        
        if not self.openai_client:
            return step.get('comprehensive_instructions', f"{agent_name} 작업 수행")
        
        instruction_prompt = f"""
        에이전트: {agent_name}
        에이전트 능력: {json.dumps(agent_capability, ensure_ascii=False)}
        
        현재 단계 목적: {step.get('purpose', '')}
        기본 작업: {step.get('comprehensive_instructions', '')}
        
        사용자 의도:
        - 액션 타입: {user_intent.get('action_type')}
        - 주요 목표: {user_intent.get('main_goal')}
        - 구체적 요구사항: {user_intent.get('specific_requirements')}
        - 기대 결과: {user_intent.get('expected_outcomes')}
        
        도메인 컨텍스트:
        - 분야: {context.get('domain')}
        - 전문성 수준: {context.get('expertise_level')}
        - 특별 고려사항: {context.get('special_considerations')}
        
        이전 단계에서 발견된 인사이트:
        {json.dumps(previous_insights, ensure_ascii=False)}
        
        위 정보를 종합하여 {agent_name}이 수행해야 할 매우 구체적이고 
        컨텍스트에 맞는 작업 지시를 작성하세요.
        
        포함사항:
        1. 정확히 무엇을 분석/처리해야 하는지
        2. 이전 결과를 어떻게 활용해야 하는지
        3. 어떤 형태의 결과물을 생성해야 하는지
        4. 주의사항과 품질 기준
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0.3,
                max_tokens=1500,
                timeout=60.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Contextual instruction generation failed: {e}")
            return step.get('comprehensive_instructions', f"{agent_name} 작업 수행")
    
    def _extract_key_insights(self, previous_results: Dict) -> Dict:
        """이전 결과에서 핵심 인사이트 추출"""
        insights = {}
        for agent_name, result in previous_results.items():
            if result.get('status') == 'success':
                insights[agent_name] = result.get('summary', '작업 완료')
            else:
                insights[agent_name] = f"오류: {result.get('error', '알 수 없는 오류')}"
        return insights
    
    async def _assess_content_richness(self, agent_results: Dict) -> Dict:
        """생성된 콘텐츠의 풍부함을 평가하고 활용 방안 결정"""
        
        if not self.openai_client:
            return {
                "has_visualizations": False,
                "visualization_details": [],
                "key_metrics": {},
                "critical_findings": ["기본 분석 결과"],
                "data_quality_score": 5,
                "recommended_inclusion": ["분석 요약"]
            }
        
        # 결과 요약 생성
        results_summary = {}
        for agent, result in agent_results.items():
            if result.get('validation', {}).get('is_valid', False):
                results_summary[agent] = {
                    'status': result.get('status'),
                    'has_artifacts': bool(result.get('result', {}).get('artifacts')),
                    'data_sources': result.get('validation', {}).get('data_sources', [])
                }
        
        assessment_prompt = f"""
        다음 분석 결과들을 평가하세요:
        {json.dumps(results_summary, ensure_ascii=False)}
        
        평가할 항목:
        1. 시각화 자료 (차트, 그래프)의 존재와 중요도
        2. 구체적인 수치나 통계 데이터
        3. 발견된 패턴이나 이상치
        4. 실무적 인사이트의 가치
        5. 사용자가 놓치면 아까울 중요 정보
        
        {{
            "has_visualizations": true/false,
            "visualization_details": ["어떤 시각화가 있는지"],
            "key_metrics": {{"메트릭명": "값"}},
            "critical_findings": ["놓치면 안 되는 발견사항"],
            "data_quality_score": 1-10,
            "recommended_inclusion": ["반드시 포함해야 할 요소들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            assessment = json.loads(response.choices[0].message.content)
            logger.info(f"🎨 Content richness assessed: score {assessment.get('data_quality_score', 5)}/10")
            return assessment
            
        except Exception as e:
            logger.warning(f"Content richness assessment failed: {e}")
            return {
                "has_visualizations": False,
                "visualization_details": [],
                "key_metrics": {},
                "critical_findings": ["기본 분석 결과"],
                "data_quality_score": 5,
                "recommended_inclusion": ["분석 요약"]
            }
    
    async def _create_evidence_based_response(self, 
                                            user_input: str,
                                            validated_results: Dict[str, Dict],
                                            user_intent: Dict) -> str:
        """검증된 데이터만을 기반으로 한 증거 기반 응답 생성"""
        
        # 신뢰할 수 있는 결과만 필터링
        reliable_results = {
            agent: result
            for agent, result in validated_results.items()
            if result.get('validation', {}).get('is_valid', False)
        }
        
        if not reliable_results:
            return "충분한 분석 데이터를 수집하지 못했습니다. 다시 시도해 주세요."
        
        if not self.openai_client:
            return self._create_fallback_synthesis(user_input, reliable_results)
        
        evidence_prompt = f"""
        사용자 요청: "{user_input}"
        사용자 의도: {json.dumps(user_intent, ensure_ascii=False)}
        
        검증된 분석 결과:
        {self._structure_reliable_results(reliable_results)}
        
        위의 검증된 데이터만을 사용하여 응답을 작성하세요.
        
        엄격한 규칙:
        1. 위에 없는 데이터는 절대 만들지 마세요
        2. 추측이나 일반론은 금지
        3. 모든 주장은 위 데이터에서 직접 인용
        4. 데이터가 없는 부분은 "데이터 없음"으로 명시
        5. 구체적인 수치와 출처를 항상 명시
        
        응답 구조:
        1. 핵심 발견사항 (데이터 기반)
        2. 상세 분석 (각 주장마다 근거 명시)
        3. 한계점 (분석되지 않은 부분 명시)
        4. 결론 (사용자 질문에 직접 답변)
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": evidence_prompt}],
                temperature=0.1,
                max_tokens=3000,
                timeout=90.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"증거 기반 응답 생성 실패: {e}")
            return self._create_fallback_synthesis(user_input, reliable_results)
    
    async def _create_intelligent_final_response(self,
                                               user_input: str,
                                               user_intent: Dict,
                                               execution_result: Dict,
                                               content_assessment: Dict,
                                               context: Dict) -> str:
        """지능적인 최종 응답 생성"""
        
        if not self.openai_client:
            return self._create_fallback_synthesis(user_input, execution_result['results'])
        
        # 검증된 결과만 사용
        reliable_results = {
            agent: result
            for agent, result in execution_result['results'].items()
            if result.get('validation', {}).get('is_valid', False)
        }
        
        response_prompt = f"""
        당신은 {context.get('adopted_perspective', '데이터 분석 전문가')}입니다.
        
        사용자 요청: "{user_input}"
        
        사용자 의도 분석:
        - 액션 타입: {user_intent['action_type']}
        - 주요 목표: {user_intent['main_goal']}
        - 기대 결과: {json.dumps(user_intent['expected_outcomes'], ensure_ascii=False)}
        
        검증된 분석 결과:
        {self._structure_reliable_results(reliable_results)}
        
        실행 요약:
        - 총 단계: {len(execution_result['history'])}
        - 성공률: {len(reliable_results)}/{len(execution_result['results'])}
        - 리플래닝: {execution_result['replanning_count']}회
        - 완료 방식: {execution_result['completion_reason']}
        
        콘텐츠 평가:
        - 데이터 품질: {content_assessment.get('data_quality_score')}/10
        - 핵심 발견: {len(content_assessment.get('critical_findings', []))}개
        - 시각화: {content_assessment.get('has_visualizations')}
        
        응답 작성 지침:
        1. 사용자의 {user_intent['action_type']} 요청에 정확히 답변
        2. 검증된 데이터만 사용 (위에 없는 내용 창작 금지)
        3. {context.get('response_style', 'professional')} 톤 유지
        4. 모든 주장에 구체적 근거 제시
        5. 한계점이 있다면 명확히 명시
        
        {self._get_action_specific_instructions(user_intent['action_type'])}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.3,
                max_tokens=4000,
                timeout=90.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Intelligent response generation failed: {e}")
            return self._create_fallback_synthesis(user_input, reliable_results)
    
    def _structure_reliable_results(self, reliable_results: Dict) -> str:
        """신뢰할 수 있는 결과를 구조화"""
        structured = "### 검증된 분석 결과\n\n"
        
        for agent_name, result in reliable_results.items():
            validation = result.get('validation', {})
            confidence = validation.get('confidence_score', 0)
            
            structured += f"#### {agent_name} (신뢰도: {confidence}%)\n"
            
            # 주요 결과 추출
            if 'summary' in result:
                structured += f"- **요약**: {result['summary']}\n"
            
            # 구체적 데이터 추출
            if 'result' in result and isinstance(result['result'], dict):
                result_data = result['result']
                
                # 아티팩트 정보
                if 'artifacts' in result_data:
                    artifacts = result_data['artifacts']
                    structured += f"- **생성된 데이터**: {len(artifacts)}개\n"
                    
                    for artifact in artifacts[:2]:  # 주요 2개만
                        if isinstance(artifact, dict):
                            name = artifact.get('name', 'unnamed')
                            content_type = artifact.get('metadata', {}).get('content_type', 'unknown')
                            structured += f"  - {name} ({content_type})\n"
            
            # 핵심 수치 추출
            if validation.get('data_sources'):
                structured += f"- **데이터 소스**: {len(validation['data_sources'])}개\n"
            
            structured += "\n"
        
        return structured
    
    async def _verify_response_matches_intent(self, response: str, user_intent: Dict) -> bool:
        """응답이 사용자 의도와 일치하는지 검증"""
        
        if not self.openai_client:
            return True  # 검증 불가시 통과
        
        verification_prompt = f"""
        사용자 의도:
        - 액션 타입: {user_intent['action_type']}
        - 주요 목표: {user_intent['main_goal']}
        - 기대 결과: {json.dumps(user_intent['expected_outcomes'], ensure_ascii=False)}
        
        생성된 응답:
        {response[:1000]}...
        
        이 응답이 사용자 의도와 일치하는지 평가하세요:
        
        {{
            "matches_intent": true/false,
            "missing_elements": ["부족한 요소들"],
            "alignment_score": 0-100
        }}
        """
        
        try:
            response_obj = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": verification_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=30.0
            )
            
            verification = json.loads(response_obj.choices[0].message.content)
            return verification.get('matches_intent', True)
            
        except Exception as e:
            logger.warning(f"Intent verification failed: {e}")
            return True
    
    async def _regenerate_response_for_intent(self,
                                            user_input: str,
                                            validated_results: Dict,
                                            user_intent: Dict) -> str:
        """의도에 맞게 응답 재생성"""
        
        if not self.openai_client:
            return self._create_fallback_synthesis(user_input, validated_results)
        
        regeneration_prompt = f"""
        사용자가 정확히 원하는 것:
        - 액션: {user_intent['action_type']}
        - 목표: {user_intent['main_goal']}
        - 기대 결과: {json.dumps(user_intent['expected_outcomes'], ensure_ascii=False)}
        
        원본 요청: "{user_input}"
        
        사용 가능한 데이터:
        {self._structure_reliable_results(validated_results)}
        
        위 데이터를 사용하여 사용자 의도에 정확히 맞는 응답을 작성하세요.
        {self._get_action_specific_instructions(user_intent['action_type'])}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": regeneration_prompt}],
                temperature=0.2,
                max_tokens=4000,
                timeout=90.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Response regeneration failed: {e}")
            return self._create_fallback_synthesis(user_input, validated_results)
    
    def _get_action_specific_instructions(self, action_type: str) -> str:
        """액션 타입별 구체적 지시사항"""
        instructions = {
            'analyze': """
            분석 결과 구조:
            1. 핵심 발견사항 요약
            2. 상세 분석 (데이터 기반)
            3. 패턴과 트렌드
            4. 결론 및 시사점
            """,
            
            'verify': """
            검증 결과 구조:
            1. 검증 대상 명확화
            2. 사실 여부 판단 (O/X)
            3. 판단 근거 (데이터 인용)
            4. 신뢰도 평가
            """,
            
            'recommend': """
            추천 결과 구조:
            1. 추천 사항 (우선순위 포함)
            2. 각 추천의 근거
            3. 예상 효과
            4. 실행 시 고려사항
            """,
            
            'diagnose': """
            진단 결과 구조:
            1. 현재 상태 평가
            2. 문제점 식별
            3. 원인 분석
            4. 개선 방향
            """,
            
            'predict': """
            예측 결과 구조:
            1. 예측 결과
            2. 예측 근거
            3. 신뢰도 및 불확실성
            4. 시나리오별 전망
            """,
            
            'compare': """
            비교 결과 구조:
            1. 비교 대상 명확화
            2. 주요 차이점
            3. 장단점 분석
            4. 상황별 추천
            """,
            
            'explain': """
            설명 결과 구조:
            1. 개념/현상 정의
            2. 작동 원리/메커니즘
            3. 실제 사례
            4. 추가 참고사항
            """
        }
        
        return instructions.get(action_type, "사용자 요청에 맞는 구조로 답변하세요.")
    
    async def _find_alternative_agent(self, user_input: str, available_agents: Dict) -> Optional[str]:
        """대체 에이전트 찾기"""
        if not self.openai_client or not available_agents:
            return None
        
        try:
            agent_list = list(available_agents.keys())
            prompt = f"""
            사용자 요청: "{user_input}"
            
            사용 가능한 에이전트: {agent_list}
            
            이 요청을 처리하기에 가장 적합한 에이전트를 선택하세요.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
                timeout=30.0
            )
            
            selected = response.choices[0].message.content.strip()
            
            # 선택된 에이전트가 실제로 존재하는지 확인
            for agent in agent_list:
                if agent in selected:
                    return agent
            
            return agent_list[0] if agent_list else None
            
        except Exception as e:
            logger.warning(f"Alternative agent selection failed: {e}")
            return list(available_agents.keys())[0] if available_agents else None
    
    async def _improve_instruction_based_on_failure(self, 
                                                  original_instruction: str,
                                                  warnings: List[str],
                                                  user_intent: Dict) -> str:
        """실패 원인을 바탕으로 지시 개선"""
        
        if not self.openai_client:
            return original_instruction + "\n\n더 구체적이고 상세한 분석을 수행하세요."
        
        improvement_prompt = f"""
        원래 지시: {original_instruction}
        
        실패 원인: {warnings}
        사용자 의도: {user_intent['main_goal']}
        
        위 실패 원인을 해결할 수 있도록 지시를 개선하세요.
        더 구체적이고 명확한 지시를 작성하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.3,
                max_tokens=1500,
                timeout=60.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Instruction improvement failed: {e}")
            return original_instruction + "\n\n더 구체적이고 상세한 분석을 수행하세요."
    
    async def _handle_with_alternative_approach(self,
                                              user_input: str,
                                              user_intent: Dict,
                                              task_updater: StreamingTaskUpdater):
        """대체 접근 방식으로 처리"""
        
        if self.openai_client:
            try:
                # LLM으로 직접 응답 생성
                alternative_prompt = f"""
                사용자 요청: "{user_input}"
                
                에이전트 실행이 실패했습니다.
                사용 가능한 정보와 일반적인 지식을 바탕으로
                최선의 답변을 제공하세요.
                
                주의: 추측이나 가정은 최소화하고,
                확실한 정보만 제공하세요.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": alternative_prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=60.0
                )
                
                answer = response.choices[0].message.content
                answer = f"⚠️ 에이전트 실행에 실패하여 제한된 정보로 답변드립니다.\n\n{answer}"
                
                await task_updater.update_status(
                    TaskState.completed,
                    message=task_updater.new_agent_message(parts=[TextPart(text=answer)])
                )
                
            except Exception as e:
                logger.error(f"Alternative approach failed: {e}")
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text="죄송합니다. 요청을 처리할 수 없습니다.")]
                    )
                )
        else:
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="에이전트 실행 실패 및 대체 방안도 사용할 수 없습니다.")]
                )
            )
    
    async def _save_plan_artifact(self, plan: Dict, task_updater: StreamingTaskUpdater):
        """계획을 아티팩트로 저장"""
        
        plan_artifact = {
            "execution_strategy": plan.get('execution_strategy', ''),
            "agent_selection_reasoning": plan.get('agent_selection_reasoning', ''),
            "plan_executed": [
                {
                    "step": i + 1,
                    "agent": step.get('agent', 'unknown'),
                    "purpose": step.get('purpose', ''),
                    "comprehensive_instructions": step.get('comprehensive_instructions', ''),
                    "expected_deliverables": step.get('expected_deliverables', {})
                }
                for i, step in enumerate(plan.get('steps', []))
            ]
        }
        
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(plan_artifact, ensure_ascii=False, indent=2))],
            name="comprehensive_execution_plan.json",
            metadata={
                "content_type": "application/json",
                "plan_type": "universal_intelligent_orchestration",
                "description": "Universal Intelligent Orchestrator 실행 계획"
            }
        )
    
    async def _save_execution_summary(self,
                                    execution_result: Dict,
                                    content_assessment: Dict,
                                    task_updater: StreamingTaskUpdater):
        """실행 요약을 아티팩트로 저장"""
        
        execution_summary = {
            "total_steps_executed": len(execution_result['history']),
            "successful_agents": len([r for r in execution_result['results'].values() 
                                    if r.get('validation', {}).get('is_valid', False)]),
            "replanning_count": execution_result['replanning_count'],
            "completion_reason": execution_result['completion_reason'],
            "content_assessment": content_assessment,
            "execution_strategy": "universal_intelligent_orchestration_v7"
        }
        
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(execution_summary, ensure_ascii=False, indent=2))],
            name="execution_summary.json",
            metadata={
                "content_type": "application/json",
                "summary_type": "universal_system_execution",
                "description": "Universal System 실행 요약"
            }
        )
    
    def _create_beautiful_plan_display(self, execution_plan: Dict, user_intent: Dict) -> str:
        """예쁜 실행 계획 표시 생성"""
        
        plan_display = f"""
## 📋 Intelligent Execution Plan

### 🎯 분석 개요
- **목표**: {user_intent.get('main_goal', '데이터 분석')}
- **액션 타입**: {user_intent.get('action_type', 'analyze')}
- **총 단계**: {len(execution_plan.get('steps', []))}개

### 🚀 실행 전략
{execution_plan.get('execution_strategy', '사용자 요청에 최적화된 분석')}

### 📊 단계별 계획

"""
        
        for i, step in enumerate(execution_plan.get('steps', [])):
            step_num = i + 1
            agent_name = step.get('agent', 'unknown')
            purpose = step.get('purpose', '')
            
            plan_display += f"""**{step_num}. {agent_name}**
   - 🎯 **목적**: {purpose}
   - 📝 **기대 결과**: 
     - 최소: {step.get('expected_deliverables', {}).get('minimum', '기본 결과')}
     - 표준: {step.get('expected_deliverables', {}).get('standard', '상세 결과')}
     - 탁월: {step.get('expected_deliverables', {}).get('exceptional', '심층 인사이트')}

"""
        
        plan_display += f"""
### 🧠 선택 근거
{execution_plan.get('agent_selection_reasoning', '요청에 최적화된 에이전트 선택')}

---
"""
        
        return plan_display
    
    def _create_fallback_plan(self, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """LLM이 사용 불가능할 때의 폴백 계획"""
        steps = []
        
        # 기본적인 데이터 분석 워크플로우
        basic_workflow = [
            ("data_loader", "데이터 로드 및 기본 검증", "데이터 파일을 로드하고 기본적인 구조를 파악합니다"),
            ("data_cleaning", "데이터 품질 확인 및 정리", "데이터의 품질을 확인하고 필요한 정리 작업을 수행합니다"),
            ("eda_tools", "탐색적 데이터 분석 수행", "데이터의 분포와 패턴을 탐색하고 기초 통계를 생성합니다"),
            ("data_visualization", "데이터 시각화 및 인사이트 도출", "데이터를 시각화하여 인사이트를 도출합니다")
        ]
        
        for agent_name, purpose, task_desc in basic_workflow:
            if agent_name in available_agents:
                steps.append({
                    "step_number": len(steps) + 1,
                    "agent": agent_name,
                    "purpose": purpose,
                    "comprehensive_instructions": task_desc,
                    "expected_deliverables": {
                        "minimum": f"{purpose} 기본 결과",
                        "standard": f"{purpose} 상세 결과",
                        "exceptional": f"{purpose} 심층 분석"
                    },
                    "success_criteria": "분석 완료",
                    "context_for_next": ["분석 결과", "데이터 정보"]
                })
        
        return {
            "execution_strategy": "표준 데이터 분석 워크플로우",
            "agent_selection_reasoning": "기본적인 데이터 분석을 위한 표준 에이전트 선택",
            "steps": steps,
            "final_synthesis_strategy": "모든 분석 결과를 종합하여 인사이트 도출",
            "potential_insights": ["데이터 패턴", "통계적 특성", "시각적 발견"]
        }
    
    def _create_fallback_synthesis(self, original_request: str, all_results: Dict) -> str:
        """폴백 최종 답변 생성"""
        
        successful_agents = [name for name, result in all_results.items() 
                           if result.get('status') == 'success']
        failed_agents = [name for name, result in all_results.items() 
                        if result.get('status') == 'failed']
        
        synthesis = f"""## 📊 데이터 분석 결과 종합

### 🎯 요청 사항
{original_request}

### ✅ 완료된 분석 단계
"""
        
        for agent_name in successful_agents:
            result = all_results[agent_name]
            synthesis += f"- **{agent_name}**: {result.get('summary', '작업 완료')}\n"
        
        if failed_agents:
            synthesis += f"\n### ⚠️ 일부 제한사항\n"
            for agent_name in failed_agents:
                result = all_results[agent_name]
                synthesis += f"- **{agent_name}**: {result.get('error', '알 수 없는 오류')}\n"
        
        synthesis += f"""

### 🎉 결론
총 {len(successful_agents)}개의 분석 단계가 성공적으로 완료되었습니다. 
각 에이전트가 생성한 결과물과 아티팩트를 확인하시기 바랍니다.

분석 결과에 대한 추가 질문이나 더 자세한 분석이 필요하시면 언제든 요청해 주세요.
"""
        
        return synthesis
    
    def _infer_use_cases(self, agent_name: str) -> List[str]:
        """에이전트 이름에서 사용 사례 추론"""
        use_cases = {
            "data_loader": ["데이터 파일 로드", "데이터 구조 파악", "기본 검증"],
            "data_cleaning": ["결측값 처리", "중복 제거", "데이터 정제"],
            "eda_tools": ["통계 분석", "분포 확인", "상관관계 분석"],
            "data_visualization": ["차트 생성", "시각화", "패턴 발견"],
            "data_wrangling": ["데이터 변환", "형태 변경", "집계"],
            "feature_engineering": ["특성 생성", "차원 축소", "특성 선택"],
            "sql_database": ["SQL 쿼리", "데이터베이스 조회", "조인"],
            "h2o_modeling": ["머신러닝", "예측 모델", "AutoML"],
            "mlflow_tracking": ["실험 추적", "모델 버전 관리", "성능 비교"]
        }
        
        return use_cases.get(agent_name, ["데이터 처리"])


def create_universal_intelligent_orchestrator_server():
    """Universal Intelligent Orchestrator 서버 생성"""
    
    agent_card = AgentCard(
        name="Universal Intelligent Orchestrator v7.0",
        description="LLM 기반 범용 지능형 오케스트레이터 - 적응적 처리, 동적 리플래닝, 실시간 스트리밍 지원",
        url="http://localhost:8100",
        version="7.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="adaptive_processing",
                name="Adaptive Request Processing",
                description="요청 복잡도에 따라 즉답, 단일 에이전트, 복잡한 워크플로우를 자동 선택",
                tags=["adaptive", "intelligent", "complexity-aware"]
            ),
            AgentSkill(
                id="dynamic_replanning",
                name="Dynamic Replanning System",
                description="실행 중 상황 변화에 따라 계획을 동적으로 수정하고 최적화",
                tags=["replanning", "optimization", "recovery"]
            ),
            AgentSkill(
                id="llm_powered_orchestration",
                name="LLM Powered Orchestration",
                description="LLM이 사용자 의도를 정확히 파악하고 최적의 실행 계획을 생성",
                tags=["llm", "ai", "intent-understanding"]
            ),
            AgentSkill(
                id="evidence_based_synthesis",
                name="Evidence Based Response Generation",
                description="검증된 데이터만을 기반으로 할루시네이션 없는 정확한 응답 생성",
                tags=["evidence-based", "validation", "accuracy"]
            )
        ]
    )
    
    executor = UniversalIntelligentOrchestrator()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )


def main():
    """메인 실행 함수"""
    logger.info("🚀 Starting Universal Intelligent Orchestrator v7.0")
    
    app = create_universal_intelligent_orchestrator_server()
    
    uvicorn.run(
        app.build(),
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main()