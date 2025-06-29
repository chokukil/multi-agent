#!/usr/bin/env python3
"""
A2A Orchestrator v6.0 - LLM Powered Dynamic Context-Aware Orchestrator
LLM 기반 동적 컨텍스트 인식 오케스트레이터
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

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
        """최종 응답을 청크로 나누어 스트리밍"""
        # Markdown 섹션별로 스트리밍
        sections = response.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                await self.update_status(
                    TaskState.working,
                    message=self.new_agent_message(parts=[TextPart(text=section)])
                )
                await asyncio.sleep(0.1)  # 부드러운 스트리밍
        
        # 완료 상태 업데이트
        await self.update_status(
            TaskState.completed,
            message=self.new_agent_message(parts=[TextPart(text="✅ 분석이 완료되었습니다.")])
        )


class LLMPoweredOrchestratorExecutor(AgentExecutor):
    """LLM 기반 동적 컨텍스트 인식 오케스트레이터"""
    
    def __init__(self):
        # OpenAI 클라이언트 초기화 (옵션)
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 LLM Powered Dynamic Orchestrator v6 with OpenAI integration")
            else:
                self.openai_client = None
                logger.info("📊 Standard Orchestrator v6 (OpenAI API key not found)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        self.available_agents = {}
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """LLM이 전체 프로세스를 이해하고 조정하는 실행"""
        task_updater = StreamingTaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            logger.info(f"📥 Processing orchestration query: {user_input}")
            
            if not user_input:
                user_input = "Please provide an analysis request."
            
            # 에이전트 발견
            await task_updater.stream_update("🔍 AI DS Team 에이전트들을 발견하고 있습니다...")
            self.available_agents = await self._discover_agents()
            
            if not self.available_agents:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용 가능한 A2A 에이전트를 찾을 수 없습니다.")])
                )
                return
            
            await task_updater.stream_update(f"✅ {len(self.available_agents)}개 에이전트 발견 완료")
            
            # 1. 사용자 입력을 LLM이 완전히 이해
            await task_updater.stream_update("🧠 사용자 요청을 심층 분석하고 있습니다...")
            request_understanding = await self._understand_request(user_input)
            
            # 2. 동적 실행 계획 생성
            await task_updater.stream_update("📋 동적 실행 계획을 생성하고 있습니다...")
            execution_plan = await self._create_dynamic_plan(
                request_understanding, 
                self.available_agents
            )
            
            if not execution_plan or not execution_plan.get('steps'):
                execution_plan = self._create_fallback_plan(self.available_agents)
            
            await task_updater.stream_update(f"✅ {len(execution_plan.get('steps', []))}단계 실행 계획 완료")
            
            # 계획을 아티팩트로 전송 (클라이언트 파싱용)
            plan_artifact = {
                "plan_executed": [
                    {
                        "step": i + 1,
                        "agent": step.get('agent', step.get('agent_name', 'unknown')),
                        "task_description": step.get('enriched_task', step.get('purpose', '')),
                        "reasoning": step.get('purpose', ''),
                        "expected_output": step.get('expected_output', '')
                    }
                    for i, step in enumerate(execution_plan.get('steps', []))
                ]
            }
            
            # 아티팩트로 계획 전송
            try:
                await task_updater.add_artifact(
                    parts=[TextPart(text=json.dumps(plan_artifact, ensure_ascii=False))],
                    name="execution_plan",
                    metadata={
                        "content_type": "application/json",
                        "plan_type": "ai_ds_team_orchestration"
                    }
                )
            except Exception as artifact_error:
                logger.warning(f"Failed to send plan artifact: {artifact_error}")
            
            # 3. 각 에이전트 실행 (컨텍스트 전달)
            agent_results = {}
            for i, step in enumerate(execution_plan.get('steps', [])):
                step_num = i + 1
                agent_name = step.get('agent', step.get('agent_name', 'unknown'))
                
                await task_updater.stream_update(f"🔄 단계 {step_num}: {agent_name} 에이전트 실행 중...")
                
                try:
                    result = await self._execute_agent_with_context(
                        agent_name=agent_name,
                        task=step.get('enriched_task', step.get('task_description', '')),
                        full_context=request_understanding,
                        previous_results=agent_results
                    )
                    agent_results[agent_name] = result
                    
                    # 실시간 피드백
                    summary = result.get('summary', 'Processing completed')
                    await task_updater.stream_update(f"✅ {agent_name}: {summary}")
                    
                except Exception as agent_error:
                    logger.warning(f"Agent {agent_name} execution failed: {agent_error}")
                    agent_results[agent_name] = {
                        'status': 'failed',
                        'error': str(agent_error),
                        'summary': f"에이전트 실행 중 오류 발생: {str(agent_error)}"
                    }
                    await task_updater.stream_update(f"⚠️ {agent_name}: 오류 발생하였으나 계속 진행합니다")
            
            # 4. LLM이 모든 결과를 종합하여 최종 답변 생성
            await task_updater.stream_update("🎯 모든 분석 결과를 종합하여 최종 답변을 생성하고 있습니다...")
            
            final_response = await self._synthesize_with_llm(
                original_request=user_input,
                understanding=request_understanding,
                all_results=agent_results,
                task_updater=task_updater
            )
            
            # 5. 스트리밍으로 최종 답변 전달
            await task_updater.stream_final_response(final_response)
            
        except Exception as e:
            logger.error(f"Error in LLM Powered Orchestrator: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"오류 발생: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

    async def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """A2A 표준 에이전트 발견"""
        available_agents = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
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

    async def _understand_request(self, user_input: str) -> Dict[str, Any]:
        """LLM이 요청을 깊이 이해하고 구조화"""
        
        if not self.openai_client:
            # Fallback: 기본 구조 반환
            return {
                "domain": "데이터 분석",
                "expertise_claimed": "일반 사용자",
                "key_objectives": ["데이터 분석 수행"],
                "required_outputs": ["분석 결과"],
                "domain_context": "일반적인 데이터 분석",
                "data_mentioned": "업로드된 데이터",
                "analysis_depth": "intermediate",
                "tone": "technical"
            }
        
        prompt = f"""다음 사용자 요청을 분석하여 구조화하세요:

{user_input}

다음 JSON 형식으로 응답하세요:
{{
    "domain": "식별된 도메인 (예: 반도체, 금융, 의료, 제조업 등)",
    "expertise_claimed": "사용자가 언급한 전문성이나 역할",
    "key_objectives": ["목표1", "목표2"],
    "required_outputs": ["필요한 산출물1", "산출물2"],
    "domain_context": "도메인 특화 지식이나 규칙 요약",
    "data_mentioned": "언급된 데이터나 파일",
    "analysis_depth": "요구되는 분석 깊이 (basic/intermediate/expert)",
    "tone": "응답 톤 (technical/business/educational)"
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Request understanding failed: {e}")
            return {
                "domain": "데이터 분석",
                "expertise_claimed": "일반 사용자",
                "key_objectives": ["데이터 분석 수행"],
                "required_outputs": ["분석 결과"],
                "domain_context": user_input,
                "data_mentioned": "업로드된 데이터",
                "analysis_depth": "intermediate",
                "tone": "technical"
            }

    async def _create_dynamic_plan(self, understanding: Dict, 
                                  available_agents: Dict) -> Dict:
        """요청에 따라 동적으로 최적의 실행 계획 생성"""
        
        if not self.openai_client:
            return self._create_fallback_plan(available_agents)
        
        agents_info = json.dumps(
            {name: info['description'] for name, info in available_agents.items()},
            ensure_ascii=False
        )
        
        planning_prompt = f"""
사용자 요청 분석:
{json.dumps(understanding, ensure_ascii=False, indent=2)}

사용 가능한 에이전트:
{agents_info}

이 요청을 처리하기 위한 최적의 에이전트 실행 순서를 계획하세요.
각 단계에서 어떤 정보를 다음 단계로 전달해야 하는지도 명시하세요.

다음 JSON 형식으로 응답하세요:
{{
    "reasoning": "이 계획을 선택한 이유",
    "steps": [
        {{
            "agent": "에이전트명",
            "purpose": "이 단계의 목적",
            "enriched_task": "구체적인 작업 지시 (도메인 컨텍스트 포함)",
            "expected_output": "예상 산출물",
            "pass_to_next": ["다음 단계로 전달할 정보들"]
        }}
    ],
    "final_synthesis_guide": "최종 종합 시 중점사항"
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": planning_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Dynamic planning failed: {e}")
            return self._create_fallback_plan(available_agents)

    async def _execute_agent_with_context(self, agent_name: str, task: str, 
                                        full_context: Dict, previous_results: Dict) -> Dict:
        """각 에이전트에 풍부한 컨텍스트와 함께 작업 전달"""
        
        if agent_name not in self.available_agents:
            return {
                'status': 'failed',
                'error': f'Agent {agent_name} not available',
                'summary': f'에이전트 {agent_name}를 찾을 수 없습니다'
            }
        
        # LLM이 이전 결과를 바탕으로 현재 에이전트의 작업을 보강
        enriched_prompt = await self._enrich_agent_task(
            agent_name, task, full_context, previous_results
        )
        
        # 에이전트 실행
        agent_url = self.available_agents[agent_name]['url']
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"req_{agent_name}_{int(time.time())}",
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": enriched_prompt
                    }]
                }
            },
            "id": f"req_{agent_name}_{int(time.time())}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    agent_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._parse_agent_response(result, agent_name)
                else:
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

    async def _enrich_agent_task(self, agent_name: str, base_task: str, 
                                context: Dict, previous_results: Dict) -> str:
        """LLM이 각 에이전트의 작업을 컨텍스트에 맞게 보강"""
        
        if not self.openai_client:
            return base_task
        
        # 이전 에이전트들의 핵심 발견사항 추출
        previous_insights = self._extract_key_insights(previous_results)
        
        enrichment_prompt = f"""
에이전트: {agent_name}
기본 작업: {base_task}

전체 맥락:
- 도메인: {context['domain']}
- 사용자 역할/전문성: {context.get('expertise_claimed', '일반 사용자')}
- 최종 목표: {context['key_objectives']}
- 도메인 컨텍스트: {context.get('domain_context', '')}

이전 분석 결과:
{json.dumps(previous_insights, ensure_ascii=False)}

위 정보를 바탕으로 {agent_name} 에이전트가 수행해야 할 구체적인 작업을 작성하세요.
도메인 지식과 이전 결과를 반영하여 더 정확하고 유용한 분석이 되도록 지시하세요.
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 빠른 응답을 위해
                messages=[{"role": "user", "content": enrichment_prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Task enrichment failed: {e}")
            return base_task

    async def _synthesize_with_llm(self, original_request: str, 
                                  understanding: Dict, all_results: Dict,
                                  task_updater: StreamingTaskUpdater) -> str:
        """LLM이 모든 정보를 종합하여 최종 답변 생성"""
        
        if not self.openai_client:
            return self._create_fallback_synthesis(original_request, all_results)
        
        # 각 에이전트 결과를 요약
        results_summary = json.dumps(all_results, ensure_ascii=False, indent=2)
        
        synthesis_prompt = f"""당신은 전문 데이터 분석 팀의 수석 분석가입니다.
        
사용자의 원래 요청:
{original_request}

요청 분석:
- 도메인: {understanding['domain']}
- 목표: {', '.join(understanding['key_objectives'])}
- 필요 산출물: {', '.join(understanding['required_outputs'])}

각 에이전트의 분석 결과:
{results_summary}

위 정보를 종합하여:
1. 사용자가 요청한 모든 사항에 대해 답변하세요
2. 도메인 지식을 활용하여 전문적인 해석을 제공하세요
3. 구체적인 인사이트와 실행 가능한 권고사항을 포함하세요
4. {understanding['tone']} 톤으로 작성하세요

중요: 사용자가 특정 역할이나 전문성을 언급했다면 ({understanding.get('expertise_claimed', 'N/A')}), 
그 관점에서 분석하고 답변하세요."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 다양한 도메인의 전문 지식을 종합하여 인사이트를 도출하는 전문가입니다."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._create_fallback_synthesis(original_request, all_results)

    def _extract_key_insights(self, previous_results: Dict) -> Dict:
        """이전 결과에서 핵심 인사이트 추출"""
        insights = {}
        for agent_name, result in previous_results.items():
            if result.get('status') == 'success':
                insights[agent_name] = result.get('summary', '작업 완료')
            else:
                insights[agent_name] = f"오류: {result.get('error', '알 수 없는 오류')}"
        return insights

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

    def _create_fallback_plan(self, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """A2A 표준 폴백 계획 생성"""
        steps = []
        step_number = 1
        
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
                    "agent": agent_name,
                    "purpose": purpose,
                    "enriched_task": task_desc,
                    "expected_output": f"{purpose} 결과",
                    "pass_to_next": ["분석 결과", "데이터 정보"]
                })
                step_number += 1
        
        return {
            "reasoning": "사용자 요청에 대한 포괄적인 데이터 분석을 수행하기 위한 표준 워크플로우",
            "steps": steps,
            "final_synthesis_guide": "모든 분석 결과를 종합하여 사용자에게 유용한 인사이트 제공"
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


def create_llm_powered_orchestrator_server():
    """LLM 기반 동적 컨텍스트 인식 오케스트레이터 서버 생성"""
    
    agent_card = AgentCard(
        name="AI DS Team LLM Powered Dynamic Orchestrator v6",
        description="LLM 기반 동적 컨텍스트 인식 멀티 에이전트 오케스트레이터 - 실시간 스트리밍 지원",
        url="http://localhost:8100",
        version="6.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="llm_powered_orchestration",
                name="LLM Powered Dynamic Context-Aware Orchestration",
                description="LLM이 사용자 요청을 깊이 이해하고 동적으로 최적의 실행 계획을 생성하여 AI DS Team 에이전트들을 조정합니다. 실시간 스트리밍과 컨텍스트 인식 기능을 제공합니다.",
                tags=["llm-powered", "dynamic-orchestration", "context-aware", "streaming", "multi-agent", "data-science"]
            )
        ]
    )
    
    executor = LLMPoweredOrchestratorExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    return app


def main():
    """메인 실행 함수"""
    logger.info("🚀 Starting LLM Powered Dynamic Context-Aware Orchestrator v6.0")
    
    app = create_llm_powered_orchestrator_server()
    
    uvicorn.run(
        app.build(),
        host="localhost",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main() 