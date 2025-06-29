#!/usr/bin/env python3
"""
A2A Orchestrator v6.0 Stable - LLM Enhanced Orchestrator (Based on v5)
안정적인 LLM 강화 오케스트레이터 (v5 기반)
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
    "feature_engineering": 8310,
    "sql_database": 8311,
    "eda_tools": 8312,
    "h2o_ml": 8313,
    "mlflow_tools": 8314,
}


class LLMEnhancedOrchestratorExecutor(AgentExecutor):
    """LLM 강화 안정적인 오케스트레이터 (v5 기반)"""
    
    def __init__(self):
        # OpenAI 클라이언트 초기화 (옵션)
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 LLM Enhanced Orchestrator v6 with OpenAI integration")
            else:
                self.openai_client = None
                logger.info("📊 Standard Orchestrator v6 (OpenAI API key not found)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
    
    async def _enhance_context_with_llm(self, user_input: str, agent_name: str) -> str:
        """LLM을 사용하여 에이전트별 컨텍스트 강화 (옵션)"""
        
        if not self.openai_client:
            return user_input
            
        try:
            prompt = f"""다음 요청을 {agent_name} 에이전트에게 최적화하여 전달하세요:

원본 요청: {user_input}

{agent_name} 에이전트의 역할에 맞게 구체적이고 명확한 지시사항으로 변환해주세요.
기술적 세부사항과 기대하는 결과물을 포함하세요."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            enhanced_input = response.choices[0].message.content.strip()
            logger.info(f"✨ Enhanced context for {agent_name}")
            return enhanced_input
            
        except Exception as e:
            logger.warning(f"Context enhancement failed for {agent_name}: {e}")
            return user_input
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 표준 준수 실행 - Artifact로 계획 반환"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_query = context.get_user_input()
            logger.info(f"📥 Processing orchestration query: {user_query}")
            
            if not user_query:
                user_query = "Please provide an analysis request."
            
            # 에이전트 발견
            available_agents = await self._discover_agents()
            
            if not available_agents:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용 가능한 A2A 에이전트를 찾을 수 없습니다.")])
                )
                return
            
            # 실제 실행 계획 생성
            execution_plan = await self._generate_execution_plan(user_query, available_agents)
            
            if not execution_plan:
                execution_plan = self._create_fallback_plan(available_agents)
            
            # A2A 표준: 실행 계획을 Artifact로 반환
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(execution_plan, ensure_ascii=False, indent=2))],
                name="execution_plan",
                metadata={
                    "content_type": "application/json",
                    "plan_type": "ai_ds_team_orchestration"
                }
            )
            
            # 완료 메시지
            completion_message = f"""✅ AI DS Team 오케스트레이션 계획 생성 완료

🤖 **발견된 에이전트**: {len(available_agents)}개
📝 **분석 목표**: {execution_plan.get('objective', '데이터 분석 수행')}
🔄 **실행 단계**: {len(execution_plan.get('steps', []))}개

실행 계획이 Artifact로 생성되었습니다."""
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=completion_message)])
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
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

    async def _generate_execution_plan(self, user_query: str, available_agents: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """LLM을 사용하여 A2A 표준 실행 계획 생성"""
        if not self.openai_client:
            logger.warning("OpenAI API key not found, using fallback plan")
            return None
        
        try:
            agent_descriptions = []
            for name, info in available_agents.items():
                agent_descriptions.append(f"- {name}: {info['description']}")
            
            agents_text = "\n".join(agent_descriptions)
            
            system_prompt = f"""당신은 AI Data Science Team의 A2A 표준 오케스트레이터입니다.
사용자의 요청을 분석하여 적절한 A2A 에이전트들을 순서대로 실행하는 계획을 세우세요.

사용 가능한 A2A 에이전트들:
{agents_text}

다음 JSON 형식으로 응답하세요:
{{
    "plan_type": "ai_ds_team_orchestration",
    "objective": "분석 목표 요약",
    "reasoning": "이 계획을 선택한 이유",
    "steps": [
        {{
            "step_number": 1,
            "agent_name": "에이전트_이름",
            "task_description": "구체적인 작업 설명",
            "reasoning": "이 단계가 필요한 이유"
        }}
    ]
}}"""

            user_prompt = f"""사용자 요청: {user_query}

위 요청에 대해 AI DS Team A2A 에이전트들을 활용한 실행 계획을 JSON 형식으로 생성해주세요."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # JSON 추출
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', plan_text, re.DOTALL)
            if json_match:
                plan_text = json_match.group(1)
            elif plan_text.startswith('{'):
                pass
            else:
                start_idx = plan_text.find('{')
                end_idx = plan_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    plan_text = plan_text[start_idx:end_idx]
            
            execution_plan = json.loads(plan_text)
            logger.info(f"✅ LLM 기반 A2A 표준 실행 계획 생성 완료: {len(execution_plan.get('steps', []))}개 단계")
            return execution_plan
            
        except Exception as e:
            logger.error(f"❌ LLM 계획 생성 실패: {e}")
            return None

    def _create_fallback_plan(self, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """A2A 표준 폴백 계획 생성"""
        steps = []
        step_number = 1
        
        # 기본적인 데이터 분석 워크플로우
        basic_workflow = [
            ("data_loader", "데이터 로드 및 기본 검증"),
            ("data_cleaning", "데이터 품질 확인 및 정리"),
            ("eda_tools", "탐색적 데이터 분석 수행"),
            ("data_visualization", "데이터 시각화 및 인사이트 도출")
        ]
        
        for agent_name, task_desc in basic_workflow:
            if agent_name in available_agents:
                steps.append({
                    "step_number": step_number,
                    "agent_name": agent_name,
                    "task_description": task_desc,
                    "reasoning": f"{available_agents[agent_name]['description']} 전문 역량 활용"
                })
                step_number += 1
        
        return {
            "plan_type": "ai_ds_team_orchestration",
            "objective": "기본 데이터 분석 워크플로우 실행",
            "reasoning": "사용자 요청에 대한 포괄적인 데이터 분석을 수행하기 위한 표준 A2A 단계들",
            "steps": steps
        }


def create_standard_orchestrator_server():
    """A2A SDK 0.2.9 완전 표준 준수 오케스트레이터 서버 생성"""
    
    agent_card = AgentCard(
        name="AI DS Team LLM Enhanced Orchestrator v6 Stable",
        description="AI Data Science Team의 A2A 표준 준수 멀티 에이전트 오케스트레이터",
        url="http://localhost:8100",
        version="5.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="orchestrate_analysis",
                name="AI DS Team A2A Orchestration",
                description="AI Data Science Team A2A 에이전트들을 조정하여 표준 준수 데이터 분석을 실행합니다.",
                tags=["orchestration", "multi-agent", "data-science", "a2a-standard"]
            )
        ]
    )
    
    executor = LLMEnhancedOrchestratorExecutor()
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
    logger.info("🚀 Starting A2A Standard Orchestrator Server v5.0 (A2A SDK 0.2.9 Full Compliance)")
    
    app = create_standard_orchestrator_server()
    
    uvicorn.run(
        app.build(),
        host="localhost",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main() 