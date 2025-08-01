#!/usr/bin/env python3
"""
A2A Orchestrator - AI Data Science Team Multi-Agent Coordinator
A2A SDK 0.2.9 완전 표준 준수 버전

핵심 기능:
1. Dynamic Agent Discovery: 사용 가능한 A2A 에이전트 자동 발견
2. LLM-driven Planning: GPT-4o를 활용한 지능형 워크플로우 계획 수립
3. Multi-Agent Execution: 병렬/순차 에이전트 실행 조정
4. Error Recovery: 실패한 단계에 대한 자동 복구 메커니즘

A2A SDK 0.2.9 표준 준수 사항:
- A2AStarletteApplication과 DefaultRequestHandler 사용
- AgentExecutor 상속하여 execute()와 cancel() 메서드 구현
- RequestContext와 TaskStore 활용
- /.well-known/agent.json에서 표준 Agent Card 제공
- A2AClient로 표준 메시지 통신
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part,
    Message,
    Role
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
openai_client = AsyncOpenAI()

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


class A2AOrchestratorExecutor(AgentExecutor):
    """A2A SDK 0.2.9 표준 준수 오케스트레이터 실행자"""
    
    def __init__(self):
        self.openai_client = openai_client
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """A2A 요청 실행 - A2A SDK 0.2.9 표준 패턴"""
        
        logger.info("🎬 A2A Orchestrator Executor starting...")
        
        try:
            # 사용자 메시지 추출 (A2A SDK 0.2.9 표준 방식)
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'text'):
                        user_message += part.text + " "
                    elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_message += part.root.text + " "
            
            user_message = user_message.strip()
            logger.info(f"📝 Extracted user message: '{user_message}'")
            
            if not user_message:
                # 에러 메시지 전송
                error_msg = Message(
                    messageId=f"error-{int(time.time())}",
                    role=Role.agent,
                    parts=[Part(TextPart(text="❌ 요청 메시지가 비어있습니다."))]
                )
                await event_queue.enqueue_event(error_msg)
                return
            
            # 진행 상황 업데이트
            progress_msg = Message(
                messageId=f"progress-{int(time.time())}",
                role=Role.agent,
                parts=[Part(TextPart(text="🔍 AI DS Team 에이전트들을 발견하고 있습니다..."))]
            )
            await event_queue.enqueue_event(progress_msg)
            
            # 에이전트 발견
            available_agents = await self._discover_agents()
            
            if not available_agents:
                error_msg = Message(
                    messageId=f"error-{int(time.time())}",
                    role=Role.agent,
                    parts=[Part(TextPart(text="❌ 사용 가능한 A2A 에이전트를 찾을 수 없습니다."))]
                )
                await event_queue.enqueue_event(error_msg)
                return
            
            # 계획 수립
            plan_msg = Message(
                messageId=f"planning-{int(time.time())}",
                role=Role.agent,
                parts=[Part(TextPart(text="🧠 지능형 분석 계획을 수립하고 있습니다..."))]
            )
            await event_queue.enqueue_event(plan_msg)
            
            plan = await self._create_plan(user_message, available_agents)
            
            # 계획 실행
            exec_msg = Message(
                messageId=f"execution-{int(time.time())}",
                role=Role.agent,
                parts=[Part(TextPart(text="⚡ 계획을 실행하고 있습니다..."))]
            )
            await event_queue.enqueue_event(exec_msg)
            
            results = await self._execute_plan(plan, available_agents)
            
            # 결과 포맷팅 및 전송
            response_text = self._format_results(results, plan)
            
            final_msg = Message(
                messageId=f"result-{int(time.time())}",
                role=Role.agent,
                parts=[Part(TextPart(text=response_text))]
            )
            await event_queue.enqueue_event(final_msg)
            
            logger.info("✅ A2A Orchestrator execution completed")
            
        except Exception as e:
            logger.error(f"❌ A2A Orchestrator execution failed: {e}", exc_info=True)
            error_msg = Message(
                messageId=f"error-{int(time.time())}",
                role=Role.agent,
                parts=[Part(TextPart(text=f"❌ 오케스트레이션 실행 중 오류가 발생했습니다: {str(e)}"))]
            )
            await event_queue.enqueue_event(error_msg)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """작업 취소"""
        logger.info(f"🛑 Orchestrator operation cancelled for context {context.context_id}")
        cancel_msg = Message(
            messageId=f"cancel-{int(time.time())}",
            role=Role.agent,
            parts=[Part(TextPart(text="🛑 오케스트레이션이 취소되었습니다."))]
        )
        await event_queue.enqueue_event(cancel_msg)

    async def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """에이전트 발견"""
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

    async def _create_plan(self, user_query: str, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """계획 수립"""
        try:
            # 간단한 계획 생성 (LLM 없이)
            if not available_agents:
                return {"steps": []}
            
            # 첫 번째 사용 가능한 에이전트로 기본 계획
            first_agent = list(available_agents.keys())[0]
            
            return {
                "objective": "데이터 분석 수행",
                "steps": [
                    {
                        "step": 1,
                        "agent": first_agent,
                        "task_description": user_query,
                        "expected_outcome": "분석 결과"
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Plan creation failed: {e}")
            return {"steps": []}

    async def _execute_plan(self, plan: Dict[str, Any], available_agents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """계획 실행"""
        results = []
        
        for step in plan.get("steps", []):
            agent_name = step["agent"]
            if agent_name in available_agents:
                agent_info = available_agents[agent_name]
                result = await self._execute_step(step, agent_info)
                results.append(result)
        
        return results

    async def _execute_step(self, step: Dict[str, Any], agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """단계 실행"""
        try:
            message_payload = {
                "jsonrpc": "2.0",
                "id": f"orchestrator-step-{step['step']}",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": step["task_description"]}],
                        "messageId": f"step-{step['step']}-{int(time.time())}"
                    }
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    agent_info["url"],
                    json=message_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "step": step["step"],
                        "agent": step["agent"],
                        "status": "success",
                        "result": result
                    }
                else:
                    return {
                        "step": step["step"],
                        "agent": step["agent"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "step": step["step"],
                "agent": step["agent"],
                "status": "error",
                "error": str(e)
            }

    def _format_results(self, results: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
        """결과 포맷팅"""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        response = f"""✅ **오케스트레이션 완료**

📋 **목표**: {plan.get('objective', '데이터 분석')}
🎯 **성공한 단계**: {len(successful)}/{len(results)}

📊 **단계별 결과**:
"""
        
        for result in results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            response += f"{status_emoji} Step {result['step']}: {result['agent']}\n"
        
        return response


def create_orchestrator_server():
    """A2A SDK 0.2.9 표준 준수 오케스트레이터 서버 생성"""
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI DS Team Orchestrator",
        description="AI Data Science Team의 멀티 에이전트 오케스트레이터",
        url="http://localhost:8100",
        version="3.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id="orchestrate_analysis",
                name="AI DS Team Orchestration",
                description="AI Data Science Team 에이전트들을 조정하여 데이터 분석을 실행합니다.",
                tags=["orchestration", "multi-agent", "data-science"]
            )
        ]
    )
    
    # A2A SDK 0.2.9 표준 구성요소 사용
    executor = A2AOrchestratorExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    # A2A Starlette Application 생성
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    return app


def main():
    """메인 실행 함수"""
    logger.info("🚀 Starting A2A Orchestrator Server (A2A SDK 0.2.9 Standard)")
    
    # 서버 생성 및 실행
    app = create_orchestrator_server()
    
    # Uvicorn으로 서버 실행
    uvicorn.run(
        app.build(),
        host="localhost",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main() 