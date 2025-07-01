#!/usr/bin/env python3
"""
CherryAI v8 - Universal Intelligent Orchestrator
v7 장점 + A2A 프로토콜 극대화 + v8 실시간 스트리밍 통합
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

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
from a2a.client import A2ACardResolver, A2AClient

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

@dataclass
class DiscoveredAgent:
    """A2A 프로토콜로 발견된 에이전트 정보"""
    name: str
    url: str
    description: str
    skills: Dict[str, Any]
    capabilities: Dict[str, Any]
    agent_card: AgentCard
    last_seen: datetime
    health_status: str = "healthy"


class RealTimeStreamingTaskUpdater(TaskUpdater):
    """v8: 진정한 실시간 스트리밍 구현"""
    
    async def stream_update(self, content: str):
        """중간 결과 스트리밍"""
        await self.update_status(
            TaskState.working,
            message=self.new_agent_message(parts=[TextPart(text=content)])
        )
    
    async def stream_response_progressively(self, response: str, 
                                          chunk_size: int = 50,
                                          delay: float = 0.03) -> None:
        """문자 단위 실시간 스트리밍"""
        
        await self.stream_update("📝 답변을 실시간으로 생성하고 있습니다...")
        await asyncio.sleep(0.1)
        
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            await self.update_status(
                TaskState.working,
                message=self.new_agent_message(parts=[TextPart(text=chunk)])
            )
            await asyncio.sleep(delay)
        
        await self.stream_update("✅ 답변 생성이 완료되었습니다!")
        await asyncio.sleep(0.2)
        
        await self.update_status(
            TaskState.completed,
            message=self.new_agent_message(parts=[TextPart(text=response)])
        )


class CherryAI_v8_UniversalIntelligentOrchestrator(AgentExecutor):
    """CherryAI v8 - Universal Intelligent Orchestrator"""
    
    def __init__(self):
        super().__init__()
        self.openai_client = self._initialize_openai_client()
        self.discovered_agents = {}
        self.streaming_updater = None
        logger.info("🚀 CherryAI v8 Universal Intelligent Orchestrator 초기화 완료")
    
    def _initialize_openai_client(self) -> Optional[AsyncOpenAI]:
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return None
            return AsyncOpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            return None
    
    async def execute(self, context: RequestContext) -> None:
        """v8 메인 실행 로직"""
        
        self.streaming_updater = RealTimeStreamingTaskUpdater(
            context.task_updater.task_store,
            context.task_updater.task_id,
            context.task_updater.event_queue
        )
        
        try:
            user_input = self._extract_user_input(context)
            if not user_input:
                await self.streaming_updater.stream_update("❌ 사용자 입력을 찾을 수 없습니다.")
                return
            
            logger.info(f"📝 사용자 요청: {user_input[:100]}...")
            
            # A2A 동적 에이전트 발견
            await self.streaming_updater.stream_update("🔍 A2A 프로토콜로 에이전트를 발견하고 있습니다...")
            await self._discover_agents()
            
            # 복잡도 분석
            await self.streaming_updater.stream_update("🧠 요청 복잡도를 분석하고 있습니다...")
            complexity_assessment = await self._assess_request_complexity(user_input)
            
            complexity_level = complexity_assessment.get('complexity_level', 'complex')
            await self.streaming_updater.stream_update(
                f"📊 요청 복잡도: {complexity_level}"
            )
            
            # 복잡도별 처리
            if complexity_level == 'simple':
                await self._handle_simple_request(user_input)
            else:
                await self._handle_complex_request(user_input)
                
        except Exception as e:
            logger.error(f"❌ v8 실행 중 오류: {e}")
            await self.streaming_updater.stream_update(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """사용자 입력 추출"""
        try:
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part.root, 'kind') and part.root.kind == 'text':
                        return part.root.text
                    elif hasattr(part.root, 'type') and part.root.type == 'text':
                        return part.root.text
            return ""
        except Exception as e:
            logger.error(f"사용자 입력 추출 실패: {e}")
            return ""
    
    async def _discover_agents(self):
        """A2A 에이전트 발견"""
        self.discovered_agents = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for port in AGENT_PORTS.values():
                try:
                    endpoint = f"http://localhost:{port}"
                    resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                    agent_card = await resolver.get_agent_card()
                    
                    if agent_card:
                        self.discovered_agents[agent_card.name] = {
                            'url': endpoint,
                            'card': agent_card
                        }
                        logger.info(f"✅ A2A Agent discovered: {agent_card.name}")
                        
                except Exception as e:
                    logger.warning(f"Agent discovery failed for port {port}: {e}")
        
        await self.streaming_updater.stream_update(
            f"✅ {len(self.discovered_agents)}개의 A2A 에이전트를 발견했습니다!"
        )
    
    async def _assess_request_complexity(self, user_input: str) -> Dict:
        """복잡도 분석"""
        word_count = len(user_input.split())
        question_marks = user_input.count('?')
        
        if word_count < 10 and question_marks > 0:
            return {'complexity_level': 'simple'}
        else:
            return {'complexity_level': 'complex'}
    
    async def _handle_simple_request(self, user_input: str):
        """Simple 요청 처리"""
        await self.streaming_updater.stream_update("💡 간단한 요청으로 판단되어 즉시 답변을 생성합니다...")
        
        if not self.openai_client:
            await self.streaming_updater.stream_response_progressively(
                "죄송합니다. OpenAI API가 설정되지 않아 답변을 생성할 수 없습니다."
            )
            return
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"다음 질문에 답변하세요: {user_input}"}],
                temperature=0.5,
                timeout=60.0
            )
            
            answer = response.choices[0].message.content
            await self.streaming_updater.stream_response_progressively(answer)
            
        except Exception as e:
            await self.streaming_updater.stream_response_progressively(
                f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def _handle_complex_request(self, user_input: str):
        """Complex 요청 처리"""
        await self.streaming_updater.stream_update("🔄 복합적인 요청으로 에이전트들과 협력하여 처리합니다...")
        
        # 기본 에이전트 순서로 실행
        agent_sequence = ["Data Loader Agent", "EDA Tools Agent", "Data Visualization Agent"]
        results = []
        
        for agent_name in agent_sequence:
            if agent_name in self.discovered_agents:
                await self.streaming_updater.stream_update(f"🤖 {agent_name} 실행 중...")
                
                result = await self._execute_agent(agent_name, user_input)
                results.append(result)
                
                if result.get('status') == 'success':
                    await self.streaming_updater.stream_update(f"✅ {agent_name} 완료")
                else:
                    await self.streaming_updater.stream_update(f"⚠️ {agent_name} 실패: {result.get('error', '')}")
        
        # 최종 응답 생성
        await self._generate_final_response(results, user_input)
    
    async def _execute_agent(self, agent_name: str, instruction: str) -> Dict:
        """단일 에이전트 실행"""
        agent_info = self.discovered_agents.get(agent_name)
        if not agent_info:
            return {'status': 'failed', 'error': f'에이전트 {agent_name}을 찾을 수 없습니다'}
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                a2a_client = A2AClient(httpx_client=client, base_url=agent_info['url'])
                
                message = {
                    "parts": [{"kind": "text", "text": instruction}],
                    "messageId": f"msg_{int(time.time())}",
                    "role": "user"
                }
                
                response = await a2a_client.send_message(message)
                
                if response and hasattr(response, 'parts') and response.parts:
                    result_text = ""
                    for part in response.parts:
                        if hasattr(part.root, 'text'):
                            result_text += part.root.text
                    
                    return {'status': 'success', 'result': result_text}
                else:
                    return {'status': 'failed', 'error': '에이전트에서 응답을 받지 못했습니다'}
                    
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    async def _generate_final_response(self, results: List[Dict], user_input: str):
        """최종 응답 생성"""
        await self.streaming_updater.stream_update("📝 최종 종합 분석을 생성하고 있습니다...")
        
        if not self.openai_client:
            # 폴백 응답
            response = f"# {user_input}에 대한 분석 결과\n\n"
            for i, result in enumerate(results):
                response += f"## 단계 {i+1} 결과\n\n"
                if result.get('status') == 'success':
                    response += f"{result.get('result', '결과 없음')}\n\n"
                else:
                    response += f"⚠️ 실행 실패: {result.get('error', '알 수 없는 오류')}\n\n"
            
            await self.streaming_updater.stream_response_progressively(response)
            return
        
        # 성공한 결과들만 수집
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            await self.streaming_updater.stream_response_progressively(
                "죄송합니다. 모든 에이전트 실행이 실패했습니다."
            )
            return
        
        # LLM으로 최종 종합
        results_text = "\n\n".join([r.get('result', '') for r in successful_results])
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user", 
                    "content": f"""
                    사용자 요청: {user_input}
                    
                    에이전트 실행 결과들:
                    {results_text}
                    
                    위 결과들을 종합하여 사용자 요청에 대한 완전한 답변을 생성하세요.
                    구조화된 Markdown 형식으로 작성하세요.
                    """
                }],
                temperature=0.5,
                timeout=90.0
            )
            
            final_answer = response.choices[0].message.content
            await self.streaming_updater.stream_response_progressively(final_answer)
            
        except Exception as e:
            logger.error(f"최종 응답 생성 실패: {e}")
            await self.streaming_updater.stream_response_progressively(
                f"최종 응답 생성 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info("🛑 CherryAI v8 작업이 취소되었습니다")
        if self.streaming_updater:
            await self.streaming_updater.stream_update("🛑 작업이 사용자에 의해 취소되었습니다.")


def create_agent_card() -> AgentCard:
    """CherryAI v8 Agent Card 생성"""
    return AgentCard(
        name="CherryAI v8 Universal Intelligent Orchestrator",
        description="v7 장점 + A2A 프로토콜 극대화 + v8 실시간 스트리밍을 통합한 범용 지능형 오케스트레이터",
        skills=[
            AgentSkill(
                id="universal_analysis",
                name="범용 데이터 분석",
                description="모든 종류의 데이터 분석 요청을 A2A 에이전트들과 협력하여 처리",
                tags=["data-analysis", "orchestration", "a2a", "streaming"],
                examples=[
                    "데이터셋의 패턴과 이상치를 분석해주세요",
                    "매출 데이터의 트렌드를 예측해주세요", 
                    "고객 세분화 분석을 수행해주세요"
                ]
            ),
            AgentSkill(
                id="realtime_streaming",
                name="실시간 스트리밍",
                description="분석 과정과 결과를 실시간으로 스트리밍하여 사용자 경험 향상",
                tags=["streaming", "realtime", "progressive"],
                examples=[
                    "분석 진행 상황을 실시간으로 확인",
                    "결과를 점진적으로 받아보기"
                ]
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            supportsAuthenticatedExtendedCard=False
        )
    )


if __name__ == "__main__":
    task_store = InMemoryTaskStore()
    event_queue = EventQueue()
    
    orchestrator = CherryAI_v8_UniversalIntelligentOrchestrator()
    
    request_handler = DefaultRequestHandler(
        agent_executor=orchestrator,
        task_store=task_store,
        event_queue=event_queue
    )
    
    app = A2AStarletteApplication(
        agent_card=create_agent_card(),
        request_handler=request_handler
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info") 