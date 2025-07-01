#!/usr/bin/env python3
"""
CherryAI Simple Orchestrator - A2A SDK v0.2.9 호환
안정적이고 간단한 오케스트레이터
"""

import asyncio
import logging
import os
import uvicorn
from typing import Optional

from openai import AsyncOpenAI

# A2A SDK imports
from a2a.server.application import A2AStarletteApplication
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.events.event_queue import EventQueue
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, TextPart, RequestContext, TaskState
from a2a.server.request_handler import DefaultRequestHandler
from a2a.client.types import new_agent_text_message
from a2a.server.executor import AgentExecutor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CherryAI_SimpleOrchestrator(AgentExecutor):
    """CherryAI 간단한 오케스트레이터"""
    
    def __init__(self):
        super().__init__()
        self.openai_client = self._initialize_openai_client()
        logger.info("🚀 CherryAI Simple Orchestrator 초기화 완료")
    
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
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 표준 프로토콜 기반 실행 메서드"""
        # TaskUpdater 초기화
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await updater.start_work()
            
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            logger.info(f"📝 사용자 요청: {user_input[:100]}...")
            
            # 진행 상황 업데이트
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧠 요청을 분석하고 있습니다...")
            )
            
            # 응답 생성
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📝 답변을 생성하고 있습니다...")
            )
            
            response = await self._generate_response(user_input)
            
            # 최종 결과 전송
            await updater.add_artifact(
                [TextPart(text=response)],
                name="comprehensive_analysis"
            )
            
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("✅ 분석이 완료되었습니다!")
            )
            
            await updater.complete()
            
        except Exception as e:
            logger.error(f"오케스트레이터 실행 중 오류 발생: {str(e)}")
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"오류 발생: {str(e)}")
            )
            raise
    
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
    
    async def _generate_response(self, user_input: str) -> str:
        """응답 생성"""
        try:
            if not self.openai_client:
                return self._generate_fallback_response(user_input)
            
            # OpenAI를 사용한 응답 생성
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """당신은 CherryAI 시스템의 전문 데이터 분석 어시스턴트입니다.
                        
사용자의 요청을 분석하여 다음과 같은 형태로 응답해주세요:

# 요청 분석 결과

## 📊 요청 내용 분석
- 사용자가 요청한 내용을 명확히 파악하고 설명

## 🔍 분석 접근법
- 해당 요청을 처리하기 위한 적절한 분석 방법론 제시

## 📈 권장 분석 단계
1. 첫 번째 단계
2. 두 번째 단계  
3. 세 번째 단계

## 💡 기대 결과
- 분석을 통해 얻을 수 있는 인사이트와 가치

## 🚀 다음 단계
- 실제 분석을 위해 필요한 데이터나 추가 정보

항상 전문적이고 실용적인 조언을 제공해주세요."""
                    },
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI 응답 생성 실패: {e}")
            return self._generate_fallback_response(user_input)
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """기본 응답 생성 (OpenAI 없을 때)"""
        return f"""# {user_input}에 대한 분석 결과

## 📊 요청 내용 분석
사용자께서 "{user_input}"에 대한 분석을 요청하셨습니다.

## 🔍 분석 접근법
CherryAI 시스템에서는 다음과 같은 단계로 분석을 진행합니다:

## 📈 권장 분석 단계
1. **데이터 수집 및 로딩**: 필요한 데이터를 수집하고 시스템에 로드
2. **탐색적 데이터 분석 (EDA)**: 데이터의 구조와 특성을 파악
3. **데이터 전처리**: 분석에 적합하도록 데이터 정리 및 변환
4. **분석 실행**: 요청된 분석 수행
5. **결과 해석 및 시각화**: 분석 결과를 이해하기 쉽게 정리

## 💡 기대 결과
- 데이터 기반의 객관적인 인사이트 제공
- 시각화를 통한 직관적인 결과 표현
- 실무에 적용 가능한 구체적인 권장사항

## 🚀 다음 단계
구체적인 분석을 위해 다음 정보가 필요합니다:
- 분석하고자 하는 데이터셋
- 구체적인 분석 목표
- 원하는 결과 형태

CherryAI 시스템이 도움을 드릴 준비가 되어있습니다!"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info("❌ CherryAI Simple 작업이 취소되었습니다")
        raise Exception('cancel not supported')


def create_agent_card() -> AgentCard:
    """CherryAI Simple 에이전트 카드 생성"""
    return AgentCard(
        name="CherryAI Simple Orchestrator",
        description="A2A SDK v0.2.9 호환 안정적인 간단 오케스트레이터",
        url="http://localhost:8100",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="simple_analysis",
                name="Simple Data Analysis",
                description="간단하고 안정적인 데이터 분석 및 요청 처리",
                tags=["analysis", "simple", "stable"],
                examples=[
                    "데이터를 분석해주세요",
                    "이 요청을 처리해주세요",
                    "분석 계획을 세워주세요"
                ]
            )
        ],
        supportsAuthenticatedExtendedCard=False
    )


async def main():
    """CherryAI Simple 서버 시작"""
    try:
        # 에이전트 카드 생성
        agent_card = create_agent_card()
        
        # 태스크 스토어 및 실행자 초기화
        task_store = InMemoryTaskStore()
        agent_executor = CherryAI_SimpleOrchestrator()
        
        # 요청 핸들러 생성
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store,
        )
        
        # A2A 애플리케이션 생성
        app_builder = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        app = app_builder.build()
        
        # 서버 시작
        print("🚀 CherryAI Simple Orchestrator 시작")
        print(f"📍 Agent Card: http://localhost:8100/.well-known/agent.json")
        print("🛑 종료하려면 Ctrl+C를 누르세요")
        
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌서버 시작 실패: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 