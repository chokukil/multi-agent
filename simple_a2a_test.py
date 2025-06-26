#!/usr/bin/env python3
"""공식 A2A SDK 패턴 기반 간단한 테스트 서버"""

import asyncio
import logging
import uvicorn
import click

from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities,
    Message, Part, Role, TextPart, TaskState
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.apps import A2AStarletteApplication  
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTestAgentExecutor(AgentExecutor):
    """간단한 테스트용 AgentExecutor - 공식 A2A SDK 패턴"""
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK v0.2.6+ 표준 TaskUpdater 패턴을 사용한 실행"""
        logger.info("🎯 SimpleTestAgentExecutor.execute() 호출됨")
        
        # TaskUpdater 초기화 (A2A SDK v0.2.6+ 필수 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Task 제출 및 시작
            task_updater.submit()
            task_updater.start_work()
            
            # 사용자 입력 추출
            user_message = context.get_user_input() if context else "테스트 요청"
            logger.info(f"📝 사용자 입력: {user_message}")
            
            # 간단한 작업 시뮬레이션
            await asyncio.sleep(1)  # 실제 작업처럼 보이게
            
            # 결과 생성
            result = f"✅ 테스트 완료! 입력받은 메시지: '{user_message}'"
            
            # Task 완료 처리 (TaskUpdater 패턴)
            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
            logger.info("✅ Task completed successfully with TaskUpdater")
            
        except Exception as e:
            logger.error(f"❌ Error in execute: {e}", exc_info=True)
            # Task 실패 처리
            task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"❌ 실행 중 오류가 발생했습니다: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task 취소 처리 (A2A SDK v0.2.6+ TaskUpdater 패턴)"""
        logger.info("🛑 Task cancellation requested")
        
        # TaskUpdater 패턴으로 취소 처리
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        task_updater.update_status(
            TaskState.canceled,
            message=task_updater.new_agent_message(parts=[TextPart(text="❌ 분석이 취소되었습니다.")])
        )

def get_agent_card() -> AgentCard:
    """Agent Card 생성 (공식 A2A 표준 메타데이터)"""
    return AgentCard(
        name='Simple Test Agent',
        description='Simple test agent for A2A SDK validation',
        url='http://localhost:10003/',
        version='1.0.0',
        defaultInputModes=['text/plain'],
        defaultOutputModes=['text/plain'],
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        skills=[
            AgentSkill(
                id='simple_test',
                name='Simple Test',
                description='Simple test functionality',
                tags=['test', 'simple'],
                examples=[
                    "테스트 해줘",
                    "간단한 테스트",
                    "hello world"
                ]
            )
        ]
    )

@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10003)
def main(host: str, port: int):
    """서버 시작"""
    print(f"🚀 간단한 A2A 테스트 서버 시작: {host}:{port}")
    
    # Agent Executor 생성
    agent_executor = SimpleTestAgentExecutor()
    
    # Task Store 생성 
    task_store = InMemoryTaskStore()
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store
    )
    
    # A2A Application 생성
    agent_card = get_agent_card()
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    # 서버 실행
    try:
        uvicorn.run(a2a_app.build(), host=host, port=port)
    except KeyboardInterrupt:
        print("🛑 서버 종료")

if __name__ == "__main__":
    main() 