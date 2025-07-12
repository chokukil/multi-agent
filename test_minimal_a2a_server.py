#!/usr/bin/env python3
"""
최소한의 A2A 서버 테스트

A2A SDK 0.2.9 호환성 테스트를 위한 최소 구현
"""

import asyncio
import uvicorn
from typing import Any

# A2A SDK 0.2.9 Import
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities


class MinimalAgentExecutor(AgentExecutor):
    """최소한의 에이전트 실행기"""
    
    def __init__(self):
        print("✅ Minimal Agent Executor 초기화")
    
    async def cancel(self) -> None:
        """취소 메서드"""
        print("🛑 Minimal Agent Executor 취소 요청")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """실행 메서드"""
        try:
            print(f"📝 실행 요청 받음: {context}")
            
            # 상태 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🔄 최소 에이전트가 작업을 시작합니다..."
            )
            
            # 간단한 응답
            await task_updater.update_status(
                TaskState.completed,
                message="✅ 최소 에이전트 작업이 완료되었습니다!",
                final=True
            )
            
        except Exception as e:
            print(f"❌ 실행 오류: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=f"❌ 에러가 발생했습니다: {str(e)}",
                final=True
            )


async def create_minimal_server():
    """최소 A2A 서버 생성"""
    
    # Agent Skills
    skills_list = [
        AgentSkill(
            id="minimal_test",
            name="minimal_test",
            description="최소 테스트 스킬",
            tags=["test", "minimal"]
        )
    ]
    
    # Agent Card 설정
    agent_card = AgentCard(
        name="Minimal Test Agent",
        description="A2A SDK 0.2.9 호환성 테스트용 최소 에이전트",
        version="1.0.0",
        url="http://localhost:8316",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=skills_list,
        capabilities=AgentCapabilities(
            skills=skills_list
        )
    )
    
    # A2A 애플리케이션 생성
    executor = MinimalAgentExecutor()
    task_store = InMemoryTaskStore()
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(executor, task_store)
    )
    
    return app


if __name__ == "__main__":
    import sys
    
    # 명령행 인자 확인
    port = 8316
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ 잘못된 포트 번호: {sys.argv[1]}, 기본값 8316 사용")
    
    # 서버 정보 출력
    print(f"🚀 Minimal A2A 서버 시작")
    print(f"📍 주소: http://0.0.0.0:{port}")
    print(f"🔧 Agent Card: http://0.0.0.0:{port}/.well-known/agent.json")
    
    # 앱 생성
    app = asyncio.run(create_minimal_server())
    
    # 서버 실행
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("🛑 서버가 종료되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}") 