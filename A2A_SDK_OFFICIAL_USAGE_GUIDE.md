# A2A SDK 0.2.9 공식 사용법 가이드

## 📋 개요

공식 A2A Protocol 문서와 예제를 바탕으로 정확한 A2A SDK 사용법을 정리한 가이드입니다.

## 🔧 설치

```bash
# 기본 설치
pip install a2a-sdk

# gRPC 지원 포함
pip install "a2a-sdk[grpc]"
```

**요구사항**: Python 3.10+

## 📚 핵심 구성 요소

### 1. AgentExecutor 패턴

모든 A2A 에이전트는 `AgentExecutor` 클래스를 상속받아 구현합니다:

```python
from typing_extensions import override
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

class YourAgentExecutor(AgentExecutor):
    """사용자 정의 AgentExecutor"""
    
    def __init__(self):
        self.agent = YourAgent()  # 실제 에이전트 로직
    
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 사용자 입력 처리
        query = context.get_user_input()
        
        # 에이전트 실행
        result = await self.agent.invoke(query)
        
        # 결과를 이벤트 큐에 전송
        event_queue.enqueue_event(new_agent_text_message(result))
    
    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        # 취소 로직 구현
        raise Exception('cancel not supported')
```

### 2. 기본 Agent 클래스

```python
class YourAgent:
    """실제 에이전트 로직을 담은 클래스"""
    
    async def invoke(self, query: str) -> str:
        # 실제 처리 로직
        return f"Processed: {query}"
```

### 3. 서버 설정

```python
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

def main():
    # Agent Skill 정의
    skill = AgentSkill(
        id="your_skill_id",
        name="Your Skill Name",
        description="상세한 스킬 설명",
        tags=["tag1", "tag2"],
        examples=["예시1", "예시2"]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Your Agent Name",
        description="에이전트 설명",
        url="http://localhost:8080/",  # 실제 실행 포트와 일치
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=YourAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # 서버 실행
    uvicorn.run(server.build(), host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
```

## 🚀 고급 패턴: 스트리밍 및 TaskUpdater

### 1. 스트리밍 에이전트

```python
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_task, new_text_artifact

class StreamingAgentExecutor(AgentExecutor):
    """스트리밍을 지원하는 AgentExecutor"""
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        task = context.current_task
        
        # Task 초기화
        if not task:
            task = new_task(context.message)
            event_queue.enqueue_event(task)
        
        # 스트리밍 처리
        async for event in self.agent.stream(query, task.contextId):
            if event['is_working']:
                # 진행 중 상태 업데이트
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.working)
                    )
                )
            
            elif event['is_task_complete']:
                # 최종 결과 전송
                event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        artifact=new_text_artifact(
                            name='final_result',
                            text=event['content']
                        )
                    )
                )
                
                # 완료 상태로 업데이트
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed)
                    )
                )
```

### 2. Event-based Agent

```python
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState

class EventAgentExecutor(AgentExecutor):
    """이벤트 기반 AgentExecutor"""
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # TaskUpdater 생성
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Task 제출 및 시작
        task_updater.submit()
        task_updater.start_work()
        
        # 실제 작업 수행
        user_message = context.message.parts[0].root.text
        
        # 작업 시뮬레이션
        await asyncio.sleep(1)
        
        # 결과에 따른 상태 업데이트
        if "success" in user_message:
            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="작업이 성공적으로 완료되었습니다.")]
                ),
            )
        else:
            task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="작업 처리 중 오류가 발생했습니다.")]
                ),
            )
```

## 📋 클라이언트 테스트

```python
import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

async def test_agent():
    base_url = 'http://localhost:8080'
    
    async with httpx.AsyncClient() as httpx_client:
        # Agent Card 조회
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        public_card = await resolver.get_agent_card()
        
        # A2A Client 초기화
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=public_card
        )
        
        # 메시지 전송
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'Hello, please help me!'}
                ],
                'messageId': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**send_message_payload)
        )
        
        # 응답 받기
        response = await client.send_message(request)
        return response.model_dump(mode='json', exclude_none=True)

# 실행
if __name__ == '__main__':
    result = asyncio.run(test_agent())
    print(result)
```

## 🔍 핵심 원칙

### 1. 정확한 Import 패턴
```python
# 필수 imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

# TaskUpdater 사용 시
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TaskState

# 스트리밍 사용 시
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus
)
from a2a.utils import new_task, new_text_artifact
```

### 2. Agent Card URL 일치
- Agent Card의 `url` 필드는 실제 실행 포트와 **정확히 일치**해야 함
- 예: 8306 포트 → `http://localhost:8306/`

### 3. 비동기 처리
- 모든 `execute()` 메서드는 `async def`로 정의
- `await`를 사용한 비동기 작업 처리

### 4. Event Queue 사용
- 모든 응답은 `event_queue.enqueue_event()`를 통해 전송
- `new_agent_text_message()` 유틸리티 사용 권장

## ⚠️ 주의사항

1. **TaskState 관리**: 적절한 상태 전환 (submit → start_work → completed/failed)
2. **Context 처리**: `context.get_user_input()` 또는 `context.message.parts[0].root.text` 사용
3. **오류 처리**: try/catch로 적절한 오류 상태 관리
4. **포트 일치**: Agent Card URL과 실제 서버 포트 일치 필수

이 가이드를 바탕으로 모든 A2A 에이전트를 일관성 있게 구현할 수 있습니다.