import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

# 실제 DataLoaderToolsAgent를 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_data_science_team.agents.data_loader_tools_agent import DataLoaderToolsAgent
from core.llm_factory import create_llm_instance

class DataLoaderAgent:
    """Real Data Loader Agent using ai_data_science_team."""

    def __init__(self):
        self.llm = create_llm_instance()
        self.agent = DataLoaderToolsAgent(model=self.llm)

    async def invoke(self, user_message: str) -> str:
        try:
            # 실제 LangGraph 에이전트 실행
            await self.agent.ainvoke_agent(user_instructions=user_message)
            
            # 결과 가져오기
            ai_response = self.agent.get_ai_message(markdown=False)
            artifacts = self.agent.get_artifacts(as_dataframe=False)
            
            # 결과 포맷팅
            result = f"**Data Loader Response:**\n\n{ai_response}"
            
            if artifacts:
                result += f"\n\n**Generated Artifacts:**\n```json\n{artifacts}\n```"
            
            return result
            
        except Exception as e:
            return f"Error processing data loading request: {str(e)}"


class DataLoaderExecutor(AgentExecutor):
    """Data Loader AgentExecutor Implementation with TaskUpdater pattern."""

    def __init__(self):
        self.agent = DataLoaderAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # TaskUpdater 초기화 (A2A SDK v0.2.6+ 필수 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Task 제출 및 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_message = context.get_user_input() if context else "No message provided"
            
            if not user_message:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ No valid message provided")])
                )
                return
            
            # 진행 상황 알림
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="🔄 Initializing Data Loader Agent...")])
            )
            
            # 에이전트 실행
            result = await self.agent.invoke(user_message)
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
        except Exception as e:
            # Task 실패 처리
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"❌ Error: {str(e)}")])
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # TaskUpdater 패턴으로 취소 처리
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.update_status(
            TaskState.canceled,
            message=task_updater.new_agent_message(parts=[TextPart(text="❌ Data loading operation cancelled")])
        )


if __name__ == '__main__':
    # AgentSkill 정의
    skill = AgentSkill(
        id='data_loader',
        name='Load data from files and directories',
        description='Loads data from various file sources, inspects file systems, and provides data loading capabilities using advanced tools',
        tags=['data', 'loader', 'filesystem', 'csv', 'json', 'parquet'],
        examples=[
            'load data from file.csv', 
            'list directory contents', 
            'read file information',
            'load all CSV files from directory',
            'inspect data structure'
        ],
    )

    # AgentCard 정의
    agent_card = AgentCard(
        name='AI Data Loader Agent',
        description='Advanced data loading agent powered by LangGraph that can load data from various sources, inspect file systems, and provide comprehensive data loading capabilities',
        url='http://localhost:8001/',
        version='2.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # RequestHandler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataLoaderExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # 서버 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # 서버 실행
    uvicorn.run(server.build(), host='0.0.0.0', port=8001) 