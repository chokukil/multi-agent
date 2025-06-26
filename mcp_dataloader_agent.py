import asyncio
import logging
import os
import sys
import uvicorn
import uuid
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, Message, Task, AgentCapabilities
from a2a.utils.message import new_agent_text_message, get_message_text

# Import core modules
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from core.data_manager import DataManager

logger = logging.getLogger(__name__)

# Global instance
data_manager = DataManager()

# 1. Define the core agent (Hello World Agent 패턴)
class DataLoaderAgent:
    """데이터 로더 에이전트 (Hello World Agent 패턴)"""
    
    async def invoke(self, user_input: str = "") -> str:
        """데이터 로드 수행 (Hello World Agent의 invoke 패턴)"""
        try:
            # 고정된 스킬 실행 - 데이터 로드
            return self.load_data(user_input)
        except Exception as e:
            logger.error(f"Error in DataLoaderAgent.invoke: {e}")
            return f"❌ 데이터 로드 중 오류가 발생했습니다: {str(e)}"

    def load_data(self, user_request: str = "", **kwargs) -> str:
        """Load data into DataManager"""
        logger.info(f"load_data called with request: {user_request}")
        
        try:
            # 샘플 데이터 로드 시뮬레이션
            success_count = 0
            
            # Check if Titanic data is available
            titanic_path = os.path.join(project_root, "artifacts", "data", "shared_dataframes", "titanic.csv.pkl")
            if os.path.exists(titanic_path):
                data_manager.add_dataframe(df_id="titanic", df_path=titanic_path)
                success_count += 1
                logger.info(f"Loaded Titanic dataset from {titanic_path}")
            
            # Check for other sample datasets
            data_dir = os.path.join(project_root, "artifacts", "data", "shared_dataframes")
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('.csv') or file.endswith('.pkl'):
                        file_path = os.path.join(data_dir, file)
                        df_id = file.replace('.csv', '').replace('.pkl', '')
                        if df_id != "titanic":  # Already handled above
                            data_manager.add_dataframe(df_id=df_id, df_path=file_path)
                            success_count += 1
                            logger.info(f"Loaded dataset {df_id} from {file_path}")
            
            # Return status
            available_dfs = data_manager.list_dataframes()
            return f"""✅ **데이터 로드 완료**

**로드된 데이터셋**: {success_count}개
**사용 가능한 데이터프레임**: {', '.join(available_dfs) if available_dfs else '없음'}

**다음 단계**: 
1. 📊 **EDA 분석** 페이지로 이동
2. 🎯 데이터 분석 요청

**상태**: 준비 완료 ✨
"""
        
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            return f"❌ 데이터 로드 실패: {str(e)}"

# 2. AgentExecutor 구현 (Hello World Agent 패턴)
class DataLoaderAgentExecutor(AgentExecutor):
    """Hello World Agent 패턴을 사용하는 DataLoader AgentExecutor"""
    
    def __init__(self):
        self.agent = DataLoaderAgent()
        logger.info("DataLoaderAgentExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (Hello World Agent 패턴)"""
        logger.info("DataLoaderAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출 (Hello World Agent 패턴)
            user_message = context.get_user_input()
            logger.info(f"사용자 입력: {user_message}")
            
            # 에이전트 실행 (Hello World Agent 패턴)
            result = await self.agent.invoke(user_message)
            
            # 결과 전송 (공식 패턴 - 중요: await 추가!)
            message = new_agent_text_message(result)
            await event_queue.enqueue_event(message)
            
            logger.info("Task completed successfully")
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            error_message = new_agent_text_message(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task 취소 처리 (Hello World Agent 패턴)"""
        logger.info("DataLoaderAgentExecutor.cancel() 호출됨")
        raise Exception("Cancel not supported")

# 3. Agent Card 생성
def create_agent_card() -> AgentCard:
    """A2A 표준 Agent Card 생성"""
    skill = AgentSkill(
        id="load_data",
        name="Load Data",
        description="Load sample datasets into the data manager",
        tags=["data", "load", "management"],
        examples=["Load data", "Import datasets", "Prepare data"]
    )
    
    return AgentCard(
        name="Data Loader Agent", 
        description="Loads and manages datasets for analysis",
        url="http://localhost:8001/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

# 4. Wire everything together
def main():
    """A2A 표준 Data Loader 서버 실행"""
    logging.basicConfig(level=logging.INFO)
    logger.info("🚀 Starting Data Loader A2A Server...")
    
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # RequestHandler 초기화
    request_handler = DefaultRequestHandler(
        agent_executor=DataLoaderAgentExecutor(),
        task_store=InMemoryTaskStore()
    )
    
    # A2A Starlette Application 생성
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info("🌐 Server starting at http://localhost:8001")
    logger.info("📋 Agent Card available at /.well-known/agent.json")
    
    # Uvicorn으로 서버 실행
    uvicorn.run(a2a_app.build(), host="localhost", port=8001)

if __name__ == "__main__":
    main() 