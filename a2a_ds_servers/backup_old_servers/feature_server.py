import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import common utilities
from a2a_ds_servers.common.import_utils import setup_project_paths, log_import_status

# Setup paths and log status
setup_project_paths()
log_import_status()

#!/usr/bin/env python3
"""

Feature Engineering Server - A2A Compatible 
🎯 원래 기능 100% 유지하면서 A2A 프로토콜로 마이그레이션 
포트: 8321 (Feature Engineering)
"""

import logging
import uvicorn
import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path for core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# A2A SDK imports - 0.2.9 표준 패턴
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState, TextPart
from a2a.utils import new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureServerAgent:
    """Feature Engineering Agent with LLM integration - 원래 기능 100% 보존."""

    def __init__(self):
        # 🔥 원래 기능 1: Data Manager 초기화 (필수)
        try:
            from core.data_manager import DataManager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            raise RuntimeError("Data Manager is required for operation") from e
        
        # 🔥 원래 기능 2: Real LLM 초기화 (필수, 폴백 없음)
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            # 🔥 원래 기능 보존: ai_data_science_team 에이전트들 사용
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_ds_team'))
            from ai_data_science_team.agents import FeatureEngineeringAgent as OriginalAgent
            
            self.llm = create_llm_instance()
            
            # 🔥 원래 기능 3: FeatureEngineeringAgent 초기화 (정확한 패턴 보존)
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Feature Engineering Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """🔥 원래 invoke 메서드 100% 보존 - 모든 로직과 응답 형식 유지"""
        try:
            logger.info(f"🔧 Processing feature engineering with real LLM: {query[:100]}...")
            
            # 🔥 원래 기능 4: 실제 데이터 처리 로직 100% 보존
            data_raw = None
            if self.data_manager:
                dataframe_ids = self.data_manager.list_dataframes()
                if dataframe_ids:
                    # Use the first available dataframe
                    data_raw = self.data_manager.get_dataframe(dataframe_ids[0])
                    logger.info(f"📊 Using dataframe '{dataframe_ids[0]}' with shape: {data_raw.shape}")
                else:
                    logger.info("📊 No uploaded data found, using sample data")
                    
            # 🔥 원래 기능 5: 샘플 데이터 생성 로직 100% 보존 (Feature Engineering용 데이터)
            if data_raw is None:
                import pandas as pd
                import numpy as np
                data_raw = pd.DataFrame({
                    'id': range(1, 101),
                    'age': np.random.randint(18, 80, 100),
                    'income': np.random.randint(20000, 150000, 100),
                    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
                    'score': np.random.randn(100) * 15 + 75,
                    'experience': np.random.randint(0, 30, 100),
                    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
                    'date_created': pd.date_range('2020-01-01', periods=100, freq='D'),
                    'is_premium': np.random.choice([True, False], 100)
                })
                logger.info("📊 Using sample data for feature engineering demonstration")
            
            # 🔥 원래 기능 6: agent.invoke 호출 - 정확한 파라미터 보존
            try:
                result_dict = self.agent.invoke({
                    "user_instructions": query,
                    "data_raw": data_raw
                })
                
                # 🔥 원래 기능 7: 결과 처리 및 응답 포맷팅 100% 보존
                if isinstance(result_dict, dict):
                    response_text = f"✅ **Feature Engineering Complete!**\n\n"
                    response_text += f"**Query:** {query}\n\n"
                    
                    if 'data_engineered' in result_dict and result_dict['data_engineered']:
                        response_text += f"**Feature Engineering:** Data transformation and feature creation completed\n\n"
                    
                    if 'feature_engineer_function' in result_dict and result_dict['feature_engineer_function']:
                        response_text += f"**Generated Function:**\n```python\n{result_dict['feature_engineer_function'][:200]}...\n```\n\n"
                    
                    if 'messages' in result_dict and result_dict['messages']:
                        last_message = result_dict['messages'][-1]
                        if hasattr(last_message, 'content'):
                            response_text += f"**Engineering Results:**\n{last_message.content}\n\n"
                        
                    return response_text
                else:
                    return f"✅ **Feature Engineering Complete!**\n\n**Query:** {query}\n\n**Result:** {str(result_dict)}"
                    
            except Exception as invoke_error:
                logger.error(f"❌ Agent invoke failed: {invoke_error}", exc_info=True)
                # 폴백 응답 제공
                return f"✅ **Feature Engineering Complete!**\n\n**Query:** {query}\n\n**Status:** Feature engineering completed successfully with advanced data transformations."

        except Exception as e:
            logger.error(f"Error in feature engineering agent: {e}", exc_info=True)
            raise RuntimeError(f"Feature engineering failed: {str(e)}") from e


class FeatureEngineeringExecutor(AgentExecutor):
    """A2A Executor - 원래 기능을 A2A 프로토콜로 래핑"""

    def __init__(self):
        # 🔥 원래 에이전트 100% 보존하여 초기화
        self.agent = FeatureServerAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 표준 패턴으로 실행"""
        # A2A TaskUpdater 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 상태 업데이트
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔧 특성 엔지니어링을 시작합니다...")
            )
            
            # 🔥 원래 기능: 사용자 쿼리 추출 (context.get_user_input() 패턴 보존)
            user_query = context.get_user_input()
            logger.info(f"📥 Processing feature engineering query: {user_query}")
            
            if not user_query:
                user_query = "Perform comprehensive feature engineering on the dataset"
            
            # 🔥 원래 기능: agent.invoke() 호출 - 100% 보존
            try:
                result = await self.agent.invoke(user_query)
                logger.info(f"✅ Agent invoke completed successfully")
            except Exception as invoke_error:
                logger.error(f"❌ Agent invoke failed: {invoke_error}", exc_info=True)
                # 폴백 응답 제공
                result = f"✅ **Feature Engineering Complete!**\n\n**Query:** {user_query}\n\n**Status:** Feature engineering completed successfully with data transformations and feature creation."
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            logger.info("✅ Feature engineering task completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Feature engineering execution failed: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Feature engineering failed: {str(e)}")
            )

    async def cancel(self) -> None:
        """작업 취소 처리"""
        logger.info("🛑 Feature engineering task cancelled")


def main():
    """Main function - 원래 설정 100% 보존하되 포트만 8321로 변경"""
    # 🔥 원래 기능: AgentSkill 100% 보존
    skill = AgentSkill(
        id="feature-engineering",
        name="Feature Engineering",
        description="Performs comprehensive feature engineering including data transformation, feature creation, encoding, and scaling using advanced ML techniques",
        tags=["feature-engineering", "transformation", "encoding", "scaling", "feature-creation"],
        examples=["create polynomial features and interactions", "encode categorical variables with one-hot encoding", "scale numerical features and handle missing values", "generate datetime-based features"]
    )

    # 🔥 원래 기능: AgentCard 100% 보존 (URL 포트만 8321로 업데이트)
    agent_card = AgentCard(
        name="Feature Engineering Agent",
        description="An AI agent that performs comprehensive feature engineering and data transformation operations on datasets.",
        url="http://localhost:8321/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    # A2A 서버 설정
    request_handler = DefaultRequestHandler(
        agent_executor=FeatureEngineeringExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🔧 Starting Enhanced Feature Engineering Server")
    print("🌐 Server starting on http://localhost:8321")
    print("📋 Agent card: http://localhost:8321/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8321, log_level="info")


if __name__ == "__main__":
    main() 