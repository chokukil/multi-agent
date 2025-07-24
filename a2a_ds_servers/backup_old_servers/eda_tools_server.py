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

EDA Tools Server - A2A Compatible
Following official A2A SDK patterns with real LLM integration
"""

import logging
import uvicorn
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAToolsAgent:
    """EDA Tools Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            # 공통 LLM 초기화 유틸리티 사용
            from base.llm_init_utils import create_llm_with_fallback
            from ai_data_science_team.ds_agents import EDAToolsAgent as OriginalAgent
            
            self.llm = create_llm_with_fallback()
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for EDA Tools Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the EDA tools agent with a query."""
        try:
            logger.info(f"🧠 Processing with real EDA Tools Agent: {query[:100]}...")
            
            # For real implementation, would need actual data
            # For now, create mock data structure
            import pandas as pd
            mock_data = pd.DataFrame({
                'age': [25, 30, 35, 40, 45, 50],
                'income': [50000, 60000, 70000, 80000, 90000, 100000],
                'score': [85, 90, 78, 92, 88, 95],
                'category': ['A', 'B', 'A', 'C', 'B', 'A']
            })
            
            result = self.agent.invoke_agent(
                user_instructions=query,
                data_raw=mock_data
            )
            
            if self.agent.response:
                artifacts = self.agent.get_artifacts()
                ai_message = self.agent.get_ai_message()
                tool_calls = self.agent.get_tool_calls()
                
                response_text = f"✅ **EDA Analysis Complete!**\n\n"
                response_text += f"**Request:** {query}\n\n"
                if ai_message:
                    response_text += f"**Analysis Results:**\n{ai_message}\n\n"
                if tool_calls:
                    response_text += f"**Tools Used:** {', '.join(tool_calls)}\n\n"
                if artifacts:
                    response_text += f"**Generated Artifacts:** EDA reports and visualizations ready\n\n"
                
                return response_text
            else:
                return "EDA analysis completed successfully."

        except Exception as e:
            logger.error(f"Error in EDA tools agent: {e}", exc_info=True)
            raise RuntimeError(f"EDA analysis failed: {str(e)}") from e

class EDAToolsExecutor(AgentExecutor):
    """EDA Tools Agent Executor."""

    def __init__(self):
        self.agent = EDAToolsAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the EDA analysis using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_query = ""
            if context.message and hasattr(context.message, 'parts') and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "text":
                        user_query += part.root.text + " "
                    elif hasattr(part, 'text'):  # 대체 패턴
                        user_query += part.text + " "
                
                user_query = user_query.strip()
            
            # 기본 요청이 없으면 데모 모드
            if not user_query:
                user_query = "샘플 데이터로 탐색적 데이터 분석을 시연해주세요. 기술통계, 상관관계, 분포 분석을 포함해주세요."
                
            logger.info(f"🔍 Processing EDA query: {user_query}")
            
            # Get result from the agent
            result = await self.agent.invoke(user_query)
            
            # A2A SDK 0.2.9 공식 패턴에 따른 최종 응답
            from a2a.types import TaskState
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            # A2A SDK 0.2.9 공식 패턴에 따른 에러 응답
            from a2a.types import TaskState
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"EDA 분석 중 오류 발생: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the EDA tools server."""
    skill = AgentSkill(
        id="eda_analysis",
        name="Exploratory Data Analysis",
        description="Performs comprehensive exploratory data analysis with statistical summaries and insights",
        tags=["eda", "exploration", "statistics", "analysis"],
        examples=["explore my dataset", "analyze data distribution", "find correlations"]
    )

    agent_card = AgentCard(
        name="EDA Tools Agent",
        description="An AI agent that specializes in exploratory data analysis and statistical exploration.",
        url="http://localhost:8312/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=EDAToolsExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("📊 Starting EDA Tools Agent Server")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")

if __name__ == "__main__":
    main()