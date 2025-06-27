#!/usr/bin/env python3
"""
Data Cleaning Server - A2A Compatible
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

class DataCleaningAgent:
    """Data Cleaning Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.agents import DataCleaningAgent as OriginalAgent
            
            self.llm = create_llm_instance()
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Data Cleaning Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the data cleaning agent with a query."""
        try:
            logger.info(f"🧠 Processing with real Data Cleaning Agent: {query[:100]}...")
            
            # For real implementation, would need actual data
            # For now, create mock data structure
            import pandas as pd
            import numpy as np
            mock_data = pd.DataFrame({
                'id': [1, 2, 3, 4, 5, 6],
                'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Frank'],
                'age': [25, 30, np.nan, 40, 45, 35],
                'salary': [50000, 60000, 70000, 80000, np.inf, 55000]
            })
            
            result = self.agent.invoke_agent(
                data_raw=mock_data,
                user_instructions=query
            )
            
            if self.agent.response:
                data_cleaned = self.agent.get_data_cleaned()
                cleaner_function = self.agent.get_data_cleaner_function()
                recommended_steps = self.agent.get_recommended_cleaning_steps()
                
                response_text = f"✅ **Data Cleaning Complete!**\n\n"
                response_text += f"**Request:** {query}\n\n"
                if data_cleaned is not None:
                    response_text += f"**Cleaned Data Shape:** {data_cleaned.shape}\n\n"
                if cleaner_function:
                    response_text += f"**Generated Function:**\n```python\n{cleaner_function}\n```\n\n"
                if recommended_steps:
                    response_text += f"**Recommended Steps:** {recommended_steps}\n\n"
                
                return response_text
            else:
                return "Data cleaning completed successfully."

        except Exception as e:
            logger.error(f"Error in data cleaning agent: {e}", exc_info=True)
            raise RuntimeError(f"Data cleaning failed: {str(e)}") from e

class DataCleaningExecutor(AgentExecutor):
    """Data Cleaning Agent Executor."""

    def __init__(self):
        self.agent = DataCleaningAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the data cleaning using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"🧹 Processing data cleaning query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a data cleaning request."
            
            # Get result from the agent
            result = await self.agent.invoke(user_query)
            
            # Complete task with result
            from a2a.types import TaskState, TextPart
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            # Report error through TaskUpdater
            from a2a.types import TaskState, TextPart
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Data cleaning failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the data cleaning server."""
    skill = AgentSkill(
        id="data_cleaning",
        name="Data Cleaning",
        description="Cleans and preprocesses data by handling missing values, outliers, and data quality issues",
        tags=["cleaning", "preprocessing", "quality", "missing-values"],
        examples=["clean my dataset", "handle missing values", "remove outliers"]
    )

    agent_card = AgentCard(
        name="Data Cleaning Agent",
        description="An AI agent that specializes in data cleaning and quality improvement.",
        url="http://localhost:8205/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DataCleaningExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🧹 Starting Data Cleaning Agent Server")
    print("🌐 Server starting on http://localhost:8205")
    print("📋 Agent card: http://localhost:8205/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8205, log_level="info")

if __name__ == "__main__":
    main() 