#!/usr/bin/env python3
"""
Feature Engineering Server - A2A Compatible
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

class FeatureEngineeringAgent:
    """Feature Engineering Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.agents import FeatureEngineeringAgent as OriginalAgent
            
            self.llm = create_llm_instance()
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Feature Engineering Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the feature engineering agent with a query."""
        try:
            logger.info(f"🧠 Processing with real Feature Engineering Agent: {query[:100]}...")
            
            # For real implementation, would need actual data
            # For now, create mock data structure
            import pandas as pd
            mock_data = pd.DataFrame({
                'numeric_feature_1': [1.2, 3.4, 5.6, 7.8, 9.0, 2.1],
                'numeric_feature_2': [10, 20, 30, 40, 50, 15],
                'categorical_feature': ['A', 'B', 'A', 'C', 'B', 'A'],
                'target': [0, 1, 0, 1, 1, 0]
            })
            
            result = self.agent.invoke_agent(
                data_raw=mock_data,
                user_instructions=query,
                target_variable="target"
            )
            
            if self.agent.response:
                data_engineered = self.agent.get_data_engineered()
                feature_function = self.agent.get_feature_engineer_function()
                recommended_steps = self.agent.get_recommended_feature_engineering_steps()
                
                response_text = f"✅ **Feature Engineering Complete!**\n\n"
                response_text += f"**Request:** {query}\n\n"
                if data_engineered is not None:
                    response_text += f"**Engineered Data Shape:** {data_engineered.shape}\n\n"
                if feature_function:
                    response_text += f"**Generated Function:**\n```python\n{feature_function}\n```\n\n"
                if recommended_steps:
                    response_text += f"**Recommended Steps:** {recommended_steps}\n\n"
                
                return response_text
            else:
                return "Feature engineering completed successfully."

        except Exception as e:
            logger.error(f"Error in feature engineering agent: {e}", exc_info=True)
            raise RuntimeError(f"Feature engineering failed: {str(e)}") from e

class FeatureEngineeringExecutor(AgentExecutor):
    """Feature Engineering Agent Executor."""

    def __init__(self):
        self.agent = FeatureEngineeringAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the feature engineering using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"🔧 Processing feature engineering query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a feature engineering request."
            
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
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Feature engineering failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the feature engineering server."""
    skill = AgentSkill(
        id="feature_engineering",
        name="Feature Engineering",
        description="Creates and transforms features for machine learning through advanced feature engineering techniques",
        tags=["features", "engineering", "preprocessing", "transformation"],
        examples=["create new features", "transform variables", "engineer features for ML"]
    )

    agent_card = AgentCard(
        name="Feature Engineering Agent",
        description="An AI agent that specializes in feature engineering and data transformation for machine learning.",
        url="http://localhost:8204/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=FeatureEngineeringExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🔧 Starting Feature Engineering Agent Server")
    print("🌐 Server starting on http://localhost:8204")
    print("📋 Agent card: http://localhost:8204/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8204, log_level="info")

if __name__ == "__main__":
    main() 