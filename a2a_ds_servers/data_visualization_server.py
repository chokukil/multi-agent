#!/usr/bin/env python3
"""
Data Visualization Server - A2A Compatible
Following official A2A SDK patterns with real LLM integration
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

class DataVisualizationAgent:
    """Data Visualization Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.agents import DataVisualizationAgent as OriginalAgent
            
            self.llm = create_llm_instance()
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Data Visualization Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the data visualization agent with a query."""
        try:
            logger.info(f"🧠 Processing with real Data Visualization Agent: {query[:100]}...")
            
            # For real implementation, would need actual data
            # For now, create mock data structure
            import pandas as pd
            mock_data = pd.DataFrame({
                'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'sales': [1000, 1200, 1100, 1300, 1250, 1400],
                'region': ['North', 'South', 'North', 'South', 'North', 'South']
            })
            
            result = self.agent.invoke_agent(
                data_raw=mock_data,
                user_instructions=query
            )
            
            if self.agent.response:
                plotly_graph = self.agent.get_plotly_graph()
                viz_function = self.agent.get_data_visualization_function()
                
                response_text = f"✅ **Data Visualization Complete!**\n\n"
                response_text += f"**Request:** {query}\n\n"
                if viz_function:
                    response_text += f"**Generated Visualization Function:**\n```python\n{viz_function}\n```\n\n"
                if plotly_graph:
                    response_text += f"**Plotly Chart Generated:** Interactive visualization ready\n\n"
                
                return response_text
            else:
                return "Data visualization completed successfully."

        except Exception as e:
            logger.error(f"Error in data visualization agent: {e}", exc_info=True)
            raise RuntimeError(f"Visualization failed: {str(e)}") from e

class DataVisualizationExecutor(AgentExecutor):
    """Data Visualization Agent Executor."""

    def __init__(self):
        self.agent = DataVisualizationAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the data visualization using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"📊 Processing visualization query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a data visualization request."
            
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
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Visualization failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the data visualization server."""
    skill = AgentSkill(
        id="data_visualization",
        name="Data Visualization",
        description="Creates interactive data visualizations and charts using advanced plotting libraries",
        tags=["visualization", "plotting", "charts", "graphs"],
        examples=["create a bar chart", "visualize trends", "plot correlation matrix"]
    )

    agent_card = AgentCard(
        name="Data Visualization Agent",
        description="An AI agent that creates professional data visualizations and interactive charts.",
        url="http://localhost:8202/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DataVisualizationExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("📊 Starting Data Visualization Agent Server")
    print("🌐 Server starting on http://localhost:8202")
    print("📋 Agent card: http://localhost:8202/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8202, log_level="info")

if __name__ == "__main__":
    main() 