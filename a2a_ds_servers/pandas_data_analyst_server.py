#!/usr/bin/env python3
"""
Pandas Data Analyst Server - A2A Compatible
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

class PandasDataAnalystAgent:
    """Pandas Data Analyst Agent with LLM integration."""

    def __init__(self):
        # Initialize data manager for real data processing
        try:
            from core.data_manager import DataManager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            raise RuntimeError("Data Manager is required for operation") from e
        
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.multiagents import PandasDataAnalyst
            from ai_data_science_team.agents import DataWranglingAgent, DataVisualizationAgent
            
            self.llm = create_llm_instance()
            
            # Initialize sub-agents
            data_wrangling_agent = DataWranglingAgent(model=self.llm)
            data_visualization_agent = DataVisualizationAgent(model=self.llm)
            
            # Initialize the pandas data analyst with sub-agents
            self.agent = PandasDataAnalyst(
                model=self.llm,
                data_wrangling_agent=data_wrangling_agent,
                data_visualization_agent=data_visualization_agent
            )
            logger.info("✅ Real LLM initialized for Pandas Data Analyst")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the pandas data analyst with a query."""
        try:
            logger.info(f"🧠 Processing with real LLM: {query[:100]}...")
            
            # Get actual data for analysis
            data_raw = None
            if self.data_manager:
                dataframe_ids = self.data_manager.list_dataframes()
                if dataframe_ids:
                    # Use the first available dataframe
                    data_raw = self.data_manager.get_dataframe(dataframe_ids[0])
                    logger.info(f"📊 Using dataframe '{dataframe_ids[0]}' with shape: {data_raw.shape}")
                else:
                    logger.info("📊 No uploaded data found, using sample data")
                    
            # If no data available, create sample data for demonstration
            if data_raw is None:
                import pandas as pd
                data_raw = pd.DataFrame({
                    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
                    'value': [10, 20, 15, 30, 25, 12],
                    'date': pd.date_range('2024-01-01', periods=6)
                })
                logger.info("📊 Using sample data for demonstration")
            
            # Invoke the agent with proper parameters
            self.agent.invoke_agent(
                user_instructions=query,
                data_raw=data_raw
            )
            
            if self.agent.response:
                # Extract results from the response
                messages = self.agent.response.get("messages", [])
                if messages:
                    # Get the last message content
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    
                # Try to get specific outputs
                data_wrangled = self.agent.get_data_wrangled()
                plotly_graph = self.agent.get_plotly_graph()
                wrangler_function = self.agent.get_data_wrangler_function()
                viz_function = self.agent.get_data_visualization_function()
                
                response_text = f"✅ **Pandas Data Analysis Complete!**\n\n"
                response_text += f"**Query:** {query}\n\n"
                
                if data_wrangled is not None:
                    response_text += f"**Data Shape:** {data_wrangled.shape}\n\n"
                if wrangler_function:
                    response_text += f"**Data Processing:**\n```python\n{wrangler_function}\n```\n\n"
                if plotly_graph:
                    response_text += f"**Visualization:** Interactive chart generated\n\n"
                if viz_function:
                    response_text += f"**Visualization Code:**\n```python\n{viz_function}\n```\n\n"
                    
                return response_text
            else:
                return "Analysis completed successfully."

        except Exception as e:
            logger.error(f"Error in pandas analyst: {e}", exc_info=True)
            raise RuntimeError(f"Analysis failed: {str(e)}") from e

class PandasDataAnalystExecutor(AgentExecutor):
    """Pandas Data Analyst Agent Executor."""

    def __init__(self):
        self.agent = PandasDataAnalystAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the pandas data analysis using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"📥 Processing query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a data analysis query."
            
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
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Analysis failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the pandas analyst server."""
    skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas Data Analysis",
        description="Performs comprehensive data analysis using pandas library on uploaded datasets",
        tags=["pandas", "data-analysis", "statistics", "eda"],
        examples=["analyze my data", "show me sales trends", "calculate statistics", "perform EDA on uploaded dataset"]
    )

    agent_card = AgentCard(
        name="Pandas Data Analyst",
        description="An AI agent that specializes in data analysis using the pandas library with real uploaded data.",
        url="http://localhost:8200/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=PandasDataAnalystExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🐼 Starting Pandas Data Analyst Server")
    print("🌐 Server starting on http://localhost:8200")
    print("📋 Agent card: http://localhost:8200/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8200, log_level="info")

if __name__ == "__main__":
    main() 