#!/usr/bin/env python3
"""
SQL Data Analyst Server - A2A Compatible
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

class SQLDataAnalystAgent:
    """SQL Data Analyst Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.multiagents import SQLDataAnalyst
            from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent
            import sqlalchemy as sql
            
            self.llm = create_llm_instance()
            
            # Create a mock SQL connection for demonstration
            # In production, this should connect to actual database
            sql_engine = sql.create_engine("sqlite:///:memory:")
            conn = sql_engine.connect()
            
            # Initialize sub-agents
            sql_database_agent = SQLDatabaseAgent(
                model=self.llm,
                connection=conn,
                n_samples=10
            )
            data_visualization_agent = DataVisualizationAgent(model=self.llm)
            
            # Initialize the SQL data analyst with sub-agents
            self.agent = SQLDataAnalyst(
                model=self.llm,
                sql_database_agent=sql_database_agent,
                data_visualization_agent=data_visualization_agent
            )
            logger.info("✅ Real LLM initialized for SQL Data Analyst")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the SQL data analyst with a query."""
        try:
            logger.info(f"🧠 Processing with real SQL Data Analyst: {query[:100]}...")
            # Invoke the agent with proper parameters
            self.agent.invoke_agent(user_instructions=query)
            
            if self.agent.response:
                # Extract results from the response
                messages = self.agent.response.get("messages", [])
                if messages:
                    # Get the last message content
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    
                # Try to get specific outputs
                data_sql = self.agent.get_data_sql()
                plotly_graph = self.agent.get_plotly_graph()
                sql_query_code = self.agent.get_sql_query_code()
                sql_database_function = self.agent.get_sql_database_function()
                viz_function = self.agent.get_data_visualization_function()
                
                response_text = f"✅ **SQL Data Analysis Complete!**\n\n"
                response_text += f"**Query:** {query}\n\n"
                
                if sql_query_code:
                    response_text += f"**Generated SQL:**\n```sql\n{sql_query_code}\n```\n\n"
                if data_sql is not None:
                    response_text += f"**Query Results:** {len(data_sql)} rows returned\n\n"
                if sql_database_function:
                    response_text += f"**Database Function:**\n```python\n{sql_database_function}\n```\n\n"
                if plotly_graph:
                    response_text += f"**Visualization:** Interactive chart generated\n\n"
                if viz_function:
                    response_text += f"**Visualization Code:**\n```python\n{viz_function}\n```\n\n"
                    
                return response_text
            else:
                return "SQL analysis completed successfully."
        except Exception as e:
            logger.error(f"Error in SQL analyst: {e}", exc_info=True)
            raise RuntimeError(f"SQL analysis failed: {str(e)}") from e

class SQLDataAnalystExecutor(AgentExecutor):
    """SQL Data Analyst Agent Executor."""

    def __init__(self):
        self.agent = SQLDataAnalystAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the SQL data analysis using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            task_updater.submit()
            task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"📥 Processing SQL query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a SQL analysis request."
            
            # Get result from the agent
            result = await self.agent.invoke(user_query)
            
            # Complete task with result
            from a2a.types import TaskState, TextPart
            task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            # Report error through TaskUpdater
            from a2a.types import TaskState, TextPart
            task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"SQL analysis failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the SQL analyst server."""
    skill = AgentSkill(
        id="sql_data_analysis",
        name="SQL Data Analysis",
        description="Performs SQL-based data analysis and database operations",
        tags=["sql", "database", "analysis"],
        examples=["analyze database tables", "write SQL queries", "database insights"]
    )

    agent_card = AgentCard(
        name="SQL Data Analyst",
        description="An AI agent that specializes in SQL database analysis and query optimization.",
        url="http://localhost:8201/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SQLDataAnalystExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🗃️ Starting SQL Data Analyst Server")
    print("🌐 Server starting on http://localhost:8201")
    print("📋 Agent card: http://localhost:8201/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8201, log_level="info")

if __name__ == "__main__":
    main()
