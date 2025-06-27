#!/usr/bin/env python3
"""
Pandas Data Analyst Server - A2A Compatible
Following official A2A SDK patterns
"""

import logging
import uvicorn

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

# Mock implementation for testing A2A protocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PandasDataAnalystAgent:
    """Pandas Data Analyst Agent."""

    def __init__(self):
        # Simplified to avoid LLM dependencies for testing
        pass

    async def invoke(self, query: str) -> str:
        """Invoke the pandas data analyst with a query."""
        try:
            # Mock response for testing A2A protocol
            return f"📊 **Pandas Data Analysis Result**\n\nQuery: {query}\n\n✅ Mock analysis completed successfully!\n\n🔍 **Sample Analysis:**\n- Data shape: (891, 12)\n- Missing values: 177 in Age column\n- Survival rate: 38.4%\n\n💡 This is a mock response to test A2A protocol functionality."
        except Exception as e:
            logger.error(f"Error in pandas analyst: {e}", exc_info=True)
            return f"Error occurred during analysis: {str(e)}"

class PandasDataAnalystExecutor(AgentExecutor):
    """Pandas Data Analyst Agent Executor."""

    def __init__(self):
        self.agent = PandasDataAnalystAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the pandas data analysis."""
        # Extract user message
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'text'):
                    user_query += part.text
        
        if not user_query:
            user_query = "Please provide a data analysis query."
        
        logger.info(f"Processing query: {user_query}")
        
        # Get result from the agent
        result = await self.agent.invoke(user_query)
        
        # Send result back via event queue
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        logger.warning(f"Cancel called for context {context.context_id}")
        await event_queue.enqueue_event(new_agent_text_message("Operation cancelled."))

def main():
    """Main function to start the pandas analyst server."""
    skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas Data Analysis",
        description="Performs data analysis using pandas library",
        tags=["pandas", "data-analysis", "statistics"],
        examples=["analyze my data", "show me sales trends", "calculate statistics"]
    )

    agent_card = AgentCard(
        name="Pandas Data Analyst",
        description="An AI agent that specializes in data analysis using the pandas library.",
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