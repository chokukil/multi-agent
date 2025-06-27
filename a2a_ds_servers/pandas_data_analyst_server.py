#!/usr/bin/env python3
"""
Pandas Data Analyst Server - A2A Compatible
Following official A2A SDK patterns with real LLM integration
"""

import logging
import uvicorn
import os

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PandasDataAnalystAgent:
    """Pandas Data Analyst Agent with LLM integration."""

    def __init__(self):
        # Try to initialize with real LLM if API key is available
        self.use_real_llm = False
        self.llm = None
        self.agent = None
        
        try:
            if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY'):
                from core.llm_factory import create_llm_instance
                from ai_data_science_team.multiagents import PandasDataAnalyst
                
                self.llm = create_llm_instance()
                self.agent = PandasDataAnalyst(llm=self.llm)
                self.use_real_llm = True
                logger.info("✅ Real LLM initialized for Pandas Data Analyst")
            else:
                logger.info("⚠️  No LLM API key found, using mock responses")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize LLM, falling back to mock: {e}")

    async def invoke(self, query: str) -> str:
        """Invoke the pandas data analyst with a query."""
        try:
            if self.use_real_llm and self.agent:
                # Use real LLM
                logger.info(f"🧠 Processing with real LLM: {query[:100]}...")
                result = self.agent.invoke({"question": query})
                if isinstance(result, dict) and "answer" in result:
                    return result["answer"]
                elif isinstance(result, str):
                    return result
                else:
                    return "Analysis completed successfully."
            else:
                # Use mock response
                logger.info(f"🤖 Processing with mock: {query[:100]}...")
                return f"""📊 **Pandas Data Analysis Result**

**Query:** {query}

✅ **Analysis Completed Successfully!**

🔍 **Sample Analysis Results:**
- Dataset loaded and processed
- Shape: (891, 12) - 891 rows, 12 columns  
- Missing values detected in Age (177), Cabin (687), Embarked (2)
- Survival rate: 38.4% (342/891 passengers survived)

📈 **Key Insights:**
- Higher survival rates for females (74.2%) vs males (18.9%)
- First class passengers had better survival chances (62.9%)
- Age distribution shows most passengers were 20-40 years old

💡 **Recommendations:**
- Focus analysis on gender and passenger class correlations
- Investigate age group survival patterns
- Consider family size impact on survival

*Note: This is enhanced mock data for demonstration. Enable LLM integration for real analysis.*"""

        except Exception as e:
            logger.error(f"Error in pandas analyst: {e}", exc_info=True)
            return f"Error occurred during analysis: {str(e)}"

class PandasDataAnalystExecutor(AgentExecutor):
    """Pandas Data Analyst Agent Executor."""

    def __init__(self):
        self.agent = PandasDataAnalystAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the pandas data analysis."""
        # Debug: Check different ways to extract user input
        logger.info(f"🔍 Debug context: {context}")
        logger.info(f"🔍 Debug context.message: {context.message}")
        if context.message:
            logger.info(f"🔍 Debug context.message.parts: {context.message.parts}")
        
        # Extract user message using the official A2A pattern
        user_query = context.get_user_input()
        logger.info(f"🔍 get_user_input() returned: '{user_query}'")
        
        # Alternative extraction methods if get_user_input fails
        if not user_query and context.message and context.message.parts:
            logger.info("🔍 Trying alternative extraction...")
            for i, part in enumerate(context.message.parts):
                logger.info(f"🔍 Part {i}: {part}")
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text
                    logger.info(f"🔍 Found text via part.root.text: {part.root.text}")
                elif hasattr(part, 'text'):
                    user_query += part.text
                    logger.info(f"🔍 Found text via part.text: {part.text}")
        
        if not user_query:
            user_query = "Please provide a data analysis query."
        
        logger.info(f"📥 Final query to process: {user_query}")
        
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