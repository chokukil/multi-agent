#!/usr/bin/env python3
"""
AI Data Science Orchestrator Server - A2A Compatible
Orchestrates multi-step data science workflows using specialized agents
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

class OrchestratorAgent:
    """AI Data Science Orchestrator Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.planner_node = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.plan_execute.planner import planner_node
            from langchain_core.messages import HumanMessage
            
            self.planner_node = planner_node
            self.HumanMessage = HumanMessage
            logger.info("✅ Real LLM initialized for Orchestrator")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM planner: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the orchestrator with a query."""
        try:
            logger.info(f"🎯 Planning multi-step analysis: {query[:100]}...")
            
            # Use real LLM for planning
            result = self.planner_node.invoke({"messages": [self.HumanMessage(content=query)]})
            
            if result and "plan" in result:
                return f"""🎯 **Data Science Analysis Plan Created**

**Query:** {query}

**Multi-Step Analysis Plan:**
{result['plan']}

**Status:** Plan created successfully. Ready for step-by-step execution."""
            else:
                return f"""🎯 **Analysis Plan**

**Query:** {query}

**Plan:** Multi-step data science workflow initiated.

**Next Steps:** Sequential agent execution will begin."""

        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            raise RuntimeError(f"Orchestration failed: {str(e)}") from e

class OrchestratorExecutor(AgentExecutor):
    """Orchestrator Agent Executor."""

    def __init__(self):
        self.agent = OrchestratorAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the orchestration using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            task_updater.submit()
            task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"📥 Processing orchestration query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a data science analysis request."
            
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
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Orchestration failed: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the orchestrator server."""
    skill = AgentSkill(
        id="data_science_orchestration",
        name="Data Science Orchestration",
        description="Orchestrates multi-step data science workflows by coordinating specialized agents",
        tags=["orchestration", "planning", "workflow", "multi-agent"],
        examples=["analyze my dataset comprehensively", "perform complete data science workflow", "coordinate agents for analysis"]
    )

    agent_card = AgentCard(
        name="AI Data Science Orchestrator",
        description="An AI orchestrator that plans and coordinates multi-step data science analyses using specialized agents.",
        url="http://localhost:8100/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("🎯 Starting AI Data Science Orchestrator Server")
    print("🌐 Server starting on http://localhost:8100")
    print("📋 Agent card: http://localhost:8100/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8100, log_level="info")

if __name__ == "__main__":
    main()
