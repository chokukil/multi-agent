#!/usr/bin/env python3
"""
AI Data Science Orchestrator Server - A2A Compatible
Following official A2A SDK patterns
"""

import logging
import uvicorn
import asyncio
import httpx
from uuid import uuid4

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message

# A2A Client imports
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams

# Mock implementation for testing A2A protocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """AI Data Science Orchestrator Agent."""

    async def invoke(self, query: str) -> str:
        """Orchestrate data analysis across multiple agents."""
        try:
            # Mock orchestration response for testing A2A protocol
            plan_summary = "🎯 **데이터 분석 실행 계획**\n\n"
            plan_summary += "**Step 1**: Pandas Data Analyst\n- Perform exploratory data analysis on the dataset\n\n"
            plan_summary += "**Step 2**: Data Visualization Agent\n- Create visualizations for key insights\n\n"
            plan_summary += "**Step 3**: Statistical Analysis Agent\n- Conduct statistical tests and analysis\n\n"
            
            result_summary = plan_summary
            
            # Try to call pandas agent if available
            agent_urls = {
                "pandas_data_analyst": "http://localhost:8200",
            }
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_urls["pandas_data_analyst"])
                    agent_card = await resolver.get_agent_card()
                    client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                    
                    # Use the original query for the pandas agent
                    send_message_payload = {
                        'message': {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': query}],
                            'messageId': uuid4().hex,
                        },
                    }
                    
                    request = SendMessageRequest(
                        id=str(uuid4()), 
                        params=MessageSendParams(**send_message_payload)
                    )
                    
                    response = await client.send_message(request)
                    
                    # Extract response text
                    response_text = ""
                    if hasattr(response, 'result') and response.result:
                        if hasattr(response.result, 'parts') and response.result.parts:
                            for part in response.result.parts:
                                if hasattr(part, 'text'):
                                    response_text += part.text
                    
                    if response_text:
                        result_summary += f"\n---\n\n**✅ Step 1 완료**: Pandas Data Analyst\n\n{response_text}\n"
                    else:
                        result_summary += f"\n---\n\n**❌ Step 1**: Pandas Data Analyst - No response received\n"
            
            except Exception as step_e:
                logger.error(f"Error calling pandas agent: {step_e}")
                result_summary += f"\n---\n\n**❌ Step 1**: Pandas Data Analyst - Error: {step_e}\n"
            
            result_summary += "\n\n🎉 **오케스트레이션 완료!**"
            return result_summary
            
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            return f"Orchestration failed: {str(e)}"

class OrchestratorExecutor(AgentExecutor):
    """AI Data Science Orchestrator Agent Executor."""

    def __init__(self):
        self.agent = OrchestratorAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the orchestration."""
        # Extract user message
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'text'):
                    user_query += part.text
        
        if not user_query:
            user_query = "Please provide a data analysis request."
        
        logger.info(f"Orchestrating query: {user_query}")
        
        # Get result from the agent
        result = await self.agent.invoke(user_query)
        
        # Send result back via event queue
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        logger.warning(f"Cancel called for context {context.context_id}")
        await event_queue.enqueue_event(new_agent_text_message("Orchestration cancelled."))

def main():
    """Main function to start the orchestrator server."""
    skill = AgentSkill(
        id="data_orchestration",
        name="Data Science Orchestration",
        description="Understands user requests, creates a data analysis plan, and coordinates multiple AI agents to execute the plan.",
        tags=["orchestration", "planning", "multi-agent"],
        examples=["analyze my data", "create a comprehensive report", "show me sales trends"]
    )

    agent_card = AgentCard(
        name="AI Data Science Orchestrator",
        description="The central coordinator for the AI Data Science Team. It creates analysis plans and manages specialized agents to fulfill user requests.",
        url="http://localhost:8100/",
        version="2.0.0",
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

    print("🚀 Starting AI Data Science Orchestrator Server")
    print("🌐 Server starting on http://localhost:8100")
    print("📋 Agent card: http://localhost:8100/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8100, log_level="info")

if __name__ == "__main__":
    main()
