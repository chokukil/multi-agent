#!/usr/bin/env python3
"""
AI Data Science Orchestrator Server - A2A Compatible
Following official A2A SDK patterns with real LLM integration
"""

import logging
import uvicorn
import asyncio
import httpx
import os
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
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """AI Data Science Orchestrator Agent with LLM integration."""

    def __init__(self):
        # Try to initialize with real LLM if API key is available
        self.use_real_llm = False
        self.planner_node = None
        
        try:
            if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY'):
                from core.plan_execute.planner import planner_node
                from langchain_core.messages import HumanMessage
                
                self.planner_node = planner_node
                self.HumanMessage = HumanMessage
                self.use_real_llm = True
                logger.info("✅ Real LLM initialized for Orchestrator")
            else:
                logger.info("⚠️  No LLM API key found, using mock planning")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize LLM planner, falling back to mock: {e}")

    async def invoke(self, query: str) -> str:
        """Orchestrate data analysis across multiple agents."""
        try:
            if self.use_real_llm and self.planner_node:
                # Use real LLM for planning
                logger.info(f"🧠 Creating real plan for: {query[:100]}...")
                
                initial_state = {
                    "messages": [self.HumanMessage(content=query)],
                    "session_id": str(uuid4()),
                }
                
                plan_state = await asyncio.to_thread(self.planner_node, initial_state)
                plan = plan_state.get("plan")
                
                if not plan:
                    plan_summary = "🎯 **데이터 분석 실행 계획**\n\n**기본 계획**: Pandas Data Analyst를 사용한 분석\n\n"
                else:
                    plan_summary = "🎯 **데이터 분석 실행 계획**\n\n"
                    for i, step in enumerate(plan, 1):
                        agent_name = step.get("agent_name", "Unknown Agent")
                        reasoning = step.get("reasoning", "No reasoning provided")
                        plan_summary += f"**Step {i}**: {agent_name}\n- {reasoning}\n\n"
            else:
                # Use mock planning
                logger.info(f"🤖 Creating mock plan for: {query[:100]}...")
                plan_summary = "🎯 **데이터 분석 실행 계획** (Mock)\n\n"
                plan_summary += "**Step 1**: Pandas Data Analyst\n- Perform comprehensive exploratory data analysis\n- Identify patterns, correlations, and missing values\n\n"
                plan_summary += "**Step 2**: Statistical Analysis Agent\n- Conduct statistical tests and hypothesis testing\n- Generate descriptive and inferential statistics\n\n"
                plan_summary += "**Step 3**: Visualization Agent\n- Create informative charts and plots\n- Generate executive summary dashboards\n\n"
            
            result_summary = plan_summary
            
            # Execute Step 1: Try to call pandas agent if available
            agent_urls = {
                "pandas_data_analyst": "http://localhost:8200",
                "sql_data_analyst": "http://localhost:8201",
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
                    
                    logger.info(f"🔗 Calling Pandas Data Analyst...")
                    response = await client.send_message(request)
                    
                    # Extract response text (using the correct pattern we learned)
                    response_text = ""
                    actual_response = response.root if hasattr(response, 'root') else response
                    if hasattr(actual_response, 'result') and actual_response.result:
                        if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                            for part in actual_response.result.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += part.root.text
                    
                    if response_text:
                        result_summary += f"\n---\n\n**✅ Step 1 완료**: Pandas Data Analyst\n\n{response_text}\n"
                    else:
                        result_summary += f"\n---\n\n**❌ Step 1**: Pandas Data Analyst - No response received\n"
            
            except Exception as step_e:
                logger.error(f"Error calling pandas agent: {step_e}")
                result_summary += f"\n---\n\n**❌ Step 1**: Pandas Data Analyst - Error: {step_e}\n"
            
            # Mock execution for remaining steps (if using real LLM, could call actual agents)
            if not self.use_real_llm:
                result_summary += f"\n---\n\n**⏭️ Step 2**: Statistical Analysis Agent - Ready for implementation\n"
                result_summary += f"\n---\n\n**⏭️ Step 3**: Visualization Agent - Ready for implementation\n"
            
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
        # Extract user message using the official A2A pattern
        user_query = context.get_user_input()
        
        if not user_query:
            user_query = "Please provide a data analysis request."
        
        logger.info(f"🎯 Orchestrating query: {user_query}")
        
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
