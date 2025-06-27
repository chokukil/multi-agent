#!/usr/bin/env python3
"""
AI Data Science Orchestrator Server
A2A SDK v0.2.9 Compatible
"""

# A2A SDK imports (v0.2.9)
import uvicorn
from a2a.server.apps import A2AStarletteApplication  
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message


class OrchestratorAgent:
    """AI Data Science Team Orchestrator Agent."""
    
    async def invoke(self, user_message: str) -> str:
        """Process user request and orchestrate appropriate agents."""
        
        # Simple orchestration logic
        request_lower = user_message.lower()
        
        if any(word in request_lower for word in ["analyze", "analysis", "eda", "explore", "타이타닉", "titanic"]):
            return f"""🧠 Orchestrating comprehensive analysis workflow:

📋 Analysis Strategy:
1. EDA Tools Agent - Exploratory data analysis
2. Pandas Analyst Agent - Statistical analysis  
3. Data Visualization Agent - Charts and insights

🔄 Workflow Steps:
• Data profiling and quality assessment
• Statistical summaries and distributions  
• Pattern detection and correlation analysis
• Interactive visualizations

💡 Status: Multi-agent analysis workflow in progress
Request: {user_message}"""
        else:
            return f"""🎯 Orchestrating intelligent analysis workflow:

🤔 Analysis Strategy:
Based on your request, I'll coordinate multiple agents to provide comprehensive insights.

📋 Recommended Workflow:
1. Data assessment and validation
2. Exploratory data analysis (EDA)
3. Statistical analysis and patterns
4. Visualization and insights

💡 Status: Multi-agent orchestration in progress
Request: {user_message}"""


class OrchestratorAgentExecutor(AgentExecutor):
    """Orchestrator Agent Executor Implementation."""
    
    def __init__(self):
        self.agent = OrchestratorAgent()
        
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute orchestrator request."""
        # Extract user message from context
        user_message = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'text'):
                    user_message += part.text
        
        # Process the request
        result = await self.agent.invoke(user_message)
        
        # Send response
        await event_queue.enqueue_event(new_agent_text_message(result))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel orchestrator operation."""
        raise Exception('cancel not supported')


def main():
    """Main function to start the orchestrator server."""
    
    # Define orchestrator skills
    skill = AgentSkill(
        id="data_orchestration",
        name="Data Science Orchestration",
        description="Coordinates multiple AI data science agents for comprehensive analysis",
        tags=["orchestration", "coordination", "data-science", "multi-agent"],
        examples=["analyze my data", "create a comprehensive report", "orchestrate data analysis"]
    )
    
    # Create public agent card
    public_agent_card = AgentCard(
        name="AI Data Science Orchestrator",
        description="Central orchestrator for AI Data Science Team - coordinates multiple specialized agents for comprehensive data analysis",
        url="http://localhost:8100/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"], 
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=OrchestratorAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler
    )
    
    print("🚀 Starting AI Data Science Orchestrator Server")
    print("🌐 Server starting on http://localhost:8100")
    print("📋 Agent card: http://localhost:8100/.well-known/agent.json")
    
    # Run server
    uvicorn.run(
        server.build(),
        host="0.0.0.0", 
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main() 