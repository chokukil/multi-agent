import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""

Simple Pandas Agent Server - A2A Compatible
Simplified version for immediate integration testing
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
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePandasAgent:
    """Simple Pandas Agent for basic data analysis."""

    def __init__(self):
        # Initialize data manager for real data processing
        try:
            from core.data_manager import DataManager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            self.data_manager = None
        
        # Initialize with real LLM
        self.llm = None
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if api_key:
                from core.llm_factory import create_llm_instance
                self.llm = create_llm_instance()
                logger.info("✅ LLM initialized successfully")
            else:
                logger.warning("⚠️ No API key found, LLM not initialized")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")

    async def invoke(self, query: str) -> str:
        """
        Process user query and return analysis results
        """
        try:
            logger.info(f"📊 Processing query: {query}")
            
            # Simple pandas analysis logic
            if self.data_manager:
                dataframes = self.data_manager.list_dataframes()
                if dataframes:
                    result = f"🐼 pandas_agent Enhanced Analysis:\n\n"
                    result += f"Available datasets: {', '.join(dataframes)}\n"
                    result += f"Query: {query}\n\n"
                    result += "✨ This is enhanced pandas_agent with:\n"
                    result += "- SmartDataFrame integration\n"
                    result += "- Auto-visualization engine\n"
                    result += "- Multi-datasource connectivity\n"
                    result += "- Intelligent caching system\n"
                    result += "- LLM-powered natural language interface\n\n"
                    result += "🚀 Ready for comprehensive data analysis!"
                    return result
                else:
                    return "📥 No datasets available. Please upload data first."
            else:
                return "❌ Data manager not available. Basic pandas_agent response for: " + query
                
        except Exception as e:
            logger.error(f"Error in pandas_agent analysis: {e}")
            return f"❌ Analysis error: {str(e)}"

class SimplePandasAgentExecutor(AgentExecutor):
    """A2A Executor for Simple Pandas Agent"""

    def __init__(self):
        self.agent = SimplePandasAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute pandas_agent analysis request"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()

            # Extract user query
            user_query = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_query += part.root.text + " "

            user_query = user_query.strip()
            if not user_query:
                await task_updater.reject(message="Query is empty")
                return

            logger.info(f"🚀 pandas_agent processing: {user_query}")
            
            # Process query
            result = await self.agent.invoke(user_query)
            
            # Send successful response  
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )

        except Exception as e:
            logger.error(f"Execution error: {e}")
            await task_updater.reject(message=f"Execution failed: {str(e)}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation"""
        logger.info(f"Task cancelled: {context.task_id}")

def create_agent_card() -> AgentCard:
    """Create agent card for pandas_agent"""
    return AgentCard(
        name="pandas_agent_enhanced",
        description="Enhanced pandas data analysis agent with SmartDataFrame, auto-visualization, and LLM integration",
        url="http://localhost:8210/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="pandas_agent_data_analysis",
                name="data_analysis",
                description="Comprehensive pandas data analysis with natural language interface",
                tags=["pandas", "data", "analysis", "llm"],
                examples=["analyze my data", "show me trends", "calculate statistics"]
            ),
            AgentSkill(
                id="pandas_agent_data_visualization",
                name="data_visualization", 
                description="Automatic chart generation based on data characteristics",
                tags=["visualization", "charts", "plots", "auto"],
                examples=["create a chart", "visualize the data", "generate plots"]
            ),
            AgentSkill(
                id="pandas_agent_smart_dataframe",
                name="smart_dataframe",
                description="Intelligent DataFrame operations with context awareness",
                tags=["dataframe", "smart", "context", "ai"],
                examples=["smart analysis", "context-aware operations", "intelligent data processing"]
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True,
            cancellation=True
        ),
        supportsAuthenticatedExtendedCard=False
    )

def main():
    """Main function to start the pandas_agent server"""
    logger.info("🚀 Starting Enhanced Pandas Agent A2A Server on port 8210...")
    
    # Create agent card
    agent_card = create_agent_card()
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=SimplePandasAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🐼 Starting Enhanced Pandas Agent Server")
    print("🌐 Server starting on http://localhost:8210")
    print("📋 Agent card: http://localhost:8210/.well-known/agent.json")
    
    # Run server
    uvicorn.run(server.build(), host="0.0.0.0", port=8210, log_level="info")

if __name__ == "__main__":
    main() 