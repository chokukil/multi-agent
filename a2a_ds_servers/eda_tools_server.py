# A2A Data Science Servers - EDA Tools Agent Server
# Exploratory Data Analysis with comprehensive statistical insights
# Compatible with A2A Protocol v0.2.9

import asyncio
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# A2A SDK imports
from a2a.server.request_handlers import DefaultRequestHandler  
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.core.data_structures import AgentCard, AgentSkill, TaskState

# Core imports
import sys
sys.path.append('..')
from core.llm_factory import create_llm_instance

# AI Data Science Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent

# Local utilities
from utils.logging import setup_a2a_logger, log_agent_execution

# Setup logging
logger = setup_a2a_logger("eda_tools_server", log_file="logs/eda_tools.log")

class EDAAnalystAgent:
    """A2A wrapper for the EDA Tools Agent."""
    
    def __init__(self):
        self.name = "EDA Tools Analyst"
        self.llm = create_llm_instance()
        
        # Initialize EDA agent
        self.eda_agent = EDAToolsAgent(
            model=self.llm,
            log=True,
            log_path="./artifacts/eda/",
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False
        )
        
        logger.info("EDA Tools Agent initialized successfully")

    async def perform_eda(self, user_instructions: str, data_raw: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform exploratory data analysis."""
        
        start_time = datetime.now()
        
        try:
            # Load default data if none provided
            if data_raw is None:
                default_data_path = "./artifacts/data/shared_dataframes/sample_data.csv"
                if os.path.exists(default_data_path):
                    df = pd.read_csv(default_data_path)
                    data_raw = df.to_dict()
                else:
                    # Create sample data
                    import numpy as np
                    sample_df = pd.DataFrame({
                        'feature_1': np.random.normal(100, 15, 1000),
                        'feature_2': np.random.exponential(2, 1000),
                        'category': np.random.choice(['A', 'B', 'C'], 1000),
                        'target': np.random.normal(50, 10, 1000)
                    })
                    data_raw = sample_df.to_dict()
            
            # Execute EDA
            logger.info("Starting EDA analysis")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.eda_agent.invoke_agent,
                user_instructions,
                data_raw,
                3,  # max_retries
                0   # retry_count
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"EDA completed in {execution_time:.2f}s")
            
            return {"status": "success", "response": response, "execution_time": execution_time}
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"EDA failed after {execution_time:.2f}s: {str(e)}")
            raise e

class EDAToolsExecutor:
    """A2A Executor for EDA Tools Agent."""
    
    def __init__(self):
        self.agent = EDAAnalystAgent()
    
    async def execute(self, context):
        """Execute EDA analysis with A2A TaskUpdater pattern."""
        
        event_queue = context.get_event_queue()
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="🔍 Performing exploratory data analysis...")]
                )
            )
            
            # Execute EDA
            results = await self.agent.perform_eda(user_input)
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="✅ EDA Analysis Complete with comprehensive insights")]
                )
            )
            
            return results
            
        except Exception as e:
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text=f"❌ EDA Analysis failed: {str(e)}")]
                )
            )
            raise e

# Create A2A Server
def create_eda_tools_app():
    """Create the A2A application for EDA Tools Agent."""
    
    agent_card = AgentCard(
        name="EDA Tools Analyst",
        description="Comprehensive exploratory data analysis with statistical insights and automated report generation",
        instructions="Send your data analysis request. I specialize in:\n"
                    "• Statistical summaries and distributions\n"
                    "• Feature correlation analysis\n" 
                    "• Data quality assessment\n"
                    "• Automated EDA reports\n"
                    "• Pattern discovery and outlier detection\n\n"
                    "Example: 'Analyze this dataset and provide comprehensive EDA insights'",
        skills=[
            AgentSkill(name="statistical_analysis", description="Advanced statistical analysis"),
            AgentSkill(name="data_profiling", description="Comprehensive data profiling"),
            AgentSkill(name="correlation_analysis", description="Feature correlation analysis"),
            AgentSkill(name="outlier_detection", description="Automated outlier detection"),
        ],
        streaming=True,
        version="1.0.0"
    )
    
    executor = EDAToolsExecutor()
    request_handler = DefaultRequestHandler(executor)
    task_store = InMemoryTaskStore()
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        request_handler=request_handler,
        task_store=task_store
    )
    
    logger.info("EDA Tools A2A Server created")
    return app

# Create the app instance
app = create_eda_tools_app()

if __name__ == "__main__":
    import uvicorn
    print("🚀 EDA Tools A2A Server starting...")
    uvicorn.run(app, host="localhost", port=8003, log_level="info") 