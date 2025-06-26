# A2A Data Science Servers - Data Visualization Agent Server
# Advanced data visualization with interactive charts and dashboards
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
from ai_data_science_team.agents import DataVisualizationAgent

# Local utilities
from utils.plotly_streamlit import export_chart_data
from utils.logging import setup_a2a_logger, log_agent_execution

# Setup logging
logger = setup_a2a_logger("data_visualization_server", log_file="logs/data_viz.log")

class DataVisualizationAnalystAgent:
    """A2A wrapper for the Data Visualization Agent."""
    
    def __init__(self):
        self.name = "Data Visualization Analyst"
        self.llm = create_llm_instance()
        
        # Initialize visualization agent
        self.viz_agent = DataVisualizationAgent(
            model=self.llm,
            log=True,
            log_path="./artifacts/plots/",
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False
        )
        
        logger.info("Data Visualization Agent initialized successfully")

    async def create_visualization(self, user_instructions: str, data_raw: Optional[Dict] = None) -> Dict[str, Any]:
        """Create interactive data visualizations."""
        
        start_time = datetime.now()
        
        try:
            # Load sample data if none provided
            if data_raw is None:
                # Create sample visualization data
                import numpy as np
                sample_df = pd.DataFrame({
                    'x_values': np.random.normal(0, 1, 100),
                    'y_values': np.random.normal(0, 1, 100),
                    'categories': np.random.choice(['Category A', 'Category B', 'Category C'], 100),
                    'sizes': np.random.randint(10, 100, 100),
                    'colors': np.random.uniform(0, 1, 100)
                })
                data_raw = sample_df.to_dict()
            
            # Execute visualization creation
            logger.info("Starting data visualization creation")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.viz_agent.invoke_agent,
                user_instructions,
                data_raw,
                3,  # max_retries
                0   # retry_count
            )
            
            # Process visualization results
            results = self._process_visualization_results(response)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Visualization completed in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Visualization failed after {execution_time:.2f}s: {str(e)}")
            raise e

    def _process_visualization_results(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process visualization results and create artifacts."""
        
        results = {
            "status": "success",
            "visualization_summary": "Interactive visualization created successfully",
            "artifacts": [],
            "charts": [],
            "generated_code": {}
        }
        
        # Extract visualization if available
        if hasattr(self.viz_agent, 'get_plotly_graph'):
            try:
                plotly_fig = self.viz_agent.get_plotly_graph()
                if plotly_fig is not None:
                    # Export chart data
                    chart_artifact = export_chart_data(plotly_fig, "interactive_chart")
                    
                    results["charts"].append({
                        "type": "plotly_chart",
                        "data": chart_artifact["data"],
                        "metadata": chart_artifact["metadata"]
                    })
                    
                    # Save as artifact
                    chart_path = f"./artifacts/plots/{chart_artifact['filename']}"
                    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
                    with open(chart_path, 'w') as f:
                        json.dump(chart_artifact["data"], f, indent=2)
                    
                    results["artifacts"].append({
                        "type": "interactive_chart",
                        "filename": chart_artifact["filename"],
                        "path": chart_path,
                        "description": "Interactive Plotly visualization",
                        "metadata": chart_artifact["metadata"]
                    })
                    
            except Exception as e:
                logger.warning(f"Could not extract visualization: {e}")
        
        # Extract generated code
        if hasattr(self.viz_agent, 'get_data_visualization_function'):
            try:
                viz_code = self.viz_agent.get_data_visualization_function()
                if viz_code:
                    results["generated_code"]["visualization"] = viz_code
                    
                    # Save as artifact
                    code_path = f"./artifacts/python/visualizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                    os.makedirs(os.path.dirname(code_path), exist_ok=True)
                    with open(code_path, 'w') as f:
                        f.write(viz_code)
                    
                    results["artifacts"].append({
                        "type": "python_code",
                        "filename": os.path.basename(code_path),
                        "path": code_path,
                        "description": "Generated visualization code",
                        "metadata": {"language": "python", "lines": len(viz_code.split('\n'))}
                    })
                    
            except Exception as e:
                logger.warning(f"Could not extract generated code: {e}")
        
        return results

class DataVisualizationExecutor:
    """A2A Executor for Data Visualization Agent."""
    
    def __init__(self):
        self.agent = DataVisualizationAnalystAgent()
    
    async def execute(self, context):
        """Execute data visualization with A2A TaskUpdater pattern."""
        
        event_queue = context.get_event_queue()
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="🎨 Creating interactive data visualizations...")]
                )
            )
            
            # Execute visualization
            results = await self.agent.create_visualization(user_input)
            
            # Prepare response
            response_parts = []
            response_parts.append(
                task_updater.new_text_part(
                    text=f"## 🎨 Data Visualization Complete\n\n{results['visualization_summary']}"
                )
            )
            
            if results.get("charts"):
                charts_count = len(results["charts"])
                response_parts.append(
                    task_updater.new_text_part(
                        text=f"\n### 📊 Interactive Charts Created: {charts_count}"
                    )
                )
            
            artifacts_count = len(results.get("artifacts", []))
            response_parts.append(
                task_updater.new_text_part(
                    text=f"\n### 📁 Generated Artifacts: {artifacts_count} files"
                )
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=response_parts)
            )
            
            return results
            
        except Exception as e:
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text=f"❌ Visualization failed: {str(e)}")]
                )
            )
            raise e

# Create A2A Server
def create_data_visualization_app():
    """Create the A2A application for Data Visualization Agent."""
    
    agent_card = AgentCard(
        name="Data Visualization Analyst",
        description="Advanced interactive data visualization specialist creating stunning charts, plots, and dashboards with Plotly and other modern visualization libraries",
        instructions="Send your visualization request. I specialize in:\n"
                    "• Interactive Plotly charts and dashboards\n"
                    "• Statistical plots and distributions\n"
                    "• Multi-dimensional data visualization\n"
                    "• Custom color schemes and styling\n"
                    "• Responsive and mobile-friendly charts\n\n"
                    "Example: 'Create an interactive scatter plot showing the relationship between sales and profit by region'",
        skills=[
            AgentSkill(name="interactive_charts", description="Plotly interactive visualizations"),
            AgentSkill(name="statistical_plots", description="Advanced statistical plotting"),
            AgentSkill(name="dashboard_creation", description="Interactive dashboard development"),
            AgentSkill(name="custom_styling", description="Custom chart styling and theming"),
        ],
        streaming=True,
        version="1.0.0"
    )
    
    executor = DataVisualizationExecutor()
    request_handler = DefaultRequestHandler(executor)
    task_store = InMemoryTaskStore()
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        request_handler=request_handler,
        task_store=task_store
    )
    
    logger.info("Data Visualization A2A Server created")
    return app

# Create the app instance
app = create_data_visualization_app()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Data Visualization A2A Server starting...")
    uvicorn.run(app, host="localhost", port=8004, log_level="info") 