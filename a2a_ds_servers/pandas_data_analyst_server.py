# A2A Data Science Servers - Pandas Data Analyst Server
# Full-featured Pandas Data Analyst with data wrangling and visualization capabilities
# Compatible with A2A Protocol v0.2.9

import asyncio
import traceback
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# A2A SDK imports
from a2a.server.request_handlers import DefaultRequestHandler  
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.core.data_structures import AgentCard, AgentSkill, TaskState

# Core imports for LLM integration
import sys
sys.path.append('..')
from core.llm_factory import create_llm_instance

# AI Data Science Team imports - we'll wrap these instead of modifying
from ai_data_science_team.multiagents import PandasDataAnalyst
from ai_data_science_team.agents import DataWranglingAgent, DataVisualizationAgent

# Local utilities
from utils.plotly_streamlit import plotly_from_dict, export_chart_data
from utils.messages import format_agent_response, create_status_message, format_artifacts
from utils.logging import setup_a2a_logger, log_agent_execution

# Setup logging
logger = setup_a2a_logger("pandas_data_analyst_server", log_file="logs/pandas_analyst.log")

class PandasDataAnalystAgent:
    """
    A2A wrapper for the Pandas Data Analyst multi-agent.
    Combines data wrangling and visualization capabilities.
    """
    
    def __init__(self):
        self.name = "Pandas Data Analyst"
        self.description = "Advanced data analysis with pandas wrangling and interactive visualizations"
        
        # Initialize LLM
        self.llm = create_llm_instance()
        
        # Initialize the underlying agents
        self.data_wrangling_agent = DataWranglingAgent(
            model=self.llm,
            log=True,
            log_path="./artifacts/python/",
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False
        )
        
        self.data_visualization_agent = DataVisualizationAgent(
            model=self.llm,
            log=True,
            log_path="./artifacts/plots/",
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False
        )
        
        # Initialize the multi-agent
        self.pandas_analyst = PandasDataAnalyst(
            model=self.llm,
            data_wrangling_agent=self.data_wrangling_agent,
            data_visualization_agent=self.data_visualization_agent
        )
        
        logger.info("Pandas Data Analyst initialized successfully")

    async def analyze_data(
        self, 
        user_instructions: str, 
        data_raw: Union[Dict, List, None] = None,
        data_file_path: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis with wrangling and visualization.
        
        Parameters:
        ----------
        user_instructions : str
            User's analysis instructions.
        data_raw : Union[Dict, List, None], optional
            Raw data as dictionary or list.
        data_file_path : str, optional
            Path to data file to load.
        max_retries : int, optional
            Maximum retry attempts.
            
        Returns:
        -------
        Dict[str, Any]
            Analysis results with artifacts.
        """
        
        start_time = datetime.now()
        
        try:
            # Load data if file path provided
            if data_file_path and os.path.exists(data_file_path):
                logger.info(f"Loading data from {data_file_path}")
                if data_file_path.endswith('.csv'):
                    df = pd.read_csv(data_file_path)
                    data_raw = df.to_dict()
                elif data_file_path.endswith('.json'):
                    with open(data_file_path, 'r') as f:
                        data_raw = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {data_file_path}")
            
            # Validate we have data
            if data_raw is None:
                # Try to load default data
                default_data_path = "./artifacts/data/shared_dataframes/titanic.csv.pkl"
                if os.path.exists(default_data_path):
                    logger.info(f"Loading default data from {default_data_path}")
                    df = pd.read_pickle(default_data_path)
                    data_raw = df.to_dict()
                else:
                    raise ValueError("No data provided and no default data available")
            
            # Execute the analysis
            logger.info("Starting Pandas Data Analyst execution")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.pandas_analyst.invoke_agent,
                user_instructions,
                data_raw,
                max_retries,
                0  # retry_count
            )
            
            # Process results
            results = self._process_analysis_results(response)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis failed after {execution_time:.2f}s: {str(e)}")
            raise e

    def _process_analysis_results(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and format analysis results for A2A protocol.
        
        Parameters:
        ----------
        response : Dict[str, Any]
            Raw response from pandas analyst.
            
        Returns:
        -------
        Dict[str, Any]
            Processed results with artifacts.
        """
        
        results = {
            "status": "success",
            "analysis_summary": "Data analysis completed successfully",
            "artifacts": [],
            "data_summary": {},
            "visualizations": [],
            "generated_code": {},
        }
        
        # Extract wrangled data
        if response and hasattr(self.pandas_analyst, 'get_data_wrangled'):
            try:
                wrangled_df = self.pandas_analyst.get_data_wrangled()
                if wrangled_df is not None:
                    results["data_summary"] = {
                        "shape": wrangled_df.shape,
                        "columns": wrangled_df.columns.tolist(),
                        "dtypes": wrangled_df.dtypes.to_dict(),
                        "sample_data": wrangled_df.head().to_dict('records')
                    }
                    
                    # Save as artifact
                    artifact_path = f"./artifacts/data/wrangled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    wrangled_df.to_csv(artifact_path, index=False)
                    
                    results["artifacts"].append({
                        "type": "processed_data",
                        "filename": os.path.basename(artifact_path),
                        "path": artifact_path,
                        "description": "Processed and wrangled data",
                        "metadata": {"rows": len(wrangled_df), "columns": len(wrangled_df.columns)}
                    })
                    
            except Exception as e:
                logger.warning(f"Could not extract wrangled data: {e}")
        
        # Extract visualization
        if hasattr(self.pandas_analyst, 'get_plotly_graph'):
            try:
                plotly_fig = self.pandas_analyst.get_plotly_graph()
                if plotly_fig is not None:
                    # Export chart data
                    chart_artifact = export_chart_data(plotly_fig, "analysis_chart")
                    
                    results["visualizations"].append({
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
                        "type": "visualization",
                        "filename": chart_artifact["filename"],
                        "path": chart_path,
                        "description": "Interactive data visualization",
                        "metadata": chart_artifact["metadata"]
                    })
                    
            except Exception as e:
                logger.warning(f"Could not extract visualization: {e}")
        
        # Extract generated code
        try:
            if hasattr(self.pandas_analyst, 'get_data_wrangler_function'):
                wrangler_code = self.pandas_analyst.get_data_wrangler_function()
                if wrangler_code:
                    results["generated_code"]["data_wrangling"] = wrangler_code
                    
                    # Save as artifact
                    code_path = f"./artifacts/python/data_wrangler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                    os.makedirs(os.path.dirname(code_path), exist_ok=True)
                    with open(code_path, 'w') as f:
                        f.write(wrangler_code)
                    
                    results["artifacts"].append({
                        "type": "python_code",
                        "filename": os.path.basename(code_path),
                        "path": code_path,
                        "description": "Generated data wrangling code",
                        "metadata": {"language": "python", "lines": len(wrangler_code.split('\n'))}
                    })
            
            if hasattr(self.pandas_analyst, 'get_data_visualization_function'):
                viz_code = self.pandas_analyst.get_data_visualization_function()
                if viz_code:
                    results["generated_code"]["visualization"] = viz_code
                    
                    # Save as artifact
                    code_path = f"./artifacts/python/data_visualizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
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

class PandasDataAnalystExecutor:
    """A2A Executor for Pandas Data Analyst."""
    
    def __init__(self):
        self.agent = PandasDataAnalystAgent()
    
    async def execute(self, context):
        """Execute pandas data analysis with A2A TaskUpdater pattern."""
        
        # Get EventQueue for TaskUpdater
        event_queue = context.get_event_queue()
        
        # Initialize TaskUpdater  
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit task
            await task_updater.submit()
            
            # Start work
            await task_updater.start_work()
            
            # Get user input
            user_input = context.get_user_input()
            
            # Update status - analyzing request
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="🔄 Analyzing your data analysis request...")]
                )
            )
            
            # Parse user request for data file path
            data_file_path = None
            if "load" in user_input.lower() or "file" in user_input.lower():
                # Try to extract file path (simple heuristic)
                import re
                file_patterns = [
                    r'load\s+(\S+\.csv)',
                    r'file\s+(\S+\.csv)',
                    r'data\s+(\S+\.csv)',
                    r'(\S+\.csv)',
                    r'(\S+\.pkl)',
                ]
                for pattern in file_patterns:
                    match = re.search(pattern, user_input, re.IGNORECASE)
                    if match:
                        potential_path = match.group(1)
                        if os.path.exists(potential_path) or os.path.exists(f"./artifacts/data/shared_dataframes/{potential_path}"):
                            data_file_path = potential_path if os.path.exists(potential_path) else f"./artifacts/data/shared_dataframes/{potential_path}"
                            break
            
            # Update status - processing
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text="⚙️ Processing data with advanced analytics...")]
                )
            )
            
            # Execute analysis
            results = await self.agent.analyze_data(
                user_instructions=user_input,
                data_file_path=data_file_path
            )
            
            # Prepare final response
            response_parts = []
            
            # Add summary
            response_parts.append(
                task_updater.new_text_part(
                    text=f"## 📊 Pandas Data Analysis Complete\n\n{results['analysis_summary']}"
                )
            )
            
            # Add data summary if available
            if results.get("data_summary"):
                summary = results["data_summary"]
                response_parts.append(
                    task_updater.new_text_part(
                        text=f"\n### 📈 Data Summary:\n"
                             f"- **Shape**: {summary.get('shape', 'N/A')}\n"
                             f"- **Columns**: {len(summary.get('columns', []))}\n"
                             f"- **Sample Data**: {len(summary.get('sample_data', []))} rows shown"
                    )
                )
            
            # Add visualization info
            if results.get("visualizations"):
                viz_count = len(results["visualizations"])
                response_parts.append(
                    task_updater.new_text_part(
                        text=f"\n### 📊 Visualizations Created: {viz_count}"
                    )
                )
            
            # Add generated code info
            if results.get("generated_code"):
                code_types = list(results["generated_code"].keys())
                response_parts.append(
                    task_updater.new_text_part(
                        text=f"\n### 🔧 Generated Code: {', '.join(code_types)}"
                    )
                )
            
            # Add artifacts summary
            artifacts_count = len(results.get("artifacts", []))
            response_parts.append(
                task_updater.new_text_part(
                    text=f"\n### 📁 Generated Artifacts: {artifacts_count} files"
                )
            )
            
            # Complete task
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=response_parts)
            )
            
            # Log successful execution
            log_agent_execution(
                logger, "PandasDataAnalyst", "data_analysis", "completed",
                details={"artifacts_count": artifacts_count}
            )
            
            return results
            
        except Exception as e:
            error_message = f"Analysis failed: {str(e)}"
            logger.error(f"Execution failed: {error_message}", exc_info=True)
            
            # Update task as failed
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[task_updater.new_text_part(text=f"❌ {error_message}")]
                )
            )
            
            # Log failed execution
            log_agent_execution(
                logger, "PandasDataAnalyst", "data_analysis", "failed",
                details={"error": error_message}
            )
            
            raise e
    
    async def cancel(self, context):
        """Cancel the pandas data analysis task."""
        logger.info("Pandas Data Analyst task cancelled")
        return {"status": "cancelled"}

# A2A Server Setup
def create_pandas_analyst_app():
    """Create the A2A Starlett application for Pandas Data Analyst."""
    
    # Define agent card
    agent_card = AgentCard(
        name="Pandas Data Analyst",
        description="Advanced multi-agent data analysis system combining pandas data wrangling with interactive visualizations. Capable of processing datasets, performing complex transformations, and generating beautiful charts.",
        instructions="Send your data analysis request. I can:\n"
                    "• Load and process CSV/JSON data files\n" 
                    "• Perform data cleaning and wrangling\n"
                    "• Generate interactive visualizations\n"
                    "• Create reusable Python code\n"
                    "• Handle complex multi-step analysis workflows\n\n"
                    "Example: 'Analyze the sales data and create a visualization showing trends by region'",
        skills=[
            AgentSkill(
                name="data_wrangling",
                description="Advanced pandas data manipulation and cleaning"
            ),
            AgentSkill(
                name="data_visualization", 
                description="Interactive plotly chart generation"
            ),
            AgentSkill(
                name="code_generation",
                description="Reusable Python data analysis code"
            ),
            AgentSkill(
                name="multi_step_analysis",
                description="Complex analytical workflows"
            )
        ],
        streaming=True,
        version="1.0.0"
    )
    
    # Initialize request handler
    executor = PandasDataAnalystExecutor()
    request_handler = DefaultRequestHandler(executor)
    
    # Create task store
    task_store = InMemoryTaskStore()
    
    # Create A2A application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        request_handler=request_handler,
        task_store=task_store
    )
    
    logger.info("Pandas Data Analyst A2A Server created successfully")
    return app

# Server startup
if __name__ == "__main__":
    import uvicorn
    
    # Create the app
    app = create_pandas_analyst_app()
    
    # Add startup message
    @app.on_event("startup")
    async def startup_event():
        logger.info("🚀 Pandas Data Analyst A2A Server starting...")
        logger.info("📊 Ready to perform advanced data analysis!")
        
        # Create required directories
        os.makedirs("./artifacts/data", exist_ok=True)
        os.makedirs("./artifacts/plots", exist_ok=True)
        os.makedirs("./artifacts/python", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
    
    # Run the server
    uvicorn.run(app, host="localhost", port=8001, log_level="info") 