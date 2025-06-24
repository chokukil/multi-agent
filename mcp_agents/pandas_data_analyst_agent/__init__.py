import sys
import os
from google.adk.models.lite_llm import LiteLlm

# To resolve the ModuleNotFoundError, we explicitly add the project's root
# directory to Python's path. This ensures that absolute imports can be found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the refactored agents and the orchestrator class
from mcp_agents.mcp_datawrangling_agent.agent import DataWranglingAgent
from mcp_agents.mcp_datavisualization_agent.agent import DataVisualizationAgent
from .agent import PandasDataAnalyst

# Configure the LLM using environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

model_config = LiteLlm(
    model=OPENAI_MODEL,
    api_base=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY
)

# Instantiate the specialist agents
data_wrangling_agent = DataWranglingAgent(model=model_config)
data_visualization_agent = DataVisualizationAgent(model=model_config)

# Instantiate the orchestrator with the specialist agents
root_agent = PandasDataAnalyst(
    model=model_config,
    data_wrangling_agent=data_wrangling_agent,
    data_visualization_agent=data_visualization_agent,
) 