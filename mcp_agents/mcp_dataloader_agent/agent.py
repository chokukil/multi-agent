from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
import pandas as pd
import os

def load_data(file_path: str, data_identifier: str, tool_context: ToolContext) -> str:
    """
    Loads data from a specified file into a DataFrame.

    :param file_path: The path to the file to load.
    :param data_identifier: The identifier to assign to the loaded data.
    :return: A message indicating the result of the operation.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'")

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")
        
        # Convert DataFrame to a JSON string to ensure serializability
        df_json = df.to_json(orient='split')
        
        # Store the JSON string in the agent's data context
        tool_context.state[data_identifier] = df_json
        
        return f"Successfully loaded data from '{file_path}' into '{data_identifier}'."

    except Exception as e:
        return f"Error loading data: {e}"

root_agent = Agent(
    name="mcp_dataloader_agent",
    description="An agent that loads data from files into the system.",
    tools=[FunctionTool(load_data)]
)

# The agent is now defined. It can be run using the ADK CLI.
# For example, from the project root:
#
# adk web
#
# Then select 'mcp_dataloader_agent' in the web UI.
#
# Or to run from the command line:
#
# adk run mcp_agents.mcp_dataloader_agent
#
# Note: For the CLI to find the agent, the parent directory of 'mcp_agents'
# should be in the Python path. 