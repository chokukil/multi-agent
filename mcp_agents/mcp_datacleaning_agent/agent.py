import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Configure the LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Define the Agent
root_agent = Agent(
    name="mcp_datacleaning_agent",
    description="An agent expert in data cleaning tasks like handling missing values.",
    instruction=(
        "You are an expert in data cleaning. "
        "Given a dataset, you can perform tasks like handling missing values, "
        "removing duplicates, and correcting data types."
    ),
    model=LiteLlm(
        model=OPENAI_MODEL,
        api_base=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY
    ),
    # tools=[...], # Add any specific data cleaning tools here
) 