import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Configure the LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Define the Agent
root_agent = Agent(
    name="mcp_h2o_modeling_agent",
    description="An agent for training models with H2O AutoML.",
    instruction=(
        "You are an expert in automated machine learning with H2O."
    ),
    model=LiteLlm(
        model=OPENAI_MODEL,
        api_base=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY
    ),
) 