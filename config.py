# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# A2A Registry Server Configuration
REGISTRY_HOST = "127.0.0.1"
REGISTRY_PORT = 8000
REGISTRY_URL = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

# Agent Server Configurations
AGENT_HOST = "127.0.0.1"
AGENT_SERVERS = {
    "pandas_analyst": {
        "host": AGENT_HOST,
        "port": 8001,
        "url": f"http://{AGENT_HOST}:8001"
    },
    "sql_analyst": {
        "host": AGENT_HOST,
        "port": 8002,
        "url": f"http://{AGENT_HOST}:8002"
    },
    # Add other agents here
}

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Logging Configuration
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True) 