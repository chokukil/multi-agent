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
    "data_loader": {
        "host": AGENT_HOST,
        "port": 8000,
        "url": f"http://{AGENT_HOST}:8000",
        "name": "Data Loader Agent",
        "description": "Data loading and processing with file operations"
    },
    "pandas_analyst": {
        "host": AGENT_HOST,
        "port": 8001,
        "url": f"http://{AGENT_HOST}:8001",
        "name": "Pandas Data Analyst",
        "description": "Advanced pandas data analysis with interactive visualizations"
    },
    "sql_analyst": {
        "host": AGENT_HOST,
        "port": 8002,
        "url": f"http://{AGENT_HOST}:8002",
        "name": "SQL Data Analyst",
        "description": "SQL database analysis with query generation"
    },
    "eda_tools": {
        "host": AGENT_HOST,
        "port": 8003,
        "url": f"http://{AGENT_HOST}:8003",
        "name": "EDA Tools Analyst",
        "description": "Comprehensive exploratory data analysis and statistical insights"
    },
    "data_visualization": {
        "host": AGENT_HOST,
        "port": 8004,
        "url": f"http://{AGENT_HOST}:8004",
        "name": "Data Visualization Analyst",
        "description": "Interactive chart and dashboard creation with Plotly"
    },
    "orchestrator": {
        "host": AGENT_HOST,
        "port": 8100,
        "url": f"http://{AGENT_HOST}:8100",
        "name": "Data Science Orchestrator",
        "description": "Central management and orchestration of all data science agents"
    }
}

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Logging Configuration
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True) 