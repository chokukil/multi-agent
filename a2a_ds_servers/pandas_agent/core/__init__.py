"""
Core engine components for pandas_agent

This module contains the core functionality of the pandas_agent including
the main agent class, LLM integration, smart dataframe, and connectors.
"""

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from core.agent import PandasAgent
from core.llm import LLMEngine
from core.smart_dataframe import SmartDataFrame

__all__ = [
    "PandasAgent",
    "LLMEngine", 
    "SmartDataFrame",
] 