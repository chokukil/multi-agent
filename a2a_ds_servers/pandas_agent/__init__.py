"""
🐼 Pandas Agent Package

A2A SDK 기반 자연어 데이터 분석 에이전트
PandasAI 참고 구현을 통한 안전한 멀티 데이터프레임 처리

Author: CherryAI Team
License: MIT License
"""

from .pandas_agent_server import PandasAgentExecutor
from .multi_dataframe_handler import MultiDataFrameHandler
from .natural_language_processor import NaturalLanguageProcessor

__version__ = "1.0.0"
__author__ = "CherryAI Team"
__license__ = "MIT"

__all__ = [
    "PandasAgentExecutor",
    "MultiDataFrameHandler", 
    "NaturalLanguageProcessor"
] 