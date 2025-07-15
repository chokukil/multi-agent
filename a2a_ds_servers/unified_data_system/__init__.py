"""
🍒 CherryAI 통합 데이터 로딩 시스템 (Unified Data Loading System)

pandas_agent 패턴을 기준으로 한 12개 A2A 에이전트 표준화 통합 시스템

핵심 원칙:
- LLM First 원칙 완전 준수
- A2A SDK 0.2.9 표준 완벽 적용  
- 100% 기능 유지, Mock 사용 금지
- pandas_agent FileConnector 패턴 기준

Author: CherryAI Team
License: MIT License
"""

from .core.unified_data_interface import UnifiedDataInterface, LoadingStrategy, A2AContext
from .core.llm_first_data_engine import LLMFirstDataEngine
from .core.smart_dataframe import SmartDataFrame, DataProfile, QualityReport
from .core.cache_manager import CacheManager
from .connectors.file_connector import EnhancedFileConnector

__all__ = [
    "UnifiedDataInterface",
    "LoadingStrategy",
    "A2AContext",
    "LLMFirstDataEngine", 
    "SmartDataFrame",
    "DataProfile",
    "QualityReport",
    "CacheManager",
    "EnhancedFileConnector"
]

__version__ = "1.0.0" 