"""
통합 데이터 인터페이스 (Unified Data Interface)

pandas_agent 패턴을 기준으로 모든 A2A 에이전트가 구현해야 할 표준 데이터 인터페이스

핵심 원칙:
- LLM First: 모든 데이터 관련 결정을 LLM이 담당
- A2A 표준: SDK 0.2.9 완벽 준수
- 기능 보존: 100% 기능 유지, Mock 사용 금지
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime


class DataIntentType(Enum):
    """데이터 처리 의도 타입"""
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization" 
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    MODELING = "modeling"
    FEATURE_ENGINEERING = "feature_engineering"
    SQL_QUERY = "sql_query"
    REPORTING = "reporting"
    EDA = "eda"
    ORCHESTRATION = "orchestration"


@dataclass
class DataIntent:
    """LLM이 분석한 데이터 처리 의도"""
    intent_type: DataIntentType
    confidence: float
    file_preferences: List[str]
    operations: List[str]
    constraints: Dict[str, Any]
    priority: int = 1
    requires_visualization: bool = False
    estimated_complexity: str = "medium"  # low, medium, high


@dataclass
class DataProfile:
    """데이터 프로파일 정보"""
    shape: tuple
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    memory_usage: int
    encoding: str
    file_size: int
    sample_data: Optional[Dict[str, Any]] = None
    column_stats: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[float] = None


@dataclass
class QualityReport:
    """데이터 품질 리포트"""
    overall_score: float
    completeness: float
    consistency: float 
    validity: float
    accuracy: float
    uniqueness: float
    issues: List[str]
    recommendations: List[str]
    passed_checks: List[str]
    failed_checks: List[str]


@dataclass
class LoadingStrategy:
    """데이터 로딩 전략"""
    encoding: str = "utf-8"
    chunk_size: Optional[int] = None
    sample_ratio: Optional[float] = None
    use_cache: bool = True
    cache_ttl: int = 3600
    fallback_encodings: List[str] = None
    preprocessing_steps: List[str] = None
    
    def __post_init__(self):
        if self.fallback_encodings is None:
            self.fallback_encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []


class A2AContext:
    """A2A 컨텍스트 래퍼"""
    
    def __init__(self, context: Any):
        self.original_context = context
        self.session_id = self._extract_session_id(context)
        self.user_id = self._extract_user_id(context)
        self.request_id = self._extract_request_id(context)
        
    def _extract_session_id(self, context: Any) -> Optional[str]:
        """세션 ID 추출"""
        try:
            if hasattr(context, 'request') and context.request:
                return context.request.get("session_id")
            return None
        except:
            return None
    
    def _extract_user_id(self, context: Any) -> Optional[str]:
        """사용자 ID 추출"""
        try:
            if hasattr(context, 'request') and context.request:
                return context.request.get("user_id")
            return None
        except:
            return None
    
    def _extract_request_id(self, context: Any) -> Optional[str]:
        """요청 ID 추출"""
        try:
            if hasattr(context, 'request') and context.request:
                return context.request.get("request_id")
            return str(datetime.now().timestamp())
        except:
            return str(datetime.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat()
        }


class UnifiedDataInterface(ABC):
    """
    모든 A2A 에이전트가 구현해야 할 표준 데이터 인터페이스
    
    pandas_agent 패턴을 기준으로 설계된 통합 인터페이스
    LLM First 원칙과 A2A SDK 0.2.9 표준을 완벽히 준수
    """
    
    @abstractmethod
    async def load_data(self, intent: DataIntent, context: A2AContext) -> 'SmartDataFrame':
        """
        데이터 로딩 (필수 구현)
        
        Args:
            intent: LLM이 분석한 데이터 처리 의도
            context: A2A 컨텍스트 정보
            
        Returns:
            SmartDataFrame: 지능형 DataFrame 객체
        """
        pass
    
    @abstractmethod
    async def get_data_info(self) -> DataProfile:
        """
        데이터 정보 조회 (필수 구현)
        
        Returns:
            DataProfile: 데이터 프로파일 정보
        """
        pass
    
    @abstractmethod
    async def validate_data_quality(self) -> QualityReport:
        """
        데이터 품질 검증 (필수 구현)
        
        Returns:
            QualityReport: 데이터 품질 리포트
        """
        pass
    
    # 선택적 구현 메서드들
    async def transform_data(self, operations: List[Dict[str, Any]]) -> 'SmartDataFrame':
        """
        데이터 변환 (선택적 구현)
        
        Args:
            operations: LLM이 생성한 변환 작업 리스트
            
        Returns:
            SmartDataFrame: 변환된 DataFrame
        """
        raise NotImplementedError(f"Agent {self.__class__.__name__} doesn't support data transformation")
    
    async def cache_data(self, key: str, ttl: int = 3600) -> bool:
        """
        데이터 캐싱 (선택적 구현)
        
        Args:
            key: 캐시 키
            ttl: 캐시 유효 시간 (초)
            
        Returns:
            bool: 캐싱 성공 여부
        """
        return False  # 기본적으로 캐싱 비활성화
    
    async def get_loading_strategy(self, file_path: str, intent: DataIntent) -> LoadingStrategy:
        """
        데이터 로딩 전략 수립 (선택적 구현)
        
        Args:
            file_path: 파일 경로
            intent: 데이터 처리 의도
            
        Returns:
            LoadingStrategy: 최적화된 로딩 전략
        """
        return LoadingStrategy()  # 기본 전략 반환
    
    async def create_analysis_context(self, intent: DataIntent, smart_df: 'SmartDataFrame') -> Dict[str, Any]:
        """
        분석 컨텍스트 생성 (선택적 구현)
        
        Args:
            intent: 데이터 처리 의도
            smart_df: 지능형 DataFrame
            
        Returns:
            Dict[str, Any]: 분석 컨텍스트
        """
        return {
            "intent_type": intent.intent_type.value,
            "data_shape": smart_df.shape,
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_insights(self, results: Dict[str, Any], intent: DataIntent) -> List[str]:
        """
        인사이트 생성 (선택적 구현)
        
        Args:
            results: 분석 결과
            intent: 데이터 처리 의도
            
        Returns:
            List[str]: LLM이 생성한 인사이트 리스트
        """
        return []  # 기본적으로 빈 인사이트
    
    def extract_user_query(self, context: A2AContext) -> str:
        """
        A2A 컨텍스트에서 사용자 쿼리 추출
        
        Args:
            context: A2A 컨텍스트
            
        Returns:
            str: 사용자 쿼리
        """
        user_query = ""
        try:
            if hasattr(context.original_context, 'message') and context.original_context.message:
                if hasattr(context.original_context.message, 'parts'):
                    for part in context.original_context.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            user_query += part.root.text + " "
            return user_query.strip()
        except Exception as e:
            return ""
    
    def supports_feature(self, feature: str) -> bool:
        """
        에이전트가 특정 기능을 지원하는지 확인
        
        Args:
            feature: 기능 이름
            
        Returns:
            bool: 지원 여부
        """
        feature_mapping = {
            "data_loading": True,  # 모든 에이전트 기본 지원
            "data_transformation": hasattr(self, 'transform_data'),
            "caching": hasattr(self, 'cache_data'),
            "visualization": False,  # 기본적으로 미지원
            "modeling": False,
            "sql": False
        }
        return feature_mapping.get(feature, False)
    
    def get_supported_file_types(self) -> List[str]:
        """
        지원하는 파일 타입 리스트
        
        Returns:
            List[str]: 지원 파일 타입들
        """
        return ['csv', 'excel', 'xlsx', 'xls', 'json', 'parquet', 'feather']
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        에이전트 역량 정보
        
        Returns:
            Dict[str, Any]: 에이전트 역량 정보
        """
        return {
            "agent_name": self.__class__.__name__,
            "supported_intents": [intent.value for intent in DataIntentType],
            "supported_file_types": self.get_supported_file_types(),
            "features": {
                "data_loading": self.supports_feature("data_loading"),
                "data_transformation": self.supports_feature("data_transformation"),
                "caching": self.supports_feature("caching"),
                "visualization": self.supports_feature("visualization"),
                "modeling": self.supports_feature("modeling"),
                "sql": self.supports_feature("sql")
            },
            "llm_first": True,
            "a2a_compliant": True
        } 