"""
Base Server Class for A2A Data Science Servers

모든 A2A 서버가 상속받아 사용하는 기본 서버 클래스
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

# Try to import optional dependencies
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

try:
    from agents_sdk_server import ServeAgent
    from agents_sdk_server.models import AgentCard
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    ServeAgent = None
    AgentCard = None

logger = logging.getLogger(__name__)


class BaseA2AServer(ABC):
    """A2A 서버 베이스 클래스"""
    
    def __init__(
        self,
        server_name: str,
        host: str = "localhost",
        port: int = 8000,
        description: str = "",
        category: str = "data_science",
        tags: Optional[List[str]] = None
    ):
        """
        베이스 서버 초기화
        
        Args:
            server_name: 서버 이름
            host: 서버 호스트
            port: 서버 포트
            description: 서버 설명
            category: 서버 카테고리
            tags: 서버 태그 리스트
        """
        self.server_name = server_name
        self.host = host
        self.port = port
        self.description = description
        self.category = category
        self.tags = tags or []
        
        # 기본 로거 설정
        self._setup_logging()
        
        # AgentCard 초기화 (optional)
        if AGENTS_SDK_AVAILABLE:
            self.agent_card = self._create_agent_card()
        else:
            self.agent_card = None
        
        # Tools 초기화
        self.tools = []
        
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.server_name)
        
    def _create_agent_card(self):
        """AgentCard 생성"""
        if not AGENTS_SDK_AVAILABLE:
            return None
        return AgentCard(
            name=self.server_name,
            description=self.description,
            author="CherryAI",
            category=self.category,
            tags=self.tags
        )
        
    @abstractmethod
    def setup_tools(self) -> List[Any]:
        """
        서버별 도구 설정 - 각 서버에서 구현 필요
        
        Returns:
            도구 리스트
        """
        pass
        
    @abstractmethod
    def get_additional_dependencies(self) -> List[str]:
        """
        서버별 추가 의존성 반환 - 각 서버에서 구현 필요
        
        Returns:
            추가 의존성 리스트
        """
        pass
        
    def get_base_dependencies(self) -> List[str]:
        """
        기본 의존성 반환
        
        Returns:
            기본 의존성 리스트
        """
        return [
            "pydantic",
            "agents-sdk-server",
            "pandas",
            "numpy",
            "aiofiles"
        ]
        
    def get_all_dependencies(self) -> List[str]:
        """
        전체 의존성 반환
        
        Returns:
            전체 의존성 리스트
        """
        base_deps = self.get_base_dependencies()
        additional_deps = self.get_additional_dependencies()
        return list(set(base_deps + additional_deps))
        
    def create_serve_agent(self):
        """
        ServeAgent 인스턴스 생성
        
        Returns:
            ServeAgent 인스턴스 또는 None
        """
        if not AGENTS_SDK_AVAILABLE:
            self.logger.warning("agents_sdk_server not available")
            return None
            
        # 도구 설정
        self.tools = self.setup_tools()
        
        # ServeAgent 생성
        agent = ServeAgent(
            agent_card=self.agent_card,
            tools=self.tools,
            host=self.host,
            port=self.port
        )
        
        return agent
        
    def run(self):
        """서버 실행"""
        try:
            self.logger.info(f"🚀 {self.server_name} 서버 시작 중...")
            self.logger.info(f"📍 주소: http://{self.host}:{self.port}")
            
            # AgentCard 정보 로깅
            self.logger.info(f"📋 서버 정보:")
            self.logger.info(f"   - 이름: {self.agent_card.name}")
            self.logger.info(f"   - 설명: {self.agent_card.description}")
            self.logger.info(f"   - 카테고리: {self.agent_card.category}")
            self.logger.info(f"   - 태그: {', '.join(self.agent_card.tags)}")
            
            # 도구 정보 로깅
            agent = self.create_serve_agent()
            self.logger.info(f"🔧 등록된 도구: {len(self.tools)}개")
            for tool in self.tools:
                if hasattr(tool, '__name__'):
                    self.logger.info(f"   - {tool.__name__}")
                else:
                    self.logger.info(f"   - {tool}")
                    
            # 서버 실행
            if agent:
                agent.serve()
            else:
                self.logger.error("Cannot run server: ServeAgent creation failed")
            
        except Exception as e:
            self.logger.error(f"❌ 서버 실행 오류: {e}")
            raise
            
    async def health_check(self) -> Dict[str, Any]:
        """
        헬스 체크
        
        Returns:
            헬스 체크 결과
        """
        return {
            "status": "healthy",
            "server_name": self.server_name,
            "host": self.host,
            "port": self.port,
            "tools_count": len(self.tools)
        }
        
    def get_server_info(self) -> Dict[str, Any]:
        """
        서버 정보 반환
        
        Returns:
            서버 정보 딕셔너리
        """
        return {
            "name": self.server_name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "host": self.host,
            "port": self.port,
            "dependencies": self.get_all_dependencies(),
            "tools_count": len(self.tools)
        }


if PYDANTIC_AVAILABLE:
    class ToolResponse(BaseModel):
        """도구 응답 기본 모델"""
        success: bool
        message: str
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        
        
    class DataFrameInfo(BaseModel):
        """DataFrame 정보 모델"""
        shape: tuple
        columns: List[str]
        dtypes: Dict[str, str]
        memory_usage: str
else:
    # Fallback classes when Pydantic is not available
    class ToolResponse:
        def __init__(self, success: bool, message: str, data=None, error=None):
            self.success = success
            self.message = message
            self.data = data
            self.error = error
            
    class DataFrameInfo:
        def __init__(self, shape: tuple, columns: List[str], dtypes: Dict[str, str], memory_usage: str):
            self.shape = shape
            self.columns = columns
            self.dtypes = dtypes
            self.memory_usage = memory_usage
    
    
class ErrorHandler:
    """공통 에러 핸들러"""
    
    @staticmethod
    def handle_tool_error(func):
        """도구 에러 핸들링 데코레이터"""
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"도구 실행 오류 - {func.__name__}: {e}")
                return ToolResponse(
                    success=False,
                    message=f"도구 실행 실패: {func.__name__}",
                    error=str(e)
                )
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper