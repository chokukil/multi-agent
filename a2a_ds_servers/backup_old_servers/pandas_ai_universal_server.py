#!/usr/bin/env python3
"""
🤖 Universal Pandas-AI A2A Server

A2A SDK 0.2.9 준수 pandas-ai 통합 서버
범용적인 멀티 에이전트 데이터 분석 플랫폼 구현

Key Features:
- pandas-ai Agent 클래스 기반 자연어 데이터 분석
- A2A SDK 0.2.9 완전 호환
- UserFileTracker 통합 지원
- 실시간 스트리밍 응답
- 멀티턴 대화 및 컨텍스트 유지
- 범용 데이터 포맷 지원 (CSV, Excel, JSON 등)
"""

import asyncio
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from contextlib import asynccontextmanager

# A2A SDK 0.2.9 Import
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn
from starlette.responses import StreamingResponse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# pandas-ai 라이브러리 Import
try:
    import pandasai as pai
    from pandasai import Agent, DataFrame as PandasAIDataFrame
    from pandasai.config import Config
    from pandasai.llm.openai import OpenAI
    from pandasai.sandbox import Sandbox
    PANDAS_AI_AVAILABLE = True
    print("✅ pandas-ai 라이브러리 로드 성공")
except ImportError as e:
    PANDAS_AI_AVAILABLE = False
    print(f"❌ pandas-ai 라이브러리 로드 실패: {e}")
    print("📦 설치 명령: pip install pandasai")

# Enhanced Tracking System (선택적)
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False

# UserFileTracker 통합 (선택적)
try:
    from core.user_file_tracker import get_user_file_tracker
    from core.session_data_manager import SessionDataManager
    USER_FILE_TRACKER_AVAILABLE = True
except ImportError:
    USER_FILE_TRACKER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalPandasAIAgent:
    """
    범용 pandas-ai 에이전트 래퍼
    
    A2A SDK 0.2.9 호환 pandas-ai Agent 통합 클래스
    멀티턴 대화, 컨텍스트 유지, 실시간 스트리밍 지원
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.agent: Optional[Agent] = None
        self.dataframes: List[PandasAIDataFrame] = []
        self.conversation_history: List[Dict] = []
        self.session_id: Optional[str] = None
        
        # Enhanced Tracking 초기화
        self.enhanced_tracer = None
        if ENHANCED_TRACKING_AVAILABLE:
            try:
                self.enhanced_tracer = get_enhanced_tracer()
                logger.info("✅ Enhanced Langfuse Tracking 활성화")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Tracking 초기화 실패: {e}")
        
        # UserFileTracker 초기화
        self.user_file_tracker = None
        self.session_data_manager = None
        if USER_FILE_TRACKER_AVAILABLE:
            try:
                self.user_file_tracker = get_user_file_tracker()
                self.session_data_manager = SessionDataManager()
                logger.info("✅ UserFileTracker 통합 활성화")
            except Exception as e:
                logger.warning(f"⚠️ UserFileTracker 초기화 실패: {e}")
        
        # pandas-ai Config 설정
        self._setup_pandas_ai_config()
    
    def _setup_pandas_ai_config(self):
        """pandas-ai 설정 초기화"""
        try:
            # LLM 설정 (OpenAI 기본)
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return
            
            # pandas-ai 전역 설정
            pai.config.set("llm", OpenAI(api_token=openai_api_key))
            pai.config.set("verbose", self.config.get("verbose", True))
            pai.config.set("save_logs", self.config.get("save_logs", True))
            pai.config.set("save_charts", self.config.get("save_charts", True))
            pai.config.set("save_charts_path", self.config.get("save_charts_path", "artifacts/charts"))
            
            logger.info("✅ pandas-ai Config 설정 완료")
            
        except Exception as e:
            logger.error(f"❌ pandas-ai Config 설정 실패: {e}")
    
    async def load_data_from_file(self, file_path: str, **kwargs) -> bool:
        """
        파일에서 데이터를 로드하여 pandas-ai DataFrame으로 변환
        
        Args:
            file_path: 데이터 파일 경로
            **kwargs: 파일 로딩 옵션
            
        Returns:
            bool: 로딩 성공 여부
        """
        try:
            logger.info(f"🔄 데이터 로딩 시작: {file_path}")
            
            # Enhanced Tracking: 데이터 로딩 추적
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "data_loading",
                    {"file_path": file_path, "kwargs": kwargs},
                    f"Loading data from {file_path}"
                )
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"❌ 파일이 존재하지 않음: {file_path}")
                return False
            
            # 파일 확장자별 로딩
            if file_path.suffix.lower() in ['.csv']:
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.json']:
                df = pd.read_json(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.parquet']:
                df = pd.read_parquet(file_path, **kwargs)
            else:
                logger.error(f"❌ 지원하지 않는 파일 형식: {file_path.suffix}")
                return False
            
            # pandas-ai DataFrame으로 변환
            pandas_ai_df = PandasAIDataFrame(
                df,
                name=file_path.stem,
                description=f"Data loaded from {file_path.name}"
            )
            
            self.dataframes.append(pandas_ai_df)
            
            # Agent 재생성 (새로운 데이터프레임 포함)
            await self._recreate_agent()
            
            logger.info(f"✅ 데이터 로딩 완료: {file_path.name} ({df.shape[0]}행, {df.shape[1]}열)")
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def load_data_from_dataframe(self, df: pd.DataFrame, name: str = "DataFrame", description: str = None) -> bool:
        """
        pandas DataFrame에서 직접 로드
        
        Args:
            df: pandas DataFrame
            name: DataFrame 이름
            description: DataFrame 설명
            
        Returns:
            bool: 로딩 성공 여부
        """
        try:
            logger.info(f"🔄 DataFrame 로딩 시작: {name}")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "dataframe_loading",
                    {"name": name, "shape": df.shape, "columns": list(df.columns)},
                    f"Loading DataFrame {name}"
                )
            
            # pandas-ai DataFrame으로 변환
            pandas_ai_df = PandasAIDataFrame(
                df,
                name=name,
                description=description or f"DataFrame: {name}"
            )
            
            self.dataframes.append(pandas_ai_df)
            
            # Agent 재생성
            await self._recreate_agent()
            
            logger.info(f"✅ DataFrame 로딩 완료: {name} ({df.shape[0]}행, {df.shape[1]}열)")
            return True
            
        except Exception as e:
            logger.error(f"❌ DataFrame 로딩 실패: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _recreate_agent(self):
        """Agent 재생성 (새로운 데이터프레임들로)"""
        try:
            if not self.dataframes:
                logger.warning("⚠️ 로딩된 데이터프레임이 없습니다")
                return
            
            # pandas-ai Agent 생성
            self.agent = Agent(
                self.dataframes,
                description="Universal Data Analysis Agent powered by pandas-ai",
                memory_size=self.config.get("memory_size", 10)
            )
            
            logger.info(f"✅ Agent 재생성 완료 (데이터프레임 {len(self.dataframes)}개)")
            
        except Exception as e:
            logger.error(f"❌ Agent 재생성 실패: {e}")
            logger.error(traceback.format_exc())
    
    async def chat(self, query: str, session_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        자연어 쿼리 처리 (실시간 스트리밍)
        
        Args:
            query: 자연어 질문
            session_id: 세션 ID
            
        Yields:
            Dict: 스트리밍 응답 청크
        """
        if not self.agent:
            yield {
                "type": "error",
                "content": "에이전트가 초기화되지 않았습니다. 먼저 데이터를 로드해주세요.",
                "final": True
            }
            return
        
        self.session_id = session_id or self.session_id
        
        try:
            # Enhanced Tracking: 대화 시작
            if self.enhanced_tracer:
                self.enhanced_tracer.log_agent_communication(
                    source_agent="User",
                    target_agent="UniversalPandasAI",
                    message=query,
                    metadata={
                        "session_id": self.session_id,
                        "query_length": len(query),
                        "dataframes_count": len(self.dataframes)
                    }
                )
            
            # 진행 상태 알림
            yield {
                "type": "status",
                "content": "🤖 pandas-ai 에이전트가 분석 중입니다...",
                "final": False
            }
            
            # pandas-ai Agent를 통한 처리
            response = self.agent.chat(query)
            
            # 대화 기록 저장
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": str(response),
                "session_id": self.session_id
            })
            
            # 결과 분석
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # DataFrame 또는 리스트 결과
                yield {
                    "type": "data",
                    "content": f"📊 **분석 결과**\n\n{response}",
                    "data": response,
                    "final": False
                }
            else:
                # 텍스트 결과
                yield {
                    "type": "message",
                    "content": f"💡 **분석 결과**\n\n{response}",
                    "final": False
                }
            
            # 생성된 코드가 있다면 표시
            if hasattr(self.agent, 'last_code_generated') and self.agent.last_code_generated:
                yield {
                    "type": "code",
                    "content": f"```python\n{self.agent.last_code_generated}\n```",
                    "code": self.agent.last_code_generated,
                    "language": "python",
                    "final": False
                }
                
                # Enhanced Tracking: 코드 생성 추적
                if self.enhanced_tracer:
                    self.enhanced_tracer.log_code_generation(
                        prompt=query,
                        generated_code=self.agent.last_code_generated,
                        metadata={
                            "session_id": self.session_id,
                            "agent": "UniversalPandasAI"
                        }
                    )
            
            # 최종 완료 알림
            yield {
                "type": "completion",
                "content": "✅ 분석이 완료되었습니다.",
                "final": True
            }
            
        except Exception as e:
            logger.error(f"❌ 쿼리 처리 실패: {e}")
            logger.error(traceback.format_exc())
            
            yield {
                "type": "error",
                "content": f"분석 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "final": True
            }
    
    async def follow_up(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Follow-up 대화 (컨텍스트 유지)
        
        Args:
            query: Follow-up 질문
            
        Yields:
            Dict: 스트리밍 응답 청크
        """
        if not self.agent:
            yield {
                "type": "error",
                "content": "활성 대화가 없습니다. 먼저 chat()을 사용해 대화를 시작하세요.",
                "final": True
            }
            return
        
        try:
            yield {
                "type": "status",
                "content": "🔄 이전 대화 컨텍스트를 바탕으로 분석 중...",
                "final": False
            }
            
            # pandas-ai의 follow_up 기능 사용
            response = self.agent.follow_up(query)
            
            # 대화 기록 저장
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": f"[FOLLOW-UP] {query}",
                "response": str(response),
                "session_id": self.session_id
            })
            
            yield {
                "type": "message",
                "content": f"💡 **Follow-up 분석 결과**\n\n{response}",
                "final": True
            }
            
        except Exception as e:
            logger.error(f"❌ Follow-up 처리 실패: {e}")
            yield {
                "type": "error",
                "content": f"Follow-up 분석 중 오류가 발생했습니다: {str(e)}",
                "final": True
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """대화 기록 반환"""
        return self.conversation_history
    
    def clear_conversation(self):
        """대화 기록 및 메모리 정리"""
        try:
            if self.agent and hasattr(self.agent, '_state') and hasattr(self.agent._state, 'memory'):
                self.agent._state.memory.clear()
            self.conversation_history.clear()
            logger.info("✅ 대화 기록 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 대화 기록 정리 실패: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "agent_type": "UniversalPandasAI",
            "pandas_ai_available": PANDAS_AI_AVAILABLE,
            "enhanced_tracking": ENHANCED_TRACKING_AVAILABLE,
            "user_file_tracker": USER_FILE_TRACKER_AVAILABLE,
            "dataframes_loaded": len(self.dataframes),
            "conversation_turns": len(self.conversation_history),
            "session_id": self.session_id,
            "agent_active": self.agent is not None
        }

# A2A Server 구현
class UniversalPandasAIServer:
    """
    A2A SDK 0.2.9 준수 Universal Pandas-AI Server
    """
    
    def __init__(self):
        self.agent = UniversalPandasAIAgent()
        self.server_info = {
            "name": "Universal Pandas-AI Agent",
            "version": "1.0.0",
            "description": "범용 pandas-ai 기반 자연어 데이터 분석 에이전트",
            "capabilities": [
                "자연어 데이터 분석",
                "멀티턴 대화",
                "컨텍스트 유지",
                "범용 데이터 포맷 지원",
                "실시간 스트리밍",
                "코드 생성 및 실행",
                "시각화 생성"
            ],
            "data_formats": ["CSV", "Excel", "JSON", "Parquet"],
            "llm_backend": "OpenAI GPT",
            "a2a_sdk_version": "0.2.9"
        }
    
    async def load_data_from_session(self, session_id: str) -> bool:
        """
        UserFileTracker/SessionDataManager를 통해 세션 데이터 로드
        
        Args:
            session_id: 세션 ID
            
        Returns:
            bool: 로딩 성공 여부
        """
        if not USER_FILE_TRACKER_AVAILABLE:
            logger.warning("⚠️ UserFileTracker를 사용할 수 없습니다")
            return False
        
        try:
            # SessionDataManager를 통해 파일 정보 가져오기
            file_path, reason = self.agent.session_data_manager.get_file_for_a2a_agent(
                user_request="데이터 분석을 위한 파일 로딩",
                session_id=session_id,
                agent_name="UniversalPandasAI"
            )
            
            if not file_path:
                logger.warning(f"⚠️ 세션 {session_id}에서 파일을 찾을 수 없음: {reason}")
                return False
            
            # 파일 로딩
            success = await self.agent.load_data_from_file(file_path)
            
            if success:
                self.agent.session_id = session_id
                logger.info(f"✅ 세션 {session_id} 데이터 로딩 완료: {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 세션 데이터 로딩 실패: {e}")
            return False

class UniversalPandasAIExecutor(AgentExecutor):
    """A2A SDK 0.2.9 호환 Universal Pandas-AI Executor"""
    
    def __init__(self):
        self.agent = UniversalPandasAIAgent()
        logger.info("✅ Universal Pandas-AI Executor 초기화 완료")
    
    async def cancel(self) -> None:
        """A2A SDK 0.2.9 표준 cancel 메서드"""
        logger.info("🛑 Universal Pandas-AI Executor 취소 요청")
        # 필요시 정리 작업 수행
        if hasattr(self.agent, 'clear_conversation'):
            self.agent.clear_conversation()
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """A2A SDK 0.2.9 표준 execute 메서드"""
        try:
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            
            # 세션 데이터 로딩 시도
            session_id = context.request.get("session_id")
            if session_id and USER_FILE_TRACKER_AVAILABLE:
                await self._load_session_data(session_id, task_updater)
            
            # 상태 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🤖 pandas-ai 에이전트가 분석을 시작합니다..."
            )
            
            # pandas-ai 처리
            response_parts = []
            async for chunk in self.agent.chat(user_input, session_id):
                # 스트리밍 응답을 A2A 형식으로 변환
                if chunk.get("type") == "message":
                    response_parts.append(TextPart(text=chunk["content"]))
                elif chunk.get("type") == "code":
                    response_parts.append(TextPart(text=f"```python\n{chunk.get('code', '')}\n```"))
                elif chunk.get("type") == "data":
                    response_parts.append(TextPart(text=chunk["content"]))
                elif chunk.get("type") == "error":
                    await task_updater.update_status(
                        TaskState.failed,
                        message=chunk["content"]
                    )
                    return [TextPart(text=chunk["content"])]
                
                # 중간 상태 업데이트
                if not chunk.get("final", False):
                    await task_updater.update_status(
                        TaskState.working,
                        message=chunk.get("content", "처리 중...")
                    )
            
            # 최종 완료 상태
            await task_updater.update_status(
                TaskState.completed,
                message="✅ pandas-ai 분석이 완료되었습니다!"
            )
            
            return response_parts
            
        except Exception as e:
            error_msg = f"Universal Pandas-AI 실행 실패: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            await task_updater.update_status(
                TaskState.failed,
                message=error_msg
            )
            
            return [TextPart(text=error_msg)]
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """RequestContext에서 사용자 입력 추출"""
        try:
            if hasattr(context.request, 'messages') and context.request.messages:
                # 마지막 메시지에서 텍스트 추출
                last_message = context.request.messages[-1]
                if hasattr(last_message, 'parts') and last_message.parts:
                    text_parts = [part.text for part in last_message.parts if hasattr(part, 'text')]
                    return " ".join(text_parts)
            
            # 폴백: request 전체를 문자열로 변환
            return str(context.request)
            
        except Exception as e:
            logger.warning(f"사용자 입력 추출 실패: {e}")
            return "데이터를 분석해주세요"
    
    async def _load_session_data(self, session_id: str, task_updater: TaskUpdater):
        """세션 데이터 로딩"""
        try:
            await task_updater.update_status(
                TaskState.working,
                message=f"📂 세션 {session_id} 데이터를 로딩하는 중..."
            )
            
            # SessionDataManager를 통해 파일 정보 가져오기
            file_path, reason = self.agent.session_data_manager.get_file_for_a2a_agent(
                user_request="데이터 분석을 위한 파일 로딩",
                session_id=session_id,
                agent_name="UniversalPandasAI"
            )
            
            if file_path:
                success = await self.agent.load_data_from_file(file_path)
                if success:
                    self.agent.session_id = session_id
                    logger.info(f"✅ 세션 {session_id} 데이터 로딩 완료: {file_path}")
                    await task_updater.update_status(
                        TaskState.working,
                        message=f"✅ 데이터 로딩 완료: {os.path.basename(file_path)}"
                    )
                else:
                    logger.warning(f"⚠️ 세션 {session_id} 데이터 로딩 실패")
            else:
                logger.warning(f"⚠️ 세션 {session_id}에서 파일을 찾을 수 없음: {reason}")
                
        except Exception as e:
            logger.error(f"❌ 세션 데이터 로딩 실패: {e}")

def create_agent_card() -> AgentCard:
    """A2A SDK 0.2.9 호환 에이전트 카드 생성"""
    return AgentCard(
        name="Universal Pandas-AI Agent",
        avatar="🤖",
        description="범용 pandas-ai 기반 자연어 데이터 분석 에이전트",
        skills=[
            AgentSkill(
                name="자연어 데이터 분석",
                description="자연어로 데이터를 질문하면 pandas-ai가 코드를 생성하고 실행하여 답변합니다"
            ),
            AgentSkill(
                name="멀티턴 대화",
                description="이전 대화 맥락을 기억하여 연속적인 분석이 가능합니다"
            ),
            AgentSkill(
                name="범용 데이터 지원",
                description="CSV, Excel, JSON, Parquet 등 다양한 데이터 형식을 지원합니다"
            ),
            AgentSkill(
                name="코드 생성 및 실행",
                description="사용자 질문에 맞는 Python 코드를 자동으로 생성하고 실행합니다"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_media=False,
            supports_files=True
        )
    )

def create_app():
    """A2A SDK 0.2.9 호환 앱 생성"""
    # 에이전트 executor 생성
    executor = UniversalPandasAIExecutor()
    
    # Task store 생성
    task_store = InMemoryTaskStore()
    
    # Request handler 생성 (executor와 task_store 필요)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    # Agent card 설정
    agent_card = create_agent_card()
    
    # A2A Starlette 앱 생성
    app = A2AStarletteApplication(
        agent_card=agent_card,
        agent_executor=executor,
        task_store=task_store,
        request_handler=request_handler
    )
    
    return app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8315))
    
    print("🚀 Universal Pandas-AI A2A Server 시작 중...")
    print(f"📡 포트: {port}")
    print(f"🤖 pandas-ai 사용 가능: {PANDAS_AI_AVAILABLE}")
    print(f"📊 Enhanced Tracking: {ENHANCED_TRACKING_AVAILABLE}")
    print(f"📂 UserFileTracker: {USER_FILE_TRACKER_AVAILABLE}")
    print("=" * 60)
    
    # A2A 앱 생성 및 실행
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port) 