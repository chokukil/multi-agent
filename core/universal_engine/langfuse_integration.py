"""
Langfuse v2 통합 - SessionBasedTracer 및 EMP_NO=2055186 통합

Requirements 13에 따른 구현:
- SessionBasedTracer: 세션 ID 형식 user_query_{timestamp}_{user_id}
- LangfuseEnhancedA2AExecutor: 에이전트 실행 자동 추적
- RealTimeStreamingTaskUpdater: 스트리밍 중 trace 컨텍스트 유지
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime
from functools import wraps
import os

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not installed. Tracing will be disabled.")

logger = logging.getLogger(__name__)

# 고정 사용자 ID (EMP_NO)
DEFAULT_USER_ID = "2055186"


class SessionBasedTracer:
    """
    Langfuse v2 세션 기반 추적기
    
    세션 ID 형식: user_query_{timestamp}_{user_id}
    EMP_NO=2055186을 기본 user_id로 사용
    """
    
    def __init__(self, user_id: str = DEFAULT_USER_ID):
        """SessionBasedTracer 초기화"""
        self.user_id = user_id
        self.langfuse = None
        self.callback_handler = None
        self.current_session_id = None
        self.trace_data = {}
        
        if LANGFUSE_AVAILABLE:
            try:
                # Langfuse 클라이언트 초기화
                self.langfuse = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
                
                # 콜백 핸들러 생성
                self.callback_handler = CallbackHandler(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                    user_id=self.user_id
                )
                
                logger.info(f"Langfuse SessionBasedTracer initialized with user_id={user_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {str(e)}")
                self.langfuse = None
    
    def create_session(self, query: str) -> str:
        """
        새로운 세션 생성
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            생성된 세션 ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 쿼리의 처음 30자를 세션 ID에 포함 (특수문자 제거)
        query_snippet = "".join(c for c in query[:30] if c.isalnum() or c.isspace()).replace(" ", "_")
        
        self.current_session_id = f"user_query_{timestamp}_{self.user_id}_{query_snippet}"
        
        if self.langfuse:
            try:
                # Langfuse에 세션 시작 기록
                self.langfuse.trace(
                    id=self.current_session_id,
                    name="query_session",
                    user_id=self.user_id,
                    metadata={
                        "query": query,
                        "start_time": datetime.now().isoformat(),
                        "emp_no": self.user_id
                    }
                )
            except Exception as e:
                logger.error(f"Failed to create Langfuse session: {str(e)}")
        
        logger.info(f"Created session: {self.current_session_id}")
        return self.current_session_id
    
    def add_span(
        self, 
        name: str, 
        input_data: Any = None, 
        output_data: Any = None,
        metadata: Dict[str, Any] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """
        현재 세션에 span 추가
        
        Args:
            name: Span 이름
            input_data: 입력 데이터
            output_data: 출력 데이터
            metadata: 추가 메타데이터
            start_time: 시작 시간
            end_time: 종료 시간
        """
        if not self.current_session_id or not self.langfuse:
            return
        
        try:
            span_data = {
                "name": name,
                "trace_id": self.current_session_id,
                "user_id": self.user_id
            }
            
            if input_data is not None:
                span_data["input"] = input_data
            if output_data is not None:
                span_data["output"] = output_data
            if metadata:
                span_data["metadata"] = metadata
            if start_time:
                span_data["start_time"] = start_time
            if end_time:
                span_data["end_time"] = end_time
            
            self.langfuse.span(**span_data)
            
        except Exception as e:
            logger.error(f"Failed to add span: {str(e)}")
    
    def add_generation(
        self,
        name: str,
        model: str,
        prompt: Any,
        completion: Any,
        metadata: Dict[str, Any] = None,
        usage: Dict[str, Any] = None
    ):
        """
        LLM 생성 추가
        
        Args:
            name: Generation 이름
            model: 사용된 모델
            prompt: 프롬프트
            completion: 완성된 텍스트
            metadata: 추가 메타데이터
            usage: 토큰 사용량 정보
        """
        if not self.current_session_id or not self.langfuse:
            return
        
        try:
            gen_data = {
                "name": name,
                "trace_id": self.current_session_id,
                "model": model,
                "model_parameters": metadata or {},
                "input": prompt,
                "output": completion,
                "user_id": self.user_id
            }
            
            if usage:
                gen_data["usage"] = usage
            
            self.langfuse.generation(**gen_data)
            
        except Exception as e:
            logger.error(f"Failed to add generation: {str(e)}")
    
    def add_event(
        self,
        name: str,
        level: str = "DEFAULT",
        message: str = None,
        metadata: Dict[str, Any] = None
    ):
        """
        이벤트 추가
        
        Args:
            name: 이벤트 이름
            level: 로그 레벨
            message: 메시지
            metadata: 추가 메타데이터
        """
        if not self.current_session_id or not self.langfuse:
            return
        
        try:
            self.langfuse.event(
                trace_id=self.current_session_id,
                name=name,
                level=level,
                message=message,
                metadata=metadata,
                user_id=self.user_id
            )
        except Exception as e:
            logger.error(f"Failed to add event: {str(e)}")
    
    def end_session(self, metadata: Dict[str, Any] = None):
        """
        현재 세션 종료
        
        Args:
            metadata: 세션 종료 시 추가할 메타데이터
        """
        if not self.current_session_id or not self.langfuse:
            return
        
        try:
            # 세션 종료 이벤트 추가
            self.add_event(
                name="session_end",
                level="DEFAULT",
                message="Session completed",
                metadata={
                    **(metadata or {}),
                    "end_time": datetime.now().isoformat(),
                    "session_duration": self._calculate_session_duration()
                }
            )
            
            # Langfuse 플러시
            if self.langfuse:
                self.langfuse.flush()
            
        except Exception as e:
            logger.error(f"Failed to end session: {str(e)}")
        finally:
            self.current_session_id = None
    
    def _calculate_session_duration(self) -> float:
        """세션 지속 시간 계산"""
        # 세션 ID에서 타임스탬프 추출
        if self.current_session_id:
            try:
                parts = self.current_session_id.split("_")
                timestamp_str = f"{parts[2]}_{parts[3]}"  # YYYYMMDD_HHMMSS
                start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                duration = (datetime.now() - start_time).total_seconds()
                return duration
            except:
                pass
        return 0.0
    
    def get_callback_handler(self) -> Optional[CallbackHandler]:
        """Langfuse 콜백 핸들러 반환"""
        return self.callback_handler


class LangfuseEnhancedA2AExecutor:
    """
    Langfuse 강화 A2A 실행기
    
    A2A 에이전트 실행을 자동으로 추적
    """
    
    def __init__(self, tracer: SessionBasedTracer):
        """LangfuseEnhancedA2AExecutor 초기화"""
        self.tracer = tracer
    
    def trace_agent_execution(self, agent_name: str):
        """
        에이전트 실행 추적 데코레이터
        
        Args:
            agent_name: 에이전트 이름
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = datetime.now()
                
                # 에이전트 실행 시작 span
                self.tracer.add_span(
                    name=f"agent_execution_{agent_name}",
                    input_data={
                        "agent": agent_name,
                        "args": str(args)[:500],  # 긴 인자는 잘라냄
                        "kwargs": str(kwargs)[:500]
                    },
                    start_time=start_time
                )
                
                try:
                    # 실제 에이전트 실행
                    result = await func(*args, **kwargs)
                    
                    # 성공 span
                    self.tracer.add_span(
                        name=f"agent_success_{agent_name}",
                        output_data=str(result)[:1000],  # 결과 요약
                        metadata={
                            "execution_time": (datetime.now() - start_time).total_seconds()
                        },
                        end_time=datetime.now()
                    )
                    
                    return result
                    
                except Exception as e:
                    # 오류 이벤트
                    self.tracer.add_event(
                        name=f"agent_error_{agent_name}",
                        level="error",
                        message=str(e),
                        metadata={
                            "error_type": type(e).__name__,
                            "execution_time": (datetime.now() - start_time).total_seconds()
                        }
                    )
                    raise
            
            return wrapper
        return decorator
    
    async def execute_with_tracing(
        self,
        agent_name: str,
        agent_func,
        *args,
        **kwargs
    ) -> Any:
        """
        추적과 함께 에이전트 실행
        
        Args:
            agent_name: 에이전트 이름
            agent_func: 실행할 에이전트 함수
            *args, **kwargs: 에이전트 함수 인자
            
        Returns:
            에이전트 실행 결과
        """
        start_time = datetime.now()
        
        # 실행 시작 기록
        self.tracer.add_span(
            name=f"agent_start_{agent_name}",
            input_data={
                "agent": agent_name,
                "timestamp": start_time.isoformat()
            }
        )
        
        try:
            result = await agent_func(*args, **kwargs)
            
            # 실행 완료 기록
            self.tracer.add_span(
                name=f"agent_complete_{agent_name}",
                output_data={
                    "success": True,
                    "duration": (datetime.now() - start_time).total_seconds()
                },
                metadata={"agent": agent_name}
            )
            
            return result
            
        except Exception as e:
            # 오류 기록
            self.tracer.add_event(
                name=f"agent_failure_{agent_name}",
                level="error",
                message=str(e),
                metadata={
                    "agent": agent_name,
                    "error_type": type(e).__name__,
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            )
            raise


class RealTimeStreamingTaskUpdater:
    """
    실시간 스트리밍 작업 업데이터
    
    스트리밍 중 trace 컨텍스트 유지
    """
    
    def __init__(self, tracer: SessionBasedTracer):
        """RealTimeStreamingTaskUpdater 초기화"""
        self.tracer = tracer
        self.stream_metadata = {}
    
    async def stream_with_tracing(
        self,
        stream_name: str,
        stream_generator: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """
        추적과 함께 스트림 처리
        
        Args:
            stream_name: 스트림 이름
            stream_generator: 스트림 생성기
            
        Yields:
            스트림 데이터
        """
        start_time = datetime.now()
        chunk_count = 0
        
        # 스트림 시작 기록
        self.tracer.add_span(
            name=f"stream_start_{stream_name}",
            input_data={"stream": stream_name},
            start_time=start_time
        )
        
        try:
            async for chunk in stream_generator:
                chunk_count += 1
                
                # 주기적으로 진행 상황 기록 (매 10개 청크마다)
                if chunk_count % 10 == 0:
                    self.tracer.add_event(
                        name=f"stream_progress_{stream_name}",
                        level="DEFAULT",
                        message=f"Processed {chunk_count} chunks",
                        metadata={
                            "elapsed_time": (datetime.now() - start_time).total_seconds()
                        }
                    )
                
                yield chunk
            
            # 스트림 완료 기록
            self.tracer.add_span(
                name=f"stream_complete_{stream_name}",
                output_data={
                    "total_chunks": chunk_count,
                    "duration": (datetime.now() - start_time).total_seconds()
                },
                end_time=datetime.now()
            )
            
        except Exception as e:
            # 스트림 오류 기록
            self.tracer.add_event(
                name=f"stream_error_{stream_name}",
                level="error",
                message=str(e),
                metadata={
                    "chunks_processed": chunk_count,
                    "error_type": type(e).__name__,
                    "duration": (datetime.now() - start_time).total_seconds()
                }
            )
            raise
    
    def update_stream_metadata(self, key: str, value: Any):
        """스트림 메타데이터 업데이트"""
        self.stream_metadata[key] = value
        
        # Langfuse에 메타데이터 이벤트 추가
        self.tracer.add_event(
            name="stream_metadata_update",
            level="DEFAULT",
            metadata={
                "key": key,
                "value": str(value)[:500]  # 긴 값은 잘라냄
            }
        )
    
    def get_stream_summary(self) -> Dict[str, Any]:
        """스트림 요약 정보 반환"""
        return {
            "metadata": self.stream_metadata,
            "session_id": self.tracer.current_session_id,
            "user_id": self.tracer.user_id
        }


# 글로벌 트레이서 인스턴스
_global_tracer = None

def get_global_tracer() -> SessionBasedTracer:
    """글로벌 트레이서 인스턴스 반환"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = SessionBasedTracer()
    return _global_tracer

def reset_global_tracer(user_id: str = DEFAULT_USER_ID):
    """글로벌 트레이서 리셋"""
    global _global_tracer
    _global_tracer = SessionBasedTracer(user_id=user_id)