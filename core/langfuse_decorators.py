"""
Langfuse Decorators Helper

@observe 데코레이터를 쉽게 적용할 수 있는 헬퍼 함수들
- 안전한 데코레이터 적용
- 조건부 활성화
- 성능 메트릭 자동 수집

Author: CherryAI Team
License: MIT License
"""

import functools
import time
import logging
from typing import Any, Callable, Dict, Optional, Union
import os

# Langfuse imports with fallback
try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    
    # Fallback decorators that do nothing
    def observe(func: Callable = None, *, name: str = None, **kwargs) -> Callable:
        """Fallback observe decorator when Langfuse is not available"""
        def decorator(f: Callable) -> Callable:
            return f
        
        if func is None:
            return decorator
        return decorator(func)
    
    langfuse_context = None

logger = logging.getLogger(__name__)


def safe_observe(
    name: str = None,
    capture_input: bool = True,
    capture_output: bool = True,
    transform_to_string: bool = True,
    **kwargs
) -> Callable:
    """
    안전한 @observe 데코레이터 - Langfuse 사용 불가능한 경우에도 안전하게 작동
    
    Args:
        name: 추적할 작업 이름
        capture_input: 입력 캡처 여부
        capture_output: 출력 캡처 여부
        transform_to_string: 문자열 변환 여부
        **kwargs: 추가 observe 옵션
    """
    def decorator(func: Callable) -> Callable:
        if not LANGFUSE_AVAILABLE:
            # Langfuse 사용 불가능한 경우 원본 함수 반환
            return func
        
        # Langfuse 사용 가능한 경우 observe 적용
        actual_name = name or f"{func.__module__}.{func.__name__}"
        
        @observe(
            name=actual_name,
            capture_input=capture_input,
            capture_output=capture_output,
            transform_to_string=transform_to_string,
            **kwargs
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def observe_agent_execution(agent_name: str = None) -> Callable:
    """
    에이전트 실행 전용 observe 데코레이터
    
    Args:
        agent_name: 에이전트 이름
    """
    def decorator(func: Callable) -> Callable:
        actual_agent_name = agent_name or "Unknown Agent"
        
        @safe_observe(
            name=f"Agent Execution: {actual_agent_name}",
            capture_input=True,
            capture_output=True
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 성공 메트릭 기록
                execution_time = time.time() - start_time
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "agent_name": actual_agent_name,
                            "execution_time": execution_time,
                            "status": "success"
                        }
                    )
                
                return result
                
            except Exception as e:
                # 오류 메트릭 기록
                execution_time = time.time() - start_time
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "agent_name": actual_agent_name,
                            "execution_time": execution_time,
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )
                raise
        
        return wrapper
    
    return decorator


def observe_data_operation(operation_type: str = "data_processing") -> Callable:
    """
    데이터 처리 작업 전용 observe 데코레이터
    
    Args:
        operation_type: 작업 유형 (data_processing, analysis, visualization 등)
    """
    def decorator(func: Callable) -> Callable:
        @safe_observe(
            name=f"Data Operation: {operation_type}",
            capture_input=False,  # 데이터가 클 수 있으므로 입력 캡처 비활성화
            capture_output=False,  # 출력도 클 수 있으므로 비활성화
            transform_to_string=False
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 입력 데이터 크기 추정
            input_size = 0
            try:
                if args and hasattr(args[0], 'shape'):  # pandas DataFrame이나 numpy array
                    input_size = args[0].shape[0] if hasattr(args[0], 'shape') else len(args[0])
            except:
                pass
            
            try:
                result = func(*args, **kwargs)
                
                # 성공 메트릭 기록
                execution_time = time.time() - start_time
                output_size = 0
                
                try:
                    if hasattr(result, 'shape'):
                        output_size = result.shape[0] if hasattr(result, 'shape') else len(result)
                except:
                    pass
                
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "operation_type": operation_type,
                            "execution_time": execution_time,
                            "input_size": input_size,
                            "output_size": output_size,
                            "status": "success"
                        }
                    )
                
                return result
                
            except Exception as e:
                # 오류 메트릭 기록
                execution_time = time.time() - start_time
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "operation_type": operation_type,
                            "execution_time": execution_time,
                            "input_size": input_size,
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )
                raise
        
        return wrapper
    
    return decorator


def observe_llm_call(model_name: str = "unknown") -> Callable:
    """
    LLM 호출 전용 observe 데코레이터
    
    Args:
        model_name: LLM 모델 이름
    """
    def decorator(func: Callable) -> Callable:
        @safe_observe(
            name=f"LLM Call: {model_name}",
            capture_input=True,
            capture_output=True
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 입력 토큰 수 추정
            input_tokens = 0
            try:
                if args and isinstance(args[0], str):
                    input_tokens = len(args[0].split()) * 1.3  # 대략적인 토큰 수
            except:
                pass
            
            try:
                result = func(*args, **kwargs)
                
                # 성공 메트릭 기록
                execution_time = time.time() - start_time
                output_tokens = 0
                
                try:
                    if isinstance(result, str):
                        output_tokens = len(result.split()) * 1.3
                    elif hasattr(result, 'content'):
                        output_tokens = len(str(result.content).split()) * 1.3
                except:
                    pass
                
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "model_name": model_name,
                            "execution_time": execution_time,
                            "estimated_input_tokens": int(input_tokens),
                            "estimated_output_tokens": int(output_tokens),
                            "status": "success"
                        }
                    )
                
                return result
                
            except Exception as e:
                # 오류 메트릭 기록
                execution_time = time.time() - start_time
                if LANGFUSE_AVAILABLE and langfuse_context:
                    langfuse_context.update_current_observation(
                        metadata={
                            "model_name": model_name,
                            "execution_time": execution_time,
                            "estimated_input_tokens": int(input_tokens),
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )
                raise
        
        return wrapper
    
    return decorator


def observe_if_enabled(condition: Union[bool, Callable[[], bool]] = None) -> Callable:
    """
    조건부 observe 데코레이터 - 특정 조건에서만 추적 활성화
    
    Args:
        condition: 추적 활성화 조건 (bool 또는 callable)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 조건 확인
            should_trace = True
            if condition is not None:
                if callable(condition):
                    should_trace = condition()
                else:
                    should_trace = bool(condition)
            
            # 환경변수로도 제어 가능
            env_trace = os.getenv("LANGFUSE_TRACE_ENABLED", "true").lower() == "true"
            should_trace = should_trace and env_trace
            
            if should_trace and LANGFUSE_AVAILABLE:
                # 추적 활성화
                @observe(name=f"{func.__module__}.{func.__name__}")
                def traced_func(*args, **kwargs):
                    return func(*args, **kwargs)
                
                return traced_func(*args, **kwargs)
            else:
                # 추적 비활성화
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 편의 함수들
def is_langfuse_available() -> bool:
    """Langfuse 사용 가능 여부 확인"""
    return LANGFUSE_AVAILABLE


def get_trace_context() -> Optional[Any]:
    """현재 추적 컨텍스트 반환"""
    if LANGFUSE_AVAILABLE and langfuse_context:
        return langfuse_context.get_current_observation()
    return None


def update_trace_metadata(metadata: Dict[str, Any]) -> None:
    """현재 추적에 메타데이터 추가"""
    if LANGFUSE_AVAILABLE and langfuse_context:
        try:
            langfuse_context.update_current_observation(metadata=metadata)
        except Exception as e:
            logger.warning(f"Failed to update trace metadata: {e}")


# 사용 예시들
def example_usage():
    """사용 예시 함수들"""
    
    @safe_observe(name="Example Function")
    def example_function(data):
        return data * 2
    
    @observe_agent_execution("DataAnalysisAgent")
    async def example_agent_execution():
        return {"result": "analysis complete"}
    
    @observe_data_operation("data_cleaning")
    def example_data_processing(df):
        return df.dropna()
    
    @observe_llm_call("gpt-4")
    async def example_llm_call(prompt):
        return f"Response to: {prompt}"
    
    @observe_if_enabled(lambda: os.getenv("DEBUG_MODE") == "true")
    def example_conditional_trace():
        return "traced only in debug mode"


if __name__ == "__main__":
    # 간단한 테스트
    print(f"Langfuse Available: {is_langfuse_available()}")
    
    @safe_observe(name="Test Function")
    def test_function(x):
        return x + 1
    
    result = test_function(5)
    print(f"Test result: {result}") 