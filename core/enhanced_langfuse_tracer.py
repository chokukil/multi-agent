"""
🔍 Enhanced Langfuse Tracer with Advanced Multi-Agent Logging
웹 검색 결과를 바탕으로 구현된 고급 Langfuse v2 추적 시스템

주요 기능:
- 네스팅된 스팬 구조로 에이전트 내부 로직 완전 추적
- 코드 생성 과정과 실행 결과 상세 로깅
- 멀티 에이전트 협업 추적
- 실시간 성능 모니터링
- 사용자 피드백 수집 준비
"""

import os
import time
import json
import uuid
import traceback
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import asyncio

# Langfuse SDK imports
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("⚠️ Langfuse SDK not available. Enhanced tracing disabled.")

class EnhancedLangfuseTracer:
    """고급 Langfuse 추적 시스템"""
    
    def __init__(self, 
                 public_key: str = None, 
                 secret_key: str = None, 
                 host: str = None,
                 project_name: str = "CherryAI-MultiAgent-v2"):
        """
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            project_name: 프로젝트 이름
        """
        self.enabled = LANGFUSE_AVAILABLE
        self.project_name = project_name
        self.current_session_id = None
        self.current_trace = None
        self.span_stack = []  # 네스팅된 스팬 관리
        self.agent_contexts = {}  # 에이전트별 컨텍스트
        
        if self.enabled:
            self._initialize_client(public_key, secret_key, host)
        
    def _initialize_client(self, public_key: str, secret_key: str, host: str):
        """Langfuse 클라이언트 초기화"""
        try:
            self.client = Langfuse(
                public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                host=host or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
            )
            print(f"✅ Enhanced Langfuse Tracer initialized - Project: {self.project_name}")
        except Exception as e:
            self.enabled = False
            print(f"❌ Langfuse initialization failed: {e}")
    
    def start_user_session(self, 
                          user_query: str, 
                          user_id: str = None,
                          session_metadata: Dict[str, Any] = None) -> str:
        """사용자 세션 시작 - 최상위 추적 컨텍스트"""
        if not self.enabled:
            return "disabled"
            
        self.current_session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        user_id = user_id or os.getenv("EMP_NO", "anonymous")
        
        # 메인 trace 생성
        self.current_trace = self.client.trace(
            name=f"User Query: {user_query[:50]}...",
            id=self.current_session_id,
            user_id=user_id,
            metadata={
                "session_id": self.current_session_id,
                "user_query": user_query,
                "query_length": len(user_query),
                "start_time": time.time(),
                "project": self.project_name,
                **(session_metadata or {})
            },
            tags=["user-session", "multi-agent", "cherry-ai"]
        )
        
        print(f"🔍 Started user session: {self.current_session_id}")
        return self.current_session_id
    
    @contextmanager
    def trace_agent_execution(self, 
                             agent_name: str,
                             task_description: str,
                             metadata: Dict[str, Any] = None):
        """에이전트 실행 추적 - 네스팅된 스팬 지원"""
        if not self.enabled or not self.current_trace:
            yield None
            return
            
        span_id = f"agent_{agent_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        start_time = time.time()
        
        # 에이전트 스팬 생성
        agent_span = self.current_trace.span(
            name=f"Agent: {agent_name}",
            id=span_id,
            metadata={
                "agent_name": agent_name,
                "task_description": task_description,
                "start_time": start_time,
                "session_id": self.current_session_id,
                **(metadata or {})
            },
            tags=["agent-execution", agent_name.lower()]
        )
        
        # 스팬 스택에 추가
        self.span_stack.append(agent_span)
        self.agent_contexts[agent_name] = {
            "span": agent_span,
            "start_time": start_time,
            "steps": []
        }
        
        try:
            yield agent_span
        except Exception as e:
            # 에러 추적
            agent_span.update(
                level="ERROR",
                status_message=f"Agent execution failed: {str(e)}",
                metadata={
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
        finally:
            # 실행 완료 추적
            duration = time.time() - start_time
            agent_span.update(
                end_time=time.time(),
                metadata={
                    "duration_seconds": duration,
                    "completed": True
                }
            )
            self.span_stack.pop()
            print(f"✅ Agent {agent_name} execution completed in {duration:.2f}s")
    
    @contextmanager
    def trace_internal_step(self, 
                           step_name: str,
                           step_type: str = "processing",
                           metadata: Dict[str, Any] = None):
        """에이전트 내부 단계 추적 - 세밀한 추적"""
        if not self.enabled or not self.span_stack:
            yield None
            return
            
        parent_span = self.span_stack[-1]
        step_id = f"step_{step_name}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        start_time = time.time()
        
        # 내부 단계 스팬 생성
        step_span = parent_span.span(
            name=f"Step: {step_name}",
            id=step_id,
            metadata={
                "step_name": step_name,
                "step_type": step_type,
                "start_time": start_time,
                **(metadata or {})
            },
            tags=["internal-step", step_type]
        )
        
        self.span_stack.append(step_span)
        
        try:
            yield step_span
        except Exception as e:
            step_span.update(
                level="ERROR",
                status_message=f"Step failed: {str(e)}",
                metadata={"error": str(e)}
            )
            raise
        finally:
            duration = time.time() - start_time
            step_span.update(
                end_time=time.time(),
                metadata={"duration_seconds": duration}
            )
            self.span_stack.pop()
    
    def log_code_generation(self, 
                           prompt: str,
                           generated_code: str,
                           execution_result: Any = None,
                           metadata: Dict[str, Any] = None):
        """코드 생성 과정 상세 로깅"""
        if not self.enabled or not self.span_stack:
            return
            
        current_span = self.span_stack[-1]
        
        # 코드 생성 이벤트 로깅
        current_span.event(
            name="code_generation",
            metadata={
                "prompt": prompt,
                "generated_code": generated_code,
                "code_length": len(generated_code),
                "execution_result": str(execution_result) if execution_result else None,
                "timestamp": time.time(),
                **(metadata or {})
            },
            tags=["code-generation"]
        )
        
        print(f"📝 Code generation logged: {len(generated_code)} characters")
    
    def log_llm_interaction(self,
                           model_name: str,
                           prompt: str,
                           response: str,
                           token_usage: Dict[str, int] = None,
                           metadata: Dict[str, Any] = None):
        """LLM 상호작용 상세 로깅"""
        if not self.enabled or not self.span_stack:
            return
            
        current_span = self.span_stack[-1]
        
        # LLM 상호작용 이벤트 로깅
        current_span.event(
            name="llm_interaction",
            metadata={
                "model_name": model_name,
                "prompt": prompt,
                "response": response,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "token_usage": token_usage,
                "timestamp": time.time(),
                **(metadata or {})
            },
            tags=["llm-interaction", model_name]
        )
        
        print(f"🤖 LLM interaction logged: {model_name}")
    
    def log_data_operation(self,
                          operation_type: str,
                          data_info: Dict[str, Any],
                          result_summary: str = None,
                          metadata: Dict[str, Any] = None):
        """데이터 처리 작업 로깅"""
        if not self.enabled or not self.span_stack:
            return
            
        current_span = self.span_stack[-1]
        
        # 데이터 작업 이벤트 로깅
        current_span.event(
            name="data_operation",
            metadata={
                "operation_type": operation_type,
                "data_info": data_info,
                "result_summary": result_summary,
                "timestamp": time.time(),
                **(metadata or {})
            },
            tags=["data-operation", operation_type]
        )
        
        print(f"📊 Data operation logged: {operation_type}")
    
    def log_agent_communication(self,
                               source_agent: str,
                               target_agent: str,
                               message: str,
                               response: str = None,
                               metadata: Dict[str, Any] = None):
        """에이전트 간 통신 로깅"""
        if not self.enabled or not self.current_trace:
            return
            
        # 에이전트 통신 이벤트 로깅
        self.current_trace.event(
            name="agent_communication",
            metadata={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "message": message,
                "response": response,
                "timestamp": time.time(),
                **(metadata or {})
            },
            tags=["agent-communication"]
        )
        
        print(f"💬 Agent communication logged: {source_agent} -> {target_agent}")
    
    def add_user_feedback(self,
                         feedback_type: str,
                         feedback_value: Any,
                         metadata: Dict[str, Any] = None):
        """사용자 피드백 추가"""
        if not self.enabled or not self.current_trace:
            return
            
        self.current_trace.score(
            name=feedback_type,
            value=feedback_value,
            metadata=metadata or {}
        )
        
        print(f"👍 User feedback added: {feedback_type} = {feedback_value}")
    
    def end_session(self, summary: str = None, metadata: Dict[str, Any] = None):
        """세션 종료"""
        if not self.enabled or not self.current_trace:
            return
            
        # 세션 종료 정보 업데이트
        self.current_trace.update(
            metadata={
                "session_summary": summary,
                "end_time": time.time(),
                "total_agents": len(self.agent_contexts),
                "agent_list": list(self.agent_contexts.keys()),
                **(metadata or {})
            }
        )
        
        print(f"🏁 Session ended: {self.current_session_id}")
        
        # 컨텍스트 정리
        self.current_session_id = None
        self.current_trace = None
        self.span_stack = []
        self.agent_contexts = {}

# 전역 인스턴스 생성 및 헬퍼 함수
_enhanced_tracer = None

def get_enhanced_tracer() -> EnhancedLangfuseTracer:
    """Enhanced Langfuse Tracer 인스턴스 가져오기"""
    global _enhanced_tracer
    if _enhanced_tracer is None:
        _enhanced_tracer = EnhancedLangfuseTracer()
    return _enhanced_tracer

def init_enhanced_tracer(public_key: str = None, 
                        secret_key: str = None, 
                        host: str = None) -> EnhancedLangfuseTracer:
    """Enhanced Langfuse Tracer 초기화"""
    global _enhanced_tracer
    _enhanced_tracer = EnhancedLangfuseTracer(public_key, secret_key, host)
    return _enhanced_tracer

# 데코레이터 함수들
def trace_agent_method(method_name: str = None):
    """에이전트 메서드 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_enhanced_tracer()
            name = method_name or func.__name__
            
            with tracer.trace_internal_step(name, "method_call"):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def trace_data_operation(operation_type: str = None):
    """데이터 처리 작업 추적 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_enhanced_tracer()
            op_type = operation_type or func.__name__
            
            with tracer.trace_internal_step(op_type, "data_operation"):
                result = func(*args, **kwargs)
                # 결과 정보 로깅
                if hasattr(result, 'shape'):  # pandas DataFrame 등
                    tracer.log_data_operation(op_type, {"shape": result.shape})
                return result
        return wrapper
    return decorator 