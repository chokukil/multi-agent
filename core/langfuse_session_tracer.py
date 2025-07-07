"""
🔍 Langfuse Session-Based Tracing System
SDK v3를 사용한 session 기반 추적 시스템으로 하나의 사용자 질문에 대한 
모든 연쇄적 작업을 통합 추적

핵심 특징:
- Session-Based Grouping: 하나의 사용자 질문 = 하나의 session
- Hierarchical Tracing: 사용자 질문 → 에이전트 → 내부 로직 → 세부 분석
- A2A Agent Visibility: 각 A2A 에이전트 내부 처리 과정 완전 가시화
- SDK v3 OpenTelemetry: 자동 컨텍스트 전파 및 분산 추적
"""

import time
import json
import uuid
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from contextlib import contextmanager

# Langfuse SDK v2/v3 호환 import
try:
    # v3 import 시도
    from langfuse import Langfuse, get_client
    try:
        from langfuse.decorators import observe, langfuse_context
        LANGFUSE_V3 = True
    except ImportError:
        # v3 decorators를 import할 수 없는 경우 v2로 폴백
        LANGFUSE_V3 = False
        observe = None
        langfuse_context = None
    LANGFUSE_AVAILABLE = True
except ImportError:
    # v2 import 시도
    try:
        from langfuse import Langfuse
        get_client = None
        observe = None
        langfuse_context = None
        LANGFUSE_V3 = False
        LANGFUSE_AVAILABLE = True
    except ImportError:
        LANGFUSE_AVAILABLE = False
        print("⚠️ Langfuse SDK not available. Tracing will be disabled.")

class SessionBasedTracer:
    """Session 기반 langfuse 추적 시스템"""
    
    def __init__(self, public_key: str = None, secret_key: str = None, host: str = None):
        """
        Langfuse SDK v2/v3 호환 추적 시스템 초기화
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key  
            host: Langfuse host URL
        """
        self.enabled = LANGFUSE_AVAILABLE
        self.current_session_id: Optional[str] = None
        self.current_session_trace = None
        self.agent_spans: Dict[str, Any] = {}
        
        if self.enabled:
            try:
                # Langfuse 클라이언트 초기화
                if public_key and secret_key:
                    self.client = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host or "http://localhost:3000"
                    )
                else:
                    # 환경변수에서 자동 초기화
                    self.client = Langfuse()
                
                print(f"✅ Langfuse SDK {'v3' if LANGFUSE_V3 else 'v2'} 클라이언트 초기화 성공")
            except Exception as e:
                self.enabled = False
                print(f"❌ Langfuse 초기화 실패: {e}")
        else:
            self.client = None
            print("🔕 Langfuse 추적 비활성화됨")
    
    @property
    def trace_client(self):
        """현재 세션 trace 객체 반환 (wrapper 호환성용)"""
        return self.current_session_trace
    
    def start_user_session(self, user_query: str, user_id: str = "anonymous", 
                          session_metadata: Dict[str, Any] = None) -> str:
        """
        사용자 질문 세션 시작
        
        Args:
            user_query: 사용자 질문
            user_id: 사용자 ID
            session_metadata: 세션 메타데이터
            
        Returns:
            session_id: 생성된 세션 ID
        """
        if not self.enabled:
            return f"session_{int(time.time())}"
        
        try:
            # 고유한 세션 ID 생성
            timestamp = int(time.time())
            self.current_session_id = f"user_query_{timestamp}_{user_id}"
            
            # 세션 레벨 trace 시작
            self.current_session_trace = self.client.trace(
                name=f"User Query Session: {self.current_session_id}",
                user_id=user_id,
                session_id=self.current_session_id,
                input={"user_query": user_query},
                metadata={
                    "session_id": self.current_session_id,
                    "user_id": user_id,
                    "start_time": datetime.now().isoformat(),
                    "query_length": len(user_query),
                    "query_complexity": self._assess_query_complexity(user_query),
                    **(session_metadata or {})
                }
            )
            
            print(f"🎯 Session 시작: {self.current_session_id}")
            return self.current_session_id
            
        except Exception as e:
            print(f"❌ Session 시작 실패: {e}")
            return f"session_fallback_{int(time.time())}"
    
    @contextmanager
    def trace_agent_execution(self, agent_name: str, task_description: str, 
                             agent_metadata: Dict[str, Any] = None):
        """
        A2A 에이전트 실행 추적 컨텍스트 매니저
        
        Args:
            agent_name: 에이전트 이름
            task_description: 작업 설명
            agent_metadata: 에이전트 메타데이터
        """
        if not self.enabled or not self.current_session_trace:
            yield None
            return
        
        agent_span = None
        start_time = time.time()
        
        try:
            # 에이전트 레벨 span 생성
            agent_span = self.current_session_trace.span(
                name=f"Agent: {agent_name}",
                input={"task": task_description},
                metadata={
                    "agent_name": agent_name,
                    "task_description": task_description,
                    "session_id": self.current_session_id,
                    "start_time": datetime.now().isoformat(),
                    **(agent_metadata or {})
                }
            )
            
            # 에이전트 span 저장
            self.agent_spans[agent_name] = agent_span
            
            print(f"🤖 Agent 추적 시작: {agent_name}")
            yield agent_span
            
        except Exception as e:
            print(f"❌ Agent 추적 오류 ({agent_name}): {e}")
            yield None
        finally:
            execution_time = time.time() - start_time
            if agent_span:
                agent_span.update(
                    output={"execution_time": execution_time},
                    metadata={"completed_at": datetime.now().isoformat()}
                )
                agent_span.end()
            print(f"✅ Agent 추적 완료: {agent_name} ({execution_time:.2f}s)")
    
    def trace_agent_internal_logic(self, agent_name: str, operation: str, 
                                  input_data: Any, operation_metadata: Dict[str, Any] = None):
        """
        에이전트 내부 로직 추적 (v2/v3 호환)
        
        Args:
            agent_name: 에이전트 이름
            operation: 내부 작업명
            input_data: 입력 데이터
            operation_metadata: 작업 메타데이터
            
        Returns:
            operation_context: 작업 컨텍스트 (결과 기록용)
        """
        if not self.enabled:
            return {"enabled": False}
        
        try:
            # 에이전트 span 가져오기
            agent_span = self.agent_spans.get(agent_name)
            
            if agent_span:
                # 내부 로직 span 생성
                operation_span = agent_span.span(
                    name=f"{operation}",
                    input=self._process_input_data(input_data),
                    metadata={
                        "agent_name": agent_name,
                        "operation": operation,
                        "session_id": self.current_session_id,
                        "timestamp": datetime.now().isoformat(),
                        **(operation_metadata or {})
                    }
                )
                
                print(f"🔧 Internal Logic: {agent_name}.{operation}")
                
                return {
                    "enabled": True,
                    "span": operation_span,
                    "operation": operation,
                    "start_time": time.time()
                }
            
        except Exception as e:
            print(f"❌ Internal Logic 추적 오류: {e}")
        
        return {"enabled": False}
    
    def record_agent_result(self, agent_name: str, result: Dict[str, Any], 
                          confidence: float = 0.8, artifacts: List[Dict] = None):
        """
        에이전트 실행 결과 기록
        
        Args:
            agent_name: 에이전트 이름
            result: 실행 결과
            confidence: 신뢰도 점수
            artifacts: 생성된 아티팩트 목록
        """
        if not self.enabled:
            return
        
        try:
            agent_span = self.agent_spans.get(agent_name)
            if agent_span:
                # 결과 데이터 처리
                processed_result = self._process_output_data(result)
                
                # 아티팩트 정보 처리
                artifact_summary = []
                if artifacts:
                    for artifact in artifacts:
                        artifact_summary.append({
                            "name": artifact.get("name", "unknown"),
                            "type": artifact.get("type", "unknown"),
                            "size": len(str(artifact.get("content", "")))
                        })
                
                # 결과 업데이트
                agent_span.update(
                    output={
                        "result": processed_result,
                        "confidence": confidence,
                        "artifacts_count": len(artifacts) if artifacts else 0,
                        "artifacts_summary": artifact_summary
                    },
                    metadata={
                        "success": result.get("success", True),
                        "completed_at": datetime.now().isoformat()
                    }
                )
                
                print(f"📊 Agent 결과 기록: {agent_name} (신뢰도: {confidence:.1%})")
                
        except Exception as e:
            print(f"❌ Agent 결과 기록 오류: {e}")
    
    def record_internal_operation_result(self, operation_context: Dict[str, Any], 
                                       result: Any, success: bool = True):
        """
        내부 작업 결과 기록
        
        Args:
            operation_context: trace_agent_internal_logic에서 반환된 컨텍스트
            result: 작업 결과
            success: 성공 여부
        """
        if not self.enabled or not operation_context.get("enabled"):
            return
        
        try:
            span = operation_context.get("span")
            if span:
                execution_time = time.time() - operation_context.get("start_time", time.time())
                
                # 결과 데이터 처리
                processed_result = self._process_output_data(result)
                
                span.update(
                    output={
                        "result": processed_result,
                        "success": success,
                        "execution_time": execution_time
                    },
                    metadata={
                        "completed_at": datetime.now().isoformat(),
                        "operation": operation_context.get("operation")
                    }
                )
                span.end()
                
                print(f"✅ 내부 작업 완료: {operation_context.get('operation')} ({execution_time:.3f}s)")
                
        except Exception as e:
            print(f"❌ 내부 작업 결과 기록 오류: {e}")
    
    def end_user_session(self, final_result: Dict[str, Any] = None, 
                        session_summary: Dict[str, Any] = None):
        """
        사용자 질문 세션 종료
        
        Args:
            final_result: 최종 결과
            session_summary: 세션 요약 정보
        """
        if not self.enabled or not self.current_session_trace:
            return
        
        try:
            # 세션 요약 데이터 준비
            summary_data = {
                "session_id": self.current_session_id,
                "end_time": datetime.now().isoformat(),
                "agents_used": list(self.agent_spans.keys()),
                "total_agents": len(self.agent_spans),
                **(session_summary or {})
            }
            
            # 최종 결과 처리
            if final_result:
                processed_result = self._process_output_data(final_result)
                self.current_session_trace.update(
                    output=processed_result,
                    metadata=summary_data
                )
            
            print(f"🏁 Session 완료: {self.current_session_id}")
            
            # 리소스 정리
            self.current_session_trace = None
            self.current_session_id = None
            self.agent_spans.clear()
            
            # 이벤트 플러시 (단기 실행 스크립트의 경우)
            if hasattr(self.client, 'flush'):
                self.client.flush()
                
        except Exception as e:
            print(f"❌ Session 종료 오류: {e}")
    
    def _assess_query_complexity(self, query: str) -> str:
        """쿼리 복잡도 평가"""
        if len(query) < 100:
            return "simple"
        elif len(query) < 500:
            return "medium"
        elif len(query) < 1500:
            return "complex"
        else:
            return "very_complex"
    
    def _process_input_data(self, data: Any) -> Dict[str, Any]:
        """입력 데이터 처리 (크기 제한 및 안전성)"""
        try:
            if isinstance(data, str):
                return {"text": data[:1000] + "..." if len(data) > 1000 else data}
            elif isinstance(data, dict):
                return {k: str(v)[:500] for k, v in list(data.items())[:10]}
            elif isinstance(data, (list, tuple)):
                return {"items": [str(x)[:200] for x in data[:5]], "total_count": len(data)}
            else:
                return {"type": type(data).__name__, "value": str(data)[:500]}
        except Exception:
            return {"error": "Failed to process input data"}
    
    def _process_output_data(self, data: Any) -> Dict[str, Any]:
        """출력 데이터 처리 (크기 제한 및 안전성)"""
        try:
            if isinstance(data, dict):
                processed = {}
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 1000:
                        processed[k] = v[:1000] + "..."
                    elif isinstance(v, (list, tuple)) and len(v) > 10:
                        processed[k] = list(v[:10]) + [f"... and {len(v)-10} more items"]
                    else:
                        processed[k] = v
                return processed
            else:
                return self._process_input_data(data)
        except Exception:
            return {"error": "Failed to process output data"}

# 전역 트레이서 인스턴스
_global_tracer: Optional[SessionBasedTracer] = None

def get_session_tracer() -> SessionBasedTracer:
    """전역 session tracer 인스턴스 반환"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = SessionBasedTracer()
    return _global_tracer

def init_session_tracer(public_key: str = None, secret_key: str = None, 
                       host: str = None) -> SessionBasedTracer:
    """Session tracer 초기화"""
    global _global_tracer
    _global_tracer = SessionBasedTracer(public_key, secret_key, host)
    return _global_tracer 