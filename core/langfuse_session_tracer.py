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

# Langfuse SDK v3 호환 import
try:
    from langfuse import get_client
    LANGFUSE_AVAILABLE = True
    print("✅ Langfuse SDK v3 import 성공")
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("⚠️ Langfuse SDK not available. Tracing will be disabled.")

class SessionBasedTracer:
    """Session 기반 langfuse 추적 시스템"""
    
    def __init__(self, public_key: str = None, secret_key: str = None, host: str = None):
        """
        Langfuse SDK v3 호환 추적 시스템 초기화
        
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
                # Langfuse v3 클라이언트 초기화
                if public_key and secret_key:
                    # 환경변수 설정
                    import os
                    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
                    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
                    if host:
                        os.environ["LANGFUSE_HOST"] = host
                
                # 전역 클라이언트 가져오기
                self.client = get_client()
                
                # 연결 테스트
                if self.client.auth_check():
                    print("✅ Langfuse SDK v3 클라이언트 초기화 성공")
                else:
                    print("❌ Langfuse 인증 실패")
                    self.enabled = False
                    
            except Exception as e:
                self.enabled = False
                print(f"❌ Langfuse 초기화 실패: {e}")
        else:
            self.client = None
            print("🔕 Langfuse 추적 비활성화됨")
    
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
            
            # 세션 레벨 span 시작 (v3 API)
            self.current_session_trace = self.client.start_as_current_span(
                name=f"User Query Session: {self.current_session_id}"
            )
            
            # 추가 메타데이터 설정
            if self.current_session_trace:
                self.current_session_trace.update_trace(
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
        
        start_time = time.time()
        
        try:
            # 에이전트 레벨 span 생성 (v3 API)
            with self.client.start_as_current_span(
                name=f"Agent: {agent_name}"
            ) as agent_span:
                # 메타데이터 설정
                agent_span.update(
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
            print(f"✅ Agent 추적 완료: {agent_name} ({execution_time:.2f}s)")
    
    def trace_agent_internal_logic(self, agent_name: str, operation: str, 
                                  input_data: Any, operation_metadata: Dict[str, Any] = None):
        """
        에이전트 내부 로직 추적 (v3 호환)
        
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
            # 내부 로직 span 생성 (v3 API)
            operation_span = self.client.start_as_current_span(
                name=f"{agent_name}.{operation}"
            )
            
            if operation_span:
                operation_span.update(
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
                
                # 결과 업데이트 (v3 API)
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
                
                # 결과 업데이트 (v3 API)
                span.update(
                    output={
                        "result": processed_result,
                        "execution_time": execution_time,
                        "success": success
                    },
                    metadata={
                        "completed_at": datetime.now().isoformat()
                    }
                )
                
                # span 종료
                span.end()
                
        except Exception as e:
            print(f"❌ Internal Operation 결과 기록 오류: {e}")
    
    def end_user_session(self, final_result: Dict[str, Any] = None, 
                        session_summary: Dict[str, Any] = None):
        """
        사용자 세션 종료
        
        Args:
            final_result: 최종 결과
            session_summary: 세션 요약
        """
        if not self.enabled or not self.current_session_trace:
            return
        
        try:
            # 세션 종료 업데이트 (v3 API)
            self.current_session_trace.update_trace(
                output={
                    "final_result": self._process_output_data(final_result) if final_result else None,
                    "session_summary": session_summary or {},
                    "session_id": self.current_session_id
                },
                metadata={
                    "session_ended_at": datetime.now().isoformat(),
                    "total_agents": len(self.agent_spans)
                }
            )
            
            # 세션 정리
            self.current_session_trace = None
            self.current_session_id = None
            self.agent_spans = {}
            
            print(f"🏁 Session 종료: {self.current_session_id}")
            
        except Exception as e:
            print(f"❌ Session 종료 오류: {e}")
    
    def _assess_query_complexity(self, query: str) -> str:
        """쿼리 복잡도 평가"""
        if len(query) < 20:
            return "simple"
        elif len(query) < 100:
            return "medium"
        else:
            return "complex"
    
    def _process_input_data(self, data: Any) -> Dict[str, Any]:
        """입력 데이터 처리"""
        try:
            if isinstance(data, dict):
                return data
            elif isinstance(data, str):
                return {"content": data}
            else:
                return {"data": str(data)}
        except Exception:
            return {"data": "processing_error"}
    
    def _process_output_data(self, data: Any) -> Dict[str, Any]:
        """출력 데이터 처리"""
        try:
            if isinstance(data, dict):
                return data
            elif isinstance(data, str):
                return {"content": data}
            else:
                return {"data": str(data)}
        except Exception:
            return {"data": "processing_error"}

# 전역 인스턴스 관리
_session_tracer = None

def get_session_tracer() -> SessionBasedTracer:
    """전역 세션 추적기 가져오기"""
    global _session_tracer
    if _session_tracer is None:
        _session_tracer = SessionBasedTracer()
    return _session_tracer

def init_session_tracer(public_key: str = None, secret_key: str = None, 
                       host: str = None) -> SessionBasedTracer:
    """세션 추적기 초기화"""
    global _session_tracer
    _session_tracer = SessionBasedTracer(public_key, secret_key, host)
    return _session_tracer 