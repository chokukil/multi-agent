"""
🔍 Langfuse AI Data Science Team Wrapper
AI-Data-Science-Team 라이브러리의 내부 처리 과정을 Langfuse에서 추적하기 위한 wrapper

이 모듈은 다음 기능을 제공합니다:
- LLM 단계별 처리 과정 추적 (recommend_steps, create_code, execute_code)
- 생성된 Python 코드 및 실행 결과 아티팩트로 저장
- 실행 시간, 성능 메트릭, 에러 정보 추적
- 세션 기반 계층적 span 구조로 완전한 가시성 제공
- AI-Data-Science-Team 내부 워크플로우 단계별 세부 추적
"""

import time
import json
import traceback
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import pandas as pd

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    def observe(func):
        return func

class LangfuseAIDataScienceTeamWrapper:
    """
    AI-Data-Science-Team 라이브러리의 내부 처리 과정을 Langfuse에서 추적하는 wrapper
    """
    
    def __init__(self, session_tracer=None, agent_name: str = "Unknown Agent"):
        """
        Args:
            session_tracer: Langfuse session tracer 객체
            agent_name: 추적할 에이전트 이름
        """
        self.session_tracer = session_tracer
        self.agent_name = agent_name
        self.current_span = None
        self.step_counter = 0
        self.nested_spans = {}  # 중첩된 span 관리
        self.is_available = LANGFUSE_AVAILABLE and session_tracer is not None
        
        if not self.is_available:
            print(f"⚠️ Langfuse tracking disabled for {agent_name} (tracer: {session_tracer is not None}, available: {LANGFUSE_AVAILABLE})")
        
    def _safe_span_operation(self, operation_name: str, operation_func):
        """안전한 span 작업 실행"""
        if not self.is_available:
            return None
            
        try:
            return operation_func()
        except Exception as e:
            print(f"❌ Langfuse {operation_name} 실패: {e}")
            return None
        
    def create_agent_span(self, operation_name: str, input_data: Dict[str, Any]) -> Optional[Any]:
        """에이전트 작업에 대한 메인 span 생성"""
        def _create_span():
            if not hasattr(self.session_tracer, 'trace_client') or self.session_tracer.trace_client is None:
                print(f"❌ Session tracer에 trace_client가 없습니다.")
                return None
                
            span = self.session_tracer.trace_client.span(
                name=f"{self.agent_name} - {operation_name}",
                input=input_data,
                metadata={
                    "agent_type": self.agent_name,
                    "operation": operation_name,
                    "timestamp": time.time(),
                    "workflow_version": "v2.0_enhanced"
                }
            )
            self.current_span = span
            return span
            
        return self._safe_span_operation("agent span 생성", _create_span)
    
    def create_nested_span(self, span_name: str, parent_span_id: str = None, 
                          input_data: Dict[str, Any] = None) -> Optional[Any]:
        """계층적 중첩 span 생성"""
        def _create_nested_span():
            if not hasattr(self.session_tracer, 'trace_client') or self.session_tracer.trace_client is None:
                return None
                
            parent_id = parent_span_id
            if not parent_id and self.current_span and hasattr(self.current_span, 'id'):
                parent_id = self.current_span.id
            
            span = self.session_tracer.trace_client.span(
                name=span_name,
                input=input_data or {},
                metadata={
                    "agent_type": self.agent_name,
                    "nested_level": len(self.nested_spans) + 1,
                    "timestamp": time.time()
                },
                parent_observation_id=parent_id
            )
            
            # 중첩된 span 관리
            self.nested_spans[span_name] = span
            return span
            
        return self._safe_span_operation("nested span 생성", _create_nested_span)
    
    def trace_ai_ds_workflow_start(self, operation_type: str, input_data: Dict[str, Any]):
        """AI-Data-Science-Team 워크플로우 시작 추적"""
        if not self.is_available:
            return None
            
        workflow_span = self.create_nested_span(
            f"AI-DS-Team Workflow: {operation_type}",
            input_data={
                "operation_type": operation_type,
                "input_data": input_data,
                "workflow_stage": "initialization"
            }
        )
        
        if workflow_span:
            self.nested_spans['workflow'] = workflow_span
        return workflow_span
    
    def trace_data_analysis_step(self, data_summary: str, analysis_type: str = "data_inspection"):
        """데이터 분석 단계 추적"""
        if not self.is_available:
            return
            
        def _trace_analysis():
            self.step_counter += 1
            analysis_span = self.create_nested_span(
                f"Data Analysis: {analysis_type}",
                input_data={
                    "analysis_type": analysis_type,
                    "step_number": self.step_counter
                }
            )
            
            if analysis_span and hasattr(analysis_span, 'end'):
                analysis_span.end(
                    output={"data_summary": data_summary[:1000] + "..." if len(data_summary) > 1000 else data_summary}
                )
                
                # 데이터 요약을 아티팩트로 저장 (v2 호환)
                self._save_artifact_safe(
                    analysis_span,
                    f"data_summary_{analysis_type}.txt",
                    data_summary,
                    "text/plain",
                    {"type": "data_analysis", "step": self.step_counter}
                )
                
        self._safe_span_operation("데이터 분석 단계 추적", _trace_analysis)
    
    def _safe_span_end(self, span, output_data: Dict[str, Any] = None):
        """안전한 span 종료"""
        if not span or not hasattr(span, 'end'):
            return False
            
        try:
            if output_data:
                span.end(output=output_data)
            else:
                span.end()
            return True
        except Exception as e:
            print(f"❌ Span 종료 실패: {e}")
            return False
    
    def _save_artifact_safe(self, span, name: str, data: str, content_type: str = "text/plain", metadata: dict = None):
        """SDK v2/v3 호환 아티팩트 저장"""
        if not span:
            return False
            
        try:
            if hasattr(span, 'create_artifact'):
                # v3 방식
                span.create_artifact(
                    name=name,
                    data=data,
                    content_type=content_type,
                    metadata=metadata or {}
                )
                return True
            elif hasattr(span, 'update'):
                # v2 방식: metadata에 포함
                current_metadata = getattr(span, 'metadata', {}) or {}
                truncated_data = data[:500] + "..." if len(data) > 500 else data
                span.update(
                    metadata={
                        **current_metadata,
                        f"artifact_{name}": truncated_data,
                        f"artifact_{name}_type": content_type,
                        f"artifact_{name}_metadata": metadata or {}
                    }
                )
                return True
            else:
                print(f"⚠️ Span에 아티팩트 저장 메서드가 없습니다: {type(span)}")
                return False
        except Exception as e:
            print(f"⚠️ 아티팩트 저장 실패 ({name}): {e}")
            return False
    
    def trace_llm_recommendation_step(self, prompt: str, response: str, step_type: str = "recommendation"):
        """LLM 추천 단계 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            llm_span = self.create_nested_span(
                f"LLM Recommendation: {step_type}",
                input_data={
                    "step_type": step_type,
                    "step_number": self.step_counter,
                    "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
                }
            )
            
            if llm_span:
                llm_span.end(
                    output={
                        "response_preview": response[:200] + "..." if len(response) > 200 else response,
                        "response_length": len(response)
                    }
                )
                
                # LLM 상호작용을 아티팩트로 저장
                llm_span.create_artifact(
                    name=f"llm_prompt_{step_type}_{self.step_counter}.txt",
                    data=prompt,
                    content_type="text/plain",
                    metadata={"type": "llm_prompt", "step": self.step_counter}
                )
                
                llm_span.create_artifact(
                    name=f"llm_response_{step_type}_{self.step_counter}.txt",
                    data=response,
                    content_type="text/plain",
                    metadata={"type": "llm_response", "step": self.step_counter}
                )
                
        except Exception as e:
            print(f"❌ LLM 추천 단계 추적 실패: {e}")
    
    def trace_code_generation_step(self, prompt: str, generated_code: str, 
                                 code_type: str = "data_processing"):
        """코드 생성 단계 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            code_span = self.create_nested_span(
                f"Code Generation: {code_type}",
                input_data={
                    "code_type": code_type,
                    "step_number": self.step_counter,
                    "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
                }
            )
            
            if code_span:
                # 코드 분석
                code_lines = len(generated_code.split('\n'))
                imports = [line.strip() for line in generated_code.split('\n') 
                          if line.strip().startswith('import') or line.strip().startswith('from')]
                
                code_span.end(
                    output={
                        "code_lines": code_lines,
                        "imports_count": len(imports),
                        "imports": imports[:5],  # 처음 5개만 표시
                        "code_preview": generated_code[:300] + "..." if len(generated_code) > 300 else generated_code
                    }
                )
                
                # 생성된 코드를 아티팩트로 저장
                code_span.create_artifact(
                    name=f"generated_code_{code_type}_{self.step_counter}.py",
                    data=generated_code,
                    content_type="text/x-python",
                    metadata={
                        "type": "generated_code",
                        "step": self.step_counter,
                        "code_type": code_type,
                        "lines": code_lines
                    }
                )
                
                # 프롬프트도 저장
                code_span.create_artifact(
                    name=f"code_generation_prompt_{self.step_counter}.txt",
                    data=prompt,
                    content_type="text/plain",
                    metadata={"type": "code_generation_prompt", "step": self.step_counter}
                )
                
        except Exception as e:
            print(f"❌ 코드 생성 단계 추적 실패: {e}")
    
    def trace_code_execution_step(self, code: str, execution_result: Any = None, 
                                execution_time: float = 0.0, error: Optional[str] = None):
        """코드 실행 단계 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            exec_span = self.create_nested_span(
                "Code Execution",
                input_data={
                    "step_number": self.step_counter,
                    "code_preview": code[:200] + "..." if len(code) > 200 else code,
                    "execution_time": execution_time
                }
            )
            
            if exec_span:
                success = error is None
                
                # 실행 결과 분석
                result_summary = self._analyze_execution_result(execution_result)
                
                exec_span.end(
                    output={
                        "success": success,
                        "execution_time": execution_time,
                        "error": error,
                        "result_summary": result_summary
                    }
                )
                
                # 실행 결과를 아티팩트로 저장
                if success and execution_result is not None:
                    exec_span.create_artifact(
                        name=f"execution_result_{self.step_counter}.json",
                        data=json.dumps(result_summary, indent=2, default=str),
                        content_type="application/json",
                        metadata={"type": "execution_result", "step": self.step_counter}
                    )
                
                if error:
                    exec_span.create_artifact(
                        name=f"execution_error_{self.step_counter}.txt",
                        data=error,
                        content_type="text/plain",
                        metadata={"type": "execution_error", "step": self.step_counter}
                    )
                
        except Exception as e:
            print(f"❌ 코드 실행 단계 추적 실패: {e}")
    
    def trace_data_transformation_step(self, input_data: Any, output_data: Any, 
                                     transformation_type: str = "data_processing"):
        """데이터 변환 단계 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            
            # 데이터 변환 분석
            input_summary = self._get_data_summary(input_data, "input")
            output_summary = self._get_data_summary(output_data, "output")
            
            transform_span = self.create_nested_span(
                f"Data Transformation: {transformation_type}",
                input_data={
                    "transformation_type": transformation_type,
                    "step_number": self.step_counter,
                    "input_summary": input_summary
                }
            )
            
            if transform_span:
                transform_span.end(
                    output={
                        "output_summary": output_summary,
                        "transformation_applied": transformation_type
                    }
                )
                
                # 변환 전후 데이터 샘플 저장
                if input_data is not None:
                    input_sample = self._get_data_sample(input_data, 3)
                    transform_span.create_artifact(
                        name=f"input_sample_{self.step_counter}.json",
                        data=json.dumps(input_sample, indent=2, default=str),
                        content_type="application/json",
                        metadata={"type": "input_sample", "step": self.step_counter}
                    )
                
                if output_data is not None:
                    output_sample = self._get_data_sample(output_data, 3)
                    transform_span.create_artifact(
                        name=f"output_sample_{self.step_counter}.json",
                        data=json.dumps(output_sample, indent=2, default=str),
                        content_type="application/json",
                        metadata={"type": "output_sample", "step": self.step_counter}
                    )
                
        except Exception as e:
            print(f"❌ 데이터 변환 단계 추적 실패: {e}")
    
    def trace_workflow_completion(self, final_result: Any, workflow_summary: str = ""):
        """워크플로우 완료 단계 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            if 'workflow' in self.nested_spans:
                workflow_span = self.nested_spans['workflow']
                
                result_summary = self._analyze_execution_result(final_result)
                
                workflow_span.end(
                    output={
                        "workflow_completed": True,
                        "total_steps": self.step_counter,
                        "final_result_summary": result_summary,
                        "workflow_summary": workflow_summary[:500] + "..." if len(workflow_summary) > 500 else workflow_summary
                    }
                )
                
                # 워크플로우 요약 저장
                if workflow_summary:
                    workflow_span.create_artifact(
                        name="workflow_summary.md",
                        data=workflow_summary,
                        content_type="text/markdown",
                        metadata={"type": "workflow_summary", "total_steps": self.step_counter}
                    )
                
        except Exception as e:
            print(f"❌ 워크플로우 완료 추적 실패: {e}")
    
    def _analyze_execution_result(self, result: Any) -> Dict[str, Any]:
        """실행 결과 분석"""
        try:
            if result is None:
                return {"type": "None", "value": None}
            
            if isinstance(result, pd.DataFrame):
                return {
                    "type": "pandas.DataFrame",
                    "shape": result.shape,
                    "columns": list(result.columns),
                    "memory_usage_mb": round(result.memory_usage(deep=True).sum() / 1024**2, 2),
                    "null_counts": result.isnull().sum().to_dict()
                }
            elif isinstance(result, dict):
                return {
                    "type": "dict",
                    "keys": list(result.keys())[:10],  # 처음 10개 키만
                    "size": len(result)
                }
            elif isinstance(result, (list, tuple)):
                return {
                    "type": type(result).__name__,
                    "length": len(result),
                    "sample_items": result[:3] if len(result) > 0 else []
                }
            else:
                return {
                    "type": type(result).__name__,
                    "value": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                }
        except Exception:
            return {"type": "unknown", "error": "analysis_failed"}
    
    # 기존 메서드들 유지
    def trace_llm_step(self, step_name: str, prompt: str, response: str, 
                      execution_time: float = 0.0, metadata: Dict[str, Any] = None):
        """LLM 단계 추적 (recommend_steps, create_code 등)"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            step_metadata = {
                "step_number": self.step_counter,
                "step_type": "llm_processing",
                "execution_time_seconds": execution_time,
                "prompt_length": len(prompt),
                "response_length": len(response),
                **(metadata or {})
            }
            
            # 현재 span의 하위 span으로 LLM 단계 기록
            if self.current_span:
                llm_span = self.session_tracer.trace_client.span(
                    name=f"LLM Step: {step_name}",
                    input={"prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt},
                    output={"response": response[:500] + "..." if len(response) > 500 else response},
                    metadata=step_metadata,
                    parent_observation_id=self.current_span.id
                )
                
                # 전체 prompt와 response를 아티팩트로 저장
                self._save_llm_artifacts(llm_span, step_name, prompt, response)
                
        except Exception as e:
            print(f"❌ Langfuse LLM 단계 추적 실패: {e}")
    
    def trace_code_execution(self, code: str, result: Any, execution_time: float = 0.0, 
                           error: Optional[str] = None, metadata: Dict[str, Any] = None):
        """생성된 코드 실행 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            exec_metadata = {
                "step_number": self.step_counter,
                "step_type": "code_execution",
                "execution_time_seconds": execution_time,
                "code_lines": len(code.split('\n')),
                "success": error is None,
                "error": error,
                **(metadata or {})
            }
            
            if self.current_span:
                exec_span = self.session_tracer.trace_client.span(
                    name="Code Execution",
                    input={"generated_code": code},
                    output={"success": error is None, "error": error},
                    metadata=exec_metadata,
                    parent_observation_id=self.current_span.id
                )
                
                # 코드와 결과를 아티팩트로 저장
                self._save_code_artifacts(exec_span, code, result, error)
                
        except Exception as e:
            print(f"❌ Langfuse 코드 실행 추적 실패: {e}")
    
    def trace_data_transformation(self, input_data: Any, output_data: Any, 
                                operation: str, metadata: Dict[str, Any] = None):
        """데이터 변환 과정 추적"""
        if not self.session_tracer or not LANGFUSE_AVAILABLE:
            return
            
        try:
            self.step_counter += 1
            
            # 데이터 요약 정보 생성
            input_summary = self._get_data_summary(input_data, "input")
            output_summary = self._get_data_summary(output_data, "output")
            
            transform_metadata = {
                "step_number": self.step_counter,
                "step_type": "data_transformation",
                "operation": operation,
                "input_summary": input_summary,
                "output_summary": output_summary,
                **(metadata or {})
            }
            
            if self.current_span:
                transform_span = self.session_tracer.trace_client.span(
                    name=f"Data Transform: {operation}",
                    input=input_summary,
                    output=output_summary,
                    metadata=transform_metadata,
                    parent_observation_id=self.current_span.id
                )
                
                # 데이터 샘플을 아티팩트로 저장
                self._save_data_artifacts(transform_span, input_data, output_data, operation)
                
        except Exception as e:
            print(f"❌ Langfuse 데이터 변환 추적 실패: {e}")
    
    def finalize_agent_span(self, final_result: Any, success: bool = True, 
                          error: Optional[str] = None):
        """에이전트 span 완료"""
        if not self.current_span or not LANGFUSE_AVAILABLE:
            return
            
        try:
            # 최종 결과 요약
            result_summary = self._get_data_summary(final_result, "final_result")
            
            self.current_span.end(
                output={
                    "success": success,
                    "error": error,
                    "total_steps": self.step_counter,
                    "final_result_summary": result_summary
                }
            )
            
            # 최종 결과를 아티팩트로 저장
            if final_result is not None:
                self._save_final_result_artifact(final_result)
                
        except Exception as e:
            print(f"❌ Langfuse agent span 완료 실패: {e}")
        finally:
            # 모든 nested span들도 정리
            for span_name, span in self.nested_spans.items():
                try:
                    if hasattr(span, 'end'):
                        span.end()
                except:
                    pass
            
            self.nested_spans.clear()
            self.current_span = None
            self.step_counter = 0
    
    def _save_llm_artifacts(self, span, step_name: str, prompt: str, response: str):
        """LLM 프롬프트와 응답을 아티팩트로 저장"""
        try:
            # Prompt 아티팩트
            span.create_artifact(
                name=f"{step_name}_prompt.txt",
                data=prompt,
                content_type="text/plain",
                metadata={"type": "llm_prompt", "step": step_name}
            )
            
            # Response 아티팩트
            span.create_artifact(
                name=f"{step_name}_response.txt",
                data=response,
                content_type="text/plain",
                metadata={"type": "llm_response", "step": step_name}
            )
        except Exception as e:
            print(f"❌ LLM 아티팩트 저장 실패: {e}")
    
    def _save_code_artifacts(self, span, code: str, result: Any, error: Optional[str]):
        """생성된 코드와 실행 결과를 아티팩트로 저장"""
        try:
            # 생성된 코드 저장
            span.create_artifact(
                name="generated_code.py",
                data=code,
                content_type="text/x-python",
                metadata={"type": "generated_code"}
            )
            
            # 실행 결과 저장
            if error:
                span.create_artifact(
                    name="execution_error.txt",
                    data=error,
                    content_type="text/plain",
                    metadata={"type": "execution_error"}
                )
            elif result is not None:
                result_str = self._serialize_result(result)
                span.create_artifact(
                    name="execution_result.json",
                    data=result_str,
                    content_type="application/json",
                    metadata={"type": "execution_result"}
                )
        except Exception as e:
            print(f"❌ 코드 아티팩트 저장 실패: {e}")
    
    def _save_data_artifacts(self, span, input_data: Any, output_data: Any, operation: str):
        """데이터 변환 과정의 입력/출력 샘플을 아티팩트로 저장"""
        try:
            # 입력 데이터 샘플
            if input_data is not None:
                input_sample = self._get_data_sample(input_data)
                span.create_artifact(
                    name=f"{operation}_input_sample.json",
                    data=json.dumps(input_sample, indent=2, default=str),
                    content_type="application/json",
                    metadata={"type": "input_sample", "operation": operation}
                )
            
            # 출력 데이터 샘플
            if output_data is not None:
                output_sample = self._get_data_sample(output_data)
                span.create_artifact(
                    name=f"{operation}_output_sample.json",
                    data=json.dumps(output_sample, indent=2, default=str),
                    content_type="application/json",
                    metadata={"type": "output_sample", "operation": operation}
                )
        except Exception as e:
            print(f"❌ 데이터 아티팩트 저장 실패: {e}")
    
    def _save_final_result_artifact(self, final_result: Any):
        """최종 결과를 아티팩트로 저장"""
        try:
            if self.current_span:
                result_data = self._serialize_result(final_result)
                self.current_span.create_artifact(
                    name="final_result.json",
                    data=result_data,
                    content_type="application/json",
                    metadata={"type": "final_result", "agent": self.agent_name}
                )
        except Exception as e:
            print(f"❌ 최종 결과 아티팩트 저장 실패: {e}")
    
    def _get_data_summary(self, data: Any, data_type: str) -> Dict[str, Any]:
        """데이터 요약 정보 생성"""
        try:
            if data is None:
                return {"type": "None", "data_type": data_type}
            
            if isinstance(data, pd.DataFrame):
                return {
                    "type": "pandas.DataFrame",
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024**2, 2),
                    "data_type": data_type
                }
            elif isinstance(data, dict):
                return {
                    "type": "dict",
                    "keys": list(data.keys()) if len(data.keys()) <= 20 else f"{len(data.keys())} keys",
                    "size": len(data),
                    "data_type": data_type
                }
            elif isinstance(data, (list, tuple)):
                return {
                    "type": type(data).__name__,
                    "length": len(data),
                    "sample_types": [type(item).__name__ for item in data[:5]],
                    "data_type": data_type
                }
            else:
                return {
                    "type": type(data).__name__,
                    "value_preview": str(data)[:100],
                    "data_type": data_type
                }
        except Exception:
            return {"type": "unknown", "error": "summary_failed", "data_type": data_type}
    
    def _get_data_sample(self, data: Any, max_rows: int = 5) -> Any:
        """데이터 샘플 추출"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.head(max_rows).to_dict()
            elif isinstance(data, dict):
                if len(data) <= 10:
                    return data
                else:
                    # 처음 몇 개 키만 샘플로 반환
                    sample_keys = list(data.keys())[:max_rows]
                    return {k: data[k] for k in sample_keys}
            elif isinstance(data, (list, tuple)):
                return data[:max_rows]
            else:
                return str(data)[:500]
        except Exception:
            return {"error": "sample_extraction_failed"}
    
    def _serialize_result(self, result: Any) -> str:
        """결과를 JSON 문자열로 직렬화"""
        try:
            if isinstance(result, pd.DataFrame):
                return json.dumps({
                    "type": "pandas.DataFrame",
                    "shape": result.shape,
                    "data_sample": result.head(10).to_dict(),
                    "columns": list(result.columns),
                    "dtypes": result.dtypes.to_dict()
                }, indent=2, default=str)
            else:
                return json.dumps(result, indent=2, default=str)
        except Exception:
            return json.dumps({"type": type(result).__name__, "error": "serialization_failed"}, indent=2)


def trace_ai_ds_team_operation(wrapper: LangfuseAIDataScienceTeamWrapper, operation_name: str):
    """
    AI-Data-Science-Team 작업을 추적하는 데코레이터
    
    Usage:
        @trace_ai_ds_team_operation(wrapper, "data_cleaning")
        def invoke_data_cleaning_agent(user_instructions, data_raw):
            # agent 실행 코드
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            start_time = time.time()
            
            # 입력 데이터 수집
            input_data = {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            # 메인 span 시작
            span = wrapper.create_agent_span(operation_name, input_data)
            
            try:
                # 원본 함수 실행
                result = func(*args, **kwargs)
                
                # 성공적으로 완료
                execution_time = time.time() - start_time
                wrapper.finalize_agent_span(result, success=True)
                
                return result
                
            except Exception as e:
                # 에러 발생 시
                execution_time = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                wrapper.finalize_agent_span(None, success=False, error=error_msg)
                
                # 원본 에러 재발생
                raise
                
        return wrapper_func
    return decorator


# 편의 함수들
def create_ai_ds_team_wrapper(session_tracer, agent_name: str) -> LangfuseAIDataScienceTeamWrapper:
    """AI-Data-Science-Team wrapper 생성 편의 함수"""
    return LangfuseAIDataScienceTeamWrapper(session_tracer, agent_name)

def trace_llm_recommendation_step(wrapper: LangfuseAIDataScienceTeamWrapper, 
                                step_name: str, prompt: str, response: str, 
                                execution_time: float = 0.0):
    """LLM 권장 단계 추적 편의 함수"""
    wrapper.trace_llm_step(
        step_name=step_name, 
        prompt=prompt, 
        response=response, 
        execution_time=execution_time,
        metadata={"category": "recommendation"}
    )

def trace_llm_code_generation_step(wrapper: LangfuseAIDataScienceTeamWrapper, 
                                 step_name: str, prompt: str, response: str, 
                                 execution_time: float = 0.0):
    """LLM 코드 생성 단계 추적 편의 함수"""
    wrapper.trace_llm_step(
        step_name=step_name, 
        prompt=prompt, 
        response=response, 
        execution_time=execution_time,
        metadata={"category": "code_generation"}
    ) 