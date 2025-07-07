"""
🔧 A2A Agent Internal Tracing Helpers
A2A 에이전트 내부에서 사용할 수 있는 추적 헬퍼 함수들

핵심 특징:
- Easy Integration: 기존 A2A 에이전트에 쉽게 통합 가능한 헬퍼
- Detailed Visibility: 에이전트 내부 로직의 상세한 가시성 제공
- Performance Tracking: 각 단계별 성능 및 처리 시간 추적
- Error Handling: 오류 발생 시에도 안전한 추적 보장
"""

import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
from contextlib import contextmanager

try:
    from .langfuse_session_tracer import get_session_tracer
    TRACER_AVAILABLE = True
except ImportError:
    TRACER_AVAILABLE = False

def trace_agent_operation(operation_name: str, agent_name: str = None):
    """
    A2A 에이전트 내부 작업을 추적하는 데코레이터
    
    사용 예:
    @trace_agent_operation("data_loading", "📁 Data Loader")
    def load_data(self, file_path):
        # 실제 데이터 로딩 로직
        return data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not TRACER_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_session_tracer()
            start_time = time.time()
            
            # 에이전트 이름 추출 (첫 번째 인자가 self인 경우)
            actual_agent_name = agent_name
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                if not actual_agent_name:
                    actual_agent_name = class_name
            
            # 입력 파라미터 처리
            input_data = {
                "args_count": len(args),
                "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
                "function_name": func.__name__
            }
            
            # 내부 로직 추적 시작
            operation_context = tracer.trace_agent_internal_logic(
                agent_name=actual_agent_name or "Unknown Agent",
                operation=operation_name,
                input_data=input_data,
                operation_metadata={
                    "function": func.__name__,
                    "module": func.__module__
                }
            )
            
            try:
                # 실제 함수 실행
                result = func(*args, **kwargs)
                
                # 결과 기록
                tracer.record_internal_operation_result(
                    operation_context=operation_context,
                    result=result,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 오류 기록
                tracer.record_internal_operation_result(
                    operation_context=operation_context,
                    result={"error": str(e), "type": type(e).__name__},
                    success=False
                )
                raise
                
        return wrapper
    return decorator

@contextmanager
def trace_data_analysis(agent_name: str, analysis_type: str, data_info: Dict[str, Any] = None):
    """
    데이터 분석 과정을 추적하는 컨텍스트 매니저
    
    사용 예:
    with trace_data_analysis("🔍 EDA Tools", "correlation_analysis", {"rows": len(df), "cols": len(df.columns)}):
        correlation_matrix = df.corr()
        # ... 분석 로직 ...
        yield {"correlation_matrix": correlation_matrix}
    """
    if not TRACER_AVAILABLE:
        yield None
        return
    
    tracer = get_session_tracer()
    start_time = time.time()
    
    # 분석 시작 추적
    operation_context = tracer.trace_agent_internal_logic(
        agent_name=agent_name,
        operation=f"data_analysis_{analysis_type}",
        input_data=data_info or {},
        operation_metadata={
            "analysis_type": analysis_type,
            "start_time": time.time()
        }
    )
    
    try:
        yield operation_context
    except Exception as e:
        # 분석 실패 기록
        tracer.record_internal_operation_result(
            operation_context=operation_context,
            result={"error": str(e), "analysis_failed": True},
            success=False
        )
        raise

def record_data_analysis_result(operation_context: Dict[str, Any], 
                              analysis_result: Dict[str, Any],
                              data_summary: Dict[str, Any] = None):
    """
    데이터 분석 결과를 기록
    
    Args:
        operation_context: trace_data_analysis에서 반환된 컨텍스트
        analysis_result: 분석 결과
        data_summary: 데이터 요약 정보
    """
    if not TRACER_AVAILABLE or not operation_context:
        return
    
    tracer = get_session_tracer()
    
    # 결과 데이터 처리
    processed_result = {
        "analysis_result": analysis_result,
        "data_summary": data_summary or {},
        "processing_time": time.time() - operation_context.get("start_time", time.time())
    }
    
    tracer.record_internal_operation_result(
        operation_context=operation_context,
        result=processed_result,
        success=True
    )

def trace_ml_model_training(agent_name: str, model_type: str, 
                           training_data_info: Dict[str, Any]):
    """
    머신러닝 모델 훈련 과정 추적
    
    사용 예:
    with trace_ml_model_training("🤖 H2O ML", "AutoML", {"rows": 1000, "features": 20}):
        model = h2o.automl.H2OAutoML()
        model.train(...)
        yield {"model_id": model.model_id, "accuracy": 0.95}
    """
    return trace_data_analysis(agent_name, f"ml_training_{model_type}", training_data_info)

def trace_data_visualization(agent_name: str, chart_type: str, 
                           data_info: Dict[str, Any] = None):
    """
    데이터 시각화 과정 추적
    
    사용 예:
    with trace_data_visualization("📊 Data Visualization", "scatter_plot", {"data_points": 500}):
        fig = px.scatter(df, x='x', y='y')
        yield {"chart_created": True, "chart_type": "scatter"}
    """
    return trace_data_analysis(agent_name, f"visualization_{chart_type}", data_info)

def trace_dataframe_operation(operation_name: str):
    """
    DataFrame 조작 작업 추적 데코레이터
    
    사용 예:
    @trace_dataframe_operation("remove_outliers")
    def remove_outliers(self, df, column):
        # 이상치 제거 로직
        return cleaned_df
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not TRACER_AVAILABLE:
                return func(*args, **kwargs)
            
            # DataFrame 정보 추출
            df_info = {}
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    df_info[f"df_{i}"] = {
                        "shape": arg.shape,
                        "memory_mb": round(arg.memory_usage(deep=True).sum() / 1024**2, 2),
                        "columns": list(arg.columns[:10])  # 처음 10개 컬럼만
                    }
            
            tracer = get_session_tracer()
            
            # DataFrame 작업 추적
            operation_context = tracer.trace_agent_internal_logic(
                agent_name="DataFrame Operation",
                operation=operation_name,
                input_data=df_info,
                operation_metadata={
                    "function": func.__name__,
                    "operation_type": operation_name
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # 결과 DataFrame 정보
                result_info = {}
                if isinstance(result, pd.DataFrame):
                    result_info = {
                        "result_shape": result.shape,
                        "result_memory_mb": round(result.memory_usage(deep=True).sum() / 1024**2, 2),
                        "shape_change": (result.shape[0] - df_info.get('df_0', {}).get('shape', [0])[0], 
                                       result.shape[1] - df_info.get('df_0', {}).get('shape', [0, 0])[1])
                    }
                
                tracer.record_internal_operation_result(
                    operation_context=operation_context,
                    result=result_info,
                    success=True
                )
                
                return result
                
            except Exception as e:
                tracer.record_internal_operation_result(
                    operation_context=operation_context,
                    result={"error": str(e), "operation_failed": True},
                    success=False
                )
                raise
                
        return wrapper
    return decorator

def trace_sql_query(agent_name: str, query_type: str = "unknown"):
    """
    SQL 쿼리 실행 추적
    
    사용 예:
    with trace_sql_query("🗄️ SQL Database", "SELECT") as context:
        result = engine.execute(query)
        record_sql_result(context, {"rows_returned": len(result), "execution_time": 0.1})
    """
    return trace_data_analysis(agent_name, f"sql_query_{query_type}")

def record_sql_result(operation_context: Dict[str, Any], query_result: Dict[str, Any]):
    """SQL 쿼리 결과 기록"""
    record_data_analysis_result(operation_context, query_result)

def trace_feature_engineering(agent_name: str, feature_type: str):
    """
    피처 엔지니어링 과정 추적
    
    사용 예:
    with trace_feature_engineering("⚙️ Feature Engineering", "polynomial_features"):
        new_features = create_polynomial_features(df)
        record_feature_result(context, {"new_features_count": len(new_features.columns)})
    """
    return trace_data_analysis(agent_name, f"feature_engineering_{feature_type}")

def record_feature_result(operation_context: Dict[str, Any], feature_result: Dict[str, Any]):
    """피처 엔지니어링 결과 기록"""
    record_data_analysis_result(operation_context, feature_result)

# A2A 에이전트용 편의 함수들
class A2AAgentTracer:
    """A2A 에이전트를 위한 통합 추적 클래스"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.tracer = get_session_tracer() if TRACER_AVAILABLE else None
    
    def start_task(self, task_description: str, task_metadata: Dict[str, Any] = None):
        """에이전트 작업 시작"""
        if self.tracer:
            return self.tracer.trace_agent_execution(
                agent_name=self.agent_name,
                task_description=task_description,
                agent_metadata=task_metadata
            )
        else:
            return None
    
    def log_step(self, step_name: str, input_data: Any = None, metadata: Dict[str, Any] = None):
        """작업 단계 로깅"""
        if self.tracer:
            return self.tracer.trace_agent_internal_logic(
                agent_name=self.agent_name,
                operation=step_name,
                input_data=input_data or {},
                operation_metadata=metadata
            )
        return {"enabled": False}
    
    def record_result(self, result: Dict[str, Any], confidence: float = 0.8, 
                     artifacts: List[Dict] = None):
        """에이전트 결과 기록"""
        if self.tracer:
            self.tracer.record_agent_result(
                agent_name=self.agent_name,
                result=result,
                confidence=confidence,
                artifacts=artifacts
            )

# 전역 헬퍼 함수
def create_agent_tracer(agent_name: str) -> A2AAgentTracer:
    """에이전트별 추적기 생성"""
    return A2AAgentTracer(agent_name)

def is_tracing_enabled() -> bool:
    """추적 활성화 여부 확인"""
    return TRACER_AVAILABLE and get_session_tracer().enabled 