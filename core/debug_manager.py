# File: core/debug_manager.py
# Location: ./core/debug_manager.py

import os
import traceback
import inspect
import time
import threading
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

class DebugManager:
    """통합 디버깅 관리자"""
    
    def __init__(self):
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.debug_logs: List[Dict] = []
        self.performance_metrics: Dict[str, List] = {}
        self.error_history: List[Dict] = []
        self.function_call_stack: List[str] = []
        self.max_logs = 1000  # 메모리 관리를 위한 최대 로그 수
        
    def log_debug(self, message: str, category: str = "GENERAL", data: dict = None):
        """디버그 로그 기록"""
        timestamp = datetime.now().isoformat()
        caller_info = self._get_caller_info()
        
        debug_entry = {
            "timestamp": timestamp,
            "category": category,
            "message": message,
            "caller": caller_info,
            "data": data or {},
            "thread_id": threading.current_thread().name
        }
        
        self.debug_logs.append(debug_entry)
        
        # 메모리 관리 - 최대 로그 수 제한
        if len(self.debug_logs) > self.max_logs:
            self.debug_logs = self.debug_logs[-self.max_logs:]
        
        if self.debug_mode:
            print(f"🐛 DEBUG [{category}] {timestamp} | {caller_info['function']}:{caller_info['line']} | {message}")
            if data:
                print(f"    📊 Data: {data}")
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """성능 메트릭 기록"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "details": details or {}
        }
        
        self.performance_metrics[operation].append(metric_entry)
        
        # 메모리 관리 - 작업당 최대 100개 메트릭
        if len(self.performance_metrics[operation]) > 100:
            self.performance_metrics[operation] = self.performance_metrics[operation][-100:]
        
        if self.debug_mode:
            print(f"⏱️ PERF [{operation}] {duration:.3f}s | {details}")
    
    def log_error(self, error: Exception, context: str = "", extra_data: dict = None):
        """에러 상세 기록"""
        timestamp = datetime.now().isoformat()
        caller_info = self._get_caller_info()
        error_trace = traceback.format_exc()
        
        error_entry = {
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "caller": caller_info,
            "traceback": error_trace,
            "extra_data": extra_data or {},
            "thread_id": threading.current_thread().name
        }
        
        self.error_history.append(error_entry)
        
        # 메모리 관리 - 최대 100개 에러
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        print(f"❌ ERROR [{context}] {timestamp} | {caller_info['function']}:{caller_info['line']}")
        print(f"    💥 {type(error).__name__}: {str(error)}")
        if self.debug_mode:
            print(f"    📍 Traceback:\n{error_trace}")
            if extra_data:
                print(f"    📊 Extra Data: {extra_data}")
    
    def _get_caller_info(self):
        """호출자 정보 추출"""
        frame = inspect.currentframe()
        try:
            # 2단계 위로 올라가서 실제 호출자 찾기
            caller_frame = frame.f_back.f_back
            if caller_frame:
                return {
                    "function": caller_frame.f_code.co_name,
                    "filename": caller_frame.f_code.co_filename.split('/')[-1],
                    "line": caller_frame.f_lineno
                }
            else:
                return {"function": "unknown", "filename": "unknown", "line": 0}
        except:
            return {"function": "unknown", "filename": "unknown", "line": 0}
        finally:
            del frame
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """디버그 정보 요약"""
        return {
            "debug_mode": self.debug_mode,
            "total_logs": len(self.debug_logs),
            "total_errors": len(self.error_history),
            "performance_operations": len(self.performance_metrics),
            "recent_errors": self.error_history[-5:] if self.error_history else [],
            "function_call_depth": len(self.function_call_stack)
        }
    
    @contextmanager
    def measure_performance(self, operation: str, details: dict = None):
        """성능 측정 컨텍스트 매니저"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_performance(operation, duration, details)
    
    def debug_decorator(self, category: str = "FUNCTION"):
        """디버깅 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                self.log_debug(f"Entering {func_name}", category, 
                             {"args_count": len(args), "kwargs_keys": list(kwargs.keys())})
                
                self.function_call_stack.append(func_name)
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.log_debug(f"Exiting {func_name} successfully", category, 
                                 {"duration": duration, "result_type": type(result).__name__})
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_error(e, f"{func_name} execution", 
                                 {"duration": duration, "args_count": len(args)})
                    raise
                finally:
                    if self.function_call_stack and self.function_call_stack[-1] == func_name:
                        self.function_call_stack.pop()
                        
            return wrapper
        return decorator
    
    def clear_logs(self):
        """모든 로그 초기화"""
        self.debug_logs = []
        self.performance_metrics = {}
        self.error_history = []
        self.function_call_stack = []
        print("🧹 All debug logs cleared")
    
    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """성능 요약 정보"""
        if operation and operation in self.performance_metrics:
            metrics = self.performance_metrics[operation]
            if metrics:
                durations = [m['duration'] for m in metrics]
                return {
                    "operation": operation,
                    "count": len(metrics),
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "total_duration": sum(durations)
                }
        elif operation is None:
            # 전체 요약
            summary = {}
            for op, metrics in self.performance_metrics.items():
                if metrics:
                    durations = [m['duration'] for m in metrics]
                    summary[op] = {
                        "count": len(metrics),
                        "avg_duration": sum(durations) / len(durations)
                    }
            return summary
        
        return {"error": "No metrics found for operation"}

# 전역 디버그 매니저 인스턴스
debug_manager = DebugManager()

# 편의 함수들
def debug_log(message: str, category: str = "GENERAL", data: dict = None):
    """간단한 디버그 로그 함수"""
    debug_manager.log_debug(message, category, data)

def debug_performance(operation: str):
    """성능 측정 컨텍스트 매니저"""
    return debug_manager.measure_performance(operation)

def debug_function(category: str = "FUNCTION"):
    """함수 디버깅 데코레이터"""
    return debug_manager.debug_decorator(category)

# 로깅 설정 함수
def setup_logging():
    """향상된 로깅 설정"""
    import logging
    
    log_level = logging.DEBUG if debug_manager.debug_mode else logging.INFO
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # 파일 핸들러
    file_handler = logging.FileHandler('debug.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # 파일은 항상 DEBUG 레벨
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 특정 로거 레벨 조정 (너무 시끄러운 것들)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return root_logger