"""
Advanced Code Generation and Execution Tracking System

고급 코드 생성 및 실행 추적 시스템
- 에이전트 생성 코드 추적
- 코드 실행 모니터링
- 코드 품질 분석
- 실행 결과 추적
- 성능 메트릭 수집
- 코드 버전 관리
- 보안 및 안전성 검사

Author: CherryAI Team
Date: 2024-12-30
"""

import ast
import sys
import traceback
import time
import threading
import subprocess
import tempfile
import os
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Optional imports for enhanced functionality
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Our imports
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    from core.auto_data_profiler import get_auto_data_profiler
    CORE_SYSTEMS_AVAILABLE = True
except ImportError:
    CORE_SYSTEMS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeType(Enum):
    """코드 유형 분류"""
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    MACHINE_LEARNING = "machine_learning"
    DATA_PROCESSING = "data_processing"
    UTILITY = "utility"
    STATISTICAL = "statistical"
    DATABASE = "database"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    UNKNOWN = "unknown"


class ExecutionStatus(Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CodeQuality(Enum):
    """코드 품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DANGEROUS = "dangerous"


@dataclass
class CodeMetrics:
    """코드 메트릭스"""
    lines_of_code: int
    complexity_score: float
    readability_score: float
    security_score: float
    performance_score: float
    
    # 상세 메트릭
    function_count: int = 0
    import_count: int = 0
    comment_ratio: float = 0.0
    
    # 잠재적 이슈
    potential_issues: List[str] = None
    
    # 사용된 라이브러리
    libraries_used: List[str] = None


@dataclass
class ExecutionResult:
    """코드 실행 결과"""
    status: ExecutionStatus
    execution_time: float
    memory_usage: Optional[float] = None
    
    # 출력 정보
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    
    # 생성된 파일들
    generated_files: List[str] = None
    
    # 오류 정보
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_info: Optional[str] = None
    
    # 리소스 사용량
    cpu_usage: Optional[float] = None
    peak_memory: Optional[float] = None


@dataclass
class CodeExecution:
    """코드 실행 추적 정보"""
    execution_id: str
    code_id: str
    agent_id: str
    session_id: str
    
    # 코드 정보
    source_code: str
    code_type: CodeType
    language: str = "python"
    
    # 실행 환경
    execution_context: Dict[str, Any] = None
    input_variables: Dict[str, Any] = None
    
    # 메트릭스 및 결과
    code_metrics: CodeMetrics = None
    execution_result: ExecutionResult = None
    
    # 메타데이터
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # 태그 및 분류
    tags: List[str] = None
    priority: int = 5  # 1-10


class SecurityChecker:
    """코드 보안 검사기"""
    
    DANGEROUS_IMPORTS = {
        'os', 'subprocess', 'eval', 'exec', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input', 'sys'
    }
    
    DANGEROUS_FUNCTIONS = {
        'exec', 'eval', 'compile', '__import__', 'getattr', 'setattr',
        'delattr', 'hasattr', 'globals', 'locals', 'vars'
    }
    
    DANGEROUS_KEYWORDS = {
        'import os', 'import sys', 'import subprocess', 'from os',
        'from sys', 'from subprocess', '__file__', '__name__'
    }
    
    @classmethod
    def analyze_code_security(cls, code: str) -> Tuple[float, List[str]]:
        """코드 보안성 분석"""
        issues = []
        score = 100.0
        
        # 위험한 import 검사
        for dangerous_import in cls.DANGEROUS_IMPORTS:
            if f'import {dangerous_import}' in code or f'from {dangerous_import}' in code:
                issues.append(f"위험한 import 발견: {dangerous_import}")
                score -= 20
        
        # 위험한 함수 호출 검사
        for dangerous_func in cls.DANGEROUS_FUNCTIONS:
            if f'{dangerous_func}(' in code:
                issues.append(f"위험한 함수 호출: {dangerous_func}")
                score -= 15
        
        # 위험한 키워드 검사
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in code:
                issues.append(f"위험한 키워드: {keyword}")
                score -= 10
        
        # 파일 시스템 접근 검사
        if any(pattern in code for pattern in ['open(', 'file(', '.write(', '.read(']):
            issues.append("파일 시스템 접근 감지")
            score -= 5
        
        # 네트워크 접근 검사
        if any(pattern in code for pattern in ['urllib', 'requests', 'http', 'socket']):
            issues.append("네트워크 접근 감지")
            score -= 5
        
        return max(0, score) / 100.0, issues


class CodeAnalyzer:
    """코드 분석기"""
    
    @staticmethod
    def analyze_complexity(code: str) -> float:
        """코드 복잡도 분석 (순환 복잡도 기반)"""
        try:
            tree = ast.parse(code)
            complexity = 1  # 기본 복잡도
            
            for node in ast.walk(tree):
                # 조건문, 반복문, 예외처리 등이 복잡도 증가
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, (ast.And, ast.Or)):
                    complexity += 1
            
            # 정규화 (1-10 스케일)
            return min(10.0, max(1.0, complexity / 3.0))
        
        except Exception:
            return 5.0  # 파싱 실패 시 중간값
    
    @staticmethod
    def analyze_readability(code: str) -> float:
        """코드 가독성 분석"""
        try:
            lines = code.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            if not non_empty_lines:
                return 0.0
            
            score = 100.0
            
            # 주석 비율
            comment_lines = [line for line in non_empty_lines if line.strip().startswith('#')]
            comment_ratio = len(comment_lines) / len(non_empty_lines)
            
            if comment_ratio < 0.1:
                score -= 20
            elif comment_ratio > 0.3:
                score += 10
            
            # 라인 길이 검사
            long_lines = [line for line in non_empty_lines if len(line) > 100]
            if long_lines:
                score -= min(30, len(long_lines) * 5)
            
            # 들여쓰기 일관성
            indentation_levels = []
            for line in non_empty_lines:
                if line.strip():
                    indentation = len(line) - len(line.lstrip())
                    if indentation > 0:
                        indentation_levels.append(indentation)
            
            if indentation_levels:
                # 들여쓰기가 4의 배수가 아니면 감점
                irregular_indent = [level for level in indentation_levels if level % 4 != 0]
                if irregular_indent:
                    score -= 10
            
            # 함수/클래스 정의 비율
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            if len(functions) + len(classes) > 0:
                score += 10  # 구조화된 코드 가산점
            
            return max(0, min(100, score)) / 100.0
        
        except Exception:
            return 0.5  # 파싱 실패 시 중간값
    
    @staticmethod
    def detect_code_type(code: str) -> CodeType:
        """코드 유형 탐지"""
        code_lower = code.lower()
        
        # 시각화 관련
        if any(keyword in code_lower for keyword in ['plot', 'chart', 'graph', 'matplotlib', 'seaborn', 'plotly']):
            return CodeType.VISUALIZATION
        
        # 머신러닝 관련
        if any(keyword in code_lower for keyword in ['sklearn', 'tensorflow', 'keras', 'model', 'train', 'predict']):
            return CodeType.MACHINE_LEARNING
        
        # 데이터 처리 관련
        if any(keyword in code_lower for keyword in ['pandas', 'dataframe', 'groupby', 'merge', 'join']):
            return CodeType.DATA_PROCESSING
        
        # 통계 분석 관련
        if any(keyword in code_lower for keyword in ['mean', 'std', 'corr', 'describe', 'statistical']):
            return CodeType.STATISTICAL
        
        # 데이터베이스 관련
        if any(keyword in code_lower for keyword in ['sql', 'database', 'query', 'select', 'insert', 'update']):
            return CodeType.DATABASE
        
        # 파일 작업 관련
        if any(keyword in code_lower for keyword in ['open', 'read', 'write', 'file', 'csv', 'json']):
            return CodeType.FILE_OPERATION
        
        # API 호출 관련
        if any(keyword in code_lower for keyword in ['requests', 'http', 'api', 'get', 'post']):
            return CodeType.API_CALL
        
        # 일반 데이터 분석
        if any(keyword in code_lower for keyword in ['data', 'analysis', 'numpy', 'array']):
            return CodeType.DATA_ANALYSIS
        
        return CodeType.UNKNOWN
    
    @staticmethod
    def extract_libraries(code: str) -> List[str]:
        """사용된 라이브러리 추출"""
        libraries = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        libraries.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        libraries.append(node.module.split('.')[0])
        
        except Exception:
            # 정규식 백업 방법
            import re
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, code)
                libraries.extend(matches)
        
        return list(set(libraries))  # 중복 제거


class SafeExecutor:
    """안전한 코드 실행기"""
    
    def __init__(self, timeout: int = 30, memory_limit: int = 100):  # MB
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'sklearn', 'scipy', 'statsmodels', 'math', 'statistics',
            'datetime', 'json', 're', 'collections', 'itertools'
        }
    
    @contextmanager
    def capture_output(self):
        """출력 캡처 컨텍스트"""
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                yield stdout_capture, stderr_capture
        finally:
            pass
    
    def execute_code(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """안전한 코드 실행"""
        start_time = time.time()
        
        try:
            # 보안 검사
            security_score, security_issues = SecurityChecker.analyze_code_security(code)
            if security_score < 0.5:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    execution_time=0.0,
                    error_type="SecurityError",
                    error_message=f"보안 위험 코드: {'; '.join(security_issues)}",
                    stderr=f"Security issues: {security_issues}"
                )
            
            # 실행 환경 설정
            exec_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed
                }
            }
            
            # 허용된 모듈 추가
            if PANDAS_AVAILABLE:
                exec_globals['pd'] = pd
                exec_globals['pandas'] = pd
            if NUMPY_AVAILABLE:
                exec_globals['np'] = np
                exec_globals['numpy'] = np
            if MATPLOTLIB_AVAILABLE:
                exec_globals['plt'] = plt
                exec_globals['matplotlib'] = matplotlib
            
            # 컨텍스트 변수 추가
            if context:
                exec_globals.update(context)
            if variables:
                exec_globals.update(variables)
            
            # 출력 캡처하며 실행
            with self.capture_output() as (stdout_capture, stderr_capture):
                try:
                    # 타임아웃 제한으로 실행
                    result = self._execute_with_timeout(code, exec_globals, self.timeout)
                    
                    execution_time = time.time() - start_time
                    
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        execution_time=execution_time,
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        return_value=result
                    )
                
                except TimeoutError:
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        execution_time=self.timeout,
                        error_type="TimeoutError",
                        error_message=f"실행 시간 초과 ({self.timeout}초)",
                        stderr="Execution timed out"
                    )
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    return ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        execution_time=execution_time,
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        traceback_info=traceback.format_exc()
                    )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=execution_time,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_info=traceback.format_exc()
            )
    
    def _execute_with_timeout(self, code: str, globals_dict: dict, timeout: int) -> Any:
        """타임아웃 제한으로 코드 실행"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                # exec 대신 compile + eval 조합 사용
                compiled = compile(code, '<string>', 'exec')
                exec(compiled, globals_dict)
                
                # 마지막 표현식의 결과 반환 시도
                try:
                    lines = code.strip().split('\n')
                    last_line = lines[-1].strip()
                    if last_line and not any(last_line.startswith(kw) for kw in ['if', 'for', 'while', 'def', 'class', 'import', 'from']):
                        if '=' not in last_line or last_line.count('=') == last_line.count('=='):
                            # 표현식일 가능성이 높음
                            compiled_expr = compile(last_line, '<string>', 'eval')
                            result[0] = eval(compiled_expr, globals_dict)
                except:
                    pass  # 표현식이 아니면 무시
                    
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # 스레드가 여전히 실행 중이면 타임아웃
            raise TimeoutError("Code execution timed out")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


class AdvancedCodeTracker:
    """
    고급 코드 생성 및 실행 추적 시스템
    
    에이전트가 생성한 코드를 종합적으로 추적하고 분석
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enhanced_tracer = None
        self.data_profiler = None
        
        # 실행 기록 저장소
        self.executions: Dict[str, CodeExecution] = {}
        self.execution_history: List[str] = []
        
        # 코드 실행기
        self.executor = SafeExecutor(
            timeout=self.config.get('execution_timeout', 30),
            memory_limit=self.config.get('memory_limit', 100)
        )
        
        # 통계 정보
        self.statistics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'code_types': {},
            'agents': {},
            'daily_stats': {}
        }
        
        # 시스템 초기화
        self._initialize_systems()
        
        logger.info("🔧 Advanced Code Tracker 초기화 완료")
    
    def _initialize_systems(self):
        """핵심 시스템 초기화"""
        if not CORE_SYSTEMS_AVAILABLE:
            logger.warning("⚠️ Core systems not available")
            return
        
        try:
            self.enhanced_tracer = get_enhanced_tracer()
            self.data_profiler = get_auto_data_profiler()
            logger.info("✅ Enhanced tracking systems activated")
        except Exception as e:
            logger.warning(f"⚠️ System initialization failed: {e}")
    
    def track_code_generation(
        self,
        agent_id: str,
        session_id: str,
        source_code: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """코드 생성 추적"""
        try:
            # 고유 ID 생성
            code_id = self._generate_code_id(source_code)
            execution_id = f"{code_id}_{int(time.time())}"
            
            # 코드 분석
            code_metrics = self._analyze_code(source_code)
            code_type = CodeAnalyzer.detect_code_type(source_code)
            
            # 실행 추적 객체 생성
            execution = CodeExecution(
                execution_id=execution_id,
                code_id=code_id,
                agent_id=agent_id,
                session_id=session_id,
                source_code=source_code,
                code_type=code_type,
                language="python",
                execution_context=context or {},
                code_metrics=code_metrics,
                created_at=datetime.now().isoformat(),
                tags=tags or []
            )
            
            # 저장
            self.executions[execution_id] = execution
            self.execution_history.append(execution_id)
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "code_generation_tracked",
                    {
                        "execution_id": execution_id,
                        "agent_id": agent_id,
                        "code_type": code_type.value,
                        "complexity": code_metrics.complexity_score,
                        "lines_of_code": code_metrics.lines_of_code
                    },
                    "Code generation tracked successfully"
                )
            
            logger.info(f"📝 코드 생성 추적 완료: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"❌ 코드 생성 추적 실패: {e}")
            return None
    
    def execute_tracked_code(
        self,
        execution_id: str,
        input_variables: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """추적된 코드 실행"""
        try:
            if execution_id not in self.executions:
                raise ValueError(f"실행 ID를 찾을 수 없습니다: {execution_id}")
            
            execution = self.executions[execution_id]
            
            # 실행 시작 기록
            execution.started_at = datetime.now().isoformat()
            execution.input_variables = input_variables or {}
            
            if additional_context:
                execution.execution_context.update(additional_context)
            
            logger.info(f"🚀 코드 실행 시작: {execution_id}")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "code_execution_start",
                    {
                        "execution_id": execution_id,
                        "agent_id": execution.agent_id,
                        "code_type": execution.code_type.value
                    },
                    "Code execution started"
                )
            
            # 코드 실행
            result = self.executor.execute_code(
                execution.source_code,
                context=execution.execution_context,
                variables=input_variables
            )
            
            # 실행 결과 저장
            execution.execution_result = result
            execution.completed_at = datetime.now().isoformat()
            
            # 통계 업데이트
            self._update_statistics(execution, result)
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "code_execution_complete",
                    {
                        "execution_id": execution_id,
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                        "success": result.status == ExecutionStatus.SUCCESS
                    },
                    f"Code execution completed: {result.status.value}"
                )
            
            logger.info(f"✅ 코드 실행 완료: {execution_id} ({result.status.value})")
            return result
            
        except Exception as e:
            logger.error(f"❌ 코드 실행 실패 {execution_id}: {e}")
            
            # 오류 결과 생성
            error_result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=0.0,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_info=traceback.format_exc()
            )
            
            if execution_id in self.executions:
                self.executions[execution_id].execution_result = error_result
                self.executions[execution_id].completed_at = datetime.now().isoformat()
            
            return error_result
    
    def track_and_execute_code(
        self,
        agent_id: str,
        session_id: str,
        source_code: str,
        input_variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[str, ExecutionResult]:
        """코드 추적 및 실행 통합 메서드"""
        execution_id = self.track_code_generation(
            agent_id, session_id, source_code, context, tags
        )
        
        if not execution_id:
            error_result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=0.0,
                error_type="TrackingError",
                error_message="코드 추적 실패"
            )
            return None, error_result
        
        result = self.execute_tracked_code(execution_id, input_variables, context)
        return execution_id, result
    
    def track_and_execute(
        self,
        agent_id: str = None,
        session_id: str = None,
        code: str = None,
        source_code: str = None,
        variables: Optional[Dict[str, Any]] = None,
        input_variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        safe_execution: bool = True,
        **kwargs
    ) -> Tuple[str, ExecutionResult]:
        """편의 메서드: 다양한 매개변수 형태를 지원하는 track_and_execute"""
        # 매개변수 정규화
        final_agent_id = agent_id or "unknown_agent"
        final_session_id = session_id or "unknown_session"
        final_code = code or source_code or ""
        final_variables = variables or input_variables or {}
        
        if not final_code:
            error_result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                execution_time=0.0,
                error_type="ParameterError",
                error_message="코드가 제공되지 않았습니다."
            )
            return None, error_result
        
        return self.track_and_execute_code(
            agent_id=final_agent_id,
            session_id=final_session_id,
            source_code=final_code,
            input_variables=final_variables,
            context=context,
            tags=tags
        )
    
    def _generate_code_id(self, code: str) -> str:
        """코드 해시 기반 ID 생성"""
        code_hash = hashlib.md5(code.encode()).hexdigest()[:12]
        return f"code_{code_hash}"
    
    def _analyze_code(self, code: str) -> CodeMetrics:
        """코드 종합 분석"""
        try:
            # 기본 메트릭
            lines = code.strip().split('\n')
            lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            # 복잡도 분석
            complexity_score = CodeAnalyzer.analyze_complexity(code)
            
            # 가독성 분석
            readability_score = CodeAnalyzer.analyze_readability(code)
            
            # 보안 분석
            security_score, security_issues = SecurityChecker.analyze_code_security(code)
            
            # 성능 점수 (단순 추정)
            performance_score = self._estimate_performance_score(code)
            
            # 상세 분석
            function_count = code.count('def ')
            import_count = code.count('import ') + code.count('from ')
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            comment_ratio = len(comment_lines) / len(lines) if lines else 0
            
            # 사용된 라이브러리
            libraries_used = CodeAnalyzer.extract_libraries(code)
            
            # 잠재적 이슈
            potential_issues = security_issues.copy()
            if complexity_score > 8:
                potential_issues.append("높은 복잡도")
            if readability_score < 0.6:
                potential_issues.append("낮은 가독성")
            if len(lines) > 100:
                potential_issues.append("긴 코드")
            
            return CodeMetrics(
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                readability_score=readability_score,
                security_score=security_score,
                performance_score=performance_score,
                function_count=function_count,
                import_count=import_count,
                comment_ratio=comment_ratio,
                potential_issues=potential_issues,
                libraries_used=libraries_used
            )
            
        except Exception as e:
            logger.warning(f"⚠️ 코드 분석 실패: {e}")
            return CodeMetrics(
                lines_of_code=0,
                complexity_score=5.0,
                readability_score=0.5,
                security_score=0.5,
                performance_score=0.5,
                potential_issues=["분석 실패"]
            )
    
    def _estimate_performance_score(self, code: str) -> float:
        """성능 점수 추정"""
        score = 100.0
        code_lower = code.lower()
        
        # 비효율적인 패턴 검사
        if 'for' in code_lower and 'range(len(' in code_lower:
            score -= 20  # 비효율적인 반복문
        
        if code_lower.count('for') > 3:
            score -= 15  # 중첩 반복문 가능성
        
        if 'sleep(' in code_lower:
            score -= 10  # 블로킹 호출
        
        if any(pattern in code_lower for pattern in ['.append(', '.extend(']):
            if code_lower.count('for') > 1:
                score -= 10  # 반복문 내 리스트 확장
        
        # 효율적인 패턴 가산점
        if any(pattern in code_lower for pattern in ['numpy', 'pandas', 'vectorized']):
            score += 10  # 벡터화 연산
        
        if 'list comprehension' in code_lower or '[' in code and 'for' in code and ']' in code:
            score += 5  # 리스트 컴프리헨션
        
        return max(0, min(100, score)) / 100.0
    
    def _update_statistics(self, execution: CodeExecution, result: ExecutionResult):
        """통계 정보 업데이트"""
        self.statistics['total_executions'] += 1
        
        if result.status == ExecutionStatus.SUCCESS:
            self.statistics['successful_executions'] += 1
        else:
            self.statistics['failed_executions'] += 1
        
        self.statistics['total_execution_time'] += result.execution_time
        
        # 코드 타입별 통계
        code_type = execution.code_type.value
        if code_type not in self.statistics['code_types']:
            self.statistics['code_types'][code_type] = {
                'count': 0, 'success_count': 0, 'total_time': 0.0
            }
        
        self.statistics['code_types'][code_type]['count'] += 1
        if result.status == ExecutionStatus.SUCCESS:
            self.statistics['code_types'][code_type]['success_count'] += 1
        self.statistics['code_types'][code_type]['total_time'] += result.execution_time
        
        # 에이전트별 통계
        agent_id = execution.agent_id
        if agent_id not in self.statistics['agents']:
            self.statistics['agents'][agent_id] = {
                'count': 0, 'success_count': 0, 'total_time': 0.0
            }
        
        self.statistics['agents'][agent_id]['count'] += 1
        if result.status == ExecutionStatus.SUCCESS:
            self.statistics['agents'][agent_id]['success_count'] += 1
        self.statistics['agents'][agent_id]['total_time'] += result.execution_time
        
        # 일일 통계
        today = datetime.now().date().isoformat()
        if today not in self.statistics['daily_stats']:
            self.statistics['daily_stats'][today] = {
                'count': 0, 'success_count': 0, 'total_time': 0.0
            }
        
        self.statistics['daily_stats'][today]['count'] += 1
        if result.status == ExecutionStatus.SUCCESS:
            self.statistics['daily_stats'][today]['success_count'] += 1
        self.statistics['daily_stats'][today]['total_time'] += result.execution_time
    
    def get_execution_info(self, execution_id: str) -> Optional[CodeExecution]:
        """실행 정보 조회"""
        return self.executions.get(execution_id)
    
    def get_execution_history(
        self, 
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        code_type: Optional[CodeType] = None,
        limit: int = 50
    ) -> List[CodeExecution]:
        """실행 히스토리 조회"""
        executions = []
        
        for exec_id in reversed(self.execution_history[-limit:]):
            if exec_id in self.executions:
                execution = self.executions[exec_id]
                
                # 필터 적용
                if agent_id and execution.agent_id != agent_id:
                    continue
                if session_id and execution.session_id != session_id:
                    continue
                if code_type and execution.code_type != code_type:
                    continue
                
                executions.append(execution)
        
        return executions
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 조회"""
        stats = self.statistics.copy()
        
        # 계산된 메트릭 추가
        if stats['total_executions'] > 0:
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_executions']
        else:
            stats['success_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        # 코드 타입별 성공률 계산
        for code_type, type_stats in stats['code_types'].items():
            if type_stats['count'] > 0:
                type_stats['success_rate'] = type_stats['success_count'] / type_stats['count']
                type_stats['average_time'] = type_stats['total_time'] / type_stats['count']
        
        # 에이전트별 성공률 계산
        for agent_id, agent_stats in stats['agents'].items():
            if agent_stats['count'] > 0:
                agent_stats['success_rate'] = agent_stats['success_count'] / agent_stats['count']
                agent_stats['average_time'] = agent_stats['total_time'] / agent_stats['count']
        
        return stats
    
    def analyze_code_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """코드 품질 트렌드 분석"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        recent_executions = []
        for exec_id in self.execution_history:
            if exec_id in self.executions:
                execution = self.executions[exec_id]
                if execution.created_at:
                    created_date = datetime.fromisoformat(execution.created_at)
                    if start_date <= created_date <= end_date:
                        recent_executions.append(execution)
        
        if not recent_executions:
            return {
                'period': f"{start_date.date()} ~ {end_date.date()}",
                'total_executions': 0,
                'trends': {}
            }
        
        # 품질 메트릭 집계
        complexity_scores = []
        readability_scores = []
        security_scores = []
        performance_scores = []
        
        for execution in recent_executions:
            if execution.code_metrics:
                complexity_scores.append(execution.code_metrics.complexity_score)
                readability_scores.append(execution.code_metrics.readability_score)
                security_scores.append(execution.code_metrics.security_score)
                performance_scores.append(execution.code_metrics.performance_score)
        
        trends = {}
        if complexity_scores:
            trends['complexity'] = {
                'average': sum(complexity_scores) / len(complexity_scores),
                'min': min(complexity_scores),
                'max': max(complexity_scores)
            }
        
        if readability_scores:
            trends['readability'] = {
                'average': sum(readability_scores) / len(readability_scores),
                'min': min(readability_scores),
                'max': max(readability_scores)
            }
        
        if security_scores:
            trends['security'] = {
                'average': sum(security_scores) / len(security_scores),
                'min': min(security_scores),
                'max': max(security_scores)
            }
        
        if performance_scores:
            trends['performance'] = {
                'average': sum(performance_scores) / len(performance_scores),
                'min': min(performance_scores),
                'max': max(performance_scores)
            }
        
        return {
            'period': f"{start_date.date()} ~ {end_date.date()}",
            'total_executions': len(recent_executions),
            'trends': trends
        }
    
    def generate_code_report(self, execution_id: str) -> str:
        """코드 실행 보고서 생성"""
        if execution_id not in self.executions:
            return f"실행 ID를 찾을 수 없습니다: {execution_id}"
        
        execution = self.executions[execution_id]
        metrics = execution.code_metrics
        result = execution.execution_result
        
        complexity_str = f"{metrics.complexity_score:.2f}/10" if metrics else 'N/A'
        readability_str = f"{metrics.readability_score:.1%}" if metrics else 'N/A'
        security_str = f"{metrics.security_score:.1%}" if metrics else 'N/A'
        performance_str = f"{metrics.performance_score:.1%}" if metrics else 'N/A'
        exec_time_str = f"{result.execution_time:.2f}초" if result else 'N/A'
        memory_str = f"{result.memory_usage or 'N/A'}MB" if result else 'N/A'
        
        report = f"""# 코드 실행 보고서

## 기본 정보
- **실행 ID**: {execution.execution_id}
- **에이전트**: {execution.agent_id}
- **세션**: {execution.session_id}
- **코드 유형**: {execution.code_type.value}
- **생성 시간**: {execution.created_at}
- **실행 시간**: {execution.started_at}
- **완료 시간**: {execution.completed_at}

## 코드 메트릭스
- **코드 라인 수**: {metrics.lines_of_code if metrics else 'N/A'}
- **복잡도 점수**: {complexity_str}
- **가독성 점수**: {readability_str}
- **보안 점수**: {security_str}
- **성능 점수**: {performance_str}

## 실행 결과
- **상태**: {result.status.value if result else 'N/A'}
- **실행 시간**: {exec_time_str}
- **메모리 사용량**: {memory_str}

"""
        
        if result and result.status == ExecutionStatus.ERROR:
            report += f"""## 오류 정보
- **오류 유형**: {result.error_type}
- **오류 메시지**: {result.error_message}
"""
        
        if metrics and metrics.potential_issues:
            report += f"""## 잠재적 이슈
{chr(10).join(f'- {issue}' for issue in metrics.potential_issues)}
"""
        
        if metrics and metrics.libraries_used:
            report += f"""## 사용된 라이브러리
{chr(10).join(f'- {lib}' for lib in metrics.libraries_used)}
"""
        
        return report
    
    def cleanup_old_executions(self, days: int = 30):
        """오래된 실행 기록 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        to_remove = []
        for exec_id, execution in self.executions.items():
            if execution.created_at:
                created_date = datetime.fromisoformat(execution.created_at)
                if created_date < cutoff_date:
                    to_remove.append(exec_id)
        
        for exec_id in to_remove:
            del self.executions[exec_id]
            if exec_id in self.execution_history:
                self.execution_history.remove(exec_id)
        
        logger.info(f"🧹 {len(to_remove)}개의 오래된 실행 기록 정리 완료")


# 전역 인스턴스
_tracker_instance = None


def get_advanced_code_tracker(config: Optional[Dict] = None) -> AdvancedCodeTracker:
    """Advanced Code Tracker 인스턴스 반환"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = AdvancedCodeTracker(config)
    return _tracker_instance


# 편의 함수들
def track_and_execute(
    agent_id: str,
    session_id: str,
    code: str,
    variables: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> Tuple[str, ExecutionResult]:
    """코드 추적 및 실행 편의 함수"""
    tracker = get_advanced_code_tracker()
    return tracker.track_and_execute_code(agent_id, session_id, code, variables, context, tags)


def get_execution_stats() -> Dict[str, Any]:
    """실행 통계 조회 편의 함수"""
    tracker = get_advanced_code_tracker()
    return tracker.get_execution_statistics()


# CLI 테스트 함수
def test_advanced_code_tracker():
    """Advanced Code Tracker 테스트"""
    print("🔧 Advanced Code Tracker 테스트 시작\n")
    
    tracker = get_advanced_code_tracker()
    
    # 테스트 코드들
    test_codes = [
        {
            "code": """
import pandas as pd
import numpy as np

# 데이터 생성
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# 기본 통계
mean_x = data['x'].mean()
mean_y = data['y'].mean()

print(f"Mean X: {mean_x:.2f}")
print(f"Mean Y: {mean_y:.2f}")

# 상관관계
correlation = data['x'].corr(data['y'])
print(f"Correlation: {correlation:.2f}")
""",
            "agent": "data_analysis_agent",
            "tags": ["statistics", "correlation"]
        },
        {
            "code": """
import matplotlib.pyplot as plt
import numpy as np

# 시각화 테스트
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print("시각화 생성 완료")
""",
            "agent": "visualization_agent",
            "tags": ["matplotlib", "visualization"]
        },
        {
            "code": """
# 간단한 계산
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 피보나치 수열 계산
result = fibonacci(10)
print(f"Fibonacci(10) = {result}")

# 리스트 컴프리헨션
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")
""",
            "agent": "utility_agent",
            "tags": ["algorithm", "fibonacci"]
        },
        {
            "code": """
# 보안 위험 코드 (테스트용)
import os
import subprocess

# 위험한 코드
os.system("echo 'test'")
subprocess.run(["echo", "dangerous"])
""",
            "agent": "dangerous_agent",
            "tags": ["security", "test"]
        }
    ]
    
    print("📝 코드 추적 및 실행 테스트:")
    
    for i, test_case in enumerate(test_codes):
        print(f"\n🔄 테스트 {i+1}: {test_case['agent']}")
        
        execution_id, result = tracker.track_and_execute_code(
            agent_id=test_case['agent'],
            session_id=f"test_session_{i}",
            source_code=test_case['code'],
            tags=test_case['tags']
        )
        
        if execution_id:
            print(f"  ✅ 실행 ID: {execution_id}")
            print(f"  📊 상태: {result.status.value}")
            print(f"  ⏱️ 실행시간: {result.execution_time:.2f}초")
            
            if result.status == ExecutionStatus.ERROR:
                print(f"  ❌ 오류: {result.error_message}")
            
            if result.stdout:
                print(f"  📄 출력: {result.stdout.strip()}")
        else:
            print(f"  ❌ 추적 실패: {result.error_message}")
    
    # 통계 확인
    print(f"\n📊 실행 통계:")
    stats = tracker.get_execution_statistics()
    print(f"  총 실행: {stats['total_executions']}")
    print(f"  성공률: {stats['success_rate']:.1%}")
    print(f"  평균 실행시간: {stats['average_execution_time']:.2f}초")
    
    if stats['code_types']:
        print(f"  코드 타입별:")
        for code_type, type_stats in stats['code_types'].items():
            print(f"    {code_type}: {type_stats['count']}회 (성공률: {type_stats.get('success_rate', 0):.1%})")
    
    # 품질 트렌드 분석
    print(f"\n📈 코드 품질 트렌드 (최근 7일):")
    trends = tracker.analyze_code_quality_trends(7)
    print(f"  분석 기간: {trends['period']}")
    print(f"  총 실행: {trends['total_executions']}")
    
    for metric, values in trends['trends'].items():
        print(f"  {metric}: 평균 {values['average']:.2f} (범위: {values['min']:.2f}-{values['max']:.2f})")
    
    # 실행 히스토리
    print(f"\n📋 최근 실행 히스토리:")
    history = tracker.get_execution_history(limit=3)
    for execution in history[:3]:
        print(f"  {execution.execution_id}: {execution.agent_id} ({execution.code_type.value})")
    
    print(f"\n✅ Advanced Code Tracker 테스트 완료!")


if __name__ == "__main__":
    test_advanced_code_tracker() 