"""
Unit Tests for Advanced Code Tracker

고급 코드 생성 및 실행 추적 시스템 단위 테스트
- 코드 분석 기능 검증
- 보안 검사 기능 검증
- 코드 실행 및 추적 검증
- 통계 및 메트릭 검증

Author: CherryAI Team
Date: 2024-12-30
"""

import pytest
import time
import tempfile
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Our imports
from core.advanced_code_tracker import (
    AdvancedCodeTracker,
    SafeExecutor,
    CodeAnalyzer,
    SecurityChecker,
    CodeType,
    ExecutionStatus,
    CodeQuality,
    CodeMetrics,
    ExecutionResult,
    CodeExecution,
    get_advanced_code_tracker,
    track_and_execute,
    get_execution_stats
)


class TestCodeType:
    """Test CodeType enum"""
    
    def test_enum_values(self):
        """Test CodeType enum values"""
        expected_types = [
            "data_analysis", "visualization", "machine_learning",
            "data_processing", "utility", "statistical", "database",
            "file_operation", "api_call", "unknown"
        ]
        
        for type_name in expected_types:
            assert any(ct.value == type_name for ct in CodeType)


class TestExecutionStatus:
    """Test ExecutionStatus enum"""
    
    def test_enum_values(self):
        """Test ExecutionStatus enum values"""
        expected_statuses = [
            "pending", "running", "success", "error", "timeout", "cancelled"
        ]
        
        for status_name in expected_statuses:
            assert any(es.value == status_name for es in ExecutionStatus)


class TestCodeMetrics:
    """Test CodeMetrics dataclass"""
    
    def test_creation(self):
        """Test CodeMetrics creation"""
        metrics = CodeMetrics(
            lines_of_code=50,
            complexity_score=3.5,
            readability_score=0.8,
            security_score=0.9,
            performance_score=0.7,
            function_count=3,
            import_count=2,
            comment_ratio=0.15,
            potential_issues=["complexity"],
            libraries_used=["pandas", "numpy"]
        )
        
        assert metrics.lines_of_code == 50
        assert metrics.complexity_score == 3.5
        assert metrics.readability_score == 0.8
        assert metrics.security_score == 0.9
        assert metrics.performance_score == 0.7
        assert len(metrics.libraries_used) == 2


class TestExecutionResult:
    """Test ExecutionResult dataclass"""
    
    def test_creation_success(self):
        """Test ExecutionResult creation for successful execution"""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            execution_time=2.5,
            memory_usage=64.0,
            stdout="Output text",
            stderr="",
            return_value=42
        )
        
        assert result.status == ExecutionStatus.SUCCESS
        assert result.execution_time == 2.5
        assert result.memory_usage == 64.0
        assert result.stdout == "Output text"
        assert result.return_value == 42
    
    def test_creation_error(self):
        """Test ExecutionResult creation for error execution"""
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            execution_time=1.0,
            error_type="ValueError",
            error_message="Invalid input",
            traceback_info="Traceback information"
        )
        
        assert result.status == ExecutionStatus.ERROR
        assert result.error_type == "ValueError"
        assert result.error_message == "Invalid input"
        assert result.traceback_info == "Traceback information"


class TestSecurityChecker:
    """Test SecurityChecker class"""
    
    def test_safe_code(self):
        """Test security analysis of safe code"""
        safe_code = """
import pandas as pd
import numpy as np

data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
result = data.mean()
print(result)
"""
        
        score, issues = SecurityChecker.analyze_code_security(safe_code)
        
        assert score >= 0.5  # Should be reasonably safe
        assert len(issues) <= 2  # May have minor issues
    
    def test_dangerous_imports(self):
        """Test detection of dangerous imports"""
        dangerous_code = """
import os
import subprocess
import sys

os.system("rm -rf /")
subprocess.call(["dangerous_command"])
"""
        
        score, issues = SecurityChecker.analyze_code_security(dangerous_code)
        
        assert score < 0.5  # Should be marked as dangerous
        assert len(issues) > 0
        assert any("import" in issue for issue in issues)
    
    def test_dangerous_functions(self):
        """Test detection of dangerous functions"""
        dangerous_code = """
user_input = "malicious_code"
exec(user_input)
eval("dangerous_expression")
"""
        
        score, issues = SecurityChecker.analyze_code_security(dangerous_code)
        
        assert score < 0.8  # Should be penalized
        assert any("함수" in issue for issue in issues)
    
    def test_file_system_access(self):
        """Test detection of file system access"""
        file_code = """
with open('file.txt', 'w') as f:
    f.write('data')

data = open('another_file.txt').read()
"""
        
        score, issues = SecurityChecker.analyze_code_security(file_code)
        
        assert any("파일" in issue for issue in issues)
    
    def test_network_access(self):
        """Test detection of network access"""
        network_code = """
import requests
import urllib

response = requests.get('http://example.com')
data = urllib.request.urlopen('http://malicious.com')
"""
        
        score, issues = SecurityChecker.analyze_code_security(network_code)
        
        assert any("네트워크" in issue for issue in issues)


class TestCodeAnalyzer:
    """Test CodeAnalyzer class"""
    
    def test_complexity_simple(self):
        """Test complexity analysis of simple code"""
        simple_code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
        
        complexity = CodeAnalyzer.analyze_complexity(simple_code)
        
        assert 1.0 <= complexity <= 3.0  # Should be low complexity
    
    def test_complexity_complex(self):
        """Test complexity analysis of complex code"""
        complex_code = """
def complex_function(data):
    if data is None:
        return None
    
    result = []
    for item in data:
        if item > 0:
            for i in range(item):
                if i % 2 == 0:
                    result.append(i)
                else:
                    try:
                        result.append(i * 2)
                    except Exception:
                        continue
    
    return result if result else None
"""
        
        complexity = CodeAnalyzer.analyze_complexity(complex_code)
        
        assert complexity > 3.0  # Should be high complexity
    
    def test_readability_good(self):
        """Test readability analysis of well-written code"""
        readable_code = """
# This function calculates the mean of a list
def calculate_mean(numbers):
    '''Calculate the arithmetic mean of a list of numbers'''
    if not numbers:
        return 0
    
    # Sum all numbers and divide by count
    total = sum(numbers)
    count = len(numbers)
    
    return total / count

# Example usage
data = [1, 2, 3, 4, 5]
mean_value = calculate_mean(data)
print(f"Mean: {mean_value}")
"""
        
        readability = CodeAnalyzer.analyze_readability(readable_code)
        
        assert readability >= 0.7  # Should be highly readable
    
    def test_readability_poor(self):
        """Test readability analysis of poorly written code"""
        unreadable_code = """
def f(x):return sum([i*2for i in x if i>0])if x else 0
y=[f([1,2,3]),f([4,5,6]),f([7,8,9])]
print(y)
z=lambda a,b,c:a+b+c if a and b and c else 0
"""
        
        readability = CodeAnalyzer.analyze_readability(unreadable_code)
        
        assert readability < 0.5  # Should be poorly readable
    
    def test_detect_code_type_visualization(self):
        """Test code type detection for visualization"""
        viz_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
"""
        
        code_type = CodeAnalyzer.detect_code_type(viz_code)
        
        assert code_type == CodeType.VISUALIZATION
    
    def test_detect_code_type_data_processing(self):
        """Test code type detection for data processing"""
        data_code = """
import pandas as pd

df = pd.read_csv('data.csv')
df_grouped = df.groupby('category').mean()
df_merged = df.merge(df_grouped, on='category')
"""
        
        code_type = CodeAnalyzer.detect_code_type(data_code)
        
        assert code_type == CodeType.DATA_PROCESSING
    
    def test_detect_code_type_machine_learning(self):
        """Test code type detection for machine learning"""
        ml_code = """
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
"""
        
        code_type = CodeAnalyzer.detect_code_type(ml_code)
        
        assert code_type == CodeType.MACHINE_LEARNING
    
    def test_extract_libraries(self):
        """Test library extraction"""
        code_with_imports = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import json
"""
        
        libraries = CodeAnalyzer.extract_libraries(code_with_imports)
        
        expected_libs = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'json']
        for lib in expected_libs:
            assert lib in libraries


class TestSafeExecutor:
    """Test SafeExecutor class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.executor = SafeExecutor(timeout=5, memory_limit=50)
    
    def test_execute_simple_code(self):
        """Test execution of simple code"""
        simple_code = """
result = 2 + 3
print(f"Result: {result}")
"""
        
        result = self.executor.execute_code(simple_code)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "Result: 5" in result.stdout
        assert result.execution_time >= 0
    
    def test_execute_with_return_value(self):
        """Test execution with return value"""
        code_with_return = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(8)
"""
        
        result = self.executor.execute_code(code_with_return)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert result.return_value == 21  # fibonacci(8) = 21
    
    def test_execute_with_variables(self):
        """Test execution with input variables"""
        code = """
result = x * y + z
print(f"Calculation: {x} * {y} + {z} = {result}")
"""
        
        variables = {'x': 5, 'y': 3, 'z': 2}
        result = self.executor.execute_code(code, variables=variables)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "Calculation: 5 * 3 + 2 = 17" in result.stdout
    
    def test_execute_with_context(self):
        """Test execution with context"""
        code = """
import pandas as pd
df = pd.DataFrame(data)
result = len(df)
print(f"DataFrame length: {result}")
"""
        
        context = {'data': {'a': [1, 2, 3], 'b': [4, 5, 6]}}
        result = self.executor.execute_code(code, context=context)
        
        # May succeed or fail depending on pandas availability
        if result.status == ExecutionStatus.SUCCESS:
            assert "DataFrame length: 3" in result.stdout
    
    def test_execute_syntax_error(self):
        """Test execution of code with syntax error"""
        invalid_code = """
def broken_function(
    print("Missing closing parenthesis")
"""
        
        result = self.executor.execute_code(invalid_code)
        
        assert result.status == ExecutionStatus.ERROR
        assert result.error_type == "SyntaxError"
    
    def test_execute_runtime_error(self):
        """Test execution of code with runtime error"""
        error_code = """
x = 10
y = 0
result = x / y  # Division by zero
"""
        
        result = self.executor.execute_code(error_code)
        
        assert result.status == ExecutionStatus.ERROR
        assert result.error_type == "ZeroDivisionError"
    
    def test_execute_dangerous_code(self):
        """Test execution of dangerous code"""
        dangerous_code = """
import os
os.system("echo 'dangerous'")
"""
        
        result = self.executor.execute_code(dangerous_code)
        
        assert result.status == ExecutionStatus.ERROR
        assert result.error_type == "SecurityError"
        assert "보안 위험" in result.error_message
    
    def test_execute_infinite_loop_timeout(self):
        """Test timeout handling for infinite loop"""
        infinite_code = """
while True:
    pass
"""
        
        # Use short timeout for test
        executor = SafeExecutor(timeout=1)
        result = executor.execute_code(infinite_code)
        
        assert result.status == ExecutionStatus.TIMEOUT
        assert result.execution_time >= 1.0


class TestAdvancedCodeTracker:
    """Test AdvancedCodeTracker class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.tracker = AdvancedCodeTracker()
    
    def test_initialization(self):
        """Test tracker initialization"""
        assert isinstance(self.tracker.executions, dict)
        assert isinstance(self.tracker.execution_history, list)
        assert isinstance(self.tracker.statistics, dict)
        assert self.tracker.executor is not None
    
    def test_track_code_generation(self):
        """Test code generation tracking"""
        code = """
def hello():
    print("Hello, World!")

hello()
"""
        
        execution_id = self.tracker.track_code_generation(
            agent_id="test_agent",
            session_id="test_session",
            source_code=code,
            tags=["test", "hello_world"]
        )
        
        assert execution_id is not None
        assert execution_id in self.tracker.executions
        
        execution = self.tracker.executions[execution_id]
        assert execution.agent_id == "test_agent"
        assert execution.session_id == "test_session"
        assert execution.source_code == code
        assert "test" in execution.tags
        assert execution.code_metrics is not None
    
    def test_execute_tracked_code(self):
        """Test execution of tracked code"""
        code = """
x = 5
y = 3
result = x + y
print(f"Sum: {result}")
"""
        
        # First track the code
        execution_id = self.tracker.track_code_generation(
            agent_id="test_agent",
            session_id="test_session",
            source_code=code
        )
        
        # Then execute it
        result = self.tracker.execute_tracked_code(execution_id)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "Sum: 8" in result.stdout
        
        # Check that execution was recorded
        execution = self.tracker.executions[execution_id]
        assert execution.execution_result is not None
        assert execution.started_at is not None
        assert execution.completed_at is not None
    
    def test_track_and_execute_code(self):
        """Test combined tracking and execution"""
        code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
print(f"Average: {average}")
"""
        
        execution_id, result = self.tracker.track_and_execute_code(
            agent_id="math_agent",
            session_id="math_session",
            source_code=code,
            tags=["math", "statistics"]
        )
        
        assert execution_id is not None
        assert result.status == ExecutionStatus.SUCCESS
        assert "Average: 3.0" in result.stdout
        
        # Verify tracking
        execution = self.tracker.executions[execution_id]
        assert execution.agent_id == "math_agent"
        assert execution.session_id == "math_session"
        assert "math" in execution.tags
    
    def test_track_code_with_variables(self):
        """Test tracking and execution with input variables"""
        code = """
total = sum(data)
count = len(data)
average = total / count if count > 0 else 0
print(f"Data: {data}")
print(f"Average: {average}")
"""
        
        variables = {"data": [10, 20, 30, 40, 50]}
        
        execution_id, result = self.tracker.track_and_execute_code(
            agent_id="data_agent",
            session_id="data_session",
            source_code=code,
            input_variables=variables
        )
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "Average: 30.0" in result.stdout
    
    def test_get_execution_info(self):
        """Test execution info retrieval"""
        code = "print('test')"
        
        execution_id = self.tracker.track_code_generation(
            agent_id="info_agent",
            session_id="info_session",
            source_code=code
        )
        
        info = self.tracker.get_execution_info(execution_id)
        
        assert info is not None
        assert info.execution_id == execution_id
        assert info.agent_id == "info_agent"
        assert info.session_id == "info_session"
    
    def test_get_execution_info_nonexistent(self):
        """Test execution info retrieval for non-existent ID"""
        info = self.tracker.get_execution_info("nonexistent_id")
        
        assert info is None
    
    def test_get_execution_history(self):
        """Test execution history retrieval"""
        # Execute multiple codes
        codes = [
            "print('first')",
            "print('second')",
            "print('third')"
        ]
        
        for i, code in enumerate(codes):
            self.tracker.track_and_execute_code(
                agent_id=f"agent_{i}",
                session_id=f"session_{i}",
                source_code=code
            )
        
        # Get history
        history = self.tracker.get_execution_history(limit=5)
        
        assert len(history) >= 3
        assert all(isinstance(exec, CodeExecution) for exec in history)
    
    def test_get_execution_history_filtered(self):
        """Test filtered execution history retrieval"""
        # Execute codes with different agents
        self.tracker.track_and_execute_code(
            agent_id="agent_a",
            session_id="session_1",
            source_code="print('a')"
        )
        
        self.tracker.track_and_execute_code(
            agent_id="agent_b",
            session_id="session_1",
            source_code="print('b')"
        )
        
        self.tracker.track_and_execute_code(
            agent_id="agent_a",
            session_id="session_2",
            source_code="print('a2')"
        )
        
        # Filter by agent
        agent_a_history = self.tracker.get_execution_history(agent_id="agent_a")
        assert all(exec.agent_id == "agent_a" for exec in agent_a_history)
        assert len(agent_a_history) >= 2
        
        # Filter by session
        session_1_history = self.tracker.get_execution_history(session_id="session_1")
        assert all(exec.session_id == "session_1" for exec in session_1_history)
        assert len(session_1_history) >= 2
    
    def test_get_execution_statistics(self):
        """Test execution statistics"""
        # Execute some codes
        successful_code = "print('success')"
        failing_code = "undefined_variable"
        
        self.tracker.track_and_execute_code(
            agent_id="stats_agent",
            session_id="stats_session",
            source_code=successful_code
        )
        
        self.tracker.track_and_execute_code(
            agent_id="stats_agent",
            session_id="stats_session",
            source_code=failing_code
        )
        
        stats = self.tracker.get_execution_statistics()
        
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats
        assert "success_rate" in stats
        assert "average_execution_time" in stats
        assert stats["total_executions"] >= 2
    
    def test_analyze_code_quality_trends(self):
        """Test code quality trend analysis"""
        # Execute some codes to build history
        codes = [
            "print('simple')",  # Simple code
            """
# Complex function
def complex_calc(x):
    if x > 0:
        result = 0
        for i in range(x):
            if i % 2 == 0:
                result += i
        return result
    return 0
""",  # Complex code
        ]
        
        for code in codes:
            self.tracker.track_and_execute_code(
                agent_id="trend_agent",
                session_id="trend_session",
                source_code=code
            )
        
        trends = self.tracker.analyze_code_quality_trends(days=1)
        
        assert "period" in trends
        assert "total_executions" in trends
        assert "trends" in trends
        assert trends["total_executions"] >= 2
        
        if trends["trends"]:
            # Should have some quality metrics
            possible_metrics = ["complexity", "readability", "security", "performance"]
            assert any(metric in trends["trends"] for metric in possible_metrics)
    
    def test_generate_code_report(self):
        """Test code report generation"""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial: {result}")
"""
        
        execution_id, _ = self.tracker.track_and_execute_code(
            agent_id="report_agent",
            session_id="report_session",
            source_code=code
        )
        
        report = self.tracker.generate_code_report(execution_id)
        
        assert "코드 실행 보고서" in report
        assert "기본 정보" in report
        assert "코드 메트릭스" in report
        assert "실행 결과" in report
        assert execution_id in report
    
    def test_generate_code_report_nonexistent(self):
        """Test code report generation for non-existent execution"""
        report = self.tracker.generate_code_report("nonexistent_id")
        
        assert "실행 ID를 찾을 수 없습니다" in report
    
    def test_cleanup_old_executions(self):
        """Test cleanup of old executions"""
        # Create some executions
        code = "print('cleanup test')"
        
        execution_id = self.tracker.track_code_generation(
            agent_id="cleanup_agent",
            session_id="cleanup_session",
            source_code=code
        )
        
        initial_count = len(self.tracker.executions)
        
        # Cleanup with very recent cutoff (should remove nothing)
        self.tracker.cleanup_old_executions(days=0)
        
        # Should not remove recent executions
        assert len(self.tracker.executions) == initial_count
        assert execution_id in self.tracker.executions


class TestFactoryFunctions:
    """Test factory functions and convenience functions"""
    
    def test_get_advanced_code_tracker_singleton(self):
        """Test that get_advanced_code_tracker returns singleton"""
        tracker1 = get_advanced_code_tracker()
        tracker2 = get_advanced_code_tracker()
        
        assert tracker1 is tracker2
    
    def test_get_advanced_code_tracker_with_config(self):
        """Test get_advanced_code_tracker with custom config"""
        config = {"execution_timeout": 60, "memory_limit": 200}
        
        # Note: This will still return singleton, but we test the concept
        tracker = get_advanced_code_tracker(config)
        assert isinstance(tracker, AdvancedCodeTracker)
    
    def test_track_and_execute_convenience(self):
        """Test track_and_execute convenience function"""
        code = """
name = "World"
greeting = f"Hello, {name}!"
print(greeting)
"""
        
        execution_id, result = track_and_execute(
            agent_id="convenience_agent",
            session_id="convenience_session",
            code=code,
            tags=["greeting", "test"]
        )
        
        assert execution_id is not None
        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello, World!" in result.stdout
    
    def test_get_execution_stats_convenience(self):
        """Test get_execution_stats convenience function"""
        # Execute some code first
        track_and_execute(
            agent_id="stats_convenience_agent",
            session_id="stats_convenience_session",
            code="print('stats test')"
        )
        
        stats = get_execution_stats()
        
        assert isinstance(stats, dict)
        assert "total_executions" in stats
        assert "success_rate" in stats
        assert stats["total_executions"] >= 1


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.tracker = AdvancedCodeTracker()
    
    def test_data_analysis_workflow(self):
        """Test complete data analysis workflow"""
        analysis_codes = [
            {
                "code": """
# Data loading simulation
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Loaded {len(data)} data points")
""",
                "agent": "data_loader",
                "tags": ["loading"]
            },
            {
                "code": """
# Basic statistics
mean_val = sum(data) / len(data)
max_val = max(data)
min_val = min(data)
print(f"Mean: {mean_val}, Max: {max_val}, Min: {min_val}")
""",
                "agent": "statistics_agent",
                "tags": ["statistics"],
                "variables": {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            },
            {
                "code": """
# Data visualization (text-based)
bars = ['*' * int(val) for val in data[:5]]
for i, bar in enumerate(bars):
    print(f"{i+1}: {bar}")
""",
                "agent": "visualization_agent",
                "tags": ["visualization"],
                "variables": {"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            }
        ]
        
        results = []
        for analysis in analysis_codes:
            execution_id, result = self.tracker.track_and_execute_code(
                agent_id=analysis["agent"],
                session_id="workflow_session",
                source_code=analysis["code"],
                input_variables=analysis.get("variables"),
                tags=analysis["tags"]
            )
            results.append((execution_id, result))
        
        # All should succeed
        assert all(result.status == ExecutionStatus.SUCCESS for _, result in results)
        
        # Check tracking
        history = self.tracker.get_execution_history(session_id="workflow_session")
        assert len(history) >= 3
        
        # Check statistics
        stats = self.tracker.get_execution_statistics()
        assert stats["total_executions"] >= 3
    
    def test_error_handling_workflow(self):
        """Test error handling in workflow"""
        error_codes = [
            {
                "code": "print('This works')",
                "expected_status": ExecutionStatus.SUCCESS
            },
            {
                "code": "undefined_variable + 1",
                "expected_status": ExecutionStatus.ERROR
            },
            {
                "code": "import os; os.system('echo test')",
                "expected_status": ExecutionStatus.ERROR  # Security error
            },
            {
                "code": "print('Recovery after error')",
                "expected_status": ExecutionStatus.SUCCESS
            }
        ]
        
        for i, error_case in enumerate(error_codes):
            execution_id, result = self.tracker.track_and_execute_code(
                agent_id=f"error_agent_{i}",
                session_id="error_session",
                source_code=error_case["code"]
            )
            
            assert result.status == error_case["expected_status"]
        
        # Check that errors were properly tracked
        stats = self.tracker.get_execution_statistics()
        assert stats["failed_executions"] >= 2
        assert stats["successful_executions"] >= 2
    
    def test_concurrent_tracking(self):
        """Test concurrent code tracking"""
        import threading
        
        results = []
        
        def track_code(thread_id):
            code = f"""
thread_id = {thread_id}
result = thread_id * 2
print(f"Thread {{thread_id}} result: {{result}}")
"""
            execution_id, result = self.tracker.track_and_execute_code(
                agent_id=f"thread_agent_{thread_id}",
                session_id=f"thread_session_{thread_id}",
                source_code=code
            )
            results.append((execution_id, result))
        
        # Create and start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=track_code, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        assert all(result.status == ExecutionStatus.SUCCESS for _, result in results)
        
        # Check tracking
        stats = self.tracker.get_execution_statistics()
        assert stats["total_executions"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 