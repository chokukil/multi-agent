#!/usr/bin/env python3
"""
CherryAI Python REPL Agent 테스트
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List
import tempfile
import pandas as pd
import numpy as np

# 테스트 대상 모듈들
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a_ds_servers.ai_ds_team_python_repl_server import (
    SafeCodeExecutor,
    LangGraphCodeInterpreter,
    CherryAI_PythonREPL_Agent,
    CodeExecutionState
)


class TestSafeCodeExecutor:
    """안전한 코드 실행기 테스트"""
    
    @pytest.fixture
    def code_executor(self):
        """코드 실행기 인스턴스 생성"""
        return SafeCodeExecutor()
    
    def test_validate_safe_code(self, code_executor):
        """안전한 코드 검증 테스트"""
        safe_code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.head())
"""
        assert code_executor.validate_code(safe_code) is True
    
    def test_validate_dangerous_code(self, code_executor):
        """위험한 코드 검증 테스트"""
        dangerous_codes = [
            "import os; os.system('rm -rf /')",
            "eval('malicious_code')",
            "exec('dangerous_code')",
            "open('/etc/passwd', 'r')",
            "__import__('subprocess')",
            "globals()['secret']"
        ]
        
        for dangerous_code in dangerous_codes:
            assert code_executor.validate_code(dangerous_code) is False
    
    @pytest.mark.asyncio
    async def test_execute_simple_code(self, code_executor):
        """간단한 코드 실행 테스트"""
        code = """
x = 10
y = 20
result = x + y
print(f"결과: {result}")
"""
        result = await code_executor.execute_code_chunk(code)
        
        assert result['status'] == 'success'
        assert '결과: 30' in result['output']
        assert result['error'] is None
        assert result['context_updated'] is True
    
    @pytest.mark.asyncio
    async def test_execute_pandas_code(self, code_executor):
        """Pandas 코드 실행 테스트"""
        code = """
import pandas as pd
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
print(f"DataFrame 형태: {df.shape}")
print(df.head())
"""
        result = await code_executor.execute_code_chunk(code)
        
        assert result['status'] == 'success'
        assert 'DataFrame 형태: (3, 2)' in result['output']
        assert len(result['artifacts']) > 0
        
        # DataFrame 아티팩트 확인
        df_artifacts = [a for a in result['artifacts'] if a['type'] == 'dataframe']
        assert len(df_artifacts) > 0
        assert df_artifacts[0]['shape'] == [3, 2]
    
    @pytest.mark.asyncio
    async def test_execute_matplotlib_code(self, code_executor):
        """Matplotlib 코드 실행 테스트"""
        code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sin Wave')
plt.show()
print("그래프 생성 완료")
"""
        result = await code_executor.execute_code_chunk(code)
        
        assert result['status'] == 'success'
        assert '그래프 생성 완료' in result['output']
        
        # Matplotlib 아티팩트 확인
        plot_artifacts = [a for a in result['artifacts'] if a['type'] == 'matplotlib_plot']
        assert len(plot_artifacts) > 0
        assert 'file_path' in plot_artifacts[0]
    
    @pytest.mark.asyncio
    async def test_execute_error_code(self, code_executor):
        """오류 코드 실행 테스트"""
        code = """
x = 10
y = 0
result = x / y  # ZeroDivisionError
print(result)
"""
        result = await code_executor.execute_code_chunk(code)
        
        assert result['status'] == 'error'
        assert 'ZeroDivisionError' in result['error']
        assert result['context_updated'] is False
    
    @pytest.mark.asyncio
    async def test_execute_unsafe_code(self, code_executor):
        """안전하지 않은 코드 실행 테스트"""
        code = "import os; os.system('echo unsafe')"
        result = await code_executor.execute_code_chunk(code)
        
        assert result['status'] == 'error'
        assert '안전하지 않은 코드가 감지되었습니다' in result['error']
    
    def test_get_context_summary(self, code_executor):
        """실행 컨텍스트 요약 테스트"""
        # 컨텍스트에 변수 추가
        code_executor.execution_context['test_var'] = 42
        code_executor.execution_context['test_df'] = pd.DataFrame({'A': [1, 2, 3]})
        code_executor.execution_context['test_func'] = lambda x: x * 2
        
        summary = code_executor.get_context_summary()
        
        assert len(summary['variables']) > 0
        assert len(summary['dataframes']) > 0
        assert any(var['name'] == 'test_var' for var in summary['variables'])
        assert any(df['name'] == 'test_df' for df in summary['dataframes'])


class TestLangGraphCodeInterpreter:
    """LangGraph 코드 해석기 테스트"""
    
    @pytest.fixture
    def code_interpreter(self):
        """코드 해석기 인스턴스 생성"""
        return LangGraphCodeInterpreter()
    
    @pytest.fixture
    def mock_openai_client(self):
        """OpenAI 클라이언트 모킹"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        
        # 코드 생성 응답
        mock_response.choices[0].message.content = """
import pandas as pd
import numpy as np

# 샘플 데이터 생성
data = np.random.randn(100, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

# 기본 통계
print("데이터 형태:", df.shape)
print("기본 통계:")
print(df.describe())
"""
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_generate_basic_code_data_analysis(self, code_interpreter):
        """기본 데이터 분석 코드 생성 테스트"""
        code = code_interpreter._generate_basic_code("데이터를 분석해주세요")
        
        assert "pandas" in code
        assert "DataFrame" in code
        assert "describe()" in code
        assert "분석이 완료되었습니다" in code
    
    def test_generate_basic_code_visualization(self, code_interpreter):
        """기본 시각화 코드 생성 테스트"""
        code = code_interpreter._generate_basic_code("그래프를 그려주세요")
        
        assert "matplotlib" in code
        assert "plt.figure" in code
        assert "plt.plot" in code
        assert "시각화가 완료되었습니다" in code
    
    def test_generate_basic_code_general(self, code_interpreter):
        """일반 요청 코드 생성 테스트"""
        code = code_interpreter._generate_basic_code("안녕하세요")
        
        assert "Python REPL 에이전트입니다" in code
        assert "안녕하세요" in code
    
    def test_create_basic_response_success(self, code_interpreter):
        """성공 시 기본 응답 생성 테스트"""
        state = {
            'user_request': '데이터 분석',
            'execution_status': 'success',
            'execution_result': '분석 완료\n결과: 성공',
            'artifacts': [
                {'type': 'dataframe', 'name': 'test_df'},
                {'type': 'matplotlib_plot', 'file_path': '/tmp/test.png'}
            ]
        }
        
        response = code_interpreter._create_basic_response(state)
        
        assert "# 🐍 Python 코드 실행 결과" in response
        assert "## 📊 실행 요약" in response
        assert "## 📈 결과 분석" in response
        assert "분석 완료" in response
        assert "2개" in response  # artifacts count
    
    def test_create_basic_response_failure(self, code_interpreter):
        """실패 시 기본 응답 생성 테스트"""
        state = {
            'user_request': '잘못된 코드',
            'execution_status': 'failed',
            'error_message': 'SyntaxError: invalid syntax'
        }
        
        response = code_interpreter._create_basic_response(state)
        
        assert "# 🐍 Python 코드 실행 결과" in response
        assert "## ⚠️ 실행 오류" in response
        assert "SyntaxError" in response
        assert "## 🔧 해결 방안" in response
    
    def test_format_artifacts(self, code_interpreter):
        """아티팩트 포맷팅 테스트"""
        artifacts = [
            {'type': 'matplotlib_plot', 'file_path': '/tmp/plot1.png'},
            {'type': 'dataframe', 'name': 'df1', 'shape': [100, 5]},
            {'type': 'custom', 'name': 'custom_result'}
        ]
        
        formatted = code_interpreter._format_artifacts(artifacts)
        
        assert "📊 시각화 1" in formatted
        assert "📋 데이터프레임 2" in formatted
        assert "📄 결과물 3" in formatted
        assert "형태: [100, 5]" in formatted
    
    @pytest.mark.asyncio
    async def test_analyze_request(self, code_interpreter):
        """요청 분석 단계 테스트"""
        state = {
            'user_request': '데이터를 분석해주세요',
            'step_history': []
        }
        
        result_state = await code_interpreter._analyze_request(state)
        
        assert len(result_state['step_history']) == 1
        assert result_state['step_history'][0]['step'] == 'analyze_request'
        assert '사용자 요청을 분석' in result_state['step_history'][0]['description']
    
    @pytest.mark.asyncio
    async def test_validate_result_success(self, code_interpreter):
        """성공적인 결과 검증 테스트"""
        state = {
            'execution_status': 'success',
            'execution_result': '분석 완료',
            'artifacts': [],
            'step_history': []
        }
        
        result_state = await code_interpreter._validate_result(state)
        
        assert len(result_state['step_history']) == 1
        assert result_state['step_history'][0]['step'] == 'validate_result'
        assert result_state['step_history'][0]['validation_status'] == 'validated'
    
    @pytest.mark.asyncio
    async def test_validate_result_partial(self, code_interpreter):
        """부분 성공 결과 검증 테스트"""
        state = {
            'execution_status': 'success',
            'execution_result': '',  # 출력 없음
            'artifacts': [{'type': 'plot'}],  # 하지만 아티팩트 있음
            'step_history': []
        }
        
        result_state = await code_interpreter._validate_result(state)
        
        assert result_state['step_history'][0]['validation_status'] == 'partial_success'
    
    @pytest.mark.asyncio
    async def test_process_request_basic_workflow(self, code_interpreter):
        """기본 워크플로우 처리 테스트"""
        result = await code_interpreter.process_request("간단한 계산을 해주세요")
        
        assert 'user_request' in result
        assert 'execution_status' in result
        assert 'step_history' in result
        assert 'final_response' in result
        assert len(result['step_history']) > 0
    
    @pytest.mark.asyncio
    async def test_generate_code_with_llm(self, mock_openai_client):
        """LLM을 활용한 코드 생성 테스트"""
        interpreter = LangGraphCodeInterpreter(mock_openai_client)
        
        code = await interpreter._generate_code_with_llm("데이터 분석을 해주세요")
        
        assert "pandas" in code
        assert "DataFrame" in code
        mock_openai_client.chat.completions.create.assert_called_once()


class TestCherryAI_PythonREPL_Agent:
    """CherryAI Python REPL 에이전트 테스트"""
    
    @pytest.fixture
    def agent(self):
        """에이전트 인스턴스 생성"""
        return CherryAI_PythonREPL_Agent()
    
    def test_agent_initialization(self, agent):
        """에이전트 초기화 테스트"""
        assert agent.code_interpreter is not None
        assert hasattr(agent, 'openai_client')
    
    def test_extract_user_input(self, agent):
        """사용자 입력 추출 테스트"""
        # Mock RequestContext 생성
        mock_context = Mock()
        mock_message = Mock()
        mock_part = Mock()
        mock_part.root.kind = 'text'
        mock_part.root.text = 'Python 코드를 실행해주세요'
        mock_message.parts = [mock_part]
        mock_context.message = mock_message
        
        user_input = agent._extract_user_input(mock_context)
        assert user_input == 'Python 코드를 실행해주세요'
    
    def test_extract_user_input_type_attribute(self, agent):
        """type 속성을 가진 사용자 입력 추출 테스트"""
        # Mock RequestContext 생성 (type 속성 사용)
        mock_context = Mock()
        mock_message = Mock()
        mock_part = Mock()
        
        # kind 속성이 없고 type 속성만 있는 경우
        del mock_part.root.kind
        mock_part.root.type = 'text'
        mock_part.root.text = 'type 속성 테스트'
        mock_message.parts = [mock_part]
        mock_context.message = mock_message
        
        user_input = agent._extract_user_input(mock_context)
        assert user_input == 'type 속성 테스트'
    
    def test_extract_user_input_empty(self, agent):
        """빈 입력 추출 테스트"""
        mock_context = Mock()
        mock_context.message = None
        
        user_input = agent._extract_user_input(mock_context)
        assert user_input == ""


@pytest.mark.integration
class TestPythonREPLAgent_Integration:
    """Python REPL 에이전트 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_analysis(self):
        """엔드투엔드 데이터 분석 테스트"""
        interpreter = LangGraphCodeInterpreter()
        
        result = await interpreter.process_request(
            "샘플 데이터를 생성하고 기본 통계를 보여주세요"
        )
        
        assert result['execution_status'] in ['success', 'failed']
        assert 'final_response' in result
        assert len(result['step_history']) >= 4  # 최소 4단계 워크플로우
    
    @pytest.mark.asyncio
    async def test_end_to_end_visualization(self):
        """엔드투엔드 시각화 테스트"""
        interpreter = LangGraphCodeInterpreter()
        
        result = await interpreter.process_request(
            "간단한 선 그래프를 그려주세요"
        )
        
        assert result['execution_status'] in ['success', 'failed']
        assert 'final_response' in result
        
        # 성공한 경우 시각화 아티팩트 확인
        if result['execution_status'] == 'success':
            plot_artifacts = [
                a for a in result.get('artifacts', [])
                if a.get('type') == 'matplotlib_plot'
            ]
            # 시각화 요청이므로 plot 아티팩트가 있을 수 있음
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """오류 처리 테스트"""
        interpreter = LangGraphCodeInterpreter()
        
        result = await interpreter.process_request(
            "잘못된 문법의 코드를 실행해주세요: print(未定義変数)"
        )
        
        # 오류가 발생해도 적절히 처리되어야 함
        assert 'execution_status' in result
        assert 'final_response' in result
        assert len(result['step_history']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 