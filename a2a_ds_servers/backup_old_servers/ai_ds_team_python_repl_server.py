#!/usr/bin/env python3
"""
CherryAI Python REPL Agent - LangGraph 기반 코드 실행 에이전트
A2A SDK v0.2.9 표준 준수 + 비동기 스트리밍 + 실시간 코드 실행
"""

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Union
import subprocess
import tempfile
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안함

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part
)
from a2a.utils import new_agent_text_message, new_task

# LangGraph 라이브러리 (설치되어 있다고 가정)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # LangGraph가 없는 경우 기본 구현 사용
    LANGGRAPH_AVAILABLE = False
    from typing import Dict as TypedDict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExecutionState(TypedDict):
    """코드 실행 상태 관리"""
    user_request: str
    generated_code: str
    execution_result: str
    execution_status: str
    artifacts: List[Dict]
    context_variables: Dict[str, Any]
    error_message: Optional[str]
    step_history: List[Dict]


class SafeCodeExecutor:
    """안전한 코드 실행 환경"""
    
    def __init__(self):
        self.allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
            'scipy', 'sklearn', 'statsmodels', 'json', 'csv',
            'datetime', 'math', 'statistics', 're', 'os', 'sys'
        }
        self.execution_context = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'go': go,
            'px': px,
            'json': json
        }
        self.output_buffer = StringIO()
        self.artifacts = []
    
    def validate_code(self, code: str) -> bool:
        """코드 안전성 검증"""
        dangerous_patterns = [
            'import os', '__import__', 'eval(', 'exec(',
            'open(', 'file(', 'input(', 'raw_input(',
            'subprocess', 'system', 'popen', 'getattr',
            'setattr', 'delattr', 'globals(', 'locals(',
            'dir(', 'vars(', '__builtins__'
        ]
        
        # 허용된 import만 체크
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # 허용된 모듈인지 확인
                if not any(module in line for module in self.allowed_modules):
                    if not any(safe in line for safe in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly']):
                        return False
            
            # 위험한 패턴 체크
            for pattern in dangerous_patterns:
                if pattern in line:
                    return False
        
        return True
    
    async def execute_code_chunk(self, code: str, context_vars: Dict = None) -> Dict:
        """코드 청크 실행"""
        try:
            if not self.validate_code(code):
                return {
                    'status': 'error',
                    'output': '',
                    'error': '안전하지 않은 코드가 감지되었습니다.',
                    'artifacts': []
                }
        
            # 실행 컨텍스트 업데이트
            if context_vars:
                self.execution_context.update(context_vars)
        
            # stdout 캡처 설정
            old_stdout = sys.stdout
            old_stderr = sys.stderr
        
            captured_output = StringIO()
            sys.stdout = captured_output
            sys.stderr = captured_output
        
            try:
                # 코드 실행
                exec(code, self.execution_context)
                output = captured_output.getvalue()
        
                # 아티팩트 생성 (그래프, 데이터 등)
                artifacts = self._collect_artifacts()
        
                return {
                    'status': 'success',
                    'output': output,
                    'error': None,
                    'artifacts': artifacts,
                    'context_updated': True
                }
        
            except Exception as e:
                error_output = captured_output.getvalue()
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
        
                return {
                    'status': 'error',
                    'output': error_output,
                    'error': error_msg,
                    'artifacts': [],
                    'context_updated': False
                }
        
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        except Exception as e:
            return {
                'status': 'error',
                'output': '',
                'error': f'코드 실행 중 예상치 못한 오류: {str(e)}',
                'artifacts': [],
                'context_updated': False
            }
    
    def _collect_artifacts(self) -> List[Dict]:
        """실행 결과 아티팩트 수집"""
        artifacts = []
    
        # Matplotlib 그래프 수집
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
        
                # 임시 파일로 저장
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                fig.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        
                artifacts.append({
                    'type': 'matplotlib_plot',
                    'file_path': temp_file.name,
                    'figure_number': fig_num,
                    'timestamp': datetime.now().isoformat()
                })
        
                plt.close(fig)
    
        # DataFrame 결과 수집
        for var_name, var_value in self.execution_context.items():
            if isinstance(var_value, pd.DataFrame) and not var_name.startswith('_'):
                artifacts.append({
                    'type': 'dataframe',
                    'name': var_name,
                    'shape': var_value.shape,
                    'columns': list(var_value.columns),
                    'head': var_value.head().to_dict(),
                    'timestamp': datetime.now().isoformat()
                })
    
        return artifacts
    
    def get_context_summary(self) -> Dict:
        """실행 컨텍스트 요약"""
        summary = {
            'variables': [],
            'dataframes': [],
            'functions': []
        }
    
        for name, value in self.execution_context.items():
            if not name.startswith('_'):
                if isinstance(value, pd.DataFrame):
                    summary['dataframes'].append({
                        'name': name,
                        'type': 'DataFrame',
                        'shape': value.shape
                    })
                elif callable(value) and hasattr(value, '__name__'):
                    summary['functions'].append({
                        'name': name,
                        'type': 'function'
                    })
                else:
                    summary['variables'].append({
                        'name': name,
                        'type': type(value).__name__
                    })
       
        return summary


class LangGraphCodeInterpreter:
    """LangGraph 기반 코드 해석기"""
    
    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = openai_client
        self.code_executor = SafeCodeExecutor()
        self.memory_saver = MemorySaver() if LANGGRAPH_AVAILABLE else None
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
    
    def _build_graph(self) -> Optional[StateGraph]:
        """LangGraph 워크플로우 그래프 구성"""
        if not LANGGRAPH_AVAILABLE:
            return None
    
        workflow = StateGraph(CodeExecutionState)
    
        # 노드 추가
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("execute_code", self._execute_code)
        workflow.add_node("validate_result", self._validate_result)
        workflow.add_node("create_response", self._create_response)
    
        # 엣지 추가
        workflow.set_entry_point("analyze_request")
        workflow.add_edge("analyze_request", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "validate_result")
        workflow.add_edge("validate_result", "create_response")
        workflow.add_edge("create_response", END)
    
        return workflow.compile(checkpointer=self.memory_saver)
    
    async def _analyze_request(self, state: CodeExecutionState) -> CodeExecutionState:
        """사용자 요청 분석"""
        user_request = state["user_request"]
    
        # 요청 분석 로깅
        logger.info(f"📝 사용자 요청 분석: {user_request[:100]}...")
    
        state["step_history"].append({
            "step": "analyze_request",
            "description": "사용자 요청을 분석하여 필요한 코드 유형을 파악",
            "timestamp": datetime.now().isoformat()
        })
    
        return state
    
    async def _generate_code(self, state: CodeExecutionState) -> CodeExecutionState:
        """코드 생성"""
        user_request = state["user_request"]
    
        try:
            if self.openai_client:
                code = await self._generate_code_with_llm(user_request)
            else:
                code = self._generate_basic_code(user_request)
    
            state["generated_code"] = code
            state["step_history"].append({
                "step": "generate_code", 
                "description": "사용자 요청에 맞는 Python 코드 생성",
                "timestamp": datetime.now().isoformat(),
                "code_preview": code[:200] + "..." if len(code) > 200 else code
            })
    
        except Exception as e:
            state["error_message"] = f"코드 생성 실패: {str(e)}"
            state["execution_status"] = "failed"
    
        return state
    
    async def _execute_code(self, state: CodeExecutionState) -> CodeExecutionState:
        """코드 실행"""
        code = state["generated_code"]
    
        try:
            result = await self.code_executor.execute_code_chunk(
                code, state["context_variables"]
            )
    
            state["execution_result"] = result["output"]
            state["execution_status"] = result["status"]
            state["artifacts"].extend(result["artifacts"])
    
            if result["error"]:
                state["error_message"] = result["error"]
    
            state["step_history"].append({
                "step": "execute_code",
                "description": "생성된 코드를 안전한 환경에서 실행",
                "timestamp": datetime.now().isoformat(),
                "status": result["status"],
                "artifacts_count": len(result["artifacts"])
            })
    
        except Exception as e:
            state["execution_status"] = "failed"
            state["error_message"] = f"코드 실행 실패: {str(e)}"
    
        return state
    
    async def _validate_result(self, state: CodeExecutionState) -> CodeExecutionState:
        """실행 결과 검증"""
        if state["execution_status"] == "success" and state["execution_result"]:
            validation_status = "validated"
        elif state["artifacts"]:
            validation_status = "partial_success"  # 출력은 없지만 아티팩트가 생성됨
        else:
            validation_status = "failed"
    
        state["step_history"].append({
            "step": "validate_result",
            "description": "실행 결과의 유효성 검증",
            "timestamp": datetime.now().isoformat(),
            "validation_status": validation_status
        })
    
        return state
    
    async def _create_response(self, state: CodeExecutionState) -> CodeExecutionState:
        """최종 응답 생성"""
        if self.openai_client and state["execution_status"] == "success":
            response = await self._create_intelligent_response(state)
        else:
            response = self._create_basic_response(state)
    
        state["final_response"] = response
        state["step_history"].append({
            "step": "create_response",
            "description": "실행 결과를 바탕으로 최종 응답 생성",
            "timestamp": datetime.now().isoformat()
        })
    
        return state
    
    async def _generate_code_with_llm(self, user_request: str) -> str:
        """LLM을 활용한 코드 생성"""
        code_generation_prompt = f"""
다음 사용자 요청에 맞는 Python 코드를 생성해주세요.

**사용자 요청**: {user_request}

**사용 가능한 라이브러리**: pandas, numpy, matplotlib, seaborn, plotly
**코드 요구사항**:
1. 안전하고 실행 가능한 코드
2. 결과를 명확히 출력하는 코드
3. 필요시 시각화 포함
4. 주석으로 설명 추가

**응답 형식**: Python 코드만 반환 (```python 태그 없이)
"""
    
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 전문 Python 개발자이며 데이터 분석 코드 생성 전문가입니다."},
                {"role": "user", "content": code_generation_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
    
        return response.choices[0].message.content.strip()
    
    def _generate_basic_code(self, user_request: str) -> str:
        """기본 코드 생성 (LLM 없을 때)"""
        # 키워드 기반 기본 코드 생성
        if "시각화" in user_request or "그래프" in user_request or "plot" in user_request:
            return """
# 샘플 데이터 생성 및 시각화
import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('샘플 데이터 시각화')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

print("시각화가 완료되었습니다.")
"""
        elif "데이터" in user_request or "분석" in user_request:
            return """
# 샘플 데이터 분석
import pandas as pd
import numpy as np

# 샘플 데이터 생성
data = {
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100)
}
df = pd.DataFrame(data)

# 기본 분석
print("데이터 형태:", df.shape)
print("\n기본 통계:")
print(df.describe())
print("\n범주형 변수 분포:")
print(df['C'].value_counts())

print("\n데이터 분석이 완료되었습니다.")
"""
        else:
            return f"""
# 사용자 요청: {user_request}
print("안녕하세요! Python REPL 에이전트입니다.")
print("요청: {user_request}")
print("더 구체적인 요청을 해주시면 맞춤형 코드를 생성해드리겠습니다.")
"""
    
    async def _create_intelligent_response(self, state: CodeExecutionState) -> str:
        """지능형 응답 생성"""
        response_prompt = f"""
Python 코드 실행 결과를 바탕으로 사용자에게 전달할 응답을 생성해주세요.

**사용자 요청**: {state['user_request']}
**실행된 코드**: {state['generated_code']}
**실행 결과**: {state['execution_result']}
**생성된 아티팩트**: {len(state['artifacts'])}개

**응답 요구사항**:
1. 실행 결과 요약
2. 생성된 시각화나 데이터 설명
3. 추가 분석 제안
4. 사용자 친화적 설명

다음 구조로 응답해주세요:

# 🐍 Python 코드 실행 결과

## 📊 실행 요약
(실행된 작업 설명)

## 📈 결과 분석
(구체적인 결과 해석)

## 💡 추가 제안
(추가 분석이나 개선 제안)
"""
    
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 친근하고 전문적인 Python 데이터 분석 어시스턴트입니다."},
                {"role": "user", "content": response_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
    
        return response.choices[0].message.content
    
    def _create_basic_response(self, state: CodeExecutionState) -> str:
        """기본 응답 생성"""
        if state["execution_status"] == "success":
            response = f"""# 🐍 Python 코드 실행 결과

## 📊 실행 요약
사용자 요청 '{state['user_request']}'에 대한 Python 코드가 성공적으로 실행되었습니다.

## 📈 결과 분석
**실행 결과:**
```
{state['execution_result']}
```

**생성된 아티팩트:** {len(state['artifacts'])}개
{self._format_artifacts(state['artifacts'])}

## 💡 추가 제안
- 추가적인 분석이나 시각화가 필요하시면 언제든지 요청해주세요
- 코드 수정이나 개선사항이 있으면 알려주세요
"""
        else:
            response = f"""# 🐍 Python 코드 실행 결과

## ⚠️ 실행 오류
코드 실행 중 오류가 발생했습니다.

**오류 메시지:**
```
{state.get('error_message', '알 수 없는 오류')}
```

## 🔧 해결 방안
1. 코드 문법을 확인해주세요
2. 사용하는 라이브러리가 지원되는지 확인해주세요
3. 더 구체적인 요청으로 다시 시도해주세요
"""
       
        return response
    
    def _format_artifacts(self, artifacts: List[Dict]) -> str:
        """아티팩트 포맷팅"""
        if not artifacts:
            return ""
    
        formatted = "\n\n**생성된 결과물:**\n"
        for i, artifact in enumerate(artifacts, 1):
            if artifact['type'] == 'matplotlib_plot':
                formatted += f"- 📊 시각화 {i}: Matplotlib 그래프\n"
            elif artifact['type'] == 'dataframe':
                formatted += f"- 📋 데이터프레임 {i}: {artifact['name']} (형태: {artifact['shape']})\n"
            else:
                formatted += f"- 📄 결과물 {i}: {artifact['type']}\n"
    
        return formatted
    
    async def process_request(self, user_request: str) -> Dict:
        """사용자 요청 처리"""
        # 초기 상태 설정
        initial_state = {
            "user_request": user_request,
            "generated_code": "",
            "execution_result": "",
            "execution_status": "pending",
            "artifacts": [],
            "context_variables": {},
            "error_message": None,
            "step_history": [],
            "final_response": ""
        }
    
        if self.graph and LANGGRAPH_AVAILABLE:
            # LangGraph 워크플로우 실행
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
    
            final_state = await self.graph.ainvoke(
                initial_state, 
                config=config
            )
        else:
            # 기본 워크플로우 실행
            final_state = await self._execute_basic_workflow(initial_state)
    
        return final_state
    
    async def _execute_basic_workflow(self, state: CodeExecutionState) -> CodeExecutionState:
        """기본 워크플로우 실행 (LangGraph 없을 때)"""
        try:
            # 1. 요청 분석
            state = await self._analyze_request(state)
    
            # 2. 코드 생성
            state = await self._generate_code(state)
    
            # 3. 코드 실행
            state = await self._execute_code(state)
    
            # 4. 결과 검증
            state = await self._validate_result(state)
    
            # 5. 응답 생성
            state = await self._create_response(state)
    
        except Exception as e:
            state["execution_status"] = "failed"
            state["error_message"] = f"워크플로우 실행 실패: {str(e)}"
            state["final_response"] = f"오류 발생: {str(e)}"
    
        return state


class CherryAI_PythonREPL_Agent(AgentExecutor):
    """CherryAI Python REPL 에이전트"""
    
    def __init__(self):
        super().__init__()
        self.openai_client = self._initialize_openai_client()
        self.code_interpreter = LangGraphCodeInterpreter(self.openai_client)
        logger.info("CherryAI Python REPL Agent 초기화 완료")
    
    def _initialize_openai_client(self) -> Optional[AsyncOpenAI]:
        """OpenAI 클라이언트 초기화"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return None
            return AsyncOpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            return None
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 표준 프로토콜 기반 실행"""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
    
        try:
            # 첫 번째 태스크인 경우 submit 호출
            if not context.current_task:
                await updater.submit()
    
            # 작업 시작
            await updater.start_work()
    
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            logger.info(f"🐍 Python REPL 요청: {user_input[:100]}...")
    
            # 단계별 진행 상황 업데이트
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 사용자 요청을 분석하고 있습니다...")
            )
    
            # LangGraph 코드 해석기로 처리
            result = await self.code_interpreter.process_request(user_input)
    
            # 진행 상황 스트리밍
            for step in result.get("step_history", []):
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(f"⚡ {step['description']}")
                )
                await asyncio.sleep(0.5)  # 시각적 효과
    
            # 코드 실행 결과 전송
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("✅ Python 코드 실행이 완료되었습니다!")
            )
    
            # 아티팩트 전송
            artifacts = []
    
            # 생성된 코드 아티팩트
            if result.get("generated_code"):
                artifacts.append(TextPart(text=result["generated_code"]))
    
            # 실행 결과 아티팩트
            if result.get("execution_result"):
                artifacts.append(TextPart(text=f"실행 결과:\n{result['execution_result']}"))
    
            # 최종 응답 아티팩트
            if result.get("final_response"):
                artifacts.append(TextPart(text=result["final_response"]))
    
            await updater.add_artifact(
                artifacts,
                name="python_repl_execution",
                metadata={
                    "execution_status": result.get("execution_status", "unknown"),
                    "artifacts_count": len(result.get("artifacts", [])),
                    "step_count": len(result.get("step_history", [])),
                    "has_visualizations": any(a.get("type") == "matplotlib_plot" for a in result.get("artifacts", [])),
                    "code_length": len(result.get("generated_code", "")),
                    "version": "v1.0"
                }
            )
    
            await updater.complete()
    
        except Exception as e:
            logger.error(f"Python REPL 에이전트 실행 중 오류: {str(e)}")
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"오류 발생: {str(e)}")
            )
            raise
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """사용자 입력 추출"""
        try:
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part.root, 'kind') and part.root.kind == 'text':
                        return part.root.text
                    elif hasattr(part.root, 'type') and part.root.type == 'text':
                        return part.root.text
            return ""
        except Exception as e:
            logger.error(f"사용자 입력 추출 실패: {e}")
            return ""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info("❌ Python REPL 작업이 취소되었습니다")
        raise Exception('cancel not supported')


def create_agent_card() -> AgentCard:
    """Python REPL 에이전트 카드 생성"""
    return AgentCard(
        name="AI_DS_Team PythonREPLAgent",
        description="LangGraph 기반 Python 코드 실행 에이전트 - 비동기 스트리밍 + 실시간 결과 표시",
        url="http://localhost:8315",
        version="1.0.0",
        provider={
            "organization": "CherryAI Team",
            "url": "https://github.com/CherryAI"
        },
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id="python_code_execution",
                name="Python Code Execution",
                description="사용자 요청에 따른 Python 코드 생성, 실행, 결과 분석 및 시각화",
                tags=["python", "code_execution", "data_analysis", "visualization", "langgraph"],
                examples=[
                    "간단한 데이터 분석을 해주세요",
                    "그래프를 그려주세요",
                    "pandas로 데이터를 처리해주세요",
                    "통계 분석을 수행해주세요"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            )
        ],
        supportsAuthenticatedExtendedCard=False
    )


async def main():
    """Python REPL 에이전트 서버 시작"""
    try:
        # 에이전트 카드 생성
        agent_card = create_agent_card()
    
        # 태스크 스토어 및 실행자 초기화
        task_store = InMemoryTaskStore()
        agent_executor = CherryAI_PythonREPL_Agent()
    
        # 요청 핸들러 생성
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store,
        )
    
        # A2A 애플리케이션 생성
        app_builder = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        app = app_builder.build()
    
        # 서버 시작
        print("🐍 CherryAI Python REPL Agent 시작")
        print(f"📍 Agent Card: http://localhost:8315/.well-known/agent.json")
        print("🛑 종료하려면 Ctrl+C를 누르세요")
    
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8315,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 