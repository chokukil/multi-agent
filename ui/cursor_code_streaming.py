"""
Cursor-Style Real-Time Code Streaming UI
A2A SDK 0.2.9 + SSE 기반 실시간 코드 스트리밍 구현
"""

import streamlit as st
import time
import uuid
import json
import asyncio
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import queue
import logging

# A2A SDK 0.2.9 imports
try:
    from a2a.types import TextPart, DataPart, FilePart
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
    from a2a.server.application import A2AFastAPIApplication
    from a2a.server.request_handler import DefaultRequestHandler
    from a2a.server.agent_executor import AgentExecutor
    from a2a.server.request_context import RequestContext
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logging.warning("A2A SDK not available - using mock implementation")
    
    # Mock implementations for when A2A SDK is not available
    class AgentExecutor:
        """Mock AgentExecutor for when A2A SDK is not available"""
        async def execute(self, context, task_updater):
            pass
        
        async def cancel(self, context):
            pass
    
    class TaskUpdater:
        """Mock TaskUpdater for when A2A SDK is not available"""
        async def update_status(self, status, message=""):
            pass
        
        async def add_artifact(self, parts, name=None, metadata=None):
            pass
    
    class RequestContext:
        """Mock RequestContext for when A2A SDK is not available"""
        def __init__(self):
            self.message = None
    
    class TextPart:
        """Mock TextPart for when A2A SDK is not available"""
        def __init__(self, text):
            self.text = text
    
    class DataPart:
        """Mock DataPart for when A2A SDK is not available"""
        def __init__(self, data):
            self.data = data
    
    class FilePart:
        """Mock FilePart for when A2A SDK is not available"""
        def __init__(self, file_path):
            self.file_path = file_path

class CodeStreamingStatus(Enum):
    """코드 스트리밍 상태"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CodeBlockType(Enum):
    """코드 블록 타입"""
    FUNCTION = "function"
    CLASS = "class"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    EXECUTION = "execution"
    OUTPUT = "output"

@dataclass
class CodeLine:
    """개별 코드 라인"""
    line_id: str
    line_number: int
    content: str
    block_type: CodeBlockType
    is_highlighted: bool = False
    is_executed: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class CodeBlock:
    """코드 블록 (함수, 클래스 등)"""
    block_id: str
    block_type: CodeBlockType
    title: str
    description: str
    lines: List[CodeLine] = field(default_factory=list)
    status: CodeStreamingStatus = CodeStreamingStatus.PENDING
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CodePlan:
    """Cursor 스타일 코드 계획"""
    plan_id: str
    title: str
    description: str
    blocks: List[CodeBlock] = field(default_factory=list)
    status: CodeStreamingStatus = CodeStreamingStatus.PENDING
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class StreamingEvent:
    """SSE 스트리밍 이벤트"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class A2ACodeStreamingExecutor(AgentExecutor):
    """A2A SDK 기반 코드 스트리밍 실행기"""
    
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self.is_streaming = False
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """코드 스트리밍 실행"""
        try:
            # 요청 분석
            user_input = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_input += part.root.text
            
            # 코드 생성 계획 수립
            await task_updater.update_status("working", "코드 생성 계획을 수립하고 있습니다...")
            
            plan = await self._create_code_plan(user_input)
            
            # 계획을 아티팩트로 전송
            plan_artifact = TextPart(text=json.dumps(asdict(plan), ensure_ascii=False))
            await task_updater.add_artifact(
                parts=[plan_artifact],
                name="code_plan",
                metadata={"content_type": "application/json"}
            )
            
            # 코드 스트리밍 시작
            await task_updater.update_status("working", "코드를 실시간으로 생성하고 있습니다...")
            
            async for event in self._stream_code_generation(plan):
                # SSE 이벤트 큐에 추가
                self.event_queue.put(event)
                
                # A2A로 진행 상황 업데이트
                if event.event_type == "block_completed":
                    block_data = event.data
                    await task_updater.update_status(
                        "working", 
                        f"코드 블록 완료: {block_data.get('title', 'Unknown')}"
                    )
            
            # 최종 완료
            await task_updater.update_status("completed", "코드 생성이 완료되었습니다!")
            
        except Exception as e:
            await task_updater.update_status("failed", f"코드 생성 중 오류 발생: {str(e)}")
            raise
    
    async def cancel(self, context: RequestContext) -> None:
        """코드 스트리밍 취소"""
        self.is_streaming = False
        
    async def _create_code_plan(self, user_input: str) -> CodePlan:
        """사용자 입력을 바탕으로 코드 계획 생성"""
        plan_id = str(uuid.uuid4())
        
        # 간단한 예제 계획 (실제로는 LLM을 활용)
        plan = CodePlan(
            plan_id=plan_id,
            title="데이터 분석 코드 생성",
            description=f"사용자 요청: {user_input}",
            blocks=[
                CodeBlock(
                    block_id=str(uuid.uuid4()),
                    block_type=CodeBlockType.IMPORT,
                    title="필요한 라이브러리 임포트",
                    description="pandas, numpy, matplotlib 등 필요한 라이브러리를 임포트합니다"
                ),
                CodeBlock(
                    block_id=str(uuid.uuid4()),
                    block_type=CodeBlockType.FUNCTION,
                    title="데이터 로드 함수",
                    description="CSV 파일에서 데이터를 로드하는 함수를 생성합니다"
                ),
                CodeBlock(
                    block_id=str(uuid.uuid4()),
                    block_type=CodeBlockType.FUNCTION,
                    title="데이터 전처리 함수",
                    description="결측치 처리 및 데이터 정제 함수를 생성합니다"
                ),
                CodeBlock(
                    block_id=str(uuid.uuid4()),
                    block_type=CodeBlockType.EXECUTION,
                    title="분석 실행",
                    description="실제 데이터 분석을 실행하고 결과를 출력합니다"
                )
            ]
        )
        
        return plan
    
    async def _stream_code_generation(self, plan: CodePlan):
        """코드 생성 스트리밍"""
        self.is_streaming = True
        
        for block in plan.blocks:
            if not self.is_streaming:
                break
                
            # 블록 시작 이벤트
            yield StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type="block_started",
                data=asdict(block)
            )
            
            # 코드 라인별 생성
            sample_lines = self._generate_sample_code(block)
            
            for i, line_content in enumerate(sample_lines):
                if not self.is_streaming:
                    break
                    
                line = CodeLine(
                    line_id=str(uuid.uuid4()),
                    line_number=i + 1,
                    content=line_content,
                    block_type=block.block_type
                )
                
                block.lines.append(line)
                
                # 타이핑 효과를 위한 문자별 스트리밍
                for char_idx in range(len(line_content)):
                    partial_content = line_content[:char_idx + 1]
                    
                    yield StreamingEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="code_typing",
                        data={
                            "block_id": block.block_id,
                            "line_id": line.line_id,
                            "line_number": line.line_number,
                            "partial_content": partial_content,
                            "is_complete": char_idx == len(line_content) - 1
                        }
                    )
                    
                    await asyncio.sleep(0.05)  # 타이핑 속도 조절
                
                # 라인 완료 이벤트
                yield StreamingEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="line_completed",
                    data=asdict(line)
                )
                
                await asyncio.sleep(0.2)  # 라인 간 간격
            
            # 블록 완료 이벤트
            block.status = CodeStreamingStatus.COMPLETED
            block.end_time = time.time()
            block.progress = 1.0
            
            yield StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type="block_completed",
                data=asdict(block)
            )
            
            await asyncio.sleep(0.5)  # 블록 간 간격
    
    def _generate_sample_code(self, block: CodeBlock) -> List[str]:
        """블록 타입에 따른 샘플 코드 생성"""
        if block.block_type == CodeBlockType.IMPORT:
            return [
                "import pandas as pd",
                "import numpy as np",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "from datetime import datetime"
            ]
        elif block.block_type == CodeBlockType.FUNCTION:
            if "로드" in block.title:
                return [
                    "def load_data(file_path: str) -> pd.DataFrame:",
                    "    \"\"\"CSV 파일에서 데이터를 로드합니다\"\"\"",
                    "    try:",
                    "        df = pd.read_csv(file_path)",
                    "        print(f'데이터 로드 완료: {df.shape}')",
                    "        return df",
                    "    except Exception as e:",
                    "        print(f'데이터 로드 실패: {e}')",
                    "        return None"
                ]
            else:
                return [
                    "def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:",
                    "    \"\"\"데이터 전처리를 수행합니다\"\"\"",
                    "    # 결측치 처리",
                    "    df = df.dropna()",
                    "    # 중복 제거",
                    "    df = df.drop_duplicates()",
                    "    print(f'전처리 완료: {df.shape}')",
                    "    return df"
                ]
        elif block.block_type == CodeBlockType.EXECUTION:
            return [
                "# 데이터 분석 실행",
                "df = load_data('data.csv')",
                "if df is not None:",
                "    df_clean = preprocess_data(df)",
                "    print(df_clean.describe())",
                "    # 시각화",
                "    plt.figure(figsize=(10, 6))",
                "    df_clean.hist(bins=30)",
                "    plt.tight_layout()",
                "    plt.show()"
            ]
        else:
            return ["# 코드 생성 중..."]

class CursorCodeStreamingManager:
    """Cursor 스타일 코드 스트리밍 관리자"""
    
    def __init__(self):
        self.current_plan: Optional[CodePlan] = None
        self.event_queue = queue.Queue()
        self.is_streaming = False
        self.container = None
        self.executor = None
        
        if A2A_AVAILABLE:
            self.executor = A2ACodeStreamingExecutor(self.event_queue)
    
    def initialize_container(self):
        """Streamlit 컨테이너 초기화"""
        if 'cursor_code_streaming' not in st.session_state:
            st.session_state.cursor_code_streaming = {
                'current_plan': None,
                'streaming_active': False,
                'events': []
            }
        
        self.container = st.container()
        self._apply_cursor_styles()
    
    def _apply_cursor_styles(self):
        """Cursor 스타일 CSS 적용"""
        st.markdown("""
        <style>
        .cursor-code-plan {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #333;
        }
        
        .cursor-plan-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        .cursor-plan-title {
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
            margin: 0;
        }
        
        .cursor-plan-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .status-pending { background: #333; color: #999; }
        .status-planning { background: #1a4f8a; color: #64b5f6; }
        .status-generating { background: #1a4f8a; color: #64b5f6; }
        .status-executing { background: #2e7d32; color: #81c784; }
        .status-completed { background: #388e3c; color: #a5d6a7; }
        .status-failed { background: #d32f2f; color: #ef5350; }
        
        .cursor-code-block {
            background: #2d2d2d;
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
            border: 1px solid #444;
        }
        
        .cursor-block-header {
            background: #333;
            padding: 10px 15px;
            border-bottom: 1px solid #444;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .cursor-block-title {
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
        }
        
        .cursor-block-type {
            font-size: 11px;
            color: #999;
            background: #444;
            padding: 2px 8px;
            border-radius: 12px;
        }
        
        .cursor-code-content {
            padding: 0;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 13px;
            line-height: 1.4;
        }
        
        .cursor-code-line {
            display: flex;
            align-items: center;
            padding: 2px 0;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }
        
        .cursor-code-line.highlighted {
            background: #264f78;
            border-left-color: #007acc;
        }
        
        .cursor-code-line.executed {
            background: #1a4f1a;
            border-left-color: #4caf50;
        }
        
        .cursor-code-line.error {
            background: #4f1a1a;
            border-left-color: #f44336;
        }
        
        .cursor-line-number {
            color: #6e7681;
            font-size: 12px;
            min-width: 40px;
            text-align: right;
            padding-right: 15px;
            user-select: none;
        }
        
        .cursor-line-content {
            color: #e1e4e8;
            flex: 1;
            padding-right: 15px;
        }
        
        .cursor-typing-cursor {
            display: inline-block;
            width: 2px;
            height: 14px;
            background: #007acc;
            animation: cursor-blink 1s infinite;
        }
        
        @keyframes cursor-blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .cursor-progress-bar {
            background: #333;
            height: 4px;
            border-radius: 2px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .cursor-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007acc, #0099ff);
            transition: width 0.3s ease;
            animation: progress-shine 2s infinite;
        }
        
        @keyframes progress-shine {
            0% { background-position: -200px 0; }
            100% { background-position: 200px 0; }
        }
        
        .cursor-todo-item {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            margin: 4px 0;
            background: #2d2d2d;
            border-radius: 4px;
            border-left: 3px solid #333;
        }
        
        .cursor-todo-item.pending {
            border-left-color: #666;
        }
        
        .cursor-todo-item.in-progress {
            border-left-color: #007acc;
            background: #1a2833;
        }
        
        .cursor-todo-item.completed {
            border-left-color: #4caf50;
            background: #1a2f1a;
        }
        
        .cursor-todo-icon {
            margin-right: 8px;
            font-size: 14px;
        }
        
        .cursor-todo-text {
            flex: 1;
            font-size: 13px;
            color: #e1e4e8;
        }
        
        .cursor-todo-time {
            font-size: 11px;
            color: #6e7681;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def start_code_streaming(self, user_request: str):
        """코드 스트리밍 시작"""
        if self.is_streaming:
            return
            
        self.is_streaming = True
        st.session_state.cursor_code_streaming['streaming_active'] = True
        
        # 새로운 계획 생성
        plan = self._create_sample_plan(user_request)
        self.current_plan = plan
        st.session_state.cursor_code_streaming['current_plan'] = asdict(plan)
        
        # 스트리밍 시뮬레이션 시작
        self._simulate_streaming()
    
    def _create_sample_plan(self, user_request: str) -> CodePlan:
        """샘플 코드 계획 생성"""
        plan_id = str(uuid.uuid4())
        
        # Cursor 스타일 할일 목록 생성
        blocks = [
            CodeBlock(
                block_id=str(uuid.uuid4()),
                block_type=CodeBlockType.IMPORT,
                title="📦 Import Dependencies",
                description="필요한 라이브러리들을 임포트합니다"
            ),
            CodeBlock(
                block_id=str(uuid.uuid4()),
                block_type=CodeBlockType.FUNCTION,
                title="📊 Data Loading Function",
                description="데이터를 로드하는 함수를 생성합니다"
            ),
            CodeBlock(
                block_id=str(uuid.uuid4()),
                block_type=CodeBlockType.FUNCTION,
                title="🔧 Data Processing Function",
                description="데이터 전처리 함수를 생성합니다"
            ),
            CodeBlock(
                block_id=str(uuid.uuid4()),
                block_type=CodeBlockType.EXECUTION,
                title="🚀 Execute Analysis",
                description="분석을 실행하고 결과를 출력합니다"
            )
        ]
        
        return CodePlan(
            plan_id=plan_id,
            title=f"🎯 {user_request}",
            description="A2A SDK와 SSE를 활용한 실시간 코드 생성",
            blocks=blocks
        )
    
    def _simulate_streaming(self):
        """스트리밍 시뮬레이션"""
        if not self.current_plan:
            return
            
        # 계획 단계별 실행
        for block in self.current_plan.blocks:
            if not self.is_streaming:
                break
                
            # 블록 시작
            block.status = CodeStreamingStatus.GENERATING
            self._update_display()
            
            # 샘플 코드 생성
            sample_lines = self._get_sample_code_lines(block.block_type)
            
            for i, line_content in enumerate(sample_lines):
                if not self.is_streaming:
                    break
                    
                line = CodeLine(
                    line_id=str(uuid.uuid4()),
                    line_number=i + 1,
                    content=line_content,
                    block_type=block.block_type
                )
                
                block.lines.append(line)
                
                # 타이핑 효과 시뮬레이션
                for char_idx in range(len(line_content)):
                    if not self.is_streaming:
                        break
                        
                    # 부분 텍스트 표시
                    partial_content = line_content[:char_idx + 1]
                    if char_idx < len(line_content) - 1:
                        partial_content += "⚡"  # 타이핑 커서
                    
                    # 임시로 라인 내용 업데이트
                    line.content = partial_content
                    self._update_display()
                    
                    time.sleep(0.03)  # 타이핑 속도
                
                # 최종 라인 내용 설정
                line.content = line_content
                line.is_highlighted = True
                self._update_display()
                
                time.sleep(0.1)  # 라인 간 간격
            
            # 블록 완료
            block.status = CodeStreamingStatus.COMPLETED
            block.end_time = time.time()
            block.progress = 1.0
            
            # 모든 라인 하이라이트 해제
            for line in block.lines:
                line.is_highlighted = False
                line.is_executed = True
            
            self._update_display()
            time.sleep(0.5)  # 블록 간 간격
        
        # 전체 계획 완료
        self.current_plan.status = CodeStreamingStatus.COMPLETED
        self.current_plan.progress = 1.0
        self.is_streaming = False
        st.session_state.cursor_code_streaming['streaming_active'] = False
        self._update_display()
    
    def _get_sample_code_lines(self, block_type: CodeBlockType) -> List[str]:
        """블록 타입별 샘플 코드 라인"""
        if block_type == CodeBlockType.IMPORT:
            return [
                "import pandas as pd",
                "import numpy as np", 
                "import matplotlib.pyplot as plt",
                "from datetime import datetime",
                "import seaborn as sns"
            ]
        elif block_type == CodeBlockType.FUNCTION:
            return [
                "def analyze_data(df: pd.DataFrame) -> dict:",
                "    \"\"\"데이터 분석 함수\"\"\"",
                "    results = {}",
                "    results['shape'] = df.shape",
                "    results['columns'] = df.columns.tolist()",
                "    results['dtypes'] = df.dtypes.to_dict()",
                "    return results"
            ]
        elif block_type == CodeBlockType.EXECUTION:
            return [
                "# 데이터 분석 실행",
                "df = pd.read_csv('data.csv')",
                "results = analyze_data(df)",
                "print('Analysis Results:')",
                "print(f'Shape: {results[\"shape\"]}')",
                "print(f'Columns: {results[\"columns\"]}')"
            ]
        else:
            return ["# 코드 생성 중..."]
    
    def _update_display(self):
        """화면 업데이트"""
        if self.container and self.current_plan:
            # 세션 상태 업데이트
            st.session_state.cursor_code_streaming['current_plan'] = asdict(self.current_plan)
            
            # 화면 렌더링
            self.render_code_plan()
    
    def render_code_plan(self):
        """코드 계획 렌더링"""
        if not self.current_plan:
            return
            
        with self.container:
            # 계획 헤더
            st.markdown(f"""
            <div class="cursor-code-plan">
                <div class="cursor-plan-header">
                    <h3 class="cursor-plan-title">{self.current_plan.title}</h3>
                    <span class="cursor-plan-status status-{self.current_plan.status.value}">
                        {self.current_plan.status.value}
                    </span>
                </div>
                <p style="color: #999; margin-bottom: 20px;">{self.current_plan.description}</p>
            """, unsafe_allow_html=True)
            
            # 진행률 표시
            progress_percentage = self.current_plan.progress * 100
            st.markdown(f"""
            <div class="cursor-progress-bar">
                <div class="cursor-progress-fill" style="width: {progress_percentage}%"></div>
            </div>
            <p style="color: #999; font-size: 12px; text-align: center;">
                {progress_percentage:.1f}% 완료
            </p>
            """, unsafe_allow_html=True)
            
            # 할일 목록 (Cursor 스타일)
            st.markdown("### 📋 Code Generation Plan")
            for block in self.current_plan.blocks:
                status_class = "pending"
                status_icon = "⏳"
                
                if block.status == CodeStreamingStatus.GENERATING:
                    status_class = "in-progress"
                    status_icon = "🔄"
                elif block.status == CodeStreamingStatus.COMPLETED:
                    status_class = "completed"
                    status_icon = "✅"
                elif block.status == CodeStreamingStatus.FAILED:
                    status_class = "failed"
                    status_icon = "❌"
                
                st.markdown(f"""
                <div class="cursor-todo-item {status_class}">
                    <span class="cursor-todo-icon">{status_icon}</span>
                    <span class="cursor-todo-text">{block.title}</span>
                    <span class="cursor-todo-time">
                        {block.progress * 100:.0f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # 코드 블록들 렌더링
            for block in self.current_plan.blocks:
                if block.lines:  # 라인이 있는 블록만 표시
                    self._render_code_block(block)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _render_code_block(self, block: CodeBlock):
        """개별 코드 블록 렌더링"""
        st.markdown(f"""
        <div class="cursor-code-block">
            <div class="cursor-block-header">
                <h4 class="cursor-block-title">{block.title}</h4>
                <span class="cursor-block-type">{block.block_type.value}</span>
            </div>
            <div class="cursor-code-content">
        """, unsafe_allow_html=True)
        
        # 코드 라인들 렌더링
        for line in block.lines:
            line_class = ""
            if line.is_highlighted:
                line_class = "highlighted"
            elif line.is_executed:
                line_class = "executed"
            elif line.error_message:
                line_class = "error"
            
            st.markdown(f"""
            <div class="cursor-code-line {line_class}">
                <span class="cursor-line-number">{line.line_number}</span>
                <span class="cursor-line-content">{line.content}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False
        st.session_state.cursor_code_streaming['streaming_active'] = False
        if self.current_plan:
            self.current_plan.status = CodeStreamingStatus.CANCELLED
    
    def clear_plan(self):
        """계획 지우기"""
        self.current_plan = None
        self.is_streaming = False
        st.session_state.cursor_code_streaming = {
            'current_plan': None,
            'streaming_active': False,
            'events': []
        }

# 싱글톤 인스턴스
_cursor_code_streaming_manager = None

def get_cursor_code_streaming() -> CursorCodeStreamingManager:
    """Cursor 코드 스트리밍 관리자 싱글톤 인스턴스 반환"""
    global _cursor_code_streaming_manager
    if _cursor_code_streaming_manager is None:
        _cursor_code_streaming_manager = CursorCodeStreamingManager()
    return _cursor_code_streaming_manager 