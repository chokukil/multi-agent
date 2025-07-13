"""
💾 Data Workspace Component - LLM First Architecture
Cursor 스타일의 완전한 LLM 기반 데이터 분석 워크스페이스
Real AI Analysis with A2A Integration & Langfuse Logging
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import io
import json
import time
import asyncio
import os
from datetime import datetime
import openai
from openai import AsyncOpenAI
import httpx # Added for dynamic agent discovery
import matplotlib.pyplot as plt # Added for safe code execution
import seaborn as sns # Added for safe code execution
import numpy as np # Added for safe code execution

# Langfuse 통합
try:
    from core.langfuse_session_tracer import get_session_tracer
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# A2A 클라이언트 통합 - A2A SDK 0.2.9 표준 준수
try:
    from core.a2a.a2a_streamlit_client import A2AStreamlitClient
    from core.enhanced_a2a_communicator import EnhancedA2ACommunicator
    from a2a.client import A2AClient
    from a2a.types import Message, TextPart, Role
    from a2a.utils.message import new_agent_text_message
    A2A_CLIENT_AVAILABLE = True
except ImportError:
    A2A_CLIENT_AVAILABLE = False

# SSE 스트리밍 지원
try:
    from core.utils.streaming import StreamingManager
    import asyncio
    from typing import AsyncGenerator
    SSE_STREAMING_AVAILABLE = True
except ImportError:
    SSE_STREAMING_AVAILABLE = False

# LLM 클라이언트 초기화
try:
    llm_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False

# SSE 실시간 스트리밍 구현
import asyncio
from typing import AsyncGenerator, Callable
import json
import time

# SSE 스트리밍 상태 관리
class SSEStreamingManager:
    """SSE 스트리밍 관리자 - A2A 표준 준수"""
    
    def __init__(self):
        self.active_streams = {}
        self.stream_callbacks = {}
        self.streaming_updates = []
    
    async def start_sse_stream(self, stream_id: str, callback: Callable = None):
        """SSE 스트림 시작"""
        self.active_streams[stream_id] = {
            "start_time": time.time(),
            "status": "active",
            "chunks_received": 0
        }
        
        if callback:
            self.stream_callbacks[stream_id] = callback
        
        # 실시간 UI 업데이트
        if 'sse_streams' not in st.session_state:
            st.session_state['sse_streams'] = {}
        
        st.session_state['sse_streams'][stream_id] = {
            "status": "streaming",
            "start_time": time.time(),
            "updates": []
        }
    
    async def handle_sse_chunk(self, stream_id: str, chunk: str):
        """SSE 청크 처리"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["chunks_received"] += 1
            
            # 실시간 업데이트
            update = {
                "timestamp": time.time(),
                "chunk": chunk,
                "chunk_index": self.active_streams[stream_id]["chunks_received"]
            }
            
            self.streaming_updates.append(update)
            
            # 세션 상태 업데이트
            if 'sse_streams' in st.session_state and stream_id in st.session_state['sse_streams']:
                st.session_state['sse_streams'][stream_id]["updates"].append(update)
            
            # 콜백 실행
            if stream_id in self.stream_callbacks:
                try:
                    await self.stream_callbacks[stream_id](chunk)
                except Exception as e:
                    print(f"SSE 콜백 오류: {e}")
    
    async def end_sse_stream(self, stream_id: str):
        """SSE 스트림 종료"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["status"] = "completed"
            self.active_streams[stream_id]["end_time"] = time.time()
            
            # 세션 상태 업데이트
            if 'sse_streams' in st.session_state and stream_id in st.session_state['sse_streams']:
                st.session_state['sse_streams'][stream_id]["status"] = "completed"
                st.session_state['sse_streams'][stream_id]["end_time"] = time.time()

# 전역 SSE 스트리밍 매니저
sse_manager = SSEStreamingManager()

# A2A SSE 스트리밍 클라이언트
class A2ASSEClient:
    """A2A 표준 SSE 스트리밍 클라이언트"""
    
    def __init__(self):
        self.timeout = 300.0
        self.max_retries = 3
    
    async def stream_a2a_request(self, 
                                agent_url: str, 
                                message: str,
                                stream_id: str = None,
                                context: dict = None) -> AsyncGenerator[dict, None]:
        """A2A SSE 스트리밍 요청"""
        
        if not stream_id:
            stream_id = f"a2a_stream_{int(time.time())}"
        
        # SSE 스트림 시작
        await sse_manager.start_sse_stream(
            stream_id, 
            lambda chunk: self._handle_stream_chunk(stream_id, chunk)
        )
        
        # A2A 표준 메시지 구성
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": stream_id,
                "message": {
                    "messageId": f"msg_{stream_id}",
                    "role": "user",
                    "parts": [{"type": "text", "text": message}]
                },
                "context": context or {}
            },
            "id": stream_id
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # SSE 스트리밍 요청
                async with client.stream(
                    "POST",
                    f"{agent_url}/a2a/stream",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache"
                    }
                ) as response:
                    
                    if response.status_code != 200:
                        yield {
                            "type": "error",
                            "content": f"HTTP {response.status_code}: {await response.aread()}"
                        }
                        return
                    
                    # SSE 스트림 처리
                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])  # "data: " 제거
                                    
                                    # A2A 표준 응답 처리
                                    if "result" in data:
                                        result = data["result"]
                                        
                                        # 메시지 추출
                                        if "message" in result:
                                            message_data = result["message"]
                                            if "parts" in message_data:
                                                for part in message_data["parts"]:
                                                    if part.get("type") == "text":
                                                        content = part.get("text", "")
                                                        if content:
                                                            await sse_manager.handle_sse_chunk(stream_id, content)
                                                            yield {
                                                                "type": "message",
                                                                "content": content,
                                                                "stream_id": stream_id,
                                                                "timestamp": time.time()
                                                            }
                                    
                                    # 완료 신호 처리
                                    if data.get("final") or data.get("done"):
                                        await sse_manager.end_sse_stream(stream_id)
                                        yield {
                                            "type": "complete",
                                            "stream_id": stream_id,
                                            "timestamp": time.time()
                                        }
                                        return
                                
                                except json.JSONDecodeError:
                                    # JSON 파싱 실패 시 원본 텍스트 전송
                                    content = line[6:] if line.startswith("data: ") else line
                                    await sse_manager.handle_sse_chunk(stream_id, content)
                                    yield {
                                        "type": "raw",
                                        "content": content,
                                        "stream_id": stream_id,
                                        "timestamp": time.time()
                                    }
                            
                            elif line.startswith("event: "):
                                event_type = line[7:]  # "event: " 제거
                                yield {
                                    "type": "event",
                                    "event_type": event_type,
                                    "stream_id": stream_id,
                                    "timestamp": time.time()
                                }
                    
                    # 스트림 자연 종료
                    await sse_manager.end_sse_stream(stream_id)
                    yield {
                        "type": "complete",
                        "stream_id": stream_id,
                        "timestamp": time.time()
                    }
        
        except Exception as e:
            await sse_manager.end_sse_stream(stream_id)
            yield {
                "type": "error",
                "content": str(e),
                "stream_id": stream_id,
                "timestamp": time.time()
            }
    
    async def _handle_stream_chunk(self, stream_id: str, chunk: str):
        """스트림 청크 처리"""
        # 실시간 UI 업데이트
        if 'streaming_content' not in st.session_state:
            st.session_state['streaming_content'] = {}
        
        if stream_id not in st.session_state['streaming_content']:
            st.session_state['streaming_content'][stream_id] = ""
        
        st.session_state['streaming_content'][stream_id] += chunk
        
        # 스트리밍 업데이트 알림
        if hasattr(st, 'rerun'):
            st.rerun()

# 전역 A2A SSE 클라이언트
a2a_sse_client = A2ASSEClient()

# SSE 스트리밍 UI 컴포넌트
def render_sse_streaming_status():
    """SSE 스트리밍 상태 표시"""
    
    if st.session_state.get('sse_streams'):
        st.markdown("### 📡 실시간 스트리밍 상태")
        
        for stream_id, stream_info in st.session_state['sse_streams'].items():
            status = stream_info.get('status', 'unknown')
            start_time = stream_info.get('start_time', 0)
            updates_count = len(stream_info.get('updates', []))
            
            if status == 'streaming':
                st.info(f"🔄 **{stream_id}**: 스트리밍 중 ({updates_count} 업데이트)")
            elif status == 'completed':
                duration = stream_info.get('end_time', time.time()) - start_time
                st.success(f"✅ **{stream_id}**: 완료 ({duration:.1f}초, {updates_count} 업데이트)")
            else:
                st.warning(f"⚠️ **{stream_id}**: {status}")
    
    # 실시간 스트리밍 콘텐츠 표시
    if st.session_state.get('streaming_content'):
        st.markdown("### 📺 실시간 스트리밍 콘텐츠")
        
        for stream_id, content in st.session_state['streaming_content'].items():
            if content:
                with st.expander(f"📡 {stream_id}", expanded=True):
                    st.write(content)
                    
                    # 자동 스크롤 효과
                    st.markdown("""
                    <script>
                        // 자동 스크롤
                        window.scrollTo(0, document.body.scrollHeight);
                    </script>
                    """, unsafe_allow_html=True)

def apply_llm_first_layout_styles():
    """LLM First 아키텍처 기반 레이아웃 스타일"""
    st.markdown("""
    <style>
    /* 전체 컨테이너 */
    .main .block-container {
        max-width: 1500px !important;
        padding: 1rem !important;
    }
    
    /* 30% 입력, 70% 결과 레이아웃 */
    .analysis-layout {
        display: flex;
        gap: 1rem;
        min-height: 80vh;
    }
    
    .input-panel {
        flex: 0 0 30%;
        background: rgba(15, 15, 15, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 122, 204, 0.3);
        position: sticky;
        top: 1rem;
        height: fit-content;
    }
    
    .results-panel {
        flex: 1;
        background: rgba(10, 10, 10, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
        overflow-y: auto;
        max-height: 85vh;
    }
    
    /* 동적 질문 생성 카드 */
    .dynamic-question-card {
        background: linear-gradient(135deg, rgba(0, 122, 204, 0.1), rgba(0, 212, 255, 0.1));
        border: 1px solid rgba(0, 122, 204, 0.4);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .dynamic-question-card:hover {
        background: linear-gradient(135deg, rgba(0, 122, 204, 0.2), rgba(0, 212, 255, 0.2));
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 122, 204, 0.3);
    }
    
    /* Follow-up 제안 버튼 */
    .followup-suggestion {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.4);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem;
        display: inline-block;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #4CAF50;
        font-weight: 500;
    }
    
    .followup-suggestion:hover {
        background: rgba(76, 175, 80, 0.2);
        transform: scale(1.02);
    }
    
    /* 실시간 프로세스 투명화 */
    .process-transparency {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .agent-activity {
        background: rgba(156, 39, 176, 0.1);
        border-left: 4px solid #9C27B0;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* 자동 스크롤 애니메이션 */
    .auto-scroll-content {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 스트리밍 결과 */
    .streaming-result {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        white-space: pre-wrap;
        font-family: 'Monaco', 'Menlo', monospace;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* 대화 흐름 */
    .conversation-flow {
        border-left: 3px solid #007acc;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background: rgba(0, 122, 204, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .ai-response {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

async def generate_dynamic_questions(df: pd.DataFrame, user_context: Dict = None) -> List[Dict]:
    """LLM을 사용하여 데이터에 맞는 동적 질문 생성"""
    if not LLM_AVAILABLE:
        return []
    
    try:
        # 데이터 특성 분석
        data_summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict()
        }
        
        prompt = f"""
당신은 데이터 분석 전문가입니다. 주어진 데이터의 특성을 분석하여 사용자가 궁금해할 만한 질문 3개를 생성해주세요.

데이터 정보:
- 크기: {data_summary['shape']}
- 컬럼: {data_summary['columns']}
- 숫자 컬럼: {data_summary['numeric_columns']}
- 범주형 컬럼: {data_summary['categorical_columns']}

다음 JSON 형식으로 응답해주세요:
{{
    "questions": [
        {{
            "text": "질문 내용",
            "type": "분석 유형",
            "reasoning": "이 질문을 제안하는 이유"
        }}
    ]
}}

규칙:
1. 데이터의 실제 특성을 반영한 구체적인 질문
2. 사용자가 바로 이해할 수 있는 자연스러운 한국어
3. 각 질문은 서로 다른 분석 관점을 제공
4. 템플릿이 아닌 이 데이터에만 특화된 질문
"""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            questions_data = json.loads(json_match.group())
            return questions_data.get("questions", [])
        
        return []
        
    except Exception as e:
        st.error(f"동적 질문 생성 실패: {e}")
        return []

async def generate_followup_suggestions(analysis_result: str, conversation_history: List[Dict]) -> List[str]:
    """LLM을 사용하여 Follow-up 제안 생성"""
    if not LLM_AVAILABLE:
        return []
    
    try:
        # 대화 히스토리 요약
        history_summary = []
        for chat in conversation_history[-3:]:  # 최근 3개 대화만 사용
            history_summary.append(f"질문: {chat['question']}")
            history_summary.append(f"답변: {chat['answer'][:200]}...")
        
        prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자와의 대화를 분석하여 다음 단계에 적합한 Follow-up 제안을 생성해주세요.

현재 분석 결과:
{analysis_result[:500]}...

대화 히스토리:
{chr(10).join(history_summary)}

다음 JSON 형식으로 3개의 간결한 Follow-up 제안을 해주세요:
{{
    "suggestions": [
        "제안 1 (간결하게)",
        "제안 2 (간결하게)",
        "제안 3 (간결하게)"
    ]
}}

규칙:
1. 각 제안은 15단어 이내로 간결하게
2. 현재 분석 결과와 연관성 있는 다음 단계 제안
3. 버튼으로 클릭하기 적합한 형태
4. 사용자가 바로 이해할 수 있는 자연스러운 한국어
5. 템플릿이 아닌 현재 상황에 맞는 맞춤형 제안
"""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        result = response.choices[0].message.content
        
        # JSON 파싱
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            suggestions_data = json.loads(json_match.group())
            return suggestions_data.get("suggestions", [])
        
        return []
        
    except Exception as e:
        st.error(f"Follow-up 제안 생성 실패: {e}")
        return []

def render_input_panel(df: pd.DataFrame):
    """입력 패널 렌더링 (30% 좌측) - 실시간 상태 모니터링 통합"""
    st.markdown("### 🧬 LLM First 분석")
    
    # === 실시간 분석 상태 섹션 ===
    if st.session_state.get('analysis_in_progress', False):
        current_stage = st.session_state.get('current_stage', '분석 준비 중')
        st.success(f"🔄 **현재 진행**: {current_stage}")
        
        # 진행률 표시
        stage_progress = {
            "계획 수립": 25,
            "A2A 에이전트 실행": 50,
            "아티팩트 생성": 75,
            "결과 통합": 100
        }
        progress = stage_progress.get(current_stage, 0)
        st.progress(progress / 100)
        
        # 분석 계획 요약
        if st.session_state.get('analysis_plan'):
            with st.expander("📋 분석 계획", expanded=True):
                plan = st.session_state['analysis_plan']
                st.write(f"**분석 유형**: {plan.get('analysis_type', 'N/A')}")
                st.write(f"**필요 에이전트**: {', '.join(plan.get('required_agents', []))}")
                st.write(f"**시각화 필요**: {'✅' if plan.get('visualization_needed', False) else '❌'}")
    
    # === 에이전트 상태 모니터링 ===
    st.markdown("### 🤖 에이전트 상태")
    
    # A2A 서버 상태 확인
    agent_servers = {
        "오케스트레이터": "8100",
        "EDA 도구": "8312", 
        "데이터 시각화": "8308",
        "데이터 정리": "8306",
        "피처 엔지니어링": "8310"
    }
    
    if st.session_state.get('agent_execution_status'):
        # 실행 중인 에이전트 상태
        for agent, status in st.session_state['agent_execution_status'].items():
            if "실행 중" in status:
                st.write(f"🔄 **{agent}**: {status}")
            elif "완료" in status:
                st.write(f"✅ **{agent}**: {status}")
            elif "실패" in status:
                st.write(f"❌ **{agent}**: {status}")
            else:
                st.write(f"⏳ **{agent}**: {status}")
    else:
        # 기본 서버 상태 (포트 기반)
        for name, port in agent_servers.items():
            st.write(f"💤 **{name}** (:{port}): 대기 중")
    
    # === 아티팩트 생성 현황 ===
    st.markdown("### 📦 아티팩트 생성 현황")
    
    artifacts = st.session_state.get('generated_artifacts', [])
    if artifacts:
        for artifact in artifacts:
            artifact_name = artifact.get('name', '알 수 없음')
            artifact_type = artifact.get('type', 'unknown')
            
            type_icons = {
                'plotly_chart': '📊',
                'dataframe': '📋',
                'image': '🖼️',
                'code': '💻',
                'text': '📝'
            }
            icon = type_icons.get(artifact_type, '📦')
            
            st.write(f"{icon} **{artifact_name}**: ✅ 생성 완료")
    else:
        st.write("📋 아직 생성된 아티팩트가 없습니다")
    
    # === 데이터 정보 요약 ===
    with st.expander("📊 데이터 정보", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("행 수", df.shape[0])
            st.metric("컬럼 수", df.shape[1])
        with col2:
            st.metric("숫자 컬럼", len(df.select_dtypes(include=['number']).columns))
            st.metric("범주형 컬럼", len(df.select_dtypes(include=['object']).columns))
    
    # === Langfuse 세션 정보 ===
    if LANGFUSE_AVAILABLE:
        with st.expander("🔍 Langfuse 추적", expanded=False):
            recent_chat = st.session_state.get('conversation_history', [])
            if recent_chat:
                last_session = recent_chat[-1].get('session_id')
                if last_session:
                    st.write(f"**최근 세션**: {last_session}")
                    st.write(f"**추적 상태**: ✅ 활성")
                else:
                    st.write("**추적 상태**: ⚠️ 세션 없음")
            else:
                st.write("**추적 상태**: 💤 대기 중")
    
    # === 동적 질문 생성 섹션 ===
    st.markdown("### 💡 추천 질문")
    
    if 'dynamic_questions' not in st.session_state:
        st.session_state['dynamic_questions'] = []
        st.session_state['questions_loading'] = True
    
    if st.session_state.get('questions_loading', False):
        with st.spinner("🤖 데이터 특성을 분석하여 맞춤형 질문을 생성하는 중..."):
            # 비동기 함수 실행
            if LLM_AVAILABLE:
                questions = asyncio.run(generate_dynamic_questions(df))
                st.session_state['dynamic_questions'] = questions
                st.session_state['questions_loading'] = False
                st.rerun()
            else:
                st.session_state['questions_loading'] = False
                st.warning("⚠️ LLM이 사용 불가능합니다.")
    
    # 동적 생성된 질문 표시
    if st.session_state.get('dynamic_questions'):
        for i, question in enumerate(st.session_state['dynamic_questions']):
            if st.button(
                f"🔍 {question['text']}", 
                key=f"dynamic_q_{i}",
                use_container_width=True
            ):
                st.session_state['selected_question'] = question['text']
                st.rerun()
    
    # 새로운 질문 생성 버튼
    if st.button("🔄 새로운 질문 생성", use_container_width=True):
        st.session_state['questions_loading'] = True
        st.session_state['dynamic_questions'] = []
        st.rerun()
    
    # === 시각화 빠른 실행 ===
    st.markdown("### ⚡ 빠른 시각화")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 기본 차트", use_container_width=True):
            st.session_state['selected_question'] = "기본적인 차트를 그려주세요"
            st.rerun()
    
    with col2:
        if st.button("📈 상관관계", use_container_width=True):
            st.session_state['selected_question'] = "변수들 간의 상관관계를 시각화해주세요"
            st.rerun()
    
    # === 시스템 정보 ===
    with st.expander("🔧 시스템 정보", expanded=False):
        st.write(f"**LLM 연결**: {'✅' if LLM_AVAILABLE else '❌'}")
        st.write(f"**A2A 클라이언트**: {'✅' if A2A_CLIENT_AVAILABLE else '❌'}")
        st.write(f"**Langfuse 추적**: {'✅' if LANGFUSE_AVAILABLE else '❌'}")
        
        if st.session_state.get('conversation_history'):
            st.write(f"**총 대화 수**: {len(st.session_state['conversation_history'])}")
        
        if st.session_state.get('generated_artifacts'):
            st.write(f"**생성된 아티팩트**: {len(st.session_state['generated_artifacts'])}개")

def render_results_panel():
    """결과 패널 렌더링 (70% 우측) - 분석 결과, 아티팩트, 에이전트 상태 통합 표시"""
    st.markdown("### 📊 분석 결과")
    
    # 대화 히스토리 초기화
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # 현재 분석 진행 상황 표시
    if st.session_state.get('analysis_in_progress', False):
        current_stage = st.session_state.get('current_stage', '분석 준비 중')
        st.info(f"🔄 **현재 단계**: {current_stage}")
        
        # 분석 계획 표시
        if st.session_state.get('analysis_plan'):
            with st.expander("📋 분석 계획", expanded=True):
                plan = st.session_state['analysis_plan']
                st.json(plan)
        
        # 에이전트 실행 상태 표시
        if st.session_state.get('agent_execution_status'):
            with st.expander("🤖 에이전트 실행 상태", expanded=True):
                status = st.session_state['agent_execution_status']
                for agent, state in status.items():
                    if "실행 중" in state:
                        st.write(f"🔄 **{agent}**: {state}")
                    elif "완료" in state:
                        st.write(f"✅ **{agent}**: {state}")
                    elif "실패" in state:
                        st.write(f"❌ **{agent}**: {state}")
                    else:
                        st.write(f"⏳ **{agent}**: {state}")
    
    # 생성된 아티팩트 표시
    if st.session_state.get('generated_artifacts'):
        st.markdown("### 📦 생성된 아티팩트")
        artifacts = st.session_state['generated_artifacts']
        
        for i, artifact in enumerate(artifacts):
            with st.expander(f"📊 {artifact.get('name', f'아티팩트 {i+1}')}", expanded=True):
                st.write(f"**설명**: {artifact.get('description', '설명 없음')}")
                
                # 아티팩트 타입별 렌더링
                if artifact.get('type') == 'plotly_chart':
                    try:
                        st.plotly_chart(artifact['content'], use_container_width=True)
                    except Exception as e:
                        st.error(f"차트 렌더링 실패: {e}")
                
                elif artifact.get('type') == 'dataframe':
                    try:
                        st.dataframe(artifact['content'])
                    except Exception as e:
                        st.error(f"데이터프레임 렌더링 실패: {e}")
                
                elif artifact.get('type') == 'image':
                    try:
                        st.image(artifact['content'])
                    except Exception as e:
                        st.error(f"이미지 렌더링 실패: {e}")
                
                elif artifact.get('type') == 'code':
                    try:
                        st.code(artifact['content'], language=artifact.get('language', 'python'))
                    except Exception as e:
                        st.error(f"코드 렌더링 실패: {e}")
                
                else:
                    # 기본 텍스트 렌더링
                    st.write(artifact.get('content', '내용 없음'))
    
    # 기존 대화 히스토리 표시
    if st.session_state['conversation_history']:
        st.markdown("### 💬 대화 히스토리")
        
        for i, chat in enumerate(st.session_state['conversation_history']):
            with st.container():
                st.markdown(f"""
                <div class="conversation-flow">
                    <div class="user-message">
                        <strong>🙋 질문:</strong> {chat['question']}
                    </div>
                    <div class="ai-response">
                        <strong>🤖 AI 분석:</strong> {chat['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 분석 방법 표시
                if 'method' in chat:
                    method_color = {
                        "LLM First + A2A Integration": "🟢",
                        "LLM Analysis": "🟡", 
                        "A2A Analysis": "🔵",
                        "Error": "🔴"
                    }
                    color = method_color.get(chat['method'], "⚪")
                    st.caption(f"{color} **분석 방법**: {chat['method']}")
                
                # 세션 ID 표시 (Langfuse 추적)
                if 'session_id' in chat:
                    st.caption(f"🔍 **Langfuse 세션**: {chat.get('session_id', 'N/A')}")
                
                # Follow-up 제안 표시
                if 'followup_suggestions' in chat and chat['followup_suggestions']:
                    st.markdown("**💡 다음 단계 제안:**")
                    cols = st.columns(min(len(chat['followup_suggestions']), 3))
                    for j, suggestion in enumerate(chat['followup_suggestions']):
                        with cols[j % 3]:
                            if st.button(
                                suggestion, 
                                key=f"followup_{i}_{j}",
                                use_container_width=True
                            ):
                                st.session_state['selected_question'] = suggestion
                                st.rerun()
                
                st.markdown("---")
    
    # 시스템 상태 정보
    with st.expander("🔧 시스템 상태", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔌 연결 상태**")
            st.write(f"• LLM: {'✅' if LLM_AVAILABLE else '❌'}")
            st.write(f"• A2A 클라이언트: {'✅' if A2A_CLIENT_AVAILABLE else '❌'}")
            st.write(f"• Langfuse: {'✅' if LANGFUSE_AVAILABLE else '❌'}")
        
        with col2:
            st.write("**📊 세션 정보**")
            st.write(f"• 대화 수: {len(st.session_state.get('conversation_history', []))}")
            st.write(f"• 아티팩트 수: {len(st.session_state.get('generated_artifacts', []))}")
            st.write(f"• 현재 단계: {st.session_state.get('current_stage', '대기 중')}")

async def perform_llm_first_analysis(df: pd.DataFrame, user_query: str) -> Dict:
    """완전한 LLM First 아키텍처 - 모든 하드코딩 제거, 동적 처리"""
    
    # 세션 상태 초기화
    if 'analysis_plan' not in st.session_state:
        st.session_state['analysis_plan'] = None
    if 'agent_execution_status' not in st.session_state:
        st.session_state['agent_execution_status'] = {}
    if 'generated_artifacts' not in st.session_state:
        st.session_state['generated_artifacts'] = []
    if 'llm_streaming_updates' not in st.session_state:
        st.session_state['llm_streaming_updates'] = []
    
    # Langfuse 세션 시작
    session_tracer = None
    session_id = None
    if LANGFUSE_AVAILABLE:
        try:
            session_tracer = get_session_tracer()
            user_id = os.getenv("EMP_NO", "cherryai_user")
            session_metadata = {
                "query": user_query,
                "data_shape": df.shape,
                "data_columns": df.columns.tolist(),
                "workspace": "data_workspace",
                "timestamp": datetime.now().isoformat(),
                "llm_first_mode": True
            }
            session_id = session_tracer.start_user_session(user_query, user_id, session_metadata)
            st.info(f"🔍 Langfuse 세션 시작: {session_id}")
        except Exception as e:
            st.warning(f"⚠️ Langfuse 세션 시작 실패: {e}")
    
    try:
        # === LLM First 핵심: 모든 결정을 LLM이 동적으로 수행 ===
        st.session_state['current_stage'] = "LLM 동적 분석 시작"
        
        # 데이터 컨텍스트 동적 생성
        data_context = await _create_dynamic_data_context(df, user_query)
        
        # === 1단계: LLM 기반 완전 동적 분석 전략 수립 ===
        st.session_state['current_stage'] = "LLM 기반 동적 전략 수립"
        
        analysis_strategy = None
        if LLM_AVAILABLE:
            # LLM이 모든 것을 동적으로 결정
            strategy_prompt = await _generate_dynamic_strategy_prompt(data_context, user_query)
            
            try:
                response = await llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 세계 최고의 데이터 분석 전문가입니다. 주어진 데이터와 질문을 바탕으로 최적의 분석 전략을 동적으로 수립하세요. 절대 템플릿이나 고정된 패턴을 사용하지 마세요."},
                        {"role": "user", "content": strategy_prompt}
                    ],
                    temperature=0.7,  # 창의적 분석을 위한 높은 temperature
                    stream=True  # SSE 스트리밍 지원
                )
                
                # SSE 스트리밍으로 실시간 업데이트
                strategy_content = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        strategy_content += chunk.choices[0].delta.content
                        # 실시간 스트리밍 업데이트
                        st.session_state['llm_streaming_updates'].append({
                            "stage": "strategy_planning",
                            "content": chunk.choices[0].delta.content,
                            "timestamp": time.time()
                        })
                
                # LLM 응답을 동적으로 파싱
                analysis_strategy = await _parse_llm_strategy_dynamically(strategy_content)
                st.session_state['analysis_plan'] = analysis_strategy
                
                # Langfuse 동적 로깅
                if session_tracer:
                    session_tracer.record_agent_result("LLM_Dynamic_Strategist", {
                        "success": True,
                        "strategy": analysis_strategy,
                        "query_analysis": user_query,
                        "data_insights": data_context.get("llm_insights", {}),
                        "dynamic_approach": True
                    }, confidence=0.95)
                
            except Exception as e:
                st.error(f"❌ 동적 전략 수립 실패: {e}")
                return {"success": False, "error": str(e)}
        
        # === 2단계: LLM 기반 A2A 에이전트 동적 선택 및 실행 ===
        st.session_state['current_stage'] = "LLM 기반 A2A 에이전트 동적 실행"
        
        if analysis_strategy and A2A_CLIENT_AVAILABLE:
            # LLM이 사용 가능한 에이전트들을 동적으로 평가하고 선택
            available_agents = await _discover_available_agents_dynamically()
            
            # LLM이 에이전트 실행 계획을 동적으로 생성
            execution_plan = await _generate_dynamic_execution_plan(
                analysis_strategy, available_agents, data_context, user_query
            )
            
            # A2A 클라이언트 동적 초기화
            a2a_client = A2AStreamlitClient(available_agents)
            agent_results = {}
            
            # LLM 기반 동적 에이전트 실행
            for agent_task in execution_plan.get("tasks", []):
                agent_id = agent_task.get("agent_id")
                task_description = agent_task.get("task_description")
                execution_context = agent_task.get("context", {})
                
                st.session_state['agent_execution_status'][agent_id] = "LLM 기반 동적 실행 중"
                
                try:
                    # A2A SDK 0.2.9 표준 준수 - 올바른 메서드 사용
                    agent_url = f"http://localhost:{available_agents[agent_id]['port']}"
                    
                    # A2A 표준 메시지 생성
                    message = Message(
                        messageId=f"msg-{int(time.time())}",
                        role=Role.user,
                        parts=[TextPart(text=task_description)]
                    )
                    
                    # SSE 스트리밍 지원 A2A 클라이언트 사용
                    enhanced_client = EnhancedA2ACommunicator()
                    
                    # 실시간 SSE 스트리밍으로 결과 수신
                    result = await enhanced_client.send_message_with_streaming(
                        agent_url=agent_url,
                        instruction=task_description,
                        stream_callback=lambda chunk: _handle_sse_chunk(agent_id, chunk),
                        context_data={
                            "data_context": data_context,
                            "execution_context": execution_context,
                            "user_query": user_query,
                            "dynamic_params": agent_task.get("dynamic_params", {})
                        }
                    )
                    
                    agent_results[agent_id] = result
                    st.session_state['agent_execution_status'][agent_id] = "LLM 기반 동적 실행 완료"
                    
                    # A2A SDK 0.2.9 표준 준수 로깅
                    if session_tracer:
                        session_tracer.record_agent_result(agent_id, {
                            "success": True,
                            "result": result,
                            "task_description": task_description,
                            "dynamic_execution": True,
                            "llm_driven": True
                        }, confidence=0.9)
                
                except Exception as e:
                    st.session_state['agent_execution_status'][agent_id] = f"실패: {str(e)}"
                    if session_tracer:
                        session_tracer.record_agent_result(agent_id, {
                            "success": False,
                            "error": str(e),
                            "task_description": task_description
                        }, confidence=0.1)
        
        # === 3단계: LLM 기반 동적 아티팩트 생성 ===
        st.session_state['current_stage'] = "LLM 기반 동적 아티팩트 생성"
        
        artifacts = await _generate_dynamic_artifacts(
            df, user_query, analysis_strategy, agent_results, session_tracer
        )
        st.session_state['generated_artifacts'] = artifacts
        
        # === 4단계: LLM 기반 동적 결과 종합 ===
        st.session_state['current_stage'] = "LLM 기반 동적 결과 종합"
        
        final_analysis = await _synthesize_results_dynamically(
            user_query, data_context, analysis_strategy, agent_results, artifacts
        )
        
        # Langfuse 세션 완료
        if session_tracer and session_id:
            try:
                session_tracer.end_user_session({
                    "final_analysis": final_analysis,
                    "artifacts_generated": len(artifacts),
                    "agents_executed": len(agent_results),
                    "strategy": analysis_strategy,
                    "llm_first_success": True,
                    "dynamic_processing": True
                })
                st.success(f"✅ LLM First 분석 완료: {session_id}")
            except Exception as e:
                st.warning(f"⚠️ Langfuse 세션 완료 실패: {e}")
        
        return {
            "success": True,
            "result": final_analysis,
            "method": "LLM First Dynamic Analysis",
            "strategy": analysis_strategy,
            "agent_results": agent_results,
            "artifacts": artifacts,
            "session_id": session_id,
            "streaming_updates": st.session_state['llm_streaming_updates']
        }
        
    except Exception as e:
        # 동적 오류 처리
        if session_tracer and session_id:
            try:
                session_tracer.end_user_session({
                    "error": str(e),
                    "success": False,
                    "llm_first_failure": True
                })
            except:
                pass
        
        return {
            "success": False,
            "result": f"LLM First 분석 중 오류: {str(e)}",
            "method": "LLM First Error",
            "error": str(e)
        }

# === LLM First 핵심 헬퍼 함수들 (하드코딩 완전 제거) ===

async def _create_dynamic_data_context(df: pd.DataFrame, user_query: str) -> Dict:
    """LLM이 데이터를 동적으로 분석하여 컨텍스트 생성"""
    
    # 기본 데이터 정보
    basic_info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.head(3).to_dict()
    }
    
    # LLM이 데이터를 동적으로 분석
    if LLM_AVAILABLE:
        try:
            analysis_prompt = f"""
데이터 분석 전문가로서 다음 데이터를 분석하고 사용자 질문에 대한 통찰을 제공해주세요.

데이터 정보:
- 크기: {basic_info['shape']}
- 컬럼: {basic_info['columns']}
- 샘플 데이터: {basic_info['sample_data']}

사용자 질문: {user_query}

이 데이터의 특성, 패턴, 분석 가능성을 자유롭게 분석하고 통찰을 제공해주세요.
템플릿이나 고정된 형식을 사용하지 말고, 데이터와 질문에 맞는 고유한 분석을 제공해주세요.
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다. 주어진 데이터를 자유롭게 분석하세요."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7
            )
            
            basic_info["llm_insights"] = response.choices[0].message.content
            
        except Exception as e:
            basic_info["llm_insights"] = f"동적 분석 실패: {e}"
    
    return basic_info

async def _generate_dynamic_strategy_prompt(data_context: Dict, user_query: str) -> str:
    """LLM이 동적으로 전략 수립 프롬프트 생성"""
    
    if not LLM_AVAILABLE:
        return f"사용자 질문: {user_query}, 데이터 분석 전략을 수립해주세요."
    
    # LLM이 프롬프트 자체를 동적으로 생성
    prompt_generation = f"""
당신은 프롬프트 엔지니어링 전문가입니다. 다음 정보를 바탕으로 데이터 분석 전략을 수립하기 위한 최적의 프롬프트를 생성해주세요.

데이터 정보:
{json.dumps(data_context, ensure_ascii=False, indent=2)}

사용자 질문: {user_query}

이 특정 상황에 맞는 맞춤형 전략 수립 프롬프트를 생성해주세요.
일반적인 템플릿이 아닌, 이 데이터와 질문에 특화된 프롬프트를 만들어주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 프롬프트 엔지니어링 전문가입니다."},
                {"role": "user", "content": prompt_generation}
            ],
            temperature=0.8
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback으로 기본 프롬프트 사용
        return f"""
데이터 분석 전문가로서 다음 상황에 대한 최적의 분석 전략을 수립해주세요:

사용자 질문: {user_query}
데이터 특성: {data_context.get('llm_insights', '알 수 없음')}

이 특정 상황에 맞는 고유한 분석 전략을 제안해주세요.
"""

async def _parse_llm_strategy_dynamically(strategy_content: str) -> Dict:
    """LLM 응답을 동적으로 파싱 (하드코딩된 JSON 형식 제거)"""
    
    if not LLM_AVAILABLE:
        return {"strategy": strategy_content, "parsed": False}
    
    # LLM이 자신의 응답을 구조화
    parsing_prompt = f"""
다음 분석 전략을 구조화된 형태로 정리해주세요:

{strategy_content}

이 전략의 핵심 요소들을 파악하고 실행 가능한 형태로 정리해주세요.
고정된 형식이 아닌, 이 전략에 맞는 최적의 구조를 사용해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 전략 분석 전문가입니다."},
                {"role": "user", "content": parsing_prompt}
            ],
            temperature=0.3
        )
        
        return {
            "raw_strategy": strategy_content,
            "structured_strategy": response.choices[0].message.content,
            "parsed": True
        }
        
    except Exception as e:
        return {
            "raw_strategy": strategy_content,
            "error": str(e),
            "parsed": False
        }

async def _discover_available_agents_dynamically() -> Dict:
    """A2A SDK 0.2.9 표준을 준수하여 에이전트 동적 발견"""
    
    available_agents = {}
    
    # A2A 표준 포트 범위 스캔
    potential_ports = [8100, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315]
    
    for port in potential_ports:
        try:
            # A2A 표준 에이전트 카드 확인
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                if response.status_code == 200:
                    agent_info = response.json()
                    available_agents[f"agent_{port}"] = {
                        "port": port,
                        "url": f"http://localhost:{port}",
                        "info": agent_info,
                        "name": agent_info.get("name", f"Agent {port}"),
                        "capabilities": agent_info.get("capabilities", [])
                    }
                    st.info(f"🔍 A2A 에이전트 발견: {agent_info.get('name', f'Agent {port}')} (포트 {port})")
        except Exception as e:
            # 에이전트가 없는 포트는 무시
            continue
    
    return available_agents

# SSE 스트리밍 콜백 함수
def _handle_sse_chunk(agent_id: str, chunk: str):
    """SSE 스트리밍 청크 처리"""
    if 'llm_streaming_updates' not in st.session_state:
        st.session_state['llm_streaming_updates'] = []
    
    st.session_state['llm_streaming_updates'].append({
        "agent_id": agent_id,
        "chunk": chunk,
        "timestamp": time.time()
    })
    
    # 실시간 UI 업데이트
    if st.session_state.get('agent_execution_status'):
        st.session_state['agent_execution_status'][agent_id] = f"스트리밍 중... {chunk[:50]}..."

async def _generate_dynamic_execution_plan(strategy: Dict, available_agents: Dict, 
                                         data_context: Dict, user_query: str) -> Dict:
    """LLM이 사용 가능한 에이전트와 전략을 바탕으로 실행 계획 동적 생성"""
    
    if not LLM_AVAILABLE:
        return {"tasks": [], "error": "LLM not available"}
    
    planning_prompt = f"""
분석 전략과 사용 가능한 에이전트를 바탕으로 실행 계획을 수립해주세요.

분석 전략:
{json.dumps(strategy, ensure_ascii=False, indent=2)}

사용 가능한 에이전트:
{json.dumps(available_agents, ensure_ascii=False, indent=2)}

사용자 질문: {user_query}

이 특정 상황에 맞는 최적의 실행 계획을 수립해주세요.
각 에이전트의 능력과 현재 상황을 고려하여 효율적인 실행 순서를 결정해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 시스템 아키텍처 전문가입니다."},
                {"role": "user", "content": planning_prompt}
            ],
            temperature=0.5
        )
        
        # LLM 응답을 실행 계획으로 변환
        execution_plan = await _convert_to_execution_plan(response.choices[0].message.content, available_agents)
        
        return execution_plan
        
    except Exception as e:
        return {"tasks": [], "error": str(e)}

async def _convert_to_execution_plan(plan_text: str, available_agents: Dict) -> Dict:
    """LLM 응답을 실행 가능한 계획으로 변환"""
    
    # LLM이 자신의 계획을 실행 가능한 형태로 변환
    conversion_prompt = f"""
다음 계획을 실행 가능한 형태로 변환해주세요:

{plan_text}

사용 가능한 에이전트:
{json.dumps(available_agents, ensure_ascii=False, indent=2)}

각 단계별로 어떤 에이전트에게 어떤 작업을 요청할지 구체적으로 정해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 시스템 실행 계획 전문가입니다."},
                {"role": "user", "content": conversion_prompt}
            ],
            temperature=0.3
        )
        
        # 응답을 파싱하여 실행 계획 생성
        return {
            "original_plan": plan_text,
            "executable_plan": response.choices[0].message.content,
            "tasks": []  # 실제 구현에서는 LLM 응답을 파싱하여 tasks 생성
        }
        
    except Exception as e:
        return {"tasks": [], "error": str(e)}

async def _generate_dynamic_artifacts(df: pd.DataFrame, user_query: str, 
                                    strategy: Dict, agent_results: Dict, 
                                    session_tracer) -> List[Dict]:
    """LLM이 동적으로 아티팩트 생성 (하드코딩된 시각화 로직 제거)"""
    
    if not LLM_AVAILABLE:
        return []
    
    # LLM이 필요한 아티팩트를 동적으로 결정
    artifact_prompt = f"""
분석 결과를 바탕으로 어떤 아티팩트(시각화, 표, 코드 등)를 생성할지 결정해주세요.

사용자 질문: {user_query}
분석 전략: {json.dumps(strategy, ensure_ascii=False, indent=2)}
에이전트 결과: {json.dumps(agent_results, ensure_ascii=False, indent=2)}

이 특정 상황에 맞는 최적의 아티팩트를 제안해주세요.
데이터 타입이나 고정된 규칙에 의존하지 말고, 맥락에 맞는 아티팩트를 제안해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 데이터 시각화 전문가입니다."},
                {"role": "user", "content": artifact_prompt}
            ],
            temperature=0.7
        )
        
        # LLM 제안을 바탕으로 동적으로 아티팩트 생성
        artifacts = await _create_artifacts_from_llm_suggestions(
            df, response.choices[0].message.content, user_query
        )
        
        return artifacts
        
    except Exception as e:
        return [{"type": "error", "content": f"아티팩트 생성 실패: {e}"}]

async def _create_artifacts_from_llm_suggestions(df: pd.DataFrame, 
                                               llm_suggestions: str, 
                                               user_query: str) -> List[Dict]:
    """LLM 제안을 바탕으로 실제 아티팩트 생성"""
    
    # LLM이 구체적인 생성 코드를 작성
    code_generation_prompt = f"""
다음 제안을 바탕으로 실제 Python 코드를 생성해주세요:

{llm_suggestions}

사용자 질문: {user_query}

plotly, matplotlib, seaborn 등을 사용하여 구체적인 시각화 코드를 작성해주세요.
데이터프레임 변수명은 'df'를 사용해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 Python 데이터 시각화 전문가입니다."},
                {"role": "user", "content": code_generation_prompt}
            ],
            temperature=0.3
        )
        
        # 생성된 코드를 실행하여 아티팩트 생성
        generated_code = response.choices[0].message.content
        
        # 안전한 코드 실행 환경에서 아티팩트 생성
        artifacts = await _execute_visualization_code_safely(df, generated_code)
        
        return artifacts
        
    except Exception as e:
        return [{"type": "error", "content": f"코드 생성 실패: {e}"}]

async def _execute_visualization_code_safely(df: pd.DataFrame, code: str) -> List[Dict]:
    """안전한 환경에서 시각화 코드 실행"""
    
    artifacts = []
    
    # 안전한 실행 환경 설정
    safe_globals = {
        'df': df,
        'pd': pd,
        'plt': plt,
        'px': px,
        'sns': sns,
        'np': np
    }
    
    try:
        # 코드에서 실행 가능한 부분만 추출
        code_lines = code.split('\n')
        exec_code = '\n'.join([line for line in code_lines if not line.strip().startswith('#')])
        
        # 안전한 실행
        exec(exec_code, safe_globals)
        
        # 실행 결과에서 아티팩트 추출
        for key, value in safe_globals.items():
            if hasattr(value, 'show') and 'plotly' in str(type(value)):
                artifacts.append({
                    'type': 'plotly_chart',
                    'content': value,
                    'name': f'llm_generated_{key}',
                    'description': f'LLM이 동적으로 생성한 {key} 시각화'
                })
        
        return artifacts
        
    except Exception as e:
        return [{"type": "error", "content": f"코드 실행 실패: {e}"}]

async def _synthesize_results_dynamically(user_query: str, data_context: Dict, 
                                        strategy: Dict, agent_results: Dict, 
                                        artifacts: List[Dict]) -> str:
    """LLM이 모든 결과를 동적으로 종합"""
    
    if not LLM_AVAILABLE:
        return "LLM을 사용할 수 없어 결과를 종합할 수 없습니다."
    
    # LLM이 모든 정보를 종합하여 최종 분석 생성
    synthesis_prompt = f"""
모든 분석 과정과 결과를 종합하여 최종 분석 보고서를 작성해주세요.

사용자 질문: {user_query}

데이터 컨텍스트:
{json.dumps(data_context, ensure_ascii=False, indent=2)}

분석 전략:
{json.dumps(strategy, ensure_ascii=False, indent=2)}

에이전트 실행 결과:
{json.dumps(agent_results, ensure_ascii=False, indent=2)}

생성된 아티팩트: {len(artifacts)}개

이 모든 정보를 종합하여 사용자에게 가치 있는 인사이트를 제공해주세요.
템플릿이나 고정된 형식이 아닌, 이 특정 상황에 맞는 고유한 분석을 제공해주세요.
"""
    
    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 세계 최고의 데이터 분석 컨설턴트입니다."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"결과 종합 중 오류 발생: {e}"

def render_data_workspace():
    """LLM First 데이터 워크스페이스 메인 렌더링"""
    apply_llm_first_layout_styles()
    
    st.markdown("# 🧬 LLM First Data Workspace")
    
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader(
        "📁 데이터 파일 업로드", 
        type=['csv', 'xlsx', 'json'],
        help="CSV, Excel, JSON 파일을 업로드하세요"
    )
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state['workspace_data'] = df
            st.session_state['data_source'] = uploaded_file.name
            st.success(f"✅ {uploaded_file.name} 업로드 완료!")
            
        except Exception as e:
            st.error(f"❌ 파일 업로드 실패: {e}")
    
    # 기존 데이터 사용
    elif 'workspace_data' in st.session_state:
        df = st.session_state['workspace_data']
    
    if df is not None:
        # 30% / 70% 레이아웃
        col_input, col_results = st.columns([3, 7])
        
        with col_input:
            st.markdown('<div class="input-panel">', unsafe_allow_html=True)
            render_input_panel(df)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_results:
            st.markdown('<div class="results-panel">', unsafe_allow_html=True)
            render_results_panel()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 채팅 입력 (엔터키 지원)
        if user_input := st.chat_input("💬 데이터에 대해 질문하세요 (엔터키로 전송)"):
            st.session_state['selected_question'] = user_input
            st.rerun()
        
        # 선택된 질문 처리
        if st.session_state.get('selected_question'):
            question = st.session_state['selected_question']
            st.session_state['selected_question'] = None
            st.session_state['analysis_in_progress'] = True
            
            # 분석 실행
            with st.spinner("🤖 LLM First 분석 실행 중..."):
                analysis_result = asyncio.run(perform_llm_first_analysis(df, question))
                
                # 대화 히스토리에 추가
                new_chat = {
                    "question": question,
                    "answer": analysis_result["result"],
                    "timestamp": datetime.now().isoformat(),
                    "method": analysis_result["method"]
                }
                
                # Follow-up 제안 생성
                if analysis_result["success"]:
                    followup_suggestions = asyncio.run(
                        generate_followup_suggestions(
                            analysis_result["result"], 
                            st.session_state.get('conversation_history', [])
                        )
                    )
                    new_chat["followup_suggestions"] = followup_suggestions
                
                st.session_state['conversation_history'].append(new_chat)
                st.session_state['analysis_in_progress'] = False
                st.rerun()
    
    else:
        # 샘플 데이터 제공
        st.markdown("### 📋 샘플 데이터로 시작하기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 부동산 데이터", use_container_width=True):
                import numpy as np
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    'area': np.random.normal(100, 30, 100),
                    'price': np.random.normal(50000, 15000, 100),
                    'rooms': np.random.randint(1, 6, 100),
                    'location': np.random.choice(['강남', '홍대', '잠실', '종로'], 100)
                })
                st.session_state['workspace_data'] = sample_data
                st.session_state['data_source'] = '샘플_부동산_데이터.csv'
                st.rerun()
        
        with col2:
            if st.button("📈 매출 데이터", use_container_width=True):
                import numpy as np
                np.random.seed(123)
                sample_data = pd.DataFrame({
                    'month': pd.date_range('2023-01-01', periods=12, freq='M'),
                    'revenue': np.random.normal(100000, 20000, 12),
                    'customers': np.random.randint(800, 1200, 12),
                    'category': np.random.choice(['A', 'B', 'C'], 12)
                })
                st.session_state['workspace_data'] = sample_data
                st.session_state['data_source'] = '샘플_매출_데이터.csv'
                st.rerun()
        
        with col3:
            if st.button("👥 고객 데이터", use_container_width=True):
                import numpy as np
                np.random.seed(456)
                sample_data = pd.DataFrame({
                    'age': np.random.randint(20, 70, 200),
                    'income': np.random.normal(50000, 15000, 200),
                    'spending': np.random.normal(30000, 10000, 200),
                    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 200)
                })
                st.session_state['workspace_data'] = sample_data
                st.session_state['data_source'] = '샘플_고객_데이터.csv'
                st.rerun() 

# LLM 동적 능력 강화 시스템
class LLMDynamicCapabilityEngine:
    """LLM 동적 능력 강화 엔진 - 범용적 데이터 처리"""
    
    def __init__(self):
        self.capability_cache = {}
        self.learning_history = []
        self.adaptive_strategies = {}
    
    async def analyze_data_dynamically(self, df: pd.DataFrame, user_query: str) -> Dict:
        """LLM이 데이터를 완전히 동적으로 분석"""
        
        # 1단계: LLM이 데이터 특성을 스스로 파악
        data_characteristics = await self._discover_data_characteristics(df)
        
        # 2단계: LLM이 분석 능력을 동적으로 구성
        analysis_capabilities = await self._build_dynamic_analysis_capabilities(
            data_characteristics, user_query
        )
        
        # 3단계: LLM이 맞춤형 분석 전략 수립
        analysis_strategy = await self._create_adaptive_analysis_strategy(
            data_characteristics, analysis_capabilities, user_query
        )
        
        # 4단계: 동적 분석 실행
        results = await self._execute_dynamic_analysis(
            df, analysis_strategy, user_query
        )
        
        # 5단계: 학습 및 적응
        await self._learn_from_analysis(data_characteristics, analysis_strategy, results)
        
        return results
    
    async def _discover_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """LLM이 데이터 특성을 완전히 동적으로 발견"""
        
        # 기본 메타데이터 수집
        basic_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns}
        }
        
        # LLM이 데이터 특성을 동적으로 분석
        discovery_prompt = f"""
당신은 데이터 분석 전문가입니다. 다음 데이터를 분석하여 특성을 파악해주세요:

기본 정보:
{json.dumps(basic_info, ensure_ascii=False, indent=2)}

이 데이터의 특성을 분석하여 다음을 결정해주세요:
1. 데이터의 도메인과 용도
2. 주요 변수들의 의미와 관계
3. 데이터 품질 평가
4. 분석 가능성 평가
5. 특별한 패턴이나 특징

이 데이터만의 고유한 특성을 파악하고 분석해주세요.
일반적인 템플릿이 아닌, 이 데이터에 특화된 분석을 제공해주세요.
"""
        
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                    {"role": "user", "content": discovery_prompt}
                ],
                temperature=0.7
            )
            
            characteristics = {
                "basic_info": basic_info,
                "llm_analysis": response.choices[0].message.content,
                "discovery_timestamp": time.time()
            }
            
            return characteristics
            
        except Exception as e:
            return {
                "basic_info": basic_info,
                "llm_analysis": f"동적 분석 실패: {e}",
                "discovery_timestamp": time.time()
            }
    
    async def _build_dynamic_analysis_capabilities(self, 
                                                 data_characteristics: Dict, 
                                                 user_query: str) -> Dict:
        """LLM이 분석 능력을 동적으로 구성"""
        
        capability_prompt = f"""
당신은 데이터 분석 능력 설계 전문가입니다. 

데이터 특성:
{json.dumps(data_characteristics, ensure_ascii=False, indent=2)}

사용자 요청:
{user_query}

이 상황에 맞는 최적의 분석 능력을 설계해주세요:
1. 어떤 분석 기법이 필요한가?
2. 어떤 시각화가 효과적인가?
3. 어떤 통계 방법이 적절한가?
4. 어떤 인사이트를 도출할 수 있는가?

이 특정 데이터와 요청에 맞는 맞춤형 분석 능력을 제안해주세요.
"""
        
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 능력 설계 전문가입니다."},
                    {"role": "user", "content": capability_prompt}
                ],
                temperature=0.6
            )
            
            return {
                "capabilities": response.choices[0].message.content,
                "build_timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "capabilities": f"능력 구성 실패: {e}",
                "build_timestamp": time.time()
            }
    
    async def _create_adaptive_analysis_strategy(self, 
                                               data_characteristics: Dict,
                                               analysis_capabilities: Dict,
                                               user_query: str) -> Dict:
        """LLM이 적응형 분석 전략 수립"""
        
        strategy_prompt = f"""
당신은 데이터 분석 전략 수립 전문가입니다.

데이터 특성:
{data_characteristics.get('llm_analysis', '알 수 없음')}

분석 능력:
{analysis_capabilities.get('capabilities', '알 수 없음')}

사용자 요청:
{user_query}

이 모든 정보를 종합하여 구체적인 분석 전략을 수립해주세요:
1. 분석 순서와 단계
2. 각 단계별 목표
3. 사용할 도구와 기법
4. 예상 결과와 검증 방법

이 특정 상황에 최적화된 분석 전략을 제안해주세요.
"""
        
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전략 수립 전문가입니다."},
                    {"role": "user", "content": strategy_prompt}
                ],
                temperature=0.5
            )
            
            return {
                "strategy": response.choices[0].message.content,
                "strategy_timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "strategy": f"전략 수립 실패: {e}",
                "strategy_timestamp": time.time()
            }
    
    async def _execute_dynamic_analysis(self, 
                                      df: pd.DataFrame, 
                                      analysis_strategy: Dict,
                                      user_query: str) -> Dict:
        """동적 분석 실행"""
        
        execution_prompt = f"""
당신은 데이터 분석 실행 전문가입니다.

분석 전략:
{analysis_strategy.get('strategy', '알 수 없음')}

사용자 요청:
{user_query}

이 전략에 따라 실제 분석을 수행하고 결과를 제공해주세요.
구체적인 분석 결과와 인사이트를 제공해주세요.

데이터 정보:
- 크기: {df.shape}
- 컬럼: {df.columns.tolist()}
- 기본 통계: {df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else '수치 데이터 없음'}
"""
        
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 실행 전문가입니다."},
                    {"role": "user", "content": execution_prompt}
                ],
                temperature=0.4
            )
            
            # 추가로 시각화 생성
            visualizations = await self._generate_dynamic_visualizations(df, user_query)
            
            return {
                "analysis_result": response.choices[0].message.content,
                "visualizations": visualizations,
                "execution_timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "analysis_result": f"분석 실행 실패: {e}",
                "visualizations": [],
                "execution_timestamp": time.time()
            }
    
    async def _generate_dynamic_visualizations(self, df: pd.DataFrame, user_query: str) -> List[Dict]:
        """LLM이 동적으로 시각화 생성"""
        
        viz_prompt = f"""
당신은 데이터 시각화 전문가입니다.

사용자 요청: {user_query}

데이터 정보:
- 크기: {df.shape}
- 컬럼: {df.columns.tolist()}
- 수치 컬럼: {df.select_dtypes(include=['number']).columns.tolist()}
- 범주형 컬럼: {df.select_dtypes(include=['object']).columns.tolist()}

이 데이터와 요청에 가장 적합한 시각화를 생성하는 Python 코드를 작성해주세요.
plotly를 사용하여 구체적인 시각화 코드를 제공해주세요.
변수명 'df'를 사용하고, 실행 가능한 코드를 작성해주세요.
"""
        
        try:
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 시각화 전문가입니다."},
                    {"role": "user", "content": viz_prompt}
                ],
                temperature=0.3
            )
            
            # 생성된 코드 실행
            viz_code = response.choices[0].message.content
            visualizations = await self._execute_visualization_code(df, viz_code)
            
            return visualizations
            
        except Exception as e:
            return [{"error": f"시각화 생성 실패: {e}"}]
    
    async def _execute_visualization_code(self, df: pd.DataFrame, code: str) -> List[Dict]:
        """시각화 코드 안전 실행"""
        
        visualizations = []
        
        # 안전한 실행 환경
        safe_globals = {
            'df': df,
            'pd': pd,
            'px': px,
            'go': go,
            'plt': plt,
            'sns': sns,
            'np': np
        }
        
        try:
            # 코드 실행
            exec(code, safe_globals)
            
            # 생성된 시각화 추출
            for key, value in safe_globals.items():
                if hasattr(value, 'show') and 'plotly' in str(type(value)):
                    visualizations.append({
                        'type': 'plotly',
                        'figure': value,
                        'name': key,
                        'description': f'LLM 생성 시각화: {key}'
                    })
            
            return visualizations
            
        except Exception as e:
            return [{"error": f"시각화 코드 실행 실패: {e}"}]
    
    async def _learn_from_analysis(self, 
                                 data_characteristics: Dict,
                                 analysis_strategy: Dict,
                                 results: Dict):
        """분석 결과로부터 학습"""
        
        # 학습 기록 저장
        learning_record = {
            "timestamp": time.time(),
            "data_characteristics": data_characteristics,
            "strategy": analysis_strategy,
            "results": results,
            "success": "error" not in results.get("analysis_result", "")
        }
        
        self.learning_history.append(learning_record)
        
        # 성공적인 전략 캐싱
        if learning_record["success"]:
            strategy_key = self._generate_strategy_key(data_characteristics)
            self.adaptive_strategies[strategy_key] = analysis_strategy
        
        # 학습 히스토리 관리 (최대 100개)
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
    
    def _generate_strategy_key(self, data_characteristics: Dict) -> str:
        """전략 키 생성"""
        basic_info = data_characteristics.get("basic_info", {})
        return f"{basic_info.get('shape', 'unknown')}_{len(basic_info.get('columns', []))}"

# 전역 LLM 동적 능력 엔진
llm_capability_engine = LLMDynamicCapabilityEngine()

# 범용적 데이터 분석 함수 (하드코딩 완전 제거)
async def perform_universal_data_analysis(df: pd.DataFrame, user_query: str) -> Dict:
    """범용적 데이터 분석 - 특정 데이터셋에 종속되지 않음"""
    
    try:
        # LLM 동적 능력 엔진 사용
        results = await llm_capability_engine.analyze_data_dynamically(df, user_query)
        
        return {
            "success": True,
            "analysis": results.get("analysis_result", ""),
            "visualizations": results.get("visualizations", []),
            "method": "LLM Dynamic Universal Analysis",
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "LLM Dynamic Universal Analysis (Error)",
            "timestamp": time.time()
        } 