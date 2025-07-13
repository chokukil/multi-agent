"""
🍒 CherryAI - Realtime Chat Container
실시간 스트리밍 채팅 컨테이너 모듈

A2A + MCP 통합 실시간 스트리밍을 위한 ChatGPT/Claude 스타일 채팅 인터페이스
"""

import streamlit as st
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class StreamingMessage:
    """스트리밍 메시지 데이터 클래스"""
    message_id: str
    source: str  # 'a2a' or 'mcp' or 'user' or 'assistant'
    agent_type: str  # 'pandas', 'visualization', 'orchestrator', etc.
    content: str
    metadata: Dict[str, Any]
    status: str  # 'streaming', 'completed', 'error'
    timestamp: float
    is_final: bool = False

class RealtimeChatContainer:
    """실시간 채팅 컨테이너 - ChatGPT/Claude 스타일"""
    
    def __init__(self, container_key: str = "main_chat"):
        self.container_key = container_key
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.messages: List[StreamingMessage] = []
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        session_key = f"{self.container_key}_messages"
        if session_key not in st.session_state:
            st.session_state[session_key] = []
        
        active_streams_key = f"{self.container_key}_active_streams"
        if active_streams_key not in st.session_state:
            st.session_state[active_streams_key] = {}
            
        self.messages = st.session_state[session_key]
        self.active_streams = st.session_state[active_streams_key]
    
    def add_user_message(self, content: str) -> str:
        """사용자 메시지 추가"""
        message_id = f"user_{int(time.time() * 1000)}"
        
        message = StreamingMessage(
            message_id=message_id,
            source="user",
            agent_type="user",
            content=content,
            metadata={"type": "user_input"},
            status="completed",
            timestamp=time.time(),
            is_final=True
        )
        
        self.messages.append(message)
        self._save_session_state()
        return message_id
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> str:
        """어시스턴트 메시지 추가"""
        message_id = f"assistant_{int(time.time() * 1000)}"
        
        message = StreamingMessage(
            message_id=message_id,
            source="assistant",
            agent_type="assistant",
            content=content,
            metadata=metadata or {},
            status="completed",
            timestamp=time.time(),
            is_final=True
        )
        
        self.messages.append(message)
        self._save_session_state()
        return message_id
    
    def add_streaming_message(self, source: str, agent_type: str, initial_content: str = "") -> str:
        """실시간 스트리밍 메시지 컨테이너 추가"""
        message_id = f"{source}_{agent_type}_{int(time.time() * 1000)}"
        
        # 스트리밍 메시지 생성
        message = StreamingMessage(
            message_id=message_id,
            source=source,
            agent_type=agent_type,
            content=initial_content,
            metadata={
                'type': 'streaming',
                'source': source,
                'agent_type': agent_type,
                'started_at': time.time()
            },
            status="streaming",
            timestamp=time.time(),
            is_final=False
        )
        
        self.messages.append(message)
        
        # 활성 스트림 추적
        self.active_streams[message_id] = {
            'placeholder': None,  # Streamlit placeholder는 나중에 설정
            'message': message,
            'last_update': time.time()
        }
        
        self._save_session_state()
        return message_id
    
    def update_streaming_message(self, message_id: str, chunk: str, is_final: bool = False):
        """스트리밍 메시지 실시간 업데이트"""
        if message_id not in self.active_streams:
            return
        
        # 메시지 찾기
        message = None
        for msg in self.messages:
            if msg.message_id == message_id:
                message = msg
                break
        
        if message:
            # 컨텐츠 누적
            message.content += chunk
            message.is_final = is_final
            message.metadata['last_update'] = time.time()
            
            # 활성 스트림 업데이트 (finalize 전에 수행)
            if message_id in self.active_streams:
                self.active_streams[message_id]['last_update'] = time.time()
            
            if is_final:
                message.status = "completed"
                # 활성 스트림에 메시지 업데이트 (finalize 전에 수행)
                if message_id in self.active_streams:
                    self.active_streams[message_id]['message'] = message
                self.finalize_streaming_message(message_id)
            else:
                message.status = "streaming"
                # 활성 스트림에 메시지 업데이트
                if message_id in self.active_streams:
                    self.active_streams[message_id]['message'] = message
            
            self._save_session_state()
    
    def finalize_streaming_message(self, message_id: str):
        """스트리밍 메시지 완료 처리"""
        if message_id in self.active_streams:
            message = self.active_streams[message_id]['message']
            message.status = "completed"
            message.is_final = True
            message.metadata['completed_at'] = time.time()
            
            # 활성 스트림에서 제거
            del self.active_streams[message_id]
            
            self._save_session_state()
    
    def render(self):
        """채팅 컨테이너 렌더링"""
        
        # 컨테이너 스타일 (ChatGPT/Claude 스타일)
        st.markdown("""
        <style>
        .cherry-chat-container {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #30363d;
        }
        
        .cherry-message {
            margin: 0.75rem 0;
            padding: 1rem;
            border-radius: 8px;
            max-width: 100%;
        }
        
        .cherry-message-user {
            background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
            color: white;
            margin-left: 20%;
        }
        
        .cherry-message-assistant {
            background: linear-gradient(135deg, #da3633 0%, #a21e1e 100%);
            color: white;
            margin-right: 20%;
        }
        
        .cherry-message-streaming {
            background: linear-gradient(135deg, #6f42c1 0%, #5a2d91 100%);
            color: white;
            margin-right: 20%;
            border-left: 4px solid #79c0ff;
        }
        
        .cherry-message-header {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            opacity: 0.9;
        }
        
        .cherry-agent-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
        }
        
        .cherry-status-indicator {
            font-size: 0.8rem;
            opacity: 0.8;
        }
        
        .cherry-streaming-indicator {
            animation: pulse 1.5s ease-in-out infinite alternate;
        }
        
        @keyframes pulse {
            from { opacity: 0.5; }
            to { opacity: 1.0; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 영역
        with st.container():
            st.markdown('<div class="cherry-chat-container">', unsafe_allow_html=True)
            
            # 메시지 렌더링
            for message in self.messages:
                self._render_message(message)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 자동 스크롤 (최신 메시지로)
        if self.messages:
            st.markdown("""
            <script>
            // 자동 스크롤을 위한 JavaScript
            setTimeout(function() {
                window.parent.document.querySelector('.main').scrollTop = 
                window.parent.document.querySelector('.main').scrollHeight;
            }, 100);
            </script>
            """, unsafe_allow_html=True)
    
    def _render_message(self, message: StreamingMessage):
        """개별 메시지 렌더링"""
        
        # 메시지 타입별 스타일 결정
        if message.source == "user":
            css_class = "cherry-message-user"
            header_icon = "👤"
        elif message.status == "streaming":
            css_class = "cherry-message-streaming"
            header_icon = "🔄"
        else:
            css_class = "cherry-message-assistant"
            header_icon = "🍒"
        
        # 상태 표시기
        if message.status == "streaming":
            status_indicator = '<span class="cherry-streaming-indicator">🔄 스트리밍 중...</span>'
        elif message.status == "completed":
            status_indicator = '✅ 완료'
        elif message.status == "error":
            status_indicator = '❌ 오류'
        else:
            status_indicator = ""
        
        # 에이전트 배지 (A2A/MCP용)
        agent_badge = ""
        if message.source in ["a2a", "mcp"]:
            agent_badge = f'<span class="cherry-agent-badge">{message.agent_type}</span>'
        
        # 메시지 헤더
        header = f"""
        <div class="cherry-message-header">
            {header_icon} {agent_badge}
            <span class="cherry-status-indicator">{status_indicator}</span>
        </div>
        """
        
        # 메시지 렌더링
        st.markdown(f"""
        <div class="cherry-message {css_class}">
            {header}
            <div class="cherry-message-content">
                {message.content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 특별한 메타데이터 처리 (아티팩트, 계획 등)
        if message.metadata.get('type') == 'orchestrator_plan':
            self._render_orchestrator_plan(message.metadata.get('plan', {}))
        elif message.metadata.get('type') == 'artifacts':
            self._render_artifacts_preview(message.metadata.get('artifacts', []))
    
    def _render_orchestrator_plan(self, plan: Dict[str, Any]):
        """오케스트레이터 계획 렌더링"""
        if not plan:
            return
            
        with st.expander("📋 실행 계획 상세", expanded=False):
            st.markdown(f"**분석 전략:** {plan.get('strategy', 'N/A')}")
            
            if plan.get('steps'):
                st.markdown("**실행 단계:**")
                for i, step in enumerate(plan['steps'], 1):
                    st.markdown(f"{i}. {step}")
            
            if plan.get('tools'):
                st.markdown("**사용 도구:**")
                for tool in plan['tools']:
                    st.markdown(f"• {tool}")
    
    def _render_artifacts_preview(self, artifacts: List[Dict]):
        """아티팩트 미리보기 렌더링"""
        if not artifacts:
            return
            
        st.markdown("**생성된 아티팩트:**")
        for artifact in artifacts:
            st.markdown(f"📊 {artifact.get('name', 'Unnamed Artifact')}")
    
    def _save_session_state(self):
        """세션 상태 저장"""
        session_key = f"{self.container_key}_messages"
        st.session_state[session_key] = self.messages
        
        active_streams_key = f"{self.container_key}_active_streams"
        st.session_state[active_streams_key] = self.active_streams
    
    def get_messages(self) -> List[StreamingMessage]:
        """모든 메시지 반환"""
        return self.messages
    
    def clear_messages(self):
        """모든 메시지 정리"""
        self.messages.clear()
        self.active_streams.clear()
        self._save_session_state()
    
    def get_active_streams_count(self) -> int:
        """활성 스트림 수 반환"""
        return len(self.active_streams)
    
    def cleanup_inactive_streams(self, timeout_seconds: int = 300):
        """비활성 스트림 정리 (5분 타임아웃)"""
        current_time = time.time()
        inactive_streams = []
        
        for stream_id, stream_data in self.active_streams.items():
            last_update = stream_data.get('last_update', 0)
            if current_time - last_update > timeout_seconds:
                inactive_streams.append(stream_id)
        
        for stream_id in inactive_streams:
            self.finalize_streaming_message(stream_id) 