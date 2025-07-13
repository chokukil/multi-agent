"""
🍒 CherryAI 현대적 채팅 인터페이스
Tailwind CSS + Stylable Containers 기반
LLM First 철학을 반영한 세계 최고 수준의 A2A + MCP 플랫폼 UI
"""
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from typing import List, Dict, Any, Optional
import time
import uuid
from datetime import datetime
import json
import re
import html

class ChatMessage:
    """채팅 메시지 클래스 - CherryAI 최적화"""
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.message_id = message_id or str(uuid.uuid4())
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'message_id': self.message_id,
            'metadata': self.metadata
        }

class CherryAIChatInterface:
    """🍒 CherryAI 현대적 채팅 인터페이스"""
    
    def __init__(self, session_key: str = "cherry_ai_chat"):
        self.session_key = session_key
        self.messages_key = f"{session_key}_messages"
        
        # 세션 상태 초기화
        if self.messages_key not in st.session_state:
            st.session_state[self.messages_key] = []
        
        # Tailwind CSS 초기화
        self._inject_tailwind_css()
    
    def _inject_tailwind_css(self):
        """Tailwind CSS + CherryAI 브랜딩 시스템 주입"""
        st.markdown("""
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            'cherry': {
                                50: '#fef2f2',
                                100: '#fee2e2',
                                200: '#fecaca',
                                300: '#fca5a5',
                                400: '#f87171',
                                500: '#ef4444',
                                600: '#dc2626',
                                700: '#b91c1c',
                                800: '#991b1b',
                                900: '#7f1d1d'
                            },
                            'ai': {
                                50: '#eff6ff',
                                100: '#dbeafe',
                                200: '#bfdbfe',
                                300: '#93c5fd',
                                400: '#60a5fa',
                                500: '#3b82f6', 
                                600: '#2563eb',
                                700: '#1d4ed8',
                                800: '#1e40af',
                                900: '#1e3a8a'
                            },
                            'a2a': {
                                50: '#ecfdf5',
                                100: '#d1fae5',
                                200: '#a7f3d0',
                                300: '#6ee7b7',
                                400: '#34d399',
                                500: '#10b981',
                                600: '#059669', 
                                700: '#047857',
                                800: '#065f46',
                                900: '#064e3b'
                            }
                        },
                        fontFamily: {
                            'cherry': ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif']
                        },
                        animation: {
                            'cherry-bounce': 'bounce 1s infinite',
                            'fade-in': 'fadeIn 0.5s ease-out',
                            'slide-up': 'slideUp 0.3s ease-out'
                        },
                        keyframes: {
                            fadeIn: {
                                '0%': { opacity: '0', transform: 'translateY(10px)' },
                                '100%': { opacity: '1', transform: 'translateY(0)' }
                            },
                            slideUp: {
                                '0%': { transform: 'translateY(20px)', opacity: '0' },
                                '100%': { transform: 'translateY(0)', opacity: '1' }
                            }
                        }
                    }
                }
            }
        </script>
        """, unsafe_allow_html=True)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """메시지 추가"""
        message = ChatMessage(role, content, metadata=metadata)
        st.session_state[self.messages_key].append(message)
    
    def add_user_message(self, content: str) -> None:
        """사용자 메시지 추가"""
        self.add_message("🧑🏻", content)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """어시스턴트 메시지 추가"""
        self.add_message("🍒", content, metadata)
    
    def add_system_message(self, content: str) -> None:
        """시스템 메시지 추가"""
        self.add_message("⚙️", content)
    
    def add_message_with_inline_suggestions(self, content: str, suggestions: List[str] = None) -> None:
        """LLM 생성 메시지와 인라인 제안 버튼들을 함께 추가"""
        metadata = {}
        if suggestions:
            metadata = {
                'type': 'message_with_suggestions',
                'suggestions': suggestions
            }
        self.add_assistant_message(content, metadata)
    
    def _process_content_for_display(self, content: str) -> str:
        """컨텐츠 처리 - Markdown + HTML 지원"""
        if not content.strip():
            return ""
            
        # HTML 이스케이프 처리
        processed = html.escape(content)
        
        # 마크다운 처리
        processed = re.sub(r'\*\*(.*?)\*\*', r'<span class="font-bold text-cherry-600">\1</span>', processed)
        processed = re.sub(r'\*(.*?)\*', r'<span class="italic text-ai-600">\1</span>', processed)
        processed = re.sub(r'`([^`]+)`', r'<code class="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-cherry-700">\1</code>', processed)
        
        # 헤더 처리
        processed = re.sub(r'^### (.*?)$', r'<h3 class="text-lg font-semibold text-ai-700 mt-4 mb-2">\1</h3>', processed, flags=re.MULTILINE)
        processed = re.sub(r'^## (.*?)$', r'<h2 class="text-xl font-bold text-ai-800 mt-4 mb-2">\1</h2>', processed, flags=re.MULTILINE)
        processed = re.sub(r'^# (.*?)$', r'<h1 class="text-2xl font-bold text-ai-900 mt-4 mb-2">\1</h1>', processed, flags=re.MULTILINE)
        
        # 목록 처리
        lines = processed.split('\n')
        in_list = False
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('* '):
                if not in_list:
                    result_lines.append('<ul class="list-disc list-inside ml-4 space-y-1">')
                    in_list = True
                result_lines.append(f'<li class="text-gray-700">{stripped[2:]}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                if stripped:
                    result_lines.append(f'<p class="mb-2">{line}</p>')
                else:
                    result_lines.append('<br>')
        
        if in_list:
            result_lines.append('</ul>')
        
        return '\n'.join(result_lines)
    
    def _render_user_message(self, message: ChatMessage) -> None:
        """사용자 메시지 렌더링"""
        timestamp = message.timestamp.strftime("%H:%M")
        
        with stylable_container(
            key=f"user_msg_{message.message_id}",
            css_styles="""
            div[data-testid="stVerticalBlock"] > div:first-child {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 0.75rem;
            }
            """
        ):
            st.markdown(f"""
            <div class="max-w-3xl">
                <div class="bg-gradient-to-r from-blue-700 to-blue-800 text-white rounded-2xl rounded-tr-sm px-3 py-2.5 shadow-lg animate-fade-in border border-blue-600">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-lg">{message.role}</span>
                        <span class="text-xs opacity-75">{timestamp}</span>
                    </div>
                    <div class="text-white leading-relaxed">
                        {self._process_content_for_display(message.content)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_assistant_message(self, message: ChatMessage) -> None:
        """어시스턴트 메시지 렌더링"""
        timestamp = message.timestamp.strftime("%H:%M")
        metadata = message.metadata or {}
        
        with stylable_container(
            key=f"assistant_msg_{message.message_id}",
            css_styles="""
            div[data-testid="stVerticalBlock"] > div:first-child {
                display: flex;
                justify-content: flex-start;
                margin-bottom: 0.75rem;
            }
            """
        ):
            # 특별한 메타데이터 처리
            if metadata.get('type') == 'orchestrator_plan':
                self._render_orchestrator_plan(message, timestamp)
            elif metadata.get('type') == 'agent_status':
                self._render_agent_status(message, timestamp)
            elif metadata.get('type') == 'artifacts':
                self._render_artifacts(message, timestamp)
            elif metadata.get('type') == 'message_with_suggestions':
                self._render_message_with_suggestions(message, timestamp)
            else:
                # 일반 메시지
                st.markdown(f"""
                <div class="max-w-4xl">
                    <div class="bg-gradient-to-r from-cherry-800 to-cherry-900 border border-cherry-700 rounded-2xl rounded-tl-sm px-3 py-2.5 shadow-lg animate-fade-in">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-lg text-white">{message.role}</span>
                            <span class="text-xs text-cherry-200">{timestamp}</span>
                        </div>
                        <div class="text-white leading-relaxed">
                            {self._process_content_for_display(message.content)}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_orchestrator_plan(self, message: ChatMessage, timestamp: str) -> None:
        """오케스트레이터 플랜 렌더링"""
        plan = message.metadata.get('plan', {})
        
        st.markdown(f"""
        <div class="max-w-4xl">
            <div class="bg-gradient-to-r from-slate-800 to-slate-700 border border-a2a-400 rounded-2xl rounded-tl-sm px-3 py-2.5 shadow-lg animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg text-white">{message.role}</span>
                        <span class="bg-a2a-600 text-white px-2 py-1 rounded-full text-xs font-semibold">오케스트레이터 플랜</span>
                    </div>
                    <span class="text-xs text-slate-300">{timestamp}</span>
                </div>
                <div class="text-white">
                    <h3 class="font-bold text-a2a-300 mb-2">🎯 실행 계획</h3>
                    {self._process_content_for_display(message.content)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_agent_status(self, message: ChatMessage, timestamp: str) -> None:
        """에이전트 상태 렌더링"""
        st.markdown(f"""
        <div class="max-w-4xl">
            <div class="bg-gradient-to-r from-slate-800 to-slate-700 border border-cherry-400 rounded-2xl rounded-tl-sm px-3 py-2.5 shadow-lg animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg text-white">{message.role}</span>
                        <span class="bg-cherry-600 text-white px-2 py-1 rounded-full text-xs font-semibold">에이전트 상태</span>
                    </div>
                    <span class="text-xs text-slate-300">{timestamp}</span>
                </div>
                <div class="text-white">
                    <div class="flex items-center space-x-2">
                        <div class="w-2 h-2 bg-a2a-400 rounded-full animate-pulse"></div>
                        <span class="font-medium">A2A 에이전트 실행 중...</span>
                    </div>
                    {self._process_content_for_display(message.content)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_artifacts(self, message: ChatMessage, timestamp: str) -> None:
        """아티팩트 렌더링"""
        artifacts = message.metadata.get('artifacts', [])
        
        st.markdown(f"""
        <div class="max-w-4xl">
            <div class="bg-gradient-to-r from-slate-800 to-slate-700 border border-ai-400 rounded-2xl rounded-tl-sm px-3 py-2.5 shadow-lg animate-fade-in">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg text-white">{message.role}</span>
                        <span class="bg-ai-600 text-white px-2 py-1 rounded-full text-xs font-semibold">아티팩트 생성</span>
                    </div>
                    <span class="text-xs text-slate-300">{timestamp}</span>
                </div>
                <div class="text-white">
                    <h3 class="font-bold text-ai-300 mb-2">📊 생성된 아티팩트</h3>
                    <div class="grid grid-cols-2 gap-2 mb-2">
                        <div class="bg-slate-600 p-2 rounded-lg border border-slate-500">
                            <span class="text-sm text-white">차트</span>
                            <p class="font-semibold text-cherry-300">3개</p>
                        </div>
                        <div class="bg-slate-600 p-2 rounded-lg border border-slate-500">
                            <span class="text-sm text-white">테이블</span>
                            <p class="font-semibold text-ai-300">2개</p>
                        </div>
                    </div>
                    {self._process_content_for_display(message.content)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_system_message(self, message: ChatMessage) -> None:
        """시스템 메시지 렌더링"""
        timestamp = message.timestamp.strftime("%H:%M")
        
        with stylable_container(
            key=f"system_msg_{message.message_id}",
            css_styles="""
            div[data-testid="stVerticalBlock"] > div:first-child {
                display: flex;
                justify-content: center;
                margin: 0.4rem 0;
            }
            """
        ):
            st.markdown(f"""
            <div class="max-w-2xl">
                <div class="bg-slate-600 border border-slate-500 rounded-xl px-2.5 py-1.5 text-center shadow-sm animate-fade-in">
                    <div class="flex items-center justify-center space-x-2">
                        <span class="text-sm text-white">{message.role}</span>
                        <span class="text-xs text-slate-300">{timestamp}</span>
                    </div>
                    <div class="text-sm text-white mt-1">
                        {self._process_content_for_display(message.content)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_message_with_suggestions(self, message: ChatMessage, timestamp: str) -> None:
        """LLM 생성 메시지와 인라인 제안 버튼들을 함께 렌더링"""
        suggestions = message.metadata.get('suggestions', [])
        
        with stylable_container(
            key=f"msg_with_suggestions_{message.message_id}",
            css_styles="""
            div[data-testid="stVerticalBlock"] > div:first-child {
                display: flex;
                justify-content: flex-start;
                margin-bottom: 0.75rem;
            }
            """
        ):
            # 메시지 렌더링
            st.markdown(f"""
            <div class="max-w-4xl">
                <div class="bg-gradient-to-r from-cherry-800 to-cherry-900 border border-cherry-700 rounded-2xl rounded-tl-sm px-3 py-2.5 shadow-lg animate-fade-in">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-lg text-white">{message.role}</span>
                        <span class="text-xs text-cherry-200">{timestamp}</span>
                    </div>
                    <div class="text-white leading-relaxed">
                        {self._process_content_for_display(message.content)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 인라인 제안 버튼들 렌더링 (ChatGPT/Claude 스타일)
            if suggestions:
                st.markdown("""
                <div class="max-w-4xl mt-3 ml-1">
                    <div class="flex flex-wrap gap-2">
                """, unsafe_allow_html=True)
                
                # 제안 버튼들을 컬럼으로 배치 (반응형)
                cols = st.columns(min(len(suggestions), 3))  # 최대 3개씩 한 줄에
                
                for i, suggestion in enumerate(suggestions):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if st.button(
                            suggestion, 
                            key=f"suggestion_{message.message_id}_{i}",
                            use_container_width=True,
                            help="클릭하여 질문하기"
                        ):
                            # 제안 버튼 클릭 시 세션 상태에 저장
                            st.session_state.suggested_question = suggestion
                            st.rerun()
                
                st.markdown("</div></div>", unsafe_allow_html=True)
    
    def render_chat_interface(self) -> None:
        """🍒 CherryAI 채팅 인터페이스 렌더링"""
        
        # 채팅 컨테이너
        with stylable_container(
            key="cherry_chat_container",
            css_styles="""
            div[data-testid="stVerticalBlock"] {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 1rem;
                padding: 0.75rem;
                min-height: 300px;
                max-height: 500px;
                overflow-y: auto;
                border: 2px solid #475569;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
                color: white;
            }
            """
        ):
            # 메시지 렌더링
            messages = st.session_state[self.messages_key]
            
            if not messages:
                # 환영 메시지
                st.markdown("""
                <div class="flex flex-col items-center justify-center py-6 text-center">
                    <div class="text-6xl mb-4 animate-cherry-bounce">🍒</div>
                    <h2 class="text-2xl font-bold text-white mb-2">CherryAI에 오신 것을 환영합니다!</h2>
                    <p class="text-white max-w-md">
                        세계 최초 A2A + MCP 통합 플랫폼에서<br>
                        11개 A2A 에이전트와 7개 MCP 도구가 함께<br>
                        여러분의 데이터 분석을 도와드립니다.
                    </p>
                    <div class="mt-4 flex space-x-2">
                        <span class="bg-cherry-100 text-cherry-700 px-3 py-1 rounded-full text-sm">LLM First</span>
                        <span class="bg-ai-100 text-ai-700 px-3 py-1 rounded-full text-sm">A2A Protocol</span>
                        <span class="bg-a2a-100 text-a2a-700 px-3 py-1 rounded-full text-sm">MCP Integration</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in messages:
                    if message.role == "🧑🏻":
                        self._render_user_message(message)
                    elif message.role == "🍒":
                        self._render_assistant_message(message)
                    else:
                        self._render_system_message(message)
            
            # 자동 스크롤을 위한 앵커
            st.markdown('<div id="chat-end"></div>', unsafe_allow_html=True)
    
    def get_messages(self) -> List[ChatMessage]:
        """메시지 목록 반환"""
        return st.session_state[self.messages_key]
    
    def clear_messages(self) -> None:
        """메시지 초기화"""
        st.session_state[self.messages_key] = []
    
    def get_last_message(self) -> Optional[ChatMessage]:
        """마지막 메시지 반환"""
        messages = st.session_state[self.messages_key]
        return messages[-1] if messages else None

def create_chat_interface(session_key: str = "cherry_ai_chat") -> CherryAIChatInterface:
    """🍒 CherryAI 채팅 인터페이스 인스턴스 생성"""
    return CherryAIChatInterface(session_key=session_key) 