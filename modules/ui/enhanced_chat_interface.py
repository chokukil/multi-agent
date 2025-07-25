"""
Enhanced Chat Interface - ChatGPT/Claude Style

Comprehensive chat interface with enhanced UI/UX features including:
- Enhanced message bubbles with distinct styling
- Real-time typing indicators with agent-specific animations  
- Auto-scroll with smooth animations
- Session persistence across browser refresh
- Advanced keyboard shortcuts and visual feedback
"""

import streamlit as st
import time
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import json
import uuid

from ..models import (
    EnhancedChatMessage, 
    StreamingResponse, 
    AgentProgressInfo, 
    TaskState,
    OneClickRecommendation
)


class EnhancedChatInterface:
    """Enhanced ChatGPT/Claude-style chat interface with comprehensive UI/UX features"""
    
    def __init__(self):
        """Initialize enhanced chat interface"""
        self.message_history: List[EnhancedChatMessage] = []
        self.typing_indicator_active = False
        self.auto_scroll_enabled = True
        self.session_key = "cherry_ai_chat_session"
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for chat persistence"""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                'messages': [],
                'session_id': str(uuid.uuid4()),
                'last_activity': datetime.now().isoformat()
            }
        
        # Restore message history from session state
        self.message_history = [
            self._deserialize_message(msg) 
            for msg in st.session_state[self.session_key]['messages']
        ]
    
    def render_chat_container(self) -> None:
        """
        Render enhanced chat area with:
        - User messages: right-aligned speech bubbles with distinct styling
        - AI responses: left-aligned with Cherry AI avatar and branding
        - Real-time typing indicators with agent-specific animations
        - Auto-scroll to bottom with smooth animations
        - Session persistence across browser refresh
        """
        # Create chat container with custom CSS
        self._inject_chat_css()
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Render all messages
            for message in self.message_history:
                self._render_message(message)
            
            # Show typing indicator if active
            if self.typing_indicator_active:
                self._render_typing_indicator()
            
            # Auto-scroll placeholder (pushes content up)
            if self.auto_scroll_enabled:
                st.empty()
    
    def _inject_chat_css(self):
        """Inject custom CSS for enhanced chat styling"""
        st.markdown("""
        <style>
        /* Enhanced Chat Interface Styles */
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            margin-right: 0;
            border-bottom-right-radius: 4px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        .assistant-message {
            background: #f8f9fa;
            color: #2c3e50;
            margin-left: 0;
            margin-right: auto;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .cherry-ai-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin-right: 0.5rem;
            font-size: 14px;
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #6c757d;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { opacity: 0.3; }
            30% { opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-timestamp {
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 0.25rem;
            text-align: right;
        }
        
        .agent-collaboration-panel {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: white;
        }
        
        .agent-progress-bar {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            height: 8px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        
        .agent-progress-fill {
            background: #28a745;
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .recommendation-card {
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .recommendation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-color: #007bff;
        }
        
        .execute-button {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .execute-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_message(self, message: EnhancedChatMessage):
        """Render individual message with enhanced styling"""
        if message.role == "user":
            self._render_user_message(message)
        elif message.role == "assistant":
            self._render_assistant_message(message)
        elif message.role == "system":
            self._render_system_message(message)
    
    def _render_user_message(self, message: EnhancedChatMessage):
        """Render user message with right-aligned speech bubble"""
        st.markdown(f"""
        <div class="chat-message user-message">
            <div>{message.content}</div>
            <div class="message-timestamp">
                {message.timestamp.strftime("%H:%M")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_assistant_message(self, message: EnhancedChatMessage):
        """Render AI response with Cherry AI avatar and branding"""
        # Agent collaboration panel if progress info exists
        if message.progress_info:
            self._render_agent_collaboration_panel(message.progress_info)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="display: flex; align-items: flex-start;">
                <div class="cherry-ai-avatar">🍒</div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #495057; margin-bottom: 0.25rem;">
                        Cherry AI Assistant
                    </div>
                    <div>{message.content}</div>
                    <div class="message-timestamp">
                        {message.timestamp.strftime("%H:%M")}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render artifacts if any
        if message.artifacts:
            self._render_message_artifacts(message.artifacts)
        
        # Render recommendations if any
        if message.recommendations:
            self._render_recommendations(message.recommendations)
    
    def _render_system_message(self, message: EnhancedChatMessage):
        """Render system message with neutral styling"""
        st.info(f"🤖 {message.content}")
    
    def _render_typing_indicator(self):
        """Render animated typing indicator"""
        st.markdown("""
        <div class="typing-indicator">
            <div class="cherry-ai-avatar">🍒</div>
            <div>
                <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.25rem;">
                    Cherry AI is thinking...
                </div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_agent_collaboration_panel(self, progress_info):
        """Render real-time agent collaboration visualization"""
        st.markdown("""
        <div class="agent-collaboration-panel">
            <div style="font-weight: 500; margin-bottom: 0.5rem;">
                🤝 Agent Collaboration in Progress
            </div>
        """, unsafe_allow_html=True)
        
        for agent in progress_info.agents_working:
            status_icon = self._get_status_icon(agent.status)
            progress_color = self._get_progress_color(agent.status)
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                    <span style="margin-right: 0.5rem;">{status_icon}</span>
                    <span style="font-weight: 500;">{agent.name}</span>
                    <span style="margin-left: auto; font-size: 0.8rem;">
                        {agent.progress_percentage:.0f}%
                    </span>
                </div>
                <div class="agent-progress-bar">
                    <div class="agent-progress-fill" style="width: {agent.progress_percentage}%; background: {progress_color};"></div>
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {agent.current_task}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _render_message_artifacts(self, artifacts):
        """Render artifacts associated with a message"""
        for artifact in artifacts:
            st.write("📎 **Artifact Generated:**")
            # This will be handled by the artifact rendering system
            st.json({"type": artifact.type.value, "id": artifact.id})
    
    def _render_recommendations(self, recommendations: List[OneClickRecommendation]):
        """Render one-click recommendations"""
        st.markdown("### 💡 Recommended Next Steps")
        
        for rec in recommendations:
            with st.expander(f"{rec.icon} {rec.title}", expanded=False):
                st.markdown(f"**Description:** {rec.description}")
                st.markdown(f"**Estimated Time:** {rec.estimated_time} seconds")
                st.markdown(f"**Complexity:** {rec.complexity_level}")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"*{rec.expected_result_preview}*")
                with col2:
                    if st.button(rec.execution_button_text, key=f"exec_{rec.title}"):
                        self._execute_recommendation(rec)
    
    def handle_user_input(self, on_message_callback: Optional[Callable] = None) -> Optional[str]:
        """
        Enhanced input handling with:
        - Shift+Enter for line breaks, Enter for sending
        - Auto-resize text area based on content  
        - Placeholder text and visual feedback
        - Attachment button integration
        """
        # Input container with enhanced styling
        input_container = st.container()
        
        with input_container:
            col1, col2, col3 = st.columns([1, 6, 1])
            
            with col1:
                # Attachment button
                if st.button("📎", help="Attach files", key="attach_files"):
                    st.session_state["show_file_upload"] = True
            
            with col2:
                # Enhanced text input with proper keyboard handling
                user_input = st.text_area(
                    label="Message",
                    placeholder="여기에 메시지를 입력하세요... (Shift+Enter로 줄바꿈, Enter로 전송)",
                    key="user_message_input",
                    height=100,
                    label_visibility="collapsed"
                )
            
            with col3:
                # Send button with visual feedback
                send_clicked = st.button("📤", help="Send message", key="send_message")
            
            # Handle keyboard shortcuts with JavaScript
            self._inject_keyboard_handler()
            
            # Process input
            if send_clicked and user_input.strip():
                self._add_user_message(user_input.strip())
                if on_message_callback:
                    on_message_callback(user_input.strip())
                
                # Clear input
                st.session_state["user_message_input"] = ""
                st.experimental_rerun()
                
                return user_input.strip()
        
        return None
    
    def _inject_keyboard_handler(self):
        """Inject JavaScript for enhanced keyboard handling"""
        st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            const textarea = document.querySelector('textarea[aria-label="Message"]');
            if (textarea && document.activeElement === textarea) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const sendButton = document.querySelector('button[title="Send message"]');
                    if (sendButton) {
                        sendButton.click();
                    }
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)
    
    def display_streaming_response(self, response_generator, agent_info: Optional[List[AgentProgressInfo]] = None):
        """
        Display streaming response with:
        - Natural typing effects with intelligent chunking
        - Agent collaboration visualization
        - Progress bars and status indicators
        """
        self.typing_indicator_active = True
        response_placeholder = st.empty()
        
        accumulated_content = ""
        
        try:
            for chunk in response_generator:
                if isinstance(chunk, StreamingResponse):
                    accumulated_content += chunk.content
                    
                    # Update typing indicator with content
                    with response_placeholder.container():
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div style="display: flex; align-items: flex-start;">
                                <div class="cherry-ai-avatar">🍒</div>
                                <div style="flex: 1;">
                                    <div style="font-weight: 500; color: #495057; margin-bottom: 0.25rem;">
                                        Cherry AI Assistant
                                    </div>
                                    <div>{accumulated_content}<span class="typing-cursor">|</span></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show agent progress if available
                    if chunk.progress_info:
                        self._render_agent_collaboration_panel(chunk.progress_info)
                    
                    # Natural typing delay
                    time.sleep(0.001)
                
        finally:
            self.typing_indicator_active = False
            
            # Add final message to history
            final_message = EnhancedChatMessage(
                id=str(uuid.uuid4()),
                content=accumulated_content,
                role="assistant",
                timestamp=datetime.now()
            )
            
            self._add_message_to_history(final_message)
            
            # Clear placeholder and show final message
            response_placeholder.empty()
            self._render_assistant_message(final_message)
    
    def _add_user_message(self, content: str):
        """Add user message to chat history"""
        message = EnhancedChatMessage(
            id=str(uuid.uuid4()),
            content=content,
            role="user",
            timestamp=datetime.now()
        )
        self._add_message_to_history(message)
    
    def _add_message_to_history(self, message: EnhancedChatMessage):
        """Add message to history and persist to session state"""
        self.message_history.append(message)
        
        # Update session state
        st.session_state[self.session_key]['messages'] = [
            self._serialize_message(msg) for msg in self.message_history
        ]
        st.session_state[self.session_key]['last_activity'] = datetime.now().isoformat()
    
    def _serialize_message(self, message: EnhancedChatMessage) -> Dict[str, Any]:
        """Serialize message for session storage"""
        return {
            'id': message.id,
            'content': message.content,
            'role': message.role,
            'timestamp': message.timestamp.isoformat(),
            'ui_metadata': message.ui_metadata
        }
    
    def _deserialize_message(self, data: Dict[str, Any]) -> EnhancedChatMessage:
        """Deserialize message from session storage"""
        return EnhancedChatMessage(
            id=data['id'],
            content=data['content'],
            role=data['role'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            ui_metadata=data.get('ui_metadata', {})
        )
    
    def _get_status_icon(self, status: TaskState) -> str:
        """Get status icon for agent state"""
        icons = {
            TaskState.PENDING: "⏳",
            TaskState.WORKING: "⚡",
            TaskState.COMPLETED: "✅",
            TaskState.FAILED: "❌"
        }
        return icons.get(status, "❓")
    
    def _get_progress_color(self, status: TaskState) -> str:
        """Get progress bar color for agent state"""
        colors = {
            TaskState.PENDING: "#ffc107",
            TaskState.WORKING: "#007bff", 
            TaskState.COMPLETED: "#28a745",
            TaskState.FAILED: "#dc3545"
        }
        return colors.get(status, "#6c757d")
    
    def _execute_recommendation(self, recommendation: OneClickRecommendation):
        """Execute a one-click recommendation"""
        st.success(f"Executing: {recommendation.title}")
        # This will be handled by the orchestration system
        st.session_state[f"execute_recommendation"] = recommendation
    
    def clear_chat_history(self):
        """Clear chat history and session state"""
        self.message_history.clear()
        if self.session_key in st.session_state:
            st.session_state[self.session_key]['messages'] = []
        st.experimental_rerun()
    
    def export_chat_history(self) -> str:
        """Export chat history as JSON"""
        export_data = {
            'session_id': st.session_state[self.session_key].get('session_id'),
            'export_time': datetime.now().isoformat(),
            'messages': [self._serialize_message(msg) for msg in self.message_history]
        }
        return json.dumps(export_data, indent=2)