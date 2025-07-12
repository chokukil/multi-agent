"""
A2A SSE Streaming System Demo
Demonstrates A2A SDK 0.2.9 compliant SSE streaming with Streamlit UI
"""

import streamlit as st
import asyncio
import json
import time
import uuid
from datetime import datetime
import requests
import sys
import os

# Add the ui directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

from a2a_sse_streaming_system import get_a2a_sse_streaming_system, A2ASSEStreamingSystem


def main():
    """Main demo application"""
    st.set_page_config(
        page_title="A2A SSE Streaming Demo",
        page_icon="🌊",
        layout="wide"
    )
    
    # Apply custom CSS for A2A theming
    st.markdown("""
    <style>
    .a2a-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .a2a-status {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .a2a-event {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    
    .a2a-completed {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    
    .a2a-error {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="a2a-header">
        <h1>🌊 A2A SSE Streaming Demo</h1>
        <p>A2A SDK 0.2.9 표준 준수 Server-Sent Events 스트리밍 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'streaming_system' not in st.session_state:
        st.session_state.streaming_system = get_a2a_sse_streaming_system()
    
    if 'server_running' not in st.session_state:
        st.session_state.server_running = False
    
    if 'stream_events' not in st.session_state:
        st.session_state.stream_events = []
    
    # Sidebar - Server Control
    with st.sidebar:
        st.markdown("### 🔧 Server Control")
        
        if not st.session_state.server_running:
            if st.button("🚀 Start A2A Server", type="primary"):
                with st.spinner("Starting server..."):
                    # Start server in background (simulation)
                    st.session_state.server_running = True
                    st.success("Server started on localhost:8000")
        else:
            if st.button("⏹️ Stop Server", type="secondary"):
                st.session_state.server_running = False
                st.success("Server stopped")
        
        st.markdown("### 📊 Server Status")
        if st.session_state.server_running:
            st.success("🟢 Server Running")
            st.info("Port: 8000")
        else:
            st.error("🔴 Server Stopped")
        
        st.markdown("### 🔗 A2A Endpoints")
        st.code("""
GET /.well-known/agent.json
POST /stream
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 A2A Request")
        
        # Request configuration
        with st.expander("⚙️ Request Configuration", expanded=True):
            jsonrpc = st.text_input("JSON-RPC Version", value="2.0")
            request_id = st.text_input("Request ID", value=str(uuid.uuid4())[:8])
            method = st.selectbox("Method", ["message/stream", "message/send"])
            session_id = st.text_input("Session ID", value=str(uuid.uuid4())[:8])
        
        # Message input
        st.markdown("#### 💬 Message")
        user_message = st.text_area(
            "User Input",
            value="Hello, please process this request with streaming updates",
            height=100
        )
        
        # Request preview
        request_data = {
            "jsonrpc": jsonrpc,
            "id": request_id,
            "method": method,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": user_message
                        }
                    ]
                },
                "sessionId": session_id
            }
        }
        
        with st.expander("📋 Request Preview"):
            st.json(request_data)
        
        # Send request
        if st.button("📤 Send A2A Request", type="primary", disabled=not st.session_state.server_running):
            if st.session_state.server_running:
                with st.spinner("Sending request..."):
                    # Simulate SSE streaming
                    st.session_state.stream_events = []
                    
                    # Simulate streaming events
                    events = [
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "status_update",
                            "data": {
                                "state": "working",
                                "message": "Analyzing your request...",
                                "progress": 0.2,
                                "step": "analysis"
                            }
                        },
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "status_update",
                            "data": {
                                "state": "working",
                                "message": "Processing request...",
                                "progress": 0.5,
                                "step": "processing"
                            }
                        },
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "status_update",
                            "data": {
                                "state": "working",
                                "message": "Generating response...",
                                "progress": 0.8,
                                "step": "generating"
                            }
                        },
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "artifact_update",
                            "data": {
                                "artifact": {
                                    "artifactId": str(uuid.uuid4()),
                                    "name": "response",
                                    "parts": [
                                        {
                                            "type": "text",
                                            "text": f"Processed: {user_message}"
                                        }
                                    ],
                                    "metadata": {"processing_time": 3.0}
                                },
                                "final": True
                            }
                        },
                        {
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "status_update",
                            "data": {
                                "state": "completed",
                                "message": "Task completed successfully!",
                                "progress": 1.0,
                                "step": "completed"
                            }
                        }
                    ]
                    
                    # Add events to session state
                    for event in events:
                        st.session_state.stream_events.append(event)
                    
                    st.success("Request sent successfully!")
            else:
                st.error("Server is not running. Please start the server first.")
    
    with col2:
        st.markdown("### 🌊 SSE Stream Response")
        
        # Stream monitoring
        if st.session_state.stream_events:
            st.markdown("#### 📡 Real-time Events")
            
            # Display events
            for i, event in enumerate(st.session_state.stream_events):
                event_type = event.get("event_type", "unknown")
                data = event.get("data", {})
                timestamp = event.get("timestamp", "")
                
                if event_type == "status_update":
                    state = data.get("state", "unknown")
                    message = data.get("message", "")
                    progress = data.get("progress", 0)
                    step = data.get("step", "")
                    
                    if state == "completed":
                        st.markdown(f"""
                        <div class="a2a-completed">
                            <strong>✅ {state.upper()}</strong><br>
                            {message}<br>
                            <small>Step: {step} | Progress: {progress:.1%} | {timestamp}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="a2a-event">
                            <strong>⚡ {state.upper()}</strong><br>
                            {message}<br>
                            <small>Step: {step} | Progress: {progress:.1%} | {timestamp}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(progress)
                
                elif event_type == "artifact_update":
                    artifact = data.get("artifact", {})
                    artifact_name = artifact.get("name", "Unknown")
                    
                    st.markdown(f"""
                    <div class="a2a-event">
                        <strong>📦 ARTIFACT UPDATE</strong><br>
                        Artifact: {artifact_name}<br>
                        <small>{timestamp}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show artifact content
                    with st.expander(f"📄 Artifact: {artifact_name}"):
                        st.json(artifact)
        
        else:
            st.markdown("""
            <div class="a2a-status">
                <h4>⏳ Waiting for stream events...</h4>
                <p>Send a request to see real-time SSE streaming in action.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear events
        if st.session_state.stream_events:
            if st.button("🗑️ Clear Events"):
                st.session_state.stream_events = []
                st.success("Events cleared!")
    
    # Bottom section - A2A Protocol Information
    st.markdown("---")
    st.markdown("### 📚 A2A Protocol Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔧 A2A SDK 0.2.9 Features
        - **AgentExecutor**: 표준 에이전트 실행기
        - **TaskUpdater**: 태스크 상태 업데이트
        - **SSE Streaming**: 실시간 서버 전송 이벤트
        - **JSON-RPC 2.0**: 표준 프로토콜 준수
        """)
    
    with col2:
        st.markdown("""
        #### 📡 SSE Event Types
        - **status_update**: 태스크 상태 업데이트
        - **artifact_update**: 아티팩트 생성/업데이트
        - **heartbeat**: 연결 유지 신호
        - **error**: 오류 발생 알림
        """)
    
    with col3:
        st.markdown("""
        #### 🎯 Task States
        - **pending**: 대기 중
        - **working**: 처리 중
        - **input-required**: 입력 필요
        - **completed**: 완료
        - **failed**: 실패
        - **cancelled**: 취소됨
        """)
    
    # Example curl command
    st.markdown("### 🔗 cURL Example")
    curl_command = f"""
curl -X POST http://localhost:8000/stream \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(request_data, indent=2)}'
    """
    st.code(curl_command, language="bash")
    
    # Agent card example
    st.markdown("### 🎴 Agent Card")
    agent_card = {
        "name": "A2A SSE Streaming Agent",
        "description": "A2A SDK 0.2.9 compliant SSE streaming agent",
        "url": "http://localhost:8000",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "skills": [
            {
                "id": "sse_streaming",
                "name": "SSE Streaming",
                "description": "Real-time server-sent events streaming",
                "examples": ["Stream my request", "Process with updates"]
            }
        ]
    }
    
    with st.expander("📋 Agent Card JSON"):
        st.json(agent_card)


if __name__ == "__main__":
    main() 