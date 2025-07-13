#!/usr/bin/env python3
"""
CherryAI UI 테스트용 간단한 메인 애플리케이션
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import time
from typing import List, Dict, Any
import uuid

def apply_cursor_theme():
    """커서 테마 적용"""
    st.markdown("""
    <style>
    /* 커서 테마 글로벌 스타일 */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1400px !important;
    }
    
    /* 헤더 스타일 */
    .cherry-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    
    /* 채팅 메시지 스타일 */
    .chat-message {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #da3633 0%, #a21e1e 100%);
        color: white;
        margin-right: 20%;
    }
    
    /* 입력 요소 개선 */
    .stTextInput input, .stTextArea textarea {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        color: #f0f6fc !important;
        border-radius: 10px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """테스트용 간단한 메인 애플리케이션"""
    try:
        # 페이지 설정
        st.set_page_config(
            page_title="🍒 CherryAI - 테스트",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # 커서 테마 적용
        apply_cursor_theme()
        
        # 헤더
        st.markdown("""
        <div class="cherry-header">
            <h1>🍒 CherryAI - UI 테스트 모드</h1>
            <h3>🌟 세계 최초 A2A + MCP 통합 | 11개 A2A 에이전트 + 7개 MCP 도구</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 상태 표시
        st.markdown("""
        <div style="
            background: rgba(0,255,0,0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #00ff00;
        ">
            🟢 <strong>시스템 상태: 정상</strong> | 
            🤖 11개 A2A 에이전트 대기 중 | 
            🔧 7개 MCP 도구 준비 완료
        </div>
        """, unsafe_allow_html=True)
        
        # 세션 상태 초기화
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        # 메인 컨테이너 - 7:3 비율
        main_col, sidebar_col = st.columns([7, 3])
        
        with main_col:
            st.markdown("### 💬 채팅 인터페이스")
            
            # 채팅 메시지 표시
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 사용자:</strong><br>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🍒 CherryAI:</strong><br>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
            
            # 채팅 입력
            user_input = st.chat_input("CherryAI에게 질문하세요...")
            
            if user_input:
                # 사용자 메시지 추가
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # 시뮬레이션된 AI 응답
                ai_response = f"""
                안녕하세요! 사용자님의 질문 "{user_input}"을 받았습니다.
                
                현재 테스트 모드에서 실행 중입니다. 다음과 같은 기능들을 테스트할 수 있습니다:
                
                📊 **데이터 분석**: CSV, Excel 파일 업로드 및 분석
                🤖 **A2A 에이전트**: 11개의 전문 AI 에이전트
                🔧 **MCP 도구**: 7개의 Model Context Protocol 도구
                📈 **실시간 스트리밍**: ChatGPT/Claude 스타일 대화
                
                파일을 업로드하시거나 구체적인 질문을 해주시면 더 자세한 도움을 드릴 수 있습니다!
                """
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                st.rerun()
        
        with sidebar_col:
            st.markdown("### 📁 파일 업로드")
            
            uploaded_files = st.file_uploader(
                "CSV, Excel, JSON 파일을 업로드하세요",
                type=['csv', 'xlsx', 'xls', 'json'],
                accept_multiple_files=True,
                key="file_uploader"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)}개 파일 업로드됨")
                st.session_state.uploaded_files = uploaded_files
                
                # 파일 정보 표시
                for file in uploaded_files:
                    st.markdown(f"""
                    📄 **{file.name}**
                    - 크기: {file.size:,} bytes
                    - 타입: {file.type}
                    """)
            
            st.markdown("### 🎯 테스트 시나리오")
            
            test_scenarios = [
                "데이터 분석을 해주세요",
                "데이터 시각화를 보여주세요", 
                "탐색적 데이터 분석을 실행하세요",
                "특성 엔지니어링을 수행하세요",
                "머신러닝 모델을 만들어주세요",
                "통계 분석 결과를 알려주세요"
            ]
            
            for scenario in test_scenarios:
                if st.button(scenario, key=f"scenario_{scenario[:10]}"):
                    # 시나리오 실행
                    st.session_state.messages.append({
                        "role": "user",
                        "content": scenario
                    })
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"'{scenario}' 시나리오를 테스트 모드에서 실행합니다. 실제 A2A 에이전트와 MCP 도구들이 연동되어 처리할 예정입니다."
                    })
                    
                    st.rerun()
            
            st.markdown("### 📊 시스템 정보")
            st.json({
                "mode": "test",
                "a2a_agents": 11,
                "mcp_tools": 7,
                "streaming": "active",
                "messages": len(st.session_state.messages),
                "files": len(st.session_state.uploaded_files)
            })
    
    except Exception as e:
        st.error(f"""
        ❌ **오류 발생**
        
        {str(e)}
        
        이는 테스트 모드 UI입니다. 전체 시스템을 테스트하려면 main.py를 사용하세요.
        """)
        
        if st.button("🔄 새로고침", type="primary"):
            st.rerun()

if __name__ == "__main__":
    main() 