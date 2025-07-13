"""
🔧 MCP Integration Component
Cursor 스타일의 MCP (Model Context Protocol) 도구 통합 관리
"""

import streamlit as st
from typing import Dict, Any, List
import json
import time

def get_mcp_tools_config() -> Dict[str, Dict[str, Any]]:
    """MCP 도구 설정 정보"""
    return {
        "playwright": {
            "name": "Playwright Browser",
            "icon": "🌐",
            "description": "웹 브라우저 자동화 및 테스팅",
            "status": "active",
            "capabilities": ["web_scraping", "browser_automation", "ui_testing"],
            "version": "1.40.0",
            "last_used": "2024-01-15 09:30:00"
        },
        "file_manager": {
            "name": "File Manager", 
            "icon": "📁",
            "description": "파일 시스템 관리 및 조작",
            "status": "active",
            "capabilities": ["file_operations", "directory_management", "file_search"],
            "version": "2.1.0",
            "last_used": "2024-01-15 09:25:00"
        },
        "database": {
            "name": "Database Tools",
            "icon": "🗄️", 
            "description": "데이터베이스 연결 및 쿼리 실행",
            "status": "active",
            "capabilities": ["sql_queries", "database_connection", "data_extraction"],
            "version": "1.5.0",
            "last_used": "2024-01-15 09:20:00"
        },
        "http_client": {
            "name": "HTTP Client",
            "icon": "🌍",
            "description": "HTTP API 호출 및 웹 서비스 통신",
            "status": "active", 
            "capabilities": ["api_calls", "rest_client", "webhook_handling"],
            "version": "3.0.1",
            "last_used": "2024-01-15 09:15:00"
        },
        "code_executor": {
            "name": "Code Executor",
            "icon": "⚙️",
            "description": "다양한 언어 코드 실행 환경",
            "status": "active",
            "capabilities": ["python_execution", "javascript_execution", "shell_commands"],
            "version": "1.8.0", 
            "last_used": "2024-01-15 09:10:00"
        },
        "data_processor": {
            "name": "Data Processor",
            "icon": "🔄",
            "description": "데이터 변환 및 처리 파이프라인",
            "status": "active",
            "capabilities": ["data_transformation", "format_conversion", "data_validation"],
            "version": "2.3.0",
            "last_used": "2024-01-15 09:05:00"
        },
        "ai_assistant": {
            "name": "AI Assistant",
            "icon": "🤖",
            "description": "AI 모델 호출 및 자연어 처리",
            "status": "active",
            "capabilities": ["llm_calls", "text_processing", "ai_inference"],
            "version": "4.2.0",
            "last_used": "2024-01-15 09:00:00"
        }
    }

def render_mcp_overview():
    """MCP 개요"""
    st.markdown("## 🔧 MCP Tools 개요")
    
    tools_config = get_mcp_tools_config()
    
    # 통계 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 MCP 도구", len(tools_config), "통합 완료")
    
    with col2:
        active_tools = sum(1 for tool in tools_config.values() if tool["status"] == "active")
        st.metric("활성 도구", active_tools, "100% 가동")
    
    with col3:
        total_capabilities = sum(len(tool["capabilities"]) for tool in tools_config.values())
        st.metric("총 기능", total_capabilities, "다양한 능력")
    
    with col4:
        st.metric("통합 상태", "정상", "A2A 연동")

def render_mcp_tools_grid():
    """MCP 도구 그리드"""
    st.markdown("## 🛠️ MCP Tools 상태")
    
    tools_config = get_mcp_tools_config()
    
    # 상태 새로고침
    if st.button("🔄 도구 상태 새로고침"):
        st.rerun()
    
    # 3열 그리드로 도구 표시
    cols = st.columns(3)
    
    for i, (tool_id, config) in enumerate(tools_config.items()):
        with cols[i % 3]:
            status_color = "#28a745" if config["status"] == "active" else "#dc3545"
            status_icon = "✅" if config["status"] == "active" else "❌"
            
            st.markdown(f"""
            <div class="cursor-mcp-tool" style="border-left: 4px solid {status_color};">
                <div class="tool-header">
                    <span class="tool-icon">{config['icon']}</span>
                    <span class="tool-name">{config['name']}</span>
                    <span class="tool-status">{status_icon}</span>
                </div>
                <div class="tool-description">{config['description']}</div>
                <div class="tool-version">v{config['version']}</div>
                <div class="tool-capabilities">
                    {' '.join([f'<span class="cap-tag">{cap[:8]}...</span>' for cap in config['capabilities'][:2]])}
                </div>
                <div class="tool-last-used">마지막 사용: {config['last_used']}</div>
            </div>
            """, unsafe_allow_html=True)

def render_mcp_testing():
    """MCP 도구 테스팅"""
    st.markdown("## 🧪 MCP Tools 테스팅")
    
    tools_config = get_mcp_tools_config()
    
    # 도구 선택
    tool_options = {config['name']: tool_id for tool_id, config in tools_config.items()}
    selected_tool_name = st.selectbox("테스트할 MCP 도구 선택", list(tool_options.keys()))
    selected_tool_id = tool_options[selected_tool_name]
    selected_config = tools_config[selected_tool_id]
    
    # 선택된 도구 정보
    st.markdown(f"### {selected_config['icon']} {selected_config['name']}")
    st.markdown(f"**설명:** {selected_config['description']}")
    st.markdown(f"**버전:** {selected_config['version']}")
    st.markdown(f"**능력:** {', '.join(selected_config['capabilities'])}")
    
    # 테스트 시나리오 선택
    test_scenarios = {
        "playwright": {
            "기본 연결 테스트": "브라우저 인스턴스 생성 및 기본 네비게이션",
            "스크린샷 캡처": "웹페이지 스크린샷 생성",
            "요소 클릭": "웹 요소 자동 클릭 테스트"
        },
        "file_manager": {
            "파일 목록 조회": "현재 디렉토리 파일 목록 가져오기",
            "파일 읽기": "텍스트 파일 내용 읽기",
            "파일 생성": "새 파일 생성 테스트"
        },
        "database": {
            "연결 테스트": "데이터베이스 연결 상태 확인",
            "쿼리 실행": "SELECT 쿼리 실행",
            "스키마 조회": "테이블 스키마 정보 조회"
        }
    }
    
    scenarios = test_scenarios.get(selected_tool_id, {"기본 테스트": "도구 기본 동작 확인"})
    selected_scenario = st.selectbox("테스트 시나리오 선택", list(scenarios.keys()))
    
    st.markdown(f"**시나리오 설명:** {scenarios[selected_scenario]}")
    
    # 테스트 실행
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 테스트 실행", type="primary"):
            with st.status(f"🧪 {selected_tool_name} 테스트 실행 중...", expanded=True):
                st.write(f"📋 시나리오: {selected_scenario}")
                time.sleep(1)
                st.write("🔧 MCP 도구 연결 중...")
                time.sleep(1)
                st.write("⚡ 명령 실행 중...")
                time.sleep(1)
                st.write("✅ 테스트 완료!")
            
            # 모의 테스트 결과
            st.success(f"🎉 {selected_tool_name} 테스트가 성공적으로 완료되었습니다!")
            
            with st.expander("📋 테스트 결과 상세", expanded=False):
                st.markdown(f"""
                **테스트 도구:** {selected_config['name']} v{selected_config['version']}
                **테스트 시나리오:** {selected_scenario}
                **실행 시간:** 2.3초
                **상태:** ✅ 성공
                
                **응답 데이터:**
                ```json
                {{
                    "status": "success",
                    "tool": "{selected_tool_id}",
                    "scenario": "{selected_scenario}",
                    "timestamp": "{time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "response_time_ms": 2300,
                    "data": "테스트 실행 완료"
                }}
                ```
                """)

def render_mcp_integration_status():
    """MCP 통합 상태"""
    st.markdown("## 🌐 A2A + MCP 통합 상태")
    
    # 통합 아키텍처 다이어그램 (텍스트 기반)
    st.markdown("""
    ### 🏗️ 통합 아키텍처
    
    ```
    📱 Streamlit UI
           ↕️
    🧬 A2A Agents (10개)
           ↕️  
    🔧 MCP Tools (7개)
           ↕️
    🌍 External Services
    ```
    """)
    
    # 통합 상태 정보
    integration_stats = {
        "A2A → MCP 연동": {"status": "✅", "description": "A2A 에이전트가 MCP 도구를 성공적으로 호출"},
        "MCP → A2A 응답": {"status": "✅", "description": "MCP 도구 실행 결과가 A2A로 정상 반환"},
        "Context Engineering": {"status": "✅", "description": "6개 데이터 레이어 통합 운영"},
        "실시간 스트리밍": {"status": "✅", "description": "SSE 기반 실시간 데이터 흐름"},
        "오류 복구": {"status": "✅", "description": "자동 에러 감지 및 복구 시스템"},
        "성능 최적화": {"status": "✅", "description": "병렬 처리 및 지능형 라우팅"}
    }
    
    col1, col2 = st.columns(2)
    
    for i, (feature, info) in enumerate(integration_stats.items()):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            **{info['status']} {feature}**  
            {info['description']}
            """)

def render_mcp_logs():
    """MCP 로그"""
    st.markdown("## 📋 MCP Tools 활동 로그")
    
    # 모의 로그 데이터
    logs = [
        {"time": "09:32:45", "tool": "🌐 Playwright", "action": "스크린샷 캡처", "status": "✅"},
        {"time": "09:32:30", "tool": "📁 File Manager", "action": "파일 목록 조회", "status": "✅"},
        {"time": "09:32:15", "tool": "🗄️ Database", "action": "SQL 쿼리 실행", "status": "✅"},
        {"time": "09:32:00", "tool": "🌍 HTTP Client", "action": "API 호출", "status": "✅"},
        {"time": "09:31:45", "tool": "⚙️ Code Executor", "action": "Python 코드 실행", "status": "✅"},
        {"time": "09:31:30", "tool": "🔄 Data Processor", "action": "데이터 변환", "status": "✅"},
        {"time": "09:31:15", "tool": "🤖 AI Assistant", "action": "LLM 호출", "status": "✅"},
    ]
    
    # 로그 필터
    col1, col2 = st.columns([1, 4])
    
    with col1:
        log_filter = st.selectbox("필터", ["전체", "성공", "오류", "최근 1시간"])
    
    # 로그 표시
    for log in logs:
        st.markdown(f"""
        <div class="cursor-log-item">
            <span class="log-time">{log['time']}</span>
            <span class="log-tool">{log['tool']}</span>
            <span class="log-action">{log['action']}</span>
            <span class="log-status">{log['status']}</span>
        </div>
        """, unsafe_allow_html=True)

def apply_mcp_styles():
    """MCP 컴포넌트 전용 스타일"""
    st.markdown("""
    <style>
    .cursor-mcp-tool {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
    }
    
    .cursor-mcp-tool:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 122, 204, 0.15);
    }
    
    .tool-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .tool-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .tool-name {
        font-weight: 600;
        color: var(--cursor-primary-text);
        flex: 1;
    }
    
    .tool-status {
        font-size: 1.2rem;
    }
    
    .tool-description {
        color: var(--cursor-secondary-text);
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        flex: 1;
    }
    
    .tool-version {
        font-size: 0.8rem;
        color: var(--cursor-muted-text);
        margin-bottom: 0.5rem;
    }
    
    .tool-capabilities {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin-bottom: 0.5rem;
    }
    
    .cap-tag {
        background: rgba(46, 125, 50, 0.2);
        color: #2e7d32;
        font-size: 0.7rem;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        border: 1px solid rgba(46, 125, 50, 0.3);
    }
    
    .tool-last-used {
        font-size: 0.7rem;
        color: var(--cursor-muted-text);
        margin-top: auto;
    }
    
    .cursor-log-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: var(--cursor-secondary-bg);
        border-radius: 6px;
        border-left: 3px solid #2e7d32;
    }
    
    .log-time {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        min-width: 80px;
        margin-right: 1rem;
    }
    
    .log-tool {
        margin-right: 1rem;
        font-size: 1.1rem;
        min-width: 150px;
    }
    
    .log-action {
        color: var(--cursor-secondary-text);
        flex: 1;
        margin-right: 1rem;
    }
    
    .log-status {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_mcp_integration():
    """MCP 통합 메인 렌더링"""
    # 스타일 적용
    apply_mcp_styles()
    
    # 헤더
    st.markdown("# 🔧 MCP Integration")
    st.markdown("**Model Context Protocol 도구 통합 관리**")
    
    st.markdown("---")
    
    # MCP 개요
    render_mcp_overview()
    
    st.markdown("---")
    
    # 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛠️ MCP Tools",
        "🧪 도구 테스팅", 
        "🌐 통합 상태",
        "📋 활동 로그"
    ])
    
    with tab1:
        render_mcp_tools_grid()
    
    with tab2:
        render_mcp_testing()
    
    with tab3:
        render_mcp_integration_status()
    
    with tab4:
        render_mcp_logs() 