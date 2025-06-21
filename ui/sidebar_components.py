# File: ui/sidebar_components.py
# Location: ./ui/sidebar_components.py

import streamlit as st
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import asyncio
import logging
from datetime import datetime

def render_mcp_config_section():
    """MCP Server Configuration 섹션 렌더링"""
    from core.utils.config import load_mcp_configs, save_mcp_config, delete_mcp_config
    from core.tools.mcp_tools import get_available_mcp_tools_info
    
    with st.expander("🔧 MCP Server Configuration", expanded=False):
        st.markdown("### MCP 도구 설정 관리")
        
        # 현재 저장된 설정들 표시
        saved_configs = load_mcp_configs()
        
        if saved_configs:
            st.markdown("#### 저장된 설정")
            config_names = [config.get('name', config['config_name']) for config in saved_configs]
            selected_config = st.selectbox(
                "설정 선택",
                ["None"] + config_names,
                help="저장된 MCP 설정을 선택하세요"
            )
            
            if selected_config != "None":
                config_data = next((c for c in saved_configs if c.get('name', c['config_name']) == selected_config), None)
                if config_data:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 상태 확인", key="check_mcp_status"):
                            with st.spinner("MCP 서버 상태 확인 중..."):
                                tools_info = get_available_mcp_tools_info(config_data['config_name'])
                                if tools_info['available']:
                                    st.success(f"✅ {tools_info['available_servers']}/{tools_info['total_servers']} 서버 사용 가능")
                                    for tool in tools_info['tools']:
                                        status_icon = "🟢" if tool['status'] == "available" else "🔴"
                                        st.text(f"{status_icon} {tool['server_name']}")
                                else:
                                    st.error("❌ 사용 가능한 서버가 없습니다")
                    
                    with col2:
                        if st.button("📝 수정", key="edit_mcp_config"):
                            st.session_state.editing_mcp_config = config_data
                    
                    with col3:
                        if st.button("🗑️ 삭제", key="delete_mcp_config"):
                            if delete_mcp_config(config_data['config_name']):
                                st.success(f"✅ '{selected_config}' 삭제됨")
                                st.rerun()
                            else:
                                st.error("❌ 삭제 실패")
        
        # 새 설정 추가 또는 수정
        st.markdown("#### 새 설정 추가")
        
        with st.form("mcp_config_form"):
            config_name = st.text_input(
                "설정 이름",
                value=st.session_state.get('editing_mcp_config', {}).get('name', ''),
                placeholder="예: my_data_tools"
            )
            
            config_description = st.text_area(
                "설명",
                value=st.session_state.get('editing_mcp_config', {}).get('description', ''),
                placeholder="이 설정에 대한 간단한 설명을 입력하세요"
            )
            
            # JSON 설정 입력
            default_json = json.dumps(
                st.session_state.get('editing_mcp_config', {}).get('mcpServers', {
                    "data_science_tools": {
                        "url": "http://localhost:8007/sse", 
                        "transport": "sse",
                        "description": "데이터 분석 도구"
                    }
                }),
                indent=2
            )
            
            json_config = st.text_area(
                "MCP 서버 설정 (JSON)",
                value=default_json,
                height=200,
                help="MCP 서버 설정을 JSON 형식으로 입력하세요"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                save_button = st.form_submit_button("💾 저장", type="primary")
            with col2:
                if st.form_submit_button("❌ 취소"):
                    if 'editing_mcp_config' in st.session_state:
                        del st.session_state.editing_mcp_config
                    st.rerun()
            
            if save_button:
                if not config_name:
                    st.error("❌ 설정 이름을 입력하세요")
                else:
                    try:
                        # JSON 유효성 검사
                        servers_config = json.loads(json_config)
                        
                        # 설정 데이터 구성
                        config_data = {
                            "name": config_name,
                            "description": config_description,
                            "mcpServers": servers_config,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        # 저장
                        if save_mcp_config(config_name, config_data):
                            st.success(f"✅ '{config_name}' 설정이 저장되었습니다")
                            if 'editing_mcp_config' in st.session_state:
                                del st.session_state.editing_mcp_config
                            st.rerun()
                        else:
                            st.error("❌ 설정 저장 실패")
                    
                    except json.JSONDecodeError as e:
                        st.error(f"❌ JSON 형식 오류: {e}")

def render_data_upload_section():
    """데이터 업로드 섹션 렌더링"""
    from core import data_manager, data_lineage_tracker
    
    st.markdown("### 강화된 데이터 분석")
    
    # SSOT 데이터 상태 표시
    current_status = data_manager.get_status_message()
    
    if data_manager.is_data_loaded():
        st.success(current_status)
        
        # 데이터 정보 표시
        with st.expander("📈 현재 데이터 정보", expanded=False):
            info = data_manager.get_data_info()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("행 수", f"{info['row_count']:,}")
                st.metric("메모리", f"{info['memory_mb']:.2f}MB")
            
            with col2:
                st.metric("열 수", f"{info['col_count']:,}")
                st.metric("출처", info['source'])
            
            st.write("**컬럼 정보:**")
            cols_display = ', '.join(info['columns'][:8])
            if len(info['columns']) > 8:
                cols_display += f" ... (+{len(info['columns']) - 8}개)"
            st.text(cols_display)
            
            st.write("**통계:**")
            st.text(f"수치형: {len(info['numeric_cols'])}개, 범주형: {len(info['categorical_cols'])}개, 결측값: {info['null_count']:,}개")
        
        # 데이터 관리 도구들
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 새로고침", use_container_width=True):
                if hasattr(st.session_state, 'uploaded_data') and st.session_state.uploaded_data is not None:
                    success = data_manager.set_data(st.session_state.uploaded_data, "세션 데이터 새로고침")
                    if success:
                        st.success("✅ 데이터가 새로고침되었습니다!")
                        st.rerun()
        
        with col2:
            if st.button("✅ 일관성 검증", use_container_width=True):
                is_valid, message = data_manager.validate_data_consistency()
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    else:
        st.warning(current_status)
    
    # sandbox/datasets 폴더 생성
    DATASETS_DIR = "./sandbox/datasets"
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    uploaded_csv = st.file_uploader("📂 CSV 파일 업로드", type=["csv"], help="데이터 분석을 위한 CSV 파일을 업로드하세요")
    if uploaded_csv:
        try:
            # 파일을 datasets 폴더에 저장
            file_path = os.path.join(DATASETS_DIR, uploaded_csv.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_csv.getvalue())
                
            # 저장된 파일을 DataFrame으로 로드
            df = pd.read_csv(file_path)
            
            # 강화된 SSOT에 데이터 설정
            success = data_manager.set_data(df, f"업로드된 파일: {uploaded_csv.name}")
            
            if success:
                # 데이터 계보 추적 시작
                original_hash = data_lineage_tracker.set_original_data(df)
                
                # 세션 상태에도 백업 저장
                st.session_state.uploaded_data = df
                st.session_state.original_data_hash = original_hash
                
                st.success(f"✅ 강화된 SSOT 설정 완료: {uploaded_csv.name}")
                
                # 업데이트된 데이터 정보 표시
                with st.expander("📈 업로드된 데이터 정보", expanded=True):
                    info = data_manager.get_data_info()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("행 수", f"{info['row_count']:,}")
                    with col2:
                        st.metric("열 수", f"{info['col_count']:,}")
                    with col3:
                        st.metric("메모리", f"{info['memory_mb']:.2f}MB")
                    
                    st.write("**파일 경로**: `" + file_path + "`")
                    st.write("**컬럼 목록**: " + ', '.join(info['columns'][:5]) + ("..." if len(info['columns']) > 5 else ""))
                    
                    # 샘플 데이터 미리보기
                    st.write("**미리보기:**")
                    st.dataframe(df.head(3), use_container_width=True)
                
                # 시스템 초기화 안내
                if st.session_state.get("graph_initialized", False):
                    st.info("💡 모든 Executor가 동일한 데이터에 접근할 수 있습니다!")
            else:
                st.error("❌ 강화된 SSOT 데이터 설정 실패")
                
        except Exception as e:
            # SSOT 초기화
            data_manager.clear_data()
            if hasattr(st.session_state, 'uploaded_data'):
                del st.session_state.uploaded_data
            st.error(f"CSV 업로드 실패: {e}")
    
    # 데이터 삭제 옵션
    if data_manager.is_data_loaded():
        if st.button("🗑️ 데이터 삭제", use_container_width=True, type="secondary"):
            data_manager.clear_data()
            if hasattr(st.session_state, 'uploaded_data'):
                del st.session_state.uploaded_data
            st.success("✅ 데이터가 삭제되었습니다.")
            st.rerun()

def render_system_settings():
    """시스템 설정 섹션 렌더링"""
    with st.expander("⚙️ System Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_recursion_limit = st.number_input(
                "Recursion Limit",
                min_value=5,
                max_value=50,
                value=st.session_state.get("recursion_limit", 30),
                step=1,
                help="Maximum number of recursive calls to prevent infinite loops"
            )
        
        with col2:
            new_timeout_seconds = st.number_input(
                "Query Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=st.session_state.get("timeout_seconds", 180),
                step=30,
                help="Maximum time to wait for response (30-600 seconds)"
            )
        
        if st.button("Apply Settings", use_container_width=True):
            st.session_state.recursion_limit = new_recursion_limit
            st.session_state.timeout_seconds = new_timeout_seconds
            st.success(f"✅ Settings updated - Recursion: {new_recursion_limit}, Timeout: {new_timeout_seconds}s")
            st.rerun()

def render_executor_creation_form():
    """Executor 생성 폼 렌더링 - MCP 도구 선택 기능 포함"""
    from core.utils.config import load_mcp_configs
    from core.tools.mcp_tools import get_available_mcp_tools_info
    
    st.subheader("➕ Create New Executor")
    
    # Enhanced agent creation with detailed options
    with st.form("create_executor_form"):
        # Executor name
        executor_name = st.text_input(
            "Executor Name", 
            placeholder="e.g., Data Analyst",
            help="Choose a descriptive name for the executor"
        )
        
        # Enhanced role templates
        role_templates = {
            "Custom": "",
            "EDA Specialist": """You are an Exploratory Data Analysis Expert who uncovers hidden patterns and insights in data. 
You focus on understanding data structure, distributions, relationships, and anomalies.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Perform ALL your analysis on `df` in the SAME Python tool execution.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
            "Visualization Expert": """You are a Data Visualization Expert who creates compelling and insightful charts, graphs, and dashboards.
You excel at choosing the right visualization for the data and message.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Create visualizations using matplotlib, seaborn, or plotly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
            "ML Engineer": """You are a Machine Learning Engineer who builds, trains, and evaluates predictive models.
You handle the full ML pipeline from data preparation to model deployment.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use sklearn or other ML libraries for modeling.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
            "Data Preprocessor": """You are a Data Preprocessing Expert who cleans, transforms, and prepares data for analysis.
You handle missing values, outliers, encoding, and feature scaling.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Document all transformations clearly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
            "Statistical Analyst": """You are a Statistical Analysis Expert who performs rigorous statistical tests and modeling.
You derive meaningful insights through hypothesis testing and statistical inference.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use scipy.stats or statsmodels for analysis.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
            "Report Writer": """You are a Report Writing Expert who creates comprehensive analysis reports.
You summarize findings and communicate insights to stakeholders.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool if data analysis is needed.
2. Focus on clear, actionable insights.
3. End with 'TASK COMPLETED: [Summary]' when finished."""
        }
        
        selected_role = st.selectbox(
            "Select Role Template",
            list(role_templates.keys()),
            help="Choose a predefined role or create custom"
        )
        
        if selected_role == "Custom":
            prompt_text = st.text_area(
                "Role Description",
                height=150,
                placeholder="Describe the executor's role... Include 'TASK COMPLETED:' instruction at the end."
            )
        else:
            prompt_text = st.text_area(
                "Role Description",
                value=role_templates[selected_role],
                height=150
            )
        
        # Tool selection
        st.subheader("🔧 Tool Selection")
        
        # Python tool (always included for data science tasks)
        use_python = st.checkbox(
            "Enhanced Python Tool (SSOT)", 
            value=True,
            help="강화된 SSOT 기반 데이터 분석 환경"
        )
        
        # MCP Tools Selection
        st.markdown("#### MCP 도구 선택")
        saved_configs = load_mcp_configs()
        
        if saved_configs:
            config_names = ["None"] + [config.get('name', config['config_name']) for config in saved_configs]
            selected_mcp_config = st.selectbox(
                "MCP 설정 선택",
                config_names,
                help="사용할 MCP 도구 설정을 선택하세요"
            )
            
            selected_mcp_tools = []
            if selected_mcp_config != "None":
                config_data = next((c for c in saved_configs if c.get('name', c['config_name']) == selected_mcp_config), None)
                if config_data:
                    # 실시간 서버 상태 확인
                    tools_info = get_available_mcp_tools_info(config_data['config_name'])
                    
                    if tools_info['available']:
                        st.success(f"✅ {tools_info['available_servers']}/{tools_info['total_servers']} MCP 서버 사용 가능")
                        
                        # 사용 가능한 도구들을 체크박스로 표시
                        for tool in tools_info['tools']:
                            if tool['status'] == 'available':
                                if st.checkbox(f"🟢 {tool['server_name']}", value=True, key=f"mcp_tool_{tool['server_name']}"):
                                    selected_mcp_tools.append(tool['server_name'])
                            else:
                                st.checkbox(f"🔴 {tool['server_name']} (사용 불가)", value=False, disabled=True, key=f"mcp_tool_disabled_{tool['server_name']}")
                    else:
                        st.warning("⚠️ 선택한 설정의 MCP 서버들이 실행되지 않았습니다")
        else:
            st.info("💡 MCP 설정을 먼저 생성하세요")
        
        # Submit button
        submitted = st.form_submit_button("✨ Create Executor", type="primary", use_container_width=True)
        
        if submitted:
            if not executor_name:
                st.error("❌ Please enter an executor name")
            elif not prompt_text:
                st.error("❌ Please enter a role description")
            elif executor_name in st.session_state.get("executors", {}):
                st.error(f"❌ Executor '{executor_name}' already exists")
            else:
                # Initialize executors dict if not exists
                if "executors" not in st.session_state:
                    st.session_state.executors = {}
                
                # 기본 도구 목록 구성
                tools = ["python_repl_ast"] if use_python else []
                
                # MCP 도구 설정 추가
                mcp_config = {}
                if 'selected_mcp_config' in locals() and selected_mcp_config != "None" and selected_mcp_tools:
                    config_data = next((c for c in saved_configs if c.get('name', c['config_name']) == selected_mcp_config), None)
                    if config_data:
                        mcp_config = {
                            "config_name": selected_mcp_config,
                            "selected_tools": selected_mcp_tools,
                            "mcpServers": {tool: config_data['mcpServers'].get(tool, {}) for tool in selected_mcp_tools}
                        }
                        # MCP 도구들을 도구 목록에 추가
                        for tool in selected_mcp_tools:
                            tools.append(f"mcp:{selected_mcp_config}:{tool}")
                
                # Create executor configuration
                executor_config = {
                    "prompt": prompt_text,
                    "tools": tools,
                    "mcp_config": mcp_config,
                    "created_at": datetime.now().isoformat()
                }
                
                st.session_state.executors[executor_name] = executor_config
                st.success(f"✅ Executor '{executor_name}' created successfully!")
                
                logging.info(f"Created executor: {executor_name} with tools: {tools}")
                st.rerun()

def render_saved_systems():
    """저장된 시스템 관리 섹션 렌더링"""
    st.markdown("### 📂 Saved Multi-Agent Systems")
    
    saved_configs = load_multi_agent_configs()
    
    if saved_configs:
        selected_config = st.selectbox(
            "Load a saved system",
            ["None"] + [f"{config['name']} ({config['executors_count']} executors)" for config in saved_configs],
            help="Select a saved multi-agent system to load"
        )
        
        if selected_config != "None":
            config_name = selected_config.split(" (")[0]
            config_to_load = next((c for c in saved_configs if c['name'] == config_name), None)
            
            if config_to_load:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Load", use_container_width=True):
                        config_data = load_multi_agent_config(config_to_load['file'])
                        if config_data:
                            st.session_state.executors = config_data.get("executors", {})
                            st.session_state.current_system_name = config_data.get("name", "")
                            st.success(f"✅ Loaded '{config_name}' successfully!")
                            st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete", use_container_width=True):
                        if delete_multi_agent_config(config_to_load['file']):
                            st.success(f"✅ Deleted '{config_name}'")
                            st.rerun()
    else:
        st.info("No saved systems yet. Create and save your first one!")

def render_quick_templates():
    """빠른 시작 템플릿 렌더링 - MCP 도구 자동 할당 강화"""
    from core.tools.mcp_tools import test_mcp_server_availability, get_role_mcp_tools
    
    with st.expander("🚀 Quick Start Templates", expanded=True):
        st.markdown("### Pre-configured Systems")
        
        if st.button("🔬 Data Science Team", use_container_width=True):
            # MCP 서버 가용성 확인
            with st.spinner("MCP 서버 상태 확인 중..."):
                try:
                    # 비동기 함수를 동기적으로 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    available_servers = loop.run_until_complete(test_mcp_server_availability())
                    loop.close()
                except Exception as e:
                    logging.warning(f"MCP server availability check failed: {e}")
                    available_servers = {}
            
            # 기본 Data Science 팀 구성 with MCP tools
            st.session_state.executors = {}
            
            team_roles = [
                ("Data_Preprocessor", "Data_Preprocessor"),
                ("EDA_Specialist", "EDA_Specialist"), 
                ("Visualization_Expert", "Visualization_Expert"),
                ("ML_Engineer", "ML_Engineer"),
                ("Statistical_Analyst", "Statistical_Analyst"),
                ("Report_Writer", "Report_Writer")
            ]
            
            for executor_name, role_name in team_roles:
                # 역할별 MCP 도구 할당
                tools, mcp_config = get_role_mcp_tools(role_name, available_servers)
                
                role_prompts = {
                    "Data_Preprocessor": """You are a Data Preprocessing Expert who cleans, transforms, and prepares data for analysis.
You handle missing values, outliers, encoding, and feature scaling.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Document all transformations clearly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "EDA_Specialist": """You are an Exploratory Data Analysis Expert who uncovers hidden patterns and insights in data. 
You focus on understanding data structure, distributions, relationships, and anomalies.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Perform ALL your analysis on `df` in the SAME Python tool execution.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "Visualization_Expert": """You are a Data Visualization Expert who creates compelling and insightful charts, graphs, and dashboards.
You excel at choosing the right visualization for the data and message.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Create visualizations using matplotlib, seaborn, or plotly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "ML_Engineer": """You are a Machine Learning Engineer who builds, trains, and evaluates predictive models.
You handle the full ML pipeline from data preparation to model deployment.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use sklearn or other ML libraries for modeling.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "Statistical_Analyst": """You are a Statistical Analysis Expert who performs rigorous statistical tests and modeling.
You derive meaningful insights through hypothesis testing and statistical inference.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use scipy.stats or statsmodels for analysis.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "Report_Writer": """You are a Report Writing Expert who creates comprehensive analysis reports.
You summarize findings and communicate insights to stakeholders.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool if data analysis is needed.
2. Focus on clear, actionable insights.
3. End with 'TASK COMPLETED: [Summary]' when finished."""
                }
                
                st.session_state.executors[executor_name] = {
                    "prompt": role_prompts[role_name],
                    "tools": tools,
                    "mcp_config": mcp_config,
                    "created_at": datetime.now().isoformat()
                }
            
            # 사용 가능한 MCP 서버 개수 표시
            available_count = sum(available_servers.values())
            total_count = len(available_servers)
            
            if available_count > 0:
                st.success(f"✅ Data Science Team template loaded! ({available_count}/{total_count} MCP servers available)")
            else:
                st.success("✅ Data Science Team template loaded! (Python tools only)")
                st.info("💡 MCP 서버를 실행하면 더 많은 도구를 사용할 수 있습니다")
            
            st.rerun()

# Helper functions
def load_multi_agent_configs():
    """Load all saved multi-agent configurations"""
    config_dir = Path("multi-agent-configs")
    configs = []
    
    if config_dir.exists():
        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    config = json.load(f)
                    configs.append({
                        "name": config.get("name", json_file.stem),
                        "file": json_file.stem,
                        "created_at": config.get("created_at", "Unknown"),
                        "executors_count": len(config.get("executors", {})),
                        "description": config.get("description", "")
                    })
            except Exception as e:
                logging.error(f"Failed to load config {json_file}: {e}")
    
    return sorted(configs, key=lambda x: x.get("created_at", ""), reverse=True)

def load_multi_agent_config(filename: str):
    """Load a specific multi-agent configuration"""
    config_dir = Path("multi-agent-configs")
    config_file = config_dir / f"{filename}.json"
    
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return json.load(f)
    return None

def delete_multi_agent_config(filename: str):
    """Delete a multi-agent configuration"""
    config_dir = Path("multi-agent-configs")
    config_file = config_dir / f"{filename}.json"
    
    if config_file.exists():
        config_file.unlink()
        return True
    return False

def save_multi_agent_config(name: str, config: dict):
    """Save multi-agent configuration to file"""
    config_dir = Path("multi-agent-configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / f"{name}.json"
    
    # Save configuration
    save_data = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "executors": config.get("executors", {}),
        "description": config.get("description", "")
    }
    
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    return config_file