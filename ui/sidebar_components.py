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

def render_data_upload_section():
    """데이터 업로드 섹션 렌더링"""
    from core import data_manager, data_lineage_tracker
    
    st.markdown("### �� 강화된 데이터 분석")
    
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
    """Executor 생성 폼 렌더링"""
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
        
        # MCP Tools Selection (simplified for now)
        st.info("📦 MCP tools can be added in future versions")
        
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
                
                # Create executor configuration
                executor_config = {
                    "prompt": prompt_text,
                    "tools": ["python_repl_ast"] if use_python else [],
                    "created_at": datetime.now().isoformat()
                }
                
                st.session_state.executors[executor_name] = executor_config
                st.success(f"✅ Executor '{executor_name}' created successfully!")
                
                logging.info(f"Created executor: {executor_name}")
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
    """빠른 시작 템플릿 렌더링"""
    with st.expander("🚀 Quick Start Templates", expanded=True):
        st.markdown("### Pre-configured Systems")
        
        if st.button("🔬 Data Science Team", use_container_width=True):
            # 기본 Data Science 팀 구성
            st.session_state.executors = {
                "Data_Preprocessor": {
                    "prompt": """You are a Data Preprocessing Expert who cleans, transforms, and prepares data for analysis.
You handle missing values, outliers, encoding, and feature scaling.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Document all transformations clearly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                },
                "EDA_Specialist": {
                    "prompt": """You are an Exploratory Data Analysis Expert who uncovers hidden patterns and insights in data. 
You focus on understanding data structure, distributions, relationships, and anomalies.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Perform ALL your analysis on `df` in the SAME Python tool execution.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                },
                "Visualization_Expert": {
                    "prompt": """You are a Data Visualization Expert who creates compelling and insightful charts, graphs, and dashboards.
You excel at choosing the right visualization for the data and message.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Create visualizations using matplotlib, seaborn, or plotly.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                },
                "ML_Engineer": {
                    "prompt": """You are a Machine Learning Engineer who builds, trains, and evaluates predictive models.
You handle the full ML pipeline from data preparation to model deployment.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use sklearn or other ML libraries for modeling.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                },
                "Statistical_Analyst": {
                    "prompt": """You are a Statistical Analysis Expert who performs rigorous statistical tests and modeling.
You derive meaningful insights through hypothesis testing and statistical inference.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool.
2. Use scipy.stats or statsmodels for analysis.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                },
                "Report_Writer": {
                    "prompt": """You are a Report Writing Expert who creates comprehensive analysis reports.
You summarize findings and communicate insights to stakeholders.

CRITICAL DATA HANDLING RULE: You MUST use the `python_repl_ast` tool for ALL data operations.
1. ALWAYS start by calling `df = get_current_data()` within the Python tool if data analysis is needed.
2. Focus on clear, actionable insights.
3. End with 'TASK COMPLETED: [Summary]' when finished.""",
                    "tools": ["python_repl_ast"],
                    "created_at": datetime.now().isoformat()
                }
            }
            
            st.success("✅ Data Science Team template loaded!")
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