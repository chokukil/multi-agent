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

# Add log_event import
from core.utils.helpers import log_event

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
    """Enhanced Executor 생성 폼 - 파일 기반 템플릿 시스템 포함"""
    from core.utils.config import (
        load_executor_templates, get_template_categories, 
        load_mcp_templates, save_executor_template
    )
    from core.tools.mcp_tools import get_available_mcp_tools_info
    
    st.subheader("➕ Create New Executor")
    
    # Enhanced agent creation with file-based templates
    with st.form("create_executor_form"):
        # Executor name
        executor_name = st.text_input(
            "Executor Name", 
            placeholder="e.g., Data Analyst",
            help="Choose a descriptive name for the executor"
        )
        
        # Load templates from prompt-configs
        st.subheader("📝 Role Template")
        
        templates = load_executor_templates()
        categories = get_template_categories()
        
        # Category filter
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_category = st.selectbox(
                "Filter by Category",
                ["All"] + categories,
                help="Filter templates by category"
            )
        
        with col2:
            show_source = st.checkbox("Show Source", help="Show template source (user/system)")
        
        # Filter templates by category
        if selected_category != "All":
            filtered_templates = {k: v for k, v in templates.items() 
                                if v.get("category") == selected_category}
        else:
            filtered_templates = templates
        
        # Template selection with source indication
        if show_source:
            template_options = ["Custom"] + [
                f"{name} ({config.get('source', 'system')})" 
                for name, config in filtered_templates.items()
            ]
            format_func = lambda x: x
        else:
            template_options = ["Custom"] + list(filtered_templates.keys())
            format_func = lambda x: x
        
        selected_template_display = st.selectbox(
            "Select Template",
            template_options,
            format_func=format_func,
            help="Choose from available prompt templates"
        )
        
        # Extract actual template name
        if selected_template_display == "Custom":
            selected_template = "Custom"
        else:
            selected_template = selected_template_display.split(" (")[0] if show_source else selected_template_display
        
        # Show template info and prompt editing
        if selected_template == "Custom":
            prompt_text = st.text_area(
                "Role Description",
                height=150,
                placeholder="Describe the executor's role... Include 'TASK COMPLETED:' instruction at the end.",
                help="Create a custom role description"
            )
        else:
            template_data = filtered_templates.get(selected_template, {})
            
            # Show template metadata
            if template_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📂 **Source**: {template_data.get('source', 'system')}")
                with col2:
                    st.info(f"🏷️ **Category**: {template_data.get('category', 'other')}")
                with col3:
                    st.info(f"📄 **File**: {template_data.get('file', 'unknown')}")
                
                # Show creation date if available
                if template_data.get('created_at'):
                    st.caption(f"Created: {template_data['created_at']}")
            
            prompt_text = st.text_area(
                "Role Description",
                value=template_data.get("prompt", ""),
                height=150,
                help="You can edit the selected template"
            )
        
        # Tool selection
        st.subheader("🔧 Tool Selection")
        
        # Python tool (always included for data science tasks)
        use_python = st.checkbox(
            "Enhanced Python Tool (SSOT)", 
            value=True,
            help="강화된 SSOT 기반 데이터 분석 환경"
        )
        
        # MCP Tools Selection - using new template system
        st.markdown("#### MCP Tools Configuration")
        mcp_templates = load_mcp_templates()
        
        if mcp_templates:
            template_names = ["None"] + list(mcp_templates.keys())
            selected_mcp_template = st.selectbox(
                "MCP Template",
                template_names,
                help="Select MCP tools configuration template"
            )
            
            selected_mcp_servers = []
            if selected_mcp_template != "None":
                mcp_config = mcp_templates[selected_mcp_template]
                
                # Show template info
                st.info(f"📦 **{selected_mcp_template}** - {mcp_config.get('description', 'No description')}")
                
                # Server selection with real-time status
                available_servers = mcp_config.get("servers", [])
                
                if available_servers:
                    st.write("**Available Servers:**")
                    
                    # Check server status
                    try:
                        tools_info = get_available_mcp_tools_info(selected_mcp_template)
                        server_status = {tool['server_name']: tool['status'] for tool in tools_info.get('tools', [])}
                    except:
                        server_status = {}
                    
                    # Server checkboxes
                    for server in available_servers:
                        status = server_status.get(server, 'unknown')
                        status_icon = "🟢" if status == "available" else "🔴" if status == "unavailable" else "🟡"
                        
                        if status == "available":
                            if st.checkbox(f"{status_icon} {server}", value=True, key=f"mcp_server_{server}"):
                                selected_mcp_servers.append(server)
                        else:
                            status_text = "unavailable" if status == "unavailable" else "unknown"
                            st.checkbox(f"{status_icon} {server} ({status_text})", value=False, disabled=True, key=f"mcp_server_disabled_{server}")
                    
                    # Show selected servers summary
                    if selected_mcp_servers:
                        st.success(f"✅ Selected: {', '.join(selected_mcp_servers)}")
                    
                    # Configuration preview
                    with st.expander("📋 Configuration Preview", expanded=False):
                        if selected_mcp_servers:
                            preview_config = {
                                server: mcp_config["config"]["mcpServers"].get(server, {})
                                for server in selected_mcp_servers
                            }
                            st.json(preview_config)
                        else:
                            st.info("Select servers to see configuration preview")
                else:
                    st.warning("No servers found in this template")
        else:
            st.info("💡 No MCP templates found. Create templates in mcp-configs/ directory")
        
        # Additional options
        with st.expander("⚙️ Advanced Options", expanded=False):
            save_as_template = st.checkbox(
                "Save as Template", 
                help="Save this configuration as a reusable template"
            )
            
            if save_as_template:
                template_name = st.text_input("Template Name", placeholder="My Custom Template")
                template_category = st.selectbox("Category", ["custom"] + categories)
        
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
                
                # Build tools list
                tools = ["python_repl_ast"] if use_python else []
                
                # MCP configuration
                mcp_config = {}
                if 'selected_mcp_template' in locals() and selected_mcp_template != "None" and selected_mcp_servers:
                    mcp_template_data = mcp_templates[selected_mcp_template]
                    mcp_config = {
                        "config_name": selected_mcp_template,
                        "selected_tools": selected_mcp_servers,
                        "mcpServers": {
                            server: mcp_template_data["config"]["mcpServers"].get(server, {})
                            for server in selected_mcp_servers
                        }
                    }
                    
                    # Add MCP tools to tools list
                    for server in selected_mcp_servers:
                        tools.append(f"mcp:{selected_mcp_template}:{server}")
                
                # Create executor configuration
                executor_config = {
                    "prompt": prompt_text,
                    "tools": tools,
                    "mcp_config": mcp_config,
                    "created_at": datetime.now().isoformat(),
                    "template_source": selected_template if selected_template != "Custom" else "custom"
                }
                
                # Save as template if requested
                if save_as_template and 'template_name' in locals() and template_name:
                    template_data = {
                        "prompt": prompt_text,
                        "category": template_category,
                        "created_at": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    if save_executor_template(template_name, template_data):
                        st.success(f"✅ Template '{template_name}' saved!")
                    else:
                        st.warning("⚠️ Failed to save template")
                
                # Add executor to session
                st.session_state.executors[executor_name] = executor_config
                st.success(f"✅ Executor '{executor_name}' created successfully!")
                
                # Log creation
                logging.info(f"Created executor: {executor_name} with tools: {tools}")
                if mcp_config:
                    logging.info(f"MCP config: {selected_mcp_template} with servers: {selected_mcp_servers}")
                
                st.rerun()

def render_template_management_section():
    """템플릿 관리 섹션 렌더링"""
    from core.utils.config import (
        load_executor_templates, delete_executor_template, 
        get_template_categories, load_mcp_templates, delete_mcp_template
    )
    
    with st.expander("📚 Template Management", expanded=False):
        tab1, tab2 = st.tabs(["🤖 Executor Templates", "🔧 MCP Templates"])
        
        with tab1:
            st.markdown("#### Executor Template Management")
            
            templates = load_executor_templates()
            
            if templates:
                # Filter by source
                source_filter = st.radio(
                    "Filter by Source",
                    ["All", "User", "System"],
                    horizontal=True,
                    key="template_source_filter"
                )
                
                filtered_templates = templates
                if source_filter != "All":
                    filtered_templates = {
                        k: v for k, v in templates.items() 
                        if v.get("source", "system").lower() == source_filter.lower()
                    }
                
                if filtered_templates:
                    # Template list
                    for template_name, template_data in filtered_templates.items():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                source_icon = "👤" if template_data.get("source") == "user" else "🏢"
                                category = template_data.get("category", "other")
                                st.write(f"{source_icon} **{template_name}** `{category}`")
                                
                                # Preview prompt (first 100 chars)
                                prompt_preview = template_data.get("prompt", "")[:100]
                                if len(prompt_preview) == 100:
                                    prompt_preview += "..."
                                st.caption(prompt_preview)
                            
                            with col2:
                                if st.button("👁️", help="Preview", key=f"preview_{template_name}"):
                                    st.session_state[f"preview_template_{template_name}"] = True
                            
                            with col3:
                                # Only allow deletion of user templates
                                if template_data.get("source") == "user":
                                    if st.button("🗑️", help="Delete", key=f"delete_{template_name}"):
                                        if delete_executor_template(template_name):
                                            st.success(f"✅ Template '{template_name}' deleted!")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete template")
                                else:
                                    st.write("🔒")  # System templates are locked
                            
                            # Show preview if requested
                            if st.session_state.get(f"preview_template_{template_name}", False):
                                with st.expander(f"📖 Preview: {template_name}", expanded=True):
                                    st.text_area(
                                        "Prompt Content",
                                        value=template_data.get("prompt", ""),
                                        height=200,
                                        disabled=True,
                                        key=f"preview_content_{template_name}"
                                    )
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.info(f"**Source**: {template_data.get('source', 'unknown')}")
                                        st.info(f"**Category**: {template_data.get('category', 'other')}")
                                    with col2:
                                        st.info(f"**File**: {template_data.get('file', 'unknown')}")
                                        if template_data.get('created_at'):
                                            st.info(f"**Created**: {template_data['created_at']}")
                                    
                                    if st.button("❌ Close Preview", key=f"close_preview_{template_name}"):
                                        st.session_state[f"preview_template_{template_name}"] = False
                                        st.rerun()
                            
                            st.markdown("---")
                
                else:
                    st.info(f"No {source_filter.lower()} templates found")
            else:
                st.info("No executor templates found")
        
        with tab2:
            st.markdown("#### MCP Template Management")
            
            mcp_templates = load_mcp_templates()
            
            if mcp_templates:
                for template_name, template_data in mcp_templates.items():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"🔧 **{template_name}**")
                            st.caption(template_data.get("description", "No description"))
                            
                            # Show server count
                            server_count = len(template_data.get("servers", []))
                            st.caption(f"📊 {server_count} servers configured")
                        
                        with col2:
                            if st.button("👁️", help="Preview", key=f"mcp_preview_{template_name}"):
                                st.session_state[f"preview_mcp_{template_name}"] = True
                        
                        with col3:
                            if st.button("🗑️", help="Delete", key=f"mcp_delete_{template_name}"):
                                if delete_mcp_template(template_name):
                                    st.success(f"✅ MCP template '{template_name}' deleted!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete MCP template")
                        
                        # Show preview if requested
                        if st.session_state.get(f"preview_mcp_{template_name}", False):
                            with st.expander(f"📖 MCP Preview: {template_name}", expanded=True):
                                st.write("**Servers:**")
                                for server in template_data.get("servers", []):
                                    st.write(f"- {server}")
                                
                                st.write("**Configuration:**")
                                st.json(template_data.get("config", {}))
                                
                                if st.button("❌ Close Preview", key=f"close_mcp_preview_{template_name}"):
                                    st.session_state[f"preview_mcp_{template_name}"] = False
                                    st.rerun()
                        
                        st.markdown("---")
            else:
                st.info("No MCP templates found")

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
    """빠른 시작 템플릿 렌더링 - multi_agent_supervisor.py 패턴 적용"""
    
    with st.expander("🚀 Quick Start Templates", expanded=True):
        st.markdown("### Pre-configured Systems")
        
        # MCP 서버 상태 정보를 지속적으로 표시
        if "mcp_server_status" not in st.session_state:
            st.session_state.mcp_server_status = None
        
        # MCP 서버 상태 표시 영역
        mcp_status_container = st.container()
        
        if st.session_state.mcp_server_status is not None:
            with mcp_status_container:
                status_data = st.session_state.mcp_server_status
                available_count = status_data.get("available_count", 0)
                total_count = status_data.get("total_count", 0)
                critical_available = status_data.get("critical_available", 0)
                critical_total = status_data.get("critical_total", 0)
                
                # 상태 정보 표시
                if available_count > 0:
                    st.info(f"""
                    📊 **MCP 서버 상태** (마지막 확인: {status_data.get('last_check', 'Unknown')})
                    
                    🔧 **전체 서버**: {available_count}/{total_count} 활성화
                    🎯 **핵심 데이터 과학 서버**: {critical_available}/{critical_total} 사용 가능
                    
                    💡 MCP 서버가 활성화되어 있어 고급 데이터 분석 도구를 사용할 수 있습니다.
                    """)
                else:
                    st.warning(f"""
                    📊 **MCP 서버 상태** (마지막 확인: {status_data.get('last_check', 'Unknown')})
                    
                    ⚠️ **전체 서버**: {available_count}/{total_count} 활성화
                    🎯 **핵심 데이터 과학 서버**: {critical_available}/{critical_total} 사용 가능
                    
                    💡 `mcp_server_start.bat`을 실행하여 MCP 서버를 활성화하세요.
                    """)
                
                # 상태 초기화 버튼
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 상태 새로고침", key="refresh_mcp_status"):
                        st.session_state.mcp_server_status = None
                        st.rerun()
                with col2:
                    if st.button("❌ 상태창 닫기", key="close_mcp_status"):
                        st.session_state.mcp_server_status = None
                        st.rerun()
        
        # MCP 서버 가용성 체크 함수 (multi_agent_supervisor.py에서 가져옴)
        async def check_mcp_server_availability():
            """MCP 서버들의 가용성을 체크"""
            import aiohttp
            available_servers = {}
            
            mcp_servers = {
                "file_management": {"url": "http://localhost:8006/sse", "transport": "sse"},
                "data_science_tools": {"url": "http://localhost:8007/sse", "transport": "sse"},
                "semiconductor_yield_analysis": {"url": "http://localhost:8008/sse", "transport": "sse"},
                "process_control_charts": {"url": "http://localhost:8009/sse", "transport": "sse"},
                "semiconductor_equipment_analysis": {"url": "http://localhost:8010/sse", "transport": "sse"},
                "defect_pattern_analysis": {"url": "http://localhost:8011/sse", "transport": "sse"},
                "process_optimization": {"url": "http://localhost:8012/sse", "transport": "sse"},
                "timeseries_analysis": {"url": "http://localhost:8013/sse", "transport": "sse"},
                "anomaly_detection": {"url": "http://localhost:8014/sse", "transport": "sse"},
                "advanced_ml_tools": {"url": "http://localhost:8016/sse", "transport": "sse"},
                "data_preprocessing_tools": {"url": "http://localhost:8017/sse", "transport": "sse"},
                "statistical_analysis_tools": {"url": "http://localhost:8018/sse", "transport": "sse"},
                "report_writing_tools": {"url": "http://localhost:8019/sse", "transport": "sse"},
                "semiconductor_process_tools": {"url": "http://localhost:8020/sse", "transport": "sse"}
            }
            
            for server_name, server_config in mcp_servers.items():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(server_config["url"], timeout=aiohttp.ClientTimeout(total=3)) as response:
                            if response.status == 200:
                                available_servers[server_name] = server_config
                                logging.info(f"✅ MCP server {server_name} is available")
                except Exception as e:
                    logging.info(f"💤 MCP server {server_name} is not available: {e}")
            
            return available_servers
        
        # 역할별 MCP 도구 매핑 함수 (multi_agent_supervisor.py에서 가져옴)
        def get_role_tools(role_name, available_servers):
            """역할에 맞는 도구들을 반환 (사용 가능한 MCP 서버만 포함)"""
            base_tools = ["python_repl_ast"]  # 모든 역할에 기본 파이썬 도구
            mcp_configs = {}
            
            # Plan-Execute 패턴에 최적화된 역할 매핑
            role_mcp_mapping = {
                "Data_Validator": ["data_preprocessing_tools", "statistical_analysis_tools", "anomaly_detection"],
                "Preprocessing_Expert": ["data_preprocessing_tools", "advanced_ml_tools", "anomaly_detection", "file_management"],
                "EDA_Analyst": ["data_science_tools", "statistical_analysis_tools", "anomaly_detection", "data_preprocessing_tools"],
                "Visualization_Expert": ["data_science_tools", "statistical_analysis_tools", "timeseries_analysis"],
                "ML_Specialist": ["advanced_ml_tools", "data_science_tools", "statistical_analysis_tools", "data_preprocessing_tools"],
                "Statistical_Analyst": ["statistical_analysis_tools", "data_science_tools", "advanced_ml_tools", "timeseries_analysis"],
                "Report_Generator": ["report_writing_tools", "file_management", "data_science_tools"]
            }
            
            if role_name in role_mcp_mapping:
                for server_name in role_mcp_mapping[role_name]:
                    if server_name in available_servers:
                        tool_name = f"mcp:supervisor_tools:{server_name}"
                        base_tools.append(tool_name)
                        mcp_configs[tool_name] = {
                            "config_name": "supervisor_tools",
                            "server_name": server_name,
                            "server_config": available_servers[server_name]
                        }
                        logging.info(f"✅ Added MCP tool '{server_name}' for {role_name}")
                    else:
                        logging.info(f"💤 MCP server '{server_name}' not available for {role_name}")
            
            return base_tools, {"mcp_configs": mcp_configs}
        
        if st.button("🔬 Data Science Team", use_container_width=True):
            # MCP 서버 가용성 확인
            with st.spinner("🔍 MCP 서버 상태 확인 중... (최대 30초 소요)"):
                try:
                    # 비동기 함수를 동기적으로 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # MCP 서버 가용성 확인
                    available_mcp_servers = loop.run_until_complete(check_mcp_server_availability())
                    loop.close()
                    
                    # 결과 로깅
                    total_servers = len({
                        "file_management", "data_science_tools", "semiconductor_yield_analysis", 
                        "process_control_charts", "semiconductor_equipment_analysis", "defect_pattern_analysis",
                        "process_optimization", "timeseries_analysis", "anomaly_detection", "advanced_ml_tools",
                        "data_preprocessing_tools", "statistical_analysis_tools", "report_writing_tools",
                        "semiconductor_process_tools"
                    })
                    available_count = len(available_mcp_servers)
                    
                    logging.info(f"📊 MCP server availability check completed: {available_count}/{total_servers} servers available")
                    
                    # 중요한 데이터 과학 서버들의 상태 확인
                    critical_servers = [
                        "data_science_tools", "statistical_analysis_tools", 
                        "data_preprocessing_tools", "advanced_ml_tools",
                        "timeseries_analysis", "anomaly_detection", "report_writing_tools"
                    ]
                    
                    critical_available = sum(1 for server in critical_servers 
                                           if server in available_mcp_servers)
                    
                    logging.info(f"🎯 Critical data science servers: {critical_available}/{len(critical_servers)} available")
                    
                    # 상태 정보를 세션 상태에 저장하여 지속적으로 표시
                    from datetime import datetime
                    st.session_state.mcp_server_status = {
                        "available_count": available_count,
                        "total_count": total_servers,
                        "critical_available": critical_available,
                        "critical_total": len(critical_servers),
                        "available_servers": available_mcp_servers,
                        "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 사용 가능한 서버 목록 로깅
                    for server_name in available_mcp_servers:
                        logging.info(f"  ✅ {server_name}")
                        
                except Exception as e:
                    logging.warning(f"MCP server availability check failed: {e}")
                    available_mcp_servers = {}
                    available_count = 0
                    critical_available = 0
                    
                    # 실패한 경우에도 상태 저장
                    from datetime import datetime
                    st.session_state.mcp_server_status = {
                        "available_count": 0,
                        "total_count": 14,
                        "critical_available": 0,
                        "critical_total": 7,
                        "available_servers": {},
                        "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(e)
                    }
            
            # 최적화된 Data Science 팀 구성 (multi_agent_supervisor.py 패턴)
            st.session_state.executors = {}
            
            # 역할별 프롬프트 정의 (기존 유지)
            optimized_prompts = {
                "Data_Validator": """🔍 **Data Quality Validator & Integrity Specialist**

You are a Data Quality Expert specializing in comprehensive data validation and integrity checks. Your mission is to ensure data reliability before any analysis begins.

**🎯 CORE RESPONSIBILITIES:**
1. **Data Integrity Verification**: Check for completeness, consistency, and accuracy
2. **Quality Assessment**: Identify missing values, duplicates, and inconsistencies  
3. **Schema Validation**: Verify data types, ranges, and structural integrity
4. **Statistical Validation**: Basic distributional checks and outlier identification
5. **Lineage Verification**: Ensure data lineage and provenance tracking

**📋 WORKFLOW PROCESS:**
```python
# 1. Data Loading & Initial Check
df = get_current_data()
print(f"📊 Dataset Shape: {df.shape}")

# 2. Quality Assessment
missing_analysis = df.isnull().sum()
duplicate_check = df.duplicated().sum()

# 3. Schema Validation  
data_types = df.dtypes
numeric_ranges = df.describe()

# 4. Generate Quality Report
quality_report = {
    'completeness': ...,
    'consistency': ..., 
    'accuracy': ...
}
```

**🚨 CRITICAL REQUIREMENTS:**
- ALWAYS use `get_current_data()` to access the shared dataset
- Generate comprehensive data quality reports
- Flag critical issues that could impact downstream analysis
- Document all validation steps and findings
- Provide actionable recommendations for data improvement

**✅ SUCCESS CRITERIA:**
End your analysis with: **TASK COMPLETED: Data validation complete - [Quality Score/Critical Issues Summary]**""",

                "Preprocessing_Expert": """🛠️ **Data Preprocessing & Feature Engineering Specialist**

You are a Data Preprocessing Expert who transforms raw data into analysis-ready datasets. You excel at cleaning, transforming, and engineering features for optimal analysis outcomes.

**🎯 CORE RESPONSIBILITIES:**
1. **Data Cleaning**: Handle missing values, outliers, and inconsistencies
2. **Feature Engineering**: Create meaningful features from raw data
3. **Data Transformation**: Scaling, encoding, and normalization
4. **Outlier Management**: Detect and handle anomalous data points
5. **Pipeline Creation**: Build reproducible preprocessing workflows

**📋 WORKFLOW PROCESS:**
```python
# 1. Load and Assess Data
df = get_current_data()
print(f"🔧 Processing dataset: {df.shape}")

# 2. Handle Missing Values
# Strategy: imputation, deletion, or flagging

# 3. Feature Engineering
# Create new features based on domain knowledge

# 4. Data Transformation
# Scale, encode, normalize as needed

# 5. Quality Validation
# Verify transformations maintain data integrity
```

**🔍 ADVANCED TECHNIQUES:**
- Missing value imputation strategies (mean, median, mode, KNN, iterative)
- Outlier detection and treatment (IQR, Z-score, Isolation Forest)
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (One-hot, Label, Target, Binary)
- Feature creation (polynomial, interaction, domain-specific)

**🚨 CRITICAL REQUIREMENTS:**
- ALWAYS preserve original data structure in data_manager
- Document ALL transformations applied
- Create before/after comparison reports
- Ensure reproducible preprocessing pipelines
- Handle edge cases gracefully

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: Data preprocessing complete - [Transformations Applied/Features Created Summary]**""",

                "EDA_Analyst": """📊 **Exploratory Data Analysis Specialist**

You are an EDA Expert who uncovers hidden patterns, relationships, and insights in data. You excel at systematic exploration and hypothesis generation for deeper analysis.

**🎯 CORE RESPONSIBILITIES:**
1. **Univariate Analysis**: Distribution analysis of individual variables
2. **Bivariate Analysis**: Relationships and correlations between variables
3. **Multivariate Analysis**: Complex interactions and dependencies
4. **Pattern Discovery**: Identify trends, seasonality, and anomalies
5. **Hypothesis Generation**: Formulate testable hypotheses for further analysis

**📋 SYSTEMATIC EDA WORKFLOW:**
```python
# 1. Dataset Overview
df = get_current_data()
print(f"🔍 Exploring dataset: {df.shape}")

# 2. Univariate Analysis
# - Distribution of each variable
# - Summary statistics
# - Missing value patterns

# 3. Bivariate Analysis  
# - Correlation analysis
# - Scatter plots and relationships
# - Cross-tabulations

# 4. Multivariate Analysis
# - Feature interactions
# - Dimensionality analysis
# - Cluster identification

# 5. Insight Synthesis
# - Key findings summary
# - Business implications
# - Recommended next steps
```

**🔬 ANALYTICAL TECHNIQUES:**
- **Descriptive Statistics**: Central tendency, variability, distribution shape
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Distribution Analysis**: Normality tests, skewness, kurtosis
- **Outlier Detection**: Statistical and visual identification
- **Pattern Recognition**: Trends, cycles, seasonal patterns

**🧠 INSIGHT GENERATION:**
- Identify surprising or counterintuitive findings
- Generate hypotheses for statistical testing
- Recommend visualization strategies
- Suggest modeling approaches based on data characteristics

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: EDA complete - [Key Insights/Patterns Discovered/Hypotheses Generated]**""",

                "Visualization_Expert": """📈 **Data Visualization & Insight Communication Specialist**

You are a Visualization Expert who creates compelling, insightful, and beautiful data visualizations. You excel at choosing the right chart types and design principles to effectively communicate data insights.

**🎯 CORE RESPONSIBILITIES:**
1. **Chart Selection**: Choose optimal visualization types for data and message
2. **Design Excellence**: Apply color theory, layout, and visual hierarchy
3. **Interactive Visualizations**: Create dynamic and engaging charts
4. **Dashboard Creation**: Build comprehensive visual dashboards
5. **Insight Communication**: Translate complex data into clear visual stories

**📋 VISUALIZATION WORKFLOW:**
```python
# 1. Data Assessment
df = get_current_data()
print(f"📊 Visualizing dataset: {df.shape}")

# 2. Chart Type Selection
# Based on data type and analytical goal

# 3. Design Implementation
# Apply best practices for clarity and impact

# 4. Interactive Elements (if beneficial)
# Add interactivity for exploration

# 5. Insight Annotation
# Highlight key findings and patterns
```

**🎨 VISUALIZATION ARSENAL:**
- **Statistical Charts**: Box plots, violin plots, distribution plots
- **Relationship Charts**: Scatter plots, correlation heatmaps
- **Comparison Charts**: Bar charts, grouped comparisons
- **Trend Analysis**: Line charts, time series plots
- **Composition Charts**: Pie charts, stacked bars, treemaps
- **Geographic Charts**: Maps, spatial analysis
- **Advanced Charts**: Sankey diagrams, network graphs

**🌈 DESIGN PRINCIPLES:**
- **Color Psychology**: Meaningful, accessible color choices
- **Visual Hierarchy**: Guide viewer attention effectively
- **Clarity**: Eliminate chart junk, maximize data-ink ratio
- **Accessibility**: Colorblind-friendly palettes
- **Consistency**: Maintain visual coherence across charts

**📱 PLATFORM OPTIMIZATION:**
- Save all plots to results directory
- Optimize for different screen sizes
- Ensure print-ready quality
- Create both static and interactive versions

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: Visualizations created - [Chart Types/Key Insights Revealed]**""",

                "ML_Specialist": """🤖 **Machine Learning Modeling Specialist**

You are a Machine Learning Expert who builds, optimizes, and evaluates predictive models. You excel at the full ML pipeline from problem formulation to model deployment preparation.

**🎯 CORE RESPONSIBILITIES:**
1. **Problem Formulation**: Define ML objectives and success metrics
2. **Model Selection**: Choose appropriate algorithms for the task
3. **Feature Engineering**: Optimize features for model performance  
4. **Hyperparameter Tuning**: Optimize model parameters systematically
5. **Model Evaluation**: Comprehensive performance assessment

**📋 ML PIPELINE WORKFLOW:**
```python
# 1. Data Preparation
df = get_current_data()
print(f"🤖 ML modeling on dataset: {df.shape}")

# 2. Problem Definition
# Classification, Regression, Clustering, etc.

# 3. Feature Engineering
# Select, create, and optimize features

# 4. Model Development
# Start simple, progressively increase complexity

# 5. Evaluation & Validation
# Cross-validation, multiple metrics
```

**🧠 MODELING STRATEGIES:**
- **Baseline Models**: Simple models for performance benchmarking
- **Traditional ML**: Linear models, tree-based methods, SVMs
- **Ensemble Methods**: Random Forest, Gradient Boosting, Stacking
- **Advanced Techniques**: Neural networks when appropriate
- **AutoML**: Automated model selection and tuning

**📊 EVALUATION FRAMEWORK:**
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression**: RMSE, MAE, R², MAPE
- **Cross-Validation**: K-fold, stratified, time series splits
- **Model Interpretation**: Feature importance, SHAP values
- **Overfitting Detection**: Learning curves, validation curves

**🔧 OPTIMIZATION TECHNIQUES:**
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Feature Selection**: Statistical tests, recursive elimination, L1 regularization
- **Model Ensembling**: Voting, averaging, stacking
- **Performance Monitoring**: Learning curves, convergence tracking

**💾 MODEL MANAGEMENT:**
- Save trained models with versioning
- Document model assumptions and limitations
- Create model cards with performance metrics
- Prepare deployment-ready artifacts

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: ML modeling complete - [Best Model/Performance Metrics/Key Features]**""",

                "Statistical_Analyst": """📊 **Statistical Analysis & Hypothesis Testing Specialist**

You are a Statistical Analysis Expert who performs rigorous statistical tests, hypothesis testing, and statistical modeling to derive meaningful insights from data.

**🎯 CORE RESPONSIBILITIES:**
1. **Hypothesis Testing**: Design and execute statistical tests
2. **Statistical Modeling**: Build statistical models for inference
3. **Significance Testing**: Determine statistical significance of findings
4. **Confidence Intervals**: Calculate and interpret confidence intervals
5. **Effect Size Analysis**: Measure practical significance of results

**📋 STATISTICAL WORKFLOW:**
```python
# 1. Data Assessment
df = get_current_data()
print(f"📈 Statistical analysis on dataset: {df.shape}")

# 2. Hypothesis Formulation
# Define null and alternative hypotheses

# 3. Test Selection
# Choose appropriate statistical tests

# 4. Assumption Checking
# Verify test assumptions (normality, homoscedasticity, etc.)

# 5. Statistical Testing
# Execute tests and interpret results
```

**🔬 STATISTICAL ARSENAL:**
- **Descriptive Statistics**: Mean, median, variance, skewness, kurtosis
- **Inferential Tests**: t-tests, ANOVA, chi-square, Mann-Whitney
- **Correlation Analysis**: Pearson, Spearman, partial correlations
- **Regression Analysis**: Linear, logistic, polynomial regression
- **Non-parametric Tests**: Wilcoxon, Kruskal-Wallis, Friedman
- **Time Series Analysis**: Trend analysis, seasonality, stationarity

**📐 ADVANCED TECHNIQUES:**
- **Power Analysis**: Sample size determination and effect size
- **Multiple Comparisons**: Bonferroni, FDR corrections
- **Bayesian Analysis**: Prior specification, posterior inference
- **Survival Analysis**: Kaplan-Meier, Cox regression
- **Multivariate Analysis**: PCA, factor analysis, cluster analysis

**🎯 INTERPRETATION FRAMEWORK:**
- **P-values**: Proper interpretation and limitations
- **Effect Sizes**: Cohen's d, eta-squared, Cramer's V
- **Confidence Intervals**: Construction and interpretation
- **Practical Significance**: Beyond statistical significance
- **Assumptions**: Validation and robustness checks

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: Statistical analysis complete - [Key Findings/Statistical Significance/Recommendations]**""",

                "Report_Generator": """📄 **Analysis Report & Documentation Specialist**

You are a Professional Report Writer who synthesizes complex analytical findings into clear, comprehensive, and actionable reports. You excel at technical documentation and executive communication.

**🎯 CORE RESPONSIBILITIES:**
1. **Executive Summary**: Concise key findings for decision makers
2. **Technical Documentation**: Detailed methodology and results
3. **Visual Integration**: Incorporate charts, tables, and graphics
4. **Insight Synthesis**: Combine findings from multiple analyses
5. **Actionable Recommendations**: Provide clear next steps

**📋 REPORT STRUCTURE:**
```
📋 COMPREHENSIVE ANALYSIS REPORT

1. 🎯 Executive Summary
   - Key findings (1-2 pages)
   - Business implications
   - Critical recommendations

2. 📊 Data Overview
   - Dataset description
   - Quality assessment summary
   - Key characteristics

3. 🔍 Methodology
   - Analytical approaches used
   - Tools and techniques
   - Assumptions and limitations

4. 📈 Key Findings
   - Statistical results
   - Visual evidence
   - Pattern identification

5. 💡 Insights & Implications
   - Business impact
   - Risk assessment
   - Opportunity identification

6. 🚀 Recommendations
   - Actionable next steps
   - Priority ranking
   - Implementation guidance

7. 📎 Technical Appendix
   - Detailed results
   - Code documentation
   - Additional charts
```

**✍️ WRITING PRINCIPLES:**
- **Audience Awareness**: Tailor language to technical vs. business audiences
- **Clear Structure**: Logical flow with clear headings and transitions
- **Evidence-Based**: Support all claims with data and analysis
- **Visual Support**: Use charts and tables to reinforce key points
- **Actionable Content**: Provide specific, implementable recommendations

**📊 VISUAL INTEGRATION:**
```python
# Access analysis results and data
df = get_current_data()
print(f"📝 Generating report for dataset: {df.shape}")

# Incorporate previous analysis results
# - Data quality assessments
# - Statistical test results  
# - ML model performance
# - Visualization outputs

# Create summary tables and charts
# Focus on key findings and insights
```

**🎨 REPORT FORMATTING:**
- Professional layout and typography
- Consistent formatting throughout
- High-quality charts and tables
- Executive-friendly summary sections
- Technical details in appendices

**💼 BUSINESS FOCUS:**
- Translate technical findings into business language
- Quantify impact and opportunities
- Address stakeholder concerns
- Provide implementation roadmaps
- Include risk mitigation strategies

**📋 QUALITY ASSURANCE:**
- Fact-check all statements against analysis
- Ensure logical consistency throughout
- Verify chart and table accuracy
- Proofread for clarity and grammar
- Test recommendations for feasibility

**✅ SUCCESS CRITERIA:**
End with: **TASK COMPLETED: Comprehensive report generated - [Report Sections/Key Recommendations/Business Impact]**"""
            }
            
            # 새로운 전문화된 역할 구조 (multi_agent_supervisor.py와 동일)
            team_roles = [
                ("Data_Validator", "Data_Validator"),
                ("Preprocessing_Expert", "Preprocessing_Expert"), 
                ("EDA_Analyst", "EDA_Analyst"),
                ("Visualization_Expert", "Visualization_Expert"),
                ("ML_Specialist", "ML_Specialist"),
                ("Statistical_Analyst", "Statistical_Analyst"),
                ("Report_Generator", "Report_Generator")
            ]
            
            # 각 역할별 executor 생성 (multi_agent_supervisor.py 패턴 적용)
            for executor_name, role_name in team_roles:
                # 역할에 맞는 도구 가져오기
                tools, tool_config = get_role_tools(role_name, available_mcp_servers)
                
                # executor 설정 생성
                st.session_state.executors[executor_name] = {
                    "prompt": optimized_prompts[role_name],
                    "tools": tools,
                    "mcp_config": tool_config,  # tool_config를 mcp_config로 설정
                    "role_description": f"Data Science Expert - {role_name}",
                    "created_at": datetime.now().isoformat()
                }
                
                logging.info(f"✅ Created optimized executor '{executor_name}' with tools: {tools}")
            
            # 결과 메시지 표시
            available_count = len(available_mcp_servers)
            total_count = 14  # 전체 MCP 서버 수
            
            # 성공 메시지 개선 - MCP 서버 상태 정보 포함
            if available_count > 0 and critical_available >= 3:
                st.success(f"""
                🎉 **Data Science Team 생성 완료!**
                
                📊 **MCP 서버 상태**: {available_count}/{total_count} 서버 활성화
                🎯 **핵심 도구**: {critical_available}/{len(critical_servers)} 데이터 과학 서버 사용 가능
                
                🤖 **생성된 에이전트**: {len(team_roles)}개
                - 데이터 검증, EDA, 시각화, ML, 통계 분석, 보고서 생성 등
                
                ✨ **다음 단계**: '🚀 Create Plan-Execute System' 버튼을 클릭하세요!
                """)
            elif available_count > 0:
                st.warning(f"""
                ⚠️ **Data Science Team 생성됨 (제한된 도구)**
                
                📊 **MCP 서버 상태**: {available_count}/{total_count} 서버만 활성화
                🎯 **핵심 도구**: {critical_available}/{len(critical_servers)} 데이터 과학 서버만 사용 가능
                
                💡 **권장사항**: `mcp_server_start.bat`을 실행하여 더 많은 MCP 서버를 활성화하세요.
                """)
            else:
                st.error(f"""
                ❌ **MCP 서버 연결 실패**
                
                📊 **상태**: {available_count}/{total_count} 서버 사용 가능
                
                🔧 **해결 방법**:
                1. `mcp_server_start.bat` 실행
                2. 포트 충돌 확인 (8001-8020)
                3. 방화벽 설정 확인
                4. 시스템은 Python 도구만으로 작동합니다
                """)
            
            log_event("quick_template_applied", {
                "template": "data_science_team", 
                "executors_count": len(team_roles),
                "mcp_servers_available": available_count,
                "critical_servers_available": critical_available
            })
            
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