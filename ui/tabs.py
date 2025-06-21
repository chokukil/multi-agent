# File: ui/tabs.py
# Location: ./ui/tabs.py

import streamlit as st
import json
from datetime import datetime
from core import data_manager, data_lineage_tracker, debug_manager

def render_bottom_tabs():
    """하단 탭 렌더링"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 System Logs", 
        "💻 Generated Code", 
        "📊 Session Statistics", 
        "🔒 SSOT Status", 
        "🐛 Debug Console"
    ])
    
    with tab1:
        render_system_logs_tab()
    
    with tab2:
        render_generated_code_tab()
    
    with tab3:
        render_session_statistics_tab()
    
    with tab4:
        render_ssot_status_tab()
    
    with tab5:
        render_debug_console_tab()

def render_system_logs_tab():
    """시스템 로그 탭"""
    st.markdown("### 📋 System Event Logs")
    
    logs = st.session_state.get("logs", [])
    
    if logs:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_types = list(set(log.get("type", "") for log in logs))
            selected_types = st.multiselect(
                "Filter by Event Type", 
                log_types, 
                default=log_types
            )
        
        with col2:
            show_count = st.selectbox(
                "Show Last N Logs",
                [10, 20, 50, 100, "All"],
                index=0
            )
        
        with col3:
            if st.button("🗑️ Clear Logs"):
                st.session_state.logs = []
                st.success("✅ Logs cleared!")
                st.rerun()
        
        # Display logs
        filtered_logs = [log for log in logs if log.get("type", "") in selected_types]
        
        if show_count != "All":
            filtered_logs = filtered_logs[-show_count:]
        
        for log in reversed(filtered_logs):
            with st.expander(f"🕐 {log.get('timestamp', 'N/A')} - {log.get('type', 'Unknown')}", expanded=False):
                st.json(log.get('content', {}))
    else:
        st.info("🔍 No system logs yet.")

def render_generated_code_tab():
    """생성된 코드 탭"""
    st.markdown("### 💻 Generated Code History")
    
    generated_code = st.session_state.get("generated_code", [])
    
    if generated_code:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            executors = list(set(code.get("executor", "") for code in generated_code))
            selected_executors = st.multiselect(
                "Filter by Executor",
                executors,
                default=executors
            )
        
        with col2:
            show_count = st.selectbox(
                "Show Last N Code Blocks",
                [5, 10, 20, "All"],
                index=1,
                key="code_count"
            )
        
        with col3:
            if st.button("🗑️ Clear Code History"):
                st.session_state.generated_code = []
                st.success("✅ Code history cleared!")
                st.rerun()
        
        # Display code
        filtered_code = [code for code in generated_code if code.get("executor", "") in selected_executors]
        
        if show_count != "All":
            filtered_code = filtered_code[-show_count:]
        
        for code_entry in reversed(filtered_code):
            with st.expander(f"🤖 {code_entry.get('executor', 'Unknown')} - {code_entry.get('timestamp', 'N/A')}", expanded=False):
                st.code(code_entry.get('code', ''), language='python')
                
                # Download button
                st.download_button(
                    "💾 Download",
                    data=code_entry.get('code', ''),
                    file_name=f"{code_entry.get('executor', 'code')}_{code_entry.get('timestamp', 'unknown')[:19].replace(':', '-')}.py",
                    mime="text/python"
                )
    else:
        st.info("💻 No code generated yet.")

def render_session_statistics_tab():
    """세션 통계 탭"""
    st.markdown("### 📊 Session Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Executors", len(st.session_state.get("executors", {})))
    
    with col2:
        st.metric("Chat Messages", len(st.session_state.get("history", [])))
    
    with col3:
        st.metric("System Logs", len(st.session_state.get("logs", [])))
    
    with col4:
        st.metric("Code Blocks", len(st.session_state.get("generated_code", [])))
    
    # Session information
    st.markdown("### 🔧 Session Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Configuration:**")
        executors = st.session_state.get("executors", {})
        if executors:
            st.success(f"🤖 Executors: {', '.join(list(executors.keys())[:3])}{'...' if len(executors) > 3 else ''}")
        else:
            st.warning("⚠️ No executors configured")
    
    with col2:
        st.markdown("**System Status:**")
        if st.session_state.get("graph_initialized"):
            st.success("✅ System Initialized")
        else:
            st.warning("⚠️ System Not Initialized")
        
        if data_manager.is_data_loaded():
            info = data_manager.get_data_info()
            st.success(f"✅ Data: {info['row_count']:,} × {info['col_count']:,}")
        else:
            st.info("📂 No data loaded")

def render_ssot_status_tab():
    """SSOT 상태 탭"""
    st.markdown("### 🔒 SSOT (Single Source of Truth) Status")
    
    # SSOT 상태 개요
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_manager.is_data_loaded():
            st.success("🟢 **SSOT Active**")
        else:
            st.error("🔴 **SSOT Empty**")
    
    with col2:
        if data_manager.is_data_loaded():
            is_valid, message = data_manager.validate_data_consistency()
            if is_valid:
                st.success("✅ **Consistent**")
            else:
                st.error("❌ **Inconsistent**")
        else:
            st.info("➖ **No Data**")
    
    with col3:
        if data_manager.is_data_loaded():
            info = data_manager.get_data_info()
            last_update = info.get('last_update', 'Unknown')
            st.info(f"🕐 **Updated**: {last_update[:19]}")
        else:
            st.info("🕐 **Never Updated**")
    
    # Data Lineage
    if data_manager.is_data_loaded():
        st.markdown("#### 📊 Data Lineage")
        
        lineage_summary = data_lineage_tracker.get_lineage_summary()
        
        if lineage_summary and "error" not in lineage_summary:
            # Original data
            st.info(f"**Original Data**: Shape {lineage_summary['original_data']['shape']}, "
                   f"{lineage_summary['original_data']['columns']} columns")
            
            # Transformations
            if lineage_summary['total_transformations'] > 0:
                st.warning(f"**Transformations**: {lineage_summary['total_transformations']} operations by "
                          f"{', '.join(lineage_summary['executors_involved'])}")
            
            # Final data
            st.success(f"**Current Data**: Shape {lineage_summary['final_data']['shape']}, "
                      f"{lineage_summary['final_data']['columns']} columns")
            
            # Suspicious patterns
            if lineage_summary['suspicious_patterns']:
                st.error("⚠️ **Suspicious Patterns Detected:**")
                for pattern in lineage_summary['suspicious_patterns']:
                    st.write(f"- {pattern['description']}")
        else:
            st.info("No data transformations tracked yet")
        
        # Validation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Validate Consistency", use_container_width=True):
                is_valid, message = data_manager.validate_data_consistency()
                if is_valid:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            if st.button("📋 Show Full Info", use_container_width=True):
                st.json(data_manager.get_data_info())
    else:
        st.info("📂 No data loaded. Upload CSV in sidebar to start.")

def render_debug_console_tab():
    """디버그 콘솔 탭"""
    st.markdown("### 🐛 Debug Console")
    
    if not hasattr(debug_manager, 'debug_mode'):
        st.error("🚨 Debug Manager not initialized")
        return
    
    # Debug mode toggle
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_debug_mode = st.checkbox(
            "🔍 Debug Mode", 
            value=debug_manager.debug_mode
        )
        if new_debug_mode != debug_manager.debug_mode:
            debug_manager.debug_mode = new_debug_mode
            st.success(f"✅ Debug mode {'enabled' if new_debug_mode else 'disabled'}")
    
    with col2:
        if st.button("🗑️ Clear Debug Logs"):
            debug_manager.debug_logs = []
            debug_manager.error_history = []
            debug_manager.performance_metrics = {}
            st.success("✅ Debug logs cleared!")
            st.rerun()
    
    with col3:
        if st.button("📊 Show Summary"):
            st.json(debug_manager.get_debug_summary())
    
    # Debug logs
    st.markdown("#### 🔍 Recent Debug Logs")
    
    if debug_manager.debug_logs:
        log_count = st.selectbox(
            "Show Last N Logs",
            [10, 20, 50, "All"],
            index=0
        )
        
        logs_to_show = debug_manager.debug_logs
        if log_count != "All":
            logs_to_show = logs_to_show[-log_count:]
        
        for log in reversed(logs_to_show):
            with st.expander(f"🐛 [{log['category']}] {log['timestamp']} - {log['message'][:50]}...", expanded=False):
                st.json(log)
    else:
        st.info("📝 No debug logs yet.")