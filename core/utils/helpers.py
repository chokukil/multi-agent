# File: utils/helpers.py
# Location: ./utils/helpers.py

import json
import logging
from datetime import datetime
from pathlib import Path

def log_event(event_type: str, content: dict):
    """로그 이벤트 기록"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "type": event_type,
        "content": content
    }
    
    # Session state에 추가 (Streamlit이 사용 가능한 경우에만)
    try:
        import streamlit as st
        # ScriptRunContext 존재 여부 확인
        if hasattr(st, 'session_state') and st.runtime.scriptrunner.get_script_run_ctx() is not None:
            if 'logs' not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(log_entry)
    except (ImportError, AttributeError, Exception):
        # Streamlit이 사용 불가능하거나 컨텍스트가 없는 경우 무시
        pass
    
    # 파일에도 저장
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"plan_execute_{datetime.now():%Y%m%d}.log"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to write log to file: {e}")

def safe_get_session_state(key: str, default=None):
    """안전한 session_state 접근"""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and st.runtime.scriptrunner.get_script_run_ctx() is not None:
            return getattr(st.session_state, key, default)
    except (ImportError, AttributeError, Exception):
        pass
    return default

def safe_set_session_state(key: str, value):
    """안전한 session_state 설정"""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and st.runtime.scriptrunner.get_script_run_ctx() is not None:
            setattr(st.session_state, key, value)
            return True
    except (ImportError, AttributeError, Exception):
        pass
    return False

def save_code(code: str, executor_name: str):
    """생성된 코드 저장"""
    code_entry = {
        "timestamp": datetime.now().isoformat(),
        "executor": executor_name,
        "code": code
    }
    
    # Session state에 추가
    if hasattr(st, 'session_state') and 'generated_code' in st.session_state:
        st.session_state.generated_code.append(code_entry)
    
    # 파일에도 저장
    try:
        code_dir = Path("generated_code")
        code_dir.mkdir(exist_ok=True)
        code_file = code_dir / f"{executor_name}_{datetime.now():%Y%m%d_%H%M%S}.py"
        
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        
        return code_file
    except Exception as e:
        logging.error(f"Failed to save code to file: {e}")
        return None