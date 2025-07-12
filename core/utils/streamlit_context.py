"""
Streamlit Context Utility Functions
Streamlit 컨텍스트 체크 및 안전한 함수 호출을 위한 유틸리티
"""

import logging
import contextlib
from typing import Any, Optional, Dict, Union
from functools import wraps

logger = logging.getLogger(__name__)

def has_streamlit_context() -> bool:
    """
    Streamlit 컨텍스트가 있는지 확인
    
    Returns:
        bool: Streamlit 컨텍스트가 있으면 True, 없으면 False
    """
    try:
        import streamlit as st
        # ScriptRunContext 존재 여부 확인
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        return ctx is not None
    except (ImportError, AttributeError, Exception):
        return False

def safe_streamlit_call(func_name: str, *args, **kwargs) -> Any:
    """
    안전한 Streamlit 함수 호출
    
    Args:
        func_name: 호출할 Streamlit 함수명
        *args: 함수 인자
        **kwargs: 함수 키워드 인자
        
    Returns:
        함수 결과 또는 None
    """
    if not has_streamlit_context():
        logger.debug(f"Streamlit context not available, skipping {func_name}")
        return None
    
    try:
        import streamlit as st
        func = getattr(st, func_name)
        return func(*args, **kwargs)
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"Failed to call st.{func_name}: {e}")
        return None

def safe_rerun():
    """안전한 st.rerun() 호출"""
    return safe_streamlit_call('rerun')

def safe_error(message: str):
    """안전한 st.error() 호출"""
    return safe_streamlit_call('error', message)

def safe_warning(message: str):
    """안전한 st.warning() 호출"""
    return safe_streamlit_call('warning', message)

def safe_success(message: str):
    """안전한 st.success() 호출"""
    return safe_streamlit_call('success', message)

def safe_info(message: str):
    """안전한 st.info() 호출"""
    return safe_streamlit_call('info', message)

def safe_pyplot(fig=None, clear_figure=True, **kwargs):
    """안전한 st.pyplot() 호출"""
    if not has_streamlit_context():
        logger.debug("Streamlit context not available, skipping pyplot")
        return None
    
    try:
        import streamlit as st
        return st.pyplot(fig, clear_figure=clear_figure, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to call st.pyplot: {e}")
        return None

def safe_plotly_chart(fig, use_container_width=True, **kwargs):
    """안전한 st.plotly_chart() 호출"""
    if not has_streamlit_context():
        logger.debug("Streamlit context not available, skipping plotly_chart")
        return None
    
    try:
        import streamlit as st
        return st.plotly_chart(fig, use_container_width=use_container_width, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to call st.plotly_chart: {e}")
        return None

def safe_dataframe(data, **kwargs):
    """안전한 st.dataframe() 호출"""
    return safe_streamlit_call('dataframe', data, **kwargs)

def safe_json(data, **kwargs):
    """안전한 st.json() 호출"""
    return safe_streamlit_call('json', data, **kwargs)

def safe_markdown(text: str, **kwargs):
    """안전한 st.markdown() 호출"""
    return safe_streamlit_call('markdown', text, **kwargs)

def safe_write(*args, **kwargs):
    """안전한 st.write() 호출"""
    return safe_streamlit_call('write', *args, **kwargs)

def with_streamlit_context(func):
    """
    Streamlit 컨텍스트가 있을 때만 함수를 실행하는 데코레이터
    
    Usage:
        @with_streamlit_context
        def my_function():
            st.write("This only runs with Streamlit context")
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if has_streamlit_context():
            return func(*args, **kwargs)
        else:
            logger.debug(f"Streamlit context not available, skipping {func.__name__}")
            return None
    return wrapper

@contextlib.contextmanager
def suppress_streamlit_warnings():
    """
    Streamlit 경고를 임시로 억제하는 컨텍스트 매니저
    
    Usage:
        with suppress_streamlit_warnings():
            # Some code that might cause Streamlit warnings
            pass
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
        yield

def get_session_state_safely(key: str, default=None):
    """
    안전한 세션 상태 접근
    
    Args:
        key: 세션 상태 키
        default: 기본값
        
    Returns:
        세션 상태 값 또는 기본값
    """
    if not has_streamlit_context():
        return default
    
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception as e:
        logger.warning(f"Failed to access session state {key}: {e}")
        return default

def set_session_state_safely(key: str, value: Any) -> bool:
    """
    안전한 세션 상태 설정
    
    Args:
        key: 세션 상태 키
        value: 설정할 값
        
    Returns:
        성공 여부
    """
    if not has_streamlit_context():
        return False
    
    try:
        import streamlit as st
        st.session_state[key] = value
        return True
    except Exception as e:
        logger.warning(f"Failed to set session state {key}: {e}")
        return False

# 로그 레벨을 줄여 경고 메시지 최소화
logging.getLogger("streamlit").setLevel(logging.ERROR) 