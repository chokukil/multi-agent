# File: core/tools/python_tool.py
# Location: ./core/tools/python_tool.py

import sys
import io
import traceback
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonAstREPLTool

# Matplotlib 설정
matplotlib.use('Agg')  # GUI 백엔드 비활성화
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 기본 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 한글 폰트 설정 시도
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS
    except:
        pass  # 기본 폰트 사용

def create_enhanced_python_tool() -> Tool:
    """
    강화된 Python REPL 도구 생성 (SSOT 통합)
    데이터 접근, 시각화, 에러 처리가 개선된 버전
    """
    
    # 기본 Python REPL 도구
    base_tool = PythonAstREPLTool()
    
    # SSOT 데이터 접근 함수들 임포트
    from ..data_manager import (
        get_current_df, 
        check_data_status, 
        show_data_info,
        load_data
    )
    
    # 전역 네임스페이스에 필요한 것들 추가
    base_tool.globals.update({
        # 데이터 분석 라이브러리
        'pd': pd,
        'np': np,
        'plt': plt,
        
        # SSOT 데이터 접근 함수
        'get_current_data': get_current_df,
        'check_data_status': check_data_status,
        'show_data_info': show_data_info,
        'load_data': load_data,
        
        # 추가 유틸리티
        'matplotlib': matplotlib,
    })
    
    # 스트림릿 통합을 위한 plt.show() 오버라이드
    original_show = plt.show
    
    def custom_show(*args, **kwargs):
        """plt.show()를 streamlit과 통합"""
        try:
            import streamlit as st
            # 현재 figure를 streamlit에 표시
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)  # 메모리 관리
        except:
            # streamlit이 없으면 원래 show 사용
            original_show(*args, **kwargs)
    
    # plt.show를 커스텀 버전으로 교체
    plt.show = custom_show
    base_tool.globals['plt'].show = custom_show
    
    def enhanced_run(code: str) -> str:
        """
        강화된 코드 실행 함수
        - 더 나은 에러 처리
        - 자동 데이터 검증
        - 시각화 통합
        """
        # 출력 캡처를 위한 StringIO
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # 데이터 상태 자동 체크 (코드에 data 관련 내용이 있으면)
        data_keywords = ['df', 'data', 'get_current_data', 'load_data']
        if any(keyword in code for keyword in data_keywords):
            # 데이터 상태 확인
            from ..data_manager import data_manager
            if not data_manager.is_data_loaded():
                return """❌ No data is currently loaded!

Please ensure data is uploaded before running analysis code.

Available commands:
- check_data_status(): Check current data status
- show_data_info(): Show detailed data information
- load_data('path/to/file.csv'): Load data from file

Example:
```python
# First, check data status
status = check_data_status()
print(status)

# If data is loaded, get it
df = get_current_data()
print(df.head())
```"""
        
        try:
            # 표준 출력 리다이렉션
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(error_buffer):
                    # 코드 실행
                    result = base_tool.run(code)
            
            # 출력 수집
            stdout_output = output_buffer.getvalue()
            stderr_output = error_buffer.getvalue()
            
            # 결과 조합
            final_output = []
            
            if stdout_output:
                final_output.append("📤 Output:")
                final_output.append(stdout_output)
            
            if stderr_output and "UserWarning" not in stderr_output:
                final_output.append("\n⚠️ Warnings:")
                final_output.append(stderr_output)
            
            if result is not None and str(result).strip() != stdout_output.strip():
                final_output.append("\n📊 Result:")
                if isinstance(result, pd.DataFrame):
                    final_output.append(result.to_string())
                else:
                    final_output.append(str(result))
            
            # 성공 메시지
            if not final_output:
                final_output.append("✅ Code executed successfully (no output)")
            
            return "\n".join(final_output)
            
        except Exception as e:
            # 에러 처리
            error_type = type(e).__name__
            error_msg = str(e)
            
            # 사용자 친화적인 에러 메시지
            error_output = [f"❌ {error_type}: {error_msg}"]
            
            # 일반적인 에러에 대한 도움말
            if "NameError" in error_type and "df" in error_msg:
                error_output.append("\n💡 Hint: Use `df = get_current_data()` to load the current dataset")
            elif "KeyError" in error_type:
                error_output.append("\n💡 Hint: Check column names with `df.columns`")
            elif "AttributeError" in error_type:
                error_output.append("\n💡 Hint: Check available methods with `dir(object)`")
            
            # 전체 트레이스백 (디버그 모드에서만)
            import os
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                error_output.append("\n📍 Full Traceback:")
                error_output.append(traceback.format_exc())
            
            return "\n".join(error_output)
    
    # 강화된 도구 생성
    enhanced_tool = Tool(
        name="python_repl_ast",
        description="""Enhanced Python shell for data analysis with SSOT integration.
        
Key features:
- Automatic data access via get_current_data()
- Integrated visualization with matplotlib
- Built-in error handling and hints
- Full pandas, numpy, and scipy support

Always start with:
```python
df = get_current_data()
```""",
        func=enhanced_run
    )
    
    return enhanced_tool
