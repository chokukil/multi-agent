"""
🧠 Smart Data Analyst (A2A Version)
A2A 프로토콜 기반 데이터 분석 시스템
"""

import streamlit as st
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 페이지 설정
st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 페이지 렌더링
def main():
    try:
        # 필요한 모듈 동적 import
        from ui.data_analysis_ui import DataAnalysisUI
        
        # DataAnalysisUI 인스턴스 생성 및 렌더링
        data_analysis_ui = DataAnalysisUI()
        data_analysis_ui.render_analysis_interface()
        
    except ImportError as e:
        st.error(f"""
        **모듈 임포트 오류가 발생했습니다:**
        
        ```
        {str(e)}
        ```
        
        **해결 방법:**
        1. 프로젝트 루트 디렉토리에서 실행하고 있는지 확인하세요
        2. 필요한 의존성이 설치되어 있는지 확인하세요
        3. 다음 명령으로 시스템을 시작해보세요:
           ```bash
           streamlit run app.py
           ```
        """)
        
    except Exception as e:
        st.error(f"""
        **예상하지 못한 오류가 발생했습니다:**
        
        ```
        {str(e)}
        ```
        
        오류 타입: {type(e).__name__}
        """)
        
        # 디버깅을 위한 추가 정보
        if st.checkbox("🔍 디버깅 정보 표시"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 