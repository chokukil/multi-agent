"""
🧬 AI_DS_Team Orchestrator Enhanced - Advanced Data Science with A2A Protocol + Smart UI
Smart Display Manager와 A2A Orchestration UI가 통합된 차세대 데이터 사이언스 시스템

핵심 특징:
- Smart Display Manager: 타입별 자동 렌더링
- A2A Orchestration UI: 지능형 오케스트레이션 시각화
- Real-time Streaming: 누적형 스트리밍 컨테이너
- Enhanced UX: 아름다운 UI/UX 컴포넌트
"""

import streamlit as st
import sys
import os
import asyncio
import logging
import platform
from datetime import datetime
from dotenv import load_dotenv
import nest_asyncio
import pandas as pd
import json
import httpx
import time
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from typing import Dict, Any, Tuple
import traceback
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 핵심 컴포넌트 임포트
from core.a2a.a2a_streamlit_client import A2AStreamlitClient
from core.utils.logging import setup_logging
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

# 새로운 Smart UI 컴포넌트 임포트
try:
    from core.ui.smart_display import SmartDisplayManager, AccumulativeStreamContainer
    from core.ui.a2a_orchestration_ui import A2AOrchestrationDashboard
    SMART_UI_AVAILABLE = True
    print("✅ Smart UI 컴포넌트 로드 성공")
except ImportError as e:
    SMART_UI_AVAILABLE = False
    print(f"⚠️ Smart UI 컴포넌트 로드 실패: {e}")

def main():
    """향상된 메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="🧬 AI DS Team Enhanced",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 헤더
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🧬 AI DS Team Enhanced</h1>
        <p style="font-size: 1.2em; opacity: 0.8;">
            Smart Display Manager + A2A Orchestration UI 통합 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Smart UI 상태 표시
    if SMART_UI_AVAILABLE:
        st.success("✅ Smart UI 컴포넌트가 성공적으로 로드되었습니다!")
        
        # Smart Display Manager 테스트
        st.markdown("### 🎨 Smart Display Manager 테스트")
        
        smart_display = SmartDisplayManager()
        
        # 다양한 콘텐츠 타입 테스트
        test_code = '''
import pandas as pd
import plotly.express as px

# 데이터 로드
df = pd.read_csv('data.csv')

# 시각화 생성
fig = px.scatter(df, x='x', y='y', color='category')
fig.show()
        '''
        
        st.markdown("#### 코드 렌더링 테스트")
        smart_display.smart_display_content(test_code)
        
        st.markdown("#### 마크다운 렌더링 테스트")
        test_markdown = """
# 데이터 분석 결과

## 주요 발견사항
- **상관관계**: X와 Y 변수 간 강한 양의 상관관계 발견
- **이상치**: 총 5개의 이상치 탐지
- **분포**: 정규분포에 가까운 패턴

> 추가 분석이 필요한 영역: 카테고리별 세부 분석
        """
        smart_display.smart_display_content(test_markdown)
        
        st.markdown("#### JSON 데이터 렌더링 테스트")
        test_json = {
            "type": "analysis_result",
            "name": "상관관계 분석",
            "description": "변수 간 상관관계 분석 결과",
            "status": "completed",
            "result": {
                "correlation_coefficient": 0.85,
                "p_value": 0.001,
                "significance": "highly_significant"
            }
        }
        smart_display.smart_display_content(test_json)
        
    else:
        st.error("❌ Smart UI 컴포넌트를 로드할 수 없습니다.")
        st.info("기본 Streamlit 기능을 사용합니다.")

if __name__ == "__main__":
    main()
