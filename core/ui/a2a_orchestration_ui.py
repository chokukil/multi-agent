"""
🧠 A2A Orchestration UI - 지능형 오케스트레이션 시각화
A2A 오케스트레이터의 지능형 요소들을 아름답게 시각화하는 고급 UI 컴포넌트
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
import time
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class ComplexityLevel(Enum):
    SIMPLE = "simple"
    SINGLE_AGENT = "single_agent"
    COMPLEX = "complex"

@dataclass
class AgentStatus:
    name: str
    port: int
    status: str
    capabilities: List[str]
    last_activity: datetime
    response_time: float
    success_rate: float

@dataclass
class TaskStep:
    id: str
    agent_name: str
    description: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    artifacts: List[Dict]

class A2AOrchestrationDashboard:
    """A2A 오케스트레이션 실시간 대시보드"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.agents_status = {}
        self.current_task_steps = []
        self.complexity_history = []
        
    def render_complexity_analyzer(self, user_input: str, complexity_result: Dict) -> None:
        """복잡도 분석 결과를 시각화"""
        
        st.markdown("### 🧠 지능형 복잡도 분석")
        
        # 복잡도 레벨 표시
        complexity_level = complexity_result.get('level', 'unknown')
        complexity_score = complexity_result.get('score', 0)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 복잡도 게이지 차트
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = complexity_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "복잡도 점수"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, key=f"complexity_gauge_{self.session_id}", use_container_width=True)
        
        with col2:
            # 복잡도 레벨 표시
            if complexity_level == ComplexityLevel.SIMPLE.value:
                st.success("🟢 단순 질문")
                st.metric("예상 처리 시간", "< 5초")
            elif complexity_level == ComplexityLevel.SINGLE_AGENT.value:
                st.warning("🟡 단일 에이전트")
                st.metric("예상 처리 시간", "5-30초")
            else:
                st.error("🔴 복합 태스크")
                st.metric("예상 처리 시간", "30초+")
        
        with col3:
            # 매칭된 패턴들
            patterns = complexity_result.get('matched_patterns', [])
            if patterns:
                st.markdown("**매칭 패턴:**")
                for pattern in patterns[:3]:  # 최대 3개만 표시
                    st.markdown(f"• {pattern}")
        
        # 상세 분석 결과 (확장 가능)
        with st.expander("🔍 상세 분석 결과", expanded=False):
            analysis_data = {
                "입력 텍스트 길이": len(user_input),
                "키워드 수": len(user_input.split()),
                "복잡도 점수": f"{complexity_score:.3f}",
                "분류 결과": complexity_level,
                "매칭된 패턴": patterns
            }
            st.json(analysis_data)


def create_orchestration_dashboard() -> A2AOrchestrationDashboard:
    """오케스트레이션 대시보드 인스턴스 생성"""
    return A2AOrchestrationDashboard()


# 전역 대시보드 인스턴스
if 'orchestration_dashboard' not in st.session_state:
    st.session_state.orchestration_dashboard = create_orchestration_dashboard()

# 에이전트 이름 매핑 (계획에서 사용하는 이름 -> 실제 에이전트 이름)
AGENT_NAME_MAPPING = {
    "data_loader": "📁 Data Loader",
    "data_cleaning": "🧹 Data Cleaning", 
    "data_wrangling": "🔧 Data Wrangling",
    "eda_tools": "🔍 EDA Tools",
    "data_visualization": "📊 Data Visualization",
    "feature_engineering": "⚙️ Feature Engineering",
    "sql_database": "🗄️ SQL Database",
    "h2o_ml": "🤖 H2O ML",
    "mlflow_tools": "📈 MLflow Tools",
    # 오케스트레이터 계획에서 사용하는 이름들 추가
    "AI_DS_Team EDAToolsAgent": "🔍 EDA Tools",
    "AI_DS_Team DataLoaderToolsAgent": "📁 Data Loader",
    "AI_DS_Team DataCleaningAgent": "🧹 Data Cleaning",
    "AI_DS_Team DataVisualizationAgent": "📊 Data Visualization",
    "AI_DS_Team SQLDatabaseAgent": "🗄️ SQL Database",
    "AI_DS_Team DataWranglingAgent": "�� Data Wrangling"
}
