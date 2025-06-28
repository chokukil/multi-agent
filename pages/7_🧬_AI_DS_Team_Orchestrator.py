"""
🧬 AI_DS_Team Orchestrator - Advanced Data Science with A2A Protocol
Smart Data Analyst의 우수한 패턴을 기반으로 한 AI_DS_Team 통합 시스템

핵심 특징:
- AI_DS_Team Integration: 9개 전문 에이전트 활용
- A2A Orchestration: LLM 기반 지능형 에이전트 선택
- Real-time Processing: 실시간 작업 진행 상황 모니터링  
- Professional Results: 전문적인 데이터 과학 결과 제공
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

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from core.utils.logging import setup_logging

# AI_DS_Team 유틸리티 임포트
try:
    sys.path.insert(0, os.path.join(project_root, "ai_ds_team"))
    from ai_data_science_team.tools.dataframe import get_dataframe_summary
    from ai_data_science_team.utils.html import open_html_file_in_browser
    from ai_data_science_team.utils.plotly import plotly_from_dict
    from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
    AI_DS_TEAM_UTILS_AVAILABLE = True
except ImportError as e:
    st.warning(f"AI_DS_Team 유틸리티 일부 기능 제한: {e}")
    AI_DS_TEAM_UTILS_AVAILABLE = False
    
    # 기본 함수 정의
    def get_dataframe_summary(df, n_sample=5):
        """기본 데이터프레임 요약 함수"""
        try:
            summary = f"""
**Shape**: {df.shape[0]:,} rows × {df.shape[1]:,} columns

**Columns**: {', '.join(df.columns.tolist())}

**Data Types**:
{df.dtypes.to_string()}

**Sample Data**:
{df.head(n_sample).to_string()}

**Missing Values**:
{df.isnull().sum().to_string()}
"""
            return [summary]
        except Exception as e:
            return [f"데이터 요약 생성 중 오류: {str(e)}"]

# --- 초기 설정 ---
setup_logging()

def setup_environment():
    """환경 설정"""
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    nest_asyncio.apply()
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def apply_custom_styling():
    """AI_DS_Team 전용 스타일링"""
    st.markdown("""
    <style>
        /* 메인 배경 */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* 컨테이너 스타일 */
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* 에이전트 카드 스타일 */
        .agent-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .agent-card:hover {
            transform: translateY(-5px);
        }
        
        /* 진행 상태 표시 */
        .progress-indicator {
            background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* 결과 표시 영역 */
        .result-container {
            background: rgba(255, 255, 255, 0.98);
            color: #333;
            padding: 2rem;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
        }
        
        /* 버튼 스타일 개선 */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ai_ds_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "data_id" not in st.session_state:
        st.session_state.data_id = None

# --- AI_DS_Team 에이전트 정보 ---
AI_DS_TEAM_AGENTS = {
    "🧹 Data Cleaning": {
        "port": 8306,
        "description": "누락값 처리, 이상치 제거, 데이터 품질 개선",
        "capabilities": ["missing_value_handling", "outlier_detection", "data_validation"],
        "color": "#FF6B6B"
    },
    "📊 Data Visualization": {
        "port": 8308, 
        "description": "Plotly, Matplotlib 기반 고급 시각화",
        "capabilities": ["interactive_charts", "statistical_plots", "dashboards"],
        "color": "#4ECDC4"
    },
    "🔍 EDA Tools": {
        "port": 8312,
        "description": "missingno, sweetviz, correlation funnel 활용 EDA",
        "capabilities": ["missing_data_analysis", "sweetviz_reports", "correlation_analysis"],
        "color": "#45B7D1"
    },
    "📁 Data Loader": {
        "port": 8307,
        "description": "다양한 데이터 소스 로딩 및 전처리", 
        "capabilities": ["file_loading", "database_connection", "api_integration"],
        "color": "#96CEB4"
    },
    "🔧 Data Wrangling": {
        "port": 8309,
        "description": "Pandas 기반 데이터 변환 및 조작",
        "capabilities": ["data_transformation", "aggregation", "merging"],
        "color": "#FFEAA7"
    },
    "⚙️ Feature Engineering": {
        "port": 8310,
        "description": "고급 피처 생성 및 선택",
        "capabilities": ["feature_creation", "feature_selection", "encoding"],
        "color": "#DDA0DD"
    },
    "🗄️ SQL Database": {
        "port": 8311,
        "description": "SQL 데이터베이스 쿼리 및 분석",
        "capabilities": ["sql_queries", "database_analysis", "data_extraction"],
        "color": "#F39C12"
    },
    "🤖 H2O ML": {
        "port": 8313,
        "description": "H2O AutoML 기반 머신러닝",
        "capabilities": ["automl", "model_training", "model_evaluation"],
        "color": "#9B59B6"
    },
    "📈 MLflow Tools": {
        "port": 8314,
        "description": "MLflow 기반 실험 관리 및 모델 추적",
        "capabilities": ["experiment_tracking", "model_registry", "deployment"],
        "color": "#E74C3C"
    }
}

def check_ai_ds_team_agents():
    """AI_DS_Team 에이전트들의 상태 확인"""
    status_results = {}
    
    st.markdown("### 🧬 AI_DS_Team 에이전트 상태")
    
    # 그리드 레이아웃으로 에이전트 카드 표시
    cols = st.columns(3)
    
    for idx, (name, info) in enumerate(AI_DS_TEAM_AGENTS.items()):
        col = cols[idx % 3]
        
        with col:
            try:
                url = f"http://localhost:{info['port']}"
                with httpx.Client(timeout=3.0) as client:
                    response = client.get(f"{url}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        agent_name = agent_card.get('name', name)
                        
                        # 성공 카드
                        st.markdown(f"""
                        <div class="agent-card" style="background: linear-gradient(135deg, {info['color']}88, {info['color']}CC);">
                            <h4>✅ {name}</h4>
                            <p><small>Port: {info['port']}</small></p>
                            <p>{info['description']}</p>
                            <div style="font-size: 0.8em; opacity: 0.9;">
                                <strong>기능:</strong><br>
                                {', '.join(info['capabilities'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        status_results[name] = True
                    else:
                        # 오류 카드
                        st.markdown(f"""
                        <div class="agent-card" style="background: linear-gradient(135deg, #FF6B6B88, #FF6B6BCC);">
                            <h4>❌ {name}</h4>
                            <p><small>Port: {info['port']}</small></p>
                            <p>HTTP {response.status_code}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        status_results[name] = False
            except Exception as e:
                # 연결 실패 카드
                st.markdown(f"""
                <div class="agent-card" style="background: linear-gradient(135deg, #FF6B6B88, #FF6B6BCC);">
                    <h4>❌ {name}</h4>
                    <p><small>Port: {info['port']}</small></p>
                    <p>연결 실패</p>
                    <p><small>{str(e)[:30]}...</small></p>
                </div>
                """, unsafe_allow_html=True)
                status_results[name] = False
    
    # 전체 상태 요약
    active_agents = sum(status_results.values())
    total_agents = len(AI_DS_TEAM_AGENTS)
    
    st.markdown(f"""
    <div class="progress-indicator">
        <h3>🎯 시스템 상태: {active_agents}/{total_agents} 에이전트 활성</h3>
        <p>활성 에이전트들을 통해 포괄적인 데이터 사이언스 분석이 가능합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    return status_results

def handle_data_upload_with_ai_ds_team():
    """AI_DS_Team 통합 데이터 업로드"""
    st.markdown("""
    <div class="main-container">
        <h3>📊 데이터 업로드 & AI_DS_Team 분석</h3>
        <p>AI_DS_Team의 9개 전문 에이전트가 데이터를 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV, Excel, JSON 파일을 업로드하세요",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="업로드된 데이터는 AI_DS_Team 전체 에이전트들이 활용할 수 있습니다."
    )
    
    # 샘플 데이터 옵션들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚢 타이타닉 데이터", type="secondary"):
            load_sample_data("titanic")
    
    with col2:
        if st.button("💼 고객 이탈 데이터", type="secondary"):
            load_sample_data("churn")
    
    with col3:
        if st.button("🏠 주택 가격 데이터", type="secondary"):
            load_sample_data("housing")
    
    if uploaded_file is not None:
        try:
            # 파일 타입에 따른 데이터 로드
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            
            # 데이터 저장
            save_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/{uploaded_file.name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path, index=False)
            
            st.session_state.uploaded_data = data
            st.session_state.data_id = uploaded_file.name
            
            # AI_DS_Team 스타일 데이터 요약
            display_data_summary_ai_ds_team(data, uploaded_file.name)
            
            st.success(f"✅ 데이터 업로드 완료: {uploaded_file.name}")
            st.rerun()
            
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")

def load_sample_data(dataset_name):
    """샘플 데이터 로드"""
    try:
        if dataset_name == "titanic":
            # 타이타닉 데이터 로드
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/titanic.csv"
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                # 기본 타이타닉 데이터 생성
                data = pd.DataFrame({
                    'PassengerId': range(1, 101),
                    'Survived': [0, 1] * 50,
                    'Pclass': [1, 2, 3] * 33 + [1],
                    'Sex': ['male', 'female'] * 50,
                    'Age': [25, 30, 35, 40] * 25,
                    'Fare': [50, 75, 100, 125] * 25
                })
                data.to_csv(data_path, index=False)
        
        elif dataset_name == "churn":
            # 고객 이탈 데이터 (ai_ds_team 예제에서 사용)
            churn_path = "ai_ds_team/data/churn_data.csv"
            if os.path.exists(churn_path):
                data = pd.read_csv(churn_path)
                save_path = "a2a_ds_servers/artifacts/data/shared_dataframes/churn_data.csv"
                data.to_csv(save_path, index=False)
            else:
                st.error("Churn 데이터를 찾을 수 없습니다.")
                return
        
        elif dataset_name == "housing":
            # 주택 가격 데이터 생성
            import numpy as np
            np.random.seed(42)
            n_samples = 1000
            data = pd.DataFrame({
                'sqft': np.random.normal(2000, 500, n_samples),
                'bedrooms': np.random.randint(1, 6, n_samples),
                'bathrooms': np.random.randint(1, 4, n_samples),
                'age': np.random.randint(0, 50, n_samples),
                'price': np.random.normal(300000, 100000, n_samples)
            })
            save_path = "a2a_ds_servers/artifacts/data/shared_dataframes/housing_data.csv"
            data.to_csv(save_path, index=False)
        
        st.session_state.uploaded_data = data
        st.session_state.data_id = f"{dataset_name}_data"
        
        display_data_summary_ai_ds_team(data, f"{dataset_name} 샘플 데이터")
        st.success(f"✅ {dataset_name} 샘플 데이터 로드 완료!")
        st.rerun()
        
    except Exception as e:
        st.error(f"샘플 데이터 로드 실패: {e}")

def display_data_summary_ai_ds_team(data, dataset_name):
    """AI_DS_Team 스타일 데이터 요약 표시"""
    try:
        # AI_DS_Team의 get_dataframe_summary 활용
        summary = get_dataframe_summary(data, n_sample=10, skip_stats=False)
        
        st.markdown(f"""
        <div class="result-container">
            <h3>📋 {dataset_name} 데이터 요약</h3>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>📊 기본 정보</h4>
                <ul>
                    <li><strong>행 수:</strong> {data.shape[0]:,}</li>
                    <li><strong>열 수:</strong> {data.shape[1]:,}</li>
                    <li><strong>메모리 사용량:</strong> {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 데이터 미리보기
        st.markdown("#### 📊 데이터 미리보기")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 데이터 타입 정보
        st.markdown("#### 🔍 컬럼 정보")
        col_info = pd.DataFrame({
            '컬럼명': data.columns,
            '데이터 타입': data.dtypes.astype(str),  # PyArrow 호환성을 위해 문자열로 변환
            '누락값 수': data.isnull().sum().values,
            '누락값 비율(%)': (data.isnull().sum() / len(data) * 100).round(2).values,
            '고유값 수': data.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)
        
    except Exception as e:
        st.error(f"데이터 요약 생성 실패: {e}")
        # 기본 pandas 요약으로 fallback
        st.dataframe(data.describe(), use_container_width=True)

async def process_ai_ds_team_query(prompt: str):
    """AI_DS_Team 통합 쿼리 처리"""
    try:
        # Universal AI Orchestrator를 통한 처리
        orchestrator_url = "http://localhost:8100"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{orchestrator_url}/invoke",
                json={
                    "message": {
                        "parts": [{"text": prompt}]
                    },
                    "context_id": st.session_state.session_id,
                    "task_id": f"ai_ds_team_{int(time.time())}"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", {}).get("parts", [{}])[0].get("text", "응답을 받지 못했습니다.")
            else:
                return f"오케스트레이터 오류: HTTP {response.status_code}"
                
    except Exception as e:
        return f"처리 중 오류 발생: {str(e)}"

def render_ai_ds_team_chat():
    """AI_DS_Team 채팅 인터페이스"""
    st.markdown("""
    <div class="main-container">
        <h3>💬 AI_DS_Team 채팅</h3>
        <p>9개의 전문 에이전트와 대화하며 데이터 사이언스 작업을 수행하세요.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("AI_DS_Team에게 데이터 분석을 요청하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 처리
        with st.chat_message("assistant"):
            with st.spinner("AI_DS_Team이 작업 중입니다..."):
                try:
                    response = asyncio.run(process_ai_ds_team_query(prompt))
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"오류 발생: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """메인 애플리케이션"""
    setup_environment()
    apply_custom_styling()
    initialize_session_state()
    
    # 페이지 헤더
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h1>🧬 AI_DS_Team Orchestrator</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Advanced Data Science with A2A Protocol</p>
        <p style="opacity: 0.8;">9개 전문 에이전트의 협업으로 완성되는 데이터 사이언스</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 - 에이전트 상태 및 제어
    with st.sidebar:
        st.markdown("## 🎛️ 시스템 제어")
        
        # 에이전트 상태 확인
        if st.button("🔄 에이전트 상태 새로고침"):
            st.rerun()
        
        # 서버 시작/중지 버튼들
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 시스템 시작"):
                with st.spinner("AI_DS_Team 시스템 시작 중..."):
                    # 시스템 시작 스크립트 실행
                    st.info("시스템 시작 중입니다. 잠시 후 에이전트 상태를 확인해주세요.")
        
        with col2:
            if st.button("🛑 시스템 중지"):
                st.warning("시스템 중지 기능은 터미널에서 수행해주세요.")
        
        # 세션 정보
        st.markdown("---")
        st.markdown("### 📋 세션 정보")
        st.text(f"세션 ID: {st.session_state.session_id}")
        if st.session_state.data_id:
            st.text(f"데이터: {st.session_state.data_id}")
        
        # 로그 초기화
        if st.button("🗑️ 채팅 기록 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    # 메인 콘텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 대시보드", "📊 데이터 업로드", "💬 AI 채팅", "📈 결과 분석"])
    
    with tab1:
        # 에이전트 상태 대시보드
        agent_status = check_ai_ds_team_agents()
        
        # 시스템 메트릭
        st.markdown("### 📊 시스템 메트릭")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("활성 에이전트", sum(agent_status.values()), f"/{len(agent_status)}")
        with col2:
            st.metric("처리된 작업", len(st.session_state.messages), "개")
        with col3:
            st.metric("업로드된 데이터", 1 if st.session_state.uploaded_data is not None else 0, "개")
        with col4:
            st.metric("세션 시간", f"{(time.time() - time.mktime(datetime.now().timetuple())) // 3600:.0f}", "시간")
    
    with tab2:
        handle_data_upload_with_ai_ds_team()
    
    with tab3:
        render_ai_ds_team_chat()
    
    with tab4:
        st.markdown("""
        <div class="main-container">
            <h3>📈 분석 결과 대시보드</h3>
            <p>AI_DS_Team의 분석 결과를 종합적으로 확인할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 생성된 아티팩트들 표시
        artifacts_path = "a2a_ds_servers/artifacts/"
        
        if os.path.exists(artifacts_path):
            # 플롯 파일들
            plots_path = os.path.join(artifacts_path, "plots/")
            if os.path.exists(plots_path):
                plot_files = [f for f in os.listdir(plots_path) if f.endswith(('.png', '.jpg', '.html'))]
                if plot_files:
                    st.markdown("#### 📊 생성된 차트들")
                    for plot_file in plot_files[-5:]:  # 최근 5개
                        st.text(f"📈 {plot_file}")
            
            # HTML 리포트들
            html_files = []
            for root, dirs, files in os.walk(artifacts_path):
                for file in files:
                    if file.endswith('.html'):
                        html_files.append(os.path.join(root, file))
            
            if html_files:
                st.markdown("#### 📄 생성된 리포트들")
                for html_file in html_files[-3:]:  # 최근 3개
                    st.text(f"📄 {os.path.basename(html_file)}")
        
        # 데이터 현황
        if st.session_state.uploaded_data is not None:
            st.markdown("#### 📊 현재 데이터 현황")
            display_data_summary_ai_ds_team(
                st.session_state.uploaded_data, 
                st.session_state.data_id or "업로드된 데이터"
            )

if __name__ == "__main__":
    main() 