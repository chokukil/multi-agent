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
from typing import Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 신규 A2A 클라이언트 및 유틸리티 임포트
from core.a2a.a2a_streamlit_client import A2AStreamlitClient
from core.utils.logging import setup_logging
from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults # 기존 클래스 활용 가능


# AI_DS_Team 유틸리티 임포트
try:
    sys.path.insert(0, os.path.join(project_root, "ai_ds_team"))
    from ai_data_science_team.tools.dataframe import get_dataframe_summary
    AI_DS_TEAM_UTILS_AVAILABLE = True
except ImportError as e:
    st.warning(f"AI_DS_Team 유틸리티 일부 기능 제한: {e}")
    AI_DS_TEAM_UTILS_AVAILABLE = False
    def get_dataframe_summary(df): return [f"Shape: {df.shape}"]

# --- 초기 설정 ---
setup_logging()

def setup_environment():
    """환경 설정"""
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    nest_asyncio.apply()
    load_dotenv()

def apply_custom_styling():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .agent-card { background: rgba(255, 255, 255, 0.1); color: white; padding: 1.5rem; border-radius: 12px; margin: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease; }
        .agent-card:hover { transform: translateY(-5px); background: rgba(255, 255, 255, 0.2); }
        .stButton > button { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-radius: 8px; padding: 0.7rem 1.5rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- AI_DS_Team 에이전트 정보 ---
AI_DS_TEAM_AGENTS = {
    "Orchestrator": {"port": 8100, "description": "AI DS Team을 지휘하는 마에스트로", "capabilities": ["planning", "delegation"], "color": "#FAD02E"},
    "🧹 Data Cleaning": {"port": 8306, "description": "누락값 처리, 이상치 제거", "capabilities": ["missing_value", "outlier"], "color": "#FF6B6B"},
    "📊 Data Visualization": {"port": 8308, "description": "고급 시각화 생성", "capabilities": ["charts", "plots"], "color": "#4ECDC4"},
    "🔍 EDA Tools": {"port": 8312, "description": "자동 EDA 및 상관관계 분석", "capabilities": ["eda", "correlation"], "color": "#45B7D1"},
    "📁 Data Loader": {"port": 8307, "description": "다양한 데이터 소스 로딩", "capabilities": ["load_file", "connect_db"], "color": "#96CEB4"},
    "🔧 Data Wrangling": {"port": 8309, "description": "데이터 변환 및 조작", "capabilities": ["transform", "aggregate"], "color": "#FFEAA7"},
    "⚙️ Feature Engineering": {"port": 8310, "description": "고급 피처 생성 및 선택", "capabilities": ["feature_creation", "selection"], "color": "#DDA0DD"},
    "🗄️ SQL Database": {"port": 8311, "description": "SQL 데이터베이스 분석", "capabilities": ["sql_query", "db_analysis"], "color": "#F39C12"},
    "🤖 H2O ML": {"port": 8313, "description": "H2O AutoML 기반 머신러닝", "capabilities": ["automl", "model_training"], "color": "#9B59B6"},
    "📈 MLflow Tools": {"port": 8314, "description": "MLflow 실험 관리", "capabilities": ["experiment_tracking", "model_registry"], "color": "#E74C3C"}
}

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_id" not in st.session_state: st.session_state.session_id = f"ui_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if "uploaded_data" not in st.session_state: st.session_state.uploaded_data = None
    if "data_id" not in st.session_state: st.session_state.data_id = None
    if "a2a_client" not in st.session_state: st.session_state.a2a_client = A2AStreamlitClient(AI_DS_TEAM_AGENTS)
    if "agent_status" not in st.session_state: st.session_state.agent_status = {}
    if "active_agent" not in st.session_state: st.session_state.active_agent = None

async def check_agents_status_async():
    """AI_DS_Team 에이전트 상태 비동기 확인 (수정된 버전)"""
    async with httpx.AsyncClient(timeout=2.0) as client:
        # 각 에이전트에 대한 비동기 GET 요청 리스트 생성
        tasks = [client.get(f"http://localhost:{info['port']}/.well-known/agent.json") for info in AI_DS_TEAM_AGENTS.values()]
        
        # 모든 요청을 병렬로 실행하고 결과 수집
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        # 원래 에이전트 정보와 응답 결과를 순서대로 매칭
        for (name, info), resp in zip(AI_DS_TEAM_AGENTS.items(), responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                results[name] = {"status": "✅", "description": info['description']}
            else:
                results[name] = {"status": "❌", "description": info['description']}
        return results

def display_agent_status():
    st.markdown("### 🧬 AI_DS_Team 에이전트 상태")
    cols = st.columns(3)
    status = st.session_state.agent_status
    sorted_agents = sorted(status.items(), key=lambda x: (x[1]['status'] == "❌", x[0]))
    for idx, (name, info) in enumerate(sorted_agents):
        border_style = "2px solid #f093fb" if st.session_state.active_agent == name else "1px solid rgba(255, 255, 255, 0.2)"
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="agent-card" style="border: {border_style};">
                <h4>{info['status']} {name}</h4>
                <p><small>{info['description']}</small></p>
            </div>""", unsafe_allow_html=True)

def render_artifact(artifact_data: Dict[str, Any]):
    st.markdown("---"); st.success("🎉 결과가 도착했습니다!")
    content_type = artifact_data.get("contentType", "text/plain")
    data = artifact_data.get("data")
    if not data: st.warning("결과 데이터가 비어있습니다."); return
    try:
        if content_type == "application/vnd.plotly.v1+json": st.plotly_chart(pio.from_json(json.dumps(data)), use_container_width=True)
        elif content_type == "application/vnd.dataresource+json": st.dataframe(pd.DataFrame(**data))
        elif content_type.startswith("image/"): st.image(data)
        elif content_type == "text/html": st.components.v1.html(data, height=400, scrolling=True)
        elif content_type == "text/markdown": st.markdown(data)
        else: st.text(str(data))
    except Exception as e:
        st.error(f"결과 렌더링 오류: {e}"); st.text(f"Raw data: {str(data)}")

async def process_query_streaming(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant", avatar="🧬"):
        plan_container = st.container(); plan_container.info("🧠 오케스트레이터와 협의하여 실행 계획을 수립 중...")
        try:
            plan = await st.session_state.a2a_client.get_plan(prompt)
            if not plan or not plan.get("steps"):
                st.error("계획 수립 실패. Orchestrator 상태를 확인하세요."); return
        except Exception as e: st.error(f"계획 수립 중 오류: {e}"); return

        plan_container.empty(); placeholders = {}
        with plan_container:
            st.markdown("### 📝 실행 계획"); 
            for i, step in enumerate(plan["steps"]):
                id, agent, desc = step.get("id", f"s_{i}"), step.get("agent"), step.get("description")
                exp = st.expander(f"**단계 {i+1}: {agent}** - {desc} [⏳ 대기중]", False)
                placeholders[id] = {"exp": exp, "log": exp.empty(), "res": exp.empty()}
        
        final_summary = []
        for i, step in enumerate(plan["steps"]):
            id, agent, desc, task_prompt = step.get("id", f"s_{i}"), step.get("agent"), step.get("description"), step.get("prompt")
            ph = placeholders[id]
            ph["exp"].expanded = True
            st.session_state.active_agent = agent; st.rerun()

            log_container = ph["log"].container(); log_container.write("---")
            ph["exp"]._label = f"**단계 {i+1}: {agent}** - {desc} [⚙️ 작업중]"
            try:
                async for event in st.session_state.a2a_client.stream_task(agent, task_prompt, st.session_state.data_id):
                    if event["type"] == "message": log_container.info(event["content"]["text"])
                    elif event["type"] == "artifact":
                        with ph["res"].container(): render_artifact(event["content"])
                        final_summary.append({"step": i+1, "agent": agent, "result": event["content"]})
                ph["exp"]._label = f"**단계 {i+1}: {agent}** - {desc} [✅ 완료]"
            except Exception as e:
                ph["exp"]._label = f"**단계 {i+1}: {agent}** - {desc} [❌ 실패]"
                ph["log"].error(f"오류: {e}"); break
        st.session_state.active_agent = None

def main():
    setup_environment(); apply_custom_styling(); initialize_session_state()
    st.title("🧬 AI_DS_Team Orchestrator")
    st.markdown("> A2A 프로토콜 기반, 9개 전문 데이터 과학 에이전트 팀의 실시간 협업 시스템")

    if st.button("🔄 에이전트 상태 새로고침") or not st.session_state.agent_status:
        st.session_state.agent_status = asyncio.run(check_agents_status_async())
    display_agent_status()

    with st.container(border=True):
        st.subheader("📂 데이터 소스")
        handle_data_upload_with_ai_ds_team() # 기존 함수 재사용
        if st.session_state.uploaded_data is not None:
            display_data_summary_ai_ds_team(st.session_state.uploaded_data) # 기존 함수 재사용

    st.subheader("💬 AI DS Team과 대화하기")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("데이터에 대해 질문하세요..."):
        asyncio.run(process_query_streaming(prompt))

# handle_data_upload_with_ai_ds_team and display_data_summary_ai_ds_team need to be defined
# For brevity, we assume they exist and function as before.
def handle_data_upload_with_ai_ds_team():
    cols = st.columns([2, 1])
    with cols[0]:
        uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요.", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_data = df
                st.session_state.data_id = uploaded_file.name
                st.success(f"'{uploaded_file.name}' 업로드 성공!")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류 발생: {e}")

def display_data_summary_ai_ds_team(data):
    if data is not None and AI_DS_TEAM_UTILS_AVAILABLE:
        st.markdown("---")
        st.write("데이터 미리보기:")
        st.dataframe(data.head())
        summaries = get_dataframe_summary(data)
        for summary in summaries:
            st.text(summary)

if __name__ == "__main__":
    main()