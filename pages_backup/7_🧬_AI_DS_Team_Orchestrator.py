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

# get_all_agents에서 에이전트 정보 가져오기 (신규)
from pages.get_all_agents import AI_DS_TEAM_AGENTS

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from core.utils.logging import setup_logging

# A2A 스트리밍 클라이언트 (신규 추가)
from core.a2a.a2a_streamlit_client import A2AStreamlitClient

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
    if "a2a_client" not in st.session_state:
        # 클라이언트 초기화 시 에이전트 정보 전달
        st.session_state.a2a_client = A2AStreamlitClient(AI_DS_TEAM_AGENTS)
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {}

async def check_ai_ds_team_agents_async():
    """AI_DS_Team 에이전트들의 상태를 비동기적으로 확인"""
    status_results = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        tasks = []
        for name, info in AI_DS_TEAM_AGENTS.items():
            url = f"http://localhost:{info['port']}/.well-known/agent.json"
            tasks.append(client.get(url, follow_redirects=True))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, info), response in zip(AI_DS_TEAM_AGENTS.items(), responses):
            if isinstance(response, httpx.Response) and response.status_code == 200:
                try:
                    agent_card = response.json()
                    status_results[name] = {
                        "status": "✅ 온라인",
                        "name": agent_card.get('name', name),
                        "description": agent_card.get('description', info['description']),
                        "skills": [skill.get('name', 'N/A') for skill in agent_card.get('skills', [])]
                    }
                except json.JSONDecodeError:
                    status_results[name] = {"status": "❌ 오류", "description": "Agent Card가 유효한 JSON이 아닙니다."}
            else:
                status_results[name] = {"status": "❌ 오프라인", "description": info['description']}
    return status_results

def display_agent_status(status_results):
    """에이전트 상태를 UI에 표시"""
    st.markdown("### 🧬 AI_DS_Team 에이전트 상태")
    cols = st.columns(3)
    sorted_agents = sorted(status_results.items(), key=lambda x: (x[1]['status'] == "❌ 오프라인", x[0]))

    for idx, (name, info) in enumerate(sorted_agents):
        col = cols[idx % 3]
        border_color = "#667eea" if st.session_state.get("active_agent") == name else AI_DS_TEAM_AGENTS[name]['color']
        
        with col:
            st.markdown(f"""
            <div class="agent-card" style="border: 2px solid {border_color};">
                <h4>{info['status']} {name}</h4>
                <p><small>{info['description']}</small></p>
            </div>
            """, unsafe_allow_html=True)

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
    """AI_DS_Team용 데이터 요약 표시"""
    if not AI_DS_TEAM_UTILS_AVAILABLE or data is None:
        return

    container = st.container(border=True)
    container.subheader(f"📄 '{dataset_name}' 데이터 미리보기")
    
    try:
        summaries = get_dataframe_summary(data)
        for summary in summaries:
            container.text(summary)
    except Exception as e:
        container.error(f"데이터 요약 표시 중 오류: {e}")

async def process_ai_ds_team_query(prompt: str):
    """
    AI DS Team 오케스트레이션의 새로운 스트리밍 방식 처리
    """
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🧬"):
        # 1. 오케스트레이터로부터 계획 수신
        plan_container = st.container()
        plan_container.write("🧠 오케스트레이터에게 작업을 요청하고 실행 계획을 수립하는 중...")

        try:
            plan = await st.session_state.a2a_client.get_plan(prompt)
            if not plan or "steps" not in plan or not plan["steps"]:
                 st.error("실행 계획을 수립하지 못했습니다. 오케스트레이터 에이전트(8305)가 실행 중인지 확인하세요.")
                 return
        except Exception as e:
            st.error(f"계획 수립 중 오류 발생: {e}")
            logging.error(f"Failed to get plan: {e}")
            return

        # 2. 계획을 UI에 즉시 시각화
        plan_container.empty() # "계획 중..." 메시지 제거
        with plan_container:
            st.markdown("### 📝 실행 계획")
            plan_placeholders = {}
            for i, step in enumerate(plan["steps"]):
                step_id = step.get("id", f"step_{i+1}")
                agent_name = step.get("agent", "Unknown Agent")
                description = step.get("description", "No description provided.")
                
                expander = st.expander(f"**단계 {i+1}: {agent_name}** - {description} [⏳ 대기중]", expanded=False)
                with expander:
                    ph_status = st.empty()
                    ph_logs = st.empty()
                    ph_artifact = st.empty()
                    plan_placeholders[step_id] = {
                        "expander": expander,
                        "status": ph_status, 
                        "logs": ph_logs,
                        "artifact": ph_artifact
                    }
        
        # 3. 계획의 각 단계를 순차적으로 실행
        final_results_summary = []
        for i, step in enumerate(plan["steps"]):
            step_id = step.get("id", f"step_{i+1}")
            agent_name = step.get("agent", "Unknown Agent")
            description = step.get("description", "...")
            task_prompt = step.get("prompt", "")
            
            placeholders = plan_placeholders[step_id]
            placeholders["expander"].expanded = True
            placeholders["expander"]._label = f"**단계 {i+1}: {agent_name}** - {description} [⚙️ 작업중]"

            st.session_state.active_agent = agent_name
            # Rerun to update agent card highlight
            st.rerun() 

            log_container = placeholders["logs"].container()
            log_container.markdown("---")
            log_container.write("실시간 로그:")

            try:
                # 4. A2A 클라이언트를 통해 스트리밍 작업 실행
                async for event in st.session_state.a2a_client.stream_task(agent_name, task_prompt, st.session_state.data_id):
                    if event["type"] == "message":
                        log_container.info(event["content"]["text"])
                    elif event["type"] == "artifact":
                        with placeholders["artifact"].container():
                            render_artifact(event["content"])
                            final_results_summary.append({
                                "step": i + 1,
                                "agent": agent_name,
                                "result": event["content"]
                            })
                    elif event["type"] == "status":
                         placeholders["status"].info(f"상태 업데이트: {event['state']}")

                placeholders["expander"]._label = f"**단계 {i+1}: {agent_name}** - {description} [✅ 완료]"

            except Exception as e:
                placeholders["expander"]._label = f"**단계 {i+1}: {agent_name}** - {description} [❌ 실패]"
                placeholders["status"].error(f"작업 처리 중 오류 발생: {e}")
                logging.error(f"Error processing step {step_id} with {agent_name}: {e}")
                break # 오류 발생 시 중단

        st.session_state.active_agent = None
        # 최종 요약 표시
        display_final_summary(final_results_summary)

def render_artifact(artifact_data: Dict[str, Any]):
    """A2A 아티팩트를 Streamlit UI에 렌더링"""
    st.markdown("---")
    st.success("🎉 결과가 도착했습니다!")

    content_type = artifact_data.get("contentType", "text/plain")
    data = artifact_data.get("data")

    if not data:
        st.warning("결과 데이터가 비어있습니다.")
        return

    try:
        if content_type == "application/json":
            st.json(data)
        elif content_type == "application/vnd.dataresource+json": # Tabular data
             df = pd.DataFrame(**data)
             st.dataframe(df)
        elif content_type == "application/vnd.plotly.v1+json":
            fig = pio.from_json(json.dumps(data))
            st.plotly_chart(fig, use_container_width=True)
        elif content_type.startswith("image/"):
            st.image(data, caption="생성된 이미지", use_column_width=True)
        elif content_type == "text/html":
            st.components.v1.html(data, height=400, scrolling=True)
            st.download_button("HTML 다운로드", data, file_name="report.html")
        elif content_type == "text/markdown":
            st.markdown(data)
        else: # Default to text
            st.text(str(data))
    except Exception as e:
        st.error(f"결과를 렌더링하는 중 오류가 발생했습니다: {e}")
        st.text("Raw data:")
        st.text(str(data))

def display_final_summary(results: list):
    """모든 작업 완료 후 최종 요약 보고"""
    with st.container(border=True):
        st.markdown("## 🚀 최종 실행 요약")
        for result in results:
            agent = result['agent']
            st.markdown(f"### 단계 {result['step']}: {agent}")
            render_artifact(result['result'])

def render_ai_ds_team_chat():
    """AI_DS_Team 챗 인터페이스 렌더링"""
    st.subheader("💬 AI DS Team과 대화하기")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("데이터 분석에 대해 무엇이든 물어보세요..."):
        asyncio.run(process_ai_ds_team_query(prompt))

def render_performance_monitoring_tab():
    """Phase 4: 성능 모니터링 탭 렌더링"""
    st.markdown("""
    <div class="main-container">
        <h3>🔍 시스템 성능 모니터링</h3>
        <p>A2A 에이전트 시스템의 실시간 성능을 모니터링합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from core.performance_monitor import performance_monitor
        
        # 모니터링 활성화 체크
        if not performance_monitor.monitoring_active:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.warning("⚠️ 성능 모니터링이 비활성화되어 있습니다.")
            with col2:
                if st.button("🔍 모니터링 시작"):
                    performance_monitor.start_monitoring()
                    st.success("모니터링이 시작되었습니다!")
                    st.rerun()
        
        # 성능 대시보드 렌더링
        performance_monitor.render_performance_dashboard()
        
    except Exception as e:
        st.error(f"성능 모니터링 로드 실패: {e}")
        st.info("기본 성능 정보를 표시합니다...")
        
        # 기본 성능 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("시스템 상태", "🟢 정상")
        with col2:
            agent_status = check_ai_ds_team_agents_async()
            active_agents = sum(1 for status in agent_status.values() if status)
            st.metric("활성 에이전트", f"{active_agents}/{len(agent_status)}")
        with col3:
            st.metric("총 메시지", len(st.session_state.messages))
        with col4:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU 사용률", f"{cpu_percent:.1f}%")
            except:
                st.metric("CPU 사용률", "N/A")

def main():
    """메인 실행 함수"""
    setup_environment()
    apply_custom_styling()
    initialize_session_state()

    st.title("🧬 AI_DS_Team Orchestrator")
    st.markdown("""
    **A2A(Agent-to-Agent) 프로토콜**을 기반으로 9개의 전문 데이터 과학 에이전트 팀을 지휘하여 복잡한 분석 작업을 수행합니다.
    """)
    
    # 에이전트 상태 비동기적으로 확인 및 표시
    if st.button("🔄 에이전트 상태 새로고침"):
        with st.spinner("에이전트 상태를 확인하는 중..."):
            st.session_state.agent_status = asyncio.run(check_ai_ds_team_agents_async())
    
    if not st.session_state.agent_status:
         st.session_state.agent_status = asyncio.run(check_ai_ds_team_agents_async())
         
    display_agent_status(st.session_state.agent_status)

    with st.container(border=True):
        st.subheader("📂 데이터 소스 선택")
        # 데이터 업로드 및 샘플 선택
        handle_data_upload_with_ai_ds_team()
        
    # 데이터 요약 표시
    if st.session_state.uploaded_data is not None:
        display_data_summary_ai_ds_team(st.session_state.uploaded_data, st.session_state.get('data_id', 'Uploaded Data'))
        
    # 챗 인터페이스
    render_ai_ds_team_chat()

if __name__ == "__main__":
    main() 