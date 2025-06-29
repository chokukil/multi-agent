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
from typing import Dict, Any, Tuple
import traceback
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가 (ai.py는 프로젝트 루트에 위치)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 신규 A2A 클라이언트 및 유틸리티 임포트
from core.a2a.a2a_streamlit_client import A2AStreamlitClient
from core.utils.logging import setup_logging
from core.data_manager import DataManager  # DataManager 추가
from core.session_data_manager import SessionDataManager  # 세션 기반 데이터 관리자 추가
from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults # 기존 클래스 활용 가능

# 디버깅 로거 설정
debug_logger = logging.getLogger("ai_ds_debug")
debug_logger.setLevel(logging.DEBUG)
if not debug_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    debug_logger.addHandler(handler)

def debug_log(message: str, level: str = "info"):
    """향상된 디버깅 로그 - 사이드바 설정에 따라 제어"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # 디버깅 모드 확인 (세션 상태에서)
    debug_enabled = getattr(st.session_state, 'debug_enabled', False)
    
    # 로그 메시지 포맷
    if level == "error":
        log_msg = f"[{timestamp}] ❌ ERROR: {message}"
        debug_logger.error(message)
    elif level == "warning":
        log_msg = f"[{timestamp}] ⚠️  WARNING: {message}"
        debug_logger.warning(message)
    elif level == "success":
        log_msg = f"[{timestamp}] ✅ SUCCESS: {message}"
        debug_logger.info(message)
    else:
        log_msg = f"[{timestamp}] ℹ️  INFO: {message}"
        debug_logger.info(message)
    
    # 터미널 출력 (항상 표시)
    print(log_msg)
    
    # 파일에도 기록 (디버깅용)
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/streamlit_debug.log", "a", encoding="utf-8") as f:
            f.write(f"{log_msg}\n")
            f.flush()
    except Exception as e:
        print(f"[{timestamp}] ❌ 로그 파일 기록 실패: {e}")
    
    # Streamlit UI에는 디버깅 모드가 켜져있을 때만 표시
    if debug_enabled:
        try:
            if level == "error":
                st.error(f"🐛 DEBUG: {message}")
            elif level == "warning":
                st.warning(f"🐛 DEBUG: {message}")
            elif level == "success":
                st.success(f"🐛 DEBUG: {message}")
            else:
                st.info(f"🐛 DEBUG: {message}")
        except:
            pass  # Streamlit 컨텍스트가 없을 때는 무시

# AI_DS_Team 유틸리티 임포트
try:
    # 디버깅 정보 출력
    ai_ds_team_path = os.path.join(project_root, "ai_ds_team")
    ai_data_science_team_path = os.path.join(ai_ds_team_path, "ai_data_science_team")
    tools_path = os.path.join(ai_data_science_team_path, "tools")
    dataframe_path = os.path.join(tools_path, "dataframe.py")
    
    debug_log(f"🔍 프로젝트 루트: {project_root}")
    debug_log(f"🔍 ai_ds_team 경로: {ai_ds_team_path} (존재: {os.path.exists(ai_ds_team_path)})")
    debug_log(f"🔍 ai_data_science_team 경로: {ai_data_science_team_path} (존재: {os.path.exists(ai_data_science_team_path)})")
    debug_log(f"🔍 dataframe.py 경로: {dataframe_path} (존재: {os.path.exists(dataframe_path)})")
    
    # ai_ds_team 폴더 안의 ai_data_science_team 모듈 경로 추가
    if ai_ds_team_path not in sys.path:
        sys.path.insert(0, ai_ds_team_path)
        debug_log(f"✅ Python path에 추가됨: {ai_ds_team_path}")
    
    from ai_data_science_team.tools.dataframe import get_dataframe_summary
    AI_DS_TEAM_UTILS_AVAILABLE = True
    debug_log("✅ AI_DS_Team 유틸리티 로드 성공")
except ImportError as e:
    # 완전히 조용한 fallback (경고 메시지 제거)
    AI_DS_TEAM_UTILS_AVAILABLE = False
    def get_dataframe_summary(df): return [f"Shape: {df.shape}"]
    # 터미널에만 로그 출력, UI에는 표시하지 않음
    print(f"[INFO] AI_DS_Team 유틸리티 로드 실패 (정상 동작): {e}")
except Exception as e:
    # 예상치 못한 오류의 경우에만 로그
    AI_DS_TEAM_UTILS_AVAILABLE = False
    def get_dataframe_summary(df): return [f"Shape: {df.shape}"]
    debug_log(f"⚠️ AI_DS_Team 유틸리티 예상치 못한 오류: {e}", "warning")

# --- 초기 설정 ---
setup_logging()

def setup_environment():
    """환경 설정"""
    debug_log("환경 설정 시작")
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        debug_log("Windows 이벤트 루프 정책 설정")
    nest_asyncio.apply()
    load_dotenv()
    debug_log("환경 설정 완료")

def apply_custom_styling():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .agent-card { background: rgba(255, 255, 255, 0.1); color: white; padding: 1.5rem; border-radius: 12px; margin: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease; }
        .agent-card:hover { transform: translateY(-5px); background: rgba(255, 255, 255, 0.2); }
        .stButton > button { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-radius: 8px; padding: 0.7rem 1.5rem; font-weight: 600; }
        .debug-box { background: rgba(255, 255, 255, 0.1); border: 1px solid #ffd700; border-radius: 8px; padding: 10px; margin: 5px 0; }
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
    "mlflow_tools": "📈 MLflow Tools"
}

def map_agent_name(plan_agent_name: str) -> str:
    """계획에서 사용하는 에이전트 이름을 실제 에이전트 이름으로 매핑"""
    return AGENT_NAME_MAPPING.get(plan_agent_name, plan_agent_name)

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_id" not in st.session_state: st.session_state.session_id = f"ui_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if "uploaded_data" not in st.session_state: st.session_state.uploaded_data = None
    if "data_id" not in st.session_state: st.session_state.data_id = None
    if "a2a_client" not in st.session_state: st.session_state.a2a_client = A2AStreamlitClient(AI_DS_TEAM_AGENTS)
    if "agent_status" not in st.session_state: st.session_state.agent_status = {}
    if "active_agent" not in st.session_state: st.session_state.active_agent = None
    if "data_manager" not in st.session_state: st.session_state.data_manager = DataManager()  # DataManager 추가
    if "session_data_manager" not in st.session_state: st.session_state.session_data_manager = SessionDataManager()  # 세션 기반 데이터 관리자 추가

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
                # 포트 정보를 포함하여 A2AStreamlitClient가 올바르게 작동하도록 수정
                results[name] = {
                    "status": "✅", 
                    "description": info['description'],
                    "port": info['port'],  # 🔥 핵심 수정: 포트 정보 추가
                    "capabilities": info.get('capabilities', []),
                    "color": info.get('color', '#ffffff')
                }
            else:
                results[name] = {
                    "status": "❌", 
                    "description": info['description'],
                    "port": info['port'],  # 🔥 실패한 경우에도 포트 정보 유지
                    "capabilities": info.get('capabilities', []),
                    "color": info.get('color', '#ffffff')
                }
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
    """
    아티팩트를 적절한 형식으로 렌더링하는 통합 함수
    Plotly, Matplotlib, 이미지, 코드, 텍스트 등 다양한 형식 지원
    """
    try:
        content_type = artifact_data.get('contentType', artifact_data.get('metadata', {}).get('content_type', 'text/plain'))
        data = artifact_data.get('data', '')
        metadata = artifact_data.get('metadata', {})
        
        debug_log(f"🎨 아티팩트 렌더링: {content_type}")
        
        # 1. Plotly 차트 렌더링
        if content_type == "application/vnd.plotly.v1+json":
            try:
                import plotly.io as pio
                import json
                
                if isinstance(data, str):
                    chart_data = json.loads(data)
                else:
                    chart_data = data
                    
                fig = pio.from_json(json.dumps(chart_data))
                st.plotly_chart(fig, use_container_width=True)
                debug_log("✅ Plotly 차트 렌더링 성공")
                return
                
            except Exception as plotly_error:
                debug_log(f"❌ Plotly 렌더링 실패: {plotly_error}", "error")
                st.error(f"Plotly 차트 렌더링 오류: {plotly_error}")
        
        # 2. Matplotlib/이미지 렌더링
        elif content_type.startswith("image/"):
            try:
                if metadata.get('encoding') == 'base64':
                    import base64
                    image_data = base64.b64decode(data)
                    st.image(image_data, caption=metadata.get('description', 'Generated Chart'))
                else:
                    st.image(data, caption=metadata.get('description', 'Chart'))
                debug_log("✅ 이미지 렌더링 성공")
                return
                
            except Exception as img_error:
                debug_log(f"❌ 이미지 렌더링 실패: {img_error}", "error")
                st.error(f"이미지 렌더링 오류: {img_error}")
        
        # 3. Python 코드 렌더링
        elif content_type == "text/x-python":
            try:
                st.code(data, language='python')
                debug_log("✅ Python 코드 렌더링 성공")
                return
                
            except Exception as code_error:
                debug_log(f"❌ 코드 렌더링 실패: {code_error}", "error")
                st.error(f"코드 렌더링 오류: {code_error}")
        
        # 4. HTML 렌더링
        elif content_type == "text/html":
            try:
                st.components.v1.html(data, height=600, scrolling=True)
                debug_log("✅ HTML 렌더링 성공")
                return
                
            except Exception as html_error:
                debug_log(f"❌ HTML 렌더링 실패: {html_error}", "error")
                st.error(f"HTML 렌더링 오류: {html_error}")
        
        # 5. 마크다운 렌더링 (최종 분석 보고서용)
        elif content_type == "text/markdown":
            try:
                st.markdown(data)
                debug_log("✅ 마크다운 렌더링 성공")
                return
                
            except Exception as md_error:
                debug_log(f"❌ 마크다운 렌더링 실패: {md_error}", "error")
                st.error(f"마크다운 렌더링 오류: {md_error}")
        
        # 6. JSON 데이터 렌더링
        elif content_type == "application/json":
            try:
                if isinstance(data, str):
                    json_data = json.loads(data)
                else:
                    json_data = data
                st.json(json_data)
                debug_log("✅ JSON 렌더링 성공")
                return
                
            except Exception as json_error:
                debug_log(f"❌ JSON 렌더링 실패: {json_error}", "error")
                st.error(f"JSON 렌더링 오류: {json_error}")
        
        # 7. 기본 텍스트 렌더링
        else:
            try:
                if isinstance(data, (dict, list)):
                    st.json(data)
                else:
                    st.text(str(data))
                debug_log("✅ 텍스트 렌더링 성공")
                return
                
            except Exception as text_error:
                debug_log(f"❌ 텍스트 렌더링 실패: {text_error}", "error")
                st.error(f"텍스트 렌더링 오류: {text_error}")
        
    except Exception as e:
        debug_log(f"💥 아티팩트 렌더링 치명적 오류: {e}", "error")
        st.error(f"아티팩트 렌더링 중 오류가 발생했습니다: {e}")
        
        # 최후의 폴백: 원시 데이터 표시
        with st.expander("🔍 원시 아티팩트 데이터", expanded=False):
            st.write("**Content Type:**", content_type)
            st.write("**Data Type:**", type(data))
            st.write("**Metadata:**", metadata)
            st.write("**Data Preview:**", str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data))

async def process_query_streaming(prompt: str):
    """A2A 프로토콜을 사용한 스트리밍 쿼리 처리"""
    debug_log(f"🚀 A2A 스트리밍 쿼리 처리 시작: {prompt[:100]}...")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🧬"):
        try:
            # 1. A2A 클라이언트 초기화
            debug_log("🔧 A2A 클라이언트 초기화 중...")
            a2a_client = A2AStreamlitClient(st.session_state.agent_status, timeout=180.0)
            
            # 2. 세션 기반 데이터 정보 확인
            debug_log("📊 세션 데이터 정보 확인 중...")
            session_manager = st.session_state.session_data_manager
            current_session_id = session_manager.get_current_session_id()
            
            if current_session_id:
                active_file, selection_reason = session_manager.get_active_file_info(current_session_id)
                debug_log(f"📁 활성 파일: {active_file}, 선택 이유: {selection_reason}")
            else:
                debug_log("⚠️ 현재 세션이 없습니다", "warning")
                active_file = None
            
            # 3. 오케스트레이터에게 계획 요청
            debug_log("🧠 오케스트레이터에게 계획 요청 중...")
            try:
                plan_response = await a2a_client.get_plan(prompt)
                debug_log(f"📋 계획 응답 수신: {type(plan_response)}")
                debug_log(f"📋 계획 응답 키: {list(plan_response.keys()) if isinstance(plan_response, dict) else 'Not a dict'}")
                
                # 응답 내용을 자세히 로깅
                if isinstance(plan_response, dict):
                    for key, value in plan_response.items():
                        if isinstance(value, (str, int, float, bool)):
                            debug_log(f"  📝 {key}: {value}")
                        elif isinstance(value, (list, dict)):
                            debug_log(f"  📝 {key}: {type(value)} (길이: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                        else:
                            debug_log(f"  📝 {key}: {type(value)}")
                
            except Exception as plan_error:
                debug_log(f"❌ 계획 요청 실패: {plan_error}", "error")
                debug_log(f"🔍 계획 요청 스택 트레이스: {traceback.format_exc()}", "error")
                st.error(f"계획 생성 실패: {plan_error}")
                return
            
            # 4. 계획 파싱
            debug_log("🔍 계획 파싱 시작...")
            try:
                plan_steps = a2a_client.parse_orchestration_plan(plan_response)
                debug_log(f"📊 파싱된 계획 단계 수: {len(plan_steps)}")
                
                for i, step in enumerate(plan_steps):
                    debug_log(f"  📋 단계 {i+1}: {step.get('agent_name', 'unknown')} - {step.get('task_description', '')[:50]}...")
                
            except Exception as parse_error:
                debug_log(f"❌ 계획 파싱 실패: {parse_error}", "error")
                debug_log(f"🔍 파싱 스택 트레이스: {traceback.format_exc()}", "error")
                st.error(f"계획 파싱 실패: {parse_error}")
                return
            
            # 5. 계획이 비어있는지 확인
            if not plan_steps:
                debug_log("❌ 유효한 계획 단계가 없습니다", "error")
                st.error("오케스트레이터가 유효한 계획을 생성하지 못했습니다.")
                
                # 오케스트레이터 응답을 자세히 표시
                with st.expander("🔍 오케스트레이터 응답 디버깅", expanded=True):
                    st.json(plan_response)
                return
            
            # 6. 계획 실행
            debug_log(f"🚀 {len(plan_steps)}개 단계 실행 시작...")
            
            # 결과 컨테이너
            results_container = st.container()
            
            # ThinkingStream과 PlanVisualization 초기화
            thinking_stream = ThinkingStream()
            plan_viz = PlanVisualization()
            
            thinking_stream.start_thinking("AI_DS_Team이 최적의 분석 계획을 수립하고 있습니다...")
            
            # 계획 시각화
            plan_viz.display_plan(plan_steps, "🧬 AI_DS_Team 실행 계획")
            
            # 각 단계 실행
            all_results = []
            
            for step_idx, step in enumerate(plan_steps):
                step_num = step_idx + 1
                agent_name = step.get('agent_name', 'unknown')
                task_description = step.get('task_description', '')
                
                debug_log(f"🎯 단계 {step_num}/{len(plan_steps)} 실행: {agent_name}")
                
                thinking_stream.add_thought(f"단계 {step_num}: {agent_name}에게 작업을 요청하고 있습니다...", "working")
                
                try:
                    # A2A 스트리밍 실행
                    step_results = []
                    async for chunk in a2a_client.stream_task(agent_name, task_description, active_file):
                        debug_log(f"📦 청크 수신: {chunk.get('type', 'unknown')}")
                        step_results.append(chunk)
                        
                        # 실시간 업데이트 표시
                        if chunk.get('type') == 'progress':
                            thinking_stream.add_thought(chunk.get('content', ''), "working")
                        elif chunk.get('type') == 'result':
                            thinking_stream.add_thought(f"{agent_name} 작업 완료!", "success")
                    
                    # 단계 결과 저장
                    all_results.append({
                        'step': step_num,
                        'agent': agent_name,
                        'task': task_description,
                        'results': step_results
                    })
                    
                    debug_log(f"✅ 단계 {step_num} 완료: {len(step_results)}개 청크 수신", "success")
                    
                except Exception as step_error:
                    debug_log(f"❌ 단계 {step_num} 실행 실패: {step_error}", "error")
                    thinking_stream.add_thought(f"단계 {step_num} 실행 중 오류 발생: {step_error}", "error")
                    
                    # 오류가 있어도 다음 단계 계속 진행
                    all_results.append({
                        'step': step_num,
                        'agent': agent_name,
                        'task': task_description,
                        'error': str(step_error)
                    })
            
            thinking_stream.finish_thinking("AI_DS_Team 분석이 완료되었습니다!")
            
            # 7. 최종 결과 표시
            debug_log("📊 최종 결과 표시 중...")
            
            # 🔍 오케스트레이터 아티팩트 디버깅
            orchestrator_artifacts = []
            total_artifacts = 0
            
            for result in all_results:
                step_results = result.get('results', [])
                agent_name = result['agent']
                
                for chunk in step_results:
                    if chunk.get('type') == 'artifact':
                        total_artifacts += 1
                        artifact = chunk.get('content', {})
                        artifact_name = artifact.get('name', 'Unknown')
                        
                        debug_log(f"🔍 아티팩트 발견: {artifact_name} (from {agent_name})")
                        
                        # 오케스트레이터의 최종 분석 보고서 확인
                        if 'final_analysis_report' in artifact_name.lower():
                            orchestrator_artifacts.append(artifact)
                            debug_log(f"🎯 오케스트레이터 최종 보고서 발견: {artifact_name}")
            
            debug_log(f"📊 총 아티팩트 수: {total_artifacts}, 오케스트레이터 보고서: {len(orchestrator_artifacts)}")
            
            # 🎯 오케스트레이터로부터 최종 종합 분석 보고서 요청
            if not orchestrator_artifacts:
                debug_log("🔍 오케스트레이터 최종 보고서가 없어서 직접 요청합니다...")
                try:
                    # 모든 단계 결과를 요약하여 오케스트레이터에게 최종 분석 요청
                    summary_prompt = f"""
다음은 AI_DS_Team이 수행한 {len(plan_steps)}단계 분석의 결과입니다:

{chr(10).join([f"단계 {r['step']}: {r['agent']} - {'성공' if 'error' not in r else '실패'}" for r in all_results])}

총 {total_artifacts}개의 아티팩트가 생성되었습니다.

이 모든 분석 결과를 종합하여 사용자에게 제공할 최종 분석 보고서를 작성해주세요.
보고서는 마크다운 형식으로 작성하고, 다음을 포함해야 합니다:
1. 분석 개요 및 목적
2. 주요 발견사항
3. 각 단계별 핵심 결과 요약
4. 전체적인 인사이트와 결론
5. 추가 분석 권장사항

사용자 원본 요청: {prompt}
"""
                    
                    # 오케스트레이터에게 최종 보고서 요청
                    final_report_chunks = []
                    async for chunk in a2a_client.stream_task("Orchestrator", summary_prompt):
                        final_report_chunks.append(chunk)
                        debug_log(f"📝 최종 보고서 청크 수신: {chunk.get('type', 'unknown')}")
                    
                    # 최종 보고서 아티팩트 추출
                    for chunk in final_report_chunks:
                        if chunk.get('type') == 'artifact':
                            artifact = chunk.get('content', {})
                            if 'final' in artifact.get('name', '').lower() or 'report' in artifact.get('name', '').lower():
                                orchestrator_artifacts.append(artifact)
                                debug_log(f"✅ 오케스트레이터 최종 보고서 수신: {artifact.get('name', 'Unknown')}")
                
                except Exception as final_report_error:
                    debug_log(f"⚠️ 최종 보고서 요청 실패: {final_report_error}", "warning")
            
            with results_container:
                st.markdown("### 🎯 AI_DS_Team 분석 결과")
                
                for result in all_results:
                    step_num = result['step']
                    agent_name = result['agent']
                    
                    with st.expander(f"📋 단계 {step_num}: {agent_name}", expanded=True):
                        if 'error' in result:
                            st.error(f"오류: {result['error']}")
                        else:
                            step_results = result.get('results', [])
                            
                            if not step_results:
                                st.info(f"{agent_name}에서 결과를 받지 못했습니다.")
                                continue
                            
                            # 메시지와 아티팩트 분리 처리
                            messages = []
                            artifacts = []
                            
                            for chunk in step_results:
                                chunk_type = chunk.get('type', 'unknown')
                                chunk_content = chunk.get('content', {})
                                
                                if chunk_type == 'message' and chunk_content.get('text'):
                                    text = chunk_content['text']
                                    if text and not text.startswith('✅') and len(text.strip()) > 5:
                                        messages.append(text)
                                elif chunk_type == 'artifact':
                                    artifacts.append(chunk_content)
                            
                            # 메시지 표시
                            if messages:
                                st.markdown("#### 💬 에이전트 응답")
                                for msg in messages:
                                    st.markdown(msg)
                            
                            # 아티팩트 렌더링
                            if artifacts:
                                st.markdown("#### 📦 생성된 아티팩트")
                                for i, artifact in enumerate(artifacts):
                                    artifact_name = artifact.get('name', f'Artifact {i+1}')
                                    with st.expander(f"📄 {artifact_name}", expanded=True):
                                        render_artifact(artifact)
            
            # 🎯 오케스트레이터 최종 보고서 표시 (단순한 마크다운 렌더링)
            if orchestrator_artifacts:
                st.markdown("---")
                st.markdown("## 🎯 최종 분석 보고서")
                
                for artifact in orchestrator_artifacts:
                    if 'parts' in artifact and artifact['parts']:
                        for part in artifact['parts']:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                content = part.root.text
                            elif isinstance(part, dict) and 'text' in part:
                                content = part['text']
                            elif isinstance(part, dict) and 'root' in part:
                                content = part['root'].get('text', str(part))
                            else:
                                content = str(part)
                            
                            # 단순한 마크다운 렌더링
                            st.markdown(content)
            
            # 세션 상태 업데이트
            response_summary = f"AI_DS_Team이 {len(plan_steps)}단계 분석을 완료했습니다."
            st.session_state.messages.append({"role": "assistant", "content": response_summary})
            
            debug_log("🎉 A2A 스트리밍 처리 완료!", "success")
            
        except Exception as e:
            debug_log(f"💥 A2A 스트리밍 처리 중 치명적 오류: {e}", "error")
            debug_log(f"🔍 전체 스택 트레이스: {traceback.format_exc()}", "error")
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            
        finally:
            # 클라이언트 정리
            try:
                await a2a_client.close()
                debug_log("🧹 A2A 클라이언트 정리 완료")
            except Exception as cleanup_error:
                debug_log(f"⚠️ 클라이언트 정리 중 오류: {cleanup_error}", "warning")

def get_file_size_info(file_id: str) -> str:
    """파일 크기 정보를 반환하는 헬퍼 함수"""
    try:
        # DataManager에서 파일 정보 가져오기
        if hasattr(st.session_state, 'data_manager'):
            df = st.session_state.data_manager.get_dataframe(file_id)
            if df is not None:
                memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                return f"{df.shape[0]}행 × {df.shape[1]}열, {memory_mb}MB"
        
        # SessionDataManager에서 메타데이터 가져오기
        if hasattr(st.session_state, 'session_data_manager'):
            session_id = st.session_state.session_data_manager.get_current_session_id()
            if session_id and session_id in st.session_state.session_data_manager._session_metadata:
                session_meta = st.session_state.session_data_manager._session_metadata[session_id]
                for file_meta in session_meta.uploaded_files:
                    if file_meta.data_id == file_id:
                        size_mb = round(file_meta.file_size / 1024**2, 2)
                        return f"{file_meta.file_type}, {size_mb}MB"
        
        return "크기 정보 없음"
    except Exception as e:
        debug_log(f"파일 크기 정보 조회 오류: {e}", "warning")
        return "크기 조회 실패"

def handle_file_name_conflict(new_file_name: str, session_id: str) -> Tuple[str, bool]:
    """파일명 중복 처리 UI"""
    try:
        session_manager = st.session_state.session_data_manager
        existing_files = session_manager.get_session_files(session_id)
        
        if new_file_name in existing_files:
            st.warning(f"⚠️ **파일명 중복**: `{new_file_name}`이 이미 존재합니다.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 덮어쓰기", key=f"overwrite_{new_file_name}"):
                    st.info(f"기존 `{new_file_name}` 파일을 덮어씁니다.")
                    return new_file_name, True
            
            with col2:
                if st.button("📝 새 이름으로 저장", key=f"rename_{new_file_name}"):
                    # 자동으로 새 이름 생성
                    base_name = Path(new_file_name).stem
                    extension = Path(new_file_name).suffix
                    counter = 1
                    
                    while f"{base_name}_{counter}{extension}" in existing_files:
                        counter += 1
                    
                    new_name = f"{base_name}_{counter}{extension}"
                    st.success(f"새 이름으로 저장: `{new_name}`")
                    return new_name, True
            
            # 사용자가 아직 선택하지 않음
            return new_file_name, False
        
        # 중복 없음
        return new_file_name, True
        
    except Exception as e:
        debug_log(f"파일명 중복 처리 오류: {e}", "error")
        return new_file_name, True

def display_session_status():
    """세션 상태 표시"""
    try:
        if hasattr(st.session_state, 'session_data_manager'):
            session_manager = st.session_state.session_data_manager
            current_session_id = session_manager.get_current_session_id()
            
            if current_session_id:
                session_age_info = session_manager.check_session_age(current_session_id)
                
                # 세션 상태에 따른 색상 표시
                if session_age_info["status"] == "active":
                    status_color = "🟢"
                elif session_age_info["status"] == "warning":
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                
                st.sidebar.markdown(f"""
                ### {status_color} 세션 상태
                - **세션 ID**: `{current_session_id}`
                - **상태**: {session_age_info["status"]}
                - **생성 시간**: {session_age_info["age_hours"]:.1f}시간 전
                - **정리까지**: {session_age_info["hours_until_cleanup"]:.1f}시간
                """)
                
                # 파일 목록
                files = session_manager.get_session_files(current_session_id)
                if files:
                    st.sidebar.markdown("### 📁 세션 파일")
                    for file_id in files:
                        file_info = get_file_size_info(file_id)
                        active_file, _ = session_manager.get_active_file_info(current_session_id)
                        
                        if file_id == active_file:
                            st.sidebar.markdown(f"🎯 **{file_id}** ({file_info})")
                        else:
                            st.sidebar.markdown(f"📄 {file_id} ({file_info})")
            else:
                st.sidebar.markdown("### ⚪ 세션 없음")
                st.sidebar.info("파일을 업로드하면 새 세션이 생성됩니다.")
    
    except Exception as e:
        debug_log(f"세션 상태 표시 오류: {e}", "warning")

def handle_data_upload_with_ai_ds_team():
    """SessionDataManager를 사용한 세션 기반 파일 업로드 처리"""
    data_manager = st.session_state.data_manager
    session_data_manager = st.session_state.session_data_manager
    
    # 현재 로드된 데이터셋 표시
    loaded_data_info = data_manager.list_dataframe_info()
    
    if loaded_data_info:
        st.success(f"✅ {len(loaded_data_info)}개의 데이터셋이 로드되었습니다.")
        
        # 세션 동기화 확인 및 복구
        current_session_id = session_data_manager.get_current_session_id()
        if not current_session_id or current_session_id not in session_data_manager._session_metadata:
            # 세션이 없거나 메타데이터가 없는 경우 복구
            debug_log("세션 정보가 없어 데이터로부터 세션을 복구합니다.")
            first_data_id = loaded_data_info[0]['data_id']
            df = data_manager.get_dataframe(first_data_id)
            if df is not None:
                new_session_id = session_data_manager.create_session_with_data(
                    data_id=first_data_id,
                    data=df,
                    user_instructions="기존 데이터로부터 세션 복구"
                )
                debug_log(f"세션 복구 완료: {new_session_id}")
                st.info(f"🔄 세션이 복구되었습니다: {new_session_id}")
        
        # 로드된 데이터셋 목록을 expander로 표시
        with st.expander("📋 로드된 데이터셋 보기", expanded=False):
            for info in loaded_data_info:
                data_id = info['data_id']
                shape = info['shape']
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{data_id}** (형태: {shape[0]}행 x {shape[1]}열)")
                with col2:
                    if st.button(f"🗑️ 삭제", key=f"del_{data_id}"):
                        if data_manager.delete_dataframe(data_id):
                            st.toast(f"'{data_id}'가 삭제되었습니다.")
                            st.rerun()
                        else:
                            st.toast(f"'{data_id}' 삭제 실패.", icon="❌")
        
        # 첫 번째 데이터셋을 기본으로 설정
        if not st.session_state.data_id or st.session_state.data_id not in [info['data_id'] for info in loaded_data_info]:
            st.session_state.data_id = loaded_data_info[0]['data_id']
            st.session_state.uploaded_data = data_manager.get_dataframe(st.session_state.data_id)
    else:
        st.info("현재 로드된 데이터가 없습니다. CSV 또는 Excel 파일을 업로드해주세요.")
    
    # 파일 업로드 (다중 파일 지원)
    uploaded_files = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요 (다중 선택 가능)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="여러 파일을 한 번에 업로드할 수 있습니다."
    )
    
    if uploaded_files:
        # 이미 로드된 파일들 확인
        existing_df_ids = set(data_manager.list_dataframes())
        files_to_process = []
        
        for file in uploaded_files:
            file_id = file.name
            if file_id not in existing_df_ids:
                files_to_process.append(file)
        
        if files_to_process:
            files_loaded = 0
            
            for file in files_to_process:
                try:
                    with st.spinner(f"'{file.name}' 처리 중..."):
                        # 파일 읽기
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        # DataManager에 추가 (자동으로 shared_dataframes에 저장됨)
                        file_id = file.name
                        data_manager.add_dataframe(data_id=file_id, data=df, source="File Upload")
                        
                        # 세션 기반 AI DS Team 환경 준비
                        # 이 파일이 AI DS Team에서 사용될 예정이므로 세션에 추가
                        session_id = session_data_manager.create_session_with_data(
                            data_id=file_id,
                            data=df,
                            user_instructions="파일 업로드를 통한 데이터 로드"
                        )
                        
                        # AI DS Team 환경 준비 (ai_ds_team/data/ 폴더에 파일 배치)
                        env_info = session_data_manager.prepare_ai_ds_team_environment(session_id)
                        
                        files_loaded += 1
                        debug_log(f"파일 업로드 성공: {file_id}, shape={df.shape}, session={session_id}")
                        
                        # 첫 번째 파일을 기본 데이터로 설정
                        if not st.session_state.data_id:
                            st.session_state.data_id = file_id
                            st.session_state.uploaded_data = df
                            st.session_state.current_session_id = session_id
                            
                except Exception as e:
                    st.error(f"'{file.name}' 로드 중 오류: {e}")
                    debug_log(f"파일 업로드 실패: {file.name} - {e}", "error")
            
            if files_loaded > 0:
                st.toast(f"✅ {files_loaded}개의 파일이 성공적으로 로드되었습니다!", icon="🎉")
                st.success("🔄 AI DS Team 환경이 준비되었습니다. 이제 에이전트들이 올바른 데이터를 사용할 수 있습니다.")
                st.rerun()
        else:
            # 모든 파일이 이미 로드됨
            file_names = [f.name for f in uploaded_files]
            if len(file_names) == 1:
                st.info(f"'{file_names[0]}'는 이미 로드되어 있습니다.")
            else:
                st.info(f"선택된 {len(file_names)}개 파일이 모두 이미 로드되어 있습니다.")

def display_data_summary_ai_ds_team(data):
    """DataManager 기반 데이터 요약 표시"""
    if data is not None:
        st.markdown("---")
        st.markdown("### 📊 데이터 미리보기")
        
        # 기본 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("행 수", f"{data.shape[0]:,}")
        with col2:
            st.metric("열 수", f"{data.shape[1]:,}")
        with col3:
            st.metric("메모리 사용량", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # 데이터 미리보기
        st.dataframe(data.head(10), use_container_width=True)
        
        # AI_DS_Team 유틸리티가 사용 가능한 경우 추가 정보
        if AI_DS_TEAM_UTILS_AVAILABLE:
            with st.expander("📈 상세 정보", expanded=False):
                summaries = get_dataframe_summary(data)
                for summary in summaries:
                    st.text(summary)

def main():
    """메인 Streamlit 애플리케이션"""
    st.set_page_config(
        page_title="🧬 AI DS Team - 통합 데이터 사이언스 플랫폼",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 사이드바에 디버깅 제어 추가
    with st.sidebar:
        st.markdown("### 🔧 시스템 설정")
        
        # 디버깅 토글
        debug_enabled = st.toggle(
            "🐛 디버깅 모드",
            value=getattr(st.session_state, 'debug_enabled', False),
            help="디버깅 메시지를 UI에 표시할지 선택합니다. 터미널과 로그 파일에는 항상 기록됩니다."
        )
        st.session_state.debug_enabled = debug_enabled
        
        if debug_enabled:
            st.success("🐛 디버깅 모드 활성화")
        else:
            st.info("🔇 디버깅 메시지 숨김")
    
    # 강화된 디버깅 로깅
    debug_log("🚀 Streamlit 애플리케이션 시작", "success")
    
    try:
        # 1. 세션 상태 초기화
        debug_log("🔧 세션 상태 초기화 시작...")
        initialize_session_state()
        debug_log("✅ 세션 상태 초기화 완료", "success")
        
        # 2. 에이전트 상태 확인 및 초기화
        debug_log("🤖 에이전트 상태 확인 시작...")
        if "agent_status" not in st.session_state or not st.session_state.agent_status:
            debug_log("⚠️ 에이전트 상태가 없음, 새로 초기화...", "warning")
            
            try:
                agent_status = asyncio.run(check_agents_status_async())
                st.session_state.agent_status = agent_status
                
                # 에이전트 상태 상세 로깅
                debug_log(f"📊 총 {len(agent_status)}개 에이전트 상태 확인됨")
                available_count = sum(1 for status in agent_status.values() if status.get("status") == "✅")
                debug_log(f"✅ 사용 가능한 에이전트: {available_count}개")
                
                for agent_name, status in agent_status.items():
                    if status.get("status") == "✅":
                        debug_log(f"  ✅ {agent_name} (포트: {status.get('port')})")
                    else:
                        debug_log(f"  ❌ {agent_name} - {status.get('description')}", "warning")
                        
            except Exception as e:
                debug_log(f"❌ 에이전트 상태 확인 실패: {e}", "error")
                import traceback
                debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
                st.session_state.agent_status = {}
        else:
            debug_log(f"✅ 기존 에이전트 상태 사용: {len(st.session_state.agent_status)}개 에이전트")
        
        # 3. UI 렌더링
        debug_log("🎨 UI 렌더링 시작...")
        st.title("🧬 AI_DS_Team Orchestrator")
        st.markdown("> A2A 프로토콜 기반, 9개 전문 데이터 과학 에이전트 팀의 실시간 협업 시스템")
        
        # 세션 상태 표시
        display_session_status()

        if st.button("🔄 에이전트 상태 새로고침") or not st.session_state.agent_status:
            st.session_state.agent_status = asyncio.run(check_agents_status_async())
        display_agent_status()

        with st.container(border=True):
            st.subheader("📂 데이터 소스")
            handle_data_upload_with_ai_ds_team()
            if st.session_state.uploaded_data is not None:
                display_data_summary_ai_ds_team(st.session_state.uploaded_data)

        st.subheader("💬 AI DS Team과 대화하기")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input("데이터에 대해 질문하세요..."):
            debug_log(f"📝 사용자 입력: {prompt[:100]}...")
            # Streamlit에서 비동기 함수를 안전하게 실행
            try:
                debug_log("🔄 비동기 처리 시작...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_query_streaming(prompt))
                loop.close()
                debug_log("✅ 비동기 처리 완료", "success")
            except Exception as e:
                debug_log(f"❌ 비동기 처리 중 오류: {e}", "error")
                import traceback
                debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
                
                st.error(f"처리 중 오류 발생: {e}")
                # 동기 버전으로 폴백
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("assistant", avatar="🧬"):
                    st.error("비동기 처리에 실패했습니다. 시스템 관리자에게 문의하세요.")
        
        debug_log("✅ UI 렌더링 완료", "success")
        
    except Exception as e:
        debug_log(f"💥 메인 함수 실행 중 오류: {e}", "error")
        import traceback
        debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
        
        # 사용자에게 오류 표시
        st.error(f"🚨 애플리케이션 초기화 중 오류가 발생했습니다: {e}")
        st.error("개발자 도구에서 콘솔 로그를 확인하거나 페이지를 새로고침해보세요.")
        
        # 기본 UI라도 표시
        st.title("🧬 AI DS Team")
        st.warning("시스템 초기화 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.")

if __name__ == "__main__":
    main()