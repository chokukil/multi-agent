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

# 새로운 UI 컴포넌트 임포트
try:
    from core.ui.smart_display import SmartDisplayManager, AccumulativeStreamContainer
    from core.ui.a2a_orchestration_ui import A2AOrchestrationDashboard
    SMART_UI_AVAILABLE = True
    debug_log("✅ Smart UI 컴포넌트 로드 성공")
except ImportError as e:
    SMART_UI_AVAILABLE = False
    debug_log(f"⚠️ Smart UI 컴포넌트 로드 실패: {e}", "warning")

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
    """아티팩트를 Smart Display Manager로 렌더링"""
    debug_log(f"🎨 Smart Display로 아티팩트 렌더링 시작")
    
    try:
        # Smart Display Manager 사용 가능 여부 확인
        if SMART_UI_AVAILABLE:
            smart_display = SmartDisplayManager()
            
            # 아티팩트 메타데이터 추출
            name = artifact_data.get("name", "Unknown Artifact")
            parts = artifact_data.get("parts", [])
            metadata = artifact_data.get("metadata", {})
            
            # 아티팩트 헤더 표시
            st.markdown(f"### 📄 {name}")
            
            # 각 part 처리
            for i, part in enumerate(parts):
                part_type = part.get("kind", "unknown")
                
                if part_type == "text":
                    # 텍스트 콘텐츠를 Smart Display로 처리
                    text_content = part.get("text", "")
                    smart_display.smart_display_content(text_content)
                    
                elif part_type == "data":
                    # 데이터 콘텐츠 처리
                    data_content = part.get("data", {})
                    content_type = data_content.get("type", "application/json")
                    actual_data = data_content.get("data", {})
                    
                    # Plotly 차트 특별 처리 (고유 키 사용)
                    if content_type == "application/vnd.plotly.v1+json":
                        try:
                            import plotly.io as pio
                            import json
                            
                            if isinstance(actual_data, str):
                                chart_data = json.loads(actual_data)
                            else:
                                chart_data = actual_data
                                
                            fig = pio.from_json(json.dumps(chart_data))
                            
                            # 고유 키로 Plotly 차트 렌더링
                            chart_id = f"artifact_{name}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                            st.plotly_chart(fig, key=chart_id, use_container_width=True)
                            debug_log("✅ Plotly 차트 Smart Display 렌더링 성공")
                            
                        except Exception as plotly_error:
                            debug_log(f"❌ Plotly Smart Display 렌더링 실패: {plotly_error}", "error")
                            st.error(f"Plotly 차트 렌더링 오류: {plotly_error}")
                    else:
                        # 기타 데이터 타입은 Smart Display로 처리
                        smart_display.smart_display_content(actual_data)
                
                else:
                    # 알 수 없는 타입은 Smart Display로 처리
                    smart_display.smart_display_content(part)
            
            debug_log("✅ Smart Display 아티팩트 렌더링 완료")
            
        else:
            # Smart Display를 사용할 수 없는 경우 기존 방식 사용
            debug_log("⚠️ Smart Display 사용 불가, 기존 방식 사용", "warning")
            _render_artifact_fallback(artifact_data)
            
    except Exception as e:
        debug_log(f"💥 Smart Display 아티팩트 렌더링 오류: {e}", "error")
        st.error(f"아티팩트 렌더링 중 오류가 발생했습니다: {e}")
        
        # 최후의 폴백: 원시 데이터 표시
        with st.expander("🔍 원시 아티팩트 데이터", expanded=False):
            st.json(artifact_data)

def _render_artifact_fallback(artifact_data: Dict[str, Any]):
    """Smart Display를 사용할 수 없을 때의 폴백 렌더링"""
    content_type = artifact_data.get('contentType', artifact_data.get('metadata', {}).get('content_type', 'text/plain'))
    data = artifact_data.get('data', '')
    metadata = artifact_data.get('metadata', {})
    
    debug_log(f"🎨 폴백 아티팩트 렌더링: {content_type}")
    
    # Plotly 차트 렌더링
    if content_type == "application/vnd.plotly.v1+json":
        try:
            import plotly.io as pio
            import json
            
            if isinstance(data, str):
                chart_data = json.loads(data)
            else:
                chart_data = data
                
            fig = pio.from_json(json.dumps(chart_data))
            
            # 고유 키 생성하여 렌더링
            chart_id = f"fallback_chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            st.plotly_chart(fig, key=chart_id, use_container_width=True)
            debug_log("✅ 폴백 Plotly 차트 렌더링 성공")
            
        except Exception as plotly_error:
            debug_log(f"❌ 폴백 Plotly 렌더링 실패: {plotly_error}", "error")
            st.error(f"Plotly 차트 렌더링 오류: {plotly_error}")
    
    # 이미지 렌더링
    elif content_type.startswith("image/"):
        try:
            if metadata.get('encoding') == 'base64':
                import base64
                image_data = base64.b64decode(data)
                st.image(image_data, caption=metadata.get('description', 'Generated Chart'))
            else:
                st.image(data, caption=metadata.get('description', 'Chart'))
            debug_log("✅ 폴백 이미지 렌더링 성공")
            
        except Exception as img_error:
            debug_log(f"❌ 폴백 이미지 렌더링 실패: {img_error}", "error")
            st.error(f"이미지 렌더링 오류: {img_error}")
    
    # 코드 렌더링
    elif content_type == "text/x-python":
        st.code(data, language='python')
        debug_log("✅ 폴백 코드 렌더링 성공")
    
    # 마크다운 렌더링
    elif content_type == "text/markdown":
        st.markdown(data)
        debug_log("✅ 폴백 마크다운 렌더링 성공")
    
    # JSON 렌더링
    elif content_type == "application/json":
        try:
            if isinstance(data, str):
                json_data = json.loads(data)
            else:
                json_data = data
            st.json(json_data)
            debug_log("✅ 폴백 JSON 렌더링 성공")
            
        except Exception as json_error:
            debug_log(f"❌ 폴백 JSON 렌더링 실패: {json_error}", "error")
            st.text(str(data))
    
    # 기본 텍스트 렌더링
    else:
        if isinstance(data, (dict, list)):
            st.json(data)
        else:
            st.text(str(data))
        debug_log("✅ 폴백 텍스트 렌더링 성공")

async def process_query_streaming(prompt: str):
    """A2A 프로토콜을 사용한 실시간 스트리밍 쿼리 처리"""
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
                
            except Exception as plan_error:
                debug_log(f"❌ 계획 요청 실패: {plan_error}", "error")
                st.error(f"계획 생성 실패: {plan_error}")
                return
            
            # 4. 계획 파싱
            debug_log("🔍 계획 파싱 시작...")
            try:
                plan_steps = a2a_client.parse_orchestration_plan(plan_response)
                debug_log(f"📊 파싱된 계획 단계 수: {len(plan_steps)}")
                
            except Exception as parse_error:
                debug_log(f"❌ 계획 파싱 실패: {parse_error}", "error")
                st.error(f"계획 파싱 실패: {parse_error}")
                return
            
            # 5. CherryAI v8 오케스트레이터 단일 응답 처리
            if not plan_steps:
                debug_log("❌ 유효한 계획 단계가 없습니다", "error")
                
                # CherryAI v8 오케스트레이터의 comprehensive_analysis 아티팩트 확인
                if isinstance(plan_response, dict) and "result" in plan_response:
                    result = plan_response["result"]
                    if "artifacts" in result:
                        for artifact in result["artifacts"]:
                            if artifact.get("name") == "comprehensive_analysis":
                                debug_log("🧠 CherryAI v8 종합 분석 결과 발견!", "success")
                                
                                # 실시간 스트리밍 컨테이너 생성
                                streaming_container = st.empty()
                                
                                # v8 분석 결과를 스트리밍으로 표시
                                parts = artifact.get("parts", [])
                                for part in parts:
                                    if part.get("kind") == "text":
                                        analysis_text = part.get("text", "")
                                        if analysis_text:
                                            # 텍스트를 문장 단위로 분할하여 스트리밍
                                            sentences = analysis_text.split('. ')
                                            displayed_text = ""
                                            
                                            streaming_container.markdown("### 🧠 CherryAI v8 Universal Intelligence 분석 결과")
                                            text_container = st.empty()
                                            
                                            for i, sentence in enumerate(sentences):
                                                if sentence.strip():
                                                    displayed_text += sentence
                                                    if i < len(sentences) - 1:
                                                        displayed_text += ". "
                                                    
                                                    # 실시간 업데이트
                                                    text_container.markdown(displayed_text)
                                                    
                                                    # 스트리밍 효과
                                                    import asyncio
                                                    await asyncio.sleep(0.3)
                                            
                                            debug_log("✅ v8 분석 결과 스트리밍 완료", "success")
                                            return
                
                st.error("오케스트레이터가 유효한 계획을 생성하지 못했습니다.")
                return
            
            # 6. 다단계 계획 실행 - 실시간 스트리밍
            debug_log(f"🚀 {len(plan_steps)}개 단계 실행 시작...")
            
            # 실시간 스트리밍 컨테이너들
            plan_container = st.container()
            streaming_container = st.empty()
            results_container = st.container()
            
            # 계획 시각화
            with plan_container:
                st.markdown("### 🧬 AI_DS_Team 실행 계획")
                plan_cols = st.columns(len(plan_steps))
                
                for i, step in enumerate(plan_steps):
                    with plan_cols[i]:
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center;">
                            <h4>단계 {i+1}</h4>
                            <p><strong>{step.get('agent_name', 'Unknown')}</strong></p>
                            <p style="font-size: 0.8em;">{step.get('task_description', '')[:50]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # 각 단계 실시간 실행
            all_results = []
            
            for step_idx, step in enumerate(plan_steps):
                step_num = step_idx + 1
                agent_name = step.get('agent_name', 'unknown')
                task_description = step.get('task_description', '')
                
                debug_log(f"🎯 단계 {step_num}/{len(plan_steps)} 실행: {agent_name}")
                
                # 실시간 진행 상황 표시
                with streaming_container:
                    st.markdown(f"### 🔄 단계 {step_num}/{len(plan_steps)} 진행 중...")
                    st.markdown(f"**에이전트**: {agent_name}")
                    st.markdown(f"**작업**: {task_description}")
                    
                    # 실시간 스트리밍 텍스트 컨테이너
                    live_text_container = st.empty()
                    live_artifacts_container = st.empty()
                
                try:
                    # A2A 스트리밍 실행
                    step_results = []
                    displayed_text = ""
                    step_artifacts = []
                    
                    async for chunk in a2a_client.stream_task(agent_name, task_description, active_file):
                        debug_log(f"📦 청크 수신: {chunk.get('type', 'unknown')}")
                        step_results.append(chunk)
                        
                        chunk_type = chunk.get('type', 'unknown')
                        chunk_content = chunk.get('content', {})
                        is_final = chunk.get('final', False)
                        
                        # 실시간 메시지 스트리밍 표시
                        if chunk_type == 'message':
                            text = chunk_content.get('text', '')
                            if text and not text.startswith('✅'):  # 완료 메시지 제외
                                # Smart UI 사용 가능 시 누적형 컨테이너 사용
                                if SMART_UI_AVAILABLE:
                                    # 누적형 스트리밍 컨테이너가 없으면 생성
                                    if 'current_stream_container' not in locals():
                                        current_stream_container = AccumulativeStreamContainer(f"🤖 {agent_name} 실시간 응답")
                                    
                                    # 청크를 누적하여 추가
                                    current_stream_container.add_chunk(text, "message")
                                    
                                else:
                                    # 기존 방식 (청크가 사라지는 문제 있음)
                                    displayed_text += text + " "
                                    
                                    with live_text_container:
                                        st.markdown(f"**{agent_name} 응답:**")
                                        st.markdown(displayed_text)
                        
                        # 아티팩트 실시간 표시
                        elif chunk_type == 'artifact':
                            step_artifacts.append(chunk_content)
                            
                            if SMART_UI_AVAILABLE:
                                # Smart Display로 아티팩트 렌더링
                                if 'current_stream_container' not in locals():
                                    current_stream_container = AccumulativeStreamContainer(f"🤖 {agent_name} 실시간 응답")
                                
                                current_stream_container.add_chunk(chunk_content, "artifact")
                                
                            else:
                                # 기존 방식
                                with live_artifacts_container:
                                    st.markdown("**생성된 아티팩트:**")
                                    for i, artifact in enumerate(step_artifacts):
                                        with st.expander(f"📄 {artifact.get('name', f'Artifact {i+1}')}", expanded=True):
                                            render_artifact(artifact)
                        
                        # final 플래그 확인
                        if is_final:
                            debug_log(f"✅ 단계 {step_num} 최종 청크 수신", "success")
                            break
                    
                    # 단계 결과 저장
                    all_results.append({
                        'step': step_num,
                        'agent': agent_name,
                        'task': task_description,
                        'results': step_results,
                        'displayed_text': displayed_text,
                        'artifacts': step_artifacts
                    })
                    
                    debug_log(f"✅ 단계 {step_num} 완료: {len(step_results)}개 청크 수신", "success")
                    
                except Exception as step_error:
                    debug_log(f"❌ 단계 {step_num} 실행 실패: {step_error}", "error")
                    
                    with live_text_container:
                        st.error(f"단계 {step_num} 실행 중 오류 발생: {step_error}")
                    
                    all_results.append({
                        'step': step_num,
                        'agent': agent_name,
                        'task': task_description,
                        'error': str(step_error)
                    })
            
            # 7. 최종 결과 정리 표시
            debug_log("📊 최종 결과 정리 중...")
            
            with streaming_container:
                st.markdown("### ✅ 모든 단계 완료!")
                st.success("AI_DS_Team 분석이 성공적으로 완료되었습니다.")
            
            # 8. 종합 결과 표시
            with results_container:
                st.markdown("---")
                st.markdown("### 🎯 AI_DS_Team 분석 종합 결과")
                
                # 성공한 단계들의 결과 요약
                successful_steps = [r for r in all_results if 'error' not in r]
                total_artifacts = sum(len(r.get('artifacts', [])) for r in successful_steps)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("완료된 단계", f"{len(successful_steps)}/{len(plan_steps)}")
                with col2:
                    st.metric("생성된 아티팩트", total_artifacts)
                with col3:
                    st.metric("처리 시간", f"{len(plan_steps) * 5}초 (예상)")
                
                # 각 단계별 상세 결과
                for result in all_results:
                    step_num = result['step']
                    agent_name = result['agent']
                    
                    with st.expander(f"📋 단계 {step_num}: {agent_name}", expanded=True):
                        if 'error' in result:
                            st.error(f"오류: {result['error']}")
                        else:
                            # 최종 텍스트 응답 표시
                            if result.get('displayed_text'):
                                st.markdown("#### 💬 에이전트 응답")
                                st.markdown(result['displayed_text'])
                            
                            # 아티팩트 표시
                            if result.get('artifacts'):
                                st.markdown("#### 📦 생성된 아티팩트")
                                for artifact in result['artifacts']:
                                    artifact_name = artifact.get('name', 'Unknown')
                                    with st.expander(f"📄 {artifact_name}", expanded=True):
                                        render_artifact(artifact)
            
            debug_log("🎉 전체 스트리밍 프로세스 완료!", "success")
            
        except Exception as e:
            debug_log(f"💥 전체 프로세스 오류: {e}", "error")
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            import traceback
            debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")

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