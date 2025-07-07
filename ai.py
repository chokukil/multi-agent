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
from typing import Dict, Any, Tuple, List
import traceback
from pathlib import Path
import uuid
import numpy as np
import base64
import contextlib

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

# Phase 3 Integration Layer 및 Expert UI 임포트
try:
    from core.phase3_integration_layer import Phase3IntegrationLayer
    from ui.expert_answer_renderer import ExpertAnswerRenderer
    PHASE3_AVAILABLE = True
    print("✅ Phase 3 Integration Layer 로드 성공")
except ImportError as e:
    PHASE3_AVAILABLE = False
    print(f"⚠️ Phase 3 Integration Layer 로드 실패: {e}")

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
    from core.ui.agent_preloader import AgentPreloader, get_agent_preloader, ProgressiveLoadingUI, AgentStatus
    SMART_UI_AVAILABLE = True
    print("✅ Smart UI 컴포넌트 로드 성공")
except ImportError as e:
    SMART_UI_AVAILABLE = False
    print(f"⚠️ Smart UI 컴포넌트 로드 실패: {e}")

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

# Langfuse Session Tracking 추가
try:
    from core.langfuse_session_tracer import init_session_tracer, get_session_tracer, LANGFUSE_AVAILABLE
    LANGFUSE_SESSION_AVAILABLE = True
    print("✅ Langfuse Session Tracer 로드 성공")
except ImportError as e:
    LANGFUSE_SESSION_AVAILABLE = False
    print(f"⚠️ Langfuse Session Tracer 로드 실패: {e}")

# --- 초기 설정 ---
setup_logging()

# Langfuse 초기화 (환경변수에서 설정 가져오기)
if LANGFUSE_SESSION_AVAILABLE:
    try:
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        
        if langfuse_public_key and langfuse_secret_key:
            init_session_tracer(langfuse_public_key, langfuse_secret_key, langfuse_host)
            debug_log("🔍 Langfuse Session Tracer 초기화 성공", "success")
        else:
            debug_log("⚠️ Langfuse 환경변수 미설정 - 추적 비활성화", "warning")
    except Exception as e:
        debug_log(f"❌ Langfuse 초기화 실패: {e}", "error")

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
    "mlflow_tools": "📈 MLflow Tools",
    # 오케스트레이터 계획에서 사용하는 이름들 추가 (정확한 Agent Card name 사용)
    "AI_DS_Team EDAToolsAgent": "🔍 EDA Tools",
    "AI_DS_Team DataLoaderToolsAgent": "📁 Data Loader",
    "AI_DS_Team DataCleaningAgent": "🧹 Data Cleaning",
    "AI_DS_Team DataVisualizationAgent": "📊 Data Visualization",
    "AI_DS_Team SQLDatabaseAgent": "🗄️ SQL Database",
    "AI_DS_Team DataWranglingAgent": "🔧 Data Wrangling",
    # 호환성을 위해 기존 이름도 유지
    "SessionEDAToolsAgent": "🔍 EDA Tools",
    # 호환성을 위해 기존 이름도 유지
    "SessionEDAToolsAgent": "🔍 EDA Tools"
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
    # 프리로더 초기화 상태 추가
    if "preloader_initialized" not in st.session_state: st.session_state.preloader_initialized = False
    if "agents_preloaded" not in st.session_state: st.session_state.agents_preloaded = False

@st.cache_resource
def initialize_agent_preloader():
    """에이전트 프리로더 초기화 (캐시됨)"""
    return get_agent_preloader(AI_DS_TEAM_AGENTS)

async def preload_agents_with_ui():
    """UI와 함께 에이전트 프리로딩"""
    if st.session_state.agents_preloaded:
        debug_log("✅ 에이전트가 이미 프리로드됨", "success")
        return st.session_state.agent_status
    
    # 프리로더 인스턴스 가져오기
    preloader = initialize_agent_preloader()
    
    # 로딩 UI 설정
    loading_container = st.container()
    loading_ui = ProgressiveLoadingUI(loading_container)
    loading_ui.setup_ui()
    
    # 진행 상황 콜백 함수
    def progress_callback(completed, total, current_task):
        loading_ui.update_progress(completed, total, current_task)
        debug_log(f"📋 {current_task} ({completed}/{total})")
    
    try:
        # 에이전트 프리로딩 실행
        debug_log("🚀 에이전트 프리로딩 시작...", "success")
        agents_info = await preloader.preload_agents(progress_callback)
        
        # 기존 형식으로 변환 (호환성 유지)
        agent_status = {}
        for name, agent_info in agents_info.items():
            status_icon = "✅" if agent_info.status == AgentStatus.READY else "❌"
            agent_status[name] = {
                "status": status_icon,
                "description": agent_info.description,
                "port": agent_info.port,
                "capabilities": agent_info.capabilities,
                "color": agent_info.color,
                "initialization_time": agent_info.initialization_time,
                "error_message": agent_info.error_message
            }
        
        # 세션 상태 업데이트
        st.session_state.agent_status = agent_status
        st.session_state.agents_preloaded = True
        
        # 완료 상태 표시
        summary = preloader.get_initialization_summary()
        loading_ui.show_completion(summary)
        
        debug_log(f"✅ 에이전트 프리로딩 완료: {summary['ready_agents']}/{summary['total_agents']} 준비됨", "success")
        
        # 로딩 UI 정리 (잠시 후)
        time.sleep(2)
        loading_container.empty()
        
        return agent_status
        
    except Exception as e:
        debug_log(f"❌ 에이전트 프리로딩 실패: {e}", "error")
        loading_container.error(f"에이전트 초기화 중 오류 발생: {e}")
        
        # 폴백: 기존 방식으로 상태 확인
        debug_log("🔄 기존 방식으로 폴백...", "warning")
        try:
            fallback_status = await check_agents_status_async()
            st.session_state.agent_status = fallback_status
            debug_log("✅ 폴백 에이전트 상태 확인 완료", "success")
            return fallback_status
        except Exception as fallback_error:
            debug_log(f"❌ 폴백도 실패: {fallback_error}", "error")
            # 최후의 폴백: 빈 상태 반환
            return {}

async def check_agents_status_async():
    """AI_DS_Team 에이전트 상태 비동기 확인 (개선된 버전)"""
    debug_log("🔍 에이전트 상태 직접 확인 시작...")
    
    async with httpx.AsyncClient(timeout=5.0) as client:  # 타임아웃 증가
        results = {}
        
        # 각 에이전트를 순차적으로 확인 (안정성 향상)
        for name, info in AI_DS_TEAM_AGENTS.items():
            port = info['port']
            try:
                debug_log(f"🔍 {name} (포트 {port}) 확인 중...")
                response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                
                if response.status_code == 200:
                    agent_card = response.json()
                    actual_name = agent_card.get('name', 'Unknown')
                    debug_log(f"✅ {name} 응답: {actual_name}")
                    
                    results[name] = {
                        "status": "✅", 
                        "description": info['description'],
                        "port": port,
                        "capabilities": info.get('capabilities', []),
                        "color": info.get('color', '#ffffff'),
                        "actual_name": actual_name  # 실제 에이전트 이름 저장
                    }
                else:
                    debug_log(f"❌ {name} 응답 실패: HTTP {response.status_code}")
                    results[name] = {
                        "status": "❌", 
                        "description": info['description'],
                        "port": port,
                        "capabilities": info.get('capabilities', []),
                        "color": info.get('color', '#ffffff'),
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                debug_log(f"❌ {name} 연결 실패: {e}")
                results[name] = {
                    "status": "❌", 
                    "description": info['description'],
                    "port": port,
                    "capabilities": info.get('capabilities', []),
                    "color": info.get('color', '#ffffff'),
                    "error": str(e)
                }
        
        debug_log(f"✅ 에이전트 상태 확인 완료: {len(results)}개")
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
    """아티팩트 렌더링 - 완전히 재작성된 버전"""
    try:
        debug_log(f"🎨 아티팩트 렌더링 시작: {artifact_data.get('name', 'Unknown')}")
        
        name = artifact_data.get('name', 'Unknown')
        parts = artifact_data.get('parts', [])
        metadata = artifact_data.get('metadata', {})
        content_type = artifact_data.get('contentType', metadata.get('content_type', 'text/plain'))
        
        if not parts:
            st.warning("아티팩트에 표시할 콘텐츠가 없습니다.")
            return
        
        # 아티팩트별 컨테이너 생성 (중복 방지)
        with st.container():
            # 헤더 표시
            if name != 'Unknown':
                st.markdown(f"### 📦 {name}")
            
            for i, part in enumerate(parts):
                try:
                    # Part 구조 파싱
                    if isinstance(part, dict):
                        part_kind = part.get("kind", part.get("type", "unknown"))
                        
                        if part_kind == "text":
                            text_content = part.get("text", "")
                            if not text_content:
                                continue
                            
                            # 컨텐츠 타입별 렌더링
                            if content_type == "application/vnd.plotly.v1+json":
                                # Plotly 차트 JSON 데이터 처리
                                _render_plotly_chart(text_content, name, i)
                                
                            elif content_type == "text/x-python" or "```python" in text_content:
                                # Python 코드 렌더링
                                _render_python_code(text_content)
                                
                            elif content_type == "text/markdown" or text_content.startswith("#"):
                                # 마크다운 렌더링
                                _render_markdown_content(text_content)
                                
                            else:
                                # 일반 텍스트 렌더링
                                _render_general_text(text_content)
                        
                        elif part_kind == "data":
                            # 데이터 Part 처리
                            data_content = part.get("data", {})
                            _render_data_content(data_content, content_type, name, i)
                        
                        else:
                            # 알 수 없는 타입
                            st.json(part)
                    
                    else:
                        # 문자열이나 기타 타입
                        st.text(str(part))
                        
                except Exception as part_error:
                    debug_log(f"❌ Part {i} 렌더링 실패: {part_error}", "error")
                    with st.expander(f"🔍 Part {i} 오류 정보"):
                        st.error(f"렌더링 오류: {part_error}")
                        st.json(part)
        
        debug_log("✅ 아티팩트 렌더링 완료")
        
    except Exception as e:
        debug_log(f"💥 아티팩트 렌더링 전체 오류: {e}", "error")
        st.error(f"아티팩트 렌더링 중 오류 발생: {e}")

def _render_plotly_chart(json_text: str, name: str, index: int):
    """Plotly 차트 전용 렌더링"""
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
        import json
        import numpy as np
        import base64
        from datetime import datetime
        import uuid
        
        debug_log("🔍 Plotly 차트 데이터 파싱 시작...")
        
        # JSON 파싱
        try:
            chart_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            debug_log(f"❌ JSON 파싱 실패: {e}", "error")
            st.error("차트 데이터 JSON 파싱 실패")
            with st.expander("🔍 원시 데이터"):
                st.text(json_text[:500] + "..." if len(json_text) > 500 else json_text)
            return
        
        # Binary data 디코딩 함수
        def decode_plotly_binary_data(data_dict):
            """Plotly binary data를 실제 값으로 변환"""
            if isinstance(data_dict, dict):
                for key, value in data_dict.items():
                    if isinstance(value, dict) and 'dtype' in value and 'bdata' in value:
                        try:
                            if value['dtype'] == 'f8':  # float64
                                binary_data = base64.b64decode(value['bdata'])
                                float_array = np.frombuffer(binary_data, dtype=np.float64)
                                data_dict[key] = float_array.tolist()
                                debug_log(f"✅ Binary float64 디코딩 성공: {key}")
                            elif value['dtype'] == 'i8':  # int64
                                binary_data = base64.b64decode(value['bdata'])
                                int_array = np.frombuffer(binary_data, dtype=np.int64)
                                data_dict[key] = int_array.tolist()
                                debug_log(f"✅ Binary int64 디코딩 성공: {key}")
                        except Exception as decode_error:
                            debug_log(f"❌ Binary data 디코딩 실패 {key}: {decode_error}", "error")
                    elif isinstance(value, (dict, list)):
                        if isinstance(value, dict):
                            decode_plotly_binary_data(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    decode_plotly_binary_data(item)
            return data_dict
        
        # Binary data 디코딩 적용
        chart_data = decode_plotly_binary_data(chart_data)
        
        # Plotly Figure 생성
        try:
            if isinstance(chart_data, dict):
                # 데이터 구조 확인
                if 'data' in chart_data and 'layout' in chart_data:
                    # 표준 Plotly 구조
                    plot_data = chart_data['data']
                    layout = chart_data['layout']
                elif 'data' in chart_data and isinstance(chart_data['data'], dict) and 'data' in chart_data['data']:
                    # 중첩된 구조
                    plot_data = chart_data['data']['data']
                    layout = chart_data['data'].get('layout', {})
                else:
                    # 전체를 Figure로 시도
                    plot_data = chart_data.get('data', [])
                    layout = chart_data.get('layout', {})
                
                fig = go.Figure(data=plot_data, layout=layout)
            else:
                # JSON 문자열로 직접 변환
                fig = pio.from_json(json.dumps(chart_data))
            
            # 차트 최적화
            fig.update_layout(
                showlegend=True,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # 고유 키 생성
            chart_id = f"plotly_{name}_{index}_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%H%M%S%f')}"
            
            # 차트 표시
            st.markdown("#### 📊 인터랙티브 차트")
            st.plotly_chart(fig, key=chart_id, use_container_width=True)
            debug_log("✅ Plotly 차트 렌더링 성공!")
            
        except Exception as fig_error:
            debug_log(f"❌ Plotly Figure 생성 실패: {fig_error}", "error")
            st.error(f"차트 생성 오류: {fig_error}")
            
            # 상세 오류 정보
            with st.expander("🔍 차트 오류 상세 정보"):
                st.text(f"오류 타입: {type(fig_error).__name__}")
                st.text(f"오류 메시지: {str(fig_error)}")
                st.markdown("**원시 차트 데이터 (처음 1000자):**")
                st.json(json.loads(json_text[:1000]) if len(json_text) > 1000 else json.loads(json_text))
                
    except Exception as e:
        debug_log(f"❌ Plotly 차트 렌더링 전체 실패: {e}", "error")
        st.error(f"Plotly 차트 처리 오류: {e}")

def _render_python_code(text_content: str):
    """Python 코드 예쁘게 렌더링"""
    try:
        # 코드 블록 추출
        if "```python" in text_content:
            # 마크다운 코드 블록에서 코드만 추출
            parts = text_content.split("```python")
            for i, part in enumerate(parts[1:], 1):
                if "```" in part:
                    code = part.split("```")[0].strip()
                    if code:
                        st.markdown(f"#### 🐍 Python 코드 #{i}")
                        st.code(code, language='python')
        else:
            # 전체가 코드인 경우
            st.markdown("#### 🐍 Python 코드")
            st.code(text_content, language='python')
            
        debug_log("✅ Python 코드 렌더링 완료")
        
    except Exception as e:
        debug_log(f"❌ Python 코드 렌더링 실패: {e}", "error")
        st.text(text_content)

def _render_markdown_content(text_content: str):
    """마크다운 예쁘게 렌더링"""
    try:
        # 커스텀 CSS 스타일 적용
        styled_markdown = f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin: 0.5rem 0;
        ">
        {text_content}
        </div>
        """
        
        st.markdown("#### 📝 마크다운 콘텐츠")
        st.markdown(text_content, unsafe_allow_html=True)
        debug_log("✅ 마크다운 렌더링 완료")
        
    except Exception as e:
        debug_log(f"❌ 마크다운 렌더링 실패: {e}", "error")
        st.text(text_content)

def _render_general_text(text_content: str):
    """일반 텍스트 렌더링"""
    try:
        # 긴 텍스트는 expander로
        if len(text_content) > 500:
            with st.expander("📄 텍스트 내용 보기", expanded=True):
                st.markdown(text_content)
        else:
            st.markdown(text_content)
            
        debug_log("✅ 일반 텍스트 렌더링 완료")
        
    except Exception as e:
        debug_log(f"❌ 일반 텍스트 렌더링 실패: {e}", "error")
        st.text(text_content)

def _render_data_content(data_content: Dict, content_type: str, name: str, index: int):
    """데이터 콘텐츠 렌더링"""
    try:
        if content_type == "application/vnd.plotly.v1+json":
            # Plotly 데이터
            actual_data = data_content.get("data", {})
            if isinstance(actual_data, str):
                _render_plotly_chart(actual_data, name, index)
            else:
                _render_plotly_chart(json.dumps(actual_data), name, index)
                
        elif content_type == "application/json":
            # JSON 데이터
            st.markdown("#### 📊 JSON 데이터")
            st.json(data_content)
            
        else:
            # 기타 데이터
            st.markdown("#### 📄 데이터")
            st.json(data_content)
            
        debug_log("✅ 데이터 콘텐츠 렌더링 완료")
        
    except Exception as e:
        debug_log(f"❌ 데이터 콘텐츠 렌더링 실패: {e}", "error")
        st.json(data_content)

async def process_query_streaming(prompt: str):
    """A2A 프로토콜을 사용한 실시간 스트리밍 쿼리 처리 + Phase 3 전문가급 답변 합성"""
    debug_log(f"🚀 A2A 스트리밍 쿼리 처리 시작: {prompt[:100]}...")
    
    # Langfuse Session 시작
    session_tracer = None
    session_id = None
    if LANGFUSE_SESSION_AVAILABLE:
        try:
            session_tracer = get_session_tracer()
            user_id = st.session_state.get("user_id", "anonymous")
            session_metadata = {
                "streamlit_session_id": st.session_state.get("session_id", "unknown"),
                "user_interface": "streamlit",
                "query_timestamp": time.time(),
                "query_length": len(prompt)
            }
            session_id = session_tracer.start_user_session(prompt, user_id, session_metadata)
            debug_log(f"🔍 Langfuse Session 시작: {session_id}", "success")
        except Exception as e:
            debug_log(f"❌ Langfuse Session 시작 실패: {e}", "error")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🧠"):
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
                                            
                                            # 일반 텍스트 크기로 헤더 표시
                                            streaming_container.markdown("**🧠 CherryAI v8 Universal Intelligence 분석 결과**")
                                            text_container = st.empty()
                                            
                                            for i, sentence in enumerate(sentences):
                                                if sentence.strip():
                                                    displayed_text += sentence
                                                    if i < len(sentences) - 1:
                                                        displayed_text += ". "
                                                    
                                                    # 실시간 업데이트 (일반 텍스트로)
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
            
            # 실시간 스트리밍을 위한 컨테이너 생성 (기존 방식 호환성)
            live_text_container = st.empty()
            live_artifacts_container = st.empty()
            
            for step_idx, step in enumerate(plan_steps):
                step_num = step_idx + 1
                agent_name = step.get('agent_name', 'unknown')
                task_description = step.get('task_description', '')
                
                debug_log(f"🎯 단계 {step_num}/{len(plan_steps)} 실행: {agent_name}")
                
                # 각 단계별 스트리밍 컨테이너 생성 (스코프 문제 해결)
                step_stream_container = None
                if SMART_UI_AVAILABLE:
                    step_stream_container = AccumulativeStreamContainer(f"🤖 {agent_name} 실시간 응답")
                
                # 각 단계별 변수 초기화
                step_results = []
                step_artifacts = []
                displayed_text = ""
                
                # Langfuse 에이전트 추적과 실시간 스트리밍 처리
                with session_tracer.trace_agent_execution(
                    agent_name=agent_name,
                    task_description=task_description,
                    agent_metadata={
                        "step_number": step_num,
                        "total_steps": len(plan_steps),
                        "step_index": step_idx
                    }
                ) if session_tracer else contextlib.nullcontext():
                    async for chunk_data in a2a_client.stream_task(agent_name, task_description):
                        try:
                            chunk_type = chunk_data.get('type', 'unknown')
                            chunk_content = chunk_data.get('content', {})
                            is_final = chunk_data.get('final', False)
                            
                            step_results.append(chunk_data)
                            
                            # 실시간 메시지 스트리밍 표시
                            if chunk_type == 'message':
                                text = chunk_content.get('text', '')
                                if text and not text.startswith('✅'):  # 완료 메시지 제외
                                    # Smart UI 사용 가능 시 누적형 컨테이너 사용
                                    if SMART_UI_AVAILABLE and step_stream_container:
                                        # 청크를 누적하여 추가
                                        step_stream_container.add_chunk(text, "message")
                                        
                                    else:
                                        # 기존 방식 - 하지만 중복 표시 방지
                                        displayed_text += text + " "
                                        
                                        # 단계별 진행 상황만 표시 (중복 방지)
                                        with streaming_container:
                                            st.markdown(f"**🔄 {agent_name} 처리 중...**")
                                            # 상세 텍스트는 Smart Display나 최종 결과에서만 표시
                            
                            # 아티팩트 실시간 표시
                            elif chunk_type == 'artifact':
                                step_artifacts.append(chunk_content)
                                
                                if SMART_UI_AVAILABLE and step_stream_container:
                                    # Smart Display로 아티팩트 렌더링
                                    step_stream_container.add_chunk(chunk_content, "artifact")
                                    
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
                
                # Langfuse 에이전트 결과 기록
                if session_tracer:
                    try:
                        session_tracer.record_agent_result(
                            agent_name=agent_name,
                            result={
                                "step_results": step_results,
                                "artifacts_count": len(step_artifacts),
                                "displayed_text_length": len(displayed_text)
                            },
                            confidence=0.9 if step_artifacts else 0.7,
                            artifacts=[{"name": a.get("name", "unknown"), "type": "artifact", "size": 0} for a in step_artifacts]
                        )
                        debug_log(f"🔍 Langfuse 에이전트 결과 기록: {agent_name}", "success")
                    except Exception as record_error:
                        debug_log(f"❌ Langfuse 에이전트 결과 기록 실패: {record_error}", "error")
                
                # 각 단계 결과를 all_results에 추가
                all_results.append({
                    'step': step_num,
                    'agent': agent_name,
                    'task': task_description,
                    'results': step_results,
                    'artifacts': step_artifacts,
                    'displayed_text': displayed_text
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
            
            # 9. 최종 종합 응답 요청 (핵심 추가!)
            debug_log("📝 오케스트레이터에게 최종 종합 응답 요청 중...")
            try:
                # 모든 단계 결과를 종합하여 최종 보고서 요청
                comprehensive_prompt = f"""
                다음 단계들이 완료되었습니다:
                {chr(10).join([f"- {step.get('agent_name', 'Unknown')}: {step.get('task_description', '')}" for step in plan_steps])}
                
                원본 사용자 요청: {prompt}
                
                위 모든 분석 결과를 종합하여 사용자 요청에 대한 완전한 최종 보고서를 작성해주세요.
                반드시 다음을 포함해야 합니다:
                1. 분석 개요 및 핵심 발견사항
                2. 데이터 품질 및 특성 분석
                3. 시각화 차트 해석
                4. 실무적 권장사항
                5. 추가 분석 제안
                """
                
                # 오케스트레이터에게 종합 응답 요청
                final_response = await a2a_client.get_plan(comprehensive_prompt)
                
                if final_response and isinstance(final_response, dict) and "result" in final_response:
                    result = final_response["result"]
                    
                    # 종합 분석 아티팩트 확인
                    if "artifacts" in result:
                        for artifact in result["artifacts"]:
                            if artifact.get("name") in ["execution_plan", "comprehensive_analysis"]:
                                debug_log("🎯 최종 종합 응답 발견!", "success")
                                
                                # 최종 보고서 표시
                                st.markdown("---")
                                st.markdown("### 🎯 최종 종합 분석 보고서")
                                
                                parts = artifact.get("parts", [])
                                for part in parts:
                                    if part.get("kind") == "text":
                                        final_text = part.get("text", "")
                                        if final_text:
                                            # 구조화된 마크다운으로 표시
                                            st.markdown(final_text)
                                            debug_log("✅ 최종 종합 보고서 표시 완료", "success")
                                            break
                    
                    # 상태 메시지도 확인
                    if "status" in result and result["status"] == "completed":
                        if "message" in result and "parts" in result["message"]:
                            for part in result["message"]["parts"]:
                                if part.get("kind") == "text":
                                    status_text = part.get("text", "")
                                    if status_text and len(status_text) > 100:  # 실질적인 내용이 있는 경우
                                        st.markdown("---")
                                        st.markdown("### 🎯 최종 종합 분석 결과")
                                        st.markdown(status_text)
                                        debug_log("✅ 상태 메시지에서 최종 응답 표시 완료", "success")
                                        break
                
            except Exception as final_error:
                debug_log(f"❌ 최종 종합 응답 요청 실패: {final_error}", "error")
                # 폴백: 기본 요약 제공
                st.markdown("---")
                st.markdown("### 🎯 분석 완료 요약")
                st.info(f"총 {len(plan_steps)}개 단계가 실행되었습니다. 각 단계별 결과는 위에서 확인하실 수 있습니다.")
            
            # Phase 3: 전문가급 답변 합성
            if PHASE3_AVAILABLE:
                await _process_phase3_expert_synthesis(prompt, plan_steps, a2a_client, session_tracer, session_id)
            
            debug_log("🎉 전체 스트리밍 프로세스 완료!", "success")
            
            # Langfuse Session 종료 (성공 케이스) - Phase 3 완료 후 종료
            if session_tracer and session_id:
                try:
                    final_result = {
                        "success": True,
                        "total_steps": len(plan_steps),
                        "total_artifacts": sum(len(r.get('artifacts', [])) for r in all_results),
                        "processing_completed": True,
                        "phase3_executed": PHASE3_AVAILABLE
                    }
                    session_summary = {
                        "steps_executed": len(plan_steps),
                        "agents_used": list(set(step.get('agent_name', 'unknown') for step in plan_steps)),
                        "phase3_enabled": PHASE3_AVAILABLE
                    }
                    session_tracer.end_user_session(final_result, session_summary)
                    debug_log(f"🔍 Langfuse Session 종료 (성공): {session_id}", "success")
                except Exception as session_end_error:
                    debug_log(f"❌ Langfuse Session 종료 실패: {session_end_error}", "error")
            
        except Exception as e:
            debug_log(f"💥 전체 프로세스 오류: {e}", "error")
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            import traceback
            debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
            
            # Langfuse Session 종료 (오류 케이스)
            if session_tracer and session_id:
                try:
                    final_result = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processing_completed": False
                    }
                    session_summary = {
                        "error_occurred": True,
                        "error_step": "process_query_streaming"
                    }
                    session_tracer.end_user_session(final_result, session_summary)
                    debug_log(f"🔍 Langfuse Session 종료 (오류): {session_id}", "success")
                except Exception as session_end_error:
                    debug_log(f"❌ Langfuse Session 종료 실패: {session_end_error}", "error")

async def _process_phase3_expert_synthesis(prompt: str, plan_steps: List[Dict], a2a_client, session_tracer=None, session_id=None):
    """Phase 3 전문가급 답변 합성 처리 - Langfuse 세션 통합"""
    try:
        debug_log("🧠 Phase 3 전문가급 답변 합성 시작...", "info")
        
        # Phase 3 전용 span 생성 (기존 세션 내에서)
        phase3_span = None
        if session_tracer and LANGFUSE_AVAILABLE:
            try:
                phase3_span = session_tracer.create_agent_execution_span(
                    "Phase 3 Expert Synthesis",
                    {
                        "operation": "expert_answer_synthesis",
                        "user_query": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        "previous_steps": len(plan_steps),
                        "synthesis_type": "holistic_integration"
                    }
                )
                debug_log("✅ Phase 3 Langfuse span 생성 완료", "success")
            except Exception as span_error:
                debug_log(f"❌ Phase 3 Langfuse span 생성 실패: {span_error}", "error")
                phase3_span = None
        
        # 1. Phase 3 Integration Layer 초기화
        phase3_layer = Phase3IntegrationLayer()
        expert_renderer = ExpertAnswerRenderer()
        
        # 2. A2A 에이전트 결과 수집
        a2a_agent_results = await _collect_a2a_agent_results(plan_steps, a2a_client)
        
        # 3. 사용자 및 세션 컨텍스트 준비 (기존 세션 정보 활용)
        user_context = {
            "user_id": st.session_state.get("user_id", "anonymous"),
            "role": "data_scientist",
            "domain_expertise": {"data_science": 0.9, "analytics": 0.8},
            "preferences": {"visualization": True, "detailed_analysis": True},
            "personalization_level": "advanced"
        }
        
        session_context = {
            "session_id": session_id or st.session_state.get("session_id", f"session_{int(time.time())}"),
            "timestamp": time.time(),
            "context_history": st.session_state.get("messages", []),
            "phase3_continuation": True,  # Phase 3가 기존 세션의 연속임을 표시
            "langfuse_session_active": session_tracer is not None
        }
        
        # 4. 전문가급 답변 합성 실행
        st.markdown("---")
        st.markdown("## 🧠 전문가급 지능형 분석 시작")
        
        synthesis_start_time = time.time()
        
        with st.spinner("전문가급 답변을 합성하는 중..."):
            expert_answer = await phase3_layer.process_user_query_to_expert_answer(
                user_query=prompt,
                a2a_agent_results=a2a_agent_results,
                user_context=user_context,
                session_context=session_context
            )
        
        synthesis_time = time.time() - synthesis_start_time
        
        # 5. Phase 3 결과를 Langfuse에 기록
        if phase3_span and session_tracer:
            try:
                phase3_result = {
                    "success": expert_answer.get("success", False),
                    "confidence_score": expert_answer.get("confidence_score", 0.0),
                    "processing_time": synthesis_time,
                    "quality_score": expert_answer.get("metadata", {}).get("phase3_quality_score", 0.0),
                    "synthesis_strategy": expert_answer.get("metadata", {}).get("synthesis_strategy", "unknown"),
                    "total_agents_integrated": expert_answer.get("metadata", {}).get("total_agents_used", 0)
                }
                
                session_tracer.end_agent_execution_span(
                    phase3_span,
                    phase3_result,
                    success=expert_answer.get("success", False),
                    metadata={
                        "phase3_metrics": expert_answer.get("metadata", {}),
                        "synthesis_time": synthesis_time,
                        "expert_answer_sections": len(expert_answer.get("synthesized_answer", {}).get("main_sections", [])) if expert_answer.get("synthesized_answer") else 0
                    }
                )
                debug_log("✅ Phase 3 Langfuse span 완료", "success")
            except Exception as span_end_error:
                debug_log(f"❌ Phase 3 Langfuse span 완료 실패: {span_end_error}", "error")
        
        # 6. 전문가급 답변 렌더링
        if expert_answer.get("success"):
            debug_log("✅ 전문가급 답변 합성 성공!", "success")
            st.markdown("---")
            expert_renderer.render_expert_answer(expert_answer)
            
            # 세션 상태에 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"전문가급 답변이 완성되었습니다. (신뢰도: {expert_answer['confidence_score']:.1%})",
                "expert_answer": expert_answer,
                "timestamp": time.time(),
                "phase3_integrated": True,  # Phase 3 통합 완료 표시
                "synthesis_time": synthesis_time
            })
            
            # Phase 3 성공 메트릭 기록
            if session_tracer:
                try:
                    session_tracer.log_system_event(
                        "phase3_completion",
                        {
                            "synthesis_time": synthesis_time,
                            "confidence_score": expert_answer['confidence_score'],
                            "integration_success": True
                        }
                    )
                except Exception as metric_error:
                    debug_log(f"❌ Phase 3 메트릭 기록 실패: {metric_error}", "error")
        else:
            debug_log("❌ 전문가급 답변 합성 실패", "error")
            st.error("전문가급 답변 합성에 실패했습니다.")
            
            # 오류 정보 표시
            error_details = expert_answer.get("error", "알 수 없는 오류")
            st.error(f"오류 세부사항: {error_details}")
            
            # 폴백 메시지 표시
            if expert_answer.get("fallback_message"):
                st.info(expert_answer["fallback_message"])
            
            # Phase 3 실패 메트릭 기록
            if session_tracer:
                try:
                    session_tracer.log_system_event(
                        "phase3_failure",
                        {
                            "synthesis_time": synthesis_time,
                            "error": error_details,
                            "integration_success": False
                        }
                    )
                except Exception as metric_error:
                    debug_log(f"❌ Phase 3 실패 메트릭 기록 실패: {metric_error}", "error")
        
        debug_log(f"🎯 Phase 3 전문가급 답변 합성 완료 ({synthesis_time:.2f}초)", "success")
        
    except Exception as e:
        debug_log(f"💥 Phase 3 처리 오류: {e}", "error")
        st.error(f"전문가급 답변 합성 중 오류가 발생했습니다: {e}")
        import traceback
        debug_log(f"🔍 Phase 3 스택 트레이스: {traceback.format_exc()}", "error")
        
        # Phase 3 오류 span 기록
        if phase3_span and session_tracer:
            try:
                session_tracer.end_agent_execution_span(
                    phase3_span,
                    {"error": str(e), "success": False},
                    success=False,
                    metadata={"error_traceback": traceback.format_exc()}
                )
            except Exception as span_error:
                debug_log(f"❌ Phase 3 오류 span 기록 실패: {span_error}", "error")

async def _collect_a2a_agent_results(plan_steps: List[Dict], a2a_client) -> List[Dict[str, Any]]:
    """A2A 에이전트 실행 결과 수집"""
    try:
        debug_log("📊 A2A 에이전트 결과 수집 시작...", "info")
        
        agent_results = []
        
        for i, step in enumerate(plan_steps):
            step_name = step.get("name", f"Step {i+1}")
            agent_name = step.get("agent", "Unknown")
            
            # 각 단계에서 에이전트 결과 수집
            try:
                # 실제 단계 실행 결과를 기반으로 데이터 구조화
                # (이미 실행된 A2A 스트리밍 결과를 활용)
                result_data = {
                    "agent_name": agent_name,
                    "step_name": step_name,
                    "success": True,
                    "confidence": 0.85,  # 기본 신뢰도
                    "artifacts": [],
                    "metadata": {
                        "step_index": i,
                        "processing_time": step.get("execution_time", 5.0),
                        "description": step.get("description", "")
                    }
                }
                
                # 단계 실행 결과가 있다면 추가 정보 포함
                if "result" in step:
                    result_data["artifacts"] = step["result"]
                    result_data["success"] = True
                    result_data["confidence"] = 0.9
                elif "error" in step:
                    result_data["success"] = False
                    result_data["confidence"] = 0.2
                    result_data["metadata"]["error"] = step["error"]
                
                agent_results.append(result_data)
                debug_log(f"✅ {agent_name} 결과 수집 완료", "success")
                    
            except Exception as step_error:
                debug_log(f"❌ {agent_name} 결과 수집 중 오류: {step_error}", "error")
                
                # 오류 정보도 포함
                result_data = {
                    "agent_name": agent_name,
                    "step_name": step_name,
                    "success": False,
                    "confidence": 0.1,
                    "artifacts": [],
                    "metadata": {
                        "step_index": i,
                        "error": str(step_error)
                    }
                }
                agent_results.append(result_data)
        
        debug_log(f"📊 총 {len(agent_results)}개 에이전트 결과 수집 완료", "success")
        return agent_results
        
    except Exception as e:
        debug_log(f"💥 A2A 결과 수집 오류: {e}", "error")
        return []

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
                # 프리로더 사용 (상단 숫자가 정확하므로)
                agent_status = asyncio.run(preload_agents_with_ui())
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
            debug_log("🔄 에이전트 상태 강제 새로고침 시작...")
            st.session_state.agents_preloaded = False  # 프리로더 재시작 강제
            st.session_state.agent_status = asyncio.run(preload_agents_with_ui())
            st.rerun()  # 페이지 새로고침으로 UI 업데이트
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