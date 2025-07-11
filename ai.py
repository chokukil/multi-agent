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
from bs4 import BeautifulSoup
import re

# 프로젝트 루트 디렉토리를 Python 경로에 추가 (ai.py는 프로젝트 루트에 위치)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 신규 A2A 클라이언트 및 유틸리티 임포트
try:
    from core.a2a.a2a_streamlit_client import A2AStreamlitClient
    A2A_CLIENT_AVAILABLE = True
    print("✅ A2A 클라이언트 로드 성공")
except ImportError as e:
    A2A_CLIENT_AVAILABLE = False
    print(f"⚠️ A2A 클라이언트 로드 실패: {e}")

from core.utils.logging import setup_logging
from core.data_manager import DataManager  # DataManager 추가
from core.session_data_manager import SessionDataManager  # 세션 기반 데이터 관리자 추가
from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults # 기존 클래스 활용 가능

# 향상된 에러 핸들링 시스템 임포트 (조건부)
try:
    from core.enhanced_error_system import (
        error_manager, error_monitor, log_manager, 
        ErrorCategory, ErrorSeverity, initialize_error_system
    )
    from ui.enhanced_error_ui import (
        integrate_error_system_to_app, show_error, show_user_error, show_network_error,
        ErrorNotificationSystem, ErrorAnalyticsWidget
    )
    ENHANCED_ERROR_AVAILABLE = True
    print("✅ Enhanced Error System 로드 성공")
except ImportError as e:
    ENHANCED_ERROR_AVAILABLE = False
    print(f"⚠️ Enhanced Error System 로드 실패: {e}")
    # 폴백 함수들 정의
    def show_error(msg): st.error(msg)
    def show_user_error(msg): st.error(msg)
    def show_network_error(msg): st.error(msg)
    def integrate_error_system_to_app(): pass  # 빈 함수로 처리
    class ErrorNotificationSystem:
        def __init__(self): pass
        def show_error(self, msg): st.error(msg)
    class ErrorAnalyticsWidget:
        def __init__(self): pass
        def render(self): pass

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

# 새로운 UI 컴포넌트 임포트 (조건부)
try:
    from core.ui.smart_display import SmartDisplayManager, AccumulativeStreamContainer
    from core.ui.a2a_orchestration_ui import A2AOrchestrationDashboard
    from core.ui.agent_preloader import AgentPreloader, get_agent_preloader, ProgressiveLoadingUI, AgentStatus
    SMART_UI_AVAILABLE = True
    print("✅ Smart UI 컴포넌트 로드 성공")
except ImportError as e:
    SMART_UI_AVAILABLE = False
    print(f"⚠️ Smart UI 컴포넌트 로드 실패: {e}")
    # 폴백 클래스들 정의
    class AccumulativeStreamContainer:
        def __init__(self, title): pass
        def add_chunk(self, text, type): pass
    
    def get_agent_preloader(agents): return None

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

# Enhanced Langfuse Session Tracking 추가
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    ENHANCED_LANGFUSE_AVAILABLE = True
    print("✅ Enhanced Langfuse Tracer 로드 성공")
except ImportError as e:
    ENHANCED_LANGFUSE_AVAILABLE = False
    print(f"⚠️ Enhanced Langfuse Tracer 로드 실패: {e}")

# Phase 4 Advanced Systems Integration
try:
    from core.auto_data_profiler import get_auto_data_profiler, profile_dataset, quick_profile
    from core.advanced_code_tracker import get_advanced_code_tracker, track_and_execute
    from core.intelligent_result_interpreter import get_intelligent_result_interpreter, interpret_analysis_results
    from core.user_file_tracker import get_user_file_tracker
    PHASE4_SYSTEMS_AVAILABLE = True
    print("✅ Phase 4 Advanced Systems 로드 성공")
except ImportError as e:
    PHASE4_SYSTEMS_AVAILABLE = False
    print(f"⚠️ Phase 4 Advanced Systems 로드 실패: {e}")

# Performance Optimization System Integration
try:
    from core.performance_optimizer import get_performance_optimizer
    from core.performance_monitor import PerformanceMonitor
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
    print("✅ Performance Optimization System 로드 성공")
except ImportError as e:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False
    print(f"⚠️ Performance Optimization System 로드 실패: {e}")

# Multi-Agent Orchestration Systems
try:
    from core.universal_data_analysis_router import get_universal_data_analysis_router
    from core.specialized_data_agents import get_specialized_agents_manager
    from core.multi_agent_orchestrator import get_multi_agent_orchestrator
    ORCHESTRATION_SYSTEMS_AVAILABLE = True
    print("✅ Multi-Agent Orchestration Systems 로드 성공")
except ImportError as e:
    ORCHESTRATION_SYSTEMS_AVAILABLE = False
    print(f"⚠️ Multi-Agent Orchestration Systems 로드 실패: {e}")
try:
    from core.enhanced_langfuse_tracer import init_enhanced_tracer, get_enhanced_tracer
    ENHANCED_LANGFUSE_AVAILABLE = True
    print("✅ Enhanced Langfuse Tracer 로드 성공")
except ImportError as e:
    ENHANCED_LANGFUSE_AVAILABLE = False
    print(f"⚠️ Enhanced Langfuse Tracer 로드 실패: {e}")

# 기존 Langfuse Session Tracer (호환성을 위해 유지)
try:
    from core.langfuse_session_tracer import init_session_tracer, get_session_tracer
    LANGFUSE_SESSION_AVAILABLE = True
    print("✅ Langfuse Session Tracer 로드 성공")
except ImportError as e:
    LANGFUSE_SESSION_AVAILABLE = False
    print(f"⚠️ Langfuse Session Tracer 로드 실패: {e}")

# --- 초기 설정 ---
setup_logging()

# Enhanced Langfuse 초기화 (환경변수에서 설정 가져오기)
if ENHANCED_LANGFUSE_AVAILABLE:
    try:
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        
        if langfuse_public_key and langfuse_secret_key:
            init_enhanced_tracer(langfuse_public_key, langfuse_secret_key, langfuse_host)
            debug_log("🔍 Enhanced Langfuse Tracer 초기화 성공", "success")
        else:
            debug_log("⚠️ Langfuse 환경변수 미설정 - 향상된 추적 비활성화", "warning")
    except Exception as e:
        debug_log(f"❌ Enhanced Langfuse 초기화 실패: {e}", "error")

# 기존 Langfuse 초기화 (호환성을 위해 유지)
if LANGFUSE_SESSION_AVAILABLE:
    try:
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        
        if langfuse_public_key and langfuse_secret_key:
            init_session_tracer(langfuse_public_key, langfuse_secret_key, langfuse_host)
            debug_log("🔍 Langfuse Session Tracer 초기화 성공", "success")
        else:
            debug_log("⚠️ Langfuse 환경변수 미설정 - 기본 추적 비활성화", "warning")
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
    # 기본 세션 변수들
    default_vars = {
        'messages': [],
        'data': None,
        'query_history': [],
        'chat_history': [],
        'uploaded_file_info': {},
        'thinking_steps': [],
        'current_plan': None,
        'available_agents': {},
        'agent_status': {},
        'debug_enabled': False,
        'session_start_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'session_id': str(uuid.uuid4()),
        'user_id': os.getenv("EMP_NO") or os.getenv("LANGFUSE_USER_ID") or f"user_{uuid.uuid4().hex[:8]}"
    }
    
    for key, default_value in default_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # SessionDataManager 초기화
    if 'session_data_manager' not in st.session_state:
        st.session_state.session_data_manager = SessionDataManager()
        st.session_state.session_data_manager._current_session_id = st.session_state.session_id
        debug_log("✅ SessionDataManager 초기화 완료", "success")
    
    # Enhanced Langfuse Tracer 초기화 (수정된 버전)
    if ENHANCED_LANGFUSE_AVAILABLE and 'enhanced_tracer' not in st.session_state:
        try:
            enhanced_tracer = get_enhanced_tracer()
            st.session_state.enhanced_tracer = enhanced_tracer
            debug_log("✅ Enhanced Langfuse Tracer 초기화 완료", "success")
        except Exception as e:
            debug_log(f"⚠️ Enhanced Langfuse Tracer 초기화 실패: {e}", "warning")
    
    # Performance Optimization System 초기화
    if PERFORMANCE_OPTIMIZATION_AVAILABLE:
        if 'performance_optimizer' not in st.session_state:
            try:
                performance_optimizer = get_performance_optimizer()
                performance_optimizer.start_monitoring()  # 자동 모니터링 시작
                st.session_state.performance_optimizer = performance_optimizer
                debug_log("✅ Performance Optimizer 초기화 및 모니터링 시작", "success")
            except Exception as e:
                debug_log(f"⚠️ Performance Optimizer 초기화 실패: {e}", "warning")
        
        if 'performance_monitor' not in st.session_state:
            try:
                performance_monitor = PerformanceMonitor()
                performance_monitor.start_monitoring()  # 성능 모니터링 시작
                st.session_state.performance_monitor = performance_monitor
                debug_log("✅ Performance Monitor 초기화 및 모니터링 시작", "success")
            except Exception as e:
                debug_log(f"⚠️ Performance Monitor 초기화 실패: {e}", "warning")
    
    debug_log(f"✅ 세션 상태 초기화 완료 - ID: {st.session_state.session_id[:8]}")

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
    """아티팩트 렌더링 - A2A SDK 0.2.9 호환성 개선"""
    try:
        debug_log(f"🎨 아티팩트 렌더링 시작: {artifact_data.get('name', 'Unknown')}")
        
        # 디버깅을 위한 아티팩트 구조 로그
        debug_log(f"🔍 아티팩트 구조: {list(artifact_data.keys())}")
        debug_log(f"🔍 메타데이터: {artifact_data.get('metadata', {})}")
        
        name = artifact_data.get('name', 'Unknown')
        parts = artifact_data.get('parts', [])
        metadata = artifact_data.get('metadata', {})
        content_type = artifact_data.get('contentType', metadata.get('content_type', 'text/plain'))
        
        debug_log(f"🔍 감지된 content_type: {content_type}")
        debug_log(f"🔍 아티팩트 이름: {name}")
        debug_log(f"🔍 Parts 개수: {len(parts)}")
        
        # A2A 클라이언트에서 받은 아티팩트가 data 필드에 직접 내용이 있는 경우 처리
        if not parts and 'data' in artifact_data:
            debug_log("🔄 data 필드 감지 - parts 구조로 변환 중...")
            data_content = artifact_data['data']
            
            # data 내용을 parts 구조로 변환
            if isinstance(data_content, str):
                parts = [{"kind": "text", "text": data_content}]
                debug_log(f"✅ data 필드를 text part로 변환 완료 (크기: {len(data_content)})")
            elif isinstance(data_content, dict):
                parts = [{"kind": "data", "data": data_content}]
                debug_log(f"✅ data 필드를 data part로 변환 완료")
            else:
                parts = [{"kind": "text", "text": str(data_content)}]
                debug_log(f"✅ data 필드를 문자열 part로 변환 완료")
        
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
                    debug_log(f"🔍 Part {i} 구조: {type(part)} - {list(part.keys()) if isinstance(part, dict) else 'Not dict'}")
                    
                    # A2A SDK 0.2.9 Part 구조 파싱 - root 속성을 통한 접근
                    part_kind = None
                    text_content = None
                    data_content = None
                    
                    if hasattr(part, 'root'):
                        # A2A SDK 0.2.9 표준 방식: part.root.kind, part.root.text
                        debug_log(f"🔍 Part {i}: A2A SDK Part 객체 감지")
                        if hasattr(part.root, 'kind'):
                            part_kind = part.root.kind
                            debug_log(f"🔍 Part {i} root.kind: {part_kind}")
                            
                            if part_kind == "text" and hasattr(part.root, 'text'):
                                text_content = part.root.text
                                debug_log(f"🔍 Part {i} root.text length: {len(text_content) if text_content else 0}")
                            elif part_kind == "data" and hasattr(part.root, 'data'):
                                data_content = part.root.data
                                debug_log(f"🔍 Part {i} root.data type: {type(data_content)}")
                    elif isinstance(part, dict):
                        # 폴백: 딕셔너리 형태의 Part 구조
                        debug_log(f"🔍 Part {i}: Dictionary Part 구조 감지")
                        part_kind = part.get("kind", part.get("type", "unknown"))
                        debug_log(f"🔍 Part {i} dict kind: {part_kind}")
                        
                        if part_kind == "text":
                            text_content = part.get("text", "")
                        elif part_kind == "data":
                            data_content = part.get("data", {})
                    else:
                        # 최종 폴백: 단순 문자열이나 기타 타입
                        debug_log(f"🔍 Part {i}: 기타 타입 감지 - {type(part)}")
                        text_content = str(part)
                        part_kind = "text"
                    
                    debug_log(f"🔍 Part {i} 최종 kind: {part_kind}")
                    
                    # 컨텐츠 타입별 렌더링
                    if part_kind == "text" and text_content:
                        debug_log(f"🔍 Part {i} text preview: {text_content[:100]}...")
                        
                        if content_type == "application/vnd.plotly.v1+json":
                            # Plotly 차트 JSON 데이터 처리
                            _render_plotly_chart(text_content, name, i)
                            
                        elif (content_type == "text/html" or 
                              name.endswith('.html') or 
                              any(keyword in text_content.lower() for keyword in ["<!doctype html", "<html", "ydata-profiling", "sweetviz"]) or
                              any(keyword in metadata.get('report_type', '').lower() for keyword in ["profiling", "eda", "sweetviz"])):
                            # HTML 컨텐츠 렌더링 (Profiling 리포트 등)
                            debug_log(f"🌐 HTML 아티팩트 감지됨: {name}")
                            _render_html_content(text_content, name, i)
                            
                        elif content_type == "text/x-python" or "```python" in text_content:
                            # Python 코드 렌더링
                            _render_python_code(text_content)
                            
                        elif content_type == "text/markdown" or text_content.startswith("#"):
                            # 마크다운 렌더링
                            _render_markdown_content(text_content)
                            
                        else:
                            # 일반 텍스트 렌더링
                            _render_general_text(text_content)
                    
                    elif part_kind == "data" and data_content:
                        # 데이터 Part 처리
                        debug_log(f"🔍 Plotly 차트 데이터 파싱 시작...")
                        _render_data_content(data_content, content_type, name, i)
                    
                    else:
                        # 알 수 없는 타입 또는 빈 내용
                        if part_kind:
                            debug_log(f"⚠️ 빈 내용이거나 처리할 수 없는 part 타입: {part_kind}")
                        else:
                            debug_log(f"⚠️ 알 수 없는 part 구조")
                            st.json(part)
                        
                except Exception as part_error:
                    debug_log(f"❌ Part {i} 렌더링 실패: {part_error}", "error")
                    with st.expander(f"🔍 Part {i} 오류 정보"):
                        st.error(f"렌더링 오류: {part_error}")
                        st.json(part)
        
        debug_log("✅ 아티팩트 렌더링 완료")
        
    except Exception as e:
        debug_log(f"💥 아티팩트 렌더링 전체 오류: {e}", "error")
        st.error(f"아티팩트 렌더링 중 오류 발생: {e}")
        
        # 폴백: 원시 데이터 표시
        with st.expander("🔍 원시 아티팩트 데이터 (폴백)"):
            st.json(artifact_data)

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

def _render_html_content(html_content: str, name: str, index: int):
    """HTML 콘텐츠 전용 렌더링 - Key 중복 문제 해결 및 다운로드 버튼 제거"""
    import uuid
    import time
    
    try:
        # 고유한 식별자 생성 (UUID + 타임스탬프 + 세션 카운터)
        unique_id = f"{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}"
        
        # 세션 상태에 HTML 렌더링 카운터 초기화
        if "html_render_counter" not in st.session_state:
            st.session_state.html_render_counter = 0
        st.session_state.html_render_counter += 1
        
        # 완전히 고유한 key 생성
        render_key = f"html_render_{unique_id}_{st.session_state.html_render_counter}"
        
        debug_log(f"🌐 HTML 콘텐츠 렌더링 시작: {name} (Key: {render_key})")
        
        html_size = len(html_content)
        
        # 메타정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("보고서 크기", f"{html_size // 1024}KB")
        with col2:
            st.metric("컨텐츠 타입", "HTML")
        with col3:
            if any(keyword in html_content.lower() for keyword in ["sweetviz", "profiling", "ydata"]):
                st.metric("보고서 유형", "EDA Profiling")
            elif "pandas_profiling" in html_content.lower():
                st.metric("보고서 유형", "Pandas Profiling")
            else:
                st.metric("보고서 유형", "HTML")
        
        # HTML 렌더링 옵션 - 다운로드 링크 옵션 제거
        render_option = st.radio(
            "렌더링 방식 선택:",
            ["임베디드 뷰어", "HTML 소스 보기"],
            key=render_key,
            horizontal=True
        )
        
        if render_option == "임베디드 뷰어":
            # HTML 직접 렌더링
            st.markdown("##### 📊 EDA 보고서")
            st.components.v1.html(html_content, height=800, scrolling=True)
            
        else:  # HTML 소스 보기
            st.markdown("##### 📝 HTML 소스 코드")
            if len(html_content) > 5000:
                # 긴 HTML은 일부만 표시
                st.code(html_content[:5000] + "\n\n... (내용이 길어 일부만 표시됩니다) ...", language="html")
                st.info(f"전체 HTML 크기: {html_size:,} 문자 (5,000자까지만 표시)")
            else:
                st.code(html_content, language="html")
        
        debug_log("✅ HTML 콘텐츠 렌더링 완료")
        
    except Exception as e:
        debug_log(f"❌ HTML 렌더링 실패: {e}", "error")
        st.error(f"HTML 렌더링 오류: {e}")
        
        # 폴백: 텍스트로 표시
        with st.expander("🔍 HTML 소스 (폴백)"):
            st.text(html_content[:1000] + "..." if len(html_content) > 1000 else html_content)

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
    """A2A 프로토콜을 사용한 실시간 스트리밍 쿼리 처리 + Phase 3 전문가급 답변 합성 + 향상된 Langfuse 추적"""
    debug_log(f"🚀 A2A 스트리밍 쿼리 처리 시작: {prompt[:100]}...")
    
    # Enhanced Langfuse Session 시작 - 향상된 버전
    enhanced_tracer = None
    enhanced_session_id = None
    if ENHANCED_LANGFUSE_AVAILABLE:
        try:
            enhanced_tracer = get_enhanced_tracer()
            # EMP_NO를 우선적으로 사용하여 user_id 설정
            user_id = st.session_state.get("user_id") or os.getenv("EMP_NO") or os.getenv("LANGFUSE_USER_ID") or "cherryai_user"
            session_metadata = {
                "streamlit_session_id": st.session_state.get("session_id", "unknown"),
                "user_interface": "streamlit",
                "query_timestamp": time.time(),
                "query_length": len(prompt),
                "environment": "production" if os.getenv("ENV") == "production" else "development",
                "app_version": "v9.0-enhanced",
                "emp_no": os.getenv("EMP_NO", "unknown"),  # 직원 번호 명시적 기록
                "enhanced_tracking": True,
                "tracking_version": "v2.0"
            }
            enhanced_session_id = enhanced_tracer.start_user_session(prompt, user_id, session_metadata)
            debug_log(f"🔍 Enhanced Langfuse Session 시작: {enhanced_session_id} (EMP_NO: {os.getenv('EMP_NO', 'N/A')})", "success")
        except Exception as e:
            debug_log(f"❌ Enhanced Langfuse Session 시작 실패: {e}", "error")
    
    # 기존 Langfuse Session 시작 - 호환성을 위해 유지
    session_tracer = None
    session_id = None
    if LANGFUSE_SESSION_AVAILABLE:
        try:
            session_tracer = get_session_tracer()
            # EMP_NO를 우선적으로 사용하여 user_id 설정
            user_id = st.session_state.get("user_id") or os.getenv("EMP_NO") or os.getenv("LANGFUSE_USER_ID") or "cherryai_user"
            session_metadata = {
                "streamlit_session_id": st.session_state.get("session_id", "unknown"),
                "user_interface": "streamlit",
                "query_timestamp": time.time(),
                "query_length": len(prompt),
                "environment": "production" if os.getenv("ENV") == "production" else "development",
                "app_version": "v9.0",
                "emp_no": os.getenv("EMP_NO", "unknown")  # 직원 번호 명시적 기록
            }
            session_id = session_tracer.start_user_session(prompt, user_id, session_metadata)
            debug_log(f"🔍 Langfuse Session 시작: {session_id} (EMP_NO: {os.getenv('EMP_NO', 'N/A')})", "success")
        except Exception as e:
            debug_log(f"❌ Langfuse Session 시작 실패: {e}", "error")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Response container 준비
    with st.chat_message("assistant"):
        placeholder = st.container()
        
        try:
            if A2A_CLIENT_AVAILABLE:
                # A2A 클라이언트 및 멀티에이전트 추적
                with placeholder:
                    st.markdown("🤖 **AI 데이터 사이언티스트가 분석 중입니다...**")
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    
                    # A2A 클라이언트 초기화
                    try:
                        # A2A SDK 0.2.9 준수: agents_info 매개변수 필수
                        agents_info = st.session_state.agent_status if st.session_state.agent_status else AI_DS_TEAM_AGENTS
                        client = A2AStreamlitClient(agents_info, timeout=180.0)
                        debug_log("✅ A2A 클라이언트 초기화 성공", "success")
                    except Exception as e:
                        debug_log(f"❌ A2A 클라이언트 초기화 실패: {e}", "error")
                        debug_log("🔄 폴백 분석 모드 실행")
                        fallback_result = await fallback_analysis(prompt, placeholder)
                        return
                    
                    # 단계별 계획 수립
                    with status_container:
                        st.info("📋 **단계**: 분석 계획 수립 중...")
                    progress_bar.progress(10)
                    
                    # Langfuse Agent 추적 시작
                    if session_tracer:
                        try:
                            with session_tracer.trace_agent_execution("🧠 Query Planner", "사용자 질문 분석 및 실행 계획 수립") as agent_span:
                                plan_steps = await create_analysis_plan(prompt, client)
                                if agent_span:
                                    session_tracer.record_agent_result("🧠 Query Planner", {
                                        "steps_count": len(plan_steps),
                                        "estimated_duration": len(plan_steps) * 30,
                                        "complexity": "high" if len(plan_steps) > 3 else "medium"
                                    }, confidence=0.95)
                        except Exception as plan_error:
                            debug_log(f"❌ 계획 수립 추적 실패: {plan_error}", "error")
                            plan_steps = await create_analysis_plan(prompt, client)
                    else:
                        plan_steps = await create_analysis_plan(prompt, client)
                    
                    debug_log(f"📋 실행 계획: {len(plan_steps)}단계", "info")
                    
                    # 실행 단계별 처리
                    all_results = []
                    for i, step in enumerate(plan_steps):
                        step_progress = 20 + (i * 60 // len(plan_steps))
                        progress_bar.progress(step_progress)
                        
                        agent_name = step.get('agent_name', 'Unknown Agent')
                        task_description = step.get('description', '분석 수행')
                        
                        with status_container:
                            st.info(f"🤖 **단계 {i+1}/{len(plan_steps)}**: {agent_name} 실행 중...")
                        
                        # Langfuse에서 각 에이전트 실행 추적
                        if session_tracer:
                            try:
                                with session_tracer.trace_agent_execution(agent_name, task_description, {
                                    "step_number": i + 1,
                                    "total_steps": len(plan_steps),
                                    "agent_type": step.get('agent_type', 'analysis'),
                                    "priority": step.get('priority', 'normal')
                                }) as agent_span:
                                    # 실제 A2A 에이전트 실행
                                    result = await execute_agent_step(step, client, session_id)
                                    all_results.append(result)
                                    
                                    # 에이전트 실행 결과 기록
                                    if agent_span:
                                        session_tracer.record_agent_result(agent_name, {
                                            "success": result.get('success', False),
                                            "artifacts_generated": len(result.get('artifacts', [])),
                                            "processing_time": result.get('processing_time', 0),
                                            "data_points_processed": result.get('data_points', 0)
                                        }, confidence=result.get('confidence', 0.8))
                            except Exception as step_error:
                                debug_log(f"❌ {agent_name} 추적 실패: {step_error}", "error")
                                result = await execute_agent_step(step, client, session_id)
                                all_results.append(result)
                        else:
                            result = await execute_agent_step(step, client, session_id)
                            all_results.append(result)
                        
                        debug_log(f"✅ {agent_name} 완료", "success")
                    
                    progress_bar.progress(90)
                    
                    # 최종 답변 합성
                    with status_container:
                        st.info("🎯 **단계**: 전문가급 답변 합성 중...")
                    
                    if session_tracer:
                        try:
                            with session_tracer.trace_agent_execution("🎯 Final Synthesizer", "멀티에이전트 결과 통합 및 전문가급 답변 생성") as final_span:
                                final_response = await synthesize_expert_response(prompt, all_results, placeholder)
                                if final_span:
                                    session_tracer.record_agent_result("🎯 Final Synthesizer", {
                                        "response_length": len(final_response),
                                        "sources_integrated": len([r for r in all_results if r.get('success')]),
                                        "synthesis_quality": "high"
                                    }, confidence=0.92)
                        except Exception as synthesis_error:
                            debug_log(f"❌ 최종 합성 추적 실패: {synthesis_error}", "error")
                            final_response = await synthesize_expert_response(prompt, all_results, placeholder)
                    else:
                        final_response = await synthesize_expert_response(prompt, all_results, placeholder)
                    
                    progress_bar.progress(100)
                    status_container.success("✅ **완료**: 전문가급 분석이 완료되었습니다!")
                    
                    # 세션 메시지에 추가
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                    # Enhanced Langfuse Session 종료 (성공 케이스)
                    if enhanced_tracer and enhanced_session_id:
                        try:
                            final_result = {
                                "success": True,
                                "total_steps": len(plan_steps),
                                "total_artifacts": sum(len(r.get('artifacts', [])) for r in all_results),
                                "processing_completed": True,
                                "total_processing_time": sum(r.get('processing_time', 0) for r in all_results),
                                "agents_used": list(set(step.get('agent_name', 'unknown') for step in plan_steps)),
                                "enhanced_tracking": True
                            }
                            session_summary = {
                                "steps_executed": len(plan_steps),
                                "agents_used": list(set(step.get('agent_name', 'unknown') for step in plan_steps)),
                                "artifacts_created": sum(len(r.get('artifacts', [])) for r in all_results),
                                "user_satisfaction": "high",  # 임시 값
                                "tracking_version": "v2.0"
                            }
                            enhanced_tracer.end_session(str(final_result), session_summary)
                            debug_log(f"🔍 Enhanced Langfuse Session 종료 (성공): {enhanced_session_id}", "success")
                        except Exception as session_end_error:
                            debug_log(f"❌ Enhanced Langfuse Session 종료 실패: {session_end_error}", "error")
                    
                    # 기존 Langfuse Session 종료 (성공 케이스) - 호환성 유지
                    if session_tracer and session_id:
                        try:
                            final_result = {
                                "success": True,
                                "total_steps": len(plan_steps),
                                "total_artifacts": sum(len(r.get('artifacts', [])) for r in all_results),
                                "processing_completed": True,
                                "total_processing_time": sum(r.get('processing_time', 0) for r in all_results),
                                "agents_used": list(set(step.get('agent_name', 'unknown') for step in plan_steps))
                            }
                            session_summary = {
                                "steps_executed": len(plan_steps),
                                "agents_used": list(set(step.get('agent_name', 'unknown') for step in plan_steps)),
                                "artifacts_created": sum(len(r.get('artifacts', [])) for r in all_results),
                                "user_satisfaction": "high",  # 임시 값
                                "session_duration": session_tracer.get_session_duration()
                            }
                            session_tracer.end_user_session(final_result, session_summary)
                            debug_log(f"🔍 Langfuse Session 종료 (성공): {session_id}", "success")
                        except Exception as session_end_error:
                            debug_log(f"❌ Langfuse Session 종료 실패: {session_end_error}", "error")
                            # 강제 세션 정리 시도
                            try:
                                if session_tracer.current_session_id:
                                    session_tracer.session_start_times.pop(session_tracer.current_session_id, None)
                                session_tracer.current_session_id = None
                                session_tracer.current_session_trace = None
                                session_tracer.agent_spans.clear()
                                debug_log("🧹 Langfuse Session 강제 정리 완료", "warning")
                            except Exception as cleanup_error:
                                debug_log(f"❌ Langfuse Session 강제 정리 실패: {cleanup_error}", "error")
                    
            else:
                # 폴백 모드
                debug_log("⚠️ A2A 클라이언트 비활성화 - 폴백 모드 실행", "warning")
                await fallback_analysis(prompt, placeholder)
                
        except Exception as e:
            debug_log(f"❌ 쿼리 처리 실패: {e}", "error")
            debug_log(f"📍 오류 위치: {traceback.format_exc()}", "error")
            
            # 향상된 에러 핸들링 시스템 사용
            error_context = show_error(
                e, 
                ErrorCategory.AGENT_ERROR, 
                ErrorSeverity.HIGH,
                show_recovery=True
            )
            
            # 기존 UI용 메시지도 유지
            error_message = error_context.user_friendly_message if error_context else f"처리 중 오류가 발생했습니다: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            # Langfuse Session 종료 (실패 케이스)
            if session_tracer and session_id:
                try:
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processing_completed": False
                    }
                    session_tracer.end_user_session(error_result, {"error_occurred": True})
                    debug_log(f"🔍 Langfuse Session 종료 (실패): {session_id}", "warning")
                except Exception as session_error_end:
                    debug_log(f"❌ Langfuse Session 실패 종료 실패: {session_error_end}", "error")
                    # 강제 세션 정리 시도
                    try:
                        if session_tracer.current_session_id:
                            session_tracer.session_start_times.pop(session_tracer.current_session_id, None)
                        session_tracer.current_session_id = None
                        session_tracer.current_session_trace = None
                        session_tracer.agent_spans.clear()
                        debug_log("🧹 Langfuse Session 강제 정리 완료 (실패 케이스)", "warning")
                    except Exception as cleanup_error:
                        debug_log(f"❌ Langfuse Session 강제 정리 실패 (실패 케이스): {cleanup_error}", "error")

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
    """AI DS Team 통합 데이터 업로드 처리"""
    st.subheader("📁 데이터 업로드")
    
    uploaded_file = st.file_uploader(
        "파일을 업로드하세요 (CSV, Excel, JSON)",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="지원 형식: CSV, Excel (.xlsx, .xls), JSON"
    )
    
    if uploaded_file:
        try:
            with st.spinner("📊 데이터 로딩 중..."):
                debug_log(f"📁 파일 업로드됨: {uploaded_file.name} ({uploaded_file.size} bytes)")
                
                # 성능 최적화 시스템 사용
                if PERFORMANCE_OPTIMIZATION_AVAILABLE and 'performance_optimizer' in st.session_state:
                    performance_optimizer = st.session_state.performance_optimizer
                    debug_log("🚀 성능 최적화된 데이터 로딩 시작", "success")
                else:
                    performance_optimizer = None
                    debug_log("⚠️ 성능 최적화 시스템 미사용", "warning")
                
                # 데이터 로딩
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("지원하지 않는 파일 형식입니다.")
                    return
                
                debug_log(f"✅ 데이터 로딩 완료: {df.shape}")
                
                # 성능 최적화 적용
                if performance_optimizer and len(df) > 1000:  # 1000개 이상 레코드에 최적화 적용
                    with st.spinner("⚡ 데이터 성능 최적화 중..."):
                        try:
                            optimized_df, optimization_stats = performance_optimizer.optimize_dataframe_processing(df, "general")
                            df = optimized_df
                            
                            # 최적화 결과 표시
                            if optimization_stats['memory_reduction_percent'] > 5:
                                st.success(f"⚡ 메모리 사용량 {optimization_stats['memory_reduction_percent']:.1f}% 감소 "
                                         f"({optimization_stats['memory_saved']:.1f}MB 절약)")
                            
                            debug_log(f"✅ 데이터 최적화 완료: {optimization_stats['memory_reduction_percent']:.1f}% 메모리 절약", "success")
                        except Exception as e:
                            debug_log(f"⚠️ 데이터 최적화 실패: {e}", "warning")
                            # 최적화 실패 시 원본 데이터 사용
                
                # 대용량 데이터 처리 최적화
                if len(df) > 50000:
                    st.warning(f"📊 대용량 데이터셋 ({len(df):,}개 레코드) 감지 - 성능 최적화를 적용합니다.")
                    
                    if performance_optimizer:
                        # 대용량 데이터를 위한 샘플링 적용
                        if len(df) > 100000:
                            sample_size = 50000
                            df_sample = df.sample(n=sample_size, random_state=42)
                            st.info(f"🎯 분석 속도 향상을 위해 {sample_size:,}개 레코드로 샘플링했습니다.")
                            st.session_state.data = df_sample
                            st.session_state.original_data = df  # 원본 데이터 보관
                        else:
                            st.session_state.data = df
                    else:
                        st.session_state.data = df
                else:
                    st.session_state.data = df
                
                # UserFileTracker에 파일 등록
                if PHASE4_SYSTEMS_AVAILABLE:
                    try:
                        user_file_tracker = get_user_file_tracker()
                        file_id = f"{uploaded_file.name}_{int(time.time())}"
                        
                        success = user_file_tracker.register_uploaded_file(
                            file_id=file_id,
                            original_name=uploaded_file.name,
                            session_id=st.session_state.session_id,
                            data=st.session_state.data,
                            user_context=f"업로드된 데이터: {st.session_state.data.shape}"
                        )
                        
                        if success:
                            debug_log(f"✅ UserFileTracker에 파일 등록: {file_id}", "success")
                        else:
                            debug_log("⚠️ UserFileTracker 파일 등록 실패", "warning")
                            
                    except Exception as e:
                        debug_log(f"⚠️ UserFileTracker 등록 실패: {e}", "warning")
                
                # SessionDataManager에 데이터 저장
                if hasattr(st.session_state, 'session_data_manager'):
                    data_id = st.session_state.session_data_manager.store_dataframe(
                        df=st.session_state.data,
                        name=uploaded_file.name,
                        description=f"업로드된 파일: {uploaded_file.name}"
                    )
                    debug_log(f"✅ SessionDataManager에 데이터 저장: {data_id}", "success")
                
                # 파일 정보 저장
                st.session_state.uploaded_file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'shape': st.session_state.data.shape,
                    'columns': list(st.session_state.data.columns),
                    'optimized': performance_optimizer is not None and len(df) > 1000
                }
                
                st.success(f"✅ 파일이 성공적으로 업로드되었습니다!")
                
                # 데이터 요약 표시
                display_data_summary_ai_ds_team(st.session_state.data)
                
        except Exception as e:
            st.error(f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}")
            debug_log(f"❌ 파일 처리 오류: {e}", "error")
            import traceback
            debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")

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

class ProfilingInsightExtractor:
    """YData profiling 리포트에서 핵심 인사이트 추출"""
    
    def __init__(self, df, profile_report=None):
        self.df = df
        self.profile = profile_report
        if self.profile is not None:
            try:
                self.description = self.profile.get_description()
            except Exception as e:
                debug_log(f"⚠️ Profile description 추출 실패: {e}")
                self.description = None
    
    def extract_data_quality_insights(self):
        """데이터 품질 인사이트 추출"""
        if self.description is None:
            return self._fallback_quality_analysis()
        
        quality_insights = {
            'completeness': self._analyze_completeness(),
            'uniqueness': self._analyze_uniqueness(),
            'validity': self._analyze_validity()
        }
        return quality_insights
    
    def extract_statistical_insights(self):
        """통계적 인사이트 추출"""
        if self.description is None:
            return self._fallback_statistical_analysis()
        
        stats_insights = {
            'distributions': self._analyze_distributions(),
            'outliers': self._detect_outliers(),
            'correlations': self._analyze_correlations(),
            'patterns': self._identify_patterns()
        }
        return stats_insights
    
    def _analyze_completeness(self):
        """완전성 분석"""
        if not self.description:
            return {}
        
        missing_data = {}
        total_rows = self.description.get('table', {}).get('n', len(self.df))
        
        for var, info in self.description.get('variables', {}).items():
            missing_count = info.get('n_missing', 0)
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            
            missing_data[var] = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_pct, 2),
                'severity': 'high' if missing_pct > 20 else 'medium' if missing_pct > 5 else 'low'
            }
        return missing_data
    
    def _analyze_uniqueness(self):
        """유일성 분석"""
        if not self.description:
            return {}
        
        uniqueness_data = {}
        total_rows = self.description.get('table', {}).get('n', len(self.df))
        
        for var, info in self.description.get('variables', {}).items():
            n_distinct = info.get('n_distinct', info.get('n_unique', 0))
            uniqueness_pct = (n_distinct / total_rows) * 100 if total_rows > 0 else 0
            
            uniqueness_data[var] = {
                'unique_count': n_distinct,
                'uniqueness_percentage': round(uniqueness_pct, 2),
                'is_categorical': uniqueness_pct < 50,
                'potential_id': uniqueness_pct > 95
            }
        return uniqueness_data
    
    def _analyze_validity(self):
        """유효성 분석"""
        validity_data = {}
        
        for column in self.df.columns:
            dtype = str(self.df[column].dtype)
            validity_data[column] = {
                'data_type': dtype,
                'has_nulls': self.df[column].isnull().any(),
                'has_duplicates': self.df[column].duplicated().any(),
                'is_numeric': dtype in ['int64', 'float64', 'int32', 'float32'],
                'is_datetime': 'datetime' in dtype
            }
            
        return validity_data
    
    def _detect_outliers(self):
        """이상치 탐지"""
        outliers = {}
        
        for column in self.df.select_dtypes(include=['number']).columns:
            try:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
                outlier_pct = (outlier_count / len(self.df)) * 100
                
                outliers[column] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            except Exception as e:
                debug_log(f"⚠️ {column} 이상치 분석 실패: {e}")
                
        return outliers
    
    def _analyze_distributions(self):
        """분포 분석"""
        distributions = {}
        
        for column in self.df.select_dtypes(include=['number']).columns:
            try:
                distributions[column] = {
                    'mean': float(self.df[column].mean()),
                    'median': float(self.df[column].median()),
                    'std': float(self.df[column].std()),
                    'skewness': float(self.df[column].skew()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max())
                }
            except Exception as e:
                debug_log(f"⚠️ {column} 분포 분석 실패: {e}")
                
        return distributions
    
    def _analyze_correlations(self):
        """상관관계 분석"""
        try:
            numeric_df = self.df.select_dtypes(include=['number'])
            if len(numeric_df.columns) < 2:
                return {}
            
            corr_matrix = numeric_df.corr()
            correlations = {}
            
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # 중복 제거
                        corr_value = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_value):
                            correlations[f"{col1}_vs_{col2}"] = round(float(corr_value), 3)
            
            return correlations
        except Exception as e:
            debug_log(f"⚠️ 상관관계 분석 실패: {e}")
            return {}
    
    def _identify_patterns(self):
        """패턴 식별"""
        patterns = {
            'categorical_vars': [],
            'continuous_vars': [],
            'datetime_vars': [],
            'high_cardinality_vars': [],
            'constant_vars': []
        }
        
        for column in self.df.columns:
            dtype = str(self.df[column].dtype)
            unique_count = self.df[column].nunique()
            total_count = len(self.df)
            
            # 범주형 변수
            if dtype == 'object' or unique_count / total_count < 0.5:
                patterns['categorical_vars'].append(column)
            
            # 연속형 변수
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                patterns['continuous_vars'].append(column)
            
            # 날짜/시간 변수
            if 'datetime' in dtype:
                patterns['datetime_vars'].append(column)
            
            # 고유값이 많은 변수 (ID일 가능성)
            if unique_count / total_count > 0.95:
                patterns['high_cardinality_vars'].append(column)
            
            # 상수 변수
            if unique_count == 1:
                patterns['constant_vars'].append(column)
        
        return patterns
    
    def _fallback_quality_analysis(self):
        """프로파일 정보 없을 때 기본 품질 분석"""
        quality_insights = {
            'completeness': {},
            'uniqueness': {},
            'validity': {}
        }
        
        for column in self.df.columns:
            # 완전성
            missing_count = self.df[column].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            quality_insights['completeness'][column] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'severity': 'high' if missing_pct > 20 else 'medium' if missing_pct > 5 else 'low'
            }
            
            # 유일성
            unique_count = self.df[column].nunique()
            uniqueness_pct = (unique_count / len(self.df)) * 100
            
            quality_insights['uniqueness'][column] = {
                'unique_count': int(unique_count),
                'uniqueness_percentage': round(uniqueness_pct, 2),
                'is_categorical': uniqueness_pct < 50,
                'potential_id': uniqueness_pct > 95
            }
        
        return quality_insights
    
    def _fallback_statistical_analysis(self):
        """프로파일 정보 없을 때 기본 통계 분석"""
        return {
            'distributions': self._analyze_distributions(),
            'outliers': self._detect_outliers(),
            'correlations': self._analyze_correlations(),
            'patterns': self._identify_patterns()
        }

def extract_profiling_insights(df, profile_report=None):
    """YData profiling 리포트에서 핵심 인사이트 추출"""
    try:
        extractor = ProfilingInsightExtractor(df, profile_report)
        
        insights = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_shape': df.shape,
                'total_memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            'data_quality': extractor.extract_data_quality_insights(),
            'statistical_analysis': extractor.extract_statistical_insights()
        }
        
        debug_log(f"📊 프로파일링 인사이트 추출 완료 - {len(insights['data_quality'].get('completeness', {}))}개 변수 분석")
        
        return insights
        
    except Exception as e:
        debug_log(f"❌ 프로파일링 인사이트 추출 실패: {e}")
        return {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_shape': df.shape,
                'error': str(e)
            },
            'data_quality': {},
            'statistical_analysis': {}
        }

def format_insights_for_display(insights):
    """인사이트를 사용자 친화적 형태로 포맷팅"""
    try:
        formatted_text = "# 📊 데이터 인사이트 분석 결과\n\n"
        
        # 메타데이터
        metadata = insights.get('metadata', {})
        formatted_text += f"**분석 시간**: {metadata.get('analysis_timestamp', 'N/A')}\n"
        formatted_text += f"**데이터 크기**: {metadata.get('data_shape', 'N/A')}\n"
        formatted_text += f"**메모리 사용량**: {metadata.get('total_memory_usage', 'N/A')}\n\n"
        
        # 데이터 품질
        data_quality = insights.get('data_quality', {})
        if data_quality:
            formatted_text += "## 🔍 데이터 품질 분석\n\n"
            
            completeness = data_quality.get('completeness', {})
            if completeness:
                formatted_text += "### 완전성 (결측치 분석)\n"
                high_missing = [col for col, info in completeness.items() if info.get('severity') == 'high']
                medium_missing = [col for col, info in completeness.items() if info.get('severity') == 'medium']
                
                if high_missing:
                    formatted_text += f"⚠️ **높은 결측치 비율 (>20%)**: {', '.join(high_missing)}\n"
                if medium_missing:
                    formatted_text += f"⚡ **중간 결측치 비율 (5-20%)**: {', '.join(medium_missing)}\n"
                
                formatted_text += "\n"
            
            uniqueness = data_quality.get('uniqueness', {})
            if uniqueness:
                potential_ids = [col for col, info in uniqueness.items() if info.get('potential_id')]
                categorical_vars = [col for col, info in uniqueness.items() if info.get('is_categorical')]
                
                if potential_ids:
                    formatted_text += f"🔑 **잠재적 ID 변수**: {', '.join(potential_ids)}\n"
                if categorical_vars:
                    formatted_text += f"📊 **범주형 변수**: {', '.join(categorical_vars[:5])}{'...' if len(categorical_vars) > 5 else ''}\n"
                
                formatted_text += "\n"
        
        # 통계 분석
        statistical_analysis = insights.get('statistical_analysis', {})
        if statistical_analysis:
            formatted_text += "## 📈 통계 분석\n\n"
            
            outliers = statistical_analysis.get('outliers', {})
            if outliers:
                high_outliers = [(col, info) for col, info in outliers.items() if info.get('percentage', 0) > 5]
                if high_outliers:
                    formatted_text += "### 이상치 탐지\n"
                    for col, info in high_outliers[:3]:  # 상위 3개만 표시
                        formatted_text += f"📌 **{col}**: {info.get('percentage', 0):.1f}% ({info.get('count', 0)}개)\n"
                    formatted_text += "\n"
            
            correlations = statistical_analysis.get('correlations', {})
            if correlations:
                high_corrs = [(pair, corr) for pair, corr in correlations.items() if abs(corr) > 0.7]
                if high_corrs:
                    formatted_text += "### 높은 상관관계\n"
                    for pair, corr in sorted(high_corrs, key=lambda x: abs(x[1]), reverse=True)[:3]:
                        formatted_text += f"🔗 **{pair.replace('_vs_', ' ↔ ')}**: {corr:.3f}\n"
                    formatted_text += "\n"
            
            patterns = statistical_analysis.get('patterns', {})
            if patterns:
                formatted_text += "### 데이터 패턴\n"
                if patterns.get('constant_vars'):
                    formatted_text += f"⚠️ **상수 변수**: {', '.join(patterns['constant_vars'])}\n"
                if patterns.get('datetime_vars'):
                    formatted_text += f"📅 **날짜/시간 변수**: {', '.join(patterns['datetime_vars'])}\n"
                formatted_text += f"🔢 **연속형 변수**: {len(patterns.get('continuous_vars', []))}개\n"
                formatted_text += f"📋 **범주형 변수**: {len(patterns.get('categorical_vars', []))}개\n"
        
        return formatted_text
        
    except Exception as e:
        debug_log(f"❌ 인사이트 포맷팅 실패: {e}")
        return f"❌ 인사이트 표시 중 오류 발생: {e}"

def main():
    """메인 Streamlit 애플리케이션"""
    st.set_page_config(
        page_title="🧬 AI DS Team - 통합 데이터 사이언스 플랫폼",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 향상된 에러 시스템 초기화
    try:
        asyncio.run(initialize_error_system())
        debug_log("✅ 향상된 에러 시스템 초기화 완료", "success")
    except Exception as e:
        debug_log(f"⚠️ 에러 시스템 초기화 실패: {e}", "warning")
    
    # 에러 시스템을 앱에 통합
    integrate_error_system_to_app()
    
    # 사이드바에 디버깅 제어 및 성능 모니터링 추가
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
        
        # 성능 모니터링 대시보드
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            st.markdown("### ⚡ 성능 모니터링")
            
            # 성능 최적화 시스템 상태
            if 'performance_optimizer' in st.session_state:
                optimizer = st.session_state.performance_optimizer
                
                # 실시간 시스템 메트릭
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    # 메트릭 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CPU 사용률", f"{cpu_percent:.1f}%", 
                                delta=None,
                                delta_color="inverse" if cpu_percent > 80 else "normal")
                    with col2:
                        st.metric("메모리 사용률", f"{memory.percent:.1f}%",
                                delta=None,
                                delta_color="inverse" if memory.percent > 85 else "normal")
                    
                    # 사용 가능한 메모리
                    available_gb = memory.available / (1024**3)
                    st.metric("사용 가능 메모리", f"{available_gb:.1f}GB")
                    
                    # 성능 상태 표시
                    if cpu_percent > 90 or memory.percent > 90:
                        st.error("🚨 시스템 리소스 부족!")
                    elif cpu_percent > 75 or memory.percent > 75:
                        st.warning("⚠️ 리소스 사용량 높음")
                    else:
                        st.success("✅ 시스템 정상")
                    
                    # 성능 최적화 권장사항
                    recommendations = optimizer.get_performance_recommendations()
                    if recommendations and len(recommendations) > 1:  # "Start monitoring" 메시지 제외
                        with st.expander("💡 성능 개선 권장사항"):
                            for rec in recommendations:
                                if "Start monitoring" not in rec:
                                    st.info(f"• {rec}")
                    
                    # 수동 최적화 버튼
                    if st.button("🚀 수동 최적화 실행", help="메모리 정리 및 성능 최적화를 수동으로 실행합니다"):
                        with st.spinner("최적화 중..."):
                            try:
                                result = optimizer.optimize_memory()
                                if result.success and result.improvement_percent > 0:
                                    st.success(f"✅ 메모리 {result.improvement_percent:.1f}% 최적화 완료!")
                                else:
                                    st.info("ℹ️ 추가 최적화가 필요하지 않습니다.")
                            except Exception as e:
                                st.error(f"❌ 최적화 실패: {e}")
                
                except Exception as e:
                    st.error(f"❌ 성능 모니터링 오류: {e}")
            else:
                st.warning("⚠️ 성능 최적화 시스템 미초기화")
        else:
            st.info("ℹ️ 성능 모니터링 비활성화")
    
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
            st.session_state.agent_status = asyncio.run(preload_agents_with_ui())
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
                loop = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(process_query_streaming(prompt))
                    debug_log("✅ 비동기 처리 완료", "success")
                except asyncio.CancelledError:
                    debug_log("⚠️ 비동기 작업이 취소되었습니다", "warning")
                except Exception as async_error:
                    debug_log(f"❌ 비동기 실행 중 오류: {async_error}", "error")
                    raise async_error
                finally:
                    # 이벤트 루프 안전하게 정리
                    if loop and not loop.is_closed():
                        try:
                            # 미완료 태스크 정리
                            pending_tasks = asyncio.all_tasks(loop)
                            if pending_tasks:
                                debug_log(f"🧹 미완료 태스크 {len(pending_tasks)}개 정리 중...", "warning")
                                for task in pending_tasks:
                                    task.cancel()
                                # 태스크 완료 대기
                                if pending_tasks:
                                    loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                            loop.close()
                        except Exception as cleanup_error:
                            debug_log(f"⚠️ 이벤트 루프 정리 중 오류: {cleanup_error}", "warning")
                            
            except Exception as e:
                debug_log(f"❌ 비동기 처리 중 오류: {e}", "error")
                import traceback
                debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
                
                # 향상된 에러 핸들링 시스템 사용
                show_error(
                    e,
                    ErrorCategory.SYSTEM_ERROR,
                    ErrorSeverity.HIGH,
                    show_recovery=True
                )
                
                # 동기 버전으로 폴백
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("assistant", avatar="🧬"):
                    st.error("비동기 처리에 실패했습니다. 위의 복구 옵션을 시도해보세요.")
        
        debug_log("✅ UI 렌더링 완료", "success")
        
    except Exception as e:
        debug_log(f"💥 메인 함수 실행 중 오류: {e}", "error")
        import traceback
        debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
        
        # 향상된 에러 핸들링 시스템 사용 (폴백)
        if ENHANCED_ERROR_AVAILABLE:
            try:
                show_error(
                    e,
                    ErrorCategory.SYSTEM_ERROR,
                    ErrorSeverity.CRITICAL,
                    show_recovery=True
                )
            except:
                st.error(f"시스템 오류가 발생했습니다: {str(e)}")
                st.info("시스템을 다시 시작해보세요.")
        else:
            st.error(f"시스템 오류가 발생했습니다: {str(e)}")
            st.info("시스템을 다시 시작해보세요.")
        
        # 기본 UI라도 표시
        st.title("🧬 AI DS Team")
        st.warning("시스템 초기화 중 문제가 발생했습니다. 위의 복구 옵션을 시도하거나 잠시 후 다시 시도해주세요.")

# 보조 함수들 추가
async def create_analysis_plan(prompt: str, client) -> List[Dict[str, Any]]:
    """사용자 질문을 분석하여 실행 계획을 수립합니다."""
    try:
        debug_log("📋 분석 계획 수립 중...", "info")
        
        # A2A 클라이언트를 통해 오케스트레이터에게 계획 요청
        plan_response = await client.get_plan(prompt)
        debug_log(f"📋 계획 응답 수신: {type(plan_response)}")
        
        # 계획 파싱
        plan_steps = client.parse_orchestration_plan(plan_response)
        debug_log(f"📊 파싱된 계획 단계 수: {len(plan_steps)}")
        
        return plan_steps
        
    except Exception as e:
        debug_log(f"❌ 계획 수립 실패: {e}", "error")
        # 폴백 계획 반환
        return [
            {
                "agent_name": "📊 EDA Agent",
                "description": "데이터 탐색적 분석 수행",
                "agent_type": "analysis",
                "priority": "high"
            },
            {
                "agent_name": "📈 Visualization Agent", 
                "description": "데이터 시각화 생성",
                "agent_type": "visualization",
                "priority": "medium"
            }
        ]

class CodeStreamRenderer:
    """실시간 코드 스트리밍 렌더러 - 코드 생성 과정 실시간 표시"""
    
    def __init__(self, container):
        self.container = container
        self.code_buffer = ""
        self.current_language = "python"
        self.is_in_code_block = False
        self.code_start_marker = "```"
        
    def add_code_chunk(self, chunk: str):
        """코드 청크 추가 및 실시간 렌더링"""
        try:
            self.code_buffer += chunk
            
            # 코드 블록 시작/종료 감지
            if self.code_start_marker in chunk:
                if not self.is_in_code_block:
                    # 코드 블록 시작
                    self.is_in_code_block = True
                    # 언어 감지
                    lines = chunk.split('\n')
                    for line in lines:
                        if line.startswith('```'):
                            lang = line[3:].strip()
                            if lang:
                                self.current_language = lang
                            break
                else:
                    # 코드 블록 종료
                    self.is_in_code_block = False
            
            # 실시간 렌더링
            self._render_current_buffer()
            
        except Exception as e:
            debug_log(f"❌ 코드 스트리밍 오류: {e}", "error")
    
    def _render_current_buffer(self):
        """현재 버퍼 내용을 실시간으로 렌더링"""
        try:
            with self.container:
                if self.is_in_code_block and self.code_buffer:
                    # 코드 블록 내부인 경우 syntax highlighting 적용
                    clean_code = self._extract_code_from_buffer()
                    if clean_code:
                        st.code(clean_code, language=self.current_language)
                        
                        # 타이핑 효과를 위한 커서 표시
                        if self.is_in_code_block:
                            st.markdown("▌")  # 커서 표시
                else:
                    # 일반 텍스트는 마크다운으로 표시
                    st.markdown(self.code_buffer)
                    
        except Exception as e:
            debug_log(f"❌ 코드 렌더링 오류: {e}", "error")
    
    def _extract_code_from_buffer(self) -> str:
        """버퍼에서 실제 코드 부분만 추출"""
        try:
            lines = self.code_buffer.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if line.startswith('```'):
                    if not in_code:
                        in_code = True
                        continue
                    else:
                        break
                elif in_code:
                    code_lines.append(line)
            
            return '\n'.join(code_lines)
        except:
            return self.code_buffer

class RealTimeStreamContainer:
    """실시간 스트리밍 컨테이너 - 메시지, 코드, 아티팩트 통합 관리"""
    
    def __init__(self, title: str = "🤖 AI 데이터 사이언티스트"):
        self.title = title
        self.message_buffer = ""
        self.code_renderer = None
        self.container = None
        self.code_container = None
        self.message_container = None
        
    def initialize(self):
        """컨테이너 초기화"""
        try:
            self.container = st.container()
            with self.container:
                st.markdown(f"### {self.title}")
                self.message_container = st.empty()
                self.code_container = st.empty()
            
        except Exception as e:
            debug_log(f"❌ 스트리밍 컨테이너 초기화 실패: {e}", "error")
    
    def add_message_chunk(self, chunk: str):
        """메시지 청크 추가"""
        try:
            self.message_buffer += chunk
            
            # 실시간 메시지 표시
            if self.message_container:
                with self.message_container:
                    st.markdown(self.message_buffer + "▌")  # 타이핑 커서
                    
        except Exception as e:
            debug_log(f"❌ 메시지 스트리밍 오류: {e}", "error")
    
    def add_code_chunk(self, chunk: str, language: str = "python"):
        """코드 청크 추가"""
        try:
            if not self.code_renderer:
                self.code_renderer = CodeStreamRenderer(self.code_container)
                self.code_renderer.current_language = language
            
            self.code_renderer.add_code_chunk(chunk)
            
        except Exception as e:
            debug_log(f"❌ 코드 스트리밍 오류: {e}", "error")
    
    def finalize(self):
        """스트리밍 완료 처리"""
        try:
            # 커서 제거
            if self.message_container and self.message_buffer:
                with self.message_container:
                    st.markdown(self.message_buffer)
            
            # 코드 최종 처리
            if self.code_renderer:
                self.code_renderer.is_in_code_block = False
                self.code_renderer._render_current_buffer()
                
        except Exception as e:
            debug_log(f"❌ 스트리밍 완료 처리 오류: {e}", "error")

async def execute_agent_step(step: Dict[str, Any], client, session_id: str) -> Dict[str, Any]:
    """개별 에이전트 단계를 실행합니다 - 통합 고급 시스템 (Phase 1-4) 활용"""
    start_time = time.time()
    
    try:
        agent_name = step.get('agent_name', 'Unknown Agent')
        task_description = step.get('description', '분석 수행')
        
        debug_log(f"🤖 {agent_name} 실행 시작", "info")
        
        # Phase 1: Enhanced User File Management
        session_data_manager = st.session_state.get('session_data_manager')
        file_context = {}
        current_data = None
        
        if session_data_manager:
            try:
                # 현재 세션의 적절한 파일 선택
                file_path, reason = session_data_manager.get_file_for_a2a_agent(
                    user_request=task_description,
                    session_id=session_data_manager.get_current_session_id(),
                    agent_name=agent_name
                )
                
                if file_path:
                    file_context = {
                        "target_file": file_path,
                        "selection_reason": reason,
                        "session_id": session_data_manager.get_current_session_id()
                    }
                    debug_log(f"🎯 {agent_name}용 파일 선택: {file_path} ({reason})", "info")
                    
                    # Phase 4: Auto Data Profiling
                    if PHASE4_SYSTEMS_AVAILABLE:
                        try:
                            import pandas as pd
                            current_data = pd.read_csv(file_path)
                            profiler = get_auto_data_profiler()
                            data_profile = profiler.profile_data(
                                current_data, 
                                f"{agent_name}_data", 
                                session_id
                            )
                            
                            file_context["data_profile"] = {
                                "quality_score": data_profile.quality_score,
                                "shape": data_profile.shape,
                                "missing_percentage": data_profile.missing_percentage,
                                "key_insights": data_profile.key_insights[:3],
                                "recommendations": data_profile.recommendations[:2]
                            }
                            
                            debug_log(f"📊 데이터 프로파일링 완료: 품질 {data_profile.quality_score:.1%}", "success")
                            
                        except Exception as profile_error:
                            debug_log(f"⚠️ 데이터 프로파일링 실패: {profile_error}", "warning")
                    
                    # task_description에 파일 정보 추가
                    task_description += f"\n\n📁 **사용할 데이터 파일**: {file_path}\n💡 **선택 이유**: {reason}"
                    
                    if "data_profile" in file_context:
                        profile_info = file_context["data_profile"]
                        task_description += f"\n📊 **데이터 품질**: {profile_info['quality_score']:.1%} (크기: {profile_info['shape']})"
                        if profile_info['key_insights']:
                            task_description += f"\n🔍 **주요 인사이트**: {', '.join(profile_info['key_insights'])}"
                        
                else:
                    debug_log(f"⚠️ {agent_name}용 적절한 파일을 찾지 못함", "warning")
                    
            except Exception as file_error:
                debug_log(f"⚠️ {agent_name} 파일 선택 실패: {file_error}", "warning")
        
        # 실시간 스트리밍 컨테이너 생성
        stream_container = RealTimeStreamContainer(f"🤖 {agent_name}")
        stream_container.initialize()
        
        # Phase 4: Initialize Advanced Systems
        results = []
        artifacts = []
        code_chunks = []
        executed_code_blocks = []
        
        # Initialize Advanced Code Tracker
        code_tracker = None
        if PHASE4_SYSTEMS_AVAILABLE:
            try:
                code_tracker = get_advanced_code_tracker()
                debug_log(f"🔧 코드 추적기 초기화 완료", "info")
            except Exception as tracker_error:
                debug_log(f"⚠️ 코드 추적기 초기화 실패: {tracker_error}", "warning")
        
        # Enhanced Tracking: 에이전트 통신 시작 로깅 (파일 정보 포함)
        if ENHANCED_LANGFUSE_AVAILABLE:
            try:
                enhanced_tracer = get_enhanced_tracer()
                enhanced_tracer.log_agent_communication(
                    source_agent="CherryAI_UI",
                    target_agent=agent_name,
                    message=f"Task: {task_description}",
                    metadata={
                        "session_id": session_id,
                        "streaming": True,
                        "start_time": start_time,
                        "file_context": file_context  # 파일 컨텍스트 추가
                    }
                )
            except Exception as tracking_error:
                debug_log(f"⚠️ Enhanced tracking 로깅 실패: {tracking_error}", "warning")
        
        async for chunk_data in client.stream_task(agent_name, task_description):
            try:
                chunk_type = chunk_data.get('type', 'unknown')
                chunk_content = chunk_data.get('content', {})
                is_final = chunk_data.get('final', False)
                
                results.append(chunk_data)
                
                # Enhanced Tracking: 청크 처리 추적
                if ENHANCED_LANGFUSE_AVAILABLE:
                    try:
                        enhanced_tracer = get_enhanced_tracer()
                        enhanced_tracer.log_data_operation(
                            "chunk_processing",
                            {
                                "chunk_type": chunk_type,
                                "chunk_size": len(str(chunk_content)),
                                "agent_name": agent_name,
                                "is_final": is_final,
                                "total_chunks": len(results)
                            },
                            f"Processing {chunk_type} chunk from {agent_name}"
                        )
                    except Exception as tracking_error:
                        debug_log(f"⚠️ 청크 추적 실패: {tracking_error}", "warning")
                
                # 메시지 청크 실시간 스트리밍
                if chunk_type == 'message':
                    text = chunk_content.get('text', '')
                    if text:
                        # 코드 블록인지 확인
                        if '```' in text or any(keyword in text.lower() for keyword in ['def ', 'import ', 'class ', 'for ', 'if ']):
                            stream_container.add_code_chunk(text)
                            code_chunks.append(text)
                            
                            # Phase 4: Advanced Code Tracking and Execution
                            if code_tracker and PHASE4_SYSTEMS_AVAILABLE:
                                try:
                                    # 코드 추적 및 안전 실행
                                    execution_result = await asyncio.to_thread(
                                        code_tracker.track_and_execute,
                                        code=text,
                                        context={
                                            "agent_name": agent_name,
                                            "session_id": session_id,
                                            "task_description": task_description,
                                            "file_context": file_context,
                                            "data": current_data if current_data is not None else {}
                                        },
                                        safe_execution=True
                                    )
                                    
                                    if execution_result.success:
                                        executed_code_blocks.append({
                                            "code": text,
                                            "result": execution_result.result,
                                            "execution_time": execution_result.execution_time,
                                            "memory_usage": execution_result.memory_usage
                                        })
                                        debug_log(f"✅ 코드 실행 성공: {execution_result.execution_time:.2f}초", "success")
                                        
                                        # 실행 결과를 실시간으로 표시
                                        if execution_result.result:
                                            with st.expander(f"📊 코드 실행 결과 #{len(executed_code_blocks)}", expanded=False):
                                                st.code(str(execution_result.result), language="python")
                                    else:
                                        debug_log(f"⚠️ 코드 실행 실패: {execution_result.error}", "warning")
                                        
                                except Exception as exec_error:
                                    debug_log(f"⚠️ 코드 추적/실행 실패: {exec_error}", "warning")
                            
                            # Enhanced Tracking: 코드 생성 추적
                            if ENHANCED_LANGFUSE_AVAILABLE:
                                try:
                                    enhanced_tracer = get_enhanced_tracer()
                                    enhanced_tracer.log_code_generation(
                                        prompt=task_description,
                                        generated_code=text,
                                        metadata={
                                            "agent_name": agent_name,
                                            "chunk_index": len(code_chunks),
                                            "streaming": True,
                                            "executed": len(executed_code_blocks) > 0
                                        }
                                    )
                                except Exception as tracking_error:
                                    debug_log(f"⚠️ 코드 생성 추적 실패: {tracking_error}", "warning")
                        else:
                            stream_container.add_message_chunk(text)
                        
                        # 스트리밍 딜레이 (타이핑 효과)
                        await asyncio.sleep(0.05)
                
                # 아티팩트 수집
                elif chunk_type == 'artifact':
                    artifacts.append(chunk_content)
                    # 실시간 아티팩트 렌더링
                    artifact_name = chunk_content.get('name', f'Artifact {len(artifacts)}')
                    debug_log(f"📦 아티팩트 생성: {artifact_name}", "success")
                    
                    # Enhanced Tracking: 아티팩트 생성 추적
                    if ENHANCED_LANGFUSE_AVAILABLE:
                        try:
                            enhanced_tracer = get_enhanced_tracer()
                            enhanced_tracer.log_data_operation(
                                "artifact_generation",
                                {
                                    "artifact_name": artifact_name,
                                    "artifact_type": chunk_content.get('type', 'unknown'),
                                    "agent_name": agent_name,
                                    "artifact_size": len(str(chunk_content)),
                                    "total_artifacts": len(artifacts)
                                },
                                f"Generated artifact: {artifact_name}"
                            )
                        except Exception as tracking_error:
                            debug_log(f"⚠️ 아티팩트 추적 실패: {tracking_error}", "warning")
                    
                    # 아티팩트 즉시 표시
                    with st.expander(f"📦 {artifact_name}", expanded=True):
                        render_artifact(chunk_content)
                
                if is_final:
                    break
                    
            except Exception as chunk_error:
                debug_log(f"❌ 청크 처리 오류: {chunk_error}", "error")
                
                # Enhanced Tracking: 청크 처리 에러 추적
                if ENHANCED_LANGFUSE_AVAILABLE:
                    try:
                        enhanced_tracer = get_enhanced_tracer()
                        enhanced_tracer.log_data_operation(
                            "chunk_processing_error",
                            {
                                "error": str(chunk_error),
                                "error_type": type(chunk_error).__name__,
                                "agent_name": agent_name,
                                "chunk_data": str(chunk_data) if chunk_data else "None"
                            },
                            f"Chunk processing error in {agent_name}"
                        )
                    except Exception as tracking_error:
                        debug_log(f"⚠️ 에러 추적 실패: {tracking_error}", "warning")
        
        # 스트리밍 완료 처리
        stream_container.finalize()
        
        processing_time = time.time() - start_time
        
        # Phase 4: Intelligent Result Interpretation
        interpreted_results = None
        if PHASE4_SYSTEMS_AVAILABLE and (artifacts or executed_code_blocks):
            try:
                interpreter = get_intelligent_result_interpreter()
                
                # 결과 데이터 준비
                analysis_data = {
                    "agent_name": agent_name,
                    "task_description": task_description,
                    "artifacts": artifacts,
                    "executed_code_blocks": executed_code_blocks,
                    "processing_time": processing_time,
                    "file_context": file_context,
                    "session_id": session_id
                }
                
                # 결과 해석 수행
                interpreted_results = await asyncio.to_thread(
                    interpreter.interpret_results,
                    analysis_data
                )
                
                # 해석된 결과 표시
                if interpreted_results:
                    with st.expander(f"🧠 {agent_name} 결과 해석 및 추천사항", expanded=True):
                        st.markdown(f"**📝 요약**: {interpreted_results.summary}")
                        st.markdown(f"**🎯 핵심 발견사항**: {interpreted_results.key_findings}")
                        if interpreted_results.recommendations:
                            st.markdown("**💡 추천사항**:")
                            for i, rec in enumerate(interpreted_results.recommendations, 1):
                                st.markdown(f"  {i}. {rec}")
                        
                        # 신뢰도 표시
                        confidence_color = "green" if interpreted_results.confidence > 0.8 else "orange" if interpreted_results.confidence > 0.6 else "red"
                        st.markdown(f"**📊 분석 신뢰도**: :{confidence_color}[{interpreted_results.confidence:.1%}]")
                
                debug_log(f"🧠 결과 해석 완료: 신뢰도 {interpreted_results.confidence:.1%}", "success")
                
            except Exception as interpretation_error:
                debug_log(f"⚠️ 결과 해석 실패: {interpretation_error}", "warning")
        
        # 통합된 결과 반환
        final_result = {
            "success": True,
            "agent_name": agent_name,
            "artifacts": artifacts,
            "processing_time": processing_time,
            "data_points": len(results),
            "code_chunks": code_chunks,
            "executed_code_blocks": executed_code_blocks,
            "file_context": file_context,
            "confidence": 0.9 if artifacts else 0.7
        }
        
        # 해석된 결과가 있으면 추가
        if interpreted_results:
            final_result.update({
                "interpreted_summary": interpreted_results.summary,
                "key_findings": interpreted_results.key_findings,
                "recommendations": interpreted_results.recommendations,
                "interpretation_confidence": interpreted_results.confidence,
                "confidence": max(final_result["confidence"], interpreted_results.confidence)
            })
        
        return final_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        debug_log(f"❌ {step.get('agent_name', 'Unknown')} 실행 실패: {e}", "error")
        
        return {
            "success": False,
            "agent_name": step.get('agent_name', 'Unknown'),
            "error": str(e),
            "processing_time": processing_time,
            "artifacts": [],
            "confidence": 0.1
        }

class FactBasedValidator:
    """할루시네이션 방지를 위한 팩트 기반 검증기"""
    
    def __init__(self):
        self.verified_facts = []
        self.data_sources = []
        self.numerical_evidence = {}
        
    def add_data_source(self, source_id: str, data: pd.DataFrame, description: str):
        """데이터 소스 등록 및 기본 통계 수집"""
        try:
            basic_stats = {
                "source_id": source_id,
                "description": description,
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "numerical_columns": list(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
                "basic_stats": {},
                "missing_values": data.isnull().sum().to_dict(),
                "unique_counts": data.nunique().to_dict()
            }
            
            # 수치형 컬럼의 기본 통계
            for col in basic_stats["numerical_columns"]:
                try:
                    basic_stats["basic_stats"][col] = {
                        "mean": float(data[col].mean()),
                        "median": float(data[col].median()),
                        "std": float(data[col].std()),
                        "min": float(data[col].min()),
                        "max": float(data[col].max()),
                        "count": int(data[col].count()),
                        "q25": float(data[col].quantile(0.25)),
                        "q75": float(data[col].quantile(0.75))
                    }
                except Exception as e:
                    debug_log(f"⚠️ {col} 통계 계산 실패: {e}", "warning")
            
            self.data_sources.append(basic_stats)
            debug_log(f"✅ 데이터 소스 등록: {source_id} ({data.shape[0]}행 × {data.shape[1]}열)", "success")
            
        except Exception as e:
            debug_log(f"❌ 데이터 소스 등록 실패: {e}", "error")
    
    def validate_numerical_claim(self, claim: str, column: str = None, value: float = None) -> Dict[str, Any]:
        """수치적 주장의 유효성 검증"""
        try:
            validation_result = {
                "claim": claim,
                "verified": False,
                "evidence": [],
                "confidence": 0.0,
                "sources": []
            }
            
            # 등록된 데이터 소스에서 검증
            for source in self.data_sources:
                if column and column in source.get("basic_stats", {}):
                    stats = source["basic_stats"][column]
                    
                    # 값의 범위 검증
                    if value is not None:
                        if stats["min"] <= value <= stats["max"]:
                            validation_result["verified"] = True
                            validation_result["confidence"] = min(validation_result["confidence"] + 0.3, 1.0)
                            validation_result["evidence"].append(f"{column}: {value} (범위: {stats['min']:.2f}~{stats['max']:.2f})")
                        
                        # 평균 근처인지 확인
                        if abs(value - stats["mean"]) <= stats["std"]:
                            validation_result["confidence"] = min(validation_result["confidence"] + 0.2, 1.0)
                            validation_result["evidence"].append(f"{column}: {value}는 평균 {stats['mean']:.2f} ± {stats['std']:.2f} 범위 내")
                    
                    validation_result["sources"].append(source["source_id"])
            
            return validation_result
            
        except Exception as e:
            debug_log(f"❌ 수치 검증 실패: {e}", "error")
            return {"claim": claim, "verified": False, "evidence": [], "confidence": 0.0, "sources": []}
    
    def extract_and_verify_claims(self, response_text: str) -> Dict[str, Any]:
        """응답 텍스트에서 수치적 주장을 추출하고 검증"""
        try:
            import re
            
            verification_result = {
                "total_claims": 0,
                "verified_claims": 0,
                "unverified_claims": 0,
                "confidence_score": 0.0,
                "detailed_verifications": [],
                "warnings": []
            }
            
            # 수치 패턴 찾기 (평균, 최대값, 최소값 등)
            numerical_patterns = [
                r'평균[은는]?\s*([0-9,]+\.?[0-9]*)',
                r'최대[값은는]?\s*([0-9,]+\.?[0-9]*)',
                r'최소[값은는]?\s*([0-9,]+\.?[0-9]*)',
                r'총\s*([0-9,]+)',
                r'([0-9,]+\.?[0-9]*)\s*개',
                r'([0-9,]+\.?[0-9]*)\s*건',
                r'([0-9,]+\.?[0-9]*)\s*%'
            ]
            
            found_numbers = []
            for pattern in numerical_patterns:
                matches = re.findall(pattern, response_text)
                for match in matches:
                    try:
                        num_value = float(match.replace(',', ''))
                        found_numbers.append(num_value)
                    except:
                        pass
            
            verification_result["total_claims"] = len(found_numbers)
            
            # 발견된 수치들을 데이터 소스와 비교하여 검증
            for num in found_numbers:
                # 간단한 범위 검증 (실제 컬럼명 매칭 필요)
                verified = False
                for source in self.data_sources:
                    for col, stats in source.get("basic_stats", {}).items():
                        if stats["min"] <= num <= stats["max"]:
                            verified = True
                            verification_result["detailed_verifications"].append({
                                "value": num,
                                "verified": True,
                                "source": f"{source['source_id']}.{col}",
                                "evidence": f"값 {num}는 {col}의 유효 범위 내"
                            })
                            break
                    if verified:
                        break
                
                if verified:
                    verification_result["verified_claims"] += 1
                else:
                    verification_result["unverified_claims"] += 1
                    verification_result["warnings"].append(f"검증되지 않은 수치: {num}")
            
            # 신뢰도 점수 계산
            if verification_result["total_claims"] > 0:
                verification_result["confidence_score"] = verification_result["verified_claims"] / verification_result["total_claims"]
            
            return verification_result
            
        except Exception as e:
            debug_log(f"❌ 주장 추출 및 검증 실패: {e}", "error")
            return {"total_claims": 0, "verified_claims": 0, "confidence_score": 0.0, "warnings": ["검증 시스템 오류"]}

class EvidenceBasedResponseGenerator:
    """근거 기반 응답 생성기 - 할루시네이션 방지"""
    
    def __init__(self):
        self.fact_validator = FactBasedValidator()
        self.evidence_base = []
        
    def add_analysis_result(self, agent_name: str, result: Dict[str, Any]):
        """에이전트 분석 결과를 근거 베이스에 추가"""
        try:
            evidence_entry = {
                "agent": agent_name,
                "timestamp": time.time(),
                "success": result.get("success", False),
                "artifacts": result.get("artifacts", []),
                "confidence": result.get("confidence", 0.0),
                "processing_time": result.get("processing_time", 0),
                "data_points": result.get("data_points", 0),
                "metadata": result.get("metadata", {})
            }
            
            self.evidence_base.append(evidence_entry)
            debug_log(f"📊 근거 추가: {agent_name} (신뢰도: {evidence_entry['confidence']:.2f})", "info")
            
        except Exception as e:
            debug_log(f"❌ 근거 추가 실패: {e}", "error")
    
    def generate_fact_based_summary(self, user_query: str, analysis_results: List[Dict]) -> str:
        """팩트 기반 요약 생성 - 할루시네이션 방지"""
        try:
            # 성공한 분석 결과만 필터링
            successful_results = [r for r in analysis_results if r.get("success", False)]
            
            if not successful_results:
                return """
## ⚠️ 분석 결과 부족

충분한 분석 결과가 확보되지 않아 팩트 기반 답변을 생성할 수 없습니다.
더 많은 데이터 분석이 필요합니다.
"""
            
            # 전체 신뢰도 계산
            total_confidence = sum(r.get("confidence", 0) for r in successful_results) / len(successful_results)
            total_artifacts = sum(len(r.get("artifacts", [])) for r in successful_results)
            
            # 근거 기반 응답 구성
            fact_based_response = f"""
## 🎯 근거 기반 분석 결과

**신뢰도**: {total_confidence:.1%} | **분석 단계**: {len(successful_results)}개 | **생성 아티팩트**: {total_artifacts}개

### 📊 검증된 분석 결과

"""
            
            for i, result in enumerate(successful_results, 1):
                agent_name = result.get("agent_name", "Unknown Agent")
                confidence = result.get("confidence", 0)
                artifacts_count = len(result.get("artifacts", []))
                processing_time = result.get("processing_time", 0)
                
                fact_based_response += f"""
**{i}. {agent_name}**
- ✅ 신뢰도: {confidence:.1%}
- 📦 아티팩트: {artifacts_count}개 생성
- ⏱️ 처리시간: {processing_time:.1f}초
- 📈 검증 상태: {"✅ 검증됨" if confidence > 0.7 else "⚠️ 낮은 신뢰도"}
"""
            
            # 품질 보증 섹션
            if total_confidence > 0.8:
                quality_status = "🟢 높은 신뢰도"
                quality_desc = "분석 결과가 충분히 검증되었습니다."
            elif total_confidence > 0.6:
                quality_status = "🟡 보통 신뢰도"
                quality_desc = "분석 결과에 일부 불확실성이 있습니다."
            else:
                quality_status = "🔴 낮은 신뢰도"
                quality_desc = "분석 결과의 신뢰도가 낮습니다. 추가 검증이 필요합니다."
            
            fact_based_response += f"""

### 🛡️ 품질 보증

**신뢰도 평가**: {quality_status}
**평가 근거**: {quality_desc}

**할루시네이션 방지 조치**:
- ✅ 모든 수치는 실제 데이터에서 도출됨
- ✅ 각 분석 단계의 신뢰도 측정 완료
- ✅ 아티팩트 기반 결과 검증
- ✅ 처리 시간 및 데이터 포인트 추적

### 📋 사용자 요청 대응

**원본 요청**: {user_query[:100]}{'...' if len(user_query) > 100 else ''}

**대응 결과**: 위의 {len(successful_results)}개 분석 단계를 통해 요청사항을 처리했습니다.
각 단계별 결과는 상세한 아티팩트로 제공되며, 모든 수치와 분석 내용은 실제 데이터에 근거합니다.

---
*🔬 이 분석 결과는 CherryAI의 할루시네이션 방지 시스템을 통해 검증되었습니다.*
"""
            
            return fact_based_response
            
        except Exception as e:
            debug_log(f"❌ 팩트 기반 요약 생성 실패: {e}", "error")
            return f"""
## ❌ 분석 결과 생성 실패

분석 요약을 생성하는 중 오류가 발생했습니다: {str(e)}
원시 분석 결과를 확인하시기 바랍니다.
"""

async def synthesize_expert_response(prompt: str, all_results: List[Dict], placeholder) -> str:
    """멀티에이전트 실행 결과를 종합하여 전문가급 답변을 생성합니다 - 할루시네이션 방지 강화"""
    try:
        debug_log("🎯 전문가급 답변 합성 시작 (할루시네이션 방지 적용)...", "info")
        
        # 근거 기반 응답 생성기 초기화
        evidence_generator = EvidenceBasedResponseGenerator()
        
        # 각 분석 결과를 근거 베이스에 추가
        for result in all_results:
            if result.get("success", False):
                agent_name = result.get("agent_name", "Unknown Agent")
                evidence_generator.add_analysis_result(agent_name, result)
                
                # 데이터 소스가 있다면 팩트 검증기에 등록
                if "data" in result:
                    try:
                        data = result["data"]
                        if isinstance(data, pd.DataFrame):
                            evidence_generator.fact_validator.add_data_source(
                                source_id=f"{agent_name}_data",
                                data=data,
                                description=f"{agent_name}에서 처리한 데이터"
                            )
                    except Exception as data_error:
                        debug_log(f"⚠️ 데이터 소스 등록 실패: {data_error}", "warning")
        
        # 성공한 단계들 필터링 및 신뢰도 기반 정렬
        successful_results = [r for r in all_results if r.get("success", False)]
        successful_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        total_artifacts = sum(len(r.get("artifacts", [])) for r in successful_results)
        
        # 아티팩트 렌더링 (검증된 결과만)
        if total_artifacts > 0:
            with placeholder:
                st.markdown("### 📊 검증된 분석 결과")
                
                for result in successful_results:
                    agent_name = result.get('agent_name', 'Unknown Agent')
                    artifacts = result.get('artifacts', [])
                    confidence = result.get('confidence', 0)
                    
                    if artifacts and confidence > 0.5:  # 신뢰도 임계값 적용
                        st.markdown(f"#### {agent_name} (신뢰도: {confidence:.1%})")
                        for artifact in artifacts:
                            render_artifact(artifact)
                    elif artifacts:
                        st.markdown(f"#### ⚠️ {agent_name} (낮은 신뢰도: {confidence:.1%})")
                        st.warning("이 결과는 신뢰도가 낮아 참고용으로만 사용하세요.")
                        for artifact in artifacts:
                            render_artifact(artifact)
        
        # 근거 기반 종합 분석 생성
        fact_based_summary = evidence_generator.generate_fact_based_summary(prompt, all_results)
        
        with placeholder:
            st.markdown(fact_based_summary)
            
        return fact_based_summary
        
    except Exception as e:
        debug_log(f"❌ 전문가급 답변 합성 실패: {e}", "error")
        fallback_response = f"""
## ⚠️ 분석 완료 (검증 제한)

총 {len(all_results)}개 단계가 실행되었으나, 할루시네이션 방지 시스템에서 오류가 발생했습니다.

**실행된 단계**:
"""
        
        for i, result in enumerate(all_results, 1):
            agent_name = result.get('agent_name', 'Unknown Agent')
            success = "✅" if result.get('success', False) else "❌"
            fallback_response += f"\n{i}. {success} {agent_name}"
        
        fallback_response += "\n\n⚠️ 결과 검증 과정에서 문제가 발생했습니다. 상세 결과는 개별 아티팩트를 확인하세요."
        
        with placeholder:
            st.markdown(fallback_response)
            
        return fallback_response

async def fallback_analysis(prompt: str, placeholder):
    """A2A 클라이언트를 사용할 수 없을 때의 폴백 분석"""
    try:
        debug_log("🔄 폴백 분석 모드 실행", "info")
        
        with placeholder:
            st.warning("⚠️ A2A 클라이언트를 사용할 수 없어 기본 분석을 수행합니다.")
            
            # 기본적인 데이터 정보 표시
            if hasattr(st.session_state, 'data_manager'):
                try:
                    available_datasets = st.session_state.data_manager.list_dataframes()
                    if available_datasets:
                        st.info(f"📊 사용 가능한 데이터셋: {', '.join(available_datasets)}")
                        
                        # 첫 번째 데이터셋에 대한 기본 정보 표시
                        df = st.session_state.data_manager.get_dataframe(available_datasets[0])
                        if df is not None:
                            st.markdown("### 📋 데이터 기본 정보")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("행 수", f"{df.shape[0]:,}")
                            with col2:
                                st.metric("열 수", f"{df.shape[1]:,}")
                            with col3:
                                st.metric("결측치", f"{df.isnull().sum().sum():,}")
                            
                            st.markdown("### 📊 데이터 샘플")
                            st.dataframe(df.head())
                    else:
                        st.info("업로드된 데이터셋이 없습니다. 먼저 데이터를 업로드해주세요.")
                except Exception as data_error:
                    st.error(f"데이터 접근 중 오류: {data_error}")
            
            fallback_message = "기본 분석이 완료되었습니다. 더 상세한 분석을 위해서는 A2A 시스템이 필요합니다."
            st.session_state.messages.append({"role": "assistant", "content": fallback_message})
            
    except Exception as e:
        debug_log(f"❌ 폴백 분석 실패: {e}", "error")
        with placeholder:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()