from __future__ import annotations
import json
import os
import sys
import asyncio
import logging
import functools
from typing import Any, Dict, List, Literal, Sequence, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import operator
from typing_extensions import TypedDict, Annotated

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import nest_asyncio
import platform
import yaml
import time
import uuid
from urllib.parse import urlsplit
import threading

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonAstREPLTool

# MCP imports - optional
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP adapters not available. Install langchain-mcp-adapters to use MCP tools.")
    
    # Dummy class for compatibility
    class MultiServerMCPClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def get_tools(self):
            return []

from pydantic import BaseModel

# Visualization - with fallback
try:
    import networkx as nx
    import plotly.graph_objects as go
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization packages not available. Install networkx and plotly for agent structure visualization.")

# Import utilities
try:
    from utils import (
        astream_graph, random_uuid, generate_followups, get_followup_llm,
        create_enhanced_python_tool
    )
except ImportError:
    # Fallback if utils.py doesn't have all functions
    import uuid
    def random_uuid():
        return str(uuid.uuid4())
    
    # Create dummy functions for missing imports
    async def astream_graph(*args, **kwargs):
        pass
    
    async def generate_followups(*args, **kwargs):
        return []
    
    def get_followup_llm(*args, **kwargs):
        return None
    
    # Use the updated version we just created in mcp_data_science_tools.py
    def create_enhanced_python_tool(df=None):
        """
        전문적인 데이터 분석을 위한 향상된 Python 도구를 생성합니다.
        동적으로 최신 데이터를 참조합니다.
        """
        # matplotlib의 원본 show 함수 백업
        original_show = plt.show
        
        def streamlit_show(*args, **kwargs):
            """plt.show()를 Streamlit 환경에서 자동으로 st.pyplot()으로 변환"""
            try:
                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig, clear_figure=False)
                    plt.figure()
                else:
                    original_show(*args, **kwargs)
            except Exception as e:
                print(f"Streamlit show error: {e}")
                original_show(*args, **kwargs)
        
        # matplotlib show 함수를 패치
        plt.show = streamlit_show
        
        # 한글 폰트 설정
        def setup_korean_font():
            """한글 폰트 자동 설정"""
            try:
                import platform
                if platform.system() == 'Windows':
                    plt.rcParams['font.family'] = 'Malgun Gothic'
                elif platform.system() == 'Darwin':
                    plt.rcParams['font.family'] = 'AppleGothic'
                else:
                    plt.rcParams['font.family'] = 'NanumGothic'
                plt.rcParams['axes.unicode_minus'] = False
                return True
            except:
                return False
        
        setup_korean_font()
        
        # 분석 환경 구성
        try:
            import scipy
            from scipy import stats
        except ImportError:
            scipy = None
            stats = None
            
        try:
            import sklearn
        except ImportError:
            sklearn = None
            
        import warnings
        warnings.filterwarnings('ignore')
        
        # 샘플 데이터 생성 함수
        def create_sample_data():
            """샘플 데이터 생성"""
            np.random.seed(42)
            n_samples = 1000
            
            # 다양한 타입의 샘플 데이터 생성
            sample_df = pd.DataFrame({
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 15000, n_samples),
                'education_years': np.random.randint(8, 20, n_samples),
                'satisfaction_score': np.random.randint(1, 11, n_samples),
                'department': np.random.choice(['IT', 'Sales', 'HR', 'Marketing', 'Finance'], n_samples),
                'is_manager': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
                'performance_rating': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
            })
            
            # 일부 결측치 추가
            sample_df.loc[np.random.choice(sample_df.index, 50), 'income'] = np.nan
            sample_df.loc[np.random.choice(sample_df.index, 30), 'satisfaction_score'] = np.nan
            
            return sample_df
        
        # 동적으로 현재 데이터 가져오기 - 개선된 버전
        def get_current_df():
            """현재 세션의 데이터프레임 반환 - 안전한 접근"""
            try:
                # Streamlit 세션 상태 접근
                if hasattr(st, 'session_state'):
                    # 업로드된 데이터가 있는지 확인
                    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
                        return st.session_state.uploaded_data
                    
                    # 백업으로 CSV 경로 확인 (다른 파일에서 사용하는 패턴)
                    if 'uploaded_csv_path' in st.session_state and st.session_state.uploaded_csv_path:
                        try:
                            return pd.read_csv(st.session_state.uploaded_csv_path)
                        except Exception as e:
                            print(f"⚠️ CSV 파일 로드 실패: {e}")
            except Exception as e:
                print(f"⚠️ 세션 상태 접근 실패: {e}")
            
            # 모든 경우에 대해 샘플 데이터 반환
            return create_sample_data()
        
        # 데이터 로드 헬퍼 함수 (개선된 버전)
        def load_data():
            """최신 데이터 로드 - 업로드된 데이터가 있으면 사용, 없으면 샘플 데이터"""
            try:
                # 직접 streamlit 세션 상태에서 데이터 확인
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'uploaded_data'):
                    uploaded_data = st.session_state.uploaded_data
                    if uploaded_data is not None and hasattr(uploaded_data, 'shape'):
                        print(f"✅ 업로드된 데이터 로드 완료: {uploaded_data.shape[0]} rows, {uploaded_data.shape[1]} columns")
                        return uploaded_data
                
                # 백업: get_current_df 함수 사용
                current_df = get_current_df()
                data_source = "업로드된 데이터" if (
                    hasattr(st, 'session_state') and 
                    'uploaded_data' in st.session_state and 
                    st.session_state.uploaded_data is not None
                ) else "샘플 데이터"
                print(f"✅ 데이터 로드 완료 ({data_source}): {current_df.shape[0]} rows, {current_df.shape[1]} columns")
                return current_df
            except Exception as e:
                print(f"⚠️ 데이터 로드 중 오류 발생: {e}")
                # 최후의 수단: 샘플 데이터 생성
                sample_data = create_sample_data()
                print(f"🔄 샘플 데이터로 대체: {sample_data.shape[0]} rows, {sample_data.shape[1]} columns")
                return sample_data
        
        # 데이터 정보 표시 함수
        def show_data_info():
            """현재 로드된 데이터 정보 표시"""
            try:
                current_df = get_current_df()
                
                # 데이터 소스 확인
                data_source = "샘플 데이터"
                if hasattr(st, 'session_state'):
                    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
                        data_source = "업로드된 데이터"
                    elif 'uploaded_csv_path' in st.session_state and st.session_state.uploaded_csv_path:
                        data_source = f"업로드된 데이터 (파일: {st.session_state.uploaded_csv_path})"
                
                print(f"📊 현재 사용 중인 데이터: {data_source}")
                print(f"📐 데이터 형태: {current_df.shape}")
                print(f"📝 컬럼: {list(current_df.columns)}")
                print(f"🔢 데이터 타입:")
                for col in current_df.columns:
                    print(f"  - {col}: {current_df[col].dtype}")
                
                # 기본 통계 정보
                print(f"\n📈 기본 통계:")
                print(f"  - 수치형 컬럼: {len(current_df.select_dtypes(include=[np.number]).columns)}개")
                print(f"  - 범주형 컬럼: {len(current_df.select_dtypes(include=['object', 'category']).columns)}개")
                print(f"  - 결측값: {current_df.isnull().sum().sum()}개")
                
                return current_df
            except Exception as e:
                error_msg = f"데이터 정보 표시 중 오류 발생: {e}"
                print(f"❌ {error_msg}")
                return create_sample_data()
        
        # df를 동적으로 참조하는 래퍼 클래스 (성능 최적화 버전)
        class DataFrameProxy:
            """DataFrame 프록시 클래스 - 항상 최신 데이터를 참조하며 성능 최적화 적용"""
            
            def __init__(self, cache_ttl=1.0):
                self._cache = None
                self._cache_time = 0
                self._cache_ttl = cache_ttl  # 캐시 유지 시간 (초)
                self._access_count = 0
                self._cache_hits = 0
                
            def _get_fresh_df(self):
                """캐시를 이용해 데이터프레임을 효율적으로 가져오기 - 성능 최적화"""
                import time
                current_time = time.time()
                self._access_count += 1
                
                # 캐시 유효성 검사 (설정 가능한 TTL)
                if self._cache is not None and (current_time - self._cache_time) < self._cache_ttl:
                    self._cache_hits += 1
                    return self._cache
                
                # 새로운 데이터 로드 (성능 측정)
                load_start = time.time()
                try:
                    # 직접 streamlit 세션 상태에서 데이터 확인 (우선순위)
                    if hasattr(st, 'session_state') and hasattr(st.session_state, 'uploaded_data'):
                        uploaded_data = st.session_state.uploaded_data
                        if uploaded_data is not None and hasattr(uploaded_data, 'shape'):
                            self._cache = uploaded_data
                            self._cache_time = current_time
                            load_time = time.time() - load_start
                            
                            # 로드 시간이 긴 경우 경고
                            if load_time > 0.1:  # 100ms 이상
                                print(f"⚠️ 데이터 로드 시간이 길어졌습니다: {load_time*1000:.2f}ms")
                            
                            return self._cache
                    
                    # 백업: get_current_df 함수 사용
                    self._cache = get_current_df()
                    self._cache_time = current_time
                    load_time = time.time() - load_start
                    
                    # 로드 시간이 긴 경우 경고
                    if load_time > 0.1:  # 100ms 이상
                        print(f"⚠️ 데이터 로드 시간이 길어졌습니다: {load_time*1000:.2f}ms")
                    
                    return self._cache
                except Exception as e:
                    print(f"⚠️ 데이터 로드 실패: {e}")
                    # 최후의 수단: 샘플 데이터
                    self._cache = create_sample_data()
                    self._cache_time = current_time
                    return self._cache
            
            def get_performance_stats(self):
                """성능 통계 반환"""
                cache_hit_rate = (self._cache_hits / self._access_count * 100) if self._access_count > 0 else 0
                return {
                    'access_count': self._access_count,
                    'cache_hits': self._cache_hits,
                    'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                    'cache_ttl': self._cache_ttl
                }
            
            def clear_cache(self):
                """캐시 강제 초기화"""
                self._cache = None
                self._cache_time = 0
                print("🗑️ 캐시가 초기화되었습니다.")
            
            def __getattr__(self, name):
                try:
                    current_df = self._get_fresh_df()
                    if current_df is not None and hasattr(current_df, name):
                        return getattr(current_df, name)
                    else:
                        # 데이터가 None이거나 속성이 없는 경우 명확한 안내
                        if current_df is None:
                            print("⚠️ 현재 데이터프레임이 None입니다. 데이터를 업로드하거나 load_data()를 실행하세요.")
                            # 샘플 데이터로 fallback
                            fallback_df = create_sample_data()
                            if hasattr(fallback_df, name):
                                print("🔄 샘플 데이터로 대체하여 실행합니다.")
                                return getattr(fallback_df, name)
                        raise AttributeError(f"DataFrame에 '{name}' 속성이 없습니다. 사용 가능한 주요 메서드: info(), describe(), head(), shape, columns")
                except Exception as e:
                    print(f"⚠️ DataFrameProxy 접근 오류: {e}")
                    # 최후의 수단으로 샘플 데이터 사용
                    try:
                        fallback_df = create_sample_data()
                        if hasattr(fallback_df, name):
                            print("🔄 샘플 데이터로 대체하여 실행합니다.")
                            return getattr(fallback_df, name)
                    except:
                        pass
                    raise AttributeError(f"DataFrame 접근 실패. 데이터를 다시 업로드하거나 load_data()를 실행하세요.")
            
            def __getitem__(self, key):
                current_df = self._get_fresh_df()
                return current_df[key]
            
            def __repr__(self):
                current_df = self._get_fresh_df()
                return repr(current_df)
            
            def __str__(self):
                current_df = self._get_fresh_df()
                return str(current_df)
            
            def __len__(self):
                current_df = self._get_fresh_df()
                return len(current_df)
            
            def __call__(self):
                """함수로 호출 시 실제 DataFrame 반환"""
                return self._get_fresh_df()
        
        # 데이터 상태 확인 함수
        def check_data_status():
            """현재 데이터 상태를 확인하고 상세 정보를 반환"""
            try:
                has_uploaded = False
                has_csv_path = False
                
                if hasattr(st, 'session_state'):
                    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
                        has_uploaded = True
                        print("✅ 업로드된 데이터가 session_state.uploaded_data에 있습니다.")
                    
                    if 'uploaded_csv_path' in st.session_state and st.session_state.uploaded_csv_path:
                        has_csv_path = True
                        print(f"✅ CSV 파일 경로가 설정되어 있습니다: {st.session_state.uploaded_csv_path}")
                
                if not has_uploaded and not has_csv_path:
                    print("ℹ️ 업로드된 데이터가 없어 샘플 데이터를 사용합니다.")
                
                # 실제 데이터 로드 테스트
                current_df = get_current_df()
                print(f"📊 현재 로드된 데이터: {current_df.shape}")
                return True
                
            except Exception as e:
                print(f"❌ 데이터 상태 확인 중 오류: {e}")
                return False

        analysis_env = {
            # 기본 패키지
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "plt": plt,
            "matplotlib": plt,
            "sns": sns,
            "seaborn": sns,
            "st": st,  # streamlit 접근 가능
            
            # 데이터프레임 - 동적 참조 (DataFrameProxy로 메서드 접근 가능)
            "df": DataFrameProxy(),  # 프록시 객체로 항상 최신 데이터 참조하며 df.info() 등 직접 호출 가능
            "data": DataFrameProxy(),  # 프록시 객체로 항상 최신 데이터 참조
            "get_df": get_current_df,  # 함수로 최신 데이터 접근
            "load_data": load_data,  # 명시적 데이터 로드
            "show_data_info": show_data_info,  # 데이터 정보 표시
            "check_data_status": check_data_status,  # 데이터 상태 확인
            
            # 추가 데이터 접근 헬퍼 함수들
            "get_uploaded_data": lambda: st.session_state.get('uploaded_data') if hasattr(st, 'session_state') else None,
            "has_uploaded_data": lambda: hasattr(st, 'session_state') and 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None,
            "get_data_safely": lambda: st.session_state.uploaded_data if (hasattr(st, 'session_state') and 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None) else create_sample_data(),
            
            # 유틸리티
            "warnings": warnings,
            "setup_korean_font": setup_korean_font,
            "create_sample_data": create_sample_data,
        }
        
        # 선택적 패키지 추가
        if scipy is not None:
            analysis_env["scipy"] = scipy
            analysis_env["stats"] = stats
        
        if sklearn is not None:
            analysis_env["sklearn"] = sklearn
        
        # 동적으로 현재 데이터 정보 가져오기
        current_df = get_current_df()
        data_source = "업로드된 데이터" if (
            hasattr(st, 'session_state') and 
            'uploaded_data' in st.session_state and 
            st.session_state.uploaded_data is not None
        ) else "샘플 데이터"
        
        description = f"""
🚀 **전문 데이터 분석 환경**

📊 **현재 로드된 데이터:**
- 데이터 소스: {data_source}
- 데이터 형태: {current_df.shape}
- 컬럼 수: {len(current_df.columns)}

📦 **사용 가능한 패키지:**
- pandas (pd), numpy (np)
- matplotlib (plt), seaborn (sns)
- scipy, sklearn
- streamlit (st) - 자동 시각화 지원

✨ **특별 기능:**
- plt.show() 자동 Streamlit 변환
- 한글 폰트 자동 설정
- 동적 데이터 로드 지원

💡 **데이터 접근 방법:**
```python
# 🎯 **FIXED! 업로드된 데이터 직접 접근 (권장)**
# df는 이제 DataFrame처럼 직접 메서드 호출이 가능합니다!
df.info()        # 데이터 정보 확인
df.describe()    # 기본 통계
df.head()        # 상위 5개 행
df.shape         # 데이터 크기
df.columns       # 컬럼 목록
df.isnull().sum()  # 결측치 확인

# 📊 데이터 상태 확인
has_uploaded_data()  # True/False로 업로드 여부 확인
get_uploaded_data()  # 업로드된 데이터만 가져오기 (None 가능)
get_data_safely()    # 안전하게 데이터 가져오기 (항상 DataFrame 반환)

# ✅ 추천 EDA 패턴
print("=== 데이터 기본 정보 ===")
df.info()
print("\\n=== 기본 통계 ===")
df.describe()
print("\\n=== 결측치 확인 ===")
print(df.isnull().sum())

# 기본 데이터 확인 함수들
check_data_status()      # 데이터 상태 점검
show_data_info()         # 상세 데이터 정보
load_data()              # 명시적 데이터 로드
```

🎯 **FIXED! 직접 메서드 접근**: df.info(), df.describe() 등 DataFrame 메서드 직접 호출 가능!
✅ **업로드 데이터 자동 연결**: 파일 업로드 시 자동으로 df 객체에 연결되어 바로 사용 가능!
"""
        
        return PythonAstREPLTool(
            locals=analysis_env,
            description=description,
            name="python_repl_ast",
            handle_tool_error=True
        )

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Apply nest_asyncio for Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# Page config
st.set_page_config(
    page_title="🔬 Data Science Multi-Agent System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

# Custom CSS for better visualization
st.markdown("""
<style>
    .agent-card {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .supervisor-card {
        border-color: #4CAF50;
        background-color: #e8f5e9;
    }
    .member-card {
        border-color: #2196F3;
        background-color: #e3f2fd;
    }
    .unregistered-card {
        border-color: #FF9800;
        background-color: #fff3e0;
        opacity: 0.7;
    }
    .tool-call-container {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f5f5f5;
    }
    .log-entry {
        font-family: monospace;
        font-size: 12px;
        background-color: #f0f0f0;
        padding: 5px;
        margin: 2px 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

if "agents" not in st.session_state:
    st.session_state.agents = {}  # {name: {type: "supervisor"|"member", prompt: str, tools: list}}

if "supervisor" not in st.session_state:
    st.session_state.supervisor = None

if "multi_agent_graph" not in st.session_state:
    st.session_state.multi_agent_graph = None

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

if "logging_config" not in st.session_state:
    st.session_state.logging_config = {
        "local": True,
        "langsmith": False,
        "langfuse": False
    }

if "logs" not in st.session_state:
    st.session_state.logs = []

if "generated_code" not in st.session_state:
    st.session_state.generated_code = []

if "result_files" not in st.session_state:
    st.session_state.result_files = []

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Route Response Model for Supervisor
class RouteResponse(BaseModel):
    next: str  # Will be validated dynamically

# Helper Functions
def log_event(event_type: str, content: dict):
    """로그 이벤트 기록"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "type": event_type,
        "content": content
    }
    st.session_state.logs.append(log_entry)
    
    # Local logging if enabled
    if st.session_state.logging_config["local"]:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"multi_agent_{datetime.now():%Y%m%d}.log"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def save_code(code: str, agent_name: str):
    """생성된 코드 저장"""
    code_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "code": code
    }
    st.session_state.generated_code.append(code_entry)
    
    # Save to file
    code_dir = Path("generated_code")
    code_dir.mkdir(exist_ok=True)
    code_file = code_dir / f"{agent_name}_{datetime.now():%Y%m%d_%H%M%S}.py"
    
    with open(code_file, "w", encoding="utf-8") as f:
        f.write(code)
    
    return code_file

def create_agent_node(state, agent, name):
    """Create an agent node for the graph"""
    log_event("agent_call", {"agent": name, "input": state.get("messages", [])[-1].content if state.get("messages") else ""})
    
    agent_response = agent.invoke(state)
    
    # Extract and save code if present
    if hasattr(agent_response.get("messages", [])[-1], "content"):
        content = agent_response["messages"][-1].content
        if "```python" in content:
            code_blocks = content.split("```python")
            for block in code_blocks[1:]:
                code = block.split("```")[0].strip()
                if code:
                    save_code(code, name)
    
    log_event("agent_response", {"agent": name, "output": agent_response["messages"][-1].content})
    
    return {
        "messages": [
            HumanMessage(content=agent_response["messages"][-1].content, name=name)
        ]
    }

def visualize_agent_structure():
    """Visualize the multi-agent structure using Plotly"""
    if not VISUALIZATION_AVAILABLE:
        st.warning("📊 Visualization packages not installed. Agent structure:")
        # Text-based visualization
        if st.session_state.supervisor:
            st.success(f"👑 **Supervisor**: {st.session_state.supervisor}")
        else:
            st.error("⚠️ **Supervisor**: Not Set - Please create a supervisor agent")
        
        members = [name for name, config in st.session_state.agents.items() 
                  if config["type"] == "member"]
        if members:
            st.info("🤖 **Members**:")
            for member in members:
                tools = st.session_state.agents[member].get("tools", [])
                st.write(f"  - {member} (Tools: {', '.join(tools)})")
        else:
            st.warning("⚠️ **Members**: None - Please create member agents")
        return None
    
    # VISUALIZATION_AVAILABLE is True, so create the figure
    fig = go.Figure()
    
    # Check if supervisor exists
    has_supervisor = st.session_state.supervisor is not None
    member_agents = [name for name, config in st.session_state.agents.items() 
                     if config["type"] == "member"]
    
    # Supervisor node
    if has_supervisor:
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.8],
            mode='markers+text',
            marker=dict(size=80, color="#4CAF50", line=dict(width=3, color='darkgreen')),
            text=[f"👑 {st.session_state.supervisor}"],
            textposition="middle center",
            textfont=dict(size=14, color="white"),
            name="Supervisor",
            hovertemplate="<b>Supervisor</b><br>%{text}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.8],
            mode='markers+text',
            marker=dict(size=80, color="#ffcccc", line=dict(width=3, color='#ff6666')),
            text=["❌ Supervisor Not Set"],
            textposition="middle center",
            textfont=dict(size=12, color="#cc0000"),
            name="Supervisor",
            hovertemplate="<b>Supervisor Required</b><br>Click sidebar to create<extra></extra>"
        ))
    
    # Member agent nodes
    if member_agents:
        num_agents = len(member_agents)
        x_positions = np.linspace(0.1, 0.9, num_agents)
        
        for i, agent_name in enumerate(member_agents):
            agent_config = st.session_state.agents[agent_name]
            tools = agent_config.get("tools", [])
            
            # Tool display
            if len(tools) > 3:
                tools_text = ", ".join(tools[:3]) + f"... (+{len(tools)-3})"
            else:
                tools_text = ", ".join(tools) if tools else "No tools"
            
            # Node color based on tools
            if "python_repl_ast" in tools:
                node_color = "#2196F3"
            else:
                node_color = "#90CAF9"
            
            fig.add_trace(go.Scatter(
                x=[x_positions[i]], y=[0.2],
                mode='markers+text',
                marker=dict(size=60, color=node_color, line=dict(width=2, color='darkblue')),
                text=[f"🤖 {agent_name}"],
                textposition="top center",
                textfont=dict(size=12),
                name=agent_name,
                hovertemplate=f"<b>{agent_name}</b><br>Tools: {tools_text}<extra></extra>"
            ))
            
            # Tool labels
            fig.add_annotation(
                x=x_positions[i], y=0.1,
                text=f"🔧 {tools_text}",
                showarrow=False,
                font=dict(size=10, color="gray")
            )
            
            # Add edges from supervisor to agents
            if has_supervisor:
                fig.add_trace(go.Scatter(
                    x=[0.5, x_positions[i]], y=[0.75, 0.25],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.5)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    else:
        # No members indicator
        fig.add_annotation(
            x=0.5, y=0.2,
            text="⚠️ No Member Agents<br>Create agents in sidebar",
            showarrow=False,
            font=dict(size=14, color="#FF9800"),
            bgcolor="rgba(255,152,0,0.1)",
            bordercolor="#FF9800",
            borderwidth=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "🔬 Data Science Multi-Agent Structure",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=400,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

async def initialize_mcp_tools(tool_config):
    """Initialize MCP tools from configuration"""
    if not MCP_AVAILABLE:
        logging.warning("MCP not available, skipping tool initialization")
        return []
        
    if not tool_config:
        return []
    
    try:
        # Check if MCP servers are already running
        import aiohttp
        import asyncio
        
        connections = tool_config.get("mcpServers", tool_config)
        tools = []
        
        # For SSE-based MCP servers
        for server_name, server_config in connections.items():
            if "url" in server_config and server_config.get("transport") == "sse":
                try:
                    # Test connection to SSE endpoint
                    async with aiohttp.ClientSession() as session:
                        async with session.get(server_config["url"], timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                logging.info(f"Connected to MCP server: {server_name} at {server_config['url']}")
                            else:
                                logging.warning(f"MCP server {server_name} returned status {response.status}")
                except Exception as e:
                    logging.warning(f"Could not connect to MCP server {server_name}: {e}")
                    st.warning(f"⚠️ MCP server '{server_name}' is not running. Please start it with: python mcp_{server_name}.py")
        
        # Initialize MCP client
        client = MultiServerMCPClient(connections)
        await client.__aenter__()
        tools = client.get_tools()
        st.session_state.mcp_client = client
        
        logging.info(f"Initialized {len(tools)} MCP tools")
        return tools
        
    except Exception as e:
        st.error(f"Failed to initialize MCP tools: {e}")
        logging.error(f"MCP initialization error: {e}", exc_info=True)
        return []

def setup_logging():
    """Setup logging based on configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    if st.session_state.logging_config["local"]:
        # Local file logging
        file_handler = logging.FileHandler(
            log_dir / f"multi_agent_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
    
    if st.session_state.logging_config["langsmith"]:
        # LangSmith setup
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "data-science-multi-agent"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    
    if st.session_state.logging_config["langfuse"]:
        # Langfuse setup
        try:
            from langfuse import Langfuse
            langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            st.session_state.langfuse = langfuse
        except ImportError:
            st.warning("Langfuse not installed. Run: pip install langfuse")

def load_prompts():
    """Load prompts from storage"""
    prompt_dir = Path("prompt-config")
    emp_no = os.getenv("EMP_NO", "default")
    prompt_file = prompt_dir / f"{emp_no}.json"
    
    if prompt_file.exists():
        with open(prompt_file, encoding="utf-8") as f:
            return json.load(f).get("prompts", {})
    return {}

def load_tools_configs():
    """Load MCP tool configurations"""
    mcp_dir = Path("mcp-config")
    configs = {}
    
    if mcp_dir.exists():
        for json_file in mcp_dir.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                configs[json_file.stem] = json.load(f)
    
    return configs

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Configuration")
    
    # Logging configuration
    with st.expander("📊 Logging Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            local_logging = st.checkbox(
                "Local", 
                value=st.session_state.logging_config["local"],
                help="Save logs to local files"
            )
        
        with col2:
            langsmith_logging = st.checkbox(
                "LangSmith", 
                value=st.session_state.logging_config["langsmith"],
                help="Enable LangSmith tracing"
            )
        
        with col3:
            langfuse_logging = st.checkbox(
                "Langfuse", 
                value=st.session_state.logging_config["langfuse"],
                help="Enable Langfuse tracing"
            )
        
        if st.button("Apply Logging Settings", use_container_width=True):
            st.session_state.logging_config["local"] = local_logging
            st.session_state.logging_config["langsmith"] = langsmith_logging
            st.session_state.logging_config["langfuse"] = langfuse_logging
            setup_logging()
            st.success("✅ Logging settings applied!")
    
    st.markdown("---")
    
    # Agent creation section
    st.subheader("➕ Create New Agent")
    
    # Load prompts and tools
    prompts = load_prompts()
    tool_configs = load_tools_configs()
    
    # Agent name
    agent_name = st.text_input(
        "Agent Name", 
        placeholder="e.g., Visualization Expert",
        help="Give your agent a descriptive name"
    )
    
    # Agent type
    agent_type = st.radio(
        "Agent Type", 
        ["member", "supervisor"],
        format_func=lambda x: "👑 Supervisor" if x == "supervisor" else "🤖 Member",
        help="Supervisor coordinates other agents, Members perform specific tasks"
    )
    
    # Prompt selection
    prompt_names = list(prompts.keys())
    selected_prompt = st.selectbox(
        "Select Prompt Template",
        ["Custom..."] + prompt_names,
        help="Choose a pre-defined prompt or create custom"
    )
    
    # Show prompt content
    if selected_prompt == "Custom...":
        prompt_text = st.text_area(
            "Prompt Content",
            height=150,
            placeholder="Enter your custom prompt here...",
            help="Define the agent's role and capabilities"
        )
    else:
        prompt_text = st.text_area(
            "Prompt Content",
            value=prompts[selected_prompt]["prompt"],
            height=150,
            help="You can edit the selected prompt"
        )
    
    # Tool selection (only for member agents)
    selected_tools = []
    tool_config_name = "None"  # Initialize with default value
    
    if agent_type == "member":
        st.subheader("🔧 Select Tools")
        
        # Enhanced Python tool (default)
        col1, col2 = st.columns([3, 1])
        with col1:
            use_python = st.checkbox(
                "Enhanced Python Tool", 
                value=True,
                help="Professional data analysis environment with pandas, numpy, matplotlib, etc."
            )
        
        if use_python:
            selected_tools.append("python_repl_ast")
        
        # MCP tools
        tool_config_name = st.selectbox(
            "MCP Tool Configuration",
            ["None"] + list(tool_configs.keys()),
            help="Select additional MCP tools"
        )
        
        if tool_config_name != "None":
            st.info(f"📦 Selected MCP config: {tool_config_name}")
            selected_tools.append(f"mcp:{tool_config_name}")
        
        # Display selected tools
        if selected_tools:
            st.success(f"Selected tools: {', '.join(selected_tools)}")
    
    # Create agent button
    if st.button("🚀 Create Agent", type="primary", use_container_width=True):
        if not agent_name:
            st.error("❌ Please enter an agent name")
        elif not prompt_text:
            st.error("❌ Please enter a prompt")
        elif agent_name in st.session_state.agents:
            st.error(f"❌ Agent '{agent_name}' already exists")
        else:
            # Create agent configuration
            agent_config = {
                "type": agent_type,
                "prompt": prompt_text,
                "tools": selected_tools,
                "tool_config": {}
            }
            
            # Add tool config for member agents if MCP tools selected
            if agent_type == "member" and tool_config_name != "None":
                agent_config["tool_config"] = tool_configs.get(tool_config_name, {})
            
            if agent_type == "supervisor":
                st.session_state.supervisor = agent_name
            
            st.session_state.agents[agent_name] = agent_config
            st.success(f"✅ Agent '{agent_name}' created successfully!")
            
            log_event("agent_created", {
                "name": agent_name,
                "type": agent_type,
                "tools": selected_tools
            })
            
            # Clear form by rerunning
            st.rerun()
    
    st.markdown("---")
    
    # Edit existing agents
    if st.session_state.agents:
        st.subheader("✏️ Edit Agents")
        
        # Group agents by type
        supervisors = [name for name, config in st.session_state.agents.items() 
                      if config["type"] == "supervisor"]
        members = [name for name, config in st.session_state.agents.items() 
                  if config["type"] == "member"]
        
        # Create tabs for different agent types
        tab1, tab2 = st.tabs(["👑 Supervisors", "🤖 Members"])
        
        with tab1:
            if supervisors:
                for supervisor_name in supervisors:
                    with st.expander(f"Edit {supervisor_name}", expanded=False):
                        agent_config = st.session_state.agents[supervisor_name]
                        
                        # Edit prompt
                        new_prompt = st.text_area(
                            "Prompt",
                            value=agent_config["prompt"],
                            height=150,
                            key=f"edit_prompt_{supervisor_name}"
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"💾 Save", key=f"save_{supervisor_name}"):
                                st.session_state.agents[supervisor_name]["prompt"] = new_prompt
                                st.success("✅ Agent updated!")
                                log_event("agent_updated", {"name": supervisor_name})
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🗑️ Delete", key=f"delete_{supervisor_name}"):
                                del st.session_state.agents[supervisor_name]
                                if st.session_state.supervisor == supervisor_name:
                                    st.session_state.supervisor = None
                                st.success("✅ Agent deleted!")
                                log_event("agent_deleted", {"name": supervisor_name})
                                st.rerun()
            else:
                st.info("No supervisors created yet")
        
        with tab2:
            if members:
                for member_name in members:
                    with st.expander(f"Edit {member_name}", expanded=False):
                        agent_config = st.session_state.agents[member_name]
                        
                        # Edit prompt
                        new_prompt = st.text_area(
                            "Prompt",
                            value=agent_config["prompt"],
                            height=150,
                            key=f"edit_prompt_{member_name}"
                        )
                        
                        # Show tools
                        st.info(f"🔧 Tools: {', '.join(agent_config.get('tools', []))}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"💾 Save", key=f"save_{member_name}"):
                                st.session_state.agents[member_name]["prompt"] = new_prompt
                                st.success("✅ Agent updated!")
                                log_event("agent_updated", {"name": member_name})
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🗑️ Delete", key=f"delete_{member_name}"):
                                del st.session_state.agents[member_name]
                                st.success("✅ Agent deleted!")
                                log_event("agent_deleted", {"name": member_name})
                                st.rerun()
            else:
                st.info("No member agents created yet")

# Main area
st.title("🔬 Data Science Multi-Agent System")
st.markdown("### Advanced AI-Powered Data Analysis Platform")

# Info box
if not st.session_state.supervisor or not any(a["type"] == "member" for a in st.session_state.agents.values()):
    st.info("""
    👋 **Welcome to the Data Science Multi-Agent System!**
    
    To get started:
    1. Create a **Supervisor** agent in the sidebar (coordinates the team)
    2. Create one or more **Member** agents (specialized data scientists)
    3. Click "Create Multi-Agent System" to initialize
    4. Start your data analysis journey!
    
    **Available Agent Specializations:**
    - 📊 **Visualization Expert**: Advanced charts and graphs
    - 📝 **Report Writer**: Professional analysis reports
    - 🔍 **EDA Specialist**: Exploratory data analysis
    - ⏰ **Time Series Analyst**: Temporal data patterns
    - 🚨 **Anomaly Detector**: Outlier detection
    - 🤖 **ML Engineer**: Machine learning models
    - 🧹 **Data Preprocessor**: Data cleaning and preparation
    - 📈 **Statistical Analyst**: Statistical tests and analysis
    - 🔢 **Numerical Analyst**: Complex computations
    - 🎯 **Model Interpreter**: Model explanation and optimization
    """)

# Visualize agent structure
st.markdown("### 🏗️ Agent Structure")
viz_result = visualize_agent_structure()
if viz_result is not None:
    st.plotly_chart(viz_result, use_container_width=True)

# Agent cards
if st.session_state.agents:
    st.markdown("### 📋 Registered Agents")
    
    # Separate supervisors and members
    supervisors = [(name, config) for name, config in st.session_state.agents.items() 
                  if config["type"] == "supervisor"]
    members = [(name, config) for name, config in st.session_state.agents.items() 
              if config["type"] == "member"]
    
    # Display supervisors
    if supervisors:
        st.markdown("#### 👑 Supervisors")
        cols = st.columns(min(len(supervisors), 3))
        for i, (name, config) in enumerate(supervisors):
            with cols[i % 3]:
                st.markdown(f"""
                <div class='agent-card supervisor-card'>
                    <h4>👑 {name}</h4>
                    <p><b>Type:</b> Supervisor</p>
                    <p><b>Role:</b> Coordinates agent team</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display members
    if members:
        st.markdown("#### 🤖 Member Agents")
        cols = st.columns(min(len(members), 3))
        for i, (name, config) in enumerate(members):
            with cols[i % 3]:
                tools = config.get("tools", [])
                tools_display = []
                for tool in tools:
                    if tool == "python_repl_ast":
                        tools_display.append("🐍 Python")
                    elif tool.startswith("mcp:"):
                        tools_display.append(f"📦 {tool[4:]}")
                    else:
                        tools_display.append(tool)
                
                tools_str = ", ".join(tools_display) if tools_display else "No tools"
                
                st.markdown(f"""
                <div class='agent-card member-card'>
                    <h4>🤖 {name}</h4>
                    <p><b>Type:</b> Member</p>
                    <p><b>Tools:</b> {tools_str}</p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# Create multi-agent system button
if st.session_state.supervisor and any(a["type"] == "member" for a in st.session_state.agents.values()):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Multi-Agent System", type="primary", use_container_width=True):
            with st.spinner("🔧 Building your data science team..."):
                # Build the multi-agent graph
                async def build_multi_agent_system():
                    # Get member agents
                    members = [name for name, config in st.session_state.agents.items() 
                              if config["type"] == "member"]
                    
                    # Update RouteResponse with dynamic members
                    members_list = ["FINISH"] + members
                    
                    class DynamicRouteResponse(BaseModel):
                        next: str
                        
                        def __init__(self, **data):
                            super().__init__(**data)
                            if self.next not in members_list:
                                raise ValueError(f"Invalid next value: {self.next}. Must be one of {members_list}")
                    
                    # Create supervisor prompt
                    supervisor_config = st.session_state.agents[st.session_state.supervisor]
                    system_prompt = supervisor_config["prompt"]
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                        ("system", f"""Given the conversation above, who should act next? Or should we FINISH? 
                        Select one of: {members_list}
                        
                        Remember:
                        - Assign tasks to the most appropriate specialist
                        - Ensure thorough analysis before finishing
                        - Coordinate multiple agents for complex tasks""")
                    ]).partial(members=", ".join(members))
                    
                    # Initialize LLM
                    llm = ChatOpenAI(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                        temperature=0.1
                    )
                    
                    # Create supervisor agent
                    def supervisor_agent(state):
                        supervisor_chain = prompt | llm.with_structured_output(DynamicRouteResponse)
                        result = supervisor_chain.invoke(state)
                        return {"next": result.next}
                    
                    # Create member agents
                    agent_nodes = {}
                    
                    for member_name in members:
                        member_config = st.session_state.agents[member_name]
                        
                        # Initialize tools
                        tools = []
                        
                        # Add enhanced Python tool if selected
                        if "python_repl_ast" in member_config.get("tools", []):
                            # 동적 데이터 참조를 위해 df 매개변수 없이 호출
                            tools.append(create_enhanced_python_tool())
                        
                        # Add MCP tools
                        for tool in member_config.get("tools", []):
                            if tool.startswith("mcp:"):
                                tool_config_name = tool[4:]
                                if tool_config_name in tool_configs:
                                    mcp_tools = await initialize_mcp_tools(tool_configs[tool_config_name])
                                    tools.extend(mcp_tools)
                        
                        # Create agent
                        agent = create_react_agent(
                            llm,
                            tools=tools,
                            state_modifier=member_config["prompt"]
                        )
                        
                        # Create node function
                        agent_nodes[member_name] = functools.partial(
                            create_agent_node, 
                            agent=agent, 
                            name=member_name
                        )
                    
                    # Build graph
                    workflow = StateGraph(AgentState)
                    
                    # Add nodes
                    workflow.add_node("Supervisor", supervisor_agent)
                    for member_name, node_func in agent_nodes.items():
                        workflow.add_node(member_name, node_func)
                    
                    # Add edges from members to supervisor
                    for member in members:
                        workflow.add_edge(member, "Supervisor")
                    
                    # Conditional routing from supervisor
                    conditional_map = {k: k for k in members}
                    conditional_map["FINISH"] = END
                    
                    def get_next(state):
                        return state["next"]
                    
                    workflow.add_conditional_edges("Supervisor", get_next, conditional_map)
                    workflow.add_edge(START, "Supervisor")
                    
                    # Compile
                    graph = workflow.compile(checkpointer=MemorySaver())
                    
                    return graph
                
                # Build the system
                try:
                    st.session_state.multi_agent_graph = st.session_state.event_loop.run_until_complete(
                        build_multi_agent_system()
                    )
                    st.success("✅ Multi-Agent System created successfully! Your data science team is ready.")
                    st.session_state.session_initialized = True
                    log_event("system_created", {
                        "supervisor": st.session_state.supervisor,
                        "members": [name for name, config in st.session_state.agents.items() 
                                   if config["type"] == "member"]
                    })
                except Exception as e:
                    st.error(f"❌ Failed to create multi-agent system: {e}")
                    logging.error(f"Multi-agent creation error: {e}", exc_info=True)

else:
    st.warning("⚠️ Please create at least one Supervisor and one Member agent to build the system.")

st.markdown("---")

# File upload section
if st.session_state.get("session_initialized"):
    with st.expander("📁 Upload Data for Analysis", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload your data file for analysis"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Loaded {uploaded_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show preview
                st.dataframe(df.head(), use_container_width=True)
                
                log_event("data_uploaded", {
                    "filename": uploaded_file.name,
                    "shape": df.shape,
                    "columns": df.columns.tolist()
                })
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

# Chat interface
if st.session_state.get("session_initialized") and st.session_state.multi_agent_graph:
    st.markdown("### 💬 Data Analysis Chat")
    
    # Display chat history
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑").write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
                
                # Show agent info
                if "agent" in message:
                    st.caption(f"Agent: {message['agent']}")
                
                # Show tool calls
                if "tool_calls" in message and message["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.markdown(f"""
                            <div class='tool-call-container'>
                                <b>Agent:</b> {tool_call.get('agent', 'Unknown')}<br>
                                <b>Tool:</b> {tool_call.get('tool', 'Unknown')}<br>
                                <b>Arguments:</b> <code>{json.dumps(tool_call.get('arguments', {}), ensure_ascii=False)}</code>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show generated images
                if "images" in message:
                    for img_path in message["images"]:
                        if Path(img_path).exists():
                            st.image(img_path, use_column_width=True)
    
    # Chat input
    user_input = st.chat_input("Ask your data science question...")
    
    if user_input:
        # Add user message
        st.session_state.history.append({"role": "user", "content": user_input})
        
        # Display user message
        st.chat_message("user", avatar="🧑").write(user_input)
        
        # Process with multi-agent system
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            agent_placeholder = st.empty()
            tool_placeholder = st.empty()
            
            async def process_multi_agent_query():
                config = RunnableConfig(
                    recursion_limit=20,
                    configurable={"thread_id": st.session_state.thread_id}
                )
                
                # Add Langfuse/LangSmith callbacks if enabled
                callbacks = []
                if st.session_state.logging_config["langsmith"]:
                    # LangSmith callback is handled by environment variables
                    pass
                
                if st.session_state.logging_config["langfuse"] and hasattr(st.session_state, "langfuse"):
                    # Add Langfuse callback
                    pass
                
                if callbacks:
                    config["callbacks"] = callbacks
                
                # Log the query
                log_event("user_query", {"content": user_input})
                
                final_response = ""
                tool_calls = []
                current_agent = None
                generated_images = []
                
                # Custom callback for streaming
                def streaming_callback(chunk):
                    nonlocal final_response, current_agent
                    
                    # Extract node and content
                    if isinstance(chunk, dict):
                        node = chunk.get("node", "")
                        content = chunk.get("content", "")
                    else:
                        return
                    
                    # Update agent display
                    if node and node != current_agent:
                        current_agent = node
                        agent_placeholder.info(f"🤖 **{node}** is analyzing...")
                    
                    # Process content
                    if isinstance(content, AIMessage):
                        if content.content:
                            final_response = content.content
                            response_placeholder.markdown(final_response)
                        
                        # Handle tool calls
                        if hasattr(content, 'tool_calls') and content.tool_calls:
                            for tool_call in content.tool_calls:
                                tool_id = tool_call.get("id", str(uuid.uuid4()))
                                
                                tool_calls.append({
                                    "id": tool_id,
                                    "agent": current_agent,
                                    "tool": tool_call.get("name", "unknown"),
                                    "arguments": tool_call.get("args", {})
                                })
                                
                                # Display tool call
                                with tool_placeholder.container():
                                    with st.expander(f"🔧 {current_agent} → {tool_call.get('name', 'unknown')}", expanded=True):
                                        st.json(tool_call.get("args", {}))
                                
                                # Log tool call
                                log_event("tool_call", {
                                    "agent": current_agent,
                                    "tool": tool_call.get("name", "unknown"),
                                    "arguments": tool_call.get("args", {})
                                })
                    
                    # Check for generated images
                    result_dir = Path("results")
                    if result_dir.exists():
                        for img_file in result_dir.glob("*.png"):
                            if img_file not in generated_images:
                                generated_images.append(img_file)
                                st.image(str(img_file), use_column_width=True)
                
                # Stream the response
                await astream_graph(
                    st.session_state.multi_agent_graph,
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    callback=streaming_callback,
                    stream_mode="messages"
                )
                
                # Log completion
                log_event("response_complete", {
                    "agent": current_agent,
                    "tool_calls_count": len(tool_calls)
                })
                
                return final_response, tool_calls, current_agent, generated_images
            
            # Run the query
            try:
                response, tools, agent, images = st.session_state.event_loop.run_until_complete(
                    process_multi_agent_query()
                )
                
                # Add to history
                history_entry = {
                    "role": "assistant",
                    "content": response,
                    "agent": agent,
                    "tool_calls": tools
                }
                
                if images:
                    history_entry["images"] = [str(img) for img in images]
                
                st.session_state.history.append(history_entry)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                logging.error(f"Query processing error: {e}", exc_info=True)
                log_event("error", {"type": "query_processing", "error": str(e)})

else:
    # Show example queries
    st.markdown("### 💡 Example Analysis Queries")
    
    example_queries = [
        "📊 Analyze the uploaded dataset and create comprehensive visualizations",
        "🔍 Perform exploratory data analysis and identify key patterns",
        "🚨 Detect anomalies and outliers in the data",
        "📈 Build a predictive model and evaluate its performance",
        "⏰ Analyze time series patterns and forecast future values",
        "📝 Generate a professional data analysis report",
        "🧹 Clean and preprocess the data for machine learning",
        "🔢 Perform statistical tests and correlation analysis"
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        with cols[i % 2]:
            st.info(query)

# Sidebar - View logs and generated code
with st.sidebar:
    st.markdown("---")
    
    # View logs
    with st.expander("📋 View Logs", expanded=False):
        if st.session_state.logs:
            # Filter options
            log_types = list(set(log["type"] for log in st.session_state.logs))
            selected_types = st.multiselect("Filter by type", log_types, default=log_types)
            
            # Display filtered logs
            filtered_logs = [log for log in st.session_state.logs if log["type"] in selected_types]
            
            for log in filtered_logs[-10:]:  # Show last 10 logs
                st.markdown(f"""
                <div class='log-entry'>
                    <b>{log['timestamp']}</b> - {log['type']}<br>
                    {json.dumps(log['content'], ensure_ascii=False)[:200]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No logs yet")
    
    # View generated code
    with st.expander("💻 Generated Code", expanded=False):
        if st.session_state.generated_code:
            for code_entry in st.session_state.generated_code[-5:]:  # Show last 5
                st.markdown(f"**{code_entry['agent']}** - {code_entry['timestamp']}")
                st.code(code_entry['code'][:500] + "..." if len(code_entry['code']) > 500 else code_entry['code'], 
                       language='python')
        else:
            st.info("No code generated yet")

# Footer
st.markdown("---")
st.caption("🔬 Data Science Multi-Agent System - Powered by LangGraph | Professional AI-Driven Data Analysis")

# MCP Server main block
if __name__ == "__main__":
    import argparse
    
    # Check if running as MCP server
    parser = argparse.ArgumentParser(description="Data Science Tools MCP Server")
    parser.add_argument("--port", type=int, default=8007, help="Port to run the MCP server on")
    parser.add_argument("--mcp", action="store_true", help="Run as MCP server instead of Streamlit app")
    
    args = parser.parse_args()
    
    if args.mcp or "--port" in sys.argv:
        # Run as MCP server
        from mcp.server.fastmcp import FastMCP
        import uvicorn
        
        # Create MCP server
        mcp = FastMCP(
            "DataScienceTools",
            instructions="Provides advanced data science tools and analysis capabilities."
        )
        
        # ────────────────────────────────────────────────────────────
        # 🔧 Utility: Streamlit context injector to silence warnings
        # ────────────────────────────────────────────────────────────
        try:
            from streamlit.runtime.scriptrunner import script_run_context as _src
            _MAIN_CTX = _src.get_script_run_ctx()

            def _ensure_streamlit_context():
                """Attach the main ScriptRunContext to the current thread if missing."""
                try:
                    if _src.get_script_run_ctx() is None and _MAIN_CTX is not None:
                        _src.add_script_run_ctx(threading.current_thread(), _MAIN_CTX)
                except Exception:
                    # Fallback ‑ simply ignore if internal API changed
                    pass
        except Exception:
            def _ensure_streamlit_context():
                pass  # Streamlit internals unavailable (e.g., pure MCP run)

        # ────────────────────────────────────────────────────────────
        # 📊 Helper: Flexible argument parser
        # ────────────────────────────────────────────────────────────
        def _parse_input(first: Any = None, **kwargs) -> Dict[str, Any]:
            """Accept string / dict / JSON and merge with explicit kwargs."""
            data: Dict[str, Any] = {}

            # 1) positional value provided
            if first is not None:
                if isinstance(first, str):
                    # try JSON → else naive ':' split (key1=value1,key2=value2)
                    try:
                        data.update(json.loads(first))
                    except Exception:
                        if ":" in first:
                            # format key1:value1,key2:value2
                            for part in first.split(","):
                                if ":" in part:
                                    k, v = [p.strip() for p in part.split(":", 1)]
                                    data[k] = v
                        else:
                            # treat as free-form description
                            data["text"] = first
                elif isinstance(first, dict):
                    data.update(first)
            # 2) explicit kwargs override
            for k, v in kwargs.items():
                if v is not None:
                    data[k] = v
            return data

        # ────────────────────────────────────────────────────────────
        # 🧮 analyze_data – quick descriptive / statistical EDA helper
        # ────────────────────────────────────────────────────────────
        @mcp.tool()
        async def analyze_data(
            payload: Any = None,
            *,
            analysis_type: str | None = None,
            target_columns: str | List[str] | None = None,
            summary_only: bool = True,
        ) -> str:
            """Flexible data analysis.

            Parameters accepted in many formats:
            1. JSON / dict with keys {analysis_type, target_columns, summary_only}
            2. Free-form description string
            3. Explicit keyword args
            """

            args = _parse_input(payload, analysis_type=analysis_type, target_columns=target_columns, summary_only=summary_only)

            df_proxy = st.session_state.get("uploaded_data") if hasattr(st, "session_state") else None
            # NEW: SSOT fallback using UnifiedDataManager when session_state is empty
            if df_proxy is None:
                try:
                    from multi_agent_supervisor import UnifiedDataManager  # Local import to avoid circular issues
                    manager = UnifiedDataManager()
                    if manager.is_data_loaded():
                        df_proxy = manager.get_data(prefer_session=True, safe_copy=True)
                except Exception:
                    df_proxy = None

            if df_proxy is None:
                return "❌ No dataframe available. Please upload CSV via sidebar or ensure SSOT is initialised."

            df = df_proxy  # actual DataFrame

            atype = args.get("analysis_type", "general")
            cols = args.get("target_columns") or df.columns.tolist()
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",") if c.strip()]

            result: Dict[str, Any] = {"analysis_type": atype, "columns": cols}

            try:
                if atype == "describe":
                    result["describe"] = df[cols].describe().to_dict()
                elif atype == "nulls":
                    result["nulls"] = df[cols].isnull().sum().to_dict()
                else:  # general
                    result["head"] = df[cols].head(3).to_dict(orient="list")
                    result["shape"] = df.shape
            except Exception as e:
                result["error"] = str(e)

            return json.dumps(result, ensure_ascii=False, indent=2)

        # # ────────────────────────────────────────────────────────────
        # # 📈 create_visualization – handles code OR spec inputs
        # # ────────────────────────────────────────────────────────────
        # @mcp.tool()
        # async def create_visualization(
        #     payload: Any = None,
        #     *,
        #     chart_type: str | None = None,
        #     data_columns: str | List[str] | None = None,
        #     code: str | None = None,
        #     save_path: str | None = None,
        # ) -> str:
        #     """Generate a matplotlib/seaborn plot and return file path.

        #     Accepts:
        #     • Raw python `code` (string)
        #     • JSON / dict payload {chart_type, data_columns}
        #     • Explicit keyword args
        #     • Simple shorthand "hist:col1,col2"
        #     """

        #     _ensure_streamlit_context()

        #     spec = _parse_input(payload, chart_type=chart_type, data_columns=data_columns, code=code, save_path=save_path)

        #     # Load dataframe
        #     df_proxy = st.session_state.get("uploaded_data") if hasattr(st, "session_state") else None
        #     if df_proxy is None:
        #         return "❌ No dataframe available in session_state.uploaded_data"

        #     df = df_proxy

        #     # If raw code provided, exec in safe namespace
        #     if spec.get("code"):
        #         local_env = {"df": df, "pd": pd, "plt": plt, "sns": sns, "np": np}
        #         try:
        #             exec(spec["code"], {}, local_env)  # no globals leakage
        #         except Exception as e:
        #             return f"❌ Error executing code: {e}"
        #     else:
        #         ctype = spec.get("chart_type", "hist")
        #         cols = spec.get("data_columns") or df.columns.tolist()
        #         if isinstance(cols, str):
        #             cols = [c.strip() for c in cols.split(",") if c.strip()]

        #         try:
        #             plt.figure(figsize=(10,6))
        #             if ctype in {"hist", "histogram"}:
        #                 df[cols].hist()
        #             elif ctype in {"box", "boxplot"}:
        #                 df[cols].plot(kind="box")
        #             elif ctype in {"scatter"} and len(cols) >= 2:
        #                 df.plot.scatter(x=cols[0], y=cols[1])
        #             else:
        #                 return f"❌ Unsupported chart_type '{ctype}' or insufficient columns"
        #         except Exception as e:
        #             return f"❌ Visualization error: {e}"

        #     # Save figure
        #     try:
        #         import uuid, os
        #         out_dir = Path("results")
        #         out_dir.mkdir(exist_ok=True)
        #         file_path = out_dir / f"viz_{uuid.uuid4().hex[:8]}.png"
        #         plt.savefig(file_path, dpi=150, bbox_inches="tight")
        #         plt.close()
        #         return str(file_path)
        #     except Exception as e:
        #         return f"❌ Failed to save image: {e}"

        # ────────────────────────────────────────────────────────────
        # 📊 run_statistical_test – very simple wrapper for scipy.stats
        # ────────────────────────────────────────────────────────────
        @mcp.tool()
        async def run_statistical_test(
            payload: Any = None,
            *,
            test_type: str | None = None,
            variables: str | List[str] | None = None,
        ) -> str:
            """Run a quick statistical test (t-test, chi2) using scipy.stats."""

            from scipy import stats as _st

            args = _parse_input(payload, test_type=test_type, variables=variables)
            ttype = args.get("test_type", "ttest")
            cols = args.get("variables")
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",") if c.strip()]

            df_proxy = st.session_state.get("uploaded_data") if hasattr(st, "session_state") else None
            if df_proxy is None:
                return "❌ No dataframe available in session_state.uploaded_data"

            df = df_proxy
            if not cols or any(c not in df.columns for c in cols):
                return "❌ Variables not provided or not in dataframe"

            try:
                if ttype.startswith("t") and len(cols) == 2:
                    stat, p = _st.ttest_ind(df[cols[0]].dropna(), df[cols[1]].dropna(), equal_var=False)
                elif ttype in {"chi2", "chi-square", "chi"} and len(cols) == 2:
                    contingency = pd.crosstab(df[cols[0]], df[cols[1]])
                    stat, p, _, _ = _st.chi2_contingency(contingency)
                else:
                    return f"❌ Unsupported test_type '{ttype}' or wrong variables"
            except Exception as e:
                return f"❌ Statistical test error: {e}"

            return json.dumps({"test": ttype, "statistic": stat, "pvalue": p}, ensure_ascii=False)

        # # ────────────────────────────────────────────────────────────
        # # 📝 generate_report – stub combining earlier results
        # # ────────────────────────────────────────────────────────────
        # @mcp.tool()
        # async def generate_report(summary_text: str = "", additional_findings: str | None = None) -> str:
        #     """Return a simple markdown report string."""
        #     report = f"""# 📄 Analysis Report\n\n{summary_text}\n\n## Additional Findings\n{additional_findings or 'N/A'}\n"""
        #     return report

        # # ────────────────────────────────────────────────────────────
        # # 🧹 housekeeping tool: truncate debug log & reset execution counts
        # # ────────────────────────────────────────────────────────────
        # @mcp.tool()
        # async def housekeeping(max_debug_lines: int = 200) -> str:
        #     """Clean up oversized logs and reset agent execution counts."""
        #     # Truncate debug.log
        #     try:
        #         dbg_file = Path("debug.log")
        #         if dbg_file.exists():
        #             lines = dbg_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        #             if len(lines) > max_debug_lines:
        #                 dbg_file.write_text("\n".join(lines[-max_debug_lines:]), encoding="utf-8")
        #         cleared = True
        #     except Exception:
        #         cleared = False

        #     # Reset execution counts in session state
        #     reset_done = False
        #     try:
        #         if hasattr(st, "session_state") and "agent_execution_state" in st.session_state:
        #             st.session_state.agent_execution_state = {}
        #             reset_done = True
        #     except Exception:
        #         pass

        #     return f"Housekeeping complete – debug.log trimmed: {cleared}, execution_state_reset: {reset_done}"

        # Start MCP server
        logging.info(f"Starting Data Science Tools MCP Server on port {args.port}")
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Run as Streamlit app (original behavior)
        # This block is empty because the Streamlit app runs automatically when the file is executed
        pass