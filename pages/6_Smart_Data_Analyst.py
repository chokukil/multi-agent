# pages/6_🧠_Smart_Data_Analyst.py
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import io
import base64
from datetime import datetime

# Third-party libraries
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for modern UI with thinking process visualization
st.markdown("""
<style>
/* Main container */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Hero section */
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    text-align: center;
}

/* File upload area */
.upload-zone {
    border: 3px dashed #667eea;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    background: linear-gradient(135deg, #f8faff 0%, #e6f3ff 100%);
    margin-bottom: 2rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.upload-zone:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    transform: translateY(-2px);
}

/* Thinking process container */
.thinking-container {
    background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    border-left: 6px solid #667eea;
}

/* Workflow step */
.workflow-step {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    border-left: 5px solid #28a745;
    position: relative;
    overflow: hidden;
}

.workflow-step::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
}

.workflow-step.thinking::before {
    background: linear-gradient(90deg, #ffc107 0%, #fd7e14 100%);
    animation: thinking-pulse 2s infinite;
}

.workflow-step.completed::before {
    background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
}

.workflow-step.error::before {
    background: linear-gradient(90deg, #dc3545 0%, #e74c3c 100%);
}

/* Result container */
.result-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    border-top: 4px solid #667eea;
}

/* Animation for thinking process */
@keyframes thinking-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'thinking_process' not in st.session_state:
    st.session_state.thinking_process = [
        {"title": "기본 분석 계획", "description": "EDA 분석 전략을 수립합니다.", "status": "pending"},
        {"title": "데이터 탐색", "description": "데이터 구조와 품질을 분석합니다.", "status": "pending"},
        {"title": "통계 분석", "description": "기본 통계량과 분포를 계산합니다.", "status": "pending"},
    ]
if 'analysis_workflow' not in st.session_state:
    st.session_state.analysis_workflow = [
        {"title": "데이터 로딩", "agent": "Data Loader", "status": "pending", "tools": ["CSV Reader", "Data Validator"]},
        {"title": "EDA 수행", "agent": "EDA Specialist", "status": "pending", "tools": ["Statistical Analysis", "Data Profiling"]},
        {"title": "시각화", "agent": "Visualization Expert", "status": "pending", "tools": ["Plotly", "Matplotlib"]},
    ]
if 'orchestrator_connected' not in st.session_state:
    st.session_state.orchestrator_connected = False

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def check_orchestrator_status() -> bool:
    """Check if orchestrator is available"""
    try:
        response = requests.get("http://localhost:8100/.well-known/agent.json", timeout=2)
        return response.status_code == 200
    except:
        return False

def process_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV, Excel, JSON 파일만 업로드 가능합니다.")
            return None
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        return None

def save_uploaded_data_to_shared(df: pd.DataFrame, filename: str) -> str:
    """Save uploaded data to shared directory for A2A agents"""
    try:
        # Create shared data directory
        shared_dir = Path("a2a_ds_servers/artifacts/data/shared_dataframes")
        shared_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        name_without_ext = Path(filename).stem
        shared_filename = f"uploaded_{name_without_ext}_{timestamp}.csv"
        shared_path = shared_dir / shared_filename
        
        # Save as CSV for A2A agents to access
        df.to_csv(shared_path, index=False)
        
        return shared_filename
    except Exception as e:
        st.error(f"데이터 저장 중 오류: {str(e)}")
        return filename

async def send_orchestrator_request(query: str, data_filename: str) -> Dict:
    """Send analysis request to orchestrator using A2A SDK v0.2.9"""
    
    try:
        # Import A2A SDK components dynamically to avoid module-level import issues
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import MessageSendParams, SendMessageRequest
        from uuid import uuid4
        
        orchestrator_url = "http://localhost:8100"  # Orchestrator port
        
        enhanced_query = f"""
데이터 분석 요청: {query}

업로드된 데이터: {data_filename}

다음과 같이 분석을 진행해주세요:
1. 먼저 분석 계획을 수립하고 어떤 에이전트들을 사용할지 결정해주세요
2. 각 단계별로 어떤 도구를 사용할지 명시해주세요
3. 분석 과정과 결과를 상세히 설명해주세요
4. 시각화가 필요한 경우 적절한 차트를 생성해주세요

분석 과정을 단계별로 스트리밍으로 보여주세요.
"""
        
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            # A2A SDK를 사용한 올바른 방식
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=orchestrator_url,
            )
            
            # Agent Card 가져오기
            try:
                agent_card = await resolver.get_agent_card()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Agent card fetch failed: {str(e)}",
                    "data": {}
                }
            
            # A2AClient 초기화
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            
            # 올바른 A2A 메시지 페이로드 구성
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': enhanced_query}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            # SendMessageRequest 생성
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # A2A SDK를 통한 메시지 전송
            response = await client.send_message(request)
            
            # 올바른 A2A SDK 응답 파싱
            if response and hasattr(response, 'root'):
                # SendMessageResponse -> root -> result -> parts
                root_response = response.root
                if hasattr(root_response, 'result'):
                    result_data = root_response.result
                    response_text = ""
                    
                    # Message 객체에서 parts 추출
                    if hasattr(result_data, 'parts') and result_data.parts:
                        for part in result_data.parts:
                            # Part 객체에서 root.text 추출
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                response_text += part.root.text
                    
                    # 응답이 있으면 성공으로 처리
                    if response_text:
                        return {
                            "success": True,
                            "data": {
                                "response": response_text,
                                "messageId": getattr(result_data, 'messageId', ''),
                                "role": str(getattr(result_data, 'role', 'agent'))
                            }
                        }
            
            # 응답이 없거나 형식이 잘못된 경우
            return {
                "success": False,
                "error": f"No valid response content received from orchestrator. Response type: {type(response)}",
                "data": {}
            }
            
    except ImportError as e:
        return {
            "success": False,
            "error": f"A2A SDK import failed: {str(e)}. Please install a2a-sdk",
            "data": {}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "data": {}
        }

def display_thinking_process(thinking_steps: List[Dict]):
    """Display orchestrator's thinking process"""
    if not thinking_steps:
        return
    
    st.markdown("""
    <div class="thinking-container">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🤔 오케스트레이터 사고 과정</h3>
    </div>
    """, unsafe_allow_html=True)
    
    for i, step in enumerate(thinking_steps):
        status_class = step.get('status', 'thinking')
        
        st.markdown(f"""
        <div class="workflow-step {status_class}">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; background: #ffc107; color: white;">
                    {step.get('status', 'PENDING')}
                </span>
                <h4 style="margin: 0; color: #495057;">
                    단계 {i+1}: {step.get('title', 'Processing...')}
                </h4>
            </div>
            <p style="margin: 0; color: #6c757d;">
                {step.get('description', 'Processing...')}
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_workflow_timeline(workflow_steps: List[Dict]):
    """Display workflow timeline with agent assignments"""
    if not workflow_steps:
        return
    
    st.markdown("### 📋 분석 워크플로우")
    
    for i, step in enumerate(workflow_steps):
        status = step.get('status', 'pending')
        agent_name = step.get('agent', 'Unknown Agent')
        tools = step.get('tools', [])
        title = step.get('title', f'Step {i+1}')
        
        # Status emoji and color
        if status == 'completed':
            status_emoji = "✅"
        elif status == 'running':
            status_emoji = "🔄"
        else:
            status_emoji = "⏳"
        
        st.markdown(f"{status_emoji} **{title}**")

def perform_analysis(user_query: str, df: pd.DataFrame, saved_filename: str):
    """Perform the actual analysis with improved error handling"""
    
    # Add user message to conversation
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_query,
        "timestamp": time.time()
    })
    
    # Display thinking process
    display_thinking_process(st.session_state.thinking_process)
    
    # Display workflow
    display_workflow_timeline(st.session_state.analysis_workflow)
    
    # Send request to orchestrator
    with st.spinner("🤖 오케스트레이터가 분석을 수행하고 있습니다..."):
        try:
            # Send actual request to orchestrator
            result = asyncio.run(send_orchestrator_request(user_query, saved_filename))
            
            # Update workflow steps based on orchestrator response
            if result.get("success", False):
                # Parse A2A SDK response
                response_data = result.get("data", {})
                response_content = response_data.get("response", "")
                
                if response_content:
                    # Update thinking process based on response
                    for i in range(len(st.session_state.thinking_process)):
                        st.session_state.thinking_process[i]["status"] = "completed"
                    
                    # Update workflow steps
                    for i in range(len(st.session_state.analysis_workflow)):
                        st.session_state.analysis_workflow[i]["status"] = "completed"
                    
                    # Add the orchestrator's response to chat
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": time.time()
                    })
                    
                    st.success("✅ 오케스트레이터 분석이 완료되었습니다!")
                    
                    # Display results in clean format
                    st.markdown("### 🔍 분석 결과")
                    st.markdown(response_content)
                    
                    return True
                else:
                    st.error("❌ 오케스트레이터로부터 응답을 받지 못했습니다.")
                    return False
            else:
                # Handle A2A SDK errors
                error_msg = result.get("error", "Unknown error occurred")
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": f"❌ 분석 중 오류가 발생했습니다: {error_msg}",
                    "timestamp": time.time()
                })
                
                st.error(f"❌ 오케스트레이터 연결 실패: {error_msg}")
                return False
                        
        except Exception as e:
            st.error(f"❌ 시스템 오류: {str(e)}")
            return False

# ------------------------------------------------------------------------------
# Main UI Layout
# ------------------------------------------------------------------------------

# Hero Section with Status
st.markdown("""
<div class="main-container">
    <div class="hero-section">
        <h1 style="color: white; margin-bottom: 0; font-size: 3rem;">🧠 Smart Data Analyst</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.3rem; margin-bottom: 1rem;">
            AI 오케스트레이터가 자동으로 최적의 분석 전략을 수립하고 실행합니다
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Check orchestrator status at start
st.session_state.orchestrator_connected = check_orchestrator_status()

# Status indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.uploaded_data is not None:
        st.success("📁 데이터 로드됨")
    else:
        st.info("📁 데이터 대기 중")

with col2:
    if st.session_state.orchestrator_connected:
        st.success("🤖 오케스트레이터 연결됨")
    else:
        st.error("🤖 오케스트레이터 연결 실패")

with col3:
    if any(step['status'] == 'completed' for step in st.session_state.analysis_workflow):
        st.success("✅ 분석 완료")
    else:
        st.info("⏳ 분석 대기 중")

with col4:
    if len(st.session_state.conversation_history) > 0:
        st.success("💬 대화 활성화")
    else:
        st.info("💬 대화 대기 중")

st.markdown("---")

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 데이터 분석", "💬 대화 기록", "🚀 빠른 분석"])

with tab1:
    # File upload area
    st.markdown("""
    <div class="upload-zone">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📁 데이터 업로드</h3>
        <p style="color: #6c757d;">CSV, Excel, JSON 파일을 지원합니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "분석할 데이터를 업로드하세요",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="CSV, Excel, JSON 형식의 파일을 업로드할 수 있습니다."
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        df = process_uploaded_file(uploaded_file)
        
        if df is not None:
            st.session_state.uploaded_data = df
            
            # Display basic data info
            st.success(f"✅ 데이터 업로드 완료: {uploaded_file.name}")
            
            # Save to shared directory for A2A agents
            saved_filename = save_uploaded_data_to_shared(df, uploaded_file.name)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("행 수", len(df))
            with col2:
                st.metric("열 수", len(df.columns))
            with col3:
                st.metric("결측값", df.isnull().sum().sum())
            
            # Data preview
            with st.expander("📖 데이터 미리보기", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Analysis request input
            st.markdown("### 🤖 AI 분석 요청")
            st.markdown("자연어로 분석을 요청하면 오케스트레이터가 최적의 분석 전략을 수립하고 실행합니다.")
            
            # Create a form for better enter key support
            with st.form(key="analysis_form", clear_on_submit=True):
                user_query = st.text_area(
                    "분석 요청을 입력하세요:",
                    placeholder="예: 이 데이터의 패턴을 분석하고 시각화해주세요",
                    height=100,
                    key="query_input"
                )
                
                # Form submit button
                submit_button = st.form_submit_button("🚀 분석 시작", type="primary")
                
                # Check if orchestrator is available before allowing analysis
                if submit_button:
                    if not st.session_state.orchestrator_connected:
                        st.error("❌ 오케스트레이터에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
                    elif not user_query.strip():
                        st.warning("❗ 분석 요청을 입력해주세요.")
                    else:
                        # Perform analysis. The form will be cleared automatically on rerun.
                        perform_analysis(user_query, df, saved_filename)
    
    # If no data uploaded yet
    else:
        st.info("📁 데이터를 업로드하여 AI 분석을 시작하세요.")

with tab2:
    st.markdown("### 💬 대화 기록")
    
    if st.session_state.conversation_history:
        for message in st.session_state.conversation_history:
            timestamp = datetime.fromtimestamp(message['timestamp']).strftime("%H:%M:%S")
            
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 20px; padding: 1.5rem; margin: 1rem 0;">
                    <strong>👤 사용자:</strong><br>
                    {message['content']}
                    <br><small style="color: #999;">{timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; padding: 1.5rem; margin: 1rem 0;">
                    <strong>🤖 오케스트레이터:</strong><br>
                    {message['content']}
                    <br><small style="color: #999;">{timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("아직 대화 기록이 없습니다. 분석을 요청해보세요!")

with tab3:
    st.markdown("### 🚀 빠른 분석")
    
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        
        # Quick analysis buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 기본 통계"):
                st.markdown("#### 📊 기본 통계 정보")
                st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            if st.button("🔍 데이터 품질"):
                st.markdown("#### 🔍 데이터 품질 분석")
                quality_info = {
                    "컬럼명": df.columns.tolist(),
                    "데이터 타입": df.dtypes.tolist(),
                    "결측값 개수": df.isnull().sum().tolist(),
                    "유니크 값 개수": df.nunique().tolist()
                }
                quality_df = pd.DataFrame(quality_info)
                st.dataframe(quality_df, use_container_width=True)
        
        with col3:
            if st.button("📊 시각화"):
                st.markdown("#### 📊 빠른 시각화")
                # Get numeric and categorical columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # 1. Distribution plot for first numeric column
                if numeric_cols:
                    fig_hist = px.histogram(
                        df, 
                        x=numeric_cols[0], 
                        title=f"{numeric_cols[0]} 분포",
                        color_discrete_sequence=['#667eea']
                    )
                    fig_hist.update_layout(
                        template="plotly_white",
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # 2. Correlation heatmap if multiple numeric columns
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        title="변수 간 상관관계",
                        color_continuous_scale='RdBu'
                    )
                    fig_corr.update_layout(
                        template="plotly_white",
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    else:
        st.info("데이터를 먼저 업로드해주세요.") 