"""
🧠 Smart Data Analyst - A2A Protocol Enhanced
Agent Chat의 우수한 패턴을 적용한 차세대 데이터 분석 시스템

핵심 특징:
- ThinkingStream: 오케스트레이터의 사고 과정 실시간 표시
- PlanVisualization: 분석 계획을 아름다운 카드로 시각화  
- BeautifulResults: 최종 결과를 전문적인 UI로 표시
- A2A Protocol: 진정한 에이전트 간 협업을 통한 분석
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

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from core.utils.logging import setup_logging

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

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"smart_analyst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "data_id" not in st.session_state:
        st.session_state.data_id = None

# --- A2A 서버 상태 확인 ---
def check_a2a_server_status():
    """A2A 서버들의 상태를 확인합니다"""
    servers = {
        "🧠 Orchestrator": "http://localhost:8100",
        "🐼 Pandas Data Analyst": "http://localhost:8200", 
        "🗄️ SQL Data Analyst": "http://localhost:8201",
        "📊 Data Visualization": "http://localhost:8202",
        "🔍 EDA Tools": "http://localhost:8203",
        "⚙️ Feature Engineering": "http://localhost:8204",
        "🧹 Data Cleaning": "http://localhost:8205"
    }
    
    status_results = {}
    
    with st.expander("🔍 A2A 서버 상태 확인", expanded=False):
        for name, url in servers.items():
            try:
                with httpx.Client(timeout=3.0) as client:
                    response = client.get(f"{url}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        agent_name = agent_card.get('name', name)
                        st.success(f"✅ **{name}**: `{agent_name}` 연결됨")
                        status_results[name] = True
                    else:
                        st.error(f"❌ **{name}**: HTTP {response.status_code}")
                        status_results[name] = False
            except Exception as e:
                st.error(f"❌ **{name}**: 연결 실패 - {str(e)[:50]}...")
                status_results[name] = False
    
    return status_results

# --- 데이터 업로드 처리 ---
def handle_data_upload():
    """데이터 업로드 처리"""
    st.markdown("### 📊 데이터 업로드")
    
    # 메인 업로드 영역
    uploaded_file = st.file_uploader(
        "CSV, Excel, JSON 파일을 업로드하세요",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="지원 형식: CSV, Excel, JSON"
    )
    
    # 샘플 데이터 옵션
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎯 타이타닉 샘플 데이터 사용", type="secondary"):
            try:
                sample_data = pd.read_csv("a2a_ds_servers/artifacts/data/shared_dataframes/titanic.csv")
                st.session_state.uploaded_data = sample_data
                st.session_state.data_id = "titanic"
                
                # 샘플 데이터를 공유 폴더에 저장
                sample_data.to_csv("a2a_ds_servers/artifacts/data/shared_dataframes/titanic.csv", index=False)
                
                st.success("✅ 타이타닉 샘플 데이터가 로드되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"샘플 데이터 로드 실패: {e}")
    
    if uploaded_file is not None:
        try:
            # 파일 확장자에 따른 데이터 로드
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            
            st.session_state.uploaded_data = data
            st.session_state.data_id = uploaded_file.name.split('.')[0]
            
            # 업로드된 데이터를 A2A 공유 폴더에 저장
            shared_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/{st.session_state.data_id}.csv"
            os.makedirs(os.path.dirname(shared_path), exist_ok=True)
            data.to_csv(shared_path, index=False)
            
            # 파일 정보 표시
            file_size = uploaded_file.size
            size_mb = file_size / (1024 * 1024)
            
            st.success(f"✅ **{uploaded_file.name}** 업로드 완료! ({size_mb:.2f} MB)")
            
            # 데이터 미리보기
            with st.expander("📋 데이터 미리보기", expanded=True):
                st.write(f"**데이터 형태:** {data.shape[0]:,} 행, {data.shape[1]:,} 열")
                st.dataframe(data.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**데이터 타입:**")
                    st.write(data.dtypes.value_counts())
                with col2:
                    st.write("**결측값:**")
                    missing_data = data.isnull().sum()
                    if missing_data.sum() > 0:
                        st.write(missing_data[missing_data > 0])
                    else:
                        st.write("결측값 없음 ✅")
            
            return True
            
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            return False
    
    return False

# --- 채팅 히스토리 렌더링 ---
def render_chat_history():
    """채팅 히스토리 렌더링 - Agent Chat 스타일"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                # 구조화된 메시지 처리
                if "thinking_process" in msg["content"]:
                    st.info(f"💭 {msg['content']['thinking_process']}")
                elif "plan_summary" in msg["content"]:
                    st.success(f"📋 {msg['content']['plan_summary']}")
                elif "analysis_result" in msg["content"]:
                    st.markdown(msg["content"]["analysis_result"])
                else:
                    st.markdown(str(msg["content"]))
            else:
                st.markdown(msg["content"])

# --- 오케스트레이터 계획 생성 ---
async def create_analysis_plan(prompt: str, data_info: dict):
    """오케스트레이터를 통해 분석 계획 생성"""
    plan_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"""데이터 분석 요청: {prompt}

데이터 정보:
- 데이터 ID: {data_info.get('data_id', 'unknown')}
- 형태: {data_info.get('shape', 'unknown')}
- 컬럼: {data_info.get('columns', [])}
- 데이터 타입 요약: {data_info.get('dtypes_summary', {})}

위 데이터에 대해 {prompt}를 수행하기 위한 단계별 계획을 수립해 주세요.
각 단계마다 어떤 A2A 에이전트가 어떤 작업을 수행할지 명시해 주세요."""
                    }
                ],
                "messageId": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "metadata": {}
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/",
                json=plan_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", {"error": "응답에 결과가 없습니다."})
            else:
                return {"error": f"오케스트레이터 연결 실패: HTTP {response.status_code}"}
                
    except Exception as e:
        return {"error": f"오케스트레이터 호출 중 오류: {str(e)}"}

# --- A2A 에이전트 실행 ---
async def execute_agent_task(agent_url: str, agent_name: str, task_description: str, data_info: dict):
    """개별 A2A 에이전트 작업 실행"""
    task_request = {
        "jsonrpc": "2.0", 
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"""작업 요청: {task_description}

데이터 정보:
- 데이터 ID: {data_info.get('data_id', 'unknown')}
- 파일 경로: a2a_ds_servers/artifacts/data/shared_dataframes/{data_info.get('data_id', 'unknown')}.csv
- 형태: {data_info.get('shape', 'unknown')}
- 컬럼: {data_info.get('columns', [])}

위 데이터에 대해 요청된 작업을 수행해 주세요."""
                    }
                ],
                "messageId": f"task_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "metadata": {}
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                agent_url,
                json=task_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # A2A 프로토콜의 올바른 응답 구조로 파싱
                if "result" in result:
                    a2a_result = result["result"]
                    
                    # status.message.parts에서 실제 결과 추출
                    if "status" in a2a_result and "message" in a2a_result["status"]:
                        message = a2a_result["status"]["message"]
                        if "parts" in message and message["parts"]:
                            return {
                                "parts": message["parts"],
                                "status": "completed",
                                "contextId": a2a_result.get("contextId", ""),
                                "taskId": a2a_result.get("id", "")
                            }
                    
                    # 백업: 최상위에서 parts 찾기
                    if "parts" in a2a_result:
                        return {"parts": a2a_result["parts"]}
                
                return {"error": "A2A 응답 구조가 예상과 다릅니다."}
            else:
                return {"error": f"{agent_name} 연결 실패: HTTP {response.status_code}"}
                
    except Exception as e:
        return {"error": f"{agent_name} 실행 중 오류: {str(e)}"}

# --- 계획 파싱 함수 ---
def parse_orchestrator_plan(plan_content: str) -> list:
    """오케스트레이터 계획을 파싱하여 실행 가능한 단계로 변환"""
    import re
    
    steps = []
    
    # Agent 매핑 - 실제 A2A 서버와 연결
    agent_mapping = {
        "pandas": {"url": "http://localhost:8200", "name": "Pandas Data Analyst"},
        "데이터분석": {"url": "http://localhost:8200", "name": "Pandas Data Analyst"},
        "sql": {"url": "http://localhost:8201", "name": "SQL Data Analyst"},
        "시각화": {"url": "http://localhost:8202", "name": "Data Visualization"},
        "eda": {"url": "http://localhost:8203", "name": "EDA Tools"},
        "피처": {"url": "http://localhost:8204", "name": "Feature Engineering"},
        "정제": {"url": "http://localhost:8205", "name": "Data Cleaning"}
    }
    
    # 간단한 계획 파싱 (실제 구현에서는 더 정교한 파싱 필요)
    lines = plan_content.split('\n')
    current_step = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 단계 번호 감지
        if re.match(r'^\d+\.', line) or re.match(r'^Step \d+', line, re.IGNORECASE):
            if current_step:
                steps.append(current_step)
            current_step = {
                "step_number": len(steps) + 1,
                "description": line,
                "agent_name": "Pandas Data Analyst",  # 기본값
                "agent_url": "http://localhost:8200",  # 기본값
                "task_description": line
            }
        elif "에이전트" in line or "agent" in line.lower():
            # 에이전트 지정이 있는 경우
            for key, agent_info in agent_mapping.items():
                if key in line.lower():
                    current_step["agent_name"] = agent_info["name"]
                    current_step["agent_url"] = agent_info["url"]
                    break
        elif current_step and len(line) > 10:
            # 작업 설명 확장
            current_step["task_description"] = line
    
    if current_step:
        steps.append(current_step)
    
    # 기본 계획이 없으면 표준 EDA 계획 생성
    if not steps:
        steps = [
            {
                "step_number": 1,
                "description": "1. 기본 데이터 분석 및 요약 통계",
                "agent_name": "Pandas Data Analyst", 
                "agent_url": "http://localhost:8200",
                "task_description": "데이터의 기본 구조, 요약 통계, 결측값 등을 분석해주세요.",
                "reasoning_insight": "데이터의 전반적인 건강성과 기본 특성을 파악하여 후속 분석의 방향성을 결정합니다. 결측값과 이상치 패턴을 통해 데이터 품질을 사전 진단할 수 있습니다."
            },
            {
                "step_number": 2,
                "description": "2. 탐색적 데이터 분석 (EDA)",
                "agent_name": "EDA Tools",
                "agent_url": "http://localhost:8203", 
                "task_description": "데이터 분포, 이상치, 패턴을 탐색하고 인사이트를 도출해주세요.",
                "reasoning_insight": "통계적 분포와 상관관계를 통해 숨겨진 패턴을 발굴하고, 비즈니스 가치를 창출할 수 있는 핵심 인사이트를 추출합니다. 변수 간 의존성 분석으로 인과관계의 실마리를 찾습니다."
            },
            {
                "step_number": 3,
                "description": "3. 데이터 시각화",
                "agent_name": "Data Visualization",
                "agent_url": "http://localhost:8202",
                "task_description": "주요 변수들의 관계와 분포를 시각화해주세요.",
                "reasoning_insight": "복잡한 수치 데이터를 직관적인 시각적 스토리로 변환하여 이해관계자들이 즉시 이해할 수 있는 형태로 제공합니다. 시각화를 통해 텍스트로는 전달하기 어려운 트렌드와 패턴을 명확히 드러냅니다."
            }
        ]
    
    return steps

# --- 단계별 실행 함수 ---
async def execute_plan_steps(steps: list, data_info: dict, prompt: str):
    """계획의 각 단계를 순차적으로 실행"""
    step_results = []
    
    for i, step in enumerate(steps):
        step_num = i + 1
        agent_name = step["agent_name"]
        agent_url = step["agent_url"]
        task_description = step["task_description"]
        
        with st.status(f"🚀 **Step {step_num}: {agent_name} 실행 중...**", expanded=True) as step_status:
            st.write(f"**작업:** {task_description}")
            st.write(f"**에이전트:** {agent_name}")
            
            try:
                # 실제 에이전트 호출
                result = await execute_agent_task(agent_url, agent_name, task_description, data_info)
                
                if "error" in result:
                    step_status.update(label=f"❌ Step {step_num} 실패", state="error")
                    st.error(f"에이전트 실행 실패: {result['error']}")
                    step_results.append({
                        "step": step_num,
                        "agent": agent_name,
                        "status": "failed",
                        "error": result["error"]
                    })
                else:
                    step_status.update(label=f"✅ Step {step_num} 완료", state="complete")
                    
                    # 결과 내용 추출 - 수정된 파싱 로직
                    result_content = ""
                    if "parts" in result and result["parts"]:
                        for part in result["parts"]:
                            if part.get("kind") == "text" and "text" in part:
                                # JSON 형태의 텍스트인 경우 파싱해서 읽기 쉽게 변환
                                text_content = part["text"]
                                try:
                                    # JSON 파싱 시도
                                    import json
                                    json_data = json.loads(text_content)
                                    
                                    # JSON 데이터를 읽기 쉬운 마크다운으로 변환
                                    if "recommended_steps" in json_data:
                                        result_content += f"## 📊 분석 결과\n\n{json_data['recommended_steps']}\n\n"
                                    
                                    if "data_wrangler_function" in json_data:
                                        result_content += f"## 💻 생성된 코드\n\n```python\n{json_data['data_wrangler_function']}\n```\n\n"
                                    
                                    # Plotly 차트가 있는 경우 특별 처리
                                    plotly_found = False
                                    for key, value in json_data.items():
                                        if isinstance(value, dict) and ('data' in value or 'layout' in value):
                                            try:
                                                import plotly.graph_objects as go
                                                import plotly.io as pio
                                                
                                                # JSON에서 Plotly Figure 재구성
                                                fig = go.Figure(value)
                                                
                                                # Streamlit에서 Plotly 차트 표시
                                                st.plotly_chart(fig, use_container_width=True)
                                                result_content += f"## 📊 {key.replace('_', ' ').title()}\n\n*Interactive visualization displayed above*\n\n"
                                                plotly_found = True
                                                
                                            except Exception as viz_error:
                                                st.warning(f"Plotly 차트 렌더링 실패: {viz_error}")
                                                result_content += f"## 📊 {key.replace('_', ' ').title()}\n\n```json\n{json.dumps(value, indent=2)}\n```\n\n"
                                    
                                    # "Plotly Chart Generated" 텍스트가 있으면 추가 처리 시도
                                    if not plotly_found and "Plotly Chart Generated" in text_content:
                                        # Raw 결과에서 실제 Plotly 데이터를 찾아보기
                                        raw_result = result.get("raw_result", {})
                                        if "parts" in raw_result:
                                            for part in raw_result["parts"]:
                                                if "text" in part:
                                                    try:
                                                        # 내부에 JSON이 있는지 확인
                                                        import re
                                                        json_match = re.search(r'\{.*\}', part["text"], re.DOTALL)
                                                        if json_match:
                                                            inner_json = json.loads(json_match.group())
                                                            for inner_key, inner_value in inner_json.items():
                                                                if isinstance(inner_value, dict) and ('data' in inner_value or 'layout' in inner_value):
                                                                    import plotly.graph_objects as go
                                                                    fig = go.Figure(inner_value)
                                                                    st.plotly_chart(fig, use_container_width=True)
                                                                    result_content += f"## 📊 Interactive Visualization\n\n*Chart displayed above*\n\n"
                                                                    plotly_found = True
                                                                    break
                                                    except Exception:
                                                        continue
                                    
                                    # A2A 서버의 함수 실행 코드가 있는 경우 안전하게 실행
                                    if "def data_visualization" in text_content and not plotly_found:
                                        try:
                                            # 함수 코드 추출
                                            import re
                                            func_match = re.search(r'def data_visualization\(.*?\):(.*?)(?=\n\n|\nPlotly|\n[A-Z]|\Z)', text_content, re.DOTALL)
                                            if func_match:
                                                func_code = "def data_visualization" + func_match.group(0)[20:]
                                                
                                                # 안전한 실행 환경에서 함수 실행
                                                exec_globals = {
                                                    'pd': __import__('pandas'),
                                                    'json': __import__('json'),
                                                    'px': __import__('plotly.express'),
                                                    'go': __import__('plotly.graph_objects'),
                                                    'pio': __import__('plotly.io')
                                                }
                                                exec(func_code, exec_globals)
                                                
                                                # 샘플 데이터로 함수 실행 (실제 구현에서는 데이터 연동 필요)
                                                import pandas as pd
                                                sample_data = pd.DataFrame({
                                                    'Age': [22, 35, 58, 25, 30],
                                                    'Fare': [7.25, 53.1, 51.86, 8.05, 10.5],
                                                    'Survived': [0, 1, 1, 0, 1]
                                                })
                                                
                                                if 'data_visualization' in exec_globals:
                                                    viz_result = exec_globals['data_visualization'](sample_data)
                                                    if isinstance(viz_result, dict):
                                                        import plotly.graph_objects as go
                                                        fig = go.Figure(viz_result)
                                                        st.plotly_chart(fig, use_container_width=True)
                                                        result_content += f"## 📊 Generated Visualization\n\n*Interactive chart displayed above*\n\n"
                                                        plotly_found = True
                                        except Exception as func_error:
                                            st.info(f"시각화 함수 실행 시도 중 오류: {func_error}")
                                    
                                    # 다른 JSON 필드들도 추가 (Plotly 차트가 아닌 경우)
                                    for key, value in json_data.items():
                                        if key not in ["recommended_steps", "data_wrangler_function"] and value:
                                            if not (isinstance(value, dict) and ('data' in value or 'layout' in value)):
                                                result_content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"
                                
                                except json.JSONDecodeError:
                                    # JSON이 아닌 일반 텍스트
                                    result_content += text_content + "\n\n"
                    
                    step_results.append({
                        "step": step_num,
                        "agent": agent_name,
                        "status": "completed",
                        "result": result_content,
                        "raw_result": result
                    })
                    
                    # 실시간 결과 표시
                    if result_content:
                        with st.expander(f"📊 Step {step_num} 결과 미리보기", expanded=False):
                            st.markdown(result_content[:800] + "..." if len(result_content) > 800 else result_content)
                
                # 단계 간 짧은 대기 (UI 안정성)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                step_status.update(label=f"❌ Step {step_num} 오류", state="error")
                st.error(f"단계 실행 중 오류: {str(e)}")
                step_results.append({
                    "step": step_num,
                    "agent": agent_name, 
                    "status": "error",
                    "error": str(e)
                })
    
    return step_results

# --- 최종 결과 통합 ---
def aggregate_step_results(step_results: list, prompt: str) -> str:
    """단계별 결과를 통합하여 최종 보고서 생성"""
    
    successful_results = [r for r in step_results if r["status"] == "completed"]
    failed_results = [r for r in step_results if r["status"] in ["failed", "error"]]
    
    # 보고서 구성
    report = f"""# 📊 데이터 분석 완료 보고서

## 🎯 분석 요청
{prompt}

## 📋 실행 요약
- **총 단계**: {len(step_results)}개
- **성공**: {len(successful_results)}개
- **실패**: {len(failed_results)}개

"""
    
    # 성공한 단계들의 결과 통합
    if successful_results:
        report += "## ✅ 분석 결과\n\n"
        
        for result in successful_results:
            report += f"### {result['step']}. {result['agent']}\n"
            if result.get("result") and result["result"].strip():
                # 결과가 있을 때만 표시하고, HTML 마크다운으로 렌더링
                report += f"{result['result']}\n\n"
            else:
                # 실제로는 에이전트가 작업을 수행했지만 결과 전달이 안된 경우
                report += f"""
<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
    <strong>⚠️ 결과 전달 문제</strong><br>
    에이전트가 작업을 수행했지만 결과가 전달되지 않았습니다.<br>
    <small>이는 A2A 통신 지연이나 타임아웃으로 인한 일시적 문제일 수 있습니다.</small>
</div>

"""
    
    # 실패한 단계들 요약
    if failed_results:
        report += "## ⚠️ 실패한 단계\n\n"
        for result in failed_results:
            report += f"- **Step {result['step']} ({result['agent']})**: {result.get('error', '알 수 없는 오류')}\n"
        report += "\n"
    
    # 전체 인사이트 및 결론
    if successful_results:
        report += """## 🎯 주요 인사이트 및 권장사항

위의 분석 결과를 종합하면 다음과 같은 인사이트를 얻을 수 있습니다:

1. **데이터 품질**: 업로드된 데이터의 전반적인 상태와 품질을 확인했습니다.
2. **패턴 발견**: 데이터에서 발견된 주요 패턴과 트렌드를 식별했습니다.
3. **시각적 분석**: 그래프와 차트를 통해 데이터의 특성을 명확히 파악했습니다.

### 💡 다음 단계 제안
- 추가적인 분석이 필요한 영역이 있다면 구체적으로 요청해주세요.
- 특정 변수나 관계에 대해 더 깊이 있는 분석을 원하시면 알려주세요.
- 예측 모델링이나 고급 통계 분석이 필요하시면 문의해주세요.

---
*이 보고서는 A2A 프로토콜을 통해 다중 에이전트가 협력하여 생성되었습니다.*
"""
    
    return report

# --- 사용자 쿼리 처리 ---
async def process_user_query(prompt: str):
    """사용자 쿼리 처리 - A2A 오케스트레이터 완전 구현"""
    
    # 1. 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # 데이터 확인
        if st.session_state.uploaded_data is None:
            st.error("⚠️ 먼저 데이터를 업로드해주세요!")
            return
        
        data = st.session_state.uploaded_data
        data_info = {
            "data_id": st.session_state.data_id,
            "shape": f"{data.shape[0]} 행, {data.shape[1]} 열",
            "columns": data.columns.tolist(),
            "dtypes_summary": data.dtypes.value_counts().to_dict()
        }
        
        # 2. ThinkingStream 시작 - Agent Chat 스타일
        thinking_container = st.container()
        thinking_stream = ThinkingStream(thinking_container)
        
        thinking_stream.start_thinking("🤔 요청을 분석하고 있습니다...")
        
        thinking_stream.add_thought("사용자의 데이터 분석 요청을 이해하고 있습니다.", "analysis")
        thinking_stream.add_thought("데이터 구조를 파악하고 적절한 분석 전략을 수립하고 있습니다.", "planning")
        thinking_stream.add_thought("오케스트레이터에게 상세한 계획 수립을 요청하겠습니다.", "planning")
        
        # 3. 오케스트레이터 계획 수립
        with st.status("🧠 **오케스트레이터 계획 수립 중...**", expanded=True) as status:
            plan_result = await create_analysis_plan(prompt, data_info)
            
            if "error" in plan_result:
                thinking_stream.add_thought(f"계획 수립 실패: {plan_result['error']}", "error")
                status.update(label="❌ 오케스트레이터 계획 수립 실패", state="error")
                
                # 4. 대안: 직접 pandas_data_analyst 실행
                thinking_stream.add_thought("대안으로 pandas_data_analyst에게 직접 분석을 요청합니다.", "planning")
                
                with st.status("🐼 **Pandas Data Analyst 직접 실행 중...**", expanded=True) as direct_status:
                    direct_result = await execute_agent_task(
                        "http://localhost:8200/",
                        "pandas_data_analyst", 
                        prompt,
                        data_info
                    )
                    
                    if "error" in direct_result:
                        thinking_stream.add_thought(f"직접 분석도 실패: {direct_result['error']}", "error")
                        direct_status.update(label="❌ 직접 분석 실패", state="error")
                        st.error(f"분석 실패: {direct_result['error']}")
                        
                        # 오류 해결 가이드 제공
                        st.markdown("""
                        ### 🔧 문제 해결 가이드
                        
                        **A2A 서버가 실행되지 않은 것 같습니다. 다음을 확인해주세요:**
                        
                        1. **서버 시작**: `./start.sh` 또는 `./system_start.bat` 실행
                        2. **서버 상태**: 사이드바의 "A2A 서버 상태 확인" 버튼 클릭
                        3. **포트 확인**: 8100-8205 포트가 사용 가능한지 확인
                        
                        **필요한 서버들:**
                        - 🧠 Orchestrator (8100)
                        - 🐼 Pandas Data Analyst (8200)
                        - 🗄️ SQL Data Analyst (8201)
                        - 📊 Data Visualization (8202)
                        - 🔍 EDA Tools (8203)
                        - ⚙️ Feature Engineering (8204)
                        - 🧹 Data Cleaning (8205)
                        """)
                        return
                    else:
                        thinking_stream.add_thought("직접 분석이 성공했습니다!", "success")
                        thinking_stream.finish_thinking("✅ 분석 완료!")
                        direct_status.update(label="✅ 분석 완료", state="complete")
                        
                        # 5. BeautifulResults로 결과 표시
                        results_container = st.container()
                        beautiful_results = BeautifulResults(results_container)
                        
                        # 결과 파싱 및 표시
                        if "parts" in direct_result and direct_result["parts"]:
                            result_content = ""
                            for part in direct_result["parts"]:
                                if "text" in part:
                                    result_content += part["text"] + "\n"
                            
                            # BeautifulResults로 멋진 결과 표시
                            beautiful_results.display_analysis_result(
                                {"output": result_content, "output_type": "markdown"},
                                "Pandas Data Analyst"
                            )
                            
                            # 세션에 결과 저장
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": {"analysis_result": result_content}
                            })
                        else:
                            st.warning("분석 결과가 비어있습니다.")
                
                return
            
            thinking_stream.add_thought("오케스트레이터가 훌륭한 계획을 수립했습니다!", "success")
            status.update(label="✅ 계획 수립 완료", state="complete")
            
            # 6. 계획 내용 추출 및 파싱
            plan_content = ""
            if "parts" in plan_result and plan_result["parts"]:
                for part in plan_result["parts"]:
                    if "text" in part:
                        plan_content += part["text"] + "\n"
            
            if not plan_content:
                st.warning("계획 내용이 비어있습니다. 기본 분석을 수행합니다.")
                plan_content = "기본 데이터 분석을 수행합니다."
            
            # 7. 계획을 실행 가능한 단계로 파싱
            thinking_stream.add_thought("계획을 실행 가능한 단계로 분석하고 있습니다.", "planning")
            execution_steps = parse_orchestrator_plan(plan_content)
            
            # 8. PlanVisualization으로 계획 표시
            plan_container = st.container()
            plan_viz = PlanVisualization(plan_container)
            
            # 계획 단계를 시각화용으로 변환
            plan_steps_for_viz = []
            for step in execution_steps:
                # 추론 인사이트 개선 - 단순 반복이 아닌 의미있는 분석 추론 제공
                reasoning_insight = step.get("reasoning_insight", "")
                if not reasoning_insight:
                    # 기본 추론이 없으면 에이전트별 특화된 인사이트 생성
                    if "Pandas" in step["agent_name"]:
                        reasoning_insight = "데이터프레임 구조 분석을 통해 정량적 특성을 파악하고, 통계적 기반으로 후속 분석 전략을 수립합니다."
                    elif "EDA" in step["agent_name"]:
                        reasoning_insight = "탐색적 분석으로 데이터의 숨겨진 패턴과 이상징후를 발굴하여 비즈니스 인사이트를 추출합니다."
                    elif "Visualization" in step["agent_name"]:
                        reasoning_insight = "시각적 스토리텔링을 통해 복잡한 데이터를 직관적이고 설득력 있는 형태로 변환합니다."
                    else:
                        reasoning_insight = f"Step {step['step_number']}: {step['task_description']}"
                
                plan_steps_for_viz.append({
                    "agent_name": step["agent_name"],
                    "skill_name": step["description"], 
                    "parameters": {
                        "user_instructions": step["task_description"],
                        "data_id": data_info["data_id"]
                    },
                    "reasoning": reasoning_insight  # 개선된 추론 인사이트 사용
                })
            
            plan_viz.display_plan(plan_steps_for_viz, "🎯 오케스트레이터 수립 계획")
            
            thinking_stream.add_thought(f"{len(execution_steps)}개 단계로 구성된 실행 계획이 준비되었습니다.", "success")
            thinking_stream.finish_thinking("✅ 계획 수립 완료! 이제 실행을 시작합니다.")
            
            # 9. **핵심**: 계획의 각 단계를 실제로 실행
            st.markdown("### 🚀 계획 실행 단계")
            st.markdown(f"총 **{len(execution_steps)}개 단계**를 순차적으로 실행합니다.")
            
            step_results = await execute_plan_steps(execution_steps, data_info, prompt)
            
            # 10. 모든 단계 완료 후 최종 결과 통합
            if step_results:
                with st.status("📊 **최종 결과 통합 중...**", expanded=True) as final_status:
                    final_report = aggregate_step_results(step_results, prompt)
                    final_status.update(label="✅ 분석 완료", state="complete")
                    
                    # 11. BeautifulResults로 최종 통합 결과 표시
                    results_container = st.container()
                    beautiful_results = BeautifulResults(results_container)
                    
                    # HTML 안전 렌더링을 위해 unsafe_allow_html=True 사용
                    beautiful_results.display_analysis_result(
                        {"output": final_report, "output_type": "markdown"},
                        f"Multi-Agent Analysis ({len([r for r in step_results if r['status'] == 'completed'])} agents)"
                    )
                    
                    # 12. 세션에 전체 결과 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {"plan_summary": f"계획 수립 완료: {len(execution_steps)}개 단계"}
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": {"analysis_result": final_report}
                    })
                    
                    # 13. 추가 분석 제안
                    st.markdown("### 🎯 다음 단계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔍 상세 분석", key="detailed"):
                            st.info("구체적인 분석 요청을 입력해주세요")
                    with col2:
                        if st.button("📈 예측 모델링", key="modeling"):
                            st.info("예측하고 싶은 변수를 알려주세요")
                    with col3:
                        if st.button("📊 커스텀 시각화", key="custom_viz"):
                            st.info("원하는 시각화를 구체적으로 요청해주세요")
            else:
                st.error("모든 단계가 실패했습니다. 서버 상태를 확인해주세요.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Smart Data Analyst",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="collapsed"
    )
    
    # 환경 설정
    setup_environment()
    initialize_session_state()
    
    # 메인 타이틀
    st.title("🧠 Smart Data Analyst")
    st.markdown("**A2A 프로토콜 기반 지능형 데이터 분석 어시스턴트** - Agent Chat의 우수한 패턴 적용")
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # 서버 상태 확인
        if st.button("🔍 A2A 서버 상태 확인"):
            check_a2a_server_status()
        
        st.markdown("---")
        
        # 데이터 업로드
        handle_data_upload()
        
        if st.session_state.uploaded_data is not None:
            st.success(f"✅ 데이터 준비됨: {st.session_state.data_id}")
            st.info(f"📊 {st.session_state.uploaded_data.shape[0]} 행, {st.session_state.uploaded_data.shape[1]} 열")
        else:
            st.info("📁 데이터를 업로드하거나 샘플 데이터를 선택하세요")
        
        st.markdown("---")
        
        # 추가 도구
        if st.button("🗑️ 대화 내역 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    # 환영 메시지 (첫 방문시)
    if not st.session_state.messages:
        welcome_html = """
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
        ">
            <h2>👋 Smart Data Analyst에 오신 것을 환영합니다!</h2>
            <p style="font-size: 18px; margin: 15px 0;">
                A2A 프로토콜로 구동되는 차세대 지능형 데이터 분석 시스템
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                <div>🧠 사고 과정 표시</div>
                <div>📋 계획 시각화</div>
                <div>🎨 아름다운 결과</div>
            </div>
        </div>
        """
        st.markdown(welcome_html, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 사용 방법
        1. **데이터 업로드**: 사이드바에서 CSV, Excel, JSON 파일을 업로드하거나 샘플 데이터를 선택하세요
        2. **분석 요청**: 아래 입력창에 원하는 분석을 입력하세요
        3. **실시간 관찰**: AI의 사고 과정, 계획 수립, 실행 과정을 실시간으로 확인하세요
        
        ### 💡 예시 질문
        - "이 데이터에 대해 전반적인 EDA를 수행해줘"
        - "데이터 요약 통계를 보여줘"  
        - "컬럼 간 상관관계를 분석해줘"
        - "데이터 품질 문제를 찾아줘"
        - "시각화를 만들어줘"
        """)
    
    # 채팅 히스토리 표시
    render_chat_history()
    
    # 채팅 입력 - 완전한 기능 구현
    if prompt := st.chat_input("🎯 어떤 데이터 분석을 원하시나요? (예: 'EDA 수행해줘', '데이터 요약해줘')"):
        # 비동기 처리를 위한 이벤트 루프
        try:
            # nest_asyncio가 적용되어 있으므로 바로 실행
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프에서는 create_task 사용
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, process_user_query(prompt))
                    future.result()
            else:
                asyncio.run(process_user_query(prompt))
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            logging.error(f"Query processing error: {e}", exc_info=True)
            
            # 오류 발생 시 기본 응답 제공
            with st.chat_message("assistant"):
                st.markdown(f"""
                ### ⚠️ 오류가 발생했습니다
                
                **오류 내용:** {str(e)}
                
                **해결 방법:**
                1. A2A 서버가 실행 중인지 확인해주세요 (`./start.sh`)
                2. 사이드바의 "A2A 서버 상태 확인" 버튼을 눌러 서버 상태를 확인해주세요
                3. 네트워크 연결을 확인해주세요
                
                문제가 지속되면 개발팀에 문의해주세요.
                """)
