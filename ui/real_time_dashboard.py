import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import logging

# Import required modules
from core.plan_execute.state import PlanExecuteState
from core.plan_execute.router import TASK_EXECUTOR_MAPPING
from core.data_lineage import data_lineage_tracker
from core.data_manager import data_manager

def render_real_time_dashboard():
    """실시간 프로세스 시각화 대시보드"""
    st.markdown("### 🚀 실시간 프로세스 모니터링")
    
    # 레이아웃 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 계획 진행 상황 표시
        render_plan_progress()
        
        # 데이터 변환 플로우 시각화
        render_data_transformation_flow()
    
    with col2:
        # 에이전트 활동 상태 패널
        render_agent_activity_panel()
        
        # 시스템 메트릭
        render_system_metrics()

def render_plan_progress():
    """계획 단계별 진행 상황 표시"""
    st.markdown("#### 📋 실행 계획 진행 상황")
    
    # 세션 상태에서 계획 정보 가져오기
    plan = st.session_state.get("current_plan", [])
    current_step = st.session_state.get("current_step", 0)
    step_results = st.session_state.get("step_results", {})
    
    if not plan:
        st.info("🔄 계획이 아직 생성되지 않았습니다.")
        return
    
    # 전체 진행률 계산
    completed_steps = len([r for r in step_results.values() if r.get("completed", False)])
    total_steps = len(plan)
    progress_percent = (completed_steps / total_steps) if total_steps > 0 else 0
    
    # 진행률 표시
    st.metric("전체 진행률", f"{completed_steps}/{total_steps}", f"{progress_percent:.1%}")
    st.progress(progress_percent)
    
    # 단계별 상태 표시
    for i, step in enumerate(plan):
        status_col, detail_col = st.columns([1, 3])
        
        with status_col:
            # 상태 아이콘 결정
            if i < current_step:
                status_icon = "✅"
                status_color = "green"
            elif i == current_step:
                status_icon = "🔄"
                status_color = "orange"
            else:
                status_icon = "⚪"
                status_color = "gray"
            
            st.markdown(f"**{status_icon} {i+1}**")
        
        with detail_col:
            # 단계 정보 표시
            task_type = step.get("type", "unknown")
            executor = TASK_EXECUTOR_MAPPING.get(task_type, "Unknown")
            
            # 단계 결과 표시
            if i in step_results:
                result = step_results[i]
                if result.get("completed"):
                    status_text = f"✅ **완료** ({result.get('execution_time', 0):.1f}s)"
                elif result.get("error"):
                    status_text = f"❌ **오류** - {result.get('error')[:50]}..."
                else:
                    status_text = "🔄 **진행 중**"
            else:
                status_text = "⏳ **대기 중**"
            
            st.markdown(f"""
            **{step.get('task', 'Unknown Task')}**  
            👤 {executor} | 🏷️ {task_type}  
            {status_text}
            """)
            
            # 에러가 있으면 상세 정보 표시
            if i in step_results and step_results[i].get("error"):
                with st.expander("❌ 오류 상세 정보"):
                    st.error(step_results[i]["error"])
                    
                    # 재시도 버튼
                    if st.button(f"🔄 단계 {i+1} 재시도", key=f"retry_{i}"):
                        st.session_state.current_step = i
                        st.session_state[f"retry_step_{i}"] = True
                        st.rerun()

def render_agent_activity_panel():
    """에이전트별 활동 상태 패널"""
    st.markdown("#### 🤖 에이전트 활동 상태")
    
    # 등록된 에이전트들
    executors = st.session_state.get("executors", {})
    current_step = st.session_state.get("current_step", 0)
    plan = st.session_state.get("current_plan", [])
    step_results = st.session_state.get("step_results", {})
    
    if not executors:
        st.warning("⚠️ 등록된 에이전트가 없습니다.")
        return
    
    # 각 에이전트의 상태 표시
    for executor_name, executor_config in executors.items():
        # 에이전트 상태 결정
        agent_status = _get_agent_status(executor_name, current_step, plan, step_results)
        
        # 상태별 색상 및 아이콘
        status_info = {
            "active": {"icon": "🟢", "color": "green", "text": "활성"},
            "waiting": {"icon": "⚪", "color": "gray", "text": "대기"},
            "completed": {"icon": "✅", "color": "blue", "text": "완료"},
            "error": {"icon": "🔴", "color": "red", "text": "오류"}
        }
        
        info = status_info.get(agent_status, status_info["waiting"])
        
        # 에이전트 카드 표시
        with st.container():
            st.markdown(f"""
            <div class="agent-status-card" style="
                border: 2px solid {info['color']};
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
                background-color: rgba(0,0,0,0.05);
            ">
                <h4>{info['icon']} {executor_name}</h4>
                <p><strong>상태:</strong> {info['text']}</p>
                <p><strong>도구:</strong> {len(executor_config.get('tools', []))}개</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 에이전트별 진행률 표시
        if agent_status == "active":
            # 활성 에이전트의 경우 진행률 애니메이션
            progress_value = (time.time() % 3) / 3  # 3초 주기 애니메이션
            st.progress(progress_value)

def _get_agent_status(executor_name: str, current_step: int, plan: List[Dict], 
                     step_results: Dict[int, Dict]) -> str:
    """에이전트 상태 결정"""
    if not plan or current_step >= len(plan):
        return "waiting"
    
    # 현재 단계의 실행자 확인
    current_task = plan[current_step]
    current_executor = TASK_EXECUTOR_MAPPING.get(current_task.get("type", ""), "")
    
    if current_executor == executor_name:
        # 현재 실행 중인 에이전트
        if current_step in step_results:
            result = step_results[current_step]
            if result.get("error"):
                return "error"
            elif result.get("completed"):
                return "completed"
        return "active"
    
    # 이전 단계에서 완료한 에이전트인지 확인
    for i in range(current_step):
        if i < len(plan):
            task = plan[i]
            if TASK_EXECUTOR_MAPPING.get(task.get("type", ""), "") == executor_name:
                if i in step_results and step_results[i].get("completed"):
                    return "completed"
                elif i in step_results and step_results[i].get("error"):
                    return "error"
    
    return "waiting"

def render_data_transformation_flow():
    """데이터 변환 과정 시각화"""
    st.markdown("#### 🔄 데이터 변환 플로우")
    
    transformations = data_lineage_tracker.transformations
    
    if len(transformations) <= 1:
        st.info("🔄 데이터 변환이 아직 진행되지 않았습니다.")
        return
    
    # Mermaid 다이어그램 생성
    mermaid_diagram = _create_mermaid_diagram(transformations)
    
    # Mermaid 다이어그램 표시
    st.markdown("**📊 변환 플로우 다이어그램**")
    st.markdown(f"""
    ```mermaid
    {mermaid_diagram}
    ```
    """)
    
    # 변환 상세 정보
    with st.expander("🔍 변환 상세 정보"):
        for i, transform in enumerate(transformations[1:], 1):  # 초기 로드 제외
            st.markdown(f"""
            **단계 {i}: {transform['operation']}**
            - 👤 실행자: {transform['executor']}
            - 📅 시간: {transform['timestamp'][:19]}
            - 📊 변경사항: 
              - 행 변화: {transform['changes']['rows_changed']:+d}
              - 열 변화: {transform['changes']['cols_changed']:+d}
              - 메모리 변화: {transform['changes']['memory_change']:+.2f}MB
            - 📝 설명: {transform['description']}
            """)
            st.divider()

def _create_mermaid_diagram(transformations: List[Dict]) -> str:
    """Mermaid 다이어그램 생성"""
    diagram_lines = ["graph TD"]
    
    for i, transform in enumerate(transformations):
        if i == 0:
            diagram_lines.append(f"    A{i}[원본 데이터<br/>{transform['metadata']['shape']}]")
        else:
            shape = transform['metadata']['shape']
            operation = transform['operation']
            executor = transform['executor']
            
            # 노드 생성
            diagram_lines.append(f"    A{i}[{operation}<br/>{executor}<br/>{shape}]")
            
            # 연결선 생성
            diagram_lines.append(f"    A{i-1} --> A{i}")
    
    return "\n".join(diagram_lines)

def render_system_metrics():
    """시스템 메트릭 표시"""
    st.markdown("#### 📊 시스템 메트릭")
    
    # 데이터 품질 점수
    if data_manager.is_data_loaded():
        data_quality = _calculate_data_quality_score()
        st.metric("데이터 품질", f"{data_quality:.1%}", 
                 delta=None, delta_color="normal")
    
    # 메모리 사용량
    memory_info = _get_memory_usage()
    st.metric("메모리 사용량", f"{memory_info['used_mb']:.1f}MB", 
             f"{memory_info['usage_percent']:.1f}%")
    
    # 변환 횟수
    transform_count = len(data_lineage_tracker.transformations) - 1  # 초기 로드 제외
    st.metric("데이터 변환", f"{transform_count}회")
    
    # 의심스러운 패턴 감지
    suspicious_patterns = data_lineage_tracker.detect_suspicious_patterns()
    if suspicious_patterns:
        st.warning(f"⚠️ {len(suspicious_patterns)}개의 의심스러운 패턴 감지")
        
        with st.expander("🔍 의심스러운 패턴 상세"):
            for pattern in suspicious_patterns:
                st.markdown(f"""
                **{pattern['type']}** (단계 {pattern['step']})
                - 👤 실행자: {pattern['executor']}
                - 📝 설명: {pattern['description']}
                """)
    else:
        st.success("✅ 데이터 무결성 양호")

def _calculate_data_quality_score() -> float:
    """데이터 품질 점수 계산"""
    if not data_manager.is_data_loaded():
        return 0.0
    
    data = data_manager.get_data()
    if data is None or data.empty:
        return 0.0
    
    # 품질 점수 계산 요소들
    scores = []
    
    # 1. 결측값 비율 (낮을수록 좋음)
    null_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
    null_score = 1.0 - min(null_ratio, 1.0)
    scores.append(null_score)
    
    # 2. 중복값 비율 (낮을수록 좋음)
    duplicate_ratio = data.duplicated().sum() / len(data)
    duplicate_score = 1.0 - min(duplicate_ratio, 1.0)
    scores.append(duplicate_score)
    
    # 3. 데이터 타입 일관성 (숫자 컬럼의 숫자 타입 비율)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        type_consistency = len(numeric_cols) / len(data.columns)
    else:
        type_consistency = 0.5  # 기본값
    scores.append(type_consistency)
    
    # 전체 점수 (가중평균)
    weights = [0.4, 0.3, 0.3]  # 결측값, 중복값, 타입 일관성
    total_score = sum(score * weight for score, weight in zip(scores, weights))
    
    return total_score

def _get_memory_usage() -> Dict[str, float]:
    """메모리 사용량 정보"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        used_mb = memory_info.rss / 1024 / 1024  # MB
        
        # 시스템 전체 메모리 대비 사용률
        system_memory = psutil.virtual_memory()
        usage_percent = (memory_info.rss / system_memory.total) * 100
        
        return {
            "used_mb": used_mb,
            "usage_percent": usage_percent
        }
    except ImportError:
        # psutil이 없으면 기본값 반환
        return {
            "used_mb": 50.0,
            "usage_percent": 5.0
        }

def render_real_time_alerts():
    """실시간 알림 표시"""
    st.markdown("#### 🚨 실시간 알림")
    
    alerts = []
    
    # 데이터 품질 알림
    if data_manager.is_data_loaded():
        quality_score = _calculate_data_quality_score()
        if quality_score < 0.7:
            alerts.append({
                "type": "warning",
                "message": f"데이터 품질이 낮습니다 ({quality_score:.1%})",
                "action": "데이터 전처리를 권장합니다"
            })
    
    # 의심스러운 패턴 알림
    suspicious_patterns = data_lineage_tracker.detect_suspicious_patterns()
    if suspicious_patterns:
        alerts.append({
            "type": "warning", 
            "message": f"{len(suspicious_patterns)}개의 의심스러운 패턴이 감지됨",
            "action": "데이터 변환 과정을 검토하세요"
        })
    
    # 메모리 사용량 알림
    memory_info = _get_memory_usage()
    if memory_info["usage_percent"] > 80:
        alerts.append({
            "type": "error",
            "message": f"메모리 사용량이 높습니다 ({memory_info['usage_percent']:.1f}%)",
            "action": "시스템 최적화가 필요합니다"
        })
    
    # 알림 표시
    if alerts:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"🔴 {alert['message']} - {alert['action']}")
            elif alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']} - {alert['action']}")
            else:
                st.info(f"ℹ️ {alert['message']} - {alert['action']}")
    else:
        st.success("✅ 모든 시스템이 정상 작동 중입니다")

def update_real_time_state(state: Dict[str, Any]):
    """실시간 상태 업데이트"""
    if "plan" in state:
        st.session_state.current_plan = state["plan"]
    if "current_step" in state:
        st.session_state.current_step = state["current_step"]
    if "step_results" in state:
        st.session_state.step_results = state["step_results"]

# CSS 스타일링
def apply_dashboard_styles():
    """대시보드 스타일 적용"""
    st.markdown("""
    <style>
    .agent-status-card {
        transition: all 0.3s ease;
    }
    .agent-status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .plan-step {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
    .plan-step.active {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .plan-step.completed {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .plan-step.error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)