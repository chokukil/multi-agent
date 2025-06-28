"""
Real-time Orchestration UI - A2A 오케스트레이션 실시간 모니터링 UI

Streamlit 고급 패턴 연구 결과를 바탕으로 구현:
- st.empty() 컨테이너를 활용한 실시간 업데이트
- st.columns()를 통한 반응형 레이아웃
- st.metric()으로 KPI 표시
- st.progress()와 st.status()를 활용한 진행 상황 시각화
- 멀티모달 아티팩트 렌더링 (텍스트, 데이터, 차트, 파일)
"""

import streamlit as st
import asyncio
import time
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import base64
import io

class RealTimeOrchestrationUI:
    """실시간 오케스트레이션 UI 컨트롤러"""
    
    def __init__(self):
        self.execution_container = None
        self.progress_container = None
        self.results_container = None
        self.metrics_container = None
        
    def initialize_ui(self):
        """UI 컨테이너 초기화"""
        st.markdown("### 🚀 실시간 오케스트레이션 실행")
        
        # 메트릭 영역
        self.metrics_container = st.container()
        
        # 진행 상황 영역
        self.progress_container = st.container()
        
        # 실행 상태 영역  
        self.execution_container = st.empty()
        
        # 결과 영역
        self.results_container = st.container()
        
    def display_execution_metrics(self, execution_data: Dict[str, Any]):
        """실행 메트릭 표시"""
        with self.metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "진행 단계",
                    f"{execution_data.get('steps_completed', 0)}/{execution_data.get('total_steps', 0)}",
                    delta=execution_data.get('steps_completed', 0)
                )
            
            with col2:
                completion_rate = (execution_data.get('steps_completed', 0) / max(execution_data.get('total_steps', 1), 1)) * 100
                st.metric(
                    "완료율",
                    f"{completion_rate:.1f}%",
                    delta=f"{completion_rate:.1f}%" if completion_rate > 0 else None
                )
            
            with col3:
                st.metric(
                    "실행 시간",
                    f"{execution_data.get('execution_time', 0):.1f}초",
                    delta=None
                )
            
            with col4:
                status = execution_data.get('status', 'unknown')
                status_emoji = {
                    'executing': '🔄',
                    'completed': '✅', 
                    'failed': '❌',
                    'unknown': '❓'
                }.get(status, '❓')
                
                st.metric(
                    "상태",
                    f"{status_emoji} {status.title()}",
                    delta=None
                )
    
    def display_progress_timeline(self, execution_data: Dict[str, Any]):
        """진행 상황 타임라인 표시"""
        with self.progress_container:
            st.markdown("#### 📋 실행 단계")
            
            steps = execution_data.get('step_results', [])
            total_steps = execution_data.get('total_steps', 0)
            current_step = execution_data.get('steps_completed', 0)
            
            # 전체 진행률
            if total_steps > 0:
                progress = current_step / total_steps
                st.progress(progress, text=f"전체 진행률: {progress*100:.1f}%")
            
            # 단계별 상태 표시
            for i, step_result in enumerate(steps):
                step_num = i + 1
                agent_name = step_result.get('agent_name', f'Step {step_num}')
                status = step_result.get('status', 'pending')
                
                # 상태별 아이콘과 색상
                if status == 'completed':
                    icon = "✅"
                    color = "green"
                elif status == 'failed':
                    icon = "❌" 
                    color = "red"
                elif status == 'executing':
                    icon = "🔄"
                    color = "blue"
                else:
                    icon = "⏳"
                    color = "gray"
                
                st.markdown(f"""
                <div style="
                    padding: 0.5rem 1rem;
                    margin: 0.25rem 0;
                    border-left: 4px solid {color};
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 0 8px 8px 0;
                ">
                    <strong>{icon} Step {step_num}: {agent_name}</strong><br/>
                    <small>Status: {status}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def display_current_execution(self, message: str, step_info: Optional[Dict] = None):
        """현재 실행 중인 작업 표시"""
        with self.execution_container.container():
            st.markdown("#### 🔄 현재 실행 중")
            
            # 현재 메시지
            st.info(message)
            
            # 단계 정보가 있으면 표시
            if step_info:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**에이전트**: {step_info.get('agent_name', 'Unknown')}")
                    st.markdown(f"**작업**: {step_info.get('task_description', 'Processing...')}")
                
                with col2:
                    # 스피너 애니메이션
                    with st.spinner("실행 중..."):
                        time.sleep(0.1)  # 짧은 지연으로 애니메이션 효과
    
    def display_execution_results(self, execution_data: Dict[str, Any]):
        """실행 결과 표시"""
        with self.results_container:
            st.markdown("#### 📊 실행 결과")
            
            if execution_data.get('status') == 'completed':
                st.success("🎉 오케스트레이션이 성공적으로 완료되었습니다!")
                
                # 최종 아티팩트 표시
                artifacts = execution_data.get('final_artifacts', [])
                if artifacts:
                    self._render_artifacts(artifacts)
                
                # 단계별 결과 요약
                self._display_step_summary(execution_data.get('step_results', []))
                
            elif execution_data.get('status') == 'failed':
                st.error(f"❌ 실행 실패: {execution_data.get('error', 'Unknown error')}")
                
                # 실패한 단계까지의 결과 표시
                completed_steps = [
                    step for step in execution_data.get('step_results', [])
                    if step.get('status') == 'completed'
                ]
                if completed_steps:
                    st.markdown("**완료된 단계들의 결과:**")
                    self._display_step_summary(completed_steps)
    
    def _render_artifacts(self, artifacts: List[Dict]):
        """아티팩트 렌더링"""
        st.markdown("### 🎯 생성된 아티팩트")
        
        for i, artifact in enumerate(artifacts):
            artifact_type = artifact.get('type', 'unknown')
            content = artifact.get('content', {})
            metadata = artifact.get('metadata', {})
            
            with st.expander(f"📄 아티팩트 {i+1}: {artifact_type.title()}", expanded=True):
                if artifact_type == 'text':
                    st.markdown(content)
                    
                elif artifact_type == 'data':
                    self._render_data_artifact(content, metadata)
                    
                elif artifact_type == 'file':
                    self._render_file_artifact(content, metadata)
                    
                else:
                    st.json(artifact)
    
    def _render_data_artifact(self, data: Dict, metadata: Dict):
        """데이터 아티팩트 렌더링"""
        try:
            # DataFrame인 경우
            if isinstance(data, dict) and 'columns' in data and 'data' in data:
                df = pd.DataFrame(data['data'], columns=data['columns'])
                st.dataframe(df, use_container_width=True)
                
                # 기본 통계 표시
                if len(df) > 0:
                    st.markdown("**기본 통계:**")
                    st.dataframe(df.describe(), use_container_width=True)
            
            # Plotly 차트인 경우
            elif isinstance(data, dict) and ('data' in data or 'layout' in data):
                try:
                    fig = go.Figure(data)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.json(data)
            
            # 일반 딕셔너리
            else:
                st.json(data)
                
        except Exception as e:
            st.error(f"데이터 렌더링 오류: {e}")
            st.json(data)
    
    def _render_file_artifact(self, file_info: Dict, metadata: Dict):
        """파일 아티팩트 렌더링"""
        try:
            file_name = file_info.get('name', 'unknown_file')
            mime_type = file_info.get('mimeType', 'application/octet-stream')
            
            if 'bytes' in file_info:
                # Base64 인코딩된 파일
                file_bytes = base64.b64decode(file_info['bytes'])
                
                if mime_type.startswith('image/'):
                    st.image(io.BytesIO(file_bytes), caption=file_name)
                elif mime_type == 'text/html':
                    st.components.v1.html(file_bytes.decode('utf-8'), height=400)
                else:
                    st.download_button(
                        label=f"📥 {file_name} 다운로드",
                        data=file_bytes,
                        file_name=file_name,
                        mime=mime_type
                    )
            
            elif 'uri' in file_info:
                # URI 참조
                st.markdown(f"**파일 위치**: {file_info['uri']}")
                
        except Exception as e:
            st.error(f"파일 렌더링 오류: {e}")
            st.json(file_info)
    
    def _display_step_summary(self, step_results: List[Dict]):
        """단계별 결과 요약"""
        st.markdown("### 📋 단계별 실행 요약")
        
        for i, step_result in enumerate(step_results):
            step_num = i + 1
            agent_name = step_result.get('agent_name', f'Step {step_num}')
            status = step_result.get('status', 'unknown')
            
            with st.expander(f"Step {step_num}: {agent_name} ({status})", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"**에이전트**: {agent_name}")
                    st.markdown(f"**상태**: {status}")
                    if step_result.get('error'):
                        st.error(f"오류: {step_result['error']}")
                
                with col2:
                    artifacts = step_result.get('artifacts', [])
                    st.markdown(f"**생성된 아티팩트**: {len(artifacts)}개")
                    
                    if artifacts:
                        for j, artifact in enumerate(artifacts[:3]):  # 최대 3개만 미리보기
                            artifact_type = artifact.get('type', 'unknown')
                            st.markdown(f"  - {artifact_type.title()} 아티팩트")

class StreamlitProgressCallback:
    """Streamlit용 진행 상황 콜백"""
    
    def __init__(self, ui_controller: RealTimeOrchestrationUI):
        self.ui = ui_controller
        self.current_step_info = {}
    
    def __call__(self, message: str, step_info: Optional[Dict] = None):
        """진행 상황 업데이트"""
        if step_info:
            self.current_step_info = step_info
        
        # UI 업데이트
        self.ui.display_current_execution(message, self.current_step_info)
        
        # Streamlit 강제 업데이트
        time.sleep(0.1)

def create_orchestration_ui() -> RealTimeOrchestrationUI:
    """오케스트레이션 UI 생성"""
    ui = RealTimeOrchestrationUI()
    ui.initialize_ui()
    return ui 