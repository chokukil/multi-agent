"""
아티팩트 표시 UI 통합 시스템

채팅 인터페이스와 아티팩트 렌더링 시스템을 통합하여 
실시간으로 에이전트 작업 결과를 표시하는 시스템
"""

import streamlit as st
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from modules.artifacts.a2a_artifact_extractor import A2AArtifactExtractor, Artifact, ArtifactType
from modules.ui.real_time_artifact_renderer import RealTimeArtifactRenderer

logger = logging.getLogger(__name__)

class ArtifactDisplayManager:
    """아티팩트 표시 관리자"""
    
    def __init__(self):
        self.extractor = A2AArtifactExtractor()
        self.renderer = RealTimeArtifactRenderer()
        self.active_containers = {}
        self.artifact_history = []
        
    def initialize_artifact_display_area(self):
        """아티팩트 표시 영역 초기화"""
        try:
            # 세션 상태 초기화
            if 'artifact_containers' not in st.session_state:
                st.session_state.artifact_containers = {}
            
            if 'artifact_history' not in st.session_state:
                st.session_state.artifact_history = []
            
            # 메인 아티팩트 표시 영역
            st.markdown("## 📦 생성된 아티팩트")
            
            # 아티팩트 필터 및 검색
            self._render_artifact_controls()
            
            # 아티팩트 표시 컨테이너
            self.main_container = st.container()
            
            # 실시간 업데이트를 위한 플레이스홀더
            self.status_placeholder = st.empty()
            
            logger.info("Artifact display area initialized")
            
        except Exception as e:
            logger.error(f"Error initializing artifact display area: {str(e)}")
            st.error(f"아티팩트 표시 영역 초기화 실패: {str(e)}")
    
    def _render_artifact_controls(self):
        """아티팩트 제어 패널 렌더링"""
        try:
            with st.expander("🔧 아티팩트 제어 패널", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 아티팩트 타입 필터
                    selected_types = st.multiselect(
                        "아티팩트 타입 필터",
                        options=[t.value for t in ArtifactType],
                        default=[t.value for t in ArtifactType],
                        key="artifact_type_filter"
                    )
                
                with col2:
                    # 에이전트 필터
                    agent_sources = list(set([
                        artifact.agent_source 
                        for artifact in st.session_state.get('artifact_history', [])
                    ]))
                    
                    selected_agents = st.multiselect(
                        "에이전트 필터",
                        options=agent_sources,
                        default=agent_sources,
                        key="artifact_agent_filter"
                    )
                
                with col3:
                    # 정렬 옵션
                    sort_option = st.selectbox(
                        "정렬 기준",
                        options=["최신순", "오래된순", "타입별", "에이전트별"],
                        key="artifact_sort_option"
                    )
                
                # 전체 삭제 버튼
                if st.button("🗑️ 모든 아티팩트 삭제", key="clear_all_artifacts"):
                    self.clear_all_artifacts()
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Error rendering artifact controls: {str(e)}")
    
    async def process_agent_response(self, response: Dict[str, Any], agent_source: str):
        """에이전트 응답을 처리하여 아티팩트 추출 및 표시"""
        try:
            logger.info(f"Processing agent response from {agent_source}")
            
            # 로딩 상태 표시
            with self.status_placeholder:
                st.info(f"🔄 {agent_source}에서 아티팩트 처리 중...")
            
            # 아티팩트 추출
            artifacts = await self.extractor.extract_from_a2a_response(response, agent_source)
            
            if not artifacts:
                with self.status_placeholder:
                    st.warning(f"⚠️ {agent_source}에서 아티팩트를 찾을 수 없습니다.")
                return
            
            # 아티팩트 히스토리에 추가
            st.session_state.artifact_history.extend(artifacts)
            
            # 실시간 렌더링
            await self._render_new_artifacts(artifacts)
            
            # 상태 업데이트
            with self.status_placeholder:
                st.success(f"✅ {agent_source}에서 {len(artifacts)}개 아티팩트 생성 완료")
            
            logger.info(f"Successfully processed {len(artifacts)} artifacts from {agent_source}")
            
        except Exception as e:
            logger.error(f"Error processing agent response: {str(e)}")
            with self.status_placeholder:
                st.error(f"❌ {agent_source} 아티팩트 처리 실패: {str(e)}")
    
    async def _render_new_artifacts(self, artifacts: List[Artifact]):
        """새로운 아티팩트들을 실시간으로 렌더링"""
        try:
            with self.main_container:
                for artifact in artifacts:
                    # 고유 컨테이너 키 생성
                    container_key = f"artifact_{artifact.id}"
                    
                    # 아티팩트 렌더링
                    self.renderer.render_artifact_immediately(artifact, container_key)
                    
                    # 컨테이너 추가 애니메이션 효과
                    self._add_artifact_animation(artifact)
                    
        except Exception as e:
            logger.error(f"Error rendering new artifacts: {str(e)}")
            st.error(f"새 아티팩트 렌더링 실패: {str(e)}")
    
    def _add_artifact_animation(self, artifact: Artifact):
        """아티팩트 추가 시 애니메이션 효과"""
        try:
            # CSS 애니메이션 효과
            st.markdown(f"""
            <style>
            .artifact-{artifact.id} {{
                animation: slideIn 0.5s ease-in-out;
                border-left: 4px solid #00ff88;
                padding-left: 10px;
                margin: 10px 0;
            }}
            
            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateY(-20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            </style>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error adding artifact animation: {str(e)}")
    
    def render_artifact_summary(self):
        """아티팩트 요약 정보 렌더링"""
        try:
            artifacts = st.session_state.get('artifact_history', [])
            
            if not artifacts:
                st.info("아직 생성된 아티팩트가 없습니다.")
                return
            
            # 통계 정보
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 아티팩트", len(artifacts))
            
            with col2:
                chart_count = len([a for a in artifacts if a.type == ArtifactType.PLOTLY_CHART])
                st.metric("차트", chart_count)
            
            with col3:
                table_count = len([a for a in artifacts if a.type == ArtifactType.DATAFRAME])
                st.metric("테이블", table_count)
            
            with col4:
                agent_count = len(set([a.agent_source for a in artifacts]))
                st.metric("활성 에이전트", agent_count)
            
            # 최근 활동
            if artifacts:
                latest_artifact = max(artifacts, key=lambda x: x.timestamp)
                st.caption(f"최근 활동: {latest_artifact.agent_source} - {latest_artifact.timestamp.strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.error(f"Error rendering artifact summary: {str(e)}")
    
    def render_filtered_artifacts(self):
        """필터링된 아티팩트 렌더링"""
        try:
            artifacts = st.session_state.get('artifact_history', [])
            
            if not artifacts:
                return
            
            # 필터 적용
            filtered_artifacts = self._apply_filters(artifacts)
            
            if not filtered_artifacts:
                st.info("필터 조건에 맞는 아티팩트가 없습니다.")
                return
            
            # 정렬 적용
            sorted_artifacts = self._apply_sorting(filtered_artifacts)
            
            # 렌더링
            for artifact in sorted_artifacts:
                self.renderer.render_artifact_immediately(artifact)
                
        except Exception as e:
            logger.error(f"Error rendering filtered artifacts: {str(e)}")
    
    def _apply_filters(self, artifacts: List[Artifact]) -> List[Artifact]:
        """아티팩트 필터 적용"""
        try:
            filtered = artifacts
            
            # 타입 필터
            selected_types = st.session_state.get('artifact_type_filter', [])
            if selected_types:
                filtered = [a for a in filtered if a.type.value in selected_types]
            
            # 에이전트 필터
            selected_agents = st.session_state.get('artifact_agent_filter', [])
            if selected_agents:
                filtered = [a for a in filtered if a.agent_source in selected_agents]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return artifacts
    
    def _apply_sorting(self, artifacts: List[Artifact]) -> List[Artifact]:
        """아티팩트 정렬 적용"""
        try:
            sort_option = st.session_state.get('artifact_sort_option', '최신순')
            
            if sort_option == "최신순":
                return sorted(artifacts, key=lambda x: x.timestamp, reverse=True)
            elif sort_option == "오래된순":
                return sorted(artifacts, key=lambda x: x.timestamp)
            elif sort_option == "타입별":
                return sorted(artifacts, key=lambda x: x.type.value)
            elif sort_option == "에이전트별":
                return sorted(artifacts, key=lambda x: x.agent_source)
            else:
                return artifacts
                
        except Exception as e:
            logger.error(f"Error applying sorting: {str(e)}")
            return artifacts
    
    def clear_all_artifacts(self):
        """모든 아티팩트 삭제"""
        try:
            st.session_state.artifact_history = []
            st.session_state.artifact_containers = {}
            self.active_containers = {}
            self.artifact_history = []
            
            logger.info("All artifacts cleared")
            
        except Exception as e:
            logger.error(f"Error clearing artifacts: {str(e)}")
    
    def export_artifacts(self, format: str = "json"):
        """아티팩트 내보내기"""
        try:
            artifacts = st.session_state.get('artifact_history', [])
            
            if not artifacts:
                st.warning("내보낼 아티팩트가 없습니다.")
                return
            
            if format == "json":
                export_data = []
                for artifact in artifacts:
                    export_data.append({
                        "id": artifact.id,
                        "type": artifact.type.value,
                        "agent_source": artifact.agent_source,
                        "timestamp": artifact.timestamp.isoformat(),
                        "metadata": artifact.metadata
                    })
                
                import json
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 JSON으로 내보내기",
                    data=json_data,
                    file_name=f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
        except Exception as e:
            logger.error(f"Error exporting artifacts: {str(e)}")
            st.error(f"아티팩트 내보내기 실패: {str(e)}")

class RealTimeProgressTracker:
    """실시간 진행 상황 추적기"""
    
    def __init__(self):
        self.active_agents = {}
        self.progress_container = None
        
    def initialize_progress_display(self):
        """진행 상황 표시 영역 초기화"""
        try:
            st.markdown("## 🔄 에이전트 작업 진행 상황")
            self.progress_container = st.container()
            
        except Exception as e:
            logger.error(f"Error initializing progress display: {str(e)}")
    
    def update_agent_status(self, agent_name: str, status: str, progress: float = 0.0):
        """에이전트 상태 업데이트"""
        try:
            self.active_agents[agent_name] = {
                "status": status,
                "progress": progress,
                "last_update": datetime.now()
            }
            
            self._render_progress_display()
            
        except Exception as e:
            logger.error(f"Error updating agent status: {str(e)}")
    
    def _render_progress_display(self):
        """진행 상황 표시 렌더링"""
        try:
            if not self.progress_container:
                return
            
            with self.progress_container:
                for agent_name, info in self.active_agents.items():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.text(f"🤖 {agent_name}: {info['status']}")
                        if info['progress'] > 0:
                            st.progress(info['progress'])
                    
                    with col2:
                        time_diff = datetime.now() - info['last_update']
                        st.caption(f"{time_diff.seconds}초 전")
                        
        except Exception as e:
            logger.error(f"Error rendering progress display: {str(e)}")
    
    def complete_agent_task(self, agent_name: str):
        """에이전트 작업 완료 처리"""
        try:
            if agent_name in self.active_agents:
                self.active_agents[agent_name]["status"] = "완료"
                self.active_agents[agent_name]["progress"] = 1.0
                self._render_progress_display()
                
        except Exception as e:
            logger.error(f"Error completing agent task: {str(e)}")

# 전역 인스턴스
artifact_display_manager = ArtifactDisplayManager()
progress_tracker = RealTimeProgressTracker()

def integrate_artifact_display_to_chat():
    """채팅 인터페이스에 아티팩트 표시 시스템 통합"""
    try:
        # 아티팩트 표시 영역 초기화
        artifact_display_manager.initialize_artifact_display_area()
        
        # 진행 상황 추적기 초기화
        progress_tracker.initialize_progress_display()
        
        # 아티팩트 요약 표시
        artifact_display_manager.render_artifact_summary()
        
        # 필터링된 아티팩트 표시
        artifact_display_manager.render_filtered_artifacts()
        
        logger.info("Artifact display integrated to chat interface")
        
    except Exception as e:
        logger.error(f"Error integrating artifact display: {str(e)}")
        st.error(f"아티팩트 표시 시스템 통합 실패: {str(e)}")

async def process_agent_artifact_response(response: Dict[str, Any], agent_source: str):
    """에이전트 아티팩트 응답 처리 (비동기)"""
    try:
        await artifact_display_manager.process_agent_response(response, agent_source)
        progress_tracker.complete_agent_task(agent_source)
        
    except Exception as e:
        logger.error(f"Error processing agent artifact response: {str(e)}")
        st.error(f"에이전트 응답 처리 실패: {str(e)}")

def update_agent_progress(agent_name: str, status: str, progress: float = 0.0):
    """에이전트 진행 상황 업데이트 (동기)"""
    try:
        progress_tracker.update_agent_status(agent_name, status, progress)
        
    except Exception as e:
        logger.error(f"Error updating agent progress: {str(e)}")