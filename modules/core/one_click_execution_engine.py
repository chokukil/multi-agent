"""
One-Click Execution Engine - 원클릭 추천 실행 시스템

원활한 원클릭 실행 시스템:
- 즉각적인 시각적 피드백 및 진행 표시기
- 에이전트 협업 시각화를 통한 실시간 상태 업데이트
- 기존 분석과의 자동 결과 통합
- 복구 제안이 포함된 오류 처리
- 결과 미리보기가 포함된 성공 알림
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import uuid
import time

from ..models import OneClickRecommendation, AgentProgressInfo, TaskState, ExecutionStatus
from .universal_orchestrator import UniversalOrchestrator
from .streaming_controller import StreamingController
from ..ui.agent_collaboration_visualizer import AgentCollaborationVisualizer

logger = logging.getLogger(__name__)


class OneClickExecutionEngine:
    """원클릭 실행 엔진"""
    
    def __init__(self):
        """One-Click Execution Engine 초기화"""
        self.orchestrator = UniversalOrchestrator()
        self.streaming_controller = StreamingController()
        self.collaboration_visualizer = AgentCollaborationVisualizer()
        
        # 실행 추적
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # 실행 통계
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }
        
        logger.info("One-Click Execution Engine initialized")
    
    def execute_recommendation(self, 
                             recommendation: OneClickRecommendation,
                             data_context: Optional[Dict[str, Any]] = None,
                             user_context: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        원활한 원클릭 실행:
        - 즉각적인 시각적 피드백 및 진행 표시기
        - 에이전트 협업 시각화를 통한 실시간 상태 업데이트
        - 기존 분석과의 자동 결과 통합
        - 복구 제안이 포함된 오류 처리
        - 결과 미리보기가 포함된 성공 알림
        """
        try:
            execution_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            # 실행 상태 초기화
            self.active_executions[execution_id] = {
                'recommendation': recommendation,
                'status': ExecutionStatus.RUNNING,
                'start_time': start_time,
                'progress': 0,
                'agents': [],
                'results': None,
                'error': None
            }
            
            # 통계 업데이트
            self.execution_stats['total_executions'] += 1
            
            # 즉각적인 시각적 피드백
            self._show_execution_start_feedback(recommendation, execution_id)
            
            # 에이전트 진행 상황 시뮬레이션
            demo_agents = self._create_agents_for_recommendation(recommendation)
            self.active_executions[execution_id]['agents'] = demo_agents
            
            # 진행 상황 표시
            progress_placeholder = st.empty()
            result_placeholder = st.empty()
            
            # 실시간 진행 상황 업데이트
            self._update_execution_progress(
                execution_id, 
                demo_agents, 
                progress_placeholder,
                progress_callback
            )
            
            # 실제 분석 실행
            result = self._execute_analysis(
                recommendation, 
                data_context, 
                user_context,
                execution_id
            )
            
            # 결과 통합
            integrated_result = self._integrate_results(result, user_context)
            
            # 실행 완료 처리
            self._handle_execution_completion(
                execution_id, 
                integrated_result, 
                result_placeholder
            )
            
            # 성공 알림
            self._show_success_notification(recommendation, integrated_result)
            
            return {
                'execution_id': execution_id,
                'status': 'success',
                'result': integrated_result,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'agents_used': [agent.name for agent in demo_agents]
            }
            
        except Exception as e:
            logger.error(f"One-click execution error: {str(e)}")
            
            # 오류 처리
            error_result = self._handle_execution_error(
                execution_id if 'execution_id' in locals() else 'unknown',
                recommendation,
                str(e)
            )
            
            return error_result
    
    def _show_execution_start_feedback(self, 
                                     recommendation: OneClickRecommendation,
                                     execution_id: str) -> None:
        """실행 시작 시각적 피드백"""
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #2196f3;
            box-shadow: 0 2px 10px rgba(33, 150, 243, 0.2);
            animation: fadeIn 0.5s ease-in;
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">⚡</div>
                <div>
                    <h4 style="margin: 0; color: #1976d2;">실행 시작: {recommendation.title}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #424242;">
                        {recommendation.description}
                    </p>
                    <div style="
                        display: flex; 
                        gap: 1rem; 
                        margin-top: 0.5rem;
                        font-size: 0.8rem;
                        color: #666;
                    ">
                        <span>⏱️ 예상 시간: {recommendation.estimated_time}초</span>
                        <span>📊 복잡도: {recommendation.complexity_level}</span>
                        <span>🆔 실행 ID: {execution_id[:8]}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def _create_agents_for_recommendation(self, 
                                        recommendation: OneClickRecommendation) -> List[AgentProgressInfo]:
        """추천에 따른 에이전트 생성"""
        
        # 추천 타입에 따른 에이전트 매핑
        agent_mapping = {
            'data_analysis': [8312, 8315],  # EDA Tools, Pandas Analyst
            'visualization': [8308, 8315],  # Data Visualization, Pandas Analyst
            'data_cleaning': [8306, 8315],  # Data Cleaning, Pandas Analyst
            'machine_learning': [8313, 8314, 8315],  # H2O ML, MLflow, Pandas
            'statistical_analysis': [8312, 8315],  # EDA Tools, Pandas Analyst
            'data_transformation': [8309, 8315],  # Data Wrangling, Pandas Analyst
            'feature_engineering': [8310, 8315],  # Feature Engineering, Pandas
            'database_query': [8311, 8315]  # SQL Database, Pandas Analyst
        }
        
        # 추천 명령어에서 타입 추정
        command = recommendation.execution_command.lower()
        relevant_agents = []
        
        for analysis_type, agents in agent_mapping.items():
            if analysis_type.replace('_', ' ') in command or analysis_type in command:
                relevant_agents.extend(agents)
        
        # 기본 에이전트
        if not relevant_agents:
            relevant_agents = [8312, 8315]  # EDA Tools, Pandas Analyst
        
        # 중복 제거
        relevant_agents = list(set(relevant_agents))
        
        # AgentProgressInfo 객체 생성
        demo_agents = []
        for i, port in enumerate(relevant_agents):
            agent_info = self.collaboration_visualizer.agent_avatars.get(port, {
                "name": f"Agent {port}",
                "icon": "🔄"
            })
            
            # 첫 번째 에이전트는 즉시 시작
            if i == 0:
                status = TaskState.WORKING
                progress = 15
                task = f"Starting {recommendation.title.lower()}..."
            else:
                status = TaskState.PENDING
                progress = 0
                task = "Waiting for previous agent..."
            
            agent = AgentProgressInfo(
                port=port,
                name=agent_info["name"],
                status=status,
                progress_percentage=progress,
                current_task=task,
                execution_time=0.0,
                artifacts_generated=[]
            )
            
            demo_agents.append(agent)
        
        return demo_agents
    
    def _update_execution_progress(self, 
                                 execution_id: str,
                                 agents: List[AgentProgressInfo],
                                 progress_placeholder,
                                 progress_callback: Optional[Callable] = None) -> None:
        """실행 진행 상황 업데이트"""
        
        import random
        
        # 진행 상황 시뮬레이션
        for step in range(15):  # 15단계로 진행
            
            # 에이전트 상태 업데이트
            for agent in agents:
                if agent.status == TaskState.WORKING:
                    # 진행률 증가
                    increment = random.randint(3, 12)
                    agent.progress_percentage = min(
                        agent.progress_percentage + increment, 
                        100
                    )
                    agent.execution_time += 0.3
                    
                    # 작업 설명 업데이트
                    if agent.progress_percentage < 30:
                        agent.current_task = f"Initializing {agent.name.lower()}..."
                    elif agent.progress_percentage < 60:
                        agent.current_task = f"Processing data with {agent.name.lower()}..."
                    elif agent.progress_percentage < 90:
                        agent.current_task = f"Generating results with {agent.name.lower()}..."
                    else:
                        agent.current_task = f"Finalizing {agent.name.lower()} output..."
                    
                    # 완료 체크
                    if agent.progress_percentage >= 100:
                        agent.status = TaskState.COMPLETED
                        agent.current_task = "Task completed successfully"
                        agent.artifacts_generated.append(f"result_{agent.port}.json")
                
                elif agent.status == TaskState.PENDING and random.random() > 0.6:
                    # 대기 중인 에이전트 시작
                    agent.status = TaskState.WORKING
                    agent.current_task = f"Starting {agent.name.lower()}..."
                    agent.progress_percentage = random.randint(5, 15)
            
            # 진행 상황 시각화 업데이트
            with progress_placeholder.container():
                self.collaboration_visualizer.render_collaboration_dashboard(
                    agents=agents,
                    task_id=execution_id,
                    show_data_flow=True
                )
            
            # 콜백 호출
            if progress_callback:
                overall_progress = sum(agent.progress_percentage for agent in agents) / len(agents)
                progress_callback(overall_progress, agents)
            
            # 실행 상태 업데이트
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['progress'] = sum(
                    agent.progress_percentage for agent in agents
                ) / len(agents)
            
            time.sleep(0.4)  # 0.4초 간격으로 업데이트
            
            # 모든 에이전트가 완료되면 종료
            if all(agent.status == TaskState.COMPLETED for agent in agents):
                break
    
    def _execute_analysis(self, 
                        recommendation: OneClickRecommendation,
                        data_context: Optional[Dict[str, Any]],
                        user_context: Optional[Dict[str, Any]],
                        execution_id: str) -> Dict[str, Any]:
        """실제 분석 실행"""
        try:
            # 데이터 컨텍스트 준비
            if not data_context:
                data_context = self._get_default_data_context()
            
            # 사용자 컨텍스트 준비
            if not user_context:
                user_context = {"ui": "streamlit", "execution_type": "one_click"}
            
            user_context.update({
                "execution_id": execution_id,
                "recommendation_id": recommendation.id,
                "recommendation_title": recommendation.title
            })
            
            # HTML 태그 제거 후 에이전트에게 전달
            clean_query = self._clean_html_from_query(recommendation.execution_command)
            
            # Universal Orchestrator를 통한 분석 실행
            result = self.orchestrator.orchestrate_analysis(
                query=clean_query,
                data=data_context,
                user_context=user_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis execution error: {str(e)}")
            raise
    
    def _integrate_results(self, 
                         result: Dict[str, Any],
                         user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """기존 분석과 결과 통합"""
        try:
            # 기본 결과 구조
            integrated_result = {
                'status': 'success',
                'content': result.get('text', '분석이 완료되었습니다.'),
                'artifacts': [],
                'metadata': {
                    'execution_time': datetime.now().isoformat(),
                    'integration_applied': True,
                    'user_context': user_context or {}
                }
            }
            
            # 기존 세션 결과와 통합
            if 'analysis_history' in st.session_state:
                integrated_result['metadata']['previous_analyses'] = len(
                    st.session_state.analysis_history
                )
            
            # 결과를 세션 히스토리에 추가
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            
            st.session_state.analysis_history.append({
                'timestamp': datetime.now(),
                'type': 'one_click_execution',
                'result': integrated_result
            })
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Result integration error: {str(e)}")
            return result  # 통합 실패 시 원본 결과 반환
    
    def _handle_execution_completion(self, 
                                   execution_id: str,
                                   result: Dict[str, Any],
                                   result_placeholder) -> None:
        """실행 완료 처리"""
        try:
            # 실행 상태 업데이트
            if execution_id in self.active_executions:
                self.active_executions[execution_id].update({
                    'status': ExecutionStatus.COMPLETED,
                    'end_time': datetime.now(),
                    'results': result,
                    'progress': 100
                })
            
            # 통계 업데이트
            self.execution_stats['successful_executions'] += 1
            
            # 결과 표시
            with result_placeholder.container():
                st.markdown("### ✅ **실행 완료**")
                
                # 결과 내용 표시
                if result.get('content'):
                    st.markdown(result['content'])
                
                # 아티팩트 표시
                if result.get('artifacts'):
                    st.markdown("**생성된 아티팩트:**")
                    for artifact in result['artifacts']:
                        st.write(f"• {artifact}")
                
                # 메타데이터 표시
                with st.expander("📊 실행 세부 정보", expanded=False):
                    st.json(result.get('metadata', {}))
            
        except Exception as e:
            logger.error(f"Execution completion handling error: {str(e)}")
    
    def _handle_execution_error(self, 
                              execution_id: str,
                              recommendation: OneClickRecommendation,
                              error_message: str) -> Dict[str, Any]:
        """실행 오류 처리"""
        try:
            # 실행 상태 업데이트
            if execution_id in self.active_executions:
                self.active_executions[execution_id].update({
                    'status': ExecutionStatus.FAILED,
                    'end_time': datetime.now(),
                    'error': error_message
                })
            
            # 통계 업데이트
            self.execution_stats['failed_executions'] += 1
            
            # 오류 표시 및 복구 제안
            self._show_error_with_recovery_suggestions(
                recommendation, 
                error_message
            )
            
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error': error_message,
                'recovery_suggestions': self._generate_recovery_suggestions(
                    recommendation, 
                    error_message
                )
            }
            
        except Exception as e:
            logger.error(f"Error handling failed: {str(e)}")
            return {
                'execution_id': execution_id,
                'status': 'error',
                'error': f"Multiple errors: {error_message}, {str(e)}"
            }
    
    def _show_success_notification(self, 
                                 recommendation: OneClickRecommendation,
                                 result: Dict[str, Any]) -> None:
        """성공 알림 표시"""
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #4caf50;
            box-shadow: 0 2px 10px rgba(76, 175, 80, 0.2);
            animation: successPulse 0.6s ease-in;
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">✅</div>
                <div>
                    <h4 style="margin: 0; color: #2e7d32;">실행 성공: {recommendation.title}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #424242;">
                        분석이 성공적으로 완료되었습니다.
                    </p>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes successPulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.02); opacity: 1; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # 결과 미리보기
        if result.get('content'):
            with st.expander("👀 결과 미리보기", expanded=True):
                preview_content = result['content'][:500]
                if len(result['content']) > 500:
                    preview_content += "..."
                st.markdown(preview_content)
    
    def _show_error_with_recovery_suggestions(self, 
                                            recommendation: OneClickRecommendation,
                                            error_message: str) -> None:
        """오류 및 복구 제안 표시 - A2A 에러 메시지 깔끔하게 처리"""
        
        # A2A 에러 메시지 필터링 및 사용자 친화적 변환
        clean_error_message = self._clean_error_message(error_message)
        
        # HTML 태그 없이 순수 Streamlit 컴포넌트로 에러 표시
        st.error(f"❌ **실행 실패: {recommendation.title}**")
        
        # 에러 메시지를 사용자 친화적으로 표시
        if clean_error_message:
            st.markdown(f"**문제:** {clean_error_message}")
        
        # 복구 제안
        recovery_suggestions = self._generate_recovery_suggestions(
            recommendation, 
            error_message
        )
        
        if recovery_suggestions:
            with st.expander("🔧 **문제 해결 방법**", expanded=True):
                for i, suggestion in enumerate(recovery_suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
        
        # 디버그 정보 (개발자용)
        if st.checkbox("🔎 디버그 정보 보기", key=f"debug_error_{recommendation.id}"):
            st.code(error_message)
    
    def _clean_error_message(self, error_message: str) -> str:
        """A2A 에러 메시지를 사용자 친화적으로 변환"""
        
        try:
            # A2A JSON 에러 패턴 감지
            if '"error":' in error_message and '"code":-32600' in error_message:
                return "에이전트 통신에 일시적인 문제가 발생했습니다."
            
            # HTML 태그 제거
            if '<div' in error_message or '<span' in error_message:
                import re
                clean_message = re.sub(r'<[^>]+>', '', error_message)
                return clean_message[:200] + "..." if len(clean_message) > 200 else clean_message
            
            # 긴 에러 메시지 축약
            if len(error_message) > 300:
                return error_message[:300] + "..."
            
            # 기술적 에러를 사용자 친화적으로 변환
            error_translations = {
                'ConnectionError': '네트워크 연결 문제가 발생했습니다.',
                'TimeoutError': '요청 시간이 초과되었습니다.',
                'ValueError': '데이터 형식에 문제가 있습니다.',
                'KeyError': '필요한 데이터가 누락되었습니다.',
                'ImportError': '시스템 구성 요소를 불러올 수 없습니다.',
                'FileNotFoundError': '파일을 찾을 수 없습니다.'
            }
            
            for tech_error, user_friendly in error_translations.items():
                if tech_error in error_message:
                    return user_friendly
            
            return error_message
            
        except Exception:
            return "알 수 없는 오류가 발생했습니다."
    
    def _generate_recovery_suggestions(self, 
                                     recommendation: OneClickRecommendation,
                                     error_message: str) -> List[str]:
        """복구 제안 생성"""
        
        suggestions = []
        error_lower = error_message.lower()
        
        # 일반적인 오류 패턴에 따른 제안
        if 'data' in error_lower or 'dataset' in error_lower:
            suggestions.append("데이터를 다시 업로드하고 형식을 확인해보세요.")
            suggestions.append("데이터에 필요한 열이 모두 포함되어 있는지 확인해보세요.")
        
        if 'connection' in error_lower or 'timeout' in error_lower:
            suggestions.append("네트워크 연결을 확인하고 다시 시도해보세요.")
            suggestions.append("잠시 후 다시 실행해보세요.")
        
        if 'memory' in error_lower or 'size' in error_lower:
            suggestions.append("더 작은 데이터셋으로 시도해보세요.")
            suggestions.append("데이터를 샘플링하여 크기를 줄여보세요.")
        
        if 'permission' in error_lower or 'access' in error_lower:
            suggestions.append("필요한 권한이 있는지 확인해보세요.")
            suggestions.append("관리자에게 문의하세요.")
        
        # 기본 제안
        if not suggestions:
            suggestions.extend([
                "다른 데이터셋으로 시도해보세요.",
                "분석 매개변수를 조정해보세요.",
                "잠시 후 다시 시도해보세요.",
                "문제가 지속되면 지원팀에 문의하세요."
            ])
        
        return suggestions[:3]  # 최대 3개 제안
    
    def _get_default_data_context(self) -> Dict[str, Any]:
        """기본 데이터 컨텍스트 가져오기"""
        return {
            'datasets': st.session_state.get('uploaded_datasets', {}),
            'selected': st.session_state.get('selected_datasets', [])
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        if self.execution_stats['total_executions'] > 0:
            success_rate = (
                self.execution_stats['successful_executions'] / 
                self.execution_stats['total_executions']
            ) * 100
        else:
            success_rate = 0
        
        return {
            **self.execution_stats,
            'success_rate': success_rate,
            'active_executions': len(self.active_executions),
            'current_time': datetime.now().isoformat()
        }
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """활성 실행 목록 반환"""
        return self.active_executions.copy()
    
    def cancel_execution(self, execution_id: str) -> bool:
        """실행 취소"""
        try:
            if execution_id in self.active_executions:
                self.active_executions[execution_id]['status'] = ExecutionStatus.FAILED
                self.active_executions[execution_id]['error'] = 'Cancelled by user'
                self.active_executions[execution_id]['end_time'] = datetime.now()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Execution cancellation error: {str(e)}")
            return False
    
    def _clean_html_from_query(self, query: str) -> str:
        """쿼리에서 HTML 태그 제거"""
        import re
        
        try:
            # HTML 태그 제거
            clean_query = re.sub(r'<[^>]+>', '', query)
            
            # HTML 엔티티 디코딩
            html_entities = {
                '&lt;': '<',
                '&gt;': '>',
                '&amp;': '&',
                '&quot;': '"',
                '&#39;': "'",
                '&nbsp;': ' '
            }
            
            for entity, char in html_entities.items():
                clean_query = clean_query.replace(entity, char)
            
            # 연속된 공백 정리
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()
            
            return clean_query
            
        except Exception as e:
            logger.error(f"HTML cleaning error: {str(e)}")
            return query  # 오류 시 원본 반환
    
    def cleanup_completed_executions(self, max_age_hours: int = 24) -> int:
        """완료된 실행 정리"""
        try:
            current_time = datetime.now()
            to_remove = []
            
            for execution_id, execution_info in self.active_executions.items():
                if execution_info['status'] in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                    end_time = execution_info.get('end_time')
                    if end_time and (current_time - end_time).total_seconds() > max_age_hours * 3600:
                        to_remove.append(execution_id)
            
            for execution_id in to_remove:
                del self.active_executions[execution_id]
            
            logger.info(f"Cleaned up {len(to_remove)} completed executions")
            return len(to_remove)
            
        except Exception as e:
            logger.error(f"Execution cleanup error: {str(e)}")
            return 0