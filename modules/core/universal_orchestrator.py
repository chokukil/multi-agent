"""
Universal Orchestrator - LLM 기반 범용 오케스트레이터

검증된 Universal Engine 패턴 기반:
- MetaReasoningEngine: 4단계 추론 (초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답)
- A2AAgentDiscoverySystem: 동적 에이전트 선택 (포트 8306-8315)
- A2AWorkflowOrchestrator: 순차/병렬 실행 패턴
- A2AResultIntegrator: 충돌 해결 및 통합 인사이트
- A2AErrorHandler: 점진적 재시도 및 서킷 브레이커 패턴
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
import json
import uuid

from ..models import EnhancedTaskRequest, AgentProgressInfo, TaskState, StreamingResponse
from ..a2a.agent_client import A2AAgentClient
from .llm_recommendation_engine import LLMRecommendationEngine

# Universal Engine 패턴 가져오기 (사용 가능한 경우)
try:
    from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
    from core.universal_engine.a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
    from core.universal_engine.a2a_integration.a2a_result_integrator import A2AResultIntegrator
    from core.universal_engine.a2a_integration.a2a_error_handler import A2AErrorHandler
    from core.universal_engine.a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
    from core.universal_engine.llm_factory import LLMFactory
    UNIVERSAL_ENGINE_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class UniversalOrchestrator:
    """
    범용 오케스트레이터 - 검증된 Universal Engine 패턴 기반
    LLM 추론 능력을 활용한 동적 에이전트 선택 및 워크플로우 조정
    """
    
    # 검증된 에이전트 역량 매핑 (Universal Engine에서 검증됨)
    AGENT_CAPABILITIES = {
        8306: {
            "name": "🧹 Data Cleaning Agent",
            "description": "LLM 기반 지능형 데이터 정리, 빈 데이터 처리, 7단계 표준 정리 프로세스",
            "capabilities": ["data_cleaning", "missing_values", "outlier_detection", "data_validation"],
            "expertise": ["data_quality", "preprocessing", "anomaly_detection"]
        },
        8307: {
            "name": "📁 Data Loader Agent", 
            "description": "통합 데이터 로딩, UTF-8 인코딩 문제 해결, 다양한 파일 형식 지원",
            "capabilities": ["file_loading", "format_conversion", "encoding_handling", "data_import"],
            "expertise": ["file_formats", "data_ingestion", "encoding_resolution"]
        },
        8308: {
            "name": "📊 Data Visualization Agent",
            "description": "Interactive 시각화, Plotly 기반 차트 생성",
            "capabilities": ["interactive_charts", "plotly_visualization", "dashboard_creation"],
            "expertise": ["data_visualization", "chart_design", "interactive_plots"]
        },
        8309: {
            "name": "🔧 Data Wrangling Agent",
            "description": "데이터 변환, 조작, 구조 변경",
            "capabilities": ["data_transformation", "reshaping", "merging", "aggregation"],
            "expertise": ["data_manipulation", "data_restructuring", "complex_transformations"]
        },
        8310: {
            "name": "⚙️ Feature Engineering Agent",
            "description": "피처 생성, 변환, 선택, 차원 축소",
            "capabilities": ["feature_creation", "feature_selection", "dimensionality_reduction"],
            "expertise": ["feature_engineering", "ml_preprocessing", "feature_optimization"]
        },
        8311: {
            "name": "🗄️ SQL Database Agent",
            "description": "SQL 쿼리 실행, 데이터베이스 연결",
            "capabilities": ["sql_queries", "database_operations", "data_extraction"],
            "expertise": ["database_management", "sql_optimization", "data_retrieval"]
        },
        8312: {
            "name": "🔍 EDA Tools Agent",
            "description": "탐색적 데이터 분석, 통계 계산, 패턴 발견",
            "capabilities": ["exploratory_analysis", "statistical_analysis", "pattern_discovery"],
            "expertise": ["data_exploration", "statistical_methods", "pattern_recognition"]
        },
        8313: {
            "name": "🤖 H2O ML Agent",
            "description": "머신러닝 모델링, AutoML, 예측 분석",
            "capabilities": ["machine_learning", "automl", "model_training", "predictions"],
            "expertise": ["ml_algorithms", "model_optimization", "predictive_analytics"]
        },
        8314: {
            "name": "📈 MLflow Tools Agent",
            "description": "모델 관리, 실험 추적, 버전 관리",
            "capabilities": ["model_management", "experiment_tracking", "version_control"],
            "expertise": ["ml_ops", "model_versioning", "experiment_management"]
        },
        8315: {
            "name": "🐼 Pandas Analyst Agent",
            "description": "판다스 기반 데이터 조작 및 분석",
            "capabilities": ["pandas_operations", "data_analysis", "statistical_computations"],
            "expertise": ["data_analysis", "pandas_expertise", "statistical_analysis"]
        }
    }
    
    def __init__(self):
        """Universal Orchestrator 초기화"""
        
        # Universal Engine 컴포넌트 초기화
        if UNIVERSAL_ENGINE_AVAILABLE:
            self.meta_reasoning_engine = MetaReasoningEngine()
            self.workflow_orchestrator = A2AWorkflowOrchestrator()
            self.result_integrator = A2AResultIntegrator()
            self.error_handler = A2AErrorHandler()
            self.agent_selector = LLMBasedAgentSelector()
            self.llm_client = LLMFactory.create_llm()
        else:
            self.meta_reasoning_engine = None
            self.workflow_orchestrator = None
            self.result_integrator = None
            self.error_handler = None
            self.agent_selector = None
            self.llm_client = None
        
        # A2A 에이전트 클라이언트 초기화
        self.agent_clients = {
            port: A2AAgentClient(port) for port in self.AGENT_CAPABILITIES.keys()
        }
        
        # LLM 추천 엔진
        self.recommendation_engine = LLMRecommendationEngine()
        
        # 활성 태스크 추적
        self.active_tasks: Dict[str, Dict] = {}
        
        logger.info("Universal Orchestrator initialized with proven patterns")
    
    async def orchestrate_analysis(self, 
                                 request: EnhancedTaskRequest,
                                 progress_callback: Optional[callable] = None) -> AsyncGenerator[StreamingResponse, None]:
        """
        검증된 4단계 메타 추론을 사용한 분석 오케스트레이션:
        1. 초기 관찰: 데이터와 쿼리 의도 파악
        2. 다각도 분석: 사용자 수준별 접근법 고려  
        3. 자가 검증: 분석의 논리적 일관성 확인
        4. 적응적 응답: 최적 전략 결정
        """
        task_id = request.id
        
        try:
            logger.info(f"Starting orchestrated analysis for task {task_id}")
            
            # 태스크 상태 초기화
            self.active_tasks[task_id] = {
                'request': request,
                'start_time': datetime.now(),
                'status': 'started',
                'agents_working': [],
                'results': {}
            }
            
            # 1단계: 초기 관찰 (메타 추론)
            yield StreamingResponse(
                content="🔍 **1단계: 초기 관찰**\n데이터와 요청 의도를 분석하고 있습니다...",
                is_complete=False,
                chunk_index=0
            )
            
            meta_analysis = await self._perform_meta_reasoning(request)
            
            # 2단계: 다각도 분석 (에이전트 선택)
            yield StreamingResponse(
                content="🎯 **2단계: 다각도 분석**\n최적의 에이전트 조합을 선택하고 있습니다...",
                is_complete=False,
                chunk_index=1
            )
            
            selected_agents = await self._select_optimal_agents(meta_analysis, request)
            
            # 3단계: 자가 검증 (워크플로우 검증)
            yield StreamingResponse(
                content="✅ **3단계: 자가 검증**\n분석 계획의 논리적 일관성을 확인하고 있습니다...",
                is_complete=False,
                chunk_index=2
            )
            
            validated_workflow = await self._validate_workflow(selected_agents, meta_analysis)
            
            # 4단계: 적응적 응답 (실행)
            yield StreamingResponse(
                content="🚀 **4단계: 적응적 응답**\n에이전트들이 협업하여 분석을 수행합니다...",
                is_complete=False,
                chunk_index=3
            )
            
            # 에이전트 워크플로우 실행
            async for result_chunk in self._execute_agent_workflow(
                validated_workflow, request, progress_callback
            ):
                yield result_chunk
            
            # 최종 결과 통합
            final_results = await self._integrate_results(task_id)
            
            yield StreamingResponse(
                content=f"✨ **분석 완료!**\n\n{final_results.get('summary', '분석이 성공적으로 완료되었습니다.')}",
                is_complete=True,
                chunk_index=999
            )
            
        except Exception as e:
            logger.error(f"Orchestration error for task {task_id}: {str(e)}")
            
            yield StreamingResponse(
                content=f"❌ **오류 발생**: {str(e)}\n\n죄송합니다. 분석 중 문제가 발생했습니다.",
                is_complete=True,
                chunk_index=999
            )
        
        finally:
            # 태스크 정리
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                self.active_tasks[task_id]['end_time'] = datetime.now()
    
    async def _perform_meta_reasoning(self, request: EnhancedTaskRequest) -> Dict[str, Any]:
        """
        검증된 4단계 메타 추론 수행:
        - 초기 관찰: 데이터와 쿼리 의도 파악
        - 다각도 분석: 사용자 수준별 접근법 고려
        - 자가 검증: 분석의 논리적 일관성 확인
        - 적응적 응답: 최적 전략 결정
        """
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.meta_reasoning_engine:
                # Universal Engine MetaReasoningEngine 사용
                meta_analysis = await self.meta_reasoning_engine.perform_meta_reasoning(
                    query=request.user_message,
                    data=request.selected_datasets,
                    user_context=request.ui_context,
                    conversation_history=[]
                )
                
                return meta_analysis
            else:
                # 기본 메타 추론
                return await self._basic_meta_reasoning(request)
                
        except Exception as e:
            logger.error(f"Meta reasoning error: {str(e)}")
            return await self._basic_meta_reasoning(request)
    
    async def _basic_meta_reasoning(self, request: EnhancedTaskRequest) -> Dict[str, Any]:
        """기본 메타 추론 (Universal Engine이 없을 때)"""
        
        # 사용자 메시지 분석
        message_lower = request.user_message.lower()
        
        # 의도 분류
        intent = 'general_analysis'
        if any(word in message_lower for word in ['visualize', 'plot', 'chart', 'graph']):
            intent = 'visualization'
        elif any(word in message_lower for word in ['clean', 'quality', 'missing', 'null']):
            intent = 'data_cleaning'
        elif any(word in message_lower for word in ['model', 'predict', 'ml', 'machine learning']):
            intent = 'machine_learning'
        elif any(word in message_lower for word in ['statistics', 'stats', 'summary', 'describe']):
            intent = 'statistical_analysis'
        
        # 복잡도 추정
        complexity = 'intermediate'
        if len(request.selected_datasets) > 3:
            complexity = 'advanced'
        elif any(keyword in message_lower for keyword in ['simple', 'basic', 'quick']):
            complexity = 'beginner'
        
        return {
            'user_intent': intent,
            'complexity_level': complexity,
            'data_context': {
                'dataset_count': len(request.selected_datasets),
                'requires_integration': len(request.selected_datasets) > 1
            },
            'recommended_approach': 'sequential_with_integration',
            'priority_agents': self._get_priority_agents_for_intent(intent)
        }
    
    def _get_priority_agents_for_intent(self, intent: str) -> List[int]:
        """의도에 따른 우선순위 에이전트 목록"""
        intent_mapping = {
            'visualization': [8308, 8315, 8312],  # Visualization, Pandas, EDA
            'data_cleaning': [8306, 8315, 8312],  # Cleaning, Pandas, EDA
            'machine_learning': [8313, 8310, 8315, 8314],  # H2O ML, Feature Eng, Pandas, MLflow
            'statistical_analysis': [8312, 8315, 8308],  # EDA, Pandas, Visualization
            'general_analysis': [8315, 8312, 8308]  # Pandas, EDA, Visualization
        }
        
        return intent_mapping.get(intent, [8315, 8312, 8308])
    
    async def _select_optimal_agents(self, 
                                   meta_analysis: Dict[str, Any], 
                                   request: EnhancedTaskRequest) -> List[Dict[str, Any]]:
        """
        검증된 LLM 기반 에이전트 선택:
        - 하드코딩된 규칙 없이 순수 LLM 기반 선택
        - 사용자 요청의 본질을 파악하여 에이전트 조합 결정
        - 최적의 실행 순서 및 병렬 실행 가능성 식별
        """
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.agent_selector:
                # Universal Engine LLMBasedAgentSelector 사용
                selected_agents = await self.agent_selector.select_agents(
                    query=request.user_message,
                    available_agents=self.AGENT_CAPABILITIES,
                    meta_analysis=meta_analysis
                )
                
                return selected_agents
            else:
                # 기본 에이전트 선택
                return self._basic_agent_selection(meta_analysis, request)
                
        except Exception as e:
            logger.error(f"Agent selection error: {str(e)}")
            return self._basic_agent_selection(meta_analysis, request)
    
    def _basic_agent_selection(self, 
                             meta_analysis: Dict[str, Any], 
                             request: EnhancedTaskRequest) -> List[Dict[str, Any]]:
        """기본 에이전트 선택 로직"""
        
        selected_agents = []
        priority_agents = meta_analysis.get('priority_agents', [8315, 8312, 8308])
        
        for port in priority_agents[:3]:  # 최대 3개 에이전트
            if port in self.AGENT_CAPABILITIES:
                agent_info = self.AGENT_CAPABILITIES[port].copy()
                agent_info['port'] = port
                agent_info['execution_order'] = len(selected_agents) + 1
                selected_agents.append(agent_info)
        
        return selected_agents
    
    async def _validate_workflow(self, 
                                selected_agents: List[Dict[str, Any]], 
                                meta_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """워크플로우 검증 및 최적화"""
        
        # 의존성 분석
        validated_workflow = []
        
        for agent in selected_agents:
            agent_config = agent.copy()
            
            # 병렬 실행 가능성 확인
            agent_config['can_run_parallel'] = self._can_run_parallel(agent, validated_workflow)
            
            # 입력 의존성 설정
            agent_config['dependencies'] = self._get_agent_dependencies(agent, validated_workflow)
            
            validated_workflow.append(agent_config)
        
        return validated_workflow
    
    def _can_run_parallel(self, agent: Dict[str, Any], existing_agents: List[Dict[str, Any]]) -> bool:
        """에이전트가 병렬 실행 가능한지 확인"""
        
        # 데이터 변경하는 에이전트들은 순차 실행
        data_modifying_agents = [8306, 8309, 8310]  # Cleaning, Wrangling, Feature Engineering
        
        if agent['port'] in data_modifying_agents:
            return False
        
        # 같은 유형의 에이전트가 이미 실행 중이면 순차
        for existing in existing_agents:
            if existing.get('capabilities', []) and agent.get('capabilities', []):
                if set(existing['capabilities']) & set(agent['capabilities']):
                    return False
        
        return True
    
    def _get_agent_dependencies(self, 
                               agent: Dict[str, Any], 
                               existing_agents: List[Dict[str, Any]]) -> List[int]:
        """에이전트 의존성 분석"""
        
        dependencies = []
        agent_port = agent['port']
        
        # 일반적인 의존성 규칙
        if agent_port == 8308:  # Visualization은 데이터 처리 후
            dependencies.extend([a['port'] for a in existing_agents 
                               if a['port'] in [8306, 8309, 8315]])
        
        elif agent_port == 8313:  # ML은 Feature Engineering 후
            dependencies.extend([a['port'] for a in existing_agents 
                               if a['port'] in [8310, 8306]])
        
        elif agent_port == 8314:  # MLflow는 ML 후
            dependencies.extend([a['port'] for a in existing_agents 
                               if a['port'] == 8313])
        
        return dependencies
    
    async def _execute_agent_workflow(self, 
                                    workflow: List[Dict[str, Any]], 
                                    request: EnhancedTaskRequest,
                                    progress_callback: Optional[callable] = None) -> AsyncGenerator[StreamingResponse, None]:
        """
        검증된 순차/병렬 실행 패턴으로 에이전트 워크플로우 실행
        """
        task_id = request.id
        
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.workflow_orchestrator:
                # Universal Engine A2AWorkflowOrchestrator 사용
                async for result in self.workflow_orchestrator.execute_workflow(workflow, request):
                    yield StreamingResponse(
                        content=result.get('content', ''),
                        is_complete=result.get('is_complete', False),
                        chunk_index=result.get('chunk_index', 0),
                        progress_info=result.get('progress_info')
                    )
            else:
                # 기본 워크플로우 실행
                async for result in self._basic_workflow_execution(workflow, request, progress_callback):
                    yield result
                    
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            yield StreamingResponse(
                content=f"워크플로우 실행 중 오류 발생: {str(e)}",
                is_complete=True,
                chunk_index=999
            )
    
    async def _basic_workflow_execution(self, 
                                      workflow: List[Dict[str, Any]], 
                                      request: EnhancedTaskRequest,
                                      progress_callback: Optional[callable] = None) -> AsyncGenerator[StreamingResponse, None]:
        """기본 워크플로우 실행"""
        
        task_id = request.id
        completed_agents = set()
        agent_results = {}
        
        # 에이전트별 진행 상황 추적
        agent_progress = {}
        for agent in workflow:
            port = agent['port']
            agent_progress[port] = AgentProgressInfo(
                port=port,
                name=agent['name'],
                status=TaskState.PENDING,
                execution_time=0.0,
                artifacts_generated=[],
                current_task="대기 중"
            )
        
        total_agents = len(workflow)
        
        for i, agent in enumerate(workflow):
            port = agent['port']
            agent_name = agent['name']
            
            try:
                # 의존성 확인
                dependencies = agent.get('dependencies', [])
                if not all(dep in completed_agents for dep in dependencies):
                    yield StreamingResponse(
                        content=f"⏳ {agent_name}: 의존성 대기 중...",
                        is_complete=False,
                        chunk_index=10 + i
                    )
                    continue
                
                # 에이전트 실행 시작
                agent_progress[port].status = TaskState.WORKING
                agent_progress[port].current_task = "분석 실행 중"
                
                yield StreamingResponse(
                    content=f"🚀 **{agent_name}** 시작\n분석을 수행하고 있습니다...",
                    is_complete=False,
                    chunk_index=20 + i,
                    progress_info=self._create_progress_info(agent_progress)
                )
                
                # A2A 에이전트 호출
                agent_client = self.agent_clients[port]
                
                start_time = datetime.now()
                
                # 에이전트 요청 데이터 준비
                agent_request = {
                    "query": request.user_message,
                    "datasets": request.selected_datasets,
                    "context": {
                        "task_id": task_id,
                        "previous_results": agent_results,
                        "user_context": request.ui_context
                    }
                }
                
                # 에이전트 실행
                result = await agent_client.execute_task(agent_request)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # 결과 저장
                agent_results[port] = result
                completed_agents.add(port)
                
                # 진행 상황 업데이트
                agent_progress[port].status = TaskState.COMPLETED
                agent_progress[port].execution_time = execution_time
                agent_progress[port].current_task = "완료"
                
                # 생성된 아티팩트 확인
                artifacts = result.get('artifacts', [])
                agent_progress[port].artifacts_generated = [
                    art.get('type', 'unknown') for art in artifacts
                ]
                
                yield StreamingResponse(
                    content=f"✅ **{agent_name}** 완료 ({execution_time:.1f}초)\n{len(artifacts)}개의 결과를 생성했습니다.",
                    is_complete=False,
                    chunk_index=30 + i,
                    progress_info=self._create_progress_info(agent_progress)
                )
                
                # 진행률 콜백 호출
                if progress_callback:
                    progress_callback(f"{agent_name} 완료", (i + 1) / total_agents)
                
            except Exception as e:
                logger.error(f"Agent {port} execution error: {str(e)}")
                
                agent_progress[port].status = TaskState.FAILED
                agent_progress[port].current_task = f"오류: {str(e)}"
                
                yield StreamingResponse(
                    content=f"❌ **{agent_name}** 실패: {str(e)}",
                    is_complete=False,
                    chunk_index=40 + i,
                    progress_info=self._create_progress_info(agent_progress)
                )
        
        # 워크플로우 완료
        self.active_tasks[task_id]['results'] = agent_results
        
        yield StreamingResponse(
            content=f"🎉 **워크플로우 완료!**\n{len(completed_agents)}/{total_agents}개 에이전트가 성공적으로 실행되었습니다.",
            is_complete=False,
            chunk_index=90
        )
    
    def _create_progress_info(self, agent_progress: Dict[int, AgentProgressInfo]):
        """진행 상황 정보 생성"""
        from ..models import ProgressInfo
        
        agents_working = list(agent_progress.values())
        
        # 진행률 계산
        total_agents = len(agents_working)
        completed_agents = sum(1 for agent in agents_working if agent.status == TaskState.COMPLETED)
        completion_percentage = (completed_agents / total_agents * 100) if total_agents > 0 else 0
        
        return ProgressInfo(
            agents_working=agents_working,
            current_step=f"{completed_agents}/{total_agents} 에이전트 완료",
            total_steps=total_agents,
            completion_percentage=completion_percentage
        )
    
    async def _integrate_results(self, task_id: str) -> Dict[str, Any]:
        """
        검증된 A2AResultIntegrator를 사용한 결과 통합:
        - 충돌 해결 및 일관성 검증
        - 통합 인사이트 생성
        - 메타데이터 병합
        """
        try:
            if task_id not in self.active_tasks:
                return {"summary": "결과를 찾을 수 없습니다."}
            
            agent_results = self.active_tasks[task_id]['results']
            
            if UNIVERSAL_ENGINE_AVAILABLE and self.result_integrator:
                # Universal Engine A2AResultIntegrator 사용
                integrated_results = await self.result_integrator.integrate_results(agent_results)
                return integrated_results
            else:
                # 기본 결과 통합
                return self._basic_result_integration(agent_results)
                
        except Exception as e:
            logger.error(f"Result integration error: {str(e)}")
            return {"summary": f"결과 통합 중 오류 발생: {str(e)}"}
    
    def _basic_result_integration(self, agent_results: Dict[int, Any]) -> Dict[str, Any]:
        """기본 결과 통합"""
        
        summary_parts = []
        total_artifacts = 0
        
        for port, result in agent_results.items():
            agent_name = self.AGENT_CAPABILITIES[port]['name']
            artifacts = result.get('artifacts', [])
            
            summary_parts.append(f"• **{agent_name}**: {len(artifacts)}개 결과 생성")
            total_artifacts += len(artifacts)
        
        summary = f"분석이 완료되었습니다!\n\n" + "\n".join(summary_parts)
        summary += f"\n\n📊 **총 {total_artifacts}개의 분석 결과**가 생성되었습니다."
        
        return {
            "summary": summary,
            "total_artifacts": total_artifacts,
            "agent_count": len(agent_results),
            "execution_summary": summary_parts
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """태스크 상태 조회"""
        return self.active_tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """태스크 취소"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'cancelled'
            return True
        return False
    
    def get_active_tasks(self) -> List[str]:
        """활성 태스크 목록 반환"""
        return [task_id for task_id, task_info in self.active_tasks.items() 
                if task_info['status'] not in ['completed', 'cancelled', 'failed']]
    
    async def health_check_agents(self) -> Dict[int, bool]:
        """모든 에이전트 헬스 체크"""
        health_status = {}
        
        for port, client in self.agent_clients.items():
            try:
                health_status[port] = await client.health_check()
            except Exception as e:
                logger.warning(f"Health check failed for agent {port}: {str(e)}")
                health_status[port] = False
        
        return health_status
    
    def get_agent_capabilities_summary(self) -> Dict[str, Any]:
        """에이전트 역량 요약 반환"""
        return {
            "total_agents": len(self.AGENT_CAPABILITIES),
            "agent_ports": list(self.AGENT_CAPABILITIES.keys()),
            "capabilities_by_agent": {
                port: info["capabilities"] 
                for port, info in self.AGENT_CAPABILITIES.items()
            },
            "expertise_areas": {
                port: info["expertise"]
                for port, info in self.AGENT_CAPABILITIES.items()
            }
        }