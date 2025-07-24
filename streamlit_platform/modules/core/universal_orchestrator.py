"""
Universal Orchestrator for Cherry AI Streamlit Platform
Based on proven MetaReasoningEngine + LLMBasedAgentSelector patterns from Universal Engine
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import proven Universal Engine patterns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core'))

from universal_engine.meta_reasoning_engine import MetaReasoningEngine
from universal_engine.a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
from universal_engine.a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from universal_engine.a2a_integration.a2a_result_integrator import A2AResultIntegrator
from universal_engine.a2a_integration.a2a_error_handler import A2AErrorHandler

from .data_models import (
    TaskRequest, StreamingResponse, AgentProgressInfo, 
    EnhancedArtifact, OneClickRecommendation, UserContext,
    TaskState, UserExpertiseLevel
)

logger = logging.getLogger(__name__)

class UniversalOrchestrator:
    """
    Universal Orchestrator using proven Universal Engine patterns
    - MetaReasoningEngine: 4-stage reasoning (초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답)
    - LLMBasedAgentSelector: Zero-hardcoding agent selection
    - A2AWorkflowOrchestrator: Sequential/parallel execution patterns
    """
    
    # Validated agent capabilities from Universal Engine
    AGENT_CAPABILITIES = {
        8306: "🧹 LLM 기반 지능형 데이터 정리, 빈 데이터 처리, 7단계 표준 정리 프로세스",
        8307: "📁 통합 데이터 로딩, UTF-8 인코딩 문제 해결, 다양한 파일 형식 지원", 
        8308: "📊 Interactive 시각화, Plotly 기반 차트 생성",
        8309: "🔧 데이터 변환, 조작, 구조 변경",
        8310: "⚙️ 피처 생성, 변환, 선택, 차원 축소",
        8311: "🗄️ SQL 쿼리 실행, 데이터베이스 연결",
        8312: "🔍 탐색적 데이터 분석, 통계 계산, 패턴 발견",
        8313: "🤖 머신러닝 모델링, AutoML, 예측 분석",
        8314: "📈 모델 관리, 실험 추적, 버전 관리",
        8315: "🐼 판다스 기반 데이터 조작 및 분석"
    }
    
    def __init__(self):
        """Initialize Universal Orchestrator with proven patterns"""
        
        # Core proven components from Universal Engine
        self.meta_reasoning_engine = MetaReasoningEngine()
        self.agent_selector = LLMBasedAgentSelector()
        self.workflow_orchestrator = A2AWorkflowOrchestrator()
        self.agent_discovery = A2AAgentDiscoverySystem()
        self.result_integrator = A2AResultIntegrator()
        self.error_handler = A2AErrorHandler()
        
        # Available agents (validated ports from Universal Engine)
        self.available_agents = self._load_validated_agents()
        
        logger.info("UniversalOrchestrator initialized with proven Universal Engine patterns")
    
    def _load_validated_agents(self) -> Dict[int, Dict]:
        """Load validated agent configurations from Universal Engine"""
        agents = {}
        for port, capability in self.AGENT_CAPABILITIES.items():
            agents[port] = {
                "port": port,
                "name": self._get_agent_name(port),
                "capability": capability,
                "status": "available",
                "health_check_url": f"http://localhost:{port}/.well-known/agent.json"
            }
        return agents
    
    def _get_agent_name(self, port: int) -> str:
        """Get agent name from port using validated mapping"""
        agent_names = {
            8306: "Data Cleaning Agent",
            8307: "Data Loader Agent", 
            8308: "Data Visualization Agent",
            8309: "Data Wrangling Agent",
            8310: "Feature Engineering Agent",
            8311: "SQL Database Agent",
            8312: "EDA Tools Agent",
            8313: "H2O ML Agent",
            8314: "MLflow Tools Agent",
            8315: "Pandas Analyst Agent"
        }
        return agent_names.get(port, f"Unknown Agent {port}")
    
    async def perform_meta_reasoning(self, 
                                   query: str, 
                                   data: Any, 
                                   user_context: UserContext,
                                   conversation_history: List) -> Dict:
        """
        Perform 4-stage meta-reasoning using proven Universal Engine patterns
        1. 초기 관찰: 데이터와 쿼리 의도 파악
        2. 다각도 분석: 사용자 수준별 접근법 고려  
        3. 자가 검증: 분석의 논리적 일관성 확인
        4. 적응적 응답: 최적 전략 결정
        """
        
        try:
            logger.info(f"Starting meta-reasoning for query: {query[:100]}...")
            
            # Use proven MetaReasoningEngine patterns
            meta_analysis = await self.meta_reasoning_engine.perform_comprehensive_reasoning(
                query=query,
                data_context=data,
                user_expertise_level=user_context.expertise_level.value,
                conversation_history=conversation_history,
                domain_context=user_context.domain_knowledge
            )
            
            # Extract reasoning stages (proven pattern from Universal Engine)
            reasoning_result = {
                "initial_observation": meta_analysis.get("stage_1_observation", ""),
                "multi_perspective_analysis": meta_analysis.get("stage_2_analysis", ""),
                "self_verification": meta_analysis.get("stage_3_verification", ""),
                "adaptive_strategy": meta_analysis.get("stage_4_strategy", ""),
                "confidence_scores": meta_analysis.get("confidence_scores", {}),
                "user_adaptation_insights": meta_analysis.get("user_adaptation", {}),
                "recommended_approach": meta_analysis.get("recommended_approach", ""),
                "complexity_assessment": meta_analysis.get("complexity_assessment", "medium")
            }
            
            logger.info("Meta-reasoning completed successfully")
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Meta-reasoning failed: {e}")
            # Fallback to basic analysis (proven error handling pattern)
            return {
                "initial_observation": f"Basic analysis of: {query}",
                "multi_perspective_analysis": "Multiple approaches available",
                "self_verification": "Standard validation needed",
                "adaptive_strategy": "Use conservative approach",
                "confidence_scores": {"overall": 0.7},
                "user_adaptation_insights": {"level": user_context.expertise_level.value},
                "recommended_approach": "sequential_analysis",
                "complexity_assessment": "medium"
            }
    
    async def select_optimal_agents(self, 
                                  meta_analysis: Dict, 
                                  available_agents: Dict) -> List[Dict]:
        """
        Select optimal agents using validated LLM-based selection patterns
        - 하드코딩된 규칙 없이 순수 LLM 기반 선택
        - 사용자 요청의 본질을 파악하여 에이전트 조합 결정
        - 최적의 실행 순서 및 병렬 실행 가능성 식별
        """
        
        try:
            logger.info("Starting LLM-based agent selection...")
            
            # Use proven LLMBasedAgentSelector patterns
            selected_agents = await self.agent_selector.select_agents_for_task(
                meta_analysis=meta_analysis,
                available_agents=available_agents,
                agent_capabilities=self.AGENT_CAPABILITIES
            )
            
            # Validate selection results (proven pattern)
            validated_agents = []
            for agent in selected_agents:
                if agent.get("port") in self.available_agents:
                    agent_info = {
                        "port": agent["port"], 
                        "name": self._get_agent_name(agent["port"]),
                        "reason": agent.get("selection_reason", "LLM recommended"),
                        "execution_order": agent.get("execution_order", 1),
                        "parallel_group": agent.get("parallel_group", 0),
                        "expected_output": agent.get("expected_output", ""),
                        "confidence": agent.get("confidence", 0.8)
                    }
                    validated_agents.append(agent_info)
            
            logger.info(f"Selected {len(validated_agents)} agents: {[a['port'] for a in validated_agents]}")
            return validated_agents
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            # Fallback to default agents (proven error handling)
            return [
                {"port": 8315, "name": "Pandas Analyst Agent", "reason": "Default fallback", 
                 "execution_order": 1, "parallel_group": 0, "confidence": 0.6}
            ]
    
    async def orchestrate_analysis(self, 
                                 task_request: TaskRequest) -> StreamingResponse:
        """
        Main orchestration method using proven Universal Engine workflow patterns
        """
        
        task_id = task_request.id
        logger.info(f"Starting orchestrated analysis for task: {task_id}")
        
        try:
            # Stage 1: Meta-reasoning (proven 4-stage pattern)
            meta_analysis = await self.perform_meta_reasoning(
                query=task_request.user_message,
                data=task_request.context,
                user_context=task_request.context.quality_summary or {},  # Use available context
                conversation_history=[]
            )
            
            # Stage 2: Agent selection (proven LLM-based selection)
            selected_agents = await self.select_optimal_agents(
                meta_analysis=meta_analysis,
                available_agents=self.available_agents
            )
            
            # Stage 3: Workflow execution (proven A2A orchestration patterns)
            execution_result = await self.workflow_orchestrator.execute_agent_workflow(
                selected_agents=selected_agents,
                query=task_request.user_message,
                data=task_request.context.datasets,
                meta_analysis=meta_analysis
            )
            
            # Stage 4: Result integration (proven integration patterns)
            integrated_results = await self.result_integrator.integrate_agent_results(
                execution_result=execution_result,
                meta_analysis=meta_analysis,
                user_context=task_request.context
            )
            
            # Create streaming response with progress tracking
            streaming_response = StreamingResponse(
                task_id=task_id,
                content_chunks=integrated_results.get("response_chunks", []),
                agent_progress=self._create_agent_progress_info(selected_agents, execution_result),
                artifacts=self._extract_artifacts(integrated_results),
                recommendations=self._generate_recommendations(integrated_results, meta_analysis),
                is_complete=True,
                total_execution_time=execution_result.get("total_execution_time", 0.0)
            )
            
            logger.info(f"Analysis orchestration completed for task: {task_id}")
            return streaming_response
            
        except Exception as e:
            logger.error(f"Analysis orchestration failed for task {task_id}: {e}")
            
            # Error recovery using proven patterns
            error_response = await self.error_handler.handle_orchestration_error(
                error=e,
                task_request=task_request,
                partial_results={}
            )
            
            return StreamingResponse(
                task_id=task_id,
                content_chunks=[f"Analysis encountered an error: {str(e)}"],
                agent_progress=[],
                artifacts=[],
                recommendations=[],
                is_complete=True,
                error_occurred=True
            )
    
    def _create_agent_progress_info(self, 
                                  selected_agents: List[Dict], 
                                  execution_result: Dict) -> List[AgentProgressInfo]:
        """Create agent progress information for UI visualization"""
        
        progress_info = []
        agent_results = execution_result.get("agent_results", {})
        
        for agent in selected_agents:
            port = agent["port"]
            agent_result = agent_results.get(str(port), {})
            
            progress = AgentProgressInfo(
                port=port,
                name=agent["name"],
                status=TaskState.COMPLETED if agent_result.get("success") else TaskState.FAILED,
                execution_time=agent_result.get("execution_time", 0.0),
                artifacts_generated=agent_result.get("artifacts", []),
                progress_percentage=100.0 if agent_result.get("success") else 0.0,
                current_task=agent_result.get("last_task", "Analysis complete"),
                avatar_icon=self._get_agent_icon(port),
                status_color=self._get_status_color(agent_result.get("success", False))
            )
            progress_info.append(progress)
        
        return progress_info
    
    def _get_agent_icon(self, port: int) -> str:
        """Get agent icon based on port"""
        icons = {
            8306: "🧹", 8307: "📁", 8308: "📊", 8309: "🔧", 8310: "⚙️",
            8311: "🗄️", 8312: "🔍", 8313: "🤖", 8314: "📈", 8315: "🐼"
        }
        return icons.get(port, "🔧")
    
    def _get_status_color(self, success: bool) -> str:
        """Get status color based on success"""
        return "#22c55e" if success else "#ef4444"  # green for success, red for failure
    
    def _extract_artifacts(self, integrated_results: Dict) -> List[EnhancedArtifact]:
        """Extract artifacts from integrated results"""
        artifacts = []
        
        # Extract artifacts from agent results (proven pattern)
        agent_artifacts = integrated_results.get("artifacts", [])
        
        for i, artifact_data in enumerate(agent_artifacts):
            artifact = EnhancedArtifact(
                id=f"artifact_{i}_{datetime.now().timestamp()}",
                type=artifact_data.get("type", "text"),
                content=artifact_data.get("content"),
                metadata=artifact_data.get("metadata", {}),
                source_agent=artifact_data.get("source_agent", 0),
                timestamp=datetime.now(),
                display_title=artifact_data.get("title", f"Analysis Result {i+1}"),
                description=artifact_data.get("description", "")
            )
            artifacts.append(artifact)
        
        return artifacts
    
    def _generate_recommendations(self, 
                                integrated_results: Dict, 
                                meta_analysis: Dict) -> List[OneClickRecommendation]:
        """Generate one-click recommendations based on results"""
        recommendations = []
        
        # Extract recommendations from meta-analysis (proven pattern)
        suggested_actions = meta_analysis.get("suggested_follow_ups", [])
        
        for i, action in enumerate(suggested_actions[:3]):  # Limit to 3 recommendations
            recommendation = OneClickRecommendation(
                title=action.get("title", f"Follow-up Analysis {i+1}"),
                description=action.get("description", "Additional analysis recommended"),
                action_type=action.get("action_type", "analysis"),
                parameters=action.get("parameters", {}),
                estimated_time=action.get("estimated_time", 30),
                confidence_score=action.get("confidence", 0.8),
                complexity_level=UserExpertiseLevel.INTERMEDIATE,
                expected_result_preview=action.get("expected_result", "Enhanced insights"),
                icon=action.get("icon", "🔍"),
                color_theme=action.get("color", "#3b82f6"),
                execution_button_text=action.get("button_text", "Execute Analysis")
            )
            recommendations.append(recommendation)
        
        return recommendations