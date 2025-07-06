"""
Phase 3 Integration Layer

This module provides the integration layer between the Streamlit UI (ai.py) 
and the Phase 3 answer synthesis pipeline. It takes raw A2A agent results
and processes them through the complete Phase 3 pipeline to produce 
expert-level synthesized answers.

Author: CherryAI Development Team  
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Phase 1 imports
from core.query_processing import (
    IntelligentQueryProcessor,
    MultiPerspectiveIntentAnalyzer, 
    DomainKnowledgeExtractor,
    AnswerStructurePredictor,
    ContextualQueryEnhancer
)

# Phase 2 imports  
from core.query_processing import (
    DomainAwareAgentSelector,
    A2AAgentExecutionOrchestrator,
    MultiAgentResultIntegrator,
    ExecutionPlanManager
)

# Phase 3 imports
from core.query_processing import (
    HolisticAnswerSynthesisEngine,
    DomainSpecificAnswerFormatter, 
    UserPersonalizedResultOptimizer,
    AnswerQualityValidator,
    FinalAnswerStructuring,
    # Data types
    StructureType,
    ExportFormat,
    PresentationMode,
    PersonalizationLevel,
    UserRole,
    ValidationStrategy,
    QualityMetric,
    StructuringContext,
    UserProfile,
    OptimizationContext,
    QualityValidationContext
)

logger = logging.getLogger(__name__)


class Phase3IntegrationLayer:
    """Phase 3 통합 레이어 - A2A 결과를 전문가급 답변으로 합성"""
    
    def __init__(self):
        """Initialize Phase 3 Integration Layer"""
        logger.info("🔄 Phase 3 Integration Layer 초기화 중...")
        
        # Phase 1 components
        self.query_processor = IntelligentQueryProcessor()
        self.intent_analyzer = MultiPerspectiveIntentAnalyzer()
        self.domain_extractor = DomainKnowledgeExtractor()
        self.answer_predictor = AnswerStructurePredictor()
        self.query_enhancer = ContextualQueryEnhancer()
        
        # Phase 2 components
        self.agent_selector = DomainAwareAgentSelector()
        self.orchestrator = A2AAgentExecutionOrchestrator()
        self.result_integrator = MultiAgentResultIntegrator()
        self.plan_manager = ExecutionPlanManager()
        
        # Phase 3 components
        self.synthesis_engine = HolisticAnswerSynthesisEngine()
        self.formatter = DomainSpecificAnswerFormatter()
        self.optimizer = UserPersonalizedResultOptimizer()
        self.validator = AnswerQualityValidator()
        self.structuring_engine = FinalAnswerStructuring()
        
        logger.info("✅ Phase 3 Integration Layer 초기화 완료")
    
    async def process_user_query_to_expert_answer(
        self,
        user_query: str,
        a2a_agent_results: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        사용자 쿼리와 A2A 에이전트 결과를 전문가급 답변으로 합성
        
        Args:
            user_query: 원본 사용자 쿼리
            a2a_agent_results: A2A 에이전트들의 실행 결과
            user_context: 사용자 컨텍스트 (선택사항)
            session_context: 세션 컨텍스트 (선택사항)
            
        Returns:
            Dict containing the expert-level synthesized answer
        """
        start_time = time.time()
        logger.info(f"🚀 전문가급 답변 합성 시작: {user_query[:100]}...")
        
        try:
            # 1. Phase 1: Enhanced Query Processing
            logger.info("🔄 Phase 1: Query Processing 시작...")
            phase1_results = await self._execute_phase1(user_query, session_context)
            
            # 2. Phase 2: Knowledge-Aware Orchestration (A2A 결과 활용)
            logger.info("🔄 Phase 2: Orchestration 처리 중...")
            phase2_results = await self._execute_phase2(
                phase1_results, a2a_agent_results, session_context
            )
            
            # 3. Phase 3: Holistic Answer Synthesis
            logger.info("🔄 Phase 3: Answer Synthesis 시작...")
            phase3_results = await self._execute_phase3(
                phase1_results, phase2_results, user_context, session_context
            )
            
            processing_time = time.time() - start_time
            
            # 4. 최종 결과 패키징
            expert_answer = {
                "success": True,
                "processing_time": processing_time,
                "user_query": user_query,
                "enhanced_query": phase1_results["enhanced_query"],
                "domain_analysis": phase1_results["domain_knowledge"],
                "agent_results_summary": self._summarize_agent_results(a2a_agent_results),
                "synthesized_answer": phase3_results["final_structured_answer"],
                "quality_report": phase3_results["quality_report"],
                "confidence_score": phase3_results["final_structured_answer"].confidence_score,
                "metadata": {
                    "phase1_score": phase1_results.get("confidence_score", 0.8),
                    "phase2_integration_score": phase2_results.get("integration_score", 0.8),
                    "phase3_quality_score": phase3_results["quality_report"].overall_score,
                    "total_agents_used": len(a2a_agent_results),
                    "synthesis_strategy": "holistic_integration"
                }
            }
            
            logger.info(f"✅ 전문가급 답변 합성 완료 ({processing_time:.2f}s)")
            return expert_answer
            
        except Exception as e:
            logger.error(f"❌ 전문가급 답변 합성 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "user_query": user_query,
                "fallback_message": "전문가급 답변 합성 중 오류가 발생했습니다. 기본 결과를 표시합니다."
            }
    
    async def _execute_phase1(self, user_query: str, session_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 1: Enhanced Query Processing 실행 (Fallback Mode)"""
        
        try:
            # 1.1: Intelligent Query Processing
            enhanced_query = await self.query_processor.process_query(user_query)
            
            # 1.2: Multi-Perspective Intent Analysis (Fallback)
            try:
                intent_analysis = await self.intent_analyzer.analyze_intent_comprehensive(enhanced_query.original_query)
            except Exception as e:
                logger.warning(f"Intent analysis fallback: {e}")
                intent_analysis = enhanced_query.intent_analysis
            
            # 1.3: Domain Knowledge Extraction (Fallback)
            try:
                domain_knowledge = await self.domain_extractor.extract_comprehensive_domain_knowledge(enhanced_query.original_query)
            except Exception as e:
                logger.warning(f"Domain knowledge extraction fallback: {e}")
                domain_knowledge = enhanced_query.domain_knowledge
            
            # 1.4: Answer Structure Prediction (Fallback)
            try:
                answer_template = await self.answer_predictor.predict_optimal_structure(
                    intent_analysis, domain_knowledge
                )
            except Exception as e:
                logger.warning(f"Answer structure prediction fallback: {e}")
                answer_template = enhanced_query.answer_structure
            
            # 1.5: Contextual Query Enhancement (Fallback)
            try:
                query_enhancement = await self.query_enhancer.enhance_query_comprehensively(
                    enhanced_query.original_query, intent_analysis, domain_knowledge, answer_template
                )
            except Exception as e:
                logger.warning(f"Query enhancement fallback: {e}")
                query_enhancement = enhanced_query.enhanced_queries[0] if enhanced_query.enhanced_queries else user_query
            
            return {
                "enhanced_query": enhanced_query,
                "intent_analysis": intent_analysis,
                "domain_knowledge": domain_knowledge,
                "answer_template": answer_template,
                "query_enhancement": query_enhancement,
                "confidence_score": getattr(enhanced_query, 'intent_analysis', type('obj', (object,), {'confidence_score': 0.8})).confidence_score
            }
        
        except Exception as e:
            logger.error(f"Phase 1 execution failed, using minimal fallback: {e}")
            return self._create_fallback_phase1_results(user_query)
    
    async def _execute_phase2(
        self, 
        phase1_results: Dict[str, Any], 
        a2a_agent_results: List[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Phase 2: Knowledge-Aware Orchestration (A2A 결과 활용) (Fallback Mode)"""
        
        try:
            enhanced_query = phase1_results["enhanced_query"]
            domain_knowledge = phase1_results["domain_knowledge"]
            
            # 2.1: Domain-Aware Agent Selection (Fallback)
            try:
                agent_selection = await self.agent_selector.select_optimal_agents(
                    enhanced_query, domain_knowledge
                )
            except Exception as e:
                logger.warning(f"Agent selection fallback: {e}")
                agent_selection = type('MockAgentSelection', (), {
                    'selected_agents': [r.get("agent_name", "unknown") for r in a2a_agent_results],
                    'confidence_score': 0.8
                })()
            
            # 2.2: A2A Agent Execution Orchestration (A2A 결과 래핑)
            execution_result = self._wrap_a2a_results_as_execution_result(a2a_agent_results)
            
            # 2.3: Multi-Agent Result Integration (Fallback)
            try:
                integration_result = await self.result_integrator.integrate_agent_results(
                    agent_selection, execution_result, enhanced_query, domain_knowledge
                )
            except Exception as e:
                logger.warning(f"Result integration fallback: {e}")
                integration_result = type('MockIntegrationResult', (), {
                    'confidence_score': 0.8,
                    'integration_strategy': 'simple_aggregation'
                })()
            
            # 2.4: Execution Plan Management (Fallback)
            try:
                managed_plan = await self.plan_manager.create_managed_plan(
                    agent_selection, enhanced_query, domain_knowledge
                )
            except Exception as e:
                logger.warning(f"Plan management fallback: {e}")
                managed_plan = type('MockManagedPlan', (), {
                    'plan_id': f"fallback_plan_{int(time.time())}",
                    'execution_steps': len(a2a_agent_results)
                })()
            
            return {
                "agent_selection": agent_selection,
                "execution_result": execution_result,
                "integration_result": integration_result,
                "managed_plan": managed_plan,
                "integration_score": getattr(integration_result, 'confidence_score', 0.8)
            }
        
        except Exception as e:
            logger.error(f"Phase 2 execution failed, using fallback: {e}")
            return self._create_fallback_phase2_results(a2a_agent_results)
    
    async def _execute_phase3(
        self,
        phase1_results: Dict[str, Any],
        phase2_results: Dict[str, Any],
        user_context: Optional[Dict[str, Any]],
        session_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Phase 3: Holistic Answer Synthesis 실행"""
        
        # 3.1: Holistic Answer Synthesis
        holistic_answer = await self.synthesis_engine.synthesize_holistic_answer(
            enhanced_query=phase1_results["enhanced_query"],
            domain_knowledge=phase1_results["domain_knowledge"],
            answer_template=phase1_results["answer_template"],
            agent_selection_result=phase2_results["agent_selection"],
            execution_result=phase2_results["execution_result"],
            integration_result=phase2_results["integration_result"],
            managed_plan=phase2_results["managed_plan"]
        )
        
        # 3.2: Domain-Specific Answer Formatting
        formatting_context = self.formatter.create_formatting_context(
            phase1_results["domain_knowledge"],
            phase1_results["intent_analysis"]
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer,
            phase1_results["domain_knowledge"],
            phase1_results["intent_analysis"],
            formatting_context
        )
        
        # 3.3: User-Personalized Result Optimization
        user_profile = self._create_user_profile(user_context)
        optimization_context = OptimizationContext(
            user_profile=user_profile,
            current_query=phase1_results["enhanced_query"].original_query,
            domain_context=phase1_results["domain_knowledge"],
            intent_analysis=phase1_results["intent_analysis"]
        )
        
        optimized_result = self.optimizer.optimize_result(
            formatted_answer, user_profile.user_id, optimization_context
        )
        
        # 3.4: Answer Quality Validation
        validation_context = QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            required_metrics=list(QualityMetric),
            strict_mode=False,
            include_improvements=True
        )
        
        quality_report = self.validator.validate_optimized_result(
            optimized_result, validation_context
        )
        
        # 3.5: Final Answer Structuring
        structuring_context = StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.MARKDOWN,
            presentation_mode=PresentationMode.STATIC,
            target_audience=user_profile.role.value,
            use_case=phase1_results["domain_knowledge"].taxonomy.primary_domain.value
        )
        
        final_structured_answer = self.structuring_engine.structure_final_answer(
            holistic_answer=holistic_answer,
            formatted_answer=formatted_answer,
            optimized_result=optimized_result,
            quality_report=quality_report,
            structuring_context=structuring_context,
            user_id=user_profile.user_id,
            session_id=session_context.get("session_id", "unknown") if session_context else "unknown"
        )
        
        return {
            "holistic_answer": holistic_answer,
            "formatted_answer": formatted_answer,
            "optimized_result": optimized_result,
            "quality_report": quality_report,
            "final_structured_answer": final_structured_answer
        }
    
    def _wrap_a2a_results_as_execution_result(self, a2a_agent_results: List[Dict[str, Any]]):
        """A2A 에이전트 결과를 ExecutionResult 객체로 래핑"""
        from core.query_processing import ExecutionResult, ExecutionStatus
        
        # A2A 결과를 Phase 2 형식으로 변환
        agent_results = {}
        overall_confidence = 0.0
        
        for result in a2a_agent_results:
            agent_name = result.get("agent_name", "unknown")
            confidence = result.get("confidence", 0.8)
            status = "completed" if result.get("success", True) else "failed"
            
            agent_results[agent_name] = {
                "status": status,
                "confidence": confidence,
                "data": result.get("artifacts", []),
                "execution_time": result.get("execution_time", 0),
                "metadata": result.get("metadata", {})
            }
            overall_confidence += confidence
        
        if a2a_agent_results:
            overall_confidence /= len(a2a_agent_results)
        
        return ExecutionResult(
            execution_id=f"a2a_exec_{int(time.time())}",
            status=ExecutionStatus.COMPLETED,
            agent_results=agent_results,
            confidence_score=overall_confidence,
            execution_time=sum(r.get("execution_time", 0) for r in a2a_agent_results),
            metadata={"source": "a2a_integration", "agent_count": len(a2a_agent_results)}
        )
    
    def _create_user_profile(self, user_context: Optional[Dict[str, Any]]) -> UserProfile:
        """사용자 컨텍스트로부터 UserProfile 생성"""
        if not user_context:
            user_context = {}
        
        return UserProfile(
            user_id=user_context.get("user_id", "anonymous"),
            role=UserRole(user_context.get("role", "engineer")),
            domain_expertise=user_context.get("domain_expertise", {"general": 0.7}),
            preferences=user_context.get("preferences", {}),
            interaction_history=user_context.get("interaction_history", []),
            learning_weights=user_context.get("learning_weights", {}),
            personalization_level=PersonalizationLevel(user_context.get("personalization_level", "advanced"))
        )
    
    def _summarize_agent_results(self, a2a_agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """A2A 에이전트 결과 요약"""
        return {
            "total_agents": len(a2a_agent_results),
            "successful_agents": len([r for r in a2a_agent_results if r.get("success", True)]),
            "total_artifacts": sum(len(r.get("artifacts", [])) for r in a2a_agent_results),
            "average_confidence": sum(r.get("confidence", 0.8) for r in a2a_agent_results) / len(a2a_agent_results) if a2a_agent_results else 0,
            "agents_used": [r.get("agent_name", "unknown") for r in a2a_agent_results]
        }
    
    def _create_fallback_phase1_results(self, user_query: str) -> Dict[str, Any]:
        """Phase 1 실패 시 fallback 결과 생성"""
        from core.query_processing import (
            IntentAnalysis, QueryType, DomainKnowledge, DomainType, AnswerStructure, AnswerFormat
        )
        
        # Mock objects 생성
        mock_enhanced_query = type('MockEnhancedQuery', (), {
            'original_query': user_query,
            'enhanced_queries': [user_query],
            'intent_analysis': type('MockIntentAnalysis', (), {'confidence_score': 0.8})()
        })()
        
        mock_intent_analysis = IntentAnalysis(
            primary_intent="Data analysis request",
            data_scientist_perspective="Statistical analysis required",
            domain_expert_perspective="Domain knowledge needed", 
            technical_implementer_perspective="Technical implementation required",
            query_type=QueryType.ANALYTICAL,
            urgency_level=0.5,
            complexity_score=0.6,
            confidence_score=0.8
        )
        
        mock_domain_knowledge = DomainKnowledge(
            domain_type=DomainType.GENERAL,
            key_concepts=["analysis", "data"],
            technical_terms=["statistics", "modeling"],
            business_context="General business analysis",
            required_expertise=["data_science"],
            relevant_methodologies=["statistical_analysis"],
            success_metrics=["accuracy", "insight_quality"],
            potential_challenges=["data_quality", "complexity"]
        )
        
        mock_answer_template = AnswerStructure(
            expected_format=AnswerFormat.STRUCTURED_REPORT,
            key_sections=["overview", "analysis", "conclusions"],
            required_visualizations=["charts", "tables"],
            success_criteria=["completeness", "accuracy"],
            expected_deliverables=["report", "insights"],
            quality_checkpoints=["validation", "review"]
        )
        
        return {
            "enhanced_query": mock_enhanced_query,
            "intent_analysis": mock_intent_analysis,
            "domain_knowledge": mock_domain_knowledge,
            "answer_template": mock_answer_template,
            "query_enhancement": user_query,
            "confidence_score": 0.8
        }
    
    def _create_fallback_phase2_results(self, a2a_agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 2 실패 시 fallback 결과 생성"""
        from core.query_processing import ExecutionResult, ExecutionStatus
        
        # Mock objects 생성
        mock_agent_selection = type('MockAgentSelection', (), {
            'selected_agents': [r.get("agent_name", "unknown") for r in a2a_agent_results],
            'confidence_score': 0.8
        })()
        
        execution_result = self._wrap_a2a_results_as_execution_result(a2a_agent_results)
        
        mock_integration_result = type('MockIntegrationResult', (), {
            'confidence_score': 0.8,
            'integration_strategy': 'simple_aggregation'
        })()
        
        mock_managed_plan = type('MockManagedPlan', (), {
            'plan_id': f"fallback_plan_{int(time.time())}",
            'execution_steps': len(a2a_agent_results)
        })()
        
        return {
            "agent_selection": mock_agent_selection,
            "execution_result": execution_result,
            "integration_result": mock_integration_result,
            "managed_plan": mock_managed_plan,
            "integration_score": 0.8
        } 