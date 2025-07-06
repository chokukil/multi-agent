"""
테스트: Holistic Answer Synthesis Engine

Phase 3.1 전체론적 답변 합성 엔진 테스트
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List, Any

from .holistic_answer_synthesis_engine import (
    HolisticAnswerSynthesisEngine,
    AnswerStyle,
    AnswerPriority,
    SynthesisStrategy,
    SynthesisContext,
    AnswerSection,
    HolisticAnswer
)

# Phase 1 imports for test data
from .intelligent_query_processor import EnhancedQuery, IntentAnalysis, QueryType
from .domain_extractor import DomainKnowledge, DomainTaxonomy, KnowledgeItem
from .answer_predictor import AnswerTemplate

# Phase 2 imports for test data
from .domain_aware_agent_selector import AgentSelectionResult, AgentSelection, AgentType
from .a2a_agent_execution_orchestrator import ExecutionResult, ExecutionStatus, ExecutionTask
from .multi_agent_result_integration import IntegrationResult, IntegratedInsight


class TestHolisticAnswerSynthesisEngine:
    """전체론적 답변 합성 엔진 테스트"""
    
    def create_sample_enhanced_query(self) -> EnhancedQuery:
        """샘플 향상된 쿼리 생성"""
        from .intelligent_query_processor import DomainKnowledge, AnswerStructure, DomainType, AnswerFormat
        
        # 필요한 컴포넌트들을 생성
        intent_analysis = IntentAnalysis(
            primary_intent="공정 이상 분석 및 조치",
            data_scientist_perspective="통계적 공정 관리 방법론 적용",
            domain_expert_perspective="반도체 제조 공정 품질 관리 전문 지식 활용",
            technical_implementer_perspective="A2A 에이전트 기반 분석 파이프라인 구축",
            query_type=QueryType.ANALYTICAL,
            urgency_level=0.8,
            complexity_score=0.85,
            confidence_score=0.90
        )
        
        domain_knowledge = DomainKnowledge(
            domain_type=DomainType.MANUFACTURING,
            key_concepts=["통계적 공정 관리", "근본원인 분석", "품질 관리"],
            technical_terms=["LOT 히스토리", "계측값", "공정 이상"],
            business_context="반도체 제조 공정 품질 관리 시스템",
            required_expertise=["반도체 공정 전문가", "품질 관리 전문가"],
            relevant_methodologies=["SPC", "RCA", "Six Sigma"],
            success_metrics=["공정 이상 감지 정확도", "조치 시간 단축"],
            potential_challenges=["데이터 품질", "실시간 분석"]
        )
        
        answer_structure = AnswerStructure(
            expected_format=AnswerFormat.STRUCTURED_REPORT,
            key_sections=["상황 분석", "이상 감지", "근본원인 분석", "조치 방향"],
            required_visualizations=["관리도", "히스토그램", "파레토 차트"],
            success_criteria=["명확한 원인 규명", "실행 가능한 조치 방향"],
            expected_deliverables=["분석 보고서", "조치 계획서"],
            quality_checkpoints=["데이터 검증", "분석 결과 검토"]
        )
        
        return EnhancedQuery(
            original_query="LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고 조치 방향을 제안해주세요",
            intent_analysis=intent_analysis,
            domain_knowledge=domain_knowledge,
            answer_structure=answer_structure,
            enhanced_queries=[
                "반도체 제조 공정에서 LOT 히스토리 데이터와 계측값 데이터를 통계적 품질 관리(SPC) 방법론을 활용하여 분석하고, 공정 이상 상태를 조기 감지하여 근본원인 분석(RCA)을 수행한 후 구체적인 기술적 조치 방향을 제안해주세요."
            ],
            execution_strategy="multi_agent_analysis",
            context_requirements={"data_quality": "high", "real_time": True}
        )
    
    def create_sample_domain_knowledge(self):
        """샘플 도메인 지식 생성"""
        from .domain_extractor import (
            EnhancedDomainKnowledge, DomainTaxonomy, KnowledgeItem, 
            MethodologyMap, RiskAssessment, DomainType, KnowledgeConfidence, KnowledgeSource
        )
        
        return EnhancedDomainKnowledge(
            taxonomy=DomainTaxonomy(
                primary_domain=DomainType.MANUFACTURING,
                sub_domains=["semiconductor_manufacturing", "process_control"],
                industry_sector="semiconductor_industry",
                business_function="process_engineering",
                technical_area="statistical_process_control",
                confidence_score=0.89
            ),
            key_concepts={
                "Statistical Process Control": KnowledgeItem(
                    item="통계적 공정 관리 방법론",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.EXPLICIT_MENTION,
                    explanation="품질 관리의 핵심 기법"
                ),
                "Root Cause Analysis": KnowledgeItem(
                    item="근본원인 분석 방법론",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.CONTEXTUAL_INFERENCE,
                    explanation="문제 해결의 체계적 접근"
                )
            },
            technical_terms={
                "LOT 히스토리": KnowledgeItem(
                    item="제조 로트 이력 데이터",
                    confidence=KnowledgeConfidence.VERY_HIGH,
                    source=KnowledgeSource.EXPLICIT_MENTION,
                    explanation="공정 추적을 위한 핵심 데이터"
                )
            },
            methodology_map=MethodologyMap(
                standard_methodologies=["SPC", "Six Sigma", "Lean Manufacturing"],
                best_practices=["실시간 모니터링", "예방적 유지보수"],
                tools_and_technologies=["관리도", "파레토 차트", "히스토그램"],
                quality_standards=["ISO 9001", "IATF 16949"],
                compliance_requirements=["FDA 규정", "ISO 14001"]
            ),
            risk_assessment=RiskAssessment(
                technical_risks=["데이터 품질 이슈", "시스템 장애"],
                business_risks=["생산 중단", "품질 문제"],
                operational_risks=["인력 부족", "교육 부족"],
                compliance_risks=["규정 위반", "감사 실패"],
                mitigation_strategies=["백업 시스템", "정기 교육"]
            ),
            success_metrics=["공정 이상 감지 정확도", "조치 시간 단축", "품질 개선율"],
            stakeholder_map={
                "primary": ["공정 엔지니어", "품질 관리자"],
                "secondary": ["생산 관리자", "경영진"]
            },
            business_context="반도체 제조 공정의 품질 관리 및 이상 감지 시스템 구축",
            extraction_confidence=0.89
        )
    
    def create_sample_answer_template(self) -> AnswerTemplate:
        """샘플 답변 템플릿 생성"""
        from .answer_predictor import (
            AnswerTemplate, AnswerFormat, SectionSpecification, 
            ContentSection, VisualizationType, QualityCheckpoint
        )
        
        # 섹션 사양 생성
        sections = [
            SectionSpecification(
                section_type=ContentSection.EXECUTIVE_SUMMARY,
                title="요약",
                description="핵심 결과 요약",
                required_content=["주요 발견사항", "권고사항"],
                optional_content=["배경"],
                visualizations=[],
                priority=1,
                estimated_length="short",
                dependencies=[]
            ),
            SectionSpecification(
                section_type=ContentSection.DETAILED_ANALYSIS,
                title="상세 분석",
                description="데이터 분석 결과",
                required_content=["분석 방법", "결과"],
                optional_content=["기술적 세부사항"],
                visualizations=[VisualizationType.CONTROL_CHART, VisualizationType.PARETO_CHART],
                priority=2,
                estimated_length="long",
                dependencies=[ContentSection.EXECUTIVE_SUMMARY]
            )
        ]
        
        return AnswerTemplate(
            format_type=AnswerFormat.STRUCTURED_REPORT,
            target_audience="Technical Team",
            sections=sections,
            required_visualizations=[VisualizationType.CONTROL_CHART, VisualizationType.HISTOGRAM],
            quality_checkpoints=[QualityCheckpoint.DATA_VALIDATION, QualityCheckpoint.DOMAIN_EXPERT_REVIEW],
            estimated_completion_time="2-3 hours",
            complexity_score=0.81
        )
    
    def create_sample_agent_selection_result(self) -> AgentSelectionResult:
        """샘플 에이전트 선택 결과 생성"""
        return AgentSelectionResult(
            selected_agents=[
                AgentSelection(
                    agent_type=AgentType.EDA_TOOLS,
                    confidence=0.88,
                    reasoning="탐색적 데이터 분석 필요",
                    priority=1,
                    dependencies=[],
                    expected_outputs=["statistical_reports", "analysis_insights"],
                    domain_relevance=0.92,
                    task_fit=0.85
                ),
                AgentSelection(
                    agent_type=AgentType.DATA_VISUALIZATION,
                    confidence=0.75,
                    reasoning="데이터 시각화 필요",
                    priority=2,
                    dependencies=[AgentType.EDA_TOOLS],
                    expected_outputs=["interactive_charts", "visual_reports"],
                    domain_relevance=0.80,
                    task_fit=0.75
                )
            ],
            selection_strategy="domain_driven",
            total_confidence=0.70,
            reasoning="반도체 공정 분석을 위한 최적의 에이전트 조합 선택",
            execution_order=[AgentType.EDA_TOOLS, AgentType.DATA_VISUALIZATION],
            estimated_duration="45-60 minutes",
            success_probability=0.85,
            alternative_options=[]
        )
    
    def create_sample_execution_result(self) -> ExecutionResult:
        """샘플 실행 결과 생성"""
        return ExecutionResult(
            result_id="exec_001",
            execution_plan_id="plan_001",
            execution_status=ExecutionStatus.COMPLETED,
            completed_tasks=3,
            total_tasks=3,
            task_results=[
                ExecutionTask(
                    task_id="task_001",
                    agent_type=AgentType.EDA_TOOLS,
                    task_description="탐색적 데이터 분석 수행",
                    status=ExecutionStatus.COMPLETED,
                    confidence_score=0.92,
                    execution_time=15.5,
                    result_data={"analysis_complete": True}
                ),
                ExecutionTask(
                    task_id="task_002",
                    agent_type=AgentType.DATA_VISUALIZATION,
                    task_description="데이터 시각화 생성",
                    status=ExecutionStatus.COMPLETED,
                    confidence_score=0.87,
                    execution_time=12.3,
                    result_data={"charts_created": 5}
                )
            ],
            confidence_score=0.90,
            execution_time=45.2,
            generated_at=datetime.now()
        )
    
    def create_sample_integration_result(self) -> IntegrationResult:
        """샘플 통합 결과 생성"""
        from .multi_agent_result_integration import IntegrationStrategy, CrossValidationResult, QualityMetric
        
        return IntegrationResult(
            integration_id="integration_001",
            strategy=IntegrationStrategy.HIERARCHICAL,
            agent_results=[],
            cross_validation=CrossValidationResult(
                consistency_score=0.85,
                conflicting_findings=[],
                supporting_evidence=[],
                validation_notes="교차 검증 완료",
                confidence_adjustment=0.0
            ),
            integrated_insights=[
                IntegratedInsight(
                    insight_type="pattern",
                    content="공정 데이터에서 주기적 변동 패턴 발견",
                    confidence=0.88,
                    supporting_agents=["EDA_TOOLS"],
                    evidence_strength=0.85,
                    actionable_items=["패턴 분석 보고서 작성"],
                    priority=1
                ),
                IntegratedInsight(
                    insight_type="anomaly",
                    content="특정 시간대 품질 지표 이상 감지",
                    confidence=0.92,
                    supporting_agents=["EDA_TOOLS", "DATA_VISUALIZATION"],
                    evidence_strength=0.90,
                    actionable_items=["이상 구간 집중 분석"],
                    priority=1
                )
            ],
            quality_assessment={
                QualityMetric.COMPLETENESS: 0.85,
                QualityMetric.CONSISTENCY: 0.88,
                QualityMetric.ACCURACY: 0.90
            },
            synthesis_report="통합 분석 보고서",
            recommendations=[
                "공정 모니터링 시스템 강화",
                "품질 관리 절차 개선",
                "예측 모델 구축"
            ],
            confidence_score=0.86,
            integration_time=8.7
        )
    
    @pytest.mark.asyncio
    async def test_holistic_answer_synthesis_engine_init(self):
        """전체론적 답변 합성 엔진 초기화 테스트"""
        engine = HolisticAnswerSynthesisEngine()
        
        assert engine is not None
        assert engine.default_style == AnswerStyle.COMPREHENSIVE
        assert engine.default_priority == AnswerPriority.INSIGHTS
        assert engine.default_strategy == SynthesisStrategy.INTEGRATED
        assert len(engine.synthesis_history) == 0
        
        print("✅ 전체론적 답변 합성 엔진 초기화 성공")
    
    @pytest.mark.asyncio
    async def test_create_synthesis_context(self):
        """합성 컨텍스트 생성 테스트"""
        engine = HolisticAnswerSynthesisEngine()
        
        enhanced_query = self.create_sample_enhanced_query()
        domain_knowledge = self.create_sample_domain_knowledge()
        integration_result = self.create_sample_integration_result()
        
        context = await engine._create_synthesis_context(
            enhanced_query, domain_knowledge, integration_result
        )
        
        assert context is not None
        assert context.user_intent is not None
        assert context.domain_context is not None
        assert context.urgency_level in ["low", "medium", "high"]
        assert context.target_audience is not None
        assert isinstance(context.answer_style, AnswerStyle)
        assert isinstance(context.answer_priority, AnswerPriority)
        assert isinstance(context.synthesis_strategy, SynthesisStrategy)
        
        print(f"✅ 합성 컨텍스트 생성 성공: {context.user_intent}")
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self):
        """종합 분석 테스트"""
        engine = HolisticAnswerSynthesisEngine()
        
        # 테스트 데이터 준비
        enhanced_query = self.create_sample_enhanced_query()
        domain_knowledge = self.create_sample_domain_knowledge()
        answer_template = self.create_sample_answer_template()
        agent_selection_result = self.create_sample_agent_selection_result()
        execution_result = self.create_sample_execution_result()
        integration_result = self.create_sample_integration_result()
        
        synthesis_context = SynthesisContext(
            user_intent="공정 이상 분석 및 조치",
            domain_context="semiconductor_manufacturing",
            urgency_level="high",
            target_audience="technical",
            answer_style=AnswerStyle.TECHNICAL,
            answer_priority=AnswerPriority.ACTIONS,
            synthesis_strategy=SynthesisStrategy.ANALYTICAL
        )
        
        analysis_result = await engine._perform_comprehensive_analysis(
            enhanced_query, domain_knowledge, answer_template,
            agent_selection_result, execution_result, integration_result,
            synthesis_context
        )
        
        assert analysis_result is not None
        assert "situation_summary" in analysis_result
        assert "key_findings" in analysis_result
        assert isinstance(analysis_result["key_findings"], list)
        
        print(f"✅ 종합 분석 성공: {len(analysis_result.get('key_findings', []))}개 발견사항")
    
    @pytest.mark.asyncio
    async def test_full_synthesis_pipeline(self):
        """전체 합성 파이프라인 테스트"""
        engine = HolisticAnswerSynthesisEngine()
        
        # 테스트 데이터 준비
        enhanced_query = self.create_sample_enhanced_query()
        domain_knowledge = self.create_sample_domain_knowledge()
        answer_template = self.create_sample_answer_template()
        agent_selection_result = self.create_sample_agent_selection_result()
        execution_result = self.create_sample_execution_result()
        integration_result = self.create_sample_integration_result()
        
        print("🔄 전체 합성 파이프라인 테스트 시작...")
        
        # 전체론적 답변 합성 실행
        holistic_answer = await engine.synthesize_holistic_answer(
            enhanced_query=enhanced_query,
            domain_knowledge=domain_knowledge,
            answer_template=answer_template,
            agent_selection_result=agent_selection_result,
            execution_result=execution_result,
            integration_result=integration_result
        )
        
        # 결과 검증
        assert holistic_answer is not None
        assert holistic_answer.answer_id is not None
        assert holistic_answer.query_summary is not None
        assert holistic_answer.executive_summary is not None
        assert isinstance(holistic_answer.main_sections, list)
        assert isinstance(holistic_answer.key_insights, list)
        assert isinstance(holistic_answer.recommendations, list)
        assert isinstance(holistic_answer.next_steps, list)
        assert 0.0 <= holistic_answer.confidence_score <= 1.0
        assert isinstance(holistic_answer.quality_metrics, dict)
        assert holistic_answer.synthesis_time > 0
        
        # 품질 메트릭 확인
        required_metrics = [
            "completeness", "consistency", "actionability", 
            "relevance", "clarity", "comprehensiveness"
        ]
        for metric in required_metrics:
            assert metric in holistic_answer.quality_metrics
            assert 0.0 <= holistic_answer.quality_metrics[metric] <= 1.0
        
        print(f"✅ 전체 합성 파이프라인 테스트 성공!")
        print(f"   - 답변 ID: {holistic_answer.answer_id}")
        print(f"   - 신뢰도: {holistic_answer.confidence_score:.3f}")
        print(f"   - 합성 시간: {holistic_answer.synthesis_time:.2f}초")
        print(f"   - 섹션 수: {len(holistic_answer.main_sections)}")
        print(f"   - 인사이트 수: {len(holistic_answer.key_insights)}")
        print(f"   - 권고사항 수: {len(holistic_answer.recommendations)}")
        print(f"   - 다음 단계 수: {len(holistic_answer.next_steps)}")
        
        # 섹션별 상세 정보
        print("\n📋 생성된 답변 섹션:")
        for i, section in enumerate(holistic_answer.main_sections, 1):
            print(f"   {i}. {section.title} (신뢰도: {section.confidence:.3f})")
        
        # 핵심 인사이트
        print("\n💡 핵심 인사이트:")
        for i, insight in enumerate(holistic_answer.key_insights, 1):
            print(f"   {i}. {insight[:100]}...")
        
        # 권고사항
        print("\n📝 권고사항:")
        for i, recommendation in enumerate(holistic_answer.recommendations, 1):
            print(f"   {i}. {recommendation[:100]}...")
        
        # 품질 메트릭
        print("\n📊 품질 메트릭:")
        for metric, value in holistic_answer.quality_metrics.items():
            print(f"   - {metric}: {value:.3f}")
        
        return holistic_answer
    
    @pytest.mark.asyncio
    async def test_synthesis_history_management(self):
        """합성 이력 관리 테스트"""
        engine = HolisticAnswerSynthesisEngine()
        
        # 첫 번째 답변 생성
        enhanced_query = self.create_sample_enhanced_query()
        domain_knowledge = self.create_sample_domain_knowledge()
        answer_template = self.create_sample_answer_template()
        agent_selection_result = self.create_sample_agent_selection_result()
        execution_result = self.create_sample_execution_result()
        integration_result = self.create_sample_integration_result()
        
        answer1 = await engine.synthesize_holistic_answer(
            enhanced_query, domain_knowledge, answer_template,
            agent_selection_result, execution_result, integration_result
        )
        
        # 이력 확인
        history = await engine.get_synthesis_history()
        assert len(history) == 1
        assert history[0].answer_id == answer1.answer_id
        
        # 답변 요약 조회
        summary = await engine.get_answer_summary(answer1.answer_id)
        assert summary is not None
        assert summary["answer_id"] == answer1.answer_id
        assert summary["confidence_score"] == answer1.confidence_score
        
        print("✅ 합성 이력 관리 테스트 성공")
        print(f"   - 이력 항목 수: {len(history)}")
        print(f"   - 답변 요약 조회 성공: {summary['answer_id']}")


async def run_synthesis_engine_tests():
    """합성 엔진 테스트 실행"""
    test_instance = TestHolisticAnswerSynthesisEngine()
    
    print("🚀 Phase 3.1: Holistic Answer Synthesis Engine 테스트 시작")
    print("=" * 60)
    
    try:
        # 초기화 테스트
        await test_instance.test_holistic_answer_synthesis_engine_init()
        print()
        
        # 합성 컨텍스트 테스트
        await test_instance.test_create_synthesis_context()
        print()
        
        # 종합 분석 테스트
        await test_instance.test_comprehensive_analysis()
        print()
        
        # 전체 파이프라인 테스트
        holistic_answer = await test_instance.test_full_synthesis_pipeline()
        print()
        
        # 이력 관리 테스트
        await test_instance.test_synthesis_history_management()
        print()
        
        print("=" * 60)
        print("🎉 Phase 3.1 테스트 완료!")
        print(f"✅ 전체론적 답변 합성 엔진 검증 완료")
        print(f"📈 최종 신뢰도: {holistic_answer.confidence_score:.1%}")
        
        return holistic_answer
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 테스트 실행
    result = asyncio.run(run_synthesis_engine_tests())
    
    if result:
        print("\n🎯 Phase 3.1 구현 완료!")
        print("다음 단계: Phase 3.2 - Domain-Specific Answer Formatter") 