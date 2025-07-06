"""
Test suite for DomainAwareAgentSelector

This test suite validates the intelligent agent selection functionality
based on domain knowledge and task requirements.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from core.query_processing.domain_aware_agent_selector import (
    DomainAwareAgentSelector, AgentType, AgentCapability, AgentSelection, AgentSelectionResult
)
from core.query_processing.intelligent_query_processor import (
    EnhancedQuery, IntentAnalysis, DomainKnowledge, AnswerStructure,
    QueryType, DomainType, AnswerFormat
)
from core.query_processing.domain_extractor import (
    EnhancedDomainKnowledge, DomainTaxonomy, KnowledgeItem, 
    MethodologyMap, RiskAssessment, KnowledgeConfidence, KnowledgeSource
)
from core.query_processing.intent_analyzer import DetailedIntentAnalysis


class TestDomainAwareAgentSelector:
    """Test cases for DomainAwareAgentSelector"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "data_requirements": ["tabular data", "time series data"],
            "analysis_types": ["statistical analysis", "exploratory analysis", "visualization"],
            "output_formats": ["reports", "charts", "insights"],
            "technical_constraints": ["standard processing"],
            "domain_specific_needs": ["manufacturing analytics", "process monitoring"],
            "complexity_factors": ["medium complexity", "statistical analysis"],
            "success_criteria": ["actionable insights", "statistical validation"],
            "timeline_requirements": ["standard processing time"]
        })
        mock_llm.ainvoke.return_value = mock_response
        return mock_llm
    
    @pytest.fixture
    def sample_enhanced_query(self):
        """Sample enhanced query for testing"""
        intent_analysis = IntentAnalysis(
            primary_intent="process_analysis",
            data_scientist_perspective="Statistical analysis needed",
            domain_expert_perspective="Manufacturing process monitoring",
            technical_implementer_perspective="Implementation challenges",
            query_type=QueryType.ANALYTICAL,
            urgency_level=0.7,
            complexity_score=0.8,
            confidence_score=0.9
        )
        
        domain_knowledge = DomainKnowledge(
            domain_type=DomainType.MANUFACTURING,
            key_concepts=["Statistical Process Control", "LOT History"],
            technical_terms=["계측값 데이터", "공정 이상"],
            business_context="Manufacturing process monitoring",
            required_expertise=["Statistical Analysis", "Process Control"],
            relevant_methodologies=["SPC", "Control Charts"],
            success_metrics=["Process Stability", "Quality Improvement"],
            potential_challenges=["Data Quality", "System Integration"]
        )
        
        answer_structure = AnswerStructure(
            expected_format=AnswerFormat.STRUCTURED_REPORT,
            key_sections=["Analysis", "Recommendations"],
            required_visualizations=["Control Charts", "Trend Analysis"],
            success_criteria=["Actionable Insights", "Statistical Validation"],
            expected_deliverables=["Report", "Visualizations"],
            quality_checkpoints=["Data Quality", "Statistical Validity"]
        )
        
        return EnhancedQuery(
            original_query="LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단해주세요",
            intent_analysis=intent_analysis,
            domain_knowledge=domain_knowledge,
            answer_structure=answer_structure,
            enhanced_queries=["반도체 제조 공정에서 LOT 히스토리와 계측값 데이터를 통합 분석하여 Statistical Process Control 방법론을 적용한 공정 이상 탐지 및 근본 원인 분석을 수행하고, 기술적 개선 방안을 제시하는 종합적인 분석 보고서를 작성해주세요."],
            execution_strategy="comprehensive_analysis",
            context_requirements={"data_quality": "high", "technical_expertise": "statistical_analysis"}
        )
    
    @pytest.fixture
    def sample_domain_knowledge(self):
        """Sample domain knowledge for testing"""
        return EnhancedDomainKnowledge(
            taxonomy=DomainTaxonomy(
                primary_domain=DomainType.MANUFACTURING,
                sub_domains=["Semiconductor Industry", "Quality Control"],
                industry_sector="Semiconductor",
                business_function="Process Control",
                technical_area="Statistical Analysis",
                confidence_score=0.85
            ),
            key_concepts={
                "spc": KnowledgeItem(
                    item="Statistical Process Control",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.EXPLICIT_MENTION,
                    explanation="Process monitoring and control methodology",
                    related_items=["control_charts", "process_monitoring"]
                )
            },
            technical_terms={
                "lot_data": KnowledgeItem(
                    item="계측값 데이터",
                    confidence=KnowledgeConfidence.MEDIUM,
                    source=KnowledgeSource.EXPLICIT_MENTION,
                    explanation="Measurement data from manufacturing process",
                    related_items=["process_data", "quality_metrics"]
                )
            },
            methodology_map=MethodologyMap(
                standard_methodologies=["Six Sigma", "SPC"],
                best_practices=["Control Charts", "Process Monitoring"],
                tools_and_technologies=["Statistical Software", "Data Analysis"],
                quality_standards=["ISO 9001", "Quality Management"],
                compliance_requirements=["Manufacturing Standards"]
            ),
            risk_assessment=RiskAssessment(
                technical_risks=["Data Quality Issues"],
                business_risks=["Process Downtime"],
                operational_risks=["System Integration"],
                compliance_risks=["Regulatory Requirements"],
                mitigation_strategies=["Data Validation", "System Testing"]
            ),
            success_metrics=["Process Stability", "Quality Improvement"],
            stakeholder_map={"engineering": ["process_engineer", "quality_manager"]},
            business_context="Manufacturing process monitoring and quality control",
            extraction_confidence=0.89
        )
    
    @pytest.fixture
    def sample_intent_analysis(self):
        """Sample intent analysis for testing"""
        return DetailedIntentAnalysis(
            primary_intent="process_analysis",
            secondary_intents=["anomaly_detection", "reporting"],
            task_type="statistical_analysis",
            complexity_level="medium",
            urgency_level="standard",
            expected_outcome="comprehensive_report",
            key_requirements=["data_analysis", "visualization", "recommendations"],
            analysis_confidence=0.76,
            perspectives=[]
        )
    
    @pytest.fixture
    def agent_selector(self, mock_llm):
        """Create agent selector with mocked LLM"""
        selector = DomainAwareAgentSelector()
        selector.llm = mock_llm
        return selector
    
    def test_agent_catalog_initialization(self, agent_selector):
        """Test agent catalog initialization"""
        catalog = agent_selector.agent_catalog
        
        # Check that all expected agents are present
        expected_agents = [
            AgentType.DATA_LOADER,
            AgentType.DATA_CLEANING,
            AgentType.DATA_WRANGLING,
            AgentType.EDA_TOOLS,
            AgentType.DATA_VISUALIZATION,
            AgentType.FEATURE_ENGINEERING,
            AgentType.H2O_ML,
            AgentType.MLFLOW_TOOLS,
            AgentType.SQL_DATABASE
        ]
        
        for agent_type in expected_agents:
            assert agent_type in catalog
            capability = catalog[agent_type]
            assert isinstance(capability, AgentCapability)
            assert capability.name
            assert capability.description
            assert capability.primary_skills
            assert capability.task_types
    
    def test_get_agent_capability(self, agent_selector):
        """Test getting specific agent capability"""
        # Test valid agent type
        eda_capability = agent_selector.get_agent_capability(AgentType.EDA_TOOLS)
        assert eda_capability is not None
        assert eda_capability.agent_type == AgentType.EDA_TOOLS
        assert "exploratory_analysis" in eda_capability.task_types
        
        # Test invalid agent type (should return None)
        invalid_capability = agent_selector.get_agent_capability("invalid_agent")
        assert invalid_capability is None
    
    def test_list_available_agents(self, agent_selector):
        """Test listing all available agents"""
        agents = agent_selector.list_available_agents()
        assert len(agents) == 9  # Should have 9 agents
        assert all(isinstance(agent, AgentCapability) for agent in agents)
    
    @pytest.mark.asyncio
    async def test_select_agents_success(
        self, agent_selector, sample_enhanced_query, sample_domain_knowledge, sample_intent_analysis
    ):
        """Test successful agent selection"""
        result = await agent_selector.select_agents(
            sample_enhanced_query, sample_domain_knowledge, sample_intent_analysis
        )
        
        assert isinstance(result, AgentSelectionResult)
        assert len(result.selected_agents) > 0
        assert result.total_confidence > 0.0
        assert result.reasoning
        assert result.execution_order
        assert result.estimated_duration
        assert result.success_probability > 0.0
        
        # Check that selected agents have proper structure
        for agent in result.selected_agents:
            assert isinstance(agent, AgentSelection)
            assert agent.agent_type in AgentType
            assert agent.confidence > 0.0
            assert agent.reasoning
            assert agent.priority > 0
    
    @pytest.mark.asyncio
    async def test_requirements_analysis(
        self, agent_selector, sample_enhanced_query, sample_domain_knowledge, sample_intent_analysis
    ):
        """Test requirements analysis"""
        requirements = await agent_selector._analyze_requirements(
            sample_enhanced_query, sample_domain_knowledge, sample_intent_analysis
        )
        
        assert isinstance(requirements, dict)
        expected_keys = [
            "data_requirements", "analysis_types", "output_formats",
            "technical_constraints", "domain_specific_needs", "complexity_factors",
            "success_criteria", "timeline_requirements"
        ]
        
        for key in expected_keys:
            assert key in requirements
            assert isinstance(requirements[key], list)
    
    @pytest.mark.asyncio
    async def test_agent_candidate_generation(self, agent_selector):
        """Test agent candidate generation"""
        requirements = {
            "data_requirements": ["tabular data", "time series"],
            "analysis_types": ["statistical analysis", "exploratory analysis"],
            "output_formats": ["reports", "visualizations"],
            "technical_constraints": ["standard processing"],
            "domain_specific_needs": ["process monitoring"],
            "complexity_factors": ["medium complexity"],
            "success_criteria": ["actionable insights"],
            "timeline_requirements": ["standard"]
        }
        
        candidates = await agent_selector._generate_agent_candidates(requirements)
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(agent, AgentType) for agent in candidates)
        
        # Should include EDA and visualization agents for this type of analysis
        assert AgentType.EDA_TOOLS in candidates
        assert AgentType.DATA_VISUALIZATION in candidates
    
    @pytest.mark.asyncio
    async def test_domain_relevance_calculation(self, agent_selector, sample_domain_knowledge):
        """Test domain relevance calculation"""
        eda_capability = agent_selector.get_agent_capability(AgentType.EDA_TOOLS)
        
        domain_score = await agent_selector._calculate_domain_relevance(
            eda_capability, sample_domain_knowledge
        )
        
        assert 0.0 <= domain_score <= 1.0
        # EDA should have good relevance for statistical analysis
        assert domain_score > 0.3
    
    @pytest.mark.asyncio
    async def test_task_fit_calculation(self, agent_selector, sample_intent_analysis):
        """Test task fit calculation"""
        requirements = {
            "analysis_types": ["statistical analysis", "exploratory analysis"],
            "output_formats": ["reports", "insights"]
        }
        
        eda_capability = agent_selector.get_agent_capability(AgentType.EDA_TOOLS)
        
        task_score = await agent_selector._calculate_task_fit(
            eda_capability, requirements, sample_intent_analysis
        )
        
        assert 0.0 <= task_score <= 1.0
        # EDA should have good task fit for statistical analysis
        assert task_score > 0.3
    
    @pytest.mark.asyncio
    async def test_execution_strategy_determination(self, agent_selector):
        """Test execution strategy determination"""
        scored_agents = [
            AgentSelection(
                agent_type=AgentType.DATA_LOADER,
                confidence=0.8,
                reasoning="Data loading capability",
                priority=1
            ),
            AgentSelection(
                agent_type=AgentType.EDA_TOOLS,
                confidence=0.7,
                reasoning="Statistical analysis capability",
                priority=2
            ),
            AgentSelection(
                agent_type=AgentType.DATA_VISUALIZATION,
                confidence=0.6,
                reasoning="Visualization capability",
                priority=3
            )
        ]
        
        requirements = {"complexity_factors": ["medium"]}
        
        strategy = await agent_selector._determine_execution_strategy(scored_agents, requirements)
        
        assert isinstance(strategy, dict)
        assert "execution_order" in strategy
        assert "strategy" in strategy
        assert "estimated_duration" in strategy
        
        # Check execution order follows logical pipeline
        execution_order = strategy["execution_order"]
        if AgentType.DATA_LOADER in execution_order and AgentType.EDA_TOOLS in execution_order:
            assert execution_order.index(AgentType.DATA_LOADER) < execution_order.index(AgentType.EDA_TOOLS)
    
    @pytest.mark.asyncio
    async def test_fallback_selection(self, agent_selector, sample_enhanced_query):
        """Test fallback selection mechanism"""
        result = await agent_selector._create_fallback_selection(sample_enhanced_query)
        
        assert isinstance(result, AgentSelectionResult)
        assert len(result.selected_agents) > 0
        assert result.selection_strategy == "fallback_default"
        assert result.total_confidence > 0.0
        
        # Should include basic agents
        agent_types = [agent.agent_type for agent in result.selected_agents]
        assert AgentType.DATA_LOADER in agent_types
        assert AgentType.EDA_TOOLS in agent_types
    
    def test_duration_estimation(self, agent_selector):
        """Test duration estimation"""
        # Test different numbers of agents
        assert "minutes" in agent_selector._estimate_duration(1)
        assert "minutes" in agent_selector._estimate_duration(5)
        
        # Test large number of agents
        duration = agent_selector._estimate_duration(15)
        assert "h" in duration or "minutes" in duration
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent_selector):
        """Test error handling in agent selection"""
        # Test with invalid LLM response
        agent_selector.llm.ainvoke.side_effect = Exception("LLM error")
        
        result = await agent_selector.select_agents(
            None, None, None  # Invalid inputs
        )
        
        # Should return fallback selection
        assert isinstance(result, AgentSelectionResult)
        assert result.selection_strategy == "fallback_default"
    
    def test_agent_type_enum(self):
        """Test AgentType enum"""
        # Test all agent types exist
        expected_types = [
            "data_loader", "data_cleaning", "data_wrangling", "eda_tools",
            "data_visualization", "feature_engineering", "h2o_ml",
            "mlflow_tools", "sql_database"
        ]
        
        for agent_type in expected_types:
            assert any(at.value == agent_type for at in AgentType)
    
    def test_agent_capability_dataclass(self):
        """Test AgentCapability dataclass"""
        capability = AgentCapability(
            agent_type=AgentType.EDA_TOOLS,
            name="Test Agent",
            description="Test description",
            primary_skills=["skill1", "skill2"],
            secondary_skills=["skill3"],
            domain_expertise=["domain1"],
            task_types=["task1", "task2"],
            port=8000
        )
        
        assert capability.agent_type == AgentType.EDA_TOOLS
        assert capability.name == "Test Agent"
        assert len(capability.primary_skills) == 2
        assert capability.port == 8000
        assert len(capability.prerequisites) == 0  # default
        assert len(capability.outputs) == 0  # default
    
    def test_agent_selection_dataclass(self):
        """Test AgentSelection dataclass"""
        selection = AgentSelection(
            agent_type=AgentType.DATA_VISUALIZATION,
            confidence=0.85,
            reasoning="Good visualization capability",
            priority=1
        )
        
        assert selection.agent_type == AgentType.DATA_VISUALIZATION
        assert selection.confidence == 0.85
        assert selection.priority == 1
        assert len(selection.dependencies) == 0  # default
        assert len(selection.expected_outputs) == 0  # default
        assert selection.domain_relevance == 0.0  # default
        assert selection.task_fit == 0.0  # default


@pytest.mark.asyncio
async def test_integration_with_phase1_modules():
    """Integration test with Phase 1 modules"""
    # This test verifies that the agent selector works with actual Phase 1 outputs
    selector = DomainAwareAgentSelector()
    
    # Mock the LLM response for requirements analysis
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "data_requirements": ["manufacturing data", "process data"],
        "analysis_types": ["statistical process control", "anomaly detection"],
        "output_formats": ["technical reports", "visualizations"],
        "technical_constraints": ["real-time processing"],
        "domain_specific_needs": ["semiconductor manufacturing"],
        "complexity_factors": ["complex statistical analysis"],
        "success_criteria": ["process improvement"],
        "timeline_requirements": ["urgent processing"]
    })
    mock_llm.ainvoke.return_value = mock_response
    selector.llm = mock_llm
    
    # Create sample inputs that would come from Phase 1
    intent_analysis = IntentAnalysis(
        primary_intent="process_analysis",
        data_scientist_perspective="Statistical analysis needed",
        domain_expert_perspective="Manufacturing process monitoring",
        technical_implementer_perspective="Implementation challenges",
        query_type=QueryType.ANALYTICAL,
        urgency_level=0.7,
        complexity_score=0.8,
        confidence_score=0.9
    )
    
    domain_knowledge = DomainKnowledge(
        domain_type=DomainType.MANUFACTURING,
        key_concepts=["Statistical Process Control"],
        technical_terms=["계측값 데이터", "공정 이상"],
        business_context="Manufacturing process monitoring",
        required_expertise=["Statistical Analysis"],
        relevant_methodologies=["SPC"],
        success_metrics=["Process Stability"],
        potential_challenges=["Data Quality"]
    )
    
    answer_structure = AnswerStructure(
        expected_format=AnswerFormat.STRUCTURED_REPORT,
        key_sections=["Analysis", "Recommendations"],
        required_visualizations=["Control Charts"],
        success_criteria=["Actionable Insights"],
        expected_deliverables=["Report"],
        quality_checkpoints=["Data Quality"]
    )
    
    enhanced_query = EnhancedQuery(
        original_query="제조 공정 데이터 분석",
        intent_analysis=intent_analysis,
        domain_knowledge=domain_knowledge,
        answer_structure=answer_structure,
        enhanced_queries=["반도체 제조 공정에서 Statistical Process Control을 적용한 공정 이상 탐지 분석"],
        execution_strategy="comprehensive_analysis",
        context_requirements={"data_quality": "high"}
    )
    
    enhanced_domain_knowledge = EnhancedDomainKnowledge(
        taxonomy=DomainTaxonomy(
            primary_domain=DomainType.MANUFACTURING,
            sub_domains=["Semiconductor"],
            industry_sector="Semiconductor",
            business_function="Process Control",
            technical_area="Statistical Analysis",
            confidence_score=0.85
        ),
        key_concepts={},
        technical_terms={},
        methodology_map=MethodologyMap(
            standard_methodologies=[],
            best_practices=[],
            tools_and_technologies=[],
            quality_standards=[],
            compliance_requirements=[]
        ),
        risk_assessment=RiskAssessment(
            technical_risks=[],
            business_risks=[],
            operational_risks=[],
            compliance_risks=[],
            mitigation_strategies=[]
        ),
        success_metrics=[],
        stakeholder_map={},
        business_context="",
        extraction_confidence=0.85
    )
    
    detailed_intent_analysis = DetailedIntentAnalysis(
        primary_intent="process_analysis",
        secondary_intents=["quality_control"],
        task_type="statistical_analysis",
        complexity_level="high",
        urgency_level="standard",
        expected_outcome="technical_report",
        key_requirements=["data_analysis", "visualization"],
        analysis_confidence=0.8,
        perspectives=[]
    )
    
    # Test the integration
    result = await selector.select_agents(enhanced_query, enhanced_domain_knowledge, detailed_intent_analysis)
    
    assert isinstance(result, AgentSelectionResult)
    assert len(result.selected_agents) > 0
    assert result.total_confidence > 0.0
    
    # Should select appropriate agents for manufacturing analysis
    selected_types = [agent.agent_type for agent in result.selected_agents]
    assert AgentType.EDA_TOOLS in selected_types  # For statistical analysis
    assert AgentType.DATA_VISUALIZATION in selected_types  # For visualization
    
    print(f"✅ Integration test passed: {len(result.selected_agents)} agents selected")
    print(f"Selected agents: {[agent.agent_type.value for agent in result.selected_agents]}")
    print(f"Total confidence: {result.total_confidence:.2f}")


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(test_integration_with_phase1_modules()) 