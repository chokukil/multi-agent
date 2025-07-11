import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Our imports
from core.universal_data_analysis_router import (
    UniversalDataAnalysisRouter,
    AnalysisType,
    RouteDecision,
    AgentCapability,
    get_universal_router
)


class TestAnalysisType:
    """Test AnalysisType enum"""
    
    def test_enum_values(self):
        """Test that all expected enum values are defined"""
        expected_values = [
            "pandas_ai", "eda", "visualization", "statistics", 
            "ml", "cleaning", "loading", "features", "database", "general"
        ]
        
        for value in expected_values:
            assert any(item.value == value for item in AnalysisType)
    
    def test_enum_accessibility(self):
        """Test enum members are accessible"""
        assert AnalysisType.PANDAS_AI.value == "pandas_ai"
        assert AnalysisType.EDA.value == "eda"
        assert AnalysisType.VISUALIZATION.value == "visualization"


class TestRouteDecision:
    """Test RouteDecision dataclass"""
    
    def test_creation(self):
        """Test RouteDecision creation"""
        decision = RouteDecision(
            analysis_type=AnalysisType.EDA,
            confidence=0.8,
            reasoning="Test reasoning",
            recommended_agent="eda",
            parameters={"test": "value"},
            fallback_agents=["pandas_ai"]
        )
        
        assert decision.analysis_type == AnalysisType.EDA
        assert decision.confidence == 0.8
        assert decision.reasoning == "Test reasoning"
        assert decision.recommended_agent == "eda"
        assert decision.parameters == {"test": "value"}
        assert decision.fallback_agents == ["pandas_ai"]
    
    def test_optional_fields(self):
        """Test RouteDecision with optional fields as None"""
        decision = RouteDecision(
            analysis_type=AnalysisType.PANDAS_AI,
            confidence=0.5,
            reasoning="Basic routing",
            recommended_agent="pandas_ai"
        )
        
        assert decision.parameters is None
        assert decision.fallback_agents is None


class TestAgentCapability:
    """Test AgentCapability dataclass"""
    
    def test_creation(self):
        """Test AgentCapability creation"""
        capability = AgentCapability(
            agent_name="Test Agent",
            analysis_types=[AnalysisType.EDA, AnalysisType.STATISTICS],
            strengths=["Fast analysis", "Good insights"],
            limitations=["Limited visualization"],
            endpoint="http://localhost:8000",
            priority=7
        )
        
        assert capability.agent_name == "Test Agent"
        assert len(capability.analysis_types) == 2
        assert AnalysisType.EDA in capability.analysis_types
        assert capability.priority == 7
        assert capability.endpoint == "http://localhost:8000"


class TestUniversalDataAnalysisRouter:
    """Test UniversalDataAnalysisRouter class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.router = UniversalDataAnalysisRouter()
    
    def test_initialization(self):
        """Test router initialization"""
        assert self.router.config == {}
        assert isinstance(self.router.agent_capabilities, dict)
        assert len(self.router.agent_capabilities) > 0
        assert self.router.routing_history == []
    
    def test_agent_capabilities_structure(self):
        """Test agent capabilities are properly structured"""
        capabilities = self.router.agent_capabilities
        
        # Check required agents exist
        required_agents = ["pandas_ai", "eda", "visualization", "data_cleaning", "ml"]
        for agent in required_agents:
            assert agent in capabilities
            
            # Check capability structure
            cap = capabilities[agent]
            assert isinstance(cap.agent_name, str)
            assert isinstance(cap.analysis_types, list)
            assert isinstance(cap.strengths, list)
            assert isinstance(cap.limitations, list)
            assert isinstance(cap.endpoint, str)
            assert isinstance(cap.priority, int)
    
    def test_normalize_agent_name(self):
        """Test agent name normalization"""
        # Test exact matches
        assert self.router._normalize_agent_name("pandas_ai") == "pandas_ai"
        assert self.router._normalize_agent_name("eda") == "eda"
        
        # Test full name matches
        assert self.router._normalize_agent_name("Universal Pandas-AI Agent") == "pandas_ai"
        assert self.router._normalize_agent_name("Enhanced EDA Tools Agent") == "eda"
        assert self.router._normalize_agent_name("Data Visualization Agent") == "visualization"
        
        # Test partial matches
        assert self.router._normalize_agent_name("ML Agent") == "ml"
        assert self.router._normalize_agent_name("visualization tool") == "visualization"
        
        # Test unknown agent (should return default)
        assert self.router._normalize_agent_name("Unknown Agent") == "pandas_ai"
    
    def test_rule_based_analysis(self):
        """Test rule-based analysis method"""
        # Test visualization keywords
        decision = self.router._rule_based_analysis("차트를 그려주세요")
        assert decision.analysis_type == AnalysisType.VISUALIZATION
        assert decision.recommended_agent == "visualization"
        assert decision.confidence > 0
        
        # Test EDA keywords
        decision = self.router._rule_based_analysis("데이터 분포를 보여주세요")
        assert decision.analysis_type == AnalysisType.EDA
        assert decision.recommended_agent == "eda"
        
        # Test ML keywords
        decision = self.router._rule_based_analysis("예측 모델을 만들어주세요")
        assert decision.analysis_type == AnalysisType.MACHINE_LEARNING
        assert decision.recommended_agent == "ml"
        
        # Test cleaning keywords
        decision = self.router._rule_based_analysis("결측값을 처리해주세요")
        assert decision.analysis_type == AnalysisType.DATA_CLEANING
        assert decision.recommended_agent == "data_cleaning"
        
        # Test unknown query (should default to pandas_ai)
        decision = self.router._rule_based_analysis("안녕하세요")
        assert decision.analysis_type == AnalysisType.PANDAS_AI
        assert decision.recommended_agent == "pandas_ai"
    
    def test_create_agent_summary(self):
        """Test agent summary creation"""
        summary = self.router._create_agent_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Check that all agents are mentioned
        for agent_id in self.router.agent_capabilities.keys():
            assert agent_id in summary
    
    @pytest.mark.asyncio
    async def test_analyze_query_intent_without_llm(self):
        """Test query intent analysis without LLM (rule-based fallback)"""
        # Mock no OpenAI client
        self.router.openai_client = None
        
        decision = await self.router.analyze_query_intent("시각화해주세요")
        
        assert isinstance(decision, RouteDecision)
        assert decision.analysis_type == AnalysisType.VISUALIZATION
        assert decision.recommended_agent == "visualization"
        assert decision.confidence > 0
        assert isinstance(decision.reasoning, str)
        
        # Check routing history is updated
        assert len(self.router.routing_history) == 1
    
    @pytest.mark.asyncio
    async def test_route_query_success(self):
        """Test successful query routing"""
        result = await self.router.route_query("데이터를 시각화해주세요")
        
        assert result["success"] is True
        assert "decision" in result
        assert "agent_info" in result
        assert "context" in result
        assert "timestamp" in result
        
        decision = result["decision"]
        assert "analysis_type" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        assert "recommended_agent" in decision
        assert "agent_endpoint" in decision
        
        agent_info = result["agent_info"]
        assert "name" in agent_info
        assert "strengths" in agent_info
        assert "limitations" in agent_info
        assert "endpoint" in agent_info
        assert "priority" in agent_info
    
    @pytest.mark.asyncio
    async def test_route_query_with_session_context(self):
        """Test routing with session context"""
        # Mock session data manager
        with patch('core.universal_data_analysis_router.SessionDataManager') as mock_sdm:
            mock_instance = Mock()
            mock_instance.get_uploaded_files.return_value = ["test.csv"]
            mock_sdm.return_value = mock_instance
            
            # Reinitialize router to pick up mocked dependencies
            router = UniversalDataAnalysisRouter()
            
            result = await router.route_query(
                "데이터를 분석해주세요", 
                session_id="test_session",
                context={"extra": "info"}
            )
            
            assert result["success"] is True
            assert "context" in result
    
    @pytest.mark.asyncio
    async def test_route_query_error_handling(self):
        """Test routing error handling"""
        # Force an error by mocking a failing method
        with patch.object(self.router, 'analyze_query_intent', side_effect=Exception("Test error")):
            result = await self.router.route_query("test query")
            
            assert result["success"] is False
            assert "error" in result
            assert "fallback" in result
            assert result["fallback"]["recommended_agent"] == "pandas_ai"
    
    def test_get_routing_statistics_empty(self):
        """Test routing statistics when no routing history"""
        stats = self.router.get_routing_statistics()
        
        assert stats["total_queries"] == 0
        assert stats["agent_distribution"] == {}
    
    @pytest.mark.asyncio
    async def test_get_routing_statistics_with_data(self):
        """Test routing statistics with actual routing data"""
        # Perform some routing to generate history
        await self.router.route_query("시각화해주세요")
        await self.router.route_query("모델을 만들어주세요")
        await self.router.route_query("차트를 그려주세요")
        
        stats = self.router.get_routing_statistics()
        
        assert stats["total_queries"] == 3
        assert isinstance(stats["agent_distribution"], dict)
        assert isinstance(stats["analysis_type_distribution"], dict)
        assert isinstance(stats["average_confidence"], float)
        assert stats["routing_history_size"] == 3
        assert stats["average_confidence"] > 0
    
    def test_clear_routing_history(self):
        """Test clearing routing history"""
        # Add some fake history
        self.router.routing_history = [{"test": "data"}]
        assert len(self.router.routing_history) == 1
        
        self.router.clear_routing_history()
        assert len(self.router.routing_history) == 0


class TestRouterIntegration:
    """Test router integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_multiple_queries_workflow(self):
        """Test multiple queries routing workflow"""
        router = UniversalDataAnalysisRouter()
        
        queries = [
            ("데이터를 시각화해주세요", "visualization"),
            ("평균값을 계산해주세요", "eda"),
            ("예측 모델을 만들어주세요", "ml"),
            ("결측값을 처리해주세요", "data_cleaning"),
            ("일반적인 질문입니다", "pandas_ai")
        ]
        
        for query, expected_agent in queries:
            result = await router.route_query(query)
            assert result["success"] is True
            
            # Check that routing is reasonable (may not be exact due to LLM variability)
            recommended_agent = result["decision"]["recommended_agent"]
            assert recommended_agent in router.agent_capabilities.keys()
        
        # Check statistics
        stats = router.get_routing_statistics()
        assert stats["total_queries"] == len(queries)
        assert len(stats["agent_distribution"]) > 0
    
    def test_get_universal_router_singleton(self):
        """Test singleton pattern for router instance"""
        router1 = get_universal_router()
        router2 = get_universal_router()
        
        # Should be the same instance
        assert router1 is router2
        
        # Test with config
        router3 = get_universal_router({"test": "config"})
        assert router3 is router1  # Still same instance
    
    @pytest.mark.asyncio 
    async def test_enhanced_tracking_integration(self):
        """Test integration with enhanced tracking system"""
        # Mock enhanced tracer
        with patch('core.universal_data_analysis_router.get_enhanced_tracer') as mock_tracer:
            mock_tracer_instance = Mock()
            mock_tracer.return_value = mock_tracer_instance
            
            router = UniversalDataAnalysisRouter()
            await router.route_query("테스트 질문")
            
            # Verify tracking was called (if available)
            if router.enhanced_tracer:
                mock_tracer_instance.log_data_operation.assert_called()


class TestRouterErrorScenarios:
    """Test router error handling scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.router = UniversalDataAnalysisRouter()
    
    @pytest.mark.asyncio
    async def test_malformed_llm_response_handling(self):
        """Test handling of malformed LLM responses"""
        # Mock OpenAI client with malformed response
        with patch('core.universal_data_analysis_router.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Invalid JSON response"
            mock_client.chat.completions.create.return_value = mock_response
            
            self.router.openai_client = mock_client
            
            decision = await self.router._llm_based_analysis("test query")
            
            # Should fall back to rule-based analysis
            assert isinstance(decision, RouteDecision)
            assert decision.recommended_agent in self.router.agent_capabilities.keys()
    
    @pytest.mark.asyncio
    async def test_openai_api_error_handling(self):
        """Test handling of OpenAI API errors"""
        # Mock OpenAI client that raises an exception
        with patch('core.universal_data_analysis_router.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            self.router.openai_client = mock_client
            
            decision = await self.router._llm_based_analysis("test query")
            
            # Should fall back to rule-based analysis
            assert isinstance(decision, RouteDecision)
            assert decision.recommended_agent in self.router.agent_capabilities.keys()
    
    @pytest.mark.asyncio
    async def test_session_context_error_handling(self):
        """Test handling of session context gathering errors"""
        # Mock session data manager that raises an exception
        with patch.object(self.router, 'session_data_manager') as mock_sdm:
            mock_sdm.get_uploaded_files.side_effect = Exception("Session error")
            
            # Should not crash and return empty context
            context = await self.router._gather_session_context("test_session", {})
            assert isinstance(context, dict)


if __name__ == "__main__":
    pytest.main([__file__]) 