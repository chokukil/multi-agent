import pytest
import asyncio
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Our imports
from core.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationStrategy,
    AgentTask,
    OrchestrationPlan,
    OrchestrationResult,
    MultiAgentA2AExecutor,
    get_multi_agent_orchestrator,
    create_multi_agent_card,
    create_multi_agent_a2a_app
)


class TestOrchestrationStrategy:
    """Test OrchestrationStrategy enum"""
    
    def test_enum_values(self):
        """Test that all expected enum values are defined"""
        expected_values = [
            "single_agent", "sequential", "parallel", "hierarchical", "collaborative"
        ]
        
        for value in expected_values:
            assert any(item.value == value for item in OrchestrationStrategy)
    
    def test_enum_accessibility(self):
        """Test enum members are accessible"""
        assert OrchestrationStrategy.SINGLE_AGENT.value == "single_agent"
        assert OrchestrationStrategy.SEQUENTIAL.value == "sequential"
        assert OrchestrationStrategy.PARALLEL.value == "parallel"


class TestAgentTask:
    """Test AgentTask dataclass"""
    
    def test_creation(self):
        """Test AgentTask creation"""
        task = AgentTask(
            agent_id="test_agent",
            agent_type="specialized",
            task_description="Test task",
            input_data={"test": "data"},
            dependencies=["dep1"],
            priority=8,
            timeout=120,
            retry_count=2
        )
        
        assert task.agent_id == "test_agent"
        assert task.agent_type == "specialized"
        assert task.task_description == "Test task"
        assert task.input_data == {"test": "data"}
        assert task.dependencies == ["dep1"]
        assert task.priority == 8
        assert task.timeout == 120
        assert task.retry_count == 2
    
    def test_default_values(self):
        """Test AgentTask with default values"""
        task = AgentTask(
            agent_id="simple_agent",
            agent_type="llm",
            task_description="Simple task",
            input_data={}
        )
        
        assert task.dependencies is None
        assert task.priority == 5
        assert task.timeout == 300
        assert task.retry_count == 3


class TestOrchestrationPlan:
    """Test OrchestrationPlan dataclass"""
    
    def test_creation(self):
        """Test OrchestrationPlan creation"""
        task = AgentTask(
            agent_id="test_agent",
            agent_type="specialized",
            task_description="Test task",
            input_data={}
        )
        
        plan = OrchestrationPlan(
            strategy=OrchestrationStrategy.SEQUENTIAL,
            tasks=[task],
            estimated_duration=60,
            confidence=0.9,
            reasoning="Test plan"
        )
        
        assert plan.strategy == OrchestrationStrategy.SEQUENTIAL
        assert len(plan.tasks) == 1
        assert plan.estimated_duration == 60
        assert plan.confidence == 0.9
        assert plan.reasoning == "Test plan"


class TestOrchestrationResult:
    """Test OrchestrationResult dataclass"""
    
    def test_creation(self):
        """Test OrchestrationResult creation"""
        result = OrchestrationResult(
            success=True,
            results={"agent1": {"data": "value"}},
            insights=["Insight 1"],
            recommendations=["Rec 1"],
            execution_time=5.2,
            errors=None,
            metadata={"meta": "data"}
        )
        
        assert result.success is True
        assert result.results == {"agent1": {"data": "value"}}
        assert result.insights == ["Insight 1"]
        assert result.recommendations == ["Rec 1"]
        assert result.execution_time == 5.2
        assert result.errors is None
        assert result.metadata == {"meta": "data"}


class TestMultiAgentOrchestrator:
    """Test MultiAgentOrchestrator class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.orchestrator = MultiAgentOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        assert self.orchestrator.config == {}
        assert isinstance(self.orchestrator.agent_endpoints, dict)
        assert len(self.orchestrator.agent_endpoints) > 0
        assert self.orchestrator.execution_history == []
    
    def test_agent_endpoints_structure(self):
        """Test agent endpoints are properly configured"""
        endpoints = self.orchestrator.agent_endpoints
        
        required_agents = ["pandas_ai", "eda_tools", "data_visualization", "data_cleaning"]
        for agent in required_agents:
            assert agent in endpoints
            assert endpoints[agent].startswith("http://localhost:")
    
    def test_assess_query_complexity(self):
        """Test query complexity assessment"""
        # Simple query
        assert self.orchestrator._assess_query_complexity("안녕하세요") == "simple"
        
        # Complex query
        assert self.orchestrator._assess_query_complexity("데이터를 분석해주세요") == "complex"
        
        # Multi-step query
        assert self.orchestrator._assess_query_complexity("데이터를 분석하고 다음 차트를 그려주세요") == "multi_step"
    
    def test_requires_data_analysis(self):
        """Test data analysis requirement detection"""
        # Requires data analysis
        assert self.orchestrator._requires_data_analysis("데이터를 분석해주세요") is True
        assert self.orchestrator._requires_data_analysis("차트를 그려주세요") is True
        
        # Does not require data analysis
        assert self.orchestrator._requires_data_analysis("안녕하세요") is False
        assert self.orchestrator._requires_data_analysis("오늘 날씨는?") is False
    
    @pytest.mark.asyncio
    async def test_prepare_data_without_session(self):
        """Test data preparation without session"""
        prepared = await self.orchestrator._prepare_data(
            data={"test": "data"},
            session_id=None,
            context={"extra": "info"}
        )
        
        assert prepared["provided_data"] == {"test": "data"}
        assert prepared["session_files"] == []
        assert prepared["context"] == {"extra": "info"}
    
    @pytest.mark.asyncio
    async def test_analyze_and_route_query_without_router(self):
        """Test query analysis when router is not available"""
        # Mock no router
        self.orchestrator.universal_router = None
        
        result = await self.orchestrator._analyze_and_route_query(
            "test query", {}, None, None
        )
        
        assert "error" in result
        assert result["error"] == "Universal router not available"
    
    @pytest.mark.asyncio
    async def test_execute_llm_task(self):
        """Test LLM task execution"""
        task = AgentTask(
            agent_id="llm_test",
            agent_type="llm",
            task_description="LLM test",
            input_data={"query": "테스트 질문"}
        )
        
        result = await self.orchestrator._execute_llm_task(task)
        
        assert result["success"] is True
        assert "response" in result
        assert result["type"] == "llm_direct"
        assert result["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_execute_detector_task_without_detector(self):
        """Test detector task when detector is not available"""
        # Mock no detector
        self.orchestrator.data_type_detector = None
        
        task = AgentTask(
            agent_id="detector_test",
            agent_type="detector",
            task_description="Detector test",
            input_data={"data": {"provided_data": "test"}}
        )
        
        result = await self.orchestrator._execute_detector_task(task)
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_execute_specialized_agent_task_without_data(self):
        """Test specialized agent task without data"""
        task = AgentTask(
            agent_id="specialized_test",
            agent_type="specialized",
            task_description="Specialized test",
            input_data={"query": "test", "data": {}}
        )
        
        result = await self.orchestrator._execute_specialized_agent_task(task)
        
        assert result["success"] is False
        assert "No data provided" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_universal_agent_task(self):
        """Test universal agent (pandas-ai) task execution"""
        task = AgentTask(
            agent_id="pandas_ai_test",
            agent_type="universal",
            task_description="Pandas-AI test",
            input_data={"query": "테스트 분석"}
        )
        
        result = await self.orchestrator._execute_universal_agent_task(task)
        
        assert result["success"] is True
        assert "response" in result
        assert result["type"] == "pandas_ai"
        assert result["confidence"] == 0.9
        assert "code_generated" in result
    
    @pytest.mark.asyncio
    async def test_create_orchestration_plan_simple(self):
        """Test orchestration plan creation for simple query"""
        routing_result = {
            "complexity": "simple",
            "requires_data": False
        }
        
        plan = await self.orchestrator._create_orchestration_plan(
            "안녕하세요", routing_result, {}
        )
        
        assert plan.strategy == OrchestrationStrategy.SINGLE_AGENT
        assert len(plan.tasks) == 1
        assert plan.tasks[0].agent_type == "llm"
        assert plan.estimated_duration == 10
    
    @pytest.mark.asyncio
    async def test_create_orchestration_plan_complex(self):
        """Test orchestration plan creation for complex query"""
        routing_result = {
            "complexity": "complex",
            "requires_data": True,
            "routing": {
                "success": True,
                "decision": {"recommended_agent": "eda"}
            }
        }
        
        plan = await self.orchestrator._create_orchestration_plan(
            "복잡한 데이터 분석", routing_result, {}
        )
        
        assert plan.strategy == OrchestrationStrategy.SEQUENTIAL
        assert len(plan.tasks) == 2
        assert plan.estimated_duration == 120
        assert plan.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_create_orchestration_plan_multi_step(self):
        """Test orchestration plan creation for multi-step query"""
        routing_result = {
            "complexity": "multi_step",
            "requires_data": True
        }
        
        plan = await self.orchestrator._create_orchestration_plan(
            "데이터를 분석하고 그 다음 모델을 만들어주세요", routing_result, {}
        )
        
        assert plan.strategy == OrchestrationStrategy.HIERARCHICAL
        assert len(plan.tasks) == 4
        assert plan.estimated_duration == 180
        assert plan.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_execute_orchestration_plan_single_agent(self):
        """Test executing single agent orchestration plan"""
        task = AgentTask(
            agent_id="test_agent",
            agent_type="llm",
            task_description="Test task",
            input_data={"query": "test"}
        )
        
        plan = OrchestrationPlan(
            strategy=OrchestrationStrategy.SINGLE_AGENT,
            tasks=[task],
            estimated_duration=30,
            confidence=0.8,
            reasoning="Test"
        )
        
        results = await self.orchestrator._execute_orchestration_plan(plan, None)
        
        assert "test_agent" in results
        assert results["test_agent"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_orchestration_plan_sequential(self):
        """Test executing sequential orchestration plan"""
        tasks = [
            AgentTask(
                agent_id="first_agent",
                agent_type="llm",
                task_description="First task",
                input_data={"query": "first"},
                priority=10
            ),
            AgentTask(
                agent_id="second_agent", 
                agent_type="llm",
                task_description="Second task",
                input_data={"query": "second"},
                dependencies=["first_agent"],
                priority=5
            )
        ]
        
        plan = OrchestrationPlan(
            strategy=OrchestrationStrategy.SEQUENTIAL,
            tasks=tasks,
            estimated_duration=60,
            confidence=0.9,
            reasoning="Test sequential"
        )
        
        results = await self.orchestrator._execute_orchestration_plan(plan, None)
        
        assert "first_agent" in results
        assert "second_agent" in results
        assert results["first_agent"]["success"] is True
        assert results["second_agent"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_integrate_and_interpret_results_success(self):
        """Test result integration with successful results"""
        execution_results = {
            "agent1": {
                "success": True,
                "insights": ["Insight 1", "Insight 2"],
                "recommendations": ["Rec 1"]
            },
            "agent2": {
                "success": True,
                "response": "Agent 2 response"
            }
        }
        
        plan = OrchestrationPlan(
            strategy=OrchestrationStrategy.SEQUENTIAL,
            tasks=[],
            estimated_duration=60,
            confidence=0.9,
            reasoning="Test"
        )
        
        result = await self.orchestrator._integrate_and_interpret_results(
            "test query", execution_results, plan
        )
        
        assert result.success is True
        assert len(result.insights) == 3  # 2 insights + 1 response
        assert len(result.recommendations) == 1
        assert result.errors is None
        assert result.metadata["successful_tasks"] == 2
    
    @pytest.mark.asyncio
    async def test_integrate_and_interpret_results_with_errors(self):
        """Test result integration with errors"""
        execution_results = {
            "agent1": {
                "success": True,
                "insights": ["Insight 1"]
            },
            "agent2": {
                "success": False,
                "error": "Agent 2 failed"
            }
        }
        
        plan = OrchestrationPlan(
            strategy=OrchestrationStrategy.SEQUENTIAL,
            tasks=[],
            estimated_duration=60,
            confidence=0.9,
            reasoning="Test"
        )
        
        result = await self.orchestrator._integrate_and_interpret_results(
            "test query", execution_results, plan
        )
        
        assert result.success is False  # Has errors
        assert len(result.errors) == 1
        assert "Agent 2 failed" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_orchestrate_analysis_simple_workflow(self):
        """Test complete orchestration workflow with simple query"""
        result = await self.orchestrator.orchestrate_analysis(
            user_query="안녕하세요",
            data=None,
            session_id="test_session"
        )
        
        assert isinstance(result, OrchestrationResult)
        assert result.execution_time > 0
        # Results may vary based on system availability
    
    def test_get_orchestration_statistics_empty(self):
        """Test statistics when no execution history"""
        stats = self.orchestrator.get_orchestration_statistics()
        
        assert stats["total_executions"] == 0
        assert stats["success_rate"] == 0
        assert stats["average_execution_time"] == 0
        assert stats["strategy_distribution"] == {}
        assert stats["recent_executions"] == []
    
    @pytest.mark.asyncio
    async def test_get_orchestration_statistics_with_data(self):
        """Test statistics with execution history"""
        # Run some orchestrations
        await self.orchestrator.orchestrate_analysis("테스트 1", session_id="s1")
        await self.orchestrator.orchestrate_analysis("테스트 2", session_id="s2")
        
        stats = self.orchestrator.get_orchestration_statistics()
        
        assert stats["total_executions"] == 2
        assert isinstance(stats["success_rate"], float)
        assert isinstance(stats["average_execution_time"], float)
        assert isinstance(stats["strategy_distribution"], dict)
        assert len(stats["recent_executions"]) <= 5
    
    def test_clear_execution_history(self):
        """Test clearing execution history"""
        # Add some fake history
        self.orchestrator.execution_history = [{"test": "data"}]
        assert len(self.orchestrator.execution_history) == 1
        
        self.orchestrator.clear_execution_history()
        assert len(self.orchestrator.execution_history) == 0


class TestMultiAgentA2AExecutor:
    """Test MultiAgentA2AExecutor class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.orchestrator = MultiAgentOrchestrator()
        self.executor = MultiAgentA2AExecutor(self.orchestrator)
    
    def test_initialization(self):
        """Test executor initialization"""
        assert self.executor.orchestrator is self.orchestrator
    
    def test_extract_user_input_empty_context(self):
        """Test user input extraction with empty context"""
        mock_context = Mock()
        mock_context.current_task = None
        
        user_input = self.executor._extract_user_input(mock_context)
        assert user_input == "데이터 분석을 수행해주세요"
    
    def test_extract_user_input_with_message(self):
        """Test user input extraction with message"""
        mock_part = Mock()
        mock_part.text = "사용자 질문입니다"
        
        mock_message = Mock()
        mock_message.parts = [mock_part]
        
        mock_task = Mock()
        mock_task.message = mock_message
        
        mock_context = Mock()
        mock_context.current_task = mock_task
        
        user_input = self.executor._extract_user_input(mock_context)
        assert user_input == "사용자 질문입니다"


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_get_multi_agent_orchestrator(self):
        """Test orchestrator singleton"""
        orchestrator1 = get_multi_agent_orchestrator()
        orchestrator2 = get_multi_agent_orchestrator()
        
        assert orchestrator1 is orchestrator2  # Should be same instance
    
    def test_get_multi_agent_orchestrator_with_config(self):
        """Test orchestrator creation with config"""
        config = {"test_param": "test_value"}
        
        # Note: This will still return the singleton, but we can test the concept
        orchestrator = get_multi_agent_orchestrator(config)
        assert isinstance(orchestrator, MultiAgentOrchestrator)
    
    def test_create_multi_agent_card(self):
        """Test A2A card creation"""
        card = create_multi_agent_card()
        
        # May be None if A2A SDK is not available
        if card:
            assert card.name == "Multi-Agent Data Analysis Orchestrator"
            assert "지능형 멀티 에이전트" in card.description
            assert len(card.capabilities.skills) > 0
    
    def test_create_multi_agent_a2a_app(self):
        """Test A2A app creation"""
        app = create_multi_agent_a2a_app()
        
        # May be None if A2A SDK is not available
        # Just test that it doesn't crash
        assert app is None or hasattr(app, 'app')


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_simple_query(self):
        """Test end-to-end simple query workflow"""
        orchestrator = MultiAgentOrchestrator()
        
        result = await orchestrator.orchestrate_analysis(
            user_query="안녕하세요",
            data=None,
            session_id="integration_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_analysis(self):
        """Test end-to-end data analysis workflow"""
        orchestrator = MultiAgentOrchestrator()
        
        # Sample data
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10]
        })
        
        result = await orchestrator.orchestrate_analysis(
            user_query="데이터를 분석해주세요",
            data=df,
            session_id="data_analysis_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        # Result success may vary based on system availability
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_multiple_orchestrations(self):
        """Test multiple orchestrations"""
        orchestrator = MultiAgentOrchestrator()
        
        queries = [
            "간단한 질문",
            "데이터 분석",
            "복잡한 분석과 시각화"
        ]
        
        results = []
        for query in queries:
            result = await orchestrator.orchestrate_analysis(
                user_query=query,
                session_id=f"multi_test_{len(results)}"
            )
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, OrchestrationResult) for r in results)
        
        # Check statistics
        stats = orchestrator.get_orchestration_statistics()
        assert stats["total_executions"] == 3


if __name__ == "__main__":
    pytest.main([__file__]) 