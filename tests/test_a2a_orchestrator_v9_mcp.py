#!/usr/bin/env python3
"""
A2A Orchestrator v9.0 MCP Enhanced 테스트 스위트

A2A Message Router v9.0의 MCP 통합 및 지능형 라우팅 기능을 종합적으로 테스트

Test Coverage:
- 지능형 의도 분석 테스트
- MCP 통합 에이전트 발견 테스트
- Enhanced 협업 엔진 테스트
- 워크플로우 실행 테스트
- A2A + MCP 통합 테스트
- Context Engineering 6 레이어 테스트
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers'))

# 테스트 대상 임포트
from a2a_orchestrator_v9_mcp_enhanced import (
    IntelligentIntentAnalyzer,
    MCPEnhancedAgentDiscovery,
    EnhancedCollaborationEngine,
    UniversalIntelligentOrchestratorV9,
    IntentAnalysisResult,
    EnhancedWorkflowStep,
    CollaborationSession,
    AGENT_PORTS,
    MCP_TOOL_PORTS
)

class TestIntelligentIntentAnalyzer:
    """지능형 의도 분석기 테스트"""
    
    @pytest.fixture
    def intent_analyzer(self):
        """의도 분석기 픽스처"""
        return IntelligentIntentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_intent_data_analysis(self, intent_analyzer):
        """데이터 분석 의도 분석 테스트"""
        user_request = "데이터를 분석하고 통계를 보여주세요"
        
        result = await intent_analyzer.analyze_intent(user_request)
        
        assert isinstance(result, IntentAnalysisResult)
        assert result.primary_intent == "data_analysis"
        assert result.confidence > 0
        assert "pandas_collaboration_hub" in result.required_agents
        assert "data_analyzer" in result.required_mcp_tools
        assert result.workflow_type in ["simple", "complex", "collaborative"]
        assert result.estimated_complexity in ["low", "medium", "high"]
    
    @pytest.mark.asyncio
    async def test_analyze_intent_web_scraping(self, intent_analyzer):
        """웹 스크래핑 의도 분석 테스트"""
        user_request = "웹사이트에서 데이터를 수집해주세요"
        
        result = await intent_analyzer.analyze_intent(user_request)
        
        assert result.primary_intent == "web_scraping"
        assert "data_loader" in result.required_agents
        assert "playwright" in result.required_mcp_tools
        assert result.workflow_type != "simple"  # 웹 스크래핑은 복잡한 작업
    
    @pytest.mark.asyncio
    async def test_analyze_intent_comprehensive(self, intent_analyzer):
        """종합 분석 의도 분석 테스트"""
        user_request = "모든 데이터를 종합적으로 분석하고 시각화해주세요"
        
        result = await intent_analyzer.analyze_intent(user_request)
        
        assert result.primary_intent == "comprehensive_analysis"
        assert len(result.required_agents) >= 2
        assert len(result.required_mcp_tools) >= 2
        assert result.workflow_type == "collaborative"
        assert result.estimated_complexity == "high"
    
    @pytest.mark.asyncio
    async def test_analyze_intent_machine_learning(self, intent_analyzer):
        """머신러닝 의도 분석 테스트"""
        user_request = "머신러닝 모델을 만들고 예측해주세요"
        
        result = await intent_analyzer.analyze_intent(user_request)
        
        assert result.primary_intent == "machine_learning"
        assert "h2o_ml" in result.required_agents
        assert "llm_gateway" in result.required_mcp_tools
        assert result.priority >= 5  # 머신러닝은 우선순위 높음
    
    def test_calculate_keyword_score(self, intent_analyzer):
        """키워드 점수 계산 테스트"""
        text = "데이터를 분석하고 통계를 보여주세요"
        keywords = ["분석", "통계", "데이터"]
        
        score = intent_analyzer._calculate_keyword_score(text, keywords)
        
        assert score == 1.0  # 모든 키워드가 매치
    
    def test_determine_workflow_type(self, intent_analyzer):
        """워크플로우 타입 결정 테스트"""
        # Simple workflow
        simple_type = intent_analyzer._determine_workflow_type(["agent1"], ["tool1"])
        assert simple_type == "simple"
        
        # Complex workflow  
        complex_type = intent_analyzer._determine_workflow_type(["agent1", "agent2"], ["tool1", "tool2"])
        assert complex_type == "complex"
        
        # Collaborative workflow
        collab_type = intent_analyzer._determine_workflow_type(
            ["agent1", "agent2", "agent3"], 
            ["tool1", "tool2", "tool3"]
        )
        assert collab_type == "collaborative"
    
    def test_estimate_complexity(self, intent_analyzer):
        """복잡성 추정 테스트"""
        # High complexity
        high_complexity = intent_analyzer._estimate_complexity(
            "모든 데이터를 종합적으로 분석", 
            ["agent1", "agent2", "agent3"], 
            ["tool1", "tool2", "tool3"]
        )
        assert high_complexity == "high"
        
        # Low complexity
        low_complexity = intent_analyzer._estimate_complexity(
            "간단한 분석", 
            ["agent1"], 
            ["tool1"]
        )
        assert low_complexity == "low"

class TestMCPEnhancedAgentDiscovery:
    """MCP 통합 에이전트 발견 테스트"""
    
    @pytest.fixture
    def agent_discovery(self):
        """에이전트 발견 픽스처"""
        return MCPEnhancedAgentDiscovery()
    
    @pytest.mark.asyncio
    async def test_discover_all_resources(self, agent_discovery):
        """전체 리소스 발견 테스트"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock HTTP 응답
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "Test Agent",
                "description": "Test Description",
                "capabilities": {},
                "skills": []
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # MCP 통합 mock
            with patch.object(agent_discovery.mcp_integration, 'initialize_mcp_tools') as mock_mcp:
                mock_mcp.return_value = {
                    "total_tools": 3,
                    "available_tools": 3,
                    "tool_details": {
                        "playwright": {"name": "Playwright", "type": "browser", "status": "available"},
                        "file_manager": {"name": "File Manager", "type": "file_system", "status": "available"},
                        "data_analyzer": {"name": "Data Analyzer", "type": "analysis", "status": "available"}
                    }
                }
                
                result = await agent_discovery.discover_all_resources()
                
                assert "a2a_agents" in result
                assert "mcp_tools" in result
                assert result["total_resources"] > 0
                assert result["integration_status"] in ["enhanced", "partial"]
    
    @pytest.mark.asyncio
    async def test_discover_a2a_agents(self, agent_discovery):
        """A2A 에이전트 발견 테스트"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful agent response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "Test Agent",
                "description": "Test Description",
                "capabilities": {"streaming": True},
                "skills": [{"id": "test_skill", "name": "Test Skill"}]
            }
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            agents = await agent_discovery._discover_a2a_agents()
            
            assert len(agents) > 0
            for agent_id, agent_info in agents.items():
                assert agent_info["type"] == "a2a_agent"
                assert agent_info["status"] == "available"
                assert "url" in agent_info
                assert "port" in agent_info
    
    @pytest.mark.asyncio
    async def test_discover_mcp_tools(self, agent_discovery):
        """MCP 도구 발견 테스트"""
        with patch.object(agent_discovery.mcp_integration, 'initialize_mcp_tools') as mock_mcp:
            mock_mcp.return_value = {
                "total_tools": 5,
                "available_tools": 5,
                "tool_details": {
                    "playwright": {"name": "Playwright", "type": "browser", "status": "available"},
                    "data_analyzer": {"name": "Data Analyzer", "type": "analysis", "status": "available"},
                    "chart_generator": {"name": "Chart Generator", "type": "visualization", "status": "available"}
                }
            }
            
            tools = await agent_discovery._discover_mcp_tools()
            
            assert len(tools) == 3
            for tool_id, tool_info in tools.items():
                assert tool_info["type"] == "mcp_tool"
                assert "tool_type" in tool_info
                assert "capabilities" in tool_info
                assert "status" in tool_info

class TestEnhancedCollaborationEngine:
    """향상된 협업 엔진 테스트"""
    
    @pytest.fixture
    def collaboration_engine(self):
        """협업 엔진 픽스처"""
        return EnhancedCollaborationEngine()
    
    @pytest.fixture
    def sample_intent_analysis(self):
        """샘플 의도 분석 결과"""
        return IntentAnalysisResult(
            primary_intent="data_analysis",
            confidence=0.8,
            required_agents=["pandas_collaboration_hub", "eda_tools"],
            required_mcp_tools=["data_analyzer", "chart_generator"],
            workflow_type="complex",
            priority=7,
            estimated_complexity="medium",
            context_requirements={}
        )
    
    @pytest.mark.asyncio
    async def test_create_collaboration_session(self, collaboration_engine, sample_intent_analysis):
        """협업 세션 생성 테스트"""
        user_request = "데이터를 분석해주세요"
        
        session = await collaboration_engine.create_collaboration_session(
            user_request, sample_intent_analysis
        )
        
        assert isinstance(session, CollaborationSession)
        assert session.user_request == user_request
        assert session.intent_analysis == sample_intent_analysis
        assert len(session.workflow_steps) > 0
        assert session.status == "pending"
        assert len(session.active_agents) > 0
        assert len(session.active_mcp_tools) > 0
    
    @pytest.mark.asyncio
    async def test_generate_workflow_steps(self, collaboration_engine, sample_intent_analysis):
        """워크플로우 단계 생성 테스트"""
        user_request = "데이터를 분석하고 시각화해주세요"
        
        steps = await collaboration_engine._generate_workflow_steps(
            user_request, sample_intent_analysis
        )
        
        assert len(steps) > 0
        
        # MCP 단계 확인
        mcp_steps = [step for step in steps if step.step_type == "mcp_tool"]
        assert len(mcp_steps) > 0
        
        # 에이전트 단계 확인
        agent_steps = [step for step in steps if step.step_type == "agent"]
        assert len(agent_steps) > 0
        
        # 통합 단계 확인 (여러 단계가 있는 경우)
        if len(steps) > 1:
            integration_steps = [step for step in steps if step.step_type == "collaboration"]
            assert len(integration_steps) > 0
    
    def test_determine_mcp_action(self, collaboration_engine):
        """MCP 액션 결정 테스트"""
        # 브라우저 도구 액션
        action = collaboration_engine._determine_mcp_action("playwright", "데이터 추출")
        assert action == "extract_data"
        
        # 파일 관리자 액션
        action = collaboration_engine._determine_mcp_action("file_manager", "파일 읽기")
        assert action == "read_file"
        
        # 데이터 분석기 액션
        action = collaboration_engine._determine_mcp_action("data_analyzer", "분석")
        assert action == "statistical_analysis"
    
    def test_generate_mcp_parameters(self, collaboration_engine):
        """MCP 매개변수 생성 테스트"""
        user_request = "데이터를 분석해주세요"
        
        params = collaboration_engine._generate_mcp_parameters(
            "data_analyzer", "statistical_analysis", user_request
        )
        
        assert "user_request" in params
        assert "timestamp" in params
        assert "analysis_type" in params
        assert params["user_request"] == user_request
    
    def test_create_enhanced_request(self, collaboration_engine):
        """향상된 요청 생성 테스트"""
        original_request = "데이터를 분석해주세요"
        mcp_results = {
            "mcp_1_data_analyzer": {
                "success": True,
                "result": {"mean": 42.5, "std": 15.2}
            },
            "mcp_2_chart_generator": {
                "success": True,
                "result": {"chart_path": "/tmp/chart.png"}
            }
        }
        
        enhanced_request = collaboration_engine._create_enhanced_request(
            original_request, mcp_results
        )
        
        assert original_request in enhanced_request
        assert "MCP 도구 분석 결과" in enhanced_request
        assert "mcp_1_data_analyzer" in enhanced_request
        assert "mcp_2_chart_generator" in enhanced_request

class TestUniversalIntelligentOrchestratorV9:
    """Universal Intelligent Orchestrator v9.0 테스트"""
    
    @pytest.fixture
    def orchestrator(self):
        """오케스트레이터 픽스처"""
        return UniversalIntelligentOrchestratorV9()
    
    def test_orchestrator_initialization(self, orchestrator):
        """오케스트레이터 초기화 테스트"""
        assert orchestrator.intent_analyzer is not None
        assert orchestrator.resource_discovery is not None
        assert orchestrator.collaboration_engine is not None
        assert isinstance(orchestrator.available_resources, dict)
        assert isinstance(orchestrator.active_sessions, dict)
        assert isinstance(orchestrator.performance_metrics, dict)
    
    def test_extract_user_request(self, orchestrator):
        """사용자 요청 추출 테스트"""
        # Mock RequestContext 생성
        mock_context = MagicMock()
        mock_part = MagicMock()
        mock_part.root.text = "테스트 요청"
        mock_context.message.parts = [mock_part]
        
        result = orchestrator._extract_user_request(mock_context)
        
        assert result == "테스트 요청"
    
    def test_generate_final_response(self, orchestrator):
        """최종 응답 생성 테스트"""
        user_request = "데이터 분석 요청"
        
        intent_analysis = IntentAnalysisResult(
            primary_intent="data_analysis",
            confidence=0.85,
            required_agents=["pandas_collaboration_hub"],
            required_mcp_tools=["data_analyzer"],
            workflow_type="complex",
            priority=8,
            estimated_complexity="medium",
            context_requirements={}
        )
        
        execution_results = {
            "session_id": "test_session",
            "mcp_results": {"mcp_1": {"success": True}},
            "agent_results": {"agent_1": {"success": True}},
            "execution_timeline": [{"step_id": "step1", "duration": 2.0}]
        }
        
        session = CollaborationSession(
            session_id="test_session",
            user_request=user_request,
            intent_analysis=intent_analysis,
            workflow_steps=[],
            active_agents={"pandas_collaboration_hub"},
            active_mcp_tools={"data_analyzer"},
            status="completed",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            results=execution_results
        )
        
        response = orchestrator._generate_final_response(
            user_request, intent_analysis, execution_results, session
        )
        
        assert "A2A v9.0 MCP Enhanced" in response
        assert user_request in response
        assert "data_analysis" in response
        assert "pandas_collaboration_hub" in response
        assert "data_analyzer" in response
        assert "test_session" in response

class TestDataStructures:
    """데이터 구조 테스트"""
    
    def test_intent_analysis_result(self):
        """의도 분석 결과 데이터 구조 테스트"""
        result = IntentAnalysisResult(
            primary_intent="test_intent",
            confidence=0.9,
            required_agents=["agent1", "agent2"],
            required_mcp_tools=["tool1", "tool2"],
            workflow_type="complex",
            priority=5,
            estimated_complexity="high",
            context_requirements={"key": "value"}
        )
        
        assert result.primary_intent == "test_intent"
        assert result.confidence == 0.9
        assert len(result.required_agents) == 2
        assert len(result.required_mcp_tools) == 2
        assert result.workflow_type == "complex"
        assert result.priority == 5
        assert result.estimated_complexity == "high"
        assert result.context_requirements == {"key": "value"}
    
    def test_enhanced_workflow_step(self):
        """향상된 워크플로우 단계 데이터 구조 테스트"""
        step = EnhancedWorkflowStep(
            step_id="test_step",
            step_type="mcp_tool",
            executor="data_analyzer",
            action="analyze",
            parameters={"param1": "value1"},
            dependencies=["step1", "step2"],
            estimated_duration=3.0,
            parallel_execution=True
        )
        
        assert step.step_id == "test_step"
        assert step.step_type == "mcp_tool"
        assert step.executor == "data_analyzer"
        assert step.action == "analyze"
        assert step.parameters == {"param1": "value1"}
        assert step.dependencies == ["step1", "step2"]
        assert step.estimated_duration == 3.0
        assert step.parallel_execution is True
    
    def test_collaboration_session(self):
        """협업 세션 데이터 구조 테스트"""
        intent_analysis = IntentAnalysisResult(
            primary_intent="test",
            confidence=0.8,
            required_agents=["agent1"],
            required_mcp_tools=["tool1"],
            workflow_type="simple",
            priority=5,
            estimated_complexity="low",
            context_requirements={}
        )
        
        session = CollaborationSession(
            session_id="test_session",
            user_request="test request",
            intent_analysis=intent_analysis,
            workflow_steps=[],
            active_agents={"agent1"},
            active_mcp_tools={"tool1"},
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            results={}
        )
        
        assert session.session_id == "test_session"
        assert session.user_request == "test request"
        assert session.intent_analysis == intent_analysis
        assert session.status == "pending"
        assert "agent1" in session.active_agents
        assert "tool1" in session.active_mcp_tools

class TestConstants:
    """상수 및 설정 테스트"""
    
    def test_agent_ports(self):
        """에이전트 포트 매핑 테스트"""
        assert isinstance(AGENT_PORTS, dict)
        assert len(AGENT_PORTS) > 0
        assert "pandas_collaboration_hub" in AGENT_PORTS
        assert AGENT_PORTS["pandas_collaboration_hub"] == 8315
        
        # 포트 번호 유효성 검사
        for agent_id, port in AGENT_PORTS.items():
            assert isinstance(port, int)
            assert 8000 <= port <= 9000
    
    def test_mcp_tool_ports(self):
        """MCP 도구 포트 매핑 테스트"""
        assert isinstance(MCP_TOOL_PORTS, dict)
        assert len(MCP_TOOL_PORTS) > 0
        assert "playwright" in MCP_TOOL_PORTS
        assert MCP_TOOL_PORTS["playwright"] == 3000
        
        # 포트 번호 유효성 검사
        for tool_id, port in MCP_TOOL_PORTS.items():
            assert isinstance(port, int)
            assert 3000 <= port <= 4000

if __name__ == "__main__":
    # 테스트 실행
    import subprocess
    import sys
    
    print("🔗 A2A Orchestrator v9.0 MCP Enhanced 테스트 실행 중...")
    
    # pytest 실행
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\n🎯 테스트 결과: {'✅ 성공' if result.returncode == 0 else '❌ 실패'}")
    
    if result.returncode == 0:
        print("🌟 A2A Orchestrator v9.0 MCP Enhanced가 모든 테스트를 통과했습니다!")
        print("✨ 지능형 의도 분석, MCP 통합, Enhanced 협업 엔진이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.") 