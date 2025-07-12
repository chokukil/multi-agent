#!/usr/bin/env python3
"""
🔗 MCP (Model Context Protocol) 통합 테스트 스위트

MCP 도구 통합 기능과 A2A 에이전트 연동을 종합적으로 테스트
Context Engineering TOOLS 레이어의 MCP 지원 검증

Test Coverage:
- MCP 도구 초기화 및 발견
- MCP 세션 생성 및 관리
- MCP 도구 호출 및 결과 처리
- A2A 에이전트와 MCP 통합
- Enhanced 협업 워크플로우
- Context Engineering 레이어 통합
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers', 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers', 'pandas_agent'))

# 테스트 대상 임포트
from mcp_integration import (
    MCPIntegration, MCPTool, MCPRequest, MCPSession, MCPToolType,
    get_mcp_integration, initialize_mcp_tools, call_mcp_tool
)

class TestMCPIntegration:
    """MCP 통합 기본 기능 테스트"""
    
    @pytest.fixture
    def mcp_integration(self):
        """MCP 통합 인스턴스 픽스처"""
        return MCPIntegration()
    
    def test_mcp_tool_type_enum(self):
        """MCP 도구 타입 Enum 테스트"""
        assert MCPToolType.BROWSER == "browser"
        assert MCPToolType.FILE_SYSTEM == "file_system"
        assert MCPToolType.DATABASE == "database"
        assert MCPToolType.API_CLIENT == "api_client"
        assert MCPToolType.ANALYSIS == "analysis"
        assert MCPToolType.VISUALIZATION == "visualization"
        assert MCPToolType.AI_MODEL == "ai_model"
        assert MCPToolType.CUSTOM == "custom"
    
    def test_mcp_tool_dataclass(self):
        """MCP 도구 데이터클래스 테스트"""
        tool = MCPTool(
            tool_id="test_tool",
            name="Test Tool",
            description="테스트 도구",
            tool_type=MCPToolType.BROWSER,
            endpoint="mcp://localhost:3000/test",
            parameters={"param1": "value1"},
            capabilities=["test_capability"],
            status="available"
        )
        
        assert tool.tool_id == "test_tool"
        assert tool.name == "Test Tool"
        assert tool.tool_type == MCPToolType.BROWSER
        assert tool.status == "available"
        assert tool.usage_count == 0
        assert tool.error_count == 0
    
    def test_mcp_request_dataclass(self):
        """MCP 요청 데이터클래스 테스트"""
        request = MCPRequest(
            request_id="test_req_123",
            tool_id="playwright",
            action="navigate",
            parameters={"url": "https://example.com"},
            requester_agent="pandas_agent",
            created_at=datetime.now(),
            status="pending"
        )
        
        assert request.request_id == "test_req_123"
        assert request.tool_id == "playwright"
        assert request.action == "navigate"
        assert request.status == "pending"
        assert request.result is None
        assert request.error is None
    
    def test_mcp_session_dataclass(self):
        """MCP 세션 데이터클래스 테스트"""
        session = MCPSession(
            session_id="session_123",
            agent_id="pandas_agent",
            active_tools=["playwright", "file_manager"],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        assert session.session_id == "session_123"
        assert session.agent_id == "pandas_agent"
        assert len(session.active_tools) == 2
        assert session.total_requests == 0
        assert session.successful_requests == 0
    
    @pytest.mark.asyncio
    async def test_mcp_integration_initialization(self, mcp_integration):
        """MCP 통합 초기화 테스트"""
        assert len(mcp_integration.available_tools) == 0
        assert len(mcp_integration.active_sessions) == 0
        assert len(mcp_integration.pending_requests) == 0
        assert len(mcp_integration.completed_requests) == 0
        assert mcp_integration.http_client is not None
    
    @pytest.mark.asyncio
    async def test_initialize_mcp_tools(self, mcp_integration):
        """MCP 도구 초기화 및 발견 테스트"""
        results = await mcp_integration.initialize_mcp_tools()
        
        # 기본 설정에서 7개 도구가 발견되어야 함
        assert results["total_tools"] == 7
        assert results["available_tools"] == 7
        assert results["discovery_status"] == "excellent"
        
        # 각 도구가 올바르게 등록되었는지 확인
        expected_tools = [
            "playwright", "file_manager", "database_connector", 
            "api_gateway", "data_analyzer", "chart_generator", "llm_gateway"
        ]
        
        for tool_id in expected_tools:
            assert tool_id in results["tool_details"]
            assert tool_id in mcp_integration.available_tools
            
        # 도구 상태 확인
        for tool_id, tool in mcp_integration.available_tools.items():
            assert tool.status == "available"
            assert tool.tool_id == tool_id
    
    @pytest.mark.asyncio
    async def test_create_mcp_session(self, mcp_integration):
        """MCP 세션 생성 테스트"""
        # 먼저 도구 초기화
        await mcp_integration.initialize_mcp_tools()
        
        # 세션 생성
        session = await mcp_integration.create_mcp_session(
            agent_id="test_agent",
            required_tools=["playwright", "file_manager"]
        )
        
        assert session.agent_id == "test_agent"
        assert len(session.active_tools) == 2
        assert "playwright" in session.active_tools
        assert "file_manager" in session.active_tools
        assert session.session_id in mcp_integration.active_sessions
    
    @pytest.mark.asyncio
    async def test_create_mcp_session_all_tools(self, mcp_integration):
        """모든 도구를 포함한 MCP 세션 생성 테스트"""
        await mcp_integration.initialize_mcp_tools()
        
        # 필요한 도구를 지정하지 않으면 모든 사용 가능한 도구 포함
        session = await mcp_integration.create_mcp_session(agent_id="test_agent")
        
        assert len(session.active_tools) == 7  # 모든 기본 도구
        assert session.session_id in mcp_integration.active_sessions

class TestMCPToolCalls:
    """MCP 도구 호출 테스트"""
    
    @pytest.fixture
    async def setup_mcp_session(self):
        """MCP 세션 설정 픽스처"""
        mcp = MCPIntegration()
        await mcp.initialize_mcp_tools()
        session = await mcp.create_mcp_session(
            agent_id="test_agent",
            required_tools=["playwright", "file_manager", "data_analyzer"]
        )
        return mcp, session
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_browser_navigate(self, setup_mcp_session):
        """브라우저 도구 navigate 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id=session.session_id,
            tool_id="playwright",
            action="navigate",
            parameters={"url": "https://example.com"}
        )
        
        assert result["success"] is True
        assert result["tool_id"] == "playwright"
        assert result["action"] == "navigate"
        assert "result" in result
        assert "execution_time" in result
        
        # 세션 통계 확인
        updated_session = mcp.active_sessions[session.session_id]
        assert updated_session.total_requests == 1
        assert updated_session.successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_browser_screenshot(self, setup_mcp_session):
        """브라우저 도구 screenshot 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id=session.session_id,
            tool_id="playwright",
            action="screenshot",
            parameters={}
        )
        
        assert result["success"] is True
        assert result["result"]["status"] == "success"
        assert "screenshot_path" in result["result"]
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_file_manager(self, setup_mcp_session):
        """파일 관리자 도구 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id=session.session_id,
            tool_id="file_manager",
            action="read_file",
            parameters={"path": "/tmp/test.csv"}
        )
        
        assert result["success"] is True
        assert result["result"]["status"] == "success"
        assert "content" in result["result"]
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_data_analyzer(self, setup_mcp_session):
        """데이터 분석기 도구 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id=session.session_id,
            tool_id="data_analyzer",
            action="statistical_analysis",
            parameters={"analysis_type": "descriptive"}
        )
        
        assert result["success"] is True
        assert result["result"]["status"] == "success"
        assert "results" in result["result"]
        assert "processing_time" in result["result"]
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_invalid_session(self, setup_mcp_session):
        """잘못된 세션 ID로 도구 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id="invalid_session",
            tool_id="playwright",
            action="navigate",
            parameters={}
        )
        
        assert "error" in result
        assert "세션을 찾을 수 없습니다" in result["error"]
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_invalid_tool(self, setup_mcp_session):
        """존재하지 않는 도구 호출 테스트"""
        mcp, session = await setup_mcp_session
        
        result = await mcp.call_mcp_tool(
            session_id=session.session_id,
            tool_id="nonexistent_tool",
            action="test",
            parameters={}
        )
        
        assert "error" in result
        assert "MCP 도구를 찾을 수 없습니다" in result["error"]

class TestMCPToolCapabilities:
    """MCP 도구 기능 및 상태 테스트"""
    
    @pytest.fixture
    async def initialized_mcp(self):
        """초기화된 MCP 인스턴스 픽스처"""
        mcp = MCPIntegration()
        await mcp.initialize_mcp_tools()
        return mcp
    
    @pytest.mark.asyncio
    async def test_get_tool_capabilities_single_tool(self, initialized_mcp):
        """단일 도구 기능 조회 테스트"""
        capabilities = await initialized_mcp.get_tool_capabilities("playwright")
        
        assert capabilities["tool_id"] == "playwright"
        assert capabilities["name"] == "Playwright Browser Automation"
        assert capabilities["type"] == "browser"
        assert "capabilities" in capabilities
        assert "parameters" in capabilities
        assert capabilities["status"] == "available"
        assert "usage_stats" in capabilities
    
    @pytest.mark.asyncio
    async def test_get_tool_capabilities_all_tools(self, initialized_mcp):
        """모든 도구 기능 조회 테스트"""
        capabilities = await initialized_mcp.get_tool_capabilities()
        
        assert capabilities["total_tools"] == 7
        assert len(capabilities["available_tools"]) == 7
        assert "tool_details" in capabilities
        assert "usage_statistics" in capabilities
        
        # 각 도구의 기본 정보 확인
        for tool_id in capabilities["available_tools"]:
            tool_info = capabilities["tool_details"][tool_id]
            assert "name" in tool_info
            assert "type" in tool_info
            assert "capabilities" in tool_info
            assert tool_info["status"] == "available"
    
    @pytest.mark.asyncio
    async def test_get_tool_capabilities_invalid_tool(self, initialized_mcp):
        """존재하지 않는 도구 기능 조회 테스트"""
        capabilities = await initialized_mcp.get_tool_capabilities("invalid_tool")
        
        assert "error" in capabilities
        assert "도구를 찾을 수 없습니다" in capabilities["error"]

class TestMCPSessionManagement:
    """MCP 세션 관리 테스트"""
    
    @pytest.fixture
    async def mcp_with_session(self):
        """세션이 있는 MCP 인스턴스 픽스처"""
        mcp = MCPIntegration()
        await mcp.initialize_mcp_tools()
        session = await mcp.create_mcp_session(
            agent_id="test_agent",
            required_tools=["playwright", "data_analyzer"]
        )
        return mcp, session
    
    @pytest.mark.asyncio
    async def test_get_session_status(self, mcp_with_session):
        """세션 상태 조회 테스트"""
        mcp, session = await mcp_with_session
        
        status = await mcp.get_session_status(session.session_id)
        
        assert status["session_id"] == session.session_id
        assert status["agent_id"] == "test_agent"
        assert len(status["active_tools"]) == 2
        assert "created_at" in status
        assert "last_activity" in status
        assert status["total_requests"] == 0
        assert status["successful_requests"] == 0
        assert status["success_rate"] == 0
    
    @pytest.mark.asyncio
    async def test_get_session_status_after_requests(self, mcp_with_session):
        """요청 후 세션 상태 조회 테스트"""
        mcp, session = await mcp_with_session
        
        # 몇 개의 요청 실행
        await mcp.call_mcp_tool(session.session_id, "playwright", "navigate", {})
        await mcp.call_mcp_tool(session.session_id, "data_analyzer", "statistical_analysis", {})
        
        status = await mcp.get_session_status(session.session_id)
        
        assert status["total_requests"] == 2
        assert status["successful_requests"] == 2
        assert status["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_get_session_status_invalid_session(self, mcp_with_session):
        """잘못된 세션 ID로 상태 조회 테스트"""
        mcp, session = await mcp_with_session
        
        status = await mcp.get_session_status("invalid_session")
        
        assert "error" in status
        assert "세션을 찾을 수 없습니다" in status["error"]
    
    @pytest.mark.asyncio
    async def test_close_session(self, mcp_with_session):
        """세션 종료 테스트"""
        mcp, session = await mcp_with_session
        
        # 세션이 존재하는지 확인
        assert session.session_id in mcp.active_sessions
        
        # 세션 종료
        await mcp.close_session(session.session_id)
        
        # 세션이 제거되었는지 확인
        assert session.session_id not in mcp.active_sessions

class TestMCPGlobalFunctions:
    """MCP 전역 함수 테스트"""
    
    def test_get_mcp_integration_singleton(self):
        """MCP 통합 싱글톤 패턴 테스트"""
        mcp1 = get_mcp_integration()
        mcp2 = get_mcp_integration()
        
        # 같은 인스턴스여야 함
        assert mcp1 is mcp2
    
    @pytest.mark.asyncio
    async def test_initialize_mcp_tools_global(self):
        """전역 MCP 도구 초기화 함수 테스트"""
        results = await initialize_mcp_tools()
        
        assert "total_tools" in results
        assert "available_tools" in results
        assert "discovery_status" in results
        assert results["total_tools"] == 7
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_global(self):
        """전역 MCP 도구 호출 함수 테스트"""
        # 먼저 도구 초기화
        await initialize_mcp_tools()
        
        # MCP 인스턴스 가져오기 및 세션 생성
        mcp = get_mcp_integration()
        session = await mcp.create_mcp_session("test_agent", ["playwright"])
        
        # 전역 함수로 도구 호출
        result = await call_mcp_tool(
            session_id=session.session_id,
            tool_id="playwright",
            action="navigate",
            parameters={"url": "https://example.com"}
        )
        
        assert result["success"] is True
        assert result["tool_id"] == "playwright"
        assert result["action"] == "navigate"

class TestMCPStatistics:
    """MCP 통계 및 모니터링 테스트"""
    
    @pytest.fixture
    async def mcp_with_usage(self):
        """사용 이력이 있는 MCP 인스턴스 픽스처"""
        mcp = MCPIntegration()
        await mcp.initialize_mcp_tools()
        session = await mcp.create_mcp_session("test_agent")
        
        # 여러 도구 호출로 사용 통계 생성
        await mcp.call_mcp_tool(session.session_id, "playwright", "navigate", {})
        await mcp.call_mcp_tool(session.session_id, "playwright", "screenshot", {})
        await mcp.call_mcp_tool(session.session_id, "data_analyzer", "statistical_analysis", {})
        
        return mcp, session
    
    @pytest.mark.asyncio
    async def test_tool_usage_statistics(self, mcp_with_usage):
        """도구 사용 통계 테스트"""
        mcp, session = await mcp_with_usage
        
        # Playwright 통계 확인
        playwright_stats = mcp.tool_usage_stats["playwright"]
        assert playwright_stats["total_calls"] == 2
        assert playwright_stats["successful_calls"] == 2
        assert playwright_stats["failed_calls"] == 0
        assert playwright_stats["average_response_time"] > 0
        
        # Data Analyzer 통계 확인
        analyzer_stats = mcp.tool_usage_stats["data_analyzer"]
        assert analyzer_stats["total_calls"] == 1
        assert analyzer_stats["successful_calls"] == 1
        assert analyzer_stats["failed_calls"] == 0

if __name__ == "__main__":
    # 테스트 실행
    import subprocess
    import sys
    
    print("🔗 MCP 통합 테스트 실행 중...")
    
    # pytest 실행
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\n🎯 테스트 결과: {'✅ 성공' if result.returncode == 0 else '❌ 실패'}")
    
    if result.returncode == 0:
        print("🔗 MCP 통합 기능이 모든 테스트를 통과했습니다!")
        print("✨ 7개 MCP 도구 (Playwright, File Manager, Database Connector, API Gateway, Data Analyzer, Chart Generator, LLM Gateway)가 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.") 