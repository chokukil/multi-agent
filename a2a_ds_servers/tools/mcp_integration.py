#!/usr/bin/env python3
"""
🔗 MCP (Model Context Protocol) 통합 모듈

A2A 기반 Context Engineering 플랫폼에서 MCP 도구들을 지원하는 통합 레이어
TOOLS Data Layer의 핵심 구성 요소로 다양한 MCP 도구들을 A2A 에이전트들이 활용할 수 있게 함

Key Features:
- MCP 서버 자동 발견 및 연결
- MCP 도구 호출 및 결과 처리  
- A2A 에이전트와 MCP 도구 간 브리지 역할
- 비동기 MCP 작업 처리 및 스트리밍 지원
- MCP 도구 상태 모니터링 및 오류 처리
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

class MCPToolType(Enum):
    """MCP 도구 타입 분류"""
    BROWSER = "browser"          # 웹 브라우징 도구
    FILE_SYSTEM = "file_system"  # 파일 시스템 조작
    DATABASE = "database"        # 데이터베이스 연결
    API_CLIENT = "api_client"    # API 호출 도구
    ANALYSIS = "analysis"        # 데이터 분석 도구
    VISUALIZATION = "visualization"  # 시각화 도구
    AI_MODEL = "ai_model"       # AI 모델 호출
    CUSTOM = "custom"           # 커스텀 도구

@dataclass
class MCPTool:
    """MCP 도구 정보"""
    tool_id: str
    name: str
    description: str
    tool_type: MCPToolType
    endpoint: str
    parameters: Dict[str, Any]
    capabilities: List[str]
    status: str  # 'available', 'busy', 'offline', 'error'
    last_used: Optional[datetime] = None
    usage_count: int = 0
    error_count: int = 0

@dataclass
class MCPRequest:
    """MCP 요청 정보"""
    request_id: str
    tool_id: str
    action: str
    parameters: Dict[str, Any]
    requester_agent: str
    created_at: datetime
    status: str  # 'pending', 'processing', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class MCPSession:
    """MCP 세션 정보"""
    session_id: str
    agent_id: str
    active_tools: List[str]
    created_at: datetime
    last_activity: datetime
    total_requests: int = 0
    successful_requests: int = 0

class MCPIntegration:
    """
    MCP (Model Context Protocol) 통합 클래스
    
    A2A 에이전트들이 다양한 MCP 도구들을 활용할 수 있도록 하는 중앙 통합 시스템
    Context Engineering TOOLS 레이어의 핵심 구성 요소
    """
    
    # 기본 MCP 도구 레지스트리 
    DEFAULT_MCP_TOOLS = {
        "playwright": {
            "name": "Playwright Browser Automation",
            "description": "웹 브라우저 자동화 및 웹페이지 상호작용",
            "tool_type": MCPToolType.BROWSER,
            "endpoint": "mcp://localhost:3000/playwright",
            "capabilities": [
                "webpage_navigation",
                "element_interaction", 
                "screenshot_capture",
                "form_filling",
                "data_extraction"
            ],
            "parameters": {
                "browser_types": ["chromium", "firefox", "webkit"],
                "headless_mode": True,
                "timeout": 30000
            }
        },
        "file_manager": {
            "name": "File System Manager",
            "description": "파일 시스템 조작 및 파일 관리",
            "tool_type": MCPToolType.FILE_SYSTEM,
            "endpoint": "mcp://localhost:3001/filesystem",
            "capabilities": [
                "file_read_write",
                "directory_operations",
                "file_search",
                "file_monitoring",
                "batch_operations"
            ],
            "parameters": {
                "allowed_extensions": [".txt", ".csv", ".json", ".xlsx", ".pdf"],
                "max_file_size": "100MB",
                "sandbox_mode": True
            }
        },
        "database_connector": {
            "name": "Database Connector",
            "description": "다양한 데이터베이스 연결 및 쿼리 실행",
            "tool_type": MCPToolType.DATABASE,
            "endpoint": "mcp://localhost:3002/database",
            "capabilities": [
                "sql_execution",
                "schema_introspection",
                "data_migration",
                "query_optimization",
                "connection_pooling"
            ],
            "parameters": {
                "supported_dbs": ["postgresql", "mysql", "sqlite", "mongodb"],
                "connection_timeout": 30,
                "query_timeout": 300
            }
        },
        "api_gateway": {
            "name": "API Gateway",
            "description": "외부 API 호출 및 응답 처리",
            "tool_type": MCPToolType.API_CLIENT,
            "endpoint": "mcp://localhost:3003/api",
            "capabilities": [
                "rest_api_calls",
                "graphql_queries",
                "authentication_handling",
                "rate_limiting",
                "response_parsing"
            ],
            "parameters": {
                "supported_auth": ["bearer", "api_key", "oauth2"],
                "max_retries": 3,
                "timeout": 60
            }
        },
        "data_analyzer": {
            "name": "Advanced Data Analyzer",
            "description": "고급 데이터 분석 및 통계 처리",
            "tool_type": MCPToolType.ANALYSIS,
            "endpoint": "mcp://localhost:3004/analysis",
            "capabilities": [
                "statistical_analysis",
                "time_series_analysis",
                "correlation_analysis",
                "outlier_detection",
                "predictive_modeling"
            ],
            "parameters": {
                "analysis_engines": ["pandas", "numpy", "scipy", "scikit-learn"],
                "max_dataset_size": "1GB",
                "parallel_processing": True
            }
        },
        "chart_generator": {
            "name": "Advanced Chart Generator", 
            "description": "고급 데이터 시각화 및 차트 생성",
            "tool_type": MCPToolType.VISUALIZATION,
            "endpoint": "mcp://localhost:3005/visualization",
            "capabilities": [
                "interactive_charts",
                "dashboard_creation",
                "3d_visualization",
                "geospatial_plotting",
                "animation_support"
            ],
            "parameters": {
                "chart_libraries": ["plotly", "d3js", "bokeh", "matplotlib"],
                "export_formats": ["png", "svg", "pdf", "html"],
                "max_data_points": 1000000
            }
        },
        "llm_gateway": {
            "name": "LLM Gateway",
            "description": "다양한 LLM 모델 호출 및 통합",
            "tool_type": MCPToolType.AI_MODEL,
            "endpoint": "mcp://localhost:3006/llm",
            "capabilities": [
                "multi_model_support",
                "prompt_optimization",
                "response_streaming",
                "model_switching",
                "cost_optimization"
            ],
            "parameters": {
                "supported_models": ["gpt-4o", "gemma3", "claude-3", "llama3"],
                "max_tokens": 4096,
                "streaming_enabled": True
            }
        }
    }
    
    def __init__(self):
        # MCP 도구 관리
        self.available_tools: Dict[str, MCPTool] = {}
        self.active_sessions: Dict[str, MCPSession] = {}
        self.pending_requests: Dict[str, MCPRequest] = {}
        self.completed_requests: Dict[str, MCPRequest] = {}
        
        # HTTP 클라이언트 (MCP over HTTP 지원)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # 모니터링 및 통계
        self.tool_usage_stats = {}
        self.performance_metrics = {}
        
        logger.info("🔗 MCP 통합 시스템 초기화 완료")
    
    async def initialize_mcp_tools(self) -> Dict[str, Any]:
        """
        MCP 도구들 초기화 및 발견
        
        사용 가능한 MCP 도구들을 검색하고 연결을 설정
        """
        logger.info("🔍 MCP 도구 초기화 및 발견 중...")
        
        discovery_results = {
            "total_tools": 0,
            "available_tools": 0,
            "tool_details": {},
            "discovery_status": "initializing"
        }
        
        for tool_id, tool_config in self.DEFAULT_MCP_TOOLS.items():
            try:
                # MCP 도구 연결성 확인 (HTTP 기반)
                # 실제 MCP 서버가 없으므로 mock discovery
                mcp_tool = MCPTool(
                    tool_id=tool_id,
                    name=tool_config["name"],
                    description=tool_config["description"],
                    tool_type=tool_config["tool_type"],
                    endpoint=tool_config["endpoint"],
                    parameters=tool_config["parameters"],
                    capabilities=tool_config["capabilities"],
                    status="available"  # Mock으로 사용 가능 상태로 설정
                )
                
                self.available_tools[tool_id] = mcp_tool
                discovery_results["tool_details"][tool_id] = {
                    "name": tool_config["name"],
                    "type": tool_config["tool_type"].value,
                    "status": "available",
                    "capabilities": tool_config["capabilities"]
                }
                discovery_results["available_tools"] += 1
                
                # 사용 통계 초기화
                self.tool_usage_stats[tool_id] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "average_response_time": 0.0
                }
                
                logger.info(f"✅ {tool_id}: MCP 도구 연결 확인")
                
            except Exception as e:
                logger.warning(f"❌ {tool_id}: MCP 도구 연결 실패 - {e}")
                discovery_results["tool_details"][tool_id] = {
                    "name": tool_config["name"],
                    "status": "connection_failed",
                    "error": str(e)
                }
            
            discovery_results["total_tools"] += 1
        
        # 발견 상태 결정
        if discovery_results["available_tools"] >= 5:
            discovery_results["discovery_status"] = "excellent"
        elif discovery_results["available_tools"] >= 3:
            discovery_results["discovery_status"] = "good"
        elif discovery_results["available_tools"] >= 1:
            discovery_results["discovery_status"] = "limited"
        else:
            discovery_results["discovery_status"] = "no_tools"
        
        logger.info(f"🎯 MCP 도구 초기화 완료: {discovery_results['available_tools']}/{discovery_results['total_tools']} 도구 사용 가능")
        
        return discovery_results
    
    async def create_mcp_session(self, agent_id: str, required_tools: List[str] = None) -> MCPSession:
        """
        MCP 세션 생성
        
        특정 에이전트를 위한 MCP 도구 사용 세션을 생성
        """
        session_id = f"mcp_session_{uuid.uuid4().hex[:12]}"
        
        # 필요한 도구들이 사용 가능한지 확인
        available_required_tools = []
        if required_tools:
            for tool_id in required_tools:
                if tool_id in self.available_tools and self.available_tools[tool_id].status == "available":
                    available_required_tools.append(tool_id)
                else:
                    logger.warning(f"⚠️ 요청된 MCP 도구 사용 불가: {tool_id}")
        else:
            # 모든 사용 가능한 도구 포함
            available_required_tools = [
                tool_id for tool_id, tool in self.available_tools.items() 
                if tool.status == "available"
            ]
        
        session = MCPSession(
            session_id=session_id,
            agent_id=agent_id,
            active_tools=available_required_tools,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"🚀 MCP 세션 생성: {session_id} (에이전트: {agent_id}, 도구: {len(available_required_tools)}개)")
        
        return session
    
    async def call_mcp_tool(self, session_id: str, tool_id: str, action: str, 
                           parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MCP 도구 호출
        
        특정 MCP 도구의 기능을 호출하고 결과를 반환
        """
        if session_id not in self.active_sessions:
            return {"error": "세션을 찾을 수 없습니다", "session_id": session_id}
        
        if tool_id not in self.available_tools:
            return {"error": f"MCP 도구를 찾을 수 없습니다: {tool_id}"}
        
        session = self.active_sessions[session_id]
        tool = self.available_tools[tool_id]
        
        # 도구가 세션에서 활성화되어 있는지 확인
        if tool_id not in session.active_tools:
            return {"error": f"도구가 현재 세션에서 활성화되지 않음: {tool_id}"}
        
        # 요청 생성
        request_id = f"mcp_req_{uuid.uuid4().hex[:12]}"
        mcp_request = MCPRequest(
            request_id=request_id,
            tool_id=tool_id,
            action=action,
            parameters=parameters or {},
            requester_agent=session.agent_id,
            created_at=datetime.now(),
            status="processing"
        )
        
        self.pending_requests[request_id] = mcp_request
        
        logger.info(f"📞 MCP 도구 호출: {tool_id}.{action} (요청: {request_id})")
        
        try:
            start_time = time.time()
            
            # MCP 도구별 처리 로직 (Mock 구현)
            result = await self._execute_mcp_action(tool, action, parameters or {})
            
            execution_time = time.time() - start_time
            
            # 성공 처리
            mcp_request.status = "completed"
            mcp_request.result = result
            
            # 통계 업데이트
            self._update_tool_statistics(tool_id, execution_time, success=True)
            
            # 세션 업데이트
            session.last_activity = datetime.now()
            session.total_requests += 1
            session.successful_requests += 1
            
            # 요청을 완료된 목록으로 이동
            self.completed_requests[request_id] = mcp_request
            del self.pending_requests[request_id]
            
            logger.info(f"✅ MCP 도구 호출 완료: {tool_id}.{action} ({execution_time:.2f}초)")
            
            return {
                "success": True,
                "request_id": request_id,
                "tool_id": tool_id,
                "action": action,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            # 실패 처리
            mcp_request.status = "failed"
            mcp_request.error = str(e)
            
            self._update_tool_statistics(tool_id, 0, success=False)
            
            session.last_activity = datetime.now()
            session.total_requests += 1
            
            self.completed_requests[request_id] = mcp_request
            del self.pending_requests[request_id]
            
            logger.error(f"❌ MCP 도구 호출 실패: {tool_id}.{action} - {e}")
            
            return {
                "success": False,
                "request_id": request_id,
                "tool_id": tool_id,
                "action": action,
                "error": str(e)
            }
    
    async def _execute_mcp_action(self, tool: MCPTool, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP 도구별 액션 실행 (Mock 구현)
        
        실제 환경에서는 각 MCP 도구의 실제 API를 호출
        """
        # Mock 지연 시간 (실제 처리 시뮬레이션)
        await asyncio.sleep(0.1)
        
        if tool.tool_type == MCPToolType.BROWSER:
            return await self._mock_browser_action(action, parameters)
        elif tool.tool_type == MCPToolType.FILE_SYSTEM:
            return await self._mock_filesystem_action(action, parameters)
        elif tool.tool_type == MCPToolType.DATABASE:
            return await self._mock_database_action(action, parameters)
        elif tool.tool_type == MCPToolType.API_CLIENT:
            return await self._mock_api_action(action, parameters)
        elif tool.tool_type == MCPToolType.ANALYSIS:
            return await self._mock_analysis_action(action, parameters)
        elif tool.tool_type == MCPToolType.VISUALIZATION:
            return await self._mock_visualization_action(action, parameters)
        elif tool.tool_type == MCPToolType.AI_MODEL:
            return await self._mock_ai_model_action(action, parameters)
        else:
            return {"result": f"Mock result for {action} with parameters: {parameters}"}
    
    async def _mock_browser_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """브라우저 도구 Mock 액션"""
        if action == "navigate":
            return {
                "status": "success",
                "url": parameters.get("url", "https://example.com"),
                "title": "Example Page",
                "content_length": 1234
            }
        elif action == "screenshot":
            return {
                "status": "success", 
                "screenshot_path": "/tmp/screenshot.png",
                "dimensions": {"width": 1920, "height": 1080}
            }
        elif action == "extract_data":
            return {
                "status": "success",
                "extracted_data": {"title": "Sample Data", "items": 42},
                "extraction_time": 0.5
            }
        else:
            return {"status": "error", "message": f"Unknown browser action: {action}"}
    
    async def _mock_filesystem_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """파일시스템 도구 Mock 액션"""
        if action == "read_file":
            return {
                "status": "success",
                "file_path": parameters.get("path", "/tmp/example.txt"),
                "content": "Mock file content",
                "size": 1024
            }
        elif action == "list_directory":
            return {
                "status": "success",
                "directory": parameters.get("path", "/tmp"),
                "files": ["file1.txt", "file2.csv", "data.json"],
                "total_files": 3
            }
        else:
            return {"status": "error", "message": f"Unknown filesystem action: {action}"}
    
    async def _mock_database_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스 도구 Mock 액션"""
        if action == "execute_query":
            return {
                "status": "success",
                "query": parameters.get("query", "SELECT * FROM users"),
                "rows_affected": 150,
                "execution_time": 0.2
            }
        elif action == "get_schema":
            return {
                "status": "success",
                "tables": ["users", "orders", "products"],
                "total_tables": 3
            }
        else:
            return {"status": "error", "message": f"Unknown database action: {action}"}
    
    async def _mock_api_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """API 도구 Mock 액션"""
        if action == "http_request":
            return {
                "status": "success",
                "method": parameters.get("method", "GET"),
                "url": parameters.get("url", "https://api.example.com"),
                "status_code": 200,
                "response_size": 2048
            }
        else:
            return {"status": "error", "message": f"Unknown API action: {action}"}
    
    async def _mock_analysis_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """분석 도구 Mock 액션"""
        if action == "statistical_analysis":
            return {
                "status": "success",
                "analysis_type": "descriptive_statistics",
                "results": {
                    "mean": 42.5,
                    "std": 15.2,
                    "correlation": 0.75
                },
                "processing_time": 1.2
            }
        else:
            return {"status": "error", "message": f"Unknown analysis action: {action}"}
    
    async def _mock_visualization_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 도구 Mock 액션"""
        if action == "create_chart":
            return {
                "status": "success",
                "chart_type": parameters.get("type", "line"),
                "chart_path": "/tmp/chart.png",
                "data_points": 100
            }
        else:
            return {"status": "error", "message": f"Unknown visualization action: {action}"}
    
    async def _mock_ai_model_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """AI 모델 도구 Mock 액션"""
        if action == "generate_text":
            return {
                "status": "success",
                "model": parameters.get("model", "gpt-4o"),
                "generated_text": "Mock AI generated response",
                "tokens_used": 150
            }
        else:
            return {"status": "error", "message": f"Unknown AI model action: {action}"}
    
    def _update_tool_statistics(self, tool_id: str, execution_time: float, success: bool):
        """도구 사용 통계 업데이트"""
        if tool_id not in self.tool_usage_stats:
            self.tool_usage_stats[tool_id] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_response_time": 0.0
            }
        
        stats = self.tool_usage_stats[tool_id]
        stats["total_calls"] += 1
        
        if success:
            stats["successful_calls"] += 1
            # 평균 응답 시간 업데이트
            current_avg = stats["average_response_time"]
            total_successful = stats["successful_calls"]
            stats["average_response_time"] = (current_avg * (total_successful - 1) + execution_time) / total_successful
        else:
            stats["failed_calls"] += 1
    
    async def get_tool_capabilities(self, tool_id: str = None) -> Dict[str, Any]:
        """MCP 도구 기능 조회"""
        if tool_id:
            if tool_id in self.available_tools:
                tool = self.available_tools[tool_id]
                return {
                    "tool_id": tool_id,
                    "name": tool.name,
                    "type": tool.tool_type.value,
                    "capabilities": tool.capabilities,
                    "parameters": tool.parameters,
                    "status": tool.status,
                    "usage_stats": self.tool_usage_stats.get(tool_id, {})
                }
            else:
                return {"error": f"도구를 찾을 수 없습니다: {tool_id}"}
        else:
            # 모든 도구 정보 반환
            return {
                "total_tools": len(self.available_tools),
                "available_tools": [tool_id for tool_id, tool in self.available_tools.items() if tool.status == "available"],
                "tool_details": {
                    tool_id: {
                        "name": tool.name,
                        "type": tool.tool_type.value,
                        "capabilities": tool.capabilities,
                        "status": tool.status
                    }
                    for tool_id, tool in self.available_tools.items()
                },
                "usage_statistics": self.tool_usage_stats
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """MCP 세션 상태 조회"""
        if session_id not in self.active_sessions:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "agent_id": session.agent_id,
            "active_tools": session.active_tools,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_requests": session.total_requests,
            "successful_requests": session.successful_requests,
            "success_rate": session.successful_requests / session.total_requests if session.total_requests > 0 else 0,
            "pending_requests": len([req for req in self.pending_requests.values() if req.requester_agent == session.agent_id])
        }
    
    async def close_session(self, session_id: str):
        """MCP 세션 종료"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🔚 MCP 세션 종료: {session_id}")
    
    async def close(self):
        """리소스 정리"""
        await self.http_client.aclose()
        logger.info("🔚 MCP 통합 시스템 종료")

# 전역 MCP 통합 인스턴스
_mcp_integration = None

def get_mcp_integration() -> MCPIntegration:
    """MCP 통합 인스턴스 반환 (싱글톤 패턴)"""
    global _mcp_integration
    if _mcp_integration is None:
        _mcp_integration = MCPIntegration()
    return _mcp_integration

async def initialize_mcp_tools():
    """MCP 도구 초기화 (편의 함수)"""
    mcp = get_mcp_integration()
    return await mcp.initialize_mcp_tools()

async def call_mcp_tool(session_id: str, tool_id: str, action: str, parameters: Dict[str, Any] = None):
    """MCP 도구 호출 (편의 함수)"""
    mcp = get_mcp_integration()
    return await mcp.call_mcp_tool(session_id, tool_id, action, parameters) 