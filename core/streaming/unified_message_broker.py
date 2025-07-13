#!/usr/bin/env python3
"""
🔄 Unified Message Broker

11개 A2A 에이전트 + 7개 MCP 도구 통합 메시지 브로커
실시간 스트리밍, 세션 관리, 로드 밸런싱 지원

Architecture:
- A2A Agent Integration (SSE 기반)
- MCP Tool Integration (SSE + STDIO 지원)
- Message Routing & Orchestration
- Real-time Streaming Pipeline
- Session & State Management
- Error Handling & Recovery
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, AsyncGenerator, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import httpx

# 기존 스트리밍 컴포넌트들
from .a2a_sse_client import A2ASSEClient
from .mcp_stdio_bridge import get_mcp_stdio_bridge, MCPSTDIOBridge

# 연결 풀 시스템 추가
from ..performance.connection_pool import (
    get_connection_pool_manager, 
    ConnectionPoolManager,
    get_a2a_connection,
    get_mcp_sse_connection,
    release_connection_with_pool
)

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """에이전트/도구 타입"""
    A2A_AGENT = "a2a_agent"
    MCP_SSE = "mcp_sse"
    MCP_STDIO = "mcp_stdio"

class MessagePriority(Enum):
    """메시지 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentEndpoint:
    """에이전트/도구 엔드포인트 정보"""
    agent_id: str
    name: str
    agent_type: AgentType
    endpoint: str
    capabilities: List[str]
    status: str = "unknown"  # online, offline, error
    load_factor: float = 0.0  # 0.0 (idle) ~ 1.0 (overloaded)
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedMessage:
    """통합 메시지 형식"""
    message_id: str
    session_id: str
    source_agent: str
    target_agent: str
    message_type: str  # request, response, stream, error
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None
    correlation_id: Optional[str] = None
    final: bool = False

@dataclass
class BrokerSession:
    """브로커 세션 정보"""
    session_id: str
    user_query: str
    active_agents: Set[str]
    message_history: List[UnifiedMessage]
    created_at: datetime
    last_activity: datetime
    status: str = "active"  # active, completed, failed, timeout
    context: Dict[str, Any] = field(default_factory=dict)

class UnifiedMessageBroker:
    """통합 메시지 브로커"""
    
    def __init__(self):
        # 에이전트/도구 레지스트리
        self.agents: Dict[str, AgentEndpoint] = {}
        
        # 통신 클라이언트들
        self.a2a_client: Optional[A2ASSEClient] = None
        self.mcp_stdio_bridge: MCPSTDIOBridge = get_mcp_stdio_bridge()
        
        # 연결 풀 매니저 (성능 최적화)
        self.connection_pool_manager: ConnectionPoolManager = get_connection_pool_manager()
        
        # 세션 관리
        self.active_sessions: Dict[str, BrokerSession] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # 통계 및 모니터링
        self.stats = {
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0,
            'active_sessions': 0,
            'connection_pool_hits': 0,  # 연결 풀 활용도
            'connection_pool_misses': 0,
            'uptime_start': datetime.now()
        }
        
        # 초기화
        self._register_default_agents()
    
    def _register_default_agents(self):
        """기본 A2A 에이전트들과 MCP 도구들 등록"""
        
        # 11개 A2A 에이전트 등록
        a2a_agents = [
            AgentEndpoint(
                agent_id="orchestrator",
                name="A2A Orchestrator",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8100",
                capabilities=["orchestration", "planning", "coordination", "context_engineering"]
            ),
            AgentEndpoint(
                agent_id="data_cleaning",
                name="Data Cleaning Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8306",
                capabilities=["data_cleaning", "missing_values", "duplicates", "data_quality"]
            ),
            AgentEndpoint(
                agent_id="data_loader",
                name="Data Loader Agent", 
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8307",
                capabilities=["data_loading", "file_formats", "data_import", "data_validation"]
            ),
            AgentEndpoint(
                agent_id="data_visualization",
                name="Data Visualization Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8308",
                capabilities=["plotting", "charts", "graphs", "visual_analytics"]
            ),
            AgentEndpoint(
                agent_id="data_wrangling",
                name="Data Wrangling Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8309",
                capabilities=["data_transformation", "feature_engineering", "data_reshaping"]
            ),
            AgentEndpoint(
                agent_id="eda",
                name="EDA Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8310",
                capabilities=["exploratory_analysis", "statistics", "data_profiling"]
            ),
            AgentEndpoint(
                agent_id="feature_engineering",
                name="Feature Engineering Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8311",
                capabilities=["feature_creation", "feature_selection", "dimensionality_reduction"]
            ),
            AgentEndpoint(
                agent_id="h2o_modeling",
                name="H2O Modeling Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8312",
                capabilities=["automl", "model_training", "hyperparameter_tuning"]
            ),
            AgentEndpoint(
                agent_id="mlflow",
                name="MLflow Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8313",
                capabilities=["experiment_tracking", "model_versioning", "model_deployment"]
            ),
            AgentEndpoint(
                agent_id="sql_database",
                name="SQL Database Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8314",
                capabilities=["sql_queries", "database_operations", "data_extraction"]
            ),
            AgentEndpoint(
                agent_id="pandas",
                name="Pandas Agent",
                agent_type=AgentType.A2A_AGENT,
                endpoint="http://localhost:8315",
                capabilities=["dataframe_operations", "data_analysis", "statistical_computing"]
            )
        ]
        
        # 7개 MCP 도구 등록 (SSE 기반)
        mcp_sse_tools = [
            AgentEndpoint(
                agent_id="playwright_browser",
                name="Playwright Browser Automation",
                agent_type=AgentType.MCP_SSE,
                endpoint="http://localhost:3000",
                capabilities=["web_automation", "browser_control", "web_scraping", "ui_testing"]
            ),
            AgentEndpoint(
                agent_id="file_manager",
                name="File System Manager",
                agent_type=AgentType.MCP_SSE,
                endpoint="http://localhost:3001",
                capabilities=["file_operations", "directory_management", "file_search"]
            ),
            AgentEndpoint(
                agent_id="database_connector",
                name="Database Connector",
                agent_type=AgentType.MCP_SSE,
                endpoint="http://localhost:3002",
                capabilities=["multi_database", "connection_pooling", "query_optimization"]
            ),
            AgentEndpoint(
                agent_id="api_gateway",
                name="API Gateway",
                agent_type=AgentType.MCP_SSE,
                endpoint="http://localhost:3003",
                capabilities=["api_calls", "rest_apis", "graphql", "webhook_handling"]
            )
        ]
        
        # MCP STDIO 도구들 (방금 구현한 브리지를 통해)
        mcp_stdio_tools = [
            AgentEndpoint(
                agent_id="advanced_analyzer",
                name="Advanced Data Analyzer", 
                agent_type=AgentType.MCP_STDIO,
                endpoint="stdio://pandas_stdio",
                capabilities=["advanced_statistics", "machine_learning", "pattern_analysis"]
            ),
            AgentEndpoint(
                agent_id="chart_generator",
                name="Chart Generator",
                agent_type=AgentType.MCP_STDIO,
                endpoint="stdio://data_analyzer_stdio",
                capabilities=["advanced_plotting", "interactive_charts", "custom_visualizations"]
            ),
            AgentEndpoint(
                agent_id="llm_gateway",
                name="LLM Gateway",
                agent_type=AgentType.MCP_STDIO,
                endpoint="stdio://web_scraper_stdio",
                capabilities=["multi_llm", "model_selection", "prompt_optimization"]
            )
        ]
        
        # 모든 에이전트/도구 등록
        all_agents = a2a_agents + mcp_sse_tools + mcp_stdio_tools
        for agent in all_agents:
            self.agents[agent.agent_id] = agent
        
        logger.info(f"📋 통합 브로커 초기화: {len(all_agents)}개 에이전트/도구 등록")
    
    async def initialize(self):
        """브로커 초기화 - 모든 에이전트/도구와 연결 확인"""
        logger.info("🚀 Unified Message Broker 초기화 시작...")
        
        # 연결 풀 매니저 초기화 (성능 최적화)
        await self.connection_pool_manager.initialize()
        
        # A2A 클라이언트 초기화
        a2a_endpoints = {
            agent.agent_id: agent.endpoint 
            for agent in self.agents.values() 
            if agent.agent_type == AgentType.A2A_AGENT
        }
        
        self.a2a_client = A2ASSEClient(
            base_url="http://localhost:8100",
            agents=a2a_endpoints
        )
        
        # 모든 에이전트 상태 확인
        await self._health_check_all_agents()
        
        logger.info("✅ Unified Message Broker 초기화 완료 (연결 풀 활성화)")
    
    async def _health_check_all_agents(self):
        """모든 에이전트/도구 상태 확인"""
        health_tasks = []
        
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(self._check_agent_health(agent_id))
            health_tasks.append(task)
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        online_count = sum(1 for r in results if r is True)
        logger.info(f"💊 상태 확인 완료: {online_count}/{len(self.agents)} 에이전트 온라인")
    
    async def _check_agent_health(self, agent_id: str) -> bool:
        """개별 에이전트 상태 확인"""
        agent = self.agents[agent_id]
        
        try:
            if agent.agent_type == AgentType.A2A_AGENT:
                # A2A 에이전트 상태 확인
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{agent.endpoint}/.well-known/agent.json",
                        timeout=5.0
                    )
                    agent.status = "online" if response.status_code == 200 else "error"
            
            elif agent.agent_type == AgentType.MCP_SSE:
                # MCP SSE 도구 상태 확인
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{agent.endpoint}/health",
                        timeout=5.0
                    )
                    agent.status = "online" if response.status_code == 200 else "error"
            
            elif agent.agent_type == AgentType.MCP_STDIO:
                # MCP STDIO 도구 상태 확인 (브리지를 통해)
                bridge_health = await self.mcp_stdio_bridge.health_check()
                service_id = agent.endpoint.replace("stdio://", "")
                service_status = bridge_health.get('service_status', {}).get(service_id, 'unavailable')
                agent.status = "online" if service_status == 'available' else "offline"
            
            agent.last_health_check = datetime.now()
            return agent.status == "online"
            
        except Exception as e:
            agent.status = "error"
            agent.last_health_check = datetime.now()
            logger.warning(f"⚠️ {agent_id} 상태 확인 실패: {e}")
            return False
    
    async def create_session(self, user_query: str, session_id: Optional[str] = None) -> str:
        """새로운 브로커 세션 생성"""
        if session_id is None:
            session_id = f"unified_{uuid.uuid4().hex[:8]}"
        
        session = BrokerSession(
            session_id=session_id,
            user_query=user_query,
            active_agents=set(),
            message_history=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        self.message_queues[session_id] = asyncio.Queue()
        
        self.stats['active_sessions'] = len(self.active_sessions)
        
        logger.info(f"🆕 브로커 세션 생성: {session_id}")
        return session_id
    
    async def route_message(self, message: UnifiedMessage) -> AsyncGenerator[Dict[str, Any], None]:
        """메시지 라우팅 및 실시간 스트리밍"""
        
        if message.session_id not in self.active_sessions:
            yield {
                'event': 'error',
                'data': {'error': f'Session not found: {message.session_id}', 'final': True}
            }
            return
        
        session = self.active_sessions[message.session_id]
        session.message_history.append(message)
        session.last_activity = datetime.now()
        session.active_agents.add(message.target_agent)
        
        self.stats['total_messages'] += 1
        
        target_agent = self.agents.get(message.target_agent)
        if not target_agent:
            yield {
                'event': 'error',
                'data': {'error': f'Unknown agent: {message.target_agent}', 'final': True}
            }
            return
        
        try:
            # 메시지 라우팅 로그
            yield {
                'event': 'routing',
                'data': {
                    'message_id': message.message_id,
                    'from': message.source_agent,
                    'to': message.target_agent,
                    'type': target_agent.agent_type.value,
                    'final': False
                }
            }
            
            # 에이전트 타입별 라우팅
            if target_agent.agent_type == AgentType.A2A_AGENT:
                async for event in self._route_to_a2a_agent(message, target_agent):
                    yield event
            
            elif target_agent.agent_type == AgentType.MCP_SSE:
                async for event in self._route_to_mcp_sse(message, target_agent):
                    yield event
            
            elif target_agent.agent_type == AgentType.MCP_STDIO:
                async for event in self._route_to_mcp_stdio(message, target_agent):
                    yield event
            
            self.stats['successful_messages'] += 1
            
        except Exception as e:
            logger.error(f"❌ 메시지 라우팅 오류: {e}")
            self.stats['failed_messages'] += 1
            
            yield {
                'event': 'error',
                'data': {
                    'error': str(e),
                    'message_id': message.message_id,
                    'target_agent': message.target_agent,
                    'final': True
                }
            }
    
    async def _route_to_a2a_agent(
        self, 
        message: UnifiedMessage, 
        agent: AgentEndpoint
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """A2A 에이전트로 메시지 라우팅 (연결 풀 최적화)"""
        
        connection = None
        start_time = time.time()
        
        try:
            # 연결 풀에서 A2A 연결 가져오기
            connection = await get_a2a_connection(agent.endpoint)
            
            if not connection:
                self.stats['connection_pool_misses'] += 1
                # 연결 풀 실패 시 기존 방식 사용
                if not self.a2a_client:
                    yield {
                        'event': 'error',
                        'data': {'error': 'A2A client not initialized', 'final': True}
                    }
                    return
                
                # 기존 A2A 클라이언트 사용
                async for sse_event in self.a2a_client.stream_request(
                    agent.agent_id, 
                    {
                        'messageId': message.message_id,
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': json.dumps(message.content)}]
                    }
                ):
                    unified_event = {
                        'event': 'a2a_response',
                        'data': {
                            'agent_id': agent.agent_id,
                            'message_id': message.message_id,
                            'content': sse_event,
                            'final': sse_event.get('final', False)
                        }
                    }
                    yield unified_event
                return
            
            # 연결 풀 히트 - 성능 메트릭스 수집
            self.stats['connection_pool_hits'] += 1
            
            # A2A 메시지 형식으로 변환
            a2a_request = {
                'messageId': message.message_id,
                'role': 'user',
                'parts': [{'kind': 'text', 'text': json.dumps(message.content)}]
            }
            
            # 풀링된 연결 사용하여 요청
            if connection.client:
                response = await connection.client.post(
                    f"{agent.endpoint}/stream",
                    json=a2a_request,
                    headers={'Accept': 'text/event-stream'}
                )
                
                # SSE 응답 처리
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            sse_data = json.loads(line[6:])
                            
                            # A2A 응답을 통합 형식으로 변환
                            unified_event = {
                                'event': 'a2a_response',
                                'data': {
                                    'agent_id': agent.agent_id,
                                    'message_id': message.message_id,
                                    'content': sse_data,
                                    'final': sse_data.get('final', False),
                                    'pooled_connection': connection.connection_id
                                }
                            }
                            yield unified_event
                            
                            if sse_data.get('final'):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # 응답 시간 메트릭스 수집
            response_time = time.time() - start_time
            connection.metrics.response_times.append(response_time)
            connection.metrics.total_requests += 1
            
        except Exception as e:
            logger.error(f"❌ A2A 연결 풀 라우팅 오류: {e}")
            yield {
                'event': 'a2a_error',
                'data': {
                    'error': str(e),
                    'agent_id': agent.agent_id,
                    'message_id': message.message_id,
                    'final': True
                }
            }
        
        finally:
            # 연결 반환 (풀로)
            if connection:
                success = True  # 에러 여부에 따라 조정 가능
                await release_connection_with_pool('a2a_agents', connection, success)
    
    async def _route_to_mcp_sse(
        self, 
        message: UnifiedMessage, 
        agent: AgentEndpoint
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """MCP SSE 도구로 메시지 라우팅 (연결 풀 최적화)"""
        
        connection = None
        start_time = time.time()
        
        try:
            # 연결 풀에서 MCP SSE 연결 가져오기
            connection = await get_mcp_sse_connection(agent.endpoint)
            
            if not connection:
                self.stats['connection_pool_misses'] += 1
                # 연결 풀 실패 시 기존 방식 사용
                async with httpx.AsyncClient() as client:
                    sse_url = f"{agent.endpoint}/stream"
                    
                    async with client.stream(
                        'POST',
                        sse_url,
                        json=message.content,
                        headers={'Accept': 'text/event-stream'},
                        timeout=30.0
                    ) as response:
                        
                        async for line in response.aiter_lines():
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    
                                    unified_event = {
                                        'event': 'mcp_sse_response',
                                        'data': {
                                            'agent_id': agent.agent_id,
                                            'message_id': message.message_id,
                                            'content': data,
                                            'final': data.get('final', False)
                                        }
                                    }
                                    yield unified_event
                                    
                                    if data.get('final'):
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                return
            
            # 연결 풀 히트 - 성능 메트릭스 수집
            self.stats['connection_pool_hits'] += 1
            
            # 풀링된 연결 사용하여 MCP SSE API 호출
            if connection.client:
                sse_url = f"{agent.endpoint}/stream"
                
                async with connection.client.post(
                    sse_url,
                    json=message.content,
                    headers={'Accept': 'text/event-stream'}
                ) as response:
                    
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])
                                
                                unified_event = {
                                    'event': 'mcp_sse_response',
                                    'data': {
                                        'agent_id': agent.agent_id,
                                        'message_id': message.message_id,
                                        'content': data,
                                        'final': data.get('final', False),
                                        'pooled_connection': connection.connection_id
                                    }
                                }
                                yield unified_event
                                
                                if data.get('final'):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
            
            # 응답 시간 메트릭스 수집
            response_time = time.time() - start_time
            connection.metrics.response_times.append(response_time)
            connection.metrics.total_requests += 1
                                
        except Exception as e:
            logger.error(f"❌ MCP SSE 연결 풀 라우팅 오류: {e}")
            yield {
                'event': 'mcp_sse_error',
                'data': {
                    'error': str(e),
                    'agent_id': agent.agent_id,
                    'message_id': message.message_id,
                    'final': True
                }
            }
        
        finally:
            # 연결 반환 (풀로)
            if connection:
                success = True  # 에러 여부에 따라 조정 가능
                await release_connection_with_pool('mcp_sse_tools', connection, success)
    
    async def _route_to_mcp_stdio(
        self, 
        message: UnifiedMessage, 
        agent: AgentEndpoint
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """MCP STDIO 도구로 메시지 라우팅 (브리지 사용)"""
        
        try:
            service_id = agent.endpoint.replace("stdio://", "")
            
            # MCP STDIO 브리지를 통한 스트리밍
            async for stdio_event in self.mcp_stdio_bridge.stream_mcp_request(
                session_id=f"{message.session_id}_{agent.agent_id}",
                method=message.content.get('method', 'execute'),
                params=message.content.get('params', {})
            ):
                # STDIO 응답을 통합 형식으로 변환
                unified_event = {
                    'event': 'mcp_stdio_response',
                    'data': {
                        'agent_id': agent.agent_id,
                        'message_id': message.message_id,
                        'content': stdio_event,
                        'final': stdio_event['data'].get('final', False)
                    }
                }
                yield unified_event
                
                if stdio_event['data'].get('final'):
                    break
                    
        except Exception as e:
            yield {
                'event': 'mcp_stdio_error',
                'data': {
                    'error': str(e),
                    'agent_id': agent.agent_id, 
                    'message_id': message.message_id,
                    'final': True
                }
            }
    
    async def orchestrate_multi_agent_query(
        self, 
        session_id: str, 
        user_query: str,
        required_capabilities: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """멀티 에이전트 쿼리 오케스트레이션"""
        
        # 1. 필요한 에이전트들 식별
        selected_agents = self._select_agents_for_capabilities(required_capabilities or [])
        
        yield {
            'event': 'orchestration_start',
            'data': {
                'session_id': session_id,
                'selected_agents': [agent.agent_id for agent in selected_agents],
                'capabilities': required_capabilities,
                'final': False
            }
        }
        
        # 2. 오케스트레이터에게 먼저 전달
        orchestrator_message = UnifiedMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            source_agent="user",
            target_agent="orchestrator",
            message_type="request",
            content={
                'query': user_query,
                'available_agents': [agent.agent_id for agent in selected_agents],
                'capabilities': required_capabilities
            }
        )
        
        # 3. 오케스트레이터 응답 처리
        async for event in self.route_message(orchestrator_message):
            yield event
            
            # 오케스트레이터가 다른 에이전트 호출을 요청하는 경우
            if event.get('event') == 'a2a_response':
                content = event['data'].get('content', {})
                if 'delegate_to' in content:
                    # 위임된 에이전트에게 메시지 전달
                    delegated_message = UnifiedMessage(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        source_agent="orchestrator",
                        target_agent=content['delegate_to'],
                        message_type="request",
                        content=content.get('delegated_content', {})
                    )
                    
                    async for delegated_event in self.route_message(delegated_message):
                        yield delegated_event
    
    def _select_agents_for_capabilities(self, capabilities: List[str]) -> List[AgentEndpoint]:
        """필요한 기능에 맞는 에이전트들 선택"""
        if not capabilities:
            # 기본 에이전트들 반환
            return [agent for agent in self.agents.values() if agent.status == "online"]
        
        selected = []
        for agent in self.agents.values():
            if agent.status == "online":
                # 에이전트의 기능과 요구사항 매칭
                if any(cap in agent.capabilities for cap in capabilities):
                    selected.append(agent)
        
        return selected
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'status': session.status,
            'user_query': session.user_query,
            'active_agents': list(session.active_agents),
            'message_count': len(session.message_history),
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'duration_seconds': (datetime.now() - session.created_at).total_seconds()
        }
    
    async def get_broker_stats(self) -> Dict[str, Any]:
        """브로커 통계 정보"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        agent_stats = {}
        for agent_type in AgentType:
            agents = [a for a in self.agents.values() if a.agent_type == agent_type]
            online_count = sum(1 for a in agents if a.status == "online")
            agent_stats[agent_type.value] = {
                'total': len(agents),
                'online': online_count,
                'offline': len(agents) - online_count
            }
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_agents': len(self.agents),
            'agent_breakdown': agent_stats,
            'active_sessions': len(self.active_sessions),
            'message_stats': {
                'total': self.stats['total_messages'],
                'successful': self.stats['successful_messages'],
                'failed': self.stats['failed_messages'],
                'success_rate': (
                    self.stats['successful_messages'] / max(1, self.stats['total_messages'])
                ) * 100
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup_expired_sessions(self, max_idle_hours: int = 2):
        """만료된 세션 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_idle_hours)
        
        to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            if session_id in self.message_queues:
                del self.message_queues[session_id]
            del self.active_sessions[session_id]
            logger.info(f"🧹 만료된 세션 정리: {session_id}")
        
        self.stats['active_sessions'] = len(self.active_sessions)
    
    async def shutdown(self):
        """브로커 종료"""
        logger.info("🔚 Unified Message Broker 종료 시작...")
        
        # 모든 세션 정리
        self.active_sessions.clear()
        self.message_queues.clear()
        
        # MCP STDIO 브리지 종료
        await self.mcp_stdio_bridge.shutdown()
        
        # 연결 풀 매니저 종료 (성능 최적화)
        await self.connection_pool_manager.shutdown()
        
        logger.info("✅ Unified Message Broker 종료 완료 (연결 풀 정리 완료)")


# 전역 브로커 인스턴스
_unified_broker = None

def get_unified_message_broker() -> UnifiedMessageBroker:
    """전역 통합 메시지 브로커 인스턴스"""
    global _unified_broker
    if _unified_broker is None:
        _unified_broker = UnifiedMessageBroker()
    return _unified_broker


if __name__ == "__main__":
    # 테스트 코드
    async def demo():
        broker = UnifiedMessageBroker()
        await broker.initialize()
        
        print("🔄 Unified Message Broker Demo")
        
        # 브로커 통계
        stats = await broker.get_broker_stats()
        print(f"📊 브로커 통계: {stats}")
        
        # 테스트 세션 생성
        session_id = await broker.create_session("테스트 데이터 분석을 해주세요")
        print(f"🆕 테스트 세션: {session_id}")
        
        # 멀티 에이전트 쿼리 테스트
        async for event in broker.orchestrate_multi_agent_query(
            session_id, 
            "판다스를 사용해서 데이터를 분석하고 시각화해주세요",
            ["data_processing", "plotting"]
        ):
            print(f"📨 이벤트: {event}")
            if event['data'].get('final'):
                break
        
        await broker.shutdown()
    
    asyncio.run(demo()) 