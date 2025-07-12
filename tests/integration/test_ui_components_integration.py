"""
UI Components Integration Tests
pytest 기반 통합 테스트 - 여러 컴포넌트 간의 상호작용 검증
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List
import uuid
import streamlit as st
from contextlib import asynccontextmanager

# 시스템 경로 설정
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ui'))

# 테스트 대상 임포트
from a2a_sse_streaming_system import (
    A2ASSEStreamingSystem,
    A2ASSEStreamingExecutor,
    get_a2a_sse_streaming_system
)

from cursor_sse_realtime import (
    CursorSSERealtimeManager,
    CursorSSEManager,
    SSEEventType,
    SSEMessage,
    get_cursor_sse_realtime
)

from cursor_collaboration_network import (
    CursorCollaborationNetwork,
    NodeType,
    NodeStatus,
    ConnectionType,
    get_cursor_collaboration_network
)

# Mock Streamlit 환경
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Streamlit 환경 모킹"""
    with patch('streamlit.session_state', {}):
        yield


class TestA2ASSEIntegration:
    """A2A SSE 시스템 통합 테스트"""
    
    @pytest.fixture
    def sse_system(self):
        """SSE 스트리밍 시스템"""
        return get_a2a_sse_streaming_system()
    
    @pytest.fixture
    def realtime_manager(self):
        """실시간 관리자"""
        return get_cursor_sse_realtime()
    
    @pytest.mark.asyncio
    async def test_sse_system_realtime_integration(self, sse_system, realtime_manager):
        """SSE 시스템과 실시간 관리자 통합 테스트"""
        # 시스템 초기화 검증
        assert sse_system is not None
        assert realtime_manager is not None
        
        # 실시간 관리자가 SSE 관리자를 사용하는지 확인
        assert realtime_manager.sse_manager is not None
        assert realtime_manager.sync_system is not None
        
        # 시스템 상태 확인
        status = realtime_manager.get_system_status()
        assert "running" in status
        assert "subscriptions" in status
        assert "components" in status
    
    @pytest.mark.asyncio
    async def test_event_flow_integration(self, sse_system, realtime_manager):
        """이벤트 플로우 통합 테스트"""
        # 이벤트 전송 테스트
        await realtime_manager.send_agent_update(
            "test-agent",
            "working",
            {"message": "Processing request", "progress": 0.5}
        )
        
        # 이벤트 큐 확인
        assert realtime_manager.sse_manager.a2a_integration.event_queue.qsize() > 0
        
        # 사고 과정 업데이트
        await realtime_manager.send_thought_update(
            "test-thought",
            "analyzing",
            "Analyzing data structure..."
        )
        
        # MCP 도구 업데이트
        await realtime_manager.send_mcp_update(
            "test-mcp",
            "active",
            {"cpu": 0.8, "memory": 0.6, "requests": 10}
        )
        
        # 코드 스트리밍 업데이트
        await realtime_manager.send_code_update(
            "test-code",
            "streaming",
            "import pandas as pd\ndf = pd.read_csv('data.csv')"
        )
        
        # 이벤트 큐에 여러 이벤트가 쌓였는지 확인
        event_queue_size = realtime_manager.sse_manager.a2a_integration.event_queue.qsize()
        assert event_queue_size >= 4  # 4개 이상의 이벤트
    
    @pytest.mark.asyncio
    async def test_a2a_streaming_request_handling(self, sse_system):
        """A2A 스트리밍 요청 처리 통합 테스트"""
        # A2A 표준 요청 생성
        request = {
            "jsonrpc": "2.0",
            "id": "test-integration-123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Please analyze the uploaded dataset and provide insights"
                        }
                    ]
                },
                "sessionId": "integration-session-123"
            }
        }
        
        # 스트리밍 응답 생성
        response = await sse_system.handle_streaming_request(request)
        
        # 응답 검증
        assert hasattr(response, 'body_iterator')
        assert response.media_type == "text/event-stream"
        
        # 응답 헤더 검증
        headers = dict(response.headers)
        assert "cache-control" in headers
        assert "connection" in headers
        assert "access-control-allow-origin" in headers


class TestCollaborationNetworkIntegration:
    """협업 네트워크 통합 테스트"""
    
    @pytest.fixture
    def collaboration_network(self):
        """협업 네트워크"""
        return get_cursor_collaboration_network()
    
    @pytest.fixture
    def realtime_manager(self):
        """실시간 관리자"""
        return get_cursor_sse_realtime()
    
    def test_network_realtime_integration(self, collaboration_network, realtime_manager):
        """네트워크와 실시간 관리자 통합 테스트"""
        # 네트워크가 실시간 관리자를 사용하는지 확인
        assert collaboration_network.realtime_manager is not None
        
        # 기본 네트워크 구조 확인
        assert len(collaboration_network.nodes) > 0
        assert len(collaboration_network.connections) > 0
        
        # 노드 타입 확인
        node_types = [node.type for node in collaboration_network.nodes.values()]
        assert NodeType.ORCHESTRATOR in node_types
        assert NodeType.AGENT in node_types
        assert NodeType.MCP_TOOL in node_types
    
    def test_network_message_flow(self, collaboration_network):
        """네트워크 메시지 흐름 테스트"""
        # 메시지 전송 테스트
        message_id = collaboration_network.send_message(
            "orchestrator",
            "pandas_agent",
            "analysis_request",
            {
                "task": "data_analysis",
                "data": {"columns": ["A", "B", "C"]},
                "priority": "high"
            }
        )
        
        assert message_id is not None
        assert len(collaboration_network.message_flows) > 0
        
        # 메시지 흐름 확인
        message_flow = collaboration_network.message_flows[-1]
        assert message_flow.source == "orchestrator"
        assert message_flow.target == "pandas_agent"
        assert message_flow.message_type == "analysis_request"
        assert message_flow.payload["task"] == "data_analysis"
    
    def test_network_node_status_update(self, collaboration_network):
        """네트워크 노드 상태 업데이트 테스트"""
        # 노드 상태 업데이트
        collaboration_network.update_node_status("pandas_agent", NodeStatus.WORKING)
        
        # 상태 확인
        pandas_node = collaboration_network.nodes["pandas_agent"]
        assert pandas_node.status == NodeStatus.WORKING
        
        # 연결 활성화
        connection_id = "orchestrator_pandas_agent"
        collaboration_network.activate_connection(
            connection_id,
            {"message": "Analysis started", "timestamp": time.time()}
        )
        
        # 연결 상태 확인
        connection = collaboration_network.connections[connection_id]
        assert connection.active == True
        assert connection.message_data is not None
    
    def test_network_statistics(self, collaboration_network):
        """네트워크 통계 테스트"""
        # 메시지 시뮬레이션
        for i in range(5):
            collaboration_network.simulate_message_flow()
        
        # 통계 확인
        stats = collaboration_network.get_network_stats()
        
        assert stats["total_nodes"] > 0
        assert stats["total_connections"] > 0
        assert stats["message_flows"] >= 5
        assert "node_types" in stats
        assert "connection_types" in stats
        
        # 노드 타입별 통계 확인
        assert stats["node_types"]["orchestrator"] >= 1
        assert stats["node_types"]["agent"] >= 1
        assert stats["node_types"]["mcp_tool"] >= 1


class TestEndToEndWorkflow:
    """End-to-End 워크플로우 테스트"""
    
    @pytest.fixture
    def full_system(self):
        """전체 시스템"""
        return {
            "sse_system": get_a2a_sse_streaming_system(),
            "realtime_manager": get_cursor_sse_realtime(),
            "collaboration_network": get_cursor_collaboration_network()
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, full_system):
        """완전한 워크플로우 테스트"""
        sse_system = full_system["sse_system"]
        realtime_manager = full_system["realtime_manager"]
        network = full_system["collaboration_network"]
        
        # 1. 사용자 요청 생성
        user_request = {
            "jsonrpc": "2.0",
            "id": "workflow-test-123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Analyze sales data and create visualization"
                        }
                    ]
                },
                "sessionId": "workflow-session-123"
            }
        }
        
        # 2. A2A 스트리밍 시작
        response = await sse_system.handle_streaming_request(user_request)
        assert response is not None
        
        # 3. 실시간 상태 업데이트 시뮬레이션
        workflow_steps = [
            ("orchestrator", "analyzing", "Analyzing user request..."),
            ("data_loader", "loading", "Loading sales data..."),
            ("pandas_agent", "processing", "Processing data..."),
            ("viz_agent", "creating", "Creating visualization...")
        ]
        
        for agent, status, message in workflow_steps:
            # 에이전트 상태 업데이트
            await realtime_manager.send_agent_update(
                agent,
                status,
                {"message": message, "timestamp": time.time()}
            )
            
            # 네트워크 노드 상태 업데이트
            if agent in network.nodes:
                if status == "analyzing":
                    network.update_node_status(agent, NodeStatus.THINKING)
                elif status in ["loading", "processing", "creating"]:
                    network.update_node_status(agent, NodeStatus.WORKING)
        
        # 4. 메시지 플로우 시뮬레이션
        message_flows = [
            ("orchestrator", "data_loader", "load_request"),
            ("data_loader", "pandas_agent", "data_ready"),
            ("pandas_agent", "viz_agent", "analysis_complete"),
            ("viz_agent", "orchestrator", "visualization_ready")
        ]
        
        for source, target, msg_type in message_flows:
            network.send_message(source, target, msg_type, {
                "workflow_id": "workflow-test-123",
                "step": msg_type,
                "timestamp": time.time()
            })
        
        # 5. 최종 상태 확인
        # 이벤트 큐에 이벤트들이 쌓였는지 확인
        event_queue_size = realtime_manager.sse_manager.a2a_integration.event_queue.qsize()
        assert event_queue_size >= len(workflow_steps)
        
        # 네트워크 메시지 흐름 확인
        assert len(network.message_flows) >= len(message_flows)
        
        # 시스템 상태 확인
        system_status = realtime_manager.get_system_status()
        assert system_status["subscriptions"] >= 0
        
        # 네트워크 통계 확인 (message_flows 개수로 확인)
        network_stats = network.get_network_stats()
        assert network_stats["message_flows"] >= len(message_flows)
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, full_system):
        """에러 처리 워크플로우 테스트"""
        realtime_manager = full_system["realtime_manager"]
        network = full_system["collaboration_network"]
        
        # 에러 시나리오 시뮬레이션
        error_scenarios = [
            ("data_loader", "error", "Failed to load data: File not found"),
            ("pandas_agent", "error", "Processing error: Invalid data format"),
            ("viz_agent", "error", "Visualization error: Missing required columns")
        ]
        
        for agent, status, error_message in error_scenarios:
            # 에러 상태 업데이트
            await realtime_manager.send_agent_update(
                agent,
                status,
                {"error": error_message, "timestamp": time.time()}
            )
            
            # 네트워크 노드 상태를 ERROR로 업데이트
            if agent in network.nodes:
                network.update_node_status(agent, NodeStatus.ERROR)
        
        # 에러 복구 시뮬레이션
        recovery_scenarios = [
            ("data_loader", "working", "Retrying data load..."),
            ("pandas_agent", "working", "Retrying with cleaned data..."),
            ("viz_agent", "completed", "Visualization created successfully")
        ]
        
        for agent, status, message in recovery_scenarios:
            await realtime_manager.send_agent_update(
                agent,
                status,
                {"message": message, "timestamp": time.time()}
            )
            
            if agent in network.nodes:
                if status == "working":
                    network.update_node_status(agent, NodeStatus.WORKING)
                elif status == "completed":
                    network.update_node_status(agent, NodeStatus.COMPLETED)
        
        # 최종 상태 확인
        event_queue_size = realtime_manager.sse_manager.a2a_integration.event_queue.qsize()
        assert event_queue_size >= len(error_scenarios) + len(recovery_scenarios)
        
        # 네트워크 상태 확인
        error_nodes = [node for node in network.nodes.values() if node.status == NodeStatus.ERROR]
        completed_nodes = [node for node in network.nodes.values() if node.status == NodeStatus.COMPLETED]
        
        # 일부 노드는 복구되어야 함
        assert len(completed_nodes) > 0


class TestPerformanceIntegration:
    """성능 통합 테스트"""
    
    @pytest.fixture
    def performance_system(self):
        """성능 테스트용 시스템"""
        return {
            "realtime_manager": get_cursor_sse_realtime(),
            "collaboration_network": get_cursor_collaboration_network()
        }
    
    @pytest.mark.asyncio
    async def test_high_volume_events(self, performance_system):
        """대용량 이벤트 처리 테스트"""
        realtime_manager = performance_system["realtime_manager"]
        
        # 대용량 이벤트 생성
        num_events = 100
        start_time = time.time()
        
        for i in range(num_events):
            await realtime_manager.send_agent_update(
                f"agent-{i % 10}",  # 10개 에이전트 순환
                "working",
                {"message": f"Processing task {i}", "batch": i // 10}
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 성능 검증 (100개 이벤트를 1초 내에 처리)
        assert processing_time < 1.0, f"Processing time too slow: {processing_time:.2f}s"
        
        # 이벤트 큐 크기 확인
        event_queue_size = realtime_manager.sse_manager.a2a_integration.event_queue.qsize()
        assert event_queue_size >= num_events
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, performance_system):
        """동시 작업 처리 테스트"""
        realtime_manager = performance_system["realtime_manager"]
        network = performance_system["collaboration_network"]
        
        # 동시 작업 정의
        async def agent_task(agent_id: str, task_count: int):
            for i in range(task_count):
                await realtime_manager.send_agent_update(
                    agent_id,
                    "working",
                    {"task": i, "timestamp": time.time()}
                )
                await asyncio.sleep(0.01)  # 작은 지연
        
        def network_task(node_count: int):
            for i in range(node_count):
                network.simulate_message_flow()
        
        # 동시 실행
        start_time = time.time()
        
        # 비동기 작업과 동기 작업 동시 실행
        await asyncio.gather(
            agent_task("agent-1", 20),
            agent_task("agent-2", 20),
            agent_task("agent-3", 20)
        )
        
        # 네트워크 작업 실행
        network_task(30)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 성능 검증
        assert total_time < 2.0, f"Concurrent operations too slow: {total_time:.2f}s"
        
        # 결과 확인
        event_queue_size = realtime_manager.sse_manager.a2a_integration.event_queue.qsize()
        assert event_queue_size >= 60  # 3개 에이전트 × 20개 작업
        
        network_stats = network.get_network_stats()
        assert network_stats["message_flows"] >= 30
    
    def test_memory_usage_stability(self, performance_system):
        """메모리 사용량 안정성 테스트"""
        import gc
        import psutil
        import os
        
        # 초기 메모리 사용량
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        realtime_manager = performance_system["realtime_manager"]
        network = performance_system["collaboration_network"]
        
        # 반복 작업 실행
        for cycle in range(10):
            # 이벤트 생성
            for i in range(50):
                asyncio.run(realtime_manager.send_agent_update(
                    f"agent-{i % 5}",
                    "working",
                    {"cycle": cycle, "task": i}
                ))
            
            # 네트워크 메시지 생성
            for i in range(20):
                network.simulate_message_flow()
            
            # 가비지 컬렉션
            gc.collect()
        
        # 최종 메모리 사용량
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 사용량이 과도하게 증가하지 않았는지 확인 (50MB 제한)
        assert memory_increase < 50, f"Memory usage increased too much: {memory_increase:.2f}MB"
        
        # 시스템 상태 확인
        system_status = realtime_manager.get_system_status()
        assert system_status["subscriptions"] >= 0
        
        network_stats = network.get_network_stats()
        assert network_stats["message_flows"] >= 200  # 10 cycles × 20 messages


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 