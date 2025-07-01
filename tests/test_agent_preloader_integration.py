"""
에이전트 프리로더 통합 테스트
실제 AI DS Team 에이전트들과의 통합을 테스트합니다.
"""

import pytest
import asyncio
import httpx
import time
from unittest.mock import patch, Mock
from core.ui.agent_preloader import (
    AgentPreloader, 
    get_agent_preloader,
    ProgressiveLoadingUI,
    AgentStatus
)

# 실제 AI DS Team 에이전트 설정
REAL_AGENTS_CONFIG = {
    "Orchestrator": {"port": 8100, "description": "AI DS Team을 지휘하는 마에스트로", "capabilities": ["planning", "delegation"], "color": "#FAD02E"},
    "🧹 Data Cleaning": {"port": 8306, "description": "누락값 처리, 이상치 제거", "capabilities": ["missing_value", "outlier"], "color": "#FF6B6B"},
    "📊 Data Visualization": {"port": 8308, "description": "고급 시각화 생성", "capabilities": ["charts", "plots"], "color": "#4ECDC4"},
    "🔍 EDA Tools": {"port": 8312, "description": "자동 EDA 및 상관관계 분석", "capabilities": ["eda", "correlation"], "color": "#45B7D1"},
}

class TestAgentPreloaderIntegration:
    """에이전트 프리로더 통합 테스트"""
    
    @pytest.fixture
    def preloader(self):
        """실제 에이전트 설정으로 프리로더 생성"""
        return AgentPreloader(REAL_AGENTS_CONFIG)
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, preloader):
        """실제 에이전트 헬스 체크 통합 테스트"""
        # 실제 에이전트 서버가 실행 중인지 확인
        available_agents = []
        
        async with httpx.AsyncClient(timeout=2.0) as client:
            for agent_name, config in REAL_AGENTS_CONFIG.items():
                try:
                    url = f"http://localhost:{config['port']}/.well-known/agent.json"
                    response = await client.get(url)
                    if response.status_code == 200:
                        available_agents.append(agent_name)
                except httpx.RequestError:
                    pass  # 에이전트가 실행 중이지 않음
        
        if not available_agents:
            pytest.skip("실행 중인 AI DS Team 에이전트가 없습니다. 통합 테스트를 건너뜁니다.")
        
        # 사용 가능한 에이전트들에 대해 초기화 테스트
        for agent_name in available_agents:
            result = await preloader._initialize_single_agent(agent_name)
            assert result == True
            
            agent_info = preloader.get_agent_status(agent_name)
            assert agent_info.status == AgentStatus.READY
            assert agent_info.initialization_time is not None
            assert agent_info.initialization_time > 0
    
    @pytest.mark.asyncio
    async def test_preload_with_real_agents(self, preloader):
        """실제 에이전트들과 프리로딩 테스트"""
        # 실행 중인 에이전트 확인
        running_agents_count = 0
        async with httpx.AsyncClient(timeout=2.0) as client:
            for agent_name, config in REAL_AGENTS_CONFIG.items():
                try:
                    url = f"http://localhost:{config['port']}/.well-known/agent.json"
                    response = await client.get(url)
                    if response.status_code == 200:
                        running_agents_count += 1
                except httpx.RequestError:
                    pass
        
        if running_agents_count == 0:
            pytest.skip("실행 중인 AI DS Team 에이전트가 없습니다.")
        
        # 프로그레스 추적
        progress_updates = []
        def progress_callback(completed, total, current_task):
            progress_updates.append((completed, total, current_task))
        
        # 프리로딩 실행
        start_time = time.time()
        agents_info = await preloader.preload_agents(progress_callback)
        total_time = time.time() - start_time
        
        # 결과 검증
        assert len(agents_info) == len(REAL_AGENTS_CONFIG)
        assert preloader.is_initialization_complete() == True
        
        # 성능 검증 (10초 이내 완료)
        assert total_time < 10.0
        
        # 프로그레스 콜백 검증
        assert len(progress_updates) > 0
        
        # 요약 정보 검증
        summary = preloader.get_initialization_summary()
        assert summary["total_agents"] == len(REAL_AGENTS_CONFIG)
        assert summary["ready_agents"] >= 0
        assert summary["ready_agents"] <= len(REAL_AGENTS_CONFIG)
        
        # 실행 중인 에이전트 수만큼 성공해야 함
        expected_ready = min(running_agents_count, len(REAL_AGENTS_CONFIG))
        if expected_ready > 0:
            assert summary["ready_agents"] >= expected_ready * 0.8  # 80% 이상 성공
    
    @pytest.mark.asyncio
    async def test_priority_loading_order(self, preloader):
        """우선순위 로딩 순서 테스트"""
        progress_updates = []
        
        def progress_callback(completed, total, current_task):
            progress_updates.append(current_task)
        
        with patch.object(preloader, '_initialize_single_agent') as mock_init:
            # Mock이 항상 성공하도록 설정
            mock_init.return_value = True
            
            await preloader.preload_agents(progress_callback)
            
            # 호출된 순서 확인
            called_agents = [call[0][0] for call in mock_init.call_args_list]
            
            # 우선순위 에이전트들이 먼저 호출되었는지 확인
            priority_agents = ["Orchestrator", "🔍 EDA Tools", "📊 Data Visualization"]
            
            for priority_agent in priority_agents:
                if priority_agent in called_agents:
                    priority_index = called_agents.index(priority_agent)
                    
                    # 우선순위 에이전트가 다른 에이전트들보다 먼저 호출되었는지 확인
                    for agent_name in REAL_AGENTS_CONFIG.keys():
                        if agent_name not in priority_agents and agent_name in called_agents:
                            other_index = called_agents.index(agent_name)
                            # 우선순위 에이전트는 앞쪽 3개 슬롯 중 하나에 있어야 함
                            assert priority_index < 3
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """캐싱 동작 테스트"""
        # 첫 번째 프리로더 인스턴스
        preloader1 = get_agent_preloader(REAL_AGENTS_CONFIG)
        
        # 두 번째 프리로더 인스턴스 (캐시됨)
        preloader2 = get_agent_preloader(REAL_AGENTS_CONFIG)
        
        # 같은 인스턴스여야 함
        assert preloader1 is preloader2
        
        # 다른 설정으로는 다른 인스턴스
        different_config = {"test": {"port": 9999, "description": "Test"}}
        preloader3 = get_agent_preloader(different_config)
        
        assert preloader3 is not preloader1

class TestPerformanceIntegration:
    """성능 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization_performance(self):
        """동시 초기화 성능 테스트"""
        preloader = AgentPreloader(REAL_AGENTS_CONFIG)
        
        with patch.object(preloader, '_initialize_single_agent') as mock_init:
            # Mock 지연 시뮬레이션 (각 에이전트 초기화에 0.5초)
            async def mock_initialize(agent_name):
                await asyncio.sleep(0.5)
                return True
            
            mock_init.side_effect = mock_initialize
            
            start_time = time.time()
            await preloader.preload_agents()
            total_time = time.time() - start_time
            
            # 병렬 처리로 인해 순차 처리(2초)보다 빨라야 함
            # 우선순위 에이전트 3개(1.5초) + 병렬 처리 1개(0.5초) = 약 2초
            assert total_time < 3.0  # 여유를 두고 3초 이내
            
            # 하지만 최소한의 시간은 걸려야 함
            assert total_time > 1.0  # 최소 1초 이상

class TestErrorHandlingIntegration:
    """오류 처리 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """부분 실패 처리 테스트"""
        # 일부 포트는 사용 불가능하도록 설정
        mixed_config = {
            "Working Agent": {"port": 8100, "description": "Should work"},
            "Broken Agent": {"port": 9999, "description": "Should fail"}  # 사용하지 않는 포트
        }
        
        preloader = AgentPreloader(mixed_config)
        
        # 실제 네트워크 요청으로 테스트
        agents_info = await preloader.preload_agents()
        
        # 모든 에이전트가 처리되어야 함
        assert len(agents_info) == 2
        
        # 적어도 하나는 실패해야 함 (포트 9999는 사용되지 않음)
        failed_agents = preloader.get_failed_agents()
        assert len(failed_agents) >= 1
        assert "Broken Agent" in failed_agents
        
        # 요약 정보에 실패가 반영되어야 함
        summary = preloader.get_initialization_summary()
        assert summary["failed_agents"] >= 1
        assert summary["success_rate"] < 100.0
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """타임아웃 처리 테스트"""
        preloader = AgentPreloader(REAL_AGENTS_CONFIG)
        
        with patch('httpx.AsyncClient') as mock_client:
            # 타임아웃 시뮬레이션
            mock_client_instance = Mock()
            mock_client_instance.get.side_effect = httpx.TimeoutException("Request timeout")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # 프리로딩 실행
            agents_info = await preloader.preload_agents()
            
            # 모든 에이전트가 실패해야 함
            assert len(agents_info) == len(REAL_AGENTS_CONFIG)
            
            failed_agents = preloader.get_failed_agents()
            assert len(failed_agents) == len(REAL_AGENTS_CONFIG)
            
            # 모든 에이전트의 오류 메시지가 설정되어야 함
            for agent_name in REAL_AGENTS_CONFIG.keys():
                agent_info = preloader.get_agent_status(agent_name)
                assert agent_info.status == AgentStatus.FAILED
                assert "timeout" in agent_info.error_message.lower()

class TestUIIntegration:
    """UI 통합 테스트"""
    
    def test_progressive_loading_ui_integration(self):
        """프로그레시브 로딩 UI 통합 테스트"""
        # Mock Streamlit 컨테이너
        mock_container = Mock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        
        # UI 컴포넌트 생성
        loading_ui = ProgressiveLoadingUI(mock_container)
        
        # 진행 상황 시뮬레이션
        total_agents = 4
        for i in range(total_agents + 1):
            loading_ui.update_progress(i, total_agents, f"Processing agent {i}")
        
        # 완료 상태 표시
        mock_summary = {
            "total_agents": 4,
            "ready_agents": 3,
            "failed_agents": 1,
            "success_rate": 75.0,
            "initialization_time": 2.5,
            "is_complete": True
        }
        
        loading_ui.show_completion(mock_summary)
        
        # UI 메서드들이 호출되었는지 확인 (실제 Streamlit 없이는 제한적)
        assert mock_container.__enter__.called
        assert mock_container.__exit__.called 