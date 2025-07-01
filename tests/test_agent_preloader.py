"""
에이전트 프리로더 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import httpx
from core.ui.agent_preloader import (
    AgentPreloader, 
    AgentInfo, 
    AgentStatus, 
    get_agent_preloader,
    ProgressiveLoadingUI
)

class TestAgentInfo:
    """AgentInfo 클래스 테스트"""
    
    def test_agent_info_creation(self):
        """AgentInfo 생성 테스트"""
        agent = AgentInfo(
            name="Test Agent",
            port=8100,
            description="Test Description",
            capabilities=["test"],
            color="#FF0000"
        )
        
        assert agent.name == "Test Agent"
        assert agent.port == 8100
        assert agent.description == "Test Description"
        assert agent.capabilities == ["test"]
        assert agent.color == "#FF0000"
        assert agent.status == AgentStatus.UNKNOWN
        assert agent.health_url is None

class TestAgentPreloader:
    """AgentPreloader 클래스 테스트"""
    
    @pytest.fixture
    def sample_agents_config(self):
        """테스트용 에이전트 설정"""
        return {
            "Orchestrator": {
                "port": 8100,
                "description": "Test Orchestrator",
                "capabilities": ["planning"],
                "color": "#FAD02E"
            },
            "Data Loader": {
                "port": 8307,
                "description": "Test Data Loader",
                "capabilities": ["loading"],
                "color": "#96CEB4"
            }
        }
    
    @pytest.fixture
    def preloader(self, sample_agents_config):
        """테스트용 프리로더 인스턴스"""
        return AgentPreloader(sample_agents_config)
    
    def test_initialization(self, preloader, sample_agents_config):
        """프리로더 초기화 테스트"""
        assert len(preloader.agents) == 2
        assert "Orchestrator" in preloader.agents
        assert "Data Loader" in preloader.agents
        
        orchestrator = preloader.agents["Orchestrator"]
        assert orchestrator.name == "Orchestrator"
        assert orchestrator.port == 8100
        assert orchestrator.description == "Test Orchestrator"
        assert orchestrator.health_url == "http://localhost:8100/.well-known/agent.json"
    
    def test_get_agent_status(self, preloader):
        """개별 에이전트 상태 조회 테스트"""
        agent = preloader.get_agent_status("Orchestrator")
        assert agent is not None
        assert agent.name == "Orchestrator"
        
        # 존재하지 않는 에이전트
        agent = preloader.get_agent_status("NonExistent")
        assert agent is None
    
    def test_get_all_agents_status(self, preloader):
        """모든 에이전트 상태 조회 테스트"""
        agents = preloader.get_all_agents_status()
        assert len(agents) == 2
        assert "Orchestrator" in agents
        assert "Data Loader" in agents
    
    def test_get_ready_agents_empty(self, preloader):
        """준비된 에이전트가 없을 때 테스트"""
        ready_agents = preloader.get_ready_agents()
        assert len(ready_agents) == 0
    
    def test_get_failed_agents_empty(self, preloader):
        """실패한 에이전트가 없을 때 테스트"""
        failed_agents = preloader.get_failed_agents()
        assert len(failed_agents) == 0
    
    def test_initialization_summary_initial(self, preloader):
        """초기 상태의 요약 정보 테스트"""
        summary = preloader.get_initialization_summary()
        assert summary["total_agents"] == 2
        assert summary["ready_agents"] == 0
        assert summary["failed_agents"] == 0
        assert summary["success_rate"] == 0
        assert summary["is_complete"] == False
    
    @pytest.mark.asyncio
    async def test_single_agent_initialization_success(self, preloader):
        """단일 에이전트 초기화 성공 테스트"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock 응답 설정
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "Test Agent"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # 에이전트 초기화 실행
            result = await preloader._initialize_single_agent("Orchestrator")
            
            assert result == True
            agent = preloader.get_agent_status("Orchestrator")
            assert agent.status == AgentStatus.READY
            assert agent.initialization_time is not None
            assert agent.initialization_time > 0
    
    @pytest.mark.asyncio
    async def test_single_agent_initialization_failure(self, preloader):
        """단일 에이전트 초기화 실패 테스트"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock 응답 설정 (실패)
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.RequestError("Connection failed")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # 에이전트 초기화 실행
            result = await preloader._initialize_single_agent("Orchestrator")
            
            assert result == False
            agent = preloader.get_agent_status("Orchestrator")
            assert agent.status == AgentStatus.FAILED
            assert agent.error_message == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_preload_agents_with_callback(self, preloader):
        """콜백과 함께 에이전트 프리로딩 테스트"""
        callback_calls = []
        
        def progress_callback(completed, total, current_task):
            callback_calls.append((completed, total, current_task))
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock 응답 설정
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "Test Agent"}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # 에이전트 프리로딩 실행
            agents_info = await preloader.preload_agents(progress_callback)
            
            # 결과 검증
            assert len(agents_info) == 2
            assert preloader.is_initialization_complete() == True
            
            # 콜백 호출 검증
            assert len(callback_calls) == 2  # 2개 에이전트
            
            # 요약 정보 검증
            summary = preloader.get_initialization_summary()
            assert summary["ready_agents"] == 2
            assert summary["success_rate"] == 100.0

class TestProgressiveLoadingUI:
    """ProgressiveLoadingUI 클래스 테스트"""
    
    @pytest.fixture
    def mock_container(self):
        """Mock Streamlit 컨테이너"""
        return Mock()
    
    def test_ui_initialization(self, mock_container):
        """UI 초기화 테스트"""
        loading_ui = ProgressiveLoadingUI(mock_container)
        assert loading_ui.container == mock_container
        assert loading_ui.progress_bar is None
        assert loading_ui.status_text is None
    
    @patch('streamlit.progress')
    @patch('streamlit.empty')
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_setup_ui(self, mock_markdown, mock_expander, mock_empty, mock_progress, mock_container):
        """UI 설정 테스트"""
        # Mock 컨테이너의 __enter__와 __exit__ 메서드 설정
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=None)
        
        loading_ui = ProgressiveLoadingUI(mock_container)
        loading_ui.setup_ui()
        
        # UI 컴포넌트가 생성되었는지 확인
        mock_markdown.assert_called_once()
        mock_progress.assert_called_once_with(0)
        mock_empty.assert_called_once()
        mock_expander.assert_called_once()

class TestCachedPreloader:
    """캐시된 프리로더 테스트"""
    
    def test_get_agent_preloader_caching(self):
        """프리로더 캐싱 테스트"""
        config = {"test": {"port": 8100, "description": "Test"}}
        
        # 첫 번째 호출
        preloader1 = get_agent_preloader(config)
        
        # 두 번째 호출 (같은 인스턴스여야 함)
        preloader2 = get_agent_preloader(config)
        
        assert preloader1 is preloader2  # 같은 인스턴스

class TestIntegrationScenarios:
    """통합 시나리오 테스트"""
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenario(self):
        """성공과 실패가 섞인 시나리오 테스트"""
        config = {
            "Success Agent": {"port": 8100, "description": "Will succeed"},
            "Fail Agent": {"port": 8101, "description": "Will fail"}
        }
        
        preloader = AgentPreloader(config)
        
        with patch('httpx.AsyncClient') as mock_client:
            def side_effect_func(url):
                if "8100" in url:
                    # 성공 응답
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"name": "Success Agent"}
                    return mock_response
                else:
                    # 실패
                    raise httpx.RequestError("Connection failed")
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = side_effect_func
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # 프리로딩 실행
            agents_info = await preloader.preload_agents()
            
            # 결과 검증
            assert len(agents_info) == 2
            
            success_agent = preloader.get_agent_status("Success Agent")
            fail_agent = preloader.get_agent_status("Fail Agent")
            
            assert success_agent.status == AgentStatus.READY
            assert fail_agent.status == AgentStatus.FAILED
            
            # 요약 정보 검증
            summary = preloader.get_initialization_summary()
            assert summary["ready_agents"] == 1
            assert summary["failed_agents"] == 1
            assert summary["success_rate"] == 50.0 