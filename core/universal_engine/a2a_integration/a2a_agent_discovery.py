"""
A2A Agent Discovery System - A2A 에이전트 자동 발견 시스템

요구사항에 따른 구현:
- 포트 8306-8315 에이전트 자동 발견
- /.well-known/agent.json 엔드포인트 검증
- 에이전트 상태 모니터링 및 헬스 체크
- 하드코딩 없는 동적 에이전트 관리
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class A2AAgentInfo:
    """A2A 에이전트 정보"""
    id: str
    name: str
    port: int
    base_url: str
    capabilities: List[str]
    description: str
    version: str
    status: str = "unknown"
    last_health_check: Optional[datetime] = None
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0


class A2AAgentDiscoverySystem:
    """
    A2A 에이전트 자동 발견 및 관리 시스템
    - 포트 스캔을 통한 에이전트 발견
    - 헬스 체크 및 상태 모니터링
    - 동적 에이전트 등록/해제
    """
    
    def __init__(self, host: str = "localhost", port_range: range = range(8306, 8316)):
        """
        A2AAgentDiscoverySystem 초기화
        
        Args:
            host: 호스트 주소
            port_range: 스캔할 포트 범위 (기본: 8306-8315)
        """
        self.host = host
        self.port_range = port_range
        self.discovered_agents: Dict[str, A2AAgentInfo] = {}
        self.health_check_interval = 60  # 60초마다 헬스 체크
        self.timeout = 5  # 5초 타임아웃
        self._discovery_task = None
        self._health_check_task = None
        logger.info(f"A2AAgentDiscoverySystem initialized for {host}:{port_range}")
    
    async def start_discovery(self) -> None:
        """에이전트 발견 및 모니터링 시작"""
        logger.info("Starting A2A agent discovery and monitoring")
        
        # 초기 발견
        await self.discover_agents()
        
        # 백그라운드 모니터링 시작
        self._health_check_task = asyncio.create_task(self._continuous_health_monitoring())
    
    async def stop_discovery(self) -> None:
        """에이전트 발견 및 모니터링 중지"""
        logger.info("Stopping A2A agent discovery and monitoring")
        
        if self._discovery_task:
            self._discovery_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
    
    async def discover_agents(self) -> Dict[str, A2AAgentInfo]:
        """
        포트 범위를 스캔하여 A2A 에이전트 발견
        
        Returns:
            발견된 에이전트들의 딕셔너리
        """
        logger.info(f"Discovering A2A agents on {self.host}:{self.port_range}")
        
        discovered = {}
        tasks = []
        
        # 병렬로 포트 스캔
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for port in self.port_range:
                task = asyncio.create_task(
                    self._check_agent_at_port(session, port)
                )
                tasks.append(task)
            
            # 모든 포트 검사 완료 대기
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, A2AAgentInfo):
                    discovered[result.id] = result
                    self.discovered_agents[result.id] = result
                    logger.info(f"Discovered agent: {result.name} at port {result.port}")
        
        logger.info(f"Discovery complete. Found {len(discovered)} agents")
        return discovered
    
    async def _check_agent_at_port(self, session: aiohttp.ClientSession, port: int) -> Optional[A2AAgentInfo]:
        """
        특정 포트에서 A2A 에이전트 확인
        
        Args:
            session: HTTP 세션
            port: 확인할 포트
            
        Returns:
            발견된 에이전트 정보 또는 None
        """
        base_url = f"http://{self.host}:{port}"
        agent_info_url = f"{base_url}/.well-known/agent.json"
        
        try:
            start_time = datetime.now()
            
            async with session.get(agent_info_url) as response:
                if response.status == 200:
                    agent_data = await response.json()
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    # A2A 표준 에이전트 정보 파싱
                    agent_info = A2AAgentInfo(
                        id=agent_data.get('id', f'agent_{port}'),
                        name=agent_data.get('name', f'Agent at {port}'),
                        port=port,
                        base_url=base_url,
                        capabilities=agent_data.get('capabilities', []),
                        description=agent_data.get('description', ''),
                        version=agent_data.get('version', '1.0.0'),
                        status="active",
                        last_health_check=datetime.now(),
                        response_time=response_time
                    )
                    
                    return agent_info
                    
        except asyncio.TimeoutError:
            logger.debug(f"Timeout checking agent at port {port}")
        except aiohttp.ClientError as e:
            logger.debug(f"Connection error checking port {port}: {e}")
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON from port {port}: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error checking port {port}: {e}")
        
        return None
    
    async def health_check_agent(self, agent: A2AAgentInfo) -> bool:
        """
        개별 에이전트 헬스 체크
        
        Args:
            agent: 체크할 에이전트 정보
            
        Returns:
            헬스 체크 성공 여부
        """
        health_url = f"{agent.base_url}/health"
        
        try:
            start_time = datetime.now()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(health_url) as response:
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if response.status == 200:
                        agent.status = "active"
                        agent.last_health_check = datetime.now()
                        agent.response_time = response_time
                        agent.success_count += 1
                        return True
                    else:
                        agent.status = "unhealthy"
                        agent.error_count += 1
                        return False
                        
        except Exception as e:
            logger.warning(f"Health check failed for agent {agent.name}: {e}")
            agent.status = "offline"
            agent.error_count += 1
            return False
    
    async def _continuous_health_monitoring(self) -> None:
        """지속적인 헬스 모니터링 백그라운드 태스크"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if self.discovered_agents:
                    logger.debug("Running periodic health checks")
                    
                    health_tasks = [
                        self.health_check_agent(agent) 
                        for agent in self.discovered_agents.values()
                    ]
                    
                    results = await asyncio.gather(*health_tasks, return_exceptions=True)
                    
                    healthy_count = sum(1 for result in results if result is True)
                    logger.debug(f"Health check complete: {healthy_count}/{len(self.discovered_agents)} agents healthy")
                    
            except asyncio.CancelledError:
                logger.info("Health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    def get_available_agents(self, include_offline: bool = False) -> Dict[str, A2AAgentInfo]:
        """
        사용 가능한 에이전트 목록 조회
        
        Args:
            include_offline: 오프라인 에이전트 포함 여부
            
        Returns:
            사용 가능한 에이전트들
        """
        if include_offline:
            return self.discovered_agents.copy()
        
        return {
            agent_id: agent 
            for agent_id, agent in self.discovered_agents.items() 
            if agent.status == "active"
        }
    
    def get_agents_by_capability(self, capability: str) -> Dict[str, A2AAgentInfo]:
        """
        특정 능력을 가진 에이전트 검색
        
        Args:
            capability: 찾고자 하는 능력
            
        Returns:
            해당 능력을 가진 에이전트들
        """
        return {
            agent_id: agent 
            for agent_id, agent in self.discovered_agents.items() 
            if capability.lower() in [cap.lower() for cap in agent.capabilities]
            and agent.status == "active"
        }
    
    def get_agent_statistics(self) -> Dict:
        """
        에이전트 통계 정보 조회
        
        Returns:
            전체 에이전트 통계
        """
        total_agents = len(self.discovered_agents)
        active_agents = len([a for a in self.discovered_agents.values() if a.status == "active"])
        offline_agents = len([a for a in self.discovered_agents.values() if a.status == "offline"])
        
        avg_response_time = 0
        if self.discovered_agents:
            response_times = [a.response_time for a in self.discovered_agents.values() if a.response_time > 0]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        # 능력별 에이전트 수
        capability_counts = {}
        for agent in self.discovered_agents.values():
            for capability in agent.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'offline_agents': offline_agents,
            'unhealthy_agents': total_agents - active_agents - offline_agents,
            'average_response_time': avg_response_time,
            'capability_distribution': capability_counts,
            'last_discovery': max(
                [a.last_health_check for a in self.discovered_agents.values() if a.last_health_check],
                default=None
            )
        }
    
    async def rediscover_agents(self) -> Dict[str, A2AAgentInfo]:
        """
        에이전트 재발견
        
        Returns:
            새로 발견된 에이전트들
        """
        logger.info("Re-discovering A2A agents")
        old_agents = set(self.discovered_agents.keys())
        
        await self.discover_agents()
        
        new_agents = set(self.discovered_agents.keys())
        newly_discovered = new_agents - old_agents
        disappeared = old_agents - new_agents
        
        if newly_discovered:
            logger.info(f"Newly discovered agents: {newly_discovered}")
        if disappeared:
            logger.info(f"Agents no longer available: {disappeared}")
            # 사라진 에이전트들 제거
            for agent_id in disappeared:
                del self.discovered_agents[agent_id]
        
        return {
            agent_id: self.discovered_agents[agent_id] 
            for agent_id in newly_discovered
        }
    
    async def register_manual_agent(self, agent_info: A2AAgentInfo) -> bool:
        """
        수동으로 에이전트 등록
        
        Args:
            agent_info: 등록할 에이전트 정보
            
        Returns:
            등록 성공 여부
        """
        logger.info(f"Manually registering agent: {agent_info.name}")
        
        # 헬스 체크 실행
        is_healthy = await self.health_check_agent(agent_info)
        
        if is_healthy:
            self.discovered_agents[agent_info.id] = agent_info
            logger.info(f"Successfully registered agent: {agent_info.name}")
            return True
        else:
            logger.warning(f"Failed to register agent {agent_info.name}: health check failed")
            return False
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        에이전트 제거
        
        Args:
            agent_id: 제거할 에이전트 ID
            
        Returns:
            제거 성공 여부
        """
        if agent_id in self.discovered_agents:
            removed_agent = self.discovered_agents.pop(agent_id)
            logger.info(f"Removed agent: {removed_agent.name}")
            return True
        else:
            logger.warning(f"Agent {agent_id} not found for removal")
            return False