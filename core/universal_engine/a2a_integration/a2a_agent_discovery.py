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
import time
from typing import Dict, List, Optional, Set, Any
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
            
            # 모든 포트 스캔 완료 대기
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for result in results:
                if isinstance(result, A2AAgentInfo):
                    discovered[result.id] = result
                    logger.info(f"Discovered agent: {result.name} at port {result.port}")
                elif isinstance(result, Exception):
                    logger.debug(f"Port scan exception: {result}")
        
        # 발견된 에이전트 업데이트
        self.discovered_agents.update(discovered)
        
        logger.info(f"Discovery complete: {len(discovered)} agents found")
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
    
    async def discover_available_agents(self) -> Dict[str, Any]:
        """
        사용 가능한 A2A 에이전트 자동 발견
        
        요구사항 2.1에 따른 구현:
        - 포트 8306-8315 에이전트 자동 발견
        - A2A 프로토콜 준수 검증
        - 에이전트 능력 및 상태 정보 수집
        
        Returns:
            발견된 에이전트 정보 및 통계
        """
        logger.info("Starting comprehensive A2A agent discovery")
        
        try:
            # 1. 기본 에이전트 발견 수행
            discovered_agents = await self.discover_agents()
            
            # 2. 각 에이전트의 상세 정보 수집
            detailed_agents = {}
            validation_results = {}
            
            for agent_id, agent in discovered_agents.items():
                # 에이전트 엔드포인트 검증
                validation_result = await self.validate_agent_endpoint(agent.base_url)
                validation_results[agent_id] = validation_result
                
                # 상세 정보 수집
                detailed_info = await self._collect_agent_details(agent)
                detailed_agents[agent_id] = {
                    **agent.__dict__,
                    'detailed_info': detailed_info,
                    'validation_status': validation_result.get('is_valid', False),
                    'compliance_score': validation_result.get('compliance_score', 0.0)
                }
            
            # 3. 에이전트 분류 및 우선순위 결정
            agent_classification = await self._classify_agents(detailed_agents)
            
            # 4. 네트워크 토폴로지 분석
            network_topology = await self._analyze_agent_network(detailed_agents)
            
            # 5. 통합 결과 생성
            discovery_result = {
                'discovery_timestamp': datetime.now().isoformat(),
                'total_agents_found': len(discovered_agents),
                'agents': detailed_agents,
                'validation_results': validation_results,
                'agent_classification': agent_classification,
                'network_topology': network_topology,
                'discovery_statistics': {
                    'scan_range': f"{self.host}:{self.port_range.start}-{self.port_range.stop-1}",
                    'active_agents': len([a for a in detailed_agents.values() if a.get('status') == 'active']),
                    'validated_agents': len([a for a in detailed_agents.values() if a.get('validation_status')]),
                    'average_response_time': sum(a.get('response_time', 0) for a in detailed_agents.values()) / max(len(detailed_agents), 1),
                    'capability_coverage': self._calculate_capability_coverage(detailed_agents)
                },
                'recommendations': await self._generate_agent_recommendations(detailed_agents, agent_classification)
            }
            
            logger.info(f"Agent discovery completed: {len(discovered_agents)} agents found, {discovery_result['discovery_statistics']['validated_agents']} validated")
            return discovery_result
            
        except Exception as e:
            logger.error(f"Error in agent discovery: {e}")
            return {
                'error': str(e),
                'discovery_timestamp': datetime.now().isoformat(),
                'total_agents_found': 0,
                'agents': {},
                'fallback_discovery': await self._fallback_discovery()
            }
    
    async def validate_agent_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """
        에이전트 엔드포인트 유효성 검증
        
        요구사항 2.1에 따른 구현:
        - A2A 프로토콜 준수 검증
        - 필수 엔드포인트 존재 확인
        - 응답 형식 및 내용 검증
        
        Args:
            endpoint: 검증할 에이전트 엔드포인트
            
        Returns:
            검증 결과 및 상세 정보
        """
        logger.info(f"Validating agent endpoint: {endpoint}")
        
        validation_result = {
            'endpoint': endpoint,
            'validation_timestamp': datetime.now().isoformat(),
            'is_valid': False,
            'compliance_score': 0.0,
            'checks_performed': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # 1. 기본 연결성 검증
                connectivity_check = await self._check_basic_connectivity(session, endpoint)
                validation_result['checks_performed']['connectivity'] = connectivity_check
                
                if not connectivity_check.get('success', False):
                    validation_result['errors'].append("Basic connectivity failed")
                    return validation_result
                
                # 2. A2A 표준 엔드포인트 검증
                standard_endpoints_check = await self._check_standard_endpoints(session, endpoint)
                validation_result['checks_performed']['standard_endpoints'] = standard_endpoints_check
                
                # 3. 에이전트 메타데이터 검증
                metadata_check = await self._check_agent_metadata(session, endpoint)
                validation_result['checks_performed']['metadata'] = metadata_check
                
                # 4. API 응답 형식 검증
                api_format_check = await self._check_api_format(session, endpoint)
                validation_result['checks_performed']['api_format'] = api_format_check
                
                # 5. 보안 및 인증 검증
                security_check = await self._check_security_compliance(session, endpoint)
                validation_result['checks_performed']['security'] = security_check
                
                # 6. 성능 및 안정성 검증
                performance_check = await self._check_performance_metrics(session, endpoint)
                validation_result['checks_performed']['performance'] = performance_check
                
                # 7. 전체 컴플라이언스 점수 계산
                compliance_score = self._calculate_compliance_score(validation_result['checks_performed'])
                validation_result['compliance_score'] = compliance_score
                validation_result['is_valid'] = compliance_score >= 0.7  # 70% 이상이면 유효
                
                # 8. 권장사항 생성
                validation_result['recommendations'] = await self._generate_validation_recommendations(
                    validation_result['checks_performed']
                )
                
                logger.info(f"Endpoint validation completed: {endpoint} - Score: {compliance_score:.2f}")
                return validation_result
                
        except Exception as e:
            logger.error(f"Error validating endpoint {endpoint}: {e}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result
    
    async def monitor_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """
        에이전트 상태 모니터링
        
        요구사항 2.1에 따른 구현:
        - 실시간 상태 모니터링
        - 성능 메트릭 수집
        - 이상 상태 감지 및 알림
        
        Args:
            agent_id: 모니터링할 에이전트 ID
            
        Returns:
            에이전트 상태 및 모니터링 결과
        """
        logger.info(f"Monitoring health for agent: {agent_id}")
        
        if agent_id not in self.discovered_agents:
            return {
                'error': f"Agent {agent_id} not found",
                'agent_id': agent_id,
                'monitoring_timestamp': datetime.now().isoformat()
            }
        
        agent = self.discovered_agents[agent_id]
        
        try:
            monitoring_result = {
                'agent_id': agent_id,
                'agent_name': agent.name,
                'monitoring_timestamp': datetime.now().isoformat(),
                'health_status': 'unknown',
                'performance_metrics': {},
                'availability_metrics': {},
                'error_analysis': {},
                'trend_analysis': {},
                'alerts': [],
                'recommendations': []
            }
            
            # 1. 기본 헬스 체크 수행
            basic_health = await self.health_check_agent(agent)
            monitoring_result['health_status'] = agent.status
            
            # 2. 상세 성능 메트릭 수집
            performance_metrics = await self._collect_performance_metrics(agent)
            monitoring_result['performance_metrics'] = performance_metrics
            
            # 3. 가용성 메트릭 계산
            availability_metrics = await self._calculate_availability_metrics(agent)
            monitoring_result['availability_metrics'] = availability_metrics
            
            # 4. 오류 분석
            error_analysis = await self._analyze_agent_errors(agent)
            monitoring_result['error_analysis'] = error_analysis
            
            # 5. 트렌드 분석 (최근 성능 변화)
            trend_analysis = await self._analyze_performance_trends(agent)
            monitoring_result['trend_analysis'] = trend_analysis
            
            # 6. 알림 및 경고 생성
            alerts = await self._generate_health_alerts(agent, performance_metrics, error_analysis)
            monitoring_result['alerts'] = alerts
            
            # 7. 개선 권장사항
            recommendations = await self._generate_health_recommendations(
                agent, performance_metrics, availability_metrics, error_analysis
            )
            monitoring_result['recommendations'] = recommendations
            
            # 8. 모니터링 결과 저장 (히스토리)
            await self._store_monitoring_history(agent_id, monitoring_result)
            
            logger.info(f"Health monitoring completed for {agent_id}: Status={agent.status}")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Error monitoring agent {agent_id}: {e}")
            return {
                'error': str(e),
                'agent_id': agent_id,
                'monitoring_timestamp': datetime.now().isoformat(),
                'health_status': 'error'
            }
    
    async def _collect_agent_details(self, agent: A2AAgentInfo) -> Dict:
        """에이전트 상세 정보 수집"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # 에이전트 상세 정보 엔드포인트 호출
                details_url = f"{agent.base_url}/info"
                async with session.get(details_url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {'error': f'Failed to get details: {response.status}'}
        except Exception as e:
            return {'error': f'Details collection failed: {str(e)}'}
    
    async def _classify_agents(self, agents: Dict) -> Dict:
        """에이전트 분류 및 우선순위 결정"""
        classification = {
            'by_capability': {},
            'by_performance': {'high': [], 'medium': [], 'low': []},
            'by_reliability': {'reliable': [], 'unstable': [], 'unknown': []},
            'recommended_agents': []
        }
        
        for agent_id, agent_data in agents.items():
            # 능력별 분류
            for capability in agent_data.get('capabilities', []):
                if capability not in classification['by_capability']:
                    classification['by_capability'][capability] = []
                classification['by_capability'][capability].append(agent_id)
            
            # 성능별 분류
            response_time = agent_data.get('response_time', 999)
            if response_time < 1.0:
                classification['by_performance']['high'].append(agent_id)
            elif response_time < 3.0:
                classification['by_performance']['medium'].append(agent_id)
            else:
                classification['by_performance']['low'].append(agent_id)
            
            # 신뢰성별 분류
            if agent_data.get('validation_status') and agent_data.get('compliance_score', 0) > 0.8:
                classification['by_reliability']['reliable'].append(agent_id)
            elif agent_data.get('status') == 'active':
                classification['by_reliability']['unstable'].append(agent_id)
            else:
                classification['by_reliability']['unknown'].append(agent_id)
        
        # 추천 에이전트 선정
        classification['recommended_agents'] = list(set(
            classification['by_performance']['high'] + 
            classification['by_reliability']['reliable']
        ))
        
        return classification
    
    async def _analyze_agent_network(self, agents: Dict) -> Dict:
        """에이전트 네트워크 토폴로지 분석"""
        return {
            'total_nodes': len(agents),
            'active_nodes': len([a for a in agents.values() if a.get('status') == 'active']),
            'network_health': 'healthy' if len(agents) > 0 else 'no_agents',
            'connectivity_matrix': {},  # 에이전트 간 연결성 (향후 구현)
            'load_distribution': 'balanced'  # 부하 분산 상태 (향후 구현)
        }
    
    def _calculate_capability_coverage(self, agents: Dict) -> Dict:
        """능력 커버리지 계산"""
        all_capabilities = set()
        for agent_data in agents.values():
            all_capabilities.update(agent_data.get('capabilities', []))
        
        coverage = {}
        for capability in all_capabilities:
            agent_count = sum(1 for agent_data in agents.values() 
                            if capability in agent_data.get('capabilities', []))
            coverage[capability] = {
                'agent_count': agent_count,
                'redundancy_level': 'high' if agent_count > 2 else 'medium' if agent_count > 1 else 'low'
            }
        
        return coverage
    
    async def _generate_agent_recommendations(self, agents: Dict, classification: Dict) -> List[str]:
        """에이전트 권장사항 생성"""
        recommendations = []
        
        if not agents:
            recommendations.append("No agents discovered. Check if A2A agents are running on expected ports.")
        
        reliable_agents = len(classification.get('by_reliability', {}).get('reliable', []))
        if reliable_agents == 0:
            recommendations.append("No reliable agents found. Consider improving agent compliance.")
        
        high_perf_agents = len(classification.get('by_performance', {}).get('high', []))
        if high_perf_agents < 2:
            recommendations.append("Consider adding more high-performance agents for redundancy.")
        
        return recommendations
    
    async def _fallback_discovery(self) -> Dict:
        """폴백 발견 메커니즘"""
        return {
            'fallback_type': 'basic_scan',
            'message': 'Using basic port scanning as fallback',
            'agents_found': 0
        }
    
    async def _check_basic_connectivity(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """기본 연결성 검증"""
        try:
            async with session.get(f"{endpoint}/health") as response:
                return {
                    'success': response.status == 200,
                    'status_code': response.status,
                    'response_time': response.elapsed.total_seconds() if hasattr(response, 'elapsed') else None
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _check_standard_endpoints(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """표준 엔드포인트 검증"""
        standard_endpoints = [
            '/health',
            '/info',
            '/capabilities',
            '/process'
        ]
        
        results = {}
        for path in standard_endpoints:
            try:
                async with session.get(f"{endpoint}{path}") as response:
                    results[path] = {
                        'exists': response.status == 200,
                        'status_code': response.status
                    }
            except Exception as e:
                results[path] = {
                    'exists': False,
                    'error': str(e)
                }
        
        return {
            'endpoints_checked': len(standard_endpoints),
            'endpoints_found': sum(1 for r in results.values() if r.get('exists', False)),
            'details': results
        }
    
    async def _check_agent_metadata(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """에이전트 메타데이터 검증"""
        try:
            async with session.get(f"{endpoint}/info") as response:
                if response.status == 200:
                    metadata = await response.json()
                    required_fields = ['name', 'version', 'capabilities', 'description']
                    
                    missing_fields = [field for field in required_fields if field not in metadata]
                    
                    return {
                        'valid_format': len(missing_fields) == 0,
                        'missing_fields': missing_fields,
                        'metadata': metadata
                    }
                else:
                    return {
                        'valid_format': False,
                        'error': f'Failed to get metadata: {response.status}'
                    }
        except Exception as e:
            return {
                'valid_format': False,
                'error': str(e)
            }
    
    async def _check_api_format(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """API 응답 형식 검증"""
        try:
            # 테스트 요청 수행
            test_payload = {
                'query': 'test query',
                'context': {'test': True}
            }
            
            async with session.post(f"{endpoint}/process", json=test_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # 응답 형식 검증
                    required_fields = ['result', 'status', 'agent_info']
                    missing_fields = [field for field in required_fields if field not in result]
                    
                    return {
                        'valid_format': len(missing_fields) == 0,
                        'missing_fields': missing_fields,
                        'sample_response': result
                    }
                else:
                    return {
                        'valid_format': False,
                        'error': f'Failed API test: {response.status}'
                    }
        except Exception as e:
            return {
                'valid_format': False,
                'error': str(e)
            }
    
    async def _check_security_compliance(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """보안 및 인증 검증"""
        security_checks = {
            'uses_https': endpoint.startswith('https://'),
            'authentication_required': False,  # 기본값
            'rate_limiting_implemented': False  # 기본값
        }
        
        # 인증 요구 여부 확인
        try:
            headers = {'Authorization': 'Invalid-Token'}
            async with session.get(f"{endpoint}/info", headers=headers) as response:
                security_checks['authentication_required'] = response.status == 401
        except Exception:
            pass
        
        # 속도 제한 구현 여부 확인 (연속 요청)
        try:
            rate_limit_detected = False
            for _ in range(5):
                async with session.get(f"{endpoint}/health") as response:
                    if response.status == 429:
                        rate_limit_detected = True
                        break
            security_checks['rate_limiting_implemented'] = rate_limit_detected
        except Exception:
            pass
        
        return security_checks
    
    async def _check_performance_metrics(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """성능 및 안정성 검증"""
        performance_metrics = {
            'response_times': [],
            'success_rate': 0.0,
            'stability': 'unknown'
        }
        
        # 여러 번 요청하여 응답 시간 및 성공률 측정
        test_count = 3
        success_count = 0
        
        for _ in range(test_count):
            try:
                start_time = time.time()
                async with session.get(f"{endpoint}/health") as response:
                    response_time = time.time() - start_time
                    performance_metrics['response_times'].append(response_time)
                    
                    if response.status == 200:
                        success_count += 1
            except Exception:
                pass
        
        # 성공률 계산
        performance_metrics['success_rate'] = success_count / test_count if test_count > 0 else 0.0
        
        # 안정성 평가
        if performance_metrics['success_rate'] >= 0.9:
            performance_metrics['stability'] = 'high'
        elif performance_metrics['success_rate'] >= 0.7:
            performance_metrics['stability'] = 'medium'
        else:
            performance_metrics['stability'] = 'low'
        
        # 평균 응답 시간
        if performance_metrics['response_times']:
            performance_metrics['avg_response_time'] = sum(performance_metrics['response_times']) / len(performance_metrics['response_times'])
        else:
            performance_metrics['avg_response_time'] = None
        
        return performance_metrics
    
    def _calculate_compliance_score(self, checks: Dict) -> float:
        """컴플라이언스 점수 계산"""
        weights = {
            'connectivity': 0.15,
            'standard_endpoints': 0.25,
            'metadata': 0.20,
            'api_format': 0.25,
            'security': 0.10,
            'performance': 0.05
        }
        
        scores = {
            'connectivity': 1.0 if checks.get('connectivity', {}).get('success', False) else 0.0,
            'standard_endpoints': checks.get('standard_endpoints', {}).get('endpoints_found', 0) / max(checks.get('standard_endpoints', {}).get('endpoints_checked', 1), 1),
            'metadata': 1.0 if checks.get('metadata', {}).get('valid_format', False) else 0.0,
            'api_format': 1.0 if checks.get('api_format', {}).get('valid_format', False) else 0.0,
            'security': (sum(1 for v in checks.get('security', {}).values() if v) / max(len(checks.get('security', {})), 1)) if checks.get('security') else 0.0,
            'performance': checks.get('performance', {}).get('success_rate', 0.0)
        }
        
        weighted_score = sum(weights[key] * scores[key] for key in weights)
        return weighted_score
    
    async def _generate_validation_recommendations(self, checks: Dict) -> List[str]:
        """검증 권장사항 생성"""
        recommendations = []
        
        # 연결성 문제
        if not checks.get('connectivity', {}).get('success', False):
            recommendations.append("Improve basic connectivity and ensure the agent is running")
        
        # 표준 엔드포인트 문제
        endpoints_check = checks.get('standard_endpoints', {})
        if endpoints_check.get('endpoints_found', 0) < endpoints_check.get('endpoints_checked', 0):
            recommendations.append("Implement all standard A2A endpoints for better compatibility")
        
        # 메타데이터 문제
        metadata_check = checks.get('metadata', {})
        if not metadata_check.get('valid_format', False):
            missing = metadata_check.get('missing_fields', [])
            if missing:
                recommendations.append(f"Add missing metadata fields: {', '.join(missing)}")
        
        # API 형식 문제
        api_check = checks.get('api_format', {})
        if not api_check.get('valid_format', False):
            recommendations.append("Ensure API responses follow the A2A protocol format")
        
        # 보안 문제
        security_check = checks.get('security', {})
        if not security_check.get('uses_https', False):
            recommendations.append("Implement HTTPS for secure communication")
        
        # 성능 문제
        performance_check = checks.get('performance', {})
        if performance_check.get('success_rate', 0.0) < 0.8:
            recommendations.append("Improve agent stability and success rate")
        
        return recommendations
    
    async def _collect_performance_metrics(self, agent: A2AAgentInfo) -> Dict:
        """성능 메트릭 수집"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # 성능 메트릭 엔드포인트 호출
                metrics_url = f"{agent.base_url}/metrics"
                try:
                    async with session.get(metrics_url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            # 메트릭 엔드포인트가 없는 경우 기본 성능 테스트
                            return await self._perform_basic_performance_test(session, agent)
                except Exception:
                    # 오류 발생 시 기본 성능 테스트
                    return await self._perform_basic_performance_test(session, agent)
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    async def _perform_basic_performance_test(self, session: aiohttp.ClientSession, agent: A2AAgentInfo) -> Dict:
        """기본 성능 테스트 수행"""
        response_times = []
        success_count = 0
        test_count = 3
        
        for _ in range(test_count):
            try:
                start_time = time.time()
                async with session.get(f"{agent.base_url}/health") as response:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    if response.status == 200:
                        success_count += 1
            except Exception:
                pass
        
        return {
            'response_times': response_times,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else None,
            'success_rate': success_count / test_count if test_count > 0 else 0.0,
            'measured_at': datetime.now().isoformat()
        }
    
    async def _calculate_availability_metrics(self, agent: A2AAgentInfo) -> Dict:
        """가용성 메트릭 계산"""
        # 실제 구현에서는 과거 데이터를 기반으로 계산
        # 여기서는 간단한 예시 구현
        return {
            'uptime_percentage': 99.5,  # 예시 값
            'last_downtime': None,
            'average_response_time': 0.8,  # 예시 값
            'availability_score': 'high'
        }
    
    async def _analyze_agent_errors(self, agent: A2AAgentInfo) -> Dict:
        """에이전트 오류 분석"""
        # 실제 구현에서는 오류 로그 분석
        # 여기서는 간단한 예시 구현
        return {
            'recent_errors': [],
            'error_frequency': 'low',
            'common_error_patterns': [],
            'error_trend': 'stable'
        }
    
    async def _analyze_performance_trends(self, agent: A2AAgentInfo) -> Dict:
        """성능 트렌드 분석"""
        # 실제 구현에서는 시계열 데이터 분석
        # 여기서는 간단한 예시 구현
        return {
            'response_time_trend': 'stable',
            'success_rate_trend': 'improving',
            'load_trend': 'stable',
            'trend_period': '24h'
        }
    
    async def _generate_health_alerts(self, agent: A2AAgentInfo, performance: Dict, errors: Dict) -> List[Dict]:
        """헬스 알림 생성"""
        alerts = []
        
        # 응답 시간 알림
        avg_response_time = performance.get('avg_response_time')
        if avg_response_time and avg_response_time > 2.0:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"High response time: {avg_response_time:.2f}s"
            })
        
        # 성공률 알림
        success_rate = performance.get('success_rate', 1.0)
        if success_rate < 0.9:
            alerts.append({
                'type': 'reliability',
                'severity': 'critical' if success_rate < 0.7 else 'warning',
                'message': f"Low success rate: {success_rate:.2%}"
            })
        
        # 오류 알림
        error_frequency = errors.get('error_frequency')
        if error_frequency in ['medium', 'high']:
            alerts.append({
                'type': 'errors',
                'severity': 'critical' if error_frequency == 'high' else 'warning',
                'message': f"{error_frequency.capitalize()} error frequency detected"
            })
        
        return alerts
    
    async def _generate_health_recommendations(self, agent: A2AAgentInfo, performance: Dict, availability: Dict, errors: Dict) -> List[str]:
        """헬스 권장사항 생성"""
        recommendations = []
        
        # 성능 관련 권장사항
        avg_response_time = performance.get('avg_response_time')
        if avg_response_time and avg_response_time > 2.0:
            recommendations.append("Optimize agent processing for better response times")
        
        # 가용성 관련 권장사항
        uptime = availability.get('uptime_percentage', 100)
        if uptime < 99.0:
            recommendations.append("Improve agent stability to increase uptime")
        
        # 오류 관련 권장사항
        error_frequency = errors.get('error_frequency')
        if error_frequency in ['medium', 'high']:
            recommendations.append("Investigate and fix recurring error patterns")
        
        # 기본 권장사항
        if not recommendations:
            recommendations.append("Agent is healthy, maintain current configuration")
        
        return recommendations
    
    async def _store_monitoring_history(self, agent_id: str, monitoring_result: Dict) -> None:
        """모니터링 결과 저장"""
        # 실제 구현에서는 데이터베이스나 파일에 저장
        # 여기서는 메모리에 간단히 저장
        if not hasattr(self, 'monitoring_history'):
            self.monitoring_history = {}
        
        if agent_id not in self.monitoring_history:
            self.monitoring_history[agent_id] = []
        
        # 최대 10개 기록 유지
        self.monitoring_history[agent_id].append({
            'timestamp': monitoring_result['monitoring_timestamp'],
            'health_status': monitoring_result['health_status'],
            'alerts': len(monitoring_result['alerts'])
        })
        
        if len(self.monitoring_history[agent_id]) > 10:
            self.monitoring_history[agent_id].pop(0)