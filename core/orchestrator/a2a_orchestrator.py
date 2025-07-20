"""
A2A Orchestrator - 멀티 에이전트 협업 관리
A2A 프로토콜 기반 동적 에이전트 오케스트레이션

Features:
- 동적 에이전트 관리 및 선택
- LLM 기반 분석 계획 수립
- 실시간 에이전트 상태 모니터링
- 장애 감지 및 자동 복구
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import httpx
import json

from config.agents_config import AgentConfigLoader, AgentConfig, AgentStatus

logger = logging.getLogger(__name__)

@dataclass
class AnalysisPlan:
    """분석 계획 데이터 모델"""
    id: str
    user_query: str
    selected_agents: List[str]
    execution_sequence: List['ExecutionStep']
    estimated_duration: timedelta
    data_requirements: Dict[str, Any]
    expected_outputs: List[str]

@dataclass
class ExecutionStep:
    """실행 단계 데이터 모델"""
    step_id: str
    agent_name: str
    task_description: str
    input_data: Dict[str, Any]
    expected_output: str
    dependencies: List[str]
    timeout: timedelta

@dataclass
class ExecutionResult:
    """실행 결과 데이터 모델"""
    id: str
    plan_id: str
    status: str  # 'success', 'failed', 'partial', 'running'
    artifacts: List[Dict[str, Any]]
    generated_code: List[Dict[str, str]]
    execution_time: timedelta
    agent_contributions: Dict[str, Dict[str, Any]]
    error_details: Optional[str] = None

class A2AOrchestrator:
    """A2A 프로토콜 기반 멀티 에이전트 오케스트레이터"""
    
    def __init__(self, config_path: str = "config/agents.json"):
        self.config_path = config_path
        self.agent_loader = AgentConfigLoader(config_path)
        self.agents = {}
        self.agent_status = {}
        self.active_executions = {}
        self.health_check_tasks = {}
        
    async def initialize(self):
        """오케스트레이터 초기화"""
        await self.reload_agents_config()
        await self.verify_all_agents_online()
        self._start_health_monitoring()
        logger.info("A2A Orchestrator initialized successfully")
    
    async def reload_agents_config(self) -> None:
        """런타임 중 에이전트 설정 재로드"""
        self.agents = await self.agent_loader.reload_config()
        logger.info(f"Reloaded {len(self.agents)} agent configurations")
    
    async def add_agent_dynamically(self, agent_config: AgentConfig) -> bool:
        """런타임 중 새 에이전트 추가"""
        success = await self.agent_loader.add_agent(agent_config)
        if success:
            await self.reload_agents_config()
            await self._start_agent_health_check(agent_config.id)
        return success
    
    async def remove_agent_dynamically(self, agent_id: str) -> bool:
        """런타임 중 에이전트 제거"""
        success = await self.agent_loader.remove_agent(agent_id)
        if success:
            await self._stop_agent_health_check(agent_id)
            self.agents.pop(agent_id, None)
            self.agent_status.pop(agent_id, None)
        return success
    
    async def create_analysis_plan(self, user_query: str, data_context: Dict) -> AnalysisPlan:
        """LLM 기반 분석 계획 수립"""
        try:
            # LLM을 사용하여 사용자 쿼리 분석
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 에이전트 선택 로직 (LLM 기반으로 확장 예정)
            selected_agents = await self._select_optimal_agents(user_query, data_context)
            
            # 실행 순서 계획
            execution_sequence = await self._create_execution_sequence(selected_agents, user_query, data_context)
            
            # 예상 실행 시간 계산
            estimated_duration = self._estimate_execution_time(execution_sequence)
            
            plan = AnalysisPlan(
                id=plan_id,
                user_query=user_query,
                selected_agents=selected_agents,
                execution_sequence=execution_sequence,
                estimated_duration=estimated_duration,
                data_requirements=data_context,
                expected_outputs=["analysis_result", "visualizations", "summary"]
            )
            
            logger.info(f"Created analysis plan {plan_id} with {len(selected_agents)} agents")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating analysis plan: {e}")
            raise
    
    async def execute_plan(self, plan: AnalysisPlan) -> ExecutionResult:
        """분석 계획 실행"""
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            self.active_executions[execution_id] = {
                'plan': plan,
                'status': 'running',
                'start_time': start_time,
                'current_step': 0
            }
            
            artifacts = []
            generated_code = []
            agent_contributions = {}
            
            # 실행 단계별 처리
            for i, step in enumerate(plan.execution_sequence):
                self.active_executions[execution_id]['current_step'] = i
                
                logger.info(f"Executing step {i+1}/{len(plan.execution_sequence)}: {step.task_description}")
                
                # 에이전트 실행
                step_result = await self._execute_step(step)
                
                if step_result['status'] == 'success':
                    artifacts.extend(step_result.get('artifacts', []))
                    generated_code.extend(step_result.get('code', []))
                    agent_contributions[step.agent_name] = step_result
                else:
                    # 단계 실패 처리
                    logger.error(f"Step {i+1} failed: {step_result.get('error')}")
                    break
            
            execution_time = datetime.now() - start_time
            
            result = ExecutionResult(
                id=execution_id,
                plan_id=plan.id,
                status='success' if len(artifacts) > 0 else 'failed',
                artifacts=artifacts,
                generated_code=generated_code,
                execution_time=execution_time,
                agent_contributions=agent_contributions
            )
            
            self.active_executions[execution_id]['status'] = 'completed'
            logger.info(f"Execution {execution_id} completed in {execution_time}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing plan {plan.id}: {e}")
            self.active_executions[execution_id]['status'] = 'failed'
            
            return ExecutionResult(
                id=execution_id,
                plan_id=plan.id,
                status='failed',
                artifacts=[],
                generated_code=[],
                execution_time=datetime.now() - start_time,
                agent_contributions={},
                error_details=str(e)
            )
    
    async def monitor_execution(self, execution_id: str) -> Dict[str, Any]:
        """실행 상태 모니터링"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return {'status': 'not_found'}
        
        return {
            'status': execution['status'],
            'progress': f"{execution['current_step']}/{len(execution['plan'].execution_sequence)}",
            'elapsed_time': str(datetime.now() - execution['start_time']),
            'current_step': execution.get('current_step', 0)
        }
    
    async def verify_all_agents_online(self) -> Dict[str, bool]:
        """모든 에이전트 온라인 상태 확인"""
        status_results = {}
        
        for agent_id, agent_config in self.agents.items():
            if not agent_config.enabled:
                continue
                
            is_online = await self._check_agent_health(agent_config)
            status_results[agent_id] = is_online
            
            self.agent_status[agent_id] = AgentStatus(
                agent_id=agent_id,
                name=agent_config.name,
                status='online' if is_online else 'offline',
                capabilities=agent_config.capabilities,
                current_load=0.0,
                last_heartbeat=datetime.now() if is_online else datetime.min,
                performance_metrics={}
            )
        
        online_count = sum(status_results.values())
        total_count = len(status_results)
        logger.info(f"Agent status check: {online_count}/{total_count} agents online")
        
        return status_results
    
    async def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """에이전트 능력 조회"""
        agent_config = self.agents.get(agent_name)
        return agent_config.capabilities if agent_config else []
    
    async def handle_agent_failure(self, agent_id: str, error: Exception) -> Dict[str, Any]:
        """에이전트 장애 처리"""
        logger.error(f"Agent {agent_id} failed: {error}")
        
        # 에이전트 상태 업데이트
        if agent_id in self.agent_status:
            self.agent_status[agent_id].status = 'error'
        
        # 대체 에이전트 찾기
        failed_agent = self.agents.get(agent_id)
        if failed_agent:
            alternative_agents = [
                agent for agent in self.agents.values()
                if (agent.category == failed_agent.category and 
                    agent.id != agent_id and 
                    agent.enabled and
                    self.agent_status.get(agent.id, {}).get('status') == 'online')
            ]
            
            return {
                'action': 'failover',
                'alternative_agents': [agent.id for agent in alternative_agents],
                'error': str(error)
            }
        
        return {'action': 'none', 'error': str(error)}
    
    # Private methods
    
    async def _select_optimal_agents(self, user_query: str, data_context: Dict) -> List[str]:
        """최적 에이전트 선택 (LLM 기반으로 확장 예정)"""
        # 현재는 기본 로직, 추후 LLM으로 개선
        selected = []
        
        # 데이터 로딩이 필요한 경우
        if 'file_path' in data_context or 'data_source' in data_context:
            selected.append('data_loader')
        
        # 데이터 분석 기본 에이전트
        selected.extend(['eda_tools', 'pandas_agent'])
        
        # 시각화 필요시
        if 'visualization' in user_query.lower() or 'chart' in user_query.lower():
            selected.append('data_visualization')
        
        # 머신러닝 관련
        if any(word in user_query.lower() for word in ['model', 'predict', 'machine learning', 'ml']):
            selected.append('h2o_ml')
        
        # 온라인 상태인 에이전트만 필터링
        online_agents = [agent_id for agent_id in selected 
                        if self.agent_status.get(agent_id, {}).get('status') == 'online']
        
        return online_agents
    
    async def _create_execution_sequence(self, selected_agents: List[str], user_query: str, data_context: Dict) -> List[ExecutionStep]:
        """실행 순서 생성"""
        sequence = []
        
        for i, agent_id in enumerate(selected_agents):
            step = ExecutionStep(
                step_id=f"step_{i+1}",
                agent_name=agent_id,
                task_description=f"Execute {agent_id} for: {user_query}",
                input_data=data_context,
                expected_output="analysis_result",
                dependencies=[],
                timeout=timedelta(seconds=300)
            )
            sequence.append(step)
        
        return sequence
    
    def _estimate_execution_time(self, sequence: List[ExecutionStep]) -> timedelta:
        """실행 시간 예측"""
        total_seconds = sum(step.timeout.total_seconds() for step in sequence)
        return timedelta(seconds=total_seconds * 0.7)  # 70% 예상 시간
    
    async def _execute_step(self, step: ExecutionStep) -> Dict[str, Any]:
        """개별 단계 실행"""
        agent_config = self.agents.get(step.agent_name)
        if not agent_config:
            return {'status': 'failed', 'error': f'Agent {step.agent_name} not found'}
        
        try:
            async with httpx.AsyncClient(timeout=step.timeout.total_seconds()) as client:
                response = await client.post(
                    f"{agent_config.endpoint}/process",
                    json={
                        'query': step.task_description,
                        'data': step.input_data
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'status': 'success',
                        'artifacts': result.get('artifacts', []),
                        'code': result.get('code', []),
                        'output': result.get('output', '')
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': f'HTTP {response.status_code}: {response.text}'
                    }
                    
        except Exception as e:
            logger.error(f"Error executing step with agent {step.agent_name}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _check_agent_health(self, agent_config: AgentConfig) -> bool:
        """에이전트 헬스 체크"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{agent_config.endpoint}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    def _start_health_monitoring(self):
        """헬스 모니터링 시작"""
        for agent_id, agent_config in self.agents.items():
            if agent_config.enabled:
                asyncio.create_task(self._start_agent_health_check(agent_id))
    
    async def _start_agent_health_check(self, agent_id: str):
        """개별 에이전트 헬스 체크 시작"""
        agent_config = self.agents.get(agent_id)
        if not agent_config:
            return
        
        async def health_check_loop():
            while agent_id in self.agents:
                is_healthy = await self._check_agent_health(agent_config)
                
                if agent_id in self.agent_status:
                    self.agent_status[agent_id].status = 'online' if is_healthy else 'offline'
                    self.agent_status[agent_id].last_heartbeat = datetime.now() if is_healthy else self.agent_status[agent_id].last_heartbeat
                
                await asyncio.sleep(agent_config.health_check_interval)
        
        self.health_check_tasks[agent_id] = asyncio.create_task(health_check_loop())
    
    async def _stop_agent_health_check(self, agent_id: str):
        """개별 에이전트 헬스 체크 중지"""
        task = self.health_check_tasks.pop(agent_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass