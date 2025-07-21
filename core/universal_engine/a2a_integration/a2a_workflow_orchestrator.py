"""
A2A Workflow Orchestrator - A2A 워크플로우 조정자

요구사항에 따른 구현:
- 순차 실행 및 병렬 실행 워크플로우 관리
- 에이전트 간 데이터 흐름 및 의존성 처리
- 실시간 진행률 추적 및 상태 업데이트
- 오류 처리 및 복구 메커니즘
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback

from .a2a_agent_discovery import A2AAgentInfo
from .llm_based_agent_selector import AgentSelectionResult
from .a2a_communication_protocol import A2ACommunicationProtocol

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """워크플로우 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """개별 태스크 상태"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """워크플로우 태스크"""
    id: str
    agent_id: str
    agent_info: A2AAgentInfo
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowExecution:
    """워크플로우 실행 정보"""
    id: str
    tasks: List[WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class A2AWorkflowOrchestrator:
    """
    A2A 워크플로우 조정자
    - 에이전트 간 복잡한 워크플로우 관리
    - 병렬/순차 실행 최적화
    - 실시간 진행 상황 추적
    - 오류 처리 및 복구
    """
    
    def __init__(self, communication_protocol: A2ACommunicationProtocol):
        """
        A2AWorkflowOrchestrator 초기화
        
        Args:
            communication_protocol: A2A 통신 프로토콜
        """
        self.communication_protocol = communication_protocol
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_history: List[WorkflowExecution] = []
        self.progress_callbacks: List[Callable] = []
        logger.info("A2AWorkflowOrchestrator initialized")
    
    async def execute_workflow(
        self,
        selection_result: AgentSelectionResult,
        query: str,
        data: Any,
        workflow_id: Optional[str] = None
    ) -> WorkflowExecution:
        """
        워크플로우 실행
        
        Args:
            selection_result: 에이전트 선택 결과
            query: 사용자 쿼리
            data: 처리할 데이터
            workflow_id: 워크플로우 ID (선택사항)
            
        Returns:
            워크플로우 실행 결과
        """
        if workflow_id is None:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        try:
            # 1. 워크플로우 태스크 생성
            tasks = await self._create_workflow_tasks(
                selection_result, query, data
            )
            
            # 2. 워크플로우 실행 객체 생성
            workflow = WorkflowExecution(
                id=workflow_id,
                tasks=tasks,
                total_steps=len(tasks),
                start_time=datetime.now()
            )
            
            self.active_workflows[workflow_id] = workflow
            workflow.status = WorkflowStatus.RUNNING
            
            # 3. 태스크 실행
            await self._execute_tasks(workflow)
            
            # 4. 결과 통합
            workflow.results = await self._integrate_results(workflow)
            
            # 5. 워크플로우 완료
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.progress = 100.0
            
            # 6. 이력에 저장
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            return workflow
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.status = WorkflowStatus.FAILED
                workflow.end_time = datetime.now()
                workflow.errors.append(str(e))
                
                # 실패한 워크플로우도 이력에 저장
                self.workflow_history.append(workflow)
                del self.active_workflows[workflow_id]
                
                return workflow
            raise
    
    async def execute_workflow_with_streaming(
        self,
        selection_result: AgentSelectionResult,
        query: str,
        data: Any,
        workflow_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 방식의 워크플로우 실행
        
        Args:
            selection_result: 에이전트 선택 결과
            query: 사용자 쿼리  
            data: 처리할 데이터
            workflow_id: 워크플로우 ID
            
        Yields:
            워크플로우 진행 상황 업데이트
        """
        if workflow_id is None:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting streaming workflow execution: {workflow_id}")
        
        try:
            # 초기 상태 전송
            yield {
                'type': 'workflow_started',
                'workflow_id': workflow_id,
                'total_agents': len(selection_result.selected_agents),
                'timestamp': datetime.now().isoformat()
            }
            
            # 워크플로우 태스크 생성
            tasks = await self._create_workflow_tasks(
                selection_result, query, data
            )
            
            workflow = WorkflowExecution(
                id=workflow_id,
                tasks=tasks,
                total_steps=len(tasks),
                start_time=datetime.now()
            )
            
            self.active_workflows[workflow_id] = workflow
            workflow.status = WorkflowStatus.RUNNING
            
            yield {
                'type': 'tasks_created',
                'task_count': len(tasks),
                'execution_plan': selection_result.execution_plan,
                'timestamp': datetime.now().isoformat()
            }
            
            # 태스크 실행 (스트리밍)
            async for progress_update in self._execute_tasks_streaming(workflow):
                yield progress_update
            
            # 결과 통합
            yield {
                'type': 'integrating_results',
                'message': '에이전트 결과 통합 중...',
                'timestamp': datetime.now().isoformat()
            }
            
            workflow.results = await self._integrate_results(workflow)
            
            # 완료
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.progress = 100.0
            
            yield {
                'type': 'workflow_completed',
                'workflow_id': workflow_id,
                'results': workflow.results,
                'duration': (workflow.end_time - workflow.start_time).total_seconds(),
                'timestamp': workflow.end_time.isoformat()
            }
            
            # 이력에 저장
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
        except Exception as e:
            logger.error(f"Streaming workflow {workflow_id} failed: {e}")
            
            yield {
                'type': 'workflow_failed',
                'workflow_id': workflow_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.status = WorkflowStatus.FAILED
                workflow.end_time = datetime.now()
                workflow.errors.append(str(e))
                
                self.workflow_history.append(workflow)
                del self.active_workflows[workflow_id]
    
    async def _create_workflow_tasks(
        self,
        selection_result: AgentSelectionResult,
        query: str,
        data: Any
    ) -> List[WorkflowTask]:
        """
        워크플로우 태스크 생성
        """
        tasks = []
        
        # 각 선택된 에이전트에 대해 태스크 생성
        for i, agent in enumerate(selection_result.selected_agents):
            task_id = f"task_{i}_{agent.id}"
            
            # 에이전트별 입력 데이터 준비
            input_data = {
                'query': query,
                'data': data,
                'context': {
                    'workflow_id': f"pending",  # 실행 시 업데이트됨
                    'task_index': i,
                    'total_tasks': len(selection_result.selected_agents)
                }
            }
            
            # 의존성 설정
            dependencies = selection_result.dependencies.get(agent.id, [])
            
            task = WorkflowTask(
                id=task_id,
                agent_id=agent.id,
                agent_info=agent,
                input_data=input_data,
                dependencies=dependencies
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _execute_tasks(self, workflow: WorkflowExecution) -> None:
        """
        태스크 실행 (비스트리밍)
        """
        # 병렬 그룹별로 실행
        for group in self._get_execution_groups(workflow.tasks):
            # 그룹 내 태스크들을 병렬 실행
            group_tasks = [
                self._execute_single_task(workflow, task)
                for task in group
            ]
            
            await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # 진행률 업데이트
            completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
            workflow.progress = (completed_tasks / workflow.total_steps) * 100
            workflow.completed_steps = completed_tasks
    
    async def _execute_tasks_streaming(
        self, 
        workflow: WorkflowExecution
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        태스크 실행 (스트리밍)
        """
        execution_groups = self._get_execution_groups(workflow.tasks)
        
        for group_index, group in enumerate(execution_groups):
            yield {
                'type': 'group_started',
                'group_index': group_index + 1,
                'total_groups': len(execution_groups),
                'group_size': len(group),
                'agents': [task.agent_info.name for task in group],
                'timestamp': datetime.now().isoformat()
            }
            
            # 그룹 내 태스크들을 병렬 실행
            group_tasks = [
                self._execute_single_task_streaming(workflow, task)
                for task in group
            ]
            
            # 병렬 실행 중 진행 상황 스트리밍
            completed = 0
            total = len(group_tasks)
            
            async def track_group_progress():
                nonlocal completed
                while completed < total:
                    await asyncio.sleep(1)  # 1초마다 체크
                    current_completed = sum(
                        1 for task in group 
                        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                    )
                    if current_completed > completed:
                        completed = current_completed
                        yield {
                            'type': 'group_progress',
                            'group_index': group_index + 1,
                            'completed': completed,
                            'total': total,
                            'timestamp': datetime.now().isoformat()
                        }
            
            # 진행 상황 추적과 태스크 실행을 병렬로 진행
            progress_task = asyncio.create_task(
                self._async_generator_to_list(track_group_progress())
            )
            
            # 그룹 태스크 실행
            await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # 진행 상황 추적 종료
            progress_task.cancel()
            
            yield {
                'type': 'group_completed',
                'group_index': group_index + 1,
                'timestamp': datetime.now().isoformat()
            }
            
            # 전체 진행률 업데이트
            completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
            workflow.progress = (completed_tasks / workflow.total_steps) * 100
            workflow.completed_steps = completed_tasks
            
            yield {
                'type': 'overall_progress',
                'progress': workflow.progress,
                'completed_steps': workflow.completed_steps,
                'total_steps': workflow.total_steps,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_single_task(self, workflow: WorkflowExecution, task: WorkflowTask) -> None:
        """
        개별 태스크 실행
        """
        logger.info(f"Executing task {task.id} on agent {task.agent_info.name}")
        
        try:
            # 의존성 확인
            await self._wait_for_dependencies(workflow, task)
            
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.now()
            
            # A2A 통신으로 에이전트 호출
            result = await self.communication_protocol.send_request(
                task.agent_info,
                task.input_data
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            
            # 재시도 로직
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
                await asyncio.sleep(2 ** task.retry_count)  # 지수 백오프
                await self._execute_single_task(workflow, task)
    
    async def _execute_single_task_streaming(
        self, 
        workflow: WorkflowExecution, 
        task: WorkflowTask
    ) -> None:
        """
        개별 태스크 실행 (스트리밍 버전)
        """
        # 기본적으로는 일반 실행과 동일하지만, 
        # 향후 개별 태스크의 스트리밍 결과를 처리할 수 있도록 확장 가능
        await self._execute_single_task(workflow, task)
    
    async def _wait_for_dependencies(self, workflow: WorkflowExecution, task: WorkflowTask) -> None:
        """
        태스크 의존성 대기
        """
        if not task.dependencies:
            return
        
        logger.debug(f"Task {task.id} waiting for dependencies: {task.dependencies}")
        
        while True:
            dependency_tasks = [
                t for t in workflow.tasks 
                if t.agent_id in task.dependencies
            ]
            
            all_completed = all(
                dep_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                for dep_task in dependency_tasks
            )
            
            if all_completed:
                # 의존성 태스크 중 실패한 것이 있는지 확인
                failed_dependencies = [
                    dep_task.agent_id for dep_task in dependency_tasks
                    if dep_task.status == TaskStatus.FAILED
                ]
                
                if failed_dependencies:
                    logger.warning(
                        f"Task {task.id} dependencies failed: {failed_dependencies}"
                    )
                    # 의존성 실패 시 태스크 스킵
                    task.status = TaskStatus.SKIPPED
                    task.error = f"Dependencies failed: {failed_dependencies}"
                    return
                
                break
            
            await asyncio.sleep(0.1)  # 100ms 대기
    
    def _get_execution_groups(self, tasks: List[WorkflowTask]) -> List[List[WorkflowTask]]:
        """
        실행 그룹 생성 (의존성 기반)
        """
        groups = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # 현재 실행 가능한 태스크들 (의존성이 없거나 모두 완료된 태스크들)
            ready_tasks = []
            
            for task in remaining_tasks:
                if not task.dependencies:
                    ready_tasks.append(task)
                else:
                    # 의존성 태스크들이 모두 이전 그룹에서 완료되었는지 확인
                    dependencies_satisfied = all(
                        any(completed_task.agent_id == dep_id 
                            for group in groups 
                            for completed_task in group)
                        for dep_id in task.dependencies
                    )
                    
                    if dependencies_satisfied:
                        ready_tasks.append(task)
            
            if not ready_tasks:
                # 순환 의존성이나 잘못된 의존성 설정
                logger.warning("No ready tasks found - possible circular dependencies")
                ready_tasks = remaining_tasks  # 강제로 실행
            
            groups.append(ready_tasks)
            
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return groups
    
    async def _integrate_results(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """
        워크플로우 결과 통합
        """
        integrated_results = {
            'workflow_id': workflow.id,
            'execution_summary': {
                'total_tasks': len(workflow.tasks),
                'completed_tasks': sum(1 for t in workflow.tasks if t.status == TaskStatus.COMPLETED),
                'failed_tasks': sum(1 for t in workflow.tasks if t.status == TaskStatus.FAILED),
                'skipped_tasks': sum(1 for t in workflow.tasks if t.status == TaskStatus.SKIPPED),
                'duration': (workflow.end_time - workflow.start_time).total_seconds() if workflow.end_time else 0
            },
            'agent_results': {},
            'consolidated_insights': {},
            'recommendations': []
        }
        
        # 각 에이전트 결과 수집
        for task in workflow.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                integrated_results['agent_results'][task.agent_info.name] = {
                    'agent_id': task.agent_id,
                    'execution_time': (task.end_time - task.start_time).total_seconds(),
                    'result': task.result
                }
        
        return integrated_results
    
    async def _async_generator_to_list(self, async_gen) -> List:
        """AsyncGenerator를 리스트로 변환하는 헬퍼 메서드"""
        result = []
        try:
            async for item in async_gen:
                result.append(item)
        except asyncio.CancelledError:
            pass
        return result
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """
        워크플로우 상태 조회
        
        Args:
            workflow_id: 워크플로우 ID
            
        Returns:
            워크플로우 실행 정보
        """
        return self.active_workflows.get(workflow_id)
    
    def get_workflow_statistics(self) -> Dict:
        """
        워크플로우 통계 조회
        
        Returns:
            워크플로우 통계 정보
        """
        total_workflows = len(self.workflow_history)
        if total_workflows == 0:
            return {'message': 'No workflow history available'}
        
        completed_workflows = sum(
            1 for w in self.workflow_history 
            if w.status == WorkflowStatus.COMPLETED
        )
        
        avg_duration = sum(
            (w.end_time - w.start_time).total_seconds()
            for w in self.workflow_history
            if w.end_time and w.start_time
        ) / total_workflows if total_workflows > 0 else 0
        
        return {
            'total_workflows': total_workflows,
            'active_workflows': len(self.active_workflows),
            'success_rate': completed_workflows / total_workflows,
            'average_duration': avg_duration,
            'most_used_agents': self._get_most_used_agents()
        }
    
    def _get_most_used_agents(self) -> List[tuple]:
        """가장 많이 사용된 에이전트 조회"""
        agent_usage = {}
        
        for workflow in self.workflow_history:
            for task in workflow.tasks:
                agent_name = task.agent_info.name
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
        
        return sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    
    async def execute_agent_workflow(
        self,
        agents: List[A2AAgentInfo],
        query: str,
        data: Any,
        execution_mode: str = "sequential"
    ) -> Dict[str, Any]:
        """
        에이전트 워크플로우 실행
        
        Args:
            agents: 실행할 에이전트 목록
            query: 사용자 쿼리
            data: 처리할 데이터
            execution_mode: 실행 모드 ("sequential" 또는 "parallel")
            
        Returns:
            워크플로우 실행 결과
        """
        workflow_id = f"agent_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Executing agent workflow: {workflow_id} with {len(agents)} agents")
        
        try:
            # 워크플로우 태스크 생성
            tasks = []
            for i, agent in enumerate(agents):
                task = WorkflowTask(
                    id=f"task_{i}_{agent.id}",
                    agent_id=agent.id,
                    agent_info=agent,
                    input_data={
                        'query': query,
                        'data': data,
                        'context': {
                            'workflow_id': workflow_id,
                            'task_index': i,
                            'execution_mode': execution_mode
                        }
                    },
                    dependencies=[] if execution_mode == "parallel" else ([f"task_{i-1}_{agents[i-1].id}"] if i > 0 else [])
                )
                tasks.append(task)
            
            # 워크플로우 실행
            workflow = WorkflowExecution(
                id=workflow_id,
                tasks=tasks,
                total_steps=len(tasks),
                start_time=datetime.now()
            )
            
            self.active_workflows[workflow_id] = workflow
            workflow.status = WorkflowStatus.RUNNING
            
            # 태스크 실행
            await self._execute_tasks(workflow)
            
            # 결과 통합
            workflow.results = await self._integrate_results(workflow)
            
            # 완료 처리
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.progress = 100.0
            
            # 이력 저장
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            logger.info(f"Agent workflow {workflow_id} completed successfully")
            return workflow.results
            
        except Exception as e:
            logger.error(f"Agent workflow {workflow_id} failed: {e}")
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.status = WorkflowStatus.FAILED
                workflow.end_time = datetime.now()
                workflow.errors.append(str(e))
                self.workflow_history.append(workflow)
                del self.active_workflows[workflow_id]
            raise
    
    async def coordinate_agents(
        self,
        primary_agent: A2AAgentInfo,
        supporting_agents: List[A2AAgentInfo],
        query: str,
        data: Any
    ) -> Dict[str, Any]:
        """
        다중 에이전트 협업 조율
        
        Args:
            primary_agent: 주 에이전트
            supporting_agents: 지원 에이전트들
            query: 사용자 쿼리
            data: 처리할 데이터
            
        Returns:
            협업 결과
        """
        coordination_id = f"coordination_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Coordinating agents: primary={primary_agent.name}, supporting={[a.name for a in supporting_agents]}")
        
        try:
            # 1단계: 지원 에이전트들 병렬 실행
            supporting_tasks = []
            for i, agent in enumerate(supporting_agents):
                task = WorkflowTask(
                    id=f"support_{i}_{agent.id}",
                    agent_id=agent.id,
                    agent_info=agent,
                    input_data={
                        'query': query,
                        'data': data,
                        'context': {
                            'coordination_id': coordination_id,
                            'role': 'supporting',
                            'primary_agent': primary_agent.name
                        }
                    }
                )
                supporting_tasks.append(task)
            
            # 지원 에이전트 실행
            support_results = {}
            for task in supporting_tasks:
                try:
                    task.status = TaskStatus.RUNNING
                    task.start_time = datetime.now()
                    
                    result = await self.communication_protocol.send_request(
                        task.agent_info,
                        task.input_data
                    )
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.end_time = datetime.now()
                    
                    support_results[task.agent_info.name] = result
                    
                except Exception as e:
                    logger.error(f"Supporting agent {task.agent_info.name} failed: {e}")
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    task.end_time = datetime.now()
            
            # 2단계: 주 에이전트 실행 (지원 결과 포함)
            primary_task = WorkflowTask(
                id=f"primary_{primary_agent.id}",
                agent_id=primary_agent.id,
                agent_info=primary_agent,
                input_data={
                    'query': query,
                    'data': data,
                    'supporting_results': support_results,
                    'context': {
                        'coordination_id': coordination_id,
                        'role': 'primary',
                        'supporting_agents': [a.name for a in supporting_agents]
                    }
                }
            )
            
            primary_task.status = TaskStatus.RUNNING
            primary_task.start_time = datetime.now()
            
            primary_result = await self.communication_protocol.send_request(
                primary_agent,
                primary_task.input_data
            )
            
            primary_task.result = primary_result
            primary_task.status = TaskStatus.COMPLETED
            primary_task.end_time = datetime.now()
            
            # 결과 통합
            coordination_result = {
                'coordination_id': coordination_id,
                'primary_result': primary_result,
                'supporting_results': support_results,
                'coordination_summary': {
                    'primary_agent': primary_agent.name,
                    'supporting_agents': [a.name for a in supporting_agents],
                    'success_rate': (
                        len([r for r in support_results.values() if r]) + (1 if primary_result else 0)
                    ) / (len(supporting_agents) + 1),
                    'execution_time': (datetime.now() - primary_task.start_time).total_seconds()
                }
            }
            
            logger.info(f"Agent coordination {coordination_id} completed successfully")
            return coordination_result
            
        except Exception as e:
            logger.error(f"Agent coordination {coordination_id} failed: {e}")
            raise
    
    async def manage_dependencies(
        self,
        workflow_tasks: List[WorkflowTask]
    ) -> Dict[str, List[str]]:
        """
        워크플로우 의존성 관리
        
        Args:
            workflow_tasks: 워크플로우 태스크 목록
            
        Returns:
            의존성 맵 {task_id: [dependency_task_ids]}
        """
        logger.info(f"Managing dependencies for {len(workflow_tasks)} tasks")
        
        dependency_map = {}
        
        # 각 태스크의 의존성 분석
        for task in workflow_tasks:
            task_dependencies = []
            
            # 에이전트 타입 기반 의존성 추론
            if hasattr(task.agent_info, 'capabilities'):
                capabilities = task.agent_info.capabilities
                
                # 데이터 로딩이 필요한 에이전트는 데이터 로더에 의존
                if any(cap in ['analysis', 'visualization', 'modeling'] for cap in capabilities):
                    data_loader_tasks = [
                        t for t in workflow_tasks 
                        if t != task and 'data_loading' in getattr(t.agent_info, 'capabilities', [])
                    ]
                    task_dependencies.extend([t.id for t in data_loader_tasks])
                
                # 시각화는 분석 결과에 의존
                if 'visualization' in capabilities:
                    analysis_tasks = [
                        t for t in workflow_tasks 
                        if t != task and any(cap in ['analysis', 'eda', 'statistics'] 
                                           for cap in getattr(t.agent_info, 'capabilities', []))
                    ]
                    task_dependencies.extend([t.id for t in analysis_tasks])
                
                # 모델링은 특성 엔지니어링에 의존
                if 'modeling' in capabilities:
                    feature_tasks = [
                        t for t in workflow_tasks 
                        if t != task and 'feature_engineering' in getattr(t.agent_info, 'capabilities', [])
                    ]
                    task_dependencies.extend([t.id for t in feature_tasks])
            
            # 명시적 의존성 추가
            task_dependencies.extend(task.dependencies)
            
            # 중복 제거
            dependency_map[task.id] = list(set(task_dependencies))
            
            # 태스크 객체 업데이트
            task.dependencies = dependency_map[task.id]
        
        # 순환 의존성 검사
        self._validate_dependencies(dependency_map)
        
        logger.info(f"Dependency management completed: {len(dependency_map)} tasks with dependencies")
        return dependency_map
    
    def _validate_dependencies(self, dependency_map: Dict[str, List[str]]) -> None:
        """
        의존성 순환 검사
        
        Args:
            dependency_map: 의존성 맵
            
        Raises:
            ValueError: 순환 의존성이 발견된 경우
        """
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_map.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for task_id in dependency_map:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    raise ValueError(f"Circular dependency detected involving task: {task_id}")
        
        logger.debug("Dependency validation passed - no circular dependencies found")