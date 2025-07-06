"""
A2A Agent Execution Orchestrator

이 모듈은 DomainAwareAgentSelector에서 선택된 에이전트들을 실제로 실행하는 
오케스트레이션 엔진입니다.

주요 기능:
- A2A 프로토콜 기반 에이전트 실행
- 순차적/병렬 실행 관리
- 실행 상태 모니터링
- 오류 처리 및 복구
- 실행 결과 수집 및 관리
"""

import asyncio
import httpx
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from core.llm_factory import create_llm_instance
from .domain_aware_agent_selector import AgentSelectionResult, AgentSelection, AgentType

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionStrategy(Enum):
    """실행 전략"""
    SEQUENTIAL = "sequential"  # 순차 실행
    PARALLEL = "parallel"      # 병렬 실행
    PIPELINE = "pipeline"      # 파이프라인 실행 (출력이 다음 입력)


@dataclass
class A2AAgentConfig:
    """A2A 에이전트 설정"""
    agent_type: AgentType
    name: str
    port: int
    url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class ExecutionTask:
    """실행 태스크"""
    task_id: str
    agent_config: A2AAgentConfig
    instruction: str
    dependencies: List[str]
    context: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class ExecutionPlan:
    """실행 계획"""
    plan_id: str
    objective: str
    strategy: ExecutionStrategy
    tasks: List[ExecutionTask]
    context: Dict[str, Any]
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    overall_status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_execution_time: Optional[float] = None


@dataclass
class ExecutionResult:
    """실행 결과"""
    plan_id: str
    objective: str
    overall_status: ExecutionStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time: float
    task_results: List[Dict[str, Any]]
    aggregated_results: Dict[str, Any]
    execution_summary: str
    confidence_score: float


class A2AAgentExecutionOrchestrator:
    """A2A 에이전트 실행 오케스트레이터"""
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.agent_configs = self._initialize_agent_configs()
        self.client_timeout = httpx.Timeout(30.0, connect=10.0)
    
    def _initialize_agent_configs(self) -> Dict[AgentType, A2AAgentConfig]:
        """A2A 에이전트 설정 초기화"""
        configs = {}
        
        # A2A 에이전트 포트 매핑
        agent_ports = {
            AgentType.DATA_LOADER: 8310,
            AgentType.DATA_CLEANING: 8311,
            AgentType.DATA_WRANGLING: 8312,
            AgentType.EDA_TOOLS: 8313,
            AgentType.DATA_VISUALIZATION: 8314,
            AgentType.FEATURE_ENGINEERING: 8315,
            AgentType.H2O_ML: 8316,
            AgentType.MLFLOW_TOOLS: 8317,
            AgentType.SQL_DATABASE: 8318
        }
        
        for agent_type, port in agent_ports.items():
            configs[agent_type] = A2AAgentConfig(
                agent_type=agent_type,
                name=f"AI_DS_Team {agent_type.value}",
                port=port,
                url=f"http://localhost:{port}",
                timeout=30.0,
                max_retries=3,
                retry_delay=1.0
            )
        
        return configs
    
    async def create_execution_plan(
        self,
        agent_selection_result: AgentSelectionResult,
        enhanced_query: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        에이전트 선택 결과로부터 실행 계획 생성
        
        Args:
            agent_selection_result: 에이전트 선택 결과
            enhanced_query: 향상된 쿼리
            context: 실행 컨텍스트
            
        Returns:
            ExecutionPlan: 실행 계획
        """
        plan_id = f"plan_{int(time.time())}"
        
        # 실행 태스크 생성
        tasks = []
        for i, selection in enumerate(agent_selection_result.selected_agents):
            task_id = f"task_{plan_id}_{i}"
            
            agent_config = self.agent_configs.get(selection.agent_type)
            if not agent_config:
                logger.warning(f"Agent config not found for {selection.agent_type}")
                continue
            
            # 에이전트별 맞춤 명령어 생성
            instruction = await self._generate_agent_instruction(
                selection, enhanced_query, context
            )
            
            task = ExecutionTask(
                task_id=task_id,
                agent_config=agent_config,
                instruction=instruction,
                dependencies=selection.dependencies,
                context=context
            )
            tasks.append(task)
        
        # 실행 전략 결정
        strategy = self._determine_execution_strategy(agent_selection_result)
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            objective=f"Execute {len(tasks)} selected agents for comprehensive analysis",
            strategy=strategy,
            tasks=tasks,
            context=context,
            total_tasks=len(tasks)
        )
        
        self.active_plans[plan_id] = plan
        return plan
    
    async def _generate_agent_instruction(
        self,
        selection: AgentSelection,
        enhanced_query: str,
        context: Dict[str, Any]
    ) -> str:
        """에이전트별 맞춤 명령어 생성"""
        
        prompt = f"""다음 정보를 바탕으로 {selection.agent_type.value} 에이전트에게 전달할 구체적인 명령어를 생성해주세요:

**원본 쿼리**: {enhanced_query}

**에이전트 정보**:
- 타입: {selection.agent_type.value}
- 신뢰도: {selection.confidence:.2f}
- 선택 이유: {selection.reasoning}
- 우선순위: {selection.priority}
- 예상 출력: {', '.join(selection.expected_outputs)}

**실행 컨텍스트**:
{json.dumps(context, ensure_ascii=False, indent=2)}

**명령어 생성 원칙**:
1. 해당 에이전트의 전문 분야에 최적화
2. 구체적이고 실행 가능한 지시사항
3. 필요한 입력 데이터 및 출력 형식 명시
4. 에러 처리 및 대안 제시

명령어를 생성해주세요:"""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate instruction: {e}")
            return f"다음 작업을 수행해주세요: {enhanced_query}"
    
    def _determine_execution_strategy(self, agent_selection_result: AgentSelectionResult) -> ExecutionStrategy:
        """실행 전략 결정"""
        if agent_selection_result.selection_strategy == "pipeline":
            return ExecutionStrategy.PIPELINE
        elif agent_selection_result.selection_strategy == "parallel":
            return ExecutionStrategy.PARALLEL
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[callable] = None
    ) -> ExecutionResult:
        """
        실행 계획 실행
        
        Args:
            plan: 실행 계획
            progress_callback: 진행 상황 콜백
            
        Returns:
            ExecutionResult: 실행 결과
        """
        logger.info(f"🚀 Starting execution plan: {plan.plan_id}")
        
        plan.start_time = time.time()
        plan.overall_status = ExecutionStatus.RUNNING
        
        try:
            if plan.strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(plan, progress_callback)
            elif plan.strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(plan, progress_callback)
            elif plan.strategy == ExecutionStrategy.PIPELINE:
                await self._execute_pipeline(plan, progress_callback)
            
            # 실행 완료 처리
            plan.end_time = time.time()
            plan.total_execution_time = plan.end_time - plan.start_time
            
            # 전체 상태 결정
            if plan.failed_tasks > 0:
                plan.overall_status = ExecutionStatus.FAILED
            else:
                plan.overall_status = ExecutionStatus.COMPLETED
            
            # 결과 생성
            result = await self._create_execution_result(plan)
            
            logger.info(f"✅ Execution plan completed: {plan.plan_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Execution plan failed: {plan.plan_id} - {e}")
            plan.overall_status = ExecutionStatus.FAILED
            plan.end_time = time.time()
            plan.total_execution_time = plan.end_time - plan.start_time
            
            # 오류 결과 생성
            result = await self._create_execution_result(plan)
            return result
    
    async def _execute_sequential(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[callable] = None
    ):
        """순차 실행"""
        logger.info(f"📋 Sequential execution: {len(plan.tasks)} tasks")
        
        for i, task in enumerate(plan.tasks):
            if progress_callback:
                progress_callback(f"실행 중: {task.agent_config.name} ({i+1}/{len(plan.tasks)})")
            
            await self._execute_task(task)
            
            if task.status == ExecutionStatus.COMPLETED:
                plan.completed_tasks += 1
            elif task.status == ExecutionStatus.FAILED:
                plan.failed_tasks += 1
                # 순차 실행에서는 실패시 중단할지 결정
                if self._should_stop_on_failure(plan, task):
                    break
    
    async def _execute_parallel(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[callable] = None
    ):
        """병렬 실행"""
        logger.info(f"⚡ Parallel execution: {len(plan.tasks)} tasks")
        
        if progress_callback:
            progress_callback(f"병렬 실행 시작: {len(plan.tasks)}개 태스크")
        
        # 의존성이 없는 태스크들을 병렬로 실행
        independent_tasks = [task for task in plan.tasks if not task.dependencies]
        
        if independent_tasks:
            await asyncio.gather(*[self._execute_task(task) for task in independent_tasks])
        
        # 의존성이 있는 태스크들 처리
        dependent_tasks = [task for task in plan.tasks if task.dependencies]
        for task in dependent_tasks:
            await self._execute_task(task)
        
        # 결과 집계
        for task in plan.tasks:
            if task.status == ExecutionStatus.COMPLETED:
                plan.completed_tasks += 1
            elif task.status == ExecutionStatus.FAILED:
                plan.failed_tasks += 1
    
    async def _execute_pipeline(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[callable] = None
    ):
        """파이프라인 실행"""
        logger.info(f"🔄 Pipeline execution: {len(plan.tasks)} tasks")
        
        pipeline_context = {}
        
        for i, task in enumerate(plan.tasks):
            if progress_callback:
                progress_callback(f"파이프라인 실행: {task.agent_config.name} ({i+1}/{len(plan.tasks)})")
            
            # 이전 태스크의 결과를 현재 태스크 컨텍스트에 추가
            if i > 0:
                prev_task = plan.tasks[i-1]
                if prev_task.result:
                    pipeline_context[f"prev_result_{i-1}"] = prev_task.result
                    task.context.update(pipeline_context)
            
            await self._execute_task(task)
            
            if task.status == ExecutionStatus.COMPLETED:
                plan.completed_tasks += 1
            elif task.status == ExecutionStatus.FAILED:
                plan.failed_tasks += 1
                # 파이프라인에서는 실패시 중단
                break
    
    async def _execute_task(self, task: ExecutionTask):
        """개별 태스크 실행"""
        task.start_time = time.time()
        task.status = ExecutionStatus.RUNNING
        
        logger.info(f"🔧 Executing task: {task.task_id} - {task.agent_config.name}")
        
        for attempt in range(task.agent_config.max_retries + 1):
            try:
                # A2A 에이전트 호출
                result = await self._call_a2a_agent(task)
                
                if result:
                    task.result = result
                    task.status = ExecutionStatus.COMPLETED
                    task.end_time = time.time()
                    task.execution_time = task.end_time - task.start_time
                    
                    logger.info(f"✅ Task completed: {task.task_id} ({task.execution_time:.2f}s)")
                    return
                else:
                    raise Exception("No result received from agent")
                    
            except Exception as e:
                task.retry_count = attempt
                error_msg = f"Task execution failed (attempt {attempt + 1}): {e}"
                logger.warning(error_msg)
                
                if attempt < task.agent_config.max_retries:
                    await asyncio.sleep(task.agent_config.retry_delay * (attempt + 1))
                else:
                    task.error = error_msg
                    task.status = ExecutionStatus.FAILED
                    task.end_time = time.time()
                    task.execution_time = task.end_time - task.start_time
                    
                    logger.error(f"❌ Task failed: {task.task_id} - {error_msg}")
                    return
    
    async def _call_a2a_agent(self, task: ExecutionTask) -> Optional[Dict[str, Any]]:
        """A2A 에이전트 호출"""
        try:
            # A2A 메시지 구성
            message_payload = {
                "jsonrpc": "2.0",
                "id": f"orchestrator-{task.task_id}",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": task.instruction
                            }
                        ],
                        "messageId": f"msg-{task.task_id}"
                    },
                    "metadata": task.context
                }
            }
            
            # HTTP 요청 전송
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.post(
                    task.agent_config.url,
                    json=message_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "agent_name": task.agent_config.name,
                        "agent_type": task.agent_config.agent_type.value,
                        "result": result.get("result", {}),
                        "response": result,
                        "execution_time": task.execution_time
                    }
                else:
                    logger.error(f"A2A agent call failed: HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"A2A agent call exception: {e}")
            return None
    
    def _should_stop_on_failure(self, plan: ExecutionPlan, failed_task: ExecutionTask) -> bool:
        """실패시 중단 여부 결정"""
        # 중요한 태스크나 의존성이 있는 태스크 실패시 중단
        if failed_task.dependencies or plan.failed_tasks > len(plan.tasks) // 2:
            return True
        return False
    
    async def _create_execution_result(self, plan: ExecutionPlan) -> ExecutionResult:
        """실행 결과 생성"""
        
        # 태스크 결과 수집
        task_results = []
        for task in plan.tasks:
            task_result = {
                "task_id": task.task_id,
                "agent_name": task.agent_config.name,
                "agent_type": task.agent_config.agent_type.value,
                "status": task.status.value,
                "execution_time": task.execution_time,
                "result": task.result,
                "error": task.error
            }
            task_results.append(task_result)
        
        # 결과 집계
        aggregated_results = await self._aggregate_results(plan)
        
        # 실행 요약 생성
        summary = await self._generate_execution_summary(plan, aggregated_results)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(plan)
        
        return ExecutionResult(
            plan_id=plan.plan_id,
            objective=plan.objective,
            overall_status=plan.overall_status,
            total_tasks=plan.total_tasks,
            completed_tasks=plan.completed_tasks,
            failed_tasks=plan.failed_tasks,
            execution_time=plan.total_execution_time or 0,
            task_results=task_results,
            aggregated_results=aggregated_results,
            execution_summary=summary,
            confidence_score=confidence
        )
    
    async def _aggregate_results(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """결과 집계"""
        
        # 성공한 태스크들의 결과 수집
        successful_results = []
        for task in plan.tasks:
            if task.status == ExecutionStatus.COMPLETED and task.result:
                successful_results.append(task.result)
        
        # 결과 집계 LLM 분석
        if successful_results:
            aggregation_prompt = f"""다음 A2A 에이전트 실행 결과들을 종합 분석해주세요:

**실행 목표**: {plan.objective}

**에이전트 실행 결과들**:
{json.dumps(successful_results, ensure_ascii=False, indent=2)}

**집계 분석 요구사항**:
1. 각 에이전트별 주요 결과 요약
2. 전체 목표 달성도 평가
3. 발견된 핵심 인사이트
4. 결과 간 연관성 분석
5. 실행 품질 및 완성도 평가

JSON 형식으로 종합 분석 결과를 제공해주세요."""

            try:
                response = await self.llm.ainvoke(aggregation_prompt)
                
                # JSON 추출 시도
                content = response.content.strip()
                if content.startswith('{') and content.endswith('}'):
                    return json.loads(content)
                else:
                    # JSON 블록 추출
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    else:
                        return {"analysis": content, "raw_results": successful_results}
                        
            except Exception as e:
                logger.warning(f"Result aggregation failed: {e}")
                return {"raw_results": successful_results}
        
        return {"message": "No successful results to aggregate"}
    
    async def _generate_execution_summary(self, plan: ExecutionPlan, aggregated_results: Dict[str, Any]) -> str:
        """실행 요약 생성"""
        
        summary_prompt = f"""다음 A2A 에이전트 오케스트레이션 실행 결과를 요약해주세요:

**실행 계획**: {plan.objective}
**전체 태스크**: {plan.total_tasks}개
**완료된 태스크**: {plan.completed_tasks}개
**실패한 태스크**: {plan.failed_tasks}개
**실행 시간**: {plan.total_execution_time:.2f}초
**실행 전략**: {plan.strategy.value}

**집계된 결과**:
{json.dumps(aggregated_results, ensure_ascii=False, indent=2)}

**요약 요구사항**:
1. 실행 성공률 및 효율성 평가
2. 주요 성과 및 제한사항
3. 목표 달성도 분석
4. 개선 방안 제시

간결하고 명확한 한국어 요약을 작성해주세요."""

        try:
            response = await self.llm.ainvoke(summary_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"실행 완료: {plan.completed_tasks}/{plan.total_tasks} 태스크 성공"
    
    def _calculate_confidence(self, plan: ExecutionPlan) -> float:
        """신뢰도 계산"""
        if plan.total_tasks == 0:
            return 0.0
        
        # 기본 성공률
        success_rate = plan.completed_tasks / plan.total_tasks
        
        # 실행 시간 페널티 (너무 빠르거나 느리면 신뢰도 하락)
        time_penalty = 0.0
        if plan.total_execution_time:
            expected_time = plan.total_tasks * 10  # 태스크당 10초 예상
            time_ratio = plan.total_execution_time / expected_time
            if time_ratio < 0.1 or time_ratio > 5.0:  # 너무 빠르거나 느림
                time_penalty = 0.1
        
        # 전략별 보정
        strategy_bonus = 0.0
        if plan.strategy == ExecutionStrategy.PIPELINE and plan.failed_tasks == 0:
            strategy_bonus = 0.1
        
        confidence = success_rate - time_penalty + strategy_bonus
        return max(0.0, min(1.0, confidence))
    
    async def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """실행 상태 조회"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return None
        
        return {
            "plan_id": plan_id,
            "objective": plan.objective,
            "status": plan.overall_status.value,
            "total_tasks": plan.total_tasks,
            "completed_tasks": plan.completed_tasks,
            "failed_tasks": plan.failed_tasks,
            "execution_time": plan.total_execution_time,
            "task_statuses": [
                {
                    "task_id": task.task_id,
                    "agent_name": task.agent_config.name,
                    "status": task.status.value,
                    "execution_time": task.execution_time
                }
                for task in plan.tasks
            ]
        }
    
    async def cancel_execution(self, plan_id: str) -> bool:
        """실행 취소"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return False
        
        plan.overall_status = ExecutionStatus.CANCELLED
        
        # 실행 중인 태스크들 취소
        for task in plan.tasks:
            if task.status == ExecutionStatus.RUNNING:
                task.status = ExecutionStatus.CANCELLED
        
        logger.info(f"🛑 Execution cancelled: {plan_id}")
        return True 