"""
Execution Plan Manager

이 모듈은 A2A 에이전트 실행 계획의 전체 생명주기를 관리합니다.

주요 기능:
- 실행 계획 생성 및 검증
- 실시간 실행 모니터링
- 계획 최적화 및 재조정
- 성능 분석 및 메트릭 수집
- 실행 이력 관리
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from core.llm_factory import create_llm_instance
from .domain_aware_agent_selector import AgentSelectionResult
from .a2a_agent_execution_orchestrator import (
    A2AAgentExecutionOrchestrator, 
    ExecutionPlan,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStrategy
)
from .multi_agent_result_integration import (
    MultiAgentResultIntegrator,
    IntegrationResult,
    IntegrationStrategy
)

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """계획 상태"""
    DRAFT = "draft"
    VALIDATED = "validated"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OPTIMIZED = "optimized"


class MonitoringLevel(Enum):
    """모니터링 레벨"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class OptimizationStrategy(Enum):
    """최적화 전략"""
    PERFORMANCE = "performance"    # 성능 최적화
    RELIABILITY = "reliability"   # 신뢰성 최적화
    COST = "cost"                 # 비용 최적화
    BALANCED = "balanced"         # 균형 최적화


@dataclass
class PlanMetrics:
    """계획 메트릭"""
    total_execution_time: float
    average_agent_time: float
    success_rate: float
    error_rate: float
    throughput: float
    resource_utilization: float
    cost_efficiency: float
    quality_score: float


@dataclass
class MonitoringEvent:
    """모니터링 이벤트"""
    timestamp: datetime
    event_type: str
    plan_id: str
    agent_name: Optional[str]
    message: str
    severity: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """최적화 권고사항"""
    optimization_type: str
    description: str
    expected_improvement: float
    implementation_effort: str
    priority: int
    estimated_impact: Dict[str, float] = field(default_factory=dict)


@dataclass
class ManagedExecutionPlan:
    """관리되는 실행 계획"""
    plan_id: str
    original_selection: AgentSelectionResult
    execution_plan: ExecutionPlan
    status: PlanStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_result: Optional[ExecutionResult] = None
    integration_result: Optional[IntegrationResult] = None
    metrics: Optional[PlanMetrics] = None
    monitoring_events: List[MonitoringEvent] = field(default_factory=list)
    optimization_recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionPlanManager:
    """실행 계획 관리자"""
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.orchestrator = A2AAgentExecutionOrchestrator()
        self.integrator = MultiAgentResultIntegrator()
        
        # 관리되는 계획들
        self.managed_plans: Dict[str, ManagedExecutionPlan] = {}
        self.active_monitors: Dict[str, asyncio.Task] = {}
        
        # 설정
        self.monitoring_level = MonitoringLevel.STANDARD
        self.optimization_enabled = True
        self.max_history_size = 100
        
        logger.info("ExecutionPlanManager initialized")
    
    async def create_managed_plan(
        self,
        agent_selection: AgentSelectionResult,
        enhanced_query: str,
        context: Optional[Dict[str, Any]] = None,
        monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    ) -> ManagedExecutionPlan:
        """
        관리되는 실행 계획 생성
        
        Args:
            agent_selection: 에이전트 선택 결과
            enhanced_query: 향상된 쿼리
            context: 실행 컨텍스트
            monitoring_level: 모니터링 레벨
            
        Returns:
            ManagedExecutionPlan: 관리되는 실행 계획
        """
        # 실행 계획 생성
        execution_plan = await self.orchestrator.create_execution_plan(
            agent_selection, enhanced_query, context or {}
        )
        
        # 계획 검증
        validation_result = await self._validate_plan(execution_plan, agent_selection)
        
        if not validation_result["valid"]:
            logger.warning(f"Plan validation failed: {validation_result['issues']}")
        
        # 관리되는 계획 생성
        managed_plan = ManagedExecutionPlan(
            plan_id=execution_plan.plan_id,
            original_selection=agent_selection,
            execution_plan=execution_plan,
            status=PlanStatus.VALIDATED if validation_result["valid"] else PlanStatus.DRAFT,
            created_at=datetime.now(),
            metadata={
                "enhanced_query": enhanced_query,
                "context": context or {},
                "monitoring_level": monitoring_level.value,
                "validation_result": validation_result
            }
        )
        
        self.managed_plans[execution_plan.plan_id] = managed_plan
        
        # 초기 모니터링 이벤트
        await self._add_monitoring_event(
            managed_plan,
            "plan_created",
            f"Execution plan created with {len(execution_plan.tasks)} tasks",
            "info"
        )
        
        logger.info(f"📋 Created managed execution plan: {execution_plan.plan_id}")
        return managed_plan
    
    async def execute_managed_plan(
        self,
        plan_id: str,
        monitoring_callback: Optional[Callable] = None
    ) -> IntegrationResult:
        """
        관리되는 계획 실행
        
        Args:
            plan_id: 계획 ID
            monitoring_callback: 모니터링 콜백
            
        Returns:
            IntegrationResult: 통합 결과
        """
        managed_plan = self.managed_plans.get(plan_id)
        if not managed_plan:
            raise ValueError(f"Plan not found: {plan_id}")
        
        if managed_plan.status not in [PlanStatus.VALIDATED, PlanStatus.SCHEDULED]:
            raise ValueError(f"Plan not ready for execution: {managed_plan.status}")
        
        try:
            # 실행 시작
            managed_plan.status = PlanStatus.EXECUTING
            managed_plan.started_at = datetime.now()
            
            await self._add_monitoring_event(
                managed_plan,
                "execution_started",
                f"Plan execution started",
                "info"
            )
            
            # 모니터링 시작
            if self.monitoring_level != MonitoringLevel.MINIMAL:
                monitor_task = asyncio.create_task(
                    self._monitor_execution(managed_plan, monitoring_callback)
                )
                self.active_monitors[plan_id] = monitor_task
            
            # 실행 진행 콜백
            def progress_callback(message: str):
                asyncio.create_task(self._add_monitoring_event(
                    managed_plan, "execution_progress", message, "info"
                ))
                if monitoring_callback:
                    monitoring_callback(message)
            
            # 에이전트 실행
            execution_result = await self.orchestrator.execute_plan(
                managed_plan.execution_plan,
                progress_callback
            )
            
            managed_plan.execution_result = execution_result
            
            # 결과 통합
            integration_result = await self.integrator.integrate_results(
                execution_result,
                IntegrationStrategy.HIERARCHICAL,
                managed_plan.metadata.get("context", {})
            )
            
            managed_plan.integration_result = integration_result
            
            # 실행 완료
            managed_plan.completed_at = datetime.now()
            managed_plan.status = PlanStatus.COMPLETED if execution_result.overall_status == ExecutionStatus.COMPLETED else PlanStatus.FAILED
            
            # 메트릭 계산
            managed_plan.metrics = await self._calculate_plan_metrics(managed_plan)
            
            # 최적화 권고 생성
            if self.optimization_enabled:
                managed_plan.optimization_recommendations = await self._generate_optimization_recommendations(managed_plan)
            
            # 모니터링 종료
            if plan_id in self.active_monitors:
                self.active_monitors[plan_id].cancel()
                del self.active_monitors[plan_id]
            
            await self._add_monitoring_event(
                managed_plan,
                "execution_completed",
                f"Plan execution completed with status: {managed_plan.status.value}",
                "info"
            )
            
            logger.info(f"✅ Completed managed execution: {plan_id}")
            return integration_result
            
        except Exception as e:
            managed_plan.status = PlanStatus.FAILED
            managed_plan.completed_at = datetime.now()
            
            await self._add_monitoring_event(
                managed_plan,
                "execution_failed",
                f"Plan execution failed: {str(e)}",
                "error",
                {"error": str(e)}
            )
            
            # 모니터링 종료
            if plan_id in self.active_monitors:
                self.active_monitors[plan_id].cancel()
                del self.active_monitors[plan_id]
            
            logger.error(f"❌ Failed managed execution: {plan_id} - {e}")
            raise
    
    async def _validate_plan(
        self,
        execution_plan: ExecutionPlan,
        agent_selection: AgentSelectionResult
    ) -> Dict[str, Any]:
        """계획 검증"""
        
        validation_prompt = f"""다음 실행 계획을 검증해주세요:

**원본 에이전트 선택**:
- 선택된 에이전트 수: {len(agent_selection.selected_agents)}
- 실행 전략: {agent_selection.selection_strategy}
- 전체 신뢰도: {agent_selection.total_confidence:.2f}

**실행 계획**:
- 계획 ID: {execution_plan.plan_id}
- 목표: {execution_plan.objective}
- 실행 전략: {execution_plan.strategy.value}
- 태스크 수: {len(execution_plan.tasks)}

**태스크 세부사항**:
{json.dumps([{
    'task_id': task.task_id,
    'agent': task.agent_config.name,
    'dependencies': task.dependencies
} for task in execution_plan.tasks], ensure_ascii=False, indent=2)}

**검증 항목**:
1. 태스크 수 일치성 확인
2. 의존성 순환 참조 검사
3. 에이전트 포트 충돌 검사
4. 실행 전략 적합성 평가
5. 리소스 요구사항 검토

다음 JSON 형식으로 응답해주세요:
{{
  "valid": true,
  "issues": [],
  "warnings": ["권고사항들"],
  "score": 0.9,
  "recommendations": ["개선사항들"]
}}"""

        try:
            response = await self.llm.ainvoke(validation_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                validation_result = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    validation_result = json.loads(json_match.group(1))
                else:
                    validation_result = {"valid": True, "issues": [], "warnings": [], "score": 0.5}
            
            return validation_result
            
        except Exception as e:
            logger.warning(f"Plan validation failed: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "score": 0.0,
                "recommendations": []
            }
    
    async def _monitor_execution(
        self,
        managed_plan: ManagedExecutionPlan,
        callback: Optional[Callable] = None
    ):
        """실행 모니터링"""
        
        monitoring_interval = 1.0  # 1초마다 모니터링
        
        try:
            while managed_plan.status == PlanStatus.EXECUTING:
                await asyncio.sleep(monitoring_interval)
                
                # 실행 상태 체크
                if managed_plan.execution_result:
                    # 에이전트별 상태 모니터링
                    for task_result in managed_plan.execution_result.task_results:
                        if task_result.get("status") == "failed":
                            await self._add_monitoring_event(
                                managed_plan,
                                "agent_failed",
                                f"Agent {task_result['agent_name']} failed",
                                "warning",
                                {"task_result": task_result}
                            )
                
                # 타임아웃 체크
                if managed_plan.started_at:
                    elapsed = datetime.now() - managed_plan.started_at
                    if elapsed > timedelta(minutes=30):  # 30분 타임아웃
                        await self._add_monitoring_event(
                            managed_plan,
                            "execution_timeout",
                            "Execution exceeded 30 minute timeout",
                            "error"
                        )
                        break
                
                if callback:
                    callback(f"Monitoring execution: {managed_plan.plan_id}")
                    
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for plan: {managed_plan.plan_id}")
        except Exception as e:
            logger.error(f"Monitoring error for plan {managed_plan.plan_id}: {e}")
    
    async def _add_monitoring_event(
        self,
        managed_plan: ManagedExecutionPlan,
        event_type: str,
        message: str,
        severity: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """모니터링 이벤트 추가"""
        
        event = MonitoringEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            plan_id=managed_plan.plan_id,
            agent_name=None,
            message=message,
            severity=severity,
            metadata=metadata or {}
        )
        
        managed_plan.monitoring_events.append(event)
        
        # 이벤트 수 제한
        if len(managed_plan.monitoring_events) > 1000:
            managed_plan.monitoring_events = managed_plan.monitoring_events[-800:]
        
        # 로그 출력
        if severity == "error":
            logger.error(f"[{managed_plan.plan_id}] {message}")
        elif severity == "warning":
            logger.warning(f"[{managed_plan.plan_id}] {message}")
        else:
            logger.info(f"[{managed_plan.plan_id}] {message}")
    
    async def _calculate_plan_metrics(self, managed_plan: ManagedExecutionPlan) -> PlanMetrics:
        """계획 메트릭 계산"""
        
        if not managed_plan.execution_result:
            return PlanMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        execution_result = managed_plan.execution_result
        
        # 기본 메트릭
        total_execution_time = execution_result.execution_time
        
        # 에이전트별 실행 시간
        agent_times = []
        success_count = 0
        error_count = 0
        
        for task_result in execution_result.task_results:
            if task_result.get("execution_time"):
                agent_times.append(task_result["execution_time"])
            
            if task_result.get("status") == "completed":
                success_count += 1
            else:
                error_count += 1
        
        average_agent_time = sum(agent_times) / len(agent_times) if agent_times else 0
        success_rate = success_count / len(execution_result.task_results) if execution_result.task_results else 0
        error_rate = error_count / len(execution_result.task_results) if execution_result.task_results else 0
        
        # 처리량 (태스크/초)
        throughput = len(execution_result.task_results) / total_execution_time if total_execution_time > 0 else 0
        
        # 리소스 활용도 (병렬 실행 효율성)
        ideal_time = max(agent_times) if agent_times else 0
        resource_utilization = ideal_time / total_execution_time if total_execution_time > 0 else 0
        
        # 비용 효율성 (성공률 / 실행시간)
        cost_efficiency = success_rate / (total_execution_time / 60) if total_execution_time > 0 else 0
        
        # 품질 점수
        quality_score = execution_result.confidence_score
        
        return PlanMetrics(
            total_execution_time=total_execution_time,
            average_agent_time=average_agent_time,
            success_rate=success_rate,
            error_rate=error_rate,
            throughput=throughput,
            resource_utilization=resource_utilization,
            cost_efficiency=cost_efficiency,
            quality_score=quality_score
        )
    
    async def _generate_optimization_recommendations(
        self,
        managed_plan: ManagedExecutionPlan
    ) -> List[OptimizationRecommendation]:
        """최적화 권고사항 생성"""
        
        if not managed_plan.metrics or not managed_plan.execution_result:
            return []
        
        optimization_prompt = f"""다음 실행 계획의 성능을 분석하고 최적화 권고사항을 제안해주세요:

**실행 메트릭**:
- 총 실행 시간: {managed_plan.metrics.total_execution_time:.2f}초
- 평균 에이전트 시간: {managed_plan.metrics.average_agent_time:.2f}초
- 성공률: {managed_plan.metrics.success_rate:.2%}
- 오류율: {managed_plan.metrics.error_rate:.2%}
- 처리량: {managed_plan.metrics.throughput:.2f} 태스크/초
- 리소스 활용도: {managed_plan.metrics.resource_utilization:.2%}
- 비용 효율성: {managed_plan.metrics.cost_efficiency:.2f}
- 품질 점수: {managed_plan.metrics.quality_score:.2f}

**실행 결과**:
- 계획된 태스크: {managed_plan.execution_result.total_tasks}
- 완료된 태스크: {managed_plan.execution_result.completed_tasks}
- 실패한 태스크: {managed_plan.execution_result.failed_tasks}

**모니터링 이벤트 수**: {len(managed_plan.monitoring_events)}

**최적화 권고사항 생성**:
1. 성능 개선 방안
2. 신뢰성 향상 방안
3. 비용 절감 방안
4. 리소스 최적화 방안

다음 JSON 형식으로 응답해주세요:
{{
  "recommendations": [
    {{
      "optimization_type": "performance",
      "description": "병렬 실행으로 처리 시간 단축",
      "expected_improvement": 0.3,
      "implementation_effort": "medium",
      "priority": 1,
      "estimated_impact": {{"time_reduction": 0.3, "cost_increase": 0.1}}
    }}
  ]
}}"""

        try:
            response = await self.llm.ainvoke(optimization_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                recommendations_data = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    recommendations_data = json.loads(json_match.group(1))
                else:
                    recommendations_data = {"recommendations": []}
            
            # OptimizationRecommendation 객체로 변환
            recommendations = []
            for rec_data in recommendations_data.get("recommendations", []):
                recommendation = OptimizationRecommendation(
                    optimization_type=rec_data.get("optimization_type", "general"),
                    description=rec_data.get("description", ""),
                    expected_improvement=rec_data.get("expected_improvement", 0.0),
                    implementation_effort=rec_data.get("implementation_effort", "unknown"),
                    priority=rec_data.get("priority", 5),
                    estimated_impact=rec_data.get("estimated_impact", {})
                )
                recommendations.append(recommendation)
            
            # 우선순위별 정렬
            recommendations.sort(key=lambda x: x.priority)
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Optimization recommendations generation failed: {e}")
            return []
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """계획 상태 조회"""
        managed_plan = self.managed_plans.get(plan_id)
        if not managed_plan:
            return None
        
        return {
            "plan_id": plan_id,
            "status": managed_plan.status.value,
            "created_at": managed_plan.created_at.isoformat(),
            "started_at": managed_plan.started_at.isoformat() if managed_plan.started_at else None,
            "completed_at": managed_plan.completed_at.isoformat() if managed_plan.completed_at else None,
            "total_tasks": len(managed_plan.execution_plan.tasks),
            "completed_tasks": managed_plan.execution_result.completed_tasks if managed_plan.execution_result else 0,
            "failed_tasks": managed_plan.execution_result.failed_tasks if managed_plan.execution_result else 0,
            "confidence_score": managed_plan.execution_result.confidence_score if managed_plan.execution_result else 0,
            "metrics": {
                "execution_time": managed_plan.metrics.total_execution_time if managed_plan.metrics else 0,
                "success_rate": managed_plan.metrics.success_rate if managed_plan.metrics else 0,
                "quality_score": managed_plan.metrics.quality_score if managed_plan.metrics else 0
            } if managed_plan.metrics else {},
            "event_count": len(managed_plan.monitoring_events),
            "optimization_count": len(managed_plan.optimization_recommendations)
        }
    
    async def get_plan_analytics(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """계획 분석 정보 조회"""
        managed_plan = self.managed_plans.get(plan_id)
        if not managed_plan:
            return None
        
        analytics = {
            "plan_overview": {
                "plan_id": plan_id,
                "status": managed_plan.status.value,
                "objective": managed_plan.execution_plan.objective,
                "strategy": managed_plan.execution_plan.strategy.value
            },
            "execution_metrics": managed_plan.metrics.__dict__ if managed_plan.metrics else {},
            "monitoring_summary": {
                "total_events": len(managed_plan.monitoring_events),
                "error_events": len([e for e in managed_plan.monitoring_events if e.severity == "error"]),
                "warning_events": len([e for e in managed_plan.monitoring_events if e.severity == "warning"]),
                "info_events": len([e for e in managed_plan.monitoring_events if e.severity == "info"])
            },
            "optimization_recommendations": [
                {
                    "type": rec.optimization_type,
                    "description": rec.description,
                    "improvement": rec.expected_improvement,
                    "effort": rec.implementation_effort,
                    "priority": rec.priority
                }
                for rec in managed_plan.optimization_recommendations
            ]
        }
        
        if managed_plan.integration_result:
            analytics["integration_summary"] = {
                "confidence_score": managed_plan.integration_result.confidence_score,
                "insight_count": len(managed_plan.integration_result.integrated_insights),
                "recommendation_count": len(managed_plan.integration_result.recommendations)
            }
        
        return analytics
    
    async def cancel_plan(self, plan_id: str) -> bool:
        """계획 취소"""
        managed_plan = self.managed_plans.get(plan_id)
        if not managed_plan:
            return False
        
        if managed_plan.status not in [PlanStatus.EXECUTING, PlanStatus.SCHEDULED]:
            return False
        
        # 실행 중인 계획 취소
        if managed_plan.status == PlanStatus.EXECUTING:
            await self.orchestrator.cancel_execution(plan_id)
        
        # 모니터링 중지
        if plan_id in self.active_monitors:
            self.active_monitors[plan_id].cancel()
            del self.active_monitors[plan_id]
        
        managed_plan.status = PlanStatus.CANCELLED
        managed_plan.completed_at = datetime.now()
        
        await self._add_monitoring_event(
            managed_plan,
            "plan_cancelled",
            "Plan execution cancelled by user",
            "warning"
        )
        
        logger.info(f"🛑 Cancelled plan: {plan_id}")
        return True
    
    async def get_all_plans(self) -> List[Dict[str, Any]]:
        """모든 계획 목록 조회"""
        plans = []
        for plan_id, managed_plan in self.managed_plans.items():
            plan_summary = await self.get_plan_status(plan_id)
            if plan_summary:
                plans.append(plan_summary)
        
        # 생성 시간 역순 정렬
        plans.sort(key=lambda x: x["created_at"], reverse=True)
        return plans
    
    def cleanup_old_plans(self, max_age_days: int = 7):
        """오래된 계획 정리"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        plans_to_remove = []
        for plan_id, managed_plan in self.managed_plans.items():
            if (managed_plan.completed_at and managed_plan.completed_at < cutoff_date) or \
               (not managed_plan.completed_at and managed_plan.created_at < cutoff_date):
                plans_to_remove.append(plan_id)
        
        for plan_id in plans_to_remove:
            del self.managed_plans[plan_id]
            if plan_id in self.active_monitors:
                self.active_monitors[plan_id].cancel()
                del self.active_monitors[plan_id]
        
        if plans_to_remove:
            logger.info(f"🧹 Cleaned up {len(plans_to_remove)} old plans")
        
        return len(plans_to_remove) 