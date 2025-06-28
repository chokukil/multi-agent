"""
A2A Task Executor - 실제 A2A 에이전트 태스크 실행 엔진

Phase 1, 2, 4, 5 통합 구현:
- JSON-RPC 2.0 기반 message/send 메서드 사용 (Phase 1)
- 고급 아티팩트 처리 (Phase 2)
- 에러 복구 및 Circuit Breaker (Phase 4)
- 지능형 계획 생성 (Phase 5)
"""

import asyncio
import httpx
import json
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class A2ATask:
    """A2A Task 객체"""
    id: str
    agent_url: str
    agent_name: str
    message: str
    status: TaskStatus = TaskStatus.SUBMITTED
    artifacts: List[Dict] = None
    error: Optional[str] = None
    progress: float = 0.0
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

@dataclass
class ExecutionPlan:
    """오케스트레이션 실행 계획"""
    objective: str
    reasoning: str
    steps: List[Dict]
    selected_agents: List[str]

class A2ATaskExecutor:
    """A2A 프로토콜 기반 태스크 실행 엔진 - Phase 4, 5 통합"""
    
    def __init__(self):
        self.active_tasks: Dict[str, A2ATask] = {}
        self.client_timeout = httpx.Timeout(30.0, connect=10.0)
        
        # Phase 4: 에러 복구 매니저 통합
        from core.error_recovery import error_recovery_manager
        self.error_recovery = error_recovery_manager
        
        # Phase 4: 성능 모니터링 통합
        from core.performance_monitor import performance_monitor
        self.performance_monitor = performance_monitor
        
    async def execute_orchestration_plan(
        self, 
        plan: ExecutionPlan,
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        오케스트레이션 계획을 실제로 실행 - Phase 4, 5 기능 통합
        
        Args:
            plan: 실행할 오케스트레이션 계획
            data_context: 데이터 컨텍스트
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            실행 결과 딕셔너리
        """
        logger.info(f"🚀 오케스트레이션 계획 실행 시작: {plan.objective}")
        
        # Phase 4: 성능 모니터링 시작
        execution_id = f"exec_{int(asyncio.get_event_loop().time())}"
        
        results = {
            "objective": plan.objective,
            "status": "executing",
            "steps_completed": 0,
            "total_steps": len(plan.steps),
            "step_results": [],
            "final_artifacts": [],
            "execution_time": 0,
            "execution_id": execution_id
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 단계별 실행 (Phase 4: 에러 복구 포함)
            for i, step in enumerate(plan.steps):
                if progress_callback:
                    progress_callback(f"📋 Step {i+1}/{len(plan.steps)}: {step.get('task_description', 'Processing...')}")
                
                # Phase 4: 에러 복구 기능과 함께 단계 실행
                step_result = await self._execute_step_with_recovery(step, data_context, progress_callback)
                results["step_results"].append(step_result)
                results["steps_completed"] = i + 1
                
                # 단계 실패 시 처리
                if step_result["status"] == "failed":
                    if step_result.get("recovery_applied"):
                        # 복구 시도했지만 실패
                        logger.warning(f"⚠️ Step {i+1} 복구 실패, 계속 진행")
                    else:
                        # 복구 불가능한 실패
                        results["status"] = "failed"
                        results["error"] = step_result.get("error", "Unknown error")
                        break
                elif step_result["status"] == "skipped":
                    logger.info(f"⏭️ Step {i+1} 건너뛰기")
                    
                # 아티팩트 수집 (Phase 2: 고급 아티팩트 처리)
                if step_result.get("artifacts"):
                    processed_artifacts = self._process_artifacts(step_result["artifacts"], step.get("agent_name"))
                    results["final_artifacts"].extend(processed_artifacts)
            
            if results["status"] != "failed":
                results["status"] = "completed"
                
        except Exception as e:
            logger.error(f"❌ 오케스트레이션 실행 실패: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        results["execution_time"] = asyncio.get_event_loop().time() - start_time
        
        # Phase 4: 성능 메트릭 기록
        self.performance_monitor._add_metric(
            "orchestration_execution_time", 
            results["execution_time"], 
            "seconds"
        )
        self.performance_monitor._add_metric(
            "orchestration_steps_completed", 
            results["steps_completed"], 
            "count"
        )
        
        logger.info(f"✅ 오케스트레이션 완료: {results['status']} ({results['execution_time']:.2f}s)")
        
        return results
    
    async def _execute_step_with_recovery(
        self, 
        step: Dict, 
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Phase 4: 에러 복구 기능이 포함된 단계 실행"""
        agent_name = step.get("agent_name")
        task_description = step.get("task_description")
        
        logger.info(f"🔧 단계 실행: {agent_name} - {task_description}")
        
        # Phase 4: 에러 복구 매니저를 통한 실행
        try:
            result = await self.error_recovery.execute_with_recovery(
                agent_name=agent_name,
                task_func=self._execute_single_step,
                task_args=(step, data_context, progress_callback),
                context={"step": step, "data_context": data_context}
            )
            
            return {
                "agent_name": agent_name,
                "status": result["status"],
                "artifacts": result.get("result", {}).get("artifacts", []),
                "error": result.get("error"),
                "recovery_applied": result.get("recovery_applied", False),
                "recovery_strategy": result.get("recovery_strategy")
            }
            
        except Exception as e:
            logger.error(f"❌ 단계 실행 완전 실패 ({agent_name}): {e}")
            return {
                "agent_name": agent_name,
                "status": "failed", 
                "error": str(e),
                "artifacts": [],
                "recovery_applied": False
            }
    
    async def _execute_single_step(
        self, 
        step: Dict, 
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """개별 단계 실행 (에러 복구 매니저에서 호출)"""
        agent_name = step.get("agent_name")
        task_description = step.get("task_description")
        
        # 에이전트 URL 매핑
        agent_port_map = {
            "AI_DS_Team DataLoaderToolsAgent": 8307,
            "AI_DS_Team DataCleaningAgent": 8306,
            "AI_DS_Team EDAToolsAgent": 8312,
            "AI_DS_Team DataVisualizationAgent": 8308,
            "AI_DS_Team DataWranglingAgent": 8309,
            "AI_DS_Team FeatureEngineeringAgent": 8310,
            "AI_DS_Team SQLDatabaseAgent": 8311,
            "AI_DS_Team H2OMLAgent": 8313,
            "AI_DS_Team MLflowToolsAgent": 8314,
        }
        
        agent_port = agent_port_map.get(agent_name)
        if not agent_port:
            raise Exception(f"Unknown agent: {agent_name}")
        
        agent_url = f"http://localhost:{agent_port}"
        
        # A2A 태스크 생성 및 실행
        task = A2ATask(
            id=str(uuid.uuid4()),
            agent_url=agent_url,
            agent_name=agent_name,
            message=self._prepare_task_message(task_description, data_context)
        )
        
        # Phase 4: 성능 모니터링 - A2A 호출 추적
        call_id = self.performance_monitor.start_a2a_call(
            task.id, agent_name, len(task.message)
        )
        
        try:
            result = await self._execute_a2a_task(task, progress_callback)
            
            # Phase 4: 성능 모니터링 - 성공 기록
            self.performance_monitor.end_a2a_call(
                call_id, "completed", 
                response_size=len(str(result))
            )
            
            return result
            
        except Exception as e:
            # Phase 4: 성능 모니터링 - 실패 기록
            self.performance_monitor.end_a2a_call(
                call_id, "failed", 
                error_message=str(e)
            )
            raise e
    
    async def _execute_a2a_task(
        self, 
        task: A2ATask, 
        progress_callback=None
    ) -> Dict[str, Any]:
        """실제 A2A 태스크 실행"""
        self.active_tasks[task.id] = task
        
        try:
            # A2A message/send 요청 구성
            request_payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": task.id,
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": task.message
                            }
                        ]
                    }
                },
                "id": 1
            }
            
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                if progress_callback:
                    progress_callback(f"🔄 {task.agent_name} 에이전트와 통신 중...")
                
                # A2A 요청 전송
                response = await client.post(
                    task.agent_url,
                    json=request_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "result" in result:
                        # A2A 응답 파싱
                        a2a_result = result["result"]
                        
                        if isinstance(a2a_result, dict) and "parts" in a2a_result:
                            # Parts에서 아티팩트 추출
                            artifacts = self._extract_artifacts_from_parts(a2a_result["parts"])
                            
                            task.status = TaskStatus.COMPLETED
                            task.artifacts = artifacts
                            task.progress = 100.0
                            
                            if progress_callback:
                                progress_callback(f"✅ {task.agent_name} 작업 완료")
                            
                            return {
                                "status": "completed",
                                "artifacts": artifacts,
                                "raw_response": a2a_result
                            }
                    
                    # 에러 응답 처리
                    if "error" in result:
                        error_msg = result["error"].get("message", "Unknown A2A error")
                        task.status = TaskStatus.FAILED
                        task.error = error_msg
                        
                        raise Exception(error_msg)
                
                # HTTP 에러
                task.status = TaskStatus.FAILED
                task.error = f"HTTP {response.status_code}: {response.text}"
                
                raise Exception(task.error)
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"❌ A2A 태스크 실행 실패: {e}")
            raise e
        
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    def _prepare_task_message(self, task_description: str, data_context: Optional[Dict] = None) -> str:
        """태스크 메시지 준비 - Phase 5: 지능형 컨텍스트 포함"""
        message = task_description
        
        if data_context:
            # Phase 5: 더 지능적인 컨텍스트 정보 제공
            context_summary = {
                "dataset_shape": data_context.get("dataset_info", "Unknown"),
                "columns": data_context.get("columns", [])[:10],  # 처음 10개만
                "data_types": list(data_context.get("dtypes", {}).keys())[:5],  # 처음 5개만
                "sample_available": bool(data_context.get("sample_data"))
            }
            
            context_info = f"\n\n=== 데이터 컨텍스트 ===\n{json.dumps(context_summary, ensure_ascii=False, indent=2)}"
            message += context_info
            
        return message
    
    def _extract_artifacts_from_parts(self, parts: List[Dict]) -> List[Dict]:
        """A2A Parts에서 아티팩트 추출 - Phase 2: 고급 처리"""
        artifacts = []
        
        for part in parts:
            if isinstance(part, dict):
                part_kind = part.get("kind", part.get("type", "unknown"))
                
                if part_kind == "text":
                    artifacts.append({
                        "type": "text",
                        "title": part.get("title", "텍스트 결과"),
                        "content": part.get("text", ""),
                        "metadata": {
                            **part.get("metadata", {}),
                            "timestamp": asyncio.get_event_loop().time(),
                            "source": "a2a_agent"
                        }
                    })
                elif part_kind == "data":
                    artifacts.append({
                        "type": "data",
                        "title": part.get("title", "데이터 결과"), 
                        "content": part.get("data", {}),
                        "metadata": {
                            **part.get("metadata", {}),
                            "timestamp": asyncio.get_event_loop().time(),
                            "source": "a2a_agent"
                        }
                    })
                elif part_kind == "file":
                    artifacts.append({
                        "type": "file",
                        "title": part.get("title", "파일 결과"),
                        "content": part.get("file", {}),
                        "metadata": {
                            **part.get("metadata", {}),
                            "timestamp": asyncio.get_event_loop().time(),
                            "source": "a2a_agent"
                        }
                    })
                elif part_kind == "chart" or "plot" in part_kind.lower():
                    artifacts.append({
                        "type": "chart",
                        "title": part.get("title", "차트 결과"),
                        "content": part.get("data", part.get("chart", {})),
                        "metadata": {
                            **part.get("metadata", {}),
                            "chart_type": part.get("chart_type", "unknown"),
                            "timestamp": asyncio.get_event_loop().time(),
                            "source": "a2a_agent"
                        }
                    })
        
        return artifacts
    
    def _process_artifacts(self, artifacts: List[Dict], agent_name: str) -> List[Dict]:
        """Phase 2: 아티팩트 후처리"""
        processed = []
        
        for artifact in artifacts:
            # 메타데이터 보강
            if "metadata" not in artifact:
                artifact["metadata"] = {}
            
            artifact["metadata"].update({
                "agent_name": agent_name,
                "processed_at": asyncio.get_event_loop().time(),
                "processor": "a2a_task_executor"
            })
            
            # 제목 자동 생성
            if not artifact.get("title"):
                artifact_type = artifact.get("type", "unknown")
                artifact["title"] = f"{agent_name} {artifact_type.title()} 결과"
            
            processed.append(artifact)
        
        return processed

# 글로벌 실행 엔진 인스턴스
task_executor = A2ATaskExecutor()
