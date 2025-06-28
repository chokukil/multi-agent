"""
A2A Task Executor - 실제 A2A 에이전트 태스크 실행 엔진

A2A 프로토콜 연구 결과를 바탕으로 구현된 실행 엔진:
- JSON-RPC 2.0 기반 message/send 메서드 사용
- Server-Sent Events (SSE) 스트리밍 지원
- Task 생명주기 관리 (submitted -> working -> completed/failed)
- Multi-modal Parts 및 Artifacts 처리
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
    """A2A 프로토콜 기반 태스크 실행 엔진"""
    
    def __init__(self):
        self.active_tasks: Dict[str, A2ATask] = {}
        self.client_timeout = httpx.Timeout(30.0, connect=10.0)
        
    async def execute_orchestration_plan(
        self, 
        plan: ExecutionPlan,
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        오케스트레이션 계획을 실제로 실행
        
        Args:
            plan: 실행할 오케스트레이션 계획
            data_context: 데이터 컨텍스트 (타이타닉 데이터 등)
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            실행 결과 딕셔너리
        """
        logger.info(f"🚀 오케스트레이션 계획 실행 시작: {plan.objective}")
        
        results = {
            "objective": plan.objective,
            "status": "executing",
            "steps_completed": 0,
            "total_steps": len(plan.steps),
            "step_results": [],
            "final_artifacts": [],
            "execution_time": 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 단계별 실행
            for i, step in enumerate(plan.steps):
                if progress_callback:
                    progress_callback(f"📋 Step {i+1}/{len(plan.steps)}: {step.get('task_description', 'Processing...')}")
                
                step_result = await self._execute_step(step, data_context, progress_callback)
                results["step_results"].append(step_result)
                results["steps_completed"] = i + 1
                
                # 단계 실패 시 중단
                if step_result["status"] == "failed":
                    results["status"] = "failed"
                    results["error"] = step_result.get("error", "Unknown error")
                    break
                    
                # 아티팩트 수집
                if step_result.get("artifacts"):
                    results["final_artifacts"].extend(step_result["artifacts"])
            
            if results["status"] != "failed":
                results["status"] = "completed"
                
        except Exception as e:
            logger.error(f"❌ 오케스트레이션 실행 실패: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        results["execution_time"] = asyncio.get_event_loop().time() - start_time
        logger.info(f"✅ 오케스트레이션 완료: {results['status']} ({results['execution_time']:.2f}s)")
        
        return results
    
    async def _execute_step(
        self, 
        step: Dict, 
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """개별 단계 실행"""
        agent_name = step.get("agent_name")
        task_description = step.get("task_description")
        
        logger.info(f"🔧 단계 실행: {agent_name} - {task_description}")
        
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
            return {
                "agent_name": agent_name,
                "status": "failed",
                "error": f"Unknown agent: {agent_name}",
                "artifacts": []
            }
        
        agent_url = f"http://localhost:{agent_port}"
        
        # A2A 태스크 생성 및 실행
        task = A2ATask(
            id=str(uuid.uuid4()),
            agent_url=agent_url,
            agent_name=agent_name,
            message=self._prepare_task_message(task_description, data_context)
        )
        
        try:
            result = await self._execute_a2a_task(task, progress_callback)
            return {
                "agent_name": agent_name,
                "task_id": task.id,
                "status": result["status"],
                "artifacts": result.get("artifacts", []),
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"❌ 단계 실행 실패 ({agent_name}): {e}")
            return {
                "agent_name": agent_name,
                "status": "failed", 
                "error": str(e),
                "artifacts": []
            }
    
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
                        
                        return {
                            "status": "failed",
                            "error": error_msg
                        }
                
                # HTTP 에러
                task.status = TaskStatus.FAILED
                task.error = f"HTTP {response.status_code}: {response.text}"
                
                return {
                    "status": "failed",
                    "error": task.error
                }
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"❌ A2A 태스크 실행 실패: {e}")
            
            return {
                "status": "failed",
                "error": str(e)
            }
        
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    def _prepare_task_message(self, task_description: str, data_context: Optional[Dict] = None) -> str:
        """태스크 메시지 준비"""
        message = task_description
        
        if data_context:
            context_info = f"\ndata_info: {json.dumps(data_context)}"
            message += context_info
            
        return message
    
    def _extract_artifacts_from_parts(self, parts: List[Dict]) -> List[Dict]:
        """A2A Parts에서 아티팩트 추출"""
        artifacts = []
        
        for part in parts:
            if isinstance(part, dict):
                part_kind = part.get("kind", part.get("type", "unknown"))
                
                if part_kind == "text":
                    artifacts.append({
                        "type": "text",
                        "content": part.get("text", ""),
                        "metadata": part.get("metadata", {})
                    })
                elif part_kind == "data":
                    artifacts.append({
                        "type": "data", 
                        "content": part.get("data", {}),
                        "metadata": part.get("metadata", {})
                    })
                elif part_kind == "file":
                    artifacts.append({
                        "type": "file",
                        "content": part.get("file", {}),
                        "metadata": part.get("metadata", {})
                    })
        
        return artifacts

# 글로벌 실행 엔진 인스턴스
task_executor = A2ATaskExecutor() 