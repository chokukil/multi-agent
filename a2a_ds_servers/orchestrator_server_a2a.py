#!/usr/bin/env python3
"""
AI Data Science Team Orchestrator Server
A2A SDK 0.2.9 기반 구현
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from pydantic import BaseModel

# A2A SDK 0.2.9 올바른 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part,
    InternalError,
    InvalidParamsError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI DS Team 에이전트 레지스트리 (모든 9개 에이전트)
AI_DS_TEAM_REGISTRY = {
    "data_cleaning": {
        "name": "Data Cleaning Agent",
        "url": "http://localhost:8306",
        "skills": ["data_cleaning", "data_validation", "outlier_detection"],
        "description": "데이터 정제 및 검증을 수행하는 에이전트"
    },
    "data_loader": {
        "name": "Data Loader Tools Agent", 
        "url": "http://localhost:8307",
        "skills": ["data_loading", "file_processing", "format_conversion"],
        "description": "다양한 형식의 데이터를 로드하고 처리하는 에이전트"
    },
    "data_visualization": {
        "name": "Data Visualization Agent",
        "url": "http://localhost:8308", 
        "skills": ["plotting", "charting", "visualization"],
        "description": "데이터 시각화 및 차트 생성을 담당하는 에이전트"
    },
    "data_wrangling": {
        "name": "Data Wrangling Agent",
        "url": "http://localhost:8309",
        "skills": ["data_transformation", "feature_engineering", "data_reshaping"],
        "description": "데이터 변환 및 특성 엔지니어링을 수행하는 에이전트"
    },
    "feature_engineering": {
        "name": "Feature Engineering Agent", 
        "url": "http://localhost:8310",
        "skills": ["feature_creation", "feature_selection", "dimensionality_reduction"],
        "description": "특성 생성 및 선택을 수행하는 에이전트"
    },
    "sql_database": {
        "name": "SQL Database Agent",
        "url": "http://localhost:8311",
        "skills": ["sql_queries", "database_operations", "data_extraction"],
        "description": "SQL 데이터베이스 작업을 수행하는 에이전트"
    },
    "eda_tools": {
        "name": "EDA Tools Agent",
        "url": "http://localhost:8312", 
        "skills": ["exploratory_analysis", "statistical_analysis", "data_profiling"],
        "description": "탐색적 데이터 분석을 수행하는 에이전트"
    },
    "h2o_ml": {
        "name": "H2O ML Agent",
        "url": "http://localhost:8313",
        "skills": ["machine_learning", "automl", "model_training"],
        "description": "H2O를 이용한 머신러닝 모델링을 수행하는 에이전트"
    },
    "mlflow_tools": {
        "name": "MLflow Tools Agent", 
        "url": "http://localhost:8314",
        "skills": ["experiment_tracking", "model_registry", "model_deployment"],
        "description": "MLflow를 이용한 실험 추적 및 모델 관리를 수행하는 에이전트"
    }
}

class AgentDiscoveryService:
    """동적 에이전트 발견 서비스"""
    
    def __init__(self, registry: Dict[str, Dict[str, Any]]):
        self.registry = registry
        self.httpx_client = httpx.AsyncClient(timeout=5.0)
    
    async def discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """사용 가능한 에이전트들을 발견합니다."""
        available_agents = {}
        
        for agent_id, agent_info in self.registry.items():
            try:
                # Agent Card 확인
                response = await self.httpx_client.get(
                    f"{agent_info['url']}/.well-known/agent.json"
                )
                if response.status_code == 200:
                    agent_card = response.json()
                    available_agents[agent_id] = {
                        **agent_info,
                        "status": "available",
                        "agent_card": agent_card
                    }
                    logger.info(f"✅ {agent_info['name']} 발견됨: {agent_info['url']}")
                else:
                    logger.warning(f"❌ {agent_info['name']} 응답 없음: {agent_info['url']}")
            except Exception as e:
                logger.warning(f"❌ {agent_info['name']} 연결 실패: {e}")
        
        return available_agents

class IntelligentPlanner:
    """LLM 기반 지능형 계획 수립기"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        self.discovery_service = discovery_service
    
    async def create_orchestration_plan(self, user_query: str, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """사용자 쿼리에 대한 오케스트레이션 계획을 수립합니다."""
        
        # 쿼리 유형 분석
        query_lower = user_query.lower()
        
        if any(keyword in query_lower for keyword in ["eda", "exploratory", "분석", "explore", "분석해"]):
            return await self._create_eda_plan(available_agents)
        elif any(keyword in query_lower for keyword in ["clean", "정제", "cleaning"]):
            return await self._create_cleaning_plan(available_agents)
        elif any(keyword in query_lower for keyword in ["visualiz", "시각화", "plot", "chart"]):
            return await self._create_visualization_plan(available_agents)
        elif any(keyword in query_lower for keyword in ["model", "ml", "machine learning", "모델"]):
            return await self._create_modeling_plan(available_agents)
        else:
            # 기본 종합 분석 계획
            return await self._create_comprehensive_plan(available_agents)
    
    async def _create_eda_plan(self, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """EDA 중심 계획"""
        plan = []
        
        if "data_loader" in available_agents:
            plan.append({
                "step": 1,
                "agent": "data_loader",
                "task": "데이터 로딩 및 기본 정보 확인",
                "description": "데이터셋을 로드하고 기본 구조를 파악합니다."
            })
        
        if "data_cleaning" in available_agents:
            plan.append({
                "step": 2,
                "agent": "data_cleaning", 
                "task": "데이터 품질 검사 및 기본 정제",
                "description": "결측값, 중복값, 이상값을 확인하고 기본 정제를 수행합니다."
            })
        
        if "eda_tools" in available_agents:
            plan.append({
                "step": 3,
                "agent": "eda_tools",
                "task": "탐색적 데이터 분석 수행", 
                "description": "통계적 요약, 분포 분석, 상관관계 분석을 수행합니다."
            })
        
        if "data_visualization" in available_agents:
            plan.append({
                "step": 4,
                "agent": "data_visualization",
                "task": "데이터 시각화",
                "description": "히스토그램, 산점도, 상관관계 매트릭스 등을 생성합니다."
            })
        
        if "eda_tools" in available_agents:
            plan.append({
                "step": 5,
                "agent": "eda_tools", 
                "task": "종합 EDA 리포트 생성",
                "description": "분석 결과를 종합하여 최종 리포트를 생성합니다."
            })
        
        return plan
    
    async def _create_comprehensive_plan(self, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """종합 분석 계획"""
        plan = []
        step = 1
        
        # 1. 데이터 로딩
        if "data_loader" in available_agents:
            plan.append({
                "step": step,
                "agent": "data_loader",
                "task": "데이터 로딩",
                "description": "데이터를 로드하고 기본 구조를 확인합니다."
            })
            step += 1
        
        # 2. 데이터 정제
        if "data_cleaning" in available_agents:
            plan.append({
                "step": step,
                "agent": "data_cleaning",
                "task": "데이터 정제",
                "description": "데이터 품질을 검사하고 정제 작업을 수행합니다."
            })
            step += 1
        
        # 3. EDA
        if "eda_tools" in available_agents:
            plan.append({
                "step": step,
                "agent": "eda_tools", 
                "task": "탐색적 데이터 분석",
                "description": "데이터의 통계적 특성과 패턴을 분석합니다."
            })
            step += 1
        
        # 4. 시각화
        if "data_visualization" in available_agents:
            plan.append({
                "step": step,
                "agent": "data_visualization",
                "task": "데이터 시각화",
                "description": "분석 결과를 시각적으로 표현합니다."
            })
            step += 1
        
        # 5. 특성 엔지니어링 (선택적)
        if "feature_engineering" in available_agents:
            plan.append({
                "step": step,
                "agent": "feature_engineering",
                "task": "특성 엔지니어링",
                "description": "새로운 특성을 생성하고 기존 특성을 개선합니다."
            })
            step += 1
        
        return plan
    
    async def _create_cleaning_plan(self, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """데이터 정제 중심 계획"""
        plan = []
        
        if "data_loader" in available_agents:
            plan.append({
                "step": 1,
                "agent": "data_loader",
                "task": "데이터 로딩",
                "description": "원본 데이터를 로드합니다."
            })
        
        if "data_cleaning" in available_agents:
            plan.append({
                "step": 2,
                "agent": "data_cleaning",
                "task": "심층 데이터 정제",
                "description": "결측값 처리, 이상값 제거, 데이터 타입 최적화를 수행합니다."
            })
        
        return plan
    
    async def _create_visualization_plan(self, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """시각화 중심 계획"""
        plan = []
        
        if "data_loader" in available_agents:
            plan.append({
                "step": 1,
                "agent": "data_loader", 
                "task": "데이터 로딩",
                "description": "시각화할 데이터를 로드합니다."
            })
        
        if "data_visualization" in available_agents:
            plan.append({
                "step": 2,
                "agent": "data_visualization",
                "task": "종합 시각화",
                "description": "다양한 차트와 그래프를 생성합니다."
            })
        
        return plan
    
    async def _create_modeling_plan(self, available_agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """모델링 중심 계획"""
        plan = []
        step = 1
        
        if "data_loader" in available_agents:
            plan.append({
                "step": step,
                "agent": "data_loader",
                "task": "데이터 로딩",
                "description": "모델링용 데이터를 로드합니다."
            })
            step += 1
        
        if "data_cleaning" in available_agents:
            plan.append({
                "step": step,
                "agent": "data_cleaning",
                "task": "데이터 전처리",
                "description": "모델링을 위한 데이터 정제를 수행합니다."
            })
            step += 1
        
        if "feature_engineering" in available_agents:
            plan.append({
                "step": step,
                "agent": "feature_engineering",
                "task": "특성 엔지니어링",
                "description": "모델 성능 향상을 위한 특성을 생성합니다."
            })
            step += 1
        
        if "h2o_ml" in available_agents:
            plan.append({
                "step": step,
                "agent": "h2o_ml",
                "task": "머신러닝 모델 훈련",
                "description": "H2O AutoML을 사용하여 모델을 훈련합니다."
            })
            step += 1
        
        if "mlflow_tools" in available_agents:
            plan.append({
                "step": step,
                "agent": "mlflow_tools",
                "task": "모델 추적 및 관리",
                "description": "MLflow로 실험을 추적하고 모델을 관리합니다."
            })
            step += 1
        
        return plan

class A2ATaskExecutor:
    """A2A 프로토콜 기반 태스크 실행기"""
    
    def __init__(self):
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
    
    async def execute_agent_task(self, agent_info: Dict[str, Any], task_description: str, context_id: str) -> Dict[str, Any]:
        """개별 에이전트에게 태스크를 실행시킵니다."""
        try:
            # A2A 메시지 구성
            message_payload = {
                "jsonrpc": "2.0",
                "id": f"orchestrator-{context_id}",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": task_description
                            }
                        ],
                        "messageId": f"msg-{context_id}"
                    },
                    "metadata": {}
                }
            }
            
            # A2A 요청 전송
            response = await self.httpx_client.post(
                f"{agent_info['url']}",
                json=message_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "agent": agent_info["name"],
                    "result": result.get("result", {}),
                    "status": "completed"
                }
            else:
                logger.error(f"❌ {agent_info['name']} 실행 실패: HTTP {response.status_code}")
                return {
                    "success": False,
                    "agent": agent_info["name"],
                    "error": f"HTTP {response.status_code}",
                    "status": "failed"
                }
                
        except Exception as e:
            logger.error(f"❌ {agent_info['name']} 실행 중 오류: {e}")
            return {
                "success": False,
                "agent": agent_info["name"], 
                "error": str(e),
                "status": "failed"
            }

class AIDataScienceOrchestratorExecutor(AgentExecutor):
    """AI 데이터 과학팀 오케스트레이터 실행기"""

    def __init__(self):
        super().__init__()
        self.discovery_service = AgentDiscoveryService(AI_DS_TEAM_REGISTRY)
        self.planner = IntelligentPlanner(self.discovery_service)
        self.task_executor = A2ATaskExecutor()
        self.active_tasks = {}  # {context_id: [asyncio.Task]}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """오케스트레이션 로직 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # 1. 입력에서 프롬프트와 에이전트 목록 추출 (수정된 부분)
            user_prompt = None
            available_agents_from_client = None
            for part in context.message.parts:
                if part.root.kind == 'text':
                    user_prompt = part.root.text
                elif part.root.kind == 'json':
                    available_agents_from_client = part.root.json

            if not user_prompt:
                raise InvalidParamsError("요청에서 프롬프트를 찾을 수 없습니다.")
            if not available_agents_from_client:
                 # 클라이언트가 에이전트 목록을 보내지 않은 경우, 서버에서 동적으로 탐색
                logger.info("클라이언트로부터 에이전트 목록을 받지 못했습니다. 서버에서 에이전트를 탐색합니다.")
                available_agents_from_client = await self.discovery_service.discover_agents()

            logger.info(f"수신된 프롬프트: {user_prompt}")
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=f"사용자 요청 접수: '{user_prompt}'")]
                )
            )

            # 2. 지능형 계획 수립
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="지능형 계획 수립 시작...")]
                )
            )
            
            plan = await self.planner.create_orchestration_plan(user_prompt, available_agents_from_client)
            
            if not plan:
                raise ServerError("사용자 요청에 대한 계획을 수립할 수 없습니다.")

            logger.info(f"수립된 계획: {plan}")
            
            # 계획을 JSON 아티팩트로 전송 (올바른 방법)
            plan_json_str = json.dumps({"steps": plan}, ensure_ascii=False, indent=2)
            plan_parts = [TextPart(text=plan_json_str)]
            await task_updater.add_artifact(
                parts=plan_parts,
                name="execution_plan",
                metadata={"content_type": "application/json"}
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=f"총 {len(plan)} 단계의 계획 수립 완료.")]
                )
            )

        except Exception as e:
            logger.error(f"오케스트레이션 중 오류 발생: {e}", exc_info=True)
            error_message = f"오케스트레이션 실패: {e}"
            await task_updater.reject(message=error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """태스크 취소"""
        logger.info("🛑 오케스트레이션 태스크가 취소되었습니다.")
        # 필요시 진행 중인 에이전트 태스크들도 취소할 수 있음

def main():
    """메인 함수"""
    try:
        # Agent Skills 정의
        orchestration_skill = AgentSkill(
            id='ai_ds_orchestration',
            name='AI Data Science Team Orchestration',
            description='AI 데이터 사이언스 팀의 여러 에이전트들을 지능적으로 오케스트레이션합니다.',
            tags=['orchestration', 'data science', 'multi-agent', 'ai team'],
            examples=[
                'EDA 분석을 수행해주세요',
                '데이터를 정제하고 시각화해주세요', 
                '머신러닝 모델을 훈련해주세요',
                '종합적인 데이터 분석을 해주세요'
            ]
        )
        
        # Agent Card 생성
        agent_card = AgentCard(
            name='AI Data Science Team Orchestrator',
            description='AI 데이터 사이언스 팀의 9개 전문 에이전트들을 지능적으로 조율하여 복합적인 데이터 사이언스 작업을 수행합니다.',
            url='http://localhost:8100/',
            version='1.0.0',
            defaultInputModes=['text'],
            defaultOutputModes=['text', 'application/json'],
            capabilities=AgentCapabilities(streaming=True),
            skills=[orchestration_skill]
        )
        
        # Request Handler 설정
        request_handler = DefaultRequestHandler(
            agent_executor=AIDataScienceOrchestratorExecutor(),
            task_store=InMemoryTaskStore()
        )
        
        # A2A Server 생성
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        logger.info("🚀 AI Data Science Team Orchestrator 서버를 시작합니다...")
        logger.info("🌐 서버 주소: http://localhost:8100")
        logger.info("📋 Agent Card: http://localhost:8100/.well-known/agent.json")
        
        # 서버 실행
        uvicorn.run(server.build(), host='0.0.0.0', port=8100, log_level='info')
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
