"""
A2A SDK 기반 AI Data Science Team Orchestrator
AgentOrchestra 논문의 원칙을 적용한 Hierarchical Multi-Agent Framework

핵심 원칙:
1. Extensibility: 새로운 에이전트 추가/제거 용이
2. Multimodality: 텍스트, 이미지, 데이터 등 다중 모달 지원
3. Modularity: 각 에이전트의 독립성과 재사용성
4. Coordination: LLM 기반 지능형 조정
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessagePart,
    TextPart,
)
from a2a.utils import new_agent_text_message
from openai import AsyncOpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI_DS_Team 에이전트 레지스트리 (동적 발견 기반)
AI_DS_TEAM_REGISTRY = {
    "data_loader": {
        "url": "http://localhost:8307",
        "capabilities": ["file_loading", "database_connection", "api_integration"],
        "description": "다양한 데이터 소스 로딩 및 전처리"
    },
    "data_cleaning": {
        "url": "http://localhost:8306", 
        "capabilities": ["missing_value_handling", "outlier_detection", "data_validation"],
        "description": "누락값 처리, 이상치 제거, 데이터 품질 개선"
    },
    "data_wrangling": {
        "url": "http://localhost:8309",
        "capabilities": ["data_transformation", "aggregation", "merging"], 
        "description": "Pandas 기반 데이터 변환 및 조작"
    },
    "eda_tools": {
        "url": "http://localhost:8312",
        "capabilities": ["missing_data_analysis", "sweetviz_reports", "correlation_analysis"],
        "description": "missingno, sweetviz, correlation funnel 활용 EDA"
    },
    "data_visualization": {
        "url": "http://localhost:8308",
        "capabilities": ["interactive_charts", "statistical_plots", "dashboards"],
        "description": "Plotly, Matplotlib 기반 고급 시각화"
    },
    "feature_engineering": {
        "url": "http://localhost:8310",
        "capabilities": ["feature_creation", "feature_selection", "encoding"],
        "description": "고급 피처 생성 및 선택"
    },
    "h2o_ml": {
        "url": "http://localhost:8313",
        "capabilities": ["automl", "model_training", "model_evaluation"],
        "description": "H2O AutoML 기반 머신러닝"
    },
    "mlflow_tools": {
        "url": "http://localhost:8314",
        "capabilities": ["experiment_tracking", "model_registry", "deployment"],
        "description": "MLflow 기반 실험 관리 및 모델 추적"
    },
    "sql_database": {
        "url": "http://localhost:8311",
        "capabilities": ["sql_queries", "database_analysis", "data_extraction"],
        "description": "SQL 데이터베이스 쿼리 및 분석"
    }
}


class AgentDiscoveryService:
    """동적 에이전트 발견 서비스 (MCP 스타일)"""
    
    async def discover_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """사용 가능한 에이전트들을 동적으로 발견"""
        available_agents = {}
        
        for agent_name, agent_info in AI_DS_TEAM_REGISTRY.items():
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{agent_info['url']}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        available_agents[agent_name] = {
                            "status": "available",
                            "agent_card": agent_card,
                            "url": agent_info["url"],
                            "capabilities": agent_info["capabilities"],
                            "description": agent_info["description"]
                        }
                        logger.info(f"✅ Discovered agent: {agent_name}")
                    else:
                        logger.warning(f"❌ Agent {agent_name} not available: HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"❌ Failed to discover agent {agent_name}: {e}")
        
        return available_agents


class IntelligentPlanner:
    """LLM 기반 지능형 계획 생성기 (AgentOrchestra 논문 기반)"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "sk-test"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
        )
    
    async def create_orchestration_plan(
        self, 
        user_query: str, 
        available_agents: Dict[str, Dict[str, Any]],
        data_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """LLM을 활용한 지능형 오케스트레이션 계획 생성"""
        
        # 에이전트 능력 요약
        agent_capabilities = []
        for agent_name, agent_info in available_agents.items():
            if agent_info["status"] == "available":
                capabilities_str = ", ".join(agent_info["capabilities"])
                agent_capabilities.append(
                    f"- {agent_name}: {agent_info['description']} (기능: {capabilities_str})"
                )
        
        # 데이터 컨텍스트 정보
        data_info = ""
        if data_context:
            data_info = f"""
현재 로드된 데이터 정보:
- 데이터 형태: {data_context.get('dataset_info', 'Unknown')}
- 컬럼: {', '.join(data_context.get('columns', [])[:10])}{'...' if len(data_context.get('columns', [])) > 10 else ''}
- 데이터 타입: {len(data_context.get('dtypes', {}))}개 컬럼
"""
        
        # LLM 프롬프트 (AgentOrchestra 스타일)
        prompt = f"""
당신은 AI Data Science Team의 중앙 오케스트레이터입니다. 사용자의 요청을 분석하여 최적의 다단계 실행 계획을 생성해야 합니다.

사용자 요청: {user_query}

{data_info}

사용 가능한 전문 에이전트들:
{chr(10).join(agent_capabilities)}

다음 원칙에 따라 계획을 수립하세요:

1. **Hierarchical Decomposition**: 복잡한 작업을 논리적인 단계로 분해
2. **Agent Specialization**: 각 에이전트의 전문성을 최대한 활용
3. **Sequential Dependencies**: 단계 간 의존성 고려 (예: 데이터 로딩 → 정제 → 분석)
4. **Multimodal Support**: 텍스트, 차트, 리포트 등 다양한 출력 형태 고려
5. **Adaptive Planning**: 데이터 특성에 맞는 적응적 계획

JSON 형식으로 응답하세요:
{{
    "objective": "전체 목표 설명",
    "reasoning": "계획 수립 논리와 에이전트 선택 이유",
    "estimated_duration": "예상 소요 시간 (분)",
    "steps": [
        {{
            "step_number": 1,
            "agent_name": "선택된 에이전트명",
            "task_description": "구체적인 작업 설명",
            "expected_output": "예상 출력 형태",
            "dependencies": ["이전 단계 번호들"],
            "priority": "high|medium|low"
        }}
    ],
    "success_criteria": "성공 판단 기준",
    "fallback_strategy": "실패 시 대안"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert AI orchestrator for data science workflows."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            if plan_text.startswith("```json"):
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif plan_text.startswith("```"):
                plan_text = plan_text.split("```")[1].strip()
            
            plan = json.loads(plan_text)
            
            # 계획 검증 및 보완
            if not isinstance(plan.get("steps"), list):
                raise ValueError("Invalid plan structure: steps must be a list")
            
            # 선택된 에이전트 목록 생성
            plan["selected_agents"] = list(set(
                step["agent_name"] for step in plan["steps"] 
                if step["agent_name"] in available_agents
            ))
            
            logger.info(f"✅ Generated orchestration plan with {len(plan['steps'])} steps")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Failed to generate plan: {e}")
            # 폴백 계획
            return self._create_fallback_plan(user_query, available_agents)
    
    def _create_fallback_plan(self, user_query: str, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """LLM 실패 시 기본 폴백 계획"""
        return {
            "objective": f"사용자 요청 처리: {user_query}",
            "reasoning": "LLM 계획 생성 실패로 인한 기본 계획 적용",
            "estimated_duration": "5-10분",
            "steps": [
                {
                    "step_number": 1,
                    "agent_name": "eda_tools",
                    "task_description": "기본 탐색적 데이터 분석 수행",
                    "expected_output": "데이터 요약 및 시각화",
                    "dependencies": [],
                    "priority": "high"
                }
            ],
            "selected_agents": ["eda_tools"],
            "success_criteria": "기본 분석 완료",
            "fallback_strategy": "수동 분석 가이드 제공"
        }


class A2ATaskExecutor:
    """A2A 프로토콜 기반 작업 실행기"""
    
    async def execute_plan_step(
        self, 
        step: Dict[str, Any], 
        agent_info: Dict[str, Any],
        data_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """개별 계획 단계를 A2A 프로토콜로 실행"""
        
        try:
            # A2A 메시지 구성
            message_parts = [
                TextPart(text=step["task_description"])
            ]
            
            # 데이터 컨텍스트가 있으면 추가
            if data_context:
                context_text = f"\n\n데이터 컨텍스트:\n{json.dumps(data_context, ensure_ascii=False, indent=2)}"
                message_parts.append(TextPart(text=context_text))
            
            # A2A 요청 구성
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send", 
                "params": {
                    "message": {
                        "messageId": f"step_{step['step_number']}_{int(time.time())}",
                        "role": "user",
                        "parts": [part.model_dump() for part in message_parts]
                    }
                },
                "id": step["step_number"]
            }
            
            logger.info(f"🚀 Executing step {step['step_number']} with {step['agent_name']}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    agent_info["url"],
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Step {step['step_number']} completed successfully")
                    
                    return {
                        "status": "success",
                        "step_number": step["step_number"],
                        "agent_name": step["agent_name"],
                        "result": result,
                        "execution_time": time.time()
                    }
                else:
                    logger.error(f"❌ Step {step['step_number']} failed: HTTP {response.status_code}")
                    return {
                        "status": "error",
                        "step_number": step["step_number"],
                        "agent_name": step["agent_name"],
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "execution_time": time.time()
                    }
                    
        except Exception as e:
            logger.error(f"❌ Step {step['step_number']} execution failed: {e}")
            return {
                "status": "error",
                "step_number": step["step_number"],
                "agent_name": step["agent_name"],
                "error": str(e),
                "execution_time": time.time()
            }


class AIDataScienceOrchestrator:
    """AI Data Science Team 오케스트레이터 (AgentOrchestra 기반)"""
    
    def __init__(self):
        self.discovery_service = AgentDiscoveryService()
        self.planner = IntelligentPlanner()
        self.executor = A2ATaskExecutor()
    
    async def orchestrate(
        self, 
        user_query: str, 
        data_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """메인 오케스트레이션 로직"""
        
        start_time = time.time()
        
        try:
            # 1. 에이전트 발견
            available_agents = await self.discovery_service.discover_available_agents()
            
            if not available_agents:
                return {
                    "status": "error",
                    "error": "사용 가능한 에이전트가 없습니다",
                    "execution_time": time.time() - start_time
                }
            
            # 2. LLM 기반 계획 생성
            plan = await self.planner.create_orchestration_plan(
                user_query, available_agents, data_context
            )
            
            # 3. 계획 실행
            execution_results = []
            
            for step in plan["steps"]:
                agent_name = step["agent_name"]
                
                if agent_name not in available_agents:
                    logger.warning(f"⚠️ Agent {agent_name} not available, skipping step {step['step_number']}")
                    continue
                
                agent_info = available_agents[agent_name]
                
                step_result = await self.executor.execute_plan_step(
                    step, agent_info, data_context
                )
                
                execution_results.append(step_result)
                
                # 실패 시 조기 종료 (선택적)
                if step_result["status"] == "error" and step.get("priority") == "high":
                    logger.error(f"❌ Critical step {step['step_number']} failed, stopping execution")
                    break
            
            # 4. 결과 종합
            successful_steps = [r for r in execution_results if r["status"] == "success"]
            failed_steps = [r for r in execution_results if r["status"] == "error"]
            
            return {
                "status": "completed" if len(failed_steps) == 0 else "partial",
                "plan": plan,
                "execution_results": execution_results,
                "summary": {
                    "total_steps": len(plan["steps"]),
                    "successful_steps": len(successful_steps),
                    "failed_steps": len(failed_steps),
                    "execution_time": time.time() - start_time
                },
                "discovered_agents": len(available_agents)
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }


class AIDataScienceOrchestratorExecutor(AgentExecutor):
    """A2A SDK 기반 오케스트레이터 실행자"""
    
    def __init__(self):
        self.orchestrator = AIDataScienceOrchestrator()
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """A2A 요청 실행"""
        
        # 사용자 메시지 추출
        user_message = ""
        for part in context.message.parts:
            if hasattr(part, 'text'):
                user_message += part.text + " "
        
        user_message = user_message.strip()
        
        if not user_message:
            await event_queue.enqueue_event(
                new_agent_text_message("❌ 요청 메시지가 비어있습니다.")
            )
            return
        
        # 진행 상황 스트리밍
        await event_queue.enqueue_event(
            new_agent_text_message("🔍 사용 가능한 AI DS Team 에이전트들을 발견하고 있습니다...")
        )
        
        # 데이터 컨텍스트 준비 (실제 구현에서는 세션에서 가져오기)
        data_context = None  # TODO: 실제 데이터 컨텍스트 연결
        
        # 오케스트레이션 실행
        result = await self.orchestrator.orchestrate(user_message, data_context)
        
        # 계획 정보 스트리밍
        if result["status"] in ["completed", "partial"] and "plan" in result:
            plan = result["plan"]
            summary = result["summary"]
            
            await event_queue.enqueue_event(
                new_agent_text_message(f"📋 {result['discovered_agents']}개 에이전트 발견. 지능형 오케스트레이션 계획을 수립하고 있습니다...")
            )
            
            # 🔥 핵심 수정: 계획을 아티팩트로 반환
            plan_artifact = {
                "name": "orchestration_plan",
                "metadata": {
                    "content_type": "application/json",
                    "description": "AI DS Team 오케스트레이션 실행 계획",
                    "total_steps": summary["total_steps"],
                    "discovered_agents": result["discovered_agents"]
                },
                "parts": [TextPart(text=json.dumps(plan, ensure_ascii=False, indent=2))]
            }
            
            # 아티팩트 이벤트 생성 및 전송
            from a2a.server.events import ArtifactEvent
            await event_queue.enqueue_event(ArtifactEvent(artifact=plan_artifact))
            
        # 결과 요약 메시지 스트리밍
        if result["status"] == "completed":
            summary = result["summary"]
            response = f"✅ A2A 오케스트레이션 완료! {summary['total_steps']}단계 계획 수립됨"
        elif result["status"] == "partial":
            summary = result["summary"]
            response = f"⚠️ A2A 오케스트레이션 부분 완료! {summary['successful_steps']}/{summary['total_steps']} 단계 성공"
        else:
            response = f"❌ A2A 오케스트레이션 실패: {result.get('error', 'Unknown error')}"
        
        await event_queue.enqueue_event(new_agent_text_message(response))
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """A2A SDK 0.2.9 올바른 패턴으로 작업 취소"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        task_updater.update_status(
            state="cancelled",
            message=new_agent_text_message("❌ 오케스트레이션이 취소되었습니다.")
        )


def create_orchestrator_server():
    """오케스트레이터 서버 생성"""
    
    # AgentSkill 정의
    orchestration_skill = AgentSkill(
        id='ai_ds_team_orchestration',
        name='AI Data Science Team Orchestration',
        description='AI-driven orchestration of multi-agent data science workflows with dynamic agent discovery and intelligent task decomposition',
        tags=['orchestration', 'ai-driven', 'data-science', 'multi-agent', 'a2a-protocol'],
        examples=[
            'analyze my dataset comprehensively',
            'perform complete EDA on my data',
            'coordinate agents for data analysis',
            'create intelligent workflow for my data',
            'orchestrate specialized agents for insights',
            'plan multi-step analysis strategy'
        ]
    )
    
    # AgentCard 정의
    agent_card = AgentCard(
        name='Universal AI Data Science Orchestrator',
        description='An AI-driven orchestrator that dynamically discovers A2A agents and creates intelligent multi-agent collaboration plans using LLM reasoning.',
        url='http://localhost:8100/',
        version='2.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[orchestration_skill],
        supportsAuthenticatedExtendedCard=False,
    )
    
    # RequestHandler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=AIDataScienceOrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A 애플리케이션 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    return server


if __name__ == '__main__':
    # 서버 실행
    server = create_orchestrator_server()
    
    logger.info("🚀 Starting AI Data Science Team Orchestrator Server...")
    logger.info("📋 Agent Card: Universal AI Data Science Orchestrator")
    logger.info("🌐 URL: http://localhost:8100")
    logger.info("🎯 Capabilities: AI-driven multi-agent orchestration")
    
    uvicorn.run(
        server.build(), 
        host='0.0.0.0', 
        port=8100,
        log_level='info'
    ) 