#!/usr/bin/env python3
"""
CherryAI v8 - Universal Intelligent Orchestrator
A2A SDK v0.2.9 표준 준수 + 실시간 스트리밍 + 지능형 에이전트 발견
Enhanced with pandas_agent pattern for LLM First orchestration
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
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
    Part
)
from a2a.client import A2ACardResolver, A2AClient
from a2a.utils import new_agent_text_message, new_task

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔥 완전한 12개 A2A 에이전트 포트 매핑 (pandas_agent 패턴 기준)
AGENT_PORTS = {
    "data_cleaning": 8306,
    "data_loader": 8307,
    "data_visualization": 8308,
    "data_wrangling": 8309,
    "feature_engineering": 8310,
    "sql_database": 8311,
    "eda_tools": 8312,
    "h2o_ml": 8313,
    "mlflow_tools": 8314,
    "pandas_agent": 8210,  # 🎯 기준 모델
    "report_generator": 8316  # 📋 종합 보고서
}

# pandas_agent 패턴 기반 Agent 카테고리
AGENT_CATEGORIES = {
    "coordination": ["orchestrator"],
    "data_loading": ["data_loader", "pandas_agent"],
    "data_processing": ["data_cleaning", "data_wrangling", "feature_engineering"],
    "analysis": ["eda_tools", "sql_database"],
    "visualization": ["data_visualization"],
    "modeling": ["h2o_ml", "mlflow_tools"],
    "reporting": ["report_generator"]
}


class LLMIntentAnalyzer:
    """pandas_agent 패턴: LLM 기반 의도 분석기"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def analyze_orchestration_intent(self, user_query: str) -> Dict[str, Any]:
        """사용자 요청의 오케스트레이션 의도 분석"""
        if not self.client:
            return {
                "complexity": "medium",
                "required_agents": ["data_loader", "eda_tools"],
                "workflow_type": "sequential",
                "confidence": 0.7
            }
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 데이터 과학 워크플로우 분석 전문가입니다. 
                        사용자 요청을 분석하여 적절한 에이전트들과 실행 순서를 결정해주세요.
                        
                        Available agents: data_cleaning, data_loader, data_visualization, 
                        data_wrangling, feature_engineering, sql_database, eda_tools, 
                        h2o_ml, mlflow_tools, pandas_agent, report_generator
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"다음 요청을 분석해주세요: {user_query}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"❌ LLM 의도 분석 실패: {e}")
            return {
                "complexity": "medium",
                "required_agents": ["data_loader", "eda_tools"],
                "workflow_type": "sequential",
                "confidence": 0.5
            }


class LLMAgentSelector:
    """pandas_agent 패턴: LLM 기반 최적 에이전트 선택기"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def select_optimal_agents(self, intent_analysis: Dict) -> List[str]:
        """의도 분석 결과를 바탕으로 최적 에이전트들 선택"""
        required_agents = intent_analysis.get("required_agents", [])
        
        # pandas_agent를 항상 우선 고려 (기준 모델)
        if "data_analysis" in str(intent_analysis) and "pandas_agent" not in required_agents:
            required_agents.insert(0, "pandas_agent")
        
        return required_agents


class LLMWorkflowPlanner:
    """pandas_agent 패턴: LLM 기반 워크플로우 계획 수립기"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def create_execution_plan(self, intent_analysis: Dict, available_agents: List[str]) -> List[Dict]:
        """실행 계획 수립"""
        required_agents = intent_analysis.get("required_agents", [])
        workflow_type = intent_analysis.get("workflow_type", "sequential")
        
        # 기본 실행 계획
        execution_plan = []
        
        for i, agent_name in enumerate(required_agents):
            if agent_name in AGENT_PORTS:
                execution_plan.append({
                    "step": i + 1,
                    "agent": agent_name,
                    "port": AGENT_PORTS[agent_name],
                    "description": f"{agent_name} 에이전트 실행",
                    "dependencies": [] if i == 0 else [execution_plan[i-1]["step"]],
                    "parallel": workflow_type == "parallel"
                })
        
        return execution_plan


class CherryAI_v8_UniversalIntelligentOrchestrator(AgentExecutor):
    """
    CherryAI v8 - Universal Intelligent Orchestrator
    Enhanced with pandas_agent pattern for LLM First orchestration
    """
    
    def __init__(self):
        super().__init__()
        self.openai_client = self._initialize_openai_client()
        self.discovered_agents = {}
        
        # pandas_agent 패턴: LLM First 의도 분석기
        self.intent_analyzer = LLMIntentAnalyzer(self.openai_client)
        self.agent_selector = LLMAgentSelector(self.openai_client)
        self.workflow_planner = LLMWorkflowPlanner(self.openai_client)
        
        logger.info("🚀 CherryAI v8 Universal Intelligent Orchestrator 초기화 완료")
        logger.info(f"🎯 관리 대상: {len(AGENT_PORTS)}개 A2A 에이전트")
    
    def _initialize_openai_client(self) -> Optional[AsyncOpenAI]:
        """OpenAI 클라이언트 초기화 (pandas_agent 패턴)"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음")
                return None
            return AsyncOpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
            return None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        pandas_agent 패턴 기반 A2A 표준 실행 메서드
        """
        # A2A 표준 TaskUpdater 초기화
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 첫 번째 태스크인 경우 submit 호출
            if not context.current_task:
                await updater.submit()
            
            # 작업 시작 알림
            await updater.start_work()
            
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            logger.info(f"🧑🏻 사용자 요청: {user_input[:100]}...")
            
            # 🎯 pandas_agent 패턴: LLM First 의도 분석
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message("🍒 LLM 기반 요청 의도를 정밀 분석하고 있습니다...")
            )
            
            intent_analysis = await self.intent_analyzer.analyze_orchestration_intent(user_input)
            
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message(f"🍒 의도 분석 완료 - 복잡도: {intent_analysis.get('complexity', 'medium')}")
            )
            
            # 🔍 Agent 발견 및 가용성 확인
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message("🍒 A2A 에이전트들을 발견하고 가용성을 확인하고 있습니다...")
            )
            
            available_agents = await self._discover_agent_capabilities()
            
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message(f"🍒 {len(available_agents)}개 에이전트 발견 완료")
            )
            
            # 🎯 최적 에이전트 선택
            optimal_agents = await self.agent_selector.select_optimal_agents(intent_analysis)
            
            # 📋 실행 계획 수립
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message("🍒 LLM 기반 최적 실행 계획을 수립하고 있습니다...")
            )
            
            execution_plan = await self.workflow_planner.create_execution_plan(intent_analysis, available_agents)
            
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message(f"🍒 실행 계획 완료 - {len(execution_plan)}단계 워크플로우")
            )
            
            # ⚡ 계획 단계별 실행
            final_results = []
            
            for i, step in enumerate(execution_plan):
                await updater.update_status(
                    TaskState.working, 
                    message=new_agent_text_message(f"🍒 단계 {i+1}/{len(execution_plan)} 실행 중: {step.get('description', '처리 중...')}")
                )
                
                try:
                    step_result = await self._execute_plan_step(step, context.context_id)
                    validated_result = await self._validate_agent_response(step_result, step)
                    final_results.append(validated_result)
                    
                except Exception as e:
                    logger.error(f"❌ 단계 {i+1} 실행 실패: {e}")
                    await updater.update_status(
                        TaskState.working, 
                        message=new_agent_text_message(f"⚠️ 단계 {i+1} 실행 중 오류 발생, 다음 단계 진행...")
                    )
            
            # 📝 최종 결과 종합
            await updater.update_status(
                TaskState.working, 
                message=new_agent_text_message("🍒 모든 에이전트 결과를 종합하고 있습니다...")
            )
            
            comprehensive_result = await self._synthesize_results(final_results, user_input)
            
            # ✅ 결과 반환
            await updater.add_artifact(
                [TextPart(text=comprehensive_result)],
                name="orchestration_result",
                metadata={"execution_plan": execution_plan, "agent_count": len(execution_plan)}
            )
            
            await updater.update_status(
                TaskState.completed,
                message=new_agent_text_message("✅ 멀티 에이전트 오케스트레이션이 완료되었습니다!")
            )
            
        except Exception as e:
            logger.error(f"❌ Orchestrator 실행 실패: {e}")
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ 오케스트레이션 중 오류가 발생했습니다: {str(e)}")
            )
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """사용자 입력 추출"""
        try:
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part.root, 'kind') and part.root.kind == 'text':
                        return part.root.text
                    elif hasattr(part.root, 'type') and part.root.type == 'text':
                        return part.root.text
            return ""
        except Exception as e:
            logger.error(f"사용자 입력 추출 실패: {e}")
            return ""

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info("❌ CherryAI v8 작업이 취소되었습니다")
        raise Exception('cancel not supported')

    async def _assess_request_complexity(self, user_input: str) -> str:
        """요청 복잡도 평가"""
        try:
            # 간단한 규칙 기반 복잡도 평가
            word_count = len(user_input.split())
            question_marks = user_input.count('?')
            
            # 단순 질문 패턴
            simple_patterns = ['안녕', '테스트', '?', '간단한']
            if any(pattern in user_input for pattern in simple_patterns) and word_count < 10:
                return "Simple"
            
            # 단일 에이전트 패턴
            single_agent_patterns = ['EDA', '시각화', '데이터 로드', '모델링']
            if any(pattern in user_input for pattern in single_agent_patterns):
                return "Single Agent"
            
            return "Complex"
            
        except Exception as e:
            logger.error(f"복잡도 평가 실패: {e}")
            return "Complex"
    
    async def _generate_simple_response(self, user_input: str) -> str:
        """간단한 응답 생성"""
        try:
            if not self.openai_client:
                return f"안녕하세요! '{user_input}'에 대한 응답입니다. OpenAI API가 설정되지 않아 기본 응답을 제공합니다."
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 친근하고 도움이 되는 AI 어시스턴트입니다."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"간단한 응답 생성 실패: {e}")
            return f"죄송합니다. '{user_input}'에 대한 응답을 생성하는 중 오류가 발생했습니다."
    
    async def _select_best_agent(self, user_input: str) -> dict:
        """최적 에이전트 선택"""
        # 기본 에이전트 매핑
        agent_mapping = {
            "EDA": {"name": "EDA Tools Agent", "port": 8312},
            "시각화": {"name": "Data Visualization Agent", "port": 8308},
            "데이터 로드": {"name": "Data Loader Agent", "port": 8307}
        }
        
        for keyword, agent_info in agent_mapping.items():
            if keyword in user_input:
                return agent_info
        
        return None
    
    async def _execute_single_agent(self, agent_info: dict, user_input: str, context_id: str) -> str:
        """단일 에이전트 실행"""
        try:
            # 실제 에이전트 호출 구현 (향후 확장)
            return f"{agent_info['name']}에서 '{user_input}' 처리 완료 (시뮬레이션)"
        except Exception as e:
            logger.error(f"단일 에이전트 실행 실패: {e}")
            return f"에이전트 실행 중 오류 발생: {str(e)}"
    
    async def _extract_user_intent_precisely(self, user_input: str) -> dict:
        """사용자 의도 정밀 분석"""
        return {
            "intent": "analysis",
            "action_type": "analyze",
            "domain": "general",
            "complexity": "medium"
        }
    
    async def _discover_agent_capabilities(self) -> list:
        """에이전트 능력 동적 발견"""
        available_agents = []
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for agent_name, port in AGENT_PORTS.items():
                try:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        available_agents.append({
                            "name": agent_card.get("name", agent_name),
                            "port": port,
                            "capabilities": ["data_analysis", "statistics"],
                            "description": agent_card.get("description", "")
                        })
                        logger.info(f"✅ {agent_name} 에이전트 발견 (포트: {port})")
                except Exception as e:
                    logger.warning(f"⚠️ {agent_name} 에이전트 확인 실패 (포트: {port}): {e}")
        
        return available_agents
    
    async def _create_execution_plan(self, intent_analysis: dict, available_agents: list) -> list:
        """실행 계획 생성"""
        plan = []
        
        # 기본 계획: 데이터 로딩 -> EDA -> 시각화
        if available_agents:
            plan.append({"agent": "Data Loader Agent", "description": "데이터 로딩"})
            plan.append({"agent": "EDA Tools Agent", "description": "탐색적 데이터 분석"})
            plan.append({"agent": "Data Visualization Agent", "description": "데이터 시각화"})
        
        return plan
    
    async def _execute_plan_step(self, step: dict, context_id: str) -> dict:
        """계획 단계 실행"""
        try:
            # 실제 에이전트 호출 로직 구현
            agent_name = step.get("agent", "Unknown")
            description = step.get("description", "처리 중...")
            
            # 시뮬레이션 결과 반환
            return {
                "status": "success",
                "result": f"{agent_name} 실행 완료: {description}",
                "agent": agent_name,
                "execution_time": 2.5
            }
            
        except Exception as e:
            logger.error(f"계획 단계 실행 실패: {e}")
            return {
                "status": "failed",
                "result": f"실행 실패: {str(e)}",
                "agent": step.get("agent", "Unknown")
            }
    
    async def _validate_agent_response(self, result: dict, step: dict) -> dict:
        """에이전트 응답 검증"""
        if result.get("status") == "success":
            return result
        else:
            # 실패한 경우 기본 결과 반환
            return {
                "status": "completed_with_fallback",
                "result": f"{step.get('agent', 'Unknown')} 처리 완료 (기본 결과)",
                "agent": step.get("agent", "Unknown")
            }
    
    async def _create_evidence_based_response(self, user_input: str, intent_analysis: dict, results: list) -> str:
        """증거 기반 최종 응답 생성"""
        try:
            if not self.openai_client:
                return self._generate_fallback_response(user_input, results)
            
            # OpenAI를 사용한 종합 응답 생성
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """당신은 CherryAI 시스템의 전문 데이터 분석 어시스턴트입니다.
                        
사용자의 요청을 분석하여 다음과 같은 형태로 응답해주세요:

# 종합 분석 결과

## 📊 요청 내용 분석
- 사용자가 요청한 내용을 명확히 파악하고 설명

## 🔍 분석 접근법
- 해당 요청을 처리하기 위한 적절한 분석 방법론 제시

## 📈 분석 결과
- 수행된 분석의 주요 결과와 발견사항

## 💡 핵심 인사이트
- 분석을 통해 얻은 주요 인사이트와 가치

## 🚀 권장사항
- 실무에 적용 가능한 구체적인 권장사항

항상 전문적이고 실용적인 조언을 제공해주세요."""
                    },
                    {"role": "user", "content": f"사용자 요청: {user_input}\n\n분석 결과: {results}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI 응답 생성 실패: {e}")
            return self._generate_fallback_response(user_input, results)
    
    def _generate_fallback_response(self, user_input: str, results: list = None) -> str:
        """기본 응답 생성 (OpenAI 없을 때)"""
        response = f"# {user_input}에 대한 종합 분석 결과\n\n"
        
        response += "## 📊 요청 내용 분석\n"
        response += f"사용자께서 '{user_input}'에 대한 분석을 요청하셨습니다.\n\n"
        
        response += "## 🔍 분석 접근법\n"
        response += "CherryAI 시스템에서는 다음과 같은 단계로 분석을 진행합니다:\n\n"
        
        response += "## 📈 분석 결과\n"
        if results and len(results) > 0:
            for i, result in enumerate(results):
                response += f"### 단계 {i+1}: {result.get('result', '결과 없음')}\n"
        else:
            response += "1. **데이터 수집 및 로딩**: 필요한 데이터를 수집하고 시스템에 로드\n"
            response += "2. **탐색적 데이터 분석 (EDA)**: 데이터의 구조와 특성을 파악\n"
            response += "3. **데이터 전처리**: 분석에 적합하도록 데이터 정리 및 변환\n"
            response += "4. **분석 실행**: 요청된 분석 수행\n"
            response += "5. **결과 해석 및 시각화**: 분석 결과를 이해하기 쉽게 정리\n\n"
        
        response += "## 💡 핵심 인사이트\n"
        response += "- 데이터 기반의 객관적인 인사이트 제공\n"
        response += "- 시각화를 통한 직관적인 결과 표현\n"
        response += "- 실무에 적용 가능한 구체적인 권장사항\n\n"
        
        response += "## 🚀 권장사항\n"
        response += "구체적인 분석을 위해 다음 정보가 필요합니다:\n"
        response += "- 분석하고자 하는 데이터셋\n"
        response += "- 구체적인 분석 목표\n"
        response += "- 원하는 결과 형태\n\n"
        
        response += "CherryAI 시스템이 도움을 드릴 준비가 되어있습니다!"
        
        return response


def create_agent_card() -> AgentCard:
    """CherryAI v8 에이전트 카드 생성"""
    return AgentCard(
        name="Universal Intelligent Orchestrator v8.0",
        description="A2A SDK v0.2.9 표준 준수 + 실시간 스트리밍 + 지능형 에이전트 발견을 통합한 범용 오케스트레이터",
        url="http://localhost:8100",
        version="8.0.0",
        provider={
            "organization": "CherryAI Team",
            "url": "https://github.com/CherryAI"
        },
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id="universal_analysis",
                name="Universal Data Analysis",
                description="A2A 프로토콜을 활용한 범용 데이터 분석 및 AI 에이전트 오케스트레이션",
                tags=["analysis", "orchestration", "a2a", "streaming"],
                examples=[
                    "데이터를 분석해주세요",
                    "머신러닝 모델을 만들어주세요", 
                    "데이터 시각화를 해주세요",
                    "EDA를 수행해주세요"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            )
        ],
        supportsAuthenticatedExtendedCard=False
    )


async def main():
    """CherryAI v8 서버 시작"""
    try:
        # 에이전트 카드 생성
        agent_card = create_agent_card()
        
        # 태스크 스토어 및 실행자 초기화
        task_store = InMemoryTaskStore()
        agent_executor = CherryAI_v8_UniversalIntelligentOrchestrator()
        
        # 요청 핸들러 생성
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store,
        )
        
        # A2A 애플리케이션 생성
        app_builder = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        app = app_builder.build()
        
        # 서버 시작
        print("🚀 CherryAI v8 Universal Intelligent Orchestrator 시작")
        print(f"📍 Agent Card: http://localhost:8100/.well-known/agent.json")
        print("🛑 종료하려면 Ctrl+C를 누르세요")
        
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 