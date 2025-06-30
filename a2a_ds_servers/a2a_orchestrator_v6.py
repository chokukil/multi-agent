#!/usr/bin/env python3
"""
A2A Orchestrator v6.0 - LLM Powered Dynamic Context-Aware Orchestrator
LLM 기반 동적 컨텍스트 인식 오케스트레이터
"""

import asyncio
import json
import logging
import os
import re
import time
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
    TextPart
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI DS Team 에이전트 포트 매핑
AGENT_PORTS = {
    "data_cleaning": 8306,
    "data_loader": 8307, 
    "data_visualization": 8308,
    "data_wrangling": 8309,
    "eda_tools": 8310,
    "feature_engineering": 8311,
    "h2o_modeling": 8312,
    "mlflow_tracking": 8313,
    "sql_database": 8314
}


class StreamingTaskUpdater(TaskUpdater):
    """실시간 스트리밍 업데이트 지원"""
    
    async def stream_update(self, content: str):
        """중간 결과 스트리밍"""
        await self.update_status(
            TaskState.working,
            message=self.new_agent_message(parts=[TextPart(text=content)])
        )
    
    async def stream_final_response(self, response: str):
        """최종 응답을 청크로 나누어 스트리밍"""
        # Markdown 섹션별로 스트리밍
        sections = response.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                await self.update_status(
                    TaskState.working,
                    message=self.new_agent_message(parts=[TextPart(text=section)])
                )
                await asyncio.sleep(0.1)  # 부드러운 스트리밍
        
        # 완료 상태 업데이트
        await self.update_status(
            TaskState.completed,
            message=self.new_agent_message(parts=[TextPart(text="✅ 분석이 완료되었습니다.")])
        )


class LLMPoweredOrchestratorExecutor(AgentExecutor):
    """LLM 기반 동적 컨텍스트 인식 오케스트레이터"""
    
    def __init__(self):
        # OpenAI 클라이언트 초기화 (옵션)
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 LLM Powered Dynamic Orchestrator v6 with OpenAI integration")
            else:
                self.openai_client = None
                logger.info("📊 Standard Orchestrator v6 (OpenAI API key not found)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        self.available_agents = {}
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """LLM이 전체 프로세스를 이해하고 조정하는 실행"""
        task_updater = StreamingTaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            logger.info(f"📥 Processing orchestration query: {user_input}")
            
            if not user_input:
                user_input = "Please provide an analysis request."
            
            # 에이전트 발견
            await task_updater.stream_update("🔍 AI DS Team 에이전트들을 발견하고 있습니다...")
            self.available_agents = await self._discover_agents()
            
            if not self.available_agents:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용 가능한 A2A 에이전트를 찾을 수 없습니다.")])
                )
                return
            
            await task_updater.stream_update(f"✅ {len(self.available_agents)}개 에이전트 발견 완료")
            
            # 1. 사용자 입력을 LLM이 완전히 이해
            await task_updater.stream_update("🧠 사용자 요청을 심층 분석하고 있습니다...")
            request_understanding = await self._understand_request(user_input)
            
            # 🎯 NEW: 질문에서 답변 구조 추출
            await task_updater.stream_update("🎯 질문에서 필요한 답변 구조를 추출하고 있습니다...")
            answer_structure = await self._extract_answer_structure_from_question(user_input)
            
            # 도메인 자동 적응
            domain_adaptation = await self._auto_adapt_to_domain(user_input)
            
            # 2. 🎯 NEW: 구조를 고려한 동적 실행 계획 생성
            await task_updater.stream_update("📋 답변 구조에 맞춘 실행 계획을 생성하고 있습니다...")
            execution_plan = await self._create_structure_aware_plan(
                request_understanding, 
                answer_structure,
                self.available_agents
            )
            
            if not execution_plan or not execution_plan.get('steps'):
                execution_plan = self._create_fallback_plan(self.available_agents)
            
            await task_updater.stream_update(f"✅ {len(execution_plan.get('steps', []))}단계 실행 계획 완료")
            
            # 📋 예쁜 계획 표시를 위한 스트리밍 메시지
            plan_display = self._create_beautiful_plan_display(execution_plan, request_understanding)
            await task_updater.stream_update(plan_display)
            
            # 계획을 아티팩트로 전송 (클라이언트 파싱용)
            plan_artifact = {
                "plan_executed": [
                    {
                        "step": i + 1,
                        "agent": step.get('agent', step.get('agent_name', 'unknown')),
                        "task_description": step.get('enriched_task', step.get('purpose', '')),
                        "reasoning": step.get('purpose', ''),
                        "expected_output": step.get('expected_output', '')
                    }
                    for i, step in enumerate(execution_plan.get('steps', []))
                ]
            }
            
            # 📋 실행 계획을 Artifact로 먼저 전송
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(plan_artifact, ensure_ascii=False, indent=2))],
                name="execution_plan.json",
                metadata={
                    "content_type": "application/json",
                    "plan_type": "ai_ds_team_orchestration",
                    "description": "LLM 기반 동적 실행 계획"
                }
            )
            
            # 계획 확인 시간 제공
            await asyncio.sleep(2)
            await task_updater.stream_update("🚀 실행 계획에 따라 분석을 시작합니다...")
            
            # 3. 각 에이전트 실행 (컨텍스트 전달)
            agent_results = {}
            for i, step in enumerate(execution_plan.get('steps', [])):
                step_num = i + 1
                agent_name = step.get('agent', step.get('agent_name', 'unknown'))
                
                # 단계별 상세 정보 표시
                step_info = f"""
🔄 **단계 {step_num}/{len(execution_plan.get('steps', []))}: {agent_name} 실행**

📝 **작업**: {step.get('enriched_task', step.get('purpose', ''))[:100]}...
🎯 **목적**: {step.get('purpose', '')}
"""
                await task_updater.stream_update(step_info)
                
                try:
                    # 🎯 NEW: 구조 컨텍스트와 함께 에이전트 실행
                    result = await self._execute_agent_with_structure_context(
                        agent_name=agent_name,
                        step=step,
                        answer_structure=answer_structure,
                        full_context=request_understanding,
                        previous_results=agent_results
                    )
                    agent_results[agent_name] = result
                    
                    # 실시간 피드백
                    summary = result.get('summary', 'Processing completed')
                    success_msg = f"✅ **{agent_name} 완료**: {summary}"
                    await task_updater.stream_update(success_msg)
                    
                except Exception as agent_error:
                    logger.warning(f"Agent {agent_name} execution failed: {agent_error}")
                    agent_results[agent_name] = {
                        'status': 'failed',
                        'error': str(agent_error),
                        'summary': f"에이전트 실행 중 오류 발생: {str(agent_error)}"
                    }
                    error_msg = f"⚠️ **{agent_name} 오류**: {str(agent_error)[:100]}... (계속 진행)"
                    await task_updater.stream_update(error_msg)
            
            # 4. LLM이 모든 결과를 종합하여 최종 답변 생성
            await task_updater.stream_update("🎯 모든 분석 결과를 종합하여 최종 답변을 생성하고 있습니다...")
            
            final_response = await self._synthesize_with_llm(
                original_request=user_input,
                understanding=request_understanding,
                all_results=agent_results,
                task_updater=task_updater
            )
            
            # 📊 최종 답변을 먼저 스트리밍으로 표시
            final_display = self._create_beautiful_final_display(
                final_response, 
                execution_plan, 
                agent_results, 
                request_understanding
            )
            await task_updater.stream_update(final_display)
            
            # 📊 최종 답변을 Artifact로도 전송
            await task_updater.add_artifact(
                parts=[TextPart(text=final_response)],
                name="final_analysis_report.md",
                metadata={
                    "content_type": "text/markdown",
                    "report_type": "comprehensive_analysis",
                    "description": "LLM 기반 종합 분석 보고서"
                }
            )
            
            # 5. 완료 메시지 (더 상세하고 예쁘게)
            completion_summary = f"""## 🎉 LLM 기반 동적 오케스트레이션 완료

### 📊 실행 결과 요약
- **🤖 에이전트 발견**: {len(self.available_agents)}개
- **📋 실행 계획**: {len(execution_plan.get('steps', []))}단계
- **✅ 성공한 에이전트**: {len([r for r in agent_results.values() if r.get('status') == 'success'])}개
- **❌ 실패한 에이전트**: {len([r for r in agent_results.values() if r.get('status') == 'failed'])}개
- **🏢 도메인**: {request_understanding.get('domain', '데이터 분석')}
- **📈 분석 깊이**: {request_understanding.get('analysis_depth', 'intermediate')}

### 📋 생성된 아티팩트
1. **execution_plan.json**: 동적 실행 계획 (JSON 형식)
2. **final_analysis_report.md**: 종합 분석 보고서 (Markdown 형식)

### 🎯 분석 완료
모든 분석이 완료되었습니다. 위의 상세한 결과와 아티팩트에서 전체 분석 내용을 확인하세요.

---
*🤖 Powered by LLM Dynamic Context-Aware Orchestrator v6*"""
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=completion_summary)])
            )
            
        except Exception as e:
            logger.error(f"Error in LLM Powered Orchestrator: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"오류 발생: {str(e)}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

    async def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """A2A 표준 에이전트 발견"""
        available_agents = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:  # 타임아웃 증가
            for agent_name, port in AGENT_PORTS.items():
                try:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        available_agents[agent_name] = {
                            "name": agent_card.get("name", agent_name),
                            "url": f"http://localhost:{port}",
                            "port": port,
                            "description": agent_card.get("description", ""),
                            "status": "available"
                        }
                        logger.info(f"✅ {agent_name} agent discovered on port {port}")
                except Exception as e:
                    logger.warning(f"⚠️ {agent_name} agent on port {port} not available: {e}")
        
        logger.info(f"🔍 Total discovered agents: {len(available_agents)}")
        return available_agents

    async def _understand_request(self, user_input: str) -> Dict[str, Any]:
        """강화된 LLM 기반 요청 이해 - 사용자 의도를 완전히 파악"""
        
        if not self.openai_client:
            return {
                "domain": "general",
                "analysis_type": "exploratory",
                "analysis_depth": "intermediate",
                "tone": "technical",
                "intent_category": "exploratory_analysis",
                "specific_questions": [user_input],
                "business_context": "일반적인 데이터 분석 요구"
            }

        understanding_prompt = f"""다음 사용자 요청을 깊이 분석하여 완전히 이해하세요:

"{user_input}"

사용자의 명시적/암시적 요구사항을 모두 파악하고, 
어떤 종류의 분석과 결과물을 원하는지 정확히 판단하세요.

특히 다음을 중점적으로 분석하세요:
1. 사용자가 속한 도메인/업계는 무엇인가?
2. 어떤 수준의 전문성을 가지고 있는가?
3. 어떤 구체적인 문제를 해결하려고 하는가?
4. 어떤 형태의 결과물을 기대하는가?
5. 비즈니스/업무 컨텍스트는 무엇인가?

JSON 형식으로 응답하세요:
{{
    "domain": "감지된 도메인 (예: semiconductor, finance, healthcare, general)",
    "analysis_type": "분석 유형 (descriptive/diagnostic/predictive/prescriptive)",
    "analysis_depth": "분석 깊이 (basic/intermediate/expert)",
    "urgency": "긴급도 (low/medium/high)",
    "tone": "적절한 응답 톤 (casual/professional/technical/academic)",
    "intent_category": "의도 카테고리",
    "specific_questions": ["사용자가 답을 원하는 구체적 질문들"],
    "business_context": "비즈니스/업무 맥락",
    "expertise_claimed": "사용자가 주장하는 전문성 수준",
    "expected_deliverables": ["기대하는 결과물 유형들"],
    "stakeholder_considerations": ["고려해야 할 이해관계자들"]
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": understanding_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            understanding = json.loads(response.choices[0].message.content)
            logger.info(f"📋 Request understanding: {understanding.get('domain')} domain, {understanding.get('analysis_depth')} depth")
            return understanding
            
        except Exception as e:
            logger.warning(f"Request understanding failed: {e}")
            return {
                "domain": "general",
                "analysis_type": "exploratory", 
                "analysis_depth": "intermediate",
                "tone": "technical",
                "intent_category": "exploratory_analysis",
                "specific_questions": [user_input],
                "business_context": "일반적인 데이터 분석 요구"
            }

    async def _extract_answer_structure_from_question(self, user_input: str) -> Dict:
        """🎯 NEW: 사용자 질문에서 필요한 답변 구조를 동적으로 추출"""
        
        if not self.openai_client:
            return {
                "required_sections": [
                    {
                        "name": "분석 결과",
                        "purpose": "데이터 기반 분석 수행",
                        "required_data": ["기본 통계", "패턴 분석"],
                        "expected_format": "텍스트"
                    }
                ],
                "overall_structure": "기본 분석 보고서",
                "key_questions_to_answer": [user_input]
            }
        
        structure_extraction_prompt = f"""
        사용자 질문을 분석하여 어떤 구조의 답변을 원하는지 파악하세요.
        
        사용자 질문: "{user_input}"
        
        질문에서 요구하는 것들을 추출하여 답변 구조를 생성하세요.
        예를 들어:
        - "공정 이상 여부를 판단하고" → 이상 여부 진단 섹션 필요
        - "원인을 설명하며" → 원인 분석 섹션 필요  
        - "조치 방향을 제안" → 조치 방안 섹션 필요
        
        하지만 이것은 예시일 뿐입니다. 
        사용자가 "트렌드를 분석해줘"라고 하면 트렌드 분석 구조가 필요하고,
        "A와 B를 비교해줘"라고 하면 비교 분석 구조가 필요합니다.
        
        JSON 형식으로 응답:
        {{
            "required_sections": [
                {{
                    "name": "섹션 이름",
                    "purpose": "이 섹션의 목적",
                    "required_data": ["필요한 데이터 유형들"],
                    "expected_format": "표/그래프/텍스트/목록 등"
                }}
            ],
            "overall_structure": "전체적인 답변 흐름",
            "key_questions_to_answer": ["답해야 할 핵심 질문들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": structure_extraction_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            answer_structure = json.loads(response.choices[0].message.content)
            logger.info(f"🎯 Answer structure extracted: {len(answer_structure.get('required_sections', []))} sections")
            return answer_structure
            
        except Exception as e:
            logger.warning(f"Answer structure extraction failed: {e}")
            return {
                "required_sections": [
                    {
                        "name": "분석 결과",
                        "purpose": "데이터 기반 분석 수행",
                        "required_data": ["기본 통계", "패턴 분석"],
                        "expected_format": "텍스트"
                    }
                ],
                "overall_structure": "기본 분석 보고서",
                "key_questions_to_answer": [user_input]
            }

    async def _auto_adapt_to_domain(self, user_input: str) -> Dict:
        """어떤 도메인이든 자동으로 적응"""
        
        if not self.openai_client:
            return {"adaptation": "기본 데이터 분석 접근법"}
        
        adaptation_prompt = f"""
다음 요청을 분석하여 자동으로 도메인과 필요한 분석 방법을 파악하세요:

{user_input}

이 요청이 어떤 도메인인지, 어떤 종류의 분석이 필요한지, 
어떤 전문 지식이 필요한지 파악하여 최적의 접근 방법을 제시하세요.

특히 사용자가 특정 역할이나 전문성을 언급했다면, 
그에 맞는 분석 깊이와 용어를 사용해야 합니다.

JSON 형식으로 응답하세요:
{{
    "detected_domain": "감지된 도메인",
    "required_expertise": "필요한 전문성",
    "analysis_approach": "권장 분석 접근법",
    "terminology_level": "용어 수준 (basic/intermediate/expert)",
    "special_considerations": ["특별 고려사항들"]
}}
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": adaptation_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0  # 타임아웃 증가
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Domain adaptation failed: {e}")
            return {"adaptation": "기본 데이터 분석 접근법"}

    async def _create_structure_aware_plan(self, understanding: Dict, 
                                          answer_structure: Dict,
                                          available_agents: Dict) -> Dict:
        """완전 LLM 기반 동적 계획 생성 - 하드코딩 제거, 범용적 접근"""
        
        if not self.openai_client:
            return self._create_fallback_plan(available_agents)
        
        # 에이전트들의 상세 정보 구조화
        agents_details = {}
        for name, info in available_agents.items():
            agents_details[name] = {
                "description": info.get('description', ''),
                "capabilities": info.get('capabilities', []),
                "typical_use_cases": info.get('use_cases', [])
            }
        
        planning_prompt = f"""당신은 데이터 분석 워크플로우 설계 전문가입니다.
사용자의 요청을 분석하고, 가장 효과적인 에이전트 실행 계획을 수립해야 합니다.

## 📋 사용자 요청 분석 결과
{json.dumps(understanding, ensure_ascii=False, indent=2)}

## 🤖 사용 가능한 에이전트들
{json.dumps(agents_details, ensure_ascii=False, indent=2)}

## 🎯 계획 수립 지침
1. **요청 중심 접근**: 사용자가 원하는 결과에 집중하여 필요한 에이전트만 선택
2. **논리적 순서**: 데이터 흐름과 의존성을 고려한 순서 결정
3. **효율성 최적화**: 불필요한 단계 제거, 핵심 분석에 집중
4. **도메인 적응**: {understanding.get('domain', '일반')} 도메인 특성 반영
5. **사용자 수준 고려**: {understanding.get('expertise_claimed', '일반 사용자')} 수준에 맞는 분석 깊이

## 🚀 동적 에이전트 선택 기준
- 사용자 질문의 핵심 의도가 무엇인가?
- 어떤 종류의 분석이 실제로 필요한가?
- 각 에이전트가 제공할 수 있는 가치는 무엇인가?
- 최소한의 단계로 최대한의 인사이트를 얻으려면?

다음 JSON 형식으로 응답하세요:
{{
    "analysis_strategy": "이 요청에 대한 전체적 분석 전략",
    "agent_selection_reasoning": "선택된 에이전트들과 그 이유",
    "steps": [
        {{
            "step_number": 1,
            "agent": "선택된_에이전트명",
            "purpose": "이 단계의 구체적 목적",
            "enriched_task": "에이전트에게 전달할 상세한 작업 지시 (도메인 컨텍스트 포함)",
            "expected_output": "이 단계에서 기대하는 구체적 결과",
            "success_criteria": "성공 판단 기준",
            "context_for_next": "다음 단계로 전달할 핵심 정보"
        }}
    ],
    "final_synthesis_strategy": "모든 결과를 어떻게 종합할 것인가",
    "potential_insights": "이 계획으로 얻을 수 있는 예상 인사이트들"
}}

중요: 템플릿이나 고정된 패턴을 사용하지 말고, 이 특정 요청에 최적화된 계획을 수립하세요."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 데이터 분석 워크플로우 최적화 전문가입니다. 각 요청의 고유한 특성을 파악하고, 가장 효율적이고 효과적인 분석 경로를 설계합니다. 하드코딩된 템플릿이 아닌, 요청별 맞춤형 접근을 사용합니다."
                    },
                    {"role": "user", "content": planning_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,  # 창의적 계획 수립을 위해 약간 높임
                max_tokens=3000,
                timeout=90.0
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            # 계획 검증 및 보정
            validated_plan = self._validate_and_enhance_plan(plan, available_agents, understanding)
            
            return validated_plan
            
        except Exception as e:
            logger.warning(f"Dynamic planning failed: {e}")
            return self._create_fallback_plan(available_agents)

    def _validate_and_enhance_plan(self, plan: Dict, available_agents: Dict, understanding: Dict) -> Dict:
        """생성된 계획을 검증하고 보강"""
        try:
            # 기본 구조 검증
            if not plan.get('steps'):
                logger.warning("Plan has no steps, using fallback")
                return self._create_fallback_plan(available_agents)
            
            # 에이전트 존재 여부 확인 및 보정
            valid_steps = []
            for step in plan.get('steps', []):
                agent_name = step.get('agent', '')
                if agent_name in available_agents:
                    # 필수 필드 보완
                    enhanced_step = {
                        "agent": agent_name,
                        "purpose": step.get('purpose', f'{agent_name} 분석 수행'),
                        "enriched_task": step.get('enriched_task', step.get('purpose', f'{agent_name} 작업 수행')),
                        "expected_output": step.get('expected_output', f'{agent_name} 분석 결과'),
                        "pass_to_next": step.get('context_for_next', step.get('pass_to_next', ['분석 결과']))
                    }
                    valid_steps.append(enhanced_step)
                else:
                    logger.warning(f"Agent {agent_name} not available, skipping step")
            
            if not valid_steps:
                logger.warning("No valid steps after validation, using fallback")
                return self._create_fallback_plan(available_agents)
            
            # 향상된 계획 반환
            enhanced_plan = {
                "reasoning": plan.get('analysis_strategy', plan.get('reasoning', '사용자 요청에 최적화된 분석 워크플로우')),
                "steps": valid_steps,
                "final_synthesis_guide": plan.get('final_synthesis_strategy', plan.get('final_synthesis_guide', '모든 결과를 종합하여 사용자에게 유용한 인사이트 제공')),
                "potential_insights": plan.get('potential_insights', ['데이터 기반 인사이트 도출']),
                "agent_selection_reasoning": plan.get('agent_selection_reasoning', '요청에 최적화된 에이전트 선택')
            }
            
            return enhanced_plan
            
        except Exception as e:
            logger.error(f"Plan validation failed: {e}")
            return self._create_fallback_plan(available_agents)

    async def _execute_agent_with_structure_context(self, agent_name: str, step: Dict, 
                                                    answer_structure: Dict,
                                                    full_context: Dict,
                                                    previous_results: Dict) -> Dict:
        """각 에이전트에 풍부한 컨텍스트와 함께 작업 전달"""
        
        if agent_name not in self.available_agents:
            return {
                'status': 'failed',
                'error': f'Agent {agent_name} not available',
                'summary': f'에이전트 {agent_name}를 찾을 수 없습니다'
            }
        
        # LLM이 이전 결과를 바탕으로 현재 에이전트의 작업을 보강
        enriched_prompt = await self._enrich_agent_task(
            agent_name, step.get('enriched_task', step.get('purpose', '')),
            full_context, previous_results
        )
        
        # 에이전트 실행
        agent_url = self.available_agents[agent_name]['url']
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"req_{agent_name}_{int(time.time())}",
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": enriched_prompt
                    }]
                }
            },
            "id": f"req_{agent_name}_{int(time.time())}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:  # 3분으로 대폭 증가
                response = await client.post(
                    agent_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._parse_agent_response(result, agent_name)
                else:
                    return {
                        'status': 'failed',
                        'error': f'HTTP {response.status_code}',
                        'summary': f'에이전트 호출 실패 (HTTP {response.status_code})'
                    }
                    
        except Exception as e:
            logger.error(f"Agent {agent_name} execution error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'summary': f'에이전트 실행 중 오류: {str(e)}'
            }

    async def _enrich_agent_task(self, agent_name: str, base_task: str, 
                                context: Dict, previous_results: Dict) -> str:
        """LLM이 각 에이전트의 작업을 컨텍스트에 맞게 보강"""
        
        if not self.openai_client:
            return base_task
        
        # 이전 에이전트들의 핵심 발견사항 추출
        previous_insights = self._extract_key_insights(previous_results)
        
        enrichment_prompt = f"""
에이전트: {agent_name}
기본 작업: {base_task}

전체 맥락:
- 도메인: {context['domain']}
- 사용자 역할/전문성: {context.get('expertise_claimed', '일반 사용자')}
- 최종 목표: {context['key_objectives']}
- 도메인 컨텍스트: {context.get('domain_context', '')}

이전 분석 결과:
{json.dumps(previous_insights, ensure_ascii=False)}

위 정보를 바탕으로 {agent_name} 에이전트가 수행해야 할 구체적인 작업을 작성하세요.
도메인 지식과 이전 결과를 반영하여 더 정확하고 유용한 분석이 되도록 지시하세요.
"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 빠른 응답을 위해
                messages=[{"role": "user", "content": enrichment_prompt}],
                temperature=0.2,
                max_tokens=1000,
                timeout=60.0  # 타임아웃 증가
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Task enrichment failed: {e}")
            return base_task

    async def _synthesize_with_llm(self, original_request: str, 
                                  understanding: Dict, all_results: Dict,
                                  task_updater: StreamingTaskUpdater) -> str:
        """🎯 NEW: Question-Driven Dynamic Structure 방식으로 답변 생성"""
        
        logger.info("🎯 Question-Driven 합성 시작")
        
        if not self.openai_client:
            logger.warning("❌ OpenAI 클라이언트 없음, fallback 사용")
            return self._create_fallback_synthesis(original_request, all_results)
        
        try:
            # 1단계: 질문에서 답변 구조 추출
            logger.info("📋 1단계: 질문에서 답변 구조 추출")
            answer_structure = await self._extract_answer_structure_from_question(original_request)
            logger.info(f"✅ 답변 구조 추출 완료: {len(answer_structure.get('required_sections', []))} 섹션")
            
            # 2단계: 실제 데이터 컨텍스트 추출 (할루시네이션 방지)
            logger.info("📊 2단계: 데이터 컨텍스트 추출")
            data_context = await self._extract_data_context(all_results)
            logger.info(f"✅ 데이터 컨텍스트 추출 완료: {data_context.get('data_quality', 'unknown')} 품질")
            
            # 3단계: 에이전트 결과 구조화
            logger.info("🔍 3단계: 에이전트 결과 구조화")
            structured_results = self._structure_agent_results(all_results)
            logger.info(f"✅ 결과 구조화 완료: {len(structured_results)} 문자")
            
            # 4단계: 🎯 NEW - 동적 구조 기반 프롬프트 생성
            logger.info("🎨 4단계: 동적 구조 기반 프롬프트 생성")
            synthesis_prompt = f"""당신은 {understanding.get('domain', '데이터 분석')} 분야의 전문가입니다.

## 🎯 사용자의 원본 질문
"{original_request}"

## 📋 사용자가 원하는 답변 구조 (질문에서 추출)
전체 답변 흐름: {answer_structure.get('overall_structure', '직접 답변')}

필요한 섹션들:
{json.dumps(answer_structure.get('required_sections', []), ensure_ascii=False, indent=2)}

답해야 할 핵심 질문들:
{json.dumps(answer_structure.get('key_questions_to_answer', []), ensure_ascii=False, indent=2)}

## 📊 실제 분석된 데이터 정보 (할루시네이션 방지)
- 사용 가능한 데이터: {len(data_context.get('available_data', []))}개 소스
- 데이터 품질: {data_context.get('data_quality', 'unknown')}
- 통계적 증거: {', '.join(data_context.get('statistical_evidence', [])[:10])}
- 데이터 제한사항: {', '.join(data_context.get('limitations', []))}

## 🔍 각 에이전트 분석 결과
{structured_results}

## ✅ 필수 준수사항 (Question-Driven 방식)
1. **질문 구조 완전 준수**: 위에서 추출한 답변 구조를 정확히 따르세요
2. **섹션별 맞춤 작성**: 각 required_section의 purpose와 expected_format에 맞게 작성
3. **실제 데이터만 사용**: 위 분석 결과만 사용하고, 추측하지 마세요
4. **핵심 질문 완전 답변**: key_questions_to_answer의 모든 질문에 답하세요
5. **구체적 근거 제시**: 모든 주장에 대해 분석 결과 기반 근거 제시

## ❌ 절대 금지
- 미리 정의된 템플릿 사용 (사용자 질문 구조와 다른 경우)
- 분석되지 않은 내용 추측
- 질문에서 요구하지 않은 섹션 추가
- 막연한 표현 ("일반적으로", "보통", "대체로" 등)

🎯 중요: 사용자가 질문에서 요구한 구조 그대로 답변하세요. 
예를 들어 "이상 여부를 판단하고 원인을 설명하며 조치를 제안"이라고 했다면, 
정확히 그 3가지 섹션으로 구성하세요."""

            logger.info(f"✅ 프롬프트 생성 완료: {len(synthesis_prompt)} 문자")
            
            # 5단계: LLM 호출
            logger.info("🤖 5단계: LLM 호출")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=4000,
                temperature=0.3,
                timeout=180
            )
            
            llm_response = response.choices[0].message.content
            logger.info(f"✅ LLM 응답 수신: {len(llm_response)} 문자")
            
            # 6단계: 품질 검증 (할루시네이션 체크)
            logger.info("🔍 6단계: 품질 검증")
            quality_ok = await self._validate_response_quality(llm_response, data_context, original_request)
            
            if quality_ok:
                logger.info("✅ 품질 검증 통과, 최종 응답 반환")
                return llm_response
            else:
                logger.warning("⚠️ 품질 검증 실패, 강화 프롬프트로 재시도")
                # 품질이 부족하면 더 강한 프롬프트로 재시도
                retry_result = await self._retry_with_stronger_prompt(
                    original_request, understanding, structured_results, data_context, answer_structure
                )
                logger.info("✅ 재시도 완료")
                return retry_result
                                                           
        except Exception as e:
            logger.error(f"❌ Question-Driven 합성 실패: {e}", exc_info=True)
            logger.warning("🔄 fallback_synthesis로 전환")
            return self._create_fallback_synthesis(original_request, all_results)

    async def _validate_response_quality(self, response: str, data_context: Dict, 
                                       original_request: str) -> bool:
        """🔍 응답 품질 검증 - 할루시네이션 및 피상적 답변 감지"""
        
        # 기본 품질 체크
        if len(response) < 300:
            return False
            
        # 할루시네이션 감지 패턴
        hallucination_patterns = [
            r'일반적으로 알려진', r'보통 \w+는', r'대체로', r'통상적으로',
            r'경험상', r'일반적인 경우', r'보편적으로'
        ]
        
        import re
        for pattern in hallucination_patterns:
            if re.search(pattern, response):
                logger.warning(f"할루시네이션 패턴 감지: {pattern}")
                return False
        
        # 피상적 답변 감지
        superficial_patterns = [
            r'분석을 통해 확인', r'결과를 바탕으로', r'데이터를 통해',
            r'추가 분석이 필요', r'향후 연구가 필요'
        ]
        
        superficial_count = sum(1 for pattern in superficial_patterns 
                               if re.search(pattern, response))
        
        if superficial_count > 2:
            logger.warning(f"피상적 표현 과다 감지: {superficial_count}개")
            return False
        
        # 사용자 질문 관련성 체크
        key_terms = original_request.split()[:5]  # 첫 5개 단어
        relevance_score = sum(1 for term in key_terms if term in response)
        
        if relevance_score < 2:
            logger.warning(f"질문 관련성 부족: {relevance_score}")
            return False
            
        return True

    async def _retry_with_stronger_prompt(self, original_request: str, understanding: Dict,
                                        structured_results: str, data_context: Dict, answer_structure: Dict) -> str:
        """🔥 품질 부족 시 더 강한 프롬프트로 재시도"""
        
        stronger_prompt = f"""🚨 중요: 이전 답변이 품질 기준을 충족하지 못했습니다. 다시 작성해주세요.

사용자 질문: "{original_request}"

## 🎯 반드시 준수해야 할 요구사항
1. 사용자 질문의 핵심 키워드를 반드시 포함하여 직접 답변
2. 아래 실제 분석 결과만 사용 (추측 금지)
3. 구체적인 수치나 발견사항이 있다면 명시
4. "일반적으로", "보통", "대체로" 같은 막연한 표현 사용 금지
5. 최소 500단어 이상의 상세한 답변

## 📊 실제 분석 데이터
{structured_results}

## 🔍 데이터 컨텍스트
- 통계적 증거: {', '.join(data_context.get('statistical_evidence', []))}
- 데이터 제한사항: {', '.join(data_context.get('limitations', []))}

사용자가 정확히 무엇을 원하는지 파악하고, 분석된 실제 데이터를 바탕으로 구체적이고 실용적인 답변을 제공하세요."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": stronger_prompt}],
                max_tokens=4000,
                temperature=0.2,
                timeout=180
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"재시도 실패: {e}")
            return self._create_fallback_synthesis(original_request, {})

    async def _analyze_user_intent_and_structure(self, user_input: str, understanding: Dict) -> Dict:
        """🎯 사용자 의도를 분석하여 최적의 답변 구조를 동적 생성"""
        
        if not self.openai_client:
            return {"structure_type": "direct_answer", "guidelines": []}
        
        try:
            structure_prompt = f"""사용자의 질문을 분석하여 가장 적합한 답변 구조를 결정하세요.

사용자 질문: "{user_input}"
질문 의도: {understanding}

다음 중에서 사용자가 원하는 답변 형태를 선택하고 구조를 제안하세요:

1. **직접 답변형**: 질문에 바로 답하는 형태 (예: "어떤 파라미터가 중요한가?")
2. **분석 보고서형**: 체계적인 분석 결과 (예: "데이터를 분석해줘")
3. **실행 가이드형**: 구체적 행동 방안 (예: "개선 방안을 제시해줘")
4. **비교 분석형**: 여러 옵션 비교 (예: "어떤 방법이 더 좋은가?")
5. **문제 해결형**: 문제 진단 및 해결책 (예: "불량률을 줄이고 싶어")
6. **탐색적 분석형**: 패턴 발견 및 인사이트 (예: "숨겨진 패턴을 찾아줘")

JSON 형식으로 응답하세요:
{{
    "structure_type": "선택된 답변 형태",
    "reasoning": "선택 이유",
    "key_elements": ["답변에 반드시 포함해야 할 핵심 요소들"],
    "tone": "답변 톤 (professional/technical/conversational/urgent)",
    "focus_areas": ["집중해야 할 영역들"],
    "avoid": ["피해야 할 내용들"]
}}"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": structure_prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            # JSON 파싱을 안전하게 처리
            response_content = response.choices[0].message.content.strip()
            logger.info(f"🔍 구조 분석 응답: {response_content[:200]}...")
            
            # JSON 블록 추출 (```json...``` 형태일 수 있음)
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            elif "```" in response_content:
                json_start = response_content.find("```") + 3
                json_end = response_content.find("```", json_start)
                if json_end != -1:
                    response_content = response_content[json_start:json_end].strip()
            
            try:
                structure_info = json.loads(response_content)
                logger.info(f"✅ 구조 분석 성공: {structure_info.get('structure_type', 'unknown')}")
                return structure_info
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON 파싱 실패: {json_error}, 원본 응답: {response_content[:200]}")
                # JSON 파싱 실패 시 기본값 반환
                return {
                    "structure_type": "direct_answer",
                    "reasoning": "JSON 파싱 실패로 기본 구조 사용",
                    "key_elements": ["구체적 답변", "데이터 기반 근거"],
                    "tone": "professional",
                    "focus_areas": ["사용자 질문 직접 답변"],
                    "avoid": ["불필요한 형식적 구조"]
                }
            
        except Exception as e:
            logger.warning(f"구조 분석 실패, 기본 구조 사용: {e}")
            return {
                "structure_type": "direct_answer",
                "reasoning": "기본 구조 사용",
                "key_elements": ["구체적 답변", "데이터 기반 근거"],
                "tone": "professional",
                "focus_areas": ["사용자 질문 직접 답변"],
                "avoid": ["불필요한 형식적 구조"]
            }

    async def _extract_data_context(self, all_results: Dict) -> Dict:
        """📊 분석 결과에서 실제 데이터 컨텍스트 추출 - 할루시네이션 방지"""
        
        data_context = {
            "available_data": [],
            "key_findings": [],
            "statistical_evidence": [],
            "limitations": [],
            "data_quality": "unknown"
        }
        
        try:
            for agent_name, result in all_results.items():
                if isinstance(result, dict):
                    # 실제 데이터 정보 추출
                    if 'artifacts' in result:
                        for artifact in result['artifacts']:
                            if isinstance(artifact, dict):
                                data_context["available_data"].append({
                                    "source": agent_name,
                                    "type": artifact.get('contentType', 'unknown'),
                                    "description": artifact.get('name', 'unnamed')
                                })
                    
                    # 핵심 발견사항 추출
                    if 'response' in result:
                        response_text = str(result['response'])
                        # 통계적 증거나 구체적 수치 추출
                        numbers = re.findall(r'\d+\.?\d*%|\d+\.?\d*', response_text)
                        if numbers:
                            data_context["statistical_evidence"].extend(numbers[:5])  # 최대 5개
                        
                        # 핵심 발견사항 키워드 추출
                        keywords = re.findall(r'(중요|핵심|주요|발견|결과|분석|상관관계|패턴)', response_text)
                        if keywords:
                            data_context["key_findings"].append(f"{agent_name}에서 {len(keywords)}개 핵심 발견")
            
            # 데이터 품질 평가
            if len(data_context["available_data"]) > 3:
                data_context["data_quality"] = "good"
            elif len(data_context["available_data"]) > 1:
                data_context["data_quality"] = "moderate"
            else:
                data_context["data_quality"] = "limited"
                data_context["limitations"].append("제한된 데이터 소스")
                
        except Exception as e:
            logger.warning(f"데이터 컨텍스트 추출 실패: {e}")
            data_context["limitations"].append("데이터 컨텍스트 추출 제한")
        
        return data_context

    def _structure_agent_results(self, all_results: Dict) -> str:
        """에이전트 결과를 LLM이 이해하기 쉽게 구조화"""
        structured = "### 에이전트별 상세 분석 결과\n\n"
        
        for agent_name, result in all_results.items():
            status = result.get('status', 'unknown')
            structured += f"#### 🤖 {agent_name} 에이전트\n"
            structured += f"- **실행 상태**: {status}\n"
            
            if status == 'success':
                # 성공한 경우 결과 상세 정보 추출
                agent_result = result.get('result', {})
                structured += f"- **요약**: {result.get('summary', '작업 완료')}\n"
                
                # 결과에서 핵심 정보 추출
                if isinstance(agent_result, dict):
                    if 'artifacts' in agent_result:
                        artifacts = agent_result['artifacts']
                        structured += f"- **생성된 아티팩트**: {len(artifacts)}개\n"
                        for artifact in artifacts[:3]:  # 최대 3개까지만 표시
                            artifact_name = artifact.get('name', '이름 없음')
                            artifact_type = artifact.get('metadata', {}).get('content_type', '타입 미지정')
                            structured += f"  - {artifact_name} ({artifact_type})\n"
                    
                    if 'message' in agent_result:
                        message = agent_result['message']
                        if isinstance(message, dict) and 'parts' in message:
                            parts = message['parts']
                            if parts and len(parts) > 0:
                                first_part = parts[0]
                                if hasattr(first_part, 'text'):
                                    text_preview = first_part.text[:200] + "..." if len(first_part.text) > 200 else first_part.text
                                    structured += f"- **결과 미리보기**: {text_preview}\n"
                
                # 원시 결과 데이터도 포함 (JSON 형태)
                structured += f"- **상세 결과**: {json.dumps(agent_result, ensure_ascii=False, indent=2)[:500]}...\n"
                
            else:
                # 실패한 경우 오류 정보
                error_msg = result.get('error', '알 수 없는 오류')
                structured += f"- **오류 내용**: {error_msg}\n"
                structured += f"- **영향**: 이 에이전트의 결과는 최종 분석에서 제외됩니다\n"
            
            structured += "\n"
        
        # 전체 요약 정보
        total_agents = len(all_results)
        successful_agents = len([r for r in all_results.values() if r.get('status') == 'success'])
        failed_agents = total_agents - successful_agents
        
        structured += f"### 📊 전체 실행 요약\n"
        structured += f"- **총 에이전트 수**: {total_agents}개\n"
        structured += f"- **성공**: {successful_agents}개\n"
        structured += f"- **실패**: {failed_agents}개\n"
        structured += f"- **성공률**: {(successful_agents/total_agents*100):.1f}%\n\n"
        
        return structured

    def _extract_key_insights(self, previous_results: Dict) -> Dict:
        """이전 결과에서 핵심 인사이트 추출"""
        insights = {}
        for agent_name, result in previous_results.items():
            if result.get('status') == 'success':
                insights[agent_name] = result.get('summary', '작업 완료')
            else:
                insights[agent_name] = f"오류: {result.get('error', '알 수 없는 오류')}"
        return insights

    def _parse_agent_response(self, response: Dict, agent_name: str) -> Dict:
        """에이전트 응답 파싱"""
        try:
            if 'result' in response:
                result = response['result']
                if isinstance(result, dict):
                    # A2A 표준 응답 처리
                    status = result.get('status', {})
                    if isinstance(status, dict) and status.get('state') == 'completed':
                        return {
                            'status': 'success',
                            'result': result,
                            'summary': f'{agent_name} 에이전트 작업 완료'
                        }
                    else:
                        return {
                            'status': 'partial',
                            'result': result,
                            'summary': f'{agent_name} 에이전트 부분 완료'
                        }
                else:
                    return {
                        'status': 'success',
                        'result': result,
                        'summary': f'{agent_name} 에이전트 작업 완료'
                    }
            else:
                return {
                    'status': 'failed',
                    'error': 'No result in response',
                    'summary': f'{agent_name} 에이전트 응답 오류'
                }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'summary': f'{agent_name} 에이전트 응답 파싱 오류'
            }

    def _create_fallback_plan(self, available_agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """LLM이 사용 불가능할 때의 폴백 계획"""
        steps = []
        step_number = 1
        
        # 기본적인 데이터 분석 워크플로우
        basic_workflow = [
            ("data_loader", "데이터 로드 및 기본 검증", "데이터 파일을 로드하고 기본적인 구조를 파악합니다"),
            ("data_cleaning", "데이터 품질 확인 및 정리", "데이터의 품질을 확인하고 필요한 정리 작업을 수행합니다"),
            ("eda_tools", "탐색적 데이터 분석 수행", "데이터의 분포와 패턴을 탐색하고 기초 통계를 생성합니다"),
            ("data_visualization", "데이터 시각화 및 인사이트 도출", "데이터를 시각화하여 인사이트를 도출합니다")
        ]
        
        for agent_name, purpose, task_desc in basic_workflow:
            if agent_name in available_agents:
                steps.append({
                    "agent": agent_name,
                    "purpose": purpose,
                    "enriched_task": task_desc,
                    "expected_output": f"{purpose} 결과",
                    "pass_to_next": ["분석 결과", "데이터 정보"]
                })
                step_number += 1
        
        return {
            "reasoning": "사용자 요청에 대한 포괄적인 데이터 분석을 수행하기 위한 표준 워크플로우",
            "steps": steps,
            "final_synthesis_guide": "모든 분석 결과를 종합하여 사용자에게 유용한 인사이트 제공"
        }

    def _create_fallback_synthesis(self, original_request: str, all_results: Dict) -> str:
        """폴백 최종 답변 생성"""
        
        successful_agents = [name for name, result in all_results.items() 
                           if result.get('status') == 'success']
        failed_agents = [name for name, result in all_results.items() 
                        if result.get('status') == 'failed']
        
        synthesis = f"""## 📊 데이터 분석 결과 종합

### 🎯 요청 사항
{original_request}

### ✅ 완료된 분석 단계
"""
        
        for agent_name in successful_agents:
            result = all_results[agent_name]
            synthesis += f"- **{agent_name}**: {result.get('summary', '작업 완료')}\n"
        
        if failed_agents:
            synthesis += f"\n### ⚠️ 일부 제한사항\n"
            for agent_name in failed_agents:
                result = all_results[agent_name]
                synthesis += f"- **{agent_name}**: {result.get('error', '알 수 없는 오류')}\n"
        
        synthesis += f"""

### 🎉 결론
총 {len(successful_agents)}개의 분석 단계가 성공적으로 완료되었습니다. 
각 에이전트가 생성한 결과물과 아티팩트를 확인하시기 바랍니다.

분석 결과에 대한 추가 질문이나 더 자세한 분석이 필요하시면 언제든 요청해 주세요.
"""
        
        return synthesis

    def _create_beautiful_plan_display(self, execution_plan: Dict, understanding: Dict) -> str:
        """예쁜 실행 계획 표시 생성"""
        
        plan_display = f"""
## 📋 LLM 기반 동적 실행 계획

### 🎯 분석 개요
- **도메인**: {understanding.get('domain', '데이터 분석')}
- **목표**: {', '.join(understanding.get('key_objectives', ['데이터 분석 수행']))}
- **분석 깊이**: {understanding.get('analysis_depth', 'intermediate')}
- **총 단계**: {len(execution_plan.get('steps', []))}개

### 🚀 실행 단계별 계획

"""
        
        for i, step in enumerate(execution_plan.get('steps', [])):
            step_num = i + 1
            agent_name = step.get('agent', step.get('agent_name', 'unknown'))
            purpose = step.get('purpose', '')
            task = step.get('enriched_task', step.get('task_description', ''))
            expected = step.get('expected_output', '')
            
            plan_display += f"""**{step_num}. {agent_name} 에이전트**
   - 🎯 **목적**: {purpose}
   - 📝 **작업**: {task[:150]}{'...' if len(task) > 150 else ''}
   - 📊 **예상 결과**: {expected}

"""
        
        plan_display += f"""
### 🧠 계획 근거
{execution_plan.get('reasoning', '사용자 요청에 최적화된 분석 워크플로우')}

---
"""
        
        return plan_display

    def _create_beautiful_final_display(self, final_response: str, execution_plan: Dict, 
                                      agent_results: Dict, understanding: Dict) -> str:
        """예쁜 최종 결과 표시 생성"""
        
        successful_agents = [name for name, result in agent_results.items() 
                           if result.get('status') == 'success']
        failed_agents = [name for name, result in agent_results.items() 
                        if result.get('status') == 'failed']
        
        final_display = f"""
## 🎉 LLM 기반 종합 분석 결과

### 📈 실행 성과
- ✅ **성공한 단계**: {len(successful_agents)}개
- ❌ **실패한 단계**: {len(failed_agents)}개
- 📊 **전체 성공률**: {(len(successful_agents) / len(agent_results) * 100):.1f}%

### 🔍 단계별 실행 결과
"""
        
        for i, step in enumerate(execution_plan.get('steps', [])):
            step_num = i + 1
            agent_name = step.get('agent', step.get('agent_name', 'unknown'))
            result = agent_results.get(agent_name, {})
            status = result.get('status', 'unknown')
            summary = result.get('summary', '결과 없음')
            
            status_icon = "✅" if status == 'success' else "❌" if status == 'failed' else "⚠️"
            
            final_display += f"""
**{step_num}. {agent_name}** {status_icon}
   - 📝 **결과**: {summary}
"""
        
        final_display += f"""

### 🎯 최종 종합 분석

{final_response}

---
*🤖 AI DS Team LLM Powered Dynamic Orchestrator v6*
"""
        
        return final_display


def create_llm_powered_orchestrator_server():
    """LLM 기반 동적 컨텍스트 인식 오케스트레이터 서버 생성"""
    
    agent_card = AgentCard(
        name="AI DS Team LLM Powered Dynamic Orchestrator v6",
        description="LLM 기반 동적 컨텍스트 인식 멀티 에이전트 오케스트레이터 - 실시간 스트리밍 지원",
        url="http://localhost:8100",
        version="6.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="llm_powered_orchestration",
                name="LLM Powered Dynamic Context-Aware Orchestration",
                description="LLM이 사용자 요청을 깊이 이해하고 동적으로 최적의 실행 계획을 생성하여 AI DS Team 에이전트들을 조정합니다. 실시간 스트리밍과 컨텍스트 인식 기능을 제공합니다.",
                tags=["llm-powered", "dynamic-orchestration", "context-aware", "streaming", "multi-agent", "data-science"]
            )
        ]
    )
    
    executor = LLMPoweredOrchestratorExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )


def main():
    """메인 실행 함수"""
    logger.info("🚀 Starting LLM Powered Dynamic Context-Aware Orchestrator v6.0")
    
    app = create_llm_powered_orchestrator_server()
    
    uvicorn.run(
        app.build(),
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main() 