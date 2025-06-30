#!/usr/bin/env python3
"""
A2A Orchestrator v7.0 - Universal LLM Powered Dynamic System
완전 범용 LLM 기반 동적 시스템
- Universal Request Analyzer
- Adaptive Context Builder  
- Smart Question Expander
- Flexible Response Generator
- Rich Information Extraction Planning
- Dynamic Content Assessment
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
        """🎯 Universal LLM Powered Dynamic System - 완전 범용 실행"""
        task_updater = StreamingTaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            user_input = context.get_user_input()
            logger.info(f"🎯 Universal System Processing: {user_input}")
            
            if not user_input:
                user_input = "Please provide an analysis request."
            
            # 🎯 Step 1: Universal Request Analyzer
            await task_updater.stream_update("🧠 Universal Request Analyzer 실행 중...")
            request_analysis = await self._analyze_request_depth(user_input)
            
            # 🎯 Step 2: Adaptive Context Builder
            await task_updater.stream_update("🎭 Adaptive Context Builder 실행 중...")
            adaptive_context = await self._build_adaptive_context(user_input, request_analysis)
            
            # 🎯 Step 3: Smart Question Expander
            await task_updater.stream_update("📈 Smart Question Expander 실행 중...")
            expanded_request = await self._expand_simple_requests(user_input, request_analysis)
            
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
            
            # 🎯 Step 4: Rich Information Extraction Planning
            await task_updater.stream_update("📋 Rich Information Extraction Planning 실행 중...")
            
            # 기본 요청 이해
            request_understanding = await self._understand_request(expanded_request)
            
            # 종합적 실행 계획 생성
            execution_plan = await self._create_comprehensive_execution_plan(
                expanded_request,
                request_understanding,
                self.available_agents
            )
            
            if not execution_plan or not execution_plan.get('steps'):
                execution_plan = self._create_fallback_plan(self.available_agents)
            
            await task_updater.stream_update(f"✅ 종합적 실행 계획 완료: {len(execution_plan.get('steps', []))}단계")
            
            # 📋 계획 표시
            plan_display = self._create_beautiful_plan_display(execution_plan, request_understanding)
            await task_updater.stream_update(plan_display)
            
            # 계획을 아티팩트로 전송
            plan_artifact = {
                "execution_strategy": execution_plan.get('execution_strategy', 'comprehensive_value_extraction'),
                "plan_executed": [
                    {
                        "step": i + 1,
                        "agent": step.get('agent', 'unknown'),
                        "comprehensive_instructions": step.get('comprehensive_instructions', ''),
                        "expected_deliverables": step.get('expected_deliverables', {})
                    }
                    for i, step in enumerate(execution_plan.get('steps', []))
                ]
            }
            
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(plan_artifact, ensure_ascii=False, indent=2))],
                name="comprehensive_execution_plan.json",
                metadata={
                    "content_type": "application/json",
                    "plan_type": "universal_llm_orchestration",
                    "description": "Universal LLM 기반 종합적 실행 계획"
                }
            )
            
            await asyncio.sleep(2)
            await task_updater.stream_update("🚀 Universal System 실행 시작...")
            
            # 🎯 Step 5: Execute Agents with Rich Context
            agent_results = {}
            for i, step in enumerate(execution_plan.get('steps', [])):
                step_num = i + 1
                agent_name = step.get('agent', 'unknown')
                
                step_info = f"""
🔄 **단계 {step_num}/{len(execution_plan.get('steps', []))}: {agent_name} 실행**

📝 **종합적 지시사항**: {step.get('comprehensive_instructions', '')[:200]}...

🎯 **기대 성과**:
- 최소: {step.get('expected_deliverables', {}).get('minimum', '기본 분석')}
- 표준: {step.get('expected_deliverables', {}).get('standard', '품질 분석')}
- 탁월: {step.get('expected_deliverables', {}).get('exceptional', '인사이트 도출')}
"""
                await task_updater.stream_update(step_info)
                
                # 에이전트 실행 (종합적 지시사항 사용)
                agent_result = await self._execute_agent_with_comprehensive_instructions(
                    agent_name, 
                    step,
                    adaptive_context,
                    agent_results
                )
                
                agent_results[agent_name] = agent_result
                
                if agent_result.get('status') == 'success':
                    await task_updater.stream_update(f"✅ {agent_name} 실행 완료")
                else:
                    await task_updater.stream_update(f"⚠️ {agent_name} 실행 이슈: {agent_result.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1)
            
            # 🎯 Step 6: Dynamic Content Assessment
            await task_updater.stream_update("🎨 Dynamic Content Assessment 실행 중...")
            content_assessment = await self._assess_content_richness(agent_results)
            
            # 🎯 Step 7: Flexible Response Generation
            await task_updater.stream_update("🎯 Flexible Response Generation 실행 중...")
            
            # 시각화 추출
            visualizations = self._extract_visualizations(agent_results)
            
            # 유연한 응답 생성
            base_response = await self._generate_flexible_response(
                user_input,  # 원본 요청 사용
                request_analysis,
                adaptive_context,
                agent_results
            )
            
            # 🎨 Rich Details Injection
            enriched_response = await self._inject_rich_details(
                base_response,
                content_assessment,
                agent_results,
                request_analysis
            )
            
            # 🎨 Visualization Integration
            if visualizations and content_assessment.get('has_visualizations'):
                enriched_response = await self._integrate_visualizations(
                    enriched_response,
                    visualizations
                )
            
            # 🎨 Smart Default Enrichment
            final_response = await self._enrich_unless_explicitly_simple(
                user_input,
                enriched_response,
                {
                    'visualizations': visualizations,
                    'metrics': content_assessment.get('key_metrics', {}),
                    'findings': content_assessment.get('critical_findings', [])
                }
            )
            
            # 🎯 Final Delivery
            await task_updater.stream_update("🎉 Universal System 분석 완료!")
            
            # 최종 응답 전달
            if request_analysis.get('detail_level', 5) < 3:
                # 간단한 응답은 한 번에
                await task_updater.update_status(
                    TaskState.completed,
                    message=task_updater.new_agent_message(parts=[TextPart(text=final_response)])
                )
            else:
                # 상세한 응답은 스트리밍
                await task_updater.stream_final_response(final_response)
            
            # 실행 결과 요약 아티팩트
            execution_summary = {
                "request_analysis": request_analysis,
                "adaptive_context": adaptive_context,
                "content_assessment": content_assessment,
                "visualizations_found": len(visualizations),
                "agents_executed": len(agent_results),
                "response_length": len(final_response),
                "execution_strategy": "universal_llm_powered_dynamic_system"
            }
            
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(execution_summary, ensure_ascii=False, indent=2))],
                name="execution_summary.json",
                metadata={
                    "content_type": "application/json",
                    "summary_type": "universal_system_execution",
                    "description": "Universal System 실행 요약"
                }
            )
            
        except Exception as e:
            error_msg = f"Universal System 실행 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=error_msg)])
            )
    
    async def _execute_agent_with_comprehensive_instructions(self, agent_name: str, step: Dict, 
                                                           adaptive_context: Dict,
                                                           previous_results: Dict) -> Dict:
        """🎯 NEW: 종합적 지시사항으로 에이전트 실행"""
        
        if agent_name not in self.available_agents:
            return {
                'status': 'failed',
                'error': f'Agent {agent_name} not available',
                'summary': f'에이전트 {agent_name}를 찾을 수 없습니다'
            }
        
        # 종합적 지시사항 사용
        comprehensive_instructions = step.get('comprehensive_instructions', f'{agent_name}에 대한 분석을 수행하세요.')
        
        # 에이전트 실행
        agent_url = self.available_agents[agent_name]['url']
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"universal_req_{agent_name}_{int(time.time())}",
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": comprehensive_instructions
                    }]
                }
            },
            "id": f"universal_req_{agent_name}_{int(time.time())}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
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

    async def _analyze_request_depth(self, user_input: str) -> Dict:
        """🎯 NEW: 요청의 깊이와 특성을 LLM이 자동 분석"""
        
        if not self.openai_client:
            return {
                "detail_level": 5,
                "has_role_description": False,
                "role_description": "",
                "explicit_requirements": ["기본 분석"],
                "implicit_needs": ["데이터 이해"],
                "suggested_response_depth": "moderate",
                "needs_clarification": []
            }
        
        analysis_prompt = f"""
        다음 사용자 요청을 분석하세요:
        "{user_input}"
        
        분석할 내용:
        1. 요청의 구체성 수준 (1-10)
        2. 역할이나 전문성 언급 여부
        3. 명시적 요구사항 vs 암시적 니즈
        4. 예상되는 답변의 깊이
        5. 추가 컨텍스트가 필요한지 여부
        
        JSON 응답:
        {{
            "detail_level": 1-10,
            "has_role_description": true/false,
            "role_description": "있다면 어떤 역할인지",
            "explicit_requirements": ["명시적으로 요청한 것들"],
            "implicit_needs": ["맥락상 필요할 것으로 보이는 것들"],
            "suggested_response_depth": "brief/moderate/comprehensive",
            "needs_clarification": ["명확히 할 필요가 있는 부분들"],
            "explicitly_wants_brief": true/false
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"📊 Request depth analysis: level {analysis.get('detail_level', 5)}/10")
            return analysis
            
        except Exception as e:
            logger.warning(f"Request depth analysis failed: {e}")
            return {
                "detail_level": 5,
                "has_role_description": False,
                "role_description": "",
                "explicit_requirements": ["기본 분석"],
                "implicit_needs": ["데이터 이해"],
                "suggested_response_depth": "moderate",
                "needs_clarification": []
            }

    async def _build_adaptive_context(self, user_input: str, request_analysis: Dict) -> Dict:
        """🎯 NEW: 요청 분석에 따라 적응적 컨텍스트 구축"""
        
        if not self.openai_client:
            return {
                "adopted_perspective": "일반 데이터 분석가",
                "analysis_approach": "표준 분석",
                "response_style": "professional",
                "focus_areas": ["기본 통계"],
                "depth_strategy": "moderate"
            }
        
        context_prompt = f"""
        사용자 요청: "{user_input}"
        요청 분석: {json.dumps(request_analysis, ensure_ascii=False)}
        
        이 정보를 바탕으로 적절한 분석 컨텍스트를 구축하세요:
        
        1. 역할이 명시되었다면: 그 역할의 관점 채택
        2. 역할이 없다면: 요청 내용에서 적절한 전문성 수준 추론
        3. 상세한 요청이면: 깊이 있는 분석 준비
        4. 간단한 요청이면: 핵심만 간결하게
        
        다음 형식으로 컨텍스트를 만드세요:
        {{
            "adopted_perspective": "채택할 관점",
            "analysis_approach": "분석 접근 방식",
            "response_style": "답변 스타일",
            "focus_areas": ["집중할 영역들"],
            "depth_strategy": "얼마나 깊이 들어갈지"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": context_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            context = json.loads(response.choices[0].message.content)
            logger.info(f"🎭 Adaptive context: {context.get('adopted_perspective', 'unknown')}")
            return context
            
        except Exception as e:
            logger.warning(f"Adaptive context building failed: {e}")
            return {
                "adopted_perspective": "일반 데이터 분석가",
                "analysis_approach": "표준 분석",
                "response_style": "professional",
                "focus_areas": ["기본 통계"],
                "depth_strategy": "moderate"
            }

    async def _expand_simple_requests(self, user_input: str, request_analysis: Dict) -> str:
        """🎯 NEW: 간단한 요청을 지능적으로 확장 (필요한 경우만)"""
        
        if request_analysis['detail_level'] >= 7:
            # 이미 충분히 상세함
            return user_input
        
        if request_analysis['needs_clarification'] and not self.openai_client:
            return user_input
        
        if request_analysis['needs_clarification']:
            expansion_prompt = f"""
            사용자의 간단한 요청: "{user_input}"
            
            이 요청에서 사용자가 암묵적으로 알고 싶어할 수 있는 것들을 추론하세요.
            하지만 과도하게 확장하지 말고, 맥락상 합리적인 수준에서만 보완하세요.
            
            예시:
            - "데이터 분석해줘" → 기본 통계, 패턴, 이상치 정도
            - "불량 원인 찾아줘" → 데이터 기반 원인 추정, 가능성 순위
            
            추론된 상세 요구사항:
            """
            
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    timeout=60.0
                )
                
                expanded = response.choices[0].message.content
                logger.info(f"📈 Request expanded: {len(expanded)} chars")
                return expanded
                
            except Exception as e:
                logger.warning(f"Request expansion failed: {e}")
                return user_input
        
        return user_input

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

    async def _create_comprehensive_execution_plan(self, expanded_request: str, 
                                                  request_understanding: Dict,
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
{json.dumps(request_understanding, ensure_ascii=False, indent=2)}

## 🤖 사용 가능한 에이전트들
{json.dumps(agents_details, ensure_ascii=False, indent=2)}

## 🎯 계획 수립 지침
1. **요청 중심 접근**: 사용자가 원하는 결과에 집중하여 필요한 에이전트만 선택
2. **논리적 순서**: 데이터 흐름과 의존성을 고려한 순서 결정
3. **효율성 최적화**: 불필요한 단계 제거, 핵심 분석에 집중
4. **도메인 적응**: {request_understanding.get('domain', '일반')} 도메인 특성 반영
5. **사용자 수준 고려**: {request_understanding.get('expertise_claimed', '일반 사용자')} 수준에 맞는 분석 깊이

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
            validated_plan = self._validate_and_enhance_plan(plan, available_agents, request_understanding)
            
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

    async def _assess_content_richness(self, agent_results: Dict) -> Dict:
        """🎨 NEW: 생성된 콘텐츠의 풍부함을 평가하고 활용 방안 결정"""
        
        if not self.openai_client:
            return {
                "has_visualizations": False,
                "visualization_details": [],
                "key_metrics": {},
                "critical_findings": ["기본 분석 결과"],
                "data_quality_score": 5,
                "recommended_inclusion": ["분석 요약"]
            }
        
        assessment_prompt = f"""
        다음 분석 결과들을 평가하세요:
        {json.dumps(agent_results, ensure_ascii=False, indent=2)}
        
        평가할 항목:
        1. 시각화 자료 (차트, 그래프)의 존재와 중요도
        2. 구체적인 수치나 통계 데이터
        3. 발견된 패턴이나 이상치
        4. 실무적 인사이트의 가치
        5. 사용자가 놓치면 아까울 중요 정보
        
        {{
            "has_visualizations": true/false,
            "visualization_details": ["어떤 시각화가 있는지"],
            "key_metrics": {{"메트릭명": "값"}},
            "critical_findings": ["놓치면 안 되는 발견사항"],
            "data_quality_score": 1-10,
            "recommended_inclusion": ["반드시 포함해야 할 요소들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=60.0
            )
            
            assessment = json.loads(response.choices[0].message.content)
            logger.info(f"🎨 Content richness assessed: score {assessment.get('data_quality_score', 5)}/10")
            return assessment
            
        except Exception as e:
            logger.warning(f"Content richness assessment failed: {e}")
            return {
                "has_visualizations": False,
                "visualization_details": [],
                "key_metrics": {},
                "critical_findings": ["기본 분석 결과"],
                "data_quality_score": 5,
                "recommended_inclusion": ["분석 요약"]
            }

    async def _inject_rich_details(self, 
                              base_response: str,
                              content_assessment: Dict,
                              agent_results: Dict,
                              user_request_analysis: Dict) -> str:
        """🎨 NEW: 기본 응답에 풍부한 디테일을 적응적으로 주입"""
        
        if not self.openai_client:
            return base_response
        
        injection_prompt = f"""
        기본 응답: {base_response}
        
        사용 가능한 풍부한 콘텐츠:
        - 시각화: {content_assessment['visualization_details']}
        - 핵심 수치: {content_assessment['key_metrics']}
        - 중요 발견사항: {content_assessment['critical_findings']}
        
        사용자 요청 특성:
        - 상세도: {user_request_analysis['detail_level']}/10
        - 명시적 간단 요청 여부: {user_request_analysis.get('explicitly_wants_brief', False)}
        
        지침:
        1. 사용자가 명시적으로 "간단히"를 요청하지 않았다면, 중요한 디테일 포함
        2. 시각화가 있다면 반드시 언급하고 주요 인사이트 설명
        3. 구체적 수치를 텍스트로 포함 (예: "TW 평균값이 3,622로 상한선 4,080의 88.8%")
        4. 데이터가 풍부할 때는 섹션을 나누어 체계적으로 제시
        5. 중요한 발견은 강조 (볼드, 불릿 포인트 등)
        
        향상된 응답을 작성하세요. 원본의 톤은 유지하되, 가치 있는 정보는 빠뜨리지 마세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": injection_prompt}],
                temperature=0.3,
                timeout=90.0
            )
            
            enhanced_response = response.choices[0].message.content
            logger.info(f"💎 Rich details injected: {len(enhanced_response)} chars")
            return enhanced_response
            
        except Exception as e:
            logger.warning(f"Rich detail injection failed: {e}")
            return base_response

    async def _integrate_visualizations(self, 
                                  text_response: str,
                                  visualizations: List[Dict]) -> str:
        """🎨 NEW: 시각화를 텍스트 응답에 자연스럽게 통합"""
        
        if not visualizations or not self.openai_client:
            return text_response
        
        integration_prompt = f"""
        텍스트 응답: {text_response}
        
        사용 가능한 시각화:
        {json.dumps(visualizations, ensure_ascii=False)}
        
        각 시각화에 대해:
        1. 적절한 위치에 참조 추가
        2. 시각화가 보여주는 핵심 인사이트 설명
        3. 중요한 데이터 포인트 텍스트로도 명시
        
        예시:
        "아래 시계열 차트에서 볼 수 있듯이, HAE4026 장비의 TW 값이 
        1월 5일 3,706에서 1월 7일 7,010으로 89% 증가했습니다.
        특히 IS CARBON IMP 공정에서 급격한 상승이 관찰됩니다."
        
        시각화와 텍스트가 상호보완적이 되도록 통합하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": integration_prompt}],
                temperature=0.2,
                timeout=90.0
            )
            
            integrated_response = response.choices[0].message.content
            logger.info(f"📊 Visualizations integrated successfully")
            return integrated_response
            
        except Exception as e:
            logger.warning(f"Visualization integration failed: {e}")
            return text_response

    async def _enrich_unless_explicitly_simple(self,
                                         user_input: str,
                                         initial_response: str,
                                         available_content: Dict) -> str:
        """🎨 NEW: 명시적 간단 요청이 아니면 자동으로 풍부하게"""
        
        # 간단함을 명시적으로 요청했는지 확인
        simplicity_indicators = ["간단히", "요약만", "briefly", "summary only", "한 줄로"]
        explicitly_simple = any(indicator in user_input.lower() for indicator in simplicity_indicators)
        
        if explicitly_simple:
            return initial_response
        
        if not self.openai_client:
            return initial_response
        
        # 풍부한 콘텐츠 자동 추가
        enrichment_prompt = f"""
        사용자가 특별히 간단함을 요구하지 않았으므로, 
        분석의 가치를 최대한 전달하세요.
        
        현재 응답: {initial_response}
        
        추가 가능한 콘텐츠:
        {json.dumps(available_content, ensure_ascii=False)}
        
        다음을 포함하여 응답을 풍부하게 만드세요:
        1. 📊 시각화 결과와 그 의미
        2. 🔢 구체적인 수치와 비율
        3. 📈 트렌드와 패턴
        4. ⚠️ 주의가 필요한 발견사항
        5. 💡 실무적 인사이트
        
        보고서처럼 섹션을 나누어도 좋습니다:
        - 핵심 요약
        - 상세 분석 결과
        - 시각화 인사이트
        - 권장 조치사항
        
        사용자가 "어렵게 분석한" 결과를 충분히 활용할 수 있게 하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": enrichment_prompt}],
                temperature=0.4,
                timeout=120.0
            )
            
            enriched_response = response.choices[0].message.content
            logger.info(f"🌟 Response enriched: {len(enriched_response)} chars")
            return enriched_response
            
        except Exception as e:
            logger.warning(f"Response enrichment failed: {e}")
            return initial_response

    def _extract_visualizations(self, agent_results: Dict) -> List[Dict]:
        """분석 결과에서 시각화 정보 추출"""
        visualizations = []
        
        for agent_name, result in agent_results.items():
            if isinstance(result, dict):
                # 아티팩트에서 시각화 찾기
                artifacts = result.get('artifacts', [])
                for artifact in artifacts:
                    if isinstance(artifact, dict):
                        name = artifact.get('name', '')
                        if any(ext in name.lower() for ext in ['.png', '.jpg', '.svg', '.html', 'chart', 'plot', 'graph']):
                            visualizations.append({
                                'agent': agent_name,
                                'name': name,
                                'type': self._infer_chart_type(name),
                                'description': artifact.get('description', ''),
                                'data_points': self._extract_data_points(artifact)
                            })
        
        return visualizations

    def _infer_chart_type(self, filename: str) -> str:
        """파일명에서 차트 타입 추론"""
        filename_lower = filename.lower()
        if 'histogram' in filename_lower or 'hist' in filename_lower:
            return '히스토그램'
        elif 'scatter' in filename_lower:
            return '산점도'
        elif 'line' in filename_lower or 'time' in filename_lower:
            return '시계열 차트'
        elif 'box' in filename_lower:
            return '박스플롯'
        elif 'bar' in filename_lower:
            return '막대 차트'
        else:
            return '차트'

    def _extract_data_points(self, artifact: Dict) -> Dict:
        """아티팩트에서 핵심 데이터 포인트 추출"""
        # 메타데이터나 설명에서 수치 정보 추출 시도
        description = artifact.get('description', '')
        metadata = artifact.get('metadata', {})
        
        data_points = {}
        
        # 간단한 패턴 매칭으로 수치 추출
        import re
        numbers = re.findall(r'(\w+):\s*([0-9,]+\.?[0-9]*)', description)
        for key, value in numbers:
            try:
                data_points[key] = float(value.replace(',', ''))
            except:
                data_points[key] = value
        
        return data_points

    async def _generate_flexible_response(self,
                                    user_input: str,
                                    request_analysis: Dict,
                                    context: Dict,
                                    agent_results: Dict) -> str:
        """🎯 NEW: 요청 특성에 맞는 유연한 응답 생성"""
        
        if not self.openai_client:
            return self._create_fallback_synthesis(user_input, agent_results)
        
        # 기본 프롬프트 구성
        base_prompt = f"""
        사용자 요청: "{user_input}"
        
        분석된 데이터:
        {self._structure_agent_results(agent_results)}
        """
        
        # 역할이 있는 경우 추가
        if request_analysis['has_role_description']:
            role_prompt = f"""
            당신은 {request_analysis['role_description']}의 관점에서 응답하세요.
            해당 분야의 전문 용어와 관심사를 반영하세요.
            """
        else:
            role_prompt = """
            전문적이지만 이해하기 쉬운 방식으로 응답하세요.
            기술적 정확성과 실용성의 균형을 맞추세요.
            """
        
        # 상세도에 따른 지시
        if request_analysis['detail_level'] < 3:
            depth_prompt = """
            핵심만 간단명료하게 답변하세요.
            불필요한 세부사항은 제외하고 중요한 결과만 전달하세요.
            """
        elif request_analysis['detail_level'] < 7:
            depth_prompt = """
            적절한 수준의 상세함으로 답변하세요.
            주요 발견사항과 그 의미를 설명하되, 과도하게 기술적이지 않게 하세요.
            """
        else:
            depth_prompt = """
            포괄적이고 상세한 분석을 제공하세요.
            모든 관련 데이터, 패턴, 인사이트를 포함하세요.
            필요하다면 기술적 세부사항도 설명하세요.
            """
        
        # 최종 프롬프트 조합
        final_prompt = f"""
        {base_prompt}
        
        {role_prompt}
        
        {depth_prompt}
        
        답변 지침:
        - 사용자가 명시적으로 요청한 것: {request_analysis['explicit_requirements']}
        - 추가로 도움될 수 있는 정보: {request_analysis['implicit_needs']}
        
        형식에 얽매이지 말고, 상황에 맞는 가장 자연스러운 방식으로 응답하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.4,
                timeout=120.0
            )
            
            flexible_response = response.choices[0].message.content
            logger.info(f"🎯 Flexible response generated: {len(flexible_response)} chars")
            return flexible_response
            
        except Exception as e:
            logger.warning(f"Flexible response generation failed: {e}")
            return self._create_fallback_synthesis(user_input, agent_results)

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
                                      agent_results: Dict, request_understanding: Dict) -> str:
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