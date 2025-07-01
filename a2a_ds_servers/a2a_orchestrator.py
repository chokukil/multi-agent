#!/usr/bin/env python3
"""
A2A Orchestrator v8.0 - Universal Intelligent Orchestrator
A2A SDK 0.2.9 표준 기능 극대화 + 실시간 스트리밍 혁신
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Set

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
    Message,
    Part,
    SendMessageRequest,
    MessageSendParams
)

# A2A 클라이언트 및 디스커버리
from a2a.client import A2ACardResolver, A2AClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeStreamingTaskUpdater(TaskUpdater):
    """실시간 문자 단위 스트리밍을 지원하는 향상된 TaskUpdater"""
    
    def __init__(self, event_queue: EventQueue, task_id: str, context_id: str):
        super().__init__(event_queue, task_id, context_id)
        self._buffer = ""
        self._last_update_time = 0
        self._min_update_interval = 0.05  # 50ms 최소 간격
    
    async def stream_character(self, char: str):
        """단일 문자 스트리밍 (버퍼링 포함)"""
        self._buffer += char
        current_time = time.time()
        
        # 최소 간격이 지났거나 특정 문자인 경우 즉시 전송
        if (current_time - self._last_update_time >= self._min_update_interval or
            char in ['\n', '.', '!', '?', ':', ';']):
            await self._flush_buffer()
    
    async def stream_chunk(self, chunk: str):
        """청크 단위 스트리밍"""
        for char in chunk:
            await self.stream_character(char)
    
    async def stream_line(self, line: str):
        """라인 단위 스트리밍 (즉시 플러시)"""
        self._buffer += line + '\n'
        await self._flush_buffer()
    
    async def _flush_buffer(self):
        """버퍼 플러시"""
        if self._buffer:
            await self.update_status(
                TaskState.working,
                message=self.new_agent_message(parts=[TextPart(text=self._buffer)])
            )
            self._buffer = ""
            self._last_update_time = time.time()
    
    async def stream_markdown_section(self, section_type: str, content: str):
        """Markdown 섹션별 스트리밍"""
        if section_type == "header":
            await self.stream_line(f"\n{content}\n")
        elif section_type == "bullet":
            await self.stream_line(f"- {content}")
        elif section_type == "code":
            await self.stream_line(f"```\n{content}\n```")
        elif section_type == "quote":
            await self.stream_line(f"> {content}")
        else:
            await self.stream_chunk(content)
    
    async def stream_final_response(self, response: str):
        """최종 응답 완료 (버퍼 플러시 후)"""
        await self._flush_buffer()
        await self.update_status(
            TaskState.completed,
            message=self.new_agent_message(parts=[TextPart(text=response)])
        )
    
    async def stream_from_llm(self, stream: AsyncGenerator):
        """LLM 스트림 직접 연동"""
        try:
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    await self.stream_chunk(chunk.choices[0].delta.content)
            await self._flush_buffer()
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            await self._flush_buffer()
    
    # A2A SDK 0.2.9 표준: add_artifact 메서드 추가
    async def add_artifact(self, parts: List[Part], artifact_id: str = None, 
                          name: str = None, metadata: Dict = None) -> None:
        """A2A SDK 0.2.9 표준 add_artifact 메서드 구현"""
        try:
            # 부모 클래스의 add_artifact 메서드 호출
            await super().add_artifact(parts, artifact_id, name, metadata)
            logger.info(f"✅ Artifact '{name}' 전송 완료 (parts: {len(parts)})")
        except Exception as e:
            logger.error(f"❌ Artifact 전송 실패: {e}")
            # 폴백: 스트리밍으로 내용 전송
            if parts and len(parts) > 0:
                for part in parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        await self.stream_line(f"📋 Artifact '{name}': {part.root.text[:200]}...")
                    elif hasattr(part, 'text'):
                        await self.stream_line(f"📋 Artifact '{name}': {part.text[:200]}...")


class DynamicAgentDiscovery:
    """A2A CardResolver를 활용한 동적 에이전트 발견 시스템"""
    
    def __init__(self):
        self.discovered_agents = {}
        self.agent_health_status = {}
        self.last_discovery_time = {}
        self.discovery_interval = 60  # 60초마다 재발견
    
    async def discover_all_agents(self, base_ports: List[int] = None) -> Dict[str, Dict]:
        """모든 에이전트 동적 발견"""
        if base_ports is None:
            # AI DS Team 표준 포트 + 추가 스캔 범위
            base_ports = list(range(8300, 8320))
        
        discovered = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for port in base_ports:
                agent_url = f"http://localhost:{port}"
                
                try:
                    # A2A 표준 에이전트 카드 조회
                    card_resolver = A2ACardResolver(
                        httpx_client=client,
                        base_url=agent_url
                    )
                    agent_card = await card_resolver.get_agent_card()
                    
                    if agent_card:
                        agent_name = agent_card.name
                        
                        discovered[agent_name] = {
                            'url': agent_url,
                            'port': port,
                            'card': agent_card,
                            'skills': [skill.model_dump() for skill in agent_card.skills],
                            'capabilities': agent_card.capabilities.model_dump() if agent_card.capabilities else {},
                            'description': agent_card.description,
                            'version': agent_card.version,
                            'discovered_at': datetime.now().isoformat()
                        }
                        
                        # 헬스 체크
                        health = await self._check_agent_health(agent_url)
                        self.agent_health_status[agent_name] = health
                        
                        logger.info(f"✅ Discovered: {agent_name} on port {port} (Health: {health['status']})")
                        
                except Exception as e:
                    logger.debug(f"Port {port} scan failed: {e}")
        
        self.discovered_agents = discovered
        self.last_discovery_time[datetime.now()] = len(discovered)
        
        logger.info(f"🔍 Total discovered agents: {len(discovered)}")
        return discovered
    
    async def _check_agent_health(self, agent_url: str) -> Dict:
        """에이전트 헬스 체크"""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                start_time = time.time()
                
                # 간단한 헬스 체크 - Agent Card 조회로 대체
                card_resolver = A2ACardResolver(
                    httpx_client=client,
                    base_url=agent_url
                )
                agent_card = await card_resolver.get_agent_card()
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                if agent_card:
                    return {
                        'status': 'healthy',
                        'response_time_ms': round(response_time, 2),
                        'last_check': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': 'No agent card',
                        'last_check': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    async def find_best_agent_for_task(self, task_description: str, required_skills: List[str]) -> Optional[str]:
        """작업에 가장 적합한 에이전트 찾기"""
        best_match = None
        best_score = 0
        
        for agent_name, agent_info in self.discovered_agents.items():
            # 헬스 체크
            if self.agent_health_status.get(agent_name, {}).get('status') != 'healthy':
                continue
            
            score = 0
            
            # 스킬 매칭
            agent_skills = {skill.get('id', '') for skill in agent_info.get('skills', [])}
            for required_skill in required_skills:
                if required_skill in agent_skills:
                    score += 10
            
            # 설명 매칭
            description = agent_info.get('description', '').lower()
            task_lower = task_description.lower()
            
            # 키워드 매칭
            keywords = task_lower.split()
            for keyword in keywords:
                if len(keyword) > 3 and keyword in description:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = agent_name
        
        return best_match
    
    async def get_healthy_agents(self) -> List[str]:
        """건강한 에이전트 목록 반환"""
        healthy = []
        for agent_name, health in self.agent_health_status.items():
            if health.get('status') == 'healthy':
                healthy.append(agent_name)
        return healthy


class StandardA2ACommunicator:
    """A2A Client를 활용한 표준 통신 시스템"""
    
    def __init__(self):
        self.clients = {}  # 에이전트별 클라이언트 캐시
    
    async def get_client(self, agent_url: str) -> A2AClient:
        """에이전트별 A2A 클라이언트 획득 (캐싱)"""
        if agent_url not in self.clients:
            async with httpx.AsyncClient() as client:
                self.clients[agent_url] = A2AClient(
                    httpx_client=client,
                    url=agent_url
                )
        return self.clients[agent_url]
    
    async def send_message(self, agent_url: str, message: str, 
                          stream_callback=None) -> Dict:
        """표준 A2A 메시지 전송"""
        try:
            async with httpx.AsyncClient() as client:
                a2a_client = A2AClient(
                    httpx_client=client,
                    url=agent_url
                )
                
                # 메시지 생성
                msg = Message(
                    messageId=f"orchestrator_{int(time.time() * 1000)}",
                    role="user",
                    parts=[TextPart(text=message)]
                )
                
                # SendMessageRequest 생성
                params = MessageSendParams(message=msg)
                request = SendMessageRequest(
                    id=f"req_{int(time.time() * 1000)}",
                    jsonrpc="2.0",
                    method="message/send",
                    params=params
                )
                
                # 스트리밍 대신 일반 전송 사용 (임시 수정)
                # TODO: A2A SDK 스트리밍 이슈 해결 후 다시 활성화
                response = await a2a_client.send_message(request)
                result = self._parse_a2a_response(response)
                
                # 스트리밍 콜백이 있으면 전체 응답을 한 번에 전달
                if stream_callback and result.get('status') == 'success':
                    content = ""
                    if hasattr(result.get('result'), 'history'):
                        # A2A 응답에서 마지막 에이전트 메시지 추출
                        for msg in result['result'].history:
                            if msg.role == 'agent' and msg.parts:
                                for part in msg.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        content += part.root.text
                                    elif hasattr(part, 'text'):
                                        content += part.text
                    
                    if content:
                        await stream_callback(content)
                
                return result
            
        except Exception as e:
            logger.error(f"A2A communication error with {agent_url}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'summary': f'통신 오류: {str(e)}'
            }
    
    async def _handle_streaming_response(self, client: A2AClient, 
                                       request: SendMessageRequest, 
                                       stream_callback) -> Dict:
        """스트리밍 응답 처리"""
        try:
            # A2A 스트리밍 응답 처리 - send_message_streaming 사용
            full_response = ""
            async for chunk in client.send_message_streaming(request):
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                    if content:
                        full_response += content
                        await stream_callback(content)
                elif hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        full_response += content
                        await stream_callback(content)
            
            return {
                'status': 'success',
                'result': {'content': full_response},
                'summary': '스트리밍 완료'
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'summary': '스트리밍 오류'
            }
    
    def _parse_a2a_response(self, response: Any) -> Dict:
        """A2A 응답 파싱"""
        try:
            if hasattr(response, 'status'):
                return {
                    'status': 'success' if response.status.state == 'completed' else 'partial',
                    'result': response,
                    'summary': '작업 완료'
                }
            else:
                return {
                    'status': 'success',
                    'result': response,
                    'summary': '작업 완료'
                }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'summary': '응답 파싱 오류'
            }


class UniversalIntelligentOrchestratorV8(AgentExecutor):
    """v8.0 - A2A SDK 극대화 + 실시간 스트리밍 혁신"""
    
    def __init__(self):
        # OpenAI 클라이언트
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 Universal Intelligent Orchestrator v8.0 with LLM")
            else:
                self.openai_client = None
                logger.info("📊 Universal Orchestrator v8.0 (No LLM)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        # A2A 시스템 초기화
        self.agent_discovery = DynamicAgentDiscovery()
        self.a2a_communicator = StandardA2ACommunicator()
        
        # 상태 관리
        self.available_agents = {}
        self.agent_capabilities = {}
        self.execution_monitor = None
        self.replanning_engine = None
        
        # 백그라운드 태스크
        self._discovery_task = None
    
    async def initialize(self):
        """시스템 초기화 및 첫 발견"""
        logger.info("🚀 Initializing Universal Orchestrator v8.0...")
        
        # 초기 에이전트 발견
        self.available_agents = await self.agent_discovery.discover_all_agents()
        
        # 백그라운드 발견 태스크 시작
        if not self._discovery_task:
            self._discovery_task = asyncio.create_task(
                self._periodic_agent_discovery()
            )
        
        logger.info(f"✅ Initialization complete with {len(self.available_agents)} agents")
    
    async def _periodic_agent_discovery(self):
        """주기적 에이전트 재발견"""
        while True:
            await asyncio.sleep(60)  # 60초마다
            try:
                new_agents = await self.agent_discovery.discover_all_agents()
                if len(new_agents) != len(self.available_agents):
                    logger.info(f"🔄 Agent landscape changed: {len(new_agents)} agents")
                    self.available_agents = new_agents
            except Exception as e:
                logger.error(f"Periodic discovery error: {e}")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 표준 준수 실행"""
        
        # 실시간 스트리밍 TaskUpdater 생성
        task_updater = RealTimeStreamingTaskUpdater(
            event_queue, 
            context.task_id, 
            context.context_id
        )
        
        try:
            # 사용자 입력 추출
            user_input = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "text":
                        user_input += part.root.text
                    elif hasattr(part, 'text'):
                        user_input += part.text
            
            if not user_input.strip():
                await task_updater.stream_line("❌ 입력 메시지가 없습니다.")
                await task_updater.stream_final_response("오류: 빈 메시지")
                return
            
            # 실시간 복잡도 평가
            await task_updater.stream_markdown_section("header", "### 🧠 요청 분석 중...")
            complexity_result = await self._assess_request_complexity_streaming(user_input, task_updater)
            
            # 구조화된 응답을 Artifact로 전송
            await self._send_structured_response_as_artifact(user_input, complexity_result, task_updater)
            
            # 복잡도별 처리
            complexity = complexity_result.get('level', 'complex')
            
            if complexity == 'simple':
                await self._handle_simple_request_streaming(user_input, task_updater)
            elif complexity == 'single_agent':
                recommended_agent = complexity_result.get('recommended_agent')
                if recommended_agent:
                    await self._handle_single_agent_streaming(user_input, recommended_agent, task_updater)
                else:
                    # 대안 에이전트 찾기
                    alternative = await self._find_alternative_agent_streaming(user_input, task_updater)
                    if alternative:
                        await self._handle_single_agent_streaming(user_input, alternative, task_updater)
                    else:
                        await self._handle_complex_request_streaming(user_input, task_updater)
            else:
                await self._handle_complex_request_streaming(user_input, task_updater)
            
        except Exception as e:
            logger.error(f"Orchestrator execution error: {e}")
            await task_updater.stream_line(f"❌ 실행 중 오류: {str(e)}")
            await task_updater.stream_final_response("오류 발생으로 인해 처리를 완료할 수 없습니다.")
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소 처리"""
        logger.info(f"Task {context.task_id} cancellation requested")
        # 취소 로직 구현 (필요시 에이전트 통신 중단 등)
    
    async def _send_structured_response_as_artifact(self, 
                                                  user_input: str, 
                                                  complexity_result: Dict,
                                                  task_updater: RealTimeStreamingTaskUpdater):
        """A2A SDK 0.2.9 표준: 구조화된 응답을 artifact로 전송"""
        
        try:
            # Streamlit 클라이언트 호환 실행 계획 생성
            execution_plan = {
                "plan_type": "ai_ds_team_orchestration",
                "complexity": complexity_result.get('level', 'complex'),
                "reasoning": complexity_result.get('reasoning', ''),
                "steps": []
            }
            
            if complexity_result.get('level', 'complex') == 'simple':
                # Simple 요청: 직접 응답으로 처리됨을 명시
                execution_plan["steps"] = [{
                    "step_number": 1,
                    "agent_name": "🧠 Universal Orchestrator v8.0",
                    "task_description": "직접 응답 제공",
                    "reasoning": "일반 지식 질문으로 LLM이 직접 응답",
                    "expected_result": "완료된 응답",
                    "status": "completed"
                }]
                
            elif complexity_result.get('level', 'complex') == 'single_agent':
                # Single Agent 요청
                agent_name = complexity_result.get('recommended_agent', 'EDA Tools Agent')
                execution_plan["steps"] = [{
                    "step_number": 1,
                    "agent_name": agent_name,
                    "task_description": user_input,
                    "reasoning": f"{agent_name}의 전문 역량으로 처리",
                    "expected_result": "전문 분석 결과",
                    "status": "completed"
                }]
                
            else:  # complex
                # Complex 요청: 다중 에이전트 계획
                available_agents = await self.agent_discovery.get_healthy_agents()
                
                # 표준 데이터 분석 워크플로우
                standard_workflow = [
                    ("AI_DS_Team DataLoaderToolsAgent", "데이터 로딩 및 전처리"),
                    ("AI_DS_Team DataCleaningAgent", "데이터 정리 및 품질 개선"),
                    ("SessionEDAToolsAgent", "데이터 탐색적 분석"),
                    ("AI_DS_Team DataVisualizationAgent", "데이터 시각화"),
                    ("AI_DS_Team SQLDatabaseAgent", "장비 간 분포 비교 및 이상 탐지"),
                    ("AI_DS_Team DataWranglingAgent", "이상 원인 해석 및 조치 방향 제안")
                ]
                
                for i, (agent, purpose) in enumerate(standard_workflow, 1):
                    execution_plan["steps"].append({
                        "step_number": i,
                        "agent_name": agent,
                        "task_description": purpose,
                        "reasoning": f"{agent}의 전문성을 활용하여 {purpose} 수행",
                        "expected_result": f"{purpose} 완료 결과",
                        "status": "planned"
                    })
            
            # A2A SDK 0.2.9: add_artifact 메서드로 구조화된 계획 전송
            await task_updater.add_artifact(
                parts=[Part(root=TextPart(text=json.dumps(execution_plan, ensure_ascii=False, indent=2)))],
                name="execution_plan",
                metadata={
                    "content_type": "application/json",
                    "plan_type": "ai_ds_team_orchestration",
                    "complexity": complexity_result.get('level', 'complex')
                }
            )
            
            logger.info(f"✅ 구조화된 실행 계획 artifact 전송 완료: {complexity_result.get('level', 'complex')}")
            
        except Exception as e:
            logger.error(f"❌ Structured response artifact 전송 실패: {e}")
            # 폴백: 기본 텍스트 응답
            await task_updater.stream_line(f"⚠️ 구조화된 응답 생성 실패, 기본 응답으로 대체")
    
    async def _assess_request_complexity_streaming(self, 
                                                 user_input: str,
                                                 task_updater: RealTimeStreamingTaskUpdater) -> Dict:
        """요청 복잡도 평가 (데이터 컨텍스트 고려) - 스트리밍"""
        
        if not self.openai_client:
            await task_updater.stream_line("- LLM 없이 기본 복잡도로 처리합니다.")
            return {'level': 'complex', 'reasoning': 'No LLM available'}
        
        # 1단계: 데이터 관련성 사전 검사
        await task_updater.stream_line("- 데이터 관련성을 확인하고 있습니다...")
        data_related = await self._check_data_requirement(user_input)
        
        if data_related:
            await task_updater.stream_line("- 📊 데이터 접근이 필요한 요청으로 감지됨")
            # 데이터 관련 요청은 최소 single_agent 이상
            return await self._assess_data_request_complexity(user_input, task_updater)
        else:
            await task_updater.stream_line("- 💭 일반적 지식 기반 요청으로 감지됨")
            # 일반적 지식만으로 답변 가능한 요청
            return await self._assess_general_request_complexity(user_input, task_updater)
    
    async def _check_data_requirement(self, user_input: str) -> bool:
        """데이터 접근 필요성 검사"""
        try:
            check_prompt = f"""
            다음 요청이 데이터 접근을 필요로 하는지 판단하세요:
            "{user_input}"
            
            데이터 접근이 필요한 경우:
            - 특정 데이터셋의 값, 개수, 통계 조회
            - 데이터 분석, 시각화, 모델링
            - 데이터 기반 비교, 트렌드 분석
            - "이 데이터", "데이터셋", "LOT", "컬럼" 등 언급
            
            일반 지식으로 답변 가능한 경우:
            - 개념 정의, 용어 설명
            - 일반적 방법론, 이론 설명
            - 추상적 질문, 의견 요청
            
            JSON 응답:
            {{
                "requires_data": true/false,
                "reasoning": "판단 근거",
                "data_keywords": ["감지된 데이터 관련 키워드들"]
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": check_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=15.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('requires_data', False)
            
        except Exception as e:
            logger.warning(f"Data requirement check failed: {e}")
            # 실패 시 안전하게 데이터 필요로 가정
            return True
    
    async def _assess_data_request_complexity(self, 
                                            user_input: str,
                                            task_updater: RealTimeStreamingTaskUpdater) -> Dict:
        """데이터 관련 요청의 복잡도 평가"""
        
        assessment_prompt = f"""
        다음은 데이터 접근이 필요한 요청입니다. 복잡도를 평가하세요:
        "{user_input}"
        
        평가 기준:
        1. **single_agent**: 단순 데이터 조회/계산 (개수, 기본 통계, 단일 컬럼 분석)
        2. **complex**: 복합 분석 (다중 변수 분석, 시각화, 모델링, 종합 리포트)
        
        중요: 데이터 관련 요청은 'simple' 분류 금지!
        
        JSON 응답:
        {{
            "level": "single_agent/complex",
            "reasoning": "판단 근거",
            "recommended_agent": "single_agent인 경우 추천 에이전트",
            "data_operations": ["필요한 데이터 작업들"]
        }}
        """
        
        try:
            await task_updater.stream_line("- 데이터 요청 복잡도를 분석하고 있습니다...")
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=30.0
            )
            
            complexity = json.loads(response.choices[0].message.content)
            
            # 결과 스트리밍
            level_emoji = {
                'single_agent': '🤖', 
                'complex': '🎯'
            }
            
            await task_updater.stream_line(
                f"- 복잡도: {level_emoji.get(complexity['level'], '📊')} "
                f"**{complexity['level'].upper()}** (데이터 요청)"
            )
            await task_updater.stream_line(f"- 판단 근거: {complexity['reasoning']}")
            
            return complexity
            
        except Exception as e:
            logger.warning(f"Data request complexity assessment failed: {e}")
            await task_updater.stream_line("- 복잡도 평가 실패, 복합 분석 모드로 진행")
            return {'level': 'complex', 'reasoning': 'Assessment failed - defaulting to complex for data requests'}
    
    async def _assess_general_request_complexity(self, 
                                               user_input: str,
                                               task_updater: RealTimeStreamingTaskUpdater) -> Dict:
        """일반적 지식 기반 요청의 복잡도 평가"""
        
        assessment_prompt = f"""
        다음은 일반적 지식으로 답변 가능한 요청입니다. 복잡도를 평가하세요:
        "{user_input}"
        
        평가 기준:
        1. **simple**: 간단한 정의, 개념 설명, 사실 확인
        2. **single_agent**: 전문적 설명, 방법론 가이드, 비교 분석
        3. **complex**: 다면적 분석, 종합적 가이드, 복합 개념 설명
        
        JSON 응답:
        {{
            "level": "simple/single_agent/complex",
            "reasoning": "판단 근거",
            "recommended_agent": "single_agent인 경우 추천 에이전트",
            "knowledge_areas": ["필요한 지식 영역들"]
        }}
        """
        
        try:
            await task_updater.stream_line("- 일반 요청 복잡도를 분석하고 있습니다...")
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": assessment_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=30.0
            )
            
            complexity = json.loads(response.choices[0].message.content)
            
            # 결과 스트리밍
            level_emoji = {
                'simple': '💬',
                'single_agent': '🤖', 
                'complex': '🎯'
            }
            
            await task_updater.stream_line(
                f"- 복잡도: {level_emoji.get(complexity['level'], '📊')} "
                f"**{complexity['level'].upper()}** (일반 지식)"
            )
            await task_updater.stream_line(f"- 판단 근거: {complexity['reasoning']}")
            
            return complexity
            
        except Exception as e:
            logger.warning(f"General request complexity assessment failed: {e}")
            await task_updater.stream_line("- 복잡도 평가 실패, 기본 모드로 진행")
            return {'level': 'simple', 'reasoning': 'Assessment failed - defaulting to simple for general requests'}
    
    async def _handle_simple_request_streaming(self,
                                             user_input: str,
                                             task_updater: RealTimeStreamingTaskUpdater):
        """간단한 요청 즉답 (스트리밍)"""
        await task_updater.stream_markdown_section(
            "header", "### 💬 즉시 답변"
        )
        
        if self.openai_client:
            try:
                # 스트리밍 응답
                stream = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": "당신은 도움이 되는 AI 어시스턴트입니다. 간결하고 정확하게 답변하세요."
                    }, {
                        "role": "user",
                        "content": user_input
                    }],
                    temperature=0.3,
                    max_tokens=1000,
                    stream=True
                )
                
                await task_updater.stream_line("")
                full_response = ""
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        await task_updater.stream_chunk(content)
                
                await task_updater._flush_buffer()
                
                # 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text=full_response)]
                    )
                )
                
            except Exception as e:
                logger.error(f"Simple request streaming failed: {e}")
                await task_updater.stream_line(f"\n❌ 오류: {str(e)}")
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text=f"오류 발생: {str(e)}")]
                    )
                )
        else:
            response = "죄송합니다. LLM이 설정되지 않아 즉답을 제공할 수 없습니다."
            await task_updater.stream_line(response)
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=response)]
                )
            )
    
    async def _handle_single_agent_streaming(self,
                                           user_input: str,
                                           agent_name: str,
                                           task_updater: RealTimeStreamingTaskUpdater):
        """단일 에이전트 처리 (스트리밍)"""
        await task_updater.stream_markdown_section(
            "header", f"### 🤖 {agent_name} 에이전트 실행"
        )
        
        # 에이전트 확인
        if agent_name not in self.available_agents:
            # 대체 에이전트 찾기
            await task_updater.stream_line("- 지정된 에이전트를 찾을 수 없어 대체 에이전트를 검색합니다...")
            
            agent_name = await self._find_alternative_agent_streaming(
                user_input, task_updater
            )
            
            if not agent_name:
                await task_updater.stream_line("❌ 적합한 에이전트를 찾을 수 없습니다.")
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(
                        parts=[TextPart(text="에이전트 없음")]
                    )
                )
                return
        
        # 에이전트 정보
        agent_info = self.available_agents[agent_name]
        agent_url = agent_info['url']
        
        await task_updater.stream_line(f"- 에이전트: **{agent_name}**")
        await task_updater.stream_line(f"- 상태: ✅ {agent_info.get('version', 'unknown')}")
        
        # 사용자 의도 추출
        await task_updater.stream_line("- 요청 내용을 분석하고 있습니다...")
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        # 정밀한 지시 생성
        await task_updater.stream_line("- 최적의 작업 지시를 생성하고 있습니다...")
        instruction = await self._create_precise_instruction_for_agent(
            agent_name, user_intent, agent_info
        )
        
        # A2A 표준 통신으로 실행 (스트리밍)
        await task_updater.stream_line(f"\n**실행 중...**")
        
        # 스트리밍 콜백
        async def stream_callback(content: str):
            await task_updater.stream_chunk(content)
        
        result = await self.a2a_communicator.send_message(
            agent_url,
            instruction,
            stream_callback=stream_callback
        )
        
        # 결과 검증
        if result['status'] == 'success':
            await task_updater.stream_line(f"\n✅ {agent_name} 작업 완료")
            
            # 최종 응답 생성
            final_response = await self._create_final_response_single(
                user_input, agent_name, result, user_intent
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=final_response)]
                )
            )
        else:
            await task_updater.stream_line(f"\n❌ 실행 실패: {result.get('error', 'Unknown')}")
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text=f"에이전트 실행 실패: {result.get('error')}")]
                )
            )
    
    async def _handle_complex_request_streaming(self,
                                              user_input: str,
                                              task_updater: RealTimeStreamingTaskUpdater):
        """복잡한 요청 처리 (스트리밍)"""
        await task_updater.stream_markdown_section(
            "header", "### 🎯 복잡한 분석 시작"
        )
        
        # Phase 1: 깊이 있는 분석
        await task_updater.stream_line("\n**1단계: 요청 심층 분석**")
        
        request_analysis = await self._analyze_request_depth(user_input)
        user_intent = await self._extract_user_intent_precisely(user_input)
        
        await task_updater.stream_line(f"- 주요 목표: {user_intent['main_goal']}")
        await task_updater.stream_line(f"- 액션 타입: {user_intent['action_type']}")
        
        # Phase 2: 에이전트 준비
        await task_updater.stream_line("\n**2단계: AI 에이전트 준비**")
        
        # 건강한 에이전트만 사용
        healthy_agents = await self.agent_discovery.get_healthy_agents()
        await task_updater.stream_line(f"- 사용 가능한 에이전트: {len(healthy_agents)}개")
        
        for agent_name in healthy_agents[:5]:  # 상위 5개만 표시
            health = self.agent_discovery.agent_health_status.get(agent_name, {})
            response_time = health.get('response_time_ms', 'N/A')
            await task_updater.stream_line(f"  - ✅ {agent_name} ({response_time}ms)")
        
        # Phase 3: 실행 계획 수립
        await task_updater.stream_line("\n**3단계: 최적 실행 계획 수립**")
        
        plan = await self._create_streaming_execution_plan(
            user_input, user_intent, healthy_agents, task_updater
        )
        
        if not plan or not plan.get('steps'):
            await task_updater.stream_line("❌ 실행 계획 수립 실패")
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(
                    parts=[TextPart(text="계획 수립 실패")]
                )
            )
            return
        
        # 계획 표시
        await task_updater.stream_line(f"\n📋 **실행 계획** ({len(plan['steps'])}단계)")
        for i, step in enumerate(plan['steps']):
            await task_updater.stream_line(
                f"{i+1}. **{step['agent']}**: {step['purpose']}"
            )
        
        # Phase 4: 적응적 실행
        await task_updater.stream_line("\n**4단계: 실행 시작**")
        
        execution_result = await self._execute_with_streaming_and_replanning(
            plan, user_intent, task_updater
        )
        
        # Phase 5: 결과 종합
        await task_updater.stream_line("\n**5단계: 결과 종합**")
        
        final_response = await self._create_comprehensive_response_streaming(
            user_input, user_intent, execution_result, task_updater
        )
        
        # 완료
        await task_updater.stream_line("\n🎉 **분석 완료!**")
        
        await task_updater.update_status(
            TaskState.completed,
            message=task_updater.new_agent_message(
                parts=[TextPart(text=final_response)]
            )
        )
    
    async def _find_alternative_agent_streaming(self,
                                              user_input: str,
                                              task_updater: RealTimeStreamingTaskUpdater) -> Optional[str]:
        """대체 에이전트 찾기 (지능형 매칭) - 스트리밍"""
        
        # 건강한 에이전트 목록 확인
        healthy_agents = await self.agent_discovery.get_healthy_agents()
        if not healthy_agents:
            await task_updater.stream_line("- ❌ 사용 가능한 에이전트가 없습니다")
            return None
        
        # LLM 기반 지능형 에이전트 매칭
        if self.openai_client:
            try:
                # 에이전트 정보 수집
                agent_details = {}
                for agent_name in healthy_agents:
                    if agent_name in self.available_agents:
                        info = self.available_agents[agent_name]
                        agent_details[agent_name] = {
                            'description': info.get('description', ''),
                            'skills': [s.get('name', '') for s in info.get('skills', [])]
                        }
                
                matching_prompt = f"""
                사용자 요청: "{user_input}"
                
                사용 가능한 에이전트들:
                {json.dumps(agent_details, ensure_ascii=False, indent=2)}
                
                이 요청에 가장 적합한 에이전트를 선택하세요.
                
                선택 기준:
                1. 데이터 조회/계산: data_loader, eda_tools
                2. 데이터 정제: data_cleaning, data_wrangling  
                3. 시각화: data_visualization
                4. 분석: eda_tools, feature_engineering
                5. 모델링: h2o_ml, mlflow_tools
                6. SQL: sql_database
                
                JSON 응답:
                {{
                    "selected_agent": "선택된 에이전트명",
                    "reasoning": "선택 근거",
                    "confidence": 0.0-1.0
                }}
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": matching_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    timeout=20.0
                )
                
                result = json.loads(response.choices[0].message.content)
                selected_agent = result.get('selected_agent')
                confidence = result.get('confidence', 0.5)
                reasoning = result.get('reasoning', '')
                
                if selected_agent in healthy_agents and confidence > 0.3:
                    await task_updater.stream_line(f"- 🎯 최적 에이전트 선택: **{selected_agent}** (신뢰도: {confidence:.1%})")
                    await task_updater.stream_line(f"- 선택 근거: {reasoning}")
                    return selected_agent
                    
            except Exception as e:
                logger.warning(f"Intelligent agent matching failed: {e}")
                await task_updater.stream_line("- ⚠️ 지능형 매칭 실패, 키워드 기반으로 진행")
        
        # 폴백: 키워드 기반 매칭
        await task_updater.stream_line("- 키워드 기반 에이전트 선택 중...")
        
        # 데이터 관련 키워드 우선순위 매칭
        data_keywords = {
            'count': ['data_loader', 'eda_tools'],
            'lot': ['data_loader', 'eda_tools'], 
            'column': ['data_loader', 'eda_tools'],
            'statistic': ['eda_tools', 'data_visualization'],
            'analysis': ['eda_tools', 'feature_engineering'],
            'visual': ['data_visualization'],
            'chart': ['data_visualization'],
            'plot': ['data_visualization'],
            'clean': ['data_cleaning', 'data_wrangling'],
            'model': ['h2o_ml', 'mlflow_tools'],
            'sql': ['sql_database'],
            'database': ['sql_database']
        }
        
        user_lower = user_input.lower()
        matched_agents = []
        
        for keyword, agents in data_keywords.items():
            if keyword in user_lower:
                for agent in agents:
                    if agent in healthy_agents:
                        matched_agents.append(agent)
        
        if matched_agents:
            # 가장 많이 매칭된 에이전트 선택
            from collections import Counter
            agent_counts = Counter(matched_agents)
            best_agent = agent_counts.most_common(1)[0][0]
            
            await task_updater.stream_line(f"- 키워드 매칭 결과: **{best_agent}**")
            return best_agent
        
        # 최종 폴백: 데이터 관련 기본 에이전트
        default_data_agents = ['data_loader', 'eda_tools', 'data_cleaning']
        for agent in default_data_agents:
            if agent in healthy_agents:
                await task_updater.stream_line(f"- 기본 데이터 에이전트 사용: **{agent}**")
                return agent
        
        # 마지막 폴백: 첫 번째 건강한 에이전트
        if healthy_agents:
            await task_updater.stream_line(f"- 첫 번째 사용 가능 에이전트: **{healthy_agents[0]}**")
            return healthy_agents[0]
        
        return None
    
    async def _create_streaming_execution_plan(self,
                                             user_input: str,
                                             user_intent: Dict,
                                             healthy_agents: List[str],
                                             task_updater: RealTimeStreamingTaskUpdater) -> Dict:
        """실행 계획 수립 (스트리밍 피드백)"""
        
        if not self.openai_client or not healthy_agents:
            return self._create_basic_plan(healthy_agents)
        
        await task_updater.stream_line("- AI가 최적의 실행 경로를 설계하고 있습니다...")
        
        # 에이전트 정보 수집
        agent_details = {}
        for agent_name in healthy_agents:
            if agent_name in self.available_agents:
                info = self.available_agents[agent_name]
                agent_details[agent_name] = {
                    'description': info.get('description', ''),
                    'skills': [s.get('name', '') for s in info.get('skills', [])]
                }
        
        planning_prompt = f"""
        사용자 요청: {user_input}
        사용자 의도: {json.dumps(user_intent, ensure_ascii=False)}
        
        사용 가능한 에이전트:
        {json.dumps(agent_details, ensure_ascii=False, indent=2)}
        
        이 요청을 처리하기 위한 최적의 실행 계획을 수립하세요.
        
        중요:
        - 사용자 의도에 필요한 에이전트만 선택
        - 논리적인 실행 순서 고려
        - 각 단계별 명확한 목적 설정
        
        JSON 형식:
        {{
            "execution_strategy": "전체 전략",
            "steps": [
                {{
                    "agent": "에이전트명",
                    "purpose": "이 단계의 목적",
                    "instruction": "구체적 작업 지시"
                }}
            ]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": planning_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            # 계획 검증
            valid_steps = []
            for step in plan.get('steps', []):
                if step.get('agent') in healthy_agents:
                    valid_steps.append(step)
            
            plan['steps'] = valid_steps
            
            await task_updater.stream_line("- ✅ 실행 계획 수립 완료")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            await task_updater.stream_line("- ⚠️ 기본 계획으로 진행합니다")
            return self._create_basic_plan(healthy_agents)
    
    async def _execute_with_streaming_and_replanning(self,
                                                    plan: Dict,
                                                    user_intent: Dict,
                                                    task_updater: RealTimeStreamingTaskUpdater) -> Dict:
        """적응적 실행 - 스트리밍과 재계획"""
        
        execution_results = []
        current_data_file = None  # 현재 사용 중인 데이터 파일 추적
        
        for i, step in enumerate(plan['steps']):
            agent_name = step['agent']
            purpose = step['purpose']
            
            await task_updater.stream_line(f"\n### 단계 {i+1}: {agent_name}")
            await task_updater.stream_line(f"**목적**: {purpose}")
            
            # 에이전트 정보 확인
            if agent_name not in self.available_agents:
                await task_updater.stream_line(f"❌ 에이전트 '{agent_name}' 사용 불가능")
                continue
            
            agent_info = self.available_agents[agent_name]
            agent_url = agent_info['url']
            
            # 데이터 파일 정보를 포함한 지시사항 생성
            instruction = await self._create_data_aware_instruction(
                agent_name, purpose, user_intent, current_data_file
            )
            
            await task_updater.stream_line(f"**실행 중...**")
            
            # 스트리밍 콜백
            async def agent_stream_callback(content: str):
                # 에이전트 출력을 들여쓰기로 표시
                lines = content.split('\n')
                for line in lines:
                    if line.strip():
                        await task_updater.stream_chunk(f"  {line}\n")
            
            try:
                result = await self.a2a_communicator.send_message(
                    agent_url,
                    instruction,
                    stream_callback=agent_stream_callback
                )
                
                if result['status'] == 'success':
                    await task_updater.stream_line(f"✅ {agent_name} 완료")
                    
                    # Data Loader 결과에서 데이터 파일명 추출
                    if "data loader" in agent_name.lower() or "dataloader" in agent_name.lower():
                        extracted_file = self._extract_data_file_from_result(result)
                        if extracted_file:
                            current_data_file = extracted_file
                            await task_updater.stream_line(f"📁 데이터 파일 설정: {current_data_file}")
                    
                    execution_results.append({
                        'step': i + 1,
                        'agent': agent_name,
                        'purpose': purpose,
                        'result': result,
                        'status': 'success',
                        'data_file': current_data_file  # 데이터 파일 정보 포함
                    })
                else:
                    await task_updater.stream_line(f"❌ {agent_name} 실패: {result.get('error', 'Unknown')}")
                    execution_results.append({
                        'step': i + 1,
                        'agent': agent_name,
                        'purpose': purpose,
                        'result': result,
                        'status': 'failed',
                        'data_file': current_data_file
                    })
                
            except Exception as e:
                await task_updater.stream_line(f"❌ {agent_name} 오류: {str(e)}")
                execution_results.append({
                    'step': i + 1,
                    'agent': agent_name,
                    'purpose': purpose,
                    'error': str(e),
                    'status': 'error',
                    'data_file': current_data_file
                })
        
        return {
            'steps': execution_results,
            'final_data_file': current_data_file,
            'success_count': len([r for r in execution_results if r['status'] == 'success']),
            'total_count': len(execution_results)
        }

    async def _create_data_aware_instruction(self,
                                           agent_name: str,
                                           purpose: str,
                                           user_intent: Dict,
                                           current_data_file: Optional[str]) -> str:
        """데이터 인식 지시 생성 - 명시적 데이터 파일 지정"""
        
        base_instruction = f"{purpose} 작업을 수행하세요."
        
        # 1. 사용자 요청에서 특정 데이터 파일 추출
        user_input = user_intent.get('original_request', '')
        specified_file = None
        
        # 사용자가 명시한 파일명 확인
        import re
        file_patterns = [
            r'([a-zA-Z0-9_]+\.csv)',
            r'([a-zA-Z0-9_]+\.xlsx)',
            r'([a-zA-Z0-9_]+\.pkl)',
            r'ion_implant[^\\s]*',
            r'dataset[^\\s]*'
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                specified_file = match.group(1) if match.group(1).endswith(('.csv', '.xlsx', '.pkl')) else match.group(0)
                break
        
        # 2. 데이터 파일 지시 추가
        if specified_file:
            data_instruction = f"""

🔬 **사용할 데이터 파일**: {specified_file}
📁 **데이터 파일 우선순위**: 
   1. {specified_file} (사용자 지정)
   2. ion_implant 관련 파일 (반도체 분석용)
   3. 가장 최근 파일

⚠️ **중요**: 반드시 위 우선순위에 따라 데이터 파일을 선택하고, 선택된 파일명을 응답에 명시해주세요.
"""
        elif current_data_file:
            data_instruction = f"""

🔬 **사용할 데이터 파일**: {current_data_file}
📁 **데이터 연속성**: 이전 단계에서 사용된 파일과 동일한 파일을 사용하세요.

⚠️ **중요**: 반드시 {current_data_file} 파일을 사용하고, 응답에 파일명을 명시해주세요.
"""
        else:
            data_instruction = f"""

🔬 **데이터 파일 선택 기준**:
   1. ion_implant 관련 파일 우선 (반도체 분석 특화)
   2. 가장 최근 수정된 파일
   3. 사용 가능한 첫 번째 파일

⚠️ **중요**: 선택된 데이터 파일명을 응답에 반드시 명시해주세요.
"""
        
        # 3. 에이전트별 맞춤 지시
        agent_specific_instructions = {
            "AI_DS_Team DataLoaderToolsAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- 데이터 파일을 로드하고 기본 정보를 확인
- 데이터 크기, 컬럼 정보, 데이터 타입 확인
- 사용된 파일명을 명확히 표시
""",
            "AI_DS_Team DataCleaningAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- 데이터 품질 검사 및 정리
- 결측값, 중복값, 이상값 확인
- 사용된 파일명을 명확히 표시
""",
            "SessionEDAToolsAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- 탐색적 데이터 분석 수행
- 기술통계, 분포 분석, 상관관계 분석
- 사용된 파일명을 명확히 표시
""",
            "AI_DS_Team DataVisualizationAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- 데이터 시각화 차트 생성
- 분포도, 트렌드 차트, 상관관계 히트맵
- 사용된 파일명을 명확히 표시
""",
            "AI_DS_Team DataWranglingAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- 데이터 변환 및 가공
- 파생 변수 생성, 데이터 형태 변환
- 사용된 파일명을 명확히 표시
""",
            "AI_DS_Team SQLDatabaseAgent": f"""
{base_instruction}

{data_instruction}

📋 **작업 세부사항**:
- SQL 쿼리를 통한 데이터 분석
- 집계, 필터링, 조인 등의 데이터 처리
- 사용된 파일명을 명확히 표시
"""
        }
        
        # 4. 최종 지시 반환
        final_instruction = agent_specific_instructions.get(agent_name, f"{base_instruction}{data_instruction}")
        
        return final_instruction

    def _extract_data_file_from_result(self, result: Dict) -> Optional[str]:
        """에이전트 결과에서 데이터 파일명 추출"""
        try:
            # 응답 텍스트에서 파일명 패턴 찾기
            response_text = str(result.get('content', ''))
            
            # CSV 파일 패턴 검색
            import re
            csv_pattern = r'([a-zA-Z0-9_]+\.csv)'
            matches = re.findall(csv_pattern, response_text)
            
            if matches:
                # ion_implant 파일 우선 반환
                for match in matches:
                    if 'ion_implant' in match.lower():
                        return match
                # 첫 번째 매치 반환
                return matches[0]
            
            # 특정 키워드로 파일명 추론
            if 'ion_implant' in response_text.lower():
                return 'ion_implant_3lot_dataset.csv'
            
            return None
            
        except Exception as e:
            logger.warning(f"데이터 파일명 추출 실패: {e}")
            return None
    
    async def _create_comprehensive_response_streaming(self,
                                                     user_input: str,
                                                     user_intent: Dict,
                                                     execution_result: Dict,
                                                     task_updater: RealTimeStreamingTaskUpdater) -> str:
        """종합 응답 생성 (스트리밍)"""
        
        await task_updater.stream_line("- 분석 결과를 종합하고 있습니다...")
        
        if not self.openai_client:
            return self._create_basic_summary(execution_result)
        
        # execution_result 구조 수정: steps에서 성공한 결과만 추출
        successful_results = {}
        failed_agents = []
        
        if 'steps' in execution_result:
            for step in execution_result['steps']:
                agent_name = step.get('agent', 'Unknown')
                if step.get('status') == 'success':
                    successful_results[agent_name] = step
                elif step.get('status') == 'failed':
                    failed_agents.append(f"- {agent_name}: {step.get('error', 'Unknown error')}")
        
        if not successful_results:
            # LLM을 사용하여 대안 응답 생성
            if self.openai_client:
                try:
                    fallback_prompt = f"""
                    사용자 요청: {user_input}
                    
                    AI 에이전트들과의 통신이 실패했지만, 당신의 지식을 바탕으로 사용자 요청에 최대한 도움이 되는 답변을 제공하세요.
                    
                    실패한 에이전트들:
                    {chr(10).join(failed_agents) if failed_agents else '- 모든 에이전트 통신 실패'}
                    
                    지침:
                    1. 사용자 요청의 핵심 내용에 대해 직접 답변
                    2. 일반적인 접근 방법과 고려사항 제시
                    3. 실무에서 활용할 수 있는 구체적인 조언 포함
                    4. 추가 정보가 필요한 부분 명시
                    """
                    
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": fallback_prompt}],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    
                    await task_updater.stream_line("- ⚠️ 에이전트 통신 실패, LLM 기반 대안 응답 생성")
                    return response.choices[0].message.content
                    
                except Exception as e:
                    logger.error(f"Fallback response generation failed: {e}")
            
            # 최후 수단: 기본 응답
            return f"""
## 분석 요청 처리 결과

⚠️ **AI 에이전트 시스템과의 통신에 문제가 발생했습니다.**

### 요청 내용
{user_input[:200]}{"..." if len(user_input) > 200 else ""}

### 발생한 문제
{chr(10).join(failed_agents) if failed_agents else "- 모든 AI 에이전트와의 통신이 실패했습니다."}

### 권장 조치
1. 잠시 후 다시 시도해 주세요
2. 요청을 더 구체적으로 작성해 보세요
3. 단계별로 나누어 요청해 보세요

죄송합니다. 현재 시스템 상태로는 완전한 분석을 제공할 수 없습니다.
            """
        
        synthesis_prompt = f"""
        사용자 요청: {user_input}
        사용자 의도: {json.dumps(user_intent, ensure_ascii=False)}
        
        분석 결과:
        {json.dumps(successful_results, ensure_ascii=False, indent=2)[:3000]}
        
        위 결과를 바탕으로 사용자 요청에 대한 종합적인 답변을 작성하세요.
        
        지침:
        1. 사용자의 {user_intent['action_type']} 요청에 직접 답변
        2. 구체적인 데이터와 근거 제시
        3. 핵심 발견사항 강조
        4. 실용적인 인사이트 제공
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            await task_updater.stream_line("- ✅ 종합 완료")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._create_basic_summary(execution_result)
    
    async def _extract_user_intent_precisely(self, user_input: str) -> Dict:
        """사용자 의도 정밀 추출"""
        
        if not self.openai_client:
            return {
                'main_goal': user_input,
                'action_type': 'analyze',
                'specific_requirements': [],
                'expected_outcomes': ['분석 결과']
            }
        
        intent_prompt = f"""
        사용자 입력: "{user_input}"
        
        의도를 추출하세요:
        1. action_type: analyze/verify/recommend/diagnose/predict/compare/explain
        2. main_goal: 한 문장 요약
        3. specific_requirements: 구체적 요구사항
        4. expected_outcomes: 기대 결과
        
        JSON 형식으로 응답하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": intent_prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Intent extraction failed: {e}")
            return {
                'main_goal': user_input,
                'action_type': 'analyze',
                'specific_requirements': [],
                'expected_outcomes': ['분석 결과']
            }
    
    async def _analyze_request_depth(self, user_input: str) -> Dict:
        """요청 깊이 분석"""
        
        if not self.openai_client:
            return {
                'detail_level': 5,
                'has_role_description': False,
                'explicit_requirements': ['기본 분석'],
                'implicit_needs': ['데이터 이해']
            }
        
        try:
            analysis_prompt = f"""
            요청 분석: "{user_input}"
            
            분석 항목:
            1. 구체성 수준 (1-10)
            2. 명시적 vs 암시적 요구사항
            3. 예상 응답 깊이
            
            JSON 응답 요청
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"Depth analysis failed: {e}")
            return {
                'detail_level': 5,
                'explicit_requirements': ['기본 분석']
            }
    
    async def _create_precise_instruction_for_agent(self,
                                                  agent_name: str,
                                                  user_intent: Dict,
                                                  agent_info: Dict) -> str:
        """에이전트별 정밀 지시 생성"""
        
        if not self.openai_client:
            return f"{user_intent['main_goal']}을 수행해주세요."
        
        # 에이전트 스킬 정보
        skills = agent_info.get('skills', [])
        skill_names = [s.get('name', '') for s in skills]
        
        instruction_prompt = f"""
        에이전트: {agent_name}
        에이전트 스킬: {skill_names}
        
        사용자 목표: {user_intent['main_goal']}
        액션 타입: {user_intent['action_type']}
        
        이 에이전트가 수행해야 할 구체적인 작업 지시를 작성하세요.
        에이전트의 스킬을 최대한 활용하도록 지시하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Instruction generation failed: {e}")
            return f"{user_intent['main_goal']}을 수행해주세요."
    
    async def _create_final_response_single(self,
                                          user_input: str,
                                          agent_name: str,
                                          result: Dict,
                                          user_intent: Dict) -> str:
        """단일 에이전트 최종 응답"""
        
        if not self.openai_client:
            return f"{agent_name} 에이전트 실행 완료:\n{result.get('summary', '작업 완료')}"
        
        response_prompt = f"""
        사용자 요청: {user_input}
        사용자 의도: {user_intent['main_goal']}
        
        {agent_name} 에이전트 실행 결과:
        {json.dumps(result, ensure_ascii=False)[:2000]}
        
        사용자 요청에 직접 답변하는 응답을 작성하세요.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"{agent_name} 작업 완료"
    
    def _create_basic_plan(self, available_agents: List[str]) -> Dict:
        """기본 실행 계획"""
        steps = []
        
        # 기본 워크플로우
        basic_flow = [
            ('data_loader', '데이터 로드'),
            ('data_cleaning', '데이터 정제'),
            ('eda_tools', '탐색적 분석'),
            ('data_visualization', '시각화')
        ]
        
        for agent, purpose in basic_flow:
            if agent in available_agents:
                steps.append({
                    'agent': agent,
                    'purpose': purpose,
                    'instruction': f'{purpose} 작업을 수행하세요'
                })
        
        return {
            'execution_strategy': '표준 데이터 분석',
            'steps': steps
        }
    
    def _create_basic_summary(self, execution_result: Dict) -> str:
        """기본 요약"""
        total = execution_result.get('total_count', 0)
        success = execution_result.get('success_count', 0)
        
        summary = f"## 분석 완료\n\n"
        summary += f"- 총 {total}단계 중 {success}단계 성공\n"
        
        # steps 구조에서 정보 추출
        if 'steps' in execution_result:
            for step in execution_result['steps']:
                agent_name = step.get('agent', 'Unknown')
                if step.get('status') == 'success':
                    summary += f"- ✅ {agent_name}: 완료\n"
                else:
                    summary += f"- ❌ {agent_name}: 실패\n"
        
        return summary


def create_orchestrator_v8_server():
    """v8.0 서버 생성"""
    
    agent_card = AgentCard(
        name="Universal Intelligent Orchestrator v8.0",
        description="A2A SDK 0.2.9 표준 기능 극대화 + 실시간 스트리밍 혁신",
        url="http://localhost:8100",
        version="8.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="real_time_streaming",
                name="Real-Time Character Streaming",
                description="문자 단위 실시간 스트리밍으로 즉각적인 피드백 제공",
                tags=["streaming", "real-time", "responsive"]
            ),
            AgentSkill(
                id="dynamic_discovery",
                name="Dynamic Agent Discovery with A2A",
                description="A2A CardResolver를 활용한 실시간 에이전트 발견 및 헬스 체크",
                tags=["discovery", "a2a", "health-check"]
            ),
            AgentSkill(
                id="standard_a2a_communication",
                name="Standard A2A Protocol Communication",
                description="A2AClient를 통한 표준화된 에이전트 통신",
                tags=["a2a", "protocol", "standard"]
            ),
            AgentSkill(
                id="adaptive_complexity_handling",
                name="Adaptive Complexity Processing",
                description="요청 복잡도에 따른 적응적 처리 (Simple/Single/Complex)",
                tags=["adaptive", "intelligent", "complexity"]
            ),
            AgentSkill(
                id="llm_streaming_integration",
                name="LLM Streaming Integration",
                description="OpenAI 스트리밍과 직접 통합된 실시간 응답",
                tags=["llm", "streaming", "openai"]
            )
        ]
    )
    
    executor = UniversalIntelligentOrchestratorV8()
    
    # 초기화를 위한 비동기 태스크
    async def startup():
        await executor.initialize()
    
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    # Starlette 앱에 startup 이벤트 추가
    starlette_app = app.build()
    starlette_app.add_event_handler("startup", startup)
    
    return starlette_app


def main():
    """메인 실행"""
    logger.info("🚀 Starting Universal Intelligent Orchestrator v8.0")
    logger.info("📡 A2A SDK 0.2.9 Features: Dynamic Discovery + Real-Time Streaming")
    
    app = create_orchestrator_v8_server()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )


if __name__ == "__main__":
    main()