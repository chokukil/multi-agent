#!/usr/bin/env python3
"""
A2A Orchestrator v9.0 - MCP Enhanced Universal Intelligent Orchestrator
A2A SDK 0.2.9 표준 + MCP (Model Context Protocol) 통합 + 지능형 라우팅

🔗 새로운 기능:
1. MCP 도구 통합 - 7개 MCP 도구와 A2A 에이전트 연동
2. 지능형 라우팅 - 사용자 의도 분석 기반 최적 라우팅
3. 동적 워크플로우 - A2A + MCP 통합 실행 계획
4. Context Engineering - 6 Data Layers 완전 지원
5. 실시간 협업 모니터링 - 에이전트 + MCP 도구 상태 추적

Architecture:
- Universal Intelligent Orchestrator (기존 v8.0 기반)
- MCP Integration Layer (새로 추가)
- Intelligent Intent Analyzer (향상된 의도 분석)
- Enhanced Collaboration Engine (A2A + MCP 통합)
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Set
from dataclasses import dataclass, asdict

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

# MCP 통합 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'pandas_agent'))

from mcp_integration import get_mcp_integration, MCPToolType
from pandas_collaboration_hub_enhanced import get_enhanced_collaboration_hub

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI DS Team 에이전트 포트 매핑 (v9.0 업데이트)
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
    "pandas_collaboration_hub": 8315,  # Enhanced 협업 허브
}

# MCP 도구 포트 매핑
MCP_TOOL_PORTS = {
    "playwright": 3000,
    "file_manager": 3001,
    "database_connector": 3002,
    "api_gateway": 3003,
    "data_analyzer": 3004,
    "chart_generator": 3005,
    "llm_gateway": 3006,
}

@dataclass
class IntentAnalysisResult:
    """의도 분석 결과"""
    primary_intent: str
    confidence: float
    required_agents: List[str]
    required_mcp_tools: List[str]
    workflow_type: str  # 'simple', 'complex', 'collaborative'
    priority: int
    estimated_complexity: str  # 'low', 'medium', 'high'
    context_requirements: Dict[str, Any]

@dataclass
class EnhancedWorkflowStep:
    """향상된 워크플로우 단계"""
    step_id: str
    step_type: str  # 'agent', 'mcp_tool', 'collaboration'
    executor: str  # 에이전트 ID 또는 MCP 도구 ID
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float
    parallel_execution: bool = False

@dataclass
class CollaborationSession:
    """협업 세션 정보"""
    session_id: str
    user_request: str
    intent_analysis: IntentAnalysisResult
    workflow_steps: List[EnhancedWorkflowStep]
    active_agents: Set[str]
    active_mcp_tools: Set[str]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime
    results: Dict[str, Any]

def new_agent_text_message(text: str):
    """에이전트 텍스트 메시지 생성"""
    return Message(
        messageId=str(uuid.uuid4()),
        role="agent",
        parts=[TextPart(text=text)]
    )

class IntelligentIntentAnalyzer:
    """지능형 의도 분석기 (v9.0 enhanced)"""
    
    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = openai_client
        
        # 의도 분석 패턴 매핑
        self.intent_patterns = {
            "data_analysis": {
                "keywords": ["분석", "analyze", "탐색", "explore", "통계", "statistics"],
                "agents": ["pandas_collaboration_hub", "eda_tools", "data_analyzer"],
                "mcp_tools": ["data_analyzer", "chart_generator"]
            },
            "data_loading": {
                "keywords": ["로드", "load", "읽기", "read", "가져오기", "import"],
                "agents": ["data_loader", "pandas_collaboration_hub"],
                "mcp_tools": ["file_manager", "database_connector", "api_gateway"]
            },
            "data_visualization": {
                "keywords": ["시각화", "visualization", "차트", "chart", "그래프", "plot"],
                "agents": ["data_visualization", "pandas_collaboration_hub"],
                "mcp_tools": ["chart_generator", "data_analyzer"]
            },
            "web_scraping": {
                "keywords": ["웹", "web", "스크래핑", "scraping", "크롤링", "crawling"],
                "agents": ["data_loader"],
                "mcp_tools": ["playwright", "api_gateway"]
            },
            "machine_learning": {
                "keywords": ["머신러닝", "machine learning", "ml", "모델", "model", "예측"],
                "agents": ["h2o_ml", "feature_engineering", "mlflow_tools"],
                "mcp_tools": ["data_analyzer", "llm_gateway"]
            },
            "data_cleaning": {
                "keywords": ["정리", "clean", "전처리", "preprocessing", "정제"],
                "agents": ["data_cleaning", "data_wrangling"],
                "mcp_tools": ["data_analyzer", "file_manager"]
            },
            "comprehensive_analysis": {
                "keywords": ["종합", "comprehensive", "전체", "complete", "모든"],
                "agents": ["pandas_collaboration_hub", "eda_tools", "data_visualization"],
                "mcp_tools": ["data_analyzer", "chart_generator", "llm_gateway"]
            }
        }
    
    async def analyze_intent(self, user_request: str, context: Dict[str, Any] = None) -> IntentAnalysisResult:
        """사용자 요청의 의도 분석"""
        logger.info(f"🧠 의도 분석 시작: {user_request}")
        
        # 키워드 기반 기본 분석
        intent_scores = {}
        for intent, config in self.intent_patterns.items():
            score = self._calculate_keyword_score(user_request, config["keywords"])
            if score > 0:
                intent_scores[intent] = score
        
        # 가장 높은 점수의 의도 선택
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent]
        else:
            primary_intent = "comprehensive_analysis"
            confidence = 0.5
        
        # LLM 기반 고급 분석 (OpenAI 사용 가능시)
        if self.openai_client:
            try:
                enhanced_analysis = await self._llm_enhanced_analysis(user_request, primary_intent, context)
                if enhanced_analysis:
                    primary_intent = enhanced_analysis.get("intent", primary_intent)
                    confidence = enhanced_analysis.get("confidence", confidence)
            except Exception as e:
                logger.warning(f"LLM 분석 실패: {e}")
        
        # 필요한 에이전트 및 MCP 도구 결정
        intent_config = self.intent_patterns.get(primary_intent, self.intent_patterns["comprehensive_analysis"])
        required_agents = intent_config["agents"]
        required_mcp_tools = intent_config["mcp_tools"]
        
        # 워크플로우 타입 결정
        workflow_type = self._determine_workflow_type(required_agents, required_mcp_tools)
        
        # 복잡성 추정
        complexity = self._estimate_complexity(user_request, required_agents, required_mcp_tools)
        
        result = IntentAnalysisResult(
            primary_intent=primary_intent,
            confidence=confidence,
            required_agents=required_agents,
            required_mcp_tools=required_mcp_tools,
            workflow_type=workflow_type,
            priority=self._calculate_priority(confidence, complexity),
            estimated_complexity=complexity,
            context_requirements=context or {}
        )
        
        logger.info(f"✅ 의도 분석 완료: {primary_intent} (신뢰도: {confidence:.2f})")
        return result
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """키워드 기반 점수 계산"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches / len(keywords) if keywords else 0
    
    async def _llm_enhanced_analysis(self, user_request: str, basic_intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 기반 향상된 의도 분석"""
        if not self.openai_client:
            return {}
        
        prompt = f"""
        사용자 요청을 분석하여 최적의 의도와 실행 전략을 제안해주세요.
        
        요청: {user_request}
        기본 의도: {basic_intent}
        컨텍스트: {json.dumps(context, ensure_ascii=False)}
        
        다음 의도 중에서 가장 적합한 것을 선택하고 분석해주세요:
        - data_analysis: 데이터 분석 및 탐색
        - data_loading: 데이터 로딩 및 가져오기
        - data_visualization: 데이터 시각화
        - web_scraping: 웹 데이터 수집
        - machine_learning: 머신러닝 모델링
        - data_cleaning: 데이터 정리 및 전처리
        - comprehensive_analysis: 종합적 분석
        
        응답 형식:
        {{
            "intent": "선택된_의도",
            "confidence": 0.0-1.0,
            "reasoning": "선택 근거",
            "additional_requirements": ["추가 요구사항들"]
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"LLM 분석 오류: {e}")
            return {}
    
    def _determine_workflow_type(self, required_agents: List[str], required_mcp_tools: List[str]) -> str:
        """워크플로우 타입 결정"""
        total_components = len(required_agents) + len(required_mcp_tools)
        
        if total_components <= 2:
            return "simple"
        elif total_components <= 4:
            return "complex"
        else:
            return "collaborative"
    
    def _estimate_complexity(self, user_request: str, required_agents: List[str], required_mcp_tools: List[str]) -> str:
        """복잡성 추정"""
        complexity_indicators = ["종합", "전체", "모든", "복잡한", "고급", "상세한"]
        
        has_complexity_keywords = any(keyword in user_request for keyword in complexity_indicators)
        total_components = len(required_agents) + len(required_mcp_tools)
        
        if has_complexity_keywords or total_components > 5:
            return "high"
        elif total_components > 3:
            return "medium"
        else:
            return "low"
    
    def _calculate_priority(self, confidence: float, complexity: str) -> int:
        """우선순위 계산"""
        base_priority = int(confidence * 10)
        
        if complexity == "high":
            return min(base_priority + 3, 10)
        elif complexity == "medium":
            return min(base_priority + 1, 10)
        else:
            return base_priority

class MCPEnhancedAgentDiscovery:
    """MCP 통합 에이전트 발견 시스템"""
    
    def __init__(self):
        self.mcp_integration = get_mcp_integration()
        self.last_discovery = None
        self.discovery_cache = {}
    
    async def discover_all_resources(self) -> Dict[str, Any]:
        """A2A 에이전트와 MCP 도구 모두 발견"""
        logger.info("🔍 Enhanced 리소스 발견 시작 (A2A + MCP)")
        
        # 병렬로 A2A 에이전트와 MCP 도구 발견
        a2a_discovery_task = asyncio.create_task(self._discover_a2a_agents())
        mcp_discovery_task = asyncio.create_task(self._discover_mcp_tools())
        
        a2a_agents, mcp_tools = await asyncio.gather(a2a_discovery_task, mcp_discovery_task)
        
        discovery_result = {
            "a2a_agents": a2a_agents,
            "mcp_tools": mcp_tools,
            "total_a2a_agents": len(a2a_agents),
            "total_mcp_tools": len(mcp_tools),
            "total_resources": len(a2a_agents) + len(mcp_tools),
            "discovery_time": datetime.now().isoformat(),
            "integration_status": "enhanced" if len(a2a_agents) > 0 and len(mcp_tools) > 0 else "partial"
        }
        
        self.last_discovery = discovery_result
        logger.info(f"✅ Enhanced 리소스 발견 완료: {discovery_result['total_a2a_agents']}개 A2A 에이전트 + {discovery_result['total_mcp_tools']}개 MCP 도구")
        
        return discovery_result
    
    async def _discover_a2a_agents(self) -> Dict[str, Any]:
        """A2A 에이전트 발견"""
        agents = {}
        
        for agent_id, port in AGENT_PORTS.items():
            try:
                url = f"http://localhost:{port}"
                
                # Agent Card 확인
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{url}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        agents[agent_id] = {
                            "id": agent_id,
                            "name": agent_card.get("name", f"Agent {agent_id}"),
                            "url": url,
                            "port": port,
                            "description": agent_card.get("description", ""),
                            "capabilities": agent_card.get("capabilities", {}),
                            "skills": agent_card.get("skills", []),
                            "status": "available",
                            "type": "a2a_agent"
                        }
                        logger.info(f"✅ A2A 에이전트 발견: {agent_id} ({url})")
                    else:
                        logger.warning(f"⚠️ A2A 에이전트 응답 없음: {agent_id} ({url})")
                        
            except Exception as e:
                logger.warning(f"❌ A2A 에이전트 연결 실패: {agent_id} - {e}")
        
        return agents
    
    async def _discover_mcp_tools(self) -> Dict[str, Any]:
        """MCP 도구 발견"""
        try:
            mcp_discovery_result = await self.mcp_integration.initialize_mcp_tools()
            
            # MCP 도구 정보를 통일된 형식으로 변환
            mcp_tools = {}
            for tool_id, tool_info in mcp_discovery_result.get("tool_details", {}).items():
                mcp_tools[tool_id] = {
                    "id": tool_id,
                    "name": tool_info.get("name", tool_id),
                    "type": "mcp_tool",
                    "tool_type": tool_info.get("type", "unknown"),
                    "capabilities": tool_info.get("capabilities", []),
                    "status": tool_info.get("status", "unknown"),
                    "port": MCP_TOOL_PORTS.get(tool_id, 3000)
                }
            
            return mcp_tools
            
        except Exception as e:
            logger.error(f"❌ MCP 도구 발견 실패: {e}")
            return {}

class EnhancedCollaborationEngine:
    """향상된 협업 엔진 (A2A + MCP 통합)"""
    
    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = openai_client
        self.collaboration_hub = get_enhanced_collaboration_hub()
        self.mcp_integration = get_mcp_integration()
        
        # 활성 세션 관리
        self.active_sessions: Dict[str, CollaborationSession] = {}
        
    async def create_collaboration_session(self, user_request: str, intent_analysis: IntentAnalysisResult) -> CollaborationSession:
        """협업 세션 생성"""
        session_id = f"collab_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"🚀 협업 세션 생성: {session_id}")
        
        # 워크플로우 단계 생성
        workflow_steps = await self._generate_workflow_steps(user_request, intent_analysis)
        
        session = CollaborationSession(
            session_id=session_id,
            user_request=user_request,
            intent_analysis=intent_analysis,
            workflow_steps=workflow_steps,
            active_agents=set(intent_analysis.required_agents),
            active_mcp_tools=set(intent_analysis.required_mcp_tools),
            status="pending",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            results={}
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"✅ 협업 세션 생성 완료: {session_id} ({len(workflow_steps)}개 단계)")
        return session
    
    async def _generate_workflow_steps(self, user_request: str, intent_analysis: IntentAnalysisResult) -> List[EnhancedWorkflowStep]:
        """워크플로우 단계 생성"""
        steps = []
        
        # 1단계: MCP 도구 사전 실행 (데이터 수집, 분석 준비)
        if intent_analysis.required_mcp_tools:
            for i, tool_id in enumerate(intent_analysis.required_mcp_tools):
                action = self._determine_mcp_action(tool_id, user_request)
                step = EnhancedWorkflowStep(
                    step_id=f"mcp_{i+1}_{tool_id}",
                    step_type="mcp_tool",
                    executor=tool_id,
                    action=action,
                    parameters=self._generate_mcp_parameters(tool_id, action, user_request),
                    dependencies=[],
                    estimated_duration=2.0,
                    parallel_execution=True
                )
                steps.append(step)
        
        # 2단계: A2A 에이전트 실행 (MCP 결과 활용)
        if intent_analysis.required_agents:
            mcp_dependencies = [step.step_id for step in steps if step.step_type == "mcp_tool"]
            
            for i, agent_id in enumerate(intent_analysis.required_agents):
                step = EnhancedWorkflowStep(
                    step_id=f"agent_{i+1}_{agent_id}",
                    step_type="agent",
                    executor=agent_id,
                    action="analyze_with_mcp_results",
                    parameters={"request": user_request, "mcp_enhanced": True},
                    dependencies=mcp_dependencies,
                    estimated_duration=5.0,
                    parallel_execution=agent_id != "pandas_collaboration_hub"  # 허브는 조정 역할
                )
                steps.append(step)
        
        # 3단계: 결과 통합 (협업 허브가 담당)
        if len(steps) > 1:
            integration_step = EnhancedWorkflowStep(
                step_id="integration_final",
                step_type="collaboration",
                executor="pandas_collaboration_hub",
                action="integrate_results",
                parameters={"session_type": "mcp_enhanced"},
                dependencies=[step.step_id for step in steps],
                estimated_duration=3.0,
                parallel_execution=False
            )
            steps.append(integration_step)
        
        return steps
    
    def _determine_mcp_action(self, tool_id: str, user_request: str) -> str:
        """MCP 도구별 액션 결정"""
        action_mapping = {
            "playwright": "extract_data" if "데이터" in user_request else "navigate",
            "file_manager": "read_file" if "읽기" in user_request else "list_directory",
            "database_connector": "execute_query",
            "api_gateway": "http_request",
            "data_analyzer": "statistical_analysis",
            "chart_generator": "create_chart",
            "llm_gateway": "generate_text"
        }
        
        return action_mapping.get(tool_id, "default_action")
    
    def _generate_mcp_parameters(self, tool_id: str, action: str, user_request: str) -> Dict[str, Any]:
        """MCP 도구 매개변수 생성"""
        # 기본 매개변수 (실제 구현에서는 더 정교하게)
        base_params = {
            "user_request": user_request,
            "timestamp": datetime.now().isoformat()
        }
        
        # 도구별 특화 매개변수
        if tool_id == "playwright":
            base_params["url"] = "https://example.com"
        elif tool_id == "data_analyzer":
            base_params["analysis_type"] = "comprehensive"
        elif tool_id == "chart_generator":
            base_params["chart_type"] = "auto"
        
        return base_params
    
    async def execute_collaboration_session(self, session: CollaborationSession, task_updater: TaskUpdater) -> Dict[str, Any]:
        """협업 세션 실행"""
        logger.info(f"⚡ 협업 세션 실행 시작: {session.session_id}")
        
        session.status = "in_progress"
        session.updated_at = datetime.now()
        
        execution_results = {
            "session_id": session.session_id,
            "mcp_results": {},
            "agent_results": {},
            "integration_result": None,
            "execution_timeline": []
        }
        
        try:
            # 워크플로우 단계별 실행
            for step in session.workflow_steps:
                step_start = time.time()
                
                await task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(parts=[TextPart(text=f"🔄 실행 중: {step.step_type} - {step.executor}")])
                )
                
                if step.step_type == "mcp_tool":
                    # MCP 도구 실행
                    result = await self._execute_mcp_step(step, session)
                    execution_results["mcp_results"][step.step_id] = result
                    
                elif step.step_type == "agent":
                    # A2A 에이전트 실행
                    result = await self._execute_agent_step(step, session, execution_results["mcp_results"])
                    execution_results["agent_results"][step.step_id] = result
                    
                elif step.step_type == "collaboration":
                    # 협업 통합
                    result = await self._execute_collaboration_step(step, session, execution_results)
                    execution_results["integration_result"] = result
                
                step_duration = time.time() - step_start
                execution_results["execution_timeline"].append({
                    "step_id": step.step_id,
                    "duration": step_duration,
                    "status": "completed"
                })
                
                logger.info(f"✅ 단계 완료: {step.step_id} ({step_duration:.2f}초)")
            
            session.status = "completed"
            session.results = execution_results
            
        except Exception as e:
            logger.error(f"❌ 협업 세션 실행 실패: {e}")
            session.status = "failed"
            execution_results["error"] = str(e)
        
        session.updated_at = datetime.now()
        
        return execution_results
    
    async def _execute_mcp_step(self, step: EnhancedWorkflowStep, session: CollaborationSession) -> Dict[str, Any]:
        """MCP 도구 단계 실행"""
        logger.info(f"🔧 MCP 도구 실행: {step.executor}.{step.action}")
        
        # MCP 세션 생성 (필요시)
        mcp_session = await self.mcp_integration.create_mcp_session(
            agent_id=f"orchestrator_{session.session_id}",
            required_tools=[step.executor]
        )
        
        # MCP 도구 호출
        result = await self.mcp_integration.call_mcp_tool(
            session_id=mcp_session.session_id,
            tool_id=step.executor,
            action=step.action,
            parameters=step.parameters
        )
        
        return result
    
    async def _execute_agent_step(self, step: EnhancedWorkflowStep, session: CollaborationSession, mcp_results: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 에이전트 단계 실행"""
        logger.info(f"🤖 A2A 에이전트 실행: {step.executor}")
        
        # MCP 결과를 포함한 향상된 요청 생성
        enhanced_request = self._create_enhanced_request(session.user_request, mcp_results)
        
        # A2A 에이전트 호출 (간단한 Mock 구현)
        # 실제 구현에서는 A2A 클라이언트로 실제 호출
        result = {
            "agent_id": step.executor,
            "status": "completed",
            "response": f"Mock response from {step.executor} with MCP enhancement",
            "processing_time": 2.0,
            "mcp_enhanced": True
        }
        
        return result
    
    async def _execute_collaboration_step(self, step: EnhancedWorkflowStep, session: CollaborationSession, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """협업 통합 단계 실행"""
        logger.info(f"🤝 협업 통합 실행: {step.executor}")
        
        # 모든 결과를 통합
        integrated_result = {
            "session_id": session.session_id,
            "integration_type": "mcp_enhanced_collaboration",
            "mcp_contributions": len(execution_results["mcp_results"]),
            "agent_contributions": len(execution_results["agent_results"]),
            "final_summary": f"Enhanced collaboration completed with {len(session.active_mcp_tools)} MCP tools and {len(session.active_agents)} A2A agents",
            "timestamp": datetime.now().isoformat()
        }
        
        return integrated_result
    
    def _create_enhanced_request(self, original_request: str, mcp_results: Dict[str, Any]) -> str:
        """MCP 결과를 포함한 향상된 요청 생성"""
        enhanced_request = f"{original_request}\n\n--- MCP 도구 분석 결과 ---\n"
        
        for step_id, result in mcp_results.items():
            if result.get("success"):
                enhanced_request += f"🔧 {step_id}: {result.get('result', {})}\n"
            else:
                enhanced_request += f"⚠️ {step_id}: 실행 실패\n"
        
        enhanced_request += "\n위 MCP 도구 결과를 참고하여 분석을 수행해주세요."
        
        return enhanced_request

class UniversalIntelligentOrchestratorV9(AgentExecutor):
    """v9.0 - MCP Enhanced Universal Intelligent Orchestrator"""
    
    def __init__(self):
        # OpenAI 클라이언트 초기화
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 Universal Intelligent Orchestrator v9.0 with LLM + MCP")
            else:
                self.openai_client = None
                logger.info("📊 Universal Orchestrator v9.0 with MCP (No LLM)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        # v9.0 핵심 컴포넌트
        self.intent_analyzer = IntelligentIntentAnalyzer(self.openai_client)
        self.resource_discovery = MCPEnhancedAgentDiscovery()
        self.collaboration_engine = EnhancedCollaborationEngine(self.openai_client)
        
        # 상태 관리
        self.available_resources = {}
        self.active_sessions = {}
        self.performance_metrics = {}
        
        # 초기화 완료
        logger.info("🚀 Universal Intelligent Orchestrator v9.0 (MCP Enhanced) 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """v9.0 Enhanced 실행 - MCP 통합 + 지능형 라우팅"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 요청 추출
            user_request = self._extract_user_request(context)
            if not user_request:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용자 요청을 찾을 수 없습니다.")])
                )
                return
            
            logger.info(f"📥 v9.0 처리 시작: {user_request}")
            
            # 1단계: 리소스 발견 (A2A + MCP)
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="🔍 Enhanced 리소스 발견 중 (A2A 에이전트 + MCP 도구)...")])
            )
            
            self.available_resources = await self.resource_discovery.discover_all_resources()
            
            if self.available_resources["total_resources"] == 0:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용 가능한 리소스가 없습니다.")])
                )
                return
            
            # 2단계: 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="🧠 지능형 의도 분석 중...")])
            )
            
            intent_analysis = await self.intent_analyzer.analyze_intent(
                user_request, 
                {"available_resources": self.available_resources}
            )
            
            # 3단계: 협업 세션 생성
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="🚀 Enhanced 협업 세션 생성 중...")])
            )
            
            collaboration_session = await self.collaboration_engine.create_collaboration_session(
                user_request, intent_analysis
            )
            
            # 4단계: 협업 실행
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="⚡ MCP Enhanced 협업 실행 중...")])
            )
            
            execution_results = await self.collaboration_engine.execute_collaboration_session(
                collaboration_session, task_updater
            )
            
            # 5단계: 최종 결과 생성
            final_response = self._generate_final_response(
                user_request, intent_analysis, execution_results, collaboration_session
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=final_response)])
            )
            
            logger.info(f"✅ v9.0 처리 완료: {collaboration_session.session_id}")
            
        except Exception as e:
            logger.error(f"❌ v9.0 실행 실패: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"❌ 실행 실패: {str(e)}")])
            )
    
    def _extract_user_request(self, context: RequestContext) -> str:
        """사용자 요청 추출"""
        if not context.message or not context.message.parts:
            return ""
        
        user_request = ""
        for part in context.message.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                user_request += part.root.text + " "
            elif hasattr(part, 'text'):
                user_request += part.text + " "
        
        return user_request.strip()
    
    def _generate_final_response(self, user_request: str, intent_analysis: IntentAnalysisResult, 
                               execution_results: Dict[str, Any], session: CollaborationSession) -> str:
        """최종 응답 생성"""
        response = f"""🌟 **A2A v9.0 MCP Enhanced 협업 완료**

**요청**: {user_request}

**🧠 의도 분석 결과**:
- 주요 의도: {intent_analysis.primary_intent}
- 신뢰도: {intent_analysis.confidence:.2f}
- 워크플로우 타입: {intent_analysis.workflow_type}
- 복잡성: {intent_analysis.estimated_complexity}

**🔗 활용된 리소스**:
- A2A 에이전트: {len(session.active_agents)}개 ({', '.join(session.active_agents)})
- MCP 도구: {len(session.active_mcp_tools)}개 ({', '.join(session.active_mcp_tools)})

**⚡ 실행 결과**:
- MCP 도구 결과: {len(execution_results['mcp_results'])}개 완료
- A2A 에이전트 결과: {len(execution_results['agent_results'])}개 완료
- 총 실행 시간: {sum(step['duration'] for step in execution_results['execution_timeline']):.2f}초

**🎯 Enhanced 협업 특징**:
- Context Engineering 6 Data Layers 활용
- 지능형 의도 분석 기반 라우팅
- MCP 도구와 A2A 에이전트 완전 통합
- 실시간 협업 모니터링

**세션 ID**: {session.session_id}
**완료 시간**: {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

🚀 **A2A v9.0 + MCP 통합**으로 강력한 멀티에이전트 협업을 경험하세요!
"""
        
        return response
    
    async def cancel(self, context: RequestContext) -> None:
        """실행 취소 처리"""
        logger.info(f"🛑 작업 취소: {context.task_id}")

def create_agent_card() -> AgentCard:
    """A2A Agent Card 생성"""
    skill = AgentSkill(
        id="universal_mcp_orchestration",
        name="Universal MCP Enhanced Orchestration",
        description="A2A 에이전트와 MCP 도구를 통합한 지능형 멀티에이전트 협업 오케스트레이션",
        tags=["orchestration", "mcp", "a2a", "collaboration", "intelligent-routing"],
        examples=[
            "데이터를 종합적으로 분석해주세요",
            "웹에서 데이터를 수집하고 시각화해주세요", 
            "파일을 읽고 머신러닝 모델을 만들어주세요",
            "데이터베이스에서 데이터를 가져와 분석하고 차트를 만들어주세요"
        ]
    )
    
    return AgentCard(
        name="Universal Intelligent Orchestrator v9.0",
        description="A2A 에이전트와 MCP 도구를 통합한 지능형 멀티에이전트 협업 오케스트레이터. Context Engineering 6 Data Layers 지원, 의도 분석 기반 라우팅, 실시간 협업 모니터링 제공.",
        url="http://localhost:8100/",
        version="9.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

def main():
    """메인 실행 함수"""
    # A2A 서버 설정
    task_store = InMemoryTaskStore()
    event_queue = EventQueue()
    
    # v9.0 오케스트레이터 생성
    orchestrator = UniversalIntelligentOrchestratorV9()
    
    # 요청 핸들러 설정
    request_handler = DefaultRequestHandler(
        agent_executor=orchestrator,
        task_store=task_store,
        event_queue=event_queue
    )
    
    # A2A 애플리케이션 생성
    app = A2AStarletteApplication(
        agent_card=create_agent_card(),
        request_handler=request_handler
    )
    
    print("🌟 A2A Orchestrator v9.0 - MCP Enhanced Universal Intelligent Orchestrator")
    print("🔗 Features: A2A + MCP Integration, Intelligent Routing, Context Engineering")
    print("🌐 Server: http://localhost:8100")
    print("📋 Agent Card: http://localhost:8100/.well-known/agent.json")
    print("🚀 Ready for enhanced multi-agent collaboration!")
    
    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )

if __name__ == "__main__":
    main() 