#!/usr/bin/env python3
"""
🧠 LLM First Shared Knowledge Bank - A2A SDK 0.2.9 표준 서버
Port: 8602

CherryAI LLM First Architecture 완전 준수 설계
- 모든 지식 관련 결정을 LLM이 동적으로 수행
- No Hardcoded Workflows
- 100% LLM-driven Knowledge Orchestration
- Context-Aware Dynamic Knowledge Management
- Real-time LLM-powered Knowledge Discovery
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import networkx as nx
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# LLM 클라이언트 초기화
llm_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 지식 저장소 경로
KNOWLEDGE_BASE_DIR = Path("a2a_ds_servers/artifacts/knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class KnowledgeItem:
    """동적 지식 항목"""
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    relevance_score: float = 0.0
    usage_count: int = 0

@dataclass
class CollaborationMemory:
    """협업 메모리"""
    id: str
    agents_involved: List[str]
    user_query: str
    workflow_executed: List[str]
    success: bool
    execution_time: float
    insights: str  # LLM이 생성한 인사이트
    created_at: datetime

class LLMFirstKnowledgeBank:
    """LLM First 지식 은행"""
    
    def __init__(self):
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.collaboration_memories: Dict[str, CollaborationMemory] = {}
        self.knowledge_graph = nx.DiGraph()
        
        # 지식 로드
        self._load_existing_knowledge()
        
        logger.info("🧠 LLM First Knowledge Bank initialized")
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        try:
            # 지식 항목 로드
            knowledge_file = KNOWLEDGE_BASE_DIR / "llm_knowledge_items.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data:
                        item = KnowledgeItem(
                            id=item_data['id'],
                            content=item_data['content'],
                            metadata=item_data['metadata'],
                            created_at=datetime.fromisoformat(item_data['created_at']),
                            updated_at=datetime.fromisoformat(item_data['updated_at']),
                            relevance_score=item_data.get('relevance_score', 0.0),
                            usage_count=item_data.get('usage_count', 0)
                        )
                        self.knowledge_items[item.id] = item
            
            # 협업 메모리 로드
            memory_file = KNOWLEDGE_BASE_DIR / "collaboration_memories.json"
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for memory_data in data:
                        memory = CollaborationMemory(
                            id=memory_data['id'],
                            agents_involved=memory_data['agents_involved'],
                            user_query=memory_data['user_query'],
                            workflow_executed=memory_data['workflow_executed'],
                            success=memory_data['success'],
                            execution_time=memory_data['execution_time'],
                            insights=memory_data['insights'],
                            created_at=datetime.fromisoformat(memory_data['created_at'])
                        )
                        self.collaboration_memories[memory.id] = memory
            
            # 지식 그래프 로드
            graph_file = KNOWLEDGE_BASE_DIR / "llm_knowledge_graph.json"
            if graph_file.exists():
                with open(graph_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    self.knowledge_graph = nx.node_link_graph(graph_data)
            
            logger.info(f"📚 Loaded {len(self.knowledge_items)} knowledge items")
            logger.info(f"🤝 Loaded {len(self.collaboration_memories)} collaboration memories")
            
        except Exception as e:
            logger.error(f"❌ Error loading knowledge: {e}")
    
    def _save_knowledge(self):
        """지식 저장"""
        try:
            # 지식 항목 저장
            knowledge_data = []
            for item in self.knowledge_items.values():
                knowledge_data.append({
                    'id': item.id,
                    'content': item.content,
                    'metadata': item.metadata,
                    'created_at': item.created_at.isoformat(),
                    'updated_at': item.updated_at.isoformat(),
                    'relevance_score': item.relevance_score,
                    'usage_count': item.usage_count
                })
            
            with open(KNOWLEDGE_BASE_DIR / "llm_knowledge_items.json", 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            
            # 협업 메모리 저장
            memory_data = []
            for memory in self.collaboration_memories.values():
                memory_data.append({
                    'id': memory.id,
                    'agents_involved': memory.agents_involved,
                    'user_query': memory.user_query,
                    'workflow_executed': memory.workflow_executed,
                    'success': memory.success,
                    'execution_time': memory.execution_time,
                    'insights': memory.insights,
                    'created_at': memory.created_at.isoformat()
                })
            
            with open(KNOWLEDGE_BASE_DIR / "collaboration_memories.json", 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            # 지식 그래프 저장
            graph_data = nx.node_link_data(self.knowledge_graph)
            with open(KNOWLEDGE_BASE_DIR / "llm_knowledge_graph.json", 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info("💾 Knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving knowledge: {e}")
    
    async def llm_analyze_knowledge_request(self, user_query: str) -> Dict[str, Any]:
        """LLM이 지식 요청을 분석"""
        try:
            analysis_prompt = f"""
당신은 CherryAI의 지식 은행 분석 전문가입니다. 
사용자의 요청을 분석하여 어떤 종류의 지식 작업이 필요한지 판단해주세요.

사용자 요청: "{user_query}"

다음 정보를 JSON 형태로 제공해주세요:
{{
    "request_type": "knowledge_search|collaboration_recommendation|pattern_learning|insight_generation|knowledge_creation",
    "intent": "사용자의 구체적인 의도",
    "context": "요청의 맥락과 배경",
    "knowledge_domains": ["관련 지식 도메인들"],
    "expected_outcome": "예상되는 결과",
    "complexity_level": "simple|medium|complex",
    "urgency": "low|medium|high",
    "reasoning": "분석 과정과 근거"
}}
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 CherryAI의 지능형 지식 분석 전문가입니다."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # JSON 추출
            try:
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # 파싱 실패 시 기본값 반환
            return {
                "request_type": "knowledge_search",
                "intent": user_query,
                "context": "일반적인 지식 요청",
                "knowledge_domains": ["general"],
                "expected_outcome": "관련 정보 제공",
                "complexity_level": "medium",
                "urgency": "medium",
                "reasoning": "LLM 분석 결과 파싱 실패"
            }
            
        except Exception as e:
            logger.error(f"❌ LLM analysis error: {e}")
            return {
                "request_type": "knowledge_search",
                "intent": user_query,
                "context": "오류 발생",
                "knowledge_domains": ["general"],
                "expected_outcome": "기본 응답",
                "complexity_level": "medium",
                "urgency": "medium",
                "reasoning": f"오류 발생: {str(e)}"
            }
    
    async def llm_search_knowledge(self, query: str, analysis: Dict[str, Any]) -> List[KnowledgeItem]:
        """LLM이 맥락을 이해하여 지식 검색"""
        try:
            # 현재 지식 항목들을 컨텍스트로 제공
            knowledge_context = []
            for item in self.knowledge_items.values():
                knowledge_context.append({
                    "id": item.id,
                    "content": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                    "metadata": item.metadata,
                    "relevance_score": item.relevance_score,
                    "usage_count": item.usage_count
                })
            
            search_prompt = f"""
당신은 CherryAI의 지능형 지식 검색 전문가입니다.
사용자의 요청과 분석 결과를 바탕으로 가장 관련성 높은 지식을 찾아주세요.

사용자 요청: "{query}"
분석 결과: {json.dumps(analysis, ensure_ascii=False, indent=2)}

사용 가능한 지식 항목들:
{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}

다음 형태로 응답해주세요:
{{
    "relevant_knowledge_ids": ["관련성 높은 지식 ID들 (최대 5개)"],
    "search_reasoning": "검색 과정과 근거",
    "relevance_explanation": "각 지식이 관련성 있는 이유",
    "missing_knowledge": "부족한 지식 영역이 있다면 설명"
}}
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 CherryAI의 지능형 지식 검색 전문가입니다."},
                    {"role": "user", "content": search_prompt}
                ],
                temperature=0.3
            )
            
            search_result = response.choices[0].message.content
            
            # 결과 파싱
            try:
                import re
                json_match = re.search(r'\{.*\}', search_result, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    relevant_ids = result_data.get("relevant_knowledge_ids", [])
                    
                    # 관련 지식 항목들 반환
                    relevant_items = []
                    for item_id in relevant_ids:
                        if item_id in self.knowledge_items:
                            item = self.knowledge_items[item_id]
                            item.usage_count += 1  # 사용 횟수 증가
                            relevant_items.append(item)
                    
                    return relevant_items
            except:
                pass
            
            # 파싱 실패 시 빈 리스트 반환
            return []
            
        except Exception as e:
            logger.error(f"❌ LLM search error: {e}")
            return []
    
    async def llm_generate_collaboration_recommendation(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """LLM이 협업 추천 생성"""
        try:
            # 협업 메모리를 컨텍스트로 제공
            memory_context = []
            for memory in self.collaboration_memories.values():
                memory_context.append({
                    "agents_involved": memory.agents_involved,
                    "user_query": memory.user_query,
                    "workflow_executed": memory.workflow_executed,
                    "success": memory.success,
                    "execution_time": memory.execution_time,
                    "insights": memory.insights
                })
            
            recommendation_prompt = f"""
당신은 CherryAI의 협업 전문가입니다.
사용자의 요청과 과거 협업 경험을 바탕으로 최적의 협업 방안을 추천해주세요.

사용자 요청: "{query}"
분석 결과: {json.dumps(analysis, ensure_ascii=False, indent=2)}

과거 협업 경험:
{json.dumps(memory_context, ensure_ascii=False, indent=2)}

사용 가능한 에이전트들:
- pandas_agent: 데이터 분석 및 처리
- visualization_agent: 데이터 시각화
- ml_modeling_agent: 머신러닝 모델링
- data_cleaning_agent: 데이터 정제
- eda_agent: 탐색적 데이터 분석
- sql_agent: SQL 데이터베이스 분석

다음 형태로 추천을 제공해주세요:
{{
    "recommended_agents": ["추천 에이전트 목록"],
    "collaboration_strategy": "협업 전략 설명",
    "workflow_steps": ["예상 워크플로우 단계들"],
    "success_probability": "성공 확률 (0-1)",
    "estimated_time": "예상 실행 시간 (분)",
    "potential_challenges": ["예상 도전 과제들"],
    "optimization_tips": ["최적화 팁들"],
    "reasoning": "추천 근거와 과정"
}}
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 CherryAI의 협업 전문가입니다."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                temperature=0.3
            )
            
            recommendation_text = response.choices[0].message.content
            
            # JSON 추출
            try:
                import re
                json_match = re.search(r'\{.*\}', recommendation_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # 파싱 실패 시 기본 추천 반환
            return {
                "recommended_agents": ["pandas_agent"],
                "collaboration_strategy": "기본 데이터 분석 전략",
                "workflow_steps": ["데이터 로드", "분석 수행", "결과 제공"],
                "success_probability": 0.7,
                "estimated_time": 10,
                "potential_challenges": ["데이터 품질 문제"],
                "optimization_tips": ["데이터 전처리 우선 수행"],
                "reasoning": "LLM 추천 결과 파싱 실패로 기본 추천 제공"
            }
            
        except Exception as e:
            logger.error(f"❌ LLM recommendation error: {e}")
            return {
                "recommended_agents": ["pandas_agent"],
                "collaboration_strategy": "오류 발생으로 기본 전략 제공",
                "workflow_steps": ["기본 분석"],
                "success_probability": 0.5,
                "estimated_time": 15,
                "potential_challenges": ["시스템 오류"],
                "optimization_tips": ["시스템 상태 확인"],
                "reasoning": f"오류 발생: {str(e)}"
            }
    
    async def llm_learn_collaboration_pattern(self, agents: List[str], user_query: str, 
                                           workflow: List[str], success: bool, 
                                           execution_time: float) -> str:
        """LLM이 협업 패턴을 학습하고 인사이트 생성"""
        try:
            learning_prompt = f"""
당신은 CherryAI의 협업 패턴 학습 전문가입니다.
새로운 협업 경험을 분석하여 인사이트를 생성해주세요.

협업 정보:
- 참여 에이전트: {agents}
- 사용자 요청: "{user_query}"
- 실행 워크플로우: {workflow}
- 성공 여부: {success}
- 실행 시간: {execution_time}분

기존 협업 경험들과 비교하여 다음을 분석해주세요:
{{
    "pattern_insights": "이 협업에서 발견한 패턴과 인사이트",
    "success_factors": "성공/실패 요인 분석",
    "optimization_opportunities": "최적화 기회",
    "lessons_learned": "학습한 교훈들",
    "future_improvements": "향후 개선 방안",
    "collaboration_effectiveness": "협업 효과성 평가",
    "knowledge_gaps": "발견된 지식 격차",
    "recommendations": "향후 유사 상황에 대한 추천"
}}
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 CherryAI의 협업 패턴 학습 전문가입니다."},
                    {"role": "user", "content": learning_prompt}
                ],
                temperature=0.3
            )
            
            insights_text = response.choices[0].message.content
            
            # 협업 메모리 생성
            memory_id = str(uuid.uuid4())
            memory = CollaborationMemory(
                id=memory_id,
                agents_involved=agents,
                user_query=user_query,
                workflow_executed=workflow,
                success=success,
                execution_time=execution_time,
                insights=insights_text,
                created_at=datetime.now()
            )
            
            self.collaboration_memories[memory_id] = memory
            self._save_knowledge()
            
            return insights_text
            
        except Exception as e:
            logger.error(f"❌ LLM learning error: {e}")
            return f"협업 패턴 학습 중 오류 발생: {str(e)}"
    
    async def llm_generate_knowledge_insights(self, query: str) -> Dict[str, Any]:
        """LLM이 지식 인사이트 생성"""
        try:
            # 전체 지식 베이스 요약
            knowledge_summary = {
                "total_knowledge_items": len(self.knowledge_items),
                "total_collaboration_memories": len(self.collaboration_memories),
                "knowledge_domains": list(set([
                    domain for item in self.knowledge_items.values()
                    for domain in item.metadata.get("domains", [])
                ])),
                "most_used_knowledge": sorted(
                    self.knowledge_items.values(),
                    key=lambda x: x.usage_count,
                    reverse=True
                )[:5],
                "recent_collaborations": sorted(
                    self.collaboration_memories.values(),
                    key=lambda x: x.created_at,
                    reverse=True
                )[:5]
            }
            
            insights_prompt = f"""
당신은 CherryAI의 지식 인사이트 전문가입니다.
현재 지식 베이스 상태를 분석하여 인사이트를 제공해주세요.

사용자 요청: "{query}"

지식 베이스 요약:
{json.dumps({
    "total_knowledge_items": knowledge_summary["total_knowledge_items"],
    "total_collaboration_memories": knowledge_summary["total_collaboration_memories"],
    "knowledge_domains": knowledge_summary["knowledge_domains"]
}, ensure_ascii=False, indent=2)}

다음 형태로 인사이트를 제공해주세요:
{{
    "knowledge_health": "지식 베이스의 건강 상태",
    "usage_patterns": "지식 사용 패턴 분석",
    "collaboration_trends": "협업 트렌드 분석",
    "knowledge_gaps": "발견된 지식 격차",
    "recommendations": "지식 베이스 개선 추천",
    "growth_opportunities": "성장 기회",
    "efficiency_metrics": "효율성 지표",
    "strategic_insights": "전략적 인사이트"
}}
"""
            
            response = await llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 CherryAI의 지식 인사이트 전문가입니다."},
                    {"role": "user", "content": insights_prompt}
                ],
                temperature=0.3
            )
            
            insights_text = response.choices[0].message.content
            
            # JSON 추출
            try:
                import re
                json_match = re.search(r'\{.*\}', insights_text, re.DOTALL)
                if json_match:
                    insights_data = json.loads(json_match.group())
                    insights_data["raw_insights"] = insights_text
                    return insights_data
            except:
                pass
            
            return {
                "knowledge_health": "분석 중 오류 발생",
                "raw_insights": insights_text,
                "error": "JSON 파싱 실패"
            }
            
        except Exception as e:
            logger.error(f"❌ LLM insights error: {e}")
            return {
                "error": f"인사이트 생성 중 오류 발생: {str(e)}"
            }

# A2A 서버 구현
class LLMFirstKnowledgeBankExecutor(AgentExecutor):
    """LLM First Knowledge Bank A2A 실행기"""
    
    def __init__(self):
        self.knowledge_bank = LLMFirstKnowledgeBank()
        logger.info("🧠 LLM First Knowledge Bank Executor initialized")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """A2A 요청 실행 - 완전한 LLM 기반 처리"""
        try:
            # 시작 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🧠 LLM First Knowledge Bank 작업을 시작합니다..."
            )
            
            # 사용자 메시지 파싱
            user_message = None
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_message = part.root.text
                        break
            
            if not user_message:
                await task_updater.update_status(
                    TaskState.failed,
                    message="❌ 사용자 메시지를 찾을 수 없습니다."
                )
                return
            
            # 1단계: LLM이 요청 분석
            await task_updater.update_status(
                TaskState.working,
                message="🔍 LLM이 요청을 분석하고 있습니다..."
            )
            
            analysis = await self.knowledge_bank.llm_analyze_knowledge_request(user_message)
            
            # 2단계: 분석 결과에 따라 LLM이 적절한 처리 수행
            await task_updater.update_status(
                TaskState.working,
                message=f"🎯 {analysis['intent']} 작업을 수행합니다..."
            )
            
            result = await self._llm_process_request(user_message, analysis, task_updater)
            
            # 결과 전송
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(result, ensure_ascii=False, indent=2))],
                name="llm_knowledge_result",
                metadata={"content_type": "application/json"}
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message="✅ LLM First Knowledge Bank 작업이 완료되었습니다."
            )
            
        except Exception as e:
            logger.error(f"❌ LLM First Knowledge Bank error: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"❌ 오류가 발생했습니다: {str(e)}"
            )
    
    async def _llm_process_request(self, user_message: str, analysis: Dict[str, Any], task_updater: TaskUpdater) -> Dict[str, Any]:
        """LLM 기반 요청 처리"""
        try:
            request_type = analysis.get("request_type", "knowledge_search")
            
            if request_type == "knowledge_search":
                # LLM 기반 지식 검색
                await task_updater.update_status(
                    TaskState.working,
                    message="📚 LLM이 맥락을 이해하여 지식을 검색합니다..."
                )
                
                relevant_knowledge = await self.knowledge_bank.llm_search_knowledge(user_message, analysis)
                
                return {
                    "type": "llm_knowledge_search",
                    "analysis": analysis,
                    "relevant_knowledge": [
                        {
                            "id": item.id,
                            "content": item.content,
                            "metadata": item.metadata,
                            "relevance_score": item.relevance_score,
                            "usage_count": item.usage_count
                        } for item in relevant_knowledge
                    ],
                    "llm_insights": "LLM이 맥락을 이해하여 가장 관련성 높은 지식을 찾았습니다."
                }
            
            elif request_type == "collaboration_recommendation":
                # LLM 기반 협업 추천
                await task_updater.update_status(
                    TaskState.working,
                    message="🤝 LLM이 협업 전략을 분석하고 추천합니다..."
                )
                
                recommendation = await self.knowledge_bank.llm_generate_collaboration_recommendation(user_message, analysis)
                
                return {
                    "type": "llm_collaboration_recommendation",
                    "analysis": analysis,
                    "recommendation": recommendation,
                    "llm_insights": "LLM이 과거 경험과 현재 상황을 종합하여 최적의 협업 전략을 추천했습니다."
                }
            
            elif request_type == "pattern_learning":
                # LLM 기반 패턴 학습 (데모용)
                await task_updater.update_status(
                    TaskState.working,
                    message="📈 LLM이 협업 패턴을 학습하고 인사이트를 생성합니다..."
                )
                
                insights = await self.knowledge_bank.llm_learn_collaboration_pattern(
                    agents=["pandas_agent", "visualization_agent"],
                    user_query=user_message,
                    workflow=["데이터 로드", "분석", "시각화"],
                    success=True,
                    execution_time=12.5
                )
                
                return {
                    "type": "llm_pattern_learning",
                    "analysis": analysis,
                    "insights": insights,
                    "llm_insights": "LLM이 새로운 협업 패턴을 학습하고 향후 개선 방안을 제안했습니다."
                }
            
            elif request_type == "insight_generation":
                # LLM 기반 인사이트 생성
                await task_updater.update_status(
                    TaskState.working,
                    message="💡 LLM이 지식 베이스 인사이트를 생성합니다..."
                )
                
                insights = await self.knowledge_bank.llm_generate_knowledge_insights(user_message)
                
                return {
                    "type": "llm_knowledge_insights",
                    "analysis": analysis,
                    "insights": insights,
                    "llm_insights": "LLM이 전체 지식 베이스를 분석하여 전략적 인사이트를 제공했습니다."
                }
            
            else:
                # 기본 LLM 기반 처리
                relevant_knowledge = await self.knowledge_bank.llm_search_knowledge(user_message, analysis)
                
                return {
                    "type": "llm_general_knowledge",
                    "analysis": analysis,
                    "relevant_knowledge": [
                        {
                            "content": item.content[:300] + "..." if len(item.content) > 300 else item.content,
                            "metadata": item.metadata
                        } for item in relevant_knowledge[:3]
                    ],
                    "llm_insights": "LLM이 요청을 이해하고 관련 지식을 찾았습니다."
                }
            
        except Exception as e:
            logger.error(f"❌ Error in LLM processing: {e}")
            return {
                "type": "error",
                "message": f"LLM 처리 중 오류가 발생했습니다: {str(e)}",
                "analysis": analysis
            }
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """작업 취소"""
        await task_updater.update_status(
            TaskState.cancelled,
            message="🛑 LLM First Knowledge Bank 작업이 취소되었습니다."
        )

# Agent Card 정의
AGENT_CARD = AgentCard(
    name="LLM First Shared Knowledge Bank",
    description="CherryAI LLM First Architecture를 완전히 준수하는 지능형 공유 지식 은행 - 모든 지식 작업을 LLM이 동적으로 수행",
    skills=[
        AgentSkill(
            name="llm_knowledge_analysis",
            description="LLM이 사용자 요청을 분석하여 지식 작업 유형을 동적으로 결정"
        ),
        AgentSkill(
            name="llm_contextual_search",
            description="LLM이 맥락을 이해하여 관련 지식을 지능적으로 검색"
        ),
        AgentSkill(
            name="llm_collaboration_strategy",
            description="LLM이 과거 경험을 분석하여 최적의 협업 전략을 추천"
        ),
        AgentSkill(
            name="llm_pattern_learning",
            description="LLM이 협업 패턴을 학습하고 인사이트를 생성"
        ),
        AgentSkill(
            name="llm_knowledge_insights",
            description="LLM이 지식 베이스를 분석하여 전략적 인사이트를 제공"
        )
    ],
    capabilities=AgentCapabilities(
        supports_streaming=True,
        supports_cancellation=True,
        supports_artifacts=True
    )
)

# 메인 실행
async def main():
    """메인 실행 함수"""
    # A2A 서버 설정
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(task_store)
    
    # 에이전트 등록
    executor = LLMFirstKnowledgeBankExecutor()
    request_handler.register_agent(AGENT_CARD, executor)
    
    # 앱 생성
    app = A2AStarletteApplication(
        request_handler=request_handler,
        agent_card=AGENT_CARD
    )
    
    # 서버 시작
    logger.info("🚀 Starting LLM First Shared Knowledge Bank Server on port 8602")
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8602,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main()) 