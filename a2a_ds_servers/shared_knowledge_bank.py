#!/usr/bin/env python3
"""
🧠 Shared Knowledge Bank - A2A SDK 0.2.9 표준 서버
Port: 8602

Context Engineering MEMORY 레이어의 핵심 구현
- 에이전트 간 공유 지식 저장 및 검색
- 협업 패턴 학습 및 최적화
- 임베딩 기반 고급 검색 시스템
- 지식 그래프 구축 및 관리
- 사용자별 에이전트 조합 선호도 학습
"""

import asyncio
import json
import logging
import os
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import uvicorn

# 간단한 텍스트 임베딩 구현 (sentence-transformers 대신)
import hashlib
import re
from collections import Counter
import math

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

# 데이터 저장 경로 설정
KNOWLEDGE_BASE_DIR = Path("a2a_ds_servers/artifacts/knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_DIR = KNOWLEDGE_BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = KNOWLEDGE_BASE_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

PATTERNS_DIR = KNOWLEDGE_BASE_DIR / "patterns"
PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

class SimpleTextEmbedding:
    """간단한 텍스트 임베딩 구현"""
    
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.vocab = {}
        self.vocab_size = 0
    
    def _preprocess_text(self, text):
        """텍스트 전처리"""
        # 한국어와 영어 모두 처리
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
        words = text.split()
        return words
    
    def _get_word_vector(self, word):
        """단어 벡터 생성"""
        if word not in self.vocab:
            # 단어 해시 기반 벡터 생성
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            np.random.seed(hash_value % (2**32))
            vector = np.random.randn(self.vector_size)
            self.vocab[word] = vector
            self.vocab_size += 1
        return self.vocab[word]
    
    def encode(self, text):
        """텍스트를 벡터로 변환"""
        if isinstance(text, str):
            words = self._preprocess_text(text)
            if not words:
                return np.zeros(self.vector_size)
            
            # 단어 벡터들의 평균 계산
            word_vectors = [self._get_word_vector(word) for word in words]
            return np.mean(word_vectors, axis=0)
        elif isinstance(text, list):
            return [self.encode(t) for t in text]
        else:
            return np.zeros(self.vector_size)

def cosine_similarity_simple(vec1, vec2):
    """코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

class KnowledgeType(Enum):
    """지식 타입 정의"""
    AGENT_EXPERTISE = "agent_expertise"
    COLLABORATION_PATTERN = "collaboration_pattern"
    USER_PREFERENCE = "user_preference"
    MESSAGE_OPTIMIZATION = "message_optimization"
    CROSS_AGENT_INSIGHT = "cross_agent_insight"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    WORKFLOW_TEMPLATE = "workflow_template"
    ERROR_SOLUTION = "error_solution"

@dataclass
class KnowledgeEntry:
    """지식 항목 정의"""
    id: str
    type: KnowledgeType
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    related_entries: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    usage_count: int = 0
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.related_entries is None:
            self.related_entries = []

@dataclass
class CollaborationPattern:
    """협업 패턴 정의"""
    id: str
    agents: List[str]
    user_query_type: str
    success_rate: float
    average_execution_time: float
    typical_workflow: List[str]
    common_errors: List[str]
    optimization_tips: List[str]
    usage_frequency: int
    created_at: datetime
    updated_at: datetime

@dataclass
class UserPreference:
    """사용자 선호도 정의"""
    user_id: str
    preferred_agents: List[str]
    preferred_workflows: List[str]
    communication_style: str
    complexity_preference: str
    favorite_analysis_types: List[str]
    avoided_agents: List[str]
    created_at: datetime
    updated_at: datetime

class SharedKnowledgeBank:
    """공유 지식 은행 핵심 클래스"""
    
    def __init__(self):
        self.knowledge_entries: Dict[str, KnowledgeEntry] = {}
        self.collaboration_patterns: Dict[str, CollaborationPattern] = {}
        self.user_preferences: Dict[str, UserPreference] = {}
        self.knowledge_graph = nx.DiGraph()
        
        # 임베딩 모델 초기화
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 기존 지식 로드
        self._load_existing_knowledge()
        
        logger.info("🧠 Shared Knowledge Bank initialized")
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        try:
            # 지식 항목 로드
            knowledge_file = KNOWLEDGE_BASE_DIR / "knowledge_entries.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = KnowledgeEntry(
                            id=entry_data['id'],
                            type=KnowledgeType(entry_data['type']),
                            title=entry_data['title'],
                            content=entry_data['content'],
                            metadata=entry_data['metadata'],
                            embedding=entry_data.get('embedding'),
                            related_entries=entry_data.get('related_entries', []),
                            created_at=datetime.fromisoformat(entry_data['created_at']),
                            updated_at=datetime.fromisoformat(entry_data['updated_at']),
                            usage_count=entry_data.get('usage_count', 0),
                            relevance_score=entry_data.get('relevance_score', 0.0)
                        )
                        self.knowledge_entries[entry.id] = entry
            
            # 협업 패턴 로드
            patterns_file = KNOWLEDGE_BASE_DIR / "collaboration_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = CollaborationPattern(
                            id=pattern_data['id'],
                            agents=pattern_data['agents'],
                            user_query_type=pattern_data['user_query_type'],
                            success_rate=pattern_data['success_rate'],
                            average_execution_time=pattern_data['average_execution_time'],
                            typical_workflow=pattern_data['typical_workflow'],
                            common_errors=pattern_data['common_errors'],
                            optimization_tips=pattern_data['optimization_tips'],
                            usage_frequency=pattern_data['usage_frequency'],
                            created_at=datetime.fromisoformat(pattern_data['created_at']),
                            updated_at=datetime.fromisoformat(pattern_data['updated_at'])
                        )
                        self.collaboration_patterns[pattern.id] = pattern
            
            # 사용자 선호도 로드
            prefs_file = KNOWLEDGE_BASE_DIR / "user_preferences.json"
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pref_data in data:
                        pref = UserPreference(
                            user_id=pref_data['user_id'],
                            preferred_agents=pref_data['preferred_agents'],
                            preferred_workflows=pref_data['preferred_workflows'],
                            communication_style=pref_data['communication_style'],
                            complexity_preference=pref_data['complexity_preference'],
                            favorite_analysis_types=pref_data['favorite_analysis_types'],
                            avoided_agents=pref_data['avoided_agents'],
                            created_at=datetime.fromisoformat(pref_data['created_at']),
                            updated_at=datetime.fromisoformat(pref_data['updated_at'])
                        )
                        self.user_preferences[pref.user_id] = pref
            
            # 지식 그래프 로드
            graph_file = KNOWLEDGE_BASE_DIR / "knowledge_graph.gpickle"
            if graph_file.exists():
                self.knowledge_graph = nx.read_gpickle(graph_file)
            
            logger.info(f"📚 Loaded {len(self.knowledge_entries)} knowledge entries")
            logger.info(f"🤝 Loaded {len(self.collaboration_patterns)} collaboration patterns")
            logger.info(f"👤 Loaded {len(self.user_preferences)} user preferences")
            
        except Exception as e:
            logger.error(f"❌ Error loading existing knowledge: {e}")
    
    def _save_knowledge(self):
        """지식 저장"""
        try:
            # 지식 항목 저장
            knowledge_data = []
            for entry in self.knowledge_entries.values():
                knowledge_data.append({
                    'id': entry.id,
                    'type': entry.type.value,
                    'title': entry.title,
                    'content': entry.content,
                    'metadata': entry.metadata,
                    'embedding': entry.embedding,
                    'related_entries': entry.related_entries,
                    'created_at': entry.created_at.isoformat(),
                    'updated_at': entry.updated_at.isoformat(),
                    'usage_count': entry.usage_count,
                    'relevance_score': entry.relevance_score
                })
            
            with open(KNOWLEDGE_BASE_DIR / "knowledge_entries.json", 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            
            # 협업 패턴 저장
            patterns_data = []
            for pattern in self.collaboration_patterns.values():
                patterns_data.append({
                    'id': pattern.id,
                    'agents': pattern.agents,
                    'user_query_type': pattern.user_query_type,
                    'success_rate': pattern.success_rate,
                    'average_execution_time': pattern.average_execution_time,
                    'typical_workflow': pattern.typical_workflow,
                    'common_errors': pattern.common_errors,
                    'optimization_tips': pattern.optimization_tips,
                    'usage_frequency': pattern.usage_frequency,
                    'created_at': pattern.created_at.isoformat(),
                    'updated_at': pattern.updated_at.isoformat()
                })
            
            with open(KNOWLEDGE_BASE_DIR / "collaboration_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, indent=2, ensure_ascii=False)
            
            # 사용자 선호도 저장
            prefs_data = []
            for pref in self.user_preferences.values():
                prefs_data.append({
                    'user_id': pref.user_id,
                    'preferred_agents': pref.preferred_agents,
                    'preferred_workflows': pref.preferred_workflows,
                    'communication_style': pref.communication_style,
                    'complexity_preference': pref.complexity_preference,
                    'favorite_analysis_types': pref.favorite_analysis_types,
                    'avoided_agents': pref.avoided_agents,
                    'created_at': pref.created_at.isoformat(),
                    'updated_at': pref.updated_at.isoformat()
                })
            
            with open(KNOWLEDGE_BASE_DIR / "user_preferences.json", 'w', encoding='utf-8') as f:
                json.dump(prefs_data, f, indent=2, ensure_ascii=False)
            
            # 지식 그래프 저장
            nx.write_gpickle(self.knowledge_graph, KNOWLEDGE_BASE_DIR / "knowledge_graph.gpickle")
            
            logger.info("💾 Knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving knowledge: {e}")
    
    async def add_knowledge_entry(self, entry: KnowledgeEntry) -> str:
        """지식 항목 추가"""
        try:
            # 임베딩 생성
            if entry.embedding is None:
                text_to_embed = f"{entry.title} {entry.content}"
                entry.embedding = self.embedding_model.encode(text_to_embed).tolist()
            
            # 관련 항목 찾기
            related_entries = await self._find_related_entries(entry)
            entry.related_entries = related_entries
            
            # 지식 저장
            self.knowledge_entries[entry.id] = entry
            
            # 지식 그래프 업데이트
            self.knowledge_graph.add_node(entry.id, **asdict(entry))
            for related_id in related_entries:
                if related_id in self.knowledge_entries:
                    self.knowledge_graph.add_edge(entry.id, related_id)
            
            # 저장
            self._save_knowledge()
            
            logger.info(f"📝 Added knowledge entry: {entry.title}")
            return entry.id
            
        except Exception as e:
            logger.error(f"❌ Error adding knowledge entry: {e}")
            raise
    
    async def _find_related_entries(self, entry: KnowledgeEntry, threshold: float = 0.7) -> List[str]:
        """관련 항목 찾기"""
        related_entries = []
        
        if entry.embedding is None:
            return related_entries
        
        try:
            for existing_id, existing_entry in self.knowledge_entries.items():
                if existing_entry.embedding is None:
                    continue
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    [entry.embedding], 
                    [existing_entry.embedding]
                )[0][0]
                
                if similarity >= threshold:
                    related_entries.append(existing_id)
            
            return related_entries[:10]  # 상위 10개만 반환
            
        except Exception as e:
            logger.error(f"❌ Error finding related entries: {e}")
            return []
    
    async def search_knowledge(self, query: str, knowledge_type: Optional[KnowledgeType] = None, limit: int = 10) -> List[KnowledgeEntry]:
        """지식 검색"""
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 유사도 계산
            results = []
            for entry in self.knowledge_entries.values():
                if knowledge_type and entry.type != knowledge_type:
                    continue
                
                if entry.embedding is None:
                    continue
                
                similarity = cosine_similarity(
                    [query_embedding], 
                    [entry.embedding]
                )[0][0]
                
                results.append((entry, similarity))
            
            # 유사도 기준 정렬
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 결과 반환
            return [entry for entry, _ in results[:limit]]
            
        except Exception as e:
            logger.error(f"❌ Error searching knowledge: {e}")
            return []
    
    async def learn_collaboration_pattern(self, agents: List[str], user_query: str, success: bool, execution_time: float, workflow: List[str], errors: List[str] = None) -> str:
        """협업 패턴 학습"""
        try:
            # 패턴 ID 생성
            pattern_key = f"{'-'.join(sorted(agents))}_{hash(user_query) % 10000}"
            
            if pattern_key in self.collaboration_patterns:
                # 기존 패턴 업데이트
                pattern = self.collaboration_patterns[pattern_key]
                pattern.usage_frequency += 1
                
                # 성공률 업데이트
                total_attempts = pattern.usage_frequency
                current_successes = pattern.success_rate * (total_attempts - 1)
                if success:
                    current_successes += 1
                pattern.success_rate = current_successes / total_attempts
                
                # 평균 실행 시간 업데이트
                pattern.average_execution_time = (
                    pattern.average_execution_time * (total_attempts - 1) + execution_time
                ) / total_attempts
                
                # 워크플로우 업데이트
                if workflow not in pattern.typical_workflow:
                    pattern.typical_workflow.append(workflow)
                
                # 오류 패턴 업데이트
                if errors:
                    pattern.common_errors.extend(errors)
                    pattern.common_errors = list(set(pattern.common_errors))
                
                pattern.updated_at = datetime.now()
                
            else:
                # 새 패턴 생성
                pattern = CollaborationPattern(
                    id=pattern_key,
                    agents=agents,
                    user_query_type=user_query,
                    success_rate=1.0 if success else 0.0,
                    average_execution_time=execution_time,
                    typical_workflow=[workflow],
                    common_errors=errors or [],
                    optimization_tips=[],
                    usage_frequency=1,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self.collaboration_patterns[pattern_key] = pattern
            
            # 최적화 팁 생성
            await self._generate_optimization_tips(pattern)
            
            # 저장
            self._save_knowledge()
            
            logger.info(f"🤝 Learned collaboration pattern: {pattern_key}")
            return pattern_key
            
        except Exception as e:
            logger.error(f"❌ Error learning collaboration pattern: {e}")
            raise
    
    async def _generate_optimization_tips(self, pattern: CollaborationPattern):
        """최적화 팁 생성"""
        tips = []
        
        # 성공률 기반 팁
        if pattern.success_rate < 0.5:
            tips.append("낮은 성공률을 개선하기 위해 다른 에이전트 조합을 고려하세요.")
        
        # 실행 시간 기반 팁
        if pattern.average_execution_time > 30:
            tips.append("실행 시간이 길므로 병렬 처리를 고려하세요.")
        
        # 오류 패턴 기반 팁
        if pattern.common_errors:
            tips.append(f"자주 발생하는 오류: {', '.join(pattern.common_errors[:3])}")
        
        pattern.optimization_tips = tips
    
    async def get_collaboration_recommendations(self, user_query: str, available_agents: List[str]) -> List[Dict[str, Any]]:
        """협업 추천"""
        try:
            recommendations = []
            
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode(user_query).tolist()
            
            # 유사한 패턴 찾기
            for pattern in self.collaboration_patterns.values():
                # 사용 가능한 에이전트와 겹치는지 확인
                if not any(agent in available_agents for agent in pattern.agents):
                    continue
                
                # 유사도 계산
                pattern_text = f"{pattern.user_query_type} {' '.join(pattern.agents)}"
                pattern_embedding = self.embedding_model.encode(pattern_text).tolist()
                
                similarity = cosine_similarity(
                    [query_embedding], 
                    [pattern_embedding]
                )[0][0]
                
                if similarity > 0.5:
                    recommendations.append({
                        'pattern_id': pattern.id,
                        'agents': pattern.agents,
                        'success_rate': pattern.success_rate,
                        'average_time': pattern.average_execution_time,
                        'similarity': similarity,
                        'optimization_tips': pattern.optimization_tips
                    })
            
            # 성공률과 유사도 기준 정렬
            recommendations.sort(
                key=lambda x: (x['success_rate'], x['similarity']), 
                reverse=True
            )
            
            return recommendations[:5]  # 상위 5개 추천
            
        except Exception as e:
            logger.error(f"❌ Error getting collaboration recommendations: {e}")
            return []
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """사용자 선호도 업데이트"""
        try:
            if user_id in self.user_preferences:
                pref = self.user_preferences[user_id]
                pref.updated_at = datetime.now()
            else:
                pref = UserPreference(
                    user_id=user_id,
                    preferred_agents=[],
                    preferred_workflows=[],
                    communication_style="standard",
                    complexity_preference="medium",
                    favorite_analysis_types=[],
                    avoided_agents=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.user_preferences[user_id] = pref
            
            # 선호도 업데이트
            for key, value in preferences.items():
                if hasattr(pref, key):
                    setattr(pref, key, value)
            
            self._save_knowledge()
            
            logger.info(f"👤 Updated user preferences for: {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Error updating user preferences: {e}")
            raise
    
    async def get_knowledge_graph_insights(self, node_id: str) -> Dict[str, Any]:
        """지식 그래프 인사이트 제공"""
        try:
            if node_id not in self.knowledge_graph:
                return {}
            
            insights = {
                'node_info': dict(self.knowledge_graph.nodes[node_id]),
                'connected_nodes': list(self.knowledge_graph.neighbors(node_id)),
                'centrality': nx.betweenness_centrality(self.knowledge_graph).get(node_id, 0),
                'degree': self.knowledge_graph.degree(node_id),
                'clustering': nx.clustering(self.knowledge_graph.to_undirected()).get(node_id, 0)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error getting knowledge graph insights: {e}")
            return {}

# A2A 서버 구현
class SharedKnowledgeBankExecutor(AgentExecutor):
    """Shared Knowledge Bank A2A 실행기"""
    
    def __init__(self):
        self.knowledge_bank = SharedKnowledgeBank()
        logger.info("🧠 SharedKnowledgeBankExecutor initialized")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """A2A 요청 실행"""
        try:
            # 시작 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🧠 Shared Knowledge Bank 작업을 시작합니다..."
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
            
            # 메시지 분석 및 적절한 기능 실행
            result = await self._process_request(user_message, task_updater)
            
            # 결과 전송
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(result, ensure_ascii=False, indent=2))],
                name="knowledge_bank_result",
                metadata={"content_type": "application/json"}
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message="✅ Shared Knowledge Bank 작업이 완료되었습니다."
            )
            
        except Exception as e:
            logger.error(f"❌ SharedKnowledgeBankExecutor error: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"❌ 오류가 발생했습니다: {str(e)}"
            )
    
    async def _process_request(self, user_message: str, task_updater: TaskUpdater) -> Dict[str, Any]:
        """요청 처리"""
        try:
            # 요청 타입 분석
            request_lower = user_message.lower()
            
            if "검색" in request_lower or "search" in request_lower:
                # 지식 검색
                await task_updater.update_status(
                    TaskState.working,
                    message="🔍 지식 검색을 수행합니다..."
                )
                
                results = await self.knowledge_bank.search_knowledge(user_message, limit=5)
                
                return {
                    "type": "knowledge_search",
                    "query": user_message,
                    "results": [
                        {
                            "id": entry.id,
                            "title": entry.title,
                            "content": entry.content,
                            "type": entry.type.value,
                            "usage_count": entry.usage_count,
                            "relevance_score": entry.relevance_score
                        } for entry in results
                    ],
                    "total_results": len(results)
                }
            
            elif "추천" in request_lower or "recommend" in request_lower:
                # 협업 추천
                await task_updater.update_status(
                    TaskState.working,
                    message="🤝 협업 패턴 추천을 생성합니다..."
                )
                
                available_agents = [
                    "pandas_agent", "data_cleaning_agent", "visualization_agent",
                    "eda_agent", "ml_modeling_agent", "sql_agent"
                ]
                
                recommendations = await self.knowledge_bank.get_collaboration_recommendations(
                    user_message, available_agents
                )
                
                return {
                    "type": "collaboration_recommendations",
                    "query": user_message,
                    "recommendations": recommendations
                }
            
            elif "학습" in request_lower or "learn" in request_lower:
                # 패턴 학습 (데모용)
                await task_updater.update_status(
                    TaskState.working,
                    message="📚 협업 패턴을 학습합니다..."
                )
                
                # 데모 데이터로 학습
                pattern_id = await self.knowledge_bank.learn_collaboration_pattern(
                    agents=["pandas_agent", "visualization_agent"],
                    user_query=user_message,
                    success=True,
                    execution_time=15.5,
                    workflow=["데이터 로드", "분석", "시각화"],
                    errors=[]
                )
                
                return {
                    "type": "pattern_learning",
                    "pattern_id": pattern_id,
                    "message": "협업 패턴이 성공적으로 학습되었습니다."
                }
            
            elif "통계" in request_lower or "stats" in request_lower:
                # 지식 은행 통계
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 지식 은행 통계를 생성합니다..."
                )
                
                return {
                    "type": "knowledge_bank_stats",
                    "statistics": {
                        "total_knowledge_entries": len(self.knowledge_bank.knowledge_entries),
                        "collaboration_patterns": len(self.knowledge_bank.collaboration_patterns),
                        "user_preferences": len(self.knowledge_bank.user_preferences),
                        "knowledge_graph_nodes": self.knowledge_bank.knowledge_graph.number_of_nodes(),
                        "knowledge_graph_edges": self.knowledge_bank.knowledge_graph.number_of_edges(),
                        "knowledge_types": {
                            ktype.value: len([
                                e for e in self.knowledge_bank.knowledge_entries.values() 
                                if e.type == ktype
                            ])
                            for ktype in KnowledgeType
                        }
                    }
                }
            
            else:
                # 기본 지식 검색
                results = await self.knowledge_bank.search_knowledge(user_message, limit=3)
                
                return {
                    "type": "general_knowledge_query",
                    "query": user_message,
                    "results": [
                        {
                            "title": entry.title,
                            "content": entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
                            "type": entry.type.value
                        } for entry in results
                    ],
                    "suggestion": "더 구체적인 검색을 위해 '검색:', '추천:', '학습:', '통계:' 중 하나를 앞에 붙여주세요."
                }
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            return {
                "type": "error",
                "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """작업 취소"""
        await task_updater.update_status(
            TaskState.cancelled,
            message="🛑 Shared Knowledge Bank 작업이 취소되었습니다."
        )

# Agent Card 정의
AGENT_CARD = AgentCard(
    name="Shared Knowledge Bank",
    description="A2A 에이전트 간 공유 지식 은행 - 협업 패턴 학습, 지식 검색, 사용자 선호도 관리",
    skills=[
        AgentSkill(
            name="knowledge_search",
            description="임베딩 기반 지식 검색 및 관련 정보 제공"
        ),
        AgentSkill(
            name="collaboration_learning",
            description="에이전트 간 협업 패턴 학습 및 최적화 제안"
        ),
        AgentSkill(
            name="preference_management",
            description="사용자별 선호도 학습 및 개인화 추천"
        ),
        AgentSkill(
            name="knowledge_graph_analysis",
            description="지식 그래프 분석 및 인사이트 제공"
        ),
        AgentSkill(
            name="cross_agent_insights",
            description="크로스 에이전트 인사이트 축적 및 활용"
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
    executor = SharedKnowledgeBankExecutor()
    request_handler.register_agent(AGENT_CARD, executor)
    
    # 앱 생성
    app = A2AStarletteApplication(
        request_handler=request_handler,
        agent_card=AGENT_CARD
    )
    
    # 서버 시작
    logger.info("🚀 Starting Shared Knowledge Bank Server on port 8602")
    
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