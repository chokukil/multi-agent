"""
Shared Knowledge Bank - 고급 기능 구현
지식 그래프, 임베딩 검색, 자동 업데이트 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import sqlite3
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeNode:
    """지식 그래프 노드"""
    
    def __init__(self, node_id: str, content: str, node_type: str, metadata: Dict[str, Any] = None):
        self.node_id = node_id
        self.content = content
        self.node_type = node_type
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.access_count = 0
        self.relevance_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content': self.content,
            'node_type': self.node_type,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'access_count': self.access_count,
            'relevance_score': self.relevance_score
        }

class KnowledgeGraph:
    """지식 그래프 관리자"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.embeddings = {}
        self.last_update = datetime.now()
    
    def add_node(self, node: KnowledgeNode) -> None:
        """노드 추가"""
        self.graph.add_node(node.node_id, **node.to_dict())
        self.nodes[node.node_id] = node
        self.last_update = datetime.now()
    
    def add_edge(self, source_id: str, target_id: str, relationship: str, weight: float = 1.0) -> None:
        """엣지 추가"""
        if source_id in self.nodes and target_id in self.nodes:
            self.graph.add_edge(source_id, target_id, relationship=relationship, weight=weight)
            self.last_update = datetime.now()
    
    def find_related_nodes(self, node_id: str, max_depth: int = 2) -> List[str]:
        """관련 노드 찾기"""
        if node_id not in self.graph:
            return []
        
        related = []
        for depth in range(1, max_depth + 1):
            for neighbor in nx.single_source_shortest_path(self.graph, node_id, cutoff=depth):
                if neighbor != node_id and neighbor not in related:
                    related.append(neighbor)
        
        return related
    
    def get_node_centrality(self) -> Dict[str, float]:
        """노드 중심성 계산"""
        try:
            return nx.betweenness_centrality(self.graph)
        except:
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """그래프를 딕셔너리로 변환"""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'relationship': self.graph[edge[0]][edge[1]].get('relationship', ''),
                    'weight': self.graph[edge[0]][edge[1]].get('weight', 1.0)
                }
                for edge in self.graph.edges()
            ],
            'last_update': self.last_update.isoformat()
        }

class EmbeddingSearchEngine:
    """임베딩 기반 검색 엔진"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.document_ids = []
        self.embeddings = None
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def add_document(self, doc_id: str, content: str) -> None:
        """문서 추가"""
        if self.model is None:
            return
        
        self.documents.append(content)
        self.document_ids.append(doc_id)
        
        # 임베딩 계산
        if self.embeddings is None:
            self.embeddings = self.model.encode([content])
        else:
            new_embedding = self.model.encode([content])
            self.embeddings = np.vstack([self.embeddings, new_embedding])
        
        # FAISS 인덱스 업데이트
        self._update_faiss_index()
    
    def _update_faiss_index(self):
        """FAISS 인덱스 업데이트"""
        if self.embeddings is not None and len(self.embeddings) > 0:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # 정규화
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.index.add(normalized_embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """유사도 검색"""
        if self.model is None or self.index is None:
            return []
        
        try:
            # 쿼리 임베딩
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # 검색 수행
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_ids):
                    results.append((self.document_ids[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """유사 문서 찾기"""
        if doc_id not in self.document_ids:
            return []
        
        idx = self.document_ids.index(doc_id)
        content = self.documents[idx]
        
        return self.search(content, top_k + 1)[1:]  # 자기 자신 제외

class SharedKnowledgeBank:
    """공유 지식 은행 - 고급 기능"""
    
    def __init__(self, db_path: str = "knowledge_bank.db"):
        self.db_path = db_path
        self.knowledge_graph = KnowledgeGraph()
        self.embedding_engine = EmbeddingSearchEngine()
        self.session_key = "shared_knowledge_bank"
        
        self.init_database()
        self.load_knowledge_graph()
        
        # 세션 상태 초기화
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                'last_update': datetime.now(),
                'auto_update_enabled': True,
                'search_history': [],
                'knowledge_stats': {}
            }
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 지식 노드 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    relevance_score REAL DEFAULT 0.0
                )
            ''')
            
            # 관계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    source_id TEXT,
                    target_id TEXT,
                    relationship TEXT,
                    weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, relationship)
                )
            ''')
            
            # 검색 히스토리 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    results TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_knowledge(self, content: str, node_type: str, metadata: Dict[str, Any] = None) -> str:
        """지식 추가"""
        # 노드 ID 생성
        node_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # 중복 체크
        if node_id in self.knowledge_graph.nodes:
            return node_id
        
        # 노드 생성
        node = KnowledgeNode(node_id, content, node_type, metadata)
        self.knowledge_graph.add_node(node)
        
        # 임베딩 엔진에 추가
        self.embedding_engine.add_document(node_id, content)
        
        # 데이터베이스 저장
        self.save_node_to_db(node)
        
        # 자동 관계 생성
        self.auto_create_relationships(node_id)
        
        return node_id
    
    def save_node_to_db(self, node: KnowledgeNode):
        """노드를 데이터베이스에 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_nodes 
                (node_id, content, node_type, metadata, created_at, updated_at, access_count, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.node_id,
                node.content,
                node.node_type,
                json.dumps(node.metadata),
                node.created_at,
                node.updated_at,
                node.access_count,
                node.relevance_score
            ))
            conn.commit()
    
    def auto_create_relationships(self, node_id: str):
        """자동 관계 생성"""
        if node_id not in self.knowledge_graph.nodes:
            return
        
        # 유사한 노드 찾기
        similar_nodes = self.embedding_engine.get_similar_documents(node_id, top_k=3)
        
        for similar_id, similarity in similar_nodes:
            if similarity > 0.7:  # 임계값
                self.knowledge_graph.add_edge(node_id, similar_id, "similar", similarity)
                self.save_relationship_to_db(node_id, similar_id, "similar", similarity)
    
    def save_relationship_to_db(self, source_id: str, target_id: str, relationship: str, weight: float):
        """관계를 데이터베이스에 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_relationships 
                (source_id, target_id, relationship, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (source_id, target_id, relationship, weight, datetime.now()))
            conn.commit()
    
    def load_knowledge_graph(self):
        """지식 그래프 로드"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 노드 로드
                cursor.execute('SELECT * FROM knowledge_nodes')
                for row in cursor.fetchall():
                    node = KnowledgeNode(
                        node_id=row[0],
                        content=row[1],
                        node_type=row[2],
                        metadata=json.loads(row[3] or '{}')
                    )
                    node.created_at = datetime.fromisoformat(row[4])
                    node.updated_at = datetime.fromisoformat(row[5])
                    node.access_count = row[6]
                    node.relevance_score = row[7]
                    
                    self.knowledge_graph.add_node(node)
                    self.embedding_engine.add_document(node.node_id, node.content)
                
                # 관계 로드
                cursor.execute('SELECT * FROM knowledge_relationships')
                for row in cursor.fetchall():
                    self.knowledge_graph.add_edge(row[0], row[1], row[2], row[3])
                
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
    
    def search_knowledge(self, query: str, search_type: str = "embedding") -> List[Dict[str, Any]]:
        """지식 검색"""
        results = []
        
        if search_type == "embedding":
            # 임베딩 검색
            search_results = self.embedding_engine.search(query, top_k=10)
            for node_id, score in search_results:
                if node_id in self.knowledge_graph.nodes:
                    node = self.knowledge_graph.nodes[node_id]
                    results.append({
                        'node_id': node_id,
                        'content': node.content,
                        'type': node.node_type,
                        'score': score,
                        'metadata': node.metadata
                    })
        
        elif search_type == "graph":
            # 그래프 기반 검색
            for node_id, node in self.knowledge_graph.nodes.items():
                if query.lower() in node.content.lower():
                    related_nodes = self.knowledge_graph.find_related_nodes(node_id)
                    results.append({
                        'node_id': node_id,
                        'content': node.content,
                        'type': node.node_type,
                        'score': len(related_nodes),
                        'related_count': len(related_nodes),
                        'metadata': node.metadata
                    })
        
        # 검색 히스토리 저장
        self.save_search_history(query, results)
        
        return results
    
    def save_search_history(self, query: str, results: List[Dict[str, Any]]):
        """검색 히스토리 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history (query, results)
                VALUES (?, ?)
            ''', (query, json.dumps(results)))
            conn.commit()
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """지식 통계"""
        stats = {
            'total_nodes': len(self.knowledge_graph.nodes),
            'total_relationships': len(self.knowledge_graph.graph.edges()),
            'node_types': {},
            'most_connected_nodes': [],
            'recent_additions': []
        }
        
        # 노드 타입별 통계
        for node in self.knowledge_graph.nodes.values():
            stats['node_types'][node.node_type] = stats['node_types'].get(node.node_type, 0) + 1
        
        # 가장 연결된 노드들
        centrality = self.knowledge_graph.get_node_centrality()
        if centrality:
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['most_connected_nodes'] = [
                {
                    'node_id': node_id,
                    'centrality': score,
                    'content': self.knowledge_graph.nodes[node_id].content[:100] + "..."
                }
                for node_id, score in sorted_nodes
            ]
        
        # 최근 추가된 노드들
        recent_nodes = sorted(
            self.knowledge_graph.nodes.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:5]
        
        stats['recent_additions'] = [
            {
                'node_id': node.node_id,
                'content': node.content[:100] + "...",
                'created_at': node.created_at.isoformat()
            }
            for node in recent_nodes
        ]
        
        return stats
    
    def auto_update_knowledge(self):
        """자동 지식 업데이트"""
        if not st.session_state[self.session_key]['auto_update_enabled']:
            return
        
        # 관련성 점수 업데이트
        self.update_relevance_scores()
        
        # 오래된 지식 정리
        self.cleanup_old_knowledge()
        
        # 새로운 관계 발견
        self.discover_new_relationships()
        
        # 업데이트 시간 기록
        st.session_state[self.session_key]['last_update'] = datetime.now()
    
    def update_relevance_scores(self):
        """관련성 점수 업데이트"""
        for node in self.knowledge_graph.nodes.values():
            # 접근 횟수 기반 점수 계산
            base_score = node.access_count * 0.1
            
            # 관계 수 기반 점수
            degree = self.knowledge_graph.graph.degree(node.node_id)
            relation_score = degree * 0.2
            
            # 최근성 점수
            days_old = (datetime.now() - node.created_at).days
            recency_score = max(0, 1 - days_old * 0.01)
            
            # 총 점수
            node.relevance_score = base_score + relation_score + recency_score
            
            # 데이터베이스 업데이트
            self.save_node_to_db(node)
    
    def cleanup_old_knowledge(self, days_threshold: int = 90):
        """오래된 지식 정리"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        nodes_to_remove = []
        for node_id, node in self.knowledge_graph.nodes.items():
            if (node.created_at < cutoff_date and 
                node.access_count < 5 and 
                node.relevance_score < 0.5):
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self.remove_knowledge(node_id)
    
    def discover_new_relationships(self):
        """새로운 관계 발견"""
        # 기존 관계가 없는 노드들 간의 유사도 계산
        nodes_list = list(self.knowledge_graph.nodes.keys())
        
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                if not self.knowledge_graph.graph.has_edge(node1, node2):
                    # 임베딩 유사도 계산
                    similar_docs = self.embedding_engine.get_similar_documents(node1, top_k=10)
                    
                    for similar_id, similarity in similar_docs:
                        if similar_id == node2 and similarity > 0.6:
                            self.knowledge_graph.add_edge(node1, node2, "discovered", similarity)
                            self.save_relationship_to_db(node1, node2, "discovered", similarity)
    
    def remove_knowledge(self, node_id: str):
        """지식 제거"""
        if node_id in self.knowledge_graph.nodes:
            # 그래프에서 제거
            self.knowledge_graph.graph.remove_node(node_id)
            del self.knowledge_graph.nodes[node_id]
            
            # 데이터베이스에서 제거
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM knowledge_nodes WHERE node_id = ?', (node_id,))
                cursor.execute('DELETE FROM knowledge_relationships WHERE source_id = ? OR target_id = ?', (node_id, node_id))
                conn.commit()
    
    def render_knowledge_dashboard(self):
        """지식 대시보드 렌더링"""
        st.markdown("## 🧠 Shared Knowledge Bank")
        
        # 통계 표시
        stats = self.get_knowledge_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 지식 노드", stats['total_nodes'])
        
        with col2:
            st.metric("총 관계", stats['total_relationships'])
        
        with col3:
            st.metric("노드 타입", len(stats['node_types']))
        
        with col4:
            auto_update = st.checkbox(
                "자동 업데이트",
                value=st.session_state[self.session_key]['auto_update_enabled']
            )
            st.session_state[self.session_key]['auto_update_enabled'] = auto_update
        
        # 검색 인터페이스
        st.markdown("### 🔍 지식 검색")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("검색어 입력", key="knowledge_search")
        
        with col2:
            search_type = st.selectbox("검색 방식", ["embedding", "graph"], key="search_type")
        
        if st.button("검색") and search_query:
            results = self.search_knowledge(search_query, search_type)
            
            if results:
                st.markdown(f"### 📊 검색 결과 ({len(results)}개)")
                
                for result in results:
                    with st.expander(f"📄 {result['content'][:100]}...", expanded=False):
                        st.markdown(f"**타입:** {result['type']}")
                        st.markdown(f"**점수:** {result['score']:.3f}")
                        st.markdown(f"**내용:** {result['content']}")
                        
                        if result.get('metadata'):
                            st.json(result['metadata'])
            else:
                st.info("검색 결과가 없습니다.")
        
        # 지식 추가 인터페이스
        st.markdown("### ➕ 지식 추가")
        
        with st.form("add_knowledge_form"):
            new_content = st.text_area("지식 내용", height=100)
            knowledge_type = st.selectbox("지식 타입", ["insight", "pattern", "rule", "fact", "procedure"])
            
            if st.form_submit_button("지식 추가"):
                if new_content.strip():
                    node_id = self.add_knowledge(new_content, knowledge_type)
                    st.success(f"지식이 추가되었습니다 (ID: {node_id})")
                    st.rerun()
        
        # 지식 그래프 시각화
        if st.button("지식 그래프 시각화"):
            self.render_knowledge_graph()
    
    def render_knowledge_graph(self):
        """지식 그래프 시각화"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # 그래프 데이터 준비
            edge_x = []
            edge_y = []
            
            # 노드 위치 계산
            pos = nx.spring_layout(self.knowledge_graph.graph, k=1, iterations=50)
            
            # 엣지 그리기
            for edge in self.knowledge_graph.graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # 엣지 트레이스
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # 노드 데이터
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node_id in self.knowledge_graph.graph.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                node = self.knowledge_graph.nodes[node_id]
                node_text.append(f"{node.content[:50]}...")
                
                # 노드 타입별 색상
                color_map = {
                    'insight': '#ff6b6b',
                    'pattern': '#4ecdc4',
                    'rule': '#45b7d1',
                    'fact': '#96ceb4',
                    'procedure': '#feca57'
                }
                node_colors.append(color_map.get(node.node_type, '#95a5a6'))
            
            # 노드 트레이스
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=10,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # 레이아웃
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='지식 그래프',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="지식 노드와 관계를 시각화한 그래프",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002 ) ],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                           ))
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.warning("Plotly가 설치되지 않아 그래프 시각화를 표시할 수 없습니다.")
        except Exception as e:
            st.error(f"그래프 시각화 오류: {e}")


def create_shared_knowledge_bank() -> SharedKnowledgeBank:
    """공유 지식 은행 인스턴스 생성"""
    return SharedKnowledgeBank() 