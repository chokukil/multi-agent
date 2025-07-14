#!/usr/bin/env python3
"""
🧪 CherryAI Shared Knowledge Bank 상세 테스트

고급 임베딩 검색 시스템의 기능을 단계별로 검증합니다.
"""

import asyncio
import os
import sys
import logging
import time
import shutil

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(__file__))

from core.shared_knowledge_bank import (
    AdvancedSharedKnowledgeBank,
    KnowledgeType,
    SearchStrategy,
    initialize_shared_knowledge_bank,
    add_user_file_knowledge,
    add_agent_memory,
    search_relevant_knowledge
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_embedding_functionality():
    """임베딩 기능 테스트"""
    print("\n🧪 Step 1: 임베딩 기능 테스트")
    
    # 테스트용 임시 디렉토리
    test_dir = "./test_detailed_kb"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    kb = AdvancedSharedKnowledgeBank(
        persist_directory=test_dir,
        embedding_model="all-MiniLM-L6-v2",
        max_chunk_size=200
    )
    
    # 간단한 임베딩 테스트
    test_text = "CherryAI는 A2A 프로토콜을 사용하는 AI 플랫폼입니다."
    embedding = await kb._generate_embedding(test_text)
    
    print(f"✅ 임베딩 생성 성공: {len(embedding)}차원")
    print(f"   샘플 값: {embedding[:5]}")
    
    return kb

async def test_knowledge_addition(kb):
    """지식 추가 테스트"""
    print("\n🧪 Step 2: 지식 추가 테스트")
    
    test_knowledge = [
        {
            "content": "CherryAI는 A2A 프로토콜과 MCP 도구를 통합한 첫 번째 AI 플랫폼입니다. LLM First 원칙을 따르며 하드코딩을 금지합니다.",
            "type": KnowledgeType.DOMAIN_KNOWLEDGE,
            "agent": "system",
            "title": "CherryAI 소개"
        },
        {
            "content": "A2A는 Agent-to-Agent의 약자로, 에이전트 간 통신을 위한 표준 프로토콜입니다. 실시간 스트리밍과 SSE를 지원합니다.",
            "type": KnowledgeType.DOMAIN_KNOWLEDGE,
            "agent": "a2a_expert",
            "title": "A2A 프로토콜"
        },
        {
            "content": "MCP는 Model Context Protocol의 약자로, AI 모델과 도구 간의 표준 인터페이스를 제공합니다. 다양한 도구를 통합할 수 있습니다.",
            "type": KnowledgeType.DOMAIN_KNOWLEDGE,
            "agent": "mcp_expert",
            "title": "MCP 프로토콜"
        },
        {
            "content": "데이터 분석을 위해서는 pandas, numpy, matplotlib 등의 라이브러리가 필요합니다. 탐색적 데이터 분석이 중요합니다.",
            "type": KnowledgeType.BEST_PRACTICE,
            "agent": "data_analyst",
            "title": "데이터 분석 베스트 프랙티스"
        }
    ]
    
    entry_ids = []
    for i, knowledge in enumerate(test_knowledge):
        start_time = time.time()
        entry_id = await kb.add_knowledge(
            content=knowledge["content"],
            knowledge_type=knowledge["type"],
            source_agent=knowledge["agent"],
            title=knowledge["title"]
        )
        elapsed = time.time() - start_time
        
        entry_ids.append(entry_id)
        print(f"✅ 지식 {i+1} 추가 완료: {knowledge['title']} ({elapsed:.3f}초)")
    
    return entry_ids

async def test_search_functionality(kb, entry_ids):
    """검색 기능 테스트"""
    print("\n🧪 Step 3: 검색 기능 테스트")
    
    test_queries = [
        "A2A 프로토콜이란 무엇인가요?",
        "CherryAI 플랫폼 특징",
        "MCP 도구 통합",
        "데이터 분석 방법",
        "에이전트 통신",
        "AI 플랫폼"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 검색 {i+1}: '{query}'")
        start_time = time.time()
        
        results = await kb.search_knowledge(
            query=query,
            strategy=SearchStrategy.HYBRID,
            max_results=3,
            min_similarity=0.1  # 낮은 임계값으로 테스트
        )
        
        elapsed = time.time() - start_time
        print(f"   결과: {len(results)}개 ({elapsed:.3f}초)")
        
        for j, result in enumerate(results):
            print(f"   {j+1}. {result.title} (유사도: {result.similarity_score:.3f})")
            print(f"      내용: {result.context_snippet}")
            print(f"      소스: {result.source_agent}")

async def test_agent_specific_search(kb):
    """에이전트별 검색 테스트"""
    print("\n🧪 Step 4: 에이전트별 검색 테스트")
    
    agents = ["system", "a2a_expert", "mcp_expert", "data_analyst"]
    
    for agent in agents:
        results = await kb.get_agent_knowledge(agent, limit=10)
        print(f"✅ {agent} 지식: {len(results)}개")
        
        for result in results:
            print(f"   - {result.title}")

async def test_stats_and_performance(kb):
    """통계 및 성능 테스트"""
    print("\n🧪 Step 5: 통계 및 성능 테스트")
    
    stats = await kb.get_stats()
    print(f"✅ 총 항목: {stats.total_entries}개")
    print(f"✅ 평균 응답시간: {stats.avg_response_time:.3f}초")
    print(f"✅ 저장소 크기: {stats.storage_size_mb:.2f}MB")
    print(f"✅ 총 검색: {stats.total_searches}회")
    
    # 타입별 통계
    print("\n📊 타입별 통계:")
    for knowledge_type, count in stats.entries_by_type.items():
        print(f"   {knowledge_type.value}: {count}개")

async def test_direct_chromadb_query(kb):
    """ChromaDB 직접 쿼리 테스트"""
    print("\n🧪 Step 6: ChromaDB 직접 쿼리 테스트")
    
    # 모든 데이터 조회
    all_results = kb.collection.get(include=["documents", "metadatas"])
    print(f"✅ 저장된 총 청크: {len(all_results['documents'])}개")
    
    for i, (doc, metadata) in enumerate(zip(all_results["documents"][:3], all_results["metadatas"][:3])):
        print(f"\n청크 {i+1}:")
        print(f"   제목: {metadata.get('title', 'N/A')}")
        print(f"   타입: {metadata.get('knowledge_type', 'N/A')}")
        print(f"   에이전트: {metadata.get('source_agent', 'N/A')}")
        print(f"   내용: {doc[:100]}...")
    
    # 임베딩 검색 테스트
    test_embedding = await kb._generate_embedding("A2A 프로토콜")
    raw_results = kb.collection.query(
        query_embeddings=[test_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\n🔍 원시 임베딩 검색 결과: {len(raw_results['documents'][0])}개")
    for i, (doc, metadata, distance) in enumerate(zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0]
    )):
        similarity = 1 - distance
        print(f"   {i+1}. 거리: {distance:.3f}, 유사도: {similarity:.3f}")
        print(f"      제목: {metadata.get('title', 'N/A')}")
        print(f"      내용: {doc[:80]}...")

async def main():
    """메인 테스트 함수"""
    print("🚀 CherryAI Shared Knowledge Bank 상세 테스트 시작")
    
    try:
        # Step 1: 임베딩 기능 테스트
        kb = await test_embedding_functionality()
        
        # Step 2: 지식 추가 테스트
        entry_ids = await test_knowledge_addition(kb)
        
        # Step 3: 검색 기능 테스트
        await test_search_functionality(kb, entry_ids)
        
        # Step 4: 에이전트별 검색 테스트
        await test_agent_specific_search(kb)
        
        # Step 5: 통계 및 성능 테스트
        await test_stats_and_performance(kb)
        
        # Step 6: ChromaDB 직접 쿼리 테스트
        await test_direct_chromadb_query(kb)
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 정리
        test_dir = "./test_detailed_kb"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("\n🧹 테스트 디렉토리 정리 완료")

if __name__ == "__main__":
    asyncio.run(main()) 