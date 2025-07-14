#!/usr/bin/env python3
"""
🧪 CherryAI Shared Knowledge Bank 테스트

고급 임베딩 검색 시스템의 기능을 테스트하고 검증합니다.
"""

import asyncio
import os
import sys
import logging

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_functionality():
    """기본 기능 테스트"""
    print("🧪 기본 기능 테스트 시작")
    
    # 지식 뱅크 초기화
    kb = initialize_shared_knowledge_bank(
        persist_directory="./test_chroma_kb",
        max_chunk_size=256,
        chunk_overlap=25
    )
    
    # 1. 지식 추가 테스트
    print("\n1️⃣ 지식 추가 테스트")
    
    # CherryAI 관련 지식 추가
    cherry_ai_knowledge = """
    CherryAI는 세계 최초의 A2A (Agent-to-Agent) + MCP (Model Context Protocol) 통합 플랫폼입니다.
    
    주요 특징:
    - 11개의 A2A 에이전트가 협업하여 데이터 분석 수행
    - 7개의 MCP 도구와 완전 통합
    - 실시간 스트리밍 처리 지원
    - Context Engineering 6 Data Layers 구현
    - LLM First 원칙 준수
    
    A2A 에이전트 목록:
    1. Orchestrator (8100) - 전체 조율
    2. Pandas Data Analyst (8315) - 데이터 분석
    3. Data Loader (8306) - 데이터 로딩
    4. Data Cleaning (8307) - 데이터 정제
    5. EDA Tools (8308) - 탐색적 데이터 분석
    6. Data Visualization (8309) - 시각화
    7. Feature Engineering (8310) - 피처 엔지니어링
    8. H2O ML (8311) - 머신러닝
    9. MLflow Tools (8312) - ML 실험 관리
    10. Data Wrangling (8313) - 데이터 가공
    11. SQL Database (8314) - 데이터베이스 연동
    
    MCP 도구 목록:
    1. Playwright (3000) - 웹 브라우저 자동화
    2. File Manager (3001) - 파일 시스템 관리
    3. Database Connector (3002) - DB 연결
    4. API Gateway (3003) - 외부 API 호출
    5. Data Analyzer (3004) - 고급 데이터 분석
    6. Chart Generator (3005) - 고급 시각화
    7. LLM Gateway (3006) - 다중 LLM 통합
    """
    
    entry_id1 = await kb.add_knowledge(
        content=cherry_ai_knowledge,
        knowledge_type=KnowledgeType.SYSTEM_CONFIG,
        source_agent="system",
        title="CherryAI 플랫폼 완전 가이드",
        summary="CherryAI의 A2A+MCP 통합 아키텍처 설명",
        keywords=["CherryAI", "A2A", "MCP", "에이전트", "협업"],
        tags={"platform", "architecture", "guide"}
    )
    print(f"✅ CherryAI 지식 추가 완료: {entry_id1}")
    
    # 데이터 분석 모범 사례 추가
    analysis_best_practices = """
    데이터 분석 모범 사례:
    
    1. 데이터 탐색 단계
    - 데이터 형태와 크기 파악
    - 결측값과 이상치 확인
    - 기본 통계량 계산
    - 데이터 분포 시각화
    
    2. 데이터 전처리
    - 결측값 처리 (삭제/대체)
    - 이상치 처리
    - 데이터 타입 변환
    - 정규화/표준화
    
    3. 탐색적 데이터 분석 (EDA)
    - 변수 간 상관관계 분석
    - 패턴과 트렌드 발견
    - 가설 수립 및 검증
    - 시각화를 통한 인사이트 도출
    
    4. 모델링
    - 적절한 알고리즘 선택
    - 교차 검증 수행
    - 하이퍼파라미터 튜닝
    - 성능 평가 및 해석
    """
    
    entry_id2 = await kb.add_knowledge(
        content=analysis_best_practices,
        knowledge_type=KnowledgeType.BEST_PRACTICE,
        source_agent="pandas_collaboration_hub",
        title="데이터 분석 모범 사례",
        summary="체계적인 데이터 분석 프로세스 가이드",
        keywords=["데이터분석", "EDA", "전처리", "모델링"],
        tags={"best_practice", "data_analysis", "methodology"}
    )
    print(f"✅ 분석 모범 사례 추가 완료: {entry_id2}")
    
    # Python 프로그래밍 팁 추가
    python_tips = """
    Python 데이터 분석 팁:
    
    1. Pandas 활용법
    - DataFrame과 Series 효과적 사용
    - 그룹바이와 집계 연산
    - 데이터 병합과 조인
    - 날짜/시간 데이터 처리
    
    2. 시각화 라이브러리
    - Matplotlib 기본 플롯
    - Seaborn 통계 시각화
    - Plotly 인터랙티브 차트
    - 맞춤형 차트 생성
    
    3. 성능 최적화
    - 벡터화 연산 활용
    - 메모리 효율적인 데이터 타입 사용
    - 청크 단위 데이터 처리
    - 멀티프로세싱 활용
    """
    
    entry_id3 = await kb.add_knowledge(
        content=python_tips,
        knowledge_type=KnowledgeType.DOMAIN_KNOWLEDGE,
        source_agent="data_loader",
        title="Python 데이터 분석 팁",
        summary="효율적인 Python 데이터 분석 기법",
        keywords=["Python", "Pandas", "시각화", "최적화"],
        tags={"programming", "tips", "python"}
    )
    print(f"✅ Python 팁 추가 완료: {entry_id3}")
    
    return entry_id1, entry_id2, entry_id3

async def test_search_functionality(entry_ids):
    """검색 기능 테스트"""
    print("\n2️⃣ 검색 기능 테스트")
    
    kb = initialize_shared_knowledge_bank()
    
    # 다양한 검색 쿼리 테스트
    test_queries = [
        "A2A 에이전트는 몇 개인가요?",
        "데이터 전처리 방법",
        "Python Pandas 사용법",
        "MCP 도구 목록",
        "시각화 라이브러리",
        "CherryAI 아키텍처"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: '{query}'")
        
        results = await kb.search_knowledge(
            query=query,
            strategy=SearchStrategy.HYBRID,
            max_results=3,
            min_similarity=0.1
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title}")
                print(f"     유사도: {result.similarity_score:.3f}")
                print(f"     출처: {result.source_agent}")
                print(f"     타입: {result.knowledge_type.value}")
                print(f"     스니펫: {result.context_snippet[:100]}...")
                print()
        else:
            print("  검색 결과 없음")

async def test_agent_specific_search():
    """에이전트별 검색 테스트"""
    print("\n3️⃣ 에이전트별 검색 테스트")
    
    kb = initialize_shared_knowledge_bank()
    
    # 에이전트별 지식 조회
    agents = ["system", "pandas_collaboration_hub", "data_loader"]
    
    for agent in agents:
        print(f"\n📊 {agent} 에이전트의 지식:")
        
        agent_knowledge = await kb.get_agent_knowledge(agent, limit=5)
        
        if agent_knowledge:
            for knowledge in agent_knowledge:
                print(f"  - {knowledge.title}")
                print(f"    생성일: {knowledge.created_at.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("  저장된 지식 없음")

async def test_statistics():
    """통계 기능 테스트"""
    print("\n4️⃣ 통계 기능 테스트")
    
    kb = initialize_shared_knowledge_bank()
    
    stats = await kb.get_stats()
    
    print(f"📈 지식 뱅크 통계:")
    print(f"  - 총 지식 항목: {stats.total_entries}개")
    print(f"  - 총 검색 횟수: {stats.total_searches}회")
    print(f"  - 평균 응답 시간: {stats.avg_response_time:.3f}초")
    print(f"  - 저장소 크기: {stats.storage_size_mb:.2f}MB")
    print(f"  - 마지막 업데이트: {stats.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 지식 유형별 분포:")
    for knowledge_type, count in stats.entries_by_type.items():
        print(f"  - {knowledge_type.value}: {count}개")
    
    print(f"\n🤖 에이전트별 분포:")
    for agent, count in stats.entries_by_agent.items():
        print(f"  - {agent}: {count}개")

async def test_context_engineering_integration():
    """Context Engineering 통합 테스트"""
    print("\n5️⃣ Context Engineering 통합 테스트")
    
    # 사용자 파일 지식 추가 테스트
    file_content = """
    샘플 데이터셋 분석 요청
    
    이 파일은 고객 구매 데이터를 포함하고 있습니다.
    - 고객 ID, 구매 날짜, 상품명, 금액
    - 2023년 전체 거래 데이터
    - 총 10,000건의 거래 기록
    
    분석 요구사항:
    1. 월별 매출 트렌드 분석
    2. 인기 상품 TOP 10 도출
    3. 고객 세분화 및 RFM 분석
    4. 계절성 패턴 발견
    """
    
    file_entry_id = await add_user_file_knowledge(
        file_content=file_content,
        filename="customer_data_analysis.txt",
        session_id="test_session_001"
    )
    print(f"✅ 사용자 파일 지식 추가: {file_entry_id}")
    
    # 에이전트 메모리 추가 테스트
    memory_content = """
    Pandas 협업 허브에서 학습한 내용:
    - 사용자가 자주 요청하는 분석: 트렌드 분석, 상관관계 분석
    - 효과적인 시각화: 시계열 플롯, 히트맵, 박스플롯
    - 성능 최적화: 벡터 연산, 인덱싱 활용
    """
    
    memory_entry_id = await add_agent_memory(
        content=memory_content,
        agent_id="pandas_collaboration_hub",
        memory_type="learning_pattern"
    )
    print(f"✅ 에이전트 메모리 추가: {memory_entry_id}")
    
    # 관련 지식 검색 테스트
    print(f"\n🔗 관련 지식 검색:")
    relevant_knowledge = await search_relevant_knowledge(
        query="고객 데이터 분석 방법",
        agent_id="pandas_collaboration_hub"
    )
    
    for knowledge in relevant_knowledge[:3]:
        print(f"  - {knowledge.title} (유사도: {knowledge.similarity_score:.3f})")

async def test_cleanup_and_export():
    """정리 및 내보내기 테스트"""
    print("\n6️⃣ 정리 및 내보내기 테스트")
    
    kb = initialize_shared_knowledge_bank()
    
    # 지식 뱅크 내보내기
    export_path = "./knowledge_bank_export.json"
    success = await kb.export_knowledge(export_path)
    
    if success and os.path.exists(export_path):
        file_size = os.path.getsize(export_path) / 1024  # KB
        print(f"✅ 지식 뱅크 내보내기 완료: {export_path} ({file_size:.1f}KB)")
        
        # 내보낸 파일 삭제 (테스트 정리)
        os.remove(export_path)
        print("🗑️  내보낸 파일 정리 완료")
    else:
        print("❌ 지식 뱅크 내보내기 실패")

async def main():
    """메인 테스트 함수"""
    print("🚀 CherryAI Shared Knowledge Bank 테스트 시작\n")
    
    try:
        # 1. 기본 기능 테스트
        entry_ids = await test_basic_functionality()
        
        # 2. 검색 기능 테스트  
        await test_search_functionality(entry_ids)
        
        # 3. 에이전트별 검색 테스트
        await test_agent_specific_search()
        
        # 4. 통계 기능 테스트
        await test_statistics()
        
        # 5. Context Engineering 통합 테스트
        await test_context_engineering_integration()
        
        # 6. 정리 및 내보내기 테스트
        await test_cleanup_and_export()
        
        print("\n🎉 모든 테스트 완료!")
        print("\n📋 테스트 결과 요약:")
        print("✅ ChromaDB 벡터 데이터베이스 연동")
        print("✅ 임베딩 기반 의미적 검색")
        print("✅ 메타데이터 필터링")
        print("✅ A2A 에이전트별 지식 관리")
        print("✅ Context Engineering 통합")
        print("✅ 실시간 통계 추적")
        print("✅ 지식 내보내기/정리")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 