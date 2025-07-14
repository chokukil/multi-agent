#!/usr/bin/env python3
"""
🚀 CherryAI Shared Knowledge Bank 성능 벤치마크 테스트

대용량 데이터와 다양한 시나리오에서의 성능을 검증합니다.
"""

import asyncio
import os
import sys
import logging
import time
import shutil
import random
import statistics
from typing import List, Dict

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(__file__))

from core.shared_knowledge_bank import (
    AdvancedSharedKnowledgeBank,
    KnowledgeType,
    SearchStrategy,
    initialize_shared_knowledge_bank
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """성능 벤치마크 테스트"""
    
    def __init__(self):
        self.test_dir = "./perf_test_kb"
        self.kb = None
        self.test_data = []
        self.performance_metrics = {}
    
    async def setup(self):
        """테스트 환경 설정"""
        print("🔧 테스트 환경 설정 중...")
        
        # 기존 테스트 디렉토리 정리
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # 지식 뱅크 초기화
        self.kb = AdvancedSharedKnowledgeBank(
            persist_directory=self.test_dir,
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            max_chunk_size=300,  # 성능 테스트용 작은 청크
            enable_cache=True
        )
        
        # 테스트 데이터 생성
        self.test_data = self._generate_test_data()
        print(f"✅ 테스트 데이터 생성 완료: {len(self.test_data)}개")
    
    def _generate_test_data(self) -> List[Dict]:
        """테스트용 지식 데이터 생성"""
        test_data = []
        
        # 기본 도메인 지식
        base_knowledge = [
            {
                "content": "CherryAI는 A2A 프로토콜과 MCP 도구를 통합한 AI 플랫폼입니다. LLM First 원칙을 따르며 실시간 스트리밍을 지원합니다.",
                "type": KnowledgeType.DOMAIN_KNOWLEDGE,
                "agent": "system",
                "title": "CherryAI 플랫폼 개요"
            },
            {
                "content": "A2A는 Agent-to-Agent 통신 프로토콜로, 에이전트 간 실시간 메시지 교환을 가능하게 합니다. SSE 스트리밍을 사용합니다.",
                "type": KnowledgeType.DOMAIN_KNOWLEDGE,
                "agent": "a2a_expert",
                "title": "A2A 프로토콜 소개"
            },
            {
                "content": "MCP는 Model Context Protocol로, AI 모델과 외부 도구 간의 표준 인터페이스를 제공합니다. 모듈화와 확장성이 핵심입니다.",
                "type": KnowledgeType.DOMAIN_KNOWLEDGE,
                "agent": "mcp_expert",
                "title": "MCP 프로토콜 설명"
            },
            {
                "content": "데이터 분석에서는 pandas, numpy, matplotlib을 주로 사용합니다. 탐색적 데이터 분석(EDA)이 중요한 첫 단계입니다.",
                "type": KnowledgeType.BEST_PRACTICE,
                "agent": "data_analyst",
                "title": "데이터 분석 도구"
            },
            {
                "content": "머신러닝 모델 훈련 시 교차 검증과 하이퍼파라미터 튜닝이 필수입니다. 과적합 방지를 위한 정규화도 중요합니다.",
                "type": KnowledgeType.BEST_PRACTICE,
                "agent": "ml_engineer",
                "title": "ML 모델 훈련 베스트 프랙티스"
            }
        ]
        
        # 기본 데이터 추가
        test_data.extend(base_knowledge)
        
        # 대용량 데이터 생성 (확장성 테스트용)
        domains = ["AI", "데이터사이언스", "소프트웨어", "클라우드", "보안"]
        agents = ["specialist_1", "specialist_2", "analyst", "engineer", "researcher"]
        
        for i in range(50):  # 50개 추가 데이터
            domain = random.choice(domains)
            agent = random.choice(agents)
            
            content = f"{domain} 분야에서는 다양한 기술과 방법론이 활용됩니다. " \
                     f"특히 {agent}가 담당하는 영역에서는 전문적인 지식이 필요합니다. " \
                     f"실무 경험과 이론적 배경이 모두 중요한 요소입니다. " \
                     f"최신 동향과 기술 발전을 지속적으로 모니터링해야 합니다."
            
            test_data.append({
                "content": content,
                "type": random.choice([KnowledgeType.DOMAIN_KNOWLEDGE, KnowledgeType.BEST_PRACTICE, KnowledgeType.AGENT_MEMORY]),
                "agent": agent,
                "title": f"{domain} 전문 지식 #{i+1}"
            })
        
        return test_data
    
    async def test_bulk_insertion_performance(self):
        """대량 데이터 삽입 성능 테스트"""
        print("\n🧪 대량 데이터 삽입 성능 테스트")
        
        insertion_times = []
        start_total = time.time()
        
        for i, data in enumerate(self.test_data):
            start_time = time.time()
            
            entry_id = await self.kb.add_knowledge(
                content=data["content"],
                knowledge_type=data["type"],
                source_agent=data["agent"],
                title=data["title"]
            )
            
            elapsed = time.time() - start_time
            insertion_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                avg_time = statistics.mean(insertion_times[-10:])
                print(f"   진행: {i+1}/{len(self.test_data)} (최근 10개 평균: {avg_time:.3f}초)")
        
        total_time = time.time() - start_total
        
        self.performance_metrics["insertion"] = {
            "total_items": len(self.test_data),
            "total_time": total_time,
            "avg_time_per_item": statistics.mean(insertion_times),
            "median_time": statistics.median(insertion_times),
            "max_time": max(insertion_times),
            "min_time": min(insertion_times)
        }
        
        print(f"✅ 삽입 완료: {len(self.test_data)}개 항목")
        print(f"   총 시간: {total_time:.3f}초")
        print(f"   평균 시간: {statistics.mean(insertion_times):.3f}초/항목")
        print(f"   처리량: {len(self.test_data)/total_time:.1f}항목/초")
    
    async def test_search_performance(self):
        """검색 성능 테스트"""
        print("\n🧪 검색 성능 테스트")
        
        search_queries = [
            "CherryAI 플랫폼",
            "A2A 프로토콜",
            "MCP 도구",
            "데이터 분석",
            "머신러닝",
            "AI 기술",
            "소프트웨어 개발",
            "클라우드 컴퓨팅",
            "보안 시스템",
            "전문 지식"
        ]
        
        search_times = []
        result_counts = []
        
        for query in search_queries:
            start_time = time.time()
            
            results = await self.kb.search_knowledge(
                query=query,
                strategy=SearchStrategy.HYBRID,
                max_results=10,
                min_similarity=0.15
            )
            
            elapsed = time.time() - start_time
            search_times.append(elapsed)
            result_counts.append(len(results))
            
            print(f"   '{query}': {len(results)}개 결과 ({elapsed:.3f}초)")
        
        self.performance_metrics["search"] = {
            "total_queries": len(search_queries),
            "avg_search_time": statistics.mean(search_times),
            "median_search_time": statistics.median(search_times),
            "max_search_time": max(search_times),
            "min_search_time": min(search_times),
            "avg_results": statistics.mean(result_counts),
            "total_results": sum(result_counts)
        }
        
        print(f"✅ 검색 테스트 완료")
        print(f"   평균 검색 시간: {statistics.mean(search_times):.3f}초")
        print(f"   평균 결과 수: {statistics.mean(result_counts):.1f}개")
    
    async def test_concurrent_operations(self):
        """동시성 테스트"""
        print("\n🧪 동시성 테스트")
        
        # 동시 검색 테스트
        search_tasks = []
        queries = ["AI", "데이터", "프로토콜", "분석", "기술"] * 4  # 20개 동시 검색
        
        start_time = time.time()
        
        for query in queries:
            task = self.kb.search_knowledge(
                query=query,
                max_results=5,
                min_similarity=0.15
            )
            search_tasks.append(task)
        
        # 모든 검색 동시 실행
        results = await asyncio.gather(*search_tasks)
        
        elapsed = time.time() - start_time
        total_results = sum(len(result) for result in results)
        
        self.performance_metrics["concurrency"] = {
            "concurrent_searches": len(search_tasks),
            "total_time": elapsed,
            "avg_time_per_search": elapsed / len(search_tasks),
            "total_results": total_results,
            "throughput": len(search_tasks) / elapsed
        }
        
        print(f"✅ 동시성 테스트 완료")
        print(f"   동시 검색: {len(search_tasks)}개")
        print(f"   총 시간: {elapsed:.3f}초")
        print(f"   처리량: {len(search_tasks)/elapsed:.1f}검색/초")
    
    async def test_memory_usage(self):
        """메모리 사용량 테스트"""
        print("\n🧪 메모리 사용량 테스트")
        
        try:
            import psutil
            process = psutil.Process()
            
            # 초기 메모리
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 통계 조회
            stats = await self.kb.get_stats()
            
            # 현재 메모리
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = current_memory - initial_memory
            
            self.performance_metrics["memory"] = {
                "initial_memory_mb": initial_memory,
                "current_memory_mb": current_memory,
                "memory_usage_mb": memory_usage,
                "storage_size_mb": stats.storage_size_mb,
                "items_count": stats.total_entries
            }
            
            print(f"✅ 메모리 사용량 분석 완료")
            print(f"   현재 메모리: {current_memory:.1f}MB")
            print(f"   저장소 크기: {stats.storage_size_mb:.1f}MB")
            print(f"   항목당 메모리: {memory_usage/stats.total_entries:.3f}MB" if stats.total_entries > 0 else "   항목당 메모리: 0MB")
            
        except ImportError:
            print("⚠️ psutil 패키지가 없어서 메모리 테스트를 건너뜁니다")
            self.performance_metrics["memory"] = {"error": "psutil not available"}
    
    async def test_scaling_characteristics(self):
        """확장성 특성 테스트"""
        print("\n🧪 확장성 특성 테스트")
        
        # 데이터 크기별 검색 성능
        batch_sizes = [10, 25, 50]  # 현재 데이터 기준
        scaling_results = {}
        
        for batch_size in batch_sizes:
            if batch_size <= len(self.test_data):
                # 해당 크기까지의 데이터로 검색 테스트
                start_time = time.time()
                
                results = await self.kb.search_knowledge(
                    query="전문 지식",
                    max_results=batch_size,
                    min_similarity=0.1
                )
                
                elapsed = time.time() - start_time
                scaling_results[batch_size] = {
                    "search_time": elapsed,
                    "result_count": len(results)
                }
                
                print(f"   {batch_size}개 결과 검색: {elapsed:.3f}초")
        
        self.performance_metrics["scaling"] = scaling_results
    
    def print_performance_summary(self):
        """성능 요약 출력"""
        print("\n📊 성능 테스트 종합 결과")
        print("=" * 50)
        
        # 삽입 성능
        if "insertion" in self.performance_metrics:
            insertion = self.performance_metrics["insertion"]
            print(f"🔧 데이터 삽입 성능:")
            print(f"   처리량: {insertion['total_items']/insertion['total_time']:.1f} 항목/초")
            print(f"   평균 지연: {insertion['avg_time_per_item']*1000:.1f}ms")
        
        # 검색 성능
        if "search" in self.performance_metrics:
            search = self.performance_metrics["search"]
            print(f"\n🔍 검색 성능:")
            print(f"   평균 응답시간: {search['avg_search_time']*1000:.1f}ms")
            print(f"   평균 결과 수: {search['avg_results']:.1f}개")
        
        # 동시성
        if "concurrency" in self.performance_metrics:
            concurrency = self.performance_metrics["concurrency"]
            print(f"\n⚡ 동시성 성능:")
            print(f"   동시 처리량: {concurrency['throughput']:.1f} 요청/초")
            print(f"   평균 지연: {concurrency['avg_time_per_search']*1000:.1f}ms")
        
        # 메모리
        if "memory" in self.performance_metrics and "error" not in self.performance_metrics["memory"]:
            memory = self.performance_metrics["memory"]
            print(f"\n💾 메모리 사용량:")
            print(f"   현재 메모리: {memory['current_memory_mb']:.1f}MB")
            print(f"   저장소 크기: {memory['storage_size_mb']:.1f}MB")
        
        # 성능 등급 평가
        self._evaluate_performance_grade()
    
    def _evaluate_performance_grade(self):
        """성능 등급 평가"""
        print(f"\n🏆 성능 등급 평가:")
        
        grade_points = 0
        max_points = 0
        
        # 검색 응답시간 평가 (50ms 이하: A+, 100ms 이하: A, 200ms 이하: B, 그 이상: C)
        if "search" in self.performance_metrics:
            search_time_ms = self.performance_metrics["search"]["avg_search_time"] * 1000
            max_points += 4
            if search_time_ms <= 50:
                grade_points += 4
                print(f"   검색 응답시간: A+ ({search_time_ms:.1f}ms)")
            elif search_time_ms <= 100:
                grade_points += 3
                print(f"   검색 응답시간: A ({search_time_ms:.1f}ms)")
            elif search_time_ms <= 200:
                grade_points += 2
                print(f"   검색 응답시간: B ({search_time_ms:.1f}ms)")
            else:
                grade_points += 1
                print(f"   검색 응답시간: C ({search_time_ms:.1f}ms)")
        
        # 삽입 처리량 평가 (10 items/s 이상: A+, 5 items/s 이상: A, 2 items/s 이상: B, 그 이하: C)
        if "insertion" in self.performance_metrics:
            insertion = self.performance_metrics["insertion"]
            throughput = insertion['total_items'] / insertion['total_time']
            max_points += 4
            if throughput >= 10:
                grade_points += 4
                print(f"   삽입 처리량: A+ ({throughput:.1f} items/s)")
            elif throughput >= 5:
                grade_points += 3
                print(f"   삽입 처리량: A ({throughput:.1f} items/s)")
            elif throughput >= 2:
                grade_points += 2
                print(f"   삽입 처리량: B ({throughput:.1f} items/s)")
            else:
                grade_points += 1
                print(f"   삽입 처리량: C ({throughput:.1f} items/s)")
        
        # 동시성 처리량 평가
        if "concurrency" in self.performance_metrics:
            concurrent_throughput = self.performance_metrics["concurrency"]["throughput"]
            max_points += 4
            if concurrent_throughput >= 20:
                grade_points += 4
                print(f"   동시성 처리량: A+ ({concurrent_throughput:.1f} req/s)")
            elif concurrent_throughput >= 10:
                grade_points += 3
                print(f"   동시성 처리량: A ({concurrent_throughput:.1f} req/s)")
            elif concurrent_throughput >= 5:
                grade_points += 2
                print(f"   동시성 처리량: B ({concurrent_throughput:.1f} req/s)")
            else:
                grade_points += 1
                print(f"   동시성 처리량: C ({concurrent_throughput:.1f} req/s)")
        
        # 종합 점수
        if max_points > 0:
            final_score = (grade_points / max_points) * 100
            if final_score >= 90:
                final_grade = "A+"
            elif final_score >= 80:
                final_grade = "A"
            elif final_score >= 70:
                final_grade = "B"
            else:
                final_grade = "C"
            
            print(f"\n🎯 종합 성능 등급: {final_grade} ({final_score:.1f}점)")
        
        # 권장사항
        print(f"\n💡 성능 최적화 권장사항:")
        if "search" in self.performance_metrics:
            search_time = self.performance_metrics["search"]["avg_search_time"] * 1000
            if search_time > 100:
                print(f"   - 검색 응답시간 개선 필요 (현재: {search_time:.1f}ms)")
                print(f"   - 임베딩 캐시 크기 증가 고려")
                print(f"   - 더 작은 임베딩 모델 사용 검토")
    
    async def cleanup(self):
        """테스트 환경 정리"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("\n🧹 테스트 환경 정리 완료")

async def main():
    """메인 성능 테스트 함수"""
    print("🚀 CherryAI Shared Knowledge Bank 성능 벤치마크 테스트 시작")
    
    benchmark = PerformanceBenchmark()
    
    try:
        # 테스트 실행
        await benchmark.setup()
        await benchmark.test_bulk_insertion_performance()
        await benchmark.test_search_performance()
        await benchmark.test_concurrent_operations()
        await benchmark.test_memory_usage()
        await benchmark.test_scaling_characteristics()
        
        # 결과 출력
        benchmark.print_performance_summary()
        
    except Exception as e:
        print(f"\n❌ 성능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 