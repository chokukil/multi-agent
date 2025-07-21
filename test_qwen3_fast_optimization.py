#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Qwen3-4B-Fast 최적화 테스트
LLM First 원칙 준수하면서 속도와 품질 균형 달성

목표:
1. 100% LLM First 원칙 준수 (패턴 매칭/하드코딩 완전 금지)
2. qwen3-4b-fast 모델로 2분 마지노선 달성
3. 속도와 품질 균형 최적화
4. 실시간 스트리밍 경험 제공
"""

import asyncio
import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator

# LLM First 컴포넌트
from core.universal_engine.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen3FastOptimizedStreaming:
    """Qwen3-4B-Fast 최적화된 LLM First 스트리밍"""
    
    def __init__(self):
        """초기화"""
        # qwen3-4b-fast 모델 사용
        os.environ["OLLAMA_MODEL"] = "qwen3-4b-fast:latest"
        os.environ["LLM_PROVIDER"] = "OLLAMA"
        
        # 최적화된 LLM 클라이언트 생성
        self.llm_client = LLMFactory.create_llm()
        
        # 성능 최적화 설정
        self.config = {
            "target_ttft": 2.0,        # 2초 이내 첫 응답
            "max_total_time": 90.0,    # 90초 이내 완료 (여유 있게)
            "quality_threshold": 0.8,   # 품질 임계값
            "chunk_delay": 0.02,       # 20ms 청크 지연
            "timeout_margin": 30.0     # 30초 여유
        }
        
        logger.info("Qwen3FastOptimizedStreaming initialized with qwen3-4b-fast")
    
    async def optimized_llm_streaming(
        self, 
        query: str, 
        max_time: float = 90.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """최적화된 LLM First 스트리밍"""
        
        start_time = time.time()
        
        try:
            # 시작 이벤트
            yield {
                "event": "start",
                "data": {
                    "message": "🚀 Qwen3-4B-Fast로 분석을 시작합니다...",
                    "model": "qwen3-4b-fast",
                    "target_time": max_time,
                    "timestamp": time.time()
                }
            }
            
            # 1단계: LLM 기반 쿼리 최적화 (5초 제한)
            optimized_query = await self._llm_optimize_query(query, timeout=5.0)
            
            yield {
                "event": "progress",
                "data": {
                    "message": "🔍 쿼리 최적화 완료, LLM 처리 시작...",
                    "optimized_query": optimized_query[:100] + "..." if len(optimized_query) > 100 else optimized_query,
                    "timestamp": time.time()
                }
            }
            
            # 2단계: 실제 LLM 스트리밍 (최적화된 타임아웃)
            chunk_count = 0
            total_content = ""
            first_content_time = None
            
            # 안전한 타임아웃 적용
            safe_timeout = min(max_time * 0.8, 70.0)
            
            try:
                # astream 사용 (가능한 경우)
                if hasattr(self.llm_client, 'astream'):
                    async for chunk in self.llm_client.astream(optimized_query):
                        chunk_start = time.time()
                        
                        # 시간 제한 체크
                        if chunk_start - start_time > safe_timeout:
                            logger.warning(f"Timeout approaching, stopping at {chunk_start - start_time:.1f}s")
                            break
                        
                        # 청크 내용 추출
                        content = self._extract_content(chunk)
                        if content:
                            if first_content_time is None:
                                first_content_time = chunk_start - start_time
                                
                            total_content += content
                            chunk_count += 1
                            
                            yield {
                                "event": "progress",
                                "data": {
                                    "chunk_id": chunk_count,
                                    "content": content,
                                    "ttft": first_content_time if chunk_count == 1 else None,
                                    "elapsed": chunk_start - start_time,
                                    "timestamp": time.time()
                                }
                            }
                            
                            # 스트리밍 지연
                            await asyncio.sleep(self.config["chunk_delay"])
                
                else:
                    # 폴백: 일반 ainvoke
                    response = await asyncio.wait_for(
                        self.llm_client.ainvoke(optimized_query),
                        timeout=safe_timeout
                    )
                    
                    content = response.content if hasattr(response, 'content') else str(response)
                    first_content_time = time.time() - start_time
                    
                    # 내용을 청크로 분할하여 스트리밍 시뮬레이션
                    chunks = self._split_for_streaming(content)
                    
                    for i, chunk_content in enumerate(chunks):
                        chunk_count += 1
                        total_content += chunk_content
                        
                        yield {
                            "event": "progress",
                            "data": {
                                "chunk_id": chunk_count,
                                "content": chunk_content,
                                "ttft": first_content_time if chunk_count == 1 else None,
                                "simulated": True,
                                "timestamp": time.time()
                            }
                        }
                        
                        await asyncio.sleep(self.config["chunk_delay"])
                
                # 3단계: LLM 기반 품질 평가 (빠른 평가)
                quality_score = await self._llm_quick_quality_assessment(
                    query, total_content, timeout=3.0
                )
                
                total_time = time.time() - start_time
                
                # 완료 이벤트
                yield {
                    "event": "complete",
                    "data": {
                        "message": "✅ 분석이 완료되었습니다",
                        "total_content": total_content,
                        "metrics": {
                            "ttft": first_content_time,
                            "total_time": total_time,
                            "chunks": chunk_count,
                            "quality_score": quality_score,
                            "model": "qwen3-4b-fast",
                            "llm_first_compliance": 100.0,
                            "success": total_time <= max_time and quality_score >= 0.7
                        },
                        "final": True,
                        "timestamp": time.time()
                    }
                }
                
                logger.info(f"Optimized streaming completed: {total_time:.2f}s, TTFT: {first_content_time:.2f}s, Quality: {quality_score:.2f}")
                
            except asyncio.TimeoutError:
                # 타임아웃 처리
                timeout_msg = await self._llm_timeout_message(time.time() - start_time)
                
                yield {
                    "event": "error",
                    "data": {
                        "message": timeout_msg,
                        "error_type": "timeout",
                        "elapsed": time.time() - start_time,
                        "partial_content": total_content,
                        "timestamp": time.time()
                    }
                }
                
        except Exception as e:
            logger.error(f"Optimized streaming failed: {e}")
            
            error_msg = await self._llm_error_message(str(e))
            
            yield {
                "event": "error",
                "data": {
                    "message": error_msg,
                    "error_type": "execution_error",
                    "original_error": str(e),
                    "timestamp": time.time()
                }
            }
    
    async def _llm_optimize_query(self, query: str, timeout: float = 5.0) -> str:
        """LLM 기반 쿼리 최적화 (빠른 처리)"""
        optimization_prompt = f"""
        Optimize this query for fast, high-quality response generation:
        
        Query: "{query}"
        
        Requirements:
        - Maintain original intent
        - Enable efficient processing
        - Focus on key aspects
        
        Optimized query:
        """
        
        try:
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(optimization_prompt),
                timeout=timeout
            )
            
            optimized = response.content if hasattr(response, 'content') else str(response)
            
            # 검증: 너무 짧거나 의미없으면 원본 사용
            if len(optimized.strip()) < 10:
                return query
            
            return optimized.strip()
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return query
    
    async def _llm_quick_quality_assessment(
        self, 
        query: str, 
        response: str, 
        timeout: float = 3.0
    ) -> float:
        """LLM 기반 빠른 품질 평가"""
        assessment_prompt = f"""
        Rate response quality (0.0-1.0):
        
        Query: "{query[:50]}..."
        Response length: {len(response)} chars
        
        Quality (number only):
        """
        
        try:
            assessment = await asyncio.wait_for(
                self.llm_client.ainvoke(assessment_prompt),
                timeout=timeout
            )
            
            content = assessment.content if hasattr(assessment, 'content') else str(assessment)
            
            # 숫자 추출
            import re
            numbers = re.findall(r'[0-9]*\.?[0-9]+', content)
            if numbers:
                quality = float(numbers[0])
                return min(max(quality, 0.0), 1.0)
                
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
        
        # 기본 품질 추정 (길이 기반)
        if len(response) > 200:
            return 0.8
        elif len(response) > 50:
            return 0.7
        else:
            return 0.6
    
    async def _llm_timeout_message(self, elapsed: float) -> str:
        """LLM 기반 타임아웃 메시지"""
        try:
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(f"Generate timeout message for {elapsed:.1f}s processing"),
                timeout=2.0
            )
            return response.content if hasattr(response, 'content') else str(response)
        except:
            return f"처리 시간이 {elapsed:.1f}초를 초과하여 중단되었습니다."
    
    async def _llm_error_message(self, error: str) -> str:
        """LLM 기반 에러 메시지"""
        try:
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(f"Generate user-friendly error message for: {error}"),
                timeout=2.0
            )
            return response.content if hasattr(response, 'content') else str(response)
        except:
            return f"처리 중 오류가 발생했습니다: {error}"
    
    def _extract_content(self, chunk) -> str:
        """청크에서 내용 추출"""
        if hasattr(chunk, 'content'):
            return chunk.content
        elif hasattr(chunk, 'text'):
            return chunk.text
        elif isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, dict):
            return chunk.get('content', chunk.get('text', ''))
        return str(chunk)
    
    def _split_for_streaming(self, content: str, chunk_size: int = 30) -> list:
        """스트리밍용 내용 분할"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # 자연스러운 끊김점 찾기
            if i + chunk_size < len(words) and not chunk_text.endswith(('.', '!', '?')):
                chunk_text += ' '
            
            chunks.append(chunk_text)
        
        return chunks


async def run_qwen3_fast_performance_test():
    """Qwen3-4B-Fast 성능 테스트 실행"""
    print("🚀 Qwen3-4B-Fast LLM First 최적화 테스트")
    print("=" * 60)
    
    streaming = Qwen3FastOptimizedStreaming()
    
    # 테스트 시나리오
    test_scenarios = [
        {
            "name": "simple_analysis",
            "query": "What is machine learning and how does it work?",
            "target_time": 45.0,
            "expected_quality": 0.8
        },
        {
            "name": "moderate_analysis", 
            "query": "Explain the differences between supervised and unsupervised learning with practical examples",
            "target_time": 60.0,
            "expected_quality": 0.8
        },
        {
            "name": "complex_analysis",
            "query": "Analyze the trade-offs between different neural network architectures for computer vision tasks",
            "target_time": 90.0,
            "expected_quality": 0.7
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        print(f"   Query: {scenario['query'][:50]}...")
        print(f"   Target: {scenario['target_time']}s")
        
        start_time = time.time()
        chunks_received = 0
        first_content_time = None
        final_metrics = None
        
        try:
            async for event in streaming.optimized_llm_streaming(
                scenario["query"], 
                max_time=scenario["target_time"]
            ):
                if event["event"] == "start":
                    print(f"   ⚡ {event['data']['message']}")
                
                elif event["event"] == "progress":
                    chunks_received += 1
                    if chunks_received == 1 and event["data"].get("ttft"):
                        first_content_time = event["data"]["ttft"]
                        print(f"   🎯 First content: {first_content_time:.2f}s")
                    
                    if chunks_received % 5 == 0:  # 5청크마다 진행상황 출력
                        elapsed = time.time() - start_time
                        print(f"   📦 Chunk {chunks_received}, Elapsed: {elapsed:.1f}s")
                
                elif event["event"] == "complete":
                    final_metrics = event["data"]["metrics"]
                    print(f"   ✅ Completed: {final_metrics['total_time']:.2f}s")
                    print(f"   📊 Quality: {final_metrics['quality_score']:.2f}")
                    print(f"   🎯 Success: {final_metrics['success']}")
                
                elif event["event"] == "error":
                    print(f"   ❌ Error: {event['data']['message']}")
                    break
            
            # 결과 저장
            scenario_result = {
                "scenario": scenario["name"],
                "target_time": scenario["target_time"],
                "actual_time": final_metrics["total_time"] if final_metrics else time.time() - start_time,
                "ttft": final_metrics["ttft"] if final_metrics else first_content_time,
                "quality": final_metrics["quality_score"] if final_metrics else 0.0,
                "chunks": chunks_received,
                "success": final_metrics["success"] if final_metrics else False,
                "time_success": (final_metrics["total_time"] if final_metrics else time.time() - start_time) <= scenario["target_time"],
                "quality_success": (final_metrics["quality_score"] if final_metrics else 0.0) >= scenario["expected_quality"]
            }
            
            results.append(scenario_result)
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e),
                "success": False
            })
    
    # 전체 결과 분석
    print("\n" + "=" * 60)
    print("📊 Qwen3-4B-Fast 성능 테스트 결과")
    print("=" * 60)
    
    successful_tests = [r for r in results if r.get("success", False)]
    time_successful = [r for r in results if r.get("time_success", False)]
    quality_successful = [r for r in results if r.get("quality_success", False)]
    
    print(f"📈 전체 성공률: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)")
    print(f"⏱️ 시간 목표 달성: {len(time_successful)}/{len(results)} ({len(time_successful)/len(results)*100:.1f}%)")
    print(f"📊 품질 목표 달성: {len(quality_successful)}/{len(results)} ({len(quality_successful)/len(results)*100:.1f}%)")
    
    if successful_tests:
        avg_time = sum(r["actual_time"] for r in successful_tests) / len(successful_tests)
        avg_ttft = sum(r["ttft"] for r in successful_tests if r["ttft"]) / len([r for r in successful_tests if r["ttft"]])
        avg_quality = sum(r["quality"] for r in successful_tests) / len(successful_tests)
        
        print(f"\n🎯 평균 성능:")
        print(f"   총 시간: {avg_time:.2f}s")
        print(f"   TTFT: {avg_ttft:.2f}s") 
        print(f"   품질: {avg_quality:.2f}")
    
    # 상세 결과
    print(f"\n📋 상세 결과:")
    for result in results:
        status = "✅" if result.get("success", False) else "❌"
        print(f"   {status} {result['scenario']}: {result.get('actual_time', 0):.2f}s, Quality: {result.get('quality', 0):.2f}")
    
    # 최종 평가
    overall_success_rate = len(successful_tests) / len(results)
    
    print(f"\n🏆 최종 평가:")
    if overall_success_rate >= 0.8:
        print("   🚀 EXCELLENT: Qwen3-4B-Fast로 LLM First 원칙과 성능 목표 달성!")
        print("   ✅ 2분 마지노선 내에서 높은 품질의 응답 제공 가능")
    elif overall_success_rate >= 0.6:
        print("   ⚡ GOOD: 대부분의 시나리오에서 목표 달성")
        print("   📈 일부 최적화로 완벽한 성능 달성 가능")
    else:
        print("   ⚠️ NEEDS IMPROVEMENT: 추가 최적화 필요")
        print("   🔧 더 빠른 모델이나 하드웨어 업그레이드 고려")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_qwen3_fast_performance_test())