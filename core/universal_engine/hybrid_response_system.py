#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 하이브리드 응답 시스템
LLM First 원칙을 유지하면서 실용적 성능을 달성하는 지능형 응답 시스템

핵심 전략:
1. 계층화된 응답 (즉시 → 빠른 LLM → 상세 LLM)
2. 지능형 프리컴퓨팅 (자주 묻는 질문 미리 처리)
3. 적응적 품질 조정 (사용자 요구에 따른 동적 조정)
4. 백그라운드 개선 (점진적 품질 향상)
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
import re
from pathlib import Path

from .llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResponseMetadata:
    """응답 메타데이터"""
    response_type: str  # immediate, quick_llm, detailed_llm, fallback
    generation_time: float
    quality_estimate: float
    source: str
    cached: bool = False
    background_improvement: bool = False

class PatternMatcher:
    """패턴 기반 즉시 응답 매처"""
    
    def __init__(self):
        """패턴 매처 초기화"""
        self.patterns = {
            # 기본 정의 질문
            r"what\s+is\s+(ai|artificial\s+intelligence)": {
                "template": "AI (Artificial Intelligence) refers to computer systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, and problem-solving.",
                "quality": 0.75,
                "expand_prompt": "Provide a comprehensive explanation of artificial intelligence, including its applications and importance"
            },
            r"what\s+is\s+(ml|machine\s+learning)": {
                "template": "Machine Learning is a subset of AI that enables computers to learn and improve from data without being explicitly programmed for each task.",
                "quality": 0.75,
                "expand_prompt": "Explain machine learning in detail, including types, algorithms, and real-world applications"
            },
            r"what\s+is\s+(data\s+science|data\s+analysis)": {
                "template": "Data science combines statistics, programming, and domain expertise to extract insights from data for decision-making.",
                "quality": 0.7,
                "expand_prompt": "Provide a detailed overview of data science, methodologies, and career aspects"
            },
            
            # 비교 질문
            r"(difference|compare)\s+between\s+(ai|artificial\s+intelligence)\s+and\s+(ml|machine\s+learning)": {
                "template": "AI is the broader concept of intelligent machines, while ML is a specific approach to achieve AI through data-driven learning algorithms.",
                "quality": 0.8,
                "expand_prompt": "Compare and contrast AI and ML with examples, applications, and technical differences"
            },
            
            # 방법 질문
            r"how\s+to\s+(optimize|improve)\s+(performance|speed)": {
                "template": "Performance optimization typically involves: 1) Identifying bottlenecks, 2) Optimizing algorithms and data structures, 3) Using caching and parallel processing, 4) Hardware optimization.",
                "quality": 0.7,
                "expand_prompt": "Provide comprehensive performance optimization strategies with practical examples and implementation details"
            },
            
            # 기술적 질문
            r"(explain|what\s+is)\s+(algorithm|neural\s+network|deep\s+learning)": {
                "template": "This is a complex technical topic that requires detailed explanation. Let me provide a comprehensive response.",
                "quality": 0.5,
                "expand_prompt": "Provide detailed technical explanation with examples, mathematical foundations, and practical applications"
            }
        }
        
        logger.info(f"PatternMatcher initialized with {len(self.patterns)} patterns")
    
    def match_pattern(self, query: str) -> Optional[Dict[str, Any]]:
        """쿼리에 대한 패턴 매칭"""
        query_lower = query.lower().strip()
        
        for pattern, response_data in self.patterns.items():
            if re.search(pattern, query_lower):
                return {
                    "template_response": response_data["template"],
                    "quality_estimate": response_data["quality"],
                    "expand_prompt": response_data["expand_prompt"],
                    "pattern_matched": pattern
                }
        
        return None

class PrecomputedResponseManager:
    """프리컴퓨팅된 응답 관리자"""
    
    def __init__(self):
        """프리컴퓨팅 매니저 초기화"""
        self.precomputed_responses = {}
        self.common_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What is data science?",
            "How to optimize performance?",
            "What is the difference between AI and ML?",
            "Explain neural networks",
            "What is deep learning?",
            "How to implement machine learning?",
            "What are the applications of AI?",
            "Explain data analysis techniques"
        ]
        self.generation_in_progress = set()
        
        logger.info(f"PrecomputedResponseManager initialized with {len(self.common_queries)} common queries")
    
    async def initialize_precomputed_responses(self):
        """백그라운드에서 프리컴퓨팅된 응답 생성"""
        logger.info("Starting background precomputation of common responses...")
        
        llm_client = LLMFactory.create_llm()
        
        for query in self.common_queries:
            if query not in self.precomputed_responses and query not in self.generation_in_progress:
                self.generation_in_progress.add(query)
                
                try:
                    # 백그라운드 태스크로 실행
                    asyncio.create_task(self._generate_precomputed_response(llm_client, query))
                    
                    # CPU 과부하 방지를 위한 작은 지연
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to start precomputation for '{query}': {e}")
                    self.generation_in_progress.discard(query)
    
    async def _generate_precomputed_response(self, llm_client, query: str):
        """단일 쿼리에 대한 프리컴퓨팅된 응답 생성"""
        try:
            start_time = time.time()
            
            # 타임아웃을 길게 설정 (백그라운드 작업이므로)
            response = await asyncio.wait_for(
                llm_client.ainvoke(query),
                timeout=30.0
            )
            
            generation_time = time.time() - start_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # 품질 추정 (간단한 휴리스틱)
            quality_estimate = min(len(response_content) / 200, 1.0)  # 길이 기반 품질 추정
            
            self.precomputed_responses[query] = {
                'response': response_content,
                'generated_at': time.time(),
                'generation_time': generation_time,
                'quality_estimate': quality_estimate,
                'usage_count': 0
            }
            
            logger.info(f"Precomputed response for '{query[:50]}...' in {generation_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"Failed to precompute response for '{query}': {e}")
        
        finally:
            self.generation_in_progress.discard(query)
    
    def get_precomputed_response(self, query: str) -> Optional[Dict[str, Any]]:
        """프리컴퓨팅된 응답 검색"""
        # 정확한 매칭
        if query in self.precomputed_responses:
            response_data = self.precomputed_responses[query]
            response_data['usage_count'] += 1
            return response_data
        
        # 유사한 쿼리 검색 (간단한 키워드 매칭)
        query_lower = query.lower()
        for precomputed_query, response_data in self.precomputed_responses.items():
            precomputed_lower = precomputed_query.lower()
            
            # 키워드 겹치는 정도 확인
            query_words = set(query_lower.split())
            precomputed_words = set(precomputed_lower.split())
            
            overlap = len(query_words.intersection(precomputed_words))
            if overlap >= 2 and overlap / len(query_words) >= 0.5:
                response_data['usage_count'] += 1
                return response_data
        
        return None

class HybridResponseSystem:
    """하이브리드 응답 시스템"""
    
    def __init__(self):
        """하이브리드 시스템 초기화"""
        self.pattern_matcher = PatternMatcher()
        self.precomputed_manager = PrecomputedResponseManager()
        self.llm_client = LLMFactory.create_llm()
        
        # 성능 메트릭
        self.metrics = {
            "total_requests": 0,
            "immediate_responses": 0,
            "quick_llm_responses": 0,
            "detailed_llm_responses": 0,
            "fallback_responses": 0,
            "background_improvements": 0
        }
        
        # 백그라운드 개선 큐
        self.improvement_queue = asyncio.Queue()
        
        logger.info("HybridResponseSystem initialized")
    
    async def initialize(self):
        """시스템 초기화"""
        logger.info("Initializing HybridResponseSystem...")
        
        # 프리컴퓨팅된 응답 초기화 (백그라운드)
        asyncio.create_task(self.precomputed_manager.initialize_precomputed_responses())
        
        # 백그라운드 개선 워커 시작
        asyncio.create_task(self._background_improvement_worker())
        
        logger.info("HybridResponseSystem initialization completed")
    
    async def get_response(
        self, 
        query: str, 
        max_time: float = 8.0,
        quality_preference: str = "balanced"  # fast, balanced, detailed
    ) -> Dict[str, Any]:
        """하이브리드 응답 생성"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        logger.info(f"Processing query: '{query[:50]}...' with {quality_preference} quality preference")
        
        try:
            # 1단계: 즉시 응답 시도 (패턴 매칭)
            immediate_response = await self._try_immediate_response(query)
            if immediate_response and quality_preference == "fast":
                return self._create_response_result(
                    immediate_response["response"],
                    time.time() - start_time,
                    "immediate",
                    immediate_response["metadata"]
                )
            
            # 2단계: 프리컴퓨팅된 응답 확인
            precomputed_response = await self._try_precomputed_response(query)
            if precomputed_response:
                # 백그라운드에서 개선된 응답 생성 시작
                if quality_preference in ["balanced", "detailed"]:
                    asyncio.create_task(self._queue_background_improvement(query, precomputed_response))
                
                return self._create_response_result(
                    precomputed_response["response"],
                    time.time() - start_time,
                    "precomputed",
                    ResponseMetadata(
                        response_type="precomputed",
                        generation_time=precomputed_response["generation_time"],
                        quality_estimate=precomputed_response["quality_estimate"],
                        source="precomputed_cache",
                        cached=True
                    )
                )
            
            # 3단계: LLM 응답 (품질 선호도에 따라)
            if quality_preference == "fast":
                # 빠른 LLM 응답
                llm_response = await self._try_quick_llm_response(query, max_time / 2)
            elif quality_preference == "balanced":
                # 균형 잡힌 LLM 응답
                llm_response = await self._try_balanced_llm_response(query, max_time)
            else:  # detailed
                # 상세한 LLM 응답
                llm_response = await self._try_detailed_llm_response(query, max_time)
            
            if llm_response:
                return llm_response
            
            # 4단계: 폴백 응답
            return await self._get_fallback_response(query, time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error in hybrid response generation: {e}")
            return await self._get_fallback_response(query, time.time() - start_time, str(e))
    
    async def _try_immediate_response(self, query: str) -> Optional[Dict[str, Any]]:
        """즉시 응답 시도"""
        pattern_match = self.pattern_matcher.match_pattern(query)
        if pattern_match:
            self.metrics["immediate_responses"] += 1
            
            # 백그라운드에서 더 상세한 응답 준비
            asyncio.create_task(self._queue_background_improvement(
                query, 
                {"response": pattern_match["template_response"], "expand_prompt": pattern_match["expand_prompt"]}
            ))
            
            return {
                "response": pattern_match["template_response"],
                "metadata": ResponseMetadata(
                    response_type="immediate",
                    generation_time=0.001,  # 즉시 응답
                    quality_estimate=pattern_match["quality_estimate"],
                    source="pattern_matching",
                    background_improvement=True
                )
            }
        
        return None
    
    async def _try_precomputed_response(self, query: str) -> Optional[Dict[str, Any]]:
        """프리컴퓨팅된 응답 시도"""
        precomputed = self.precomputed_manager.get_precomputed_response(query)
        if precomputed:
            self.metrics["quick_llm_responses"] += 1
            return precomputed
        
        return None
    
    async def _try_quick_llm_response(self, query: str, max_time: float) -> Optional[Dict[str, Any]]:
        """빠른 LLM 응답 시도"""
        try:
            # 간소화된 프롬프트
            quick_prompt = f"Briefly explain: {query}"
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(quick_prompt),
                timeout=max_time
            )
            
            generation_time = time.time() - start_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            self.metrics["quick_llm_responses"] += 1
            
            return self._create_response_result(
                response_content,
                generation_time,
                "quick_llm",
                ResponseMetadata(
                    response_type="quick_llm",
                    generation_time=generation_time,
                    quality_estimate=0.7,  # 추정 품질
                    source="quick_llm_call"
                )
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Quick LLM response timeout after {max_time}s")
            return None
        except Exception as e:
            logger.error(f"Quick LLM response failed: {e}")
            return None
    
    async def _try_balanced_llm_response(self, query: str, max_time: float) -> Optional[Dict[str, Any]]:
        """균형 잡힌 LLM 응답 시도"""
        try:
            start_time = time.time()
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(query),
                timeout=max_time
            )
            
            generation_time = time.time() - start_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            self.metrics["detailed_llm_responses"] += 1
            
            return self._create_response_result(
                response_content,
                generation_time,
                "balanced_llm",
                ResponseMetadata(
                    response_type="balanced_llm",
                    generation_time=generation_time,
                    quality_estimate=0.85,
                    source="balanced_llm_call"
                )
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Balanced LLM response timeout after {max_time}s")
            return None
        except Exception as e:
            logger.error(f"Balanced LLM response failed: {e}")
            return None
    
    async def _try_detailed_llm_response(self, query: str, max_time: float) -> Optional[Dict[str, Any]]:
        """상세한 LLM 응답 시도"""
        try:
            # 상세한 프롬프트
            detailed_prompt = f"Provide a comprehensive and detailed explanation for: {query}. Include examples, technical details, and practical applications where relevant."
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(detailed_prompt),
                timeout=max_time
            )
            
            generation_time = time.time() - start_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            self.metrics["detailed_llm_responses"] += 1
            
            return self._create_response_result(
                response_content,
                generation_time,
                "detailed_llm",
                ResponseMetadata(
                    response_type="detailed_llm",
                    generation_time=generation_time,
                    quality_estimate=0.95,
                    source="detailed_llm_call"
                )
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Detailed LLM response timeout after {max_time}s")
            return None
        except Exception as e:
            logger.error(f"Detailed LLM response failed: {e}")
            return None
    
    async def _get_fallback_response(self, query: str, elapsed_time: float, error: Optional[str] = None) -> Dict[str, Any]:
        """폴백 응답 생성"""
        self.metrics["fallback_responses"] += 1
        
        fallback_message = f"I understand you're asking about '{query}'. While I'm processing a detailed response, here's what I can tell you immediately: This appears to be a complex topic that requires careful analysis. I'm working on providing you with a comprehensive answer."
        
        if error:
            fallback_message += f" (Technical note: {error})"
        
        # 백그라운드에서 실제 응답 생성 시작
        asyncio.create_task(self._queue_background_improvement(query, {"response": fallback_message}))
        
        return self._create_response_result(
            fallback_message,
            elapsed_time,
            "fallback",
            ResponseMetadata(
                response_type="fallback",
                generation_time=elapsed_time,
                quality_estimate=0.3,
                source="fallback_system",
                background_improvement=True
            )
        )
    
    async def _queue_background_improvement(self, query: str, current_response: Dict[str, Any]):
        """백그라운드 개선 큐에 추가"""
        try:
            await self.improvement_queue.put({
                "query": query,
                "current_response": current_response,
                "queued_at": time.time()
            })
        except Exception as e:
            logger.warning(f"Failed to queue background improvement: {e}")
    
    async def _background_improvement_worker(self):
        """백그라운드 개선 워커"""
        logger.info("Background improvement worker started")
        
        while True:
            try:
                # 개선 작업 대기
                improvement_task = await self.improvement_queue.get()
                
                query = improvement_task["query"]
                current_response = improvement_task["current_response"]
                
                logger.info(f"Processing background improvement for: '{query[:50]}...'")
                
                # 더 상세한 프롬프트로 개선된 응답 생성
                improved_prompt = current_response.get("expand_prompt", 
                    f"Provide a comprehensive, detailed, and high-quality response to: {query}")
                
                try:
                    start_time = time.time()
                    response = await asyncio.wait_for(
                        self.llm_client.ainvoke(improved_prompt),
                        timeout=30.0  # 백그라운드이므로 더 긴 타임아웃
                    )
                    
                    generation_time = time.time() - start_time
                    response_content = response.content if hasattr(response, 'content') else str(response)
                    
                    # 개선된 응답을 프리컴퓨팅 캐시에 저장
                    self.precomputed_manager.precomputed_responses[query] = {
                        'response': response_content,
                        'generated_at': time.time(),
                        'generation_time': generation_time,
                        'quality_estimate': 0.9,  # 개선된 응답의 높은 품질
                        'usage_count': 0,
                        'background_improved': True
                    }
                    
                    self.metrics["background_improvements"] += 1
                    logger.info(f"Background improvement completed for '{query[:50]}...' in {generation_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"Background improvement failed for '{query}': {e}")
                
                # 다음 작업 전 잠시 대기 (시스템 부하 방지)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Background improvement worker error: {e}")
                await asyncio.sleep(5.0)  # 에러 발생시 더 긴 대기
    
    def _create_response_result(
        self, 
        response: str, 
        execution_time: float, 
        response_type: str,
        metadata: ResponseMetadata
    ) -> Dict[str, Any]:
        """응답 결과 생성"""
        return {
            "response": response,
            "execution_time": execution_time,
            "response_type": response_type,
            "metadata": metadata.__dict__,
            "success": True,
            "hybrid_system": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 반환"""
        total_requests = self.metrics["total_requests"]
        if total_requests == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "immediate_response_rate": self.metrics["immediate_responses"] / total_requests,
            "quick_llm_rate": self.metrics["quick_llm_responses"] / total_requests,
            "detailed_llm_rate": self.metrics["detailed_llm_responses"] / total_requests,
            "fallback_rate": self.metrics["fallback_responses"] / total_requests,
            "precomputed_cache_size": len(self.precomputed_manager.precomputed_responses)
        }

# 전역 하이브리드 시스템 인스턴스
_global_hybrid_system = None

async def get_hybrid_system() -> HybridResponseSystem:
    """전역 하이브리드 시스템 반환"""
    global _global_hybrid_system
    if _global_hybrid_system is None:
        _global_hybrid_system = HybridResponseSystem()
        await _global_hybrid_system.initialize()
    return _global_hybrid_system

async def get_hybrid_response(
    query: str, 
    max_time: float = 8.0,
    quality_preference: str = "balanced"
) -> Dict[str, Any]:
    """편의 함수: 하이브리드 응답 생성"""
    hybrid_system = await get_hybrid_system()
    return await hybrid_system.get_response(query, max_time, quality_preference)