"""
Hybrid Meta-Reasoning Engine - 적응적 품질-성능 균형 시스템

Ultra-Deep 분석 기반 하이브리드 접근:
- 쿼리 복잡도에 따른 적응적 엔진 선택
- 시간 제한 내 최고 품질 보장
- 스트리밍 + 백그라운드 고품질 분석
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from .llm_factory import LLMFactory
from .meta_reasoning_engine_fast import FastMetaReasoningEngine
from .optimized_meta_reasoning_engine import OptimizedMetaReasoningEngine

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """쿼리 복잡도 레벨"""
    SIMPLE = "simple"      # Fast 엔진 (10-20초)
    MODERATE = "moderate"  # 최적화 엔진 (30-60초)  
    COMPLEX = "complex"    # 전체 최적화 (60-90초)


class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"          # 신속 응답 (품질 70%)
    BALANCED = "balanced"  # 균형점 (품질 80%)
    PREMIUM = "premium"    # 최고 품질 (품질 90%)


class HybridMetaReasoningEngine:
    """
    적응적 품질-성능 균형 메타 추론 엔진
    
    핵심 전략:
    1. 쿼리 복잡도 자동 분석
    2. 시간 제약 내 최적 엔진 선택  
    3. 스트리밍 + 백그라운드 개선
    4. 적응적 품질 레벨 조정
    """
    
    def __init__(self, 
                 default_time_limit: int = 60,
                 default_quality: QualityLevel = QualityLevel.BALANCED):
        """HybridMetaReasoningEngine 초기화"""
        
        self.llm_client = LLMFactory.create_llm()
        self.default_time_limit = default_time_limit
        self.default_quality = default_quality
        
        # 다중 엔진 초기화
        self.fast_engine = FastMetaReasoningEngine()
        self.optimized_engine = OptimizedMetaReasoningEngine()
        
        # 성능 추적
        self.performance_history = []
        
        logger.info(f"HybridMetaReasoningEngine initialized with {default_time_limit}s limit")
    
    async def analyze_request(self, 
                            query: str, 
                            data: Any, 
                            context: Dict,
                            time_limit: Optional[int] = None,
                            quality_level: Optional[QualityLevel] = None) -> Dict:
        """
        적응적 메타 추론 분석
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트
            time_limit: 시간 제한 (초)
            quality_level: 원하는 품질 레벨
            
        Returns:
            최적화된 메타 추론 결과
        """
        start_time = time.time()
        time_limit = time_limit or self.default_time_limit
        quality_level = quality_level or self.default_quality
        
        logger.info(f"Hybrid analysis: {query[:50]}... (limit: {time_limit}s, quality: {quality_level.value})")
        
        try:
            # 1. 쿼리 복잡도 빠른 분석 (5초 이내)
            complexity = await self._analyze_query_complexity(query, data)
            
            # 2. 최적 엔진 전략 결정
            strategy = self._determine_engine_strategy(
                complexity, time_limit, quality_level
            )
            
            # 3. 전략에 따른 실행
            if strategy['approach'] == 'fast_only':
                return await self._execute_fast_analysis(query, data, context)
                
            elif strategy['approach'] == 'optimized_with_fallback':
                return await self._execute_optimized_with_fallback(
                    query, data, context, strategy['allocated_time']
                )
                
            elif strategy['approach'] == 'streaming_hybrid':
                return await self._execute_streaming_hybrid(
                    query, data, context, time_limit
                )
                
            else:  # 'adaptive_quality'
                return await self._execute_adaptive_quality(
                    query, data, context, time_limit, quality_level
                )
                
        except Exception as e:
            logger.error(f"Hybrid reasoning failed: {e}")
            # 폴백: Fast 엔진으로 기본 응답
            return await self._execute_fallback_analysis(query, data, context)
    
    async def _analyze_query_complexity(self, query: str, data: Any) -> QueryComplexity:
        """쿼리 복잡도 빠른 분석 (5초 내)"""
        
        # 휴리스틱 기반 빠른 분류
        complexity_score = 0
        
        # 쿼리 길이
        if len(query) > 100:
            complexity_score += 1
        
        # 키워드 기반
        complex_keywords = ['분석', '인사이트', '패턴', '트렌드', '예측', '최적화', '전략']
        complexity_score += sum(1 for word in complex_keywords if word in query)
        
        # 데이터 복잡도
        try:
            if hasattr(data, '__len__') and len(data) > 100:
                complexity_score += 1
            elif hasattr(data, 'columns'):  # DataFrame
                complexity_score += 2
        except:
            pass
        
        # 질문의 추상도 (간단한 LLM 호출)
        abstractness_prompt = f"Query complexity (1-3): '{query[:100]}'"
        try:
            response = await asyncio.wait_for(
                self.llm_client.ainvoke(abstractness_prompt), 
                timeout=5.0
            )
            content = response.content if hasattr(response, 'content') else str(response)
            if '3' in content:
                complexity_score += 2
            elif '2' in content:
                complexity_score += 1
        except:
            pass  # 타임아웃 시 휴리스틱으로만 판단
        
        # 복잡도 결정
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 5:
            return QueryComplexity.MODERATE  
        else:
            return QueryComplexity.COMPLEX
    
    def _determine_engine_strategy(self, 
                                 complexity: QueryComplexity,
                                 time_limit: int,
                                 quality_level: QualityLevel) -> Dict:
        """최적 엔진 전략 결정"""
        
        strategy = {
            'approach': 'fast_only',
            'allocated_time': time_limit,
            'expected_quality': 0.7,
            'reasoning': 'default'
        }
        
        # 시간 제약이 엄격한 경우 (30초 미만)
        if time_limit < 30:
            strategy.update({
                'approach': 'fast_only',
                'expected_quality': 0.65,
                'reasoning': 'tight_time_constraint'
            })
            
        # 단순 쿼리 + 빠른 품질 요구
        elif complexity == QueryComplexity.SIMPLE and quality_level == QualityLevel.FAST:
            strategy.update({
                'approach': 'fast_only', 
                'expected_quality': 0.75,
                'reasoning': 'simple_query_fast_quality'
            })
            
        # 중간 복잡도 + 충분한 시간
        elif complexity == QueryComplexity.MODERATE and time_limit >= 60:
            strategy.update({
                'approach': 'optimized_with_fallback',
                'allocated_time': min(time_limit - 10, 90),  # 10초 버퍼
                'expected_quality': 0.80,
                'reasoning': 'moderate_complexity_good_time'
            })
            
        # 복잡한 쿼리 + 프리미엄 품질 요구
        elif complexity == QueryComplexity.COMPLEX and quality_level == QualityLevel.PREMIUM:
            strategy.update({
                'approach': 'streaming_hybrid',
                'expected_quality': 0.85,
                'reasoning': 'complex_query_premium_quality'
            })
            
        # 일반적인 경우 - 적응적 품질
        else:
            strategy.update({
                'approach': 'adaptive_quality',
                'expected_quality': 0.75,
                'reasoning': 'general_adaptive_approach'
            })
        
        logger.info(f"Strategy: {strategy['approach']} (complexity: {complexity.value}, "
                   f"time: {time_limit}s, quality: {quality_level.value})")
        
        return strategy
    
    async def _execute_fast_analysis(self, query: str, data: Any, context: Dict) -> Dict:
        """Fast 엔진 실행"""
        logger.info("Executing fast analysis")
        
        result = await self.fast_engine.analyze_request(query, data, context)
        
        # 메타데이터 추가
        result.update({
            'engine_used': 'fast',
            'execution_time': 20,  # 추정값
            'quality_level': 'fast',
            'trade_off': 'optimized_for_speed'
        })
        
        return result
    
    async def _execute_optimized_with_fallback(self, 
                                             query: str, 
                                             data: Any, 
                                             context: Dict,
                                             allocated_time: int) -> Dict:
        """최적화 엔진 + 폴백 실행"""
        logger.info(f"Executing optimized analysis with {allocated_time}s timeout")
        
        try:
            result = await asyncio.wait_for(
                self.optimized_engine.analyze_request(query, data, context),
                timeout=allocated_time
            )
            
            result.update({
                'engine_used': 'optimized',
                'quality_level': 'balanced',
                'trade_off': 'quality_performance_balance'
            })
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Optimized engine timeout ({allocated_time}s), falling back to fast")
            
            # Fast 엔진으로 폴백
            result = await self.fast_engine.analyze_request(query, data, context)
            result.update({
                'engine_used': 'fast_fallback',
                'quality_level': 'fast',
                'trade_off': 'timeout_fallback',
                'original_timeout': allocated_time
            })
            
            return result
    
    async def _execute_streaming_hybrid(self, 
                                      query: str, 
                                      data: Any, 
                                      context: Dict,
                                      time_limit: int) -> Dict:
        """스트리밍 + 백그라운드 고품질 분석"""
        logger.info("Executing streaming hybrid analysis")
        
        # 1. 즉시 Fast 결과 제공
        fast_result = await self.fast_engine.analyze_request(query, data, context)
        
        # 2. 백그라운드에서 고품질 분석 시작 (non-blocking)
        background_task = asyncio.create_task(
            self._background_quality_analysis(query, data, context, time_limit - 25)
        )
        
        # 3. Fast 결과에 개선 프로미스 추가
        result = fast_result.copy()
        result.update({
            'engine_used': 'streaming_hybrid',
            'quality_level': 'progressive',
            'trade_off': 'immediate_response_with_background_improvement',
            'background_analysis': 'in_progress',
            'initial_confidence': result.get('confidence_level', 0.7)
        })
        
        # 4. 백그라운드 분석이 시간 내 완료되면 결과 개선
        try:
            enhanced_result = await asyncio.wait_for(background_task, timeout=2.0)
            if enhanced_result and enhanced_result.get('confidence_level', 0) > result.get('confidence_level', 0):
                logger.info("Background analysis completed, upgrading result")
                result.update({
                    'enhanced_analysis': enhanced_result,
                    'background_analysis': 'completed',
                    'quality_improvement': enhanced_result.get('confidence_level', 0) - result.get('confidence_level', 0)
                })
        except asyncio.TimeoutError:
            result['background_analysis'] = 'continuing'
        
        return result
    
    async def _execute_adaptive_quality(self, 
                                       query: str, 
                                       data: Any, 
                                       context: Dict,
                                       time_limit: int,
                                       quality_level: QualityLevel) -> Dict:
        """적응적 품질 분석"""
        logger.info(f"Executing adaptive quality analysis (limit: {time_limit}s)")
        
        # 시간에 따른 품질 조정
        if time_limit < 45:
            return await self._execute_fast_analysis(query, data, context)
        elif time_limit < 90:
            return await self._execute_optimized_with_fallback(query, data, context, time_limit - 10)
        else:
            # 충분한 시간이 있으면 스트리밍 하이브리드
            return await self._execute_streaming_hybrid(query, data, context, time_limit)
    
    async def _background_quality_analysis(self, 
                                         query: str, 
                                         data: Any, 
                                         context: Dict,
                                         time_limit: int) -> Optional[Dict]:
        """백그라운드 고품질 분석"""
        try:
            result = await asyncio.wait_for(
                self.optimized_engine.analyze_request(query, data, context),
                timeout=time_limit
            )
            return result
        except:
            return None
    
    async def _execute_fallback_analysis(self, query: str, data: Any, context: Dict) -> Dict:
        """최종 폴백 분석"""
        logger.warning("Executing emergency fallback analysis")
        
        try:
            return await self.fast_engine.analyze_request(query, data, context)
        except:
            # 최종 폴백: 기본 응답
            return {
                'analysis': {'intent': 'analysis', 'domain': 'general'},
                'strategy': {'depth': 'shallow', 'user_level': 'beginner'},
                'confidence_level': 0.5,
                'user_profile': {'expertise': 'beginner'},
                'domain_context': 'general',
                'data_characteristics': str(type(data).__name__),
                'engine_used': 'emergency_fallback',
                'quality_level': 'minimal'
            }
    
    async def get_recommendation(self, query: str, data: Any, available_time: int) -> Dict:
        """사용자를 위한 추천 설정"""
        
        complexity = await self._analyze_query_complexity(query, data)
        
        recommendations = {
            'complexity': complexity.value,
            'recommended_time': 30,
            'recommended_quality': QualityLevel.BALANCED.value,
            'expected_confidence': 0.75
        }
        
        if complexity == QueryComplexity.SIMPLE:
            recommendations.update({
                'recommended_time': 20,
                'recommended_quality': QualityLevel.FAST.value,
                'expected_confidence': 0.70,
                'message': '간단한 쿼리입니다. 빠른 응답이 가능합니다.'
            })
        elif complexity == QueryComplexity.COMPLEX:
            recommendations.update({
                'recommended_time': 80,
                'recommended_quality': QualityLevel.PREMIUM.value,
                'expected_confidence': 0.85,
                'message': '복잡한 쿼리입니다. 충분한 시간을 주면 더 나은 결과를 얻을 수 있습니다.'
            })
        else:
            recommendations['message'] = '표준적인 분석이 필요한 쿼리입니다.'
        
        return recommendations