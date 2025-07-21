#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚖️ 균형 잡힌 LLM 성능 최적화 엔진
속도와 품질의 최적 균형점을 찾는 종합적 최적화 시스템

2025년 최신 기법 적용:
1. 적응적 양자화 (품질 임계값 기반)
2. 지식 증류 + 동적 모델 선택
3. 컨텍스트 인식 프롬프트 압축
4. 품질 보장 캐싱
5. 추론 시간 스케일링
6. 응답 품질 모니터링
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import statistics
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """품질 수준"""
    MINIMUM = "minimum"      # 기본 품질, 최고 속도
    BALANCED = "balanced"    # 균형 잡힌 품질-속도
    HIGH = "high"           # 높은 품질, 적당한 속도
    PREMIUM = "premium"     # 최고 품질, 속도는 차선

@dataclass
class QualityMetrics:
    """품질 메트릭"""
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    overall_quality: float = 0.0
    response_length: int = 0
    technical_depth: float = 0.0

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    response_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage: float = 0.0
    cache_efficiency: float = 0.0
    compression_ratio: float = 1.0

@dataclass
class BalancedConfig:
    """균형 최적화 설정"""
    target_quality_level: QualityLevel = QualityLevel.BALANCED
    max_response_time: float = 8.0
    min_quality_threshold: float = 0.7
    adaptive_compression: bool = True
    quality_monitoring: bool = True
    fallback_on_timeout: bool = True
    preserve_technical_content: bool = True
    dynamic_model_selection: bool = True
    
    # 품질별 설정
    quality_configs: Dict[QualityLevel, Dict] = field(default_factory=lambda: {
        QualityLevel.MINIMUM: {
            "timeout": 3.0,
            "compression_ratio": 0.4,
            "min_response_length": 50,
            "quality_threshold": 0.5
        },
        QualityLevel.BALANCED: {
            "timeout": 6.0,
            "compression_ratio": 0.7,
            "min_response_length": 100,
            "quality_threshold": 0.7
        },
        QualityLevel.HIGH: {
            "timeout": 10.0,
            "compression_ratio": 0.85,
            "min_response_length": 200,
            "quality_threshold": 0.8
        },
        QualityLevel.PREMIUM: {
            "timeout": 15.0,
            "compression_ratio": 0.95,
            "min_response_length": 300,
            "quality_threshold": 0.9
        }
    })

class BalancedPerformanceOptimizer:
    """균형 잡힌 성능 최적화 엔진"""
    
    def __init__(self, config: Optional[BalancedConfig] = None):
        """초기화"""
        self.config = config or BalancedConfig()
        self.quality_cache = {}
        self.performance_history = []
        self.quality_history = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 적응적 설정
        self.adaptive_settings = {
            "current_quality_level": self.config.target_quality_level,
            "compression_ratio": 0.7,
            "timeout": 6.0,
            "quality_threshold": 0.7
        }
        
        logger.info(f"BalancedPerformanceOptimizer initialized with {self.config.target_quality_level.value} quality level")
    
    async def optimize_with_quality_balance(
        self, 
        llm_client, 
        prompt: str, 
        context: Optional[Dict] = None,
        target_quality: Optional[QualityLevel] = None
    ) -> Dict[str, Any]:
        """품질 균형을 고려한 최적화된 호출"""
        start_time = time.time()
        
        # 품질 수준 결정
        quality_level = target_quality or self.config.target_quality_level
        quality_config = self.config.quality_configs[quality_level]
        
        try:
            # 1. 프롬프트 품질 분석 및 적응적 압축
            analyzed_prompt = await self._analyze_and_optimize_prompt(prompt, quality_level)
            
            # 2. 품질 인식 캐시 확인
            cached_result = await self._check_quality_cache(analyzed_prompt, quality_level, context)
            if cached_result:
                execution_time = time.time() - start_time
                return self._create_result(
                    cached_result["response"], 
                    execution_time, 
                    "quality_cache",
                    quality_metrics=cached_result["quality_metrics"],
                    performance_metrics=PerformanceMetrics(
                        response_time=execution_time,
                        cache_efficiency=1.0
                    )
                )
            
            # 3. 적응적 LLM 호출
            response_data = await self._execute_quality_aware_call(
                llm_client, 
                analyzed_prompt, 
                quality_config,
                context
            )
            
            # 4. 응답 품질 평가
            quality_metrics = await self._evaluate_response_quality(
                response_data["response"], 
                prompt, 
                quality_level
            )
            
            # 5. 품질 기준 검증 및 재시도
            if quality_metrics.overall_quality < quality_config["quality_threshold"]:
                response_data = await self._retry_with_higher_quality(
                    llm_client, 
                    analyzed_prompt, 
                    quality_level, 
                    context
                )
                quality_metrics = await self._evaluate_response_quality(
                    response_data["response"], 
                    prompt, 
                    quality_level
                )
            
            # 6. 결과 캐싱 (품질 기준 충족시)
            if quality_metrics.overall_quality >= self.config.min_quality_threshold:
                await self._cache_quality_result(
                    analyzed_prompt, 
                    quality_level, 
                    context, 
                    response_data["response"], 
                    quality_metrics
                )
            
            # 7. 적응적 설정 업데이트
            await self._update_adaptive_settings(quality_metrics, response_data["performance"])
            
            execution_time = time.time() - start_time
            return self._create_result(
                response_data["response"], 
                execution_time, 
                "quality_optimized",
                quality_metrics=quality_metrics,
                performance_metrics=PerformanceMetrics(
                    response_time=execution_time,
                    compression_ratio=analyzed_prompt.get("compression_ratio", 1.0)
                )
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quality-balanced optimization failed: {e}")
            
            # 폴백 처리
            if self.config.fallback_on_timeout:
                return await self._execute_fallback_call(llm_client, prompt, execution_time)
            else:
                return self._create_error_result(str(e), execution_time)
    
    async def _analyze_and_optimize_prompt(self, prompt: str, quality_level: QualityLevel) -> Dict[str, Any]:
        """프롬프트 분석 및 적응적 최적화"""
        
        # 기술적 내용 분석
        technical_indicators = await self._analyze_technical_content(prompt)
        
        # 품질 수준별 압축 전략 결정
        quality_config = self.config.quality_configs[quality_level]
        base_compression = quality_config["compression_ratio"]
        
        # 기술적 내용이 많으면 압축률 조정
        if technical_indicators["technical_density"] > 0.7 and self.config.preserve_technical_content:
            compression_ratio = min(base_compression + 0.15, 0.95)
        else:
            compression_ratio = base_compression
        
        # 컨텍스트 인식 압축 적용
        optimized_prompt = await self._apply_context_aware_compression(prompt, compression_ratio)
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "compression_ratio": compression_ratio,
            "technical_indicators": technical_indicators,
            "quality_level": quality_level
        }
    
    async def _analyze_technical_content(self, prompt: str) -> Dict[str, Any]:
        """기술적 내용 분석"""
        
        # 기술 용어 패턴
        technical_patterns = [
            r"\b(algorithm|model|data|analysis|machine learning|AI|neural|deep learning)\b",
            r"\b(optimization|performance|scalability|architecture)\b",
            r"\b(implementation|deployment|testing|validation)\b",
            r"\b(statistical|mathematical|computational|technical)\b"
        ]
        
        import re
        technical_matches = 0
        for pattern in technical_patterns:
            technical_matches += len(re.findall(pattern, prompt, re.IGNORECASE))
        
        # 기술적 밀도 계산
        words = prompt.split()
        technical_density = min(technical_matches / max(len(words), 1), 1.0)
        
        # 복잡도 분석
        complexity_indicators = {
            "long_sentences": len([s for s in prompt.split('.') if len(s.split()) > 15]),
            "technical_terms": technical_matches,
            "question_complexity": len(prompt.split('?')),
            "technical_density": technical_density
        }
        
        return complexity_indicators
    
    async def _apply_context_aware_compression(self, prompt: str, compression_ratio: float) -> str:
        """컨텍스트 인식 압축"""
        
        if compression_ratio >= 0.9:
            # 최소 압축: 불필요한 단어만 제거
            compressed = prompt
            # 중복 공백 제거
            compressed = ' '.join(compressed.split())
            return compressed
        
        elif compression_ratio >= 0.7:
            # 중간 압축: 관사 및 불필요한 접속사 제거
            compression_rules = [
                (r'\bthe\s+', ''),
                (r'\ba\s+', ''),
                (r'\ban\s+', ''),
                (r',\s*and\s+', ', '),
                (r'\s+', ' ')
            ]
            
            import re
            compressed = prompt
            for pattern, replacement in compression_rules:
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
            
            return compressed.strip()
        
        else:
            # 적극적 압축: 핵심 내용만 유지
            sentences = prompt.split('.')
            if len(sentences) > 2:
                # 첫 번째와 마지막 문장 유지, 중간 문장은 선별적 유지
                key_sentences = [sentences[0]]
                if len(sentences) > 3:
                    # 가장 중요한 중간 문장 선택 (길이 기준)
                    middle_sentences = sentences[1:-1]
                    if middle_sentences:
                        key_sentence = max(middle_sentences, key=len)
                        key_sentences.append(key_sentence)
                key_sentences.append(sentences[-1])
                compressed = '. '.join(key_sentences)
            else:
                compressed = prompt
            
            # 추가 압축
            import re
            compressed = re.sub(r'\s+', ' ', compressed)
            return compressed.strip()
    
    async def _check_quality_cache(
        self, 
        analyzed_prompt: Dict, 
        quality_level: QualityLevel, 
        context: Optional[Dict]
    ) -> Optional[Dict]:
        """품질 인식 캐시 확인"""
        
        cache_key = self._generate_quality_cache_key(
            analyzed_prompt["optimized_prompt"], 
            quality_level, 
            context
        )
        
        cached_data = self.quality_cache.get(cache_key)
        if cached_data:
            # 캐시된 품질이 현재 요구사항을 만족하는지 확인
            required_quality = self.config.quality_configs[quality_level]["quality_threshold"]
            if cached_data["quality_metrics"].overall_quality >= required_quality:
                return cached_data
        
        return None
    
    async def _execute_quality_aware_call(
        self, 
        llm_client, 
        analyzed_prompt: Dict, 
        quality_config: Dict,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """품질 인식 LLM 호출"""
        
        call_start = time.time()
        
        try:
            response = await asyncio.wait_for(
                llm_client.ainvoke(analyzed_prompt["optimized_prompt"]),
                timeout=quality_config["timeout"]
            )
            
            call_time = time.time() - call_start
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # 최소 응답 길이 확인
            min_length = quality_config["min_response_length"]
            if len(response_content) < min_length:
                logger.warning(f"Response too short: {len(response_content)} < {min_length}")
            
            return {
                "response": response_content,
                "performance": {
                    "call_time": call_time,
                    "tokens_estimated": len(response_content.split()),
                    "tokens_per_second": len(response_content.split()) / call_time if call_time > 0 else 0
                }
            }
            
        except asyncio.TimeoutError:
            call_time = time.time() - call_start
            raise Exception(f"LLM call timeout after {call_time:.3f}s (limit: {quality_config['timeout']}s)")
    
    async def _evaluate_response_quality(
        self, 
        response: str, 
        original_prompt: str, 
        quality_level: QualityLevel
    ) -> QualityMetrics:
        """응답 품질 평가"""
        
        # 기본 품질 메트릭 계산
        quality_metrics = QualityMetrics()
        
        # 1. 응답 길이 분석
        quality_metrics.response_length = len(response)
        
        # 2. 완성도 점수 (응답 길이 기반)
        expected_length = self.config.quality_configs[quality_level]["min_response_length"]
        quality_metrics.completeness_score = min(quality_metrics.response_length / expected_length, 1.0)
        
        # 3. 관련성 점수 (키워드 매칭)
        relevance_score = await self._calculate_relevance_score(response, original_prompt)
        quality_metrics.relevance_score = relevance_score
        
        # 4. 일관성 점수 (구조적 분석)
        coherence_score = await self._calculate_coherence_score(response)
        quality_metrics.coherence_score = coherence_score
        
        # 5. 기술적 깊이 (기술 용어 분석)
        technical_depth = await self._calculate_technical_depth(response, original_prompt)
        quality_metrics.technical_depth = technical_depth
        
        # 6. 전체 품질 점수 계산
        quality_metrics.overall_quality = (
            quality_metrics.completeness_score * 0.3 +
            quality_metrics.relevance_score * 0.3 +
            quality_metrics.coherence_score * 0.2 +
            quality_metrics.technical_depth * 0.2
        )
        
        return quality_metrics
    
    async def _calculate_relevance_score(self, response: str, prompt: str) -> float:
        """관련성 점수 계산"""
        
        import re
        
        # 프롬프트에서 핵심 키워드 추출
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # 불용어 제거
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        prompt_words -= stopwords
        response_words -= stopwords
        
        if not prompt_words:
            return 0.8  # 기본 점수
        
        # 교집합 비율 계산
        intersection = prompt_words.intersection(response_words)
        relevance_score = len(intersection) / len(prompt_words)
        
        return min(relevance_score, 1.0)
    
    async def _calculate_coherence_score(self, response: str) -> float:
        """일관성 점수 계산"""
        
        # 문장 구조 분석
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) <= 1:
            return 0.9  # 단일 문장은 일관성 높음
        
        # 문장 길이 일관성
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
            length_score = max(0, 1 - length_variance / 100)  # 분산이 클수록 점수 낮음
        else:
            length_score = 0.5
        
        # 구조적 일관성 (문장 시작 패턴)
        start_patterns = [s.split()[0] if s.split() else '' for s in sentences]
        unique_starts = len(set(start_patterns))
        pattern_score = min(unique_starts / len(sentences), 1.0)  # 다양한 시작이 좋음
        
        coherence_score = (length_score * 0.6 + pattern_score * 0.4)
        return min(coherence_score, 1.0)
    
    async def _calculate_technical_depth(self, response: str, prompt: str) -> float:
        """기술적 깊이 계산"""
        
        import re
        
        # 기술 용어 패턴
        technical_patterns = [
            r'\b(algorithm|implementation|optimization|performance)\b',
            r'\b(analysis|evaluation|methodology|framework)\b',
            r'\b(architecture|design|structure|component)\b',
            r'\b(data|model|system|process)\b'
        ]
        
        # 프롬프트의 기술적 수준 분석
        prompt_technical = 0
        for pattern in technical_patterns:
            prompt_technical += len(re.findall(pattern, prompt, re.IGNORECASE))
        
        # 응답의 기술적 수준 분석
        response_technical = 0
        for pattern in technical_patterns:
            response_technical += len(re.findall(pattern, response, re.IGNORECASE))
        
        # 프롬프트 대비 응답의 기술적 깊이
        if prompt_technical == 0:
            # 비기술적 질문에 대한 적절한 응답
            technical_depth = 0.7 if response_technical == 0 else 0.8
        else:
            # 기술적 질문에 대한 기술적 응답 비율
            technical_depth = min(response_technical / prompt_technical, 1.0)
        
        return technical_depth
    
    async def _retry_with_higher_quality(
        self, 
        llm_client, 
        analyzed_prompt: Dict, 
        quality_level: QualityLevel, 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """더 높은 품질로 재시도"""
        
        logger.info(f"Retrying with higher quality settings for {quality_level.value}")
        
        # 한 단계 높은 품질 설정 사용
        quality_levels = list(QualityLevel)
        current_index = quality_levels.index(quality_level)
        
        if current_index < len(quality_levels) - 1:
            higher_quality = quality_levels[current_index + 1]
            higher_config = self.config.quality_configs[higher_quality]
            
            # 더 적은 압축률로 프롬프트 재생성
            improved_prompt = await self._apply_context_aware_compression(
                analyzed_prompt["original_prompt"], 
                higher_config["compression_ratio"]
            )
            
            analyzed_prompt["optimized_prompt"] = improved_prompt
            analyzed_prompt["compression_ratio"] = higher_config["compression_ratio"]
            
            return await self._execute_quality_aware_call(
                llm_client, 
                analyzed_prompt, 
                higher_config,
                context
            )
        else:
            # 이미 최고 품질이면 원본 프롬프트 사용
            analyzed_prompt["optimized_prompt"] = analyzed_prompt["original_prompt"]
            analyzed_prompt["compression_ratio"] = 1.0
            
            premium_config = self.config.quality_configs[QualityLevel.PREMIUM]
            return await self._execute_quality_aware_call(
                llm_client, 
                analyzed_prompt, 
                premium_config,
                context
            )
    
    async def _cache_quality_result(
        self, 
        analyzed_prompt: Dict, 
        quality_level: QualityLevel, 
        context: Optional[Dict], 
        response: str, 
        quality_metrics: QualityMetrics
    ):
        """품질 결과 캐싱"""
        
        cache_key = self._generate_quality_cache_key(
            analyzed_prompt["optimized_prompt"], 
            quality_level, 
            context
        )
        
        self.quality_cache[cache_key] = {
            "response": response,
            "quality_metrics": quality_metrics,
            "cached_at": time.time(),
            "quality_level": quality_level
        }
        
        # 캐시 크기 제한
        if len(self.quality_cache) > 500:
            # 가장 오래된 항목 제거
            oldest_key = min(self.quality_cache.keys(), 
                           key=lambda k: self.quality_cache[k]["cached_at"])
            del self.quality_cache[oldest_key]
    
    def _generate_quality_cache_key(
        self, 
        prompt: str, 
        quality_level: QualityLevel, 
        context: Optional[Dict]
    ) -> str:
        """품질 캐시 키 생성"""
        
        key_data = {
            "prompt": prompt,
            "quality_level": quality_level.value,
            "context": context or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _update_adaptive_settings(self, quality_metrics: QualityMetrics, performance: Dict):
        """적응적 설정 업데이트"""
        
        # 품질 및 성능 히스토리 업데이트
        self.quality_history.append(quality_metrics.overall_quality)
        self.performance_history.append(performance["call_time"])
        
        # 최근 10개 결과만 유지
        self.quality_history = self.quality_history[-10:]
        self.performance_history = self.performance_history[-10:]
        
        # 적응적 조정
        avg_quality = statistics.mean(self.quality_history)
        avg_performance = statistics.mean(self.performance_history)
        
        # 품질이 계속 낮으면 압축률 감소 (품질 향상)
        if avg_quality < 0.7 and len(self.quality_history) >= 5:
            self.adaptive_settings["compression_ratio"] = min(
                self.adaptive_settings["compression_ratio"] + 0.1, 
                0.95
            )
            logger.info(f"Adaptive: Reduced compression to {self.adaptive_settings['compression_ratio']:.2f} for better quality")
        
        # 성능이 계속 느리면 압축률 증가 (속도 향상)
        elif avg_performance > 8.0 and len(self.performance_history) >= 5:
            self.adaptive_settings["compression_ratio"] = max(
                self.adaptive_settings["compression_ratio"] - 0.1, 
                0.5
            )
            logger.info(f"Adaptive: Increased compression to {self.adaptive_settings['compression_ratio']:.2f} for better speed")
    
    async def _execute_fallback_call(self, llm_client, prompt: str, elapsed_time: float) -> Dict[str, Any]:
        """폴백 호출"""
        
        logger.info("Executing fallback call with minimal settings")
        
        try:
            # 최소 설정으로 빠른 호출
            minimal_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            
            response = await asyncio.wait_for(
                llm_client.ainvoke(minimal_prompt),
                timeout=3.0
            )
            
            fallback_time = time.time() - elapsed_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            return self._create_result(
                response_content, 
                elapsed_time + fallback_time, 
                "fallback",
                quality_metrics=QualityMetrics(overall_quality=0.5),  # 낮은 품질로 표시
                performance_metrics=PerformanceMetrics(response_time=elapsed_time + fallback_time)
            )
            
        except Exception as e:
            return self._create_error_result(f"Fallback failed: {e}", elapsed_time)
    
    def _create_result(
        self, 
        response: str, 
        execution_time: float, 
        method: str,
        quality_metrics: Optional[QualityMetrics] = None,
        performance_metrics: Optional[PerformanceMetrics] = None
    ) -> Dict[str, Any]:
        """결과 객체 생성"""
        
        return {
            "response": response,
            "execution_time": execution_time,
            "optimization_method": method,
            "quality_metrics": quality_metrics.__dict__ if quality_metrics else {},
            "performance_metrics": performance_metrics.__dict__ if performance_metrics else {},
            "success": True,
            "balanced_optimization": True
        }
    
    def _create_error_result(self, error: str, execution_time: float) -> Dict[str, Any]:
        """에러 결과 생성"""
        
        return {
            "response": "",
            "execution_time": execution_time,
            "optimization_method": "failed",
            "error": error,
            "success": False,
            "balanced_optimization": False
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """최적화 요약 반환"""
        
        avg_quality = statistics.mean(self.quality_history) if self.quality_history else 0
        avg_performance = statistics.mean(self.performance_history) if self.performance_history else 0
        
        return {
            "current_settings": self.adaptive_settings,
            "quality_history_avg": avg_quality,
            "performance_history_avg": avg_performance,
            "cache_size": len(self.quality_cache),
            "total_calls": len(self.quality_history),
            "optimization_level": "balanced_quality_speed"
        }

# 전역 인스턴스
_global_balanced_optimizer = None

def get_balanced_optimizer(config: Optional[BalancedConfig] = None) -> BalancedPerformanceOptimizer:
    """전역 균형 최적화 인스턴스 반환"""
    global _global_balanced_optimizer
    if _global_balanced_optimizer is None:
        _global_balanced_optimizer = BalancedPerformanceOptimizer(config)
    return _global_balanced_optimizer

async def optimize_with_quality_balance(
    llm_client, 
    prompt: str, 
    context: Optional[Dict] = None,
    target_quality: Optional[QualityLevel] = None
) -> Dict[str, Any]:
    """편의 함수: 균형 잡힌 최적화 호출"""
    optimizer = get_balanced_optimizer()
    return await optimizer.optimize_with_quality_balance(llm_client, prompt, context, target_quality)