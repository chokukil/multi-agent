"""
LLM First 실시간 분석 강화 시스템
Phase 2.4: LLM 분석 비율 90% 이상 달성

핵심 원칙:
- 폴백 로직 최소화 (LLM 우선)
- 실시간 LLM 분석 강화
- 동적 에이전트 라우팅 개선
- LLM 응답 품질 모니터링
- 적응적 LLM 전략 선택
"""

import asyncio
import time
import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import uuid
from pathlib import Path

# LLM 클라이언트
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)

class AnalysisStrategy(Enum):
    """분석 전략"""
    LLM_ONLY = "llm_only"           # 100% LLM 분석
    LLM_PREFERRED = "llm_preferred"  # LLM 우선, 필요시 폴백
    HYBRID = "hybrid"               # LLM + 룰 기반 하이브리드
    FALLBACK_ONLY = "fallback_only" # 폴백만 (비상시)

class LLMProvider(Enum):
    """LLM 제공자"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    LOCAL_MODEL = "local_model"

class AnalysisQuality(Enum):
    """분석 품질"""
    EXCELLENT = "excellent"   # 90-100점
    GOOD = "good"            # 70-89점
    ACCEPTABLE = "acceptable" # 50-69점
    POOR = "poor"            # 30-49점
    FAILED = "failed"        # 0-29점

@dataclass
class LLMAnalysisRequest:
    """LLM 분석 요청"""
    id: str
    user_query: str
    context: Dict[str, Any]
    priority: int = 5  # 1(highest) - 10(lowest)
    max_tokens: int = 2000
    temperature: float = 0.7
    strategy: AnalysisStrategy = AnalysisStrategy.LLM_PREFERRED
    timeout_seconds: float = 30.0
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMAnalysisResponse:
    """LLM 분석 응답"""
    request_id: str
    success: bool
    content: str
    analysis_quality: AnalysisQuality
    response_time: float
    token_usage: int
    llm_provider: LLMProvider
    confidence_score: float
    reasoning_steps: List[str]
    fallback_used: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMMetrics:
    """LLM 성능 메트릭"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_token_usage: int = 0
    avg_confidence_score: float = 0.0
    llm_usage_ratio: float = 0.0  # LLM vs 폴백 비율
    quality_distribution: Dict[AnalysisQuality, int] = field(default_factory=lambda: defaultdict(int))
    fallback_rate: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def llm_first_score(self) -> float:
        """LLM First 준수도 점수 (0-100)"""
        base_score = self.llm_usage_ratio * 100
        
        # 품질 가중치 적용
        quality_weight = 0
        total_quality_requests = sum(self.quality_distribution.values())
        
        if total_quality_requests > 0:
            quality_weight = (
                self.quality_distribution[AnalysisQuality.EXCELLENT] * 1.0 +
                self.quality_distribution[AnalysisQuality.GOOD] * 0.8 +
                self.quality_distribution[AnalysisQuality.ACCEPTABLE] * 0.6 +
                self.quality_distribution[AnalysisQuality.POOR] * 0.3 +
                self.quality_distribution[AnalysisQuality.FAILED] * 0.0
            ) / total_quality_requests
        
        # 최종 점수 (비율 70% + 품질 30%)
        return base_score * 0.7 + quality_weight * 100 * 0.3

class AdaptiveLLMRouter:
    """적응적 LLM 라우팅"""
    
    def __init__(self):
        # LLM 제공자별 성능 추적
        self.provider_metrics: Dict[LLMProvider, LLMMetrics] = {
            provider: LLMMetrics() for provider in LLMProvider
        }
        
        # 적응적 라우팅 설정
        self.routing_strategy = "performance_based"  # "round_robin", "least_latency", "highest_quality"
        self.fallback_chain = [
            LLMProvider.OPENAI_GPT4,
            LLMProvider.OPENAI_GPT35,
            LLMProvider.LOCAL_MODEL
        ]
        
        # 성능 기반 라우팅
        self.performance_window = 50  # 최근 50개 요청 기준
        self.performance_history: Dict[LLMProvider, deque] = {
            provider: deque(maxlen=self.performance_window) 
            for provider in LLMProvider
        }
        
        # 부하 밸런싱
        self.current_loads: Dict[LLMProvider, int] = defaultdict(int)
        self.max_concurrent_requests = 10
    
    def select_optimal_provider(self, request: LLMAnalysisRequest) -> LLMProvider:
        """최적 LLM 제공자 선택"""
        if request.strategy == AnalysisStrategy.FALLBACK_ONLY:
            return LLMProvider.LOCAL_MODEL
        
        if self.routing_strategy == "performance_based":
            return self._select_by_performance(request)
        elif self.routing_strategy == "least_latency":
            return self._select_by_latency()
        elif self.routing_strategy == "highest_quality":
            return self._select_by_quality()
        else:
            return self._round_robin_selection()
    
    def _select_by_performance(self, request: LLMAnalysisRequest) -> LLMProvider:
        """성능 기반 선택"""
        scores = {}
        
        for provider in LLMProvider:
            metrics = self.provider_metrics[provider]
            current_load = self.current_loads[provider]
            
            # 기본 점수 계산
            success_score = metrics.success_rate * 100
            response_time_score = max(0, 100 - metrics.avg_response_time * 2)
            quality_score = metrics.avg_confidence_score * 100
            
            # 부하 고려
            load_penalty = (current_load / self.max_concurrent_requests) * 20
            
            # 우선순위 고려 (높은 우선순위 요청은 좋은 모델 선호)
            priority_bonus = 0
            if request.priority <= 3 and provider == LLMProvider.OPENAI_GPT4:
                priority_bonus = 15
            elif request.priority <= 5 and provider in [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35]:
                priority_bonus = 10
            
            total_score = (success_score * 0.4 + 
                          response_time_score * 0.3 + 
                          quality_score * 0.3 + 
                          priority_bonus - load_penalty)
            
            scores[provider] = total_score
        
        # 최고 점수 제공자 선택
        best_provider = max(scores.items(), key=lambda x: x[1])[0]
        return best_provider
    
    def _select_by_latency(self) -> LLMProvider:
        """지연시간 기반 선택"""
        best_provider = LLMProvider.OPENAI_GPT4
        best_latency = float('inf')
        
        for provider, metrics in self.provider_metrics.items():
            if metrics.avg_response_time < best_latency and metrics.success_rate > 0.8:
                best_latency = metrics.avg_response_time
                best_provider = provider
        
        return best_provider
    
    def _select_by_quality(self) -> LLMProvider:
        """품질 기반 선택"""
        best_provider = LLMProvider.OPENAI_GPT4
        best_quality = 0.0
        
        for provider, metrics in self.provider_metrics.items():
            if metrics.avg_confidence_score > best_quality and metrics.success_rate > 0.7:
                best_quality = metrics.avg_confidence_score
                best_provider = provider
        
        return best_provider
    
    def _round_robin_selection(self) -> LLMProvider:
        """라운드 로빈 선택"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        providers = list(LLMProvider)
        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        
        return provider
    
    def update_provider_metrics(self, provider: LLMProvider, response: LLMAnalysisResponse):
        """제공자 메트릭 업데이트"""
        metrics = self.provider_metrics[provider]
        
        metrics.total_requests += 1
        
        if response.success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # 이동 평균 업데이트
        alpha = 0.1
        metrics.avg_response_time = (alpha * response.response_time + 
                                   (1 - alpha) * metrics.avg_response_time)
        metrics.avg_token_usage = int(alpha * response.token_usage + 
                                    (1 - alpha) * metrics.avg_token_usage)
        metrics.avg_confidence_score = (alpha * response.confidence_score + 
                                      (1 - alpha) * metrics.avg_confidence_score)
        
        # 품질 분포 업데이트
        metrics.quality_distribution[response.analysis_quality] += 1
        
        # 성능 히스토리 업데이트
        self.performance_history[provider].append({
            "timestamp": datetime.now(),
            "response_time": response.response_time,
            "success": response.success,
            "confidence": response.confidence_score
        })
        
        # 부하 감소
        self.current_loads[provider] = max(0, self.current_loads[provider] - 1)
    
    def record_load_increase(self, provider: LLMProvider):
        """부하 증가 기록"""
        self.current_loads[provider] += 1
    
    def get_routing_status(self) -> Dict[str, Any]:
        """라우팅 상태 반환"""
        return {
            "routing_strategy": self.routing_strategy,
            "provider_metrics": {
                provider.value: {
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "avg_confidence": metrics.avg_confidence_score,
                    "current_load": self.current_loads[provider],
                    "llm_first_score": metrics.llm_first_score
                }
                for provider, metrics in self.provider_metrics.items()
            },
            "fallback_chain": [p.value for p in self.fallback_chain]
        }

class LLMQualityAssessor:
    """LLM 응답 품질 평가"""
    
    def __init__(self):
        self.quality_criteria = {
            "relevance": 0.3,      # 관련성
            "completeness": 0.25,   # 완성도
            "accuracy": 0.25,      # 정확성
            "clarity": 0.2         # 명확성
        }
        
        # 품질 평가 패턴
        self.quality_indicators = {
            "excellent": [
                "구체적인 데이터 분석",
                "단계별 추론 과정",
                "수치적 근거 제시",
                "전문적 인사이트",
                "actionable recommendations"
            ],
            "poor": [
                "모호한 응답",
                "관련성 부족",
                "오류 포함",
                "불완전한 분석",
                "generic responses"
            ]
        }
    
    async def assess_response_quality(self, 
                                    request: LLMAnalysisRequest, 
                                    response_content: str,
                                    context: Dict[str, Any] = None) -> Tuple[AnalysisQuality, float, List[str]]:
        """응답 품질 평가"""
        
        # 기본 품질 점수
        scores = {}
        reasoning_steps = []
        
        # 1. 관련성 평가
        relevance_score = await self._assess_relevance(request.user_query, response_content)
        scores["relevance"] = relevance_score
        reasoning_steps.append(f"관련성 점수: {relevance_score:.2f}")
        
        # 2. 완성도 평가
        completeness_score = self._assess_completeness(response_content)
        scores["completeness"] = completeness_score
        reasoning_steps.append(f"완성도 점수: {completeness_score:.2f}")
        
        # 3. 정확성 평가
        accuracy_score = self._assess_accuracy(response_content, context)
        scores["accuracy"] = accuracy_score
        reasoning_steps.append(f"정확성 점수: {accuracy_score:.2f}")
        
        # 4. 명확성 평가
        clarity_score = self._assess_clarity(response_content)
        scores["clarity"] = clarity_score
        reasoning_steps.append(f"명확성 점수: {clarity_score:.2f}")
        
        # 가중 평균 계산
        total_score = sum(scores[criterion] * weight 
                         for criterion, weight in self.quality_criteria.items())
        
        # 품질 등급 결정
        if total_score >= 0.9:
            quality = AnalysisQuality.EXCELLENT
        elif total_score >= 0.7:
            quality = AnalysisQuality.GOOD
        elif total_score >= 0.5:
            quality = AnalysisQuality.ACCEPTABLE
        elif total_score >= 0.3:
            quality = AnalysisQuality.POOR
        else:
            quality = AnalysisQuality.FAILED
        
        reasoning_steps.append(f"최종 품질 등급: {quality.value} (점수: {total_score:.2f})")
        
        return quality, total_score, reasoning_steps
    
    async def _assess_relevance(self, query: str, response: str) -> float:
        """관련성 평가"""
        # 간단한 키워드 매칭 + 의미 분석
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # 키워드 중복도
        overlap = len(query_words.intersection(response_words))
        keyword_score = min(1.0, overlap / max(len(query_words), 1))
        
        # 길이 기반 관련성 (너무 짧거나 너무 긴 응답 페널티)
        response_length = len(response)
        if response_length < 50:
            length_penalty = 0.5
        elif response_length > 5000:
            length_penalty = 0.8
        else:
            length_penalty = 1.0
        
        return keyword_score * length_penalty
    
    def _assess_completeness(self, response: str) -> float:
        """완성도 평가"""
        # 응답 구조 분석
        structure_score = 0.0
        
        # 단락 구조
        paragraphs = response.split('\n\n')
        if len(paragraphs) >= 2:
            structure_score += 0.3
        
        # 목록이나 단계 포함
        if any(marker in response for marker in ['1.', '2.', '•', '-', '*']):
            structure_score += 0.3
        
        # 결론 포함
        conclusion_markers = ['결론', '요약', '정리하면', '따라서', 'conclusion']
        if any(marker in response.lower() for marker in conclusion_markers):
            structure_score += 0.2
        
        # 수치 데이터 포함
        import re
        numbers = re.findall(r'\d+\.?\d*%?', response)
        if len(numbers) >= 3:
            structure_score += 0.2
        
        return min(1.0, structure_score)
    
    def _assess_accuracy(self, response: str, context: Dict[str, Any] = None) -> float:
        """정확성 평가"""
        accuracy_score = 0.8  # 기본 점수
        
        # 기본적인 정확성 체크
        error_patterns = [
            r'error|오류|에러',
            r'failed|실패',
            r'unknown|알 수 없음',
            r'not found|찾을 수 없음'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, response.lower()):
                accuracy_score -= 0.2
        
        # 긍정적 지표
        positive_patterns = [
            r'분석 결과',
            r'데이터에 따르면',
            r'통계적으로',
            r'구체적으로'
        ]
        
        for pattern in positive_patterns:
            if re.search(pattern, response.lower()):
                accuracy_score += 0.1
        
        return max(0.0, min(1.0, accuracy_score))
    
    def _assess_clarity(self, response: str) -> float:
        """명확성 평가"""
        clarity_score = 0.0
        
        # 문장 길이 분석
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 10 <= avg_sentence_length <= 25:  # 적절한 문장 길이
            clarity_score += 0.4
        
        # 전문 용어 vs 이해하기 쉬운 설명
        technical_terms = ['분석', '데이터', '통계', '모델', '알고리즘']
        explanatory_terms = ['즉', '다시 말해', '예를 들어', '구체적으로']
        
        tech_count = sum(1 for term in technical_terms if term in response)
        explain_count = sum(1 for term in explanatory_terms if term in response)
        
        if tech_count > 0 and explain_count > 0:
            clarity_score += 0.3
        
        # 구조적 명확성
        if '1.' in response or '첫째' in response:
            clarity_score += 0.3
        
        return min(1.0, clarity_score)

class LLMFirstAnalyzer:
    """LLM First 실시간 분석기"""
    
    def __init__(self):
        # 핵심 컴포넌트
        self.llm_router = AdaptiveLLMRouter()
        self.quality_assessor = LLMQualityAssessor()
        
        # LLM 클라이언트들
        self.openai_client = AsyncOpenAI()
        self.llm_clients = {}
        
        # 분석 큐 및 우선순위 관리
        self.analysis_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_analyses: Dict[str, LLMAnalysisRequest] = {}
        
        # 메트릭 및 모니터링
        self.global_metrics = LLMMetrics()
        self.analysis_history: deque = deque(maxlen=1000)
        
        # 설정
        self.max_concurrent_analyses = 10
        self.fallback_enabled = True
        self.target_llm_ratio = 0.9  # 90% LLM 분석 목표
        
        # 폴백 방지 전략
        self.fallback_prevention = {
            "retry_with_simpler_prompt": True,
            "use_alternative_model": True,
            "reduce_complexity": True,
            "chunk_large_requests": True
        }
        
        # 결과 저장 경로
        self.results_dir = Path("monitoring/llm_analysis_performance")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 워커 태스크
        self.workers: List[asyncio.Task] = []
        self.is_running = False
    
    async def initialize(self):
        """분석기 초기화"""
        logger.info("🧠 LLM First 분석기 초기화 중...")
        
        # 워커 태스크 시작
        self.is_running = True
        for i in range(self.max_concurrent_analyses):
            worker = asyncio.create_task(self._analysis_worker(f"worker_{i}"))
            self.workers.append(worker)
        
        logger.info(f"✅ LLM First 분석기 초기화 완료 ({len(self.workers)}개 워커)")
    
    async def analyze_realtime(self, 
                             user_query: str, 
                             context: Dict[str, Any] = None,
                             priority: int = 5,
                             strategy: AnalysisStrategy = AnalysisStrategy.LLM_PREFERRED) -> LLMAnalysisResponse:
        """실시간 LLM 분석 요청"""
        
        request = LLMAnalysisRequest(
            id=str(uuid.uuid4()),
            user_query=user_query,
            context=context or {},
            priority=priority,
            strategy=strategy,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # 우선순위 큐에 추가
        await self.analysis_queue.put((priority, time.time(), request))
        
        # 요청 추적
        self.active_analyses[request.id] = request
        
        logger.info(f"📝 LLM 분석 요청 추가: {request.id} (우선순위: {priority})")
        
        # 응답 대기 (비동기적으로 처리됨)
        return await self._wait_for_response(request.id)
    
    async def _analysis_worker(self, worker_name: str):
        """분석 워커"""
        logger.info(f"🔧 분석 워커 시작: {worker_name}")
        
        while self.is_running:
            try:
                # 큐에서 요청 가져오기
                priority, timestamp, request = await asyncio.wait_for(
                    self.analysis_queue.get(), timeout=1.0
                )
                
                logger.info(f"🔍 {worker_name}가 분석 시작: {request.id}")
                
                # LLM 분석 실행
                response = await self._execute_llm_analysis(request)
                
                # 응답 저장 및 메트릭 업데이트
                await self._process_analysis_response(request, response)
                
                logger.info(f"✅ {worker_name}가 분석 완료: {request.id} (품질: {response.analysis_quality.value})")
                
            except asyncio.TimeoutError:
                # 큐가 비어있음 - 정상
                continue
            except Exception as e:
                logger.error(f"❌ {worker_name} 오류: {e}")
                await asyncio.sleep(1)
    
    async def _execute_llm_analysis(self, request: LLMAnalysisRequest) -> LLMAnalysisResponse:
        """LLM 분석 실행"""
        start_time = time.time()
        
        # 최적 LLM 제공자 선택
        provider = self.llm_router.select_optimal_provider(request)
        self.llm_router.record_load_increase(provider)
        
        try:
            # LLM 분석 시도
            if request.strategy == AnalysisStrategy.LLM_ONLY or request.strategy == AnalysisStrategy.LLM_PREFERRED:
                response = await self._try_llm_analysis(request, provider)
                
                if response.success:
                    return response
                elif request.strategy == AnalysisStrategy.LLM_ONLY:
                    # LLM 전용 모드에서는 실패해도 폴백하지 않음
                    return response
                else:
                    # LLM_PREFERRED 모드에서는 폴백 시도
                    logger.warning(f"LLM 분석 실패, 폴백 시도: {request.id}")
                    return await self._try_fallback_analysis(request)
            
            else:
                # HYBRID 또는 FALLBACK_ONLY
                return await self._try_fallback_analysis(request)
                
        except Exception as e:
            logger.error(f"분석 실행 오류: {e}")
            
            return LLMAnalysisResponse(
                request_id=request.id,
                success=False,
                content="",
                analysis_quality=AnalysisQuality.FAILED,
                response_time=time.time() - start_time,
                token_usage=0,
                llm_provider=provider,
                confidence_score=0.0,
                reasoning_steps=[],
                fallback_used=False,
                error_message=str(e)
            )
    
    async def _try_llm_analysis(self, request: LLMAnalysisRequest, provider: LLMProvider) -> LLMAnalysisResponse:
        """LLM 분석 시도"""
        start_time = time.time()
        
        try:
            # 프롬프트 최적화
            optimized_prompt = await self._optimize_prompt(request)
            
            # LLM 호출
            if provider == LLMProvider.OPENAI_GPT4:
                response_content, token_usage = await self._call_openai_gpt4(optimized_prompt, request)
            elif provider == LLMProvider.OPENAI_GPT35:
                response_content, token_usage = await self._call_openai_gpt35(optimized_prompt, request)
            else:
                # 다른 제공자들은 추후 구현
                raise NotImplementedError(f"Provider {provider.value} not implemented")
            
            response_time = time.time() - start_time
            
            # 품질 평가
            quality, confidence, reasoning = await self.quality_assessor.assess_response_quality(
                request, response_content, request.context
            )
            
            response = LLMAnalysisResponse(
                request_id=request.id,
                success=True,
                content=response_content,
                analysis_quality=quality,
                response_time=response_time,
                token_usage=token_usage,
                llm_provider=provider,
                confidence_score=confidence,
                reasoning_steps=reasoning,
                fallback_used=False
            )
            
            # 라우터 메트릭 업데이트
            self.llm_router.update_provider_metrics(provider, response)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM 분석 실패 ({provider.value}): {e}")
            
            response_time = time.time() - start_time
            
            response = LLMAnalysisResponse(
                request_id=request.id,
                success=False,
                content="",
                analysis_quality=AnalysisQuality.FAILED,
                response_time=response_time,
                token_usage=0,
                llm_provider=provider,
                confidence_score=0.0,
                reasoning_steps=[],
                fallback_used=False,
                error_message=str(e)
            )
            
            self.llm_router.update_provider_metrics(provider, response)
            return response
    
    async def _optimize_prompt(self, request: LLMAnalysisRequest) -> str:
        """프롬프트 최적화"""
        base_prompt = f"""당신은 데이터 분석 전문가입니다. 다음 요청을 분석해주세요:

사용자 질문: {request.user_query}

컨텍스트 정보:
{json.dumps(request.context, indent=2, ensure_ascii=False)}

분석 요구사항:
1. 구체적이고 실용적인 분석 제공
2. 데이터 기반 인사이트 도출
3. 명확한 결론과 권장사항 제시
4. 단계별 추론 과정 설명

다음 형식으로 답변해주세요:
## 분석 결과
[구체적인 분석 내용]

## 주요 인사이트
[핵심 발견사항]

## 권장사항
[실행 가능한 제안]
"""
        
        # 프롬프트 길이 최적화
        if len(base_prompt) > 3000:
            # 컨텍스트 축약
            context_summary = str(request.context)[:500] + "..."
            base_prompt = base_prompt.replace(
                json.dumps(request.context, indent=2, ensure_ascii=False),
                context_summary
            )
        
        return base_prompt
    
    async def _call_openai_gpt4(self, prompt: str, request: LLMAnalysisRequest) -> Tuple[str, int]:
        """OpenAI GPT-4 호출"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout_seconds
            )
            
            content = response.choices[0].message.content
            token_usage = response.usage.total_tokens
            
            return content, token_usage
            
        except Exception as e:
            logger.error(f"OpenAI GPT-4 호출 오류: {e}")
            raise
    
    async def _call_openai_gpt35(self, prompt: str, request: LLMAnalysisRequest) -> Tuple[str, int]:
        """OpenAI GPT-3.5 호출"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=request.timeout_seconds
            )
            
            content = response.choices[0].message.content
            token_usage = response.usage.total_tokens
            
            return content, token_usage
            
        except Exception as e:
            logger.error(f"OpenAI GPT-3.5 호출 오류: {e}")
            raise
    
    async def _try_fallback_analysis(self, request: LLMAnalysisRequest) -> LLMAnalysisResponse:
        """폴백 분석 시도"""
        start_time = time.time()
        
        logger.info(f"폴백 분석 시작: {request.id}")
        
        # 간단한 키워드 기반 분석 (폴백)
        fallback_content = self._generate_fallback_response(request)
        
        response_time = time.time() - start_time
        
        # 폴백 응답의 품질은 일반적으로 낮음
        quality, confidence, reasoning = await self.quality_assessor.assess_response_quality(
            request, fallback_content, request.context
        )
        
        return LLMAnalysisResponse(
            request_id=request.id,
            success=True,
            content=fallback_content,
            analysis_quality=quality,
            response_time=response_time,
            token_usage=0,
            llm_provider=LLMProvider.LOCAL_MODEL,  # 폴백은 로컬로 처리
            confidence_score=confidence,
            reasoning_steps=reasoning,
            fallback_used=True,
            metadata={"fallback_reason": "LLM analysis failed"}
        )
    
    def _generate_fallback_response(self, request: LLMAnalysisRequest) -> str:
        """폴백 응답 생성"""
        query_lower = request.user_query.lower()
        
        # 키워드 기반 간단 응답
        if any(word in query_lower for word in ['분석', 'analyze', '요약', 'summary']):
            return f"""## 기본 분석 결과

요청하신 '{request.user_query}' 에 대한 기본 분석을 제공합니다.

## 제공된 정보 요약
- 분석 대상: {request.context.get('data_info', '데이터 정보 없음')}
- 요청 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 권장사항
더 상세한 분석을 위해서는 구체적인 데이터나 분석 목적을 제공해주시기 바랍니다.

*참고: 이는 기본 응답입니다. 더 정확한 분석을 위해 다시 시도해주세요.*
"""
        
        else:
            return f"""## 응답

'{request.user_query}' 요청을 처리했습니다.

현재 기본 모드로 응답하고 있습니다. 더 상세하고 정확한 분석을 위해 다음을 고려해주세요:

1. 더 구체적인 질문 제시
2. 관련 데이터나 컨텍스트 제공
3. 분석 목적 명확화

*참고: 이는 간소화된 응답입니다.*
"""
    
    async def _process_analysis_response(self, request: LLMAnalysisRequest, response: LLMAnalysisResponse):
        """분석 응답 처리"""
        # 글로벌 메트릭 업데이트
        self.global_metrics.total_requests += 1
        
        if response.success:
            self.global_metrics.successful_requests += 1
        else:
            self.global_metrics.failed_requests += 1
        
        # LLM vs 폴백 비율 업데이트
        if not response.fallback_used:
            # LLM 사용
            llm_count = sum(1 for h in self.analysis_history if not h.get('fallback_used', False))
            self.global_metrics.llm_usage_ratio = llm_count / max(len(self.analysis_history), 1)
        
        # 품질 분포 업데이트
        self.global_metrics.quality_distribution[response.analysis_quality] += 1
        
        # 이동 평균 업데이트
        alpha = 0.1
        self.global_metrics.avg_response_time = (
            alpha * response.response_time + 
            (1 - alpha) * self.global_metrics.avg_response_time
        )
        self.global_metrics.avg_confidence_score = (
            alpha * response.confidence_score + 
            (1 - alpha) * self.global_metrics.avg_confidence_score
        )
        
        # 히스토리 추가
        self.analysis_history.append({
            "request_id": request.id,
            "timestamp": datetime.now(),
            "success": response.success,
            "quality": response.analysis_quality.value,
            "response_time": response.response_time,
            "fallback_used": response.fallback_used,
            "provider": response.llm_provider.value
        })
        
        # 활성 분석에서 제거
        if request.id in self.active_analyses:
            del self.active_analyses[request.id]
    
    async def _wait_for_response(self, request_id: str) -> LLMAnalysisResponse:
        """응답 대기 (실제로는 워커에서 비동기 처리됨)"""
        # 이 구현에서는 단순화를 위해 즉시 처리
        # 실제로는 결과를 큐나 캐시에서 가져와야 함
        await asyncio.sleep(0.1)  # 최소 대기
        
        # 임시 구현 - 실제로는 결과 저장소에서 가져오기
        return LLMAnalysisResponse(
            request_id=request_id,
            success=True,
            content="분석이 처리 중입니다.",
            analysis_quality=AnalysisQuality.ACCEPTABLE,
            response_time=0.1,
            token_usage=0,
            llm_provider=LLMProvider.OPENAI_GPT4,
            confidence_score=0.8,
            reasoning_steps=["분석 큐에 추가됨"]
        )
    
    async def shutdown(self):
        """분석기 종료"""
        logger.info("🛑 LLM First 분석기 종료 중...")
        
        self.is_running = False
        
        # 워커 태스크 정리
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("✅ LLM First 분석기 종료 완료")
    
    def get_llm_first_status(self) -> Dict[str, Any]:
        """LLM First 상태 반환"""
        return {
            "global_metrics": {
                "total_requests": self.global_metrics.total_requests,
                "success_rate": self.global_metrics.success_rate,
                "llm_usage_ratio": self.global_metrics.llm_usage_ratio,
                "llm_first_score": self.global_metrics.llm_first_score,
                "avg_response_time": self.global_metrics.avg_response_time,
                "avg_confidence": self.global_metrics.avg_confidence_score,
                "fallback_rate": 1 - self.global_metrics.llm_usage_ratio
            },
            "router_status": self.llm_router.get_routing_status(),
            "active_analyses": len(self.active_analyses),
            "queue_size": self.analysis_queue.qsize(),
            "target_llm_ratio": self.target_llm_ratio,
            "system_status": "running" if self.is_running else "stopped"
        }


# 사용 예시 및 테스트
async def test_llm_first_analyzer():
    """LLM First 분석기 테스트"""
    analyzer = LLMFirstAnalyzer()
    
    try:
        await analyzer.initialize()
        
        # 테스트 분석 요청들
        test_queries = [
            "데이터의 전반적인 패턴을 분석해주세요",
            "이상치를 탐지하고 원인을 설명해주세요", 
            "예측 모델을 위한 특성을 추천해주세요",
            "데이터 품질 문제를 식별해주세요"
        ]
        
        # 병렬 분석 테스트
        tasks = []
        for i, query in enumerate(test_queries):
            context = {"data_info": f"테스트 데이터셋 {i+1}"}
            task = analyzer.analyze_realtime(query, context, priority=i+1)
            tasks.append(task)
        
        # 결과 수집
        responses = await asyncio.gather(*tasks)
        
        # 결과 출력
        for i, response in enumerate(responses):
            print(f"📊 분석 {i+1}: {response.analysis_quality.value} "
                  f"(응답시간: {response.response_time:.2f}초, "
                  f"신뢰도: {response.confidence_score:.2f})")
        
        # 전체 상태 확인
        status = analyzer.get_llm_first_status()
        print(f"\n🎯 LLM First 점수: {status['global_metrics']['llm_first_score']:.1f}/100")
        print(f"LLM 사용 비율: {status['global_metrics']['llm_usage_ratio']:.2%}")
        
    finally:
        await analyzer.shutdown()

if __name__ == "__main__":
    asyncio.run(test_llm_first_analyzer()) 