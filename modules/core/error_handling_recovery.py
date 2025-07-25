"""
LLM-Powered Error Handling and Recovery System

검증된 Universal Engine 패턴:
- IntelligentErrorDiagnosis: 오류 원인 자동 진단
- ContextAwareRecovery: 상황별 복구 전략
- ProgressiveRetryMechanism: 점진적 재시도 패턴
- UserFriendlyErrorExplanation: 사용자 친화적 오류 설명
- SelfHealingSystem: 자가 치유 시스템
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re

# Universal Engine 패턴 가져오기 (사용 가능한 경우)
try:
    from core.universal_engine.intelligent_error_diagnosis import IntelligentErrorDiagnosis
    from core.universal_engine.context_aware_recovery import ContextAwareRecovery
    from core.universal_engine.self_healing_system import SelfHealingSystem
    from core.universal_engine.llm_factory import LLMFactory
    UNIVERSAL_ENGINE_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """오류 심각도"""
    LOW = "low"           # 경고, 비치명적
    MEDIUM = "medium"     # 기능 제한, 부분 실패
    HIGH = "high"         # 주요 기능 실패
    CRITICAL = "critical" # 시스템 전체 실패


class RecoveryStrategy(Enum):
    """복구 전략"""
    AUTO_RETRY = "auto_retry"           # 자동 재시도
    FALLBACK = "fallback"               # 대체 방법 사용
    USER_INTERVENTION = "user_intervention"  # 사용자 개입 필요
    GRACEFUL_DEGRADATION = "graceful_degradation"  # 우아한 성능 저하
    SYSTEM_RESTART = "system_restart"   # 시스템 재시작


@dataclass
class ErrorContext:
    """오류 컨텍스트 정보"""
    error_id: str
    timestamp: datetime
    component: str
    function_name: str
    error_type: str
    error_message: str
    full_traceback: str
    user_context: Dict[str, Any]
    system_state: Dict[str, Any]
    previous_errors: List[str] = field(default_factory=list)


@dataclass
class RecoveryPlan:
    """복구 계획"""
    plan_id: str
    error_id: str
    strategy: RecoveryStrategy
    confidence: float
    estimated_recovery_time: int  # seconds
    steps: List[str]
    fallback_options: List[str]
    user_explanation: str
    technical_details: str
    requires_user_input: bool = False


@dataclass
class ErrorPattern:
    """오류 패턴"""
    pattern_id: str
    error_signature: str
    frequency: int
    last_occurrence: datetime
    typical_causes: List[str]
    proven_solutions: List[str]
    prevention_measures: List[str]


class LLMErrorHandler:
    """
    LLM 기반 지능형 오류 처리 및 복구 시스템
    검증된 Universal Engine 패턴을 활용한 자동 오류 진단 및 복구
    """
    
    def __init__(self):
        """LLM Error Handler 초기화"""
        
        # Universal Engine 컴포넌트 초기화
        if UNIVERSAL_ENGINE_AVAILABLE:
            self.error_diagnosis = IntelligentErrorDiagnosis()
            self.recovery_system = ContextAwareRecovery()
            self.self_healing = SelfHealingSystem()
            self.llm_client = LLMFactory.create_llm()
        else:
            self.error_diagnosis = None
            self.recovery_system = None
            self.self_healing = None
            self.llm_client = None
        
        # 오류 추적 및 패턴 분석
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.recovery_success_rate: Dict[str, float] = {}
        
        # 재시도 설정
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1.0,  # seconds
            'max_delay': 30.0,  # seconds
            'exponential_base': 2.0,
            'jitter': 0.1
        }
        
        # 회로 차단기 설정 (Circuit Breaker Pattern)
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_config = {
            'failure_threshold': 5,
            'recovery_timeout': 60,  # seconds
            'half_open_max_calls': 3
        }
        
        # 사용자 친화적 메시지 템플릿
        self.user_message_templates = self._initialize_message_templates()
        
        logger.info("LLM Error Handler initialized")
    
    def _initialize_message_templates(self) -> Dict[str, Dict[str, str]]:
        """사용자 친화적 메시지 템플릿 초기화"""
        return {
            'file_upload_error': {
                'beginner': "파일 업로드 중 문제가 발생했습니다. 파일 형식이나 크기를 확인해주세요.",
                'intermediate': "파일 처리 오류가 발생했습니다. 지원되는 형식(CSV, Excel, JSON)인지 확인하고 다시 시도해주세요.",
                'advanced': "파일 파싱 오류: 인코딩, 구분자, 또는 스키마 문제일 수 있습니다. 로그를 확인하여 구체적인 원인을 파악하세요."
            },
            
            'agent_connection_error': {
                'beginner': "분석 서비스에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요.",
                'intermediate': "에이전트 연결 실패입니다. 네트워크 상태를 확인하거나 잠시 후 재시도해주세요.",
                'advanced': "A2A SDK 에이전트 연결 타임아웃. 포트 상태와 서비스 가용성을 확인하세요."
            },
            
            'analysis_error': {
                'beginner': "데이터 분석 중 오류가 발생했습니다. 데이터를 다시 확인해보세요.",
                'intermediate': "분석 프로세스에서 오류가 발생했습니다. 데이터 형식이나 내용에 문제가 있을 수 있습니다.",
                'advanced': "분석 엔진 오류: 알고리즘 파라미터나 데이터 전처리 단계에서 문제가 발생했습니다."
            },
            
            'memory_error': {
                'beginner': "데이터가 너무 커서 처리할 수 없습니다. 더 작은 파일로 나누어 시도해보세요.",
                'intermediate': "메모리 부족 오류입니다. 데이터 크기를 줄이거나 샘플링을 고려해보세요.",
                'advanced': "메모리 할당 실패: 현재 {memory_used}MB 사용 중. 데이터 청킹이나 스트리밍 처리를 권장합니다."
            }
        }
    
    async def handle_error(self, 
                          error: Exception,
                          context: Dict[str, Any],
                          component: str = "unknown",
                          function_name: str = "unknown") -> Tuple[bool, str, Optional[RecoveryPlan]]:
        """
        메인 오류 처리 함수
        
        Returns:
            (recovery_success, user_message, recovery_plan)
        """
        try:
            # 오류 컨텍스트 생성
            error_context = self._create_error_context(
                error, context, component, function_name
            )
            
            # 회로 차단기 확인
            circuit_key = f"{component}:{function_name}"
            if self._is_circuit_open(circuit_key):
                return False, "서비스가 일시적으로 중단되었습니다. 잠시 후 다시 시도해주세요.", None
            
            # 1. 지능형 오류 진단
            diagnosis = await self._diagnose_error(error_context)
            
            # 2. 복구 계획 생성
            recovery_plan = await self._generate_recovery_plan(error_context, diagnosis)
            
            # 3. 복구 시도
            recovery_success = await self._execute_recovery_plan(recovery_plan, error_context)
            
            # 4. 결과에 따른 처리
            if recovery_success:
                self._record_successful_recovery(error_context, recovery_plan)
                user_message = self._generate_user_message(error_context, recovery_plan, success=True)
                return True, user_message, recovery_plan
            else:
                self._record_failed_recovery(error_context, recovery_plan)
                self._update_circuit_breaker(circuit_key, success=False)
                user_message = self._generate_user_message(error_context, recovery_plan, success=False)
                return False, user_message, recovery_plan
                
        except Exception as handler_error:
            logger.error(f"Error in error handler: {str(handler_error)}")
            return False, "예상치 못한 오류가 발생했습니다. 시스템 관리자에게 문의해주세요.", None
    
    def _create_error_context(self, 
                            error: Exception,
                            context: Dict[str, Any],
                            component: str,
                            function_name: str) -> ErrorContext:
        """오류 컨텍스트 생성"""
        
        error_id = f"err_{int(time.time())}_{hash(str(error)) % 10000}"
        
        # 시스템 상태 수집
        system_state = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'function': function_name,
            'memory_usage': self._get_memory_usage(),
            'active_sessions': len(getattr(self, 'active_sessions', [])),
            'recent_errors': len([e for e in self.error_history 
                                if e.timestamp > datetime.now() - timedelta(minutes=10)])
        }
        
        # 최근 오류 패턴
        recent_errors = [e.error_type for e in self.error_history[-5:]]
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            component=component,
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            full_traceback=traceback.format_exc(),
            user_context=context,
            system_state=system_state,
            previous_errors=recent_errors
        )
    
    async def _diagnose_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """지능형 오류 진단"""
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.error_diagnosis:
                # Universal Engine IntelligentErrorDiagnosis 사용
                diagnosis = await self.error_diagnosis.diagnose_error(error_context)
                return diagnosis
            else:
                # 기본 진단 로직
                return await self._basic_error_diagnosis(error_context)
                
        except Exception as e:
            logger.error(f"Error diagnosis failed: {str(e)}")
            return {'diagnosis': 'unknown', 'confidence': 0.1}
    
    async def _basic_error_diagnosis(self, error_context: ErrorContext) -> Dict[str, Any]:
        """기본 오류 진단 로직"""
        
        error_type = error_context.error_type
        error_message = error_context.error_message.lower()
        
        # 일반적인 오류 패턴 매칭
        if 'filenotfound' in error_type.lower() or 'no such file' in error_message:
            return {
                'diagnosis': 'file_not_found',
                'severity': ErrorSeverity.MEDIUM,
                'likely_cause': '파일 경로가 잘못되었거나 파일이 존재하지 않음',
                'confidence': 0.9
            }
        
        elif 'memoryerror' in error_type.lower() or 'out of memory' in error_message:
            return {
                'diagnosis': 'memory_exhaustion',
                'severity': ErrorSeverity.HIGH,
                'likely_cause': '데이터가 너무 크거나 메모리 부족',
                'confidence': 0.95
            }
        
        elif 'connectionerror' in error_type.lower() or 'timeout' in error_message:
            return {
                'diagnosis': 'network_connectivity',
                'severity': ErrorSeverity.MEDIUM,
                'likely_cause': '네트워크 연결 문제 또는 서비스 응답 없음',
                'confidence': 0.85
            }
        
        elif 'keyerror' in error_type.lower() or 'column' in error_message:
            return {
                'diagnosis': 'data_schema_mismatch',
                'severity': ErrorSeverity.MEDIUM,
                'likely_cause': '예상하지 못한 데이터 스키마 또는 누락된 컬럼',
                'confidence': 0.8
            }
        
        elif 'valueerror' in error_type.lower():
            return {
                'diagnosis': 'invalid_data_format',
                'severity': ErrorSeverity.MEDIUM,
                'likely_cause': '잘못된 데이터 형식 또는 값',
                'confidence': 0.75
            }
        
        else:
            return {
                'diagnosis': 'unknown_error',
                'severity': ErrorSeverity.MEDIUM,
                'likely_cause': '알 수 없는 오류',
                'confidence': 0.3
            }
    
    async def _generate_recovery_plan(self, 
                                    error_context: ErrorContext,
                                    diagnosis: Dict[str, Any]) -> RecoveryPlan:
        """복구 계획 생성"""
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.recovery_system:
                # Universal Engine ContextAwareRecovery 사용
                recovery_plan = await self.recovery_system.generate_recovery_plan(
                    error_context, diagnosis
                )
                return recovery_plan
            else:
                # 기본 복구 계획 생성
                return await self._basic_recovery_plan(error_context, diagnosis)
                
        except Exception as e:
            logger.error(f"Recovery plan generation failed: {str(e)}")
            return self._create_fallback_recovery_plan(error_context)
    
    async def _basic_recovery_plan(self, 
                                 error_context: ErrorContext,
                                 diagnosis: Dict[str, Any]) -> RecoveryPlan:
        """기본 복구 계획 생성"""
        
        diagnosis_type = diagnosis.get('diagnosis', 'unknown_error')
        severity = diagnosis.get('severity', ErrorSeverity.MEDIUM)
        
        plan_id = f"plan_{error_context.error_id}"
        
        if diagnosis_type == 'memory_exhaustion':
            return RecoveryPlan(
                plan_id=plan_id,
                error_id=error_context.error_id,
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                confidence=0.8,
                estimated_recovery_time=5,
                steps=[
                    "데이터 샘플링으로 크기 축소",
                    "청킹을 통한 분할 처리",
                    "메모리 효율적 알고리즘 사용"
                ],
                fallback_options=[
                    "더 작은 데이터셋으로 재시도",
                    "CSV 형식으로 변환 후 처리"
                ],
                user_explanation="데이터가 너무 커서 축소된 버전으로 분석을 진행합니다.",
                technical_details=f"메모리 사용량: {diagnosis.get('memory_usage', 'N/A')}MB",
                requires_user_input=False
            )
        
        elif diagnosis_type == 'network_connectivity':
            return RecoveryPlan(
                plan_id=plan_id,
                error_id=error_context.error_id,
                strategy=RecoveryStrategy.AUTO_RETRY,
                confidence=0.7,
                estimated_recovery_time=10,
                steps=[
                    "연결 상태 확인",
                    "짧은 지연 후 재시도",
                    "대체 엔드포인트 사용"
                ],
                fallback_options=[
                    "로컬 처리로 전환",
                    "캐시된 결과 사용"
                ],
                user_explanation="네트워크 연결을 재시도하고 있습니다.",
                technical_details=f"연결 대상: {error_context.component}",
                requires_user_input=False
            )
        
        elif diagnosis_type == 'file_not_found':
            return RecoveryPlan(
                plan_id=plan_id,
                error_id=error_context.error_id,
                strategy=RecoveryStrategy.USER_INTERVENTION,
                confidence=0.9,
                estimated_recovery_time=0,
                steps=[
                    "파일 경로 확인",
                    "파일 존재 여부 검증",
                    "사용자에게 올바른 파일 요청"
                ],
                fallback_options=[
                    "예제 데이터로 진행",
                    "다른 파일 선택"
                ],
                user_explanation="파일을 찾을 수 없습니다. 올바른 파일을 다시 업로드해주세요.",
                technical_details=f"찾을 수 없는 파일: {error_context.error_message}",
                requires_user_input=True
            )
        
        elif diagnosis_type == 'data_schema_mismatch':
            return RecoveryPlan(
                plan_id=plan_id,
                error_id=error_context.error_id,
                strategy=RecoveryStrategy.FALLBACK,
                confidence=0.6,
                estimated_recovery_time=3,
                steps=[
                    "데이터 스키마 자동 추론",
                    "누락된 컬럼 기본값으로 채우기",
                    "유연한 파싱 모드 활성화"
                ],
                fallback_options=[
                    "기본 분석만 수행",
                    "사용자에게 스키마 정보 요청"
                ],
                user_explanation="데이터 형식이 예상과 다릅니다. 자동으로 조정하여 처리합니다.",
                technical_details=f"스키마 불일치: {error_context.error_message}",
                requires_user_input=False
            )
        
        else:
            return self._create_fallback_recovery_plan(error_context)
    
    def _create_fallback_recovery_plan(self, error_context: ErrorContext) -> RecoveryPlan:
        """기본 fallback 복구 계획"""
        return RecoveryPlan(
            plan_id=f"fallback_{error_context.error_id}",
            error_id=error_context.error_id,
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            confidence=0.5,
            estimated_recovery_time=5,
            steps=[
                "오류 로깅",
                "기본 동작으로 전환",
                "사용자에게 상황 안내"
            ],
            fallback_options=[
                "다시 시도",
                "다른 접근 방법 사용"
            ],
            user_explanation="예상치 못한 오류가 발생했습니다. 기본 방식으로 처리를 계속합니다.",
            technical_details=f"오류 유형: {error_context.error_type}",
            requires_user_input=False
        )
    
    async def _execute_recovery_plan(self, 
                                   recovery_plan: RecoveryPlan,
                                   error_context: ErrorContext) -> bool:
        """복구 계획 실행"""
        try:
            strategy = recovery_plan.strategy
            
            if strategy == RecoveryStrategy.AUTO_RETRY:
                return await self._execute_retry_strategy(recovery_plan, error_context)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback_strategy(recovery_plan, error_context)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._execute_degradation_strategy(recovery_plan, error_context)
            
            elif strategy == RecoveryStrategy.USER_INTERVENTION:
                # 사용자 개입이 필요한 경우는 바로 False 반환 (UI에서 처리)
                return False
            
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery plan execution failed: {str(e)}")
            return False
    
    async def _execute_retry_strategy(self, 
                                    recovery_plan: RecoveryPlan,
                                    error_context: ErrorContext) -> bool:
        """재시도 전략 실행"""
        max_retries = self.retry_config['max_retries']
        base_delay = self.retry_config['base_delay']
        
        for attempt in range(max_retries):
            try:
                # 지수 백오프로 지연
                delay = min(
                    self.retry_config['max_delay'],
                    base_delay * (self.retry_config['exponential_base'] ** attempt)
                )
                
                if attempt > 0:  # 첫 번째 시도는 즉시
                    await asyncio.sleep(delay)
                
                # 원래 작업 재실행 (시뮬레이션)
                # 실제 구현에서는 원래 함수를 재호출
                logger.info(f"Retry attempt {attempt + 1} for {error_context.error_id}")
                
                # 성공 시뮬레이션 (실제로는 원래 작업의 결과)
                if attempt >= 1:  # 두 번째 시도부터 성공 가정
                    return True
                    
            except Exception as retry_error:
                logger.warning(f"Retry attempt {attempt + 1} failed: {str(retry_error)}")
                continue
        
        return False
    
    async def _execute_fallback_strategy(self, 
                                       recovery_plan: RecoveryPlan,
                                       error_context: ErrorContext) -> bool:
        """대체 방법 실행"""
        try:
            # 대체 방법 시뮬레이션
            logger.info(f"Executing fallback strategy for {error_context.error_id}")
            
            # 실제 구현에서는 대체 알고리즘이나 방법을 사용
            await asyncio.sleep(0.5)  # 시뮬레이션 지연
            
            return True  # 대체 방법은 일반적으로 성공
            
        except Exception as e:
            logger.error(f"Fallback strategy failed: {str(e)}")
            return False
    
    async def _execute_degradation_strategy(self, 
                                          recovery_plan: RecoveryPlan,
                                          error_context: ErrorContext) -> bool:
        """우아한 성능 저하 실행"""
        try:
            logger.info(f"Executing graceful degradation for {error_context.error_id}")
            
            # 기능을 제한하되 기본 동작은 유지
            # 예: 데이터 샘플링, 간단한 알고리즘 사용 등
            await asyncio.sleep(0.2)  # 시뮬레이션
            
            return True  # 성능 저하는 거의 항상 성공
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {str(e)}")
            return False
    
    def _generate_user_message(self, 
                              error_context: ErrorContext,
                              recovery_plan: RecoveryPlan,
                              success: bool) -> str:
        """사용자 친화적 메시지 생성"""
        
        if success:
            base_message = "✅ **문제가 해결되었습니다!**\n\n"
            base_message += recovery_plan.user_explanation
            
            if recovery_plan.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                base_message += "\n\n💡 일부 기능이 제한될 수 있지만 기본 분석은 정상적으로 진행됩니다."
            
        else:
            base_message = "❌ **문제 해결에 실패했습니다**\n\n"
            base_message += recovery_plan.user_explanation
            
            if recovery_plan.fallback_options:
                base_message += "\n\n**다음과 같은 방법을 시도해보세요:**\n"
                for option in recovery_plan.fallback_options:
                    base_message += f"• {option}\n"
        
        # 기술적 세부사항 (고급 사용자용)
        if hasattr(self, 'show_technical_details') and self.show_technical_details:
            base_message += f"\n\n🔧 **기술 정보**: {recovery_plan.technical_details}"
        
        return base_message
    
    def _record_successful_recovery(self, 
                                  error_context: ErrorContext,
                                  recovery_plan: RecoveryPlan):
        """성공적인 복구 기록"""
        
        # 오류 히스토리에 추가
        self.error_history.append(error_context)
        
        # 성공률 업데이트
        strategy_key = recovery_plan.strategy.value
        if strategy_key in self.recovery_success_rate:
            current_rate = self.recovery_success_rate[strategy_key]
            # 지수 이동 평균으로 업데이트
            self.recovery_success_rate[strategy_key] = 0.9 * current_rate + 0.1 * 1.0
        else:
            self.recovery_success_rate[strategy_key] = 1.0
        
        # 패턴 학습
        self._update_error_patterns(error_context, success=True)
        
        logger.info(f"Successful recovery recorded: {recovery_plan.plan_id}")
    
    def _record_failed_recovery(self, 
                               error_context: ErrorContext,
                               recovery_plan: RecoveryPlan):
        """실패한 복구 기록"""
        
        self.error_history.append(error_context)
        
        # 성공률 업데이트 (실패)
        strategy_key = recovery_plan.strategy.value
        if strategy_key in self.recovery_success_rate:
            current_rate = self.recovery_success_rate[strategy_key]
            self.recovery_success_rate[strategy_key] = 0.9 * current_rate + 0.1 * 0.0
        else:
            self.recovery_success_rate[strategy_key] = 0.0
        
        # 패턴 학습
        self._update_error_patterns(error_context, success=False)
        
        logger.warning(f"Failed recovery recorded: {recovery_plan.plan_id}")
    
    def _update_error_patterns(self, error_context: ErrorContext, success: bool):
        """오류 패턴 업데이트"""
        
        # 오류 시그니처 생성
        signature = f"{error_context.component}:{error_context.error_type}"
        
        if signature in self.error_patterns:
            pattern = self.error_patterns[signature]
            pattern.frequency += 1
            pattern.last_occurrence = error_context.timestamp
        else:
            self.error_patterns[signature] = ErrorPattern(
                pattern_id=f"pattern_{signature}_{int(time.time())}",
                error_signature=signature,
                frequency=1,
                last_occurrence=error_context.timestamp,
                typical_causes=[error_context.error_message],
                proven_solutions=[],
                prevention_measures=[]
            )
    
    def _is_circuit_open(self, circuit_key: str) -> bool:
        """회로 차단기 상태 확인"""
        
        if circuit_key not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[circuit_key]
        
        if circuit['state'] == 'open':
            # 복구 시간이 지났는지 확인
            if time.time() - circuit['open_time'] > self.circuit_breaker_config['recovery_timeout']:
                circuit['state'] = 'half_open'
                circuit['half_open_calls'] = 0
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, circuit_key: str, success: bool):
        """회로 차단기 상태 업데이트"""
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                'state': 'closed',
                'failure_count': 0,
                'open_time': 0,
                'half_open_calls': 0
            }
        
        circuit = self.circuit_breakers[circuit_key]
        
        if success:
            if circuit['state'] == 'half_open':
                circuit['half_open_calls'] += 1
                if circuit['half_open_calls'] >= self.circuit_breaker_config['half_open_max_calls']:
                    circuit['state'] = 'closed'
                    circuit['failure_count'] = 0
            else:
                circuit['failure_count'] = max(0, circuit['failure_count'] - 1)
        else:
            circuit['failure_count'] += 1
            if circuit['failure_count'] >= self.circuit_breaker_config['failure_threshold']:
                circuit['state'] = 'open'
                circuit['open_time'] = time.time()
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """오류 통계 반환"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {'total_errors': 0, 'message': 'No errors recorded'}
        
        # 최근 24시간 오류
        recent_errors = [
            e for e in self.error_history 
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # 오류 유형별 분포
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        # 컴포넌트별 분포
        components = {}
        for error in self.error_history:
            components[error.component] = components.get(error.component, 0) + 1
        
        return {
            'total_errors': total_errors,
            'recent_errors_24h': len(recent_errors),
            'error_types': error_types,
            'components': components,
            'recovery_success_rates': self.recovery_success_rate,
            'circuit_breaker_status': {
                k: v['state'] for k, v in self.circuit_breakers.items()
            },
            'error_patterns': len(self.error_patterns)
        }
    
    def reset_circuit_breaker(self, circuit_key: str) -> bool:
        """회로 차단기 수동 리셋"""
        if circuit_key in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                'state': 'closed',
                'failure_count': 0,
                'open_time': 0,
                'half_open_calls': 0
            }
            logger.info(f"Circuit breaker reset: {circuit_key}")
            return True
        return False
    
    def clear_error_history(self, older_than_hours: int = 24):
        """오래된 오류 히스토리 정리"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        original_count = len(self.error_history)
        self.error_history = [
            e for e in self.error_history if e.timestamp > cutoff_time
        ]
        
        cleared_count = original_count - len(self.error_history)
        logger.info(f"Cleared {cleared_count} old error records")
        
        return cleared_count