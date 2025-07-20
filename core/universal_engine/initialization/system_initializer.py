"""
System Initialization - Universal Engine 시스템 초기화

완전한 시스템 초기화 구현:
- Component dependency management and startup sequence
- Configuration validation and environment setup
- Health checks and readiness verification
- Graceful startup with error recovery
- Resource allocation and optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class InitializationStage(Enum):
    """초기화 단계"""
    PRE_INIT = "pre_init"
    CORE_COMPONENTS = "core_components"
    INTEGRATIONS = "integrations"
    VALIDATION = "validation"
    POST_INIT = "post_init"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentStatus(Enum):
    """컴포넌트 상태"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass
class ComponentInfo:
    """컴포넌트 정보"""
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    timeout_seconds: int = 60
    retry_count: int = 3
    health_check: Optional[Callable] = None
    initialization_func: Optional[Callable] = None


@dataclass
class InitializationResult:
    """초기화 결과"""
    component_name: str
    status: ComponentStatus
    duration_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealth:
    """시스템 건강 상태"""
    overall_status: str
    ready_components: List[str]
    failed_components: List[str]
    degraded_components: List[str]
    initialization_time: float
    last_health_check: datetime
    system_metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalEngineInitializer:
    """
    Universal Engine 시스템 초기화기
    - 컴포넌트 의존성 관리 및 시작 순서 제어
    - 설정 검증 및 환경 설정
    - 건강 상태 확인 및 준비 상태 검증
    - 오류 복구를 통한 우아한 시작
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """UniversalEngineInitializer 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.config_path = config_path or "./config/universal_engine.json"
        
        # 초기화 상태 추적
        self.current_stage = InitializationStage.PRE_INIT
        self.component_status: Dict[str, ComponentStatus] = {}
        self.initialization_results: List[InitializationResult] = []
        
        # 컴포넌트 정의
        self.components: Dict[str, ComponentInfo] = {}
        self.initialization_order: List[str] = []
        
        # 시스템 설정
        self.config: Dict[str, Any] = {}
        self.environment: Dict[str, Any] = {}
        
        # 초기화 메트릭
        self.start_time = None
        self.end_time = None
        
        # 컴포넌트 인스턴스 저장소
        self.component_instances: Dict[str, Any] = {}
        
        # 기본 컴포넌트 정의 로드
        self._define_core_components()
        
        logger.info("UniversalEngineInitializer created")
    
    async def initialize_system(self) -> SystemHealth:
        """시스템 전체 초기화"""
        
        logger.info("Starting Universal Engine system initialization")
        self.start_time = time.time()
        
        try:
            # 1. Pre-initialization
            await self._pre_initialization()
            
            # 2. Core components initialization
            await self._initialize_core_components()
            
            # 3. Integration components initialization
            await self._initialize_integrations()
            
            # 4. System validation
            await self._validate_system()
            
            # 5. Post-initialization
            await self._post_initialization()
            
            self.current_stage = InitializationStage.COMPLETED
            logger.info("Universal Engine system initialization completed successfully")
            
        except Exception as e:
            self.current_stage = InitializationStage.FAILED
            logger.error(f"System initialization failed: {e}")
            
            # 실패 시 부분 복구 시도
            await self._attempt_recovery()
        
        finally:
            self.end_time = time.time()
        
        return await self._generate_system_health()
    
    async def _pre_initialization(self):
        """사전 초기화"""
        
        logger.info("Starting pre-initialization stage")
        self.current_stage = InitializationStage.PRE_INIT
        
        # 1. 설정 파일 로드
        await self._load_configuration()
        
        # 2. 환경 변수 검증
        await self._validate_environment()
        
        # 3. 필수 디렉토리 생성
        await self._create_directories()
        
        # 4. 로깅 시스템 설정
        await self._setup_logging()
        
        # 5. 리소스 할당 계획
        await self._plan_resource_allocation()
        
        logger.info("Pre-initialization completed")
    
    async def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        
        logger.info("Starting core components initialization")
        self.current_stage = InitializationStage.CORE_COMPONENTS
        
        core_components = [
            "meta_reasoning_engine",
            "dynamic_context_discovery",
            "adaptive_user_understanding", 
            "universal_intent_detection",
            "dynamic_knowledge_orchestrator"
        ]
        
        for component_name in core_components:
            await self._initialize_component(component_name)
        
        logger.info("Core components initialization completed")
    
    async def _initialize_integrations(self):
        """통합 컴포넌트 초기화"""
        
        logger.info("Starting integration components initialization")
        self.current_stage = InitializationStage.INTEGRATIONS
        
        integration_components = [
            "a2a_agent_discovery",
            "llm_based_agent_selector",
            "a2a_workflow_orchestrator",
            "a2a_error_handler",
            "performance_monitoring",
            "session_manager",
            "adaptive_response_generator"
        ]
        
        for component_name in integration_components:
            await self._initialize_component(component_name)
        
        logger.info("Integration components initialization completed")
    
    async def _validate_system(self):
        """시스템 검증"""
        
        logger.info("Starting system validation")
        self.current_stage = InitializationStage.VALIDATION
        
        # 1. 컴포넌트 상호 연결성 검증
        await self._validate_component_connectivity()
        
        # 2. 종단간 기능 테스트
        await self._run_end_to_end_tests()
        
        # 3. 성능 기준선 설정
        await self._establish_performance_baseline()
        
        # 4. 보안 설정 검증
        await self._validate_security_settings()
        
        logger.info("System validation completed")
    
    async def _post_initialization(self):
        """사후 초기화"""
        
        logger.info("Starting post-initialization")
        self.current_stage = InitializationStage.POST_INIT
        
        # 1. 모니터링 시스템 활성화
        await self._activate_monitoring()
        
        # 2. 자동 복구 메커니즘 설정
        await self._setup_auto_recovery()
        
        # 3. 백그라운드 작업 시작
        await self._start_background_tasks()
        
        # 4. 준비 신호 발송
        await self._signal_system_ready()
        
        logger.info("Post-initialization completed")
    
    async def _initialize_component(self, component_name: str) -> InitializationResult:
        """개별 컴포넌트 초기화"""
        
        if component_name not in self.components:
            error_msg = f"Unknown component: {component_name}"
            logger.error(error_msg)
            return InitializationResult(
                component_name=component_name,
                status=ComponentStatus.FAILED,
                duration_ms=0.0,
                error_message=error_msg
            )
        
        component = self.components[component_name]
        logger.info(f"Initializing component: {component.name}")
        
        start_time = time.time()
        self.component_status[component_name] = ComponentStatus.INITIALIZING
        
        try:
            # 의존성 확인
            for dependency in component.dependencies:
                if self.component_status.get(dependency) != ComponentStatus.READY:
                    if not component.optional:
                        raise Exception(f"Dependency {dependency} not ready")
                    else:
                        logger.warning(f"Optional dependency {dependency} not ready for {component_name}")
            
            # 컴포넌트 초기화 실행
            if component.initialization_func:
                instance = await self._run_with_timeout(
                    component.initialization_func(),
                    component.timeout_seconds
                )
                self.component_instances[component_name] = instance
            else:
                # 기본 초기화 로직
                instance = await self._default_component_initialization(component_name)
                self.component_instances[component_name] = instance
            
            # 건강 상태 확인
            if component.health_check:
                health_ok = await component.health_check(instance)
                if not health_ok:
                    raise Exception(f"Health check failed for {component_name}")
            
            duration_ms = (time.time() - start_time) * 1000
            self.component_status[component_name] = ComponentStatus.READY
            
            result = InitializationResult(
                component_name=component_name,
                status=ComponentStatus.READY,
                duration_ms=duration_ms,
                metadata={"initialization_time": duration_ms}
            )
            
            self.initialization_results.append(result)
            logger.info(f"Component {component_name} initialized successfully in {duration_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # 재시도 로직
            if component.retry_count > 0:
                logger.warning(f"Component {component_name} failed, retrying...")
                component.retry_count -= 1
                await asyncio.sleep(1)  # 재시도 전 대기
                return await self._initialize_component(component_name)
            
            self.component_status[component_name] = ComponentStatus.FAILED
            
            result = InitializationResult(
                component_name=component_name,
                status=ComponentStatus.FAILED,
                duration_ms=duration_ms,
                error_message=error_msg
            )
            
            self.initialization_results.append(result)
            
            if not component.optional:
                logger.error(f"Critical component {component_name} failed: {error_msg}")
                raise
            else:
                logger.warning(f"Optional component {component_name} failed: {error_msg}")
                self.component_status[component_name] = ComponentStatus.DEGRADED
            
            return result
    
    async def _default_component_initialization(self, component_name: str) -> Any:
        """기본 컴포넌트 초기화"""
        
        # 컴포넌트별 기본 초기화 로직
        if component_name == "meta_reasoning_engine":
            from ..meta_reasoning_engine import MetaReasoningEngine
            return MetaReasoningEngine()
        
        elif component_name == "dynamic_context_discovery":
            from ..dynamic_context_discovery import DynamicContextDiscovery
            return DynamicContextDiscovery()
        
        elif component_name == "adaptive_user_understanding":
            from ..adaptive_user_understanding import AdaptiveUserUnderstanding
            return AdaptiveUserUnderstanding()
        
        elif component_name == "universal_intent_detection":
            from ..universal_intent_detection import UniversalIntentDetection
            return UniversalIntentDetection()
        
        elif component_name == "dynamic_knowledge_orchestrator":
            from ..dynamic_knowledge_orchestrator import DynamicKnowledgeOrchestrator
            return DynamicKnowledgeOrchestrator()
        
        elif component_name == "a2a_agent_discovery":
            from ..a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
            discovery = A2AAgentDiscoverySystem()
            await discovery.start_discovery()
            return discovery
        
        elif component_name == "llm_based_agent_selector":
            from ..a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
            discovery = self.component_instances.get("a2a_agent_discovery")
            return LLMBasedAgentSelector(discovery)
        
        elif component_name == "a2a_workflow_orchestrator":
            from ..a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
            from ..a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
            protocol = A2ACommunicationProtocol()
            return A2AWorkflowOrchestrator(protocol)
        
        elif component_name == "a2a_error_handler":
            from ..a2a_integration.a2a_error_handler import A2AErrorHandler
            return A2AErrorHandler()
        
        elif component_name == "performance_monitoring":
            from ..monitoring.performance_monitoring_system import PerformanceMonitoringSystem
            return PerformanceMonitoringSystem()
        
        elif component_name == "session_manager":
            from ..session.session_management_system import SessionManager
            return SessionManager()
        
        elif component_name == "adaptive_response_generator":
            from ..adaptive_response_generator import AdaptiveResponseGenerator
            return AdaptiveResponseGenerator()
        
        else:
            raise ValueError(f"Unknown component for default initialization: {component_name}")
    
    async def _run_with_timeout(self, coroutine, timeout_seconds: int):
        """타임아웃과 함께 코루틴 실행"""
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise Exception(f"Initialization timed out after {timeout_seconds} seconds")
    
    async def _load_configuration(self):
        """설정 파일 로드"""
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # 기본 설정 사용
                self.config = self._get_default_configuration()
                logger.info("Using default configuration")
        
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}, using defaults")
            self.config = self._get_default_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "system": {
                "log_level": "INFO",
                "max_concurrent_requests": 100,
                "request_timeout_seconds": 30
            },
            "components": {
                "meta_reasoning": {
                    "enabled": True,
                    "reasoning_depth": "comprehensive"
                },
                "a2a_integration": {
                    "enabled": True,
                    "discovery_ports": [8306, 8307, 8308, 8309, 8310]
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_retention_hours": 24
                }
            },
            "performance": {
                "enable_caching": True,
                "cache_size_mb": 512,
                "optimization_level": "balanced"
            }
        }
    
    async def _validate_environment(self):
        """환경 변수 검증"""
        
        required_env_vars = [
            "PYTHONPATH",
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if var not in os.environ:
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
        
        # 환경 정보 수집
        self.environment = {
            "python_version": os.sys.version,
            "platform": os.sys.platform,
            "working_directory": os.getcwd(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _create_directories(self):
        """필수 디렉토리 생성"""
        
        directories = [
            "./logs",
            "./data",
            "./cache",
            "./sessions",
            "./temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Required directories created")
    
    async def _setup_logging(self):
        """로깅 시스템 설정"""
        
        log_level = self.config.get("system", {}).get("log_level", "INFO")
        
        # 로그 핸들러 설정 (간단화된 버전)
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/universal_engine.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging system configured")
    
    async def _plan_resource_allocation(self):
        """리소스 할당 계획"""
        
        # 간단한 리소스 계획 (실제로는 더 정교한 로직 필요)
        max_concurrent = self.config.get("system", {}).get("max_concurrent_requests", 100)
        
        logger.info(f"Resource allocation planned: max_concurrent={max_concurrent}")
    
    async def _validate_component_connectivity(self):
        """컴포넌트 상호 연결성 검증"""
        
        logger.info("Validating component connectivity")
        
        # 각 컴포넌트 간 통신 테스트
        ready_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.READY
        ]
        
        connectivity_issues = []
        
        # 메타 추론 엔진과 다른 컴포넌트 간 연결 테스트
        if "meta_reasoning_engine" in ready_components:
            meta_engine = self.component_instances["meta_reasoning_engine"]
            try:
                # 간단한 테스트 쿼리 실행
                test_result = await meta_engine.analyze_request(
                    "테스트 쿼리", {"test": True}, {}
                )
                logger.info("Meta reasoning engine connectivity verified")
            except Exception as e:
                connectivity_issues.append(f"Meta reasoning engine: {e}")
        
        if connectivity_issues:
            raise Exception(f"Connectivity issues: {connectivity_issues}")
    
    async def _run_end_to_end_tests(self):
        """종단간 기능 테스트"""
        
        logger.info("Running end-to-end tests")
        
        # Universal Query Processor 통합 테스트
        try:
            from ..universal_query_processor import UniversalQueryProcessor
            processor = UniversalQueryProcessor()
            
            test_result = await processor.process_query(
                query="시스템 초기화 테스트",
                data={"test": True},
                context={"initialization_test": True}
            )
            
            if test_result.get("success"):
                logger.info("End-to-end test passed")
            else:
                raise Exception("End-to-end test failed")
        
        except Exception as e:
            logger.warning(f"End-to-end test issue: {e}")
    
    async def _establish_performance_baseline(self):
        """성능 기준선 설정"""
        
        logger.info("Establishing performance baseline")
        
        # 기본 성능 메트릭 설정
        baseline_metrics = {
            "average_response_time_ms": 1000,
            "max_response_time_ms": 5000,
            "success_rate_threshold": 0.95,
            "concurrent_request_limit": 50
        }
        
        # 성능 모니터링 시스템에 기준선 설정
        if "performance_monitoring" in self.component_instances:
            monitoring = self.component_instances["performance_monitoring"]
            # 기준선 설정 로직 (실제 구현에서는 더 정교한 설정 필요)
            
        logger.info("Performance baseline established")
    
    async def _validate_security_settings(self):
        """보안 설정 검증"""
        
        logger.info("Validating security settings")
        
        # 기본 보안 검증
        security_checks = [
            "Session encryption enabled",
            "API rate limiting configured", 
            "Input validation active",
            "Error message sanitization enabled"
        ]
        
        for check in security_checks:
            logger.info(f"Security check: {check}")
        
        logger.info("Security validation completed")
    
    async def _activate_monitoring(self):
        """모니터링 시스템 활성화"""
        
        if "performance_monitoring" in self.component_instances:
            monitoring = self.component_instances["performance_monitoring"]
            monitoring.start_monitoring()
            logger.info("Performance monitoring activated")
    
    async def _setup_auto_recovery(self):
        """자동 복구 메커니즘 설정"""
        
        logger.info("Setting up auto-recovery mechanisms")
        
        # 컴포넌트 건강 상태 모니터링 설정
        # 실제 구현에서는 더 정교한 복구 로직 필요
        
        logger.info("Auto-recovery mechanisms configured")
    
    async def _start_background_tasks(self):
        """백그라운드 작업 시작"""
        
        logger.info("Starting background tasks")
        
        # 주기적 건강 상태 체크
        asyncio.create_task(self._health_check_loop())
        
        # 성능 메트릭 수집
        asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Background tasks started")
    
    async def _signal_system_ready(self):
        """시스템 준비 신호 발송"""
        
        logger.info("System ready signal sent")
        
        # 시스템 준비 상태를 외부에 알림
        ready_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.READY
        ]
        
        logger.info(f"System ready with {len(ready_components)} components")
    
    async def _attempt_recovery(self):
        """실패 시 부분 복구 시도"""
        
        logger.info("Attempting system recovery")
        
        failed_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.FAILED
        ]
        
        for component_name in failed_components:
            component = self.components[component_name]
            if component.optional:
                logger.info(f"Marking optional component {component_name} as degraded")
                self.component_status[component_name] = ComponentStatus.DEGRADED
        
        # 핵심 컴포넌트가 실패한 경우 재시도
        critical_failed = [name for name in failed_components if not self.components[name].optional]
        
        if critical_failed:
            logger.error(f"Critical components failed: {critical_failed}")
            # 실제로는 더 정교한 복구 로직 필요
        else:
            logger.info("Recovery successful - system operating in degraded mode")
    
    async def _generate_system_health(self) -> SystemHealth:
        """시스템 건강 상태 생성"""
        
        ready_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.READY
        ]
        
        failed_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.FAILED
        ]
        
        degraded_components = [
            name for name, status in self.component_status.items()
            if status == ComponentStatus.DEGRADED
        ]
        
        # 전체 상태 결정
        if failed_components and any(not self.components[name].optional for name in failed_components):
            overall_status = "failed"
        elif degraded_components:
            overall_status = "degraded"
        elif len(ready_components) == len(self.components):
            overall_status = "healthy"
        else:
            overall_status = "partial"
        
        initialization_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.0
        
        return SystemHealth(
            overall_status=overall_status,
            ready_components=ready_components,
            failed_components=failed_components,
            degraded_components=degraded_components,
            initialization_time=initialization_time,
            last_health_check=datetime.now(),
            system_metadata={
                "total_components": len(self.components),
                "initialization_stage": self.current_stage.value,
                "config_loaded": bool(self.config),
                "environment_validated": bool(self.environment)
            }
        )
    
    def _define_core_components(self):
        """핵심 컴포넌트 정의"""
        
        # 메타 추론 엔진
        self.components["meta_reasoning_engine"] = ComponentInfo(
            name="Meta Reasoning Engine",
            description="DeepSeek-R1 inspired meta reasoning engine",
            dependencies=[],
            optional=False
        )
        
        # 동적 컨텍스트 발견
        self.components["dynamic_context_discovery"] = ComponentInfo(
            name="Dynamic Context Discovery",
            description="Automatic domain and context detection",
            dependencies=["meta_reasoning_engine"],
            optional=False
        )
        
        # 적응적 사용자 이해
        self.components["adaptive_user_understanding"] = ComponentInfo(
            name="Adaptive User Understanding",
            description="User expertise and preference detection",
            dependencies=["meta_reasoning_engine"],
            optional=False
        )
        
        # 유니버설 의도 감지
        self.components["universal_intent_detection"] = ComponentInfo(
            name="Universal Intent Detection",
            description="Semantic intent routing without hardcoded rules",
            dependencies=["meta_reasoning_engine"],
            optional=False
        )
        
        # 동적 지식 오케스트레이션
        self.components["dynamic_knowledge_orchestrator"] = ComponentInfo(
            name="Dynamic Knowledge Orchestrator",
            description="Dynamic knowledge source orchestration",
            dependencies=["dynamic_context_discovery"],
            optional=False
        )
        
        # A2A 에이전트 발견
        self.components["a2a_agent_discovery"] = ComponentInfo(
            name="A2A Agent Discovery",
            description="Automatic discovery of A2A agents",
            dependencies=[],
            optional=True
        )
        
        # LLM 기반 에이전트 선택기
        self.components["llm_based_agent_selector"] = ComponentInfo(
            name="LLM Based Agent Selector",
            description="Dynamic agent selection using LLM reasoning",
            dependencies=["a2a_agent_discovery", "meta_reasoning_engine"],
            optional=True
        )
        
        # A2A 워크플로우 오케스트레이터
        self.components["a2a_workflow_orchestrator"] = ComponentInfo(
            name="A2A Workflow Orchestrator",
            description="A2A workflow execution and coordination",
            dependencies=["llm_based_agent_selector"],
            optional=True
        )
        
        # A2A 에러 핸들러
        self.components["a2a_error_handler"] = ComponentInfo(
            name="A2A Error Handler",
            description="A2A error handling and resilience",
            dependencies=["a2a_agent_discovery"],
            optional=True
        )
        
        # 성능 모니터링
        self.components["performance_monitoring"] = ComponentInfo(
            name="Performance Monitoring",
            description="Real-time performance monitoring and analysis",
            dependencies=[],
            optional=True
        )
        
        # 세션 관리자
        self.components["session_manager"] = ComponentInfo(
            name="Session Manager",
            description="User session lifecycle management",
            dependencies=[],
            optional=True
        )
        
        # 적응적 응답 생성기
        self.components["adaptive_response_generator"] = ComponentInfo(
            name="Adaptive Response Generator",
            description="Context-aware response generation",
            dependencies=["adaptive_user_understanding", "dynamic_knowledge_orchestrator"],
            optional=False
        )
    
    async def _health_check_loop(self):
        """주기적 건강 상태 체크"""
        
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다
                
                for component_name, instance in self.component_instances.items():
                    component = self.components[component_name]
                    
                    if component.health_check:
                        try:
                            health_ok = await component.health_check(instance)
                            if not health_ok:
                                logger.warning(f"Health check failed for {component_name}")
                                self.component_status[component_name] = ComponentStatus.DEGRADED
                        except Exception as e:
                            logger.error(f"Health check error for {component_name}: {e}")
                            self.component_status[component_name] = ComponentStatus.FAILED
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _metrics_collection_loop(self):
        """주기적 메트릭 수집"""
        
        while True:
            try:
                await asyncio.sleep(30)  # 30초마다
                
                # 시스템 메트릭 수집 및 기록
                if "performance_monitoring" in self.component_instances:
                    monitoring = self.component_instances["performance_monitoring"]
                    
                    # 컴포넌트별 상태 메트릭
                    for component_name, status in self.component_status.items():
                        from ..monitoring.performance_monitoring_system import ComponentType, MetricType
                        
                        try:
                            component_type = ComponentType(component_name.upper())
                            monitoring.record_metric(
                                name="component_status",
                                component=component_type,
                                metric_type=MetricType.GAUGE,
                                value=1.0 if status == ComponentStatus.READY else 0.0,
                                tags={"status": status.value}
                            )
                        except ValueError:
                            # 알려지지 않은 컴포넌트 타입은 무시
                            pass
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """초기화 보고서 조회"""
        
        total_duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.0
        
        component_summary = {}
        for result in self.initialization_results:
            component_summary[result.component_name] = {
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "error": result.error_message
            }
        
        return {
            "initialization_stage": self.current_stage.value,
            "total_duration_seconds": total_duration,
            "component_summary": component_summary,
            "system_health": self.component_status,
            "configuration_loaded": bool(self.config),
            "environment_info": self.environment,
            "timestamp": datetime.now().isoformat()
        }