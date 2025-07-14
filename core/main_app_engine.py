#!/usr/bin/env python3
"""
🚀 CherryAI Main Application Engine

모든 비즈니스 로직을 담당하는 핵심 엔진
main.py에서 비즈니스 로직을 분리하여 테스트 가능하고 LLM First 원칙을 준수

Key Features:
- A2A + MCP 통합 처리
- 실시간 스트리밍 관리
- 파일 처리 및 데이터 준비
- 쿼리 처리 및 응답 생성
- 에이전트 오케스트레이션
- 에러 처리 및 복구

Architecture:
- Business Layer: 핵심 비즈니스 로직
- Integration Layer: A2A + MCP 통합
- Processing Layer: 데이터 및 쿼리 처리
- Orchestration Layer: 에이전트 협업 관리
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

# 프로젝트 임포트
from core.app_components.main_app_controller import (
    initialize_app_controller, 
    get_app_controller,
    SystemStatus
)
from core.app_components.realtime_streaming_handler import (
    get_streaming_handler,
    process_query_with_streaming
)
from core.app_components.file_upload_processor import (
    get_file_upload_processor,
    process_and_prepare_files_for_a2a
)
from core.shared_knowledge_bank import (
    get_shared_knowledge_bank,
    add_user_file_knowledge,
    search_relevant_knowledge
)
from core.llm_first_engine import (
    get_llm_first_engine,
    analyze_intent,
    make_decision,
    assess_quality,
    DecisionType,
    UserIntent,
    DynamicDecision,
    QualityAssessment
)

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """처리 단계"""
    INITIALIZATION = "initialization"
    INTENT_ANALYSIS = "intent_analysis"
    FILE_PROCESSING = "file_processing"
    AGENT_SELECTION = "agent_selection"
    EXECUTION = "execution"
    QUALITY_CHECK = "quality_check"
    COMPLETION = "completion"

@dataclass
class ProcessingContext:
    """처리 컨텍스트"""
    request_id: str
    user_input: str
    uploaded_files: List[Any] = field(default_factory=list)
    processed_files: List[Any] = field(default_factory=list)
    user_intent: Optional[UserIntent] = None
    agent_decision: Optional[DynamicDecision] = None
    execution_result: Optional[str] = None
    quality_assessment: Optional[QualityAssessment] = None
    stage: ProcessingStage = ProcessingStage.INITIALIZATION
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """처리 결과"""
    request_id: str
    success: bool
    response: str
    processing_time: float
    stages_completed: List[ProcessingStage]
    quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CherryAIMainEngine:
    """
    🚀 CherryAI 메인 애플리케이션 엔진
    
    LLM First 원칙을 준수하며 모든 비즈니스 로직을 처리
    """
    
    def __init__(self, 
                 config_manager = None,
                 session_manager = None):
        """
        메인 엔진 초기화
        
        Args:
            config_manager: 설정 관리자
            session_manager: 세션 관리자
        """
        self.config_manager = config_manager
        self.session_manager = session_manager
        
        # 엔진 상태
        self.engine_state = {
            "initialized": False,
            "last_health_check": None,
            "total_requests": 0,
            "successful_requests": 0,
            "error_count": 0,
            "average_processing_time": 0.0
        }
        
        # 컴포넌트 초기화
        self.app_controller = None
        self.streaming_handler = None
        self.file_processor = None
        self.knowledge_bank = None
        self.llm_engine = None
        
        logger.info("🚀 CherryAI Main Engine 초기화 시작")

    async def initialize(self) -> bool:
        """엔진 초기화"""
        try:
            # 컴포넌트 초기화
            self.app_controller = initialize_app_controller()
            self.streaming_handler = get_streaming_handler()
            self.file_processor = get_file_upload_processor()
            self.knowledge_bank = get_shared_knowledge_bank()
            self.llm_engine = get_llm_first_engine()
            
            # 상태 업데이트
            self.engine_state["initialized"] = True
            self.engine_state["last_health_check"] = datetime.now()
            
            logger.info("✅ CherryAI Main Engine 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 엔진 초기화 실패: {e}")
            return False

    async def process_user_request(self, 
                                 user_input: str, 
                                 uploaded_files: List[Any] = None) -> AsyncGenerator[str, None]:
        """
        사용자 요청 처리 (LLM First)
        
        Args:
            user_input: 사용자 입력
            uploaded_files: 업로드된 파일들
            
        Yields:
            처리 과정의 실시간 스트리밍 응답
        """
        # 처리 컨텍스트 생성
        context = ProcessingContext(
            request_id=str(uuid.uuid4()),
            user_input=user_input,
            uploaded_files=uploaded_files or []
        )
        
        self.engine_state["total_requests"] += 1
        
        try:
            # Stage 1: 사용자 의도 분석 (LLM First)
            yield "🧠 사용자 의도를 분석하고 있습니다..."
            context = await self._analyze_user_intent(context)
            yield f"✅ 의도 분석 완료: {context.user_intent.primary_intent}"
            
            # Stage 2: 파일 처리 (업로드된 파일이 있는 경우)
            if context.uploaded_files:
                yield f"📁 {len(context.uploaded_files)}개 파일을 처리하고 있습니다..."
                context = await self._process_files(context)
                yield f"✅ 파일 처리 완료: {len(context.processed_files)}개 파일 준비됨"
            
            # Stage 3: 에이전트 선택 (LLM First)
            yield "🤖 최적의 에이전트를 선택하고 있습니다..."
            context = await self._select_agents(context)
            yield f"✅ 에이전트 선택 완료: {context.agent_decision.decision}"
            
            # Stage 4: 실행
            yield "🚀 A2A + MCP 하이브리드 시스템이 작업을 수행하고 있습니다..."
            async for execution_chunk in self._execute_with_agents(context):
                context.execution_result = execution_chunk
                yield execution_chunk
            
            # Stage 5: 품질 검증 (LLM First)
            yield "📊 결과 품질을 검증하고 있습니다..."
            context = await self._assess_quality(context)
            
            # 최종 결과 반환
            final_result = await self._finalize_result(context)
            
            if final_result.success:
                self.engine_state["successful_requests"] += 1
            else:
                self.engine_state["error_count"] += 1
            
            # 성능 메트릭 업데이트
            processing_time = (datetime.now() - context.start_time).total_seconds()
            self._update_performance_metrics(processing_time)
            
        except Exception as e:
            error_msg = f"❌ 처리 중 오류 발생: {str(e)}"
            logger.error(f"사용자 요청 처리 오류: {e}")
            self.engine_state["error_count"] += 1
            yield error_msg

    async def _analyze_user_intent(self, context: ProcessingContext) -> ProcessingContext:
        """사용자 의도 분석 (LLM First)"""
        context.stage = ProcessingStage.INTENT_ANALYSIS
        
        # 컨텍스트 구성
        analysis_context = {
            "has_files": len(context.uploaded_files) > 0,
            "file_count": len(context.uploaded_files),
            "platform": "CherryAI",
            "capabilities": ["data_analysis", "visualization", "ml", "web_scraping"]
        }
        
        # LLM First 의도 분석
        context.user_intent = await analyze_intent(context.user_input, analysis_context)
        
        # 지식 뱅크에 쿼리 저장
        await self.knowledge_bank.add_knowledge(
            content=context.user_input,
            knowledge_type="user_query",
            source_agent="user",
            title=f"사용자 요청 #{context.request_id[:8]}",
            metadata={"intent": context.user_intent.primary_intent}
        )
        
        return context

    async def _process_files(self, context: ProcessingContext) -> ProcessingContext:
        """파일 처리"""
        context.stage = ProcessingStage.FILE_PROCESSING
        
        try:
            # 파일 처리
            context.processed_files = process_and_prepare_files_for_a2a(context.uploaded_files)
            
            # 지식 뱅크에 파일 정보 저장
            for i, file in enumerate(context.uploaded_files):
                file_content = file.getvalue().decode('utf-8') if hasattr(file, 'getvalue') else str(file)
                await add_user_file_knowledge(
                    file_content=file_content[:1000],  # 첫 1000자만 저장
                    filename=getattr(file, 'name', f'file_{i}'),
                    session_id=context.request_id
                )
            
        except Exception as e:
            logger.error(f"파일 처리 오류: {e}")
            context.metadata["file_processing_error"] = str(e)
        
        return context

    async def _select_agents(self, context: ProcessingContext) -> ProcessingContext:
        """에이전트 선택 (LLM First)"""
        context.stage = ProcessingStage.AGENT_SELECTION
        
        # 사용 가능한 에이전트 정보
        available_agents = [
            "pandas_collaboration_hub",
            "data_cleaning",
            "data_loader", 
            "data_visualization",
            "eda_tools",
            "feature_engineering",
            "h2o_ml",
            "mlflow_tools",
            "sql_database",
            "data_wrangling"
        ]
        
        # 의사결정 컨텍스트
        decision_context = {
            "user_intent": context.user_intent.primary_intent,
            "complexity": context.user_intent.complexity_level,
            "has_data": len(context.processed_files) > 0,
            "data_requirements": context.user_intent.data_requirements,
            "expected_outputs": context.user_intent.expected_outputs
        }
        
        # LLM First 에이전트 선택
        context.agent_decision = await make_decision(
            DecisionType.AGENT_SELECTION,
            decision_context,
            available_agents
        )
        
        return context

    async def _execute_with_agents(self, context: ProcessingContext) -> AsyncGenerator[str, None]:
        """에이전트와 함께 실행 - 실제 SSE 스트리밍 통합"""
        context.stage = ProcessingStage.EXECUTION
        
        try:
            # Unified Message Broker로 실제 SSE 스트리밍 처리
            from core.streaming.unified_message_broker import get_unified_message_broker
            broker = get_unified_message_broker()
            
            # 세션 생성
            session_id = await broker.create_session(context.user_input)
            
            # 멀티 에이전트 오케스트레이션으로 실시간 스트리밍
            async for broker_event in broker.orchestrate_multi_agent_query(
                session_id=session_id,
                user_query=context.user_input,
                required_capabilities=context.agent_decision.required_capabilities if context.agent_decision else None
            ):
                # 브로커 이벤트를 UI용 텍스트로 변환
                if broker_event.get('event') == 'orchestration_start':
                    agents = broker_event.get('data', {}).get('selected_agents', [])
                    yield f"🤖 선택된 에이전트: {', '.join(agents)}"
                    
                elif broker_event.get('event') == 'agent_response':
                    data = broker_event.get('data', {})
                    agent_name = data.get('agent', 'unknown')
                    content = data.get('content', '')
                    
                    if content.strip():
                        yield f"📊 {agent_name}: {content}"
                        
                elif broker_event.get('event') == 'stream_chunk':
                    chunk_content = broker_event.get('data', {}).get('content', '')
                    if chunk_content.strip():
                        yield chunk_content
                        
                elif broker_event.get('event') == 'error':
                    error_msg = broker_event.get('data', {}).get('error', 'Unknown error')
                    yield f"❌ 오류: {error_msg}"
                    
        except Exception as e:
            error_msg = f"실행 오류: {str(e)}"
            logger.error(f"에이전트 실행 오류: {e}")
            yield error_msg

    async def _assess_quality(self, context: ProcessingContext) -> ProcessingContext:
        """품질 평가 (LLM First)"""
        context.stage = ProcessingStage.QUALITY_CHECK
        
        if context.execution_result:
            # 품질 평가 기준
            quality_criteria = [
                "사용자 요청 충족도",
                "결과의 정확성",
                "설명의 명확성",
                "실용적 가치",
                "완전성"
            ]
            
            # 평가 컨텍스트
            assessment_context = {
                "user_intent": context.user_intent.primary_intent,
                "original_request": context.user_input,
                "processing_time": (datetime.now() - context.start_time).total_seconds()
            }
            
            # LLM First 품질 평가
            context.quality_assessment = await assess_quality(
                context.execution_result,
                quality_criteria,
                assessment_context
            )
        
        return context

    async def _finalize_result(self, context: ProcessingContext) -> ProcessingResult:
        """결과 마무리"""
        context.stage = ProcessingStage.COMPLETION
        
        processing_time = (datetime.now() - context.start_time).total_seconds()
        
        # 완료된 단계들
        stages_completed = [
            ProcessingStage.INITIALIZATION,
            ProcessingStage.INTENT_ANALYSIS
        ]
        
        if context.uploaded_files:
            stages_completed.append(ProcessingStage.FILE_PROCESSING)
        
        stages_completed.extend([
            ProcessingStage.AGENT_SELECTION,
            ProcessingStage.EXECUTION,
            ProcessingStage.QUALITY_CHECK,
            ProcessingStage.COMPLETION
        ])
        
        # 성공 여부 판단
        success = (
            context.execution_result is not None and
            len(context.execution_result.strip()) > 50 and
            "오류" not in context.execution_result
        )
        
        # 품질 점수
        quality_score = (
            context.quality_assessment.overall_score 
            if context.quality_assessment else 0.5
        )
        
        return ProcessingResult(
            request_id=context.request_id,
            success=success,
            response=context.execution_result or "처리 결과를 생성할 수 없습니다.",
            processing_time=processing_time,
            stages_completed=stages_completed,
            quality_score=quality_score,
            metadata={
                "intent": context.user_intent.primary_intent if context.user_intent else "unknown",
                "complexity": context.user_intent.complexity_level if context.user_intent else "unknown",
                "agent_selected": context.agent_decision.decision if context.agent_decision else "none",
                "files_processed": len(context.processed_files)
            }
        )

    def _update_performance_metrics(self, processing_time: float):
        """성능 메트릭 업데이트"""
        # 평균 처리 시간 계산
        total_requests = self.engine_state["total_requests"]
        current_avg = self.engine_state["average_processing_time"]
        
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.engine_state["average_processing_time"] = new_avg

    async def get_system_status(self) -> SystemStatus:
        """시스템 상태 조회"""
        if self.app_controller:
            return self.app_controller.get_system_status()
        else:
            # 기본 상태 반환
            return SystemStatus(
                overall_health=50.0,
                a2a_agents_count=0,
                mcp_tools_count=0,
                last_check=datetime.now()
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        success_rate = (
            self.engine_state["successful_requests"] / self.engine_state["total_requests"]
            if self.engine_state["total_requests"] > 0 else 0.0
        )
        
        return {
            "initialized": self.engine_state["initialized"],
            "total_requests": self.engine_state["total_requests"],
            "successful_requests": self.engine_state["successful_requests"],
            "error_count": self.engine_state["error_count"],
            "success_rate": success_rate,
            "average_processing_time": self.engine_state["average_processing_time"],
            "last_health_check": self.engine_state["last_health_check"].isoformat() if self.engine_state["last_health_check"] else None
        }

    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        try:
            # 각 컴포넌트 상태 확인
            health_status = {
                "engine_initialized": self.engine_state["initialized"],
                "app_controller": self.app_controller is not None,
                "streaming_handler": self.streaming_handler is not None,
                "file_processor": self.file_processor is not None,
                "knowledge_bank": self.knowledge_bank is not None,
                "llm_engine": self.llm_engine is not None
            }
            
            # 전체 건강도 계산
            healthy_components = sum(health_status.values())
            total_components = len(health_status)
            overall_health = (healthy_components / total_components) * 100
            
            self.engine_state["last_health_check"] = datetime.now()
            
            return {
                "overall_health": overall_health,
                "component_status": health_status,
                "last_check": self.engine_state["last_health_check"].isoformat(),
                "error_count": self.engine_state["error_count"],
                "uptime_status": "healthy" if overall_health > 80 else "degraded"
            }
            
        except Exception as e:
            logger.error(f"헬스 체크 오류: {e}")
            return {
                "overall_health": 0.0,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
                "uptime_status": "critical"
            }

    async def restart_components(self) -> bool:
        """컴포넌트 재시작"""
        try:
            logger.info("🔄 엔진 컴포넌트 재시작 중...")
            
            # 컴포넌트 재초기화
            success = await self.initialize()
            
            if success:
                logger.info("✅ 엔진 컴포넌트 재시작 완료")
            else:
                logger.error("❌ 엔진 컴포넌트 재시작 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"컴포넌트 재시작 오류: {e}")
            return False

# 전역 인스턴스 관리
_global_main_engine: Optional[CherryAIMainEngine] = None

def get_main_engine() -> CherryAIMainEngine:
    """전역 메인 엔진 인스턴스 조회"""
    global _global_main_engine
    if _global_main_engine is None:
        _global_main_engine = CherryAIMainEngine()
    return _global_main_engine

def initialize_main_engine(**kwargs) -> CherryAIMainEngine:
    """메인 엔진 초기화"""
    global _global_main_engine
    _global_main_engine = CherryAIMainEngine(**kwargs)
    return _global_main_engine

async def initialize_and_start_engine() -> CherryAIMainEngine:
    """엔진 초기화 및 시작"""
    engine = get_main_engine()
    await engine.initialize()
    return engine 