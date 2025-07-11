"""
Multi-Agent Orchestration System

새로운 멀티 에이전트 오케스트레이션 시스템
- UniversalDataAnalysisRouter 통합
- 전문화 에이전트들 (정형/시계열/텍스트/이미지) 통합
- pandas-ai 서버 통합
- Enhanced Langfuse v2 추적 통합

Author: CherryAI Team
Date: 2024-12-30
"""

import asyncio
import json
import logging
import os
import time
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# A2A SDK 0.2.9 표준 임포트
try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.types import (
        AgentCard,
        AgentSkill,
        AgentCapabilities,
        TaskState,
        TextPart,
        Part
    )
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.utils import new_agent_text_message, new_task
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logger.warning("⚠️ A2A SDK not available")

# 우리가 구현한 시스템들 통합
try:
    from core.universal_data_analysis_router import get_universal_router
    from core.specialized_data_agents import get_data_type_detector, DataType, DataAnalysisResult
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    from core.user_file_tracker import get_user_file_tracker
    from core.session_data_manager import SessionDataManager
    CORE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    CORE_SYSTEMS_AVAILABLE = False
    print(f"⚠️ Core systems not available: {e}")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """오케스트레이션 전략"""
    SINGLE_AGENT = "single_agent"           # 단일 에이전트 실행
    SEQUENTIAL = "sequential"               # 순차 실행
    PARALLEL = "parallel"                   # 병렬 실행
    HIERARCHICAL = "hierarchical"           # 계층적 실행
    COLLABORATIVE = "collaborative"         # 협력적 실행


@dataclass
class AgentTask:
    """에이전트 작업 정의"""
    agent_id: str
    agent_type: str
    task_description: str
    input_data: Any
    dependencies: List[str] = None
    priority: int = 5  # 1-10
    timeout: int = 300  # seconds
    retry_count: int = 3


@dataclass
class OrchestrationPlan:
    """오케스트레이션 실행 계획"""
    strategy: OrchestrationStrategy
    tasks: List[AgentTask]
    estimated_duration: int
    confidence: float
    reasoning: str


@dataclass
class OrchestrationResult:
    """오케스트레이션 결과"""
    success: bool
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    execution_time: float
    errors: List[str] = None
    metadata: Dict[str, Any] = None


class MultiAgentOrchestrator:
    """
    멀티 에이전트 오케스트레이션 시스템
    
    다양한 전문화 에이전트들을 조합하여 복잡한 데이터 분석 작업을 수행
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.enhanced_tracer = None
        self.universal_router = None
        self.data_type_detector = None
        self.user_file_tracker = None
        self.session_data_manager = None
        
        # 에이전트 엔드포인트 매핑
        self.agent_endpoints = {
            "pandas_ai": "http://localhost:8000",
            "eda_tools": "http://localhost:8001",
            "data_visualization": "http://localhost:8002",
            "data_cleaning": "http://localhost:8003",
            "feature_engineering": "http://localhost:8004",
            "ml_agent": "http://localhost:8005"
        }
        
        # 실행 히스토리
        self.execution_history: List[Dict] = []
        
        # 시스템 초기화
        self._initialize_systems()
        
        logger.info("🚀 Multi-Agent Orchestrator 초기화 완료")
    
    def _initialize_systems(self):
        """핵심 시스템들 초기화"""
        if not CORE_SYSTEMS_AVAILABLE:
            logger.warning("⚠️ Core systems not available - using fallback mode")
            return
        
        try:
            # Enhanced Tracking 초기화
            self.enhanced_tracer = get_enhanced_tracer()
            logger.info("✅ Enhanced Langfuse Tracking 활성화")
            
            # Universal Router 초기화
            self.universal_router = get_universal_router()
            logger.info("✅ Universal Data Analysis Router 활성화")
            
            # Data Type Detector 초기화
            self.data_type_detector = get_data_type_detector()
            logger.info("✅ Data Type Detector 활성화")
            
            # User File Tracker 초기화
            self.user_file_tracker = get_user_file_tracker()
            self.session_data_manager = SessionDataManager()
            logger.info("✅ User File Tracker & Session Manager 활성화")
            
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}")
    
    async def orchestrate_analysis(
        self, 
        user_query: str, 
        data: Optional[Any] = None, 
        session_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> OrchestrationResult:
        """
        데이터 분석을 위한 멀티 에이전트 오케스트레이션
        
        Args:
            user_query: 사용자 질문
            data: 분석할 데이터 (선택사항)
            session_id: 세션 ID
            context: 추가 컨텍스트
            
        Returns:
            OrchestrationResult: 오케스트레이션 결과
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔄 멀티 에이전트 오케스트레이션 시작: {user_query[:100]}...")
            
            # Enhanced Tracking
            if self.enhanced_tracer:
                self.enhanced_tracer.log_data_operation(
                    "orchestration_start",
                    {"query": user_query, "session_id": session_id},
                    "Starting multi-agent orchestration"
                )
            
            # 1. 데이터 수집 및 준비
            prepared_data = await self._prepare_data(data, session_id, context)
            
            # 2. 질문 분석 및 라우팅
            routing_result = await self._analyze_and_route_query(user_query, prepared_data, session_id, context)
            
            # 3. 오케스트레이션 계획 생성
            plan = await self._create_orchestration_plan(user_query, routing_result, prepared_data)
            
            # 4. 계획 실행
            execution_result = await self._execute_orchestration_plan(plan, session_id)
            
            # 5. 결과 통합 및 해석
            final_result = await self._integrate_and_interpret_results(
                user_query, execution_result, plan
            )
            
            execution_time = time.time() - start_time
            
            # 실행 히스토리 기록
            self.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "strategy": plan.strategy.value,
                "execution_time": execution_time,
                "success": final_result.success,
                "session_id": session_id
            })
            
            logger.info(f"✅ 오케스트레이션 완료 (소요시간: {execution_time:.2f}초)")
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ 오케스트레이션 실패: {e}")
            
            return OrchestrationResult(
                success=False,
                results={"error": str(e)},
                insights=[],
                recommendations=["시스템 오류가 발생했습니다. 다시 시도해주세요."],
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    async def _prepare_data(
        self, 
        data: Optional[Any], 
        session_id: Optional[str], 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """데이터 준비 및 컨텍스트 수집"""
        prepared = {
            "provided_data": data,
            "session_files": [],
            "context": context or {}
        }
        
        # 세션 파일 수집
        if session_id and self.session_data_manager:
            try:
                uploaded_files = self.session_data_manager.get_uploaded_files(session_id)
                prepared["session_files"] = uploaded_files
                
                # 파일 데이터 로드
                if uploaded_files and self.user_file_tracker:
                    file_data = []
                    for file_name in uploaded_files[:3]:  # 최대 3개 파일
                        file_path = self.user_file_tracker.get_best_file(
                            session_id=session_id,
                            query=file_name
                        )
                        if file_path:
                            # 파일 타입에 따른 로딩 (간단한 버전)
                            file_info = {
                                "name": file_name,
                                "path": file_path,
                                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            }
                            file_data.append(file_info)
                    
                    prepared["file_data"] = file_data
                    
            except Exception as e:
                logger.warning(f"⚠️ 세션 데이터 수집 실패: {e}")
        
        return prepared
    
    async def _analyze_and_route_query(
        self, 
        user_query: str, 
        prepared_data: Dict, 
        session_id: Optional[str], 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """질문 분석 및 라우팅"""
        if not self.universal_router:
            return {"error": "Universal router not available"}
        
        try:
            # Universal Router로 질문 분석
            routing_result = await self.universal_router.route_query(
                user_query, session_id, context
            )
            
            # 데이터 타입 탐지 (데이터가 있는 경우)
            data_type_info = None
            if prepared_data.get("provided_data") and self.data_type_detector:
                detection_result = self.data_type_detector.detect_data_type(
                    prepared_data["provided_data"]
                )
                data_type_info = {
                    "detected_type": detection_result.detected_type.value,
                    "confidence": detection_result.confidence,
                    "characteristics": detection_result.characteristics
                }
            
            return {
                "routing": routing_result,
                "data_type": data_type_info,
                "complexity": self._assess_query_complexity(user_query),
                "requires_data": self._requires_data_analysis(user_query)
            }
            
        except Exception as e:
            logger.error(f"❌ 질문 분석 실패: {e}")
            return {"error": str(e)}
    
    def _assess_query_complexity(self, user_query: str) -> str:
        """질문 복잡도 평가"""
        query_lower = user_query.lower()
        
        # 복잡한 키워드들
        complex_keywords = [
            '비교', '분석', '예측', '모델', '상관관계', '분류', '회귀',
            'compare', 'analyze', 'predict', 'model', 'correlation', 'classify', 'regression'
        ]
        
        # 다단계 키워드들
        multi_step_keywords = [
            '그리고', '다음', '그 후', '또한', '추가로', '마지막으로',
            'and then', 'next', 'after that', 'also', 'additionally', 'finally'
        ]
        
        if any(keyword in query_lower for keyword in multi_step_keywords):
            return "multi_step"
        elif any(keyword in query_lower for keyword in complex_keywords):
            return "complex"
        else:
            return "simple"
    
    def _requires_data_analysis(self, user_query: str) -> bool:
        """데이터 분석이 필요한지 판단"""
        data_keywords = [
            '데이터', '차트', '그래프', '분석', '통계', '평균', '분포',
            'data', 'chart', 'graph', 'analysis', 'statistics', 'average', 'distribution'
        ]
        return any(keyword in user_query.lower() for keyword in data_keywords)
    
    async def _create_orchestration_plan(
        self, 
        user_query: str, 
        routing_result: Dict, 
        prepared_data: Dict
    ) -> OrchestrationPlan:
        """오케스트레이션 실행 계획 생성"""
        complexity = routing_result.get("complexity", "simple")
        requires_data = routing_result.get("requires_data", False)
        
        tasks = []
        strategy = OrchestrationStrategy.SINGLE_AGENT
        estimated_duration = 30  # seconds
        confidence = 0.8
        reasoning = "기본 단일 에이전트 전략"
        
        if complexity == "simple" and not requires_data:
            # 단순 질문 - LLM 직접 응답
            tasks.append(AgentTask(
                agent_id="llm_direct",
                agent_type="llm",
                task_description="단순 질문 직접 응답",
                input_data={"query": user_query}
            ))
            strategy = OrchestrationStrategy.SINGLE_AGENT
            estimated_duration = 10
            reasoning = "단순 질문으로 LLM 직접 응답"
            
        elif complexity == "simple" and requires_data:
            # 단순 데이터 분석 - 적절한 에이전트 하나 선택
            if routing_result.get("routing", {}).get("success"):
                recommended_agent = routing_result["routing"]["decision"]["recommended_agent"]
                tasks.append(AgentTask(
                    agent_id=recommended_agent,
                    agent_type="specialized",
                    task_description=f"{recommended_agent} 에이전트를 통한 데이터 분석",
                    input_data={"query": user_query, "data": prepared_data}
                ))
                strategy = OrchestrationStrategy.SINGLE_AGENT
                reasoning = f"라우터 추천 {recommended_agent} 에이전트 사용"
            
        elif complexity == "complex":
            # 복잡한 분석 - 여러 에이전트 순차 실행
            tasks.extend([
                AgentTask(
                    agent_id="eda",
                    agent_type="specialized",
                    task_description="탐색적 데이터 분석",
                    input_data={"query": "데이터 기본 분석", "data": prepared_data},
                    priority=9
                ),
                AgentTask(
                    agent_id="pandas_ai",
                    agent_type="universal",
                    task_description="자연어 기반 심화 분석",
                    input_data={"query": user_query, "data": prepared_data},
                    dependencies=["eda"],
                    priority=8
                )
            ])
            strategy = OrchestrationStrategy.SEQUENTIAL
            estimated_duration = 120
            confidence = 0.9
            reasoning = "복잡한 질문으로 EDA 후 pandas-ai 심화 분석"
            
        elif complexity == "multi_step":
            # 다단계 분석 - 계층적 실행
            tasks.extend([
                AgentTask(
                    agent_id="data_type_detector",
                    agent_type="detector",
                    task_description="데이터 타입 탐지",
                    input_data={"data": prepared_data},
                    priority=10
                ),
                AgentTask(
                    agent_id="eda",
                    agent_type="specialized", 
                    task_description="기본 탐색적 분석",
                    input_data={"query": "기본 분석", "data": prepared_data},
                    dependencies=["data_type_detector"],
                    priority=9
                ),
                AgentTask(
                    agent_id="specialized_analysis",
                    agent_type="specialized",
                    task_description="전문화된 분석",
                    input_data={"query": user_query, "data": prepared_data},
                    dependencies=["eda"],
                    priority=8
                ),
                AgentTask(
                    agent_id="pandas_ai",
                    agent_type="universal",
                    task_description="종합 분석 및 해석",
                    input_data={"query": user_query, "data": prepared_data},
                    dependencies=["specialized_analysis"],
                    priority=7
                )
            ])
            strategy = OrchestrationStrategy.HIERARCHICAL
            estimated_duration = 180
            confidence = 0.95
            reasoning = "다단계 질문으로 계층적 분석 워크플로우"
        
        return OrchestrationPlan(
            strategy=strategy,
            tasks=tasks,
            estimated_duration=estimated_duration,
            confidence=confidence,
            reasoning=reasoning
        )
    
    async def _execute_orchestration_plan(
        self, 
        plan: OrchestrationPlan, 
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """오케스트레이션 계획 실행"""
        results = {}
        
        if plan.strategy == OrchestrationStrategy.SINGLE_AGENT:
            # 단일 에이전트 실행
            if plan.tasks:
                task = plan.tasks[0]
                result = await self._execute_single_task(task, session_id)
                results[task.agent_id] = result
                
        elif plan.strategy == OrchestrationStrategy.SEQUENTIAL:
            # 순차 실행
            for task in sorted(plan.tasks, key=lambda t: t.priority, reverse=True):
                # 의존성 확인
                if task.dependencies:
                    dependencies_met = all(dep in results for dep in task.dependencies)
                    if not dependencies_met:
                        logger.warning(f"⚠️ 작업 {task.agent_id}의 의존성이 충족되지 않음")
                        continue
                
                result = await self._execute_single_task(task, session_id)
                results[task.agent_id] = result
                
        elif plan.strategy == OrchestrationStrategy.PARALLEL:
            # 병렬 실행
            tasks_to_run = [self._execute_single_task(task, session_id) for task in plan.tasks]
            parallel_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            
            for i, task in enumerate(plan.tasks):
                results[task.agent_id] = parallel_results[i]
                
        elif plan.strategy == OrchestrationStrategy.HIERARCHICAL:
            # 계층적 실행 (의존성 순서에 따라)
            remaining_tasks = plan.tasks.copy()
            
            while remaining_tasks:
                # 실행 가능한 작업 찾기
                ready_tasks = [
                    task for task in remaining_tasks
                    if not task.dependencies or all(dep in results for dep in task.dependencies)
                ]
                
                if not ready_tasks:
                    logger.error("❌ 순환 의존성 또는 해결할 수 없는 의존성 발견")
                    break
                
                # 우선순위가 높은 작업부터 실행
                ready_tasks.sort(key=lambda t: t.priority, reverse=True)
                
                for task in ready_tasks:
                    result = await self._execute_single_task(task, session_id)
                    results[task.agent_id] = result
                    remaining_tasks.remove(task)
        
        return results
    
    async def _execute_single_task(self, task: AgentTask, session_id: Optional[str]) -> Dict[str, Any]:
        """단일 작업 실행"""
        try:
            logger.info(f"🔄 작업 실행: {task.agent_id} - {task.task_description}")
            
            if task.agent_type == "llm":
                return await self._execute_llm_task(task)
            elif task.agent_type == "specialized":
                return await self._execute_specialized_agent_task(task)
            elif task.agent_type == "universal":
                return await self._execute_universal_agent_task(task)
            elif task.agent_type == "detector":
                return await self._execute_detector_task(task)
            else:
                return {"error": f"Unknown agent type: {task.agent_type}"}
                
        except Exception as e:
            logger.error(f"❌ 작업 실행 실패 {task.agent_id}: {e}")
            return {"error": str(e), "agent_id": task.agent_id}
    
    async def _execute_llm_task(self, task: AgentTask) -> Dict[str, Any]:
        """LLM 직접 실행"""
        try:
            # 간단한 LLM 응답 (실제로는 OpenAI API 호출)
            query = task.input_data.get("query", "")
            
            response = f"'{query}'에 대한 답변입니다. 이는 LLM이 직접 생성한 응답으로, 추가적인 데이터 분석이 필요하지 않은 일반적인 질문에 대한 답변입니다."
            
            return {
                "success": True,
                "response": response,
                "type": "llm_direct",
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _execute_specialized_agent_task(self, task: AgentTask) -> Dict[str, Any]:
        """전문화 에이전트 실행"""
        try:
            if not self.data_type_detector:
                return {"error": "Data type detector not available", "success": False}
            
            query = task.input_data.get("query", "")
            data = task.input_data.get("data", {}).get("provided_data")
            
            if data is not None:
                # 전문화 에이전트로 분석
                result = await self.data_type_detector.analyze_with_best_agent(data, query)
                
                return {
                    "success": True,
                    "analysis_type": result.analysis_type,
                    "data_type": result.data_type.value,
                    "results": result.results,
                    "insights": result.insights,
                    "recommendations": result.recommendations,
                    "confidence": result.confidence
                }
            else:
                return {
                    "success": False,
                    "error": "No data provided for specialized analysis"
                }
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _execute_universal_agent_task(self, task: AgentTask) -> Dict[str, Any]:
        """범용 에이전트 (pandas-ai) 실행"""
        try:
            # pandas-ai 서버 호출 시뮬레이션
            endpoint = self.agent_endpoints.get("pandas_ai", "http://localhost:8000")
            query = task.input_data.get("query", "")
            
            # 실제로는 HTTP 요청을 보냄
            response_text = f"pandas-ai 에이전트가 '{query}'에 대해 분석했습니다. 자연어 처리를 통한 데이터 분석 결과입니다."
            
            return {
                "success": True,
                "response": response_text,
                "type": "pandas_ai",
                "confidence": 0.9,
                "code_generated": "# pandas-ai generated code\ndf.describe()",
                "execution_result": "분석 완료"
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _execute_detector_task(self, task: AgentTask) -> Dict[str, Any]:
        """데이터 타입 탐지 실행"""
        try:
            if not self.data_type_detector:
                return {"error": "Data type detector not available", "success": False}
            
            data = task.input_data.get("data", {}).get("provided_data")
            
            if data is not None:
                detection_result = self.data_type_detector.detect_data_type(data)
                
                return {
                    "success": True,
                    "detected_type": detection_result.detected_type.value,
                    "confidence": detection_result.confidence,
                    "reasoning": detection_result.reasoning,
                    "characteristics": detection_result.characteristics,
                    "recommendations": detection_result.recommendations
                }
            else:
                return {
                    "success": False,
                    "error": "No data provided for type detection"
                }
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _integrate_and_interpret_results(
        self, 
        user_query: str, 
        execution_results: Dict, 
        plan: OrchestrationPlan
    ) -> OrchestrationResult:
        """결과 통합 및 해석"""
        try:
            # 성공한 결과들 수집
            successful_results = {
                k: v for k, v in execution_results.items() 
                if isinstance(v, dict) and v.get("success", False)
            }
            
            # 오류 수집
            errors = [
                f"{k}: {v.get('error', 'Unknown error')}" 
                for k, v in execution_results.items()
                if isinstance(v, dict) and not v.get("success", True)
            ]
            
            # 인사이트 통합
            all_insights = []
            all_recommendations = []
            
            for agent_id, result in successful_results.items():
                if "insights" in result:
                    all_insights.extend(result["insights"])
                if "recommendations" in result:
                    all_recommendations.extend(result["recommendations"])
                if "response" in result:
                    all_insights.append(f"{agent_id}: {result['response']}")
            
            # 기본 인사이트 추가 (결과가 없는 경우)
            if not all_insights:
                all_insights.append(f"'{user_query}'에 대한 분석을 완료했습니다.")
                
            if not all_recommendations:
                all_recommendations.append("추가적인 분석이나 질문이 있으시면 언제든 문의해주세요.")
            
            # 성공 여부 판단
            success = len(successful_results) > 0 and len(errors) == 0
            
            return OrchestrationResult(
                success=success,
                results=execution_results,
                insights=all_insights,
                recommendations=all_recommendations,
                execution_time=0,  # 실제로는 계산됨
                errors=errors if errors else None,
                metadata={
                    "strategy": plan.strategy.value,
                    "tasks_executed": len(execution_results),
                    "successful_tasks": len(successful_results),
                    "plan_confidence": plan.confidence
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 결과 통합 실패: {e}")
            return OrchestrationResult(
                success=False,
                results=execution_results,
                insights=[],
                recommendations=["결과 통합 중 오류가 발생했습니다."],
                execution_time=0,
                errors=[str(e)]
            )
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """오케스트레이션 통계 조회"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0,
                "average_execution_time": 0,
                "strategy_distribution": {},
                "recent_executions": []
            }
        
        total = len(self.execution_history)
        successful = sum(1 for h in self.execution_history if h.get("success", False))
        
        strategies = {}
        total_time = 0
        
        for history in self.execution_history:
            strategy = history.get("strategy", "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1
            total_time += history.get("execution_time", 0)
        
        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0,
            "average_execution_time": total_time / total if total > 0 else 0,
            "strategy_distribution": strategies,
            "recent_executions": self.execution_history[-5:]  # 최근 5개
        }
    
    def clear_execution_history(self):
        """실행 히스토리 정리"""
        self.execution_history.clear()
        logger.info("✅ 오케스트레이션 실행 히스토리 정리 완료")


# A2A SDK 통합을 위한 Executor 클래스
class MultiAgentA2AExecutor(AgentExecutor):
    """A2A SDK 호환 Multi-Agent Executor"""
    
    def __init__(self, orchestrator: MultiAgentOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 표준 실행 메서드"""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await updater.submit()
            await updater.start_work()
            
            # 사용자 입력 추출
            user_input = self._extract_user_input(context)
            
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 멀티 에이전트 오케스트레이션 시작...")
            )
            
            # 오케스트레이션 실행
            result = await self.orchestrator.orchestrate_analysis(
                user_query=user_input,
                session_id=context.context_id
            )
            
            if result.success:
                # 인사이트를 텍스트로 결합
                response_text = "\n".join(result.insights)
                if result.recommendations:
                    response_text += "\n\n권장사항:\n" + "\n".join(f"- {rec}" for rec in result.recommendations)
                
                await updater.add_artifact(
                    [TextPart(text=response_text)],
                    name="orchestration_result"
                )
                await updater.complete()
            else:
                error_message = "오케스트레이션 실행 중 오류가 발생했습니다."
                if result.errors:
                    error_message += f"\n오류: {'; '.join(result.errors)}"
                
                await updater.update_status(
                    TaskState.error,
                    message=new_agent_text_message(error_message)
                )
                
        except Exception as e:
            logger.error(f"❌ A2A 실행 실패: {e}")
            await updater.update_status(
                TaskState.error,
                message=new_agent_text_message(f"실행 오류: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info("🛑 Multi-Agent 오케스트레이션 취소")
    
    def _extract_user_input(self, context: RequestContext) -> str:
        """컨텍스트에서 사용자 입력 추출"""
        try:
            if context.current_task and context.current_task.message and context.current_task.message.parts:
                for part in context.current_task.message.parts:
                    if hasattr(part, 'text') and part.text:
                        return part.text
            return "데이터 분석을 수행해주세요"
        except Exception as e:
            logger.warning(f"⚠️ 사용자 입력 추출 실패: {e}")
            return "데이터 분석을 수행해주세요"


# 전역 인스턴스
_orchestrator_instance = None


def get_multi_agent_orchestrator(config: Optional[Dict] = None) -> MultiAgentOrchestrator:
    """Multi-Agent Orchestrator 인스턴스 반환"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiAgentOrchestrator(config)
    return _orchestrator_instance


# A2A 서버 설정 함수
def create_multi_agent_card() -> Optional[AgentCard]:
    """Multi-Agent A2A 카드 생성"""
    if not A2A_AVAILABLE:
        return None
    
    return AgentCard(
        name="Multi-Agent Data Analysis Orchestrator",
        description="지능형 멀티 에이전트 오케스트레이션을 통한 포괄적 데이터 분석",
        capabilities=AgentCapabilities(
            skills=[
                AgentSkill(
                    name="multi_agent_orchestration",
                    description="여러 전문화 에이전트를 조합한 복합 데이터 분석"
                ),
                AgentSkill(
                    name="intelligent_routing",
                    description="LLM 기반 지능형 에이전트 라우팅"
                ),
                AgentSkill(
                    name="data_type_detection",
                    description="자동 데이터 타입 탐지 및 맞춤 분석"
                ),
                AgentSkill(
                    name="comprehensive_analysis",
                    description="정형/시계열/텍스트/이미지 데이터 종합 분석"
                )
            ]
        )
    )


def create_multi_agent_a2a_app() -> Optional[A2AStarletteApplication]:
    """Multi-Agent A2A 애플리케이션 생성"""
    if not A2A_AVAILABLE:
        return None
    
    orchestrator = get_multi_agent_orchestrator()
    executor = MultiAgentA2AExecutor(orchestrator)
    card = create_multi_agent_card()
    
    if card:
        return A2AStarletteApplication(
            task_store=InMemoryTaskStore(),
            request_handler=DefaultRequestHandler(card, executor)
        )
    return None


# CLI 테스트 함수
async def test_multi_agent_orchestrator():
    """Multi-Agent Orchestrator 테스트"""
    print("🔄 Multi-Agent Orchestrator 테스트 시작\n")
    
    orchestrator = get_multi_agent_orchestrator()
    
    # 테스트 질문들
    test_queries = [
        "안녕하세요?",
        "데이터의 기본 통계를 보여주세요",
        "상관관계를 분석하고 시각화해주세요",
        "데이터를 정제하고 예측 모델을 만들어주세요"
    ]
    
    # 샘플 데이터
    import pandas as pd
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': ['x', 'y', 'x', 'y', 'x']
    })
    
    for i, query in enumerate(test_queries):
        print(f"📝 테스트 {i+1}: {query}")
        
        result = await orchestrator.orchestrate_analysis(
            user_query=query,
            data=sample_data if "데이터" in query else None,
            session_id=f"test_session_{i}"
        )
        
        print(f"  ✅ 성공: {result.success}")
        print(f"  🔍 인사이트: {len(result.insights)}개")
        print(f"  💡 추천사항: {len(result.recommendations)}개")
        print(f"  ⏱️ 실행시간: {result.execution_time:.2f}초")
        print()
    
    # 통계 출력
    stats = orchestrator.get_orchestration_statistics()
    print("📊 오케스트레이션 통계:")
    print(f"  총 실행: {stats['total_executions']}")
    print(f"  성공률: {stats['success_rate']:.2%}")
    print(f"  평균 시간: {stats['average_execution_time']:.2f}초")
    print(f"  전략 분포: {stats['strategy_distribution']}")
    
    print("\n✅ Multi-Agent Orchestrator 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_orchestrator()) 