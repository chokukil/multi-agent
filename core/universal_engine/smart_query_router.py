"""
Smart Query Router - LLM 기반 지능형 쿼리 라우팅 시스템

Requirements 6.1에 따른 구현:
- LLM 기반 쿼리 복잡도 분석
- fast_track (5-10초) / balanced (10-20초) / thorough (30-60초) / expert_mode
- 오케스트레이터 오버헤드 없는 직접 응답
- 단일 에이전트 처리
- 멀티 에이전트 오케스트레이션
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
import json

from ..llm_factory import LLMFactory
from .universal_query_processor import UniversalQueryProcessor
from ..a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from ..a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
from ..a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from ..monitoring.performance_monitoring_system import PerformanceMonitor
from .langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor, get_global_tracer

logger = logging.getLogger(__name__)

# 처리 모드 타입
ProcessingMode = Literal["fast_track", "balanced", "thorough", "expert_mode"]


class SmartQueryRouter:
    """
    지능형 쿼리 라우터 - LLM First 원칙에 따른 동적 라우팅
    
    하드코딩된 규칙 없이 순수 LLM 판단으로:
    1. 쿼리 복잡도 평가
    2. 최적 처리 경로 결정
    3. 리소스 할당 최적화
    """
    
    def __init__(self):
        """SmartQueryRouter 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.universal_processor = UniversalQueryProcessor()
        self.agent_discovery = A2AAgentDiscoverySystem()
        self.agent_selector = LLMBasedAgentSelector(self.agent_discovery)
        self.workflow_orchestrator = None  # 필요시 초기화
        self.performance_monitor = PerformanceMonitor()
        
        # Langfuse 통합 완료
        self.langfuse_tracer = get_global_tracer()
        self.enhanced_executor = LangfuseEnhancedA2AExecutor(self.langfuse_tracer)
        
        logger.info("SmartQueryRouter initialized with LLM-first architecture")
    
    async def route_query(
        self, 
        query: str, 
        data: Any = None, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        쿼리를 분석하고 최적의 처리 경로로 라우팅
        
        Args:
            query: 사용자 쿼리
            data: 분석할 데이터
            context: 추가 컨텍스트
            
        Returns:
            처리 결과
        """
        start_time = datetime.now()
        context = context or {}
        
        # Langfuse 세션 생성
        session_id = context.get('session_id')
        if not session_id:
            session_id = self.langfuse_tracer.create_session(query)
            context['session_id'] = session_id
        
        try:
            # 라우팅 시작 추적
            self.langfuse_tracer.add_span(
                name="smart_query_routing_start",
                input_data={"query": query[:200]},  # 쿼리 일부만 기록
                start_time=start_time
            )
            
            # 1. 빠른 복잡도 사전 판단
            complexity_assessment = await self.quick_complexity_assessment(query, data)
            processing_mode = complexity_assessment['mode']
            
            # 복잡도 평가 추적
            self.langfuse_tracer.add_span(
                name="complexity_assessment",
                input_data={"query": query[:100]},
                output_data=complexity_assessment
            )
            
            logger.info(f"Query complexity: {complexity_assessment['complexity']} -> Mode: {processing_mode}")
            
            # 2. 처리 모드에 따른 라우팅
            if processing_mode == "fast_track":
                # 직접 응답 (5-10초)
                result = await self.direct_response(query, data, context)
                
            elif processing_mode == "balanced":
                # 단일 에이전트 (10-20초)
                result = await self.single_agent_response(query, data, context, complexity_assessment)
                
            elif processing_mode in ["thorough", "expert_mode"]:
                # 멀티 에이전트 오케스트레이션 (30-60초)
                result = await self.orchestrated_response(query, data, context, complexity_assessment)
                
            else:
                # 기본값: balanced
                result = await self.single_agent_response(query, data, context, complexity_assessment)
            
            # 처리 시간 기록
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            result['processing_mode'] = processing_mode
            
            # 성능 모니터링
            await self.performance_monitor.record_query_processing(
                query=query,
                mode=processing_mode,
                processing_time=processing_time,
                success=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query routing: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            await self.performance_monitor.record_query_processing(
                query=query,
                mode="error",
                processing_time=processing_time,
                success=False
            )
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def quick_complexity_assessment(
        self, 
        query: str, 
        data: Any = None
    ) -> Dict[str, Any]:
        """
        LLM 기반 빠른 복잡도 사전 판단
        
        Returns:
            {
                "complexity": "trivial|simple|medium|complex",
                "mode": "fast_track|balanced|thorough|expert_mode",
                "confidence": 0.0-1.0,
                "reasoning": "판단 근거"
            }
        """
        # 데이터 크기 정보 준비
        data_info = self._get_data_info(data)
        
        # LLM 프롬프트
        prompt = f"""
You are an AI assistant specialized in analyzing query complexity for data analysis tasks.
Analyze the following query and determine its complexity level and appropriate processing mode.

Query: "{query}"

Data Information:
{json.dumps(data_info, indent=2)}

Classify the query into one of these complexity levels:
- trivial: Very simple queries that can be answered directly (e.g., "What is the shape of data?", "Show first 5 rows")
- simple: Basic single-operation queries (e.g., "Calculate mean of column X", "Filter data by condition Y")
- medium: Queries requiring multiple steps or moderate analysis (e.g., "Analyze trends and patterns", "Compare two datasets")
- complex: Queries requiring deep analysis, multiple agents, or domain expertise (e.g., "Perform comprehensive analysis with ML models", "Generate detailed report with insights")

Based on the complexity, assign a processing mode:
- fast_track: For trivial queries (5-10 seconds)
- balanced: For simple queries (10-20 seconds)
- thorough: For medium queries (30-60 seconds)
- expert_mode: For complex queries requiring full capabilities (30-60 seconds)

Respond in JSON format:
{{
    "complexity": "trivial|simple|medium|complex",
    "mode": "fast_track|balanced|thorough|expert_mode",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your assessment"
}}
"""
        
        try:
            response = await self.llm_client.agenerate([prompt])
            result_text = response.generations[0][0].text.strip()
            
            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            assessment = json.loads(result_text)
            
            # 검증 및 기본값
            if assessment['complexity'] not in ['trivial', 'simple', 'medium', 'complex']:
                assessment['complexity'] = 'simple'
            if assessment['mode'] not in ['fast_track', 'balanced', 'thorough', 'expert_mode']:
                assessment['mode'] = 'balanced'
            if 'confidence' not in assessment:
                assessment['confidence'] = 0.8
                
            return assessment
            
        except Exception as e:
            logger.warning(f"Error in complexity assessment: {str(e)}, using default")
            return {
                "complexity": "simple",
                "mode": "balanced",
                "confidence": 0.5,
                "reasoning": "Default assessment due to error"
            }
    
    async def direct_response(
        self, 
        query: str, 
        data: Any = None, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        직접 LLM 응답 - 오케스트레이터 오버헤드 없음 (5-10초)
        
        trivial 쿼리를 위한 빠른 처리
        """
        logger.info("Processing via direct response (fast_track)")
        
        # 간단한 세션 추적
        session_id = context.get('session_id', f"direct_{datetime.now().timestamp()}")
        
        # 데이터 정보 준비
        data_info = self._get_data_info(data)
        
        # 직접 LLM 응답 생성
        prompt = f"""
You are a helpful data analysis assistant. Answer the following query directly and concisely.

Query: {query}

Data Information:
{json.dumps(data_info, indent=2)}

Provide a clear and direct answer. If the query asks for specific data, provide it.
If it's a question about the data, answer it directly.
"""
        
        try:
            response = await self.llm_client.agenerate([prompt])
            answer = response.generations[0][0].text.strip()
            
            return {
                "success": True,
                "result": answer,
                "mode": "fast_track",
                "session_id": session_id,
                "agents_used": []
            }
            
        except Exception as e:
            logger.error(f"Error in direct response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": "fast_track",
                "session_id": session_id
            }
    
    async def single_agent_response(
        self, 
        query: str, 
        data: Any = None, 
        context: Dict[str, Any] = None,
        complexity_assessment: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        단일 에이전트 응답 (10-20초)
        
        LLM이 선택한 가장 적합한 단일 에이전트로 처리
        """
        logger.info("Processing via single agent (balanced)")
        
        session_id = context.get('session_id', f"single_{datetime.now().timestamp()}")
        
        try:
            # LLM 기반 최적 에이전트 선택
            selected_agents = await self.agent_selector.select_agents_for_task(
                query=query,
                context={
                    **context,
                    "mode": "single_agent",
                    "complexity": complexity_assessment
                }
            )
            
            if not selected_agents:
                # 에이전트 선택 실패시 직접 응답으로 폴백
                logger.warning("No suitable agent found, falling back to direct response")
                return await self.direct_response(query, data, context)
            
            # 첫 번째 에이전트 사용
            agent = selected_agents[0]
            logger.info(f"Selected agent: {agent['name']} on port {agent['port']}")
            
            # 에이전트 실행
            result = await self._execute_single_agent(agent, query, data)
            
            return {
                "success": True,
                "result": result,
                "mode": "balanced",
                "session_id": session_id,
                "agents_used": [agent['name']]
            }
            
        except Exception as e:
            logger.error(f"Error in single agent response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": "balanced",
                "session_id": session_id
            }
    
    async def orchestrated_response(
        self, 
        query: str, 
        data: Any = None, 
        context: Dict[str, Any] = None,
        complexity_assessment: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        멀티 에이전트 오케스트레이션 응답 (30-60초)
        
        복잡한 쿼리를 위한 완전한 오케스트레이션
        """
        logger.info("Processing via multi-agent orchestration (thorough/expert)")
        
        session_id = context.get('session_id', f"orchestrated_{datetime.now().timestamp()}")
        
        # 워크플로우 오케스트레이터 초기화 (필요시)
        if not self.workflow_orchestrator:
            from ..a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
            comm_protocol = A2ACommunicationProtocol()
            self.workflow_orchestrator = A2AWorkflowOrchestrator(comm_protocol)
        
        try:
            # LLM First Optimized Orchestrator 사용 (Phase 2에서 구현)
            # 현재는 기본 Universal Query Processor 사용
            result = await self.universal_processor.process_query(
                query=query,
                data=data,
                context={
                    **context,
                    "mode": complexity_assessment.get('mode', 'thorough'),
                    "complexity_assessment": complexity_assessment
                }
            )
            
            return {
                "success": True,
                "result": result.get('final_answer', result),
                "mode": complexity_assessment.get('mode', 'thorough'),
                "session_id": session_id,
                "agents_used": result.get('agents_used', []),
                "reasoning_steps": result.get('reasoning_steps', [])
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrated response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": complexity_assessment.get('mode', 'thorough'),
                "session_id": session_id
            }
    
    def _get_data_info(self, data: Any) -> Dict[str, Any]:
        """데이터 정보 추출"""
        if data is None:
            return {"type": "none", "description": "No data provided"}
        
        data_info = {"type": type(data).__name__}
        
        # pandas DataFrame 처리
        if hasattr(data, 'shape'):
            data_info.update({
                "shape": data.shape,
                "columns": list(data.columns) if hasattr(data, 'columns') else None,
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()} if hasattr(data, 'dtypes') else None
            })
        
        # 리스트/딕셔너리 처리
        elif isinstance(data, (list, dict)):
            data_info["length"] = len(data)
            if isinstance(data, list) and data:
                data_info["sample"] = str(data[0])[:100]
        
        # 문자열 처리
        elif isinstance(data, str):
            data_info["length"] = len(data)
            data_info["preview"] = data[:200]
        
        return data_info
    
    async def _execute_single_agent(
        self, 
        agent: Dict[str, Any], 
        query: str, 
        data: Any
    ) -> Any:
        """단일 에이전트 실행"""
        # TODO: 실제 A2A 에이전트 호출 구현
        # 현재는 시뮬레이션
        return f"Result from {agent['name']}: Processed query '{query}'"
    
    async def route_query_with_streaming(self, query: str, data: Any = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 방식으로 쿼리 처리
        
        Args:
            query: 사용자 쿼리
            data: 선택적 데이터
            
        Yields:
            Dict[str, Any]: 스트리밍 응답 청크
        """
        logger.info(f"🚀 Streaming query processing started: {query[:50]}...")
        
        try:
            # 복잡도 평가
            yield {
                "type": "status",
                "content": "🔍 쿼리 복잡도를 분석하고 있습니다...",
                "step": "complexity_assessment"
            }
            
            assessment = await self.quick_complexity_assessment(query)
            
            yield {
                "type": "status", 
                "content": f"📊 복잡도: {assessment['complexity']} | 모드: {assessment['mode']}",
                "step": "complexity_result"
            }
            
            # 모드별 처리
            if assessment['mode'] == 'fast_track':
                yield {
                    "type": "status",
                    "content": "⚡ 빠른 응답 모드로 처리 중...",
                    "step": "direct_processing"
                }
                
                result = await self.direct_response(query, data, {"session_id": f"stream_{datetime.now().timestamp()}"})
                
                yield {
                    "type": "result",
                    "content": result['result'],
                    "agent": "LLM_Direct",
                    "processing_time": result['processing_time']
                }
                
            elif assessment['mode'] == 'balanced':
                yield {
                    "type": "status",
                    "content": "🎯 단일 에이전트 모드로 처리 중...",
                    "step": "single_agent_processing"
                }
                
                result = await self.single_agent_response(query, data, {"session_id": f"stream_{datetime.now().timestamp()}"})
                
                yield {
                    "type": "result",
                    "content": result['result'],
                    "agent": result.get('agent_used', 'Unknown'),
                    "processing_time": result['processing_time']
                }
                
            else:  # thorough or expert_mode
                yield {
                    "type": "status",
                    "content": "🎭 다중 에이전트 협업 모드로 처리 중...",
                    "step": "orchestrated_processing"
                }
                
                result = await self.orchestrated_response(query, data, {"session_id": f"stream_{datetime.now().timestamp()}"})
                
                yield {
                    "type": "result",
                    "content": result['result'],
                    "agent": "Multi-Agent",
                    "processing_time": result['processing_time']
                }
                
        except Exception as e:
            logger.error(f"Streaming query processing error: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"처리 중 오류가 발생했습니다: {str(e)}"
            }