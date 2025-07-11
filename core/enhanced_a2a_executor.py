"""
🔍 Enhanced A2A Executor with Deep Internal Tracking
웹 검색 결과를 바탕으로 구현된 고급 A2A Executor

핵심 기능:
- 에이전트 내부 처리 과정 완전 추적
- 코드 생성 및 실행 과정 상세 로깅
- LLM 상호작용 실시간 모니터링
- 네스팅된 스팬 구조로 계층적 추적
- AI_DS_Team 에이전트와의 완벽한 통합
"""

import asyncio
import json
import time
import traceback
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager
import pandas as pd

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import TextPart, TaskState

# Enhanced tracking imports
try:
    from core.enhanced_langfuse_tracer import get_enhanced_tracer, EnhancedLangfuseTracer
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False
    print("⚠️ Enhanced tracking not available")

# Data management imports
try:
    from core.data_manager import DataManager
    from core.session_data_manager import SessionDataManager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    print("⚠️ Data manager not available")

class EnhancedA2AExecutor(AgentExecutor):
    """향상된 A2A Executor with Deep Internal Tracking"""
    
    def __init__(self, 
                 agent_name: str,
                 agent_description: str = None,
                 enable_enhanced_tracking: bool = True):
        """
        Args:
            agent_name: 에이전트 이름
            agent_description: 에이전트 설명
            enable_enhanced_tracking: 향상된 추적 활성화 여부
        """
        super().__init__()
        self.agent_name = agent_name
        self.agent_description = agent_description or f"{agent_name} Agent"
        self.enable_enhanced_tracking = enable_enhanced_tracking and ENHANCED_TRACKING_AVAILABLE
        
        # 추적 및 데이터 관리 시스템 초기화
        self.tracer = get_enhanced_tracer() if self.enable_enhanced_tracking else None
        self.data_manager = DataManager() if DATA_MANAGER_AVAILABLE else None
        self.session_data_manager = SessionDataManager() if DATA_MANAGER_AVAILABLE else None
        
        # 에이전트 상태 추적
        self.current_agent_span = None
        self.execution_context = {}
        
        print(f"🚀 Enhanced A2A Executor initialized - Agent: {self.agent_name}")
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        print(f"🚫 Enhanced A2A Executor task cancelled: {self.agent_name}")
        
        # Enhanced Tracking: 취소 이벤트 로깅
        if self.enable_enhanced_tracking and self.tracer:
            self.tracer.log_data_operation(
                "task_cancellation",
                {
                    "agent_name": self.agent_name,
                    "context_id": getattr(context, 'context_id', 'unknown'),
                    "task_id": getattr(context, 'task_id', 'unknown')
                },
                f"Task cancelled for agent {self.agent_name}"
            )
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """메인 실행 함수 - 하위 클래스에서 오버라이드"""
        if self.enable_enhanced_tracking:
            with self.tracer.trace_agent_execution(
                self.agent_name, 
                "Processing user request",
                {"agent_description": self.agent_description}
            ) as span:
                self.current_agent_span = span
                return await self._execute_with_tracking(context, task_updater)
        else:
            return await self._execute_with_tracking(context, task_updater)
    
    async def _execute_with_tracking(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """실제 실행 로직 - 하위 클래스에서 구현"""
        raise NotImplementedError("Subclasses must implement _execute_with_tracking")
    
    @contextmanager
    def trace_processing_step(self, step_name: str, metadata: Dict[str, Any] = None):
        """처리 단계 추적"""
        if self.enable_enhanced_tracking and self.tracer:
            with self.tracer.trace_internal_step(step_name, "processing", metadata) as span:
                yield span
        else:
            yield None
    
    def log_user_input_analysis(self, user_input: str, analysis_result: Dict[str, Any]):
        """사용자 입력 분석 로깅"""
        if self.enable_enhanced_tracking and self.tracer:
            self.tracer.log_data_operation(
                "user_input_analysis",
                {
                    "input": user_input,
                    "input_length": len(user_input),
                    "analysis": analysis_result
                },
                f"Analyzed user input: {analysis_result.get('intent', 'unknown')}"
            )
    
    def log_data_loading(self, data_source: str, data_info: Dict[str, Any]):
        """데이터 로딩 로깅"""
        if self.enable_enhanced_tracking and self.tracer:
            self.tracer.log_data_operation(
                "data_loading",
                {
                    "source": data_source,
                    "data_info": data_info
                },
                f"Loaded data from {data_source}"
            )
    
    def log_code_generation_process(self, 
                                   prompt: str, 
                                   generated_code: str,
                                   execution_result: Any = None,
                                   error: str = None):
        """코드 생성 과정 로깅"""
        if self.enable_enhanced_tracking and self.tracer:
            metadata = {}
            if error:
                metadata["error"] = error
                metadata["success"] = False
            else:
                metadata["success"] = True
            
            self.tracer.log_code_generation(
                prompt=prompt,
                generated_code=generated_code,
                execution_result=execution_result,
                metadata=metadata
            )
    
    def log_llm_call(self, 
                    model_name: str,
                    prompt: str,
                    response: str,
                    token_usage: Dict[str, int] = None,
                    metadata: Dict[str, Any] = None):
        """LLM 호출 로깅"""
        if self.enable_enhanced_tracking and self.tracer:
            self.tracer.log_llm_interaction(
                model_name=model_name,
                prompt=prompt,
                response=response,
                token_usage=token_usage,
                metadata=metadata
            )
    
    def log_data_analysis_result(self, 
                                analysis_type: str,
                                data_summary: Dict[str, Any],
                                results: Dict[str, Any]):
        """데이터 분석 결과 로깅"""
        if self.enable_enhanced_tracking and self.tracer:
            self.tracer.log_data_operation(
                f"data_analysis_{analysis_type}",
                {
                    "analysis_type": analysis_type,
                    "data_summary": data_summary,
                    "results": results
                },
                f"Completed {analysis_type} analysis"
            )
    
    def get_user_input_from_context(self, context: RequestContext) -> str:
        """컨텍스트에서 사용자 입력 추출"""
        try:
            if hasattr(context, 'message') and context.message:
                if hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            return part.root.text
                        elif hasattr(part, 'text'):
                            return part.text
            return "No user input found"
        except Exception as e:
            print(f"⚠️ Error extracting user input: {e}")
            return "Error extracting user input"
    
    def format_response_for_a2a(self, response_content: str) -> List[TextPart]:
        """A2A 프로토콜용 응답 포맷팅"""
        return [TextPart(text=response_content)]
    
    async def stream_response_chunk(self, 
                                   task_updater: TaskUpdater,
                                   chunk: str,
                                   is_final: bool = False):
        """응답 청크 스트리밍"""
        try:
            await task_updater.update_status(
                TaskState.working if not is_final else TaskState.completed,
                message=chunk
            )
            
            # 스트리밍 로깅
            if self.enable_enhanced_tracking and self.tracer:
                self.tracer.log_data_operation(
                    "response_streaming",
                    {
                        "chunk_length": len(chunk),
                        "is_final": is_final,
                        "agent": self.agent_name
                    },
                    f"Streamed chunk: {len(chunk)} characters"
                )
        except Exception as e:
            print(f"⚠️ Error streaming response: {e}")

class EnhancedAIDataScienceExecutor(EnhancedA2AExecutor):
    """AI Data Science 전용 Enhanced Executor"""
    
    def __init__(self, 
                 agent_name: str,
                 ai_ds_agent_class,
                 llm_instance=None):
        """
        Args:
            agent_name: 에이전트 이름
            ai_ds_agent_class: AI_DS_Team 에이전트 클래스
            llm_instance: LLM 인스턴스
        """
        super().__init__(agent_name, f"AI Data Science {agent_name}")
        self.ai_ds_agent_class = ai_ds_agent_class
        self.llm_instance = llm_instance
        self.ai_ds_agent = None
        
        # AI_DS_Team 에이전트 초기화
        self._initialize_ai_ds_agent()
    
    def _initialize_ai_ds_agent(self):
        """AI_DS_Team 에이전트 초기화"""
        try:
            if self.llm_instance:
                self.ai_ds_agent = self.ai_ds_agent_class(model=self.llm_instance)
            else:
                self.ai_ds_agent = self.ai_ds_agent_class()
            print(f"✅ AI_DS_Team {self.agent_name} initialized")
        except Exception as e:
            print(f"❌ Failed to initialize AI_DS_Team {self.agent_name}: {e}")
    
    async def _execute_with_tracking(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """AI Data Science 에이전트 실행"""
        try:
            # 1. 사용자 입력 분석
            user_input = self.get_user_input_from_context(context)
            
            with self.trace_processing_step("user_input_analysis"):
                analysis_result = await self._analyze_user_input(user_input)
                self.log_user_input_analysis(user_input, analysis_result)
            
            # 2. 데이터 컨텍스트 준비
            with self.trace_processing_step("data_context_preparation"):
                data_context = await self._prepare_data_context(analysis_result)
            
            # 3. AI_DS_Team 에이전트 실행
            with self.trace_processing_step("ai_ds_agent_execution"):
                result = await self._execute_ai_ds_agent(user_input, data_context, task_updater)
            
            # 4. 결과 후처리
            with self.trace_processing_step("result_post_processing"):
                final_result = await self._post_process_result(result)
            
            return final_result
            
        except Exception as e:
            error_msg = f"❌ {self.agent_name} execution failed: {str(e)}"
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
            
            await task_updater.update_status(
                TaskState.failed,
                message=error_msg
            )
            return self.format_response_for_a2a(error_msg)
    
    async def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력 분석"""
        # 간단한 분석 로직 - 실제로는 더 복잡할 수 있음
        analysis = {
            "intent": "data_analysis",
            "input_length": len(user_input),
            "contains_data_keywords": any(keyword in user_input.lower() 
                                        for keyword in ["analyze", "chart", "plot", "data", "statistics"]),
            "complexity": "medium"
        }
        return analysis
    
    async def _prepare_data_context(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 컨텍스트 준비"""
        data_context = {
            "available_data": [],
            "session_id": None,
            "data_manager_available": self.data_manager is not None
        }
        
        if self.data_manager:
            try:
                # 사용 가능한 데이터 목록 가져오기
                data_context["available_data"] = self.data_manager.list_available_data()
            except Exception as e:
                print(f"⚠️ Error preparing data context: {e}")
        
        return data_context
    
    async def _execute_ai_ds_agent(self, 
                                  user_input: str,
                                  data_context: Dict[str, Any],
                                  task_updater: TaskUpdater) -> Any:
        """AI_DS_Team 에이전트 실행"""
        if not self.ai_ds_agent:
            raise Exception("AI_DS_Team agent not initialized")
        
        # 프롬프트 준비
        enhanced_prompt = self._prepare_enhanced_prompt(user_input, data_context)
        
        # LLM 호출 추적
        if hasattr(self.ai_ds_agent, 'model'):
            model_name = getattr(self.ai_ds_agent.model, 'model_name', 'unknown')
            self.log_llm_call(
                model_name=model_name,
                prompt=enhanced_prompt,
                response="[AI_DS_Agent processing...]"
            )
        
        # AI_DS_Team 에이전트 실행
        result = await asyncio.to_thread(
            self.ai_ds_agent.run,
            enhanced_prompt
        )
        
        # 결과 로깅
        self.log_data_analysis_result(
            analysis_type=self.agent_name,
            data_summary=data_context,
            results={"result_type": type(result).__name__, "result_length": len(str(result))}
        )
        
        return result
    
    def _prepare_enhanced_prompt(self, user_input: str, data_context: Dict[str, Any]) -> str:
        """향상된 프롬프트 준비"""
        enhanced_prompt = f"""
User Request: {user_input}

Available Data Context:
{json.dumps(data_context, indent=2)}

Please provide a comprehensive analysis with:
1. Data exploration and understanding
2. Relevant visualizations
3. Statistical insights
4. Actionable recommendations

Ensure all code is properly documented and results are clearly explained.
"""
        return enhanced_prompt
    
    async def _post_process_result(self, result: Any) -> List[TextPart]:
        """결과 후처리"""
        try:
            if isinstance(result, str):
                response_content = result
            elif isinstance(result, dict):
                response_content = json.dumps(result, indent=2)
            elif isinstance(result, list):
                response_content = "\n".join(str(item) for item in result)
            else:
                response_content = str(result)
            
            return self.format_response_for_a2a(response_content)
            
        except Exception as e:
            error_msg = f"Error processing result: {str(e)}"
            print(f"⚠️ {error_msg}")
            return self.format_response_for_a2a(error_msg)

# 헬퍼 함수들
def create_enhanced_executor(agent_name: str, 
                           ai_ds_agent_class,
                           llm_instance=None) -> EnhancedAIDataScienceExecutor:
    """Enhanced AI Data Science Executor 생성"""
    return EnhancedAIDataScienceExecutor(
        agent_name=agent_name,
        ai_ds_agent_class=ai_ds_agent_class,
        llm_instance=llm_instance
    ) 