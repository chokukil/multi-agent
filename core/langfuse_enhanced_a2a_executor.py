"""
🔍 Langfuse Enhanced A2A Executor
AI-Data-Science-Team 라이브러리 내부 처리 과정 추적이 통합된 A2A Executor

이 모듈은 다음 기능을 제공합니다:
- A2A AgentExecutor에 Langfuse 추적 기능 통합
- AI-Data-Science-Team 라이브러리 내부 처리 과정 자동 추적
- LLM 단계별 프롬프트/응답 및 코드 생성/실행 과정 완전 가시화
- 세션 기반 계층적 추적 구조
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater, TaskState
from a2a.utils import new_agent_text_message

try:
    from core.langfuse_session_tracer import get_session_tracer
    from core.langfuse_ai_ds_team_wrapper import LangfuseAIDataScienceTeamWrapper
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class LangfuseEnhancedA2AExecutor(AgentExecutor):
    """
    Langfuse 추적 기능이 통합된 A2A Executor 기본 클래스
    
    AI-Data-Science-Team 라이브러리 사용 시 내부 처리 과정을 자동으로 추적합니다.
    """
    
    def __init__(self, agent_name: str = "AI_DS_Agent"):
        """
        Args:
            agent_name: 추적할 에이전트 이름
        """
        super().__init__()
        self.agent_name = agent_name
        self.session_tracer = None
        self.ai_ds_wrapper = None
        
        # Langfuse 초기화
        self._initialize_langfuse()
        
    def _initialize_langfuse(self):
        """Langfuse 추적 시스템 초기화"""
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse 모듈을 사용할 수 없습니다. 추적이 비활성화됩니다.")
            return
            
        try:
            self.session_tracer = get_session_tracer()
            if self.session_tracer:
                logger.info(f"✅ {self.agent_name} - Langfuse 추적 시스템 초기화 성공")
            else:
                logger.warning(f"⚠️ {self.agent_name} - Langfuse session tracer를 가져올 수 없습니다")
        except Exception as e:
            logger.error(f"❌ {self.agent_name} - Langfuse 초기화 실패: {e}")
    
    def create_ai_ds_wrapper(self, operation_name: str) -> Optional[LangfuseAIDataScienceTeamWrapper]:
        """AI-Data-Science-Team 작업을 위한 wrapper 생성"""
        if not self.session_tracer:
            return None
            
        wrapper = LangfuseAIDataScienceTeamWrapper(self.session_tracer, self.agent_name)
        wrapper.create_agent_span(operation_name, {
            "agent_name": self.agent_name,
            "operation": operation_name,
            "timestamp": time.time()
        })
        return wrapper
    
    def trace_ai_ds_team_invoke(self, agent, method_name: str, **kwargs):
        """
        AI-Data-Science-Team 에이전트 메서드 호출을 추적하는 래퍼
        
        Usage:
            result = self.trace_ai_ds_team_invoke(
                self.agent, 
                'invoke_agent',
                user_instructions=user_instructions,
                data_raw=df
            )
        """
        if not self.ai_ds_wrapper:
            # 추적이 비활성화된 경우 원본 메서드 호출
            method = getattr(agent, method_name)
            return method(**kwargs)
        
        start_time = time.time()
        
        try:
            # 메서드 호출 전 추적 시작
            self.ai_ds_wrapper.trace_data_transformation(
                input_data=kwargs,
                output_data=None,
                operation=f"{method_name}_start"
            )
            
            # AI-Data-Science-Team 에이전트 원본 메서드 호출
            method = getattr(agent, method_name)
            result = method(**kwargs)
            
            execution_time = time.time() - start_time
            
            # 메서드 호출 후 결과 추적
            self.ai_ds_wrapper.trace_data_transformation(
                input_data=kwargs,
                output_data=result,
                operation=f"{method_name}_complete",
                metadata={"execution_time": execution_time}
            )
            
            # 추가적인 정보 추적 (가능한 경우)
            self._trace_agent_internal_data(agent)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            # 에러 추적
            self.ai_ds_wrapper.trace_code_execution(
                code="AI-Data-Science-Team method call",
                result=None,
                execution_time=execution_time,
                error=error_msg
            )
            
            # 원본 에러 재발생
            raise
    
    def _trace_agent_internal_data(self, agent):
        """AI-Data-Science-Team 에이전트의 내부 데이터 추적"""
        if not self.ai_ds_wrapper:
            return
            
        try:
            # 생성된 함수 코드 추적
            if hasattr(agent, 'get_data_cleaner_function'):
                code = agent.get_data_cleaner_function()
                if code:
                    self.ai_ds_wrapper.trace_code_execution(
                        code=code,
                        result="Code generated successfully",
                        metadata={"type": "data_cleaner_function"}
                    )
            elif hasattr(agent, 'get_data_wrangler_function'):
                code = agent.get_data_wrangler_function()
                if code:
                    self.ai_ds_wrapper.trace_code_execution(
                        code=code,
                        result="Code generated successfully",
                        metadata={"type": "data_wrangler_function"}
                    )
            elif hasattr(agent, 'get_feature_engineer_function'):
                code = agent.get_feature_engineer_function()
                if code:
                    self.ai_ds_wrapper.trace_code_execution(
                        code=code,
                        result="Code generated successfully",
                        metadata={"type": "feature_engineer_function"}
                    )
            elif hasattr(agent, 'get_data_visualization_function'):
                code = agent.get_data_visualization_function()
                if code:
                    self.ai_ds_wrapper.trace_code_execution(
                        code=code,
                        result="Code generated successfully",
                        metadata={"type": "data_visualization_function"}
                    )
            
            # 권장 단계 추적
            if hasattr(agent, 'get_recommended_cleaning_steps'):
                steps = agent.get_recommended_cleaning_steps()
                if steps:
                    self.ai_ds_wrapper.trace_llm_step(
                        step_name="recommended_cleaning_steps",
                        prompt="Generate cleaning recommendations",
                        response=steps,
                        metadata={"type": "recommendations"}
                    )
            elif hasattr(agent, 'get_recommended_wrangling_steps'):
                steps = agent.get_recommended_wrangling_steps()
                if steps:
                    self.ai_ds_wrapper.trace_llm_step(
                        step_name="recommended_wrangling_steps",
                        prompt="Generate wrangling recommendations",
                        response=steps,
                        metadata={"type": "recommendations"}
                    )
            elif hasattr(agent, 'get_recommended_feature_engineering_steps'):
                steps = agent.get_recommended_feature_engineering_steps()
                if steps:
                    self.ai_ds_wrapper.trace_llm_step(
                        step_name="recommended_feature_engineering_steps",
                        prompt="Generate feature engineering recommendations",
                        response=steps,
                        metadata={"type": "recommendations"}
                    )
            
            # 워크플로우 요약 추적
            if hasattr(agent, 'get_workflow_summary'):
                try:
                    summary = agent.get_workflow_summary(markdown=True)
                    if summary:
                        self.ai_ds_wrapper.trace_llm_step(
                            step_name="workflow_summary",
                            prompt="Generate workflow summary",
                            response=summary,
                            metadata={"type": "summary"}
                        )
                except:
                    pass  # get_workflow_summary 메서드가 실패하는 경우 무시
                    
        except Exception as e:
            logger.warning(f"AI-Data-Science-Team 내부 데이터 추적 실패: {e}")
    
    def finalize_langfuse_tracking(self, final_result: Any = None, success: bool = True, error: Optional[str] = None):
        """Langfuse 추적 완료"""
        if self.ai_ds_wrapper:
            try:
                self.ai_ds_wrapper.finalize_agent_span(final_result, success, error)
            except Exception as e:
                logger.warning(f"Langfuse 추적 완료 실패: {e}")
            finally:
                self.ai_ds_wrapper = None


class EnhancedDataCleaningExecutor(LangfuseEnhancedA2AExecutor):
    """Langfuse 추적이 통합된 Data Cleaning Executor"""
    
    def __init__(self):
        super().__init__("Data Cleaning Agent")
        
        # LLM 설정 (langfuse 콜백은 LLM 팩토리에서 자동 처리)
        from core.llm_factory import create_llm_instance
        from ai_data_science_team.agents import DataCleaningAgent
        from core.data_manager import DataManager
        
        self.llm = create_llm_instance()
        self.agent = DataCleaningAgent(model=self.llm)
        self.data_manager = DataManager()
        
        logger.info("EnhancedDataCleaningExecutor initialized with Langfuse tracking")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Langfuse 추적이 통합된 실행 메서드"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # AI-Data-Science-Team wrapper 생성
        self.ai_ds_wrapper = self.create_ai_ds_wrapper("data_cleaning")
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_instructions = ""
            data_reference = None
            
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                    elif part.root.kind == "data" and hasattr(part.root, 'data'):
                        data_reference = part.root.data.get('data_reference', {})
                
                user_instructions = user_instructions.strip()
                logger.info(f"Processing data cleaning request: {user_instructions}")
                
                # 데이터 로딩
                available_data = self.data_manager.list_dataframes()
                
                if not available_data:
                    response_text = """## ❌ 데이터 없음

데이터 클리닝을 수행하려면 먼저 데이터를 업로드해야 합니다.

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "data.xlsx 파일을 클리닝해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

**현재 상태**: 사용 가능한 데이터가 없습니다.
"""
                else:
                    # 요청된 파일 찾기
                    data_file = None
                    if data_reference and 'data_id' in data_reference:
                        requested_id = data_reference['data_id']
                        if requested_id in available_data:
                            data_file = requested_id
                    
                    if data_file is None:
                        response_text = f"""## ❌ 요청된 데이터를 찾을 수 없음

**사용 가능한 데이터**: {', '.join(available_data)}

**해결 방법**:
1. 사용 가능한 데이터 중 하나를 선택하여 요청하세요
2. 원하는 파일을 먼저 업로드해주세요

**요청**: {user_instructions}
"""
                    else:
                        # 데이터 로드 및 처리
                        df = self.data_manager.get_dataframe(data_file)
                        if df is not None:
                            # 🔍 Langfuse 추적이 통합된 AI_DS_Team DataCleaningAgent 실행
                            result = self.trace_ai_ds_team_invoke(
                                self.agent,
                                'invoke_agent',
                                user_instructions=user_instructions,
                                data_raw=df
                            )
                            
                            # 결과 처리
                            cleaned_data = self.agent.get_data_cleaned()
                            
                            # 워크플로우 요약 생성
                            try:
                                workflow_summary = self.agent.get_workflow_summary(markdown=True)
                            except:
                                workflow_summary = f"✅ 데이터 정제 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                            
                            response_text = f"""## 🧹 데이터 정리 완료 (Langfuse 추적됨)

### 📋 작업 요약
{workflow_summary}

### 🔍 Langfuse 추적 정보
- **에이전트**: {self.agent_name}
- **세션 ID**: 현재 세션에서 추적됨
- **LLM 단계**: 권장 사항 생성, 코드 생성, 실행 추적
- **코드 아티팩트**: 생성된 Python 코드가 저장됨
- **데이터 변환**: 입력/출력 데이터 샘플이 저장됨

### 🧹 Data Cleaning Agent 기능
- **결측값 처리**: fillna, dropna, 보간법 등
- **중복 제거**: drop_duplicates 최적화
- **이상값 탐지**: IQR, Z-score, Isolation Forest
- **데이터 타입 변환**: 메모리 효율적인 타입 선택
- **텍스트 정리**: 공백 제거, 대소문자 통일
- **날짜 형식 표준화**: datetime 변환 및 검증

💡 **Langfuse에서 확인 가능한 정보**: 
- LLM이 생성한 데이터 정리 단계별 추천 사항
- 실제 생성된 Python 코드 (함수 형태)
- 코드 실행 과정 및 결과
- 데이터 변환 전후 비교
"""
                        else:
                            response_text = f"❌ 데이터 로드 실패: {data_file}"
                            self.finalize_langfuse_tracking(None, False, "데이터 로드 실패")
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(response_text)
            )
            
            # Langfuse 추적 완료
            self.finalize_langfuse_tracking(response_text, True)
            
        except Exception as e:
            logger.error(f"Error in EnhancedDataCleaningExecutor: {e}")
            error_msg = f"데이터 정리 중 오류 발생: {str(e)}"
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(error_msg)
            )
            
            # Langfuse 추적 완료 (에러)
            self.finalize_langfuse_tracking(None, False, error_msg)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"EnhancedDataCleaningExecutor task cancelled: {context.task_id}")
        self.finalize_langfuse_tracking(None, False, "작업 취소됨")


class EnhancedDataWranglingExecutor(LangfuseEnhancedA2AExecutor):
    """Langfuse 추적이 통합된 Data Wrangling Executor"""
    
    def __init__(self):
        super().__init__("Data Wrangling Agent")
        
        from core.llm_factory import create_llm_instance
        from ai_data_science_team.agents import DataWranglingAgent
        from core.data_manager import DataManager
        
        self.llm = create_llm_instance()
        self.agent = DataWranglingAgent(model=self.llm)
        self.data_manager = DataManager()
        
        logger.info("EnhancedDataWranglingExecutor initialized with Langfuse tracking")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Langfuse 추적이 통합된 실행 메서드"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # AI-Data-Science-Team wrapper 생성
        self.ai_ds_wrapper = self.create_ai_ds_wrapper("data_wrangling")
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 처리 로직 (기존과 동일)
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"Processing data wrangling request: {user_instructions}")
                
                # 데이터 로딩 및 처리
                available_data = self.data_manager.list_dataframes()
                
                if available_data:
                    # 첫 번째 사용 가능한 데이터 사용
                    data_file = list(available_data.keys())[0]
                    df = self.data_manager.get_dataframe(data_file)
                    
                    if df is not None:
                        # 🔍 Langfuse 추적이 통합된 AI_DS_Team DataWranglingAgent 실행
                        result = self.trace_ai_ds_team_invoke(
                            self.agent,
                            'invoke_agent',
                            user_instructions=user_instructions,
                            data_raw=df
                        )
                        
                        response_text = f"""## 🔧 데이터 랭글링 완료 (Langfuse 추적됨)

### 📋 요청 처리 완료
{user_instructions}

### 🔍 Langfuse 추적 정보
- **에이전트**: {self.agent_name}
- **랭글링 단계**: 추천 → 코드 생성 → 실행 추적
- **코드 아티팩트**: 생성된 데이터 변환 함수 저장
- **실행 결과**: 변환된 데이터 샘플 저장

**Langfuse에서 확인 가능**: LLM의 데이터 랭글링 전략, 생성된 코드, 실행 과정
"""
                    else:
                        response_text = "❌ 데이터 로드 실패"
                        self.finalize_langfuse_tracking(None, False, "데이터 로드 실패")
                else:
                    response_text = "❌ 사용 가능한 데이터가 없습니다"
                    self.finalize_langfuse_tracking(None, False, "데이터 없음")
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(response_text)
            )
            
            # Langfuse 추적 완료
            self.finalize_langfuse_tracking(response_text, True)
            
        except Exception as e:
            logger.error(f"Error in EnhancedDataWranglingExecutor: {e}")
            error_msg = f"데이터 랭글링 중 오류 발생: {str(e)}"
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(error_msg)
            )
            
            self.finalize_langfuse_tracking(None, False, error_msg)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"EnhancedDataWranglingExecutor task cancelled: {context.task_id}")
        self.finalize_langfuse_tracking(None, False, "작업 취소됨") 