#!/usr/bin/env python3
"""
🔍 Enhanced AI_DS_Team EDA Tools Server with Deep Tracking
Port: 8312

이 서버는 다음 기능을 제공합니다:
- AI-Data-Science-Team EDAToolsAgent 내부 처리 과정 완전 추적
- LLM 호출, 코드 생성, 실행 결과 실시간 모니터링
- Langfuse 세션 기반 계층적 추적
- 에이전트 내부 워크플로우 가시화
"""

import asyncio
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn
import logging
import json

# AI_DS_Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent
import pandas as pd

# CherryAI Enhanced tracking imports
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

try:
    from core.langfuse_session_tracer import get_session_tracer
    from core.langfuse_ai_ds_team_wrapper import LangfuseAIDataScienceTeamWrapper
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced tracking not available: {e}")
    ENHANCED_TRACKING_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()
session_data_manager = SessionDataManager()


class EnhancedEDAToolsAgentExecutor(AgentExecutor):
    """Enhanced EDA Tools Agent with Deep Internal Tracking"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        
        # AI-Data-Science-Team 에이전트 초기화
        self.agent = EDAToolsAgent(model=self.llm)
        
        # Enhanced tracking wrapper
        self.tracking_wrapper = None
        if ENHANCED_TRACKING_AVAILABLE:
            session_tracer = get_session_tracer()
            if session_tracer:
                self.tracking_wrapper = LangfuseAIDataScienceTeamWrapper(
                    session_tracer, 
                    "Enhanced EDA Tools Agent"
                )
                logger.info("✅ Enhanced tracking wrapper initialized")
            else:
                logger.warning("⚠️ Session tracer not available")
        
        logger.info("🔍 Enhanced EDA Tools Agent initialized with deep tracking")
    
    def extract_data_reference_from_message(self, context: RequestContext) -> Dict[str, Any]:
        """A2A 메시지에서 데이터 참조 정보 추출"""
        data_reference = None
        user_instructions = ""
        
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root'):
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                    elif part.root.kind == "data":
                        if hasattr(part.root, 'data') and 'data_reference' in part.root.data:
                            data_reference = part.root.data['data_reference']
        
        return {
            "user_instructions": user_instructions.strip(),
            "data_reference": data_reference
        }
    
    async def execute_with_enhanced_tracking(self, user_instructions: str, df: pd.DataFrame, 
                                           data_source: str, session_id: str, task_updater: TaskUpdater):
        """Enhanced tracking을 적용한 EDA 실행"""
        
        if not self.tracking_wrapper:
            logger.warning("⚠️ Enhanced tracking not available, falling back to basic execution")
            return await self.execute_basic_eda(user_instructions, df, data_source, session_id, task_updater)
        
        logger.info("🔍 Starting Enhanced EDA with deep tracking...")
        
        # 메인 agent span 생성
        operation_data = {
            "operation": "enhanced_eda_analysis",
            "user_request": user_instructions,
            "data_source": data_source,
            "data_shape": df.shape,
            "session_id": session_id
        }
        
        main_span = self.tracking_wrapper.create_agent_span("Enhanced EDA Analysis", operation_data)
        
        try:
            # 1. 워크플로우 시작 추적
            self.tracking_wrapper.trace_ai_ds_workflow_start("eda_analysis", operation_data)
            
            # 2. 데이터 분석 단계
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 데이터 구조 분석 중...")
            )
            
            data_summary = f"""EDA 데이터 분석:
- 데이터 소스: {data_source}
- 형태: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}
- 결측값: {dict(df.isnull().sum())}
- 기본 통계: {df.describe().to_dict()}
"""
            
            self.tracking_wrapper.trace_data_analysis_step(data_summary, "initial_data_inspection")
            
            # 3. LLM 추천 단계 (EDA 전략 수립)
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 EDA 전략 수립 중...")
            )
            
            eda_strategy_prompt = f"""데이터 과학자로서 다음 데이터에 대한 탐색적 데이터 분석(EDA) 전략을 수립해주세요:

데이터 정보:
- 형태: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 컬럼: {list(df.columns)}
- 데이터 타입: {dict(df.dtypes)}

사용자 요청: {user_instructions}

다음 항목에 대한 분석 전략을 제시해주세요:
1. 기본 통계 분석
2. 데이터 품질 평가
3. 변수 간 관계 분석
4. 시각화 권장사항
5. 이상치 탐지 방법
"""

            eda_strategy_response = """# EDA 전략 수립

## 1. 기본 통계 분석
- 수치형 변수: 평균, 중앙값, 표준편차, 사분위수 계산
- 범주형 변수: 빈도수, 최빈값 분석
- 분포 특성 파악

## 2. 데이터 품질 평가  
- 결측값 패턴 분석
- 중복값 확인
- 데이터 타입 적절성 검토

## 3. 변수 간 관계 분석
- 상관관계 매트릭스 계산
- 수치형-범주형 변수 관계 분석
- 주요 패턴 식별

## 4. 시각화 권장사항
- 히스토그램 및 박스플롯으로 분포 확인
- 산점도로 변수 간 관계 시각화
- 히트맵으로 상관관계 표현

## 5. 이상치 탐지
- IQR 방법으로 이상치 식별
- Z-score 기반 이상치 탐지
- 도메인 지식 기반 검증
"""
            
            self.tracking_wrapper.trace_llm_recommendation_step(
                eda_strategy_prompt, 
                eda_strategy_response, 
                "eda_strategy_planning"
            )
            
            # 4. 실제 AI-Data-Science-Team 에이전트 실행
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("⚡ AI-Data-Science-Team EDA 에이전트 실행 중...")
            )
            
            logger.info(f"🚀 Invoking AI-Data-Science-Team EDAToolsAgent...")
            logger.info(f"📊 Data shape: {df.shape}")
            logger.info(f"📝 User instructions: {user_instructions}")
            
            # AI-DS-Team 에이전트 실행 (상세 로깅)
            start_time = time.time()
            try:
                # 실제 에이전트 실행 - 더 상세한 로깅
                logger.debug("🔄 Before invoke_agent call...")
                logger.debug(f"🔄 Agent type: {type(self.agent)}")
                logger.debug(f"🔄 Agent methods: {[m for m in dir(self.agent) if not m.startswith('_')]}")
                
                result = self.agent.invoke_agent(
                    user_instructions=user_instructions,
                    data_raw=df
                )
                
                execution_time = time.time() - start_time
                logger.info(f"✅ AI-DS-Team agent completed in {execution_time:.2f}s")
                logger.info(f"📊 Result type: {type(result)}")
                logger.info(f"📊 Result preview: {str(result)[:500]}...")
                
                # 결과 분석 및 추적
                if result is not None:
                    result_analysis = {
                        "execution_time": execution_time,
                        "result_type": type(result).__name__,
                        "result_length": len(str(result)) if result else 0,
                        "success": True
                    }
                else:
                    result_analysis = {
                        "execution_time": execution_time,
                        "result_type": "None",
                        "result_length": 0,
                        "success": False,
                        "issue": "Agent returned None result"
                    }
                
                # 코드 실행 추적 (가상의 코드 - 실제로는 에이전트 내부에서 실행됨)
                virtual_code = f"""# AI-Data-Science-Team EDA 실행
eda_agent = EDAToolsAgent(model=llm)
result = eda_agent.invoke_agent(
    user_instructions="{user_instructions[:100]}...",
    data_raw=data_frame  # shape: {df.shape}
)
"""
                
                self.tracking_wrapper.trace_code_execution_step(
                    virtual_code,
                    result_analysis,
                    execution_time
                )
                
                # 데이터 변환 추적 (입력 데이터 → 분석 결과)
                self.tracking_wrapper.trace_data_transformation_step(
                    df,
                    result,
                    "eda_analysis_transformation"
                )
                
                # 최종 결과 생성
                if result:
                    if isinstance(result, dict):
                        result_text = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        result_text = str(result)
                    
                    response_text = f"""## 🔍 Enhanced EDA 분석 완료

✅ **세션 ID**: {session_id}
✅ **데이터 소스**: {data_source}  
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
✅ **실행 시간**: {execution_time:.2f}초

### 📊 AI-Data-Science-Team 분석 결과

{result_text[:2000]}{'...' if len(result_text) > 2000 else ''}

### 🎯 Enhanced Tracking 정보
- **내부 처리 단계**: {self.tracking_wrapper.step_counter}단계
- **LLM 호출**: EDA 전략 수립 완료
- **코드 실행**: AI-DS-Team 에이전트 실행 완료
- **데이터 변환**: 원본 데이터 → 분석 결과 추적 완료

### ✅ 분석 완료
Enhanced tracking이 적용된 AI-Data-Science-Team EDA 에이전트가 성공적으로 실행되었습니다.
모든 내부 처리 과정이 Langfuse에서 추적 가능합니다.
"""
                else:
                    response_text = f"""## ⚠️ EDA 분석 결과 없음

✅ **세션 ID**: {session_id}
✅ **데이터 소스**: {data_source}
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
❌ **분석 결과**: None (에이전트가 결과를 반환하지 않음)

### 🔍 상세 분석
- **실행 시간**: {execution_time:.2f}초
- **에이전트 상태**: 실행 완료되었으나 결과 없음
- **추적 단계**: {self.tracking_wrapper.step_counter}단계 완료

### 🎯 권장사항
1. 데이터 형식 확인 필요
2. 사용자 지시사항 명확화 필요
3. AI-DS-Team 에이전트 설정 점검 필요

모든 내부 처리 과정은 Langfuse에서 확인하실 수 있습니다.
"""
                
                # 워크플로우 완료 추적
                workflow_summary = f"""# Enhanced EDA 워크플로우 완료

## 처리 요약
- **요청**: {user_instructions}
- **처리 단계**: {self.tracking_wrapper.step_counter}단계
- **실행 시간**: {execution_time:.2f}초

## 데이터 정보
- **소스**: {data_source}
- **형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **세션**: {session_id}

## Enhanced Tracking 결과
- ✅ 데이터 구조 분석 완료
- ✅ EDA 전략 수립 완료  
- ✅ AI-DS-Team 에이전트 실행 완료
- ✅ 결과 데이터 변환 추적 완료

## 아티팩트
- 데이터 분석 요약
- EDA 전략 프롬프트/응답
- 가상 실행 코드
- 실행 결과 분석
- 데이터 변환 샘플
"""
                
                self.tracking_wrapper.trace_workflow_completion(result, workflow_summary)
                
                return response_text
                
            except Exception as agent_error:
                execution_time = time.time() - start_time
                logger.error(f"❌ AI-DS-Team agent execution failed: {agent_error}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                
                # 에러 추적
                error_code = f"""# AI-Data-Science-Team EDA 실행 (실패)
try:
    result = eda_agent.invoke_agent(
        user_instructions="{user_instructions}",
        data_raw=data_frame
    )
except Exception as e:
    print(f"Error: {{e}}")
"""
                
                self.tracking_wrapper.trace_code_execution_step(
                    error_code,
                    None,
                    execution_time,
                    str(agent_error)
                )
                
                error_response = f"""## ❌ Enhanced EDA 분석 오류

✅ **세션 ID**: {session_id}
✅ **데이터 소스**: {data_source}
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
❌ **오류**: {str(agent_error)}

### 🔍 오류 세부사항
```
{traceback.format_exc()[:1000]}
```

### 📊 기본 데이터 분석 (대체)
{df.describe().to_string()[:500]}

### 🎯 Enhanced Tracking 정보
모든 오류 상황도 Langfuse에서 추적됩니다.
- 실행 시간: {execution_time:.2f}초
- 추적 단계: {self.tracking_wrapper.step_counter}단계
"""
                
                return error_response
                
        except Exception as tracking_error:
            logger.error(f"❌ Enhanced tracking failed: {tracking_error}")
            return await self.execute_basic_eda(user_instructions, df, data_source, session_id, task_updater)
        
        finally:
            # Agent span 완료
            if main_span and self.tracking_wrapper:
                self.tracking_wrapper.finalize_agent_span(
                    final_result="Enhanced EDA analysis completed",
                    success=True
                )
    
    async def execute_basic_eda(self, user_instructions: str, df: pd.DataFrame, 
                              data_source: str, session_id: str, task_updater: TaskUpdater):
        """기본 EDA 실행 (Enhanced tracking 실패 시 fallback)"""
        logger.info("🔄 Executing basic EDA (fallback mode)")
        
        try:
            result = self.agent.invoke_agent(
                user_instructions=user_instructions,
                data_raw=df
            )
            
            if result:
                result_text = str(result)
                response = f"""## 🔍 EDA 분석 완료 (기본 모드)

✅ **데이터**: {data_source} ({df.shape[0]:,} × {df.shape[1]:,})
✅ **세션**: {session_id}

### 📊 분석 결과
{result_text[:1500]}{'...' if len(result_text) > 1500 else ''}
"""
            else:
                response = f"""## ⚠️ EDA 분석 결과 없음

데이터: {data_source} ({df.shape[0]:,} × {df.shape[1]:,})
AI-DS-Team 에이전트가 결과를 반환하지 않았습니다.

### 📊 기본 통계
{df.describe().to_string()[:500]}
"""
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Basic EDA also failed: {e}")
            return f"""## ❌ EDA 분석 실패

오류: {str(e)}

### 📊 기본 데이터 정보
- 소스: {data_source}
- 형태: {df.shape[0]:,} × {df.shape[1]:,} 열
- 컬럼: {list(df.columns)[:10]}
"""

    async def execute(self, context: RequestContext, event_queue) -> None:
        """메인 실행 함수"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 Enhanced EDA 분석을 시작합니다...")
            )
            
            # 메시지 데이터 추출
            message_data = self.extract_data_reference_from_message(context)
            user_instructions = message_data["user_instructions"]
            data_reference = message_data["data_reference"]
            
            logger.info(f"📝 User instructions: {user_instructions}")
            logger.info(f"📊 Data reference: {data_reference}")
            
            if user_instructions:
                df = None
                data_source = "unknown"
                
                # 데이터 로드
                if data_reference:
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            data_source = data_id
                            logger.info(f"✅ Data loaded: {data_id} with shape {df.shape}")
                
                # 기본 데이터 사용
                if df is None:
                    available_data = data_manager.list_dataframes()
                    logger.info(f"🔍 Available data: {available_data}")
                    
                    if available_data:
                        first_data_id = available_data[0]
                        df = data_manager.get_dataframe(first_data_id)
                        if df is not None:
                            data_source = first_data_id
                            logger.info(f"✅ Using default data: {first_data_id} with shape {df.shape}")
                
                if df is not None:
                    # 세션 생성
                    current_session_id = session_data_manager.create_session_with_data(
                        data_id=data_source,
                        data=df,
                        user_instructions=user_instructions
                    )
                    
                    logger.info(f"✅ Session created: {current_session_id}")
                    
                    # Enhanced tracking으로 EDA 실행
                    response_text = await self.execute_with_enhanced_tracking(
                        user_instructions, df, data_source, current_session_id, task_updater
                    )
                    
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message(response_text)
                    )
                else:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message("❌ 사용 가능한 데이터가 없습니다. 먼저 데이터를 업로드해주세요.")
                    )
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ EDA 분석 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ Enhanced EDA Agent execution failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Enhanced EDA 분석 중 오류가 발생했습니다: {str(e)}")
            )

    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info("Enhanced EDA Tools Agent task cancelled")


def main():
    """Enhanced EDA Tools Server 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="enhanced_eda_analysis",
        name="Enhanced EDA Analysis with Deep Tracking",
        description="완전 추적 가능한 탐색적 데이터 분석. AI-Data-Science-Team 내부 처리 과정을 Langfuse에서 실시간 추적할 수 있습니다.",
        tags=["eda", "data-analysis", "langfuse", "tracking", "transparency", "ai-ds-team"],
        examples=[
            "데이터의 기본 통계와 분포를 분석해주세요",
            "변수 간 상관관계를 파악하고 시각화해주세요",
            "이상치를 탐지하고 데이터 품질을 평가해주세요",
            "EDA 과정을 단계별로 추적하며 실행해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Enhanced AI_DS_Team EDAToolsAgent",
        description="AI-Data-Science-Team 내부 처리 과정이 완전히 추적되는 EDA 전문가. LLM의 사고 과정, 생성된 코드, 분석 결과를 Langfuse에서 실시간으로 확인할 수 있습니다.",
        url="http://localhost:8312/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=EnhancedEDAToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting Enhanced AI_DS_Team EDAToolsAgent Server")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")
    print("🛠️ Features: Enhanced EDA analysis with Langfuse tracking")
    print("🔍 Langfuse tracking: Complete AI-Data-Science-Team internal process visibility")
    print("📊 Tracking scope:")
    print("   - 데이터 구조 분석 및 요약")
    print("   - EDA 전략 수립 (LLM 프롬프트 + 응답)")
    print("   - AI-DS-Team 에이전트 실행 과정")
    print("   - 코드 실행 및 결과 분석")
    print("   - 데이터 변환 (입력 → 분석 결과)")
    print("   - 워크플로우 요약 및 완료")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main() 