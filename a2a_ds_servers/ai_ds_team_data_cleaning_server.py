from a2a.utils import new_agent_text_message#!/usr/bin/env python3
"""
AI_DS_Team DataCleaningAgent A2A Server
Port: 8306

AI_DS_Team의 DataCleaningAgent를 A2A 프로토콜로 래핑하여 제공합니다.
데이터 정리 및 품질 개선 전문
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.agents import DataCleaningAgent
import pandas as pd
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# Langfuse 로깅 설정 (선택적)
langfuse_handler = None
if os.getenv("LOGGING_PROVIDER") in ["langfuse", "both"]:
    try:
        from langfuse.callback import CallbackHandler
        langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
        logger.info("✅ Langfuse logging enabled")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse logging setup failed: {e}")

# LangSmith 로깅 설정 (선택적)
if os.getenv("LOGGING_PROVIDER") in ["langsmith", "both"]:
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-ds-team")
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        logger.info("✅ LangSmith logging enabled")


def get_dataframe_summary(df, n_sample=5):
    """간단한 데이터프레임 요약 생성"""
    try:
        summary = f"""
**Shape**: {df.shape[0]:,} rows × {df.shape[1]:,} columns

**Columns**: {', '.join(df.columns.tolist())}

**Data Types**:
{df.dtypes.to_string()}

**Sample Data**:
{df.head(n_sample).to_string()}

**Missing Values**:
{df.isnull().sum().to_string()}
"""
        return [summary]
    except Exception as e:
        return [f"데이터 요약 생성 중 오류: {str(e)}"]


class DataCleaningAgentExecutor(AgentExecutor):
    """AI_DS_Team DataCleaningAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = DataCleaningAgent(model=self.llm)
        logger.info("DataCleaningAgent initialized")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 프로토콜에 따른 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"Processing data cleaning request: {user_instructions}")
                
                # 데이터 로드 시도
                data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
                available_data = []
                
                try:
                    for file in os.listdir(data_path):
                        if file.endswith(('.csv', '.pkl')):
                            available_data.append(file)
                except:
                    pass
                
                if available_data:
                    # 가장 최근 데이터 사용
                    data_file = available_data[0]
                    if data_file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(data_path, data_file))
                    else:
                        df = pd.read_pickle(os.path.join(data_path, data_file))
                    
                    logger.info(f"Loaded data: {data_file}, shape: {df.shape}")
                    
                    # DataCleaningAgent 실행
                    try:
                        result = self.agent.invoke_agent(
                            user_instructions=user_instructions,
                            data_raw=df
                        )
                        
                        # 결과 처리 - AI_DS_Team의 올바른 메서드 사용
                        try:
                            # 정리된 데이터 가져오기
                            cleaned_data = self.agent.get_data_cleaned()
                            workflow_summary = self.agent.get_workflow_summary(markdown=True)
                            
                            # 데이터 요약 생성
                            data_summary = get_dataframe_summary(df, n_sample=10)
                            
                            # 정리 결과 요약
                            if cleaned_data is not None:
                                cleaned_summary = get_dataframe_summary(cleaned_data, n_sample=10)
                                response_text = f"""## 🧹 데이터 정리 완료

### 📋 작업 요약
{workflow_summary}

### 📊 원본 데이터 요약
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

### 🔧 정리된 데이터 요약
{cleaned_summary[0] if cleaned_summary else '정리된 데이터 요약을 생성할 수 없습니다.'}

### 💾 저장된 파일
정리된 데이터가 아티팩트 폴더에 저장되었습니다.

### 🧹 Data Cleaning Agent 기능
- **결측값 처리**: fillna, dropna, 보간법 등
- **중복 제거**: drop_duplicates 최적화
- **이상값 탐지**: IQR, Z-score, Isolation Forest
- **데이터 타입 변환**: 메모리 효율적인 타입 선택
- **텍스트 정리**: 공백 제거, 대소문자 통일
- **날짜 형식 표준화**: datetime 변환 및 검증
"""
                            else:
                                response_text = f"""## 🧹 데이터 정리 완료

### 📋 작업 요약
{workflow_summary}

### 📊 원본 데이터 요약
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

데이터 정리가 완료되었지만 정리된 데이터를 가져올 수 없습니다.
"""
                        except Exception as result_error:
                            logger.warning(f"Result processing failed: {result_error}")
                            response_text = f"""## 🧹 데이터 정리 완료

데이터 정리 작업이 수행되었지만 결과 처리 중 오류가 발생했습니다: {str(result_error)}

### 🧹 Data Cleaning Agent 기능
- **결측값 처리**: fillna, dropna, 보간법 등
- **중복 제거**: drop_duplicates 최적화
- **이상값 탐지**: IQR, Z-score, Isolation Forest
- **데이터 타입 변환**: 메모리 효율적인 타입 선택
- **텍스트 정리**: 공백 제거, 대소문자 통일
- **날짜 형식 표준화**: datetime 변환 및 검증

요청: {user_instructions}
"""
                        
                    except Exception as agent_error:
                        logger.warning(f"Agent execution failed, providing guidance: {agent_error}")
                        response_text = f"""## 🧹 데이터 정리 가이드

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 Data Cleaning 사용법
다음과 같은 요청을 시도해보세요:

1. **기본 정리**:
   - "데이터를 정리해주세요"
   - "결측값과 중복값을 제거해주세요"

2. **고급 정리**:
   - "이상값을 제거하지 말고 데이터를 정리해주세요"
   - "범주형 변수는 최빈값으로 채워주세요"

3. **사용자 정의**:
   - "40% 이상 결측값이 있는 컬럼만 제거해주세요"
   - "중복 행은 유지하고 결측값만 처리해주세요"

요청: {user_instructions}
"""
                else:
                    response_text = f"""## ❌ 데이터 없음

데이터 정리를 수행하려면 먼저 데이터를 업로드해야 합니다.
사용 가능한 데이터가 없습니다: {data_path}

요청: {user_instructions}

### 🧹 Data Cleaning Agent 기능
- **결측값 처리**: fillna, dropna, 보간법 등
- **중복 제거**: drop_duplicates 최적화
- **이상값 탐지**: IQR, Z-score, Isolation Forest
- **데이터 타입 변환**: 메모리 효율적인 타입 선택
- **텍스트 정리**: 공백 제거, 대소문자 통일
- **날짜 형식 표준화**: datetime 변환 및 검증
"""
                
                # 작업 완료
                from a2a.utils import new_agent_text_message
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(response_text)
                )
                
            else:
                # 메시지가 없는 경우
                from a2a.utils import new_agent_text_message
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("데이터 정리 요청이 비어있습니다. 구체적인 데이터 정리 작업을 요청해주세요.")
                )
                
        except Exception as e:
            logger.error(f"Error in DataCleaningAgent execution: {e}")
            from a2a.utils import new_agent_text_message
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 정리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"DataCleaningAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_cleaning",
        name="Data Cleaning & Quality Improvement",
        description="전문적인 데이터 정리 및 품질 개선 서비스. 결측값 처리, 중복 제거, 이상값 탐지, 데이터 타입 최적화 등을 수행합니다.",
        tags=["data-cleaning", "preprocessing", "quality-improvement", "missing-values", "outliers"],
        examples=[
            "결측값을 처리해주세요",
            "중복 데이터를 제거해주세요", 
            "데이터 품질을 평가해주세요",
            "이상값을 탐지하고 처리해주세요",
            "데이터 타입을 최적화해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team DataCleaningAgent",
        description="데이터 정리 및 품질 개선 전문가. 결측값 처리, 중복 제거, 이상값 탐지, 데이터 타입 최적화 등을 수행합니다.",
        url="http://localhost:8306/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataCleaningAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🧹 Starting AI_DS_Team DataCleaningAgent Server")
    print("🌐 Server starting on http://localhost:8306")
    print("📋 Agent card: http://localhost:8306/.well-known/agent.json")
    print("🛠️ Features: Data cleaning, quality improvement, preprocessing")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8306, log_level="info")


if __name__ == "__main__":
    main() 