from a2a.utils import new_agent_text_message#!/usr/bin/env python3
"""
AI_DS_Team DataLoaderToolsAgent A2A Server
Port: 8307

AI_DS_Team의 DataLoaderToolsAgent를 A2A 프로토콜로 래핑하여 제공합니다.
다양한 데이터 소스 로딩 및 전처리 전문
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
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.agents import DataLoaderToolsAgent
import pandas as pd
import json
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class DataLoaderToolsAgentExecutor(AgentExecutor):
    """AI_DS_Team DataLoaderToolsAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = DataLoaderToolsAgent(model=self.llm)
        logger.info("DataLoaderToolsAgent initialized")
    
    async def execute(self, context: RequestContext, event_queue) -> None:
        """A2A 프로토콜에 따른 실행"""
        # event_queue passed as parameter
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
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
                logger.info(f"Processing data loading request: {user_instructions}")
                
                # DataManager를 통한 데이터 관리
                data_manager = DataManager()
                available_data_ids = data_manager.list_dataframes()
                
                response_text = ""
                
                # 요청된 데이터 확인
                if data_reference and 'data_id' in data_reference:
                    requested_data_id = data_reference['data_id']
                    logger.info(f"Requested data: {requested_data_id}")
                    
                    if requested_data_id in available_data_ids:
                        # 요청된 데이터가 이미 로드되어 있음
                        df = data_manager.get_dataframe(requested_data_id)
                        if df is not None:
                            response_text = f"""## 📁 데이터 로딩 완료
✅ 요청하신 데이터가 이미 로드되어 있습니다.

**요청**: {user_instructions}

### 📊 로드된 데이터 정보
- **데이터 ID**: `{requested_data_id}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

### 📋 데이터 미리보기
```
{df.head().to_string()}
```

### 🔍 데이터 정보
```
{df.info()}
```
"""
                        else:
                            response_text = f"""## ❌ 데이터 로드 실패

요청하신 데이터 '{requested_data_id}'를 DataManager에서 로드할 수 없습니다.

**해결 방법**: 
1. UI에서 파일을 다시 업로드해주세요
2. 다른 사용 가능한 데이터를 선택해주세요
"""
                    else:
                        # 요청된 데이터가 없는 경우
                        if available_data_ids:
                            response_text = f"""## ❌ 요청된 데이터를 찾을 수 없음

요청하신 데이터 파일 '{requested_data_id}'을 찾을 수 없습니다.

### 📁 사용 가능한 데이터
{chr(10).join([f"- {data_id}" for data_id in available_data_ids])}

**해결 방법**:
1. 위의 사용 가능한 데이터 중 하나를 선택하여 요청하세요
2. 원하는 파일을 먼저 업로드해주세요

**요청**: {user_instructions}
"""
                        else:
                            response_text = f"""## ❌ 데이터 없음

데이터 로딩을 수행하려면 먼저 데이터를 업로드해야 합니다.

**요청**: {user_instructions}

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "{requested_data_id} 파일로 분석해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

**현재 상태**: 사용 가능한 데이터가 없습니다.
"""
                else:
                    # 데이터 참조가 없는 경우 - 일반적인 데이터 로딩 가이드
                    if available_data_ids:
                        # 첫 번째 데이터를 로드하지 말고 사용자에게 선택하도록 안내
                        response_text = f"""## 📁 데이터 로딩 가이드

**요청**: {user_instructions}

### 📁 사용 가능한 데이터
{chr(10).join([f"- {data_id}" for data_id in available_data_ids])}

### 💡 데이터 로딩 방법
구체적인 파일명을 명시하여 요청해주세요:

**예시**:
- "sales_data.csv 파일을 로드해주세요"
- "employee_data.csv로 분석을 시작해주세요"

### 🛠️ Data Loader Tools 기능
- **파일 로딩**: CSV, Excel, JSON, Parquet 등 다양한 형식 지원
- **데이터 검증**: 로드된 데이터의 품질 및 형식 검증
- **자동 타입 추론**: 컬럼 타입 자동 감지 및 변환
"""
                    else:
                        response_text = f"""## 📁 데이터 로딩 가이드

**요청**: {user_instructions}

### ❌ 사용 가능한 데이터가 없습니다

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "data.xlsx 파일을 로드해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

### 🛠️ Data Loader Tools 기능
- **파일 로딩**: CSV, Excel, JSON, Parquet 등 다양한 형식 지원
- **데이터베이스 연결**: SQL 데이터베이스 연결 및 쿼리
- **API 통합**: REST API를 통한 데이터 수집
- **데이터 검증**: 로드된 데이터의 품질 및 형식 검증
- **자동 타입 추론**: 컬럼 타입 자동 감지 및 변환
"""
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(response_text)
                )
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("데이터 로딩 요청이 비어있습니다. 로드할 데이터 파일이나 소스를 지정해주세요.")
                )
                
        except Exception as e:
            logger.error(f"Error in DataLoaderToolsAgent execution: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 로딩 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"DataLoaderToolsAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_loading",
        name="Data Loading & File Processing",
        description="다양한 데이터 소스 로딩 및 전처리 전문가. 파일, 데이터베이스, API 등에서 데이터를 로드하고 DataFrame으로 변환합니다.",
        tags=["data-loading", "etl", "file-processing", "database", "api-integration"],
        examples=[
            "CSV 파일을 로드해주세요",
            "데이터베이스에서 고객 테이블을 가져와주세요",
            "API에서 실시간 데이터를 수집해주세요",
            "Excel 파일의 특정 시트를 읽어주세요",
            "사용 가능한 데이터 파일들을 보여주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team DataLoaderToolsAgent",
        description="다양한 데이터 소스 로딩 및 전처리 전문가. 파일, 데이터베이스, API 등에서 데이터를 로드하고 DataFrame으로 변환합니다.",
        url="http://localhost:8307/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataLoaderToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📁 Starting AI_DS_Team DataLoaderToolsAgent Server")
    print("🌐 Server starting on http://localhost:8307")
    print("📋 Agent card: http://localhost:8307/.well-known/agent.json")
    print("🛠️ Features: Data loading, file processing, database integration")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8307, log_level="info")


if __name__ == "__main__":
    main() 