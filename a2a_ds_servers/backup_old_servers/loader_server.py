import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import common utilities
from a2a_ds_servers.common.import_utils import setup_project_paths, log_import_status

# Setup paths and log status
setup_project_paths()
log_import_status()

#!/usr/bin/env python3
"""

Data Loader Server
Port: 8322

DataLoaderToolsAgent를 A2A 프로토콜로 래핑하여 제공합니다.
다양한 데이터 소스(CSV, Excel, JSON 등)에서 데이터를 로드하고 전처리하는 전문 에이전트입니다.
"""

import asyncio
import sys
import os
import json
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
from a2a.utils import new_agent_text_message
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.agents import DataLoaderToolsAgent
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoaderAgent:
    """DataLoaderToolsAgent를 사용한 래퍼 클래스"""
    
    def __init__(self):
        # 🔥 원래 기능 1: Data Manager 초기화 (필수)
        try:
            from core.data_manager import DataManager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            raise RuntimeError("Data Manager is required for operation") from e
        
        # 🔥 원래 기능 2: Real LLM 초기화 (필수, 폴백 없음)
        self.llm = None
        self.agent = None
        
        try:
            # 공통 LLM 초기화 유틸리티 사용
            from base.llm_init_utils import create_llm_with_fallback
            
            self.llm = create_llm_with_fallback()
            
            # 🔥 원래 기능 보존: ai_data_science_team 에이전트들 사용
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_ds_team'))
            from agents import DataLoaderToolsAgent as OriginalAgent
            
            # 🔥 원래 기능 3: DataLoaderToolsAgent 초기화 (정확한 패턴 보존)
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Data Loader Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e
    
    async def process_data_loading(self, user_input: str) -> str:
        """데이터 로딩 처리 실행"""
        try:
            logger.info(f"Processing data loading request: {user_input}")
            
            # DataLoaderToolsAgent 실행
            result = self.agent.invoke_agent(user_instructions=user_input)
            
            # 워크플로우 요약 생성 (메서드가 없는 경우 대체)
            try:
                workflow_summary = self.agent.get_workflow_summary(markdown=True)
            except AttributeError:
                workflow_summary = "워크플로우가 성공적으로 실행되었습니다."
            
            # 로드된 데이터 정보 확인
            loaded_data_info = ""
            artifacts_info = ""
            
            # 에이전트가 데이터를 로드했는지 확인
            if hasattr(self.agent, 'data') and self.agent.data is not None:
                df = self.agent.data
                
                # 데이터를 공유 폴더에 저장
                data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
                os.makedirs(data_path, exist_ok=True)
                
                import time
                timestamp = int(time.time())
                output_file = f"loaded_data_{timestamp}.csv"
                output_path = os.path.join(data_path, output_file)
                
                df.to_csv(output_path, index=False)
                logger.info(f"Data saved to: {output_path}")
                
                loaded_data_info = f"""

### 📊 로드된 데이터 정보
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

### 📋 데이터 미리보기
```
{df.head().to_string()}
```

### 🔍 컬럼 정보
```
{df.dtypes.to_string()}
```
"""
            
            # 생성된 아티팩트 확인 (메서드가 없는 경우 대체)
            artifacts_info = ""
            try:
                artifacts = self.agent.get_artifacts(as_dataframe=False)
                if artifacts:
                    artifacts_info = f"""

### 📁 생성된 아티팩트
```json
{json.dumps(artifacts, indent=2, ensure_ascii=False)}
```
"""
            except AttributeError:
                artifacts_info = ""
            
            # 최종 응답 포맷팅
            final_response = f"""**Data Loading Complete!**

### 📥 데이터 로딩 요청
{user_input}

### 🔄 처리 과정
{workflow_summary}
{loaded_data_info}
{artifacts_info}

✅ 데이터 로딩이 성공적으로 완료되었습니다."""
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in data loading processing: {e}")
            return f"❌ 데이터 로딩 중 오류 발생: {str(e)}"


class LoaderExecutor(AgentExecutor):
    """Data Loader AgentExecutor Implementation"""

    def __init__(self):
        self.agent = LoaderAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue,
    ) -> None:
        # TaskUpdater 초기화 (A2A SDK v0.2.6+ 필수 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Task 제출 및 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_message += part.root.text + " "
                
                user_message = user_message.strip()
            
            if not user_message:
                await task_updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message("❌ No valid message provided")
                )
                return
            
            # 진행 상황 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📁 데이터 로딩을 시작합니다...")
            )
            
            # 로딩 처리 실행
            result = await self.agent.process_data_loading(user_message)
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            # Task 실패 처리
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ Error: {str(e)}")
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue,
    ) -> None:
        """작업 취소 처리"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.update_status(
            TaskState.cancelled,
            message=new_agent_text_message("작업이 취소되었습니다.")
        )


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_loading",
        name="Data Loading & File Processing",
        description="다양한 데이터 소스(CSV, Excel, JSON, parquet)에서 데이터를 로드하고 전처리하는 전문가",
        tags=["data-loading", "etl", "file-processing", "csv", "excel", "json"],
        examples=[
            "CSV 파일을 로드해주세요",
            "Excel 파일의 특정 시트를 읽어주세요", 
            "JSON 데이터를 DataFrame으로 변환해주세요",
            "사용 가능한 데이터 파일들을 보여주세요",
            "파일 형식을 자동으로 감지하여 로드해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Data Loader Agent",
        description="다양한 데이터 소스에서 데이터를 로드하고 전처리하는 전문 에이전트",
        url="http://localhost:8322/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=LoaderExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📁 Starting Data Loader Agent Server")
    print("🌐 Server starting on http://localhost:8322")
    print("📋 Agent card: http://localhost:8322/.well-known/agent.json")
    print("🛠️ Features: CSV, Excel, JSON, parquet loading and preprocessing")
    
    # 서버 실행
    uvicorn.run(server.build(), host="0.0.0.0", port=8322, log_level="info")


if __name__ == "__main__":
    main()