from a2a.utils import new_agent_text_message#!/usr/bin/env python3
"""
AI_DS_Team EDAToolsAgent A2A Server
Port: 8312

AI_DS_Team의 EDAToolsAgent를 A2A 프로토콜로 래핑하여 제공합니다.
탐색적 데이터 분석(EDA) 도구 전문
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any

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
from ai_data_science_team.ds_agents import EDAToolsAgent
import pandas as pd
import json

# CherryAI imports
from core.data_manager import DataManager  # DataManager 추가
from core.session_data_manager import SessionDataManager  # 세션 기반 데이터 관리자 추가

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

# 전역 DataManager 및 SessionDataManager 인스턴스
data_manager = DataManager()
session_data_manager = SessionDataManager()

class EDAToolsAgentExecutor(AgentExecutor):
    """EDA Tools Agent A2A Executor with DataManager integration"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = EDAToolsAgent(model=self.llm)
        logger.info("EDAToolsAgent initialized with LLM integration")
    
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
                        # 데이터 참조 정보 추출
                        if hasattr(part.root, 'data') and 'data_reference' in part.root.data:
                            data_reference = part.root.data['data_reference']
                            logger.info(f"Found data reference: {data_reference.get('data_id', 'unknown')}")
        
        return {
            "user_instructions": user_instructions.strip(),
            "data_reference": data_reference
        }

    async def execute(self, context: RequestContext, event_queue) -> None:
        """세션 기반 EDA 분석 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 탐색적 데이터 분석을 시작합니다...")
            )
            
            # A2A 메시지에서 정보 추출
            message_data = self.extract_data_reference_from_message(context)
            user_instructions = message_data["user_instructions"]
            data_reference = message_data["data_reference"]
            
            if user_instructions:
                logger.info(f"Processing EDA request: {user_instructions}")
                
                # SessionDataManager를 통한 AI DS Team 환경 준비
                current_session_id = session_data_manager.get_current_session_id()
                
                # 사용자 지시사항에서 파일명 추출하여 세션 생성
                if not current_session_id and data_reference:
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            # 새 세션 생성
                            current_session_id = session_data_manager.create_session_with_data(
                                data_id=data_id,
                                data=df,
                                user_instructions=user_instructions
                            )
                            logger.info(f"Created new session: {current_session_id}")
                
                # AI DS Team 환경 준비 (ai_ds_team/data/ 폴더에 올바른 데이터 배치)
                if current_session_id:
                    env_info = session_data_manager.prepare_ai_ds_team_environment(current_session_id)
                    logger.info(f"Prepared AI DS Team environment: {env_info['main_data_directory']}")
                
                # 데이터 로드 확인
                df = None
                data_description = ""
                
                if data_reference:
                    # A2A 메시지에서 전달된 데이터 참조 사용
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            data_description = f"✅ 요청된 데이터 로드됨: {data_id} (형태: {df.shape})"
                            logger.info(f"Loaded data from DataManager: {data_id}")
                        else:
                            logger.warning(f"Data not found in DataManager: {data_id}")
                
                if df is None:
                    # 데이터가 없으면 명확한 오류 메시지
                    available_data_ids = data_manager.list_dataframes()
                    if not available_data_ids:
                        response_text = f"""## ❌ 데이터 없음

탐색적 데이터 분석을 수행하려면 먼저 데이터를 업로드해야 합니다.

**요청**: {user_instructions}

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "ion_implant_3lot_dataset.xlsx 파일로 분석해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

**현재 상태**: 사용 가능한 데이터가 없습니다.
"""
                    else:
                        # 요청된 데이터를 찾을 수 없는 경우
                        requested_file = data_reference.get('data_id') if data_reference else '알 수 없음'
                        response_text = f"""## ❌ 요청된 데이터를 찾을 수 없음

요청하신 데이터 파일 '{requested_file}'을 찾을 수 없습니다.

**사용 가능한 데이터**: {', '.join(available_data_ids)}

**해결 방법**:
1. 사용 가능한 데이터 중 하나를 선택하여 요청하세요
2. 원하는 파일을 먼저 업로드해주세요

**요청**: {user_instructions}
"""
                
                if df is not None:
                    # 세션 컨텍스트와 함께 EDAToolsAgent 실행
                    try:
                        # 세션 컨텍스트 정보 추가
                        enhanced_instructions = user_instructions
                        if current_session_id:
                            session_context = session_data_manager.get_session_context(current_session_id)
                            if session_context:
                                enhanced_instructions = f"{user_instructions}\n\n[컨텍스트] 세션: {current_session_id}, 데이터: {session_context.get('data_id', 'unknown')}"
                        
                        result = self.agent.invoke_agent(
                            user_instructions=enhanced_instructions,
                            data_raw=df
                        )
                        
                        # 결과 처리
                        try:
                            workflow_summary = self.agent.get_workflow_summary(markdown=True)
                        except AttributeError:
                            workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                        except Exception as e:
                            logger.warning(f"Error getting workflow summary: {e}")
                            workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                        
                        # 생성된 EDA 보고서 정보 수집
                        eda_info = ""
                        artifacts_path = "a2a_ds_servers/artifacts/eda/"
                        os.makedirs(artifacts_path, exist_ok=True)
                        
                        # EDA 파일 저장 확인
                        saved_files = []
                        try:
                            if os.path.exists(artifacts_path):
                                for file in os.listdir(artifacts_path):
                                    if file.endswith(('.html', '.png', '.json')):
                                        saved_files.append(file)
                        except:
                            pass
                        
                        if saved_files:
                            eda_info += f"""
### 💾 생성된 EDA 보고서
{chr(10).join([f"- {file}" for file in saved_files[-5:]])}
"""
                        
                        # 데이터 요약 생성
                        data_summary = get_dataframe_summary(df, n_sample=10)
                        
                        response_text = f"""## 🔍 탐색적 데이터 분석(EDA) 완료

{data_description}

{workflow_summary}

{eda_info}

### 📋 분석된 데이터 요약
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

### 🧰 EDA Tools Agent 기능
- **데이터 프로파일링**: 자동 데이터 품질 분석
- **분포 분석**: 변수별 분포 및 통계 분석
- **상관관계 분석**: Correlation Funnel 및 히트맵
- **결측값 분석**: Missingno 시각화
- **자동 보고서**: Sweetviz, Pandas Profiling
- **통계적 검정**: 가설 검정 및 통계 분석
"""
                        
                    except Exception as agent_error:
                        logger.warning(f"Agent execution failed, providing guidance: {agent_error}")
                        response_text = f"""## 🔍 탐색적 데이터 분석(EDA) 가이드

{data_description}

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 EDA Tools 사용법
다음과 같은 요청을 시도해보세요:

1. **기본 EDA**:
   - "데이터의 기본 통계와 분포를 분석해주세요"
   - "결측값과 이상값을 확인해주세요"

2. **고급 EDA**:
   - "변수들 간의 상관관계를 분석해주세요"
   - "Sweetviz 보고서를 생성해주세요"

3. **특화 분석**:
   - "Correlation Funnel로 타겟 변수와의 관계를 보여주세요"
   - "Pandas Profiling 보고서를 만들어주세요"

요청: {user_instructions}
"""
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("EDA 요청이 비어있습니다. 구체적인 탐색적 데이터 분석 요청을 해주세요.")
                )
                
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(response_text)
            )
            
        except Exception as e:
            logger.error(f"Error in EDAToolsAgent execution: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"탐색적 데이터 분석 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"EDAToolsAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="eda-tools",
        name="Exploratory Data Analysis Tools",
        description="전문적인 탐색적 데이터 분석 도구. 데이터 프로파일링, 통계 분석, 시각화, 자동 보고서 생성을 제공합니다.",
        tags=["eda", "data-profiling", "statistics", "correlation", "visualization"],
        examples=[
            "데이터의 기본 통계와 분포를 분석해주세요",
            "결측값과 이상값을 확인해주세요",
            "변수들 간의 상관관계를 분석해주세요",
            "Sweetviz 보고서를 생성해주세요",
            "Correlation Funnel로 타겟 변수와의 관계를 보여주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team EDAToolsAgent",
        description="전문적인 탐색적 데이터 분석 도구. 데이터 프로파일링, 통계 분석, 시각화, 자동 보고서 생성을 제공합니다.",
        url="http://localhost:8312/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=EDAToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting AI_DS_Team EDAToolsAgent Server")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")
    print("🧰 Features: Data profiling, statistics, correlation, auto reports")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main() 