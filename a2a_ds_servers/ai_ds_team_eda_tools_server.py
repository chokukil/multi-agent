#!/usr/bin/env python3
"""
AI_DS_Team EDAToolsAgent A2A Server (Enhanced)
Port: 8312
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
from a2a.utils import new_agent_text_message
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent
import pandas as pd
import json

# CherryAI imports
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()
session_data_manager = SessionDataManager()

class SessionEDAToolsAgentExecutor(AgentExecutor):
    """Enhanced EDA Tools Agent A2A Executor"""
    
    def __init__(self):
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = EDAToolsAgent(model=self.llm)
        logger.info("Enhanced SessionEDAToolsAgent initialized")
    
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

    async def execute(self, context: RequestContext, event_queue) -> None:
        """Enhanced EDA 분석 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 Enhanced EDA 분석을 시작합니다...")
            )
            
            message_data = self.extract_data_reference_from_message(context)
            user_instructions = message_data["user_instructions"]
            data_reference = message_data["data_reference"]
            
            if user_instructions:
                df = None
                current_session_id = None
                data_source = "unknown"
                
                # 데이터 로드 시도
                if data_reference:
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            data_source = data_id
                            logger.info(f"📊 데이터 로드 성공: {data_id}")
                
                # 기본 데이터 찾기
                if df is None:
                    available_data = data_manager.list_dataframes()
                    logger.info(f"🔍 사용 가능한 데이터: {available_data}")
                    
                    if available_data:
                        first_data_id = available_data[0]
                        df = data_manager.get_dataframe(first_data_id)
                        if df is not None:
                            data_source = first_data_id
                            logger.info(f"📊 기본 데이터 사용: {first_data_id}")
                
                if df is not None:
                    # 세션 생성
                    current_session_id = session_data_manager.create_session_with_data(
                        data_id=data_source,
                        data=df,
                        user_instructions=user_instructions
                    )
                    env_info = session_data_manager.prepare_ai_ds_team_environment(current_session_id)
                    logger.info(f"✅ Session {current_session_id} created")
                    
                    # AI DS Team EDA 실행
                    logger.info("🚀 AI DS Team EDA 에이전트 실행 중...")
                    
                    try:
                        result = self.agent.invoke_agent(
                            user_instructions=user_instructions,
                            data_raw=df
                        )
                        
                        # 결과 처리
                        if isinstance(result, dict):
                            result_text = json.dumps(result, ensure_ascii=False, indent=2)
                        else:
                            result_text = str(result)
                        
                        response_text = f"""## 🔍 Enhanced EDA 분석 완료

✅ **세션 ID**: {current_session_id}
✅ **데이터 소스**: {data_source}
✅ **데이터 형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열

### 📊 분석 결과

{result_text[:1500]}{'...' if len(result_text) > 1500 else ''}

### 🎯 분석 완료
AI DS Team EDA 에이전트가 성공적으로 데이터 분석을 완료했습니다.
"""
                        
                        await task_updater.update_status(
                            TaskState.completed,
                            message=new_agent_text_message(response_text)
                        )
                        
                    except Exception as eda_error:
                        logger.error(f"❌ EDA 실행 오류: {eda_error}")
                        
                        # 기본 분석 제공
                        basic_analysis = f"""## ⚠️ 기본 데이터 분석

### 📊 데이터 정보
- **소스**: {data_source}
- **형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}

### 🔍 기본 통계
{df.describe().to_string()[:500]}

### ⚠️ 참고
AI DS Team 에이전트 실행 중 오류가 발생했습니다: {str(eda_error)}
"""
                        
                        await task_updater.update_status(
                            TaskState.completed,
                            message=new_agent_text_message(basic_analysis)
                        )
                else:
                    response_text = """❌ **데이터를 찾을 수 없습니다**

데이터를 먼저 업로드해주세요."""
                    
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message(response_text)
                    )
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("EDA 분석 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"Error in Enhanced EDA Agent: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"EDA 분석 중 오류: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        logger.info(f"Enhanced EDA Agent cancelled: {context.task_id}")


def main():
    skill = AgentSkill(
        id="enhanced_eda",
        name="Enhanced EDA",
        description="Enhanced 탐색적 데이터 분석",
        tags=["eda", "enhanced"],
        examples=["데이터 EDA를 진행해주세요"]
    )
    
    agent_card = AgentCard(
        name="AI_DS_Team EDAToolsAgent",
        description="Enhanced EDA 전문가",
        url="http://localhost:8312/",
        version="3.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=SessionEDAToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting Enhanced EDA Agent Server on port 8312")
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main()
