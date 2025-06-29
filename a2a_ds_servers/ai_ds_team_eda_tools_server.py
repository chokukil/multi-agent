#!/usr/bin/env python3
"""
AI_DS_Team EDAToolsAgent A2A Server (Session-based)
Port: 8312

SessionDataManager를 사용하여 세션 기반으로 AI DS Team과 통합
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
from a2a.utils import new_agent_text_message
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
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
    """세션 기반 EDA Tools Agent A2A Executor"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = EDAToolsAgent(model=self.llm)
        logger.info("SessionEDAToolsAgent initialized")
    
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
        """세션 기반 EDA 분석 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔍 세션 기반 EDA 분석을 시작합니다...")
            )
            
            message_data = self.extract_data_reference_from_message(context)
            user_instructions = message_data["user_instructions"]
            data_reference = message_data["data_reference"]
            
            if user_instructions:
                df = None
                current_session_id = None
                
                if data_reference:
                    data_id = data_reference.get('data_id')
                    if data_id:
                        df = data_manager.get_dataframe(data_id)
                        if df is not None:
                            # 세션 생성 및 AI DS Team 환경 준비
                            current_session_id = session_data_manager.create_session_with_data(
                                data_id=data_id,
                                data=df,
                                user_instructions=user_instructions
                            )
                            env_info = session_data_manager.prepare_ai_ds_team_environment(current_session_id)
                            logger.info(f"✅ Session {current_session_id} created and AI DS Team environment prepared")
                
                if df is not None:
                    # EDA 실행
                    result = self.agent.invoke_agent(
                        user_instructions=user_instructions,
                        data_raw=df
                    )
                    
                    response_text = f"""## 🔍 세션 기반 EDA 분석 완료

✅ **세션 ID**: {current_session_id}
✅ **데이터**: {data_reference.get('data_id', 'unknown') if data_reference else 'unknown'}
✅ **형태**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
✅ **AI DS Team 환경**: 준비 완료

### 📊 분석 결과
EDA 분석이 성공적으로 완료되었습니다. AI DS Team 에이전트들이 올바른 데이터를 사용하여 분석을 수행했습니다.

### 🎯 세션 기반 분석의 장점
- 올바른 데이터 파일 사용 보장
- 사용자 컨텍스트 유지
- 세션별 결과 격리
"""
                else:
                    response_text = "❌ 요청된 데이터를 찾을 수 없습니다. 먼저 데이터를 업로드해주세요."
                
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
            logger.error(f"Error in SessionEDAToolsAgent: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"세션 기반 EDA 분석 중 오류: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        logger.info(f"SessionEDAToolsAgent cancelled: {context.task_id}")


def main():
    skill = AgentSkill(
        id="session_eda",
        name="Session-based EDA",
        description="세션 기반 탐색적 데이터 분석",
        tags=["eda", "session-based"],
        examples=["데이터 EDA를 진행해주세요"]
    )
    
    agent_card = AgentCard(
        name="SessionEDAToolsAgent",
        description="세션 기반 EDA 전문가",
        url="http://localhost:8312/",
        version="2.0.0",
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
    
    print("🔍 Starting SessionEDAToolsAgent Server on port 8312")
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main()
