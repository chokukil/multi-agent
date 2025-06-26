#!/usr/bin/env python3
"""
작동하는 A2A 패턴을 적용한 Pandas Data Analyst 서버
mcp_dataloader_agent.py와 동일한 구조로 구현
"""

import asyncio
import uvicorn
import logging
import pandas as pd
import os
import sys
from typing import Dict, Any
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A2A SDK imports (작동하는 패턴)
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, Message, Task
from a2a.utils.message import new_agent_text_message, get_message_text
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

# CherryAI imports
from core.data_manager import DataManager
from core.llm_factory import create_llm_instance

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pandas_server_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 객체 초기화
data_manager = DataManager()
llm = create_llm_instance()

# Skill Functions (단순한 함수 기반 접근)
async def analyze_data_skill(**kwargs) -> Message:
    """데이터 분석 스킬 함수"""
    logger.info("🎯 analyze_data_skill 함수 호출됨")
    
    try:
        # 파라미터에서 메시지 텍스트 추출
        prompt = kwargs.get('prompt', 'Analyze the available dataset')
        user_request = kwargs.get('user_request', prompt)
        
        logger.info(f"📝 분석 요청: {user_request}")
        
        # 사용 가능한 데이터프레임 확인
        available_dfs = data_manager.list_dataframes()
        logger.info(f"💾 사용 가능한 데이터: {available_dfs}")
        
        if not available_dfs:
            return new_agent_text_message("""❌ **데이터 없음**

**문제**: 아직 업로드된 데이터셋이 없습니다.

**해결방법:**
1. 🔄 **데이터 로더** 페이지로 이동
2. 📁 CSV, Excel 등의 데이터 파일 업로드  
3. 📊 다시 돌아와서 데이터 분석 요청

**현재 사용 가능한 데이터셋**: 없음
""")
        
        # 첫 번째 데이터프레임 사용
        df_id = available_dfs[0]
        df = data_manager.get_dataframe(df_id)
        
        if df is None:
            return new_agent_text_message(f"❌ 데이터셋 '{df_id}'를 로드할 수 없습니다.")
        
        logger.info(f"✅ 데이터셋 로드 완료: {df_id} ({df.shape})")
        
        # 기본 분석 수행
        analysis_result = await perform_comprehensive_analysis(df, df_id, user_request)
        
        return new_agent_text_message(analysis_result)
        
    except Exception as e:
        logger.error(f"💥 분석 실패: {e}", exc_info=True)
        return new_agent_text_message(f"❌ 분석 실패: {str(e)}")

async def perform_comprehensive_analysis(df: pd.DataFrame, df_id: str, prompt: str) -> str:
    """포괄적인 데이터 분석 수행"""
    import numpy as np
    
    logger.info(f"🔍 {df_id}에 대한 종합 분석 시작")
    
    # 기본 정보
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 데이터 품질
    missing_data = df.isnull().sum()
    completeness = ((total_rows * total_cols - missing_data.sum()) / (total_rows * total_cols)) * 100
    
    # 최종 보고서 구성 (LLM 없이 기본 분석)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    final_result = f"""# 📊 데이터 분석 보고서

**분석 대상**: {df_id}  
**분석 일시**: {timestamp}  
**요청**: {prompt}

## 📋 데이터 개요

- **크기**: {total_rows:,}행 × {total_cols}열
- **완성도**: {completeness:.1f}%
- **숫자형 변수**: {len(numeric_cols)}개
- **범주형 변수**: {len(categorical_cols)}개

## 🔍 기본 통계

{df.describe().round(2).to_markdown() if not df.select_dtypes(include=[np.number]).empty else "숫자형 데이터 없음"}

## 💡 주요 관찰

1. **데이터 규모**: {total_rows:,}개 관측값으로 {"충분한" if total_rows > 1000 else "제한적인"} 분석 가능
2. **데이터 품질**: {completeness:.1f}%로 {"우수" if completeness > 95 else "보통" if completeness > 80 else "개선 필요"}
3. **변수 구성**: 다양한 분석 관점 제공

---
**분석 엔진**: Pandas Data Analyst (Fixed)  
**상태**: ✅ 기본 분석 완료
"""
    
    logger.info("✅ 기본 분석 완료")
    return final_result

# Agent Executor 구현 (작동하는 패턴 정확히 적용)
class PandasSkillExecutor(AgentExecutor):
    def __init__(self, skill_handlers: Dict[str, Any]):
        self._skill_handlers = skill_handlers
        logger.info("🔧 PandasSkillExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (작동하는 패턴)"""
        logger.info("🎯 PandasSkillExecutor.execute() 호출됨")
        
        try:
            # skill_id 추출 (작동하는 패턴)
            skill_id = getattr(context, 'method', 'analyze_data')
            logger.info(f"🔧 실행할 스킬: {skill_id}")
            
            # 스킬 핸들러 확인
            handler = self._skill_handlers.get(skill_id)
            if not handler:
                logger.error(f"❌ 스킬 '{skill_id}' 찾을 수 없음")
                error_message = new_agent_text_message(f"스킬 '{skill_id}'를 찾을 수 없습니다.")
                await event_queue.enqueue_event(error_message)
                return

            # 파라미터 추출 (작동하는 패턴)
            params = getattr(context, 'params', {}) or {}
            
            # 메시지에서 추가 정보 추출
            if hasattr(context, 'request') and context.request:
                if hasattr(context.request, 'params') and hasattr(context.request.params, 'message'):
                    message = context.request.params.message
                    if message.parts:
                        user_text = ""
                        for part in message.parts:
                            if hasattr(part, 'text'):
                                user_text += part.text + " "
                        params['user_request'] = user_text.strip()
            
            logger.info(f"📝 파라미터: {params}")
            
            # 스킬 실행
            result = await handler(**params)
            
            # 결과 전송 (올바른 A2A API)
            await event_queue.enqueue_event(result)
            logger.info("✅ 스킬 실행 및 응답 전송 완료")
            
        except Exception as e:
            logger.error(f"💥 스킬 실행 실패: {e}", exc_info=True)
            error_message = new_agent_text_message(f"스킬 실행 오류: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """취소 처리 (필수 구현)"""
        logger.info("🛑 작업 취소 요청")
        pass

# Skill Handlers 매핑
skill_handlers: Dict[str, Any] = {
    "analyze_data": analyze_data_skill,
}

# 서버 설정
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 10001

# Agent Card 생성 (작동하는 패턴)
agent_card = AgentCard(
    name="Pandas Data Analyst (Fixed)",
    description="Expert data analyst using pandas for comprehensive dataset analysis - Fixed Version",
    version="1.0.1",
    url=f"http://{SERVER_HOST}:{SERVER_PORT}",
    capabilities={"streaming": True, "pushNotifications": False, "stateTransitionHistory": True},
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="analyze_data",
            name="Data Analysis",
            description="Analyze datasets using pandas and provide comprehensive insights",
            tags=["data", "analysis", "pandas", "statistics"],
            examples=["analyze the titanic dataset", "show me insights about sales data"]
        ),
    ],
    provider={
        "organization": "CherryAI",
        "description": "AI-powered data analysis platform",
        "url": f"http://{SERVER_HOST}:{SERVER_PORT}"
    }
)

# A2A 서버 구성 (JSON-RPC 프로토콜 사용)
agent_executor = PandasSkillExecutor(skill_handlers=skill_handlers)
task_store = InMemoryTaskStore()

# A2A SDK JSON-RPC 핸들러 사용
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
jsonrpc_handler = JSONRPCHandler(agent_executor=agent_executor, task_store=task_store)

a2a_app = A2AFastAPIApplication(
    agent_card=agent_card, 
    jsonrpc_handler=jsonrpc_handler
)
app = a2a_app.build()

if __name__ == "__main__":
    logger.info("🚀 Pandas Data Analyst A2A Server (Fixed) 시작...")
    logger.info(f"🌐 서버 주소: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info("📊 분석 준비 완료!")
    
    try:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        logger.exception(f"💥 서버 시작 실패: {e}")
        exit(1)
