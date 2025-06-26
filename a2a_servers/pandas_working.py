#!/usr/bin/env python3
"""
작동하는 A2A 패턴 기반 Pandas Data Analyst 서버
mcp_dataloader_agent.py의 검증된 구조를 사용
"""

import pandas as pd
import os
import sys
import uvicorn
import logging
from typing import Dict, Any
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A2A SDK imports (검증된 패턴)
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, Message, Task
from a2a.utils.message import new_agent_text_message, get_message_text
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

# CherryAI imports
from core.data_manager import DataManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 데이터 매니저
data_manager = DataManager()

# 1. Skill Functions 정의
def analyze_data(prompt: str = "Analyze this dataset", data_id: str = None) -> Message:
    """데이터 분석 스킬 - mcp_dataloader_agent 패턴 적용"""
    logger.info(f"🎯 analyze_data 스킬 실행: {prompt}")
    
    try:
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
        
        # 첫 번째 데이터프레임 사용 (또는 지정된 data_id)
        df_id = data_id if data_id and data_id in available_dfs else available_dfs[0]
        df = data_manager.get_dataframe(df_id)
        
        if df is None:
            return new_agent_text_message(f"❌ 데이터셋 '{df_id}'를 로드할 수 없습니다.")
        
        logger.info(f"✅ 데이터셋 로드 완료: {df_id} ({df.shape})")
        
        # 기본 분석 수행
        analysis_result = perform_analysis(df, df_id, prompt)
        
        return new_agent_text_message(analysis_result)
        
    except Exception as e:
        logger.error(f"💥 분석 실패: {e}", exc_info=True)
        return new_agent_text_message(f"❌ 분석 실패: {str(e)}")

def perform_analysis(df: pd.DataFrame, df_id: str, prompt: str) -> str:
    """실제 데이터 분석 수행"""
    import numpy as np
    
    logger.info(f"🔍 {df_id}에 대한 분석 시작")
    
    # 기본 정보
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 데이터 품질
    missing_data = df.isnull().sum()
    completeness = ((total_rows * total_cols - missing_data.sum()) / (total_rows * total_cols)) * 100
    
    # 상세 분석 보고서
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 기본 통계 생성
    stats_table = ""
    if not df.select_dtypes(include=[np.number]).empty:
        stats_table = df.describe().round(2).to_markdown()
    else:
        stats_table = "숫자형 데이터가 없습니다."
    
    # 결측값 분석
    missing_info = ""
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        missing_info = "\n".join([f"- **{col}**: {count}개 ({count/total_rows*100:.1f}%)" 
                                  for col, count in missing_values.items() if count > 0])
    else:
        missing_info = "✅ 결측값이 없습니다."
    
    # 범주형 변수 분석
    categorical_info = ""
    for col in categorical_cols[:3]:  # 상위 3개만
        value_counts = df[col].value_counts().head(5)
        categorical_info += f"\n**{col}**:\n"
        for value, count in value_counts.items():
            categorical_info += f"- {value}: {count}개 ({count/total_rows*100:.1f}%)\n"
    
    # 최종 보고서
    final_result = f"""# 📊 데이터 분석 보고서

**분석 대상**: {df_id}  
**분석 일시**: {timestamp}  
**요청**: {prompt}

## 📋 데이터 개요

| 항목 | 값 |
|------|-----|
| 데이터 크기 | {total_rows:,} 행 × {total_cols} 열 |
| 완성도 | {completeness:.1f}% |
| 숫자형 변수 | {len(numeric_cols)}개 |
| 범주형 변수 | {len(categorical_cols)}개 |
| 메모리 사용량 | {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB |

## 🔍 기본 통계

{stats_table}

## ❌ 결측값 현황

{missing_info}

## 📊 범주형 변수 분포

{categorical_info}

## 💡 주요 인사이트

1. **데이터 규모**: {total_rows:,}개 관측값으로 {"충분한" if total_rows > 1000 else "제한적인"} 분석이 가능합니다.
2. **데이터 품질**: {completeness:.1f}%의 완성도로 {"우수한" if completeness > 95 else "보통" if completeness > 80 else "개선이 필요한"} 수준입니다.
3. **변수 구성**: {len(numeric_cols)}개의 숫자형 변수와 {len(categorical_cols)}개의 범주형 변수로 다양한 분석이 가능합니다.

## 📋 추천 분석 방향

1. **상관관계 분석**: 숫자형 변수 간의 관계 탐색
2. **분포 분석**: 각 변수의 분포 패턴 확인
3. **이상값 탐지**: 데이터 품질 개선 포인트 식별
4. **시각화**: 주요 패턴의 그래프 표현

---
**분석 엔진**: Pandas Data Analyst (Working)  
**상태**: ✅ 분석 완료
**처리 시간**: < 1초
"""
    
    logger.info("✅ 분석 완료")
    return final_result

# 2. AgentExecutor 구현 (검증된 패턴)
class SkillBasedAgentExecutor(AgentExecutor):
    def __init__(self, skill_handlers: Dict[str, Any]):
        self._skill_handlers = skill_handlers

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        skill_id = context.method
        handler = self._skill_handlers.get(skill_id)
        
        if not handler:
            error_message = new_agent_text_message(f"Skill '{skill_id}' not found.")
            await event_queue.put(error_message)
            return

        try:
            params = context.params or {}
            result = handler(**params)
            await event_queue.put(result)
        except Exception as e:
            error_message = new_agent_text_message(f"Error executing skill '{skill_id}': {e}")
            await event_queue.put(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Not implemented for this simple agent
        pass

# 3. 서버 구성 (검증된 패턴)
skill_handlers: Dict[str, Any] = {
    "analyze_data": analyze_data,
}

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 10001

agent_card = AgentCard(
    name="Pandas Data Analyst (Working)",
    description="Expert data analyst using pandas for comprehensive dataset analysis - Working Version",
    version="1.0.2",
    url=f"http://{SERVER_HOST}:{SERVER_PORT}",
    capabilities={"streaming": False},
    defaultInputModes=["application/json"],
    defaultOutputModes=["application/json"],
    skills=[
        AgentSkill(
            id="analyze_data",
            name="Data Analysis",
            description="Analyze datasets using pandas and provide comprehensive insights",
            tags=["data", "analysis", "pandas", "statistics"],
            examples=["analyze the titanic dataset", "show me insights about sales data"]
        ),
    ]
)

# A2A 서버 구성 (검증된 패턴)
agent_executor = SkillBasedAgentExecutor(skill_handlers=skill_handlers)
task_store = InMemoryTaskStore()
handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)
a2a_app = A2AFastAPIApplication(agent_card=agent_card, http_handler=handler)
app = a2a_app.build()

if __name__ == "__main__":
    logger.info("🚀 Pandas Data Analyst A2A Server (Working) 시작...")
    logger.info(f"🌐 서버 주소: http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info("📊 검증된 패턴으로 구현된 안정적인 분석 서버!")
    
    try:
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        logger.exception(f"💥 서버 시작 실패: {e}")
        exit(1) 