#!/usr/bin/env python3
"""
실제로 작동하는 A2A 패턴 기반 Pandas Data Analyst 서버
mcp_dataloader_agent.py의 검증된 구조를 완전히 복사하여 구현
"""

import asyncio
import logging
import os
import sys
import uvicorn
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# A2A SDK imports (mcp_dataloader_agent.py와 동일)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, Message, Task, AgentCapabilities
from a2a.utils.message import new_agent_text_message, get_message_text

# CherryAI imports
from core.data_manager import DataManager

logger = logging.getLogger(__name__)

# Global instance
data_manager = DataManager()

# 1. Define the core agent (mcp_dataloader_agent.py 패턴 정확히 복사)
class PandasAnalysisAgent:
    """Pandas 데이터 분석 에이전트 (mcp_dataloader_agent.py 패턴)"""
    
    async def invoke(self, user_input: str = "") -> str:
        """데이터 분석 수행 (mcp_dataloader_agent.py의 invoke 패턴)"""
        try:
            # 고정된 스킬 실행 - 데이터 분석
            return self.analyze_data(user_input)
        except Exception as e:
            logger.error(f"Error in PandasAnalysisAgent.invoke: {e}")
            return f"❌ 데이터 분석 중 오류가 발생했습니다: {str(e)}"

    def analyze_data(self, user_request: str = "", **kwargs) -> str:
        """데이터 분석 스킬 (mcp_dataloader_agent.py 패턴)"""
        try:
            logger.info(f"🔍 데이터 분석 요청: {user_request}")
            
            # 1. 데이터 로드
            df, df_id = self._load_latest_dataset()
            if df is None:
                return "❌ 분석할 데이터가 없습니다. 먼저 데이터를 업로드해주세요."
            
            logger.info(f"✅ 데이터 로드 완료: {df.shape}")
            
            # 2. 기본 정보 생성
            basic_info = f"""# 📊 **데이터 분석 결과**

## 🔍 **데이터 개요**
- **데이터셋**: {df_id}
- **크기**: {df.shape[0]:,}행 × {df.shape[1]}열
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

## 📋 **컬럼 정보**
{self._get_column_info(df)}

## 📊 **기술통계**
{self._get_descriptive_stats(df)}

## 🔍 **데이터 품질**
{self._get_data_quality(df)}

## 🎯 **주요 인사이트**
{self._get_key_insights(df, user_request)}
"""
            
            return basic_info
            
        except Exception as e:
            logger.error(f"❌ 데이터 분석 오류: {e}")
            return f"❌ 데이터 분석 중 오류가 발생했습니다: {str(e)}"
    
    def _load_latest_dataset(self):
        """최신 데이터셋 로드"""
        try:
            datasets = data_manager.list_dataframes()
            if not datasets:
                return None, None
            
            # 가장 최근 데이터셋 ID 사용 (list_dataframes는 ID 목록을 반환)
            latest_dataset_id = datasets[0] 
            df = data_manager.get_dataframe(latest_dataset_id)
            
            logger.info(f"✅ 데이터셋 로드 성공: {latest_dataset_id}")
            return df, latest_dataset_id
            
        except Exception as e:
            logger.error(f"❌ 데이터셋 로드 오류: {e}")
            return None, None
    
    def _get_column_info(self, df):
        """컬럼 정보 생성"""
        try:
            info_lines = []
            for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
                non_null = df[col].count()
                null_count = df[col].isnull().sum()
                info_lines.append(f"  {i+1:2d}. **{col}**: {dtype} ({non_null:,} non-null, {null_count:,} null)")
            
            return "\n".join(info_lines)
        except:
            return "  (컬럼 정보 생성 오류)"
    
    def _get_descriptive_stats(self, df):
        """기술통계 생성"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "  - 수치형 컬럼이 없습니다."
            
            stats_lines = []
            for col in numeric_cols[:5]:  # 상위 5개만
                series = df[col]
                stats_lines.append(f"  **{col}**: 평균 {series.mean():.2f}, 표준편차 {series.std():.2f}, 최소값 {series.min():.2f}, 최대값 {series.max():.2f}")
            
            return "\n".join(stats_lines)
        except:
            return "  (기술통계 생성 오류)"
    
    def _get_data_quality(self, df):
        """데이터 품질 정보"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            null_percentage = (null_cells / total_cells) * 100
            
            return f"""  - **결측값**: {null_cells:,}개 ({null_percentage:.1f}%)
  - **중복행**: {df.duplicated().sum():,}개
  - **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)}개 숫자형, {len(df.select_dtypes(include=['object']).columns)}개 텍스트형"""
        except:
            return "  (데이터 품질 정보 생성 오류)"
    
    def _get_key_insights(self, df, user_request):
        """주요 인사이트 생성"""
        try:
            insights = []
            
            # 데이터 크기 기반 인사이트
            if df.shape[0] > 10000:
                insights.append("- 📊 **대용량 데이터셋**: 10,000행 이상의 데이터로 통계적 신뢰성이 높습니다.")
            
            # 결측값 기반 인사이트
            null_cols = df.isnull().sum()
            high_null_cols = null_cols[null_cols > len(df) * 0.3]
            if len(high_null_cols) > 0:
                insights.append(f"- ⚠️ **주의**: {len(high_null_cols)}개 컬럼에 30% 이상 결측값 존재")
            
            # 수치형 데이터 기반 인사이트
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                insights.append(f"- 🔢 **수치 분석 가능**: {len(numeric_cols)}개 수치형 컬럼 발견")
            
            if not insights:
                insights.append("- ✅ 데이터가 정상적으로 로드되었습니다.")
            
            return "\n".join(insights)
        except:
            return "- (인사이트 생성 오류)"

# 2. AgentExecutor 구현 (mcp_dataloader_agent.py 패턴 정확히 복사)
class PandasAnalysisAgentExecutor(AgentExecutor):
    """mcp_dataloader_agent.py 패턴을 사용하는 Pandas AgentExecutor"""
    
    def __init__(self):
        self.agent = PandasAnalysisAgent()
        logger.info("PandasAnalysisAgentExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (mcp_dataloader_agent.py 패턴 정확히 복사)"""
        logger.info("PandasAnalysisAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출 (mcp_dataloader_agent.py 패턴)
            user_message = context.get_user_input()
            logger.info(f"사용자 입력: {user_message}")
            
            # 에이전트 실행 (mcp_dataloader_agent.py 패턴)
            result = await self.agent.invoke(user_message)
            
            # 결과 전송 (공식 패턴 - 중요: await 추가!)
            message = new_agent_text_message(result)
            await event_queue.enqueue_event(message)
            
            logger.info("Task completed successfully")
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            error_message = new_agent_text_message(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task 취소 처리 (mcp_dataloader_agent.py 패턴)"""
        logger.info("PandasAnalysisAgentExecutor.cancel() 호출됨")
        raise Exception("Cancel not supported")

# 3. Agent Card 생성 (mcp_dataloader_agent.py 패턴)
def create_agent_card() -> AgentCard:
    """Agent Card 생성 (mcp_dataloader_agent.py 패턴)"""
    
    skill = AgentSkill(
        id="pandas_analysis",
        name="Pandas Data Analysis",
        description="Comprehensive data analysis using pandas with statistical insights",
        tags=["data", "analysis", "pandas", "statistics", "eda"],
        examples=["데이터를 분석해주세요", "EDA를 수행해주세요", "데이터 인사이트를 보여주세요"]
    )
    
    return AgentCard(
        name="Pandas Data Analyst (Working)",
        description="A working data analysis agent using proven patterns",
        url="http://localhost:10002/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),  # 실제로는 스트리밍 안함
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

# 4. Wire everything together (mcp_dataloader_agent.py 패턴 정확히 복사)
def main():
    """A2A 표준 Pandas Analysis 서버 실행"""
    logging.basicConfig(level=logging.INFO)
    logger.info("🚀 Starting Pandas Analysis A2A Server...")
    
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # RequestHandler 초기화
    request_handler = DefaultRequestHandler(
        agent_executor=PandasAnalysisAgentExecutor(),
        task_store=InMemoryTaskStore()
    )
    
    # A2A Starlette Application 생성
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info("🌐 Server starting at http://localhost:10002")
    logger.info("📋 Agent Card available at /.well-known/agent.json")
    logger.info("✅ Using proven mcp_dataloader_agent pattern")
    
    # Uvicorn으로 서버 실행
    uvicorn.run(a2a_app.build(), host="localhost", port=10002)

if __name__ == "__main__":
    main() 