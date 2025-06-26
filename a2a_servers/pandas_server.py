import asyncio
import logging
import os
import pandas as pd
import re
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
import uvicorn
import click

# A2A SDK 공식 컴포넌트 사용 (공식 Hello World Agent 패턴)
import uuid
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, Message, Task, AgentCapabilities
from a2a.utils.message import new_agent_text_message, get_message_text

from langchain_ollama import ChatOllama

# Import core modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.data_manager import DataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data manager instance
data_manager = DataManager()

# 1. Define the core agent (공식 Hello World Agent 패턴)
class PandasDataAnalysisAgent:
    """Pandas 데이터 분석 에이전트 (공식 Hello World Agent 패턴)"""
    
    async def invoke(self, user_input: str = "") -> str:
        """
        데이터 분석 수행 (공식 Hello World Agent의 invoke 패턴)
        """
        logger.info(f"🎯 PandasDataAnalysisAgent.invoke() called with: {user_input}")
        
        try:
            # 사용 가능한 데이터프레임 확인
            available_dfs = data_manager.list_dataframes()
            logger.info(f"💾 Available dataframes: {available_dfs}")
            
            if not available_dfs:
                result_text = """❌ **데이터 없음**

**문제**: 아직 업로드된 데이터셋이 없습니다.

**해결방법:**
1. 🔄 **데이터 로더** 페이지로 이동
2. 📁 CSV, Excel 등의 데이터 파일 업로드  
3. 📊 다시 돌아와서 데이터 분석 요청

**현재 사용 가능한 데이터셋**: 없음
"""
                return result_text
            
            # 첫 번째 데이터프레임 사용
            df_id = available_dfs[0]
            df = data_manager.get_dataframe(df_id)
            
            if df is None:
                return "❌ 데이터프레임을 로드할 수 없습니다."
            
            logger.info(f"📊 Analyzing dataframe: {df_id}, shape: {df.shape}")
            
            # 데이터 분석 수행
            analysis_parts = []
            
            # 1. 기본 정보
            analysis_parts.append("# 📊 **데이터 분석 보고서**\n")
            analysis_parts.append(f"**데이터셋**: {df_id}")
            analysis_parts.append(f"**크기**: {df.shape[0]:,}행 × {df.shape[1]}열")
            analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 2. 데이터 개요
            analysis_parts.append("## 📋 **데이터 개요**")
            analysis_parts.append("**컬럼 정보:**")
            for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
                analysis_parts.append(f"{i}. **{col}** ({dtype})")
            analysis_parts.append("")
            
            # 3. 기본 통계
            analysis_parts.append("## 📈 **기본 통계**")
            desc = df.describe()
            if not desc.empty:
                analysis_parts.append("**수치형 변수 통계:**")
                for col in desc.columns[:3]:  # 처음 3개 컬럼만
                    analysis_parts.append(f"- **{col}**: 평균 {desc.loc['mean', col]:.2f}, 표준편차 {desc.loc['std', col]:.2f}")
            
            # 4. 결측치 분석
            missing = df.isnull().sum()
            if missing.sum() > 0:
                analysis_parts.append("\n## ⚠️ **결측치 분석**")
                for col, count in missing.items():
                    if count > 0:
                        pct = (count / len(df)) * 100
                        analysis_parts.append(f"- **{col}**: {count}개 ({pct:.1f}%)")
            else:
                analysis_parts.append("\n## ✅ **결측치**: 없음")
            
            # 5. 특별 분석 (Titanic 데이터셋인 경우)
            if 'Survived' in df.columns:
                analysis_parts.append("\n## 🚢 **타이타닉 생존 분석**")
                survival_rate = df['Survived'].mean() * 100
                analysis_parts.append(f"- **전체 생존율**: {survival_rate:.1f}%")
                
                if 'Sex' in df.columns:
                    survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
                    for sex, rate in survival_by_sex.items():
                        analysis_parts.append(f"- **{sex} 생존율**: {rate:.1f}%")
                
                if 'Pclass' in df.columns:
                    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
                    for pclass, rate in survival_by_class.items():
                        analysis_parts.append(f"- **{pclass}등석 생존율**: {rate:.1f}%")
            
            # 6. 추천사항
            analysis_parts.append("\n## 💡 **분석 추천사항**")
            analysis_parts.append("1. 🔍 **상관관계 분석**: 수치형 변수들 간의 관계 탐색")
            analysis_parts.append("2. 📊 **시각화**: 히스토그램, 상자그림 등으로 분포 확인")
            analysis_parts.append("3. 🎯 **세분화 분석**: 카테고리별 상세 분석 수행")
            
            result_text = "\n".join(analysis_parts)
            
            logger.info(f"✅ Analysis completed, length: {len(result_text)} characters")
            return result_text
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_data: {e}", exc_info=True)
            return f"❌ 분석 중 오류가 발생했습니다: {str(e)}"

# 2. AgentExecutor 구현 (공식 Hello World Agent 패턴)
class PandasAgentExecutor(AgentExecutor):
    """공식 Hello World Agent 패턴을 사용하는 AgentExecutor"""
    
    def __init__(self):
        self.agent = PandasDataAnalysisAgent()
        logger.info("🔧 PandasAgentExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (공식 Hello World Agent 패턴)"""
        logger.info("🎯 PandasAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출 (공식 패턴)
            user_message = context.get_user_input()
            logger.info(f"📝 사용자 입력: {user_message}")
            
            # 에이전트 실행 (공식 패턴)
            result = await self.agent.invoke(user_message)
            
            # 결과 전송 (공식 패턴 - 중요: await 추가!)
            message = new_agent_text_message(result)
            await event_queue.enqueue_event(message)
            
            logger.info("✅ Task completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in execute: {e}", exc_info=True)
            error_message = new_agent_text_message(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task 취소 처리 (공식 Hello World Agent 패턴)"""
        logger.info("🛑 PandasAgentExecutor.cancel() 호출됨")
        raise Exception("Cancel not supported")

# 3. Agent Card 생성 (공식 A2A 표준 메타데이터)
def create_agent_card() -> AgentCard:
    """A2A 표준 Agent Card 생성 (공식 Hello World Agent 패턴)"""
    
    # 기본 스킬 정의 (공식 패턴)
    skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas Data Analysis",
        description="Performs comprehensive data analysis on uploaded datasets using pandas",
        tags=["data", "analysis", "pandas", "statistics", "EDA"],
        examples=["Analyze my data", "What insights can you find?", "Show me data statistics"]
    )
    
    return AgentCard(
        name="Pandas Data Analyst",
        description="A comprehensive data analysis agent powered by pandas and AI",
        url="http://localhost:10001/",
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

# 4. Wire everything together (공식 Hello World Agent 패턴)
@click.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=10001, help='Port to bind to')
def main(host: str, port: int):
    """A2A 표준 Pandas 서버 실행 (공식 Hello World Agent 패턴)"""
    
    logger.info("🚀 Starting Pandas A2A Server...")
    
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # RequestHandler 초기화 (공식 패턴)
    request_handler = DefaultRequestHandler(
        agent_executor=PandasAgentExecutor(),
        task_store=InMemoryTaskStore()
    )
    
    # A2A Starlette Application 생성 (공식 패턴)
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info(f"🌐 Server starting at http://{host}:{port}")
    logger.info("📋 Agent Card available at /.well-known/agent.json")
    
    # Uvicorn으로 서버 실행
    uvicorn.run(a2a_app.build(), host=host, port=port)

if __name__ == "__main__":
    main() 