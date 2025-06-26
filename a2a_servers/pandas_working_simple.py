#!/usr/bin/env python3
"""
작동하는 A2A 패턴으로 구현한 Pandas 분석 서버
mcp_dataloader_agent.py 패턴을 pandas 분석용으로 수정
"""

import asyncio
import logging
import os
import sys
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# A2A SDK imports
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

# 1. Core Agent Implementation
class PandasAnalysisAgent:
    """Pandas 데이터 분석 에이전트 (작동하는 패턴)"""
    
    async def invoke(self, user_input: str = "") -> str:
        """데이터 분석 수행"""
        try:
            return self.analyze_data(user_input)
        except Exception as e:
            logger.error(f"Error in invoke: {e}")
            return f"❌ 분석 중 오류가 발생했습니다: {str(e)}"

    def analyze_data(self, user_request: str = "", **kwargs) -> str:
        """실제 데이터 분석 로직"""
        try:
            logger.info(f"🔍 Pandas 데이터 분석 시작: {user_request}")
            
            # 1. 데이터 로드
            datasets = data_manager.list_datasets()
            if not datasets:
                return """❌ **분석할 데이터가 없습니다**

📋 **데이터 업로드 방법:**
1. CherryAI UI 사이드바에서 **"📁 데이터 업로드"** 클릭
2. CSV 파일을 선택하여 업로드
3. 업로드 완료 후 다시 분석 요청

💡 **추천**: 먼저 `titanic.csv`, `sales_data.csv` 등의 샘플 데이터를 업로드해보세요!"""

            # 최신 데이터셋 사용
            latest_dataset = datasets[0]
            df_id = latest_dataset["id"]
            df = data_manager.get_dataset(df_id)
            
            if df is None:
                return "❌ 데이터셋을 로드할 수 없습니다."
            
            logger.info(f"✅ 데이터 로드 성공: {df.shape}")
            
            # 2. 종합 분석 수행
            return self._generate_comprehensive_analysis(df, df_id, user_request)
            
        except Exception as e:
            logger.error(f"❌ analyze_data 오류: {e}")
            return f"❌ 데이터 분석 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_comprehensive_analysis(self, df: pd.DataFrame, df_id: str, user_request: str) -> str:
        """종합적인 데이터 분석 생성"""
        try:
            analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 기본 정보
            basic_info = self._get_basic_info(df, df_id)
            
            # 수치형 분석
            numeric_analysis = self._get_numeric_analysis(df)
            
            # 범주형 분석
            categorical_analysis = self._get_categorical_analysis(df)
            
            # 데이터 품질
            quality_analysis = self._get_quality_analysis(df)
            
            # 인사이트
            insights = self._get_insights(df, user_request)
            
            # 추천사항
            recommendations = self._get_recommendations(df)
            
            # 종합 리포트 생성
            comprehensive_report = f"""# 📊 **종합 데이터 분석 결과**

**분석 요청**: {user_request}  
**분석 시간**: {analysis_time}  
**처리 시간**: ~2초

---

## 🔍 **데이터 개요**
{basic_info}

## 📊 **수치형 데이터 분석**
{numeric_analysis}

## 📋 **범주형 데이터 분석**
{categorical_analysis}

## 🔍 **데이터 품질 분석**
{quality_analysis}

## 💡 **주요 인사이트**
{insights}

## 🎯 **분석 추천사항**
{recommendations}

---

## ⚙️ **기술 정보**
- **분석 엔진**: Pandas + NumPy
- **데이터 처리**: 메모리 효율적 처리
- **분석 깊이**: 종합 EDA (탐색적 데이터 분석)

✅ **분석 완료!** 추가 질문이나 심화 분석이 필요하시면 언제든 요청해주세요.
"""
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ 종합 분석 생성 오류: {e}")
            return f"❌ 분석 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _get_basic_info(self, df: pd.DataFrame, df_id: str) -> str:
        """기본 정보 생성"""
        return f"""- **데이터셋**: {df_id}
- **크기**: {df.shape[0]:,}행 × {df.shape[1]}열
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
- **컬럼**: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}"""
    
    def _get_numeric_analysis(self, df: pd.DataFrame) -> str:
        """수치형 데이터 분석"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "- 수치형 컬럼이 없습니다."
        
        lines = [f"- **수치형 컬럼 개수**: {len(numeric_cols)}개"]
        for col in numeric_cols[:3]:
            series = df[col]
            lines.append(f"- **{col}**: 평균 {series.mean():.2f}, 표준편차 {series.std():.2f}")
        
        if len(numeric_cols) > 3:
            lines.append(f"- *(+{len(numeric_cols)-3}개 컬럼 추가 분석 가능)*")
        
        return '\n'.join(lines)
    
    def _get_categorical_analysis(self, df: pd.DataFrame) -> str:
        """범주형 데이터 분석"""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            return "- 범주형 컬럼이 없습니다."
        
        lines = [f"- **범주형 컬럼 개수**: {len(cat_cols)}개"]
        for col in cat_cols[:3]:
            unique_count = df[col].nunique()
            lines.append(f"- **{col}**: {unique_count}개 고유값")
        
        return '\n'.join(lines)
    
    def _get_quality_analysis(self, df: pd.DataFrame) -> str:
        """데이터 품질 분석"""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells) * 100
        duplicate_rows = df.duplicated().sum()
        
        return f"""- **결측값**: {null_cells:,}개 ({null_percentage:.1f}%)
- **중복행**: {duplicate_rows:,}개
- **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)}개 숫자형, {len(df.select_dtypes(include=['object']).columns)}개 텍스트형
- **전체 완전성**: {100-null_percentage:.1f}%"""
    
    def _get_insights(self, df: pd.DataFrame, user_request: str) -> str:
        """주요 인사이트 생성"""
        insights = []
        
        # 데이터 크기 인사이트
        if df.shape[0] > 1000:
            insights.append("📊 **충분한 데이터**: 1,000행 이상으로 통계적 분석에 적합합니다.")
        
        # 결측값 인사이트
        null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if null_percentage > 10:
            insights.append(f"⚠️ **주의**: 결측값이 {null_percentage:.1f}%로 높습니다. 전처리가 필요할 수 있습니다.")
        else:
            insights.append("✅ **양질의 데이터**: 결측값이 적어 분석에 적합합니다.")
        
        # 컬럼 다양성
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        if numeric_ratio > 0.7:
            insights.append("🔢 **수치 중심**: 수치형 데이터가 많아 통계 분석에 유리합니다.")
        elif numeric_ratio < 0.3:
            insights.append("📝 **범주 중심**: 범주형 데이터가 많아 분류 분석에 적합합니다.")
        else:
            insights.append("⚖️ **균형**: 수치형과 범주형 데이터가 균형잡혀 있습니다.")
        
        return '\n'.join(insights) if insights else "- 추가 분석을 통해 더 많은 인사이트를 발견할 수 있습니다."
    
    def _get_recommendations(self, df: pd.DataFrame) -> str:
        """분석 추천사항"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            recommendations.append("📈 **상관관계 분석**: 수치형 변수들 간의 상관관계를 분석해보세요.")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            recommendations.append("📊 **범주별 분석**: 범주형 변수별 분포와 패턴을 살펴보세요.")
        
        if df.shape[0] > 5000:
            recommendations.append("🤖 **머신러닝**: 데이터가 충분하니 예측 모델링을 시도해보세요.")
        
        recommendations.append("📋 **시각화**: 차트와 그래프로 데이터를 시각화해보세요.")
        
        return '\n'.join(recommendations)

# 2. AgentExecutor Implementation
class PandasAnalysisAgentExecutor(AgentExecutor):
    """작동하는 패턴의 Pandas AgentExecutor"""
    
    def __init__(self):
        self.agent = PandasAnalysisAgent()
        logger.info("PandasAnalysisAgentExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (작동하는 패턴)"""
        logger.info("PandasAnalysisAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출
            user_message = context.get_user_input()
            logger.info(f"사용자 입력: {user_message}")
            
            # 에이전트 실행
            result = await self.agent.invoke(user_message)
            
            # 결과 전송
            message = new_agent_text_message(result)
            await event_queue.enqueue_event(message)
            
            logger.info("✅ Pandas 분석 완료")
            
        except Exception as e:
            logger.error(f"❌ 실행 오류: {e}", exc_info=True)
            error_message = new_agent_text_message(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """취소 처리"""
        logger.info("취소 요청")
        raise Exception("Cancel not supported")

# 3. Agent Card Creation
def create_agent_card() -> AgentCard:
    """Agent Card 생성"""
    skill = AgentSkill(
        id="pandas_analysis",
        name="Pandas Data Analysis",
        description="Comprehensive data analysis using pandas with statistical insights and EDA",
        tags=["data", "analysis", "pandas", "statistics", "eda", "insights"],
        examples=["데이터를 분석해주세요", "EDA를 수행해주세요", "데이터 인사이트를 보여주세요", "종합 분석해주세요"]
    )
    
    return AgentCard(
        name="Pandas Data Analyst",
        description="Comprehensive data analysis agent using pandas and statistical methods",
        url="http://localhost:10001/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

# 4. Server Setup
def main():
    """A2A Pandas 분석 서버 실행"""
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
    
    logger.info("🌐 Server starting at http://localhost:10001")
    logger.info("📋 Agent Card available at /.well-known/agent.json")
    logger.info("✅ Using proven working pattern")
    
    # Uvicorn으로 서버 실행
    uvicorn.run(a2a_app.build(), host="localhost", port=10001)

if __name__ == "__main__":
    main() 