import asyncio
import logging
import os
import pandas as pd
import re
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
import uvicorn

# A2A SDK 공식 컴포넌트 사용 (완전한 표준 구현)
import uuid
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard, AgentSkill, Message, Task, TaskState, TextPart, Role
)
from a2a.utils.message import new_agent_text_message

from langchain_ollama import ChatOllama

# Import core modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'core'))

from utils.logging import setup_logging
from data_manager import DataManager

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Initialize Global Components ---
try:
    llm = ChatOllama(model="gemma3:latest", temperature=0)
    data_manager = DataManager()
    logger.info("✅ Global components initialized successfully")
except Exception as e:
    logger.exception(f"💥 Critical error during initialization: {e}")
    exit(1)

class PandasAgentExecutor(AgentExecutor):
    """A2A SDK 표준을 완전히 준수하는 Pandas 분석 AgentExecutor"""
    
    def __init__(self, data_manager: DataManager, llm):
        self.data_manager = data_manager
        self.llm = llm
        logger.info("🔧 PandasAgentExecutor initialized")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 인터페이스 - 실시간 피드백 강화"""
        logger.info("🎯 A2A AGENT EXECUTE METHOD CALLED!")
        logger.info(f"📥 Request message_id: {getattr(context.message, 'messageId', 'unknown')}")
        logger.info(f"📥 Request user: {getattr(context.message, 'role', 'unknown')}")
        
        try:
            # 메시지에서 텍스트 추출
            message_text = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'text') and part.text:
                        message_text += part.text + " "
            
            message_text = message_text.strip()
            logger.info(f"📝 FULL ANALYSIS REQUEST: {message_text}")
            
            # 데이터 분석 수행 (A2A 표준 방식)
            logger.info("🔍 Starting comprehensive data analysis...")
            
            result = await self.analyze_data(message_text)
            logger.info(f"✅ Analysis completed successfully. Result length: {len(result)} chars")
            
            # A2A 표준 메시지 응답 생성 및 전송 (작동하는 패턴 적용)
            response_message = new_agent_text_message(result)
            await event_queue.put(response_message)
            
            logger.info("📤 Analysis result sent via EventQueue successfully")
            
        except Exception as e:
            logger.error(f"💥 A2A Agent execution failed: {e}", exc_info=True)
            
            # A2A 표준 오류 메시지 생성 및 전송
            error_message = new_agent_text_message(f"""❌ **Analysis Failed**

**Error Details:** {str(e)}

**Troubleshooting:**
1. Check if the dataset is properly loaded
2. Verify the analysis request format
3. Try again with a simpler request

Please contact support if the issue persists.
""")
            await event_queue.put(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 취소 인터페이스"""
        logger.info(f"🛑 Cancelling task")
        # 현재 구현에서는 취소 로직이 필요하지 않음
        pass



    async def analyze_data(self, prompt: str = "Analyze this dataset") -> str:
        """pandas 데이터 분석 실행"""
        logger.info(f"🎯 ANALYZE_DATA SKILL CALLED")
        logger.debug(f"📝 Prompt: {prompt}")
        
        try:
            # 데이터 ID 추출
            df_id = self._extract_data_id(prompt)
            available_dfs = self.data_manager.list_dataframes()
            
            logger.info(f"💾 Available dataframes: {available_dfs}")
            
            if not available_dfs:
                return """❌ **No Data Available**

**Issue:** No dataset has been uploaded yet.

**To use the Pandas Data Analyst:**
1. 🔄 Go to the **Data Loader** page first
2. 📁 Upload a CSV, Excel, or other data file  
3. 📊 Return here to analyze your uploaded data

**Available datasets:** None (please upload data first)
"""
            
            # 데이터 ID 자동 할당
            if not df_id:
                df_id = available_dfs[0]
                logger.info(f"🔧 Auto-assigned dataframe: '{df_id}'")
            
            # 데이터프레임 로드
            df = self.data_manager.get_dataframe(df_id)
            if df is None:
                return f"""❌ **Dataset Not Found: '{df_id}'**

**Available datasets:**
{chr(10).join(f"• `{df_id}`" for df_id in available_dfs)}

**Solution:** Use one of the available dataset IDs above, or upload new data via the Data Loader page.
"""
            
            # 데이터 분석 수행
            analysis_result = await self._perform_analysis(df, df_id, prompt)
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            return f"Analysis failed: {str(e)}"

    def _extract_data_id(self, prompt: str) -> str:
        """프롬프트에서 데이터 ID 추출"""
        if not prompt:
            return None
            
        # Pattern 1: Explicit "Data ID: something"
        data_id_match = re.search(r"Data ID:\s*([^\n\r\s]+)", prompt, re.IGNORECASE)
        if data_id_match:
            return data_id_match.group(1).strip().strip("'\"")
        
        # Pattern 2: "dataset with ID 'something'"
        id_pattern2 = re.search(r"dataset\s+with\s+ID\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
        if id_pattern2:
            return id_pattern2.group(1).strip()
        
        # Pattern 3: Common dataset names
        common_patterns = [
            r"titanic",
            r"customer_data", 
            r"sales_data",
            r"([a-zA-Z0-9_-]+\.(?:csv|xlsx|json|parquet))"
        ]
        for pattern in common_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(0).strip()
                
        return None

    async def _perform_analysis(self, df: pd.DataFrame, df_id: str, prompt: str) -> str:
        """실제 데이터 분석 수행 - 상세한 분석 리포트 생성"""
        import numpy as np
        from datetime import datetime
        
        logger.info(f"🔍 Starting comprehensive analysis for {df_id}")
        
        # 1. 기본 데이터 프로파일링
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 2. 데이터 품질 메트릭
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_data_summary = df.isnull().sum()
        completeness = ((total_rows * total_cols - missing_data_summary.sum()) / (total_rows * total_cols)) * 100
        
        # 3. 상세 통계 분석
        analysis_results = []
        
        # 숫자형 컬럼 분석
        if numeric_cols:
            numeric_summary = df[numeric_cols].describe()
            correlations = df[numeric_cols].corr() if len(numeric_cols) > 1 else None
        
        # 범주형 컬럼 분석  
        categorical_summary = {}
        for col in categorical_cols[:5]:  # 상위 5개 컬럼만
            value_counts = df[col].value_counts().head(10)
            categorical_summary[col] = {
                'unique_count': df[col].nunique(),
                'top_values': value_counts.to_dict()
            }
        
        # 4. 고급 분석 생성
        advanced_prompt = f"""
당신은 전문 데이터 분석가입니다. 다음 데이터셋에 대해 상세하고 통찰력 있는 분석 보고서를 작성해주세요:

**사용자 요청**: {prompt}

**데이터셋 정보**:
- 데이터셋명: {df_id}
- 전체 크기: {total_rows:,}행 × {total_cols}열
- 데이터 완성도: {completeness:.1f}%
- 숫자형 컬럼: {len(numeric_cols)}개 ({numeric_cols[:5]})
- 범주형 컬럼: {len(categorical_cols)}개 ({categorical_cols[:5]})

**숫자형 데이터 요약**:
{numeric_summary.to_string() if numeric_cols else "숫자형 데이터 없음"}

**범주형 데이터 요약**:
{str(categorical_summary) if categorical_summary else "범주형 데이터 없음"}

**분석 요구사항**:
1. 📊 **데이터 개요 및 구조 분석**
2. 🔍 **데이터 품질 평가** (결측값, 이상값, 데이터 타입 적절성)
3. 📈 **주요 통계적 특성** (분포, 중심경향, 변동성)
4. 🔗 **변수 간 관계 분석** (상관관계, 패턴)
5. 💡 **핵심 인사이트 및 비즈니스 함의**
6. 📋 **추가 분석 권장사항**

**출력 형식**: 마크다운으로 구조화된 상세 보고서
**톤**: 전문적이면서도 이해하기 쉽게
**목표**: 실무진이 의사결정에 활용할 수 있는 실용적 인사이트 제공
        """
        
        try:
            # LLM을 통한 전문 분석 생성
            logger.info("🧠 Generating AI-powered analysis...")
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(advanced_prompt)
            )
            
            # 분석 결과에 메타데이터 추가
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            final_result = f"""# 📊 데이터 분석 보고서

**분석 대상**: {df_id}  
**분석 일시**: {timestamp}  
**요청 내용**: {prompt}

---

{response.content}

---

## 📋 분석 메타데이터

| 항목 | 값 |
|-----|-----|
| 데이터셋 크기 | {total_rows:,} 행 × {total_cols} 열 |
| 데이터 완성도 | {completeness:.1f}% |
| 숫자형 변수 | {len(numeric_cols)}개 |
| 범주형 변수 | {len(categorical_cols)}개 |
| 결측값 총량 | {missing_data_summary.sum()} 개 |

**분석 엔진**: Pandas Data Analyst (A2A Protocol)  
**버전**: 1.0.0
"""
            
            logger.info("✅ Comprehensive analysis completed")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Advanced analysis failed, falling back to basic: {e}")
            
            # 기본 분석 결과로 대체 (더 상세하게)
            return f"""# 📊 데이터 분석 보고서

**분석 대상**: {df_id}  
**분석 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**요청 내용**: {prompt}

## 📋 데이터 개요

### 기본 정보
- **데이터셋 크기**: {total_rows:,} 행 × {total_cols} 열
- **데이터 완성도**: {completeness:.1f}%
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

### 변수 구성
| 변수 타입 | 개수 | 컬럼명 |
|----------|------|--------|
| 숫자형 | {len(numeric_cols)} | {', '.join(numeric_cols[:5])} |
| 범주형 | {len(categorical_cols)} | {', '.join(categorical_cols[:5])} |
| 날짜형 | {len(datetime_cols)} | {', '.join(datetime_cols[:5])} |

## 🔍 데이터 품질 분석

### 결측값 현황
{chr(10).join(f"- **{col}**: {count:,}개 ({count/total_rows*100:.1f}%)" for col, count in missing_data_summary.items() if count > 0) or "✅ 결측값이 없습니다."}

### 숫자형 변수 요약 통계
{numeric_summary.round(2).to_markdown() if not numeric_summary.empty else "숫자형 변수가 없습니다."}

## 💡 주요 관찰점

1. **데이터 크기**: {total_rows:,}개의 관측값으로 {"충분한" if total_rows > 1000 else "제한적인"} 분석 가능
2. **데이터 완성도**: {completeness:.1f}%로 {"우수한" if completeness > 95 else "보통" if completeness > 80 else "개선 필요한"} 수준
3. **변수 다양성**: {total_cols}개 변수로 {"다양한" if total_cols > 10 else "기본적인"} 분석 차원 제공

## 📈 추천 분석 방향

1. **탐색적 데이터 분석**: 변수별 분포 및 패턴 확인
2. **상관관계 분석**: 변수 간 연관성 탐색
3. **이상값 탐지**: 데이터 품질 개선
4. **시각화**: 주요 패턴의 시각적 표현

---
**분석 엔진**: Pandas Data Analyst (A2A Protocol)  
**상태**: 기본 분석 완료 ✅
"""

def create_agent_card() -> AgentCard:
    """A2A 표준 Agent Card 생성"""
    skill = AgentSkill(
        id="analyze_data",
        name="Data Analysis",
        description="Analyze datasets using pandas and provide comprehensive insights",
        tags=["data", "analysis", "pandas", "statistics"],
        examples=["analyze the titanic dataset", "show me insights about sales data"]
    )
    
    return AgentCard(
        name="Pandas Data Analyst",
        description="Expert data analyst using pandas for comprehensive dataset analysis",
        url="http://localhost:10001",
        version="1.0.0",
        capabilities={
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        authentication={"schemes": ["none"]},  # 인증 없음
        skills=[skill],
        provider={
            "organization": "CherryAI",
            "description": "AI-powered data analysis platform",
            "url": "http://localhost:10001"
        }
    )

def create_a2a_server() -> A2AFastAPIApplication:
    """A2A SDK를 사용한 완전한 표준 서버 생성"""
    
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # AgentExecutor 생성
    agent_executor = PandasAgentExecutor(data_manager, llm)
    
    # TaskStore 생성
    task_store = InMemoryTaskStore()
    
    # A2A 표준 RequestHandler 생성
    http_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store
    )
    
    # A2A FastAPI 애플리케이션 생성
    server = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=http_handler
    )
    
    logger.info("✅ A2A 서버가 표준 SDK로 생성되었습니다")
    return server

if __name__ == "__main__":
    logger.info("🚀 Starting Pandas Data Analyst A2A Server...")
    
    try:
        # A2A 표준 서버 생성
        server = create_a2a_server()
        app = server.build()
        
        # 서버 시작
        logger.info("🌐 Server starting on http://0.0.0.0:10001")
        uvicorn.run(app, host="0.0.0.0", port=10001)
        
    except Exception as e:
        logger.exception(f"💥 Server startup failed: {e}")
        exit(1) 