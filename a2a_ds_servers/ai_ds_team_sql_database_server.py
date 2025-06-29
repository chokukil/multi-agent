from a2a.utils import new_agent_text_message#!/usr/bin/env python3
"""
AI_DS_Team SQLDatabaseAgent A2A Server
Port: 8311

AI_DS_Team의 SQLDatabaseAgent를 A2A 프로토콜로 래핑하여 제공합니다.
SQL 데이터베이스 분석 및 쿼리 생성 전문
"""

import asyncio
import sys
import os
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
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.agents import SQLDatabaseAgent
import pandas as pd
import json
import sqlalchemy as sql

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


class SQLDatabaseAgentExecutor(AgentExecutor):
    """AI_DS_Team SQLDatabaseAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        
        # SQLite 인메모리 데이터베이스 연결 생성
        self.engine = sql.create_engine("sqlite:///:memory:")
        self.connection = self.engine.connect()
        
        self.agent = SQLDatabaseAgent(model=self.llm, connection=self.connection)
        logger.info("SQLDatabaseAgent initialized with in-memory SQLite database")
    
    async def execute(self, context: RequestContext, event_queue) -> None:
        """A2A 프로토콜에 따른 실행"""
        # event_queue passed as parameter
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"Processing SQL database request: {user_instructions}")
                
                # 데이터 로드 시도
                data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
                available_data = []
                
                try:
                    for file in os.listdir(data_path):
                        if file.endswith(('.csv', '.pkl')):
                            available_data.append(file)
                except:
                    pass
                
                if available_data:
                    # 가장 최근 데이터 사용하여 SQLite에 로드
                    # FALLBACK REMOVED - data_file = available_data[0]
                    if data_file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(data_path, data_file))
                    else:
                        df = pd.read_pickle(os.path.join(data_path, data_file))
                    
                    # 데이터프레임을 SQLite 테이블로 로드
                    table_name = data_file.split('.')[0]
                    df.to_sql(table_name, self.connection, if_exists='replace', index=False)
                    
                    logger.info(f"Loaded data: {data_file}, shape: {df.shape}, table: {table_name}")
                    
                    # SQLDatabaseAgent 실행
                    try:
                        result = self.agent.invoke_agent(
                            user_instructions=user_instructions
                        )
                        
                        # 결과 처리
                        # 결과 처리 (안전한 방식으로 workflow summary 가져오기)

                        try:

                            workflow_summary = self.agent.get_workflow_summary(markdown=True)

                        except AttributeError:

                            # get_workflow_summary 메서드가 없는 경우 기본 요약 생성

                            workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"

                        except Exception as e:

                            logger.warning(f"Error getting workflow summary: {e}")

                            workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                        
                        # SQL 쿼리 결과 수집
                        sql_info = ""
                        try:
                            sql_query = self.agent.get_sql_query_code()
                            sql_data = self.agent.get_data_sql()
                            
                            if sql_query:
                                sql_info += f"""
### 📝 생성된 SQL 쿼리
```sql
{sql_query}
```
"""
                            
                            if sql_data:
                                sql_info += f"""
### 📊 쿼리 결과
결과 행 수: {len(sql_data) if isinstance(sql_data, list) else 'N/A'}
"""
                        except:
                            sql_info = "\n### ℹ️ SQL 분석이 완료되었습니다."
                        
                        # 데이터 요약 생성
                        data_summary = get_dataframe_summary(df, n_sample=10)
                        
                        response_text = f"""## 🗄️ SQL 데이터베이스 분석 완료

{workflow_summary}

{sql_info}

### 📋 분석된 데이터 요약
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

### 🗄️ SQL Database Agent 기능
- **자동 SQL 생성**: 자연어를 SQL 쿼리로 변환
- **데이터베이스 분석**: 테이블 구조 및 관계 분석
- **복잡한 쿼리**: JOIN, 집계, 서브쿼리 등 고급 SQL
- **쿼리 최적화**: 효율적인 쿼리 작성 및 성능 개선
- **결과 분석**: 쿼리 결과 해석 및 인사이트 제공
"""
                        
                    except Exception as agent_error:
                        logger.warning(f"Agent execution failed, providing guidance: {agent_error}")
                        response_text = f"""## 🗄️ SQL 데이터베이스 분석 가이드

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 SQL Database Agent 사용법
다음과 같은 요청을 시도해보세요:

1. **기본 쿼리**:
   - "모든 데이터를 조회해주세요"
   - "상위 10개 행을 보여주세요"

2. **집계 분석**:
   - "카테고리별 평균값을 계산해주세요"
   - "월별 매출 합계를 구해주세요"

3. **복잡한 분석**:
   - "조건에 맞는 데이터를 필터링해주세요"
   - "두 테이블을 조인해서 분석해주세요"

테이블: {table_name}
요청: {user_instructions}
"""
                
                else:
                    response_text = f"""## ❌ 데이터 없음

SQL 데이터베이스 분석을 수행하려면 먼저 데이터를 업로드해야 합니다.
사용 가능한 데이터가 없습니다: {data_path}

요청: {user_instructions}

### 🗄️ SQL Database Agent 기능
- **SQL 쿼리 생성**: 자연어를 SQL로 변환
- **데이터베이스 분석**: 스키마 및 관계 분석
- **복잡한 조인**: 여러 테이블 간 관계 분석
- **집계 함수**: COUNT, SUM, AVG, MAX, MIN 등
- **조건부 쿼리**: WHERE, HAVING, ORDER BY 등
"""
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(response_text)
                )
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("SQL 분석 요청이 비어있습니다. 구체적인 데이터베이스 쿼리 요청을 해주세요.")
                )
                
        except Exception as e:
            logger.error(f"Error in SQLDatabaseAgent execution: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"SQL 데이터베이스 분석 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"SQLDatabaseAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="sql-database",
        name="SQL Database Analysis & Query Generation",
        description="전문적인 SQL 데이터베이스 분석 및 쿼리 생성 서비스. 자연어를 SQL로 변환하고 복잡한 데이터베이스 분석을 수행합니다.",
        tags=["sql", "database", "query", "analysis", "data"],
        examples=[
            "모든 데이터를 조회해주세요",
            "카테고리별 평균값을 계산해주세요",
            "상위 10개 행을 보여주세요",
            "조건에 맞는 데이터를 필터링해주세요",
            "월별 매출 합계를 구해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team SQLDatabaseAgent",
        description="전문적인 SQL 데이터베이스 분석 및 쿼리 생성 서비스. 자연어를 SQL로 변환하고 복잡한 데이터베이스 분석을 수행합니다.",
        url="http://localhost:8311/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=SQLDatabaseAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🗄️ Starting AI_DS_Team SQLDatabaseAgent Server")
    print("🌐 Server starting on http://localhost:8311")
    print("📋 Agent card: http://localhost:8311/.well-known/agent.json")
    print("🗄️ Features: SQL generation, database analysis, complex queries")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8311, log_level="info")


if __name__ == "__main__":
    main() 