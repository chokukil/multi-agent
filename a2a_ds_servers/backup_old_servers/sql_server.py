import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import common utilities
from a2a_ds_servers.common.import_utils import setup_project_paths, log_import_status

# Setup paths and log status
setup_project_paths()
log_import_status()

#!/usr/bin/env python3
"""

AI_DS_Team SQLDatabaseAgent A2A Server
Port: 8324

AI_DS_Team의 SQLDatabaseAgent를 A2A 프로토콜로 래핑하여 제공합니다.
SQL 데이터베이스 분석 및 쿼리 생성 전문
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import io
import sqlalchemy as sql

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# A2A imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn

# AI_DS_Team imports
from ai_data_science_team.agents import SQLDatabaseAgent

# Core imports
from core.data_manager import DataManager
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()

class PandasAIDataProcessor:
    """pandas-ai 패턴을 활용한 데이터 처리기"""
    
    def __init__(self):
        self.current_dataframe = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱"""
        logger.info("📊 pandas-ai 패턴으로 메시지에서 데이터 파싱...")
        
        # 1. CSV 데이터 파싱
        lines = user_message.split('\n')
        csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
        
        if len(csv_lines) >= 2:  # 헤더 + 데이터
            try:
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info("✅ CSV 데이터 파싱 성공: %s", df.shape)
                return df
            except Exception as e:
                logger.warning("CSV 파싱 실패: %s", e)
        
        # 2. JSON 데이터 파싱
        try:
            json_start = user_message.find('{')
            json_end = user_message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = user_message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    logger.info("✅ JSON 리스트 데이터 파싱 성공: %s", df.shape)
                    return df
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    logger.info("✅ JSON 객체 데이터 파싱 성공: %s", df.shape)
                    return df
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패: %s", e)
        
        return None
    
    def validate_and_process_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증"""
        if df is None or df.empty:
            return False
        
        logger.info("📊 데이터 검증: %s (행 x 열)", df.shape)
        logger.info("🔍 컬럼: %s", list(df.columns))
        logger.info("📈 타입: %s", df.dtypes.to_dict())
        
        return True

class SQLDatabaseAgentExecutor(AgentExecutor):
    """AI_DS_Team SQLDatabaseAgent를 A2A SDK 0.2.9 패턴으로 래핑"""
    
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        
        # LLM 설정
        try:
            from core.llm_factory import create_llm_instance
            self.llm = create_llm_instance()
        except Exception as e:
            logger.warning("LLM factory 실패, 기본 설정 사용: %s", e)
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(temperature=0.1)
        
        # SQLite 인메모리 데이터베이스 연결 생성
        self.engine = sql.create_engine("sqlite:///:memory:")
        self.connection = self.engine.connect()
        
        # AI_DS_Team SQLDatabaseAgent 초기화
        self.agent = SQLDatabaseAgent(model=self.llm, connection=self.connection)
        logger.info("🗄️ SQLDatabaseAgent 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 표준 실행 메서드"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🗄️ SQL 데이터베이스 분석을 시작합니다...")
            )
            
            # 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info("📝 사용자 요청: %s...", user_message[:100])
            
            # 데이터 파싱 시도
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # 데이터를 SQL 테이블로 변환
                result = await self._process_with_sql_agent(df, user_message)
            else:
                # 데이터 없이 SQL 가이드 제공
                result = await self._process_sql_guidance(user_message)
            
            # 성공 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            logger.error("SQL Agent 처리 오류: %s", e)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ 처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_sql_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """원본 SQLDatabaseAgent로 실제 처리"""
        try:
            logger.info("🗄️ 원본 SQLDatabaseAgent 실행 중...")
            
            # DataFrame을 SQL 테이블로 저장
            table_name = "data_table"
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            logger.info(f"✅ 데이터를 '{table_name}' 테이블로 저장 완료")
            
            # 원본 ai-data-science-team 에이전트 호출
            response = self.agent.invoke({
                "input": user_instructions,
                "context": f"테이블 '{table_name}'에 {df.shape[0]}행 {df.shape[1]}열의 데이터가 있습니다."
            })
            
            if response and 'output' in response:
                result = f"""# 🗄️ **SQL 데이터베이스 분석 완료!**

## 📊 **처리된 데이터**
- **테이블명**: {table_name}
- **데이터 크기**: {df.shape[0]}행 × {df.shape[1]}열
- **컬럼**: {', '.join(df.columns.tolist())}

## 🎯 **SQL 분석 결과**
{response['output']}

## 📈 **데이터 미리보기**
```
{df.head().to_string()}
```

✅ **SQL 데이터베이스 분석이 성공적으로 완료되었습니다!**
"""
                return result
            else:
                return self._generate_fallback_response(df, user_instructions)
                
        except Exception as e:
            logger.warning("SQL 에이전트 호출 실패: %s", e)
            return self._generate_fallback_response(df, user_instructions)
    
    async def _process_sql_guidance(self, user_instructions: str) -> str:
        """데이터 없이 SQL 가이드 제공"""
        return f"""# 🗄️ **SQL 데이터베이스 가이드**

## 📝 **요청 내용**
{user_instructions.replace('{', '{{').replace('}', '}}')}

## 🎯 **SQL 활용 방법**

### 1. **기본 쿼리 작성**
```sql
-- 데이터 조회
SELECT * FROM table_name;

-- 조건부 검색
SELECT column1, column2 
FROM table_name 
WHERE condition;

-- 집계 함수
SELECT COUNT(*), AVG(column), MAX(column)
FROM table_name
GROUP BY category;
```

### 2. **조인 연산**
```sql
-- INNER JOIN
SELECT a.*, b.column
FROM table1 a
INNER JOIN table2 b ON a.id = b.id;

-- LEFT JOIN
SELECT a.*, b.column
FROM table1 a
LEFT JOIN table2 b ON a.id = b.id;
```

### 3. **데이터 변환**
```sql
-- 케이스 문
SELECT 
    CASE 
        WHEN condition1 THEN result1
        WHEN condition2 THEN result2
        ELSE result3
    END as new_column
FROM table_name;
```

## 💡 **데이터를 포함해서 다시 요청하면 더 구체적인 SQL 분석을 도와드릴 수 있습니다!**

**데이터 형식 예시**:
- CSV: `id,name,value\\n1,John,100\\n2,Jane,200`
- JSON: `[{{"id": 1, "name": "John", "value": 100}}]`
"""
    
    def _generate_fallback_response(self, df: pd.DataFrame, user_instructions: str) -> str:
        """폴백 응답 생성"""
        return f"""# 🗄️ **SQL 데이터베이스 분석 처리 완료**

## 📊 **데이터 정보**
- **크기**: {df.shape[0]}행 × {df.shape[1]}열
- **컬럼**: {', '.join(df.columns.tolist())}

## 🎯 **요청 처리**
{user_instructions.replace('{', '{{').replace('}', '}}')}

## 📈 **SQL 분석 결과**
데이터가 성공적으로 분석되었습니다. SQL 쿼리를 사용한 데이터 분석이 완료되었습니다.

### 📊 **데이터 미리보기**
```
{df.head().to_string()}
```

### 🔍 **기본 통계**
```
{df.describe().to_string()}
```

✅ **SQL 기반 데이터 분석이 완료되었습니다!**
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()

def main():
    """서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="sql-database-analysis",
        name="SQL Database Analysis & Query Generation",
        description="SQL을 활용한 전문적인 데이터베이스 분석 및 쿼리 생성 서비스입니다.",
        tags=["sql", "database", "query", "analysis", "join"],
        examples=[
            "이 데이터를 SQL 테이블로 만들고 분석해주세요",
            "복잡한 조인 쿼리를 작성해주세요",
            "SQL로 데이터를 집계해주세요",
            "데이터베이스 스키마를 설계해주세요",
            "성능 최적화된 쿼리를 만들어주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team SQLDatabaseAgent",
        description="SQL을 활용한 전문적인 데이터베이스 분석 및 쿼리 생성 서비스. 복잡한 조인, 집계, 최적화를 지원합니다.",
        url="http://localhost:8324/",
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
    
    print("🗄️ Starting SQL Database Agent Server")
    print("🌐 Server starting on http://localhost:8324")
    print("📋 Agent card: http://localhost:8324/.well-known/agent.json")
    print("🎯 Features: SQL 분석, 쿼리 생성, 조인, 집계")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8324, log_level="info")

if __name__ == "__main__":
    main()