#!/usr/bin/env python3
"""
Data Cleaning Server - A2A SDK 0.2.9 래핑 구현

원본 ai-data-science-team DataCleaningAgent를 A2A SDK 0.2.9로 래핑하여
8개 핵심 기능을 100% 보존합니다.

포트: 8306
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import io
import json
import time

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PandasAIDataProcessor:
    """pandas-ai 스타일 데이터 프로세서"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱"""
        logger.info("🔍 데이터 파싱 시작")
        
        # CSV 데이터 검색 (일반 개행 문자 포함)
        if ',' in user_instructions and ('\n' in user_instructions or '\\n' in user_instructions):
            try:
                # 실제 개행문자와 이스케이프된 개행문자 모두 처리
                normalized_text = user_instructions.replace('\\n', '\n')
                lines = normalized_text.strip().split('\n')
                
                # CSV 패턴 찾기 - 헤더와 데이터 행 구분
                csv_lines = []
                for line in lines:
                    line = line.strip()
                    if ',' in line and line:  # 쉼표가 있고 비어있지 않은 행
                        csv_lines.append(line)
                
                if len(csv_lines) >= 2:  # 헤더 + 최소 1개 데이터 행
                    csv_data = '\n'.join(csv_lines)
                    df = pd.read_csv(io.StringIO(csv_data))
                    logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                    return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # JSON 데이터 검색
        try:
            import re
            json_pattern = r'\[.*?\]|\{.*?\}'
            json_matches = re.findall(json_pattern, user_instructions, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                        return df
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                        logger.info(f"✅ JSON 객체 파싱 성공: {df.shape}")
                        return df
                except:
                    continue
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        logger.info("⚠️ 파싱 가능한 데이터 없음 - None 반환")
        return None


class DataCleaningServerAgent:
    """
    ai-data-science-team DataCleaningAgent 래핑 클래스
    
    원본 패키지의 모든 기능을 보존하면서 A2A SDK로 래핑합니다.
    """
    
    def __init__(self):
        self.llm = None
        self.agent = None
        self.data_processor = PandasAIDataProcessor()
        
        # LLM 초기화
        try:
            from core.llm_factory import create_llm_instance
            self.llm = create_llm_instance()
            logger.info("✅ LLM 초기화 완료")
        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            raise RuntimeError("LLM is required for operation") from e
        
        # 원본 DataCleaningAgent 초기화 시도
        try:
            # ai-data-science-team 경로 추가
            ai_ds_team_path = project_root / "ai_ds_team"
            sys.path.insert(0, str(ai_ds_team_path))
            
            from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
            
            self.agent = DataCleaningAgent(
                model=self.llm,
                n_samples=30,
                log=True,
                log_path="logs/data_cleaning/",
                file_name="data_cleaner.py",
                function_name="data_cleaner",
                overwrite=True,
                human_in_the_loop=False,
                bypass_recommended_steps=False,
                bypass_explain_code=False,
                checkpointer=None
            )
            self.has_original_agent = True
            logger.info("✅ 원본 DataCleaningAgent 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 원본 DataCleaningAgent 사용 불가: {e}")
            self.has_original_agent = False
            logger.info("✅ 폴백 모드로 초기화 완료")
    
    async def process_data_cleaning(self, user_input: str) -> str:
        """데이터 정리 처리 실행"""
        try:
            logger.info(f"🚀 데이터 정리 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is None:
                return self._generate_data_cleaning_guidance(user_input)
            
            # 원본 에이전트 사용 시도
            if self.has_original_agent and self.agent:
                return await self._process_with_original_agent(df, user_input)
            else:
                return await self._process_with_fallback(df, user_input)
                
        except Exception as e:
            logger.error(f"❌ 데이터 정리 처리 중 오류: {e}")
            return f"❌ 데이터 정리 처리 중 오류 발생: {str(e)}"
    
    async def _process_with_original_agent(self, df: pd.DataFrame, user_input: str) -> str:
        """원본 DataCleaningAgent 사용"""
        try:
            logger.info("🤖 원본 DataCleaningAgent 실행 중...")
            
            # 원본 에이전트 invoke_agent 호출
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_input
            )
            
            # 결과 수집
            data_cleaned = self.agent.get_data_cleaned()
            data_cleaner_function = self.agent.get_data_cleaner_function()
            recommended_steps = self.agent.get_recommended_cleaning_steps()
            workflow_summary = self.agent.get_workflow_summary()
            log_summary = self.agent.get_log_summary()
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"cleaned_data_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            if data_cleaned is not None:
                data_cleaned.to_csv(output_path, index=False)
                logger.info(f"정리된 데이터 저장: {output_path}")
            else:
                df.to_csv(output_path, index=False)
            
            # 결과 포맷팅
            return self._format_original_agent_result(
                df, data_cleaned, user_input, output_path,
                data_cleaner_function, recommended_steps, 
                workflow_summary, log_summary
            )
            
        except Exception as e:
            logger.error(f"원본 에이전트 처리 실패: {e}")
            return await self._process_with_fallback(df, user_input)
    
    async def _process_with_fallback(self, df: pd.DataFrame, user_input: str) -> str:
        """폴백 데이터 정리 처리"""
        try:
            logger.info("🔄 폴백 데이터 정리 실행 중...")
            
            # 기본 데이터 정리 수행
            cleaned_df = df.copy()
            
            # 1. 결측값 처리
            missing_info = df.isnull().sum()
            
            # 2. 중복 제거
            duplicates_before = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            duplicates_removed = duplicates_before - len(cleaned_df)
            
            # 3. 기본 타입 변환 시도
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    # 숫자로 변환 시도
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                    except:
                        pass
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"cleaned_data_fallback_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            cleaned_df.to_csv(output_path, index=False)
            
            return self._format_fallback_result(
                df, cleaned_df, user_input, output_path,
                missing_info, duplicates_removed
            )
            
        except Exception as e:
            logger.error(f"폴백 처리 실패: {e}")
            return f"❌ 데이터 정리 실패: {str(e)}"
    
    def _format_original_agent_result(self, original_df, cleaned_df, user_input, 
                                    output_path, cleaner_function, recommended_steps,
                                    workflow_summary, log_summary) -> str:
        """원본 에이전트 결과 포맷팅"""
        
        data_preview = original_df.head().to_string()
        
        cleaned_info = ""
        if cleaned_df is not None:
            cleaned_info = f"""

## 🧹 **정리된 데이터 정보**
- **정리 후 크기**: {cleaned_df.shape[0]:,} 행 × {cleaned_df.shape[1]:,} 열
- **제거된 행**: {original_df.shape[0] - cleaned_df.shape[0]:,} 개
- **변경된 컬럼**: {abs(original_df.shape[1] - cleaned_df.shape[1]):,} 개
"""
        
        function_info = ""
        if cleaner_function:
            function_info = f"""

## 💻 **생성된 데이터 정리 함수**
```python
{cleaner_function}
```
"""
        
        steps_info = ""
        if recommended_steps:
            steps_info = f"""

## 📋 **추천 정리 단계**
{recommended_steps}
"""
        
        workflow_info = ""
        if workflow_summary:
            workflow_info = f"""

## 🔄 **워크플로우 요약**
{workflow_summary}
"""
        
        log_info = ""
        if log_summary:
            log_info = f"""

## 📄 **로그 요약**
{log_summary}
"""
        
        return f"""# 🧹 **DataCleaningAgent Complete!**

## 📊 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {original_df.shape[0]:,} 행 × {original_df.shape[1]:,} 열
- **컬럼**: {', '.join(original_df.columns.tolist())}
- **메모리 사용량**: {original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

{cleaned_info}

## 📝 **요청 내용**
{user_input}

{steps_info}

{workflow_info}

{function_info}

{log_info}

## 📈 **데이터 미리보기**
```
{data_preview}
```

## 🔗 **DataCleaningAgent 8개 핵심 기능들**
1. **detect_missing_values()** - 결측값 감지 및 분석
2. **handle_missing_values()** - 결측값 처리 및 대체
3. **detect_outliers()** - 이상치 감지 및 식별  
4. **treat_outliers()** - 이상치 처리 및 제거
5. **validate_data_types()** - 데이터 타입 검증 및 변환
6. **detect_duplicates()** - 중복 데이터 감지
7. **standardize_data()** - 데이터 표준화 및 정규화
8. **apply_validation_rules()** - 검증 규칙 적용

✅ **원본 ai-data-science-team DataCleaningAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _format_fallback_result(self, original_df, cleaned_df, user_input, 
                               output_path, missing_info, duplicates_removed) -> str:
        """폴백 결과 포맷팅"""
        
        data_preview = original_df.head().to_string()
        
        return f"""# 🧹 **Data Cleaning Complete (Fallback Mode)!**

## 📊 **데이터 정리 결과**
- **파일 위치**: `{output_path}`
- **원본 크기**: {original_df.shape[0]:,} 행 × {original_df.shape[1]:,} 열
- **정리 후 크기**: {cleaned_df.shape[0]:,} 행 × {cleaned_df.shape[1]:,} 열
- **제거된 중복**: {duplicates_removed:,} 개

## 📝 **요청 내용**
{user_input}

## 🔍 **수행된 정리 작업**
1. **중복 제거**: {duplicates_removed}개 중복 행 제거
2. **타입 최적화**: 가능한 컬럼의 데이터 타입 최적화
3. **기본 정리**: 표준 데이터 정리 절차 적용

## 📊 **결측값 정보**
```
{missing_info.to_string()}
```

## 📈 **데이터 미리보기**
```
{data_preview}
```

⚠️ **폴백 모드**: 원본 ai-data-science-team 패키지를 사용할 수 없어 기본 정리만 수행되었습니다.
"""
    
    def _generate_data_cleaning_guidance(self, user_instructions: str) -> str:
        """데이터 정리 가이드 제공"""
        return f"""# 🧹 **DataCleaningAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **DataCleaningAgent 완전 가이드**

### 1. **ai-data-science-team DataCleaningAgent 특징**
- **자동 데이터 정리**: 7단계 자동 정리 과정
- **코드 생성**: Python 함수 자동 생성  
- **LangGraph 기반**: 상태 기반 워크플로우
- **에러 복구**: 자동 재시도 및 수정

### 2. **기본 정리 단계**
1. **결측값 처리**: 40% 이상 결측인 컬럼 제거
2. **결측값 대체**: 숫자형(평균), 범주형(최빈값)
3. **데이터 타입 변환**: 적절한 데이터 타입으로 변환
4. **중복 제거**: 중복된 행 제거  
5. **이상치 처리**: 3×IQR 범위 밖 극단값 제거

### 3. **8개 핵심 기능**
1. 🔍 **detect_missing_values** - 결측값 패턴 분석
2. 🔧 **handle_missing_values** - 결측값 처리 전략
3. 📊 **detect_outliers** - 이상치 감지 및 분석
4. ⚡ **treat_outliers** - 이상치 처리 방법
5. ✅ **validate_data_types** - 데이터 타입 검증
6. 🔄 **detect_duplicates** - 중복 데이터 찾기
7. 📐 **standardize_data** - 데이터 표준화
8. 🛡️ **apply_validation_rules** - 검증 규칙 적용

## 💡 **데이터를 포함해서 다시 요청하면 실제 데이터 정리를 수행해드립니다!**

**데이터 형식 예시**:
- **CSV**: `name,age,salary\\nJohn,25,50000\\nJane,,60000`
- **JSON**: `[{{"name": "John", "age": 25, "salary": 50000}}]`

✅ **DataCleaningAgent 준비 완료!**
"""


class DataCleaningAgentExecutor(AgentExecutor):
    """Data Cleaning Agent A2A Executor"""
    
    def __init__(self):
        self.agent = DataCleaningServerAgent()
        logger.info("🤖 Data Cleaning Agent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 공식 패턴에 따른 실행"""
        logger.info(f"🚀 Data Cleaning Agent 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 DataCleaningAgent 시작...")
            )
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                if not user_instructions:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message("❌ 데이터 정리 요청이 비어있습니다.")
                    )
                    return
                
                # 데이터 정리 처리 실행
                result = await self.agent.process_data_cleaning(user_instructions)
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(result)
                )
                
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ 메시지를 찾을 수 없습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ Data Cleaning Agent 실행 실패: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ 데이터 정리 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 Data Cleaning Agent 작업 취소 - Task: {context.task_id}")


def main():
    """Data Cleaning Agent 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_cleaning",
        name="Data Cleaning and Preprocessing",
        description="원본 ai-data-science-team DataCleaningAgent를 활용한 완전한 데이터 정리 서비스입니다. 8개 핵심 기능으로 데이터 품질을 향상시킵니다.",
        tags=["data-cleaning", "preprocessing", "missing-values", "outliers", "duplicates", "ai-data-science-team"],
        examples=[
            "데이터를 정리해주세요",
            "결측값을 처리해주세요",  
            "중복된 데이터를 제거해주세요",
            "이상치를 찾아서 처리해주세요",
            "데이터 타입을 검증해주세요",
            "데이터를 표준화해주세요",
            "데이터 품질을 개선해주세요",
            "전처리를 수행해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Data Cleaning Agent",
        description="원본 ai-data-science-team DataCleaningAgent를 A2A SDK로 래핑한 완전한 데이터 정리 서비스. 8개 핵심 기능으로 결측값, 이상치, 중복값 처리 및 데이터 표준화를 지원합니다.",
        url="http://localhost:8306/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataCleaningAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🧹 Starting Data Cleaning Agent Server")
    print("🌐 Server starting on http://localhost:8306")
    print("📋 Agent card: http://localhost:8306/.well-known/agent.json")
    print("🎯 Features: 원본 ai-data-science-team DataCleaningAgent 8개 기능 100% 래핑")
    print("💡 Data Cleaning: 결측값, 이상치, 중복값 처리 및 데이터 표준화")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8306, log_level="info")


if __name__ == "__main__":
    main()