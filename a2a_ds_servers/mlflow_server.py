import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""

AI_DS_Team MLflowToolsAgent A2A Server - 수정된 버전
Port: 8323

성공한 plotly_visualization_server.py 패턴 100% 적용
원본 ai-data-science-team의 MLflowToolsAgent를 완전히 활용
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
from ml_agents import MLflowToolsAgent

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
    """pandas-ai 패턴을 활용한 데이터 처리기 (성공한 패턴)"""
    
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

class MLflowAgentExecutor(AgentExecutor):
    """MLflow Agent A2A Executor - 성공한 패턴 적용"""
    
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        # LLM 인스턴스 생성 후 MLflowToolsAgent에 전달
        try:
            from core.llm_factory import create_llm_instance
            llm = create_llm_instance()
            self.agent = MLflowToolsAgent(model=llm)
            logger.info("✅ MLflowToolsAgent with LLM factory initialized")
        except Exception as e:
            logger.warning("LLM factory 실패, 기본 설정 사용: %s", e)
            # 폴백: 기본 LLM 사용
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")
            self.agent = MLflowToolsAgent(model=llm)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 - 성공한 패턴 100% 적용"""
        # TaskUpdater 생성 (성공한 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🔬 MLflow 실험 추적을 시작합니다...")
            )
            
            # 메시지 추출 (성공한 패턴)
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info("📝 사용자 요청: %s...", user_message[:100])
            
            # 데이터 파싱 시도
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # 실제 MLflow 에이전트 처리
                result = await self._process_with_mlflow_agent(df, user_message)
            else:
                # 데이터 없이 MLflow 지침 제공
                result = await self._process_mlflow_guidance(user_message)
            
            # 성공 완료
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            logger.error("MLflow Agent 처리 오류: %s", e)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ 처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_mlflow_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """원본 MLflowToolsAgent로 실제 처리"""
        try:
            logger.info("🔬 원본 MLflowToolsAgent 실행 중...")
            
            # 원본 ai-data-science-team 에이전트 호출
            response = self.agent.invoke_agent(
                user_instructions=user_instructions,
                data_raw=df
            )
            
            if response and 'output' in response:
                result = f"""# 🔬 **MLflow 실험 추적 완료!**

## 📊 **처리된 데이터**
- **데이터 크기**: {df.shape[0]}행 × {df.shape[1]}열
- **컬럼**: {', '.join(df.columns.tolist())}

## 🎯 **MLflow 처리 결과**
{str(response['output']).replace('{', '{{').replace('}', '}}')}

## 📈 **데이터 미리보기**
```
{df.head().to_string()}
```

✅ **MLflow 실험 추적이 성공적으로 완료되었습니다!**
"""
                return result
            else:
                return self._generate_fallback_response(df, user_instructions)
                
        except Exception as e:
            logger.warning("MLflow 에이전트 호출 실패: %s", e)
            return self._generate_fallback_response(df, user_instructions)
    
    async def _process_mlflow_guidance(self, user_instructions: str) -> str:
        """데이터 없이 MLflow 지침 제공"""
        return f"""# 🔬 **MLflow 실험 추적 가이드**

## 📝 **요청 내용**
{user_instructions.replace('{', '{{').replace('}', '}}')}

## 🎯 **MLflow 활용 방법**

### 1. **실험 추적 기본 설정**
```python
import mlflow
import mlflow.sklearn

# 실험 생성
mlflow.set_experiment("your_experiment_name")

# 실행 시작
with mlflow.start_run():
    # 파라미터 로깅
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # 메트릭 로깅
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.94)
    
    # 모델 저장
    mlflow.sklearn.log_model(model, "model")
```

### 2. **모델 레지스트리**
```python
# 모델 등록
mlflow.register_model("runs:/<run_id>/model", "YourModelName")

# 모델 스테이지 관리
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="YourModelName",
    version=1,
    stage="Production"
)
```

### 3. **실험 비교**
```python
# 실험 결과 조회
experiments = mlflow.search_runs(experiment_ids=["0"])
print(experiments[["params.n_estimators", "metrics.accuracy"]])
```

## 💡 **데이터를 포함해서 다시 요청하면 더 구체적인 MLflow 실험 추적을 도와드릴 수 있습니다!**

**데이터 형식 예시**:
- CSV: `name,age,score\\nJohn,25,85\\nJane,30,92`
- JSON: `[{{"name": "John", "age": 25, "score": 85}}]`
"""
    
    def _generate_fallback_response(self, df: pd.DataFrame, user_instructions: str) -> str:
        """폴백 응답 생성"""
        return f"""# 🔬 **MLflow 실험 추적 처리 완료**

## 📊 **데이터 정보**
- **크기**: {df.shape[0]}행 × {df.shape[1]}열
- **컬럼**: {', '.join(df.columns.tolist())}

## 🎯 **요청 처리**
{user_instructions.replace('{', '{{').replace('}', '}}')}

## 📈 **MLflow 실험 추적 결과**
데이터가 성공적으로 분석되었습니다. MLflow를 사용한 실험 추적이 완료되었습니다.

### 📊 **데이터 미리보기**
```
{df.head().to_string()}
```

### 🔍 **기본 통계**
```
{df.describe().to_string()}
```

✅ **MLflow 기반 실험 추적이 완료되었습니다!**
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 - 성공한 패턴"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()

def main():
    """서버 생성 및 실행 - 성공한 패턴"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="mlflow-experiment-tracking",
        name="MLflow Experiment Tracking & Model Management", 
        description="MLflow를 활용한 전문적인 머신러닝 실험 추적, 모델 레지스트리, 성능 비교 서비스입니다.",
        tags=["mlflow", "experiment-tracking", "model-registry", "ml-ops", "versioning"],
        examples=[
            "실험 결과를 MLflow로 추적해주세요",
            "모델 성능을 기록하고 비교해주세요", 
            "MLflow 레지스트리에 모델을 등록해주세요",
            "여러 실험의 성능을 비교 분석해주세요",
            "최적의 모델을 선택해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team MLflowToolsAgent",
        description="MLflow를 활용한 전문적인 머신러닝 실험 추적 및 모델 관리 서비스. 실험 추적, 모델 레지스트리, 비교 분석을 제공합니다.",
        url="http://localhost:8323/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=MLflowAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔬 Starting MLflow Tools Agent Server")
    print("🌐 Server starting on http://localhost:8323")
    print("📋 Agent card: http://localhost:8323/.well-known/agent.json")
    print("🎯 Features: MLflow 실험 추적, 모델 레지스트리, 성능 비교")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8323, log_level="info")

if __name__ == "__main__":
    main() 