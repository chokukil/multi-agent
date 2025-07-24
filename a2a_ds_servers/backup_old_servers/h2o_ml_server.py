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

H2O ML Agent Server
Port: 8323

원본 ai-data-science-team의 H2OMLAgent를 A2A 프로토콜로 래핑하여 제공합니다.
H2O AutoML을 활용한 완전한 자동화된 머신러닝 모델링 서비스입니다.
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
import io
from pathlib import Path

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
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")


class PandasAIDataProcessor:
    """pandas-ai 스타일 데이터 프로세서 - 100% LLM First, 샘플 데이터 생성 절대 금지"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱 - 절대 샘플 데이터 생성 안함"""
        logger.info("🔍 데이터 파싱 시작 (샘플 데이터 생성 절대 금지)")
        
        # CSV 데이터 검색
        if ',' in user_instructions and '\n' in user_instructions:
            try:
                lines = user_instructions.strip().split('\n')
                csv_lines = [line for line in lines if ',' in line and any(c.isdigit() for c in line)]
                
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
        
        # 절대 샘플 데이터 생성 안함
        logger.info("⚠️ 파싱 가능한 데이터 없음 - None 반환 (샘플 데이터 생성 금지)")
        return None


class H2OMLServerAgent:
    """H2OMLAgent를 사용한 래퍼 클래스 - 성공한 loader_server.py 패턴 + 원본 100% 기능 구현"""
    
    def __init__(self):
        # 🔥 성공한 패턴 1: Data Manager 초기화 (필수)
        try:
            from core.data_manager import DataManager
            self.data_manager = DataManager()
            logger.info("✅ Data Manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Manager: {e}")
            raise RuntimeError("Data Manager is required for operation") from e
        
        # 🔥 성공한 패턴 2: Real LLM 초기화 (필수, 폴백 없음)
        self.llm = None
        self.agent = None
        
        try:
            # 공통 LLM 초기화 유틸리티 사용
            from base.llm_init_utils import create_llm_with_fallback
            
            self.llm = create_llm_with_fallback()
            from ai_data_science_team.ml_agents import H2OMLAgent
            
            # 🔥 원본 H2OMLAgent 초기화 (100% 원본 파라미터 보존)
            self.agent = H2OMLAgent(
                model=self.llm,
                log=True,
                log_path="logs/h2o/",
                model_directory="models/h2o/",
                overwrite=True,
                human_in_the_loop=False,
                bypass_recommended_steps=False,
                bypass_explain_code=False,
                enable_mlflow=False,
                mlflow_experiment_name="H2O AutoML",
                checkpointer=None
            )
            logger.info("✅ Real LLM initialized for H2O ML Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e
    
    def _detect_target_variable(self, df: pd.DataFrame, user_instructions: str) -> str:
        """타겟 변수 감지"""
        instructions_lower = user_instructions.lower()
        
        # 명시적 타겟 지정 확인
        for col in df.columns:
            if f"타겟은 {col}" in instructions_lower or f"target is {col}" in instructions_lower:
                return col
            if f"predict {col}" in instructions_lower or f"예측 {col}" in instructions_lower:
                return col
        
        # 일반적인 타겟 컬럼명 확인
        common_targets = ['target', 'label', 'y', 'class', 'prediction', 'result', 'churn', 'outcome']
        for target in common_targets:
            if target in df.columns:
                return target
        
        # 마지막 컬럼을 타겟으로 가정
        return df.columns[-1]
    
    async def process_h2o_ml(self, user_input: str) -> str:
        """H2O ML 처리 실행 - 원본 H2OMLAgent 100% 기능 구현"""
        try:
            logger.info(f"Processing H2O ML request: {user_input}")
            
            # 데이터 파싱 (성공한 패턴)
            data_processor = PandasAIDataProcessor()
            df = data_processor.parse_data_from_message(user_input)
            
            if df is None:
                # 데이터 없이 H2O 가이드 제공
                return self._generate_h2o_guidance(user_input)
            
            # 타겟 변수 감지
            target_variable = self._detect_target_variable(df, user_input)
            
            # 🔥 원본 H2OMLAgent.invoke_agent() 100% 호출
            logger.info(f"🎯 타겟 변수: {target_variable}")
            logger.info("🤖 원본 H2OMLAgent.invoke_agent 실행 중...")
            
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_input,
                target_variable=target_variable
            )
            
            # 🔥 원본 H2OMLAgent 모든 메서드 활용
            leaderboard = self.agent.get_leaderboard()
            best_model_id = self.agent.get_best_model_id()
            model_path = self.agent.get_model_path()
            h2o_function = self.agent.get_h2o_train_function()
            recommended_steps = self.agent.get_recommended_ml_steps()
            workflow_summary = self.agent.get_workflow_summary()
            log_summary = self.agent.get_log_summary()
            
            # 데이터를 공유 폴더에 저장 (성공한 패턴)
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            import time
            timestamp = int(time.time())
            output_file = f"h2o_ml_data_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to: {output_path}")
            
            # 🔥 원본 기능 100% 보존한 결과 구성
            leaderboard_info = ""
            if leaderboard is not None and not leaderboard.empty:
                leaderboard_info = f"""

### 🏆 **H2O AutoML Leaderboard**
```
{leaderboard.head().to_string()}
```

**총 모델 수**: {len(leaderboard)} 개
**최고 모델**: {best_model_id if best_model_id else "N/A"}
"""
            
            model_info = ""
            if model_path:
                model_info = f"""

### 💾 **모델 저장 정보**
- **모델 경로**: `{model_path}`
- **모델 ID**: {best_model_id if best_model_id else "N/A"}
"""
            
            h2o_function_info = ""
            if h2o_function:
                h2o_function_info = f"""

### 💻 **생성된 H2O AutoML 함수**
```python
{h2o_function}
```
"""
            
            recommended_info = ""
            if recommended_steps:
                recommended_info = f"""

### 📋 **추천 ML 단계**
{recommended_steps}
"""
            
            workflow_info = ""
            if workflow_summary:
                workflow_info = f"""

### 🔄 **워크플로우 요약**
{workflow_summary}
"""
            
            # 데이터 미리보기 안전하게 생성
            data_preview = df.head().to_string()
            
            # 최종 결과 구성
            result = f"""# 🤖 **H2O AutoML Complete!**

## 📊 **처리된 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **타겟 변수**: {target_variable}
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

{leaderboard_info}

{model_info}

{recommended_info}

{workflow_info}

{h2o_function_info}

### 📈 **데이터 미리보기**
```
{data_preview}
```

## 🔗 **활용 가능한 메서드들**
- `get_leaderboard()` - H2O AutoML 리더보드
- `get_best_model_id()` - 최고 성능 모델 ID
- `get_model_path()` - 저장된 모델 경로
- `get_h2o_train_function()` - 생성된 H2O 함수 코드
- `get_recommended_ml_steps()` - ML 추천 단계
- `get_workflow_summary()` - 워크플로우 요약
- `get_log_summary()` - 상세 로그 요약

✅ **원본 ai-data-science-team H2OMLAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
            
            logger.info("✅ H2O ML 처리 완료")
            return result
            
        except Exception as e:
            logger.error(f"H2O ML 처리 중 오류: {e}")
            return f"❌ H2O AutoML 처리 중 오류 발생: {str(e)}\n\n**해결 방법**: H2O 라이브러리 설치 확인 (`pip install h2o`)"
    
    def _generate_h2o_guidance(self, user_instructions: str) -> str:
        """H2O AutoML 가이드 제공"""
        return f"""# 🤖 **H2O AutoML 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **H2O AutoML 완전 가이드**

### 1. **H2O AutoML 설치 및 초기화**
```python
# H2O 설치
pip install h2o

# H2O 초기화
import h2o
h2o.init()
```

### 2. **기본 분류 모델**
```python
from h2o.automl import H2OAutoML

# 데이터 로드
train = h2o.import_file("train.csv")
test = h2o.import_file("test.csv")

# 특성과 타겟 정의
x = train.columns
y = "target_column"
x.remove(y)

# AutoML 실행
aml = H2OAutoML(max_models=20, seed=1, max_runtime_secs=300)
aml.train(x=x, y=y, training_frame=train)

# 리더보드 확인
print(aml.leaderboard.head())
```

### 3. **원본 H2OMLAgent 기능들**
```python
from ml_agents import H2OMLAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
agent = H2OMLAgent(model=llm, log=True)

# 모델 훈련
agent.invoke_agent(
    data_raw=df,
    user_instructions="분류 모델 생성",
    target_variable="target"
)

# 결과 확인
leaderboard = agent.get_leaderboard()
best_model = agent.get_best_model_id()
model_path = agent.get_model_path()
```

## 💡 **데이터를 포함해서 다시 요청하면 실제 H2O AutoML 모델링을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `feature1,feature2,target\\n1.0,2.0,1\\n1.5,2.5,0`
- **JSON**: `[{"feature1": 1.0, "feature2": 2.0, "target": 1}]`

### 🔗 **추가 리소스**
- H2O 공식 문서: https://docs.h2o.ai/
- H2O AutoML 가이드: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- H2O 튜토리얼: https://github.com/h2oai/h2o-tutorials

✅ **H2O AutoML 준비 완료!**
"""

    # 🔥 원본 H2OMLAgent 모든 메서드들 구현
    def get_leaderboard(self):
        """원본 H2OMLAgent.get_leaderboard() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_leaderboard()
        return None
    
    def get_best_model_id(self):
        """원본 H2OMLAgent.get_best_model_id() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_best_model_id()
        return None
    
    def get_model_path(self):
        """원본 H2OMLAgent.get_model_path() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_model_path()
        return None
    
    def get_data_raw(self):
        """원본 H2OMLAgent.get_data_raw() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_raw()
        return None
    
    def get_h2o_train_function(self, markdown=False):
        """원본 H2OMLAgent.get_h2o_train_function() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_h2o_train_function(markdown=markdown)
        return None
    
    def get_recommended_ml_steps(self, markdown=False):
        """원본 H2OMLAgent.get_recommended_ml_steps() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_recommended_ml_steps(markdown=markdown)
        return None
    
    def get_workflow_summary(self, markdown=False):
        """원본 H2OMLAgent.get_workflow_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_workflow_summary(markdown=markdown)
        return None
    
    def get_log_summary(self, markdown=False):
        """원본 H2OMLAgent.get_log_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_log_summary(markdown=markdown)
        return None


class H2OMLAgentExecutor(AgentExecutor):
    """H2O ML Agent A2A Executor with Langfuse integration"""
    
    def __init__(self):
        self.agent = H2OMLServerAgent()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ H2OMLAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("🤖 H2O ML Agent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """H2O ML Agent 실행 - 성공한 loader_server.py 패턴"""
        logger.info(f"🚀 H2O ML Agent 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화 (성공한 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 원본 ai-data-science-team H2OMLAgent 시작...")
            )
            
            # 사용자 메시지 추출 (성공한 패턴)
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
                        message=new_agent_text_message("❌ H2O AutoML 요청이 비어있습니다.")
                    )
                    return
                
                # H2O ML 처리 실행
                result = await self.agent.process_h2o_ml(user_instructions)
                
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
            logger.error(f"❌ H2O ML Agent 실행 실패: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ H2O ML 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 H2O ML Agent 작업 취소 - Task: {context.task_id}")


def main():
    """H2O ML Agent 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="h2o_automl",
        name="H2O AutoML Modeling", 
        description="원본 ai-data-science-team H2OMLAgent를 활용한 완전한 AutoML 서비스입니다. H2O AutoML을 통해 자동으로 최적의 머신러닝 모델을 찾고 훈련합니다.",
        tags=["h2o", "automl", "machine-learning", "modeling", "prediction", "ai-data-science-team", "classification", "regression"],
        examples=[
            "H2O AutoML로 분류 모델을 구축해주세요",
            "회귀 분석 모델을 자동으로 생성해주세요", 
            "최적의 머신러닝 모델을 찾아주세요",
            "H2O를 사용해서 모델 성능을 비교해주세요",
            "AutoML로 예측 모델을 만들어주세요",
            "고객 이탈 예측 모델을 생성해주세요",
            "매출 예측을 위한 회귀 모델을 만들어주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="H2O ML Agent",
        description="원본 ai-data-science-team H2OMLAgent를 활용한 완전한 AutoML 서비스. H2O AutoML을 통해 자동으로 최적의 머신러닝 모델을 찾고 훈련하며, 모델 평가 및 저장을 지원합니다.",
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
        agent_executor=H2OMLAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🤖 Starting H2O ML Agent Server")
    print("🌐 Server starting on http://localhost:8313")
    print("📋 Agent card: http://localhost:8313/.well-known/agent.json")
    print("🎯 Features: 원본 ai-data-science-team H2OMLAgent 100% + 성공한 A2A 패턴")
    print("💡 H2O AutoML: 자동 모델 선택, 하이퍼파라미터 튜닝, 성능 평가")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8313, log_level="info")


if __name__ == "__main__":
    main()