#!/usr/bin/env python3
"""
AI_DS_Team MLflowToolsAgent A2A Server
Port: 8314

AI_DS_Team의 MLflowToolsAgent를 A2A 프로토콜로 래핑하여 제공합니다.
MLflow 기반 머신러닝 실험 관리 및 모델 추적 전문
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
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.ml_agents import MLflowToolsAgent
import pandas as pd
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# Langfuse 로깅 설정 (선택적)
langfuse_handler = None
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler()
    logger.info("✅ Langfuse 로깅 활성화")
except ImportError:
    logger.info("⚠️ Langfuse 사용 불가 (선택적)")

# 유틸리티 함수 import
from a2a_ds_servers.utils.safe_data_loader import load_data_safely, create_safe_data_response

class MLflowToolsAgentExecutor(AgentExecutor):
    """MLflow Tools Agent A2A Executor"""
    
    def __init__(self):
        # LLM 설정 (langfuse 콜백은 LLM 팩토리에서 자동 처리)
        from core.llm_factory import create_llm_instance
        llm = create_llm_instance()
        self.agent = MLflowToolsAgent(model=llm)
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """MLflow Tools Agent 실행"""
        try:
            logger.info(f"🔬 MLflow Tools Agent 실행 시작: {context.task_id}")
            
            # 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message="🔄 MLflow 실험 관리를 시작합니다..."
            )
            
            # 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"사용자 요청: {user_instructions}")
                
                # 안전한 데이터 로딩 적용
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 데이터를 안전하게 로딩하고 있습니다..."
                )
                
                # 사용 가능한 데이터 스캔
                data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
                available_data = []
                
                try:
                    for file in os.listdir(data_path):
                        if file.endswith(('.csv', '.pkl')):
                            available_data.append(file)
                except:
                    pass
                
                # 안전한 데이터 로딩 적용
                df, data_file, error_msg = load_data_safely(
                    available_data=available_data,
                    preferred_file=None,
                    fallback_strategy='latest'
                )
                
                if df is not None and data_file is not None:
                    logger.info(f"✅ 안전한 데이터 로딩 성공: {data_file}, shape: {df.shape}")
                    
                    # MLflow Tools Agent 실행
                    await task_updater.update_status(
                        TaskState.working,
                        message="🔬 MLflow 실험 추적을 시작합니다..."
                    )
                    
                    try:
                        result = self.agent.invoke_agent(
                            user_instructions=user_instructions,
                            data_raw=df
                        )
                        
                        # 결과 처리 (안전한 방식으로 workflow summary 가져오기)
                        try:
                            workflow_summary = self.agent.get_workflow_summary(markdown=True)
                        except AttributeError:
                            # get_workflow_summary 메서드가 없는 경우 기본 요약 생성
                            workflow_summary = f"✅ MLflow 실험 추적이 완료되었습니다.\n\n**요청**: {user_instructions}"
                        except Exception as e:
                            logger.warning(f"워크플로우 요약 생성 오류: {e}")
                            workflow_summary = f"✅ MLflow 실험 추적이 완료되었습니다.\n\n**요청**: {user_instructions}"
                        
                        # 생성된 실험 정보 수집
                        experiments_info = ""
                        artifacts_path = "a2a_ds_servers/artifacts/plots/"
                        os.makedirs(artifacts_path, exist_ok=True)
                        
                        # 실험 파일 저장 확인
                        saved_files = []
                        try:
                            if os.path.exists(artifacts_path):
                                for file in os.listdir(artifacts_path):
                                    if file.endswith(('.png', '.jpg', '.html', '.json')):
                                        saved_files.append(file)
                        except:
                            pass
                        
                        if saved_files:
                            experiments_info += f"""
### 💾 저장된 실험 파일들
{chr(10).join([f"- {file}" for file in saved_files[-5:]])}
"""
                        
                        # 데이터 요약 생성
                        data_summary = get_dataframe_summary(df, n_sample=10)
                        
                        response_text = f"""## 🔬 MLflow 실험 관리 완료

{workflow_summary}

{experiments_info}

### 📋 사용된 데이터 요약
**파일**: {data_file}
**형태**: {df.shape}
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

### 🎯 MLflow 기능
- **실험 추적**: 모델 성능 및 매개변수 추적
- **모델 레지스트리**: 모델 버전 관리
- **아티팩트 저장**: 모델 및 결과 저장
- **비교 분석**: 실험 간 성능 비교
- **재현성**: 실험 재현 가능성 보장
"""
                        
                    except Exception as agent_error:
                        logger.warning(f"MLflow Tools Agent 실행 실패, 가이드 제공: {agent_error}")
                        response_text = f"""## 🔬 MLflow 실험 관리 가이드

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 MLflow 사용법
다음과 같은 요청을 시도해보세요:

1. **실험 추적**:
   - "모델 성능을 MLflow로 추적해주세요"
   - "실험 결과를 기록하고 비교해주세요"

2. **모델 관리**:
   - "모델을 MLflow 레지스트리에 등록해주세요"
   - "모델 버전을 관리해주세요"

3. **비교 분석**:
   - "여러 실험의 성능을 비교해주세요"
   - "최적 모델을 선택해주세요"

**사용된 데이터**: {data_file}
**데이터 형태**: {df.shape}
**요청**: {user_instructions}
"""
                
                else:
                    # 데이터 로딩 실패 시 안전한 응답 생성
                    response_text = create_safe_data_response(
                        df, data_file, user_instructions, "MLflow 실험 관리"
                    )
                    
                    if error_msg:
                        response_text += f"\n\n**오류 상세**: {error_msg}"
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(response_text)
                )
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("MLflow 요청이 비어있습니다. 구체적인 실험 관리 요청을 해주세요.")
                )
                
        except Exception as e:
            logger.error(f"MLflow Tools Agent 실행 중 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"MLflow 실험 관리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"MLflow Tools Agent 작업 취소: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="mlflow-tools",
        name="MLflow Experiment Tracking",
        description="MLflow를 활용한 전문적인 머신러닝 실험 추적 및 모델 관리 서비스. 실험 추적, 모델 레지스트리, 비교 분석을 제공합니다.",
        tags=["mlflow", "experiment-tracking", "model-registry", "ml-ops", "versioning"],
        examples=[
            "모델 성능을 MLflow로 추적해주세요",
            "실험 결과를 기록하고 비교해주세요",
            "모델을 MLflow 레지스트리에 등록해주세요",
            "여러 실험의 성능을 비교해주세요",
            "최적 모델을 선택해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team MLflowToolsAgent",
        description="MLflow를 활용한 전문적인 머신러닝 실험 추적 및 모델 관리 서비스. 실험 추적, 모델 레지스트리, 비교 분석을 제공합니다.",
        url="http://localhost:8314/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=MLflowToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔬 Starting AI_DS_Team MLflowToolsAgent Server")
    print("🌐 Server starting on http://localhost:8314")
    print("📋 Agent card: http://localhost:8314/.well-known/agent.json")
    print("🎯 Features: MLflow 실험 추적, 모델 레지스트리, 성능 비교")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8314, log_level="info")


if __name__ == "__main__":
    main() 