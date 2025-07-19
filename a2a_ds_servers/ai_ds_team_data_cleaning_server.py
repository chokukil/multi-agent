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

from a2a.utils import new_agent_text_message
#!/usr/bin/env python3
"""

AI_DS_Team DataCleaningAgent A2A Server
Port: 8306

AI_DS_Team의 DataCleaningAgent를 A2A 프로토콜로 래핑하여 제공합니다.
데이터 정리 및 품질 개선 전문
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
import uvicorn
import logging

# AI_DS_Team imports
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.agents import DataCleaningAgent
import pandas as pd
import json
from core.data_manager import DataManager
from core.session_data_manager import SessionDataManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()
session_data_manager = SessionDataManager()

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
langsmith_handler = None
if os.getenv("LOGGING_PROVIDER") in ["langsmith", "both"]:
    try:
        from langsmith.run_helpers import CallbackHandler
        langsmith_handler = CallbackHandler(
            project_name=os.getenv("LANGSMITH_PROJECT_NAME", "ai-ds-team"),
            api_key=os.getenv("LANGSMITH_API_KEY"),
        )
        logger.info("✅ LangSmith logging enabled")
    except Exception as e:
        logger.warning(f"⚠️ LangSmith logging setup failed: {e}")

# 유틸리티 함수 import
from a2a_ds_servers.utils.safe_data_loader import load_data_safely, create_safe_data_response

class DataCleaningAgentExecutor(AgentExecutor):
    """AI_DS_Team DataCleaningAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정 (langfuse 콜백은 LLM 팩토리에서 자동 처리)
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = DataCleaningAgent(
            model=self.llm,
            log=True,
            log_path="logs/generated_code/"
        )
        logger.info("✅ DataCleaningAgent 초기화 완료 (LLM 팩토리 사용)")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 프로토콜에 따른 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧹 Data Cleaning Agent 분석을 시작합니다...")
            )
            
            # 사용자 메시지 추출
            user_instructions = ""
            data_reference = None
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                    elif part.root.kind == "data" and hasattr(part.root, 'data'):
                        data_reference = part.root.data.get('data_reference', {})
                
                user_instructions = user_instructions.strip()
                logger.info(f"사용자 요청: {user_instructions}")
                
                # 안전한 데이터 로딩 적용
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("📊 데이터를 안전하게 로딩하고 있습니다...")
                )
                
                # 데이터 로딩 - DataManager 사용
                available_data = data_manager.list_dataframes()
                
                # 안전한 데이터 선택 로직 적용
                df = None
                selected_data_id = None
                
                if not available_data:
                    result = create_safe_data_response(
                        None, None, user_instructions, "Data Cleaning Agent"
                    )
                else:
                    # 1. 명시적 데이터 요청 확인
                    if data_reference and 'data_id' in data_reference:
                        requested_id = data_reference['data_id']
                        if requested_id in available_data:
                            selected_data_id = requested_id
                    
                    # 2. 데이터 선택 안전성 확인
                    if selected_data_id is None and available_data:
                        # 가장 최근 데이터를 기본으로 선택
                        selected_data_id = available_data[0]
                        logger.info(f"기본 데이터 선택: {selected_data_id}")
                    
                    # 3. 데이터 로딩
                    if selected_data_id:
                        df = data_manager.get_dataframe(selected_data_id)
                        
                        if df is not None:
                            logger.info(f"✅ 데이터 로딩 성공: {selected_data_id}, shape: {df.shape}")
                            
                            # Data Cleaning Agent 실행
                            await task_updater.update_status(
                                TaskState.working,
                                message=new_agent_text_message("🧹 데이터 정리 작업을 실행하고 있습니다...")
                            )
                            
                            try:
                                result = self.agent.invoke_agent(
                                    user_instructions=user_instructions,
                                    data_raw=df
                                )
                                
                                # 결과 처리
                                try:
                                    ai_message = self.agent.get_ai_message(markdown=True)
                                except AttributeError:
                                    ai_message = "✅ 데이터 정리가 완료되었습니다."
                                except Exception as e:
                                    logger.warning(f"AI 메시지 생성 오류: {e}")
                                    ai_message = "✅ 데이터 정리가 완료되었습니다."
                                
                                # 정리된 데이터 저장
                                cleaned_data_info = ""
                                if hasattr(self.agent, 'data') and self.agent.data is not None:
                                    # 정리된 데이터 공유 폴더에 저장
                                    output_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/cleaned_data_{context.task_id}.csv"
                                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                    self.agent.data.to_csv(output_path, index=False)
                                    
                                    # 정리 전후 비교
                                    original_shape = df.shape
                                    cleaned_shape = self.agent.data.shape
                                    
                                    # 데이터 요약 생성
                                    try:
                                        data_summary = get_dataframe_summary(self.agent.data, n_sample=10)
                                        summary_text = data_summary[0] if data_summary else "데이터 요약 생성 불가"
                                    except Exception as e:
                                        logger.warning(f"데이터 요약 생성 오류: {e}")
                                        summary_text = "데이터 요약 생성 불가"
                                    
                                    cleaned_data_info = f"""
### 📊 데이터 정리 결과
- **원본 데이터**: {original_shape[0]:,} 행 × {original_shape[1]:,} 열
- **정리된 데이터**: {cleaned_shape[0]:,} 행 × {cleaned_shape[1]:,} 열
- **변화**: {cleaned_shape[0] - original_shape[0]:+,} 행, {cleaned_shape[1] - original_shape[1]:+,} 열

### 📋 정리된 데이터 요약
{summary_text}

### 💾 저장된 파일
- **경로**: {output_path}
"""
                                
                                # 최종 응답 생성
                                response_text = f"""## 🧹 데이터 정리 완료

{ai_message}

{cleaned_data_info}

### 📋 사용된 데이터 정보
**파일**: {selected_data_id}
**원본 형태**: {df.shape}

### 🛠️ Data Cleaning Agent 기능
- **결측값 처리**: 지능적인 결측값 대체 및 제거
- **중복 제거**: 중복 행 탐지 및 제거
- **이상값 탐지**: 통계적 이상값 식별 및 처리
- **데이터 타입 최적화**: 메모리 효율성 개선
- **품질 평가**: 데이터 품질 지표 제공
"""
                                
                                result = response_text
                                
                            except Exception as agent_error:
                                logger.warning(f"Data Cleaning Agent 실행 실패: {agent_error}")
                                result = f"""## 🧹 데이터 정리 가이드

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 Data Cleaning Agent 사용법
다음과 같은 요청을 시도해보세요:

1. **결측값 처리**:
   - "결측값을 처리해주세요"
   - "빈 값을 적절히 채워주세요"

2. **중복 제거**:
   - "중복된 행을 제거해주세요"
   - "중복 데이터를 찾아서 정리해주세요"

3. **이상값 처리**:
   - "이상값을 탐지하고 처리해주세요"
   - "통계적 이상값을 찾아주세요"

4. **데이터 품질 개선**:
   - "데이터 품질을 평가하고 개선해주세요"
   - "데이터 타입을 최적화해주세요"

**사용된 데이터**: {selected_data_id}
**데이터 형태**: {df.shape}
**요청**: {user_instructions}
"""
                        else:
                            result = f"❌ 데이터 로드 실패: {selected_data_id}"
                    else:
                        result = f"""## ❌ 적절한 데이터를 찾을 수 없음

**사용 가능한 데이터**: {', '.join(available_data)}

**해결 방법**:
1. 사용 가능한 데이터 중 하나를 선택하여 요청하세요
2. 원하는 파일을 먼저 업로드해주세요

**요청**: {user_instructions}
"""
                
                # 최종 응답 전송
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(result)
                )
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ 데이터 정리 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ Data Cleaning Agent 실행 중 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 정리 중 오류 발생: {str(e)}")
            )

    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"Data Cleaning Agent 작업 취소: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_cleaning",
        name="Data Cleaning & Quality Improvement",
        description="전문적인 데이터 정리 및 품질 개선 서비스. 결측값 처리, 중복 제거, 이상값 탐지, 데이터 타입 최적화 등을 수행합니다.",
        tags=["data-cleaning", "preprocessing", "quality-improvement", "missing-values", "outliers"],
        examples=[
            "결측값을 처리해주세요",
            "중복 데이터를 제거해주세요", 
            "데이터 품질을 평가해주세요",
            "이상값을 탐지하고 처리해주세요",
            "데이터 타입을 최적화해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team DataCleaningAgent",
        description="데이터 정리 및 품질 개선 전문가. 결측값 처리, 중복 제거, 이상값 탐지, 데이터 타입 최적화 등을 수행합니다.",
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
    
    print("🧹 Starting AI_DS_Team DataCleaningAgent Server")
    print("🌐 Server starting on http://localhost:8306")
    print("📋 Agent card: http://localhost:8306/.well-known/agent.json")
    print("🛠️ Features: Data cleaning, quality improvement, preprocessing")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8306, log_level="info")


if __name__ == "__main__":
    main() 