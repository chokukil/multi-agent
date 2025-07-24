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

from a2a.utils import new_agent_text_message#!/usr/bin/env python3
"""

AI_DS_Team DataWranglingAgent A2A Server
Port: 8309

AI_DS_Team의 DataWranglingAgent를 A2A 프로토콜로 래핑하여 제공합니다.
데이터 시각화 및 차트 생성 전문
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
from ai_data_science_team.agents import DataWranglingAgent
from ai_data_science_team.utils.plotly import plotly_from_dict
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


class DataWranglingAgentExecutor(AgentExecutor):
    """AI_DS_Team DataWranglingAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정 (langfuse 콜백은 LLM 팩토리에서 자동 처리)
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = DataWranglingAgent(model=self.llm)
        logger.info("DataWranglingAgent initialized with LLM factory (langfuse auto-enabled)")
    
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
                logger.info(f"Processing data visualization request: {user_instructions}")
                
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
                    # 데이터 파일 선택 로직 개선
                    data_file = None
                    
                    # 1. 사용자 요청에서 특정 데이터 파일 언급 확인
                    user_lower = user_instructions.lower()
                    for file in available_data:
                        file_name_lower = file.lower()
                        file_base = file_name_lower.replace('.csv', '').replace('.pkl', '')
                        
                        # 파일명이 사용자 요청에 포함되어 있는지 확인
                        if (file_base in user_lower or 
                            any(keyword in file_name_lower for keyword in user_lower.split() if len(keyword) > 3)):
                            data_file = file
                            logger.info(f"🎯 사용자 요청에서 언급된 데이터 파일 선택: {data_file}")
                            break
                    
                    # 2. ion_implant 데이터 우선 선택 (반도체 분석 특화)
                    if not data_file:
                        for file in available_data:
                            if "ion_implant" in file.lower():
                                data_file = file
                                logger.info(f"🔬 반도체 분석용 ion_implant 데이터 선택: {data_file}")
                                break
                    
                    # 3. 가장 최근 수정된 파일 선택 (fallback)
                    if not data_file:
                        try:
                            file_times = []
                            for file in available_data:
                                file_path = os.path.join(data_path, file)
                                if os.path.exists(file_path):
                                    mtime = os.path.getmtime(file_path)
                                    file_times.append((file, mtime))
                            
                            if file_times:
                                file_times.sort(key=lambda x: x[1], reverse=True)
                                data_file = file_times[0][0]
                                logger.info(f"📅 가장 최근 파일 선택: {data_file}")
                            else:
                                data_file = available_data[0]
                                logger.info(f"📁 첫 번째 사용 가능한 파일 선택: {data_file}")
                        except Exception as e:
                            logger.warning(f"파일 시간 정렬 실패: {e}")
                            data_file = available_data[0]
                            logger.info(f"📁 기본 파일 선택: {data_file}")
                    
                    if data_file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(data_path, data_file))
                    else:
                        df = pd.read_pickle(os.path.join(data_path, data_file))
                    
                    logger.info(f"Loaded data: {data_file}, shape: {df.shape}")
                    
                    # DataWranglingAgent 실행
                    try:
                        result = self.agent.invoke_agent(
                            user_instructions=user_instructions,
                            data_raw=df
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
                        
                        # 생성된 차트 정보 수집
                        charts_info = ""
                        artifacts_path = "a2a_ds_servers/artifacts/plots/"
                        os.makedirs(artifacts_path, exist_ok=True)
                        
                        # 차트 파일 저장 확인
                        saved_files = []
                        try:
                            if os.path.exists(artifacts_path):
                                for file in os.listdir(artifacts_path):
                                    if file.endswith(('.png', '.jpg', '.html', '.json')):
                                        saved_files.append(file)
                        except:
                            pass
                        
                        if saved_files:
                            charts_info += f"""
### 💾 저장된 차트 파일들
{chr(10).join([f"- {file}" for file in saved_files[-5:]])}
"""
                        
                        # 데이터 요약 생성
                        data_summary = get_dataframe_summary(df, n_sample=10)
                        
                        response_text = f"""## 📊 데이터 시각화 완료

{workflow_summary}

{charts_info}

### 📋 사용된 데이터 요약
{data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'}

### 🎨 Data Visualization Agent 기능
- **Plotly 차트**: 인터랙티브 차트 생성
- **Matplotlib 차트**: 고품질 정적 차트
- **통계 시각화**: 분포, 상관관계, 트렌드 분석
- **대시보드**: 복합 시각화 대시보드
- **커스텀 차트**: 요구사항에 맞는 맞춤형 시각화
"""
                        
                    except Exception as agent_error:
                        logger.warning(f"Agent execution failed, providing guidance: {agent_error}")
                        response_text = f"""## 📊 데이터 시각화 가이드

요청을 처리하는 중 문제가 발생했습니다: {str(agent_error)}

### 💡 데이터 시각화 사용법
다음과 같은 요청을 시도해보세요:

1. **기본 차트**:
   - "매출 데이터의 월별 트렌드를 선 그래프로 그려주세요"
   - "고객 나이대별 분포를 히스토그램으로 보여주세요"

2. **고급 시각화**:
   - "변수들 간의 상관관계를 히트맵으로 표시해주세요"
   - "카테고리별 박스플롯을 생성해주세요"

3. **인터랙티브 차트**:
   - "Plotly를 사용해서 인터랙티브 차트를 만들어주세요"
   - "대시보드 형태로 여러 차트를 조합해주세요"

요청: {user_instructions}
"""
                
                else:
                    response_text = f"""## ❌ 데이터 없음

데이터 시각화를 수행하려면 먼저 데이터를 업로드해야 합니다.
사용 가능한 데이터가 없습니다: {data_path}

요청: {user_instructions}

### 📊 Data Visualization Agent 기능
- **차트 유형**: 선그래프, 막대그래프, 히스토그램, 산점도, 박스플롯 등
- **인터랙티브**: Plotly 기반 동적 차트
- **정적 차트**: Matplotlib 기반 고품질 이미지
- **통계 시각화**: 분포 분석, 상관관계 분석
- **대시보드**: 복합 시각화 레이아웃
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
                    message=new_agent_text_message("시각화 요청이 비어있습니다. 구체적인 차트나 그래프 요청을 해주세요.")
                )
                
        except Exception as e:
            logger.error(f"Error in DataWranglingAgent execution: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 시각화 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"DataWranglingAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data-wrangling",
        name="Data Visualization & Chart Creation",
        description="전문적인 데이터 시각화 및 차트 생성 서비스. Plotly, Matplotlib을 활용하여 인터랙티브 및 정적 차트를 생성합니다.",
        tags=["data-visualization", "plotly", "matplotlib", "charts", "dashboard"],
        examples=[
            "매출 데이터의 월별 트렌드를 선 그래프로 그려주세요",
            "고객 나이대별 분포를 히스토그램으로 보여주세요",
            "변수들 간의 상관관계를 히트맵으로 표시해주세요",
            "카테고리별 박스플롯을 생성해주세요",
            "인터랙티브 대시보드를 만들어주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI_DS_Team DataWranglingAgent",
        description="전문적인 데이터 시각화 및 차트 생성 서비스. Plotly, Matplotlib을 활용하여 인터랙티브 및 정적 차트를 생성합니다.",
        url="http://localhost:8309/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataWranglingAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📊 Starting AI_DS_Team DataWranglingAgent Server")
    print("🌐 Server starting on http://localhost:8309")
    print("📋 Agent card: http://localhost:8309/.well-known/agent.json")
    print("🎨 Features: Interactive charts, static plots, dashboards")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8309, log_level="info")


if __name__ == "__main__":
    main() 