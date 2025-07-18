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

AI_DS_Team DataVisualizationAgent A2A Server
Port: 8308

AI_DS_Team의 DataVisualizationAgent를 A2A 프로토콜로 래핑하여 제공합니다.
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
from ai_data_science_team.agents import DataVisualizationAgent
from ai_data_science_team.utils.plotly import plotly_from_dict
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


class DataVisualizationAgentExecutor(AgentExecutor):
    """AI_DS_Team DataVisualizationAgent를 A2A 프로토콜로 래핑"""
    
    def __init__(self):
        # LLM 설정 (langfuse 콜백은 LLM 팩토리에서 자동 처리)
        from core.llm_factory import create_llm_instance
        self.llm = create_llm_instance()
        self.agent = DataVisualizationAgent(model=self.llm)
        logger.info("DataVisualizationAgent initialized with LLM factory (langfuse auto-enabled)")
    
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
                
                # 데이터 로딩
                data_manager = DataManager()
                available_data = data_manager.list_dataframes()
                
                # df 변수 초기화
                df = None
                data_file = None
                
                if not available_data:
                    result = """## ❌ 데이터 없음

데이터 시각화를 수행하려면 먼저 데이터를 업로드해야 합니다.

### 📤 데이터 업로드 방법
1. **UI에서 파일 업로드**: 메인 페이지에서 CSV, Excel 파일을 업로드하세요
2. **파일명 명시**: 자연어로 "data.xlsx 파일을 시각화해줘"와 같이 요청하세요
3. **지원 형식**: CSV, Excel (.xlsx, .xls), JSON, Pickle

**현재 상태**: 사용 가능한 데이터가 없습니다.
"""
                    
                    # 최종 응답 메시지 전송
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message(result)
                    )
                    return
                    
                else:
                    # 요청된 파일 찾기 또는 첫 번째 사용 가능한 데이터 사용
                    for part in context.message.parts:
                        if part.root.kind == "data" and hasattr(part.root, 'data'):
                            data_ref = part.root.data.get('data_reference', {})
                            requested_id = data_ref.get('data_id')
                            if requested_id and requested_id in available_data:
                                data_file = requested_id
                                break
                    
                    # 데이터 참조가 없으면 첫 번째 사용 가능한 데이터 사용
                    if data_file is None and available_data:
                        data_file = available_data[0]  # available_data는 List[str]
                        logger.info(f"데이터 참조가 없어서 첫 번째 사용 가능한 데이터 사용: {data_file}")
                    
                    if data_file:
                        # 데이터 로드 및 처리
                        df = data_manager.get_dataframe(data_file)
                        if df is not None:
                            logger.info(f"데이터 로드 성공: {data_file}, 형태: {df.shape}")
                            
                            # AI_DS_Team DataVisualizationAgent 실행
                            result = self.agent.invoke_agent(
                                user_instructions=user_instructions,
                                data_raw=df
                            )
                            
                            # 🔥 핵심 개선: Plotly 차트 아티팩트 생성 및 전송
                            try:
                                # 디버깅: agent response 확인
                                logger.info(f"🔍 DEBUG: Agent response keys: {list(self.agent.response.keys()) if self.agent.response else 'None'}")
                                if self.agent.response:
                                    logger.info(f"🔍 DEBUG: plotly_graph in response: {'plotly_graph' in self.agent.response}")
                                    if 'plotly_graph' in self.agent.response:
                                        logger.info(f"🔍 DEBUG: plotly_graph value: {type(self.agent.response['plotly_graph'])}")
                                
                                # Plotly 그래프 가져오기 - response에서 직접 가져오기
                                plotly_graph_raw = self.agent.response.get('plotly_graph') if self.agent.response else None
                                logger.info(f"🔍 DEBUG: plotly_graph_raw type: {type(plotly_graph_raw)} - {plotly_graph_raw is not None}")
                                
                                if plotly_graph_raw:
                                    # plotly_graph가 이미 dict인 경우 그대로 사용, Figure인 경우 변환
                                    if isinstance(plotly_graph_raw, dict):
                                        chart_json = json.dumps(plotly_graph_raw)
                                    else:
                                        # Figure 객체인 경우 JSON으로 변환
                                        import plotly.io as pio
                                        chart_json = pio.to_json(plotly_graph_raw)
                                    
                                    logger.info(f"🔍 DEBUG: Chart JSON length: {len(chart_json)}")
                                    
                                    # A2A 아티팩트로 전송
                                    await task_updater.add_artifact(
                                        parts=[TextPart(text=chart_json)],
                                        name="interactive_chart.json",
                                        metadata={
                                            "content_type": "application/vnd.plotly.v1+json",
                                            "chart_type": "plotly",
                                            "description": "Interactive Plotly chart"
                                        }
                                    )
                                    
                                    logger.info("✅ Plotly 차트 아티팩트 전송 완료")
                                else:
                                    logger.warning("⚠️ Plotly 그래프를 가져올 수 없음")
                                    
                            except Exception as chart_error:
                                logger.error(f"❌ 차트 아티팩트 생성 실패: {chart_error}", exc_info=True)
                            
                            # 워크플로우 요약 및 기본 응답 메시지
                            try:
                                # 결과 처리 (안전한 방식으로 workflow summary 가져오기)
                                try:
                                    workflow_summary = self.agent.get_workflow_summary(markdown=True)
                                except AttributeError:
                                    # get_workflow_summary 메서드가 없는 경우 기본 요약 생성
                                    workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                                except Exception as e:
                                    logger.warning(f"Error getting workflow summary: {e}")
                                    workflow_summary = f"✅ 작업이 완료되었습니다.\n\n**요청**: {user_instructions}"
                                
                                # 🔥 핵심 수정: Markdown 객체를 문자열로 변환
                                if workflow_summary is not None:
                                    if hasattr(workflow_summary, 'data'):
                                        workflow_summary = workflow_summary.data
                                    elif hasattr(workflow_summary, '_repr_markdown_'):
                                        workflow_summary = workflow_summary._repr_markdown_()
                                    elif hasattr(workflow_summary, '__str__'):
                                        workflow_summary = str(workflow_summary)
                                    else:
                                        workflow_summary = "✅ 📊 Data Visualization 작업 완료"
                                else:
                                    workflow_summary = "✅ 📊 Data Visualization 작업 완료"
                                    
                            except AttributeError as ae:
                                logger.warning(f"AttributeError getting workflow summary: {ae}")
                                workflow_summary = "✅ 📊 Data Visualization 작업 완료"
                            except Exception as e:
                                logger.warning(f"Error getting workflow summary: {e}")
                                workflow_summary = "✅ 📊 Data Visualization 작업 완료"
                            
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
                            
                            # 데이터 요약 생성 (df가 None이 아닌 경우에만)
                            data_summary_text = ""
                            if df is not None:
                                try:
                                    data_summary = get_dataframe_summary(df, n_sample=10)
                                    data_summary_text = data_summary[0] if data_summary else '데이터 요약을 생성할 수 없습니다.'
                                except Exception as e:
                                    logger.warning(f"데이터 요약 생성 실패: {e}")
                                    data_summary_text = f"데이터 형태: {df.shape}, 컬럼: {list(df.columns)}"
                            
                            result = f"""## 📊 데이터 시각화 완료

{workflow_summary}

{charts_info}

### 📋 사용된 데이터 요약
{data_summary_text}

### 🎨 Data Visualization Agent 기능
- **Plotly 차트**: 인터랙티브 차트 생성
- **Matplotlib 차트**: 고품질 정적 차트
- **통계 시각화**: 분포, 상관관계, 트렌드 분석
- **대시보드**: 복합 시각화 대시보드
- **커스텀 차트**: 요구사항에 맞는 맞춤형 시각화
"""
                            
                            # 최종 응답 메시지 전송 (문자열 보장)
                            final_message = str(result) if result else "✅ 📊 Data Visualization 작업 완료"
                            await task_updater.update_status(
                                TaskState.completed,
                                message=new_agent_text_message(final_message)
                            )
                            
                        else:
                            await task_updater.update_status(
                                TaskState.failed,
                                message=new_agent_text_message(f"❌ 데이터 로드 실패: {data_file}")
                            )
                    else:
                        await task_updater.update_status(
                            TaskState.failed,
                            message=new_agent_text_message("❌ 사용 가능한 데이터를 찾을 수 없습니다.")
                        )
                
            else:
                # 메시지가 없는 경우
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("시각화 요청이 비어있습니다. 구체적인 차트나 그래프 요청을 해주세요.")
                )
                
        except Exception as e:
            logger.error(f"Error in DataVisualizationAgent execution: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 시각화 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext) -> None:
        """작업 취소"""
        logger.info(f"DataVisualizationAgent task cancelled: {context.task_id}")


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_visualization",
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
        name="AI_DS_Team DataVisualizationAgent",
        description="전문적인 데이터 시각화 및 차트 생성 서비스. Plotly, Matplotlib을 활용하여 인터랙티브 및 정적 차트를 생성합니다.",
        url="http://localhost:8308/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataVisualizationAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📊 Starting AI_DS_Team DataVisualizationAgent Server")
    print("🌐 Server starting on http://localhost:8308")
    print("📋 Agent card: http://localhost:8308/.well-known/agent.json")
    print("🎨 Features: Interactive charts, static plots, dashboards")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8308, log_level="info")


if __name__ == "__main__":
    main() 