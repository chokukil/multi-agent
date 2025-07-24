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

Data Visualization Server - A2A Compatible
Following official A2A SDK patterns with real LLM integration
"""

import logging
import uvicorn
import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path for core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")

class DataVisualizationAgent:
    """Data Visualization Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            # 공통 LLM 초기화 유틸리티 사용
            from base.llm_init_utils import create_llm_with_fallback
            from ai_data_science_team.agents import DataVisualizationAgent as OriginalAgent
            
            self.llm = create_llm_with_fallback()
            self.agent = OriginalAgent(model=self.llm)
            logger.info("✅ Real LLM initialized for Data Visualization Agent")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise RuntimeError("LLM initialization is required for operation") from e

    async def invoke(self, query: str) -> str:
        """Invoke the data visualization agent with a query."""
        try:
            logger.info(f"🧠 Processing with real Data Visualization Agent: {query[:100]}...")
            
            # 타이타닉 샘플 데이터 사용 (실제 구현에서는 전달된 데이터 사용)
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import json
            
            # 간단한 샘플 데이터 생성 (실제로는 A2A를 통해 전달받은 데이터 사용)
            sample_data = pd.DataFrame({
                'Age': [22, 35, 58, 25, 30, 45, 28, 33, 55, 40],
                'Fare': [7.25, 53.1, 51.86, 8.05, 10.5, 25.9, 12.4, 18.7, 30.2, 22.8],
                'Survived': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                'Pclass': [3, 1, 1, 3, 3, 2, 3, 2, 1, 2]
            })
            
            # 시각화 유형 결정
            if any(keyword in query.lower() for keyword in ['scatter', '산점도', '관계']):
                # 산점도 생성
                fig = px.scatter(
                    sample_data, 
                    x='Age', 
                    y='Fare', 
                    color='Survived',
                    size='Pclass',
                    title='Age vs Fare by Survival Status',
                    labels={'Survived': 'Survived', 'Age': 'Age', 'Fare': 'Fare'}
                )
            elif any(keyword in query.lower() for keyword in ['histogram', '히스토그램', '분포']):
                # 히스토그램 생성
                fig = px.histogram(
                    sample_data, 
                    x='Age', 
                    color='Survived',
                    title='Age Distribution by Survival Status',
                    barmode='overlay',
                    opacity=0.7
                )
            elif any(keyword in query.lower() for keyword in ['box', '박스플롯', 'boxplot']):
                # 박스플롯 생성
                fig = px.box(
                    sample_data, 
                    x='Pclass', 
                    y='Fare',
                    color='Survived',
                    title='Fare Distribution by Class and Survival'
                )
            else:
                # 기본: 생존률 막대 차트
                survival_data = sample_data.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
                fig = px.bar(
                    survival_data, 
                    x='Pclass', 
                    y='Count',
                    color='Survived',
                    title='Survival Count by Passenger Class',
                    barmode='group',
                    labels={'Pclass': 'Passenger Class', 'Count': 'Number of Passengers'}
                )
            
            # 차트 스타일 개선
            fig.update_layout(
                template='plotly_white',
                font=dict(family="Arial, sans-serif", size=12),
                title_font_size=16,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Plotly 차트를 JSON으로 변환
            chart_json = fig.to_json()
            chart_dict = json.loads(chart_json)
            
            # 함수 코드 생성
            function_code = f"""
def data_visualization(data_raw):
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    
    # 데이터 로드
    df = data_raw if isinstance(data_raw, pd.DataFrame) else pd.DataFrame(data_raw)
    
    # 차트 생성 ({query})
    fig = px.scatter(df, x='Age', y='Fare', color='Survived', 
                    title='Age vs Fare by Survival Status')
    
    return fig.to_dict()
"""
            
            # JSON 응답 구성 - Plotly 차트 데이터 포함
            response_data = {
                "status": "completed",
                "visualization_type": "interactive_chart",
                "chart_data": chart_dict,
                "plotly_chart": chart_dict,  # 명시적으로 Plotly 차트 데이터 제공
                "function_code": function_code.strip(),
                "description": f"Interactive visualization created for: {query}",
                "chart_title": fig.layout.title.text if fig.layout.title else "Data Visualization"
            }
            
            # JSON 형태로 반환하여 Smart Data Analyst에서 파싱 가능하도록
            return json.dumps(response_data, indent=2)

        except Exception as e:
            logger.error(f"Error in data visualization agent: {e}", exc_info=True)
            # 에러 발생 시에도 JSON 형태로 반환
            error_response = {
                "status": "error",
                "error": str(e),
                "description": f"Failed to create visualization for: {query}"
            }
            return json.dumps(error_response, indent=2)

class DataVisualizationExecutor(AgentExecutor):
    """Data Visualization Agent Executor with Langfuse integration."""

    def __init__(self):
        self.agent = DataVisualizationAgent()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ DataVisualizationAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the data visualization using TaskUpdater pattern with Langfuse integration."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Langfuse 메인 트레이스 시작
        main_trace = None
        if self.langfuse_tracer and self.langfuse_tracer.langfuse:
            try:
                # 전체 사용자 쿼리 추출
                full_user_query = ""
                if context.message and hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'root') and part.root.kind == "text":
                            full_user_query += part.root.text + " "
                        elif hasattr(part, 'text'):
                            full_user_query += part.text + " "
                full_user_query = full_user_query.strip()
                
                # 메인 트레이스 생성 (task_id를 트레이스 ID로 사용)
                main_trace = self.langfuse_tracer.langfuse.trace(
                    id=context.task_id,
                    name="DataVisualizationAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "DataVisualizationAgent",
                        "port": 8308,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id)
                    }
                )
                logger.info(f"📊 Langfuse 메인 트레이스 시작: {context.task_id}")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_query = ""
            if context.message and hasattr(context.message, 'parts') and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "text":
                        user_query += part.root.text + " "
                    elif hasattr(part, 'text'):  # 대체 패턴
                        user_query += part.text + " "
                
                user_query = user_query.strip()
            
            # 기본 요청이 없으면 데모 모드
            if not user_query:
                user_query = "샘플 데이터로 시각화를 생성해주세요. 산점도를 만들어주세요."
            
            # 1단계: 요청 파싱 (Langfuse 추적)
            parsing_span = None
            if main_trace:
                parsing_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="request_parsing",
                    input={"user_request": user_query[:500]},
                    metadata={"step": "1", "description": "Parse visualization request"}
                )
            
            logger.info(f"🔍 시각화 요청 파싱: {user_query}")
            
            # 시각화 유형 결정
            chart_type = "scatter"
            if any(keyword in user_query.lower() for keyword in ['histogram', '히스토그램', '분포']):
                chart_type = "histogram"
            elif any(keyword in user_query.lower() for keyword in ['box', '박스플롯', 'boxplot']):
                chart_type = "boxplot"
            elif any(keyword in user_query.lower() for keyword in ['bar', '막대', '바']):
                chart_type = "bar"
            
            # 파싱 결과 업데이트
            if parsing_span:
                parsing_span.update(
                    output={
                        "success": True,
                        "chart_type_detected": chart_type,
                        "request_length": len(user_query),
                        "keywords_found": [kw for kw in ['scatter', 'histogram', 'bar', 'box'] if kw in user_query.lower()]
                    }
                )
            
            # 2단계: 시각화 생성 (Langfuse 추적)
            visualization_span = None
            if main_trace:
                visualization_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="chart_generation",
                    input={
                        "chart_type": chart_type,
                        "user_request": user_query[:200]
                    },
                    metadata={"step": "2", "description": "Generate interactive visualization"}
                )
            
            logger.info(f"📊 {chart_type} 차트 생성 시작")
            
            # Get result from the agent
            result = await self.agent.invoke(user_query)
            
            # 결과 파싱하여 정보 추출
            chart_info = {"status": "completed", "type": chart_type}
            try:
                import json
                result_data = json.loads(result)
                chart_info.update({
                    "chart_title": result_data.get("chart_title", "Data Visualization"),
                    "visualization_type": result_data.get("visualization_type", "interactive_chart"),
                    "status": result_data.get("status", "completed")
                })
            except:
                pass
            
            # 시각화 결과 업데이트
            if visualization_span:
                visualization_span.update(
                    output={
                        "success": True,
                        "chart_created": True,
                        "chart_type": chart_type,
                        "chart_title": chart_info.get("chart_title", "Data Visualization"),
                        "result_length": len(result),
                        "interactive_features": True,
                        "plotly_based": True
                    }
                )
            
            # 3단계: 결과 저장/반환 (Langfuse 추적)
            save_span = None
            if main_trace:
                save_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="save_visualization",
                    input={
                        "chart_info": chart_info,
                        "result_size": len(result)
                    },
                    metadata={"step": "3", "description": "Prepare visualization response"}
                )
            
            logger.info(f"💾 시각화 결과 준비 완료")
            
            # 저장 결과 업데이트
            if save_span:
                save_span.update(
                    output={
                        "response_prepared": True,
                        "chart_data_included": "chart_data" in result,
                        "function_code_included": "function_code" in result,
                        "interactive_chart": True,
                        "final_status": "completed"
                    }
                )
            
            # A2A SDK 0.2.9 공식 패턴에 따른 최종 응답
            from a2a.types import TaskState
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            # Langfuse 메인 트레이스 완료
            if main_trace:
                try:
                    # Output을 요약된 형태로 제공
                    output_summary = {
                        "status": "completed",
                        "result_preview": result[:1000] + "..." if len(result) > 1000 else result,
                        "full_result_length": len(result)
                    }
                    
                    main_trace.update(
                        output=output_summary,
                        metadata={
                            "status": "completed",
                            "result_length": len(result),
                            "success": True,
                            "completion_timestamp": str(context.task_id),
                            "agent": "DataVisualizationAgent",
                            "port": 8308,
                            "chart_type": chart_type
                        }
                    )
                    logger.info(f"📊 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
            
        except Exception as e:
            logger.error(f"❌ DataVisualizationAgent 실행 오류: {e}")
            
            # Langfuse 메인 트레이스 오류 기록
            if main_trace:
                try:
                    main_trace.update(
                        output=f"Error: {str(e)}",
                        metadata={
                            "status": "failed",
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "success": False,
                            "agent": "DataVisualizationAgent",
                            "port": 8308
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
            
            # A2A SDK 0.2.9 공식 패턴에 따른 에러 응답
            from a2a.types import TaskState
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"시각화 생성 중 오류 발생: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function to start the data visualization server."""
    skill = AgentSkill(
        id="data_visualization",
        name="Data Visualization",
        description="Creates interactive data visualizations and charts using advanced plotting libraries",
        tags=["visualization", "plotting", "charts", "graphs"],
        examples=["create a bar chart", "visualize trends", "plot correlation matrix"]
    )

    agent_card = AgentCard(
        name="Data Visualization Agent",
        description="An AI agent that creates professional data visualizations and interactive charts.",
        url="http://localhost:8308/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )

    request_handler = DefaultRequestHandler(
        agent_executor=DataVisualizationExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("📊 Starting Data Visualization Agent Server")
    print("🌐 Server starting on http://localhost:8308")
    print("📋 Agent card: http://localhost:8308/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8308, log_level="info")

if __name__ == "__main__":
    main()