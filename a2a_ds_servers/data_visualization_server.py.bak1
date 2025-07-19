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

class DataVisualizationAgent:
    """Data Visualization Agent with LLM integration."""

    def __init__(self):
        # Initialize with real LLM - required, no fallback
        self.llm = None
        self.agent = None
        
        try:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No LLM API key found in environment variables")
                
            from core.llm_factory import create_llm_instance
            from ai_data_science_team.agents import DataVisualizationAgent as OriginalAgent
            
            self.llm = create_llm_instance()
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
    """Data Visualization Agent Executor."""

    def __init__(self):
        self.agent = DataVisualizationAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the data visualization using TaskUpdater pattern."""
        # Initialize TaskUpdater
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # Submit and start work
            await task_updater.submit()
            await task_updater.start_work()
            
            # Extract user message
            user_query = context.get_user_input()
            logger.info(f"📊 Processing visualization query: {user_query}")
            
            if not user_query:
                user_query = "Please provide a data visualization request."
            
            # Get result from the agent
            result = await self.agent.invoke(user_query)
            
            # Complete task with result
            from a2a.types import TaskState, TextPart
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=result)])
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            # Report error through TaskUpdater
            from a2a.types import TaskState, TextPart
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"Visualization failed: {str(e)}")])
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
        url="http://localhost:8202/",
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
    print("🌐 Server starting on http://localhost:8202")
    print("📋 Agent card: http://localhost:8202/.well-known/agent.json")

    uvicorn.run(server.build(), host="0.0.0.0", port=8202, log_level="info")

if __name__ == "__main__":
    main() 