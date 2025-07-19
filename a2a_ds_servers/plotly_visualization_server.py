import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""

AI_DS_Team DataVisualizationAgent A2A Server - 원본 100% LLM First 패턴
Port: 8318

원본 ai-data-science-team의 DataVisualizationAgent를 100% 그대로 사용하면서
성공한 A2A 에이전트들의 데이터 처리 패턴을 적용한 완전한 LLM First 구현
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

# 원본 AI_DS_Team imports - 100% 원본 패턴
from agents import DataVisualizationAgent
from tools.dataframe import get_dataframe_summary

# Core imports - 성공한 에이전트 패턴
from core.data_manager import DataManager
from core.llm_factory import create_llm_instance
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()

class PandasAIDataProcessor:
    """성공한 A2A 에이전트들의 pandas-ai 패턴을 활용한 데이터 처리기"""
    
    def __init__(self):
        self.current_dataframe = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱 - 성공한 에이전트 패턴"""
        logger.info("📊 성공한 A2A 패턴으로 메시지에서 데이터 파싱...")
        
        # CSV 데이터 파싱
        lines = user_message.split('\n')
        csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
        
        if len(csv_lines) >= 2:  # 헤더 + 데이터
            try:
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # JSON 데이터 파싱
        try:
            json_start = user_message.find('{')
            json_end = user_message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = user_message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    raise ValueError("지원되지 않는 JSON 형태")
                    
                logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        return None

class AIDataScienceVisualizationAgent:
    """원본 ai-data-science-team DataVisualizationAgent 100% 원본 패턴 사용"""

    def __init__(self):
        # 성공한 에이전트 패턴 - LLM Factory 사용
        self.llm = create_llm_instance()
        
        # 원본 ai-data-science-team DataVisualizationAgent 100% 그대로 사용
        self.agent = DataVisualizationAgent(
            model=self.llm,
            n_samples=30,  # 원본 기본값
            log=True,
            log_path="a2a_ds_servers/artifacts/logs/",
            human_in_the_loop=False,  # A2A에서는 비활성화
            bypass_recommended_steps=False,  # 원본 LLM First 워크플로우 유지
            bypass_explain_code=False
        )
        
        # 데이터 처리기 - 성공한 에이전트 패턴
        self.data_processor = PandasAIDataProcessor()
        
        logger.info("✅ 원본 ai-data-science-team DataVisualizationAgent 100% 초기화 완료")

    async def invoke(self, user_message: str) -> dict:
        """
        원본 DataVisualizationAgent 100% 패턴으로 시각화 실행
        성공한 A2A 에이전트들의 데이터 처리 결합
        """
        try:
            # 1단계: 데이터 파싱 (성공한 에이전트 패턴)
            df = self.data_processor.parse_data_from_message(user_message)
            
            # 2단계: DataManager 폴백 (성공한 에이전트 패턴)
            if df is None:
                available_data = data_manager.list_dataframes()
                if available_data:
                    selected_id = available_data[0]
                    df = data_manager.get_dataframe(selected_id)
                    logger.info(f"✅ DataManager 폴백 사용: {selected_id}")
                else:
                    raise ValueError("시각화할 데이터가 없습니다. 데이터를 포함해서 요청해주세요.")
            
            if df is None or df.empty:
                raise ValueError("유효한 데이터를 찾을 수 없습니다.")
            
            # 3단계: 원본 DataVisualizationAgent 100% 실행
            logger.info("🎨 원본 ai-data-science-team DataVisualizationAgent 실행...")
            
            # 원본 패턴 그대로: invoke_agent 호출
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_message,
                max_retries=3,
                retry_count=0
            )
            
            # 4단계: 원본 패턴으로 결과 추출
            response = self.agent.get_response()
            
            if not response:
                raise ValueError("DataVisualizationAgent가 응답을 생성하지 못했습니다.")
            
            # 5단계: 원본 결과 구조 그대로 반환
            result = {
                'dataframe': df,
                'plotly_graph': self.agent.get_plotly_graph(),
                'data_visualization_function': self.agent.get_data_visualization_function(),
                'recommended_steps': self.agent.get_recommended_visualization_steps(),
                'workflow_summary': self.agent.get_workflow_summary(),
                'log_summary': self.agent.get_log_summary(),
                'response': response
            }
            
            logger.info("✅ 원본 DataVisualizationAgent 실행 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ DataVisualizationAgent 실행 오류: {e}")
            raise

    def generate_response(self, viz_result: dict, user_instructions: str) -> str:
        """원본 ai-data-science-team 스타일 응답 생성"""
        df = viz_result['dataframe']
        plotly_graph = viz_result.get('plotly_graph')
        
        # 기본 데이터 정보
        response = f"""# 🎨 **Plotly Interactive Visualization Complete!**
*원본 ai-data-science-team DataVisualizationAgent 100% 패턴 적용*

## 📊 **시각화 결과**
- **데이터**: {len(df)}행 × {len(df.columns)}열
- **차트 엔진**: Plotly (인터랙티브 웹 친화적)
- **컬럼**: {', '.join(df.columns.tolist())}
- **숫자형 컬럼**: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}

## 🔍 **데이터 미리보기**
```
{df.head().to_string()}
```

## 📈 **기본 통계**
```
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "숫자형 데이터 없음"}
```
"""

        # 원본 워크플로우 정보 추가
        if viz_result.get('workflow_summary'):
            response += f"\n## 🔄 **원본 Agent 워크플로우**\n{viz_result['workflow_summary']}\n"
        
        # 추천 단계 추가
        if viz_result.get('recommended_steps'):
            response += f"\n## 📋 **LLM 생성 추천 단계**\n{viz_result['recommended_steps']}\n"
        
        # 생성된 함수 코드 추가
        if viz_result.get('data_visualization_function'):
            response += f"\n## 💻 **원본 LLM 생성 시각화 함수**\n```python\n{viz_result['data_visualization_function']}\n```\n"
        
        # Plotly 그래프 정보
        if plotly_graph:
            response += f"\n## 🌐 **인터랙티브 Plotly 차트**\n**특징**: 줌, 팬, 호버 툴팁, 범례 클릭 등 완전한 인터랙티브 기능\n**크기**: {len(str(plotly_graph))} 바이트\n"
        
        # 로그 정보 추가
        if viz_result.get('log_summary'):
            response += f"\n## 📝 **로그 정보**\n{viz_result['log_summary']}\n"

        response += f"""
---
**💬 사용자 요청**: {user_instructions}
**🎯 엔진**: 원본 ai-data-science-team DataVisualizationAgent (100% LLM First)
**🕒 생성 시간**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**🌟 장점**: 완전 LLM 동적 생성, 웹 친화적 인터랙티브, 범용적
"""
        return response

class PlotlyVisualizationAgentExecutor(AgentExecutor):
    """원본 100% + 성공한 A2A 패턴 결합 Executor"""
    
    def __init__(self):
        self.agent = AIDataScienceVisualizationAgent()
        logger.info("🎨 PlotlyVisualizationAgent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """원본 DataVisualizationAgent 100% + 성공한 A2A 패턴 실행"""
        logger.info(f"🚀 원본 DataVisualizationAgent 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화 (성공한 에이전트 패턴)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🎨 원본 ai-data-science-team DataVisualizationAgent 100% 패턴 시작...")
            )
            
            # 사용자 메시지 추출 (성공한 에이전트 패턴)
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                if not user_instructions:
                    await task_updater.update_status(
                        TaskState.failed,
                        message=new_agent_text_message("❌ 시각화 요청이 비어있습니다.")
                    )
                    return
                
                # 진행 상황 업데이트
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("📊 성공한 A2A 패턴으로 데이터 분석 중...")
                )
                
                # 원본 DataVisualizationAgent 100% 실행
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("🎨 원본 LLM First 패턴으로 Plotly 차트 생성 중...")
                )
                
                visualization_result = await self.agent.invoke(user_instructions)
                
                # 데이터 저장 (성공한 에이전트 패턴)
                df = visualization_result['dataframe']
                output_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/visualization_data_{context.task_id}.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"📁 데이터 저장: {output_path}")
                
                # 최종 응답 생성
                result = self.agent.generate_response(visualization_result, user_instructions)
                
                # 작업 완료
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(result)
                )
                
            else:
                await task_updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message("❌ 시각화 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ PlotlyVisualizationAgent 실행 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"시각화 생성 중 오류 발생: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 (성공한 에이전트 패턴)"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"PlotlyVisualizationAgent 작업 취소: {context.task_id}")

def main():
    """A2A 서버 생성 및 실행 - 성공한 A2A 에이전트 패턴"""
    
    # AgentSkill 정의 (성공 패턴)
    skill = AgentSkill(
        id="plotly_visualization",
        name="AI Data Science Team Plotly Visualization",
        description="원본 ai-data-science-team DataVisualizationAgent 100% LLM First 패턴 기반 전문 Plotly 시각화",
        tags=["plotly", "visualization", "interactive", "ai-data-science-team", "llm-first"],
        examples=[
            "다음 데이터로 인터랙티브 차트를 만들어주세요",
            "산점도를 그려서 상관관계를 확인해주세요", 
            "매출 데이터를 막대 차트로 시각화해주세요",
            "시계열 데이터를 선 그래프로 그려주세요"
        ]
    )
    
    # Agent Card 정의 (성공 패턴)
    agent_card = AgentCard(
        name="AI Data Science Team Plotly Visualization Agent",
        description="원본 ai-data-science-team DataVisualizationAgent 100% LLM First 패턴 기반 전문 Plotly 시각화 에이전트",
        url="http://localhost:8318/",
        version="3.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성 (성공 패턴)
    request_handler = DefaultRequestHandler(
        agent_executor=PlotlyVisualizationAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성 (성공 패턴)
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🎨 Starting AI Data Science Team Plotly Visualization Agent Server")
    print("🌐 Server starting on http://localhost:8318")
    print("📋 Agent card: http://localhost:8318/.well-known/agent.json")
    print("✨ Features: 원본 ai-data-science-team DataVisualizationAgent 100% + 성공한 A2A 패턴")
    print("🧠 Architecture: LLM First + 동적 생성 + 인터랙티브 Plotly")
    
    # 서버 실행 (성공 패턴)
    uvicorn.run(server.build(), host="0.0.0.0", port=8318, log_level="info")

if __name__ == "__main__":
    main() 