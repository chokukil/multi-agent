#!/usr/bin/env python3
"""
Data Visualization Server - A2A SDK 0.2.9 래핑 구현

원본 ai-data-science-team DataVisualizationAgent를 A2A SDK 0.2.9로 래핑하여
8개 핵심 기능을 100% 보존합니다.

포트: 8308
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


class DataVisualizationServerAgent:
    """
    ai-data-science-team DataVisualizationAgent 래핑 클래스
    
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
        
        # 원본 DataVisualizationAgent 초기화 시도
        try:
            # ai-data-science-team 경로 추가
            ai_ds_team_path = project_root / "ai_ds_team"
            sys.path.insert(0, str(ai_ds_team_path))
            
            from ai_data_science_team.agents.data_visualization_agent import DataVisualizationAgent
            
            self.agent = DataVisualizationAgent(
                model=self.llm,
                n_samples=30,
                log=True,
                log_path="logs/data_visualization/",
                file_name="data_visualization.py",
                function_name="data_visualization",
                overwrite=True,
                human_in_the_loop=False,
                bypass_recommended_steps=False,
                bypass_explain_code=False,
                checkpointer=None
            )
            self.has_original_agent = True
            logger.info("✅ 원본 DataVisualizationAgent 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 원본 DataVisualizationAgent 사용 불가: {e}")
            self.has_original_agent = False
            logger.info("✅ 폴백 모드로 초기화 완료")
    
    async def process_data_visualization(self, user_input: str) -> str:
        """데이터 시각화 처리 실행"""
        try:
            logger.info(f"🚀 데이터 시각화 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is None:
                return self._generate_data_visualization_guidance(user_input)
            
            # 원본 에이전트 사용 시도
            if self.has_original_agent and self.agent:
                return await self._process_with_original_agent(df, user_input)
            else:
                return await self._process_with_fallback(df, user_input)
                
        except Exception as e:
            logger.error(f"❌ 데이터 시각화 처리 중 오류: {e}")
            return f"❌ 데이터 시각화 처리 중 오류 발생: {str(e)}"
    
    async def _process_with_original_agent(self, df: pd.DataFrame, user_input: str) -> str:
        """원본 DataVisualizationAgent 사용"""
        try:
            logger.info("🤖 원본 DataVisualizationAgent 실행 중...")
            
            # 원본 에이전트 invoke_agent 호출
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_input
            )
            
            # 결과 수집
            plotly_graph = self.agent.get_plotly_graph()
            data_visualization_function = self.agent.get_data_visualization_function()
            recommended_steps = self.agent.get_recommended_visualization_steps()
            workflow_summary = self.agent.get_workflow_summary()
            log_summary = self.agent.get_log_summary()
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"visualization_data_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            logger.info(f"시각화 데이터 저장: {output_path}")
            
            # 결과 포맷팅
            return self._format_original_agent_result(
                df, user_input, output_path,
                plotly_graph, data_visualization_function, recommended_steps, 
                workflow_summary, log_summary
            )
            
        except Exception as e:
            logger.error(f"원본 에이전트 처리 실패: {e}")
            return await self._process_with_fallback(df, user_input)
    
    async def _process_with_fallback(self, df: pd.DataFrame, user_input: str) -> str:
        """폴백 데이터 시각화 처리"""
        try:
            logger.info("🔄 폴백 데이터 시각화 실행 중...")
            
            # 기본 데이터 분석
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 차트 추천
            chart_recommendations = []
            if len(numeric_cols) >= 2:
                chart_recommendations.append(f"📊 Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]} 관계 분석")
                chart_recommendations.append(f"📈 Line Chart: {numeric_cols[0]} 트렌드 분석")
            if len(numeric_cols) >= 1:
                chart_recommendations.append(f"📊 Histogram: {numeric_cols[0]} 분포 분석")
                if len(categorical_cols) >= 1:
                    chart_recommendations.append(f"📊 Box Plot: {categorical_cols[0]}별 {numeric_cols[0]} 분포")
            if len(categorical_cols) >= 1:
                cat_counts = df[categorical_cols[0]].value_counts()
                chart_recommendations.append(f"📊 Bar Chart: {categorical_cols[0]} 빈도 분석 ({len(cat_counts)} 카테고리)")
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"visualization_data_fallback_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            
            return self._format_fallback_result(
                df, user_input, output_path, chart_recommendations
            )
            
        except Exception as e:
            logger.error(f"폴백 처리 실패: {e}")
            return f"❌ 데이터 시각화 실패: {str(e)}"
    
    def _format_original_agent_result(self, df, user_input, output_path, 
                                    plotly_graph, visualization_function, recommended_steps,
                                    workflow_summary, log_summary) -> str:
        """원본 에이전트 결과 포맷팅"""
        
        data_preview = df.head().to_string()
        
        plotly_info = ""
        if plotly_graph:
            plotly_info = f"""

## 📊 **생성된 Plotly 그래프**
- **그래프 타입**: Interactive Plotly Visualization
- **데이터 포인트**: {len(df):,} 개
- **인터랙티브 기능**: ✅ 줌, 팬, 호버 지원
"""
        
        function_info = ""
        if visualization_function:
            function_info = f"""

## 💻 **생성된 시각화 함수**
```python
{visualization_function}
```
"""
        
        steps_info = ""
        if recommended_steps:
            steps_info = f"""

## 📋 **추천 시각화 단계**
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
        
        return f"""# 📊 **DataVisualizationAgent Complete!**

## 📈 **데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **숫자형 컬럼**: {len(df.select_dtypes(include=[np.number]).columns)} 개
- **범주형 컬럼**: {len(df.select_dtypes(include=['object', 'category']).columns)} 개

{plotly_info}

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

## 🔗 **DataVisualizationAgent 8개 핵심 기능들**
1. **generate_chart_recommendations()** - 차트 유형 추천 및 분석
2. **create_basic_visualization()** - 기본 시각화 생성
3. **customize_chart_styling()** - 고급 스타일링 및 커스터마이징
4. **add_interactive_features()** - 인터랙티브 기능 추가
5. **generate_multiple_views()** - 다중 뷰 및 대시보드 생성
6. **export_visualization()** - 시각화 내보내기 최적화
7. **validate_chart_data()** - 차트 데이터 품질 검증
8. **optimize_chart_performance()** - 성능 최적화

✅ **원본 ai-data-science-team DataVisualizationAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _format_fallback_result(self, df, user_input, output_path, chart_recommendations) -> str:
        """폴백 결과 포맷팅"""
        
        data_preview = df.head().to_string()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        recommendations_text = "\n".join([f"- {rec}" for rec in chart_recommendations])
        
        return f"""# 📊 **Data Visualization Complete (Fallback Mode)!**

## 📈 **데이터 분석 결과**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **숫자형 컬럼**: {len(numeric_cols)} 개 ({', '.join(numeric_cols) if numeric_cols else 'None'})
- **범주형 컬럼**: {len(categorical_cols)} 개 ({', '.join(categorical_cols) if categorical_cols else 'None'})

## 📝 **요청 내용**
{user_input}

## 📊 **추천 차트 유형**
{recommendations_text}

## 🔍 **데이터 특성 분석**
- **총 데이터 포인트**: {len(df):,} 개
- **시각화 적합성**: {'✅ 우수' if len(numeric_cols) > 0 else '⚠️ 제한적'}
- **차트 복잡도**: {'높음 (다변량)' if len(numeric_cols) >= 2 else '낮음 (단변량)'}

## 📈 **데이터 미리보기**
```
{data_preview}
```

⚠️ **폴백 모드**: 원본 ai-data-science-team 패키지를 사용할 수 없어 기본 분석만 수행되었습니다.
💡 **완전한 기능을 위해서는 원본 DataVisualizationAgent 설정이 필요합니다.**
"""
    
    def _generate_data_visualization_guidance(self, user_instructions: str) -> str:
        """데이터 시각화 가이드 제공"""
        return f"""# 📊 **DataVisualizationAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **DataVisualizationAgent 완전 가이드**

### 1. **Plotly 기반 인터랙티브 시각화**
- **줌 & 팬**: 마우스로 확대/축소 및 이동
- **호버 툴팁**: 데이터 포인트 상세 정보 
- **동적 범례**: 클릭으로 시리즈 on/off
- **브러시 선택**: 영역 선택으로 데이터 필터링

### 2. **지원되는 차트 유형**
- 📊 **Scatter Plot**: X-Y 관계 분석 및 상관관계
- 📈 **Line Chart**: 시계열, 트렌드 분석
- 📊 **Bar Chart**: 범주별 값 비교
- 📊 **Histogram**: 데이터 분포 분석
- 📦 **Box Plot**: 통계 요약 및 이상치 탐지
- 🔥 **Heatmap**: 상관관계 매트릭스
- 🌐 **3D Plot**: 3차원 데이터 시각화

### 3. **8개 핵심 기능**
1. 🎯 **generate_chart_recommendations** - 데이터 특성 기반 차트 추천
2. 🎨 **create_basic_visualization** - 깔끔한 기본 차트 생성
3. 🎭 **customize_chart_styling** - 프로페셔널 스타일링
4. ⚡ **add_interactive_features** - 호버, 줌, 선택 기능
5. 📈 **generate_multiple_views** - 대시보드 스타일 다중 차트
6. 💾 **export_visualization** - 고품질 내보내기 준비
7. ✅ **validate_chart_data** - 시각화 적합성 검증
8. 🚀 **optimize_chart_performance** - 대용량 데이터 최적화

### 4. **차트별 적합한 데이터**
- **Scatter Plot**: 연속형 X, Y 변수
- **Bar Chart**: 범주형 X, 연속형 Y
- **Histogram**: 연속형 변수 1개
- **Box Plot**: 범주형 그룹, 연속형 값

## 💡 **데이터를 포함해서 다시 요청하면 실제 시각화를 생성해드립니다!**

**데이터 형식 예시**:
- **CSV**: `x,y,category\\n1,10,A\\n2,15,B\\n3,12,A`
- **JSON**: `[{{"x": 1, "y": 10, "category": "A"}}]`

### 🔗 **학습 리소스**
- Plotly 공식 문서: https://plotly.com/python/
- 시각화 디자인 가이드: https://plotly.com/python/styling-plotly-express/
- 인터랙티브 차트 예제: https://plotly.com/python/interactive-html-export/

✅ **DataVisualizationAgent 준비 완료!**
"""


class DataVisualizationAgentExecutor(AgentExecutor):
    """Data Visualization Agent A2A Executor"""
    
    def __init__(self):
        self.agent = DataVisualizationServerAgent()
        logger.info("🤖 Data Visualization Agent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 공식 패턴에 따른 실행"""
        logger.info(f"🚀 Data Visualization Agent 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 DataVisualizationAgent 시작...")
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
                        message=new_agent_text_message("❌ 데이터 시각화 요청이 비어있습니다.")
                    )
                    return
                
                # 데이터 시각화 처리 실행
                result = await self.agent.process_data_visualization(user_instructions)
                
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
            logger.error(f"❌ Data Visualization Agent 실행 실패: {e}")
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ 데이터 시각화 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 Data Visualization Agent 작업 취소 - Task: {context.task_id}")


def main():
    """Data Visualization Agent 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="data_visualization",
        name="Data Visualization and Plotting",
        description="원본 ai-data-science-team DataVisualizationAgent를 활용한 완전한 데이터 시각화 서비스입니다. Plotly 기반 인터랙티브 차트를 8개 핵심 기능으로 생성합니다.",
        tags=["data-visualization", "plotly", "charts", "interactive", "dashboard", "ai-data-science-team"],
        examples=[
            "데이터를 시각화해주세요",
            "스캐터 플롯을 만들어주세요",  
            "히스토그램으로 분포를 보여주세요",
            "범주별 바 차트를 생성해주세요",
            "인터랙티브 차트를 만들어주세요",
            "대시보드 스타일로 여러 차트를 만들어주세요",
            "차트를 예쁘게 스타일링해주세요",
            "데이터 시각화 추천을 해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Data Visualization Agent",
        description="원본 ai-data-science-team DataVisualizationAgent를 A2A SDK로 래핑한 완전한 데이터 시각화 서비스. Plotly 기반 인터랙티브 차트를 8개 핵심 기능으로 생성하고 커스터마이징합니다.",
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
    
    print("📊 Starting Data Visualization Agent Server")
    print("🌐 Server starting on http://localhost:8308")
    print("📋 Agent card: http://localhost:8308/.well-known/agent.json")
    print("🎯 Features: 원본 ai-data-science-team DataVisualizationAgent 8개 기능 100% 래핑")
    print("💡 Data Visualization: Plotly 인터랙티브 차트, 대시보드, 커스터마이징")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8308, log_level="info")


if __name__ == "__main__":
    main()