#!/usr/bin/env python3
"""
Plotly Enhanced Visualization Server - A2A Compatible 
🎯 원본 ai-data-science-team의 plotly 패턴 + LLM 동적 코드 생성 + A2A 프로토콜
포트: 8319 (Plotly Enhanced)
"""

import logging
import uvicorn
import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
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
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState, TextPart
from a2a.utils import new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotlyEnhancedDataVisualizationAgent:
    """Plotly Enhanced Data Visualization Agent - 원본 ai-data-science-team 패턴"""

    def __init__(self):
        # LLM 초기화
        self._setup_llm()
        # Plotly 라이브러리 초기화
        self._setup_plotly_libraries()
        logger.info("✅ Plotly Enhanced Data Visualization Agent initialized")
        
    def _setup_llm(self):
        """LLM 초기화 (원본 패턴)"""
        try:
            from core.llm_factory import create_llm_instance
            self.llm = create_llm_instance()
            logger.info("✅ LLM initialized for dynamic code generation")
        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            # Fallback to predefined templates
            self.llm = None
            logger.warning("⚠️ LLM 없이 템플릿 기반으로 동작")
        
    def _setup_plotly_libraries(self):
        """Plotly 라이브러리 초기화"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            import pandas as pd
            import numpy as np
            
            self.px = px
            self.go = go
            self.pio = pio
            self.pd = pd
            self.np = np
            
            # 아티팩트 디렉토리 생성
            self.artifacts_dir = Path("artifacts/plots")
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Plotly 라이브러리 초기화 완료")
        except Exception as e:
            logger.error(f"❌ Plotly 라이브러리 초기화 실패: {e}")
            raise
            
    async def create_plotly_visualization(self, user_query: str) -> dict:
        """Plotly 시각화 생성 (원본 ai-data-science-team 패턴)"""
        # CSV 데이터 추출
        csv_data = self._extract_csv_from_query(user_query)
        
        if csv_data is None:
            raise ValueError("시각화할 데이터가 없습니다")
            
        # DataFrame 생성
        df = self.pd.read_csv(self.pd.io.common.StringIO(csv_data))
        
        # LLM으로 동적 코드 생성 또는 템플릿 사용
        if self.llm:
            plotly_code = await self._generate_plotly_code_with_llm(user_query, df)
        else:
            plotly_code = self._generate_plotly_code_template(user_query, df)
        
        # 코드 실행
        plotly_dict = self._execute_plotly_code(plotly_code, df)
        
        # HTML 파일 저장 (인터랙티브)
        html_path = self._save_plotly_html(plotly_dict, user_query)
        
        return {
            'dataframe': df,
            'plotly_dict': plotly_dict,
            'plotly_code': plotly_code,
            'html_path': html_path,
            'data_summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'numeric_columns': df.select_dtypes(include=[self.np.number]).columns.tolist()
            }
        }
        
    def _extract_csv_from_query(self, query: str) -> str:
        """쿼리에서 CSV 데이터 추출"""
        lines = query.split('\n')
        csv_lines = []
        in_csv = False
        
        for line in lines:
            if ',' in line and ('=' not in line or line.count(',') > line.count('=')):
                csv_lines.append(line.strip())
                in_csv = True
            elif in_csv and line.strip() == '':
                break
                
        if len(csv_lines) >= 2:  # 헤더 + 최소 1행
            return '\n'.join(csv_lines)
        return None
        
    async def _generate_plotly_code_with_llm(self, user_query: str, df) -> str:
        """LLM으로 동적 Plotly 코드 생성 (원본 패턴)"""
        from langchain.prompts import PromptTemplate
        
        prompt_template = PromptTemplate(
            template="""
            You are a chart generator agent that is an expert in generating plotly charts. You must use plotly or plotly.express to produce plots.
    
            Your job is to produce python code to generate visualizations with a function named create_plotly_chart.
            
            USER INSTRUCTIONS: 
            {user_instructions}
            
            DATA SUMMARY: 
            Columns: {columns}
            Shape: {shape}
            Sample Data:
            {sample_data}
            
            RETURN:
            
            Return Python code in ```python ``` format with a single function definition, create_plotly_chart(data_raw), that includes all imports inside the function.
            
            Return the plotly chart as a dictionary.
            
            def create_plotly_chart(data_raw):
                import pandas as pd
                import numpy as np
                import json
                import plotly.express as px
                import plotly.graph_objects as go
                import plotly.io as pio
                
                # Your chart generation code here
                
                fig_json = pio.to_json(fig)
                fig_dict = json.loads(fig_json)
                
                return fig_dict
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            """,
            input_variables=["user_instructions", "columns", "shape", "sample_data"],
        )

        try:
            from core.parsers.python_output_parser import PythonOutputParser
            
            chain = prompt_template | self.llm | PythonOutputParser()
            
            response = await chain.ainvoke({
                "user_instructions": user_query,
                "columns": ', '.join(df.columns.tolist()),
                "shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
                "sample_data": df.head().to_string()
            })
            
            return response
        except Exception as e:
            logger.error(f"❌ LLM 코드 생성 실패: {e}")
            # Fallback to template
            return self._generate_plotly_code_template(user_query, df)
        
    def _generate_plotly_code_template(self, user_query: str, df) -> str:
        """템플릿 기반 Plotly 코드 생성"""
        # 차트 타입 결정
        chart_type = self._determine_chart_type(user_query, df)
        
        if chart_type == 'bar':
            return self._get_bar_chart_template()
        elif chart_type == 'line':
            return self._get_line_chart_template()
        elif chart_type == 'scatter':
            return self._get_scatter_chart_template()
        elif chart_type == 'pie':
            return self._get_pie_chart_template()
        elif chart_type == 'histogram':
            return self._get_histogram_template()
        else:
            return self._get_default_chart_template()
            
    def _determine_chart_type(self, query: str, df) -> str:
        """쿼리와 데이터를 기반으로 차트 타입 결정"""
        query_lower = query.lower()
        
        if 'bar' in query_lower or '막대' in query_lower:
            return 'bar'
        elif 'line' in query_lower or '선' in query_lower or 'trend' in query_lower:
            return 'line'
        elif 'scatter' in query_lower or '산점도' in query_lower:
            return 'scatter'
        elif 'pie' in query_lower or '파이' in query_lower:
            return 'pie'
        elif 'hist' in query_lower or '히스토그램' in query_lower:
            return 'histogram'
        else:
            # 데이터 기반 자동 선택
            numeric_cols = df.select_dtypes(include=[self.np.number]).columns
            if len(numeric_cols) >= 2:
                return 'scatter'
            elif len(numeric_cols) == 1:
                return 'histogram'
            else:
                return 'bar'
                
    def _get_bar_chart_template(self) -> str:
        """막대 차트 템플릿"""
        return """
def create_plotly_chart(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.express as px
    import plotly.io as pio
    
    # 첫 번째와 두 번째 컬럼 사용
    x_col = data_raw.columns[0]
    y_col = data_raw.columns[1] if len(data_raw.columns) > 1 else data_raw.columns[0]
    
    fig = px.bar(data_raw, x=x_col, y=y_col, 
                 title=f'Bar Chart: {x_col} vs {y_col}',
                 color_discrete_sequence=['#3381ff'])
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=13.2,
        font=dict(size=8.8),
        hoverlabel=dict(font_size=8.8)
    )
    
    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)
    
    return fig_dict
"""
    
    def _get_scatter_chart_template(self) -> str:
        """산점도 템플릿"""
        return """
def create_plotly_chart(data_raw):
    import pandas as pd
    import numpy as np
    import json
    import plotly.express as px
    import plotly.io as pio
    
    numeric_cols = data_raw.select_dtypes(include=[np.number]).columns
    x_col = numeric_cols[0] if len(numeric_cols) > 0 else data_raw.columns[0]
    y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    
    fig = px.scatter(data_raw, x=x_col, y=y_col,
                     title=f'Scatter Plot: {x_col} vs {y_col}',
                     color_discrete_sequence=['#3381ff'])
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=13.2,
        font=dict(size=8.8),
        hoverlabel=dict(font_size=8.8)
    )
    
    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)
    
    return fig_dict
"""

    def _get_default_chart_template(self) -> str:
        """기본 차트 템플릿"""
        return self._get_bar_chart_template()
        
    def _execute_plotly_code(self, plotly_code: str, df) -> dict:
        """Plotly 코드 실행"""
        try:
            # 코드 실행을 위한 namespace 준비
            namespace = {'data_raw': df}
            
            # 코드 실행
            exec(plotly_code, namespace)
            
            # 함수 호출
            if 'create_plotly_chart' in namespace:
                result = namespace['create_plotly_chart'](df)
                return result
            else:
                raise ValueError("create_plotly_chart 함수가 생성되지 않았습니다")
                
        except Exception as e:
            logger.error(f"❌ Plotly 코드 실행 실패: {e}")
            # 기본 차트 생성
            return self._create_fallback_chart(df)
            
    def _create_fallback_chart(self, df) -> dict:
        """폴백 차트 생성"""
        try:
            if len(df.columns) >= 2:
                fig = self.px.bar(df, x=df.columns[0], y=df.columns[1], 
                                  title="데이터 시각화")
            else:
                fig = self.px.histogram(df, x=df.columns[0], 
                                       title="데이터 분포")
            
            fig_json = self.pio.to_json(fig)
            return json.loads(fig_json)
            
        except Exception as e:
            logger.error(f"❌ 폴백 차트 생성 실패: {e}")
            return {"error": str(e)}
    
    def _save_plotly_html(self, plotly_dict: dict, user_query: str) -> str:
        """Plotly 차트를 HTML 파일로 저장"""
        try:
            # 고유 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_id = str(uuid.uuid4())[:8]
            filename = f"plotly_chart_{timestamp}_{chart_id}.html"
            html_path = self.artifacts_dir / filename
            
            # Plotly 딕셔너리를 Figure 객체로 변환
            fig = self.pio.from_json(json.dumps(plotly_dict))
            
            # HTML로 저장
            fig.write_html(html_path)
            
            return str(html_path)
            
        except Exception as e:
            logger.error(f"❌ HTML 저장 실패: {e}")
            return f"HTML 저장 실패: {str(e)}"
    
    def generate_plotly_response(self, viz_result: dict, user_query: str) -> str:
        """Plotly 시각화 결과 응답 생성"""
        df = viz_result['dataframe']
        
        return f"""# 🎨 **Plotly Interactive Visualization Complete!**

## 📊 **시각화 결과**
- **데이터**: {viz_result['data_summary']['rows']}행 × {viz_result['data_summary']['columns']}열
- **차트 엔진**: Plotly (인터랙티브)
- **컬럼**: {', '.join(viz_result['data_summary']['column_names'])}
- **숫자형 컬럼**: {', '.join(viz_result['data_summary']['numeric_columns'])}

## 🔍 **데이터 미리보기**
```
{df.head().to_string()}
```

## 📈 **기본 통계**
```
{df.describe().to_string() if len(viz_result['data_summary']['numeric_columns']) > 0 else "숫자형 데이터 없음"}
```

## 🌐 **인터랙티브 차트**
**HTML 파일**: `{viz_result['html_path']}`
**특징**: 줌, 팬, 호버 툴팁, 범례 클릭 등 인터랙티브 기능

## 💻 **생성된 Plotly 코드**
```python
{viz_result['plotly_code']}
```

## 📊 **Plotly JSON 데이터**
**크기**: {len(str(viz_result['plotly_dict']))} 바이트

---
**💬 사용자 요청**: {user_query}
**🎯 엔진**: Plotly Express/Graph Objects
**🕒 생성 시간**: {self.pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**🌟 장점**: 웹 친화적, 인터랙티브, 확대/축소 가능
"""

    def generate_no_data_response(self, user_query: str) -> str:
        """데이터 없음 응답 생성"""
        return f"""# ❌ **시각화할 데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**:
   ```
   product,sales,profit
   A,100,20
   B,150,30
   C,120,25
   ```

2. **지원하는 Plotly 차트 타입**:
   - Bar Chart (막대 차트) - 인터랙티브
   - Line Chart (선 그래프) - 확대/축소 가능
   - Scatter Plot (산점도) - 호버 툴팁
   - Pie Chart (파이 차트) - 범례 클릭
   - Histogram (히스토그램) - 구간 조정

3. **LLM 동적 생성**:
   - 사용자 요청에 맞는 최적 차트 자동 생성
   - 스타일링과 레이아웃 자동 최적화

**요청**: {user_query}
**💡 팁**: 데이터와 함께 원하는 차트 타입을 명시하면 더 정확한 시각화를 생성합니다!
"""

class PlotlyVisualizationExecutor(AgentExecutor):
    """A2A Executor - Plotly Enhanced 시각화 기능"""

    def __init__(self):
        self.agent = PlotlyEnhancedDataVisualizationAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 표준 패턴으로 실행"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🎨 Plotly 인터랙티브 시각화를 시작합니다...")
            )
            
            user_query = context.get_user_input()
            logger.info(f"📥 Processing Plotly visualization query: {user_query}")
            
            if not user_query:
                user_query = "Create an interactive visualization of the data"
            
            # Plotly 시각화 기능 구현
            try:
                visualization_result = await self.agent.create_plotly_visualization(user_query)
                result = self.agent.generate_plotly_response(visualization_result, user_query)
                logger.info(f"✅ Plotly 시각화 생성 완료: {visualization_result['html_path']}")
            except Exception as viz_error:
                logger.error(f"❌ Plotly 시각화 생성 실패: {viz_error}", exc_info=True)
                result = self.agent.generate_no_data_response(user_query)
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Plotly visualization failed: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the operation."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"Operation cancelled for context {context.context_id}")

def main():
    """Main function"""
    skill = AgentSkill(
        id="plotly_enhanced_visualization",
        name="Plotly Enhanced Data Visualization",
        description="Creates interactive data visualizations using Plotly with LLM-generated code",
        tags=["plotly", "interactive", "visualization", "charts", "llm-generated"],
        examples=[
            "다음 데이터로 인터랙티브 막대 차트를 만들어주세요: name,sales\\nA,100\\nB,150",
            "산점도를 만들어서 확대/축소가 가능하게 해주세요",
            "파이 차트를 만들어서 범례를 클릭할 수 있게 해주세요",
            "시계열 데이터로 선 그래프를 만들어주세요"
        ]
    )

    card = AgentCard(
        name="Plotly Enhanced Data Visualization Agent",
        description="Creates interactive data visualizations using Plotly with LLM-generated code",
        url="http://localhost:8323/",
        version="2.0.0-plotly",
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )

    # A2A 서버 설정
    app = A2AStarletteApplication(
        agent_cards=[card],
        agents=[PlotlyVisualizationExecutor()],
        task_store=InMemoryTaskStore(),
        request_handler=DefaultRequestHandler()
    )

    print(f"🎨 Starting Plotly Enhanced Data Visualization Agent Server")
    print(f"🌐 Server starting on http://localhost:8323")
    print(f"📋 Agent card: http://localhost:8323/.well-known/agent.json")
    print(f"🛠️ Features: Plotly, 인터랙티브, LLM 동적 생성, HTML 출력")

    uvicorn.run(app, host="0.0.0.0", port=8323)

if __name__ == "__main__":
    main() 