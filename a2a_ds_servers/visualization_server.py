#!/usr/bin/env python3
"""
Enhanced Visualization Server - A2A Compatible 
🎯 실제 matplotlib/seaborn 시각화 기능 구현 (성공 패턴 적용)
포트: 8318
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
from datetime import datetime

# 프로젝트 루트 경로 추가 (성공 패턴)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# A2A imports (성공 패턴 순서)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn

# AI_DS_Team imports (성공 패턴)
try:
    from ai_data_science_team.agents import DataVisualizationAgent
except ImportError:
    logger.warning("DataVisualizationAgent를 찾을 수 없습니다. 기본 에이전트를 사용합니다.")
    DataVisualizationAgent = None

# Core imports (성공 패턴)
from core.data_manager import DataManager
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 전역 인스턴스 (성공 패턴)
data_manager = DataManager()

class PandasAIDataProcessor:
    """pandas-ai 패턴을 활용한 데이터 처리기 (성공 패턴)"""
    
    def __init__(self):
        self.current_dataframe = None
        self.pandasai_df = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱 (성공 패턴)"""
        logger.info("📊 pandas-ai 패턴으로 메시지에서 데이터 파싱...")
        
        # 1. CSV 데이터 파싱 (성공 패턴)
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
        
        # 2. JSON 데이터 파싱 (성공 패턴)
        try:
            json_start = user_message.find('{')
            json_end = user_message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = user_message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    logger.info(f"✅ JSON 리스트 데이터 파싱 성공: {df.shape}")
                    return df
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    logger.info(f"✅ JSON 객체 데이터 파싱 성공: {df.shape}")
                    return df
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        # 3. 샘플 데이터 요청 감지 (성공 패턴)
        if any(keyword in user_message.lower() for keyword in ["샘플", "sample", "테스트", "test"]):
            logger.info("📊 샘플 데이터 생성")
            return self._generate_sample_data()
        
        return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """사용자 요청에 의한 샘플 데이터 생성 (LLM First 원칙)"""
        logger.info("🔧 사용자 요청으로 샘플 데이터 생성...")
        
        # LLM First 원칙: 하드코딩 대신 동적 생성
        try:
            # 간단한 예시 데이터 (최소한의 구조만)
            df = pd.DataFrame({
                'category': ['A', 'B', 'C', 'D', 'E'],
                'value': np.random.randint(10, 100, 5)
            })
            return df
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            return pd.DataFrame()
    
    def validate_and_process_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증 (성공 패턴)"""
        if df is None or df.empty:
            return False
        
        logger.info(f"📊 데이터 검증: {df.shape} (행 x 열)")
        logger.info(f"🔍 컬럼: {list(df.columns)}")
        logger.info(f"📈 타입: {df.dtypes.to_dict()}")
        
        return True

class EnhancedDataVisualizationAgent:
    """Enhanced Data Visualization Agent - 실제 matplotlib/seaborn 구현"""

    def __init__(self):
        # 시각화 라이브러리 초기화
        self._setup_visualization_libraries()
        logger.info("✅ Enhanced Data Visualization Agent initialized")
        
    def _setup_visualization_libraries(self):
        """시각화 라이브러리 초기화"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 백엔드 설정
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.plt = plt
            self.sns = sns
            
            # 스타일 설정
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 아티팩트 디렉토리 생성
            self.artifacts_dir = Path("artifacts/plots")
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ 시각화 라이브러리 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 시각화 라이브러리 초기화 실패: {e}")
            raise
            
    async def create_visualization(self, df: pd.DataFrame, user_query: str) -> dict:
        """실제 시각화 생성 (성공 패턴)"""
        try:
            # 차트 타입 결정
            chart_type = self._determine_chart_type(user_query, df)
            
            # 시각화 생성
            chart_path = self._generate_chart(df, chart_type, user_query)
            
            return {
                'dataframe': df,
                'chart_type': chart_type,
                'chart_path': chart_path,
                'data_summary': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
                }
            }
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            raise
        
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return 'scatter'
            elif len(numeric_cols) == 1:
                return 'histogram'
            else:
                return 'bar'
                
    def _generate_chart(self, df, chart_type: str, query: str) -> str:
        """실제 차트 생성 및 저장"""
        # 고유 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_id = str(uuid.uuid4())[:8]
        filename = f"chart_{chart_type}_{timestamp}_{chart_id}.png"
        chart_path = self.artifacts_dir / filename
        
        # 차트 생성
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if chart_type == 'bar':
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1] if df.columns[1] in numeric_cols else (numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1])
                df.plot(x=x_col, y=y_col, kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'Bar Chart: {x_col} vs {y_col}')
                
        elif chart_type == 'line':
            if len(numeric_cols) >= 1:
                df[numeric_cols].plot(kind='line', ax=ax)
                ax.set_title('Line Chart')
                
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7)
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                
        elif chart_type == 'pie':
            if len(df.columns) >= 2:
                value_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
                label_col = df.columns[0]
                ax.pie(df[value_col], labels=df[label_col], autopct='%1.1f%%')
                ax.set_title('Pie Chart')
                
        elif chart_type == 'histogram':
            if len(numeric_cols) >= 1:
                df[numeric_cols[0]].hist(ax=ax, bins=10, alpha=0.7)
                ax.set_title(f'Histogram: {numeric_cols[0]}')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel('Frequency')
        
        self.plt.tight_layout()
        self.plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        self.plt.close()
        
        return str(chart_path)

class DataVisualizationExecutor(AgentExecutor):
    """Data Visualization A2A Executor (성공 패턴)"""
    
    def __init__(self):
        # 성공 패턴: 데이터 프로세서와 에이전트 초기화
        self.data_processor = PandasAIDataProcessor()
        self.agent = EnhancedDataVisualizationAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 (성공 패턴)"""
        # 성공 패턴: TaskUpdater 올바른 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 성공 패턴: 작업 시작 (Data Cleaning Server 패턴)
            await task_updater.submit()
            await task_updater.start_work()
            
            # 성공 패턴: 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Data Visualization 작업을 시작합니다...")
            )
            
            # 성공 패턴: 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            logger.info(f"📥 Processing visualization query: {user_message}")
            
            # 성공 패턴: 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # 성공 패턴: 실제 처리 로직
                result = await self._process_with_agent(df, user_message)
            else:
                # 성공 패턴: 데이터 없음 응답
                result = self._generate_no_data_response(user_message)
            
            # 성공 패턴: 성공 완료 (new_agent_text_message 래핑)
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
        except Exception as e:
            # 성공 패턴: 오류 처리
            logger.error(f"Data Visualization 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"처리 중 오류 발생: {str(e)}")
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """시각화 처리 (성공 패턴)"""
        try:
            # 성공 패턴: 에이전트 호출
            viz_result = await self.agent.create_visualization(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if viz_result and 'chart_path' in viz_result:
                return self._generate_response(viz_result, user_instructions)
            else:
                return self._generate_fallback_response(user_instructions)
                
        except Exception as e:
            # 성공 패턴: 폴백 메커니즘
            logger.warning(f"시각화 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
    
    def _generate_response(self, viz_result: dict, user_instructions: str) -> str:
        """시각화 결과 응답 생성 (성공 패턴)"""
        df = viz_result['dataframe']
        chart_type = viz_result['chart_type']
        chart_path = viz_result['chart_path']
        
        return f"""# 🎨 **Data Visualization Complete!**

## 📊 시각화 결과

**차트 타입**: {chart_type.title()}
**데이터 크기**: {len(df)} 행 x {len(df)} 열
**생성된 파일**: {chart_path}

## 📈 데이터 요약
- **컬럼**: {', '.join(df.columns.tolist())}
- **수치형 컬럼**: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}

## 🎯 요청 내용
{user_instructions}

시각화가 성공적으로 완료되었습니다! 📊
"""
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 (성공 패턴)"""
        return f"""# ❌ **시각화할 데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**
2. **JSON 형태로 데이터 포함**  
3. **샘플 데이터 요청**: "샘플 데이터로 시각화해주세요"

**요청**: {user_instructions}
"""
    
    def _generate_fallback_response(self, user_instructions: str) -> str:
        """폴백 응답 (성공 패턴)"""
        return f"""# ⚠️ **시각화 처리 중 일시적 문제가 발생했습니다**

**요청**: {user_instructions}

**해결 방법**:
1. **다시 시도해주세요**
2. **다른 데이터로 테스트해주세요**
3. **서버를 재시작해주세요**
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 (성공 패턴)"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()

def main():
    """서버 생성 및 실행 (성공 패턴)"""
    
    # 성공 패턴: AgentSkill 정의
    skill = AgentSkill(
        id="data-visualization",
        name="Data Visualization Agent",
        description="matplotlib/seaborn 기반 데이터 시각화, 차트 생성",
        tags=["visualization", "charts", "plots", "matplotlib", "seaborn"],
        examples=[
            "막대 차트를 생성해주세요",
            "산점도를 그려주세요",
            "파이 차트를 만들어주세요"
        ]
    )
    
    # 성공 패턴: Agent Card 정의
    agent_card = AgentCard(
        name="Data Visualization Agent",
        description="Enhanced Data Visualization Agent with matplotlib/seaborn",
        url="http://localhost:8318/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 성공 패턴: Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=DataVisualizationExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # 성공 패턴: A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 Starting Data Visualization Server on http://localhost:8318")
    uvicorn.run(server.build(), host="0.0.0.0", port=8318, log_level="info")

if __name__ == "__main__":
    main() 