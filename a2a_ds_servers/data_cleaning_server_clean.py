#!/usr/bin/env python3
"""
AI_DS_Team DataCleaningAgent A2A Server (Clean Implementation)
Port: 8316

복사된 원본 ai_data_science_team 모듈을 사용하는 깔끔한 구현
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
from typing import Union, List, Dict

# 로깅 설정 (가장 먼저)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 추가 (루트 이동 후 단순화)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

# Core imports
from core.data_manager import DataManager
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# AI_DS_Team 원본 함수 직접 포함 (상대 import 문제 해결)
def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 30,
    skip_stats: bool = False,
) -> List[str]:
    """
    원본 ai_data_science_team의 get_dataframe_summary 함수
    상대 import 문제를 피하기 위해 직접 포함
    """
    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(_summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats))

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(_summarize_dataframe(df, dataset_name, n_sample, skip_stats))

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries

def _summarize_dataframe(
    df: pd.DataFrame, 
    dataset_name: str, 
    n_sample=30, 
    skip_stats=False
) -> str:
    """Generate a summary string for a single DataFrame."""
    # 1. Convert dictionary-type cells to strings
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))
    
    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    # 3. Calculate missing value stats
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])

    # 4. Get column data types
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    # 5. Get unique value counts
    unique_counts = df.nunique()
    unique_counts_summary = "\n".join([f"{col}: {count}" for col, count in unique_counts.items()])

    # 6. Generate the summary text
    if not skip_stats:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}

        Unique Value Counts:
        {unique_counts_summary}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}

        Data Description:
        {df.describe().to_string()}

        Data Info:
        {info_text}
        """
    else:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}
        """
        
    return summary_text.strip()

# 원본 함수 사용 가능 표시
AI_DS_TEAM_AVAILABLE = True
logger.info("✅ AI DS Team 원본 함수 직접 포함 완료")

# 전역 인스턴스
data_manager = DataManager()

class DataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        self.current_dataframe = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱"""
        logger.info("📊 메시지에서 데이터 파싱...")
        
        # CSV 데이터 파싱
        df = self._parse_csv_data(user_message)
        if df is not None:
            return df
            
        # JSON 데이터 파싱
        df = self._parse_json_data(user_message)
        if df is not None:
            return df
        
        # 샘플 데이터 요청 확인
        if self._is_sample_request(user_message):
            return self._create_sample_data()
        
        return None
    
    def _parse_csv_data(self, message: str) -> pd.DataFrame:
        """CSV 형태 데이터 파싱"""
        try:
            lines = message.split('\n')
            csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
            
            if len(csv_lines) >= 2:  # 헤더 + 최소 1개 데이터 행
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"CSV 파싱 실패: {e}")
        return None
    
    def _parse_json_data(self, message: str) -> pd.DataFrame:
        """JSON 형태 데이터 파싱"""
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    return None
                    
                logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        return None
    
    def _is_sample_request(self, message: str) -> bool:
        """샘플 데이터 요청인지 확인"""
        keywords = ["샘플", "테스트", "example", "demo", "sample", "test"]
        return any(keyword in message.lower() for keyword in keywords)
    
    def _create_sample_data(self) -> pd.DataFrame:
        """샘플 데이터 생성"""
        logger.info("🔧 샘플 데이터 생성...")
        
        np.random.seed(42)  # 재현 가능한 결과
        
        data = {
            'id': range(1, 101),
            'name': [f'User_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'income': np.random.randint(20000, 150000, 100),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'score': np.random.normal(75, 15, 100)
        }
        
        # 의도적으로 결측값과 이상값 추가
        df = pd.DataFrame(data)
        
        # 결측값 추가
        missing_indices = np.random.choice(df.index, 15, replace=False)
        df.loc[missing_indices[:5], 'age'] = np.nan
        df.loc[missing_indices[5:10], 'income'] = np.nan
        df.loc[missing_indices[10:], 'category'] = np.nan
        
        # 이상값 추가
        df.loc[0, 'age'] = 200  # 이상값
        df.loc[1, 'income'] = 1000000  # 이상값
        df.loc[2, 'score'] = -50  # 이상값
        
        # 중복 행 추가
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        
        logger.info(f"✅ 샘플 데이터 생성 완료: {df.shape}")
        return df

class DataCleaner:
    """데이터 클리너"""
    
    def __init__(self):
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_report = []
    
    def clean_data(self, df: pd.DataFrame, user_instructions: str = "") -> dict:
        """데이터 클리닝 실행"""
        self.original_data = df.copy()
        self.cleaned_data = df.copy()
        self.cleaning_report = []
        
        logger.info(f"🧹 데이터 클리닝 시작: {df.shape}")
        
        # 기본 정보 수집
        original_shape = df.shape
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # 클리닝 단계 실행
        self._remove_high_missing_columns()
        self._handle_missing_values()
        self._optimize_data_types()
        self._remove_duplicates()
        
        # 이상값 처리 (사용자가 금지하지 않은 경우)
        if "outlier" not in user_instructions.lower() and "이상값" not in user_instructions:
            self._handle_outliers()
        
        # 최종 결과 계산
        final_shape = self.cleaned_data.shape
        final_memory = self.cleaned_data.memory_usage(deep=True).sum() / 1024**2  # MB
        quality_score = self._calculate_quality_score()
        
        return {
            'original_data': self.original_data,
            'cleaned_data': self.cleaned_data,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'memory_saved': original_memory - final_memory,
            'cleaning_report': self.cleaning_report,
            'quality_score': quality_score
        }
    
    def _remove_high_missing_columns(self):
        """40% 이상 결측값이 있는 컬럼 제거"""
        missing_ratios = self.cleaned_data.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > 0.4].index.tolist()
        
        if high_missing_cols:
            self.cleaned_data = self.cleaned_data.drop(columns=high_missing_cols)
            self.cleaning_report.append(f"✅ 40% 이상 결측값 컬럼 제거: {high_missing_cols}")
    
    def _handle_missing_values(self):
        """결측값 처리"""
        for col in self.cleaned_data.columns:
            missing_count = self.cleaned_data[col].isnull().sum()
            if missing_count > 0:
                if self.cleaned_data[col].dtype in ['int64', 'float64']:
                    # 숫자형: 평균값으로 대체
                    mean_val = self.cleaned_data[col].mean()
                    self.cleaned_data[col].fillna(mean_val, inplace=True)
                    self.cleaning_report.append(f"✅ '{col}' 결측값 {missing_count}개를 평균값({mean_val:.2f})으로 대체")
                else:
                    # 범주형: 최빈값으로 대체
                    mode_val = self.cleaned_data[col].mode()
                    if not mode_val.empty:
                        self.cleaned_data[col].fillna(mode_val.iloc[0], inplace=True)
                        self.cleaning_report.append(f"✅ '{col}' 결측값 {missing_count}개를 최빈값('{mode_val.iloc[0]}')으로 대체")
    
    def _optimize_data_types(self):
        """데이터 타입 최적화"""
        optimized_count = 0
        
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].dtype == 'object':
                # 문자열 정규화
                self.cleaned_data[col] = self.cleaned_data[col].astype(str).str.strip()
            elif self.cleaned_data[col].dtype == 'int64':
                # 정수형 최적화
                col_min, col_max = self.cleaned_data[col].min(), self.cleaned_data[col].max()
                if col_min >= 0 and col_max < 255:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('uint8')
                    optimized_count += 1
                elif col_min >= -128 and col_max < 127:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int8')
                    optimized_count += 1
                elif col_min >= -32768 and col_max < 32767:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int16')
                    optimized_count += 1
        
        if optimized_count > 0:
            self.cleaning_report.append(f"✅ 데이터 타입 최적화: {optimized_count}개 컬럼")
    
    def _remove_duplicates(self):
        """중복 행 제거"""
        duplicates_count = self.cleaned_data.duplicated().sum()
        if duplicates_count > 0:
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            self.cleaning_report.append(f"✅ 중복 행 {duplicates_count}개 제거")
    
    def _handle_outliers(self):
        """IQR 방법으로 이상값 처리"""
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                # 이상값을 경계값으로 클리핑
                self.cleaned_data[col] = self.cleaned_data[col].clip(lower_bound, upper_bound)
                self.cleaning_report.append(f"✅ '{col}' 이상값 {outliers_count}개 처리 (IQR 1.5배 기준)")
    
    def _calculate_quality_score(self) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        score = 100.0
        
        # 결측값 비율
        total_cells = self.cleaned_data.shape[0] * self.cleaned_data.shape[1]
        missing_ratio = self.cleaned_data.isnull().sum().sum() / total_cells if total_cells > 0 else 0
        score -= missing_ratio * 40
        
        # 중복 비율
        duplicate_ratio = self.cleaned_data.duplicated().sum() / self.cleaned_data.shape[0] if self.cleaned_data.shape[0] > 0 else 0
        score -= duplicate_ratio * 30
        
        # 데이터 타입 일관성
        if len(self.cleaned_data.columns) > 0:
            numeric_ratio = len(self.cleaned_data.select_dtypes(include=[np.number]).columns) / len(self.cleaned_data.columns)
            score += numeric_ratio * 10
        
        return max(0, min(100, score))

class DataCleaningAgentExecutor(AgentExecutor):
    """데이터 클리닝 에이전트 실행기"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_cleaner = DataCleaner()
        logger.info("🧹 Clean DataCleaningAgent 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """데이터 클리닝 실행"""
        logger.info(f"🚀 DataCleaningAgent 실행 시작 - Task: {context.task_id}")
        
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧹 복사된 원본 모듈 기반 데이터 클리닝 시작...")
            )
            
            # 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                # 데이터 파싱
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("📊 데이터 분석 중...")
                )
                
                df = self.data_processor.parse_data_from_message(user_instructions)
                
                if df is not None and not df.empty:
                    # 데이터 클리닝 실행
                    await task_updater.update_status(
                        TaskState.working,
                        message=new_agent_text_message("🧹 데이터 클리닝 실행 중...")
                    )
                    
                    cleaning_results = self.data_cleaner.clean_data(df, user_instructions)
                    
                    # 결과 저장
                    output_dir = Path("a2a_ds_servers/artifacts/cleaned_data")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"cleaned_data_{context.task_id}.csv"
                    
                    cleaning_results['cleaned_data'].to_csv(output_path, index=False)
                    
                    # 원본 모듈 사용 시 추가 정보
                    dataframe_summary = ""
                    if AI_DS_TEAM_AVAILABLE:
                        try:
                            summary_list = get_dataframe_summary(cleaning_results['cleaned_data'])
                            dataframe_summary = "\n".join(summary_list) if summary_list else ""
                        except Exception as e:
                            logger.warning(f"원본 모듈 요약 생성 실패: {e}")
                    
                    # 응답 생성
                    result = self._generate_response(cleaning_results, user_instructions, str(output_path), dataframe_summary)
                    
                else:
                    result = self._generate_no_data_response(user_instructions)
                
                # 최종 응답
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(result)
                )
            else:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message("❌ 데이터 클리닝 요청이 비어있습니다.")
                )
                
        except Exception as e:
            logger.error(f"❌ DataCleaningAgent 실행 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"데이터 클리닝 중 오류 발생: {str(e)}")
            )
    
    def _generate_response(self, results: dict, user_instructions: str, output_path: str, dataframe_summary: str = "") -> str:
        """클리닝 결과 응답 생성"""
        ai_ds_status = "✅ 원본 AI DS Team 모듈 사용" if AI_DS_TEAM_AVAILABLE else "⚠️ 폴백 모드"
        
        response = f"""# 🧹 **AI DataCleaningAgent 완료** (복사된 원본 모듈 기반)

## 📊 **클리닝 결과**
- **원본 데이터**: {results['original_shape'][0]:,}행 × {results['original_shape'][1]}열
- **정리 후**: {results['final_shape'][0]:,}행 × {results['final_shape'][1]}열
- **메모리 절약**: {results['memory_saved']:.2f} MB
- **품질 점수**: {results['quality_score']:.1f}/100
- **모듈 상태**: {ai_ds_status}

## 🔧 **수행된 작업**
{chr(10).join(f"- {report}" for report in results['cleaning_report'])}

## 🔍 **정리된 데이터 미리보기**
```
{results['cleaned_data'].head().to_string()}
```

## 📈 **데이터 통계 요약**
```
{results['cleaned_data'].describe().to_string()}
```
"""

        if dataframe_summary:
            response += f"""
## 🎯 **AI DS Team 원본 모듈 분석**
```
{dataframe_summary}
```
"""

        response += f"""
## 📁 **저장 경로**
`{output_path}`

---
**💬 사용자 요청**: {user_instructions}
**🎯 처리 방식**: 복사된 원본 ai_data_science_team 모듈 기반
**🕒 처리 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return response
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 생성"""
        return f"""# ❌ **데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**:
   ```
   name,age,income
   John,25,50000
   Jane,30,60000
   ```

2. **JSON 형태로 데이터 포함**:
   ```json
   {{"name": "John", "age": 25, "income": 50000}}
   ```

3. **샘플 데이터 요청**: "샘플 데이터로 테스트해주세요"

**요청**: {user_instructions}
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info(f"DataCleaningAgent 작업 취소: {context.task_id}")

def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="clean_data_cleaning",
        name="Clean Data Cleaning with Original Modules",
        description="복사된 원본 ai_data_science_team 모듈을 사용한 전문 데이터 클리닝 서비스",
        tags=["data-cleaning", "ai-ds-team", "preprocessing", "original-modules"],
        examples=[
            "샘플 데이터로 테스트해주세요",
            "결측값을 처리해주세요",
            "이상값 제거 없이 데이터를 정리해주세요",
            "중복 데이터를 제거하고 품질을 개선해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Clean AI DataCleaningAgent",
        description="복사된 원본 ai_data_science_team 모듈 기반 데이터 클리닝 전문가",
        url="http://localhost:8316/",
        version="3.0.0",
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
    
    print("🧹 Starting Clean AI DataCleaningAgent Server")
    print("🌐 Server starting on http://localhost:8316")
    print("📋 Agent card: http://localhost:8316/.well-known/agent.json")
    print(f"✨ Features: 복사된 원본 모듈 기반 ({AI_DS_TEAM_AVAILABLE and '원본 모듈 사용' or '폴백 모드'})")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8316, log_level="info")

if __name__ == "__main__":
    main()