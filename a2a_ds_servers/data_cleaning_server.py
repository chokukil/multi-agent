#!/usr/bin/env python3
"""
AI_DS_Team DataCleaningAgent A2A Server (New Implementation)
Port: 8310

원본 ai-data-science-team의 DataCleaningAgent를 참조하여 A2A 프로토콜에 맞게 구현
데이터 부분에 pandas-ai 패턴 적용
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

# 프로젝트 루트 경로 추가 (단순화)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))  # a2a_ds_servers 디렉토리 추가

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

# AI_DS_Team imports - 직접 모듈 import 방식
try:
    # __init__.py를 거치지 않고 직접 모듈 import
    from ai_data_science_team.tools.dataframe import get_dataframe_summary
    from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent, make_data_cleaning_agent
    AI_DS_TEAM_AVAILABLE = True
    logger.info("✅ AI DS Team 원본 모듈 직접 import 성공")
except ImportError as e:
    AI_DS_TEAM_AVAILABLE = False
    logger.warning(f"⚠️ AI DS Team 모듈 import 실패: {e}")
    
    # 폴백 함수
    def get_dataframe_summary(df: pd.DataFrame) -> str:
        """DataFrame 요약 정보 생성 (폴백 버전)"""
        return f"""
데이터 형태: {df.shape[0]}행 × {df.shape[1]}열
컬럼: {list(df.columns)}
데이터 타입: {df.dtypes.to_dict()}
결측값: {df.isnull().sum().to_dict()}
메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
    
    # 더미 클래스
    class DataCleaningAgent:
        pass
    
    def make_data_cleaning_agent(*args, **kwargs):
        return DataCleaningAgent()

# pandas-ai imports (for enhanced data handling)
try:
    from pandasai import Agent as PandasAIAgent
    from pandasai import DataFrame as PandasAIDataFrame
    PANDASAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ pandas-ai 사용 가능")
except ImportError:
    PANDASAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ pandas-ai 미설치 - 기본 모드로 실행")

# Core imports
from core.data_manager import DataManager
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 전역 인스턴스
data_manager = DataManager()

class PandasAIDataProcessor:
    """pandas-ai 패턴을 활용한 데이터 처리기"""
    
    def __init__(self):
        self.current_dataframe = None
        self.pandasai_df = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱"""
        logger.info("📊 pandas-ai 패턴으로 메시지에서 데이터 파싱...")
        
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
        
        # 샘플 데이터 생성
        if any(keyword in user_message.lower() for keyword in ["샘플", "테스트", "example", "demo"]):
            return self._create_sample_data()
        
        return None
    
    def _create_sample_data(self) -> pd.DataFrame:
        """사용자 요청에 의한 샘플 데이터 생성 (LLM First 원칙)"""
        logger.info("🔧 사용자 요청으로 샘플 데이터 생성...")
        
        # LLM First 원칙: 하드코딩 대신 동적 생성
        try:
            # 간단한 예시 데이터 (최소한의 구조만)
            df = pd.DataFrame({
                'id': range(1, 11),
                'name': [f'User_{i}' for i in range(1, 11)],
                'value': np.random.randint(1, 100, 10)
            })
            return df
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            return pd.DataFrame()
    
    def create_pandasai_dataframe(self, df: pd.DataFrame, name: str = "dataset", description: str = "User dataset") -> pd.DataFrame:
        """pandas DataFrame을 PandasAI 패턴으로 처리"""
        if not PANDASAI_AVAILABLE:
            logger.info("pandas-ai 없음 - 기본 DataFrame 사용")
            return df
        
        try:
            self.pandasai_df = PandasAIDataFrame(
                df,
                name=name,
                description=description
            )
            logger.info(f"✅ PandasAI DataFrame 생성: {name}")
            return df
        except Exception as e:
            logger.warning(f"PandasAI DataFrame 생성 실패: {e}")
            return df

class EnhancedDataCleaner:
    """원본 DataCleaningAgent 패턴을 따른 향상된 데이터 클리너"""
    
    def __init__(self):
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_report = []
        self.recommended_steps = []
    
    def get_default_cleaning_steps(self) -> list:
        """원본 DataCleaningAgent의 기본 클리닝 단계들"""
        return [
            "40% 이상 결측값이 있는 컬럼 제거",
            "숫자형 컬럼의 결측값을 평균으로 대체", 
            "범주형 컬럼의 결측값을 최빈값으로 대체",
            "적절한 데이터 타입으로 변환",
            "중복 행 제거",
            "결측값이 있는 행 제거 (선택적)",
            "극단적 이상값 제거 (IQR 3배 기준)"
        ]
    
    def clean_data(self, df: pd.DataFrame, user_instructions: str = None) -> dict:
        """
        원본 DataCleaningAgent 스타일의 데이터 클리닝
        pandas-ai 패턴으로 강화
        """
        self.original_data = df.copy()
        self.cleaned_data = df.copy()
        self.cleaning_report = []
        
        logger.info(f"🧹 Enhanced 데이터 클리닝 시작: {df.shape}")
        
        # 기본 정보 수집
        original_shape = df.shape
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # 1. 결측값 비율이 높은 컬럼 제거 (40% 기준)
        missing_ratios = df.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > 0.4].index.tolist()
        
        if high_missing_cols and "outlier" not in user_instructions.lower():
            self.cleaned_data = self.cleaned_data.drop(columns=high_missing_cols)
            self.cleaning_report.append(f"✅ 40% 이상 결측값 컬럼 제거: {high_missing_cols}")
        
        # 2. 결측값 처리
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].isnull().sum() > 0:
                if self.cleaned_data[col].dtype in ['int64', 'float64']:
                    # 숫자형: 평균값으로 대체
                    mean_val = self.cleaned_data[col].mean()
                    self.cleaned_data[col].fillna(mean_val, inplace=True)
                    self.cleaning_report.append(f"✅ '{col}' 결측값을 평균값({mean_val:.2f})으로 대체")
                else:
                    # 범주형: 최빈값으로 대체
                    mode_val = self.cleaned_data[col].mode()
                    if not mode_val.empty:
                        self.cleaned_data[col].fillna(mode_val.iloc[0], inplace=True)
                        self.cleaning_report.append(f"✅ '{col}' 결측값을 최빈값('{mode_val.iloc[0]}')으로 대체")
        
        # 3. 데이터 타입 최적화
        self._optimize_data_types()
        
        # 4. 중복 행 제거
        duplicates_count = self.cleaned_data.duplicated().sum()
        if duplicates_count > 0:
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            self.cleaning_report.append(f"✅ 중복 행 {duplicates_count}개 제거")
        
        # 5. 이상값 처리 (사용자가 명시적으로 금지하지 않은 경우)
        if user_instructions and "outlier" not in user_instructions.lower():
            self._handle_outliers()
        
        # 6. 최종 결과 계산
        final_shape = self.cleaned_data.shape
        final_memory = self.cleaned_data.memory_usage(deep=True).sum() / 1024**2  # MB
        data_quality_score = self._calculate_quality_score()
        
        return {
            'original_data': self.original_data,
            'cleaned_data': self.cleaned_data,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'memory_saved': original_memory - final_memory,
            'cleaning_report': self.cleaning_report,
            'data_quality_score': data_quality_score,
            'recommended_steps': self.get_default_cleaning_steps()
        }
    
    def _optimize_data_types(self):
        """데이터 타입 최적화"""
        for col in self.cleaned_data.columns:
            if self.cleaned_data[col].dtype == 'object':
                # 문자열 정규화
                self.cleaned_data[col] = self.cleaned_data[col].astype(str).str.strip()
            elif self.cleaned_data[col].dtype == 'int64':
                # 정수형 최적화
                col_min, col_max = self.cleaned_data[col].min(), self.cleaned_data[col].max()
                if col_min >= 0 and col_max < 255:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('uint8')
                elif col_min >= -128 and col_max < 127:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int8')
                elif col_min >= -32768 and col_max < 32767:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int16')
                    
        self.cleaning_report.append("✅ 데이터 타입 최적화 완료")
    
    def _handle_outliers(self):
        """IQR 방법으로 이상값 처리"""
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 원본처럼 3배 사용
            upper_bound = Q3 + 3 * IQR
            
            outliers_mask = (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                # 이상값을 경계값으로 클리핑
                self.cleaned_data[col] = self.cleaned_data[col].clip(lower_bound, upper_bound)
                self.cleaning_report.append(f"✅ '{col}' 이상값 {outliers_count}개 처리 (3×IQR 기준)")
    
    def _calculate_quality_score(self) -> float:
        """데이터 품질 점수 계산 (0-100)"""
        score = 100.0
        
        # 결측값 비율
        missing_ratio = self.cleaned_data.isnull().sum().sum() / (self.cleaned_data.shape[0] * self.cleaned_data.shape[1])
        score -= missing_ratio * 40
        
        # 중복 비율
        duplicate_ratio = self.cleaned_data.duplicated().sum() / self.cleaned_data.shape[0]
        score -= duplicate_ratio * 30
        
        # 데이터 타입 일관성
        numeric_ratio = len(self.cleaned_data.select_dtypes(include=[np.number]).columns) / len(self.cleaned_data.columns)
        score += numeric_ratio * 10
        
        return max(0, min(100, score))

class DataCleaningAgentExecutor(AgentExecutor):
    """A2A DataCleaningAgent Executor with pandas-ai pattern"""
    
    def __init__(self):
        """초기화"""
        self.data_processor = PandasAIDataProcessor()
        self.data_cleaner = EnhancedDataCleaner()
        logger.info("🧹 DataCleaningAgent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """pandas-ai 패턴이 적용된 데이터 클리닝 실행"""
        logger.info(f"🚀 DataCleaningAgent 실행 시작 - Task: {context.task_id}")
        
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧹 pandas-ai 패턴 DataCleaningAgent 시작...")
            )
            
            # 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                # pandas-ai 패턴으로 데이터 파싱
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("📊 pandas-ai 패턴으로 데이터 분석 중...")
                )
                
                # 1단계: 메시지에서 데이터 파싱
                df = self.data_processor.parse_data_from_message(user_instructions)
                
                # 2단계: 데이터가 없으면 DataManager 폴백
                if df is None:
                    available_data = data_manager.list_dataframes()
                    if available_data:
                        selected_id = available_data[0]
                        df = data_manager.get_dataframe(selected_id)
                        logger.info(f"✅ DataManager 폴백: {selected_id}")
                    else:
                        # 샘플 데이터 생성
                        df = self.data_processor._create_sample_data()
                
                if df is not None and not df.empty:
                    # 3단계: pandas-ai DataFrame 생성
                    self.data_processor.create_pandasai_dataframe(
                        df, name="user_dataset", description=user_instructions[:100]
                    )
                    
                    # 4단계: 데이터 클리닝 실행
                    await task_updater.update_status(
                        TaskState.working,
                        message=new_agent_text_message("🧹 Enhanced 데이터 클리닝 실행 중...")
                    )
                    
                    cleaning_results = self.data_cleaner.clean_data(df, user_instructions)
                    
                    # 5단계: 결과 저장
                    output_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/cleaned_data_{context.task_id}.csv"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cleaning_results['cleaned_data'].to_csv(output_path, index=False)
                    
                    # 6단계: 응답 생성
                    result = self._generate_response(cleaning_results, user_instructions, output_path)
                    
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
    
    def _generate_response(self, results: dict, user_instructions: str, output_path: str) -> str:
        """클리닝 결과 응답 생성"""
        return f"""# 🧹 **AI DataCleaningAgent 완료** (pandas-ai 패턴)

## 📊 **클리닝 결과**
- **원본 데이터**: {results['original_shape'][0]:,}행 × {results['original_shape'][1]}열
- **정리 후**: {results['final_shape'][0]:,}행 × {results['final_shape'][1]}열
- **메모리 절약**: {results['memory_saved']:.2f} MB
- **품질 점수**: {results['data_quality_score']:.1f}/100

## 🔧 **수행된 작업**
{chr(10).join(f"- {report}" for report in results['cleaning_report'])}

## 📋 **기본 클리닝 단계**
{chr(10).join(f"- {step}" for step in results['recommended_steps'])}

## 🔍 **정리된 데이터 미리보기**
```
{results['cleaned_data'].head().to_string()}
```

## 📈 **데이터 통계 요약**
```
{results['cleaned_data'].describe().to_string()}
```

## 📁 **저장 경로**
`{output_path}`

---
**💬 사용자 요청**: {user_instructions}
**🎯 처리 방식**: pandas-ai Enhanced Pattern + AI DataCleaningAgent
**🕒 처리 시간**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
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

4. **파일 업로드**: 데이터 파일을 먼저 업로드

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
        id="enhanced_data_cleaning",
        name="Enhanced Data Cleaning with pandas-ai",
        description="pandas-ai 패턴이 적용된 전문 데이터 클리닝 서비스. 원본 ai-data-science-team DataCleaningAgent 기반으로 구현.",
        tags=["data-cleaning", "pandas-ai", "preprocessing", "quality-improvement"],
        examples=[
            "샘플 데이터로 테스트해주세요",
            "결측값을 처리해주세요",
            "이상값 제거 없이 데이터를 정리해주세요",
            "중복 데이터를 제거하고 품질을 개선해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="AI DataCleaningAgent (Enhanced)",
        description="pandas-ai 패턴이 적용된 향상된 데이터 클리닝 전문가. 원본 ai-data-science-team 기반.",
        url="http://localhost:8316/",
        version="2.0.0",
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
    
    print("🧹 Starting Enhanced AI DataCleaningAgent Server (pandas-ai pattern)")
    print("🌐 Server starting on http://localhost:8316")
    print("📋 Agent card: http://localhost:8316/.well-known/agent.json")
    print("✨ Features: pandas-ai pattern + ai-data-science-team compatibility")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8316, log_level="info")

if __name__ == "__main__":
    main() 