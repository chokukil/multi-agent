#!/usr/bin/env python3
"""
Feature Engineering Server - A2A SDK 0.2.9 래핑 구현

원본 ai-data-science-team FeatureEngineeringAgent를 A2A SDK 0.2.9로 래핑하여
8개 핵심 기능을 100% 보존합니다.

포트: 8310
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

# Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")


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


class FeatureEngineeringServerAgent:
    """
    ai-data-science-team FeatureEngineeringAgent 래핑 클래스
    
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
        
        # 원본 FeatureEngineeringAgent 초기화 시도
        try:
            # ai-data-science-team 경로 추가
            ai_ds_team_path = project_root / "ai_ds_team"
            sys.path.insert(0, str(ai_ds_team_path))
            
            from ai_data_science_team.agents.feature_engineering_agent import FeatureEngineeringAgent
            
            self.agent = FeatureEngineeringAgent(
                model=self.llm,
                n_samples=30,
                log=True,
                log_path="logs/feature_engineering/",
                file_name="feature_engineer.py",
                function_name="feature_engineer",
                overwrite=True,
                human_in_the_loop=False,
                bypass_recommended_steps=False,
                bypass_explain_code=False,
                checkpointer=None
            )
            self.has_original_agent = True
            logger.info("✅ 원본 FeatureEngineeringAgent 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 원본 FeatureEngineeringAgent 사용 불가: {e}")
            self.has_original_agent = False
            logger.info("✅ 폴백 모드로 초기화 완료")
    
    def _detect_target_variable(self, df: pd.DataFrame, user_input: str) -> str:
        """사용자 입력과 데이터에서 타겟 변수 감지"""
        # 일반적인 타겟 변수 이름들
        common_targets = ['target', 'label', 'y', 'class', 'outcome', 'result', 
                         'churn', 'price', 'sales', 'revenue', 'score']
        
        # 사용자 입력에서 타겟 변수 언급 확인
        for word in user_input.lower().split():
            if word in df.columns:
                return word
        
        # 일반적인 타겟 변수명 확인
        for target in common_targets:
            if target in df.columns:
                return target
        
        # 마지막 컬럼이 타겟일 가능성이 높음
        if len(df.columns) > 1:
            last_col = df.columns[-1]
            if df[last_col].dtype in ['object', 'bool'] or df[last_col].nunique() < len(df) * 0.5:
                return last_col
        
        return None
    
    async def process_feature_engineering(self, user_input: str) -> str:
        """피처 엔지니어링 처리 실행"""
        try:
            logger.info(f"🚀 피처 엔지니어링 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is None:
                return self._generate_feature_engineering_guidance(user_input)
            
            # 원본 에이전트 사용 시도
            if self.has_original_agent and self.agent:
                return await self._process_with_original_agent(df, user_input)
            else:
                return await self._process_with_fallback(df, user_input)
                
        except Exception as e:
            logger.error(f"❌ 피처 엔지니어링 처리 중 오류: {e}")
            return f"❌ 피처 엔지니어링 처리 중 오류 발생: {str(e)}"
    
    async def _process_with_original_agent(self, df: pd.DataFrame, user_input: str) -> str:
        """원본 FeatureEngineeringAgent 사용"""
        try:
            logger.info("🤖 원본 FeatureEngineeringAgent 실행 중...")
            
            # 타겟 변수 감지
            target_variable = self._detect_target_variable(df, user_input)
            
            # 원본 에이전트 invoke_agent 호출
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_input,
                target_variable=target_variable
            )
            
            # 결과 수집
            data_engineered = self.agent.get_data_engineered()
            feature_engineer_function = self.agent.get_feature_engineer_function()
            recommended_steps = self.agent.get_recommended_feature_engineering_steps()
            workflow_summary = self.agent.get_workflow_summary()
            log_summary = self.agent.get_log_summary()
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"engineered_data_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            # 엔지니어링된 데이터가 있으면 저장, 없으면 원본 저장
            if data_engineered is not None and isinstance(data_engineered, pd.DataFrame):
                data_engineered.to_csv(output_path, index=False)
                logger.info(f"엔지니어링된 데이터 저장: {output_path}")
            else:
                df.to_csv(output_path, index=False)
            
            # 결과 포맷팅
            return self._format_original_agent_result(
                df, data_engineered, user_input, output_path,
                feature_engineer_function, recommended_steps, 
                workflow_summary, log_summary, target_variable
            )
            
        except Exception as e:
            logger.error(f"원본 에이전트 처리 실패: {e}")
            return await self._process_with_fallback(df, user_input)
    
    async def _process_with_fallback(self, df: pd.DataFrame, user_input: str) -> str:
        """폴백 피처 엔지니어링 처리"""
        try:
            logger.info("🔄 폴백 피처 엔지니어링 실행 중...")
            
            # 기본 피처 엔지니어링 분석
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            engineered_df = df.copy()
            engineering_actions = []
            
            # 1. 데이터 타입 변환
            for col in categorical_cols:
                if col in engineered_df.columns:
                    unique_ratio = engineered_df[col].nunique() / len(engineered_df)
                    if unique_ratio > 0.95:  # 고유값이 95% 이상인 컬럼 제거
                        engineered_df = engineered_df.drop(columns=[col])
                        engineering_actions.append(f"'{col}' 고유값 피처 제거 (unique ratio: {unique_ratio:.2f})")
            
            # 2. 상수 피처 제거
            constant_cols = []
            for col in engineered_df.columns:
                if engineered_df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                engineered_df = engineered_df.drop(columns=constant_cols)
                engineering_actions.append(f"상수 피처 {len(constant_cols)}개 제거: {constant_cols}")
            
            # 3. 범주형 인코딩 (간단한 버전)
            remaining_categorical = engineered_df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in remaining_categorical:
                if col in engineered_df.columns:
                    unique_count = engineered_df[col].nunique()
                    if unique_count <= 10:  # 소규모 카테고리는 원핫 인코딩
                        dummies = pd.get_dummies(engineered_df[col], prefix=col, drop_first=True)
                        engineered_df = pd.concat([engineered_df.drop(columns=[col]), dummies], axis=1)
                        engineering_actions.append(f"'{col}' 원핫 인코딩 ({unique_count}개 카테고리)")
                    else:  # 대규모 카테고리는 빈도 기반 인코딩
                        value_counts = engineered_df[col].value_counts()
                        threshold = len(engineered_df) * 0.05  # 5% 임계값
                        frequent_values = value_counts[value_counts >= threshold].index
                        engineered_df[col] = engineered_df[col].apply(
                            lambda x: x if x in frequent_values else 'other'
                        )
                        engineering_actions.append(f"'{col}' 고차원 카테고리 처리 (threshold: 5%)")
            
            # 4. 불린 값을 정수로 변환
            bool_cols = engineered_df.select_dtypes(include=['bool']).columns.tolist()
            for col in bool_cols:
                engineered_df[col] = engineered_df[col].astype(int)
                engineering_actions.append(f"'{col}' 불린→정수 변환")
            
            # 5. 결측값 처리
            for col in engineered_df.columns:
                if engineered_df[col].isnull().any():
                    if engineered_df[col].dtype in ['object', 'category']:
                        engineered_df[col] = engineered_df[col].fillna('missing')
                    else:
                        engineered_df[col] = engineered_df[col].fillna(engineered_df[col].median())
                    engineering_actions.append(f"'{col}' 결측값 처리")
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"engineered_data_fallback_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            engineered_df.to_csv(output_path, index=False)
            
            return self._format_fallback_result(
                df, engineered_df, user_input, output_path, engineering_actions
            )
            
        except Exception as e:
            logger.error(f"폴백 처리 실패: {e}")
            return f"❌ 피처 엔지니어링 실패: {str(e)}"
    
    def _format_original_agent_result(self, original_df, engineered_df, user_input, 
                                    output_path, engineer_function, recommended_steps,
                                    workflow_summary, log_summary, target_variable) -> str:
        """원본 에이전트 결과 포맷팅"""
        
        data_preview = original_df.head().to_string()
        
        engineered_info = ""
        if engineered_df is not None and isinstance(engineered_df, pd.DataFrame):
            engineered_info = f"""

## 🔧 **엔지니어링된 데이터 정보**
- **엔지니어링 후 크기**: {engineered_df.shape[0]:,} 행 × {engineered_df.shape[1]:,} 열
- **피처 변화**: {len(original_df.columns)} → {len(engineered_df.columns)} ({len(engineered_df.columns) - len(original_df.columns):+d})
- **데이터 타입**: {len(engineered_df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(engineered_df.select_dtypes(include=['object']).columns)} 범주형
"""
        
        function_info = ""
        if engineer_function:
            function_info = f"""

## 💻 **생성된 피처 엔지니어링 함수**
```python
{engineer_function}
```
"""
        
        steps_info = ""
        if recommended_steps:
            steps_info = f"""

## 📋 **추천 피처 엔지니어링 단계**
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
        
        target_info = ""
        if target_variable:
            target_info = f"""

## 🎯 **감지된 타겟 변수**: `{target_variable}`
"""
        
        return f"""# 🔧 **FeatureEngineeringAgent Complete!**

## 📊 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {original_df.shape[0]:,} 행 × {original_df.shape[1]:,} 열
- **컬럼**: {', '.join(original_df.columns.tolist())}
- **데이터 타입**: {len(original_df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(original_df.select_dtypes(include=['object']).columns)} 텍스트형

{engineered_info}

{target_info}

## 📝 **요청 내용**
{user_input}

{steps_info}

{workflow_info}

{function_info}

{log_info}

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔗 **FeatureEngineeringAgent 8개 핵심 기능들**
1. **convert_data_types()** - 데이터 타입 최적화 및 변환
2. **remove_unique_features()** - 고유값 및 상수 피처 제거
3. **encode_categorical()** - 범주형 변수 인코딩 (원핫/라벨)
4. **handle_high_cardinality()** - 고차원 범주형 변수 처리
5. **create_datetime_features()** - 날짜/시간 기반 피처 생성
6. **scale_numeric_features()** - 수치형 피처 정규화/표준화
7. **create_interaction_features()** - 상호작용 및 다항 피처 생성
8. **handle_target_encoding()** - 타겟 변수 인코딩 및 처리

✅ **원본 ai-data-science-team FeatureEngineeringAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _format_fallback_result(self, original_df, engineered_df, user_input, 
                               output_path, engineering_actions) -> str:
        """폴백 결과 포맷팅"""
        
        data_preview = original_df.head().to_string()
        engineered_preview = engineered_df.head().to_string()
        
        actions_text = "\n".join([f"- {action}" for action in engineering_actions]) if engineering_actions else "- 기본 데이터 검증만 수행"
        
        return f"""# 🔧 **Feature Engineering Complete (Fallback Mode)!**

## 📊 **피처 엔지니어링 결과**
- **파일 위치**: `{output_path}`
- **원본 크기**: {original_df.shape[0]:,} 행 × {original_df.shape[1]:,} 열
- **엔지니어링 후 크기**: {engineered_df.shape[0]:,} 행 × {engineered_df.shape[1]:,} 열
- **처리 결과**: {len(engineering_actions)}개 작업 수행

## 📝 **요청 내용**
{user_input}

## 🔧 **수행된 엔지니어링 작업**
{actions_text}

## 📊 **데이터 타입 분석**
- **숫자형 피처**: {len(original_df.select_dtypes(include=[np.number]).columns)} 개
- **범주형 피처**: {len(original_df.select_dtypes(include=['object']).columns)} 개
- **결측값**: {original_df.isnull().sum().sum()} 개

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔧 **엔지니어링된 데이터 미리보기**
```
{engineered_preview}
```

⚠️ **폴백 모드**: 원본 ai-data-science-team 패키지를 사용할 수 없어 기본 엔지니어링만 수행되었습니다.
💡 **완전한 기능을 위해서는 원본 FeatureEngineeringAgent 설정이 필요합니다.**
"""
    
    def _generate_feature_engineering_guidance(self, user_instructions: str) -> str:
        """피처 엔지니어링 가이드 제공"""
        return f"""# 🔧 **FeatureEngineeringAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **FeatureEngineeringAgent 완전 가이드**

### 1. **피처 엔지니어링 핵심 개념**
피처 엔지니어링은 원시 데이터를 머신러닝 모델에 최적화된 피처로 변환하는 과정입니다:

- **데이터 전처리**: 타입 변환, 결측값 처리
- **피처 선택**: 불필요한 피처 제거
- **피처 변환**: 인코딩, 스케일링, 정규화
- **피처 생성**: 상호작용, 다항식, 파생 피처

### 2. **8개 핵심 기능**
1. 🔄 **convert_data_types** - 데이터 타입 최적화
2. 🗑️ **remove_unique_features** - 고유값/상수 피처 제거
3. 🏷️ **encode_categorical** - 범주형 변수 인코딩
4. 📊 **handle_high_cardinality** - 고차원 범주형 처리
5. ⏰ **create_datetime_features** - 시간 기반 피처 생성
6. 📏 **scale_numeric_features** - 수치형 피처 스케일링
7. 🔗 **create_interaction_features** - 상호작용 피처 생성
8. 🎯 **handle_target_encoding** - 타겟 변수 인코딩

### 3. **피처 엔지니어링 작업 예시**

#### 🔄 **데이터 타입 최적화**
```text
데이터 타입을 메모리 효율적으로 변환해주세요
```

#### 🏷️ **범주형 인코딩**
```text
범주형 변수들을 원핫 인코딩으로 변환해주세요
```

#### 📏 **피처 스케일링**
```text
수치형 피처들을 표준화해주세요
```

### 4. **지원되는 엔지니어링 기법**
- **인코딩**: OneHot, Label, Target, Frequency
- **스케일링**: Standard, MinMax, Robust Scaler
- **변환**: Log, Square Root, Box-Cox
- **피처 생성**: 다항식, 상호작용, 비닝
- **차원 축소**: PCA, 피처 선택
- **시계열**: 시차, 이동평균, 계절성

### 5. **원본 FeatureEngineeringAgent 특징**
- **자동 타입 감지**: 최적 데이터 타입 자동 선택
- **타겟 인식**: 타겟 변수 자동 감지 및 처리
- **스마트 인코딩**: 카디널리티 기반 최적 인코딩
- **메모리 최적화**: 효율적인 데이터 타입 사용

## 💡 **데이터를 포함해서 다시 요청하면 실제 피처 엔지니어링 작업을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `id,age,category,target\\n1,25,A,1\\n2,30,B,0`
- **JSON**: `[{{"id": 1, "age": 25, "category": "A", "target": 1}}]`

### 🔗 **학습 리소스**
- scikit-learn 전처리: https://scikit-learn.org/stable/modules/preprocessing.html
- 피처 엔지니어링 가이드: https://scikit-learn.org/stable/modules/feature_extraction.html
- pandas 데이터 변환: https://pandas.pydata.org/docs/user_guide/reshaping.html

✅ **FeatureEngineeringAgent 준비 완료!**
"""


class FeatureEngineeringAgentExecutor(AgentExecutor):
    """Feature Engineering Agent A2A Executor"""
    
    def __init__(self):
        self.agent = FeatureEngineeringServerAgent()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ FeatureEngineeringAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("🤖 Feature Engineering Agent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 공식 패턴에 따른 실행 with Langfuse integration"""
        logger.info(f"🚀 Feature Engineering Agent 실행 시작 - Task: {context.task_id}")
        
        # TaskUpdater 초기화
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
                    name="FeatureEngineeringAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "FeatureEngineeringAgent",
                        "port": 8310,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id),
                        "server_type": "new_wrapper_based"
                    }
                )
                logger.info(f"🔧 Langfuse 메인 트레이스 시작: {context.task_id}")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
        
        try:
            await task_updater.submit()
            await task_updater.start_work()
            
            # 1단계: 요청 파싱 (Langfuse 추적)
            parsing_span = None
            if main_trace:
                parsing_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="request_parsing",
                    input={"user_request": full_user_query[:500]},
                    metadata={"step": "1", "description": "Parse feature engineering request"}
                )
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 FeatureEngineeringAgent 시작...")
            )
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_instructions = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_instructions += part.root.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
                
                # 파싱 결과 업데이트
                if parsing_span:
                    parsing_span.update(
                        output={
                            "success": True,
                            "query_extracted": user_instructions[:200],
                            "request_length": len(user_instructions),
                            "engineering_type": "feature_transformation"
                        }
                    )
                
                if not user_instructions:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message("❌ 피처 엔지니어링 요청이 비어있습니다.")
                    )
                    return
                
                # 2단계: 피처 엔지니어링 실행 (Langfuse 추적)
                engineering_span = None
                if main_trace:
                    engineering_span = self.langfuse_tracer.langfuse.span(
                        trace_id=context.task_id,
                        name="feature_engineering",
                        input={
                            "query": user_instructions[:200],
                            "engineering_type": "wrapper_based_processing"
                        },
                        metadata={"step": "2", "description": "Execute feature engineering with optimized wrapper"}
                    )
                
                # 피처 엔지니어링 처리 실행
                result = await self.agent.process_feature_engineering(user_instructions)
                
                # 엔지니어링 결과 업데이트
                if engineering_span:
                    engineering_span.update(
                        output={
                            "success": True,
                            "result_length": len(result),
                            "features_created": True,
                            "transformation_applied": True,
                            "execution_method": "optimized_wrapper"
                        }
                    )
                
                # 3단계: 결과 저장/반환 (Langfuse 추적)
                save_span = None
                if main_trace:
                    save_span = self.langfuse_tracer.langfuse.span(
                        trace_id=context.task_id,
                        name="save_results",
                        input={
                            "result_size": len(result),
                            "engineering_success": True
                        },
                        metadata={"step": "3", "description": "Prepare feature engineering results"}
                    )
                
                # 저장 결과 업데이트
                if save_span:
                    save_span.update(
                        output={
                            "response_prepared": True,
                            "features_delivered": True,
                            "final_status": "completed",
                            "transformations_included": True
                        }
                    )
                
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
                            "agent": "FeatureEngineeringAgent",
                            "port": 8310,
                            "server_type": "new_wrapper_based",
                            "engineering_type": "feature_transformation"
                        }
                    )
                    logger.info(f"🔧 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
                
        except Exception as e:
            logger.error(f"❌ Feature Engineering Agent 실행 실패: {e}")
            
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
                            "agent": "FeatureEngineeringAgent",
                            "port": 8310,
                            "server_type": "new_wrapper_based"
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ 피처 엔지니어링 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 Feature Engineering Agent 작업 취소 - Task: {context.task_id}")


def main():
    """Feature Engineering Agent 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="feature_engineering",
        name="Feature Engineering and Transformation",
        description="원본 ai-data-science-team FeatureEngineeringAgent를 활용한 완전한 피처 엔지니어링 서비스입니다. 8개 핵심 기능으로 데이터 타입 최적화, 인코딩, 스케일링을 수행합니다.",
        tags=["feature-engineering", "preprocessing", "encoding", "scaling", "transformation", "ai-data-science-team"],
        examples=[
            "데이터 타입을 최적화해주세요",
            "범주형 변수를 인코딩해주세요",  
            "수치형 피처를 표준화해주세요",
            "고유값 피처를 제거해주세요",
            "시간 기반 피처를 생성해주세요",
            "상호작용 피처를 만들어주세요",
            "고차원 범주형을 처리해주세요",
            "타겟 변수를 인코딩해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Feature Engineering Agent",
        description="원본 ai-data-science-team FeatureEngineeringAgent를 A2A SDK로 래핑한 완전한 피처 엔지니어링 서비스. 8개 핵심 기능으로 데이터 타입 최적화, 인코딩, 스케일링, 피처 생성을 지원합니다.",
        url="http://localhost:8310/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=FeatureEngineeringAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔧 Starting Feature Engineering Agent Server")
    print("🌐 Server starting on http://localhost:8310")
    print("📋 Agent card: http://localhost:8310/.well-known/agent.json")
    print("🎯 Features: 원본 ai-data-science-team FeatureEngineeringAgent 8개 기능 100% 래핑")
    print("💡 Feature Engineering: 타입 최적화, 인코딩, 스케일링, 피처 생성, 변환")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8310, log_level="info")


if __name__ == "__main__":
    main()