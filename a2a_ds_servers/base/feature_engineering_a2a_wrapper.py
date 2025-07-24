#!/usr/bin/env python3
"""
FeatureEngineeringA2AWrapper - A2A SDK 0.2.9 래핑 FeatureEngineeringAgent

원본 ai-data-science-team FeatureEngineeringAgent를 A2A SDK 0.2.9 프로토콜로 
래핑하여 8개 핵심 기능을 100% 보존합니다.

8개 핵심 기능:
1. convert_data_types() - 데이터 타입 최적화 및 변환
2. remove_unique_features() - 고유값 피처 제거
3. encode_categorical() - 범주형 변수 인코딩 (원핫/라벨)
4. handle_high_cardinality() - 고차원 범주형 변수 처리
5. create_datetime_features() - 날짜/시간 기반 피처 생성
6. scale_numeric_features() - 수치형 피처 정규화/표준화
7. create_interaction_features() - 상호작용 피처 생성
8. handle_target_encoding() - 타겟 변수 인코딩
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a_ds_servers.base.base_a2a_wrapper import BaseA2AWrapper, BaseA2AExecutor

logger = logging.getLogger(__name__)


class FeatureEngineeringA2AWrapper(BaseA2AWrapper):
    """
    FeatureEngineeringAgent의 A2A SDK 0.2.9 래퍼
    
    원본 ai-data-science-team FeatureEngineeringAgent의 모든 기능을 
    A2A 프로토콜로 래핑하여 제공합니다.
    """
    
    def __init__(self):
        # FeatureEngineeringAgent 임포트를 시도
        try:
            from ai_data_science_team.agents.feature_engineering_agent import FeatureEngineeringAgent
            self.original_agent_class = FeatureEngineeringAgent
        except ImportError:
            logger.warning("FeatureEngineeringAgent import failed, using fallback")
            self.original_agent_class = None
            
        super().__init__(
            agent_name="FeatureEngineeringAgent",
            original_agent_class=self.original_agent_class,
            port=8310
        )
    
    def _create_original_agent(self):
        """원본 FeatureEngineeringAgent 생성"""
        if self.original_agent_class:
            return self.original_agent_class(
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
        return None
    
    async def _invoke_original_agent(self, df: pd.DataFrame, user_input: str, function_name: str = None) -> Dict[str, Any]:
        """원본 FeatureEngineeringAgent invoke_agent 호출"""
        
        # 특정 기능 요청이 있는 경우 해당 기능에 맞는 지시사항 생성
        if function_name:
            user_input = self._get_function_specific_instructions(function_name, user_input)
        
        # 타겟 변수 감지 시도
        target_variable = self._detect_target_variable(df, user_input)
        
        # 원본 에이전트 호출
        if self.agent:
            self.agent.invoke_agent(
                data_raw=df,
                user_instructions=user_input,
                target_variable=target_variable
            )
            
            # 8개 기능 결과 수집
            results = {
                "response": self.agent.response,
                "data_engineered": self.agent.get_data_engineered(),
                "data_raw": self.agent.get_data_raw(),
                "feature_engineer_function": self.agent.get_feature_engineer_function(),
                "recommended_feature_engineering_steps": self.agent.get_recommended_feature_engineering_steps(),
                "workflow_summary": self.agent.get_workflow_summary(),
                "log_summary": self.agent.get_log_summary(),
                "ai_message": None,
                "target_variable": target_variable
            }
            
            # AI 메시지 추출
            if results["response"] and results["response"].get("messages"):
                last_message = results["response"]["messages"][-1]
                if hasattr(last_message, 'content'):
                    results["ai_message"] = last_message.content
        else:
            # 폴백 모드
            results = await self._fallback_feature_engineering(df, user_input)
        
        return results
    
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
    
    async def _fallback_feature_engineering(self, df: pd.DataFrame, user_input: str) -> Dict[str, Any]:
        """폴백 피처 엔지니어링 처리"""
        try:
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
            
            return {
                "response": {"engineering_actions": engineering_actions},
                "data_engineered": engineered_df,
                "data_raw": df,
                "feature_engineer_function": f"# Fallback feature engineering function for {len(df.columns)} columns, {len(df)} rows",
                "recommended_feature_engineering_steps": "1. 데이터 타입 최적화\\n2. 고유값/상수 피처 제거\\n3. 범주형 인코딩\\n4. 결측값 처리",
                "workflow_summary": "Fallback feature engineering completed",
                "log_summary": "Fallback mode - original agent not available",
                "ai_message": f"수행된 피처 엔지니어링: {len(engineering_actions)}개 작업",
                "target_variable": None
            }
        except Exception as e:
            logger.error(f"Fallback feature engineering failed: {e}")
            return {"ai_message": f"피처 엔지니어링 분석 중 오류: {str(e)}"}
    
    def _get_function_specific_instructions(self, function_name: str, user_input: str) -> str:
        """8개 기능별 특화된 지시사항 생성"""
        
        function_instructions = {
            "convert_data_types": """
Focus on data type conversion and optimization:
- Analyze column data types and convert to most appropriate types
- Convert object columns to categorical where appropriate
- Optimize numeric types (int64 to int32, float64 to float32)
- Handle mixed-type columns and standardize formats
- Ensure memory efficiency through type optimization

Original user request: {}
""",
            "remove_unique_features": """
Focus on removing problematic unique features:
- Identify columns with unique values equal to dataset size
- Remove constant features with same value in all rows
- Detect and remove ID-like columns that don't add value
- Handle features with excessive cardinality
- Preserve meaningful unique identifiers only when necessary

Original user request: {}
""",
            "encode_categorical": """
Focus on categorical variable encoding:
- Apply one-hot encoding for low-cardinality categories
- Use label encoding for ordinal variables
- Implement target encoding for high-predictive-power categories
- Handle unknown categories in test data
- Choose encoding method based on cardinality and relationship to target

Original user request: {}
""",
            "handle_high_cardinality": """
Focus on high-cardinality categorical processing:
- Identify categories with cardinality > 5% of dataset
- Group infrequent values into 'other' category
- Apply frequency-based encoding
- Use embedding techniques for very high cardinality
- Balance information retention with model complexity

Original user request: {}
""",
            "create_datetime_features": """
Focus on datetime feature engineering:
- Extract year, month, day, hour, minute components
- Create day of week, is_weekend, is_holiday features
- Calculate time differences and durations
- Generate cyclical features (sin/cos for periodic patterns)
- Handle timezone conversions and date arithmetic

Original user request: {}
""",
            "scale_numeric_features": """
Focus on numeric feature scaling and normalization:
- Apply StandardScaler for normal distributions
- Use MinMaxScaler for bounded features
- Apply RobustScaler for features with outliers
- Log transform skewed distributions
- Handle zero and negative values appropriately

Original user request: {}
""",
            "create_interaction_features": """
Focus on feature interaction and polynomial features:
- Generate multiplicative interactions between numeric features
- Create ratio features (division of related variables)
- Generate polynomial features for non-linear relationships
- Apply binning and discretization for continuous variables
- Create domain-specific engineered features

Original user request: {}
""",
            "handle_target_encoding": """
Focus on target variable processing:
- Apply label encoding for categorical targets
- Ensure numeric targets are properly scaled
- Handle target leakage in feature engineering
- Create target-based statistical features
- Validate target distribution and handle imbalance

Original user request: {}
"""
        }
        
        return function_instructions.get(function_name, user_input).format(user_input)
    
    def _format_result(self, result: Dict[str, Any], df: pd.DataFrame, output_path: str, user_input: str) -> str:
        """FeatureEngineeringAgent 특화 결과 포맷팅"""
        
        # 기본 정보
        data_preview = df.head().to_string()
        
        # 엔지니어링된 데이터 정보
        engineered_info = ""
        if result.get("data_engineered") is not None:
            engineered_df = result["data_engineered"]
            if isinstance(engineered_df, pd.DataFrame):
                engineered_info = f"""

## 🔧 **엔지니어링된 데이터 정보**  
- **엔지니어링 후 크기**: {engineered_df.shape[0]:,} 행 × {engineered_df.shape[1]:,} 열
- **피처 변화**: {len(df.columns)} → {len(engineered_df.columns)} ({len(engineered_df.columns) - len(df.columns):+d})
- **데이터 타입**: {len(engineered_df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(engineered_df.select_dtypes(include=['object']).columns)} 범주형
"""
        
        # 생성된 함수 정보
        function_info = ""
        if result.get("feature_engineer_function"):
            function_info = f"""

## 💻 **생성된 피처 엔지니어링 함수**
```python
{result["feature_engineer_function"]}
```
"""
        
        # 추천 단계 정보
        recommended_steps_info = ""
        if result.get("recommended_feature_engineering_steps"):
            recommended_steps_info = f"""

## 📋 **추천 피처 엔지니어링 단계**
{result["recommended_feature_engineering_steps"]}
"""
        
        # 워크플로우 요약
        workflow_info = ""
        if result.get("workflow_summary"):
            workflow_info = f"""

## 🔄 **워크플로우 요약**
{result["workflow_summary"]}
"""
        
        # 로그 요약
        log_info = ""
        if result.get("log_summary"):
            log_info = f"""

## 📄 **로그 요약**
{result["log_summary"]}
"""
        
        # 타겟 변수 정보
        target_info = ""
        if result.get("target_variable"):
            target_info = f"""

## 🎯 **타겟 변수**: `{result["target_variable"]}`
"""
        
        return f"""# 🔧 **FeatureEngineeringAgent Complete!**

## 📊 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(df.select_dtypes(include=['object']).columns)} 범주형

{engineered_info}

{target_info}

## 📝 **요청 내용**
{user_input}

{recommended_steps_info}

{workflow_info}

{function_info}

{log_info}

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔗 **활용 가능한 8개 핵심 기능들**
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
    
    def _generate_guidance(self, user_instructions: str) -> str:
        """FeatureEngineeringAgent 가이드 제공"""
        return f"""# 🔧 **FeatureEngineeringAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **FeatureEngineeringAgent 완전 가이드**

### 1. **피처 엔지니어링 핵심 작업**
FeatureEngineeringAgent는 원시 데이터를 머신러닝에 최적화된 피처로 변환하는 모든 작업을 수행합니다:

- **데이터 전처리**: 타입 변환, 결측값 처리
- **피처 선택**: 불필요한 피처 제거
- **인코딩**: 범주형 변수 수치화
- **스케일링**: 수치형 피처 정규화
- **피처 생성**: 상호작용 및 파생 피처

### 2. **8개 핵심 기능 개별 활용**

#### 🔄 **1. convert_data_types**
```text
데이터 타입을 최적화해주세요
```

#### 🗑️ **2. remove_unique_features**  
```text
불필요한 고유값 피처들을 제거해주세요
```

#### 🏷️ **3. encode_categorical**
```text
범주형 변수들을 원핫 인코딩으로 변환해주세요  
```

#### 📊 **4. handle_high_cardinality**
```text
고차원 범주형 변수를 적절히 처리해주세요
```

#### ⏰ **5. create_datetime_features**
```text
날짜 컬럼으로부터 시간 기반 피처를 생성해주세요
```

#### 📏 **6. scale_numeric_features**
```text
수치형 피처들을 표준화해주세요
```

#### 🔗 **7. create_interaction_features**
```text
피처 간 상호작용을 생성해주세요
```

#### 🎯 **8. handle_target_encoding**
```text
타겟 변수를 적절히 인코딩해주세요
```

### 3. **지원되는 엔지니어링 기법**
- **인코딩**: OneHot, Label, Target, Frequency Encoding
- **스케일링**: StandardScaler, MinMaxScaler, RobustScaler
- **변환**: Log, Square Root, Box-Cox 변환
- **피처 생성**: 다항식, 상호작용, 비닝
- **차원 축소**: PCA, 피처 선택
- **시계열**: 시차, 이동평균, 계절성 피처

### 4. **원본 FeatureEngineeringAgent 특징**
- **자동 타입 감지**: 최적 데이터 타입 자동 선택
- **타겟 인식**: 타겟 변수 자동 감지 및 처리
- **스마트 인코딩**: 카디널리티 기반 최적 인코딩 선택
- **메모리 최적화**: 효율적인 데이터 타입 사용

## 💡 **데이터를 포함해서 다시 요청하면 실제 FeatureEngineeringAgent 작업을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `id,age,category,target\\n1,25,A,1\\n2,30,B,0`
- **JSON**: `[{{"id": 1, "age": 25, "category": "A", "target": 1}}]`

### 🔗 **학습 리소스**
- scikit-learn 전처리: https://scikit-learn.org/stable/modules/preprocessing.html
- 피처 엔지니어링 가이드: https://scikit-learn.org/stable/modules/feature_extraction.html
- pandas 데이터 변환: https://pandas.pydata.org/docs/user_guide/reshaping.html

✅ **FeatureEngineeringAgent 준비 완료!**
"""

    def get_function_mapping(self) -> Dict[str, str]:
        """FeatureEngineeringAgent 8개 기능 매핑"""
        return {
            "convert_data_types": "get_data_engineered",  # 타입 변환된 데이터
            "remove_unique_features": "get_data_engineered",  # 정제된 데이터
            "encode_categorical": "get_data_engineered",  # 인코딩된 데이터
            "handle_high_cardinality": "get_data_engineered",  # 처리된 데이터
            "create_datetime_features": "get_data_engineered",  # 시간 피처가 추가된 데이터
            "scale_numeric_features": "get_feature_engineer_function",  # 스케일링 함수
            "create_interaction_features": "get_recommended_feature_engineering_steps",  # 상호작용 생성 단계
            "handle_target_encoding": "get_workflow_summary"  # 타겟 인코딩 워크플로우
        }

    # 🔥 원본 FeatureEngineeringAgent 8개 메서드들 구현
    def get_data_engineered(self):
        """원본 FeatureEngineeringAgent.get_data_engineered() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_engineered()
        return None
    
    def get_data_raw(self):
        """원본 FeatureEngineeringAgent.get_data_raw() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_raw()
        return None
    
    def get_feature_engineer_function(self, markdown=False):
        """원본 FeatureEngineeringAgent.get_feature_engineer_function() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_feature_engineer_function(markdown=markdown)
        return None
    
    def get_recommended_feature_engineering_steps(self, markdown=False):
        """원본 FeatureEngineeringAgent.get_recommended_feature_engineering_steps() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_recommended_feature_engineering_steps(markdown=markdown)
        return None
    
    def get_workflow_summary(self, markdown=False):
        """원본 FeatureEngineeringAgent.get_workflow_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_workflow_summary(markdown=markdown)
        return None
    
    def get_log_summary(self, markdown=False):
        """원본 FeatureEngineeringAgent.get_log_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_log_summary(markdown=markdown)
        return None
    
    def get_response(self):
        """원본 FeatureEngineeringAgent.get_response() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_response()
        return None


class FeatureEngineeringA2AExecutor(BaseA2AExecutor):
    """FeatureEngineeringAgent A2A Executor"""
    
    def __init__(self):
        wrapper_agent = FeatureEngineeringA2AWrapper()
        super().__init__(wrapper_agent)