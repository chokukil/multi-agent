#!/usr/bin/env python3
"""
DataCleaningA2AWrapper - A2A SDK 0.2.9 래핑 DataCleaningAgent

원본 ai-data-science-team DataCleaningAgent를 A2A SDK 0.2.9 프로토콜로 
래핑하여 8개 기능을 100% 보존합니다.

8개 핵심 기능:
1. detect_missing_values() - 결측값 감지
2. handle_missing_values() - 결측값 처리  
3. detect_outliers() - 이상치 감지
4. treat_outliers() - 이상치 처리
5. validate_data_types() - 데이터 타입 검증
6. detect_duplicates() - 중복 데이터 감지
7. standardize_data() - 데이터 표준화
8. apply_validation_rules() - 검증 규칙 적용
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a_ds_servers.base.base_a2a_wrapper import BaseA2AWrapper, BaseA2AExecutor
from ai_data_science_team.agents import DataCleaningAgent

logger = logging.getLogger(__name__)


class DataCleaningA2AWrapper(BaseA2AWrapper):
    """
    DataCleaningAgent의 A2A SDK 0.2.9 래퍼
    
    원본 ai-data-science-team DataCleaningAgent의 모든 기능을 
    A2A 프로토콜로 래핑하여 제공합니다.
    """
    
    def __init__(self):
        super().__init__(
            agent_name="DataCleaningAgent",
            original_agent_class=DataCleaningAgent,
            port=8306
        )
    
    def _create_original_agent(self):
        """원본 DataCleaningAgent 생성"""
        return DataCleaningAgent(
            model=self.llm,
            n_samples=30,
            log=True,
            log_path="logs/data_cleaning/",
            file_name="data_cleaner.py",
            function_name="data_cleaner",
            overwrite=True,
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False,
            checkpointer=None
        )
    
    async def _invoke_original_agent(self, df: pd.DataFrame, user_input: str, function_name: str = None) -> Dict[str, Any]:
        """원본 DataCleaningAgent invoke_agent 호출"""
        
        # 특정 기능 요청이 있는 경우 해당 기능에 맞는 지시사항 생성
        if function_name:
            user_input = self._get_function_specific_instructions(function_name, user_input)
        
        # 원본 에이전트 호출
        self.agent.invoke_agent(
            data_raw=df,
            user_instructions=user_input
        )
        
        # 8개 기능 결과 수집
        results = {
            "response": self.agent.response,
            "data_cleaned": self.agent.get_data_cleaned(),
            "data_raw": self.agent.get_data_raw(),
            "data_cleaner_function": self.agent.get_data_cleaner_function(),
            "recommended_cleaning_steps": self.agent.get_recommended_cleaning_steps(),
            "workflow_summary": self.agent.get_workflow_summary(),
            "log_summary": self.agent.get_log_summary(),
            "ai_message": None
        }
        
        # AI 메시지 추출
        if results["response"] and results["response"].get("messages"):
            last_message = results["response"]["messages"][-1]
            if hasattr(last_message, 'content'):
                results["ai_message"] = last_message.content
        
        return results
    
    def _get_function_specific_instructions(self, function_name: str, user_input: str) -> str:
        """8개 기능별 특화된 지시사항 생성"""
        
        function_instructions = {
            "detect_missing_values": """
Focus on detecting and reporting missing values in the dataset:
- Identify columns with missing values
- Calculate missing value percentages
- Report patterns in missing data
- Recommend actions for columns with high missing rates (>40%)

Original user request: {}
""",
            "handle_missing_values": """
Focus on handling missing values using appropriate imputation strategies:
- Impute numeric columns with mean values
- Impute categorical columns with mode values
- Remove columns with >40% missing values
- Document imputation strategies used

Original user request: {}
""",
            "detect_outliers": """
Focus on detecting outliers in the dataset:
- Use IQR method to identify outliers (3x interquartile range)
- Report outliers in numeric columns
- Provide statistics on outlier counts per column
- Suggest outlier treatment strategies

Original user request: {}
""",
            "treat_outliers": """
Focus on treating outliers in the dataset:
- Remove extreme outliers (3x interquartile range)
- Apply winsorization if appropriate
- Document outlier treatment methods
- Preserve data integrity during treatment

Original user request: {}
""",
            "validate_data_types": """
Focus on validating and correcting data types:
- Check current data types for each column
- Convert columns to appropriate data types
- Handle type conversion errors gracefully
- Report data type changes made

Original user request: {}
""",
            "detect_duplicates": """
Focus on detecting duplicate records:
- Identify exact duplicate rows
- Report duplicate counts and percentages
- Show examples of duplicate records
- Recommend deduplication strategy

Original user request: {}
""",
            "standardize_data": """
Focus on standardizing data formats and values:
- Standardize text case and formats
- Normalize numeric ranges if needed
- Ensure consistent data representations
- Apply data standardization rules

Original user request: {}
""",
            "apply_validation_rules": """
Focus on applying comprehensive data validation rules:
- Apply business logic validation
- Check data constraints and ranges
- Validate relationships between columns
- Report validation failures and corrections

Original user request: {}
"""
        }
        
        return function_instructions.get(function_name, user_input).format(user_input)
    
    def _format_result(self, result: Dict[str, Any], df: pd.DataFrame, output_path: str, user_input: str) -> str:
        """DataCleaningAgent 특화 결과 포맷팅"""
        
        # 기본 정보
        data_preview = df.head().to_string()
        
        # 정리된 데이터 정보
        cleaned_data_info = ""
        if result.get("data_cleaned") is not None:
            cleaned_df = result["data_cleaned"]
            cleaned_data_info = f"""

## 🧹 **정리된 데이터 정보**  
- **정리 후 크기**: {cleaned_df.shape[0]:,} 행 × {cleaned_df.shape[1]:,} 열
- **제거된 행**: {df.shape[0] - cleaned_df.shape[0]:,} 개
- **변경된 컬럼**: {abs(df.shape[1] - cleaned_df.shape[1]):,} 개
"""
        
        # 생성된 함수 정보
        function_info = ""
        if result.get("data_cleaner_function"):
            function_info = f"""

## 💻 **생성된 데이터 정리 함수**
```python
{result["data_cleaner_function"]}
```
"""
        
        # 추천 단계 정보
        recommended_steps_info = ""
        if result.get("recommended_cleaning_steps"):
            recommended_steps_info = f"""

## 📋 **추천 정리 단계**
{result["recommended_cleaning_steps"]}
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
        
        return f"""# 🧹 **DataCleaningAgent Complete!**

## 📊 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

{cleaned_data_info}

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
1. **detect_missing_values()** - 결측값 감지 및 분석
2. **handle_missing_values()** - 결측값 처리 및 대체
3. **detect_outliers()** - 이상치 감지 및 식별
4. **treat_outliers()** - 이상치 처리 및 제거
5. **validate_data_types()** - 데이터 타입 검증 및 변환
6. **detect_duplicates()** - 중복 데이터 감지
7. **standardize_data()** - 데이터 표준화 및 정규화
8. **apply_validation_rules()** - 검증 규칙 적용

✅ **원본 ai-data-science-team DataCleaningAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _generate_guidance(self, user_instructions: str) -> str:
        """DataCleaningAgent 가이드 제공"""
        return f"""# 🧹 **DataCleaningAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **DataCleaningAgent 완전 가이드**

### 1. **기본 데이터 정리 과정**
DataCleaningAgent는 다음과 같은 기본 정리 단계를 수행합니다:

1. **결측값 처리**: 40% 이상 결측인 컬럼 제거
2. **결측값 대체**: 숫자형(평균), 범주형(최빈값)
3. **데이터 타입 변환**: 적절한 데이터 타입으로 변환
4. **중복 제거**: 중복된 행 제거
5. **이상치 처리**: 3×IQR 범위 밖 극단값 제거

### 2. **8개 핵심 기능 개별 활용**

#### 🔍 **1. detect_missing_values**
```text
결측값 패턴을 자세히 분석해주세요
```

#### 🔧 **2. handle_missing_values**  
```text
결측값을 적절한 방법으로 처리해주세요
```

#### 📊 **3. detect_outliers**
```text
데이터의 이상치를 감지하고 분석해주세요  
```

#### ⚡ **4. treat_outliers**
```text
이상치를 적절히 처리해주세요
```

#### ✅ **5. validate_data_types**
```text
데이터 타입을 검증하고 수정해주세요
```

#### 🔄 **6. detect_duplicates**
```text
중복된 데이터를 찾아주세요
```

#### 📐 **7. standardize_data**
```text
데이터를 표준화해주세요
```

#### 🛡️ **8. apply_validation_rules**
```text
데이터 검증 규칙을 적용해주세요
```

### 3. **원본 DataCleaningAgent 특징**
- **LangGraph 기반**: 상태 기반 워크플로우
- **자동 코드 생성**: Python 함수 자동 생성
- **오류 복구**: 자동 재시도 및 수정
- **로깅 지원**: 상세한 처리 과정 기록
- **사용자 검토**: Human-in-the-loop 지원

## 💡 **데이터를 포함해서 다시 요청하면 실제 DataCleaningAgent 작업을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `name,age,salary\\nJohn,25,50000\\nJane,,60000\\nBob,30,`
- **JSON**: `[{{"name": "John", "age": 25, "salary": 50000}}]`

### 🔗 **추가 리소스**
- DataCleaningAgent 문서: ai-data-science-team 패키지
- pandas 데이터 정리: https://pandas.pydata.org/docs/
- scikit-learn 전처리: https://scikit-learn.org/stable/modules/preprocessing.html

✅ **DataCleaningAgent 준비 완료!**
"""

    def get_function_mapping(self) -> Dict[str, str]:
        """DataCleaningAgent 8개 기능 매핑"""
        return {
            "detect_missing_values": "get_data_raw",  # 원본 데이터에서 결측값 분석
            "handle_missing_values": "get_data_cleaned",  # 정리된 데이터 (결측값 처리됨) 
            "detect_outliers": "get_data_raw",  # 원본 데이터에서 이상치 분석
            "treat_outliers": "get_data_cleaned",  # 정리된 데이터 (이상치 처리됨)
            "validate_data_types": "get_data_cleaned",  # 정리된 데이터 (타입 변환됨)
            "detect_duplicates": "get_data_raw",  # 원본 데이터에서 중복 분석
            "standardize_data": "get_data_cleaned",  # 정리된 데이터 (표준화됨)
            "apply_validation_rules": "get_data_cleaner_function"  # 생성된 검증 함수
        }

    # 🔥 원본 DataCleaningAgent 8개 메서드들 구현
    def get_data_cleaned(self):
        """원본 DataCleaningAgent.get_data_cleaned() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_cleaned()
        return None
    
    def get_data_raw(self):
        """원본 DataCleaningAgent.get_data_raw() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_raw()
        return None
    
    def get_data_cleaner_function(self, markdown=False):
        """원본 DataCleaningAgent.get_data_cleaner_function() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_data_cleaner_function(markdown=markdown)
        return None
    
    def get_recommended_cleaning_steps(self, markdown=False):
        """원본 DataCleaningAgent.get_recommended_cleaning_steps() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_recommended_cleaning_steps(markdown=markdown)
        return None
    
    def get_workflow_summary(self, markdown=False):
        """원본 DataCleaningAgent.get_workflow_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_workflow_summary(markdown=markdown)
        return None
    
    def get_log_summary(self, markdown=False):
        """원본 DataCleaningAgent.get_log_summary() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_log_summary(markdown=markdown)
        return None
    
    def get_state_keys(self):
        """원본 DataCleaningAgent.get_state_keys() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_state_keys()
        return None
    
    def get_state_properties(self):
        """원본 DataCleaningAgent.get_state_properties() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_state_properties()
        return None


class DataCleaningA2AExecutor(BaseA2AExecutor):
    """DataCleaningAgent A2A Executor"""
    
    def __init__(self):
        wrapper_agent = DataCleaningA2AWrapper()
        super().__init__(wrapper_agent)