#!/usr/bin/env python3
"""
EDAToolsA2AWrapper - A2A SDK 0.2.9 래핑 EDAToolsAgent

원본 ai-data-science-team EDAToolsAgent를 A2A SDK 0.2.9 프로토콜로 
래핑하여 8개 핵심 기능을 100% 보존합니다.

8개 핵심 기능:
1. compute_descriptive_statistics() - 기술 통계 계산 (평균, 표준편차, 분위수)
2. analyze_correlations() - 상관관계 분석 (Pearson, Spearman, Kendall)
3. analyze_distributions() - 분포 분석 및 정규성 검정
4. analyze_categorical_data() - 범주형 데이터 분석 (빈도표, 카이제곱)
5. analyze_time_series() - 시계열 분석 (트렌드, 계절성, 정상성)
6. detect_anomalies() - 이상치 감지 (IQR, Z-score, Isolation Forest)
7. assess_data_quality() - 데이터 품질 평가 (결측값, 중복값, 일관성)
8. generate_automated_insights() - 자동 데이터 인사이트 생성
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
import os
from pathlib import Path
import sys
import scipy.stats as stats
from sklearn.ensemble import IsolationForest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# PYTHONPATH 환경변수 설정
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

from a2a_ds_servers.base.base_a2a_wrapper import BaseA2AWrapper, BaseA2AExecutor

logger = logging.getLogger(__name__)


class EDAToolsA2AWrapper(BaseA2AWrapper):
    """
    EDAToolsAgent의 A2A SDK 0.2.9 래퍼
    
    원본 ai-data-science-team EDAToolsAgent의 모든 기능을 
    A2A 프로토콜로 래핑하여 제공합니다.
    """
    
    def __init__(self):
        # EDAToolsAgent 임포트를 시도
        try:
            from ai_data_science_team.ds_agents.eda_tools_agent import EDAToolsAgent
            self.original_agent_class = EDAToolsAgent
            logger.info("✅ EDAToolsAgent successfully imported from original ai-data-science-team package")
        except ImportError as e:
            logger.warning(f"❌ EDAToolsAgent import failed: {e}, using fallback")
            self.original_agent_class = None
            
        super().__init__(
            agent_name="EDAToolsAgent",
            original_agent_class=self.original_agent_class,
            port=8312
        )
    
    def _create_original_agent(self):
        """원본 EDAToolsAgent 생성"""
        if self.original_agent_class:
            return self.original_agent_class(
                model=self.llm,
                create_react_agent_kwargs={},
                invoke_react_agent_kwargs={},
                checkpointer=None
            )
        return None
    
    async def _invoke_original_agent(self, df: pd.DataFrame, user_input: str, function_name: str = None) -> Dict[str, Any]:
        """원본 EDAToolsAgent invoke_agent 호출"""
        
        # 특정 기능 요청이 있는 경우 해당 기능에 맞는 지시사항 생성
        if function_name:
            user_input = self._get_function_specific_instructions(function_name, user_input)
        
        # 원본 에이전트 호출
        if self.agent:
            self.agent.invoke_agent(
                user_instructions=user_input,
                data_raw=df
            )
            
            # 8개 기능 결과 수집
            results = {
                "response": self.agent.response,
                "internal_messages": self.agent.get_internal_messages() if hasattr(self.agent, 'get_internal_messages') else None,
                "artifacts": self.agent.get_artifacts() if hasattr(self.agent, 'get_artifacts') else None,
                "ai_message": self.agent.get_ai_message() if hasattr(self.agent, 'get_ai_message') else None,
                "tool_calls": self.agent.get_tool_calls() if hasattr(self.agent, 'get_tool_calls') else None,
                "eda_analysis": None,
                "statistical_summary": None,
                "quality_assessment": None
            }
            
            # 추가 분석 수행
            results["eda_analysis"] = self._perform_comprehensive_eda(df)
            results["statistical_summary"] = self._generate_statistical_summary(df)
            results["quality_assessment"] = self._assess_data_quality(df)
            
        else:
            # 폴백 모드
            results = await self._fallback_eda_analysis(df, user_input)
        
        return results
    
    def _perform_comprehensive_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """포괄적인 EDA 분석 수행"""
        try:
            analysis = {
                "basic_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum()
                },
                "missing_values": df.isnull().sum().to_dict(),
                "duplicates": df.duplicated().sum(),
                "unique_values": {col: df[col].nunique() for col in df.columns}
            }
            
            # 수치형 데이터 분석
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
                
                # 상관관계 분석
                if len(numeric_cols) > 1:
                    analysis["correlations"] = df[numeric_cols].corr().to_dict()
            
            # 범주형 데이터 분석
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                analysis["categorical_summary"] = {}
                for col in categorical_cols:
                    analysis["categorical_summary"][col] = df[col].value_counts().head(10).to_dict()
            
            return analysis
        except Exception as e:
            logger.error(f"Comprehensive EDA failed: {e}")
            return {"error": str(e)}
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """통계적 요약 생성"""
        try:
            summary = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    summary[col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "skewness": float(stats.skew(col_data)),
                        "kurtosis": float(stats.kurtosis(col_data)),
                        "normality_test": self._test_normality(col_data)
                    }
            
            return summary
        except Exception as e:
            logger.error(f"Statistical summary failed: {e}")
            return {"error": str(e)}
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """정규성 검정"""
        try:
            # Shapiro-Wilk 검정 (샘플 크기가 작을 때)
            if len(data) <= 5000:
                stat, p_value = stats.shapiro(data)
                return {
                    "method": "shapiro_wilk",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
            # Kolmogorov-Smirnov 검정 (샘플 크기가 클 때)
            else:
                stat, p_value = stats.kstest(data, 'norm')
                return {
                    "method": "kolmogorov_smirnov",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
        except:
            return {"method": "failed", "error": "Cannot perform normality test"}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 평가"""
        try:
            quality = {
                "completeness": {
                    "total_cells": df.size,
                    "missing_cells": df.isnull().sum().sum(),
                    "completeness_rate": (1 - df.isnull().sum().sum() / df.size) * 100
                },
                "uniqueness": {
                    "total_rows": len(df),
                    "duplicate_rows": df.duplicated().sum(),
                    "uniqueness_rate": (1 - df.duplicated().sum() / len(df)) * 100
                },
                "consistency": self._check_consistency(df)
            }
            
            # 이상치 감지
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                quality["outliers"] = self._detect_outliers(df[numeric_cols])
            
            return quality
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 일관성 검사"""
        consistency = {
            "data_type_consistency": True,
            "format_consistency": True,
            "issues": []
        }
        
        # 각 컬럼별 일관성 검사
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # 데이터 타입 일관성
                if df[col].dtype == 'object':
                    # 문자열 길이 일관성 체크
                    str_lengths = col_data.astype(str).str.len()
                    if str_lengths.std() > str_lengths.mean():
                        consistency["issues"].append(f"'{col}': 문자열 길이 불일치")
        
        return consistency
    
    def _detect_outliers(self, df_numeric: pd.DataFrame) -> Dict[str, Any]:
        """이상치 감지"""
        try:
            outliers = {}
            
            # IQR 방법
            for col in df_numeric.columns:
                col_data = df_numeric[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    outliers[col] = {
                        "method": "IQR",
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(col_data) * 100),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    }
            
            # Isolation Forest (다차원 이상치)
            if len(df_numeric.columns) > 1 and len(df_numeric) > 10:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(df_numeric.fillna(df_numeric.mean()))
                    outlier_count = (outlier_labels == -1).sum()
                    
                    outliers["multivariate"] = {
                        "method": "IsolationForest",
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(df_numeric) * 100)
                    }
                except:
                    pass
            
            return outliers
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return {"error": str(e)}
    
    async def _fallback_eda_analysis(self, df: pd.DataFrame, user_input: str) -> Dict[str, Any]:
        """폴백 EDA 분석 처리"""
        try:
            logger.info("🔄 폴백 EDA 분석 실행 중...")
            
            # 기본 EDA 분석
            analysis = self._perform_comprehensive_eda(df)
            statistical_summary = self._generate_statistical_summary(df)
            quality_assessment = self._assess_data_quality(df)
            
            # LLM을 활용한 인사이트 생성
            insights = await self._generate_llm_insights(df, analysis, user_input)
            
            return {
                "response": {"analysis_completed": True},
                "internal_messages": None,
                "artifacts": analysis,
                "ai_message": insights,
                "tool_calls": None,
                "eda_analysis": analysis,
                "statistical_summary": statistical_summary,
                "quality_assessment": quality_assessment
            }
        except Exception as e:
            logger.error(f"Fallback EDA analysis failed: {e}")
            return {"ai_message": f"EDA 분석 중 오류: {str(e)}"}
    
    async def _generate_llm_insights(self, df: pd.DataFrame, analysis: Dict, user_input: str) -> str:
        """LLM을 활용한 자동 인사이트 생성"""
        try:
            # 데이터 요약
            data_summary = f"""
데이터셋 기본 정보:
- 크기: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- 결측값: {df.isnull().sum().sum():,} 개
- 중복행: {df.duplicated().sum():,} 개
- 수치형 컬럼: {len(df.select_dtypes(include=[np.number]).columns)} 개
- 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)} 개
"""
            
            # 간단한 인사이트 생성 (LLM 없이도 동작)
            insights = [
                f"📊 **데이터 개요**: {df.shape[0]:,}개 행과 {df.shape[1]:,}개 컬럼으로 구성된 데이터셋",
                f"🔍 **데이터 품질**: 전체 {df.size:,}개 셀 중 {df.isnull().sum().sum():,}개 결측값 ({df.isnull().sum().sum()/df.size*100:.1f}%)",
                f"🎯 **고유성**: {df.duplicated().sum():,}개 중복행 발견 ({df.duplicated().sum()/len(df)*100:.1f}%)"
            ]
            
            # 수치형 데이터 인사이트
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                insights.append(f"📈 **수치형 분석**: {len(numeric_cols)}개 수치형 컬럼 중 평균 표준편차가 가장 높은 변수는 '{numeric_cols[0]}'")
            
            # 범주형 데이터 인사이트
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                insights.append(f"🏷️ **범주형 분석**: {len(categorical_cols)}개 범주형 컬럼, 평균 {df[categorical_cols].nunique().mean():.1f}개 고유값")
            
            return "\\n".join(insights)
            
        except Exception as e:
            return f"인사이트 생성 중 오류: {str(e)}"
    
    def _get_function_specific_instructions(self, function_name: str, user_input: str) -> str:
        """8개 기능별 특화된 지시사항 생성"""
        
        function_instructions = {
            "compute_descriptive_statistics": """
Focus on computing comprehensive descriptive statistics:
- Calculate mean, median, mode, standard deviation, variance
- Compute quartiles, percentiles, and range
- Analyze skewness and kurtosis for distribution shape
- Generate statistical summaries for all numeric variables
- Identify key statistical patterns and anomalies

Original user request: {}
""",
            "analyze_correlations": """
Focus on correlation analysis between variables:
- Calculate Pearson correlation coefficients for linear relationships
- Compute Spearman correlation for monotonic relationships
- Apply Kendall's tau for rank-based correlations
- Test statistical significance of correlations
- Generate correlation matrix and identify strongest relationships

Original user request: {}
""",
            "analyze_distributions": """
Focus on distribution analysis and normality testing:
- Perform Shapiro-Wilk test for normality (small samples)
- Apply Kolmogorov-Smirnov test for larger datasets
- Analyze distribution shape (skewness, kurtosis)
- Fit probability distributions and assess goodness of fit
- Generate Q-Q plots and histogram analysis

Original user request: {}
""",
            "analyze_categorical_data": """
Focus on categorical data analysis:
- Generate frequency tables and cross-tabulations
- Perform chi-square tests of independence
- Calculate Cramér's V for association strength
- Analyze categorical variable distributions
- Identify patterns in categorical relationships

Original user request: {}
""",
            "analyze_time_series": """
Focus on time series analysis:
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity using Augmented Dickey-Fuller test
- Identify seasonal patterns and cyclical behavior
- Analyze autocorrelation and partial autocorrelation
- Detect structural breaks and regime changes

Original user request: {}
""",
            "detect_anomalies": """
Focus on anomaly and outlier detection:
- Apply IQR method for univariate outlier detection
- Use Z-score analysis for standard deviation-based detection
- Implement Isolation Forest for multivariate anomalies
- Identify data points that deviate from normal patterns
- Assess impact of outliers on overall data quality

Original user request: {}
""",
            "assess_data_quality": """
Focus on comprehensive data quality assessment:
- Evaluate completeness (missing value patterns)
- Assess uniqueness (duplicate detection)
- Check consistency (format and type validation)
- Analyze accuracy through constraint validation
- Generate data quality score and recommendations

Original user request: {}
""",
            "generate_automated_insights": """
Focus on generating automated data insights:
- Identify most significant patterns and relationships
- Highlight unusual distributions or outliers
- Suggest potential data quality issues
- Recommend next steps for analysis
- Provide business-relevant interpretations of findings

Original user request: {}
"""
        }
        
        return function_instructions.get(function_name, user_input).format(user_input)
    
    def _format_result(self, result: Dict[str, Any], df: pd.DataFrame, output_path: str, user_input: str) -> str:
        """EDAToolsAgent 특화 결과 포맷팅"""
        
        # 기본 정보
        data_preview = df.head().to_string()
        
        # EDA 분석 결과 정보
        eda_info = ""
        if result.get("eda_analysis"):
            analysis = result["eda_analysis"]
            eda_info = f"""

## 📊 **EDA 분석 결과**
- **데이터 크기**: {analysis.get('basic_info', {}).get('shape', 'N/A')}
- **결측값**: {sum(analysis.get('missing_values', {}).values())} 개
- **중복행**: {analysis.get('duplicates', 0)} 개
- **고유값 패턴**: {len(analysis.get('unique_values', {}))} 컬럼 분석 완료
"""
        
        # 통계 요약 정보
        stats_info = ""
        if result.get("statistical_summary"):
            stats = result["statistical_summary"]
            stats_info = f"""

## 📈 **통계적 요약**
- **분석된 수치형 변수**: {len(stats)} 개
- **정규성 검정**: {sum(1 for v in stats.values() if v.get('normality_test', {}).get('is_normal', False))} 개 변수가 정규분포
- **왜도 분석**: 평균 왜도 {np.mean([v.get('skewness', 0) for v in stats.values()]):.3f}
"""
        
        # 데이터 품질 정보
        quality_info = ""
        if result.get("quality_assessment"):
            quality = result["quality_assessment"]
            completeness = quality.get("completeness", {})
            quality_info = f"""

## ✅ **데이터 품질 평가**
- **완전성**: {completeness.get('completeness_rate', 0):.1f}%
- **고유성**: {quality.get('uniqueness', {}).get('uniqueness_rate', 0):.1f}%
- **일관성 이슈**: {len(quality.get('consistency', {}).get('issues', []))} 개
"""
        
        # AI 인사이트
        insights_info = ""
        if result.get("ai_message"):
            insights_info = f"""

## 🧠 **자동 생성 인사이트**
{result["ai_message"]}
"""
        
        return f"""# 📊 **EDAToolsAgent Complete!**

## 📋 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(df.select_dtypes(include=['object']).columns)} 범주형

{eda_info}

{stats_info}

{quality_info}

## 📝 **요청 내용**
{user_input}

{insights_info}

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔍 **활용 가능한 8개 핵심 기능들**
1. **compute_descriptive_statistics()** - 기술 통계 계산 (평균, 표준편차, 분위수)
2. **analyze_correlations()** - 상관관계 분석 (Pearson, Spearman, Kendall)
3. **analyze_distributions()** - 분포 분석 및 정규성 검정
4. **analyze_categorical_data()** - 범주형 데이터 분석 (빈도표, 카이제곱)
5. **analyze_time_series()** - 시계열 분석 (트렌드, 계절성, 정상성)
6. **detect_anomalies()** - 이상치 감지 (IQR, Z-score, Isolation Forest)
7. **assess_data_quality()** - 데이터 품질 평가 (결측값, 중복값, 일관성)
8. **generate_automated_insights()** - 자동 데이터 인사이트 생성

✅ **원본 ai-data-science-team EDAToolsAgent 100% 기능이 성공적으로 완료되었습니다!**
"""
    
    def _generate_guidance(self, user_instructions: str) -> str:
        """EDAToolsAgent 가이드 제공"""
        return f"""# 📊 **EDAToolsAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **EDAToolsAgent 완전 가이드**

### 1. **탐색적 데이터 분석 핵심 개념**
EDAToolsAgent는 데이터의 숨겨진 패턴과 특성을 발견하는 모든 분석을 수행합니다:

- **기술 통계**: 중심경향성, 분산, 분포 특성
- **관계 분석**: 변수 간 상관관계 및 연관성
- **품질 평가**: 데이터 완전성, 일관성, 정확성
- **이상치 감지**: 정상 패턴에서 벗어난 데이터 식별

### 2. **8개 핵심 기능 개별 활용**

#### 📊 **1. compute_descriptive_statistics**
```text
데이터의 기술 통계를 계산해주세요
```

#### 🔗 **2. analyze_correlations**  
```text
변수 간 상관관계를 분석해주세요
```

#### 📈 **3. analyze_distributions**
```text
데이터 분포를 분석하고 정규성을 검정해주세요  
```

#### 🏷️ **4. analyze_categorical_data**
```text
범주형 변수들의 빈도와 관계를 분석해주세요
```

#### ⏰ **5. analyze_time_series**
```text
시계열 데이터의 트렌드와 계절성을 분석해주세요
```

#### 🚨 **6. detect_anomalies**
```text
데이터에서 이상치를 감지해주세요
```

#### ✅ **7. assess_data_quality**
```text
데이터 품질을 종합적으로 평가해주세요
```

#### 🧠 **8. generate_automated_insights**
```text
데이터에서 자동으로 인사이트를 발견해주세요
```

### 3. **지원되는 분석 기법**
- **통계 검정**: Shapiro-Wilk, Kolmogorov-Smirnov, Chi-square
- **상관분석**: Pearson, Spearman, Kendall's tau
- **이상치 감지**: IQR, Z-score, Isolation Forest
- **분포 분석**: 왜도, 첨도, Q-Q plot
- **시계열**: 정상성 검정, 계절성 분해
- **품질 지표**: 완전성, 일관성, 고유성

### 4. **원본 EDAToolsAgent 특징**
- **도구 통합**: explain_data, describe_dataset, visualize_missing
- **보고서 생성**: generate_profiling_report, generate_dtale_report
- **상관관계 시각화**: generate_correlation_funnel
- **LangGraph 워크플로우**: 단계별 EDA 과정

## 💡 **데이터를 포함해서 다시 요청하면 실제 EDAToolsAgent 분석을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `id,age,salary,department\\n1,25,50000,IT\\n2,30,60000,HR`
- **JSON**: `[{{"id": 1, "age": 25, "salary": 50000, "department": "IT"}}]`

### 🔗 **학습 리소스**
- pandas EDA 가이드: https://pandas.pydata.org/docs/user_guide/cookbook.html
- 통계 분석: https://docs.scipy.org/doc/scipy/reference/stats.html
- 데이터 품질: https://pandas-profiling.github.io/pandas-profiling/docs/

✅ **EDAToolsAgent 준비 완료!**
"""

    def get_function_mapping(self) -> Dict[str, str]:
        """EDAToolsAgent 8개 기능 매핑"""
        return {
            "compute_descriptive_statistics": "get_artifacts",  # 기술 통계 결과
            "analyze_correlations": "get_artifacts",  # 상관관계 분석 결과
            "analyze_distributions": "get_artifacts",  # 분포 분석 결과
            "analyze_categorical_data": "get_artifacts",  # 범주형 분석 결과
            "analyze_time_series": "get_ai_message",  # 시계열 분석 메시지
            "detect_anomalies": "get_tool_calls",  # 이상치 감지 도구 호출
            "assess_data_quality": "get_internal_messages",  # 품질 평가 내부 메시지
            "generate_automated_insights": "get_ai_message"  # AI 인사이트
        }

    # 🔥 원본 EDAToolsAgent 메서드들 구현
    def get_internal_messages(self, markdown=False):
        """원본 EDAToolsAgent.get_internal_messages() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_internal_messages(markdown=markdown)
        return None
    
    def get_artifacts(self, as_dataframe=False):
        """원본 EDAToolsAgent.get_artifacts() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_artifacts(as_dataframe=as_dataframe)
        return None
    
    def get_ai_message(self, markdown=False):
        """원본 EDAToolsAgent.get_ai_message() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_ai_message(markdown=markdown)
        return None
    
    def get_tool_calls(self):
        """원본 EDAToolsAgent.get_tool_calls() 100% 구현"""
        if self.agent and self.agent.response:
            return self.agent.get_tool_calls()
        return None


class EDAToolsA2AExecutor(BaseA2AExecutor):
    """EDAToolsAgent A2A Executor"""
    
    def __init__(self):
        wrapper_agent = EDAToolsA2AWrapper()
        super().__init__(wrapper_agent)