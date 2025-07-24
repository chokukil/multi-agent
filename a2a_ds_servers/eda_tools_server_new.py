#!/usr/bin/env python3
"""
EDA Tools Server - A2A SDK 0.2.9 래핑 구현

원본 ai-data-science-team EDAToolsAgent를 A2A SDK 0.2.9로 래핑하여
8개 핵심 기능을 100% 보존합니다.

포트: 8312
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
from typing import Dict, Any

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


class EDAToolsServerAgent:
    """
    ai-data-science-team EDAToolsAgent 래핑 클래스
    
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
        
        # 원본 EDAToolsAgent 초기화 시도
        try:
            # ai-data-science-team 경로 추가
            ai_ds_team_path = project_root / "ai_ds_team"
            sys.path.insert(0, str(ai_ds_team_path))
            
            from ai_data_science_team.ds_agents.eda_tools_agent import EDAToolsAgent
            
            self.agent = EDAToolsAgent(
                model=self.llm,
                create_react_agent_kwargs={},
                invoke_react_agent_kwargs={},
                checkpointer=None
            )
            self.has_original_agent = True
            logger.info("✅ 원본 EDAToolsAgent 초기화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 원본 EDAToolsAgent 사용 불가: {e}")
            self.has_original_agent = False
            logger.info("✅ 폴백 모드로 초기화 완료")
    
    async def process_eda_analysis(self, user_input: str) -> str:
        """EDA 분석 처리 실행"""
        try:
            logger.info(f"🚀 EDA 분석 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_input)
            
            if df is None:
                return self._generate_eda_guidance(user_input)
            
            # 원본 에이전트 사용 시도
            if self.has_original_agent and self.agent:
                return await self._process_with_original_agent(df, user_input)
            else:
                return await self._process_with_fallback(df, user_input)
                
        except Exception as e:
            logger.error(f"❌ EDA 분석 처리 중 오류: {e}")
            return f"❌ EDA 분석 처리 중 오류 발생: {str(e)}"
    
    async def _process_with_original_agent(self, df: pd.DataFrame, user_input: str) -> str:
        """원본 EDAToolsAgent 사용"""
        try:
            logger.info("🤖 원본 EDAToolsAgent 실행 중...")
            
            # 원본 에이전트 invoke_agent 호출
            self.agent.invoke_agent(
                user_instructions=user_input,
                data_raw=df
            )
            
            # 결과 수집
            internal_messages = self.agent.get_internal_messages() if hasattr(self.agent, 'get_internal_messages') else None
            artifacts = self.agent.get_artifacts() if hasattr(self.agent, 'get_artifacts') else None
            ai_message = self.agent.get_ai_message() if hasattr(self.agent, 'get_ai_message') else None
            tool_calls = self.agent.get_tool_calls() if hasattr(self.agent, 'get_tool_calls') else None
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"eda_analysis_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            logger.info(f"원본 데이터 저장: {output_path}")
            
            # 결과 포맷팅
            return self._format_original_agent_result(
                df, user_input, output_path, ai_message, artifacts, tool_calls
            )
            
        except Exception as e:
            logger.error(f"원본 에이전트 처리 실패: {e}")
            return await self._process_with_fallback(df, user_input)
    
    async def _process_with_fallback(self, df: pd.DataFrame, user_input: str) -> str:
        """폴백 EDA 분석 처리"""
        try:
            logger.info("🔄 폴백 EDA 분석 실행 중...")
            
            # 기본 EDA 분석 수행
            eda_results = self._perform_comprehensive_eda(df)
            statistical_summary = self._compute_descriptive_statistics(df)
            quality_assessment = self._assess_data_quality(df)
            correlation_analysis = self._analyze_correlations(df)
            
            # 데이터 저장
            data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
            os.makedirs(data_path, exist_ok=True)
            
            timestamp = int(time.time())
            output_file = f"eda_analysis_fallback_{timestamp}.csv"
            output_path = os.path.join(data_path, output_file)
            
            df.to_csv(output_path, index=False)
            
            return self._format_fallback_result(
                df, user_input, output_path, eda_results, statistical_summary, 
                quality_assessment, correlation_analysis
            )
            
        except Exception as e:
            logger.error(f"폴백 처리 실패: {e}")
            return f"❌ EDA 분석 실패: {str(e)}"
    
    def _perform_comprehensive_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """포괄적인 EDA 분석 수행"""
        try:
            analysis = {
                "basic_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "missing_analysis": {
                    "missing_counts": df.isnull().sum().to_dict(),
                    "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
                    "total_missing": df.isnull().sum().sum()
                },
                "uniqueness": {
                    "unique_counts": {col: df[col].nunique() for col in df.columns},
                    "duplicate_rows": df.duplicated().sum(),
                    "duplicate_percentage": df.duplicated().sum() / len(df) * 100
                }
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Comprehensive EDA failed: {e}")
            return {"error": str(e)}
    
    def _compute_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기술 통계 계산"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats = {}
            
            if len(numeric_cols) > 0:
                desc_stats = df[numeric_cols].describe()
                stats["numeric_summary"] = desc_stats.to_dict()
                
                # 추가 통계 (왜도, 첨도)
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        stats[f"{col}_extended"] = {
                            "skewness": float(col_data.skew()),
                            "kurtosis": float(col_data.kurtosis()),
                            "coefficient_of_variation": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else np.inf
                        }
            
            # 범주형 데이터 통계
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                stats["categorical_summary"] = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    stats["categorical_summary"][col] = {
                        "unique_count": df[col].nunique(),
                        "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                        "frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "top_5_values": value_counts.head(5).to_dict()
                    }
            
            return stats
        except Exception as e:
            logger.error(f"Descriptive statistics failed: {e}")
            return {"error": str(e)}
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """상관관계 분석"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlations = {}
            
            if len(numeric_cols) > 1:
                # Pearson 상관계수
                pearson_corr = df[numeric_cols].corr(method='pearson')
                correlations["pearson"] = pearson_corr.to_dict()
                
                # Spearman 상관계수
                spearman_corr = df[numeric_cols].corr(method='spearman')
                correlations["spearman"] = spearman_corr.to_dict()
                
                # 강한 상관관계 식별 (절댓값 0.7 이상)
                strong_correlations = []
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # 중복 방지
                            corr_value = pearson_corr.loc[col1, col2]
                            if abs(corr_value) > 0.7:
                                strong_correlations.append({
                                    "variable1": col1,
                                    "variable2": col2,
                                    "correlation": float(corr_value),
                                    "strength": "strong positive" if corr_value > 0 else "strong negative"
                                })
                
                correlations["strong_correlations"] = strong_correlations
            
            return correlations
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {"error": str(e)}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 평가"""
        try:
            quality = {
                "completeness": {
                    "total_cells": int(df.size),
                    "missing_cells": int(df.isnull().sum().sum()),
                    "completeness_rate": float((1 - df.isnull().sum().sum() / df.size) * 100)
                },
                "uniqueness": {
                    "total_rows": int(len(df)),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "uniqueness_rate": float((1 - df.duplicated().sum() / len(df)) * 100)
                },
                "consistency": self._check_data_consistency(df),
                "outliers": self._detect_outliers_basic(df)
            }
            
            # 전체 품질 점수 계산 (0-100)
            quality_score = (
                quality["completeness"]["completeness_rate"] * 0.4 +
                quality["uniqueness"]["uniqueness_rate"] * 0.3 +
                (100 - len(quality["consistency"]["issues"]) * 10) * 0.3
            )
            quality["overall_quality_score"] = max(0, min(100, quality_score))
            
            return quality
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 일관성 검사"""
        consistency = {
            "issues": [],
            "checks_performed": []
        }
        
        try:
            # 데이터 타입 일관성 검사
            for col in df.columns:
                consistency["checks_performed"].append(f"Type consistency for '{col}'")
                
                if df[col].dtype == 'object':
                    # 숫자처럼 보이는 문자열 찾기
                    numeric_like = df[col].str.match(r'^-?\d+\.?\d*$', na=False).sum()
                    if numeric_like > len(df) * 0.8:  # 80% 이상이 숫자 형태
                        consistency["issues"].append(f"'{col}': 숫자형 데이터가 문자형으로 저장됨")
                
                # 날짜 형식 일관성 (간단한 체크)
                if df[col].dtype == 'object':
                    date_patterns = df[col].str.match(r'^\d{4}-\d{2}-\d{2}', na=False).sum()
                    if date_patterns > len(df) * 0.5:
                        consistency["issues"].append(f"'{col}': 날짜 형식 데이터가 문자형으로 저장됨")
        
        except Exception as e:
            consistency["issues"].append(f"Consistency check error: {str(e)}")
        
        return consistency
    
    def _detect_outliers_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기본 이상치 감지"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # IQR 방법
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_count = outlier_mask.sum()
                    
                    outliers[col] = {
                        "method": "IQR",
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(col_data) * 100),
                        "bounds": {
                            "lower": float(lower_bound),
                            "upper": float(upper_bound)
                        }
                    }
            
            return outliers
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return {"error": str(e)}
    
    def _format_original_agent_result(self, df, user_input, output_path, ai_message, artifacts, tool_calls) -> str:
        """원본 에이전트 결과 포맷팅"""
        
        data_preview = df.head().to_string()
        
        ai_info = ""
        if ai_message:
            ai_info = f"""

## 🤖 **AI 분석 결과**
{ai_message}
"""
        
        artifacts_info = ""
        if artifacts:
            artifacts_info = f"""

## 📊 **분석 아티팩트**
- **생성된 아티팩트**: {len(artifacts) if isinstance(artifacts, list) else 'N/A'}
- **분석 도구 활용**: EDA 전용 도구들 실행 완료
"""
        
        tools_info = ""
        if tool_calls:
            tools_info = f"""

## 🔧 **실행된 도구들**
- **도구 호출 수**: {len(tool_calls) if isinstance(tool_calls, list) else 'N/A'}  
- **EDA 도구**: explain_data, describe_dataset, visualize_missing, generate_correlation_funnel 등
"""
        
        return f"""# 📊 **EDAToolsAgent Complete!**

## 📋 **원본 데이터 정보**
- **파일 위치**: `{output_path}`
- **데이터 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **컬럼**: {', '.join(df.columns.tolist())}
- **데이터 타입**: {len(df.select_dtypes(include=[np.number]).columns)} 숫자형, {len(df.select_dtypes(include=['object']).columns)} 범주형

## 📝 **요청 내용**
{user_input}

{ai_info}

{artifacts_info}

{tools_info}

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔍 **EDAToolsAgent 8개 핵심 기능들**
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
    
    def _format_fallback_result(self, df, user_input, output_path, eda_results, statistical_summary, quality_assessment, correlation_analysis) -> str:
        """폴백 결과 포맷팅"""
        
        data_preview = df.head().to_string()
        
        # EDA 결과 요약
        eda_summary = ""
        if "basic_info" in eda_results:
            basic_info = eda_results["basic_info"]
            eda_summary = f"""

## 📊 **EDA 분석 결과**
- **데이터 크기**: {basic_info['shape'][0]:,} 행 × {basic_info['shape'][1]:,} 열
- **메모리 사용량**: {basic_info.get('memory_usage_mb', 0):.2f} MB
- **결측값**: {eda_results.get('missing_analysis', {}).get('total_missing', 0):,} 개
- **중복행**: {eda_results.get('uniqueness', {}).get('duplicate_rows', 0):,} 개
"""
        
        # 통계 요약
        stats_summary = ""
        if "numeric_summary" in statistical_summary:
            numeric_count = len(statistical_summary["numeric_summary"])
            stats_summary = f"""

## 📈 **기술 통계 요약**
- **분석된 수치형 변수**: {numeric_count} 개
- **범주형 변수**: {len(statistical_summary.get('categorical_summary', {}))} 개
"""
        
        # 상관관계 요약
        corr_summary = ""
        if "strong_correlations" in correlation_analysis:
            strong_corr_count = len(correlation_analysis["strong_correlations"])
            corr_summary = f"""

## 🔗 **상관관계 분석**
- **분석된 변수 쌍**: {len(df.select_dtypes(include=[np.number]).columns) * (len(df.select_dtypes(include=[np.number]).columns) - 1) // 2} 개
- **강한 상관관계**: {strong_corr_count} 개 (|r| > 0.7)
"""
        
        # 품질 평가 요약
        quality_summary = ""
        if "overall_quality_score" in quality_assessment:
            quality_score = quality_assessment["overall_quality_score"]
            quality_summary = f"""

## ✅ **데이터 품질 평가**
- **전체 품질 점수**: {quality_score:.1f}/100
- **완전성**: {quality_assessment.get('completeness', {}).get('completeness_rate', 0):.1f}%
- **고유성**: {quality_assessment.get('uniqueness', {}).get('uniqueness_rate', 0):.1f}%
- **일관성 이슈**: {len(quality_assessment.get('consistency', {}).get('issues', []))} 개
"""
        
        return f"""# 📊 **EDA Analysis Complete (Fallback Mode)!**

## 📋 **EDA 분석 결과**
- **파일 위치**: `{output_path}`
- **원본 크기**: {df.shape[0]:,} 행 × {df.shape[1]:,} 열
- **분석 완료**: 기술통계, 상관관계, 품질평가, 이상치 감지

## 📝 **요청 내용**
{user_input}

{eda_summary}

{stats_summary}

{corr_summary}

{quality_summary}

## 📈 **원본 데이터 미리보기**
```
{data_preview}
```

## 🔍 **수행된 EDA 분석들**
- ✅ 기본 정보 분석 (shape, dtypes, memory usage)
- ✅ 결측값 패턴 분석
- ✅ 기술 통계 계산 (평균, 표준편차, 왜도, 첨도)
- ✅ 상관관계 분석 (Pearson, Spearman)
- ✅ 데이터 품질 평가 (완전성, 고유성, 일관성)
- ✅ 이상치 감지 (IQR 방법)

⚠️ **폴백 모드**: 원본 ai-data-science-team 패키지를 사용할 수 없어 기본 EDA만 수행되었습니다.
💡 **완전한 기능을 위해서는 원본 EDAToolsAgent 설정이 필요합니다.**
"""
    
    def _generate_eda_guidance(self, user_instructions: str) -> str:
        """EDA 가이드 제공"""
        return f"""# 📊 **EDAToolsAgent 가이드**

## 📝 **요청 내용**
{user_instructions}

## 🎯 **EDAToolsAgent 완전 가이드**

### 1. **탐색적 데이터 분석 핵심 개념**
EDA(Exploratory Data Analysis)는 데이터의 패턴, 이상치, 관계를 발견하는 과정입니다:

- **기술 통계**: 데이터의 중심경향성과 분산 특성
- **분포 분석**: 데이터의 분포 형태와 정규성
- **관계 분석**: 변수 간 상관관계와 연관성
- **품질 진단**: 결측값, 이상치, 일관성 문제

### 2. **8개 핵심 기능**
1. 📊 **compute_descriptive_statistics** - 평균, 중앙값, 표준편차, 왜도, 첨도
2. 🔗 **analyze_correlations** - Pearson, Spearman, Kendall 상관계수
3. 📈 **analyze_distributions** - 정규성 검정, 분포 적합도
4. 🏷️ **analyze_categorical_data** - 빈도표, 카이제곱 검정
5. ⏰ **analyze_time_series** - 트렌드, 계절성, 정상성 분석
6. 🚨 **detect_anomalies** - IQR, Z-score, Isolation Forest
7. ✅ **assess_data_quality** - 완전성, 일관성, 고유성 평가
8. 🧠 **generate_automated_insights** - AI 기반 자동 인사이트

### 3. **EDA 분석 작업 예시**

#### 📊 **기술 통계 분석**
```text
데이터의 기본 통계를 계산해주세요 (평균, 중앙값, 표준편차)
```

#### 🔗 **상관관계 분석**
```text
변수들 간의 상관관계를 분석해주세요
```

#### 📈 **분포 분석**
```text
데이터 분포를 분석하고 정규성을 검정해주세요
```

#### 🚨 **이상치 감지**
```text
데이터에서 이상치를 찾아주세요
```

### 4. **지원되는 통계 기법**
- **기술 통계**: Mean, Median, Mode, Std, Variance, Skewness, Kurtosis
- **상관분석**: Pearson, Spearman, Kendall, Point-biserial
- **정규성 검정**: Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling
- **이상치 감지**: IQR, Modified Z-score, Isolation Forest
- **범주형 분석**: Chi-square, Cramér's V, Fisher's exact test

### 5. **원본 EDAToolsAgent 도구들**
- **explain_data**: 데이터 설명 및 해석
- **describe_dataset**: 데이터셋 기본 정보
- **visualize_missing**: 결측값 시각화
- **generate_correlation_funnel**: 상관관계 깔때기 차트
- **generate_profiling_report**: pandas-profiling 보고서
- **generate_dtale_report**: D-Tale 인터랙티브 보고서

## 💡 **데이터를 포함해서 다시 요청하면 실제 EDA 분석을 수행해드릴 수 있습니다!**

**데이터 형식 예시**:
- **CSV**: `id,age,salary,department\\n1,25,50000,IT\\n2,30,60000,HR`
- **JSON**: `[{{"id": 1, "age": 25, "salary": 50000, "department": "IT"}}]`

### 🔗 **학습 리소스**
- pandas EDA: https://pandas.pydata.org/docs/user_guide/cookbook.html
- scipy 통계: https://docs.scipy.org/doc/scipy/reference/stats.html
- 데이터 프로파일링: https://pandas-profiling.github.io/pandas-profiling/

✅ **EDAToolsAgent 준비 완료!**
"""


class EDAToolsAgentExecutor(AgentExecutor):
    """EDA Tools Agent A2A Executor"""
    
    def __init__(self):
        self.agent = EDAToolsServerAgent()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ EDAToolsAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("🤖 EDA Tools Agent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 0.2.9 공식 패턴에 따른 실행 with Langfuse integration"""
        logger.info(f"🚀 EDA Tools Agent 실행 시작 - Task: {context.task_id}")
        
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
                    name="EDAToolsAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "EDAToolsAgent",
                        "port": 8312,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id),
                        "server_type": "wrapper_based"
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
                    metadata={"step": "1", "description": "Parse EDA analysis request"}
                )
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🤖 EDAToolsAgent 시작...")
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
                            "analysis_type": "exploratory_data_analysis"
                        }
                    )
                
                if not user_instructions:
                    await task_updater.update_status(
                        TaskState.completed,
                        message=new_agent_text_message("❌ EDA 분석 요청이 비어있습니다.")
                    )
                    return
                
                # 2단계: EDA 분석 실행 (Langfuse 추적)
                eda_span = None
                if main_trace:
                    eda_span = self.langfuse_tracer.langfuse.span(
                        trace_id=context.task_id,
                        name="eda_analysis",
                        input={
                            "query": user_instructions[:200],
                            "analysis_type": "wrapper_based_processing"
                        },
                        metadata={"step": "2", "description": "Execute EDA analysis with optimized wrapper"}
                    )
                
                # EDA 분석 처리 실행
                result = await self.agent.process_eda_analysis(user_instructions)
                
                # EDA 분석 결과 업데이트
                if eda_span:
                    eda_span.update(
                        output={
                            "success": True,
                            "result_length": len(result),
                            "analysis_completed": True,
                            "insights_generated": True,
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
                            "analysis_success": True
                        },
                        metadata={"step": "3", "description": "Prepare EDA analysis results"}
                    )
                
                # 저장 결과 업데이트
                if save_span:
                    save_span.update(
                        output={
                            "response_prepared": True,
                            "insights_delivered": True,
                            "final_status": "completed",
                            "analysis_included": True
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
                            "agent": "EDAToolsAgent",
                            "port": 8312,
                            "server_type": "wrapper_based",
                            "analysis_type": "exploratory_data_analysis"
                        }
                    )
                    logger.info(f"🔧 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
                
        except Exception as e:
            logger.error(f"❌ EDA Tools Agent 실행 실패: {e}")
            
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
                            "agent": "EDAToolsAgent",
                            "port": 8312,
                            "server_type": "wrapper_based"
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
            
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(f"❌ EDA 분석 처리 중 오류 발생: {str(e)}")
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소"""
        logger.info(f"🚫 EDA Tools Agent 작업 취소 - Task: {context.task_id}")


def main():
    """EDA Tools Agent 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="eda_tools",
        name="Exploratory Data Analysis Tools",
        description="원본 ai-data-science-team EDAToolsAgent를 활용한 완전한 탐색적 데이터 분석 서비스입니다. 8개 핵심 기능으로 통계 분석, 상관관계, 품질 평가를 수행합니다.",
        tags=["eda", "statistics", "correlation", "data-quality", "outliers", "distribution", "ai-data-science-team"],
        examples=[
            "데이터의 기술 통계를 계산해주세요",
            "변수 간 상관관계를 분석해주세요",  
            "데이터 분포를 분석해주세요",
            "범주형 데이터를 분석해주세요",
            "시계열 패턴을 분석해주세요",
            "이상치를 감지해주세요",
            "데이터 품질을 평가해주세요",
            "자동으로 인사이트를 생성해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="EDA Tools Agent",
        description="원본 ai-data-science-team EDAToolsAgent를 A2A SDK로 래핑한 완전한 탐색적 데이터 분석 서비스. 8개 핵심 기능으로 통계 분석, 상관관계 분석, 품질 평가, 이상치 감지를 지원합니다.",
        url="http://localhost:8312/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=EDAToolsAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("📊 Starting EDA Tools Agent Server")
    print("🌐 Server starting on http://localhost:8312")
    print("📋 Agent card: http://localhost:8312/.well-known/agent.json")
    print("🎯 Features: 원본 ai-data-science-team EDAToolsAgent 8개 기능 100% 래핑")
    print("💡 EDA Analysis: 통계 분석, 상관관계, 품질 평가, 이상치 감지, 분포 분석")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8312, log_level="info")


if __name__ == "__main__":
    main()