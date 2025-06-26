import asyncio
import logging
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
import uvicorn
import click

# A2A SDK 공식 컴포넌트 사용 (공식 Hello World Agent 패턴)
import uuid
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, Message, Task, AgentCapabilities
from a2a.utils.message import new_agent_text_message, get_message_text

from langchain_ollama import ChatOllama

# Import core modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.data_manager import DataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data manager instance
data_manager = DataManager()

# 1. Define the core agent (공식 Hello World Agent 패턴)
class PandasDataAnalysisAgent:
    """Pandas 데이터 분석 에이전트 (공식 Hello World Agent 패턴)"""
    
    async def invoke(self, user_input: str = "") -> str:
        """
        데이터 분석 수행 (사용자 지시사항에 따른 맞춤형 분석)
        """
        logger.info(f"🎯 PandasDataAnalysisAgent.invoke() called with: {user_input}")
        
        try:
            # 사용 가능한 데이터프레임 확인
            available_dfs = data_manager.list_dataframes()
            logger.info(f"💾 Available dataframes: {available_dfs}")
            
            if not available_dfs:
                result_text = """❌ **데이터 없음**

**문제**: 아직 업로드된 데이터셋이 없습니다.

**해결방법:**
1. 🔄 **데이터 로더** 페이지로 이동
2. 📁 CSV, Excel 등의 데이터 파일 업로드  
3. 📊 다시 돌아와서 데이터 분석 요청

**현재 사용 가능한 데이터셋**: 없음
"""
                return result_text
            
            # 첫 번째 데이터프레임 사용
            df_id = available_dfs[0]
            df = data_manager.get_dataframe(df_id)
            
            if df is None:
                return "❌ 데이터프레임을 로드할 수 없습니다."
            
            logger.info(f"📊 Analyzing dataframe: {df_id}, shape: {df.shape}")
            
            # 사용자 지시사항 분석하여 적절한 분석 수행
            return await self._perform_targeted_analysis(df, df_id, user_input)
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_data: {e}", exc_info=True)
            return f"❌ 분석 중 오류가 발생했습니다: {str(e)}"
    
    async def _perform_targeted_analysis(self, df, df_id: str, user_instruction: str) -> str:
        """LLM이 지시사항을 이해하고 적절한 분석을 자동으로 선택하여 수행"""
        
        # LLM에게 지시사항을 해석하고 적절한 분석을 요청
        analysis_director_prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자의 요청을 분석하여 가장 적절한 데이터 분석을 수행해주세요.

데이터셋 정보:
- 이름: {df_id}
- 크기: {df.shape[0]:,}행 × {df.shape[1]}열
- 컬럼: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}

사용자 요청: "{user_instruction}"

위 요청에 가장 적합한 분석을 수행하고, 다음 형식으로 응답해주세요:

# 📊 **[분석 제목]**

**요청**: {user_instruction}
**데이터셋**: {df_id}
**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## [적절한 섹션들]

[실제 데이터를 활용한 구체적인 분석 결과]

사용자의 요청을 정확히 이해하고, 데이터의 특성을 고려하여 가장 유용한 분석을 제공해주세요.
바이너리 타겟 변수가 있다면 해당 변수를 중심으로 한 분석을, 일반 데이터라면 적절한 EDA를 수행해주세요.
"""

        try:
            # LLM 호출을 위한 설정
            from langchain_ollama import ChatOllama
            
            # Ollama LLM 초기화
            llm = ChatOllama(
                model="qwen2.5:latest",
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            
            # 데이터 컨텍스트 준비
            data_context = self._prepare_data_context(df)
            
            # 최종 프롬프트 구성
            final_prompt = f"""{analysis_director_prompt}

데이터 컨텍스트:
{data_context}

사용자가 요청한 구체적인 분석을 데이터에 기반하여 수행해주세요."""

            # LLM에게 분석 요청
            response = await llm.ainvoke(final_prompt)
            
            # 응답 텍스트 추출
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"❌ LLM 분석 중 오류: {e}")
            # 폴백: 기본 분석
            return self._generate_comprehensive_analysis(df, df_id, user_instruction)
    
    def _prepare_data_context(self, df) -> str:
        """LLM이 데이터를 이해할 수 있도록 핵심 컨텍스트 정보 준비"""
        context_parts = []
        
        # 기본 정보
        context_parts.append(f"데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 컬럼 정보와 데이터 타입
        context_parts.append("컬럼 정보:")
        for col, dtype in zip(df.columns, df.dtypes):
            sample_values = df[col].dropna().head(3).tolist()
            context_parts.append(f"- {col} ({dtype}): 예시값 {sample_values}")
        
        # 결측값 정보
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            context_parts.append("\n결측값:")
            for col, count in missing_info.items():
                if count > 0:
                    context_parts.append(f"- {col}: {count}개 ({count/len(df)*100:.1f}%)")
        
        # 수치형 데이터 기본 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context_parts.append("\n수치형 변수 요약:")
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:3]:  # 처음 3개만
                if col in desc.columns:
                    context_parts.append(f"- {col}: 평균 {desc.loc['mean', col]:.2f}, 범위 {desc.loc['min', col]:.2f}~{desc.loc['max', col]:.2f}")
        
        # 범주형 데이터 정보
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            context_parts.append("\n범주형 변수 정보:")
            for col in categorical_cols[:3]:  # 처음 3개만
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                context_parts.append(f"- {col}: {unique_count}개 고유값, 상위값 {dict(top_values)}")
        
        # 바이너리 타겟 컬럼 자동 감지 (범용적)
        binary_target_info = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if set(unique_vals) == {0, 1}:
                    positive_rate = df[col].mean() * 100
                    binary_target_info.append(f"{col}: {positive_rate:.1f}% 양성")
        
        if binary_target_info:
            context_parts.append(f"\n바이너리 타겟: {', '.join(binary_target_info)}")
        
        return "\n".join(context_parts)

    def _generate_data_overview(self, df, df_id: str, instruction: str) -> str:
        """데이터 구조 및 개요 분석"""
        analysis_parts = []
        
        analysis_parts.append(f"# 📋 **데이터 구조 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 데이터셋 기본 정보
        analysis_parts.append("## 📊 **데이터셋 기본 정보**")
        analysis_parts.append(f"- **총 행 수**: {df.shape[0]:,}개")
        analysis_parts.append(f"- **총 열 수**: {df.shape[1]}개")
        analysis_parts.append(f"- **메모리 사용량**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 컬럼별 데이터 타입
        analysis_parts.append("\n## 🔍 **컬럼별 상세 정보**")
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            analysis_parts.append(f"{i}. **{col}** ({dtype})")
            analysis_parts.append(f"   - 유효값: {non_null_count:,}개 ({non_null_count/len(df)*100:.1f}%)")
            if null_count > 0:
                analysis_parts.append(f"   - 결측값: {null_count:,}개 ({null_count/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_descriptive_stats(self, df, df_id: str, instruction: str) -> str:
        """기술통계 및 분포 분석"""
        analysis_parts = []
        
        analysis_parts.append(f"# 📈 **기술통계 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 수치형 변수 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("## 🔢 **수치형 변수 기술통계**")
            desc = df[numeric_cols].describe()
            for col in numeric_cols:
                if col in desc.columns:
                    analysis_parts.append(f"\n**{col}**:")
                    analysis_parts.append(f"- 평균: {desc.loc['mean', col]:.2f}")
                    analysis_parts.append(f"- 중앙값: {desc.loc['50%', col]:.2f}")
                    analysis_parts.append(f"- 표준편차: {desc.loc['std', col]:.2f}")
                    analysis_parts.append(f"- 최솟값: {desc.loc['min', col]:.2f}")
                    analysis_parts.append(f"- 최댓값: {desc.loc['max', col]:.2f}")
        
        # 범주형 변수 통계
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n## 📝 **범주형 변수 빈도 분석**")
            for col in categorical_cols[:3]:  # 상위 3개만
                value_counts = df[col].value_counts().head(5)
                analysis_parts.append(f"\n**{col} (상위 5개 값):**")
                for value, count in value_counts.items():
                    analysis_parts.append(f"- {value}: {count:,}개 ({count/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_correlation_analysis(self, df, df_id: str, instruction: str) -> str:
        """상관관계 분석"""
        analysis_parts = []
        
        analysis_parts.append(f"# 🔗 **상관관계 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 수치형 변수들 간의 상관관계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            analysis_parts.append("## 📊 **수치형 변수 상관관계**")
            
            # 강한 상관관계 찾기 (|r| > 0.5)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        strong_correlations.append((col1, col2, corr_val))
            
            if strong_correlations:
                analysis_parts.append("\n**강한 상관관계 (|r| > 0.5):**")
                for col1, col2, corr_val in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                    analysis_parts.append(f"- **{col1}** ↔ **{col2}**: {corr_val:.3f}")
            else:
                analysis_parts.append("\n강한 상관관계(|r| > 0.5)를 보이는 변수 쌍이 없습니다.")
            
            # 상관관계 매트릭스 요약
            analysis_parts.append("\n**전체 상관관계 매트릭스:**")
            for col in numeric_cols[:4]:  # 상위 4개 변수만
                analysis_parts.append(f"\n**{col}과의 상관관계:**")
                correlations = corr_matrix[col].drop(col).sort_values(key=abs, ascending=False)
                for other_col, corr_val in correlations.head(3).items():
                    analysis_parts.append(f"- {other_col}: {corr_val:.3f}")
        else:
            analysis_parts.append("## ⚠️ **상관관계 분석 불가**")
            analysis_parts.append("수치형 변수가 2개 미만이어서 상관관계 분석을 수행할 수 없습니다.")
        
        return "\n".join(analysis_parts)
    
    def _generate_trend_analysis(self, df, df_id: str, instruction: str) -> str:
        """트렌드 및 패턴 분석"""
        analysis_parts = []
        
        analysis_parts.append(f"# 📈 **트렌드 및 패턴 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 범용적인 패턴 분석
        
        # 1. 바이너리 타겟 변수 패턴 분석 (범용적)
        binary_target_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if set(unique_vals) == {0, 1}:
                    binary_target_cols.append(col)
        
        if binary_target_cols:
            analysis_parts.append("## 🎯 **바이너리 타겟 변수 패턴 분석**")
            
            for target_col in binary_target_cols:
                positive_rate = df[target_col].mean() * 100
                analysis_parts.append(f"\n**{target_col} 분포:**")
                analysis_parts.append(f"- 양성(1): {df[target_col].sum():,}개 ({positive_rate:.1f}%)")
                analysis_parts.append(f"- 음성(0): {(df[target_col] == 0).sum():,}개 ({100-positive_rate:.1f}%)")
                
                # 범주형 변수와의 관계 분석
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for cat_col in categorical_cols[:2]:  # 상위 2개 범주형 변수
                    analysis_parts.append(f"\n**{cat_col}별 {target_col} 패턴:**")
                    group_stats = df.groupby(cat_col)[target_col].agg(['count', 'sum', 'mean'])
                    for category in group_stats.index[:4]:  # 상위 4개 카테고리
                        total = group_stats.loc[category, 'count']
                        positive = group_stats.loc[category, 'sum']
                        rate = group_stats.loc[category, 'mean'] * 100
                        analysis_parts.append(f"- **{category}**: {positive}/{total}개 ({rate:.1f}%)")
        
        # 2. 범주형 변수 분포 패턴
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n## 📊 **범주형 변수 분포 패턴**")
            for col in categorical_cols[:3]:  # 상위 3개만
                value_counts = df[col].value_counts()
                total_unique = df[col].nunique()
                analysis_parts.append(f"\n**{col} ({total_unique}개 고유값):**")
                for i, (value, count) in enumerate(value_counts.head(4).items()):
                    analysis_parts.append(f"{i+1}. {value}: {count:,}개 ({count/len(df)*100:.1f}%)")
        
        # 3. 수치형 변수 분포 패턴
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("\n## 📈 **수치형 변수 분포 특성**")
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:3]:  # 상위 3개만
                if col in desc.columns:
                    skewness = df[col].skew()
                    outlier_threshold = desc.loc['75%', col] + 1.5 * (desc.loc['75%', col] - desc.loc['25%', col])
                    outliers = (df[col] > outlier_threshold).sum()
                    
                    analysis_parts.append(f"\n**{col}:**")
                    analysis_parts.append(f"- 범위: {desc.loc['min', col]:.2f} ~ {desc.loc['max', col]:.2f}")
                    analysis_parts.append(f"- 분포: {'왼쪽 치우침' if skewness > 1 else '오른쪽 치우침' if skewness < -1 else '정규분포에 가까움'}")
                    if outliers > 0:
                        analysis_parts.append(f"- 이상값: {outliers}개 ({outliers/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_insights_summary(self, df, df_id: str, instruction: str) -> str:
        """핵심 인사이트 및 요약"""
        analysis_parts = []
        
        analysis_parts.append(f"# 💡 **핵심 인사이트 요약 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 데이터 품질 인사이트
        total_entries = len(df)
        missing_data = df.isnull().sum().sum()
        completeness = (1 - missing_data / (total_entries * len(df.columns))) * 100
        
        analysis_parts.append("## 🔍 **핵심 발견사항**")
        
        analysis_parts.append(f"\n**1. 데이터 품질**")
        analysis_parts.append(f"- 데이터 완성도: {completeness:.1f}%")
        analysis_parts.append(f"- 총 {total_entries:,}개 관측값으로 {'충분한' if total_entries > 1000 else '제한적인'} 분석 가능")
        
        # 범용적인 데이터 인사이트
        analysis_parts.append(f"\n**2. 핵심 데이터 인사이트**")
        
        # 바이너리 타겟 변수 인사이트
        binary_targets = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if set(unique_vals) == {0, 1}:
                    positive_rate = df[col].mean() * 100
                    binary_targets.append((col, positive_rate))
        
        if binary_targets:
            for target_col, rate in binary_targets:
                balance_status = "균형잡힌" if 40 <= rate <= 60 else "불균형한"
                analysis_parts.append(f"- {target_col}: {rate:.1f}% 양성률로 {balance_status} 분포")
        
        # 결측값 패턴 인사이트
        missing_rates = df.isnull().mean() * 100
        high_missing = missing_rates[missing_rates > 20]
        if len(high_missing) > 0:
            analysis_parts.append(f"- 결측값 주의: {list(high_missing.index)} 컬럼의 결측률이 20% 이상")
        
        # 범주형 변수 다양성 인사이트
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            high_cardinality = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.1]
            if high_cardinality:
                analysis_parts.append(f"- 고유값 과다: {high_cardinality} 컬럼은 범주 수가 매우 높음")
        
        # 데이터 구조 인사이트
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        analysis_parts.append(f"\n**3. 데이터 구조 특징**")
        analysis_parts.append(f"- 수치형 변수 {len(numeric_cols)}개, 범주형 변수 {len(categorical_cols)}개")
        analysis_parts.append(f"- 다양한 관점의 분석이 가능한 {'균형잡힌' if len(numeric_cols) > 2 and len(categorical_cols) > 2 else '단순한'} 구조")
        
        # 추천사항
        analysis_parts.append(f"\n## 📋 **추천 후속 분석**")
        analysis_parts.append("1. **시각화**: 주요 패턴을 그래프로 표현")
        analysis_parts.append("2. **예측 모델링**: 타겟 변수 예측 모델 구축")
        analysis_parts.append("3. **세분화 분석**: 특정 그룹별 상세 분석")
        analysis_parts.append("4. **이상값 분석**: 특이한 케이스 탐지")
        
        return "\n".join(analysis_parts)
    
    def _generate_comprehensive_analysis(self, df, df_id: str, instruction: str) -> str:
        """종합 분석 (기본값)"""
        analysis_parts = []
        
        analysis_parts.append("# 📊 **종합 데이터 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**크기**: {df.shape[0]:,}행 × {df.shape[1]}열")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 데이터 개요
        analysis_parts.append("## 📋 **데이터 개요**")
        analysis_parts.append("**컬럼 정보:**")
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            analysis_parts.append(f"{i}. **{col}** ({dtype})")
        
        # 기본 통계
        analysis_parts.append("\n## 📈 **기본 통계**")
        desc = df.describe()
        if not desc.empty:
            analysis_parts.append("**수치형 변수 통계:**")
            for col in desc.columns[:3]:  # 처음 3개 컬럼만
                analysis_parts.append(f"- **{col}**: 평균 {desc.loc['mean', col]:.2f}, 표준편차 {desc.loc['std', col]:.2f}")
        
        # 결측치 분석
        missing = df.isnull().sum()
        if missing.sum() > 0:
            analysis_parts.append("\n## ⚠️ **결측치 분석**")
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    analysis_parts.append(f"- **{col}**: {count}개 ({pct:.1f}%)")
        
        # 범용적인 타겟 변수 분석
        binary_target_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if set(unique_vals) == {0, 1}:
                    binary_target_cols.append(col)
        
        if binary_target_cols:
            analysis_parts.append("\n## 🎯 **타겟 변수 분석**")
            for target_col in binary_target_cols:
                positive_rate = df[target_col].mean() * 100
                analysis_parts.append(f"- **{target_col} 양성률**: {positive_rate:.1f}%")
                
                # 범주형 변수와의 관계
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for cat_col in categorical_cols[:2]:  # 상위 2개만
                    if len(df.groupby(cat_col)[target_col].mean()) > 1:
                        group_means = df.groupby(cat_col)[target_col].mean() * 100
                        top_categories = group_means.head(3)
                        analysis_parts.append(f"- **{cat_col}별 {target_col}**: {dict(top_categories.round(1))}")
        
        # 범주형 변수 분포
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n## 📊 **범주형 변수 분포**")
            for cat_col in categorical_cols[:2]:  # 상위 2개만
                value_counts = df[cat_col].value_counts()
                analysis_parts.append(f"- **{cat_col}**: {dict(value_counts.head(3))}")
        
        # 수치형 변수 분포
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("\n## 📈 **수치형 변수 특성**")
            for num_col in numeric_cols[:3]:  # 상위 3개만
                if num_col not in binary_target_cols:  # 바이너리 타겟 제외
                    skewness = df[num_col].skew()
                    outliers_count = len(df[df[num_col] > df[num_col].quantile(0.75) + 1.5 * (df[num_col].quantile(0.75) - df[num_col].quantile(0.25))])
                    analysis_parts.append(f"- **{num_col}**: {'정규분포' if abs(skewness) < 1 else '치우친 분포'}, 이상값 {outliers_count}개")
        
        # 추천사항
        analysis_parts.append("\n## 💡 **분석 추천사항**")
        analysis_parts.append("1. 🔍 **상관관계 분석**: 수치형 변수들 간의 관계 탐색")
        analysis_parts.append("2. 📊 **시각화**: 히스토그램, 상자그림 등으로 분포 확인")
        analysis_parts.append("3. 🎯 **세분화 분석**: 카테고리별 상세 분석 수행")
        
        return "\n".join(analysis_parts)

# 2. AgentExecutor 구현 (공식 Hello World Agent 패턴)
class PandasAgentExecutor(AgentExecutor):
    """공식 Hello World Agent 패턴을 사용하는 AgentExecutor"""
    
    def __init__(self):
        self.agent = PandasDataAnalysisAgent()
        logger.info("🔧 PandasAgentExecutor 초기화 완료")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A SDK 표준 실행 (공식 Hello World Agent 패턴)"""
        logger.info("🎯 PandasAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출 (공식 패턴)
            user_message = context.get_user_input()
            logger.info(f"📝 사용자 입력: {user_message}")
            
            # 에이전트 실행 (공식 패턴)
            result = await self.agent.invoke(user_message)
            
            # 결과 전송 (공식 패턴 - 중요: await 추가!)
            message = new_agent_text_message(result)
            await event_queue.enqueue_event(message)
            
            logger.info("✅ Task completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in execute: {e}", exc_info=True)
            error_message = new_agent_text_message(f"❌ 실행 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Task 취소 처리 (공식 Hello World Agent 패턴)"""
        logger.info("🛑 PandasAgentExecutor.cancel() 호출됨")
        raise Exception("Cancel not supported")

# 3. Agent Card 생성 (공식 A2A 표준 메타데이터)
def create_agent_card() -> AgentCard:
    """A2A 표준 Agent Card 생성 (공식 Hello World Agent 패턴)"""
    
    # 기본 스킬 정의 (공식 패턴)
    skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas Data Analysis",
        description="Performs comprehensive data analysis on uploaded datasets using pandas",
        tags=["data", "analysis", "pandas", "statistics", "EDA"],
        examples=["Analyze my data", "What insights can you find?", "Show me data statistics"]
    )
    
    return AgentCard(
        name="Pandas Data Analyst",
        description="A comprehensive data analysis agent powered by pandas and AI",
        url="http://localhost:10001/",
        version="2.0.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill]
    )

# 4. Wire everything together (공식 Hello World Agent 패턴)
@click.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=10001, help='Port to bind to')
def main(host: str, port: int):
    """A2A 표준 Pandas 서버 실행 (공식 Hello World Agent 패턴)"""
    
    logger.info("🚀 Starting Pandas A2A Server...")
    
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # RequestHandler 초기화 (공식 패턴)
    request_handler = DefaultRequestHandler(
        agent_executor=PandasAgentExecutor(),
        task_store=InMemoryTaskStore()
    )
    
    # A2A Starlette Application 생성 (공식 패턴)
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info(f"🌐 Server starting at http://{host}:{port}")
    logger.info("📋 Agent Card available at /.well-known/agent.json")
    
    # Uvicorn으로 서버 실행
    uvicorn.run(a2a_app.build(), host=host, port=port)

if __name__ == "__main__":
    main() 