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
import pickle
from pathlib import Path

# A2A SDK 공식 컴포넌트 사용 (공식 Hello World Agent 패턴)
import uuid
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
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
    """스트리밍 지원 범용 데이터 분석 에이전트"""
    
    def __init__(self):
        self.data_cache = {}

    async def invoke(self, user_input: str = "", stream: bool = False) -> str:
        """분석 수행 - 스트리밍 지원"""
        try:
            logger.info(f"📊 데이터 분석 요청: {user_input[:100]}...")
            
            # 데이터 로드
            df, df_id = await self._load_latest_dataset()
            if df is None:
                return "❌ 분석할 데이터가 없습니다. 먼저 데이터를 업로드해주세요."
            
            logger.info(f"✅ 데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
            
            if stream:
                # 스트리밍 모드
                return await self._perform_streaming_analysis(df, df_id, user_input)
            else:
                # 일반 모드 (기존)
                return await self._perform_targeted_analysis(df, df_id, user_input)
                
        except Exception as e:
            logger.error(f"❌ 분석 중 오류: {e}")
            return f"분석 중 오류가 발생했습니다: {str(e)}"
    
    async def _load_latest_dataset(self):
        """최신 데이터셋 로드"""
        try:
            # artifacts/data/shared_dataframes 디렉토리에서 데이터 찾기
            data_dir = Path("artifacts/data/shared_dataframes")
            if not data_dir.exists():
                logger.warning(f"데이터 디렉토리가 존재하지 않음: {data_dir}")
                return None, None
            
            # pickle 파일들 찾기
            pickle_files = list(data_dir.glob("*.pkl"))
            if not pickle_files:
                logger.warning(f"pickle 파일이 없음: {data_dir}")
                return None, None
            
            # 가장 최근 파일 선택
            latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"로드할 파일: {latest_file}")
            
            # 데이터 로드
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)
            
            # 데이터 타입 확인 및 변환
            if isinstance(data, dict):
                # dict인 경우 DataFrame으로 변환 시도
                import pandas as pd
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'df' in data:
                    df = pd.DataFrame(data['df'])
                else:
                    # dict의 첫 번째 값이 DataFrame이거나 dict인지 확인
                    first_key = list(data.keys())[0]
                    if hasattr(data[first_key], 'shape'):
                        df = data[first_key]
                    else:
                        df = pd.DataFrame(data)
            else:
                df = data
            
            # DataFrame인지 최종 확인
            if not hasattr(df, 'shape'):
                logger.error(f"로드된 데이터가 DataFrame이 아님: {type(df)}")
                return None, None
            
            df_id = latest_file.stem.replace('.csv', '')  # .csv.pkl -> .csv 제거
            logger.info(f"데이터 로드 성공: {df.shape} - {df_id}")
            
            return df, df_id
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            return None, None
    
    async def _perform_targeted_analysis(self, df, df_id: str, user_instruction: str) -> str:
        """LLM이 지시사항을 이해하고 적절한 분석을 자동으로 선택하여 수행 (기존 방식)"""
        
        try:
            # LLM 호출을 위한 설정
            from langchain_ollama import ChatOllama
            
            # Ollama LLM 초기화 - gemma3 모델 사용
            llm = ChatOllama(
                model="gemma3:latest",
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            
            # 분석 유형 선택을 위한 프롬프트
            analysis_selector_prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자의 요청을 분석하여 가장 적절한 분석 유형을 선택해주세요.

사용자 요청: "{user_instruction}"

다음 중 하나를 선택해주세요:
1. data_overview - 데이터 구조, 변수 정보, 기본 개요 분석
2. descriptive_stats - 기술통계, 분포, 요약 통계 분석  
3. correlation_analysis - 변수 간 상관관계, 관계성 분석
4. pattern_analysis - 패턴, 트렌드, 분포 특성 분석
5. insights_summary - 핵심 인사이트, 결론, 추천사항

오직 숫자만 응답해주세요 (1, 2, 3, 4, 또는 5):
"""
            
            # LLM에게 분석 유형 선택 요청
            response = await llm.ainvoke(analysis_selector_prompt)
            
            # 응답 텍스트 추출
            if hasattr(response, 'content'):
                selection = response.content.strip()
            else:
                selection = str(response).strip()
            
            logger.info(f"🎯 LLM이 선택한 분석 유형: {selection}")
            
            # 선택된 분석 함수 실행
            if selection == "1":
                return self._generate_data_overview(df, df_id, user_instruction)
            elif selection == "2":
                return self._generate_descriptive_stats(df, df_id, user_instruction)
            elif selection == "3":
                return self._generate_correlation_analysis(df, df_id, user_instruction)
            elif selection == "4":
                return self._generate_pattern_analysis(df, df_id, user_instruction)
            elif selection == "5":
                return self._generate_insights_summary(df, df_id, user_instruction)
            else:
                logger.warning(f"⚠️ 알 수 없는 선택: {selection}, 종합 분석으로 폴백")
                return await self._generate_comprehensive_streaming_analysis(df, df_id, user_instruction)
                
        except Exception as e:
            logger.error(f"❌ LLM 분석 중 오류: {e}")
            # 폴백: 기본 분석
            return await self._generate_comprehensive_streaming_analysis(df, df_id, user_instruction)
    
    async def _perform_streaming_analysis(self, df, df_id: str, user_instruction: str) -> str:
        """실시간 스트리밍 분석"""
        try:
            from langchain_ollama import ChatOllama
            
            # Ollama LLM 초기화
            llm = ChatOllama(
                model="gemma3:latest",
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            
            # 스트리밍 응답 생성
            streaming_response = []
            
            # 1. 즉시 시작 메시지
            start_msg = f"""# 📊 **실시간 데이터 분석 시작**

**요청**: {user_instruction}
**데이터셋**: {df_id} ({df.shape[0]:,}행 × {df.shape[1]}열)
**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔄 **분석 진행 중...**

"""
            streaming_response.append(start_msg)
            
            # 2. 데이터 기본 정보 (즉시 제공)
            basic_info = f"""## 📋 **데이터 기본 정보**

| 항목 | 값 |
|------|-----|
| 📏 데이터 크기 | **{df.shape[0]:,}** 행 × **{df.shape[1]}** 열 |
| 🔢 수치형 변수 | **{len(df.select_dtypes(include=[np.number]).columns)}개** |
| 📝 범주형 변수 | **{len(df.select_dtypes(include=['object', 'category']).columns)}개** |
| ✅ 완성도 | **{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%** |
| 💾 메모리 사용량 | **{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB** |

"""
            streaming_response.append(basic_info)
            
            # 3. LLM 분석 선택 (약간의 지연)
            await asyncio.sleep(1)  # 실제 처리 시뮬레이션
            
            analysis_selector_prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자의 요청을 분석하여 가장 적절한 분석 유형을 선택해주세요.

사용자 요청: "{user_instruction}"

다음 중 하나를 선택해주세요:
1. comprehensive - 종합적인 EDA 분석
2. descriptive_stats - 기술통계 및 분포 분석  
3. correlation_analysis - 변수 간 상관관계 분석
4. pattern_analysis - 패턴 및 트렌드 분석
5. data_quality - 데이터 품질 분석

오직 숫자만 응답해주세요 (1, 2, 3, 4, 또는 5):
"""
            
            response = await llm.ainvoke(analysis_selector_prompt)
            selection = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 4. 선택된 분석 수행 알림
            analysis_types = {
                "1": "종합 EDA 분석",
                "2": "기술통계 분석", 
                "3": "상관관계 분석",
                "4": "패턴 분석",
                "5": "데이터 품질 분석"
            }
            
            selected_analysis = analysis_types.get(selection, "종합 EDA 분석")
            
            progress_msg = f"""🎯 **분석 유형 선택 완료**: {selected_analysis}

🔄 **상세 분석 수행 중...**

"""
            streaming_response.append(progress_msg)
            
            # 5. 실제 분석 수행 (점진적 결과 제공)
            await asyncio.sleep(1)  # 분석 처리 시뮬레이션
            
            if selection == "1":
                detailed_analysis = await self._generate_comprehensive_streaming_analysis(df, df_id, user_instruction)
            elif selection == "2":
                detailed_analysis = self._generate_descriptive_stats(df, df_id, user_instruction)
            elif selection == "3":
                detailed_analysis = self._generate_correlation_analysis(df, df_id, user_instruction)
            elif selection == "4":
                detailed_analysis = self._generate_pattern_analysis(df, df_id, user_instruction)
            elif selection == "5":
                detailed_analysis = self._generate_data_overview(df, df_id, user_instruction)
            else:
                detailed_analysis = await self._generate_comprehensive_streaming_analysis(df, df_id, user_instruction)
            
            streaming_response.append(detailed_analysis)
            
            # 6. 완료 메시지
            completion_msg = f"""

---

✅ **분석 완료!**  
🕐 **총 처리 시간**: ~3초  
🔧 **분석 엔진**: 스트리밍 지원 범용 AI 데이터 사이언스 에이전트

"""
            streaming_response.append(completion_msg)
            
            return "".join(streaming_response)
            
        except Exception as e:
            logger.error(f"❌ 스트리밍 분석 중 오류: {e}")
            return f"스트리밍 분석 중 오류가 발생했습니다: {str(e)}"
    
    async def _generate_comprehensive_streaming_analysis(self, df, df_id: str, instruction: str) -> str:
        """종합 분석을 스트리밍으로 제공"""
        
        analysis_parts = []
        
        # 변수 유형 분석
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        analysis_parts.append("## 🔍 **변수 유형별 분석**\n")
        
        # 수치형 변수 분석
        if len(numeric_cols) > 0:
            analysis_parts.append("### 📊 **수치형 변수 분석**")
            desc = df[numeric_cols].describe()
            
            for i, col in enumerate(numeric_cols[:5], 1):  # 상위 5개
                if col in desc.columns:
                    var_name = self._get_generic_column_name(df, col, i)
                    skew_val = df[col].skew()
                    distribution = "정규분포에 가까움" if abs(skew_val) < 0.5 else "좌편향" if skew_val > 0.5 else "우편향"
                    
                    analysis_parts.append(f"\n**{var_name}**:")
                    analysis_parts.append(f"- 평균: {desc.loc['mean', col]:.2f}, 중앙값: {desc.loc['50%', col]:.2f}")
                    analysis_parts.append(f"- 범위: {desc.loc['min', col]:.2f} ~ {desc.loc['max', col]:.2f}")
                    analysis_parts.append(f"- 분포 특성: {distribution}")
                    analysis_parts.append(f"- 고유값: {df[col].nunique()}개")
        
        # 범주형 변수 분석
        if len(categorical_cols) > 0:
            analysis_parts.append("\n### 📝 **범주형 변수 분석**")
            
            for i, col in enumerate(categorical_cols[:5], 1):  # 상위 5개
                var_name = self._get_generic_column_name(df, col, i)
                unique_count = df[col].nunique()
                value_counts = df[col].value_counts().head(3)
                
                analysis_parts.append(f"\n**{var_name}**:")
                analysis_parts.append(f"- 고유값: {unique_count}개")
                analysis_parts.append(f"- 상위 3개 값:")
                for value, count in value_counts.items():
                    analysis_parts.append(f"  - {value}: {count}개 ({count/len(df)*100:.1f}%)")
        
        # 데이터 품질 분석
        analysis_parts.append("\n## 🔍 **데이터 품질 분석**\n")
        
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            analysis_parts.append("### ⚠️ **결측값 분석**")
            missing_vars = missing_data[missing_data > 0]
            for i, (col, count) in enumerate(missing_vars.items(), 1):
                var_name = self._get_generic_column_name(df, col, i)
                analysis_parts.append(f"- **{var_name}**: {count}개 ({count/len(df)*100:.1f}%)")
        else:
            analysis_parts.append("✅ **결측값 없음**: 모든 변수가 완전합니다.")
        
        # 관계 분석 (수치형 변수가 2개 이상인 경우)
        if len(numeric_cols) > 1:
            analysis_parts.append("\n## 🔗 **변수 간 관계 분석**\n")
            
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1_name = self._get_generic_column_name(df, corr_matrix.columns[i])
                        col2_name = self._get_generic_column_name(df, corr_matrix.columns[j])
                        strong_correlations.append((col1_name, col2_name, corr_val))
            
            if strong_correlations:
                analysis_parts.append("### 📈 **강한 상관관계 (|r| > 0.5)**")
                for var1, var2, corr_val in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    direction = "양의" if corr_val > 0 else "음의"
                    analysis_parts.append(f"- **{var1}** ↔ **{var2}**: {direction} 상관관계 ({corr_val:.3f})")
            else:
                analysis_parts.append("📊 **중간 정도의 상관관계**: 변수들 간에 강한 선형 관계는 없습니다.")
        
        # 핵심 인사이트
        analysis_parts.append("\n## 💡 **핵심 인사이트**\n")
        
        total_entries = len(df)
        completeness = (1 - df.isnull().sum().sum() / (total_entries * len(df.columns))) * 100
        
        analysis_parts.append(f"1. **데이터 규모**: {total_entries:,}개 관측값으로 {'충분한' if total_entries > 1000 else '적절한' if total_entries > 100 else '제한적인'} 분석이 가능합니다.")
        analysis_parts.append(f"2. **데이터 품질**: {completeness:.1f}%의 완성도로 {'우수한' if completeness > 95 else '양호한' if completeness > 85 else '개선이 필요한'} 수준입니다.")
        analysis_parts.append(f"3. **변수 구성**: 수치형 {len(numeric_cols)}개, 범주형 {len(categorical_cols)}개로 {'균형잡힌' if len(numeric_cols) > 0 and len(categorical_cols) > 0 else '단순한'} 구조입니다.")
        
        if len(numeric_cols) > 2:
            analysis_parts.append("4. **분석 가능성**: 다양한 통계 분석과 머신러닝 모델링이 가능합니다.")
        
        # 추천 후속 분석
        analysis_parts.append("\n## 📋 **추천 후속 분석**\n")
        analysis_parts.append("1. **시각화**: 분포도, 상관관계 히트맵, 박스플롯 생성")
        analysis_parts.append("2. **고급 통계**: 가설 검정, 분산 분석 수행")
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            analysis_parts.append("3. **그룹 분석**: 범주별 수치형 변수 비교 분석")
        analysis_parts.append("4. **이상값 탐지**: 통계적 이상값 식별 및 처리")
        if len(numeric_cols) > 3:
            analysis_parts.append("5. **차원 축소**: PCA, t-SNE를 통한 데이터 구조 탐색")
        
        return "\n".join(analysis_parts)
    
    def _generate_pattern_analysis(self, df, df_id: str, instruction: str) -> str:
        """범용적 패턴 분석 (바이너리 편향 제거)"""
        analysis_parts = []
        
        analysis_parts.append(f"## 📈 **데이터 패턴 분석**\n")
        
        # 수치형 변수 분포 패턴
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("### 📊 **수치형 변수 분포 패턴**")
            
            for i, col in enumerate(numeric_cols[:4], 1):
                var_name = self._get_generic_column_name(df, col, i)
                
                # 분포 특성 분석
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()
                
                # 이상값 분석
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
                
                analysis_parts.append(f"\n**{var_name}**:")
                
                # 분포 형태
                if abs(skewness) < 0.5:
                    dist_shape = "대칭적 분포"
                elif skewness > 0.5:
                    dist_shape = "우측 꼬리가 긴 분포"
                else:
                    dist_shape = "좌측 꼬리가 긴 분포"
                
                analysis_parts.append(f"- 분포 형태: {dist_shape}")
                analysis_parts.append(f"- 변동성: {'높음' if df[col].std() > df[col].mean() else '보통' if df[col].std() > df[col].mean()/2 else '낮음'}")
                
                if len(outliers) > 0:
                    analysis_parts.append(f"- 이상값: {len(outliers)}개 ({len(outliers)/len(df)*100:.1f}%)")
                else:
                    analysis_parts.append("- 이상값: 없음")
        
        # 범주형 변수 분포 패턴
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n### 📝 **범주형 변수 분포 패턴**")
            
            for i, col in enumerate(categorical_cols[:4], 1):
                var_name = self._get_generic_column_name(df, col, i)
                value_counts = df[col].value_counts()
                
                # 분포 균등성 분석
                max_freq = value_counts.max()
                min_freq = value_counts.min()
                balance_ratio = min_freq / max_freq
                
                analysis_parts.append(f"\n**{var_name}**:")
                analysis_parts.append(f"- 고유값 수: {len(value_counts)}개")
                
                if balance_ratio > 0.7:
                    balance_desc = "균등한 분포"
                elif balance_ratio > 0.3:
                    balance_desc = "약간 불균등한 분포"
                else:
                    balance_desc = "매우 불균등한 분포"
                
                analysis_parts.append(f"- 분포 균등성: {balance_desc}")
                
                # 상위 빈도 카테고리
                top_categories = value_counts.head(3)
                analysis_parts.append("- 상위 카테고리:")
                for cat, count in top_categories.items():
                    analysis_parts.append(f"  - {cat}: {count}개 ({count/len(df)*100:.1f}%)")
        
        # 전체 데이터 패턴 요약
        analysis_parts.append("\n### 🔍 **전체 데이터 패턴 요약**")
        
        # 다양성 지수
        total_unique_values = sum(df[col].nunique() for col in df.columns)
        diversity_index = total_unique_values / len(df)
        
        analysis_parts.append(f"- 데이터 다양성: {'높음' if diversity_index > 0.5 else '보통' if diversity_index > 0.2 else '낮음'}")
        
        # 완성도
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        analysis_parts.append(f"- 데이터 완성도: {completeness:.1f}%")
        
        # 구조 복잡성
        if len(numeric_cols) > 3 and len(categorical_cols) > 2:
            complexity = "복잡한 다차원 구조"
        elif len(numeric_cols) > 1 and len(categorical_cols) > 1:
            complexity = "중간 복잡도 구조"
        else:
            complexity = "단순한 구조"
        
        analysis_parts.append(f"- 구조 복잡성: {complexity}")
        
        return "\n".join(analysis_parts)
    
    def _prepare_data_context(self, df) -> str:
        """LLM이 데이터를 이해할 수 있도록 핵심 컨텍스트 정보 준비 (완전 범용화)"""
        context_parts = []
        
        # 기본 정보
        context_parts.append(f"데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 데이터 유형별 분류
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        context_parts.append(f"\n변수 구성:")
        context_parts.append(f"- 수치형 변수: {len(numeric_cols)}개")
        context_parts.append(f"- 범주형 변수: {len(categorical_cols)}개")
        context_parts.append(f"- 날짜형 변수: {len(datetime_cols)}개")
        
        # 수치형 변수 특성 (바이너리 구분 없이)
        if len(numeric_cols) > 0:
            context_parts.append("\n수치형 변수 요약:")
            desc = df[numeric_cols].describe()
            for i, col in enumerate(numeric_cols[:3], 1):  # 처음 3개만
                if col in desc.columns:
                    unique_count = df[col].nunique()
                    var_type = "이산형" if unique_count < 20 else "연속형"
                    context_parts.append(f"- 수치형{i} ({var_type}): 평균 {desc.loc['mean', col]:.2f}, 고유값 {unique_count}개")
        
        # 범주형 변수 정보
        if len(categorical_cols) > 0:
            context_parts.append("\n범주형 변수 정보:")
            for i, col in enumerate(categorical_cols[:3], 1):  # 처음 3개만
                unique_count = df[col].nunique()
                cardinality = "저" if unique_count < 10 else "중" if unique_count < 50 else "고"
                top_values = df[col].value_counts().head(2)
                context_parts.append(f"- 범주형{i} ({cardinality}카디널리티): {unique_count}개 고유값, 상위값 {dict(top_values)}")
        
        # 결측값 정보
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            context_parts.append("\n결측값:")
            missing_count = 0
            for col, count in missing_info.items():
                if count > 0:
                    missing_count += 1
                    col_type = self._get_generic_column_name(df, col, missing_count)
                    context_parts.append(f"- {col_type}: {count}개 ({count/len(df)*100:.1f}%)")
        
        # 데이터 특성 요약
        context_parts.append(f"\n데이터 특성:")
        context_parts.append(f"- 완성도: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
        context_parts.append(f"- 규모: {'대용량' if len(df) > 10000 else '중간' if len(df) > 1000 else '소규모'} 데이터셋")
        
        return "\n".join(context_parts)
    
    def _get_generic_column_name(self, df, col, index=None):
        """컬럼을 범용적 이름으로 변환"""
        if df[col].dtype in ['int64', 'float64']:
            unique_count = df[col].nunique()
            if unique_count == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                return f"이진변수{index or ''}"
            elif unique_count < 20:
                return f"이산형{index or ''}"
            else:
                return f"연속형{index or ''}"
        elif df[col].dtype in ['object', 'category']:
            unique_count = df[col].nunique()
            if unique_count < 10:
                return f"범주형{index or ''}"
            else:
                return f"고카디널리티{index or ''}"
        elif 'datetime' in str(df[col].dtype):
            return f"날짜형{index or ''}"
        else:
            return f"기타형{index or ''}"

    def _generate_data_overview(self, df, df_id: str, instruction: str) -> str:
        """데이터 구조 및 개요 분석 (완전 범용화)"""
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
        
        # 컬럼별 데이터 타입 (범용화)
        analysis_parts.append("\n## 🔍 **변수별 상세 정보**")
        
        # 변수 유형별 카운터
        numeric_count = 0
        categorical_count = 0
        binary_count = 0
        
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            
            # 변수 유형 결정
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                    binary_count += 1
                    var_name = f"바이너리{binary_count}"
                else:
                    numeric_count += 1
                    var_name = f"수치형{numeric_count}"
            else:
                categorical_count += 1
                var_name = f"범주형{categorical_count}"
            
            analysis_parts.append(f"{i}. **{var_name}** ({dtype})")
            analysis_parts.append(f"   - 유효값: {non_null_count:,}개 ({non_null_count/len(df)*100:.1f}%)")
            if null_count > 0:
                analysis_parts.append(f"   - 결측값: {null_count:,}개 ({null_count/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_descriptive_stats(self, df, df_id: str, instruction: str) -> str:
        """기술통계 및 분포 분석 (완전 범용화)"""
        analysis_parts = []
        
        analysis_parts.append(f"# 📈 **기술통계 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 수치형 변수 통계 (범용화)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("## 🔢 **수치형 변수 기술통계**")
            desc = df[numeric_cols].describe()
            
            numeric_count = 0
            binary_count = 0
            
            for col in numeric_cols:
                if col in desc.columns:
                    # 바이너리 변수와 일반 수치형 변수 구분
                    if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                        binary_count += 1
                        var_name = f"바이너리{binary_count}"
                    else:
                        numeric_count += 1
                        var_name = f"수치형{numeric_count}"
                    
                    analysis_parts.append(f"\n**{var_name}**:")
                    analysis_parts.append(f"- 평균: {desc.loc['mean', col]:.2f}")
                    analysis_parts.append(f"- 중앙값: {desc.loc['50%', col]:.2f}")
                    analysis_parts.append(f"- 표준편차: {desc.loc['std', col]:.2f}")
                    analysis_parts.append(f"- 최솟값: {desc.loc['min', col]:.2f}")
                    analysis_parts.append(f"- 최댓값: {desc.loc['max', col]:.2f}")
        
        # 범주형 변수 통계 (범용화)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n## 📝 **범주형 변수 빈도 분석**")
            for i, col in enumerate(categorical_cols[:3], 1):  # 상위 3개만
                value_counts = df[col].value_counts().head(5)
                analysis_parts.append(f"\n**범주형{i} (상위 5개 값):**")
                for value, count in value_counts.items():
                    analysis_parts.append(f"- {value}: {count:,}개 ({count/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_correlation_analysis(self, df, df_id: str, instruction: str) -> str:
        """상관관계 분석 (완전 범용화)"""
        analysis_parts = []
        
        analysis_parts.append(f"# 🔗 **상관관계 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 수치형 변수들 간의 상관관계 (범용화)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # 컬럼명을 범용적 이름으로 매핑
            col_name_mapping = {}
            numeric_count = 0
            binary_count = 0
            
            for col in numeric_cols:
                if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                    binary_count += 1
                    col_name_mapping[col] = f"바이너리{binary_count}"
                else:
                    numeric_count += 1
                    col_name_mapping[col] = f"수치형{numeric_count}"
            
            analysis_parts.append("## 📊 **수치형 변수 상관관계**")
            
            # 강한 상관관계 찾기 (|r| > 0.5)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        var1_name = col_name_mapping.get(col1, col1)
                        var2_name = col_name_mapping.get(col2, col2)
                        strong_correlations.append((var1_name, var2_name, corr_val))
            
            if strong_correlations:
                analysis_parts.append("\n**강한 상관관계 (|r| > 0.5):**")
                for var1, var2, corr_val in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                    analysis_parts.append(f"- **{var1}** ↔ **{var2}**: {corr_val:.3f}")
            else:
                analysis_parts.append("\n강한 상관관계(|r| > 0.5)를 보이는 변수 쌍이 없습니다.")
            
            # 상관관계 매트릭스 요약 (범용화)
            analysis_parts.append("\n**전체 상관관계 매트릭스:**")
            for col in numeric_cols[:4]:  # 상위 4개 변수만
                var_name = col_name_mapping.get(col, col)
                analysis_parts.append(f"\n**{var_name}과의 상관관계:**")
                correlations = corr_matrix[col].drop(col).sort_values(key=abs, ascending=False)
                for other_col, corr_val in correlations.head(3).items():
                    other_var_name = col_name_mapping.get(other_col, other_col)
                    analysis_parts.append(f"- {other_var_name}: {corr_val:.3f}")
        else:
            analysis_parts.append("## ⚠️ **상관관계 분석 불가**")
            analysis_parts.append("수치형 변수가 2개 미만이어서 상관관계 분석을 수행할 수 없습니다.")
        
        return "\n".join(analysis_parts)
    
    def _generate_trend_analysis(self, df, df_id: str, instruction: str) -> str:
        """트렌드 및 패턴 분석 (완전 범용화)"""
        analysis_parts = []
        
        analysis_parts.append(f"# 📈 **트렌드 및 패턴 분석 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 컬럼명 매핑 생성
        col_name_mapping = {}
        numeric_count = 0
        categorical_count = 0
        binary_count = 0
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                    binary_count += 1
                    col_name_mapping[col] = f"바이너리{binary_count}"
                else:
                    numeric_count += 1
                    col_name_mapping[col] = f"수치형{numeric_count}"
            else:
                categorical_count += 1
                col_name_mapping[col] = f"범주형{categorical_count}"
        
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
                target_name = col_name_mapping.get(target_col, target_col)
                positive_rate = df[target_col].mean() * 100
                analysis_parts.append(f"\n**{target_name} 분포:**")
                analysis_parts.append(f"- 양성(1): {df[target_col].sum():,}개 ({positive_rate:.1f}%)")
                analysis_parts.append(f"- 음성(0): {(df[target_col] == 0).sum():,}개 ({100-positive_rate:.1f}%)")
                
                # 범주형 변수와의 관계 분석
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                for cat_col in categorical_cols[:2]:  # 상위 2개 범주형 변수
                    cat_name = col_name_mapping.get(cat_col, cat_col)
                    analysis_parts.append(f"\n**{cat_name}별 {target_name} 패턴:**")
                    group_stats = df.groupby(cat_col)[target_col].agg(['count', 'sum', 'mean'])
                    for category in group_stats.index[:4]:  # 상위 4개 카테고리
                        total = group_stats.loc[category, 'count']
                        positive = group_stats.loc[category, 'sum']
                        rate = group_stats.loc[category, 'mean'] * 100
                        analysis_parts.append(f"- **{category}**: {positive}/{total}개 ({rate:.1f}%)")
        
        # 2. 범주형 변수 분포 패턴 (범용화)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis_parts.append("\n## 📊 **범주형 변수 분포 패턴**")
            for col in categorical_cols[:3]:  # 상위 3개만
                col_name = col_name_mapping.get(col, col)
                value_counts = df[col].value_counts()
                total_unique = df[col].nunique()
                analysis_parts.append(f"\n**{col_name} ({total_unique}개 고유값):**")
                for i, (value, count) in enumerate(value_counts.head(4).items()):
                    analysis_parts.append(f"{i+1}. {value}: {count:,}개 ({count/len(df)*100:.1f}%)")
        
        # 3. 수치형 변수 분포 패턴 (범용화)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append("\n## 📈 **수치형 변수 분포 특성**")
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:3]:  # 상위 3개만
                if col in desc.columns:
                    col_name = col_name_mapping.get(col, col)
                    skewness = df[col].skew()
                    outlier_threshold = desc.loc['75%', col] + 1.5 * (desc.loc['75%', col] - desc.loc['25%', col])
                    outliers = (df[col] > outlier_threshold).sum()
                    
                    analysis_parts.append(f"\n**{col_name}:**")
                    analysis_parts.append(f"- 범위: {desc.loc['min', col]:.2f} ~ {desc.loc['max', col]:.2f}")
                    analysis_parts.append(f"- 분포: {'왼쪽 치우침' if skewness > 1 else '오른쪽 치우침' if skewness < -1 else '정규분포에 가까움'}")
                    if outliers > 0:
                        analysis_parts.append(f"- 이상값: {outliers}개 ({outliers/len(df)*100:.1f}%)")
        
        return "\n".join(analysis_parts)
    
    def _generate_insights_summary(self, df, df_id: str, instruction: str) -> str:
        """핵심 인사이트 및 요약 (완전 범용화)"""
        analysis_parts = []
        
        analysis_parts.append(f"# 💡 **핵심 인사이트 요약 보고서**\n")
        analysis_parts.append(f"**요청**: {instruction}")
        analysis_parts.append(f"**데이터셋**: {df_id}")
        analysis_parts.append(f"**분석 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 컬럼명 매핑 생성
        col_name_mapping = {}
        numeric_count = 0
        categorical_count = 0
        binary_count = 0
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                    binary_count += 1
                    col_name_mapping[col] = f"바이너리{binary_count}"
                else:
                    numeric_count += 1
                    col_name_mapping[col] = f"수치형{numeric_count}"
            else:
                categorical_count += 1
                col_name_mapping[col] = f"범주형{categorical_count}"
        
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
        
        # 바이너리 타겟 변수 인사이트 (범용화)
        binary_targets = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if set(unique_vals) == {0, 1}:
                    positive_rate = df[col].mean() * 100
                    col_name = col_name_mapping.get(col, col)
                    binary_targets.append((col_name, positive_rate))
        
        if binary_targets:
            for target_name, rate in binary_targets:
                balance_status = "균형잡힌" if 40 <= rate <= 60 else "불균형한"
                analysis_parts.append(f"- {target_name}: {rate:.1f}% 양성률로 {balance_status} 분포")
        
        # 결측값 패턴 인사이트 (범용화)
        missing_rates = df.isnull().mean() * 100
        high_missing = missing_rates[missing_rates > 20]
        if len(high_missing) > 0:
            missing_var_names = [col_name_mapping.get(col, col) for col in high_missing.index]
            analysis_parts.append(f"- 결측값 주의: {missing_var_names} 변수의 결측률이 20% 이상")
        
        # 범주형 변수 다양성 인사이트 (범용화)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            high_cardinality = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.1]
            if high_cardinality:
                high_card_names = [col_name_mapping.get(col, col) for col in high_cardinality]
                analysis_parts.append(f"- 고유값 과다: {high_card_names} 변수는 범주 수가 매우 높음")
        
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
        """A2A SDK 표준 실행 - 스트리밍 지원 (공식 Hello World Agent 패턴)"""
        logger.info("🎯 PandasAgentExecutor.execute() 호출됨")
        
        try:
            # 사용자 입력 추출 (공식 패턴)
            user_message = context.get_user_input()
            logger.info(f"📝 사용자 입력: {user_message}")
            
            # 스트리밍 요청 확인 (키워드 기반)
            streaming_keywords = ["eda", "분석", "실시간", "스트리밍", "progress", "종합", "상세"]
            should_stream = any(keyword in user_message.lower() for keyword in streaming_keywords)
            
            # 에이전트 실행 (스트리밍 지원)
            if should_stream:
                logger.info("🔄 스트리밍 모드로 분석 수행")
                result = await self.agent.invoke(user_message, stream=True)
            else:
                logger.info("📊 일반 모드로 분석 수행")
                result = await self.agent.invoke(user_message, stream=False)
            
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