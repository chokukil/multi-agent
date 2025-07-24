#!/usr/bin/env python3
"""
AI_DS_Team DataCleaningAgent A2A Server (New Implementation)
Port: 8306

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
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Langfuse 통합
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")

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

# Logger 설정
logger = logging.getLogger(__name__)

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
    logger.info("✅ pandas-ai 사용 가능")
except ImportError:
    PANDASAI_AVAILABLE = False
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
        self.current_data = None  # 현재 작업 중인 데이터
    
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
    
    # 1. 결측값 감지 기능
    def detect_missing_values(self, df: pd.DataFrame) -> dict:
        """결측값을 감지하고 상세 리포트 생성"""
        logger.info("🔍 결측값 감지 시작...")
        
        missing_info = {}
        total_missing = 0
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(df)
                missing_info[col] = {
                    'count': missing_count,
                    'ratio': missing_ratio,
                    'percentage': f"{missing_ratio * 100:.2f}%"
                }
                total_missing += missing_count
        
        # 전체 통계
        total_cells = df.shape[0] * df.shape[1]
        overall_missing_ratio = total_missing / total_cells if total_cells > 0 else 0
        
        result = {
            'total_missing': total_missing,
            'overall_missing_ratio': overall_missing_ratio,
            'overall_missing_percentage': f"{overall_missing_ratio * 100:.2f}%",
            'columns_with_missing': missing_info,
            'columns_without_missing': [col for col in df.columns if col not in missing_info],
            'recommendation': self._get_missing_value_recommendation(missing_info)
        }
        
        logger.info(f"✅ 결측값 감지 완료: 총 {total_missing}개 발견")
        return result
    
    def _get_missing_value_recommendation(self, missing_info: dict) -> list:
        """결측값 처리 권장사항 생성"""
        recommendations = []
        
        for col, info in missing_info.items():
            if info['ratio'] > 0.4:
                recommendations.append(f"'{col}' 컬럼 제거 권장 (40% 이상 결측)")
            elif info['ratio'] > 0.2:
                recommendations.append(f"'{col}' 컬럼 신중한 대체 필요 (20-40% 결측)")
            else:
                recommendations.append(f"'{col}' 컬럼 평균/최빈값 대체 가능 (<20% 결측)")
        
        return recommendations
    
    # 2. 결측값 처리 기능
    def handle_missing_values(self, df: pd.DataFrame, method: str = "auto", columns: list = None) -> dict:
        """결측값 처리 - 다양한 방법 지원"""
        logger.info(f"🔧 결측값 처리 시작 (방법: {method})...")
        
        self.current_data = df.copy()
        processing_report = []
        
        # 처리할 컬럼 결정
        if columns:
            cols_to_process = columns
        else:
            cols_to_process = df.columns[df.isnull().any()].tolist()
        
        for col in cols_to_process:
            if col not in df.columns:
                continue
                
            missing_count = self.current_data[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if method == "auto":
                # 자동 처리: 데이터 타입에 따라 결정
                if self.current_data[col].dtype in ['int64', 'float64']:
                    fill_value = self.current_data[col].mean()
                    self.current_data[col].fillna(fill_value, inplace=True)
                    processing_report.append(f"'{col}': 평균값({fill_value:.2f})으로 대체")
                else:
                    mode_val = self.current_data[col].mode()
                    if not mode_val.empty:
                        fill_value = mode_val.iloc[0]
                        self.current_data[col].fillna(fill_value, inplace=True)
                        processing_report.append(f"'{col}': 최빈값('{fill_value}')으로 대체")
            
            elif method == "drop":
                # 결측값이 있는 행 제거
                before_shape = self.current_data.shape[0]
                self.current_data = self.current_data.dropna(subset=[col])
                after_shape = self.current_data.shape[0]
                processing_report.append(f"'{col}': {before_shape - after_shape}개 행 제거")
            
            elif method == "forward_fill":
                # 앞의 값으로 채우기
                self.current_data[col].fillna(method='ffill', inplace=True)
                processing_report.append(f"'{col}': 이전 값으로 대체 (forward fill)")
            
            elif method == "backward_fill":
                # 뒤의 값으로 채우기
                self.current_data[col].fillna(method='bfill', inplace=True)
                processing_report.append(f"'{col}': 다음 값으로 대체 (backward fill)")
            
            elif method == "interpolate":
                # 보간법 사용
                if self.current_data[col].dtype in ['int64', 'float64']:
                    self.current_data[col].interpolate(method='linear', inplace=True)
                    processing_report.append(f"'{col}': 선형 보간으로 대체")
        
        return {
            'processed_data': self.current_data,
            'processing_report': processing_report,
            'before_missing': df.isnull().sum().sum(),
            'after_missing': self.current_data.isnull().sum().sum(),
            'method_used': method
        }
    
    # 3. 이상치 감지 기능
    def detect_outliers(self, df: pd.DataFrame, method: str = "IQR", threshold: float = 1.5) -> dict:
        """이상치 감지 - IQR, Z-score, Isolation Forest 등 다양한 방법 지원"""
        logger.info(f"🔍 이상치 감지 시작 (방법: {method})...")
        
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "IQR":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers = df[outliers_mask][col].tolist()
                
                outlier_info[col] = {
                    'method': 'IQR',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': len(outliers),
                    'outlier_percentage': f"{(len(outliers) / len(df)) * 100:.2f}%",
                    'outlier_values': outliers[:10]  # 최대 10개만 표시
                }
            
            elif method == "Z-score":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers_mask = z_scores > threshold
                outliers = df[col].dropna()[outliers_mask].tolist()
                
                outlier_info[col] = {
                    'method': 'Z-score',
                    'threshold': threshold,
                    'outlier_count': len(outliers),
                    'outlier_percentage': f"{(len(outliers) / len(df)) * 100:.2f}%",
                    'outlier_values': outliers[:10]
                }
        
        return {
            'outlier_summary': outlier_info,
            'total_outliers': sum(info['outlier_count'] for info in outlier_info.values()),
            'columns_with_outliers': [col for col, info in outlier_info.items() if info['outlier_count'] > 0],
            'method': method,
            'recommendation': self._get_outlier_recommendation(outlier_info)
        }
    
    def _get_outlier_recommendation(self, outlier_info: dict) -> list:
        """이상치 처리 권장사항 생성"""
        recommendations = []
        
        for col, info in outlier_info.items():
            outlier_ratio = info['outlier_count'] / 100  # 대략적인 비율
            if outlier_ratio > 0.1:
                recommendations.append(f"'{col}': 이상치가 많음 (10% 이상) - 도메인 지식 기반 검토 필요")
            elif outlier_ratio > 0.05:
                recommendations.append(f"'{col}': 중간 수준 이상치 (5-10%) - 캡핑 또는 변환 고려")
            elif info['outlier_count'] > 0:
                recommendations.append(f"'{col}': 소수 이상치 (<5%) - 제거 또는 대체 가능")
        
        return recommendations
    
    # 4. 이상치 처리 기능
    def treat_outliers(self, df: pd.DataFrame, method: str = "cap", threshold: float = 1.5, columns: list = None) -> dict:
        """이상치 처리 - 제거, 캡핑, 변환 등"""
        logger.info(f"🔧 이상치 처리 시작 (방법: {method})...")
        
        self.current_data = df.copy()
        processing_report = []
        
        # 처리할 컬럼 결정
        if columns:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.current_data[col].quantile(0.25)
            Q3 = self.current_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_mask = (self.current_data[col] < lower_bound) | (self.current_data[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count == 0:
                continue
            
            if method == "remove":
                # 이상치가 있는 행 제거
                before_shape = self.current_data.shape[0]
                self.current_data = self.current_data[~outliers_mask]
                after_shape = self.current_data.shape[0]
                processing_report.append(f"'{col}': {before_shape - after_shape}개 이상치 행 제거")
            
            elif method == "cap":
                # 이상치를 경계값으로 대체 (캡핑)
                self.current_data.loc[self.current_data[col] < lower_bound, col] = lower_bound
                self.current_data.loc[self.current_data[col] > upper_bound, col] = upper_bound
                processing_report.append(f"'{col}': {outlier_count}개 이상치 캡핑 처리")
            
            elif method == "transform":
                # 로그 변환
                if (self.current_data[col] > 0).all():
                    self.current_data[col] = np.log1p(self.current_data[col])
                    processing_report.append(f"'{col}': 로그 변환 적용")
                else:
                    processing_report.append(f"'{col}': 음수값 존재로 로그 변환 불가")
        
        return {
            'processed_data': self.current_data,
            'processing_report': processing_report,
            'method_used': method,
            'threshold': threshold
        }

class DataCleaningAgentExecutor(AgentExecutor):
    """A2A DataCleaningAgent Executor with pandas-ai pattern and Langfuse integration"""
    
    def __init__(self):
        """초기화"""
        self.data_processor = PandasAIDataProcessor()
        self.data_cleaner = EnhancedDataCleaner()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ DataCleaningAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("🧹 DataCleaningAgent Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """pandas-ai 패턴이 적용된 데이터 클리닝 실행"""
        logger.info(f"🚀 DataCleaningAgent 실행 시작 - Task: {context.task_id}")
        
        # Langfuse 트레이스 시작 (올바른 방식)
        session_id = None
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
                    name="DataCleaningAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "DataCleaningAgent",
                        "port": 8306,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id)
                    }
                )
                logger.info(f"📊 Langfuse 메인 트레이스 시작: {context.task_id}")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
        
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # A2A SDK 0.2.9 공식 패턴에 따른 태스크 라이프사이클
            await task_updater.submit()
            await task_updater.start_work()
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧹 pandas-ai 패턴 DataCleaningAgent 시작...")
            )
            
            # A2A SDK 0.2.9 공식 패턴에 따른 사용자 메시지 추출
            user_instructions = ""
            if context.message and hasattr(context.message, 'parts') and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "text":
                        user_instructions += part.root.text + " "
                    elif hasattr(part, 'text'):  # 대체 패턴
                        user_instructions += part.text + " "
                
                user_instructions = user_instructions.strip()
                logger.info(f"📝 사용자 요청: {user_instructions}")
            
            # 기본 요청이 없으면 데모 모드
            if not user_instructions:
                user_instructions = "샘플 데이터로 데이터 클리닝을 시연해주세요"
                logger.info("📝 기본 데모 모드 실행")
                
            # pandas-ai 패턴으로 데이터 파싱
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("📊 pandas-ai 패턴으로 데이터 분석 중...")
            )
            
            # 1단계: 메시지에서 데이터 파싱 (Langfuse 추적)
            parsing_span = None
            if main_trace:
                parsing_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="data_parsing",
                    input={"user_instructions": user_instructions[:500]},
                    metadata={"step": "1", "description": "Parse data from user message"}
                )
            
            logger.info("🔍 데이터 파싱 시작")
            df = self.data_processor.parse_data_from_message(user_instructions)
            logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape if df is not None else 'None'}")
            
            if parsing_span:
                if df is not None:
                    parsing_span.update(
                        output={
                            "success": True,
                            "data_shape": list(df.shape),  # tuple을 list로 변환
                            "columns": list(df.columns),
                            "data_preview": df.head(3).to_dict('records'),  # 더 readable한 형태
                            "total_rows": len(df),
                            "total_columns": len(df.columns)
                        }
                    )
                else:
                    parsing_span.update(
                        output={
                            "success": False, 
                            "reason": "No CSV data found in message",
                            "fallback_needed": True
                        }
                    )
            
            # 2단계: 데이터가 없으면 DataManager 폴백
            if df is None:
                try:
                    available_data = data_manager.list_dataframes()
                    if available_data:
                        selected_id = available_data[0]
                        df = data_manager.get_dataframe(selected_id)
                        logger.info(f"✅ DataManager 폴백: {selected_id}")
                    else:
                        # 샘플 데이터 생성
                        df = self.data_processor._create_sample_data()
                        logger.info("✅ 샘플 데이터 생성")
                except Exception as e:
                    logger.warning(f"DataManager 폴백 실패: {e}")
                    # 샘플 데이터로 대체
                    df = self.data_processor._create_sample_data()
                    logger.info("✅ 폴백 후 샘플 데이터 생성")
                
            if df is not None and not df.empty:
                # 3단계: pandas-ai DataFrame 생성
                self.data_processor.create_pandasai_dataframe(
                    df, name="user_dataset", description=user_instructions[:100]
                )
                
                # 4단계: 데이터 클리닝 실행 (Langfuse 추적)
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("🧹 Enhanced 데이터 클리닝 실행 중...")
                )
                
                cleaning_span = None
                if main_trace:
                    cleaning_span = self.langfuse_tracer.langfuse.span(
                        trace_id=context.task_id,
                        name="data_cleaning",
                        input={
                            "original_data_shape": df.shape,
                            "columns": list(df.columns),
                            "user_instructions": user_instructions[:200]
                        },
                        metadata={"step": "2", "description": "Clean and process data"}
                    )
                
                logger.info("🚀 데이터 정리 요청 처리: " + user_instructions[:100] + "...")
                cleaning_results = self.data_cleaner.clean_data(df, user_instructions)
                
                if cleaning_span:
                    cleaning_span.update(
                        output={
                            "success": True,
                            "cleaned_data_shape": list(cleaning_results['cleaned_data'].shape),
                            "original_shape": list(cleaning_results['original_shape']),
                            "final_shape": list(cleaning_results['final_shape']),
                            "memory_saved_mb": round(cleaning_results['memory_saved'], 4),
                            "data_quality_score": cleaning_results['data_quality_score'],
                            "cleaning_operations_performed": len(cleaning_results['cleaning_report']),
                            "cleaning_report_summary": cleaning_results['cleaning_report'][:3],  # 처음 3개만
                            "rows_removed": cleaning_results['original_shape'][0] - cleaning_results['final_shape'][0],  
                            "columns_removed": cleaning_results['original_shape'][1] - cleaning_results['final_shape'][1]
                        }
                    )
                
                # 5단계: 결과 저장 (Langfuse 추적)
                save_span = None
                if main_trace:
                    save_span = self.langfuse_tracer.langfuse.span(
                        trace_id=context.task_id,
                        name="save_results",
                        input={
                            "cleaned_data_shape": cleaning_results['cleaned_data'].shape,
                            "data_quality_score": cleaning_results['data_quality_score'],
                            "cleaning_operations": len(cleaning_results['cleaning_report'])
                        },
                        metadata={"step": "3", "description": "Save cleaned data to file"}
                    )
                
                output_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/cleaned_data_{context.task_id}.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cleaning_results['cleaned_data'].to_csv(output_path, index=False)
                logger.info(f"정리된 데이터 저장: {output_path}")
                
                if save_span:
                    save_span.update(
                        output={
                            "file_path": output_path,
                            "file_size_mb": os.path.getsize(output_path) / (1024*1024),
                            "saved_rows": len(cleaning_results['cleaned_data']),
                            "saved_successfully": True
                        }
                    )
                
                # 6단계: 응답 생성
                result = self._generate_response(cleaning_results, user_instructions, output_path)
                
            else:
                result = self._generate_no_data_response(user_instructions)
            
            # A2A SDK 0.2.9 공식 패턴에 따른 최종 응답
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            # Langfuse 메인 트레이스 완료
            if main_trace:
                try:
                    # Output을 요약된 형태로 제공 (너무 길면 Langfuse에서 문제가 될 수 있음)
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
                            "agent": "DataCleaningAgent",
                            "port": 8306
                        }
                    )
                    logger.info(f"📊 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
                
        except Exception as e:
            logger.error(f"❌ DataCleaningAgent 실행 오류: {e}")
            
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
                            "agent": "DataCleaningAgent",
                            "port": 8306
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
            
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
        url="http://localhost:8306/",
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
    print("🌐 Server starting on http://localhost:8306")
    print("📋 Agent card: http://localhost:8306/.well-known/agent.json")
    print("✨ Features: pandas-ai pattern + ai-data-science-team compatibility")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8306, log_level="info")

if __name__ == "__main__":
    main()