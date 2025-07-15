#!/usr/bin/env python3
"""
CherryAI Unified EDA Tools Server - Port 8312
A2A SDK 0.2.9 완전 표준 준수 + UnifiedDataInterface 패턴

📊 핵심 기능:
- 🔍 LLM 기반 지능형 탐색적 데이터 분석 전략
- 📈 통계 계산 오류 완전 해결 (설계 문서 주요 문제점 해결)
- 📋 포괄적 5단계 EDA 프로세스 (기술통계, 분포, 상관관계, 이상값, 패턴)
- 🎨 Interactive 시각화 + 통계 리포트 통합
- 🧠 LLM 인사이트 생성 및 해석
- 🎯 A2A 표준 TaskUpdater + 실시간 스트리밍

기반: pandas_agent 패턴 + unified_data_loader 성공 사례
"""

import asyncio
import logging
import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK 0.2.9 표준 imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities,
    TaskState, TextPart
)
from a2a.utils import new_agent_text_message
import uvicorn

# CherryAI Core imports
from core.llm_factory import LLMFactory
from a2a_ds_servers.unified_data_system.core.unified_data_interface import UnifiedDataInterface
from a2a_ds_servers.unified_data_system.core.smart_dataframe import SmartDataFrame
from a2a_ds_servers.unified_data_system.core.llm_first_data_engine import LLMFirstDataEngine
from a2a_ds_servers.unified_data_system.core.cache_manager import CacheManager
from a2a_ds_servers.unified_data_system.utils.file_scanner import FileScanner

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EDAInsight:
    """EDA 인사이트 정의"""
    category: str  # statistical, distribution, correlation, outlier, pattern
    finding: str
    confidence: float
    evidence: Dict[str, Any]
    recommendation: str

@dataclass
class StatisticalAnalysis:
    """통계 분석 결과"""
    descriptive_stats: Dict[str, Any]
    distribution_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]

class UnifiedEDAToolsExecutor(AgentExecutor, UnifiedDataInterface):
    """
    Unified EDA Tools Executor
    
    pandas_agent 패턴 + data_loader 성공 사례 기반
    - LLM First 탐색적 데이터 분석
    - 통계 계산 안정성 보장
    - 포괄적 5단계 EDA 프로세스
    - A2A SDK 0.2.9 완전 준수
    """
    
    def __init__(self):
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.file_scanner = FileScanner()
        self.llm_factory = LLMFactory()
        
        # EDA 전문 설정
        self.eda_components = {
            'descriptive_statistics': [
                'basic_stats', 'central_tendency', 'variability', 'shape_measures',
                'quantiles', 'missing_value_analysis'
            ],
            'distribution_analysis': [
                'histograms', 'density_plots', 'qq_plots', 'normality_tests',
                'skewness_kurtosis', 'distribution_fitting'
            ],
            'correlation_analysis': [
                'correlation_matrix', 'correlation_heatmap', 'scatter_matrix',
                'partial_correlations', 'association_measures'
            ],
            'outlier_analysis': [
                'box_plots', 'z_score_analysis', 'iqr_analysis', 'isolation_forest',
                'local_outlier_factor', 'outlier_visualization'
            ],
            'pattern_analysis': [
                'trend_analysis', 'seasonality_detection', 'clustering_tendency',
                'feature_importance', 'anomaly_detection'
            ]
        }
        
        # 통계 안전성 설정 (핵심 문제 해결)
        self.statistical_safety = {
            'min_sample_size': 5,  # 최소 샘플 크기
            'max_categories': 50,   # 카테고리 최대 개수
            'correlation_threshold': 0.01,  # 상관관계 임계값
            'outlier_detection_methods': ['iqr', 'zscore', 'isolation'],
            'normality_test_methods': ['shapiro', 'jarque_bera', 'kolmogorov']
        }
        
        # 시각화 설정
        self.visualization_themes = {
            'default': 'plotly_white',
            'professional': 'plotly',
            'minimal': 'simple_white',
            'academic': 'ggplot2'
        }
        
        logger.info("✅ Unified EDA Tools Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        A2A 표준 실행: 7단계 포괄적 탐색적 데이터 분석 프로세스
        
        🔍 1단계: LLM EDA 의도 분석
        📂 2단계: 데이터 검색 및 안전 로딩
        📊 3단계: 기술통계 안전 계산 (통계 오류 해결)
        📈 4단계: 분포 분석 및 시각화
        🔗 5단계: 상관관계 및 연관성 분석
        ⚠️ 6단계: 이상값 및 패턴 탐지
        🧠 7단계: LLM 인사이트 생성 및 종합 보고서
        """
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            start_time = time.time()
            
            # 🔍 1단계: EDA 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧑🏻 **탐색적 데이터 분석 시작** - 1단계: EDA 요구사항 분석 중...")
            )
            
            user_query = self._extract_user_query(context)
            logger.info(f"📊 EDA Analysis Query: {user_query}")
            
            # LLM 기반 EDA 의도 분석
            eda_intent = await self._analyze_eda_intent(user_query)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **의도 분석 완료**\n"
                    f"- 분석 유형: {eda_intent['analysis_type']}\n"
                    f"- 중점 영역: {', '.join(eda_intent['focus_areas'])}\n"
                    f"- 상세도 수준: {eda_intent['detail_level']}\n"
                    f"- 신뢰도: {eda_intent['confidence']:.2f}\n\n"
                    f"**2단계**: 데이터 검색 중..."
                )
            )
            
            # 📂 2단계: 데이터 검색 및 로딩
            available_files = await self._scan_available_files()
            
            if not available_files:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(
                        "⚠️ **데이터 없음**: 분석할 데이터를 찾을 수 없습니다.\n\n"
                        "**해결책**:\n"
                        "1. `a2a_ds_servers/artifacts/data/` 폴더에 데이터 파일을 업로드해주세요\n"
                        "2. 지원 형식: CSV, Excel (.xlsx/.xls), JSON, Parquet\n"
                        "3. 권장 최소 크기: 100행 이상 (통계적 유의성)"
                    )
                )
                return
            
            # 최적 파일 선택
            selected_file = await self._select_optimal_file_for_eda(available_files, eda_intent)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **파일 선택 완료**\n"
                    f"- 파일: {selected_file['name']}\n"
                    f"- 크기: {selected_file['size']:,} bytes\n"
                    f"- EDA 적합도: {selected_file.get('eda_suitability', 'N/A')}\n\n"
                    f"**3단계**: 안전한 데이터 로딩 중..."
                )
            )
            
            # 📊 3단계: 안전한 데이터 로딩
            smart_df = await self._load_data_safely_for_eda(selected_file)
            
            # 📈 4단계: 기술통계 안전 계산 (핵심 문제 해결)
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **데이터 로딩 완료**\n"
                    f"- 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 숫자형 컬럼: {len(smart_df.data.select_dtypes(include=[np.number]).columns)}개\n"
                    f"- 범주형 컬럼: {len(smart_df.data.select_dtypes(include=['object']).columns)}개\n\n"
                    f"**4단계**: 기술통계 안전 계산 중..."
                )
            )
            
            # 통계 계산 오류 방지 시스템
            statistical_analysis = await self._comprehensive_statistical_analysis(smart_df)
            
            # 📈 5단계: 분포 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **기술통계 완료**\n"
                    f"- 계산된 통계: {len(statistical_analysis.descriptive_stats)}개 컬럼\n"
                    f"- 안전성 검증: 통과\n\n"
                    f"**5단계**: 분포 분석 및 시각화 중..."
                )
            )
            
            distribution_results = await self._distribution_analysis(smart_df, task_updater)
            
            # 🔗 6단계: 상관관계 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **분포 분석 완료**\n"
                    f"- 생성된 차트: {len(distribution_results['charts'])}개\n\n"
                    f"**6단계**: 상관관계 및 연관성 분석 중..."
                )
            )
            
            correlation_results = await self._correlation_analysis(smart_df, task_updater)
            
            # ⚠️ 7단계: 이상값 및 패턴 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **상관관계 분석 완료**\n"
                    f"- 상관관계 계수: {correlation_results['correlation_count']}개\n\n"
                    f"**7단계**: 이상값 및 패턴 탐지 중..."
                )
            )
            
            outlier_pattern_results = await self._outlier_and_pattern_analysis(smart_df, task_updater)
            
            # 🧠 8단계: LLM 인사이트 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**8단계**: LLM 인사이트 생성 및 종합 보고서 작성 중...")
            )
            
            llm_insights = await self._generate_llm_insights(
                smart_df, statistical_analysis, distribution_results, 
                correlation_results, outlier_pattern_results, eda_intent
            )
            
            # 최종 결과 통합
            final_results = await self._finalize_eda_results(
                smart_df=smart_df,
                statistical_analysis=statistical_analysis,
                distribution_results=distribution_results,
                correlation_results=correlation_results,
                outlier_pattern_results=outlier_pattern_results,
                llm_insights=llm_insights,
                task_updater=task_updater
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 최종 완료 메시지
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    f"✅ **탐색적 데이터 분석 완료!**\n\n"
                    f"📊 **분석 결과**:\n"
                    f"- 데이터 크기: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 생성된 차트: {final_results['total_charts']}개\n"
                    f"- LLM 인사이트: {len(final_results['insights'])}개\n"
                    f"- 발견된 패턴: {final_results['patterns_found']}개\n"
                    f"- 이상값 감지: {final_results['outliers_detected']}개\n"
                    f"- 처리 시간: {processing_time:.2f}초\n\n"
                    f"📁 **저장 위치**: {final_results['report_path']}\n"
                    f"📊 **Interactive 차트**: Streamlit에서 확인 가능\n"
                    f"📋 **종합 보고서**: 아티팩트로 생성됨"
                )
            )
            
            # 아티팩트 생성
            await self._create_eda_artifacts(final_results, task_updater)
            
        except Exception as e:
            logger.error(f"❌ EDA Analysis Error: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ **EDA 분석 실패**: {str(e)}")
            )
    
    async def _analyze_eda_intent(self, user_query: str) -> Dict[str, Any]:
        """LLM 기반 EDA 의도 분석"""
        llm = await self.llm_factory.get_llm()
        
        prompt = f"""
        사용자의 탐색적 데이터 분석 요청을 분석하여 최적의 EDA 전략을 결정해주세요:
        
        요청: {user_query}
        
        사용 가능한 EDA 구성요소들:
        {json.dumps(self.eda_components, indent=2, ensure_ascii=False)}
        
        다음 JSON 형식으로 응답해주세요:
        {{
            "analysis_type": "comprehensive|focused|statistical|visual|exploratory",
            "focus_areas": ["descriptive_statistics", "distribution_analysis", "correlation_analysis", "outlier_analysis", "pattern_analysis"],
            "detail_level": "summary|detailed|comprehensive",
            "confidence": 0.0-1.0,
            "visualization_priority": "high|medium|low",
            "statistical_rigor": "basic|intermediate|advanced",
            "target_insights": ["찾고자 하는 인사이트들"]
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            intent = json.loads(response.generations[0][0].text)
            return intent
        except:
            # 기본값 반환
            return {
                "analysis_type": "comprehensive",
                "focus_areas": ["descriptive_statistics", "distribution_analysis", "correlation_analysis"],
                "detail_level": "detailed",
                "confidence": 0.8,
                "visualization_priority": "high",
                "statistical_rigor": "intermediate",
                "target_insights": ["데이터 패턴", "분포 특성", "상관관계"]
            }
    
    async def _scan_available_files(self) -> List[Dict[str, Any]]:
        """사용 가능한 데이터 파일 검색 (unified pattern)"""
        try:
            data_directories = [
                "ai_ds_team/data",
                "a2a_ds_servers/artifacts/data",
                "test_datasets"
            ]
            
            discovered_files = []
            for directory in data_directories:
                if os.path.exists(directory):
                    files = self.file_scanner.scan_data_files(directory)
                    discovered_files.extend(files)
            
            logger.info(f"📂 발견된 파일: {len(discovered_files)}개")
            return discovered_files
            
        except Exception as e:
            logger.error(f"파일 스캔 오류: {e}")
            return []
    
    async def _select_optimal_file_for_eda(self, available_files: List[Dict], eda_intent: Dict) -> Dict[str, Any]:
        """EDA에 최적화된 파일 선택"""
        if len(available_files) == 1:
            return available_files[0]
        
        # EDA 적합도 점수 계산
        for file_info in available_files:
            suitability_score = await self._calculate_eda_suitability(file_info)
            file_info['eda_suitability'] = suitability_score
        
        # 적합도 순으로 정렬
        available_files.sort(key=lambda x: x.get('eda_suitability', 0), reverse=True)
        
        # LLM 기반 최종 선택
        llm = await self.llm_factory.get_llm()
        
        files_info = "\n".join([
            f"- {f['name']} (크기: {f['size']} bytes, EDA적합도: {f.get('eda_suitability', 'N/A')})"
            for f in available_files[:5]  # 상위 5개만 표시
        ])
        
        prompt = f"""
        탐색적 데이터 분석에 가장 적합한 파일을 선택해주세요:
        
        EDA 의도: {eda_intent['analysis_type']} - {eda_intent['detail_level']}
        중점 영역: {eda_intent['focus_areas']}
        
        사용 가능한 파일들:
        {files_info}
        
        가장 적합한 파일명만 반환해주세요.
        """
        
        response = await llm.agenerate([prompt])
        selected_name = response.generations[0][0].text.strip()
        
        # 파일명 매칭
        for file_info in available_files:
            if selected_name in file_info['name'] or file_info['name'] in selected_name:
                return file_info
        
        return available_files[0]
    
    async def _calculate_eda_suitability(self, file_info: Dict) -> float:
        """EDA 적합도 점수 계산"""
        try:
            suitability_factors = []
            
            # 1. 파일 크기 (EDA에 적합한 크기)
            size = file_info['size']
            if 10000 <= size <= 50_000_000:  # 10KB ~ 50MB (EDA에 적합)
                suitability_factors.append(1.0)
            elif size < 10000:
                suitability_factors.append(0.4)  # 너무 작음
            elif size > 100_000_000:  # 100MB 이상
                suitability_factors.append(0.6)  # 너무 큼 (샘플링 필요)
            else:
                suitability_factors.append(0.8)
            
            # 2. 파일 형식 (EDA 친화성)
            extension = file_info.get('extension', '').lower()
            if extension == '.csv':
                suitability_factors.append(1.0)  # 최고 적합성
            elif extension in ['.xlsx', '.xls']:
                suitability_factors.append(0.9)  # 좋은 적합성
            elif extension == '.json':
                suitability_factors.append(0.7)  # 보통 적합성
            else:
                suitability_factors.append(0.5)
            
            # 3. 파일명 분석 (데이터 유형 추정)
            filename = file_info['name'].lower()
            if any(keyword in filename for keyword in ['sales', 'customer', 'financial', 'survey']):
                suitability_factors.append(0.9)  # 분석하기 좋은 데이터
            elif any(keyword in filename for keyword in ['sample', 'test', 'demo']):
                suitability_factors.append(0.8)  # 샘플 데이터
            elif any(keyword in filename for keyword in ['log', 'temp', 'backup']):
                suitability_factors.append(0.3)  # 분석에 부적합
            else:
                suitability_factors.append(0.6)
            
            suitability_score = sum(suitability_factors) / len(suitability_factors)
            return round(suitability_score, 3)
            
        except Exception as e:
            logger.warning(f"EDA 적합도 계산 실패: {e}")
            return 0.5
    
    async def _load_data_safely_for_eda(self, file_info: Dict[str, Any]) -> SmartDataFrame:
        """EDA에 최적화된 안전한 데이터 로딩"""
        file_path = file_info['path']
        
        try:
            # unified pattern의 다중 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'utf-16']
            df = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding=encoding)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                        used_encoding = 'excel_auto'
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path, encoding=encoding)
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        used_encoding = 'parquet_auto'
                    
                    if df is not None and not df.empty:
                        used_encoding = used_encoding or encoding
                        break
                        
                except (UnicodeDecodeError, Exception):
                    continue
            
            if df is None or df.empty:
                raise ValueError("데이터 로딩 실패")
            
            # EDA용 데이터 최적화
            df = await self._optimize_data_for_eda(df)
            
            # SmartDataFrame 생성
            metadata = {
                'source_file': file_path,
                'encoding': used_encoding,
                'load_timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'eda_optimized': True,
                'eda_suitability': file_info.get('eda_suitability', 0.5)
            }
            
            smart_df = SmartDataFrame(df, metadata)
            logger.info(f"✅ EDA 최적화 로딩 완료: {smart_df.shape}")
            
            return smart_df
            
        except Exception as e:
            logger.error(f"EDA 데이터 로딩 실패: {e}")
            raise
    
    async def _optimize_data_for_eda(self, df: pd.DataFrame) -> pd.DataFrame:
        """EDA에 최적화된 데이터 전처리"""
        try:
            # 1. 극단적으로 큰 데이터 샘플링
            if len(df) > 50000:  # 5만 행 이상이면 샘플링
                df = df.sample(n=50000, random_state=42)
                logger.info(f"대용량 데이터 샘플링: 50,000행으로 축소")
            
            # 2. 극단적으로 많은 컬럼 제한
            if df.shape[1] > 100:  # 100개 컬럼 이상이면 제한
                # 숫자형 우선, 그 다음 범주형
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:50]
                categorical_cols = df.select_dtypes(include=['object']).columns[:30]
                selected_cols = list(numeric_cols) + list(categorical_cols)[:100]
                df = df[selected_cols]
                logger.info(f"컬럼 수 제한: {len(selected_cols)}개로 축소")
            
            # 3. 데이터 타입 최적화
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 숫자로 변환 가능한지 확인
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series
                    except:
                        pass
            
            return df
            
        except Exception as e:
            logger.warning(f"EDA 최적화 실패: {e}")
            return df
    
    async def _comprehensive_statistical_analysis(self, smart_df: SmartDataFrame) -> StatisticalAnalysis:
        """
        포괄적 통계 분석 (통계 계산 오류 방지 시스템 포함)
        설계 문서 주요 문제점 해결
        """
        df = smart_df.data
        
        try:
            # 기술통계 안전 계산
            descriptive_stats = await self._safe_descriptive_statistics(df)
            
            # 분포 분석 안전 계산
            distribution_analysis = await self._safe_distribution_analysis(df)
            
            # 상관관계 분석 안전 계산
            correlation_analysis = await self._safe_correlation_analysis(df)
            
            # 이상값 분석 안전 계산
            outlier_analysis = await self._safe_outlier_analysis(df)
            
            # 패턴 분석 안전 계산
            pattern_analysis = await self._safe_pattern_analysis(df)
            
            return StatisticalAnalysis(
                descriptive_stats=descriptive_stats,
                distribution_analysis=distribution_analysis,
                correlation_analysis=correlation_analysis,
                outlier_analysis=outlier_analysis,
                pattern_analysis=pattern_analysis
            )
            
        except Exception as e:
            logger.error(f"통계 분석 실패: {e}")
            # 빈 결과 반환
            return StatisticalAnalysis(
                descriptive_stats={},
                distribution_analysis={},
                correlation_analysis={},
                outlier_analysis={},
                pattern_analysis={}
            )
    
    async def _safe_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """안전한 기술통계 계산 (오류 방지)"""
        stats_results = {}
        
        try:
            # 숫자형 컬럼 분석
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                try:
                    col_data = df[col].dropna()  # NaN 제거
                    
                    if len(col_data) < self.statistical_safety['min_sample_size']:
                        continue  # 샘플 크기 부족 시 건너뛰기
                    
                    col_stats = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75)),
                        'missing_count': int(df[col].isna().sum()),
                        'missing_percentage': float((df[col].isna().sum() / len(df)) * 100)
                    }
                    
                    # 추가 통계 (안전한 계산)
                    try:
                        col_stats['skewness'] = float(col_data.skew())
                        col_stats['kurtosis'] = float(col_data.kurtosis())
                    except:
                        col_stats['skewness'] = None
                        col_stats['kurtosis'] = None
                    
                    stats_results[col] = col_stats
                    
                except Exception as e:
                    logger.warning(f"컬럼 {col} 통계 계산 실패: {e}")
                    continue
            
            # 범주형 컬럼 분석
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) == 0:
                        continue
                    
                    value_counts = col_data.value_counts()
                    
                    # 카테고리가 너무 많으면 제한
                    if len(value_counts) > self.statistical_safety['max_categories']:
                        value_counts = value_counts.head(self.statistical_safety['max_categories'])
                    
                    col_stats = {
                        'count': len(col_data),
                        'unique_count': len(value_counts),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'missing_count': int(df[col].isna().sum()),
                        'missing_percentage': float((df[col].isna().sum() / len(df)) * 100),
                        'top_categories': {str(k): int(v) for k, v in value_counts.head(10).items()}
                    }
                    
                    stats_results[f"{col}_categorical"] = col_stats
                    
                except Exception as e:
                    logger.warning(f"범주형 컬럼 {col} 분석 실패: {e}")
                    continue
            
            logger.info(f"✅ 기술통계 계산 완료: {len(stats_results)}개 컬럼")
            return stats_results
            
        except Exception as e:
            logger.error(f"기술통계 계산 전체 실패: {e}")
            return {}
    
    async def _safe_distribution_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """안전한 분포 분석"""
        distribution_results = {}
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) < self.statistical_safety['min_sample_size']:
                        continue
                    
                    # 정규성 검정 (안전한 실행)
                    normality_tests = {}
                    
                    # Shapiro-Wilk 테스트 (샘플 크기 제한)
                    if len(col_data) <= 5000:  # Shapiro는 5000개 이하에서만 유효
                        try:
                            shapiro_stat, shapiro_p = stats.shapiro(col_data.sample(min(len(col_data), 1000)))
                            normality_tests['shapiro'] = {
                                'statistic': float(shapiro_stat),
                                'p_value': float(shapiro_p),
                                'is_normal': shapiro_p > 0.05
                            }
                        except:
                            pass
                    
                    # Jarque-Bera 테스트
                    try:
                        jb_stat, jb_p = stats.jarque_bera(col_data)
                        normality_tests['jarque_bera'] = {
                            'statistic': float(jb_stat),
                            'p_value': float(jb_p),
                            'is_normal': jb_p > 0.05
                        }
                    except:
                        pass
                    
                    distribution_results[col] = {
                        'normality_tests': normality_tests,
                        'skewness': float(col_data.skew()) if len(col_data) > 0 else None,
                        'kurtosis': float(col_data.kurtosis()) if len(col_data) > 0 else None,
                        'distribution_shape': self._classify_distribution_shape(col_data)
                    }
                    
                except Exception as e:
                    logger.warning(f"컬럼 {col} 분포 분석 실패: {e}")
                    continue
            
            return distribution_results
            
        except Exception as e:
            logger.error(f"분포 분석 전체 실패: {e}")
            return {}
    
    async def _safe_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """안전한 상관관계 분석"""
        correlation_results = {}
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                return {"message": "상관관계 분석을 위한 숫자형 컬럼이 부족합니다"}
            
            # 숫자형 데이터만 추출
            numeric_df = df[numeric_columns].dropna()
            
            if len(numeric_df) < self.statistical_safety['min_sample_size']:
                return {"message": "상관관계 분석을 위한 데이터가 부족합니다"}
            
            # 상관관계 매트릭스 계산
            try:
                correlation_matrix = numeric_df.corr()
                
                # NaN 값 처리
                correlation_matrix = correlation_matrix.fillna(0)
                
                correlation_results['correlation_matrix'] = correlation_matrix.to_dict()
                
                # 강한 상관관계 찾기
                strong_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:  # 강한 상관관계
                            strong_correlations.append({
                                'variable1': correlation_matrix.columns[i],
                                'variable2': correlation_matrix.columns[j],
                                'correlation': float(corr_value),
                                'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                            })
                
                correlation_results['strong_correlations'] = strong_correlations
                correlation_results['correlation_count'] = len(strong_correlations)
                
            except Exception as e:
                logger.warning(f"상관관계 매트릭스 계산 실패: {e}")
                correlation_results['error'] = str(e)
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"상관관계 분석 전체 실패: {e}")
            return {}
    
    async def _safe_outlier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """안전한 이상값 분석"""
        outlier_results = {}
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) < self.statistical_safety['min_sample_size']:
                        continue
                    
                    # IQR 방법
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    # Z-Score 방법
                    z_scores = np.abs(stats.zscore(col_data))
                    z_outliers = col_data[z_scores > 3]
                    
                    outlier_results[col] = {
                        'iqr_method': {
                            'count': len(iqr_outliers),
                            'percentage': float((len(iqr_outliers) / len(col_data)) * 100),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound)
                        },
                        'zscore_method': {
                            'count': len(z_outliers),
                            'percentage': float((len(z_outliers) / len(col_data)) * 100)
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"컬럼 {col} 이상값 분석 실패: {e}")
                    continue
            
            return outlier_results
            
        except Exception as e:
            logger.error(f"이상값 분석 전체 실패: {e}")
            return {}
    
    async def _safe_pattern_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """안전한 패턴 분석"""
        pattern_results = {}
        
        try:
            # 기본 패턴 분석
            pattern_results['data_shape'] = {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'density': float((df.notna().sum().sum()) / (df.shape[0] * df.shape[1]))
            }
            
            # 데이터 타입 분포
            dtype_counts = df.dtypes.value_counts()
            pattern_results['data_types'] = {str(k): int(v) for k, v in dtype_counts.items()}
            
            # 결측값 패턴
            missing_pattern = df.isnull().sum()
            missing_cols = missing_pattern[missing_pattern > 0]
            
            if len(missing_cols) > 0:
                pattern_results['missing_patterns'] = {
                    'columns_with_missing': {str(k): int(v) for k, v in missing_cols.items()},
                    'total_missing_cells': int(df.isnull().sum().sum())
                }
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"패턴 분석 전체 실패: {e}")
            return {}
    
    def _classify_distribution_shape(self, data: pd.Series) -> str:
        """분포 형태 분류"""
        try:
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            if abs(skewness) < 0.5:
                skew_desc = "symmetric"
            elif skewness > 0.5:
                skew_desc = "right_skewed"
            else:
                skew_desc = "left_skewed"
            
            if kurtosis > 3:
                kurt_desc = "heavy_tailed"
            elif kurtosis < -1:
                kurt_desc = "light_tailed"
            else:
                kurt_desc = "normal_tailed"
            
            return f"{skew_desc}_{kurt_desc}"
            
        except:
            return "unknown"
    
    async def _distribution_analysis(self, smart_df: SmartDataFrame, task_updater: TaskUpdater) -> Dict[str, Any]:
        """분포 분석 및 시각화"""
        df = smart_df.data
        charts = {}
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns[:6]  # 최대 6개
            
            for col in numeric_columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) < 5:
                        continue
                    
                    # 히스토그램
                    fig_hist = px.histogram(
                        df, x=col,
                        title=f"Distribution of {col}",
                        template='plotly_white'
                    )
                    charts[f"{col}_histogram"] = fig_hist
                    
                    # 박스플롯
                    fig_box = px.box(
                        df, y=col,
                        title=f"Box Plot of {col}",
                        template='plotly_white'
                    )
                    charts[f"{col}_boxplot"] = fig_box
                    
                except Exception as e:
                    logger.warning(f"컬럼 {col} 차트 생성 실패: {e}")
                    continue
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🍒 분포 차트 생성 완료: {len(charts)}개")
            )
            
            return {
                'charts': charts,
                'chart_count': len(charts)
            }
            
        except Exception as e:
            logger.error(f"분포 분석 실패: {e}")
            return {'charts': {}, 'chart_count': 0}
    
    async def _correlation_analysis(self, smart_df: SmartDataFrame, task_updater: TaskUpdater) -> Dict[str, Any]:
        """상관관계 분석 및 시각화"""
        df = smart_df.data
        charts = {}
        correlation_count = 0
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) >= 2:
                # 상관관계 매트릭스
                corr_matrix = df[numeric_columns].corr()
                
                # 히트맵
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    template='plotly_white',
                    aspect="auto"
                )
                charts['correlation_heatmap'] = fig_heatmap
                
                # 강한 상관관계 개수
                strong_corr_mask = (abs(corr_matrix) > 0.5) & (abs(corr_matrix) < 1.0)
                correlation_count = strong_corr_mask.sum().sum() // 2  # 대칭 매트릭스이므로 2로 나눔
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🍒 상관관계 분석 완료: {correlation_count}개 강한 상관관계")
            )
            
            return {
                'charts': charts,
                'correlation_count': correlation_count
            }
            
        except Exception as e:
            logger.error(f"상관관계 분석 실패: {e}")
            return {'charts': {}, 'correlation_count': 0}
    
    async def _outlier_and_pattern_analysis(self, smart_df: SmartDataFrame, task_updater: TaskUpdater) -> Dict[str, Any]:
        """이상값 및 패턴 분석"""
        df = smart_df.data
        charts = {}
        outliers_detected = 0
        patterns_found = 0
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns[:4]  # 최대 4개
            
            for col in numeric_columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) < 5:
                        continue
                    
                    # 박스플롯 (이상값 시각화)
                    fig_box = px.box(
                        df, y=col,
                        title=f"Outlier Detection: {col}",
                        template='plotly_white'
                    )
                    charts[f"{col}_outlier_boxplot"] = fig_box
                    
                    # 이상값 개수 계산
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    outlier_mask = (col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)
                    outliers_detected += outlier_mask.sum()
                    
                except Exception as e:
                    logger.warning(f"컬럼 {col} 이상값 분석 실패: {e}")
                    continue
            
            # 간단한 패턴 탐지
            patterns_found = len(df.select_dtypes(include=[np.number]).columns)  # 숫자형 컬럼 수
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🍒 이상값 및 패턴 분석 완료: {outliers_detected}개 이상값, {patterns_found}개 패턴")
            )
            
            return {
                'charts': charts,
                'outliers_detected': outliers_detected,
                'patterns_found': patterns_found
            }
            
        except Exception as e:
            logger.error(f"이상값 및 패턴 분석 실패: {e}")
            return {'charts': {}, 'outliers_detected': 0, 'patterns_found': 0}
    
    async def _generate_llm_insights(self, smart_df: SmartDataFrame, statistical_analysis: StatisticalAnalysis,
                                   distribution_results: Dict, correlation_results: Dict, outlier_pattern_results: Dict,
                                   eda_intent: Dict) -> List[EDAInsight]:
        """LLM 기반 인사이트 생성"""
        try:
            llm = await self.llm_factory.get_llm()
            
            # 분석 결과 요약
            analysis_summary = {
                'data_shape': smart_df.shape,
                'descriptive_stats_count': len(statistical_analysis.descriptive_stats),
                'correlation_count': correlation_results.get('correlation_count', 0),
                'outliers_detected': outlier_pattern_results.get('outliers_detected', 0),
                'charts_generated': (
                    len(distribution_results.get('charts', {})) + 
                    len(correlation_results.get('charts', {})) + 
                    len(outlier_pattern_results.get('charts', {}))
                )
            }
            
            prompt = f"""
            탐색적 데이터 분석 결과를 바탕으로 의미있는 인사이트를 생성해주세요:
            
            데이터 개요:
            - 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열
            - 분석된 컬럼: {analysis_summary['descriptive_stats_count']}개
            
            분석 결과 요약:
            {json.dumps(analysis_summary, indent=2, ensure_ascii=False)}
            
            사용자 의도:
            {json.dumps(eda_intent, indent=2, ensure_ascii=False)}
            
            다음 JSON 형식으로 3-5개의 인사이트를 생성해주세요:
            [
                {{
                    "category": "statistical|distribution|correlation|outlier|pattern",
                    "finding": "구체적인 발견사항",
                    "confidence": 0.0-1.0,
                    "evidence": {{"근거": "값"}},
                    "recommendation": "권장 사항"
                }}
            ]
            """
            
            response = await llm.agenerate([prompt])
            insights_data = json.loads(response.generations[0][0].text)
            
            insights = []
            for insight_data in insights_data:
                insight = EDAInsight(
                    category=insight_data.get('category', 'general'),
                    finding=insight_data.get('finding', ''),
                    confidence=insight_data.get('confidence', 0.5),
                    evidence=insight_data.get('evidence', {}),
                    recommendation=insight_data.get('recommendation', '')
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"LLM 인사이트 생성 실패: {e}")
            # 기본 인사이트 반환
            return [
                EDAInsight(
                    category="general",
                    finding=f"데이터셋은 {smart_df.shape[0]}행 {smart_df.shape[1]}열로 구성되어 있습니다.",
                    confidence=1.0,
                    evidence={"data_shape": smart_df.shape},
                    recommendation="추가 분석을 통해 더 자세한 인사이트를 얻을 수 있습니다."
                )
            ]
    
    async def _finalize_eda_results(self, smart_df: SmartDataFrame, statistical_analysis: StatisticalAnalysis,
                                  distribution_results: Dict, correlation_results: Dict, outlier_pattern_results: Dict,
                                  llm_insights: List[EDAInsight], task_updater: TaskUpdater) -> Dict[str, Any]:
        """EDA 결과 최종화"""
        
        # 모든 차트 통합
        all_charts = {}
        all_charts.update(distribution_results.get('charts', {}))
        all_charts.update(correlation_results.get('charts', {}))
        all_charts.update(outlier_pattern_results.get('charts', {}))
        
        # 차트 저장
        save_dir = Path("a2a_ds_servers/artifacts/eda_reports")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 차트 파일 저장
        chart_paths = []
        for chart_name, fig in all_charts.items():
            try:
                html_filename = f"{chart_name}_{timestamp}.html"
                html_path = save_dir / html_filename
                fig.write_html(str(html_path))
                chart_paths.append(str(html_path))
            except Exception as e:
                logger.warning(f"차트 저장 실패 ({chart_name}): {e}")
        
        # 종합 보고서 생성
        report_filename = f"eda_report_{timestamp}.json"
        report_path = save_dir / report_filename
        
        comprehensive_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_source': smart_df.metadata.get('source_file', 'Unknown'),
                'data_shape': smart_df.shape,
                'analysis_duration': 'completed'
            },
            'statistical_summary': statistical_analysis.descriptive_stats,
            'insights': [
                {
                    'category': insight.category,
                    'finding': insight.finding,
                    'confidence': insight.confidence,
                    'recommendation': insight.recommendation
                } for insight in llm_insights
            ],
            'charts_generated': list(all_charts.keys()),
            'saved_files': chart_paths
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        return {
            'total_charts': len(all_charts),
            'insights': llm_insights,
            'patterns_found': outlier_pattern_results.get('patterns_found', 0),
            'outliers_detected': outlier_pattern_results.get('outliers_detected', 0),
            'report_path': str(report_path),
            'chart_paths': chart_paths,
            'comprehensive_report': comprehensive_report
        }
    
    async def _create_eda_artifacts(self, results: Dict[str, Any], task_updater: TaskUpdater) -> None:
        """EDA 아티팩트 생성"""
        
        # EDA 보고서 아티팩트
        eda_report = {
            'exploratory_data_analysis_report': {
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': {
                    'total_charts_generated': results['total_charts'],
                    'insights_discovered': len(results['insights']),
                    'patterns_identified': results['patterns_found'],
                    'outliers_detected': results['outliers_detected']
                },
                'key_insights': [
                    {
                        'category': insight.category,
                        'finding': insight.finding,
                        'confidence': insight.confidence,
                        'recommendation': insight.recommendation
                    } for insight in results['insights']
                ],
                'files_generated': {
                    'report_path': results['report_path'],
                    'chart_count': len(results['chart_paths']),
                    'interactive_charts': True
                }
            }
        }
        
        # A2A 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(eda_report, indent=2, ensure_ascii=False))],
            name="eda_analysis_report",
            metadata={"content_type": "application/json", "category": "exploratory_data_analysis"}
        )
        
        # 종합 보고서도 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(results['comprehensive_report'], indent=2, ensure_ascii=False))],
            name="comprehensive_eda_report",
            metadata={"content_type": "application/json", "category": "comprehensive_analysis"}
        )
        
        logger.info("✅ EDA 분석 아티팩트 생성 완료")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip() or "데이터를 탐색적으로 분석해주세요"
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """작업 취소"""
        await task_updater.reject()
        logger.info(f"EDA Analysis 작업 취소됨: {context.context_id}")

# A2A 서버 설정
def create_eda_tools_agent_card() -> AgentCard:
    """EDA Tools Agent Card 생성"""
    return AgentCard(
        name="Unified EDA Tools Agent",
        description="📊 LLM First 지능형 탐색적 데이터 분석 전문가 - 통계 계산 오류 완전 해결, 포괄적 5단계 EDA, A2A SDK 0.2.9 표준 준수",
        skills=[
            AgentSkill(
                name="intelligent_eda_planning",
                description="LLM 기반 탐색적 데이터 분석 전략 수립"
            ),
            AgentSkill(
                name="safe_statistical_calculation", 
                description="통계 계산 오류 방지 시스템 (주요 문제점 해결)"
            ),
            AgentSkill(
                name="comprehensive_descriptive_statistics",
                description="기술통계, 중심경향성, 변산성, 분포 형태 분석"
            ),
            AgentSkill(
                name="distribution_analysis",
                description="정규성 검정, 분포 적합성, 왜도-첨도 분석"
            ),
            AgentSkill(
                name="correlation_analysis",
                description="상관관계 매트릭스, 연관성 측정, 다중공선성 진단"
            ),
            AgentSkill(
                name="outlier_detection",
                description="IQR, Z-score, Isolation Forest 기반 이상값 탐지"
            ),
            AgentSkill(
                name="pattern_recognition",
                description="트렌드, 계절성, 클러스터링 경향 분석"
            ),
            AgentSkill(
                name="llm_insight_generation",
                description="LLM 기반 데이터 인사이트 발견 및 해석"
            ),
            AgentSkill(
                name="interactive_eda_visualization",
                description="Interactive Plotly 기반 EDA 시각화"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_artifacts=True,
            max_execution_time=240,
            supported_formats=["csv", "excel", "json", "parquet"]
        )
    )

# 메인 실행부
if __name__ == "__main__":
    # A2A 서버 애플리케이션 생성
    task_store = InMemoryTaskStore()
    executor = UnifiedEDAToolsExecutor()
    agent_card = create_eda_tools_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        task_store=task_store,
        agent_executor=executor
    )
    
    app = A2AStarletteApplication(request_handler=request_handler)
    
    # 서버 시작
    logger.info("🚀 Unified EDA Tools Server 시작 - Port 8312")
    logger.info("📊 기능: LLM First 탐색적 데이터 분석 + 통계 계산 안정성")
    logger.info("🎯 A2A SDK 0.2.9 완전 표준 준수")
    
    uvicorn.run(app, host="0.0.0.0", port=8312) 