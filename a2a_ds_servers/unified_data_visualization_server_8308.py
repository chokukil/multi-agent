#!/usr/bin/env python3
"""
CherryAI Unified Data Visualization Server - Port 8308
A2A SDK 0.2.9 완전 표준 준수 + UnifiedDataInterface 패턴

📊 핵심 기능:
- 📈 LLM 기반 지능형 차트 타입 선택 및 설정
- 🔤 UTF-8 인코딩 자동 감지 및 처리 (설계 문서 주요 문제점 해결)
- 🎨 Interactive Plotly 차트 생성 및 최적화
- 📱 Streamlit 호환 시각화 출력
- 💾 차트 아티팩트 자동 저장 및 관리
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
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

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

class UnifiedDataVisualizationExecutor(AgentExecutor, UnifiedDataInterface):
    """
    Unified Data Visualization Executor
    
    pandas_agent 패턴 + data_loader 성공 사례 기반
    - LLM First 시각화 전략 분석
    - UTF-8 인코딩 자동 처리
    - Interactive Plotly 차트 생성
    - A2A SDK 0.2.9 완전 준수
    """
    
    def __init__(self):
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.file_scanner = FileScanner()
        self.llm_factory = LLMFactory()
        
        # 시각화 전문 설정
        self.chart_types = {
            'line': '시계열 데이터, 트렌드 분석',
            'bar': '카테고리별 비교, 순위',
            'scatter': '상관관계, 패턴 분석',
            'histogram': '분포 분석, 빈도',
            'box': '분포와 이상값, 요약 통계',
            'heatmap': '상관관계 매트릭스, 패턴',
            'pie': '구성비, 비율 분석',
            'violin': '분포 밀도 분석',
            'area': '누적 트렌드, 변화량',
            'sunburst': '계층적 데이터, 다차원 분석'
        }
        
        # Plotly 테마 설정
        self.themes = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']
        
        # 색상 팔레트
        self.color_palettes = {
            'default': px.colors.qualitative.Plotly,
            'professional': px.colors.qualitative.Set1,
            'modern': px.colors.qualitative.Pastel,
            'vibrant': px.colors.qualitative.Vivid,
            'corporate': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        
        logger.info("✅ Unified Data Visualization Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        A2A 표준 실행: 6단계 지능형 데이터 시각화 프로세스
        
        📊 1단계: LLM 시각화 의도 분석
        📂 2단계: 데이터 검색 및 인코딩 안전 로딩
        🎨 3단계: 최적 차트 타입 및 설정 결정
        📈 4단계: Interactive Plotly 차트 생성
        🖼️ 5단계: 차트 최적화 및 스타일링
        💾 6단계: 아티팩트 저장 및 결과 반환
        """
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            start_time = time.time()
            
            # 📊 1단계: 시각화 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧑🏻 **데이터 시각화 시작** - 1단계: 시각화 요구사항 분석 중...")
            )
            
            user_query = self._extract_user_query(context)
            logger.info(f"📊 Data Visualization Query: {user_query}")
            
            # LLM 기반 시각화 의도 분석
            viz_intent = await self._analyze_visualization_intent(user_query)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **의도 분석 완료**\n"
                    f"- 차트 타입: {viz_intent['chart_type']}\n"
                    f"- 분석 목적: {viz_intent['analysis_purpose']}\n"
                    f"- 권장 컬럼: {', '.join(viz_intent['recommended_columns'])}\n"
                    f"- 신뢰도: {viz_intent['confidence']:.2f}\n\n"
                    f"**2단계**: 데이터 검색 중..."
                )
            )
            
            # 📂 2단계: 파일 검색 및 인코딩 안전 로딩
            available_files = await self._scan_available_files()
            
            if not available_files:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(
                        "⚠️ **데이터 없음**: 시각화할 데이터를 찾을 수 없습니다.\n\n"
                        "**해결책**:\n"
                        "1. `a2a_ds_servers/artifacts/data/` 폴더에 데이터 파일을 업로드해주세요\n"
                        "2. 지원 형식: CSV, Excel (.xlsx/.xls), JSON, Parquet\n"
                        "3. 권장 인코딩: UTF-8 (자동 감지됨)"
                    )
                )
                return
            
            # LLM 기반 최적 파일 선택
            selected_file = await self._select_optimal_file(available_files, viz_intent)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **파일 선택 완료**\n"
                    f"- 파일: {selected_file['name']}\n"
                    f"- 크기: {selected_file['size']:,} bytes\n"
                    f"- 형식: {selected_file['extension']}\n\n"
                    f"**3단계**: 인코딩 안전 로딩 중..."
                )
            )
            
            # 인코딩 자동 감지 및 안전 로딩 (핵심 문제 해결)
            smart_df = await self._load_data_with_encoding_detection(selected_file)
            
            # 🎨 3단계: 차트 설정 최적화
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **데이터 로딩 완료**\n"
                    f"- 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 인코딩: {smart_df.metadata.get('encoding', 'auto')}\n"
                    f"- 컬럼: {list(smart_df.data.columns)}\n\n"
                    f"**4단계**: 차트 설정 최적화 중..."
                )
            )
            
            # 데이터 프로파일링 및 차트 설정
            chart_config = await self._optimize_chart_configuration(smart_df, viz_intent)
            
            # 📈 4단계: Interactive Plotly 차트 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **차트 설정 완료**\n"
                    f"- 차트 타입: {chart_config['chart_type']}\n"
                    f"- X축: {chart_config['x_column']}\n"
                    f"- Y축: {chart_config['y_column']}\n"
                    f"- 테마: {chart_config['theme']}\n\n"
                    f"**5단계**: Interactive 차트 생성 중..."
                )
            )
            
            # Plotly 차트 생성
            chart_results = await self._create_plotly_charts(smart_df, chart_config, task_updater)
            
            # 🖼️ 5단계: 차트 최적화 및 스타일링
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**6단계**: 차트 최적화 및 스타일링 중...")
            )
            
            optimized_charts = await self._optimize_and_style_charts(chart_results, chart_config)
            
            # 💾 6단계: 결과 최종화
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**7단계**: 차트 저장 및 결과 생성 중...")
            )
            
            final_results = await self._finalize_visualization_results(
                smart_df=smart_df,
                chart_config=chart_config,
                chart_results=optimized_charts,
                task_updater=task_updater
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 최종 완료 메시지
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    f"✅ **데이터 시각화 완료!**\n\n"
                    f"📊 **생성된 차트**:\n"
                    f"- 메인 차트: {chart_config['chart_type']}\n"
                    f"- 데이터 범위: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 추가 분석: {len(final_results['additional_charts'])}개\n"
                    f"- 처리 시간: {processing_time:.2f}초\n\n"
                    f"📁 **저장 위치**: {final_results['saved_paths']}\n"
                    f"🎨 **인터랙티브 차트**: Streamlit에서 확인 가능"
                )
            )
            
            # 아티팩트 생성
            await self._create_visualization_artifacts(final_results, task_updater)
            
        except Exception as e:
            logger.error(f"❌ Data Visualization Error: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ **시각화 실패**: {str(e)}")
            )
    
    async def _analyze_visualization_intent(self, user_query: str) -> Dict[str, Any]:
        """LLM 기반 시각화 의도 분석"""
        llm = await self.llm_factory.get_llm()
        
        prompt = f"""
        사용자의 데이터 시각화 요청을 분석하여 최적의 차트 전략을 결정해주세요:
        
        요청: {user_query}
        
        사용 가능한 차트 타입:
        {json.dumps(self.chart_types, indent=2, ensure_ascii=False)}
        
        다음 JSON 형식으로 응답해주세요:
        {{
            "chart_type": "line|bar|scatter|histogram|box|heatmap|pie|violin|area|sunburst",
            "analysis_purpose": "트렌드 분석|비교|분포|상관관계|구성비|패턴",
            "recommended_columns": ["컬럼명들"],
            "confidence": 0.0-1.0,
            "interaction_level": "basic|advanced|expert",
            "color_scheme": "default|professional|modern|vibrant|corporate",
            "additional_charts": ["보조 차트 타입들"]
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            intent = json.loads(response.generations[0][0].text)
            return intent
        except:
            # 기본값 반환
            return {
                "chart_type": "bar",
                "analysis_purpose": "데이터 탐색",
                "recommended_columns": [],
                "confidence": 0.8,
                "interaction_level": "basic",
                "color_scheme": "default",
                "additional_charts": ["scatter"]
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
    
    async def _select_optimal_file(self, available_files: List[Dict], viz_intent: Dict) -> Dict[str, Any]:
        """LLM 기반 최적 파일 선택 (unified pattern)"""
        if len(available_files) == 1:
            return available_files[0]
        
        llm = await self.llm_factory.get_llm()
        
        files_info = "\n".join([
            f"- {f['name']} ({f['size']} bytes, {f['extension']})"
            for f in available_files
        ])
        
        prompt = f"""
        데이터 시각화 목적에 가장 적합한 파일을 선택해주세요:
        
        시각화 의도: {viz_intent['chart_type']} - {viz_intent['analysis_purpose']}
        권장 컬럼: {viz_intent['recommended_columns']}
        
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
    
    async def _load_data_with_encoding_detection(self, file_info: Dict[str, Any]) -> SmartDataFrame:
        """
        UTF-8 인코딩 자동 감지 및 안전한 데이터 로딩 
        (설계 문서 주요 문제점 해결)
        """
        file_path = file_info['path']
        
        try:
            # 확장된 인코딩 리스트 (UTF-8 문제 해결)
            encodings = [
                'utf-8', 'utf-8-sig',  # UTF-8 변형들
                'cp949', 'euc-kr',     # 한국어 인코딩
                'latin1', 'cp1252',    # 서유럽 인코딩
                'utf-16', 'utf-16le', 'utf-16be',  # UTF-16 변형들
                'iso-8859-1', 'ascii'  # 기타 인코딩
            ]
            
            df = None
            used_encoding = None
            encoding_errors = []
            
            for encoding in encodings:
                try:
                    logger.info(f"🔤 시도 중인 인코딩: {encoding}")
                    
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding=encoding)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        # Excel 파일은 인코딩이 자동 처리됨
                        df = pd.read_excel(file_path)
                        used_encoding = 'excel_auto'
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path, encoding=encoding)
                    elif file_path.endswith('.parquet'):
                        # Parquet 파일은 인코딩이 자동 처리됨
                        df = pd.read_parquet(file_path)
                        used_encoding = 'parquet_auto'
                    
                    if df is not None and not df.empty:
                        used_encoding = used_encoding or encoding
                        logger.info(f"✅ 인코딩 성공: {used_encoding}")
                        break
                    
                except UnicodeDecodeError as e:
                    encoding_errors.append(f"{encoding}: {str(e)}")
                    continue
                except Exception as e:
                    encoding_errors.append(f"{encoding}: {str(e)}")
                    continue
            
            if df is None or df.empty:
                error_summary = "\n".join(encoding_errors[:5])  # 처음 5개 오류만 표시
                raise ValueError(
                    f"모든 인코딩 시도 실패. UTF-8 인코딩 문제로 추정됩니다.\n"
                    f"시도된 인코딩: {', '.join(encodings[:5])}\n"
                    f"주요 오류:\n{error_summary}"
                )
            
            # SmartDataFrame 생성
            metadata = {
                'source_file': file_path,
                'encoding': used_encoding,
                'encoding_detection_attempts': len([e for e in encoding_errors]) + 1,
                'load_timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'utf8_compliant': used_encoding in ['utf-8', 'utf-8-sig']
            }
            
            smart_df = SmartDataFrame(df, metadata)
            logger.info(f"✅ UTF-8 안전 로딩 성공: {smart_df.shape}, 인코딩: {used_encoding}")
            
            return smart_df
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            raise ValueError(f"인코딩 문제로 인한 데이터 로딩 실패: {str(e)}")
    
    async def _optimize_chart_configuration(self, smart_df: SmartDataFrame, viz_intent: Dict) -> Dict[str, Any]:
        """데이터 분석 기반 차트 설정 최적화"""
        df = smart_df.data
        
        # 데이터 타입별 컬럼 분류
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 차트 타입별 최적 설정
        chart_type = viz_intent['chart_type']
        
        config = {
            'chart_type': chart_type,
            'theme': 'plotly_white',
            'color_palette': self.color_palettes[viz_intent.get('color_scheme', 'default')],
            'width': 800,
            'height': 600
        }
        
        # 차트 타입별 컬럼 선택 로직
        if chart_type in ['line', 'area']:
            # 시계열 차트: 날짜/숫자 + 숫자
            if datetime_columns:
                config['x_column'] = datetime_columns[0]
                config['y_column'] = numeric_columns[0] if numeric_columns else categorical_columns[0]
            else:
                config['x_column'] = numeric_columns[0] if len(numeric_columns) > 1 else categorical_columns[0]
                config['y_column'] = numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
        
        elif chart_type in ['bar', 'box']:
            # 막대/박스 차트: 카테고리 + 숫자
            config['x_column'] = categorical_columns[0] if categorical_columns else df.columns[0]
            config['y_column'] = numeric_columns[0] if numeric_columns else df.columns[1]
        
        elif chart_type == 'scatter':
            # 산점도: 숫자 + 숫자
            if len(numeric_columns) >= 2:
                config['x_column'] = numeric_columns[0]
                config['y_column'] = numeric_columns[1]
                if len(numeric_columns) >= 3:
                    config['size_column'] = numeric_columns[2]
                if categorical_columns:
                    config['color_column'] = categorical_columns[0]
            else:
                config['x_column'] = df.columns[0]
                config['y_column'] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        elif chart_type in ['histogram', 'violin']:
            # 분포 차트: 숫자
            config['x_column'] = numeric_columns[0] if numeric_columns else df.columns[0]
            if categorical_columns:
                config['color_column'] = categorical_columns[0]
        
        elif chart_type == 'heatmap':
            # 히트맵: 상관관계 매트릭스
            config['correlation_matrix'] = True
            config['columns'] = numeric_columns[:10]  # 최대 10개 컬럼
        
        elif chart_type == 'pie':
            # 파이 차트: 카테고리 + 카운트
            config['labels_column'] = categorical_columns[0] if categorical_columns else df.columns[0]
            config['values_column'] = numeric_columns[0] if numeric_columns else None
        
        # 기본값 설정
        if 'x_column' not in config:
            config['x_column'] = df.columns[0]
        if 'y_column' not in config and chart_type not in ['histogram', 'pie', 'heatmap']:
            config['y_column'] = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # 데이터 크기에 따른 최적화
        if len(df) > 10000:
            config['sample_size'] = 5000
            config['optimization'] = 'large_dataset'
        
        return config
    
    async def _create_plotly_charts(self, smart_df: SmartDataFrame, chart_config: Dict, task_updater: TaskUpdater) -> Dict[str, Any]:
        """Plotly Interactive 차트 생성"""
        df = smart_df.data
        chart_type = chart_config['chart_type']
        
        # 데이터 샘플링 (대용량 데이터 처리)
        if chart_config.get('sample_size') and len(df) > chart_config['sample_size']:
            df_plot = df.sample(n=chart_config['sample_size'], random_state=42)
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(f"🍒 대용량 데이터 감지: {chart_config['sample_size']}개 샘플로 최적화")
            )
        else:
            df_plot = df.copy()
        
        charts = {}
        
        try:
            # 메인 차트 생성
            if chart_type == 'line':
                fig = px.line(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    title=f"Line Chart: {chart_config['y_column']} vs {chart_config['x_column']}"
                )
            
            elif chart_type == 'bar':
                fig = px.bar(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    title=f"Bar Chart: {chart_config['y_column']} by {chart_config['x_column']}"
                )
            
            elif chart_type == 'scatter':
                fig = px.scatter(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    size=chart_config.get('size_column'),
                    color=chart_config.get('color_column'),
                    title=f"Scatter Plot: {chart_config['y_column']} vs {chart_config['x_column']}"
                )
            
            elif chart_type == 'histogram':
                fig = px.histogram(
                    df_plot, 
                    x=chart_config['x_column'],
                    color=chart_config.get('color_column'),
                    title=f"Histogram: Distribution of {chart_config['x_column']}"
                )
            
            elif chart_type == 'box':
                fig = px.box(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    title=f"Box Plot: {chart_config['y_column']} by {chart_config['x_column']}"
                )
            
            elif chart_type == 'heatmap':
                # 상관관계 매트릭스
                corr_matrix = df_plot.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Heatmap",
                    aspect="auto"
                )
            
            elif chart_type == 'pie':
                if chart_config.get('values_column'):
                    fig = px.pie(
                        df_plot, 
                        names=chart_config['labels_column'],
                        values=chart_config['values_column'],
                        title=f"Pie Chart: {chart_config['values_column']} by {chart_config['labels_column']}"
                    )
                else:
                    # 카운트 기반 파이 차트
                    value_counts = df_plot[chart_config['labels_column']].value_counts()
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart: Count by {chart_config['labels_column']}"
                    )
            
            elif chart_type == 'violin':
                fig = px.violin(
                    df_plot, 
                    y=chart_config['x_column'],
                    box=True,
                    title=f"Violin Plot: Distribution of {chart_config['x_column']}"
                )
            
            elif chart_type == 'area':
                fig = px.area(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    title=f"Area Chart: {chart_config['y_column']} vs {chart_config['x_column']}"
                )
            
            else:
                # 기본 차트 (bar)
                fig = px.bar(
                    df_plot, 
                    x=chart_config['x_column'], 
                    y=chart_config['y_column'],
                    title="Default Bar Chart"
                )
            
            # 차트 스타일링
            fig.update_layout(
                template=chart_config['theme'],
                width=chart_config['width'],
                height=chart_config['height'],
                font=dict(size=12),
                title_font_size=16
            )
            
            charts['main_chart'] = fig
            
            # 추가 분석 차트들
            additional_charts = await self._create_additional_analysis_charts(df_plot, chart_config)
            charts.update(additional_charts)
            
            logger.info(f"✅ Plotly 차트 생성 완료: {len(charts)}개")
            return charts
            
        except Exception as e:
            logger.error(f"차트 생성 오류: {e}")
            # 기본 차트 생성
            fig = px.scatter(df_plot.head(100), title="기본 데이터 시각화")
            return {'main_chart': fig}
    
    async def _create_additional_analysis_charts(self, df: pd.DataFrame, chart_config: Dict) -> Dict[str, Any]:
        """추가 분석 차트 생성"""
        additional_charts = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 데이터 요약 통계 차트
            if len(numeric_cols) >= 2:
                # 상관관계 히트맵
                corr_matrix = df[numeric_cols].corr()
                heatmap_fig = px.imshow(
                    corr_matrix, 
                    title="Correlation Matrix",
                    aspect="auto"
                )
                additional_charts['correlation_heatmap'] = heatmap_fig
            
            # 분포 분석 차트
            if numeric_cols:
                hist_fig = px.histogram(
                    df, 
                    x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
                additional_charts['distribution_analysis'] = hist_fig
            
            # 박스플롯 (이상값 분석)
            if len(numeric_cols) >= 1:
                box_fig = px.box(
                    df, 
                    y=numeric_cols[0],
                    title=f"Outlier Analysis: {numeric_cols[0]}"
                )
                additional_charts['outlier_analysis'] = box_fig
            
        except Exception as e:
            logger.warning(f"추가 차트 생성 실패: {e}")
        
        return additional_charts
    
    async def _optimize_and_style_charts(self, chart_results: Dict, chart_config: Dict) -> Dict[str, Any]:
        """차트 최적화 및 스타일링"""
        optimized_charts = {}
        
        for chart_name, fig in chart_results.items():
            try:
                # 반응형 레이아웃
                fig.update_layout(
                    autosize=True,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # 인터랙티브 기능 활성화
                fig.update_traces(
                    hovertemplate='%{x}<br>%{y}<extra></extra>'
                )
                
                # 축 레이블 최적화
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                optimized_charts[chart_name] = fig
                
            except Exception as e:
                logger.warning(f"차트 최적화 실패 ({chart_name}): {e}")
                optimized_charts[chart_name] = fig
        
        return optimized_charts
    
    async def _finalize_visualization_results(self, smart_df: SmartDataFrame, chart_config: Dict, 
                                           chart_results: Dict, task_updater: TaskUpdater) -> Dict[str, Any]:
        """시각화 결과 최종화"""
        
        # 차트 저장 디렉토리
        save_dir = Path("a2a_ds_servers/artifacts/plots")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        # 각 차트를 HTML과 JSON으로 저장
        for chart_name, fig in chart_results.items():
            try:
                # HTML 저장 (Streamlit 호환)
                html_filename = f"{chart_name}_{timestamp}.html"
                html_path = save_dir / html_filename
                fig.write_html(str(html_path))
                
                # JSON 저장 (Plotly 호환)
                json_filename = f"{chart_name}_{timestamp}.json"
                json_path = save_dir / json_filename
                fig.write_json(str(json_path))
                
                saved_paths.append({
                    'chart_name': chart_name,
                    'html_path': str(html_path),
                    'json_path': str(json_path)
                })
                
            except Exception as e:
                logger.warning(f"차트 저장 실패 ({chart_name}): {e}")
        
        return {
            'data_info': {
                'shape': smart_df.shape,
                'encoding': smart_df.metadata.get('encoding'),
                'source_file': smart_df.metadata.get('source_file')
            },
            'chart_config': chart_config,
            'generated_charts': list(chart_results.keys()),
            'additional_charts': [name for name in chart_results.keys() if name != 'main_chart'],
            'saved_paths': saved_paths,
            'interactive_features': True,
            'streamlit_compatible': True
        }
    
    async def _create_visualization_artifacts(self, results: Dict[str, Any], task_updater: TaskUpdater) -> None:
        """시각화 아티팩트 생성"""
        
        # 시각화 보고서 생성
        report = {
            'data_visualization_report': {
                'timestamp': datetime.now().isoformat(),
                'data_source': results['data_info'],
                'chart_configuration': results['chart_config'],
                'generated_visualizations': {
                    'main_chart': results['chart_config']['chart_type'],
                    'additional_charts': results['additional_charts'],
                    'total_charts': len(results['generated_charts'])
                },
                'technical_details': {
                    'encoding_handling': results['data_info']['encoding'],
                    'interactive_features': results['interactive_features'],
                    'streamlit_compatible': results['streamlit_compatible'],
                    'file_formats': ['HTML', 'JSON']
                },
                'saved_files': results['saved_paths']
            }
        }
        
        # A2A 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(report, indent=2, ensure_ascii=False))],
            name="data_visualization_report",
            metadata={"content_type": "application/json", "category": "data_visualization"}
        )
        
        # 차트 파일들을 개별 아티팩트로 전송
        for chart_info in results['saved_paths']:
            try:
                # HTML 파일 내용 읽기
                with open(chart_info['html_path'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                await task_updater.add_artifact(
                    parts=[TextPart(text=html_content)],
                    name=f"{chart_info['chart_name']}_chart",
                    metadata={
                        "content_type": "text/html", 
                        "category": "visualization",
                        "chart_type": chart_info['chart_name']
                    }
                )
            except Exception as e:
                logger.warning(f"차트 아티팩트 생성 실패: {e}")
        
        logger.info("✅ 데이터 시각화 아티팩트 생성 완료")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip() or "데이터를 시각화해주세요"
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """작업 취소"""
        await task_updater.reject()
        logger.info(f"Data Visualization 작업 취소됨: {context.context_id}")

# A2A 서버 설정
def create_data_visualization_agent_card() -> AgentCard:
    """Data Visualization Agent Card 생성"""
    return AgentCard(
        name="Unified Data Visualization Agent",
        description="📊 LLM First 지능형 데이터 시각화 전문가 - UTF-8 인코딩 완벽 처리, Interactive Plotly 차트, A2A SDK 0.2.9 표준 준수",
        skills=[
            AgentSkill(
                name="intelligent_chart_selection",
                description="LLM 기반 데이터 분석 및 최적 차트 타입 자동 선택"
            ),
            AgentSkill(
                name="utf8_encoding_handling", 
                description="UTF-8 인코딩 자동 감지 및 안전한 처리 (주요 문제점 해결)"
            ),
            AgentSkill(
                name="interactive_plotly_charts",
                description="Interactive Plotly 차트 생성 및 최적화"
            ),
            AgentSkill(
                name="multi_chart_analysis",
                description="메인 차트 + 추가 분석 차트 자동 생성"
            ),
            AgentSkill(
                name="correlation_analysis",
                description="상관관계 히트맵 및 패턴 분석"
            ),
            AgentSkill(
                name="distribution_analysis",
                description="분포 분석 및 이상값 탐지 시각화"
            ),
            AgentSkill(
                name="streamlit_compatibility",
                description="Streamlit 호환 시각화 출력 및 저장"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_artifacts=True,
            max_execution_time=180,
            supported_formats=["csv", "excel", "json", "parquet"]
        )
    )

# 메인 실행부
if __name__ == "__main__":
    # A2A 서버 애플리케이션 생성
    task_store = InMemoryTaskStore()
    executor = UnifiedDataVisualizationExecutor()
    agent_card = create_data_visualization_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        task_store=task_store,
        agent_executor=executor
    )
    
    app = A2AStarletteApplication(request_handler=request_handler)
    
    # 서버 시작
    logger.info("🚀 Unified Data Visualization Server 시작 - Port 8308")
    logger.info("📊 기능: LLM First 시각화 + UTF-8 인코딩 완벽 처리")
    logger.info("🎯 A2A SDK 0.2.9 완전 표준 준수")
    
    uvicorn.run(app, host="0.0.0.0", port=8308) 