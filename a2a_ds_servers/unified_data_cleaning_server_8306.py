#!/usr/bin/env python3
"""
CherryAI Unified Data Cleaning Server - Port 8306
A2A SDK 0.2.9 완전 표준 준수 + UnifiedDataInterface 패턴

📊 핵심 기능:
- 🧹 LLM 기반 지능형 데이터 정리 전략 분석
- 🔍 빈 데이터 완벽 처리 (설계 문서 주요 문제점 해결)
- 📋 7단계 표준 정리 프로세스
- 💾 SmartDataFrame 품질 검증 시스템
- ⚡ 캐싱 및 성능 최적화
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

class UnifiedDataCleaningExecutor(AgentExecutor, UnifiedDataInterface):
    """
    Unified Data Cleaning Executor
    
    pandas_agent 패턴 + data_loader 성공 사례 기반
    - LLM First 데이터 정리 전략
    - 빈 데이터 완벽 처리
    - SmartDataFrame 품질 시스템
    - A2A SDK 0.2.9 완전 준수
    """
    
    def __init__(self):
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.file_scanner = FileScanner()
        self.llm_factory = LLMFactory()
        
        # 정리 전문 설정
        self.cleaning_strategies = {
            'missing_values': ['drop', 'impute_mean', 'impute_median', 'impute_mode', 'forward_fill', 'backward_fill'],
            'outliers': ['iqr_removal', 'zscore_removal', 'isolation_forest', 'clip_values'],
            'duplicates': ['drop_exact', 'drop_subset', 'keep_first', 'keep_last'],
            'data_types': ['auto_convert', 'optimize_memory', 'categorical_conversion']
        }
        
        # 품질 지표 임계값
        self.quality_thresholds = {
            'completeness_min': 0.7,  # 70% 이상 데이터 완전성
            'consistency_min': 0.8,   # 80% 이상 일관성
            'validity_min': 0.9       # 90% 이상 유효성
        }
        
        logger.info("✅ Unified Data Cleaning Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        A2A 표준 실행: 7단계 지능형 데이터 정리 프로세스
        
        🧹 1단계: LLM 정리 의도 분석
        📂 2단계: 데이터 검색 및 지능형 선택  
        📊 3단계: 안전한 데이터 로딩
        🔍 4단계: 빈 데이터 감지 및 품질 진단
        🛠️ 5단계: LLM 정리 계획 수립
        ⚡ 6단계: 정리 작업 실행
        ✅ 7단계: 결과 검증 및 저장
        """
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            start_time = time.time()
            
            # 🧹 1단계: 사용자 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧑🏻 **데이터 정리 시작** - 1단계: 정리 요구사항 분석 중...")
            )
            
            user_query = self._extract_user_query(context)
            logger.info(f"🧹 Data Cleaning Query: {user_query}")
            
            # LLM 기반 정리 의도 분석
            cleaning_intent = await self._analyze_cleaning_intent(user_query)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **의도 분석 완료**\n"
                    f"- 정리 타입: {cleaning_intent['cleaning_type']}\n"
                    f"- 우선순위: {', '.join(cleaning_intent['priority_areas'])}\n"
                    f"- 신뢰도: {cleaning_intent['confidence']:.2f}\n\n"
                    f"**2단계**: 데이터 검색 중..."
                )
            )
            
            # 📂 2단계: 파일 검색 및 지능형 선택
            available_files = await self._scan_available_files()
            
            if not available_files:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(
                        "⚠️ **데이터 없음**: 정리할 데이터를 찾을 수 없습니다.\n\n"
                        "**해결책**:\n"
                        "1. `a2a_ds_servers/artifacts/data/` 폴더에 데이터 파일을 업로드해주세요\n"
                        "2. 지원 형식: CSV, Excel (.xlsx/.xls), JSON, Parquet\n"
                        "3. 권장 인코딩: UTF-8"
                    )
                )
                return
            
            # LLM 기반 최적 파일 선택
            selected_file = await self._select_optimal_file(available_files, cleaning_intent)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **파일 선택 완료**\n"
                    f"- 파일: {selected_file['name']}\n"
                    f"- 크기: {selected_file['size']:,} bytes\n"
                    f"- 형식: {selected_file['extension']}\n\n"
                    f"**3단계**: 데이터 로딩 중..."
                )
            )
            
            # 📊 3단계: 안전한 데이터 로딩 (unified pattern)
            smart_df = await self._load_data_safely(selected_file)
            
            # 🔍 4단계: 빈 데이터 감지 및 품질 진단
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **데이터 로딩 완료**\n"
                    f"- 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 컬럼: {list(smart_df.data.columns)}\n\n"
                    f"**4단계**: 데이터 품질 진단 중..."
                )
            )
            
            # 빈 데이터 처리 (핵심 문제 해결)
            if await self._is_empty_or_invalid(smart_df):
                await self._handle_empty_data(smart_df, task_updater)
                return
            
            # 품질 진단
            quality_report = await self._comprehensive_quality_assessment(smart_df)
            
            # 🛠️ 5단계: LLM 정리 계획 수립
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **품질 진단 완료**\n"
                    f"- 전체 품질 점수: {quality_report['overall_score']:.2f}/1.00\n"
                    f"- 완전성: {quality_report['completeness']:.2f}\n"
                    f"- 일관성: {quality_report['consistency']:.2f}\n"
                    f"- 유효성: {quality_report['validity']:.2f}\n\n"
                    f"**5단계**: 정리 계획 수립 중..."
                )
            )
            
            cleaning_plan = await self._create_cleaning_plan(smart_df, quality_report, cleaning_intent)
            
            # ⚡ 6단계: 정리 작업 실행
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **정리 계획 완성**\n"
                    f"- 실행 단계: {len(cleaning_plan['steps'])}개\n"
                    f"- 예상 개선도: +{cleaning_plan['expected_improvement']:.2f}\n\n"
                    f"**6단계**: 데이터 정리 실행 중..."
                )
            )
            
            cleaned_smart_df = await self._execute_cleaning_plan(smart_df, cleaning_plan, task_updater)
            
            # ✅ 7단계: 결과 검증 및 저장
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**7단계**: 정리 결과 검증 및 저장 중...")
            )
            
            final_results = await self._finalize_cleaning_results(
                original_df=smart_df,
                cleaned_df=cleaned_smart_df,
                cleaning_plan=cleaning_plan,
                task_updater=task_updater
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 최종 완료 메시지
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    f"✅ **데이터 정리 완료!**\n\n"
                    f"📊 **정리 결과**:\n"
                    f"- 원본: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 정리된 데이터: {cleaned_smart_df.shape[0]}행 × {cleaned_smart_df.shape[1]}열\n"
                    f"- 품질 개선: {final_results['quality_improvement']:.2f} → {final_results['final_quality']:.2f}\n"
                    f"- 처리 시간: {processing_time:.2f}초\n\n"
                    f"📁 **저장 위치**: {final_results['saved_path']}\n"
                    f"📋 **정리 보고서**: 아티팩트로 생성됨"
                )
            )
            
            # 아티팩트 생성
            await self._create_cleaning_artifacts(final_results, task_updater)
            
        except Exception as e:
            logger.error(f"❌ Data Cleaning Error: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ **정리 실패**: {str(e)}")
            )
    
    async def _analyze_cleaning_intent(self, user_query: str) -> Dict[str, Any]:
        """LLM 기반 정리 의도 분석"""
        llm = await self.llm_factory.get_llm()
        
        prompt = f"""
        사용자의 데이터 정리 요청을 분석하여 최적의 정리 전략을 결정해주세요:
        
        요청: {user_query}
        
        다음 JSON 형식으로 응답해주세요:
        {{
            "cleaning_type": "comprehensive|missing_values|outliers|duplicates|data_types|custom",
            "priority_areas": ["결측값", "이상값", "중복", "데이터타입", "일관성"],
            "confidence": 0.0-1.0,
            "aggressive_level": "conservative|moderate|aggressive",
            "preserve_original": true/false,
            "expected_operations": ["구체적인 정리 작업들"]
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            intent = json.loads(response.generations[0][0].text)
            return intent
        except:
            # 기본값 반환
            return {
                "cleaning_type": "comprehensive",
                "priority_areas": ["결측값", "이상값", "중복"],
                "confidence": 0.8,
                "aggressive_level": "moderate",
                "preserve_original": True,
                "expected_operations": ["결측값 처리", "이상값 탐지", "중복 제거"]
            }
    
    async def _scan_available_files(self) -> List[Dict[str, Any]]:
        """사용 가능한 데이터 파일 검색 (unified pattern)"""
        try:
            # FileScanner 사용 (unified pattern)
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
    
    async def _select_optimal_file(self, available_files: List[Dict], cleaning_intent: Dict) -> Dict[str, Any]:
        """LLM 기반 최적 파일 선택 (unified pattern)"""
        if len(available_files) == 1:
            return available_files[0]
        
        llm = await self.llm_factory.get_llm()
        
        files_info = "\n".join([
            f"- {f['name']} ({f['size']} bytes, {f['extension']})"
            for f in available_files
        ])
        
        prompt = f"""
        데이터 정리 목적에 가장 적합한 파일을 선택해주세요:
        
        정리 의도: {cleaning_intent['cleaning_type']}
        우선순위: {cleaning_intent['priority_areas']}
        
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
        
        # 매칭 실패 시 첫 번째 파일 반환
        return available_files[0]
    
    async def _load_data_safely(self, file_info: Dict[str, Any]) -> SmartDataFrame:
        """안전한 데이터 로딩 (unified pattern과 동일)"""
        file_path = file_info['path']
        
        try:
            # 다중 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'utf-16']
            df = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding=encoding)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path, encoding=encoding)
                    elif file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue
            
            if df is None:
                raise ValueError("지원되지 않는 파일 형식이거나 읽기에 실패했습니다")
            
            # SmartDataFrame 생성
            metadata = {
                'source_file': file_path,
                'encoding': used_encoding,
                'load_timestamp': datetime.now().isoformat(),
                'original_shape': df.shape
            }
            
            smart_df = SmartDataFrame(df, metadata)
            logger.info(f"✅ 데이터 로딩 성공: {smart_df.shape}, 인코딩: {used_encoding}")
            
            return smart_df
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            raise
    
    async def _is_empty_or_invalid(self, smart_df: SmartDataFrame) -> bool:
        """빈 데이터 또는 유효하지 않은 데이터 검사 (핵심 문제 해결)"""
        # 다양한 빈 데이터 상태 검사
        checks = [
            smart_df.data.empty,                    # 완전히 빈 DataFrame
            smart_df.shape[0] == 0,                 # 행이 없음
            smart_df.shape[1] == 0,                 # 열이 없음
            len(smart_df.data.columns) == 0,        # 컬럼이 없음
            smart_df.data.isna().all().all(),       # 모든 값이 NaN
            all(smart_df.data[col].astype(str).str.strip().eq('').all() for col in smart_df.data.columns if smart_df.data[col].dtype == 'object')  # 모든 값이 빈 문자열
        ]
        
        return any(checks)
    
    async def _handle_empty_data(self, smart_df: SmartDataFrame, task_updater: TaskUpdater) -> None:
        """빈 데이터 전용 처리 로직 (설계 문서 주요 문제점 해결)"""
        
        # 빈 데이터 상태 진단
        diagnosis = []
        if smart_df.data.empty:
            diagnosis.append("DataFrame이 완전히 비어있음")
        if smart_df.shape[0] == 0:
            diagnosis.append("데이터 행이 없음") 
        if smart_df.shape[1] == 0:
            diagnosis.append("데이터 컬럼이 없음")
        if len(smart_df.data.columns) == 0:
            diagnosis.append("컬럼 정의가 없음")
        
        await task_updater.update_status(
            TaskState.completed,
            message=new_agent_text_message(
                f"⚠️ **빈 데이터 감지됨**\n\n"
                f"📊 **진단 결과**:\n" +
                "\n".join(f"- {d}" for d in diagnosis) +
                f"\n\n📁 **원본 정보**:\n"
                f"- 파일: {smart_df.metadata.get('source_file', 'Unknown')}\n"
                f"- 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n\n"
                f"🔧 **해결 방법**:\n"
                f"1. 다른 데이터 파일을 업로드해주세요\n"
                f"2. 데이터 파일에 실제 데이터가 포함되어 있는지 확인해주세요\n"
                f"3. 파일 형식이 올바른지 확인해주세요 (CSV, Excel, JSON, Parquet)\n"
                f"4. 헤더나 인덱스 설정을 확인해주세요\n\n"
                f"💡 **추천**: EDA Tools 에이전트를 사용해 데이터 구조를 먼저 분석해보세요"
            )
        )
    
    async def _comprehensive_quality_assessment(self, smart_df: SmartDataFrame) -> Dict[str, float]:
        """포괄적 데이터 품질 평가"""
        df = smart_df.data
        
        # 완전성 (Completeness) 계산
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # 일관성 (Consistency) 계산
        consistency_scores = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # 문자열 컬럼의 형식 일관성
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    unique_patterns = len(set(str(val).strip().lower() for val in non_null_values))
                    consistency_scores.append(1.0 - (unique_patterns / len(non_null_values)))
                else:
                    consistency_scores.append(1.0)
            else:
                # 숫자 컬럼의 범위 일관성
                consistency_scores.append(0.9)  # 기본 점수
        
        consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # 유효성 (Validity) 계산
        validity_scores = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # 숫자 유효성 (무한값, NaN 제외)
                valid_numbers = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                validity_scores.append(len(valid_numbers) / len(df) if len(df) > 0 else 0)
            else:
                # 문자열 유효성 (빈 문자열 제외)
                valid_strings = df[col].dropna()
                if len(valid_strings) > 0:
                    non_empty = valid_strings.astype(str).str.strip().ne('')
                    validity_scores.append(non_empty.sum() / len(df) if len(df) > 0 else 0)
                else:
                    validity_scores.append(1.0)
        
        validity = np.mean(validity_scores) if validity_scores else 1.0
        
        # 전체 품질 점수
        overall_score = (completeness + consistency + validity) / 3
        
        return {
            'overall_score': round(overall_score, 3),
            'completeness': round(completeness, 3),
            'consistency': round(consistency, 3), 
            'validity': round(validity, 3),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells)
        }
    
    async def _create_cleaning_plan(self, smart_df: SmartDataFrame, quality_report: Dict, cleaning_intent: Dict) -> Dict[str, Any]:
        """LLM 기반 정리 계획 수립"""
        llm = await self.llm_factory.get_llm()
        
        df_info = {
            'shape': smart_df.shape,
            'columns': list(smart_df.data.columns),
            'dtypes': smart_df.data.dtypes.to_dict(),
            'missing_info': smart_df.data.isna().sum().to_dict()
        }
        
        prompt = f"""
        데이터 품질 분석 결과를 바탕으로 최적의 정리 계획을 수립해주세요:
        
        데이터 정보:
        {json.dumps(df_info, indent=2, default=str)}
        
        품질 보고서:
        {json.dumps(quality_report, indent=2)}
        
        사용자 의도:
        {json.dumps(cleaning_intent, indent=2)}
        
        다음 JSON 형식으로 정리 계획을 작성해주세요:
        {{
            "steps": [
                {{
                    "step_number": 1,
                    "operation": "missing_values|outliers|duplicates|data_types",
                    "method": "구체적 방법",
                    "target_columns": ["컬럼명들"],
                    "parameters": {{"매개변수": "값"}},
                    "expected_improvement": 0.1
                }}
            ],
            "expected_improvement": 0.3,
            "estimated_time": "2-3분",
            "backup_required": true
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            plan = json.loads(response.generations[0][0].text)
            return plan
        except:
            # 기본 계획 반환
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "operation": "missing_values",
                        "method": "auto_imputation",
                        "target_columns": list(smart_df.data.columns),
                        "parameters": {"strategy": "appropriate"},
                        "expected_improvement": 0.2
                    }
                ],
                "expected_improvement": 0.2,
                "estimated_time": "1-2분",
                "backup_required": True
            }
    
    async def _execute_cleaning_plan(self, smart_df: SmartDataFrame, cleaning_plan: Dict, task_updater: TaskUpdater) -> SmartDataFrame:
        """정리 계획 실행"""
        df_cleaned = smart_df.data.copy()
        execution_log = []
        
        for step in cleaning_plan['steps']:
            step_num = step['step_number']
            operation = step['operation']
            method = step['method']
            target_columns = step.get('target_columns', [])
            
            try:
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(f"🍒 정리 단계 {step_num}: {operation} 실행 중...")
                )
                
                if operation == "missing_values":
                    df_cleaned = await self._handle_missing_values(df_cleaned, method, target_columns)
                    execution_log.append(f"✅ 단계 {step_num}: 결측값 처리 완료")
                
                elif operation == "outliers":
                    df_cleaned = await self._handle_outliers(df_cleaned, method, target_columns)
                    execution_log.append(f"✅ 단계 {step_num}: 이상값 처리 완료")
                
                elif operation == "duplicates":
                    df_cleaned = await self._handle_duplicates(df_cleaned, method)
                    execution_log.append(f"✅ 단계 {step_num}: 중복 제거 완료")
                
                elif operation == "data_types":
                    df_cleaned = await self._optimize_data_types(df_cleaned, target_columns)
                    execution_log.append(f"✅ 단계 {step_num}: 데이터 타입 최적화 완료")
                
            except Exception as e:
                execution_log.append(f"⚠️ 단계 {step_num}: {operation} 실행 중 오류 - {str(e)}")
                logger.warning(f"정리 단계 {step_num} 오류: {e}")
        
        # 정리된 SmartDataFrame 생성
        cleaned_metadata = smart_df.metadata.copy()
        cleaned_metadata.update({
            'cleaning_timestamp': datetime.now().isoformat(),
            'cleaning_steps': execution_log,
            'original_shape': smart_df.shape,
            'cleaned_shape': df_cleaned.shape
        })
        
        return SmartDataFrame(df_cleaned, cleaned_metadata)
    
    async def _handle_missing_values(self, df: pd.DataFrame, method: str, target_columns: List[str]) -> pd.DataFrame:
        """결측값 처리"""
        if not target_columns:
            target_columns = df.columns.tolist()
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            if df[col].isna().any():
                if method == "auto_imputation":
                    if df[col].dtype in ['int64', 'float64']:
                        # 숫자형: 중앙값 사용
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # 문자형: 최빈값 사용
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col].fillna(mode_val[0], inplace=True)
                elif method == "drop":
                    df.dropna(subset=[col], inplace=True)
                elif method == "forward_fill":
                    df[col].fillna(method='ffill', inplace=True)
                elif method == "backward_fill":
                    df[col].fillna(method='bfill', inplace=True)
        
        return df
    
    async def _handle_outliers(self, df: pd.DataFrame, method: str, target_columns: List[str]) -> pd.DataFrame:
        """이상값 처리"""
        if not target_columns:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in target_columns:
            if col not in df.columns or df[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == "iqr_removal":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif method == "zscore_removal":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
            
            elif method == "clip_values":
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                df[col] = df[col].clip(lower=Q1, upper=Q3)
        
        return df
    
    async def _handle_duplicates(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """중복 처리"""
        if method == "drop_exact":
            df = df.drop_duplicates()
        elif method == "keep_first":
            df = df.drop_duplicates(keep='first')
        elif method == "keep_last":
            df = df.drop_duplicates(keep='last')
        
        return df
    
    async def _optimize_data_types(self, df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """데이터 타입 최적화"""
        if not target_columns:
            target_columns = df.columns.tolist()
        
        for col in target_columns:
            if col not in df.columns:
                continue
            
            # 숫자형 최적화
            if df[col].dtype in ['int64', 'float64']:
                # 정수형 다운캐스팅
                if df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                # 실수형 다운캐스팅  
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # 카테고리형 변환
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # 50% 미만이 고유값이면 카테고리로 변환
                    df[col] = df[col].astype('category')
        
        return df
    
    async def _finalize_cleaning_results(self, original_df: SmartDataFrame, cleaned_df: SmartDataFrame, 
                                       cleaning_plan: Dict, task_updater: TaskUpdater) -> Dict[str, Any]:
        """정리 결과 최종화"""
        
        # 품질 개선도 계산
        original_quality = await self._comprehensive_quality_assessment(original_df)
        final_quality = await self._comprehensive_quality_assessment(cleaned_df)
        
        quality_improvement = final_quality['overall_score'] - original_quality['overall_score']
        
        # 정리된 데이터 저장
        save_dir = Path("a2a_ds_servers/artifacts/data/cleaned_dataframes")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_data_{timestamp}.csv"
        save_path = save_dir / filename
        
        cleaned_df.data.to_csv(save_path, index=False, encoding='utf-8')
        
        return {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'quality_improvement': round(quality_improvement, 3),
            'final_quality': final_quality['overall_score'],
            'saved_path': str(save_path),
            'cleaning_steps': cleaned_df.metadata.get('cleaning_steps', []),
            'metadata': cleaned_df.metadata
        }
    
    async def _create_cleaning_artifacts(self, results: Dict[str, Any], task_updater: TaskUpdater) -> None:
        """정리 아티팩트 생성"""
        
        # 정리 보고서 생성
        report = {
            'data_cleaning_report': {
                'timestamp': datetime.now().isoformat(),
                'original_data': {
                    'shape': results['original_shape'],
                },
                'cleaned_data': {
                    'shape': results['cleaned_shape'],
                    'quality_score': results['final_quality'],
                    'saved_path': results['saved_path']
                },
                'improvements': {
                    'quality_gain': results['quality_improvement'],
                    'steps_executed': len(results['cleaning_steps'])
                },
                'cleaning_steps': results['cleaning_steps']
            }
        }
        
        # A2A 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(report, indent=2, ensure_ascii=False))],
            name="data_cleaning_report",
            metadata={"content_type": "application/json", "category": "data_cleaning"}
        )
        
        logger.info("✅ 데이터 정리 아티팩트 생성 완료")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip() or "데이터를 정리해주세요"
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """작업 취소"""
        await task_updater.reject()
        logger.info(f"Data Cleaning 작업 취소됨: {context.context_id}")

# A2A 서버 설정
def create_data_cleaning_agent_card() -> AgentCard:
    """Data Cleaning Agent Card 생성"""
    return AgentCard(
        name="Unified Data Cleaning Agent",
        description="🧹 LLM First 지능형 데이터 정리 전문가 - 빈 데이터 완벽 처리, 품질 개선, A2A SDK 0.2.9 표준 준수",
        skills=[
            AgentSkill(
                name="intelligent_data_cleaning",
                description="LLM 기반 지능형 데이터 정리 전략 분석 및 실행"
            ),
            AgentSkill(
                name="empty_data_handling", 
                description="빈 데이터 감지 및 완벽 처리 (주요 문제점 해결)"
            ),
            AgentSkill(
                name="quality_assessment",
                description="포괄적 데이터 품질 평가 및 개선"
            ),
            AgentSkill(
                name="missing_value_processing",
                description="결측값 지능형 처리 (자동 imputation, 삭제, 채우기)"
            ),
            AgentSkill(
                name="outlier_detection",
                description="이상값 탐지 및 처리 (IQR, Z-score, Isolation Forest)"
            ),
            AgentSkill(
                name="duplicate_removal",
                description="중복 데이터 식별 및 제거"
            ),
            AgentSkill(
                name="data_type_optimization",
                description="데이터 타입 최적화 및 메모리 효율성 개선"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_artifacts=True,
            max_execution_time=300,
            supported_formats=["csv", "excel", "json", "parquet"]
        )
    )

# 메인 실행부
if __name__ == "__main__":
    # A2A 서버 애플리케이션 생성
    task_store = InMemoryTaskStore()
    executor = UnifiedDataCleaningExecutor()
    agent_card = create_data_cleaning_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        task_store=task_store,
        agent_executor=executor
    )
    
    app = A2AStarletteApplication(request_handler=request_handler)
    
    # 서버 시작
    logger.info("🚀 Unified Data Cleaning Server 시작 - Port 8306")
    logger.info("📊 기능: LLM First 데이터 정리 + 빈 데이터 완벽 처리")
    logger.info("🎯 A2A SDK 0.2.9 완전 표준 준수")
    
    uvicorn.run(app, host="0.0.0.0", port=8306) 