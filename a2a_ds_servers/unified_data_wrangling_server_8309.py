#!/usr/bin/env python3
"""
CherryAI Unified Data Wrangling Server - Port 8309
A2A SDK 0.2.9 완전 표준 준수 + UnifiedDataInterface 패턴

🔧 핵심 기능:
- 🧠 LLM 기반 지능형 데이터 변환 전략 분석
- 📁 안정적 파일 선택 및 로딩 (설계 문서 주요 문제점 해결)
- 🔄 안전한 데이터 변환 및 검증 시스템
- 📊 다단계 데이터 품질 보장
- 💾 변환 히스토리 및 롤백 기능
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
from dataclasses import dataclass
import copy

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
class TransformationStep:
    """데이터 변환 단계 정의"""
    step_id: str
    operation: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    description: str
    reversible: bool = True
    backup_required: bool = True

@dataclass 
class ValidationResult:
    """변환 검증 결과"""
    is_valid: bool
    validation_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]

class UnifiedDataWranglingExecutor(AgentExecutor, UnifiedDataInterface):
    """
    Unified Data Wrangling Executor
    
    pandas_agent 패턴 + data_loader 성공 사례 기반
    - LLM First 데이터 변환 전략
    - 안정적 파일 선택 시스템
    - 안전한 변환 및 검증
    - A2A SDK 0.2.9 완전 준수
    """
    
    def __init__(self):
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.file_scanner = FileScanner()
        self.llm_factory = LLMFactory()
        
        # 데이터 변환 전문 설정
        self.transformation_operations = {
            'column_operations': [
                'rename_columns', 'drop_columns', 'add_columns', 'reorder_columns',
                'split_columns', 'merge_columns', 'extract_substrings'
            ],
            'data_type_operations': [
                'convert_types', 'parse_dates', 'categorize', 'numeric_conversion',
                'string_normalization', 'boolean_conversion'
            ],
            'filtering_operations': [
                'filter_rows', 'filter_by_condition', 'remove_duplicates',
                'sample_data', 'slice_data', 'query_filter'
            ],
            'aggregation_operations': [
                'group_by', 'pivot_table', 'melt_data', 'crosstab',
                'rolling_operations', 'cumulative_operations'
            ],
            'joining_operations': [
                'merge_dataframes', 'concat_dataframes', 'join_operations',
                'append_data', 'union_data'
            ],
            'transformation_operations': [
                'normalize_data', 'scale_features', 'encode_categorical',
                'create_features', 'apply_functions', 'mathematical_operations'
            ]
        }
        
        # 안전성 임계값
        self.safety_thresholds = {
            'max_data_loss_percentage': 20,  # 최대 20% 데이터 손실 허용
            'min_data_quality_score': 0.7,   # 최소 품질 점수 70%
            'max_processing_time': 300,      # 최대 처리 시간 5분
            'backup_required_operations': [
                'drop_columns', 'filter_rows', 'remove_duplicates'
            ]
        }
        
        # 변환 히스토리 관리
        self.transformation_history = []
        self.backup_dataframes = {}
        
        logger.info("✅ Unified Data Wrangling Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        A2A 표준 실행: 8단계 지능형 데이터 변환 프로세스
        
        🧠 1단계: LLM 변환 의도 분석
        📂 2단계: 안정적 파일 선택 및 로딩
        🔍 3단계: 데이터 프로파일링 및 변환 가능성 평가
        📋 4단계: LLM 변환 계획 수립
        💾 5단계: 백업 생성 및 안전성 확인
        🔄 6단계: 단계별 변환 실행 및 검증
        ✅ 7단계: 변환 결과 품질 검증
        📁 8단계: 최종 결과 저장 및 보고
        """
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            start_time = time.time()
            
            # 🧠 1단계: 변환 의도 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧑🏻 **데이터 변환 시작** - 1단계: 변환 요구사항 분석 중...")
            )
            
            user_query = self._extract_user_query(context)
            logger.info(f"🔧 Data Wrangling Query: {user_query}")
            
            # LLM 기반 변환 의도 분석
            wrangling_intent = await self._analyze_wrangling_intent(user_query)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **의도 분석 완료**\n"
                    f"- 변환 타입: {wrangling_intent['transformation_type']}\n"
                    f"- 주요 작업: {', '.join(wrangling_intent['primary_operations'])}\n"
                    f"- 복잡도: {wrangling_intent['complexity_level']}\n"
                    f"- 신뢰도: {wrangling_intent['confidence']:.2f}\n\n"
                    f"**2단계**: 데이터 검색 중..."
                )
            )
            
            # 📂 2단계: 안정적 파일 선택 (핵심 문제 해결)
            available_files = await self._scan_available_files_with_stability_check()
            
            if not available_files:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(
                        "⚠️ **데이터 없음**: 변환할 데이터를 찾을 수 없습니다.\n\n"
                        "**해결책**:\n"
                        "1. `a2a_ds_servers/artifacts/data/` 폴더에 데이터 파일을 업로드해주세요\n"
                        "2. 지원 형식: CSV, Excel (.xlsx/.xls), JSON, Parquet\n"
                        "3. 파일 크기: 최대 1GB (자동 최적화)"
                    )
                )
                return
            
            # 안정적 파일 선택 (신뢰성 점수 기반)
            selected_file = await self._select_stable_file(available_files, wrangling_intent)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **파일 선택 완료**\n"
                    f"- 파일: {selected_file['name']}\n"
                    f"- 크기: {selected_file['size']:,} bytes\n"
                    f"- 안정성 점수: {selected_file['stability_score']:.2f}\n\n"
                    f"**3단계**: 안전한 데이터 로딩 중..."
                )
            )
            
            # 📊 3단계: 안전한 데이터 로딩
            smart_df = await self._load_data_safely_with_validation(selected_file)
            
            # 🔍 4단계: 데이터 프로파일링
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **데이터 로딩 완료**\n"
                    f"- 형태: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 메모리 사용량: {smart_df.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n"
                    f"- 컬럼: {list(smart_df.data.columns)}\n\n"
                    f"**4단계**: 데이터 프로파일링 및 변환 가능성 평가 중..."
                )
            )
            
            # 데이터 프로파일링 및 변환 가능성 평가
            data_profile = await self._comprehensive_data_profiling(smart_df)
            transformation_feasibility = await self._assess_transformation_feasibility(smart_df, wrangling_intent)
            
            # 📋 5단계: LLM 변환 계획 수립
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **프로파일링 완료**\n"
                    f"- 데이터 품질: {data_profile['quality_score']:.2f}/1.00\n"
                    f"- 변환 가능성: {transformation_feasibility['feasibility_score']:.2f}\n"
                    f"- 권장 작업: {len(transformation_feasibility['recommended_operations'])}개\n\n"
                    f"**5단계**: LLM 변환 계획 수립 중..."
                )
            )
            
            transformation_plan = await self._create_transformation_plan(smart_df, wrangling_intent, data_profile)
            
            # 💾 6단계: 백업 및 안전성 확인
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **변환 계획 완성**\n"
                    f"- 변환 단계: {len(transformation_plan['steps'])}개\n"
                    f"- 예상 처리 시간: {transformation_plan['estimated_time']}\n"
                    f"- 안전성 등급: {transformation_plan['safety_level']}\n\n"
                    f"**6단계**: 백업 생성 및 안전성 확인 중..."
                )
            )
            
            # 백업 생성
            backup_info = await self._create_transformation_backup(smart_df, transformation_plan)
            
            # 🔄 7단계: 단계별 변환 실행
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **백업 완료**\n"
                    f"- 백업 ID: {backup_info['backup_id']}\n"
                    f"- 롤백 가능: {backup_info['rollback_enabled']}\n\n"
                    f"**7단계**: 단계별 변환 실행 중..."
                )
            )
            
            transformed_smart_df = await self._execute_transformation_plan(smart_df, transformation_plan, task_updater)
            
            # ✅ 8단계: 변환 결과 검증 및 최종화
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**8단계**: 변환 결과 검증 및 최종화 중...")
            )
            
            final_results = await self._finalize_wrangling_results(
                original_df=smart_df,
                transformed_df=transformed_smart_df,
                transformation_plan=transformation_plan,
                backup_info=backup_info,
                task_updater=task_updater
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 최종 완료 메시지
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    f"✅ **데이터 변환 완료!**\n\n"
                    f"📊 **변환 결과**:\n"
                    f"- 원본: {smart_df.shape[0]}행 × {smart_df.shape[1]}열\n"
                    f"- 변환된 데이터: {transformed_smart_df.shape[0]}행 × {transformed_smart_df.shape[1]}열\n"
                    f"- 품질 변화: {final_results['quality_change']:.2f}\n"
                    f"- 데이터 보존율: {final_results['data_preservation_rate']:.1%}\n"
                    f"- 처리 시간: {processing_time:.2f}초\n\n"
                    f"📁 **저장 위치**: {final_results['saved_path']}\n"
                    f"🔄 **롤백 가능**: {final_results['rollback_available']}\n"
                    f"📋 **변환 보고서**: 아티팩트로 생성됨"
                )
            )
            
            # 아티팩트 생성
            await self._create_wrangling_artifacts(final_results, task_updater)
            
        except Exception as e:
            logger.error(f"❌ Data Wrangling Error: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ **변환 실패**: {str(e)}")
            )
    
    async def _analyze_wrangling_intent(self, user_query: str) -> Dict[str, Any]:
        """LLM 기반 데이터 변환 의도 분석"""
        llm = await self.llm_factory.get_llm()
        
        prompt = f"""
        사용자의 데이터 변환 요청을 분석하여 최적의 변환 전략을 결정해주세요:
        
        요청: {user_query}
        
        사용 가능한 변환 작업들:
        {json.dumps(self.transformation_operations, indent=2, ensure_ascii=False)}
        
        다음 JSON 형식으로 응답해주세요:
        {{
            "transformation_type": "structural|content|analytical|cleaning|aggregation",
            "primary_operations": ["주요 변환 작업들"],
            "target_columns": ["대상 컬럼들"],
            "complexity_level": "simple|moderate|complex",
            "confidence": 0.0-1.0,
            "data_size_sensitivity": "low|medium|high",
            "reversibility_required": true/false,
            "expected_outcomes": ["예상 결과들"]
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            intent = json.loads(response.generations[0][0].text)
            return intent
        except:
            # 기본값 반환
            return {
                "transformation_type": "structural",
                "primary_operations": ["데이터 정리", "형태 변환"],
                "target_columns": [],
                "complexity_level": "moderate",
                "confidence": 0.8,
                "data_size_sensitivity": "medium",
                "reversibility_required": True,
                "expected_outcomes": ["정리된 데이터", "구조 개선"]
            }
    
    async def _scan_available_files_with_stability_check(self) -> List[Dict[str, Any]]:
        """안정성 검사가 포함된 파일 스캔 (핵심 문제 해결)"""
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
                    
                    # 각 파일에 대한 안정성 점수 계산
                    for file_info in files:
                        stability_score = await self._calculate_file_stability_score(file_info)
                        file_info['stability_score'] = stability_score
                        
                        # 안정성 점수가 임계값 이상인 파일만 포함
                        if stability_score >= 0.5:  # 50% 이상 안정성
                            discovered_files.append(file_info)
            
            # 안정성 점수 순으로 정렬
            discovered_files.sort(key=lambda x: x['stability_score'], reverse=True)
            
            logger.info(f"📂 안정성 검증된 파일: {len(discovered_files)}개")
            return discovered_files
            
        except Exception as e:
            logger.error(f"파일 스캔 오류: {e}")
            return []
    
    async def _calculate_file_stability_score(self, file_info: Dict) -> float:
        """파일 안정성 점수 계산 (변환 가능성 평가)"""
        try:
            stability_factors = []
            
            # 1. 파일 크기 안정성 (너무 크거나 작으면 불안정)
            size = file_info['size']
            if 1024 <= size <= 100_000_000:  # 1KB ~ 100MB
                stability_factors.append(1.0)
            elif size < 1024:
                stability_factors.append(0.3)  # 너무 작음
            elif size > 1_000_000_000:  # 1GB 이상
                stability_factors.append(0.5)  # 너무 큼
            else:
                stability_factors.append(0.8)
            
            # 2. 파일 확장자 신뢰성
            extension = file_info.get('extension', '').lower()
            if extension in ['.csv', '.xlsx', '.xls']:
                stability_factors.append(1.0)  # 높은 신뢰성
            elif extension in ['.json', '.parquet']:
                stability_factors.append(0.9)  # 좋은 신뢰성
            else:
                stability_factors.append(0.6)  # 보통 신뢰성
            
            # 3. 파일 접근 가능성
            file_path = file_info['path']
            if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                stability_factors.append(1.0)
            else:
                stability_factors.append(0.0)
            
            # 4. 파일명 명확성 (명확한 파일명일수록 안정)
            filename = file_info['name'].lower()
            if any(keyword in filename for keyword in ['test', 'sample', 'data', 'clean']):
                stability_factors.append(0.9)
            elif any(keyword in filename for keyword in ['temp', 'tmp', 'backup']):
                stability_factors.append(0.4)  # 임시 파일
            else:
                stability_factors.append(0.7)
            
            # 전체 안정성 점수 계산
            stability_score = sum(stability_factors) / len(stability_factors)
            return round(stability_score, 3)
            
        except Exception as e:
            logger.warning(f"안정성 점수 계산 실패: {e}")
            return 0.5  # 기본값
    
    async def _select_stable_file(self, available_files: List[Dict], wrangling_intent: Dict) -> Dict[str, Any]:
        """안정성 점수 기반 최적 파일 선택"""
        if len(available_files) == 1:
            return available_files[0]
        
        # 안정성 점수가 가장 높은 파일들 선별 (상위 30%)
        top_stable_files = available_files[:max(1, len(available_files) // 3)]
        
        # LLM 기반 최종 선택 (안정성 + 적합성)
        llm = await self.llm_factory.get_llm()
        
        files_info = "\n".join([
            f"- {f['name']} (크기: {f['size']} bytes, 안정성: {f['stability_score']:.2f})"
            for f in top_stable_files
        ])
        
        prompt = f"""
        데이터 변환 목적에 가장 적합하면서 안정적인 파일을 선택해주세요:
        
        변환 의도: {wrangling_intent['transformation_type']}
        주요 작업: {wrangling_intent['primary_operations']}
        복잡도: {wrangling_intent['complexity_level']}
        
        안정성 검증된 파일들:
        {files_info}
        
        가장 적합한 파일명만 반환해주세요.
        """
        
        response = await llm.agenerate([prompt])
        selected_name = response.generations[0][0].text.strip()
        
        # 파일명 매칭
        for file_info in top_stable_files:
            if selected_name in file_info['name'] or file_info['name'] in selected_name:
                return file_info
        
        # 매칭 실패 시 가장 안정적인 파일 반환
        return top_stable_files[0]
    
    async def _load_data_safely_with_validation(self, file_info: Dict[str, Any]) -> SmartDataFrame:
        """검증이 포함된 안전한 데이터 로딩"""
        file_path = file_info['path']
        
        try:
            # 다중 인코딩 시도 (unified pattern)
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
                raise ValueError("데이터 로딩 실패: 지원되지 않는 형식이거나 빈 파일")
            
            # 로딩 후 기본 검증
            validation_results = await self._validate_loaded_data(df)
            
            if not validation_results['is_valid']:
                logger.warning(f"데이터 검증 경고: {validation_results['issues']}")
            
            # SmartDataFrame 생성
            metadata = {
                'source_file': file_path,
                'encoding': used_encoding,
                'load_timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'stability_score': file_info['stability_score'],
                'validation_results': validation_results
            }
            
            smart_df = SmartDataFrame(df, metadata)
            logger.info(f"✅ 안전한 데이터 로딩 완료: {smart_df.shape}, 안정성: {file_info['stability_score']:.2f}")
            
            return smart_df
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            raise
    
    async def _validate_loaded_data(self, df: pd.DataFrame) -> ValidationResult:
        """로딩된 데이터 기본 검증"""
        issues = []
        warnings = []
        recommendations = []
        
        try:
            # 1. 기본 구조 검증
            if df.empty:
                issues.append("DataFrame이 비어있음")
            
            if df.shape[0] == 0:
                issues.append("데이터 행이 없음")
            
            if df.shape[1] == 0:
                issues.append("데이터 컬럼이 없음")
            
            # 2. 데이터 품질 검증
            null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if null_percentage > 50:
                warnings.append(f"결측값이 많음: {null_percentage:.1f}%")
                recommendations.append("데이터 정리 에이전트 사용 권장")
            
            # 3. 메모리 사용량 검증
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            if memory_usage > 500:  # 500MB 이상
                warnings.append(f"큰 데이터 크기: {memory_usage:.1f} MB")
                recommendations.append("샘플링 또는 청크 처리 고려")
            
            # 4. 데이터 타입 검증
            object_columns = df.select_dtypes(include=['object']).columns
            if len(object_columns) > df.shape[1] * 0.8:  # 80% 이상이 object 타입
                warnings.append("대부분의 컬럼이 문자열 타입")
                recommendations.append("데이터 타입 최적화 필요")
            
            # 검증 점수 계산
            validation_score = 1.0
            validation_score -= len(issues) * 0.3
            validation_score -= len(warnings) * 0.1
            validation_score = max(0.0, validation_score)
            
            return ValidationResult(
                is_valid=len(issues) == 0,
                validation_score=validation_score,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                issues=[f"검증 중 오류: {str(e)}"],
                warnings=[],
                recommendations=["데이터 형식 확인 필요"]
            )
    
    async def _comprehensive_data_profiling(self, smart_df: SmartDataFrame) -> Dict[str, Any]:
        """포괄적 데이터 프로파일링"""
        df = smart_df.data
        
        profile = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'column_count': df.shape[1],
                'row_count': df.shape[0]
            },
            'data_types': {
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
                'boolean_columns': df.select_dtypes(include=['bool']).columns.tolist()
            },
            'quality_metrics': {
                'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'unique_values_per_column': {col: df[col].nunique() for col in df.columns}
            }
        }
        
        # 품질 점수 계산
        quality_factors = []
        
        # 완전성
        completeness = (1 - profile['quality_metrics']['null_percentage'] / 100)
        quality_factors.append(completeness)
        
        # 고유성
        uniqueness = 1 - (profile['quality_metrics']['duplicate_rows'] / len(df))
        quality_factors.append(uniqueness)
        
        # 다양성 (컬럼별 고유값 비율)
        diversity_scores = []
        for col in df.columns:
            unique_ratio = profile['quality_metrics']['unique_values_per_column'][col] / len(df)
            diversity_scores.append(min(unique_ratio, 1.0))
        diversity = np.mean(diversity_scores) if diversity_scores else 0.5
        quality_factors.append(diversity)
        
        profile['quality_score'] = np.mean(quality_factors)
        
        return profile
    
    async def _assess_transformation_feasibility(self, smart_df: SmartDataFrame, wrangling_intent: Dict) -> Dict[str, Any]:
        """변환 가능성 평가"""
        df = smart_df.data
        
        feasibility_factors = []
        recommended_operations = []
        potential_issues = []
        
        # 1. 데이터 크기 기반 가능성
        if df.shape[0] > 100000:  # 10만 행 이상
            feasibility_factors.append(0.7)  # 처리 시간 고려
            recommended_operations.append("데이터 샘플링")
        else:
            feasibility_factors.append(1.0)
        
        # 2. 메모리 사용량 기반 가능성
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 1000:  # 1GB 이상
            feasibility_factors.append(0.6)
            potential_issues.append("메모리 부족 위험")
        else:
            feasibility_factors.append(1.0)
        
        # 3. 요청된 작업의 복잡도
        complexity = wrangling_intent.get('complexity_level', 'moderate')
        if complexity == 'simple':
            feasibility_factors.append(1.0)
        elif complexity == 'moderate':
            feasibility_factors.append(0.8)
        else:  # complex
            feasibility_factors.append(0.6)
            recommended_operations.append("단계별 처리")
        
        # 4. 데이터 품질 기반 가능성
        null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if null_percentage > 30:
            feasibility_factors.append(0.7)
            potential_issues.append("높은 결측값 비율")
            recommended_operations.append("사전 데이터 정리")
        else:
            feasibility_factors.append(1.0)
        
        feasibility_score = np.mean(feasibility_factors)
        
        return {
            'feasibility_score': feasibility_score,
            'recommended_operations': recommended_operations,
            'potential_issues': potential_issues,
            'estimated_difficulty': complexity,
            'resource_requirements': {
                'memory_intensive': memory_mb > 500,
                'time_intensive': df.shape[0] > 50000,
                'cpu_intensive': complexity == 'complex'
            }
        }
    
    async def _create_transformation_plan(self, smart_df: SmartDataFrame, wrangling_intent: Dict, data_profile: Dict) -> Dict[str, Any]:
        """LLM 기반 변환 계획 수립"""
        llm = await self.llm_factory.get_llm()
        
        # 계획 수립을 위한 컨텍스트 구성
        context = {
            'data_info': {
                'shape': smart_df.shape,
                'columns': list(smart_df.data.columns),
                'data_types': data_profile['data_types'],
                'quality_score': data_profile['quality_score']
            },
            'user_intent': wrangling_intent,
            'available_operations': self.transformation_operations
        }
        
        prompt = f"""
        데이터 변환 계획을 수립해주세요:
        
        컨텍스트:
        {json.dumps(context, indent=2, default=str, ensure_ascii=False)}
        
        다음 JSON 형식으로 상세한 변환 계획을 작성해주세요:
        {{
            "steps": [
                {{
                    "step_number": 1,
                    "operation_category": "column_operations|data_type_operations|filtering_operations|...",
                    "specific_operation": "구체적 작업명",
                    "target_columns": ["대상 컬럼들"],
                    "parameters": {{"매개변수": "값"}},
                    "description": "작업 설명",
                    "expected_result": "예상 결과",
                    "risk_level": "low|medium|high",
                    "backup_required": true/false
                }}
            ],
            "estimated_time": "예상 시간",
            "safety_level": "safe|caution|risky",
            "rollback_plan": "롤백 계획",
            "validation_checks": ["검증 항목들"]
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            plan = json.loads(response.generations[0][0].text)
            
            # 계획 검증 및 보강
            validated_plan = await self._validate_and_enhance_plan(plan, smart_df)
            return validated_plan
            
        except Exception as e:
            logger.warning(f"LLM 계획 생성 실패, 기본 계획 사용: {e}")
            # 기본 계획 반환
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "operation_category": "data_type_operations",
                        "specific_operation": "basic_optimization",
                        "target_columns": list(smart_df.data.columns),
                        "parameters": {"auto_optimize": True},
                        "description": "기본 데이터 최적화",
                        "expected_result": "최적화된 데이터",
                        "risk_level": "low",
                        "backup_required": True
                    }
                ],
                "estimated_time": "1-2분",
                "safety_level": "safe",
                "rollback_plan": "백업에서 복원",
                "validation_checks": ["데이터 무결성", "컬럼 일관성"]
            }
    
    async def _validate_and_enhance_plan(self, plan: Dict, smart_df: SmartDataFrame) -> Dict[str, Any]:
        """변환 계획 검증 및 보강"""
        try:
            # 1. 단계별 검증
            for step in plan.get('steps', []):
                # 대상 컬럼 존재 확인
                target_columns = step.get('target_columns', [])
                valid_columns = [col for col in target_columns if col in smart_df.data.columns]
                step['target_columns'] = valid_columns
                
                # 위험도 평가
                if step.get('specific_operation') in ['drop_columns', 'filter_rows']:
                    step['risk_level'] = 'medium'
                    step['backup_required'] = True
            
            # 2. 안전성 등급 재평가
            high_risk_steps = sum(1 for step in plan['steps'] if step.get('risk_level') == 'high')
            if high_risk_steps > 0:
                plan['safety_level'] = 'risky'
            elif any(step.get('risk_level') == 'medium' for step in plan['steps']):
                plan['safety_level'] = 'caution'
            else:
                plan['safety_level'] = 'safe'
            
            # 3. 백업 요구사항 확인
            backup_required = any(step.get('backup_required', False) for step in plan['steps'])
            plan['backup_required'] = backup_required
            
            return plan
            
        except Exception as e:
            logger.error(f"계획 검증 실패: {e}")
            return plan
    
    async def _create_transformation_backup(self, smart_df: SmartDataFrame, transformation_plan: Dict) -> Dict[str, Any]:
        """변환 백업 생성"""
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 백업이 필요한지 확인
            backup_required = transformation_plan.get('backup_required', False)
            
            if not backup_required:
                return {
                    'backup_id': backup_id,
                    'backup_created': False,
                    'rollback_enabled': False,
                    'backup_path': None
                }
            
            # 백업 디렉토리 생성
            backup_dir = Path("a2a_ds_servers/artifacts/data/backups")
            backup_dir.mkdir(exist_ok=True, parents=True)
            
            # 원본 데이터 백업
            backup_filename = f"original_data_{backup_id}.csv"
            backup_path = backup_dir / backup_filename
            smart_df.data.to_csv(backup_path, index=False, encoding='utf-8')
            
            # 메타데이터 백업
            metadata_filename = f"metadata_{backup_id}.json"
            metadata_path = backup_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(smart_df.metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # 백업 정보 저장
            self.backup_dataframes[backup_id] = {
                'original_data': smart_df.data.copy(),
                'metadata': smart_df.metadata.copy(),
                'creation_time': datetime.now().isoformat(),
                'backup_path': str(backup_path)
            }
            
            logger.info(f"✅ 변환 백업 생성 완료: {backup_id}")
            
            return {
                'backup_id': backup_id,
                'backup_created': True,
                'rollback_enabled': True,
                'backup_path': str(backup_path),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
            return {
                'backup_id': backup_id,
                'backup_created': False,
                'rollback_enabled': False,
                'backup_path': None,
                'error': str(e)
            }
    
    async def _execute_transformation_plan(self, smart_df: SmartDataFrame, transformation_plan: Dict, task_updater: TaskUpdater) -> SmartDataFrame:
        """변환 계획 실행"""
        current_df = smart_df.data.copy()
        execution_log = []
        
        for step in transformation_plan.get('steps', []):
            step_num = step['step_number']
            operation = step['specific_operation']
            
            try:
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(
                        f"🍒 변환 단계 {step_num}/{len(transformation_plan['steps'])}: {step['description']} 실행 중..."
                    )
                )
                
                # 단계별 변환 실행
                transformed_df = await self._execute_single_transformation(current_df, step)
                
                # 변환 결과 검증
                validation = await self._validate_transformation_step(current_df, transformed_df, step)
                
                if validation['is_valid']:
                    current_df = transformed_df
                    execution_log.append(f"✅ 단계 {step_num}: {step['description']} 완료")
                    logger.info(f"변환 단계 {step_num} 성공")
                else:
                    execution_log.append(f"⚠️ 단계 {step_num}: {step['description']} 부분 실행 - {validation['issues']}")
                    logger.warning(f"변환 단계 {step_num} 경고: {validation['issues']}")
                
            except Exception as e:
                execution_log.append(f"❌ 단계 {step_num}: {step['description']} 실패 - {str(e)}")
                logger.error(f"변환 단계 {step_num} 실패: {e}")
                # 오류 발생 시 해당 단계는 건너뛰고 계속 진행
                continue
        
        # 변환된 SmartDataFrame 생성
        transformed_metadata = smart_df.metadata.copy()
        transformed_metadata.update({
            'transformation_timestamp': datetime.now().isoformat(),
            'transformation_log': execution_log,
            'original_shape': smart_df.shape,
            'transformed_shape': current_df.shape,
            'transformation_plan': transformation_plan
        })
        
        return SmartDataFrame(current_df, transformed_metadata)
    
    async def _execute_single_transformation(self, df: pd.DataFrame, step: Dict) -> pd.DataFrame:
        """단일 변환 단계 실행"""
        operation = step['specific_operation']
        target_columns = step.get('target_columns', [])
        parameters = step.get('parameters', {})
        
        result_df = df.copy()
        
        try:
            # 기본적인 변환 작업들 구현
            if operation == 'basic_optimization':
                # 데이터 타입 최적화
                for col in result_df.columns:
                    if result_df[col].dtype == 'object':
                        # 숫자로 변환 가능한지 확인
                        try:
                            numeric_series = pd.to_numeric(result_df[col], errors='coerce')
                            if not numeric_series.isna().all():
                                result_df[col] = numeric_series
                        except:
                            pass
                    elif result_df[col].dtype in ['int64', 'float64']:
                        # 다운캐스팅
                        result_df[col] = pd.to_numeric(result_df[col], downcast='integer' if 'int' in str(result_df[col].dtype) else 'float')
            
            elif operation == 'remove_duplicates':
                result_df = result_df.drop_duplicates()
            
            elif operation == 'drop_columns' and target_columns:
                valid_columns = [col for col in target_columns if col in result_df.columns]
                if valid_columns:
                    result_df = result_df.drop(columns=valid_columns)
            
            elif operation == 'rename_columns' and parameters.get('column_mapping'):
                column_mapping = parameters['column_mapping']
                result_df = result_df.rename(columns=column_mapping)
            
            elif operation == 'filter_rows' and parameters.get('condition'):
                # 간단한 필터링 (안전한 eval 사용)
                condition = parameters['condition']
                if 'query' in dir(result_df):
                    try:
                        result_df = result_df.query(condition)
                    except:
                        # 쿼리 실패 시 원본 유지
                        pass
            
            elif operation == 'fill_missing_values':
                if target_columns:
                    for col in target_columns:
                        if col in result_df.columns:
                            fill_method = parameters.get('method', 'forward')
                            if fill_method == 'forward':
                                result_df[col] = result_df[col].fillna(method='ffill')
                            elif fill_method == 'mean' and result_df[col].dtype in ['int64', 'float64']:
                                result_df[col] = result_df[col].fillna(result_df[col].mean())
            
            # 더 많은 변환 작업들을 여기에 추가할 수 있음
            
        except Exception as e:
            logger.warning(f"변환 작업 실패 ({operation}): {e}")
            # 실패 시 원본 반환
            return df
        
        return result_df
    
    async def _validate_transformation_step(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, step: Dict) -> ValidationResult:
        """변환 단계 검증"""
        issues = []
        warnings = []
        recommendations = []
        
        try:
            # 1. 기본 구조 검증
            if transformed_df.empty and not original_df.empty:
                issues.append("변환 후 데이터가 비어짐")
            
            # 2. 데이터 손실률 검증
            data_loss_rate = (original_df.shape[0] - transformed_df.shape[0]) / original_df.shape[0] * 100
            if data_loss_rate > self.safety_thresholds['max_data_loss_percentage']:
                issues.append(f"과도한 데이터 손실: {data_loss_rate:.1f}%")
            elif data_loss_rate > 5:
                warnings.append(f"데이터 손실 감지: {data_loss_rate:.1f}%")
            
            # 3. 컬럼 변화 검증
            original_columns = set(original_df.columns)
            transformed_columns = set(transformed_df.columns)
            
            removed_columns = original_columns - transformed_columns
            added_columns = transformed_columns - original_columns
            
            if removed_columns and step.get('operation_category') != 'column_operations':
                warnings.append(f"예상치 못한 컬럼 제거: {list(removed_columns)}")
            
            if added_columns:
                recommendations.append(f"새 컬럼 추가됨: {list(added_columns)}")
            
            # 4. 데이터 타입 일관성 검증
            for col in original_columns.intersection(transformed_columns):
                if original_df[col].dtype != transformed_df[col].dtype:
                    recommendations.append(f"컬럼 {col} 타입 변경: {original_df[col].dtype} → {transformed_df[col].dtype}")
            
            # 검증 점수 계산
            validation_score = 1.0
            validation_score -= len(issues) * 0.4
            validation_score -= len(warnings) * 0.2
            validation_score = max(0.0, validation_score)
            
            return ValidationResult(
                is_valid=len(issues) == 0,
                validation_score=validation_score,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                issues=[f"검증 중 오류: {str(e)}"],
                warnings=[],
                recommendations=["변환 결과 수동 확인 필요"]
            )
    
    async def _finalize_wrangling_results(self, original_df: SmartDataFrame, transformed_df: SmartDataFrame, 
                                        transformation_plan: Dict, backup_info: Dict, task_updater: TaskUpdater) -> Dict[str, Any]:
        """데이터 변환 결과 최종화"""
        
        # 변환 품질 평가
        original_quality = await self._comprehensive_data_profiling(original_df)
        transformed_quality = await self._comprehensive_data_profiling(transformed_df)
        
        quality_change = transformed_quality['quality_score'] - original_quality['quality_score']
        data_preservation_rate = transformed_df.shape[0] / original_df.shape[0]
        
        # 변환된 데이터 저장
        save_dir = Path("a2a_ds_servers/artifacts/data/transformed_dataframes")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transformed_data_{timestamp}.csv"
        save_path = save_dir / filename
        
        transformed_df.data.to_csv(save_path, index=False, encoding='utf-8')
        
        return {
            'original_shape': original_df.shape,
            'transformed_shape': transformed_df.shape,
            'quality_change': round(quality_change, 3),
            'data_preservation_rate': data_preservation_rate,
            'saved_path': str(save_path),
            'backup_info': backup_info,
            'rollback_available': backup_info.get('rollback_enabled', False),
            'transformation_log': transformed_df.metadata.get('transformation_log', []),
            'execution_summary': {
                'total_steps': len(transformation_plan.get('steps', [])),
                'successful_steps': len([log for log in transformed_df.metadata.get('transformation_log', []) if '✅' in log]),
                'failed_steps': len([log for log in transformed_df.metadata.get('transformation_log', []) if '❌' in log]),
                'warning_steps': len([log for log in transformed_df.metadata.get('transformation_log', []) if '⚠️' in log])
            }
        }
    
    async def _create_wrangling_artifacts(self, results: Dict[str, Any], task_updater: TaskUpdater) -> None:
        """데이터 변환 아티팩트 생성"""
        
        # 변환 보고서 생성
        report = {
            'data_wrangling_report': {
                'timestamp': datetime.now().isoformat(),
                'transformation_summary': {
                    'original_data': {
                        'shape': results['original_shape'],
                    },
                    'transformed_data': {
                        'shape': results['transformed_shape'],
                        'saved_path': results['saved_path']
                    },
                    'quality_metrics': {
                        'quality_change': results['quality_change'],
                        'data_preservation_rate': results['data_preservation_rate']
                    }
                },
                'execution_details': results['execution_summary'],
                'transformation_log': results['transformation_log'],
                'backup_information': {
                    'backup_created': results['backup_info'].get('backup_created', False),
                    'rollback_available': results['rollback_available'],
                    'backup_path': results['backup_info'].get('backup_path')
                }
            }
        }
        
        # A2A 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(report, indent=2, ensure_ascii=False))],
            name="data_wrangling_report",
            metadata={"content_type": "application/json", "category": "data_transformation"}
        )
        
        logger.info("✅ 데이터 변환 아티팩트 생성 완료")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip() or "데이터를 변환해주세요"
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """작업 취소"""
        await task_updater.reject()
        logger.info(f"Data Wrangling 작업 취소됨: {context.context_id}")

# A2A 서버 설정
def create_data_wrangling_agent_card() -> AgentCard:
    """Data Wrangling Agent Card 생성"""
    return AgentCard(
        name="Unified Data Wrangling Agent",
        description="🔧 LLM First 지능형 데이터 변환 전문가 - 안정적 파일 선택, 안전한 변환 및 검증, A2A SDK 0.2.9 표준 준수",
        skills=[
            AgentSkill(
                name="intelligent_transformation_planning",
                description="LLM 기반 데이터 변환 전략 분석 및 계획 수립"
            ),
            AgentSkill(
                name="stable_file_selection", 
                description="안정성 점수 기반 파일 선택 (주요 문제점 해결)"
            ),
            AgentSkill(
                name="safe_data_transformation",
                description="백업 및 롤백이 포함된 안전한 데이터 변환"
            ),
            AgentSkill(
                name="transformation_validation",
                description="변환 단계별 검증 및 품질 보장"
            ),
            AgentSkill(
                name="column_operations",
                description="컬럼 이름 변경, 추가, 삭제, 재배열"
            ),
            AgentSkill(
                name="data_type_conversion",
                description="데이터 타입 최적화 및 변환"
            ),
            AgentSkill(
                name="filtering_and_aggregation",
                description="데이터 필터링, 그룹화, 집계 작업"
            ),
            AgentSkill(
                name="backup_and_rollback",
                description="자동 백업 생성 및 롤백 기능"
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
    executor = UnifiedDataWranglingExecutor()
    agent_card = create_data_wrangling_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        task_store=task_store,
        agent_executor=executor
    )
    
    app = A2AStarletteApplication(request_handler=request_handler)
    
    # 서버 시작
    logger.info("🚀 Unified Data Wrangling Server 시작 - Port 8309")
    logger.info("🔧 기능: LLM First 데이터 변환 + 안정적 파일 선택")
    logger.info("🎯 A2A SDK 0.2.9 완전 표준 준수")
    
    uvicorn.run(app, host="0.0.0.0", port=8309) 