#!/usr/bin/env python3
"""
🍒 CherryAI 파일 업로드 프로세서

업로드된 파일을 A2A 시스템에 연동하는 로직
- 파일 업로드 검증 (CSV, Excel, JSON)
- A2A 시스템으로 파일 전달
- 파일 메타데이터 관리
- 파일 프리뷰 기능
"""

import io
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

SUPPORTED_FILE_TYPES = {
    'csv': 'text/csv',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel',
    'json': 'application/json'
}

@dataclass
class FileMetadata:
    """파일 메타데이터"""
    file_id: str
    filename: str
    file_type: str
    file_size: int
    upload_time: datetime = field(default_factory=datetime.now)
    rows: Optional[int] = None
    columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None
    preview_data: Optional[Dict[str, Any]] = None
    processing_status: str = "uploaded"  # uploaded, processing, processed, error
    error_message: Optional[str] = None

@dataclass
class ProcessedFile:
    """처리된 파일 정보"""
    metadata: FileMetadata
    dataframe: Optional[pd.DataFrame] = None
    raw_data: Optional[Any] = None
    a2a_ready: bool = False

class FileUploadProcessor:
    """파일 업로드 프로세서"""
    
    def __init__(self):
        # 처리된 파일들 저장
        self.processed_files: Dict[str, ProcessedFile] = {}
        
        # 설정
        self.config = {
            'max_file_size_mb': 100,
            'max_preview_rows': 5,
            'max_columns_display': 10,
            'auto_detect_encoding': True,
            'validate_data_quality': True,
            'enable_file_caching': True
        }
        
        # 통계
        self.stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_size_mb': 0.0
        }
    
    def validate_uploaded_files(self, uploaded_files: List[Any]) -> List[Tuple[bool, str, Any]]:
        """업로드된 파일들 검증"""
        validation_results = []
        
        for file in uploaded_files:
            try:
                # 파일 크기 확인
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > self.config['max_file_size_mb']:
                    validation_results.append((
                        False, 
                        f"파일 크기가 너무 큽니다: {file_size_mb:.1f}MB (최대 {self.config['max_file_size_mb']}MB)",
                        file
                    ))
                    continue
                
                # 파일 타입 확인
                file_extension = file.name.split('.')[-1].lower()
                if file_extension not in SUPPORTED_FILE_TYPES:
                    validation_results.append((
                        False,
                        f"지원하지 않는 파일 형식입니다: {file_extension}",
                        file
                    ))
                    continue
                
                validation_results.append((True, "검증 성공", file))
                
            except Exception as e:
                validation_results.append((
                    False,
                    f"파일 검증 실패: {str(e)}",
                    file
                ))
        
        return validation_results
    
    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[ProcessedFile]:
        """업로드된 파일들 처리"""
        processed_files = []
        
        # 파일 검증
        validation_results = self.validate_uploaded_files(uploaded_files)
        
        for is_valid, message, file in validation_results:
            self.stats['total_uploads'] += 1
            
            if not is_valid:
                logger.error(f"파일 검증 실패: {message}")
                self.stats['failed_uploads'] += 1
                continue
            
            try:
                # 파일 처리
                processed_file = self._process_single_file(file)
                processed_files.append(processed_file)
                
                # 캐시에 저장
                self.processed_files[processed_file.metadata.file_id] = processed_file
                
                self.stats['successful_uploads'] += 1
                self.stats['total_size_mb'] += file.size / (1024 * 1024)
                
                logger.info(f"파일 처리 완료: {file.name}")
                
            except Exception as e:
                logger.error(f"파일 처리 실패: {file.name} - {str(e)}")
                self.stats['failed_uploads'] += 1
        
        return processed_files
    
    def _process_single_file(self, file) -> ProcessedFile:
        """개별 파일 처리"""
        # 파일 메타데이터 생성
        file_id = str(uuid.uuid4())
        file_extension = file.name.split('.')[-1].lower()
        
        metadata = FileMetadata(
            file_id=file_id,
            filename=file.name,
            file_type=file_extension,
            file_size=file.size,
            processing_status="processing"
        )
        
        try:
            # 파일 타입별 처리
            if file_extension == 'csv':
                df, raw_data = self._process_csv_file(file)
            elif file_extension in ['xlsx', 'xls']:
                df, raw_data = self._process_excel_file(file)
            elif file_extension == 'json':
                df, raw_data = self._process_json_file(file)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_extension}")
            
            # 메타데이터 업데이트
            if df is not None:
                metadata.rows = len(df)
                metadata.columns = len(df.columns)
                metadata.column_names = df.columns.tolist()
                metadata.data_types = df.dtypes.astype(str).to_dict()
                metadata.preview_data = self._create_preview_data(df)
            
            metadata.processing_status = "processed"
            
            # ProcessedFile 객체 생성
            processed_file = ProcessedFile(
                metadata=metadata,
                dataframe=df,
                raw_data=raw_data,
                a2a_ready=True
            )
            
            return processed_file
            
        except Exception as e:
            metadata.processing_status = "error"
            metadata.error_message = str(e)
            
            return ProcessedFile(
                metadata=metadata,
                a2a_ready=False
            )
    
    def _process_csv_file(self, file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """CSV 파일 처리"""
        try:
            # 인코딩 자동 감지
            if self.config['auto_detect_encoding']:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file, encoding='cp949')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='latin-1')
            else:
                df = pd.read_csv(file)
            
            # Raw 데이터 정보
            raw_data = {
                'encoding': 'auto-detected',
                'separator': ',',
                'header_row': 0,
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
            
            return df, raw_data
            
        except Exception as e:
            raise Exception(f"CSV 파일 처리 실패: {str(e)}")
    
    def _process_excel_file(self, file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Excel 파일 처리"""
        try:
            # Excel 파일 읽기 (첫 번째 시트)
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            # 첫 번째 시트 읽기
            df = pd.read_excel(file, sheet_name=0)
            
            # Raw 데이터 정보
            raw_data = {
                'sheet_names': sheet_names,
                'active_sheet': sheet_names[0] if sheet_names else 'Sheet1',
                'total_sheets': len(sheet_names),
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
            
            return df, raw_data
            
        except Exception as e:
            raise Exception(f"Excel 파일 처리 실패: {str(e)}")
    
    def _process_json_file(self, file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """JSON 파일 처리"""
        try:
            # JSON 데이터 읽기
            json_data = json.load(file)
            
            # DataFrame으로 변환 시도
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                if 'data' in json_data:
                    df = pd.DataFrame(json_data['data'])
                else:
                    df = pd.DataFrame([json_data])
            else:
                raise ValueError("JSON 형식을 DataFrame으로 변환할 수 없습니다")
            
            # Raw 데이터 정보
            raw_data = {
                'json_type': type(json_data).__name__,
                'structure': 'list' if isinstance(json_data, list) else 'object',
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
            
            return df, raw_data
            
        except Exception as e:
            raise Exception(f"JSON 파일 처리 실패: {str(e)}")
    
    def _create_preview_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """미리보기 데이터 생성"""
        try:
            max_rows = self.config['max_preview_rows']
            max_cols = self.config['max_columns_display']
            
            # 미리보기용 DataFrame
            preview_df = df.head(max_rows)
            if len(df.columns) > max_cols:
                preview_df = preview_df.iloc[:, :max_cols]
            
            # 기본 통계
            basic_stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'null_counts': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            return {
                'preview_table': preview_df.to_dict('records'),
                'column_info': [
                    {
                        'name': col,
                        'type': str(df[col].dtype),
                        'null_count': int(df[col].isnull().sum()),
                        'unique_count': int(df[col].nunique())
                    }
                    for col in df.columns[:max_cols]
                ],
                'basic_stats': basic_stats
            }
            
        except Exception as e:
            logger.error(f"미리보기 데이터 생성 실패: {e}")
            return {'error': str(e)}
    
    def prepare_for_a2a_system(self, file_ids: List[str]) -> Dict[str, Any]:
        """A2A 시스템으로 전달할 데이터 준비"""
        a2a_data = {
            'files': [],
            'metadata': {
                'total_files': len(file_ids),
                'preparation_time': datetime.now().isoformat(),
                'processor_version': '1.0.0'
            }
        }
        
        for file_id in file_ids:
            if file_id not in self.processed_files:
                logger.warning(f"파일 ID를 찾을 수 없습니다: {file_id}")
                continue
            
            processed_file = self.processed_files[file_id]
            
            if not processed_file.a2a_ready:
                logger.warning(f"A2A 준비가 되지 않은 파일: {file_id}")
                continue
            
            # A2A 시스템용 파일 데이터
            file_data = {
                'file_id': file_id,
                'filename': processed_file.metadata.filename,
                'file_type': processed_file.metadata.file_type,
                'rows': processed_file.metadata.rows,
                'columns': processed_file.metadata.columns,
                'column_names': processed_file.metadata.column_names,
                'data_types': processed_file.metadata.data_types,
                'dataframe_json': processed_file.dataframe.to_json(orient='records') if processed_file.dataframe is not None else None,
                'raw_data_info': processed_file.raw_data
            }
            
            a2a_data['files'].append(file_data)
        
        return a2a_data
    
    def render_file_preview(self, file_id: str) -> None:
        """파일 미리보기 렌더링"""
        if file_id not in self.processed_files:
            st.error(f"파일을 찾을 수 없습니다: {file_id}")
            return
        
        processed_file = self.processed_files[file_id]
        metadata = processed_file.metadata
        
        # 파일 정보 헤더
        st.markdown(f"""
        ### 📄 {metadata.filename}
        - **파일 타입**: {metadata.file_type.upper()}
        - **크기**: {metadata.file_size / 1024:.1f} KB
        - **행 수**: {metadata.rows:,}개
        - **열 수**: {metadata.columns}개
        - **상태**: {metadata.processing_status}
        """)
        
        if metadata.processing_status == "error":
            st.error(f"❌ 처리 오류: {metadata.error_message}")
            return
        
        if metadata.preview_data and processed_file.dataframe is not None:
            # 미리보기 테이블
            st.markdown("#### 📊 데이터 미리보기")
            preview_df = pd.DataFrame(metadata.preview_data['preview_table'])
            st.dataframe(preview_df, use_container_width=True)
            
            # 컬럼 정보
            st.markdown("#### 📋 컬럼 정보")
            col_info = metadata.preview_data['column_info']
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
            
            # 기본 통계
            basic_stats = metadata.preview_data['basic_stats']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("메모리 사용량", f"{basic_stats['memory_usage_mb']} MB")
            with col2:
                st.metric("숫자형 컬럼", len(basic_stats['numeric_columns']))
            with col3:
                st.metric("텍스트 컬럼", len(basic_stats['categorical_columns']))
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """업로드 통계 반환"""
        success_rate = 0.0
        if self.stats['total_uploads'] > 0:
            success_rate = (self.stats['successful_uploads'] / self.stats['total_uploads']) * 100
        
        return {
            "total_uploads": self.stats['total_uploads'],
            "successful_uploads": self.stats['successful_uploads'],
            "failed_uploads": self.stats['failed_uploads'],
            "success_rate": round(success_rate, 1),
            "total_size_mb": round(self.stats['total_size_mb'], 2),
            "processed_files_count": len(self.processed_files),
            "supported_formats": list(SUPPORTED_FILE_TYPES.keys())
        }

# 전역 파일 업로드 프로세서 인스턴스
_file_processor = None

def get_file_upload_processor() -> FileUploadProcessor:
    """파일 업로드 프로세서 싱글톤 인스턴스 반환"""
    global _file_processor
    if _file_processor is None:
        _file_processor = FileUploadProcessor()
    return _file_processor

def process_and_prepare_files_for_a2a(uploaded_files: List[Any]) -> Optional[Dict[str, Any]]:
    """파일들을 처리하고 A2A 시스템으로 전달할 데이터 준비"""
    if not uploaded_files:
        return None
    
    processor = get_file_upload_processor()
    
    try:
        # 파일 처리
        processed_files = processor.process_uploaded_files(uploaded_files)
        
        if not processed_files:
            st.warning("처리된 파일이 없습니다.")
            return None
        
        # A2A 시스템용 데이터 준비
        file_ids = [pf.metadata.file_id for pf in processed_files if pf.a2a_ready]
        a2a_data = processor.prepare_for_a2a_system(file_ids)
        
        return a2a_data
        
    except Exception as e:
        logger.error(f"파일 처리 및 A2A 준비 실패: {e}")
        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        return None 