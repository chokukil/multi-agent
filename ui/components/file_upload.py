"""
파일 업로드 컴포넌트
이미지 + pandas 지원 포맷 (CSV, Excel, JSON 등)
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
import io
import os
from pathlib import Path
import tempfile
import mimetypes

class FileUploadManager:
    """파일 업로드 관리자"""
    
    # pandas 지원 포맷
    PANDAS_FORMATS = {
        'csv': ['.csv'],
        'excel': ['.xlsx', '.xls'],
        'json': ['.json'],
        'parquet': ['.parquet'],
        'feather': ['.feather'],
        'pickle': ['.pkl', '.pickle'],
        'tsv': ['.tsv'],
        'txt': ['.txt']
    }
    
    # 이미지 포맷
    IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
    
    def __init__(self, session_key: str = "uploaded_files"):
        self.session_key = session_key
        
        # 세션 상태 초기화
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
    
    def get_supported_extensions(self) -> List[str]:
        """지원되는 파일 확장자 목록"""
        pandas_exts = []
        for format_name, exts in self.PANDAS_FORMATS.items():
            pandas_exts.extend(exts)
        
        return pandas_exts + self.IMAGE_FORMATS
    
    def get_file_info(self, file) -> Dict[str, Any]:
        """파일 정보 추출"""
        file_info = {
            'name': file.name,
            'size': file.size,
            'type': file.type,
            'extension': Path(file.name).suffix.lower(),
            'is_image': False,
            'is_data': False,
            'format': None
        }
        
        # 이미지 파일 체크
        if file_info['extension'] in self.IMAGE_FORMATS:
            file_info['is_image'] = True
            file_info['format'] = 'image'
        
        # 데이터 파일 체크
        for format_name, exts in self.PANDAS_FORMATS.items():
            if file_info['extension'] in exts:
                file_info['is_data'] = True
                file_info['format'] = format_name
                break
        
        return file_info
    
    def load_data_file(self, file, file_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """데이터 파일 로드"""
        try:
            format_name = file_info['format']
            
            if format_name == 'csv':
                # CSV 파일 로드 (자동 구분자 감지)
                sample = file.read(1024).decode('utf-8', errors='ignore')
                file.seek(0)
                
                # 구분자 자동 감지
                if '\t' in sample:
                    separator = '\t'
                elif ';' in sample:
                    separator = ';'
                else:
                    separator = ','
                
                df = pd.read_csv(file, sep=separator, encoding='utf-8')
                
            elif format_name == 'excel':
                df = pd.read_excel(file, engine='openpyxl')
                
            elif format_name == 'json':
                df = pd.read_json(file)
                
            elif format_name == 'parquet':
                df = pd.read_parquet(file)
                
            elif format_name == 'feather':
                df = pd.read_feather(file)
                
            elif format_name == 'pickle':
                df = pd.read_pickle(file)
                
            elif format_name == 'tsv':
                df = pd.read_csv(file, sep='\t', encoding='utf-8')
                
            elif format_name == 'txt':
                df = pd.read_csv(file, sep='\t', encoding='utf-8')
                
            else:
                return None
            
            return df
            
        except Exception as e:
            st.error(f"파일 로드 중 오류 발생: {str(e)}")
            return None
    
    def render_upload_area(self, max_files: int = 5) -> List[Dict[str, Any]]:
        """파일 업로드 영역 렌더링 - 간소화된 UI"""
        
        # 파일 업로드 위젯 - 별도 컨테이너 없이 바로 표시
        uploaded_files = st.file_uploader(
            "파일을 선택하거나 드래그하여 업로드하세요",
            accept_multiple_files=True,
            type=None,  # 모든 파일 타입 허용 (내부에서 필터링)
            key="file_uploader",
            help="지원 포맷: CSV, Excel, JSON, Parquet, TSV, JPEG, PNG, GIF 등"
        )
        
        processed_files = []
        
        if uploaded_files:
            for file in uploaded_files:
                file_info = self.get_file_info(file)
                
                # 지원되는 포맷인지 체크
                if file_info['extension'] not in self.get_supported_extensions():
                    st.error(f"❌ 지원되지 않는 파일 형식: {file.name}")
                    continue
                
                # 파일 처리
                processed_file = {
                    'file': file,
                    'info': file_info,
                    'data': None,
                    'preview': None
                }
                
                # 데이터 파일 로드
                if file_info['is_data']:
                    df = self.load_data_file(file, file_info)
                    if df is not None:
                        processed_file['data'] = df
                        processed_file['preview'] = self.generate_data_preview(df)
                
                # 이미지 파일 처리
                elif file_info['is_image']:
                    processed_file['preview'] = self.generate_image_preview(file)
                
                processed_files.append(processed_file)
        
        # 업로드된 파일 미리보기 - 접힌 상태
        if processed_files:
            self.render_file_previews_collapsed(processed_files)
        
        # 세션 상태 업데이트
        st.session_state[self.session_key] = processed_files
        
        return processed_files
    
    def generate_data_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 미리보기 생성"""
        preview = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'head': df.head().to_dict('records'),
            'info': {
                'memory_usage': df.memory_usage(deep=True).sum(),
                'null_counts': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum()
            }
        }
        return preview
    
    def generate_image_preview(self, file) -> Dict[str, Any]:
        """이미지 미리보기 생성"""
        try:
            from PIL import Image
            
            # 이미지 로드
            image = Image.open(file)
            
            preview = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            return preview
            
        except Exception as e:
            return {'error': str(e)}
    
    def render_file_previews(self, processed_files: List[Dict[str, Any]]) -> None:
        """파일 미리보기 렌더링"""
        st.markdown("### 📋 업로드된 파일")
        
        for i, processed_file in enumerate(processed_files):
            file_info = processed_file['info']
            
            with st.expander(f"📄 {file_info['name']} ({self.format_file_size(file_info['size'])})", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**파일 정보:**")
                    st.markdown(f"- 크기: {self.format_file_size(file_info['size'])}")
                    st.markdown(f"- 타입: {file_info['format']}")
                    st.markdown(f"- 확장자: {file_info['extension']}")
                
                with col2:
                    if file_info['is_data'] and processed_file['data'] is not None:
                        self.render_data_preview(processed_file['data'], processed_file['preview'])
                    elif file_info['is_image']:
                        self.render_image_preview(processed_file['file'], processed_file['preview'])
    
    def render_data_preview(self, df: pd.DataFrame, preview: Dict[str, Any]) -> None:
        """데이터 미리보기 렌더링"""
        st.markdown(f"**데이터 정보:** {preview['shape'][0]} rows × {preview['shape'][1]} columns")
        
        # 기본 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 행수", preview['shape'][0])
        with col2:
            st.metric("총 열수", preview['shape'][1])
        with col3:
            st.metric("중복 행수", preview['info']['duplicates'])
        
        # 데이터 미리보기
        st.markdown("**데이터 미리보기:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 컬럼 정보
        if len(preview['columns']) > 0:
            st.markdown("**컬럼 정보:**")
            col_info = []
            for col in preview['columns']:
                col_info.append({
                    '컬럼명': col,
                    '데이터타입': str(preview['dtypes'][col]),
                    '결측값': preview['info']['null_counts'][col]
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    def render_image_preview(self, file, preview: Dict[str, Any]) -> None:
        """이미지 미리보기 렌더링"""
        if 'error' in preview:
            st.error(f"이미지 로드 실패: {preview['error']}")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**이미지 정보:**")
            st.markdown(f"- 크기: {preview['size'][0]} × {preview['size'][1]}")
            st.markdown(f"- 모드: {preview['mode']}")
            st.markdown(f"- 포맷: {preview['format']}")
            if preview['has_transparency']:
                st.markdown("- 투명도: 있음")
        
        with col2:
            st.image(file, caption=file.name, use_container_width=True)
    
    def format_file_size(self, size_bytes: int) -> str:
        """파일 크기 포맷팅"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """업로드된 파일 목록 반환"""
        return st.session_state.get(self.session_key, [])
    
    def clear_uploaded_files(self) -> None:
        """업로드된 파일 초기화"""
        st.session_state[self.session_key] = []

    def render_file_previews_collapsed(self, processed_files: List[Dict[str, Any]]) -> None:
        """업로드된 파일 미리보기 - 접힌 상태"""
        
        # 간단한 상태 표시
        file_count = len(processed_files)
        data_files = [f for f in processed_files if f['info'].get('is_data', False)]
        image_files = [f for f in processed_files if f['info'].get('is_image', False)]
        
        # 업로드 상태 요약
        st.success(f"✅ {file_count}개 파일 업로드 완료 (데이터: {len(data_files)}개, 이미지: {len(image_files)}개)")
        
        # 상세 미리보기는 접힌 상태로
        with st.expander("📋 업로드된 파일 상세보기", expanded=False):
            for i, processed_file in enumerate(processed_files):
                file_info = processed_file['info']
                
                # 파일 기본 정보
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**📄 {file_info['name']}**")
                
                with col2:
                    st.caption(f"타입: {file_info['format']}")
                
                with col3:
                    st.caption(f"크기: {self.format_file_size(file_info['size'])}")
                
                # 데이터 파일 미리보기
                if file_info['is_data'] and processed_file['data'] is not None:
                    df = processed_file['data']
                    st.caption(f"📊 {df.shape[0]}행 × {df.shape[1]}열")
                    
                    # 매우 간단한 미리보기
                    with st.expander(f"🔍 {file_info['name']} 데이터 미리보기", expanded=False):
                        st.dataframe(df.head(), use_container_width=True)
                
                # 이미지 파일 미리보기
                elif file_info['is_image']:
                    with st.expander(f"🖼️ {file_info['name']} 이미지 미리보기", expanded=False):
                        st.image(processed_file['file'], caption=file_info['name'], use_column_width=True)
                
                if i < len(processed_files) - 1:
                    st.markdown("---")


def create_file_upload_manager(session_key: str = "uploaded_files") -> FileUploadManager:
    """파일 업로드 관리자 인스턴스 생성"""
    return FileUploadManager(session_key=session_key) 