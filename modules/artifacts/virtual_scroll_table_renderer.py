"""
Virtual Scroll Table Renderer - Advanced Features with CSV Download

Large table rendering with:
- Virtual scrolling for performance
- Column sorting and filtering
- Search functionality
- Raw CSV data download always available
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
import io

logger = logging.getLogger(__name__)


class VirtualScrollTableRenderer:
    """Virtual scrolling table renderer for large datasets"""
    
    def __init__(self):
        """Initialize table renderer"""
        self.page_size = 50  # Default rows per page
        self.max_display_columns = 20  # Maximum columns to display
        
    def render_table(self,
                    df: pd.DataFrame,
                    title: Optional[str] = None,
                    enable_search: bool = True,
                    enable_filter: bool = True,
                    enable_download: bool = True,
                    page_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Render table with virtual scrolling and advanced features
        
        Returns:
            Dict with 'raw_csv' for download
        """
        try:
            if title:
                st.markdown(f"### 📊 {title}")
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 행 수", f"{len(df):,}")
            with col2:
                st.metric("총 열 수", f"{len(df.columns):,}")
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
            
            # Search functionality
            if enable_search:
                search_query = st.text_input(
                    "🔍 테이블 검색",
                    placeholder="검색어를 입력하세요...",
                    key=f"search_{id(df)}"
                )
                
                if search_query:
                    df = self._search_dataframe(df, search_query)
                    st.info(f"검색 결과: {len(df):,}개 행")
            
            # Column filter
            if enable_filter and len(df.columns) > 5:
                selected_columns = st.multiselect(
                    "📋 표시할 열 선택",
                    options=df.columns.tolist(),
                    default=df.columns[:min(10, len(df.columns))].tolist(),
                    key=f"columns_{id(df)}"
                )
                
                if selected_columns:
                    display_df = df[selected_columns]
                else:
                    display_df = df
            else:
                display_df = df
            
            # Sorting
            if len(display_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    sort_column = st.selectbox(
                        "정렬 기준 열",
                        options=['없음'] + display_df.columns.tolist(),
                        key=f"sort_col_{id(df)}"
                    )
                with col2:
                    sort_order = st.radio(
                        "정렬 순서",
                        options=['오름차순', '내림차순'],
                        horizontal=True,
                        key=f"sort_order_{id(df)}"
                    )
                
                if sort_column != '없음':
                    display_df = display_df.sort_values(
                        by=sort_column,
                        ascending=(sort_order == '오름차순')
                    )
            
            # Virtual scrolling with pagination
            page_size = page_size or self.page_size
            total_pages = max(1, (len(display_df) - 1) // page_size + 1)
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col2:
                page = st.slider(
                    "페이지",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key=f"page_{id(df)}"
                )
            
            # Calculate slice indices
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(display_df))
            
            # Display current page
            page_df = display_df.iloc[start_idx:end_idx]
            
            # Advanced table styling
            styled_df = self._style_dataframe(page_df)
            
            # Render table
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(600, (len(page_df) + 1) * 35 + 20)
            )
            
            # Page info
            st.caption(f"📄 페이지 {page}/{total_pages} | 행 {start_idx + 1}-{end_idx} / {len(display_df):,}")
            
            # Download button
            if enable_download:
                csv_data = self._prepare_csv_download(df)  # Original full dataframe
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ 전체 데이터 CSV 다운로드",
                        data=csv_data,
                        file_name=f"{title or 'table_data'}.csv",
                        mime="text/csv",
                        key=f"download_{id(df)}"
                    )
                with col2:
                    if len(display_df) != len(df):
                        filtered_csv = self._prepare_csv_download(display_df)
                        st.download_button(
                            label="⬇️ 필터된 데이터 CSV 다운로드",
                            data=filtered_csv,
                            file_name=f"{title or 'table_data'}_filtered.csv",
                            mime="text/csv",
                            key=f"download_filtered_{id(df)}"
                        )
            
            return {
                'raw_csv': csv_data if enable_download else None,
                'displayed_rows': len(display_df),
                'total_rows': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error rendering table: {str(e)}")
            st.error(f"테이블 렌더링 오류: {str(e)}")
            return {'raw_csv': None}
    
    def _search_dataframe(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search DataFrame for query string"""
        try:
            # Convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            
            # Create mask for rows containing query
            mask = pd.Series([False] * len(df))
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # String columns
                    mask |= df[col].astype(str).str.lower().str.contains(query_lower, na=False)
                else:
                    # Numeric columns - exact match
                    try:
                        numeric_query = float(query)
                        mask |= (df[col] == numeric_query)
                    except ValueError:
                        pass
            
            return df[mask]
            
        except Exception as e:
            logger.error(f"Error searching dataframe: {str(e)}")
            return df
    
    def _style_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to dataframe"""
        try:
            # Create styler
            styler = df.style
            
            # Highlight numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Color scale for numeric columns
                for col in numeric_cols:
                    if df[col].nunique() > 1:
                        styler = styler.background_gradient(
                            subset=[col],
                            cmap='RdYlBu_r',
                            low=0.2,
                            high=0.8
                        )
            
            # Format numbers
            format_dict = {}
            for col in df.columns:
                if df[col].dtype in ['float64', 'float32']:
                    # Check if values are large or small
                    max_val = df[col].abs().max()
                    if pd.notna(max_val):
                        if max_val > 1000:
                            format_dict[col] = '{:,.0f}'
                        elif max_val < 0.01:
                            format_dict[col] = '{:.4f}'
                        else:
                            format_dict[col] = '{:.2f}'
            
            if format_dict:
                styler = styler.format(format_dict)
            
            # Highlight null values
            styler = styler.highlight_null(color='#ffcccc')
            
            # Set properties
            styler = styler.set_properties(**{
                'text-align': 'left',
                'font-size': '12px'
            })
            
            return styler
            
        except Exception as e:
            logger.error(f"Error styling dataframe: {str(e)}")
            return df
    
    def _prepare_csv_download(self, df: pd.DataFrame) -> bytes:
        """Prepare CSV data for download"""
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            return csv_buffer.getvalue().encode('utf-8-sig')
        except Exception as e:
            logger.error(f"Error preparing CSV: {str(e)}")
            return b""
    
    def render_pivot_table(self,
                          df: pd.DataFrame,
                          title: Optional[str] = None) -> Dict[str, Any]:
        """Render interactive pivot table"""
        try:
            st.markdown(f"### 📊 {title or 'Pivot Table'}")
            
            # Select columns for pivot
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not numeric_cols or not categorical_cols:
                st.warning("피벗 테이블을 생성하기 위한 적절한 열이 없습니다.")
                return {'raw_csv': None}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                index_col = st.selectbox(
                    "행 인덱스",
                    options=categorical_cols,
                    key="pivot_index"
                )
            
            with col2:
                values_col = st.selectbox(
                    "값",
                    options=numeric_cols,
                    key="pivot_values"
                )
            
            with col3:
                agg_func = st.selectbox(
                    "집계 함수",
                    options=['mean', 'sum', 'count', 'min', 'max'],
                    key="pivot_agg"
                )
            
            # Optional column selection
            columns_col = st.selectbox(
                "열 (선택사항)",
                options=['없음'] + categorical_cols,
                key="pivot_columns"
            )
            
            # Create pivot table
            if columns_col == '없음':
                pivot_df = df.pivot_table(
                    index=index_col,
                    values=values_col,
                    aggfunc=agg_func
                ).round(2)
            else:
                pivot_df = df.pivot_table(
                    index=index_col,
                    columns=columns_col,
                    values=values_col,
                    aggfunc=agg_func
                ).round(2)
            
            # Render pivot table
            return self.render_table(
                pivot_df.reset_index(),
                title=None,
                enable_search=False,
                enable_filter=False
            )
            
        except Exception as e:
            logger.error(f"Error rendering pivot table: {str(e)}")
            st.error(f"피벗 테이블 생성 오류: {str(e)}")
            return {'raw_csv': None}
    
    def render_summary_statistics(self, df: pd.DataFrame) -> None:
        """Render summary statistics for numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                st.info("숫자형 열이 없습니다.")
                return
            
            st.markdown("### 📈 요약 통계")
            
            # Calculate statistics
            stats_df = pd.DataFrame({
                '평균': numeric_df.mean(),
                '표준편차': numeric_df.std(),
                '최소값': numeric_df.min(),
                '25%': numeric_df.quantile(0.25),
                '중앙값': numeric_df.median(),
                '75%': numeric_df.quantile(0.75),
                '최대값': numeric_df.max(),
                '결측값': numeric_df.isnull().sum()
            }).round(2)
            
            # Transpose for better display
            stats_df = stats_df.T
            
            # Style the statistics table
            styled_stats = stats_df.style.background_gradient(cmap='RdYlBu_r', axis=1)
            
            st.dataframe(styled_stats, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering summary statistics: {str(e)}")
            st.error(f"요약 통계 생성 오류: {str(e)}")