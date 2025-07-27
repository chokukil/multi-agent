"""
Virtual Scroll Table Renderer - High Performance with Smart Downloads

Enhanced table rendering with smart download system:
- Virtual scrolling for performance with large datasets
- Column sorting and filtering with search functionality
- Conditional formatting with color coding for data patterns
- Statistical summaries for numerical columns
- Raw artifact download: Table Data (CSV) - always available
- Context-aware enhanced formats: Excel for business users, formatted PDF for reports
- Pagination controls with customizable page sizes
- Row selection and bulk operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
import io
from datetime import datetime
import base64

from ..interfaces import BaseRenderer
from ..models import EnhancedArtifact

logger = logging.getLogger(__name__)


class VirtualScrollTableRenderer(BaseRenderer):
    """Virtual scrolling table renderer for large datasets"""
    
    def __init__(self):
        """Initialize enhanced table renderer"""
        super().__init__(supported_types=['table', 'dataframe', 'data'])
        self.page_size = 50  # Default rows per page
        self.max_display_columns = 20  # Maximum columns to display
        self.max_rows_for_styling = 1000  # Maximum rows for advanced styling
        
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
   
    def render_artifact(self, artifact: EnhancedArtifact) -> None:
        """
        Render enhanced table with comprehensive features:
        - Virtual scrolling for large datasets
        - Advanced filtering and search
        - Statistical summaries and insights
        - Smart download system with multiple formats
        """
        try:
            # Extract table data
            df = self._extract_table_data(artifact)
            
            # Render table with enhanced features
            self._render_enhanced_table(df, artifact)
            
            # Render statistical insights
            self._render_statistical_insights(df)
            
            # Render smart download options
            self._render_smart_download_options(df, artifact)
            
        except Exception as e:
            logger.error(f"Error rendering table artifact: {str(e)}")
            st.error(f"❌ Table rendering failed: {str(e)}")
            
            # Fallback to basic display
            with st.expander("📄 Raw Table Data", expanded=False):
                if hasattr(artifact.data, 'head'):
                    st.dataframe(artifact.data.head(100))
                else:
                    st.write(artifact.data)
    
    def _extract_table_data(self, artifact: EnhancedArtifact) -> pd.DataFrame:
        """Extract and validate table data from artifact"""
        try:
            if isinstance(artifact.data, pd.DataFrame):
                return artifact.data
            elif isinstance(artifact.data, dict):
                return pd.DataFrame(artifact.data)
            elif isinstance(artifact.data, list):
                return pd.DataFrame(artifact.data)
            else:
                raise ValueError("Invalid table data format")
                
        except Exception as e:
            logger.error(f"Error extracting table data: {str(e)}")
            raise ValueError(f"Failed to parse table data: {str(e)}")
    
    def _render_enhanced_table(self, df: pd.DataFrame, artifact: EnhancedArtifact):
        """Render table with enhanced features and controls"""
        
        # Table header with metadata
        st.markdown(f"### 📊 {artifact.title}")
        if artifact.description:
            st.markdown(f"*{artifact.description}*")
        
        # Table statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", f"{len(df.columns):,}")
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Enhanced controls
        with st.expander("🔧 Table Controls", expanded=False):
            self._render_table_controls(df, artifact)
        
        # Apply filters and transformations
        filtered_df = self._apply_table_filters(df, artifact)
        
        # Render paginated table
        self._render_paginated_table(filtered_df, artifact)
    
    def _render_table_controls(self, df: pd.DataFrame, artifact: EnhancedArtifact):
        """Render advanced table controls"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Search & Filter**")
            
            # Global search
            search_query = st.text_input(
                "Search all columns",
                placeholder="Enter search term...",
                key=f"search_{artifact.id}"
            )
            
            # Column selection
            if len(df.columns) > 10:
                selected_columns = st.multiselect(
                    "Select columns to display",
                    options=df.columns.tolist(),
                    default=df.columns[:10].tolist(),
                    key=f"columns_{artifact.id}"
                )
            else:
                selected_columns = df.columns.tolist()
        
        with col2:
            st.markdown("**📊 Display Options**")
            
            # Page size
            page_size = st.selectbox(
                "Rows per page",
                options=[25, 50, 100, 200],
                index=1,
                key=f"page_size_{artifact.id}"
            )
            
            # Sorting
            sort_column = st.selectbox(
                "Sort by column",
                options=['None'] + df.columns.tolist(),
                key=f"sort_col_{artifact.id}"
            )
            
            if sort_column != 'None':
                sort_ascending = st.checkbox(
                    "Ascending order",
                    value=True,
                    key=f"sort_asc_{artifact.id}"
                )
        
        # Store filter settings in session state
        filter_key = f"table_filters_{artifact.id}"
        st.session_state[filter_key] = {
            'search_query': search_query,
            'selected_columns': selected_columns,
            'page_size': page_size,
            'sort_column': sort_column,
            'sort_ascending': sort_ascending if sort_column != 'None' else True
        }
    
    def _apply_table_filters(self, df: pd.DataFrame, artifact: EnhancedArtifact) -> pd.DataFrame:
        """Apply filters and transformations to dataframe"""
        
        filter_key = f"table_filters_{artifact.id}"
        filters = st.session_state.get(filter_key, {})
        
        filtered_df = df.copy()
        
        # Apply search filter
        search_query = filters.get('search_query', '')
        if search_query:
            filtered_df = self._search_dataframe(filtered_df, search_query)
        
        # Apply column selection
        selected_columns = filters.get('selected_columns', df.columns.tolist())
        if selected_columns:
            filtered_df = filtered_df[selected_columns]
        
        # Apply sorting
        sort_column = filters.get('sort_column', 'None')
        if sort_column != 'None' and sort_column in filtered_df.columns:
            sort_ascending = filters.get('sort_ascending', True)
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)
        
        return filtered_df
    
    def _render_paginated_table(self, df: pd.DataFrame, artifact: EnhancedArtifact):
        """Render table with pagination and advanced styling"""
        
        filter_key = f"table_filters_{artifact.id}"
        filters = st.session_state.get(filter_key, {})
        page_size = filters.get('page_size', 50)
        
        if len(df) == 0:
            st.warning("No data matches the current filters.")
            return
        
        # Calculate pagination
        total_pages = max(1, (len(df) - 1) // page_size + 1)
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_page = st.slider(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"page_{artifact.id}"
            )
        
        # Calculate slice
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(df))
        page_df = df.iloc[start_idx:end_idx]
        
        # Apply conditional formatting
        if len(page_df) <= self.max_rows_for_styling:
            styled_df = self._apply_conditional_formatting(page_df)
            st.dataframe(styled_df, use_container_width=True, height=600)
        else:
            st.dataframe(page_df, use_container_width=True, height=600)
        
        # Page information
        st.caption(f"📄 Page {current_page}/{total_pages} | Showing rows {start_idx + 1}-{end_idx} of {len(df):,}")
        
        # Show filter summary if applied
        if len(df) != len(artifact.data if hasattr(artifact, 'data') else df):
            original_count = len(artifact.data) if hasattr(artifact, 'data') else len(df)
            st.info(f"🔍 Filtered: {len(df):,} rows from {original_count:,} total")
    
    def _apply_conditional_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply conditional formatting to dataframe"""
        try:
            styler = df.style
            
            # Numeric column formatting
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].nunique() > 1 and not df[col].isnull().all():
                    # Color gradient for numeric values
                    styler = styler.background_gradient(
                        subset=[col],
                        cmap='RdYlBu_r',
                        low=0.3,
                        high=0.7
                    )
            
            # Highlight missing values
            styler = styler.highlight_null(color='#ffcccc')
            
            # Format numbers
            format_dict = {}
            for col in numeric_cols:
                if df[col].dtype in ['float64', 'float32']:
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
            
            # Set table properties
            styler = styler.set_properties(**{
                'text-align': 'left',
                'font-size': '11px',
                'padding': '4px'
            })
            
            return styler
            
        except Exception as e:
            logger.error(f"Error applying conditional formatting: {str(e)}")
            return df
    
    def _render_statistical_insights(self, df: pd.DataFrame):
        """Render statistical insights and summaries"""
        
        with st.expander("📈 Statistical Insights", expanded=False):
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("#### Numeric Columns Summary")
                
                stats_df = df[numeric_cols].describe().round(2)
                st.dataframe(stats_df, use_container_width=True)
                
                # Correlation matrix for numeric columns
                if len(numeric_cols) > 1:
                    st.markdown("#### Correlation Matrix")
                    corr_matrix = df[numeric_cols].corr().round(2)
                    
                    # Style correlation matrix
                    styled_corr = corr_matrix.style.background_gradient(
                        cmap='RdBu_r', center=0, vmin=-1, vmax=1
                    ).format('{:.2f}')
                    
                    st.dataframe(styled_corr, use_container_width=True)
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                st.markdown("#### Categorical Columns Summary")
                
                for col in categorical_cols[:5]:  # Show first 5 categorical columns
                    value_counts = df[col].value_counts().head(10)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**{col}**")
                        st.write(f"Unique values: {df[col].nunique()}")
                        st.write(f"Most frequent: {value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
                    
                    with col2:
                        if len(value_counts) > 0:
                            st.bar_chart(value_counts)
            
            # Data quality summary
            st.markdown("#### Data Quality Summary")
            
            quality_data = []
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                quality_data.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Missing Count': missing_count,
                    'Missing %': f"{missing_pct:.1f}%",
                    'Unique Values': unique_count,
                    'Completeness': f"{100 - missing_pct:.1f}%"
                })
            
            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True)
    
    def _render_smart_download_options(self, df: pd.DataFrame, artifact: EnhancedArtifact):
        """Render smart download system with two-tier approach"""
        
        st.markdown("### 💾 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔒 Raw Artifacts (Always Available)")
            
            # Raw CSV data (always available)
            csv_data = self._export_to_csv(df)
            st.download_button(
                label="📊 Table Data (CSV)",
                data=csv_data,
                file_name=f"{artifact.title.replace(' ', '_')}_data.csv",
                mime="text/csv",
                help="Raw table data in CSV format"
            )
            
            # JSON format
            json_data = self._export_to_json(df)
            st.download_button(
                label="📋 Table Data (JSON)",
                data=json_data,
                file_name=f"{artifact.title.replace(' ', '_')}_data.json",
                mime="application/json",
                help="Raw table data in JSON format"
            )
        
        with col2:
            st.markdown("#### 🎨 Enhanced Formats (Context-Based)")
            
            # Excel export for business users
            excel_data = self._export_to_excel(df, artifact)
            if excel_data:
                st.download_button(
                    label="📈 Formatted Excel",
                    data=excel_data,
                    file_name=f"{artifact.title.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Excel file with formatting and charts"
                )
            
            # Statistical summary report
            summary_data = self._export_statistical_summary(df, artifact)
            if summary_data:
                st.download_button(
                    label="📊 Statistical Report",
                    data=summary_data,
                    file_name=f"{artifact.title.replace(' ', '_')}_report.txt",
                    mime="text/plain",
                    help="Comprehensive statistical analysis report"
                )
    
    def _export_to_csv(self, df: pd.DataFrame) -> bytes:
        """Export dataframe to CSV format"""
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            return csv_buffer.getvalue().encode('utf-8-sig')
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            return b""
    
    def _export_to_json(self, df: pd.DataFrame) -> str:
        """Export dataframe to JSON format"""
        try:
            return df.to_json(orient='records', indent=2, date_format='iso')
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            return "{}"
    
    def _export_to_excel(self, df: pd.DataFrame, artifact: EnhancedArtifact) -> Optional[bytes]:
        """Export dataframe to formatted Excel file"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Write summary statistics if numeric columns exist
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = df[numeric_cols].describe()
                    stats_df.to_excel(writer, sheet_name='Statistics')
                
                # Write data quality summary
                quality_data = []
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    quality_data.append({
                        'Column': col,
                        'Type': str(df[col].dtype),
                        'Missing Count': missing_count,
                        'Missing %': f"{(missing_count / len(df)) * 100:.1f}%",
                        'Unique Values': df[col].nunique()
                    })
                
                quality_df = pd.DataFrame(quality_data)
                quality_df.to_excel(writer, sheet_name='Data Quality', index=False)
            
            output.seek(0)
            return output.read()
            
        except Exception as e:
            logger.error(f"Excel export failed: {str(e)}")
            st.error("Excel export failed. Please try CSV format.")
            return None
    
    def _export_statistical_summary(self, df: pd.DataFrame, artifact: EnhancedArtifact) -> Optional[str]:
        """Export comprehensive statistical summary"""
        try:
            report_lines = []
            report_lines.append(f"Statistical Analysis Report: {artifact.title}")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Dataset overview
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 20)
            report_lines.append(f"Rows: {len(df):,}")
            report_lines.append(f"Columns: {len(df.columns):,}")
            report_lines.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            report_lines.append("")
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                report_lines.append("MISSING DATA ANALYSIS")
                report_lines.append("-" * 25)
                for col, missing_count in missing_data[missing_data > 0].items():
                    missing_pct = (missing_count / len(df)) * 100
                    report_lines.append(f"{col}: {missing_count:,} ({missing_pct:.1f}%)")
                report_lines.append("")
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                report_lines.append("NUMERIC COLUMNS ANALYSIS")
                report_lines.append("-" * 30)
                stats_df = df[numeric_cols].describe()
                report_lines.append(stats_df.to_string())
                report_lines.append("")
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                report_lines.append("CATEGORICAL COLUMNS ANALYSIS")
                report_lines.append("-" * 35)
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                    report_lines.append(f"{col}: {unique_count} unique values, most frequent: {most_frequent}")
                report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Statistical summary export failed: {str(e)}")
            return None
    
    def get_download_options(self, artifact: EnhancedArtifact) -> List[Dict[str, Any]]:
        """Get available download options for the artifact"""
        
        options = [
            {
                'name': 'Table Data (CSV)',
                'description': 'Raw table data in CSV format',
                'format': 'csv',
                'type': 'raw',
                'icon': '📊',
                'always_available': True
            },
            {
                'name': 'Table Data (JSON)',
                'description': 'Raw table data in JSON format',
                'format': 'json',
                'type': 'raw',
                'icon': '📋',
                'always_available': True
            },
            {
                'name': 'Formatted Excel',
                'description': 'Excel file with formatting and multiple sheets',
                'format': 'xlsx',
                'type': 'enhanced',
                'icon': '📈',
                'context': 'business'
            },
            {
                'name': 'Statistical Report',
                'description': 'Comprehensive statistical analysis report',
                'format': 'txt',
                'type': 'enhanced',
                'icon': '📊',
                'context': 'analysis'
            }
        ]
        
        return options
    
    def supports_artifact_type(self, artifact_type: str) -> bool:
        """Check if renderer supports the given artifact type"""
        supported_types = ['table', 'dataframe', 'data', 'csv', 'excel']
        return artifact_type.lower() in supported_types