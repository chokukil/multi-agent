"""
Enhanced File Upload - Drag-and-Drop with Visual Feedback

Enhanced file upload with:
- Clear drag-and-drop visual boundaries
- Upload progress indicators  
- Multi-file selection support
- File format validation with visual feedback
- Immediate processing status display
"""

import streamlit as st
import pandas as pd
import json
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import uuid
from datetime import datetime

from ..models import VisualDataCard, DataQualityInfo


class EnhancedFileUpload:
    """Enhanced file upload component with comprehensive visual feedback"""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.pkl']
    
    def __init__(self):
        """Initialize enhanced file upload"""
        self.uploaded_files: List[Any] = []
        self.processing_status: Dict[str, str] = {}
        self.data_cards: List[VisualDataCard] = []
    
    def render_upload_area(self, 
                          on_files_uploaded: Optional[Callable] = None,
                          show_existing_files: bool = True) -> List[VisualDataCard]:
        """
        Render enhanced file upload area with comprehensive features
        
        Returns:
            List of VisualDataCard objects for uploaded files
        """
        upload_container = st.container()
        
        with upload_container:
            # Custom upload area with enhanced styling
            self._render_custom_upload_zone()
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload your data files",
                accept_multiple_files=True,
                type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'pkl'],
                help="Supports CSV, Excel, JSON, Parquet, and Pickle formats",
                label_visibility="collapsed",
                key="enhanced_file_uploader"
            )
            
            # Process uploaded files
            if uploaded_files:
                self.uploaded_files = uploaded_files
                self.data_cards = self._process_uploaded_files(uploaded_files)
                
                if on_files_uploaded:
                    on_files_uploaded(self.data_cards)
            
            # Show existing files if requested
            if show_existing_files and self.data_cards:
                self._render_data_cards()
        
        return self.data_cards
    
    def _render_custom_upload_zone(self):
        """Render custom upload zone with enhanced visual feedback"""
        st.markdown("""
        <div class="enhanced-upload-zone" id="upload-zone">
            <div class="upload-icon">📁</div>
            <div class="upload-title">Drag and drop your data files here</div>
            <div class="upload-subtitle">
                Supports CSV, Excel (.xlsx, .xls), JSON, Parquet, and PKL formats
            </div>
            <div class="upload-hint">or click to browse files</div>
            <div class="upload-features">
                <span class="feature-badge">📊 Auto Analysis</span>
                <span class="feature-badge">🔍 Quality Check</span>
                <span class="feature-badge">🤝 Relationship Discovery</span>
            </div>
        </div>
        
        <style>
        .enhanced-upload-zone {
            border: 3px dashed #dee2e6;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .enhanced-upload-zone:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 123, 255, 0.15);
        }
        
        .enhanced-upload-zone.drag-over {
            border-color: #28a745;
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            transform: scale(1.02);
            animation: pulse 0.5s ease-in-out;
        }
        
        .upload-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.8;
        }
        
        .upload-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }
        
        .upload-subtitle {
            color: #6c757d;
            margin-bottom: 1rem;
            font-size: 1rem;
        }
        
        .upload-hint {
            font-size: 0.9rem;
            color: #868e96;
            margin-bottom: 1.5rem;
        }
        
        .upload-features {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .feature-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        @keyframes pulse {
            0% { transform: scale(1.02); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1.02); }
        }
        
        .upload-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 4px;
            background: linear-gradient(90deg, #28a745, #20c997);
            border-radius: 0 0 16px 16px;
            transition: width 0.3s ease;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _process_uploaded_files(self, uploaded_files: List[Any]) -> List[VisualDataCard]:
        """Process uploaded files and create visual data cards"""
        data_cards = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Process individual file
                data_card = self._process_single_file(uploaded_file)
                data_cards.append(data_card)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        if data_cards:
            st.success(f"✅ Successfully processed {len(data_cards)} file(s)")
        
        return data_cards
    
    def _process_single_file(self, uploaded_file) -> VisualDataCard:
        """Process a single uploaded file and create a visual data card"""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Load data based on file type
        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.json':
                df = pd.read_json(uploaded_file)
            elif file_extension == '.parquet':
                df = pd.read_parquet(uploaded_file)
            elif file_extension == '.pkl':
                df = pd.read_pickle(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
        except Exception as e:
            raise Exception(f"Failed to read file: {str(e)}")
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(df)
        
        # Generate quality indicators
        quality_info = self._analyze_data_quality(df)
        
        # Create visual data card
        data_card = VisualDataCard(
            id=str(uuid.uuid4()),
            name=uploaded_file.name,
            file_path=uploaded_file.name,
            format=file_extension.upper().replace('.', ''),
            rows=len(df),
            columns=len(df.columns),
            memory_usage=memory_usage,
            preview=df.head(10),
            metadata={
                'upload_time': datetime.now().isoformat(),
                'file_size': uploaded_file.size if hasattr(uploaded_file, 'size') else 0,
                'column_types': df.dtypes.to_dict(),
                'column_names': df.columns.tolist()
            },
            quality_indicators=quality_info,
            selection_state=True,
            upload_progress=100.0
        )
        
        return data_card
    
    def _calculate_memory_usage(self, df: pd.DataFrame) -> str:
        """Calculate and format memory usage"""
        memory_bytes = df.memory_usage(deep=True).sum()
        
        if memory_bytes < 1024:
            return f"{memory_bytes} B"
        elif memory_bytes < 1024**2:
            return f"{memory_bytes/1024:.1f} KB"
        elif memory_bytes < 1024**3:
            return f"{memory_bytes/(1024**2):.1f} MB"
        else:
            return f"{memory_bytes/(1024**3):.1f} GB"
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> DataQualityInfo:
        """Analyze data quality and generate indicators"""
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        
        # Data types summary
        data_types_summary = df.dtypes.value_counts().to_dict()
        data_types_summary = {str(k): int(v) for k, v in data_types_summary.items()}
        
        # Quality score calculation
        quality_score = max(0, 100 - missing_percentage * 2)
        
        # Identify issues
        issues = []
        if missing_percentage > 10:
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        if len(df.columns) > 100:
            issues.append("High dimensionality: Consider feature selection")
        if len(df) < 10:
            issues.append("Small dataset: Results may not be reliable")
        
        return DataQualityInfo(
            missing_values_count=int(missing_count),
            missing_percentage=missing_percentage,
            data_types_summary=data_types_summary,
            quality_score=quality_score,
            issues=issues
        )
    
    def _render_data_cards(self):
        """Render visual data cards for uploaded files"""
        if not self.data_cards:
            return
        
        st.markdown("### 📊 Uploaded Datasets")
        
        # Cards grid
        cols = st.columns(min(len(self.data_cards), 3))
        
        for i, data_card in enumerate(self.data_cards):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                self._render_single_data_card(data_card)
    
    def _render_single_data_card(self, data_card: VisualDataCard):
        """Render a single visual data card"""
        # Card container
        with st.container():
            # Card header
            card_html = f"""
            <div class="data-card">
                <div class="card-header">
                    <div class="card-title">
                        <span class="file-icon">{self._get_file_icon(data_card.format)}</span>
                        <span class="file-name">{data_card.name}</span>
                    </div>
                    <div class="format-badge">{data_card.format}</div>
                </div>
                
                <div class="card-stats">
                    <div class="stat-item">
                        <div class="stat-value">{data_card.rows:,}</div>
                        <div class="stat-label">Rows</div>
                    </div>
                    <div class="stat-divider">×</div>
                    <div class="stat-item">
                        <div class="stat-value">{data_card.columns}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                </div>
                
                <div class="card-info">
                    <div class="info-item">
                        <span class="info-label">Memory:</span>
                        <span class="info-value">{data_card.memory_usage}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Quality:</span>
                        <span class="info-value quality-score">{data_card.quality_indicators.quality_score:.0f}%</span>
                    </div>
                </div>
                
                <div class="card-actions">
                    <button class="preview-btn" onclick="showPreview('{data_card.id}')">
                        📖 Preview
                    </button>
                </div>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Selection checkbox
            selected = st.checkbox(
                f"Include in analysis",
                value=data_card.selection_state,
                key=f"select_{data_card.id}"
            )
            data_card.selection_state = selected
            
            # Preview modal (expandable)
            with st.expander(f"📖 Preview: {data_card.name}", expanded=False):
                st.dataframe(data_card.preview, use_container_width=True)
                
                # Quality indicators
                if data_card.quality_indicators.issues:
                    st.warning("⚠️ Data Quality Issues:")
                    for issue in data_card.quality_indicators.issues:
                        st.write(f"• {issue}")
        
        # Inject card CSS
        self._inject_card_css()
    
    def _inject_card_css(self):
        """Inject CSS for data cards"""
        st.markdown("""
        <style>
        .data-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .data-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            border-color: #007bff;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .card-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .file-icon {
            font-size: 1.5rem;
        }
        
        .file-name {
            font-weight: 600;
            color: #495057;
            font-size: 1rem;
        }
        
        .format-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .card-stats {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #495057;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #6c757d;
            text-transform: uppercase;
        }
        
        .stat-divider {
            margin: 0 1rem;
            font-size: 1.2rem;
            color: #6c757d;
        }
        
        .card-info {
            margin: 1rem 0;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
        }
        
        .info-label {
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .info-value {
            font-weight: 500;
            color: #495057;
        }
        
        .quality-score {
            color: #28a745;
            font-weight: 600;
        }
        
        .card-actions {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        
        .preview-btn {
            flex: 1;
            background: linear-gradient(135deg, #17a2b8, #138496);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .preview-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(23, 162, 184, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _get_file_icon(self, format_name: str) -> str:
        """Get appropriate icon for file format"""
        icons = {
            'CSV': '📊',
            'XLSX': '📈',
            'XLS': '📈', 
            'JSON': '🗂️',
            'PARQUET': '🗃️',
            'PKL': '🗄️'
        }
        return icons.get(format_name.upper(), '📄')
    
    def get_selected_datasets(self) -> List[VisualDataCard]:
        """Get list of selected datasets"""
        return [card for card in self.data_cards if card.selection_state]
    
    def get_all_datasets(self) -> List[VisualDataCard]:
        """Get all uploaded datasets"""
        return self.data_cards.copy()
    
    def clear_uploads(self):
        """Clear all uploaded files and data cards"""
        self.uploaded_files.clear()
        self.data_cards.clear()
        self.processing_status.clear()
        
        # Clear Streamlit file uploader
        if "enhanced_file_uploader" in st.session_state:
            del st.session_state["enhanced_file_uploader"]