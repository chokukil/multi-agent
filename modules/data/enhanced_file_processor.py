"""
Enhanced File Processor - Multi-format Support with Visual Feedback

Comprehensive file processing with:
- Multi-format file handling (CSV, Excel, JSON, Parquet, PKL)
- Integration with Pandas Analyst Agent (8315)
- Automatic data profiling with quality indicators
- Visual relationship diagrams for multi-dataset scenarios
- One-click analysis suggestion buttons
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
import logging
from datetime import datetime
import uuid

from ..models import VisualDataCard, DataQualityInfo, DataRelationship, OneClickRecommendation
from ..a2a.agent_client import A2AAgentClient

logger = logging.getLogger(__name__)


class EnhancedFileProcessor:
    """Enhanced file processor with comprehensive multi-format support and LLM integration"""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.pkl']
    
    def __init__(self):
        """Initialize enhanced file processor"""
        self.pandas_agent = A2AAgentClient(port=8315)
        self.processed_datasets: Dict[str, VisualDataCard] = {}
        self.relationship_cache: Dict[str, List[DataRelationship]] = {}
    
    async def process_upload_with_ui(self, 
                                   uploaded_files: List[Any],
                                   progress_callback: Optional[callable] = None) -> List[VisualDataCard]:
        """
        Enhanced upload processing with visual feedback:
        1. Show upload progress indicators with file names and sizes
        2. Validate file formats with immediate visual feedback
        3. Send to Pandas Analyst with real-time status updates
        4. Generate visual data cards with interactive previews
        5. Perform automatic profiling with progress visualization
        6. Generate contextual analysis suggestions with one-click buttons
        """
        logger.info(f"Processing {len(uploaded_files)} uploaded files")
        
        processed_cards = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(f"Processing {uploaded_file.name}...", (i + 1) / total_files)
                
                # Process individual file
                data_card = await self._process_single_file_enhanced(uploaded_file)
                processed_cards.append(data_card)
                
                # Cache the dataset
                self.processed_datasets[data_card.id] = data_card
                
                logger.info(f"Successfully processed {uploaded_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Discover relationships between datasets
        if len(processed_cards) > 1:
            await self._discover_dataset_relationships(processed_cards)
        
        # Generate contextual analysis suggestions
        for data_card in processed_cards:
            data_card.metadata['suggestions'] = await self._generate_analysis_suggestions(data_card)
        
        return processed_cards
    
    async def _process_single_file_enhanced(self, uploaded_file) -> VisualDataCard:
        """Process a single file with enhanced features and Pandas Agent integration"""
        file_name = uploaded_file.name
        file_extension = Path(file_name).suffix.lower()
        
        # Validate file format
        if file_extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Load data with format-specific handling
        try:
            df = await self._load_data_with_format_detection(uploaded_file, file_extension)
        except Exception as e:
            raise Exception(f"Failed to load {file_name}: {str(e)}")
        
        # Send to Pandas Analyst for initial processing
        pandas_analysis = await self._analyze_with_pandas_agent(df, file_name)
        
        # Generate comprehensive data card
        data_card = await self._create_enhanced_data_card(
            uploaded_file, df, pandas_analysis
        )
        
        return data_card
    
    async def _load_data_with_format_detection(self, uploaded_file, file_extension: str) -> pd.DataFrame:
        """Load data with intelligent format detection and error handling"""
        try:
            if file_extension == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any supported encoding")
                    
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file, engine=None)
                
            elif file_extension == '.json':
                uploaded_file.seek(0)
                data = json.load(uploaded_file)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to normalize nested JSON
                    df = pd.json_normalize(data)
                    if df.empty and len(data) > 0:
                        # Fallback to simple dict conversion
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("JSON format not supported")
                    
            elif file_extension == '.parquet':
                df = pd.read_parquet(uploaded_file)
                
            elif file_extension == '.pkl':
                uploaded_file.seek(0)
                df = pd.read_pickle(uploaded_file)
                
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
            
            # Basic validation
            if df.empty:
                raise ValueError("File contains no data")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading file with extension {file_extension}: {str(e)}")
            raise
    
    async def _analyze_with_pandas_agent(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """Send data to Pandas Analyst Agent for initial analysis"""
        try:
            # Prepare request for Pandas Agent
            request_data = {
                "query": f"Analyze the uploaded dataset '{file_name}' and provide comprehensive insights",
                "data": df.to_dict('records'),
                "metadata": {
                    "file_name": file_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict()
                },
                "analysis_type": "comprehensive_profiling"
            }
            
            # Call Pandas Agent
            response = await self.pandas_agent.execute_task(request_data)
            
            return response.get('analysis', {})
            
        except Exception as e:
            logger.warning(f"Pandas Agent analysis failed: {str(e)}")
            # Return basic analysis if agent fails
            return self._generate_basic_analysis(df)
    
    def _generate_basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic analysis when Pandas Agent is unavailable"""
        try:
            analysis = {
                "summary_statistics": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "memory_usage": df.memory_usage(deep=True).to_dict()
            }
            
            # Add column analysis
            analysis["column_analysis"] = {}
            for col in df.columns:
                col_info = {
                    "type": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique())
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                        "std": float(df[col].std()) if not df[col].isnull().all() else None,
                        "min": float(df[col].min()) if not df[col].isnull().all() else None,
                        "max": float(df[col].max()) if not df[col].isnull().all() else None
                    })
                
                analysis["column_analysis"][col] = col_info
            
            return analysis
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {str(e)}")
            return {}
    
    async def _create_enhanced_data_card(self, 
                                       uploaded_file, 
                                       df: pd.DataFrame, 
                                       pandas_analysis: Dict[str, Any]) -> VisualDataCard:
        """Create enhanced visual data card with comprehensive information"""
        file_name = uploaded_file.name
        file_extension = Path(file_name).suffix.lower()
        
        # Calculate memory usage
        memory_usage = self._format_memory_usage(df.memory_usage(deep=True).sum())
        
        # Generate quality indicators
        quality_info = self._analyze_data_quality_enhanced(df, pandas_analysis)
        
        # Create preview (top 10 rows with smart column selection)
        preview_df = self._create_smart_preview(df)
        
        # Generate metadata
        metadata = {
            'upload_time': datetime.now().isoformat(),
            'file_size': getattr(uploaded_file, 'size', 0),
            'column_types': df.dtypes.to_dict(),
            'column_names': df.columns.tolist(),
            'pandas_analysis': pandas_analysis,
            'data_summary': {
                'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                'boolean_columns': len(df.select_dtypes(include=['bool']).columns)
            }
        }
        
        # Create data card
        data_card = VisualDataCard(
            id=str(uuid.uuid4()),
            name=file_name,
            file_path=file_name,
            format=file_extension.upper().replace('.', ''),
            rows=len(df),
            columns=len(df.columns),
            memory_usage=memory_usage,
            preview=preview_df,
            metadata=metadata,
            quality_indicators=quality_info,
            selection_state=True,
            upload_progress=100.0
        )
        
        return data_card
    
    def _format_memory_usage(self, memory_bytes: int) -> str:
        """Format memory usage in human-readable format"""
        if memory_bytes < 1024:
            return f"{memory_bytes} B"
        elif memory_bytes < 1024**2:
            return f"{memory_bytes/1024:.1f} KB"
        elif memory_bytes < 1024**3:
            return f"{memory_bytes/(1024**2):.1f} MB"
        else:
            return f"{memory_bytes/(1024**3):.1f} GB"
    
    def _analyze_data_quality_enhanced(self, 
                                     df: pd.DataFrame, 
                                     pandas_analysis: Dict[str, Any]) -> DataQualityInfo:
        """Enhanced data quality analysis with Pandas Agent insights"""
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        
        # Data types summary
        data_types_summary = df.dtypes.value_counts().to_dict()
        data_types_summary = {str(k): int(v) for k, v in data_types_summary.items()}
        
        # Enhanced quality score calculation
        quality_score = self._calculate_quality_score(df, pandas_analysis)
        
        # Comprehensive issue detection
        issues = self._detect_data_quality_issues(df, pandas_analysis)
        
        return DataQualityInfo(
            missing_values_count=int(missing_count),
            missing_percentage=missing_percentage,
            data_types_summary=data_types_summary,
            quality_score=quality_score,
            issues=issues
        )
    
    def _calculate_quality_score(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive quality score"""
        score = 100.0
        
        # Missing data penalty
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score -= min(missing_percentage * 1.5, 30)
        
        # Data consistency checks
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                unique_types = set(type(x).__name__ for x in df[col].dropna())
                if len(unique_types) > 1:
                    score -= 5
        
        # Size appropriateness
        if len(df) < 10:
            score -= 20  # Very small dataset
        elif len(df) > 1000000:
            score -= 5   # Very large dataset may have processing issues
        
        # Column count check
        if len(df.columns) > 100:
            score -= 10  # High dimensionality
        
        return max(0.0, min(100.0, score))
    
    def _detect_data_quality_issues(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Detect comprehensive data quality issues"""
        issues = []
        
        # Missing data issues
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            issues.append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        # Size issues
        if len(df) < 10:
            issues.append("Very small dataset: Results may not be reliable")
        elif len(df) < 100:
            issues.append("Small dataset: Consider collecting more data")
        
        # Dimensionality issues
        if len(df.columns) > 100:
            issues.append("High dimensionality: Consider feature selection")
        
        # Data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for potential numeric columns stored as strings
                try:
                    pd.to_numeric(df[col].dropna().iloc[:100])
                    issues.append(f"Column '{col}' may be numeric but stored as text")
                except:
                    pass
        
        # Duplicate detection
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                issues.append(f"Column '{col}' has potential outliers")
        
        return issues
    
    def _create_smart_preview(self, df: pd.DataFrame, max_cols: int = 10) -> pd.DataFrame:
        """Create smart preview with most important columns"""
        if len(df.columns) <= max_cols:
            return df.head(10)
        
        # Prioritize columns: non-null, diverse, informative
        column_priorities = {}
        
        for col in df.columns:
            priority = 0
            
            # Non-null data priority
            non_null_ratio = (len(df) - df[col].isnull().sum()) / len(df)
            priority += non_null_ratio * 10
            
            # Diversity priority
            unique_ratio = df[col].nunique() / len(df)
            priority += min(unique_ratio * 5, 5)
            
            # Type priority (numeric > categorical > text)
            if df[col].dtype in ['int64', 'float64']:
                priority += 3
            elif df[col].dtype == 'bool':
                priority += 2
            elif df[col].dtype == 'object':
                # Shorter strings preferred
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length < 50:
                    priority += 2
                else:
                    priority += 1
            
            column_priorities[col] = priority
        
        # Select top columns
        top_columns = sorted(column_priorities.items(), key=lambda x: x[1], reverse=True)
        selected_columns = [col for col, _ in top_columns[:max_cols]]
        
        return df[selected_columns].head(10)
    
    async def _discover_dataset_relationships(self, data_cards: List[VisualDataCard]):
        """Discover relationships between multiple datasets"""
        logger.info(f"Discovering relationships between {len(data_cards)} datasets")
        
        for i, card1 in enumerate(data_cards):
            relationships = []
            
            for j, card2 in enumerate(data_cards):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                relationship = await self._analyze_dataset_relationship(card1, card2)
                if relationship:
                    relationships.append(relationship)
            
            # Cache relationships
            self.relationship_cache[card1.id] = relationships
            card1.relationships = relationships
    
    async def _analyze_dataset_relationship(self, 
                                          card1: VisualDataCard, 
                                          card2: VisualDataCard) -> Optional[DataRelationship]:
        """Analyze relationship between two datasets"""
        try:
            cols1 = set(card1.metadata['column_names'])
            cols2 = set(card2.metadata['column_names'])
            
            # Find common columns
            common_columns = list(cols1.intersection(cols2))
            
            if not common_columns:
                return None
            
            # Calculate relationship confidence
            confidence = len(common_columns) / min(len(cols1), len(cols2))
            
            # Generate merge suggestions
            merge_suggestions = []
            for col in common_columns:
                merge_suggestions.append(f"Join on '{col}' column")
            
            # Determine relationship type
            relationship_type = "potential_join"
            if len(common_columns) > 3:
                relationship_type = "strong_relationship"
            elif confidence > 0.5:
                relationship_type = "moderate_relationship"
            
            return DataRelationship(
                target_dataset_id=card2.id,
                relationship_type=relationship_type,
                common_columns=common_columns,
                confidence_score=confidence,
                merge_suggestions=merge_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error analyzing relationship: {str(e)}")
            return None
    
    async def _generate_analysis_suggestions(self, 
                                           data_card: VisualDataCard) -> List[OneClickRecommendation]:
        """Generate contextual analysis suggestions"""
        suggestions = []
        df_info = data_card.metadata
        
        # Basic statistics suggestion
        suggestions.append(OneClickRecommendation(
            title="Basic Statistics",
            description="Generate comprehensive statistical summary of the dataset",
            action_type="statistical_analysis",
            parameters={"dataset_id": data_card.id},
            estimated_time=30,
            confidence_score=0.95,
            complexity_level="beginner",
            expected_result_preview="Summary statistics, distributions, and basic insights",
            icon="📊",
            color_theme="blue",
            execution_button_text="Generate Statistics"
        ))
        
        # Data quality check suggestion
        if data_card.quality_indicators.quality_score < 90:
            suggestions.append(OneClickRecommendation(
                title="Data Quality Assessment",
                description="Detailed analysis of data quality issues and cleaning recommendations",
                action_type="data_quality_check",
                parameters={"dataset_id": data_card.id},
                estimated_time=45,
                confidence_score=0.90,
                complexity_level="intermediate",
                expected_result_preview="Quality report with cleaning suggestions",
                icon="🔍",
                color_theme="orange",
                execution_button_text="Check Quality"
            ))
        
        # Visualization suggestion for numerical data
        numerical_cols = df_info['data_summary']['numerical_columns']
        if numerical_cols > 0:
            suggestions.append(OneClickRecommendation(
                title="Data Visualization",
                description="Create interactive charts and plots for numerical columns",
                action_type="create_visualizations",
                parameters={"dataset_id": data_card.id, "focus": "numerical"},
                estimated_time=60,
                confidence_score=0.85,
                complexity_level="beginner",
                expected_result_preview="Interactive charts showing data distributions",
                icon="📈",
                color_theme="green",
                execution_button_text="Create Charts"
            ))
        
        # Machine learning suggestion for suitable datasets
        if data_card.rows > 100 and numerical_cols > 2:
            suggestions.append(OneClickRecommendation(
                title="ML Model Training",
                description="Build and evaluate machine learning models",
                action_type="ml_analysis",
                parameters={"dataset_id": data_card.id},
                estimated_time=180,
                confidence_score=0.75,
                complexity_level="advanced",
                expected_result_preview="Model performance metrics and insights",
                icon="🤖",
                color_theme="purple",
                execution_button_text="Train Models"
            ))
        
        return suggestions[:3]  # Return maximum 3 suggestions
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[VisualDataCard]:
        """Get dataset by ID"""
        return self.processed_datasets.get(dataset_id)
    
    def get_all_datasets(self) -> List[VisualDataCard]:
        """Get all processed datasets"""
        return list(self.processed_datasets.values())
    
    def get_selected_datasets(self) -> List[VisualDataCard]:
        """Get selected datasets for analysis"""
        return [card for card in self.processed_datasets.values() if card.selection_state]