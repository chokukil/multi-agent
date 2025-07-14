"""
DataFrame information extraction utilities

This module provides comprehensive analysis and information extraction
capabilities for pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

from .logger import get_logger


class DataFrameInfo:
    """
    Comprehensive DataFrame information extractor and analyzer
    
    Provides detailed insights about DataFrame structure, content,
    data quality, and statistical properties.
    """
    
    def __init__(self, df: pd.DataFrame, name: str = "DataFrame"):
        """
        Initialize DataFrame analyzer
        
        Args:
            df: DataFrame to analyze
            name: Name for the DataFrame
        """
        self.df = df
        self.name = name
        self.logger = get_logger()
        
        # Cached analysis results
        self._basic_info = None
        self._data_quality = None
        self._statistical_summary = None
        self._column_analysis = None
        self._relationships = None
        
    def get_basic_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get basic DataFrame information
        
        Args:
            force_refresh: Force recalculation
            
        Returns:
            Dictionary with basic information
        """
        if self._basic_info is None or force_refresh:
            self._basic_info = self._calculate_basic_info()
        
        return self._basic_info
    
    def get_data_quality_report(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive data quality report
        
        Args:
            force_refresh: Force recalculation
            
        Returns:
            Dictionary with data quality metrics
        """
        if self._data_quality is None or force_refresh:
            self._data_quality = self._calculate_data_quality()
        
        return self._data_quality
    
    def get_statistical_summary(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get statistical summary for numeric columns
        
        Args:
            force_refresh: Force recalculation
            
        Returns:
            Dictionary with statistical summaries
        """
        if self._statistical_summary is None or force_refresh:
            self._statistical_summary = self._calculate_statistical_summary()
        
        return self._statistical_summary
    
    def get_column_analysis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get detailed analysis for each column
        
        Args:
            force_refresh: Force recalculation
            
        Returns:
            Dictionary with per-column analysis
        """
        if self._column_analysis is None or force_refresh:
            self._column_analysis = self._calculate_column_analysis()
        
        return self._column_analysis
    
    def get_relationships(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get analysis of relationships between columns
        
        Args:
            force_refresh: Force recalculation
            
        Returns:
            Dictionary with relationship analysis
        """
        if self._relationships is None or force_refresh:
            self._relationships = self._calculate_relationships()
        
        return self._relationships
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis report
        
        Returns:
            Complete analysis report
        """
        return {
            "name": self.name,
            "generated_at": datetime.now().isoformat(),
            "basic_info": self.get_basic_info(),
            "data_quality": self.get_data_quality_report(),
            "statistical_summary": self.get_statistical_summary(),
            "column_analysis": self.get_column_analysis(),
            "relationships": self.get_relationships()
        }
    
    def _calculate_basic_info(self) -> Dict[str, Any]:
        """Calculate basic DataFrame information"""
        try:
            # Basic shape and structure
            info = {
                "shape": self.df.shape,
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "column_names": list(self.df.columns),
                "dtypes": self.df.dtypes.to_dict(),
                "memory_usage": self.df.memory_usage(deep=True).sum(),
                "index_type": str(type(self.df.index).__name__)
            }
            
            # Data type categorization
            numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = list(self.df.select_dtypes(include=['datetime']).columns)
            boolean_cols = list(self.df.select_dtypes(include=['bool']).columns)
            
            info["data_types"] = {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols,
                "boolean": boolean_cols,
                "counts": {
                    "numeric": len(numeric_cols),
                    "categorical": len(categorical_cols),
                    "datetime": len(datetime_cols),
                    "boolean": len(boolean_cols)
                }
            }
            
            # Index information
            info["index_info"] = {
                "name": self.df.index.name,
                "dtype": str(self.df.index.dtype),
                "is_unique": self.df.index.is_unique,
                "is_monotonic": self.df.index.is_monotonic_increasing,
                "has_duplicates": self.df.index.duplicated().any()
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error calculating basic info: {e}")
            return {"error": str(e)}
    
    def _calculate_data_quality(self) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        try:
            quality = {
                "completeness": {},
                "uniqueness": {},
                "consistency": {},
                "validity": {},
                "overall_score": 0.0
            }
            
            total_cells = len(self.df) * len(self.df.columns)
            
            # Completeness analysis
            null_counts = self.df.isnull().sum()
            total_nulls = null_counts.sum()
            
            quality["completeness"] = {
                "null_counts": null_counts.to_dict(),
                "null_percentages": (null_counts / len(self.df) * 100).round(2).to_dict(),
                "total_nulls": int(total_nulls),
                "completeness_rate": round((1 - total_nulls / total_cells) * 100, 2)
            }
            
            # Uniqueness analysis
            duplicate_rows = self.df.duplicated().sum()
            quality["uniqueness"] = {
                "duplicate_rows": int(duplicate_rows),
                "duplicate_percentage": round(duplicate_rows / len(self.df) * 100, 2),
                "unique_counts": {col: int(self.df[col].nunique()) for col in self.df.columns}
            }
            
            # Column-specific quality
            quality["column_quality"] = {}
            for col in self.df.columns:
                col_quality = self._analyze_column_quality(col)
                quality["column_quality"][col] = col_quality
            
            # Overall quality score (0-100)
            completeness_score = quality["completeness"]["completeness_rate"]
            uniqueness_score = (1 - duplicate_rows / len(self.df)) * 100 if len(self.df) > 0 else 0
            
            quality["overall_score"] = round((completeness_score * 0.6 + uniqueness_score * 0.4), 2)
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality: {e}")
            return {"error": str(e)}
    
    def _analyze_column_quality(self, column: str) -> Dict[str, Any]:
        """Analyze quality of a specific column"""
        try:
            col_data = self.df[column]
            
            quality = {
                "completeness": round((1 - col_data.isnull().sum() / len(col_data)) * 100, 2),
                "uniqueness": round(col_data.nunique() / len(col_data) * 100, 2),
                "issues": []
            }
            
            # Detect common quality issues
            if col_data.isnull().sum() > len(col_data) * 0.5:
                quality["issues"].append("High missing value percentage (>50%)")
            
            if col_data.nunique() == 1:
                quality["issues"].append("Constant value (no variation)")
            
            if col_data.dtype == 'object':
                # Check for inconsistent formatting
                if col_data.nunique() > len(col_data) * 0.9:
                    quality["issues"].append("Very high cardinality for categorical data")
                
                # Check for mixed case
                if any(isinstance(x, str) and x != x.lower() and x != x.upper() for x in col_data.dropna()):
                    quality["issues"].append("Inconsistent case formatting")
            
            elif pd.api.types.is_numeric_dtype(col_data):
                # Check for outliers
                if len(col_data.dropna()) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                    
                    if outliers > len(col_data) * 0.05:
                        quality["issues"].append(f"Potential outliers detected ({outliers} values)")
            
            return quality
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_statistical_summary(self) -> Dict[str, Any]:
        """Calculate statistical summary for numeric columns"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {"message": "No numeric columns found"}
            
            # Basic statistics
            summary = {
                "basic_stats": self.df[numeric_cols].describe().to_dict(),
                "additional_stats": {},
                "distribution_info": {}
            }
            
            # Additional statistics for each numeric column
            for col in numeric_cols:
                col_data = self.df[col].dropna()
                
                if len(col_data) > 0:
                    summary["additional_stats"][col] = {
                        "variance": float(col_data.var()),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis()),
                        "range": float(col_data.max() - col_data.min()),
                        "coefficient_of_variation": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else None,
                        "outlier_count": self._count_outliers(col_data)
                    }
                    
                    # Distribution characterization
                    summary["distribution_info"][col] = self._characterize_distribution(col_data)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical summary: {e}")
            return {"error": str(e)}
    
    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return int(((data < lower_bound) | (data > upper_bound)).sum())
    
    def _characterize_distribution(self, data: pd.Series) -> Dict[str, Any]:
        """Characterize the distribution of numeric data"""
        try:
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            # Distribution shape
            if abs(skewness) < 0.5:
                skew_desc = "approximately symmetric"
            elif skewness > 0.5:
                skew_desc = "right-skewed (positive skew)"
            else:
                skew_desc = "left-skewed (negative skew)"
            
            # Kurtosis interpretation
            if kurtosis > 3:
                kurt_desc = "heavy-tailed (leptokurtic)"
            elif kurtosis < 3:
                kurt_desc = "light-tailed (platykurtic)"
            else:
                kurt_desc = "normal-tailed (mesokurtic)"
            
            return {
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "skew_description": skew_desc,
                "kurtosis_description": kurt_desc,
                "is_normal_like": abs(skewness) < 0.5 and 2 < kurtosis < 4
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_column_analysis(self) -> Dict[str, Any]:
        """Calculate detailed analysis for each column"""
        try:
            analysis = {}
            
            for col in self.df.columns:
                col_data = self.df[col]
                dtype = str(col_data.dtype)
                
                col_analysis = {
                    "dtype": dtype,
                    "null_count": int(col_data.isnull().sum()),
                    "null_percentage": round(col_data.isnull().sum() / len(col_data) * 100, 2),
                    "unique_count": int(col_data.nunique()),
                    "unique_percentage": round(col_data.nunique() / len(col_data) * 100, 2)
                }
                
                # Type-specific analysis
                if pd.api.types.is_numeric_dtype(col_data):
                    col_analysis.update(self._analyze_numeric_column(col_data))
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_analysis.update(self._analyze_datetime_column(col_data))
                else:
                    col_analysis.update(self._analyze_categorical_column(col_data))
                
                analysis[col] = col_analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating column analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_numeric_column(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"analysis_type": "numeric", "message": "No valid numeric data"}
        
        return {
            "analysis_type": "numeric",
            "min": float(clean_data.min()),
            "max": float(clean_data.max()),
            "mean": float(clean_data.mean()),
            "median": float(clean_data.median()),
            "std": float(clean_data.std()),
            "zeros_count": int((clean_data == 0).sum()),
            "negative_count": int((clean_data < 0).sum()),
            "positive_count": int((clean_data > 0).sum())
        }
    
    def _analyze_datetime_column(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"analysis_type": "datetime", "message": "No valid datetime data"}
        
        return {
            "analysis_type": "datetime",
            "min_date": clean_data.min().isoformat() if hasattr(clean_data.min(), 'isoformat') else str(clean_data.min()),
            "max_date": clean_data.max().isoformat() if hasattr(clean_data.max(), 'isoformat') else str(clean_data.max()),
            "date_range_days": (clean_data.max() - clean_data.min()).days if len(clean_data) > 1 else 0
        }
    
    def _analyze_categorical_column(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze categorical/text column"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"analysis_type": "categorical", "message": "No valid categorical data"}
        
        value_counts = clean_data.value_counts()
        
        analysis = {
            "analysis_type": "categorical",
            "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
            "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
            "top_values": value_counts.head(5).to_dict()
        }
        
        # String-specific analysis
        if data.dtype == 'object':
            str_data = clean_data.astype(str)
            analysis.update({
                "avg_length": round(str_data.str.len().mean(), 2),
                "max_length": int(str_data.str.len().max()),
                "min_length": int(str_data.str.len().min()),
                "empty_strings": int((str_data == '').sum())
            })
        
        return analysis
    
    def _calculate_relationships(self) -> Dict[str, Any]:
        """Calculate relationships between columns"""
        try:
            relationships = {
                "correlations": {},
                "strong_correlations": [],
                "potential_dependencies": []
            }
            
            # Numeric correlations
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                relationships["correlations"] = corr_matrix.to_dict()
                
                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            relationships["strong_correlations"].append({
                                "column1": corr_matrix.columns[i],
                                "column2": corr_matrix.columns[j],
                                "correlation": round(float(corr_value), 3),
                                "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                            })
            
            # Look for potential functional dependencies
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col1 in categorical_cols:
                for col2 in categorical_cols:
                    if col1 != col2:
                        # Check if col1 values uniquely determine col2 values
                        if self._check_functional_dependency(col1, col2):
                            relationships["potential_dependencies"].append({
                                "determinant": col1,
                                "dependent": col2,
                                "type": "functional_dependency"
                            })
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error calculating relationships: {e}")
            return {"error": str(e)}
    
    def _check_functional_dependency(self, col1: str, col2: str) -> bool:
        """Check if col1 functionally determines col2"""
        try:
            # Group by col1 and check if col2 has unique values for each group
            grouped = self.df.groupby(col1)[col2].nunique()
            return (grouped == 1).all()
        except:
            return False
    
    def to_json(self, include_data: bool = False) -> str:
        """
        Convert analysis to JSON string
        
        Args:
            include_data: Whether to include actual data
            
        Returns:
            JSON string representation
        """
        report = self.get_comprehensive_report()
        
        if include_data:
            # Include sample data
            report["sample_data"] = self.df.head(5).to_dict('records')
        
        return json.dumps(report, indent=2, default=str)
    
    def get_summary_text(self) -> str:
        """
        Get human-readable summary text
        
        Returns:
            Summary as formatted text
        """
        basic = self.get_basic_info()
        quality = self.get_data_quality_report()
        
        summary = f"""
DataFrame Analysis Summary: {self.name}
{'=' * 50}

Basic Information:
- Shape: {basic['shape'][0]:,} rows × {basic['shape'][1]} columns
- Memory Usage: {basic['memory_usage'] / 1024 / 1024:.2f} MB
- Data Types: {basic['data_types']['counts']}

Data Quality:
- Completeness: {quality['completeness']['completeness_rate']:.1f}%
- Duplicate Rows: {quality['uniqueness']['duplicate_rows']:,} ({quality['uniqueness']['duplicate_percentage']:.1f}%)
- Overall Quality Score: {quality['overall_score']:.1f}/100

Column Summary:
"""
        
        for col in basic['column_names'][:10]:  # Show first 10 columns
            null_pct = quality['completeness']['null_percentages'].get(col, 0)
            unique_count = quality['uniqueness']['unique_counts'].get(col, 0)
            summary += f"- {col}: {unique_count:,} unique values, {null_pct:.1f}% null\n"
        
        if len(basic['column_names']) > 10:
            summary += f"... and {len(basic['column_names']) - 10} more columns\n"
        
        return summary 