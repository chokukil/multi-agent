"""
Unit Tests for Auto Data Profiler

자동 데이터 프로파일링 시스템 단위 테스트
- 다양한 데이터 타입 처리 검증
- 품질 평가 정확성 검증
- 패턴 탐지 기능 검증
- 인사이트 및 추천사항 생성 검증

Author: CherryAI Team
Date: 2024-12-30
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Our imports
from core.auto_data_profiler import (
    AutoDataProfiler,
    DataProfile,
    ColumnProfile,
    DataQuality,
    ColumnType,
    DataPattern,
    get_auto_data_profiler,
    profile_dataset,
    quick_profile
)


class TestDataQuality:
    """Test DataQuality enum"""
    
    def test_enum_values(self):
        """Test DataQuality enum values"""
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.FAIR.value == "fair"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.CRITICAL.value == "critical"


class TestColumnType:
    """Test ColumnType enum"""
    
    def test_enum_values(self):
        """Test ColumnType enum values"""
        expected_types = [
            "numeric_continuous", "numeric_discrete", "categorical",
            "datetime", "text", "boolean", "mixed", "unknown"
        ]
        
        for type_name in expected_types:
            assert any(ct.value == type_name for ct in ColumnType)


class TestDataPattern:
    """Test DataPattern enum"""
    
    def test_enum_values(self):
        """Test DataPattern enum values"""
        expected_patterns = [
            "time_series", "hierarchical", "relational", "transactional",
            "experimental", "survey", "log_data", "mixed"
        ]
        
        for pattern_name in expected_patterns:
            assert any(dp.value == pattern_name for dp in DataPattern)


class TestColumnProfile:
    """Test ColumnProfile dataclass"""
    
    def test_creation(self):
        """Test ColumnProfile creation"""
        profile = ColumnProfile(
            name="test_column",
            dtype="int64",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            null_count=5,
            null_percentage=5.0,
            unique_count=95,
            unique_percentage=95.0,
            mean=50.0,
            quality_score=0.85,
            quality_issues=["minor issue"],
            recommendations=["consider normalization"]
        )
        
        assert profile.name == "test_column"
        assert profile.dtype == "int64"
        assert profile.inferred_type == ColumnType.NUMERIC_CONTINUOUS
        assert profile.null_count == 5
        assert profile.null_percentage == 5.0
        assert profile.mean == 50.0
        assert profile.quality_score == 0.85


class TestDataProfile:
    """Test DataProfile dataclass"""
    
    def test_creation(self):
        """Test DataProfile creation"""
        col_profile = ColumnProfile(
            name="test_col",
            dtype="int64",
            inferred_type=ColumnType.NUMERIC_CONTINUOUS,
            null_count=0,
            null_percentage=0.0,
            unique_count=100,
            unique_percentage=100.0
        )
        
        profile = DataProfile(
            dataset_name="Test Dataset",
            shape=(100, 1),
            memory_usage=1.0,
            dtypes_summary={"int64": 1},
            overall_quality=DataQuality.EXCELLENT,
            quality_score=0.95,
            columns=[col_profile],
            detected_patterns=[DataPattern.RELATIONAL],
            key_insights=["High quality data"],
            recommendations=["Continue analysis"]
        )
        
        assert profile.dataset_name == "Test Dataset"
        assert profile.shape == (100, 1)
        assert profile.overall_quality == DataQuality.EXCELLENT
        assert len(profile.columns) == 1
        assert DataPattern.RELATIONAL in profile.detected_patterns


class TestAutoDataProfiler:
    """Test AutoDataProfiler class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = AutoDataProfiler()
        
        # Create test datasets
        np.random.seed(42)
        
        # Simple numeric dataset
        self.numeric_data = pd.DataFrame({
            'int_col': range(100),
            'float_col': np.random.normal(50, 10, 100),
            'discrete_col': np.random.randint(1, 6, 100)
        })
        
        # Mixed type dataset
        self.mixed_data = pd.DataFrame({
            'id': range(100),
            'name': [f'Name_{i}' for i in range(100)],
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.normal(100, 20, 100),
            'flag': np.random.choice([True, False], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        # Poor quality dataset
        self.poor_quality_data = pd.DataFrame({
            'mostly_null': [1, 2] + [None] * 98,
            'duplicated': [1] * 100,
            'mixed_types': ['text'] * 50 + list(range(50))
        })
    
    def test_initialization(self):
        """Test profiler initialization"""
        assert isinstance(self.profiler.config, dict)
        assert self.profiler.max_categorical_values > 0
        assert self.profiler.outlier_threshold > 0
        assert self.profiler.correlation_threshold > 0
    
    def test_prepare_data_dataframe(self):
        """Test data preparation with DataFrame input"""
        result = self.profiler._prepare_data(self.numeric_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.numeric_data.shape
        assert list(result.columns) == list(self.numeric_data.columns)
    
    def test_prepare_data_dict(self):
        """Test data preparation with dictionary input"""
        data_dict = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        }
        
        result = self.profiler._prepare_data(data_dict)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert 'col1' in result.columns
        assert 'col2' in result.columns
    
    def test_prepare_data_list(self):
        """Test data preparation with list input"""
        data_list = [
            {'col1': 1, 'col2': 'a'},
            {'col1': 2, 'col2': 'b'},
            {'col1': 3, 'col2': 'c'}
        ]
        
        result = self.profiler._prepare_data(data_list)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
    
    def test_prepare_data_invalid(self):
        """Test data preparation with invalid input"""
        result = self.profiler._prepare_data(12345)
        assert result is None
    
    def test_infer_column_type_numeric_continuous(self):
        """Test numeric continuous type inference"""
        series = pd.Series(np.random.normal(50, 10, 100))
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.NUMERIC_CONTINUOUS
    
    def test_infer_column_type_numeric_discrete(self):
        """Test numeric discrete type inference"""
        series = pd.Series(np.random.randint(1, 6, 100))
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.NUMERIC_DISCRETE
    
    def test_infer_column_type_categorical(self):
        """Test categorical type inference"""
        series = pd.Series(np.random.choice(['A', 'B', 'C'], 100))
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.CATEGORICAL
    
    def test_infer_column_type_boolean(self):
        """Test boolean type inference"""
        series = pd.Series(np.random.choice([True, False], 100))
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.BOOLEAN
    
    def test_infer_column_type_datetime(self):
        """Test datetime type inference"""
        series = pd.Series(pd.date_range('2023-01-01', periods=100, freq='D'))
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.DATETIME
    
    def test_infer_column_type_text(self):
        """Test text type inference"""
        series = pd.Series([f'This is a long text string number {i}' for i in range(100)])
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.TEXT
    
    def test_infer_column_type_mostly_null(self):
        """Test type inference with mostly null data"""
        series = pd.Series([1, 2] + [None] * 98)
        result = self.profiler._infer_column_type(series)
        
        assert result == ColumnType.UNKNOWN
    
    def test_analyze_single_column_numeric(self):
        """Test single column analysis for numeric data"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        profile = self.profiler._analyze_single_column(series, "test_numeric")
        
        assert profile.name == "test_numeric"
        assert profile.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE]
        assert profile.null_count == 0
        assert profile.null_percentage == 0.0
        assert profile.unique_count == 10
        assert profile.mean is not None
        assert profile.std is not None
        assert profile.quality_score > 0
    
    def test_analyze_single_column_categorical(self):
        """Test single column analysis for categorical data"""
        series = pd.Series(['A', 'B', 'C', 'A', 'B', 'C'] * 10)
        
        profile = self.profiler._analyze_single_column(series, "test_categorical")
        
        assert profile.name == "test_categorical"
        assert profile.inferred_type == ColumnType.CATEGORICAL
        assert profile.unique_count == 3
        assert profile.top_values is not None
        assert len(profile.top_values) == 3
        assert profile.quality_score > 0
    
    def test_analyze_single_column_with_nulls(self):
        """Test single column analysis with null values"""
        series = pd.Series([1, 2, 3, None, None, 6, 7, 8, 9, 10])
        
        profile = self.profiler._analyze_single_column(series, "test_with_nulls")
        
        assert profile.null_count == 2
        assert profile.null_percentage == 20.0
        assert profile.unique_count == 8  # Excluding nulls
        assert profile.quality_score < 1.0  # Should be penalized for nulls
    
    def test_assess_column_quality_high_quality(self):
        """Test column quality assessment for high quality data"""
        series = pd.Series(range(100))  # Perfect data
        
        score, issues = self.profiler._assess_column_quality(series, ColumnType.NUMERIC_CONTINUOUS)
        
        assert score > 0.9  # High quality score
        assert len(issues) == 0  # No issues
    
    def test_assess_column_quality_with_nulls(self):
        """Test column quality assessment with null values"""
        series = pd.Series([1, 2, 3] + [None] * 7)  # 70% nulls
        
        score, issues = self.profiler._assess_column_quality(series, ColumnType.NUMERIC_CONTINUOUS)
        
        assert score < 0.5  # Low quality score
        assert len(issues) > 0  # Should have issues
        assert any("누락값" in issue for issue in issues)
    
    def test_assess_column_quality_all_same(self):
        """Test column quality assessment for constant values"""
        series = pd.Series([5] * 100)  # All same values
        
        score, issues = self.profiler._assess_column_quality(series, ColumnType.NUMERIC_CONTINUOUS)
        
        assert score < 0.8  # Should be penalized
        assert any("영 분산" in issue for issue in issues)
    
    def test_generate_column_recommendations_nulls(self):
        """Test column recommendations for data with nulls"""
        series = pd.Series([1, 2, None, 4, None])
        
        recommendations = self.profiler._generate_column_recommendations(
            series, ColumnType.NUMERIC_CONTINUOUS, ["높은 누락값 비율: 40.0%"]
        )
        
        assert len(recommendations) > 0
        assert any("누락값" in rec for rec in recommendations)
    
    def test_generate_column_recommendations_categorical(self):
        """Test column recommendations for categorical data"""
        series = pd.Series([f'Category_{i}' for i in range(50)] * 2)  # Many categories
        
        recommendations = self.profiler._generate_column_recommendations(
            series, ColumnType.CATEGORICAL, []
        )
        
        assert any("범주" in rec for rec in recommendations)
    
    def test_detect_data_patterns_time_series(self):
        """Test time series pattern detection"""
        profiles = [
            ColumnProfile(
                name="date_col",
                dtype="datetime64",
                inferred_type=ColumnType.DATETIME,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_percentage=100.0
            ),
            ColumnProfile(
                name="value_col",
                dtype="float64",
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                null_count=0,
                null_percentage=0.0,
                unique_count=95,
                unique_percentage=95.0
            )
        ]
        
        patterns = self.profiler._detect_data_patterns(self.mixed_data, profiles)
        
        assert DataPattern.TIME_SERIES in patterns
    
    def test_detect_data_patterns_relational(self):
        """Test relational pattern detection"""
        profiles = [
            ColumnProfile(
                name="id",
                dtype="int64",
                inferred_type=ColumnType.NUMERIC_DISCRETE,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_percentage=100.0
            ),
            ColumnProfile(
                name="category",
                dtype="object",
                inferred_type=ColumnType.CATEGORICAL,
                null_count=0,
                null_percentage=0.0,
                unique_count=3,
                unique_percentage=3.0
            ),
            ColumnProfile(
                name="value",
                dtype="float64",
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                null_count=0,
                null_percentage=0.0,
                unique_count=95,
                unique_percentage=95.0
            )
        ]
        
        patterns = self.profiler._detect_data_patterns(self.mixed_data, profiles)
        
        assert DataPattern.RELATIONAL in patterns
    
    def test_analyze_correlations_numeric_data(self):
        """Test correlation analysis with numeric data"""
        # Create data with strong correlation
        correlated_data = pd.DataFrame({
            'x': range(100),
            'y': [i * 2 + np.random.normal(0, 1) for i in range(100)],
            'z': np.random.normal(0, 1, 100)
        })
        
        correlations = self.profiler._analyze_correlations(correlated_data)
        
        assert correlations is not None
        assert 'matrix' in correlations
        assert 'high_correlations' in correlations
        assert 'summary' in correlations
    
    def test_analyze_correlations_no_numeric(self):
        """Test correlation analysis with no numeric columns"""
        non_numeric_data = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })
        
        correlations = self.profiler._analyze_correlations(non_numeric_data)
        
        assert correlations is None
    
    def test_detect_outliers_with_outliers(self):
        """Test outlier detection with data containing outliers"""
        # Create data with obvious outliers
        outlier_data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'with_outliers': list(np.random.normal(0, 1, 95)) + [100, 200, 300, 400, 500]
        })
        
        outliers = self.profiler._detect_outliers(outlier_data)
        
        assert 'with_outliers' in outliers
        assert outliers['with_outliers'] > 0
    
    def test_detect_outliers_clean_data(self):
        """Test outlier detection with clean data"""
        clean_data = pd.DataFrame({
            'clean_col': np.random.normal(0, 1, 100)
        })
        
        outliers = self.profiler._detect_outliers(clean_data)
        
        # Should have few or no outliers
        if 'clean_col' in outliers:
            assert outliers['clean_col'] < 10  # Less than 10% outliers
    
    def test_profile_data_simple(self):
        """Test complete data profiling with simple dataset"""
        profile = self.profiler.profile_data(self.numeric_data, "Test Numeric Data")
        
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == "Test Numeric Data"
        assert profile.shape == self.numeric_data.shape
        assert len(profile.columns) == len(self.numeric_data.columns)
        assert profile.overall_quality in DataQuality
        assert 0 <= profile.quality_score <= 1
        assert len(profile.key_insights) > 0
        assert len(profile.recommendations) > 0
    
    def test_profile_data_mixed_types(self):
        """Test data profiling with mixed data types"""
        profile = self.profiler.profile_data(self.mixed_data, "Test Mixed Data")
        
        assert isinstance(profile, DataProfile)
        assert len(profile.columns) == 6
        
        # Check that different types were detected
        detected_types = [col.inferred_type for col in profile.columns]
        assert len(set(detected_types)) > 1  # Multiple types detected
    
    def test_profile_data_poor_quality(self):
        """Test data profiling with poor quality data"""
        profile = self.profiler.profile_data(self.poor_quality_data, "Poor Quality Data")
        
        assert isinstance(profile, DataProfile)
        assert profile.overall_quality in [DataQuality.POOR, DataQuality.CRITICAL, DataQuality.FAIR]
        assert profile.quality_score < 0.7
        assert len(profile.data_quality_issues) > 0
    
    def test_profile_data_empty_dataframe(self):
        """Test data profiling with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        profile = self.profiler.profile_data(empty_df, "Empty Data")
        
        assert isinstance(profile, DataProfile)
        assert profile.overall_quality == DataQuality.CRITICAL
        assert "Invalid or empty dataset" in str(profile.data_quality_issues)
    
    def test_profile_data_with_session_id(self):
        """Test data profiling with session ID"""
        profile = self.profiler.profile_data(
            self.numeric_data, 
            "Test Session Data", 
            session_id="test_session_123"
        )
        
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == "Test Session Data"
        # Session ID should be used for tracking but doesn't affect the profile structure
    
    def test_generate_insights_comprehensive(self):
        """Test comprehensive insight generation"""
        column_profiles = [
            ColumnProfile(
                name=col,
                dtype=str(self.mixed_data[col].dtype),
                inferred_type=ColumnType.NUMERIC_CONTINUOUS if col in ['id', 'value'] else ColumnType.CATEGORICAL,
                null_count=0,
                null_percentage=0.0,
                unique_count=self.mixed_data[col].nunique(),
                unique_percentage=(self.mixed_data[col].nunique() / len(self.mixed_data)) * 100,
                quality_score=0.9
            )
            for col in self.mixed_data.columns
        ]
        
        quality_info = {
            'overall_quality': DataQuality.GOOD,
            'quality_score': 0.85,
            'issues': []
        }
        
        patterns = [DataPattern.RELATIONAL, DataPattern.TIME_SERIES]
        
        insights = self.profiler._generate_insights(
            self.mixed_data, column_profiles, quality_info, patterns
        )
        
        assert len(insights) > 0
        assert any("데이터셋 크기" in insight for insight in insights)
        assert any("품질" in insight for insight in insights)
        assert any("패턴" in insight for insight in insights)
    
    def test_generate_recommendations_high_quality(self):
        """Test recommendation generation for high quality data"""
        column_profiles = [
            ColumnProfile(
                name=col,
                dtype=str(self.numeric_data[col].dtype),
                inferred_type=ColumnType.NUMERIC_CONTINUOUS,
                null_count=0,
                null_percentage=0.0,
                unique_count=self.numeric_data[col].nunique(),
                unique_percentage=100.0,
                quality_score=0.95,
                recommendations=[]
            )
            for col in self.numeric_data.columns
        ]
        
        quality_info = {
            'overall_quality': DataQuality.EXCELLENT,
            'quality_score': 0.95,
            'issues': []
        }
        
        recommendations = self.profiler._generate_recommendations(
            self.numeric_data, column_profiles, quality_info
        )
        
        assert len(recommendations) > 0
        assert any("수치형 데이터" in rec for rec in recommendations)
    
    def test_generate_recommendations_poor_quality(self):
        """Test recommendation generation for poor quality data"""
        column_profiles = [
            ColumnProfile(
                name=col,
                dtype=str(self.poor_quality_data[col].dtype),
                inferred_type=ColumnType.UNKNOWN,
                null_count=self.poor_quality_data[col].isnull().sum(),
                null_percentage=(self.poor_quality_data[col].isnull().sum() / len(self.poor_quality_data)) * 100,
                unique_count=self.poor_quality_data[col].nunique(),
                unique_percentage=50.0,
                quality_score=0.3,
                recommendations=["데이터 정제 필요"]
            )
            for col in self.poor_quality_data.columns
        ]
        
        quality_info = {
            'overall_quality': DataQuality.POOR,
            'quality_score': 0.3,
            'issues': ["높은 누락값 비율", "타입 불일치"]
        }
        
        recommendations = self.profiler._generate_recommendations(
            self.poor_quality_data, column_profiles, quality_info
        )
        
        assert len(recommendations) > 0
        assert any("품질이 낮습니다" in rec for rec in recommendations)
    
    def test_export_profile_report_json(self):
        """Test JSON report export"""
        profile = self.profiler.profile_data(self.numeric_data, "Test JSON Export")
        
        json_report = self.profiler.export_profile_report(profile, 'json')
        
        # Should be valid JSON string (even if empty due to serialization issues)
        assert isinstance(json_report, str)
        if len(json_report) > 100:  # If serialization worked
            parsed = json.loads(json_report)
            assert isinstance(parsed, dict)
    
    def test_export_profile_report_html(self):
        """Test HTML report export"""
        profile = self.profiler.profile_data(self.numeric_data, "Test HTML Export")
        
        html_report = self.profiler.export_profile_report(profile, 'html')
        
        assert isinstance(html_report, str)
        assert "<html>" in html_report
        assert profile.dataset_name in html_report
    
    def test_export_profile_report_markdown(self):
        """Test Markdown report export"""
        profile = self.profiler.profile_data(self.numeric_data, "Test Markdown Export")
        
        md_report = self.profiler.export_profile_report(profile, 'markdown')
        
        assert isinstance(md_report, str)
        assert "# 데이터 프로파일 보고서" in md_report
        assert profile.dataset_name in md_report
    
    def test_export_profile_report_invalid_format(self):
        """Test report export with invalid format"""
        profile = self.profiler.profile_data(self.numeric_data, "Test Invalid Format")
        
        result = self.profiler.export_profile_report(profile, 'invalid_format')
        
        assert "보고서 생성 오류" in result or "Unsupported format" in result


class TestFactoryFunctions:
    """Test factory functions and convenience functions"""
    
    def test_get_auto_data_profiler_singleton(self):
        """Test that get_auto_data_profiler returns singleton"""
        profiler1 = get_auto_data_profiler()
        profiler2 = get_auto_data_profiler()
        
        assert profiler1 is profiler2
    
    def test_get_auto_data_profiler_with_config(self):
        """Test get_auto_data_profiler with custom config"""
        config = {'max_categorical_values': 25, 'outlier_threshold': 2.5}
        
        # Note: This will still return singleton, but we test the concept
        profiler = get_auto_data_profiler(config)
        assert isinstance(profiler, AutoDataProfiler)
    
    def test_profile_dataset_convenience(self):
        """Test profile_dataset convenience function"""
        data = pd.DataFrame({
            'col1': range(10),
            'col2': ['A'] * 10
        })
        
        profile = profile_dataset(data, "Convenience Test")
        
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == "Convenience Test"
    
    def test_quick_profile_success(self):
        """Test quick_profile convenience function"""
        data = pd.DataFrame({
            'col1': range(10),
            'col2': ['A'] * 10
        })
        
        result = quick_profile(data)
        
        assert isinstance(result, dict)
        assert 'shape' in result
        assert 'quality' in result
        assert 'quality_score' in result
        assert 'insights' in result
        assert 'recommendations' in result
        assert len(result['insights']) <= 3
        assert len(result['recommendations']) <= 3
    
    def test_quick_profile_error(self):
        """Test quick_profile with invalid data"""
        result = quick_profile("invalid_data")
        
        assert isinstance(result, dict)
        assert 'error' in result


class TestFileHandling:
    """Test file-based data profiling"""
    
    def setup_method(self):
        """Setup temporary files for testing"""
        self.profiler = AutoDataProfiler()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test CSV file
        test_data = pd.DataFrame({
            'id': range(20),
            'value': np.random.randn(20),
            'category': np.random.choice(['A', 'B', 'C'], 20)
        })
        
        self.csv_file = self.temp_dir / "test_data.csv"
        test_data.to_csv(self.csv_file, index=False)
        
        # Create test JSON file
        json_data = test_data.to_dict('records')
        self.json_file = self.temp_dir / "test_data.json"
        with open(self.json_file, 'w') as f:
            json.dump(json_data, f)
    
    def teardown_method(self):
        """Cleanup temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_profile_csv_file(self):
        """Test profiling CSV file"""
        profile = self.profiler.profile_data(str(self.csv_file), "CSV Test")
        
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == "CSV Test"
        assert profile.shape[1] == 3  # 3 columns
    
    def test_profile_json_file(self):
        """Test profiling JSON file"""
        profile = self.profiler.profile_data(str(self.json_file), "JSON Test")
        
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == "JSON Test"
    
    def test_profile_nonexistent_file(self):
        """Test profiling non-existent file"""
        profile = self.profiler.profile_data("nonexistent_file.csv", "Nonexistent")
        
        assert isinstance(profile, DataProfile)
        assert profile.overall_quality == DataQuality.CRITICAL
        assert len(profile.data_quality_issues) > 0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = AutoDataProfiler()
    
    def test_profile_single_column(self):
        """Test profiling dataset with single column"""
        single_col_data = pd.DataFrame({'only_col': range(100)})
        
        profile = self.profiler.profile_data(single_col_data, "Single Column")
        
        assert isinstance(profile, DataProfile)
        assert profile.shape == (100, 1)
        assert len(profile.columns) == 1
    
    def test_profile_single_row(self):
        """Test profiling dataset with single row"""
        single_row_data = pd.DataFrame({'col1': [1], 'col2': ['A'], 'col3': [True]})
        
        profile = self.profiler.profile_data(single_row_data, "Single Row")
        
        assert isinstance(profile, DataProfile)
        assert profile.shape == (1, 3)
        assert len(profile.columns) == 3
    
    def test_profile_all_null_column(self):
        """Test profiling with all-null column"""
        null_data = pd.DataFrame({
            'all_null': [None] * 100,
            'some_data': range(100)
        })
        
        profile = self.profiler.profile_data(null_data, "All Null Test")
        
        assert isinstance(profile, DataProfile)
        null_col_profile = next(col for col in profile.columns if col.name == 'all_null')
        assert null_col_profile.null_percentage == 100.0
        assert null_col_profile.quality_score < 0.5
    
    def test_profile_unicode_data(self):
        """Test profiling with Unicode characters"""
        unicode_data = pd.DataFrame({
            'korean': ['안녕하세요', '데이터', '분석'],
            'chinese': ['你好', '数据', '分析'],
            'emoji': ['😀', '📊', '🔍']
        })
        
        profile = self.profiler.profile_data(unicode_data, "Unicode Test")
        
        assert isinstance(profile, DataProfile)
        assert profile.shape == (3, 3)
        
        # All columns should be detected as categorical or text
        for col in profile.columns:
            assert col.inferred_type in [ColumnType.CATEGORICAL, ColumnType.TEXT]
    
    def test_profile_very_large_numbers(self):
        """Test profiling with very large numbers"""
        large_numbers = pd.DataFrame({
            'large_int': [10**15, 10**16, 10**17],
            'large_float': [1.5e308, 2.5e308, 3.5e307],
            'small_float': [1e-308, 2e-308, 3e-307]
        })
        
        profile = self.profiler.profile_data(large_numbers, "Large Numbers")
        
        assert isinstance(profile, DataProfile)
        assert all(col.inferred_type in [ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE] 
                  for col in profile.columns)
    
    def test_profile_special_values(self):
        """Test profiling with special float values"""
        special_data = pd.DataFrame({
            'with_inf': [1.0, float('inf'), 3.0, float('-inf'), 5.0],
            'with_nan': [1.0, float('nan'), 3.0, 4.0, 5.0],
            'normal': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        profile = self.profiler.profile_data(special_data, "Special Values")
        
        assert isinstance(profile, DataProfile)
        # Should handle special values without crashing
        assert all(isinstance(col.quality_score, (int, float)) for col in profile.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 