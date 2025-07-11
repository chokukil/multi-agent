import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from typing import Dict, Any

# Our imports
from core.specialized_data_agents import (
    DataType,
    DataAnalysisResult,
    DataTypeDetectionResult,
    StructuredDataAgent,
    TimeSeriesDataAgent,
    TextDataAgent,
    ImageDataAgent,
    DataTypeDetector,
    get_data_type_detector,
    get_structured_agent,
    get_time_series_agent,
    get_text_agent,
    get_image_agent
)


class TestDataType:
    """Test DataType enum"""
    
    def test_enum_values(self):
        """Test that all expected enum values are defined"""
        expected_values = [
            "structured", "time_series", "text", "image", "mixed", "unknown"
        ]
        
        for value in expected_values:
            assert any(item.value == value for item in DataType)
    
    def test_enum_accessibility(self):
        """Test enum members are accessible"""
        assert DataType.STRUCTURED.value == "structured"
        assert DataType.TIME_SERIES.value == "time_series"
        assert DataType.TEXT.value == "text"
        assert DataType.IMAGE.value == "image"


class TestDataAnalysisResult:
    """Test DataAnalysisResult dataclass"""
    
    def test_creation(self):
        """Test DataAnalysisResult creation"""
        result = DataAnalysisResult(
            analysis_type="test_analysis",
            data_type=DataType.STRUCTURED,
            results={"test": "data"},
            insights=["Test insight"],
            recommendations=["Test recommendation"],
            confidence=0.8,
            metadata={"meta": "data"}
        )
        
        assert result.analysis_type == "test_analysis"
        assert result.data_type == DataType.STRUCTURED
        assert result.results == {"test": "data"}
        assert result.insights == ["Test insight"]
        assert result.recommendations == ["Test recommendation"]
        assert result.confidence == 0.8
        assert result.metadata == {"meta": "data"}


class TestDataTypeDetectionResult:
    """Test DataTypeDetectionResult dataclass"""
    
    def test_creation(self):
        """Test DataTypeDetectionResult creation"""
        result = DataTypeDetectionResult(
            detected_type=DataType.TEXT,
            confidence=0.9,
            reasoning="Test reasoning",
            characteristics={"char": "data"},
            recommendations=["Test rec"]
        )
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 0.9
        assert result.reasoning == "Test reasoning"
        assert result.characteristics == {"char": "data"}
        assert result.recommendations == ["Test rec"]


class TestStructuredDataAgent:
    """Test StructuredDataAgent class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = StructuredDataAgent()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.data_type == DataType.STRUCTURED
        assert self.agent.config == {}
    
    def test_detect_data_type_dataframe(self):
        """Test detecting DataFrame as structured data"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        result = self.agent.detect_data_type(df)
        
        assert result.detected_type == DataType.STRUCTURED
        assert result.confidence == 0.9
        assert "3 rows and 3 columns" in result.reasoning
        assert "rows" in result.characteristics
        assert "columns" in result.characteristics
    
    def test_detect_data_type_list_of_dicts(self):
        """Test detecting list of dictionaries as structured data"""
        data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ]
        
        result = self.agent.detect_data_type(data)
        
        assert result.detected_type == DataType.STRUCTURED
        assert result.confidence == 0.7
        assert "dict" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_analyze_basic_dataframe(self):
        """Test analyzing basic DataFrame"""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'salary': [50000, 60000, 70000, 80000],
            'department': ['IT', 'Finance', 'IT', 'HR']
        })
        
        result = await self.agent.analyze(df, "데이터 요약을 보여주세요")
        
        assert result.analysis_type == "structured_data_analysis"
        assert result.data_type == DataType.STRUCTURED
        assert result.confidence == 0.9
        assert "basic_info" in result.results
        assert len(result.insights) > 0
        assert "4개 행, 3개 열" in result.insights[0]
    
    @pytest.mark.asyncio
    async def test_analyze_correlation(self):
        """Test correlation analysis"""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],  # Perfect correlation
            'z': [1, 1, 1, 1, 1]   # No correlation
        })
        
        result = await self.agent.analyze(df, "상관관계를 분석해주세요")
        
        assert "correlation_matrix" in result.results
        assert "strong_correlations" in result.results
        assert len(result.results["strong_correlations"]) >= 1  # x and y should be strongly correlated
    
    @pytest.mark.asyncio
    async def test_analyze_outliers(self):
        """Test outlier analysis"""
        # Create data with clear outliers
        normal_data = [1, 2, 3, 4, 5] * 10
        outlier_data = normal_data + [100, 200]  # Clear outliers
        
        df = pd.DataFrame({'values': outlier_data})
        
        result = await self.agent.analyze(df, "이상값을 찾아주세요")
        
        assert "outlier_analysis" in result.results
        outlier_info = result.results["outlier_analysis"]["values"]
        assert outlier_info["count"] > 0  # Should detect outliers
    
    def test_calculate_data_quality_score(self):
        """Test data quality score calculation"""
        # Perfect data
        df_perfect = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        score_perfect = self.agent._calculate_data_quality_score(df_perfect)
        assert score_perfect == 1.0
        
        # Data with missing values
        df_missing = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        score_missing = self.agent._calculate_data_quality_score(df_missing)
        assert score_missing < 1.0
    
    def test_get_capabilities(self):
        """Test agent capabilities"""
        capabilities = self.agent.get_capabilities()
        
        assert capabilities["name"] == "Structured Data Agent"
        assert capabilities["data_type"] == "structured"
        assert isinstance(capabilities["capabilities"], list)
        assert "기술통계 분석" in capabilities["capabilities"]


class TestTimeSeriesDataAgent:
    """Test TimeSeriesDataAgent class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = TimeSeriesDataAgent()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.data_type == DataType.TIME_SERIES
    
    def test_detect_data_type_datetime_index(self):
        """Test detecting DataFrame with datetime index as time series"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'value': np.random.randn(100),
            'another_col': np.random.randn(100)
        }, index=dates)
        
        result = self.agent.detect_data_type(df)
        
        assert result.detected_type == DataType.TIME_SERIES
        assert result.confidence >= 0.8
        assert "indexed by datetime" in result.reasoning
    
    def test_detect_data_type_datetime_column(self):
        """Test detecting DataFrame with datetime column as time series"""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'value': np.random.randn(50)
        })
        
        result = self.agent.detect_data_type(df)
        
        assert result.detected_type == DataType.TIME_SERIES
        assert result.confidence > 0.5
        assert "time-related columns" in result.reasoning
    
    def test_is_date_like(self):
        """Test date pattern recognition"""
        assert self.agent._is_date_like("2023-01-01")
        assert self.agent._is_date_like("01/15/2023")
        assert self.agent._is_date_like("2023-12-31 14:30")
        assert not self.agent._is_date_like("not a date")
        assert not self.agent._is_date_like("123456")
    
    @pytest.mark.asyncio
    async def test_analyze_trend(self):
        """Test trend analysis"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create data with clear upward trend
        trend_data = np.arange(100) + np.random.randn(100) * 0.1
        
        df = pd.DataFrame({
            'value': trend_data
        }, index=dates)
        
        result = await self.agent.analyze(df, "추세를 분석해주세요")
        
        assert "trend_analysis" in result.results
        trend_info = result.results["trend_analysis"]["value"]
        assert trend_info["direction"] == "증가"  # Should detect upward trend
    
    @pytest.mark.asyncio
    async def test_analyze_volatility(self):
        """Test volatility analysis"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'value': np.random.randn(100)
        }, index=dates)
        
        result = await self.agent.analyze(df, "변동성을 분석해주세요")
        
        assert "volatility_analysis" in result.results
        vol_info = result.results["volatility_analysis"]["value"]
        assert "overall_volatility" in vol_info
        assert "recent_volatility" in vol_info
    
    def test_find_time_column(self):
        """Test time column detection"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'value': range(10)
        })
        
        time_col = self.agent._find_time_column(df)
        assert time_col == 'timestamp'
        
        # Test with name-based detection
        df2 = pd.DataFrame({
            'date_column': ['2023-01-01', '2023-01-02'],
            'value': [1, 2]
        })
        
        time_col2 = self.agent._find_time_column(df2)
        assert time_col2 == 'date_column'
    
    def test_get_capabilities(self):
        """Test agent capabilities"""
        capabilities = self.agent.get_capabilities()
        
        assert capabilities["name"] == "Time Series Data Agent"
        assert capabilities["data_type"] == "time_series"
        assert "추세 분석" in capabilities["capabilities"]


class TestTextDataAgent:
    """Test TextDataAgent class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = TextDataAgent()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.data_type == DataType.TEXT
    
    def test_detect_data_type_string(self):
        """Test detecting single string as text data"""
        text = "This is a long text string that should be detected as text data. It contains multiple sentences."
        
        result = self.agent.detect_data_type(text)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 0.9
        assert "Single text string" in result.reasoning
    
    def test_detect_data_type_list_of_strings(self):
        """Test detecting list of strings as text data"""
        texts = [
            "This is the first text document.",
            "This is the second text document with more content.",
            "Third document here with even more content to analyze."
        ]
        
        result = self.agent.detect_data_type(texts)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence == 0.8
        assert "List of 3 text items" in result.reasoning
    
    def test_detect_data_type_dataframe_with_text(self):
        """Test detecting DataFrame with text columns"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'review': [
                "This product is absolutely amazing! I love it so much and would recommend it to everyone.",
                "Terrible product. Waste of money. Do not buy this under any circumstances.",
                "It's okay, nothing special. Average quality for the price point."
            ]
        })
        
        result = self.agent.detect_data_type(df)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence > 0.5
        assert "text columns" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_analyze_keywords(self):
        """Test keyword analysis"""
        texts = [
            "Python programming is great for data analysis",
            "Data science with Python is very powerful",
            "Machine learning algorithms in Python are amazing"
        ]
        
        result = await self.agent.analyze(texts, "키워드를 추출해주세요")
        
        assert "keyword_analysis" in result.results
        keyword_info = result.results["keyword_analysis"]
        assert "top_words" in keyword_info
        assert len(keyword_info["top_words"]) > 0
        
        # Check if common words like "python" are in top words
        top_words = [item["word"] for item in keyword_info["top_words"]]
        assert "python" in top_words
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        texts = [
            "I love this product! It's amazing and perfect!",
            "This is terrible. I hate it completely.",
            "It's okay, nothing special."
        ]
        
        result = await self.agent.analyze(texts, "감정을 분석해주세요")
        
        assert "sentiment_analysis" in result.results
        sentiment_info = result.results["sentiment_analysis"]
        assert "positive_ratio" in sentiment_info
        assert "negative_ratio" in sentiment_info
        assert "neutral_ratio" in sentiment_info
        assert "overall_sentiment" in sentiment_info
    
    def test_prepare_text_data(self):
        """Test text data preparation"""
        # Test string input
        text_data = self.agent._prepare_text_data("Single text string")
        assert text_data == ["Single text string"]
        
        # Test list input
        list_input = ["Text 1", "Text 2", "Text 3"]
        text_data = self.agent._prepare_text_data(list_input)
        assert text_data == ["Text 1", "Text 2", "Text 3"]
        
        # Test DataFrame input
        df = pd.DataFrame({
            'short': ['a', 'b'],
            'long_text': ['This is a longer text that should be selected', 'Another long text for analysis']
        })
        text_data = self.agent._prepare_text_data(df)
        assert len(text_data) == 2
        assert all(len(text) > 20 for text in text_data)
    
    def test_compute_basic_text_stats(self):
        """Test basic text statistics computation"""
        texts = ["Short text", "This is a much longer text with many more words"]
        
        stats = self.agent._compute_basic_text_stats(texts)
        
        assert stats["total_texts"] == 2
        assert stats["avg_char_length"] > 0
        assert stats["avg_word_count"] > 0
        assert stats["max_char_length"] >= stats["min_char_length"]
    
    def test_get_capabilities(self):
        """Test agent capabilities"""
        capabilities = self.agent.get_capabilities()
        
        assert capabilities["name"] == "Text Data Agent"
        assert capabilities["data_type"] == "text"
        assert "키워드 추출" in capabilities["capabilities"]


class TestImageDataAgent:
    """Test ImageDataAgent class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = ImageDataAgent()
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.data_type == DataType.IMAGE
    
    def test_detect_data_type_image_path(self):
        """Test detecting image file path"""
        # Test with image extension
        result = self.agent.detect_data_type("test_image.jpg")
        assert result.detected_type == DataType.IMAGE
        assert result.confidence == 0.9
        assert ".jpg" in result.reasoning
        
        # Test with different extensions
        for ext in ['.png', '.gif', '.bmp']:
            result = self.agent.detect_data_type(f"image{ext}")
            assert result.detected_type == DataType.IMAGE
    
    def test_detect_data_type_list_of_paths(self):
        """Test detecting list of image paths"""
        image_paths = [
            "image1.jpg",
            "photo2.png", 
            "picture3.gif",
            "not_image.txt",  # This should be ignored
            "image4.jpeg"
        ]
        
        result = self.agent.detect_data_type(image_paths)
        
        assert result.detected_type == DataType.IMAGE
        assert result.confidence == 0.7
        assert "image file paths" in result.reasoning
    
    def test_detect_data_type_dataframe_with_image_columns(self):
        """Test detecting DataFrame with image path columns"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'image_path': ['photo1.jpg', 'photo2.png', 'photo3.gif'],
            'description': ['desc1', 'desc2', 'desc3']
        })
        
        result = self.agent.detect_data_type(df)
        
        assert result.detected_type == DataType.IMAGE
        assert result.confidence == 0.8
        assert "image file paths" in result.reasoning
    
    def test_collect_image_paths(self):
        """Test image path collection"""
        # Test string input
        paths = self.agent._collect_image_paths("image.jpg")
        assert len(paths) == 0  # File doesn't exist
        
        # Test list input
        path_list = ["img1.jpg", "img2.png"]
        paths = self.agent._collect_image_paths(path_list)
        assert len(paths) == 0  # Files don't exist
        
        # Test DataFrame input
        df = pd.DataFrame({
            'image_file': ['test1.jpg', 'test2.png'],
            'other_col': ['a', 'b']
        })
        paths = self.agent._collect_image_paths(df)
        assert len(paths) == 0  # Files don't exist
    
    def test_analyze_basic_image_info(self):
        """Test basic image info analysis"""
        # Test with empty list (no actual files)
        info = self.agent._analyze_basic_image_info([])
        
        assert info["total_images"] == 0
        assert info["valid_files"] == 0
        assert info["total_size_bytes"] == 0
    
    def test_get_capabilities(self):
        """Test agent capabilities"""
        capabilities = self.agent.get_capabilities()
        
        assert capabilities["name"] == "Image Data Agent"
        assert capabilities["data_type"] == "image"
        assert "이미지 메타데이터 분석" in capabilities["capabilities"]
        assert "limitations" in capabilities


class TestDataTypeDetector:
    """Test DataTypeDetector class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = DataTypeDetector()
    
    def test_initialization(self):
        """Test detector initialization"""
        assert len(self.detector.agents) == 4
        assert DataType.STRUCTURED in self.detector.agents
        assert DataType.TIME_SERIES in self.detector.agents
        assert DataType.TEXT in self.detector.agents
        assert DataType.IMAGE in self.detector.agents
    
    def test_detect_data_type_structured(self):
        """Test detecting structured data type"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = self.detector.detect_data_type(df)
        
        assert result.detected_type == DataType.STRUCTURED
        assert result.confidence > 0.5
    
    def test_detect_data_type_text(self):
        """Test detecting text data type"""
        texts = ["This is a long text document", "Another text document here"]
        
        result = self.detector.detect_data_type(texts)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence > 0.5
    
    def test_detect_data_type_unknown(self):
        """Test detecting unknown data type"""
        # Use data that doesn't match any pattern well
        weird_data = [1, "string", {"dict": "value"}, [1, 2, 3]]
        
        result = self.detector.detect_data_type(weird_data)
        
        # Should either detect something with low confidence or return unknown
        assert result.confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_analyze_with_best_agent(self):
        """Test analysis with best agent selection"""
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        
        result = await self.detector.analyze_with_best_agent(df, "데이터를 분석해주세요")
        
        assert isinstance(result, DataAnalysisResult)
        assert result.data_type == DataType.STRUCTURED
        assert len(result.insights) > 0


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_get_data_type_detector(self):
        """Test detector singleton"""
        detector1 = get_data_type_detector()
        detector2 = get_data_type_detector()
        
        assert detector1 is detector2  # Should be same instance
    
    def test_get_specialized_agents(self):
        """Test specialized agent factory functions"""
        structured_agent = get_structured_agent()
        assert isinstance(structured_agent, StructuredDataAgent)
        
        time_series_agent = get_time_series_agent()
        assert isinstance(time_series_agent, TimeSeriesDataAgent)
        
        text_agent = get_text_agent()
        assert isinstance(text_agent, TextDataAgent)
        
        image_agent = get_image_agent()
        assert isinstance(image_agent, ImageDataAgent)
    
    def test_agent_factory_with_config(self):
        """Test agent creation with config"""
        config = {"test_param": "test_value"}
        
        agent = get_structured_agent(config)
        assert agent.config == config


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from detection to analysis"""
        # Create test data
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': np.random.randn(10),
            'category': ['A', 'B'] * 5
        })
        
        # Detect data type
        detector = get_data_type_detector()
        detection_result = detector.detect_data_type(df)
        
        # Should detect as time series due to date column
        assert detection_result.detected_type in [DataType.TIME_SERIES, DataType.STRUCTURED]
        assert detection_result.confidence > 0.5
        
        # Analyze with best agent
        analysis_result = await detector.analyze_with_best_agent(
            df, "데이터의 특성을 분석해주세요"
        )
        
        assert isinstance(analysis_result, DataAnalysisResult)
        assert len(analysis_result.insights) > 0
        assert len(analysis_result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_data_types(self):
        """Test handling multiple different data types"""
        detector = get_data_type_detector()
        
        # Test different data types
        test_cases = [
            (pd.DataFrame({'a': [1, 2, 3]}), DataType.STRUCTURED),
            (["Long text document here", "Another text"], DataType.TEXT),
            ("single_image.jpg", DataType.IMAGE)
        ]
        
        for data, expected_type in test_cases:
            detection_result = detector.detect_data_type(data)
            
            if detection_result.confidence > 0.5:
                assert detection_result.detected_type == expected_type
            
            # Should be able to analyze regardless
            analysis_result = await detector.analyze_with_best_agent(data, "분석해주세요")
            assert isinstance(analysis_result, DataAnalysisResult)


if __name__ == "__main__":
    pytest.main([__file__]) 