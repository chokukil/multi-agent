"""
아티팩트 추출 테스트

Task 4.1.1: 아티팩트 추출 테스트 - 모든 A2A 응답 파싱 시나리오 테스트
아티팩트 타입별 추출 정확성 검증 및 에러 상황 처리 테스트

테스트 시나리오:
1. Plotly Chart 추출 테스트
2. DataFrame 추출 테스트  
3. Image 추출 테스트
4. Code 추출 테스트
5. Text 추출 테스트
6. 다중 아티팩트 추출 테스트
7. 에러 상황 처리 테스트
8. 메타데이터 추출 테스트
"""

import unittest
import json
import base64
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from modules.artifacts.a2a_artifact_extractor import A2AArtifactExtractor, ArtifactInfo, ArtifactType
from modules.artifacts.artifact_parsers import (
    PlotlyArtifactParser, DataFrameArtifactParser, ImageArtifactParser,
    CodeArtifactParser, TextArtifactParser
)

class TestArtifactExtraction(unittest.TestCase):
    """아티팩트 추출 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.extractor = A2AArtifactExtractor()
        
        # 테스트 데이터
        self.sample_plotly_data = {
            "data": [
                {
                    "x": [1, 2, 3, 4],
                    "y": [10, 11, 12, 13],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Test Line"
                }
            ],
            "layout": {
                "title": "Test Chart",
                "xaxis": {"title": "X Axis"},
                "yaxis": {"title": "Y Axis"}
            }
        }
        
        self.sample_dataframe_data = {
            "columns": ["Name", "Age", "City"],
            "data": [
                ["Alice", 25, "New York"],
                ["Bob", 30, "Los Angeles"],
                ["Charlie", 35, "Chicago"]
            ],
            "index": [0, 1, 2]
        }
        
        # Base64 이미지 (1x1 투명 PNG)
        self.sample_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        self.sample_code = """
def hello_world():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
"""
        
        self.sample_markdown = """
# Test Document

## Introduction
This is a test document with **bold** text and *italic* text.

### Features
- Feature 1
- Feature 2
- Feature 3

```python
def example():
    return "test"
```
"""
    
    def test_plotly_chart_extraction(self):
        """Plotly 차트 추출 테스트"""
        
        # A2A 응답 모방
        a2a_response = {
            "agent_id": "visualization_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "plotly_chart",
                    "title": "Test Visualization",
                    "data": self.sample_plotly_data,
                    "metadata": {
                        "chart_type": "line",
                        "interactive": True,
                        "created_at": datetime.now().isoformat()
                    }
                }
            ]
        }
        
        # 추출 실행
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        # 검증
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.PLOTLY_CHART)
        self.assertEqual(artifact.title, "Test Visualization")
        self.assertIsInstance(artifact.data, dict)
        self.assertIn("data", artifact.data)
        self.assertIn("layout", artifact.data)
        self.assertEqual(artifact.metadata["chart_type"], "line")
        self.assertTrue(artifact.metadata["interactive"])
    
    def test_dataframe_extraction(self):
        """DataFrame 추출 테스트"""
        
        a2a_response = {
            "agent_id": "analysis_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "dataframe",
                    "title": "Analysis Results",
                    "data": self.sample_dataframe_data,
                    "metadata": {
                        "rows": 3,
                        "columns": 3,
                        "memory_usage": "240 bytes"
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.DATAFRAME)
        self.assertEqual(artifact.title, "Analysis Results")
        self.assertIsInstance(artifact.data, dict)
        self.assertIn("columns", artifact.data)
        self.assertIn("data", artifact.data)
        self.assertEqual(len(artifact.data["columns"]), 3)
        self.assertEqual(len(artifact.data["data"]), 3)
    
    def test_image_extraction(self):
        """이미지 추출 테스트"""
        
        a2a_response = {
            "agent_id": "visualization_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "image",
                    "title": "Generated Plot",
                    "data": self.sample_image_b64,
                    "metadata": {
                        "format": "PNG",
                        "size": "1x1",
                        "file_size": "67 bytes"
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.IMAGE)
        self.assertEqual(artifact.title, "Generated Plot")
        self.assertIsInstance(artifact.data, str)
        self.assertTrue(artifact.data.startswith("iVBORw0KGgo"))  # PNG header
        self.assertEqual(artifact.metadata["format"], "PNG")
    
    def test_code_extraction(self):
        """코드 추출 테스트"""
        
        a2a_response = {
            "agent_id": "code_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "code",
                    "title": "Solution Code",
                    "data": self.sample_code,
                    "metadata": {
                        "language": "python",
                        "lines": 7,
                        "functions": ["hello_world"]
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.CODE)
        self.assertEqual(artifact.title, "Solution Code")
        self.assertIsInstance(artifact.data, str)
        self.assertIn("def hello_world()", artifact.data)
        self.assertEqual(artifact.metadata["language"], "python")
        self.assertEqual(artifact.metadata["lines"], 7)
    
    def test_text_extraction(self):
        """텍스트 추출 테스트"""
        
        a2a_response = {
            "agent_id": "analysis_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "text",
                    "title": "Analysis Report",
                    "data": self.sample_markdown,
                    "metadata": {
                        "format": "markdown",
                        "word_count": 25,
                        "sections": 3
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.TEXT)
        self.assertEqual(artifact.title, "Analysis Report")
        self.assertIsInstance(artifact.data, str)
        self.assertIn("# Test Document", artifact.data)
        self.assertEqual(artifact.metadata["format"], "markdown")
    
    def test_multiple_artifacts_extraction(self):
        """다중 아티팩트 추출 테스트"""
        
        a2a_response = {
            "agent_id": "comprehensive_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "plotly_chart",
                    "title": "Chart 1",
                    "data": self.sample_plotly_data,
                    "metadata": {"chart_type": "line"}
                },
                {
                    "type": "dataframe",
                    "title": "Data 1",
                    "data": self.sample_dataframe_data,
                    "metadata": {"rows": 3}
                },
                {
                    "type": "text",
                    "title": "Report 1",
                    "data": self.sample_markdown,
                    "metadata": {"format": "markdown"}
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 3)
        
        # 타입 검증
        types = [artifact.type for artifact in artifacts]
        self.assertIn(ArtifactType.PLOTLY_CHART, types)
        self.assertIn(ArtifactType.DATAFRAME, types)
        self.assertIn(ArtifactType.TEXT, types)
        
        # 제목 검증
        titles = [artifact.title for artifact in artifacts]
        self.assertIn("Chart 1", titles)
        self.assertIn("Data 1", titles)
        self.assertIn("Report 1", titles)
    
    def test_invalid_artifact_type(self):
        """잘못된 아티팩트 타입 처리 테스트"""
        
        a2a_response = {
            "agent_id": "test_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "invalid_type",
                    "title": "Invalid Artifact",
                    "data": "some data",
                    "metadata": {}
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        # 잘못된 타입은 건너뛰어야 함
        self.assertEqual(len(artifacts), 0)
    
    def test_missing_required_fields(self):
        """필수 필드 누락 처리 테스트"""
        
        # 타입 누락
        a2a_response1 = {
            "agent_id": "test_agent",
            "status": "completed",
            "artifacts": [
                {
                    "title": "No Type",
                    "data": "some data",
                    "metadata": {}
                }
            ]
        }
        
        artifacts1 = self.extractor.extract_artifacts(a2a_response1)
        self.assertEqual(len(artifacts1), 0)
        
        # 데이터 누락
        a2a_response2 = {
            "agent_id": "test_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "text",
                    "title": "No Data",
                    "metadata": {}
                }
            ]
        }
        
        artifacts2 = self.extractor.extract_artifacts(a2a_response2)
        self.assertEqual(len(artifacts2), 0)
    
    def test_empty_response(self):
        """빈 응답 처리 테스트"""
        
        # 아티팩트 없음
        a2a_response1 = {
            "agent_id": "test_agent",
            "status": "completed",
            "artifacts": []
        }
        
        artifacts1 = self.extractor.extract_artifacts(a2a_response1)
        self.assertEqual(len(artifacts1), 0)
        
        # 아티팩트 키 없음
        a2a_response2 = {
            "agent_id": "test_agent",
            "status": "completed"
        }
        
        artifacts2 = self.extractor.extract_artifacts(a2a_response2)
        self.assertEqual(len(artifacts2), 0)
        
        # 빈 응답
        artifacts3 = self.extractor.extract_artifacts({})
        self.assertEqual(len(artifacts3), 0)
    
    def test_malformed_data(self):
        """잘못된 형식 데이터 처리 테스트"""
        
        # 잘못된 JSON 구조
        a2a_response = {
            "agent_id": "test_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "plotly_chart",
                    "title": "Malformed Chart",
                    "data": "invalid json structure",
                    "metadata": {}
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        # 잘못된 데이터는 건너뛰어야 함
        self.assertEqual(len(artifacts), 0)
    
    def test_metadata_extraction(self):
        """메타데이터 추출 테스트"""
        
        timestamp = datetime.now().isoformat()
        
        a2a_response = {
            "agent_id": "metadata_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "text",
                    "title": "Metadata Test",
                    "data": "Test content",
                    "metadata": {
                        "created_at": timestamp,
                        "author": "test_agent",
                        "version": "1.0",
                        "tags": ["test", "metadata"],
                        "custom_field": "custom_value"
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.metadata["created_at"], timestamp)
        self.assertEqual(artifact.metadata["author"], "test_agent")
        self.assertEqual(artifact.metadata["version"], "1.0")
        self.assertEqual(artifact.metadata["tags"], ["test", "metadata"])
        self.assertEqual(artifact.metadata["custom_field"], "custom_value")
    
    def test_large_data_handling(self):
        """대용량 데이터 처리 테스트"""
        
        # 큰 DataFrame 생성
        large_dataframe_data = {
            "columns": [f"col_{i}" for i in range(100)],
            "data": [[f"value_{i}_{j}" for j in range(100)] for i in range(1000)],
            "index": list(range(1000))
        }
        
        a2a_response = {
            "agent_id": "large_data_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "dataframe",
                    "title": "Large Dataset",
                    "data": large_dataframe_data,
                    "metadata": {
                        "rows": 1000,
                        "columns": 100,
                        "estimated_size": "800KB"
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.DATAFRAME)
        self.assertEqual(len(artifact.data["columns"]), 100)
        self.assertEqual(len(artifact.data["data"]), 1000)
        self.assertEqual(artifact.metadata["rows"], 1000)
        self.assertEqual(artifact.metadata["columns"], 100)
    
    def test_unicode_content(self):
        """유니코드 콘텐츠 처리 테스트"""
        
        unicode_text = """
# 한글 제목

## 다양한 언어 테스트
- 한국어: 안녕하세요
- 中文: 你好
- 日本語: こんにちは
- العربية: مرحبا
- русский: Привет
- Ελληνικά: Γεια σας

### 특수 문자
- 이모지: 😀 🎉 ✨
- 수학 기호: ∑ ∏ ∫ ≈ ≠
- 화폐: $ € ¥ ₩
"""
        
        a2a_response = {
            "agent_id": "unicode_agent",
            "status": "completed",
            "artifacts": [
                {
                    "type": "text",
                    "title": "Unicode Test",
                    "data": unicode_text,
                    "metadata": {
                        "encoding": "utf-8",
                        "languages": ["ko", "zh", "ja", "ar", "ru", "el"]
                    }
                }
            ]
        }
        
        artifacts = self.extractor.extract_artifacts(a2a_response)
        
        self.assertEqual(len(artifacts), 1)
        
        artifact = artifacts[0]
        self.assertEqual(artifact.type, ArtifactType.TEXT)
        self.assertIn("안녕하세요", artifact.data)
        self.assertIn("你好", artifact.data)
        self.assertIn("😀", artifact.data)
        self.assertEqual(artifact.metadata["encoding"], "utf-8")

class TestArtifactParsers(unittest.TestCase):
    """아티팩트 파서 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.plotly_parser = PlotlyArtifactParser()
        self.dataframe_parser = DataFrameArtifactParser()
        self.image_parser = ImageArtifactParser()
        self.code_parser = CodeArtifactParser()
        self.text_parser = TextArtifactParser()
    
    def test_plotly_parser_validation(self):
        """Plotly 파서 검증 테스트"""
        
        # 유효한 데이터
        valid_data = {
            "data": [{"x": [1, 2], "y": [1, 2], "type": "scatter"}],
            "layout": {"title": "Test"}
        }
        
        result = self.plotly_parser.parse(valid_data)
        self.assertTrue(result["is_valid"])
        self.assertIn("chart_info", result)
        
        # 무효한 데이터
        invalid_data = {"invalid": "structure"}
        
        result = self.plotly_parser.parse(invalid_data)
        self.assertFalse(result["is_valid"])
        self.assertIn("error", result)
    
    def test_dataframe_parser_validation(self):
        """DataFrame 파서 검증 테스트"""
        
        # 유효한 데이터
        valid_data = {
            "columns": ["A", "B"],
            "data": [[1, 2], [3, 4]],
            "index": [0, 1]
        }
        
        result = self.dataframe_parser.parse(valid_data)
        self.assertTrue(result["is_valid"])
        self.assertIn("shape", result)
        self.assertEqual(result["shape"], (2, 2))
        
        # 무효한 데이터
        invalid_data = {"columns": ["A"], "data": [[1, 2]]}  # 열 수 불일치
        
        result = self.dataframe_parser.parse(invalid_data)
        self.assertFalse(result["is_valid"])
    
    def test_image_parser_validation(self):
        """이미지 파서 검증 테스트"""
        
        # 유효한 Base64 이미지
        valid_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        result = self.image_parser.parse(valid_b64)
        self.assertTrue(result["is_valid"])
        self.assertIn("format", result)
        
        # 무효한 Base64
        invalid_b64 = "invalid_base64_string"
        
        result = self.image_parser.parse(invalid_b64)
        self.assertFalse(result["is_valid"])
    
    def test_code_parser_validation(self):
        """코드 파서 검증 테스트"""
        
        # 유효한 Python 코드
        valid_code = """
def test_function():
    return "Hello, World!"

if __name__ == "__main__":
    print(test_function())
"""
        
        result = self.code_parser.parse(valid_code)
        self.assertTrue(result["is_valid"])
        self.assertIn("language", result)
        self.assertIn("syntax_valid", result)
        
        # 무효한 구문
        invalid_code = "def invalid_syntax(:"
        
        result = self.code_parser.parse(invalid_code)
        self.assertFalse(result["syntax_valid"])
    
    def test_text_parser_validation(self):
        """텍스트 파서 검증 테스트"""
        
        # 마크다운 텍스트
        markdown_text = """
# Title
## Subtitle
- Item 1
- Item 2
"""
        
        result = self.text_parser.parse(markdown_text)
        self.assertTrue(result["is_valid"])
        self.assertIn("format", result)
        self.assertIn("word_count", result)
        self.assertIn("line_count", result)

if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)