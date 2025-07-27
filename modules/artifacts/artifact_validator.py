"""
아티팩트 유효성 검증 시스템

A2A 에이전트에서 생성된 아티팩트의 데이터 무결성을 검사하고
지원되지 않는 형식을 감지하여 적절한 에러 처리 및 폴백 메커니즘을 제공
"""

import logging
import json
import base64
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from PIL import Image
import io

from .a2a_artifact_extractor import Artifact, ArtifactType

logger = logging.getLogger(__name__)

class ValidationResult:
    """검증 결과 클래스"""
    
    def __init__(self, is_valid: bool, error_message: str = "", 
                 warnings: List[str] = None, suggestions: List[str] = None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.warnings = warnings or []
        self.suggestions = suggestions or []
        self.validation_score = 1.0 if is_valid else 0.0
    
    def add_warning(self, warning: str):
        """경고 메시지 추가"""
        self.warnings.append(warning)
        self.validation_score = max(0.5, self.validation_score - 0.1)
    
    def add_suggestion(self, suggestion: str):
        """개선 제안 추가"""
        self.suggestions.append(suggestion)

class ArtifactValidator:
    """아티팩트 유효성 검증 시스템"""
    
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_dataframe_rows = 100000
        self.max_image_dimensions = (4096, 4096)
        self.supported_image_formats = ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'TIFF']
        
    async def validate_artifact(self, artifact: Artifact) -> ValidationResult:
        """아티팩트 종합 유효성 검증"""
        try:
            # 기본 구조 검증
            basic_result = self._validate_basic_structure(artifact)
            if not basic_result.is_valid:
                return basic_result
            
            # 타입별 상세 검증
            if artifact.type == ArtifactType.PLOTLY_CHART:
                return await self._validate_plotly_chart(artifact)
            elif artifact.type == ArtifactType.DATAFRAME:
                return await self._validate_dataframe(artifact)
            elif artifact.type == ArtifactType.IMAGE:
                return await self._validate_image(artifact)
            elif artifact.type == ArtifactType.CODE:
                return await self._validate_code(artifact)
            elif artifact.type == ArtifactType.TEXT:
                return await self._validate_text(artifact)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported artifact type: {artifact.type}"
                )
                
        except Exception as e:
            logger.error(f"Error validating artifact: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )    

    def _validate_basic_structure(self, artifact: Artifact) -> ValidationResult:
        """기본 구조 검증"""
        try:
            # 필수 필드 확인
            if not artifact.id:
                return ValidationResult(False, "Artifact ID is missing")
            
            if not artifact.type:
                return ValidationResult(False, "Artifact type is missing")
            
            if artifact.data is None:
                return ValidationResult(False, "Artifact data is missing")
            
            if not artifact.agent_source:
                return ValidationResult(False, "Agent source is missing")
            
            # 메타데이터 검증
            if not isinstance(artifact.metadata, dict):
                return ValidationResult(False, "Invalid metadata format")
            
            return ValidationResult(True)
            
        except Exception as e:
            return ValidationResult(False, f"Basic structure validation error: {str(e)}")
    
    async def _validate_plotly_chart(self, artifact: Artifact) -> ValidationResult:
        """Plotly 차트 검증"""
        try:
            result = ValidationResult(True)
            plotly_data = artifact.data
            
            # JSON 구조 검증
            if not isinstance(plotly_data, dict):
                return ValidationResult(False, "Plotly data must be a dictionary")
            
            # 필수 필드 확인
            if "data" not in plotly_data:
                return ValidationResult(False, "Plotly chart missing 'data' field")
            
            if not isinstance(plotly_data["data"], list):
                return ValidationResult(False, "Plotly 'data' field must be a list")
            
            if len(plotly_data["data"]) == 0:
                return ValidationResult(False, "Plotly chart has no data traces")
            
            # 데이터 크기 검증
            total_points = 0
            for trace in plotly_data["data"]:
                if isinstance(trace, dict):
                    if "x" in trace and isinstance(trace["x"], list):
                        total_points += len(trace["x"])
                    elif "values" in trace and isinstance(trace["values"], list):
                        total_points += len(trace["values"])
            
            if total_points > 50000:
                result.add_warning(f"Large dataset with {total_points} points may affect performance")
                result.add_suggestion("Consider data sampling for better performance")
            
            return result
            
        except Exception as e:
            return ValidationResult(False, f"Plotly validation error: {str(e)}")
    
    async def _validate_dataframe(self, artifact: Artifact) -> ValidationResult:
        """DataFrame 검증"""
        try:
            result = ValidationResult(True)
            df = artifact.data
            
            # DataFrame 타입 확인
            if not isinstance(df, pd.DataFrame):
                return ValidationResult(False, "Data is not a valid DataFrame")
            
            # 빈 DataFrame 확인
            if df.empty:
                return ValidationResult(False, "DataFrame is empty")
            
            # 크기 검증
            rows, cols = df.shape
            if rows > self.max_dataframe_rows:
                result.add_warning(f"Large DataFrame with {rows} rows may affect performance")
                result.add_suggestion("Consider data sampling or pagination")
            
            # 데이터 품질 검증
            null_percentage = (df.isnull().sum().sum() / (rows * cols)) * 100
            if null_percentage > 50:
                result.add_warning(f"High null percentage: {null_percentage:.1f}%")
                result.add_suggestion("Consider data cleaning or imputation")
            
            return result
            
        except Exception as e:
            return ValidationResult(False, f"DataFrame validation error: {str(e)}")
    
    async def _validate_image(self, artifact: Artifact) -> ValidationResult:
        """이미지 검증"""
        try:
            result = ValidationResult(True)
            image_data = artifact.data
            
            # 바이트 데이터 확인
            if not isinstance(image_data, bytes):
                return ValidationResult(False, "Image data must be bytes")
            
            if len(image_data) == 0:
                return ValidationResult(False, "Empty image data")
            
            # PIL로 이미지 유효성 확인
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    # 이미지 형식 확인
                    if img.format not in self.supported_image_formats:
                        result.add_warning(f"Uncommon image format: {img.format}")
                    
                    # 이미지 크기 확인
                    width, height = img.size
                    if width > self.max_image_dimensions[0] or height > self.max_image_dimensions[1]:
                        result.add_warning(f"Large image dimensions: {width}x{height}")
                        result.add_suggestion("Consider resizing for better performance")
                    
            except Exception as e:
                return ValidationResult(False, f"Invalid image data: {str(e)}")
            
            return result
            
        except Exception as e:
            return ValidationResult(False, f"Image validation error: {str(e)}")
    
    async def _validate_code(self, artifact: Artifact) -> ValidationResult:
        """코드 검증"""
        try:
            result = ValidationResult(True)
            code = artifact.data
            
            # 문자열 확인
            if not isinstance(code, str):
                return ValidationResult(False, "Code data must be a string")
            
            if len(code.strip()) == 0:
                return ValidationResult(False, "Empty code content")
            
            # 코드 길이 검증
            line_count = len(code.split('\n'))
            if line_count > 1000:
                result.add_warning(f"Very long code: {line_count} lines")
                result.add_suggestion("Consider breaking into smaller modules")
            
            # 보안 검증
            security_result = self._check_code_security(code)
            if security_result:
                result.add_warning(security_result)
                result.add_suggestion("Review code for security implications")
            
            return result
            
        except Exception as e:
            return ValidationResult(False, f"Code validation error: {str(e)}")
    
    def _check_code_security(self, code: str) -> Optional[str]:
        """코드 보안 검사"""
        try:
            # 위험한 패턴 확인
            dangerous_patterns = [
                'eval(', 'exec(', 'os.system(', 'subprocess.call(',
                'rm -rf', 'del /f', '__import__'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    return f"Potentially dangerous pattern found: {pattern}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking code security: {str(e)}")
            return None
    
    async def _validate_text(self, artifact: Artifact) -> ValidationResult:
        """텍스트 검증"""
        try:
            result = ValidationResult(True)
            text = artifact.data
            
            # 문자열 확인
            if not isinstance(text, str):
                return ValidationResult(False, "Text data must be a string")
            
            if len(text.strip()) == 0:
                return ValidationResult(False, "Empty text content")
            
            # 텍스트 길이 검증
            word_count = len(text.split())
            if word_count > 10000:
                result.add_warning(f"Very long text: {word_count} words")
                result.add_suggestion("Consider text summarization")
            
            return result
            
        except Exception as e:
            return ValidationResult(False, f"Text validation error: {str(e)}")
    
    def get_fallback_artifact(self, original_artifact: Artifact, error_message: str) -> Artifact:
        """검증 실패 시 폴백 아티팩트 생성"""
        try:
            from datetime import datetime
            
            fallback_data = {
                "error": "Artifact validation failed",
                "original_type": original_artifact.type.value if original_artifact.type else "unknown",
                "error_message": error_message,
                "fallback_content": str(original_artifact.data)[:500] + "..." if original_artifact.data else "No data"
            }
            
            return Artifact(
                id=f"fallback_{original_artifact.id}",
                type=ArtifactType.TEXT,
                data=json.dumps(fallback_data, indent=2),
                metadata={
                    "is_fallback": True,
                    "original_type": original_artifact.type.value if original_artifact.type else "unknown",
                    "error_message": error_message
                },
                agent_source=original_artifact.agent_source,
                timestamp=datetime.now(),
                download_formats=["json", "txt"]
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback artifact: {str(e)}")
            # 최소한의 폴백
            from datetime import datetime
            return Artifact(
                id="emergency_fallback",
                type=ArtifactType.TEXT,
                data="Artifact processing failed",
                metadata={"is_emergency_fallback": True},
                agent_source="system",
                timestamp=datetime.now(),
                download_formats=["txt"]
            )