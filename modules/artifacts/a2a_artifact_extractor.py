"""
A2A 아티팩트 추출 시스템

A2A 에이전트 응답에서 다양한 타입의 아티팩트를 정확히 추출하고 분류하는 시스템
"""

import json
import logging
import base64
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)

class ArtifactType(Enum):
    """지원되는 아티팩트 타입"""
    PLOTLY_CHART = "plotly_chart"
    DATAFRAME = "dataframe"
    IMAGE = "image"
    CODE = "code"
    TEXT = "text"
    UNKNOWN = "unknown"

@dataclass
class Artifact:
    """아티팩트 기본 데이터 모델"""
    id: str
    type: ArtifactType
    data: Any
    metadata: Dict[str, Any]
    agent_source: str
    timestamp: datetime
    download_formats: List[str]
    
    def __post_init__(self):
        if not self.download_formats:
            self.download_formats = self._get_default_download_formats()
    
    def _get_default_download_formats(self) -> List[str]:
        """아티팩트 타입별 기본 다운로드 형식"""
        format_map = {
            ArtifactType.PLOTLY_CHART: ["json", "png", "html"],
            ArtifactType.DATAFRAME: ["csv", "xlsx", "json"],
            ArtifactType.IMAGE: ["png", "jpg"],
            ArtifactType.CODE: ["py", "txt"],
            ArtifactType.TEXT: ["md", "txt"],
            ArtifactType.UNKNOWN: ["txt"]
        }
        return format_map.get(self.type, ["txt"])

@dataclass
class PlotlyArtifact(Artifact):
    """Plotly 차트 아티팩트"""
    plotly_json: Dict
    chart_type: str
    interactive_features: List[str]
    
    def __post_init__(self):
        super().__post_init__()
        self.type = ArtifactType.PLOTLY_CHART

@dataclass
class DataFrameArtifact(Artifact):
    """DataFrame 아티팩트"""
    dataframe: pd.DataFrame
    summary_stats: Dict
    column_info: List[Dict]
    
    def __post_init__(self):
        super().__post_init__()
        self.type = ArtifactType.DATAFRAME

@dataclass
class ImageArtifact(Artifact):
    """이미지 아티팩트"""
    image_data: bytes
    format: str
    dimensions: tuple
    
    def __post_init__(self):
        super().__post_init__()
        self.type = ArtifactType.IMAGE

@dataclass
class CodeArtifact(Artifact):
    """코드 아티팩트"""
    code: str
    language: str
    executable: bool
    
    def __post_init__(self):
        super().__post_init__()
        self.type = ArtifactType.CODE

@dataclass
class TextArtifact(Artifact):
    """텍스트 아티팩트"""
    text: str
    format: str  # markdown, html, plain
    
    def __post_init__(self):
        super().__post_init__()
        self.type = ArtifactType.TEXT

class A2AArtifactExtractor:
    """A2A 응답에서 아티팩트를 정확히 추출하는 시스템"""
    
    def __init__(self):
        self.supported_types = {
            ArtifactType.PLOTLY_CHART: self._parse_plotly_chart,
            ArtifactType.DATAFRAME: self._parse_dataframe,
            ArtifactType.IMAGE: self._parse_image,
            ArtifactType.CODE: self._parse_code,
            ArtifactType.TEXT: self._parse_text
        }
        
    async def extract_from_a2a_response(self, response: Dict, agent_source: str = "unknown") -> List[Artifact]:
        """A2A 응답에서 모든 아티팩트 추출"""
        artifacts = []
        
        try:
            # A2A 응답 구조 분석
            if not isinstance(response, dict):
                logger.warning(f"Invalid A2A response format: {type(response)}")
                return artifacts
            
            # 다양한 A2A 응답 구조 처리
            artifacts.extend(await self._extract_from_message_parts(response, agent_source))
            artifacts.extend(await self._extract_from_artifacts_field(response, agent_source))
            artifacts.extend(await self._extract_from_content_field(response, agent_source))
            
            logger.info(f"Extracted {len(artifacts)} artifacts from {agent_source}")
            
        except Exception as e:
            logger.error(f"Error extracting artifacts from A2A response: {str(e)}")
            
        return artifacts
    
    async def _extract_from_message_parts(self, response: Dict, agent_source: str) -> List[Artifact]:
        """message.parts 구조에서 아티팩트 추출"""
        artifacts = []
        
        try:
            # A2A SDK 표준 구조: message.parts[]
            if "message" in response and "parts" in response["message"]:
                parts = response["message"]["parts"]
                for i, part in enumerate(parts):
                    if isinstance(part, dict):
                        artifact = await self._extract_from_part(part, agent_source, f"part_{i}")
                        if artifact:
                            artifacts.append(artifact)
                            
        except Exception as e:
            logger.error(f"Error extracting from message parts: {str(e)}")
            
        return artifacts
    
    async def _extract_from_artifacts_field(self, response: Dict, agent_source: str) -> List[Artifact]:
        """artifacts 필드에서 직접 추출"""
        artifacts = []
        
        try:
            if "artifacts" in response:
                artifacts_data = response["artifacts"]
                if isinstance(artifacts_data, list):
                    for i, artifact_data in enumerate(artifacts_data):
                        artifact = await self._extract_from_data(artifact_data, agent_source, f"artifact_{i}")
                        if artifact:
                            artifacts.append(artifact)
                            
        except Exception as e:
            logger.error(f"Error extracting from artifacts field: {str(e)}")
            
        return artifacts
    
    async def _extract_from_content_field(self, response: Dict, agent_source: str) -> List[Artifact]:
        """content 필드에서 추출"""
        artifacts = []
        
        try:
            if "content" in response:
                content = response["content"]
                artifact = await self._extract_from_data(content, agent_source, "content")
                if artifact:
                    artifacts.append(artifact)
                    
        except Exception as e:
            logger.error(f"Error extracting from content field: {str(e)}")
            
        return artifacts
    
    async def _extract_from_part(self, part: Dict, agent_source: str, part_id: str) -> Optional[Artifact]:
        """개별 part에서 아티팩트 추출"""
        try:
            # part 구조 분석
            if "root" in part:
                return await self._extract_from_data(part["root"], agent_source, part_id)
            elif "text" in part:
                return await self._extract_from_data(part, agent_source, part_id)
            else:
                return await self._extract_from_data(part, agent_source, part_id)
                
        except Exception as e:
            logger.error(f"Error extracting from part {part_id}: {str(e)}")
            return None
    
    async def _extract_from_data(self, data: Any, agent_source: str, data_id: str) -> Optional[Artifact]:
        """데이터에서 아티팩트 추출"""
        try:
            # 아티팩트 타입 감지
            artifact_type = self.detect_artifact_type(data)
            
            if artifact_type == ArtifactType.UNKNOWN:
                return None
            
            # 타입별 파싱
            parser = self.supported_types.get(artifact_type)
            if parser:
                return await parser(data, agent_source, data_id)
            
        except Exception as e:
            logger.error(f"Error extracting from data {data_id}: {str(e)}")
            
        return None
    
    def detect_artifact_type(self, data: Any) -> ArtifactType:
        """데이터 구조 분석하여 아티팩트 타입 감지"""
        try:
            if isinstance(data, dict):
                # Plotly 차트 감지
                if self._is_plotly_chart(data):
                    return ArtifactType.PLOTLY_CHART
                
                # DataFrame 감지 (dict 형태)
                if self._is_dataframe_dict(data):
                    return ArtifactType.DATAFRAME
                
                # 이미지 감지
                if self._is_image_data(data):
                    return ArtifactType.IMAGE
                
                # 코드 감지
                if self._is_code_data(data):
                    return ArtifactType.CODE
                
                # 텍스트 감지
                if self._is_text_data(data):
                    return ArtifactType.TEXT
            
            elif isinstance(data, pd.DataFrame):
                return ArtifactType.DATAFRAME
            
            elif isinstance(data, str):
                # 문자열에서 타입 추론
                if self._is_code_string(data):
                    return ArtifactType.CODE
                elif self._is_markdown_string(data):
                    return ArtifactType.TEXT
                elif len(data) > 100:  # 긴 텍스트
                    return ArtifactType.TEXT
            
        except Exception as e:
            logger.error(f"Error detecting artifact type: {str(e)}")
        
        return ArtifactType.UNKNOWN
    
    def _is_plotly_chart(self, data: Dict) -> bool:
        """Plotly 차트 데이터인지 확인"""
        plotly_indicators = ["data", "layout", "config", "frames"]
        return any(key in data for key in plotly_indicators) and isinstance(data.get("data"), list)
    
    def _is_dataframe_dict(self, data: Dict) -> bool:
        """DataFrame 딕셔너리인지 확인"""
        df_indicators = ["columns", "index", "data"]
        return any(key in data for key in df_indicators)
    
    def _is_image_data(self, data: Dict) -> bool:
        """이미지 데이터인지 확인"""
        image_indicators = ["image", "base64", "png", "jpg", "jpeg"]
        return any(key in data for key in image_indicators)
    
    def _is_code_data(self, data: Dict) -> bool:
        """코드 데이터인지 확인"""
        code_indicators = ["code", "language", "script", "python", "sql"]
        return any(key in data for key in code_indicators)
    
    def _is_text_data(self, data: Dict) -> bool:
        """텍스트 데이터인지 확인"""
        text_indicators = ["text", "content", "markdown", "html"]
        return any(key in data for key in text_indicators)
    
    def _is_code_string(self, text: str) -> bool:
        """문자열이 코드인지 확인"""
        code_patterns = ["def ", "import ", "from ", "class ", "SELECT ", "CREATE "]
        return any(pattern in text for pattern in code_patterns)
    
    def _is_markdown_string(self, text: str) -> bool:
        """문자열이 마크다운인지 확인"""
        markdown_patterns = ["# ", "## ", "- ", "* ", "```", "**"]
        return any(pattern in text for pattern in markdown_patterns)
    
    async def _parse_plotly_chart(self, data: Any, agent_source: str, data_id: str) -> PlotlyArtifact:
        """Plotly 차트 데이터 파싱"""
        try:
            if isinstance(data, dict):
                plotly_json = data
            else:
                plotly_json = json.loads(str(data))
            
            # 차트 타입 추론
            chart_type = self._infer_chart_type(plotly_json)
            
            # 인터랙티브 기능 감지
            interactive_features = self._detect_interactive_features(plotly_json)
            
            return PlotlyArtifact(
                id=f"{agent_source}_{data_id}_{datetime.now().timestamp()}",
                data=plotly_json,
                metadata={
                    "chart_type": chart_type,
                    "data_points": len(plotly_json.get("data", [])),
                    "has_layout": "layout" in plotly_json
                },
                agent_source=agent_source,
                timestamp=datetime.now(),
                download_formats=["json", "png", "html", "svg"],
                plotly_json=plotly_json,
                chart_type=chart_type,
                interactive_features=interactive_features
            )
            
        except Exception as e:
            logger.error(f"Error parsing Plotly chart: {str(e)}")
            raise
    
    def _infer_chart_type(self, plotly_json: Dict) -> str:
        """Plotly JSON에서 차트 타입 추론"""
        try:
            if "data" in plotly_json and plotly_json["data"]:
                first_trace = plotly_json["data"][0]
                return first_trace.get("type", "scatter")
        except:
            pass
        return "unknown"
    
    def _detect_interactive_features(self, plotly_json: Dict) -> List[str]:
        """Plotly 차트의 인터랙티브 기능 감지"""
        features = []
        
        try:
            # 기본 인터랙티브 기능
            features.extend(["zoom", "pan", "hover"])
            
            # 레이아웃 기반 기능
            layout = plotly_json.get("layout", {})
            if layout.get("showlegend", True):
                features.append("legend_toggle")
            
            # 데이터 기반 기능
            data = plotly_json.get("data", [])
            for trace in data:
                if trace.get("hovertemplate") or trace.get("hoverinfo"):
                    features.append("custom_hover")
                    break
                    
        except Exception as e:
            logger.error(f"Error detecting interactive features: {str(e)}")
        
        return features
    
    async def _parse_dataframe(self, data: Any, agent_source: str, data_id: str) -> DataFrameArtifact:
        """DataFrame 데이터 파싱"""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                # 문자열이나 다른 형태를 DataFrame으로 변환 시도
                df = pd.read_json(str(data))
            
            # 요약 통계 생성
            summary_stats = self._generate_summary_stats(df)
            
            # 컬럼 정보 생성
            column_info = self._generate_column_info(df)
            
            return DataFrameArtifact(
                id=f"{agent_source}_{data_id}_{datetime.now().timestamp()}",
                data=df,
                metadata={
                    "shape": df.shape,
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "dtypes": df.dtypes.to_dict()
                },
                agent_source=agent_source,
                timestamp=datetime.now(),
                download_formats=["csv", "xlsx", "json", "parquet"],
                dataframe=df,
                summary_stats=summary_stats,
                column_info=column_info
            )
            
        except Exception as e:
            logger.error(f"Error parsing DataFrame: {str(e)}")
            raise
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """DataFrame 요약 통계 생성"""
        try:
            stats = {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "null_counts": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
            # 수치형 컬럼 통계
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating summary stats: {str(e)}")
            return {}
    
    def _generate_column_info(self, df: pd.DataFrame) -> List[Dict]:
        """DataFrame 컬럼 정보 생성"""
        try:
            column_info = []
            
            for col in df.columns:
                info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "null_count": df[col].isnull().sum(),
                    "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                    "unique_count": df[col].nunique()
                }
                
                # 수치형 컬럼 추가 정보
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    info.update({
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "std": df[col].std()
                    })
                
                column_info.append(info)
            
            return column_info
            
        except Exception as e:
            logger.error(f"Error generating column info: {str(e)}")
            return []
    
    async def _parse_image(self, data: Any, agent_source: str, data_id: str) -> ImageArtifact:
        """이미지 데이터 파싱"""
        try:
            image_data = None
            format = "png"
            dimensions = (0, 0)
            
            if isinstance(data, dict):
                # Base64 이미지 데이터 처리
                if "base64" in data:
                    image_data = base64.b64decode(data["base64"])
                elif "image" in data:
                    image_data = base64.b64decode(data["image"])
                
                format = data.get("format", "png")
                dimensions = data.get("dimensions", (0, 0))
            
            elif isinstance(data, str):
                # Base64 문자열 직접 처리
                image_data = base64.b64decode(data)
            
            elif isinstance(data, bytes):
                image_data = data
            
            return ImageArtifact(
                id=f"{agent_source}_{data_id}_{datetime.now().timestamp()}",
                data=image_data,
                metadata={
                    "format": format,
                    "size_bytes": len(image_data) if image_data else 0,
                    "dimensions": dimensions
                },
                agent_source=agent_source,
                timestamp=datetime.now(),
                download_formats=["png", "jpg", "svg"],
                image_data=image_data,
                format=format,
                dimensions=dimensions
            )
            
        except Exception as e:
            logger.error(f"Error parsing image: {str(e)}")
            raise
    
    async def _parse_code(self, data: Any, agent_source: str, data_id: str) -> CodeArtifact:
        """코드 데이터 파싱"""
        try:
            code = ""
            language = "python"
            executable = False
            
            if isinstance(data, dict):
                code = data.get("code", data.get("script", ""))
                language = data.get("language", "python")
                executable = data.get("executable", False)
            elif isinstance(data, str):
                code = data
                language = self._detect_code_language(code)
                executable = self._is_executable_code(code)
            
            return CodeArtifact(
                id=f"{agent_source}_{data_id}_{datetime.now().timestamp()}",
                data=code,
                metadata={
                    "language": language,
                    "line_count": len(code.split('\n')),
                    "char_count": len(code),
                    "executable": executable
                },
                agent_source=agent_source,
                timestamp=datetime.now(),
                download_formats=["py", "txt", "ipynb"],
                code=code,
                language=language,
                executable=executable
            )
            
        except Exception as e:
            logger.error(f"Error parsing code: {str(e)}")
            raise
    
    def _detect_code_language(self, code: str) -> str:
        """코드 언어 감지"""
        if "SELECT" in code.upper() or "CREATE" in code.upper():
            return "sql"
        elif "import " in code or "def " in code:
            return "python"
        elif "library(" in code or "<-" in code:
            return "r"
        elif "{" in code and "}" in code:
            return "json"
        else:
            return "text"
    
    def _is_executable_code(self, code: str) -> bool:
        """실행 가능한 코드인지 확인"""
        executable_patterns = ["def ", "import ", "from ", "class "]
        return any(pattern in code for pattern in executable_patterns)
    
    async def _parse_text(self, data: Any, agent_source: str, data_id: str) -> TextArtifact:
        """텍스트 데이터 파싱"""
        try:
            text = ""
            format = "plain"
            
            if isinstance(data, dict):
                text = data.get("text", data.get("content", str(data)))
                format = data.get("format", "markdown" if self._is_markdown_string(text) else "plain")
            elif isinstance(data, str):
                text = data
                format = "markdown" if self._is_markdown_string(text) else "plain"
            else:
                text = str(data)
            
            return TextArtifact(
                id=f"{agent_source}_{data_id}_{datetime.now().timestamp()}",
                data=text,
                metadata={
                    "format": format,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "line_count": len(text.split('\n'))
                },
                agent_source=agent_source,
                timestamp=datetime.now(),
                download_formats=["md", "txt", "html"],
                text=text,
                format=format
            )
            
        except Exception as e:
            logger.error(f"Error parsing text: {str(e)}")
            raise
    
    def validate_artifact_data(self, artifact: Artifact) -> bool:
        """아티팩트 데이터 유효성 검증"""
        try:
            if not artifact or not artifact.data:
                return False
            
            # 타입별 검증
            if artifact.type == ArtifactType.PLOTLY_CHART:
                return isinstance(artifact.data, dict) and "data" in artifact.data
            elif artifact.type == ArtifactType.DATAFRAME:
                return isinstance(artifact.data, pd.DataFrame) and not artifact.data.empty
            elif artifact.type == ArtifactType.IMAGE:
                return isinstance(artifact.data, bytes) and len(artifact.data) > 0
            elif artifact.type == ArtifactType.CODE:
                return isinstance(artifact.data, str) and len(artifact.data.strip()) > 0
            elif artifact.type == ArtifactType.TEXT:
                return isinstance(artifact.data, str) and len(artifact.data.strip()) > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating artifact: {str(e)}")
            return False