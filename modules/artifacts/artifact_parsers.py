"""
아티팩트 타입별 전문 파서 시스템

각 아티팩트 타입에 특화된 파싱 로직을 제공하여 
A2A 에이전트 응답에서 정확한 데이터 추출을 보장
"""

import json
import logging
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.utils

logger = logging.getLogger(__name__)

class BaseArtifactParser(ABC):
    """아티팩트 파서 기본 클래스"""
    
    @abstractmethod
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """아티팩트 데이터 파싱"""
        pass
    
    @abstractmethod
    def validate(self, parsed_data: Dict) -> bool:
        """파싱된 데이터 유효성 검증"""
        pass

class PlotlyArtifactParser(BaseArtifactParser):
    """Plotly 차트 전문 파서"""
    
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """Plotly JSON 데이터 파싱 및 최적화"""
        try:
            # JSON 데이터 정규화
            if isinstance(data, str):
                plotly_json = json.loads(data)
            elif isinstance(data, dict):
                plotly_json = data
            else:
                raise ValueError(f"Unsupported Plotly data type: {type(data)}")
            
            # Plotly 구조 검증 및 보완
            plotly_json = self._normalize_plotly_structure(plotly_json)
            
            # 차트 메타데이터 추출
            chart_metadata = self._extract_chart_metadata(plotly_json)
            
            # 성능 최적화
            optimized_json = self._optimize_for_rendering(plotly_json)
            
            return {
                "plotly_json": optimized_json,
                "original_json": plotly_json,
                "chart_metadata": chart_metadata,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing Plotly data: {str(e)}")
            return {
                "plotly_json": {},
                "error": str(e),
                "is_valid": False
            }
    
    def _normalize_plotly_structure(self, plotly_json: Dict) -> Dict:
        """Plotly JSON 구조 정규화"""
        normalized = plotly_json.copy()
        
        # 필수 필드 보장
        if "data" not in normalized:
            normalized["data"] = []
        if "layout" not in normalized:
            normalized["layout"] = {}
        if "config" not in normalized:
            normalized["config"] = {"responsive": True}
        
        # 데이터 타입 정규화
        if not isinstance(normalized["data"], list):
            normalized["data"] = [normalized["data"]]
        
        return normalized
    
    def _extract_chart_metadata(self, plotly_json: Dict) -> Dict:
        """차트 메타데이터 추출"""
        metadata = {
            "chart_type": "unknown",
            "trace_count": 0,
            "data_points": 0,
            "has_animation": False,
            "interactive_features": [],
            "dimensions": "2d"
        }
        
        try:
            data = plotly_json.get("data", [])
            metadata["trace_count"] = len(data)
            
            if data:
                first_trace = data[0]
                metadata["chart_type"] = first_trace.get("type", "scatter")
                
                # 데이터 포인트 수 계산
                if "x" in first_trace:
                    metadata["data_points"] = len(first_trace["x"])
                elif "values" in first_trace:
                    metadata["data_points"] = len(first_trace["values"])
                
                # 3D 차트 감지
                if first_trace.get("type") in ["scatter3d", "surface", "mesh3d"]:
                    metadata["dimensions"] = "3d"
            
            # 인터랙티브 기능 감지
            layout = plotly_json.get("layout", {})
            if layout.get("showlegend", True):
                metadata["interactive_features"].append("legend")
            if "updatemenus" in layout:
                metadata["interactive_features"].append("controls")
            if "sliders" in layout:
                metadata["interactive_features"].append("sliders")
            
        except Exception as e:
            logger.error(f"Error extracting chart metadata: {str(e)}")
        
        return metadata
    
    def _optimize_for_rendering(self, plotly_json: Dict) -> Dict:
        """렌더링 성능 최적화"""
        optimized = plotly_json.copy()
        
        try:
            # 대용량 데이터 샘플링
            data = optimized.get("data", [])
            for trace in data:
                if "x" in trace and len(trace["x"]) > 10000:
                    # 10,000개 이상의 데이터 포인트는 샘플링
                    sample_size = 5000
                    indices = np.linspace(0, len(trace["x"]) - 1, sample_size, dtype=int)
                    
                    trace["x"] = [trace["x"][i] for i in indices]
                    if "y" in trace:
                        trace["y"] = [trace["y"][i] for i in indices]
                    if "z" in trace:
                        trace["z"] = [trace["z"][i] for i in indices]
            
            # 레이아웃 최적화
            layout = optimized.get("layout", {})
            layout.update({
                "autosize": True,
                "responsive": True,
                "showlegend": layout.get("showlegend", True)
            })
            
        except Exception as e:
            logger.error(f"Error optimizing Plotly chart: {str(e)}")
        
        return optimized
    
    def validate(self, parsed_data: Dict) -> bool:
        """Plotly 데이터 유효성 검증"""
        try:
            if not parsed_data.get("is_valid", False):
                return False
            
            plotly_json = parsed_data.get("plotly_json", {})
            
            # 필수 구조 확인
            if not isinstance(plotly_json.get("data"), list):
                return False
            
            # 데이터 유효성 확인
            data = plotly_json["data"]
            if not data:
                return False
            
            # 첫 번째 trace 검증
            first_trace = data[0]
            if not isinstance(first_trace, dict):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating Plotly data: {str(e)}")
            return Falseclass D
ataFrameArtifactParser(BaseArtifactParser):
    """DataFrame 전문 파서"""
    
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """DataFrame 데이터 파싱 및 최적화"""
        try:
            # DataFrame 생성
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, dict):
                df = self._dict_to_dataframe(data)
            elif isinstance(data, str):
                df = self._string_to_dataframe(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported DataFrame data type: {type(data)}")
            
            # 데이터 최적화
            optimized_df = self._optimize_dataframe(df)
            
            # 메타데이터 생성
            df_metadata = self._generate_dataframe_metadata(optimized_df)
            
            # 미리보기 데이터 생성
            preview_data = self._generate_preview_data(optimized_df)
            
            return {
                "dataframe": optimized_df,
                "original_dataframe": df,
                "metadata": df_metadata,
                "preview": preview_data,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing DataFrame: {str(e)}")
            return {
                "dataframe": pd.DataFrame(),
                "error": str(e),
                "is_valid": False
            }
    
    def _dict_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """딕셔너리를 DataFrame으로 변환"""
        try:
            # 다양한 딕셔너리 구조 처리
            if "data" in data and "columns" in data:
                # {data: [[...]], columns: [...]} 형태
                return pd.DataFrame(data["data"], columns=data["columns"])
            elif "index" in data and "columns" in data and "data" in data:
                # pandas to_dict('split') 형태
                return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
            elif all(isinstance(v, list) for v in data.values()):
                # {col1: [val1, val2], col2: [val3, val4]} 형태
                return pd.DataFrame(data)
            else:
                # 일반 딕셔너리를 단일 행 DataFrame으로
                return pd.DataFrame([data])
                
        except Exception as e:
            logger.error(f"Error converting dict to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _string_to_dataframe(self, data: str) -> pd.DataFrame:
        """문자열을 DataFrame으로 변환"""
        try:
            # JSON 문자열 시도
            try:
                json_data = json.loads(data)
                return self._dict_to_dataframe(json_data)
            except:
                pass
            
            # CSV 문자열 시도
            try:
                return pd.read_csv(io.StringIO(data))
            except:
                pass
            
            # 탭 구분 시도
            try:
                return pd.read_csv(io.StringIO(data), sep='\t')
            except:
                pass
            
            # 실패 시 빈 DataFrame
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error converting string to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 메모리 및 성능 최적화"""
        try:
            optimized_df = df.copy()
            
            # 데이터 타입 최적화
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    # 문자열 컬럼 최적화
                    try:
                        optimized_df[col] = optimized_df[col].astype('category')
                    except:
                        pass
                elif optimized_df[col].dtype in ['int64', 'float64']:
                    # 수치형 컬럼 최적화
                    try:
                        if optimized_df[col].dtype == 'int64':
                            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                        else:
                            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
                    except:
                        pass
            
            # 대용량 데이터 샘플링 (10,000행 이상)
            if len(optimized_df) > 10000:
                sample_size = 5000
                optimized_df = optimized_df.sample(n=sample_size, random_state=42)
                logger.info(f"DataFrame sampled from {len(df)} to {len(optimized_df)} rows")
            
            return optimized_df
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {str(e)}")
            return df
    
    def _generate_dataframe_metadata(self, df: pd.DataFrame) -> Dict:
        """DataFrame 메타데이터 생성"""
        try:
            metadata = {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "column_info": []
            }
            
            # 컬럼별 상세 정보
            for col in df.columns:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique()),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                
                # 수치형 컬럼 통계
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    col_info.update({
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
                    })
                
                metadata["column_info"].append(col_info)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating DataFrame metadata: {str(e)}")
            return {}
    
    def _generate_preview_data(self, df: pd.DataFrame) -> Dict:
        """DataFrame 미리보기 데이터 생성"""
        try:
            preview = {
                "head": df.head(10).to_dict('records'),
                "tail": df.tail(5).to_dict('records'),
                "sample": df.sample(min(5, len(df)), random_state=42).to_dict('records') if len(df) > 0 else [],
                "columns": df.columns.tolist(),
                "index_name": df.index.name,
                "shape": df.shape
            }
            
            return preview
            
        except Exception as e:
            logger.error(f"Error generating DataFrame preview: {str(e)}")
            return {}
    
    def validate(self, parsed_data: Dict) -> bool:
        """DataFrame 데이터 유효성 검증"""
        try:
            if not parsed_data.get("is_valid", False):
                return False
            
            df = parsed_data.get("dataframe")
            if not isinstance(df, pd.DataFrame):
                return False
            
            if df.empty:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return False

class ImageArtifactParser(BaseArtifactParser):
    """이미지 전문 파서"""
    
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """이미지 데이터 파싱 및 최적화"""
        try:
            image_data = None
            format_type = "png"
            
            # 다양한 이미지 데이터 형태 처리
            if isinstance(data, dict):
                image_data = self._extract_image_from_dict(data)
                format_type = data.get("format", "png")
            elif isinstance(data, str):
                image_data = self._decode_base64_image(data)
            elif isinstance(data, bytes):
                image_data = data
            else:
                raise ValueError(f"Unsupported image data type: {type(data)}")
            
            if not image_data:
                raise ValueError("No valid image data found")
            
            # 이미지 메타데이터 추출
            image_metadata = self._extract_image_metadata(image_data, format_type)
            
            # 이미지 최적화
            optimized_data = self._optimize_image(image_data, format_type)
            
            return {
                "image_data": optimized_data,
                "original_data": image_data,
                "format": format_type,
                "metadata": image_metadata,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing image: {str(e)}")
            return {
                "image_data": b"",
                "error": str(e),
                "is_valid": False
            }
    
    def _extract_image_from_dict(self, data: Dict) -> Optional[bytes]:
        """딕셔너리에서 이미지 데이터 추출"""
        try:
            # 다양한 키 패턴 시도
            for key in ["image", "base64", "data", "content", "png", "jpg"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        return self._decode_base64_image(value)
                    elif isinstance(value, bytes):
                        return value
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image from dict: {str(e)}")
            return None
    
    def _decode_base64_image(self, base64_str: str) -> Optional[bytes]:
        """Base64 문자열을 이미지 바이트로 디코딩"""
        try:
            # Base64 헤더 제거
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            
            return base64.b64decode(base64_str)
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            return None
    
    def _extract_image_metadata(self, image_data: bytes, format_type: str) -> Dict:
        """이미지 메타데이터 추출"""
        try:
            metadata = {
                "size_bytes": len(image_data),
                "format": format_type,
                "dimensions": (0, 0),
                "mode": "unknown",
                "has_transparency": False
            }
            
            # PIL로 이미지 정보 추출
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    metadata.update({
                        "dimensions": img.size,
                        "mode": img.mode,
                        "format": img.format or format_type,
                        "has_transparency": img.mode in ["RGBA", "LA"] or "transparency" in img.info
                    })
            except Exception as e:
                logger.warning(f"Could not extract image metadata with PIL: {str(e)}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting image metadata: {str(e)}")
            return {}
    
    def _optimize_image(self, image_data: bytes, format_type: str) -> bytes:
        """이미지 최적화"""
        try:
            # 큰 이미지 리사이징
            with Image.open(io.BytesIO(image_data)) as img:
                # 최대 크기 제한 (1920x1080)
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # 최적화된 이미지를 바이트로 변환
                    output = io.BytesIO()
                    img.save(output, format=format_type.upper(), optimize=True, quality=85)
                    return output.getvalue()
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            return image_data
    
    def validate(self, parsed_data: Dict) -> bool:
        """이미지 데이터 유효성 검증"""
        try:
            if not parsed_data.get("is_valid", False):
                return False
            
            image_data = parsed_data.get("image_data")
            if not isinstance(image_data, bytes) or len(image_data) == 0:
                return False
            
            # PIL로 이미지 유효성 확인
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    img.verify()
                return True
            except:
                return False
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False

class CodeArtifactParser(BaseArtifactParser):
    """코드 전문 파서"""
    
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """코드 데이터 파싱 및 분석"""
        try:
            code_content = ""
            language = "python"
            
            # 코드 데이터 추출
            if isinstance(data, dict):
                code_content = data.get("code", data.get("script", data.get("content", "")))
                language = data.get("language", "python")
            elif isinstance(data, str):
                code_content = data
                language = self._detect_language(code_content)
            else:
                code_content = str(data)
                language = "text"
            
            # 코드 분석
            code_analysis = self._analyze_code(code_content, language)
            
            # 코드 포맷팅
            formatted_code = self._format_code(code_content, language)
            
            return {
                "code": formatted_code,
                "original_code": code_content,
                "language": language,
                "analysis": code_analysis,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing code: {str(e)}")
            return {
                "code": "",
                "error": str(e),
                "is_valid": False
            }
    
    def _detect_language(self, code: str) -> str:
        """코드 언어 자동 감지"""
        try:
            code_lower = code.lower()
            
            # SQL 감지
            sql_keywords = ["select", "insert", "update", "delete", "create", "alter", "drop"]
            if any(keyword in code_lower for keyword in sql_keywords):
                return "sql"
            
            # Python 감지
            python_keywords = ["def ", "import ", "from ", "class ", "if __name__"]
            if any(keyword in code for keyword in python_keywords):
                return "python"
            
            # R 감지
            r_keywords = ["library(", "<-", "function(", "data.frame"]
            if any(keyword in code for keyword in r_keywords):
                return "r"
            
            # JavaScript 감지
            js_keywords = ["function(", "var ", "let ", "const ", "=>"]
            if any(keyword in code for keyword in js_keywords):
                return "javascript"
            
            # JSON 감지
            if code.strip().startswith("{") and code.strip().endswith("}"):
                try:
                    json.loads(code)
                    return "json"
                except:
                    pass
            
            return "text"
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "text"
    
    def _analyze_code(self, code: str, language: str) -> Dict:
        """코드 분석"""
        try:
            analysis = {
                "line_count": len(code.split('\n')),
                "char_count": len(code),
                "word_count": len(code.split()),
                "executable": False,
                "complexity": "low",
                "functions": [],
                "imports": [],
                "comments": []
            }
            
            lines = code.split('\n')
            
            if language == "python":
                analysis.update(self._analyze_python_code(lines))
            elif language == "sql":
                analysis.update(self._analyze_sql_code(lines))
            elif language == "r":
                analysis.update(self._analyze_r_code(lines))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {}
    
    def _analyze_python_code(self, lines: List[str]) -> Dict:
        """Python 코드 분석"""
        analysis = {
            "executable": True,
            "functions": [],
            "imports": [],
            "classes": []
        }
        
        try:
            for line in lines:
                line = line.strip()
                
                if line.startswith("def "):
                    func_name = line.split("(")[0].replace("def ", "")
                    analysis["functions"].append(func_name)
                elif line.startswith("import ") or line.startswith("from "):
                    analysis["imports"].append(line)
                elif line.startswith("class "):
                    class_name = line.split("(")[0].replace("class ", "").replace(":", "")
                    analysis["classes"].append(class_name)
            
            # 복잡도 추정
            if len(analysis["functions"]) > 5 or len(analysis["classes"]) > 2:
                analysis["complexity"] = "high"
            elif len(analysis["functions"]) > 2 or len(analysis["classes"]) > 0:
                analysis["complexity"] = "medium"
            
        except Exception as e:
            logger.error(f"Error analyzing Python code: {str(e)}")
        
        return analysis
    
    def _analyze_sql_code(self, lines: List[str]) -> Dict:
        """SQL 코드 분석"""
        analysis = {
            "executable": True,
            "queries": [],
            "tables": []
        }
        
        try:
            for line in lines:
                line_upper = line.strip().upper()
                
                if line_upper.startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
                    analysis["queries"].append(line_upper.split()[0])
                
                if "FROM " in line_upper:
                    # 테이블 이름 추출 시도
                    try:
                        table_part = line_upper.split("FROM ")[1].split()[0]
                        analysis["tables"].append(table_part)
                    except:
                        pass
            
            if len(analysis["queries"]) > 3:
                analysis["complexity"] = "high"
            elif len(analysis["queries"]) > 1:
                analysis["complexity"] = "medium"
            
        except Exception as e:
            logger.error(f"Error analyzing SQL code: {str(e)}")
        
        return analysis
    
    def _analyze_r_code(self, lines: List[str]) -> Dict:
        """R 코드 분석"""
        analysis = {
            "executable": True,
            "libraries": [],
            "functions": []
        }
        
        try:
            for line in lines:
                line = line.strip()
                
                if line.startswith("library("):
                    lib_name = line.replace("library(", "").replace(")", "")
                    analysis["libraries"].append(lib_name)
                elif "<- function(" in line:
                    func_name = line.split("<-")[0].strip()
                    analysis["functions"].append(func_name)
            
        except Exception as e:
            logger.error(f"Error analyzing R code: {str(e)}")
        
        return analysis
    
    def _format_code(self, code: str, language: str) -> str:
        """코드 포맷팅"""
        try:
            # 기본적인 들여쓰기 정리
            lines = code.split('\n')
            formatted_lines = []
            
            for line in lines:
                # 탭을 스페이스로 변환
                formatted_line = line.replace('\t', '    ')
                formatted_lines.append(formatted_line)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting code: {str(e)}")
            return code
    
    def validate(self, parsed_data: Dict) -> bool:
        """코드 데이터 유효성 검증"""
        try:
            if not parsed_data.get("is_valid", False):
                return False
            
            code = parsed_data.get("code", "")
            if not isinstance(code, str) or len(code.strip()) == 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            return False

class TextArtifactParser(BaseArtifactParser):
    """텍스트 전문 파서"""
    
    async def parse(self, data: Any, metadata: Dict = None) -> Dict[str, Any]:
        """텍스트 데이터 파싱 및 분석"""
        try:
            text_content = ""
            format_type = "plain"
            
            # 텍스트 데이터 추출
            if isinstance(data, dict):
                text_content = data.get("text", data.get("content", str(data)))
                format_type = data.get("format", self._detect_text_format(text_content))
            elif isinstance(data, str):
                text_content = data
                format_type = self._detect_text_format(text_content)
            else:
                text_content = str(data)
                format_type = "plain"
            
            # 텍스트 분석
            text_analysis = self._analyze_text(text_content, format_type)
            
            # 텍스트 정리
            cleaned_text = self._clean_text(text_content, format_type)
            
            return {
                "text": cleaned_text,
                "original_text": text_content,
                "format": format_type,
                "analysis": text_analysis,
                "is_valid": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing text: {str(e)}")
            return {
                "text": "",
                "error": str(e),
                "is_valid": False
            }
    
    def _detect_text_format(self, text: str) -> str:
        """텍스트 형식 감지"""
        try:
            # HTML 감지
            if "<html>" in text.lower() or "<div>" in text.lower():
                return "html"
            
            # Markdown 감지
            markdown_patterns = ["# ", "## ", "- ", "* ", "```", "**", "__"]
            if any(pattern in text for pattern in markdown_patterns):
                return "markdown"
            
            # JSON 감지
            if text.strip().startswith("{") and text.strip().endswith("}"):
                try:
                    json.loads(text)
                    return "json"
                except:
                    pass
            
            return "plain"
            
        except Exception as e:
            logger.error(f"Error detecting text format: {str(e)}")
            return "plain"
    
    def _analyze_text(self, text: str, format_type: str) -> Dict:
        """텍스트 분석"""
        try:
            analysis = {
                "word_count": len(text.split()),
                "char_count": len(text),
                "line_count": len(text.split('\n')),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "language": "unknown",
                "readability": "medium",
                "structure": []
            }
            
            if format_type == "markdown":
                analysis.update(self._analyze_markdown(text))
            elif format_type == "html":
                analysis.update(self._analyze_html(text))
            
            # 언어 감지 (간단한 휴리스틱)
            if self._contains_korean(text):
                analysis["language"] = "korean"
            elif self._contains_english(text):
                analysis["language"] = "english"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {}
    
    def _analyze_markdown(self, text: str) -> Dict:
        """Markdown 텍스트 분석"""
        analysis = {
            "headers": [],
            "lists": 0,
            "code_blocks": 0,
            "links": 0
        }
        
        try:
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 헤더 감지
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    header_text = line.lstrip('# ').strip()
                    analysis["headers"].append({"level": level, "text": header_text})
                
                # 리스트 감지
                elif line.startswith(('- ', '* ', '+ ')) or line.lstrip().startswith(tuple(f"{i}. " for i in range(1, 10))):
                    analysis["lists"] += 1
                
                # 코드 블록 감지
                elif line.startswith('```'):
                    analysis["code_blocks"] += 1
                
                # 링크 감지
                elif '[' in line and '](' in line:
                    analysis["links"] += line.count('](')
            
        except Exception as e:
            logger.error(f"Error analyzing markdown: {str(e)}")
        
        return analysis
    
    def _analyze_html(self, text: str) -> Dict:
        """HTML 텍스트 분석"""
        analysis = {
            "tags": [],
            "structure": "basic"
        }
        
        try:
            # 간단한 태그 감지
            import re
            tags = re.findall(r'<(\w+)', text.lower())
            analysis["tags"] = list(set(tags))
            
            if any(tag in tags for tag in ['div', 'span', 'p']):
                analysis["structure"] = "structured"
            
        except Exception as e:
            logger.error(f"Error analyzing HTML: {str(e)}")
        
        return analysis
    
    def _contains_korean(self, text: str) -> bool:
        """한국어 포함 여부 확인"""
        try:
            import re
            korean_pattern = re.compile(r'[가-힣]')
            return bool(korean_pattern.search(text))
        except:
            return False
    
    def _contains_english(self, text: str) -> bool:
        """영어 포함 여부 확인"""
        try:
            import re
            english_pattern = re.compile(r'[a-zA-Z]')
            return bool(english_pattern.search(text))
        except:
            return False
    
    def _clean_text(self, text: str, format_type: str) -> str:
        """텍스트 정리"""
        try:
            cleaned = text
            
            # 기본 정리
            cleaned = cleaned.strip()
            
            # 연속된 공백 정리
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # 연속된 줄바꿈 정리
            cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def validate(self, parsed_data: Dict) -> bool:
        """텍스트 데이터 유효성 검증"""
        try:
            if not parsed_data.get("is_valid", False):
                return False
            
            text = parsed_data.get("text", "")
            if not isinstance(text, str) or len(text.strip()) == 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating text: {str(e)}")
            return False