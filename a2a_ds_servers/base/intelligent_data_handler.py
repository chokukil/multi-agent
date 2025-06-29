import os
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """데이터 요청 구조"""
    requested_file: Optional[str]
    data_description: str
    analysis_context: str
    file_hints: List[str]
    confidence_score: float

class IntelligentDataHandler:
    """LLM 기반 지능형 데이터 처리 시스템"""
    
    def __init__(self, llm_instance=None):
        """
        Args:
            llm_instance: LLM 인스턴스 (없으면 기본 LLM 사용)
        """
        if llm_instance:
            self.llm = llm_instance
        else:
            from core.llm_factory import create_llm_instance
            self.llm = create_llm_instance()
        
        self.data_path = "a2a_ds_servers/artifacts/data/shared_dataframes/"
        logger.info("IntelligentDataHandler initialized with LLM-based data resolution")
    
    async def resolve_data_request(self, user_request: str, context) -> Tuple[Optional[pd.DataFrame], str]:
        """
        LLM을 사용하여 사용자 요청에서 데이터 의도를 파악하고 적절한 데이터 로드
        
        Args:
            user_request: 사용자의 원본 요청
            context: A2A RequestContext
            
        Returns:
            Tuple[DataFrame, description]: 로드된 데이터와 설명
        """
        
        # 1. A2A 메시지에서 명시적 데이터 정보 추출
        explicit_data = self._extract_explicit_data_from_a2a(context)
        
        # 2. 사용 가능한 데이터 파일 목록 수집
        available_files = self._get_available_files()
        
        # 3. LLM을 통한 데이터 요청 분석
        data_request = await self._analyze_data_request(user_request, explicit_data, available_files)
        
        # 4. 데이터 로드 시도
        df, description = await self._load_best_match_data(data_request, available_files)
        
        return df, description
    
    def _extract_explicit_data_from_a2a(self, context) -> Dict[str, Any]:
        """A2A 메시지에서 명시적 데이터 정보 추출"""
        explicit_data = {
            "file_parts": [],
            "data_parts": [],
            "text_mentions": []
        }
        
        if context.message and context.message.parts:
            for part in context.message.parts:
                try:
                    actual_part = part.root
                    
                    if actual_part.kind == "file":
                        # FilePart 정보
                        file_info = {
                            "uri": getattr(actual_part, 'uri', ''),
                            "filename": getattr(actual_part, 'metadata', {}).get('filename', ''),
                            "mime_type": getattr(actual_part, 'mimeType', '')
                        }
                        explicit_data["file_parts"].append(file_info)
                        
                    elif actual_part.kind == "data":
                        # DataPart 정보
                        data_info = getattr(actual_part, 'data', {})
                        explicit_data["data_parts"].append(data_info)
                        
                    elif actual_part.kind == "text":
                        # 텍스트에서 파일명 언급 추출
                        text = actual_part.text
                        file_mentions = self._extract_file_mentions(text)
                        explicit_data["text_mentions"].extend(file_mentions)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting data from A2A part: {e}")
        
        return explicit_data
    
    def _extract_file_mentions(self, text: str) -> List[str]:
        """텍스트에서 파일명 언급 추출"""
        import re
        
        # 파일 확장자 패턴
        file_patterns = [
            r'(\w+\.csv)',
            r'(\w+\.xlsx?)',
            r'(\w+\.pkl)',
            r'(\w+\.json)',
            r'(\w+_\w+\.csv)',  # underscore 포함
            r'(\w+_\w+_\w+\.csv)',  # 여러 underscore
        ]
        
        mentions = []
        for pattern in file_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)
        
        return list(set(mentions))  # 중복 제거
    
    def _get_available_files(self) -> List[Dict[str, Any]]:
        """사용 가능한 데이터 파일 목록과 메타데이터 수집"""
        files = []
        
        try:
            for filename in os.listdir(self.data_path):
                if filename.endswith(('.csv', '.pkl', '.xlsx', '.json')):
                    file_path = os.path.join(self.data_path, filename)
                    file_info = {
                        "filename": filename,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path),
                        "extension": os.path.splitext(filename)[1]
                    }
                    
                    # 파일 내용 미리보기 (작은 파일만)
                    if file_info["size"] < 10 * 1024 * 1024:  # 10MB 미만
                        try:
                            preview = self._get_file_preview(file_path)
                            file_info["preview"] = preview
                        except:
                            file_info["preview"] = "Preview not available"
                    
                    files.append(file_info)
                    
        except Exception as e:
            logger.error(f"Error scanning data directory: {e}")
        
        return files
    
    def _get_file_preview(self, file_path: str) -> str:
        """파일 미리보기 생성"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=3)
                return f"Shape: {df.shape}, Columns: {list(df.columns)}"
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
                return f"Shape: {df.shape}, Columns: {list(df.columns[:5])}"  # 처음 5개 컬럼만
            else:
                return "Binary file"
        except:
            return "Preview failed"
    
    async def _analyze_data_request(self, user_request: str, explicit_data: Dict, available_files: List[Dict]) -> DataRequest:
        """LLM을 통한 데이터 요청 분석"""
        
        # 사용 가능한 파일 요약
        files_summary = []
        for file_info in available_files[:10]:  # 최대 10개만
            summary = f"- {file_info['filename']}: {file_info.get('preview', 'No preview')}"
            files_summary.append(summary)
        
        analysis_prompt = f"""
당신은 데이터 분석 전문가입니다. 사용자의 요청을 분석하여 어떤 데이터가 필요한지 파악해주세요.

사용자 요청:
```
{user_request}
```

명시적 데이터 정보:
- 파일 Parts: {explicit_data.get('file_parts', [])}
- 데이터 Parts: {explicit_data.get('data_parts', [])}
- 텍스트 언급: {explicit_data.get('text_mentions', [])}

사용 가능한 파일들:
{chr(10).join(files_summary)}

다음 JSON 형식으로 분석 결과를 제공해주세요:

{{
    "requested_file": "명시적으로 요청된 파일명 (없으면 null)",
    "data_description": "요청된 데이터의 특성이나 도메인 설명",
    "analysis_context": "분석의 목적이나 컨텍스트",
    "file_hints": ["추천할 수 있는 파일명들"],
    "confidence_score": 0.0-1.0
}}

분석 기준:
1. 명시적 파일명이 언급되었는가?
2. 데이터의 도메인이나 특성이 설명되었는가?
3. 분석 목적이 명확한가?
4. 사용 가능한 파일 중 적합한 것이 있는가?

JSON만 반환하세요:
"""

        try:
            response = await self._call_llm_async(analysis_prompt)
            analysis_result = self._extract_json_from_response(response)
            
            return DataRequest(
                requested_file=analysis_result.get("requested_file"),
                data_description=analysis_result.get("data_description", ""),
                analysis_context=analysis_result.get("analysis_context", ""),
                file_hints=analysis_result.get("file_hints", []),
                confidence_score=float(analysis_result.get("confidence_score", 0.5))
            )
            
        except Exception as e:
            logger.warning(f"⚠️ LLM data analysis failed: {e}, using fallback")
            return self._create_fallback_data_request(user_request, explicit_data)
    
    async def _load_best_match_data(self, data_request: DataRequest, available_files: List[Dict]) -> Tuple[Optional[pd.DataFrame], str]:
        """최적 매칭 데이터 로드"""
        
        # 1. 명시적 요청 파일 우선 처리
        if data_request.requested_file:
            df = self._try_load_file(data_request.requested_file)
            if df is not None:
                return df, f"✅ 요청된 파일 로드됨: {data_request.requested_file}"
        
        # 2. 힌트 파일들 시도
        for hint_file in data_request.file_hints:
            df = self._try_load_file(hint_file)
            if df is not None:
                return df, f"✅ 추천 파일 로드됨: {hint_file}"
        
        # 3. LLM을 통한 스마트 매칭
        if data_request.data_description:
            best_match = await self._find_best_semantic_match(data_request, available_files)
            if best_match:
                df = self._try_load_file(best_match["filename"])
                if df is not None:
                    return df, f"✅ 의미적 매칭으로 로드됨: {best_match['filename']}"
        
        # 4. 최신 파일 fallback (조건부)
        if not data_request.requested_file and data_request.confidence_score < 0.3:
            latest_file = max(available_files, key=lambda x: x["modified"], default=None)
            if latest_file:
                df = self._try_load_file(latest_file["filename"])
                if df is not None:
                    return df, f"⚠️ 최신 파일로 fallback: {latest_file['filename']} (데이터 요청이 불명확함)"
        
        return None, "❌ 적절한 데이터 파일을 찾을 수 없습니다."
    
    async def _find_best_semantic_match(self, data_request: DataRequest, available_files: List[Dict]) -> Optional[Dict]:
        """의미적 매칭을 통한 최적 파일 찾기"""
        
        if not available_files:
            return None
        
        matching_prompt = f"""
데이터 요청: {data_request.data_description}
분석 컨텍스트: {data_request.analysis_context}

사용 가능한 파일들:
{chr(10).join([f"- {f['filename']}: {f.get('preview', '')}" for f in available_files])}

위 데이터 요청에 가장 적합한 파일을 선택해주세요. 파일명만 반환하세요.
만약 적합한 파일이 없다면 "NONE"을 반환하세요.
"""

        try:
            response = await self._call_llm_async(matching_prompt)
            filename = response.strip().strip('"\'')
            
            # 응답 검증
            for file_info in available_files:
                if file_info["filename"] == filename:
                    return file_info
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic matching failed: {e}")
            return None
    
    def _try_load_file(self, filename: str) -> Optional[pd.DataFrame]:
        """파일 로드 시도"""
        if not filename:
            return None
        
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            if filename.endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.endswith('.pkl'):
                return pd.read_pickle(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            elif filename.endswith('.json'):
                return pd.read_json(file_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}")
            return None
    
    async def _call_llm_async(self, prompt: str) -> str:
        """LLM 비동기 호출"""
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            elif hasattr(self.llm, 'acall'):
                return await self.llm.acall(prompt)
            else:
                import asyncio
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._call_llm_sync, prompt
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _call_llm_sync(self, prompt: str) -> str:
        """LLM 동기 호출 (fallback)"""
        if hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        elif hasattr(self.llm, 'call'):
            return self.llm.call(prompt)
        else:
            return self.llm(prompt)
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """응답에서 JSON 추출"""
        import json
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {}
    
    def _create_fallback_data_request(self, user_request: str, explicit_data: Dict) -> DataRequest:
        """fallback 데이터 요청 생성"""
        # 텍스트 언급에서 파일명 추출
        mentioned_files = explicit_data.get('text_mentions', [])
        
        return DataRequest(
            requested_file=mentioned_files[0] if mentioned_files else None,
            data_description="General data analysis",
            analysis_context=user_request,
            file_hints=mentioned_files,
            confidence_score=0.3
        ) 