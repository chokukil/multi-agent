import pandas as pd
import os
import json
import logging
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataRequestAnalysis:
    """데이터 요청 분석 결과"""
    requested_file: Optional[str] = None
    file_hints: List[str] = None
    data_description: str = ""
    confidence_score: float = 0.0
    fallback_strategy: str = "latest"
    semantic_matches: List[Dict] = None
    
    def __post_init__(self):
        if self.file_hints is None:
            self.file_hints = []
        if self.semantic_matches is None:
            self.semantic_matches = []

@dataclass 
class DataValidationResult:
    """데이터 검증 결과"""
    valid: bool
    error: str = ""
    shape: Tuple[int, int] = (0, 0)
    columns: List[str] = None
    memory_usage: int = 0
    data_types: Dict[str, str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.data_types is None:
            self.data_types = {}

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
        self.supported_formats = ['.csv', '.pkl', '.xlsx', '.xls', '.json']
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
        available_files = self._scan_available_data()
        
        # 3. LLM을 통한 데이터 요청 분석
        data_request = await self._analyze_data_request(user_request, explicit_data, available_files)
        
        # 4. 데이터 로드 시도
        df, description = await self._load_best_match_data(data_request, available_files)
        
        return df, description
    
    def _extract_explicit_data_from_a2a(self, context) -> Dict[str, Any]:
        """A2A 컨텍스트에서 명시적 데이터 정보 추출"""
        explicit_data = {}
        
        try:
            if hasattr(context, 'message') and hasattr(context.message, 'parts'):
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "data":
                        if hasattr(part.root, 'data'):
                            explicit_data.update(part.root.data)
        except Exception as e:
            logger.warning(f"A2A 데이터 추출 실패: {e}")
            
        return explicit_data
    
    def _scan_available_data(self) -> List[str]:
        """사용 가능한 데이터 파일 스캔"""
        available_files = []
        
        try:
            if os.path.exists(self.data_path):
                for filename in os.listdir(self.data_path):
                    file_path = os.path.join(self.data_path, filename)
                    if os.path.isfile(file_path):
                        # 지원되는 형식만 포함
                        if any(filename.lower().endswith(fmt) for fmt in self.supported_formats):
                            available_files.append(filename)
        except Exception as e:
            logger.error(f"파일 스캔 실패: {e}")
            
        return available_files
    
    async def _analyze_data_request(self, user_request: str, explicit_data: Dict, available_files: List[str]) -> DataRequestAnalysis:
        """LLM을 통한 데이터 요청 분석"""
        
        # 1. 명시적 데이터가 있는 경우 우선 처리
        if explicit_data.get("file_name") in available_files:
            return DataRequestAnalysis(
                requested_file=explicit_data["file_name"],
                confidence_score=1.0,
                data_description="명시적으로 지정된 파일"
            )
        
        # 2. 파일 힌트 생성
        file_hints = self._generate_file_hints(user_request)
        
        # 3. 의미적 매칭
        semantic_matches = self._find_semantic_matches(user_request, available_files)
        
        # 4. LLM 분석 (선택적)
        llm_analysis = await self._get_llm_analysis(user_request, available_files)
        
        # 5. 최종 분석 결과 구성
        analysis = DataRequestAnalysis(
            file_hints=file_hints,
            semantic_matches=semantic_matches,
            confidence_score=self._calculate_confidence_score(user_request, available_files),
            data_description=f"사용자 요청: {user_request[:100]}..."
        )
        
        # LLM 분석 결과 통합
        if llm_analysis:
            analysis.requested_file = llm_analysis.get("suggested_file")
            analysis.confidence_score = max(analysis.confidence_score, llm_analysis.get("confidence", 0.0))
        
        return analysis
    
    def _generate_file_hints(self, user_request: str) -> List[str]:
        """사용자 요청에서 파일 힌트 추출"""
        hints = []
        
        # 키워드 기반 힌트 매핑
        keyword_mappings = {
            "고객": ["customer", "client", "user"],
            "매출": ["sales", "revenue", "income"],
            "재고": ["inventory", "stock"],
            "분석": ["analysis", "analytics"],
            "보고서": ["report", "summary"],
            "ion_implant": ["ion", "implant", "semiconductor"],
            "반도체": ["semiconductor", "chip", "wafer"]
        }
        
        user_request_lower = user_request.lower()
        
        for keyword, mapped_hints in keyword_mappings.items():
            if keyword in user_request_lower:
                hints.extend(mapped_hints)
        
        return list(set(hints))  # 중복 제거
    
    def _find_semantic_matches(self, user_request: str, available_files: List[str]) -> List[Dict]:
        """의미적 파일 매칭"""
        matches = []
        
        file_hints = self._generate_file_hints(user_request)
        
        for filename in available_files:
            score = 0.0
            matched_hints = []
            
            filename_lower = filename.lower()
            
            # 파일명에서 힌트 매칭
            for hint in file_hints:
                if hint.lower() in filename_lower:
                    score += 1.0
                    matched_hints.append(hint)
            
            # ion_implant 우선순위 부여
            if "ion" in filename_lower and "implant" in filename_lower:
                score += 2.0
                matched_hints.append("ion_implant_priority")
            
            if score > 0:
                matches.append({
                    "filename": filename,
                    "score": score,
                    "matched_hints": matched_hints
                })
        
        # 점수 기준으로 정렬
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return matches
    
    async def _get_llm_analysis(self, user_request: str, available_files: List[str]) -> Optional[Dict]:
        """LLM을 통한 고급 분석 (선택적)"""
        try:
            prompt = f"""
사용자 요청을 분석하여 가장 적합한 데이터 파일을 선택해주세요.

사용자 요청: {user_request}

사용 가능한 파일들:
{', '.join(available_files)}

다음 JSON 형식으로 응답해주세요:
{{
    "suggested_file": "추천할 파일명 또는 null",
    "confidence": 0.0에서 1.0 사이의 신뢰도,
    "reasoning": "선택 이유"
}}
"""
            
            response = await self.llm.ainvoke(prompt)
            
            if hasattr(response, 'content'):
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    logger.warning("LLM 응답을 JSON으로 파싱할 수 없습니다")
                    
        except Exception as e:
            logger.warning(f"LLM 분석 실패: {e}")
            
        return None
    
    def _calculate_confidence_score(self, user_request: str, available_files: List[str]) -> float:
        """신뢰도 점수 계산"""
        score = 0.0
        
        # 명시적 파일명 언급
        for filename in available_files:
            if filename.lower() in user_request.lower():
                score += 0.5
        
        # 구체적인 키워드 존재
        specific_keywords = ["분석", "시각화", "예측", "분류", "군집화"]
        for keyword in specific_keywords:
            if keyword in user_request:
                score += 0.1
        
        # 요청 길이 (더 구체적일수록 높은 점수)
        if len(user_request) > 50:
            score += 0.2
        elif len(user_request) > 20:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _load_best_match_data(self, analysis: DataRequestAnalysis, available_files: List[str]) -> Tuple[Optional[pd.DataFrame], str]:
        """최적 매치 데이터 로드"""
        
        # 1. 명시적 요청 파일 확인
        if analysis.requested_file and analysis.requested_file in available_files:
            df = self._try_load_file(analysis.requested_file)
            if df is not None:
                validation = self._validate_loaded_data(df)
                if validation.valid:
                    return df, f"요청된 파일 로드 성공: {analysis.requested_file}"
        
        # 2. 의미적 매칭 결과 사용
        for match in analysis.semantic_matches:
            df = self._try_load_file(match["filename"])
            if df is not None:
                validation = self._validate_loaded_data(df)
                if validation.valid:
                    return df, f"의미적 매칭으로 선택: {match['filename']} (점수: {match['score']})"
        
        # 3. 폴백 전략 적용
        if available_files:
            fallback_file = self._apply_fallback_strategy(available_files, analysis.fallback_strategy)
            if fallback_file:
                df = self._try_load_file(fallback_file)
                if df is not None:
                    validation = self._validate_loaded_data(df)
                    if validation.valid:
                        return df, f"폴백 전략으로 선택: {fallback_file}"
        
        return None, "적절한 데이터 파일을 찾을 수 없습니다"
    
    def _try_load_file(self, filename: str) -> Optional[pd.DataFrame]:
        """안전한 파일 로딩"""
        if not filename:
            return None
            
        file_path = os.path.join(self.data_path, filename)
        
        try:
            if not os.path.exists(file_path):
                return None
            
            # 확장자에 따른 로딩
            if filename.lower().endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            elif filename.lower().endswith('.json'):
                return pd.read_json(file_path)
            elif filename.lower().endswith('.pkl'):
                return pd.read_pickle(file_path)
                
        except Exception as e:
            logger.warning(f"파일 로딩 실패 {filename}: {e}")
            
        return None
    
    def _validate_loaded_data(self, df: pd.DataFrame) -> DataValidationResult:
        """로드된 데이터 검증"""
        if df is None:
            return DataValidationResult(valid=False, error="데이터프레임이 None입니다")
        
        if df.empty:
            return DataValidationResult(valid=False, error="데이터프레임이 비어있습니다")
        
        try:
            shape = df.shape
            columns = list(df.columns)
            memory_usage = df.memory_usage(deep=True).sum()
            data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            return DataValidationResult(
                valid=True,
                shape=shape,
                columns=columns,
                memory_usage=memory_usage,
                data_types=data_types
            )
            
        except Exception as e:
            return DataValidationResult(valid=False, error=f"데이터 검증 실패: {e}")
    
    def _apply_fallback_strategy(self, available_files: List[str], strategy: str) -> Optional[str]:
        """폴백 전략 적용"""
        if not available_files:
            return None
        
        try:
            if strategy == "first":
                return available_files[0]
            elif strategy == "latest":
                return self._get_latest_file(available_files)
            elif strategy == "largest":
                return self._get_largest_file(available_files)
            else:
                return available_files[0]  # 기본값
                
        except Exception as e:
            logger.warning(f"폴백 전략 적용 실패: {e}")
            return available_files[0] if available_files else None
    
    def _get_latest_file(self, files: List[str]) -> Optional[str]:
        """가장 최근 수정된 파일 선택"""
        try:
            latest_file = None
            latest_time = 0
            
            for filename in files:
                file_path = os.path.join(self.data_path, filename)
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_file = filename
            
            return latest_file
            
        except Exception as e:
            logger.warning(f"최신 파일 찾기 실패: {e}")
            return files[0] if files else None
    
    def _get_largest_file(self, files: List[str]) -> Optional[str]:
        """가장 큰 파일 선택"""
        try:
            largest_file = None
            largest_size = 0
            
            for filename in files:
                file_path = os.path.join(self.data_path, filename)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    if size > largest_size:
                        largest_size = size
                        largest_file = filename
            
            return largest_file
            
        except Exception as e:
            logger.warning(f"가장 큰 파일 찾기 실패: {e}")
            return files[0] if files else None
    
    def _select_best_file(self, available_files: List[str], preferred_file: Optional[str], fallback_strategy: str) -> Optional[str]:
        """최적 파일 선택 (단위 테스트용)"""
        
        # 1. 우선 파일 확인
        if preferred_file and preferred_file in available_files:
            return preferred_file
        
        # 2. ion_implant 우선순위
        for filename in available_files:
            if "ion_implant" in filename.lower():
                return filename
        
        # 3. 폴백 전략 적용
        return self._apply_fallback_strategy(available_files, fallback_strategy)
    
    def _find_semantic_match(self, user_request: str, available_files: List[Dict]) -> Dict:
        """의미적 매칭 (단위 테스트용)"""
        matches = []
        
        for file_info in available_files:
            filename = file_info["filename"]
            metadata = file_info.get("metadata", {})
            description = metadata.get("description", "")
            
            # 간단한 키워드 매칭
            score = 0
            if "고객" in user_request and "customer" in filename.lower():
                score += 1
            if "매출" in user_request and "sales" in filename.lower():
                score += 1
            if "고객" in user_request and "고객" in description:
                score += 1
                
            matches.append({"file_info": file_info, "score": score})
        
        # 가장 높은 점수의 파일 반환
        if matches:
            best_match = max(matches, key=lambda x: x["score"])
            return best_match["file_info"]
        
        return available_files[0] if available_files else {} 