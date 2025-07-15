"""
향상된 파일 커넥터 (Enhanced File Connector)

pandas_agent의 FileConnector 패턴을 기준으로 한 향상된 파일 데이터 소스 커넥터
다중 인코딩 지원, 지능형 로딩 전략, 캐싱 등의 고급 기능 제공

핵심 원칙:
- pandas_agent FileConnector 100% 호환
- UTF-8 인코딩 문제 자동 해결
- LLM First: 로딩 전략을 LLM이 동적 결정
- 성능 최적화: 캐싱 및 청크 로딩 지원
"""

import pandas as pd
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import os

from ..core.unified_data_interface import LoadingStrategy, A2AContext
from ..core.smart_dataframe import SmartDataFrame
from ..utils.encoding_detector import EncodingDetector
from ..core.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class EnhancedFileConnector:
    """
    향상된 파일 커넥터
    
    pandas_agent의 FileConnector 패턴을 기준으로 구현된
    지능형 파일 로딩 커넥터
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        향상된 파일 커넥터 초기화
        
        Args:
            cache_manager: 캐시 매니저 (선택적)
        """
        self.encoding_detector = EncodingDetector()
        self.cache_manager = cache_manager
        
        # 지원하는 파일 형식
        self.supported_formats = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.feather': self._load_feather,
            '.txt': self._load_text,
            '.tsv': self._load_tsv
        }
        
        logger.info("✅ EnhancedFileConnector 초기화 완료")
    
    async def load_file(self, 
                       file_path: str, 
                       strategy: LoadingStrategy,
                       context: Optional[A2AContext] = None) -> SmartDataFrame:
        """
        파일 로딩 (메인 메서드)
        
        Args:
            file_path: 파일 경로
            strategy: 로딩 전략
            context: A2A 컨텍스트
            
        Returns:
            SmartDataFrame: 로딩된 지능형 DataFrame
        """
        try:
            # 캐시 확인
            if self.cache_manager and strategy.use_cache:
                cache_key = self._generate_cache_key(file_path, strategy)
                cached_df = await self.cache_manager.get(cache_key)
                
                if cached_df is not None:
                    logger.info(f"✅ 캐시에서 로딩: {file_path}")
                    return cached_df
            
            # 파일 존재 확인
            file_obj = Path(file_path)
            if not file_obj.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            # 파일 형식 감지
            file_extension = file_obj.suffix.lower()
            if file_extension not in self.supported_formats:
                raise ValueError(f"지원되지 않는 파일 형식: {file_extension}")
            
            # 파일 로딩
            loader_func = self.supported_formats[file_extension]
            df = await loader_func(file_path, strategy)
            
            # 메타데이터 생성
            metadata = await self._generate_metadata(file_path, strategy, context)
            
            # SmartDataFrame 생성
            smart_df = SmartDataFrame(df, metadata)
            
            # 캐시 저장
            if self.cache_manager and strategy.use_cache:
                await self.cache_manager.set(
                    cache_key, 
                    smart_df, 
                    ttl=strategy.cache_ttl,
                    tags={f"file:{file_obj.name}", f"ext:{file_extension}"}
                )
            
            logger.info(f"✅ 파일 로딩 완료: {file_path} ({smart_df.shape})")
            return smart_df
            
        except Exception as e:
            logger.error(f"❌ 파일 로딩 실패 {file_path}: {e}")
            raise
    
    async def _load_csv(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """CSV 파일 로딩"""
        try:
            # 인코딩 감지 또는 전략 사용
            encoding = await self._determine_encoding(file_path, strategy)
            
            # 로딩 파라미터 설정
            kwargs = {
                'encoding': encoding,
                'low_memory': False  # 타입 추론 개선
            }
            
            # 청크 로딩 지원
            if strategy.chunk_size:
                kwargs['chunksize'] = strategy.chunk_size
                
                # 청크 단위로 읽어서 합치기
                chunks = []
                for chunk in pd.read_csv(file_path, **kwargs):
                    chunks.append(chunk)
                    
                    # 샘플링이 필요한 경우
                    if strategy.sample_ratio and len(chunks) * strategy.chunk_size >= 10000:
                        break
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path, **kwargs)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"🎯 샘플링 적용: {len(df)}행 ({strategy.sample_ratio:.1%})")
            
            return df
            
        except UnicodeDecodeError as e:
            # 폴백 인코딩 시도
            logger.warning(f"⚠️ 인코딩 오류, 폴백 시도: {e}")
            return await self._load_csv_with_fallback(file_path, strategy)
    
    async def _load_csv_with_fallback(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """CSV 폴백 로딩 (다중 인코딩 시도)"""
        for encoding in strategy.fallback_encodings:
            try:
                logger.info(f"🔄 폴백 인코딩 시도: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                logger.info(f"✅ 폴백 인코딩 성공: {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # 모든 인코딩 실패 시 errors='ignore' 사용
        logger.warning("⚠️ 모든 인코딩 실패, errors='ignore' 사용")
        return pd.read_csv(file_path, encoding='utf-8', errors='ignore', low_memory=False)
    
    async def _load_excel(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """Excel 파일 로딩"""
        try:
            kwargs = {}
            
            # 엔진 자동 선택
            if file_path.endswith('.xlsx'):
                kwargs['engine'] = 'openpyxl'
            elif file_path.endswith('.xls'):
                kwargs['engine'] = 'xlrd'
            
            df = pd.read_excel(file_path, **kwargs)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Excel 로딩 실패: {e}")
            raise
    
    async def _load_json(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """JSON 파일 로딩"""
        try:
            # 인코딩 감지
            encoding = await self._determine_encoding(file_path, strategy)
            
            # JSON 로딩 시도 (여러 형식 지원)
            try:
                df = pd.read_json(file_path, encoding=encoding)
            except ValueError:
                # lines=True 시도 (JSONL 형식)
                df = pd.read_json(file_path, lines=True, encoding=encoding)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ JSON 로딩 실패: {e}")
            raise
    
    async def _load_parquet(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """Parquet 파일 로딩"""
        try:
            df = pd.read_parquet(file_path)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Parquet 로딩 실패: {e}")
            raise
    
    async def _load_feather(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """Feather 파일 로딩"""
        try:
            df = pd.read_feather(file_path)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feather 로딩 실패: {e}")
            raise
    
    async def _load_text(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """텍스트 파일 로딩 (단순 라인 기반)"""
        try:
            encoding = await self._determine_encoding(file_path, strategy)
            
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            # 라인을 DataFrame으로 변환
            df = pd.DataFrame({'text': [line.strip() for line in lines]})
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 텍스트 로딩 실패: {e}")
            raise
    
    async def _load_tsv(self, file_path: str, strategy: LoadingStrategy) -> pd.DataFrame:
        """TSV 파일 로딩"""
        try:
            encoding = await self._determine_encoding(file_path, strategy)
            
            kwargs = {
                'encoding': encoding,
                'sep': '\t',
                'low_memory': False
            }
            
            df = pd.read_csv(file_path, **kwargs)
            
            # 샘플링 적용
            if strategy.sample_ratio and len(df) > 1000:
                sample_size = int(len(df) * strategy.sample_ratio)
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ TSV 로딩 실패: {e}")
            raise
    
    async def _determine_encoding(self, file_path: str, strategy: LoadingStrategy) -> str:
        """인코딩 결정"""
        # 전략에 명시적 인코딩이 있으면 우선 사용
        if strategy.encoding and strategy.encoding != 'auto':
            return strategy.encoding
        
        # 자동 감지
        detected_encoding = await self.encoding_detector.detect_encoding(file_path)
        return detected_encoding
    
    async def _generate_metadata(self, 
                                file_path: str, 
                                strategy: LoadingStrategy,
                                context: Optional[A2AContext]) -> Dict[str, Any]:
        """메타데이터 생성"""
        file_obj = Path(file_path)
        
        metadata = {
            "source": "file",
            "file_path": str(file_obj.absolute()),
            "file_name": file_obj.name,
            "file_extension": file_obj.suffix.lower(),
            "file_size": file_obj.stat().st_size,
            "encoding": strategy.encoding,
            "loading_strategy": {
                "chunk_size": strategy.chunk_size,
                "sample_ratio": strategy.sample_ratio,
                "use_cache": strategy.use_cache,
                "cache_ttl": strategy.cache_ttl
            },
            "loaded_at": pd.Timestamp.now().isoformat()
        }
        
        # A2A 컨텍스트 정보 추가
        if context:
            metadata.update({
                "session_id": context.session_id,
                "user_id": context.user_id,
                "request_id": context.request_id
            })
        
        return metadata
    
    def _generate_cache_key(self, file_path: str, strategy: LoadingStrategy) -> str:
        """캐시 키 생성"""
        file_obj = Path(file_path)
        
        # 파일 경로, 수정 시간, 전략을 포함한 키 생성
        key_components = [
            str(file_obj.absolute()),
            str(file_obj.stat().st_mtime),
            strategy.encoding,
            str(strategy.chunk_size),
            str(strategy.sample_ratio)
        ]
        
        return "|".join(str(comp) for comp in key_components)
    
    async def validate_file(self, file_path: str) -> Dict[str, Any]:
        """파일 유효성 검증"""
        try:
            file_obj = Path(file_path)
            
            validation_result = {
                "valid": False,
                "exists": file_obj.exists(),
                "readable": False,
                "supported_format": False,
                "estimated_size_mb": 0,
                "encoding_issues": [],
                "recommendations": []
            }
            
            if not validation_result["exists"]:
                validation_result["recommendations"].append("파일이 존재하지 않습니다")
                return validation_result
            
            # 읽기 권한 확인
            validation_result["readable"] = file_obj.is_file() and os.access(file_obj, os.R_OK)
            
            # 지원 형식 확인
            file_extension = file_obj.suffix.lower()
            validation_result["supported_format"] = file_extension in self.supported_formats
            
            # 파일 크기
            file_size = file_obj.stat().st_size
            validation_result["estimated_size_mb"] = file_size / (1024 * 1024)
            
            # 인코딩 분석 (텍스트 파일만)
            if file_extension in ['.csv', '.txt', '.tsv', '.json']:
                encoding_analysis = await self.encoding_detector.analyze_file_encoding_issues(file_path)
                validation_result["encoding_issues"] = encoding_analysis.get("recommendations", [])
            
            # 권장사항 생성
            if validation_result["estimated_size_mb"] > 100:
                validation_result["recommendations"].append("대용량 파일입니다. 샘플링 옵션을 고려하세요")
            
            if not validation_result["supported_format"]:
                validation_result["recommendations"].append(f"지원되지 않는 형식입니다: {file_extension}")
            
            # 전체 유효성
            validation_result["valid"] = (
                validation_result["exists"] and 
                validation_result["readable"] and 
                validation_result["supported_format"]
            )
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "recommendations": ["파일 유효성 검증 중 오류가 발생했습니다"]
            }
    
    def get_supported_formats(self) -> List[str]:
        """지원되는 파일 형식 리스트 반환"""
        return list(self.supported_formats.keys())
    
    async def preview_file(self, file_path: str, lines: int = 10) -> Dict[str, Any]:
        """파일 미리보기"""
        try:
            # 기본 전략으로 미리보기 로딩
            strategy = LoadingStrategy(
                encoding='utf-8',
                sample_ratio=0.1 if lines > 5 else None
            )
            
            smart_df = await self.load_file(file_path, strategy)
            
            preview = {
                "shape": smart_df.shape,
                "columns": list(smart_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in smart_df.dtypes.items()},
                "sample_data": smart_df.head(lines).to_dict('records'),
                "metadata": smart_df.metadata
            }
            
            return preview
            
        except Exception as e:
            logger.error(f"❌ 파일 미리보기 실패: {e}")
            return {
                "error": str(e),
                "recommendations": ["파일을 직접 확인해보세요"]
            } 