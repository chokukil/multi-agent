"""
인코딩 감지기 (Encoding Detector)

UTF-8 인코딩 문제를 해결하기 위한 지능형 인코딩 감지 유틸리티
pandas_agent의 안정성을 기준으로 다중 인코딩 시도 패턴 구현
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import chardet

logger = logging.getLogger(__name__)


class EncodingDetector:
    """
    지능형 인코딩 감지기
    
    UTF-8 인코딩 문제를 해결하기 위해 다중 인코딩을 시도하고
    최적의 인코딩을 자동으로 감지하는 유틸리티
    """
    
    def __init__(self):
        # 우선순위 기반 인코딩 리스트 (한국어 환경 최적화)
        self.encoding_priority = [
            'utf-8',
            'cp949',      # Windows 한국어
            'euc-kr',     # Unix/Linux 한국어
            'utf-8-sig',  # BOM이 있는 UTF-8
            'latin1',     # 서유럽
            'iso-8859-1', # 라틴-1
            'ascii',      # 기본 ASCII
            'utf-16',     # Unicode 16비트
            'utf-32'      # Unicode 32비트
        ]
        
        logger.info("✅ EncodingDetector 초기화 완료")
    
    async def detect_encoding(self, file_path: str, sample_size: int = 8192) -> str:
        """
        파일 인코딩 자동 감지
        
        Args:
            file_path: 파일 경로
            sample_size: 샘플링할 바이트 크기
            
        Returns:
            str: 감지된 인코딩 (기본: utf-8)
        """
        try:
            # 1단계: chardet 라이브러리를 사용한 자동 감지
            detected_encoding = await self._detect_with_chardet(file_path, sample_size)
            
            if detected_encoding:
                # 감지된 인코딩으로 실제 읽기 테스트
                if await self._test_encoding(file_path, detected_encoding):
                    logger.info(f"✅ 인코딩 자동 감지 성공: {detected_encoding}")
                    return detected_encoding
            
            # 2단계: 우선순위 기반 순차 시도
            for encoding in self.encoding_priority:
                if await self._test_encoding(file_path, encoding):
                    logger.info(f"✅ 인코딩 순차 시도 성공: {encoding}")
                    return encoding
            
            # 3단계: 폴백 - UTF-8 강제 사용
            logger.warning(f"⚠️ 인코딩 감지 실패, UTF-8 폴백 사용: {file_path}")
            return 'utf-8'
            
        except Exception as e:
            logger.error(f"❌ 인코딩 감지 중 오류: {e}")
            return 'utf-8'
    
    async def _detect_with_chardet(self, file_path: str, sample_size: int) -> Optional[str]:
        """chardet 라이브러리를 사용한 인코딩 감지"""
        try:
            with open(file_path, 'rb') as file:
                # 파일 샘플 읽기
                sample = file.read(sample_size)
                
                if not sample:
                    return None
                
                # chardet으로 인코딩 감지
                result = chardet.detect(sample)
                
                if result and result.get('confidence', 0) > 0.7:
                    encoding = result['encoding']
                    
                    # 일반적인 인코딩 이름으로 정규화
                    encoding = self._normalize_encoding_name(encoding)
                    
                    logger.info(f"🔍 chardet 감지: {encoding} (신뢰도: {result['confidence']:.2f})")
                    return encoding
                
        except Exception as e:
            logger.debug(f"chardet 감지 실패: {e}")
            
        return None
    
    def _normalize_encoding_name(self, encoding: str) -> str:
        """인코딩 이름 정규화"""
        if not encoding:
            return 'utf-8'
        
        encoding_lower = encoding.lower()
        
        # 일반적인 인코딩 이름 매핑
        encoding_mapping = {
            'cp949': 'cp949',
            'euc-kr': 'euc-kr',
            'ks_c_5601-1987': 'cp949',
            'windows-1252': 'latin1',
            'iso-8859-1': 'latin1',
            'ascii': 'ascii',
            'utf-8': 'utf-8',
            'utf-16': 'utf-16',
            'utf-32': 'utf-32'
        }
        
        for key, value in encoding_mapping.items():
            if key in encoding_lower:
                return value
        
        return encoding
    
    async def _test_encoding(self, file_path: str, encoding: str, test_lines: int = 10) -> bool:
        """특정 인코딩으로 파일 읽기 테스트"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # 처음 몇 줄만 읽어서 테스트
                for i, line in enumerate(file):
                    if i >= test_lines:
                        break
                    
                    # 읽기 성공하면 True
                    if line:
                        pass
            
            return True
            
        except (UnicodeDecodeError, UnicodeError, LookupError):
            return False
        except Exception as e:
            logger.debug(f"인코딩 테스트 실패 {encoding}: {e}")
            return False
    
    async def get_encoding_candidates(self, file_path: str) -> List[Dict[str, Any]]:
        """
        가능한 인코딩 후보들과 신뢰도 반환
        
        Args:
            file_path: 파일 경로
            
        Returns:
            List[Dict]: 인코딩 후보 정보 리스트
        """
        candidates = []
        
        # chardet으로 감지된 인코딩 추가
        detected = await self._detect_with_chardet(file_path, 8192)
        if detected:
            candidates.append({
                "encoding": detected,
                "method": "chardet",
                "confidence": 0.8,
                "tested": await self._test_encoding(file_path, detected)
            })
        
        # 우선순위 기반 인코딩들 테스트
        for encoding in self.encoding_priority:
            if encoding != detected:  # 중복 제거
                tested = await self._test_encoding(file_path, encoding)
                candidates.append({
                    "encoding": encoding,
                    "method": "priority",
                    "confidence": 0.6 if tested else 0.2,
                    "tested": tested
                })
        
        # 신뢰도 기준 내림차순 정렬
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return candidates
    
    async def detect_with_fallback(self, file_path: str, preferred_encodings: List[str] = None) -> str:
        """
        폴백 전략을 포함한 인코딩 감지
        
        Args:
            file_path: 파일 경로
            preferred_encodings: 우선 시도할 인코딩들
            
        Returns:
            str: 감지된 인코딩
        """
        try:
            # 사용자 지정 인코딩 우선 시도
            if preferred_encodings:
                for encoding in preferred_encodings:
                    if await self._test_encoding(file_path, encoding):
                        logger.info(f"✅ 선호 인코딩 성공: {encoding}")
                        return encoding
            
            # 기본 감지 로직 실행
            return await self.detect_encoding(file_path)
            
        except Exception as e:
            logger.error(f"❌ 폴백 인코딩 감지 실패: {e}")
            return 'utf-8'
    
    async def analyze_file_encoding_issues(self, file_path: str) -> Dict[str, Any]:
        """
        파일의 인코딩 문제 분석
        
        Args:
            file_path: 파일 경로
            
        Returns:
            Dict: 인코딩 분석 결과
        """
        try:
            analysis = {
                "file_path": file_path,
                "detected_encoding": None,
                "working_encodings": [],
                "failed_encodings": [],
                "has_bom": False,
                "file_size": 0,
                "recommendations": []
            }
            
            # 파일 크기 확인
            file_obj = Path(file_path)
            if file_obj.exists():
                analysis["file_size"] = file_obj.stat().st_size
            
            # BOM 확인
            analysis["has_bom"] = await self._check_bom(file_path)
            
            # 모든 인코딩 테스트
            for encoding in self.encoding_priority:
                if await self._test_encoding(file_path, encoding):
                    analysis["working_encodings"].append(encoding)
                else:
                    analysis["failed_encodings"].append(encoding)
            
            # 최적 인코딩 감지
            analysis["detected_encoding"] = await self.detect_encoding(file_path)
            
            # 권장사항 생성
            analysis["recommendations"] = self._generate_encoding_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 인코딩 분석 실패: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "recommendations": ["파일을 UTF-8로 다시 저장해보세요."]
            }
    
    async def _check_bom(self, file_path: str) -> bool:
        """BOM(Byte Order Mark) 확인"""
        try:
            with open(file_path, 'rb') as file:
                # BOM 시그니처 확인
                bom_signatures = [
                    b'\xef\xbb\xbf',      # UTF-8 BOM
                    b'\xff\xfe',          # UTF-16 LE BOM
                    b'\xfe\xff',          # UTF-16 BE BOM
                    b'\xff\xfe\x00\x00',  # UTF-32 LE BOM
                    b'\x00\x00\xfe\xff'   # UTF-32 BE BOM
                ]
                
                first_bytes = file.read(4)
                
                for bom in bom_signatures:
                    if first_bytes.startswith(bom):
                        return True
                
                return False
                
        except Exception:
            return False
    
    def _generate_encoding_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """인코딩 분석 결과를 바탕으로 권장사항 생성"""
        recommendations = []
        
        working_count = len(analysis.get("working_encodings", []))
        
        if working_count == 0:
            recommendations.append("⚠️ 지원되는 인코딩이 없습니다. 파일이 손상되었을 가능성이 있습니다.")
            recommendations.append("💡 파일을 텍스트 에디터에서 UTF-8로 다시 저장해보세요.")
        
        elif working_count == 1:
            encoding = analysis["working_encodings"][0]
            recommendations.append(f"✅ {encoding} 인코딩을 사용하는 것이 안전합니다.")
        
        elif working_count > 1:
            recommendations.append("⚠️ 여러 인코딩이 가능합니다. 데이터 정확성을 위해 검증이 필요합니다.")
            recommendations.append(f"💡 권장 인코딩: {analysis.get('detected_encoding', 'utf-8')}")
        
        if analysis.get("has_bom"):
            recommendations.append("📝 파일에 BOM이 포함되어 있습니다. utf-8-sig 인코딩을 고려해보세요.")
        
        file_size_mb = analysis.get("file_size", 0) / (1024 * 1024)
        if file_size_mb > 100:
            recommendations.append(f"📊 대용량 파일({file_size_mb:.1f}MB)입니다. 청크 단위 읽기를 권장합니다.")
        
        return recommendations
    
    def get_encoding_priority(self) -> List[str]:
        """인코딩 우선순위 반환"""
        return self.encoding_priority.copy()
    
    async def convert_file_encoding(self, source_path: str, target_path: str, 
                                  source_encoding: str, target_encoding: str = 'utf-8') -> bool:
        """
        파일 인코딩 변환 (주의: 원본 파일 수정됨)
        
        Args:
            source_path: 원본 파일 경로
            target_path: 대상 파일 경로
            source_encoding: 원본 인코딩
            target_encoding: 대상 인코딩
            
        Returns:
            bool: 변환 성공 여부
        """
        try:
            # 원본 파일 읽기
            with open(source_path, 'r', encoding=source_encoding) as source_file:
                content = source_file.read()
            
            # 대상 인코딩으로 저장
            with open(target_path, 'w', encoding=target_encoding) as target_file:
                target_file.write(content)
            
            logger.info(f"✅ 인코딩 변환 완료: {source_encoding} → {target_encoding}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 인코딩 변환 실패: {e}")
            return False 