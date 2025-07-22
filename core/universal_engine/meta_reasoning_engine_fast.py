"""
Fast Meta-Reasoning Engine - 성능 최적화된 메타 추론 엔진

원본 대비 최적화 사항:
- 4단계 → 2단계로 축소
- 프롬프트 길이 80% 단축
- JSON 구조 단순화
- 불필요한 설명 제거
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class FastMetaReasoningEngine:
    """
    성능 최적화된 메타 추론 엔진
    - 2단계 추론: 분석 → 전략
    - 단순화된 프롬프트
    - 빠른 응답 보장
    """
    
    def __init__(self):
        """FastMetaReasoningEngine 초기화"""
        self.llm_client = LLMFactory.create_llm()
        logger.info("FastMetaReasoningEngine initialized")
    
    async def analyze_request(self, query: str, data: Any, context: Dict) -> Dict:
        """
        빠른 메타 추론 분석 (2단계)
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트
            
        Returns:
            메타 추론 분석 결과
        """
        logger.info("Starting fast meta-reasoning analysis")
        
        try:
            # 단계 1: 빠른 분석
            analysis = await self._quick_analysis(query, data)
            
            # 단계 2: 응답 전략
            strategy = await self._response_strategy(analysis, context)
            
            return {
                'analysis': analysis,
                'strategy': strategy,
                'confidence_level': strategy.get('confidence', 0.7),
                'user_profile': strategy.get('user_profile', {}),
                'domain_context': analysis.get('domain', 'general'),
                'data_characteristics': self._simple_data_characteristics(data)
            }
            
        except Exception as e:
            logger.error(f"Error in fast meta-reasoning: {e}")
            raise
    
    async def _quick_analysis(self, query: str, data: Any) -> Dict:
        """단계 1: 빠른 분석"""
        
        prompt = f"""분석 요청:
쿼리: {query}
데이터: {self._simple_data_characteristics(data)}

다음을 간단히 분석하세요:
1. 사용자 의도
2. 데이터 특성
3. 필요한 접근법

JSON 응답:
{{
    "intent": "사용자 의도",
    "domain": "도메인",
    "approach": "추천 접근법"
}}"""
        
        response = await self.llm_client.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return self._parse_json_safe(content)
    
    async def _response_strategy(self, analysis: Dict, context: Dict) -> Dict:
        """단계 2: 응답 전략"""
        
        prompt = f"""분석 결과: {json.dumps(analysis, ensure_ascii=False)}

사용자에게 어떻게 응답할지 결정하세요:
1. 설명 깊이 (shallow/deep)
2. 사용자 수준 (beginner/expert) 
3. 신뢰도 (0.0-1.0)

JSON 응답:
{{
    "depth": "shallow|deep",
    "user_level": "beginner|expert",
    "confidence": 0.8,
    "user_profile": {{"expertise": "beginner"}}
}}"""
        
        response = await self.llm_client.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return self._parse_json_safe(content)
    
    def _simple_data_characteristics(self, data: Any) -> str:
        """간단한 데이터 특성 분석"""
        try:
            if hasattr(data, '__len__'):
                return f"{type(data).__name__} with {len(data)} items"
            else:
                return type(data).__name__
        except:
            return "unknown data"
    
    def _parse_json_safe(self, response: str) -> Dict:
        """안전한 JSON 파싱"""
        try:
            return json.loads(response.strip())
        except:
            # JSON 파싱 실패 시 기본값 반환
            return {
                "intent": "analysis",
                "domain": "general", 
                "approach": "standard",
                "depth": "shallow",
                "user_level": "beginner",
                "confidence": 0.5,
                "user_profile": {"expertise": "beginner"}
            }