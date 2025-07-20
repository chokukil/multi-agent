"""
Meta-Reasoning Engine - 메타 추론 엔진

요구사항 2에 따른 DeepSeek-R1 영감을 받은 메타 추론 시스템
- 생각에 대해 생각하기
- 4단계 추론: 초기 관찰 → 다각도 분석 → 자가 검증 → 적응적 응답
- 메타 보상 패턴으로 분석 품질 평가
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MetaReasoningEngine:
    """
    메타 추론 엔진 - 생각에 대해 생각하기
    DeepSeek-R1 영감을 받은 자가 반성 추론 시스템
    """
    
    def __init__(self):
        """MetaReasoningEngine 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.reasoning_patterns = {
            'self_reflection': self._load_self_reflection_pattern(),
            'meta_rewarding': self._load_meta_rewarding_pattern(),
            'chain_of_thought': self._load_chain_of_thought_pattern(),
            'zero_shot_adaptive': self._load_zero_shot_pattern()
        }
        logger.info("MetaReasoningEngine initialized with DeepSeek-R1 patterns")
    
    def _load_self_reflection_pattern(self) -> str:
        """자가 반성 추론 패턴 로드"""
        return """
        # 자가 반성 추론 패턴
        당신은 주어진 쿼리와 데이터를 분석하는 전문가입니다.
        
        단계 1: 초기 관찰
        - 데이터를 보고 무엇을 발견하는가?
        - 사용자 쿼리의 진정한 의도는?
        - 내가 놓치고 있는 것은 없는가?
        
        단계 2: 다각도 분석
        - 이 문제를 다른 방식으로 접근한다면?
        - 사용자가 전문가라면 어떤 답을 원할까?
        - 사용자가 초보자라면 어떤 도움이 필요할까?
        
        단계 3: 자가 검증
        - 내 분석이 논리적으로 일관성이 있는가?
        - 사용자에게 실제로 도움이 되는가?
        - 확신이 없는 부분은 무엇인가?
        
        단계 4: 적응적 응답
        - 확실한 부분은 명확히 제시
        - 불확실한 부분은 명확화 질문
        - 사용자 수준에 맞는 설명 깊이 조절
        """
    
    def _load_meta_rewarding_pattern(self) -> str:
        """메타 보상 패턴 로드"""
        return """
        # 자가 평가 및 개선 패턴
        내 분석을 스스로 평가해보겠습니다:
        
        평가 기준:
        1. 정확성: 분석이 데이터를 올바르게 해석했는가?
        2. 완전성: 중요한 인사이트를 놓치지 않았는가?
        3. 적절성: 사용자 수준과 요구에 맞는가?
        4. 명확성: 설명이 이해하기 쉬운가?
        5. 실용성: 실제로 도움이 되는 조치를 제안했는가?
        
        개선점:
        - 부족한 부분은 무엇인가?
        - 어떻게 더 나은 분석을 할 수 있는가?
        - 사용자에게 추가로 필요한 정보는?
        
        이 평가를 바탕으로 응답을 개선하겠습니다.
        """
    
    def _load_chain_of_thought_pattern(self) -> str:
        """Chain of Thought 패턴 로드"""
        return """
        문제를 단계별로 생각해보겠습니다:
        
        1. 문제 정의: 정확히 무엇을 해결해야 하는가?
        2. 가능한 접근법: 어떤 방법들이 있는가?
        3. 각 접근법의 장단점: 무엇이 최선인가?
        4. 선택한 접근법 실행: 구체적으로 어떻게?
        5. 결과 검증: 올바른 결과인가?
        """
    
    def _load_zero_shot_pattern(self) -> str:
        """Zero-shot 적응형 추론 패턴 로드"""
        return """
        템플릿이나 공식에 의존하지 않고 문제 자체의 본질에 맞는 추론을 수행하겠습니다.
        
        문제 공간 정의:
        - 주어진 정보는 무엇인가?
        - 요구되는 결과는 무엇인가?
        - 제약사항은 무엇인가?
        
        추론 전략 수립:
        - 이 특정 문제에 가장 적합한 사고 방식은?
        - 어떤 관점에서 접근해야 하는가?
        
        단계별 추론 실행:
        - 각 단계의 논리적 연결성 확보
        - 가정과 제약사항 명시
        - 불확실성과 신뢰도 평가
        
        결과 통합 및 검증:
        - 모든 요구사항이 충족되었는가?
        - 논리적 일관성이 있는가?
        - 실용적 가치가 있는가?
        """
    
    async def analyze_request(self, query: str, data: Any, context: Dict) -> Dict:
        """
        요구사항 2에 따른 메타 추론 분석
        - DeepSeek-R1 영감 4단계 추론
        - 자가 평가 및 개선
        - 동적 전략 선택
        
        Args:
            query: 사용자 쿼리
            data: 분석 대상 데이터
            context: 추가 컨텍스트
            
        Returns:
            메타 추론 분석 결과
        """
        logger.info("Starting meta-reasoning analysis")
        
        try:
            # 1단계: 초기 관찰 및 분석
            initial_analysis = await self._perform_initial_observation(query, data)
            
            # 2단계: 다각도 분석
            multi_perspective_analysis = await self._perform_multi_perspective_analysis(
                initial_analysis, query, data
            )
            
            # 3단계: 자가 검증
            self_verification = await self._perform_self_verification(
                multi_perspective_analysis
            )
            
            # 4단계: 적응적 응답 전략 결정
            response_strategy = await self._determine_adaptive_strategy(
                self_verification, context
            )
            
            # 메타 보상 패턴으로 전체 분석 품질 평가
            quality_assessment = await self._assess_analysis_quality(response_strategy)
            
            return {
                'initial_analysis': initial_analysis,
                'multi_perspective': multi_perspective_analysis,
                'self_verification': self_verification,
                'response_strategy': response_strategy,
                'quality_assessment': quality_assessment,
                'confidence_level': quality_assessment.get('confidence', 0.0),
                'user_profile': response_strategy.get('estimated_user_profile', {}),
                'domain_context': initial_analysis.get('domain_context', {}),
                'data_characteristics': initial_analysis.get('data_characteristics', {})
            }
            
        except Exception as e:
            logger.error(f"Error in meta-reasoning analysis: {e}")
            raise
    
    async def _perform_initial_observation(self, query: str, data: Any) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 1: 초기 관찰
        """
        observation_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        쿼리: {query}
        데이터 특성: {self._analyze_data_characteristics(data)}
        
        단계 1: 초기 관찰
        - 데이터를 보고 무엇을 발견하는가?
        - 사용자 쿼리의 진정한 의도는?
        - 내가 놓치고 있는 것은 없는가?
        
        사전 정의된 카테고리나 패턴에 의존하지 말고,
        순수하게 관찰된 것만을 바탕으로 분석하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "data_observations": "관찰된 데이터 특성",
            "query_intent": "파악된 사용자 의도",
            "potential_blind_spots": "놓칠 수 있는 부분",
            "domain_context": "감지된 도메인 컨텍스트",
            "data_characteristics": {{
                "type": "데이터 유형",
                "structure": "데이터 구조",
                "patterns": "발견된 패턴"
            }}
        }}
        """
        
        response = await self.llm_client.agenerate(observation_prompt)
        return self._parse_json_response(response)
    
    async def _perform_multi_perspective_analysis(self, initial_analysis: Dict, query: str, data: Any) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 2: 다각도 분석
        """
        multi_perspective_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        초기 관찰 결과: {json.dumps(initial_analysis, ensure_ascii=False)}
        
        단계 2: 다각도 분석
        - 이 문제를 다른 방식으로 접근한다면?
        - 사용자가 전문가라면 어떤 답을 원할까?
        - 사용자가 초보자라면 어떤 도움이 필요할까?
        
        각 관점에서의 분석을 수행하고,
        사용자 수준 추정과 최적 접근법을 제시하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "alternative_approaches": ["접근법1", "접근법2", "접근법3"],
            "expert_perspective": {{
                "expectations": "전문가가 원하는 것",
                "technical_depth": "필요한 기술적 깊이",
                "key_insights": ["인사이트1", "인사이트2"]
            }},
            "beginner_perspective": {{
                "needs": "초보자가 필요로 하는 것",
                "simplifications": "단순화할 부분",
                "guidance": "제공할 가이드"
            }},
            "estimated_user_level": "beginner|intermediate|expert|unknown",
            "recommended_approach": "추천하는 접근법"
        }}
        """
        
        response = await self.llm_client.agenerate(multi_perspective_prompt)
        return self._parse_json_response(response)
    
    async def _perform_self_verification(self, multi_perspective_analysis: Dict) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 3: 자가 검증
        """
        verification_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        다각도 분석 결과: {json.dumps(multi_perspective_analysis, ensure_ascii=False)}
        
        단계 3: 자가 검증
        - 내 분석이 논리적으로 일관성이 있는가?
        - 사용자에게 실제로 도움이 되는가?
        - 확신이 없는 부분은 무엇인가?
        
        분석의 강점과 약점을 솔직하게 평가하고,
        불확실한 부분은 명확히 식별하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "logical_consistency": {{
                "is_consistent": true/false,
                "inconsistencies": ["불일치1", "불일치2"]
            }},
            "practical_value": {{
                "is_helpful": true/false,
                "value_points": ["가치1", "가치2"],
                "limitations": ["한계1", "한계2"]
            }},
            "uncertainties": {{
                "high_confidence_areas": ["확실한 부분1", "확실한 부분2"],
                "low_confidence_areas": ["불확실한 부분1", "불확실한 부분2"],
                "clarification_needed": ["명확화 필요1", "명확화 필요2"]
            }},
            "overall_confidence": 0.0-1.0
        }}
        """
        
        response = await self.llm_client.agenerate(verification_prompt)
        return self._parse_json_response(response)
    
    async def _determine_adaptive_strategy(self, self_verification: Dict, context: Dict) -> Dict:
        """
        요구사항 2의 자가 반성 추론 패턴 - 단계 4: 적응적 응답
        """
        strategy_prompt = f"""
        {self.reasoning_patterns['self_reflection']}
        
        자가 검증 결과: {json.dumps(self_verification, ensure_ascii=False)}
        상호작용 컨텍스트: {json.dumps(context, ensure_ascii=False)}
        
        단계 4: 적응적 응답
        - 확실한 부분은 명확히 제시
        - 불확실한 부분은 명확화 질문
        - 사용자 수준에 맞는 설명 깊이 조절
        
        최적의 응답 전략을 결정하고,
        사용자 프로필과 상호작용 방식을 제안하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "response_strategy": {{
                "approach": "직접적|점진적|대화형",
                "explanation_depth": "shallow|medium|deep",
                "technical_level": "low|medium|high",
                "interaction_style": "formal|casual|educational"
            }},
            "content_structure": {{
                "present_confidently": ["확실한 내용1", "확실한 내용2"],
                "seek_clarification": ["명확화 질문1", "명확화 질문2"],
                "progressive_disclosure": ["단계1", "단계2", "단계3"]
            }},
            "estimated_user_profile": {{
                "expertise_level": "beginner|intermediate|expert",
                "learning_style": "visual|textual|example-based",
                "domain_familiarity": "low|medium|high"
            }},
            "follow_up_recommendations": ["추천1", "추천2", "추천3"]
        }}
        """
        
        response = await self.llm_client.agenerate(strategy_prompt)
        return self._parse_json_response(response)
    
    async def _assess_analysis_quality(self, response_strategy: Dict) -> Dict:
        """
        요구사항 2의 메타 보상 패턴으로 분석 품질 평가
        """
        quality_prompt = f"""
        {self.reasoning_patterns['meta_rewarding']}
        
        분석 내용: {json.dumps(response_strategy, ensure_ascii=False)}
        
        평가 기준:
        1. 정확성: 분석이 데이터를 올바르게 해석했는가?
        2. 완전성: 중요한 인사이트를 놓치지 않았는가?
        3. 적절성: 사용자 수준과 요구에 맞는가?
        4. 명확성: 설명이 이해하기 쉬운가?
        5. 실용성: 실제로 도움이 되는 조치를 제안했는가?
        
        JSON 형식으로 평가 결과를 제공하세요:
        {{
            "accuracy_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "appropriateness_score": 0.0-1.0,
            "clarity_score": 0.0-1.0,
            "practicality_score": 0.0-1.0,
            "overall_quality": 0.0-1.0,
            "confidence": 0.0-1.0,
            "improvements_needed": ["개선점1", "개선점2"],
            "strengths": ["강점1", "강점2"],
            "next_steps": ["다음 단계1", "다음 단계2"]
        }}
        """
        
        response = await self.llm_client.agenerate(quality_prompt)
        return self._parse_json_response(response)
    
    def _analyze_data_characteristics(self, data: Any) -> str:
        """데이터 특성 분석"""
        characteristics = {
            'type': type(data).__name__,
            'size': self._get_data_size(data),
            'structure': self._get_data_structure(data)
        }
        return json.dumps(characteristics, ensure_ascii=False)
    
    def _get_data_size(self, data: Any) -> str:
        """데이터 크기 확인"""
        try:
            if hasattr(data, '__len__'):
                return f"{len(data)} items"
            elif hasattr(data, 'shape'):
                return f"shape: {data.shape}"
            else:
                return "unknown size"
        except:
            return "size calculation failed"
    
    def _get_data_structure(self, data: Any) -> str:
        """데이터 구조 확인"""
        try:
            if hasattr(data, 'columns'):
                return f"columns: {list(data.columns)[:5]}..."
            elif isinstance(data, dict):
                return f"keys: {list(data.keys())[:5]}..."
            elif isinstance(data, list):
                return f"list of {type(data[0]).__name__ if data else 'empty'}"
            else:
                return type(data).__name__
        except:
            return "structure analysis failed"
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'raw_response': response,
                'parse_error': str(e)
            }