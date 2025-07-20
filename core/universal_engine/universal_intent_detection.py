"""
Universal Intent Detection - 범용 의도 감지 시스템

요구사항 6에 따른 구현:
- 사전 정의된 카테고리 없는 의미 기반 라우팅
- 직접적/암묵적 의도 구분
- 의미 공간 탐색을 통한 최적 접근법 발견
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class UniversalIntentDetection:
    """
    범용 의도 감지 시스템
    - 템플릿이나 카테고리 없이 의도 파악
    - 의미 공간에서의 동적 탐색
    - 다층적 의도 이해
    """
    
    def __init__(self):
        """UniversalIntentDetection 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.intent_history = []
        self.semantic_cache = {}
        logger.info("UniversalIntentDetection initialized")
    
    async def detect_intent(self, query: str, context: Dict = None) -> Dict:
        """
        사용자 의도 감지
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트 정보
            
        Returns:
            감지된 의도 정보
        """
        logger.info("Detecting user intent without predefined categories")
        
        try:
            # 1. 쿼리 자체의 의미 분석
            semantic_analysis = await self._analyze_query_semantics(query)
            
            # 2. 명시적 의도 추출
            explicit_intent = await self._extract_explicit_intent(query, semantic_analysis)
            
            # 3. 암묵적 의도 추론
            implicit_intent = await self._infer_implicit_intent(
                query, semantic_analysis, context
            )
            
            # 4. 의미 공간 탐색
            semantic_exploration = await self._explore_semantic_space(
                explicit_intent, implicit_intent, semantic_analysis
            )
            
            # 5. 최적 응답 전략 결정
            response_strategy = await self._determine_response_strategy(
                explicit_intent, implicit_intent, semantic_exploration
            )
            
            # 결과 통합
            intent_result = {
                'query': query,
                'semantic_analysis': semantic_analysis,
                'explicit_intent': explicit_intent,
                'implicit_intent': implicit_intent,
                'semantic_exploration': semantic_exploration,
                'response_strategy': response_strategy,
                'confidence': self._calculate_intent_confidence(
                    explicit_intent, implicit_intent
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            # 이력 저장
            self.intent_history.append(intent_result)
            
            return intent_result
            
        except Exception as e:
            logger.error(f"Error in intent detection: {e}")
            raise
    
    async def _analyze_query_semantics(self, query: str) -> Dict:
        """
        쿼리의 의미론적 분석
        """
        prompt = f"""
        사전 정의된 카테고리나 패턴에 의존하지 않고 쿼리 자체가 말하는 것을 들어보겠습니다.
        
        쿼리: {query}
        
        이 쿼리의 의미를 다각도로 분석하세요:
        1. 표면적 의미
        2. 잠재적 의미
        3. 감정적 톤
        4. 긴급도
        5. 구체성 수준
        
        JSON 형식으로 응답하세요:
        {{
            "surface_meaning": "표면적으로 요청하는 것",
            "underlying_meanings": ["잠재 의미1", "잠재 의미2"],
            "emotional_tone": "neutral|positive|negative|urgent|confused",
            "urgency_level": "low|medium|high",
            "specificity": "vague|moderate|specific|very_specific",
            "key_concepts": ["핵심 개념1", "핵심 개념2"],
            "action_orientation": "exploratory|analytical|problem_solving|informational",
            "complexity_indicators": ["복잡도 지시자1", "복잡도 지시자2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _extract_explicit_intent(self, query: str, semantic_analysis: Dict) -> Dict:
        """
        명시적 의도 추출
        """
        prompt = f"""
        쿼리에서 직접적으로 표현된 의도를 추출하세요.
        
        쿼리: {query}
        의미 분석: {semantic_analysis}
        
        명시적으로 요청하는 것만 추출하세요 (추측하지 마세요).
        
        JSON 형식으로 응답하세요:
        {{
            "primary_intent": "주요 명시적 의도",
            "action_requested": "요청된 구체적 행동",
            "target_object": "대상 객체나 주제",
            "constraints": ["제약사항1", "제약사항2"],
            "expected_outcome": "예상 결과",
            "explicit_requirements": ["명시적 요구사항1", "명시적 요구사항2"],
            "clarity_level": "clear|moderate|unclear"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _infer_implicit_intent(self, query: str, semantic_analysis: Dict, context: Dict = None) -> Dict:
        """
        암묵적 의도 추론
        """
        context_info = context if context else {}
        
        prompt = f"""
        쿼리에서 직접 표현되지 않았지만 암시된 의도를 추론하세요.
        
        쿼리: {query}
        의미 분석: {semantic_analysis}
        컨텍스트: {context_info}
        
        사용자가 말하지 않았지만 원할 가능성이 있는 것들을 추론하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "hidden_needs": ["숨겨진 필요1", "숨겨진 필요2"],
            "underlying_problems": ["근본 문제1", "근본 문제2"],
            "unstated_assumptions": ["암묵적 가정1", "암묵적 가정2"],
            "potential_follow_ups": ["잠재적 후속 질문1", "잠재적 후속 질문2"],
            "emotional_drivers": ["감정적 동기1", "감정적 동기2"],
            "real_goal": "실제 목표 (추정)",
            "confidence_in_inference": "high|medium|low"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _explore_semantic_space(self, explicit_intent: Dict, implicit_intent: Dict, semantic_analysis: Dict) -> Dict:
        """
        의미 공간 탐색
        """
        prompt = f"""
        명시적 의도와 암묵적 의도를 바탕으로 의미 공간을 탐색하세요.
        
        명시적 의도: {explicit_intent}
        암묵적 의도: {implicit_intent}
        의미 분석: {semantic_analysis}
        
        다양한 해석 가능성과 접근 방법을 탐색하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "interpretation_space": [
                {{
                    "interpretation": "해석 1",
                    "likelihood": "high|medium|low",
                    "approach": "접근 방법",
                    "value": "이 해석의 가치"
                }}
            ],
            "alternative_perspectives": [
                {{
                    "perspective": "관점 1",
                    "insight": "이 관점에서의 통찰",
                    "relevance": "high|medium|low"
                }}
            ],
            "conceptual_connections": [
                {{
                    "from_concept": "개념 A",
                    "to_concept": "개념 B",
                    "relationship": "관계 설명",
                    "strength": "strong|moderate|weak"
                }}
            ],
            "exploration_depth": "shallow|moderate|deep",
            "most_promising_direction": "가장 유망한 방향"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _determine_response_strategy(self, explicit_intent: Dict, implicit_intent: Dict, semantic_exploration: Dict) -> Dict:
        """
        최적 응답 전략 결정
        """
        prompt = f"""
        발견된 의도들을 바탕으로 최적의 응답 전략을 결정하세요.
        
        명시적 의도: {explicit_intent}
        암묵적 의도: {implicit_intent}
        의미 공간 탐색: {semantic_exploration}
        
        사전 정의된 템플릿이 아닌, 이 특정 상황에 맞는 전략을 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "primary_strategy": {{
                "approach": "응답 접근법",
                "rationale": "이 접근법을 선택한 이유",
                "key_elements": ["핵심 요소1", "핵심 요소2"]
            }},
            "secondary_strategies": [
                {{
                    "approach": "보조 접근법",
                    "when_to_use": "사용 시점",
                    "benefit": "기대 효과"
                }}
            ],
            "content_structure": {{
                "opening": "시작 방식",
                "main_body": "본문 구조",
                "closing": "마무리 방식"
            }},
            "interaction_style": {{
                "tone": "대화 톤",
                "engagement_level": "참여 수준",
                "personalization": "개인화 정도"
            }},
            "adaptation_triggers": [
                {{
                    "condition": "조건",
                    "action": "조치"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    def _calculate_intent_confidence(self, explicit_intent: Dict, implicit_intent: Dict) -> float:
        """
        의도 감지 신뢰도 계산
        """
        confidence = 0.5  # 기본값
        
        # 명시적 의도의 명확성
        if explicit_intent.get('clarity_level') == 'clear':
            confidence += 0.3
        elif explicit_intent.get('clarity_level') == 'moderate':
            confidence += 0.15
        
        # 암묵적 의도의 신뢰도
        if implicit_intent.get('confidence_in_inference') == 'high':
            confidence += 0.2
        elif implicit_intent.get('confidence_in_inference') == 'medium':
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def refine_intent_with_clarification(self, original_intent: Dict, clarification: str) -> Dict:
        """
        명확화 응답을 통한 의도 개선
        """
        prompt = f"""
        사용자의 명확화 응답을 바탕으로 원래 의도 분석을 개선하세요.
        
        원래 의도 분석: {original_intent}
        사용자 명확화: {clarification}
        
        JSON 형식으로 개선된 의도 분석을 제공하세요:
        {{
            "refined_intent": "개선된 의도 이해",
            "corrections": ["수정사항1", "수정사항2"],
            "new_insights": ["새로운 통찰1", "새로운 통찰2"],
            "confidence_improvement": 0.0-1.0,
            "remaining_ambiguities": ["남은 모호함1", "남은 모호함2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        refined = self._parse_json_response(response)
        
        # 원래 의도와 병합
        updated_intent = {**original_intent}
        updated_intent['refinements'] = refined
        updated_intent['confidence'] = min(
            original_intent.get('confidence', 0.5) + refined.get('confidence_improvement', 0),
            1.0
        )
        
        return updated_intent
    
    async def analyze_intent_patterns(self, time_window: int = 10) -> Dict:
        """
        최근 의도 패턴 분석
        """
        recent_intents = self.intent_history[-time_window:] if len(self.intent_history) > time_window else self.intent_history
        
        if not recent_intents:
            return {'message': 'No intent history available'}
        
        prompt = f"""
        최근 사용자 의도들의 패턴을 분석하세요.
        
        의도 수: {len(recent_intents)}
        주요 의도들: {[intent.get('explicit_intent', {}).get('primary_intent', '') for intent in recent_intents]}
        
        JSON 형식으로 패턴 분석을 제공하세요:
        {{
            "common_themes": ["공통 주제1", "공통 주제2"],
            "intent_evolution": "의도 변화 패턴",
            "user_journey": "사용자 여정 설명",
            "predicted_next_intents": ["예상 다음 의도1", "예상 다음 의도2"],
            "engagement_pattern": "참여 패턴",
            "recommendations": ["추천사항1", "추천사항2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    def get_intent_statistics(self) -> Dict:
        """
        의도 감지 통계 조회
        """
        if not self.intent_history:
            return {'message': 'No intent history available'}
        
        stats = {
            'total_intents_analyzed': len(self.intent_history),
            'average_confidence': sum(i.get('confidence', 0) for i in self.intent_history) / len(self.intent_history),
            'clarity_distribution': self._calculate_clarity_distribution(),
            'action_orientation_distribution': self._calculate_action_distribution(),
            'most_common_implicit_needs': self._get_common_implicit_needs()
        }
        
        return stats
    
    def _calculate_clarity_distribution(self) -> Dict:
        """명확성 분포 계산"""
        distribution = {'clear': 0, 'moderate': 0, 'unclear': 0}
        
        for intent in self.intent_history:
            clarity = intent.get('explicit_intent', {}).get('clarity_level', 'unclear')
            if clarity in distribution:
                distribution[clarity] += 1
                
        return distribution
    
    def _calculate_action_distribution(self) -> Dict:
        """행동 지향성 분포 계산"""
        distribution = {}
        
        for intent in self.intent_history:
            action = intent.get('semantic_analysis', {}).get('action_orientation', 'unknown')
            distribution[action] = distribution.get(action, 0) + 1
            
        return distribution
    
    def _get_common_implicit_needs(self, top_n: int = 5) -> List[str]:
        """공통 암묵적 필요 추출"""
        all_needs = []
        
        for intent in self.intent_history:
            needs = intent.get('implicit_intent', {}).get('hidden_needs', [])
            all_needs.extend(needs)
        
        # 빈도 계산
        need_counts = {}
        for need in all_needs:
            need_counts[need] = need_counts.get(need, 0) + 1
        
        # 상위 N개 반환
        sorted_needs = sorted(need_counts.items(), key=lambda x: x[1], reverse=True)
        return [need for need, count in sorted_needs[:top_n]]
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        import json
        try:
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