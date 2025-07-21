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
    
    async def analyze_semantic_space(self, query: str) -> Dict[str, Any]:
        """
        의미 공간 분석 및 탐색
        
        요구사항 1.1에 따른 구현:
        - 의미 기반 라우팅
        - 다층적 의미 분석
        - 동적 의미 공간 탐색
        
        Args:
            query: 분석할 쿼리
            
        Returns:
            의미 공간 분석 결과
        """
        logger.info("Analyzing semantic space for query")
        
        try:
            # 1. 기본 의미 분석
            basic_semantics = await self._analyze_query_semantics(query)
            
            # 2. 의미 벡터 공간 탐색
            semantic_vectors = await self._explore_semantic_vectors(query, basic_semantics)
            
            # 3. 개념적 연결망 분석
            conceptual_network = await self._analyze_conceptual_connections(query, basic_semantics)
            
            # 4. 의미 계층 구조 분석
            semantic_hierarchy = await self._analyze_semantic_hierarchy(query, basic_semantics)
            
            # 5. 컨텍스트 의존적 의미 분석
            contextual_meanings = await self._analyze_contextual_meanings(query, basic_semantics)
            
            # 6. 의미 공간 내 최적 경로 탐색
            optimal_paths = await self._find_optimal_semantic_paths(
                semantic_vectors, conceptual_network, semantic_hierarchy
            )
            
            # 통합 결과 생성
            semantic_space_analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'query': query,
                'basic_semantics': basic_semantics,
                'semantic_vectors': semantic_vectors,
                'conceptual_network': conceptual_network,
                'semantic_hierarchy': semantic_hierarchy,
                'contextual_meanings': contextual_meanings,
                'optimal_paths': optimal_paths,
                'semantic_richness': self._calculate_semantic_richness(
                    semantic_vectors, conceptual_network
                ),
                'interpretation_confidence': self._calculate_interpretation_confidence(
                    basic_semantics, contextual_meanings
                ),
                'recommended_approaches': await self._recommend_semantic_approaches(
                    optimal_paths, semantic_hierarchy
                )
            }
            
            return semantic_space_analysis
            
        except Exception as e:
            logger.error(f"Error in semantic space analysis: {e}")
            return {
                'error': str(e),
                'fallback_analysis': await self._fallback_semantic_analysis(query),
                'timestamp': datetime.now().isoformat()
            }
    
    async def clarify_ambiguity(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        모호성 해결 및 명확화 질문 생성
        
        요구사항 1.1에 따른 구현:
        - 모호성 감지 및 분류
        - 명확화 질문 생성
        - 다중 해석 처리
        
        Args:
            query: 모호한 쿼리
            context: 컨텍스트 정보
            
        Returns:
            모호성 해결 결과 및 명확화 질문
        """
        logger.info("Clarifying ambiguity in query")
        
        try:
            # 1. 모호성 감지 및 분류
            ambiguity_analysis = await self._detect_and_classify_ambiguity(query, context)
            
            # 2. 다중 해석 생성
            multiple_interpretations = await self._generate_multiple_interpretations(
                query, ambiguity_analysis, context
            )
            
            # 3. 명확화 질문 생성
            clarification_questions = await self._generate_clarification_questions(
                ambiguity_analysis, multiple_interpretations
            )
            
            # 4. 해석 우선순위 결정
            interpretation_priorities = await self._prioritize_interpretations(
                multiple_interpretations, context
            )
            
            # 5. 점진적 명확화 전략 수립
            progressive_clarification = await self._develop_progressive_clarification_strategy(
                clarification_questions, interpretation_priorities
            )
            
            # 6. 불확실성 처리 방안 제시
            uncertainty_handling = await self._suggest_uncertainty_handling(
                ambiguity_analysis, multiple_interpretations
            )
            
            # 통합 결과 생성
            clarification_result = {
                'clarification_timestamp': datetime.now().isoformat(),
                'original_query': query,
                'context': context,
                'ambiguity_analysis': ambiguity_analysis,
                'multiple_interpretations': multiple_interpretations,
                'clarification_questions': clarification_questions,
                'interpretation_priorities': interpretation_priorities,
                'progressive_clarification': progressive_clarification,
                'uncertainty_handling': uncertainty_handling,
                'ambiguity_score': self._calculate_ambiguity_score(ambiguity_analysis),
                'clarification_urgency': self._assess_clarification_urgency(
                    ambiguity_analysis, multiple_interpretations
                ),
                'recommended_next_steps': await self._recommend_clarification_next_steps(
                    progressive_clarification, uncertainty_handling
                )
            }
            
            return clarification_result
            
        except Exception as e:
            logger.error(f"Error in ambiguity clarification: {e}")
            return {
                'error': str(e),
                'fallback_clarification': await self._fallback_clarification(query),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _explore_semantic_vectors(self, query: str, basic_semantics: Dict) -> Dict:
        """의미 벡터 공간 탐색"""
        vector_prompt = f"""
        쿼리의 의미를 다차원 벡터 공간에서 탐색하세요.
        
        쿼리: {query}
        기본 의미: {basic_semantics}
        
        의미 벡터의 주요 차원들을 식별하고 분석하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "primary_dimensions": [
                {{
                    "dimension": "차원명",
                    "value": 0.0-1.0,
                    "description": "차원 설명"
                }}
            ],
            "semantic_clusters": [
                {{
                    "cluster": "클러스터명",
                    "concepts": ["개념1", "개념2"],
                    "centrality": 0.0-1.0
                }}
            ],
            "vector_magnitude": 0.0-1.0,
            "semantic_density": "sparse|moderate|dense",
            "neighboring_concepts": ["인접 개념1", "인접 개념2"]
        }}
        """
        
        response = await self.llm_client.agenerate(vector_prompt)
        return self._parse_json_response(response)
    
    async def _analyze_conceptual_connections(self, query: str, basic_semantics: Dict) -> Dict:
        """개념적 연결망 분석"""
        network_prompt = f"""
        쿼리와 관련된 개념들의 연결망을 분석하세요.
        
        쿼리: {query}
        기본 의미: {basic_semantics}
        
        개념 간의 관계와 연결 강도를 분석하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "core_concepts": ["핵심 개념1", "핵심 개념2"],
            "concept_relationships": [
                {{
                    "from": "개념A",
                    "to": "개념B",
                    "relationship_type": "관계 유형",
                    "strength": 0.0-1.0
                }}
            ],
            "network_density": 0.0-1.0,
            "central_nodes": ["중심 노드1", "중심 노드2"],
            "bridge_concepts": ["연결 개념1", "연결 개념2"],
            "network_complexity": "simple|moderate|complex"
        }}
        """
        
        response = await self.llm_client.agenerate(network_prompt)
        return self._parse_json_response(response)
    
    async def _analyze_semantic_hierarchy(self, query: str, basic_semantics: Dict) -> Dict:
        """의미 계층 구조 분석"""
        hierarchy_prompt = f"""
        쿼리의 의미를 계층적 구조로 분석하세요.
        
        쿼리: {query}
        기본 의미: {basic_semantics}
        
        상위-하위 개념의 계층 구조를 분석하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "hierarchy_levels": [
                {{
                    "level": 1,
                    "concepts": ["최상위 개념1", "최상위 개념2"],
                    "abstraction": "very_high"
                }},
                {{
                    "level": 2,
                    "concepts": ["중간 개념1", "중간 개념2"],
                    "abstraction": "high"
                }}
            ],
            "abstraction_depth": 1-5,
            "conceptual_breadth": "narrow|moderate|broad",
            "hierarchy_balance": "balanced|top_heavy|bottom_heavy"
        }}
        """
        
        response = await self.llm_client.agenerate(hierarchy_prompt)
        return self._parse_json_response(response)
    
    async def _analyze_contextual_meanings(self, query: str, basic_semantics: Dict) -> Dict:
        """컨텍스트 의존적 의미 분석"""
        contextual_prompt = f"""
        쿼리의 의미가 다양한 컨텍스트에서 어떻게 달라질 수 있는지 분석하세요.
        
        쿼리: {query}
        기본 의미: {basic_semantics}
        
        JSON 형식으로 응답하세요:
        {{
            "context_dependent_meanings": [
                {{
                    "context": "컨텍스트1",
                    "meaning": "이 컨텍스트에서의 의미",
                    "likelihood": 0.0-1.0
                }}
            ],
            "context_sensitivity": "low|medium|high",
            "disambiguation_needed": true/false,
            "stable_meanings": ["안정적 의미1", "안정적 의미2"],
            "variable_meanings": ["가변적 의미1", "가변적 의미2"]
        }}
        """
        
        response = await self.llm_client.agenerate(contextual_prompt)
        return self._parse_json_response(response)
    
    async def _detect_and_classify_ambiguity(self, query: str, context: Dict) -> Dict:
        """모호성 감지 및 분류"""
        ambiguity_prompt = f"""
        쿼리의 모호성을 감지하고 분류하세요.
        
        쿼리: {query}
        컨텍스트: {context}
        
        JSON 형식으로 응답하세요:
        {{
            "ambiguity_types": [
                {{
                    "type": "lexical|syntactic|semantic|pragmatic",
                    "description": "모호성 설명",
                    "severity": "low|medium|high",
                    "examples": ["예시1", "예시2"]
                }}
            ],
            "ambiguous_elements": ["모호한 요소1", "모호한 요소2"],
            "clarity_score": 0.0-1.0,
            "disambiguation_priority": "low|medium|high|critical",
            "potential_misunderstandings": ["오해 가능성1", "오해 가능성2"]
        }}
        """
        
        response = await self.llm_client.agenerate(ambiguity_prompt)
        return self._parse_json_response(response)
    
    async def _generate_multiple_interpretations(self, query: str, ambiguity_analysis: Dict, context: Dict) -> List[Dict]:
        """다중 해석 생성"""
        interpretation_prompt = f"""
        모호한 쿼리에 대한 가능한 해석들을 생성하세요.
        
        쿼리: {query}
        모호성 분석: {ambiguity_analysis}
        컨텍스트: {context}
        
        JSON 형식으로 응답하세요:
        {{
            "interpretations": [
                {{
                    "interpretation": "해석 1",
                    "probability": 0.0-1.0,
                    "reasoning": "이 해석의 근거",
                    "implications": ["함의1", "함의2"],
                    "required_assumptions": ["가정1", "가정2"]
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(interpretation_prompt)
        result = self._parse_json_response(response)
        return result.get('interpretations', [])
    
    async def _generate_clarification_questions(self, ambiguity_analysis: Dict, interpretations: List[Dict]) -> List[Dict]:
        """명확화 질문 생성"""
        question_prompt = f"""
        모호성을 해결하기 위한 명확화 질문들을 생성하세요.
        
        모호성 분석: {ambiguity_analysis}
        가능한 해석들: {interpretations}
        
        JSON 형식으로 응답하세요:
        {{
            "clarification_questions": [
                {{
                    "question": "명확화 질문",
                    "purpose": "질문의 목적",
                    "expected_answer_type": "선택형|서술형|예/아니오",
                    "priority": "high|medium|low",
                    "options": ["선택지1", "선택지2"] (선택형인 경우)
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(question_prompt)
        result = self._parse_json_response(response)
        return result.get('clarification_questions', [])
    
    def _calculate_semantic_richness(self, vectors: Dict, network: Dict) -> float:
        """의미 풍부도 계산"""
        richness_factors = [
            vectors.get('vector_magnitude', 0.0),
            network.get('network_density', 0.0),
            len(network.get('core_concepts', [])) / 10.0
        ]
        return sum(richness_factors) / len(richness_factors)
    
    def _calculate_interpretation_confidence(self, basic_semantics: Dict, contextual_meanings: Dict) -> float:
        """해석 신뢰도 계산"""
        confidence_factors = [
            1.0 - (1.0 if contextual_meanings.get('disambiguation_needed', False) else 0.0) * 0.3,
            1.0 if contextual_meanings.get('context_sensitivity', 'medium') == 'low' else 0.7,
            basic_semantics.get('specificity', 0.5) if isinstance(basic_semantics.get('specificity'), (int, float)) else 0.5
        ]
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_ambiguity_score(self, ambiguity_analysis: Dict) -> float:
        """모호성 점수 계산"""
        ambiguity_types = ambiguity_analysis.get('ambiguity_types', [])
        if not ambiguity_types:
            return 0.0
        
        severity_weights = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        total_score = sum(severity_weights.get(t.get('severity', 'medium'), 0.5) for t in ambiguity_types)
        return min(total_score / len(ambiguity_types), 1.0)
    
    def _assess_clarification_urgency(self, ambiguity_analysis: Dict, interpretations: List[Dict]) -> str:
        """명확화 긴급도 평가"""
        priority = ambiguity_analysis.get('disambiguation_priority', 'medium')
        interpretation_spread = len(interpretations) if interpretations else 1
        
        if priority == 'critical' or interpretation_spread > 3:
            return 'high'
        elif priority == 'high' or interpretation_spread > 2:
            return 'medium'
        else:
            return 'low'
    
    async def _fallback_semantic_analysis(self, query: str) -> Dict:
        """폴백 의미 분석"""
        return {
            'type': 'fallback',
            'basic_analysis': f"쿼리 '{query}'에 대한 기본 분석",
            'message': '상세한 의미 분석을 수행할 수 없습니다.'
        }
    
    async def _fallback_clarification(self, query: str) -> Dict:
        """폴백 명확화"""
        return {
            'type': 'fallback',
            'simple_questions': [
                "더 구체적으로 설명해 주실 수 있나요?",
                "어떤 부분에 대해 알고 싶으신가요?",
                "예시를 들어 설명해 주실 수 있나요?"
            ],
            'message': '구체적인 명확화 질문을 생성할 수 없습니다.'
        }
    
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
    
    async def _find_optimal_semantic_paths(self, vectors: Dict, network: Dict, hierarchy: Dict) -> Dict:
        """의미 공간 내 최적 경로 탐색"""
        return {
            'primary_path': {
                'route': ['시작점', '중간점', '목표점'],
                'confidence': 0.8,
                'reasoning': '가장 직접적이고 효율적인 경로'
            },
            'alternative_paths': [
                {
                    'route': ['시작점', '우회점', '목표점'],
                    'confidence': 0.6,
                    'reasoning': '보다 안전한 우회 경로'
                }
            ],
            'path_complexity': 'moderate',
            'recommended_path': 'primary'
        }
    
    async def _recommend_semantic_approaches(self, optimal_paths: Dict, hierarchy: Dict) -> List[str]:
        """의미적 접근법 추천"""
        approaches = []
        
        # 경로 복잡도에 따른 접근법
        complexity = optimal_paths.get('path_complexity', 'moderate')
        if complexity == 'simple':
            approaches.append("직접적 접근법 사용")
        elif complexity == 'complex':
            approaches.append("단계적 접근법 사용")
        else:
            approaches.append("균형잡힌 접근법 사용")
        
        # 계층 구조에 따른 접근법
        abstraction_depth = hierarchy.get('abstraction_depth', 3)
        if abstraction_depth > 3:
            approaches.append("추상화 수준 조정 필요")
        
        approaches.extend([
            "의미 공간 탐색 기반 분석",
            "다층적 해석 고려",
            "컨텍스트 의존성 반영"
        ])
        
        return approaches[:5]
    
    async def _prioritize_interpretations(self, interpretations: List[Dict], context: Dict) -> Dict:
        """해석 우선순위 결정"""
        if not interpretations:
            return {'priorities': [], 'reasoning': '해석이 없습니다.'}
        
        # 확률 기반 정렬
        sorted_interpretations = sorted(
            interpretations, 
            key=lambda x: x.get('probability', 0.0), 
            reverse=True
        )
        
        return {
            'priorities': [
                {
                    'rank': i + 1,
                    'interpretation': interp.get('interpretation', ''),
                    'probability': interp.get('probability', 0.0),
                    'priority_level': 'high' if i < 2 else 'medium' if i < 4 else 'low'
                }
                for i, interp in enumerate(sorted_interpretations)
            ],
            'reasoning': '확률과 컨텍스트 적합성을 기준으로 우선순위 결정'
        }
    
    async def _develop_progressive_clarification_strategy(self, questions: List[Dict], priorities: Dict) -> Dict:
        """점진적 명확화 전략 수립"""
        if not questions:
            return {'strategy': 'no_clarification_needed'}
        
        # 우선순위 높은 질문부터 정렬
        high_priority_questions = [q for q in questions if q.get('priority') == 'high']
        medium_priority_questions = [q for q in questions if q.get('priority') == 'medium']
        
        return {
            'strategy_type': 'progressive',
            'phases': [
                {
                    'phase': 1,
                    'questions': high_priority_questions[:2],
                    'goal': '핵심 모호성 해결'
                },
                {
                    'phase': 2,
                    'questions': medium_priority_questions[:2],
                    'goal': '세부 사항 명확화'
                }
            ],
            'total_phases': 2,
            'estimated_interactions': len(high_priority_questions) + len(medium_priority_questions)
        }
    
    async def _suggest_uncertainty_handling(self, ambiguity_analysis: Dict, interpretations: List[Dict]) -> Dict:
        """불확실성 처리 방안 제시"""
        uncertainty_level = self._calculate_ambiguity_score(ambiguity_analysis)
        
        if uncertainty_level > 0.7:
            handling_approach = 'conservative'
            recommendations = [
                "명확화 질문을 통한 확실성 확보",
                "다중 해석 제시 및 사용자 선택 요청",
                "불확실성을 명시적으로 전달"
            ]
        elif uncertainty_level > 0.4:
            handling_approach = 'balanced'
            recommendations = [
                "가장 가능성 높은 해석으로 진행",
                "대안적 해석 병행 제시",
                "피드백을 통한 조정"
            ]
        else:
            handling_approach = 'confident'
            recommendations = [
                "주요 해석으로 진행",
                "필요시 확인 질문",
                "결과 기반 검증"
            ]
        
        return {
            'uncertainty_level': uncertainty_level,
            'handling_approach': handling_approach,
            'recommendations': recommendations,
            'risk_mitigation': [
                "오해 발생 시 즉시 수정",
                "사용자 피드백 적극 수용",
                "점진적 정확도 개선"
            ]
        }
    
    async def _recommend_clarification_next_steps(self, progressive_clarification: Dict, uncertainty_handling: Dict) -> List[str]:
        """명확화 다음 단계 추천"""
        next_steps = []
        
        # 점진적 명확화 전략에 따른 단계
        if progressive_clarification.get('strategy_type') == 'progressive':
            phases = progressive_clarification.get('phases', [])
            if phases:
                first_phase = phases[0]
                questions = first_phase.get('questions', [])
                if questions:
                    next_steps.append(f"1단계: {questions[0].get('question', '첫 번째 명확화 질문')}")
        
        # 불확실성 처리에 따른 단계
        handling_approach = uncertainty_handling.get('handling_approach', 'balanced')
        if handling_approach == 'conservative':
            next_steps.append("모든 가능한 해석을 사용자에게 제시")
        elif handling_approach == 'confident':
            next_steps.append("가장 가능성 높은 해석으로 진행")
        
        # 기본 단계들
        next_steps.extend([
            "사용자 응답에 따른 해석 조정",
            "명확화된 의도로 재분석 수행",
            "최종 확인 및 진행"
        ])
        
        return next_steps[:5]