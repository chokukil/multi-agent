"""
Adaptive User Understanding - 적응형 사용자 이해 시스템

요구사항 4에 따른 구현:
- 언어 사용, 용어, 질문 복잡도를 통한 사용자 전문성 추정
- 점진적 공개를 통한 이해 수준 파악
- 대화 중 동적 조정
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import re

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AdaptiveUserUnderstanding:
    """
    적응형 사용자 이해 시스템
    - 상호작용을 통한 사용자 수준 파악
    - 동적 커뮤니케이션 스타일 조정
    - 지속적인 사용자 모델 개선
    """
    
    def __init__(self):
        """AdaptiveUserUnderstanding 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.user_models = {}
        self.interaction_history = []
        logger.info("AdaptiveUserUnderstanding initialized")
    
    async def analyze_user_expertise(self, query: str, interaction_history: List[Dict] = None) -> Dict:
        """
        사용자 전문성 수준 분석
        
        Args:
            query: 사용자 쿼리
            interaction_history: 이전 상호작용 이력
            
        Returns:
            사용자 전문성 분석 결과
        """
        logger.info("Analyzing user expertise level")
        
        try:
            # 1. 언어 사용 패턴 분석
            language_analysis = await self._analyze_language_usage(query)
            
            # 2. 질문 복잡도 평가
            query_complexity = await self._assess_query_complexity(query)
            
            # 3. 기술 용어 사용 분석
            terminology_analysis = await self._analyze_terminology_usage(query)
            
            # 4. 상호작용 이력 기반 패턴 분석
            historical_patterns = None
            if interaction_history:
                historical_patterns = await self._analyze_interaction_patterns(interaction_history)
            
            # 5. 종합적 사용자 프로필 생성
            user_profile = await self._create_comprehensive_profile(
                language_analysis,
                query_complexity,
                terminology_analysis,
                historical_patterns
            )
            
            return {
                'user_profile': user_profile,
                'language_analysis': language_analysis,
                'query_complexity': query_complexity,
                'terminology_analysis': terminology_analysis,
                'historical_patterns': historical_patterns,
                'confidence': user_profile.get('confidence_level', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in user expertise analysis: {e}")
            raise
    
    async def _analyze_language_usage(self, query: str) -> Dict:
        """
        언어 사용 패턴 분석
        """
        # 기본 언어 특성 추출
        language_features = {
            'query_length': len(query),
            'word_count': len(query.split()),
            'avg_word_length': sum(len(word) for word in query.split()) / max(len(query.split()), 1),
            'question_marks': query.count('?'),
            'technical_indicators': self._count_technical_indicators(query),
            'formality_indicators': self._assess_formality(query)
        }
        
        # LLM을 통한 심층 분석
        prompt = f"""
        다음 질문의 언어 사용 패턴을 분석하세요.
        
        질문: {query}
        
        JSON 형식으로 응답하세요:
        {{
            "language_sophistication": "basic|intermediate|advanced",
            "communication_style": "casual|formal|technical|mixed",
            "clarity_level": "vague|clear|precise",
            "question_structure": "simple|compound|complex",
            "domain_familiarity_indicators": ["지시자1", "지시자2"],
            "confidence_indicators": ["자신감 지시자1", "자신감 지시자2"],
            "uncertainty_markers": ["불확실성 표현1", "불확실성 표현2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        llm_analysis = self._parse_json_response(response)
        
        return {**language_features, **llm_analysis}
    
    def _count_technical_indicators(self, text: str) -> int:
        """기술적 지시자 수 계산"""
        # 일반적인 기술 용어 패턴 (하드코딩 최소화)
        technical_patterns = [
            r'\b\w+[_]\w+\b',  # snake_case
            r'\b[A-Z]{2,}\b',  # 약어
            r'\b\d+\.\d+\b',   # 소수
            r'\b\w+\(\)',      # 함수 호출
            r'\[[^\]]+\]',     # 대괄호 표현
        ]
        
        count = 0
        for pattern in technical_patterns:
            count += len(re.findall(pattern, text))
            
        return count
    
    def _assess_formality(self, text: str) -> Dict:
        """형식성 평가"""
        informal_indicators = ['뭐', '뭔가', '좀', '약간', '아마']
        formal_indicators = ['분석', '검토', '평가', '고려', '검증']
        
        informal_count = sum(1 for word in informal_indicators if word in text)
        formal_count = sum(1 for word in formal_indicators if word in text)
        
        return {
            'informal_count': informal_count,
            'formal_count': formal_count,
            'formality_ratio': formal_count / max(informal_count + formal_count, 1)
        }
    
    async def _assess_query_complexity(self, query: str) -> Dict:
        """
        질문 복잡도 평가
        """
        prompt = f"""
        다음 질문의 복잡도를 평가하세요.
        
        질문: {query}
        
        JSON 형식으로 응답하세요:
        {{
            "complexity_level": "simple|moderate|complex|expert",
            "reasoning_required": "basic|analytical|systematic|advanced",
            "scope": "narrow|moderate|broad|comprehensive",
            "abstraction_level": "concrete|mixed|abstract",
            "multi_part_question": true/false,
            "requires_context": true/false,
            "complexity_factors": ["요인1", "요인2", "요인3"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _analyze_terminology_usage(self, query: str) -> Dict:
        """
        기술 용어 사용 분석
        """
        prompt = f"""
        다음 질문에서 사용된 전문 용어와 기술 용어를 분석하세요.
        
        질문: {query}
        
        JSON 형식으로 응답하세요:
        {{
            "technical_terms": ["용어1", "용어2"],
            "domain_specific_terms": ["도메인 용어1", "도메인 용어2"],
            "terminology_sophistication": "none|basic|intermediate|advanced",
            "correct_usage": true/false,
            "mixed_terminology": true/false,
            "terminology_consistency": "consistent|mixed|inconsistent"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _analyze_interaction_patterns(self, history: List[Dict]) -> Dict:
        """
        상호작용 이력 패턴 분석
        """
        if not history:
            return {}
            
        # 이력에서 패턴 추출
        patterns = {
            'interaction_count': len(history),
            'avg_query_length': sum(len(h.get('query', '')) for h in history) / len(history),
            'complexity_progression': self._analyze_complexity_progression(history),
            'topic_consistency': self._analyze_topic_consistency(history),
            'learning_indicators': self._identify_learning_indicators(history)
        }
        
        # LLM을 통한 종합 분석
        prompt = f"""
        사용자의 상호작용 이력을 분석하여 학습 패턴과 전문성 변화를 파악하세요.
        
        상호작용 수: {patterns['interaction_count']}
        복잡도 진행: {patterns['complexity_progression']}
        주제 일관성: {patterns['topic_consistency']}
        
        JSON 형식으로 응답하세요:
        {{
            "expertise_trajectory": "increasing|stable|decreasing|variable",
            "learning_style": "exploratory|focused|systematic|mixed",
            "engagement_level": "low|moderate|high",
            "adaptation_needed": "simplify|maintain|advance",
            "user_goals": ["추정 목표1", "추정 목표2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        llm_patterns = self._parse_json_response(response)
        
        return {**patterns, **llm_patterns}
    
    def _analyze_complexity_progression(self, history: List[Dict]) -> str:
        """복잡도 진행 분석"""
        if len(history) < 2:
            return "insufficient_data"
            
        # 간단한 복잡도 점수 계산
        complexity_scores = []
        for item in history:
            query = item.get('query', '')
            score = len(query) * 0.1 + query.count('?') * 10 + len(query.split()) * 0.5
            complexity_scores.append(score)
        
        # 추세 분석
        if len(complexity_scores) >= 2:
            trend = complexity_scores[-1] - complexity_scores[0]
            if trend > 10:
                return "increasing"
            elif trend < -10:
                return "decreasing"
            else:
                return "stable"
        
        return "stable"
    
    def _analyze_topic_consistency(self, history: List[Dict]) -> float:
        """주제 일관성 분석"""
        # 간단한 주제 일관성 점수 (0-1)
        # 실제로는 더 정교한 분석이 필요하지만 여기서는 단순화
        return 0.7  # 기본값
    
    def _identify_learning_indicators(self, history: List[Dict]) -> List[str]:
        """학습 지시자 식별"""
        indicators = []
        
        # 질문이 점점 구체적으로 변하는지 확인
        if len(history) > 2:
            recent_queries = [h.get('query', '') for h in history[-3:]]
            if all('구체적' in q or 'specific' in q.lower() for q in recent_queries):
                indicators.append("increasing_specificity")
        
        # 후속 질문 패턴
        for i in range(1, len(history)):
            if '더' in history[i].get('query', '') or 'more' in history[i].get('query', '').lower():
                indicators.append("follow_up_questions")
                break
        
        return indicators
    
    async def _create_comprehensive_profile(
        self,
        language_analysis: Dict,
        query_complexity: Dict,
        terminology_analysis: Dict,
        historical_patterns: Dict = None
    ) -> Dict:
        """
        종합적 사용자 프로필 생성
        """
        # 전문성 수준 결정
        expertise_score = 0
        
        # 언어 정교함
        if language_analysis.get('language_sophistication') == 'advanced':
            expertise_score += 3
        elif language_analysis.get('language_sophistication') == 'intermediate':
            expertise_score += 2
        else:
            expertise_score += 1
        
        # 질문 복잡도
        if query_complexity.get('complexity_level') in ['complex', 'expert']:
            expertise_score += 3
        elif query_complexity.get('complexity_level') == 'moderate':
            expertise_score += 2
        else:
            expertise_score += 1
        
        # 용어 사용
        if terminology_analysis.get('terminology_sophistication') in ['intermediate', 'advanced']:
            expertise_score += 2
        
        # 전문성 수준 결정
        if expertise_score >= 7:
            expertise_level = 'expert'
        elif expertise_score >= 4:
            expertise_level = 'intermediate'
        else:
            expertise_level = 'beginner'
        
        # 신뢰도 계산
        confidence_level = min(0.9, 0.3 + (expertise_score * 0.1))
        
        profile = {
            'expertise_level': expertise_level,
            'expertise_score': expertise_score,
            'confidence_level': confidence_level,
            'communication_preferences': {
                'preferred_style': language_analysis.get('communication_style', 'balanced'),
                'detail_level': 'high' if expertise_level == 'expert' else 'medium' if expertise_level == 'intermediate' else 'low',
                'use_technical_terms': expertise_level in ['intermediate', 'expert'],
                'needs_examples': expertise_level == 'beginner'
            },
            'learning_characteristics': {
                'learning_style': historical_patterns.get('learning_style', 'unknown') if historical_patterns else 'unknown',
                'engagement_level': historical_patterns.get('engagement_level', 'moderate') if historical_patterns else 'moderate',
                'prefers_exploration': language_analysis.get('question_structure') == 'complex'
            },
            'adaptation_recommendations': {
                'explanation_depth': 'deep' if expertise_level == 'expert' else 'moderate' if expertise_level == 'intermediate' else 'shallow',
                'use_analogies': expertise_level == 'beginner',
                'provide_context': query_complexity.get('requires_context', False),
                'offer_alternatives': expertise_level in ['intermediate', 'expert']
            }
        }
        
        return profile
    
    async def adapt_communication_style(self, content: Dict, user_profile: Dict) -> Dict:
        """
        사용자 프로필에 맞춰 커뮤니케이션 스타일 조정
        """
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        preferences = user_profile.get('communication_preferences', {})
        
        prompt = f"""
        다음 내용을 사용자 수준에 맞게 조정하세요.
        
        원본 내용: {content}
        사용자 수준: {expertise_level}
        선호 스타일: {preferences}
        
        조정 지침:
        - 초보자: 친근한 설명, 비유 사용, 단계별 안내
        - 중급자: 균형잡힌 설명, 선택적 심화 내용
        - 전문가: 기술적 정확성, 간결한 설명, 고급 옵션
        
        JSON 형식으로 응답하세요:
        {{
            "adapted_content": "조정된 내용",
            "additional_elements": {{
                "examples": ["예시1", "예시2"] (필요한 경우),
                "analogies": ["비유1", "비유2"] (필요한 경우),
                "technical_details": ["기술 상세1", "기술 상세2"] (필요한 경우)
            }},
            "interaction_prompts": ["상호작용 프롬프트1", "상호작용 프롬프트2"],
            "next_step_suggestions": ["다음 단계 제안1", "다음 단계 제안2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def update_user_model(self, user_id: str, interaction_result: Dict) -> Dict:
        """
        상호작용 결과를 바탕으로 사용자 모델 업데이트
        """
        if user_id not in self.user_models:
            self.user_models[user_id] = {
                'interactions': [],
                'profile_history': [],
                'last_updated': datetime.now()
            }
        
        model = self.user_models[user_id]
        model['interactions'].append(interaction_result)
        model['last_updated'] = datetime.now()
        
        # 프로필 재평가가 필요한지 확인
        if len(model['interactions']) % 5 == 0:  # 5번 상호작용마다 재평가
            updated_profile = await self._reevaluate_user_profile(user_id)
            model['profile_history'].append(updated_profile)
            
        return {
            'user_id': user_id,
            'model_updated': True,
            'interaction_count': len(model['interactions']),
            'profile_stable': len(model['profile_history']) > 2 and self._is_profile_stable(model['profile_history'][-3:])
        }
    
    async def _reevaluate_user_profile(self, user_id: str) -> Dict:
        """사용자 프로필 재평가"""
        model = self.user_models.get(user_id, {})
        recent_interactions = model.get('interactions', [])[-10:]  # 최근 10개 상호작용
        
        # 최근 상호작용을 바탕으로 프로필 재분석
        if recent_interactions:
            latest_query = recent_interactions[-1].get('query', '')
            return await self.analyze_user_expertise(latest_query, recent_interactions)
        
        return {}
    
    def _is_profile_stable(self, profile_history: List[Dict]) -> bool:
        """프로필 안정성 확인"""
        if len(profile_history) < 2:
            return False
            
        # 최근 프로필들의 전문성 수준이 동일한지 확인
        expertise_levels = [p.get('user_profile', {}).get('expertise_level') for p in profile_history]
        return len(set(expertise_levels)) == 1
    
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