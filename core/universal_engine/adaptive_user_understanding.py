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
from .optimizations.llm_performance_optimizer import optimize_llm_call

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
    
    async def analyze_user(self, query: str, data: Dict = None, context: Dict = None) -> Dict:
        """
        사용자 분석 - analyze_user_expertise의 별칭
        기존 코드와의 호환성을 위한 메서드
        
        Args:
            query: 사용자 쿼리
            data: 데이터 컨텍스트
            context: 추가 컨텍스트
            
        Returns:
            사용자 분석 결과
        """
        interaction_history = context.get('interaction_history', []) if context else []
        return await self.analyze_user_expertise(query, interaction_history)
    
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
        
        try:
            # 최적화된 LLM 호출 사용
            optimized_result = await optimize_llm_call(self.llm_client, prompt)
            response_content = optimized_result["response"]
            llm_analysis = self._parse_json_response(response_content)
        except Exception as e:
            logger.error(f"LLM language analysis failed: {e}")
            llm_analysis = {
                "language_sophistication": "intermediate",
                "communication_style": "mixed",
                "clarity_level": "clear",
                "question_structure": "simple",
                "domain_familiarity_indicators": [],
                "confidence_indicators": [],
                "uncertainty_markers": []
            }
        
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
        
        try:
            # 최적화된 LLM 호출 사용
            optimized_result = await optimize_llm_call(self.llm_client, prompt)
            response_content = optimized_result["response"]
            return self._parse_json_response(response_content)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"complexity_level": "medium", "complexity_indicators": []}
    
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
        
        try:
            # 최적화된 LLM 호출 사용
            optimized_result = await optimize_llm_call(self.llm_client, prompt)
            response_content = optimized_result["response"]
            return self._parse_json_response(response_content)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"complexity_level": "medium", "complexity_indicators": []}
    
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
        
        try:
            # 최적화된 LLM 호출 사용
            optimized_result = await optimize_llm_call(self.llm_client, prompt)
            response_content = optimized_result["response"]
            return self._parse_json_response(response_content)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"complexity_level": "medium", "complexity_indicators": []}
    
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
    
    async def estimate_user_level(self, query: str, interaction_history: List) -> str:
        """
        사용자 전문성 수준 추정
        
        요구사항 1.1에 따른 구현:
        - 언어 사용, 용어, 질문 복잡도를 통한 전문성 추정
        - 상호작용 기반 추론
        - 동적 수준 판단
        
        Args:
            query: 사용자 쿼리
            interaction_history: 상호작용 이력
            
        Returns:
            추정된 사용자 수준 ('beginner', 'intermediate', 'expert')
        """
        logger.info("Estimating user expertise level")
        
        try:
            # 전체 사용자 분석 수행
            analysis_result = await self.analyze_user_expertise(query, interaction_history)
            
            # 기본 전문성 수준 추출
            base_level = analysis_result.get('user_profile', {}).get('expertise_level', 'intermediate')
            
            # 추가적인 수준 세분화 분석
            refined_level = await self._perform_refined_level_estimation(
                query, interaction_history, analysis_result
            )
            
            # 신뢰도 기반 최종 결정
            confidence = analysis_result.get('confidence', 0.5)
            
            if confidence > 0.8:
                final_level = refined_level
            elif confidence > 0.6:
                final_level = base_level
            else:
                # 낮은 신뢰도일 때는 보수적으로 중간 수준으로 설정
                final_level = 'intermediate'
            
            logger.info(f"Estimated user level: {final_level} (confidence: {confidence:.2f})")
            return final_level
            
        except Exception as e:
            logger.error(f"Error in user level estimation: {e}")
            return 'intermediate'  # 기본값 반환
    
    async def adapt_response(self, content: str, user_level: str) -> str:
        """
        사용자 수준에 맞는 응답 적응
        
        요구사항 1.1에 따른 구현:
        - 사용자 수준별 응답 적응
        - 설명 깊이 조절
        - 기술 용어 사용 조정
        
        Args:
            content: 원본 응답 내용
            user_level: 사용자 수준
            
        Returns:
            적응된 응답 내용
        """
        logger.info(f"Adapting response for user level: {user_level}")
        
        try:
            # 사용자 수준별 적응 전략 결정
            adaptation_strategy = self._get_adaptation_strategy(user_level)
            
            # LLM을 통한 응답 적응
            adaptation_prompt = f"""
            다음 내용을 {user_level} 수준의 사용자에게 맞게 적응시켜주세요.
            
            원본 내용: {content}
            사용자 수준: {user_level}
            적응 전략: {adaptation_strategy}
            
            적응 지침:
            {self._get_adaptation_guidelines(user_level)}
            
            적응된 내용만 반환하세요 (JSON 형식 없이 직접 텍스트로):
            """
            
            response = await self.llm_client.ainvoke(adaptation_prompt)
            adapted_content = response.content if hasattr(response, 'content') else str(response)
            
            # 후처리: 불필요한 메타 텍스트 제거
            adapted_content = self._clean_adapted_content(adapted_content)
            
            # 추가 개선 사항 적용
            final_content = await self._apply_additional_adaptations(
                adapted_content, user_level, content
            )
            
            return final_content
            
        except Exception as e:
            logger.error(f"Error in response adaptation: {e}")
            return content  # 오류 시 원본 반환
    
    async def update_user_profile(self, interaction_data: Dict) -> Dict[str, Any]:
        """
        상호작용 기반 사용자 프로필 업데이트
        
        요구사항 1.1에 따른 구현:
        - 상호작용 데이터 기반 프로필 업데이트
        - 학습 패턴 추적
        - 동적 프로필 개선
        
        Args:
            interaction_data: 상호작용 데이터
            
        Returns:
            업데이트된 사용자 프로필 정보
        """
        logger.info("Updating user profile based on interaction data")
        
        try:
            # 상호작용 데이터에서 핵심 정보 추출
            user_id = interaction_data.get('user_id', 'anonymous')
            query = interaction_data.get('query', '')
            response_feedback = interaction_data.get('feedback', {})
            interaction_timestamp = interaction_data.get('timestamp', datetime.now())
            
            # 기존 프로필 로드 또는 생성
            current_profile = self._get_or_create_user_profile(user_id)
            
            # 새로운 상호작용 분석
            interaction_analysis = await self._analyze_new_interaction(
                query, response_feedback, current_profile
            )
            
            # 프로필 업데이트 수행
            updated_profile = await self._perform_profile_update(
                current_profile, interaction_analysis, interaction_data
            )
            
            # 학습 패턴 업데이트
            learning_patterns = await self._update_learning_patterns(
                user_id, interaction_analysis, updated_profile
            )
            
            # 적응 전략 재평가
            adaptation_strategy = await self._reevaluate_adaptation_strategy(
                updated_profile, learning_patterns
            )
            
            # 최종 프로필 저장
            final_profile = {
                'user_id': user_id,
                'profile': updated_profile,
                'learning_patterns': learning_patterns,
                'adaptation_strategy': adaptation_strategy,
                'last_updated': datetime.now().isoformat(),
                'update_confidence': interaction_analysis.get('confidence', 0.5),
                'interaction_count': current_profile.get('interaction_count', 0) + 1,
                'profile_stability': self._calculate_profile_stability(updated_profile, current_profile)
            }
            
            # 프로필 저장
            self.user_models[user_id] = final_profile
            
            return {
                'update_successful': True,
                'profile_changes': self._identify_profile_changes(current_profile, updated_profile),
                'confidence_improvement': interaction_analysis.get('confidence', 0.5) - current_profile.get('confidence_level', 0.5),
                'learning_insights': learning_patterns.get('insights', []),
                'adaptation_recommendations': adaptation_strategy.get('recommendations', []),
                'next_interaction_guidance': await self._generate_next_interaction_guidance(final_profile)
            }
            
        except Exception as e:
            logger.error(f"Error in user profile update: {e}")
            return {
                'update_successful': False,
                'error': str(e),
                'fallback_profile': self._get_default_profile()
            }
    
    async def _perform_refined_level_estimation(self, query: str, history: List, analysis: Dict) -> str:
        """세분화된 수준 추정"""
        refinement_prompt = f"""
        사용자의 질문과 분석 결과를 바탕으로 더 정확한 전문성 수준을 판단하세요.
        
        질문: {query}
        기본 분석: {analysis.get('user_profile', {})}
        상호작용 수: {len(history) if history else 0}
        
        다음 기준으로 세밀하게 판단하세요:
        - 초보자: 기본 개념 질문, 단순한 요청, 안내 필요
        - 중급자: 구체적 문제, 일부 전문 용어, 선택적 도움
        - 전문가: 복잡한 분석, 고급 용어, 독립적 작업
        
        하나의 단어로만 응답하세요: beginner, intermediate, expert
        """
        
        response = await self.llm_client.ainvoke(refinement_prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        refined_level = response.strip().lower()
        
        if refined_level in ['beginner', 'intermediate', 'expert']:
            return refined_level
        else:
            return analysis.get('user_profile', {}).get('expertise_level', 'intermediate')
    
    def _get_adaptation_strategy(self, user_level: str) -> Dict:
        """적응 전략 결정"""
        strategies = {
            'beginner': {
                'explanation_style': 'step_by_step',
                'use_analogies': True,
                'technical_terms': 'minimal',
                'examples': 'many',
                'tone': 'friendly_supportive'
            },
            'intermediate': {
                'explanation_style': 'balanced',
                'use_analogies': 'selective',
                'technical_terms': 'moderate',
                'examples': 'relevant',
                'tone': 'professional_helpful'
            },
            'expert': {
                'explanation_style': 'concise_technical',
                'use_analogies': False,
                'technical_terms': 'full',
                'examples': 'minimal',
                'tone': 'direct_efficient'
            }
        }
        
        return strategies.get(user_level, strategies['intermediate'])
    
    def _get_adaptation_guidelines(self, user_level: str) -> str:
        """적응 가이드라인 반환"""
        guidelines = {
            'beginner': """
            - 친근하고 격려하는 톤 사용
            - 복잡한 개념을 단순한 언어로 설명
            - 단계별로 차근차근 안내
            - 비유와 예시를 풍부하게 사용
            - 전문 용어 사용 시 반드시 설명 추가
            - "걱정하지 마세요", "천천히 해보세요" 같은 격려 표현 사용
            """,
            'intermediate': """
            - 전문적이면서도 접근하기 쉬운 톤 사용
            - 핵심 개념을 명확하게 설명
            - 필요한 경우에만 예시 제공
            - 적절한 수준의 전문 용어 사용
            - 선택적 심화 정보 제공
            - 사용자의 판단을 존중하는 표현 사용
            """,
            'expert': """
            - 간결하고 정확한 기술적 설명
            - 불필요한 설명 최소화
            - 전문 용어를 정확하게 사용
            - 고급 옵션과 대안 제시
            - 효율성과 정확성에 중점
            - 동료 전문가와 대화하는 톤 사용
            """
        }
        
        return guidelines.get(user_level, guidelines['intermediate'])
    
    def _clean_adapted_content(self, content: str) -> str:
        """적응된 내용 정리"""
        # 불필요한 메타 텍스트 제거
        content = content.strip()
        
        # "적응된 내용:", "다음은" 등의 메타 표현 제거
        meta_patterns = [
            r'^적응된 내용:\s*',
            r'^다음은.*?:\s*',
            r'^Here is.*?:\s*',
            r'^Adapted.*?:\s*'
        ]
        
        for pattern in meta_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        return content.strip()
    
    async def _apply_additional_adaptations(self, content: str, user_level: str, original: str) -> str:
        """추가 적응 사항 적용 - LLM 기반 동적 처리"""
        try:
            prompt = f"""
            Based on the user level '{user_level}' and content characteristics, determine what additional adaptations are needed.
            
            User level: {user_level}
            Content length: {len(content)} characters
            Original length: {len(original)} characters
            
            Return JSON with adaptation instructions:
            {{
                "needs_encouragement": boolean,
                "needs_simplification": boolean,
                "needs_conciseness": boolean,
                "additional_text": "text to add if any",
                "modification_type": "none|append|modify"
            }}
            """
            
            response = await self.llm_client.ainvoke(prompt)
            adaptation_plan = self._safe_json_parse(response.content)
            
            # LLM 결정에 따른 적응 적용
            if adaptation_plan.get("needs_encouragement") and len(content) > 200:
                encouragement = adaptation_plan.get("additional_text", "궁금한 점이 있으시면 언제든 말씀해 주세요!")
                content += f"\n\n{encouragement}"
            
            elif adaptation_plan.get("needs_conciseness") and len(content) > len(original) * 1.5:
                content = await self._make_more_concise(content)
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to apply LLM-based adaptations: {e}")
            # 폴백: 단순 반환
            return content
    
    async def _make_more_concise(self, content: str) -> str:
        """내용을 더 간결하게 만들기"""
        concise_prompt = f"""
        다음 내용을 전문가 수준에 맞게 더 간결하고 핵심적으로 요약하세요:
        
        {content}
        
        핵심 정보만 유지하고 불필요한 설명은 제거하세요.
        """
        
        response = await self.llm_client.ainvoke(concise_prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        return response.strip()
    
    def _get_or_create_user_profile(self, user_id: str) -> Dict:
        """사용자 프로필 로드 또는 생성"""
        if user_id in self.user_models:
            return self.user_models[user_id].get('profile', {})
        else:
            return self._get_default_profile()
    
    def _get_default_profile(self) -> Dict:
        """기본 사용자 프로필"""
        return {
            'expertise_level': 'intermediate',
            'confidence_level': 0.5,
            'interaction_count': 0,
            'communication_preferences': {
                'preferred_style': 'balanced',
                'detail_level': 'medium',
                'use_technical_terms': True,
                'needs_examples': False
            },
            'learning_characteristics': {
                'learning_style': 'unknown',
                'engagement_level': 'moderate',
                'prefers_exploration': False
            }
        }
    
    async def _analyze_new_interaction(self, query: str, feedback: Dict, current_profile: Dict) -> Dict:
        """새로운 상호작용 분석"""
        analysis_prompt = f"""
        새로운 상호작용을 분석하여 사용자 프로필 업데이트에 필요한 정보를 추출하세요.
        
        질문: {query}
        피드백: {feedback}
        현재 프로필: {current_profile}
        
        JSON 형식으로 응답하세요:
        {{
            "expertise_indicators": ["지시자1", "지시자2"],
            "learning_progress": "improving|stable|regressing",
            "engagement_level": "low|medium|high",
            "confidence": 0.0-1.0,
            "profile_adjustment_needed": true/false,
            "key_insights": ["인사이트1", "인사이트2"]
        }}
        """
        
        response = await self.llm_client.ainvoke(analysis_prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        return self._parse_json_response(response_content)
    
    async def _perform_profile_update(self, current_profile: Dict, interaction_analysis: Dict, interaction_data: Dict) -> Dict:
        """프로필 업데이트 수행"""
        updated_profile = current_profile.copy()
        
        # 전문성 수준 조정
        if interaction_analysis.get('profile_adjustment_needed', False):
            learning_progress = interaction_analysis.get('learning_progress', 'stable')
            
            if learning_progress == 'improving':
                updated_profile['confidence_level'] = min(1.0, updated_profile.get('confidence_level', 0.5) + 0.1)
            elif learning_progress == 'regressing':
                updated_profile['confidence_level'] = max(0.1, updated_profile.get('confidence_level', 0.5) - 0.1)
        
        # 상호작용 수 증가
        updated_profile['interaction_count'] = updated_profile.get('interaction_count', 0) + 1
        
        # 참여도 업데이트
        engagement = interaction_analysis.get('engagement_level', 'medium')
        updated_profile['learning_characteristics']['engagement_level'] = engagement
        
        return updated_profile
    
    def _calculate_profile_stability(self, new_profile: Dict, old_profile: Dict) -> float:
        """프로필 안정성 계산"""
        if not old_profile:
            return 0.0
        
        # 주요 속성들의 변화 정도 계산
        stability_score = 1.0
        
        if new_profile.get('expertise_level') != old_profile.get('expertise_level'):
            stability_score -= 0.3
        
        confidence_diff = abs(
            new_profile.get('confidence_level', 0.5) - old_profile.get('confidence_level', 0.5)
        )
        stability_score -= confidence_diff * 0.5
        
        return max(0.0, stability_score)
    
    def _identify_profile_changes(self, old_profile: Dict, new_profile: Dict) -> List[str]:
        """프로필 변경사항 식별"""
        changes = []
        
        if old_profile.get('expertise_level') != new_profile.get('expertise_level'):
            changes.append(f"전문성 수준: {old_profile.get('expertise_level')} → {new_profile.get('expertise_level')}")
        
        old_confidence = old_profile.get('confidence_level', 0.5)
        new_confidence = new_profile.get('confidence_level', 0.5)
        if abs(old_confidence - new_confidence) > 0.1:
            changes.append(f"신뢰도: {old_confidence:.2f} → {new_confidence:.2f}")
        
        return changes
    
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
        
        old_confidence = old_profile.get('confidence_level', 0.5)
        new_confidence = new_profile.get('confidence_level', 0.5)
        if abs(old_confidence - new_confidence) > 0.1:
            changes.append(f"신뢰도: {old_confidence:.2f} → {new_confidence:.2f}")
        
        return changes
    
    async def _update_learning_patterns(self, user_id: str, interaction_analysis: Dict, updated_profile: Dict) -> Dict:
        """학습 패턴 업데이트"""
        return {
            'learning_trajectory': interaction_analysis.get('learning_progress', 'stable'),
            'engagement_trend': interaction_analysis.get('engagement_level', 'medium'),
            'insights': interaction_analysis.get('key_insights', []),
            'pattern_stability': 'stable'
        }
    
    async def _reevaluate_adaptation_strategy(self, updated_profile: Dict, learning_patterns: Dict) -> Dict:
        """적응 전략 재평가"""
        expertise_level = updated_profile.get('expertise_level', 'intermediate')
        
        return {
            'primary_strategy': f"{expertise_level}_focused",
            'recommendations': [
                f"{expertise_level} 수준에 맞는 설명 제공",
                "상호작용 기반 동적 조정",
                "학습 패턴 반영"
            ],
            'adaptation_confidence': updated_profile.get('confidence_level', 0.5)
        }
    
    async def _generate_next_interaction_guidance(self, final_profile: Dict) -> List[str]:
        """다음 상호작용 가이드 생성"""
        expertise_level = final_profile.get('profile', {}).get('expertise_level', 'intermediate')
        
        guidance = {
            'beginner': [
                "단계별 설명 제공",
                "충분한 예시와 비유 사용",
                "격려와 지원 표현 포함"
            ],
            'intermediate': [
                "균형잡힌 설명 제공",
                "선택적 심화 내용 포함",
                "사용자 선택권 존중"
            ],
            'expert': [
                "간결하고 정확한 정보 제공",
                "고급 옵션과 대안 제시",
                "효율적인 커뮤니케이션"
            ]
        }
        
        return guidance.get(expertise_level, guidance['intermediate'])
    
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