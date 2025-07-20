"""
Real-time Learning System - 실시간 학습 시스템

요구사항 9와 18에 따른 구현:
- 사용자 피드백 기반 학습 로직
- 성공/실패 패턴 식별 및 일반화
- 개인화된 사용자 모델 구축
- 프라이버시 보호 학습 메커니즘
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class RealTimeLearningSystem:
    """
    실시간 학습 시스템
    - 상호작용을 통한 지속적 개선
    - 프라이버시를 보호하면서 학습
    - 패턴 인식 및 일반화
    """
    
    def __init__(self):
        """RealTimeLearningSystem 초기화"""
        self.learning_history = defaultdict(list)
        self.pattern_database = defaultdict(dict)
        self.user_models = {}
        self.success_patterns = []
        self.failure_patterns = []
        self.learning_rate = 0.1
        logger.info("RealTimeLearningSystem initialized")
    
    async def learn_from_interaction(self, interaction_data: Dict) -> Dict:
        """
        상호작용으로부터 학습
        
        Args:
            interaction_data: 상호작용 데이터
            
        Returns:
            학습 결과
        """
        logger.info("Learning from interaction")
        
        try:
            # 1. 상호작용 데이터 익명화
            anonymized_data = self._anonymize_interaction_data(interaction_data)
            
            # 2. 패턴 추출
            patterns = await self._extract_patterns(anonymized_data)
            
            # 3. 성공/실패 평가
            evaluation = await self._evaluate_interaction_success(anonymized_data)
            
            # 4. 패턴 데이터베이스 업데이트
            await self._update_pattern_database(patterns, evaluation)
            
            # 5. 사용자 모델 업데이트
            await self._update_user_model(anonymized_data, patterns, evaluation)
            
            # 6. 학습 요약 생성
            learning_summary = await self._generate_learning_summary(
                patterns, evaluation, anonymized_data
            )
            
            return {
                'success': True,
                'patterns_learned': len(patterns),
                'evaluation': evaluation,
                'learning_summary': learning_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in learning from interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _anonymize_interaction_data(self, data: Dict) -> Dict:
        """
        프라이버시 보호를 위한 데이터 익명화
        """
        # 민감한 정보 제거 및 해시화
        anonymized = {
            'interaction_id': self._generate_anonymous_id(data),
            'timestamp': data.get('timestamp'),
            'data_characteristics': data.get('data_characteristics', {}),
            'user_profile_hash': self._hash_user_profile(data.get('user_profile', {})),
            'query_type': self._categorize_query(data.get('query', '')),
            'response_metadata': self._extract_response_metadata(data.get('response', {}))
        }
        
        return anonymized
    
    def _generate_anonymous_id(self, data: Dict) -> str:
        """익명 ID 생성"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_user_profile(self, profile: Dict) -> str:
        """사용자 프로필 해시화"""
        if not profile:
            return "anonymous"
        
        # 전문성 수준과 도메인 친숙도만 유지
        simplified = {
            'expertise': profile.get('expertise_level', 'unknown'),
            'domain_familiarity': profile.get('domain_familiarity', 'unknown')
        }
        return hashlib.sha256(json.dumps(simplified).encode()).hexdigest()[:8]
    
    def _categorize_query(self, query: str) -> str:
        """쿼리 카테고리화 - 실제 내용은 저장하지 않음"""
        if not query:
            return "empty"
        
        # 쿼리 특성만 추출
        return {
            'length': 'short' if len(query) < 50 else 'medium' if len(query) < 200 else 'long',
            'complexity': self._assess_query_complexity(query),
            'has_technical_terms': self._has_technical_terms(query)
        }
    
    def _assess_query_complexity(self, query: str) -> str:
        """쿼리 복잡도 평가"""
        # 간단한 복잡도 평가
        if '?' in query and ('어떻게' in query or 'how' in query.lower()):
            return 'complex'
        elif '무엇' in query or 'what' in query.lower():
            return 'medium'
        else:
            return 'simple'
    
    def _has_technical_terms(self, query: str) -> bool:
        """기술 용어 포함 여부"""
        # 매우 일반적인 기술 용어 패턴만 확인
        technical_indicators = ['분석', '데이터', '패턴', '트렌드', 'analysis', 'pattern', 'trend']
        return any(term in query.lower() for term in technical_indicators)
    
    def _extract_response_metadata(self, response: Dict) -> Dict:
        """응답 메타데이터 추출"""
        return {
            'has_visualizations': bool(response.get('visualizations')),
            'has_recommendations': bool(response.get('recommendations')),
            'confidence_level': response.get('metadata', {}).get('confidence_level', 'unknown'),
            'response_complexity': self._assess_response_complexity(response)
        }
    
    def _assess_response_complexity(self, response: Dict) -> str:
        """응답 복잡도 평가"""
        core = response.get('core_response', {})
        insights = core.get('main_insights', [])
        recommendations = core.get('recommendations', [])
        
        total_items = len(insights) + len(recommendations)
        
        if total_items < 3:
            return 'simple'
        elif total_items < 6:
            return 'medium'
        else:
            return 'complex'
    
    async def _extract_patterns(self, data: Dict) -> List[Dict]:
        """
        상호작용에서 패턴 추출
        """
        patterns = []
        
        # 쿼리-응답 패턴
        query_response_pattern = {
            'type': 'query_response',
            'query_type': data.get('query_type'),
            'response_complexity': data.get('response_metadata', {}).get('response_complexity'),
            'user_level': data.get('user_profile_hash'),
            'success_indicators': self._extract_success_indicators(data)
        }
        patterns.append(query_response_pattern)
        
        # 사용자 수준-설명 깊이 패턴
        if data.get('user_profile_hash') != 'anonymous':
            user_level_pattern = {
                'type': 'user_level_adaptation',
                'user_profile': data.get('user_profile_hash'),
                'preferred_complexity': data.get('response_metadata', {}).get('response_complexity'),
                'engagement_level': self._estimate_engagement_level(data)
            }
            patterns.append(user_level_pattern)
        
        # 데이터 특성-분석 방법 패턴
        data_analysis_pattern = {
            'type': 'data_analysis',
            'data_characteristics': data.get('data_characteristics', {}),
            'effective_methods': self._identify_effective_methods(data)
        }
        patterns.append(data_analysis_pattern)
        
        return patterns
    
    def _extract_success_indicators(self, data: Dict) -> Dict:
        """성공 지표 추출"""
        return {
            'high_confidence': data.get('response_metadata', {}).get('confidence_level') == 'high',
            'has_actionable_insights': data.get('response_metadata', {}).get('has_recommendations', False),
            'appropriate_complexity': True  # 실제 피드백이 있을 때 업데이트
        }
    
    def _estimate_engagement_level(self, data: Dict) -> str:
        """참여도 추정"""
        # 응답의 다양한 요소 활용 여부로 참여도 추정
        metadata = data.get('response_metadata', {})
        
        engagement_score = 0
        if metadata.get('has_visualizations'):
            engagement_score += 1
        if metadata.get('has_recommendations'):
            engagement_score += 1
        if metadata.get('response_complexity') in ['medium', 'complex']:
            engagement_score += 1
            
        if engagement_score >= 2:
            return 'high'
        elif engagement_score == 1:
            return 'medium'
        else:
            return 'low'
    
    def _identify_effective_methods(self, data: Dict) -> List[str]:
        """효과적인 분석 방법 식별"""
        methods = []
        
        # 데이터 특성에 따른 효과적인 방법 추론
        data_chars = data.get('data_characteristics', {})
        
        if data_chars.get('type') in ['DataFrame', 'list', 'dict']:
            methods.append('structured_analysis')
        
        if data_chars.get('patterns'):
            methods.append('pattern_recognition')
            
        return methods
    
    async def _evaluate_interaction_success(self, data: Dict) -> Dict:
        """
        상호작용 성공 여부 평가
        """
        evaluation = {
            'overall_success': True,  # 기본값, 실제 피드백으로 업데이트
            'confidence_score': 0.0,
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # 신뢰도 기반 평가
        confidence = data.get('response_metadata', {}).get('confidence_level', 'unknown')
        if confidence == 'high':
            evaluation['confidence_score'] = 0.9
            evaluation['strengths'].append('high_confidence_response')
        elif confidence == 'medium':
            evaluation['confidence_score'] = 0.6
        else:
            evaluation['confidence_score'] = 0.3
            evaluation['areas_for_improvement'].append('improve_confidence')
        
        # 응답 완전성 평가
        if data.get('response_metadata', {}).get('has_recommendations'):
            evaluation['strengths'].append('actionable_recommendations')
        else:
            evaluation['areas_for_improvement'].append('add_actionable_insights')
        
        # 전반적 성공 여부 결정
        evaluation['overall_success'] = evaluation['confidence_score'] > 0.5
        
        return evaluation
    
    async def _update_pattern_database(self, patterns: List[Dict], evaluation: Dict):
        """
        패턴 데이터베이스 업데이트
        """
        for pattern in patterns:
            pattern_key = f"{pattern['type']}_{json.dumps(pattern, sort_keys=True)[:50]}"
            
            if pattern_key not in self.pattern_database:
                self.pattern_database[pattern_key] = {
                    'pattern': pattern,
                    'occurrences': 0,
                    'success_rate': 0.0,
                    'last_updated': datetime.now()
                }
            
            # 발생 횟수 증가
            self.pattern_database[pattern_key]['occurrences'] += 1
            
            # 성공률 업데이트 (이동 평균)
            current_success = 1.0 if evaluation['overall_success'] else 0.0
            old_rate = self.pattern_database[pattern_key]['success_rate']
            new_rate = old_rate * (1 - self.learning_rate) + current_success * self.learning_rate
            self.pattern_database[pattern_key]['success_rate'] = new_rate
            
            # 타임스탬프 업데이트
            self.pattern_database[pattern_key]['last_updated'] = datetime.now()
            
            # 성공/실패 패턴 분류
            if new_rate > 0.7 and self.pattern_database[pattern_key]['occurrences'] > 5:
                if pattern not in self.success_patterns:
                    self.success_patterns.append(pattern)
            elif new_rate < 0.3 and self.pattern_database[pattern_key]['occurrences'] > 5:
                if pattern not in self.failure_patterns:
                    self.failure_patterns.append(pattern)
    
    async def _update_user_model(self, data: Dict, patterns: List[Dict], evaluation: Dict):
        """
        사용자 모델 업데이트
        """
        user_hash = data.get('user_profile_hash', 'anonymous')
        
        if user_hash not in self.user_models:
            self.user_models[user_hash] = {
                'interactions': 0,
                'preferences': defaultdict(float),
                'success_rate': 0.0,
                'last_interaction': datetime.now()
            }
        
        model = self.user_models[user_hash]
        
        # 상호작용 횟수 증가
        model['interactions'] += 1
        
        # 선호도 업데이트
        response_complexity = data.get('response_metadata', {}).get('response_complexity', 'medium')
        model['preferences'][f'complexity_{response_complexity}'] += 1
        
        # 성공률 업데이트
        current_success = 1.0 if evaluation['overall_success'] else 0.0
        model['success_rate'] = (model['success_rate'] * (model['interactions'] - 1) + current_success) / model['interactions']
        
        # 마지막 상호작용 시간 업데이트
        model['last_interaction'] = datetime.now()
    
    async def _generate_learning_summary(self, patterns: List[Dict], evaluation: Dict, data: Dict) -> Dict:
        """
        학습 요약 생성
        """
        return {
            'patterns_identified': len(patterns),
            'pattern_types': list(set(p['type'] for p in patterns)),
            'interaction_success': evaluation['overall_success'],
            'confidence_level': evaluation['confidence_score'],
            'key_learnings': {
                'effective_approaches': evaluation.get('strengths', []),
                'improvement_areas': evaluation.get('areas_for_improvement', [])
            },
            'user_segment': self._identify_user_segment(data.get('user_profile_hash', 'anonymous')),
            'timestamp': datetime.now().isoformat()
        }
    
    def _identify_user_segment(self, user_hash: str) -> str:
        """사용자 세그먼트 식별"""
        if user_hash == 'anonymous':
            return 'anonymous'
        
        model = self.user_models.get(user_hash, {})
        
        if model.get('interactions', 0) < 3:
            return 'new_user'
        elif model.get('success_rate', 0) > 0.8:
            return 'power_user'
        elif model.get('success_rate', 0) < 0.5:
            return 'struggling_user'
        else:
            return 'regular_user'
    
    def get_learning_insights(self) -> Dict:
        """
        학습된 인사이트 조회
        """
        return {
            'total_patterns': len(self.pattern_database),
            'success_patterns': len(self.success_patterns),
            'failure_patterns': len(self.failure_patterns),
            'user_segments': self._get_user_segments_summary(),
            'top_successful_approaches': self._get_top_patterns(self.success_patterns, 5),
            'common_failure_points': self._get_top_patterns(self.failure_patterns, 5),
            'learning_statistics': {
                'total_interactions': sum(p['occurrences'] for p in self.pattern_database.values()),
                'average_success_rate': self._calculate_average_success_rate(),
                'active_users': len(self.user_models)
            }
        }
    
    def _get_user_segments_summary(self) -> Dict:
        """사용자 세그먼트 요약"""
        segments = defaultdict(int)
        
        for user_hash, model in self.user_models.items():
            segment = self._identify_user_segment(user_hash)
            segments[segment] += 1
            
        return dict(segments)
    
    def _get_top_patterns(self, patterns: List[Dict], limit: int) -> List[Dict]:
        """상위 패턴 조회"""
        # 패턴을 성공률 기준으로 정렬
        pattern_scores = []
        
        for pattern in patterns[:limit]:
            pattern_key = f"{pattern['type']}_{json.dumps(pattern, sort_keys=True)[:50]}"
            if pattern_key in self.pattern_database:
                score = self.pattern_database[pattern_key]['success_rate']
                pattern_scores.append({
                    'pattern': pattern,
                    'success_rate': score,
                    'occurrences': self.pattern_database[pattern_key]['occurrences']
                })
        
        return sorted(pattern_scores, key=lambda x: x['success_rate'], reverse=True)[:limit]
    
    def _calculate_average_success_rate(self) -> float:
        """평균 성공률 계산"""
        if not self.pattern_database:
            return 0.0
            
        total_success = sum(p['success_rate'] * p['occurrences'] for p in self.pattern_database.values())
        total_occurrences = sum(p['occurrences'] for p in self.pattern_database.values())
        
        return total_success / total_occurrences if total_occurrences > 0 else 0.0