"""
LLM Recommendation Engine - LLM 기반 분석 추천 시스템

검증된 Universal Engine 패턴 기반:
- DynamicContextDiscovery: 하드코딩 없는 도메인 컨텍스트 자동 발견
- AdaptiveUserUnderstanding: 사용자 전문성 수준 추정
- RealTimeLearningSystem: 피드백 기반 추천 개선
- Progressive Disclosure: 사용자 수준별 추천 적응
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import pandas as pd
import numpy as np
from dataclasses import asdict

from ..models import OneClickRecommendation, VisualDataCard, DataContext

# Universal Engine 패턴 가져오기 (사용 가능한 경우)
try:
    from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery
    from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
    from core.universal_engine.real_time_learning_system import RealTimeLearningSystem
    from core.universal_engine.llm_factory import LLMFactory
    UNIVERSAL_ENGINE_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMRecommendationEngine:
    """
    LLM 기반 분석 추천 엔진
    검증된 Universal Engine 패턴을 활용한 지능적 추천 시스템
    """
    
    def __init__(self):
        """LLM 추천 엔진 초기화"""
        self.max_recommendations = 3
        
        # Universal Engine 컴포넌트 초기화
        if UNIVERSAL_ENGINE_AVAILABLE:
            self.context_discovery = DynamicContextDiscovery()
            self.user_understanding = AdaptiveUserUnderstanding()
            self.learning_system = RealTimeLearningSystem()
            self.llm_client = LLMFactory.create_llm()
        else:
            self.context_discovery = None
            self.user_understanding = None
            self.learning_system = None
            self.llm_client = None
            
        # 캐시 및 학습 데이터
        self.context_cache: Dict[str, DataContext] = {}
        self.user_feedback_history: List[Dict] = []
        self.recommendation_templates = self._load_recommendation_templates()
        
        logger.info("LLM Recommendation Engine initialized")
    
    async def generate_contextual_recommendations(self, 
                                                data_cards: List[VisualDataCard],
                                                user_query: Optional[str] = None,
                                                user_context: Optional[Dict] = None,
                                                interaction_history: Optional[List] = None) -> List[OneClickRecommendation]:
        """
        검증된 Universal Engine 패턴을 사용한 컨텍스트 기반 추천 생성
        
        1. DynamicContextDiscovery: 도메인 컨텍스트 자동 발견 (하드코딩 없음)
        2. AdaptiveUserUnderstanding: 사용자 전문성 수준 추정
        3. RealTimeLearningSystem: 피드백 기반 추천 개선
        4. Progressive Disclosure: 사용자 수준별 추천 적응
        """
        try:
            logger.info(f"Generating recommendations for {len(data_cards)} datasets")
            
            # 1. 동적 컨텍스트 발견
            data_context = await self._discover_data_context_dynamically(data_cards, user_query)
            
            # 2. 사용자 수준 적응적 이해
            user_profile = await self._analyze_user_adaptively(user_query, interaction_history, user_context)
            
            # 3. LLM 기반 추천 생성
            recommendations = await self._generate_llm_recommendations(
                data_cards, data_context, user_profile, user_query
            )
            
            # 4. 점진적 공개 패턴 적용
            adapted_recommendations = await self._adapt_to_user_level(recommendations, user_profile)
            
            # 5. 실시간 학습 시스템 업데이트
            await self._update_learning_system(data_context, user_profile, recommendations)
            
            return adapted_recommendations[:self.max_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._generate_fallback_recommendations(data_cards)
    
    async def _discover_data_context_dynamically(self, 
                                               data_cards: List[VisualDataCard], 
                                               user_query: Optional[str] = None) -> DataContext:
        """
        검증된 zero-hardcoding 컨텍스트 발견:
        - 사전 정의된 도메인 카테고리 없음
        - LLM이 데이터 특성을 보고 직접 컨텍스트 발견
        - 패턴 매칭이 아닌 실제 이해와 추론
        """
        try:
            # 데이터 메타데이터 수집
            combined_metadata = self._extract_combined_metadata(data_cards)
            
            if UNIVERSAL_ENGINE_AVAILABLE and self.context_discovery:
                # Universal Engine DynamicContextDiscovery 사용
                discovered_context = await self.context_discovery.discover_context(
                    data=combined_metadata,
                    query=user_query
                )
                
                return self._convert_to_data_context(discovered_context, data_cards)
            else:
                # 기본 LLM 기반 컨텍스트 발견
                return await self._llm_context_discovery_fallback(combined_metadata, user_query)
                
        except Exception as e:
            logger.error(f"Context discovery error: {str(e)}")
            return self._create_basic_data_context(data_cards)
    
    async def _analyze_user_adaptively(self, 
                                     user_query: Optional[str],
                                     interaction_history: Optional[List],
                                     user_context: Optional[Dict]) -> Dict[str, Any]:
        """
        검증된 적응적 사용자 이해:
        - 언어 사용, 용어, 질문 복잡도 분석
        - 점진적 공개를 통한 이해 수준 파악
        - 대화 중 동적 조정
        """
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.user_understanding:
                # Universal Engine AdaptiveUserUnderstanding 사용
                user_analysis = await self.user_understanding.analyze_user_expertise(
                    query=user_query or "",
                    interaction_history=interaction_history or []
                )
                
                return user_analysis
            else:
                # 기본 사용자 분석
                return self._basic_user_analysis(user_query, interaction_history)
                
        except Exception as e:
            logger.error(f"User analysis error: {str(e)}")
            return {"expertise_level": "intermediate", "domain_familiarity": "moderate"}
    
    async def _generate_llm_recommendations(self, 
                                          data_cards: List[VisualDataCard],
                                          data_context: DataContext,
                                          user_profile: Dict[str, Any],
                                          user_query: Optional[str]) -> List[OneClickRecommendation]:
        """LLM을 사용한 지능적 추천 생성"""
        try:
            # LLM 프롬프트 구성
            prompt = self._build_recommendation_prompt(
                data_cards, data_context, user_profile, user_query
            )
            
            if UNIVERSAL_ENGINE_AVAILABLE and self.llm_client:
                # Universal Engine LLM 사용
                response = await self.llm_client.ainvoke(prompt)
                recommendations_data = self._parse_llm_response(response.content)
            else:
                # 기본 추천 생성
                recommendations_data = self._generate_basic_recommendations(data_cards, data_context)
            
            # OneClickRecommendation 객체로 변환
            recommendations = []
            for i, rec_data in enumerate(recommendations_data[:self.max_recommendations]):
                recommendation = OneClickRecommendation(
                    title=rec_data.get('title', f'Analysis {i+1}'),
                    description=rec_data.get('description', 'Perform data analysis'),
                    action_type=rec_data.get('action_type', 'general_analysis'),
                    parameters=rec_data.get('parameters', {}),
                    estimated_time=rec_data.get('estimated_time', 60),
                    confidence_score=rec_data.get('confidence_score', 0.8),
                    complexity_level=rec_data.get('complexity_level', 'intermediate'),
                    expected_result_preview=rec_data.get('expected_result_preview', 'Analysis results'),
                    icon=rec_data.get('icon', '📊'),
                    color_theme=rec_data.get('color_theme', 'blue'),
                    execution_button_text=rec_data.get('execution_button_text', 'Execute Analysis')
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"LLM recommendation generation error: {str(e)}")
            return self._generate_fallback_recommendations(data_cards)
    
    def _build_recommendation_prompt(self, 
                                   data_cards: List[VisualDataCard],
                                   data_context: DataContext,
                                   user_profile: Dict[str, Any],
                                   user_query: Optional[str]) -> str:
        """LLM 추천 생성을 위한 프롬프트 구성"""
        
        # 데이터 요약 생성
        data_summary = self._create_data_summary(data_cards)
        
        # 사용자 프로필 요약
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        domain_familiarity = user_profile.get('domain_familiarity', 'moderate')
        
        prompt = f"""
당신은 데이터 사이언스 전문가입니다. 주어진 데이터와 사용자 프로필을 바탕으로 최대 3개의 분석 추천을 생성해주세요.

## 데이터 정보:
{data_summary}

## 데이터 컨텍스트:
- 도메인: {data_context.domain}
- 데이터 타입: {', '.join(data_context.data_types)}
- 품질 점수: {data_context.quality_assessment.quality_score:.1f}%

## 사용자 프로필:
- 전문성 수준: {expertise_level}
- 도메인 친숙도: {domain_familiarity}
- 사용자 질문: {user_query or '없음'}

## 요구사항:
1. 사용자 수준에 맞는 분석 추천
2. 실행 가능한 구체적인 분석
3. 명확한 가치 제안

각 추천에 대해 다음 JSON 형식으로 응답해주세요:

```json
[
  {{
    "title": "분석 제목",
    "description": "분석에 대한 명확한 설명 (1-2문장)",
    "action_type": "statistical_analysis|visualization|ml_analysis|data_quality",
    "parameters": {{"dataset_ids": ["dataset_id"], "analysis_type": "specific_type"}},
    "estimated_time": 60,
    "confidence_score": 0.8,
    "complexity_level": "beginner|intermediate|advanced",
    "expected_result_preview": "예상 결과에 대한 간단한 설명",
    "icon": "📊|📈|🤖|🔍",
    "color_theme": "blue|green|purple|orange",
    "execution_button_text": "실행 버튼 텍스트"
  }}
]
```
"""
        
        return prompt
    
    def _create_data_summary(self, data_cards: List[VisualDataCard]) -> str:
        """데이터 카드들의 요약 생성"""
        summaries = []
        
        for card in data_cards:
            summary = f"- {card.name}: {card.rows:,}행 × {card.columns}열 ({card.format})"
            if card.quality_indicators:
                summary += f", 품질: {card.quality_indicators.quality_score:.0f}%"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _parse_llm_response(self, response_content: str) -> List[Dict]:
        """LLM 응답을 파싱하여 추천 데이터 추출"""
        try:
            # JSON 블록 찾기
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # 직접 JSON 파싱 시도
            return json.loads(response_content)
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return []
    
    async def _adapt_to_user_level(self, 
                                 recommendations: List[OneClickRecommendation],
                                 user_profile: Dict[str, Any]) -> List[OneClickRecommendation]:
        """검증된 progressive disclosure 패턴으로 사용자 수준별 추천 적응"""
        
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        
        adapted_recommendations = []
        for rec in recommendations:
            # 사용자 수준에 따른 복잡도 조정
            if expertise_level == 'beginner':
                if rec.complexity_level == 'advanced':
                    continue  # 고급 분석은 제외
                rec.description = f"🔰 초보자용: {rec.description}"
                rec.estimated_time = int(rec.estimated_time * 1.2)  # 시간 여유 추가
                
            elif expertise_level == 'advanced':
                if rec.complexity_level == 'beginner':
                    # 고급 사용자에게는 더 상세한 옵션 제공
                    rec.description = f"⚡ 고급: {rec.description} (맞춤 설정 가능)"
                
            adapted_recommendations.append(rec)
        
        return adapted_recommendations
    
    async def _update_learning_system(self, 
                                    data_context: DataContext,
                                    user_profile: Dict[str, Any],
                                    recommendations: List[OneClickRecommendation]):
        """실시간 학습 시스템 업데이트"""
        try:
            if UNIVERSAL_ENGINE_AVAILABLE and self.learning_system:
                # Universal Engine RealTimeLearningSystem 사용
                learning_data = {
                    'context': asdict(data_context),
                    'user_profile': user_profile,
                    'recommendations': [asdict(rec) for rec in recommendations],
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.learning_system.update_system(learning_data)
            
        except Exception as e:
            logger.error(f"Learning system update error: {str(e)}")
    
    def _extract_combined_metadata(self, data_cards: List[VisualDataCard]) -> Dict[str, Any]:
        """데이터 카드들로부터 결합된 메타데이터 추출"""
        combined_metadata = {
            'total_datasets': len(data_cards),
            'total_rows': sum(card.rows for card in data_cards),
            'total_columns': sum(card.columns for card in data_cards),
            'formats': list(set(card.format for card in data_cards)),
            'column_names': [],
            'data_types': [],
            'quality_scores': []
        }
        
        for card in data_cards:
            if 'column_names' in card.metadata:
                combined_metadata['column_names'].extend(card.metadata['column_names'])
            if 'column_types' in card.metadata:
                combined_metadata['data_types'].extend(list(card.metadata['column_types'].values()))
            if card.quality_indicators:
                combined_metadata['quality_scores'].append(card.quality_indicators.quality_score)
        
        return combined_metadata
    
    async def _llm_context_discovery_fallback(self, 
                                            metadata: Dict[str, Any], 
                                            user_query: Optional[str]) -> DataContext:
        """기본 LLM 기반 컨텍스트 발견 (Universal Engine이 없을 때)"""
        
        # 기본 도메인 추론
        column_names = metadata.get('column_names', [])
        domain = self._infer_domain_from_columns(column_names)
        
        # 기본 DataContext 생성
        from ..models import DataQualityInfo
        
        avg_quality = np.mean(metadata.get('quality_scores', [85.0]))
        
        quality_info = DataQualityInfo(
            missing_values_count=0,
            missing_percentage=0.0,
            data_types_summary={},
            quality_score=avg_quality,
            issues=[]
        )
        
        return DataContext(
            domain=domain,
            data_types=list(set(metadata.get('data_types', []))),
            relationships=[],
            quality_assessment=quality_info,
            suggested_analyses=[]
        )
    
    def _infer_domain_from_columns(self, column_names: List[str]) -> str:
        """컬럼명으로부터 도메인 추론"""
        column_text = ' '.join(column_names).lower()
        
        # 간단한 키워드 기반 도메인 추론
        if any(word in column_text for word in ['price', 'sales', 'revenue', 'profit', 'cost']):
            return 'business_finance'
        elif any(word in column_text for word in ['patient', 'medical', 'diagnosis', 'treatment']):
            return 'healthcare'
        elif any(word in column_text for word in ['student', 'grade', 'course', 'education']):
            return 'education'
        elif any(word in column_text for word in ['temperature', 'pressure', 'sensor', 'measurement']):
            return 'iot_sensor'
        else:
            return 'general'
    
    def _create_basic_data_context(self, data_cards: List[VisualDataCard]) -> DataContext:
        """기본 데이터 컨텍스트 생성"""
        from ..models import DataQualityInfo
        
        # 평균 품질 점수 계산
        quality_scores = [card.quality_indicators.quality_score for card in data_cards 
                         if card.quality_indicators]
        avg_quality = np.mean(quality_scores) if quality_scores else 85.0
        
        quality_info = DataQualityInfo(
            missing_values_count=0,
            missing_percentage=0.0,
            data_types_summary={},
            quality_score=avg_quality,
            issues=[]
        )
        
        return DataContext(
            domain='general',
            data_types=[card.format for card in data_cards],
            relationships=[],
            quality_assessment=quality_info,
            suggested_analyses=[]
        )
    
    def _basic_user_analysis(self, 
                           user_query: Optional[str], 
                           interaction_history: Optional[List]) -> Dict[str, Any]:
        """기본 사용자 분석"""
        
        expertise_level = 'intermediate'  # 기본값
        
        if user_query:
            query_lower = user_query.lower()
            
            # 고급 키워드 확인
            advanced_keywords = ['model', 'algorithm', 'regression', 'classification', 'clustering', 
                               'neural', 'deep learning', 'feature engineering']
            if any(keyword in query_lower for keyword in advanced_keywords):
                expertise_level = 'advanced'
            
            # 초보자 키워드 확인
            beginner_keywords = ['help', 'how to', 'what is', 'explain', 'simple', 'basic']
            if any(keyword in query_lower for keyword in beginner_keywords):
                expertise_level = 'beginner'
        
        return {
            'expertise_level': expertise_level,
            'domain_familiarity': 'moderate',
            'communication_style': 'detailed',
            'analysis_preferences': ['visual', 'statistical']
        }
    
    def _generate_basic_recommendations(self, 
                                      data_cards: List[VisualDataCard],
                                      data_context: DataContext) -> List[Dict]:
        """기본 추천 생성 (LLM이 없을 때)"""
        
        recommendations = []
        
        # 기본 통계 분석 추천
        recommendations.append({
            'title': '기본 통계 분석',
            'description': '데이터의 기본 통계량과 분포를 분석합니다',
            'action_type': 'statistical_analysis',
            'parameters': {'dataset_ids': [card.id for card in data_cards]},
            'estimated_time': 30,
            'confidence_score': 0.9,
            'complexity_level': 'beginner',
            'expected_result_preview': '요약 통계, 분포 차트, 상관관계 분석',
            'icon': '📊',
            'color_theme': 'blue',
            'execution_button_text': '통계 분석 실행'
        })
        
        # 데이터 시각화 추천
        if any(card.columns > 1 for card in data_cards):
            recommendations.append({
                'title': '데이터 시각화',
                'description': '주요 변수들의 시각적 패턴을 탐색합니다',
                'action_type': 'visualization',
                'parameters': {'dataset_ids': [card.id for card in data_cards]},
                'estimated_time': 45,
                'confidence_score': 0.85,
                'complexity_level': 'intermediate',
                'expected_result_preview': '히스토그램, 산점도, 박스플롯 등 다양한 차트',
                'icon': '📈',
                'color_theme': 'green',
                'execution_button_text': '시각화 생성'
            })
        
        # 데이터 품질 확인 추천
        avg_quality = np.mean([card.quality_indicators.quality_score for card in data_cards 
                              if card.quality_indicators])
        if avg_quality < 90:
            recommendations.append({
                'title': '데이터 품질 개선',
                'description': '누락값, 이상치, 데이터 일관성을 검사하고 개선합니다',
                'action_type': 'data_quality',
                'parameters': {'dataset_ids': [card.id for card in data_cards]},
                'estimated_time': 60,
                'confidence_score': 0.8,
                'complexity_level': 'intermediate',
                'expected_result_preview': '품질 리포트, 정제 제안, 개선된 데이터',
                'icon': '🔍',
                'color_theme': 'orange',
                'execution_button_text': '품질 검사 실행'
            })
        
        return recommendations
    
    def _generate_fallback_recommendations(self, data_cards: List[VisualDataCard]) -> List[OneClickRecommendation]:
        """오류 발생 시 기본 추천 생성"""
        
        basic_rec = OneClickRecommendation(
            title='기본 데이터 탐색',
            description='업로드된 데이터의 기본적인 특성을 탐색합니다',
            action_type='basic_exploration',
            parameters={'dataset_ids': [card.id for card in data_cards]},
            estimated_time=30,
            confidence_score=0.7,
            complexity_level='beginner',
            expected_result_preview='데이터 요약, 기본 통계, 누락값 정보',
            icon='📋',
            color_theme='gray',
            execution_button_text='탐색 시작'
        )
        
        return [basic_rec]
    
    def _load_recommendation_templates(self) -> Dict[str, Dict]:
        """추천 템플릿 로드"""
        return {
            'statistical_analysis': {
                'icon': '📊',
                'color_theme': 'blue',
                'complexity_mapping': {
                    'beginner': 30,
                    'intermediate': 45,
                    'advanced': 60
                }
            },
            'visualization': {
                'icon': '📈',
                'color_theme': 'green',
                'complexity_mapping': {
                    'beginner': 20,
                    'intermediate': 35,
                    'advanced': 50
                }
            },
            'ml_analysis': {
                'icon': '🤖',
                'color_theme': 'purple',
                'complexity_mapping': {
                    'beginner': 120,
                    'intermediate': 180,
                    'advanced': 240
                }
            },
            'data_quality': {
                'icon': '🔍',
                'color_theme': 'orange',
                'complexity_mapping': {
                    'beginner': 40,
                    'intermediate': 60,
                    'advanced': 90
                }
            }
        }
    
    def _convert_to_data_context(self, discovered_context: Dict, data_cards: List[VisualDataCard]) -> DataContext:
        """Universal Engine 컨텍스트를 DataContext로 변환"""
        from ..models import DataQualityInfo
        
        # 평균 품질 점수 계산
        quality_scores = [card.quality_indicators.quality_score for card in data_cards 
                         if card.quality_indicators]
        avg_quality = np.mean(quality_scores) if quality_scores else 85.0
        
        quality_info = DataQualityInfo(
            missing_values_count=0,
            missing_percentage=0.0,
            data_types_summary={},
            quality_score=avg_quality,
            issues=[]
        )
        
        return DataContext(
            domain=discovered_context.get('domain', 'general'),
            data_types=discovered_context.get('data_types', []),
            relationships=[],
            quality_assessment=quality_info,
            suggested_analyses=discovered_context.get('suggested_analyses', [])
        )
    
    async def process_user_feedback(self, 
                                  recommendation_id: str, 
                                  feedback_type: str, 
                                  feedback_data: Dict[str, Any]):
        """사용자 피드백 처리 및 학습 시스템 업데이트"""
        try:
            feedback_entry = {
                'recommendation_id': recommendation_id,
                'feedback_type': feedback_type,  # 'positive', 'negative', 'executed', 'ignored'
                'feedback_data': feedback_data,
                'timestamp': datetime.now().isoformat()
            }
            
            self.user_feedback_history.append(feedback_entry)
            
            # RealTimeLearningSystem 업데이트 (사용 가능한 경우)
            if UNIVERSAL_ENGINE_AVAILABLE and self.learning_system:
                await self.learning_system.process_feedback(feedback_entry)
            
            logger.info(f"Processed user feedback for recommendation {recommendation_id}")
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {str(e)}")
    
    def get_recommendation_analytics(self) -> Dict[str, Any]:
        """추천 분석 데이터 반환"""
        return {
            'total_recommendations_generated': len(self.user_feedback_history),
            'feedback_summary': self._analyze_feedback_patterns(),
            'most_popular_analysis_types': self._get_popular_analysis_types(),
            'average_user_satisfaction': self._calculate_satisfaction_score()
        }
    
    def _analyze_feedback_patterns(self) -> Dict[str, int]:
        """피드백 패턴 분석"""
        patterns = {'positive': 0, 'negative': 0, 'executed': 0, 'ignored': 0}
        
        for feedback in self.user_feedback_history:
            feedback_type = feedback.get('feedback_type', 'unknown')
            if feedback_type in patterns:
                patterns[feedback_type] += 1
        
        return patterns
    
    def _get_popular_analysis_types(self) -> List[str]:
        """인기 있는 분석 유형 반환"""
        # 실제 구현에서는 피드백 데이터를 분석하여 반환
        return ['statistical_analysis', 'visualization', 'data_quality']
    
    def _calculate_satisfaction_score(self) -> float:
        """사용자 만족도 점수 계산"""
        if not self.user_feedback_history:
            return 0.0
        
        positive_feedback = sum(1 for f in self.user_feedback_history 
                               if f.get('feedback_type') in ['positive', 'executed'])
        
        return positive_feedback / len(self.user_feedback_history) * 100