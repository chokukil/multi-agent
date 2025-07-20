"""
Adaptive Response Generator - 적응형 응답 생성기

요구사항 17에 따른 구현:
- 사용자 수준별 설명 생성 로직
- 점진적 정보 공개 메커니즘
- 대화형 명확화 질문 생성
- 후속 분석 추천 시스템
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AdaptiveResponseGenerator:
    """
    적응형 응답 생성기
    - 사용자 수준에 맞는 동적 응답 생성
    - 점진적 정보 공개
    - 대화형 상호작용 지원
    """
    
    def __init__(self):
        """AdaptiveResponseGenerator 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.response_templates = self._initialize_response_patterns()
        logger.info("AdaptiveResponseGenerator initialized")
    
    def _initialize_response_patterns(self) -> Dict:
        """응답 패턴 초기화 - 하드코딩 없이 동적 생성"""
        return {
            'progressive_disclosure': {
                'initial': "핵심 정보 먼저 제공",
                'detailed': "관심 있는 부분 상세 설명",
                'expert': "전문적 깊이 있는 분석"
            },
            'clarification': {
                'ambiguous': "명확하지 않은 부분 질문",
                'confirmation': "이해 확인 질문",
                'exploration': "추가 탐색 제안"
            }
        }
    
    async def generate_adaptive_response(
        self, 
        knowledge_result: Dict, 
        user_profile: Dict, 
        interaction_context: Dict
    ) -> Dict:
        """
        사용자 맞춤형 적응 응답 생성
        
        Args:
            knowledge_result: 지식 통합 결과
            user_profile: 사용자 프로필 정보
            interaction_context: 상호작용 컨텍스트
            
        Returns:
            적응형 응답 결과
        """
        logger.info("Generating adaptive response")
        
        try:
            # 1. 사용자 수준별 설명 전략 결정
            explanation_strategy = await self._determine_explanation_strategy(
                user_profile, knowledge_result
            )
            
            # 2. 핵심 응답 생성
            core_response = await self._generate_core_response(
                knowledge_result, explanation_strategy
            )
            
            # 3. 점진적 공개 옵션 생성
            progressive_options = await self._create_progressive_disclosure_options(
                core_response, user_profile
            )
            
            # 4. 대화형 요소 추가
            interactive_elements = await self._add_interactive_elements(
                core_response, knowledge_result, user_profile
            )
            
            # 5. 후속 추천 생성
            follow_up_recommendations = await self._generate_follow_up_recommendations(
                knowledge_result, interaction_context
            )
            
            return {
                'core_response': core_response,
                'progressive_options': progressive_options,
                'interactive_elements': interactive_elements,
                'follow_up_recommendations': follow_up_recommendations,
                'metadata': {
                    'explanation_strategy': explanation_strategy,
                    'user_level': user_profile.get('expertise_level', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in adaptive response generation: {e}")
            raise
    
    async def _determine_explanation_strategy(self, user_profile: Dict, knowledge_result: Dict) -> Dict:
        """
        사용자 수준별 설명 전략 결정
        """
        prompt = f"""
        사용자 프로필과 지식 결과를 바탕으로 최적의 설명 전략을 결정하세요.
        
        사용자 프로필: {user_profile}
        지식 복잡도: {self._assess_knowledge_complexity(knowledge_result)}
        
        다음을 고려하여 전략을 수립하세요:
        1. 사용자의 전문성 수준
        2. 선호하는 학습 스타일
        3. 도메인 친숙도
        4. 상호작용 선호도
        
        JSON 형식으로 응답하세요:
        {{
            "primary_approach": "educational|informative|consultative|collaborative",
            "explanation_depth": {{
                "initial": "shallow|medium|deep",
                "maximum": "medium|deep|expert"
            }},
            "language_style": {{
                "technicality": "low|medium|high",
                "formality": "casual|balanced|formal",
                "use_analogies": true/false,
                "use_examples": true/false
            }},
            "interaction_mode": {{
                "proactive_clarification": true/false,
                "offer_alternatives": true/false,
                "encourage_exploration": true/false
            }},
            "pacing": "fast|moderate|slow"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _generate_core_response(self, knowledge_result: Dict, strategy: Dict) -> Dict:
        """
        핵심 응답 생성
        """
        prompt = f"""
        다음 지식 결과를 바탕으로 사용자에게 제공할 핵심 응답을 생성하세요.
        
        지식 결과: {knowledge_result.get('refined_result', {})}
        설명 전략: {strategy}
        
        전략에 따라:
        - 기술 수준: {strategy.get('language_style', {}).get('technicality', 'medium')}
        - 형식성: {strategy.get('language_style', {}).get('formality', 'balanced')}
        - 유추 사용: {strategy.get('language_style', {}).get('use_analogies', False)}
        - 예시 사용: {strategy.get('language_style', {}).get('use_examples', False)}
        
        JSON 형식으로 응답하세요:
        {{
            "summary": "한 문장 요약",
            "main_insights": [
                {{
                    "insight": "핵심 인사이트 1",
                    "explanation": "설명",
                    "confidence": "high|medium|low"
                }}
            ],
            "key_findings": [
                {{
                    "finding": "주요 발견 1",
                    "evidence": "근거",
                    "implication": "시사점"
                }}
            ],
            "recommendations": [
                {{
                    "action": "권장 조치 1",
                    "rationale": "이유",
                    "priority": "high|medium|low"
                }}
            ],
            "caveats": ["주의사항 1", "주의사항 2"]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _create_progressive_disclosure_options(self, core_response: Dict, user_profile: Dict) -> Dict:
        """
        점진적 정보 공개 옵션 생성
        """
        prompt = f"""
        핵심 응답을 바탕으로 사용자가 선택적으로 더 깊이 탐색할 수 있는 옵션을 생성하세요.
        
        핵심 응답: {core_response}
        사용자 수준: {user_profile.get('expertise_level', 'unknown')}
        
        사용자가 관심 있을 만한 추가 정보를 단계별로 제공하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "explore_deeper": [
                {{
                    "topic": "깊이 탐색할 주제 1",
                    "teaser": "이 주제에 대한 흥미로운 한 줄 설명",
                    "depth_level": "intermediate|advanced|expert",
                    "estimated_time": "2-3분"
                }}
            ],
            "see_examples": [
                {{
                    "example_type": "실제 사례",
                    "description": "사례 설명",
                    "relevance": "이 사례가 도움이 되는 이유"
                }}
            ],
            "technical_details": [
                {{
                    "aspect": "기술적 측면 1",
                    "summary": "간단한 설명",
                    "warning": "복잡도 경고 (있다면)"
                }}
            ],
            "related_topics": [
                {{
                    "topic": "관련 주제 1",
                    "connection": "현재 주제와의 연관성",
                    "value": "탐색할 가치"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _add_interactive_elements(self, core_response: Dict, knowledge_result: Dict, user_profile: Dict) -> Dict:
        """
        대화형 요소 추가
        """
        prompt = f"""
        응답에 사용자와의 상호작용을 촉진할 요소를 추가하세요.
        
        핵심 응답: {core_response}
        불확실한 영역: {knowledge_result.get('refined_result', {}).get('confidence_assessment', {}).get('needs_validation', [])}
        사용자 프로필: {user_profile}
        
        대화를 이어갈 수 있는 요소를 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "clarification_questions": [
                {{
                    "question": "명확화 질문 1",
                    "purpose": "이 질문의 목적",
                    "options": ["선택지 1", "선택지 2", "선택지 3"]
                }}
            ],
            "interactive_prompts": [
                {{
                    "prompt": "대화형 프롬프트 1",
                    "action_type": "explore|validate|customize",
                    "expected_input": "예상 입력 유형"
                }}
            ],
            "feedback_requests": [
                {{
                    "aspect": "피드백 요청 측면",
                    "question": "피드백 질문",
                    "scale": "binary|scale|open"
                }}
            ],
            "quick_actions": [
                {{
                    "action": "빠른 작업 1",
                    "description": "작업 설명",
                    "icon": "🔍|📊|💡|🎯"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _generate_follow_up_recommendations(self, knowledge_result: Dict, interaction_context: Dict) -> Dict:
        """
        후속 분석 추천 생성
        """
        prompt = f"""
        현재 분석 결과를 바탕으로 사용자에게 도움이 될 후속 분석을 추천하세요.
        
        현재 분석 결과: {knowledge_result.get('refined_result', {})}
        상호작용 이력: {interaction_context}
        
        다양한 관점에서 후속 분석을 제안하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "immediate_next_steps": [
                {{
                    "action": "즉시 실행 가능한 분석 1",
                    "value": "예상 가치",
                    "effort": "low|medium|high",
                    "prerequisites": ["전제조건 1", "전제조건 2"]
                }}
            ],
            "deeper_analysis": [
                {{
                    "analysis_type": "심화 분석 유형 1",
                    "description": "분석 설명",
                    "expected_insights": ["예상 인사이트 1", "예상 인사이트 2"],
                    "complexity": "medium|high|expert"
                }}
            ],
            "alternative_perspectives": [
                {{
                    "perspective": "대안적 관점 1",
                    "rationale": "이 관점의 가치",
                    "approach": "접근 방법"
                }}
            ],
            "long_term_exploration": [
                {{
                    "topic": "장기 탐색 주제 1",
                    "timeline": "예상 기간",
                    "potential_impact": "잠재적 영향"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    def _assess_knowledge_complexity(self, knowledge_result: Dict) -> str:
        """지식 복잡도 평가"""
        # 간단한 복잡도 평가 로직
        refined = knowledge_result.get('refined_result', {})
        insights = refined.get('refined_insights', {})
        
        complexity_score = 0
        complexity_score += len(insights.get('enhanced_patterns', []))
        complexity_score += len(insights.get('deeper_analysis', []))
        complexity_score += len(refined.get('practical_recommendations', {}).get('strategic_considerations', []))
        
        if complexity_score < 5:
            return "low"
        elif complexity_score < 10:
            return "medium"
        else:
            return "high"
    
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