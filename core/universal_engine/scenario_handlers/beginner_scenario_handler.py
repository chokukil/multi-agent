"""
Beginner Scenario Handler - 초보자 사용자를 위한 시나리오 처리

Requirement 15 구현:
- "이 데이터 파일이 뭘 말하는지 전혀 모르겠어요. 도움 주세요." 시나리오 처리
- 초보자 친화적 설명과 단계별 가이드 제공
- 기술 용어 최소화 및 직관적 해석 중심
- 학습 지향적 응답 생성
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ExplanationLevel(Enum):
    """설명 수준"""
    VERY_SIMPLE = "very_simple"
    SIMPLE = "simple"
    BASIC = "basic"
    GUIDED = "guided"


class LearningStyle(Enum):
    """학습 스타일"""
    VISUAL = "visual"
    STEP_BY_STEP = "step_by_step"
    EXAMPLE_BASED = "example_based"
    STORY_TELLING = "story_telling"


@dataclass
class BeginnerGuidance:
    """초보자 가이드"""
    simplified_explanation: str
    key_concepts: List[str]
    step_by_step_guide: List[str]
    what_to_look_for: List[str]
    common_patterns: List[str]
    next_questions: List[str]
    analogies: List[str]
    warnings: List[str]


@dataclass
class BeginnerScenarioResult:
    """초보자 시나리오 처리 결과"""
    guidance: BeginnerGuidance
    confidence_level: float
    learning_path: List[str]
    simplified_insights: List[str]
    encouragement: str
    next_steps: List[str]
    resource_suggestions: List[str]


class BeginnerScenarioHandler:
    """
    초보자 시나리오 핸들러
    - 기술 용어 없는 쉬운 설명
    - 단계별 학습 가이드
    - 비유와 예시 중심 설명
    - 격려와 동기부여 포함
    """
    
    def __init__(self):
        """BeginnerScenarioHandler 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.interaction_history = []
        self.learning_progress = {}
        logger.info("BeginnerScenarioHandler initialized")
    
    async def handle_confused_data_scenario(
        self,
        data: Any,
        user_query: str,
        context: Dict[str, Any] = None
    ) -> BeginnerScenarioResult:
        """
        "이 데이터 파일이 뭘 말하는지 전혀 모르겠어요" 시나리오 처리
        
        Args:
            data: 분석할 데이터
            user_query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            초보자 친화적 설명 결과
        """
        logger.info("Handling confused data scenario for beginner")
        
        try:
            # 1. 데이터 기본 분석
            data_summary = await self._analyze_data_for_beginners(data)
            
            # 2. 초보자 친화적 설명 생성
            guidance = await self._generate_beginner_guidance(
                data_summary, user_query, context
            )
            
            # 3. 학습 경로 생성
            learning_path = await self._create_learning_path(data_summary, guidance)
            
            # 4. 격려 메시지 생성
            encouragement = await self._generate_encouragement(user_query)
            
            # 5. 다음 단계 제안
            next_steps = await self._suggest_next_steps(data_summary, guidance)
            
            # 6. 학습 자료 추천
            resources = self._recommend_learning_resources(data_summary)
            
            result = BeginnerScenarioResult(
                guidance=guidance,
                confidence_level=0.9,  # 초보자용은 높은 신뢰도로 제공
                learning_path=learning_path,
                simplified_insights=guidance.key_concepts,
                encouragement=encouragement,
                next_steps=next_steps,
                resource_suggestions=resources
            )
            
            # 7. 상호작용 이력 저장
            self._record_interaction(user_query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in beginner scenario handling: {e}")
            # 기본 친화적 응답 제공
            return self._create_fallback_response(data, user_query)
    
    async def _analyze_data_for_beginners(self, data: Any) -> Dict[str, Any]:
        """초보자를 위한 데이터 분석"""
        
        # 데이터 기본 정보 추출
        data_info = self._extract_basic_data_info(data)
        
        prompt = f"""
        다음 데이터를 완전 초보자가 이해할 수 있도록 분석하세요.
        기술 용어는 사용하지 말고, 일상 언어로 설명하세요.
        
        데이터 정보: {json.dumps(data_info, ensure_ascii=False)}
        
        초보자 관점에서 다음을 분석하세요:
        1. 이 데이터가 무엇인지 (일상 언어로)
        2. 어떤 정보를 담고 있는지
        3. 왜 이런 데이터를 모으는지
        4. 일반인에게 어떤 의미인지
        
        JSON 형식으로 응답하세요:
        {{
            "what_is_this": "이 데이터가 무엇인지 쉬운 설명",
            "main_story": "데이터가 말하는 주요 이야기",
            "why_collected": "왜 이런 데이터를 모으는지",
            "real_world_meaning": "실생활에서 어떤 의미인지",
            "interesting_parts": ["흥미로운 부분1", "흥미로운 부분2"],
            "simple_patterns": "발견할 수 있는 간단한 패턴들",
            "data_type_explanation": "데이터 종류에 대한 쉬운 설명"
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _generate_beginner_guidance(
        self,
        data_summary: Dict[str, Any],
        user_query: str,
        context: Dict[str, Any] = None
    ) -> BeginnerGuidance:
        """초보자 가이드 생성"""
        
        prompt = f"""
        완전 초보자를 위한 데이터 이해 가이드를 생성하세요.
        친근하고 격려적인 톤으로, 어려운 용어 없이 설명하세요.
        
        데이터 분석 결과: {json.dumps(data_summary, ensure_ascii=False)}
        사용자 질문: {user_query}
        
        다음을 포함한 초보자 가이드를 작성하세요:
        
        JSON 형식으로 응답하세요:
        {{
            "simplified_explanation": "5살 아이도 이해할 수 있는 설명",
            "key_concepts": ["핵심 개념1", "핵심 개념2", "핵심 개념3"],
            "step_by_step_guide": [
                "1단계: 첫 번째로 봐야 할 것",
                "2단계: 두 번째로 확인할 것",
                "3단계: 세 번째로 알아볼 것"
            ],
            "what_to_look_for": [
                "이런 것을 찾아보세요1",
                "이런 것을 찾아보세요2"
            ],
            "common_patterns": [
                "자주 나타나는 패턴1",
                "자주 나타나는 패턴2"
            ],
            "next_questions": [
                "다음에 물어볼 수 있는 질문1",
                "다음에 물어볼 수 있는 질문2"
            ],
            "analogies": [
                "일상생활 비유1",
                "일상생활 비유2"
            ],
            "warnings": [
                "주의할 점1",
                "흔한 오해2"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        guidance_data = self._parse_json_response(response)
        
        return BeginnerGuidance(
            simplified_explanation=guidance_data.get('simplified_explanation', ''),
            key_concepts=guidance_data.get('key_concepts', []),
            step_by_step_guide=guidance_data.get('step_by_step_guide', []),
            what_to_look_for=guidance_data.get('what_to_look_for', []),
            common_patterns=guidance_data.get('common_patterns', []),
            next_questions=guidance_data.get('next_questions', []),
            analogies=guidance_data.get('analogies', []),
            warnings=guidance_data.get('warnings', [])
        )
    
    async def _create_learning_path(
        self,
        data_summary: Dict[str, Any],
        guidance: BeginnerGuidance
    ) -> List[str]:
        """단계별 학습 경로 생성"""
        
        prompt = f"""
        초보자를 위한 단계별 학습 경로를 만드세요.
        각 단계는 이전 단계를 이해한 후에 진행할 수 있도록 구성하세요.
        
        데이터 정보: {json.dumps(data_summary, ensure_ascii=False)[:500]}
        핵심 개념들: {guidance.key_concepts}
        
        5-7단계의 학습 경로를 만들어주세요.
        각 단계는 구체적이고 실행 가능해야 합니다.
        
        JSON 형식으로 응답하세요:
        {{
            "learning_path": [
                "1단계: 데이터가 무엇인지 파악하기",
                "2단계: 기본 정보 읽어보기",
                "3단계: 패턴 찾아보기",
                "4단계: 질문 만들어보기",
                "5단계: 결론 도출해보기"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        path_data = self._parse_json_response(response)
        return path_data.get('learning_path', [])
    
    async def _generate_encouragement(self, user_query: str) -> str:
        """격려 메시지 생성"""
        
        prompt = f"""
        데이터를 처음 접하는 사용자에게 격려와 동기부여가 되는 메시지를 작성하세요.
        
        사용자 상황: {user_query}
        
        다음을 포함하세요:
        1. 궁금해하는 것 자체가 훌륭하다는 격려
        2. 데이터 분석은 누구나 할 수 있다는 확신
        3. 작은 발견도 의미있다는 동기부여
        4. 계속 질문하라는 권장
        
        따뜻하고 친근한 톤으로 2-3문장으로 작성하세요.
        """
        
        response = await self.llm_client.agenerate(prompt)
        return response.strip()
    
    async def _suggest_next_steps(
        self,
        data_summary: Dict[str, Any],
        guidance: BeginnerGuidance
    ) -> List[str]:
        """다음 단계 제안"""
        
        prompt = f"""
        초보자가 다음에 할 수 있는 구체적이고 쉬운 단계들을 제안하세요.
        
        데이터 정보: {json.dumps(data_summary, ensure_ascii=False)[:300]}
        가이드 정보: {guidance.simplified_explanation[:200]}
        
        3-5개의 다음 단계를 제안하세요. 각 단계는:
        1. 구체적이고 실행 가능할 것
        2. 너무 어렵지 않을 것
        3. 재미있고 흥미로울 것
        
        JSON 형식으로 응답하세요:
        {{
            "next_steps": [
                "첫 번째로 해볼 수 있는 것",
                "두 번째로 시도해볼 것",
                "세 번째로 탐색해볼 것"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        steps_data = self._parse_json_response(response)
        return steps_data.get('next_steps', [])
    
    def _recommend_learning_resources(self, data_summary: Dict[str, Any]) -> List[str]:
        """학습 자료 추천"""
        
        # 데이터 유형에 따른 기본 학습 자료
        base_resources = [
            "📊 '데이터 시각화 기초' - 차트 읽는 법 배우기",
            "📈 '숫자로 이야기하기' - 데이터 해석 기초",
            "🔍 '패턴 찾기 게임' - 재미있게 분석 연습하기",
            "💡 '일상 속 데이터' - 주변 데이터 찾아보기"
        ]
        
        # 데이터 특성에 맞는 추가 자료
        data_type = data_summary.get('data_type_explanation', '').lower()
        
        if 'table' in data_type or '표' in data_type:
            base_resources.append("📋 '표 읽기 마스터' - 표 데이터 이해하기")
        
        if 'time' in data_type or '시간' in data_type:
            base_resources.append("⏰ '시간의 흐름 보기' - 시계열 데이터 기초")
        
        if 'category' in data_type or '범주' in data_type:
            base_resources.append("🏷️ '분류하기 놀이' - 카테고리 데이터 이해")
        
        return base_resources[:5]  # 최대 5개만 추천
    
    def _extract_basic_data_info(self, data: Any) -> Dict[str, Any]:
        """데이터 기본 정보 추출"""
        
        if data is None:
            return {'type': 'none', 'description': 'No data'}
        
        data_info = {
            'type': type(data).__name__,
            'size': 'unknown'
        }
        
        try:
            # DataFrame인 경우
            if hasattr(data, 'shape'):
                data_info.update({
                    'rows': data.shape[0],
                    'columns': data.shape[1],
                    'column_names': list(data.columns) if hasattr(data, 'columns') else [],
                    'size': f"{data.shape[0]} 행 x {data.shape[1]} 열"
                })
            
            # 리스트나 배열인 경우
            elif hasattr(data, '__len__'):
                data_info.update({
                    'length': len(data),
                    'size': f"{len(data)} 개 항목"
                })
            
            # 딕셔너리인 경우
            elif isinstance(data, dict):
                data_info.update({
                    'keys': list(data.keys())[:10],  # 처음 10개 키만
                    'size': f"{len(data)} 개 키"
                })
            
            # 샘플 데이터
            if hasattr(data, 'head'):
                data_info['sample'] = data.head(3).to_dict() if hasattr(data.head(3), 'to_dict') else str(data.head(3))
            elif isinstance(data, (list, tuple)) and len(data) > 0:
                data_info['sample'] = data[:3]
            
        except Exception as e:
            logger.warning(f"Error extracting data info: {e}")
            data_info['error'] = str(e)
        
        return data_info
    
    def _create_fallback_response(
        self,
        data: Any,
        user_query: str
    ) -> BeginnerScenarioResult:
        """기본 응답 생성 (오류 시)"""
        
        basic_guidance = BeginnerGuidance(
            simplified_explanation="데이터를 분석하는 중에 문제가 생겼어요. 하지만 걱정하지 마세요! 천천히 다시 시도해볼 수 있어요.",
            key_concepts=["데이터 보기", "패턴 찾기", "질문하기"],
            step_by_step_guide=[
                "1단계: 데이터를 천천히 살펴보기",
                "2단계: 궁금한 점 적어보기",
                "3단계: 간단한 질문부터 시작하기"
            ],
            what_to_look_for=["숫자들", "패턴들", "특이한 점들"],
            common_patterns=["높낮이", "증가감소", "반복"],
            next_questions=["이게 뭘 의미하죠?", "왜 이런 패턴이 나타나죠?"],
            analogies=["데이터는 책 읽기와 비슷해요"],
            warnings=["급하게 결론내리지 마세요"]
        )
        
        return BeginnerScenarioResult(
            guidance=basic_guidance,
            confidence_level=0.5,
            learning_path=["천천히 다시 시작하기"],
            simplified_insights=["데이터 분석은 연습이 필요해요"],
            encouragement="괜찮아요! 모든 전문가도 처음에는 초보자였답니다. 😊",
            next_steps=["다시 천천히 시도해보기"],
            resource_suggestions=["초보자 가이드 읽어보기"]
        )
    
    def _record_interaction(self, user_query: str, result: BeginnerScenarioResult):
        """상호작용 이력 기록"""
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': user_query[:100],
            'confidence': result.confidence_level,
            'concepts_introduced': len(result.guidance.key_concepts),
            'learning_steps': len(result.learning_path)
        }
        
        self.interaction_history.append(interaction)
        
        # 이력 크기 제한
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-50:]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
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
            return {}
    
    async def handle_learning_progression(
        self,
        user_id: str,
        current_query: str,
        previous_interactions: List[Dict] = None
    ) -> Dict[str, Any]:
        """학습 진행 상황 관리"""
        
        if user_id not in self.learning_progress:
            self.learning_progress[user_id] = {
                'level': 'absolute_beginner',
                'concepts_learned': [],
                'interaction_count': 0,
                'last_interaction': datetime.now().isoformat()
            }
        
        progress = self.learning_progress[user_id]
        progress['interaction_count'] += 1
        progress['last_interaction'] = datetime.now().isoformat()
        
        # 학습 레벨 조정
        if progress['interaction_count'] > 5:
            progress['level'] = 'learning_beginner'
        if progress['interaction_count'] > 15:
            progress['level'] = 'progressing_beginner'
        
        return progress
    
    def get_beginner_statistics(self) -> Dict[str, Any]:
        """초보자 시나리오 통계"""
        
        if not self.interaction_history:
            return {'message': 'No beginner interactions yet'}
        
        total_interactions = len(self.interaction_history)
        avg_confidence = sum(i['confidence'] for i in self.interaction_history) / total_interactions
        
        return {
            'total_beginner_interactions': total_interactions,
            'average_confidence': avg_confidence,
            'active_learners': len(self.learning_progress),
            'recent_interactions': self.interaction_history[-5:]
        }