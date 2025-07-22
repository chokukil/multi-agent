"""
Ambiguous Query Handler - 모호한 쿼리 처리

Requirement 15 구현:
- "뭔가 이상한데요? 평소랑 다른 것 같아요." 모호한 쿼리 시나리오 처리
- 컨텍스트 기반 의도 파악 및 명확화
- 단계적 질문을 통한 요구사항 구체화
- 탐색적 분석을 통한 이상 패턴 감지
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AmbiguityLevel(Enum):
    """모호함 수준"""
    SLIGHTLY_UNCLEAR = "slightly_unclear"
    MODERATELY_AMBIGUOUS = "moderately_ambiguous"
    HIGHLY_AMBIGUOUS = "highly_ambiguous"
    COMPLETELY_VAGUE = "completely_vague"


class ClarificationStrategy(Enum):
    """명확화 전략"""
    DIRECTED_QUESTIONING = "directed_questioning"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    PATTERN_DETECTION = "pattern_detection"
    CONTEXTUAL_INFERENCE = "contextual_inference"


@dataclass
class ClarificationQuestion:
    """명확화 질문"""
    question: str
    purpose: str
    question_type: str  # "open", "closed", "multiple_choice", "ranking"
    expected_info: str
    follow_up_strategy: str


@dataclass
class PatternAnomalyResult:
    """패턴 이상 감지 결과"""
    anomaly_type: str
    confidence_score: float
    description: str
    location: str
    severity: str
    potential_causes: List[str]
    recommended_investigation: List[str]


@dataclass
class AmbiguousQueryResult:
    """모호한 쿼리 처리 결과"""
    ambiguity_assessment: Dict[str, Any]
    clarification_questions: List[ClarificationQuestion]
    exploratory_findings: Dict[str, Any]
    pattern_anomalies: List[PatternAnomalyResult]
    possible_interpretations: List[str]
    suggested_next_steps: List[str]
    confidence_level: float


class AmbiguousQueryHandler:
    """
    모호한 쿼리 핸들러
    - 모호함 수준 평가 및 분류
    - 컨텍스트 기반 의도 추론
    - 전략적 명확화 질문 생성
    - 탐색적 데이터 분석 수행
    """
    
    def __init__(self):
        """AmbiguousQueryHandler 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.clarification_history = []
        self.pattern_memory = {}
        self.context_patterns = {}
        logger.info("AmbiguousQueryHandler initialized")
    
    async def handle_vague_concern_scenario(
        self,
        vague_query: str,
        data: Any,
        context: Dict[str, Any] = None,
        conversation_history: List[Dict] = None
    ) -> AmbiguousQueryResult:
        """
        "뭔가 이상한데요? 평소랑 다른 것 같아요." 시나리오 처리
        
        Args:
            vague_query: 모호한 사용자 쿼리
            data: 분석할 데이터
            context: 추가 컨텍스트
            conversation_history: 대화 이력
            
        Returns:
            모호한 쿼리 처리 결과
        """
        logger.info(f"Handling vague concern: {vague_query[:50]}...")
        
        try:
            # 1. 모호함 수준 평가
            ambiguity_assessment = await self._assess_ambiguity_level(
                vague_query, context, conversation_history
            )
            
            # 2. 컨텍스트 기반 의도 추론
            intent_inference = await self._infer_user_intent(
                vague_query, data, context, conversation_history
            )
            
            # 3. 전략적 명확화 질문 생성
            clarification_questions = await self._generate_clarification_questions(
                vague_query, intent_inference, data
            )
            
            # 4. 탐색적 데이터 분석
            exploratory_findings = await self._perform_exploratory_analysis(
                data, intent_inference
            )
            
            # 5. 이상 패턴 자동 감지
            pattern_anomalies = await self._detect_pattern_anomalies(
                data, context, conversation_history
            )
            
            # 6. 가능한 해석들 생성
            possible_interpretations = await self._generate_possible_interpretations(
                vague_query, exploratory_findings, pattern_anomalies
            )
            
            # 7. 다음 단계 제안
            suggested_next_steps = await self._suggest_next_steps(
                ambiguity_assessment, clarification_questions, exploratory_findings
            )
            
            # 8. 신뢰도 평가
            confidence_level = self._calculate_confidence(
                ambiguity_assessment, exploratory_findings, pattern_anomalies
            )
            
            result = AmbiguousQueryResult(
                ambiguity_assessment=ambiguity_assessment,
                clarification_questions=clarification_questions,
                exploratory_findings=exploratory_findings,
                pattern_anomalies=pattern_anomalies,
                possible_interpretations=possible_interpretations,
                suggested_next_steps=suggested_next_steps,
                confidence_level=confidence_level
            )
            
            # 9. 상호작용 이력 저장
            self._record_clarification_attempt(vague_query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ambiguous query handling: {e}")
            return self._create_fallback_response(vague_query, data)
    
    async def _assess_ambiguity_level(
        self,
        query: str,
        context: Dict[str, Any],
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """모호함 수준 평가"""
        
        prompt = f"""
        다음 사용자 쿼리의 모호함 수준을 평가하고 분석하세요.
        
        사용자 쿼리: "{query}"
        컨텍스트: {json.dumps(context, ensure_ascii=False) if context else "없음"}
        대화 이력: {json.dumps(conversation_history[-3:] if conversation_history else [], ensure_ascii=False)}
        
        다음 관점에서 모호함을 분석하세요:
        1. 의도의 명확성 (무엇을 원하는가?)
        2. 범위의 구체성 (어떤 데이터, 어떤 부분?)
        3. 기대 결과의 명시성 (어떤 답을 원하는가?)
        4. 기술적 수준 표현 (전문용어 vs 일반용어)
        5. 감정적 표현 (걱정, 궁금함, 불안 등)
        
        JSON 형식으로 응답하세요:
        {{
            "ambiguity_level": "slightly_unclear|moderately_ambiguous|highly_ambiguous|completely_vague",
            "ambiguity_sources": [
                {{
                    "source": "의도 불명확",
                    "severity": "low|medium|high",
                    "description": "구체적 설명"
                }}
            ],
            "contextual_clues": [
                "컨텍스트에서 얻을 수 있는 단서1",
                "컨텍스트에서 얻을 수 있는 단서2"
            ],
            "emotional_indicators": {{
                "concern_level": "low|medium|high",
                "urgency": "low|medium|high",
                "confidence_in_observation": "low|medium|high"
            }},
            "interpretation_difficulty": {{
                "technical_complexity": "low|medium|high",
                "scope_breadth": "narrow|medium|broad",
                "specificity_needed": "low|medium|high"
            }}
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _infer_user_intent(
        self,
        query: str,
        data: Any,
        context: Dict[str, Any],
        conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """사용자 의도 추론"""
        
        data_summary = self._summarize_data_for_intent(data)
        
        prompt = f"""
        모호한 사용자 쿼리에서 의도를 추론하세요.
        
        사용자 쿼리: "{query}"
        데이터 요약: {json.dumps(data_summary, ensure_ascii=False)}
        컨텍스트: {json.dumps(context, ensure_ascii=False) if context else "없음"}
        
        다음 가능한 의도들을 평가하고 우선순위를 매기세요:
        1. 이상 패턴 감지 요청
        2. 변화/트렌드 분석 요청
        3. 성능 저하 확인 요청
        4. 비교 분석 요청 (과거 vs 현재)
        5. 예외 상황 식별 요청
        6. 품질/정확성 검증 요청
        7. 일반적 데이터 탐색 요청
        
        JSON 형식으로 응답하세요:
        {{
            "primary_intent": {{
                "type": "anomaly_detection|trend_analysis|performance_check|comparison|exception_identification|quality_verification|general_exploration",
                "confidence": 0.0-1.0,
                "reasoning": "추론 근거"
            }},
            "secondary_intents": [
                {{
                    "type": "의도 유형",
                    "confidence": 0.0-1.0,
                    "reasoning": "추론 근거"
                }}
            ],
            "likely_concerns": [
                "우려사항1",
                "우려사항2",
                "우려사항3"
            ],
            "information_gaps": [
                "부족한 정보1",
                "부족한 정보2"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _generate_clarification_questions(
        self,
        query: str,
        intent_inference: Dict[str, Any],
        data: Any
    ) -> List[ClarificationQuestion]:
        """명확화 질문 생성"""
        
        primary_intent = intent_inference.get('primary_intent', {})
        intent_type = primary_intent.get('type', 'general_exploration')
        
        prompt = f"""
        모호한 쿼리를 명확화하기 위한 전략적 질문들을 생성하세요.
        
        원본 쿼리: "{query}"
        추정 의도: {intent_type}
        추정 신뢰도: {primary_intent.get('confidence', 0.5)}
        
        다음 원칙에 따라 질문을 생성하세요:
        1. 점진적 구체화 (넓은 것부터 좁은 것으로)
        2. 비위협적 톤 (사용자가 부담스럽지 않게)
        3. 실행 가능성 (사용자가 답할 수 있는 것)
        4. 진단적 가치 (답변이 분석에 도움되는 것)
        
        3-5개의 질문을 순서대로 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "clarification_questions": [
                {{
                    "question": "구체적인 질문 내용",
                    "purpose": "이 질문의 목적",
                    "question_type": "open|closed|multiple_choice|ranking",
                    "expected_info": "얻고자 하는 정보",
                    "follow_up_strategy": "답변에 따른 후속 전략"
                }}
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        questions_data = self._parse_json_response(response)
        
        questions = []
        for q_data in questions_data.get('clarification_questions', []):
            question = ClarificationQuestion(
                question=q_data.get('question', ''),
                purpose=q_data.get('purpose', ''),
                question_type=q_data.get('question_type', 'open'),
                expected_info=q_data.get('expected_info', ''),
                follow_up_strategy=q_data.get('follow_up_strategy', '')
            )
            questions.append(question)
        
        return questions
    
    async def _perform_exploratory_analysis(
        self,
        data: Any,
        intent_inference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """탐색적 데이터 분석"""
        
        if data is None:
            return {'message': 'No data available for exploratory analysis'}
        
        # 기본 통계 정보 추출
        basic_stats = self._calculate_basic_statistics(data)
        
        # 패턴 검색
        patterns = await self._search_for_patterns(data, intent_inference)
        
        # 이상값 감지
        outliers = self._detect_outliers(data)
        
        # 시계열 분석 (가능한 경우)
        time_series_analysis = await self._analyze_time_patterns(data)
        
        return {
            'basic_statistics': basic_stats,
            'pattern_analysis': patterns,
            'outlier_detection': outliers,
            'time_series_insights': time_series_analysis,
            'data_quality_assessment': self._assess_data_quality(data)
        }
    
    async def _detect_pattern_anomalies(
        self,
        data: Any,
        context: Dict[str, Any],
        conversation_history: List[Dict]
    ) -> List[PatternAnomalyResult]:
        """패턴 이상 자동 감지"""
        
        if data is None:
            return []
        
        anomalies = []
        
        # 통계적 이상값 감지
        statistical_anomalies = self._detect_statistical_anomalies(data)
        for anomaly in statistical_anomalies:
            anomalies.append(PatternAnomalyResult(
                anomaly_type="statistical_outlier",
                confidence_score=anomaly.get('confidence', 0.7),
                description=anomaly.get('description', ''),
                location=anomaly.get('location', ''),
                severity=anomaly.get('severity', 'medium'),
                potential_causes=anomaly.get('causes', []),
                recommended_investigation=anomaly.get('investigation', [])
            ))
        
        # 분포 이상 감지
        distribution_anomalies = await self._detect_distribution_anomalies(data)
        for anomaly in distribution_anomalies:
            anomalies.append(PatternAnomalyResult(
                anomaly_type="distribution_anomaly",
                confidence_score=anomaly.get('confidence', 0.6),
                description=anomaly.get('description', ''),
                location=anomaly.get('location', ''),
                severity=anomaly.get('severity', 'medium'),
                potential_causes=anomaly.get('causes', []),
                recommended_investigation=anomaly.get('investigation', [])
            ))
        
        # 시간적 패턴 이상
        if self._has_time_component(data):
            temporal_anomalies = await self._detect_temporal_anomalies(data)
            for anomaly in temporal_anomalies:
                anomalies.append(PatternAnomalyResult(
                    anomaly_type="temporal_anomaly",
                    confidence_score=anomaly.get('confidence', 0.8),
                    description=anomaly.get('description', ''),
                    location=anomaly.get('location', ''),
                    severity=anomaly.get('severity', 'medium'),
                    potential_causes=anomaly.get('causes', []),
                    recommended_investigation=anomaly.get('investigation', [])
                ))
        
        return anomalies
    
    async def _generate_possible_interpretations(
        self,
        query: str,
        exploratory_findings: Dict[str, Any],
        pattern_anomalies: List[PatternAnomalyResult]
    ) -> List[str]:
        """가능한 해석들 생성"""
        
        findings_summary = {
            'basic_stats': exploratory_findings.get('basic_statistics', {}),
            'patterns_found': len(exploratory_findings.get('pattern_analysis', {}).get('patterns', [])),
            'anomalies_count': len(pattern_anomalies),
            'data_quality': exploratory_findings.get('data_quality_assessment', {})
        }
        
        prompt = f"""
        모호한 사용자 쿼리에 대한 가능한 해석들을 생성하세요.
        
        원본 쿼리: "{query}"
        탐색적 분석 결과: {json.dumps(findings_summary, ensure_ascii=False)}
        발견된 이상 패턴 수: {len(pattern_anomalies)}
        
        다음을 고려하여 3-5개의 가능한 해석을 제시하세요:
        1. 사용자가 느낀 "이상함"의 가능한 원인
        2. 데이터에서 발견된 실제 패턴과의 연관성
        3. 일반적으로 사용자들이 관심을 갖는 변화들
        4. 업무/도메인별 일반적 우려사항
        
        JSON 형식으로 응답하세요:
        {{
            "interpretations": [
                "해석1: 구체적이고 실용적인 해석",
                "해석2: 다른 관점에서의 해석",
                "해석3: 기술적 관점에서의 해석",
                "해석4: 비즈니스 관점에서의 해석"
            ]
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        interpretations_data = self._parse_json_response(response)
        return interpretations_data.get('interpretations', [])
    
    async def _suggest_next_steps(
        self,
        ambiguity_assessment: Dict[str, Any],
        clarification_questions: List[ClarificationQuestion],
        exploratory_findings: Dict[str, Any]
    ) -> List[str]:
        """다음 단계 제안"""
        
        ambiguity_level = ambiguity_assessment.get('ambiguity_level', 'moderately_ambiguous')
        
        if ambiguity_level in ['highly_ambiguous', 'completely_vague']:
            return [
                "먼저 몇 가지 질문에 답해주시면 더 정확한 분석을 도와드릴 수 있어요",
                "어떤 부분이 평소와 다르게 느껴지시는지 조금 더 구체적으로 설명해주세요",
                "데이터의 특정 영역이나 시간대에 관심이 있으신지 알려주세요"
            ]
        elif ambiguity_level == 'moderately_ambiguous':
            return [
                "탐색적 분석을 시작하여 이상 패턴을 찾아보겠습니다",
                "발견된 패턴들을 하나씩 검토해보시겠어요?",
                "특정 영역에 집중해서 더 자세히 분석해보시겠어요?"
            ]
        else:  # slightly_unclear
            return [
                "바로 상세 분석을 시작할 수 있습니다",
                "발견된 이상 패턴들을 우선순위별로 검토해보시겠어요?",
                "원인 분석 및 해결방안을 함께 찾아보시겠어요?"
            ]
    
    def _calculate_confidence(
        self,
        ambiguity_assessment: Dict[str, Any],
        exploratory_findings: Dict[str, Any],
        pattern_anomalies: List[PatternAnomalyResult]
    ) -> float:
        """신뢰도 계산"""
        
        # 모호함 수준에 따른 기본 신뢰도
        ambiguity_level = ambiguity_assessment.get('ambiguity_level', 'moderately_ambiguous')
        base_confidence = {
            'slightly_unclear': 0.8,
            'moderately_ambiguous': 0.6,
            'highly_ambiguous': 0.4,
            'completely_vague': 0.2
        }.get(ambiguity_level, 0.5)
        
        # 탐색적 분석 결과 신뢰도 보정
        if exploratory_findings.get('basic_statistics'):
            base_confidence += 0.1
        
        # 이상 패턴 발견 시 신뢰도 증가
        if pattern_anomalies:
            avg_anomaly_confidence = sum(a.confidence_score for a in pattern_anomalies) / len(pattern_anomalies)
            base_confidence = (base_confidence + avg_anomaly_confidence) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _summarize_data_for_intent(self, data: Any) -> Dict[str, Any]:
        """의도 추론을 위한 데이터 요약"""
        
        if data is None:
            return {'type': 'none', 'description': 'No data available'}
        
        summary = {
            'type': type(data).__name__,
            'has_time_component': self._has_time_component(data),
            'has_numeric_data': self._has_numeric_data(data),
            'estimated_complexity': 'unknown'
        }
        
        try:
            if hasattr(data, 'shape'):
                summary.update({
                    'rows': data.shape[0],
                    'columns': data.shape[1],
                    'estimated_complexity': 'high' if data.shape[0] * data.shape[1] > 10000 else 'medium'
                })
            elif hasattr(data, '__len__'):
                length = len(data)
                summary.update({
                    'length': length,
                    'estimated_complexity': 'high' if length > 1000 else 'medium'
                })
        except Exception as e:
            logger.warning(f"Error summarizing data: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def _calculate_basic_statistics(self, data: Any) -> Dict[str, Any]:
        """기본 통계 계산"""
        
        if data is None:
            return {}
        
        stats = {}
        
        try:
            if hasattr(data, 'describe'):
                # DataFrame의 경우
                desc = data.describe()
                stats['descriptive_stats'] = desc.to_dict() if hasattr(desc, 'to_dict') else str(desc)
            elif hasattr(data, '__len__') and len(data) > 0:
                # 리스트나 배열의 경우
                if all(isinstance(x, (int, float)) for x in data):
                    import statistics
                    stats['mean'] = statistics.mean(data)
                    stats['median'] = statistics.median(data)
                    stats['std'] = statistics.stdev(data) if len(data) > 1 else 0
        except Exception as e:
            logger.warning(f"Error calculating basic statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    async def _search_for_patterns(self, data: Any, intent_inference: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 검색"""
        
        patterns = {
            'trends': [],
            'cycles': [],
            'clusters': [],
            'correlations': []
        }
        
        # 간단한 트렌드 감지
        if self._has_numeric_data(data) and hasattr(data, 'shape'):
            try:
                # 첫 번째 숫자 컬럼의 트렌드 검사
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    first_col = data[numeric_cols[0]]
                    if len(first_col) > 5:
                        # 간단한 선형 트렌드 체크
                        values = first_col.dropna().values
                        if len(values) > 2:
                            slope = (values[-1] - values[0]) / (len(values) - 1)
                            if abs(slope) > 0.1:  # 임계값
                                direction = "증가" if slope > 0 else "감소"
                                patterns['trends'].append(f"{numeric_cols[0]}: {direction} 트렌드 감지")
            except Exception as e:
                logger.warning(f"Error in pattern search: {e}")
        
        return patterns
    
    def _detect_outliers(self, data: Any) -> Dict[str, Any]:
        """이상값 감지"""
        
        outliers = {'count': 0, 'locations': [], 'severity': 'none'}
        
        try:
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=['number'])
                for col in numeric_data.columns:
                    col_data = numeric_data[col].dropna()
                    if len(col_data) > 4:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                        outlier_count = outlier_mask.sum()
                        
                        if outlier_count > 0:
                            outliers['count'] += outlier_count
                            outliers['locations'].append(f"{col}: {outlier_count}개")
                
                if outliers['count'] > 0:
                    outliers['severity'] = 'high' if outliers['count'] > len(data) * 0.1 else 'medium'
        
        except Exception as e:
            logger.warning(f"Error in outlier detection: {e}")
            outliers['error'] = str(e)
        
        return outliers
    
    async def _analyze_time_patterns(self, data: Any) -> Dict[str, Any]:
        """시계열 패턴 분석"""
        
        if not self._has_time_component(data):
            return {'message': 'No time component detected'}
        
        time_analysis = {
            'has_seasonality': False,
            'has_trend': False,
            'irregular_patterns': [],
            'period_analysis': {}
        }
        
        # 시간 관련 패턴 간단 분석
        try:
            # 기본적인 시계열 패턴 감지 로직
            time_analysis['analysis_attempted'] = True
        except Exception as e:
            logger.warning(f"Error in time series analysis: {e}")
            time_analysis['error'] = str(e)
        
        return time_analysis
    
    def _assess_data_quality(self, data: Any) -> Dict[str, Any]:
        """데이터 품질 평가"""
        
        quality = {
            'completeness': 1.0,
            'consistency': 1.0,
            'validity': 1.0,
            'issues': []
        }
        
        try:
            if hasattr(data, 'isnull'):
                # 결측값 확인
                missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                quality['completeness'] = 1.0 - missing_ratio
                
                if missing_ratio > 0.1:
                    quality['issues'].append(f"결측값 비율 높음: {missing_ratio:.1%}")
            
            if hasattr(data, 'duplicated'):
                # 중복값 확인
                duplicate_ratio = data.duplicated().sum() / len(data)
                if duplicate_ratio > 0.05:
                    quality['issues'].append(f"중복 행 비율: {duplicate_ratio:.1%}")
        
        except Exception as e:
            logger.warning(f"Error in data quality assessment: {e}")
            quality['error'] = str(e)
        
        return quality
    
    def _has_time_component(self, data: Any) -> bool:
        """시간 구성요소 확인"""
        if hasattr(data, 'columns'):
            time_keywords = ['time', 'date', 'timestamp', '시간', '날짜']
            return any(any(keyword in str(col).lower() for keyword in time_keywords) for col in data.columns)
        return False
    
    def _has_numeric_data(self, data: Any) -> bool:
        """숫자 데이터 확인"""
        try:
            if hasattr(data, 'select_dtypes'):
                return len(data.select_dtypes(include=['number']).columns) > 0
            elif hasattr(data, '__len__') and len(data) > 0:
                return any(isinstance(x, (int, float)) for x in data)
        except Exception:
            pass
        return False
    
    def _detect_statistical_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """통계적 이상값 감지"""
        anomalies = []
        
        try:
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=['number'])
                for col in numeric_data.columns:
                    col_data = numeric_data[col].dropna()
                    if len(col_data) > 10:
                        # Z-score 기반 이상값
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        z_scores = abs((col_data - mean_val) / std_val)
                        extreme_outliers = (z_scores > 3).sum()
                        
                        if extreme_outliers > 0:
                            anomalies.append({
                                'description': f'{col}에서 {extreme_outliers}개의 극단적 이상값 발견',
                                'location': col,
                                'severity': 'high' if extreme_outliers > len(col_data) * 0.05 else 'medium',
                                'confidence': 0.9,
                                'causes': ['측정 오류', '데이터 입력 오류', '실제 예외 상황'],
                                'investigation': [f'{col} 컬럼의 극값들 개별 검토', '데이터 수집 과정 확인']
                            })
        except Exception as e:
            logger.warning(f"Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_distribution_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """분포 이상 감지"""
        anomalies = []
        
        # 간단한 분포 이상 감지 로직
        # 실제로는 더 정교한 통계적 테스트가 필요함
        
        return anomalies
    
    async def _detect_temporal_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """시간적 이상 감지"""
        anomalies = []
        
        # 시계열 이상 패턴 감지 로직
        # 계절성 변화, 트렌드 변화점 등
        
        return anomalies
    
    def _create_fallback_response(self, query: str, data: Any) -> AmbiguousQueryResult:
        """기본 응답 생성 (오류 시)"""
        
        fallback_question = ClarificationQuestion(
            question="어떤 부분이 평소와 다르게 느껴지시나요?",
            purpose="사용자의 우려사항 파악",
            question_type="open",
            expected_info="구체적인 관찰 내용",
            follow_up_strategy="답변에 따라 맞춤형 분석 수행"
        )
        
        return AmbiguousQueryResult(
            ambiguity_assessment={'ambiguity_level': 'highly_ambiguous'},
            clarification_questions=[fallback_question],
            exploratory_findings={'message': 'Analysis failed, manual clarification needed'},
            pattern_anomalies=[],
            possible_interpretations=["데이터 변화에 대한 일반적 우려"],
            suggested_next_steps=["더 구체적인 설명 요청"],
            confidence_level=0.3
        )
    
    def _record_clarification_attempt(self, query: str, result: AmbiguousQueryResult):
        """명확화 시도 기록"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'ambiguity_level': result.ambiguity_assessment.get('ambiguity_level'),
            'questions_generated': len(result.clarification_questions),
            'anomalies_found': len(result.pattern_anomalies),
            'confidence': result.confidence_level
        }
        
        self.clarification_history.append(record)
        
        # 이력 크기 제한
        if len(self.clarification_history) > 100:
            self.clarification_history = self.clarification_history[-100:]
    
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
    
    def get_clarification_statistics(self) -> Dict[str, Any]:
        """명확화 통계"""
        
        if not self.clarification_history:
            return {'message': 'No clarification attempts recorded'}
        
        total_attempts = len(self.clarification_history)
        
        # 모호함 수준 분포
        ambiguity_distribution = {}
        for record in self.clarification_history:
            level = record['ambiguity_level']
            ambiguity_distribution[level] = ambiguity_distribution.get(level, 0) + 1
        
        # 평균 신뢰도
        avg_confidence = sum(r['confidence'] for r in self.clarification_history) / total_attempts
        
        return {
            'total_clarification_attempts': total_attempts,
            'ambiguity_level_distribution': ambiguity_distribution,
            'average_confidence': avg_confidence,
            'average_questions_per_attempt': sum(r['questions_generated'] for r in self.clarification_history) / total_attempts,
            'recent_attempts': self.clarification_history[-5:]
        }