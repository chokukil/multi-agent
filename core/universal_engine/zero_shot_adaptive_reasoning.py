"""
Zero-Shot Adaptive Reasoning - 템플릿 없는 순수 추론 시스템

Requirement 14 구현:
- 템플릿 없는 순수 추론 시스템
- 문제 공간 정의 및 추론 전략 수립
- 단계별 추론 실행 및 결과 통합
- 가정 명시 및 불확실성 평가
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum

from ..llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ProblemDimension(Enum):
    """문제 차원 유형"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LOGICAL = "logical"
    EMPIRICAL = "empirical"
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"


class ReasoningStrategy(Enum):
    """추론 전략 유형"""
    DEDUCTIVE = "deductive"        # 연역적 추론
    INDUCTIVE = "inductive"        # 귀납적 추론
    ABDUCTIVE = "abductive"        # 가추적 추론
    ANALOGICAL = "analogical"      # 유추적 추론
    CAUSAL = "causal"             # 인과적 추론
    SYSTEMS = "systems"           # 시스템적 추론


class UncertaintyType(Enum):
    """불확실성 유형"""
    EPISTEMIC = "epistemic"        # 지식의 불완전성
    ALEATORIC = "aleatoric"        # 본질적 무작위성
    ONTOLOGICAL = "ontological"    # 존재론적 불확실성
    METHODOLOGICAL = "methodological"  # 방법론적 불확실성


@dataclass
class ProblemSpace:
    """문제 공간 정의"""
    domain: str
    dimensions: List[ProblemDimension]
    complexity_level: float  # 0.0-1.0
    known_constraints: List[str]
    unknown_factors: List[str]
    available_information: Dict[str, Any]
    information_gaps: List[str]
    success_criteria: List[str]


@dataclass
class ReasoningStep:
    """추론 단계"""
    step_id: str
    strategy_used: ReasoningStrategy
    premise: str
    reasoning_process: str
    intermediate_conclusion: str
    confidence: float
    assumptions_made: List[str]
    evidence_referenced: List[str]
    alternatives_considered: List[str]


@dataclass
class UncertaintyAssessment:
    """불확실성 평가"""
    uncertainty_type: UncertaintyType
    uncertainty_level: float  # 0.0-1.0
    sources: List[str]
    impact_on_conclusion: str
    mitigation_strategies: List[str]


@dataclass
class AdaptiveReasoningResult:
    """적응적 추론 결과"""
    problem_space: ProblemSpace
    reasoning_strategy: ReasoningStrategy
    reasoning_steps: List[ReasoningStep]
    final_conclusion: str
    confidence_level: float
    key_assumptions: List[str]
    uncertainty_assessments: List[UncertaintyAssessment]
    alternative_perspectives: List[str]
    recommendations: List[str]
    meta_reasoning: Dict[str, Any]


class ZeroShotAdaptiveReasoning:
    """
    Zero-Shot Adaptive Reasoning 엔진
    - 템플릿 없는 순수 추론
    - 동적 문제 공간 정의
    - 적응적 추론 전략 선택
    - 불확실성 평가 및 관리
    """
    
    def __init__(self):
        """ZeroShotAdaptiveReasoning 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.reasoning_history: List[Dict] = []
        self.learned_patterns: Dict[str, Any] = {}
        logger.info("ZeroShotAdaptiveReasoning initialized")
    
    async def perform_adaptive_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        available_data: Any = None
    ) -> AdaptiveReasoningResult:
        """
        적응적 추론 수행
        
        Args:
            query: 추론 대상 쿼리
            context: 컨텍스트 정보
            available_data: 사용 가능한 데이터
            
        Returns:
            적응적 추론 결과
        """
        logger.info(f"Starting adaptive reasoning for: {query[:50]}...")
        start_time = datetime.now()
        
        try:
            # 1. 문제 공간 정의
            problem_space = await self._define_problem_space(query, context, available_data)
            
            # 2. 적응적 추론 전략 선택
            reasoning_strategy = await self._select_reasoning_strategy(problem_space, query)
            
            # 3. 단계별 추론 실행
            reasoning_steps = await self._execute_reasoning_steps(
                problem_space, reasoning_strategy, query, context
            )
            
            # 4. 최종 결론 도출
            final_conclusion = await self._derive_conclusion(reasoning_steps, problem_space)
            
            # 5. 불확실성 평가
            uncertainty_assessments = await self._assess_uncertainties(
                problem_space, reasoning_steps, final_conclusion
            )
            
            # 6. 대안적 관점 탐색
            alternative_perspectives = await self._explore_alternative_perspectives(
                problem_space, reasoning_steps, final_conclusion
            )
            
            # 7. 추천사항 생성
            recommendations = await self._generate_recommendations(
                problem_space, reasoning_steps, uncertainty_assessments
            )
            
            # 8. 메타 추론 수행
            meta_reasoning = await self._perform_meta_reasoning(
                problem_space, reasoning_strategy, reasoning_steps, uncertainty_assessments
            )
            
            # 9. 신뢰도 계산
            confidence_level = self._calculate_overall_confidence(
                reasoning_steps, uncertainty_assessments, meta_reasoning
            )
            
            result = AdaptiveReasoningResult(
                problem_space=problem_space,
                reasoning_strategy=reasoning_strategy,
                reasoning_steps=reasoning_steps,
                final_conclusion=final_conclusion,
                confidence_level=confidence_level,
                key_assumptions=self._extract_key_assumptions(reasoning_steps),
                uncertainty_assessments=uncertainty_assessments,
                alternative_perspectives=alternative_perspectives,
                recommendations=recommendations,
                meta_reasoning=meta_reasoning
            )
            
            # 10. 학습 및 이력 저장
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_and_learn(query, result, execution_time)
            
            logger.info(f"Adaptive reasoning completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in adaptive reasoning: {e}")
            raise
    
    async def _define_problem_space(
        self,
        query: str,
        context: Dict[str, Any],
        available_data: Any
    ) -> ProblemSpace:
        """문제 공간 정의"""
        
        problem_analysis_prompt = f"""
        다음 쿼리와 컨텍스트를 분석하여 문제 공간을 정의하세요.
        
        쿼리: {query}
        컨텍스트: {json.dumps(context, ensure_ascii=False)[:800]}
        데이터 정보: {str(available_data)[:300] if available_data is not None else "데이터 없음"}
        
        문제 공간을 다음 관점에서 분석하세요:
        1. 문제의 도메인과 범위
        2. 문제의 차원들 (분석적, 창의적, 논리적, 경험적, 개념적, 실용적)
        3. 복잡도 수준 평가
        4. 알려진 제약사항들
        5. 불명확한 요소들
        6. 사용 가능한 정보
        7. 정보 격차
        8. 성공 기준
        
        JSON 형식으로 응답하세요:
        {{
            "domain": "문제 도메인 식별",
            "dimensions": ["analytical", "creative", "logical", "empirical", "conceptual", "practical"],
            "complexity_level": 0.0-1.0,
            "known_constraints": ["제약사항1", "제약사항2"],
            "unknown_factors": ["불명확 요소1", "불명확 요소2"],
            "available_information": {{
                "key1": "정보1",
                "key2": "정보2"
            }},
            "information_gaps": ["정보 격차1", "정보 격차2"],
            "success_criteria": ["성공 기준1", "성공 기준2"]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(problem_analysis_prompt)
            problem_data = self._parse_json_response(response)
            
            # 문제 차원 enum 변환
            dimensions = []
            for dim_str in problem_data.get('dimensions', []):
                try:
                    dimensions.append(ProblemDimension(dim_str))
                except ValueError:
                    pass  # 유효하지 않은 차원은 무시
            
            return ProblemSpace(
                domain=problem_data.get('domain', 'unknown'),
                dimensions=dimensions,
                complexity_level=problem_data.get('complexity_level', 0.5),
                known_constraints=problem_data.get('known_constraints', []),
                unknown_factors=problem_data.get('unknown_factors', []),
                available_information=problem_data.get('available_information', {}),
                information_gaps=problem_data.get('information_gaps', []),
                success_criteria=problem_data.get('success_criteria', [])
            )
            
        except Exception as e:
            logger.error(f"Error defining problem space: {e}")
            
            # 기본 문제 공간 반환
            return ProblemSpace(
                domain="general",
                dimensions=[ProblemDimension.ANALYTICAL],
                complexity_level=0.5,
                known_constraints=[],
                unknown_factors=["문제 공간 정의 실패"],
                available_information={},
                information_gaps=["전체 정보 격차"],
                success_criteria=["기본 성공 기준"]
            )
    
    async def _select_reasoning_strategy(
        self,
        problem_space: ProblemSpace,
        query: str
    ) -> ReasoningStrategy:
        """적응적 추론 전략 선택"""
        
        strategy_selection_prompt = f"""
        다음 문제 공간과 쿼리에 가장 적합한 추론 전략을 선택하세요.
        
        문제 도메인: {problem_space.domain}
        문제 차원들: {[dim.value for dim in problem_space.dimensions]}
        복잡도: {problem_space.complexity_level}
        쿼리: {query}
        
        사용 가능한 추론 전략들:
        1. deductive: 일반 원리에서 구체적 결론으로 (연역적)
        2. inductive: 구체적 관찰에서 일반 원리로 (귀납적)
        3. abductive: 최적 설명을 찾는 추론 (가추적)
        4. analogical: 유사성 기반 추론 (유추적)
        5. causal: 원인-결과 관계 추론 (인과적)
        6. systems: 시스템 전체 관점 추론 (시스템적)
        
        최적의 전략과 그 이유를 설명하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "selected_strategy": "deductive|inductive|abductive|analogical|causal|systems",
            "reasoning": "선택 이유에 대한 상세한 설명",
            "alternative_strategies": ["대안 전략1", "대안 전략2"],
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(strategy_selection_prompt)
            strategy_data = self._parse_json_response(response)
            
            strategy_str = strategy_data.get('selected_strategy', 'analytical')
            try:
                return ReasoningStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Invalid strategy: {strategy_str}, using default")
                return ReasoningStrategy.DEDUCTIVE
                
        except Exception as e:
            logger.error(f"Error selecting reasoning strategy: {e}")
            return ReasoningStrategy.DEDUCTIVE
    
    async def _execute_reasoning_steps(
        self,
        problem_space: ProblemSpace,
        strategy: ReasoningStrategy,
        query: str,
        context: Dict[str, Any]
    ) -> List[ReasoningStep]:
        """단계별 추론 실행"""
        
        reasoning_prompt = f"""
        다음 문제에 대해 {strategy.value} 추론 전략을 사용하여 단계별 추론을 수행하세요.
        
        문제 공간:
        - 도메인: {problem_space.domain}
        - 복잡도: {problem_space.complexity_level}
        - 제약사항: {problem_space.known_constraints}
        - 성공 기준: {problem_space.success_criteria}
        
        쿼리: {query}
        컨텍스트: {json.dumps(context, ensure_ascii=False)[:500]}
        
        {strategy.value} 추론 전략에 따라 3-5단계의 체계적 추론을 수행하세요.
        각 단계에서:
        1. 명확한 전제 설정
        2. 추론 과정 상세 설명
        3. 중간 결론 도출
        4. 신뢰도 평가
        5. 가정사항 명시
        6. 사용된 증거 기록
        7. 고려된 대안들 나열
        
        JSON 형식으로 응답하세요:
        {{
            "reasoning_steps": [
                {{
                    "step_id": "step_1",
                    "premise": "이 단계의 전제",
                    "reasoning_process": "추론 과정의 상세한 설명",
                    "intermediate_conclusion": "이 단계의 중간 결론",
                    "confidence": 0.0-1.0,
                    "assumptions_made": ["가정1", "가정2"],
                    "evidence_referenced": ["증거1", "증거2"],
                    "alternatives_considered": ["대안1", "대안2"]
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(reasoning_prompt)
            steps_data = self._parse_json_response(response)
            
            reasoning_steps = []
            for step_data in steps_data.get('reasoning_steps', []):
                step = ReasoningStep(
                    step_id=step_data.get('step_id', f'step_{len(reasoning_steps)+1}'),
                    strategy_used=strategy,
                    premise=step_data.get('premise', ''),
                    reasoning_process=step_data.get('reasoning_process', ''),
                    intermediate_conclusion=step_data.get('intermediate_conclusion', ''),
                    confidence=step_data.get('confidence', 0.5),
                    assumptions_made=step_data.get('assumptions_made', []),
                    evidence_referenced=step_data.get('evidence_referenced', []),
                    alternatives_considered=step_data.get('alternatives_considered', [])
                )
                reasoning_steps.append(step)
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error executing reasoning steps: {e}")
            
            # 기본 추론 단계 반환
            return [ReasoningStep(
                step_id="fallback_step",
                strategy_used=strategy,
                premise="추론 실행 실패",
                reasoning_process="추론 단계 생성 중 오류 발생",
                intermediate_conclusion="결론 도출 불가",
                confidence=0.0,
                assumptions_made=[],
                evidence_referenced=[],
                alternatives_considered=[]
            )]
    
    async def _derive_conclusion(
        self,
        reasoning_steps: List[ReasoningStep],
        problem_space: ProblemSpace
    ) -> str:
        """최종 결론 도출"""
        
        conclusion_prompt = f"""
        다음 추론 단계들을 종합하여 최종 결론을 도출하세요.
        
        추론 단계들:
        {json.dumps([{
            "step_id": step.step_id,
            "premise": step.premise,
            "reasoning_process": step.reasoning_process,
            "intermediate_conclusion": step.intermediate_conclusion,
            "confidence": step.confidence
        } for step in reasoning_steps], ensure_ascii=False)}
        
        성공 기준: {problem_space.success_criteria}
        
        모든 추론 단계를 종합하여 논리적이고 일관된 최종 결론을 도출하세요.
        결론은 구체적이고 실행 가능해야 합니다.
        """
        
        try:
            response = await self.llm_client.agenerate(conclusion_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error deriving conclusion: {e}")
            return "결론 도출 중 오류 발생"
    
    async def _assess_uncertainties(
        self,
        problem_space: ProblemSpace,
        reasoning_steps: List[ReasoningStep],
        conclusion: str
    ) -> List[UncertaintyAssessment]:
        """불확실성 평가"""
        
        uncertainty_prompt = f"""
        다음 추론 과정과 결론에 대한 불확실성을 평가하세요.
        
        문제 공간의 불명확 요소들: {problem_space.unknown_factors}
        정보 격차: {problem_space.information_gaps}
        추론 단계들의 신뢰도: {[step.confidence for step in reasoning_steps]}
        최종 결론: {conclusion}
        
        다음 유형의 불확실성을 평가하세요:
        1. epistemic: 지식의 불완전성으로 인한 불확실성
        2. aleatoric: 본질적 무작위성으로 인한 불확실성
        3. ontological: 존재론적/개념적 불확실성
        4. methodological: 방법론적 한계로 인한 불확실성
        
        JSON 형식으로 응답하세요:
        {{
            "uncertainties": [
                {{
                    "uncertainty_type": "epistemic|aleatoric|ontological|methodological",
                    "uncertainty_level": 0.0-1.0,
                    "sources": ["불확실성 원인1", "불확실성 원인2"],
                    "impact_on_conclusion": "결론에 미치는 영향 설명",
                    "mitigation_strategies": ["완화 전략1", "완화 전략2"]
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(uncertainty_prompt)
            uncertainty_data = self._parse_json_response(response)
            
            assessments = []
            for uncertainty in uncertainty_data.get('uncertainties', []):
                try:
                    uncertainty_type = UncertaintyType(uncertainty.get('uncertainty_type', 'epistemic'))
                    assessment = UncertaintyAssessment(
                        uncertainty_type=uncertainty_type,
                        uncertainty_level=uncertainty.get('uncertainty_level', 0.5),
                        sources=uncertainty.get('sources', []),
                        impact_on_conclusion=uncertainty.get('impact_on_conclusion', ''),
                        mitigation_strategies=uncertainty.get('mitigation_strategies', [])
                    )
                    assessments.append(assessment)
                except ValueError:
                    continue  # 유효하지 않은 불확실성 유형은 무시
            
            return assessments
            
        except Exception as e:
            logger.error(f"Error assessing uncertainties: {e}")
            return [UncertaintyAssessment(
                uncertainty_type=UncertaintyType.EPISTEMIC,
                uncertainty_level=0.5,
                sources=["불확실성 평가 실패"],
                impact_on_conclusion="평가 불가",
                mitigation_strategies=[]
            )]
    
    async def _explore_alternative_perspectives(
        self,
        problem_space: ProblemSpace,
        reasoning_steps: List[ReasoningStep],
        conclusion: str
    ) -> List[str]:
        """대안적 관점 탐색"""
        
        alternatives_prompt = f"""
        다음 추론 과정과 결론에 대해 대안적 관점들을 탐색하세요.
        
        현재 결론: {conclusion}
        사용된 추론 전략: {reasoning_steps[0].strategy_used.value if reasoning_steps else 'unknown'}
        
        다음과 같은 대안적 관점들을 고려하세요:
        1. 다른 추론 전략을 사용했다면?
        2. 다른 가정을 했다면?
        3. 다른 증거를 우선시했다면?
        4. 완전히 다른 접근법을 취했다면?
        5. 반대 입장에서 본다면?
        
        각 대안적 관점에서 어떤 다른 결론이 가능한지 설명하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "alternative_perspectives": [
                "관점1: 다른 추론 전략 사용시 가능한 결론",
                "관점2: 다른 가정 하에서의 가능한 결론",
                "관점3: 반대 입장에서의 가능한 결론"
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(alternatives_prompt)
            alternatives_data = self._parse_json_response(response)
            return alternatives_data.get('alternative_perspectives', [])
        except Exception as e:
            logger.error(f"Error exploring alternatives: {e}")
            return ["대안적 관점 탐색 실패"]
    
    async def _generate_recommendations(
        self,
        problem_space: ProblemSpace,
        reasoning_steps: List[ReasoningStep],
        uncertainty_assessments: List[UncertaintyAssessment]
    ) -> List[str]:
        """추천사항 생성"""
        
        recommendations_prompt = f"""
        추론 결과를 바탕으로 실행 가능한 추천사항들을 생성하세요.
        
        문제 도메인: {problem_space.domain}
        성공 기준: {problem_space.success_criteria}
        주요 불확실성들: {[ua.uncertainty_type.value for ua in uncertainty_assessments]}
        
        다음 카테고리별로 추천사항을 제시하세요:
        1. 즉시 실행 가능한 조치
        2. 불확실성 완화 방안
        3. 추가 정보 수집 방향
        4. 장기적 개선 방안
        
        JSON 형식으로 응답하세요:
        {{
            "recommendations": [
                "즉시 실행: 구체적인 행동 방안",
                "불확실성 완화: 위험 관리 방안",
                "정보 수집: 추가 조사 방향",
                "장기 개선: 지속적 개선 방안"
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(recommendations_prompt)
            recommendations_data = self._parse_json_response(response)
            return recommendations_data.get('recommendations', [])
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["추천사항 생성 실패"]
    
    async def _perform_meta_reasoning(
        self,
        problem_space: ProblemSpace,
        strategy: ReasoningStrategy,
        reasoning_steps: List[ReasoningStep],
        uncertainty_assessments: List[UncertaintyAssessment]
    ) -> Dict[str, Any]:
        """메타 추론 수행"""
        
        # 추론 품질 평가
        reasoning_quality = self._evaluate_reasoning_quality(reasoning_steps)
        
        # 전략 적합성 평가
        strategy_effectiveness = self._evaluate_strategy_effectiveness(
            problem_space, strategy, reasoning_steps
        )
        
        # 불확실성 관리 평가
        uncertainty_management = self._evaluate_uncertainty_management(
            uncertainty_assessments, reasoning_steps
        )
        
        return {
            "reasoning_quality": reasoning_quality,
            "strategy_effectiveness": strategy_effectiveness,
            "uncertainty_management": uncertainty_management,
            "overall_assessment": (reasoning_quality + strategy_effectiveness + uncertainty_management) / 3,
            "improvement_suggestions": self._suggest_improvements(
                problem_space, strategy, reasoning_steps, uncertainty_assessments
            )
        }
    
    def _evaluate_reasoning_quality(self, reasoning_steps: List[ReasoningStep]) -> float:
        """추론 품질 평가"""
        if not reasoning_steps:
            return 0.0
        
        # 신뢰도 평균
        avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        # 완전성 평가 (각 단계가 필요한 요소들을 포함하는지)
        completeness_scores = []
        for step in reasoning_steps:
            score = 0.0
            if step.premise: score += 0.2
            if step.reasoning_process: score += 0.3
            if step.intermediate_conclusion: score += 0.2
            if step.assumptions_made: score += 0.15
            if step.evidence_referenced: score += 0.15
            completeness_scores.append(score)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        
        return (avg_confidence + avg_completeness) / 2
    
    def _evaluate_strategy_effectiveness(
        self,
        problem_space: ProblemSpace,
        strategy: ReasoningStrategy,
        reasoning_steps: List[ReasoningStep]
    ) -> float:
        """전략 효과성 평가"""
        
        # 문제 복잡도와 전략 적합성
        complexity = problem_space.complexity_level
        strategy_complexity_match = {
            ReasoningStrategy.DEDUCTIVE: 0.8 if complexity < 0.5 else 0.6,
            ReasoningStrategy.INDUCTIVE: 0.7 if complexity > 0.3 else 0.5,
            ReasoningStrategy.ABDUCTIVE: 0.9 if complexity > 0.6 else 0.6,
            ReasoningStrategy.ANALOGICAL: 0.8 if complexity > 0.4 else 0.5,
            ReasoningStrategy.CAUSAL: 0.9 if complexity > 0.5 else 0.7,
            ReasoningStrategy.SYSTEMS: 0.9 if complexity > 0.7 else 0.6
        }.get(strategy, 0.5)
        
        # 추론 단계의 일관성
        if len(reasoning_steps) > 1:
            consistency = sum(1 for i in range(len(reasoning_steps)-1) 
                           if reasoning_steps[i].confidence > 0.3 and reasoning_steps[i+1].confidence > 0.3) / (len(reasoning_steps) - 1)
        else:
            consistency = 1.0
        
        return (strategy_complexity_match + consistency) / 2
    
    def _evaluate_uncertainty_management(
        self,
        uncertainty_assessments: List[UncertaintyAssessment],
        reasoning_steps: List[ReasoningStep]
    ) -> float:
        """불확실성 관리 평가"""
        
        if not uncertainty_assessments:
            return 0.5  # 불확실성 평가가 없으면 중간 점수
        
        # 불확실성 인식 정도
        recognition_score = min(1.0, len(uncertainty_assessments) / 3)  # 3개 정도가 적정
        
        # 완화 전략 제시 정도
        mitigation_score = sum(1 for ua in uncertainty_assessments if ua.mitigation_strategies) / len(uncertainty_assessments)
        
        # 추론 단계에서 가정 명시 정도
        assumption_score = sum(1 for step in reasoning_steps if step.assumptions_made) / len(reasoning_steps) if reasoning_steps else 0
        
        return (recognition_score + mitigation_score + assumption_score) / 3
    
    def _suggest_improvements(
        self,
        problem_space: ProblemSpace,
        strategy: ReasoningStrategy,
        reasoning_steps: List[ReasoningStep],
        uncertainty_assessments: List[UncertaintyAssessment]
    ) -> List[str]:
        """개선 제안"""
        
        suggestions = []
        
        # 추론 품질 개선
        if reasoning_steps:
            avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
            if avg_confidence < 0.7:
                suggestions.append("추론 단계의 신뢰도 향상을 위해 더 강력한 증거 수집 필요")
        
        # 불확실성 관리 개선
        if len(uncertainty_assessments) < 2:
            suggestions.append("더 포괄적인 불확실성 분석 필요")
        
        # 전략 개선
        if problem_space.complexity_level > 0.7 and strategy not in [ReasoningStrategy.SYSTEMS, ReasoningStrategy.ABDUCTIVE]:
            suggestions.append("복잡한 문제에 대해 시스템적 또는 가추적 추론 전략 고려")
        
        return suggestions
    
    def _calculate_overall_confidence(
        self,
        reasoning_steps: List[ReasoningStep],
        uncertainty_assessments: List[UncertaintyAssessment],
        meta_reasoning: Dict[str, Any]
    ) -> float:
        """전체 신뢰도 계산"""
        
        if not reasoning_steps:
            return 0.0
        
        # 추론 단계 신뢰도 평균
        step_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        # 불확실성 페널티
        if uncertainty_assessments:
            avg_uncertainty = sum(ua.uncertainty_level for ua in uncertainty_assessments) / len(uncertainty_assessments)
            uncertainty_penalty = avg_uncertainty * 0.3
        else:
            uncertainty_penalty = 0.1  # 불확실성 평가 없으면 소폭 페널티
        
        # 메타 추론 품질 보너스
        meta_quality = meta_reasoning.get('overall_assessment', 0.5)
        meta_bonus = (meta_quality - 0.5) * 0.2
        
        final_confidence = max(0.0, min(1.0, step_confidence - uncertainty_penalty + meta_bonus))
        
        return final_confidence
    
    def _extract_key_assumptions(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """핵심 가정 추출"""
        all_assumptions = []
        for step in reasoning_steps:
            all_assumptions.extend(step.assumptions_made)
        
        # 중복 제거 및 중요도 순 정렬 (향후 LLM으로 개선 가능)
        unique_assumptions = list(set(all_assumptions))
        return unique_assumptions[:5]  # 상위 5개만
    
    async def _record_and_learn(
        self,
        query: str,
        result: AdaptiveReasoningResult,
        execution_time: float
    ):
        """학습 및 이력 기록"""
        
        # 실행 이력 기록
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "domain": result.problem_space.domain,
            "strategy": result.reasoning_strategy.value,
            "complexity": result.problem_space.complexity_level,
            "confidence": result.confidence_level,
            "execution_time": execution_time,
            "meta_quality": result.meta_reasoning.get('overall_assessment', 0.5)
        }
        
        self.reasoning_history.append(history_entry)
        
        # 이력 크기 제한
        if len(self.reasoning_history) > 100:
            self.reasoning_history = self.reasoning_history[-100:]
        
        # 패턴 학습 (간단한 휴리스틱)
        domain = result.problem_space.domain
        strategy = result.reasoning_strategy.value
        
        if domain not in self.learned_patterns:
            self.learned_patterns[domain] = {}
        
        if strategy not in self.learned_patterns[domain]:
            self.learned_patterns[domain][strategy] = {
                "usage_count": 0,
                "avg_confidence": 0.0,
                "avg_execution_time": 0.0
            }
        
        pattern = self.learned_patterns[domain][strategy]
        pattern["usage_count"] += 1
        pattern["avg_confidence"] = (pattern["avg_confidence"] * (pattern["usage_count"] - 1) + result.confidence_level) / pattern["usage_count"]
        pattern["avg_execution_time"] = (pattern["avg_execution_time"] * (pattern["usage_count"] - 1) + execution_time) / pattern["usage_count"]
    
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
    
    def get_reasoning_analytics(self) -> Dict[str, Any]:
        """추론 분석 정보 조회"""
        
        if not self.reasoning_history:
            return {"message": "No reasoning history available"}
        
        import statistics
        
        # 도메인별 통계
        domain_stats = {}
        for entry in self.reasoning_history:
            domain = entry["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = []
            domain_stats[domain].append(entry)
        
        domain_summary = {}
        for domain, entries in domain_stats.items():
            domain_summary[domain] = {
                "total_queries": len(entries),
                "avg_confidence": statistics.mean([e["confidence"] for e in entries]),
                "avg_execution_time": statistics.mean([e["execution_time"] for e in entries]),
                "preferred_strategies": self._get_strategy_preferences(entries)
            }
        
        return {
            "total_reasoning_sessions": len(self.reasoning_history),
            "average_confidence": statistics.mean([h["confidence"] for h in self.reasoning_history]),
            "average_execution_time": statistics.mean([h["execution_time"] for h in self.reasoning_history]),
            "domain_analytics": domain_summary,
            "learned_patterns": self.learned_patterns,
            "recent_performance": self.reasoning_history[-10:]
        }
    
    def _get_strategy_preferences(self, entries: List[Dict]) -> Dict[str, int]:
        """전략 선호도 계산"""
        from collections import Counter
        strategies = [entry["strategy"] for entry in entries]
        return dict(Counter(strategies).most_common(3))