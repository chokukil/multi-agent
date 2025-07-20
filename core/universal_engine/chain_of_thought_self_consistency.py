"""
Chain-of-Thought with Self-Consistency - 다중 추론 경로 및 일관성 검증

Requirement 13 구현:
- 다중 추론 경로 생성 및 실행
- 추론 경로 간 일관성 검증 로직
- 충돌 해결 및 최종 결론 도출
- 신뢰도 평가 및 불확실성 표시
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import Counter

from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ReasoningPathType(Enum):
    """추론 경로 유형"""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"
    EMPIRICAL = "empirical"


class ConsistencyLevel(Enum):
    """일관성 수준"""
    HIGH = "high"          # 90%+ 일치
    MEDIUM = "medium"      # 70-90% 일치
    LOW = "low"           # 50-70% 일치
    CONFLICTED = "conflicted"  # <50% 일치


@dataclass
class ReasoningPath:
    """개별 추론 경로"""
    path_id: str
    path_type: ReasoningPathType
    reasoning_steps: List[str]
    conclusion: str
    confidence: float
    key_assumptions: List[str]
    evidence_used: List[str]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyAnalysis:
    """일관성 분석 결과"""
    consistency_level: ConsistencyLevel
    agreement_percentage: float
    convergent_conclusions: List[str]
    conflicting_points: List[Dict[str, Any]]
    consensus_confidence: float
    uncertainty_factors: List[str]


@dataclass
class ConflictResolutionResult:
    """충돌 해결 결과"""
    resolved_conclusion: str
    resolution_method: str
    confidence_after_resolution: float
    supporting_paths: List[str]
    dismissed_paths: List[str]
    resolution_reasoning: str


@dataclass
class SelfConsistencyResult:
    """Self-Consistency 최종 결과"""
    final_conclusion: str
    overall_confidence: float
    reasoning_paths: List[ReasoningPath]
    consistency_analysis: ConsistencyAnalysis
    conflict_resolution: Optional[ConflictResolutionResult]
    uncertainty_assessment: Dict[str, Any]
    meta_analysis: Dict[str, Any]


class ChainOfThoughtSelfConsistency:
    """
    Chain-of-Thought with Self-Consistency 엔진
    - 다중 추론 경로 생성 및 실행
    - 추론 경로 간 일관성 검증
    - 충돌 해결 및 신뢰도 평가
    """
    
    def __init__(self, num_paths: int = 5):
        """
        ChainOfThoughtSelfConsistency 초기화
        
        Args:
            num_paths: 생성할 추론 경로 수 (기본값: 5)
        """
        self.llm_client = LLMFactory.create_llm()
        self.num_paths = num_paths
        self.execution_history: List[Dict] = []
        logger.info(f"ChainOfThoughtSelfConsistency initialized with {num_paths} paths")
    
    async def perform_multi_path_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        data: Any = None
    ) -> SelfConsistencyResult:
        """
        다중 경로 추론 실행
        
        Args:
            query: 사용자 쿼리
            context: 컨텍스트 정보
            data: 분석 대상 데이터
            
        Returns:
            Self-Consistency 분석 결과
        """
        logger.info(f"Starting multi-path reasoning for query: {query[:50]}...")
        start_time = datetime.now()
        
        try:
            # 1. 다중 추론 경로 생성 및 실행
            reasoning_paths = await self._generate_multiple_reasoning_paths(
                query, context, data
            )
            
            # 2. 일관성 분석 수행
            consistency_analysis = await self._analyze_consistency(reasoning_paths)
            
            # 3. 충돌이 있는 경우 해결
            conflict_resolution = None
            if consistency_analysis.consistency_level in [ConsistencyLevel.LOW, ConsistencyLevel.CONFLICTED]:
                conflict_resolution = await self._resolve_conflicts(
                    reasoning_paths, consistency_analysis
                )
            
            # 4. 최종 결론 도출
            final_conclusion = await self._derive_final_conclusion(
                reasoning_paths, consistency_analysis, conflict_resolution
            )
            
            # 5. 불확실성 평가
            uncertainty_assessment = await self._assess_uncertainty(
                reasoning_paths, consistency_analysis
            )
            
            # 6. 메타 분석
            meta_analysis = self._perform_meta_analysis(
                reasoning_paths, consistency_analysis, conflict_resolution
            )
            
            # 7. 전체 신뢰도 계산
            overall_confidence = self._calculate_overall_confidence(
                reasoning_paths, consistency_analysis, conflict_resolution
            )
            
            result = SelfConsistencyResult(
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence,
                reasoning_paths=reasoning_paths,
                consistency_analysis=consistency_analysis,
                conflict_resolution=conflict_resolution,
                uncertainty_assessment=uncertainty_assessment,
                meta_analysis=meta_analysis
            )
            
            # 8. 실행 이력 저장
            execution_time = (datetime.now() - start_time).total_seconds()
            self._record_execution_history(query, result, execution_time)
            
            logger.info(f"Multi-path reasoning completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-path reasoning: {e}")
            raise
    
    async def _generate_multiple_reasoning_paths(
        self,
        query: str,
        context: Dict[str, Any],
        data: Any
    ) -> List[ReasoningPath]:
        """다중 추론 경로 생성"""
        
        reasoning_paths = []
        path_types = list(ReasoningPathType)
        
        # 각 추론 경로 유형별로 생성
        for i in range(self.num_paths):
            path_type = path_types[i % len(path_types)]
            
            path = await self._generate_single_reasoning_path(
                path_id=f"path_{i+1}",
                path_type=path_type,
                query=query,
                context=context,
                data=data
            )
            
            reasoning_paths.append(path)
        
        return reasoning_paths
    
    async def _generate_single_reasoning_path(
        self,
        path_id: str,
        path_type: ReasoningPathType,
        query: str,
        context: Dict[str, Any],
        data: Any
    ) -> ReasoningPath:
        """개별 추론 경로 생성"""
        
        start_time = datetime.now()
        
        # 추론 경로 유형별 프롬프트 생성
        reasoning_prompt = self._create_path_specific_prompt(
            path_type, query, context, data
        )
        
        try:
            response = await self.llm_client.agenerate(reasoning_prompt)
            parsed_response = self._parse_reasoning_response(response)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ReasoningPath(
                path_id=path_id,
                path_type=path_type,
                reasoning_steps=parsed_response.get('reasoning_steps', []),
                conclusion=parsed_response.get('conclusion', ''),
                confidence=parsed_response.get('confidence', 0.5),
                key_assumptions=parsed_response.get('key_assumptions', []),
                evidence_used=parsed_response.get('evidence_used', []),
                execution_time=execution_time,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': response[:500]  # 처음 500자만 저장
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating reasoning path {path_id}: {e}")
            
            # 오류 시 기본 경로 반환
            return ReasoningPath(
                path_id=path_id,
                path_type=path_type,
                reasoning_steps=["추론 경로 생성 중 오류 발생"],
                conclusion="결론 도출 실패",
                confidence=0.0,
                key_assumptions=[],
                evidence_used=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )
    
    def _create_path_specific_prompt(
        self,
        path_type: ReasoningPathType,
        query: str,
        context: Dict[str, Any],
        data: Any
    ) -> str:
        """추론 경로 유형별 특화 프롬프트 생성"""
        
        base_context = f"""
        사용자 쿼리: {query}
        컨텍스트: {json.dumps(context, ensure_ascii=False)[:800]}
        데이터 정보: {str(data)[:300] if data is not None else "데이터 없음"}
        """
        
        path_instructions = {
            ReasoningPathType.ANALYTICAL: """
            **분석적 추론 접근법**을 사용하세요:
            - 체계적이고 논리적인 단계별 분석
            - 데이터와 증거에 기반한 객관적 추론
            - 정량적 분석과 통계적 접근법 우선
            - 각 단계의 논리적 연결성 중시
            """,
            
            ReasoningPathType.INTUITIVE: """
            **직관적 추론 접근법**을 사용하세요:
            - 패턴 인식과 전체적 관점에서의 추론
            - 경험적 지식과 직관적 판단 활용
            - 큰 그림과 맥락적 이해 중시
            - 창의적 연결과 통찰 추구
            """,
            
            ReasoningPathType.SYSTEMATIC: """
            **체계적 추론 접근법**을 사용하세요:
            - 구조화된 프레임워크와 방법론 적용
            - 단계별 검증과 확인 과정 포함
            - 표준화된 분석 절차 준수
            - 재현 가능한 추론 과정 구성
            """,
            
            ReasoningPathType.CREATIVE: """
            **창의적 추론 접근법**을 사용하세요:
            - 혁신적이고 비전통적인 관점 탐색
            - 다양한 가능성과 대안적 해석 고려
            - 상상력과 추상적 사고 활용
            - 기존 틀을 벗어난 새로운 접근법 시도
            """,
            
            ReasoningPathType.EMPIRICAL: """
            **경험적 추론 접근법**을 사용하세요:
            - 실증적 증거와 관찰 가능한 데이터 중심
            - 실험적 검증과 테스트 가능한 가설 수립
            - 과거 경험과 사례 연구 활용
            - 측정 가능한 결과와 실용적 검증 추구
            """
        }
        
        instruction = path_instructions.get(path_type, path_instructions[ReasoningPathType.ANALYTICAL])
        
        return f"""
        {base_context}
        
        {instruction}
        
        이 접근법을 사용하여 사용자 쿼리에 대해 상세한 추론을 수행하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "reasoning_steps": [
                "1단계: 첫 번째 추론 단계 설명",
                "2단계: 두 번째 추론 단계 설명",
                "3단계: 세 번째 추론 단계 설명",
                "결론: 최종 추론 결과"
            ],
            "conclusion": "최종 결론을 명확하고 구체적으로 서술",
            "confidence": 0.0-1.0,
            "key_assumptions": ["가정1", "가정2", "가정3"],
            "evidence_used": ["증거1", "증거2", "증거3"]
        }}
        """
    
    async def _analyze_consistency(
        self,
        reasoning_paths: List[ReasoningPath]
    ) -> ConsistencyAnalysis:
        """추론 경로들 간의 일관성 분석"""
        
        # 결론들 수집
        conclusions = [path.conclusion for path in reasoning_paths if path.conclusion]
        
        if not conclusions:
            return ConsistencyAnalysis(
                consistency_level=ConsistencyLevel.CONFLICTED,
                agreement_percentage=0.0,
                convergent_conclusions=[],
                conflicting_points=[],
                consensus_confidence=0.0,
                uncertainty_factors=["추론 경로에서 유효한 결론을 찾을 수 없음"]
            )
        
        # LLM을 사용한 일관성 분석
        consistency_prompt = f"""
        다음 추론 경로들의 결론들을 분석하여 일관성을 평가하세요.
        
        추론 경로별 결론들:
        {json.dumps([{"path_id": path.path_id, "conclusion": path.conclusion, "confidence": path.confidence} for path in reasoning_paths], ensure_ascii=False)}
        
        다음을 분석하세요:
        1. 결론들 간의 일치 정도
        2. 공통적으로 나타나는 핵심 포인트들
        3. 상충하는 부분들과 그 원인
        4. 전체적인 합의 가능성
        
        JSON 형식으로 응답하세요:
        {{
            "agreement_percentage": 0.0-100.0,
            "convergent_conclusions": ["공통 결론1", "공통 결론2"],
            "conflicting_points": [
                {{
                    "issue": "충돌 이슈 설명",
                    "conflicting_paths": ["path_1", "path_2"],
                    "severity": "low|medium|high"
                }}
            ],
            "consensus_confidence": 0.0-1.0,
            "uncertainty_factors": ["불확실성 요인1", "불확실성 요인2"]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(consistency_prompt)
            analysis_data = self._parse_json_response(response)
            
            agreement_percentage = analysis_data.get('agreement_percentage', 0.0)
            
            # 일관성 수준 결정
            if agreement_percentage >= 90:
                consistency_level = ConsistencyLevel.HIGH
            elif agreement_percentage >= 70:
                consistency_level = ConsistencyLevel.MEDIUM
            elif agreement_percentage >= 50:
                consistency_level = ConsistencyLevel.LOW
            else:
                consistency_level = ConsistencyLevel.CONFLICTED
            
            return ConsistencyAnalysis(
                consistency_level=consistency_level,
                agreement_percentage=agreement_percentage,
                convergent_conclusions=analysis_data.get('convergent_conclusions', []),
                conflicting_points=analysis_data.get('conflicting_points', []),
                consensus_confidence=analysis_data.get('consensus_confidence', 0.0),
                uncertainty_factors=analysis_data.get('uncertainty_factors', [])
            )
            
        except Exception as e:
            logger.error(f"Error in consistency analysis: {e}")
            
            # 단순 일치율 계산으로 fallback
            conclusion_counter = Counter(conclusions)
            most_common = conclusion_counter.most_common(1)
            agreement_percentage = (most_common[0][1] / len(conclusions)) * 100 if most_common else 0
            
            if agreement_percentage >= 70:
                consistency_level = ConsistencyLevel.MEDIUM
            elif agreement_percentage >= 50:
                consistency_level = ConsistencyLevel.LOW
            else:
                consistency_level = ConsistencyLevel.CONFLICTED
            
            return ConsistencyAnalysis(
                consistency_level=consistency_level,
                agreement_percentage=agreement_percentage,
                convergent_conclusions=[most_common[0][0]] if most_common else [],
                conflicting_points=[],
                consensus_confidence=agreement_percentage / 100,
                uncertainty_factors=["일관성 분석 중 오류 발생"]
            )
    
    async def _resolve_conflicts(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis
    ) -> ConflictResolutionResult:
        """충돌하는 추론 경로들 간의 충돌 해결"""
        
        conflict_resolution_prompt = f"""
        다음 추론 경로들 간에 충돌이 발생했습니다. 최적의 해결 방법을 제시하세요.
        
        추론 경로들:
        {json.dumps([{
            "path_id": path.path_id,
            "path_type": path.path_type.value,
            "conclusion": path.conclusion,
            "confidence": path.confidence,
            "key_assumptions": path.key_assumptions,
            "evidence_used": path.evidence_used
        } for path in reasoning_paths], ensure_ascii=False)}
        
        일관성 분석 결과:
        {json.dumps({
            "agreement_percentage": consistency_analysis.agreement_percentage,
            "conflicting_points": consistency_analysis.conflicting_points,
            "uncertainty_factors": consistency_analysis.uncertainty_factors
        }, ensure_ascii=False)}
        
        다음 해결 방법들을 고려하여 최적의 결론을 도출하세요:
        1. 신뢰도 기반 가중 평균
        2. 증거 품질 기반 선택
        3. 합의 가능한 공통분모 추출
        4. 대안적 통합 해석
        
        JSON 형식으로 응답하세요:
        {{
            "resolved_conclusion": "통합된 최종 결론",
            "resolution_method": "사용된 해결 방법 설명",
            "confidence_after_resolution": 0.0-1.0,
            "supporting_paths": ["지지하는 경로 ID들"],
            "dismissed_paths": ["제외된 경로 ID들"],
            "resolution_reasoning": "해결 과정의 상세한 설명"
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(conflict_resolution_prompt)
            resolution_data = self._parse_json_response(response)
            
            return ConflictResolutionResult(
                resolved_conclusion=resolution_data.get('resolved_conclusion', ''),
                resolution_method=resolution_data.get('resolution_method', ''),
                confidence_after_resolution=resolution_data.get('confidence_after_resolution', 0.5),
                supporting_paths=resolution_data.get('supporting_paths', []),
                dismissed_paths=resolution_data.get('dismissed_paths', []),
                resolution_reasoning=resolution_data.get('resolution_reasoning', '')
            )
            
        except Exception as e:
            logger.error(f"Error in conflict resolution: {e}")
            
            # 가장 높은 신뢰도의 경로 선택으로 fallback
            best_path = max(reasoning_paths, key=lambda p: p.confidence)
            
            return ConflictResolutionResult(
                resolved_conclusion=best_path.conclusion,
                resolution_method="최고 신뢰도 경로 선택 (fallback)",
                confidence_after_resolution=best_path.confidence,
                supporting_paths=[best_path.path_id],
                dismissed_paths=[p.path_id for p in reasoning_paths if p.path_id != best_path.path_id],
                resolution_reasoning="충돌 해결 중 오류 발생으로 최고 신뢰도 경로 선택"
            )
    
    async def _derive_final_conclusion(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis,
        conflict_resolution: Optional[ConflictResolutionResult]
    ) -> str:
        """최종 결론 도출"""
        
        if conflict_resolution:
            return conflict_resolution.resolved_conclusion
        
        # 일관성이 높은 경우 합의된 결론 사용
        if consistency_analysis.consistency_level in [ConsistencyLevel.HIGH, ConsistencyLevel.MEDIUM]:
            if consistency_analysis.convergent_conclusions:
                return consistency_analysis.convergent_conclusions[0]
        
        # fallback: 가장 높은 신뢰도 경로의 결론
        if reasoning_paths:
            best_path = max(reasoning_paths, key=lambda p: p.confidence)
            return best_path.conclusion
        
        return "결론 도출 실패"
    
    async def _assess_uncertainty(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis
    ) -> Dict[str, Any]:
        """불확실성 평가"""
        
        # 신뢰도 분산 계산
        confidences = [path.confidence for path in reasoning_paths if path.confidence > 0]
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # 결론 다양성 계산
        unique_conclusions = len(set(path.conclusion for path in reasoning_paths))
        conclusion_diversity = unique_conclusions / len(reasoning_paths) if reasoning_paths else 0
        
        return {
            "confidence_variance": confidence_std,
            "conclusion_diversity": conclusion_diversity,
            "agreement_level": consistency_analysis.consistency_level.value,
            "uncertainty_factors": consistency_analysis.uncertainty_factors,
            "path_count": len(reasoning_paths),
            "failed_paths": len([p for p in reasoning_paths if p.confidence == 0.0])
        }
    
    def _perform_meta_analysis(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis,
        conflict_resolution: Optional[ConflictResolutionResult]
    ) -> Dict[str, Any]:
        """메타 분석 수행"""
        
        # 경로별 성능 분석
        path_performance = {
            path.path_type.value: {
                "average_confidence": statistics.mean([p.confidence for p in reasoning_paths if p.path_type == path.path_type]),
                "execution_time": statistics.mean([p.execution_time for p in reasoning_paths if p.path_type == path.path_type]),
                "success_rate": len([p for p in reasoning_paths if p.path_type == path.path_type and p.confidence > 0.5]) / len([p for p in reasoning_paths if p.path_type == path.path_type])
            }
            for path in reasoning_paths
        }
        
        return {
            "total_reasoning_time": sum(path.execution_time for path in reasoning_paths),
            "path_performance": path_performance,
            "consistency_achieved": consistency_analysis.consistency_level.value,
            "conflicts_resolved": conflict_resolution is not None,
            "overall_quality": self._calculate_overall_quality(reasoning_paths, consistency_analysis)
        }
    
    def _calculate_overall_confidence(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis,
        conflict_resolution: Optional[ConflictResolutionResult]
    ) -> float:
        """전체 신뢰도 계산"""
        
        # 기본 신뢰도 (경로들의 평균)
        base_confidence = statistics.mean([p.confidence for p in reasoning_paths if p.confidence > 0]) if reasoning_paths else 0.0
        
        # 일관성 보너스/페널티
        consistency_modifier = {
            ConsistencyLevel.HIGH: 0.2,
            ConsistencyLevel.MEDIUM: 0.1,
            ConsistencyLevel.LOW: -0.1,
            ConsistencyLevel.CONFLICTED: -0.2
        }.get(consistency_analysis.consistency_level, 0.0)
        
        # 충돌 해결 보너스
        resolution_bonus = 0.1 if conflict_resolution and conflict_resolution.confidence_after_resolution > 0.7 else 0.0
        
        # 최종 신뢰도 계산
        final_confidence = min(1.0, max(0.0, base_confidence + consistency_modifier + resolution_bonus))
        
        return final_confidence
    
    def _calculate_overall_quality(
        self,
        reasoning_paths: List[ReasoningPath],
        consistency_analysis: ConsistencyAnalysis
    ) -> float:
        """전체 품질 점수 계산"""
        
        # 다양한 품질 지표들
        completeness = len([p for p in reasoning_paths if p.confidence > 0]) / len(reasoning_paths) if reasoning_paths else 0
        consistency_score = consistency_analysis.agreement_percentage / 100
        confidence_quality = statistics.mean([p.confidence for p in reasoning_paths if p.confidence > 0]) if reasoning_paths else 0
        
        # 가중 평균
        overall_quality = (completeness * 0.3 + consistency_score * 0.4 + confidence_quality * 0.3)
        
        return overall_quality
    
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
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, Any]:
        """추론 응답 파싱"""
        parsed = self._parse_json_response(response)
        
        # 기본값 설정
        if not parsed.get('reasoning_steps'):
            parsed['reasoning_steps'] = ["추론 단계 파싱 실패"]
        if not parsed.get('conclusion'):
            parsed['conclusion'] = "결론 파싱 실패"
        if 'confidence' not in parsed:
            parsed['confidence'] = 0.0
        if not parsed.get('key_assumptions'):
            parsed['key_assumptions'] = []
        if not parsed.get('evidence_used'):
            parsed['evidence_used'] = []
        
        return parsed
    
    def _record_execution_history(
        self,
        query: str,
        result: SelfConsistencyResult,
        execution_time: float
    ):
        """실행 이력 기록"""
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # 처음 100자만
            "num_paths": len(result.reasoning_paths),
            "consistency_level": result.consistency_analysis.consistency_level.value,
            "agreement_percentage": result.consistency_analysis.agreement_percentage,
            "overall_confidence": result.overall_confidence,
            "execution_time": execution_time,
            "conflicts_resolved": result.conflict_resolution is not None
        }
        
        self.execution_history.append(history_entry)
        
        # 이력 크기 제한
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """실행 통계 조회"""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        return {
            "total_executions": len(self.execution_history),
            "average_execution_time": statistics.mean([h["execution_time"] for h in self.execution_history]),
            "average_confidence": statistics.mean([h["overall_confidence"] for h in self.execution_history]),
            "consistency_distribution": Counter([h["consistency_level"] for h in self.execution_history]),
            "conflict_resolution_rate": len([h for h in self.execution_history if h["conflicts_resolved"]]) / len(self.execution_history),
            "recent_performance": self.execution_history[-10:]  # 최근 10개
        }