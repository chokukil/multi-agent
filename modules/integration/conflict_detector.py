"""
멀티 에이전트 결과 충돌 감지 시스템

이 모듈은 여러 A2A 에이전트 간의 상충되는 결과를 감지하고,
충돌 유형을 분류하며 해결 전략을 제안하는 시스템을 제공합니다.

주요 기능:
- 에이전트 간 상충 결과 식별 및 분류
- 충돌 유형별 우선순위 결정
- 데이터 불일치 및 모순 감지
- 충돌 해결 전략 및 권장사항 제공
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import difflib
from collections import defaultdict

from .agent_result_collector import AgentResult, CollectionSession
from .result_validator import QualityMetrics

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """충돌 유형"""
    DATA_CONTRADICTION = "data_contradiction"        # 데이터 모순
    STATISTICAL_DISCREPANCY = "statistical_discrepancy"  # 통계 불일치
    CONCLUSION_DIVERGENCE = "conclusion_divergence"  # 결론 분기
    FORMAT_INCONSISTENCY = "format_inconsistency"    # 형식 불일치
    QUALITY_VARIANCE = "quality_variance"            # 품질 편차
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"  # 시간적 불일치
    SCOPE_MISMATCH = "scope_mismatch"               # 범위 불일치

class ConflictSeverity(Enum):
    """충돌 심각도"""
    CRITICAL = "critical"    # 치명적 (즉시 해결 필요)
    HIGH = "high"           # 높음 (우선 해결)
    MEDIUM = "medium"       # 보통 (검토 필요)
    LOW = "low"            # 낮음 (참고)

class ResolutionStrategy(Enum):
    """해결 전략"""
    QUALITY_BASED = "quality_based"          # 품질 기준 선택
    CONSENSUS_BASED = "consensus_based"      # 합의 기준 선택
    MERGE_RESULTS = "merge_results"          # 결과 병합
    RERUN_ANALYSIS = "rerun_analysis"        # 재분석 실행
    MANUAL_REVIEW = "manual_review"          # 수동 검토
    EXCLUDE_OUTLIER = "exclude_outlier"      # 이상치 제외

@dataclass
class ConflictInstance:
    """충돌 인스턴스"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    
    # 관련 에이전트
    involved_agents: List[str]
    primary_agent: Optional[str] = None
    
    # 충돌 상세 정보
    description: str = ""
    conflicting_data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    
    # 해결 정보
    suggested_strategy: ResolutionStrategy = ResolutionStrategy.MANUAL_REVIEW
    resolution_confidence: float = 0.0
    alternative_strategies: List[ResolutionStrategy] = field(default_factory=list)
    
    # 메타데이터
    detected_at: datetime = field(default_factory=datetime.now)
    detection_method: str = ""
    
    @property
    def agent_count(self) -> int:
        return len(self.involved_agents)

@dataclass
class ConflictAnalysis:
    """충돌 분석 결과"""
    session_id: str
    total_conflicts: int
    conflicts_by_type: Dict[ConflictType, int]
    conflicts_by_severity: Dict[ConflictSeverity, int]
    
    # 상세 충돌 목록
    conflicts: List[ConflictInstance] = field(default_factory=list)
    
    # 전체 분석
    overall_consistency_score: float = 0.0
    reliability_impact: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    
    # 통계
    agent_conflict_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)
    most_conflicted_agents: List[str] = field(default_factory=list)

class ConflictDetector:
    """멀티 에이전트 결과 충돌 감지기"""
    
    def __init__(self):
        # 감지 임계값
        self.thresholds = {
            'statistical_difference': 0.15,    # 15% 이상 차이
            'quality_variance': 0.3,           # 품질 점수 30% 이상 차이
            'text_similarity': 0.7,            # 텍스트 유사도 70% 미만
            'execution_time_variance': 5.0     # 실행 시간 5배 이상 차이
        }
        
        # 키워드 패턴
        self.conclusion_keywords = [
            'conclusion', '결론', 'summary', '요약', 'result', '결과',
            'finding', '발견', 'insight', '인사이트', 'recommendation', '권장'
        ]
        
        self.statistical_keywords = [
            'mean', 'average', '평균', 'median', '중앙값', 'std', '표준편차',
            'count', '개수', 'total', '총합', 'percentage', '비율'
        ]
        
        # 충돌 우선순위 (높을수록 우선)
        self.severity_priorities = {
            ConflictSeverity.CRITICAL: 4,
            ConflictSeverity.HIGH: 3,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 1
        }
    
    def detect_conflicts(self, 
                        session: CollectionSession,
                        quality_metrics: Dict[str, QualityMetrics] = None) -> ConflictAnalysis:
        """세션 내 모든 충돌 감지"""
        
        logger.info(f"🔍 충돌 감지 시작 - 세션 {session.session_id}, "
                   f"에이전트 {len(session.collected_results)}개")
        
        analysis = ConflictAnalysis(
            session_id=session.session_id,
            total_conflicts=0,
            conflicts_by_type=defaultdict(int),
            conflicts_by_severity=defaultdict(int)
        )
        
        if len(session.collected_results) < 2:
            logger.info("충돌 감지에 필요한 최소 에이전트 수(2개) 미만")
            return analysis
        
        try:
            results = list(session.collected_results.values())
            
            # 1. 데이터 모순 감지
            conflicts = self._detect_data_contradictions(results)
            analysis.conflicts.extend(conflicts)
            
            # 2. 통계 불일치 감지
            conflicts = self._detect_statistical_discrepancies(results)
            analysis.conflicts.extend(conflicts)
            
            # 3. 결론 분기 감지
            conflicts = self._detect_conclusion_divergence(results)
            analysis.conflicts.extend(conflicts)
            
            # 4. 형식 불일치 감지
            conflicts = self._detect_format_inconsistencies(results)
            analysis.conflicts.extend(conflicts)
            
            # 5. 품질 편차 감지
            if quality_metrics:
                conflicts = self._detect_quality_variance(results, quality_metrics)
                analysis.conflicts.extend(conflicts)
            
            # 6. 시간적 불일치 감지
            conflicts = self._detect_temporal_inconsistencies(results)
            analysis.conflicts.extend(conflicts)
            
            # 7. 범위 불일치 감지
            conflicts = self._detect_scope_mismatches(results)
            analysis.conflicts.extend(conflicts)
            
            # 분석 결과 집계
            self._aggregate_analysis(analysis)
            
            logger.info(f"✅ 충돌 감지 완료 - {analysis.total_conflicts}개 충돌 발견")
            
        except Exception as e:
            logger.error(f"❌ 충돌 감지 중 오류: {e}")
            analysis.recommended_actions.append(f"충돌 감지 과정에서 오류 발생: {str(e)}")
        
        return analysis
    
    def resolve_conflicts(self, 
                         analysis: ConflictAnalysis,
                         results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """충돌 해결 방안 제시"""
        
        logger.info(f"🔧 충돌 해결 방안 생성 - {analysis.total_conflicts}개 충돌")
        
        resolution_plan = {
            "summary": {
                "total_conflicts": analysis.total_conflicts,
                "resolvable_conflicts": 0,
                "manual_review_required": 0
            },
            "resolutions": [],
            "priority_order": [],
            "implementation_steps": []
        }
        
        try:
            # 심각도별 정렬
            sorted_conflicts = sorted(
                analysis.conflicts,
                key=lambda c: self.severity_priorities[c.severity],
                reverse=True
            )
            
            for conflict in sorted_conflicts:
                resolution = self._generate_resolution(conflict, results)
                resolution_plan["resolutions"].append(resolution)
                
                if resolution["strategy"] != ResolutionStrategy.MANUAL_REVIEW.value:
                    resolution_plan["summary"]["resolvable_conflicts"] += 1
                else:
                    resolution_plan["summary"]["manual_review_required"] += 1
                
                resolution_plan["priority_order"].append(conflict.conflict_id)
            
            # 구현 단계 생성
            resolution_plan["implementation_steps"] = self._generate_implementation_steps(
                resolution_plan["resolutions"]
            )
            
        except Exception as e:
            logger.error(f"❌ 충돌 해결 방안 생성 중 오류: {e}")
            resolution_plan["error"] = str(e)
        
        return resolution_plan
    
    def _detect_data_contradictions(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """데이터 모순 감지"""
        
        conflicts = []
        
        # 아티팩트 데이터 비교
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                
                # 동일한 타입의 아티팩트 비교
                artifacts1 = {art['type']: art for art in result1.artifacts if 'type' in art}
                artifacts2 = {art['type']: art for art in result2.artifacts if 'type' in art}
                
                common_types = set(artifacts1.keys()) & set(artifacts2.keys())
                
                for art_type in common_types:
                    contradiction = self._compare_artifacts(
                        artifacts1[art_type], artifacts2[art_type],
                        result1.agent_id, result2.agent_id
                    )
                    
                    if contradiction:
                        conflict_id = f"data_contradiction_{result1.agent_id}_{result2.agent_id}_{art_type}"
                        
                        conflict = ConflictInstance(
                            conflict_id=conflict_id,
                            conflict_type=ConflictType.DATA_CONTRADICTION,
                            severity=ConflictSeverity.HIGH,
                            involved_agents=[result1.agent_id, result2.agent_id],
                            description=f"{art_type} 데이터에서 모순 발견",
                            conflicting_data=contradiction,
                            evidence=[f"에이전트 {result1.agent_id}와 {result2.agent_id}의 {art_type} 데이터 불일치"],
                            suggested_strategy=ResolutionStrategy.QUALITY_BASED,
                            detection_method="artifact_comparison"
                        )
                        
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_statistical_discrepancies(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """통계 불일치 감지"""
        
        conflicts = []
        
        # 텍스트에서 숫자 패턴 추출
        number_pattern = r'\b\d+(?:\.\d+)?%?\b'
        
        for i, result1 in enumerate(results):
            numbers1 = re.findall(number_pattern, result1.processed_text)
            
            for j, result2 in enumerate(results[i+1:], i+1):
                numbers2 = re.findall(number_pattern, result2.processed_text)
                
                # 유사한 컨텍스트에서 발견된 숫자들 비교
                discrepancies = self._find_numerical_discrepancies(
                    numbers1, numbers2, result1.processed_text, result2.processed_text
                )
                
                if discrepancies:
                    conflict_id = f"statistical_discrepancy_{result1.agent_id}_{result2.agent_id}"
                    
                    conflict = ConflictInstance(
                        conflict_id=conflict_id,
                        conflict_type=ConflictType.STATISTICAL_DISCREPANCY,
                        severity=ConflictSeverity.MEDIUM,
                        involved_agents=[result1.agent_id, result2.agent_id],
                        description="통계 수치 불일치 발견",
                        conflicting_data={"discrepancies": discrepancies},
                        evidence=[f"에이전트 간 통계 수치 차이: {len(discrepancies)}개 항목"],
                        suggested_strategy=ResolutionStrategy.CONSENSUS_BASED,
                        detection_method="numerical_comparison"
                    )
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_conclusion_divergence(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """결론 분기 감지"""
        
        conflicts = []
        
        # 각 결과에서 결론 부분 추출
        conclusions = {}
        for result in results:
            conclusion = self._extract_conclusion(result.processed_text)
            if conclusion:
                conclusions[result.agent_id] = conclusion
        
        if len(conclusions) < 2:
            return conflicts
        
        # 결론 간 유사도 비교
        agent_ids = list(conclusions.keys())
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                
                similarity = self._calculate_text_similarity(
                    conclusions[agent1], conclusions[agent2]
                )
                
                if similarity < self.thresholds['text_similarity']:
                    conflict_id = f"conclusion_divergence_{agent1}_{agent2}"
                    
                    severity = ConflictSeverity.HIGH if similarity < 0.5 else ConflictSeverity.MEDIUM
                    
                    conflict = ConflictInstance(
                        conflict_id=conflict_id,
                        conflict_type=ConflictType.CONCLUSION_DIVERGENCE,
                        severity=severity,
                        involved_agents=[agent1, agent2],
                        description=f"결론 유사도 낮음 ({similarity:.2f})",
                        conflicting_data={
                            "similarity_score": similarity,
                            "conclusions": {agent1: conclusions[agent1], agent2: conclusions[agent2]}
                        },
                        evidence=[f"결론 유사도 {similarity:.1%} (임계값 {self.thresholds['text_similarity']:.1%})"],
                        suggested_strategy=ResolutionStrategy.MANUAL_REVIEW,
                        detection_method="text_similarity_analysis"
                    )
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_format_inconsistencies(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """형식 불일치 감지"""
        
        conflicts = []
        
        # 아티팩트 형식 분석
        format_patterns = {}
        for result in results:
            patterns = []
            for artifact in result.artifacts:
                art_type = artifact.get('type', 'unknown')
                patterns.append(art_type)
            
            format_patterns[result.agent_id] = set(patterns)
        
        if len(format_patterns) < 2:
            return conflicts
        
        # 형식 차이 감지
        agent_ids = list(format_patterns.keys())
        expected_formats = set()
        for patterns in format_patterns.values():
            expected_formats.update(patterns)
        
        for agent_id, patterns in format_patterns.items():
            missing_formats = expected_formats - patterns
            
            if missing_formats:
                conflict_id = f"format_inconsistency_{agent_id}"
                
                conflict = ConflictInstance(
                    conflict_id=conflict_id,
                    conflict_type=ConflictType.FORMAT_INCONSISTENCY,
                    severity=ConflictSeverity.LOW,
                    involved_agents=[agent_id],
                    description=f"누락된 아티팩트 형식: {missing_formats}",
                    conflicting_data={
                        "missing_formats": list(missing_formats),
                        "available_formats": list(patterns)
                    },
                    evidence=[f"예상 형식 {len(expected_formats)}개 중 {len(missing_formats)}개 누락"],
                    suggested_strategy=ResolutionStrategy.RERUN_ANALYSIS,
                    detection_method="format_pattern_analysis"
                )
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_quality_variance(self, 
                               results: List[AgentResult],
                               quality_metrics: Dict[str, QualityMetrics]) -> List[ConflictInstance]:
        """품질 편차 감지"""
        
        conflicts = []
        
        # 품질 점수 수집
        quality_scores = {}
        for result in results:
            if result.agent_id in quality_metrics:
                quality_scores[result.agent_id] = quality_metrics[result.agent_id].overall_score
        
        if len(quality_scores) < 2:
            return conflicts
        
        # 품질 편차 분석
        scores = list(quality_scores.values())
        avg_score = sum(scores) / len(scores)
        max_diff = max(scores) - min(scores)
        
        if max_diff > self.thresholds['quality_variance']:
            # 품질이 현저히 낮은 에이전트 식별
            low_quality_agents = [
                agent_id for agent_id, score in quality_scores.items()
                if score < avg_score - self.thresholds['quality_variance'] / 2
            ]
            
            if low_quality_agents:
                conflict_id = f"quality_variance_{'_'.join(low_quality_agents)}"
                
                conflict = ConflictInstance(
                    conflict_id=conflict_id,
                    conflict_type=ConflictType.QUALITY_VARIANCE,
                    severity=ConflictSeverity.MEDIUM,
                    involved_agents=low_quality_agents,
                    description=f"품질 편차 큼 (최대 차이: {max_diff:.2f})",
                    conflicting_data={
                        "quality_scores": quality_scores,
                        "average_score": avg_score,
                        "max_difference": max_diff
                    },
                    evidence=[f"품질 점수 편차 {max_diff:.1%} (임계값 {self.thresholds['quality_variance']:.1%})"],
                    suggested_strategy=ResolutionStrategy.EXCLUDE_OUTLIER,
                    detection_method="quality_variance_analysis"
                )
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_temporal_inconsistencies(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """시간적 불일치 감지"""
        
        conflicts = []
        
        # 실행 시간 분석
        execution_times = {result.agent_id: result.execution_duration for result in results}
        
        if len(execution_times) < 2:
            return conflicts
        
        times = list(execution_times.values())
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # 실행 시간 편차가 큰 경우
        if max_time > min_time * self.thresholds['execution_time_variance']:
            
            # 비정상적으로 빠르거나 느린 에이전트 식별
            outlier_agents = []
            for agent_id, time in execution_times.items():
                if time < avg_time / 3 or time > avg_time * 3:
                    outlier_agents.append(agent_id)
            
            if outlier_agents:
                conflict_id = f"temporal_inconsistency_{'_'.join(outlier_agents)}"
                
                conflict = ConflictInstance(
                    conflict_id=conflict_id,
                    conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                    severity=ConflictSeverity.LOW,
                    involved_agents=outlier_agents,
                    description=f"실행 시간 편차 큼 (비율: {max_time/min_time:.1f}배)",
                    conflicting_data={
                        "execution_times": execution_times,
                        "average_time": avg_time,
                        "time_ratio": max_time / min_time
                    },
                    evidence=[f"실행 시간 편차 {max_time/min_time:.1f}배 (임계값 {self.thresholds['execution_time_variance']:.1f}배)"],
                    suggested_strategy=ResolutionStrategy.MANUAL_REVIEW,
                    detection_method="execution_time_analysis"
                )
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_scope_mismatches(self, results: List[AgentResult]) -> List[ConflictInstance]:
        """범위 불일치 감지"""
        
        conflicts = []
        
        # 각 결과의 범위/깊이 분석
        scope_analysis = {}
        for result in results:
            analysis = {
                "text_length": len(result.processed_text),
                "artifact_count": len(result.artifacts),
                "artifact_types": set(art.get('type', 'unknown') for art in result.artifacts),
                "keyword_coverage": self._analyze_keyword_coverage(result.processed_text)
            }
            scope_analysis[result.agent_id] = analysis
        
        if len(scope_analysis) < 2:
            return conflicts
        
        # 범위 차이 감지
        text_lengths = [analysis["text_length"] for analysis in scope_analysis.values()]
        artifact_counts = [analysis["artifact_count"] for analysis in scope_analysis.values()]
        
        # 텍스트 길이 편차
        if max(text_lengths) > min(text_lengths) * 3:
            outlier_agents = []
            avg_length = sum(text_lengths) / len(text_lengths)
            
            for agent_id, analysis in scope_analysis.items():
                if analysis["text_length"] < avg_length / 2:
                    outlier_agents.append(agent_id)
            
            if outlier_agents:
                conflict_id = f"scope_mismatch_text_{'_'.join(outlier_agents)}"
                
                conflict = ConflictInstance(
                    conflict_id=conflict_id,
                    conflict_type=ConflictType.SCOPE_MISMATCH,
                    severity=ConflictSeverity.MEDIUM,
                    involved_agents=outlier_agents,
                    description="분석 범위 불일치 (텍스트 길이)",
                    conflicting_data={
                        "text_lengths": {aid: analysis["text_length"] 
                                       for aid, analysis in scope_analysis.items()},
                        "average_length": avg_length
                    },
                    evidence=[f"텍스트 길이 편차 {max(text_lengths)/min(text_lengths):.1f}배"],
                    suggested_strategy=ResolutionStrategy.RERUN_ANALYSIS,
                    detection_method="scope_analysis"
                )
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _compare_artifacts(self, 
                          artifact1: Dict[str, Any], 
                          artifact2: Dict[str, Any],
                          agent1_id: str,
                          agent2_id: str) -> Optional[Dict[str, Any]]:
        """아티팩트 비교 및 모순 감지"""
        
        contradictions = {}
        
        try:
            art_type = artifact1.get('type', '')
            
            if art_type == 'dataframe':
                # 데이터프레임 비교
                data1 = artifact1.get('data', [])
                data2 = artifact2.get('data', [])
                
                if len(data1) != len(data2):
                    contradictions['row_count'] = {
                        agent1_id: len(data1),
                        agent2_id: len(data2)
                    }
                
                cols1 = set(artifact1.get('columns', []))
                cols2 = set(artifact2.get('columns', []))
                
                if cols1 != cols2:
                    contradictions['columns'] = {
                        agent1_id: list(cols1),
                        agent2_id: list(cols2),
                        'missing_in_1': list(cols2 - cols1),
                        'missing_in_2': list(cols1 - cols2)
                    }
            
            elif art_type == 'plotly_chart':
                # 차트 비교
                layout1 = artifact1.get('layout', {})
                layout2 = artifact2.get('layout', {})
                
                title1 = layout1.get('title', {}).get('text', '')
                title2 = layout2.get('title', {}).get('text', '')
                
                if title1 and title2 and title1 != title2:
                    contradictions['chart_title'] = {
                        agent1_id: title1,
                        agent2_id: title2
                    }
            
        except Exception as e:
            logger.warning(f"아티팩트 비교 중 오류: {e}")
        
        return contradictions if contradictions else None
    
    def _find_numerical_discrepancies(self, 
                                    numbers1: List[str],
                                    numbers2: List[str],
                                    text1: str,
                                    text2: str) -> List[Dict[str, Any]]:
        """수치 불일치 감지"""
        
        discrepancies = []
        
        try:
            # 숫자를 실제 값으로 변환
            values1 = []
            for num_str in numbers1:
                try:
                    if '%' in num_str:
                        values1.append(float(num_str.replace('%', '')) / 100)
                    else:
                        values1.append(float(num_str))
                except ValueError:
                    continue
            
            values2 = []
            for num_str in numbers2:
                try:
                    if '%' in num_str:
                        values2.append(float(num_str.replace('%', '')) / 100)
                    else:
                        values2.append(float(num_str))
                except ValueError:
                    continue
            
            # 유사한 크기의 숫자들 비교
            for val1 in values1:
                for val2 in values2:
                    if abs(val1) > 0 and abs(val2) > 0:
                        diff_ratio = abs(val1 - val2) / max(abs(val1), abs(val2))
                        
                        if diff_ratio > self.thresholds['statistical_difference']:
                            discrepancies.append({
                                "value1": val1,
                                "value2": val2,
                                "difference_ratio": diff_ratio,
                                "context1": self._get_number_context(str(val1), text1),
                                "context2": self._get_number_context(str(val2), text2)
                            })
            
        except Exception as e:
            logger.warning(f"수치 불일치 감지 중 오류: {e}")
        
        return discrepancies
    
    def _extract_conclusion(self, text: str) -> Optional[str]:
        """텍스트에서 결론 부분 추출"""
        
        text_lower = text.lower()
        
        for keyword in self.conclusion_keywords:
            if keyword in text_lower:
                # 키워드 이후 텍스트 추출
                start_idx = text_lower.find(keyword)
                if start_idx != -1:
                    # 다음 문단이나 충분한 길이까지 추출
                    conclusion_part = text[start_idx:start_idx + 500]
                    return conclusion_part.strip()
        
        # 키워드가 없으면 마지막 문단 반환
        paragraphs = text.split('\n\n')
        if paragraphs:
            return paragraphs[-1].strip()
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        
        try:
            # 간단한 토큰 기반 유사도
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_keyword_coverage(self, text: str) -> Dict[str, int]:
        """키워드 커버리지 분석"""
        
        text_lower = text.lower()
        coverage = {}
        
        # 통계 키워드
        coverage['statistical'] = sum(1 for kw in self.statistical_keywords if kw in text_lower)
        
        # 결론 키워드
        coverage['conclusion'] = sum(1 for kw in self.conclusion_keywords if kw in text_lower)
        
        return coverage
    
    def _get_number_context(self, number: str, text: str, context_length: int = 50) -> str:
        """숫자 주변 컨텍스트 추출"""
        
        idx = text.find(number)
        if idx == -1:
            return ""
        
        start = max(0, idx - context_length)
        end = min(len(text), idx + len(number) + context_length)
        
        return text[start:end].strip()
    
    def _aggregate_analysis(self, analysis: ConflictAnalysis):
        """분석 결과 집계"""
        
        analysis.total_conflicts = len(analysis.conflicts)
        
        # 유형별/심각도별 집계
        for conflict in analysis.conflicts:
            analysis.conflicts_by_type[conflict.conflict_type] += 1
            analysis.conflicts_by_severity[conflict.severity] += 1
        
        # 전체 일관성 점수 계산
        if analysis.total_conflicts == 0:
            analysis.overall_consistency_score = 1.0
        else:
            # 심각도 가중 평균으로 계산
            severity_weights = {
                ConflictSeverity.CRITICAL: 1.0,
                ConflictSeverity.HIGH: 0.8,
                ConflictSeverity.MEDIUM: 0.5,
                ConflictSeverity.LOW: 0.2
            }
            
            weighted_conflicts = sum(
                severity_weights.get(conflict.severity, 0.5)
                for conflict in analysis.conflicts
            )
            
            max_possible_score = len(analysis.conflicts)
            analysis.overall_consistency_score = max(0.0, 1.0 - weighted_conflicts / max_possible_score)
        
        # 권장 조치 생성
        if analysis.total_conflicts == 0:
            analysis.recommended_actions.append("모든 에이전트 결과가 일관성을 보입니다.")
        else:
            analysis.recommended_actions.extend([
                f"총 {analysis.total_conflicts}개 충돌 해결 필요",
                f"높은 우선순위 충돌: {analysis.conflicts_by_severity.get(ConflictSeverity.HIGH, 0)}개",
                f"치명적 충돌: {analysis.conflicts_by_severity.get(ConflictSeverity.CRITICAL, 0)}개"
            ])
    
    def _generate_resolution(self, 
                           conflict: ConflictInstance, 
                           results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """개별 충돌 해결 방안 생성"""
        
        resolution = {
            "conflict_id": conflict.conflict_id,
            "strategy": conflict.suggested_strategy.value,
            "confidence": conflict.resolution_confidence,
            "steps": [],
            "expected_outcome": "",
            "risk_level": "low"
        }
        
        try:
            if conflict.suggested_strategy == ResolutionStrategy.QUALITY_BASED:
                # 품질 기반 해결
                resolution["steps"] = [
                    "각 에이전트의 품질 점수 비교",
                    "가장 높은 품질의 결과 선택",
                    "선택된 결과를 기준으로 최종 답변 구성"
                ]
                resolution["expected_outcome"] = "가장 신뢰할 수 있는 결과 채택"
                
            elif conflict.suggested_strategy == ResolutionStrategy.CONSENSUS_BASED:
                # 합의 기반 해결
                resolution["steps"] = [
                    "공통 결과 요소 식별",
                    "차이점에 대한 가중 평균 계산",
                    "합의된 결과로 통합"
                ]
                resolution["expected_outcome"] = "에이전트 간 합의된 결과 도출"
                
            elif conflict.suggested_strategy == ResolutionStrategy.EXCLUDE_OUTLIER:
                # 이상치 제외
                resolution["steps"] = [
                    "이상치 에이전트 식별",
                    "나머지 에이전트 결과로 재분석",
                    "정상 범위 결과만 활용"
                ]
                resolution["expected_outcome"] = "일관성 있는 결과 집합 확보"
                resolution["risk_level"] = "medium"
                
            else:
                # 수동 검토 필요
                resolution["steps"] = [
                    "충돌 상세 내용 검토",
                    "도메인 전문가 의견 수렴",
                    "수동으로 최적 해결 방안 결정"
                ]
                resolution["expected_outcome"] = "전문가 판단 기반 해결"
                resolution["risk_level"] = "high"
        
        except Exception as e:
            logger.error(f"해결 방안 생성 중 오류: {e}")
            resolution["error"] = str(e)
        
        return resolution
    
    def _generate_implementation_steps(self, resolutions: List[Dict[str, Any]]) -> List[str]:
        """통합 구현 단계 생성"""
        
        steps = []
        
        try:
            # 자동 해결 가능한 충돌들
            auto_resolvable = [r for r in resolutions 
                             if r["strategy"] != ResolutionStrategy.MANUAL_REVIEW.value]
            
            if auto_resolvable:
                steps.append("1. 자동 해결 가능한 충돌들 처리")
                steps.extend([f"   - {r['conflict_id']}: {r['strategy']}" 
                            for r in auto_resolvable])
            
            # 수동 검토 필요한 충돌들
            manual_review = [r for r in resolutions 
                           if r["strategy"] == ResolutionStrategy.MANUAL_REVIEW.value]
            
            if manual_review:
                steps.append("2. 수동 검토 필요한 충돌들")
                steps.extend([f"   - {r['conflict_id']}: 전문가 검토 필요" 
                            for r in manual_review])
            
            steps.append("3. 해결된 결과로 최종 답변 재구성")
            steps.append("4. 통합 결과 품질 검증")
            
        except Exception as e:
            logger.error(f"구현 단계 생성 중 오류: {e}")
            steps.append(f"오류로 인한 수동 처리 필요: {str(e)}")
        
        return steps