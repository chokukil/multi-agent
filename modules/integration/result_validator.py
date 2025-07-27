"""
멀티 에이전트 결과 검증 및 품질 평가 시스템

이 모듈은 A2A 에이전트들의 결과 데이터에 대한 무결성 검사,
완성도 평가, 신뢰도 분석을 수행하는 시스템을 제공합니다.

주요 기능:
- 데이터 무결성 검사 및 형식 검증
- 결과 완성도 및 신뢰도 평가
- 누락된 정보 식별 및 보완 제안
- 품질 기반 결과 우선순위 결정
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .agent_result_collector import AgentResult, CollectionSession

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """검증 수준"""
    BASIC = "basic"          # 기본 검증 (형식, 구조)
    STANDARD = "standard"    # 표준 검증 (내용, 완성도)
    COMPREHENSIVE = "comprehensive"  # 종합 검증 (품질, 신뢰도)

class ValidationStatus(Enum):
    """검증 상태"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationResult:
    """검증 결과"""
    status: ValidationStatus
    score: float  # 0.0 ~ 1.0
    message: str
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class QualityMetrics:
    """품질 지표"""
    # 데이터 품질
    data_integrity_score: float = 0.0
    format_validity_score: float = 0.0
    content_richness_score: float = 0.0
    
    # 완성도 지표
    completeness_score: float = 0.0
    coverage_score: float = 0.0
    depth_score: float = 0.0
    
    # 신뢰도 지표
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    reliability_score: float = 0.0
    
    # 종합 점수
    overall_score: float = 0.0
    
    def calculate_overall_score(self):
        """종합 점수 계산"""
        scores = [
            self.data_integrity_score,
            self.format_validity_score,
            self.content_richness_score,
            self.completeness_score,
            self.coverage_score,
            self.depth_score,
            self.consistency_score,
            self.accuracy_score,
            self.reliability_score
        ]
        
        # 유효한 점수만으로 평균 계산
        valid_scores = [s for s in scores if s > 0.0]
        self.overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

class ResultValidator:
    """멀티 에이전트 결과 검증기"""
    
    def __init__(self):
        # 검증 규칙 설정
        self.min_text_length = 10
        self.max_text_length = 50000
        self.expected_artifact_types = {
            'plotly_chart', 'dataframe', 'image', 'code', 'text'
        }
        
        # 품질 임계값
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
        
        # 키워드 패턴
        self.error_patterns = [
            r'error', r'exception', r'failed', r'failure',
            r'timeout', r'connection\s+refused', r'not\s+found'
        ]
        
        self.success_indicators = [
            r'successfully', r'completed', r'finished',
            r'analysis', r'result', r'conclusion'
        ]
    
    def validate_agent_result(self, 
                            result: AgentResult,
                            level: ValidationLevel = ValidationLevel.STANDARD) -> Tuple[ValidationResult, QualityMetrics]:
        """단일 에이전트 결과 검증"""
        
        logger.info(f"🔍 에이전트 결과 검증 시작 - {result.agent_name} ({level.value})")
        
        metrics = QualityMetrics()
        validations = []
        
        try:
            # 기본 검증
            if level.value in ['basic', 'standard', 'comprehensive']:
                validations.extend(self._validate_basic_structure(result, metrics))
            
            # 표준 검증
            if level.value in ['standard', 'comprehensive']:
                validations.extend(self._validate_content_quality(result, metrics))
                validations.extend(self._validate_completeness(result, metrics))
            
            # 종합 검증
            if level.value == 'comprehensive':
                validations.extend(self._validate_reliability(result, metrics))
                validations.extend(self._validate_consistency(result, metrics))
            
            # 종합 점수 계산
            metrics.calculate_overall_score()
            
            # 전체 검증 결과 종합
            overall_validation = self._aggregate_validations(validations, metrics)
            
            logger.info(f"✅ 검증 완료 - {result.agent_name}, "
                       f"점수: {metrics.overall_score:.3f}, "
                       f"상태: {overall_validation.status.value}")
            
            return overall_validation, metrics
            
        except Exception as e:
            logger.error(f"❌ 검증 중 오류 - {result.agent_name}: {e}")
            
            error_validation = ValidationResult(
                status=ValidationStatus.FAILED,
                score=0.0,
                message=f"검증 과정에서 오류 발생: {str(e)}",
                details={"error": str(e)},
                recommendations=["결과 데이터를 다시 확인하고 재실행해주세요."]
            )
            
            return error_validation, metrics
    
    def validate_session_results(self, 
                                session: CollectionSession,
                                level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Tuple[ValidationResult, QualityMetrics]]:
        """세션 내 모든 결과 검증"""
        
        logger.info(f"🔍 세션 결과 검증 시작 - {session.session_id}, "
                   f"에이전트: {len(session.collected_results)}개")
        
        validation_results = {}
        
        for agent_id, result in session.collected_results.items():
            try:
                validation, metrics = self.validate_agent_result(result, level)
                validation_results[agent_id] = (validation, metrics)
                
            except Exception as e:
                logger.error(f"❌ 에이전트 {agent_id} 검증 실패: {e}")
                
                error_validation = ValidationResult(
                    status=ValidationStatus.FAILED,
                    score=0.0,
                    message=f"검증 실패: {str(e)}",
                    details={"error": str(e)},
                    recommendations=[]
                )
                
                validation_results[agent_id] = (error_validation, QualityMetrics())
        
        logger.info(f"✅ 세션 검증 완료 - {session.session_id}")
        
        return validation_results
    
    def identify_missing_information(self, 
                                   session: CollectionSession,
                                   expected_elements: List[str] = None) -> Dict[str, List[str]]:
        """누락된 정보 식별"""
        
        expected_elements = expected_elements or [
            'data_analysis', 'visualization', 'summary', 'insights'
        ]
        
        missing_info = {}
        
        for agent_id, result in session.collected_results.items():
            agent_missing = []
            
            # 기본 요소 확인
            if not result.processed_text or len(result.processed_text) < self.min_text_length:
                agent_missing.append("충분한 텍스트 설명")
            
            # 아티팩트 확인
            if not result.artifacts:
                agent_missing.append("데이터 시각화 또는 분석 결과물")
            
            # 예상 요소 확인
            text_lower = result.processed_text.lower()
            for element in expected_elements:
                if element.replace('_', ' ') not in text_lower:
                    agent_missing.append(f"{element} 관련 내용")
            
            if agent_missing:
                missing_info[agent_id] = agent_missing
        
        return missing_info
    
    def generate_quality_report(self, 
                              validation_results: Dict[str, Tuple[ValidationResult, QualityMetrics]]) -> Dict[str, Any]:
        """품질 보고서 생성"""
        
        total_agents = len(validation_results)
        if total_agents == 0:
            return {"error": "검증할 결과가 없습니다."}
        
        # 상태별 통계
        status_counts = {}
        quality_scores = []
        
        for validation, metrics in validation_results.values():
            status = validation.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            quality_scores.append(metrics.overall_score)
        
        # 평균 품질 점수
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # 품질 등급 결정
        if avg_quality >= self.quality_thresholds['excellent']:
            quality_grade = "Excellent"
        elif avg_quality >= self.quality_thresholds['good']:
            quality_grade = "Good"
        elif avg_quality >= self.quality_thresholds['acceptable']:
            quality_grade = "Acceptable"
        else:
            quality_grade = "Poor"
        
        # 권장사항 수집
        all_recommendations = []
        for validation, _ in validation_results.values():
            all_recommendations.extend(validation.recommendations)
        
        # 중복 제거 및 우선순위 정렬
        unique_recommendations = list(set(all_recommendations))
        
        return {
            "summary": {
                "total_agents": total_agents,
                "average_quality_score": round(avg_quality, 3),
                "quality_grade": quality_grade,
                "status_distribution": status_counts
            },
            "quality_metrics": {
                "scores": quality_scores,
                "min_score": min(quality_scores),
                "max_score": max(quality_scores),
                "std_deviation": self._calculate_std_dev(quality_scores)
            },
            "recommendations": unique_recommendations[:10],  # 상위 10개
            "detailed_results": {
                agent_id: {
                    "status": validation.status.value,
                    "score": metrics.overall_score,
                    "message": validation.message
                }
                for agent_id, (validation, metrics) in validation_results.items()
            }
        }
    
    def _validate_basic_structure(self, result: AgentResult, metrics: QualityMetrics) -> List[ValidationResult]:
        """기본 구조 검증"""
        
        validations = []
        
        # 1. 응답 존재 여부
        if not result.raw_response and not result.processed_text:
            validations.append(ValidationResult(
                status=ValidationStatus.FAILED,
                score=0.0,
                message="응답 데이터가 없습니다",
                details={"issue": "no_response"},
                recommendations=["에이전트가 적절한 응답을 생성했는지 확인하세요."]
            ))
            metrics.data_integrity_score = 0.0
        else:
            metrics.data_integrity_score = 0.8
        
        # 2. 텍스트 길이 검증
        text_length = len(result.processed_text)
        if text_length < self.min_text_length:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.3,
                message=f"응답이 너무 짧습니다 ({text_length}자)",
                details={"text_length": text_length, "min_required": self.min_text_length},
                recommendations=["더 상세한 분석이나 설명을 요청하세요."]
            ))
            metrics.content_richness_score = 0.3
        elif text_length > self.max_text_length:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.7,
                message=f"응답이 너무 깁니다 ({text_length}자)",
                details={"text_length": text_length, "max_recommended": self.max_text_length},
                recommendations=["응답을 요약하거나 핵심 내용만 추출하세요."]
            ))
            metrics.content_richness_score = 0.7
        else:
            metrics.content_richness_score = 0.9
        
        # 3. 형식 유효성 검증
        format_score = 0.5
        
        # JSON 형식 체크 (아티팩트가 있는 경우)
        if result.artifacts:
            valid_artifacts = 0
            for artifact in result.artifacts:
                if artifact.get('type') in self.expected_artifact_types:
                    valid_artifacts += 1
            
            if valid_artifacts == len(result.artifacts):
                format_score = 1.0
            elif valid_artifacts > 0:
                format_score = 0.7
        
        metrics.format_validity_score = format_score
        
        if format_score < 0.7:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=format_score,
                message="일부 아티팩트의 형식이 올바르지 않습니다",
                details={"valid_artifacts": valid_artifacts, "total_artifacts": len(result.artifacts)},
                recommendations=["아티팩트 생성 과정을 다시 확인하세요."]
            ))
        
        return validations
    
    def _validate_content_quality(self, result: AgentResult, metrics: QualityMetrics) -> List[ValidationResult]:
        """내용 품질 검증"""
        
        validations = []
        text = result.processed_text.lower()
        
        # 1. 에러 패턴 감지
        error_count = 0
        for pattern in self.error_patterns:
            error_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        if error_count > 3:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.3,
                message=f"에러 관련 내용이 많이 포함되어 있습니다 ({error_count}개)",
                details={"error_count": error_count},
                recommendations=["에이전트 실행 과정에서 발생한 문제를 해결하세요."]
            ))
            metrics.accuracy_score = 0.3
        elif error_count > 0:
            metrics.accuracy_score = 0.7
        else:
            metrics.accuracy_score = 0.9
        
        # 2. 성공 지표 확인
        success_count = 0
        for pattern in self.success_indicators:
            success_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        if success_count == 0:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.4,
                message="분석 완료를 나타내는 지표가 부족합니다",
                details={"success_indicators": success_count},
                recommendations=["분석 결과와 결론을 명확히 제시하세요."]
            ))
        
        return validations
    
    def _validate_completeness(self, result: AgentResult, metrics: QualityMetrics) -> List[ValidationResult]:
        """완성도 검증"""
        
        validations = []
        
        # 1. 기본 요소 완성도
        completeness_factors = []
        
        # 텍스트 설명
        if result.processed_text and len(result.processed_text) > 50:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.3)
        
        # 아티팩트
        if result.artifacts:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.0)
        
        # 메타데이터
        if result.meta:
            completeness_factors.append(0.8)
        else:
            completeness_factors.append(0.2)
        
        # 실행 성공
        if result.error_message:
            completeness_factors.append(0.2)
        else:
            completeness_factors.append(1.0)
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        metrics.completeness_score = completeness_score
        
        if completeness_score < 0.5:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=completeness_score,
                message=f"결과 완성도가 낮습니다 ({completeness_score:.1%})",
                details={"completeness_score": completeness_score},
                recommendations=["누락된 분석 요소들을 보완하세요."]
            ))
        
        # 2. 커버리지 평가
        text = result.processed_text.lower()
        coverage_keywords = ['data', 'analysis', 'result', 'chart', 'table', 'insight']
        found_keywords = sum(1 for kw in coverage_keywords if kw in text)
        
        coverage_score = found_keywords / len(coverage_keywords)
        metrics.coverage_score = coverage_score
        
        if coverage_score < 0.3:
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=coverage_score,
                message="분석 범위가 제한적입니다",
                details={"coverage_score": coverage_score},
                recommendations=["더 포괄적인 분석을 수행하세요."]
            ))
        
        return validations
    
    def _validate_reliability(self, result: AgentResult, metrics: QualityMetrics) -> List[ValidationResult]:
        """신뢰도 검증"""
        
        validations = []
        
        # 1. 실행 시간 기반 신뢰도
        exec_time = result.execution_duration
        if exec_time < 1.0:  # 너무 빠른 실행
            reliability_score = 0.4
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.4,
                message=f"실행 시간이 너무 짧습니다 ({exec_time:.1f}초)",
                details={"execution_time": exec_time},
                recommendations=["충분한 분석이 이루어졌는지 확인하세요."]
            ))
        elif exec_time > 300.0:  # 너무 긴 실행
            reliability_score = 0.6
            validations.append(ValidationResult(
                status=ValidationStatus.WARNING,
                score=0.6,
                message=f"실행 시간이 너무 깁니다 ({exec_time:.1f}초)",
                details={"execution_time": exec_time},
                recommendations=["처리 과정을 최적화하거나 타임아웃을 조정하세요."]
            ))
        else:
            reliability_score = 0.9
        
        metrics.reliability_score = reliability_score
        
        # 2. 데이터 일관성
        if result.artifacts:
            consistent_artifacts = 0
            for artifact in result.artifacts:
                if artifact.get('metadata') and artifact.get('content'):
                    consistent_artifacts += 1
            
            consistency_ratio = consistent_artifacts / len(result.artifacts)
            metrics.consistency_score = consistency_ratio
            
            if consistency_ratio < 0.7:
                validations.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    score=consistency_ratio,
                    message="아티팩트 데이터 일관성이 부족합니다",
                    details={"consistency_ratio": consistency_ratio},
                    recommendations=["아티팩트 생성 과정의 일관성을 확보하세요."]
                ))
        else:
            metrics.consistency_score = 0.5
        
        return validations
    
    def _validate_consistency(self, result: AgentResult, metrics: QualityMetrics) -> List[ValidationResult]:
        """일관성 검증"""
        
        validations = []
        
        # 텍스트-아티팩트 일관성 확인
        if result.artifacts and result.processed_text:
            text_mentions_charts = any(word in result.processed_text.lower() 
                                     for word in ['chart', 'graph', 'plot', 'visualization'])
            has_chart_artifacts = any(art.get('type') == 'plotly_chart' 
                                    for art in result.artifacts)
            
            text_mentions_tables = any(word in result.processed_text.lower() 
                                     for word in ['table', 'dataframe', 'data'])
            has_table_artifacts = any(art.get('type') == 'dataframe' 
                                    for art in result.artifacts)
            
            consistency_issues = []
            if text_mentions_charts and not has_chart_artifacts:
                consistency_issues.append("텍스트에서 차트를 언급했지만 차트 아티팩트가 없음")
            if text_mentions_tables and not has_table_artifacts:
                consistency_issues.append("텍스트에서 테이블을 언급했지만 테이블 아티팩트가 없음")
            
            if consistency_issues:
                validations.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    score=0.6,
                    message="텍스트와 아티팩트 간 일관성 문제가 있습니다",
                    details={"issues": consistency_issues},
                    recommendations=["텍스트 설명과 생성된 아티팩트의 일치성을 확인하세요."]
                ))
        
        return validations
    
    def _aggregate_validations(self, validations: List[ValidationResult], metrics: QualityMetrics) -> ValidationResult:
        """검증 결과 집계"""
        
        if not validations:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                score=metrics.overall_score,
                message="모든 검증을 통과했습니다",
                details={"metrics": metrics.__dict__},
                recommendations=[]
            )
        
        # 상태 우선순위: FAILED > WARNING > PASSED
        has_failed = any(v.status == ValidationStatus.FAILED for v in validations)
        has_warning = any(v.status == ValidationStatus.WARNING for v in validations)
        
        if has_failed:
            overall_status = ValidationStatus.FAILED
        elif has_warning:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # 메시지 및 권장사항 수집
        messages = [v.message for v in validations]
        all_recommendations = []
        for v in validations:
            all_recommendations.extend(v.recommendations)
        
        return ValidationResult(
            status=overall_status,
            score=metrics.overall_score,
            message=f"{len(validations)}개 검증 항목 중 문제 발견: " + "; ".join(messages),
            details={"validation_count": len(validations), "metrics": metrics.__dict__},
            recommendations=list(set(all_recommendations))  # 중복 제거
        )
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """표준편차 계산"""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5