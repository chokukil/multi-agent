"""
멀티 에이전트 결과 통합 시스템

이 모듈은 여러 A2A 에이전트의 결과를 통합하여 
중복을 제거하고 일관성 있는 최종 결과를 생성하는 시스템을 제공합니다.

주요 기능:
- 에이전트 결과 통합 알고리즘
- 중복 정보 제거 및 일관성 확보
- 사용자 질문과의 연관성 분석
- 우선순위 기반 결과 선택
"""

import json
import logging
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .agent_result_collector import AgentResult, CollectionSession
from .result_validator import QualityMetrics, ValidationResult
from .conflict_detector import ConflictAnalysis, ConflictInstance, ResolutionStrategy

logger = logging.getLogger(__name__)

class IntegrationStrategy(Enum):
    """통합 전략"""
    QUALITY_WEIGHTED = "quality_weighted"      # 품질 가중 통합
    CONSENSUS_BASED = "consensus_based"        # 합의 기반 통합
    BEST_RESULT = "best_result"               # 최고 결과 선택
    COMPREHENSIVE = "comprehensive"            # 종합 통합
    CONFLICT_AWARE = "conflict_aware"          # 충돌 인식 통합

class ContentType(Enum):
    """콘텐츠 유형"""
    STATISTICAL = "statistical"               # 통계 정보
    VISUALIZATION = "visualization"           # 시각화
    TEXTUAL = "textual"                      # 텍스트 설명
    INSIGHT = "insight"                      # 인사이트
    RECOMMENDATION = "recommendation"         # 권장사항

@dataclass
class IntegratedContent:
    """통합된 콘텐츠"""
    content_type: ContentType
    content: Any
    confidence: float
    sources: List[str]  # 소스 에이전트 ID들
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationResult:
    """통합 결과"""
    session_id: str
    query: str
    strategy: IntegrationStrategy
    
    # 통합된 콘텐츠
    integrated_text: str = ""
    integrated_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    integrated_insights: List[str] = field(default_factory=list)
    
    # 품질 지표
    overall_confidence: float = 0.0
    integration_quality: float = 0.0
    coverage_score: float = 0.0
    
    # 소스 정보
    contributing_agents: List[str] = field(default_factory=list)
    excluded_agents: List[str] = field(default_factory=list)
    
    # 메타데이터
    integration_time: datetime = field(default_factory=datetime.now)
    processing_notes: List[str] = field(default_factory=list)
    
    # 성공/실패 상태
    success: bool = True
    error_message: Optional[str] = None

class MultiAgentResultIntegrator:
    """멀티 에이전트 결과 통합기"""
    
    def __init__(self):
        # 통합 설정
        self.min_confidence_threshold = 0.3
        self.duplicate_similarity_threshold = 0.85
        self.consensus_threshold = 0.6
        
        # 가중치 설정
        self.quality_weight = 0.4
        self.confidence_weight = 0.3
        self.consistency_weight = 0.3
        
        # 키워드 패턴
        self.insight_keywords = [
            '분석', 'analysis', '결론', 'conclusion', '발견', 'finding',
            '인사이트', 'insight', '패턴', 'pattern', '트렌드', 'trend'
        ]
        
        self.recommendation_keywords = [
            '권장', 'recommend', '제안', 'suggest', '개선', 'improve',
            '다음단계', 'next step', '액션', 'action', '조치', 'measure'
        ]
    
    def integrate_results(self,
                         session: CollectionSession,
                         quality_metrics: Dict[str, QualityMetrics] = None,
                         conflict_analysis: ConflictAnalysis = None,
                         strategy: IntegrationStrategy = IntegrationStrategy.COMPREHENSIVE) -> IntegrationResult:
        """멀티 에이전트 결과 통합"""
        
        logger.info(f"🔄 결과 통합 시작 - 세션 {session.session_id}, "
                   f"에이전트 {len(session.collected_results)}개, "
                   f"전략: {strategy.value}")
        
        integration_result = IntegrationResult(
            session_id=session.session_id,
            query=session.query,
            strategy=strategy
        )
        
        try:
            # 1. 유효한 결과 필터링
            valid_results = self._filter_valid_results(
                session.collected_results, quality_metrics
            )
            
            if not valid_results:
                integration_result.success = False
                integration_result.error_message = "통합할 유효한 결과가 없습니다"
                return integration_result
            
            integration_result.contributing_agents = list(valid_results.keys())
            integration_result.excluded_agents = [
                aid for aid in session.collected_results.keys()
                if aid not in valid_results
            ]
            
            # 2. 전략별 통합 실행
            if strategy == IntegrationStrategy.QUALITY_WEIGHTED:
                self._integrate_quality_weighted(integration_result, valid_results, quality_metrics)
            
            elif strategy == IntegrationStrategy.CONSENSUS_BASED:
                self._integrate_consensus_based(integration_result, valid_results)
            
            elif strategy == IntegrationStrategy.BEST_RESULT:
                self._integrate_best_result(integration_result, valid_results, quality_metrics)
            
            elif strategy == IntegrationStrategy.COMPREHENSIVE:
                self._integrate_comprehensive(integration_result, valid_results, quality_metrics)
            
            elif strategy == IntegrationStrategy.CONFLICT_AWARE:
                self._integrate_conflict_aware(integration_result, valid_results, conflict_analysis)
            
            # 3. 품질 지표 계산
            self._calculate_integration_metrics(integration_result, valid_results, quality_metrics)
            
            # 4. 후처리
            self._post_process_integration(integration_result)
            
            logger.info(f"✅ 결과 통합 완료 - 품질: {integration_result.integration_quality:.3f}, "
                       f"신뢰도: {integration_result.overall_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 결과 통합 중 오류: {e}")
            integration_result.success = False
            integration_result.error_message = str(e)
        
        return integration_result
    
    def _filter_valid_results(self,
                             results: Dict[str, AgentResult],
                             quality_metrics: Dict[str, QualityMetrics] = None) -> Dict[str, AgentResult]:
        """유효한 결과 필터링"""
        
        valid_results = {}
        
        for agent_id, result in results.items():
            # 기본 유효성 검사
            if not result.processed_text and not result.artifacts:
                logger.warning(f"에이전트 {agent_id}: 빈 결과로 제외")
                continue
            
            # 에러 상태 확인
            if result.error_message:
                logger.warning(f"에이전트 {agent_id}: 에러 상태로 제외 - {result.error_message}")
                continue
            
            # 품질 기준 확인
            if quality_metrics and agent_id in quality_metrics:
                quality_score = quality_metrics[agent_id].overall_score
                if quality_score < self.min_confidence_threshold:
                    logger.warning(f"에이전트 {agent_id}: 품질 기준 미달로 제외 ({quality_score:.3f})")
                    continue
            
            valid_results[agent_id] = result
        
        logger.info(f"유효한 결과: {len(valid_results)}/{len(results)}개")
        return valid_results
    
    def _integrate_quality_weighted(self,
                                  integration_result: IntegrationResult,
                                  results: Dict[str, AgentResult],
                                  quality_metrics: Dict[str, QualityMetrics] = None):
        """품질 가중 통합"""
        
        logger.info("품질 가중 통합 실행")
        
        # 품질 가중치 계산
        weights = {}
        total_weight = 0.0
        
        for agent_id in results.keys():
            if quality_metrics and agent_id in quality_metrics:
                weight = quality_metrics[agent_id].overall_score
            else:
                weight = 0.5  # 기본 가중치
            
            weights[agent_id] = weight
            total_weight += weight
        
        # 가중치 정규화
        if total_weight > 0:
            weights = {aid: w / total_weight for aid, w in weights.items()}
        
        # 텍스트 통합 (가중 선택)
        text_segments = []
        for agent_id, result in results.items():
            if result.processed_text:
                weight = weights.get(agent_id, 0.0)
                # 가중치가 높은 결과 우선 포함
                if weight > 0.2:  # 20% 이상 가중치
                    text_segments.append({
                        'text': result.processed_text,
                        'weight': weight,
                        'agent_id': agent_id
                    })
        
        # 가중치 순으로 정렬
        text_segments.sort(key=lambda x: x['weight'], reverse=True)
        
        # 통합 텍스트 생성
        integrated_parts = []
        for segment in text_segments:
            # 중복 제거를 위한 유사도 검사
            is_duplicate = False
            for existing_part in integrated_parts:
                if self._calculate_text_similarity(segment['text'], existing_part) > self.duplicate_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                integrated_parts.append(segment['text'])
        
        integration_result.integrated_text = '\n\n'.join(integrated_parts)
        
        # 아티팩트 통합 (품질 기준)
        self._integrate_artifacts_by_quality(integration_result, results, weights)
        
        integration_result.processing_notes.append(f"품질 가중 통합 완료 - {len(text_segments)}개 텍스트 세그먼트")
    
    def _integrate_consensus_based(self,
                                 integration_result: IntegrationResult,
                                 results: Dict[str, AgentResult]):
        """합의 기반 통합"""
        
        logger.info("합의 기반 통합 실행")
        
        # 공통 요소 추출
        common_elements = self._extract_common_elements(results)
        
        # 합의된 내용 우선 포함
        consensus_text_parts = []
        
        # 공통 키워드/주제 기반 문장 추출
        for common_element in common_elements:
            sentences = []
            for result in results.values():
                # 해당 요소를 포함하는 문장들 찾기
                text_sentences = self._extract_sentences_with_element(
                    result.processed_text, common_element
                )
                sentences.extend(text_sentences)
            
            if sentences:
                # 가장 대표적인 문장 선택
                representative_sentence = self._select_representative_sentence(sentences)
                if representative_sentence:
                    consensus_text_parts.append(representative_sentence)
        
        # 개별 특색있는 내용 추가
        unique_parts = []
        for agent_id, result in results.items():
            unique_content = self._extract_unique_content(
                result.processed_text, common_elements
            )
            if unique_content:
                unique_parts.append({
                    'content': unique_content,
                    'agent_id': agent_id
                })
        
        # 최종 텍스트 구성
        final_parts = consensus_text_parts
        
        # 고유 내용 중 가치 있는 것들 추가
        for unique_part in unique_parts[:3]:  # 최대 3개
            final_parts.append(unique_part['content'])
        
        integration_result.integrated_text = '\n\n'.join(final_parts)
        
        # 합의된 아티팩트 선택
        self._integrate_artifacts_by_consensus(integration_result, results)
        
        integration_result.processing_notes.append(f"합의 기반 통합 완료 - 공통요소 {len(common_elements)}개")
    
    def _integrate_best_result(self,
                             integration_result: IntegrationResult,
                             results: Dict[str, AgentResult],
                             quality_metrics: Dict[str, QualityMetrics] = None):
        """최고 결과 선택"""
        
        logger.info("최고 결과 선택 통합 실행")
        
        # 최고 품질 결과 선택
        best_agent_id = None
        best_score = -1.0
        
        for agent_id in results.keys():
            score = 0.0
            
            if quality_metrics and agent_id in quality_metrics:
                score = quality_metrics[agent_id].overall_score
            else:
                # 기본 점수 계산
                result = results[agent_id]
                score = (
                    (0.4 if result.processed_text else 0.0) +
                    (0.3 * min(1.0, len(result.artifacts) / 3)) +
                    (0.3 if not result.error_message else 0.0)
                )
            
            if score > best_score:
                best_score = score
                best_agent_id = agent_id
        
        if best_agent_id:
            best_result = results[best_agent_id]
            integration_result.integrated_text = best_result.processed_text
            integration_result.integrated_artifacts = best_result.artifacts.copy()
            
            # 다른 결과에서 보완 정보 추가
            supplementary_info = []
            for agent_id, result in results.items():
                if agent_id != best_agent_id and result.processed_text:
                    # 최고 결과에 없는 유니크한 정보 찾기
                    unique_info = self._find_unique_information(
                        result.processed_text, best_result.processed_text
                    )
                    if unique_info:
                        supplementary_info.append(unique_info)
            
            if supplementary_info:
                integration_result.integrated_text += "\n\n**추가 정보:**\n" + '\n'.join(supplementary_info)
            
            integration_result.processing_notes.append(f"최고 결과 선택 - 에이전트 {best_agent_id} (점수: {best_score:.3f})")
        
    def _integrate_comprehensive(self,
                               integration_result: IntegrationResult,
                               results: Dict[str, AgentResult],
                               quality_metrics: Dict[str, QualityMetrics] = None):
        """종합 통합"""
        
        logger.info("종합 통합 실행")
        
        # 여러 통합 전략 조합
        
        # 1. 품질 기반 필터링
        high_quality_results = {}
        medium_quality_results = {}
        
        for agent_id, result in results.items():
            quality_score = 0.5
            if quality_metrics and agent_id in quality_metrics:
                quality_score = quality_metrics[agent_id].overall_score
            
            if quality_score >= 0.7:
                high_quality_results[agent_id] = result
            elif quality_score >= 0.4:
                medium_quality_results[agent_id] = result
        
        # 2. 고품질 결과 우선 통합
        primary_text_parts = []
        if high_quality_results:
            for result in high_quality_results.values():
                if result.processed_text:
                    primary_text_parts.append(result.processed_text)
        
        # 3. 중품질 결과에서 보완 정보 추출
        supplementary_parts = []
        for result in medium_quality_results.values():
            if result.processed_text:
                # 기존 내용과 중복되지 않는 정보 찾기
                unique_content = result.processed_text
                for primary_part in primary_text_parts:
                    if self._calculate_text_similarity(unique_content, primary_part) > 0.7:
                        unique_content = ""
                        break
                
                if unique_content and len(unique_content) > 50:
                    supplementary_parts.append(unique_content)
        
        # 4. 텍스트 구조화
        final_text_parts = []
        
        # 주요 분석 결과
        if primary_text_parts:
            final_text_parts.append("## 🔍 주요 분석 결과\n")
            final_text_parts.extend(primary_text_parts)
        
        # 보충 정보
        if supplementary_parts:
            final_text_parts.append("\n## 📎 추가 정보\n")
            final_text_parts.extend(supplementary_parts[:2])  # 최대 2개
        
        integration_result.integrated_text = '\n\n'.join(final_text_parts)
        
        # 5. 아티팩트 종합 통합
        self._integrate_artifacts_comprehensive(integration_result, results, quality_metrics)
        
        # 6. 인사이트 추출
        integration_result.integrated_insights = self._extract_integrated_insights(results)
        
        integration_result.processing_notes.append(
            f"종합 통합 완료 - 고품질 {len(high_quality_results)}개, 중품질 {len(medium_quality_results)}개"
        )
    
    def _integrate_conflict_aware(self,
                                integration_result: IntegrationResult,
                                results: Dict[str, AgentResult],
                                conflict_analysis: ConflictAnalysis = None):
        """충돌 인식 통합"""
        
        logger.info("충돌 인식 통합 실행")
        
        if not conflict_analysis or not conflict_analysis.conflicts:
            # 충돌이 없으면 종합 통합으로 fallback
            self._integrate_comprehensive(integration_result, results)
            integration_result.processing_notes.append("충돌 없음 - 종합 통합으로 진행")
            return
        
        # 충돌 해결 기반 통합
        resolved_results = {}
        conflicted_agents = set()
        
        # 충돌 관련 에이전트 식별
        for conflict in conflict_analysis.conflicts:
            conflicted_agents.update(conflict.involved_agents)
        
        # 충돌 없는 결과들 우선 채택
        for agent_id, result in results.items():
            if agent_id not in conflicted_agents:
                resolved_results[agent_id] = result
        
        # 충돌 해결 전략 적용
        for conflict in conflict_analysis.conflicts:
            if conflict.suggested_strategy == ResolutionStrategy.QUALITY_BASED:
                # 품질 기반 선택
                best_agent = self._select_best_quality_agent(
                    conflict.involved_agents, results
                )
                if best_agent and best_agent in results:
                    resolved_results[best_agent] = results[best_agent]
            
            elif conflict.suggested_strategy == ResolutionStrategy.CONSENSUS_BASED:
                # 합의 내용만 채택
                consensus_content = self._extract_consensus_from_conflict(
                    conflict.involved_agents, results
                )
                if consensus_content:
                    # 가상 결과 생성
                    consensus_result = AgentResult(
                        agent_id="consensus",
                        agent_name="Consensus",
                        endpoint="internal",
                        status=results[conflict.involved_agents[0]].status
                    )
                    consensus_result.processed_text = consensus_content
                    resolved_results["consensus"] = consensus_result
        
        # 해결된 결과들로 통합
        if resolved_results:
            self._integrate_comprehensive(integration_result, resolved_results)
            
            # 충돌 정보 추가
            if conflict_analysis.conflicts:
                conflict_note = f"\n\n**⚠️ 충돌 해결 정보:**\n"
                conflict_note += f"- 총 {len(conflict_analysis.conflicts)}개 충돌 감지 및 해결\n"
                conflict_note += f"- 충돌 관련 에이전트: {', '.join(conflicted_agents)}\n"
                
                integration_result.integrated_text += conflict_note
        
        integration_result.processing_notes.append(
            f"충돌 인식 통합 완료 - {len(conflict_analysis.conflicts)}개 충돌 처리"
        )
    
    def _integrate_artifacts_by_quality(self,
                                      integration_result: IntegrationResult,
                                      results: Dict[str, AgentResult],
                                      weights: Dict[str, float]):
        """품질 기반 아티팩트 통합"""
        
        # 아티팩트 타입별 최고 품질 선택
        artifact_by_type = defaultdict(list)
        
        for agent_id, result in results.items():
            weight = weights.get(agent_id, 0.0)
            
            for artifact in result.artifacts:
                art_type = artifact.get('type', 'unknown')
                artifact_by_type[art_type].append({
                    'artifact': artifact,
                    'weight': weight,
                    'agent_id': agent_id
                })
        
        # 타입별 최고 품질 아티팩트 선택
        for art_type, artifacts in artifact_by_type.items():
            # 가중치 순 정렬
            artifacts.sort(key=lambda x: x['weight'], reverse=True)
            
            # 최고 품질 아티팩트 선택
            best_artifact = artifacts[0]['artifact'].copy()
            best_artifact['source_agent'] = artifacts[0]['agent_id']
            best_artifact['quality_weight'] = artifacts[0]['weight']
            
            integration_result.integrated_artifacts.append(best_artifact)
    
    def _integrate_artifacts_by_consensus(self,
                                        integration_result: IntegrationResult,
                                        results: Dict[str, AgentResult]):
        """합의 기반 아티팩트 통합"""
        
        # 아티팩트 타입별 빈도 계산
        artifact_type_count = Counter()
        artifact_by_type = defaultdict(list)
        
        for agent_id, result in results.items():
            for artifact in result.artifacts:
                art_type = artifact.get('type', 'unknown')
                artifact_type_count[art_type] += 1
                artifact_by_type[art_type].append({
                    'artifact': artifact,
                    'agent_id': agent_id
                })
        
        # 합의 임계값 이상인 아티팩트 타입만 포함
        min_consensus = max(1, int(len(results) * self.consensus_threshold))
        
        for art_type, count in artifact_type_count.items():
            if count >= min_consensus:
                # 해당 타입의 대표 아티팩트 선택
                artifacts = artifact_by_type[art_type]
                representative = artifacts[0]['artifact'].copy()  # 첫 번째를 대표로
                representative['consensus_count'] = count
                representative['total_agents'] = len(results)
                
                integration_result.integrated_artifacts.append(representative)
    
    def _integrate_artifacts_comprehensive(self,
                                         integration_result: IntegrationResult,
                                         results: Dict[str, AgentResult],
                                         quality_metrics: Dict[str, QualityMetrics] = None):
        """종합 아티팩트 통합"""
        
        # 아티팩트 수집 및 중복 제거
        unique_artifacts = {}
        
        for agent_id, result in results.items():
            quality_score = 0.5
            if quality_metrics and agent_id in quality_metrics:
                quality_score = quality_metrics[agent_id].overall_score
            
            for artifact in result.artifacts:
                art_type = artifact.get('type', 'unknown')
                art_key = f"{art_type}_{self._calculate_artifact_hash(artifact)}"
                
                if art_key not in unique_artifacts or quality_score > unique_artifacts[art_key]['quality']:
                    artifact_copy = artifact.copy()
                    artifact_copy['source_agent'] = agent_id
                    artifact_copy['quality_score'] = quality_score
                    
                    unique_artifacts[art_key] = {
                        'artifact': artifact_copy,
                        'quality': quality_score
                    }
        
        # 품질 순으로 정렬하여 추가
        sorted_artifacts = sorted(
            unique_artifacts.values(),
            key=lambda x: x['quality'],
            reverse=True
        )
        
        integration_result.integrated_artifacts = [
            item['artifact'] for item in sorted_artifacts
        ]
    
    def _calculate_integration_metrics(self,
                                     integration_result: IntegrationResult,
                                     results: Dict[str, AgentResult],
                                     quality_metrics: Dict[str, QualityMetrics] = None):
        """통합 품질 지표 계산"""
        
        # 전체 신뢰도 계산
        confidence_scores = []
        for agent_id in integration_result.contributing_agents:
            if quality_metrics and agent_id in quality_metrics:
                confidence_scores.append(quality_metrics[agent_id].overall_score)
            else:
                confidence_scores.append(0.5)
        
        if confidence_scores:
            integration_result.overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 통합 품질 계산
        quality_factors = []
        
        # 텍스트 품질
        if integration_result.integrated_text:
            text_quality = min(1.0, len(integration_result.integrated_text) / 1000)
            quality_factors.append(text_quality)
        
        # 아티팩트 품질
        if integration_result.integrated_artifacts:
            artifact_quality = min(1.0, len(integration_result.integrated_artifacts) / 5)
            quality_factors.append(artifact_quality)
        
        # 커버리지 점수
        total_agents = len(results)
        contributing_agents = len(integration_result.contributing_agents)
        integration_result.coverage_score = contributing_agents / total_agents if total_agents > 0 else 0.0
        
        if quality_factors:
            integration_result.integration_quality = sum(quality_factors) / len(quality_factors)
        
        # 성공 여부 결정
        integration_result.success = (
            integration_result.integration_quality > 0.3 and
            integration_result.overall_confidence > 0.3 and
            integration_result.coverage_score > 0.5
        )
    
    def _post_process_integration(self, integration_result: IntegrationResult):
        """통합 결과 후처리"""
        
        # 텍스트 정리
        if integration_result.integrated_text:
            # 중복 문단 제거
            paragraphs = integration_result.integrated_text.split('\n\n')
            unique_paragraphs = []
            
            for paragraph in paragraphs:
                is_duplicate = False
                for existing in unique_paragraphs:
                    if self._calculate_text_similarity(paragraph, existing) > 0.9:
                        is_duplicate = True
                        break
                
                if not is_duplicate and len(paragraph.strip()) > 10:
                    unique_paragraphs.append(paragraph)
            
            integration_result.integrated_text = '\n\n'.join(unique_paragraphs)
        
        # 아티팩트 메타데이터 정리
        for artifact in integration_result.integrated_artifacts:
            if 'metadata' not in artifact:
                artifact['metadata'] = {}
            
            artifact['metadata']['integration_time'] = integration_result.integration_time.isoformat()
            artifact['metadata']['integration_strategy'] = integration_result.strategy.value
    
    def _extract_common_elements(self, results: Dict[str, AgentResult]) -> List[str]:
        """공통 요소 추출"""
        
        common_elements = []
        
        # 모든 텍스트에서 키워드 추출
        all_words = []
        for result in results.values():
            if result.processed_text:
                # 단어 토큰화 (간단한 방식)
                words = re.findall(r'\b\w+\b', result.processed_text.lower())
                all_words.extend(words)
        
        # 빈도 기반 공통 키워드 식별
        word_counts = Counter(all_words)
        
        # 최소 절반 이상의 결과에서 나타나는 키워드
        min_frequency = max(2, len(results) // 2)
        
        for word, count in word_counts.items():
            if count >= min_frequency and len(word) > 3:
                common_elements.append(word)
        
        return common_elements[:10]  # 상위 10개
    
    def _extract_sentences_with_element(self, text: str, element: str) -> List[str]:
        """특정 요소를 포함하는 문장 추출"""
        
        sentences = re.split(r'[.!?]+', text)
        matching_sentences = []
        
        for sentence in sentences:
            if element.lower() in sentence.lower() and len(sentence.strip()) > 20:
                matching_sentences.append(sentence.strip())
        
        return matching_sentences
    
    def _select_representative_sentence(self, sentences: List[str]) -> Optional[str]:
        """대표 문장 선택"""
        
        if not sentences:
            return None
        
        # 가장 긴 문장을 대표로 선택 (더 많은 정보 포함 가능성)
        return max(sentences, key=len)
    
    def _extract_unique_content(self, text: str, common_elements: List[str]) -> str:
        """고유 콘텐츠 추출"""
        
        # 공통 요소를 제외한 독특한 내용 찾기
        sentences = re.split(r'[.!?]+', text)
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                # 공통 요소가 많이 포함되지 않은 문장
                common_count = sum(1 for element in common_elements 
                                 if element.lower() in sentence.lower())
                
                if common_count < len(common_elements) * 0.3:  # 30% 미만
                    unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences[:3])  # 최대 3문장
    
    def _find_unique_information(self, source_text: str, reference_text: str) -> str:
        """참조 텍스트에 없는 고유 정보 찾기"""
        
        source_sentences = re.split(r'[.!?]+', source_text)
        unique_info = []
        
        for sentence in source_sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # 참조 텍스트와 유사도가 낮은 문장 찾기
                similarity = self._calculate_text_similarity(sentence, reference_text)
                if similarity < 0.3:
                    unique_info.append(sentence)
        
        return '. '.join(unique_info[:2])  # 최대 2문장
    
    def _extract_integrated_insights(self, results: Dict[str, AgentResult]) -> List[str]:
        """통합 인사이트 추출"""
        
        insights = []
        
        # 각 결과에서 인사이트 관련 문장 추출
        for result in results.values():
            if result.processed_text:
                text_lower = result.processed_text.lower()
                
                for keyword in self.insight_keywords:
                    if keyword in text_lower:
                        # 키워드 주변 문장 추출
                        sentences = re.split(r'[.!?]+', result.processed_text)
                        for sentence in sentences:
                            if keyword in sentence.lower() and len(sentence) > 30:
                                insights.append(sentence.strip())
        
        # 중복 제거 및 품질 필터링
        unique_insights = []
        for insight in insights:
            is_duplicate = False
            for existing in unique_insights:
                if self._calculate_text_similarity(insight, existing) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_insights.append(insight)
        
        return unique_insights[:5]  # 최대 5개
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        
        if not text1 or not text2:
            return 0.0
        
        try:
            # 단어 집합 기반 Jaccard 유사도
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_artifact_hash(self, artifact: Dict[str, Any]) -> str:
        """아티팩트 해시 계산 (중복 감지용)"""
        
        try:
            # 주요 필드만으로 해시 생성
            key_fields = ['type', 'title', 'columns', 'shape']
            hash_data = {}
            
            for field in key_fields:
                if field in artifact:
                    hash_data[field] = artifact[field]
            
            return str(hash(json.dumps(hash_data, sort_keys=True)))
            
        except Exception:
            return str(hash(str(artifact)))
    
    def _select_best_quality_agent(self, 
                                 agent_ids: List[str], 
                                 results: Dict[str, AgentResult]) -> Optional[str]:
        """품질 기반 최고 에이전트 선택"""
        
        best_agent = None
        best_score = -1.0
        
        for agent_id in agent_ids:
            if agent_id in results:
                result = results[agent_id]
                
                # 간단한 품질 점수 계산
                score = 0.0
                if result.processed_text:
                    score += 0.5
                if result.artifacts:
                    score += 0.3
                if not result.error_message:
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
        
        return best_agent
    
    def _extract_consensus_from_conflict(self, 
                                       agent_ids: List[str], 
                                       results: Dict[str, AgentResult]) -> str:
        """충돌에서 합의 내용 추출"""
        
        if len(agent_ids) < 2:
            return ""
        
        # 관련 결과들의 텍스트 수집
        texts = []
        for agent_id in agent_ids:
            if agent_id in results and results[agent_id].processed_text:
                texts.append(results[agent_id].processed_text)
        
        if len(texts) < 2:
            return ""
        
        # 공통 키워드 기반 합의 내용 추출
        common_elements = self._extract_common_elements({
            f"agent_{i}": AgentResult(
                agent_id=f"agent_{i}",
                agent_name=f"Agent{i}",
                endpoint="temp",
                status=results[agent_ids[0]].status
            ) for i, text in enumerate(texts)
        })
        
        # 공통 요소를 포함하는 문장들로 합의 내용 구성
        consensus_sentences = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # 공통 요소를 많이 포함하는 문장
                    common_count = sum(1 for element in common_elements 
                                     if element.lower() in sentence.lower())
                    
                    if common_count >= len(common_elements) * 0.5:  # 50% 이상
                        consensus_sentences.append(sentence)
        
        # 중복 제거
        unique_sentences = []
        for sentence in consensus_sentences:
            is_duplicate = False
            for existing in unique_sentences:
                if self._calculate_text_similarity(sentence, existing) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences[:3])  # 최대 3문장