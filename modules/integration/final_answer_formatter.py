"""
최종 답변 포맷팅 시스템

이 모듈은 멀티 에이전트 분석 결과를 구조화된 최종 답변으로 포맷팅하여
사용자에게 명확하고 실행 가능한 인사이트를 전달하는 시스템을 제공합니다.

주요 기능:
- 구조화된 최종 답변 템플릿
- 마크다운 기반 전문적 포맷팅
- 아티팩트 임베딩 및 컨텍스트 설명
- Progressive Disclosure 지원
- 신뢰도 및 품질 지표 표시
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .result_integrator import IntegrationResult
from .insight_generator import InsightAnalysis, Insight, InsightType, InsightPriority
from .recommendation_generator import RecommendationPlan, Recommendation, Priority
from .agent_result_collector import AgentResult
from .result_validator import QualityMetrics

logger = logging.getLogger(__name__)

class AnswerFormat(Enum):
    """답변 형식"""
    EXECUTIVE_SUMMARY = "executive_summary"      # 경영진 요약
    DETAILED_ANALYSIS = "detailed_analysis"      # 상세 분석
    TECHNICAL_REPORT = "technical_report"        # 기술 리포트
    QUICK_INSIGHTS = "quick_insights"           # 빠른 인사이트
    PRESENTATION = "presentation"               # 프레젠테이션 형식

class DisclosureLevel(Enum):
    """공개 수준"""
    SUMMARY_ONLY = "summary_only"               # 요약만
    WITH_INSIGHTS = "with_insights"             # 인사이트 포함
    WITH_RECOMMENDATIONS = "with_recommendations" # 추천사항 포함
    FULL_DETAILS = "full_details"               # 모든 세부사항

@dataclass
class FormattedAnswer:
    """포맷팅된 최종 답변"""
    session_id: str
    query: str
    answer_format: AnswerFormat
    disclosure_level: DisclosureLevel
    
    # 메인 콘텐츠
    executive_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    detailed_analysis: str = ""
    insights_section: str = ""
    recommendations_section: str = ""
    
    # 아티팩트 섹션
    artifacts_section: str = ""
    artifact_descriptions: List[str] = field(default_factory=list)
    
    # 품질 및 신뢰도
    quality_indicators: Dict[str, Any] = field(default_factory=dict)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 메타데이터
    contributing_agents: List[str] = field(default_factory=list)
    analysis_methodology: str = ""
    limitations: List[str] = field(default_factory=list)
    
    # 최종 답변
    formatted_answer: str = ""
    
    # 생성 정보
    generated_at: datetime = field(default_factory=datetime.now)
    processing_notes: List[str] = field(default_factory=list)

class FinalAnswerFormatter:
    """최종 답변 포맷팅 시스템"""
    
    def __init__(self):
        # 포맷팅 설정
        self.max_summary_length = 500
        self.max_findings_count = 5
        self.max_insights_display = 8
        self.max_recommendations_display = 10
        
        # 마크다운 템플릿
        self.templates = {
            AnswerFormat.EXECUTIVE_SUMMARY: self._get_executive_template(),
            AnswerFormat.DETAILED_ANALYSIS: self._get_detailed_template(),
            AnswerFormat.TECHNICAL_REPORT: self._get_technical_template(),
            AnswerFormat.QUICK_INSIGHTS: self._get_quick_template(),
            AnswerFormat.PRESENTATION: self._get_presentation_template()
        }
        
        # 신뢰도 레벨 매핑
        self.confidence_levels = {
            (0.9, 1.0): "매우 높음",
            (0.7, 0.9): "높음", 
            (0.5, 0.7): "보통",
            (0.3, 0.5): "낮음",
            (0.0, 0.3): "매우 낮음"
        }
        
        # 우선순위 아이콘
        self.priority_icons = {
            InsightPriority.CRITICAL: "🚨",
            InsightPriority.HIGH: "🔴",
            InsightPriority.MEDIUM: "🟡",
            InsightPriority.LOW: "🟢",
            InsightPriority.INFORMATIONAL: "ℹ️"
        }
        
        self.recommendation_icons = {
            Priority.URGENT: "⚡",
            Priority.HIGH: "🔴",
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢",
            Priority.FUTURE: "🔮"
        }
    
    def format_final_answer(self,
                           integration_result: IntegrationResult,
                           insight_analysis: InsightAnalysis = None,
                           recommendation_plan: RecommendationPlan = None,
                           quality_metrics: Dict[str, QualityMetrics] = None,
                           answer_format: AnswerFormat = AnswerFormat.DETAILED_ANALYSIS,
                           disclosure_level: DisclosureLevel = DisclosureLevel.FULL_DETAILS) -> FormattedAnswer:
        """최종 답변 포맷팅"""
        
        logger.info(f"📝 최종 답변 포맷팅 시작 - 세션 {integration_result.session_id}, "
                   f"형식: {answer_format.value}, 수준: {disclosure_level.value}")
        
        formatted_answer = FormattedAnswer(
            session_id=integration_result.session_id,
            query=integration_result.query,
            answer_format=answer_format,
            disclosure_level=disclosure_level,
            contributing_agents=integration_result.contributing_agents.copy()
        )
        
        try:
            # 1. 경영진 요약 생성
            formatted_answer.executive_summary = self._generate_executive_summary(
                integration_result, insight_analysis, recommendation_plan
            )
            
            # 2. 주요 발견사항 생성
            formatted_answer.key_findings = self._generate_key_findings(
                integration_result, insight_analysis
            )
            
            # 3. 상세 분석 섹션 생성
            if disclosure_level.value in ['with_insights', 'with_recommendations', 'full_details']:
                formatted_answer.detailed_analysis = self._generate_detailed_analysis(
                    integration_result
                )
            
            # 4. 인사이트 섹션 생성
            if insight_analysis and disclosure_level.value in ['with_insights', 'with_recommendations', 'full_details']:
                formatted_answer.insights_section = self._generate_insights_section(
                    insight_analysis
                )
            
            # 5. 추천사항 섹션 생성
            if recommendation_plan and disclosure_level.value in ['with_recommendations', 'full_details']:
                formatted_answer.recommendations_section = self._generate_recommendations_section(
                    recommendation_plan
                )
            
            # 6. 아티팩트 섹션 생성
            if integration_result.integrated_artifacts:
                formatted_answer.artifacts_section = self._generate_artifacts_section(
                    integration_result.integrated_artifacts
                )
            
            # 7. 품질 및 신뢰도 지표 생성
            formatted_answer.quality_indicators = self._generate_quality_indicators(
                integration_result, insight_analysis, quality_metrics
            )
            
            # 8. 신뢰도 메트릭 생성
            formatted_answer.confidence_metrics = self._generate_confidence_metrics(
                integration_result, insight_analysis
            )
            
            # 9. 분석 방법론 및 제한사항
            formatted_answer.analysis_methodology = self._generate_methodology_section(
                integration_result
            )
            formatted_answer.limitations = self._generate_limitations(
                integration_result, insight_analysis
            )
            
            # 10. 최종 답변 조합
            formatted_answer.formatted_answer = self._assemble_final_answer(
                formatted_answer, answer_format, disclosure_level
            )
            
            logger.info(f"✅ 최종 답변 포맷팅 완료 - 길이: {len(formatted_answer.formatted_answer)}자")
            
        except Exception as e:
            logger.error(f"❌ 최종 답변 포맷팅 중 오류: {e}")
            formatted_answer.processing_notes.append(f"포맷팅 오류: {str(e)}")
            formatted_answer.formatted_answer = self._generate_error_fallback(
                integration_result, str(e)
            )
        
        return formatted_answer
    
    def _generate_executive_summary(self,
                                  integration_result: IntegrationResult,
                                  insight_analysis: InsightAnalysis = None,
                                  recommendation_plan: RecommendationPlan = None) -> str:
        """경영진 요약 생성"""
        
        summary_parts = []
        
        try:
            # 기본 분석 개요
            summary_parts.append(f"**분석 개요**: {integration_result.query}에 대한 종합 분석을 완료했습니다.")
            
            # 참여 에이전트 정보
            agent_count = len(integration_result.contributing_agents)
            summary_parts.append(f"총 {agent_count}개 분석 에이전트가 참여하여 다각도로 검토했습니다.")
            
            # 주요 결과 요약
            if integration_result.integrated_artifacts:
                artifact_count = len(integration_result.integrated_artifacts)
                summary_parts.append(f"{artifact_count}개의 시각화 및 데이터 분석 결과물이 생성되었습니다.")
            
            # 인사이트 요약
            if insight_analysis and insight_analysis.total_insights > 0:
                high_priority_insights = len([i for i in insight_analysis.insights 
                                            if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]])
                if high_priority_insights > 0:
                    summary_parts.append(f"**핵심 발견**: {high_priority_insights}개의 중요한 인사이트가 도출되었습니다.")
            
            # 추천사항 요약
            if recommendation_plan and recommendation_plan.total_recommendations > 0:
                urgent_recs = len([r for r in recommendation_plan.recommendations 
                                 if r.priority in [Priority.URGENT, Priority.HIGH]])
                if urgent_recs > 0:
                    summary_parts.append(f"**즉시 조치**: {urgent_recs}개의 우선순위 높은 조치사항이 식별되었습니다.")
            
            # 품질 지표
            quality_score = integration_result.integration_quality
            confidence_score = integration_result.overall_confidence
            
            quality_text = self._get_quality_description(quality_score)
            confidence_text = self._get_confidence_description(confidence_score)
            
            summary_parts.append(f"**분석 품질**: {quality_text} (신뢰도: {confidence_text})")
            
        except Exception as e:
            logger.warning(f"경영진 요약 생성 중 오류: {e}")
            summary_parts = [f"분석이 완료되었으나 요약 생성 중 일부 오류가 발생했습니다."]
        
        summary = " ".join(summary_parts)
        
        # 길이 제한
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        return summary
    
    def _generate_key_findings(self,
                             integration_result: IntegrationResult,
                             insight_analysis: InsightAnalysis = None) -> List[str]:
        """주요 발견사항 생성"""
        
        findings = []
        
        try:
            # 인사이트 기반 주요 발견사항
            if insight_analysis and insight_analysis.key_findings:
                findings.extend(insight_analysis.key_findings[:3])
            
            # 통합 결과 기반 발견사항
            if integration_result.integrated_text:
                # 텍스트에서 주요 수치나 결론 추출
                text_findings = self._extract_key_findings_from_text(
                    integration_result.integrated_text
                )
                findings.extend(text_findings[:2])
            
            # 아티팩트 기반 발견사항
            if integration_result.integrated_artifacts:
                artifact_findings = self._extract_findings_from_artifacts(
                    integration_result.integrated_artifacts
                )
                findings.extend(artifact_findings[:2])
            
            # 기본 발견사항 (없는 경우)
            if not findings:
                findings.append("다양한 각도에서 데이터 분석이 수행되었습니다")
                if integration_result.integrated_artifacts:
                    findings.append(f"{len(integration_result.integrated_artifacts)}개의 시각화 자료가 생성되었습니다")
        
        except Exception as e:
            logger.warning(f"주요 발견사항 생성 중 오류: {e}")
            findings = ["분석이 완료되었습니다"]
        
        return findings[:self.max_findings_count]
    
    def _generate_detailed_analysis(self, integration_result: IntegrationResult) -> str:
        """상세 분석 섹션 생성"""
        
        analysis_parts = []
        
        try:
            if integration_result.integrated_text:
                # 텍스트를 섹션으로 나누어 구조화
                sections = self._structure_analysis_text(integration_result.integrated_text)
                
                for section_title, content in sections.items():
                    if content.strip():
                        analysis_parts.append(f"### {section_title}\n\n{content}\n")
            
            if not analysis_parts:
                analysis_parts.append("### 분석 결과\n\n상세한 분석 결과가 아래 시각화 자료를 통해 제공됩니다.\n")
        
        except Exception as e:
            logger.warning(f"상세 분석 생성 중 오류: {e}")
            analysis_parts = ["### 분석 결과\n\n분석이 완료되었으나 상세 내용 표시 중 오류가 발생했습니다.\n"]
        
        return "\n".join(analysis_parts)
    
    def _generate_insights_section(self, insight_analysis: InsightAnalysis) -> str:
        """인사이트 섹션 생성"""
        
        if not insight_analysis.insights:
            return ""
        
        section_parts = ["## 🔍 핵심 인사이트\n"]
        
        try:
            # 우선순위별로 정렬
            sorted_insights = sorted(
                insight_analysis.insights,
                key=lambda i: (i.priority.value, -i.confidence),
                reverse=False
            )
            
            displayed_insights = sorted_insights[:self.max_insights_display]
            
            for i, insight in enumerate(displayed_insights, 1):
                icon = self.priority_icons.get(insight.priority, "📊")
                
                insight_text = f"### {icon} {insight.title}\n\n"
                insight_text += f"{insight.description}\n\n"
                
                # 신뢰도 표시
                confidence_text = self._get_confidence_description(insight.confidence)
                insight_text += f"**신뢰도**: {confidence_text} | "
                insight_text += f"**영향도**: {insight.impact_score:.1%}\n\n"
                
                # 증거 표시 (간단히)
                if insight.evidence:
                    insight_text += f"**근거**: {insight.evidence[0][:100]}...\n\n"
                
                section_parts.append(insight_text)
            
            # 추가 인사이트가 있는 경우
            if len(insight_analysis.insights) > self.max_insights_display:
                remaining = len(insight_analysis.insights) - self.max_insights_display
                section_parts.append(f"*추가로 {remaining}개의 인사이트가 더 있습니다.*\n")
        
        except Exception as e:
            logger.warning(f"인사이트 섹션 생성 중 오류: {e}")
            section_parts.append("인사이트 표시 중 오류가 발생했습니다.\n")
        
        return "\n".join(section_parts)
    
    def _generate_recommendations_section(self, recommendation_plan: RecommendationPlan) -> str:
        """추천사항 섹션 생성"""
        
        if not recommendation_plan.recommendations:
            return ""
        
        section_parts = ["## 💡 추천사항\n"]
        
        try:
            # 즉시 조치 사항
            if recommendation_plan.immediate_actions:
                section_parts.append("### ⚡ 즉시 조치 필요\n")
                for rec in recommendation_plan.immediate_actions[:3]:
                    section_parts.append(self._format_single_recommendation(rec))
                section_parts.append("")
            
            # 단기 계획
            if recommendation_plan.short_term_plan:
                section_parts.append("### 🎯 단기 계획 (1개월 내)\n")
                for rec in recommendation_plan.short_term_plan[:4]:
                    section_parts.append(self._format_single_recommendation(rec))
                section_parts.append("")
            
            # 장기 계획
            if recommendation_plan.long_term_plan:
                section_parts.append("### 🔮 장기 계획\n")
                for rec in recommendation_plan.long_term_plan[:3]:
                    section_parts.append(self._format_single_recommendation(rec))
            
        except Exception as e:
            logger.warning(f"추천사항 섹션 생성 중 오류: {e}")
            section_parts.append("추천사항 표시 중 오류가 발생했습니다.\n")
        
        return "\n".join(section_parts)
    
    def _generate_artifacts_section(self, artifacts: List[Dict[str, Any]]) -> str:
        """아티팩트 섹션 생성"""
        
        if not artifacts:
            return ""
        
        section_parts = ["## 📊 분석 결과물\n"]
        
        try:
            for i, artifact in enumerate(artifacts, 1):
                art_type = artifact.get('type', 'unknown')
                
                if art_type == 'plotly_chart':
                    title = artifact.get('layout', {}).get('title', {}).get('text', f'차트 {i}')
                    section_parts.append(f"### 📈 {title}\n")
                    section_parts.append("대화형 차트가 생성되어 데이터의 패턴과 트렌드를 시각적으로 확인할 수 있습니다.\n")
                
                elif art_type == 'dataframe':
                    rows = len(artifact.get('data', []))
                    cols = len(artifact.get('columns', []))
                    section_parts.append(f"### 📋 데이터 테이블 {i}\n")
                    section_parts.append(f"정리된 데이터 ({rows:,}행 × {cols}열)를 표 형태로 제공합니다.\n")
                
                elif art_type == 'image':
                    section_parts.append(f"### 🖼️ 이미지 분석 결과 {i}\n")
                    section_parts.append("시각적 분석 결과가 이미지로 제공됩니다.\n")
                
                else:
                    section_parts.append(f"### 📄 분석 결과 {i}\n")
                    section_parts.append(f"{art_type} 형태의 분석 결과물이 생성되었습니다.\n")
                
                section_parts.append("")
        
        except Exception as e:
            logger.warning(f"아티팩트 섹션 생성 중 오류: {e}")
            section_parts.append("분석 결과물 표시 중 오류가 발생했습니다.\n")
        
        return "\n".join(section_parts)
    
    def _generate_quality_indicators(self,
                                   integration_result: IntegrationResult,
                                   insight_analysis: InsightAnalysis = None,
                                   quality_metrics: Dict[str, QualityMetrics] = None) -> Dict[str, Any]:
        """품질 지표 생성"""
        
        indicators = {}
        
        try:
            # 통합 품질
            indicators['integration_quality'] = {
                'score': integration_result.integration_quality,
                'description': self._get_quality_description(integration_result.integration_quality)
            }
            
            # 신뢰도
            indicators['confidence'] = {
                'score': integration_result.overall_confidence,
                'description': self._get_confidence_description(integration_result.overall_confidence)
            }
            
            # 커버리지
            indicators['coverage'] = {
                'score': integration_result.coverage_score,
                'description': f"{integration_result.coverage_score:.1%} 에이전트 기여"
            }
            
            # 인사이트 품질
            if insight_analysis:
                indicators['insight_quality'] = {
                    'score': insight_analysis.overall_quality_score,
                    'count': insight_analysis.total_insights,
                    'description': f"{insight_analysis.total_insights}개 인사이트, 평균 품질 {insight_analysis.overall_quality_score:.1%}"
                }
            
            # 데이터 품질 (가능한 경우)
            if quality_metrics:
                avg_data_quality = sum(m.data_integrity_score for m in quality_metrics.values()) / len(quality_metrics)
                indicators['data_quality'] = {
                    'score': avg_data_quality,
                    'description': f"평균 데이터 품질 {avg_data_quality:.1%}"
                }
        
        except Exception as e:
            logger.warning(f"품질 지표 생성 중 오류: {e}")
            indicators['error'] = {'description': f"품질 지표 생성 오류: {str(e)}"}
        
        return indicators
    
    def _generate_confidence_metrics(self,
                                   integration_result: IntegrationResult,
                                   insight_analysis: InsightAnalysis = None) -> Dict[str, float]:
        """신뢰도 메트릭 생성"""
        
        metrics = {}
        
        try:
            # 기본 신뢰도
            metrics['overall'] = integration_result.overall_confidence
            metrics['integration'] = integration_result.integration_quality
            metrics['coverage'] = integration_result.coverage_score
            
            # 인사이트 신뢰도
            if insight_analysis and insight_analysis.insights:
                insight_confidences = [i.confidence for i in insight_analysis.insights]
                metrics['insights'] = sum(insight_confidences) / len(insight_confidences)
            
            # 종합 신뢰도 점수
            metrics['composite'] = (
                metrics['overall'] * 0.4 +
                metrics['integration'] * 0.3 +
                metrics['coverage'] * 0.2 +
                metrics.get('insights', 0.5) * 0.1
            )
        
        except Exception as e:
            logger.warning(f"신뢰도 메트릭 생성 중 오류: {e}")
            metrics['error'] = 0.0
        
        return metrics
    
    def _generate_methodology_section(self, integration_result: IntegrationResult) -> str:
        """분석 방법론 섹션 생성"""
        
        methodology_parts = []
        
        try:
            methodology_parts.append(f"**분석 전략**: {integration_result.strategy.value}")
            methodology_parts.append(f"**참여 에이전트**: {len(integration_result.contributing_agents)}개")
            
            if integration_result.integrated_artifacts:
                artifact_types = set(art.get('type', 'unknown') for art in integration_result.integrated_artifacts)
                methodology_parts.append(f"**생성 결과물**: {', '.join(artifact_types)}")
            
            methodology_parts.append(f"**통합 시간**: {integration_result.integration_time.strftime('%Y-%m-%d %H:%M')}")
        
        except Exception as e:
            logger.warning(f"방법론 섹션 생성 중 오류: {e}")
            methodology_parts = ["표준 멀티 에이전트 분석 방법론 적용"]
        
        return " | ".join(methodology_parts)
    
    def _generate_limitations(self,
                            integration_result: IntegrationResult,
                            insight_analysis: InsightAnalysis = None) -> List[str]:
        """제한사항 생성"""
        
        limitations = []
        
        try:
            # 커버리지 기반 제한사항
            if integration_result.coverage_score < 0.8:
                excluded_count = len(integration_result.excluded_agents)
                if excluded_count > 0:
                    limitations.append(f"{excluded_count}개 에이전트가 분석에 기여하지 못했습니다")
            
            # 신뢰도 기반 제한사항
            if integration_result.overall_confidence < 0.7:
                limitations.append("일부 분석 결과의 신뢰도가 제한적입니다")
            
            # 인사이트 기반 제한사항
            if insight_analysis:
                low_confidence_insights = len([i for i in insight_analysis.insights if i.confidence < 0.6])
                if low_confidence_insights > 0:
                    limitations.append(f"{low_confidence_insights}개 인사이트의 신뢰도가 상대적으로 낮습니다")
            
            # 기본 제한사항
            if not limitations:
                limitations.append("자동화된 분석으로 인한 해석의 한계가 있을 수 있습니다")
        
        except Exception as e:
            logger.warning(f"제한사항 생성 중 오류: {e}")
            limitations = ["분석 결과 해석 시 주의가 필요합니다"]
        
        return limitations
    
    def _assemble_final_answer(self,
                             formatted_answer: FormattedAnswer,
                             answer_format: AnswerFormat,
                             disclosure_level: DisclosureLevel) -> str:
        """최종 답변 조합"""
        
        try:
            template = self.templates.get(answer_format, self.templates[AnswerFormat.DETAILED_ANALYSIS])
            
            # 템플릿 변수 치환
            final_answer = template.format(
                query=formatted_answer.query,
                executive_summary=formatted_answer.executive_summary,
                key_findings=self._format_key_findings(formatted_answer.key_findings),
                detailed_analysis=formatted_answer.detailed_analysis,
                insights_section=formatted_answer.insights_section,
                recommendations_section=formatted_answer.recommendations_section,
                artifacts_section=formatted_answer.artifacts_section,
                quality_section=self._format_quality_section(formatted_answer.quality_indicators),
                methodology=formatted_answer.analysis_methodology,
                limitations=self._format_limitations(formatted_answer.limitations),
                timestamp=formatted_answer.generated_at.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            return final_answer
            
        except Exception as e:
            logger.error(f"최종 답변 조합 중 오류: {e}")
            return self._generate_error_fallback(None, str(e))
    
    def _format_single_recommendation(self, recommendation: Recommendation) -> str:
        """단일 추천사항 포맷팅"""
        
        icon = self.recommendation_icons.get(recommendation.priority, "📋")
        
        rec_text = f"#### {icon} {recommendation.title}\n\n"
        rec_text += f"{recommendation.description}\n\n"
        rec_text += f"**예상 노력**: {recommendation.estimated_effort} | "
        rec_text += f"**예상 임팩트**: {recommendation.expected_impact:.1%}\n\n"
        
        if recommendation.action_steps:
            rec_text += f"**실행 단계**: {recommendation.action_steps[0]}\n\n"
        
        return rec_text
    
    def _format_key_findings(self, findings: List[str]) -> str:
        """주요 발견사항 포맷팅"""
        
        if not findings:
            return "- 분석이 완료되었습니다"
        
        return "\n".join(f"- {finding}" for finding in findings)
    
    def _format_quality_section(self, quality_indicators: Dict[str, Any]) -> str:
        """품질 섹션 포맷팅"""
        
        if not quality_indicators:
            return "품질 지표를 사용할 수 없습니다."
        
        quality_parts = []
        
        for key, indicator in quality_indicators.items():
            if isinstance(indicator, dict) and 'description' in indicator:
                quality_parts.append(f"**{key.replace('_', ' ').title()}**: {indicator['description']}")
        
        return " | ".join(quality_parts) if quality_parts else "품질 평가 완료"
    
    def _format_limitations(self, limitations: List[str]) -> str:
        """제한사항 포맷팅"""
        
        if not limitations:
            return "특별한 제한사항 없음"
        
        return "\n".join(f"- {limitation}" for limitation in limitations)
    
    # 템플릿 메서드들
    def _get_executive_template(self) -> str:
        return """# 📊 분석 결과 요약

## 🎯 개요
{executive_summary}

## 🔍 주요 발견사항
{key_findings}

{quality_section}

---
*분석 완료: {timestamp}*
"""
    
    def _get_detailed_template(self) -> str:
        return """# 📊 {query} - 종합 분석 결과

## 🎯 경영진 요약
{executive_summary}

## 🔍 주요 발견사항
{key_findings}

{detailed_analysis}

{insights_section}

{recommendations_section}

{artifacts_section}

## 📋 분석 정보
**방법론**: {methodology}

**제한사항**: 
{limitations}

**품질 지표**: {quality_section}

---
*분석 완료 시점: {timestamp}*
"""
    
    def _get_technical_template(self) -> str:
        return """# 🔬 기술 분석 리포트: {query}

## 📈 분석 개요
{executive_summary}

## 🎯 핵심 결과
{key_findings}

{detailed_analysis}

{insights_section}

{artifacts_section}

## 📊 품질 및 신뢰도
{quality_section}

## 🔧 분석 방법론
{methodology}

## ⚠️ 제한사항 및 고려사항
{limitations}

{recommendations_section}

---
*리포트 생성: {timestamp}*
"""
    
    def _get_quick_template(self) -> str:
        return """# ⚡ {query} - 빠른 인사이트

{executive_summary}

## 🎯 핵심 발견
{key_findings}

{insights_section}

---
*{timestamp}*
"""
    
    def _get_presentation_template(self) -> str:
        return """# 🎤 {query} - 프레젠테이션

---

## 📋 AGENDA
1. 개요 및 주요 발견사항
2. 상세 분석 결과
3. 핵심 인사이트
4. 추천사항
5. Q&A

---

## 🎯 개요
{executive_summary}

---

## 🔍 주요 발견사항
{key_findings}

---

{insights_section}

---

{recommendations_section}

---

## 📊 품질 지표
{quality_section}

---

*프레젠테이션 생성: {timestamp}*
"""
    
    # 유틸리티 메서드들
    def _get_quality_description(self, score: float) -> str:
        """품질 점수 설명"""
        
        for (min_score, max_score), description in self.confidence_levels.items():
            if min_score <= score < max_score:
                return f"{description} ({score:.1%})"
        
        return f"평가 불가 ({score:.1%})"
    
    def _get_confidence_description(self, score: float) -> str:
        """신뢰도 점수 설명"""
        
        for (min_score, max_score), description in self.confidence_levels.items():
            if min_score <= score < max_score:
                return f"{description} ({score:.1%})"
        
        return f"평가 불가 ({score:.1%})"
    
    def _extract_key_findings_from_text(self, text: str) -> List[str]:
        """텍스트에서 주요 발견사항 추출"""
        
        findings = []
        
        try:
            # 수치가 포함된 문장 찾기
            import re
            
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and re.search(r'\d+(?:\.\d+)?%?', sentence):
                    findings.append(sentence)
                    if len(findings) >= 3:
                        break
        
        except Exception:
            pass
        
        return findings
    
    def _extract_findings_from_artifacts(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """아티팩트에서 발견사항 추출"""
        
        findings = []
        
        try:
            chart_count = sum(1 for art in artifacts if art.get('type') == 'plotly_chart')
            table_count = sum(1 for art in artifacts if art.get('type') == 'dataframe')
            
            if chart_count > 0:
                findings.append(f"{chart_count}개의 시각화 차트를 통한 패턴 분석 완료")
            
            if table_count > 0:
                findings.append(f"{table_count}개의 데이터 테이블로 상세 정보 제공")
        
        except Exception:
            pass
        
        return findings
    
    def _structure_analysis_text(self, text: str) -> Dict[str, str]:
        """분석 텍스트를 섹션으로 구조화"""
        
        sections = {}
        
        try:
            # 간단한 섹션 분리 (향후 더 정교한 로직으로 개선 가능)
            paragraphs = text.split('\n\n')
            
            current_section = "분석 결과"
            current_content = []
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    # 섹션 헤더 감지 (## 또는 특정 키워드)
                    if paragraph.startswith('##') or any(keyword in paragraph.lower() 
                                                        for keyword in ['결론', '요약', '분석', '발견']):
                        if current_content:
                            sections[current_section] = '\n\n'.join(current_content)
                        
                        current_section = paragraph.replace('##', '').strip()
                        current_content = []
                    else:
                        current_content.append(paragraph)
            
            # 마지막 섹션 추가
            if current_content:
                sections[current_section] = '\n\n'.join(current_content)
        
        except Exception:
            sections = {"분석 결과": text}
        
        return sections
    
    def _generate_error_fallback(self, integration_result=None, error_message: str = "") -> str:
        """오류 발생 시 폴백 답변"""
        
        fallback = f"""# ⚠️ 분석 결과

분석이 완료되었으나 결과 포맷팅 중 일부 문제가 발생했습니다.

## 상황
- 오류 내용: {error_message}
- 발생 시점: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 권장사항
1. 분석을 다시 실행해보세요
2. 데이터 품질을 확인해보세요
3. 시스템 관리자에게 문의하세요

---
*자동 생성된 오류 리포트*
"""
        
        return fallback