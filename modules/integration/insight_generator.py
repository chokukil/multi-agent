"""
멀티 에이전트 인사이트 생성 시스템

이 모듈은 통합된 A2A 에이전트 결과에서 핵심 인사이트를 추출하고,
패턴 분석 및 트렌드 식별을 통해 비즈니스 가치를 창출하는 시스템을 제공합니다.

주요 기능:
- 핵심 인사이트 자동 추출
- 데이터 패턴 및 트렌드 분석
- 비즈니스 임팩트 평가
- 상관관계 및 이상치 감지
"""

import json
import logging
import re
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import pandas as pd

from .result_integrator import IntegrationResult
from .agent_result_collector import AgentResult

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """인사이트 유형"""
    TREND = "trend"                      # 트렌드 분석
    PATTERN = "pattern"                  # 패턴 식별
    CORRELATION = "correlation"          # 상관관계
    ANOMALY = "anomaly"                 # 이상치/예외
    DISTRIBUTION = "distribution"        # 분포 분석
    COMPARISON = "comparison"            # 비교 분석
    STATISTICAL = "statistical"          # 통계적 발견
    BUSINESS_IMPACT = "business_impact"   # 비즈니스 영향

class InsightPriority(Enum):
    """인사이트 우선순위"""
    CRITICAL = "critical"    # 즉시 주목 필요
    HIGH = "high"           # 높은 우선순위
    MEDIUM = "medium"       # 보통 우선순위
    LOW = "low"            # 낮은 우선순위
    INFORMATIONAL = "informational"  # 참고용

@dataclass
class Insight:
    """개별 인사이트"""
    insight_id: str
    title: str
    description: str
    insight_type: InsightType
    priority: InsightPriority
    
    # 신뢰도 및 영향도
    confidence: float  # 0.0 ~ 1.0
    impact_score: float  # 0.0 ~ 1.0
    
    # 지원 데이터
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    
    # 메타데이터
    source_agents: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    # 액션 아이템
    recommended_actions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

@dataclass
class InsightAnalysis:
    """인사이트 분석 결과"""
    session_id: str
    total_insights: int
    insights: List[Insight] = field(default_factory=list)
    
    # 요약 통계
    insights_by_type: Dict[InsightType, int] = field(default_factory=dict)
    insights_by_priority: Dict[InsightPriority, int] = field(default_factory=dict)
    
    # 품질 지표
    average_confidence: float = 0.0
    average_impact: float = 0.0
    overall_quality_score: float = 0.0
    
    # 주요 발견사항
    key_findings: List[str] = field(default_factory=list)
    top_insights: List[Insight] = field(default_factory=list)
    
    # 추가 정보
    processing_notes: List[str] = field(default_factory=list)
    generation_time: datetime = field(default_factory=datetime.now)

class InsightGenerator:
    """멀티 에이전트 인사이트 생성기"""
    
    def __init__(self):
        # 임계값 설정
        self.thresholds = {
            'correlation_significance': 0.7,      # 상관관계 유의성
            'trend_confidence': 0.6,              # 트렌드 신뢰도
            'anomaly_threshold': 2.0,             # 이상치 임계값 (표준편차)
            'pattern_frequency': 0.3,             # 패턴 빈도
            'statistical_significance': 0.05      # 통계적 유의성
        }
        
        # 키워드 패턴
        self.trend_keywords = [
            '증가', 'increase', '감소', 'decrease', '상승', 'rise',
            '하락', 'decline', '개선', 'improve', '악화', 'worsen'
        ]
        
        self.pattern_keywords = [
            '패턴', 'pattern', '규칙', 'rule', '주기', 'cycle',
            '반복', 'repeat', '경향', 'tendency', '특성', 'characteristic'
        ]
        
        self.comparison_keywords = [
            '비교', 'compare', '차이', 'difference', '대비', 'vs',
            '높은', 'higher', '낮은', 'lower', '많은', 'more', '적은', 'less'
        ]
        
        # 통계 패턴
        self.statistical_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',                    # 백분율
            r'평균.*?(\d+(?:\.\d+)?)',                  # 평균값
            r'최대.*?(\d+(?:\.\d+)?)',                  # 최대값
            r'최소.*?(\d+(?:\.\d+)?)',                  # 최소값
            r'표준편차.*?(\d+(?:\.\d+)?)',              # 표준편차
            r'(\d+(?:\.\d+)?)\s*배',                   # 배수 관계
        ]
    
    def generate_insights(self, 
                         integration_result: IntegrationResult,
                         agent_results: Dict[str, AgentResult] = None) -> InsightAnalysis:
        """통합 결과에서 인사이트 생성"""
        
        logger.info(f"🧠 인사이트 생성 시작 - 세션 {integration_result.session_id}")
        
        analysis = InsightAnalysis(
            session_id=integration_result.session_id,
            total_insights=0
        )
        
        try:
            insights = []
            
            # 1. 텍스트 기반 인사이트 추출
            if integration_result.integrated_text:
                text_insights = self._extract_text_insights(
                    integration_result.integrated_text,
                    integration_result.contributing_agents
                )
                insights.extend(text_insights)
            
            # 2. 아티팩트 기반 인사이트 추출
            if integration_result.integrated_artifacts:
                artifact_insights = self._extract_artifact_insights(
                    integration_result.integrated_artifacts,
                    integration_result.contributing_agents
                )
                insights.extend(artifact_insights)
            
            # 3. 크로스 분석 인사이트
            if agent_results:
                cross_insights = self._extract_cross_analysis_insights(
                    agent_results, integration_result
                )
                insights.extend(cross_insights)
            
            # 4. 인사이트 품질 평가 및 필터링
            qualified_insights = self._evaluate_and_filter_insights(insights)
            
            # 5. 인사이트 우선순위 결정
            prioritized_insights = self._prioritize_insights(qualified_insights)
            
            analysis.insights = prioritized_insights
            analysis.total_insights = len(prioritized_insights)
            
            # 6. 분석 결과 집계
            self._aggregate_analysis(analysis)
            
            # 7. 주요 발견사항 생성
            analysis.key_findings = self._generate_key_findings(prioritized_insights)
            analysis.top_insights = prioritized_insights[:5]  # 상위 5개
            
            logger.info(f"✅ 인사이트 생성 완료 - {analysis.total_insights}개 인사이트, "
                       f"품질점수: {analysis.overall_quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 인사이트 생성 중 오류: {e}")
            analysis.processing_notes.append(f"생성 오류: {str(e)}")
        
        return analysis
    
    def _extract_text_insights(self, 
                             text: str, 
                             source_agents: List[str]) -> List[Insight]:
        """텍스트에서 인사이트 추출"""
        
        insights = []
        
        try:
            # 1. 트렌드 인사이트
            trend_insights = self._identify_trends(text, source_agents)
            insights.extend(trend_insights)
            
            # 2. 패턴 인사이트
            pattern_insights = self._identify_patterns(text, source_agents)
            insights.extend(pattern_insights)
            
            # 3. 비교 인사이트
            comparison_insights = self._identify_comparisons(text, source_agents)
            insights.extend(comparison_insights)
            
            # 4. 통계적 인사이트
            statistical_insights = self._identify_statistical_insights(text, source_agents)
            insights.extend(statistical_insights)
            
        except Exception as e:
            logger.warning(f"텍스트 인사이트 추출 중 오류: {e}")
        
        return insights
    
    def _extract_artifact_insights(self, 
                                 artifacts: List[Dict[str, Any]], 
                                 source_agents: List[str]) -> List[Insight]:
        """아티팩트에서 인사이트 추출"""
        
        insights = []
        
        try:
            for artifact in artifacts:
                art_type = artifact.get('type', 'unknown')
                
                if art_type == 'plotly_chart':
                    chart_insights = self._analyze_chart_insights(artifact, source_agents)
                    insights.extend(chart_insights)
                
                elif art_type == 'dataframe':
                    df_insights = self._analyze_dataframe_insights(artifact, source_agents)
                    insights.extend(df_insights)
                
                elif art_type == 'statistical_summary':
                    stats_insights = self._analyze_statistical_summary_insights(artifact, source_agents)
                    insights.extend(stats_insights)
            
        except Exception as e:
            logger.warning(f"아티팩트 인사이트 추출 중 오류: {e}")
        
        return insights
    
    def _extract_cross_analysis_insights(self, 
                                       agent_results: Dict[str, AgentResult],
                                       integration_result: IntegrationResult) -> List[Insight]:
        """크로스 분석 인사이트 추출"""
        
        insights = []
        
        try:
            # 1. 에이전트 간 일관성 분석
            consistency_insights = self._analyze_agent_consistency(
                agent_results, integration_result.contributing_agents
            )
            insights.extend(consistency_insights)
            
            # 2. 데이터 품질 인사이트
            quality_insights = self._analyze_data_quality_insights(
                agent_results, integration_result.contributing_agents
            )
            insights.extend(quality_insights)
            
            # 3. 커버리지 분석
            coverage_insights = self._analyze_coverage_insights(
                agent_results, integration_result
            )
            insights.extend(coverage_insights)
            
        except Exception as e:
            logger.warning(f"크로스 분석 인사이트 추출 중 오류: {e}")
        
        return insights
    
    def _identify_trends(self, text: str, source_agents: List[str]) -> List[Insight]:
        """트렌드 식별"""
        
        insights = []
        text_lower = text.lower()
        
        try:
            for keyword in self.trend_keywords:
                if keyword in text_lower:
                    # 키워드 주변 컨텍스트 분석
                    trend_context = self._extract_context_around_keyword(text, keyword)
                    
                    if trend_context:
                        # 수치 정보 추출
                        numbers = re.findall(r'\d+(?:\.\d+)?%?', trend_context)
                        
                        confidence = 0.6
                        impact = 0.5
                        
                        # 수치가 있으면 신뢰도 향상
                        if numbers:
                            confidence += 0.2
                            impact += 0.2
                        
                        insight = Insight(
                            insight_id=f"trend_{len(insights)}_{keyword}",
                            title=f"{keyword.capitalize()} 트렌드 발견",
                            description=trend_context,
                            insight_type=InsightType.TREND,
                            priority=InsightPriority.MEDIUM,
                            confidence=min(1.0, confidence),
                            impact_score=min(1.0, impact),
                            supporting_data={"keyword": keyword, "numbers": numbers},
                            evidence=[trend_context],
                            source_agents=source_agents,
                            recommended_actions=[f"{keyword} 트렌드에 대한 추가 분석 수행"]
                        )
                        
                        insights.append(insight)
        
        except Exception as e:
            logger.warning(f"트렌드 식별 중 오류: {e}")
        
        return insights[:3]  # 최대 3개
    
    def _identify_patterns(self, text: str, source_agents: List[str]) -> List[Insight]:
        """패턴 식별"""
        
        insights = []
        
        try:
            # 반복되는 구조 패턴 찾기
            sentences = re.split(r'[.!?]+', text)
            sentence_patterns = defaultdict(list)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # 문장 구조 패턴 (간단한 방식)
                    words = sentence.lower().split()
                    if len(words) >= 3:
                        pattern = f"{words[0]}...{words[-1]}"
                        sentence_patterns[pattern].append(sentence)
            
            # 반복되는 패턴 식별
            for pattern, sentences in sentence_patterns.items():
                if len(sentences) >= 2:  # 2번 이상 반복
                    insight = Insight(
                        insight_id=f"pattern_{len(insights)}_{hash(pattern)}",
                        title=f"반복 패턴 발견: {pattern}",
                        description=f"다음 구조가 {len(sentences)}번 반복됨: {sentences[0][:100]}...",
                        insight_type=InsightType.PATTERN,
                        priority=InsightPriority.LOW,
                        confidence=0.5 + min(0.4, len(sentences) * 0.1),
                        impact_score=0.4,
                        supporting_data={"pattern": pattern, "occurrences": len(sentences)},
                        evidence=sentences[:3],  # 최대 3개 예시
                        source_agents=source_agents,
                        recommended_actions=["패턴의 원인과 의미 분석"]
                    )
                    
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"패턴 식별 중 오류: {e}")
        
        return insights[:2]  # 최대 2개
    
    def _identify_comparisons(self, text: str, source_agents: List[str]) -> List[Insight]:
        """비교 분석 식별"""
        
        insights = []
        
        try:
            for keyword in self.comparison_keywords:
                if keyword in text.lower():
                    comparison_context = self._extract_context_around_keyword(text, keyword)
                    
                    if comparison_context:
                        # 비교 대상과 수치 추출
                        numbers = re.findall(r'\d+(?:\.\d+)?%?', comparison_context)
                        
                        if len(numbers) >= 2:  # 비교할 수치가 있는 경우
                            try:
                                # 수치 간 차이 계산
                                nums = [float(n.replace('%', '')) for n in numbers[:2]]
                                difference = abs(nums[0] - nums[1])
                                
                                impact = min(1.0, difference / max(nums) if max(nums) > 0 else 0.5)
                                
                                insight = Insight(
                                    insight_id=f"comparison_{len(insights)}_{keyword}",
                                    title=f"유의미한 차이 발견: {keyword}",
                                    description=comparison_context,
                                    insight_type=InsightType.COMPARISON,
                                    priority=InsightPriority.MEDIUM if impact > 0.3 else InsightPriority.LOW,
                                    confidence=0.7,
                                    impact_score=impact,
                                    supporting_data={
                                        "keyword": keyword,
                                        "numbers": numbers,
                                        "difference": difference
                                    },
                                    evidence=[comparison_context],
                                    source_agents=source_agents,
                                    recommended_actions=["차이의 원인 분석", "비교 기준 재검토"]
                                )
                                
                                insights.append(insight)
                                
                            except ValueError:
                                continue
        
        except Exception as e:
            logger.warning(f"비교 분석 식별 중 오류: {e}")
        
        return insights[:3]  # 최대 3개
    
    def _identify_statistical_insights(self, text: str, source_agents: List[str]) -> List[Insight]:
        """통계적 인사이트 식별"""
        
        insights = []
        
        try:
            for pattern in self.statistical_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                if matches:
                    for match in matches[:2]:  # 최대 2개
                        try:
                            value = float(match) if isinstance(match, str) else float(match[0])
                            
                            # 통계적 의미 평가
                            stat_type = self._classify_statistical_value(pattern, value)
                            
                            insight = Insight(
                                insight_id=f"statistical_{len(insights)}_{stat_type}",
                                title=f"통계적 발견: {stat_type}",
                                description=f"{stat_type} 값: {value}",
                                insight_type=InsightType.STATISTICAL,
                                priority=InsightPriority.MEDIUM,
                                confidence=0.8,
                                impact_score=self._evaluate_statistical_impact(stat_type, value),
                                supporting_data={"type": stat_type, "value": value},
                                evidence=[f"{stat_type}: {value}"],
                                source_agents=source_agents,
                                recommended_actions=[f"{stat_type} 값의 의미와 원인 분석"]
                            )
                            
                            insights.append(insight)
                            
                        except ValueError:
                            continue
        
        except Exception as e:
            logger.warning(f"통계적 인사이트 식별 중 오류: {e}")
        
        return insights[:3]  # 최대 3개
    
    def _analyze_chart_insights(self, 
                              chart_artifact: Dict[str, Any], 
                              source_agents: List[str]) -> List[Insight]:
        """차트 아티팩트 분석"""
        
        insights = []
        
        try:
            chart_data = chart_artifact.get('data', {})
            chart_layout = chart_artifact.get('layout', {})
            
            # 차트 제목에서 인사이트 단서 추출
            title = chart_layout.get('title', {}).get('text', '')
            
            if title:
                insight = Insight(
                    insight_id=f"chart_insight_{hash(title)}",
                    title=f"시각화 인사이트: {title}",
                    description=f"차트 '{title}'에서 중요한 패턴이나 트렌드가 식별됨",
                    insight_type=InsightType.PATTERN,
                    priority=InsightPriority.MEDIUM,
                    confidence=0.6,
                    impact_score=0.7,
                    supporting_data={"chart_title": title, "chart_type": chart_artifact.get('type')},
                    evidence=[f"차트 제목: {title}"],
                    source_agents=source_agents,
                    recommended_actions=["차트 데이터의 상세 분석", "트렌드 지속성 검토"]
                )
                
                insights.append(insight)
            
            # 데이터 포인트 수 분석
            if isinstance(chart_data, list) and chart_data:
                data_points = len(chart_data[0].get('y', [])) if chart_data[0].get('y') else 0
                
                if data_points > 100:
                    insight = Insight(
                        insight_id=f"chart_volume_{data_points}",
                        title="대용량 데이터셋 분석",
                        description=f"차트에 {data_points}개의 데이터 포인트가 포함됨",
                        insight_type=InsightType.STATISTICAL,
                        priority=InsightPriority.LOW,
                        confidence=0.9,
                        impact_score=0.4,
                        supporting_data={"data_points": data_points},
                        evidence=[f"데이터 포인트 수: {data_points}"],
                        source_agents=source_agents,
                        recommended_actions=["데이터 샘플링 전략 검토", "성능 최적화 고려"]
                    )
                    
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"차트 인사이트 분석 중 오류: {e}")
        
        return insights
    
    def _analyze_dataframe_insights(self, 
                                  df_artifact: Dict[str, Any], 
                                  source_agents: List[str]) -> List[Insight]:
        """데이터프레임 아티팩트 분석"""
        
        insights = []
        
        try:
            df_data = df_artifact.get('data', [])
            df_columns = df_artifact.get('columns', [])
            
            if df_data and df_columns:
                row_count = len(df_data)
                col_count = len(df_columns)
                
                # 데이터 규모 인사이트
                if row_count > 1000:
                    priority = InsightPriority.MEDIUM if row_count > 10000 else InsightPriority.LOW
                    
                    insight = Insight(
                        insight_id=f"df_scale_{row_count}_{col_count}",
                        title=f"대규모 데이터셋: {row_count:,}행 x {col_count}열",
                        description=f"분석 대상 데이터가 {row_count:,}개 레코드, {col_count}개 필드로 구성됨",
                        insight_type=InsightType.STATISTICAL,
                        priority=priority,
                        confidence=1.0,
                        impact_score=min(1.0, row_count / 10000),
                        supporting_data={"rows": row_count, "columns": col_count},
                        evidence=[f"데이터 규모: {row_count:,} x {col_count}"],
                        source_agents=source_agents,
                        recommended_actions=["데이터 품질 검증", "처리 성능 최적화"]
                    )
                    
                    insights.append(insight)
                
                # 컬럼 타입 다양성 분석
                if col_count > 10:
                    insight = Insight(
                        insight_id=f"df_complexity_{col_count}",
                        title=f"다차원 데이터 분석: {col_count}개 변수",
                        description=f"다양한 {col_count}개 변수에 대한 종합적 분석 수행됨",
                        insight_type=InsightType.PATTERN,
                        priority=InsightPriority.MEDIUM,
                        confidence=0.8,
                        impact_score=0.6,
                        supporting_data={"column_count": col_count, "columns": df_columns},
                        evidence=[f"분석 변수: {', '.join(df_columns[:5])}..."],
                        source_agents=source_agents,
                        recommended_actions=["변수 간 상관관계 분석", "차원 축소 검토"]
                    )
                    
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"데이터프레임 인사이트 분석 중 오류: {e}")
        
        return insights
    
    def _analyze_statistical_summary_insights(self, 
                                            stats_artifact: Dict[str, Any], 
                                            source_agents: List[str]) -> List[Insight]:
        """통계 요약 아티팩트 분석"""
        
        insights = []
        
        try:
            stats_data = stats_artifact.get('content', {})
            
            # 평균, 표준편차 등에서 이상치 감지
            for metric, value in stats_data.items():
                if isinstance(value, (int, float)):
                    if metric.lower() in ['std', 'standard_deviation', '표준편차']:
                        if value > 0:  # 표준편차가 있는 경우
                            insight = Insight(
                                insight_id=f"stats_variability_{metric}",
                                title=f"데이터 변동성: {metric}",
                                description=f"{metric}: {value:.3f} - 데이터의 변동성 정도",
                                insight_type=InsightType.STATISTICAL,
                                priority=InsightPriority.LOW,
                                confidence=0.9,
                                impact_score=0.3,
                                supporting_data={metric: value},
                                evidence=[f"{metric}: {value:.3f}"],
                                source_agents=source_agents,
                                recommended_actions=["변동성 원인 분석", "이상치 검출"]
                            )
                            
                            insights.append(insight)
        
        except Exception as e:
            logger.warning(f"통계 요약 인사이트 분석 중 오류: {e}")
        
        return insights
    
    def _analyze_agent_consistency(self, 
                                 agent_results: Dict[str, AgentResult],
                                 contributing_agents: List[str]) -> List[Insight]:
        """에이전트 간 일관성 분석"""
        
        insights = []
        
        try:
            # 실행 시간 일관성
            execution_times = []
            for agent_id in contributing_agents:
                if agent_id in agent_results:
                    execution_times.append(agent_results[agent_id].execution_duration)
            
            if len(execution_times) >= 2:
                time_std = statistics.stdev(execution_times)
                time_mean = statistics.mean(execution_times)
                
                cv = time_std / time_mean if time_mean > 0 else 0  # 변동계수
                
                if cv > 0.5:  # 변동계수가 50% 이상
                    insight = Insight(
                        insight_id="agent_time_inconsistency",
                        title="에이전트 성능 편차 발견",
                        description=f"에이전트 간 실행 시간 편차가 큼 (변동계수: {cv:.2f})",
                        insight_type=InsightType.ANOMALY,
                        priority=InsightPriority.MEDIUM,
                        confidence=0.8,
                        impact_score=0.5,
                        supporting_data={
                            "execution_times": execution_times,
                            "coefficient_of_variation": cv
                        },
                        evidence=[f"실행 시간 범위: {min(execution_times):.1f}~{max(execution_times):.1f}초"],
                        source_agents=contributing_agents,
                        recommended_actions=["성능 병목 지점 분석", "에이전트 최적화"]
                    )
                    
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"에이전트 일관성 분석 중 오류: {e}")
        
        return insights
    
    def _analyze_data_quality_insights(self, 
                                     agent_results: Dict[str, AgentResult],
                                     contributing_agents: List[str]) -> List[Insight]:
        """데이터 품질 인사이트 분석"""
        
        insights = []
        
        try:
            # 아티팩트 품질 분석
            total_artifacts = 0
            valid_artifacts = 0
            
            for agent_id in contributing_agents:
                if agent_id in agent_results:
                    result = agent_results[agent_id]
                    total_artifacts += len(result.artifacts)
                    
                    for artifact in result.artifacts:
                        if artifact.get('content') or artifact.get('data'):
                            valid_artifacts += 1
            
            if total_artifacts > 0:
                quality_ratio = valid_artifacts / total_artifacts
                
                if quality_ratio < 0.8:  # 80% 미만
                    insight = Insight(
                        insight_id="data_quality_issue",
                        title="데이터 품질 이슈 감지",
                        description=f"아티팩트 중 {quality_ratio:.1%}만 유효한 데이터 포함",
                        insight_type=InsightType.ANOMALY,
                        priority=InsightPriority.HIGH,
                        confidence=0.9,
                        impact_score=1.0 - quality_ratio,
                        supporting_data={
                            "total_artifacts": total_artifacts,
                            "valid_artifacts": valid_artifacts,
                            "quality_ratio": quality_ratio
                        },
                        evidence=[f"유효 아티팩트: {valid_artifacts}/{total_artifacts}"],
                        source_agents=contributing_agents,
                        recommended_actions=["데이터 품질 검증 강화", "에이전트 오류 처리 개선"]
                    )
                    
                    insights.append(insight)
        
        except Exception as e:
            logger.warning(f"데이터 품질 인사이트 분석 중 오류: {e}")
        
        return insights
    
    def _analyze_coverage_insights(self, 
                                 agent_results: Dict[str, AgentResult],
                                 integration_result: IntegrationResult) -> List[Insight]:
        """커버리지 분석 인사이트"""
        
        insights = []
        
        try:
            total_agents = len(agent_results)
            contributing_agents = len(integration_result.contributing_agents)
            
            coverage_rate = contributing_agents / total_agents if total_agents > 0 else 0
            
            if coverage_rate < 0.8:  # 80% 미만
                insight = Insight(
                    insight_id="low_coverage",
                    title="분석 커버리지 부족",
                    description=f"전체 에이전트 중 {coverage_rate:.1%}만 분석에 기여",
                    insight_type=InsightType.BUSINESS_IMPACT,
                    priority=InsightPriority.HIGH,
                    confidence=1.0,
                    impact_score=1.0 - coverage_rate,
                    supporting_data={
                        "total_agents": total_agents,
                        "contributing_agents": contributing_agents,
                        "coverage_rate": coverage_rate
                    },
                    evidence=[f"기여 에이전트: {contributing_agents}/{total_agents}"],
                    source_agents=integration_result.contributing_agents,
                    recommended_actions=["비기여 에이전트 문제 해결", "분석 범위 확대"]
                )
                
                insights.append(insight)
        
        except Exception as e:
            logger.warning(f"커버리지 분석 인사이트 중 오류: {e}")
        
        return insights
    
    def _evaluate_and_filter_insights(self, insights: List[Insight]) -> List[Insight]:
        """인사이트 품질 평가 및 필터링"""
        
        qualified_insights = []
        
        for insight in insights:
            # 품질 점수 계산
            quality_score = (
                insight.confidence * 0.4 +
                insight.impact_score * 0.3 +
                (len(insight.evidence) / 5) * 0.2 +  # 최대 5개 증거
                (len(insight.supporting_data) / 10) * 0.1  # 최대 10개 지원 데이터
            )
            
            # 최소 품질 기준
            if quality_score >= 0.3:
                qualified_insights.append(insight)
        
        return qualified_insights
    
    def _prioritize_insights(self, insights: List[Insight]) -> List[Insight]:
        """인사이트 우선순위 결정"""
        
        # 우선순위 점수 계산
        priority_scores = {
            InsightPriority.CRITICAL: 5,
            InsightPriority.HIGH: 4,
            InsightPriority.MEDIUM: 3,
            InsightPriority.LOW: 2,
            InsightPriority.INFORMATIONAL: 1
        }
        
        def calculate_priority_score(insight: Insight) -> float:
            base_score = priority_scores.get(insight.priority, 1)
            
            # 신뢰도와 영향도로 가중
            weighted_score = base_score * (
                0.6 * insight.confidence +
                0.4 * insight.impact_score
            )
            
            return weighted_score
        
        # 우선순위 점수로 정렬
        prioritized = sorted(insights, key=calculate_priority_score, reverse=True)
        
        return prioritized
    
    def _aggregate_analysis(self, analysis: InsightAnalysis):
        """분석 결과 집계"""
        
        if not analysis.insights:
            return
        
        # 유형별/우선순위별 집계
        for insight in analysis.insights:
            analysis.insights_by_type[insight.insight_type] = \
                analysis.insights_by_type.get(insight.insight_type, 0) + 1
            
            analysis.insights_by_priority[insight.priority] = \
                analysis.insights_by_priority.get(insight.priority, 0) + 1
        
        # 평균 지표 계산
        confidences = [insight.confidence for insight in analysis.insights]
        impacts = [insight.impact_score for insight in analysis.insights]
        
        analysis.average_confidence = sum(confidences) / len(confidences)
        analysis.average_impact = sum(impacts) / len(impacts)
        
        # 전체 품질 점수
        analysis.overall_quality_score = (
            analysis.average_confidence * 0.5 +
            analysis.average_impact * 0.3 +
            min(1.0, analysis.total_insights / 10) * 0.2
        )
    
    def _generate_key_findings(self, insights: List[Insight]) -> List[str]:
        """주요 발견사항 생성"""
        
        key_findings = []
        
        try:
            # 1. 가장 높은 우선순위 인사이트들
            high_priority = [i for i in insights 
                           if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]]
            
            if high_priority:
                key_findings.append(f"🚨 {len(high_priority)}개의 중요한 발견사항이 있습니다")
            
            # 2. 인사이트 유형별 요약
            type_counts = {}
            for insight in insights:
                type_counts[insight.insight_type] = type_counts.get(insight.insight_type, 0) + 1
            
            if type_counts:
                most_common_type = max(type_counts.items(), key=lambda x: x[1])
                key_findings.append(f"📊 주요 분석 영역: {most_common_type[0].value} ({most_common_type[1]}건)")
            
            # 3. 평균 신뢰도가 높은 경우
            avg_confidence = sum(i.confidence for i in insights) / len(insights)
            if avg_confidence > 0.8:
                key_findings.append(f"✅ 높은 신뢰도의 분석 결과 (평균 {avg_confidence:.1%})")
            
            # 4. 특별한 패턴이나 이상치
            anomalies = [i for i in insights if i.insight_type == InsightType.ANOMALY]
            if anomalies:
                key_findings.append(f"⚠️ {len(anomalies)}개의 이상치 또는 예외 상황 감지")
        
        except Exception as e:
            logger.warning(f"주요 발견사항 생성 중 오류: {e}")
            key_findings.append("분석 결과를 바탕으로 여러 인사이트가 도출되었습니다")
        
        return key_findings[:5]  # 최대 5개
    
    def _extract_context_around_keyword(self, text: str, keyword: str, context_length: int = 150) -> str:
        """키워드 주변 컨텍스트 추출"""
        
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        idx = text_lower.find(keyword_lower)
        if idx == -1:
            return ""
        
        start = max(0, idx - context_length)
        end = min(len(text), idx + len(keyword) + context_length)
        
        return text[start:end].strip()
    
    def _classify_statistical_value(self, pattern: str, value: float) -> str:
        """통계 값 분류"""
        
        if '%' in pattern:
            return "백분율"
        elif '평균' in pattern:
            return "평균값"
        elif '최대' in pattern:
            return "최대값"
        elif '최소' in pattern:
            return "최소값"
        elif '표준편차' in pattern:
            return "표준편차"
        elif '배' in pattern:
            return "배수관계"
        else:
            return "수치값"
    
    def _evaluate_statistical_impact(self, stat_type: str, value: float) -> float:
        """통계 값의 영향도 평가"""
        
        if stat_type == "백분율":
            if value > 90 or value < 10:
                return 0.8  # 극값
            elif value > 70 or value < 30:
                return 0.6  # 높은 편향
            else:
                return 0.4
        
        elif stat_type == "배수관계":
            if value > 5:
                return 0.9  # 5배 이상 차이
            elif value > 2:
                return 0.7  # 2배 이상 차이
            else:
                return 0.4
        
        elif stat_type == "표준편차":
            # 상대적 평가는 어려우므로 중간 값
            return 0.5
        
        else:
            return 0.5  # 기본값