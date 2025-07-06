"""
Holistic Answer Synthesis Engine

이 모듈은 Phase 1과 Phase 2의 모든 결과를 종합하여 
전체론적이고 완전한 답변을 생성하는 핵심 엔진입니다.

주요 기능:
- 다중 소스 정보 통합 및 합성
- 도메인별 전문가 수준 답변 생성
- 구조화된 답변 포맷 제공
- 사용자 의도에 맞는 답변 스타일 적응
- 품질 보장 및 일관성 검증
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from core.llm_factory import create_llm_instance

# Phase 1 imports
from .intelligent_query_processor import EnhancedQuery
from .domain_extractor import EnhancedDomainKnowledge
from .answer_predictor import AnswerTemplate

# Phase 2 imports
from .domain_aware_agent_selector import AgentSelectionResult
from .a2a_agent_execution_orchestrator import ExecutionResult
from .multi_agent_result_integration import IntegrationResult
from .execution_plan_manager import ManagedExecutionPlan

logger = logging.getLogger(__name__)


class AnswerStyle(Enum):
    """답변 스타일"""
    TECHNICAL = "technical"          # 기술적 상세 답변
    BUSINESS = "business"            # 비즈니스 중심 답변
    EXECUTIVE = "executive"          # 임원급 요약 답변
    OPERATIONAL = "operational"      # 운영 실행 중심 답변
    EDUCATIONAL = "educational"      # 교육적 설명 답변
    COMPREHENSIVE = "comprehensive"  # 종합적 상세 답변


class AnswerPriority(Enum):
    """답변 우선순위"""
    INSIGHTS = "insights"           # 인사이트 중심
    ACTIONS = "actions"            # 실행 방안 중심
    ANALYSIS = "analysis"          # 분석 결과 중심
    RECOMMENDATIONS = "recommendations"  # 권고사항 중심
    SOLUTIONS = "solutions"        # 해결책 중심


class SynthesisStrategy(Enum):
    """합성 전략"""
    LAYERED = "layered"            # 계층적 합성
    INTEGRATED = "integrated"      # 통합적 합성
    NARRATIVE = "narrative"        # 서사적 합성
    ANALYTICAL = "analytical"      # 분석적 합성


@dataclass
class AnswerSection:
    """답변 섹션"""
    title: str
    content: str
    priority: int
    section_type: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisContext:
    """합성 컨텍스트"""
    user_intent: str
    domain_context: str
    urgency_level: str
    target_audience: str
    answer_style: AnswerStyle
    answer_priority: AnswerPriority
    synthesis_strategy: SynthesisStrategy
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


@dataclass
class HolisticAnswer:
    """전체론적 답변"""
    answer_id: str
    query_summary: str
    executive_summary: str
    main_sections: List[AnswerSection]
    key_insights: List[str]
    recommendations: List[str]
    next_steps: List[str]
    confidence_score: float
    quality_metrics: Dict[str, float]
    synthesis_metadata: Dict[str, Any]
    generated_at: datetime
    synthesis_time: float


class HolisticAnswerSynthesisEngine:
    """전체론적 답변 합성 엔진"""
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.synthesis_history: List[HolisticAnswer] = []
        
        # 기본 설정
        self.default_style = AnswerStyle.COMPREHENSIVE
        self.default_priority = AnswerPriority.INSIGHTS
        self.default_strategy = SynthesisStrategy.INTEGRATED
        
        logger.info("HolisticAnswerSynthesisEngine initialized")
    
    async def synthesize_holistic_answer(
        self,
        # Phase 1 결과들
        enhanced_query: EnhancedQuery,
        domain_knowledge: EnhancedDomainKnowledge,
        answer_template: AnswerTemplate,
        
        # Phase 2 결과들
        agent_selection_result: AgentSelectionResult,
        execution_result: ExecutionResult,
        integration_result: IntegrationResult,
        managed_plan: Optional[ManagedExecutionPlan] = None,
        
        # 합성 설정
        synthesis_context: Optional[SynthesisContext] = None
    ) -> HolisticAnswer:
        """
        전체론적 답변 합성
        
        Args:
            enhanced_query: Phase 1 향상된 쿼리
            domain_knowledge: Phase 1 도메인 지식
            answer_template: Phase 1 답변 템플릿
            agent_selection_result: Phase 2 에이전트 선택 결과
            execution_result: Phase 2 실행 결과
            integration_result: Phase 2 통합 결과
            managed_plan: Phase 2 관리된 계획 (선택사항)
            synthesis_context: 합성 컨텍스트 (선택사항)
            
        Returns:
            HolisticAnswer: 전체론적 답변
        """
        start_time = time.time()
        answer_id = f"holistic_{int(start_time)}"
        
        logger.info(f"🔄 Starting holistic answer synthesis: {answer_id}")
        
        try:
            # 1. 합성 컨텍스트 준비
            if synthesis_context is None:
                synthesis_context = await self._create_synthesis_context(
                    enhanced_query, domain_knowledge, integration_result
                )
            
            # 2. 종합 분석 수행
            comprehensive_analysis = await self._perform_comprehensive_analysis(
                enhanced_query, domain_knowledge, answer_template,
                agent_selection_result, execution_result, integration_result,
                synthesis_context
            )
            
            # 3. 답변 섹션 생성
            answer_sections = await self._generate_answer_sections(
                comprehensive_analysis, synthesis_context
            )
            
            # 4. 핵심 인사이트 추출
            key_insights = await self._extract_key_insights(
                comprehensive_analysis, integration_result, synthesis_context
            )
            
            # 5. 실행 권고사항 생성
            recommendations = await self._generate_actionable_recommendations(
                comprehensive_analysis, integration_result, synthesis_context
            )
            
            # 6. 다음 단계 제안
            next_steps = await self._suggest_next_steps(
                recommendations, execution_result, synthesis_context
            )
            
            # 7. 임원 요약 생성
            executive_summary = await self._generate_executive_summary(
                comprehensive_analysis, key_insights, recommendations, synthesis_context
            )
            
            # 8. 품질 메트릭 계산
            quality_metrics = await self._calculate_quality_metrics(
                answer_sections, key_insights, recommendations, integration_result
            )
            
            # 9. 전체 신뢰도 계산
            confidence_score = self._calculate_overall_confidence(
                quality_metrics, integration_result, execution_result
            )
            
            synthesis_time = time.time() - start_time
            
            # 10. 전체론적 답변 구성
            holistic_answer = HolisticAnswer(
                answer_id=answer_id,
                query_summary=(enhanced_query.enhanced_queries[0][:200] + "..." if enhanced_query.enhanced_queries else enhanced_query.original_query[:200] + "..."),
                executive_summary=executive_summary,
                main_sections=answer_sections,
                key_insights=key_insights,
                recommendations=recommendations,
                next_steps=next_steps,
                confidence_score=confidence_score,
                quality_metrics=quality_metrics,
                synthesis_metadata={
                    "synthesis_context": synthesis_context.__dict__,
                    "source_confidence": {
                        "query_processing": getattr(enhanced_query, 'confidence_score', 0.8),
                        "agent_selection": agent_selection_result.total_confidence,
                        "execution": execution_result.confidence_score,
                        "integration": integration_result.confidence_score
                    },
                    "analysis_depth": len(answer_sections),
                    "insight_count": len(key_insights),
                    "recommendation_count": len(recommendations)
                },
                generated_at=datetime.now(),
                synthesis_time=synthesis_time
            )
            
            self.synthesis_history.append(holistic_answer)
            
            logger.info(f"✅ Holistic answer synthesis completed: {answer_id} ({synthesis_time:.2f}s)")
            return holistic_answer
            
        except Exception as e:
            logger.error(f"❌ Holistic answer synthesis failed: {answer_id} - {e}")
            raise
    
    async def _create_synthesis_context(
        self,
        enhanced_query: EnhancedQuery,
        domain_knowledge: EnhancedDomainKnowledge,
        integration_result: IntegrationResult
    ) -> SynthesisContext:
        """합성 컨텍스트 생성"""
        
        context_prompt = f"""다음 정보를 바탕으로 답변 합성을 위한 최적의 컨텍스트를 결정해주세요:

**향상된 쿼리**:
- 원본 쿼리: {enhanced_query.original_query}
- 향상된 쿼리: {enhanced_query.enhanced_queries[0][:300] + '...' if enhanced_query.enhanced_queries else enhanced_query.original_query}
- 의도 분석: {enhanced_query.intent_analysis.primary_intent if enhanced_query.intent_analysis else 'N/A'}

**도메인 지식**:
- 주요 도메인: {domain_knowledge.taxonomy.primary_domain.value}
- 기술 영역: {domain_knowledge.taxonomy.technical_area}
- 주요 개념: {', '.join(list(domain_knowledge.key_concepts.keys())[:3])}

**통합 결과**:
- 신뢰도: {integration_result.confidence_score:.2f}
- 인사이트 수: {len(integration_result.integrated_insights)}
- 권고사항 수: {len(integration_result.recommendations)}

**컨텍스트 결정 요구사항**:
1. 사용자 의도 (user_intent)
2. 도메인 컨텍스트 (domain_context)
3. 긴급도 수준 (urgency_level: low/medium/high)
4. 대상 청중 (target_audience: technical/business/mixed)
5. 답변 스타일 (answer_style: technical/business/executive/operational/educational/comprehensive)
6. 답변 우선순위 (answer_priority: insights/actions/analysis/recommendations/solutions)
7. 합성 전략 (synthesis_strategy: layered/integrated/narrative/analytical)

다음 JSON 형식으로 응답해주세요:
{{
  "user_intent": "반도체 공정 이상 분석 및 기술적 조치 방향 도출",
  "domain_context": "semiconductor_manufacturing_quality_control",
  "urgency_level": "high",
  "target_audience": "technical",
  "answer_style": "comprehensive",
  "answer_priority": "actions",
  "synthesis_strategy": "integrated",
  "quality_requirements": {{"completeness": 0.9, "accuracy": 0.95, "actionability": 0.85}},
  "constraints": ["실행 가능한 조치 중심", "기술적 정확성 필수"]
}}"""

        try:
            response = await self.llm.ainvoke(context_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                context_data = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    context_data = json.loads(json_match.group(1))
                else:
                    context_data = {}
            
            # SynthesisContext 객체 생성
            synthesis_context = SynthesisContext(
                user_intent=context_data.get("user_intent", "정보 분석 및 인사이트 도출"),
                domain_context=context_data.get("domain_context", domain_knowledge.taxonomy.primary_domain.value),
                urgency_level=context_data.get("urgency_level", "medium"),
                target_audience=context_data.get("target_audience", "mixed"),
                answer_style=AnswerStyle(context_data.get("answer_style", "comprehensive")),
                answer_priority=AnswerPriority(context_data.get("answer_priority", "insights")),
                synthesis_strategy=SynthesisStrategy(context_data.get("synthesis_strategy", "integrated")),
                quality_requirements=context_data.get("quality_requirements", {"completeness": 0.8, "accuracy": 0.9}),
                constraints=context_data.get("constraints", [])
            )
            
            return synthesis_context
            
        except Exception as e:
            logger.warning(f"Synthesis context creation failed, using defaults: {e}")
            return SynthesisContext(
                user_intent="정보 분석 및 인사이트 도출",
                domain_context=domain_knowledge.taxonomy.primary_domain.value,
                urgency_level="medium",
                target_audience="mixed",
                answer_style=self.default_style,
                answer_priority=self.default_priority,
                synthesis_strategy=self.default_strategy
            )
    
    async def _perform_comprehensive_analysis(
        self,
        enhanced_query: EnhancedQuery,
        domain_knowledge: EnhancedDomainKnowledge,
        answer_template: AnswerTemplate,
        agent_selection_result: AgentSelectionResult,
        execution_result: ExecutionResult,
        integration_result: IntegrationResult,
        synthesis_context: SynthesisContext
    ) -> Dict[str, Any]:
        """종합 분석 수행"""
        
        analysis_prompt = f"""다음 모든 정보를 종합하여 전문가 수준의 포괄적 분석을 수행해주세요:

**1. 쿼리 분석**:
- 원본 의도: {enhanced_query.original_query}
- 향상된 분석: {enhanced_query.enhanced_queries[0][:400] + '...' if enhanced_query.enhanced_queries else enhanced_query.original_query}
- 신뢰도: {getattr(enhanced_query, 'confidence_score', 0.8):.2f}

**2. 도메인 지식**:
- 도메인: {domain_knowledge.taxonomy.primary_domain.value}
- 기술 영역: {domain_knowledge.taxonomy.technical_area}
- 핵심 개념: {json.dumps(list(domain_knowledge.key_concepts.keys())[:5], ensure_ascii=False)}

**3. 에이전트 선택 및 실행**:
- 선택된 에이전트: {len(agent_selection_result.selected_agents)}개
- 실행 성공률: {execution_result.completed_tasks}/{execution_result.total_tasks}
- 실행 시간: {execution_result.execution_time:.2f}초

**4. 통합 결과**:
- 통합 인사이트: {len(integration_result.integrated_insights)}개
- 권고사항: {len(integration_result.recommendations)}개
- 신뢰도: {integration_result.confidence_score:.2f}

**5. 합성 컨텍스트**:
- 사용자 의도: {synthesis_context.user_intent}
- 답변 스타일: {synthesis_context.answer_style.value}
- 우선순위: {synthesis_context.answer_priority.value}

**종합 분석 요구사항**:
1. 전체 상황 요약 및 문제 정의
2. 수집된 데이터 및 분석 결과 통합
3. 핵심 발견사항 및 패턴 식별
4. 리스크 및 기회 요인 분석
5. 전문가적 해석 및 시사점
6. 실행 가능성 및 우선순위 평가

다음 JSON 형식으로 응답해주세요:
{{
  "situation_summary": "현재 상황에 대한 명확한 요약",
  "problem_definition": "해결해야 할 핵심 문제들",
  "data_integration": "수집된 모든 데이터의 통합 분석",
  "key_findings": ["발견사항 1", "발견사항 2", "발견사항 3"],
  "patterns_identified": ["패턴 1", "패턴 2"],
  "risk_analysis": {{"risks": ["리스크 1", "리스크 2"], "opportunities": ["기회 1", "기회 2"]}},
  "expert_interpretation": "전문가적 해석 및 시사점",
  "feasibility_assessment": "실행 가능성 평가",
  "priority_matrix": {{"high": ["우선순위 높음"], "medium": ["보통"], "low": ["낮음"]}}
}}"""

        try:
            response = await self.llm.ainvoke(analysis_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                analysis_result = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group(1))
                else:
                    analysis_result = {"error": "Failed to parse analysis"}
            
            return analysis_result
            
        except Exception as e:
            logger.warning(f"Comprehensive analysis failed: {e}")
            return {
                "situation_summary": "종합 분석 결과를 생성하는 중 오류가 발생했습니다.",
                "problem_definition": "문제 정의 실패",
                "data_integration": "데이터 통합 분석 실패",
                "key_findings": ["분석 결과 생성 실패"],
                "patterns_identified": [],
                "risk_analysis": {"risks": [], "opportunities": []},
                "expert_interpretation": "전문가 해석 생성 실패",
                "feasibility_assessment": "실행 가능성 평가 실패",
                "priority_matrix": {"high": [], "medium": [], "low": []}
            }
    
    async def _generate_answer_sections(
        self,
        comprehensive_analysis: Dict[str, Any],
        synthesis_context: SynthesisContext
    ) -> List[AnswerSection]:
        """답변 섹션 생성"""
        
        sections_prompt = f"""다음 종합 분석 결과를 바탕으로 체계적인 답변 섹션들을 생성해주세요:

**종합 분석 결과**:
{json.dumps(comprehensive_analysis, ensure_ascii=False, indent=2)}

**답변 스타일**: {synthesis_context.answer_style.value}
**우선순위**: {synthesis_context.answer_priority.value}
**대상 청중**: {synthesis_context.target_audience}

**섹션 생성 요구사항**:
1. 사용자 의도에 맞는 논리적 구조
2. 각 섹션별 명확한 목적과 내용
3. 우선순위에 따른 섹션 순서
4. 대상 청중에 적합한 상세도

다음 JSON 형식으로 응답해주세요:
{{
  "sections": [
    {{
      "title": "상황 분석 및 문제 정의",
      "content": "상세한 섹션 내용...",
      "priority": 1,
      "section_type": "analysis",
      "confidence": 0.9,
      "sources": ["comprehensive_analysis"]
    }}
  ]
}}"""

        try:
            response = await self.llm.ainvoke(sections_prompt)
            content = response.content.strip()
            
            # JSON 추출
            if content.startswith('{') and content.endswith('}'):
                sections_data = json.loads(content)
            else:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    sections_data = json.loads(json_match.group(1))
                else:
                    sections_data = {"sections": []}
            
            # AnswerSection 객체로 변환
            answer_sections = []
            for section_data in sections_data.get("sections", []):
                section = AnswerSection(
                    title=section_data.get("title", "제목 없음"),
                    content=section_data.get("content", "내용 없음"),
                    priority=section_data.get("priority", 5),
                    section_type=section_data.get("section_type", "general"),
                    confidence=section_data.get("confidence", 0.5),
                    sources=section_data.get("sources", []),
                    metadata={"generated_by": "holistic_synthesis"}
                )
                answer_sections.append(section)
            
            # 우선순위별 정렬
            answer_sections.sort(key=lambda x: x.priority)
            
            return answer_sections
            
        except Exception as e:
            logger.warning(f"Answer sections generation failed: {e}")
            return [
                AnswerSection(
                    title="분석 결과 요약",
                    content="답변 섹션 생성 중 오류가 발생했습니다.",
                    priority=1,
                    section_type="error",
                    confidence=0.0,
                    sources=["error_fallback"]
                )
            ]
    
    async def _extract_key_insights(
        self,
        comprehensive_analysis: Dict[str, Any],
        integration_result: IntegrationResult,
        synthesis_context: SynthesisContext
    ) -> List[str]:
        """핵심 인사이트 추출"""
        
        insights_prompt = f"""다음 정보들을 종합하여 가장 중요한 핵심 인사이트들을 추출해주세요:

**종합 분석**:
- 핵심 발견사항: {comprehensive_analysis.get('key_findings', [])}
- 식별된 패턴: {comprehensive_analysis.get('patterns_identified', [])}
- 전문가 해석: {comprehensive_analysis.get('expert_interpretation', '')}

**통합 인사이트**:
{json.dumps([{
    'type': insight.insight_type,
    'content': insight.content,
    'confidence': insight.confidence
} for insight in integration_result.integrated_insights], ensure_ascii=False, indent=2)}

**사용자 의도**: {synthesis_context.user_intent}
**우선순위**: {synthesis_context.answer_priority.value}

**인사이트 추출 요구사항**:
1. 실행 가능하고 구체적인 인사이트
2. 사용자 의도와 직접적 연관성
3. 높은 영향도와 실용성
4. 명확하고 이해하기 쉬운 표현

핵심 인사이트 3-7개를 생성해주세요. 각각 한 문장으로 명확하게 표현하고, 리스트 형태로 응답해주세요."""

        try:
            response = await self.llm.ainvoke(insights_prompt)
            content = response.content.strip()
            
            # 리스트 형태로 파싱
            insights = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    insights.append(line[1:].strip())
                elif line and any(char.isdigit() for char in line[:3]):
                    # 번호가 있는 항목
                    insights.append(line.split('.', 1)[-1].strip())
            
            return insights if insights else [content]
            
        except Exception as e:
            logger.warning(f"Key insights extraction failed: {e}")
            return ["핵심 인사이트 추출 중 오류가 발생했습니다."]
    
    async def _generate_actionable_recommendations(
        self,
        comprehensive_analysis: Dict[str, Any],
        integration_result: IntegrationResult,
        synthesis_context: SynthesisContext
    ) -> List[str]:
        """실행 가능한 권고사항 생성"""
        
        recommendations_prompt = f"""다음 분석 결과를 바탕으로 실행 가능한 구체적 권고사항을 생성해주세요:

**종합 분석**:
- 우선순위 매트릭스: {comprehensive_analysis.get('priority_matrix', {})}
- 실행 가능성 평가: {comprehensive_analysis.get('feasibility_assessment', '')}
- 리스크 분석: {comprehensive_analysis.get('risk_analysis', {})}

**기존 권고사항**:
{json.dumps(integration_result.recommendations, ensure_ascii=False, indent=2)}

**컨텍스트**:
- 긴급도: {synthesis_context.urgency_level}
- 제약사항: {synthesis_context.constraints}
- 품질 요구사항: {synthesis_context.quality_requirements}

**권고사항 생성 원칙**:
1. 즉시 실행 가능한 조치
2. 명확한 실행 주체 및 방법
3. 측정 가능한 결과 지표
4. 리스크 최소화 방안 포함
5. 우선순위별 단계적 접근

실행 가능한 권고사항 5-10개를 생성해주세요."""

        try:
            response = await self.llm.ainvoke(recommendations_prompt)
            content = response.content.strip()
            
            # 리스트 형태로 파싱
            recommendations = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    recommendations.append(line[1:].strip())
                elif line and any(char.isdigit() for char in line[:3]):
                    recommendations.append(line.split('.', 1)[-1].strip())
            
            return recommendations if recommendations else [content]
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            return ["권고사항 생성 중 오류가 발생했습니다."]
    
    async def _suggest_next_steps(
        self,
        recommendations: List[str],
        execution_result: ExecutionResult,
        synthesis_context: SynthesisContext
    ) -> List[str]:
        """다음 단계 제안"""
        
        next_steps_prompt = f"""다음 권고사항과 실행 결과를 바탕으로 구체적인 다음 단계를 제안해주세요:

**생성된 권고사항**:
{json.dumps(recommendations, ensure_ascii=False, indent=2)}

**실행 결과 요약**:
- 완료된 분석: {execution_result.completed_tasks}/{execution_result.total_tasks}
- 실행 시간: {execution_result.execution_time:.2f}초
- 신뢰도: {execution_result.confidence_score:.2f}

**컨텍스트**:
- 긴급도: {synthesis_context.urgency_level}
- 대상 청중: {synthesis_context.target_audience}

**다음 단계 요구사항**:
1. 시간순서에 따른 구체적 액션
2. 담당자 및 리소스 요구사항 명시
3. 예상 소요 시간 및 마일스톤
4. 성공 측정 기준
5. 리스크 대응 방안

즉시 실행 가능한 다음 단계 3-6개를 제안해주세요."""

        try:
            response = await self.llm.ainvoke(next_steps_prompt)
            content = response.content.strip()
            
            # 리스트 형태로 파싱
            next_steps = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    next_steps.append(line[1:].strip())
                elif line and any(char.isdigit() for char in line[:3]):
                    next_steps.append(line.split('.', 1)[-1].strip())
            
            return next_steps if next_steps else [content]
            
        except Exception as e:
            logger.warning(f"Next steps suggestion failed: {e}")
            return ["다음 단계 제안 생성 중 오류가 발생했습니다."]
    
    async def _generate_executive_summary(
        self,
        comprehensive_analysis: Dict[str, Any],
        key_insights: List[str],
        recommendations: List[str],
        synthesis_context: SynthesisContext
    ) -> str:
        """임원 요약 생성"""
        
        summary_prompt = f"""다음 모든 정보를 바탕으로 임원진을 위한 핵심 요약을 작성해주세요:

**상황 요약**: {comprehensive_analysis.get('situation_summary', '')}
**핵심 인사이트**: {key_insights[:3]}
**주요 권고사항**: {recommendations[:3]}
**대상 청중**: {synthesis_context.target_audience}
**긴급도**: {synthesis_context.urgency_level}

**임원 요약 요구사항**:
1. 3-4문장으로 핵심만 간결하게
2. 비즈니스 영향도 중심
3. 의사결정에 필요한 핵심 정보
4. 실행 시급성 및 우선순위 명시

전문적이고 간결한 임원 요약을 작성해주세요."""

        try:
            response = await self.llm.ainvoke(summary_prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return "임원 요약 생성 중 오류가 발생했습니다."
    
    async def _calculate_quality_metrics(
        self,
        answer_sections: List[AnswerSection],
        key_insights: List[str],
        recommendations: List[str],
        integration_result: IntegrationResult
    ) -> Dict[str, float]:
        """품질 메트릭 계산"""
        
        # 완성도: 섹션 수와 내용 길이 기반
        completeness = min(1.0, len(answer_sections) / 5.0) * 0.5 + \
                      min(1.0, sum(len(section.content) for section in answer_sections) / 2000) * 0.5
        
        # 일관성: 섹션 간 신뢰도 편차 기반
        if answer_sections:
            section_confidences = [section.confidence for section in answer_sections]
            consistency = 1.0 - (max(section_confidences) - min(section_confidences))
        else:
            consistency = 0.0
        
        # 실행가능성: 권고사항 수와 인사이트 수 기반
        actionability = min(1.0, len(recommendations) / 8.0) * 0.7 + \
                       min(1.0, len(key_insights) / 6.0) * 0.3
        
        # 관련성: 통합 결과 신뢰도 활용
        relevance = integration_result.confidence_score
        
        # 명확성: 섹션별 평균 신뢰도
        if answer_sections:
            clarity = sum(section.confidence for section in answer_sections) / len(answer_sections)
        else:
            clarity = 0.0
        
        # 포괄성: 다양한 측면의 커버리지
        comprehensiveness = min(1.0, len(answer_sections) / 6.0) * 0.4 + \
                           min(1.0, len(key_insights) / 5.0) * 0.3 + \
                           min(1.0, len(recommendations) / 7.0) * 0.3
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "actionability": actionability,
            "relevance": relevance,
            "clarity": clarity,
            "comprehensiveness": comprehensiveness
        }
    
    def _calculate_overall_confidence(
        self,
        quality_metrics: Dict[str, float],
        integration_result: IntegrationResult,
        execution_result: ExecutionResult
    ) -> float:
        """전체 신뢰도 계산"""
        
        # 품질 메트릭 평균
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        # 소스 신뢰도 가중 평균
        source_confidence = (
            integration_result.confidence_score * 0.4 +
            execution_result.confidence_score * 0.3 +
            quality_score * 0.3
        )
        
        # 실행 성공률 보정
        execution_success_rate = execution_result.completed_tasks / max(1, execution_result.total_tasks)
        
        # 최종 신뢰도 계산
        final_confidence = source_confidence * 0.8 + execution_success_rate * 0.2
        
        return max(0.0, min(1.0, final_confidence))
    
    async def get_synthesis_history(self) -> List[HolisticAnswer]:
        """합성 이력 조회"""
        return self.synthesis_history.copy()
    
    async def get_answer_summary(self, answer_id: str) -> Optional[Dict[str, Any]]:
        """답변 요약 조회"""
        for answer in self.synthesis_history:
            if answer.answer_id == answer_id:
                return {
                    "answer_id": answer.answer_id,
                    "query_summary": answer.query_summary,
                    "confidence_score": answer.confidence_score,
                    "section_count": len(answer.main_sections),
                    "insight_count": len(answer.key_insights),
                    "recommendation_count": len(answer.recommendations),
                    "synthesis_time": answer.synthesis_time,
                    "generated_at": answer.generated_at.isoformat()
                }
        return None 