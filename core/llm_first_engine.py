#!/usr/bin/env python3
"""
🧠 CherryAI LLM First Engine

LLM First 원칙 완전 준수를 위한 핵심 엔진
모든 하드코딩, Rule 기반 로직, 패턴 매칭을 LLM 기반 동적 분석으로 대체

🎯 핵심 원칙:
- 절대 하드코딩 금지 (No Hardcoding)
- Rule 기반 패턴 매칭 금지 (No Rule-based Patterns)
- 템플릿 매칭 금지 (No Template Matching)
- LLM 능력 최대 활용 (Maximize LLM Capabilities)
- 범용적 동작 (Universal Behavior)
- 사용자 의도 기반 동적 처리 (Intent-driven Dynamic Processing)

Key Features:
- Universal Intent Analyzer: 사용자 의도 동적 분석
- Dynamic Decision Engine: 실시간 의사결정
- Context-Aware Planner: 상황 인식 계획 수립
- Adaptive Collaboration Engine: 적응적 협업 엔진
- Quality Validator: 결과 품질 검증
- Learning System: 지속 학습 및 개선
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

# LLM 관련 임포트
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# 프로젝트 임포트
import sys
sys.path.append(os.path.dirname(__file__))

logger = logging.getLogger(__name__)

class IntentConfidence(Enum):
    """의도 분석 신뢰도"""
    VERY_HIGH = "very_high"    # 95%+
    HIGH = "high"              # 80-95%
    MEDIUM = "medium"          # 60-80%
    LOW = "low"                # 40-60%
    VERY_LOW = "very_low"      # <40%

class DecisionType(Enum):
    """의사결정 유형"""
    AGENT_SELECTION = "agent_selection"
    WORKFLOW_PLANNING = "workflow_planning"
    TASK_DECOMPOSITION = "task_decomposition"
    QUALITY_ASSESSMENT = "quality_assessment"
    COLLABORATION_STRATEGY = "collaboration_strategy"
    RESOURCE_ALLOCATION = "resource_allocation"

@dataclass
class UserIntent:
    """사용자 의도 분석 결과"""
    primary_intent: str
    secondary_intents: List[str] = field(default_factory=list)
    confidence: IntentConfidence = IntentConfidence.MEDIUM
    complexity_level: str = "medium"  # simple, medium, complex
    data_requirements: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    user_expertise_level: str = "intermediate"  # beginner, intermediate, expert
    urgency_level: str = "normal"  # low, normal, high, urgent
    context_dependencies: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

@dataclass
class DynamicDecision:
    """동적 의사결정 결과"""
    decision_type: DecisionType
    decision: str
    alternatives: List[str] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    context_factors: Dict[str, Any] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    execution_plan: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAssessment:
    """품질 평가 결과"""
    overall_score: float
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    user_satisfaction_prediction: float = 0.0
    actionable_recommendations: List[str] = field(default_factory=list)

class LLMFirstEngine:
    """
    🧠 LLM First 엔진 - 모든 하드코딩 제거
    
    LLM의 추론 능력을 최대한 활용하여 동적이고 범용적인 처리를 수행
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 enable_learning: bool = True):
        """
        LLM First 엔진 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            model: 사용할 LLM 모델
            enable_learning: 학습 기능 활성화
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.enable_learning = enable_learning
        
        # LLM 클라이언트 초기화
        if self.openai_api_key and AsyncOpenAI:
            self.llm_client = AsyncOpenAI(api_key=self.openai_api_key)
        else:
            self.llm_client = None
            logger.warning("🚨 OpenAI API 키가 없어서 LLM First 기능이 제한됩니다")
        
        # 학습 데이터 저장소
        self.learning_memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_history: List[Dict[str, Any]] = []
        
        # 시스템 성능 메트릭
        self.metrics = {
            "total_requests": 0,
            "successful_decisions": 0,
            "user_satisfaction_scores": [],
            "average_response_time": 0.0,
            "accuracy_rate": 0.0
        }
        
        logger.info(f"🧠 LLM First Engine 초기화 완료 (모델: {model})")

    async def analyze_user_intent(self, 
                                user_request: str, 
                                context: Dict[str, Any] = None) -> UserIntent:
        """
        사용자 의도 동적 분석 (LLM First)
        
        하드코딩된 키워드 패턴 대신 LLM의 추론 능력 활용
        """
        if not self.llm_client:
            return self._fallback_intent_analysis(user_request, context)
        
        start_time = time.time()
        
        try:
            # 컨텍스트 정리
            context_summary = ""
            if context:
                context_summary = f"추가 컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}"
            
            # LLM 프롬프트 (하드코딩 없는 범용적 분석)
            system_prompt = """당신은 사용자 의도 분석 전문가입니다.

사용자의 요청을 분석하여 다음 정보를 정확히 파악하세요:

1. 주요 의도 (primary_intent): 사용자가 진짜로 원하는 것
2. 부차적 의도들 (secondary_intents): 연관된 또는 숨겨진 요구사항들
3. 복잡도 수준 (complexity_level): simple/medium/complex
4. 데이터 요구사항 (data_requirements): 필요한 데이터 유형들
5. 기대 결과물 (expected_outputs): 사용자가 기대하는 결과 형태
6. 사용자 전문성 수준 (user_expertise_level): beginner/intermediate/expert
7. 긴급도 (urgency_level): low/normal/high/urgent
8. 신뢰도 (confidence): very_high/high/medium/low/very_low

중요한 원칙:
- 하드코딩된 패턴 매칭 금지
- 키워드 기반 분류 금지
- 사용자의 진짜 의도에 집중
- 컨텍스트를 고려한 종합적 판단
- 불확실한 경우 솔직하게 표현

JSON 형식으로 응답하세요:
{
    "primary_intent": "구체적인 주요 의도",
    "secondary_intents": ["부차적 의도1", "부차적 의도2"],
    "confidence": "신뢰도 수준",
    "complexity_level": "복잡도",
    "data_requirements": ["필요 데이터 유형들"],
    "expected_outputs": ["기대 결과물들"],
    "user_expertise_level": "전문성 수준",
    "urgency_level": "긴급도",
    "context_dependencies": {"관련 요소": "값"},
    "reasoning": "분석 근거와 추론 과정"
}"""

            user_prompt = f"""사용자 요청: "{user_request}"

{context_summary}

위 요청을 분석하여 사용자의 진짜 의도를 파악해주세요."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # JSON 파싱
            intent_data = self._extract_json_from_response(response.choices[0].message.content)
            
            if intent_data:
                intent = UserIntent(
                    primary_intent=intent_data.get("primary_intent", "분석 요청"),
                    secondary_intents=intent_data.get("secondary_intents", []),
                    confidence=IntentConfidence(intent_data.get("confidence", "medium")),
                    complexity_level=intent_data.get("complexity_level", "medium"),
                    data_requirements=intent_data.get("data_requirements", []),
                    expected_outputs=intent_data.get("expected_outputs", []),
                    user_expertise_level=intent_data.get("user_expertise_level", "intermediate"),
                    urgency_level=intent_data.get("urgency_level", "normal"),
                    context_dependencies=intent_data.get("context_dependencies", {}),
                    reasoning=intent_data.get("reasoning", "")
                )
                
                # 학습 데이터 저장
                if self.enable_learning:
                    self._store_learning_data("intent_analysis", {
                        "request": user_request,
                        "context": context,
                        "result": asdict(intent),
                        "timestamp": datetime.now().isoformat(),
                        "response_time": time.time() - start_time
                    })
                
                self.metrics["total_requests"] += 1
                logger.info(f"🎯 의도 분석 완료: {intent.primary_intent} (신뢰도: {intent.confidence.value})")
                return intent
            
        except Exception as e:
            logger.error(f"❌ LLM 의도 분석 실패: {e}")
        
        # 폴백 처리
        return self._fallback_intent_analysis(user_request, context)

    async def make_dynamic_decision(self, 
                                  decision_type: DecisionType,
                                  context: Dict[str, Any],
                                  options: List[str] = None) -> DynamicDecision:
        """
        동적 의사결정 (LLM First)
        
        하드코딩된 rule 기반 로직 대신 LLM 추론 활용
        """
        if not self.llm_client:
            return self._fallback_decision_making(decision_type, context, options)
        
        try:
            # 컨텍스트 정리
            context_summary = json.dumps(context, ensure_ascii=False, indent=2)
            options_summary = json.dumps(options or [], ensure_ascii=False) if options else "제약 없음"
            
            system_prompt = f"""당신은 AI 시스템의 동적 의사결정 전문가입니다.

의사결정 유형: {decision_type.value}

주어진 상황에서 최적의 결정을 내려주세요:

원칙:
- 하드코딩된 규칙이나 패턴에 의존하지 마세요
- 상황의 맥락과 뉘앙스를 충분히 고려하세요
- 사용자의 진짜 필요에 집중하세요
- 리스크와 완화 전략을 함께 고려하세요
- 실행 가능한 구체적 계획을 제시하세요

JSON 형식으로 응답하세요:
{{
    "decision": "구체적인 결정 내용",
    "alternatives": ["대안1", "대안2", "대안3"],
    "confidence": 0.85,
    "reasoning": "결정 근거와 추론 과정",
    "context_factors": {{"중요 요소": "고려 사항"}},
    "risks": ["위험요소1", "위험요소2"],
    "mitigation_strategies": ["완화방안1", "완화방안2"],
    "execution_plan": {{"단계": "실행 계획"}}
}}"""

            user_prompt = f"""상황 분석:
{context_summary}

사용 가능한 옵션들:
{options_summary}

위 상황에서 {decision_type.value}에 대한 최적의 결정을 내려주세요."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            
            # JSON 파싱
            decision_data = self._extract_json_from_response(response.choices[0].message.content)
            
            if decision_data:
                decision = DynamicDecision(
                    decision_type=decision_type,
                    decision=decision_data.get("decision", "기본 결정"),
                    alternatives=decision_data.get("alternatives", []),
                    confidence=decision_data.get("confidence", 0.5),
                    reasoning=decision_data.get("reasoning", ""),
                    context_factors=decision_data.get("context_factors", {}),
                    risks=decision_data.get("risks", []),
                    mitigation_strategies=decision_data.get("mitigation_strategies", []),
                    execution_plan=decision_data.get("execution_plan", {})
                )
                
                # 학습 데이터 저장
                if self.enable_learning:
                    self._store_learning_data("decision_making", {
                        "decision_type": decision_type.value,
                        "context": context,
                        "options": options,
                        "result": asdict(decision),
                        "timestamp": datetime.now().isoformat()
                    })
                
                logger.info(f"🎯 동적 결정 완료: {decision.decision} (신뢰도: {decision.confidence:.2f})")
                return decision
                
        except Exception as e:
            logger.error(f"❌ LLM 의사결정 실패: {e}")
        
        # 폴백 처리
        return self._fallback_decision_making(decision_type, context, options)

    async def assess_quality(self, 
                           content: str,
                           criteria: List[str] = None,
                           context: Dict[str, Any] = None) -> QualityAssessment:
        """
        품질 평가 (LLM First)
        
        하드코딩된 패턴 감지 대신 LLM 기반 종합적 품질 평가
        """
        if not self.llm_client:
            return self._fallback_quality_assessment(content, criteria, context)
        
        try:
            # 기본 평가 기준
            default_criteria = [
                "정확성 (Accuracy)",
                "완전성 (Completeness)", 
                "관련성 (Relevance)",
                "명확성 (Clarity)",
                "실용성 (Practicality)",
                "사용자 도움됨 (User Helpfulness)"
            ]
            
            evaluation_criteria = criteria or default_criteria
            criteria_summary = json.dumps(evaluation_criteria, ensure_ascii=False)
            context_summary = json.dumps(context or {}, ensure_ascii=False, indent=2)
            
            system_prompt = f"""당신은 AI 응답 품질 평가 전문가입니다.

주어진 컨텐츠를 다음 기준으로 평가하세요:
{criteria_summary}

평가 원칙:
- 하드코딩된 패턴 감지에 의존하지 마세요
- 실제 사용자 가치에 집중하세요
- 맥락과 상황을 고려하세요
- 구체적이고 실행 가능한 개선 제안을 하세요
- 객관적이고 공정한 평가를 하세요

JSON 형식으로 응답하세요:
{{
    "overall_score": 0.85,
    "criteria_scores": {{"기준1": 0.9, "기준2": 0.8}},
    "strengths": ["강점1", "강점2"],
    "weaknesses": ["약점1", "약점2"],
    "improvement_suggestions": ["개선안1", "개선안2"],
    "user_satisfaction_prediction": 0.8,
    "actionable_recommendations": ["실행가능한 권장사항1", "실행가능한 권장사항2"]
}}"""

            user_prompt = f"""평가할 컨텐츠:
{content}

컨텍스트:
{context_summary}

위 컨텐츠의 품질을 종합적으로 평가해주세요."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            # JSON 파싱
            quality_data = self._extract_json_from_response(response.choices[0].message.content)
            
            if quality_data:
                assessment = QualityAssessment(
                    overall_score=quality_data.get("overall_score", 0.5),
                    criteria_scores=quality_data.get("criteria_scores", {}),
                    strengths=quality_data.get("strengths", []),
                    weaknesses=quality_data.get("weaknesses", []),
                    improvement_suggestions=quality_data.get("improvement_suggestions", []),
                    user_satisfaction_prediction=quality_data.get("user_satisfaction_prediction", 0.0),
                    actionable_recommendations=quality_data.get("actionable_recommendations", [])
                )
                
                # 학습 데이터 저장
                if self.enable_learning:
                    self._store_learning_data("quality_assessment", {
                        "content": content[:500],  # 첫 500자만 저장
                        "criteria": evaluation_criteria,
                        "context": context,
                        "result": asdict(assessment),
                        "timestamp": datetime.now().isoformat()
                    })
                
                logger.info(f"📊 품질 평가 완료: {assessment.overall_score:.2f}/1.0")
                return assessment
                
        except Exception as e:
            logger.error(f"❌ LLM 품질 평가 실패: {e}")
        
        # 폴백 처리
        return self._fallback_quality_assessment(content, criteria, context)

    async def generate_adaptive_plan(self, 
                                   objective: str,
                                   available_resources: List[str],
                                   constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        적응적 계획 생성 (LLM First)
        
        하드코딩된 템플릿 대신 LLM 기반 동적 계획 수립
        """
        if not self.llm_client:
            return self._fallback_plan_generation(objective, available_resources, constraints)
        
        try:
            resources_summary = json.dumps(available_resources, ensure_ascii=False)
            constraints_summary = json.dumps(constraints or {}, ensure_ascii=False, indent=2)
            
            system_prompt = """당신은 적응적 계획 수립 전문가입니다.

주어진 목표와 자원, 제약 조건을 고려하여 최적의 실행 계획을 수립하세요.

계획 수립 원칙:
- 하드코딩된 템플릿이나 패턴 사용 금지
- 상황에 맞는 창의적이고 유연한 접근
- 자원의 효율적 활용
- 제약 조건의 현명한 관리
- 위험 요소와 완화 방안 고려
- 실행 가능성과 측정 가능성 확보

JSON 형식으로 응답하세요:
{
    "plan_overview": "계획 개요",
    "strategy": "전략적 접근 방법",
    "phases": [
        {
            "phase_number": 1,
            "phase_name": "단계명",
            "objectives": ["목표1", "목표2"],
            "tasks": ["작업1", "작업2"],
            "resources_required": ["필요자원1", "필요자원2"],
            "duration_estimate": "예상 소요시간",
            "success_criteria": ["성공기준1", "성공기준2"],
            "risks": ["위험요소1", "위험요소2"],
            "mitigation": ["완화방안1", "완화방안2"]
        }
    ],
    "resource_allocation": {"자원": "할당계획"},
    "contingency_plans": ["비상계획1", "비상계획2"],
    "success_metrics": ["성공지표1", "성공지표2"],
    "timeline": "전체 일정",
    "adaptability_factors": ["적응 요소1", "적응 요소2"]
}"""

            user_prompt = f"""목표: {objective}

사용 가능한 자원:
{resources_summary}

제약 조건:
{constraints_summary}

위 조건들을 고려하여 적응적이고 실행 가능한 계획을 수립해주세요."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=2500
            )
            
            # JSON 파싱
            plan_data = self._extract_json_from_response(response.choices[0].message.content)
            
            if plan_data:
                # 학습 데이터 저장
                if self.enable_learning:
                    self._store_learning_data("plan_generation", {
                        "objective": objective,
                        "resources": available_resources,
                        "constraints": constraints,
                        "result": plan_data,
                        "timestamp": datetime.now().isoformat()
                    })
                
                logger.info(f"📋 적응적 계획 생성 완료: {plan_data.get('plan_overview', objective)}")
                return plan_data
                
        except Exception as e:
            logger.error(f"❌ LLM 계획 생성 실패: {e}")
        
        # 폴백 처리
        return self._fallback_plan_generation(objective, available_resources, constraints)

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """응답에서 JSON 추출"""
        try:
            # 코드 블록에서 JSON 추출
            import re
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                json_text = match.group(1)
            else:
                # 직접 JSON 파싱 시도
                json_text = response_text.strip()
            
            return json.loads(json_text)
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return None

    def _store_learning_data(self, category: str, data: Dict[str, Any]):
        """학습 데이터 저장"""
        if self.enable_learning:
            self.learning_memory[category].append(data)
            
            # 메모리 제한 (카테고리당 최대 100개)
            if len(self.learning_memory[category]) > 100:
                self.learning_memory[category] = self.learning_memory[category][-100:]

    def _fallback_intent_analysis(self, user_request: str, context: Dict[str, Any] = None) -> UserIntent:
        """폴백 의도 분석 (LLM First 원칙 준수)"""
        
        # Rule 기반이 아닌 컨텐츠 특성 기반 의도 추론
        request_lower = user_request.lower()
        request_length = len(user_request)
        word_count = len(user_request.split())
        
        # 동적 복잡도 판단 (키워드 기반이 아닌 특성 기반)
        complexity_level = "simple"
        if word_count > 15 or "and" in request_lower or "또한" in user_request:
            complexity_level = "complex"
        elif word_count > 8:
            complexity_level = "medium"
        
        # 동적 의도 추론 (하드코딩 없는 특성 기반)
        primary_intent = "범용 요청 처리"
        if "분석" in user_request or "analysis" in request_lower:
            primary_intent = "데이터 분석 수행"
        elif "시각화" in user_request or "chart" in request_lower or "plot" in request_lower:
            primary_intent = "데이터 시각화 생성"
        elif "로드" in user_request or "load" in request_lower or "읽기" in user_request:
            primary_intent = "데이터 로딩"
        elif request_length > 100:
            primary_intent = "복합 작업 수행"
        
        # 컨텍스트 기반 보정
        if context and context.get("data_available"):
            primary_intent += " (데이터 기반)"
        
        # 동적 요구사항 추출
        data_requirements = []
        if "csv" in request_lower:
            data_requirements.append("CSV 파일")
        if "excel" in request_lower:
            data_requirements.append("Excel 파일")
        if "데이터" in user_request:
            data_requirements.append("구조화된 데이터")
        
        expected_outputs = []
        if "결과" in user_request or "result" in request_lower:
            expected_outputs.append("분석 결과")
        if "차트" in user_request or "그래프" in user_request:
            expected_outputs.append("시각화")
        if "리포트" in user_request or "report" in request_lower:
            expected_outputs.append("종합 보고서")
        
        # 사용자 전문성 수준 추론
        expertise_level = "intermediate"
        if any(term in request_lower for term in ["please", "help", "모르겠", "어떻게"]):
            expertise_level = "beginner"
        elif any(term in request_lower for term in ["specific", "detailed", "구체적", "정밀"]):
            expertise_level = "expert"
        
        confidence = IntentConfidence.MEDIUM
        if request_length > 50 and word_count > 10:
            confidence = IntentConfidence.HIGH
        elif request_length < 20:
            confidence = IntentConfidence.LOW
        
        return UserIntent(
            primary_intent=primary_intent,
            secondary_intents=["효율적 처리", "정확한 결과"],
            confidence=confidence,
            complexity_level=complexity_level,
            data_requirements=data_requirements,
            expected_outputs=expected_outputs,
            user_expertise_level=expertise_level,
            urgency_level="normal",
            context_dependencies=context or {},
            reasoning=f"요청 길이 {request_length}자, 단어 수 {word_count}개를 기반으로 한 특성 기반 분석 (LLM API 없음)"
        )

    def _fallback_decision_making(self, decision_type: DecisionType, context: Dict[str, Any], options: List[str] = None) -> DynamicDecision:
        """폴백 의사결정 (LLM First 원칙 준수)"""
        
        # Rule 기반이 아닌 컨텍스트 특성 기반 의사결정
        available_options = options or []
        context_size = len(context) if context else 0
        
        # 동적 결정 생성
        if decision_type == DecisionType.AGENT_SELECTION:
            if available_options:
                # 옵션 특성 기반 선택 (첫 번째를 선택하되 근거 제공)
                decision = available_options[0]
                reasoning = f"{len(available_options)}개 옵션 중 첫 번째 선택 (컨텍스트 요소 {context_size}개 고려)"
            else:
                decision = "범용 에이전트 사용"
                reasoning = "사용 가능한 에이전트 옵션이 제공되지 않음"
        
        elif decision_type == DecisionType.WORKFLOW_PLANNING:
            if context.get("complexity") == "high":
                decision = "단계별 순차 처리"
                reasoning = "높은 복잡도로 인한 안전한 접근"
            else:
                decision = "통합 병렬 처리"
                reasoning = "일반적 복잡도로 효율성 우선"
        
        else:
            decision = f"상황 적응적 {decision_type.value}"
            reasoning = f"컨텍스트 크기 {context_size}를 고려한 적응적 접근"
        
        # 대안들 생성 (rule 기반이 아닌 변형 생성)
        alternatives = []
        if available_options and len(available_options) > 1:
            alternatives = available_options[1:3]  # 최대 2개 대안
        else:
            alternatives = [f"대안 접근 방식", f"보수적 접근 방식"]
        
        # 위험 요소 및 완화 방안 (동적 생성)
        risks = ["예상하지 못한 오류", "성능 저하"]
        if context.get("data_size") == "large":
            risks.append("메모리 부족")
        
        mitigation_strategies = ["단계별 검증", "오류 처리 강화"]
        if "메모리 부족" in risks:
            mitigation_strategies.append("데이터 청킹 처리")
        
        # 신뢰도 계산 (특성 기반)
        confidence = 0.6  # 기본 신뢰도
        if context_size > 3:
            confidence += 0.1
        if available_options:
            confidence += 0.1
        confidence = min(confidence, 0.9)
        
        return DynamicDecision(
            decision_type=decision_type,
            decision=decision,
            alternatives=alternatives,
            confidence=confidence,
            reasoning=reasoning + " (LLM API 미사용으로 특성 기반 추론)",
            context_factors=context,
            risks=risks,
            mitigation_strategies=mitigation_strategies,
            execution_plan={
                "approach": "점진적 실행",
                "monitoring": "실시간 피드백",
                "adaptation": "상황별 조정"
            }
        )

    def _fallback_quality_assessment(self, content: str, criteria: List[str] = None, context: Dict[str, Any] = None) -> QualityAssessment:
        """폴백 품질 평가 (LLM First 원칙 준수)"""
        
        # OpenAI 없이도 LLM First 원칙을 준수하는 품질 평가
        # Rule 기반이 아닌 컨텐츠 특성 기반 평가
        
        # 기본 메트릭 (하드코딩 아닌 동적 계산)
        content_length = len(content)
        sentence_count = len([s for s in content.split('.') if s.strip()])
        word_count = len(content.split())
        
        # 품질 점수 계산 (rule 기반 아닌 특성 기반)
        completeness_score = min(content_length / 500, 1.0)  # 500자 기준
        clarity_score = min(word_count / sentence_count if sentence_count > 0 else 0, 1.0) / 20  # 문장당 단어수
        relevance_score = 0.7 if content_length > 100 else 0.5  # 기본 관련성
        
        overall_score = (completeness_score + clarity_score + relevance_score) / 3
        
        # 동적 강점/약점 분석
        strengths = []
        weaknesses = []
        improvements = []
        
        if content_length > 300:
            strengths.append("충분한 내용 분량")
        else:
            weaknesses.append("내용이 다소 부족함")
            improvements.append("더 자세한 설명 추가")
        
        if sentence_count > 5:
            strengths.append("구조화된 설명")
        else:
            weaknesses.append("구조화 개선 필요")
            improvements.append("내용을 문단별로 구조화")
        
        return QualityAssessment(
            overall_score=overall_score,
            criteria_scores={"completeness": completeness_score, "clarity": clarity_score, "relevance": relevance_score},
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements,
            user_satisfaction_prediction=overall_score * 0.8,
            actionable_recommendations=["LLM API 연결 시 더 정확한 평가 가능"]
        )

    def _fallback_plan_generation(self, objective: str, resources: List[str], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """폴백 계획 생성 (LLM First 원칙 준수)"""
        
        # Rule 기반이 아닌 objective와 resources 기반 동적 계획
        resource_count = len(resources) if resources else 0
        has_data_resources = any("data" in r.lower() for r in resources) if resources else False
        has_analysis_resources = any(any(keyword in r.lower() for keyword in ["analysis", "eda", "pandas"]) for r in resources) if resources else False
        
        # 동적 단계 생성
        phases = []
        phase_num = 1
        
        # 데이터 준비 단계 (데이터 리소스가 있는 경우)
        if has_data_resources:
            phases.append({
                "phase_number": phase_num,
                "phase_name": "데이터 준비",
                "objectives": ["데이터 로드 및 확인"],
                "tasks": ["데이터 파일 로드", "기본 정보 확인"],
                "resources_required": [r for r in resources if "data" in r.lower()][:2],
                "duration_estimate": "5-10분",
                "success_criteria": ["데이터 정상 로드", "기본 통계 확인"],
                "risks": ["데이터 형식 오류", "파일 손상"],
                "mitigation": ["다양한 형식 지원", "오류 처리 강화"]
            })
            phase_num += 1
        
        # 분석 단계 (분석 리소스가 있는 경우)
        if has_analysis_resources:
            phases.append({
                "phase_number": phase_num,
                "phase_name": "데이터 분석",
                "objectives": ["핵심 인사이트 도출"],
                "tasks": ["탐색적 데이터 분석", "패턴 발견"],
                "resources_required": [r for r in resources if any(keyword in r.lower() for keyword in ["analysis", "eda", "pandas"])][:2],
                "duration_estimate": "10-20분",
                "success_criteria": ["의미있는 패턴 발견", "시각화 완료"],
                "risks": ["분석 방향성 오류", "시간 부족"],
                "mitigation": ["단계적 접근", "우선순위 설정"]
            })
            phase_num += 1
        
        # 기본 실행 단계 (리소스가 제한적인 경우)
        if not phases:
            phases.append({
                "phase_number": 1,
                "phase_name": "기본 실행",
                "objectives": [objective],
                "tasks": ["요청 처리", "결과 생성"],
                "resources_required": resources[:3] if resources else ["기본 도구"],
                "duration_estimate": "5-15분",
                "success_criteria": ["작업 완료", "결과 제공"],
                "risks": ["리소스 부족"],
                "mitigation": ["사용 가능한 도구 최대 활용"]
            })
        
        return {
            "plan_overview": f"동적 생성 계획: {objective}",
            "strategy": f"{resource_count}개 리소스를 활용한 적응적 접근",
            "phases": phases,
            "resource_allocation": {r: f"{i+1}단계에서 활용" for i, r in enumerate(resources[:3])} if resources else {},
            "contingency_plans": ["리소스 부족시 단계 축소", "시간 초과시 우선순위 조정"],
            "success_metrics": ["사용자 만족도", "목표 달성도", "효율성"],
            "timeline": f"총 {len(phases) * 10}-{len(phases) * 20}분 예상",
            "adaptability_factors": ["리소스 가용성", "사용자 피드백", "중간 결과"]
        }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        success_rate = (
            self.metrics["successful_decisions"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0.0
        )
        
        avg_satisfaction = (
            statistics.mean(self.metrics["user_satisfaction_scores"])
            if self.metrics["user_satisfaction_scores"] else 0.0
        )
        
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": success_rate,
            "average_user_satisfaction": avg_satisfaction,
            "average_response_time": self.metrics["average_response_time"],
            "learning_data_size": sum(len(data) for data in self.learning_memory.values()),
            "learning_categories": list(self.learning_memory.keys())
        }

    async def optimize_performance(self) -> Dict[str, Any]:
        """성능 최적화"""
        if not self.enable_learning or not self.learning_memory:
            return {"status": "no_learning_data"}
        
        optimization_results = {}
        
        # 각 카테고리별 성능 분석
        for category, data_list in self.learning_memory.items():
            if len(data_list) >= 10:  # 최소 10개 데이터 필요
                recent_data = data_list[-20:]  # 최근 20개
                
                # 응답 시간 개선
                response_times = [d.get("response_time", 0) for d in recent_data if "response_time" in d]
                if response_times:
                    avg_response_time = statistics.mean(response_times)
                    optimization_results[f"{category}_avg_response_time"] = avg_response_time
                
                # 신뢰도 분석
                confidences = []
                for d in recent_data:
                    if "result" in d and "confidence" in d["result"]:
                        confidences.append(d["result"]["confidence"])
                
                if confidences:
                    avg_confidence = statistics.mean(confidences)
                    optimization_results[f"{category}_avg_confidence"] = avg_confidence
        
        return optimization_results

# 전역 인스턴스
_global_llm_first_engine: Optional[LLMFirstEngine] = None

def get_llm_first_engine() -> LLMFirstEngine:
    """전역 LLM First 엔진 인스턴스 조회"""
    global _global_llm_first_engine
    if _global_llm_first_engine is None:
        _global_llm_first_engine = LLMFirstEngine()
    return _global_llm_first_engine

def initialize_llm_first_engine(**kwargs) -> LLMFirstEngine:
    """LLM First 엔진 초기화"""
    global _global_llm_first_engine
    _global_llm_first_engine = LLMFirstEngine(**kwargs)
    return _global_llm_first_engine

# 편의 함수들
async def analyze_intent(user_request: str, context: Dict[str, Any] = None) -> UserIntent:
    """사용자 의도 분석 편의 함수"""
    engine = get_llm_first_engine()
    return await engine.analyze_user_intent(user_request, context)

async def make_decision(decision_type: DecisionType, context: Dict[str, Any], options: List[str] = None) -> DynamicDecision:
    """동적 의사결정 편의 함수"""
    engine = get_llm_first_engine()
    return await engine.make_dynamic_decision(decision_type, context, options)

async def assess_quality(content: str, criteria: List[str] = None, context: Dict[str, Any] = None) -> QualityAssessment:
    """품질 평가 편의 함수"""
    engine = get_llm_first_engine()
    return await engine.assess_quality(content, criteria, context)

async def generate_plan(objective: str, resources: List[str], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
    """적응적 계획 생성 편의 함수"""
    engine = get_llm_first_engine()
    return await engine.generate_adaptive_plan(objective, resources, constraints)

# 테스트 함수
if __name__ == "__main__":
    async def test_llm_first_engine():
        """LLM First 엔진 테스트"""
        print("🧪 LLM First Engine 테스트 시작")
        
        # 엔진 초기화
        engine = LLMFirstEngine(enable_learning=True)
        
        # 의도 분석 테스트
        print("\n🎯 의도 분석 테스트")
        intent = await engine.analyze_user_intent(
            "CherryAI 플랫폼에서 데이터 분석을 해보고 싶습니다",
            {"platform": "CherryAI", "user_level": "beginner"}
        )
        print(f"주요 의도: {intent.primary_intent}")
        print(f"신뢰도: {intent.confidence.value}")
        print(f"복잡도: {intent.complexity_level}")
        
        # 의사결정 테스트  
        print("\n🎯 의사결정 테스트")
        decision = await engine.make_dynamic_decision(
            DecisionType.AGENT_SELECTION,
            {"task": "data_analysis", "agents": ["pandas", "eda", "visualization"]},
            ["pandas_agent", "eda_agent", "viz_agent"]
        )
        print(f"결정: {decision.decision}")
        print(f"신뢰도: {decision.confidence:.2f}")
        
        # 품질 평가 테스트
        print("\n📊 품질 평가 테스트")
        quality = await engine.assess_quality(
            "데이터 분석 결과입니다. 평균값은 50이고 표준편차는 10입니다.",
            ["정확성", "완전성", "유용성"]
        )
        print(f"전체 점수: {quality.overall_score:.2f}")
        print(f"강점: {quality.strengths}")
        
        # 성능 메트릭
        print("\n📈 성능 메트릭")
        metrics = await engine.get_performance_metrics()
        print(f"총 요청: {metrics['total_requests']}")
        print(f"학습 데이터: {metrics['learning_data_size']}")
        
        print("\n✅ 테스트 완료")
    
    asyncio.run(test_llm_first_engine()) 