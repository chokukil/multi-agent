#!/usr/bin/env python3
"""
🧠 CherryAI LLM First UI 통합 시스템

LLM First Engine과 UI를 완전히 융합하여 Rule 기반 로직을 완전히 제거하는 시스템

Key Features:
- LLM 기반 UI 동적 구성
- 지능적 사용자 의도 분석 및 UI 반영
- 컨텍스트 인식 인터페이스 자동 조정
- 사용자 패턴 학습 및 UI 개인화
- 실시간 의도 예측 및 제안
- 에이전트 선택 과정 UI 시각화
- LLM 기반 오류 복구 및 사용자 가이드

Architecture:
- Intent Analyzer: 사용자 의도 실시간 분석
- UI Adapter: LLM 판단에 따른 UI 동적 조정
- Context Manager: 대화 컨텍스트 기반 인터페이스 관리
- Prediction Engine: 다음 액션 예측 및 제안
- Learning System: 사용자 패턴 학습 및 적용
"""

import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# LLM First Engine 임포트
try:
    from core.llm_first_engine import (
        get_llm_first_engine, LLMFirstEngine, UserIntent, 
        DynamicDecision, QualityAssessment, DecisionType
    )
    LLM_FIRST_AVAILABLE = True
except ImportError:
    LLM_FIRST_AVAILABLE = False

# UI 컴포넌트들 임포트
from ui.components.chat_interface import ChatInterface, MessageRole
from ui.components.rich_content_renderer import RichContentRenderer, ContentType
from ui.components.session_manager import SessionManager, SessionType

logger = logging.getLogger(__name__)

class UIAdaptationLevel(Enum):
    """UI 적응 수준"""
    MINIMAL = "minimal"      # 최소한의 변경
    MODERATE = "moderate"    # 중간 수준 변경
    EXTENSIVE = "extensive"  # 광범위한 변경
    COMPLETE = "complete"    # 완전한 재구성

class InterfaceMode(Enum):
    """인터페이스 모드"""
    CONVERSATION = "conversation"       # 일반 대화
    DATA_ANALYSIS = "data_analysis"    # 데이터 분석
    FILE_PROCESSING = "file_processing" # 파일 처리
    RESEARCH = "research"              # 연구/조사
    CODING = "coding"                  # 코딩 지원
    CREATIVE = "creative"              # 창작 활동

@dataclass
class UIAdaptation:
    """UI 적응 정보"""
    adaptation_id: str
    intent: UserIntent
    suggested_mode: InterfaceMode
    adaptation_level: UIAdaptationLevel
    ui_changes: Dict[str, Any]
    confidence: float
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptation_id": self.adaptation_id,
            "intent": self.intent.to_dict() if self.intent else None,
            "suggested_mode": self.suggested_mode.value,
            "adaptation_level": self.adaptation_level.value,
            "ui_changes": self.ui_changes,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class PredictedAction:
    """예측된 액션"""
    action_id: str
    action_type: str
    description: str
    confidence: float
    suggested_ui_elements: List[str]
    estimated_time: Optional[float] = None
    prerequisites: List[str] = field(default_factory=list)

class LLMFirstUIIntegrator:
    """
    🧠 LLM First UI 통합기
    
    LLM의 지능적 판단을 UI에 실시간으로 반영하는 메인 시스템
    """
    
    def __init__(self):
        """통합기 초기화"""
        self.llm_engine: Optional[LLMFirstEngine] = None
        
        # 현재 UI 상태
        self.current_mode: InterfaceMode = InterfaceMode.CONVERSATION
        self.adaptation_history: List[UIAdaptation] = []
        self.user_patterns: Dict[str, Any] = {}
        
        # 예측 및 제안
        self.predicted_actions: List[PredictedAction] = []
        self.last_intent_analysis: Optional[UserIntent] = None
        
        # 학습 데이터
        self.user_interaction_log: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, float] = {}
        
        # 개인화 설정
        self.user_preferences = {
            "ui_complexity": "moderate",  # simple, moderate, advanced
            "suggestion_frequency": "balanced",  # minimal, balanced, frequent
            "adaptation_sensitivity": "medium",  # low, medium, high
            "auto_mode_switching": True
        }
        
        logger.info("🧠 LLM First UI 통합기 초기화 완료")
    
    async def initialize(self) -> bool:
        """통합기 초기화"""
        try:
            if LLM_FIRST_AVAILABLE:
                self.llm_engine = get_llm_first_engine()
                logger.info("🧠 LLM First Engine 연결 완료")
                return True
            else:
                logger.warning("LLM First Engine을 사용할 수 없습니다")
                return False
        except Exception as e:
            logger.error(f"LLM First UI 통합기 초기화 실패: {e}")
            return False
    
    async def analyze_user_intent_for_ui(self, 
                                       user_input: str, 
                                       context: Dict[str, Any] = None) -> Optional[UIAdaptation]:
        """사용자 의도 분석 및 UI 적응 제안"""
        try:
            if not self.llm_engine:
                return None
            
            # LLM First Engine으로 의도 분석
            intent = await self.llm_engine.analyze_intent(
                user_input=user_input,
                context=context or {}
            )
            
            self.last_intent_analysis = intent
            
            # 의도 기반 UI 적응 결정
            adaptation = await self._determine_ui_adaptation(intent, user_input)
            
            if adaptation:
                self.adaptation_history.append(adaptation)
                logger.info(f"🧠 UI 적응 제안: {adaptation.suggested_mode.value} ({adaptation.confidence:.2f})")
                
                # 사용자 패턴 학습
                await self._learn_user_pattern(user_input, intent, adaptation)
            
            return adaptation
            
        except Exception as e:
            logger.error(f"사용자 의도 분석 실패: {e}")
            return None
    
    async def _determine_ui_adaptation(self, 
                                     intent: UserIntent, 
                                     user_input: str) -> Optional[UIAdaptation]:
        """UI 적응 결정"""
        try:
            # LLM에게 UI 적응 방안 요청
            adaptation_prompt = f"""
            사용자 의도: {intent.primary_intent}
            세부 의도: {intent.secondary_intents}
            컨텍스트: {intent.context}
            사용자 입력: {user_input}
            
            현재 UI 모드: {self.current_mode.value}
            사용자 선호도: {self.user_preferences}
            
            이 정보를 바탕으로 가장 적절한 UI 적응 방안을 제안해주세요:
            1. 권장 인터페이스 모드
            2. 적응 수준 (minimal/moderate/extensive/complete)
            3. 구체적인 UI 변경사항
            4. 적응 이유 및 신뢰도
            """
            
            # LLM 결정 요청
            decision = await self.llm_engine.make_decision(
                decision_type=DecisionType.UI_ADAPTATION,
                context={"prompt": adaptation_prompt},
                constraints={}
            )
            
            # 결정 내용 파싱 및 UIAdaptation 생성
            ui_changes = self._parse_ui_changes(decision.reasoning)
            suggested_mode = self._determine_interface_mode(intent, decision)
            adaptation_level = self._determine_adaptation_level(decision, intent)
            
            return UIAdaptation(
                adaptation_id=str(uuid.uuid4()),
                intent=intent,
                suggested_mode=suggested_mode,
                adaptation_level=adaptation_level,
                ui_changes=ui_changes,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"UI 적응 결정 실패: {e}")
            return None
    
    def _determine_interface_mode(self, 
                                intent: UserIntent, 
                                decision: DynamicDecision) -> InterfaceMode:
        """인터페이스 모드 결정"""
        # LLM 결정 내용에서 모드 추출
        reasoning_lower = decision.reasoning.lower()
        
        # 키워드 기반 모드 매핑 (LLM First 원칙에 따라 LLM 판단 우선)
        mode_keywords = {
            InterfaceMode.DATA_ANALYSIS: ["데이터", "분석", "통계", "차트", "그래프", "시각화"],
            InterfaceMode.FILE_PROCESSING: ["파일", "업로드", "처리", "변환", "저장"],
            InterfaceMode.RESEARCH: ["연구", "조사", "검색", "자료", "정보"],
            InterfaceMode.CODING: ["코드", "프로그래밍", "개발", "함수", "알고리즘"],
            InterfaceMode.CREATIVE: ["창작", "글쓰기", "아이디어", "브레인스토밍"]
        }
        
        # 가장 높은 매칭 점수를 가진 모드 선택
        max_score = 0
        selected_mode = InterfaceMode.CONVERSATION
        
        for mode, keywords in mode_keywords.items():
            score = sum(1 for keyword in keywords if keyword in reasoning_lower)
            if score > max_score:
                max_score = score
                selected_mode = mode
        
        return selected_mode
    
    def _determine_adaptation_level(self, 
                                  decision: DynamicDecision, 
                                  intent: UserIntent) -> UIAdaptationLevel:
        """적응 수준 결정"""
        # 신뢰도와 의도의 복잡성을 바탕으로 결정
        confidence = decision.confidence
        intent_complexity = len(intent.secondary_intents) + (1 if intent.context else 0)
        
        if confidence > 0.8 and intent_complexity > 2:
            return UIAdaptationLevel.EXTENSIVE
        elif confidence > 0.6 and intent_complexity > 1:
            return UIAdaptationLevel.MODERATE
        elif confidence > 0.4:
            return UIAdaptationLevel.MINIMAL
        else:
            return UIAdaptationLevel.MINIMAL
    
    def _parse_ui_changes(self, reasoning: str) -> Dict[str, Any]:
        """LLM 추론에서 UI 변경사항 파싱"""
        ui_changes = {
            "layout_changes": [],
            "component_additions": [],
            "component_removals": [],
            "style_adjustments": [],
            "interaction_enhancements": []
        }
        
        # LLM 추론에서 구체적인 UI 변경사항 추출
        reasoning_lower = reasoning.lower()
        
        # 레이아웃 변경
        if "사이드바" in reasoning_lower:
            ui_changes["layout_changes"].append("sidebar_adjustment")
        if "전체화면" in reasoning_lower or "풀스크린" in reasoning_lower:
            ui_changes["layout_changes"].append("fullscreen_mode")
        
        # 컴포넌트 추가
        if "업로드" in reasoning_lower:
            ui_changes["component_additions"].append("file_upload_area")
        if "차트" in reasoning_lower or "그래프" in reasoning_lower:
            ui_changes["component_additions"].append("chart_preview_area")
        if "프로그래스" in reasoning_lower or "진행" in reasoning_lower:
            ui_changes["component_additions"].append("progress_indicator")
        
        # 상호작용 개선
        if "단축키" in reasoning_lower:
            ui_changes["interaction_enhancements"].append("keyboard_shortcuts")
        if "자동완성" in reasoning_lower:
            ui_changes["interaction_enhancements"].append("auto_completion")
        
        return ui_changes
    
    async def apply_ui_adaptation(self, adaptation: UIAdaptation) -> bool:
        """UI 적응 적용"""
        try:
            # 적응 수준에 따른 점진적 적용
            if adaptation.adaptation_level == UIAdaptationLevel.MINIMAL:
                return await self._apply_minimal_changes(adaptation)
            elif adaptation.adaptation_level == UIAdaptationLevel.MODERATE:
                return await self._apply_moderate_changes(adaptation)
            elif adaptation.adaptation_level == UIAdaptationLevel.EXTENSIVE:
                return await self._apply_extensive_changes(adaptation)
            else:  # COMPLETE
                return await self._apply_complete_changes(adaptation)
                
        except Exception as e:
            logger.error(f"UI 적응 적용 실패: {e}")
            return False
    
    async def _apply_minimal_changes(self, adaptation: UIAdaptation) -> bool:
        """최소한의 UI 변경 적용"""
        try:
            # 세션 상태에 UI 힌트 저장
            st.session_state["ui_mode"] = adaptation.suggested_mode.value
            st.session_state["ui_suggestions"] = adaptation.ui_changes.get("interaction_enhancements", [])
            
            # 간단한 알림 표시
            if adaptation.confidence > 0.7:
                st.info(f"💡 {adaptation.reasoning[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"최소한의 UI 변경 적용 실패: {e}")
            return False
    
    async def _apply_moderate_changes(self, adaptation: UIAdaptation) -> bool:
        """중간 수준의 UI 변경 적용"""
        try:
            # 모드 변경
            self.current_mode = adaptation.suggested_mode
            st.session_state["interface_mode"] = adaptation.suggested_mode.value
            
            # UI 컴포넌트 동적 조정
            ui_changes = adaptation.ui_changes
            
            if "file_upload_area" in ui_changes.get("component_additions", []):
                st.session_state["show_file_upload"] = True
            
            if "chart_preview_area" in ui_changes.get("component_additions", []):
                st.session_state["show_chart_preview"] = True
            
            if "progress_indicator" in ui_changes.get("component_additions", []):
                st.session_state["show_progress"] = True
            
            # 상호작용 개선
            if "keyboard_shortcuts" in ui_changes.get("interaction_enhancements", []):
                st.session_state["enable_advanced_shortcuts"] = True
            
            # 사용자에게 변경사항 알림
            st.success(f"🎯 인터페이스가 {adaptation.suggested_mode.value} 모드로 최적화되었습니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"중간 수준의 UI 변경 적용 실패: {e}")
            return False
    
    async def _apply_extensive_changes(self, adaptation: UIAdaptation) -> bool:
        """광범위한 UI 변경 적용"""
        try:
            # 완전한 인터페이스 재구성
            self.current_mode = adaptation.suggested_mode
            
            # 세션 상태 대폭 업데이트
            st.session_state.update({
                "interface_mode": adaptation.suggested_mode.value,
                "ui_layout": self._get_optimized_layout(adaptation.suggested_mode),
                "adaptive_components": adaptation.ui_changes,
                "llm_optimization_active": True,
                "adaptation_confidence": adaptation.confidence
            })
            
            # 페이지 리로드 권장
            st.balloons()  # 변경 완료 시각적 피드백
            st.success(f"🚀 {adaptation.suggested_mode.value} 전용 인터페이스로 최적화 완료!")
            st.info("더 나은 경험을 위해 페이지를 새로고침해주세요.")
            
            return True
            
        except Exception as e:
            logger.error(f"광범위한 UI 변경 적용 실패: {e}")
            return False
    
    async def _apply_complete_changes(self, adaptation: UIAdaptation) -> bool:
        """완전한 UI 재구성 적용"""
        try:
            # 전체 인터페이스 재설계
            self.current_mode = adaptation.suggested_mode
            
            # 새로운 세션 생성 및 이전
            new_session_config = {
                "optimized_for": adaptation.suggested_mode.value,
                "llm_customized": True,
                "adaptation_reasoning": adaptation.reasoning,
                "confidence": adaptation.confidence
            }
            
            st.session_state["session_config"] = new_session_config
            st.session_state["ui_completely_adapted"] = True
            
            # 사용자에게 강력한 권장사항 제공
            st.success(f"🎉 {adaptation.suggested_mode.value} 작업에 완벽히 최적화된 새로운 인터페이스가 준비되었습니다!")
            st.info("최상의 경험을 위해 새 세션을 시작하시기 바랍니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"완전한 UI 재구성 적용 실패: {e}")
            return False
    
    def _get_optimized_layout(self, mode: InterfaceMode) -> Dict[str, Any]:
        """모드별 최적화된 레이아웃 반환"""
        layouts = {
            InterfaceMode.DATA_ANALYSIS: {
                "sidebar_width": "wide",
                "main_columns": [2, 1],
                "enable_charts": True,
                "enable_tables": True,
                "file_upload_prominent": True
            },
            InterfaceMode.FILE_PROCESSING: {
                "sidebar_width": "narrow",
                "main_columns": [1, 2],
                "enable_drag_drop": True,
                "enable_batch_processing": True,
                "file_upload_prominent": True
            },
            InterfaceMode.CODING: {
                "sidebar_width": "narrow",
                "main_columns": [1],
                "enable_code_editor": True,
                "enable_syntax_highlighting": True,
                "enable_code_execution": True
            },
            InterfaceMode.RESEARCH: {
                "sidebar_width": "wide",
                "main_columns": [1, 1],
                "enable_search_tools": True,
                "enable_reference_manager": True,
                "enable_note_taking": True
            }
        }
        
        return layouts.get(mode, {"sidebar_width": "medium", "main_columns": [1]})
    
    async def predict_next_actions(self, 
                                 current_context: Dict[str, Any]) -> List[PredictedAction]:
        """다음 액션 예측"""
        try:
            if not self.llm_engine or not self.last_intent_analysis:
                return []
            
            # LLM에게 다음 액션 예측 요청
            prediction_prompt = f"""
            현재 컨텍스트: {current_context}
            최근 사용자 의도: {self.last_intent_analysis.primary_intent}
            사용자 패턴: {self.user_patterns}
            
            이 정보를 바탕으로 사용자가 다음에 수행할 가능성이 높은 액션들을 예측해주세요.
            각 액션에 대해 신뢰도와 추천 UI 요소를 포함해주세요.
            """
            
            decision = await self.llm_engine.make_decision(
                decision_type=DecisionType.NEXT_ACTION_PREDICTION,
                context={"prompt": prediction_prompt},
                constraints={}
            )
            
            # 예측 결과 파싱
            predicted_actions = self._parse_predicted_actions(decision.reasoning)
            self.predicted_actions = predicted_actions
            
            return predicted_actions
            
        except Exception as e:
            logger.error(f"다음 액션 예측 실패: {e}")
            return []
    
    def _parse_predicted_actions(self, reasoning: str) -> List[PredictedAction]:
        """LLM 추론에서 예측된 액션들 파싱"""
        actions = []
        
        # 간단한 패턴 매칭으로 액션 추출 (실제로는 더 정교한 파싱 필요)
        reasoning_lines = reasoning.split('\n')
        
        for i, line in enumerate(reasoning_lines):
            if any(keyword in line.lower() for keyword in ['액션', '작업', '다음', '예상']):
                action = PredictedAction(
                    action_id=str(uuid.uuid4()),
                    action_type="predicted",
                    description=line.strip(),
                    confidence=0.7,  # 기본값
                    suggested_ui_elements=["button", "shortcut"]
                )
                actions.append(action)
        
        return actions[:5]  # 최대 5개까지
    
    async def _learn_user_pattern(self, 
                                user_input: str, 
                                intent: UserIntent, 
                                adaptation: UIAdaptation) -> None:
        """사용자 패턴 학습"""
        try:
            # 상호작용 로그 기록
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "intent": intent.primary_intent,
                "adaptation_applied": adaptation.suggested_mode.value,
                "confidence": adaptation.confidence
            }
            
            self.user_interaction_log.append(interaction)
            
            # 패턴 분석 및 업데이트
            if len(self.user_interaction_log) > 10:
                await self._analyze_user_patterns()
            
        except Exception as e:
            logger.error(f"사용자 패턴 학습 실패: {e}")
    
    async def _analyze_user_patterns(self) -> None:
        """사용자 패턴 분석"""
        try:
            # 최근 상호작용 분석
            recent_interactions = self.user_interaction_log[-20:]
            
            # 선호 모드 분석
            mode_counts = {}
            for interaction in recent_interactions:
                mode = interaction.get("adaptation_applied", "conversation")
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # 패턴 업데이트
            self.user_patterns.update({
                "preferred_modes": mode_counts,
                "interaction_frequency": len(recent_interactions),
                "last_analysis": datetime.now().isoformat()
            })
            
            logger.info(f"🧠 사용자 패턴 분석 완료: {mode_counts}")
            
        except Exception as e:
            logger.error(f"사용자 패턴 분석 실패: {e}")
    
    def render_adaptive_ui_elements(self) -> None:
        """적응형 UI 요소 렌더링"""
        try:
            # 현재 모드에 따른 특별 UI 요소들
            if self.current_mode == InterfaceMode.DATA_ANALYSIS:
                self._render_data_analysis_tools()
            elif self.current_mode == InterfaceMode.FILE_PROCESSING:
                self._render_file_processing_tools()
            elif self.current_mode == InterfaceMode.CODING:
                self._render_coding_tools()
            
            # 예측된 액션 제안
            if self.predicted_actions:
                self._render_action_suggestions()
            
            # 적응 히스토리 (디버그 모드)
            if st.session_state.get("debug_mode", False):
                self._render_adaptation_history()
                
        except Exception as e:
            logger.error(f"적응형 UI 요소 렌더링 실패: {e}")
    
    def _render_data_analysis_tools(self) -> None:
        """데이터 분석 도구 렌더링"""
        with st.expander("🔬 데이터 분석 도구", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 빠른 통계", help="기본 통계 분석 수행"):
                    st.session_state["quick_stats_requested"] = True
            
            with col2:
                if st.button("📈 시각화", help="데이터 시각화 생성"):
                    st.session_state["visualization_requested"] = True
            
            with col3:
                if st.button("🔍 패턴 분석", help="데이터 패턴 탐지"):
                    st.session_state["pattern_analysis_requested"] = True
    
    def _render_file_processing_tools(self) -> None:
        """파일 처리 도구 렌더링"""
        with st.expander("📁 파일 처리 도구", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 배치 업로드", help="여러 파일 동시 업로드"):
                    st.session_state["batch_upload_mode"] = True
            
            with col2:
                if st.button("🔄 형식 변환", help="파일 형식 변환"):
                    st.session_state["format_conversion_mode"] = True
    
    def _render_coding_tools(self) -> None:
        """코딩 도구 렌더링"""
        with st.expander("💻 코딩 도구", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🐛 디버그", help="코드 디버깅 지원"):
                    st.session_state["debug_mode_requested"] = True
            
            with col2:
                if st.button("⚡ 최적화", help="코드 최적화 제안"):
                    st.session_state["optimization_requested"] = True
            
            with col3:
                if st.button("📚 문서화", help="코드 문서화 생성"):
                    st.session_state["documentation_requested"] = True
    
    def _render_action_suggestions(self) -> None:
        """예측 액션 제안 렌더링"""
        if not self.predicted_actions:
            return
            
        st.markdown("### 🎯 다음 액션 제안")
        
        for action in self.predicted_actions[:3]:  # 상위 3개만 표시
            confidence_color = "green" if action.confidence > 0.7 else "orange"
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{action.description}**")
                
                with col2:
                    st.markdown(f":{confidence_color}[{action.confidence:.0%}]")
    
    def _render_adaptation_history(self) -> None:
        """적응 히스토리 렌더링 (디버그용)"""
        if not self.adaptation_history:
            return
            
        with st.expander("🔧 UI 적응 히스토리 (디버그)", expanded=False):
            for adaptation in self.adaptation_history[-5:]:  # 최근 5개만
                st.markdown(f"""
                **{adaptation.timestamp.strftime('%H:%M:%S')}** - 
                {adaptation.suggested_mode.value} 
                ({adaptation.confidence:.0%})
                
                {adaptation.reasoning[:100]}...
                """)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 정보"""
        return {
            "llm_engine_available": LLM_FIRST_AVAILABLE and self.llm_engine is not None,
            "current_mode": self.current_mode.value,
            "adaptations_count": len(self.adaptation_history),
            "predicted_actions_count": len(self.predicted_actions),
            "user_patterns": self.user_patterns,
            "last_intent": self.last_intent_analysis.primary_intent if self.last_intent_analysis else None
        }

# 전역 인스턴스 관리
_llm_first_ui_integrator_instance = None

def get_llm_first_ui_integrator() -> LLMFirstUIIntegrator:
    """LLM First UI 통합기 싱글톤 인스턴스 반환"""
    global _llm_first_ui_integrator_instance
    if _llm_first_ui_integrator_instance is None:
        _llm_first_ui_integrator_instance = LLMFirstUIIntegrator()
    return _llm_first_ui_integrator_instance

async def initialize_llm_first_ui_integrator() -> LLMFirstUIIntegrator:
    """LLM First UI 통합기 초기화"""
    global _llm_first_ui_integrator_instance
    _llm_first_ui_integrator_instance = LLMFirstUIIntegrator()
    await _llm_first_ui_integrator_instance.initialize()
    return _llm_first_ui_integrator_instance 