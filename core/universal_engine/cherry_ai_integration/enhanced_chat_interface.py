"""
Enhanced Chat Interface - 향상된 채팅 인터페이스

요구사항 3.3에 따른 구현:
- ChatGPT 스타일 메시지 표시 유지
- 메타 추론 과정 시각화
- A2A 에이전트 협업 상태 표시
- 스트리밍 응답 지원
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import json
import time

from ..universal_query_processor import UniversalQueryProcessor
from ..a2a_integration.a2a_agent_discovery import A2AAgentInfo
from ..a2a_integration.a2a_result_integrator import AgentContribution

logger = logging.getLogger(__name__)


class EnhancedChatInterface:
    """
    향상된 채팅 인터페이스
    - ChatGPT 스타일 UI/UX 유지
    - Universal Engine 메타 추론 시각화
    - A2A 에이전트 협업 실시간 표시
    - 스트리밍 응답 및 진행 상황 표시
    """
    
    def __init__(self, universal_engine: UniversalQueryProcessor):
        """EnhancedChatInterface 초기화"""
        self.universal_engine = universal_engine
        self.message_history: List[Dict] = []
        self.typing_indicators: Dict[str, bool] = {}
        logger.info("EnhancedChatInterface initialized")
    
    def render_chat_messages(self):
        """💬 기존 채팅 메시지들을 렌더링"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # 메시지 컨테이너
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                self._render_single_message(message, i)
    
    def _render_single_message(self, message: Dict, index: int):
        """개별 메시지 렌더링"""
        role = message.get("role", "user")
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        
        with st.chat_message(role):
            # 메시지 내용 표시
            st.write(content)
            
            # 타임스탬프 표시
            if timestamp:
                st.caption(f"⏰ {self._format_timestamp(timestamp)}")
            
            # Assistant 메시지인 경우 추가 정보 표시
            if role == "assistant":
                self._render_assistant_extras(message, index)
    
    def _render_assistant_extras(self, message: Dict, index: int):
        """Assistant 메시지의 추가 정보 렌더링"""
        # 메타 추론 결과 표시
        if message.get("meta_reasoning") and st.session_state.get('show_reasoning', True):
            with st.expander("🧠 메타 추론 과정", expanded=False):
                self._render_meta_reasoning_visualization(message["meta_reasoning"])
        
        # A2A 에이전트 기여도 표시
        if message.get("agent_contributions") and st.session_state.get('show_agent_details', True):
            with st.expander("🤖 에이전트 협업 상세", expanded=False):
                self._render_agent_collaboration_details(message["agent_contributions"])
        
        # 응답 품질 평가 표시
        if message.get("quality_metrics"):
            with st.expander("📊 응답 품질 평가", expanded=False):
                self._render_quality_metrics(message["quality_metrics"])
        
        # 피드백 버튼들
        self._render_feedback_buttons(message, index)
    
    def _render_meta_reasoning_visualization(self, meta_reasoning: Dict):
        """🧠 메타 추론 과정 시각화"""
        if not meta_reasoning:
            st.info("메타 추론 정보가 없습니다.")
            return
        
        # 4단계 추론 과정 시각화
        stages = [
            ("초기 관찰", "initial_analysis", "🔍"),
            ("다각도 분석", "multi_perspective", "🔄"),
            ("자가 검증", "self_verification", "✅"),
            ("적응적 응답", "response_strategy", "🎯")
        ]
        
        # 진행 바 표시
        completed_stages = sum(1 for _, key, _ in stages if key in meta_reasoning)
        progress = completed_stages / len(stages)
        st.progress(progress)
        st.caption(f"추론 완료도: {completed_stages}/{len(stages)} 단계")
        
        # 각 단계별 상세 정보
        cols = st.columns(len(stages))
        for i, (stage_name, stage_key, icon) in enumerate(stages):
            with cols[i]:
                if stage_key in meta_reasoning:
                    st.success(f"{icon} {stage_name}")
                    
                    stage_data = meta_reasoning[stage_key]
                    if isinstance(stage_data, dict):
                        # 주요 정보만 표시
                        if "confidence" in stage_data:
                            st.metric("신뢰도", f"{stage_data['confidence']:.1%}")
                        
                        if "key_insights" in stage_data:
                            st.write("**주요 인사이트:**")
                            for insight in stage_data["key_insights"][:2]:
                                st.write(f"• {insight}")
                else:
                    st.info(f"⏳ {stage_name}")
        
        # 전체 메타 분석 결과 (접기/펼치기)
        with st.expander("🔬 전체 메타 분석 데이터", expanded=False):
            st.json(meta_reasoning)
    
    def _render_agent_collaboration_details(self, agent_contributions: List):
        """🤖 A2A 에이전트 협업 상세 정보"""
        if not agent_contributions:
            st.info("에이전트 협업 정보가 없습니다.")
            return
        
        # 에이전트별 기여도 차트
        st.write("### 에이전트 기여도 분석")
        
        # 기여도 막대 차트
        contrib_data = []
        for contrib in agent_contributions:
            if hasattr(contrib, 'agent_name'):
                contrib_data.append({
                    "에이전트": contrib.agent_name,
                    "기여도": contrib.contribution_score,
                    "품질": contrib.quality_score,
                    "고유성": contrib.uniqueness_score,
                    "신뢰성": contrib.reliability_score
                })
        
        if contrib_data:
            import pandas as pd
            df = pd.DataFrame(contrib_data)
            st.bar_chart(df.set_index("에이전트"))
        
        # 개별 에이전트 상세 정보
        for i, contrib in enumerate(agent_contributions):
            if hasattr(contrib, 'agent_name'):
                with st.expander(f"🤖 {contrib.agent_name} 상세", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("전체 기여도", f"{contrib.contribution_score:.2f}")
                        st.metric("품질 점수", f"{contrib.quality_score:.2f}")
                    
                    with col2:
                        st.metric("고유성 점수", f"{contrib.uniqueness_score:.2f}")
                        st.metric("신뢰성 점수", f"{contrib.reliability_score:.2f}")
                    
                    if hasattr(contrib, 'key_insights') and contrib.key_insights:
                        st.write("**핵심 인사이트:**")
                        for insight in contrib.key_insights:
                            st.write(f"• {insight}")
    
    def _render_quality_metrics(self, quality_metrics: Dict):
        """📊 응답 품질 평가 표시"""
        st.write("### 응답 품질 메트릭")
        
        metrics_layout = [
            ("전체 품질", "overall_quality"),
            ("정확성", "accuracy"),
            ("관련성", "relevance"),
            ("유용성", "usefulness")
        ]
        
        cols = st.columns(len(metrics_layout))
        for i, (label, key) in enumerate(metrics_layout):
            with cols[i]:
                value = quality_metrics.get(key, 0.0)
                if value >= 0.8:
                    st.metric(label, f"{value:.1%}", delta="우수")
                elif value >= 0.6:
                    st.metric(label, f"{value:.1%}", delta="양호")
                else:
                    st.metric(label, f"{value:.1%}", delta="개선 필요")
    
    def _render_feedback_buttons(self, message: Dict, index: int):
        """피드백 버튼들 렌더링"""
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍 도움됨", key=f"helpful_{index}"):
                self._record_feedback(message, "helpful", True)
                st.success("피드백이 기록되었습니다!")
        
        with col2:
            if st.button("👎 별로", key=f"not_helpful_{index}"):
                self._record_feedback(message, "helpful", False)
                st.info("피드백이 기록되었습니다!")
        
        with col3:
            if st.button("🔄 다시 생성", key=f"regenerate_{index}"):
                st.session_state.regenerate_request = {
                    "message_index": index,
                    "original_query": message.get("original_query", "")
                }
                st.rerun()
        
        with col4:
            if st.button("💾 저장", key=f"save_{index}"):
                self._save_message_to_history(message)
                st.success("메시지가 저장되었습니다!")
    
    def render_typing_indicator(self, agent_name: str = "Assistant"):
        """⌨️ 타이핑 인디케이터 표시"""
        if self.typing_indicators.get(agent_name, False):
            with st.chat_message("assistant"):
                typing_placeholder = st.empty()
                
                # 애니메이션 효과
                for i in range(3):
                    typing_placeholder.write(f"🤖 {agent_name}이(가) 입력 중" + "." * (i + 1))
                    time.sleep(0.5)
                
                typing_placeholder.empty()
    
    def render_streaming_response(
        self, 
        response_generator: AsyncGenerator[Dict, None], 
        agent_info: Optional[List[A2AAgentInfo]] = None
    ):
        """🌊 스트리밍 응답 렌더링"""
        
        # 응답 컨테이너들
        response_container = st.empty()
        meta_container = st.empty()
        agent_container = st.empty()
        
        # 스트리밍 데이터 축적
        accumulated_response = {
            "content": "",
            "meta_reasoning": {},
            "agent_status": {},
            "progress": 0
        }
        
        async def process_stream():
            async for chunk in response_generator:
                chunk_type = chunk.get("type", "content")
                
                if chunk_type == "content":
                    accumulated_response["content"] += chunk.get("data", "")
                    
                elif chunk_type == "meta_reasoning":
                    accumulated_response["meta_reasoning"].update(chunk.get("data", {}))
                    
                elif chunk_type == "agent_status":
                    accumulated_response["agent_status"].update(chunk.get("data", {}))
                    
                elif chunk_type == "progress":
                    accumulated_response["progress"] = chunk.get("data", 0)
                
                # UI 업데이트
                self._update_streaming_display(
                    response_container, 
                    meta_container, 
                    agent_container, 
                    accumulated_response
                )
        
        # 스트리밍 실행
        asyncio.run(process_stream())
        
        return accumulated_response
    
    def _update_streaming_display(
        self, 
        response_container, 
        meta_container, 
        agent_container, 
        data: Dict
    ):
        """스트리밍 표시 업데이트"""
        
        # 응답 내용 업데이트
        with response_container.container():
            st.write("### 🔄 실시간 응답")
            if data["content"]:
                st.write(data["content"])
            
            # 진행률 표시
            if data["progress"] > 0:
                st.progress(data["progress"] / 100)
                st.caption(f"진행률: {data['progress']:.1f}%")
        
        # 메타 추론 상태 업데이트
        if data["meta_reasoning"]:
            with meta_container.container():
                st.write("### 🧠 추론 진행 상황")
                self._render_meta_reasoning_progress(data["meta_reasoning"])
        
        # 에이전트 상태 업데이트
        if data["agent_status"]:
            with agent_container.container():
                st.write("### 🤖 에이전트 활동")
                self._render_agent_activity(data["agent_status"])
    
    def _render_meta_reasoning_progress(self, meta_data: Dict):
        """메타 추론 진행 상황 표시"""
        stages = ["initial_analysis", "multi_perspective", "self_verification", "response_strategy"]
        completed = sum(1 for stage in stages if stage in meta_data)
        
        progress = completed / len(stages)
        st.progress(progress)
        
        cols = st.columns(len(stages))
        stage_names = ["초기 관찰", "다각도 분석", "자가 검증", "적응적 응답"]
        
        for i, (stage, name) in enumerate(zip(stages, stage_names)):
            with cols[i]:
                if stage in meta_data:
                    st.success(f"✅ {name}")
                else:
                    st.info(f"⏳ {name}")
    
    def _render_agent_activity(self, agent_status: Dict):
        """에이전트 활동 상태 표시"""
        for agent_id, status in agent_status.items():
            status_icon = {
                "active": "🟢",
                "processing": "🟡", 
                "completed": "✅",
                "error": "🔴"
            }.get(status.get("status", "unknown"), "❓")
            
            st.write(f"{status_icon} **{status.get('name', agent_id)}**: {status.get('task', 'Unknown task')}")
    
    def render_chat_input(self) -> Optional[str]:
        """💬 채팅 입력 위젯 렌더링"""
        
        # 자동 완성 제안
        if st.session_state.get('current_data') is not None:
            st.write("💡 **추천 질문:**")
            suggestions = self._get_smart_suggestions()
            
            cols = st.columns(min(len(suggestions), 3))
            for i, suggestion in enumerate(suggestions[:3]):
                with cols[i]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        return suggestion
        
        # 메인 입력
        user_input = st.chat_input("질문이나 분석 요청을 입력하세요...")
        
        # 음성 입력 지원 (향후 확장)
        if st.session_state.get('voice_input_enabled', False):
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🎤 음성 입력"):
                    st.info("음성 입력 기능은 향후 지원 예정입니다.")
        
        return user_input
    
    def _get_smart_suggestions(self) -> List[str]:
        """스마트 질문 제안 생성"""
        current_data = st.session_state.get('current_data')
        if current_data is None:
            return []
        
        import pandas as pd
        
        if isinstance(current_data, pd.DataFrame):
            suggestions = [
                "이 데이터의 주요 패턴은 무엇인가요?",
                "데이터에서 이상값을 찾아주세요",
                "컬럼 간의 상관관계를 분석해주세요"
            ]
        else:
            suggestions = [
                "이 데이터를 요약해주세요",
                "주요 인사이트를 찾아주세요",
                "데이터 품질을 평가해주세요"
            ]
        
        return suggestions
    
    def _format_timestamp(self, timestamp: str) -> str:
        """타임스탬프 포맷팅"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp
    
    def _record_feedback(self, message: Dict, feedback_type: str, value: Any):
        """피드백 기록"""
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "message_id": message.get("id", "unknown"),
            "feedback_type": feedback_type,
            "value": value,
            "message_content": message.get("content", "")[:100]  # 처음 100자만
        }
        
        st.session_state.feedback_history.append(feedback_entry)
        
        # 피드백 이력 크기 제한
        if len(st.session_state.feedback_history) > 100:
            st.session_state.feedback_history = st.session_state.feedback_history[-100:]
    
    def _save_message_to_history(self, message: Dict):
        """메시지를 이력에 저장"""
        if 'saved_messages' not in st.session_state:
            st.session_state.saved_messages = []
        
        saved_message = {
            "timestamp": datetime.now().isoformat(),
            "content": message.get("content", ""),
            "meta_reasoning": message.get("meta_reasoning"),
            "agent_contributions": message.get("agent_contributions"),
            "tags": []  # 향후 태그 기능 지원
        }
        
        st.session_state.saved_messages.append(saved_message)
    
    def set_typing_indicator(self, agent_name: str, is_typing: bool):
        """타이핑 인디케이터 설정"""
        self.typing_indicators[agent_name] = is_typing
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """대화 요약 생성"""
        messages = st.session_state.get('messages', [])
        
        if not messages:
            return {"message": "No conversation history"}
        
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "conversation_start": messages[0].get('timestamp', 'Unknown'),
            "last_message": messages[-1].get('timestamp', 'Unknown'),
            "topics_discussed": self._extract_topics(user_messages),
            "average_response_quality": self._calculate_avg_quality(assistant_messages)
        }
    
    def _extract_topics(self, user_messages: List[Dict]) -> List[str]:
        """대화 주제 추출 (간단한 키워드 기반)"""
        # 향후 LLM을 사용한 더 정교한 주제 추출로 개선 가능
        common_keywords = []
        for msg in user_messages:
            content = msg.get('content', '').lower()
            # 간단한 키워드 추출 로직
            if '분석' in content:
                common_keywords.append('데이터 분석')
            if '패턴' in content:
                common_keywords.append('패턴 발견')
            if '예측' in content:
                common_keywords.append('예측 모델링')
        
        return list(set(common_keywords))
    
    def _calculate_avg_quality(self, assistant_messages: List[Dict]) -> float:
        """평균 응답 품질 계산"""
        quality_scores = []
        for msg in assistant_messages:
            quality_metrics = msg.get('quality_metrics', {})
            if quality_metrics:
                overall_quality = quality_metrics.get('overall_quality', 0.0)
                quality_scores.append(overall_quality)
        
        if quality_scores:
            return sum(quality_scores) / len(quality_scores)
        return 0.0