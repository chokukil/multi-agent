"""
Expert Answer Renderer

Professional UI component for displaying Phase 3 synthesized expert-level answers
with comprehensive formatting, visualizations, and interactive elements.

Author: CherryAI Development Team
Version: 1.0.0
"""

import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64


class ExpertAnswerRenderer:
    """전문가급 답변을 위한 고급 렌더링 컴포넌트"""
    
    def __init__(self):
        """Initialize Expert Answer Renderer"""
        self.style_config = self._load_style_config()
    
    def render_expert_answer(self, expert_answer: Dict[str, Any]):
        """
        전문가급 답변을 전문적으로 렌더링
        
        Args:
            expert_answer: Phase 3에서 합성된 전문가급 답변
        """
        if not expert_answer.get("success"):
            self._render_error_fallback(expert_answer)
            return
        
        synthesized_answer = expert_answer["synthesized_answer"]
        quality_report = expert_answer["quality_report"]
        
        # 1. 메인 헤더
        self._render_expert_header(expert_answer)
        
        # 2. 품질 대시보드
        self._render_quality_dashboard(quality_report, expert_answer["confidence_score"])
        
        # 3. 임원 요약 (Executive Summary)
        self._render_executive_summary(synthesized_answer)
        
        # 4. 주요 인사이트
        self._render_key_insights(synthesized_answer)
        
        # 5. 상세 섹션들
        self._render_detailed_sections(synthesized_answer)
        
        # 6. 실행 권고사항
        self._render_recommendations(synthesized_answer)
        
        # 7. 다음 단계
        self._render_next_steps(synthesized_answer)
        
        # 8. 시각화 요소들
        self._render_visualizations(synthesized_answer)
        
        # 9. 기술적 부록
        self._render_technical_appendix(expert_answer)
        
        # 10. 메타데이터 및 추적 정보
        self._render_metadata_section(expert_answer)
    
    def _render_expert_header(self, expert_answer: Dict[str, Any]):
        """전문가급 답변 헤더 렌더링"""
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5rem; text-align: center;">
                🧠 CherryAI Expert Analysis
            </h1>
            <p style="text-align: center; font-size: 1.2rem; margin: 0.5rem 0;">
                전문가급 지능형 분석 결과
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 처리 시간 및 신뢰도 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "처리 시간",
                f"{expert_answer['processing_time']:.2f}s",
                delta=None
            )
        
        with col2:
            confidence = expert_answer['confidence_score']
            st.metric(
                "전체 신뢰도", 
                f"{confidence:.1%}",
                delta=f"+{(confidence-0.8)*100:.1f}%" if confidence > 0.8 else None
            )
        
        with col3:
            agents_used = expert_answer['metadata']['total_agents_used']
            st.metric(
                "활용 에이전트",
                f"{agents_used}개",
                delta=None
            )
        
        with col4:
            quality_score = expert_answer['metadata']['phase3_quality_score']
            st.metric(
                "품질 점수",
                f"{quality_score:.1%}",
                delta=f"+{(quality_score-0.8)*100:.1f}%" if quality_score > 0.8 else None
            )
    
    def _render_quality_dashboard(self, quality_report: Any, confidence_score: float):
        """품질 대시보드 렌더링"""
        
        st.markdown("### 📊 답변 품질 분석")
        
        # 품질 메트릭 시각화
        if hasattr(quality_report, 'metric_scores'):
            metrics_data = []
            for metric, score in quality_report.metric_scores.items():
                metrics_data.append({
                    'metric': metric.replace('_', ' ').title(),
                    'score': score * 100
                })
            
            if metrics_data:
                # 레이더 차트 생성
                fig = go.Figure()
                
                categories = [item['metric'] for item in metrics_data]
                values = [item['score'] for item in metrics_data]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='품질 점수',
                    line_color='rgb(102, 126, 234)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title="답변 품질 메트릭",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 품질 향상 제안
        if hasattr(quality_report, 'improvement_suggestions') and quality_report.improvement_suggestions:
            with st.expander("🔧 품질 향상 제안", expanded=False):
                for suggestion in quality_report.improvement_suggestions:
                    improvement_type = getattr(suggestion, 'improvement_type', 'general')
                    description = getattr(suggestion, 'description', str(suggestion))
                    
                    if improvement_type == 'critical':
                        st.error(f"🚨 중요: {description}")
                    elif improvement_type == 'recommendation':
                        st.warning(f"💡 권장: {description}")
                    else:
                        st.info(f"ℹ️ 참고: {description}")
    
    def _render_executive_summary(self, synthesized_answer: Any):
        """임원 요약 렌더링"""
        
        st.markdown("---")
        st.markdown("## 📋 Executive Summary")
        
        if hasattr(synthesized_answer, 'executive_summary'):
            summary = synthesized_answer.executive_summary
            
            # 요약을 하이라이트된 박스로 표시
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0 0 1rem 0; color: white;">핵심 요약</h3>
                <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_key_insights(self, synthesized_answer: Any):
        """주요 인사이트 렌더링"""
        
        if hasattr(synthesized_answer, 'key_insights') and synthesized_answer.key_insights:
            st.markdown("## 💡 핵심 인사이트")
            
            # 인사이트를 3컬럼으로 배치
            num_insights = len(synthesized_answer.key_insights)
            cols = st.columns(min(3, num_insights))
            
            for i, insight in enumerate(synthesized_answer.key_insights):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.1); 
                                border-left: 4px solid #4ECDC4; 
                                padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                        <h4 style="color: #4ECDC4; margin: 0 0 0.5rem 0;">인사이트 {i+1}</h4>
                        <p style="margin: 0;">{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_detailed_sections(self, synthesized_answer: Any):
        """상세 섹션들 렌더링"""
        
        if hasattr(synthesized_answer, 'main_sections') and synthesized_answer.main_sections:
            st.markdown("## 📖 상세 분석")
            
            for i, section in enumerate(synthesized_answer.main_sections):
                title = getattr(section, 'title', f'섹션 {i+1}')
                content = getattr(section, 'content', str(section))
                priority = getattr(section, 'priority', 1)
                confidence = getattr(section, 'confidence', 0.8)
                
                # 우선순위에 따른 확장 여부 결정
                expanded = priority <= 2
                
                with st.expander(f"📄 {title} (신뢰도: {confidence:.1%})", expanded=expanded):
                    st.markdown(content)
                    
                    # 섹션 메타데이터 표시
                    if hasattr(section, 'sources') and section.sources:
                        st.caption(f"출처: {', '.join(section.sources)}")
    
    def _render_recommendations(self, synthesized_answer: Any):
        """실행 권고사항 렌더링"""
        
        if hasattr(synthesized_answer, 'recommendations') and synthesized_answer.recommendations:
            st.markdown("## 🎯 실행 권고사항")
            
            for i, recommendation in enumerate(synthesized_answer.recommendations):
                # 권고사항을 체크박스 형태로 표시
                checkbox_key = f"recommendation_{i}"
                checked = st.checkbox(
                    recommendation,
                    key=checkbox_key,
                    help="클릭하여 완료 표시"
                )
                
                if checked:
                    st.success(f"✅ 권고사항 {i+1} 검토 완료")
    
    def _render_next_steps(self, synthesized_answer: Any):
        """다음 단계 렌더링"""
        
        if hasattr(synthesized_answer, 'next_steps') and synthesized_answer.next_steps:
            st.markdown("## 🚀 다음 단계")
            
            # 단계별 타임라인 형태로 표시
            for i, step in enumerate(synthesized_answer.next_steps):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 1rem 0;">
                    <div style="background: #667eea; color: white; 
                                width: 30px; height: 30px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;
                                margin-right: 1rem; font-weight: bold;">
                        {i+1}
                    </div>
                    <div style="flex: 1; padding: 1rem; background: rgba(255, 255, 255, 0.1);
                                border-radius: 8px;">
                        {step}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_visualizations(self, synthesized_answer: Any):
        """시각화 요소들 렌더링"""
        
        if hasattr(synthesized_answer, 'visualizations') and synthesized_answer.visualizations:
            st.markdown("## 📊 시각화")
            
            for viz in synthesized_answer.visualizations:
                if hasattr(viz, 'chart_data'):
                    # Plotly 차트 렌더링
                    try:
                        fig_dict = json.loads(viz.chart_data) if isinstance(viz.chart_data, str) else viz.chart_data
                        fig = go.Figure(fig_dict)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"시각화 렌더링 오류: {e}")
    
    def _render_technical_appendix(self, expert_answer: Dict[str, Any]):
        """기술적 부록 렌더링"""
        
        with st.expander("🔧 기술적 세부사항", expanded=False):
            
            # Phase별 처리 결과
            st.markdown("### Phase별 처리 성과")
            
            metadata = expert_answer.get("metadata", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Phase 1 점수", f"{metadata.get('phase1_score', 0):.1%}")
                st.caption("쿼리 처리 및 의도 분석")
            
            with col2:
                st.metric("Phase 2 점수", f"{metadata.get('phase2_integration_score', 0):.1%}")
                st.caption("에이전트 오케스트레이션")
            
            with col3:
                st.metric("Phase 3 점수", f"{metadata.get('phase3_quality_score', 0):.1%}")
                st.caption("답변 합성 및 품질")
            
            # 도메인 분석 결과
            if "domain_analysis" in expert_answer:
                domain_analysis = expert_answer["domain_analysis"]
                st.markdown("### 도메인 분석")
                
                if hasattr(domain_analysis, 'taxonomy'):
                    taxonomy = domain_analysis.taxonomy
                    primary_domain = getattr(taxonomy.primary_domain, 'value', 'Unknown')
                    st.write(f"**주 도메인**: {primary_domain}")
                
                if hasattr(domain_analysis, 'key_concepts'):
                    st.write("**핵심 개념들**:")
                    for concept, details in domain_analysis.key_concepts.items():
                        confidence = details.get('confidence', 0) if isinstance(details, dict) else 0
                        st.write(f"- {concept}: {confidence:.1%} 신뢰도")
            
            # 에이전트 활용 요약
            agent_summary = expert_answer.get("agent_results_summary", {})
            if agent_summary:
                st.markdown("### 에이전트 활용 현황")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("총 에이전트", agent_summary.get('total_agents', 0))
                    st.metric("성공한 에이전트", agent_summary.get('successful_agents', 0))
                
                with col2:
                    st.metric("생성된 아티팩트", agent_summary.get('total_artifacts', 0))
                    st.metric("평균 신뢰도", f"{agent_summary.get('average_confidence', 0):.1%}")
                
                if agent_summary.get('agents_used'):
                    st.write("**사용된 에이전트들**:")
                    for agent in agent_summary['agents_used']:
                        st.write(f"- {agent}")
    
    def _render_metadata_section(self, expert_answer: Dict[str, Any]):
        """메타데이터 섹션 렌더링"""
        
        with st.expander("📋 메타데이터 및 추적 정보", expanded=False):
            
            # JSON 형태로 전체 메타데이터 표시
            metadata_json = {
                "처리_시간": expert_answer.get("processing_time"),
                "신뢰도_점수": expert_answer.get("confidence_score"),
                "합성_전략": expert_answer.get("metadata", {}).get("synthesis_strategy"),
                "생성_시간": datetime.now().isoformat()
            }
            
            st.json(metadata_json)
            
            # 다운로드 버튼
            if st.button("📥 분석 결과 다운로드"):
                self._generate_download_link(expert_answer)
    
    def _render_error_fallback(self, expert_answer: Dict[str, Any]):
        """오류 발생 시 폴백 렌더링"""
        
        st.error("🚨 전문가급 답변 합성 중 오류가 발생했습니다")
        
        error_message = expert_answer.get("error", "알 수 없는 오류")
        fallback_message = expert_answer.get("fallback_message", "")
        
        st.write(f"**오류 내용**: {error_message}")
        
        if fallback_message:
            st.info(fallback_message)
        
        # 기본 정보 표시
        if "user_query" in expert_answer:
            st.write(f"**원본 쿼리**: {expert_answer['user_query']}")
        
        if "processing_time" in expert_answer:
            st.write(f"**처리 시간**: {expert_answer['processing_time']:.2f}초")
    
    def _generate_download_link(self, expert_answer: Dict[str, Any]):
        """분석 결과 다운로드 링크 생성"""
        
        # JSON 형태로 결과 직렬화
        download_data = {
            "분석_결과": expert_answer,
            "생성_시간": datetime.now().isoformat(),
            "버전": "CherryAI Expert Analysis v1.0"
        }
        
        json_str = json.dumps(download_data, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        
        href = f'<a href="data:file/json;base64,{b64}" download="cherryai_expert_analysis.json">분석 결과 JSON 다운로드</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _load_style_config(self) -> Dict[str, Any]:
        """스타일 설정 로드"""
        return {
            "primary_color": "#667eea",
            "secondary_color": "#764ba2",
            "accent_color": "#4ECDC4",
            "success_color": "#2ECC71",
            "warning_color": "#F39C12",
            "error_color": "#E74C3C"
        } 