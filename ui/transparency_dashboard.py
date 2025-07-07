#!/usr/bin/env python3
"""
Transparency Dashboard UI Component
실시간 멀티에이전트 시스템 투명성 분석 대시보드
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timedelta
import numpy as np

# 최신 연구 기반 컴포넌트 import
try:
    from core.enhanced_tracing_system import (
        enhanced_tracer, ComponentSynergyScore, ToolUtilizationEfficacy,
        TraceLevel, IssueType
    )
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

class TransparencyDashboard:
    """투명성 분석 대시보드"""
    
    def __init__(self):
        self.dashboard_id = "transparency_dashboard"
        
    def render_comprehensive_analysis(self, 
                                    trace_analysis: Dict[str, Any],
                                    agent_results: List[Dict[str, Any]],
                                    query_info: Dict[str, Any]) -> None:
        """종합 투명성 분석 렌더링"""
        
        st.markdown("## 🔍 **멀티에이전트 시스템 투명성 분석**")
        
        # 핵심 투명성 지표 요약
        self._render_transparency_metrics_summary(trace_analysis)
        
        # 탭 구성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 분석 품질", "🤝 에이전트 협업", "🔧 도구 효율성", 
            "📊 실행 플로우", "🔎 상세 추적"
        ])
        
        with tab1:
            self._render_analysis_quality_tab(trace_analysis, agent_results, query_info)
            
        with tab2:
            self._render_agent_collaboration_tab(trace_analysis)
            
        with tab3:
            self._render_tool_efficiency_tab(trace_analysis)
            
        with tab4:
            self._render_execution_flow_tab(trace_analysis)
            
        with tab5:
            self._render_detailed_tracing_tab(trace_analysis)
    
    def _render_transparency_metrics_summary(self, trace_analysis: Dict[str, Any]) -> None:
        """투명성 지표 요약 렌더링"""
        
        st.markdown("### 📊 **핵심 투명성 지표**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        transparency_metrics = trace_analysis.get("transparency_metrics", {})
        css_metrics = transparency_metrics.get("component_synergy_score", {})
        tue_metrics = transparency_metrics.get("tool_utilization_efficacy", {})
        
        with col1:
            css_score = css_metrics.get("css", 0.0)
            st.metric(
                "🤝 협업 품질 (CSS)",
                f"{css_score:.1%}",
                delta=f"{css_score - 0.75:.1%}" if css_score > 0.75 else None,
                help="Component Synergy Score - 에이전트 간 협업 품질 지표"
            )
            
        with col2:
            tue_score = tue_metrics.get("tue", 0.0)
            st.metric(
                "🔧 도구 효율성 (TUE)",
                f"{tue_score:.1%}",
                delta=f"{tue_score - 0.8:.1%}" if tue_score > 0.8 else None,
                help="Tool Utilization Efficacy - 도구 사용 효율성 지표"
            )
            
        with col3:
            success_rate = trace_analysis.get("summary", {}).get("success_rate", 0.0)
            st.metric(
                "✅ 성공률",
                f"{success_rate:.1%}",
                delta=f"{success_rate - 0.9:.1%}" if success_rate > 0.9 else None,
                help="전체 실행 단계 성공률"
            )
            
        with col4:
            issues_detected = transparency_metrics.get("issues_detected", 0)
            st.metric(
                "⚠️ 이슈 감지",
                f"{issues_detected}개",
                delta=f"-{issues_detected}" if issues_detected == 0 else None,
                help="TRAIL 프레임워크 기반 이슈 감지 수"
            )
        
        # 실시간 투명성 게이지
        self._render_transparency_gauge(css_score, tue_score, success_rate)
    
    def _render_transparency_gauge(self, css_score: float, tue_score: float, success_rate: float) -> None:
        """투명성 종합 게이지 차트"""
        
        # 종합 투명성 점수 계산
        transparency_score = (css_score * 0.3 + tue_score * 0.3 + success_rate * 0.4)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = transparency_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "🔍 종합 투명성 점수"},
            delta = {'reference': 85, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_analysis_quality_tab(self, 
                                   trace_analysis: Dict[str, Any],
                                   agent_results: List[Dict[str, Any]],
                                   query_info: Dict[str, Any]) -> None:
        """분석 품질 탭 렌더링"""
        
        st.markdown("### 🎯 **분석 품질 평가**")
        
        # 쿼리 복잡도 분석
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 **쿼리 특성 분석**")
            
            query_text = query_info.get("original_query", "")
            query_complexity = self._calculate_query_complexity(query_text)
            
            complexity_data = {
                "지표": ["문자 수", "전문 용어", "복잡한 구조", "도메인 깊이"],
                "점수": [
                    min(len(query_text) / 100, 1.0),
                    query_complexity.get("technical_terms", 0.5),
                    query_complexity.get("structural_complexity", 0.6),
                    query_complexity.get("domain_depth", 0.7)
                ]
            }
            
            df_complexity = pd.DataFrame(complexity_data)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=df_complexity["점수"],
                theta=df_complexity["지표"],
                fill='toself',
                name='쿼리 복잡도'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.markdown("#### 🧠 **에이전트 성능 분석**")
            
            agent_performance = trace_analysis.get("agent_performance", {})
            
            if agent_performance:
                performance_data = []
                for agent_id, perf in agent_performance.items():
                    error_rate = perf["errors"] / max(perf["spans"], 1)
                    avg_duration = perf["duration"] / max(perf["spans"], 1)
                    
                    performance_data.append({
                        "에이전트": agent_id.replace("_", " ").title(),
                        "오류율": error_rate,
                        "평균 응답시간": avg_duration,
                        "작업 수": perf["spans"]
                    })
                
                df_performance = pd.DataFrame(performance_data)
                
                fig_scatter = px.scatter(
                    df_performance, 
                    x="평균 응답시간", 
                    y="오류율",
                    size="작업 수",
                    hover_name="에이전트",
                    title="에이전트 성능 매트릭스"
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("에이전트 성능 데이터가 없습니다.")
        
        # 결과 품질 분석
        st.markdown("#### 📈 **결과 품질 분석**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # 신뢰도 분포
            confidence_scores = []
            for result in agent_results:
                if isinstance(result, dict) and "confidence" in result:
                    confidence_scores.append(result["confidence"])
                elif hasattr(result, 'confidence_score'):
                    confidence_scores.append(result.confidence_score)
                else:
                    confidence_scores.append(0.8)  # 기본값
            
            if confidence_scores:
                fig_hist = px.histogram(
                    x=confidence_scores,
                    nbins=10,
                    title="신뢰도 점수 분포",
                    labels={'x': '신뢰도', 'y': '빈도'}
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col4:
            # 품질 지표 레이더 차트
            quality_metrics = {
                "정확성": np.mean(confidence_scores) if confidence_scores else 0.8,
                "완전성": len(agent_results) / 5.0 if len(agent_results) <= 5 else 1.0,
                "일관성": 1.0 - np.std(confidence_scores) if confidence_scores else 0.8,
                "관련성": 0.85,  # 도메인 관련성 (계산 필요)
                "명확성": 0.82   # 답변 명확성 (계산 필요)
            }
            
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Scatterpolar(
                r=list(quality_metrics.values()),
                theta=list(quality_metrics.keys()),
                fill='toself',
                name='품질 지표'
            ))
            
            fig_quality.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=300,
                title="종합 품질 지표"
            )
            
            st.plotly_chart(fig_quality, use_container_width=True)
    
    def _render_agent_collaboration_tab(self, trace_analysis: Dict[str, Any]) -> None:
        """에이전트 협업 탭 렌더링"""
        
        st.markdown("### 🤝 **에이전트 협업 분석**")
        
        transparency_metrics = trace_analysis.get("transparency_metrics", {})
        css_metrics = transparency_metrics.get("component_synergy_score", {})
        interaction_flow = trace_analysis.get("interaction_flow", [])
        
        # 협업 품질 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cooperation_quality = css_metrics.get("cooperation_quality", 0.0)
            st.metric(
                "🤝 협업 품질",
                f"{cooperation_quality:.1%}",
                help="성공적인 상호작용 비율"
            )
        
        with col2:
            communication_efficiency = css_metrics.get("communication_efficiency", 0.0)
            st.metric(
                "💬 소통 효율성",
                f"{communication_efficiency:.1%}",
                help="응답 시간 기반 소통 효율성"
            )
        
        with col3:
            task_distribution = css_metrics.get("task_distribution", 0.0)
            st.metric(
                "⚖️ 업무 분배",
                f"{task_distribution:.1%}",
                help="에이전트 간 균형잡힌 업무 분배"
            )
        
        # 상호작용 네트워크 시각화
        if interaction_flow:
            st.markdown("#### 🌐 **에이전트 상호작용 네트워크**")
            
            # 네트워크 그래프 생성
            agents = set()
            edges = []
            
            for interaction in interaction_flow:
                source = interaction["source_agent"]
                target = interaction["target_agent"]
                agents.add(source)
                agents.add(target)
                edges.append((source, target, interaction["type"]))
            
            # 네트워크 시각화 (간단한 버전)
            agent_list = list(agents)
            interaction_matrix = np.zeros((len(agent_list), len(agent_list)))
            
            for source, target, _ in edges:
                i = agent_list.index(source)
                j = agent_list.index(target)
                interaction_matrix[i][j] += 1
            
            fig_network = px.imshow(
                interaction_matrix,
                x=agent_list,
                y=agent_list,
                title="에이전트 간 상호작용 매트릭스",
                color_continuous_scale="Blues"
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
        
        # 시간대별 상호작용 흐름
        if interaction_flow:
            st.markdown("#### ⏱️ **시간대별 상호작용 흐름**")
            
            # 상호작용 타임라인
            timeline_data = []
            for i, interaction in enumerate(interaction_flow):
                timeline_data.append({
                    "순서": i + 1,
                    "시간": datetime.fromtimestamp(interaction["timestamp"]),
                    "상호작용": f"{interaction['source_agent']} → {interaction['target_agent']}",
                    "타입": interaction["type"],
                    "데이터 크기": interaction["data_summary"]["input_size"]
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            
            fig_timeline = px.line(
                df_timeline,
                x="시간",
                y="데이터 크기",
                hover_data=["상호작용", "타입"],
                title="상호작용 타임라인",
                markers=True
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    def _render_tool_efficiency_tab(self, trace_analysis: Dict[str, Any]) -> None:
        """도구 효율성 탭 렌더링"""
        
        st.markdown("### 🔧 **도구 사용 효율성 분석**")
        
        transparency_metrics = trace_analysis.get("transparency_metrics", {})
        tue_metrics = transparency_metrics.get("tool_utilization_efficacy", {})
        
        # TUE 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_rate = tue_metrics.get("success_rate", 0.0)
            st.metric(
                "✅ 성공률",
                f"{success_rate:.1%}",
                help="도구 호출 성공률"
            )
        
        with col2:
            avg_response_time = tue_metrics.get("avg_response_time", 0.0)
            st.metric(
                "⚡ 평균 응답시간",
                f"{avg_response_time:.2f}s",
                help="도구 실행 평균 응답시간"
            )
        
        with col3:
            resource_efficiency = tue_metrics.get("resource_efficiency", 0.0)
            st.metric(
                "🎯 리소스 효율성",
                f"{resource_efficiency:.3f}",
                help="토큰 사용량 대비 성공률"
            )
        
        # 도구별 성능 분석
        spans_hierarchy = trace_analysis.get("spans_hierarchy", {})
        tool_performance = self._extract_tool_performance(spans_hierarchy)
        
        if tool_performance:
            st.markdown("#### 🛠️ **도구별 성능 분석**")
            
            df_tools = pd.DataFrame(tool_performance)
            
            col4, col5 = st.columns(2)
            
            with col4:
                # 도구별 성공률
                fig_success = px.bar(
                    df_tools,
                    x="도구명",
                    y="성공률",
                    title="도구별 성공률",
                    color="성공률",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_success, use_container_width=True)
            
            with col5:
                # 도구별 응답시간
                fig_time = px.box(
                    df_tools,
                    x="도구명",
                    y="평균 응답시간",
                    title="도구별 응답시간 분포"
                )
                st.plotly_chart(fig_time, use_container_width=True)
    
    def _render_execution_flow_tab(self, trace_analysis: Dict[str, Any]) -> None:
        """실행 플로우 탭 렌더링"""
        
        st.markdown("### 📊 **실행 플로우 분석**")
        
        spans_hierarchy = trace_analysis.get("spans_hierarchy", {})
        
        if spans_hierarchy:
            # 실행 플로우 시각화
            self._render_execution_flow_chart(spans_hierarchy)
            
            # 플로우 통계
            st.markdown("#### 📈 **플로우 통계**")
            
            flow_stats = self._calculate_flow_statistics(spans_hierarchy)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 단계", flow_stats["total_steps"])
            with col2:
                st.metric("병렬 실행", flow_stats["parallel_steps"])
            with col3:
                st.metric("최대 깊이", flow_stats["max_depth"])
            with col4:
                st.metric("평균 단계 시간", f"{flow_stats['avg_step_time']:.2f}s")
    
    def _render_detailed_tracing_tab(self, trace_analysis: Dict[str, Any]) -> None:
        """상세 추적 탭 렌더링"""
        
        st.markdown("### 🔎 **상세 추적 정보**")
        
        # 원시 트레이스 데이터 표시
        with st.expander("📋 **원시 트레이스 데이터**"):
            st.json(trace_analysis)
        
        # 이슈 분석
        transparency_metrics = trace_analysis.get("transparency_metrics", {})
        issues_detected = transparency_metrics.get("issues_detected", 0)
        issue_types = transparency_metrics.get("issue_types", [])
        
        if issues_detected > 0:
            st.markdown("#### ⚠️ **감지된 이슈**")
            
            for issue_type in issue_types:
                st.warning(f"🚨 {issue_type}: {self._get_issue_description(issue_type)}")
        else:
            st.success("✅ 감지된 이슈가 없습니다.")
        
        # 성능 개선 제안
        st.markdown("#### 💡 **성능 개선 제안**")
        
        recommendations = self._generate_recommendations(trace_analysis)
        
        for recommendation in recommendations:
            st.info(f"💡 {recommendation}")
    
    def _calculate_query_complexity(self, query_text: str) -> Dict[str, float]:
        """쿼리 복잡도 계산"""
        
        technical_terms = ["공정", "TW", "이온주입", "반도체", "계측", "장비", "레시피"]
        technical_score = sum(1 for term in technical_terms if term in query_text) / len(technical_terms)
        
        structural_indicators = ["분석", "판단", "제안", "원인", "조치"]
        structural_score = sum(1 for indicator in structural_indicators if indicator in query_text) / len(structural_indicators)
        
        domain_keywords = ["엔지니어", "도메인", "히스토리", "셋팅", "데이터"]
        domain_score = sum(1 for keyword in domain_keywords if keyword in query_text) / len(domain_keywords)
        
        return {
            "technical_terms": technical_score,
            "structural_complexity": structural_score,
            "domain_depth": domain_score
        }
    
    def _extract_tool_performance(self, spans_hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """스팬 계층에서 도구 성능 추출"""
        
        tool_performance = []
        
        def extract_from_node(node):
            span = node.get("span")
            if span and hasattr(span, 'level') and span.level.value == "tool":
                tool_performance.append({
                    "도구명": getattr(span, 'tool_name', 'Unknown'),
                    "성공률": 1.0 if not getattr(span, 'error', None) else 0.0,
                    "평균 응답시간": getattr(span, 'duration', 0.0) or 0.0,
                    "토큰 사용량": getattr(span, 'token_usage', {}).get('total_tokens', 0) if getattr(span, 'token_usage', None) else 0
                })
            
            for child in node.get("children", []):
                extract_from_node(child)
        
        for root_node in spans_hierarchy.values():
            extract_from_node(root_node)
        
        return tool_performance
    
    def _render_execution_flow_chart(self, spans_hierarchy: Dict[str, Any]) -> None:
        """실행 플로우 차트 렌더링"""
        
        # 간단한 플로우차트 생성 (Gantt 차트 스타일)
        flow_data = []
        
        def extract_flow_data(node, level=0):
            span = node.get("span")
            if span:
                start_time = getattr(span, 'start_time', 0)
                end_time = getattr(span, 'end_time', start_time + 1)
                
                flow_data.append({
                    "작업": getattr(span, 'name', 'Unknown'),
                    "시작": datetime.fromtimestamp(start_time),
                    "종료": datetime.fromtimestamp(end_time),
                    "레벨": level,
                    "타입": getattr(span, 'level', 'unknown').value if hasattr(getattr(span, 'level', None), 'value') else 'unknown'
                })
            
            for child in node.get("children", []):
                extract_flow_data(child, level + 1)
        
        for root_node in spans_hierarchy.values():
            extract_flow_data(root_node)
        
        if flow_data:
            df_flow = pd.DataFrame(flow_data)
            
            fig_gantt = px.timeline(
                df_flow,
                x_start="시작",
                x_end="종료",
                y="작업",
                color="타입",
                title="실행 플로우 타임라인"
            )
            
            st.plotly_chart(fig_gantt, use_container_width=True)
    
    def _calculate_flow_statistics(self, spans_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """플로우 통계 계산"""
        
        total_steps = 0
        max_depth = 0
        step_times = []
        
        def analyze_node(node, depth=0):
            nonlocal total_steps, max_depth
            
            total_steps += 1
            max_depth = max(max_depth, depth)
            
            span = node.get("span")
            if span and hasattr(span, 'duration') and span.duration:
                step_times.append(span.duration)
            
            for child in node.get("children", []):
                analyze_node(child, depth + 1)
        
        for root_node in spans_hierarchy.values():
            analyze_node(root_node)
        
        return {
            "total_steps": total_steps,
            "parallel_steps": len(spans_hierarchy),  # 루트 노드 수
            "max_depth": max_depth,
            "avg_step_time": np.mean(step_times) if step_times else 0.0
        }
    
    def _get_issue_description(self, issue_type: str) -> str:
        """이슈 타입별 설명"""
        
        descriptions = {
            "coordination_failure": "에이전트 간 협조 실패가 감지되었습니다.",
            "tool_misuse": "도구 사용 오류가 발생했습니다.",
            "reasoning_error": "추론 과정에서 논리적 오류가 감지되었습니다.",
            "context_loss": "컨텍스트 손실이 발생했습니다.",
            "hallucination": "환각 현상이 감지되었습니다.",
            "performance_degradation": "성능 저하가 감지되었습니다."
        }
        
        return descriptions.get(issue_type, "알 수 없는 이슈입니다.")
    
    def _generate_recommendations(self, trace_analysis: Dict[str, Any]) -> List[str]:
        """성능 개선 제안 생성"""
        
        recommendations = []
        
        transparency_metrics = trace_analysis.get("transparency_metrics", {})
        css_metrics = transparency_metrics.get("component_synergy_score", {})
        tue_metrics = transparency_metrics.get("tool_utilization_efficacy", {})
        
        # CSS 기반 제안
        if css_metrics.get("cooperation_quality", 0) < 0.7:
            recommendations.append("에이전트 간 협업 프로토콜을 개선하세요.")
        
        if css_metrics.get("communication_efficiency", 0) < 0.6:
            recommendations.append("에이전트 간 통신 지연을 최적화하세요.")
        
        if css_metrics.get("task_distribution", 0) < 0.5:
            recommendations.append("업무 분배 알고리즘을 조정하세요.")
        
        # TUE 기반 제안
        if tue_metrics.get("success_rate", 0) < 0.8:
            recommendations.append("도구 안정성을 개선하세요.")
        
        if tue_metrics.get("avg_response_time", 0) > 5.0:
            recommendations.append("도구 응답 시간을 최적화하세요.")
        
        if not recommendations:
            recommendations.append("시스템이 우수한 성능을 보이고 있습니다. 현재 설정을 유지하세요.")
        
        return recommendations

# 전역 대시보드 인스턴스
transparency_dashboard = TransparencyDashboard()

# 사용 예제 함수
def render_transparency_analysis(trace_analysis: Dict[str, Any], 
                               agent_results: List[Dict[str, Any]] = None,
                               query_info: Dict[str, Any] = None) -> None:
    """투명성 분석 렌더링 함수"""
    
    if not trace_analysis:
        st.warning("🔍 투명성 분석 데이터가 없습니다.")
        return
    
    agent_results = agent_results or []
    query_info = query_info or {"original_query": "분석 쿼리"}
    
    transparency_dashboard.render_comprehensive_analysis(
        trace_analysis, agent_results, query_info
    ) 