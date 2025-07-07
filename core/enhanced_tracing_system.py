#!/usr/bin/env python3
"""
Enhanced Tracing System for Multi-Agent Transparency
Based on TRAIL (Trace Reasoning and Agentic Issue Localization) research
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import contextvars

# 현재 trace context 관리
current_trace_context = contextvars.ContextVar('current_trace_context', default=None)

class TraceLevel(Enum):
    """트레이스 레벨 정의"""
    SYSTEM = "system"
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    INTEGRATION = "integration"

class IssueType(Enum):
    """TRAIL 연구 기반 이슈 타입"""
    COORDINATION_FAILURE = "coordination_failure"
    TOOL_MISUSE = "tool_misuse"
    REASONING_ERROR = "reasoning_error"
    CONTEXT_LOSS = "context_loss"
    HALLUCINATION = "hallucination"
    PERFORMANCE_DEGRADATION = "performance_degradation"

@dataclass
class TraceSpan:
    """개별 트레이스 스팬"""
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    level: TraceLevel
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    llm_model: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    issues: List[IssueType] = field(default_factory=list)
    
    def end_span(self, output_data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """스팬 종료 및 duration 계산"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if output_data:
            self.output_data = output_data
        if error:
            self.error = error

@dataclass
class AgentInteraction:
    """에이전트 간 상호작용 기록"""
    interaction_id: str
    source_agent: str
    target_agent: str
    interaction_type: str  # "request", "response", "delegation", "collaboration"
    timestamp: float
    data: Dict[str, Any]
    correlation_id: str

class ComponentSynergyScore:
    """에이전트 간 협업 품질 정량화 (CSS)"""
    
    @staticmethod
    def calculate_css(interactions: List[AgentInteraction], 
                     execution_time: float, 
                     success_rate: float) -> Dict[str, float]:
        """
        Component Synergy Score 계산
        - Cooperation Quality: 협업 품질
        - Communication Efficiency: 소통 효율성
        - Task Distribution: 업무 분배 균형
        """
        if not interactions:
            return {"css": 0.0, "cooperation_quality": 0.0, "communication_efficiency": 0.0, "task_distribution": 0.0}
        
        # 협업 품질 계산
        successful_interactions = sum(1 for i in interactions if i.interaction_type in ["response", "collaboration"])
        cooperation_quality = successful_interactions / len(interactions)
        
        # 소통 효율성 계산 (응답 시간 기반)
        response_times = []
        for i in range(len(interactions) - 1):
            if interactions[i].interaction_type == "request" and interactions[i+1].interaction_type == "response":
                response_times.append(interactions[i+1].timestamp - interactions[i].timestamp)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        communication_efficiency = 1.0 / (1.0 + avg_response_time) if avg_response_time > 0 else 1.0
        
        # 업무 분배 균형 계산
        agent_workload = defaultdict(int)
        for interaction in interactions:
            agent_workload[interaction.source_agent] += 1
        
        workload_variance = sum((count - len(interactions) / len(agent_workload))**2 for count in agent_workload.values()) / len(agent_workload)
        task_distribution = 1.0 / (1.0 + workload_variance)
        
        # 최종 CSS 계산
        css = (cooperation_quality * 0.4 + communication_efficiency * 0.3 + task_distribution * 0.3) * success_rate
        
        return {
            "css": css,
            "cooperation_quality": cooperation_quality,
            "communication_efficiency": communication_efficiency,
            "task_distribution": task_distribution
        }

class ToolUtilizationEfficacy:
    """도구 사용 효율성 평가 (TUE)"""
    
    @staticmethod
    def calculate_tue(tool_usage: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Tool Utilization Efficacy 계산
        - Success Rate: 도구 호출 성공률
        - Response Time: 평균 응답 시간
        - Resource Efficiency: 리소스 효율성
        """
        if not tool_usage:
            return {"tue": 0.0, "success_rate": 0.0, "avg_response_time": 0.0, "resource_efficiency": 0.0}
        
        successful_calls = sum(1 for usage in tool_usage if usage.get("success", False))
        success_rate = successful_calls / len(tool_usage)
        
        response_times = [usage.get("duration", 0) for usage in tool_usage]
        avg_response_time = sum(response_times) / len(response_times)
        
        # 리소스 효율성 (토큰 사용량 기반)
        token_usage = sum(usage.get("token_count", 0) for usage in tool_usage)
        resource_efficiency = successful_calls / max(token_usage, 1)
        
        # 최종 TUE 계산
        tue = (success_rate * 0.5 + (1.0 / max(avg_response_time, 0.1)) * 0.3 + resource_efficiency * 0.2)
        
        return {
            "tue": tue,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "resource_efficiency": resource_efficiency
        }

class EnhancedTracingSystem:
    """향상된 트레이싱 시스템"""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.interactions: Dict[str, List[AgentInteraction]] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self.logger = logging.getLogger(__name__)
        
    def create_trace(self, trace_name: str, user_id: str = None, session_id: str = None) -> str:
        """새로운 트레이스 생성"""
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = []
        self.interactions[trace_id] = []
        
        # 시스템 레벨 루트 스팬 생성
        root_span = TraceSpan(
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            trace_id=trace_id,
            name=trace_name,
            level=TraceLevel.SYSTEM,
            start_time=time.time(),
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "trace_name": trace_name
            }
        )
        
        self.traces[trace_id].append(root_span)
        self.active_spans[trace_id] = root_span
        
        # Context 설정
        current_trace_context.set(trace_id)
        
        return trace_id
    
    def start_span(self, 
                   name: str, 
                   level: TraceLevel, 
                   parent_span_id: Optional[str] = None,
                   agent_id: Optional[str] = None,
                   tool_name: Optional[str] = None,
                   llm_model: Optional[str] = None,
                   input_data: Optional[Dict[str, Any]] = None) -> str:
        """새로운 스팬 시작"""
        trace_id = current_trace_context.get()
        if not trace_id:
            raise ValueError("No active trace context found")
        
        span_id = str(uuid.uuid4())
        
        # 부모 스팬 ID 자동 결정
        if parent_span_id is None and trace_id in self.active_spans:
            parent_span_id = self.active_spans[trace_id].span_id
        
        span = TraceSpan(
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            name=name,
            level=level,
            start_time=time.time(),
            input_data=input_data,
            agent_id=agent_id,
            tool_name=tool_name,
            llm_model=llm_model
        )
        
        self.traces[trace_id].append(span)
        self.active_spans[trace_id] = span
        
        return span_id
    
    def end_span(self, span_id: str, output_data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """스팬 종료"""
        trace_id = current_trace_context.get()
        if not trace_id:
            return
        
        for span in self.traces[trace_id]:
            if span.span_id == span_id:
                span.end_span(output_data=output_data, error=error)
                break
    
    def record_interaction(self, 
                          source_agent: str, 
                          target_agent: str, 
                          interaction_type: str,
                          data: Dict[str, Any],
                          correlation_id: Optional[str] = None) -> str:
        """에이전트 간 상호작용 기록"""
        trace_id = current_trace_context.get()
        if not trace_id:
            return None
        
        interaction_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            timestamp=time.time(),
            data=data,
            correlation_id=correlation_id
        )
        
        self.interactions[trace_id].append(interaction)
        return interaction_id
    
    def analyze_trace(self, trace_id: str) -> Dict[str, Any]:
        """트레이스 분석 및 투명성 지표 계산"""
        if trace_id not in self.traces:
            return {}
        
        spans = self.traces[trace_id]
        interactions = self.interactions[trace_id]
        
        # 기본 통계
        # 안전한 duration 계산
        end_times = [span.end_time for span in spans if span.end_time]
        start_times = [span.start_time for span in spans if span.start_time]
        
        if end_times and start_times:
            total_duration = max(end_times) - min(start_times)
        else:
            total_duration = 0.0
        success_rate = sum(1 for span in spans if span.error is None) / len(spans)
        
        # CSS 계산
        css_metrics = ComponentSynergyScore.calculate_css(interactions, total_duration, success_rate)
        
        # TUE 계산
        tool_usage = []
        for span in spans:
            if span.level == TraceLevel.TOOL and span.tool_name:
                tool_usage.append({
                    "tool_name": span.tool_name,
                    "duration": span.duration,
                    "success": span.error is None,
                    "token_count": span.token_usage.get("total_tokens", 0) if span.token_usage else 0
                })
        
        tue_metrics = ToolUtilizationEfficacy.calculate_tue(tool_usage)
        
        # 이슈 분석
        issues = []
        for span in spans:
            if span.error:
                issues.extend(span.issues)
        
        # 에이전트별 성능 분석
        agent_performance = defaultdict(lambda: {"spans": 0, "errors": 0, "duration": 0})
        for span in spans:
            if span.agent_id:
                agent_performance[span.agent_id]["spans"] += 1
                if span.error:
                    agent_performance[span.agent_id]["errors"] += 1
                if span.duration:
                    agent_performance[span.agent_id]["duration"] += span.duration
        
        return {
            "trace_id": trace_id,
            "summary": {
                "total_spans": len(spans),
                "total_duration": total_duration,
                "success_rate": success_rate,
                "total_interactions": len(interactions)
            },
            "transparency_metrics": {
                "component_synergy_score": css_metrics,
                "tool_utilization_efficacy": tue_metrics,
                "issues_detected": len(issues),
                "issue_types": list(set(issues))
            },
            "agent_performance": dict(agent_performance),
            "spans_hierarchy": self._build_span_hierarchy(spans),
            "interaction_flow": self._build_interaction_flow(interactions)
        }
    
    def _build_span_hierarchy(self, spans: List[TraceSpan]) -> Dict[str, Any]:
        """스팬 계층 구조 생성"""
        hierarchy = {}
        span_map = {span.span_id: span for span in spans}
        
        for span in spans:
            if span.parent_span_id is None:
                hierarchy[span.span_id] = {
                    "span": span,
                    "children": []
                }
        
        # 자식 스팬 추가
        for span in spans:
            if span.parent_span_id and span.parent_span_id in span_map:
                parent_node = self._find_node_in_hierarchy(hierarchy, span.parent_span_id)
                if parent_node:
                    parent_node["children"].append({
                        "span": span,
                        "children": []
                    })
        
        return hierarchy
    
    def _find_node_in_hierarchy(self, hierarchy: Dict[str, Any], span_id: str) -> Optional[Dict[str, Any]]:
        """계층 구조에서 노드 찾기"""
        for node in hierarchy.values():
            if node["span"].span_id == span_id:
                return node
            result = self._find_node_in_children(node["children"], span_id)
            if result:
                return result
        return None
    
    def _find_node_in_children(self, children: List[Dict[str, Any]], span_id: str) -> Optional[Dict[str, Any]]:
        """자식 노드에서 찾기"""
        for child in children:
            if child["span"].span_id == span_id:
                return child
            result = self._find_node_in_children(child["children"], span_id)
            if result:
                return result
        return None
    
    def _build_interaction_flow(self, interactions: List[AgentInteraction]) -> List[Dict[str, Any]]:
        """상호작용 플로우 생성"""
        return [
            {
                "interaction_id": interaction.interaction_id,
                "source_agent": interaction.source_agent,
                "target_agent": interaction.target_agent,
                "type": interaction.interaction_type,
                "timestamp": interaction.timestamp,
                "correlation_id": interaction.correlation_id,
                "data_summary": {
                    "input_size": len(str(interaction.data.get("input", ""))),
                    "output_size": len(str(interaction.data.get("output", ""))),
                    "has_error": "error" in interaction.data
                }
            }
            for interaction in sorted(interactions, key=lambda x: x.timestamp)
        ]
    
    def export_trace(self, trace_id: str, format: str = "json") -> str:
        """트레이스 내보내기"""
        analysis = self.analyze_trace(trace_id)
        
        if format == "json":
            return json.dumps(analysis, indent=2, default=str)
        elif format == "langfuse":
            return self._export_to_langfuse_format(analysis)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_langfuse_format(self, analysis: Dict[str, Any]) -> str:
        """Langfuse 포맷으로 내보내기"""
        # Langfuse 호환 포맷 생성
        langfuse_format = {
            "trace_id": analysis["trace_id"],
            "name": "CherryAI_Enhanced_Trace",
            "input": {"query": "Multi-agent trace analysis"},
            "output": analysis["summary"],
            "metadata": {
                "transparency_metrics": analysis["transparency_metrics"],
                "agent_performance": analysis["agent_performance"]
            },
            "spans": []
        }
        
        # 스팬 정보 추가
        for span_id, node in analysis["spans_hierarchy"].items():
            langfuse_format["spans"].append({
                "id": span_id,
                "name": node["span"].name,
                "level": node["span"].level.value,
                "start_time": node["span"].start_time,
                "end_time": node["span"].end_time,
                "duration": node["span"].duration,
                "input": node["span"].input_data,
                "output": node["span"].output_data,
                "metadata": node["span"].metadata
            })
        
        return json.dumps(langfuse_format, indent=2, default=str)

# 전역 트레이싱 시스템 인스턴스
enhanced_tracer = EnhancedTracingSystem()

# 컨텍스트 매니저 지원
class TraceContext:
    """트레이스 컨텍스트 매니저"""
    
    def __init__(self, trace_name: str, user_id: str = None, session_id: str = None):
        self.trace_name = trace_name
        self.user_id = user_id
        self.session_id = session_id
        self.trace_id = None
    
    def __enter__(self):
        self.trace_id = enhanced_tracer.create_trace(self.trace_name, self.user_id, self.session_id)
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace_id:
            # 루트 스팬 종료
            root_span = enhanced_tracer.traces[self.trace_id][0]
            enhanced_tracer.end_span(root_span.span_id)
            
            # 분석 결과 로깅
            analysis = enhanced_tracer.analyze_trace(self.trace_id)
            enhanced_tracer.logger.info(f"Trace Analysis: {json.dumps(analysis, indent=2, default=str)}")

# 사용 예제
def example_usage():
    """사용 예제"""
    
    # 트레이스 시작
    with TraceContext("반도체_이온주입_분석", user_id="engineer_001", session_id="session_123") as trace_id:
        
        # 에이전트 스팬 시작
        agent_span_id = enhanced_tracer.start_span(
            "데이터_분석_에이전트",
            TraceLevel.AGENT,
            agent_id="data_analyst_agent",
            input_data={"query": "TW 이상 분석"}
        )
        
        # 도구 사용 스팬
        tool_span_id = enhanced_tracer.start_span(
            "통계_분석_도구",
            TraceLevel.TOOL,
            tool_name="statistical_analyzer",
            input_data={"data": "TW measurement data"}
        )
        
        # 에이전트 간 상호작용 기록
        enhanced_tracer.record_interaction(
            "data_analyst_agent",
            "process_expert_agent",
            "delegation",
            {"task": "이상 원인 분석", "data": "TW 데이터"}
        )
        
        # 스팬 종료
        enhanced_tracer.end_span(tool_span_id, output_data={"result": "분석 완료"})
        enhanced_tracer.end_span(agent_span_id, output_data={"analysis": "TW 상승 원인 식별"})
        
        print(f"트레이스 분석 완료: {trace_id}")

if __name__ == "__main__":
    example_usage() 