# core/streaming/smart_stream_manager.py
"""
스마트 스트림 관리자 - UI 최적화를 위한 메시지 필터링 및 버퍼링
"""
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import streamlit as st

from ..schemas.messages import StreamMessage, MessageType, AgentType

class UIImportance(Enum):
    """UI 표시 중요도"""
    CRITICAL = "critical"    # 반드시 표시 (최종 응답, 에러)
    HIGH = "high"           # 중요 (에이전트 시작/완료, 주요 결과)
    MEDIUM = "medium"       # 보통 (도구 실행, 진행상황)
    LOW = "low"            # 낮음 (디버그, 상세 로그)
    HIDDEN = "hidden"       # 숨김 (내부 처리)

@dataclass
class MessageBuffer:
    """메시지 버퍼"""
    messages: List[StreamMessage]
    last_update: float
    node_name: str
    importance: UIImportance

class SmartStreamManager:
    """스마트 스트림 관리자"""
    
    def __init__(self, 
                 update_interval: float = 0.5,  # 최소 업데이트 간격
                 max_buffer_size: int = 100,    # 최대 버퍼 크기
                 show_debug: bool = False):     # 디버그 모드
        
        self.update_interval = update_interval
        self.max_buffer_size = max_buffer_size
        self.show_debug = show_debug
        
        # 메시지 버퍼
        self.message_buffers: Dict[str, MessageBuffer] = {}
        
        # 노드별 실행 횟수 추적
        self.node_execution_count: Dict[str, int] = {}
        self.node_last_execution: Dict[str, float] = {}
        
        # UI 상태 추적
        self.current_workflow_status = "idle"
        self.active_agents: Set[str] = set()
        self.completed_steps: List[str] = []
        
        # 중요도 매핑
        self.importance_mapping = {
            # Critical - 반드시 보여야 할 것들
            MessageType.FINAL_RESPONSE: UIImportance.CRITICAL,
            MessageType.DIRECT_RESPONSE: UIImportance.CRITICAL,
            MessageType.ERROR: UIImportance.CRITICAL,
            
            # High - 중요한 진행 상황
            MessageType.AGENT_START: UIImportance.HIGH,
            MessageType.AGENT_END: UIImportance.HIGH,
            MessageType.VISUALIZATION: UIImportance.HIGH,
            
            # Medium - 일반적인 작업 과정
            MessageType.CODE_EXECUTION: UIImportance.MEDIUM,
            MessageType.TOOL_RESULT: UIImportance.MEDIUM,
            MessageType.PROGRESS: UIImportance.MEDIUM,
            
            # Low - 상세 정보
            MessageType.TOOL_CALL: UIImportance.LOW,
        }
        
        # 노드별 중요도 매핑
        self.node_importance = {
            'smart_router': UIImportance.MEDIUM,
            'planner': UIImportance.HIGH,
            'final_responder': UIImportance.CRITICAL,
            'direct_response': UIImportance.CRITICAL,
            'EDA_Analyst': UIImportance.HIGH,
            'replanner': UIImportance.LOW,
            'router': UIImportance.LOW,
        }
    
    def should_display_message(self, message: StreamMessage, node: str = None) -> bool:
        """메시지가 UI에 표시되어야 하는지 판단"""
        
        # 1. 메시지 타입별 중요도 확인
        msg_importance = self.importance_mapping.get(message.message_type, UIImportance.MEDIUM)
        
        # 2. 노드별 중요도 확인
        if node:
            node_importance = self.node_importance.get(node, UIImportance.MEDIUM)
            # 더 높은 중요도로 설정
            if node_importance.value == "critical":
                msg_importance = UIImportance.CRITICAL
            elif node_importance.value == "high" and msg_importance.value not in ["critical"]:
                msg_importance = UIImportance.HIGH
        
        # 3. 디버그 모드가 아니면 LOW 이하는 숨김
        if not self.show_debug and msg_importance in [UIImportance.LOW, UIImportance.HIDDEN]:
            return False
        
        # 4. 중복 노드 실행 확인
        if node and self._is_repetitive_node(node):
            return msg_importance == UIImportance.CRITICAL
        
        return True
    
    def _is_repetitive_node(self, node: str, threshold: int = 3, time_window: float = 10.0) -> bool:
        """반복적인 노드 실행인지 확인"""
        current_time = time.time()
        
        # 실행 횟수 업데이트
        if node not in self.node_execution_count:
            self.node_execution_count[node] = 0
            self.node_last_execution[node] = current_time
        
        # 시간 윈도우 내 실행 횟수 확인
        if current_time - self.node_last_execution[node] < time_window:
            self.node_execution_count[node] += 1
        else:
            # 시간 윈도우 초과시 리셋
            self.node_execution_count[node] = 1
            self.node_last_execution[node] = current_time
        
        return self.node_execution_count[node] > threshold
    
    def add_message_to_buffer(self, message: StreamMessage, node: str = None):
        """메시지를 버퍼에 추가"""
        if not self.should_display_message(message, node):
            return
        
        buffer_key = node or "default"
        current_time = time.time()
        
        if buffer_key not in self.message_buffers:
            self.message_buffers[buffer_key] = MessageBuffer(
                messages=[],
                last_update=current_time,
                node_name=buffer_key,
                importance=self.node_importance.get(buffer_key, UIImportance.MEDIUM)
            )
        
        buffer = self.message_buffers[buffer_key]
        buffer.messages.append(message)
        
        # 버퍼 크기 제한
        if len(buffer.messages) > self.max_buffer_size:
            buffer.messages = buffer.messages[-self.max_buffer_size:]
    
    def should_update_ui(self, buffer_key: str) -> bool:
        """UI 업데이트가 필요한지 확인"""
        if buffer_key not in self.message_buffers:
            return False
        
        buffer = self.message_buffers[buffer_key]
        current_time = time.time()
        
        # 중요한 메시지는 즉시 업데이트
        if buffer.importance in [UIImportance.CRITICAL, UIImportance.HIGH]:
            return True
        
        # 일반 메시지는 간격 제한
        return current_time - buffer.last_update >= self.update_interval
    
    def get_display_summary(self, node: str) -> Optional[str]:
        """노드별 표시 요약 생성"""
        if node not in self.message_buffers:
            return None
        
        buffer = self.message_buffers[node]
        if not buffer.messages:
            return None
        
        latest_message = buffer.messages[-1]
        
        # 노드별 커스텀 요약
        if node == "planner":
            return f"📋 계획 수립 완료 ({len(buffer.messages)}개 단계)"
        elif node == "EDA_Analyst":
            return f"📊 데이터 분석 진행 중... ({len(buffer.messages)}개 작업)"
        elif node == "router":
            return f"🔀 작업 라우팅 ({len(buffer.messages)}회)"
        else:
            return f"⚙️ {node} 처리 중... ({len(buffer.messages)}개 메시지)"
    
    def clear_buffer(self, node: str = None):
        """버퍼 클리어"""
        if node:
            if node in self.message_buffers:
                del self.message_buffers[node]
        else:
            self.message_buffers.clear()
            self.node_execution_count.clear()
            self.node_last_execution.clear()

# 전역 인스턴스
smart_stream_manager = SmartStreamManager(
    update_interval=0.5,
    max_buffer_size=50,
    show_debug=False  # 프로덕션에서는 False
)