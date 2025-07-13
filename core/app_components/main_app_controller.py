#!/usr/bin/env python3
"""
🍒 CherryAI Main App Controller

main.py에서 비즈니스 로직을 분리한 메인 앱 컨트롤러
- 시스템 전체 상태 관리
- 세션 및 설정 관리  
- 에러 처리 및 로깅
- A2A + MCP 통합 시스템 조율
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st

# 통합 시스템 임포트
from core.streaming.unified_message_broker import get_unified_message_broker
from core.streaming.streaming_orchestrator import get_streaming_orchestrator, StreamingConfig, ChatStyle

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """시스템 상태"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class AppSession:
    """앱 세션 정보"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    uploaded_files: List[Any] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    broker_session_id: Optional[str] = None

@dataclass
class SystemHealth:
    """시스템 건강 상태"""
    status: SystemStatus
    a2a_agents_online: int = 0
    a2a_agents_total: int = 11
    mcp_tools_online: int = 0
    mcp_tools_total: int = 7
    broker_initialized: bool = False
    orchestrator_initialized: bool = False
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None

class MainAppController:
    """메인 앱 컨트롤러 - 시스템 전체 관리"""
    
    def __init__(self):
        # 시스템 상태
        self.system_health = SystemHealth(status=SystemStatus.INITIALIZING)
        self.current_session: Optional[AppSession] = None
        
        # 통합 시스템 컴포넌트들
        self.unified_broker = None
        self.streaming_orchestrator = None
        
        # 설정
        self.config = {
            'chat_style': ChatStyle.CHATGPT,
            'enable_real_time_streaming': True,
            'show_agent_names': True,
            'enable_typing_indicator': True,
            'max_message_history': 100,
            'session_timeout_minutes': 60
        }
        
        # 통계
        self.stats = {
            'total_sessions': 0,
            'total_messages': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'uptime_start': datetime.now()
        }
    
    async def initialize_system(self) -> bool:
        """시스템 전체 초기화"""
        try:
            logger.info("🚀 CherryAI 시스템 초기화 시작...")
            self.system_health.status = SystemStatus.INITIALIZING
            
            # 1. UnifiedMessageBroker 초기화
            try:
                self.unified_broker = get_unified_message_broker()
                await self.unified_broker.initialize()
                self.system_health.broker_initialized = True
                logger.info("✅ UnifiedMessageBroker 초기화 완료")
            except Exception as e:
                logger.error(f"❌ UnifiedMessageBroker 초기화 실패: {e}")
                self.system_health.error_message = f"Broker init failed: {str(e)}"
                return False
            
            # 2. StreamingOrchestrator 초기화
            try:
                streaming_config = StreamingConfig(
                    chat_style=self.config['chat_style'],
                    enable_typing_indicator=self.config['enable_typing_indicator'],
                    enable_progress_bar=True,
                    show_agent_names=self.config['show_agent_names']
                )
                self.streaming_orchestrator = get_streaming_orchestrator(streaming_config)
                await self.streaming_orchestrator.initialize()
                self.system_health.orchestrator_initialized = True
                logger.info("✅ StreamingOrchestrator 초기화 완료")
            except Exception as e:
                logger.error(f"❌ StreamingOrchestrator 초기화 실패: {e}")
                self.system_health.error_message = f"Orchestrator init failed: {str(e)}"
                return False
            
            # 3. 시스템 건강 상태 확인
            await self.check_system_health()
            
            self.system_health.status = SystemStatus.READY
            logger.info("🎉 CherryAI 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            logger.error(f"🚨 시스템 초기화 치명적 오류: {e}")
            self.system_health.status = SystemStatus.ERROR
            self.system_health.error_message = f"Critical init error: {str(e)}"
            return False
    
    async def check_system_health(self) -> SystemHealth:
        """시스템 건강 상태 확인"""
        try:
            if not self.unified_broker:
                self.system_health.status = SystemStatus.ERROR
                return self.system_health
            
            # A2A 에이전트들 상태 확인
            a2a_agents = [agent for agent in self.unified_broker.agents.values() 
                         if agent.agent_type.value == "a2a_agent"]
            online_a2a = sum(1 for agent in a2a_agents if agent.status == "online")
            
            # MCP 도구들 상태 확인  
            mcp_tools = [agent for agent in self.unified_broker.agents.values()
                        if agent.agent_type.value in ["mcp_sse", "mcp_stdio"]]
            online_mcp = sum(1 for tool in mcp_tools if tool.status == "online")
            
            # 상태 업데이트
            self.system_health.a2a_agents_online = online_a2a
            self.system_health.a2a_agents_total = len(a2a_agents)
            self.system_health.mcp_tools_online = online_mcp
            self.system_health.mcp_tools_total = len(mcp_tools)
            self.system_health.last_health_check = datetime.now()
            
            # 전체 상태 결정
            if online_a2a == 0 and online_mcp == 0:
                self.system_health.status = SystemStatus.ERROR
            elif online_a2a > 0 or online_mcp > 0:
                self.system_health.status = SystemStatus.READY
            
            return self.system_health
            
        except Exception as e:
            logger.error(f"❌ 시스템 건강 상태 확인 실패: {e}")
            self.system_health.status = SystemStatus.ERROR
            self.system_health.error_message = str(e)
            return self.system_health
    
    def create_session(self) -> AppSession:
        """새 앱 세션 생성"""
        session = AppSession(
            session_id=str(uuid.uuid4()),
            user_preferences=self.config.copy()
        )
        self.current_session = session
        self.stats['total_sessions'] += 1
        
        logger.info(f"📱 새 세션 생성: {session.session_id[:8]}...")
        return session
    
    def get_current_session(self) -> AppSession:
        """현재 세션 가져오기 (없으면 생성)"""
        if not self.current_session:
            return self.create_session()
        
        # 세션 타임아웃 확인
        timeout_minutes = self.config['session_timeout_minutes']
        if (datetime.now() - self.current_session.last_activity).total_seconds() > timeout_minutes * 60:
            logger.info(f"⏰ 세션 타임아웃: {self.current_session.session_id[:8]}...")
            return self.create_session()
        
        return self.current_session
    
    def add_message(self, role: str, content: str) -> None:
        """메시지 추가"""
        session = self.get_current_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": session.session_id
        }
        
        session.messages.append(message)
        session.last_activity = datetime.now()
        self.stats['total_messages'] += 1
        
        # 메시지 히스토리 제한
        max_history = self.config['max_message_history']
        if len(session.messages) > max_history:
            session.messages = session.messages[-max_history:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        uptime = datetime.now() - self.stats['uptime_start']
        session = self.get_current_session()
        
        return {
            "system_status": self.system_health.status.value,
            "session_id": session.session_id[:8] if session else "none",
            "uptime_hours": round(uptime.total_seconds() / 3600, 1),
            "total_sessions": self.stats['total_sessions'],
            "total_messages": self.stats['total_messages'],
            "success_rate": round(
                (self.stats['successful_queries'] / max(1, self.stats['total_messages'])) * 100, 1
            ),
            "a2a_agents": f"{self.system_health.a2a_agents_online}/{self.system_health.a2a_agents_total}",
            "mcp_tools": f"{self.system_health.mcp_tools_online}/{self.system_health.mcp_tools_total}",
            "broker_status": "✅" if self.system_health.broker_initialized else "❌",
            "orchestrator_status": "✅" if self.system_health.orchestrator_initialized else "❌"
        }
    
    def handle_error(self, error: Exception, context: str = "") -> str:
        """에러 처리 및 사용자 친화적 메시지 반환"""
        error_msg = str(error)
        logger.error(f"🚨 {context} 오류: {error_msg}")
        
        self.stats['failed_queries'] += 1
        self.system_health.error_message = error_msg
        
        # 사용자 친화적 에러 메시지
        if "connection" in error_msg.lower():
            return "🔌 연결 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        elif "timeout" in error_msg.lower():
            return "⏰ 처리 시간이 초과되었습니다. 다시 시도해주세요."
        elif "broker" in error_msg.lower():
            return "🔄 메시지 브로커 문제가 발생했습니다. 시스템을 재시작해주세요."
        else:
            return f"❌ 처리 중 오류가 발생했습니다: {error_msg}"

# 전역 앱 컨트롤러 인스턴스
_app_controller = None

def get_app_controller() -> MainAppController:
    """앱 컨트롤러 싱글톤 인스턴스 반환"""
    global _app_controller
    if _app_controller is None:
        _app_controller = MainAppController()
    return _app_controller

def initialize_app_controller() -> MainAppController:
    """앱 컨트롤러 초기화"""
    controller = get_app_controller()
    
    # Streamlit에서 비동기 초기화
    if not hasattr(st.session_state, 'system_initialized'):
        try:
            # 비동기 초기화를 동기적으로 실행
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프가 있으면 태스크로 처리
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, controller.initialize_system())
                    success = future.result(timeout=30)
            except RuntimeError:
                # 실행 중인 루프가 없으면 새로 생성
                success = asyncio.run(controller.initialize_system())
            
            st.session_state.system_initialized = success
            if not success:
                st.error(f"시스템 초기화 실패: {controller.system_health.error_message}")
                
        except Exception as e:
            logger.error(f"앱 컨트롤러 초기화 실패: {e}")
            st.session_state.system_initialized = False
            st.error(f"시스템 초기화 중 오류: {str(e)}")
    
    return controller 