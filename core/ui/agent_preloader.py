"""
🚀 Agent Preloader - 백그라운드 에이전트 초기화 시스템
페이지 로딩 시점에 에이전트들을 미리 초기화하여 사용자 경험을 개선합니다.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import httpx
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """에이전트 상태"""
    UNKNOWN = "❓"
    INITIALIZING = "⏳"
    READY = "✅"
    FAILED = "❌"
    WARMING_UP = "🔥"

@dataclass
class AgentInfo:
    """에이전트 정보"""
    name: str
    port: int
    description: str
    capabilities: List[str] = field(default_factory=list)
    color: str = "#ffffff"
    status: AgentStatus = AgentStatus.UNKNOWN
    last_check: Optional[datetime] = None
    initialization_time: Optional[float] = None
    error_message: Optional[str] = None
    health_url: Optional[str] = None

class AgentPreloader:
    """에이전트 프리로딩 관리자"""
    
    def __init__(self, agents_config: Dict[str, Dict[str, Any]]):
        self.agents_config = agents_config
        self.agents: Dict[str, AgentInfo] = {}
        self.initialization_start_time = None
        self.initialization_complete = False
        self._lock = threading.Lock()
        
        # 에이전트 정보 초기화
        self._initialize_agent_info()
    
    def _initialize_agent_info(self):
        """에이전트 정보 초기화"""
        for name, config in self.agents_config.items():
            self.agents[name] = AgentInfo(
                name=name,
                port=config["port"],
                description=config["description"],
                capabilities=config.get("capabilities", []),
                color=config.get("color", "#ffffff"),
                health_url=f"http://localhost:{config['port']}/.well-known/agent.json"
            )
    
    async def preload_agents(self, progress_callback=None) -> Dict[str, AgentInfo]:
        """에이전트들을 백그라운드에서 프리로드"""
        self.initialization_start_time = time.time()
        
        logger.info("🚀 에이전트 프리로딩 시작")
        
        # 프로그레시브 로딩: 핵심 에이전트부터 우선순위로 초기화
        priority_agents = ["Orchestrator", "🔍 EDA Tools", "📊 Data Visualization"]
        secondary_agents = [name for name in self.agents.keys() if name not in priority_agents]
        
        total_agents = len(self.agents)
        completed = 0
        
        # 1단계: 핵심 에이전트 초기화
        logger.info("📋 1단계: 핵심 에이전트 초기화")
        for agent_name in priority_agents:
            if agent_name in self.agents:
                await self._initialize_single_agent(agent_name)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_agents, f"핵심 에이전트 초기화: {agent_name}")
        
        # 2단계: 나머지 에이전트 병렬 초기화
        logger.info("⚡ 2단계: 나머지 에이전트 병렬 초기화")
        tasks = []
        for agent_name in secondary_agents:
            if agent_name in self.agents:
                task = asyncio.create_task(self._initialize_single_agent(agent_name))
                tasks.append((agent_name, task))
        
        # 병렬 초기화 결과 수집
        for agent_name, task in tasks:
            try:
                await task
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_agents, f"에이전트 초기화: {agent_name}")
            except Exception as e:
                logger.error(f"❌ {agent_name} 초기화 실패: {e}")
                self.agents[agent_name].status = AgentStatus.FAILED
                self.agents[agent_name].error_message = str(e)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_agents, f"에이전트 초기화 실패: {agent_name}")
        
        self.initialization_complete = True
        total_time = time.time() - self.initialization_start_time
        
        # 최종 상태 요약
        ready_count = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.READY)
        failed_count = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.FAILED)
        
        logger.info(f"✅ 에이전트 프리로딩 완료: {total_time:.2f}초")
        logger.info(f"📊 준비됨: {ready_count}개, 실패: {failed_count}개, 전체: {total_agents}개")
        
        return self.agents
    
    async def _initialize_single_agent(self, agent_name: str) -> bool:
        """단일 에이전트 초기화"""
        agent = self.agents[agent_name]
        agent.status = AgentStatus.INITIALIZING
        start_time = time.time()
        
        try:
            # 헬스 체크
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(agent.health_url)
                
                if response.status_code == 200:
                    agent_card = response.json()
                    
                    # 에이전트 워밍업 (간단한 요청으로 초기화 완료 확인)
                    agent.status = AgentStatus.WARMING_UP
                    await self._warm_up_agent(agent_name, client)
                    
                    agent.status = AgentStatus.READY
                    agent.last_check = datetime.now()
                    agent.initialization_time = time.time() - start_time
                    
                    logger.info(f"✅ {agent_name} 초기화 완료 ({agent.initialization_time:.2f}초)")
                    return True
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
        except Exception as e:
            agent.status = AgentStatus.FAILED
            agent.error_message = str(e)
            agent.initialization_time = time.time() - start_time
            
            logger.warning(f"❌ {agent_name} 초기화 실패: {e}")
            return False
    
    async def _warm_up_agent(self, agent_name: str, client: httpx.AsyncClient):
        """에이전트 워밍업 (선택적)"""
        # 오케스트레이터는 워밍업 스킵 (너무 복잡한 초기화 방지)
        if agent_name == "Orchestrator":
            return
            
        try:
            # 간단한 핑 요청으로 에이전트 준비 상태 확인
            # 실제 A2A 요청은 하지 않고 헬스 체크만 재수행
            agent = self.agents[agent_name]
            response = await client.get(agent.health_url)
            
            if response.status_code == 200:
                logger.debug(f"🔥 {agent_name} 워밍업 완료")
            else:
                logger.warning(f"⚠️ {agent_name} 워밍업 실패: HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ {agent_name} 워밍업 중 오류: {e}")
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentInfo]:
        """특정 에이전트 상태 조회"""
        return self.agents.get(agent_name)
    
    def get_all_agents_status(self) -> Dict[str, AgentInfo]:
        """모든 에이전트 상태 조회"""
        return self.agents.copy()
    
    def get_ready_agents(self) -> List[str]:
        """준비된 에이전트 목록 반환"""
        return [name for name, agent in self.agents.items() if agent.status == AgentStatus.READY]
    
    def get_failed_agents(self) -> List[str]:
        """실패한 에이전트 목록 반환"""
        return [name for name, agent in self.agents.items() if agent.status == AgentStatus.FAILED]
    
    def is_initialization_complete(self) -> bool:
        """초기화 완료 여부"""
        return self.initialization_complete
    
    def get_initialization_summary(self) -> Dict[str, Any]:
        """초기화 요약 정보"""
        ready_count = len(self.get_ready_agents())
        failed_count = len(self.get_failed_agents())
        total_count = len(self.agents)
        
        return {
            "total_agents": total_count,
            "ready_agents": ready_count,
            "failed_agents": failed_count,
            "success_rate": (ready_count / total_count * 100) if total_count > 0 else 0,
            "initialization_time": time.time() - self.initialization_start_time if self.initialization_start_time else 0,
            "is_complete": self.initialization_complete
        }

@st.cache_resource
def get_agent_preloader(agents_config: Dict[str, Dict[str, Any]]) -> AgentPreloader:
    """캐시된 에이전트 프리로더 인스턴스 반환"""
    return AgentPreloader(agents_config)

class ProgressiveLoadingUI:
    """프로그레시브 로딩 UI 컴포넌트"""
    
    def __init__(self, container):
        self.container = container
        self.progress_bar = None
        self.status_text = None
        self.details_expander = None
    
    def setup_ui(self):
        """로딩 UI 설정"""
        with self.container:
            st.markdown("### 🚀 AI DS Team 에이전트 초기화 중...")
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.details_expander = st.expander("🔍 상세 진행 상황", expanded=False)
    
    def update_progress(self, completed: int, total: int, current_task: str):
        """진행 상황 업데이트"""
        progress = completed / total if total > 0 else 0
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text:
            self.status_text.text(f"📋 {current_task} ({completed}/{total})")
        
        if self.details_expander:
            with self.details_expander:
                st.text(f"진행률: {progress*100:.1f}%")
                st.text(f"완료: {completed}개")
                st.text(f"남은 작업: {total - completed}개")
    
    def show_completion(self, summary: Dict[str, Any]):
        """완료 상태 표시"""
        with self.container:
            st.success(f"✅ 에이전트 초기화 완료! ({summary['initialization_time']:.2f}초)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 에이전트", summary['total_agents'])
            with col2:
                st.metric("준비 완료", summary['ready_agents'], delta=f"{summary['success_rate']:.1f}%")
            with col3:
                st.metric("실패", summary['failed_agents']) 