"""
🍒 CherryAI - A2A SSE Client
A2A SDK 0.2.9 표준 SSE 클라이언트

A2A 에이전트들로부터 실시간 스트리밍 데이터를 수신하는 클라이언트
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# 로거 설정
logger = logging.getLogger(__name__)

class A2AMessageType(Enum):
    """A2A 메시지 타입"""
    START = "start"
    PROGRESS = "progress" 
    COMPLETE = "complete"
    ERROR = "error"
    STATUS_UPDATE = "status_update"

@dataclass
class A2AStreamEvent:
    """A2A 스트림 이벤트 데이터 클래스"""
    source: str  # 'a2a'
    agent: str  # 'pandas', 'orchestrator', etc.
    event_type: A2AMessageType
    data: Dict[str, Any]
    final: bool
    timestamp: float
    raw_data: str

class A2ASSEClient:
    """A2A SDK 0.2.9 표준 SSE 클라이언트"""
    
    def __init__(self, base_url: str, agents: Dict[str, str]):
        """
        Args:
            base_url: A2A 기본 URL
            agents: {'agent_name': 'http://localhost:port'} 형태의 에이전트 매핑
        """
        self.base_url = base_url
        self.agents = agents
        self.active_connections: Dict[str, aiohttp.ClientSession] = {}
        self.reconnect_interval = 5.0  # 재연결 간격 (초)
        self.max_reconnect_attempts = 3
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
    async def stream_agent_response(
        self, 
        agent_name: str, 
        query: str, 
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[A2AStreamEvent, None]:
        """A2A 에이전트로부터 실시간 스트리밍 수신"""
        
        if agent_name not in self.agents:
            logger.error(f"Unknown agent: {agent_name}")
            yield self._create_error_event(agent_name, f"Unknown agent: {agent_name}")
            return
        
        agent_url = self.agents[agent_name]
        sse_endpoint = f"{agent_url}/stream/{session_id}"
        
        reconnect_attempts = 0
        
        while reconnect_attempts <= self.max_reconnect_attempts:
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    # A2A 표준 요청 준비
                    headers = {
                        'Accept': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'Content-Type': 'application/json'
                    }
                    
                    # A2A 표준 페이로드
                    payload = {
                        "messageId": f"{session_id}_{int(time.time())}",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": query
                            }
                        ]
                    }
                    
                    if context:
                        payload["context"] = context
                    
                    # SSE 연결 시작
                    async with session.post(
                        sse_endpoint,
                        headers=headers,
                        json=payload
                    ) as response:
                        
                        if response.status != 200:
                            logger.error(f"A2A SSE 연결 실패: {response.status}")
                            yield self._create_error_event(
                                agent_name, 
                                f"Connection failed: HTTP {response.status}"
                            )
                            return
                        
                        logger.info(f"A2A SSE 연결 성공: {agent_name}")
                        
                        # 스트림 이벤트 처리
                        async for event in self._process_sse_stream(response, agent_name):
                            yield event
                            
                            # 최종 이벤트면 정상 종료
                            if event.final:
                                logger.info(f"A2A 스트림 정상 완료: {agent_name}")
                                return
                
            except asyncio.TimeoutError:
                logger.warning(f"A2A SSE 타임아웃: {agent_name}")
                reconnect_attempts += 1
                if reconnect_attempts <= self.max_reconnect_attempts:
                    logger.info(f"재연결 시도 {reconnect_attempts}/{self.max_reconnect_attempts}")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    yield self._create_error_event(agent_name, "Connection timeout")
                    
            except aiohttp.ClientError as e:
                logger.error(f"A2A SSE 클라이언트 오류: {e}")
                reconnect_attempts += 1
                if reconnect_attempts <= self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    yield self._create_error_event(agent_name, f"Client error: {str(e)}")
                    
            except Exception as e:
                logger.error(f"A2A SSE 예상치 못한 오류: {e}")
                yield self._create_error_event(agent_name, f"Unexpected error: {str(e)}")
                return
    
    async def _process_sse_stream(
        self, 
        response: aiohttp.ClientResponse, 
        agent_name: str
    ) -> AsyncGenerator[A2AStreamEvent, None]:
        """SSE 스트림 처리"""
        
        buffer = ""
        
        async for line in response.content:
            try:
                line_str = line.decode('utf-8').strip()
                
                if not line_str:
                    # 빈 줄은 이벤트 구분자
                    if buffer:
                        event = self._parse_sse_event(buffer, agent_name)
                        if event:
                            yield event
                        buffer = ""
                    continue
                
                buffer += line_str + "\n"
                
            except UnicodeDecodeError as e:
                logger.warning(f"SSE 디코딩 오류: {e}")
                continue
                
        # 스트림 종료 시 남은 버퍼 처리
        if buffer:
            event = self._parse_sse_event(buffer, agent_name)
            if event:
                yield event
    
    def _parse_sse_event(self, raw_data: str, agent_name: str) -> Optional[A2AStreamEvent]:
        """A2A 표준 SSE 이벤트 파싱"""
        
        try:
            lines = raw_data.strip().split('\n')
            event_type = None
            data_str = None
            
            for line in lines:
                if line.startswith('event: '):
                    event_type = line[7:].strip()
                elif line.startswith('data: '):
                    data_str = line[6:].strip()
            
            if not data_str:
                return None
            
            # A2A 표준 데이터 파싱
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 텍스트로 처리
                data = {"content": data_str, "type": "text"}
            
            # A2A 메시지 타입 결정
            if event_type:
                try:
                    msg_type = A2AMessageType(event_type)
                except ValueError:
                    msg_type = A2AMessageType.PROGRESS  # 기본값
            else:
                # 데이터 내용으로 타입 추론
                if data.get('final', False):
                    msg_type = A2AMessageType.COMPLETE
                elif 'error' in data:
                    msg_type = A2AMessageType.ERROR
                elif 'status' in data:
                    msg_type = A2AMessageType.STATUS_UPDATE
                else:
                    msg_type = A2AMessageType.PROGRESS
            
            return A2AStreamEvent(
                source="a2a",
                agent=agent_name,
                event_type=msg_type,
                data=data,
                final=data.get('final', False) or msg_type == A2AMessageType.COMPLETE,
                timestamp=time.time(),
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"SSE 이벤트 파싱 실패: {e}")
            return self._create_error_event(agent_name, f"Parse error: {str(e)}")
    
    def _create_error_event(self, agent_name: str, error_message: str) -> A2AStreamEvent:
        """에러 이벤트 생성"""
        return A2AStreamEvent(
            source="a2a",
            agent=agent_name,
            event_type=A2AMessageType.ERROR,
            data={"error": error_message, "final": True},
            final=True,
            timestamp=time.time(),
            raw_data=f"error: {error_message}"
        )
    
    async def send_a2a_request(
        self,
        agent_name: str,
        query: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """A2A 표준 요청 전송 (non-streaming)"""
        
        if agent_name not in self.agents:
            return {"error": f"Unknown agent: {agent_name}"}
        
        agent_url = self.agents[agent_name]
        endpoint = f"{agent_url}/execute"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # A2A 표준 페이로드
                payload = {
                    "messageId": f"{session_id}_{int(time.time())}",
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text", 
                            "text": query
                        }
                    ]
                }
                
                if context:
                    payload["context"] = context
                
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {
                            "error": f"HTTP {response.status}", 
                            "agent": agent_name
                        }
                        
        except Exception as e:
            logger.error(f"A2A 요청 실패: {e}")
            return {"error": str(e), "agent": agent_name}
    
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """A2A 에이전트 상태 확인"""
        
        if agent_name not in self.agents:
            return {"status": "unknown", "error": f"Unknown agent: {agent_name}"}
        
        agent_url = self.agents[agent_name]
        status_endpoint = f"{agent_url}/.well-known/agent.json"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(status_endpoint) as response:
                    if response.status == 200:
                        agent_card = await response.json()
                        return {
                            "status": "active",
                            "agent_card": agent_card,
                            "url": agent_url
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}",
                            "url": agent_url
                        }
                        
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "url": agent_url
            }
    
    async def validate_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """모든 A2A 에이전트 상태 검증"""
        
        results = {}
        
        tasks = [
            self.get_agent_status(agent_name) 
            for agent_name in self.agents.keys()
        ]
        
        agent_statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for agent_name, status in zip(self.agents.keys(), agent_statuses):
            if isinstance(status, Exception):
                results[agent_name] = {
                    "status": "error",
                    "error": str(status)
                }
            else:
                results[agent_name] = status
        
        return results
    
    def get_agent_urls(self) -> Dict[str, str]:
        """에이전트 URL 매핑 반환"""
        return self.agents.copy()
    
    def add_agent(self, agent_name: str, agent_url: str):
        """새 에이전트 추가"""
        self.agents[agent_name] = agent_url
        logger.info(f"A2A 에이전트 추가: {agent_name} -> {agent_url}")
    
    def remove_agent(self, agent_name: str):
        """에이전트 제거"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"A2A 에이전트 제거: {agent_name}")
    
    async def close(self):
        """모든 연결 정리"""
        for session in self.active_connections.values():
            if not session.closed:
                await session.close()
        self.active_connections.clear()
        logger.info("A2A SSE 클라이언트 연결 정리 완료") 