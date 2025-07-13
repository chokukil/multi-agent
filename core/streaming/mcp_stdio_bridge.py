#!/usr/bin/env python3
"""
🔗 MCP STDIO → SSE Bridge

STDIO 기반 MCP (Model Context Protocol) 서비스들을 
SSE (Server-Sent Events)로 변환하는 브리지

Purpose:
- A2A 에이전트들은 SSE 기반 통신 사용
- 일부 MCP 도구들은 STDIO 기반으로 서비스
- STDIO MCP → SSE 변환으로 A2A 호환성 확보
- 향후 STDIO 기반 MCP 확장성 대비

Architecture:
- STDIO MCP 프로세스 관리
- JSON-RPC → SSE 변환
- 실시간 스트리밍 지원
- 멀티 세션 관리
"""

import asyncio
import json
import logging
import subprocess
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import signal
import os

logger = logging.getLogger(__name__)

@dataclass
class STDIOMCPService:
    """STDIO 기반 MCP 서비스 정보"""
    service_id: str
    name: str
    command: str
    args: List[str]
    description: str
    capabilities: List[str]
    working_dir: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    timeout: int = 30

@dataclass
class MCPSession:
    """MCP STDIO 세션 정보"""
    session_id: str
    service_id: str
    process: Optional[subprocess.Popen]
    created_at: datetime
    last_activity: datetime
    active: bool = True

class MCPSTDIOBridge:
    """MCP STDIO → SSE 변환 브리지"""
    
    def __init__(self):
        self.stdio_services: Dict[str, STDIOMCPService] = {}
        self.active_sessions: Dict[str, MCPSession] = {}
        self.registered_capabilities: Set[str] = set()
        
        # 기본 STDIO MCP 서비스들 등록
        self._register_default_stdio_services()
    
    def _register_default_stdio_services(self):
        """기본 STDIO MCP 서비스들 등록"""
        
        # 예시: Python 기반 STDIO MCP 서비스들
        default_services = [
            STDIOMCPService(
                service_id="pandas_stdio",
                name="Pandas STDIO MCP",
                command="python",
                args=["-m", "mcp_pandas_stdio"],
                description="STDIO 기반 Pandas 데이터 처리",
                capabilities=["data_processing", "dataframe_operations", "statistical_analysis"]
            ),
            STDIOMCPService(
                service_id="file_manager_stdio",
                name="File Manager STDIO MCP", 
                command="python",
                args=["-m", "mcp_file_manager_stdio"],
                description="STDIO 기반 파일 시스템 관리",
                capabilities=["file_operations", "directory_management", "file_search"]
            ),
            STDIOMCPService(
                service_id="data_analyzer_stdio",
                name="Data Analyzer STDIO MCP",
                command="python", 
                args=["-m", "mcp_data_analyzer_stdio"],
                description="STDIO 기반 고급 데이터 분석",
                capabilities=["statistical_analysis", "data_mining", "pattern_recognition"]
            ),
            # Node.js 기반 MCP 서비스 예시
            STDIOMCPService(
                service_id="web_scraper_stdio",
                name="Web Scraper STDIO MCP",
                command="node",
                args=["mcp-web-scraper-stdio.js"],
                description="STDIO 기반 웹 스크래핑",
                capabilities=["web_scraping", "html_parsing", "data_extraction"]
            )
        ]
        
        for service in default_services:
            self.stdio_services[service.service_id] = service
            self.registered_capabilities.update(service.capabilities)
    
    def register_stdio_service(self, service: STDIOMCPService):
        """새로운 STDIO MCP 서비스 등록"""
        self.stdio_services[service.service_id] = service
        self.registered_capabilities.update(service.capabilities)
        logger.info(f"📝 STDIO MCP 서비스 등록: {service.name}")
    
    def get_available_services(self) -> Dict[str, STDIOMCPService]:
        """사용 가능한 STDIO MCP 서비스 목록"""
        return self.stdio_services.copy()
    
    def get_service_capabilities(self) -> Set[str]:
        """등록된 모든 서비스의 기능 목록"""
        return self.registered_capabilities.copy()
    
    async def create_session(self, service_id: str, session_id: Optional[str] = None) -> str:
        """새로운 MCP STDIO 세션 생성"""
        if service_id not in self.stdio_services:
            raise ValueError(f"Unknown STDIO MCP service: {service_id}")
        
        if session_id is None:
            session_id = f"mcp_stdio_{uuid.uuid4().hex[:8]}"
        
        service = self.stdio_services[service_id]
        
        try:
            # STDIO 프로세스 시작
            process = await asyncio.create_subprocess_exec(
                service.command,
                *service.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=service.working_dir,
                env={**os.environ, **(service.env_vars or {})}
            )
            
            session = MCPSession(
                session_id=session_id,
                service_id=service_id,
                process=process,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"🚀 MCP STDIO 세션 시작: {session_id} ({service.name})")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ MCP STDIO 세션 시작 실패: {e}")
            raise
    
    async def close_session(self, session_id: str):
        """MCP STDIO 세션 종료"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        if session.process and session.active:
            try:
                # 프로세스 정상 종료 시도
                session.process.terminate()
                await asyncio.wait_for(session.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # 강제 종료
                session.process.kill()
                await session.process.wait()
            except Exception as e:
                logger.warning(f"⚠️ 세션 종료 중 오류: {e}")
        
        session.active = False
        del self.active_sessions[session_id]
        
        logger.info(f"🔚 MCP STDIO 세션 종료: {session_id}")
    
    async def stream_mcp_request(
        self, 
        session_id: str,
        method: str,
        params: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """MCP STDIO 요청을 SSE 스트림으로 변환"""
        
        if session_id not in self.active_sessions:
            yield {
                'event': 'error',
                'data': {'error': f'Session not found: {session_id}', 'final': True}
            }
            return
        
        session = self.active_sessions[session_id]
        
        if not session.active or not session.process:
            yield {
                'event': 'error', 
                'data': {'error': f'Session not active: {session_id}', 'final': True}
            }
            return
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # JSON-RPC 요청 생성
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        try:
            # 요청 전송
            yield {
                'event': 'request_sent',
                'data': {
                    'session_id': session_id,
                    'method': method,
                    'request_id': request_id,
                    'final': False
                }
            }
            
            request_json = json.dumps(jsonrpc_request) + '\n'
            session.process.stdin.write(request_json.encode())
            await session.process.stdin.drain()
            
            session.last_activity = datetime.now()
            
            # 응답 스트리밍
            async for sse_event in self._stream_stdio_output(session, request_id):
                yield sse_event
                
        except Exception as e:
            logger.error(f"❌ MCP STDIO 요청 처리 오류: {e}")
            yield {
                'event': 'error',
                'data': {
                    'error': str(e),
                    'session_id': session_id,
                    'request_id': request_id,
                    'final': True
                }
            }
    
    async def _stream_stdio_output(
        self, 
        session: MCPSession, 
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """STDIO 출력을 SSE 이벤트로 스트리밍"""
        
        buffer = ""
        timeout_task = None
        
        try:
            # 타임아웃 설정
            service = self.stdio_services[session.service_id]
            timeout_task = asyncio.create_task(asyncio.sleep(service.timeout))
            
            while True:
                # 출력 대기 또는 타임아웃
                read_task = asyncio.create_task(session.process.stdout.read(1024))
                
                done, pending = await asyncio.wait(
                    [read_task, timeout_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 타임아웃 발생
                if timeout_task in done:
                    read_task.cancel()
                    yield {
                        'event': 'timeout',
                        'data': {
                            'message': f'Request timeout after {service.timeout}s',
                            'request_id': request_id,
                            'final': True
                        }
                    }
                    break
                
                # 데이터 읽기 완료
                data = await read_task
                timeout_task.cancel()
                
                if not data:
                    # 프로세스 종료
                    yield {
                        'event': 'process_ended',
                        'data': {
                            'message': 'MCP process ended',
                            'request_id': request_id,
                            'final': True
                        }
                    }
                    break
                
                buffer += data.decode('utf-8', errors='ignore')
                
                # 줄 단위로 처리
                lines = buffer.split('\n')
                buffer = lines[-1]  # 마지막 불완전한 줄 보관
                
                for line in lines[:-1]:
                    if line.strip():
                        await self._process_stdio_line(line.strip(), request_id, session)
                        
                        # SSE 이벤트로 변환
                        async for sse_event in self._convert_line_to_sse(line.strip(), request_id):
                            yield sse_event
                
                # 새로운 타임아웃 설정
                timeout_task = asyncio.create_task(asyncio.sleep(service.timeout))
                
        except Exception as e:
            logger.error(f"❌ STDIO 출력 스트리밍 오류: {e}")
            yield {
                'event': 'stream_error',
                'data': {
                    'error': str(e),
                    'request_id': request_id,
                    'final': True
                }
            }
        finally:
            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
    
    async def _process_stdio_line(self, line: str, request_id: str, session: MCPSession):
        """STDIO 라인 처리 (로깅, 상태 업데이트 등)"""
        session.last_activity = datetime.now()
        
        # JSON-RPC 응답인지 확인
        try:
            parsed = json.loads(line)
            if 'jsonrpc' in parsed and parsed.get('id') == request_id:
                logger.debug(f"📨 JSON-RPC 응답 수신: {session.session_id}")
        except json.JSONDecodeError:
            # 일반 텍스트 출력
            logger.debug(f"📤 STDIO 출력: {line[:100]}...")
    
    async def _convert_line_to_sse(
        self, 
        line: str, 
        request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """STDIO 라인을 SSE 이벤트로 변환"""
        
        try:
            # JSON-RPC 응답 처리
            parsed = json.loads(line)
            
            if 'jsonrpc' in parsed:
                if 'result' in parsed:
                    # 성공 응답
                    yield {
                        'event': 'mcp_result',
                        'data': {
                            'result': parsed['result'],
                            'request_id': request_id,
                            'final': True
                        }
                    }
                elif 'error' in parsed:
                    # 오류 응답
                    yield {
                        'event': 'mcp_error',
                        'data': {
                            'error': parsed['error'],
                            'request_id': request_id,
                            'final': True
                        }
                    }
                else:
                    # 진행 상황 또는 부분 응답
                    yield {
                        'event': 'mcp_progress',
                        'data': {
                            'data': parsed,
                            'request_id': request_id,
                            'final': False
                        }
                    }
            else:
                # 비표준 JSON 응답
                yield {
                    'event': 'mcp_data',
                    'data': {
                        'data': parsed,
                        'request_id': request_id,
                        'final': False
                    }
                }
                
        except json.JSONDecodeError:
            # 일반 텍스트 출력
            yield {
                'event': 'mcp_output',
                'data': {
                    'output': line,
                    'request_id': request_id,
                    'final': False
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """브리지 상태 확인"""
        active_count = len([s for s in self.active_sessions.values() if s.active])
        
        service_status = {}
        for service_id, service in self.stdio_services.items():
            # 서비스 실행 가능성 테스트 (간단한 버전 체크)
            try:
                test_process = await asyncio.create_subprocess_exec(
                    service.command, '--version',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await asyncio.wait_for(test_process.wait(), timeout=5.0)
                service_status[service_id] = 'available'
            except:
                service_status[service_id] = 'unavailable'
        
        return {
            'bridge_status': 'healthy',
            'active_sessions': active_count,
            'total_services': len(self.stdio_services),
            'service_status': service_status,
            'capabilities': list(self.registered_capabilities),
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup_inactive_sessions(self, max_idle_minutes: int = 30):
        """비활성 세션 정리"""
        cutoff_time = datetime.now().timestamp() - (max_idle_minutes * 60)
        
        to_close = []
        for session_id, session in self.active_sessions.items():
            if session.last_activity.timestamp() < cutoff_time:
                to_close.append(session_id)
        
        for session_id in to_close:
            await self.close_session(session_id)
            logger.info(f"🧹 비활성 세션 정리: {session_id}")
    
    async def shutdown(self):
        """브리지 종료 - 모든 세션 정리"""
        logger.info("🔚 MCP STDIO Bridge 종료 시작...")
        
        # 모든 세션 종료
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
        
        logger.info("✅ MCP STDIO Bridge 종료 완료")


# 전역 브리지 인스턴스
_mcp_stdio_bridge = None

def get_mcp_stdio_bridge() -> MCPSTDIOBridge:
    """전역 MCP STDIO 브리지 인스턴스 반환"""
    global _mcp_stdio_bridge
    if _mcp_stdio_bridge is None:
        _mcp_stdio_bridge = MCPSTDIOBridge()
    return _mcp_stdio_bridge


# 편의 함수들
async def stream_stdio_mcp(
    service_id: str,
    method: str,
    params: Dict[str, Any],
    session_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """STDIO MCP 요청을 SSE로 스트리밍하는 편의 함수"""
    bridge = get_mcp_stdio_bridge()
    
    if session_id is None:
        session_id = await bridge.create_session(service_id)
    
    try:
        async for event in bridge.stream_mcp_request(session_id, method, params):
            yield event
    finally:
        await bridge.close_session(session_id)


if __name__ == "__main__":
    # 테스트/데모 코드
    async def demo():
        bridge = MCPSTDIOBridge()
        
        print("🔗 MCP STDIO Bridge Demo")
        print(f"📋 등록된 서비스: {len(bridge.get_available_services())}")
        
        # 상태 확인
        health = await bridge.health_check()
        print(f"💊 브리지 상태: {health}")
        
        # 테스트 세션 생성 (pandas_stdio가 실제로 존재한다면)
        try:
            session_id = await bridge.create_session("pandas_stdio")
            print(f"🚀 테스트 세션 생성: {session_id}")
            
            # 테스트 요청
            async for event in bridge.stream_mcp_request(
                session_id, 
                "test", 
                {"message": "Hello MCP STDIO!"}
            ):
                print(f"📨 이벤트: {event}")
                if event['data'].get('final'):
                    break
            
            await bridge.close_session(session_id)
            
        except Exception as e:
            print(f"⚠️ 테스트 실패 (예상됨 - 실제 MCP 서비스 필요): {e}")
        
        await bridge.shutdown()
    
    asyncio.run(demo()) 