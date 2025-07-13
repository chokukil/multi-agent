#!/usr/bin/env python3
"""
🍒 CherryAI MCP 서버 관리 도구
Phase 1.3: MCP 서버 생명주기 관리 및 운영 도구

Features:
- 서버 재시작 자동화
- 설정 검증 및 진단
- 로그 모니터링 및 분석
- 성능 최적화 권장사항
- 장애 예방 및 복구

Author: CherryAI Team
Date: 2025-07-13
"""

import asyncio
import logging
import subprocess
import signal
import psutil
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

from .mcp_config_manager import MCPConfigManager, MCPServerDefinition, MCPServerType
from .mcp_connection_monitor import MCPConnectionMonitor
from .mcp_auto_recovery import MCPAutoRecovery

logger = logging.getLogger(__name__)

class ServerState(Enum):
    """서버 상태"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class ServerProcess:
    """서버 프로세스 정보"""
    server_id: str
    pid: Optional[int] = None
    state: ServerState = ServerState.STOPPED
    start_time: Optional[datetime] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    log_file: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class ValidationResult:
    """설정 검증 결과"""
    server_id: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    score: int = 100  # 0-100 점수

@dataclass
class LogAnalysis:
    """로그 분석 결과"""
    server_id: str
    total_lines: int = 0
    error_count: int = 0
    warning_count: int = 0
    recent_errors: List[str] = field(default_factory=list)
    performance_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MCPServerManager:
    """MCP 서버 관리 시스템"""
    
    def __init__(self, 
                 config_manager: Optional[MCPConfigManager] = None,
                 connection_monitor: Optional[MCPConnectionMonitor] = None):
        self.config_manager = config_manager or MCPConfigManager()
        self.connection_monitor = connection_monitor
        
        # 서버 프로세스 추적
        self.server_processes: Dict[str, ServerProcess] = {}
        
        # 로그 관리
        self.log_directory = Path("logs/mcp_servers")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # 모니터링 설정
        self.monitoring_active = False
        self.performance_history: Dict[str, List[Dict]] = {}
        
        logger.info("MCP Server Manager 초기화 완료")
    
    async def start_server(self, server_id: str) -> bool:
        """서버 시작"""
        try:
            server_def = self.config_manager.get_server(server_id)
            if not server_def:
                logger.error(f"서버 설정을 찾을 수 없음: {server_id}")
                return False
            
            # 이미 실행 중인지 확인
            if self._is_server_running(server_id):
                logger.warning(f"서버가 이미 실행 중: {server_id}")
                return True
            
            # 프로세스 정보 초기화
            process_info = ServerProcess(
                server_id=server_id,
                state=ServerState.STARTING,
                start_time=datetime.now(),
                log_file=str(self.log_directory / f"{server_id}.log")
            )
            self.server_processes[server_id] = process_info
            
            logger.info(f"🚀 서버 시작: {server_id} ({server_def.server_type.value})")
            
            # 서버 타입별 시작 로직
            if server_def.server_type == MCPServerType.STDIO:
                success = await self._start_stdio_server(server_def, process_info)
            elif server_def.server_type == MCPServerType.SSE:
                success = await self._start_sse_server(server_def, process_info)
            else:
                logger.error(f"지원하지 않는 서버 타입: {server_def.server_type}")
                return False
            
            if success:
                process_info.state = ServerState.RUNNING
                logger.info(f"✅ 서버 시작 성공: {server_id}")
            else:
                process_info.state = ServerState.ERROR
                logger.error(f"❌ 서버 시작 실패: {server_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"서버 시작 중 오류 {server_id}: {e}")
            if server_id in self.server_processes:
                self.server_processes[server_id].state = ServerState.ERROR
                self.server_processes[server_id].error_message = str(e)
            return False
    
    async def _start_stdio_server(self, server_def: MCPServerDefinition, process_info: ServerProcess) -> bool:
        """STDIO 서버 시작"""
        try:
            # 명령어 구성
            cmd = [server_def.command] + server_def.args
            
            # 환경변수 설정
            env = os.environ.copy()
            env.update(server_def.env)
            
            # 로그 파일 설정
            log_file = open(process_info.log_file, 'a', encoding='utf-8')
            
            # 프로세스 시작
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=server_def.cwd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                stdin=asyncio.subprocess.PIPE
            )
            
            process_info.pid = process.pid
            
            # 짧은 대기 후 프로세스 상태 확인
            await asyncio.sleep(2)
            
            if process.returncode is None:  # 여전히 실행 중
                logger.info(f"STDIO 서버 시작 성공: {server_def.server_id} (PID: {process.pid})")
                return True
            else:
                logger.error(f"STDIO 서버가 즉시 종료됨: {server_def.server_id} (코드: {process.returncode})")
                return False
                
        except Exception as e:
            logger.error(f"STDIO 서버 시작 실패 {server_def.server_id}: {e}")
            return False
    
    async def _start_sse_server(self, server_def: MCPServerDefinition, process_info: ServerProcess) -> bool:
        """SSE 서버 시작"""
        try:
            # SSE 서버의 경우 보통 외부에서 관리되므로 연결 테스트만 수행
            if server_def.url:
                import httpx
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.get(server_def.url, timeout=5.0)
                        if response.status_code == 200:
                            logger.info(f"SSE 서버 연결 확인: {server_def.server_id}")
                            return True
                        else:
                            logger.warning(f"SSE 서버 응답 비정상: {server_def.server_id} (상태: {response.status_code})")
                            return False
                    except Exception as e:
                        logger.error(f"SSE 서버 연결 실패: {server_def.server_id} - {e}")
                        return False
            else:
                logger.error(f"SSE 서버 URL이 설정되지 않음: {server_def.server_id}")
                return False
                
        except Exception as e:
            logger.error(f"SSE 서버 확인 실패 {server_def.server_id}: {e}")
            return False
    
    async def stop_server(self, server_id: str, force: bool = False) -> bool:
        """서버 중지"""
        try:
            if server_id not in self.server_processes:
                logger.warning(f"서버 프로세스 정보 없음: {server_id}")
                return True
            
            process_info = self.server_processes[server_id]
            
            if process_info.state == ServerState.STOPPED:
                logger.info(f"서버가 이미 중지됨: {server_id}")
                return True
            
            process_info.state = ServerState.STOPPING
            logger.info(f"🛑 서버 중지: {server_id}")
            
            if process_info.pid:
                try:
                    process = psutil.Process(process_info.pid)
                    
                    if force:
                        # 강제 종료
                        process.kill()
                        logger.info(f"강제 종료: {server_id} (PID: {process_info.pid})")
                    else:
                        # 정상 종료 시도
                        process.terminate()
                        
                        # 10초 대기
                        try:
                            process.wait(timeout=10)
                        except psutil.TimeoutExpired:
                            # 타임아웃 시 강제 종료
                            process.kill()
                            logger.warning(f"타임아웃으로 강제 종료: {server_id}")
                    
                    process_info.state = ServerState.STOPPED
                    process_info.pid = None
                    logger.info(f"✅ 서버 중지 완료: {server_id}")
                    return True
                    
                except psutil.NoSuchProcess:
                    logger.info(f"프로세스가 이미 종료됨: {server_id}")
                    process_info.state = ServerState.STOPPED
                    process_info.pid = None
                    return True
                    
            else:
                logger.warning(f"PID 정보 없음: {server_id}")
                process_info.state = ServerState.STOPPED
                return True
                
        except Exception as e:
            logger.error(f"서버 중지 중 오류 {server_id}: {e}")
            return False
    
    async def restart_server(self, server_id: str) -> bool:
        """서버 재시작"""
        try:
            logger.info(f"🔄 서버 재시작: {server_id}")
            
            # 서버 중지
            stop_success = await self.stop_server(server_id)
            if not stop_success:
                logger.error(f"서버 중지 실패로 재시작 중단: {server_id}")
                return False
            
            # 짧은 대기
            await asyncio.sleep(2)
            
            # 서버 시작
            start_success = await self.start_server(server_id)
            
            if start_success and server_id in self.server_processes:
                process_info = self.server_processes[server_id]
                process_info.restart_count += 1
                process_info.last_restart = datetime.now()
                logger.info(f"✅ 서버 재시작 완료: {server_id} (재시작 횟수: {process_info.restart_count})")
            
            return start_success
            
        except Exception as e:
            logger.error(f"서버 재시작 중 오류 {server_id}: {e}")
            return False
    
    def _is_server_running(self, server_id: str) -> bool:
        """서버 실행 상태 확인"""
        if server_id not in self.server_processes:
            return False
        
        process_info = self.server_processes[server_id]
        
        if not process_info.pid:
            return False
        
        try:
            process = psutil.Process(process_info.pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    async def validate_server_config(self, server_id: str) -> ValidationResult:
        """서버 설정 검증"""
        try:
            server_def = self.config_manager.get_server(server_id)
            if not server_def:
                return ValidationResult(
                    server_id=server_id,
                    is_valid=False,
                    errors=["서버 설정을 찾을 수 없음"],
                    score=0
                )
            
            result = ValidationResult(server_id=server_id, is_valid=True)
            
            # 기본 설정 검증
            if not server_def.name:
                result.errors.append("서버 이름이 설정되지 않음")
            
            if not server_def.description:
                result.warnings.append("서버 설명이 없음")
            
            # 타입별 검증
            if server_def.server_type == MCPServerType.STDIO:
                await self._validate_stdio_config(server_def, result)
            elif server_def.server_type == MCPServerType.SSE:
                await self._validate_sse_config(server_def, result)
            
            # 성능 권장사항
            self._add_performance_recommendations(server_def, result)
            
            # 점수 계산
            error_penalty = len(result.errors) * 20
            warning_penalty = len(result.warnings) * 5
            result.score = max(0, 100 - error_penalty - warning_penalty)
            
            result.is_valid = len(result.errors) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"설정 검증 중 오류 {server_id}: {e}")
            return ValidationResult(
                server_id=server_id,
                is_valid=False,
                errors=[f"검증 중 오류: {str(e)}"],
                score=0
            )
    
    async def _validate_stdio_config(self, server_def: MCPServerDefinition, result: ValidationResult):
        """STDIO 서버 설정 검증"""
        if not server_def.command:
            result.errors.append("STDIO 서버 명령어가 설정되지 않음")
            return
        
        # 명령어 실행 가능성 확인
        try:
            which_result = subprocess.run(['which', server_def.command], 
                                        capture_output=True, text=True)
            if which_result.returncode != 0:
                result.errors.append(f"명령어를 찾을 수 없음: {server_def.command}")
        except Exception:
            result.warnings.append("명령어 경로 확인 실패")
        
        # 작업 디렉토리 확인
        if not os.path.exists(server_def.cwd):
            result.errors.append(f"작업 디렉토리가 존재하지 않음: {server_def.cwd}")
        
        # 환경변수 검증
        for key, value in server_def.env.items():
            if not value:
                result.warnings.append(f"환경변수 값이 비어있음: {key}")
    
    async def _validate_sse_config(self, server_def: MCPServerDefinition, result: ValidationResult):
        """SSE 서버 설정 검증"""
        if not server_def.url:
            result.errors.append("SSE 서버 URL이 설정되지 않음")
            return
        
        # URL 형식 검증
        if not server_def.url.startswith(('http://', 'https://')):
            result.errors.append("잘못된 URL 형식")
        
        # 연결 테스트
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(server_def.url, timeout=5.0)
                if response.status_code != 200:
                    result.warnings.append(f"서버 응답 상태 비정상: {response.status_code}")
        except Exception as e:
            result.warnings.append(f"연결 테스트 실패: {str(e)}")
    
    def _add_performance_recommendations(self, server_def: MCPServerDefinition, result: ValidationResult):
        """성능 최적화 권장사항 추가"""
        # 타임아웃 설정 확인
        if server_def.timeout > 30:
            result.recommendations.append("타임아웃이 길게 설정됨 (30초 이하 권장)")
        elif server_def.timeout < 5:
            result.recommendations.append("타임아웃이 짧게 설정됨 (5초 이상 권장)")
        
        # 재시도 횟수 확인
        if server_def.retry_count > 5:
            result.recommendations.append("재시도 횟수가 많음 (3-5회 권장)")
        elif server_def.retry_count < 2:
            result.recommendations.append("재시도 횟수가 적음 (2회 이상 권장)")
        
        # 헬스체크 간격 확인
        if server_def.health_check_interval > 120:
            result.recommendations.append("헬스체크 간격이 김 (60초 이하 권장)")
        elif server_def.health_check_interval < 15:
            result.recommendations.append("헬스체크 간격이 짧음 (15초 이상 권장)")
    
    async def analyze_server_logs(self, server_id: str, lines: int = 1000) -> LogAnalysis:
        """서버 로그 분석"""
        try:
            analysis = LogAnalysis(server_id=server_id)
            
            if server_id not in self.server_processes:
                analysis.recommendations.append("서버 프로세스 정보 없음")
                return analysis
            
            process_info = self.server_processes[server_id]
            if not process_info.log_file or not os.path.exists(process_info.log_file):
                analysis.recommendations.append("로그 파일이 존재하지 않음")
                return analysis
            
            # 로그 파일 읽기
            with open(process_info.log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()[-lines:]  # 최근 N개 라인
            
            analysis.total_lines = len(log_lines)
            
            # 로그 분석
            error_patterns = [
                r'ERROR', r'FATAL', r'CRITICAL', r'Exception', r'Traceback',
                r'Failed', r'Error:', r'Cannot', r'Unable to'
            ]
            
            warning_patterns = [
                r'WARNING', r'WARN', r'Deprecated', r'Timeout', r'Retry',
                r'Slow', r'Performance'
            ]
            
            performance_patterns = [
                r'slow', r'timeout', r'memory', r'cpu', r'performance',
                r'bottleneck', r'latency'
            ]
            
            for line in log_lines:
                # 에러 검출
                for pattern in error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        analysis.error_count += 1
                        if len(analysis.recent_errors) < 10:
                            analysis.recent_errors.append(line.strip())
                        break
                
                # 경고 검출
                for pattern in warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        analysis.warning_count += 1
                        break
                
                # 성능 이슈 검출
                for pattern in performance_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        if len(analysis.performance_issues) < 10:
                            analysis.performance_issues.append(line.strip())
                        break
            
            # 권장사항 생성
            if analysis.error_count > 10:
                analysis.recommendations.append(f"에러가 많이 발생함 ({analysis.error_count}개) - 설정 검토 필요")
            
            if analysis.warning_count > 20:
                analysis.recommendations.append(f"경고가 많이 발생함 ({analysis.warning_count}개) - 튜닝 고려")
            
            if analysis.performance_issues:
                analysis.recommendations.append("성능 이슈 감지됨 - 최적화 필요")
            
            if analysis.error_count == 0 and analysis.warning_count < 5:
                analysis.recommendations.append("안정적으로 운영 중")
            
            return analysis
            
        except Exception as e:
            logger.error(f"로그 분석 중 오류 {server_id}: {e}")
            return LogAnalysis(
                server_id=server_id,
                recommendations=[f"로그 분석 실패: {str(e)}"]
            )
    
    async def get_server_performance(self, server_id: str) -> Dict[str, Any]:
        """서버 성능 정보 수집"""
        try:
            if server_id not in self.server_processes:
                return {"error": "서버 프로세스 정보 없음"}
            
            process_info = self.server_processes[server_id]
            
            if not process_info.pid:
                return {"status": "stopped", "metrics": {}}
            
            try:
                process = psutil.Process(process_info.pid)
                
                # 성능 메트릭 수집
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 네트워크 연결 수
                connections = len(process.connections())
                
                # 실행 시간
                create_time = datetime.fromtimestamp(process.create_time())
                uptime = datetime.now() - create_time
                
                metrics = {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "connections": connections,
                    "uptime_seconds": uptime.total_seconds(),
                    "restart_count": process_info.restart_count,
                    "state": process_info.state.value,
                    "last_restart": process_info.last_restart.isoformat() if process_info.last_restart else None
                }
                
                # 프로세스 정보 업데이트
                process_info.cpu_percent = cpu_percent
                process_info.memory_mb = memory_mb
                
                # 성능 히스토리 저장
                if server_id not in self.performance_history:
                    self.performance_history[server_id] = []
                
                self.performance_history[server_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "connections": connections
                })
                
                # 히스토리 제한 (최근 100개)
                if len(self.performance_history[server_id]) > 100:
                    self.performance_history[server_id] = self.performance_history[server_id][-100:]
                
                return {"status": "running", "metrics": metrics}
                
            except psutil.NoSuchProcess:
                process_info.state = ServerState.STOPPED
                process_info.pid = None
                return {"status": "process_not_found", "metrics": {}}
                
        except Exception as e:
            logger.error(f"성능 정보 수집 중 오류 {server_id}: {e}")
            return {"error": str(e)}
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """전체 시스템 요약"""
        try:
            enabled_servers = self.config_manager.get_enabled_servers()
            total_servers = len(enabled_servers)
            
            running_count = 0
            stopped_count = 0
            error_count = 0
            
            server_details = {}
            
            for server_id in enabled_servers.keys():
                if server_id in self.server_processes:
                    process_info = self.server_processes[server_id]
                    
                    if process_info.state == ServerState.RUNNING:
                        running_count += 1
                    elif process_info.state == ServerState.STOPPED:
                        stopped_count += 1
                    elif process_info.state == ServerState.ERROR:
                        error_count += 1
                    
                    # 성능 정보 수집
                    performance = await self.get_server_performance(server_id)
                    
                    server_details[server_id] = {
                        "state": process_info.state.value,
                        "pid": process_info.pid,
                        "restart_count": process_info.restart_count,
                        "performance": performance
                    }
                else:
                    stopped_count += 1
                    server_details[server_id] = {
                        "state": "unknown",
                        "pid": None,
                        "restart_count": 0,
                        "performance": {"status": "no_data"}
                    }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_servers": total_servers,
                "running": running_count,
                "stopped": stopped_count,
                "error": error_count,
                "uptime_percentage": (running_count / total_servers * 100) if total_servers > 0 else 0,
                "servers": server_details,
                "management_features": {
                    "auto_restart": True,
                    "config_validation": True,
                    "log_analysis": True,
                    "performance_monitoring": True
                }
            }
            
        except Exception as e:
            logger.error(f"시스템 요약 생성 중 오류: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# 전역 서버 관리자 인스턴스
_server_manager = None

def get_server_manager() -> MCPServerManager:
    """전역 MCP 서버 관리자 인스턴스 반환"""
    global _server_manager
    if _server_manager is None:
        _server_manager = MCPServerManager()
    return _server_manager 