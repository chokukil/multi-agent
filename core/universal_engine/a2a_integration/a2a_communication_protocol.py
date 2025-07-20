"""
A2A Communication Protocol - A2A 에이전트 통신 프로토콜

요구사항에 따른 구현:
- A2A SDK 0.2.9 표준 준수 통신 로직
- 향상된 컨텍스트로 에이전트 요청 생성
- 타임아웃 및 재시도 메커니즘
- 에이전트 응답 파싱 및 검증
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import uuid
from enum import Enum

from .a2a_agent_discovery import A2AAgentInfo

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """A2A 요청 타입"""
    ANALYSIS = "analysis"
    PROCESSING = "processing"
    VISUALIZATION = "visualization"
    QUERY = "query"
    HEALTH_CHECK = "health"


@dataclass
class A2ARequest:
    """A2A 요청 구조"""
    id: str
    type: RequestType
    payload: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: str
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'id': self.id,
            'type': self.type.value,
            'payload': self.payload,
            'context': self.context,
            'timestamp': self.timestamp,
            'timeout': self.timeout
        }


@dataclass 
class A2AResponse:
    """A2A 응답 구조"""
    request_id: str
    agent_id: str
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    execution_time: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AResponse':
        """딕셔너리에서 생성"""
        return cls(
            request_id=data.get('request_id', ''),
            agent_id=data.get('agent_id', ''),
            status=data.get('status', 'unknown'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', ''),
            execution_time=data.get('execution_time', 0.0)
        )


class A2ACommunicationProtocol:
    """
    A2A 에이전트 통신 프로토콜
    - A2A SDK 0.2.9 표준 준수
    - 향상된 컨텍스트 및 메타데이터 지원
    - 안정적인 통신 및 오류 처리
    """
    
    def __init__(self, default_timeout: int = 30, max_retries: int = 3):
        """
        A2ACommunicationProtocol 초기화
        
        Args:
            default_timeout: 기본 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.request_history: List[Dict] = []
        self.active_requests: Dict[str, A2ARequest] = {}
        logger.info("A2ACommunicationProtocol initialized")
    
    async def send_request(
        self,
        agent: A2AAgentInfo,
        payload: Dict[str, Any],
        request_type: RequestType = RequestType.ANALYSIS,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> A2AResponse:
        """
        A2A 에이전트에 요청 전송
        
        Args:
            agent: 대상 에이전트 정보
            payload: 요청 페이로드
            request_type: 요청 타입
            context: 추가 컨텍스트
            timeout: 타임아웃 (초)
            
        Returns:
            A2A 응답
        """
        request_id = str(uuid.uuid4())
        request_timeout = timeout or self.default_timeout
        
        logger.info(f"Sending A2A request {request_id} to agent {agent.name}")
        
        # A2A 요청 생성
        request = A2ARequest(
            id=request_id,
            type=request_type,
            payload=payload,
            context=self._enhance_context(context, agent),
            timestamp=datetime.now().isoformat(),
            timeout=request_timeout
        )
        
        self.active_requests[request_id] = request
        
        try:
            # 재시도 로직과 함께 요청 전송
            response = await self._send_with_retry(agent, request)
            
            # 요청 이력에 저장
            self._record_request_history(agent, request, response, success=True)
            
            return response
            
        except Exception as e:
            logger.error(f"A2A request {request_id} failed: {e}")
            
            # 실패한 요청도 이력에 저장
            error_response = A2AResponse(
                request_id=request_id,
                agent_id=agent.id,
                status="error",
                data={"error": str(e)},
                metadata={"error_type": type(e).__name__},
                timestamp=datetime.now().isoformat(),
                execution_time=0.0
            )
            
            self._record_request_history(agent, request, error_response, success=False)
            raise
            
        finally:
            # 활성 요청에서 제거
            self.active_requests.pop(request_id, None)
    
    async def _send_with_retry(
        self, 
        agent: A2AAgentInfo, 
        request: A2ARequest
    ) -> A2AResponse:
        """
        재시도 로직을 포함한 요청 전송
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # 지수 백오프 대기
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Retrying request {request.id} (attempt {attempt + 1}) after {wait_time}s")
                    await asyncio.sleep(wait_time)
                
                return await self._send_single_request(agent, request)
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Request {request.id} timed out on attempt {attempt + 1}")
                
            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(f"Network error for request {request.id} on attempt {attempt + 1}: {e}")
                
            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"Invalid JSON response for request {request.id}: {e}")
                # JSON 오류는 재시도하지 않음
                break
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Unexpected error for request {request.id}: {e}")
        
        # 모든 재시도 실패
        raise last_exception or Exception("All retry attempts failed")
    
    async def _send_single_request(
        self, 
        agent: A2AAgentInfo, 
        request: A2ARequest
    ) -> A2AResponse:
        """
        단일 요청 전송
        """
        start_time = datetime.now()
        endpoint_url = f"{agent.base_url}/process"
        
        # A2A SDK 0.2.9 표준 준수 요청 구조
        request_data = {
            "id": request.id,
            "method": "process",
            "params": {
                "type": request.type.value,
                "data": request.payload,
                "context": request.context,
                "metadata": {
                    "timestamp": request.timestamp,
                    "source": "universal_engine",
                    "version": "1.0.0"
                }
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=request.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                endpoint_url,
                json=request_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "UniversalEngine/1.0.0",
                    "X-Request-ID": request.id
                }
            ) as response:
                
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP {response.status}"
                    )
                
                response_data = await response.json()
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # A2A 표준 응답 파싱
                return self._parse_a2a_response(
                    response_data, request.id, agent.id, execution_time
                )
    
    def _enhance_context(
        self, 
        base_context: Optional[Dict[str, Any]], 
        agent: A2AAgentInfo
    ) -> Dict[str, Any]:
        """
        컨텍스트 향상
        """
        enhanced_context = {
            "universal_engine": {
                "version": "1.0.0",
                "request_time": datetime.now().isoformat(),
                "agent_selection_reason": "llm_based_dynamic_selection"
            },
            "agent_info": {
                "id": agent.id,
                "name": agent.name,
                "capabilities": agent.capabilities,
                "version": agent.version
            },
            "execution_context": {
                "workflow_id": None,  # 워크플로우에서 설정
                "task_sequence": None,  # 워크플로우에서 설정
                "parallel_execution": False  # 워크플로우에서 설정
            }
        }
        
        # 기본 컨텍스트와 병합
        if base_context:
            enhanced_context.update(base_context)
        
        return enhanced_context
    
    def _parse_a2a_response(
        self, 
        response_data: Dict[str, Any],
        request_id: str,
        agent_id: str,
        execution_time: float
    ) -> A2AResponse:
        """
        A2A 응답 파싱
        """
        # A2A SDK 0.2.9 표준 응답 구조 파싱
        if "result" in response_data:
            # 성공 응답
            result = response_data["result"]
            return A2AResponse(
                request_id=request_id,
                agent_id=agent_id,
                status="success",
                data=result.get("data", {}),
                metadata=result.get("metadata", {}),
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time
            )
        elif "error" in response_data:
            # 오류 응답
            error = response_data["error"]
            return A2AResponse(
                request_id=request_id,
                agent_id=agent_id,
                status="error",
                data={"error": error.get("message", "Unknown error")},
                metadata={"error_code": error.get("code", -1)},
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time
            )
        else:
            # 비표준 응답
            return A2AResponse(
                request_id=request_id,
                agent_id=agent_id,
                status="unknown",
                data=response_data,
                metadata={"response_type": "non_standard"},
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time
            )
    
    def _record_request_history(
        self,
        agent: A2AAgentInfo,
        request: A2ARequest,
        response: A2AResponse,
        success: bool
    ) -> None:
        """
        요청 이력 기록
        """
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent.id,
            "agent_name": agent.name,
            "request_id": request.id,
            "request_type": request.type.value,
            "success": success,
            "execution_time": response.execution_time,
            "status": response.status
        }
        
        self.request_history.append(history_entry)
        
        # 이력 크기 제한 (최근 1000개만 유지)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def send_batch_requests(
        self,
        requests: List[tuple[A2AAgentInfo, Dict[str, Any]]],
        request_type: RequestType = RequestType.ANALYSIS,
        context: Optional[Dict[str, Any]] = None
    ) -> List[A2AResponse]:
        """
        배치 요청 전송
        
        Args:
            requests: (에이전트, 페이로드) 튜플 리스트
            request_type: 요청 타입
            context: 공통 컨텍스트
            
        Returns:
            응답 리스트
        """
        logger.info(f"Sending batch requests to {len(requests)} agents")
        
        # 모든 요청을 병렬로 실행
        tasks = [
            self.send_request(agent, payload, request_type, context)
            for agent, payload in requests
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외를 응답으로 변환
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                agent = requests[i][0]
                error_response = A2AResponse(
                    request_id=f"batch_error_{i}",
                    agent_id=agent.id,
                    status="error",
                    data={"error": str(response)},
                    metadata={"error_type": type(response).__name__},
                    timestamp=datetime.now().isoformat(),
                    execution_time=0.0
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    async def health_check_agent(self, agent: A2AAgentInfo) -> bool:
        """
        에이전트 헬스 체크
        
        Args:
            agent: 확인할 에이전트
            
        Returns:
            헬스 체크 성공 여부
        """
        try:
            response = await self.send_request(
                agent=agent,
                payload={"action": "health_check"},
                request_type=RequestType.HEALTH_CHECK,
                timeout=5  # 짧은 타임아웃
            )
            
            return response.status == "success"
            
        except Exception as e:
            logger.warning(f"Health check failed for agent {agent.name}: {e}")
            return False
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """
        통신 통계 조회
        
        Returns:
            통신 통계 정보
        """
        if not self.request_history:
            return {"message": "No communication history available"}
        
        total_requests = len(self.request_history)
        successful_requests = sum(1 for h in self.request_history if h["success"])
        
        # 에이전트별 통계
        agent_stats = {}
        for history in self.request_history:
            agent_name = history["agent_name"]
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "total_execution_time": 0.0
                }
            
            agent_stats[agent_name]["total_requests"] += 1
            if history["success"]:
                agent_stats[agent_name]["successful_requests"] += 1
            agent_stats[agent_name]["total_execution_time"] += history["execution_time"]
        
        # 평균 실행 시간
        avg_execution_time = sum(h["execution_time"] for h in self.request_history) / total_requests
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / total_requests,
            "average_execution_time": avg_execution_time,
            "active_requests": len(self.active_requests),
            "agent_statistics": agent_stats,
            "recent_activity": self.request_history[-10:]  # 최근 10개
        }
    
    def cancel_request(self, request_id: str) -> bool:
        """
        요청 취소
        
        Args:
            request_id: 취소할 요청 ID
            
        Returns:
            취소 성공 여부
        """
        if request_id in self.active_requests:
            # 실제 HTTP 요청 취소는 복잡하므로, 
            # 여기서는 활성 요청 목록에서만 제거
            del self.active_requests[request_id]
            logger.info(f"Request {request_id} cancelled")
            return True
        
        return False