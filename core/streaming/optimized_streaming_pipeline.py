"""
최적화된 비동기 스트리밍 파이프라인
Phase 2.3: A2A SSE 스트리밍 성능 최적화

기능:
- 적응적 청크 크기 조정
- 백프레셔 처리 및 흐름 제어
- 지능형 버퍼링 전략
- 스트리밍 품질 모니터링
- 네트워크 조건 적응
- 에러 복구 및 재시작
"""

import asyncio
import time
import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable, Union
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class StreamingState(Enum):
    """스트리밍 상태"""
    INITIALIZING = "initializing"
    STREAMING = "streaming"
    BUFFERING = "buffering"
    BACKPRESSURE = "backpressure"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ChunkType(Enum):
    """청크 타입"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    METADATA = "metadata"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class StreamChunk:
    """스트리밍 청크"""
    id: str
    type: ChunkType
    data: Any
    timestamp: datetime
    size_bytes: int
    sequence: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse_format(self) -> str:
        """SSE 형식으로 변환"""
        sse_data = {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "metadata": self.metadata
        }
        
        json_data = json.dumps(sse_data, ensure_ascii=False, default=str)
        return f"data: {json_data}\n\n"

@dataclass
class StreamingMetrics:
    """스트리밍 메트릭"""
    total_chunks: int = 0
    total_bytes: int = 0
    avg_chunk_size: float = 0.0
    max_chunk_size: int = 0
    min_chunk_size: int = float('inf')
    throughput_bps: float = 0.0
    latency_ms: float = 0.0
    buffer_usage_percent: float = 0.0
    backpressure_events: int = 0
    error_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """스트리밍 지속 시간"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def avg_throughput_bps(self) -> float:
        """평균 처리량 (bytes per second)"""
        if self.duration_seconds > 0:
            return self.total_bytes / self.duration_seconds
        return 0.0

@dataclass
class BufferConfig:
    """버퍼 설정"""
    max_buffer_size: int = 1024 * 1024  # 1MB
    low_watermark: float = 0.3  # 30%
    high_watermark: float = 0.8  # 80%
    chunk_timeout_ms: float = 100.0
    max_chunks_in_buffer: int = 1000
    adaptive_sizing: bool = True

@dataclass
class ChunkingConfig:
    """청킹 설정"""
    min_chunk_size: int = 256
    max_chunk_size: int = 8192
    target_chunk_size: int = 2048
    adaptive_chunking: bool = True
    chunk_overlap: int = 0
    compression_threshold: int = 1024

class AdaptiveChunker:
    """적응적 청킹 시스템"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.current_chunk_size = config.target_chunk_size
        self.performance_history: deque = deque(maxlen=50)
        self.network_conditions: Dict[str, float] = {
            "latency_ms": 0.0,
            "bandwidth_bps": 0.0,
            "packet_loss_rate": 0.0
        }
    
    def calculate_optimal_chunk_size(self, content_size: int, network_conditions: Dict[str, float]) -> int:
        """최적 청크 크기 계산"""
        self.network_conditions.update(network_conditions)
        
        if not self.config.adaptive_chunking:
            return self.config.target_chunk_size
        
        # 네트워크 조건 기반 조정
        latency = network_conditions.get("latency_ms", 50.0)
        bandwidth = network_conditions.get("bandwidth_bps", 1000000.0)  # 1Mbps 기본값
        
        # 고지연 네트워크에서는 큰 청크 사용
        if latency > 100:
            size_factor = min(2.0, latency / 50)
        # 저지연 네트워크에서는 작은 청크로 더 빠른 응답성
        else:
            size_factor = max(0.5, latency / 100)
        
        # 대역폭 기반 조정
        if bandwidth < 100000:  # 100kbps 미만
            size_factor *= 0.5
        elif bandwidth > 10000000:  # 10Mbps 초과
            size_factor *= 1.5
        
        # 컨텐츠 크기 기반 조정
        if content_size < 1024:  # 작은 컨텐츠
            size_factor *= 0.7
        elif content_size > 100000:  # 큰 컨텐츠
            size_factor *= 1.3
        
        optimal_size = int(self.config.target_chunk_size * size_factor)
        optimal_size = max(self.config.min_chunk_size, min(self.config.max_chunk_size, optimal_size))
        
        self.current_chunk_size = optimal_size
        return optimal_size
    
    async def chunk_content(self, content: Union[str, bytes], chunk_type: ChunkType) -> AsyncGenerator[StreamChunk, None]:
        """컨텐츠를 최적화된 청크로 분할"""
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        content_size = len(content_bytes)
        chunk_size = self.calculate_optimal_chunk_size(
            content_size, 
            self.network_conditions
        )
        
        sequence = 0
        position = 0
        
        while position < content_size:
            # 청크 크기 동적 조정
            remaining = content_size - position
            current_chunk_size = min(chunk_size, remaining)
            
            # 오버랩 고려 (텍스트 경계 맞추기 등)
            if self.config.chunk_overlap > 0 and position > 0:
                overlap_start = max(0, position - self.config.chunk_overlap)
                chunk_data = content_bytes[overlap_start:position + current_chunk_size]
            else:
                chunk_data = content_bytes[position:position + current_chunk_size]
            
            # 텍스트 경계 조정 (UTF-8 안전성)
            if chunk_type == ChunkType.TEXT and position + current_chunk_size < content_size:
                chunk_data = self._adjust_text_boundary(chunk_data)
            
            chunk = StreamChunk(
                id=str(uuid.uuid4()),
                type=chunk_type,
                data=chunk_data.decode('utf-8') if chunk_type == ChunkType.TEXT else chunk_data,
                timestamp=datetime.now(),
                size_bytes=len(chunk_data),
                sequence=sequence,
                metadata={
                    "total_size": content_size,
                    "position": position,
                    "is_final": position + current_chunk_size >= content_size
                }
            )
            
            yield chunk
            
            position += current_chunk_size
            sequence += 1
    
    def _adjust_text_boundary(self, chunk_data: bytes) -> bytes:
        """텍스트 경계 조정 (UTF-8 안전)"""
        try:
            # 마지막 10바이트 내에서 안전한 UTF-8 경계 찾기
            for i in range(min(10, len(chunk_data))):
                try:
                    test_chunk = chunk_data[:-i] if i > 0 else chunk_data
                    test_chunk.decode('utf-8')
                    return test_chunk
                except UnicodeDecodeError:
                    continue
        except:
            pass
        
        return chunk_data
    
    def update_performance(self, throughput_bps: float, latency_ms: float):
        """성능 정보 업데이트"""
        self.performance_history.append({
            "timestamp": datetime.now(),
            "throughput_bps": throughput_bps,
            "latency_ms": latency_ms,
            "chunk_size": self.current_chunk_size
        })
        
        # 성능 기반 청크 크기 미세 조정
        if len(self.performance_history) >= 10:
            recent_performance = list(self.performance_history)[-10:]
            avg_latency = statistics.mean([p["latency_ms"] for p in recent_performance])
            avg_throughput = statistics.mean([p["throughput_bps"] for p in recent_performance])
            
            # 지연시간이 증가하면 청크 크기 증가 (fewer requests)
            if avg_latency > self.network_conditions.get("latency_ms", 50.0) * 1.2:
                self.current_chunk_size = min(
                    self.config.max_chunk_size,
                    int(self.current_chunk_size * 1.1)
                )
            # 처리량이 감소하면 청크 크기 감소 (better responsiveness)
            elif avg_throughput < self.network_conditions.get("bandwidth_bps", 1000000.0) * 0.8:
                self.current_chunk_size = max(
                    self.config.min_chunk_size,
                    int(self.current_chunk_size * 0.9)
                )

class StreamBuffer:
    """스트리밍 버퍼 관리"""
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self.buffer: deque = deque()
        self.buffer_size_bytes = 0
        self.is_draining = False
        self.backpressure_active = False
        self._buffer_lock = asyncio.Lock()
        
        # 버퍼 상태 추적
        self.buffer_stats = {
            "max_size_reached": 0,
            "backpressure_events": 0,
            "drain_events": 0,
            "overflow_events": 0
        }
    
    async def add_chunk(self, chunk: StreamChunk) -> bool:
        """청크를 버퍼에 추가"""
        async with self._buffer_lock:
            # 버퍼 용량 체크
            if self._is_buffer_full():
                self.buffer_stats["overflow_events"] += 1
                logger.warning(f"버퍼 오버플로우: {self.buffer_size_bytes}/{self.config.max_buffer_size} bytes")
                return False
            
            self.buffer.append(chunk)
            self.buffer_size_bytes += chunk.size_bytes
            
            # 백프레셔 상태 확인
            self._check_backpressure()
            
            return True
    
    async def get_chunk(self) -> Optional[StreamChunk]:
        """버퍼에서 청크 가져오기"""
        async with self._buffer_lock:
            if not self.buffer:
                return None
            
            chunk = self.buffer.popleft()
            self.buffer_size_bytes -= chunk.size_bytes
            
            # 드레인 상태 확인
            self._check_drain_status()
            
            return chunk
    
    async def get_chunks_batch(self, max_count: int = 10) -> List[StreamChunk]:
        """여러 청크를 배치로 가져오기"""
        chunks = []
        
        async with self._buffer_lock:
            for _ in range(min(max_count, len(self.buffer))):
                if self.buffer:
                    chunk = self.buffer.popleft()
                    self.buffer_size_bytes -= chunk.size_bytes
                    chunks.append(chunk)
                else:
                    break
            
            self._check_drain_status()
        
        return chunks
    
    def _is_buffer_full(self) -> bool:
        """버퍼가 가득 찼는지 확인"""
        size_full = self.buffer_size_bytes >= self.config.max_buffer_size
        count_full = len(self.buffer) >= self.config.max_chunks_in_buffer
        return size_full or count_full
    
    def _check_backpressure(self):
        """백프레셔 상태 확인"""
        usage_ratio = self.buffer_size_bytes / self.config.max_buffer_size
        
        if usage_ratio > self.config.high_watermark and not self.backpressure_active:
            self.backpressure_active = True
            self.buffer_stats["backpressure_events"] += 1
            logger.info(f"백프레셔 활성화: 버퍼 사용량 {usage_ratio:.2%}")
        
        elif usage_ratio < self.config.low_watermark and self.backpressure_active:
            self.backpressure_active = False
            logger.info(f"백프레셔 해제: 버퍼 사용량 {usage_ratio:.2%}")
    
    def _check_drain_status(self):
        """드레인 상태 확인"""
        usage_ratio = self.buffer_size_bytes / self.config.max_buffer_size
        
        if usage_ratio < self.config.low_watermark and self.is_draining:
            self.is_draining = False
            self.buffer_stats["drain_events"] += 1
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """버퍼 상태 반환"""
        usage_ratio = self.buffer_size_bytes / self.config.max_buffer_size
        
        return {
            "buffer_size_bytes": self.buffer_size_bytes,
            "buffer_chunk_count": len(self.buffer),
            "usage_ratio": usage_ratio,
            "is_draining": self.is_draining,
            "backpressure_active": self.backpressure_active,
            "stats": self.buffer_stats.copy()
        }

class OptimizedStreamingPipeline:
    """최적화된 스트리밍 파이프라인"""
    
    def __init__(self, 
                 buffer_config: Optional[BufferConfig] = None,
                 chunking_config: Optional[ChunkingConfig] = None):
        # 설정
        self.buffer_config = buffer_config or BufferConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        
        # 컴포넌트
        self.chunker = AdaptiveChunker(self.chunking_config)
        self.buffer = StreamBuffer(self.buffer_config)
        
        # 상태
        self.state = StreamingState.INITIALIZING
        self.session_id = str(uuid.uuid4())
        self.metrics = StreamingMetrics()
        
        # 스트리밍 제어
        self.is_streaming = False
        self.should_stop = False
        self._consumers: List[Callable] = []
        
        # 백프레셔 제어
        self.backpressure_threshold = 0.8
        self.flow_control_enabled = True
        
        # 에러 처리
        self.error_count = 0
        self.max_errors = 5
        self.retry_delay = 1.0
        
        # 모니터링
        self.monitoring_enabled = True
        self.quality_metrics: deque = deque(maxlen=100)
        
        # 결과 저장 경로
        self.results_dir = Path("monitoring/streaming_performance")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def start_stream(self, content_source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """최적화된 스트리밍 시작"""
        self.state = StreamingState.STREAMING
        self.metrics.start_time = datetime.now()
        self.is_streaming = True
        
        logger.info(f"🚀 최적화된 스트리밍 시작 (세션: {self.session_id})")
        
        try:
            # 프로듀서 태스크 (컨텐츠 생성 및 청킹)
            producer_task = asyncio.create_task(
                self._producer_loop(content_source)
            )
            
            # 컨슈머 태스크 (SSE 전송)
            consumer_task = asyncio.create_task(
                self._consumer_loop()
            )
            
            # 모니터링 태스크
            monitoring_task = asyncio.create_task(
                self._monitoring_loop()
            ) if self.monitoring_enabled else None
            
            # SSE 스트림 생성
            async for sse_chunk in self._sse_generator():
                if self.should_stop:
                    break
                yield sse_chunk
                
                # 백프레셔 제어
                if self.flow_control_enabled:
                    await self._handle_backpressure()
            
            # 정리
            producer_task.cancel()
            consumer_task.cancel()
            if monitoring_task:
                monitoring_task.cancel()
            
            await self._cleanup_tasks([producer_task, consumer_task, monitoring_task])
            
        except Exception as e:
            logger.error(f"스트리밍 중 오류: {e}")
            self.state = StreamingState.ERROR
            self.error_count += 1
            
            # 에러 청크 전송
            error_chunk = StreamChunk(
                id=str(uuid.uuid4()),
                type=ChunkType.ERROR,
                data={"error": str(e), "session_id": self.session_id},
                timestamp=datetime.now(),
                size_bytes=len(str(e)),
                sequence=-1
            )
            yield error_chunk.to_sse_format()
            
        finally:
            await self._finalize_stream()
    
    async def _producer_loop(self, content_source: AsyncGenerator[str, None]):
        """프로듀서 루프 - 컨텐츠 생성 및 청킹"""
        try:
            async for content in content_source:
                if self.should_stop:
                    break
                
                # 백프레셔 체크
                if self.buffer.backpressure_active:
                    await asyncio.sleep(0.1)
                    continue
                
                # 컨텐츠 청킹
                async for chunk in self.chunker.chunk_content(content, ChunkType.TEXT):
                    success = await self.buffer.add_chunk(chunk)
                    
                    if not success:
                        logger.warning("버퍼 가득참으로 청크 드롭")
                        await asyncio.sleep(0.05)  # 짧은 백오프
                        
                        # 재시도
                        if await self.buffer.add_chunk(chunk):
                            logger.info("청크 재시도 성공")
                        else:
                            logger.error("청크 재시도 실패")
                            break
                    
                    # 메트릭 업데이트
                    self.metrics.total_chunks += 1
                    self.metrics.total_bytes += chunk.size_bytes
                    
                    if self.should_stop:
                        break
                        
        except Exception as e:
            logger.error(f"프로듀서 루프 오류: {e}")
            self.error_count += 1
    
    async def _consumer_loop(self):
        """컨슈머 루프 - 버퍼에서 청크 소비"""
        try:
            while self.is_streaming and not self.should_stop:
                # 배치로 청크 가져오기
                chunks = await self.buffer.get_chunks_batch(max_count=5)
                
                if not chunks:
                    await asyncio.sleep(0.01)  # 10ms 대기
                    continue
                
                # 청크 처리
                for chunk in chunks:
                    await self._process_chunk(chunk)
                
        except Exception as e:
            logger.error(f"컨슈머 루프 오류: {e}")
            self.error_count += 1
    
    async def _sse_generator(self) -> AsyncGenerator[str, None]:
        """SSE 형식으로 청크 전송"""
        chunk_count = 0
        
        while self.is_streaming and not self.should_stop:
            chunk = await self.buffer.get_chunk()
            
            if chunk is None:
                # 하트비트 전송 (연결 유지)
                if chunk_count % 50 == 0:  # 주기적으로
                    heartbeat = StreamChunk(
                        id=str(uuid.uuid4()),
                        type=ChunkType.HEARTBEAT,
                        data={"status": "alive", "session_id": self.session_id},
                        timestamp=datetime.now(),
                        size_bytes=0,
                        sequence=-1
                    )
                    yield heartbeat.to_sse_format()
                
                await asyncio.sleep(0.01)
                continue
            
            # SSE 형식으로 변환 후 전송
            sse_data = chunk.to_sse_format()
            yield sse_data
            
            chunk_count += 1
            
            # 품질 메트릭 수집
            await self._collect_quality_metrics(chunk)
    
    async def _handle_backpressure(self):
        """백프레셔 처리"""
        buffer_status = self.buffer.get_buffer_status()
        
        if buffer_status["backpressure_active"]:
            self.state = StreamingState.BACKPRESSURE
            
            # 적응적 지연
            usage_ratio = buffer_status["usage_ratio"]
            delay = 0.01 + (usage_ratio - self.backpressure_threshold) * 0.1
            
            await asyncio.sleep(delay)
            
            logger.debug(f"백프레셔 처리: {delay:.3f}초 지연")
        
        elif self.state == StreamingState.BACKPRESSURE:
            self.state = StreamingState.STREAMING
    
    async def _process_chunk(self, chunk: StreamChunk):
        """청크 처리 (전처리, 압축 등)"""
        try:
            # 압축 고려
            if (chunk.size_bytes > self.chunking_config.compression_threshold and
                chunk.type in [ChunkType.TEXT, ChunkType.JSON]):
                # 여기에 압축 로직 추가 가능
                pass
            
            # 품질 검증
            await self._validate_chunk_quality(chunk)
            
        except Exception as e:
            logger.error(f"청크 처리 오류: {e}")
            self.error_count += 1
    
    async def _validate_chunk_quality(self, chunk: StreamChunk):
        """청크 품질 검증"""
        # 크기 검증
        if chunk.size_bytes > self.chunking_config.max_chunk_size:
            logger.warning(f"청크 크기 초과: {chunk.size_bytes} > {self.chunking_config.max_chunk_size}")
        
        # 타입 검증
        if chunk.type == ChunkType.TEXT and isinstance(chunk.data, str):
            # UTF-8 유효성 검사
            try:
                chunk.data.encode('utf-8')
            except UnicodeEncodeError as e:
                logger.error(f"UTF-8 인코딩 오류: {e}")
                raise
        
        # 순서 검증 (기본적인 체크)
        if hasattr(self, '_last_sequence') and chunk.sequence > 0:
            if chunk.sequence != self._last_sequence + 1:
                logger.warning(f"청크 순서 불일치: {chunk.sequence} (예상: {self._last_sequence + 1})")
        
        self._last_sequence = chunk.sequence
    
    async def _collect_quality_metrics(self, chunk: StreamChunk):
        """품질 메트릭 수집"""
        current_time = datetime.now()
        
        # 지연시간 계산
        latency_ms = (current_time - chunk.timestamp).total_seconds() * 1000
        
        quality_metric = {
            "timestamp": current_time,
            "chunk_id": chunk.id,
            "latency_ms": latency_ms,
            "chunk_size": chunk.size_bytes,
            "buffer_usage": self.buffer.get_buffer_status()["usage_ratio"]
        }
        
        self.quality_metrics.append(quality_metric)
        
        # 성능 정보를 청커에 피드백
        if len(self.quality_metrics) >= 10:
            recent_metrics = list(self.quality_metrics)[-10:]
            avg_latency = statistics.mean([m["latency_ms"] for m in recent_metrics])
            avg_throughput = self.metrics.avg_throughput_bps
            
            self.chunker.update_performance(avg_throughput, avg_latency)
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        try:
            while self.is_streaming and not self.should_stop:
                # 스트리밍 상태 로깅
                buffer_status = self.buffer.get_buffer_status()
                
                if self.metrics.total_chunks % 100 == 0:  # 100청크마다
                    logger.info(
                        f"스트리밍 상태 - 청크: {self.metrics.total_chunks}, "
                        f"버퍼: {buffer_status['usage_ratio']:.2%}, "
                        f"상태: {self.state.value}"
                    )
                
                await asyncio.sleep(5)  # 5초마다 모니터링
                
        except Exception as e:
            logger.error(f"모니터링 루프 오류: {e}")
    
    async def _cleanup_tasks(self, tasks: List[asyncio.Task]):
        """태스크 정리"""
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"태스크 정리 오류: {e}")
    
    async def _finalize_stream(self):
        """스트리밍 종료 처리"""
        self.is_streaming = False
        self.metrics.end_time = datetime.now()
        self.state = StreamingState.COMPLETED
        
        # 최종 메트릭 계산
        self._calculate_final_metrics()
        
        # 결과 저장
        await self._save_streaming_results()
        
        logger.info(f"✅ 스트리밍 완료 (세션: {self.session_id}) - "
                   f"청크: {self.metrics.total_chunks}, "
                   f"처리량: {self.metrics.avg_throughput_bps:.0f} bps")
    
    def _calculate_final_metrics(self):
        """최종 메트릭 계산"""
        if self.metrics.total_chunks > 0:
            self.metrics.avg_chunk_size = self.metrics.total_bytes / self.metrics.total_chunks
        
        if self.quality_metrics:
            latencies = [m["latency_ms"] for m in self.quality_metrics]
            self.metrics.latency_ms = statistics.mean(latencies)
        
        buffer_status = self.buffer.get_buffer_status()
        self.metrics.buffer_usage_percent = buffer_status["usage_ratio"] * 100
        self.metrics.backpressure_events = buffer_status["stats"]["backpressure_events"]
        self.metrics.error_count = self.error_count
    
    async def _save_streaming_results(self):
        """스트리밍 결과 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results = {
                "session_id": self.session_id,
                "metrics": {
                    "total_chunks": self.metrics.total_chunks,
                    "total_bytes": self.metrics.total_bytes,
                    "avg_chunk_size": self.metrics.avg_chunk_size,
                    "duration_seconds": self.metrics.duration_seconds,
                    "avg_throughput_bps": self.metrics.avg_throughput_bps,
                    "latency_ms": self.metrics.latency_ms,
                    "backpressure_events": self.metrics.backpressure_events,
                    "error_count": self.metrics.error_count
                },
                "buffer_config": {
                    "max_buffer_size": self.buffer_config.max_buffer_size,
                    "high_watermark": self.buffer_config.high_watermark,
                    "low_watermark": self.buffer_config.low_watermark
                },
                "chunking_config": {
                    "target_chunk_size": self.chunking_config.target_chunk_size,
                    "adaptive_chunking": self.chunking_config.adaptive_chunking
                },
                "quality_metrics": list(self.quality_metrics),
                "buffer_stats": self.buffer.get_buffer_status()["stats"]
            }
            
            file_path = self.results_dir / f"streaming_session_{timestamp}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 스트리밍 결과 저장: {file_path}")
            
        except Exception as e:
            logger.error(f"결과 저장 오류: {e}")
    
    def stop_stream(self):
        """스트리밍 중지"""
        self.should_stop = True
        logger.info(f"⏹️ 스트리밍 중지 요청 (세션: {self.session_id})")
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """스트리밍 상태 반환"""
        buffer_status = self.buffer.get_buffer_status()
        
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "is_streaming": self.is_streaming,
            "metrics": {
                "total_chunks": self.metrics.total_chunks,
                "total_bytes": self.metrics.total_bytes,
                "avg_throughput_bps": self.metrics.avg_throughput_bps,
                "error_count": self.error_count
            },
            "buffer_status": buffer_status,
            "chunker_status": {
                "current_chunk_size": self.chunker.current_chunk_size,
                "adaptive_enabled": self.chunking_config.adaptive_chunking
            }
        }


# 사용 예시 및 테스트
async def sample_content_generator() -> AsyncGenerator[str, None]:
    """샘플 컨텐츠 생성기"""
    for i in range(100):
        content = f"이것은 스트리밍 테스트 메시지입니다. 청크 번호: {i}. " * 5
        yield content
        await asyncio.sleep(0.1)  # 100ms 간격

async def test_optimized_streaming():
    """최적화된 스트리밍 테스트"""
    # 설정
    buffer_config = BufferConfig(
        max_buffer_size=1024*512,  # 512KB
        high_watermark=0.8,
        low_watermark=0.3
    )
    
    chunking_config = ChunkingConfig(
        target_chunk_size=1024,
        adaptive_chunking=True
    )
    
    # 파이프라인 생성
    pipeline = OptimizedStreamingPipeline(buffer_config, chunking_config)
    
    # 스트리밍 테스트
    content_source = sample_content_generator()
    
    chunk_count = 0
    async for sse_chunk in pipeline.start_stream(content_source):
        print(f"수신한 SSE 청크: {len(sse_chunk)} bytes")
        chunk_count += 1
        
        if chunk_count >= 50:  # 50개 청크로 제한
            pipeline.stop_stream()
            break
    
    # 결과 출력
    status = pipeline.get_streaming_status()
    print(f"🚀 스트리밍 완료: {status['metrics']['total_chunks']}개 청크, "
          f"{status['metrics']['total_bytes']} bytes")

if __name__ == "__main__":
    asyncio.run(test_optimized_streaming()) 