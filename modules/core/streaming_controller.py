"""
Streaming Controller - 실시간 스트리밍 및 에이전트 협업 시각화

SSE (Server-Sent Events) 기반의 자연스러운 텍스트 스트리밍과 
A2A SDK TaskUpdater 패턴을 사용한 실시간 에이전트 협업 시각화
"""

import asyncio
import time
import logging
from typing import AsyncGenerator, Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import re
from dataclasses import asdict

from ..models import StreamingResponse, AgentProgressInfo, TaskState, ProgressInfo

logger = logging.getLogger(__name__)


class StreamingController:
    """
    실시간 스트리밍 컨트롤러
    - 자연스러운 타이핑 효과와 지능적 청킹
    - 실시간 에이전트 협업 시각화
    - A2A SDK TaskUpdater 패턴 지원
    """
    
    def __init__(self):
        """Streaming Controller 초기화"""
        self.chunk_delay = 0.001  # 자연스러운 타이핑 속도 (1ms)
        self.sentence_boundaries = ['.', '!', '?', '\n', ':', ';']
        self.chunk_size = 3  # 한 번에 전송할 문자 수
        self.status_update_interval = 0.5  # 에이전트 상태 업데이트 간격 (0.5초)
        
        # 활성 스트림 추적
        self.active_streams: Dict[str, Dict] = {}
        
        logger.info("Streaming Controller initialized")
    
    async def stream_response(self, 
                            content: str,
                            response_id: str,
                            agent_progress: Optional[List[AgentProgressInfo]] = None,
                            chunk_callback: Optional[Callable] = None) -> AsyncGenerator[StreamingResponse, None]:
        """
        자연스러운 스트리밍 응답:
        1. 지능적 의미 단위별 청킹
        2. 구두점에서 자연스러운 멈춤
        3. 실시간 에이전트 진행 상황 시각화
        4. 중단 시 우아한 성능 저하
        """
        try:
            logger.info(f"Starting stream response for {response_id}")
            
            # 스트림 상태 초기화
            self.active_streams[response_id] = {
                'start_time': datetime.now(),
                'status': 'streaming',
                'chunks_sent': 0,
                'agent_progress': agent_progress or []
            }
            
            # 의미 단위별 청킹
            semantic_chunks = self._chunk_by_semantic_units(content)
            
            total_chunks = len(semantic_chunks)
            
            for i, chunk in enumerate(semantic_chunks):
                try:
                    # 스트림이 중단되었는지 확인
                    if response_id not in self.active_streams:
                        logger.warning(f"Stream {response_id} was interrupted")
                        break
                    
                    # 진행 상황 정보 생성
                    progress_info = self._create_progress_info(
                        agent_progress, i + 1, total_chunks
                    ) if agent_progress else None
                    
                    # 스트리밍 응답 생성
                    streaming_response = StreamingResponse(
                        content=chunk,
                        is_complete=(i == total_chunks - 1),
                        chunk_index=i,
                        total_chunks=total_chunks,
                        progress_info=progress_info
                    )
                    
                    yield streaming_response
                    
                    # 청크 콜백 호출
                    if chunk_callback:
                        chunk_callback(chunk, i + 1, total_chunks)
                    
                    # 스트림 상태 업데이트
                    self.active_streams[response_id]['chunks_sent'] = i + 1
                    
                    # 자연스러운 타이핑 지연
                    await self._calculate_natural_delay(chunk)
                    
                except Exception as e:
                    logger.error(f"Error streaming chunk {i}: {str(e)}")
                    # 오류 발생 시에도 계속 진행
                    continue
            
            # 스트림 완료
            self.active_streams[response_id]['status'] = 'completed'
            self.active_streams[response_id]['end_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Streaming error for {response_id}: {str(e)}")
            
            # 오류 발생 시 우아한 성능 저하
            yield StreamingResponse(
                content=content,  # 전체 내용을 한 번에 전송
                is_complete=True,
                chunk_index=0,
                total_chunks=1
            )
        
        finally:
            # 정리
            if response_id in self.active_streams:
                self.active_streams[response_id]['status'] = 'finished'
    
    def _chunk_by_semantic_units(self, text: str) -> List[str]:
        """
        지능적 의미 단위별 텍스트 청킹:
        - 문장 경계에서 분할
        - 목록 아이템별 분할
        - 코드 블록 보존
        - 마크다운 구조 유지
        """
        chunks = []
        
        # 마크다운 코드 블록 처리
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        
        # 코드 블록을 임시 플레이스홀더로 교체
        temp_text = text
        for i, block in enumerate(code_blocks):
            temp_text = temp_text.replace(block, f'__CODE_BLOCK_{i}__', 1)
        
        # 문장별 분할
        sentences = self._split_by_sentences(temp_text)
        
        for sentence in sentences:
            # 코드 블록 복원
            for i, block in enumerate(code_blocks):
                sentence = sentence.replace(f'__CODE_BLOCK_{i}__', block)
            
            if sentence.strip():
                chunks.append(sentence.strip())
        
        # 빈 청크 제거 및 최소 길이 보장
        filtered_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            current_chunk += chunk + " "
            
            # 최소 길이에 도달하거나 문장이 완료되면 청크 추가
            if (len(current_chunk) >= 50 or 
                any(boundary in chunk for boundary in self.sentence_boundaries)):
                
                filtered_chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # 남은 내용 추가
        if current_chunk.strip():
            filtered_chunks.append(current_chunk.strip())
        
        return filtered_chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """문장별 텍스트 분할"""
        
        # 마크다운 목록 아이템 처리
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 목록 아이템은 개별 청크로 처리
            if line.startswith(('- ', '* ', '+ ', '1. ', '2. ', '3. ')):
                sentences.append(line)
                continue
            
            # 헤더는 개별 청크로 처리
            if line.startswith('#'):
                sentences.append(line)
                continue
            
            # 일반 텍스트는 문장별로 분할
            sentence_parts = re.split(r'([.!?:;]\s+)', line)
            
            current_sentence = ""
            for part in sentence_parts:
                current_sentence += part
                
                if any(boundary in part for boundary in self.sentence_boundaries):
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # 남은 내용 추가
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
        
        return sentences
    
    async def _calculate_natural_delay(self, chunk: str):
        """자연스러운 타이핑 지연 계산"""
        
        # 기본 지연
        base_delay = self.chunk_delay
        
        # 청크 길이에 따른 조정
        length_factor = len(chunk) / 50  # 50자 기준
        adjusted_delay = base_delay * (1 + length_factor * 0.1)
        
        # 구두점에서 추가 지연
        if any(boundary in chunk for boundary in ['.', '!', '?']):
            adjusted_delay += 0.1  # 100ms 추가 지연
        
        # 목록 아이템 후 짧은 지연
        if chunk.strip().startswith(('- ', '* ', '+ ')):
            adjusted_delay += 0.05  # 50ms 추가 지연
        
        # 최대 지연 제한
        adjusted_delay = min(adjusted_delay, 0.3)  # 최대 300ms
        
        if adjusted_delay > 0:
            await asyncio.sleep(adjusted_delay)
    
    def _create_progress_info(self, 
                            agent_progress: List[AgentProgressInfo],
                            current_chunk: int,
                            total_chunks: int) -> ProgressInfo:
        """진행 상황 정보 생성"""
        
        # 에이전트별 진행률 계산
        for agent in agent_progress:
            if agent.status == TaskState.WORKING:
                # 텍스트 청킹 진행률을 에이전트 진행률에 반영
                chunk_progress = (current_chunk / total_chunks) * 100
                agent.progress_percentage = min(chunk_progress, 90)  # 최대 90%까지
            elif agent.status == TaskState.COMPLETED:
                agent.progress_percentage = 100
        
        # 전체 진행률 계산
        total_agents = len(agent_progress)
        completed_agents = sum(1 for agent in agent_progress 
                             if agent.status == TaskState.COMPLETED)
        working_agents = sum(1 for agent in agent_progress 
                           if agent.status == TaskState.WORKING)
        
        # 가중 평균으로 전체 진행률 계산
        if total_agents > 0:
            completion_percentage = (completed_agents / total_agents) * 100
            
            # 작업 중인 에이전트의 부분 진행률 추가
            if working_agents > 0:
                avg_working_progress = sum(
                    agent.progress_percentage for agent in agent_progress
                    if agent.status == TaskState.WORKING
                ) / working_agents
                
                working_contribution = (working_agents / total_agents) * (avg_working_progress / 100) * 100
                completion_percentage = min(completion_percentage + working_contribution, 95)
        else:
            completion_percentage = 0
        
        return ProgressInfo(
            agents_working=agent_progress,
            current_step=f"스트리밍 진행 중... ({current_chunk}/{total_chunks})",
            total_steps=total_chunks,
            completion_percentage=completion_percentage,
            estimated_remaining_time=self._estimate_remaining_time(
                current_chunk, total_chunks, agent_progress
            )
        )
    
    def _estimate_remaining_time(self, 
                                current_chunk: int,
                                total_chunks: int,
                                agent_progress: List[AgentProgressInfo]) -> Optional[int]:
        """남은 시간 추정"""
        
        if current_chunk == 0:
            return None
        
        # 청킹 기반 시간 추정
        avg_chunk_time = 0.05  # 평균 청크당 50ms
        remaining_chunks = total_chunks - current_chunk
        chunk_time_estimate = remaining_chunks * avg_chunk_time
        
        # 에이전트 작업 시간 추정
        working_agents = [agent for agent in agent_progress 
                         if agent.status == TaskState.WORKING]
        
        agent_time_estimate = 0
        if working_agents:
            # 가장 느린 에이전트 기준으로 추정
            max_remaining_work = max(
                (100 - agent.progress_percentage) / 100 
                for agent in working_agents
            )
            
            # 에이전트당 평균 30초 작업 시간 가정
            agent_time_estimate = max_remaining_work * 30
        
        total_estimate = max(chunk_time_estimate, agent_time_estimate)
        
        return int(total_estimate) if total_estimate > 1 else None
    
    async def stream_agent_collaboration(self, 
                                       agents: List[AgentProgressInfo],
                                       task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        실시간 에이전트 협업 시각화:
        - 개별 에이전트 진행률 (0-100%)
        - 에이전트 아바타와 상태 표시기
        - 현재 작업 설명
        - 완료 체크마크와 실행 시간
        - 에이전트 간 데이터 흐름 시각화
        """
        try:
            start_time = datetime.now()
            
            while any(agent.status == TaskState.WORKING for agent in agents):
                
                # 현재 상태 스냅샷 생성
                collaboration_snapshot = {
                    'task_id': task_id,
                    'timestamp': datetime.now().isoformat(),
                    'agents': [],
                    'overall_progress': 0,
                    'active_agents_count': 0,
                    'completed_agents_count': 0,
                    'elapsed_time': (datetime.now() - start_time).total_seconds()
                }
                
                total_progress = 0
                active_count = 0
                completed_count = 0
                
                for agent in agents:
                    agent_info = {
                        'port': agent.port,
                        'name': agent.name,
                        'status': agent.status.value,
                        'progress_percentage': agent.progress_percentage,
                        'current_task': agent.current_task,
                        'execution_time': agent.execution_time,
                        'artifacts_generated': agent.artifacts_generated,
                        'status_icon': self._get_status_icon(agent.status),
                        'status_color': self._get_status_color(agent.status),
                        'avatar_icon': agent.avatar_icon or self._get_default_avatar(agent.port)
                    }
                    
                    collaboration_snapshot['agents'].append(agent_info)
                    
                    total_progress += agent.progress_percentage
                    
                    if agent.status == TaskState.WORKING:
                        active_count += 1
                    elif agent.status == TaskState.COMPLETED:
                        completed_count += 1
                
                # 전체 진행률 계산
                collaboration_snapshot['overall_progress'] = total_progress / len(agents) if agents else 0
                collaboration_snapshot['active_agents_count'] = active_count
                collaboration_snapshot['completed_agents_count'] = completed_count
                
                yield collaboration_snapshot
                
                # 0.5초 간격으로 업데이트
                await asyncio.sleep(self.status_update_interval)
            
            # 최종 완료 상태 전송
            final_snapshot = collaboration_snapshot.copy()
            final_snapshot['status'] = 'completed'
            final_snapshot['overall_progress'] = 100
            
            yield final_snapshot
            
        except Exception as e:
            logger.error(f"Agent collaboration streaming error: {str(e)}")
            
            # 오류 상태 전송
            yield {
                'task_id': task_id,
                'status': 'error',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_status_icon(self, status: TaskState) -> str:
        """상태별 아이콘 반환"""
        status_icons = {
            TaskState.PENDING: "⏳",
            TaskState.WORKING: "⚡",
            TaskState.COMPLETED: "✅",
            TaskState.FAILED: "❌"
        }
        return status_icons.get(status, "❓")
    
    def _get_status_color(self, status: TaskState) -> str:
        """상태별 색상 반환"""
        status_colors = {
            TaskState.PENDING: "#ffc107",  # 노란색
            TaskState.WORKING: "#007bff",  # 파란색
            TaskState.COMPLETED: "#28a745",  # 초록색
            TaskState.FAILED: "#dc3545"  # 빨간색
        }
        return status_colors.get(status, "#6c757d")
    
    def _get_default_avatar(self, port: int) -> str:
        """포트별 기본 아바타 아이콘"""
        avatar_mapping = {
            8306: "🧹",  # Data Cleaning
            8307: "📁",  # Data Loader
            8308: "📊",  # Data Visualization
            8309: "🔧",  # Data Wrangling
            8310: "⚙️",  # Feature Engineering
            8311: "🗄️",  # SQL Database
            8312: "🔍",  # EDA Tools
            8313: "🤖",  # H2O ML
            8314: "📈",  # MLflow Tools
            8315: "🐼"   # Pandas Analyst
        }
        return avatar_mapping.get(port, "🔄")
    
    async def handle_stream_interruption(self, 
                                       response_id: str, 
                                       fallback_content: str = None) -> StreamingResponse:
        """
        스트림 중단 시 우아한 성능 저하:
        - 명확한 오류 메시지
        - 기본 응답으로 폴백
        - 중단 전까지의 진행 상황 보존
        """
        try:
            if response_id in self.active_streams:
                stream_info = self.active_streams[response_id]
                stream_info['status'] = 'interrupted'
                stream_info['interruption_time'] = datetime.now()
                
                chunks_sent = stream_info.get('chunks_sent', 0)
                
                fallback_message = (
                    f"⚠️ 스트리밍이 중단되었습니다 (청크 {chunks_sent}개 전송 완료).\n\n"
                    f"{fallback_content or '전체 응답을 표시합니다.'}"
                )
                
                return StreamingResponse(
                    content=fallback_message,
                    is_complete=True,
                    chunk_index=chunks_sent,
                    total_chunks=chunks_sent + 1
                )
            else:
                return StreamingResponse(
                    content=fallback_content or "응답을 생성할 수 없습니다.",
                    is_complete=True,
                    chunk_index=0,
                    total_chunks=1
                )
                
        except Exception as e:
            logger.error(f"Error handling stream interruption: {str(e)}")
            
            return StreamingResponse(
                content="시스템 오류가 발생했습니다.",
                is_complete=True,
                chunk_index=0,
                total_chunks=1
            )
    
    def get_active_streams(self) -> Dict[str, Dict]:
        """활성 스트림 목록 반환"""
        return self.active_streams.copy()
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """스트리밍 통계 반환"""
        total_streams = len(self.active_streams)
        completed_streams = sum(1 for stream in self.active_streams.values() 
                               if stream['status'] == 'completed')
        
        return {
            'total_streams': total_streams,
            'completed_streams': completed_streams,
            'active_streams': total_streams - completed_streams,
            'average_chunks_per_stream': (
                sum(stream.get('chunks_sent', 0) for stream in self.active_streams.values()) 
                / total_streams if total_streams > 0 else 0
            )
        }
    
    def cleanup_finished_streams(self, max_age_hours: int = 1):
        """완료된 스트림 정리"""
        current_time = datetime.now()
        to_remove = []
        
        for stream_id, stream_info in self.active_streams.items():
            if stream_info['status'] in ['completed', 'interrupted', 'finished']:
                end_time = stream_info.get('end_time') or stream_info.get('interruption_time')
                if end_time and (current_time - end_time).total_seconds() > max_age_hours * 3600:
                    to_remove.append(stream_id)
        
        for stream_id in to_remove:
            del self.active_streams[stream_id]
        
        logger.info(f"Cleaned up {len(to_remove)} finished streams")