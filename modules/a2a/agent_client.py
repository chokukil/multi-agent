import os, mimetypes, requests, uuid, base64, asyncio, time, json
import httpx
from typing import Any, Dict, List, Optional, AsyncGenerator
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

def _detect_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "text/plain"

class A2AAgentClient:
    """A2A SDK 0.2.9 공식 표준 준수 클라이언트"""
    
    def __init__(self, endpoint: str, timeout: int = 60):
        self.endpoint = endpoint
        self.timeout = timeout
        self.chunk_delay = 0.01  # 0.01초 자연스러운 딜레이
        
    def send_message(self, text: str, file_paths: Optional[List[str]] = None,
                     meta: Optional[Dict[str, Any]] = None, dry_run: bool=False) -> Dict[str, Any]:
        """A2A SDK 0.2.9 표준 준수 메시지 전송"""
        
        # TextPart 생성 (A2A SDK 0.2.9 표준)
        parts = [{"kind": "text", "text": text}]
        
        # FilePart 생성 (A2A SDK 0.2.9 표준)
        for p in (file_paths or []):
            try:
                with open(p, 'rb') as f:
                    file_content = f.read()
                
                # Base64 인코딩 (A2A SDK 0.2.9 요구사항)
                base64_content = base64.b64encode(file_content).decode('utf-8')
                
                parts.append({
                    "kind": "file",  # A2A SDK 0.2.9: "type" → "kind"
                    "file": {
                        "bytes": base64_content,  # A2A SDK 0.2.9: "data" → "bytes", base64 인코딩
                        "mime_type": _detect_mime(p),  # A2A SDK 0.2.9: "mimeType" → "mime_type"
                        "name": os.path.basename(p)
                    }
                })
                
            except Exception as e:
                logger.warning(f"파일 읽기 오류 {p}: {e}")
                continue
        
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": "msg",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": parts
                },
                "meta": {"source": "cherry-ai", "dry_run": dry_run, **(meta or {})}
            }
        }
        
        logger.info(f"🔍 A2A 요청: {self.endpoint} | 메시지 ID: {message_id[:8]} | 파트: {len(parts)}")
        
        r = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def stream_message(self, text: str, file_paths: Optional[List[str]] = None,
                       meta: Optional[Dict[str, Any]] = None):
        """A2A SDK 0.2.9 표준 준수 동기 스트리밍 메시지 (기존 호환성)"""
        
        # TextPart 생성 (A2A SDK 0.2.9 표준)
        parts = [{"kind": "text", "text": text}]
        
        # FilePart 생성 (A2A SDK 0.2.9 표준)
        for p in (file_paths or []):
            try:
                with open(p, 'rb') as f:
                    file_content = f.read()
                
                # Base64 인코딩 (A2A SDK 0.2.9 요구사항)
                base64_content = base64.b64encode(file_content).decode('utf-8')
                
                parts.append({
                    "kind": "file",  # A2A SDK 0.2.9: "type" → "kind"
                    "file": {
                        "bytes": base64_content,  # A2A SDK 0.2.9: "data" → "bytes", base64 인코딩
                        "mime_type": _detect_mime(p),  # A2A SDK 0.2.9: "mimeType" → "mime_type"
                        "name": os.path.basename(p)
                    }
                })
                
            except Exception as e:
                logger.warning(f"파일 읽기 오류 {p}: {e}")
                continue
        
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": "stream",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": parts
                },
                "meta": {"source": "cherry-ai", **(meta or {})}
            }
        }
        
        logger.info(f"🔍 A2A 스트리밍: {self.endpoint} | 메시지 ID: {message_id[:8]} | 파트: {len(parts)}")
        
        with requests.post(self.endpoint, json=payload, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line: 
                    # 0.01초 자연스러운 딜레이 적용
                    time.sleep(self.chunk_delay)
                    yield line

    async def stream_message_async(self, text: str, file_paths: Optional[List[str]] = None,
                                   meta: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """A2A SDK 0.2.9 표준 준수 비동기 SSE 스트리밍 (0.01초 자연스러운 딜레이)"""
        
        # TextPart 생성 (A2A SDK 0.2.9 표준)
        parts = [{"kind": "text", "text": text}]
        
        # FilePart 생성 (A2A SDK 0.2.9 표준)
        for p in (file_paths or []):
            try:
                with open(p, 'rb') as f:
                    file_content = f.read()
                
                # Base64 인코딩 (A2A SDK 0.2.9 요구사항)
                base64_content = base64.b64encode(file_content).decode('utf-8')
                
                parts.append({
                    "kind": "file",  # A2A SDK 0.2.9: "type" → "kind"
                    "file": {
                        "bytes": base64_content,  # A2A SDK 0.2.9: "data" → "bytes", base64 인코딩
                        "mime_type": _detect_mime(p),  # A2A SDK 0.2.9: "mimeType" → "mime_type"
                        "name": os.path.basename(p)
                    }
                })
                
            except Exception as e:
                logger.warning(f"파일 읽기 오류 {p}: {e}")
                continue
        
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": "stream",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": parts
                },
                "meta": {"source": "cherry-ai", **(meta or {})}
            }
        }
        
        logger.info(f"🔍 A2A 비동기 스트리밍: {self.endpoint} | 메시지 ID: {message_id[:8]} | 파트: {len(parts)}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream('POST', self.endpoint, json=payload) as response:
                    response.raise_for_status()
                    
                    # SSE 스트리밍 처리
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            # 청크를 문자열로 디코딩
                            chunk_str = chunk.decode('utf-8', errors='ignore')
                            buffer += chunk_str
                            
                            # 줄 단위로 처리
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                
                                if line:
                                    # SSE 형식 처리
                                    if line.startswith('data: '):
                                        data_content = line[6:]  # 'data: ' 제거
                                        
                                        # JSON 파싱 시도
                                        try:
                                            json_data = json.loads(data_content)
                                            # 실제 텍스트 내용 추출
                                            if isinstance(json_data, dict):
                                                text_content = json_data.get('text', data_content)
                                            else:
                                                text_content = data_content
                                        except json.JSONDecodeError:
                                            text_content = data_content
                                        
                                        # 0.01초 자연스러운 딜레이
                                        await asyncio.sleep(self.chunk_delay)
                                        yield text_content
                                    
                                    elif line and not line.startswith(':'):
                                        # 일반 텍스트 라인
                                        await asyncio.sleep(self.chunk_delay)
                                        yield line
                            
                            # 남은 버퍼 처리
                            if buffer and not buffer.endswith('\n'):
                                await asyncio.sleep(self.chunk_delay)
                                yield buffer
                                buffer = ""
                                
        except Exception as e:
            logger.error(f"A2A 비동기 스트리밍 오류: {e}")
            yield f"스트리밍 오류: {str(e)}"

    async def send_message_async(self, text: str, file_paths: Optional[List[str]] = None,
                                 meta: Optional[Dict[str, Any]] = None, dry_run: bool=False) -> Dict[str, Any]:
        """A2A SDK 0.2.9 표준 준수 비동기 메시지 전송"""
        
        # TextPart 생성 (A2A SDK 0.2.9 표준)
        parts = [{"kind": "text", "text": text}]
        
        # FilePart 생성 (A2A SDK 0.2.9 표준)
        for p in (file_paths or []):
            try:
                with open(p, 'rb') as f:
                    file_content = f.read()
                
                # Base64 인코딩 (A2A SDK 0.2.9 요구사항)
                base64_content = base64.b64encode(file_content).decode('utf-8')
                
                parts.append({
                    "kind": "file",  # A2A SDK 0.2.9: "type" → "kind"
                    "file": {
                        "bytes": base64_content,  # A2A SDK 0.2.9: "data" → "bytes", base64 인코딩
                        "mime_type": _detect_mime(p),  # A2A SDK 0.2.9: "mimeType" → "mime_type"
                        "name": os.path.basename(p)
                    }
                })
                
            except Exception as e:
                logger.warning(f"파일 읽기 오류 {p}: {e}")
                continue
        
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": "msg",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": parts
                },
                "meta": {"source": "cherry-ai", "dry_run": dry_run, **(meta or {})}
            }
        }
        
        logger.info(f"🔍 A2A 비동기 요청: {self.endpoint} | 메시지 ID: {message_id[:8]} | 파트: {len(parts)}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            return response.json()

    def get_agent_card(self) -> Dict[str, Any]:
        """A2A 에이전트 카드 조회 (/.well-known/agent.json)"""
        try:
            card_url = f"{self.endpoint}/.well-known/agent.json"
            response = requests.get(card_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"에이전트 카드 조회 실패 {self.endpoint}: {e}")
            return {}

    async def get_agent_card_async(self) -> Dict[str, Any]:
        """A2A 에이전트 카드 비동기 조회"""
        try:
            card_url = f"{self.endpoint}/.well-known/agent.json"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(card_url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"에이전트 카드 비동기 조회 실패 {self.endpoint}: {e}")
            return {}

    def health_check(self) -> bool:
        """A2A 에이전트 헬스 체크"""
        try:
            card = self.get_agent_card()
            return bool(card.get('name'))
        except:
            return False

    async def health_check_async(self) -> bool:
        """A2A 에이전트 비동기 헬스 체크"""
        try:
            card = await self.get_agent_card_async()
            return bool(card.get('name'))
        except:
            return False