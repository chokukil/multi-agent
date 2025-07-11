"""
🔍 Enhanced A2A Communicator with Distributed Tracing
Advanced A2A communication with comprehensive Langfuse + OpenTelemetry integration

Features:
- Distributed tracing across A2A agents
- Streaming support with trace context propagation
- Session-based correlation
- Performance monitoring
- Error tracking and recovery
- Multi-agent workflow visibility
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from contextlib import asynccontextmanager
import httpx

# A2A imports
from a2a.types import Message, TextPart, TaskState
from a2a.utils import new_agent_text_message

# Project imports
try:
    from core.langfuse_session_tracer import get_session_tracer
    from core.langfuse_otel_integration import get_otel_integration
    ENHANCED_TRACING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACING_AVAILABLE = False
    logging.warning("Enhanced tracing not available")

logger = logging.getLogger(__name__)


class EnhancedA2ACommunicator:
    """Enhanced A2A communicator with distributed tracing and streaming support"""
    
    def __init__(self, timeout: int = 300, max_retries: int = 3):
        """
        Initialize enhanced A2A communicator
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session_tracer = None
        self.otel_integration = None
        
        # Initialize tracing if available
        if ENHANCED_TRACING_AVAILABLE:
            self.session_tracer = get_session_tracer()
            self.otel_integration = get_otel_integration()
            logger.info("✅ Enhanced A2A communicator initialized with distributed tracing")
        else:
            logger.warning("⚠️ Enhanced tracing disabled - basic A2A communication only")
    
    async def send_message_with_streaming(
        self,
        agent_url: str,
        instruction: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send message to A2A agent with streaming support and distributed tracing
        
        Args:
            agent_url: URL of the target A2A agent
            instruction: Instruction to send
            stream_callback: Callback for streaming updates
            context_data: Additional context data
            
        Returns:
            Response from the agent
        """
        agent_name = self._extract_agent_name(agent_url)
        
        # Create distributed trace context
        trace_context = None
        session_context = None
        
        if ENHANCED_TRACING_AVAILABLE:
            # OpenTelemetry distributed tracing
            if self.otel_integration and self.otel_integration.enabled:
                trace_context = self.otel_integration.trace_a2a_agent_execution(
                    agent_name, 
                    instruction[:100] + "..." if len(instruction) > 100 else instruction,
                    context_data
                )
            
            # Session-based tracing
            if self.session_tracer:
                session_context = self.session_tracer.trace_agent_execution(
                    agent_name,
                    instruction[:100] + "..." if len(instruction) > 100 else instruction,
                    context_data
                )
        
        # Execute with both tracing systems
        async with self._multi_trace_context(trace_context, session_context) as (otel_span, session_span):
            return await self._execute_with_tracing(
                agent_url,
                instruction,
                stream_callback,
                context_data,
                otel_span,
                session_span
            )
    
    @asynccontextmanager
    async def _multi_trace_context(self, otel_context, session_context):
        """Context manager for multiple tracing systems"""
        otel_span = None
        session_span = None
        
        try:
            # Start OpenTelemetry span
            if otel_context:
                async with otel_context as span:
                    otel_span = span
                    
                    # Start session span
                    if session_context:
                        async with session_context as s_span:
                            session_span = s_span
                            yield otel_span, session_span
                    else:
                        yield otel_span, None
            elif session_context:
                async with session_context as s_span:
                    session_span = s_span
                    yield None, session_span
            else:
                yield None, None
                
        except Exception as e:
            if otel_span:
                otel_span.set_attribute("error", True)
                otel_span.set_attribute("error.message", str(e))
            if session_span:
                # Session span error handling would be done by session tracer
                pass
            raise
    
    async def _execute_with_tracing(
        self,
        agent_url: str,
        instruction: str,
        stream_callback: Optional[Callable[[str], None]],
        context_data: Optional[Dict[str, Any]],
        otel_span,
        session_span
    ) -> Dict[str, Any]:
        """Execute A2A request with comprehensive tracing"""
        
        start_time = time.time()
        agent_name = self._extract_agent_name(agent_url)
        
        # Add initial trace attributes
        if otel_span:
            otel_span.set_attribute("a2a.agent.url", agent_url)
            otel_span.set_attribute("a2a.instruction.length", len(instruction))
            otel_span.set_attribute("a2a.streaming.enabled", stream_callback is not None)
        
        try:
            # Create A2A message
            message = Message(
                messageId=str(uuid.uuid4()),
                role="user",
                parts=[TextPart(text=instruction)]
            )
            
            # Prepare headers with trace context
            headers = {"Content-Type": "application/json"}
            if self.otel_integration and self.otel_integration.enabled:
                headers = self.otel_integration.inject_trace_context(headers)
            
            # Execute request with retry logic
            response = await self._execute_with_retry(
                agent_url, message, headers, stream_callback, otel_span, session_span
            )
            
            execution_time = time.time() - start_time
            
            # Add success attributes
            if otel_span:
                otel_span.set_attribute("a2a.execution.time", execution_time)
                otel_span.set_attribute("a2a.execution.status", "success")
                otel_span.set_attribute("a2a.response.size", len(str(response)))
            
            # Log to session tracer
            if self.session_tracer and session_span:
                self.session_tracer.trace_agent_internal_logic(
                    agent_name=agent_name,
                    operation="a2a_communication",
                    input_data={"instruction": instruction[:200]},
                    output_data=response,
                    operation_metadata={
                        "execution_time": execution_time,
                        "agent_url": agent_url,
                        "streaming_enabled": stream_callback is not None
                    }
                )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Add error attributes
            if otel_span:
                otel_span.set_attribute("a2a.execution.time", execution_time)
                otel_span.set_attribute("a2a.execution.status", "error")
                otel_span.set_attribute("error", True)
                otel_span.set_attribute("error.message", str(e))
                otel_span.set_attribute("error.type", type(e).__name__)
            
            logger.error(f"❌ A2A communication failed for {agent_name}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "agent_name": agent_name,
                "execution_time": execution_time
            }
    
    async def _execute_with_retry(
        self,
        agent_url: str,
        message: Message,
        headers: Dict[str, str],
        stream_callback: Optional[Callable[[str], None]],
        otel_span,
        session_span
    ) -> Dict[str, Any]:
        """Execute A2A request with retry logic"""
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if otel_span:
                    otel_span.add_event(
                        f"a2a.request.attempt",
                        attributes={"attempt": attempt + 1, "max_attempts": self.max_retries}
                    )
                
                if stream_callback:
                    return await self._execute_streaming_request(
                        agent_url, message, headers, stream_callback, otel_span, session_span
                    )
                else:
                    return await self._execute_standard_request(
                        agent_url, message, headers, otel_span, session_span
                    )
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    
                    if otel_span:
                        otel_span.add_event(
                            "a2a.request.retry",
                            attributes={
                                "attempt": attempt + 1,
                                "error": str(e),
                                "wait_time": wait_time
                            }
                        )
                    
                    logger.warning(f"⚠️ A2A request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ A2A request failed after {self.max_retries} attempts: {e}")
        
        raise last_exception
    
    async def _execute_streaming_request(
        self,
        agent_url: str,
        message: Message,
        headers: Dict[str, str],
        stream_callback: Callable[[str], None],
        otel_span,
        session_span
    ) -> Dict[str, Any]:
        """Execute streaming A2A request"""
        
        # Prepare streaming payload
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "messageId": message.messageId,
                    "role": message.role,
                    "parts": [{"type": "text", "text": part.text} for part in message.parts]
                }
            },
            "id": str(uuid.uuid4())
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if otel_span:
                otel_span.add_event("a2a.streaming.request.start")
            
            # Make streaming request
            async with client.stream(
                "POST",
                f"{agent_url}/a2a/stream",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                chunks = []
                chunk_index = 0
                
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        chunks.append(chunk)
                        
                        # Process SSE chunk
                        if chunk.startswith("data: "):
                            try:
                                data = json.loads(chunk[6:])  # Remove "data: " prefix
                                
                                if "result" in data and "message" in data["result"]:
                                    content = self._extract_content_from_message(data["result"]["message"])
                                    
                                    if content and stream_callback:
                                        await stream_callback(content)
                                    
                                    # Add streaming trace attributes
                                    if otel_span:
                                        self.otel_integration.trace_streaming_update(
                                            otel_span, content, chunk_index
                                        )
                                    
                                    chunk_index += 1
                                    
                            except json.JSONDecodeError:
                                continue
                
                # Mark streaming as complete
                if otel_span:
                    otel_span.add_event("a2a.streaming.request.complete")
                    otel_span.set_attribute("a2a.streaming.chunk_count", chunk_index)
                
                return {
                    "status": "success",
                    "chunks": chunks,
                    "chunk_count": chunk_index,
                    "streaming": True
                }
    
    async def _execute_standard_request(
        self,
        agent_url: str,
        message: Message,
        headers: Dict[str, str],
        otel_span,
        session_span
    ) -> Dict[str, Any]:
        """Execute standard (non-streaming) A2A request"""
        
        # Prepare payload
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "messageId": message.messageId,
                    "role": message.role,
                    "parts": [{"type": "text", "text": part.text} for part in message.parts]
                }
            },
            "id": str(uuid.uuid4())
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if otel_span:
                otel_span.add_event("a2a.standard.request.start")
            
            response = await client.post(
                f"{agent_url}/a2a/tasks/send",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            
            if otel_span:
                otel_span.add_event("a2a.standard.request.complete")
                otel_span.set_attribute("a2a.response.status_code", response.status_code)
            
            return {
                "status": "success",
                "result": result,
                "streaming": False
            }
    
    def _extract_agent_name(self, agent_url: str) -> str:
        """Extract agent name from URL"""
        try:
            # Extract from URL pattern like http://localhost:8306/...
            port = agent_url.split(":")[2].split("/")[0]
            return f"Agent-{port}"
        except:
            return "UnknownAgent"
    
    def _extract_content_from_message(self, message: Dict[str, Any]) -> str:
        """Extract text content from A2A message"""
        try:
            if "parts" in message:
                for part in message["parts"]:
                    if part.get("type") == "text":
                        return part.get("text", "")
            return ""
        except:
            return ""
    
    async def health_check(self, agent_url: str) -> Dict[str, Any]:
        """Check health of A2A agent with tracing"""
        agent_name = self._extract_agent_name(agent_url)
        
        # Simple health check with minimal tracing
        if self.otel_integration and self.otel_integration.enabled:
            with self.otel_integration.trace_a2a_agent_execution(
                agent_name, "health_check"
            ) as span:
                return await self._execute_health_check(agent_url, span)
        else:
            return await self._execute_health_check(agent_url, None)
    
    async def _execute_health_check(self, agent_url: str, otel_span) -> Dict[str, Any]:
        """Execute health check request"""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{agent_url}/.well-known/agent.json")
                
                execution_time = time.time() - start_time
                
                if otel_span:
                    otel_span.set_attribute("a2a.health.status", "healthy")
                    otel_span.set_attribute("a2a.health.response_time", execution_time)
                
                return {
                    "status": "healthy",
                    "response_time": execution_time,
                    "agent_info": response.json() if response.status_code == 200 else None
                }
                
        except Exception as e:
            if otel_span:
                otel_span.set_attribute("a2a.health.status", "unhealthy")
                otel_span.set_attribute("error", True)
                otel_span.set_attribute("error.message", str(e))
            
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def discover_agents(self, agent_urls: List[str]) -> Dict[str, Any]:
        """Discover multiple agents with distributed tracing"""
        results = {}
        
        # Use OpenTelemetry for agent discovery coordination
        if self.otel_integration and self.otel_integration.enabled:
            with self.otel_integration.trace_a2a_agent_execution(
                "AgentDiscovery", "discover_multiple_agents"
            ) as span:
                if span:
                    span.set_attribute("a2a.discovery.agent_count", len(agent_urls))
                
                # Parallel health checks
                tasks = [self.health_check(url) for url in agent_urls]
                health_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, result in zip(agent_urls, health_results):
                    agent_name = self._extract_agent_name(url)
                    results[agent_name] = result if not isinstance(result, Exception) else {
                        "status": "error",
                        "error": str(result)
                    }
                
                if span:
                    healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
                    span.set_attribute("a2a.discovery.healthy_count", healthy_count)
        
        return results


# Global instance
enhanced_a2a_communicator = EnhancedA2ACommunicator()


def get_enhanced_a2a_communicator() -> EnhancedA2ACommunicator:
    """Get the global enhanced A2A communicator instance"""
    return enhanced_a2a_communicator 