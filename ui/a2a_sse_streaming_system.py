"""
A2A SDK 0.2.9 Standard SSE Streaming System
Based on official A2A protocol specifications and SSE streaming implementation
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# A2A SDK 0.2.9 imports with proper mock fallback
try:
    from a2a.types import TextPart, DataPart, FilePart, Part
    from a2a.server.tasks.task_updater import TaskUpdater
    from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
    from a2a.server.application import A2AFastAPIApplication
    from a2a.server.request_handler import DefaultRequestHandler
    from a2a.server.agent_executor import AgentExecutor
    from a2a.server.request_context import RequestContext
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logging.warning("A2A SDK not available - using mock implementation")
    
    # Complete mock implementations
    class Part:
        def __init__(self, **kwargs):
            self.root = type('Root', (), kwargs)()
    
    class TextPart(Part):
        def __init__(self, text: str):
            super().__init__(text=text, kind='text')
    
    class DataPart(Part):
        def __init__(self, data: Any):
            super().__init__(data=data, kind='data')
    
    class FilePart(Part):
        def __init__(self, file_data: Any):
            super().__init__(file=file_data, kind='file')
    
    class TaskUpdater:
        async def update_status(self, state: str, message: str = ""):
            pass
        
        async def add_artifact(self, parts: List[Part], name: str = None, metadata: Dict = None):
            pass
    
    class RequestContext:
        def __init__(self):
            self.message = None
            self.task_id = None
            self.session_id = None
    
    class AgentExecutor:
        async def execute(self, context: RequestContext, task_updater: TaskUpdater):
            pass
        
        async def cancel(self, context: RequestContext):
            pass
    
    class InMemoryTaskStore:
        def __init__(self):
            self.tasks = {}
    
    class DefaultRequestHandler:
        pass
    
    class A2AFastAPIApplication:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(Enum):
    """A2A Task States"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SSEEvent:
    """Server-Sent Event structure"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_sse_format(self) -> str:
        """Convert to SSE format"""
        return f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass
class A2AMessage:
    """A2A Message structure"""
    role: str
    parts: List[Dict[str, Any]]
    message_id: str
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class A2ATaskStatus:
    """A2A Task Status"""
    state: TaskState
    message: Optional[A2AMessage] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "state": self.state.value,
            "timestamp": self.timestamp
        }
        if self.message:
            result["message"] = asdict(self.message)
        return result


@dataclass
class A2AArtifact:
    """A2A Artifact structure"""
    artifact_id: str
    name: Optional[str] = None
    parts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    index: int = 0
    append: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "artifactId": self.artifact_id,
            "name": self.name,
            "parts": self.parts,
            "metadata": self.metadata,
            "index": self.index,
            "append": self.append
        }


class A2ASSEStreamingExecutor(AgentExecutor):
    """A2A SDK 0.2.9 compliant SSE streaming executor"""
    
    def __init__(self):
        self.is_streaming = False
        self.stream_queue = asyncio.Queue()
        self.task_store = InMemoryTaskStore()
        
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """Execute task with SSE streaming"""
        try:
            # Parse user input
            user_input = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_input += part.root.text
            
            # Start streaming
            self.is_streaming = True
            
            # Initial status update
            await task_updater.update_status(
                TaskState.WORKING.value,
                "Processing your request..."
            )
            
            # Stream processing steps
            async for event in self._stream_processing(user_input, context):
                # Add event to queue for SSE streaming
                await self.stream_queue.put(event)
                
                # Update task status through TaskUpdater
                if event.event_type == "status_update":
                    await task_updater.update_status(
                        event.data.get("state", TaskState.WORKING.value),
                        event.data.get("message", "")
                    )
                elif event.event_type == "artifact_update":
                    # Create artifact
                    artifact_data = event.data.get("artifact", {})
                    parts = [TextPart(text=str(artifact_data.get("content", "")))]
                    await task_updater.add_artifact(
                        parts=parts,
                        name=artifact_data.get("name", "result"),
                        metadata=artifact_data.get("metadata", {})
                    )
            
            # Final completion
            await task_updater.update_status(
                TaskState.COMPLETED.value,
                "Task completed successfully!"
            )
            
        except Exception as e:
            await task_updater.update_status(
                TaskState.FAILED.value,
                f"Task failed: {str(e)}"
            )
            raise
        finally:
            self.is_streaming = False
    
    async def _stream_processing(self, user_input: str, context: RequestContext) -> AsyncIterator[SSEEvent]:
        """Stream the processing steps"""
        
        # Step 1: Analysis
        yield SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="status_update",
            data={
                "state": TaskState.WORKING.value,
                "message": "Analyzing your request...",
                "progress": 0.2,
                "step": "analysis"
            }
        )
        
        await asyncio.sleep(1)  # Simulate processing time
        
        # Step 2: Processing
        yield SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="status_update",
            data={
                "state": TaskState.WORKING.value,
                "message": "Processing request...",
                "progress": 0.5,
                "step": "processing"
            }
        )
        
        await asyncio.sleep(1)
        
        # Step 3: Generating response
        yield SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="status_update",
            data={
                "state": TaskState.WORKING.value,
                "message": "Generating response...",
                "progress": 0.8,
                "step": "generating"
            }
        )
        
        await asyncio.sleep(1)
        
        # Step 4: Create artifact
        artifact = A2AArtifact(
            artifact_id=str(uuid.uuid4()),
            name="response",
            parts=[{
                "type": "text",
                "text": f"Processed: {user_input}"
            }],
            metadata={"processing_time": 3.0}
        )
        
        yield SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="artifact_update",
            data={
                "artifact": artifact.to_dict(),
                "final": True
            }
        )
        
        # Final completion
        yield SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="status_update",
            data={
                "state": TaskState.COMPLETED.value,
                "message": "Task completed successfully!",
                "progress": 1.0,
                "step": "completed"
            }
        )
    
    async def cancel(self, context: RequestContext) -> None:
        """Cancel the streaming task"""
        self.is_streaming = False
        await self.stream_queue.put(SSEEvent(
            event_id=str(uuid.uuid4()),
            event_type="status_update",
            data={
                "state": TaskState.CANCELLED.value,
                "message": "Task cancelled by user"
            }
        ))


class A2ASSEStreamingSystem:
    """A2A SDK 0.2.9 compliant SSE streaming system"""
    
    def __init__(self):
        self.executor = A2ASSEStreamingExecutor()
        self.active_streams = {}
        self.app = self._create_fastapi_app()
        
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with SSE endpoints"""
        app = FastAPI(
            title="A2A SSE Streaming System",
            description="A2A SDK 0.2.9 compliant SSE streaming",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add SSE streaming endpoint
        @app.post("/stream")
        async def stream_endpoint(request: Dict[str, Any]):
            """A2A compliant SSE streaming endpoint"""
            return await self.handle_streaming_request(request)
        
        # Add agent card endpoint
        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get A2A agent card"""
            return {
                "name": "A2A SSE Streaming Agent",
                "description": "A2A SDK 0.2.9 compliant SSE streaming agent",
                "url": "http://localhost:8000",
                "version": "1.0.0",
                "capabilities": {
                    "streaming": True,
                    "pushNotifications": False,
                    "stateTransitionHistory": True
                },
                "skills": [
                    {
                        "id": "sse_streaming",
                        "name": "SSE Streaming",
                        "description": "Real-time server-sent events streaming",
                        "examples": ["Stream my request", "Process with updates"]
                    }
                ]
            }
        
        return app
    
    async def handle_streaming_request(self, request: Dict[str, Any]) -> StreamingResponse:
        """Handle A2A streaming request"""
        
        # Extract request parameters
        jsonrpc = request.get("jsonrpc", "2.0")
        request_id = request.get("id", str(uuid.uuid4()))
        method = request.get("method", "message/stream")
        params = request.get("params", {})
        
        # Create context
        context = RequestContext()
        context.task_id = str(uuid.uuid4())
        context.session_id = params.get("sessionId", str(uuid.uuid4()))
        
        # Create message
        message_data = params.get("message", {})
        context.message = type('Message', (), {
            'parts': [
                type('Part', (), {
                    'root': type('Root', (), {
                        'text': part.get('text', ''),
                        'kind': part.get('kind', 'text')
                    })()
                })() for part in message_data.get('parts', [])
            ]
        })()
        
        # Create task updater
        task_updater = TaskUpdater()
        
        # Start streaming
        return StreamingResponse(
            self._generate_sse_stream(jsonrpc, request_id, context, task_updater),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    async def _generate_sse_stream(
        self,
        jsonrpc: str,
        request_id: str,
        context: RequestContext,
        task_updater: TaskUpdater
    ) -> AsyncIterator[str]:
        """Generate SSE stream following A2A protocol"""
        
        try:
            # Start task execution in background
            task = asyncio.create_task(
                self.executor.execute(context, task_updater)
            )
            
            # Stream events
            while not task.done():
                try:
                    # Get event from queue with timeout
                    event = await asyncio.wait_for(
                        self.executor.stream_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Format as A2A response
                    response = {
                        "jsonrpc": jsonrpc,
                        "id": request_id,
                        "result": {
                            "id": context.task_id,
                            "contextId": context.session_id,
                            "timestamp": datetime.now().isoformat(),
                            **event.data
                        }
                    }
                    
                    # Add final flag
                    if event.data.get("state") == TaskState.COMPLETED.value:
                        response["result"]["final"] = True
                    else:
                        response["result"]["final"] = False
                    
                    yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = {
                        "jsonrpc": jsonrpc,
                        "id": request_id,
                        "result": {
                            "id": context.task_id,
                            "contextId": context.session_id,
                            "heartbeat": True,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
            
            # Wait for task completion
            await task
            
            # Send final completion message
            final_response = {
                "jsonrpc": jsonrpc,
                "id": request_id,
                "result": {
                    "id": context.task_id,
                    "contextId": context.session_id,
                    "status": {
                        "state": TaskState.COMPLETED.value,
                        "timestamp": datetime.now().isoformat()
                    },
                    "final": True
                }
            }
            
            yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            # Send error response
            error_response = {
                "jsonrpc": jsonrpc,
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
            
            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application"""
        return self.app
    
    async def start_server(self, host: str = "localhost", port: int = 8000):
        """Start the SSE streaming server"""
        import uvicorn
        
        logger.info(f"Starting A2A SSE Streaming server on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()


# Singleton instance
_streaming_system = None


def get_a2a_sse_streaming_system() -> A2ASSEStreamingSystem:
    """Get the singleton A2A SSE streaming system"""
    global _streaming_system
    if _streaming_system is None:
        _streaming_system = A2ASSEStreamingSystem()
    return _streaming_system


# Demo functions
async def demo_sse_streaming():
    """Demo SSE streaming functionality"""
    system = get_a2a_sse_streaming_system()
    
    # Create a sample request
    request = {
        "jsonrpc": "2.0",
        "id": "demo-1",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Hello, please process this request with streaming updates"
                    }
                ]
            },
            "sessionId": "demo-session"
        }
    }
    
    # Process streaming request
    response = await system.handle_streaming_request(request)
    return response


if __name__ == "__main__":
    # Run demo
    async def main():
        system = get_a2a_sse_streaming_system()
        await system.start_server(host="0.0.0.0", port=8000)
    
    asyncio.run(main()) 