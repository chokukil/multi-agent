"""
🔍 Langfuse OpenTelemetry Integration
Enhanced distributed tracing for A2A multi-agent systems using OpenTelemetry backend

Features:
- OpenTelemetry trace propagation across A2A agents
- Automatic context propagation in streaming scenarios
- Enhanced multi-agent correlation
- Performance metrics collection
- Custom span attributes for A2A protocol
"""

import os
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import time

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Install: pip install opentelemetry-api opentelemetry-sdk")

# Langfuse imports
try:
    from langfuse import Langfuse
    from langfuse.opentelemetry import LangfuseSpanExporter
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available for OpenTelemetry integration")

logger = logging.getLogger(__name__)


class LangfuseOTelIntegration:
    """OpenTelemetry integration with Langfuse for distributed A2A tracing"""
    
    def __init__(self, service_name: str = "cherry-ai-a2a-system"):
        """
        Initialize OpenTelemetry with Langfuse backend
        
        Args:
            service_name: Name of the A2A service
        """
        self.service_name = service_name
        self.tracer = None
        self.enabled = OTEL_AVAILABLE and LANGFUSE_AVAILABLE
        
        if self.enabled:
            self._setup_otel_with_langfuse()
        else:
            logger.warning("OpenTelemetry + Langfuse integration disabled")
    
    def _setup_otel_with_langfuse(self):
        """Setup OpenTelemetry with Langfuse as backend"""
        try:
            # Create TracerProvider
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            
            # Setup Langfuse exporter
            langfuse_exporter = LangfuseSpanExporter(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(langfuse_exporter)
            provider.add_span_processor(span_processor)
            
            # Set up propagators for distributed tracing
            set_global_textmap(TraceContextTextMapPropagator())
            
            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)
            
            logger.info(f"✅ OpenTelemetry + Langfuse integration initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"❌ OpenTelemetry setup failed: {e}")
            self.enabled = False
    
    @contextmanager
    def trace_a2a_agent_execution(self, agent_name: str, task_description: str,
                                 context_data: Dict[str, Any] = None):
        """
        OpenTelemetry context manager for A2A agent execution
        
        Args:
            agent_name: Name of the A2A agent
            task_description: Description of the task
            context_data: Additional context data
        """
        if not self.enabled or not self.tracer:
            yield None
            return
        
        span_name = f"A2A-Agent: {agent_name}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Set standard A2A attributes
            span.set_attribute("a2a.agent.name", agent_name)
            span.set_attribute("a2a.task.description", task_description)
            span.set_attribute("a2a.protocol.version", "v0.2.9")
            span.set_attribute("service.name", self.service_name)
            
            # Add EMP_NO for user identification
            if os.getenv("EMP_NO"):
                span.set_attribute("user.id", os.getenv("EMP_NO"))
            
            # Add context data
            if context_data:
                for key, value in context_data.items():
                    span.set_attribute(f"a2a.context.{key}", str(value))
            
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
    
    def trace_streaming_update(self, span, chunk: str, chunk_index: int, final: bool = False):
        """
        Add streaming-specific attributes to current span
        
        Args:
            span: Current OpenTelemetry span
            chunk: Streaming chunk content
            chunk_index: Index of the chunk
            final: Whether this is the final chunk
        """
        if not self.enabled or not span:
            return
        
        try:
            span.add_event(
                name="streaming.chunk",
                attributes={
                    "chunk.index": chunk_index,
                    "chunk.size": len(chunk),
                    "chunk.final": final,
                    "chunk.timestamp": time.time()
                }
            )
            
            # Update span attributes for streaming
            span.set_attribute("streaming.enabled", True)
            span.set_attribute("streaming.chunk_count", chunk_index + 1)
            
            if final:
                span.set_attribute("streaming.completed", True)
                
        except Exception as e:
            logger.warning(f"Failed to add streaming attributes: {e}")
    
    def trace_llm_interaction(self, span, prompt: str, response: str, 
                             model: str, execution_time: float):
        """
        Add LLM interaction details to span
        
        Args:
            span: Current OpenTelemetry span
            prompt: LLM prompt
            response: LLM response
            model: Model name
            execution_time: Execution time in seconds
        """
        if not self.enabled or not span:
            return
        
        try:
            span.add_event(
                name="llm.interaction",
                attributes={
                    "llm.model": model,
                    "llm.prompt.length": len(prompt),
                    "llm.response.length": len(response),
                    "llm.execution_time": execution_time,
                    "llm.timestamp": time.time()
                }
            )
            
            # Add as span attributes too
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.execution_time", execution_time)
            
        except Exception as e:
            logger.warning(f"Failed to add LLM attributes: {e}")
    
    def create_child_span(self, parent_span, operation_name: str, attributes: Dict[str, Any] = None):
        """
        Create a child span for nested operations
        
        Args:
            parent_span: Parent span
            operation_name: Name of the operation
            attributes: Additional attributes
            
        Returns:
            Child span context manager
        """
        if not self.enabled or not self.tracer:
            return None
        
        @contextmanager
        def child_span_context():
            with self.tracer.start_span(operation_name, parent=parent_span) as child_span:
                if attributes:
                    for key, value in attributes.items():
                        child_span.set_attribute(key, str(value))
                yield child_span
        
        return child_span_context()
    
    def inject_trace_context(self, headers: Dict[str, str] = None) -> Dict[str, str]:
        """
        Inject trace context into HTTP headers for A2A communication
        
        Args:
            headers: Existing headers dictionary
            
        Returns:
            Headers with trace context
        """
        if not self.enabled:
            return headers or {}
        
        headers = headers or {}
        
        try:
            # Inject trace context
            from opentelemetry.propagate import inject
            inject(headers)
            
        except Exception as e:
            logger.warning(f"Failed to inject trace context: {e}")
        
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]):
        """
        Extract trace context from HTTP headers
        
        Args:
            headers: HTTP headers containing trace context
            
        Returns:
            Extracted context
        """
        if not self.enabled:
            return None
        
        try:
            from opentelemetry.propagate import extract
            return extract(headers)
            
        except Exception as e:
            logger.warning(f"Failed to extract trace context: {e}")
            return None


# Global instance
otel_integration = LangfuseOTelIntegration()


def get_otel_integration() -> LangfuseOTelIntegration:
    """Get the global OpenTelemetry integration instance"""
    return otel_integration


# Decorator for easy A2A agent tracing
def trace_a2a_agent(agent_name: str = None, task_description: str = None):
    """
    Decorator for A2A agent methods with OpenTelemetry tracing
    
    Args:
        agent_name: Name of the agent (defaults to class name)
        task_description: Description of the task
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract agent name from class if not provided
            actual_agent_name = agent_name
            if not actual_agent_name and hasattr(args[0], '__class__'):
                actual_agent_name = args[0].__class__.__name__
            
            # Extract task description from kwargs if not provided
            actual_task_description = task_description or kwargs.get('task_description', 'A2A Agent Execution')
            
            integration = get_otel_integration()
            
            if not integration.enabled:
                return func(*args, **kwargs)
            
            with integration.trace_a2a_agent_execution(actual_agent_name, actual_task_description) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    # Add result metadata if available
                    if span and isinstance(result, dict):
                        if 'status' in result:
                            span.set_attribute("a2a.result.status", result['status'])
                        if 'artifacts' in result:
                            span.set_attribute("a2a.result.artifacts_count", len(result['artifacts']))
                    
                    return result
                    
                except Exception as e:
                    if span:
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    return decorator 