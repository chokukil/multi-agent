"""
A2A SDK Integration Components for Cherry AI Universal Engine

A2A SDK 0.2.9 integration with proven Universal Engine patterns:
- agent_client: A2A communication protocol with JSON-RPC 2.0 validation
- agent_discovery: Agent discovery system with health monitoring
- workflow_orchestrator: Sequential/parallel execution patterns
"""

# Import A2A components with graceful degradation
try:
    from .agent_client import A2AAgentClient
except ImportError:
    A2AAgentClient = None

try:
    from .workflow_orchestrator import A2AWorkflowOrchestrator
except ImportError:
    A2AWorkflowOrchestrator = None

# Export available components
__all__ = [
    name for name, obj in [
        ("A2AAgentClient", A2AAgentClient),
        ("A2AWorkflowOrchestrator", A2AWorkflowOrchestrator),
    ] if obj is not None
]