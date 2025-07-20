"""
A2A Integration Package - Universal Engine과 A2A Agent 통합

Zero Hardcoding A2A Agent 통합 시스템
"""

from .a2a_agent_discovery import A2AAgentDiscoverySystem
from .llm_based_agent_selector import LLMBasedAgentSelector
from .a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from .a2a_communication_protocol import A2ACommunicationProtocol
from .a2a_result_integrator import A2AResultIntegrator

__all__ = [
    'A2AAgentDiscoverySystem',
    'LLMBasedAgentSelector', 
    'A2AWorkflowOrchestrator',
    'A2ACommunicationProtocol',
    'A2AResultIntegrator'
]