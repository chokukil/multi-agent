"""Core Agents"""
from .agent_registry import AgentRegistry

agent_registry = AgentRegistry()

__all__ = [
    "AgentRegistry",
    "agent_registry"
] 