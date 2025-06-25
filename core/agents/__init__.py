"""Core Agents"""
from typing import Dict, Optional
from a2a.types import AgentCard

class AgentRegistry:
    """A central registry to hold and manage agent instances."""
    _agents: Dict[str, AgentCard] = {}

    def register_agent(self, agent: AgentCard):
        """Registers a new agent instance."""
        if not agent.name:
            raise ValueError("Agent must have a name to be registered.")
        print(f"Registering agent: {agent.name}")
        self._agents[agent.name] = agent

    def get_agent(self, agent_name: str) -> Optional[AgentCard]:
        """Retrieves an agent by its name."""
        print(f"Looking up agent: {agent_name}")
        return self._agents.get(agent_name)

    def list_agents(self) -> Dict[str, AgentCard]:
        """Returns all registered agents."""
        return self._agents

# A global instance of the registry for easy access across the application
agent_registry = AgentRegistry()

__all__ = [
    "AgentRegistry",
    "agent_registry",
] 