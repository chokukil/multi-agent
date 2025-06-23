import os
import json
import logging
from typing import Dict, Any, List, Optional
import threading

class AgentRegistry:
    """
    A singleton class to discover, register, and manage A2A agents from configuration files.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir: str = "mcp-configs"):
        # The initializer should only run once
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

            self.config_dir = config_dir
            self.agents: Dict[str, Dict[str, Any]] = {}
            self.load_agents_from_config()
            self._initialized = True
            logging.info(f"AgentRegistry initialized. Found {len(self.agents)} agents in '{self.config_dir}'.")

    def load_agents_from_config(self):
        """Loads agent configurations from JSON files in the specified directory."""
        if not os.path.isdir(self.config_dir):
            logging.warning(f"Agent config directory not found: {self.config_dir}")
            return

        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.config_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        agent_name = config.get("agent_name")
                        if agent_name:
                            self.agents[agent_name] = config
                        else:
                            logging.warning(f"Skipping config file {filename}: 'agent_name' not found.")
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON from {filename}")
                except Exception as e:
                    logging.error(f"Error loading agent from {filename}: {e}")

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Returns the configuration info for a specific agent."""
        return self.agents.get(agent_name)

    def get_all_skills_summary(self) -> str:
        """
        Generates a structured string summary of all skills from all registered agents.
        This summary is intended for use in LLM prompts.
        """
        summary = "Available Agents and Skills:\n"
        summary += "---------------------------\n\n"
        
        if not self.agents:
            return "No agents are available."
            
        for agent_name, config in self.agents.items():
            summary += f"### Agent: {agent_name}\n"
            description = config.get('description', 'No description provided.')
            summary += f"Description: {description}\n"
            
            skills = config.get('skills', [])
            if not skills:
                summary += "- No skills listed for this agent.\n"
            else:
                summary += "Skills:\n"
                for i, skill in enumerate(skills, 1):
                    skill_name = skill.get('skill_name', f'Unnamed Skill {i}')
                    skill_desc = skill.get('description', 'No description.')
                    summary += f"- **{skill_name}**: {skill_desc}\n"
            
            summary += "\n"
            
        return summary.strip()

# Create a singleton instance for easy import
agent_registry = AgentRegistry() 