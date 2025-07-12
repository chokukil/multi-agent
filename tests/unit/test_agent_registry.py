import pytest
from core.agents import AgentRegistry
import os

@pytest.fixture
def temp_agent_configs(tmp_path):
    """Creates temporary agent config files for testing."""
    config_dir = tmp_path / "mcp-configs"
    config_dir.mkdir()
    
    config1_content = """
    {
        "agent_name": "TestAgent1",
        "description": "A test agent for unit tests.",
        "base_url": "http://localhost:8001",
        "skills": [{
            "skill_name": "test_skill_1",
            "description": "Performs test action 1."
        }]
    }
    """
    (config_dir / "test_agent_1.json").write_text(config1_content)
    
    config2_content = """
    {
        "agent_name": "TestAgent2",
        "description": "Another test agent.",
        "base_url": "http://localhost:8002",
        "skills": [
            { "skill_name": "test_skill_2a", "description": "Performs test action 2a." },
            { "skill_name": "test_skill_2b", "description": "Performs test action 2b." }
        ]
    }
    """
    (config_dir / "test_agent_2.json").write_text(config2_content)
    
    return str(config_dir)

@pytest.fixture
def registry(temp_agent_configs):
    """Provides a clean, non-singleton AgentRegistry instance for each test."""
    # Reset the singleton instance before creating a new one for the test
    AgentRegistry._instance = None
    # We instantiate a new registry for each test to ensure isolation
    return AgentRegistry(config_dir=temp_agent_configs)

def test_load_agents_from_config(registry):
    """Test that agents are loaded correctly from JSON config files."""
    assert len(registry.agents) == 2
    assert "TestAgent1" in registry.agents
    assert "TestAgent2" in registry.agents
    
    agent1_info = registry.get_agent_info("TestAgent1")
    assert agent1_info["agent_name"] == "TestAgent1"
    assert agent1_info["base_url"] == "http://localhost:8001"
    assert len(agent1_info["skills"]) == 1

def test_get_agent_info(registry):
    """Test retrieving information for a specific agent."""
    agent_info = registry.get_agent_info("TestAgent2")
    assert agent_info is not None
    assert agent_info["description"] == "Another test agent."
    
    non_existent_agent = registry.get_agent_info("NonExistentAgent")
    assert non_existent_agent is None

def test_get_all_skills_summary(registry):
    """Test generating a summary of all available skills from all agents."""
    summary = registry.get_all_skills_summary()
    
    assert isinstance(summary, str)
    assert "Agent: TestAgent1" in summary
    assert "**test_skill_1**" in summary
    assert "Agent: TestAgent2" in summary
    assert "**test_skill_2a**" in summary
    assert "**test_skill_2b**" in summary
    assert "- **test_skill_1**: Performs test action 1." in summary

def test_singleton_behavior_with_custom_path(temp_agent_configs):
    """Test that the singleton pattern works even with a custom config path."""
    # The first instance created with a custom path should be stored
    r1 = AgentRegistry(config_dir=temp_agent_configs)
    
    # Subsequent calls without a path should return the first instance
    r2 = AgentRegistry()
    
    assert r1 is r2
    assert len(r1.agents) == 2
    assert len(r2.agents) == 2
    assert "TestAgent1" in r2.agents

    # Reset singleton for other tests
    AgentRegistry._instance = None 