import pytest
from unittest.mock import MagicMock, AsyncMock
from core.plan_execute.planner import planner_node
from core.schemas.messages import A2APlanState
import json

@pytest.fixture
def mock_agent_registry():
    """Fixture to provide a mock AgentRegistry."""
    registry = MagicMock()
    registry.get_all_skills_summary.return_value = "Agent: TestAgent, Skill: test_skill"
    return registry

@pytest.fixture
def mock_llm():
    """Fixture to provide a mock LLM."""
    llm = MagicMock()
    # Use AsyncMock for the async invoke method
    llm.ainvoke = AsyncMock()
    return llm

@pytest.mark.asyncio
async def test_planner_node_success(mock_llm, mock_agent_registry):
    """
    Test that the planner_node successfully generates a valid plan
    when the LLM returns a well-formed JSON string.
    """
    # Arrange: Configure the mock LLM to return a valid JSON plan
    plan_dict = {
        "plan": [
            {"agent": "TestAgent", "skill": "test_skill", "instructions": "Do something."}
        ],
        "thought": "The plan is simple."
    }
    mock_llm.ainvoke.return_value.content = json.dumps(plan_dict)
    
    # Arrange: Prepare the initial state
    initial_state = A2APlanState(
        session_id="test_session_1",
        user_prompt="Test prompt",
        plan=[],
        plan_str="",
        thought="",
        previous_steps=[]
    )
    
    # Act: Run the planner node
    result_state = await planner_node(initial_state, llm=mock_llm, agent_registry_instance=mock_agent_registry)
    
    # Assert: Check that the state was updated correctly
    assert result_state.thought == "The plan is simple."
    assert len(result_state.plan) == 1
    assert result_state.plan[0]["agent"] == "TestAgent"
    assert result_state.plan_str == json.dumps(plan_dict, indent=2)
    mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_planner_node_json_error(mock_llm, mock_agent_registry):
    """
    Test that the planner_node handles malformed JSON from the LLM gracefully
    by setting the thought to an error message and leaving the plan empty.
    """
    # Arrange: Configure the mock LLM to return a malformed JSON string
    malformed_json = '{"plan": [{"agent": "TestAgent"}], "thought": "oops" extra_comma,}'
    mock_llm.ainvoke.return_value.content = malformed_json

    initial_state = A2APlanState(
        session_id="test_session_2",
        user_prompt="Test prompt",
        plan=[],
        plan_str="",
        thought="",
        previous_steps=[]
    )

    # Act
    result_state = await planner_node(initial_state, llm=mock_llm, agent_registry_instance=mock_agent_registry)

    # Assert
    assert "Failed to parse LLM response into JSON" in result_state.thought
    assert result_state.plan == [] # Plan should remain empty

@pytest.mark.asyncio
async def test_planner_node_missing_keys(mock_llm, mock_agent_registry):
    """
    Test that the planner_node handles JSON that is missing required keys ('plan', 'thought')
    by setting an appropriate error message.
    """
    # Arrange: Configure the mock LLM to return JSON missing the 'plan' key
    invalid_dict = {"thought": "This is a thought without a plan."}
    mock_llm.ainvoke.return_value.content = json.dumps(invalid_dict)

    initial_state = A2APlanState(
        session_id="test_session_3",
        user_prompt="Test prompt",
        plan=[],
        plan_str="",
        thought="",
        previous_steps=[]
    )

    # Act
    result_state = await planner_node(initial_state, llm=mock_llm, agent_registry_instance=mock_agent_registry)

    # Assert
    assert "LLM response must include 'plan' and 'thought' keys." in result_state.thought
    assert result_state.plan == [] 