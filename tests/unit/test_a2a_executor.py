import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.plan_execute.a2a_executor import a2a_executor_node
from core.schemas.messages import A2APlanState
from httpx import Response, RequestError
import json

@pytest.fixture
def mock_agent_registry():
    """Fixture to provide a mock AgentRegistry that returns agent base URLs."""
    registry = MagicMock()
    registry.get_agent_base_url.return_value = "http://fake-agent-url.com"
    return registry

@pytest.mark.asyncio
async def test_executor_success_flow(mock_agent_registry):
    """
    Test the executor for a successful run where the agent API call is successful.
    """
    # Arrange
    initial_state = A2APlanState(
        user_prompt="test",
        plan=[{"agent": "TestAgent", "skill": "test_skill", "instructions": "Do it."}],
        current_step=0
    )

    mock_response_data = {"result": "success", "data_id": "new_data_123"}
    mock_response = Response(200, json=mock_response_data)
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        # Act
        result_state = await a2a_executor_node(initial_state, agent_registry_instance=mock_agent_registry)

        # Assert
        assert result_state.current_step == 1
        assert len(result_state.previous_steps) == 1
        step_result = result_state.previous_steps[0]
        assert step_result[0] == "TestAgent"
        assert step_result[1] == "test_skill"
        assert step_result[2] == mock_response_data
        assert result_state.error_message is None
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_executor_api_http_error(mock_agent_registry):
    """
    Test how the executor handles a non-200 HTTP response from the agent API.
    """
    # Arrange
    initial_state = A2APlanState(
        user_prompt="test",
        plan=[{"agent": "TestAgent", "skill": "test_skill", "instructions": "Do it."}],
        current_step=0
    )
    
    mock_response = Response(500, json={"detail": "Internal Server Error"})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        # Act
        result_state = await a2a_executor_node(initial_state, agent_registry_instance=mock_agent_registry)

        # Assert
        assert result_state.current_step == 0 # Step does not advance
        assert len(result_state.previous_steps) == 0
        assert "HTTP error 500" in result_state.error_message
        assert "Internal Server Error" in result_state.error_message

@pytest.mark.asyncio
async def test_executor_network_error(mock_agent_registry):
    """
    Test how the executor handles a network error (e.g., RequestError).
    """
    # Arrange
    initial_state = A2APlanState(
        user_prompt="test",
        plan=[{"agent": "TestAgent", "skill": "test_skill", "instructions": "Do it."}],
        current_step=0
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=RequestError("Connection failed")) as mock_post:
        # Act
        result_state = await a2a_executor_node(initial_state, agent_registry_instance=mock_agent_registry)

        # Assert
        assert result_state.current_step == 0
        assert "Network error calling TestAgent" in result_state.error_message
        assert "Connection failed" in result_state.error_message

@pytest.mark.asyncio
async def test_executor_no_plan(mock_agent_registry):
    """
    Test that the executor does nothing if the plan is empty.
    """
    # Arrange
    initial_state = A2APlanState(user_prompt="test", plan=[], current_step=0)

    # Act
    result_state = await a2a_executor_node(initial_state, agent_registry_instance=mock_agent_registry)

    # Assert
    assert result_state == initial_state # State should be unchanged

@pytest.mark.asyncio
async def test_executor_plan_finished(mock_agent_registry):
    """
    Test that the executor does nothing if all plan steps are completed.
    """
    # Arrange
    initial_state = A2APlanState(
        user_prompt="test",
        plan=[{"agent": "TestAgent", "skill": "test_skill", "instructions": "Do it."}],
        current_step=1 # Already past the last step
    )

    # Act
    result_state = await a2a_executor_node(initial_state, agent_registry_instance=mock_agent_registry)

    # Assert
    assert result_state == initial_state 