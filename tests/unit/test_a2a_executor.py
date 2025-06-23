import pytest
from unittest.mock import patch, AsyncMock

from core.plan_execute.a2a_executor import A2AExecutor
from core.schemas.messages import A2APlanState
from core.agents.agent_registry import AgentRegistry

@pytest.fixture
def mock_agent_registry():
    registry = AgentRegistry()
    registry.agents = {
        "TestAgent": {
            "agent_name": "TestAgent",
            "url": "http://test.agent"
        }
    }
    return registry

@pytest.mark.asyncio
async def test_a2a_executor_success(mock_agent_registry):
    """
    Tests successful execution of a single-step plan by the A2AExecutor class.
    """
    plan = [{"agent_name": "TestAgent", "action": "do_work", "parameters": {"param1": "value1"}}]
    initial_state = A2APlanState(user_prompt="test", plan=plan)
    
    executor = A2AExecutor(agent_registry=mock_agent_registry)
    
    mock_response = {
        "status": "success",
        "message": "Work done",
        "content": [{"type": "text", "data": "Result data"}]
    }

    with patch.object(executor, '_a2a_call', new_callable=AsyncMock) as mock_a2a_call:
        mock_a2a_call.return_value = mock_response
        
        final_state = await executor.execute(initial_state)

        mock_a2a_call.assert_awaited_once()
        
        assert final_state.error_message is None
        assert len(final_state.previous_steps) == 1
        
        agent, skill, result = final_state.previous_steps[0]
        assert agent == "TestAgent"
        assert skill == "do_work"
        assert result == mock_response

@pytest.mark.asyncio
async def test_a2a_executor_step_failure(mock_agent_registry):
    """
    Tests that the executor stops and records an error if a step fails.
    """
    plan = [{"agent_name": "TestAgent", "action": "do_work", "parameters": {}}]
    initial_state = A2APlanState(user_prompt="test", plan=plan)
    
    executor = A2AExecutor(agent_registry=mock_agent_registry)
    
    mock_response = {"status": "failure", "message": "It broke"}

    with patch.object(executor, '_a2a_call', new_callable=AsyncMock) as mock_a2a_call:
        mock_a2a_call.return_value = mock_response
        
        final_state = await executor.execute(initial_state)

        mock_a2a_call.assert_awaited_once()
        
        assert "Step 1 failed: It broke" in final_state.error_message
        assert len(final_state.previous_steps) == 0

@pytest.mark.asyncio
async def test_a2a_executor_http_error(mock_agent_registry):
    """
    Tests that the executor handles HTTP errors during A2A calls.
    """
    plan = [{"agent_name": "TestAgent", "action": "do_work", "parameters": {}}]
    initial_state = A2APlanState(user_prompt="test", plan=plan)
    
    executor = A2AExecutor(agent_registry=mock_agent_registry)

    with patch.object(executor, '_a2a_call', new_callable=AsyncMock) as mock_a2a_call:
        mock_a2a_call.side_effect = Exception("Connection error")
        
        final_state = await executor.execute(initial_state)

        mock_a2a_call.assert_awaited_once()
        
        assert "An unexpected error occurred during step 1: Connection error" in final_state.error_message
        assert len(final_state.previous_steps) == 0 