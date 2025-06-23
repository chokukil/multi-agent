import pytest
import asyncio
from unittest.mock import patch, AsyncMock
import json

import pandas as pd
from core.data_manager import data_manager
from core.plan_execute.a2a_executor import A2AExecutor
from core.schemas.messages import A2APlanState, A2ARequest, ParamsContent
from core.agents.agent_registry import AgentRegistry

# a2a-sdk models for mocking
from a2a.types import Message as A2AResponse
from a2a.utils.message import new_agent_text_message
from a2a.utils.message import get_message_text

@pytest.fixture(scope="module")
def event_loop():
    """Provides an asyncio event loop for the test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def clear_data_manager_before_each_test():
    """Ensures the data_manager is clean before each test."""
    data_manager.clear()
    yield

@pytest.mark.asyncio
async def test_eda_copilot_profile_report_workflow_with_a2a_sdk():
    """
    Tests the EDA Copilot workflow now that the system uses a2a-sdk.
    This test mocks the `a2a.client.post` function to simulate a successful
    call to the EDA agent.
    """
    # 1. Setup
    df_key = "test_df_for_eda"
    data_manager.add_dataframe(df_key, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
    
    plan = [{
        "agent_name": "EDA",
        "action": "generate_profile_report",
        "parameters": {"dataframe_key": df_key},
    }]
    initial_state = A2APlanState(user_prompt="Run EDA report", plan=plan)

    # 2. Mocking the A2A Response
    mock_artifact_path = "/path/to/mock_report.html"
    # Create a realistic Message object as the mock response
    mock_response_text = json.dumps({"status": "success", "file_path": mock_artifact_path})
    # The _a2a_call now returns a dict, so we create a dict mock response
    mock_message_dict = new_agent_text_message(mock_response_text).model_dump(by_alias=True)
    # The executor checks for a 'status' key in the top-level dictionary.
    mock_response = {**mock_message_dict, "status": "success"}

    # 3. Execution with Patching the internal _a2a_call method
    with patch('core.plan_execute.a2a_executor.A2AExecutor._a2a_call', new_callable=AsyncMock, return_value=mock_response) as mock_a2a_call:
        # We need to register a dummy URL for the agent
        # Because __init__ is complex, we patch it and manually set the registry
        with patch.object(A2AExecutor, '__init__', lambda self, agent_registry=None: None):
             executor = A2AExecutor()
             # Manually create and set a dummy registry
             registry = AgentRegistry(config_dir=None) # Prevent loading from disk
             registry.agents["EDA"] = {"url": "http://localhost:8008/process"}
             executor._agent_registry = registry
        
             final_state = await executor.execute(initial_state)

    # 4. Assertions
    mock_a2a_call.assert_awaited_once()
    
    # 4.1. Verify the call arguments to _a2a_call
    call_args = mock_a2a_call.call_args
    called_agent_name = call_args[0][0]
    called_request = call_args[0][1]
    
    assert called_agent_name == "EDA"
    assert isinstance(called_request, A2ARequest)
    assert called_request.action == "generate_profile_report"
    assert called_request.contents[0].data['dataframe_key'] == df_key

    # 4.2. Verify the final state
    assert final_state.error_message is None
    result_dict = final_state.previous_steps[0][2] # (agent, action, result)
    
    # The result in previous_steps is the full mock_response dict
    assert result_dict['status'] == 'success'
    result_text = get_message_text(A2AResponse(**result_dict))
    result_data = json.loads(result_text)

    assert result_data['status'] == 'success'
    assert result_data['file_path'] == mock_artifact_path