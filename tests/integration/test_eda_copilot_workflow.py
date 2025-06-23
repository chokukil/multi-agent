import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd
from core.data_manager import data_manager
from core.plan_execute.a2a_executor import A2AExecutor
from core.schemas.messages import MediaContent, A2APlanState

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

# @pytest.mark.asyncio
# async def test_eda_copilot_profile_report_workflow():
#     """
#     Tests the EDA Copilot's 'Generate Profile Report' workflow.
#     This test confirms that the A2AExecutor correctly calls the EDA agent
#     and processes the response as expected, based on a direct analysis
#     of the A2AExecutor's source code.
#     """
#     # 1. Setup
#     df_key = "test_df_for_eda"
#     data_manager.add_dataframe(df_key, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
    
#     plan = [{
#         "agent_name": "EDA",
#         "action": "generate_profile_report",
#         "parameters": {"dataframe_key": df_key},
#     }]
#     initial_state = A2APlanState(user_prompt="Run EDA report", plan=plan)

#     # 2. Mocking
#     mock_artifact_path = "/path/to/mock_report.html"
#     mock_response_content = MediaContent(
#         type="file_path",
#         data={"file_path": mock_artifact_path}
#     )
#     # This is the dictionary that the mocked _a2a_call will return.
#     mock_a2a_result = {
#         "status": "success",
#         "content": [mock_response_content.model_dump()]
#     }

#     # 3. Execution with Patch
#     with patch.object(A2AExecutor, '_a2a_call', new_callable=AsyncMock, return_value=mock_a2a_result) as mock_a2a_call:
#         executor = A2AExecutor()
#         final_state = await executor.execute(initial_state)

#         # 4. Assertions
        
#         # 4.1. Verify the call to the agent was correct
#         mock_a2a_call.assert_awaited_once()
#         called_agent_name = mock_a2a_call.call_args[0][0]
#         called_request_body = mock_a2a_call.call_args[0][1]
        
#         assert called_agent_name == "EDA"
#         assert called_request_body['action'] == "generate_profile_report"
#         assert called_request_body['contents'][0]['data']['dataframe_key'] == df_key

#         # 4.2. Verify the final state is correct
#         assert final_state.error_message is None
#         assert len(final_state.previous_steps) == 1
        
#         agent_name, action, result = final_state.previous_steps[0]
#         assert agent_name == "EDA"
#         assert action == "generate_profile_report"
        
#         # 4.3. Verify the result structure, as confirmed from A2AExecutor source
#         assert result['status'] == "success"
#         final_artifact = result['content'][0]
#         assert final_artifact['type'] == "file_path"
#         assert final_artifact['data']['file_path'] == mock_artifact_path