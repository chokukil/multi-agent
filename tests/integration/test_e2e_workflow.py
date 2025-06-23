import pytest
import pandas as pd
import numpy as np
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from core.plan_execute.a2a_executor import A2AExecutor
from core.data_manager import DataManager
from core.schemas.messages import A2APlanState

# Import agent apps to ensure they are loaded for the test client
from mcp_agents.mcp_dataloader_agent import app as dataloader_app
from mcp_agents.mcp_datacleaning_agent import app as datacleaning_app
from mcp_agents.mcp_eda_agent import app as eda_app

@pytest.fixture(scope="module")
def test_clients():
    """Provides TestClients for all necessary agent servers."""
    return {
        "DataLoaderAgent": TestClient(dataloader_app),
        "DataCleaningAgent": TestClient(datacleaning_app),
        "EDAAgent": TestClient(eda_app),
    }

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Clear DataManager before and after each test."""
    dm = DataManager()
    dm.clear()
    yield
    dm.clear()

@pytest.mark.asyncio
async def test_load_clean_eda_workflow(test_clients):
    # 1. Define a mock plan that mirrors what the Planner would generate
    # Note the dependency syntax: {{steps[index]...}}
    mock_plan = [
        {
            "agent_name": "DataLoaderAgent",
            "action": "load_data",
            "parameters": {"file_path": "fake_data.csv", "output_df_id": "df_loaded"}
        },
        {
            "agent_name": "DataCleaningAgent",
            "action": "handle_missing_values",
            "parameters": {"input_df_id": "{{steps[0].output.contents[0].data.df_id}}", "output_df_id": "df_cleaned", "strategy": "mean"}
        },
        {
            "agent_name": "EDAAgent",
            "action": "get_descriptive_statistics",
            "parameters": {"data_id": "{{steps[1].output.contents[0].data.df_id}}"}
        }
    ]

    # 2. Setup initial state and mocks
    initial_state = A2APlanState(user_prompt="test prompt", plan=mock_plan)
    csv_data = "col1,col2\n1,10\n2,20\n,30\n4,40"
    mock_df = pd.read_csv(io.StringIO(csv_data))

    # Define a side effect function that routes calls to the correct TestClient
    async def a2a_call_side_effect(agent_name, request_body):
        client = test_clients[agent_name]
        response = client.post("/process", json=request_body)
        response.raise_for_status()
        return response.json()

    # 3. Patch external dependencies
    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=mock_df), \
         patch("core.plan_execute.a2a_executor.A2AExecutor._a2a_call", new_callable=AsyncMock) as mock_a2a_call_method:
        
        mock_a2a_call_method.side_effect = a2a_call_side_effect
        
        # 4. Initialize and run the executor
        executor = A2AExecutor()
        final_state = await executor.execute(initial_state)

        # 5. Assert the final state and results
        assert final_state.error_message is None
        assert final_state.current_step == 3
        
        dm = DataManager()
        
        # Check intermediate dataframe
        cleaned_df = dm.get_dataframe("df_cleaned")
        assert cleaned_df is not None
        assert cleaned_df['col1'].isnull().sum() == 0
        assert cleaned_df.loc[2, 'col1'] == pytest.approx((1+2+4)/3) 
        
        # Check final step's result in the state
        last_step_result = final_state.previous_steps[-1]
        assert last_step_result[0] == "EDAAgent" 
        final_content = last_step_result[2]['contents'][0]['data']
        assert 'col1' in final_content
        assert final_content['col1']['mean'] == pytest.approx((1+2+4)/3) 