import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, MagicMock, any_call
import os
import io

from mcp_agents.mcp_dataloader_agent import DataLoaderAgent
from core.data_manager import DataManager
from core.schemas.messages import A2ARequest, Message, ToolCall, ToolCallResult, Content, ParamsContent

@pytest.fixture(scope="module")
def agent():
    agent_instance = DataLoaderAgent(config_path="mcp-configs/dataloader_agent.json")
    return agent_instance

@pytest.fixture(scope="module")
def test_app(agent):
    return agent.app

@pytest.fixture(scope="function")
def client(test_app):
    with TestClient(test_app) as c:
        yield c

@pytest.fixture(autouse=True)
def cleanup_data_manager():
    """Ensure DataManager is clean before and after each test."""
    DataManager.clear()
    yield
    DataManager.clear()

def test_load_csv_from_path(client: TestClient):
    """
    Test loading a CSV from a file path.
    This test mocks file existence and reading to avoid actual file I/O.
    """
    csv_data = "col1,col2\n1,2\n3,4"
    fake_file_path = "/fake/path/to/sample_data.csv"
    
    # Create an in-memory string buffer for the CSV data
    string_io_buffer = io.StringIO(csv_data)
    
    expected_df = pd.read_csv(io.StringIO(csv_data)) # Re-read for a clean comparison object

    # Mock os.path.exists to return True for our fake path
    # Mock pandas.read_csv to read from our in-memory buffer instead of a file
    with patch('os.path.exists') as mock_exists, \
         patch('pandas.read_csv') as mock_read_csv:

        mock_exists.return_value = True
        mock_read_csv.return_value = expected_df

        request_data = A2ARequest(
            messages=[
                Message(
                    role="user",
                    content=Content(
                        content_type="params",
                        content=ParamsContent(
                            tool_name="load_csv",
                            tool_params={"filepath": fake_file_path}
                        )
                    )
                )
            ]
        )

        response = client.post("/process", json=request_data.dict())

    assert response.status_code == 200
    response_json = response.json()

    # Check that the response indicates success
    assert response_json["messages"][0]["content"]["content_type"] == "tool_code"
    
    tool_call_content = response_json["messages"][0]["content"]["content"]
    assert tool_call_content["tool_name"] == "load_csv"
    assert "Successfully loaded" in tool_call_content["tool_code"]

    # Verify the DataFrame was stored in DataManager
    loaded_df_info = DataManager.get_data_info("dataframe_")
    assert len(loaded_df_info) == 1
    data_key = list(loaded_df_info.keys())[0]
    loaded_df = DataManager.get_data(data_key)

    pd.testing.assert_frame_equal(loaded_df, expected_df)

    # Assert that os.path.exists was called with our fake file path
    # We use assert_has_calls with any_call because the agent's internal logic
    # might call os.path.exists for other reasons (e.g., checking directories).
    mock_exists.assert_has_calls([any_call(fake_file_path)], any_order=True)
    
    # Assert that read_csv was called once with the correct path
    mock_read_csv.assert_called_once_with(fake_file_path, encoding='utf-8', sep=',')

def test_file_not_found(client: TestClient):
    """Test the agent's behavior when the requested file does not exist."""
    fake_file_path = "/non/existent/file.csv"

    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False

        request_data = A2ARequest(
            messages=[
                Message(
                    role="user",
                    content=Content(
                        content_type="params",
                        content=ParamsContent(
                            tool_name="load_csv",
                            tool_params={"filepath": fake_file_path}
                        )
                    )
                )
            ]
        )

        response = client.post("/process", json=request_data.dict())

    assert response.status_code == 200 # Agent should handle the error gracefully
    response_json = response.json()
    
    # Check that the response contains a tool call result with an error message
    tool_result_content = response_json["messages"][0]["content"]["content"]
    assert "Error" in tool_result_content["status"]
    assert "File not found" in tool_result_content["result"]

    # Ensure no data was added to DataManager
    assert len(DataManager.get_data_info()) == 0 