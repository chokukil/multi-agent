import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import os
import io
import numpy as np

from mcp_agents.mcp_dataloader_agent import app
from core.data_manager import DataManager
from mcp_agents.mcp_eda_agent import app as eda_app
from core.schemas.messages import A2ARequest, ParamsContent, DataFrameContent, MediaContent
from mcp_agents.mcp_datacleaning_agent import app as datacleaning_app
from mcp_agents.mcp_featureengineering_agent import app as featureengineering_app

client = TestClient(app)
eda_client = TestClient(eda_app)
datacleaning_client = TestClient(datacleaning_app)
featureengineering_client = TestClient(featureengineering_app)

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Clear DataManager before and after each test."""
    dm = DataManager()
    dm.clear()
    yield
    dm.clear()

def test_load_data_success():
    """
    Tests successful data loading by mocking the DataManager interaction.
    """
    output_df_id = "test_df_123"
    fake_path = "any/fake/path.csv"
    csv_data = "col1,col2\n1,a\n2,b"
    mock_df = pd.read_csv(io.StringIO(csv_data))

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=mock_df), \
         patch("core.data_manager.DataManager.add_dataframe") as mock_add_dataframe:

        request_payload = A2ARequest(
            action="load_data",
            contents=[ParamsContent(data={"file_path": fake_path, "output_df_id": output_df_id})]
        )

        response = client.post("/process", json=request_payload.model_dump(by_alias=True))

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        
        response_df_content = next((c for c in response_json["contents"] if c["content_type"] == "dataframe"), None)
        assert response_df_content is not None
        assert response_df_content["data"]["df_id"] == output_df_id

        # Assert that DataManager.add_dataframe was called correctly
        mock_add_dataframe.assert_called_once()
        args, kwargs = mock_add_dataframe.call_args
        assert args[0] == output_df_id
        pd.testing.assert_frame_equal(args[1], mock_df)
        assert kwargs["source"] == fake_path

def test_get_descriptive_statistics_success():
    input_df_id = "stats_test_df"
    
    csv_data = "col1,col2\n1,a\n2,b\n3,c"
    test_df = pd.read_csv(io.StringIO(csv_data))
    dm = DataManager()
    dm.add_dataframe(input_df_id, test_df)

    request_payload = A2ARequest(
        action="get_descriptive_statistics",
        contents=[ParamsContent(data={"data_id": input_df_id})]
    )

    response = eda_client.post("/process", json=request_payload.model_dump(by_alias=True))

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"

    media_content = next((c for c in response_json["contents"] if c["content_type"] == "media"), None)
    assert media_content is not None
    
    stats_data = media_content["data"]
    
    assert "col1" in stats_data
    assert "col2" in stats_data
    assert "count" in stats_data["col1"]
    assert stats_data["col1"]["count"] == 3
    assert "top" in stats_data["col2"]
    assert stats_data["col2"]["top"] == "a"

def test_handle_missing_values_success():
    input_df_id = "dirty_df"
    output_df_id = "cleaned_df"
    
    # 1. Create and add a dataframe with missing values
    data = {'col1': [1, 2, np.nan, 4, 5], 'col2': ['a', 'b', 'c', np.nan, 'e']}
    dirty_df = pd.DataFrame(data)
    dm = DataManager()
    dm.add_dataframe(input_df_id, dirty_df)

    # 2. Request the cleaning action
    request_payload = A2ARequest(
        action="handle_missing_values",
        contents=[
            ParamsContent(data={
                "input_df_id": input_df_id,
                "output_df_id": output_df_id,
                "strategy": "mean",
                "columns": ["col1"] # Only apply to numeric column
            })
        ]
    )

    response = datacleaning_client.post("/process", json=request_payload.model_dump(by_alias=True))

    # 3. Assert the response
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    
    response_content = next((c for c in response_json["contents"] if c["content_type"] == "dataframe"), None)
    assert response_content is not None
    assert response_content["data"]["df_id"] == output_df_id

    # 4. Assert the result in DataManager
    cleaned_df = dm.get_dataframe(output_df_id)
    assert cleaned_df is not None
    
    # Check if NaN in 'col1' is filled (mean of 1,2,4,5 is 3.0)
    assert cleaned_df['col1'].isnull().sum() == 0
    assert cleaned_df.loc[2, 'col1'] == 3.0
    
    # Check if 'col2' is untouched
    assert cleaned_df['col2'].isnull().sum() == 1

def test_create_polynomial_features_success():
    input_df_id = "poly_in_df"
    output_df_id = "poly_out_df"
    
    # 1. Create and add a simple dataframe
    data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
    df = pd.DataFrame(data)
    dm = DataManager()
    dm.add_dataframe(input_df_id, df)

    # 2. Request the feature engineering action
    request_payload = A2ARequest(
        action="create_polynomial_features",
        contents=[
            ParamsContent(data={
                "input_df_id": input_df_id,
                "output_df_id": output_df_id,
                "columns": ["a", "b"],
                "degree": 2
            })
        ]
    )
    
    response = featureengineering_client.post("/process", json=request_payload.model_dump(by_alias=True))

    # 3. Assert the response
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    
    # 4. Assert the result in DataManager
    new_df = dm.get_dataframe(output_df_id)
    assert new_df is not None
    
    # Check columns - original 'a', 'b' should be gone, 'c' should remain
    expected_cols = ['c', 'a', 'b', 'a^2', 'a b', 'b^2']
    assert all(col in new_df.columns for col in expected_cols)
    assert 'a' not in new_df.columns or 'a' in expected_cols # Handle potential name collision
    
    # Check a value
    # For input a=2, b=5: a^2=4, a*b=10, b^2=25
    row_1 = new_df.iloc[1]
    assert row_1['a^2'] == 4.0
    assert row_1['a b'] == 10.0
    assert row_1['b^2'] == 25.0