import pytest
import pandas as pd
from core.data_manager import DataManager

@pytest.fixture(scope="function")
def data_manager():
    """Provides a clean DataManager instance for each test function."""
    # Using a new instance for each test ensures isolation
    dm = DataManager()
    dm.clear() # Use the correct clear method
    return dm

def test_add_and_get_dataframe(data_manager):
    """Test adding a dataframe and retrieving it."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    data_id = "test_df_1"
    
    data_manager.add_dataframe(data_id, df, source="test_source")
    
    retrieved_df = data_manager.get_dataframe(data_id)
    
    assert retrieved_df is not None
    assert isinstance(retrieved_df, pd.DataFrame)
    pd.testing.assert_frame_equal(retrieved_df, df)

def test_get_non_existent_dataframe(data_manager):
    """Test retrieving a non-existent dataframe returns None."""
    retrieved_df = data_manager.get_dataframe("non_existent_id")
    assert retrieved_df is None

def test_get_data_info(data_manager):
    """Test getting metadata information about a stored dataframe."""
    df = pd.DataFrame({'a': list(range(10))})
    data_id = "test_df_info"
    source = "test_upload"
    
    data_manager.add_dataframe(data_id, df, source=source)
    
    info = data_manager.get_data_info(data_id)
    
    assert info is not None
    assert info['data_id'] == data_id
    assert info['source'] == source
    assert info['metadata']['shape'] == (10, 1)
    assert 'created_at' in info

def test_list_dataframe_info(data_manager):
    """Test listing all stored dataframes."""
    df1 = pd.DataFrame({'a': [1]})
    df2 = pd.DataFrame({'b': [2]})
    
    data_manager.add_dataframe("df1", df1, source="s1")
    data_manager.add_dataframe("df2", df2, source="s2")
    
    all_data_info = data_manager.list_dataframe_info()
    
    assert len(all_data_info) == 2
    data_ids = [d['data_id'] for d in all_data_info]
    assert "df1" in data_ids
    assert "df2" in data_ids

def test_delete_dataframe(data_manager):
    """Test deleting a dataframe."""
    df = pd.DataFrame({'c': [5]})
    data_id = "df_to_delete"
    
    data_manager.add_dataframe(data_id, df)
    assert data_manager.get_dataframe(data_id) is not None
    
    data_manager.delete_dataframe(data_id)
    assert data_manager.get_dataframe(data_id) is None

def test_clear_all_data(data_manager):
    """Test clearing all data from the manager."""
    data_manager.add_dataframe("d1", pd.DataFrame())
    data_manager.add_dataframe("d2", pd.DataFrame())
    
    assert len(data_manager.list_dataframe_info()) == 2
    
    data_manager.clear()
    
    assert len(data_manager.list_dataframe_info()) == 0

def test_singleton_behavior():
    """Test that DataManager behaves as a singleton."""
    dm1 = DataManager()
    dm2 = DataManager()
    
    assert dm1 is dm2
    
    dm1.add_dataframe("singleton_test", pd.DataFrame({'x': [1]}))
    retrieved_df = dm2.get_dataframe("singleton_test")
    assert retrieved_df is not None
    
    # Clean up after test
    dm1.clear() 