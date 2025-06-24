# tests/integration/test_pandas_data_analyst_workflow.py
import asyncio
import os
import shutil
from pathlib import Path

import pytest
from core.agents.agent_registry import AgentRegistry
from core.data_manager import DataManager
from core.plan_execute.a2a_executor import A2AExecutor
from core.schemas.messages import Plan, Step

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    """Ensure a single event loop for the whole test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def agent_registry():
    """
    Provides a configured AgentRegistry with the agents needed for this workflow.
    These agents are expected to be running externally for this test.
    """
    registry = AgentRegistry()
    # These URLs should match the running agents' configurations
    registry.register_agent(
        "dataloader_agent",
        "http://localhost:8001",
        "Agent for loading data from various sources.",
    )
    registry.register_agent(
        "data_wrangling_agent",
        "http://localhost:8002",
        "Agent for cleaning and transforming data.",
    )
    registry.register_agent(
        "data_visualization_agent",
        "http://localhost:8003",
        "Agent for creating data visualizations.",
    )
    return registry

@pytest.fixture(scope="module")
def data_manager():
    """Provides a DataManager instance pointed to the test sandbox."""
    sandbox_dir = Path(__file__).parent.parent.parent / "sandbox"
    return DataManager(base_dir=str(sandbox_dir))

@pytest.fixture
def setup_test_data(data_manager: DataManager):
    """
    Sets up a test CSV file in the sandbox datasets directory and yields its name.
    Cleans up the file afterwards.
    """
    test_data = "col1,col2\n1,a\n2,b\n2,b\n3,c\n"
    test_csv_path = data_manager.get_artifact_dir("datasets") / "test_data_analyst.csv"
    
    with open(test_csv_path, "w") as f:
        f.write(test_data)
    
    yield test_csv_path.name
    
    # Cleanup: remove the created csv and any generated plots
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
    
    plots_dir = data_manager.get_artifact_dir("plots")
    for item in plots_dir.iterdir():
        if "test_data_analyst" in item.name or "histogram" in item.name:
            if item.is_file():
                os.remove(item)

# ------------------------------------------------------------------------------
# Integration Test
# ------------------------------------------------------------------------------

async def test_pandas_data_analyst_full_workflow(
    agent_registry: AgentRegistry,
    data_manager: DataManager,
    setup_test_data: str,
):
    """
    Tests the full "load -> wrangle -> visualize" workflow by executing a plan
    that calls three different agents in sequence.
    """
    # Arrange: Create the executor and the plan
    a2a_executor = A2AExecutor(data_manager, agent_registry)
    csv_file_name = setup_test_data

    pandas_analyst_plan = Plan(
        id="plan_pandas_analyst_workflow",
        steps=[
            Step(
                id="load_step",
                agent_name="dataloader_agent",
                skill_name="load_csv",
                parameters={"file_name": csv_file_name},
            ),
            Step(
                id="wrangle_step",
                agent_name="data_wrangling_agent",
                skill_name="remove_duplicates",
                parameters={"dataset_id": "{{load_step.data.dataset_id}}"},
            ),
            Step(
                id="visualize_step",
                agent_name="data_visualization_agent",
                skill_name="create_visualization",
                parameters={
                    "dataset_id": "{{wrangle_step.data.new_dataset_id}}",
                    "plot_type": "histogram",
                    "x_column": "col1",
                },
            ),
        ],
    )

    # Act: Execute the plan
    results = await a2a_executor.aexecute(plan=pandas_analyst_plan, timeout=30)

    # Assert: Verify the results of each step
    assert len(results) == 3, "The plan should have three output results."
    
    # Step 1: Dataloader result
    load_result = results[0]
    assert load_result.exit_code == 0
    assert "dataset_id" in load_result.data
    assert load_result.data["rows"] == 4 # Initial data has 4 rows

    # Step 2: Data wrangling result
    wrangle_result = results[1]
    assert wrangle_result.exit_code == 0
    assert "new_dataset_id" in wrangle_result.data
    assert wrangle_result.data["rows_removed"] == 1 # One duplicate row '2,b'
    
    # Step 3: Data visualization result
    viz_result = results[2]
    assert viz_result.exit_code == 0
    assert "plot_path" in viz_result.data
    
    # Verify the final artifact exists
    plot_path = viz_result.data["plot_path"]
    assert os.path.exists(plot_path), f"Plot file should be created at {plot_path}"
    assert "histogram" in Path(plot_path).name, "The filename should indicate a histogram."
    
    # Verify the content of the wrangled data
    wrangled_df = data_manager.load_dataframe(wrangle_result.data["new_dataset_id"])
    assert len(wrangled_df) == 3, "Wrangled dataframe should have 3 unique rows." 