@echo off
:: ==============================================================================
:: CherryAI - Pandas Data Analyst Workflow Test Runner (BAT)
:: ==============================================================================
:: This script automates the process of:
:: 1. Starting all required agent servers in background windows.
:: 2. Waiting for the servers to initialize.
:: 3. Running the integration test for the Pandas Data Analyst workflow.
:: 4. Shutting down all agent servers after the test is complete.
:: ==============================================================================

echo.
echo #############################################################
echo #  Running Pandas Data Analyst Workflow Test
echo #############################################################
echo.

echo [1/4] Starting agent servers in background...
:: Start agents in separate, titled, background command windows
START "DataloaderAgent" /B uv run python -m mcp_agents.mcp_dataloader_agent
START "WranglingAgent" /B uv run python -m mcp_agents.mcp_datawrangling_agent
START "VisualizationAgent" /B uv run python -m mcp_agents.mcp_datavisualization_agent
echo      - Dataloader, Wrangling, and Visualization agents started.

echo.
echo [2/4] Waiting for servers to initialize (5 seconds)...
timeout /t 5 /nobreak > nul

echo.
echo [3/4] Running Pytest for the integration test...
uv run pytest tests/integration/test_pandas_data_analyst_workflow.py -v
set TEST_EXIT_CODE=%errorlevel%

echo.
echo [4/4] Shutting down agent servers...
:: Terminate the agents by the window title we assigned them
taskkill /F /FI "WINDOWTITLE eq DataloaderAgent" > nul
taskkill /F /FI "WINDOWTITLE eq WranglingAgent" > nul
taskkill /F /FI "WINDOWTITLE eq VisualizationAgent" > nul
echo      - All agent processes terminated.

echo.
echo #############################################################
echo #  Test run finished.
echo #############################################################
echo.

exit /b %TEST_EXIT_CODE% 