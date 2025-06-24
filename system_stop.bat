@echo off
:: ==============================================================================
:: CherryAI - System Stop Script (BAT)
:: ==============================================================================
:: This script stops all the agent servers that were started by system_start.bat.
:: It uses taskkill to terminate the background processes by their window titles.
:: ==============================================================================

echo.
echo #############################################################
echo #         Stopping all CherryAI Agent Servers
echo #############################################################
echo.

echo [INFO] Terminating agent processes...
echo.

taskkill /F /FI "WINDOWTITLE eq DataloaderAgent" > nul 2>&1
echo   - Dataloader Agent terminated.

taskkill /F /FI "WINDOWTITLE eq WranglingAgent" > nul 2>&1
echo   - Data Wrangling Agent terminated.

taskkill /F /FI "WINDOWTITLE eq VisualizationAgent" > nul 2>&1
echo   - Data Visualization Agent terminated.

echo.
echo [SUCCESS] All running agent processes have been stopped.
echo.

timeout /t 2 /nobreak > nul 