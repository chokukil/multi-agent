@echo off
setlocal

echo ================================================
echo CherryAI A2A Data Science System Test
echo ================================================

echo.
echo This script will test all A2A agents for health and functionality.
echo.
echo Prerequisites:
echo - A2A servers should be running (use system_start.bat)
echo - Python environment should be activated
echo.

pause

echo.
echo Running A2A system test...
echo.

uv run python test_a2a_system.py

echo.
echo Test completed. Check the results above and a2a_test_report.json file.
echo.

pause
endlocal 