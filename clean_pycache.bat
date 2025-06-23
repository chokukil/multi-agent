@echo off
echo Cleaning __pycache__ directories...

:: Delete all __pycache__ directories recursively
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Deleting: %%d
    rd /s /q "%%d"
)

:: Also clean .pyc files that might be outside __pycache__
for /r . %%f in (*.pyc) do @if exist "%%f" (
    echo Deleting: %%f
    del /q "%%f"
)

:: Clean .pyo files as well
for /r . %%f in (*.pyo) do @if exist "%%f" (
    echo Deleting: %%f
    del /q "%%f"
)

echo.
echo Cleanup completed!
pause 