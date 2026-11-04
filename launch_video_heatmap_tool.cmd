@echo off
setlocal
cd /d "%~dp0"

python -c "import PySide6; assert PySide6.__version__ == '6.11.1'" >nul 2>&1
if not errorlevel 1 (
    python scripts\launch_heatmap_tool.py %*
    if errorlevel 1 goto launch_failed
    exit /b 0
)

where conda >nul 2>&1
if errorlevel 1 goto missing
call conda run -n video-heatmap python -c "import PySide6; assert PySide6.__version__ == '6.11.1'" >nul 2>&1
if errorlevel 1 goto missing
call conda run -n video-heatmap python scripts\launch_heatmap_tool.py %*
if errorlevel 1 goto launch_failed
exit /b 0

:launch_failed
echo.
echo Video Heatmap Tool exited with an error. Review the message above.
pause
exit /b 1

:missing
echo Video Heatmap Tool could not find an active Python with PySide6 6.11.1
echo or the project Conda environment named video-heatmap.
echo.
echo Activate/install the project environment, then try again.
pause
exit /b 2
