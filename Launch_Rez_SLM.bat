@echo off
title Rez_SLM Desktop Application Launcher
echo Launching Rez_SLM Control Center Desktop Application...
cd /d "%~dp0"
".venv\Scripts\python.exe" rez_slm_app.py
pause
