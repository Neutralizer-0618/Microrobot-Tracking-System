@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found in .venv
  echo Please install dependencies first.
  pause
  exit /b 1
)

echo Starting MicroMotorTracker-AI...
".venv\Scripts\python.exe" -m streamlit run micromotor_tracker/app.py
