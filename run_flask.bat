@echo off
cd /d C:\Github_repos\sepsis-prediction-project
python apps\sepsis_dashboard\backend\flask_backend.py > backend_output.log 2>&1
echo Backend started. PID: %ERRORLEVEL%