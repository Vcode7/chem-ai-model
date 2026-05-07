@echo off
echo Starting Cataltys Backend...
cd /d "%~dp0backend"
call E:\projects\deepfake_detection_model\deepfake_env\Scripts\activate.bat
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
