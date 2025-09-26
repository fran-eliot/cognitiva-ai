@echo off
title CognitivaAI — Streamlit Demo
setlocal ENABLEDELAYEDEXPANSION

REM Activate venv or create it
IF EXIST .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
) ELSE (
  echo [i] No virtualenv found. Creating one...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  echo [i] Installing dependencies...
  pip install --upgrade pip
  pip install streamlit pandas numpy scikit-learn==1.7.1 joblib requests
)

REM Point to models & config (adjust if needed)
set MODELS_DIR=p26_release\models
set CONFIG_PATH=p26_release\CONFIG\deployment_config.json

echo [i] Launching Streamlit...
streamlit run app.py --server.headless true
echo.
pause
