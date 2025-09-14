
@echo off
REM Lanzador Streamlit (Windows) para CognitivaAI
setlocal

if not exist .venv (
  python -m venv .venv
)
call .\.venv\Scripts\activate

pip install --quiet streamlit pandas numpy scikit-learn==1.7.1 joblib

streamlit run app.py

endlocal
