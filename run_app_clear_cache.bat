@echo off
echo Clearing Streamlit cache...
call .\venv\Scripts\activate
streamlit cache clear
echo Cache cleared!
echo Running Streamlit app...
streamlit run app.py
pause
