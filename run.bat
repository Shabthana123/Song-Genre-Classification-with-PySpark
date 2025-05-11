@echo off
echo -------------------------------------
echo Setting up and running Genre Classifier App...
echo -------------------------------------

REM Optional: create and activate virtual env
python -m venv env_bigdata
call env_bigdata\Scripts\activate

echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo Running your application...
streamlit run app.py

pause
