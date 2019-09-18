@echo off
call ..\setenv.bat

python -m pip install Flask==1.1.1
python -m pip install flask-socketio==4.2.1
python -m pip install bin/eos_py-1.1.2-cp36-cp36m-win_amd64.whl

pause
