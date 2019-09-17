@echo off
call ..\setenv.bat

python -m pip install Flask==1.1.1
python -m pip install flask-socketio==4.2.1

pause
