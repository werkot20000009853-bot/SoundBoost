@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PYW=C:\Users\PEKARNYA\AppData\Local\Python\pythoncore-3.15-64\pythonw.exe"
if not exist "%PYW%" (
    for /f "usebackq delims=" %%i in (`py -3 -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))" 2^>nul`) do set "PYW=%%i"
)
if not exist "%PYW%" set "PYW=pythonw"

start "" "%PYW%" "%~dp0soundboost.py"
exit /b
