@echo off
chcp 65001 > nul
cd /d "%~dp0"

REM 인자가 없으면 GUI 만 띄운다. pythonw 는 콘솔이 없는 파이썬이라
REM 검은 cmd 창이 뜨지 않는다. start /b 로 띄우고 배치는 바로 끝난다.
REM (콘솔 없이 완전히 깔끔하게 띄우려면 digimonUp.vbs 를 더블클릭하세요.)
if "%~1"=="" (
    start "" /b pythonw.exe launcher.py
    exit /b 0
)

REM 번호를 주면 콘솔에서 로그를 보며 실행한다:  run.bat 1  /  run.bat 2
title digimonUp macro
python launcher.py %*
if errorlevel 9009 echo [!] Python not found. Please check your Python installation.
echo.
pause
