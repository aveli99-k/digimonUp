@echo off
chcp 65001 > nul
title digimonUp template capture
cd /d "%~dp0"

python capture.py

echo.
pause
