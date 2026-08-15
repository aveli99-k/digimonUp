@echo off
chcp 65001 > nul
title digimonUp detection check
cd /d "%~dp0"

python check.py

echo.
pause
