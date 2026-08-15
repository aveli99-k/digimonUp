@echo off
chcp 65001 > nul
title digimonUp - EXE build
cd /d "%~dp0"

echo [1/3] 필요한 패키지 확인...
python -m pip install --quiet --upgrade pyinstaller opencv-python numpy pillow pyautogui pywin32
if errorlevel 1 goto :fail

echo [2/3] 테스트 실행...
python -m pytest tests -q
if errorlevel 1 (
  echo.
  echo [!] 테스트가 실패했습니다. 그래도 빌드하려면 아무 키나 누르세요.
  pause > nul
)

echo [3/3] 단일 EXE 빌드...
python -m PyInstaller digimonUp.spec --noconfirm --clean
if errorlevel 1 goto :fail

echo.
echo 빌드 완료: dist\digimonUp.exe
echo.
echo [중요] EXE 를 옮길 때는 아래 항목을 **같은 폴더**에 함께 두세요.
echo        - config.json
echo        - templates\  (폴더 전체)
echo   EXE 옆의 파일을 읽으므로, 템플릿을 새로 찍거나 설정을 바꿔도
echo   다시 빌드할 필요가 없습니다.
echo.
pause
exit /b 0

:fail
echo.
echo [!] 빌드에 실패했습니다.
pause
exit /b 1
