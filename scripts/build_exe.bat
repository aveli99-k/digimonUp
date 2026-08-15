@echo off
chcp 65001 > nul
title digimonUp - EXE build
cd /d "%~dp0.."

echo [1/4] 필요한 패키지 확인...
REM 패키지 목록은 requirements 파일 한 곳에만 둔다. 예전에는 이 줄에 직접
REM 적어 두어서, 저장소를 받은 사람이 무엇을 설치해야 하는지 알 수 없었다.
python -m pip install --quiet -r requirements.txt -r requirements-dev.txt
if errorlevel 1 goto :fail

echo [2/4] 테스트 실행...
python -m pytest tests -q
if errorlevel 1 (
  echo.
  echo [!] 테스트가 실패했습니다. 그래도 빌드하려면 아무 키나 누르세요.
  pause > nul
)

echo [3/4] 단일 EXE 빌드...
python -m PyInstaller digimonUp.spec --noconfirm --clean
if errorlevel 1 goto :fail

REM EXE 는 자기 옆의 config.json / templates 를 읽는다(EXE 안에 넣지 않는다).
REM 예전에는 이 복사를 손으로 했더니 dist 쪽 사본이 조용히 낡아서, 지운 설정이
REM 남아 있거나 새로 찍은 템플릿이 반영되지 않았다. 그래서 빌드가 직접 맞춘다.
echo [4/4] 설정과 템플릿을 dist 로 동기화...
copy /y config.json dist\config.json > nul
if errorlevel 1 goto :fail
robocopy templates dist\templates /MIR /NJH /NJS /NDL /NP /NFL > nul
REM robocopy 는 성공해도 0 이 아닌 코드를 돌려준다(8 미만이면 정상).
if errorlevel 8 goto :fail

echo.
echo 빌드 완료: dist\digimonUp.exe
echo   config.json 과 templates\ 를 dist\ 에 맞춰 두었습니다.
echo   EXE 를 다른 곳으로 옮길 때는 이 세 가지를 **같은 폴더**에 함께 두세요.
echo.
pause
exit /b 0

:fail
echo.
echo [!] 빌드에 실패했습니다.
pause
exit /b 1
