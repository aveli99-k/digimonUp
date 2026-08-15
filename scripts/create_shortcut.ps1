# digimonUp 매크로 바로가기 생성 스크립트
# 사용법:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
#         (바탕화면에도 만들려면)  ... -File create_shortcut.ps1 -Desktop
#
# dist\digimonUp.exe 가 있으면 그것을, 없으면 scripts\run.bat 을 가리킨다.
# 실행하면 GUI 가 뜨고, 거기서 1) 네트워크 / 2) 탐사 를 고른다.

param(
    [switch]$Desktop
)

# 이 스크립트는 scripts/ 안에 있고 실제 프로젝트는 그 위 폴더다.
$base = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 콘솔(cmd) 창이 뜨지 않는 것부터 순서대로 고른다.
#   1) dist\digimonUp.exe  - 창 모드로 빌드돼 있어 콘솔이 없다
#   2) pythonw.exe         - 콘솔 없는 파이썬. 바로가기가 직접 가리키므로 깜빡임도 없다
#   3) scripts\run.bat     - 최후의 수단 (cmd 창이 잠깐 보인다)
$exe = Join-Path $base "dist\digimonUp.exe"
$launcher = Join-Path $base "launcher.py"
$arguments = ""

if (Test-Path $exe) {
    $target = $exe
    $workdir = Split-Path -Parent $exe
} else {
    $pythonw = $null
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $candidate = Join-Path (Split-Path -Parent $pyCmd.Source) "pythonw.exe"
        if (Test-Path $candidate) { $pythonw = $candidate }
    }
    if ($pythonw -and (Test-Path $launcher)) {
        $target = $pythonw
        $arguments = '"{0}"' -f $launcher
        $workdir = $base
    } else {
        $target = Join-Path $base "scripts\run.bat"
        $workdir = $base
    }
}

if (-not (Test-Path $target)) {
    Write-Host "실행 대상을 찾을 수 없습니다: $target"
    exit 1
}

$locations = @($base)
if ($Desktop) {
    $locations += [Environment]::GetFolderPath("Desktop")
}

$shell = New-Object -ComObject WScript.Shell

foreach ($dir in $locations) {
    $linkPath = Join-Path $dir "digimonUp 매크로.lnk"
    $sc = $shell.CreateShortcut($linkPath)
    $sc.TargetPath       = $target
    if ($arguments -ne "") { $sc.Arguments = $arguments }
    $sc.WorkingDirectory = $workdir
    $sc.Description      = "digimonUp 매크로 (1: 네트워크 / 2: 탐사)"
    $sc.WindowStyle      = 1
    $sc.IconLocation     = ('{0}\System32\shell32.dll,137' -f $env:SystemRoot)
    $sc.Save()
    Write-Host "바로가기 생성: $linkPath  ->  $target $arguments"
}
