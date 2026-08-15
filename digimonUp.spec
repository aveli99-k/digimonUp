# PyInstaller 스펙 - 단일 EXE 빌드
#   빌드:  pyinstaller digimonUp.spec --noconfirm
#   또는:  build_exe.bat 더블클릭
#
# templates/ 와 config.json 은 EXE 안에 넣지 않고 **EXE 옆에 그대로 둔다**.
# 템플릿을 새로 찍거나 설정을 바꿀 때마다 다시 빌드하지 않아도 되게 하기 위해서다.
# (common.py / recognize.py 는 실행 파일 위치를 기준으로 이 폴더들을 찾는다.)

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[
        "win32gui", "win32ui", "win32api", "win32con",
        "tkinter", "tkinter.ttk",
        "gui", "explore", "network_macro", "board", "recognize",
        "pathfind", "overlay", "mumu_window", "settings", "common",
        "single_instance", "paths",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "matplotlib", "PyQt5", "PySide2", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="digimonUp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI 앱이라 콘솔 창을 띄우지 않는다
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
