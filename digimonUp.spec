# PyInstaller 스펙 - 단일 EXE 빌드
#   빌드:  pyinstaller digimonUp.spec --noconfirm
#   또는:  scripts\build_exe.bat 더블클릭
#
# templates/ 와 config.json 을 **EXE 안에 기본값으로 넣는다**. 그래야 받는 사람이
# 파일 하나만 받아 더블클릭하면 바로 쓸 수 있다.
#
# 그러면서도 EXE **옆에** config.json 이나 templates 폴더를 두면 그쪽이 우선한다.
# 템플릿을 자기 화면에서 다시 찍거나 설정을 고쳐도 다시 빌드할 필요가 없다.
# (우선순위 규칙은 paths.resource() 에 있다.)

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("config.json", "."),
        ("templates", "templates"),
        ("assets/icon.ico", "assets"),
    ],
    hiddenimports=[
        "win32gui", "win32ui", "win32api", "win32con",
        "tkinter", "tkinter.ttk",
        "gui", "explore", "network_macro", "board", "recognize",
        "pathfind", "overlay", "emulator_window", "settings", "common",
        "single_instance", "paths", "counters", "imgio", "version",
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
    icon="assets/icon.ico",
)
