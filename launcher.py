"""바로가기 진입점.

기본은 GUI 다. 콘솔에서 번호로만 돌리고 싶으면 인자를 준다.

    python launcher.py           -> GUI (기능 번호는 GUI 안에서 선택)
    python launcher.py 1         -> 콘솔에서 바로 네트워크
    python launcher.py 2         -> 콘솔에서 바로 탐사
    python launcher.py --console -> 콘솔에서 번호를 물어본 뒤 실행
    python launcher.py --version -> 버전만 찍고 종료
"""

from __future__ import annotations

import io
import sys


def _silence_missing_console() -> None:
    """pythonw.exe(콘솔 없는 파이썬)로 실행하면 sys.stdout 이 None 이다.

    그 상태에서 print() 를 부르면 AttributeError 로 죽는다. GUI 만 띄우는 게
    목적이므로, 표준 출력이 없으면 버리는 스트림으로 바꿔 둔다.
    (PyInstaller 로 만든 창 모드 EXE 도 마찬가지다.)
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, io.StringIO())


_silence_missing_console()

from digimonup.win import single_instance  # noqa: E402
from digimonup.win.emulator_window import enable_dpi_awareness  # noqa: E402
from digimonup.base.version import __version__, version_line  # noqa: E402

MENU = f"""
============================================
 digimonUp 매크로  v{__version__}
============================================
  1) 네트워크  - 매칭/포기 자동 클릭
  2) 탐사      - 5x5 게임판 자동 이동
  q) 종료
"""


def run_console(mode: str) -> int:
    if mode == "1":
        from digimonup.app.network_macro import NetworkMacro
        return NetworkMacro().run()
    if mode == "2":
        from digimonup.app.explore import ExploreEngine
        from digimonup.base.settings import load_explore_config
        engine = ExploreEngine(load_explore_config())
        try:
            engine.run()
        except KeyboardInterrupt:
            engine.stop()
            print("\nCtrl+C 로 중단했습니다.")
        return 0
    print("잘못된 번호입니다.")
    return 1


def main(argv: list[str]) -> int:
    enable_dpi_awareness()
    args = [a for a in argv[1:] if a]

    # 버전 확인은 중복 실행 검사보다 먼저 한다. 매크로가 이미 돌고 있어도
    # 버전은 물어볼 수 있어야 한다.
    if args and args[0] in ("--version", "-V"):
        print(version_line())
        return 0

    # 두 개가 동시에 돌면 같은 게임 창에 서로 클릭을 날려 둘 다 망가진다.
    if not single_instance.acquire():
        if single_instance.focus_existing():
            return 0
        print("digimonUp 매크로가 이미 실행 중입니다.")
        try:
            import tkinter.messagebox as mb
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            mb.showinfo("digimonUp", "매크로가 이미 실행 중입니다.")
            root.destroy()
        except Exception:
            pass
        return 0

    if args and args[0] in ("1", "2"):
        return run_console(args[0])

    if args and args[0] in ("--console", "-c"):
        print(MENU)
        choice = input("번호 선택 (1/2, q=종료): ").strip().lower()
        if choice in ("q", "quit", "exit", ""):
            return 0
        return run_console(choice)

    from digimonup.app.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
