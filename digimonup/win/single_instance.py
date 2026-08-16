"""중복 실행 방지.

바로가기를 두 번 누르면 매크로가 두 개 떠서 같은 게임 창에 서로 클릭을 날린다.
그러면 이동 확인이 계속 어긋나고 '이동할 수 없습니다'만 반복된다.

Windows 이름 있는 뮤텍스로 막는다. 프로세스가 죽으면 OS 가 알아서 풀어 주므로
잠금 파일처럼 찌꺼기가 남지 않는다.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

MUTEX_NAME = "Global\\digimonUp_macro_single_instance"
GUI_TITLE = "digimonUp 매크로"

_ERROR_ALREADY_EXISTS = 183

_kernel32 = ctypes.windll.kernel32
_user32 = ctypes.windll.user32

_handle = None      # 프로세스가 살아 있는 동안 붙잡고 있어야 한다


def acquire(name: str = MUTEX_NAME) -> bool:
    """이 프로세스가 첫 번째면 True, 이미 다른 인스턴스가 있으면 False."""
    global _handle
    _kernel32.SetLastError(0)
    handle = _kernel32.CreateMutexW(None, wintypes.BOOL(True), name)
    if not handle:
        return True                     # 뮤텍스를 못 만들면 막지 않는다
    if _kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
        _kernel32.CloseHandle(handle)
        return False
    _handle = handle                    # 참조를 유지해야 잠금이 풀리지 않는다
    return True


def focus_existing(title: str = GUI_TITLE) -> bool:
    """이미 떠 있는 창을 앞으로 가져온다. 찾으면 True."""
    hwnd = _user32.FindWindowW(None, title)
    if not hwnd:
        return False
    if _user32.IsIconic(hwnd):
        _user32.ShowWindow(hwnd, 9)     # SW_RESTORE
    _user32.SetForegroundWindow(hwnd)
    return True
