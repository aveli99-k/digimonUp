"""MuMuPlayer 창을 찾아 '고정'하고, 그 창만 캡처/클릭하는 모듈.

핵심 원칙
  - 고정 화면 좌표를 쓰지 않는다. 모든 좌표는 **고정된 HWND 의 클라이언트 좌표**다.
  - 창 제목만 믿지 않는다. 게임 화면은 계속 바뀌므로 두 가지를 함께 확인한다.
        1) 상단에 고정된 게임 탭 이미지가 보이는가
        2) 화면 안에 5x5 게임판 격자 테두리가 있는가
  - 한 번 고정한 HWND 는 실행 중에 바꾸지 않는다.

MuMuPlayer 12 의 창 구조 (실측)
    Qt5156QWindowIcon  "Android Device"      <- 최상위 창
      +- Qt5156QWindowIcon  "MuMuNxDevice"
      +- nemuwin           "nemudisplay"     <- 실제 안드로이드 화면이 그려지는 자식 창
    Qt5156QWindowIcon  "MuMuPlayer"          <- 멀티 인스턴스 관리창(게임 아님)
      +- MuMuNativeWindow "MuMuThumbnailWindow"

게임 화면은 nemudisplay 자식 창에만 그려지므로, 캡처/클릭 대상은 이 자식 창이다.
상단 툴바(키보드/음량/뒤로가기 아이콘)가 포함되지 않아 좌표 계산이 깔끔해진다.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui

# 안드로이드 화면이 그려지는 자식 창의 클래스/제목
RENDER_CLASSES = ("nemuwin",)
RENDER_TITLES = ("nemudisplay",)
# 최상위 후보 창의 클래스 접두어 (Qt 버전에 따라 숫자가 달라진다: Qt5156QWindowIcon 등)
TOPLEVEL_CLASS_PREFIX = "Qt5"
TOPLEVEL_CLASS_SUFFIX = "QWindowIcon"
# 게임이 아닌 관리창은 이 자식 창을 갖는다 -> 후보에서 제외
MANAGER_CHILD_CLASSES = ("MuMuNativeWindow",)

PW_RENDERFULLCONTENT = 0x00000002

_user32 = ctypes.windll.user32


def enable_dpi_awareness() -> None:
    """디스플레이 배율(125%, 150% 등)이 켜져 있어도 좌표가 어긋나지 않게 한다.

    이걸 안 하면 GetWindowRect 가 논리 픽셀을 돌려주고, PrintWindow 로 뜬 비트맵은
    물리 픽셀이라 캡처 이미지가 어긋난다(실측: 125% 배율에서 567px vs 709px).
    프로세스 시작 직후 반드시 한 번 호출해야 한다.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            _user32.SetProcessDPIAware()
        except Exception:
            pass


# --------------------------------------------------------------------------
# 후보 창 열거
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    """LDPlayer/MuMuPlayer 후보 창 하나에 대한 평가 결과."""
    hwnd: int                 # 캡처/클릭 대상 (렌더 자식 창)
    top_hwnd: int             # 최상위 창 (포커스 대상)
    title: str
    cls: str
    width: int
    height: int
    reasons: list[str] = field(default_factory=list)
    tab_score: float = 0.0    # 상단 고정 탭 이미지 유사도
    board_score: float = 0.0  # 5x5 격자 검출 신뢰도
    ok: bool = False

    @property
    def score(self) -> float:
        return self.tab_score + self.board_score

    def describe(self) -> str:
        mark = "O" if self.ok else "X"
        return (f"[{mark}] hwnd=0x{self.hwnd:X} {self.width}x{self.height} "
                f"'{self.title}' 탭={self.tab_score:.2f} 격자={self.board_score:.2f} "
                f"| {', '.join(self.reasons) if self.reasons else '-'}")


def _child_windows(hwnd: int) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []

    def cb(child, _):
        try:
            out.append((child, win32gui.GetClassName(child), win32gui.GetWindowText(child)))
        except Exception:
            pass
        return True

    try:
        win32gui.EnumChildWindows(hwnd, cb, None)
    except Exception:
        pass
    return out


def _find_render_child(top_hwnd: int) -> int | None:
    """최상위 창 아래에서 안드로이드 화면이 그려지는 자식 창을 찾는다."""
    best = None
    best_area = 0
    for child, cls, title in _child_windows(top_hwnd):
        if cls not in RENDER_CLASSES and title not in RENDER_TITLES:
            continue
        try:
            l, t, r, b = win32gui.GetClientRect(child)
        except Exception:
            continue
        area = (r - l) * (b - t)
        if area > best_area:
            best_area, best = area, child
    return best


def enumerate_candidates(min_size: int = 200) -> list[Candidate]:
    """화면에 떠 있는 MuMuPlayer 후보 창을 모두 찾는다 (아직 검증 전)."""
    found: list[Candidate] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            cls = win32gui.GetClassName(hwnd)
            title = win32gui.GetWindowText(hwnd)
        except Exception:
            return True

        reasons: list[str] = []
        if not (cls.startswith(TOPLEVEL_CLASS_PREFIX) and cls.endswith(TOPLEVEL_CLASS_SUFFIX)):
            return True

        children = _child_windows(hwnd)
        child_classes = {c for _, c, _ in children}
        if child_classes & set(MANAGER_CHILD_CLASSES):
            # 멀티 인스턴스 관리창(썸네일 창). 게임 화면이 아니다.
            return True

        render = _find_render_child(hwnd)
        if render is None:
            return True

        l, t, r, b = win32gui.GetClientRect(render)
        w, h = r - l, b - t
        if w < min_size or h < min_size:
            return True

        reasons.append(f"렌더창 발견({win32gui.GetClassName(render)})")
        if h > w:
            reasons.append("세로 화면")
        found.append(Candidate(hwnd=render, top_hwnd=hwnd, title=title, cls=cls,
                               width=w, height=h, reasons=reasons))
        return True

    win32gui.EnumWindows(cb, None)
    found.sort(key=lambda c: c.width * c.height, reverse=True)
    return found


# --------------------------------------------------------------------------
# 캡처
# --------------------------------------------------------------------------

def capture_client(hwnd: int) -> np.ndarray | None:
    """지정한 HWND 의 **클라이언트 영역만** BGR 배열로 캡처한다.

    PrintWindow + PW_RENDERFULLCONTENT 를 쓰므로 창이 다른 창에 가려져 있거나
    화면 밖으로 나가 있어도 내용을 받아올 수 있다(실측 확인). 전체 모니터를
    캡처하지 않으므로 창이 이동해도 좌표 기준이 흔들리지 않는다.
    """
    if not win32gui.IsWindow(hwnd):
        return None
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
    except Exception:
        return None
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        return None

    hdc = None
    src = mem = bmp = None
    try:
        hdc = win32gui.GetWindowDC(hwnd)
        src = win32ui.CreateDCFromHandle(hdc)
        mem = src.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(src, w, h)
        mem.SelectObject(bmp)

        ok = _user32.PrintWindow(hwnd, mem.GetSafeHdc(), PW_RENDERFULLCONTENT)
        if not ok:
            _user32.PrintWindow(hwnd, mem.GetSafeHdc(), 0)

        raw = bmp.GetBitmapBits(True)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None
    finally:
        try:
            if bmp is not None:
                win32gui.DeleteObject(bmp.GetHandle())
            if mem is not None:
                mem.DeleteDC()
            if src is not None:
                src.DeleteDC()
            if hdc is not None:
                win32gui.ReleaseDC(hwnd, hdc)
        except Exception:
            pass


# --------------------------------------------------------------------------
# 고정된 창 핸들
# --------------------------------------------------------------------------

class MuMuWindow:
    """검증을 통과해 '고정된' 하나의 창. 실행 중 대상이 바뀌지 않는다."""

    def __init__(self, hwnd: int, top_hwnd: int, title: str = ""):
        self.hwnd = hwnd
        self.top_hwnd = top_hwnd
        self.title = title

    # -- 상태 -------------------------------------------------------------
    def is_valid(self) -> bool:
        return bool(win32gui.IsWindow(self.hwnd) and win32gui.IsWindow(self.top_hwnd))

    def client_size(self) -> tuple[int, int]:
        l, t, r, b = win32gui.GetClientRect(self.hwnd)
        return r - l, b - t

    # -- 캡처 -------------------------------------------------------------
    def capture(self) -> np.ndarray | None:
        return capture_client(self.hwnd)

    # -- 좌표 -------------------------------------------------------------
    def client_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """클라이언트 좌표 -> 화면 좌표. 창이 이동해도 매번 새로 물어본다."""
        return win32gui.ClientToScreen(self.hwnd, (int(x), int(y)))

    # -- 포커스 -----------------------------------------------------------
    def focus(self, retries: int = 3) -> bool:
        """대상 창을 앞으로 가져온다. 실패해도 몇 번 재시도한다."""
        for _ in range(retries):
            try:
                if win32gui.IsIconic(self.top_hwnd):
                    win32gui.ShowWindow(self.top_hwnd, win32con.SW_RESTORE)
                if win32gui.GetForegroundWindow() == self.top_hwnd:
                    return True
                # SetForegroundWindow 는 다른 프로세스가 포그라운드일 때 거부될 수 있어
                # 입력 스레드를 잠깐 붙였다가 뗀다.
                fg = win32gui.GetForegroundWindow()
                cur = win32api.GetCurrentThreadId()
                tgt = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
                attached = False
                if tgt and tgt != cur:
                    attached = bool(_user32.AttachThreadInput(cur, tgt, True))
                try:
                    win32gui.SetForegroundWindow(self.top_hwnd)
                finally:
                    if attached:
                        _user32.AttachThreadInput(cur, tgt, False)
                if win32gui.GetForegroundWindow() == self.top_hwnd:
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        return win32gui.GetForegroundWindow() == self.top_hwnd

    # -- 클릭 -------------------------------------------------------------
    def click_client(self, x: int, y: int, move_duration: float = 0.06,
                     require_focus: bool = True) -> tuple[int, int] | None:
        """클라이언트 좌표 (x, y) 를 클릭한다. 실제로 클릭한 화면 좌표를 돌려준다.

        클릭 직전에 HWND 유효성을 확인하고, 포커스를 재시도한 다음 입력한다.
        """
        if not self.is_valid():
            return None
        w, h = self.client_size()
        if not (0 <= x < w and 0 <= y < h):
            return None
        if require_focus:
            self.focus()
        sx, sy = self.client_to_screen(x, y)

        # pyautogui 를 쓰지 않고 SendInput 을 직접 쓴다(FAILSAFE/PAUSE 영향 없음).
        # 커서가 실제로 그 자리에 갔는지 확인한 뒤에 누른다.
        got = move_and_verify(sx, sy, move_duration)
        if abs(got[0] - sx) > 2 or abs(got[1] - sy) > 2:
            # 여기까지 어긋나면 클릭하지 않는다. 엉뚱한 곳을 누르는 것보다 낫다.
            return None
        _left_click()
        return sx, sy


# --------------------------------------------------------------------------
# 마우스 입력 (SendInput)
# --------------------------------------------------------------------------

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
# ABSOLUTE 만 쓰면 좌표를 '주 모니터' 기준으로 해석한다. 아래 플래그를 함께 줘야
# 가상 데스크톱(모든 모니터) 기준이 된다. 이게 빠지면 보조 모니터에서 클릭이
# 엉뚱한 곳으로 간다. (실측: y=636 을 의도했는데 커서가 y=301 로 갔다.
#  주 모니터 높이 1440 / 가상 데스크톱 높이 3040 의 비율만큼 어긋난 것)
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_ABS_MOVE = _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK


def _send_mouse(flags: int, x: int = 0, y: int = 0) -> None:
    inp = _INPUT(type=_INPUT_MOUSE,
                 mi=_MOUSEINPUT(dx=x, dy=y, mouseData=0, dwFlags=flags,
                                time=0, dwExtraInfo=None))
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _abs_coords(x: int, y: int) -> tuple[int, int]:
    """가상 데스크톱 전체 기준 절대좌표(0~65535)로 변환. 멀티 모니터 대응."""
    vx = _user32.GetSystemMetrics(76)
    vy = _user32.GetSystemMetrics(77)
    vw = _user32.GetSystemMetrics(78)
    vh = _user32.GetSystemMetrics(79)
    ax = int((x - vx) * 65535 / max(1, vw - 1))
    ay = int((y - vy) * 65535 / max(1, vh - 1))
    return ax, ay


def _move_mouse(x: int, y: int, duration: float = 0.0) -> None:
    if duration > 0:
        cx, cy = win32api.GetCursorPos()
        steps = max(2, int(duration / 0.012))
        for i in range(1, steps + 1):
            ix = int(cx + (x - cx) * i / steps)
            iy = int(cy + (y - cy) * i / steps)
            ax, ay = _abs_coords(ix, iy)
            _send_mouse(_ABS_MOVE, ax, ay)
            time.sleep(duration / steps)
    else:
        ax, ay = _abs_coords(x, y)
        _send_mouse(_ABS_MOVE, ax, ay)


def move_and_verify(x: int, y: int, duration: float = 0.0,
                    tolerance: int = 2) -> tuple[int, int]:
    """커서를 옮기고 **실제로 그 자리에 갔는지 확인**해서 실제 좌표를 돌려준다.

    좌표 변환이 조금이라도 어긋나면 엉뚱한 곳을 클릭하게 되므로, 클릭 전에
    한 번 확인하고 어긋나면 다시 시도한다.
    """
    _move_mouse(x, y, duration)
    for _ in range(3):
        got = win32api.GetCursorPos()
        if abs(got[0] - x) <= tolerance and abs(got[1] - y) <= tolerance:
            return got
        _move_mouse(x, y, 0.0)
        time.sleep(0.01)
    return win32api.GetCursorPos()


def _left_click(press_sec: float = 0.06) -> None:
    _send_mouse(_MOUSEEVENTF_LEFTDOWN)
    time.sleep(press_sec)
    _send_mouse(_MOUSEEVENTF_LEFTUP)
