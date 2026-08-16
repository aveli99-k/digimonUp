"""앱플레이어(안드로이드 에뮬레이터) 창을 찾아 '고정'하고, 그 창만 캡처/클릭한다.

핵심 원칙
  - 고정 화면 좌표를 쓰지 않는다. 모든 좌표는 **고정된 HWND 의 클라이언트 좌표**다.
  - 창 제목이나 클래스 이름만 믿지 않는다. 최종 판정은 **화면 내용**으로 한다.
        1) 상단에 고정된 게임 탭 이미지가 보이는가 (템플릿을 넣은 경우)
        2) 화면 안에 5x5 게임판 격자 테두리가 있는가
  - 한 번 고정한 HWND 는 실행 중에 바꾸지 않는다.

왜 클래스 이름에 의존하지 않는가
    처음에는 MuMuPlayer 의 창 클래스(nemuwin / Qt5...QWindowIcon)를 코드에 박아 두고
    그 창만 찾았다. 그러면 LDPlayer, NoxPlayer, BlueStacks 처럼 클래스 이름이 다른
    앱플레이어에서는 후보가 아예 0개가 되어 "창을 찾지 못했습니다"만 뜬다.

    하지만 이 매크로는 어차피 **화면에 5x5 게임판이 있는지**를 보고 창을 확정한다
    (explore.pick_window). 그 검사가 진짜 판정이고 클래스 이름은 후보를 좁히는
    힌트일 뿐이다. 그래서 아래처럼 바꿨다.

      - 아는 앱플레이어는 EMULATOR_PROFILES 로 정확히 집어내고(빠르고 확실하다)
      - 모르는 앱플레이어는 '적당한 크기의 보이는 창'을 전부 후보로 올린 뒤
        게임판 검사로 거른다

    덕분에 새 앱플레이어가 나와도 코드를 고치지 않고 동작한다.

앱플레이어 창 구조 (MuMuPlayer 12 실측)
    Qt5156QWindowIcon  "Android Device"      <- 최상위 창
      +- Qt5156QWindowIcon  "MuMuNxDevice"
      +- nemuwin           "nemudisplay"     <- 실제 안드로이드 화면이 그려지는 자식 창
    Qt5156QWindowIcon  "MuMuPlayer"          <- 멀티 인스턴스 관리창(게임 아님)
      +- MuMuNativeWindow "MuMuThumbnailWindow"

안드로이드 화면은 자식 창에만 그려지는 경우가 많다. 그 자식 창을 잡으면 상단
툴바(키보드/음량/뒤로가기)가 캡처에 안 들어와 좌표 계산이 깔끔해진다.
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass, field

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


@dataclass(frozen=True)
class EmulatorProfile:
    """아는 앱플레이어 하나의 창 생김새.

    전부 '힌트'다. 여기 없는 앱플레이어도 generic 경로로 동작한다.
    """
    name: str
    render_classes: tuple[str, ...] = ()     # 안드로이드 화면이 그려지는 자식 창 클래스
    render_titles: tuple[str, ...] = ()      # 그 자식 창의 제목
    title_hints: tuple[str, ...] = ()        # 최상위 창 제목에 들어가는 말
    exclude_child_classes: tuple[str, ...] = ()   # 이 자식이 있으면 게임 창이 아니다


# 실측으로 확인한 것은 MuMuPlayer 12 뿐이다. 나머지는 공개된 창 클래스 정보를 적어
# 두었고, 틀리더라도 generic 경로로 잡히므로 동작에는 지장이 없다.
EMULATOR_PROFILES: tuple[EmulatorProfile, ...] = (
    EmulatorProfile(
        name="MuMuPlayer",
        render_classes=("nemuwin",),
        render_titles=("nemudisplay",),
        title_hints=("MuMu", "Android Device"),
        # 멀티 인스턴스 관리창(썸네일 창)은 게임 화면이 아니다.
        exclude_child_classes=("MuMuNativeWindow",),
    ),
    EmulatorProfile(
        name="LDPlayer",
        render_classes=("RenderWindow", "subWin", "TheRender"),
        title_hints=("LDPlayer", "뮤뮤", "雷电"),
    ),
    EmulatorProfile(
        name="NoxPlayer",
        render_classes=("ScreenBoardClass", "subWin", "SDL_app"),
        title_hints=("Nox", "夜神"),
    ),
    EmulatorProfile(
        name="BlueStacks",
        render_classes=("BlueStacksApp", "plrNativeInputWindowClass"),
        title_hints=("BlueStacks",),
    ),
    EmulatorProfile(
        name="MEmu",
        render_classes=("SDL_app",),
        title_hints=("MEmu",),
    ),
    EmulatorProfile(
        name="Google Play Games",
        title_hints=("Google Play Games",),
    ),
)

# 아래 두 개는 예전 이름이다. 지금은 프로필 표에서 뽑아 쓴다.

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
    """앱플레이어 후보 창 하나에 대한 평가 결과."""
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
    emulator: str = ""        # 알아본 앱플레이어 이름 (모르면 빈 문자열)

    @property
    def score(self) -> float:
        # 아는 앱플레이어를 살짝 우대한다. 게임판 점수가 같을 때만 갈린다.
        return self.tab_score + self.board_score + (0.01 if self.emulator else 0.0)

    def describe(self) -> str:
        mark = "O" if self.ok else "X"
        who = self.emulator or "알 수 없는 창"
        return (f"[{mark}] hwnd=0x{self.hwnd:X} {self.width}x{self.height} "
                f"[{who}] '{self.title}' 탭={self.tab_score:.2f} "
                f"격자={self.board_score:.2f} "
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


def _pid_of(hwnd: int) -> int:
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _client_size(hwnd: int) -> tuple[int, int]:
    try:
        x0, y0, x1, y1 = win32gui.GetClientRect(hwnd)
        return x1 - x0, y1 - y0
    except Exception:
        return 0, 0


# 앱플레이어일 수 없는 창들. 바탕화면·작업표시줄·IME 같은 셸 창이다.
SHELL_CLASSES = frozenset({
    "Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd",
    "Windows.UI.Core.CoreWindow", "MSCTFIME UI", "IME", "tooltips_class32",
    "TaskListThumbnailWnd", "ForegroundStaging", "MultitaskingViewFrame",
})


def _match_profile(title: str, children: list[tuple[int, str, str]]
                   ) -> tuple[EmulatorProfile | None, bool]:
    """어떤 앱플레이어인지 알아본다.

    반환: (프로필, 확실한가)

    **제목만 맞는 것은 확실한 근거가 아니다.** 실측: 이 저장소 페이지를 띄운
    브라우저 창의 제목에 'MuMu' 가 들어 있어서 MuMuPlayer 로 잡혔다. 그러면
    엉뚱한 창이 1순위 후보로 올라가 캡처·격자 검사를 헛돈다.

    그래서 자식 창의 클래스/제목이 맞을 때(구조가 맞을 때)만 '확실'로 보고,
    제목 힌트만 맞으면 후보로는 올리되 앱플레이어로 단정하지 않는다.
    """
    child_classes = {c for _, c, _ in children}
    child_titles = {t for _, _, t in children}
    for prof in EMULATOR_PROFILES:
        if child_classes & set(prof.render_classes):
            return prof, True
        if child_titles & set(prof.render_titles):
            return prof, True
    for prof in EMULATOR_PROFILES:
        if any(hint.lower() in title.lower() for hint in prof.title_hints if hint):
            return prof, False
    return None, False


def _find_render_child(top_hwnd: int, prof: EmulatorProfile | None,
                       children: list[tuple[int, str, str]],
                       min_size: int) -> int | None:
    """안드로이드 화면이 그려지는 자식 창을 고른다.

    1) 아는 앱플레이어면 그 프로필의 렌더 클래스/제목으로 정확히 집는다.
    2) 모르면 **가장 큰 자식 창**을 쓴다. 앱플레이어는 대체로 화면 전체를 채우는
       렌더 자식 창을 하나 갖는다.
    3) 쓸 만한 자식이 없으면 None 을 돌려준다. 그때는 최상위 창을 직접 캡처한다
       (자식 창 없이 최상위에 바로 그리는 앱플레이어도 있다).
    """
    named: list[tuple[int, int]] = []      # (면적, hwnd)
    biggest: list[tuple[int, int]] = []
    for child, cls, title in children:
        w, h = _client_size(child)
        if w < min_size or h < min_size:
            continue
        biggest.append((w * h, child))
        if prof and (cls in prof.render_classes or title in prof.render_titles):
            named.append((w * h, child))
    pool = named or biggest
    if not pool:
        return None
    return max(pool)[1]


def enumerate_candidates(min_size: int = 200, title_hint: str = "",
                         max_candidates: int = 16) -> list[Candidate]:
    """앱플레이어로 보이는 창을 모두 찾는다 (아직 게임판 검증 전).

    아는 앱플레이어든 모르는 앱플레이어든 일단 후보로 올린다. 진짜 판정은
    explore.pick_window 가 화면 안의 5x5 게임판으로 한다.

    title_hint 를 주면 창 제목에 그 말이 든 창만 본다. 앱플레이어를 여러 개
    띄워 두고 특정 창만 쓰고 싶을 때 config.json 으로 지정한다.
    """
    found: list[Candidate] = []

    own_pid = os.getpid()

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        if win32gui.IsIconic(hwnd):
            # 최소화된 창은 PrintWindow 로도 제대로 안 잡힌다.
            return True
        try:
            cls = win32gui.GetClassName(hwnd)
            title = win32gui.GetWindowText(hwnd)
        except Exception:
            return True

        if cls in SHELL_CLASSES:
            return True
        if _pid_of(hwnd) == own_pid:
            # 매크로 자신의 GUI 창. 자기를 캡처할 이유가 없다.
            return True
        if title_hint and title_hint.lower() not in title.lower():
            return True

        # 최상위 창 자체가 너무 작으면 앱플레이어일 수 없다.
        tw, th = _client_size(hwnd)
        if tw < min_size or th < min_size:
            return True

        children = _child_windows(hwnd)
        prof, sure = _match_profile(title, children)

        if prof and sure and prof.exclude_child_classes:
            if {c for _, c, _ in children} & set(prof.exclude_child_classes):
                # 멀티 인스턴스 관리창(썸네일 창). 게임 화면이 아니다.
                return True

        reasons: list[str] = []
        render = _find_render_child(hwnd, prof if sure else None, children, min_size)
        if render is None:
            # 자식 창 없이 최상위에 바로 그리는 경우. 최상위를 그대로 쓴다.
            render, w, h = hwnd, tw, th
            reasons.append("렌더 자식 창이 없어 최상위 창을 직접 캡처")
        else:
            w, h = _client_size(render)
            reasons.append(f"렌더창({win32gui.GetClassName(render)})")

        if sure:
            reasons.append(f"{prof.name} 로 인식")
        elif prof:
            reasons.append(f"제목만 {prof.name} 같음 (게임판 검사로 판정)")
        else:
            reasons.append("모르는 창 (게임판 검사로 판정)")
        if h > w:
            reasons.append("세로 화면")

        found.append(Candidate(hwnd=render, top_hwnd=hwnd, title=title, cls=cls,
                               width=w, height=h, reasons=reasons,
                               emulator=prof.name if sure else ""))
        return True

    win32gui.EnumWindows(cb, None)
    # 확실한 앱플레이어 먼저, 그다음 큰 창 먼저. 모르는 창까지 전부 캡처해서
    # 검사하면 느리므로 상위 몇 개만 넘긴다.
    found.sort(key=lambda c: (bool(c.emulator), c.width * c.height), reverse=True)
    return found[:max_candidates]


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
        x0, y0, x1, y1 = win32gui.GetClientRect(hwnd)
    except Exception:
        return None
    w, h = x1 - x0, y1 - y0
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

class EmulatorWindow:
    """검증을 통과해 '고정된' 하나의 창. 실행 중 대상이 바뀌지 않는다."""

    def __init__(self, hwnd: int, top_hwnd: int, title: str = ""):
        self.hwnd = hwnd
        self.top_hwnd = top_hwnd
        self.title = title

    # -- 상태 -------------------------------------------------------------
    def is_valid(self) -> bool:
        return bool(win32gui.IsWindow(self.hwnd) and win32gui.IsWindow(self.top_hwnd))

    def client_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = win32gui.GetClientRect(self.hwnd)
        return x1 - x0, y1 - y0

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
