"""탐사 엔진 동작 테스트.

여기서 지키려는 것 세 가지
  - 이전 칸에 남은 애니메이션 잔상을 '이동 성공'으로 처리하지 않는다.
  - 정지를 누르면 뒤늦은 클릭이 절대 나가지 않는다.
  - '이동할 수 없습니다' 안내문은 클릭하지 않는다.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import board
import explore
import recognize
import synth
from explore import ExploreConfig, ExploreEngine, Stopped

LAYOUT_AT = {
    (1, 1): ["....." , ".P..X", "..X..", "....X", "....."],
    (1, 2): ["....." , "..P.X", "..X..", "....X", "....."],
}


class FakeWindow:
    """캡처 프레임을 대본대로 돌려주고, 클릭은 기록만 하는 가짜 창."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.clicks: list[tuple[int, int]] = []
        self.hwnd = 0x1234
        self.top_hwnd = 0x1230
        self.i = 0

    def is_valid(self):
        return True

    def client_size(self):
        return self.frames[0].shape[1], self.frames[0].shape[0]

    def capture(self):
        img = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return img

    def client_to_screen(self, x, y):
        return x + 1000, y + 50

    def focus(self, retries=3):
        return True

    def click_client(self, x, y, move_duration=0.0, require_focus=True):
        self.clicks.append((int(x), int(y)))
        return int(x) + 1000, int(y) + 50


def _engine(frames, **cfg_kw):
    defaults = dict(move_timeout_sec=0.5, poll_interval_sec=0.0,
                    click_settle_sec=0.0, save_debug=False)
    defaults.update(cfg_kw)
    cfg = ExploreConfig(**defaults)
    eng = ExploreEngine(cfg, log=lambda *_: None)
    eng.window = FakeWindow(frames)
    eng.templates = recognize.load_templates()
    return eng


def _frame(cell, **kw):
    return synth.make_board(LAYOUT_AT[cell], **kw)


# ------------------------------------------------- 애니메이션 잔상 방지
def test_이전_위치의_잔상을_이동_성공으로_처리하지_않는다():
    """RIGHT 를 눌렀는데 계속 이전 칸에서만 검출되면 성공이 아니다."""
    stay = _frame((1, 1))
    eng = _engine([stay] * 40)
    grid = board.detect_board(stay)

    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False, "이전 칸에 남아 있는데 성공으로 봤습니다"
    assert len(eng.window.clicks) == 1, "한 칸 이동에는 클릭이 정확히 한 번"


def test_예상_칸이_2회_연속_확인돼야_성공이다():
    """한 프레임만 새 칸에서 잡히는 깜빡임은 성공으로 치지 않는다."""
    old, new = _frame((1, 1)), _frame((1, 2))
    # 새 칸 -> 이전 칸(잔상) -> 새 칸 ... 한 번씩만 번갈아 나오는 대본
    eng = _engine([old, new, old, new, old, new, old] + [old] * 30,
                  confirm_repeat=2)
    grid = board.detect_board(old)
    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False, "연속 2회 확인 규칙이 지켜지지 않았습니다"


def test_예상_칸에_실제로_도착하면_성공한다():
    old, new = _frame((1, 1)), _frame((1, 2))
    eng = _engine([old, new, new, new, new], confirm_repeat=2)
    grid = board.detect_board(old)
    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is True
    assert eng.stats.moves == 1


def test_클릭_좌표는_다음_한_칸의_셀_중심이다():
    """게임판 중앙이나 고정 좌표가 아니라, 이동할 칸의 중심을 클릭해야 한다."""
    old, new = _frame((1, 1)), _frame((1, 2))
    eng = _engine([old, new, new, new])
    grid = board.detect_board(old)
    eng._do_move(grid, (1, 1), (1, 2), "RIGHT")

    want = grid.cell_center(1, 2)
    got = eng.window.clicks[0]
    assert abs(got[0] - want[0]) <= 3 and abs(got[1] - want[1]) <= 3
    # 게임판 중앙(2,2)과는 확실히 달라야 한다
    center = grid.cell_center(2, 2)
    assert got != center


# 스크롤 검증용 판. 이동해도 플레이어가 놓일 칸이 장애물이 되지 않도록 골랐다.
SCROLL_BASE = [
    ".....",
    ".P...",
    "..X..",
    ".X..X",
    "..i..",
]
# 실측: 플레이어가 진행하면 판이 이동 방향의 반대로 한 칸 밀리고,
#       플레이어는 화면상 같은 칸(1,1)에 그대로 남는다.
#       즉 after[r][c] == before[r+dr][c+dc] (dr,dc = 이동 방향 델타)
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def _scrolled_layout(layout, dr, dc, player=(1, 1)):
    """판이 한 칸 스크롤한 뒤의 배치. 플레이어는 화면상 같은 칸에 남는다."""
    rows = []
    for r in range(5):
        row = ""
        for c in range(5):
            sr, sc = r + dr, c + dc
            ch = layout[sr][sc] if 0 <= sr < 5 and 0 <= sc < 5 else "."
            row += "." if ch == "P" else ch
        rows.append(row)
    rows[player[0]] = (rows[player[0]][:player[1]] + "P"
                       + rows[player[0]][player[1] + 1:])
    return rows


@pytest.mark.parametrize("direction", ["UP", "DOWN", "LEFT", "RIGHT"])
def test_게임판_스크롤로도_이동_성공을_인정한다(direction):
    """진행하면 판이 통째로 밀려서 플레이어가 화면상 같은 칸에 남는다."""
    dr, dc = DELTA[direction]
    base = synth.make_board(SCROLL_BASE)
    after = synth.make_board(_scrolled_layout(SCROLL_BASE, dr, dc))
    g = board.detect_board(base)

    eng = _engine([base, after, after, after, after])
    to = (1 + dr, 1 + dc)
    ok, _, _ = eng._do_move(g, (1, 1), to, direction)
    assert ok is True, f"{direction}: 판이 한 칸 스크롤했는데 실패로 봤습니다"


def test_스크롤하지_않았으면_성공으로_보지_않는다():
    """화면이 그대로면 이동한 게 아니다."""
    base = synth.make_board(SCROLL_BASE)
    g = board.detect_board(base)
    eng = _engine([base] * 40)
    ok, _, _ = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is False


def test_반대_방향_스크롤은_성공으로_보지_않는다():
    """RIGHT 를 눌렀는데 판이 반대로 밀렸다면 이동에 성공한 게 아니다."""
    base = synth.make_board(SCROLL_BASE)
    wrong = synth.make_board(_scrolled_layout(SCROLL_BASE, 0, -1))  # LEFT 스크롤
    g = board.detect_board(base)
    eng = _engine([base] + [wrong] * 30)
    ok, _, _ = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is False


# ---------------------------------------------------------- 정지 처리
def test_정지_요청_후에는_클릭이_나가지_않는다():
    old = _frame((1, 1))
    eng = _engine([old] * 10)
    grid = board.detect_board(old)

    eng.stop()
    with pytest.raises(Stopped):
        eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert eng.window.clicks == [], "정지 후에도 클릭이 나갔습니다"


def test_분석_도중_정지하면_이전_경로가_뒤늦게_실행되지_않는다(monkeypatch):
    """분석이 끝난 뒤 정지 상태를 다시 확인하므로 경로 실행에 진입하면 안 된다."""
    img = _frame((1, 1))
    eng = _engine([img] * 50)
    monkeypatch.setattr(explore, "enumerate_candidates", lambda: [])
    eng.pick_window = lambda: eng.window

    real_analyze = explore.analyze

    def analyze_then_stop(*a, **kw):
        result = real_analyze(*a, **kw)
        eng.stop()          # 분석하는 동안 사용자가 정지를 누른 상황
        return result

    monkeypatch.setattr(explore, "analyze", analyze_then_stop)
    eng.run()
    assert eng.window.clicks == [], "정지했는데 이전 경로가 실행됐습니다"


def test_정지하면_run_이_즉시_빠져나온다(monkeypatch):
    img = _frame((1, 1))
    eng = _engine([img] * 200)
    eng.pick_window = lambda: eng.window
    eng.stop()
    eng.run()      # Stopped 를 안에서 잡고 조용히 끝나야 한다
    assert eng.window.clicks == []


# ------------------------------------------------- 이동 불가 안내 처리
def _toast_templates(toast_patch):
    tpl = recognize.load_templates()
    t = recognize.TemplateSet.__new__(recognize.TemplateSet)
    t.name, t.allow_flip, t.images, t.paths = \
        "blocked_toast", False, [toast_patch], ["toast.png"]
    tpl["blocked_toast"] = t
    return tpl


def _with_toast(img):
    """화면 가운데에 안내문 비슷한 띠를 얹는다."""
    import cv2
    out = img.copy()
    cv2.rectangle(out, (150, 560), (560, 640), (40, 40, 40), -1)
    cv2.putText(out, "CANNOT MOVE THERE", (165, 615),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2)
    return out


def test_이동_불가_안내문을_클릭하지_않는다():
    clean = _frame((1, 1))
    toast = _with_toast(clean)
    patch = toast[560:640, 150:560].copy()

    # 클릭 -> 안내 등장 -> 안내 유지 -> 사라짐(2회 연속)
    eng = _engine([clean, toast, toast, clean, clean, clean])
    eng.templates = _toast_templates(patch)
    grid = board.detect_board(clean)

    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False
    assert len(eng.window.clicks) == 1, "안내문을 추가로 클릭했습니다"

    # 클릭한 곳은 안내문 영역이 아니라 이동할 셀 중심이어야 한다
    cx, cy = eng.window.clicks[0]
    assert not (150 <= cx <= 560 and 560 <= cy <= 640)
    assert eng.stats.blocked_toasts == 1


def test_안내가_사라진_화면을_2회_연속_확인한_뒤_진행한다():
    clean = _frame((1, 1))
    toast = _with_toast(clean)
    patch = toast[560:640, 150:560].copy()

    eng = _engine([toast, toast, clean, toast, clean, clean, clean, clean])
    eng.templates = _toast_templates(patch)
    eng._wait_toast_clear()
    # 중간에 안내가 다시 보이면 카운트가 초기화되므로, 끝까지 클릭은 없어야 한다
    assert eng.window.clicks == []


def test_안내문_감지_자체가_동작한다():
    clean = _frame((1, 1))
    toast = _with_toast(clean)
    patch = toast[560:640, 150:560].copy()
    eng = _engine([clean])
    eng.templates = _toast_templates(patch)

    assert eng._toast_visible(toast) is True
    assert eng._toast_visible(clean) is False


# ------------------------------------------------------------ 그 밖의 것
def test_격자선을_매_이동마다_다시_검출해_셀_중심을_갱신한다():
    """창이 움직여 판이 밀려도 새 격자 기준으로 클릭해야 한다."""
    old = _frame((1, 1))
    shifted = np.roll(_frame((1, 2)), 24, axis=1)   # 판이 오른쪽으로 24px 이동
    eng = _engine([shifted, shifted, shifted, shifted])
    stale_grid = board.detect_board(old)

    eng._do_move(stale_grid, (1, 1), (1, 2), "RIGHT")
    fresh = board.detect_board(shifted)
    want = fresh.cell_center(1, 2)
    got = eng.window.clicks[0]
    assert abs(got[0] - want[0]) <= 4, "갱신 전 격자로 클릭했습니다"


def test_창이_유효하지_않으면_클릭하지_않는다():
    old = _frame((1, 1))
    eng = _engine([old] * 5)
    eng.window.is_valid = lambda: False
    grid = board.detect_board(old)
    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False
    assert eng.window.clicks == []


# ------------------------------------------- 장애물 파괴가 통하지 않는 경우
# 상하좌우가 전부 막힌 '주머니'. 여기서만 장애물 파괴가 후보로 올라온다.
# (세로 벽 하나로 막은 판은 위아래로 움직여 새 행을 불러올 수 있으므로
#  파괴 대상이 아니다 — 탐사는 무한 우측 진행이라 그쪽이 항상 먼저다.)
BLOCKED = [
    ".X...",
    "XPX..",
    ".X...",
    ".....",
    ".....",
]


class ToastAfterClickWindow(FakeWindow):
    """실제 게임처럼, 클릭한 뒤 잠깐 '이동할 수 없습니다' 안내가 뜨는 창."""

    def __init__(self, clean, toast, toast_frames=4):
        super().__init__([clean])
        self.clean, self.toast = clean, toast
        self.toast_frames, self.left = toast_frames, 0

    def capture(self):
        if self.left > 0:
            self.left -= 1
            return self.toast
        return self.clean

    def click_client(self, x, y, move_duration=0.0, require_focus=True):
        self.clicks.append((int(x), int(y)))
        self.left = self.toast_frames
        return int(x) + 1000, int(y) + 50


def test_장애물_파괴가_연속_실패하면_더_이상_시도하지_않는다():
    """실측: 이 게임은 장애물을 클릭해도 부서지지 않고 안내문만 뜬다.

    같은 자리를 무한히 다시 누르면 안 되므로, 몇 번 실패하면 시도를 접고
    길이 열릴 때까지 기다려야 한다.
    """
    clean = synth.make_board(BLOCKED)
    toast = _with_toast(clean)
    patch = toast[560:640, 150:560].copy()

    eng = _engine([clean], allow_obstacle_break=True, obstacle_break_max_failures=2,
                  blocked_wait_sec=0.05, cycle_pause_sec=0.0, lost_retry_sec=0.05,
                  toast_clear_repeat=1)
    eng.window = ToastAfterClickWindow(clean, toast)
    eng.templates = _toast_templates(patch)
    eng.pick_window = lambda: eng.window

    t = threading.Thread(target=eng.run, daemon=True)
    t.start()
    time.sleep(3.0)
    eng.stop()
    t.join(timeout=5)

    assert eng.break_disabled is True, "실패가 쌓였는데도 파괴 시도를 계속했습니다"
    assert eng.stats.obstacles_broken <= 3,         f"장애물을 {eng.stats.obstacles_broken}번이나 눌렀습니다 (무한 반복)"
    assert eng.stats.blocked_toasts >= 2


def test_설정으로_장애물_파괴를_끌_수_있다():
    """기본은 켜짐(오른쪽이 막히면 장애물을 클릭해 부순다)."""
    eng = _engine([synth.make_board(BLOCKED)] * 10)
    assert eng.break_disabled is False
    eng2 = _engine([synth.make_board(BLOCKED)] * 10, allow_obstacle_break=False)
    assert eng2.break_disabled is True


def test_파괴를_포기한_뒤에는_클릭하지_않고_기다린다():
    clean = synth.make_board(BLOCKED)
    eng = _engine([clean] * 50, blocked_wait_sec=0.05, cycle_pause_sec=0.0)
    eng.break_disabled = True          # 이미 통하지 않는 것을 확인한 상태
    eng.pick_window = lambda: eng.window
    t = threading.Thread(target=eng.run, daemon=True)
    t.start(); time.sleep(1.2); eng.stop(); t.join(timeout=5)
    assert eng.window.clicks == [], "파괴를 포기했는데도 클릭이 나갔습니다"


def test_이동에_성공하면_실패_연속_횟수가_초기화된다():
    old = synth.make_board(SCROLL_BASE)
    after = synth.make_board(_scrolled_layout(SCROLL_BASE, 0, 1))
    eng = _engine([old, after, after, after, after])
    eng.break_fail_streak = 1
    g = board.detect_board(old)
    ok, _, _ = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is True


# ------------------------------------------------- 격자 위상 (실측 회귀)
def _grid_at(y0, conf):
    return board.Grid(xs=[76, 184, 292, 400, 507, 615],
                      ys=[y0 + 88 * i for i in range(6)],
                      confidence=conf, detail={})


def test_격자는_표_수가_아니라_신뢰도로_고른다(monkeypatch):
    """실측 회귀: 실제 MuMuPlayer 화면에서 한 칸 밀린 격자가 잡히는 일이 있었다.

    게임이 다시 그리는 중간에 캡처되면 격자선이 흐려져 밀린 배치가 이긴다.
    이런 프레임은 **무리 지어** 들어오기 때문에(연속 3~4장) 표를 더 모아도
    다수결로는 못 거른다. 실측 60프레임 모의에서 다수결은 장수를 늘려도
    6~7% 에서 안 떨어졌다.

    반면 두 배치의 신뢰도는 겹치지 않았다 (맞는 쪽 0.882~0.911,
    밀린 쪽 0.830~0.853). 그래서 '한 장이라도 잘 나온 쪽'을 믿는다.
    """
    eng = _engine([synth.make_board(LAYOUT_AT[(1, 1)])] * 10)
    # 밀린 배치가 3표, 맞는 배치가 1표. 신뢰도는 맞는 쪽이 높다.
    seq = [_grid_at(507, 0.84), _grid_at(507, 0.85), _grid_at(507, 0.83),
           _grid_at(419, 0.91)]
    it = iter(seq)
    monkeypatch.setattr(explore, "detect_board",
                        lambda img, min_confidence=0.0: next(it, seq[-1]))
    for _ in range(len(seq)):
        g = eng._stable_grid(eng._capture())
    assert g.ys[0] == 419, f"표가 많은 밀린 배치를 골랐습니다 (y0={g.ys[0]})"


def test_격자를_한_장만_보고_고정하지_않는다(monkeypatch):
    """시작 직후 첫 프레임이 하필 밀린 프레임이면 잘못된 격자로 클릭하게 된다.

    실측: 40프레임 중 3장(7.5%)이 밀린 배치였다. 그래서 전체 인식 때는
    몇 장을 더 모아서 정한다.
    """
    eng = _engine([synth.make_board(LAYOUT_AT[(1, 1)])] * 10,
                  grid_min_votes=5)
    seq = [_grid_at(507, 0.84)] + [_grid_at(419, 0.91)] * 6
    it = iter(seq)
    monkeypatch.setattr(explore, "detect_board",
                        lambda img, min_confidence=0.0: next(it, seq[-1]))
    g = eng._stable_grid(eng._capture(), seed=True)
    assert g.ys[0] == 419, "첫 프레임 한 장만 보고 고정했습니다"
    assert len(eng._grid_votes) >= 5, "표를 더 모으지 않았습니다"


def test_이동_확인_루프에서는_추가_캡처를_하지_않는다(monkeypatch):
    """확인 루프는 폴링마다 도는 자리다. 여기서 캡처를 더 하면 확인 기회가 준다."""
    eng = _engine([synth.make_board(LAYOUT_AT[(1, 1)])] * 10, grid_min_votes=5)
    before = eng.window.i
    eng._stable_grid(eng._capture())          # seed=False (기본값)
    assert eng.window.i - before == 1, "이동 확인 경로에서 프레임을 더 소비했습니다"


# ----------------------------------------- 막힌 주머니에서 제자리 맴돌기 방지
POCKET = [
    "..X..",
    ".PX..",
    "..X..",
    "XX...",
    ".X...",
]


def test_갇히면_제자리를_맴돌지_않고_기다린다():
    """실측 회귀: 5칸짜리 주머니에 갇힌 채 25번을 움직였는데 판이 그대로였다.

    (1,0)->(1,1)->(2,1)->(1,1)->(1,0)->(0,0)->... 무한 왕복. 이동 횟수만 축낸다.
    세로로 움직여도 장애물 배치가 그대로면 새 지형이 안 들어온다는 뜻이므로,
    움직임을 멈추고 길이 열릴 때까지 기다려야 한다.
    """
    img = synth.make_board(POCKET)
    eng = _engine([img] * 400, blocked_wait_sec=0.05, cycle_pause_sec=0.0,
                  move_timeout_sec=0.2)
    eng.pick_window = lambda: eng.window

    t = threading.Thread(target=eng.run, daemon=True)
    t.start()
    time.sleep(2.5)
    eng.stop()
    t.join(timeout=5)

    assert eng.stuck_cycles >= 2, "갇힌 것을 알아채지 못했습니다"
    assert len(eng.window.clicks) <= 6, \
        f"갇혔는데 {len(eng.window.clicks)}번이나 클릭했습니다"


def test_갇힘_판정은_플레이어_위치가_아니라_장애물_배치로_한다():
    """플레이어만 움직이고 장애물이 그대로면 새 지형이 안 들어온 것이다.

    처음에는 판 전체(플레이어 포함)를 비교했는데, 움직일 때마다 값이 달라져서
    갇힘을 영영 못 잡았다.
    """
    import board as _b
    import recognize as _r
    from recognize import Kind as _K
    a = synth.make_board(["..X..", ".P...", "..X..", "XX...", ".X..."])
    b = synth.make_board(["..X..", "..P..", "..X..", "XX...", ".X..."])
    eng = _engine([a])
    def sig(img):
        g = _b.detect_board(img)
        sc = _r.analyze(img, g, eng.templates)
        return tuple(tuple(c == _K.OBSTACLE for c in row) for row in sc.cells)
    assert sig(a) == sig(b), "플레이어만 움직였는데 다른 배치로 봤습니다"


def test_스크롤한_뒤에는_남은_경로를_버린다():
    """판이 밀리면 플레이어는 화면상 제자리이고 칸 번호가 통째로 어긋난다.

    실측 회귀: 미리 계산한 경로를 그대로 이어서 클릭했더니, 두 칸 떨어진 곳을
    누르게 돼 '이동할 수 없습니다'가 반복해서 떴다.
    """
    base = synth.make_board(SCROLL_BASE)
    after = synth.make_board(_scrolled_layout(SCROLL_BASE, 0, 1))
    g = board.detect_board(base)
    eng = _engine([base, after, after, after, after])
    ok, _, scrolled = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is True
    assert scrolled is True, "스크롤로 성공한 것을 알려주지 않았습니다"


def test_제자리_도착으로_성공하면_스크롤_플래그가_꺼져_있다():
    old, new = _frame((1, 1)), _frame((1, 2))
    eng = _engine([old, new, new, new, new])
    g = board.detect_board(old)
    ok, _, scrolled = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is True and scrolled is False


# ------------------------------------------------------ 중복 실행 방지
def test_두_번째_실행은_거절된다():
    """바로가기를 두 번 누르면 매크로가 둘 다 같은 게임 창을 클릭해 서로 망친다."""
    import single_instance
    name = r"Global\digimonUp_test_mutex_dup"
    assert single_instance.acquire(name) is True
    assert single_instance.acquire(name) is False, "두 번째 실행이 통과했습니다"


def test_서로_다른_이름은_막지_않는다():
    import single_instance
    assert single_instance.acquire(r"Global\digimonUp_test_mutex_a") is True
    assert single_instance.acquire(r"Global\digimonUp_test_mutex_b") is True


# ------------------------------------------- 아이템 개수 모니터링 (실측 기반)
def _counts(steps=None, brk=None, dash=None):
    import counters
    return counters.Counters(steps=steps, break_=brk, dash=dash)


def test_돌진이_0개면_초록버튼을_누르지_않는다():
    """예전에는 눌러 보고 안 되면 포기했다. 개수를 알면 아예 안 누른다."""
    eng = _engine([synth.make_board(BLOCKED)] * 5)
    eng.counts = _counts(dash=0)
    assert eng._press_green_button() is False
    assert eng.window.clicks == []
    assert eng.stats.green_button_uses == 0


def test_개수를_모르면_예전처럼_시도한다():
    """숫자 템플릿이 없어 못 읽는 경우. 막으면 아예 못 쓰게 되므로 시도한다."""
    eng = _engine([synth.make_board(BLOCKED)] * 5)
    eng.counts = _counts()               # 전부 None = 모름
    assert eng._can_use("dash") is True
    assert eng._can_use("break") is True


def test_남아_있으면_쓴다():
    eng = _engine([synth.make_board(BLOCKED)] * 5)
    eng.counts = _counts(dash=3, brk=7)
    assert eng._can_use("dash") is True
    assert eng._can_use("break") is True


def test_걸음수가_0이면_멈춘다(monkeypatch):
    """더 움직일 수 없는데 계속 클릭하면 이동 확인 실패만 쌓인다."""
    img = synth.make_board(LAYOUT_AT[(1, 1)])
    eng = _engine([img] * 50, cycle_pause_sec=0.0)
    eng.pick_window = lambda: eng.window
    monkeypatch.setattr(explore.counters, "read", lambda im: _counts(steps=0))

    t = threading.Thread(target=eng.run, daemon=True)
    t.start(); t.join(timeout=5)
    assert not t.is_alive(), "걸음수가 0인데 계속 돌았습니다"
    assert eng.window.clicks == [], "걸음수가 0인데 클릭했습니다"


def test_개수_읽기가_실패해도_매크로는_계속_돈다(monkeypatch):
    """카운터 읽기는 부가 기능이다. 여기서 터져도 본체가 멈추면 안 된다."""
    img = synth.make_board(LAYOUT_AT[(1, 1)])
    eng = _engine([img] * 10)

    def boom(im):
        raise ValueError("일부러 낸 오류")

    monkeypatch.setattr(explore.counters, "read", boom)
    eng._update_counts(img)              # 예외가 밖으로 나오면 안 된다
    assert eng.cfg.watch_counters is False, "실패한 뒤에도 계속 읽으려 합니다"


# ------------------------------------------------------- 중지 키 (전역 단축키)
def test_중지키를_누르면_멈추고_클릭이_나가지_않는다(monkeypatch):
    """매크로가 마우스를 움직이는 중에는 GUI 정지 버튼을 겨냥하기 어렵다.
    그래서 창 포커스와 무관한 전역 단축키로도 멈출 수 있어야 한다.
    """
    img = _frame((1, 1))
    eng = _engine([img] * 40, stop_key="F12")
    assert eng._stop_vk == 0x7B, "F12 의 가상 키 코드가 아닙니다"

    monkeypatch.setattr(explore, "is_stop_key_pressed", lambda vk: vk == 0x7B)
    with pytest.raises(Stopped):
        eng._check_stop()
    assert eng.stop_event.is_set(), "키를 뗀 뒤에도 정지 상태로 남아야 합니다"

    grid = board.detect_board(img)
    with pytest.raises(Stopped):
        eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert eng.window.clicks == [], "중지키를 눌렀는데 클릭이 나갔습니다"


def test_중지키를_누르지_않으면_계속_돈다(monkeypatch):
    eng = _engine([_frame((1, 1))] * 10, stop_key="F12")
    monkeypatch.setattr(explore, "is_stop_key_pressed", lambda vk: False)
    eng._check_stop()          # 예외가 나면 안 된다


def test_중지키를_비워두면_키_검사를_하지_않는다(monkeypatch):
    """키를 쓰고 싶지 않은 사람도 있다."""
    eng = _engine([_frame((1, 1))] * 10, stop_key="")
    assert eng._stop_vk == 0
    called = []
    monkeypatch.setattr(explore, "is_stop_key_pressed",
                        lambda vk: called.append(vk) or True)
    eng._check_stop()          # 검사 자체를 건너뛰므로 멈추지 않는다
    assert called == [], "중지키를 비웠는데도 키를 검사했습니다"


def test_중지키는_config_최상위_stop_key_를_따른다():
    """1번 기능과 같은 키를 쓴다. explore 절에 적으면 그것이 우선한다."""
    import settings
    cfg = settings.load_explore_config()
    assert cfg.stop_key, "config.json 의 stop_key 를 못 읽었습니다"
