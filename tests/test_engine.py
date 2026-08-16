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

from digimonup.vision import board
from digimonup.app import explore
from digimonup.vision import recognize
import synth
from digimonup.app.explore import ExploreConfig, ExploreEngine, Stopped

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
    # 합성 화면은 정지 이미지라 움직임으로 플레이어를 찾을 수 없다. 촬영에
    # 시간만 쓰므로 끈다(motion_frames < 2 면 건너뛴다).
    defaults = dict(move_timeout_sec=0.5, poll_interval_sec=0.0,
                    click_settle_sec=0.0, save_debug=False, motion_frames=1)
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
    # 안내문이 뜨는 시점(0.41초)을 지나야 '먹지 않았다'고 단정하므로
    # 제한 시간을 그보다 넉넉히 준다.
    eng = _engine([stay] * 200, move_timeout_sec=2.0)
    grid = board.detect_board(stay)

    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False, "이전 칸에 남아 있는데 성공으로 봤습니다"
    # 판이 하나도 안 바뀌었으므로 '먹지 않은 클릭'으로 보고 다시 누른다.
    # 아무 일도 일어나지 않았으니 두 번 움직일 위험은 없다.
    assert len(eng.window.clicks) == 1 + eng.cfg.dead_click_retries
    assert len(set(eng.window.clicks)) == 1, "다시 누를 때도 같은 자리를 눌러야 합니다"


def test_다시_누르기를_끄면_클릭은_한_번뿐이다():
    """두 번 움직일 위험이 없는지 확인하는 자리이기도 하다."""
    stay = _frame((1, 1))
    eng = _engine([stay] * 200, dead_click_retries=0, move_timeout_sec=2.0)
    grid = board.detect_board(stay)

    ok, _, _ = eng._do_move(grid, (1, 1), (1, 2), "RIGHT")
    assert ok is False
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

    # 합성 화면에는 개수 표시가 없다. 개수를 안 보는 설정으로 두어야 파괴
    # 실패 처리 자체를 시험할 수 있다(안 그러면 '모르니 안 쓴다'로 막힌다).
    eng = _engine([clean], allow_obstacle_break=True, obstacle_break_max_failures=2,
                  blocked_wait_sec=0.05, cycle_pause_sec=0.0, lost_retry_sec=0.05,
                  toast_clear_repeat=1, watch_counters=False)
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
    from digimonup.vision import board as _b
    from digimonup.vision import recognize as _r
    from digimonup.vision.recognize import Kind as _K
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


def test_빠른_추적이_헛값을_내도_스크롤을_확인한다():
    """전진은 스크롤 확인이 **유일한 증거**다. 그 앞을 막으면 안 된다.

    판이 밀리면 플레이어는 화면상 제자리에 남으므로 '목표 칸 도착'으로도
    '움직인 칸'으로도 확인되지 않는다. 예전에는 빠른 추적 결과가 출발칸이나
    None 일 때만 스크롤을 봤는데, 색 기반 추적이 엉뚱한 칸을 하나 내면 그대로
    건너뛰어 이동을 실패로 적었다.

    실측 회귀(298초): 실패로 적힌 이동 18건 중 13건이 사실은 성공이었고,
    그 13건이 제한시간 2.0~3.5초를 전부 썼다(성공한 이동은 0.59초).
    전체 시간의 12%를 여기서 버렸다.
    """
    base = synth.make_board(SCROLL_BASE)
    after = synth.make_board(_scrolled_layout(SCROLL_BASE, 0, 1))
    g = board.detect_board(base)
    eng = _engine([base, after, after, after, after])
    # 빠른 추적이 출발칸도 목표칸도 아닌 말이 안 되는 칸을 낸다고 하자.
    eng._track_player = lambda img, grid: (4, 4)
    ok, _, scrolled = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is True, "추적이 헛값을 냈다고 스크롤 확인을 건너뛰었습니다"
    assert scrolled is True


def test_걸음수가_줄면_한_번만_보고도_성공으로_친다():
    """걸음수 감소는 **한 번만 뜨는 신호**다. 연속 확인을 요구하면 안 된다.

    걸음수는 지금까지 본 최솟값과 견주므로, 줄어든 것을 한 번 보고 나면 그
    값이 새 최솟값이 되어 다음 폴링부터는 다시 뜨지 않는다. 그런데 성공을
    연속 confirm_repeat 번 확인하도록 돼 있어서, 걸음수로만 확인되는 이동은
    구조적으로 절대 성공 처리되지 않았다.

    실측 회귀(299초): 실패로 적힌 이동 12건 전부가 이 경우였다. 폴링 기록에
    ok=True 가 한 번 떴다가 다음 폴링에서 0 으로 되돌아간 것이 남아 있다.
    그 뒤 '클릭이 안 먹었다'며 같은 칸을 다시 눌렀고, 거기엔 이미 플레이어가
    서 있어서 안내문이 떴다(안내문 12번 = 실패 12번).
    """
    stay = _frame((1, 1))
    eng = _engine([stay] * 8, confirm_repeat=2, watch_counters=True)
    eng._track_player = lambda img, grid: (1, 1)      # 끝내 목표 칸을 못 찾는다
    calls = []

    def steps_dropped(img):
        calls.append(1)
        return len(calls) == 2                        # 딱 한 번만 뜬다

    eng._steps_dropped = steps_dropped
    g = board.detect_board(stay)
    ok, _, scrolled = eng._do_move(g, (1, 1), (2, 1), "DOWN")
    assert ok is True, "걸음수가 줄었는데도 실패로 봤습니다"
    assert scrolled is False, "세로 이동을 스크롤로 셌습니다"


def test_제자리_도착으로_성공하면_스크롤_플래그가_꺼져_있다():
    old, new = _frame((1, 1)), _frame((1, 2))
    eng = _engine([old, new, new, new, new])
    g = board.detect_board(old)
    ok, _, scrolled = eng._do_move(g, (1, 1), (1, 2), "RIGHT")
    assert ok is True and scrolled is False


# ------------------------------------------------------ 중복 실행 방지
def test_두_번째_실행은_거절된다():
    """바로가기를 두 번 누르면 매크로가 둘 다 같은 게임 창을 클릭해 서로 망친다."""
    from digimonup.win import single_instance
    name = r"Global\digimonUp_test_mutex_dup"
    assert single_instance.acquire(name) is True
    assert single_instance.acquire(name) is False, "두 번째 실행이 통과했습니다"


def test_서로_다른_이름은_막지_않는다():
    from digimonup.win import single_instance
    assert single_instance.acquire(r"Global\digimonUp_test_mutex_a") is True
    assert single_instance.acquire(r"Global\digimonUp_test_mutex_b") is True


# ------------------------------------------- 아이템 개수 모니터링 (실측 기반)
def _counts(steps=None, brk=None, dash=None):
    from digimonup.vision import counters
    return counters.Counters(steps=steps, break_=brk, dash=dash)


def test_돌진이_0개면_초록버튼을_누르지_않는다():
    """예전에는 눌러 보고 안 되면 포기했다. 개수를 알면 아예 안 누른다."""
    eng = _engine([synth.make_board(BLOCKED)] * 5)
    eng.counts = _counts(dash=0)
    assert eng._press_green_button() is False
    assert eng.window.clicks == []
    assert eng.stats.green_button_uses == 0


def test_개수를_못_읽으면_아껴서_쓰지_않는다():
    """실측 회귀: 돌진 줄 옆에 충전 타이머가 뜨자 줄을 두 개만 찾아 세 항목이
    전부 None 이 됐다. 예전 규칙('모르면 해 보자')이면 아무 제동 없이 계속
    쓴다는 뜻이 되고, 실제로 돌진 45개를 다 태웠다.

    아껴야 하는 자원에서 '모름'은 '써도 된다'가 아니다.
    """
    eng = _engine([synth.make_board(BLOCKED)] * 5)
    eng.counts = _counts()               # 전부 None = 못 읽음
    assert eng._can_use("dash") is False
    assert eng._can_use("break") is False


def test_개수를_아예_안_보기로_했으면_예전처럼_시도한다():
    """못 읽는 것과 애초에 안 보는 것은 다르다.

    숫자 템플릿이 없는 사람은 watch_counters 를 꺼서 예전처럼 쓸 수 있어야 한다.
    """
    eng = _engine([synth.make_board(BLOCKED)] * 5, watch_counters=False)
    eng.counts = _counts()
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
    from digimonup.base import settings
    cfg = settings.load_explore_config()
    assert cfg.stop_key, "config.json 의 stop_key 를 못 읽었습니다"


# ------------------- 칩 획득 이펙트 걸러내기 — 실측 회귀
def _scene_with_goals(goals):
    from digimonup.vision.recognize import Detection, Scene
    from digimonup.vision.recognize import Kind
    cells = [[Kind.EMPTY] * 5 for _ in range(5)]
    dets = []
    for r, c in goals:
        cells[r][c] = Kind.GOAL
        dets.append(Detection(Kind.GOAL, r, c, 0.95))
    return Scene(grid=None, cells=cells, goals=dets)


def _lock(eng, goals):
    """묶음을 잠근다. 두 프레임 연속으로 같아야 잠기므로 두 번 준다."""
    for _ in range(2):
        eng._confirm_goals(_scene_with_goals(goals))


def test_확인된_새_칩은_언제든_받아들인다():
    """실측 회귀: 묶음을 닫아 두니 그사이 오른쪽에서 들어온 진짜 칩까지 버렸다.

    180초 74사이클에서 10사이클이 칩을 무시했고, 거기엔 (0,4), (3,4) 처럼
    막 들어온 칩도 있었다. 이펙트를 가려낼 수 있는 지금은 닫아 둘 이유가 없다.
    """
    eng = _engine([])
    eng._motion_valid = True
    eng._moving_cells = set()
    eng._confirm_goals(_scene_with_goals([(1, 3)]))

    sc = _scene_with_goals([(1, 3), (0, 4)])        # 새 열에서 칩이 들어왔다
    eng._confirm_goals(sc)
    assert sorted((d.row, d.col) for d in sc.goals) == [(0, 4), (1, 3)]


def test_잠근_묶음에_없는_칩은_이펙트로_본다():
    """실측: 칩을 먹으면 디지몬 주변으로 칩이 흩어지는 이펙트가 뜬다.

    그 칩들은 템플릿 0.94~0.98, 주황 0.096~0.202 로 진짜와 같은 그림이라
    모양·색으로는 가를 수 없다. 그래서 묶음을 잠그고 새 칩을 아예 안 본다.
    """
    from digimonup.vision.recognize import Kind
    eng = _engine([])
    _lock(eng, [(1, 3)])

    sc = _scene_with_goals([(1, 3), (4, 0), (4, 1)])     # 이펙트 둘이 끼어들었다
    eng._confirm_goals(sc)
    assert sorted((d.row, d.col) for d in sc.goals) == [(1, 3)]
    assert sc.cells[4][0] is Kind.EMPTY, "경로 계산이 보는 판에서도 빠져야 합니다"
    assert sc.cells[1][3] is Kind.GOAL


def test_이펙트가_오래_남아도_끼어들지_못한다():
    """'연속 두 번 보이면 인정' 규칙은 이펙트가 두 사이클 남으면 뚫렸다."""
    eng = _engine([])
    _lock(eng, [(1, 3)])
    for _ in range(4):
        sc = _scene_with_goals([(1, 3), (2, 1)])
        eng._confirm_goals(sc)
    assert sorted((d.row, d.col) for d in sc.goals) == [(1, 3)]


def test_전진하면_칩_자리를_따라간다():
    """전진 한 번에 판이 한 열 밀린다. 알고 있던 칩도 함께 옮겨야 한다."""
    eng = _engine([])
    _lock(eng, [(2, 4)])

    eng._scrolls_since = 1
    sc = _scene_with_goals([(2, 3)])
    eng._confirm_goals(sc)
    assert [(d.row, d.col) for d in sc.goals] == [(2, 3)]


def test_안_보이는_칩은_바로_버린다():
    """실측 회귀: 안 보이는 칩을 기억해 두는 것이 유령의 원천이었다.

    300초에 '칩을 먹었다'고 판단한 22건 중 17건이 헛것이었고(상단 보유량이 안
    올랐다) 유령 칸은 전부 플레이어가 선 자리였다. 없어진 칩을 기억해 두었다가
    전진으로 그 자리가 밀려와 겹치는 순간 '먹었다'고 처리한 것이다.

    기억이 필요했던 이유는 검출을 못 믿어서였는데 지금은 믿을 수 있다
    (12프레임 눈 대조: 오탐 0 / 미검출 0).
    """
    eng = _engine([])
    eng._motion_valid = True
    eng._moving_cells = set()
    eng._confirm_goals(_scene_with_goals([(2, 2)]))

    sc = _scene_with_goals([])           # 이번 프레임엔 안 보인다
    eng._confirm_goals(sc)
    assert sc.goals == [], "안 보이는 칩을 목표로 남기면 안 됩니다"
    assert not eng.chips.locked


def test_계속_안_보이면_없어진_것으로_본다():
    eng = _engine([])
    _lock(eng, [(2, 2)])
    for _ in range(eng.chips.MISS_LIMIT):
        sc = _scene_with_goals([])
        eng._confirm_goals(sc)
    assert sc.goals == []


def test_플레이어가_선_칸의_칩은_먹은_것으로_본다():
    from digimonup.vision.recognize import Detection, Kind
    eng = _engine([])
    _lock(eng, [(2, 1)])

    sc = _scene_with_goals([])
    sc.player = Detection(Kind.PLAYER, 2, 1, 1.0)
    eng._confirm_goals(sc)
    assert not eng.chips.locked, "먹은 칩만 있었으면 묶음이 비어야 합니다"


def test_묶음을_다_먹으면_다시_읽는다():
    eng = _engine([])
    _lock(eng, [(0, 2)])
    eng.chips.collected_at((0, 2))
    assert not eng.chips.locked

    _lock(eng, [(3, 4), (1, 3)])          # 새 묶음을 읽는다
    assert eng.chips.chips == {(3, 4), (1, 3)}


def test_그림이_움직이는_칸의_칩은_이펙트로_본다():
    """실측(0.12초 간격 161프레임): 잡힌 칩 자리 15건이 전부 0.72초 안에
    사라졌고, 15건 모두 움직인 프레임이 있었다.

    판이 가라앉았을 때 움직이는 것은 디지몬과 연출뿐이다. 판에 놓인 칩은
    가만히 있는다.
    """
    from digimonup.vision.recognize import Kind
    eng = _engine([])
    _lock(eng, [(1, 3)])

    eng._moving_cells = {(4, 0), (4, 1)}         # 이펙트가 날아다니는 칸
    sc = _scene_with_goals([(1, 3), (4, 0), (4, 1)])
    eng._confirm_goals(sc)
    assert sorted((d.row, d.col) for d in sc.goals) == [(1, 3)]
    assert sc.cells[4][0] is Kind.EMPTY


def test_디지몬이_선_칸은_늘_움직이므로_빼고_본다():
    """디지몬은 제자리 애니메이션이 돌아 언제나 움직이는 칸이다."""
    from digimonup.vision.recognize import Detection, Kind
    eng = _engine([])
    _lock(eng, [(2, 1)])

    sc = _scene_with_goals([(2, 1)])
    sc.player = Detection(Kind.PLAYER, 0, 0, 1.0)
    eng._moving_cells = {(0, 0), (2, 1)}
    eng._confirm_goals(sc)
    assert eng._ghost_suspect == 1, "디지몬 칸이 아닌 (2,1) 은 의심해야 합니다"


def test_이펙트가_의심되면_묶음을_잠그지_않는다():
    """이펙트가 뜬 화면에서 잠그면 이펙트를 진짜 칩으로 굳혀 버린다."""
    eng = _engine([])
    eng._moving_cells = {(3, 2)}
    for _ in range(3):
        eng._confirm_goals(_scene_with_goals([(3, 2)]))
    assert not eng.chips.locked
    assert eng._ghost_suspect > 0


def test_이펙트가_없다고_확인되면_바로_잠근다():
    """실측: 두 프레임 연속 확인을 요구하니 칩이 처음 보인 사이클에 목표가 된
    경우가 150초 65사이클 동안 **0건**이었고, 36건이 한 사이클 이상 늦었다.

    그사이 전진해 버리면 칩이 뒤로 밀려 되돌아가서 먹어야 한다.
    움직임으로 이펙트를 직접 가려낼 수 있는 지금은 기다릴 이유가 없다.
    """
    eng = _engine([])
    eng._motion_valid = True            # 움직임 검사가 제대로 돌았다
    eng._moving_cells = set()           # 움직이는 칩이 없다
    sc = _scene_with_goals([(1, 3)])
    eng._confirm_goals(sc)
    assert [(d.row, d.col) for d in sc.goals] == [(1, 3)], \
        "이펙트가 없는 것이 확인됐으면 첫 사이클에 잡아야 합니다"


def test_이펙트_확인이_안_되면_예전처럼_두_번_본다():
    """움직임 검사가 못 돌았으면(스크롤 중 등) 함부로 믿지 않는다."""
    eng = _engine([])
    eng._motion_valid = False
    sc = _scene_with_goals([(1, 3)])
    eng._confirm_goals(sc)
    assert sc.goals == [], "확인이 안 됐으면 한 번 더 봐야 합니다"
    sc2 = _scene_with_goals([(1, 3)])
    eng._confirm_goals(sc2)
    assert [(d.row, d.col) for d in sc2.goals] == [(1, 3)]


# ------------------- 걸음수로 이동 성공을 가려내기 — 실측 회귀
def test_늦게_갱신된_걸음수를_이번_이동으로_착각하지_않는다():
    """실측: 걸음수는 반영에 0.53~2.10초가 걸리는데 이동 하나는 1.85초다.

    '클릭 직전 값보다 작으면 성공'으로 보면 직전 이동의 감소를 이번 이동의
    성공으로 착각한다. 실제로 움직이지도 않았는데 성공으로 치고 (2,1) 과
    (2,0) 을 일곱 번 오갔다.
    """
    from digimonup.vision.counters import Counters
    eng = _engine([])
    assert eng._note_steps(Counters(steps=100)) is False      # 기준 잡기
    # 화면이 늦게 갱신돼 같은 값이 계속 보인다 -> 새 이동이 아니다
    assert eng._note_steps(Counters(steps=100)) is False
    # 직전 이동분이 이제야 반영됐다 -> 이번 이동의 증거는 아니다... 가 아니라
    # 최솟값보다 작아졌으므로 '새로 줄었다'가 맞다
    assert eng._note_steps(Counters(steps=99)) is True
    # 그 값이 한동안 계속 보여도 다시 성공으로 세지 않는다
    assert eng._note_steps(Counters(steps=99)) is False
    assert eng._note_steps(Counters(steps=100)) is False, "늦은 화면은 무시한다"


def test_걸음수를_채워_넣으면_기준을_다시_잡는다():
    from digimonup.vision.counters import Counters
    eng = _engine([])
    eng._note_steps(Counters(steps=10))
    eng._note_steps(Counters(steps=900))          # 충전
    assert eng._steps_min == 900
    assert eng._note_steps(Counters(steps=899)) is True


def test_걸음수를_못_읽으면_다른_신호에_맡긴다():
    from digimonup.vision.counters import Counters
    eng = _engine([])
    assert eng._note_steps(Counters(steps=None)) is False
    assert eng._note_steps(None) is False


def test_0열_칩은_한_번_더_보고_정한다():
    """틀렸을 때의 대가가 방향에 따라 다르다.

    앞쪽 칩이 가짜면 어차피 전진하던 길이라 손해가 없지만, 0열 칩은 왼쪽으로
    되돌아가야 하므로 가짜였다면 걸음수를 두 번 버린다.
    """
    eng = _engine([])
    eng._motion_valid = True
    eng._moving_cells = set()

    sc = _scene_with_goals([(2, 0), (1, 3)])
    eng._confirm_goals(sc)
    assert sorted((d.row, d.col) for d in sc.goals) == [(1, 3)], \
        "0열 칩은 첫 사이클에 목표로 삼지 않는다"

    sc2 = _scene_with_goals([(2, 0), (1, 3)])       # 다음에도 보였다 -> 진짜다
    eng._confirm_goals(sc2)
    assert (2, 0) in {(d.row, d.col) for d in sc2.goals}


def test_걸러낸_칩이_scene_goal_로_되살아나지_않는다():
    """실측 회귀: analyze 는 가장 확실한 칩 하나를 scene.goal 에도 따로 담는다.

    추적기가 scene.goals 와 cells 만 거르고 scene.goal 을 놔두면, plan_route 의
    호환용 갈래("goals 가 비었으면 goal 을 쓴다")로 걸러낸 칩이 되살아난다.
    300초에서 좌이동 5건 중 3건이 '계획=목적지인데 칩 목록은 비어 있음' 이었다.
    """
    from digimonup.vision.recognize import Detection, Kind
    eng = _engine([])
    eng._motion_valid = True
    eng._moving_cells = {(2, 0)}                 # 이 칩은 움직인다 = 이펙트

    sc = _scene_with_goals([(2, 0)])
    sc.goal = Detection(Kind.GOAL, 2, 0, 0.95)   # analyze 가 담아 두는 값
    eng._confirm_goals(sc)

    assert sc.goals == []
    assert sc.goal is None, "걸러낸 칩이 scene.goal 에 남아 있으면 안 됩니다"


# ------------------- 돌진 (실험으로 확정한 규칙)
from digimonup.vision.counters import Counters  # noqa: E402
def _with_player(scene, row, col):
    from digimonup.vision.recognize import Detection, Kind
    scene.player = Detection(Kind.PLAYER, row, col, 1.0)
    return scene


# ------------------- 길이 있으면 아이템을 쓰지 않는다 (실측 회귀)
def _board_scene(layout):
    """'.'=빈칸 X=장애물 P=플레이어 G=칩"""
    from digimonup.vision.recognize import Detection, Kind, Scene
    sym = {".": Kind.EMPTY, "X": Kind.OBSTACLE, "P": Kind.PLAYER, "G": Kind.GOAL}
    cells = [[sym[ch] for ch in row] for row in layout]
    sc = Scene(grid=None, cells=cells)
    for r in range(5):
        for c in range(5):
            if cells[r][c] == Kind.PLAYER:
                sc.player = Detection(Kind.PLAYER, r, c, 1.0)
            elif cells[r][c] == Kind.GOAL:
                sc.goals.append(Detection(Kind.GOAL, r, c, 0.9))
    return sc


def test_공짜로_갈_수_있으면_부수지_않는다():
    """실측 회귀(276.3초): **한 번만 전진하면 공짜로 먹을 칩**에 부수기를 썼다.

    판은 오른쪽으로 갈 때마다 왼쪽으로 한 열 밀린다(19장). (3,1) 을 막고 선
    장애물은 다음 전진에 (3,0) 으로 밀려나고, (3,2) 에 있던 칩이 (3,1) 로
    온다. 2행에서 그냥 전진한 뒤 한 칸 내려가면 걸음수 둘로 먹는 자리였다.

    299초 기록의 부수기 5회 중 3회가 이런 경우였다(35.6초·168.5초·276.3초).
    나머지 2회는 정말 아무 데도 못 가는 판이었다 — 아래 test_정말_막혔으면 참고.
    """
    from digimonup.vision.recognize import Kind
    layout = [
        ".....",
        "..X..",
        ".P...",
        ".XG..",
        "...X.",
    ]
    eng = _engine([])
    eng.counts = Counters(steps=1300, break_=55, dash=2)
    sc = _board_scene(layout)
    plan = eng._plan(sc)
    hit = [(r, c) for r, c in plan.path[1:] if layout[r][c] == "X"]
    assert not hit,         f"부수지 않고도 갈 수 있는데 {hit} 를 부수려 합니다: {plan.path}"



def test_정말_막혔으면_부순다():
    """아껴 쓰되, 부수지 않으면 아무 데도 못 가는 판에서는 부순다."""
    eng = _engine([])
    eng.counts = Counters(steps=1300, break_=55, dash=2)
    plan = eng._plan(_board_scene([
        "XX...",
        "XX...",
        "XP...",
        "XX...",
        "XX...",
    ]))
    assert plan.path and len(plan.path) >= 2,         f"부수면 갈 수 있는데 멈춰 섰습니다: {plan.describe()}"



def test_다른_행의_칩을_지나치게_되면_돌진하지_않는다():
    """돌진은 세 칸을 건너뛴다. 그 칩들은 화면 밖으로 밀려 영영 못 먹는다."""
    eng = _engine([])
    eng.counts = Counters(steps=100, break_=10, dash=5)
    pressed = []
    eng._press_green_button = lambda: pressed.append(1) or True

    sc = _with_player(_scene_with_goals([(2, 3)]), 0, 1)   # 칩은 2행, 나는 0행
    assert eng._dash_if_worth(sc) is False
    assert pressed == []


def test_막힌_행의_칩은_돌진으로_챙긴다():
    """실측: 같은 행 2열 칩을 두고 돌진하니 보유량 267.8K -> 268.0K.

    단 **길이 막혔을 때만** 쓴다 (아래 test_뚫린_행에서는_걸어서_챙긴다 참고).
    """
    from digimonup.vision.recognize import Kind
    eng = _engine([])
    eng.counts = Counters(steps=100, break_=10, dash=5)
    pressed = []
    eng._press_green_button = lambda: pressed.append(1) or True

    sc = _with_player(_scene_with_goals([(0, 2), (0, 4)]), 0, 1)
    sc.cells[0][3] = Kind.OBSTACLE          # 걸어서는 (0,4) 를 못 지나간다
    assert eng._dash_if_worth(sc) is True
    assert pressed == [1]


def test_뚫린_행에서는_걸어서_챙긴다():
    """실측 회귀(299초): 돌진한 두 번 다 **장애물이 하나도 없는 행**이었다.

    63.7초 플레이어 (0,1) / 칩 (0,2)·(0,4), 258.7초 플레이어 (0,1) /
    칩 (0,2)·(0,3). 둘 다 오른쪽만 누르면 걸음수 몇 개로 전부 먹는 자리였다.
    아이템은 길이 막혔을 때 쓰는 것이지, 뚫린 길에서 걸음수를 아끼자고
    쓰는 것이 아니다.
    """
    eng = _engine([])
    eng.counts = Counters(steps=1300, break_=55, dash=2)
    eng._press_green_button = lambda: (_ for _ in ()).throw(
        AssertionError("뚫린 길인데 돌진했습니다"))
    sc = _with_player(_scene_with_goals([(0, 2), (0, 4)]), 0, 1)
    assert eng._dash_if_worth(sc) is False



def test_칩이_하나뿐이면_걸어서_챙긴다():
    """실측 회귀(67.5초): **막다른 길에서 부순 자리에 나온 칩**에 돌진을 썼다.

    (4,2) 장애물을 부쉈더니 그 자리에 칩이 드러났다. 바로 옆 칸이라 오른쪽
    한 번, 걸음수 1이면 먹는 자리였고 경로도 그렇게 잡혀 있었다. 그런데 그
    계획을 가로채 **마지막 남은 돌진(1 -> 0)** 을 썼다.

    칩 하나는 걸어가야 걸음수 최대 3이다. 돌진은 그보다 훨씬 귀하다
    (남은 양 1550 대 45). 하나짜리는 언제나 걸어간다.
    """
    eng = _engine([])
    eng.counts = Counters(steps=1540, break_=59, dash=1)
    eng._press_green_button = lambda: (_ for _ in ()).throw(
        AssertionError("한 걸음이면 되는 칩에 돌진을 썼습니다"))
    sc = _with_player(_scene_with_goals([(4, 2)]), 4, 1)
    assert eng._dash_if_worth(sc) is False



def test_챙길_칩이_없으면_돌진을_아낀다():
    """돌진 하나는 걸음수 세 개어치인데 남은 양은 걸음수 1550 대 돌진 45 다.

    빈 길에서 쓰면 걸음수 셋을 아끼자고 서른네 배 귀한 것을 버리는 셈이다.
    """
    eng = _engine([])
    eng.counts = Counters(steps=100, break_=10, dash=5)
    eng._press_green_button = lambda: (_ for _ in ()).throw(
        AssertionError("챙길 칩이 없는데 돌진했습니다"))
    assert eng._dash_if_worth(_with_player(_scene_with_goals([]), 1, 1)) is False


def test_돌진이_0개면_누르지_않는다():
    eng = _engine([])
    eng.counts = Counters(steps=100, break_=10, dash=0)
    eng._press_green_button = lambda: (_ for _ in ()).throw(
        AssertionError("돌진이 0인데 눌렀습니다"))
    assert eng._dash_if_worth(_with_player(_scene_with_goals([]), 1, 1)) is False


def test_설정으로_돌진을_끌_수_있다():
    eng = _engine([], use_dash=False)
    eng.counts = Counters(steps=100, break_=10, dash=5)
    eng._press_green_button = lambda: (_ for _ in ()).throw(
        AssertionError("꺼 두었는데 눌렀습니다"))
    assert eng._dash_if_worth(_with_player(_scene_with_goals([]), 1, 1)) is False
