"""창 선택 / 상단 탭 / 5x5 게임판 검출 테스트."""

from __future__ import annotations

import cv2
import numpy as np

import board
import explore
import emulator_window
import recognize
import synth
from emulator_window import Candidate

REAL = "tests/fixtures/explore_sample1.png"


def _tset(images):
    t = recognize.TemplateSet.__new__(recognize.TemplateSet)
    t.name, t.allow_flip, t.images = "x", False, list(images)
    t.paths = [f"t{i}.png" for i in range(len(images))]
    return t


# ---------------------------------------------------------------- 게임판 검출
def test_5x5_격자_검출_합성():
    img = synth.make_board(["....." , ".P..X", "..X..", "....X", "...i."])
    g = board.detect_board(img)
    assert g is not None
    assert len(g.xs) == 6 and len(g.ys) == 6
    for got, want in zip(g.xs, synth.XS):
        assert abs(got - want) <= 5
    for got, want in zip(g.ys, synth.YS):
        assert abs(got - want) <= 6
    assert g.confidence > 0.6


def test_5x5_격자_검출_실제캡처():
    img = cv2.imread(REAL)
    assert img is not None, "실제 캡처 fixture 가 없습니다"
    g = board.detect_board(img)
    assert g is not None
    # 손으로 잰 값: 세로선 76/184/292/400/508/616, 가로선 419/507/595/684/772/851
    assert [abs(a - b) <= 4 for a, b in zip(g.xs, [76, 184, 292, 400, 508, 616])] == [True] * 6
    assert [abs(a - b) <= 10 for a, b in zip(g.ys, [419, 507, 595, 684, 772, 851])] == [True] * 6
    assert g.confidence > 0.6


def test_게임판이_없는_창은_거부된다():
    assert board.detect_board(synth.make_non_game_window()) is None


def test_두_실제_캡처가_같은_격자를_낸다():
    """격자 위상이 한 칸 밀리는 일이 없어야 한다.

    게임판 위쪽에는 장식용 청록 블록 띠가 있어서, 안쪽 격자선 4개만 보고
    맞추면 '한 칸 위로 밀린' 배치도 점수가 비슷해진다. 실제로 같은 화면을
    두 번 캡처했을 때 서로 다른 격자가 나온 적이 있어 회귀 테스트로 남긴다.
    """
    a = cv2.imread("tests/fixtures/explore_sample1.png")
    b = cv2.imread("tests/fixtures/explore_sample2.png")
    ga, gb = board.detect_board(a), board.detect_board(b)
    assert ga is not None and gb is not None
    assert ga.ys == gb.ys and ga.xs == gb.xs
    assert abs(ga.ys[0] - 419) <= 3, "게임판 위쪽 장식 띠를 첫 행으로 잡았습니다"


def test_엄격한_셀_색_마스크로_게임판_세로_범위를_잡는다():
    img = cv2.imread("tests/fixtures/explore_sample1.png")
    ext_x, ext_y = board.board_extent(img)
    assert ext_y is not None
    # 실측: 셀 색(S>=210)이 이어지는 구간은 420~841
    assert abs(ext_y[0] - 420) <= 6
    assert ext_y[1] > 800
    assert ext_x is not None and abs(ext_x[0] - 76) <= 8


def test_장식_블록_띠는_셀_색으로_치지_않는다():
    """장식 띠는 채도가 낮다(실측 S≈159). 셀(S≈245)과 구분돼야 한다."""
    img = cv2.imread("tests/fixtures/explore_sample1.png")
    strict = board.strict_cell_mask(img)
    band = strict[350:410].mean()      # 게임판 바로 위 장식 띠
    cells = strict[430:500].mean()     # 첫 번째 행 안쪽
    assert band < 0.10 < cells


def test_격자선은_등간격_가정이_아니라_검출값을_쓴다():
    """아래로 갈수록 간격이 줄어드는 원근을 따라가는지 확인."""
    img = cv2.imread(REAL)
    g = board.detect_board(img)
    gaps = np.diff(g.ys)
    assert gaps.std() > 0, "모든 간격이 완전히 같다면 등간격 모델을 그대로 쓴 것"


# ------------------------------------------------------------- 상단 고정 탭
def test_상단_고정_탭_검출():
    base = synth.make_board(["....." , ".P...", ".....", ".....", "....."])
    tabbed = synth.make_top_tab(base)
    tab_tpl = _tset([synth.crop(tabbed, (230, 60, 470, 130))])

    score, box = recognize.find_top_tab(tabbed, tab_tpl)
    assert score > 0.9
    assert box is not None and box[1] < base.shape[0] * 0.35

    # 탭이 없는 화면에서는 유사도가 확 떨어져야 한다.
    score_none, _ = recognize.find_top_tab(base, tab_tpl)
    assert score_none < score - 0.2


# ------------------------------------------------------------- 창 선택
def test_여러_창_중_올바른_창을_고른다(monkeypatch, blank_templates):
    """게임판이 있는 창만 골라야 한다. 제목은 셋 다 MuMuPlayer 로 똑같이 둔다."""
    game = synth.make_board(["....." , ".P..X", "..X..", ".....", "....i"])
    other = synth.make_non_game_window()

    cands = [
        Candidate(hwnd=0x111, top_hwnd=0x110, title="MuMuPlayer", cls="Qt5QWindowIcon",
                  width=other.shape[1], height=other.shape[0]),
        Candidate(hwnd=0x222, top_hwnd=0x220, title="MuMuPlayer", cls="Qt5QWindowIcon",
                  width=game.shape[1], height=game.shape[0]),
        Candidate(hwnd=0x333, top_hwnd=0x330, title="MuMuPlayer", cls="Qt5QWindowIcon",
                  width=other.shape[1], height=other.shape[0]),
    ]
    frames = {0x111: other, 0x222: game, 0x333: other}

    monkeypatch.setattr(explore, "enumerate_candidates", lambda **kw: cands)
    monkeypatch.setattr(explore, "capture_client", lambda h: frames[h])

    engine = explore.ExploreEngine(log=lambda *_: None)
    win = engine.pick_window()
    assert win is not None
    assert win.hwnd == 0x222
    assert any("0x222" in line for line in engine.candidates_report)


def test_상단_탭_템플릿이_있으면_탭까지_맞아야_고른다(monkeypatch):
    """게임판은 둘 다 있지만 상단 탭이 있는 창만 골라야 한다."""
    layout = ["....." , ".P..X", ".....", "....X", "....."]
    plain = synth.make_board(layout)
    tabbed = synth.make_top_tab(plain)

    cands = [
        Candidate(hwnd=0xAAA, top_hwnd=0xAA0, title="MuMuPlayer", cls="Qt5QWindowIcon",
                  width=plain.shape[1], height=plain.shape[0]),
        Candidate(hwnd=0xBBB, top_hwnd=0xBB0, title="MuMuPlayer", cls="Qt5QWindowIcon",
                  width=tabbed.shape[1], height=tabbed.shape[0]),
    ]
    frames = {0xAAA: plain, 0xBBB: tabbed}
    monkeypatch.setattr(explore, "enumerate_candidates", lambda **kw: cands)
    monkeypatch.setattr(explore, "capture_client", lambda h: frames[h])

    engine = explore.ExploreEngine(log=lambda *_: None)
    engine.templates["top_tab"] = _tset([synth.crop(tabbed, (230, 60, 470, 130))])

    win = engine.pick_window()
    assert win is not None and win.hwnd == 0xBBB


def test_후보가_하나도_조건을_만족하지_않으면_None(monkeypatch, blank_templates):
    other = synth.make_non_game_window()
    cands = [Candidate(hwnd=0x1, top_hwnd=0x0, title="MuMuPlayer", cls="Qt5QWindowIcon",
                       width=other.shape[1], height=other.shape[0])]
    monkeypatch.setattr(explore, "enumerate_candidates", lambda **kw: cands)
    monkeypatch.setattr(explore, "capture_client", lambda h: other)
    engine = explore.ExploreEngine(log=lambda *_: None)
    assert engine.pick_window() is None


def test_관리창은_후보에서_제외되는_규칙이_있다():
    """멀티 인스턴스 관리창(MuMuNativeWindow 자식)을 거르는 목록이 살아 있는지.

    파생 별칭이 아니라 원본인 프로필을 본다. 별칭은 쓰는 곳이 없어 지웠다.
    """
    manager = {c for p in emulator_window.EMULATOR_PROFILES
               for c in p.exclude_child_classes}
    render = {c for p in emulator_window.EMULATOR_PROFILES
              for c in p.render_classes}
    assert "MuMuNativeWindow" in manager
    assert "nemuwin" in render


# ------------------------------------- 여러 앱플레이어 지원 (창 클래스에 의존하지 않기)
def test_제목만_비슷한_창은_앱플레이어로_단정하지_않는다():
    """실측 회귀: 이 저장소 페이지를 띄운 브라우저 창의 제목에 'MuMu' 가 들어 있어
    MuMuPlayer 로 잡혔다. 그러면 엉뚱한 창이 1순위 후보가 된다.

    제목은 힌트일 뿐이고, 자식 창 구조가 맞아야 앱플레이어로 확정한다.
    """
    browser = [(0x1, "Chrome_WidgetWin_1", "")]
    prof, sure = emulator_window._match_profile(
        "aveli99-k/digimonUp: MuMuPlayer 디지몬 탐사 - Whale", browser)
    assert sure is False, "제목만 보고 앱플레이어로 확정했습니다"

    real = [(0x2, "nemuwin", "nemudisplay")]
    prof, sure = emulator_window._match_profile("Android Device", real)
    assert sure is True and prof.name == "MuMuPlayer"


def test_모르는_창에는_더_높은_격자_기준을_요구한다(monkeypatch, blank_templates):
    """모르는 창까지 후보로 올리므로 무관한 창이 얻어걸릴 여지가 생겼다.

    실측: 브라우저 창이 격자 0.48~0.55 로 기본 기준 0.45 를 넘었다.
    진짜 게임 화면은 0.91 이므로, 모르는 창에만 기준을 높여 갈라낸다.
    """
    game = synth.make_board(["....." , ".P..X", "..X..", ".....", "....i"])

    # 같은 화면인데 하나는 앱플레이어로 알아본 창, 하나는 모르는 창
    known = Candidate(hwnd=0x11, top_hwnd=0x10, title="Android Device",
                      cls="Qt5QWindowIcon", width=game.shape[1],
                      height=game.shape[0], emulator="MuMuPlayer")
    unknown = Candidate(hwnd=0x22, top_hwnd=0x20, title="어떤 앱",
                        cls="Chrome_WidgetWin_1", width=game.shape[1],
                        height=game.shape[0])

    monkeypatch.setattr(explore, "capture_client", lambda h: game)

    # 게임판 신뢰도가 기본 기준은 넘지만 '모르는 창' 기준에는 못 미치는 상황
    monkeypatch.setattr(explore, "detect_board",
                        lambda img, min_confidence=0.0: board.Grid(
                            xs=[0, 1, 2, 3, 4, 5], ys=[0, 1, 2, 3, 4, 5],
                            confidence=0.55, detail={}))

    monkeypatch.setattr(explore, "enumerate_candidates", lambda **kw: [unknown])
    assert explore.ExploreEngine(log=lambda *_: None).pick_window() is None, \
        "모르는 창이 낮은 신뢰도로 통과했습니다"

    monkeypatch.setattr(explore, "enumerate_candidates", lambda **kw: [known])
    win = explore.ExploreEngine(log=lambda *_: None).pick_window()
    assert win is not None and win.hwnd == 0x11, \
        "아는 앱플레이어인데 같은 신뢰도로 거절했습니다"


def test_셸_창은_후보에서_제외한다():
    """바탕화면(Progman/WorkerW)이나 작업표시줄이 후보로 올라오면 안 된다."""
    for cls in ("Progman", "WorkerW", "Shell_TrayWnd"):
        assert cls in emulator_window.SHELL_CLASSES


def test_여러_앱플레이어_프로필이_등록돼_있다():
    names = {p.name for p in emulator_window.EMULATOR_PROFILES}
    assert {"MuMuPlayer", "LDPlayer", "NoxPlayer", "BlueStacks"} <= names


# --------------------------------------------- 격자 위상 안정성 (실측 회귀)
def test_창_가장자리에_달라붙은_배치는_격자로_보지_않는다():
    """한 칸 밀린 가짜 후보는 진짜와 점수 차이가 0.001 밖에 안 났다.

    실측: 진짜 xs=[76,184,292,400,507,615] (conf 0.780) 와
          가짜 xs=[184,292,400,507,615,708] (conf 0.779) 가 번갈아 나와,
          매크로가 매 사이클 한 칸씩 다른 곳을 클릭했다.
    가짜 쪽은 마지막 선이 클라이언트 폭(709)에 달라붙고 마지막 간격만 93px 로
    좁아진다. 점수를 매기기 전에 구조로 걸러낸다.
    """
    assert board._plausible([76, 184, 292, 400, 507, 615], 709, 108) is True
    assert board._plausible([184, 292, 400, 507, 615, 708], 709, 108) is False
    # 가장자리에 붙은 경우
    assert board._plausible([0, 108, 216, 324, 432, 540], 709, 108) is False
    # 간격이 들쭉날쭉한 경우
    assert board._plausible([76, 184, 292, 400, 507, 560], 709, 108) is False
    # 원근 때문에 조금 줄어드는 정도는 허용
    assert board._plausible([419, 507, 595, 684, 772, 851], 1260, 88) is True


def test_같은_화면을_여러_번_검출해도_격자가_같다():
    """같은 이미지에서는 당연히 같아야 하고, 실제로도 그랬어야 했다."""
    img = cv2.imread("tests/fixtures/explore_sample3.png")
    grids = [board.detect_board(img) for _ in range(5)]
    assert all(g is not None for g in grids)
    assert len({tuple(g.xs) for g in grids}) == 1
    assert len({tuple(g.ys) for g in grids}) == 1
    # 게임판은 창 가장자리에 붙지 않는다
    g = grids[0]
    assert g.xs[0] >= 2 and g.xs[-1] <= img.shape[1] - 3
    assert g.ys[0] >= 2 and g.ys[-1] <= img.shape[0] - 3
