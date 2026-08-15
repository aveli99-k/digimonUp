"""셀 상태 인식 테스트.

플레이어는 애니메이션으로 계속 움직이고 좌우로 뒤집히며, 발이 아래 셀을
침범하거나 테두리/장애물에 가려지기도 한다. 그 모든 경우에 **논리적 셀 위치**가
흔들리지 않아야 한다.
"""

from __future__ import annotations

import cv2
import pytest

import board
import recognize
import synth
from recognize import Kind

LAYOUT = ["....." , ".P..X", "..X..", "....X", "....."]


def _scene(img, templates=None):
    g = board.detect_board(img)
    assert g is not None, "게임판 검출 실패"
    return g, recognize.analyze(img, g, templates or recognize.load_templates())


def _player_cell(img, templates=None):
    _, sc = _scene(img, templates)
    assert sc.player is not None, "플레이어 미검출"
    return sc.player.row, sc.player.col


# ------------------------------------------------------------ 플레이어 자세
def test_플레이어_기본_자세():
    assert _player_cell(synth.make_board(LAYOUT)) == (1, 1)


def test_좌우_반전_플레이어():
    img = synth.make_board(LAYOUT, player_kw={"flip": True})
    assert _player_cell(img) == (1, 1)


@pytest.mark.parametrize("dx,dy,scale", [(6, -4, 1.0), (-7, 3, 1.04),
                                         (3, 6, 0.96), (-4, -6, 1.02)])
def test_애니메이션_중인_플레이어(dx, dy, scale):
    """프레임마다 조금씩 흔들려도 같은 칸으로 인식돼야 한다."""
    img = synth.make_board(LAYOUT, player_kw={"dx": dx, "dy": dy, "scale": scale})
    assert _player_cell(img) == (1, 1)


def test_발이_셀_경계를_넘은_경우():
    """발끝이 아래 칸으로 넘어가도 논리 위치는 원래 칸이어야 한다."""
    img = synth.make_board(LAYOUT, player_kw={"foot_overflow": 0.18})
    assert _player_cell(img) == (1, 1)


def test_맨_아래_행_플레이어():
    layout = ["....." , "....X", "..X..", "....X", ".P..."]
    img = synth.make_board(layout)
    assert _player_cell(img) == (4, 1)


def test_맨_아래_행에서_발이_테두리에_가려진_경우():
    layout = ["....." , "....X", "..X..", "....X", ".P..."]
    img = synth.make_board(layout, player_kw={"occlude_bottom": 16})
    assert _player_cell(img) == (4, 1)


def test_위_장애물과_아래_테두리에_동시에_가려진_플레이어():
    layout = ["....." , "....X", "..X..", ".X..X", ".P..."]
    img = synth.make_board(layout, player_kw={"occlude_top": 22, "occlude_bottom": 14})
    assert _player_cell(img) == (4, 1)


def test_전체_인식_실패시_몸통_보조_인식이_동작한다():
    """전체 템플릿은 못 맞추고 몸통 템플릿만 맞는 상황."""
    img = synth.make_board(LAYOUT)
    g = board.detect_board(img)

    tpl = recognize.load_templates()
    # 절대 안 맞는 전체 템플릿(단색) + 실제 몸통을 잘라 만든 몸통 템플릿
    x0, y0, x1, y1 = g.cell_rect(1, 1)
    body = img[y0 + int(g.cell_h * 0.35):y0 + int(g.cell_h * 0.8),
               x0 + 30:x1 - 30].copy()

    # 절대 안 맞지만 무늬는 있는(분산이 0이 아닌) 가짜 전체 템플릿
    full = recognize.TemplateSet.__new__(recognize.TemplateSet)
    full.name, full.allow_flip, full.images, full.paths = "player", True, [
        synth.make_non_game_window()[80:200, 20:140].copy()], ["fake.png"]
    bodyset = recognize.TemplateSet.__new__(recognize.TemplateSet)
    bodyset.name, bodyset.allow_flip, bodyset.images, bodyset.paths = \
        "player_body", True, [body], ["body.png"]

    tpl["player"], tpl["player_body"] = full, bodyset
    det = recognize.detect_player(img, g, tpl, [])
    assert det is not None
    assert (det.row, det.col) == (1, 1)
    assert "몸통" in det.note


def test_몸통_보조_인식은_전체_인식이_성공하면_쓰지_않는다():
    img = synth.make_board(LAYOUT)
    g = board.detect_board(img)
    x0, y0, x1, y1 = g.cell_rect(1, 1)
    whole = img[y0 - int(g.cell_h * 0.3):y1, x0:x1].copy()

    tpl = recognize.load_templates()
    full = recognize.TemplateSet.__new__(recognize.TemplateSet)
    full.name, full.allow_flip, full.images, full.paths = \
        "player", True, [whole], ["p.png"]
    bodyset = recognize.TemplateSet.__new__(recognize.TemplateSet)
    bodyset.name, bodyset.allow_flip, bodyset.images, bodyset.paths = \
        "player_body", True, [whole[10:40, 10:40]], ["b.png"]
    tpl["player"], tpl["player_body"] = full, bodyset

    det = recognize.detect_player(img, g, tpl, [])
    assert det is not None and "전체 템플릿" in det.note


# ------------------------------------------------------------------ 목적지
def test_하단_테두리에_가려진_목적지():
    """맨 아래 행 목적지가 아래쪽이 잘려도 주황 카드 색으로 찾아낸다."""
    layout = ["....." , ".P...", "..X..", "....X", "...G."]
    img = synth.make_board(layout, goal_clip=20)
    g = board.detect_board(img)
    tpl = recognize.load_templates()
    sc = recognize.analyze(img, g, tpl, orange_goal_without_template=True)
    assert sc.goal is not None
    assert (sc.goal.row, sc.goal.col) == (4, 3)


def test_목적지_템플릿이_없으면_기본적으로_주황색을_목적지로_보지_않는다():
    """일반 아이템도 주황 카드라서, 템플릿 없이 추측하면 경로가 통째로 틀어진다."""
    layout = ["....." , ".P...", "..X..", ".....", "...i."]
    img = synth.make_board(layout)
    g = board.detect_board(img)
    sc = recognize.analyze(img, g, recognize.load_templates())
    assert sc.goal is None
    assert sc.cells[4][3] == Kind.ITEM


# ------------------------------------------------------------------ 장애물
def test_장애물_인식():
    layout = ["....." , ".P..X", "..X..", "....X", "....."]
    img = synth.make_board(layout)
    _, sc = _scene(img)
    got = {(r, c) for r in range(5) for c in range(5)
           if sc.cells[r][c] == Kind.OBSTACLE}
    assert got == {(1, 4), (2, 2), (3, 4)}


def test_위_셀로_삐져나온_피라미드_꼭대기를_장애물로_세지_않는다():
    """피라미드는 위 칸으로 삐져나온다. 그 칸까지 장애물로 세면 길이 막힌다."""
    layout = ["....." , ".P...", "....X", ".....", "....."]
    img = synth.make_board(layout)
    _, sc = _scene(img)
    assert sc.cells[2][4] == Kind.OBSTACLE
    assert sc.cells[1][4] != Kind.OBSTACLE, "위 칸까지 장애물로 오인했습니다"


def test_실제_캡처_인식_결과():
    """실제 MuMuPlayer 캡처에서 손으로 확인한 배치와 일치해야 한다."""
    img = cv2.imread("tests/fixtures/explore_sample1.png")
    g, sc = _scene(img)
    assert (sc.player.row, sc.player.col) == (1, 1)
    obstacles = {(r, c) for r in range(5) for c in range(5)
                 if sc.cells[r][c] == Kind.OBSTACLE}
    assert obstacles == {(1, 3), (2, 4), (3, 2), (3, 4)}
    assert sc.cells[4][3] == Kind.ITEM
    assert set(sc.highlights) == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_플레이어의_머리와_발이_걸친_칸을_아이템으로_세지_않는다():
    img = synth.make_board(LAYOUT, player_kw={"foot_overflow": 0.2})
    _, sc = _scene(img)
    assert sc.cells[0][1] != Kind.ITEM
    assert sc.cells[2][1] != Kind.ITEM


# --------------------------------------- 강조칸 역산은 어디까지나 보조
def test_강조칸이_십자가_아니면_역산하지_않는다():
    """실측 회귀: 강조칸이 [(0,0),(1,0),(1,1)] 로 잡힌 적이 있다.

    이 셋은 한 칸을 둘러싼 십자가 될 수 없는데도(예: (1,0)은 (0,1)의 대각선)
    예전 코드는 (0,1)을 플레이어로 반환했고, 템플릿이 맞게 찾은 (2,1)을
    그 값으로 덮어써 이동 확인이 계속 실패했다.
    """
    assert recognize._player_from_highlights([(0, 0), (1, 0), (1, 1)]) is None


def test_제대로_된_십자는_역산한다():
    assert recognize._player_from_highlights([(0, 1), (1, 0), (1, 2), (2, 1)]) == (1, 1)
    assert recognize._player_from_highlights([(1, 0), (1, 2)]) == (1, 1)
    # 위와 오른쪽만 보이는 부분 십자도 유효하다
    assert recognize._player_from_highlights([(0, 0), (1, 1)]) == (1, 0)


def test_강조칸이_하나뿐이면_역산하지_않는다():
    assert recognize._player_from_highlights([(1, 1)]) is None
    assert recognize._player_from_highlights([]) is None


def test_강조칸_역산이_이미지_인식_결과를_덮어쓰지_않는다():
    """이미지에서 실제로 찾은 위치가 항상 우선이어야 한다."""
    img = synth.make_board(LAYOUT)          # 플레이어는 (1,1)
    g = board.detect_board(img)
    tpl = recognize.load_templates()
    # 일부러 엉뚱한 칸을 가리키는 십자 힌트를 넣는다
    wrong_hint = [(3, 3), (4, 4)]
    det = recognize.detect_player(img, g, tpl, wrong_hint)
    assert det is not None
    assert (det.row, det.col) == (1, 1), "힌트가 이미지 결과를 덮어썼습니다"


def test_이미지_인식이_실패하면_강조칸_역산을_쓴다():
    """스프라이트를 못 찾았을 때는 십자 힌트라도 쓰는 게 낫다."""
    img = synth.make_board(["....." , ".....", ".....", ".....", "....."])
    g = board.detect_board(img)
    tpl = recognize.load_templates()
    det = recognize.detect_player(img, g, tpl, [(0, 1), (1, 0), (1, 2), (2, 1)])
    assert det is not None and (det.row, det.col) == (1, 1)
    assert "역산" in det.note


# ----------------------------- 실제 템플릿을 쓴 경로 (색 기반만으로는 못 잡는 회귀)
def _real_templates():
    """conftest 가 비워 둔 것과 달리, 저장소에 든 진짜 템플릿을 불러온다.

    다른 테스트는 색 기반 경로를 검증하려고 템플릿을 비운다. 그래서 템플릿
    경로에서만 나는 버그(정규화 배율, 상자 위치)를 놓친 적이 있어 따로 둔다.
    """
    import glob
    import os

    import paths
    out = {}
    for name in ("player", "player_body", "goal", "obstacle", "item",
                 "top_tab", "blocked_toast", "green_button"):
        t = recognize.TemplateSet.__new__(recognize.TemplateSet)
        t.name, t.allow_flip = name, name.startswith("player")
        t.paths = sorted(glob.glob(os.path.join(paths.EXPLORE_TEMPLATE_DIR,
                                                name, "*.png")))
        t.images = [im for im in (recognize.imread_bgr(p) for p in t.paths)
                    if im is not None]
        out[name] = t
    return out


@pytest.mark.parametrize("fixture", ["explore_sample1.png", "explore_sample2.png",
                                     "explore_sample3.png"])
def test_진짜_템플릿으로도_플레이어_칸이_맞는다(fixture):
    """템플릿 상자를 그대로 발 위치로 쓰면 셀이 한 칸 밀리는 회귀가 있었다.

    - 템플릿을 셀 높이의 1.15 배로 정규화해 너무 작게 줄인 탓에 점수가 0.44 로 떨어짐
    - 정규화를 고치자 이번엔 상자가 8px 아래로 잡혀 (1,1) 이 (2,1) 로 뒤집힘
    두 경우 모두 색 기반 경로만 돌리는 테스트로는 잡히지 않았다.
    """
    tpl = _real_templates()
    if not tpl["player"]:
        pytest.skip("templates/explore/player 가 비어 있습니다")
    img = cv2.imread(f"tests/fixtures/{fixture}")
    assert img is not None
    g = board.detect_board(img)
    sc = recognize.analyze(img, g, tpl)
    assert sc.player is not None
    assert (sc.player.row, sc.player.col) == (1, 1)


def test_진짜_템플릿으로도_전체_인식이_충분히_빠르다():
    """이동 확인 루프가 아니라 사이클마다 도는 경로라 1초를 넘기면 안 된다."""
    import time
    tpl = _real_templates()
    if not tpl["player"]:
        pytest.skip("templates/explore/player 가 비어 있습니다")
    img = cv2.imread("tests/fixtures/explore_sample3.png")
    g = board.detect_board(img)
    recognize.analyze(img, g, tpl)             # 워밍업
    t = time.time()
    recognize.analyze(img, g, tpl)
    elapsed = time.time() - t
    assert elapsed < 1.0, f"전체 인식이 {elapsed*1000:.0f}ms 걸립니다"


# ------------------------------- 안내문 검사 (이동 확인 루프의 진짜 병목이었다)
# explore_toast.png 는 실제 게임에서 '해당 위치로 이동할 수 없습니다' 가 떠 있을 때
# 찍은 프레임이다. 템플릿을 잘라낸 그 프레임이 아니라 **다른 프레임**을 골랐다.
# (템플릿 원본을 쓰면 점수가 정확히 1.0 이라 매칭을 시험하는 의미가 없다.)
TOAST_FIXTURE = "tests/fixtures/explore_toast.png"


def test_안내문_검사가_이동_확인_루프를_막지_않는다():
    """확인 루프는 폴링마다 이 검사를 돈다. 여기가 느리면 이동이 전부 실패한다.

    실측 회귀: 안내문 템플릿(478x77)을 709x1260 화면 전체에서 배율 3개 x
    템플릿 3장으로 훑느라 **한 번에 346ms** 가 걸렸다. move_timeout_sec 이
    2.2초라 확인 기회가 5~6번뿐이었고, 그중 2회 연속을 맞춰야 이동 성공이라
    한 번만 어긋나도 실패로 잡혔다. 플레이어 추적만 빠르게 만들어서는
    해결되지 않는 문제였다.
    """
    import time
    tpl = _real_templates()
    if not tpl["blocked_toast"]:
        pytest.skip("templates/explore/blocked_toast 가 비어 있습니다")
    img = cv2.imread("tests/fixtures/explore_sample3.png")
    recognize.find_blocked_toast(img, tpl["blocked_toast"])      # 워밍업
    t = time.time()
    for _ in range(3):
        recognize.find_blocked_toast(img, tpl["blocked_toast"])
    elapsed = (time.time() - t) / 3
    assert elapsed < 0.15, f"안내문 검사가 {elapsed*1000:.0f}ms 걸립니다"


def test_2단계_매칭이_안내문을_놓치지_않는다():
    """속도를 위해 축소본으로 선별하더라도 판정은 그대로여야 한다.

    실제 게임 프레임 두 장으로 확인한다. 안내문이 뜬 프레임과 안 뜬 프레임이
    toast_min(0.65)을 사이에 두고 확실히 갈려야 한다.
    """
    tpl = _real_templates()
    if not tpl["blocked_toast"]:
        pytest.skip("templates/explore/blocked_toast 가 비어 있습니다")
    toasted = cv2.imread(TOAST_FIXTURE)
    clean = cv2.imread("tests/fixtures/explore_sample3.png")
    assert toasted is not None and clean is not None

    on, _ = recognize.find_blocked_toast(toasted, tpl["blocked_toast"])
    off, _ = recognize.find_blocked_toast(clean, tpl["blocked_toast"])
    assert on >= 0.65, f"안내문이 떠 있는데 점수가 {on:.3f} 입니다"
    assert off < 0.65, f"안내문이 없는데 점수가 {off:.3f} 입니다"
    # 축소 선별이 아슬아슬하게 통과하는 게 아니라 여유 있게 갈리는지도 본다.
    assert on - off > 0.15, f"안내문 유무의 점수 차가 {on - off:.3f} 뿐입니다"


def test_2단계_매칭_점수가_전체_해상도_매칭과_같다():
    """축소 선별은 위치만 좁히고, 점수는 원본 해상도에서 재야 한다.

    그래야 config.json 의 toast_min 같은 기준을 그대로 쓸 수 있다.

    허용 오차를 0 이 아니라 1e-4 로 둔 이유: matchTemplate 은 내부적으로 float32 이고,
    OpenCV 가 탐색 영역 크기에 따라 공간 상관과 DFT 중 하나를 골라 쓴다. 그래서
    같은 자리라도 마지막 자릿수가 조금 다르다(실측 1.0e-6). 판정 기준끼리의 간격은
    0.05 단위라 이 정도 오차는 판단을 바꾸지 못한다.
    """
    tpl = _real_templates()
    if not tpl["blocked_toast"]:
        pytest.skip("templates/explore/blocked_toast 가 비어 있습니다")
    img = cv2.imread(TOAST_FIXTURE)
    scales = (0.85, 1.0, 1.15)
    want, wbox, _ = recognize.match_best(img, tpl["blocked_toast"], scales=scales)
    got, gbox, _ = recognize.match_big(img, tpl["blocked_toast"], scales=scales)
    assert abs(got - want) < 1e-4, f"점수가 달라졌습니다 {want:.4f} -> {got:.4f}"
    assert gbox == wbox, f"위치가 달라졌습니다 {wbox} -> {gbox}"


def test_템플릿_리사이즈는_캐시된다():
    """셀 25칸을 훑을 때 같은 리사이즈를 25번 반복하고 있었다."""
    tpl = _real_templates()
    if not tpl["obstacle"]:
        pytest.skip("templates/explore/obstacle 이 비어 있습니다")
    a = tpl["obstacle"].prepared(88, (0.85, 1.0), 999, 999)
    b = tpl["obstacle"].prepared(88, (0.85, 1.0), 999, 999)
    assert [id(im) for im, _ in a] == [id(im) for im, _ in b], "매번 새로 만들고 있습니다"
    # 화면보다 큰 템플릿은 걸러서 낸다
    small = tpl["obstacle"].prepared(88, (0.85, 1.0), 10, 10)
    assert small == []


def test_빠른_추적은_템플릿을_쓰지_않아_아주_빠르다():
    """이동 확인은 짧은 주기로 계속 부르는 자리라 속도가 전부다.

    예전에는 여기서 템플릿 매칭을 돌려 한 번에 1~2초가 걸렸고, 2.2초 제한 안에
    '연속 2회 확인'이 불가능해 모든 이동이 실패로 잡혔다.
    """
    import time
    img = cv2.imread("tests/fixtures/explore_sample3.png")
    g = board.detect_board(img)
    m_high = recognize.mask_highlight(img)
    highlights = [(r, c) for r in range(5) for c in range(5)
                  if recognize._frac(m_high, g.cell_rect(r, c)) > 0.40]
    recognize.track_player_fast(img, g, highlights)      # 워밍업
    t = time.time()
    for _ in range(5):
        pos = recognize.track_player_fast(img, g, highlights)
    elapsed = (time.time() - t) / 5
    assert pos == (1, 1)
    assert elapsed < 0.10, f"빠른 추적이 {elapsed*1000:.0f}ms 걸립니다"
