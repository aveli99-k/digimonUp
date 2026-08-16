"""왼쪽 아래 아이템 개수 읽기 — **실제 캡처**로만 검증한다.

합성 화면을 쓰지 않는다. 여기서 갈리는 것이 '충전 타이머가 아이콘에 달라붙어
한 덩어리가 되는가' 인데, 그건 실제 화면의 채도 분포에서만 일어난다.

fixture 세 장의 진짜 값은 화면을 눈으로 읽어 확인한 것이다.

    explore_items.png          걸음수 1,705 / 부수기 231 / 돌진 25   (타이머 없음)
    explore_sample1.png        걸음수     7 / 부수기 235 / 돌진 30
                               -> **맨 위**(걸음수) 줄 왼쪽에 '05:32' 타이머
    counters_dash_timer.png    걸음수 1,708 / 부수기  65 / 돌진  0
                               -> **맨 아래**(돌진) 줄 왼쪽에 '30:20' 타이머

가려지는 줄이 위일 수도 아래일 수도 있다는 것이 요점이다. 예전 코드는 '아래가
가려진다'고 단정하고 간격으로 아래에 한 줄을 만들어 붙였고, 그래서 위가 가려진
캡처에서 세 값이 통째로 한 줄씩 밀렸다.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from digimonup.base.imgio import imread
from digimonup.vision import counters

ITEMS = "tests/fixtures/explore_items.png"
STEPS_TIMER = "tests/fixtures/explore_sample1.png"
DASH_TIMER = "tests/fixtures/counters_dash_timer.png"


def _read(path):
    img = imread(path)
    assert img is not None, f"캡처를 못 읽었습니다: {path}"
    return img, counters.read(img)


@pytest.mark.skipif(not counters.load_digits(),
                    reason="templates/counters 의 숫자 본보기가 없습니다")
@pytest.mark.parametrize("path, want", [
    (ITEMS, (1705, 231, 25)),
    (STEPS_TIMER, (7, 235, 30)),
    (DASH_TIMER, (1708, 65, 0)),
])
def test_실제_캡처의_개수를_그대로_읽는다(path, want):
    _img, got = _read(path)
    assert (got.steps, got.break_, got.dash) == want, \
        f"{path}: 화면에 보이는 값과 다릅니다 ({got.describe()})"


@pytest.mark.skipif(not counters.load_digits(),
                    reason="templates/counters 의 숫자 본보기가 없습니다")
def test_맨_위_줄에_타이머가_붙어도_값이_밀리지_않는다():
    """실측 회귀: 걸음수 7 을 235 로, 부수기 235 를 30 으로 읽었다.

    걸음수 줄이 바닥나 왼쪽에 충전 타이머('05:32', 분홍)가 떴다. 채도가 높아
    아이콘과 한 덩어리가 되고 가로로 길어져 걸러졌다. 그러면 줄이 둘만 남는데,
    예전 코드는 그 둘을 '위 둘'로 보고 아래에 한 줄을 지어 붙였다. 실제로는
    그 둘이 **아래 둘**(부수기·돌진)이었으므로 값이 통째로 밀렸다.

    걸음수 7 을 235 로 읽으면 stop_when_out_of_steps 가 영영 안 걸린다 —
    걸음수가 바닥난 채로 계속 클릭하며 실패만 쌓는다. 이 모듈을 만든 이유가
    바로 그것을 막는 것이었다.
    """
    _img, got = _read(STEPS_TIMER)
    assert got.steps == 7, f"걸음수가 밀렸습니다: {got.describe()}"
    assert got.break_ == 235
    assert got.dash == 30
    assert got.dash is not None, "돌진이 '모름'이면 _can_use 가 영영 False 다"


@pytest.mark.skipif(not counters.load_digits(),
                    reason="templates/counters 의 숫자 본보기가 없습니다")
def test_맨_아래_줄에_타이머가_붙어도_읽는다():
    """30장에서 처음 잡힌 경우. 이쪽은 예전 코드도 우연히 맞았다."""
    _img, got = _read(DASH_TIMER)
    assert (got.steps, got.break_, got.dash) == (1708, 65, 0)


def test_타이머가_붙은_덩어리에서_아이콘만_떼어_낸다():
    """타이머는 **아이콘 왼쪽**에 붙는다. 오른쪽 끝 정사각형이 아이콘이다."""
    icon = counters._icon_in_merged((100, 1062, 118, 41))
    assert icon == (100 + 118 - 41, 1062, 41, 41)
    # 되살린 상자는 원래 덩어리 안에 들어 있어야 한다.
    x, y, w, h = icon
    assert x >= 100 and x + w == 100 + 118


def test_세_줄이_다_보이면_가로로_긴_덩어리는_그냥_버린다():
    """멀쩡한 세 줄에 가짜를 하나 더 얹지 않는다."""
    img = imread(ITEMS)
    rows = counters.find_rows(img)
    assert len(rows) == 3
    ys = [r[1] for r in rows]
    assert ys == sorted(ys), "위에서부터 나와야 합니다"
    gaps = [ys[1] - ys[0], ys[2] - ys[1]]
    assert abs(gaps[0] - gaps[1]) <= 6, f"세 줄은 등간격이어야 합니다: {gaps}"


def test_되살려도_세_줄이_안_되면_모른다고_답한다():
    """**엉뚱한 숫자를 읽느니 모른다고 하는 편이 안전하다** (이 모듈의 규약).

    예전에는 여기서 없는 줄을 지어냈고, 그것이 위 회귀의 원인이었다.
    """
    img = imread(ITEMS).copy()
    rows = counters.find_rows(img)
    assert len(rows) == 3
    # 아래 두 줄의 아이콘을 주변 패널 색으로 덮어 채도를 없앤다.
    panel = img[rows[0][1] - 30, rows[0][0]].tolist()
    for x, y, w, h in rows[1:]:
        cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6),
                      [int(v) for v in panel], -1)
    left = counters.find_rows(img)
    assert len(left) < 3, f"덮었는데도 세 줄이 나왔습니다: {left}"
    got = counters.read(img)
    assert (got.steps, got.break_, got.dash) == (None, None, None), \
        f"줄을 못 찾았는데 숫자를 지어냈습니다: {got.describe()}"


def test_빈_화면에서는_아무것도_찾지_않는다():
    assert counters.find_rows(None) == []
    assert counters.find_rows(np.zeros((0, 0, 3), np.uint8)) == []
    blank = np.full((1260, 709, 3), 200, np.uint8)      # 채도 0 인 회색
    assert counters.find_rows(blank) == []
    got = counters.read(blank)
    assert (got.steps, got.break_, got.dash) == (None, None, None)
