"""판을 가리는 팝업(실패창/보상창)을 세 기능이 함께 알아보는가."""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from digimonup.base import imgio
from digimonup.vision import popup

FIX = "tests/fixtures"


def _fx(name):
    path = os.path.join(FIX, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} 이 없습니다")
    return imgio.imread_bgr(path)


def test_실패창을_알아본다():
    got = popup.find(_fx("dungeon_fail.png"))
    assert got is not None
    kind, score, box = got
    assert kind == "fail" and score >= 0.70
    assert box[2] > box[0] and box[3] > box[1]


def test_첫_줄이_달라도_알아본다():
    """실측: '던전 실패...' 와 '스테이지 실패...' 두 가지가 있다.

    바뀌지 않는 둘째 줄만 보므로 둘 다 잡힌다.
    """
    got = popup.find(_fx("dungeon_fail_stage.png"))
    assert got is not None and got[0] == "fail"


def test_팝업이_없는_화면에서는_찾지_않는다():
    """탐사 화면에서 팝업을 잘못 잡으면 멀쩡한 진행을 멈춘다."""
    for path in sorted(glob.glob(os.path.join(FIX, "explore_*.png"))):
        img = imgio.imread_bgr(path)
        assert popup.find(img) is None, f"{os.path.basename(path)} 에서 오탐"


def test_띠를_안_써도_찾는다():
    """1번 기능은 모니터 전체를 캡처해 게임 창 위치를 모른다."""
    got = popup.find(_fx("dungeon_fail.png"), use_band=False)
    assert got is not None and got[0] == "fail"


def test_닫을_자리는_왼쪽_위_바깥이다():
    """보상창 아래쪽 '포기' 버튼에서 가장 먼 자리를 골라야 한다."""
    w, h = 709, 1260
    x, y = popup.close_point(w, h)
    assert x < w * 0.2 and y < h * 0.2


def test_상자를_보고_정한_자리도_팝업_바깥이다():
    box = (238, 166, 468, 240)
    x, y = popup.close_point_for_box(box, 1920, 1080)
    assert x < box[0] and y < box[1], "글자 위를 누르면 안 됩니다"
    assert 0 < x < 1920 and 0 < y < 1080, "화면 밖으로 나가면 안 됩니다"


def test_상자가_구석에_있어도_화면_안을_누른다():
    x, y = popup.close_point_for_box((5, 5, 60, 40), 800, 600)
    assert 0 < x < 800 and 0 < y < 600
