import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from digimonup.vision import recognize


@pytest.fixture(autouse=True)
def no_user_templates(monkeypatch):
    """테스트는 사용자가 넣어둔 templates/explore 내용에 좌우되면 안 된다.

    기본적으로 빈 템플릿 세트로 돌려서 색 기반 인식 경로를 검증하고,
    템플릿이 필요한 테스트는 각자 직접 주입한다.
    """
    empty = {name: recognize.TemplateSet.__new__(recognize.TemplateSet)
             for name in ("player", "player_body", "goal", "obstacle",
                          "item", "top_tab", "blocked_toast", "green_button")}
    for name, tset in empty.items():
        tset.name = name
        tset.allow_flip = name.startswith("player")
        tset.images = []
        tset.paths = []
    monkeypatch.setattr(recognize, "load_templates", lambda: dict(empty))
    return empty


@pytest.fixture
def blank_templates(no_user_templates):
    return dict(no_user_templates)
