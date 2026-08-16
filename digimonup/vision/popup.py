"""판을 가리는 팝업(실패창/보상창)을 알아보고 닫는다. **세 기능이 함께 쓴다.**

왜 공통으로 두는가
    실패창과 보상창은 던전에서만 뜨는 것이 아니다. 어떤 기능을 돌리든 판이
    끝나면 그 위에 뜨고, 그동안 아래 화면은 클릭을 먹지 않는다. 그런데 탐사와
    네트워크는 그걸 모르니 계속 헛클릭을 하며 시간을 버린다.

    던전 기능에 이미 검증된 처리가 있었다(3번 기능). 그 규칙을 여기로 옮겨
    세 기능이 같은 코드를 쓰게 한다.

닫는 규칙 (던전에서 실측으로 정한 것 그대로)
    닫기 버튼이 없다. 어두워진 **바깥을 한 번** 누르면 닫힌다. 실측으로 둘 다
    클라이언트 (60, 60) 한 번에 닫혔다.

    **왼쪽 위 구석을 누르는 것이 중요하다.** 보상창 아래쪽에는 '포기' 버튼이
    같이 보이는데, 거기를 잘못 누르면 보상을 못 받는다. 화면에서 그 버튼과
    가장 먼 자리를 고른 것이다.

    한 번 누르고 나면 잠시 기다린다. 화면이 아직 안 바뀌었다고 연달아 누르면
    팝업이 닫히는 순간의 클릭이 그 아래 화면으로 전달돼 엉뚱한 것을 누른다.
"""

from __future__ import annotations

import numpy as np

from digimonup.base.paths import DUNGEON_TEMPLATE_DIR
from digimonup.vision.recognize import TemplateSet, match_big

# 팝업 종류: (한글 이름, 템플릿 폴더, 글자가 있는 세로 범위, 최소 유사도)
#
# 세로 범위를 두는 이유는 화면 전체를 훑지 않으려는 것이다. 실패창 글자는 위쪽,
# 보상창 글자는 가운데에 뜬다.
POPUPS: dict[str, tuple[str, tuple[float, float], float]] = {
    "fail": ("실패창", (0.02, 0.32), 0.70),
    "reward": ("보상창", (0.12, 0.45), 0.70),
}

# 팝업을 닫으려고 누를 자리 (화면 크기 대비 비율). 왼쪽 위 구석이다.
CLOSE_POINT = (0.085, 0.048)

_templates: dict[str, TemplateSet] | None = None


def load_templates() -> dict[str, TemplateSet]:
    """팝업 템플릿. 없으면 빈 것을 돌려주고, 그러면 아무것도 찾지 않는다."""
    global _templates
    if _templates is None:
        _templates = {k: TemplateSet(k, base_dir=DUNGEON_TEMPLATE_DIR)
                      for k in POPUPS}
    return _templates


def find(img: np.ndarray, templates: dict[str, TemplateSet] | None = None,
         use_band: bool = True):
    """지금 팝업이 떠 있는가. (종류, 점수, 글자상자) 또는 None.

    글자상자는 **입력 이미지 기준** 좌표다.

    use_band
        True 면 글자가 뜨는 띠만 잘라서 본다(게임 창을 그대로 캡처한 경우).
        화면 전체를 훑는 것보다 빠르고 다른 곳의 비슷한 무늬에 속지도 않는다.
        False 면 이미지 전체를 본다. 모니터 전체를 캡처해 게임 창의 위치를
        모르는 경우(1번 기능)에 쓴다.
    """
    if img is None or img.size == 0:
        return None
    tsets = templates if templates is not None else load_templates()
    h = img.shape[0]
    for kind, (_, band, min_score) in POPUPS.items():
        tset = tsets.get(kind)
        if not tset:
            continue
        top = max(0, int(h * band[0])) if use_band else 0
        bottom = min(h, int(h * band[1])) if use_band else h
        if bottom <= top:
            continue
        score, box, _ = match_big(img[top:bottom], tset, scales=(0.85, 1.0, 1.15))
        if score >= min_score and box is not None:
            return kind, float(score), (box[0], box[1] + top, box[2], box[3] + top)
    return None


def close_point(client_w: int, client_h: int) -> tuple[int, int]:
    """팝업을 닫으려고 누를 클라이언트 좌표."""
    return int(client_w * CLOSE_POINT[0]), int(client_h * CLOSE_POINT[1])


def name_of(kind: str) -> str:
    return POPUPS[kind][0] if kind in POPUPS else kind


def close_point_for_box(box: tuple[int, int, int, int], w: int, h: int,
                        margin: int = 40) -> tuple[int, int]:
    """글자상자를 보고 팝업 **바깥**을 누를 자리를 정한다.

    게임 창의 위치를 모를 때 쓴다(모니터 전체를 캡처하는 1번 기능).
    팝업의 왼쪽 위 바깥을 고른다 — 보상창 아래쪽의 '포기' 버튼에서 가장 먼
    자리이기 때문이다. 화면 밖으로 나가지 않도록 잘라 준다.
    """
    x0, y0, _, _ = box
    return (max(margin, min(w - margin, x0 - margin)),
            max(margin, min(h - margin, y0 - margin)))
