"""이미지 파일 입출력.

`cv2.imread` / `cv2.imwrite` 는 Windows 에서 **한글이 든 경로를 못 읽고 못 쓴다.**
이 프로젝트의 경로에는 한글이 들어 있어서(바탕화면/준/project/...) 전부 우회해야 한다.
그 우회 코드가 common / recognize / overlay / capture 네 곳에 각각 복사돼 있었고,
알파 채널 정규화까지 두 벌로 갈라져 있어서 여기로 모았다.

cv2 와 numpy 에만 의존한다. 그래야 인식 모듈(recognize)이 화면 캡처용 PIL 까지
끌어오지 않고 이 파일만 가져다 쓸 수 있다.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """한글 경로에서도 읽히는 cv2.imread. 실패하면 None."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imread_bgr(path: str) -> np.ndarray | None:
    """읽어서 **무조건 3채널 BGR** 로 맞춰 돌려준다.

    PNG 에 알파 채널이 있으면 투명한 부분이 템플릿 매칭을 방해하고, 흑백 PNG 는
    채널 수가 달라 matchTemplate 이 아예 실패한다. 템플릿은 전부 이걸로 읽는다.
    """
    img = imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def imwrite(path: str, img: np.ndarray) -> bool:
    """한글 경로에도 저장되는 cv2.imwrite. 상위 폴더가 없으면 만든다."""
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except (OSError, cv2.error):
        return False
