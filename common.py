"""digimonUp 매크로 공용 유틸 (화면 캡처 / 템플릿 매칭 / DPI 처리)."""

from __future__ import annotations

import ctypes
import json
import os
import sys

import cv2
import numpy as np
from PIL import ImageGrab

from paths import BASE_DIR, CONFIG_PATH  # noqa: F401  (기존 import 경로 유지)


def enable_dpi_awareness() -> None:
    """디스플레이 배율(125%, 150% 등)이 켜져 있어도 좌표가 어긋나지 않게 한다."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("match_template", "giveup_template"):
        cfg[key] = os.path.join(BASE_DIR, cfg[key].replace("/", os.sep))
    if cfg.get("log_file"):
        cfg["log_file"] = os.path.join(BASE_DIR, cfg["log_file"])
    return cfg


def grab_screen(region=None) -> np.ndarray:
    """화면을 BGR numpy 배열로 캡처. region = [left, top, width, height] 또는 None(전체)."""
    if region:
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)
    else:
        bbox = None
    img = ImageGrab.grab(bbox=bbox, all_screens=False)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """cv2.imread 는 Windows 에서 한글 경로를 못 읽으므로 우회한다."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite 의 한글 경로 우회 버전."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def load_template(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"템플릿 이미지가 없습니다: {path}\n"
            f"먼저 capture.py 를 실행해서 버튼 이미지를 잘라 저장하세요."
        )
    tpl = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        raise ValueError(f"템플릿 이미지를 읽을 수 없습니다: {path}")

    # PNG 에 알파 채널이 있으면 투명 영역이 매칭을 방해하므로 BGR 로 통일한다.
    if tpl.ndim == 2:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    elif tpl.shape[2] == 4:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
    return tpl


def _match_once(screen_gray: np.ndarray, tpl_gray: np.ndarray):
    th, tw = tpl_gray.shape[:2]
    sh, sw = screen_gray.shape[:2]
    if th > sh or tw > sw:
        return 0.0, None, (tw, th)
    res = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), max_loc, (tw, th)


def _scaled(tpl_gray: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return tpl_gray
    nw = max(1, int(round(tpl_gray.shape[1] * s)))
    nh = max(1, int(round(tpl_gray.shape[0] * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(tpl_gray, (nw, nh), interpolation=interp)


def find_template(screen: np.ndarray, tpl: np.ndarray, confidence: float,
                  multi_scale: bool = False, scales=None, lock: dict | None = None,
                  resweep_after: int = 25):
    """화면에서 템플릿을 찾는다.

    lock 에 dict 를 넘기면 '전체 배율을 훑어 확정한' 최적 배율을 기억해서, 다음
    호출부터는 그 배율 하나만 검사한다(7배 이상 빠름). 그 배율로 연속
    resweep_after 회 실패하면 창 크기가 바뀐 것일 수 있으므로 다시 전체를 훑는다.

    반환: (found: bool, center_xy: (x, y) | None, score: float)
    center_xy 는 screen 이미지 기준 좌표(= region 을 쓰면 region 내부 상대 좌표).
    """
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    all_scales = list(scales) if (multi_scale and scales) else [1.0]

    # 1차: 이미 확정된 배율이 있으면 그것만 빠르게 확인한다.
    locked = lock.get("scale") if lock else None
    if locked is not None and locked in all_scales:
        score, loc, size = _match_once(screen_gray, _scaled(tpl_gray, locked))
        if loc is not None and score >= confidence:
            lock["miss"] = 0
            return True, (loc[0] + size[0] // 2, loc[1] + size[1] // 2), score
        lock["miss"] = lock.get("miss", 0) + 1
        if lock["miss"] < resweep_after:
            return False, None, score
        # 오래 못 찾았다 -> 배율이 달라졌을 수 있으니 아래에서 전체 재탐색
        lock["miss"] = 0

    # 2차: 전체 배율 탐색 (최적 배율을 확정하는 경로)
    best = (0.0, None, (0, 0), None)
    for s in all_scales:
        score, loc, size = _match_once(screen_gray, _scaled(tpl_gray, s))
        if score > best[0]:
            best = (score, loc, size, s)

    score, loc, (tw, th), used = best
    if loc is None or score < confidence:
        return False, None, score
    if lock is not None and used is not None:
        lock["scale"] = used
        lock["miss"] = 0
    return True, (loc[0] + tw // 2, loc[1] + th // 2), score


def is_stop_key_pressed(vk_code: int) -> bool:
    """전역 단축키 감지 (창 포커스와 무관)."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


VK_CODES = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75,
    "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "ESC": 0x1B, "END": 0x23, "PAUSE": 0x13,
}


def vk_of(name: str) -> int:
    return VK_CODES.get(str(name).upper(), 0x7B)


def ensure_windows() -> None:
    if sys.platform != "win32":
        raise SystemExit("이 매크로는 Windows 전용입니다.")
