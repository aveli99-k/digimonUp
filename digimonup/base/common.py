"""digimonUp 매크로 공용 유틸 (화면 캡처 / 템플릿 매칭 / DPI 처리)."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading

import cv2
import numpy as np
from PIL import ImageGrab

from digimonup.base.imgio import imread_bgr
from digimonup.base.paths import BASE_DIR, CONFIG_PATH, resource
# DPI 처리는 emulator_window 에 정본이 있다. 예전에는 여기에도 같은 함수가 한 벌 더
# 있었는데, 한쪽만 고치면 다른 쪽이 조용히 어긋나므로 하나로 합쳤다.
from digimonup.win.emulator_window import enable_dpi_awareness  # noqa: F401  (여기서 계속 가져다 쓴다)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    # 템플릿은 읽기만 하므로 resource() 로 찾는다. EXE 옆에 있으면 그것을,
    # 없으면 EXE 안에 넣어 둔 기본값을 쓴다(파일 하나로도 돌아가게).
    for key in ("match_template", "giveup_template"):
        cfg[key] = resource(*cfg[key].replace("\\", "/").split("/"))
    # 로그는 쓰는 파일이라 항상 EXE 옆이다.
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


def load_template(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"템플릿 이미지가 없습니다: {path}\n"
            f"먼저 tools/capture.py 를 실행해서 버튼 이미지를 잘라 저장하세요."
        )
    tpl = imread_bgr(path)
    if tpl is None:
        raise ValueError(f"템플릿 이미지를 읽을 수 없습니다: {path}")
    return tpl


def load_button_templates(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """1번 기능이 쓰는 '매칭'/'포기' 버튼 템플릿 두 장.

    설정 열쇠 이름을 아는 곳을 한 군데로 둔다. 네트워크 매크로와 점검 도구가
    각자 `cfg["match_template"]` 를 적고 있었는데, 열쇠 이름이 바뀌면 한쪽만
    고쳐진다. 없으면 FileNotFoundError, 못 읽으면 ValueError 가 그대로 올라온다.
    """
    return (load_template(cfg["match_template"]),
            load_template(cfg["giveup_template"]))


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


class Stopped(Exception):
    """정지 요청. 진행 중인 모든 동작을 즉시 접는다.

    엔진마다 따로 두지 않는다. 정지의 의미가 기능마다 다를 이유가 없고,
    두 벌이 되면 한쪽에서 던진 것을 다른 쪽이 못 잡는 일이 생긴다.
    """


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


class StoppableEngine:
    """정지 요청과 중지 키를 함께 보는 부분. 기능 엔진들이 물려받는다.

    탐사(2번)와 던전(3번)에 **글자 하나까지 같은 코드**가 한 벌씩 있었다.
    정지의 뜻이 기능마다 다를 이유가 없는데도 두 벌이면 한쪽만 고쳐진다.
    `Stopped` 예외를 이미 한 벌로 모아 둔 것과 같은 이유로 여기로 합친다.

    게임을 전혀 모른다(스레드 이벤트와 키 코드만 안다). 그래서 base 에 둔다.
    """

    def __init__(self, stop_key: str = "", log=print):
        self.log = log
        self.stop_event = threading.Event()
        self.stop_key = str(stop_key or "")
        # 중지 키의 가상 키 코드. 빈 문자열이면 0 이라 키 검사를 건너뛴다.
        self._stop_vk = vk_of(self.stop_key) if self.stop_key else 0

    def stop(self) -> None:
        self.stop_event.set()

    def _check_stop(self) -> None:
        """정지가 걸렸으면 즉시 예외로 빠져나온다.

        클릭 직전, 분석 직후 등 모든 갈림길에서 호출한다. 덕분에 '분석 중에
        정지를 눌렀는데 분석이 끝난 뒤 이전 경로가 뒤늦게 실행되는' 일이 없다.

        중지 키(기본 F12)도 여기서 함께 본다. 매크로가 마우스를 계속 움직이는
        중에는 GUI 의 정지 버튼을 겨냥해서 누르기가 까다롭기 때문이다.
        전역 감지라 게임 창이 앞에 있어도 먹는다.
        """
        if self.stop_event.is_set():
            raise Stopped()
        if self._stop_vk and is_stop_key_pressed(self._stop_vk):
            # 한 번 감지하면 stop_event 를 세워 둔다. 키에서 손을 떼도 계속
            # 정지 상태이고, GUI 도 같은 깃발을 보므로 상태가 어긋나지 않는다.
            self.stop_event.set()
            self.log(f"[정지] {self.stop_key} 키를 눌러 중단합니다.")
            raise Stopped()

    def _sleep(self, sec: float) -> None:
        """정지 요청에 즉시 반응하는 sleep."""
        if self.stop_event.wait(sec):
            raise Stopped()

    def log_stop_hint(self) -> None:
        """멈추는 방법을 한 줄로 알린다. 중지 키가 없으면 아무것도 안 한다."""
        if self._stop_vk:
            self.log(f"[정지] 멈추려면 GUI 의 정지 버튼 또는 "
                     f"{self.stop_key} 키를 누르세요 (어느 창에서든 먹습니다).")


def ensure_windows() -> None:
    if sys.platform != "win32":
        raise SystemExit("이 매크로는 Windows 전용입니다.")
