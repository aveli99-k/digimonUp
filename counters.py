"""왼쪽 아래 아이템 개수 읽기 (걸음수 / 부수기 / 돌진).

게임 화면 왼쪽 아래에는 남은 아이템 개수가 세 줄로 표시된다.

    분홍 발자국   1,824   걸음수  - 한 칸 움직일 때마다 준다
    노란 발톱       234   부수기  - 장애물을 부술 때 쓴다
    초록 돌진        27   돌진    - 우측 하단 버튼으로 쓴다

이걸 읽어야 하는 이유
    지금까지 매크로는 '해 보고 안 되면 포기'하는 식이었다. 장애물을 클릭해 보고
    안내문이 뜨면 실패로 세고, 두 번 연속 실패하면 접었다. 그러면 **쓸 수 있는
    아이템이 애초에 0개일 때도 두 번은 헛클릭**하고, 안내문이 사라지길 기다리느라
    시간을 버린다. 개수를 먼저 보면 아예 시도하지 않는다.

    걸음수가 0이면 더 움직일 수 없으므로 매크로를 멈춰야 한다. 모르면 계속
    클릭하면서 '이동 확인 실패'만 쌓는다.

읽는 방법
    OCR 라이브러리를 쓰지 않는다. 배포본이 EXE 하나여야 하는데 OCR 엔진을 넣으면
    수십 MB 가 늘고 설치도 까다롭다. 대신 이 게임 글꼴의 숫자 모양이 고정이라는
    점을 이용해 **숫자 하나하나를 템플릿으로 맞춘다.**

      1. 왼쪽 아래에서 채도 높은 덩어리 세 개(아이콘)를 찾아 각 줄의 위치를 잡는다
         - 창 크기가 달라져도 아이콘을 찾아 상대 위치로 계산하므로 좌표를 박지 않는다
      2. 아이콘 오른쪽의 숫자 띠를 잘라 어두운 픽셀만 남긴다
      3. 붙어 있는 덩어리로 글자를 나눈다 (쉼표는 높이가 낮아 구분된다)
      4. 각 글자를 templates/counters/0~9.png 와 맞춰 읽는다

    자신 없는 글자가 하나라도 있으면 그 줄은 **None(모름)** 으로 돌려준다.
    엉뚱한 숫자를 읽느니 모른다고 하는 편이 안전하다. 모르면 엔진은 예전처럼
    '해 보고 판단하는' 방식으로 움직인다.

숫자 템플릿이 없으면 이 모듈은 조용히 None 만 돌려준다. 즉 **없어도 동작에는
지장이 없고**, 넣으면 헛클릭이 사라진다. 만드는 법은 tools/capture_counters.py 참고.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np

from imgio import imread_bgr
from paths import resource

DIGIT_DIR = resource("templates", "counters")

# 아이콘을 찾을 영역 (클라이언트 크기 대비). 왼쪽 아래 구석이다.
BAND_X = (0.0, 0.36)
BAND_Y = (0.80, 0.97)

ICON_MIN_AREA = 400        # 이보다 작은 덩어리는 아이콘이 아니다 (반짝임 등)
DIGIT_MIN_AREA = 12
# 글자 높이를 띠 높이로 나눈 값의 허용 범위. 실측(709x1260 화면):
#   숫자 13~14px / 띠 40px = 0.33~0.35
#   쉼표  6px    / 40      = 0.15   <- 아래로 걸러진다
#   오른쪽 원형 버튼 조각 33px / 40 = 0.83  <- 위로 걸러진다
DIGIT_H_RATIO = (0.25, 0.60)
DIGIT_MATCH_MIN = 0.72     # 이 아래면 '모르는 글자'로 취급한다
STRIP_W_RATIO = 3.2        # 숫자 띠 폭 / 아이콘 폭. 실측: 네 자리 수가 1.4배

ROW_NAMES = ("steps", "break", "dash")     # 위에서부터 걸음수 / 부수기 / 돌진
ROW_LABELS = {"steps": "걸음수", "break": "부수기", "dash": "돌진"}


@dataclass
class Counters:
    """읽어낸 개수. 못 읽은 항목은 None 이다."""
    steps: int | None = None
    break_: int | None = None
    dash: int | None = None

    def get(self, name: str) -> int | None:
        return {"steps": self.steps, "break": self.break_, "dash": self.dash}[name]

    def describe(self) -> str:
        parts = []
        for key in ROW_NAMES:
            v = self.get(key)
            parts.append(f"{ROW_LABELS[key]} {'?' if v is None else v}")
        return " / ".join(parts)


# --------------------------------------------------------------------------
# 숫자 템플릿
# --------------------------------------------------------------------------

_digits: dict[str, np.ndarray] | None = None


def load_digits(force: bool = False) -> dict[str, np.ndarray]:
    """templates/counters/<숫자>.png 를 불러 둔다. 한 번만 읽는다."""
    global _digits
    if _digits is not None and not force:
        return _digits
    out: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(DIGIT_DIR, "*.png"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if len(name) != 1 or not name.isdigit():
            continue
        img = imread_bgr(path)
        if img is not None and img.size:
            out[name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _digits = out
    return out


# --------------------------------------------------------------------------
# 줄 찾기
# --------------------------------------------------------------------------

def find_rows(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """왼쪽 아래 아이콘 세 개의 bbox 를 위에서부터 돌려준다 (클라이언트 좌표).

    아이콘은 채도가 높고, 그 주변 패널은 연회색이라 채도가 낮다. 그 차이로 찾는다.
    """
    if img is None or img.size == 0:
        return []
    h, w = img.shape[:2]
    x0, x1 = int(w * BAND_X[0]), int(w * BAND_X[1])
    y0, y1 = int(h * BAND_Y[0]), int(h * BAND_Y[1])
    band = img[y0:y1, x0:x1]
    if band.size == 0:
        return []

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 90, 90), (255, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    found = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < ICON_MIN_AREA:
            continue
        # 아이콘은 대체로 정사각형에 가깝다. 가로로 길쭉한 것은 반짝임/띠다.
        if bw > bh * 2.2 or bh > bw * 2.2:
            continue
        found.append((x0 + int(x), y0 + int(y), int(bw), int(bh)))
    found.sort(key=lambda b: b[1])
    return found


def _strip_of(img: np.ndarray, icon: tuple[int, int, int, int]) -> np.ndarray:
    """아이콘 오른쪽의 숫자 띠를 잘라 낸다."""
    x, y, bw, bh = icon
    sx0 = x + bw + int(bw * 0.20)
    # 너무 넓게 자르면 오른쪽 원형 버튼까지 들어와 숫자 분리가 망가진다(실측).
    sx1 = min(img.shape[1], sx0 + int(bw * STRIP_W_RATIO))
    cy = y + bh // 2
    sy0 = max(0, cy - int(bh * 0.55))
    sy1 = min(img.shape[0], cy + int(bh * 0.55))
    return img[sy0:sy1, sx0:sx1]


# --------------------------------------------------------------------------
# 숫자 읽기
# --------------------------------------------------------------------------

def split_glyphs(strip: np.ndarray) -> list[np.ndarray]:
    """숫자 띠에서 글자 하나하나를 왼쪽부터 잘라 낸다 (쉼표는 뺀다)."""
    if strip is None or strip.size == 0:
        return []
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    mask = (gray < 150).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    sh = strip.shape[0]
    boxes = [tuple(map(int, stats[i][:4])) for i in range(1, num)
             if stats[i][4] >= DIGIT_MIN_AREA]
    # 숫자만 남긴다. 낮은 것은 쉼표, 지나치게 큰 것은 옆 UI 조각이다.
    boxes = [b for b in boxes
             if DIGIT_H_RATIO[0] * sh <= b[3] <= DIGIT_H_RATIO[1] * sh]
    if not boxes:
        return []
    boxes.sort(key=lambda b: b[0])

    # 숫자는 서로 붙어 있다. 큰 틈이 나오면 거기서 끊는다(뒤쪽은 다른 UI).
    kept = [boxes[0]]
    for b in boxes[1:]:
        prev = kept[-1]
        if b[0] - (prev[0] + prev[2]) > prev[2] * 1.5:
            break
        kept.append(b)
    return [gray[y:y + bh, x:x + bw] for x, y, bw, bh in kept]


def _match_glyph(glyph: np.ndarray, digits: dict[str, np.ndarray]) -> str | None:
    """글자 하나를 0~9 중 하나로 읽는다. 자신 없으면 None."""
    best_name, best_score = None, 0.0
    for name, tpl in digits.items():
        t = cv2.resize(tpl, (glyph.shape[1], glyph.shape[0]),
                       interpolation=cv2.INTER_AREA)
        score = float(cv2.matchTemplate(glyph, t, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= DIGIT_MATCH_MIN else None


def read_number(img: np.ndarray, icon: tuple[int, int, int, int]) -> int | None:
    """한 줄의 숫자를 읽는다. 글자 하나라도 모르면 None."""
    digits = load_digits()
    if not digits:
        return None
    glyphs = split_glyphs(_strip_of(img, icon))
    if not glyphs:
        return None
    out = ""
    for gl in glyphs:
        d = _match_glyph(gl, digits)
        if d is None:
            return None
        out += d
    return int(out) if out else None


def read(img: np.ndarray) -> Counters:
    """세 줄을 모두 읽는다. 못 읽은 항목은 None 으로 둔다."""
    rows = find_rows(img)
    if len(rows) < 3:
        return Counters()
    values = [read_number(img, rows[i]) for i in range(3)]
    return Counters(steps=values[0], break_=values[1], dash=values[2])
