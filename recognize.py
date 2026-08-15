"""셀 상태 인식.

각 셀을 다음 다섯 가지로 분류한다.
    PLAYER    플레이어
    GOAL      목적지 또는 필수 아이템
    OBSTACLE  장애물
    ITEM      일반 아이템
    EMPTY     빈칸

인식은 두 겹이다.
  1) 템플릿 매칭  - templates/explore/<종류>/*.png 가 있으면 우선 사용.
                    여러 장, 좌우 반전, 여러 배율을 모두 시도한다.
  2) 색 기반 판정 - 템플릿이 없거나 실패했을 때의 보조 수단.
                    실측한 HSV 분포를 그대로 쓴다.

실측 HSV (기준 캡처 709x1260)
    빈칸        H≈101  S≈245  V≈100      진한 파랑
    이동가능칸  H≈102  S≈241  V≈203      밝은 파랑 (플레이어 상하좌우에 표시됨)
    장애물      H≈116  S≈110  V≈134      연보라 피라미드
    플레이어    S 낮고 따뜻한 색 픽셀이 섞임
    아이템      주황 카드 (따뜻한 색 비율 높음)

주의해야 하는 두 가지 '삐져나옴'
  - 플레이어의 발은 아래 셀로 조금 넘어간다  -> 발끝보다 살짝 위를 기준점으로 쓴다.
  - 장애물 피라미드는 위 셀로 삐져나온다     -> 셀의 아래쪽 영역만 보고 판정한다.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from board import Grid, N
from imgio import imread_bgr
from paths import EXPLORE_TEMPLATE_DIR as TEMPLATE_DIR


class Kind(str, Enum):
    EMPTY = "빈칸"
    PLAYER = "플레이어"
    GOAL = "목적지"
    OBSTACLE = "장애물"
    ITEM = "아이템"


@dataclass
class Detection:
    kind: Kind
    row: int
    col: int
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    note: str = ""


@dataclass
class Scene:
    """한 프레임의 인식 결과 전체."""
    grid: Grid
    cells: list[list[Kind]]
    player: Detection | None = None
    goal: Detection | None = None            # 가장 확실한 주황칩 하나 (호환용)
    goals: list[Detection] = field(default_factory=list)   # 판 위의 모든 주황칩
    detections: list[Detection] = field(default_factory=list)
    highlights: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def kind_at(self, row: int, col: int) -> Kind:
        return self.cells[row][col]

    def summary(self) -> str:
        rows = []
        sym = {Kind.EMPTY: ".", Kind.PLAYER: "P", Kind.GOAL: "G",
               Kind.OBSTACLE: "X", Kind.ITEM: "i"}
        for r in range(N):
            rows.append(" ".join(sym[self.cells[r][c]] for c in range(N)))
        return "\n".join(rows)


# --------------------------------------------------------------------------
# 템플릿 로딩
# --------------------------------------------------------------------------

class TemplateSet:
    """templates/explore/<이름>/ 안의 모든 PNG 를 불러 둔다."""

    def __init__(self, name: str, allow_flip: bool = False):
        self.name = name
        self.allow_flip = allow_flip
        self.images: list[np.ndarray] = []
        self.paths: list[str] = []
        self.reload()

    def reload(self) -> None:
        self.images, self.paths = [], []
        self._prepared = {}
        folder = os.path.join(TEMPLATE_DIR, self.name)
        for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
            img = imread_bgr(path)
            if img is not None and img.size:
                self.images.append(img)
                self.paths.append(path)

    def __bool__(self) -> bool:
        return bool(self.images)

    def variants(self):
        """(이미지, 라벨) 목록. allow_flip 이면 좌우 반전본도 함께 낸다."""
        for img, path in zip(self.images, self.paths):
            label = os.path.basename(path)
            yield img, label
            if self.allow_flip:
                yield cv2.flip(img, 1), label + "(반전)"

    def prepared(self, target_h: int | None, scales, max_h: int, max_w: int):
        """정규화 + 배율까지 끝낸 (이미지, 라벨) 목록.

        같은 (target_h, 배율표) 조합은 한 번만 만들고 캐시해 둔다. 셀 25칸을
        같은 템플릿으로 훑는 자리에서는 똑같은 리사이즈를 25번 반복하고 있었다
        (실측: analyze 한 번에 resize 414회). 템플릿 이미지는 실행 중 바뀌지
        않으므로 다시 만들 이유가 없다. reload() 하면 캐시를 버린다.

        max_h/max_w 보다 큰 템플릿은 그 화면에서 찾을 수 없으므로 빼고 낸다.
        """
        # 캐시 사전을 여기서 만드는 이유: TemplateSet 을 __new__ 로 만들어
        # 속성만 직접 채우는 곳(테스트)이 있어서 __init__ 을 거치지 않는다.
        cache = self.__dict__.setdefault("_prepared", {})
        key = (target_h, tuple(scales))
        built = cache.get(key)
        if built is None:
            # target_h 는 셀 높이에서 오므로 격자가 1px 흔들리면 새 항목이 생긴다.
            # 오래 돌려도 무한정 쌓이지 않게 가끔 비운다(다시 만들면 그만이다).
            if len(cache) > 32:
                cache.clear()
            built = []
            for tpl, label in self.variants():
                base = tpl
                if target_h:
                    k = target_h / tpl.shape[0]
                    nh, nw = max(1, int(tpl.shape[0] * k)), max(1, int(tpl.shape[1] * k))
                    base = cv2.resize(tpl, (nw, nh),
                                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
                for s in scales:
                    nh = max(4, int(round(base.shape[0] * s)))
                    nw = max(4, int(round(base.shape[1] * s)))
                    built.append((cv2.resize(base, (nw, nh),
                                             interpolation=cv2.INTER_AREA if s < 1
                                             else cv2.INTER_CUBIC), label))
            cache[key] = built
        return [(im, lb) for im, lb in built
                if im.shape[0] <= max_h and im.shape[1] <= max_w]


def load_templates() -> dict[str, TemplateSet]:
    return {
        "player": TemplateSet("player", allow_flip=True),
        "player_body": TemplateSet("player_body", allow_flip=True),
        "goal": TemplateSet("goal"),
        "obstacle": TemplateSet("obstacle"),
        "item": TemplateSet("item"),
        "top_tab": TemplateSet("top_tab"),
        "blocked_toast": TemplateSet("blocked_toast"),
        "green_button": TemplateSet("green_button"),
    }


# --------------------------------------------------------------------------
# 템플릿 매칭
# --------------------------------------------------------------------------

def match_best(scene_img: np.ndarray, tset: TemplateSet, scales=(0.7, 0.85, 1.0, 1.15, 1.3),
               target_h: int | None = None):
    """여러 템플릿 x 좌우반전 x 여러 배율 중 가장 잘 맞는 하나를 돌려준다.

    target_h 를 주면 템플릿을 그 높이에 맞춰 정규화한 뒤 배율을 적용한다.
    (게임판 셀 높이를 기준으로 삼으면 창 크기가 달라져도 같은 배율표를 쓸 수 있다.)

    반환: (score, bbox(x0,y0,x1,y1), label) — 못 찾으면 (0.0, None, "")
    """
    if not tset or scene_img is None or scene_img.size == 0:
        return 0.0, None, ""
    sh, sw = scene_img.shape[:2]
    best = (0.0, None, "")

    for resized, label in tset.prepared(target_h, scales, sh, sw):
        res = cv2.matchTemplate(scene_img, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            nh, nw = resized.shape[:2]
            best = (float(score), (loc[0], loc[1], loc[0] + nw, loc[1] + nh), label)
    return best


def match_big(scene_img: np.ndarray, tset: TemplateSet, scales=(0.85, 1.0, 1.15),
              shrink: float = 1 / 3, pad: int = 28):
    """넓은 화면에서 **큰 템플릿**을 찾을 때 쓰는 2단계 매칭.

    안내문 템플릿은 478x77 이고 화면은 709x1260 이다. 이걸 배율 3개 x 템플릿
    3장으로 원본 해상도에서 통째로 훑으면 한 번에 346ms 가 든다. 그런데 이
    검사는 이동 확인 루프에서 **폴링마다** 돌기 때문에, 2.2초 제한 안에 확인
    기회가 5~6번밖에 생기지 않았다. 사실상 매크로 전체에서 가장 비싼 자리였다.

    그래서 두 단계로 나눈다.
      1) 화면과 템플릿을 함께 1/3 로 줄여 대략의 위치를 찾는다 (약 1/9 비용)
      2) 그 자리 주변만 원본 해상도로 다시 맞춰 **정확한 점수**를 낸다

    점수는 결국 원본 해상도에서 재므로 판정 기준(toast_min 등)은 그대로 쓴다.
    실측 근거: 실제 캡처 102장에서 안내문이 있는 프레임의 1/3 해상도 점수는
    0.882~0.953, 없는 프레임은 0.577~0.588 로 0.29 만큼 확실히 갈린다.
    즉 축소본이 봉우리를 엉뚱한 데로 옮기지 않는다.

    2단계 창은 상자의 **중심**을 기준으로 잡는다. 같은 물체라면 배율이 달라도
    중심은 그대로지만 왼쪽 위 모서리는 크기 차이만큼 밀리기 때문이다.

    반환값은 match_best 와 같다: (score, bbox, label)
    """
    if not tset or scene_img is None or scene_img.size == 0:
        return 0.0, None, ""
    sh, sw = scene_img.shape[:2]
    if min(sh, sw) * shrink < 4:
        # 줄이면 남는 게 없는 작은 화면. 그냥 원본에서 찾는다.
        return match_best(scene_img, tset, scales=scales)

    small = cv2.resize(scene_img, None, fx=shrink, fy=shrink,
                       interpolation=cv2.INTER_AREA)
    coarse_scales = tuple(s * shrink for s in scales)
    _, box, _ = match_best(small, tset, scales=coarse_scales)
    if box is None:
        # 축소본에서조차 넣어볼 템플릿이 없었다(화면보다 크다). 원본으로 간다.
        return match_best(scene_img, tset, scales=scales)

    # 축소본의 상자 중심을 원본 좌표로 되돌리고, 가장 큰 배율의 템플릿이 통째로
    # 들어갈 만큼만 창을 잡는다.
    half_h = int(max(im.shape[0] for im in tset.images) * max(scales) / 2) + pad
    half_w = int(max(im.shape[1] for im in tset.images) * max(scales) / 2) + pad
    cx = int((box[0] + box[2]) / 2 / shrink)
    cy = int((box[1] + box[3]) / 2 / shrink)
    x0, y0 = max(0, cx - half_w), max(0, cy - half_h)
    x1, y1 = min(sw, cx + half_w), min(sh, cy + half_h)

    score, fine, label = match_best(scene_img[y0:y1, x0:x1], tset, scales=scales)
    if fine is None:
        return match_best(scene_img, tset, scales=scales)
    return score, (fine[0] + x0, fine[1] + y0, fine[2] + x0, fine[3] + y0), label


# --------------------------------------------------------------------------
# 색 마스크들
# --------------------------------------------------------------------------

def hsv_of(img: np.ndarray) -> np.ndarray:
    """BGR -> HSV. 한 프레임에서 마스크를 여러 개 쓸 때 이걸 한 번만 구해 돌려쓴다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def _range_mask(img, hsv, lo, hi) -> np.ndarray:
    """HSV 범위 안이면 1, 아니면 0 인 uint8 마스크.

    예전에는 채널을 int16 으로 바꾼 뒤 `(h>=95)&(h<=110)&...` 로 비교했다.
    비교만 하는데 int16 으로 넓힐 이유가 없었고(임계값이 전부 0~255 안이다),
    그 변환 하나에 프레임당 5MB 짜리 임시 배열이 세 개씩 생겼다. cv2.inRange 는
    한 번에 훑어서 같은 결과를 훨씬 싸게 낸다. 결과값은 0/255 라 >>7 로 0/1 로
    되돌린다(기존 마스크가 0/1 이고 _frac 이 mean() 을 쓰기 때문).
    """
    if hsv is None:
        hsv = hsv_of(img)
    return cv2.inRange(hsv, lo, hi) >> 7


def mask_highlight(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """이동 가능 강조칸(밝은 파랑)."""
    return _range_mask(img, hsv, (95, 200, 165), (110, 255, 255))


def mask_plain(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """빈칸(진한 파랑)."""
    return _range_mask(img, hsv, (95, 200, 0), (110, 255, 164))


def mask_obstacle(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """장애물(연보라 피라미드): 채도가 낮고 밝다."""
    return _range_mask(img, hsv, (106, 55, 100), (130, 185, 255))


def mask_warm(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """따뜻한 색(주황/빨강/노랑): 플레이어와 아이템 카드."""
    if hsv is None:
        hsv = hsv_of(img)
    # H 가 양 끝(빨강)으로 갈라지므로 두 구간을 합친다.
    return (cv2.inRange(hsv, (0, 90, 80), (28, 255, 255))
            | cv2.inRange(hsv, (160, 90, 80), (255, 255, 255))) >> 7


def mask_orange_card(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """주황 카드(목적지/아이템) 전용. 붉은기보다 주황~노랑에 가깝다."""
    return _range_mask(img, hsv, (5, 130, 130), (30, 255, 255))


def mask_nonboard(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """게임판 배경도 장애물도 아닌 픽셀 = 스프라이트(플레이어/아이템) 후보."""
    if hsv is None:
        hsv = hsv_of(img)
    board_like = (mask_highlight(img, hsv) | mask_plain(img, hsv)
                  | mask_obstacle(img, hsv))
    return board_like ^ 1


def _frac(mask: np.ndarray, rect: tuple[int, int, int, int],
          inset: float = 0.12, bottom_only: float = 1.0) -> float:
    """rect 안에서 마스크가 차지하는 비율.

    bottom_only < 1.0 이면 셀의 아래쪽 그만큼만 본다. 장애물이 위 셀로
    삐져나오는 것을 무시하려고 쓴다.
    """
    x0, y0, x1, y1 = rect
    iy, ix = int((y1 - y0) * inset), int((x1 - x0) * inset)
    x0, y0, x1, y1 = x0 + ix, y0 + iy, x1 - ix, y1 - iy
    if bottom_only < 1.0:
        y0 = int(y1 - (y1 - y0) * bottom_only)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = mask[max(0, y0):y1, max(0, x0):x1]
    return float(patch.mean()) if patch.size else 0.0


# --------------------------------------------------------------------------
# 플레이어
# --------------------------------------------------------------------------

# 발끝이 아래 셀로 넘어가므로, 논리 위치는 발끝보다 이만큼 위를 기준으로 잡는다.
FOOT_LIFT = 0.20        # 셀 높이 대비
# 템플릿을 셀 높이의 몇 배로 정규화할지. 플레이어 스프라이트는 셀보다 크다
# (실측: 템플릿 높이 / 셀 높이 = 1.41~1.55). 이 값을 1.15 로 잡았더니 템플릿이
# 너무 작게 줄어 매칭 점수가 0.44 까지 떨어졌다. 1.4 로 맞추면 0.85 가 나온다.
PLAYER_TARGET_H_RATIO = 1.4
PLAYER_SCALES = (0.85, 1.0, 1.15)
PLAYER_TEMPLATE_MIN = 0.62
PLAYER_BODY_MIN = 0.55


def _anchor_cell(grid: Grid, bbox: tuple[int, int, int, int],
                 centroid: tuple[float, float] | None = None) -> tuple[int, int]:
    """스프라이트로부터 논리적 셀 위치를 계산한다.

    centroid(마스크 질량 중심)가 있으면 그걸 쓴다. 스프라이트의 무게중심은
    자기 칸 중앙에 놓이는 반면(실측: 셀 중심과 2~3px 차이), bbox 의 아래끝은
    그림자·이웃 아이템·위 칸 장애물 조각이 조금만 붙어도 쉽게 흔들린다.
    (실측: 옆 칸 아이템 카드가 덩어리에 붙어 bbox 아래끝이 17px 내려가는 바람에
     (1,1) 에 있는 플레이어를 (2,1) 로 읽었고, 그래서 매크로가 자기 발밑 칸을
     클릭해 '이동할 수 없습니다'만 반복했다.)

    centroid 가 없을 때만 예전 방식(발끝보다 살짝 위)을 쓴다. 맨 아래 행에서
    발이 테두리에 가려지면 bbox 아래끝이 짧아지므로 그만큼 올려 잡는다.
    """
    x0, y0, x1, y1 = bbox
    if centroid is not None:
        return grid.clamp_locate(centroid[0], centroid[1])
    return grid.clamp_locate((x0 + x1) / 2, y1 - grid.cell_h * FOOT_LIFT)


def _player_from_highlights(highlights: list[tuple[int, int]]) -> tuple[int, int] | None:
    """이동 가능 강조칸(십자 모양)의 공통 이웃 = 플레이어 칸.

    게임이 플레이어의 상하좌우를 밝게 표시해 주므로, 강조칸이 2개 이상이면
    플레이어 위치를 좁힐 수 있다.

    다만 **강조칸 전부가 후보의 상하좌우여야 한다.** 이 조건이 없으면 십자가
    아닌 엉뚱한 밝은 칸 조합에서도 답을 하나 만들어 내 버린다.
    (실측: 강조칸이 [(0,0),(1,0),(1,1)] 로 잡혔는데 (1,0)은 (0,1)의 대각선인데도
     (0,1)을 플레이어로 반환해, 템플릿이 맞게 찾은 (2,1)을 덮어썼다.)
    """
    if len(highlights) < 2:
        return None
    hl = set(highlights)
    counts: dict[tuple[int, int], int] = {}
    for r, c in highlights:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in hl:
                counts[(nr, nc)] = counts.get((nr, nc), 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    if n < 2:
        return None
    # 십자 검증: 강조칸이 모두 best 의 상하좌우인가?
    if any(abs(r - best[0]) + abs(c - best[1]) != 1 for r, c in highlights):
        return None
    return best


def _player_blob(img: np.ndarray, grid: Grid, hsv: np.ndarray | None = None):
    """템플릿 없이 플레이어 스프라이트를 찾는 보조 수단.

    게임판 영역 안에서 '게임판 색도 장애물 색도 아닌' 가장 큰 덩어리를 고른다.
    플레이어는 셀 하나를 거의 채우므로 아이템 카드보다 훨씬 크다.

    OPEN 을 크게(7x7) 잡아 옆 칸 아이템 카드처럼 가느다랗게 이어진 것을 끊는다.
    CLOSE 는 작게(3x3) 둔다. 반대로 하면 아이템까지 한 덩어리로 붙어서
    스프라이트 상자가 실제보다 한참 커진다(실측: 폭 1.8칸).

    반환: (bbox, 셀 대비 면적, 질량중심(cx, cy)) 또는 None
    """
    bx0, by0, bx1, by1 = grid.bbox
    sub = img[by0:by1, bx0:bx1]
    if sub.size == 0:
        return None
    # 이미 구해 둔 프레임 HSV 가 있으면 같은 자리를 잘라 쓴다(재변환 방지).
    sub_hsv = hsv[by0:by1, bx0:bx1] if hsv is not None else None
    m = mask_nonboard(sub, sub_hsv)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
    cell_area = grid.cell_w * grid.cell_h
    best = None
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < cell_area * 0.18:
            continue
        if h < grid.cell_h * 0.45:
            continue
        warm = mask_warm(sub[y:y + h, x:x + w],
                         None if sub_hsv is None else sub_hsv[y:y + h, x:x + w]).mean()
        score = area / cell_area + warm
        if best is None or score > best[0]:
            best = (score,
                    (bx0 + x, by0 + y, bx0 + x + w, by0 + y + h),
                    area / cell_area,
                    (bx0 + float(centroids[i][0]), by0 + float(centroids[i][1])))
    if best is None:
        return None
    return best[1], float(np.clip(best[2], 0.0, 1.0)), best[3]


def track_player_fast(img: np.ndarray, grid: Grid,
                      highlights: list[tuple[int, int]],
                      hsv: np.ndarray | None = None) -> tuple[int, int] | None:
    """템플릿 매칭 없이 플레이어 칸만 빠르게 추적한다.

    한 칸 이동이 끝났는지 확인하려고 짧은 주기로 계속 호출하는 자리라서 속도가
    전부다. 전체 템플릿 매칭은 여기에 쓰면 안 된다.
    (실측: 템플릿 경로는 한 번에 1~2초가 걸려서, 2.2초 안에 '연속 2회 확인'이
     아예 불가능했다. 그래서 모든 이동이 실패로 잡혔다.)

    순서: 강조칸 십자(가장 쌈) -> 색 기반 덩어리.
    """
    hint = _player_from_highlights(highlights)
    if hint and len(highlights) >= 3:
        return hint
    blob = _player_blob(img, grid, hsv)
    if blob:
        return _anchor_cell(grid, blob[0], blob[2])
    return hint


def detect_player(img: np.ndarray, grid: Grid, tpl: dict[str, TemplateSet],
                  highlights: list[tuple[int, int]],
                  hsv: np.ndarray | None = None) -> Detection | None:
    """플레이어를 찾는다.

    순서:
      1) 전체 템플릿 (여러 장 / 좌우 반전 / 여러 배율)
      2) 실패하면 몸통 중앙 템플릿 (머리와 발을 뺀 부분)   <- 오탐을 줄이려고 2순위
      3) 그래도 실패하면 색 기반 덩어리 검출
      4) 이동 가능 강조칸의 공통 이웃으로 교차 검증/보정
    """
    bx0, by0, bx1, by1 = grid.bbox
    # 머리가 위 셀로, 발이 아래 셀로 삐져나올 수 있으니 게임판보다 조금 넓게 본다.
    pad_y = int(grid.cell_h * 0.6)
    sy0, sy1 = max(0, by0 - pad_y), min(img.shape[0], by1 + pad_y)

    # 템플릿 매칭은 넓은 영역에서 돌리면 매우 비싸다(실측: 게임판 전체에서
    # 1초). 먼저 색 기반 덩어리로 대략 어디인지 찾아 그 주변만 훑으면
    # 결과는 그대로면서 훨씬 빨라진다.
    blob = _player_blob(img, grid, hsv)
    if blob:
        gx0, gy0, gx1, gy1 = blob[0]
        mx, my = int(grid.cell_w * 0.6), int(grid.cell_h * 0.6)
        sx0, sx1 = max(bx0, gx0 - mx), min(bx1, gx1 + mx)
        sy0, sy1 = max(0, gy0 - my), min(img.shape[0], gy1 + my)
    else:
        sx0, sx1 = bx0, bx1
    sub = img[sy0:sy1, sx0:sx1]

    note = ""
    bbox = None
    conf = 0.0

    if tpl["player"]:
        score, box, label = match_best(sub, tpl["player"],
                                       target_h=int(grid.cell_h * PLAYER_TARGET_H_RATIO),
                                       scales=PLAYER_SCALES)
        if score >= PLAYER_TEMPLATE_MIN and box:
            bbox = (sx0 + box[0], sy0 + box[1], sx0 + box[2], sy0 + box[3])
            conf, note = score, f"전체 템플릿 {label}"

    if bbox is None and tpl["player_body"]:
        # 전체 인식이 실패했을 때만 몸통 보조 인식을 쓴다(오탐 감소).
        score, box, label = match_best(sub, tpl["player_body"],
                                       target_h=int(grid.cell_h * 0.55),
                                       scales=PLAYER_SCALES)
        if score >= PLAYER_BODY_MIN and box:
            # 몸통 아래쪽 = 발이 아니므로, 발 위치를 셀 높이만큼 아래로 추정한다.
            y1 = sy0 + box[3] + int(grid.cell_h * 0.30)
            bbox = (sx0 + box[0], sy0 + box[1], sx0 + box[2], y1)
            conf, note = score, f"몸통 보조 템플릿 {label}"

    if bbox is None and blob:
        bbox, conf = blob[0], min(0.60, blob[1])
        note = "색 기반 덩어리"

    # 위치는 **이번 프레임의 실제 실루엣(덩어리)** 을 기준으로 잡는다.
    # 템플릿은 '이게 플레이어가 맞다'를 확인하고 신뢰도를 주는 역할이고,
    # 크기를 셀 높이에 맞춰 정규화하는 과정에서 몇 px 씩 어긋나기 때문에
    # 그 상자의 아래끝을 그대로 발 위치로 쓰면 셀이 한 칸 밀릴 수 있다.
    # (실측: 템플릿 상자가 8px 아래로 잡혀 (1,1) 이 (2,1) 로 뒤집혔다.)
    if blob is not None and bbox is not None and note.startswith(("전체", "몸통")):
        note += " (위치는 실루엣 기준)"

    hint = _player_from_highlights(highlights)

    if bbox is None:
        if hint:
            return Detection(Kind.PLAYER, hint[0], hint[1], 0.5,
                             note="이동가능칸 십자에서 역산")
        return None

    row, col = _anchor_cell(grid, bbox, blob[2] if blob else None)
    if hint == (row, col):
        conf = min(1.0, conf + 0.15)
        note += " + 강조칸 일치"
    elif hint:
        # 이미지에서 실제로 찾은 위치가 우선이다. 강조칸 역산은 어디까지나 보조라서,
        # 이걸로 이미지 결과를 덮으면 멀쩡히 맞은 위치를 틀리게 만든다(실측).
        note += f" (강조칸 역산은 {hint} 였지만 이미지 결과를 따름)"
        conf = max(0.0, conf - 0.05)

    return Detection(Kind.PLAYER, row, col, float(conf), bbox, note)


# --------------------------------------------------------------------------
# 장면 전체 인식
# --------------------------------------------------------------------------

# 장애물 판정 기준 (실측: 실제 피라미드 셀 0.72~0.75,
# 위 셀로 삐져나온 꼭대기만 걸친 셀 0.10~0.11 -> 사이를 넉넉히 가른다)
OBSTACLE_FRAC = 0.32
OBSTACLE_FRAC_WEAK = 0.18     # 가려졌을 때를 위한 완화 기준
ITEM_WARM_FRAC = 0.10
GOAL_TEMPLATE_MIN = 0.62
GOAL_ORANGE_FRAC = 0.035


def _overlaps_player(rect, player: Detection | None, ratio: float = 0.25) -> bool:
    """셀 rect 가 플레이어 스프라이트 bbox 와 상당히 겹치는가.

    플레이어의 머리/날개는 위 셀로, 발은 아래 셀로 넘어간다. 그 부분을
    아이템이나 목적지로 오인하지 않도록 걸러내는 데 쓴다.
    """
    if player is None or player.bbox is None:
        return False
    ax0, ay0, ax1, ay1 = rect
    bx0, by0, bx1, by1 = player.bbox
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return (ix * iy) / area >= ratio


def analyze(img: np.ndarray, grid: Grid, tpl: dict[str, TemplateSet],
            orange_goal_without_template: bool = False) -> Scene:
    """한 프레임을 통째로 인식해서 Scene 을 만든다.

    orange_goal_without_template
        목적지 템플릿이 아예 없을 때도 '주황 카드'를 목적지로 볼지 여부.
        기본은 False. 일반 아이템 카드도 주황색이라 템플릿 없이 구분할 수 없고,
        엉뚱한 칸을 목적지로 삼으면 경로 전체가 틀어지기 때문이다.
        이 경우 목적지 미인식으로 두고 '오른쪽 끝 열' 규칙으로 진행한다.
    """
    cells = [[Kind.EMPTY for _ in range(N)] for _ in range(N)]
    detections: list[Detection] = []
    notes: list[str] = []

    # 마스크 다섯 개가 전부 같은 HSV 를 본다. 한 번만 변환해서 돌려쓴다.
    hsv = hsv_of(img)
    m_high = mask_highlight(img, hsv)
    m_obst = mask_obstacle(img, hsv)
    m_orange = mask_orange_card(img, hsv)
    m_warm = mask_warm(img, hsv)

    highlights: list[tuple[int, int]] = []
    ofrac = [[0.0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            rect = grid.cell_rect(r, c)
            if _frac(m_high, rect) > 0.40:
                highlights.append((r, c))
            ofrac[r][c] = _frac(m_obst, rect)

    # --- 1) 플레이어 먼저 ------------------------------------------------
    # 플레이어를 먼저 확정해야 그 스프라이트가 걸친 이웃 칸을 아이템/목적지로
    # 잘못 세는 일을 막을 수 있다.
    player = detect_player(img, grid, tpl, highlights, hsv)
    if player is None:
        notes.append("플레이어를 찾지 못했습니다.")

    # --- 2) 장애물 -------------------------------------------------------
    # 템플릿 매칭은 **색만으로 판정이 안 갈리는 칸에서만** 돌린다. 색 비율이
    # 이미 기준을 넘었으면 템플릿 점수가 얼마든 결과는 장애물이므로, 25칸을
    # 전부 훑을 이유가 없다(실측: analyze 한 번에 matchTemplate 283회 중 200회가
    # 이 자리였다). 판정 결과는 그대로이고, 건너뛴 칸은 신뢰도에 색 비율만 남는다.
    for r in range(N):
        for c in range(N):
            f = ofrac[r][c]
            below_is_obstacle = (r + 1 < N and ofrac[r + 1][c] >= OBSTACLE_FRAC)
            # 아래 칸에 피라미드가 없는데도 보라색이 제법 잡힌다면
            # 이 칸의 장애물이 플레이어/테두리에 가려진 경우다.
            hit = (f >= OBSTACLE_FRAC
                   or (f >= OBSTACLE_FRAC_WEAK and not below_is_obstacle))
            ts = 0.0
            if not hit and tpl["obstacle"]:
                x0, y0, x1, y1 = grid.cell_rect(r, c)
                ts, _, _ = match_best(img[y0:y1, x0:x1], tpl["obstacle"],
                                      target_h=int(grid.cell_h),
                                      scales=(0.85, 1.0, 1.15))
                hit = ts >= 0.60
            if hit:
                cells[r][c] = Kind.OBSTACLE
                detections.append(Detection(
                    Kind.OBSTACLE, r, c, max(f, ts), grid.cell_rect(r, c),
                    f"색 {f:.2f}" + (f" / 템플릿 {ts:.2f}" if ts else "")))

    # --- 3) 목적지 -------------------------------------------------------
    goal: Detection | None = None
    if tpl["goal"]:
        bx0, by0, bx1, by1 = grid.bbox
        pad = int(grid.cell_h * 0.4)
        gy0 = max(0, by0 - pad)
        sub = img[gy0:min(img.shape[0], by1 + pad), bx0:bx1]
        score, box, label = match_best(sub, tpl["goal"], target_h=int(grid.cell_h * 0.7))
        if score >= GOAL_TEMPLATE_MIN and box:
            bbox = (bx0 + box[0], gy0 + box[1], bx0 + box[2], gy0 + box[3])
            r, c = grid.clamp_locate((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            goal = Detection(Kind.GOAL, r, c, score, bbox, f"템플릿 {label}")

    goals: list[Detection] = [goal] if goal else []

    if not goals and (tpl["goal"] or orange_goal_without_template):
        # 템플릿이 없거나 가려진 경우: 주황 카드 색으로 찾는다.
        # **한 판에 주황칩이 여러 개 놓일 수 있다**(실측: 2개). 가장 진한 하나만
        # 고르면 프레임마다 다른 칩이 뽑혀 목표가 왔다 갔다 한다. 전부 모아 두고
        # 어느 칩부터 먹을지는 경로 계산에서 정한다.
        for r in range(N):
            for c in range(N):
                rect = grid.cell_rect(r, c)
                if cells[r][c] == Kind.OBSTACLE or _overlaps_player(rect, player):
                    continue
                if player and (r, c) == (player.row, player.col):
                    continue
                f = _frac(m_orange, rect, inset=0.05)
                if f >= GOAL_ORANGE_FRAC:
                    goals.append(Detection(Kind.GOAL, r, c, min(0.6, f * 6),
                                           grid.cell_rect(r, c),
                                           f"주황 카드 색 {f:.3f}"))
        if goals:
            notes.append(f"주황칩 {len(goals)}개를 색으로 찾았습니다.")

    for gdet in goals:
        cells[gdet.row][gdet.col] = Kind.GOAL
        detections.append(gdet)
    goal = max(goals, key=lambda d: d.confidence) if goals else None

    # --- 4) 일반 아이템 --------------------------------------------------
    for r in range(N):
        for c in range(N):
            rect = grid.cell_rect(r, c)
            if cells[r][c] in (Kind.OBSTACLE, Kind.GOAL):
                continue
            if player and (r, c) == (player.row, player.col):
                continue
            if _overlaps_player(rect, player):
                continue      # 플레이어의 머리/발이 걸친 칸. 아이템이 아니다.
            if tpl["item"]:
                x0, y0, x1, y1 = rect
                score, _, _ = match_best(img[y0:y1, x0:x1], tpl["item"],
                                         target_h=int(grid.cell_h * 0.5),
                                         scales=(0.85, 1.0, 1.15))
                if score >= 0.60:
                    cells[r][c] = Kind.ITEM
                    detections.append(Detection(Kind.ITEM, r, c, score, rect, "템플릿"))
                    continue
            if _frac(m_warm, rect) >= ITEM_WARM_FRAC:
                cells[r][c] = Kind.ITEM
                detections.append(Detection(Kind.ITEM, r, c, 0.5, rect, "따뜻한 색 비율"))

    # --- 5) 플레이어 칸 확정 ---------------------------------------------
    if player:
        cells[player.row][player.col] = Kind.PLAYER
        detections.append(player)

    return Scene(grid=grid, cells=cells, player=player, goal=goal, goals=goals,
                 detections=detections, highlights=highlights, notes=notes)


# --------------------------------------------------------------------------
# 창 검증에 쓰는 보조 함수
# --------------------------------------------------------------------------

def find_top_tab(img: np.ndarray, tset: TemplateSet, top_ratio: float = 0.35):
    """화면 위쪽 top_ratio 영역에서 고정된 게임 탭 이미지를 찾는다."""
    if not tset:
        return 0.0, None
    h = img.shape[0]
    band = img[:int(h * top_ratio)]
    score, box, _ = match_best(band, tset, scales=(0.8, 0.9, 1.0, 1.1, 1.2))
    return score, box


def find_blocked_toast(img: np.ndarray, tset: TemplateSet):
    """'해당 위치로 이동할 수 없습니다' 안내문을 찾는다 (클릭하면 안 되는 대상).

    이동 확인 루프에서 폴링마다 불리는 자리라 속도가 곧 이동 확인 횟수다.
    그래서 화면 전체를 원본 해상도로 훑지 않고 match_big(축소 선별 -> 원본 확인)
    을 쓴다. 점수는 원본 해상도에서 재므로 판정 기준은 그대로다.
    """
    if not tset:
        return 0.0, None
    score, box, _ = match_big(img, tset, scales=(0.85, 1.0, 1.15))
    return score, box


GREEN_BUTTON_MIN = 0.60


def find_green_button(img: np.ndarray, tset: TemplateSet | None = None
                      ) -> tuple[int, int] | None:
    """장애물을 부수는 우측 하단 초록색 버튼의 중심(클라이언트 좌표).

    템플릿이 있으면 그걸 먼저 쓰고, 없거나 실패하면 화면 아래쪽에서 초록색
    덩어리를 찾는다. 이 버튼은 사용 횟수가 정해져 있으므로(실측: 30회) 정말
    막혔을 때만 눌러야 한다.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    top = int(h * 0.75)          # 게임판이 아니라 하단 UI 영역만 본다
    band = img[top:]
    if band.size == 0:
        return None

    if tset:
        score, box, _ = match_big(band, tset, scales=(0.8, 0.9, 1.0, 1.1, 1.2))
        if score >= GREEN_BUTTON_MIN and box:
            return (box[0] + box[2]) // 2, top + (box[1] + box[3]) // 2

    green = _range_mask(band, None, (35, 90, 110), (85, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    num, _, stats, cent = cv2.connectedComponentsWithStats(green, 8)
    best = None
    for i in range(1, num):
        area = stats[i][4]
        if area < 800:           # 작은 초록 아이콘(잔여 횟수 표시 등)은 제외
            continue
        if best is None or area > best[0]:
            best = (area, (int(cent[i][0]), top + int(cent[i][1])))
    return best[1] if best else None
