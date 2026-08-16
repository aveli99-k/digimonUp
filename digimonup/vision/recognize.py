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

from digimonup.vision.board import Grid, N
from digimonup.base.imgio import hsv_of, imread_bgr
from digimonup.base.paths import EXPLORE_TEMPLATE_DIR as TEMPLATE_DIR


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
    # 플레이어일 때만 채워지는 실제 스프라이트 픽셀 마스크.
    # 이웃 칸의 칩/아이템을 셀 때 플레이어 몸을 빼는 데 쓴다.
    sprite: np.ndarray | None = None
    # 아이템일 때 그 종류 ("steps" / "break" / "dash"). 모르면 빈 문자열.
    # 템플릿 파일 이름 item_<종류>.png 에서 읽는다.
    item_kind: str = ""


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
    # (행, 열) -> 아이템 종류. 경로 계산에서 '걸음수는 들르지 않는다' 같은
    # 판단에 쓴다. 종류를 모르는 아이템은 빈 문자열이다.
    item_kinds: dict[tuple[int, int], str] = field(default_factory=dict)

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
    """<기준 폴더>/<이름>/ 안의 모든 PNG 를 불러 둔다.

    기준 폴더는 기본이 templates/explore/ 다. 던전처럼 다른 기능의 템플릿은
    base_dir 로 자기 폴더를 넘긴다. 불러오기·정규화·캐시는 기능이 달라도
    똑같아서, 여기 한 벌만 두고 폴더만 갈아끼운다.
    """

    def __init__(self, name: str, allow_flip: bool = False,
                 base_dir: str = TEMPLATE_DIR):
        self.name = name
        self.allow_flip = allow_flip
        self.base_dir = base_dir
        self.images: list[np.ndarray] = []
        self.paths: list[str] = []
        self.reload()

    def reload(self) -> None:
        self.images, self.paths = [], []
        self._prepared = {}
        folder = os.path.join(self.base_dir, self.name)
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
# 예전에는 템플릿을 셀 높이의 1.4배로 정규화했다. 그때 디지몬이 셀보다 컸기
# 때문이다(템플릿 높이/셀 높이 = 1.41~1.55). 그런데 디지몬을 바꾸면 이 비율이
# 통째로 달라진다(실측: 작은 토끼형은 0.61). 정규화하면 2배 넘게 부풀려져
# 아예 안 맞는다. 그래서 칩/아이템과 마찬가지로 정규화하지 않고 배율만 넓게 준다.
PLAYER_SCALES = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)
PLAYER_TEMPLATE_MIN = 0.62
# 색 기반 덩어리를 고를 때 '따뜻한 색 비율'에 줄 무게. 크기가 주된 근거다.
WARM_WEIGHT = 0.3
PLAYER_BODY_MIN = 0.55
# 플레이어 스프라이트의 최소 크기(셀 면적 대비). 디지몬은 진화/교체로 크기가
# 크게 달라진다. 실측: 큰 기계형 1.37, 작은 토끼형 0.13. 작은 쪽도 잡아야 한다.
PLAYER_BLOB_MIN_AREA = 0.08


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



# 움직임으로 플레이어 찾기 기준 (실측: 진짜 디지몬 칸 0.296, 나머지 전부 0.000)
MOTION_PIXEL_DIFF = 25       # 이만큼 밝기가 바뀐 픽셀을 '움직였다'고 본다
MOTION_CELL_MIN = 0.05       # 칸에서 움직인 픽셀 비율이 이 이상이어야 후보
MOTION_MAX_CELLS = 6         # 이보다 많은 칸이 움직였으면 판 전체가 움직이는 중


def motion_report(frames: list[np.ndarray], grid: Grid):
    """칸별 '움직인 픽셀 비율'을 재서 (움직인 칸 수, 가장 움직인 칸, 그 비율).

    움직인 칸이 MOTION_MAX_CELLS 를 넘으면 **판이 통째로 움직이는 중**이다.
    그런 프레임은 플레이어 위치뿐 아니라 칩/장애물 인식도 믿을 수 없다.
    (실측: 스크롤 애니메이션 중에 찍힌 프레임에서 칩이 7개로 잡혔고, 매크로가
     그 유령 칩을 먹으러 왼쪽으로 갔다. 정지 화면에서는 한 번도 안 나온다.)
    """
    if len(frames) < 2 or grid is None:
        return 0, None, 0.0, {}
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    acc = None
    for a, b in zip(grays, grays[1:]):
        d = cv2.absdiff(a, b)
        acc = d if acc is None else np.maximum(acc, d)
    moved = (acc > MOTION_PIXEL_DIFF)

    ratios: dict[tuple[int, int], float] = {}
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = grid.cell_rect(r, c)
            ratios[(r, c)] = float(moved[y0:y1, x0:x1].mean())
    busy = sum(1 for v in ratios.values() if v >= MOTION_CELL_MIN)
    cell, best = max(ratios.items(), key=lambda kv: kv[1])
    return busy, cell, best, ratios



def motion_player_cell(frames: list[np.ndarray], grid: Grid
                       ) -> tuple[tuple[int, int], float] | None:
    """연속 프레임에서 **움직인 칸**을 찾아 플레이어 칸을 돌려준다.

    이것이 플레이어를 찾는 가장 확실한 방법이다. 디지몬은 가만히 서 있을 때도
    제자리 애니메이션이 돌아가는 **판 위의 유일한 움직이는 물체**라서, 생김새와
    아무 상관이 없다. 색·모양·템플릿은 디지몬을 바꾸면 전부 무너진다.

    실측(파란 디지몬, 프레임 7장 0.18초 간격)
        (4,1) 0.296   <- 진짜 디지몬
        (3,1) 0.105   <- 머리가 위 칸으로 삐져나온 부분
        나머지 21칸 전부 0.000

    반환: ((행, 열), 움직임 비율) — 판이 통째로 움직이는 중이면 None.
    """
    busy, cell, best, _ = motion_report(frames, grid)
    if cell is None or best < MOTION_CELL_MIN:
        return None
    if busy > MOTION_MAX_CELLS:
        # 판이 스크롤하거나 화면이 통째로 바뀌는 중이다. 이때는 못 믿는다.
        return None
    # 머리는 위 칸으로, 발은 아래 칸으로 조금씩 삐져나오지만 **몸통이 있는 칸이
    # 언제나 가장 많이 움직인다.** 실측에서 몸통 0.296 대 머리 0.105 로 세 배
    # 차이가 났다. 그래서 위아래로 더 보정하지 않고 최댓값 칸을 그대로 쓴다.
    return cell, best


def _highlight_center(
    highlights: list[tuple[int, int]],
    cells: list[list[Kind]] | None = None,
) -> tuple[tuple[int, int], bool] | None:
    """강조 영역의 중심 = 플레이어 칸. (중심칸, 확실한가) 를 돌려준다.

    게임은 지금 갈 수 있는 칸을 밝게 칠하는데, 실측해 보니 두 가지가
    예전 가정과 달랐다.

      1. **플레이어 칸 자신도 강조된다.**
      2. **강조칸은 4개 고정이 아니다.** 이동 거리가 늘면 마름모로 넓어진다.

    그래서 '십자의 빈 중심'을 찾으면 안 된다. 대신 칸마다 **거기 서 있었다면
    어디가 칠해졌을지**를 만들어 실제 강조칸과 얼마나 겹치는지로 겨룬다.
    판 밖과 장애물로는 못 가므로 그 둘을 빼고 예측한다. 그래야 벽이나 장애물에
    막혀 팔이 둘뿐인 자리도 제대로 맞힌다.

    실측 예 (실제 화면에서 그대로 가져옴)
        [(3,1),(4,0),(4,2)]                     -> (4,1)  중심칸은 안 칠해짐
        [(1,1),(2,0),(2,1),(2,2),(3,1)]         -> (2,1)
        [(1,0),(1,1),(2,0),(2,1),(2,2),(3,1)]   -> (2,1)  (1,0) 하나가 더 붙음
        [(4,0),(4,1),(4,2)] + (3,1)이 장애물    -> (4,1)  겹침 3/3 으로 완벽

    이 값은 디지몬 생김새와 무관하다. 실측: 파란 디지몬으로 바꾸니 파란 배경에
    묻혀 색덩어리가 화면 반대쪽 분홍 생물체 (4,4) 를 잡았는데, 강조칸은 그때도
    (2,1) 을 정확히 가리켰다.
    """
    if len(highlights) < 2:
        return None
    hl = set(highlights)
    scored = []
    for r in range(N):
        for c in range(N):
            if cells is not None and cells[r][c] == Kind.OBSTACLE:
                continue
            # 이 칸에 플레이어가 있다면 강조됐어야 할 칸들.
            pred = {(r, c)}
            impossible = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < N and 0 <= nc < N):
                    continue            # 판 밖으로는 못 간다
                if cells is not None and cells[nr][nc] == Kind.OBSTACLE:
                    continue            # 장애물로는 못 간다
                pred.add((nr, nc))
                # **바로 옆 빈칸이 안 칠해져 있으면 그 칸에 플레이어가 있을 수 없다.**
                # 강조 범위는 넓어질 수는 있어도 1칸보다 좁아지지는 않기 때문이다.
                # 칩/아이템 칸은 그림이 강조색을 덮어 검출이 놓치므로 빼고 본다.
                # (실측 회귀: 이 검사가 없어서 (2,2) 가 빈칸인데 안 칠해졌는데도
                #  (2,1) 을 플레이어로 골라, 진짜 위치 (1,1) 을 덮어썼다.)
                if (cells is not None and cells[nr][nc] == Kind.EMPTY
                        and (nr, nc) not in hl):
                    impossible = True
                    break
            if impossible or len(pred) < 2:
                continue
            score = len(pred & hl) / len(pred | hl)
            scored.append((-score, (r, c)))
    if not scored:
        return None
    scored.sort()
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None                      # 똑같이 그럴듯하면 쓰지 않는다
    best, center = -scored[0][0], scored[0][1]
    return center, best >= HIGHLIGHT_SURE_MIN


def _player_from_highlights(highlights: list[tuple[int, int]],
                            cells: list[list[Kind]] | None = None
                            ) -> tuple[int, int] | None:
    """강조 영역에서 역산한 플레이어 칸. 자세한 근거는 _highlight_center 참고."""
    got = _highlight_center(highlights, cells)
    return None if got is None else got[0]


def _player_blob(img: np.ndarray, grid: Grid, hsv: np.ndarray | None = None,
                 exclude: set | None = None):
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
        if area < cell_area * PLAYER_BLOB_MIN_AREA:
            continue
        if h < grid.cell_h * 0.45:
            continue
        # 캐릭터는 납작하지 않다. 게임판 아래쪽 장식 블록이 76x37(가로:세로 2:1)
        # 로 잡혀서 작은 디지몬(47x47)을 제치고 플레이어로 뽑힌 적이 있다.
        if h < w * 0.55:
            continue
        # 칩/아이템으로 **확실히** 아는 칸은 플레이어 후보에서 뺀다.
        #
        # 실측 회귀: 디지몬을 작은 것으로 바꿨더니 스프라이트 면적비가 0.13 이 됐다.
        # 주황칩은 0.48 이라 크기로 겨루면 칩이 이긴다. 그래서 칩을 플레이어로
        # 잡고 진짜 플레이어를 놓쳤고, 그 뒤의 모든 이동이 엉뚱한 방향이었다.
        # 칩/아이템은 템플릿으로 이미 확정한 것이므로 후보에서 빼는 편이 확실하다.
        if exclude:
            cx = bx0 + x + w / 2
            cy = by0 + y + h / 2
            if grid.clamp_locate(cx, cy) in exclude:
                continue
        warm = mask_warm(sub[y:y + h, x:x + w],
                         None if sub_hsv is None else sub_hsv[y:y + h, x:x + w]).mean()
        # 크기가 주된 근거이고, 따뜻한 색은 거들 뿐이다.
        #
        # 예전에는 둘을 1:1 로 더했다. 그러면 **작지만 새빨간 것이 큰 것을 이긴다.**
        # 주황칩이 딱 그렇다. 플레이어 바로 옆의 칩을 플레이어로 오인하면 위치가
        # 통째로 틀어지고, 칩이 걸음수보다 중요한 지금은 손해가 크다.
        # (실측 면적비/warm: 실제 플레이어 1.37/0.17, 칩 0.37/0.04 -> 원래도 안전하지만
        #  합성 화면처럼 칩이 단색 주황(warm 1.00)이면 0.22+1.00 으로 뒤집힌다.)
        score = area / cell_area + WARM_WEIGHT * warm
        if best is None or score > best[0]:
            best = (score,
                    (bx0 + x, by0 + y, bx0 + x + w, by0 + y + h),
                    area / cell_area,
                    (bx0 + float(centroids[i][0]), by0 + float(centroids[i][1])),
                    i)
    if best is None:
        return None

    # 스프라이트가 실제로 차지한 픽셀만 담은 마스크(화면 전체 크기).
    # 이웃 칸에서 칩을 셀 때 플레이어의 주황 갈기를 빼는 데 쓴다. 칸을 통째로
    # 빼면 그 칸의 칩까지 안 보이므로, 픽셀 단위로 빼야 한다.
    sprite = np.zeros(img.shape[:2], np.uint8)
    sprite[by0:by1, bx0:bx1] = (labels == best[4]).astype(np.uint8)
    return best[1], float(np.clip(best[2], 0.0, 1.0)), best[3], sprite


def track_player_fast(img: np.ndarray, grid: Grid,
                      highlights: list[tuple[int, int]],
                      hsv: np.ndarray | None = None,
                      exclude: set | None = None) -> tuple[int, int] | None:
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
    # 전체 인식에서 알아낸 '플레이어가 아닌 칸'(장애물/칩/아이템)을 그대로 뺀다.
    # 이걸 안 빼면 이동 확인 중에 장애물 덩어리를 플레이어로 잡아, 실제로는
    # 움직였는데도 확인이 계속 실패한다(실측: 이동 1회 / 실패 9회).
    blob = _player_blob(img, grid, hsv, exclude)
    if blob:
        return _anchor_cell(grid, blob[0], blob[2])
    return hint


def detect_player(img: np.ndarray, grid: Grid, tpl: dict[str, TemplateSet],
                  highlights: list[tuple[int, int]],
                  hsv: np.ndarray | None = None,
                  exclude: set | None = None,
                  cells: list[list[Kind]] | None = None,
                  motion_cell: tuple[int, int] | None = None) -> Detection | None:
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
    blob = _player_blob(img, grid, hsv, exclude)
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
        score, box, label = match_best(sub, tpl["player"], scales=PLAYER_SCALES)
        if score >= PLAYER_TEMPLATE_MIN and box:
            bbox = (sx0 + box[0], sy0 + box[1], sx0 + box[2], sy0 + box[3])
            conf, note = score, f"전체 템플릿 {label}"

    if bbox is None and tpl["player_body"]:
        # 전체 인식이 실패했을 때만 몸통 보조 인식을 쓴다(오탐 감소).
        score, box, label = match_best(sub, tpl["player_body"], scales=PLAYER_SCALES)
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
    if blob is not None and bbox is not None and note.startswith("몸통"):
        note += " (위치는 실루엣 기준)"

    got = _highlight_center(highlights, cells)
    hint, hint_sure = got if got else (None, False)

    # **움직임이 잡혔으면 그게 답이다.** motion_player_cell 참고.
    # 생김새에 전혀 기대지 않는 유일한 신호라, 다른 모든 근거보다 우선한다.
    if motion_cell is not None:
        return Detection(Kind.PLAYER, motion_cell[0], motion_cell[1], 0.95, bbox,
                         note=("움직임으로 확인"
                               + ("" if hint in (None, motion_cell)
                                  else f" (강조칸 역산은 {hint})")),
                         sprite=blob[3] if blob else None)

    if bbox is None:
        if hint:
            return Detection(Kind.PLAYER, hint[0], hint[1], 0.5,
                             note="강조 영역 중심에서 역산")
        return None

    # 위치 기준: 템플릿이 확실히 맞았으면 **템플릿 상자**를, 아니면 실루엣 중심을 쓴다.
    #
    # 예전에는 항상 실루엣 중심을 썼다. 템플릿을 셀 높이에 맞춰 정규화하느라
    # 상자가 몇 px 씩 어긋났기 때문이다. 그 정규화를 없앤 지금은 템플릿 상자가
    # 더 정확하다.
    #
    # 반대로 실루엣은 옆에 붙은 것을 함께 삼킨다. 실측: 디지몬 오른쪽에 게임이
    # 그리는 노란 방향 표시가 덩어리에 붙어 무게중심이 한 칸 오른쪽으로 밀렸다.
    # 그러면 두 칸 떨어진 칸을 클릭해 '이동할 수 없습니다'가 뜬다.
    use_centroid = blob[2] if (blob and not note.startswith("전체 템플릿")) else None
    row, col = _anchor_cell(grid, bbox, use_centroid)
    if hint == (row, col):
        conf = min(1.0, conf + 0.15)
        note += " + 강조칸 일치"
    elif hint:
        # **강조칸으로 이미지 결과를 덮지 않는다.**
        #
        # 강조 범위는 4칸 고정이 아니라 이동력에 따라 넓어진다. 그걸 반지름 1 로
        # 놓고 맞춰 보면 엉뚱한 칸이 더 그럴듯해 보인다. 실측 회귀: 진짜 위치
        # (1,1) 을 (2,1) 로 덮어써서 UP 클릭이 자기 자신을 누르는 꼴이 됐고,
        # 게임이 거절해 같은 실패가 계속 반복됐다(120초에 안내문 14회).
        #
        # 지금은 움직임이 가장 확실한 신호이고 그 다음이 템플릿이다. 강조칸은
        # 둘 다 실패해 위치를 아예 못 잡았을 때의 최후 수단으로만 남긴다.
        note += f" (강조칸 역산 {hint} 는 참고만)"

    return Detection(Kind.PLAYER, row, col, float(conf), bbox, note,
                     sprite=blob[3] if blob else None)


# --------------------------------------------------------------------------
# 장면 전체 인식
# --------------------------------------------------------------------------

# 장애물 판정 기준 (실측: 실제 피라미드 셀 0.72~0.75,
# 위 셀로 삐져나온 꼭대기만 걸친 셀 0.10~0.11 -> 사이를 넉넉히 가른다)
OBSTACLE_FRAC = 0.32
OBSTACLE_FRAC_WEAK = 0.18     # 가려졌을 때를 위한 완화 기준
ITEM_WARM_FRAC = 0.10
ITEM_TEMPLATE_MIN = 0.78
# 칩/아이템 칸에 주황이 최소 이만큼은 있어야 한다.
# 실측: 진짜 칩 0.099 / 칩 없는 칸 0.000~0.020.
CARD_ORANGE_MIN = 0.04
# 칩/아이템이 칸 한가운데에서 이만큼까지 벗어나도 봐준다 (0.5 가 한가운데).
# 실측: 진짜 칩 0.50, 획득 연출로 튀어나온 칩 0.13 / 0.75 / 0.88.
CARD_CENTER_SLACK = 0.22
# 강조칸 역산을 '확실하다'고 볼 겹침 비율. 실측: 맞는 칸 0.75~1.00.
HIGHLIGHT_SURE_MIN = 0.6
# 플레이어는 강조칸 무리에서 이보다 멀 수 없다.
# 강조 범위는 이동력에 따라 넓어지지만 플레이어는 늘 그 한가운데에 있다.
PLAYER_HIGHLIGHT_MAX_DIST = 2


def item_kind_of(label: str) -> str:
    """템플릿 파일 이름에서 아이템 종류를 읽는다.

    item_steps.png -> "steps",  item_break.png -> "break",  item_dash.png -> "dash"
    규칙에 안 맞는 이름이면 빈 문자열(종류 모름)이다.
    """
    name = os.path.splitext(os.path.basename(label))[0]
    name = name.replace("(반전)", "")
    return name[5:] if name.startswith("item_") else ""
# 아이템 아이콘도 셀 높이에 맞춰 정규화하지 않는다(위 GOAL_SCALES 와 같은 이유).
ITEM_SCALES = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35)
GOAL_TEMPLATE_MIN = 0.62
# 칩 템플릿은 카드 전체가 아니라 **중심(번개 무늬)** 이다. 셀 높이에 맞춰
# 정규화하지 않고 배율만 넓게 준다. 템플릿을 찍은 창과 실행 창의 크기가
# 달라도 이 범위 안에서 맞는다.
GOAL_SCALES = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)
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


def _card_orange_ratio(cell_hsv: np.ndarray) -> float:
    """칸 안에서 '주황 카드' 색이 차지하는 비율.

    실측(진짜 칩 0.099 / 칩 없는 칸 0.000~0.020)으로 정한 좁은 범위다.
    """
    h, s, v = cell_hsv[:, :, 0], cell_hsv[:, :, 1], cell_hsv[:, :, 2]
    return float(((h >= 5) & (h <= 30) & (s > 150) & (v > 150)).mean())


def _match_cells(img: np.ndarray, grid: Grid, cells, tpl: dict[str, TemplateSet],
                 hsv: np.ndarray | None = None):
    """칸마다 칩/아이템 템플릿을 맞춰 본 결과를 한 번에 구한다.

    반환: {"goal": {(r,c): (점수, 라벨)}, "item": {...}}

    플레이어를 찾기 **전에** 불러야 한다. 플레이어 찾기는 '가장 큰 덩어리' 라는
    헐거운 기준이라, 칩/아이템 칸을 미리 알아야 그걸 플레이어로 오인하지 않는다.

    주황칩은 정의상 주황색이므로, 템플릿 점수만 믿지 않고 **그 칸에 주황이
    실제로 있는지** 한 번 더 확인한다. 템플릿은 모양만 보기 때문에 밝은 빈칸이나
    노란 방향 표시에서도 기준을 넘길 때가 있다.
    (실측: 칩이 (2,3) 하나뿐인 판에서 (1,2),(2,0),(3,0) 이 칩으로 잡혀 왼쪽으로
     헛걸음했다. 주황 비율은 진짜 칩 0.098, 빈칸 0.000~0.020 으로 확실히 갈린다.)

    아이템에는 걸지 않는다. 아이템은 색이 제각각이다(부수기 노랑, 돌진 초록).
    """
    if hsv is None:
        hsv = hsv_of(img)
    out = {"goal": {}, "item": {}}
    for key, tset, scales, thr in (
            ("goal", tpl["goal"], GOAL_SCALES, GOAL_TEMPLATE_MIN),
            ("item", tpl["item"], ITEM_SCALES, ITEM_TEMPLATE_MIN)):
        if not tset:
            continue
        for r in range(N):
            for c in range(N):
                if cells[r][c] == Kind.OBSTACLE:
                    continue
                x0, y0, x1, y1 = grid.cell_rect(r, c)
                score, box, label = match_best(img[y0:y1, x0:x1], tset, scales=scales)
                if score < thr:
                    continue
                # 칸 한가운데에 놓여 있어야 판 위의 물건이다.
                #
                # 칩을 먹으면 디지몬 주변으로 **칩이 튀어나오는 획득 연출**이
                # 뜬다. 그 칩들은 셀 경계에 걸쳐 비스듬히 그려지는데 템플릿
                # 점수는 0.94~0.98 로 진짜와 구별되지 않는다. 매크로가 그걸
                # 판 위의 칩으로 알고 왼쪽으로 쫓아갔다.
                # 실측 칸 안 중심 (0.50/0.50 이 한가운데)
                #     진짜 칩   0.50/0.50
                #     연출 칩   0.13/0.36, 0.75/0.88, 0.48/0.75
                if box is not None:
                    mx = (box[0] + box[2]) / 2 / max(1, x1 - x0)
                    my = (box[1] + box[3]) / 2 / max(1, y1 - y0)
                    if abs(mx - 0.5) > CARD_CENTER_SLACK or abs(my - 0.5) > CARD_CENTER_SLACK:
                        continue
                # 주황 확인은 **칩에만** 건다. 아이템은 색이 제각각이라
                # (부수기 노랑, 돌진 초록) 주황을 요구하면 놓친다.
                if (key == "goal"
                        and _card_orange_ratio(hsv[y0:y1, x0:x1]) < CARD_ORANGE_MIN):
                    continue
                out[key][(r, c)] = (score, label)
    return out


def _reject_impossible_player(player, cells, highlights, notes):
    """있을 수 없는 자리에 잡힌 플레이어는 **버린다**.

    검출은 드물게 틀린다. 실측한 사고: 디지몬이 (1,1) 에 있는데 (3,0) 피라미드
    위로 잡혔고, 그 자리에서 경로를 계산해 실제 위치와 무관한 칸들을 눌렀다.
    밖에서 보면 '있지도 않은 칩을 향해 엉뚱하게 좌우로 움직이는' 모습이 된다.
    사용자가 유령칩이라고 부른 증상이 이것이었다.

    틀린 것을 늘 막을 수는 없어도, **불가능한 자리로 행동하는 것은 거부**할 수
    있다. 여기서 None 을 돌려주면 그 사이클은 아무것도 하지 않고 다시 본다.
    한 사이클 쉬는 값은 싸고, 엉뚱한 클릭의 값은 비싸다.

    불가능한 자리
      1. 장애물 칸        - 피라미드 위에 설 수 없다
      2. 강조칸에서 멀다  - 게임은 플레이어 주변만 밝게 칠한다(20장).
                            강조칸이 하나라도 보이는데 두 칸 넘게 떨어져 있으면
                            그건 플레이어가 아니다.
    """
    if player is None:
        return None
    r, c = player.row, player.col
    if cells[r][c] == Kind.OBSTACLE:
        notes.append(f"플레이어로 잡힌 ({r},{c}) 은 장애물입니다. 버리고 다시 봅니다.")
        return None
    if highlights:
        near = min(abs(r - a) + abs(c - b) for a, b in highlights)
        if near > PLAYER_HIGHLIGHT_MAX_DIST:
            notes.append(f"플레이어로 잡힌 ({r},{c}) 이 강조칸에서 {near}칸 "
                         f"떨어져 있습니다. 버리고 다시 봅니다.")
            return None
    return player


def analyze(img: np.ndarray, grid: Grid, tpl: dict[str, TemplateSet],
            orange_goal_without_template: bool = False,
            motion_cell: tuple[int, int] | None = None) -> Scene:
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

    # --- 1) 장애물 먼저 -------------------------------------------------------
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

    # --- 3) 주황칩(목적지) ------------------------------------------------
    # 템플릿이 있으면 **칸마다** 맞춰 본다. 예전에는 게임판 전체에서 가장 잘 맞는
    # 하나만 찾았는데, 판에 칩이 여러 개 놓이는 것을 실측으로 확인했다(2개).
    # 하나만 찾으면 나머지를 놓친다. 칩은 걸음수보다 중요하므로 전부 찾아야 한다.
    #
    # 템플릿은 **칩의 중심(번개 무늬)** 이어야 한다. 카드 전체를 쓰면 맨 아래 행에서
    # 게임판 하단 테두리에 카드 아래쪽이 가려질 때 점수가 무너진다. 중심만 쓰면
    # 가려져도 그대로 맞는다(실측: 셀 아래 38px 가 잘려도 무사).

    # --- 2) 칩/아이템을 템플릿으로 확정한다 --------------------------
    # 순서가 중요하다. 예전에는 플레이어를 먼저 찾았는데, 플레이어 찾기는
    # '게임판 색이 아닌 가장 큰 덩어리' 라는 헐거운 기준이라 **칩이나 아이템을
    # 플레이어로 잘못 잡을 수 있다.**
    #
    # 실측 회귀: 디지몬을 작은 것으로 바꿨더니 스프라이트 면적비가 0.13 이 됐다.
    # 주황칩은 0.48 이라 크기로 겨루면 칩이 이긴다. 그래서 칩을 플레이어로 잡고
    # 진짜 플레이어를 놓쳤고, 그 뒤의 모든 이동이 엉뚱한 방향이었다.
    #
    # 칩/아이템은 템플릿으로 확실히 아는 것이므로 먼저 확정하고, 그 칸은
    # 플레이어 후보에서 뺀다.
    #
    # **장애물 판정보다 뒤에 와야 한다.** 장애물 칸에까지 칩 템플릿을 맞추면
    # 연보라 피라미드가 칩으로 잡힌다(실측: 한 프레임에 유령 칩 4개, 그 바람에
    # 못 가는 칸을 클릭해 '이동할 수 없습니다'가 75초에 16번 떴다).
    strong = _match_cells(img, grid, cells, tpl, hsv)
    # 장애물 칸도 뺀다. 실측: 게임판 오른쪽 아래 피라미드가 면적비 0.43 으로 잡혀
    # 작은 디지몬(0.13)을 제치고 플레이어가 됐다. 그러면 디지몬과 상관없는
    # 칸을 계속 클릭해서 '이동할 수 없습니다'만 뜬다(75초에 15번, 이동 0회).
    occupied = (set(strong["goal"]) | set(strong["item"])
                | {(r, c) for r in range(N) for c in range(N)
                   if cells[r][c] == Kind.OBSTACLE})
    player = detect_player(img, grid, tpl, highlights, hsv, exclude=occupied,
                           cells=cells, motion_cell=motion_cell)
    player = _reject_impossible_player(player, cells, highlights, notes)
    if player is None:
        notes.append("플레이어를 찾지 못했습니다.")

    # 플레이어 스프라이트가 차지한 **픽셀만** 마스크에서 지운다.
    #
    # 예전에는 '플레이어 상자와 25% 이상 겹치는 칸'을 통째로 검사에서 뺐다.
    # 그런데 실측하면 스프라이트가 175x141px 이고 셀은 108x88px 이라, 상하좌우
    # 이웃 칸이 30~36% 씩 가려진다. 즉 **플레이어 바로 옆 칸의 칩을 못 봤다.**
    # 칩이 걸음수보다 중요하므로 이건 그냥 손해다.
    #
    # 픽셀 단위로 빼면 디지몬의 주황 갈기는 그대로 걸러지면서 옆 칸의 칩은 보인다.
    if player is not None and player.sprite is not None:
        keep = 1 - player.sprite
        m_orange = m_orange * keep
        m_warm = m_warm * keep

    # 칩보다 **아이템 템플릿을 먼저** 본다.
    #
    # 실측 회귀: 판 위의 부수기 아이템(노란 발톱)이 주황칩으로 잡혔다(색 비율
    # 0.067 -> 칩 판정). 그러면 매크로가 그걸 1순위 목표로 쫓아간다. 노란색과
    # 주황색은 색만으로 가르기 어렵다. 템플릿으로 '이건 아이템이다'가 확인되면
    # 그 칸은 칩 색 판정에서 빼야 한다. 템플릿이 색보다 강한 증거다.
    # 위에서 이미 맞춰 둔 결과를 쓴다(같은 매칭을 두 번 하지 않는다).
    item_by_template: dict[tuple[int, int], Detection] = {}
    for (r, c), (score, label) in strong["item"].items():
        if cells[r][c] == Kind.OBSTACLE:
            continue
        if player and (r, c) == (player.row, player.col):
            continue
        item_by_template[(r, c)] = Detection(
            Kind.ITEM, r, c, score, grid.cell_rect(r, c),
            f"템플릿 {label} {score:.2f}", item_kind=item_kind_of(label))

    goals: list[Detection] = []
    for (r, c), (score, label) in strong["goal"].items():
        if cells[r][c] == Kind.OBSTACLE:
            continue
        if player and (r, c) == (player.row, player.col):
            continue
        # 아이템 템플릿이 더 잘 맞으면 그건 칩이 아니다.
        other = item_by_template.get((r, c))
        if other is not None and other.confidence > score:
            continue
        goals.append(Detection(Kind.GOAL, r, c, score, grid.cell_rect(r, c),
                               f"템플릿 {label} {score:.2f}"))
    if goals:
        notes.append(f"주황칩 {len(goals)}개를 템플릿으로 찾았습니다.")

    # **칩 템플릿이 있으면 색 추측은 쓰지 않는다.**
    #
    # 실측 회귀: 색 대체 경로가 한 프레임에 유령 칩을 7개나 만들었다.
    # 게임판 위아래의 장식 띠와 하단 테두리가 주황색이라 그렇다.
    #     맨 아래 행 주황 비율 0.601 / 0.543 / 0.333   (칩 기준은 0.035)
    #     위쪽 장식 띠        0.104 / 0.099
    # 그 바람에 (1) 맨 아래 진짜 칩이 유령들에 묻히고, (2) 가장 왼쪽 유령을
    # 목표로 삼아 왼쪽으로 잘못 이동하고, (3) 엉뚱한 방향으로 움직였다.
    # 사용자가 보고한 세 증상이 전부 이것 하나에서 나왔다.
    #
    # 칩 템플릿은 실제 칩을 0.95(기준 0.62)로 잡는다. 색보다 훨씬 확실하므로,
    # 템플릿이 있으면 색은 보지 않는다. 템플릿이 아예 없는 사람만 색으로 찾는다.
    if not goals and orange_goal_without_template and not tpl["goal"]:
        # 템플릿이 없는 경우의 최후 수단: 주황 카드 색으로 찾는다.
        # **한 판에 주황칩이 여러 개 놓일 수 있다**(실측: 2개). 가장 진한 하나만
        # 고르면 프레임마다 다른 칩이 뽑혀 목표가 왔다 갔다 한다. 전부 모아 두고
        # 어느 칩부터 먹을지는 경로 계산에서 정한다.
        for r in range(N):
            for c in range(N):
                rect = grid.cell_rect(r, c)
                if cells[r][c] == Kind.OBSTACLE:
                    continue
                # 스프라이트 마스크를 못 만든 경우에만 예전처럼 칸째로 뺀다.
                if player is not None and player.sprite is None                         and _overlaps_player(rect, player):
                    continue
                if player and (r, c) == (player.row, player.col):
                    continue
                if (r, c) in item_by_template:
                    continue      # 템플릿으로 아이템이 확인된 칸. 칩이 아니다.
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
            # 플레이어 픽셀은 이미 마스크에서 빠졌다. 마스크를 못 만들었을 때만
            # 예전처럼 걸친 칸을 통째로 뺀다.
            if player is not None and player.sprite is None                     and _overlaps_player(rect, player):
                continue
            # 템플릿 검사는 위(칩보다 먼저)에서 이미 끝냈다. 그 결과를 쓴다.
            found = item_by_template.get((r, c))
            if found is not None:
                cells[r][c] = Kind.ITEM
                detections.append(found)
                continue
            # 색 추측은 **아이템 템플릿이 하나도 없을 때만** 쓴다.
            # 실측: 템플릿이 있는데도 색까지 같이 보면, 디지몬 애니메이션의 주황
            # 픽셀이 순간적으로 새어 엉뚱한 칸이 2/15 프레임쯤 아이템으로 깜빡였다.
            # 템플릿이 있으면 그게 더 확실한 증거이므로 색은 보지 않는다.
            if tpl["item"]:
                continue
            if _frac(m_warm, rect) >= ITEM_WARM_FRAC:
                cells[r][c] = Kind.ITEM
                detections.append(Detection(Kind.ITEM, r, c, 0.5, rect, "따뜻한 색 비율"))

    # --- 5) 플레이어 칸 확정 ---------------------------------------------
    if player:
        cells[player.row][player.col] = Kind.PLAYER
        detections.append(player)

    return Scene(grid=grid, cells=cells, player=player, goal=goal, goals=goals,
                 detections=detections, highlights=highlights, notes=notes,
                 item_kinds={(d.row, d.col): d.item_kind
                             for d in detections if d.kind is Kind.ITEM})


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


# 안내문이 뜨는 세로 범위 (화면 높이 대비).
# 실측: 안내문 상자가 0.47~0.53 에 있었다. 넉넉히 잡아도 화면의 1/5 면 된다.
TOAST_BAND = (0.40, 0.60)


def find_blocked_toast(img: np.ndarray, tset: TemplateSet):
    """'해당 위치로 이동할 수 없습니다' 안내문을 찾는다 (클릭하면 안 되는 대상).

    이동 확인 루프에서 되풀이해 불리는 자리라 속도가 곧 이동 확인 횟수다.
    그래서 두 가지로 줄인다.

      1. **안내문이 뜨는 띠만 본다.** 화면 세로 0.40~0.60 밖은 볼 이유가 없다.
      2. match_big (축소 선별 -> 원본 확인) 으로 훑는다.

    실측 (709x1260 화면)
        전체 화면        75.7ms   안내문 0.869 / 없을 때 최대 0.549
        0.40~0.60 띠     48.2ms   안내문 0.869 / 없을 때 최대 0.266

    띠로 자르면 빨라지는 데다 **오탐 여유까지 넓어진다.** 화면 다른 곳의
    비슷한 무늬를 아예 보지 않기 때문이다.

    반환하는 상자는 **원본 화면 좌표**다(띠 오프셋을 더해 돌려준다).
    """
    if not tset or img is None or img.size == 0:
        return 0.0, None
    h = img.shape[0]
    top = int(h * TOAST_BAND[0])
    band = img[top:int(h * TOAST_BAND[1])]
    if band.size == 0:
        return 0.0, None
    score, box, _ = match_big(band, tset, scales=(0.85, 1.0, 1.15))
    if box is not None:
        box = (box[0], box[1] + top, box[2], box[3] + top)
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
