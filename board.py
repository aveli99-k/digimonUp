"""5x5 게임판 격자 검출.

게임판 격자선은 '밝기 경계'가 아니라 **채도(S)가 잠깐 내려앉는 1~2px 선**이다.
(실측: 셀 내부 S=245 -> 격자선 위 S=171, 색상 H 는 그대로 100 근처)

그래서 흑백 Sobel 로는 도시 배경/UI 글자에 묻혀 버린다. 대신
  1) 게임판 색(청록 계열 H 93~112) 픽셀만 남기고
  2) 채도 채널에 black-hat 을 걸어 '가늘고 어두운 선'만 뽑아낸 뒤
  3) 등간격 6줄 모델을 맞추고
  4) 원근 때문에 아래로 갈수록 간격이 조금씩 줄어드는 것을 보정하려고
     각 선을 실제 검출 피크에 스냅한다
는 순서로 찾는다.

단순 고정 비율은 쓰지 않는다. 실제로 검출한 격자선 좌표가 항상 우선이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from imgio import hsv_of

N = 5                     # 5x5
NLINES = N + 1            # 격자선 6개

# 게임판(청록 계열) 색 범위. 빈칸/이동가능 강조칸 모두 포함한다.
BOARD_H_LO, BOARD_H_HI = 93, 112
BOARD_S_MIN = 100
BOARD_V_MIN = 50


@dataclass
class Grid:
    """검출된 5x5 게임판."""
    xs: list[int]            # 세로 격자선 6개 (클라이언트 좌표)
    ys: list[int]            # 가로 격자선 6개
    confidence: float        # 0~1
    detail: dict             # 디버그용 세부 점수

    # -- 기본 정보 --------------------------------------------------------
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]

    @property
    def cell_w(self) -> float:
        return (self.xs[-1] - self.xs[0]) / N

    @property
    def cell_h(self) -> float:
        return (self.ys[-1] - self.ys[0]) / N

    def cell_rect(self, row: int, col: int) -> tuple[int, int, int, int]:
        return self.xs[col], self.ys[row], self.xs[col + 1], self.ys[row + 1]

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.cell_rect(row, col)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def clamp_locate(self, x: float, y: float) -> tuple[int, int]:
        """게임판을 살짝 벗어나도 가장 가까운 셀로 붙여서 돌려준다."""
        col = int(np.clip(np.searchsorted(self.xs, x, side="right") - 1, 0, N - 1))
        row = int(np.clip(np.searchsorted(self.ys, y, side="right") - 1, 0, N - 1))
        return row, col


# --------------------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------------------

def board_mask(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """게임판 색(청록 계열) 픽셀 마스크."""
    if hsv is None:
        hsv = hsv_of(img)
    # cv2.inRange 는 채널별 비교 배열을 따로 만들지 않아 훨씬 싸다.
    # 결과가 0/255 이므로 >>7 로 기존과 같은 0/1 마스크로 되돌린다.
    return cv2.inRange(hsv, (BOARD_H_LO, BOARD_S_MIN, BOARD_V_MIN),
                       (BOARD_H_HI, 255, 255)) >> 7


def strict_cell_mask(img: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    """'셀 안쪽' 색만 남기는 엄격한 마스크.

    board_mask 보다 채도 기준을 높여서, 게임판 위아래의 장식용 청록 블록 띠나
    도시 배경을 확실히 뺀다(실측: 장식 띠 S≈159, 셀 S≈245). 이걸로 게임판의
    실제 위/아래·좌/우 끝을 잡아내 격자 위상이 한 칸 밀리는 것을 막는다.
    """
    if hsv is None:
        hsv = hsv_of(img)
    return cv2.inRange(hsv, (95, 210, 0), (110, 255, 255)) >> 7


def _longest_run(coverage: np.ndarray, thresh: float = 0.15,
                 bridge: int = 10) -> tuple[int, int] | None:
    """coverage 가 thresh 를 넘는 가장 긴 구간. 짧은 끊김은 이어 붙인다."""
    on = coverage > thresh
    if not on.any():
        return None
    # 장애물이 많은 줄에서 잠깐 끊기는 것을 메운다.
    idx = np.flatnonzero(on)
    filled = on.copy()
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= bridge:
            filled[a:b] = True

    best = cur_start = None
    best_len = 0
    for i, v in enumerate(filled):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best_len:
                best, best_len = (cur_start, i - 1), i - cur_start
            cur_start = None
    if cur_start is not None and len(filled) - cur_start > best_len:
        best = (cur_start, len(filled) - 1)
    return best


def board_extent(img: np.ndarray, hsv: np.ndarray | None = None
                 ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """엄격한 셀 색 마스크로 잰 게임판의 (x범위, y범위).

    세로 범위를 먼저 구한 뒤, 가로 범위는 **그 행들 안에서만** 잰다.
    화면 전체 높이로 나누면 게임판이 차지하는 비율 자체가 낮아서, 플레이어
    스프라이트가 걸친 열 하나만으로도 구간이 끊긴다.
    (실측: 그래서 x 범위가 75~615 여야 하는데 257~614 로 잘렸고, 격자 후보
     채점이 흔들려 셀 폭을 71px 로 잘못 잡았다.)
    """
    m = strict_cell_mask(img, hsv)
    ext_y = _longest_run(m.mean(axis=1))
    if ext_y is None:
        return _longest_run(m.mean(axis=0)), None
    band = m[ext_y[0]:ext_y[1] + 1]
    if band.size == 0:
        return _longest_run(m.mean(axis=0)), ext_y
    return _longest_run(band.mean(axis=0)), ext_y


def _edge_alignment(ext: tuple[int, int] | None, lines: list[int],
                    step: int) -> float:
    """격자의 바깥 선이 게임판의 실제 가장자리와 얼마나 맞는지 (0~1).

    장애물이 많으면 셀 색이 이어지는 구간이 중간에 끊겨서 범위(IoU)만으로는
    한 칸 밀린 배치를 못 가른다(실측: 두 후보 모두 extent 0.559).
    하지만 **양 끝 중 한쪽은 대체로 정확히 잡힌다.** 그래서 가까운 쪽 가장자리에
    얼마나 붙어 있는지를 따로 본다.
    (실측: 게임판 위 끝이 y=420 인데, 맞는 배치는 419 에서 시작하고 한 칸 밀린
     배치는 332 에서 시작한다. 이 한 가지로 확실히 갈린다.)
    """
    if ext is None or step <= 0:
        return 0.5
    d = min(abs(lines[0] - ext[0]), abs(lines[-1] - ext[1]))
    return float(np.clip(1.0 - d / (step * 0.5), 0.0, 1.0))


def _interval_iou(a: tuple[int, int] | None, b: tuple[int, int]) -> float:
    if a is None:
        return 0.5      # 근거가 없으면 중립
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 0 else 0.0


def line_maps(img: np.ndarray, hsv: np.ndarray | None = None,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(세로선 강도맵, 가로선 강도맵). 채도 black-hat 을 게임판 색으로 마스킹.

    형태학 연산은 uint8 그대로 돌린다. S 값이 어차피 0~255 정수라 float32 로
    올려서 계산해도 결과가 같은데, 큰 부동소수 배열을 만드느라 시간만 더 썼다.
    닫힘(close) 결과는 항상 원본 이상이므로 uint8 뺄셈에서 음수가 날 수 없다.
    """
    if hsv is None:
        hsv = hsv_of(img)
    s = hsv[:, :, 1]                      # S 채널
    if mask is None:
        mask = board_mask(img, hsv)

    # 가로 방향으로 닫아주면 '세로로 뻗은 가는 어두운 선'만 남는다.
    vert = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((1, 11), np.uint8)) - s
    horz = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((11, 1), np.uint8)) - s
    return vert * mask, horz * mask


def _fit_candidates(profile: np.ndarray, n: int, step_lo: int, step_hi: int,
                    topk: int = 6) -> list[tuple[float, int, int]]:
    """등간격 n 줄 모델 후보를 점수 높은 순으로 topk 개 돌려준다.

    바깥 테두리가 이 강도맵에 잡히지 않는 판에서는 '한 칸 밀린' 배치도
    안쪽 선들을 똑같이 지나가서 점수가 거의 같다. 그래서 하나만 고르지 않고
    후보를 여러 개 남긴 뒤, 셀 내부가 실제 게임판 색인지까지 보고 고른다.
    """
    p = np.clip(profile - np.median(profile), 0, None)
    length = len(p)
    # 선이 1~2px 라 살짝 번지게 해서 반올림 오차에 덜 민감하게 만든다.
    p = np.maximum(p, np.roll(p, 1))
    p = np.maximum(p, np.roll(p, -1))

    found: list[tuple[float, int, int]] = []
    for step in range(step_lo, step_hi + 1):
        span = (n - 1) * step
        if span >= length:
            break
        # 시작점마다 n 개를 뽑아 더하는 대신, 프로파일을 step 만큼씩 밀어서 통째로
        # 더한다. (시작점 x n) 크기의 색인 행렬을 만들고 gather 하는 비용이 사라진다.
        # 더하는 순서가 같아 결과값은 완전히 동일하다.
        m = length - span
        scores = p[:m].copy()
        for j in range(1, n):
            scores += p[step * j:step * j + m]
        # 같은 간격에서는 위상만 다른 후보 몇 개를 남긴다.
        order = np.argsort(scores)[::-1][:3]
        for i in order:
            found.append((float(scores[i]), int(i), step))

    found.sort(key=lambda t: -t[0])
    picked: list[tuple[float, int, int]] = []
    for score, start, step in found:
        # 거의 같은 배치는 중복으로 보고 건너뛴다.
        if any(abs(start - s2) < step * 0.25 and abs(step - st2) < max(2, step * 0.06)
               for _, s2, st2 in picked):
            continue
        picked.append((score, start, step))
        if len(picked) >= topk:
            break
    return picked


def _snap(profile: np.ndarray, lines: list[int], tol: int) -> list[int]:
    """원근 때문에 간격이 균일하지 않으므로 각 선을 실제 피크로 끌어당긴다.

    멀리 있는 더 센 피크(패널 테두리 등)에 끌려가지 않도록, 등간격 모델이 예측한
    자리에서 멀어질수록 점수를 깎는 삼각 가중치를 준다.

    게임판의 바깥 테두리는 안쪽 격자선과 달리 채도 선이 아니라 패널 가장자리라서
    이 강도맵에 잡히지 않는다. 그런 선까지 억지로 스냅하면 오히려 어긋나므로,
    주변 피크가 다른 선들에 비해 충분히 세지 않으면 등간격 모델 값을 그대로 쓴다.
    """
    p = np.clip(profile - np.median(profile), 0, None)
    cand: list[tuple[int, float]] = []
    for x in lines:
        lo, hi = max(0, x - tol), min(len(p), x + tol + 1)
        if hi <= lo or p[lo:hi].max() <= 0:
            cand.append((x, 0.0))
            continue
        local = p[lo:hi].astype(np.float32)
        dist = np.abs(np.arange(lo, hi) - x)
        weighted = local * (1.0 - 0.6 * dist / max(1, tol))
        i = int(np.argmax(weighted))
        cand.append((lo + i, float(local[i])))

    strengths = [s for _, s in cand if s > 0]
    floor = 0.4 * float(np.median(strengths)) if strengths else 0.0
    out = [pos if s >= floor else model
           for (pos, s), model in zip(cand, lines)]

    # 단조 증가 보장 (스냅이 순서를 뒤집는 일은 없어야 한다)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    return out


def _line_quality(profile: np.ndarray, lines: list[int]) -> float:
    """격자선 위 에너지가 주변보다 얼마나 두드러지는지 (0~1)."""
    p = np.clip(profile - np.median(profile), 0, None)
    if p.max() <= 0:
        return 0.0
    on = np.mean([p[max(0, x - 1):x + 2].max() for x in lines])
    return float(np.clip(on / (p.max() + 1e-6), 0.0, 1.0))


def _spacing_quality(lines: list[int]) -> float:
    """간격이 얼마나 규칙적인지 (0~1). 원근으로 조금 줄어드는 건 허용."""
    d = np.diff(lines).astype(float)
    if len(d) == 0 or d.mean() <= 0:
        return 0.0
    return float(np.clip(1.0 - (d.std() / d.mean()) / 0.25, 0.0, 1.0))


def _purity(profile: np.ndarray, lines: list[int], tol: int = 2) -> float:
    """셀 '안쪽'에 격자선이 남아 있지 않은 정도 (0~1).

    이게 없으면 간격을 2배로 잡은 배치(6줄 중 3줄만 진짜 격자선인 경우)가
    더 넓은 영역을 덮는다는 이유로 이길 수 있다. 진짜 격자라면 셀 내부에는
    선이 없어야 하므로, 선 위 에너지와 셀 내부 에너지의 비를 본다.
    """
    p = np.clip(profile - np.median(profile), 0, None)
    lo, hi = lines[0], lines[-1]
    if hi - lo < 4:
        return 0.0
    inside = np.ones(hi - lo, dtype=bool)
    for x in lines:
        a, b = max(lo, x - tol) - lo, min(hi, x + tol + 1) - lo
        if b > a:
            inside[a:b] = False
    on = np.mean([p[max(0, x - tol):x + tol + 1].max() for x in lines])
    within = p[lo:hi][inside]
    off = float(within.mean()) if within.size else 0.0
    if on <= 0:
        return 0.0
    return float(np.clip(on / (on + off * 6.0), 0.0, 1.0))


# --------------------------------------------------------------------------
# 메인
# --------------------------------------------------------------------------

def _plausible(lines: list[int], limit: int, step: int) -> bool:
    """이 6줄 배치가 애초에 게임판일 수 있는가.

    한 칸 밀린 가짜 후보는 진짜와 점수가 0.001 밖에 차이 나지 않아 사실상
    동전 던지기가 된다(실측). 그래서 점수를 매기기 전에 구조적으로 불가능한
    배치를 먼저 걸러낸다.

      - 게임판은 UI 패널 안에 있어서 창 가장자리에 딱 붙지 않는다.
        가짜 후보는 마지막 선이 창 밖으로 나갔다가 스냅되면서 가장자리에
        달라붙는다(실측: xs[-1] = 708, 클라이언트 폭 709).
      - 그때 마지막 칸 간격만 눈에 띄게 좁아진다(실측: 108,108,107,108,93).
        원근으로 줄어드는 정도를 넘어서면 격자가 아니다.
    """
    if lines[0] < 2 or lines[-1] > limit - 3:
        return False
    gaps = np.diff(lines)
    if gaps.min() < step * 0.7 or gaps.max() > step * 1.3:
        return False
    return True


def _coverage(mask: np.ndarray, xs: list[int], ys: list[int]) -> float:
    """25개 셀 중 내부가 실제 게임판 색인 셀의 비율.

    도시 배경이나 UI 를 격자로 오인하는 것을 막는 가장 강한 근거다.
    """
    covered = 0
    for r in range(N):
        for c in range(N):
            x0, y0, x1, y1 = xs[c], ys[r], xs[c + 1], ys[r + 1]
            iy, ix = int((y1 - y0) * 0.2), int((x1 - x0) * 0.2)
            patch = mask[y0 + iy:y1 - iy, x0 + ix:x1 - ix]
            if patch.size and patch.mean() > 0.25:
                covered += 1
    return covered / (N * N)


def detect_board(img: np.ndarray, min_confidence: float = 0.45) -> Grid | None:
    """화면(클라이언트 캡처)에서 5x5 게임판을 찾는다.

    반환하는 좌표는 모두 입력 이미지(=클라이언트) 좌표계다.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    # HSV 변환은 한 번이면 된다. 예전에는 board_mask / strict_cell_mask /
    # line_maps 가 각자 변환해서 같은 프레임을 세 번씩 훑었다.
    hsv = hsv_of(img)
    mask = board_mask(img, hsv)
    vert, horz = line_maps(img, hsv, mask)

    col_profile = vert.mean(axis=0)
    row_profile = horz.mean(axis=1)

    # 셀 하나가 화면 폭의 8~30% 사이라고 보고 간격을 훑는다.
    col_cands = _fit_candidates(col_profile, NLINES,
                                max(8, int(w * 0.08)), max(9, int(w * 0.30)))
    row_cands = _fit_candidates(row_profile, NLINES,
                                max(8, int(h * 0.03)), max(9, int(h * 0.20)))
    if not col_cands or not row_cands:
        return None

    ext_x, ext_y = board_extent(img, hsv)
    best: Grid | None = None

    for _, cx0, cstep in col_cands:
        xs = _snap(col_profile, [cx0 + cstep * i for i in range(NLINES)],
                   max(3, cstep // 8))
        if not _plausible(xs, w, cstep):
            continue

        for _, ry0, rstep in row_cands:
            ys = _snap(row_profile, [ry0 + rstep * i for i in range(NLINES)],
                       max(3, rstep // 8))
            if not _plausible(ys, h, rstep):
                continue

            q_line = 0.5 * (_line_quality(col_profile, xs) +
                            _line_quality(row_profile, ys))
            q_space = 0.5 * (_spacing_quality(xs) + _spacing_quality(ys))
            q_cover = _coverage(mask, xs, ys)
            q_purity = 0.5 * (_purity(col_profile, xs) + _purity(row_profile, ys))
            # 격자가 실제 게임판 범위와 겹치는 정도. 위상이 한 칸 밀린 배치를
            # 걸러내는 가장 결정적인 근거다.
            q_extent = 0.5 * (_interval_iou(ext_x, (xs[0], xs[-1])) +
                              _interval_iou(ext_y, (ys[0], ys[-1])))
            q_edge = 0.5 * (_edge_alignment(ext_x, xs, cstep) +
                            _edge_alignment(ext_y, ys, rstep))
            # 셀은 대체로 정사각형에 가깝다. 원근 때문에 가로가 조금 넓어지는
            # 정도는 봐주고, 비율이 크게 어긋나면 부드럽게 감점한다.
            ratio = (cstep / rstep) if rstep else 0.0
            q_ratio = (float(np.exp(-(np.log(ratio) ** 2) / (2 * 0.35 ** 2)))
                       if ratio > 0 else 0.0)

            confidence = float(np.clip(0.15 * q_line + 0.10 * q_space +
                                       0.10 * q_cover + 0.05 * q_ratio +
                                       0.25 * q_purity + 0.10 * q_extent +
                                       0.25 * q_edge, 0.0, 1.0))
            if best is None or confidence > best.confidence:
                best = Grid(xs=xs, ys=ys, confidence=confidence,
                            detail={"line": round(q_line, 3),
                                    "spacing": round(q_space, 3),
                                    "cover": round(q_cover, 3),
                                    "purity": round(q_purity, 3),
                                    "extent": round(q_extent, 3),
                                    "edge": round(q_edge, 3),
                                    "ratio": round(ratio, 2),
                                    "cell": (cstep, rstep)})

    if best is None or best.confidence < min_confidence:
        return None
    return best
