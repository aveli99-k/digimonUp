"""테스트용 게임 화면 합성기.

실제 캡처(709x1260)에서 잰 HSV 값을 그대로 써서 게임판을 그린다.
덕분에 실물 없이도 다음 상황을 재현할 수 있다.
  - 좌우 반전 / 애니메이션 중인 플레이어
  - 발이 셀 경계를 넘은 플레이어
  - 맨 아래 행에서 테두리에 발이 가려진 플레이어
  - 위 장애물에 머리가 가려진 플레이어
  - 하단 테두리에 가려진 목적지

실측 HSV
    빈칸        (101, 245, 103)
    강조칸      (102, 241, 203)
    격자선      (100, 171, 112)   <- 채도만 내려앉는 가는 선
    장애물      (116, 110, 134)
    주황 카드   ( 15, 205, 225)
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 709, 1260
XS = [75, 184, 292, 400, 507, 615]
YS = [419, 507, 595, 684, 772, 859]

HSV_EMPTY = (101, 245, 103)
HSV_HIGH = (102, 241, 203)
HSV_LINE = (100, 171, 112)
HSV_OBSTACLE = (116, 110, 134)
HSV_ORANGE = (15, 205, 225)
# 게임판 밖(도시 배경). 게임판 색 범위(H 93~112, S>=100)에 들지 않아야
# '셀 내부가 실제 게임판 색인지' 검사가 제 역할을 한다.
HSV_BACKDROP = (128, 70, 45)


def _hsv_canvas() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = HSV_BACKDROP
    return img


def _cell(img, r, c, hsv):
    img[YS[r]:YS[r + 1], XS[c]:XS[c + 1]] = hsv


def _lines(img):
    for x in XS[1:-1]:
        img[YS[0]:YS[-1], x:x + 2] = HSV_LINE
    for y in YS[1:-1]:
        img[y:y + 2, XS[0]:XS[-1]] = HSV_LINE


def _pyramid(img, r, c):
    """장애물. 밑변은 자기 칸 바닥, 꼭대기는 위 칸으로 삐져나온다."""
    x0, x1 = XS[c], XS[c + 1]
    y1 = YS[r + 1]
    apex_y = YS[r] - int((YS[r + 1] - YS[r]) * 0.22)
    pts = np.array([[(x0 + x1) // 2, apex_y],
                    [x0 + 4, y1 - 3], [x1 - 4, y1 - 3]], np.int32)
    cv2.fillPoly(img, [pts], HSV_OBSTACLE)


def _card(img, r, c, shrink=0.45, clip_bottom=0):
    """주황 카드(목적지/아이템). clip_bottom>0 이면 아래가 잘려 가려진 모양."""
    x0, y0, x1, y1 = XS[c], YS[r], XS[c + 1], YS[r + 1]
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * (1 - shrink) / 2), int(h * (1 - shrink) / 2)
    cv2.rectangle(img, (x0 + mx, y0 + my), (x1 - mx, y1 - my - clip_bottom),
                  HSV_ORANGE, -1)


def _player(img, r, c, flip=False, dy=0, dx=0, foot_overflow=0.0,
            occlude_top=0, occlude_bottom=0, scale=1.0):
    """플레이어 스프라이트.

    몸통은 채도가 낮고, 머리/날개/무기에 따뜻한 색이 섞여 있다.
    foot_overflow 는 발이 아래 셀로 넘어가는 정도(셀 높이 비율).
    """
    x0, y0, x1, y1 = XS[c], YS[r], XS[c + 1], YS[r + 1]
    cw, ch = x1 - x0, y1 - y0
    pw, ph = int(cw * 0.62 * scale), int(ch * 1.25 * scale)
    cx = (x0 + x1) // 2 + dx
    feet = y1 + int(ch * foot_overflow) + dy
    head = feet - ph

    body = (int(cx - pw / 2), head + int(ph * 0.28), int(cx + pw / 2), feet - int(ph * 0.12))
    # 몸통: 채도 낮은 회색빛. 단색이면 템플릿 매칭이 성립하지 않으므로
    # (분산 0인 템플릿은 정규화 상관계수가 정의되지 않는다) 무늬를 넣는다.
    cv2.rectangle(img, (body[0], body[1]), (body[2], body[3]), (60, 40, 110), -1)
    bh = body[3] - body[1]
    for i, band in enumerate((0.18, 0.42, 0.66)):
        y = body[1] + int(bh * band)
        cv2.rectangle(img, (body[0] + 3, y), (body[2] - 3, y + max(3, bh // 12)),
                      (95, 90, 150 - i * 25), -1)
    cv2.circle(img, ((body[0] + body[2]) // 2, body[1] + int(bh * 0.3)),
               max(4, pw // 8), (30, 180, 200), -1)
    # 머리/날개: 따뜻한 색 (좌우 반전 시 위치가 바뀐다)
    hx = cx + (int(pw * 0.22) * (-1 if flip else 1))
    cv2.rectangle(img, (hx - int(pw * 0.3), head),
                  (hx + int(pw * 0.3), head + int(ph * 0.3)), (5, 220, 230), -1)
    # 발
    cv2.rectangle(img, (cx - int(pw * 0.35), feet - int(ph * 0.12)),
                  (cx + int(pw * 0.35), feet), (20, 200, 200), -1)

    if occlude_top:      # 위 장애물이 머리를 가림
        img[head:head + occlude_top, x0:x1] = HSV_OBSTACLE
    if occlude_bottom:   # 아래 테두리가 발을 가림
        img[max(0, feet - occlude_bottom):feet + 6, x0:x1] = HSV_BACKDROP


def make_board(layout: list[str], *, player_kw: dict | None = None,
               highlight_player: bool = True, goal_clip: int = 0) -> np.ndarray:
    """레이아웃 문자열로 게임 화면을 만든다.

    문자: '.' 빈칸  'X' 장애물  'P' 플레이어  'G' 목적지  'i' 아이템
    """
    img = _hsv_canvas()
    pr = pc = None
    for r, line in enumerate(layout):
        for c, ch in enumerate(line.replace(" ", "")):
            _cell(img, r, c, HSV_EMPTY)
            if ch == "P":
                pr, pc = r, c

    if highlight_player and pr is not None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = pr + dr, pc + dc
            if 0 <= r < 5 and 0 <= c < 5 and layout[r].replace(" ", "")[c] != "X":
                _cell(img, r, c, HSV_HIGH)

    _lines(img)

    for r, line in enumerate(layout):
        for c, ch in enumerate(line.replace(" ", "")):
            if ch == "X":
                _pyramid(img, r, c)
            elif ch == "G":
                _card(img, r, c, clip_bottom=goal_clip)
            elif ch == "i":
                _card(img, r, c, shrink=0.35)

    if pr is not None:
        _player(img, pr, pc, **(player_kw or {}))

    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def make_non_game_window() -> np.ndarray:
    """게임판이 없는 창(브라우저 등)을 흉내낸 이미지."""
    img = np.full((900, 1200, 3), 240, np.uint8)
    cv2.rectangle(img, (0, 0), (1200, 60), (200, 200, 200), -1)
    for i in range(12):
        cv2.putText(img, "some text line", (40, 120 + i * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    return img


def make_top_tab(img: np.ndarray, box=(230, 60, 470, 130)) -> np.ndarray:
    """상단에 고정된 '게임 탭' 비슷한 마크를 그려 넣는다."""
    out = img.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1, y1), (40, 90, 210), -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), (250, 250, 250), 3)
    cv2.putText(out, "TAB", (x0 + 30, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3)
    return out


def crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()
