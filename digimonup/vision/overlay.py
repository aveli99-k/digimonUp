"""디버그 오버레이 그리기.

인식 결과가 맞는지는 숫자가 아니라 **그림으로** 확인해야 한다.
오버레이에는 다음을 그린다.
  - 5x5 게임판 경계
  - 실제로 검출된 격자선 (등간격 가정이 아니라 검출값)
  - 플레이어 / 장애물 / 목적지 / 아이템 위치와 신뢰도
  - 계산된 이동 경로
"""

from __future__ import annotations

import cv2
import numpy as np

from digimonup.vision.board import Grid
from digimonup.base.imgio import imwrite
from digimonup.vision.recognize import Kind, Scene

COLOR = {
    Kind.PLAYER: (0, 215, 255),     # 주황
    Kind.GOAL: (0, 255, 0),         # 초록
    Kind.OBSTACLE: (0, 0, 255),     # 빨강
    Kind.ITEM: (255, 255, 0),       # 하늘
    Kind.EMPTY: (90, 90, 90),
}
BOARD_COLOR = (255, 255, 0)
GRID_COLOR = (200, 200, 60)
PATH_COLOR = (255, 0, 255)

# 한글은 OpenCV 기본 폰트로 못 그리므로 라벨은 알파벳/기호로 쓴다.
LABEL = {Kind.PLAYER: "P", Kind.GOAL: "GOAL", Kind.OBSTACLE: "X",
         Kind.ITEM: "item", Kind.EMPTY: ""}


def draw(img: np.ndarray, grid: Grid | None = None, scene: Scene | None = None,
         path: list[tuple[int, int]] | None = None, header: str = "") -> np.ndarray:
    out = img.copy()
    if grid is not None:
        _draw_grid(out, grid)
    if scene is not None:
        _draw_scene(out, scene)
    if grid is not None and path:
        _draw_path(out, grid, path)
    if header:
        _draw_header(out, header)
    return out


def _draw_grid(img: np.ndarray, grid: Grid) -> None:
    x0, y0, x1, y1 = grid.bbox
    for x in grid.xs:
        cv2.line(img, (x, y0), (x, y1), GRID_COLOR, 1, cv2.LINE_AA)
    for y in grid.ys:
        cv2.line(img, (x0, y), (x1, y), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), BOARD_COLOR, 2)
    cv2.putText(img, f"board conf {grid.confidence:.2f}", (x0, max(14, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOARD_COLOR, 1, cv2.LINE_AA)


def _draw_scene(img: np.ndarray, scene: Scene) -> None:
    grid = scene.grid
    for r, c in scene.highlights:
        x0, y0, x1, y1 = grid.cell_rect(r, c)
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (255, 190, 90), 1)

    for det in scene.detections:
        x0, y0, x1, y1 = grid.cell_rect(det.row, det.col)
        col = COLOR.get(det.kind, (255, 255, 255))
        cv2.rectangle(img, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), col, 2)
        cv2.putText(img, f"{LABEL.get(det.kind,'')} {det.confidence:.2f}",
                    (x0 + 5, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    if scene.player and scene.player.bbox:
        bx0, by0, bx1, by1 = scene.player.bbox
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 215, 255), 1)
        # 논리 기준점(발끝보다 살짝 위)
        anchor_y = int(by1 - grid.cell_h * 0.20)
        cv2.drawMarker(img, ((bx0 + bx1) // 2, anchor_y), (0, 215, 255),
                       cv2.MARKER_CROSS, 14, 2)


def _draw_path(img: np.ndarray, grid: Grid, path: list[tuple[int, int]]) -> None:
    pts = [grid.cell_center(r, c) for r, c in path]
    for i in range(len(pts) - 1):
        cv2.arrowedLine(img, pts[i], pts[i + 1], PATH_COLOR, 2, cv2.LINE_AA, tipLength=0.25)
    for i, (px, py) in enumerate(pts):
        cv2.circle(img, (px, py), 4, PATH_COLOR, -1)
        cv2.putText(img, str(i), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, PATH_COLOR, 1, cv2.LINE_AA)


def _draw_header(img: np.ndarray, text: str) -> None:
    lines = text.split("\n")
    h = 6 + 16 * len(lines)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)


def save(path: str, img: np.ndarray) -> bool:
    """디버그 오버레이 저장. 한글 경로 처리는 imgio 가 맡는다."""
    return imwrite(path, img)
