"""경로 계산.

탐사는 **오른쪽으로 무한히 나아가는 미니게임**이다. 정해진 종착점이 없고,
한 칸 움직일 때마다 게임판이 반대쪽으로 밀리면서 새 열/새 행이 들어온다.
이 성질이 경로 규칙 전체를 결정한다.

플레이어는 상하좌우로 한 칸씩만 움직인다. 우선순위는 다음과 같다.

  1순위  목적지가 인식되면 그리로 가는 장애물 없는 최단 경로
         (탐사에는 보통 목적지가 없다. 템플릿을 넣었을 때만 동작한다.)
  2순위  **도달 가능한 가장 오른쪽 칸**까지 가는 장애물 없는 최단 경로
  3순위  오른쪽으로 한 칸도 못 갈 때, 세로로 최대한 움직여 새 행을 불러온다
  4순위  오른쪽으로도 세로로도 갈 데가 없을 때만 장애물 클릭

2순위가 '오른쪽 끝 열(col 4)'이 아니라 '도달 가능한 가장 오른쪽 칸'인 것이 중요하다.
지금 보이는 5x5 안에서 맨 오른쪽 열에 못 닿아도 막힌 것이 아니다. 오른쪽으로 한 칸만
가도 게임판이 밀리며 새 열이 들어와 길이 이어진다. 끝 열만 목표로 삼으면, 사실은
전진할 수 있는데도 '갇혔다'고 판단해 장애물을 두드리게 된다(실제로 그랬다).

핵심: 오른쪽으로 바로 못 가더라도 위나 아래로 우회할 수 있으면 장애물을 부수지 않는다.
BFS 는 상하좌우 전부를 훑으므로 우회로가 있으면 반드시 찾아낸다.

3개 연속 장애물 특수 조건은 **가로 방향 XXX 배열에만** 적용한다.
세로로 3개가 늘어선 경우는 특수 취급하지 않는다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from board import N
from recognize import Kind, Scene

Cell = tuple[int, int]
DIRS: list[tuple[str, int, int]] = [
    ("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1),
]


class PlanKind(str, Enum):
    GOAL = "목적지"
    RIGHT_EDGE = "오른쪽 전진"
    SCROLL_VERTICAL = "세로 이동으로 새 지형"
    BREAK_OBSTACLE = "장애물 파괴"
    NONE = "없음"


@dataclass
class Plan:
    kind: PlanKind
    path: list[Cell]              # 플레이어 칸부터 시작하는 셀 목록
    target: Cell | None
    reason: str

    @property
    def moves(self) -> list[str]:
        """경로를 방향 목록으로 바꾼다."""
        out = []
        for (r0, c0), (r1, c1) in zip(self.path, self.path[1:]):
            for name, dr, dc in DIRS:
                if (r1 - r0, c1 - c0) == (dr, dc):
                    out.append(name)
                    break
        return out

    def describe(self) -> str:
        if not self.path:
            return f"{self.kind.value}: {self.reason}"
        route = " -> ".join(f"({r},{c})" for r, c in self.path)
        return f"{self.kind.value} [{len(self.path) - 1}칸] {route}  | {self.reason}"


def passable(kind: Kind) -> bool:
    """장애물만 못 지나간다. 아이템/목적지/빈칸은 지나갈 수 있다."""
    return kind != Kind.OBSTACLE


def _bfs(cells: list[list[Kind]], start: Cell) -> tuple[dict[Cell, int], dict[Cell, Cell]]:
    """장애물을 피해 상하좌우로만 이동하는 BFS. (거리, 이전칸) 을 돌려준다."""
    dist: dict[Cell, int] = {start: 0}
    prev: dict[Cell, Cell] = {}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for _, dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if (nr, nc) in dist:
                continue
            if not passable(cells[nr][nc]):
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))
    return dist, prev


def _rebuild(prev: dict[Cell, Cell], start: Cell, end: Cell) -> list[Cell]:
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _horizontal_runs(cells: list[list[Kind]], length: int = 3
                     ) -> list[tuple[int, int]]:
    """가로로 **정확히** length 개 연속인 장애물 구간의 (행, 시작열) 목록."""
    out: list[tuple[int, int]] = []
    for r in range(N):
        c = 0
        while c < N:
            if cells[r][c] != Kind.OBSTACLE:
                c += 1
                continue
            start = c
            while c < N and cells[r][c] == Kind.OBSTACLE:
                c += 1
            if c - start == length:
                out.append((r, start))
    return out


def horizontal_triples(cells: list[list[Kind]]) -> set[Cell]:
    """가로 3연속(XXX)의 **가운데 칸** 집합.

    세로 3연속은 여기에 포함하지 않는다(특수 조건은 가로에만 적용).
    4개 이상 연속도 XXX 가 아니므로 포함하지 않는다.
    """
    return {(r, start + 1) for r, start in _horizontal_runs(cells, 3)}


def horizontal_triple_members(cells: list[list[Kind]]) -> set[Cell]:
    """가로 3연속(XXX)에 속한 **모든 칸** 집합.

    가운데 칸에 닿을 수 없을 때, 같은 줄의 끝쪽이라도 우선해서 부수려고 쓴다.
    """
    return {(r, start + i) for r, start in _horizontal_runs(cells, 3)
            for i in range(3)}


def plan_route(scene: Scene) -> Plan:
    """장면을 보고 다음에 무엇을 할지 정한다."""
    if scene.player is None:
        return Plan(PlanKind.NONE, [], None, "플레이어를 찾지 못했습니다.")

    start: Cell = (scene.player.row, scene.player.col)
    cells = scene.cells
    dist, prev = _bfs(cells, start)

    # --- 1순위: 주황칩(필수 아이템) ----------------------------------------
    # 판에 칩이 여러 개일 수 있다. **가장 가까운 것부터** 먹는다. 매번 '가장
    # 진한 칩'을 고르면 프레임마다 목표가 바뀌어 제자리를 오간다.
    goal_cells = [(d.row, d.col) for d in scene.goals]
    if not goal_cells and scene.goal is not None:
        goal_cells = [(scene.goal.row, scene.goal.col)]
    reachable_goals = [g for g in goal_cells if g in dist and g != start]
    if reachable_goals:
        # 가깝고, 같으면 더 오른쪽에 있는 칩을 먼저
        goal = min(reachable_goals, key=lambda c: (dist[c], -c[1]))
        extra = f" (칩 {len(goal_cells)}개 중 가장 가까운 것)" if len(goal_cells) > 1 else ""
        return Plan(PlanKind.GOAL, _rebuild(prev, start, goal), goal,
                    f"주황칩까지 장애물 없는 최단 경로 {dist[goal]}칸{extra}")
    # 칩이 없거나 전부 막혀 있으면 오른쪽 전진 규칙으로 내려간다.

    # --- 2순위: 오른쪽으로 갈 수 있는 데까지 --------------------------------
    # 탐사는 오른쪽으로 무한히 나아가는 미니게임이고, 한 칸 이동할 때마다
    # 게임판이 밀리면서 **새 열이 하나 들어온다**. 그래서 지금 보이는 5x5 안에서
    # 맨 오른쪽 열(col 4)에 못 닿는다고 막힌 것이 아니다. 갈 수 있는 만큼만
    # 오른쪽으로 가면 새 지형이 나타나 길이 이어진다.
    #
    # 따라서 목표는 '오른쪽 끝 열'이 아니라 **도달 가능한 가장 오른쪽 칸**이다.
    reachable = [cell for cell in dist if cell != start]
    if reachable:
        # 더 오른쪽 > 더 가까움 > 원래 행에 가까움 순으로 고른다.
        best = max(reachable, key=lambda cell: (cell[1], -dist[cell],
                                                -abs(cell[0] - start[0])))
        if best[1] > start[1]:
            reason = (f"오른쪽으로 {best[1] - start[1]}열 전진 "
                      f"(도달 가능한 가장 오른쪽 칸까지 {dist[best]}칸)")
            if best[1] < N - 1:
                reason += " (한 칸 가면 새 열이 들어온다)"
            return Plan(PlanKind.RIGHT_EDGE, _rebuild(prev, start, best), best, reason)

    # --- 3순위: 세로로 움직여 새 지형을 불러온다 ----------------------------
    # 오른쪽으로 한 칸도 못 가는 '주머니'에 갇힌 경우다. 세로 이동에서도 게임판이
    # 스크롤하면서 **새 행이 들어오므로**, 위아래로 최대한 움직이면 오른쪽으로
    # 이어지는 길이 새로 나타날 수 있다. 장애물을 부수는 것보다 이쪽이 먼저다.
    vertical = [cell for cell in dist if cell[0] != start[0]]
    if vertical:
        best = max(vertical, key=lambda cell: (abs(cell[0] - start[0]), -dist[cell],
                                               cell[1]))
        return Plan(PlanKind.SCROLL_VERTICAL, _rebuild(prev, start, best), best,
                    f"오른쪽으로 갈 길이 없어 세로로 {abs(best[0] - start[0])}칸 이동 "
                    f"(새 행이 들어오면 길이 열린다)")

    # --- 4순위: 장애물 파괴 ----------------------------------------------
    # 오른쪽으로도 세로로도 갈 데가 없는, 완전히 갇힌 경우다.
    middles = horizontal_triples(cells)
    members = horizontal_triple_members(cells)
    candidates: list[tuple[int, int, int, Cell]] = []
    for r in range(N):
        for c in range(N):
            if cells[r][c] != Kind.OBSTACLE:
                continue
            # 도달 가능한 빈칸에 붙어 있는 장애물만 클릭할 수 있다.
            near = [dist[(r + dr, c + dc)]
                    for _, dr, dc in DIRS
                    if 0 <= r + dr < N and 0 <= c + dc < N and (r + dr, c + dc) in dist]
            if not near:
                continue
            # 가로 3연속(XXX) 특수 조건. 가운데를 부술 수 있으면 그게 가장 크게 뚫린다.
            # 가운데가 닿지 않으면 같은 줄의 끝쪽이라도 우선한다.
            # 세로 3연속은 여기에 해당하지 않는다.
            if (r, c) in middles:
                priority = 0
            elif (r, c) in members:
                priority = 1
            else:
                priority = 2
            # 같은 조건이면 가깝고, 그다음 오른쪽에 가까운 장애물을 먼저 부순다.
            candidates.append((priority, min(near), -c, (r, c)))

    if not candidates:
        return Plan(PlanKind.NONE, [], None,
                    "오른쪽으로 갈 길도 없고 부술 수 있는 장애물도 없습니다.")

    candidates.sort()
    priority, d, _, target = candidates[0]
    note = {0: "가로 3연속(XXX) 가운데",
            1: "가로 3연속(XXX) 중 닿을 수 있는 칸",
            2: "가장 가까운 장애물"}[priority]
    # 장애물 바로 옆까지 가는 경로 + 마지막에 장애물 칸을 클릭
    adj = min((cell for cell in
               [(target[0] + dr, target[1] + dc) for _, dr, dc in DIRS]
               if cell in dist), key=lambda cell: dist[cell])
    path = _rebuild(prev, start, adj) + [target]
    return Plan(PlanKind.BREAK_OBSTACLE, path, target,
                f"오른쪽 끝까지 우회로가 전혀 없어 {note} 파괴 (거리 {d})")
