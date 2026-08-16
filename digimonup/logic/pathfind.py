"""경로 계산.

탐사는 **오른쪽으로 무한히 나아가는 미니게임**이다. 정해진 종착점이 없고,
오른쪽으로 한 칸 움직일 때마다 게임판이 왼쪽으로 밀리면서 새 열이 들어온다.
이 성질이 경로 규칙 전체를 결정한다.

**새 지형은 오른쪽 이동으로만 들어온다.** 실측(150초 31이동)으로 확인했다.

    RIGHT  19회   스크롤 18   지형변화 18
    UP      6회   스크롤  0   지형변화  0
    DOWN    3회   스크롤  0   지형변화  0
    LEFT    3회   스크롤  0   지형변화  0

그래서 오른쪽으로 갈 수 있는 칸이 BFS 에 하나도 없으면, 그 방 안에서는 어떤
이동을 해도 길이 열리지 않는다. 장애물을 부수는 것이 유일한 해법이다.

플레이어는 상하좌우로 한 칸씩만 움직인다. 우선순위는 다음과 같다.

  1순위  주황칩까지 가는 장애물 없는 최단 경로 (칩이 걸음수보다 중요하다)
  2순위  전진 경로에서 벗어나 있는 **바로 근처(기본 2칸)** 의 부수기/돌진 아이템
         (걸음수 아이템은 들르지 않는다. 이동에 쓰는 자원을 얻자고 이동하는 셈이라
          본전이거나 손해다. 다만 가는 길에 걸리면 여전히 먹는다.)
  3순위  **도달 가능한 가장 오른쪽 칸**까지 가는 장애물 없는 최단 경로
  4순위  오른쪽으로 한 칸도 못 가면 장애물 클릭 (세로로 헤매지 않는다)

어느 경로든 **같은 길이의 갈래가 여럿이면 아이템을 밟는 쪽**을 고른다(_bfs).
거리는 절대 늘리지 않으므로, 아이템 때문에 돌아가는 일은 2순위에서만 일어나고
그것도 2칸까지다.

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

from dataclasses import dataclass
from enum import Enum

from digimonup.vision.board import N
from digimonup.vision.recognize import Kind, Scene

Cell = tuple[int, int]
DIRS: list[tuple[str, int, int]] = [
    ("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1),
]


class PlanKind(str, Enum):
    GOAL = "목적지"
    ITEM = "근처 아이템"
    RIGHT_EDGE = "오른쪽 전진"
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


# 지나가면서 주우면 좋은 것의 값어치. 길이가 같은 경로가 여럿일 때만 쓴다.
# 칩을 크게 잡은 이유는 칩이 걸음수보다 중요하기 때문이다. 목표로 가는 길에
# 다른 칩이 놓여 있으면 그 길로 간다.
PICKUP_VALUE = {Kind.GOAL: 4, Kind.ITEM: 1}

# 이 종류의 아이템은 **들르지 않는다**(가는 길에 걸리면 여전히 먹는다).
#
# 걸음수 아이템은 이동에 쓰는 자원 그 자체다. 그걸 얻으려고 걸음수를 쓰는 것은
# 앞뒤가 맞지 않는다. 두 칸 걸어가서 걸음수를 얻으면 본전이거나 손해다.
# 반면 부수기·돌진은 걸음수로는 살 수 없는 것이라 조금 돌아갈 값어치가 있다.
DETOUR_SKIP_KINDS = {"steps"}


# 실측으로 확정한 게임 규칙 (150초, 사이클 51회 / 오른쪽 이동 9회)
#
#     플레이어가 있던 열   0열 4회, 1열 47회   (2~4열은 한 번도 없다)
#     0열에서 오른쪽       스크롤 X  (1/1)
#     1열에서 오른쪽       스크롤 O  (8/8)
#
# 즉 **플레이어는 0열과 1열에만 있는다.** 1열에서 오른쪽을 누르면 플레이어가
# 2열로 가는 것이 아니라 배경이 왼쪽으로 밀리고 플레이어는 1열에 그대로 있는다.
# 그때 2열에 있던 것이 플레이어 자리로 들어오므로, 거기 칩이 있으면 먹힌다.
#
# 그래서 경로는 **0~1열만 지나갈 수 있고, 마지막 한 걸음만 2열로 들어간다.**
# 그 한 걸음이 전진(스크롤)이다. 3열, 4열을 목표로 삼는 경로는 실행될 수 없다.

# 플레이어가 서 있을 수 있는 가장 오른쪽 열 (위 실측 참고).
PLAYER_MAX_COL = 1
# 전진하려고 클릭하는 열. 여기를 누르면 배경이 밀린다.
ADVANCE_COL = PLAYER_MAX_COL + 1


# 장애물 하나를 부수는 값을 **걸음수 몇 개어치**로 볼지.
#
# 실험으로 확인한 규칙(25장): 인접한 장애물을 클릭하면 부서진다. 플레이어는
# 들어가지 않고 **걸음수는 들지 않는다.** 대신 부수기 아이템이 1 줄어든다.
#
# 그러니 장애물은 벽이 아니라 '부수기 1개짜리 통행료가 붙은 칸'이다.
# 돌아가면 걸음수를 쓰고, 부수면 부수기를 쓴다. 어느 쪽이 싼지는 **지금 가진
# 양의 비**로 정하는 것이 맞다. 걸음수 926 / 부수기 198 이면 부수기 하나가
# 걸음수 4.7개만큼 귀하다.
DEFAULT_BREAK_COST = 4.0
# 걸어서 닿는 칩을 쫓아갈 때 낼 수 있는 최대 값(걸음수 환산).
# 한도가 없으면 **벽을 여러 겹 뚫고** 칩을 쫓아간다.
#
# 한도는 **칩의 값어치 그 자체**로 둔다. 값보다 비싸게 주고 사지 않는다는,
# 설명이 필요 없는 기준이다. 예전에는 8.0 이라는 별도 숫자였는데, 그러면
# 장애물 하나를 부수고 닿는 칩(1 + 부수기값)이 값어치와 무관하게 잘려 나갔다.
GOAL_MAX_COST = 10.0

# 앞으로 들어올 것들의 값 (걸음수 몇 개어치인가).
#
# 칩이 목적이므로 가장 크게 잡는다. 걸음수 아이템은 **하나에 +5** 라서(실측)
# 네 걸음 이내로 돌아가면 이득이다. 부수기는 나중에 걸음수를 아껴 주므로 그
# 아껴 줄 양(cost_break)만큼의 값이 있다.
VALUE_CHIP = 10.0
VALUE_STEPS_ITEM = 5.0
VALUE_DASH_ITEM = 3.0
# 앞을 몇 번의 전진까지 내다볼지. 2~4열이 보이므로 세 번이 한계다.
LOOKAHEAD = 3
# 부수기 하나를 걸음수 몇 개어치로 볼지의 범위.
#
# 상한이 **칩 값어치보다 낮아야 한다.** 장애물 하나를 부수고 칩을 먹는 값은
# 1 + 상한이므로, 상한이 9 이상이면 칩(10)을 절대 못 넘어 구조적으로 부수지
# 않게 된다. 실측 298초에서 정확히 그 일이 벌어졌다 — 부수기 59개를 쥔 채
# 부수기값이 매 사이클 12.0 으로 잡혔고, **부수기만 하면 닿는 칩 20건**을
# 그대로 지나쳤다.
#
# 상한 6 은 이렇게 정했다. 실제로 걸리는 판은 '벽 하나 너머 두 걸음'까지다.
#     1(걸음) + (1 + 상한)(부수기) + 1(걸음) <= 칩값 10   ->   상한 <= 7
# 여유를 두어 6 으로 한다. 벽 **두 겹** 너머 칩은 2 + 12 = 14 > 10 이라
# 여전히 포기한다.
#
# 하한은 빈 길에서까지 헤프게 부수지 않을 만큼은 둔다. 돌아가는 길이 보통
# 2~4걸음이므로 그보다 비싸야 '그냥 돌아가는' 쪽을 고른다.
BREAK_COST_RANGE = (2.0, 6.0)


def break_cost(steps: int | None, breaks: int | None) -> float:
    """부수기 하나를 걸음수 몇 개어치로 볼지.

    남은 양의 비(걸음수/부수기)를 쓰되 위 범위로 자른다.

    **비 하나만으로는 안 된다.** 그 비는 둘이 같은 속도로 준다고 볼 때만 뜻이
    있는데, 실제로는 그렇지 않다. 실측 298초: 걸음수 142개가 나가는 동안
    부수기는 2개 나갔다(71 대 1). 남은 양은 1298 대 57 이라 비는 24 가 되지만,
    이대로 가면 걸음수가 45분, 부수기가 2.4시간 뒤에 바닥난다. **먼저 바닥나는
    쪽은 걸음수다.** 그런데 비를 그대로 쓰면 부수기를 스물네 배 귀한 것으로
    쳐서, 정작 남아도는 쪽을 아끼느라 모자란 쪽을 쓰게 된다.

    둘 중 하나라도 못 읽으면 기본값을 쓴다.
    """
    if not steps or not breaks:
        return DEFAULT_BREAK_COST
    lo, hi = BREAK_COST_RANGE
    return float(min(hi, max(lo, steps / breaks)))


def _bfs(cells: list[list[Kind]], start: Cell, max_col: int = PLAYER_MAX_COL,
         cost_break: float | None = None
         ) -> tuple[dict[Cell, float], dict[Cell, Cell]]:
    """상하좌우로 움직이는 최소 비용 탐색. (비용, 이전칸) 을 돌려준다.

    **장애물도 지나갈 수 있다.** 부수기 1개를 쓰면 되고 걸음수는 들지 않는다
    (25장 실험). 그래서 장애물 칸에 들어가는 값을 `1 + cost_break` 로 둔다.
    cost_break 가 None 이면 예전처럼 통과 불가로 본다(테스트와 호환).

    예전에는 장애물을 벽으로 두고 '갈 데가 전혀 없을 때만' 부쉈다. 그러면
    돌아가느라 걸음수를 쓰면서 정작 공짜인 파괴를 안 쓴다. 실측 300초에서
    장애물을 부순 것은 단 1회였고, 대신 칩도 없는데 왼쪽으로 도는 이동이 있었다.

    max_col 보다 오른쪽 열로는 걸어가지 않는다. 플레이어는 0~1열에만 있기
    때문이다(위 실측 참고). 예전에는 5x5 전체를 걸어 다닌다고 보고 4열까지
    가는 경로를 세웠는데, 그런 경로는 첫 걸음에서 스크롤이 나 버려지므로
    사실상 아무 의미가 없었고 첫 걸음을 엉뚱한 방향으로 쓰게 만들었다.

    같은 거리의 경로가 여러 개일 때는 **지나가며 주울 게 많은 쪽**을 고른다.
    걸음수/부수기/돌진 아이템은 판 위에 놓여 있고, 밟으면 그냥 얻어진다.

    **거리는 절대 늘리지 않는다.** 아이템을 먹으려고 돌아가거나 장애물을 부수는
    일은 없다. 어디까지나 '가는 길에 공짜로 얻어지는 것'만 챙긴다.

    실측: 이 BFS 가 훑는 칸은 평균 8.1개, 최대 10개다(0~1열 x 5행). 나오는 경로는
    27개 중 23개가 1칸, 최대 2칸이었다. 그래서 '동점 처리가 이 정도 그래프에서
    의미가 있나' 싶지만, 0~1열의 모든 배치를 전수 검사(185,960 조합)해 보니
    **6.09% 에서 결과가 달라졌다.** 예를 들어 (0,1) -> (2,0) 은 동점 처리가
    있을 때 값어치 5 를 줍고 없을 때 0 을 줍는다. 그래서 그대로 둔다.

    구현: BFS 는 거리가 커지는 순서로 꺼내므로, 어떤 칸 v 를 꺼낼 때쯤이면
    v 로 올 수 있는 같은 거리의 이전 칸들이 모두 처리돼 있다. 그래서 v 를
    처음 발견할 때뿐 아니라 '같은 거리로 또 닿았을 때'도 비교해서, 주운 게
    더 많은 쪽으로 이전 칸을 갈아끼우면 된다.
    """
    dist: dict[Cell, float] = {start: 0.0}
    prev: dict[Cell, Cell] = {}
    picked: dict[Cell, int] = {start: 0}       # 여기까지 오면서 주운 것의 합
    # 칸이 열 개뿐이라 우선순위 큐 없이 매번 가장 싼 것을 골라도 넉넉하다.
    todo = {start}
    while todo:
        cur = min(todo, key=lambda c: dist[c])
        todo.discard(cur)
        r, c = cur
        for _, dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not (0 <= nr < N and 0 <= nc <= max_col):
                continue
            kind = cells[nr][nc]
            if kind == Kind.OBSTACLE:
                if cost_break is None:
                    continue                  # 예전 방식: 통과 불가
                step = 1.0 + cost_break       # 부수고 들어간다
            else:
                step = 1.0
            nd = dist[cur] + step
            gain = picked[cur] + PICKUP_VALUE.get(kind, 0)
            old = dist.get(nxt)
            if old is None or nd < old - 1e-9:
                dist[nxt] = nd
                prev[nxt] = cur
                picked[nxt] = gain
                todo.add(nxt)
            elif abs(nd - old) < 1e-9 and gain > picked[nxt]:
                # 값이 같은데 주운 게 더 많은 길을 찾았다. 그쪽으로 갈아탄다.
                prev[nxt] = cur
                picked[nxt] = gain
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


def _cell_value(cells, scene, r: int, c: int, cost_break: float) -> float | None:
    """그 칸이 내 자리로 들어올 때의 값. 장애물이면 부수는 값이 빠진다.

    부수기를 못 쓰면(cost_break None) 그 칸으로는 전진할 수 없으므로 None.
    """
    kind = cells[r][c]
    if kind == Kind.OBSTACLE:
        return None if cost_break is None else -cost_break
    if kind == Kind.GOAL:
        return VALUE_CHIP
    if kind == Kind.ITEM:
        which = scene.item_kinds.get((r, c), "") if scene is not None else ""
        if which == "steps":
            return VALUE_STEPS_ITEM
        if which == "dash":
            return VALUE_DASH_ITEM
        return float(PICKUP_VALUE[Kind.ITEM])
    return 0.0


def _schedule_value(cells, scene, dist, row: int, k: int,
                    cost_break: float, memo: dict) -> float:
    """k 번째 전진부터 끝까지, row 행에서 시작해 얻을 수 있는 최대 값.

    **이 게임은 앞을 정확히 내다볼 수 있다.** 전진할 때마다 판이 한 열씩
    밀리므로, 지금 2열에 있는 것은 이번 전진에, 3열은 다음 전진에, 4열은
    그다음에 내 자리로 들어온다. 즉 '몇 번째 전진에 어느 행에 있을까'만 정하면
    무엇을 얻을지가 확정된다.

    그래서 매번 가장 급한 것만 좇는 대신, 상태 (전진 횟수, 행) 위에서 최적을
    계산한다. 상태가 5행 x 3전진 = 15개뿐이라 비용이 사실상 없다.

    실측(180초 57건): 이렇게 고른 행이 그리디와 11건(19%)에서 달랐다.
    """
    if k > LOOKAHEAD:
        return 0.0
    key = (row, k)
    if key in memo:
        return memo[key]
    best = None
    for r in range(N):
        stand = (r, PLAYER_MAX_COL)
        if stand not in dist:
            continue                       # 그 행의 1열까지 갈 수 없다
        gain = _cell_value(cells, scene, r, PLAYER_MAX_COL + k, cost_break)
        if gain is None:
            continue                       # 그 칸으로는 전진할 수 없다
        move = abs(r - row) * 1.0          # 행을 옮기는 걸음수
        val = -move - 1.0 + gain + _schedule_value(
            cells, scene, dist, r, k + 1, cost_break, memo)
        if best is None or val > best:
            best = val
    memo[key] = 0.0 if best is None else best
    return memo[key]


def _advance_for(cells, dist, prev, start: Cell, scene: Scene | None = None,
                 cost_break: float | None = None):
    """전진(스크롤)을 일으키는 경로를 만든다.

    전진은 **1열에 서서 2열을 클릭**하는 것 하나뿐이다. 그러려면 그 행의 2열이
    장애물이 아니어야 한다. 그래서 '2열이 뚫린 행' 중에서 하나를 골라 세로로
    이동한 뒤 오른쪽을 누른다.

    어느 행을 고를지는 **그 행에서 앞으로 들어올 것의 값어치에서 세로 이동
    걸음수를 뺀 값**으로 정한다. 칩(4)이면 두 칸 올라갈 값어치가 있고,
    아이템(1)이면 한 칸도 아깝다.

    반환: (경로, 고른 행, 세로 걸음수, 값어치) — 전진할 행이 없으면 None.
    """
    memo: dict = {}
    best = None
    for r in range(N):
        stand = (r, PLAYER_MAX_COL)
        if stand not in dist:
            continue                      # 그 행의 1열까지 갈 수 없다
        gain = _cell_value(cells, scene, r, ADVANCE_COL, cost_break)
        if gain is None:
            continue                      # 그 행 2열이 장애물인데 부술 수 없다
        cost = dist[stand]
        # **앞으로 두 번 더 전진할 것까지 함께 본다.**
        # 지금 한 칸 더 움직여 두면 그다음 전진에서 칩 두 개를 받는 자리가
        # 있을 수 있다. 매번 가장 급한 것만 좇으면 그런 자리를 놓친다.
        ahead = _schedule_value(cells, scene, dist, r, 2, cost_break, memo)
        score = gain + ahead - cost
        # 동점이면 **덜 움직이는 쪽으로 간다.**
        #
        # 한때 반대로 바꿨다가 되돌렸다. "칩이 다가오는데 4행에서 세 사이클
        # 내리 제자리 전진을 골라 놓쳤다"고 봤는데, 그건 분석기 쪽 버그였다
        # (장애물 글자를 'X' 로 찾는 바람에 장애물을 전부 빈칸으로 봤다).
        # 제대로 세어 보니 그 판은 애초에 **그 행으로 갈 수가 없었고**, 다른
        # 판은 **결국 걸어가서 먹었다**. 미루는 편이 맞다 — 그사이 새 열이
        # 들어와 더 나은 자리가 생기고, 미리 옮기면 그 걸음수를 되돌려야
        # 할 수도 있다.
        key = (score, gain, -cost, -abs(r - start[0]))
        if best is None or key > best[0]:
            best = (key, stand, cost, gain, score)
    if best is None:
        return None
    _, stand, cost, value, score = best
    path = _rebuild(prev, start, stand) + [(stand[0], ADVANCE_COL)]
    return path, stand[0], cost, value


def plan_route(scene: Scene, item_max_detour: int = 2,
               cost_break: float | None = DEFAULT_BREAK_COST) -> Plan:
    """장면을 보고 다음에 무엇을 할지 정한다.

    item_max_detour
        칩이 없을 때, 이 칸수 안에 있는 아이템은 들러서 먹는다.
        0 이면 아이템을 목표로 삼지 않는다(가는 길에 걸리면 여전히 먹는다).
    """
    if scene.player is None:
        return Plan(PlanKind.NONE, [], None, "플레이어를 찾지 못했습니다.")

    start: Cell = (scene.player.row, scene.player.col)
    cells = scene.cells
    # cost_break=None 은 '부수기를 아예 쓸 수 없다'는 뜻이다(장애물 = 벽).
    dist, prev = _bfs(cells, start, cost_break=cost_break)

    # --- 1순위: 주황칩(필수 아이템) ----------------------------------------
    # 판에 칩이 여러 개일 수 있다. **가장 가까운 것부터** 먹는다. 매번 '가장
    # 진한 칩'을 고르면 프레임마다 목표가 바뀌어 제자리를 오간다.
    goal_cells = [(d.row, d.col) for d in scene.goals]
    # scene.goal 로 되돌리는 갈래는 두지 않는다. 진실이 두 곳에 있으면
    # 한쪽만 걸러졌을 때 조용히 어긋난다(실제로 그랬다).
    # 1-a) **걸어서 닿는 칩** (0~1열). 다음 전진에 화면 밖으로 밀려나므로 급하다.
    walkable_goals = [g for g in goal_cells
                      if g in dist and g != start and dist[g] <= GOAL_MAX_COST]
    if walkable_goals:
        # 가깝고, 같으면 **더 왼쪽** 칩을 먼저 먹는다.
        # 오른쪽으로 한 칸 전진하면 열 번호가 하나씩 줄어들어 0열 칩이 화면
        # 밖으로 떨어진다. 왼쪽 것을 먼저 먹어야 둘 다 먹는다.
        goal = min(walkable_goals, key=lambda c: (dist[c], c[1]))
        extra = (f" (걸어서 닿는 칩 {len(walkable_goals)}개 중 가깝고 왼쪽 것부터)"
                 if len(walkable_goals) > 1 else "")
        return Plan(PlanKind.GOAL, _rebuild(prev, start, goal), goal,
                    f"주황칩까지 {dist[goal]}칸{extra}")

    # 1-b) **걸어서 못 닿는 칩** (2열 이상). 이건 쫓아가는 것이 아니라
    #      그 칩이 있는 **행에 서서 전진하면 제 발로 걸어 들어온다.**
    #      전진할 때마다 열이 하나씩 줄어 결국 플레이어 자리로 오기 때문이다.
    #      그래서 비용은 '그 행까지 세로로 가는 걸음수'뿐이다.
    #      어느 행에서 전진할지는 아래 '전진' 규칙이 값어치까지 따져서 정한다.

    # --- 2순위 / 3순위 -----------------------------------------------------
    # 먼저 '오른쪽 전진' 경로를 계산해 둔다. 근처 아이템이 **이미 그 경로 위에**
    # 있으면 따로 들를 이유가 없다. 지나가면서 먹고 더 나아가는 편이 낫다.
    #
    # 탐사는 오른쪽으로 무한히 나아가는 미니게임이고, 한 칸 이동할 때마다
    # 게임판이 밀리면서 **새 열이 하나 들어온다**. 그래서 지금 보이는 5x5 안에서
    # 맨 오른쪽 열(col 4)에 못 닿는다고 막힌 것이 아니다.
    # 따라서 목표는 '오른쪽 끝 열'이 아니라 **도달 가능한 가장 오른쪽 칸**이다.
    # 목표는 **걸음당 전진이 가장 좋은 칸**이다. '가장 오른쪽 칸'이 아니다.
    #
    # 실측 회귀: 예전에는 거리를 무시하고 제일 오른쪽 칸을 목표로 삼았다.
    # 그러면 6걸음을 들여 3열을 얻는 경로(0.50 열/걸음)를 고르는데, 바로 옆에
    # 2걸음에 2열을 얻는 길(1.00)이 있었다. 실제 주행에서 오른쪽 전진 계획
    # 17개 중 5개가 그랬고, 평균 0.112 열/걸음을 손해 보고 있었다.
    # 그 손해가 'UP UP UP' 처럼 세로로 몰려 움직이는 모습으로 나타났다
    # (전체 이동의 34%가 세로였다).
    #
    # 게다가 한 칸 움직일 때마다 판이 스크롤해서 남은 경로를 버리고 다시
    # 계획하므로, 사실상 **첫 한 걸음만 의미가 있다.** 먼 목표를 위해 첫 걸음을
    # 위아래로 쓰는 것은 거의 언제나 손해다. 지금 갈 수 있는 만큼 전진하고
    # 새 지형을 본 뒤에 다시 정하는 편이 낫다.
    right_plan: Plan | None = None
    got = _advance_for(cells, dist, prev, start, scene, cost_break)
    if got is not None:
        path, row, cost, value = got
        reason = (f"{row}행에서 전진 (세로 {cost}칸 이동 후 오른쪽)"
                  if cost else "제자리에서 바로 전진")
        if value:
            reason += f" — 그 행 앞쪽의 값어치 {value:.2f}"
        # 칩이 들어오는 행으로 가는 것이면 '목적지' 계획으로 부른다.
        # 값어치 숫자로 가르지 않는다. value 는 **이번 전진에 들어올 칸 하나**의
        # 값이라, 세 번 뒤에 들어올 칩을 향해 가는 중이면 0 으로 나온다.
        # 칩을 향해 가는 것은 거리와 상관없이 칩 계획이다.
        row_has_chip = any(cells[row][c] == Kind.GOAL for c in range(ADVANCE_COL, N))
        kind = PlanKind.GOAL if row_has_chip else PlanKind.RIGHT_EDGE
        right_plan = Plan(kind, path, path[-1], reason)

    # --- 2순위: 전진 경로에서 벗어나 있는 바로 근처의 아이템 -----------------
    # 걸음수/부수기/돌진 아이템은 밟으면 그냥 얻어진다. 그런데 '가는 길에 있으면
    # 줍는다'(경로 동점 처리)만으로는 **한 칸 옆에 있어도 그냥 지나친다.**
    # 실측: 아이템이 2칸, 4칸 거리에 있는데 전진 경로에 안 걸려서 둘 다 못 먹었다.
    #
    # 그래서 **아주 가까운 것만** 목표로 삼는다. 멀리 있는 것을 쫓아가면 전진이
    # 느려지므로 item_max_detour 칸까지만 본다. 장애물은 여전히 부수지 않는다
    # (BFS 가 장애물을 통과하지 않으므로 자동으로 지켜진다).
    if item_max_detour > 0:
        on_the_way = set(right_plan.path) if right_plan else set()
        kinds = scene.item_kinds
        near = [(r, c) for r in range(N) for c in range(N)
                if cells[r][c] == Kind.ITEM and (r, c) in dist
                and (r, c) != start and dist[(r, c)] <= item_max_detour
                and (r, c) not in on_the_way
                and kinds.get((r, c), "") not in DETOUR_SKIP_KINDS]
        if near:
            # 가깝고, 같으면 더 오른쪽(전진 방향)에 있는 것부터
            target = min(near, key=lambda c: (dist[c], -c[1]))
            kind = kinds.get(target, "") or "아이템"
            return Plan(PlanKind.ITEM, _rebuild(prev, start, target), target,
                        f"전진 경로에서 벗어난 {kind} 까지 {dist[target]}칸 "
                        f"(최대 {item_max_detour}칸까지만 들른다)")

    # --- 3순위: 오른쪽으로 갈 수 있는 데까지 --------------------------------
    if right_plan is not None:
        return right_plan

    # --- 4순위: 장애물 파괴 ----------------------------------------------
    # 여기까지 왔다는 것은 BFS 로 갈 수 있는 칸 중에 지금보다 오른쪽인 칸이
    # **하나도 없다**는 뜻이다. 장애물 없는 우회로는 3순위에서 이미 다 찾아봤다.
    #
    # 예전에는 이럴 때 세로로 최대한 움직였다. "세로 이동에서도 판이 스크롤하며
    # 새 행이 들어온다"고 봤기 때문인데, **사실이 아니었다.** 실측 150초에서
    # 세로·왼쪽 이동 12회 중 지형이 바뀐 것은 0회였고(오른쪽은 19회 중 18회),
    # 이 계획으로 실행된 이동 2회도 전부 지형 변화가 없었다.
    # 갇힌 방 안을 걸어 다녀 봐야 걸음수만 쓴다. 장애물을 부수는 수밖에 없다.
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
