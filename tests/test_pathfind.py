"""경로 계산 테스트.

탐사는 오른쪽으로 무한히 나아가는 미니게임이다. 한 칸 움직일 때마다 게임판이
밀리면서 새 열/새 행이 들어오므로, 지금 보이는 5x5 안에서 맨 오른쪽 열에
못 닿아도 막힌 것이 아니다.

가장 중요한 규칙: **우회하거나 조금이라도 전진할 수 있으면 장애물을 부수지 않는다.**
장애물 클릭은 상하좌우 어디로도 갈 수 없을 때만 허용한다.
"""

from __future__ import annotations

import pathfind
from board import Grid, N
from pathfind import PlanKind, plan_route
from recognize import Detection, Kind, Scene

SYM = {".": Kind.EMPTY, "X": Kind.OBSTACLE, "P": Kind.PLAYER,
       "G": Kind.GOAL, "i": Kind.ITEM}


def make_scene(layout: list[str]) -> Scene:
    cells = [[SYM[ch] for ch in row.replace(" ", "")] for row in layout]
    grid = Grid(xs=[0, 100, 200, 300, 400, 500],
                ys=[0, 90, 180, 270, 360, 450], confidence=0.9, detail={})
    player = goal = None
    for r in range(N):
        for c in range(N):
            if cells[r][c] == Kind.PLAYER:
                player = Detection(Kind.PLAYER, r, c, 0.9)
            elif cells[r][c] == Kind.GOAL:
                goal = Detection(Kind.GOAL, r, c, 0.9)
    return Scene(grid=grid, cells=cells, player=player, goal=goal)


# ----------------------------------------------------------- 1순위: 목적지
def test_목적지까지_장애물_없는_최단_경로():
    plan = plan_route(make_scene([
        "P....",
        ".....",
        "..G..",
        ".....",
        ".....",
    ]))
    assert plan.kind == PlanKind.GOAL
    assert plan.path[0] == (0, 0) and plan.path[-1] == (2, 2)
    assert len(plan.path) - 1 == 4          # 맨해튼 거리 = 최단
    assert all(cell != (9, 9) for cell in plan.path)


def test_목적지_경로가_장애물을_통과하지_않는다():
    plan = plan_route(make_scene([
        "P.X..",
        ".XX..",
        "..G..",
        ".....",
        ".....",
    ]))
    assert plan.kind == PlanKind.GOAL
    blocked = {(0, 2), (1, 1), (1, 2)}
    assert not (set(plan.path) & blocked)


# ------------------------------------ 2순위: 오른쪽으로 갈 수 있는 데까지
def test_목적지가_없으면_오른쪽으로_전진한다():
    plan = plan_route(make_scene([
        ".....",
        "..P..",
        ".....",
        ".....",
        ".....",
    ]))
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path[-1][1] == N - 1
    assert plan.moves == ["RIGHT", "RIGHT"]


def test_끝_열에_못_닿아도_갈_수_있는_만큼_오른쪽으로_간다():
    """탐사는 무한 우측 진행이라, 한 칸만 가도 새 열이 들어와 길이 이어진다.

    '오른쪽 끝 열(col 4)에 닿을 수 있는가'만 보면, 사실 전진할 수 있는데도
    갇혔다고 판단해 장애물을 두드리게 된다. 실제로 그런 판에서 멈췄었다.
    """
    scene = make_scene([
        ".XX..",
        "..X.X",
        "P.X..",
        "XX...",
        ".X.X.",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE, "전진할 수 있는데 다른 판단을 했습니다"
    assert plan.target == (2, 1)
    assert plan.moves == ["RIGHT"]
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path)


def test_우회할_수_있으면_장애물을_부수지_않는다():
    """오른쪽이 막혀 있어도 위/아래로 돌아갈 수 있으면 파괴하지 않는다."""
    scene = make_scene([
        ".....",
        "..PXX",
        ".....",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path[-1][1] == N - 1
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path)


def test_한_줄만_뚫려_있어도_우회로를_찾는다():
    scene = make_scene([
        "XXXXX",
        "XXXXX",
        "..P..",
        "XXXXX",
        "XXXXX",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path[-1] == (2, 4)


def test_먼_길로_돌아가야_해도_부수지_않는다():
    scene = make_scene([
        "....X",
        "XXX.X",
        "..P.X",
        "XXX.X",
        "....X",
    ])
    plan = plan_route(scene)
    # 끝 열은 전부 막혔지만 col 3 까지는 갈 수 있다 -> 그만큼 전진한다
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.target[1] == 3


# ------------------------------- 3순위: 세로로 움직여 새 지형을 불러온다
def test_오른쪽으로_한_칸도_못_가면_세로로_움직인다():
    """세로 이동에서도 판이 스크롤하며 새 행이 들어온다. 파괴보다 먼저다."""
    scene = make_scene([
        "..X..",
        "..X..",
        ".PX..",
        "..X..",
        "..X..",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.SCROLL_VERTICAL
    assert plan.target[0] != 2, "세로로 움직여야 합니다"
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path)


def test_세로_이동은_가장_먼_행까지_간다():
    scene = make_scene([
        "..X..",
        "..X..",
        ".PX..",
        "..X..",
        "..X..",
    ])
    plan = plan_route(scene)
    assert abs(plan.target[0] - 2) == 2


# ----------------------------- 4순위: 완전히 갇혔을 때만 장애물 클릭
def test_사방이_막혔을_때만_장애물을_클릭한다():
    scene = make_scene([
        ".X...",
        "XPX..",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.BREAK_OBSTACLE
    assert scene.cells[plan.target[0]][plan.target[1]] == Kind.OBSTACLE
    assert plan.path[-1] == plan.target
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path[:-1])


def test_가로_3연속_XXX_는_가운데를_부순다():
    """3개 연속 장애물 특수 조건은 가로 배열에만 적용한다."""
    scene = make_scene([
        "XXX..",
        "XPX..",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.BREAK_OBSTACLE
    assert plan.target == (0, 1), "가로 XXX 의 가운데 칸을 골라야 한다"
    assert "가운데" in plan.reason


def test_XXX_가운데에_닿을_수_없으면_같은_줄의_끝을_부순다():
    scene = make_scene([
        ".X...",
        "XPXXX",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.BREAK_OBSTACLE
    assert plan.target == (1, 2)
    assert "가로 3연속" in plan.reason


def test_세로_3연속은_특수_조건이_아니다():
    scene = make_scene([
        ".XX..",
        "XPX..",
        ".XX..",
        ".....",
        ".....",
    ])
    assert pathfind.horizontal_triples(scene.cells) == set()
    plan = plan_route(scene)
    assert plan.kind == PlanKind.BREAK_OBSTACLE


def test_가로_4연속은_XXX_특수_조건이_아니다():
    cells = make_scene([
        ".....",
        ".....",
        ".XXXX",
        ".....",
        ".....",
    ]).cells
    assert pathfind.horizontal_triples(cells) == set()


def test_XXX_구성원_집합():
    cells = make_scene([
        "XXX..",
        ".....",
        "..XXX",
        ".....",
        ".....",
    ]).cells
    assert pathfind.horizontal_triples(cells) == {(0, 1), (2, 3)}
    assert pathfind.horizontal_triple_members(cells) == {
        (0, 0), (0, 1), (0, 2), (2, 2), (2, 3), (2, 4)}


# ------------------------------------------------------------------ 기타
def test_플레이어를_못_찾으면_아무것도_하지_않는다():
    scene = make_scene([".....", ".....", ".....", ".....", "....."])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.NONE
    assert plan.path == []


def test_아이템과_목적지는_지나갈_수_있다():
    scene = make_scene([
        "P.i..",
        ".....",
        ".....",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.path[-1][1] == N - 1
    assert (0, 2) in plan.path
    assert pathfind.passable(Kind.ITEM)
    assert pathfind.passable(Kind.GOAL)
    assert not pathfind.passable(Kind.OBSTACLE)


def test_경로를_방향_목록으로_바꾼다():
    plan = plan_route(make_scene([
        "P....",
        ".....",
        "..G..",
        ".....",
        ".....",
    ]))
    assert set(plan.moves) <= {"UP", "DOWN", "LEFT", "RIGHT"}
    assert plan.moves.count("DOWN") == 2 and plan.moves.count("RIGHT") == 2


def test_목적지가_막혀_있으면_오른쪽_전진으로_내려간다():
    scene = make_scene([
        ".....",
        ".P.X.",
        "...XG",
        "...X.",
        ".....",
    ])
    plan = plan_route(scene)
    # 목적지 (2,4) 는 X 벽 뒤에 있지만 아래로 돌아가면 도달 가능하다
    assert plan.kind == PlanKind.GOAL
    assert plan.path[-1] == (2, 4)


# ------------------------------------- 주황칩(필수 아이템)이 여러 개인 경우
def _scene_with_chips(layout):
    from recognize import Detection
    scene = make_scene(layout)
    scene.goals = [Detection(Kind.GOAL, r, c, 0.6)
                   for r in range(N) for c in range(N)
                   if scene.cells[r][c] == Kind.GOAL]
    return scene


def test_주황칩이_여러_개면_가장_가까운_것부터_먹는다():
    """실측: 한 판에 칩이 2개 놓인다.

    매 프레임 '가장 진한 칩'을 고르면 프레임마다 목표가 바뀌어 제자리를 오간다.
    """
    scene = _scene_with_chips([
        ".....",
        ".P.G.",
        ".....",
        "....G",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.GOAL
    assert plan.target == (1, 3), "더 가까운 칩을 골라야 합니다"


def test_칩이_같은_거리면_더_오른쪽_칩을_고른다():
    scene = _scene_with_chips([
        ".G...",
        ".P...",
        ".....",
        ".....",
        ".....",
    ])
    scene.cells[2][1] = Kind.GOAL
    from recognize import Detection
    scene.goals.append(Detection(Kind.GOAL, 2, 1, 0.6))
    plan = plan_route(scene)
    assert plan.kind == PlanKind.GOAL
    assert plan.target in {(0, 1), (2, 1)}


def test_막힌_칩은_건너뛰고_오른쪽으로_전진한다():
    scene = _scene_with_chips([
        "XXXXX",
        "XG.XX",
        "XXXXX",
        "..P..",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE


def test_칩이_없으면_오른쪽_전진():
    scene = _scene_with_chips([
        ".....",
        "..P..",
        ".....",
        ".....",
        ".....",
    ])
    assert scene.goals == []
    assert plan_route(scene).kind == PlanKind.RIGHT_EDGE


# ------------------------- 가는 길에 아이템 줍기 (돌아가지는 않는다)
def _plan_for(rows):
    """글자판으로 Scene 을 만들어 경로를 계산한다.  P 플레이어 / G 칩 / i 아이템 / X 장애물"""
    sym = {".": Kind.EMPTY, "P": Kind.PLAYER, "G": Kind.GOAL,
           "X": Kind.OBSTACLE, "i": Kind.ITEM}
    cells = [[sym[ch] for ch in row] for row in rows]
    player = goals = None
    goals = []
    for r in range(5):
        for c in range(5):
            if rows[r][c] == "P":
                player = Detection(Kind.PLAYER, r, c, 1.0)
            elif rows[r][c] == "G":
                goals.append(Detection(Kind.GOAL, r, c, 1.0))
    scene = Scene(grid=None, cells=cells, player=player,
                            goal=goals[0] if goals else None, goals=goals)
    return plan_route(scene)


def test_같은_거리면_아이템을_밟는_길로_간다():
    """칩까지 두 갈래 길이 똑같이 3칸이면, 아이템이 놓인 쪽으로 간다."""
    plan = _plan_for([
        ".i..G",      # 위로 돌면 아이템을 밟는다
        "P....",
        ".....",
        ".....",
        ".....",
    ])
    # (0,4) 칩까지 최단 5칸. 위 경로는 (0,1) 아이템을 지난다.
    assert (0, 1) in plan.path, f"아이템을 지나는 길을 고르지 않았습니다: {plan.path}"
    assert len(plan.path) - 1 == 5, "최단 거리가 아닙니다"


def test_아이템을_먹으려고_돌아가지_않는다():
    """아이템이 경로 밖에 있으면 무시한다. 걸음수가 늘어나면 안 된다."""
    plan = _plan_for([
        ".....",
        "P...G",      # 칩까지 4칸 직진
        ".....",
        "..i..",      # 아이템은 두 칸 아래. 먹으러 가면 4칸이 더 든다
        ".....",
    ])
    assert len(plan.path) - 1 == 4, f"돌아갔습니다: {plan.path}"
    assert (3, 2) not in plan.path


def test_아이템_때문에_장애물을_부수지_않는다():
    """아이템이 장애물 뒤에 있어도 부수지 않는다."""
    plan = _plan_for([
        ".....",
        "P.X.G",      # 아이템은 장애물 뒤
        ".....",
        ".....",
        ".....",
    ])
    assert plan.kind != PlanKind.BREAK_OBSTACLE
    for cell in plan.path:
        assert cell != (1, 2), "장애물 칸을 지나려 했습니다"


def test_가는_길의_다른_칩도_주워간다():
    """목표 칩으로 가는 길에 다른 칩이 있으면 그 길로 간다."""
    plan = _plan_for([
        ".G..G",
        "P....",
        ".....",
        ".....",
        ".....",
    ])
    assert (0, 1) in plan.path, f"길에 있는 칩을 지나치지 않아야 합니다: {plan.path}"


def test_아이템이_없으면_예전과_같다():
    plan = _plan_for([
        ".....",
        "P...G",
        ".....",
        ".....",
        ".....",
    ])
    assert plan.kind == PlanKind.GOAL
    assert len(plan.path) - 1 == 4


# ------------------- 아이템 들르기 (걸음수는 제외) — 실측 회귀
def _plan_with_kinds(rows, kinds, **kw):
    sym = {".": Kind.EMPTY, "P": Kind.PLAYER, "G": Kind.GOAL,
           "X": Kind.OBSTACLE, "i": Kind.ITEM}
    cells = [[sym[ch] for ch in row] for row in rows]
    player = None
    for r in range(5):
        for c in range(5):
            if rows[r][c] == "P":
                player = Detection(Kind.PLAYER, r, c, 1.0)
    scene = Scene(grid=None, cells=cells, player=player, goals=[],
                  item_kinds=kinds)
    return plan_route(scene, **kw)


def test_전진_경로에서_벗어난_아이템은_들른다():
    """실측 회귀: 아이템이 2칸 거리에 있는데 전진 경로에 안 걸려서 그냥 지나쳤다."""
    plan = _plan_with_kinds([
        ".....",
        ".....",
        "..i..",       # 돌진 아이템. 전진 경로(3행)에서 벗어나 있다
        ".P...",
        ".....",
    ], {(2, 2): "dash"})
    assert plan.kind == PlanKind.ITEM
    assert plan.path[-1] == (2, 2)


def test_걸음수_아이템은_들르지_않는다():
    """걸음수는 이동에 쓰는 자원이다. 그걸 얻으려고 이동하면 본전이거나 손해다."""
    plan = _plan_with_kinds([
        ".....",
        ".....",
        "..i..",
        ".P...",
        ".....",
    ], {(2, 2): "steps"})
    assert plan.kind == PlanKind.RIGHT_EDGE, f"걸음수를 먹으러 갔습니다: {plan.describe()}"


def test_걸음수도_가는_길에_있으면_먹는다():
    """들르지 않을 뿐, 공짜로 얻어지는 것까지 피하지는 않는다."""
    plan = _plan_with_kinds([
        ".....",
        ".Pi..",       # 오른쪽 전진 경로 위에 걸음수가 놓여 있다
        ".....",
        ".....",
        ".....",
    ], {(1, 2): "steps"})
    assert (1, 2) in plan.path


def test_이미_전진_경로_위의_아이템이면_거기서_멈추지_않는다():
    """지나가며 먹고 더 나아가는 편이 낫다."""
    plan = _plan_with_kinds([
        ".....",
        ".Pi..",
        ".....",
        ".....",
        ".....",
    ], {(1, 2): "dash"})
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path[-1][1] == 4, f"아이템 칸에서 멈췄습니다: {plan.path}"
    assert (1, 2) in plan.path


def test_아이템_들르기를_끌_수_있다():
    plan = _plan_with_kinds([
        ".....", ".....", "..i..", ".P...", ".....",
    ], {(2, 2): "dash"}, item_max_detour=0)
    assert plan.kind == PlanKind.RIGHT_EDGE


def test_멀리_있는_아이템은_들르지_않는다():
    plan = _plan_with_kinds([
        "....i",       # 4칸 거리
        ".....",
        ".....",
        ".P...",
        ".....",
    ], {(0, 4): "dash"}, item_max_detour=2)
    assert plan.kind == PlanKind.RIGHT_EDGE


# --------------------- 칩이 여러 개일 때 어느 것부터 (실측 회귀)
def _plan_chips(rows):
    sym = {".": Kind.EMPTY, "P": Kind.PLAYER, "G": Kind.GOAL,
           "X": Kind.OBSTACLE, "i": Kind.ITEM}
    cells = [[sym[ch] for ch in row] for row in rows]
    player, goals = None, []
    for r in range(5):
        for c in range(5):
            if rows[r][c] == "P":
                player = Detection(Kind.PLAYER, r, c, 1.0)
            elif rows[r][c] == "G":
                goals.append(Detection(Kind.GOAL, r, c, 1.0))
    scene = Scene(grid=None, cells=cells, player=player,
                  goal=goals[0] if goals else None, goals=goals)
    return plan_route(scene)


def test_같은_거리의_칩이면_왼쪽_것부터_먹는다():
    """실측 회귀: 칩이 2개일 때 하나를 버리는 문제.

    스크롤 규칙이 after[r][c] = before[r+dr][c+dc] 라서, 오른쪽으로 한 칸 가면
    열 번호가 하나씩 줄어든다. 즉 **왼쪽 칩이 화면 밖으로 떨어진다.**
    오른쪽 칩을 먼저 먹으러 가는 동안 왼쪽 칩은 사라지므로 하나를 버리게 된다.

    왼쪽 것부터 먹으면 그동안 오른쪽 칩은 판이 밀리며 더 오래 남아 둘 다 먹는다.
    """
    plan = _plan_chips([
        ".....",
        ".....",
        "G.P.G",       # 양쪽으로 2칸씩. 왼쪽(2,0)이 먼저 사라진다
        ".....",
        ".....",
    ])
    assert plan.kind == PlanKind.GOAL
    assert plan.target == (2, 0), f"오른쪽 칩을 먼저 골랐습니다: {plan.target}"


def test_더_가까운_칩이_있으면_거리가_우선이다():
    """왼쪽 우선은 어디까지나 '같은 거리일 때' 규칙이다."""
    plan = _plan_chips([
        ".....",
        ".....",
        "G..PG",       # 왼쪽은 3칸, 오른쪽은 1칸
        ".....",
        ".....",
    ])
    assert plan.target == (2, 4), f"가까운 칩을 두고 멀리 갔습니다: {plan.target}"
