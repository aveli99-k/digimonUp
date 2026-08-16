"""경로 계산 테스트.

탐사는 오른쪽으로 무한히 나아가는 미니게임이다. 한 칸 움직일 때마다 게임판이
밀리면서 새 열/새 행이 들어오므로, 지금 보이는 5x5 안에서 맨 오른쪽 열에
못 닿아도 막힌 것이 아니다.

가장 중요한 규칙: **우회하거나 조금이라도 전진할 수 있으면 장애물을 부수지 않는다.**
장애물 클릭은 상하좌우 어디로도 갈 수 없을 때만 허용한다.
"""

from __future__ import annotations

from digimonup.logic import pathfind
from digimonup.vision.board import Grid, N
from digimonup.logic.pathfind import PlanKind, plan_route
from digimonup.vision.recognize import Detection, Kind, Scene

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


# ------------------------------------ 2순위: 전진 (1열에 서서 2열을 클릭)
def test_전진은_2열을_클릭하는_한_걸음이다():
    """실측 모델: 플레이어는 0~1열에만 있고, 1열에서 오른쪽을 누르면 배경이 밀린다.

    그래서 전진 경로는 '3열, 4열까지 걸어간다'가 아니라 **2열 클릭 한 번**이다.
    예전에는 4열까지 가는 경로를 세웠는데, 첫 걸음에서 스크롤이 나 나머지가
    통째로 버려졌고 그 바람에 첫 걸음을 엉뚱한 방향으로 쓰곤 했다.
    """
    plan = plan_route(make_scene([
        ".....",
        ".P...",
        ".....",
        ".....",
        ".....",
    ]))
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.moves == ["RIGHT"]
    assert plan.path == [(1, 1), (1, 2)]


def test_0열에_있으면_먼저_1열로_간다():
    """0열에서 오른쪽을 눌러도 스크롤은 안 난다(실측 1/1). 1열로 붙는 걸음이다."""
    plan = plan_route(make_scene([
        ".....",
        "P....",
        ".....",
        ".....",
        ".....",
    ]))
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path[:2] == [(1, 0), (1, 1)]


def test_2열이_막히면_뚫린_행으로_옮겨서_전진한다():
    """세로 이동은 '새 지형을 부르려고'가 아니라 **전진할 수 있는 행으로 가려고**
    하는 것이다. 이건 낭비가 아니라 꼭 필요한 걸음이다."""
    scene = make_scene([
        ".....",
        ".PX..",
        ".....",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.target[1] == 2, "전진은 2열 클릭이어야 합니다"
    assert plan.target[0] != 1, "2열이 막힌 행에서 전진할 수는 없습니다"
    assert plan.moves[-1] == "RIGHT"
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path)


def test_한_줄만_뚫려_있어도_전진한다():
    scene = make_scene([
        "XXXXX",
        "XXXXX",
        ".P...",
        "XXXXX",
        "XXXXX",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path == [(2, 1), (2, 2)]


def test_전진할_수_있는_가장_가까운_행으로_간다():
    scene = make_scene([
        "..X..",
        "..X..",
        ".PX..",
        "..X..",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE
    assert plan.path == [(2, 1), (3, 1), (4, 1), (4, 2)], \
        f"2열이 뚫린 4행으로 내려가 전진해야 합니다: {plan.path}"


# ------------------------------- 4순위: 오른쪽이 막히면 바로 장애물을 부순다
def test_오른쪽으로_한_칸도_못_가면_세로로_헤매지_않고_부순다():
    """실측: 새 지형은 오른쪽 이동으로만 들어온다.

    150초 31이동에서 RIGHT 는 19회 중 18회 지형이 바뀌었고, UP/DOWN/LEFT 는
    12회 전부 0회였다. 그러니 오른쪽으로 갈 칸이 없는 방 안에서 위아래로
    걸어 다니는 것은 걸음수만 쓰는 헛일이다.
    """
    scene = make_scene([
        "..X..",
        "..X..",
        ".PX..",
        "..X..",
        "..X..",
    ])
    plan = plan_route(scene)
    # 장애물은 벽이 아니라 '부수기 1개짜리 통행료가 붙은 칸'이다(25장 실험).
    # 세로로 헤매지 말고 뚫고 나가는 경로가 나와야 한다.
    assert plan.path[-1] == (2, 2), f"뚫고 전진해야 합니다: {plan.path}"
    assert scene.cells[2][2] == Kind.OBSTACLE


def test_위아래로_돌아가면_오른쪽에_닿을_때는_부수지_않는다():
    """장애물 없는 우회로가 있으면 언제나 그쪽이 먼저다.

    아래 판은 (4,2) 가 뚫려 있어 내려갔다 돌아가면 오른쪽에 닿는다.
    '오른쪽으로 한 칸도 못 간다'와 '바로 오른쪽이 막혔다'를 헷갈리면
    멀쩡한 길을 두고 장애물을 두드리게 된다.
    """
    scene = make_scene([
        "..X..",
        "..X..",
        ".PX..",
        "..X..",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE, \
        f"우회로가 있는데 {plan.kind.value} 를 골랐습니다"
    assert plan.target[1] > 1, "오른쪽으로 나아가야 합니다"
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path)


# ----------------------------- 4순위: 완전히 갇혔을 때만 장애물 클릭
def test_사방이_막혀도_뚫고_나간다():
    """예전에는 '사방이 막히면 그때만 부순다'였다. 규칙을 실험으로 확인하니
    장애물은 벽이 아니라 부수기 1개면 지나가는 칸이고 걸음수는 들지 않는다."""
    scene = make_scene([
        ".X...",
        "XPX..",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert len(plan.path) >= 2, "가만히 있으면 안 된다"
    nxt = plan.path[1]
    assert scene.cells[nxt[0]][nxt[1]] == Kind.OBSTACLE,         f"둘러싸였으면 뚫어야 합니다: {plan.path}"
    assert scene.cells[plan.target[0]][plan.target[1]] == Kind.OBSTACLE
    assert plan.path[-1] == plan.target
    assert all(scene.cells[r][c] != Kind.OBSTACLE for r, c in plan.path[:-1])


def test_가로_3연속_XXX_는_가운데를_부순다():
    # 부수기를 아예 못 쓸 때(cost_break=None)의 최후 규칙이다.
    """3개 연속 장애물 특수 조건은 가로 배열에만 적용한다."""
    scene = make_scene([
        "XXX..",
        "XPX..",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene, cost_break=None)
    assert plan.kind == PlanKind.BREAK_OBSTACLE
    assert plan.target == (0, 1), "가로 XXX 의 가운데 칸을 골라야 한다"
    assert "가운데" in plan.reason


def test_XXX_가운데에_닿을_수_없으면_같은_줄의_끝을_부순다():
    # 부수기를 아예 못 쓸 때(cost_break=None)의 최후 규칙이다.
    scene = make_scene([
        ".X...",
        "XPXXX",
        ".X...",
        ".....",
        ".....",
    ])
    plan = plan_route(scene, cost_break=None)
    assert plan.kind == PlanKind.BREAK_OBSTACLE
    assert plan.target == (1, 2)
    assert "가로 3연속" in plan.reason


def test_세로_3연속은_특수_조건이_아니다():
    # 부수기를 아예 못 쓸 때(cost_break=None)의 최후 규칙이다.
    scene = make_scene([
        ".XX..",
        "XPX..",
        ".XX..",
        ".....",
        ".....",
    ])
    assert pathfind.horizontal_triples(scene.cells) == set()
    plan = plan_route(scene, cost_break=None)
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
        "Pi...",
        ".....",
        ".....",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert (0, 1) in plan.path, "아이템 칸을 피해 가면 안 됩니다"
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


def test_코앞의_칩은_그_행으로_옮겨_받는다():
    """2열 칩은 **이번 전진**에 내 자리로 들어온다. 지금 그 행에 있어야 한다."""
    scene = make_scene([
        ".....",
        ".P...",
        "..G..",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.GOAL
    assert plan.path == [(1, 1), (2, 1), (2, 2)],         f"칩이 있는 2행으로 내려가 전진해야 합니다: {plan.path}"


def test_가로막은_장애물_하나는_부수고_칩을_먹는다():
    """실측 회귀(62.2초): **가로로 놓인 장애물 때문에 칩을 계속 지나쳤다.**

    부수기를 59개나 쥐고 있는데도 부수기값이 매 사이클 12.0 으로 잡혔다.
    칩 값어치가 10 이라 `1 + 12 = 13 > 10` — 아무리 가까운 칩이라도 구조적으로
    닿을 수 없었다. 298초 기록에서 이렇게 지나친 칩이 20건이었다.

    값은 '남은 양의 비'(1422/59 = 24)로 매기고 있었는데, 그 비는 둘이 같은
    속도로 줄 때만 뜻이 있다. 실제로는 걸음수 142개가 나가는 동안 부수기는
    2개 나갔다. **먼저 바닥나는 쪽은 걸음수**이고, 부수기는 걸음수를 한 개도
    쓰지 않는다. 그러니 부수기를 걸음수 열두 개어치로 볼 이유가 없다.
    """
    scene = make_scene([
        "XXG..",
        ".PXX.",
        ".....",
        ".XGX.",
        "..X.X",
    ])
    plan = plan_route(scene, cost_break=pathfind.BREAK_COST_RANGE[1])
    assert plan.kind == PlanKind.GOAL,         f"막아선 장애물 하나 때문에 칩을 포기했습니다: {plan.describe()}"
    assert plan.path == [(1, 1), (0, 1), (0, 2)],         f"(0,1) 을 부수고 칩 (0,2) 로 가야 합니다: {plan.path}"


def test_장애물_두_겹_너머의_칩은_포기한다():
    """부수기를 아껴 쓴다. 한 겹까지만 뚫는다 — 2 + 12 = 14 > 칩값 10."""
    scene = make_scene([
        ".G...",
        "XX...",
        "XX...",
        ".P...",
        ".....",
    ])
    plan = plan_route(scene, cost_break=pathfind.BREAK_COST_RANGE[1])
    assert plan.kind != PlanKind.GOAL,         f"벽을 두 겹이나 뚫으려 합니다: {plan.describe()}"


def test_먼_칩_때문에_미리_행을_옮기지_않는다():
    """4열 칩은 **세 번 뒤** 전진에 들어온다. 지금 옮길 이유가 없다.

    미루는 편이 낫다 — 그사이 새 열이 들어와 더 나은 자리가 생길 수 있고,
    지금 옮기면 그 걸음수를 되돌려야 할 수도 있다. 앞을 내다보는 계획이라야
    이 구분이 된다(그리디는 무조건 칩 쪽으로 움직인다).
    """
    scene = make_scene([
        ".....",
        ".P...",
        "....G",
        ".....",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.path == [(1, 1), (1, 2)],         f"제자리에서 전진해야 합니다: {plan.path}"


# ------------------------------------- 주황칩(필수 아이템)이 여러 개인 경우
def _scene_with_chips(layout):
    from digimonup.vision.recognize import Detection
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
    assert plan.target == (1, 2),         f"같은 행(1행) 칩이 공짜인데 3행까지 내려갔습니다: {plan.target}"


def test_칩이_같은_거리면_더_오른쪽_칩을_고른다():
    scene = _scene_with_chips([
        ".G...",
        ".P...",
        ".....",
        ".....",
        ".....",
    ])
    scene.cells[2][1] = Kind.GOAL
    from digimonup.vision.recognize import Detection
    scene.goals.append(Detection(Kind.GOAL, 2, 1, 0.6))
    plan = plan_route(scene)
    assert plan.kind == PlanKind.GOAL
    assert plan.target in {(0, 1), (2, 1)}


def test_너무_비싼_칩은_쫓아가지_않는다():
    """장애물이 통과 가능해진 뒤로는 **한도가 없으면 벽을 여러 겹 뚫고** 칩을
    쫓아간다. 칩 하나의 값어치보다 비싸면 그냥 전진하는 편이 낫다.

    아래는 어느 쪽으로 가도 장애물 두 겹이라 값이 11 (부수기 2 x 5 + 걸음 1) 로
    한도 8 을 넘는다. 벽 한 겹뿐이라면 뚫고 먹는 것이 맞다(아래 테스트).
    """
    scene = _scene_with_chips([
        "XG...",
        "XX...",
        "XX...",
        ".P...",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.RIGHT_EDGE,         f"벽 두 겹 뒤의 칩을 쫓아가면 안 됩니다: {plan.describe()}"


def test_벽_한_겹_뒤의_칩은_뚫고_먹는다():
    """부수기는 걸음수를 쓰지 않으므로(25장) 한 겹은 뚫는 편이 싸다."""
    scene = _scene_with_chips([
        ".....",
        ".G...",
        ".X...",
        ".P...",
        ".....",
    ])
    plan = plan_route(scene)
    assert plan.kind == PlanKind.GOAL
    assert plan.path[-1] == (1, 1)


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


def test_값어치가_같으면_덜_움직이는_행을_고른다():
    """칩이 두 행에 있으면 가까운 행에서 전진한다."""
    plan = _plan_for([
        "....G",
        ".P...",
        "..G..",
        ".....",
        ".....",
    ])
    assert plan.path[-1][0] in (0, 2)
    assert len(plan.path) - 1 == 2, f"한 칸 옆 행을 두고 멀리 갔습니다: {plan.path}"


def test_아이템을_먹으려고_돌아가지_않는다():
    """아이템이 경로 밖에 있으면 무시한다. 걸음수가 늘어나면 안 된다."""
    plan = _plan_for([
        ".....",
        ".P..G",      # 1행에서 전진하면 칩이 들어온다
        ".....",
        ".....",
        "..i..",      # 아이템은 세 칸 아래. 값어치 1 로는 갈 값어치가 없다
    ])
    assert plan.path == [(1, 1), (1, 2)], f"돌아갔습니다: {plan.path}"


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
        ".P..G",
        ".....",
        ".....",
        ".....",
    ])
    assert plan.kind == PlanKind.GOAL
    assert plan.path == [(1, 1), (1, 2)]


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
    assert plan.path == [(3, 1), (2, 1), (2, 2)],         f"한 칸 위의 돌진 아이템을 지나쳤습니다: {plan.path}"


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
    assert plan.path == [(1, 1), (1, 2)], f"제자리 전진이면 됩니다: {plan.path}"


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
        "G....",
        ".....",
        ".P...",
        ".....",
        ".....",
    ])
    assert plan.target == (0, 0),         f"걸어서 닿는 칩은 먼저 먹어야 합니다: {plan.target}"


# ------------------- 걸음수 아이템은 하나에 +5 (사용자 확인)
def _scene_items(rows, kinds):
    from digimonup.vision.recognize import Detection, Scene
    sym = {".": Kind.EMPTY, "P": Kind.PLAYER, "G": Kind.GOAL,
           "X": Kind.OBSTACLE, "i": Kind.ITEM}
    cells = [[sym[ch] for ch in row] for row in rows]
    player = None
    for r in range(5):
        for c in range(5):
            if rows[r][c] == "P":
                player = Detection(Kind.PLAYER, r, c, 1.0)
    return Scene(grid=None, cells=cells, player=player, goals=[], item_kinds=kinds)


def test_걸음수_아이템은_두_칸_옮겨서라도_받는다():
    """하나에 +5 라서 네 걸음 이내로 움직이면 이득이다."""
    scene = _scene_items([
        ".....",
        ".....",
        "..i..",
        ".....",
        ".P...",
    ], {(2, 2): "steps"})
    plan = plan_route(scene)
    assert plan.path[-1] == (2, 2), \
        f"두 칸 올라가 +5 를 받는 편이 이득입니다: {plan.path}"


def test_칩과_걸음수_아이템이_겹치면_칩이_먼저다():
    """둘 다 갈 수 있으면 칩이다. 걸음수는 +5, 칩은 그보다 값이 크다."""
    scene = _scene_items([
        "..G..",
        ".....",
        ".P...",
        ".....",
        "..i..",
    ], {(4, 2): "steps"})
    plan = plan_route(scene)
    assert plan.path[-1] == (0, 2),         f"같은 두 칸이면 칩 쪽으로 가야 합니다: {plan.path}"


def test_네_칸_옮겨_걸음수를_받는_것은_이득이다():
    """네 칸 이동(-4) + 아이템(+5) = 순이득 +1. 사용자 확인: 하나에 +5."""
    scene = _scene_items([
        "..i..",
        ".....",
        ".....",
        ".....",
        ".P...",
    ], {(0, 2): "steps"})
    plan = plan_route(scene)
    assert plan.path[-1] == (0, 2), f"이득이면 받아야 합니다: {plan.path}"
