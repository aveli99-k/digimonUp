"""칩 묶음 추적.

매 프레임 칩을 새로 판단하지 않는다. **한 번 읽어 묶음으로 잠그고**, 그 묶음을
다 먹을 때까지는 새로 보이는 칩을 쳐다보지 않는다. 다 먹으면 그때 다시 읽는다.

왜 이렇게 하나
    칩을 먹으면 디지몬 주변으로 칩이 흩어지는 **획득 이펙트**가 뜬다. 그 칩들은
    템플릿 점수 0.94~0.98, 주황 비율 0.096~0.202 로 진짜 칩과 **같은 그림**이라
    모양·색·위치로는 가를 수 없다. 프레임마다 새로 판단하는 한 언젠가는 속는다.

    묶음으로 잠그면 이펙트가 아무리 오래, 아무리 그럴듯하게 떠도 **끼어들 자리가
    없다.** 잠근 뒤로는 검출 결과를 '새 칩을 찾는 데' 쓰지 않고, 알고 있던 칩이
    제자리에 잘 있는지 **확인하는 데만** 쓴다.

칩이 어떻게 움직이는가 (19장)
    판은 오른쪽으로 전진할 때만 밀린다. 전진 한 번에 모든 칸이 한 열씩 왼쪽으로
    옮겨진다. 그래서 칩의 자리는 계산으로 정확히 따라갈 수 있다.

        전진 k 번 뒤의 자리 = (행, 원래 열 - k)

    플레이어는 0~1열에만 있으므로, 2열의 칩은 **이번 전진**에 내 자리로 들어오고
    3열의 칩은 **다음 전진**에 들어온다. 즉 칩마다 '몇 번째 전진에 먹히는지'가
    정해져 있고, 그때 그 행에 서 있기만 하면 된다.

언제 다시 읽는가
    묶음이 비면(다 먹었거나 화면 밖으로 밀려났으면) 다시 읽는다. 다만 그 순간은
    마지막 칩을 막 먹은 직후라 이펙트가 한창일 때다. 그래서 **두 프레임 연속으로
    같은 칩 집합이 보일 때만** 잠근다. 묶음 하나에 한 번뿐이라 비용도 작다.
"""

from __future__ import annotations

Cell = tuple[int, int]


class ChipTracker:
    """칩 묶음을 잠그고, 전진할 때마다 자리를 옮겨 가며 따라간다."""

    # 알고 있던 칩이 이만큼 연속으로 안 보이면 없어진 것으로 본다.
    #
    # **1 이다. 곧 지금 화면에 보이지 않으면 바로 버린다.**
    #
    # 처음에는 3 이었다. 검출이 한 프레임쯤 놓치는 일이 흔하니 살려 두자는
    # 뜻이었다. 그런데 그 관용이 유령을 만들었다.
    #
    #   실측(300초): 매크로가 '칩을 먹었다'고 판단한 22건 중 **17건이 헛것**
    #   이었다(상단 보유량이 안 올랐다). 유령 칸은 전부 1열, 곧 플레이어가 선
    #   자리였다. 없어진 칩을 기억해 두었다가, 전진으로 그 자리가 밀려와
    #   플레이어와 겹치는 순간 '먹었다'고 처리한 것이다.
    #
    # 기억이 필요했던 이유는 검출을 못 믿어서였는데, 지금은 믿을 수 있다.
    # 12프레임을 눈으로 대조해 오탐 0 / 미검출 0 이었다(28장). 그러니 기억은
    # 이득 없이 위험만 남는다.
    MISS_LIMIT = 1

    def __init__(self, cols: int = 5):
        self.cols = cols
        self.chips: set[Cell] = set()
        self._misses: dict[Cell, int] = {}
        self._pending: set[Cell] | None = None    # 잠그기 전 후보 (두 번 확인용)

    # ------------------------------------------------------------- 상태
    @property
    def locked(self) -> bool:
        return bool(self.chips)

    # ------------------------------------------------------------- 갱신
    def advanced(self, times: int = 1) -> None:
        """오른쪽으로 times 번 전진했다. 판이 그만큼 왼쪽으로 밀린다."""
        if times <= 0:
            return
        moved = {}
        for r, c in self.chips:
            nc = c - times
            if nc >= 0:                      # 0열보다 왼쪽은 화면 밖이다
                moved[(r, nc)] = self._misses.get((r, c), 0)
        self.chips = set(moved)
        self._misses = moved

    def collected_at(self, cell: Cell) -> bool:
        """플레이어가 이 칸에 섰다. 거기 있던 칩은 먹은 것이다."""
        if cell not in self.chips:
            return False
        self.chips.discard(cell)
        self._misses.pop(cell, None)
        return True

    def update(self, detected: set[Cell], trust_now: bool = False) -> set[Cell]:
        """이번 프레임 검출 결과로 추적 목록을 손질하고, 유효한 칩을 돌려준다.

        알고 있던 칩이 계속 안 보이면 (MISS_LIMIT 회) 없어진 것으로 보고 뺀다.

        trust_now
            이번 화면에 획득 이펙트가 없다는 것을 **확인했으면** True.
            그러면 두 프레임을 기다리지 않고 바로 잠근다.

            두 프레임 연속 확인은 이펙트를 직접 알아볼 방법이 없던 때의
            대용품이었다. 지금은 '움직이는 칸의 칩은 이펙트'라는 직접 증거가
            있으므로, 그 검사를 통과했다면 한 사이클을 더 기다릴 이유가 없다.

            기다리면 손해가 크다. 실측(150초 65사이클): 칩이 처음 보인 사이클에
            바로 목표가 된 경우가 **0건**이었고, 36건이 한 사이클 이상 늦었다.
            그사이 전진해 버리면 칩이 뒤로 밀려 되돌아가서 먹어야 한다.
        """
        if trust_now:
            # **이펙트가 없다고 확인했으면 보이는 칩을 다 받아들인다.**
            #
            # 묶음을 닫아 두던 예전 방식은 이펙트를 가려낼 방법이 없던 때의
            # 것이다. 그때는 '한 번 읽고 다 먹을 때까지 안 본다'가 최선이었지만,
            # 그러면 그사이 오른쪽에서 새로 들어온 **진짜 칩도 버린다.**
            # 실측(180초 74사이클): 열 1~4 에 걸쳐 10사이클에서 칩을 무시했고,
            # 거기엔 (0,4), (3,4) 처럼 막 들어온 칩도 있었다.
            for chip in detected:
                self._misses.setdefault(chip, 0)
            self.chips |= set(detected)
            self._pending = None
        elif not self.chips:
            # 이펙트 여부를 확인하지 못했다. 두 프레임 연속으로 같아야 믿는다.
            if detected and self._pending == detected:
                self.chips = set(detected)
                self._misses = dict.fromkeys(detected, 0)
                self._pending = None
            else:
                self._pending = set(detected)
            return set(self.chips)
        else:
            self._pending = None
        for chip in list(self.chips):
            if chip in detected:
                self._misses[chip] = 0
            else:
                self._misses[chip] = self._misses.get(chip, 0) + 1
                if self._misses[chip] >= self.MISS_LIMIT:
                    self.chips.discard(chip)
                    self._misses.pop(chip, None)
        return set(self.chips)
