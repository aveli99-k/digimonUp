"""탐사 매크로 본체.

한 문장 요약:
    "고정된 MuMuPlayer 창의 5x5 게임판을 영상으로 인식하고, 장애물 없는 최단 경로를
     계산한 뒤, 매 이동의 실제 성공을 확인하면서 플레이어 기준으로 한 칸씩 클릭한다."

동작 순서 (빠른 연속 이동)
    1. 전체 화면 인식
    2. 전체 경로 계산
    3. 첫 번째 한 칸 클릭
    4. 화면 변화 확인
    5. 화면이 안정될 때까지 대기
    6. 플레이어 위치만 가볍게 재추적
    7. 예상 칸 도착이 확인되면 다음 한 칸 클릭
    8. 경로가 끝나면 다시 전체 화면 인식

매 이동 후에는 전체 객체 인식은 생략하되 격자선은 가볍게 다시 검출해서
다음 클릭의 셀 중심 좌표를 갱신한다.
"""

from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from digimonup.logic import chiptrack
from digimonup.vision import counters
from digimonup.base import trace
from digimonup.vision import popup
from digimonup.vision import overlay
from digimonup.app.engine import WindowedEngine
from digimonup.base.common import Stopped
from digimonup.vision.board import Grid, N, detect_board
from digimonup.win.emulator_window import enable_dpi_awareness
from digimonup.logic.pathfind import (ADVANCE_COL, PLAYER_MAX_COL, PlanKind,
                                      plan_route)
from digimonup.logic import pathfind
from digimonup.logic.pathfind import break_cost as pathfind_break_cost
from digimonup.vision.recognize import (Kind, Scene, TemplateSet, analyze,
                       find_blocked_toast,
                       find_green_button, find_top_tab, hsv_of, load_templates,
                       mask_highlight, mask_obstacle, motion_report,
                       track_player_fast, _frac)
from digimonup.vision.recognize import MOTION_CELL_MIN, MOTION_MAX_CELLS
from digimonup.vision.recognize import OBSTACLE_FRAC_WEAK

from digimonup.base.paths import DEBUG_DIR


@dataclass
class ExploreConfig:
    # 창 검증
    top_tab_min: float = 0.60          # 상단 고정 탭 템플릿 최소 유사도
    board_min: float = 0.45            # 5x5 격자 검출 최소 신뢰도
    # 앱플레이어로 알아보지 못한 창에 요구하는 더 높은 기준.
    # 이제는 모르는 창까지 후보로 올리므로(그래야 새 앱플레이어에서도 동작한다)
    # 게임과 무관한 창이 얻어걸릴 여지가 생긴다. 실측: 브라우저 창이 0.48~0.55 로
    # 기본 기준 0.45 를 넘었다. 진짜 게임 화면은 0.91 이라 넉넉히 갈린다.
    unknown_board_min: float = 0.70
    require_top_tab: bool = True       # 탭 템플릿이 있을 때만 강제된다
    # 앱플레이어를 여러 개 띄웠거나 자동 탐지가 엉뚱한 창을 고를 때, 창 제목의
    # 일부를 적어 두면 그 창만 후보로 본다. 비워 두면 전부 본다.
    window_title_hint: str = ""
    window_min_size: int = 200         # 이보다 작은 창은 앱플레이어일 수 없다
    # 격자를 처음 고정할 때 모을 최소 표 수. 게임이 다시 그리는 중간에 캡처되면
    # 격자선이 흐려져 한 칸 밀린 배치가 이긴다. 실측 프레임 모의에서 5장이면 오판 0%.
    grid_min_votes: int = 5

    # 이동
    click_settle_sec: float = 0.12     # 클릭 직후 최소 대기
    move_timeout_sec: float = 2.2      # 한 칸 이동 확인 최대 대기
    poll_interval_sec: float = 0.06    # 이동 확인 폴링 주기
    confirm_repeat: int = 2            # 예상 칸을 몇 번 연속 확인해야 성공으로 볼지
    move_duration: float = 0.05        # 마우스 이동 시간

    # 사이클
    # 움직임으로 플레이어 찾기
    # 디지몬은 서 있을 때도 제자리 애니메이션이 도는 판 위의 유일한 움직이는
    # 물체다. 색·모양·템플릿과 달리 디지몬을 바꿔도 그대로 통한다.
    # 실측: 3장 x 0.18초면 진짜 칸 0.296 / 나머지 24칸 0.000 으로 확실히 갈렸다.
    motion_frames: int = 3
    motion_gap_sec: float = 0.18
    # 칩 획득 이펙트가 보일 때, 묶음을 새로 읽기 전에 기다리는 시간.
    # 실측: 이펙트 칩은 전부 0.72초 안에 사라졌다.
    ghost_settle_sec: float = 0.8
    # 장애물을 부순 뒤, 그 자리를 다시 읽기 전에 기다리는 시간. 부서지는 연출이
    # 끝나고 아래에 있던 칩·아이템이 드러날 때까지다. 실측으로 0.45초면 충분했다
    # (61.3초에 (4,2) 를 부수고 그 뒤 첫 인식에서 드러난 칩을 바로 찾았다).
    # 장애물을 누른 뒤 **부서진 것을 확인할 때까지** 기다리는 최대 시간.
    # 이 게임은 화면 반영이 느리다. 실측 176.3초에 2.2초가 걸렸다.
    break_confirm_sec: float = 3.0

    cycle_pause_sec: float = 0.25      # 전체 재인식 사이의 쉬는 시간
    lost_retry_sec: float = 0.8        # 인식 실패 시 재시도 간격
    max_lost_before_report: int = 5

    # 이동 불가 안내
    toast_min: float = 0.65
    toast_clear_repeat: int = 2
    # 이동 확인 루프에서 안내문을 몇 번에 한 번 볼지.
    #
    # 1(매번)로 둔다. 2 로 건너뛰어 봤더니 검사 횟수는 줄었지만 안내문을 늦게
    # 알아채 확인 루프가 더 오래 돌았고, 실주행에서 이득이 없었다.
    # 검사 자체는 띠만 보도록 고쳐 이미 절반이 됐다(67ms -> 35ms).
    toast_check_every: int = 1
    # 판이 그대로인 채로 이만큼 연속 확인되면 '클릭이 안 먹었다'로 본다.
    dead_click_polls: int = 3
    # 다만 **클릭한 지 이만큼 지나기 전에는 단정하지 않는다.**
    #
    # 실측: '이동할 수 없습니다' 안내문은 클릭 후 0.40~0.42초에 뜬다(9건 전부).
    # 0.2초 만에 '안 먹었다'고 판단하면 사실은 게임이 거부한 클릭을 못 먹은
    # 것으로 오해해 **막힌 칸을 한 번 더 누른다.** 그러면 안내문이 또 떠서
    # 사라지길 2초 더 기다린다.
    dead_click_wait_sec: float = 0.6
    # 판을 가리는 팝업(실패창/보상창)을 닫고 기다리는 시간.
    # 실측(던전): 바깥을 누르면 0.5초 안에 닫힌다.
    popup_settle_sec: float = 0.8
    # 먹지 않은 클릭을 몇 번까지 다시 눌러 볼지.
    #
    # 아무 일도 안 일어났으므로 두 번 움직일 위험이 없고, 판을 다시 인식하는
    # 것(인식 0.37초 + 움직임 촬영 0.45초)보다 싸다.
    #
    # 다만 **한 번이면 충분하다.** 실측(180초): 두 번까지 허용하니 재시도 18회로
    # 실패가 21 -> 13건으로 줄었지만, 남은 실패 하나에 드는 시간이 0.97 -> 3.03초로
    # 늘어 총합으로는 이득이 없었다. 재시도 성공률이 44% 라 두 번째는 대개 헛수고다.
    dead_click_retries: int = 1

    # 목적지 = 주황칩(필수 아이템)
    # 탐사에는 종착점이 없지만, 판에 나오는 주황칩은 반드시 먹어야 한다.
    # 그래서 주황 카드를 목적지로 취급해 1순위로 가져간다.
    orange_goal_without_template: bool = True

    # 판 위의 아이템 들르기
    # 칩이 없을 때, 전진 경로에서 벗어나 있는 아이템을 이 칸수까지는 들러서 먹는다.
    # 0 이면 목표로 삼지 않는다(가는 길에 걸리면 여전히 먹는다).
    # 걸음수 아이템은 들르지 않는다(pathfind.DETOUR_SKIP_KINDS).
    item_max_detour: int = 2

    # 아이템 개수 (왼쪽 아래 걸음수 / 부수기 / 돌진)
    # 개수를 읽어서 **0 이면 아예 시도하지 않는다.** 예전에는 해 보고 안내문이
    # 뜨면 실패로 세는 방식이라, 쓸 게 없을 때도 두 번씩 헛클릭하고 안내문이
    # 사라지길 기다렸다. 숫자 템플릿이 없어 못 읽으면 예전 방식으로 돌아간다.
    watch_counters: bool = True
    stop_when_out_of_steps: bool = True   # 걸음수가 0 이면 매크로를 멈춘다

    # 장애물 파괴
    # 오른쪽이 막히면 장애물을 직접 클릭하거나, 우측 하단 초록색 버튼을 눌러
    # 부술 수 있다. 초록 버튼은 사용 횟수가 정해져 있으므로(실측: 30회)
    # 먼저 장애물 클릭을 시도하고, 그게 안 먹힐 때만 버튼을 쓴다.
    allow_obstacle_break: bool = True
    obstacle_break_max_failures: int = 2
    use_green_button: bool = True
    # 돌진(초록 버튼)으로 세 칸씩 나아간다.
    # 실측: 돌진 1개 = 세 칸 전진, 걸음수 0, 지나가는 칩도 먹는다.
    use_dash: bool = True
    dash_cells: int = 3        # 돌진 한 번에 나아가는 칸 수 (실측)
    green_button_max_uses: int = 0      # 0 = 제한 없음
    blocked_wait_sec: float = 2.0

    # 중지 키 (창 포커스와 무관하게 어디서 눌러도 먹는다)
    # 매크로가 마우스를 계속 움직이는 중에는 GUI 의 정지 버튼을 누르기가 까다롭다.
    # 빈 문자열이면 키로는 멈추지 않는다.
    stop_key: str = "F12"

    # 디버그
    save_debug: bool = True
    # 판단 근거를 통째로 남긴다 (debug/trace/<시각>/log.jsonl).
    # 무엇을 보고 왜 그렇게 했는지 나중에 그대로 되짚을 수 있다.
    # tools/analyze_trace.py 로 읽는다.
    trace: bool = False
    trace_frames: bool = False       # 사이클마다 화면도 남길지 (용량이 크다)


# 셀 요약이 이만큼도 안 바뀌었으면 '판이 그대로'로 본다.
# 실측: 먹히지 않은 클릭 0.001~0.006 / 성공한 전진 0.07~0.17.
DEAD_CLICK_SAME = 0.02

# 돌진 한 번에 챙겨야 하는 칩의 최소 개수. 하나짜리는 걸어가는 편이 낫다
# (걸음수 최대 3 대 돌진 1, 남은 양은 1550 대 45). _dash_if_worth 참고.
DASH_MIN_CHIPS = 2


def advances(direction: str, frm: tuple[int, int]) -> bool:
    """이 이동이 게임판을 한 열 미는가 (= 전진인가).

    실측(19장): 판이 밀리는 것은 **1열에서 오른쪽을 눌렀을 때뿐**이다.
        0열에서 오른쪽   스크롤 X  (1/1)   플레이어가 1열로 걸어갈 뿐이다
        1열에서 오른쪽   스크롤 O  (8/8)
    위/아래/왼쪽은 150초 12회 전부 지형 변화가 없었다.

    이 조건을 **한 곳에만** 둔다. 전에는 걸음수 감소로 성공을 판정하는 자리에만
    적혀 있고, 화면 비교로 판정하는 자리에는 없었다. 스크롤로 잘못 세면 칩
    추적기가 칩 자리를 한 열 더 밀어, 없는 칩이 플레이어 자리로 들어와 '먹었다'로
    처리된다 — 사용자가 유령칩이라고 부른 증상이다.
    """
    return direction == "RIGHT" and frm[1] == PLAYER_MAX_COL


@dataclass
class ExploreStats:
    cycles: int = 0
    moves: int = 0
    failed_moves: int = 0
    blocked_toasts: int = 0
    obstacles_broken: int = 0
    green_button_uses: int = 0


class ExploreEngine(WindowedEngine):
    """탐사 자동화 엔진.

    GUI 든 콘솔이든 콜백만 갈아끼우면 그대로 쓸 수 있게 분리했다.
    창 고르기·캡처·정지 처리는 던전과 똑같아서 WindowedEngine 에 한 벌만 둔다.
    """

    def __init__(self, cfg: ExploreConfig | None = None,
                 log=print, status=lambda s: None, preview=lambda img: None):
        self.cfg = cfg or ExploreConfig()
        super().__init__(self.cfg.stop_key, log, status, preview)
        self.templates: dict[str, TemplateSet] = load_templates()
        self.stats = ExploreStats()
        # 장애물 파괴 연속 실패 횟수. 한도를 넘으면 파괴 시도를 접는다.
        self.break_fail_streak = 0
        self.break_disabled = not self.cfg.allow_obstacle_break
        # 막힌 주머니에서 제자리를 맴도는 것을 잡아내기 위한 상태
        self._last_layout: tuple | None = None
        # 직전 이동으로 알아낸 플레이어 자리. 인식이 말이 되는지 대조한다.
        self._expect: tuple | None = None
        # 한 번 다시 본 자리. 두 번째도 같으면 인식을 믿는다.
        self._recheck: tuple | None = None
        # 지금 배치에서 이미 서 봤던 자리들 (맴도는지 가르는 데 쓴다)
        self._seen_spots: set = set()
        self.stuck_cycles = 0
        # 고정된 격자. 게임판 패널은 화면에서 움직이지 않고 내용만 스크롤한다.
        self.locked_grid: Grid | None = None
        self._locked_size: tuple[int, int] | None = None
        self._grid_votes: deque = deque(maxlen=9)
        # 왼쪽 아래 아이템 개수. 못 읽으면 항목이 None 이다.
        self.counts = counters.Counters()
        self._last_counts_line = ""
        # 마지막 전체 인식에서 알아낸 '플레이어가 아닌 칸'(장애물/칩/아이템).
        # 이동 확인 중 빠른 추적이 그것들을 플레이어로 잡지 않도록 넘겨준다.
        self._not_player: set = set()
        # 마지막 움직임 검사에서 '판이 아직 스크롤 중'이었는지.
        # 그런 프레임은 칩/장애물 인식도 믿을 수 없어 그 사이클을 통째로 건너뛴다.
        self._board_animating = False
        # 칩 묶음 추적기. 한 번 읽어 잠그고 다 먹을 때까지 새 칩을 보지 않는다.
        # 자세한 근거는 chiptrack.py 참고.
        self.chips = chiptrack.ChipTracker(cols=N)
        self._scrolls_since = 0
        # 마지막 움직임 검사에서 그림이 움직이던 칸들.
        self._moving_cells: set = set()
        # 이번 사이클에 이펙트로 의심된 칩 수. 있으면 계획을 세우지 않고
        # 화면이 가라앉기를 기다렸다 다시 본다.
        self._ghost_suspect = 0
        # 이번 사이클에 움직임 검사가 쓸 만한 결과를 냈는가.
        # (판이 스크롤 중이었거나 캡처가 모자라면 False)
        self._motion_valid = False
        # 지금까지 본 걸음수 최솟값. 이동 성공을 가려내는 데 쓴다(_note_steps).
        self._steps_min: int | None = None
        # 이번 이동에서 '무슨 일이 있었는지 확실치 않다'는 표시.
        # 남은 경로를 버리고 다시 인식해야 한다는 뜻이다(스크롤과는 다르다).
        self._path_dirty = False
        self.trace = trace.Tracer(self.cfg.trace, self.cfg.trace_frames)
        # 직전 폴링의 '판이 그대로' 점수 (_scrolled_one_cell 이 채운다).
        self._last_same: float | None = None
        self._last_moved: float | None = None
        # 직전 사이클에 보인 0열 칩. 되돌아가야 하는 칩은 두 번 봐야 인정한다.
        self._prev_zero: set = set()
        # '부수기가 0개라 안 누른다'를 이미 알렸는가. 매 사이클 같은 줄을
        # 쏟아내지 않으려는 표시일 뿐, 판단을 붙잡아 두지는 않는다.
        self._told_no_breaks = False

    # 걸음수가 이만큼 넘게 늘면 '채워 넣었다'로 보고 기준을 다시 잡는다.
    _STEPS_REFILL = 5

    # ------------------------------------------------------------ 창 고정
    # 창 고르기의 뼈대는 WindowedEngine 에 있다. 여기서는 **무엇을 보고 고르는지**
    # 두 조건만 채운다.
    #   1) 상단에 고정된 게임 탭 이미지가 있는가 (템플릿이 있을 때만)
    #   2) 화면 안에 5x5 게임판 격자 테두리가 있는가
    def _prepare_judging(self) -> None:
        if self.templates["top_tab"] and not self.cfg.require_top_tab:
            self.log("[창] require_top_tab=false 라 상단 탭 검사를 건너뜁니다.")

    def _judge(self, img: np.ndarray, cand) -> None:
        tab_tpl = self.templates["top_tab"] if self.cfg.require_top_tab else None
        grid = detect_board(img, min_confidence=0.0)
        cand.board_score = grid.confidence if grid else 0.0
        if tab_tpl:
            cand.tab_score, _ = find_top_tab(img, tab_tpl)
        else:
            cand.tab_score = 0.0
            cand.reasons.append("상단 탭 템플릿 없음(격자만으로 판정)")

        # 앱플레이어라고 확신하지 못한 창에는 더 높은 기준을 요구한다.
        need = (self.cfg.board_min if cand.emulator
                else max(self.cfg.board_min, self.cfg.unknown_board_min))
        board_ok = cand.board_score >= need
        tab_ok = (not tab_tpl) or (cand.tab_score >= self.cfg.top_tab_min)
        if not board_ok:
            cand.reasons.append(
                f"격자 신뢰도 부족 {cand.board_score:.2f} (기준 {need:.2f}"
                + ("" if cand.emulator else ", 모르는 창이라 기준이 높음") + ")")
        if not tab_ok:
            cand.reasons.append(f"상단 탭 불일치 {cand.tab_score:.2f}")
        cand.ok = board_ok and tab_ok

    def _no_match_help(self, n_candidates: int) -> list[str]:
        return [
            f"후보 {n_candidates}개를 모두 봤지만 5x5 게임판이 있는 창이 "
            f"없습니다. 탐사 화면을 띄운 상태인지 확인하세요.",
            "창은 뜨는데 계속 실패하면 tools/detect_windows.py 를 실행해 "
            "어떤 창이 어떻게 보이는지 확인할 수 있습니다.",
        ]

    def _picked_note(self, cand) -> str:
        return f"격자={cand.board_score:.2f} 탭={cand.tab_score:.2f}"

    # ------------------------------------------------- 이동 불가 안내 처리
    def _toast_visible(self, img: np.ndarray) -> bool:
        tset = self.templates["blocked_toast"]
        if not tset or img is None:
            return False
        score, _ = find_blocked_toast(img, tset)
        return score >= self.cfg.toast_min

    def _wait_toast_clear(self) -> None:
        """'이동할 수 없습니다' 안내가 사라질 때까지 기다린다.

        이 안내는 OK 버튼이 있는 모달이 아니라 잠시 뒤 저절로 사라지는 알림이다.
        따라서 **안내문을 클릭하면 안 된다.** 클릭하면 안내가 사라지는 순간
        그 클릭이 게임판 중앙으로 전달돼 같은 오류가 반복된다.
        """
        self.stats.blocked_toasts += 1
        self.log("[안내] '이동할 수 없습니다' 감지 -> 마우스 입력 중지, 클릭하지 않고 대기")
        clear = 0
        deadline = time.time() + 6.0
        while time.time() < deadline:
            self._check_stop()
            time.sleep(0.15)
            img = self._capture()
            if img is None:
                continue
            if self._toast_visible(img):
                clear = 0
            else:
                clear += 1
                if clear >= self.cfg.toast_clear_repeat:
                    self.log("[안내] 안내가 사라진 화면을 2회 연속 확인 -> 전체 재인식")
                    return
        self.log("[안내] 안내가 오래 남아 있습니다. 전체 재인식으로 넘어갑니다.")

    # ------------------------------------------------------- 격자 고정
    @staticmethod
    def _grid_key(grid: Grid) -> tuple[int, int, int, int]:
        """몇 px 흔들리는 것은 같은 배치로 묶기 위한 열쇠."""
        return (grid.xs[0] // 8, grid.ys[0] // 8,
                int(grid.cell_w) // 8, int(grid.cell_h) // 8)

    def _stable_grid(self, img: np.ndarray, seed: bool = False) -> Grid | None:
        """최근 프레임들의 **다수결**로 격자를 정한다.

        게임판 패널은 화면에서 움직이지 않는다. 스크롤하는 건 판의 내용일 뿐
        격자선의 화면 좌표는 그대로다. 그런데 매 프레임 새로 검출하면 한 칸 밀린
        배치와 진짜 배치의 점수가 종이 한 장 차이라 위상이 왔다 갔다 한다.
        (실측: 같은 칸의 클릭 y 좌표가 375 와 463 으로 한 행씩 튀었고, 그 바람에
         주황칩이 두 칸에 걸쳐 인식되지 않고 엉뚱한 곳을 클릭했다.)

        처음 한 장만 보고 고정했더니 그 한 장이 틀렸을 때 계속 틀린 채로
        굳어 버렸다. 그래서 최근 몇 프레임을 모아 가장 많이 나온 배치를 쓴다.
        틀린 검출이 한두 번 섞여도 다수결로 걸러지고, 판이 정말 바뀌면
        몇 프레임 안에 새 배치로 갈아탄다.
        """
        size = img.shape[:2]
        if size != self._locked_size:
            self._grid_votes.clear()
            self.locked_grid = None
            self._locked_size = size

        grid = detect_board(img, min_confidence=self.cfg.board_min)
        if grid is not None:
            self._grid_votes.append((self._grid_key(grid), grid))

        # 아직 고정된 격자가 없으면 **한 장만 보고 정하지 않는다.**
        # 실측(실제 MuMuPlayer 40프레임): 게임이 다시 그리는 중간에 캡처되면
        # 격자선이 흐려져서 한 칸 밀린 배치가 이긴다. 40장 중 3장(7.5%)이 그랬다.
        # 그 한 장이 하필 첫 프레임이면 잘못된 격자로 경로를 짜서 엉뚱한 칸을
        # 클릭하게 된다. 그래서 처음에는 몇 장을 더 모아 다수결로 정한다.
        # seed 는 전체 인식(사이클 시작)에서만 켠다. 이동 확인 루프는 폴링마다
        # 도는 자리라 여기서 캡처를 더 하면 확인 기회가 줄어 손해다.
        if seed and self.locked_grid is None:
            need = self.cfg.grid_min_votes - len(self._grid_votes)
            for _ in range(max(0, need)):
                self._check_stop()
                # 연속으로 바로 찍으면 같은 중간 프레임을 또 잡을 수 있어 조금 띄운다.
                self._sleep(0.05)
                extra = self._capture()
                if extra is None or extra.shape[:2] != size:
                    break
                g2 = detect_board(extra, min_confidence=self.cfg.board_min)
                if g2 is not None:
                    self._grid_votes.append((self._grid_key(g2), g2))

        if not self._grid_votes:
            return self.locked_grid

        # 배치마다 '가장 잘 나온 프레임'의 신뢰도로 겨룬다. 표 개수로 뽑지 않는다.
        #
        # 실측(실제 MuMuPlayer 60프레임): 게임이 다시 그리는 중간에 캡처된 프레임은
        # 무리 지어 들어온다(연속 3~4장). 그래서 표를 더 모아도 다수결은 6~7% 에서
        # 안 떨어졌다. 반면 두 배치의 신뢰도는 아예 겹치지 않았다.
        #     맞는 배치  0.882 ~ 0.911
        #     밀린 배치  0.830 ~ 0.853
        # 그래서 '한 장이라도 잘 나온 쪽'을 믿는 편이 훨씬 정확하다.
        # 같은 실측 프레임으로 모의해 보면 5장 기준 오판이 6.6% -> 0% 가 된다.
        best: dict[tuple, float] = {}
        counts = Counter(k for k, _ in self._grid_votes)
        for k, g in self._grid_votes:
            if g.confidence > best.get(k, 0.0):
                best[k] = g.confidence
        best_key = max(best, key=lambda k: (best[k], counts[k]))
        chosen = max((g for k, g in self._grid_votes if k == best_key),
                     key=lambda g: g.confidence)
        votes = counts[best_key]

        if self.locked_grid is None:
            self.log(f"[격자] 고정: x={chosen.xs} y={chosen.ys} "
                     f"(신뢰도 {chosen.confidence:.2f}, {votes}/{len(self._grid_votes)}표)")
        elif (chosen.xs[0] // 8, chosen.ys[0] // 8) != (
                self.locked_grid.xs[0] // 8, self.locked_grid.ys[0] // 8):
            self.log(f"[격자] 갱신: y0 {self.locked_grid.ys[0]} -> {chosen.ys[0]} "
                     f"(신뢰도 {chosen.confidence:.2f}, {votes}/{len(self._grid_votes)}표)")

        self.locked_grid = chosen
        return chosen

    # ------------------------------------------------- 움직임으로 플레이어 찾기
    def _motion_cell(self, img: np.ndarray, grid: Grid) -> tuple[int, int] | None:
        """짧은 간격으로 몇 장 더 찍어 움직인 칸을 찾는다.

        자세한 근거는 recognize.motion_report 참고. 판이 스크롤하는 중이면
        온 화면이 움직이므로 그쪽에서 None 을 돌려주고, 그때는 기존 방식으로
        되돌아간다.
        """
        if self.cfg.motion_frames < 2:
            return None                 # 움직임 인식 끔
        self._board_animating = False
        self._motion_valid = False

        # 두 번까지 시도한다. 첫 묶음은 스크롤 직후라 화면 전체가 움직이는
        # 중일 때가 많아 못 쓴다(실측: 사이클의 절반에서 움직임을 못 썼다).
        # 그때는 조금 더 기다렸다가 한 번 더 찍으면 판이 가라앉아 있다.
        frames: list[np.ndarray] = [img]
        for attempt in range(2):
            while len(frames) < self.cfg.motion_frames * (attempt + 1):
                self._check_stop()
                time.sleep(self.cfg.motion_gap_sec)
                shot = self._capture()
                if shot is None:
                    return None
                frames.append(shot)
            recent = frames[-self.cfg.motion_frames:]
            busy, cell, ratio, ratios = motion_report(recent, grid)
            if busy > MOTION_MAX_CELLS:
                self._board_animating = True
                continue                # 아직 스크롤 중이다. 한 번 더 기다린다.
            self._board_animating = False
            self._motion_valid = True
            # 판이 가라앉은 지금 움직이는 칸 = 디지몬 + 연출.
            # 거기서 잡힌 칩은 획득 이펙트다(_confirm_goals 에서 쓴다).
            self._moving_cells = {c for c, v in ratios.items()
                                  if v >= MOTION_CELL_MIN}
            if cell is not None and ratio >= MOTION_CELL_MIN:
                return cell
            return None
        return None

    def _note_steps(self, got) -> bool:
        """읽은 걸음수를 최솟값에 반영한다. **새로 줄었으면** True.

        걸음수는 줄기만 하므로 지금까지 본 최솟값이 진짜 값에 가장 가깝다.
        화면이 늦게 갱신될 때 보이는 값은 그보다 크거나 같다.
        """
        steps = got.steps if got else None
        if steps is None:
            return False
        prev = self._steps_min
        if prev is None or steps > prev + self._STEPS_REFILL:
            # 처음이거나, 걸음수를 채워 넣어 값이 뛴 경우다. 기준을 새로 잡는다.
            self._steps_min = steps
            return False
        if steps < prev:
            self._steps_min = steps
            return True
        return False

    def _steps_dropped(self, img: np.ndarray) -> bool:
        """걸음수가 지금까지 본 것보다 더 줄었는가. 못 읽으면 False."""
        if not self.cfg.watch_counters:
            return False
        return self._note_steps(counters.read(img))

    def _arrived_by_motion(self, frames, grid: Grid,
                           to: tuple[int, int]) -> bool:
        """**움직이는 칸이 목표 칸 하나뿐**이면 도착한 것으로 본다.

        판이 멈춰 있을 때 움직이는 것은 디지몬뿐이다(20장). 색 기반 빠른
        추적과 달리 생김새에 기대지 않아 헛값이 없다.

        확인 루프는 어차피 폴링마다 캡처하므로, 그중 motion_gap_sec 만큼
        떨어진 두 장을 골라 쓰면 추가 촬영이 필요 없다. 간격이 좁으면 제자리
        애니메이션의 변화가 작아 놓친다(실측: 0.18초 5/5, 0.08초 3/5).
        """
        if len(frames) < 2 or grid is None:
            return False
        newest_t, newest = frames[-1]
        for t, old in reversed(frames):
            if newest_t - t >= self.cfg.motion_gap_sec:
                busy, cell, ratio, _ = motion_report([old, newest], grid)
                return (busy == 1 and cell == to
                        and ratio >= MOTION_CELL_MIN)
        return False

    # ------------------------------------------- 갇힘 처리 / 장애물 부수기
    def _handle_blocked(self, scene: Scene, plan) -> bool:
        """막힌 주머니에 갇혔는지 보고, 필요하면 부수거나 기다린다.

        반환: True 면 이번 사이클은 여기서 끝낸다(이동하지 않는다).

        오른쪽으로 한 칸도 못 가는 상태에서 장애물 배치까지 그대로면, 세로로
        움직여도 새 지형이 들어오지 않는 '주머니'에 갇힌 것이다. 이때 계속
        움직이면 (1,0)->(1,1)->(2,1)->(1,1)->... 처럼 제자리를 맴돌며 이동
        횟수만 축낸다(실측: 25번을 움직였는데 판이 그대로였다).

        플레이어 칸은 움직일 때마다 바뀌므로 빼고 **장애물 배치만** 본다.
        새 지형이 들어왔다면 장애물 배치가 반드시 달라진다.
        """
        layout = tuple(tuple(c == Kind.OBSTACLE for c in row)
                       for row in scene.cells)
        # 칩이나 아이템을 먹으러 가는 중이라면 '갇힘'으로 세지 않는다.
        #
        # 실측 교착: 갇힘 처리는 이동을 건너뛰고 만다. 그런데 이동을 안 했으니
        # 장애물 배치는 당연히 그대로고, 그래서 다음 사이클도 갇힘으로 판정돼
        # 또 건너뛴다. 스스로를 강화하는 교착이다. (실측: 칩 (2,3) 을 향해 UP 을
        # 계획해 놓고 P(4,4) 에서 27초 동안 11사이클을 한 발짝도 못 갔다.)
        # 갈 곳이 분명하면 갇힌 게 아니므로 그냥 가면 된다.
        worth_going = plan.kind in (PlanKind.GOAL, PlanKind.ITEM)
        # **같은 배치에서 가 본 자리로 또 가고 있으면** 그때가 맴도는 것이다.
        #
        # 배치가 그대로라는 것만으로는 부족하다. 전진할 수 있는 행까지 걸어서
        # 올라가는 동안에도 배치는 그대로다. 실측 170.8~178.3초: (3,1) 에서
        # 1행까지 네 칸을 걸어 올라가는 중이었는데 — 매 사이클 이동에
        # **성공**하고 있었는데 — 갇힘으로 세어 3사이클 만에 초록 버튼을 눌러
        # 돌진 하나를 태웠다(2 -> 1). 그때 계획은 '오른쪽 전진'이었다.
        #
        # 새 자리로 가고 있으면 나아가는 중이고, 왔던 자리로 되돌아가면 맴도는
        # 것이다. 예전에 잡았던 (1,0)->(1,1)->(2,1)->(1,1) 무한 왕복은 이
        # 기준으로도 그대로 걸린다.
        here = (scene.player.row, scene.player.col) if scene.player else None
        if layout != self._last_layout:
            self._seen_spots = set()
        looping = here is not None and here in self._seen_spots
        if here is not None:
            self._seen_spots.add(here)
        if layout == self._last_layout and looping and not worth_going:
            self.stuck_cycles += 1
        else:
            self.stuck_cycles = 0
        self._last_layout = layout

        if self.stuck_cycles >= 3:
            # 오른쪽이 막혔다. 초록 버튼으로 장애물을 부순다.
            if self._press_green_button():
                self.stuck_cycles = 0
                self._last_layout = None
                return True
            self.log(f"[갇힘] {self.stuck_cycles}사이클째 장애물 배치가 "
                     f"그대로인데 초록 버튼도 쓸 수 없습니다. "
                     f"{self.cfg.blocked_wait_sec:g}초 기다립니다.")
            self._sleep(self.cfg.blocked_wait_sec)
            return True

        # 부수기 아이템이 **0 이라고 읽었을 때만** 파괴를 접는다.
        #
        # 예전에는 _can_use() 가 False 면 접었다. 그런데 _can_use 는 개수를
        # **못 읽었을 때도** False 를 돌려준다(아껴 쓰기 규칙, 30장). 그러면
        #     개수 읽기 실패 -> "부수기 0개" -> break_disabled 영구 설정
        #     -> 장애물을 벽으로 취급 -> 부수고 갈 길을 두고 오른쪽만 시도
        # 로 이어진다. 실측: 경로 계산 32회 내내 장애물을 벽으로 봤고, 그중
        # 29회는 break_disabled 가 켜진 뒤였다. 아이템은 멀쩡히 있었다.
        #
        # 못 읽는 것은 **일시적**이다(숫자가 네 자리가 되거나 옆에 타이머가
        # 뜨면 막힌다). 그걸로 기능을 영구히 끄면 안 된다.
        #
        # **개수가 0이라고 기능을 끄지도 않는다.** 0 은 되돌아오는 상태다
        # (아이템은 다시 채워진다). 예전에는 여기서 break_disabled 를 세워
        # 버려서, 한 번 바닥나면 그 뒤로 아무리 채워 넣어도 다시는 부수지
        # 않았다 — 장애물을 영영 벽으로 보게 되는데, 그건 위 주석이 막으려던
        # 바로 그 사고다. break_disabled 는 '눌러도 안 부서지는 게임'이라는
        # 판정에만 남긴다. 그것만이 되돌아오지 않는 사실이다.
        left = self.counts.break_
        out_of_breaks = left is not None and left <= 0
        if not out_of_breaks:
            self._told_no_breaks = False
        elif plan.kind == PlanKind.BREAK_OBSTACLE and not self._told_no_breaks:
            # 매 사이클 같은 줄을 쏟아내지 않도록 한 번만 알린다.
            self.log("[장애물] 부수기 아이템이 0개라 클릭하지 않습니다.")
            self._told_no_breaks = True

        if plan.kind == PlanKind.BREAK_OBSTACLE and (self.break_disabled
                                                     or out_of_breaks):
            # 장애물 직접 클릭이 안 먹히는 것을 확인했다면 초록 버튼을 쓴다.
            if self._press_green_button():
                self._last_layout = None
                return True
            self.log(f"[경로] 우회로도 없고 장애물도 부술 수 없습니다. "
                     f"{self.cfg.blocked_wait_sec:g}초 기다렸다가 다시 봅니다.")
            self._sleep(self.cfg.blocked_wait_sec)
            return True
        return False

    def _dash_if_worth(self, scene: Scene) -> bool:
        """돌진(우측 하단 초록 버튼)을 **아껴서** 쓴다. 썼으면 True.

        실험으로 확정한 규칙
            돌진 1개 소모 / 걸음수 0 / 부수기 0 / **세 칸 전진**
            (거리 29,714m -> 29,717m, 돌진 46 -> 45, 걸음수 1699 그대로)
            그리고 **지나가는 칩을 먹는다**
            (같은 행 2열 칩을 두고 돌진하니 보유량 267.8K -> 268.0K)

        **그래서 돌진 하나의 값은 걸음수 세 개다.** 그런데 남은 양은 걸음수
        1550 대 돌진 45 로 서른네 배 차이가 난다. 빈 길에서 돌진을 쓰면
        걸음수 셋을 아끼자고 훨씬 귀한 것을 버리는 셈이다.

        그래서 **걸어서는 못 먹을 때만 쓴다.** 조건은 셋 다 맞아야 한다.
            1. 앞 세 열 **내 행**에 칩이 **둘 이상** 있다
            2. 앞 세 열 **다른 행**에는 칩이 없다    -> 지나쳐 잃을 것이 없다
            3. 그 길에 **장애물이 있다**             -> 뚫려 있으면 그냥 걸어간다

        3번을 넣은 이유 (실측 회귀, 299초에 돌진한 두 번 다)
            63.7초  플레이어 (0,1), 칩 (0,2)·(0,4), 0행에 장애물이 **하나도 없음**
            258.7초 플레이어 (0,1), 칩 (0,2)·(0,3), 마찬가지로 뚫려 있음
            둘 다 오른쪽만 누르면 걸음수 몇 개로 전부 먹는 자리였다. 아이템은
            길이 막혔을 때 쓰는 것이지, 뚫린 길에서 걸음수를 아끼자고 쓰는
            것이 아니다.

        1번이 '하나'가 아니라 '둘 이상'인 이유 (실측 회귀, 67.5초)
            막다른 길에서 (4,2) 장애물을 부쉈더니 그 자리에서 칩이 나왔다.
            바로 옆 칸이라 오른쪽 한 번이면 걸음수 1로 먹는 자리였고, 실제로
            경로도 그렇게 잡혀 있었다. 그런데 여기서 그 계획을 가로채 **마지막
            남은 돌진(1 -> 0)** 을 썼다. 걸음수 1을 아끼자고 돌진을 버린 것이다.

            칩 하나는 걸어가면 걸음수 최대 3이다. 돌진 하나가 걸음수 3보다
            귀하므로(1550 대 45) 하나짜리는 언제나 걸어가는 편이 낫다.
        """
        if not self.cfg.use_dash or not self._can_use("dash"):
            return False
        if scene.player is None:
            return False

        row = scene.player.row
        span = range(ADVANCE_COL, min(N, ADVANCE_COL + self.cfg.dash_cells))
        mine = [(d.row, d.col) for d in scene.goals if d.row == row and d.col in span]
        if len(mine) < DASH_MIN_CHIPS:
            # 하나뿐이면 걸어간다. 걸음수 최대 3이면 먹는데, 돌진은 그보다 귀하다.
            if mine:
                self.log(f"[돌진] 내 행의 칩이 {sorted(mine)} 하나뿐이라 걸어서 "
                         f"챙깁니다 (돌진을 아낍니다).")
            return False

        blocked = [c for c in span if scene.kind_at(row, c) == Kind.OBSTACLE]
        if not blocked:
            self.log(f"[돌진] {row}행 앞이 뚫려 있어 걸어서 칩 {sorted(mine)} 을(를) "
                     f"챙깁니다 (돌진을 아낍니다).")
            return False

        missed = [(d.row, d.col) for d in scene.goals
                  if d.row != row and d.col in span]
        if missed:
            self.log(f"[돌진] 다른 행의 칩 {sorted(missed)} 을(를) 지나치게 되어 "
                     f"돌진하지 않습니다.")
            return False

        if self._press_green_button():
            self.trace.write("dash", row=row, chips=[list(m) for m in mine],
                             left=self.counts.dash)
            self.log(f"[돌진] 내 행의 칩 {sorted(mine)} 을(를) 챙기며 세 칸 "
                     f"전진합니다 (남은 돌진 "
                     f"{self.counts.dash if self.counts.dash is not None else '?'}).")
            self._last_layout = None
            return True
        return False

    def _odd_position(self, here: tuple[int, int]) -> str:
        """이 자리가 말이 되는가. 이상하면 **왜 이상한지**를, 괜찮으면 빈 글자를.

        1) 플레이어는 0~1열에만 있다(19장). 2열로 잡혔다면 판이 밀리는 도중을
           찍은 것이다. 예전에는 그 자리를 곧이곧대로 믿고 **왼쪽으로 되돌리는
           이동**을 계획했다. 게임은 당연히 거부했고, 실측 209.4초·233.2초에
           2.1초씩 버렸다.

        2) 직전 이동이 성공했다면 지금 어디 있어야 하는지 정확히 안다. 전진이면
           판이 밀려 화면상 제자리이고, 아니면 누른 칸이다. 실측 세 판
           353사이클에서 이 예측은 347번 맞았고, 어긋난 여섯 번 중 넷이
           **인식 쪽이 틀린 것**이었다(44.4초·233.2초는 한 행 위로 읽었고,
           207.2초·230.9초는 2열로 읽었다).
        """
        if here[1] > PLAYER_MAX_COL:
            return f"{here[1]}열은 플레이어가 설 수 없는 자리입니다"
        if self._expect is not None and here != self._expect:
            return f"직전 이동으로 보면 {self._expect} 에 있어야 합니다"
        return ""

    def _wait_broken(self, grid: Grid, cell: tuple[int, int]):
        """장애물을 누른 뒤, 그 칸에서 장애물이 사라질 때까지 본다.

        반환: (마지막으로 찍은 화면, 부서졌는지)

        칸 하나의 보라색 비율만 보므로 판 전체를 다시 인식하는 것(0.37초)보다
        훨씬 싸다. 인식 기준은 판 인식과 같은 값을 쓴다.

        왜 시간이 아니라 결과를 기다리나 (실측 176.3초)
            파괴는 성공했는데 2.2초 뒤 재인식에서도 그 칸이 장애물로 보였다.
            하단 부수기 개수도 56 그대로였다(저장된 화면을 다시 읽어 확인).
            그래서 같은 칸을 또 눌렀는데, 그때는 이미 빈칸이라 그 클릭이
            **이동**이 됐다 — 걸음수 1223 -> 1222, 부수기는 안 줄었다.
            2.3초 사이에 같은 자리를 두 번 두드린 셈이다.
        """
        rect = grid.cell_rect(cell[0], cell[1])
        deadline = time.time() + self.cfg.break_confirm_sec
        shot = None
        while True:
            self._check_stop()
            shot = self._capture()
            if shot is not None:
                frac = _frac(mask_obstacle(shot), rect)
                if frac < OBSTACLE_FRAC_WEAK:
                    self.log(f"[장애물] {cell} 가 부서진 것을 확인했습니다 "
                             f"(보라색 {frac:.2f}).")
                    return shot, True
                # 안내문이 떴다면 부술 수 없는 것이다. 더 기다릴 이유가 없다.
                if self._toast_visible(shot):
                    return shot, False
            if time.time() >= deadline:
                return shot, False
            self._sleep(self.cfg.poll_interval_sec)

    def _plan(self, scene: Scene):
        """경로를 세운다. **길이 있으면 아이템을 쓰지 않는다.**

        먼저 장애물을 벽으로 두고(부수기 없이) 계산한다. 그것만으로 나아갈 수
        있으면 그 경로를 쓴다. 아무 데도 못 갈 때만 장애물을 '부수기 1개짜리
        통행료가 붙은 칸'으로 보고 다시 계산한다.

        왜 이렇게 하나 (실측 299초, 부수기 5회를 전부 뜯어봤다)
            다섯 번 모두 **한 번만 전진하면 공짜로 먹을 수 있는 칩**이었다.
            판은 오른쪽으로 갈 때마다 왼쪽으로 한 열 밀린다(19장). 그러니
            (r,1) 을 막고 선 장애물은 다음 전진에 0열로 밀려나고, (r,2) 에
            있던 칩이 그 자리로 들어온다. 한 박자만 기다리면 걸어서 닿는다.

            예: 176.3초 판. 플레이어 (2,1), 칩 (4,2), (3,1) 이 장애물.
                    . X . . .        전진하면              . . . . ?
                    . . X . .        ->                    . X . . ?
                    . P X . .                              P X . . ?
                    X X . . O                              X . . O ?
                    . . G . .                              . G . . ?
                (3,1) 이 비고 칩이 (4,1) 로 온다. 걸어서 두 칸이면 먹는다.
                그런데 부수기를 써 버렸고, 2.3초 뒤 같은 자리를 또 부쉈다.

        그래도 정말 막힌 판에서는 부순다. 부수지 않으면 칩을 영영 못 먹는
        경우가 있고(298초에 20건), 그때는 아껴 봐야 소용이 없다.
        """
        def usable(plan, item_free: bool) -> bool:
            if plan.kind == PlanKind.NONE or len(plan.path) < 2:
                return False
            if not item_free:
                return True
            # **장애물을 밟는 길은 공짜가 아니다.** 부수기를 못 쓴다고 알려도
            # '갈 데가 아주 없을 때'의 마지막 수단으로 장애물 칸을 내주는
            # 자리가 있어서, 그걸 공짜 길로 착각하면 안 된다.
            return not any(scene.kind_at(r, c) == Kind.OBSTACLE
                           for r, c in plan.path[1:])

        free = plan_route(scene, self.cfg.item_max_detour, cost_break=None)
        if usable(free, True):
            return free
        cost = self._break_cost()
        if cost is None:
            return free                   # 부술 수단이 없다. 이게 최선이다
        paid = plan_route(scene, self.cfg.item_max_detour, cost_break=cost)
        if usable(paid, False):
            self.log("[경로] 부수지 않고는 갈 데가 없어 장애물을 지나는 길을 "
                     "다시 세웁니다.")
            return paid
        return free

    def _break_cost(self) -> float | None:
        """장애물 하나를 부수는 값을 걸음수 몇 개어치로 볼지.

        부수기를 쓸 수 없으면(설정으로 껐거나, 개수가 0이거나, 눌러도 안
        부서지는 것을 확인했으면) None. 그러면 경로 계산이 장애물을 벽으로 본다.
        """
        if self.break_disabled or not self.cfg.allow_obstacle_break:
            return None
        left = self.counts.break_
        if left is not None and left <= 0:
            return None                   # 0 이라고 **읽었을 때만** 막는다
        if left is None and self.cfg.watch_counters:
            # 개수를 못 읽었다. 그래도 장애물을 벽으로 보면 길을 통째로 잃는다.
            # 아껴 쓰는 쪽으로 값을 비싸게 매겨 꼭 필요할 때만 부수게 한다.
            return pathfind.BREAK_COST_RANGE[1]
        return pathfind_break_cost(self.counts.steps, left)

    # --------------------------------------------- 판을 가리는 팝업 닫기
    def _close_popup_if_any(self, img: np.ndarray) -> bool:
        """실패창/보상창이 떠 있으면 바깥을 눌러 닫는다. 닫았으면 True.

        이 팝업들은 던전에서만 뜨는 것이 아니라 어떤 기능을 돌리든 판이 끝나면
        올라오고, 그동안 아래 화면은 클릭을 먹지 않는다. 모르고 계속 누르면
        헛클릭만 쌓인다. 규칙은 던전에서 검증된 것을 그대로 쓴다(popup 참고).
        """
        got = popup.find(img)
        if got is None:
            return False
        kind, score, _box = got
        w, h = self.window.client_size()
        x, y = popup.close_point(w, h)
        self.log(f"[팝업] {popup.name_of(kind)}이(가) 떠 있습니다 ({score:.2f}). "
                 f"바깥({x},{y})을 눌러 닫습니다.")
        self._check_stop()
        self.window.click_client(x, y, self.cfg.move_duration)
        self._sleep(self.cfg.popup_settle_sec)
        return True

    # ------------------------------------------------- 칩 묶음 추적
    def _confirm_goals(self, scene: Scene) -> None:
        """검출된 칩을 그대로 쓰지 않고 **추적 중인 묶음**으로 갈아끼운다.

        묶음은 한 번 읽어 잠그고, 다 먹을 때까지 새 칩을 받지 않는다. 그래서
        칩 획득 이펙트가 아무리 그럴듯해도 끼어들 자리가 없다.
        자세한 근거와 규칙은 chiptrack.py 를 보라.
        """
        # 지난 사이클 이후 전진한 만큼 판이 밀렸다. 알고 있던 칩도 함께 옮긴다.
        self.chips.advanced(self._scrolls_since)
        self._scrolls_since = 0

        # 플레이어가 서 있는 칸의 칩은 이미 먹은 것이다.
        if scene.player is not None:
            self.chips.collected_at((scene.player.row, scene.player.col))

        detected = {(d.row, d.col) for d in scene.goals}

        # **움직이는 칸의 칩은 이펙트다.**
        #
        # 판이 가라앉은 지금 움직이는 것은 디지몬과 연출뿐이다. 판 위에 놓인
        # 칩은 가만히 있는다. 실측(0.12초 간격 161프레임): 잡힌 칩 자리 15건이
        # 전부 0.72초 안에 사라졌고 15건 모두 움직인 프레임이 있었다.
        # 디지몬이 선 칸은 늘 움직이므로 빼고 본다.
        player_cell = ((scene.player.row, scene.player.col)
                       if scene.player is not None else None)
        wiggling = {c for c in (self._moving_cells - {player_cell}) if c in detected}
        if wiggling:
            self._ghost_suspect += 1
            detected -= wiggling

        # 이번 화면에 이펙트가 없다는 것을 **확인했는가.**
        #   - 움직임 검사가 제대로 돌았고 (판이 스크롤 중이 아니었고)
        #   - 검출된 칩 중 움직이는 것이 하나도 없다
        # 둘 다 맞으면 두 프레임을 기다리지 않고 바로 잠근다.
        no_effect = self._motion_valid and not wiggling

        # **0열 칩만은 한 번 더 확인한다.**
        #
        # 틀렸을 때의 대가가 방향에 따라 다르다. 앞쪽(2열 이상) 칩이 가짜면
        # 어차피 전진하던 길이라 손해가 없다. 그런데 0열 칩은 **왼쪽으로
        # 되돌아가야** 하므로, 가짜였다면 걸음수를 두 번 버린다.
        # 사용자가 계속 보고한 증상이 정확히 '유령칩 때문에 좌로 이동'이었다.
        fresh_zero = {c for c in detected
                      if c[1] == 0 and c not in self.chips.chips
                      and c not in self._prev_zero}
        self._prev_zero = {c for c in detected if c[1] == 0}
        detected -= fresh_zero

        was_locked = self.chips.locked
        valid = self.chips.update(detected, trust_now=no_effect)
        if fresh_zero:
            scene.notes.append(
                f"0열 칩 {sorted(fresh_zero)} 은(는) 되돌아가야 하므로 "
                f"한 번 더 보고 정합니다.")

        if not was_locked and valid:
            self.log(f"[칩] 묶음을 새로 읽었습니다: {sorted(valid)} "
                     f"(다 먹을 때까지 새 칩은 보지 않습니다)")

        # 판에서 지울 칸은 **검출된 것 전부** 를 기준으로 따진다.
        # 움직여서 뺀 칩(wiggling)도 판에서 지워야 경로 계산이 안 쫓아간다.
        ignored = (detected | wiggling | fresh_zero) - valid
        scene.goals = [d for d in scene.goals if (d.row, d.col) in valid]
        # **scene.goal(가장 확실한 칩 하나) 도 함께 맞춘다.**
        #
        # 이걸 빼먹으면 걸러낸 칩이 뒷문으로 되살아난다. plan_route 에 "goals 가
        # 비었으면 goal 을 쓴다"는 호환용 갈래가 있던 때, 추적기가 이펙트로
        # 판정해 버린 칩을 그대로 쫓아갔다. 실측 300초에서 좌이동 5건 중 3건이
        # '계획=목적지인데 칩 목록은 비어 있음' 이었다.
        #
        # 그 갈래는 이제 없앴지만(진실이 두 곳에 있으면 한쪽만 걸러진다),
        # 걸러낸 칩이 Scene 어디에도 남지 않게 하는 것은 그대로 지킨다.
        # 오버레이·디버그가 아직 이 값을 읽고, 언제든 새 사용처가 생긴다.
        scene.goal = max(scene.goals, key=lambda d: d.confidence,
                         default=None) if scene.goals else None
        for r, c in ignored:
            if scene.cells[r][c] == Kind.GOAL:
                scene.cells[r][c] = Kind.EMPTY
        # 안 보이는 칩을 판에 살려 두지 않는다.
        #
        # 예전에는 '한 프레임 놓쳤다고 목표를 놓으면 다 온 칩을 버린다'며
        # 살려 뒀다. 그런데 그것이 유령의 원천이었다(chiptrack.MISS_LIMIT 참고).
        # 검출이 정확한 지금은 **보이는 것만** 목표로 삼는다.
        if wiggling:
            scene.notes.append(
                f"칩 {sorted(wiggling)} 은(는) 그림이 움직이고 있습니다. "
                f"획득 이펙트로 보고 무시합니다.")
        rest = ignored - wiggling
        if rest:
            scene.notes.append(
                f"추적 중인 묶음에 없는 칩 {sorted(rest)} 은(는) 획득 이펙트로 "
                f"보고 무시합니다.")

    # ------------------------------------------------- 아이템 개수 모니터링
    def _update_counts(self, img: np.ndarray) -> None:
        """왼쪽 아래 걸음수/부수기/돌진 개수를 읽어 둔다.

        못 읽어도 그냥 넘어간다(숫자 템플릿이 없는 경우). 그때는 개수가 None 이라
        아래 _can_use 가 '모르니 해 보자'로 답한다.
        """
        if not self.cfg.watch_counters:
            return
        try:
            new = counters.read(img)
        except Exception as e:                       # 읽기 실패가 매크로를 멈추면 안 된다
            self.log(f"[개수] 읽기 실패: {type(e).__name__}: {e}")
            self.cfg.watch_counters = False
            return
        self.counts = new
        line = new.describe()
        if line != self._last_counts_line:
            self.log(f"[개수] {line}")
            self._last_counts_line = line

    def _can_use(self, name: str) -> bool:
        """그 아이템을 쓸 수 있는가.

        **개수를 보기로 해 놓고 못 읽었으면 쓰지 않는다.**

        예전에는 '모르면 해 보자'였다. 그런데 개수 읽기가 막히면(실측: 돌진 줄
        옆에 충전 타이머가 뜨자 줄을 두 개만 찾아 세 항목이 전부 None 이 됐다)
        그 말은 **아무 제동 없이 계속 쓴다**는 뜻이 된다. 실제로 돌진 45개를
        다 태워 버렸다. 아껴야 하는 자원에서 '모름'은 '써도 된다'가 아니다.

        개수를 아예 안 보기로 했다면(watch_counters=False) 예전처럼 해 본다.
        그때는 못 읽는 것이 아니라 애초에 안 보는 것이다.
        """
        if not self.cfg.watch_counters:
            return True
        left = self.counts.get(name)
        if left is None:
            return False
        return left > 0

    def _out_of_steps(self) -> bool:
        return (self.cfg.stop_when_out_of_steps
                and self.counts.steps is not None and self.counts.steps <= 0)

    # --------------------------------------------- 초록 버튼(돌진으로 장애물 부수기)
    def _press_green_button(self) -> bool:
        """우측 하단 초록색 버튼을 눌러 장애물을 부순다.

        오른쪽으로 갈 길이 없을 때만 쓴다. 사용 횟수가 정해져 있으므로
        (실측: 30회) 함부로 누르면 안 된다.
        """
        if not self.cfg.use_green_button:
            return False
        if not self._can_use("dash"):
            self.log("[초록버튼] 돌진 아이템이 0개입니다. 누르지 않습니다.")
            return False
        if (self.cfg.green_button_max_uses
                and self.stats.green_button_uses >= self.cfg.green_button_max_uses):
            self.log(f"[초록버튼] 설정한 사용 한도"
                     f"({self.cfg.green_button_max_uses}회)에 도달했습니다.")
            return False

        img = self._capture()
        if img is None:
            return False
        pos = find_green_button(img, self.templates["green_button"])
        if pos is None:
            self.log("[초록버튼] 버튼을 찾지 못했습니다.")
            return False

        self._check_stop()
        screen = self.window.click_client(pos[0], pos[1], self.cfg.move_duration)
        if screen is None:
            return False
        self.stats.green_button_uses += 1
        # 이 버튼은 **돌진**이다(29장). 예전 로그는 '장애물 부수기'라고 적었는데,
        # 실제로 줄어드는 것은 돌진 아이템이라 기록만 보면 원인을 못 찾는다.
        self.log(f"[초록버튼] 돌진 (누적 {self.stats.green_button_uses}회, 남은 돌진 "
                 f"{self.counts.dash if self.counts.dash is not None else '?'}) "
                 f"| 클라이언트{pos} -> 화면{screen}")
        self._sleep(0.6)
        self._expect = None       # 세 칸 나아갔다. 어디인지 다시 봐야 안다
        return True

    # ------------------------------------------------- 가벼운 플레이어 추적
    def _track_player(self, img: np.ndarray, grid: Grid) -> tuple[int, int] | None:
        """전체 인식을 하지 않고 플레이어 칸만 가볍게 추적한다.

        강조칸 십자로 역산하는 쪽이 훨씬 싸지만, 십자 모양이 **제대로 갖춰졌을
        때만** 쓴다. 아니면 템플릿/색 기반 검출로 내려간다.
        (전에는 강조칸을 무조건 1순위로 썼다가, 십자가 아닌 밝은 칸 조합에서
         엉뚱한 칸을 내놓아 이동 확인이 계속 실패했다.)
        """
        hsv = hsv_of(img)
        highlights = []
        m_high = mask_highlight(img, hsv)
        for r in range(N):
            for c in range(N):
                if _frac(m_high, grid.cell_rect(r, c)) > 0.40:
                    highlights.append((r, c))
        return track_player_fast(img, grid, highlights, hsv, self._not_player)

    @staticmethod
    def _cell_signature(img: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """셀 25칸의 '내용 요약'과 비교에 써도 되는 칸인지 여부.

        반환: (5x5x2 요약[장애물 비율, 평균 밝기], 5x5 유효 마스크)

        이동 가능 강조칸은 **판과 함께 스크롤하지 않고 늘 플레이어 주위에**
        표시된다. 그대로 비교에 넣으면 '판이 밀렸는지'를 흐려 놓으므로 뺀다.
        """
        hsv = hsv_of(img)
        m_obst = mask_obstacle(img, hsv)
        m_high = mask_highlight(img, hsv)
        v = hsv[:, :, 2]
        sig = np.zeros((N, N, 2), np.float32)
        valid = np.ones((N, N), bool)
        for r in range(N):
            for c in range(N):
                rect = grid.cell_rect(r, c)
                sig[r, c, 0] = _frac(m_obst, rect)
                x0, y0, x1, y1 = rect
                iy, ix = int((y1 - y0) * 0.2), int((x1 - x0) * 0.2)
                patch = v[y0 + iy:y1 - iy, x0 + ix:x1 - ix]
                sig[r, c, 1] = float(patch.mean()) / 255.0 if patch.size else 0.0
                if _frac(m_high, rect) > 0.40:
                    valid[r, c] = False
        return sig, valid

    @staticmethod
    def _shift_score(a: np.ndarray, b: np.ndarray, dr: int, dc: int,
                     va: np.ndarray | None = None,
                     vb: np.ndarray | None = None) -> float:
        """b 가 a 를 (dr, dc) 칸만큼 민 것과 얼마나 맞는지. 작을수록 잘 맞는다."""
        diffs = []
        for r in range(max(0, -dr), N - max(0, dr)):
            for c in range(max(0, -dc), N - max(0, dc)):
                if va is not None and not va[r + dr, c + dc]:
                    continue
                if vb is not None and not vb[r, c]:
                    continue
                diffs.append(float(np.abs(b[r, c] - a[r + dr, c + dc]).mean()))
        return float(np.mean(diffs)) if diffs else 1.0

    def _scrolled_one_cell(self, before: np.ndarray, after: np.ndarray,
                           grid: Grid, direction: str,
                           before_sig: tuple[np.ndarray, np.ndarray] | None = None) -> bool:
        """게임판이 이동 방향의 반대로 정확히 한 칸 스크롤했는지 판정한다.

        플레이어가 진행하면 게임판이 통째로 밀리면서 플레이어가 화면상 같은 칸에
        남는다(실측: RIGHT 이동 시 판이 왼쪽으로 한 열 밀림). 이때 '안 움직였다'고
        오판하면 안 된다.

        처음에는 위상 상관(phaseCorrelate)으로 픽셀 이동량을 쟀는데, 게임판이
        108px 주기로 반복되는 격자라 상관 피크가 주기 구조에 걸려 0 으로 잡히는
        일이 있었다. 그래서 픽셀 대신 **셀 내용이 한 칸 밀렸는지**를 본다.
        """
        delta = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        if direction not in delta:
            return False
        dr, dc = delta[direction]
        # 여기서 예외를 통째로 삼키면 안 된다. 예전에 import 누락으로 이 함수가
        # 항상 False 를 돌려주는데도 조용히 넘어가서 이동이 전부 실패로 잡혔다.
        # before 는 이동 내내 같은 사진이므로 폴링마다 다시 계산하지 않는다.
        a, va = before_sig if before_sig is not None else self._cell_signature(before, grid)
        b, vb = self._cell_signature(after, grid)
        moved = self._shift_score(a, b, dr, dc, va, vb)
        same = self._shift_score(a, b, 0, 0, va, vb)
        # 클릭이 먹었는지 판단하는 데도 쓰려고 남겨 둔다(_do_move 참고).
        self._last_same = same
        self._last_moved = moved
        # '한 칸 밀림'이 '그대로'보다 뚜렷하게 잘 맞아야 인정한다.
        return moved < 0.06 and moved < same * 0.6

    # ------------------------------------------------------------- 한 칸 이동
    def _do_move(self, grid: Grid, frm: tuple[int, int], to: tuple[int, int],
                 direction: str) -> tuple[bool, Grid, bool]:
        """플레이어 기준으로 다음 한 칸의 셀 중심을 클릭하고 실제 도착을 확인한다.

        반환: (성공 여부, 갱신된 격자, 게임판이 스크롤했는지)

        스크롤했다면 플레이어는 **화면상 같은 칸에 그대로** 있고 판 전체가 밀린
        것이므로, 미리 계산해 둔 나머지 경로의 칸 번호는 더 이상 맞지 않는다.
        그때는 고속 경로를 접고 전체 재인식을 해야 한다.
        """
        self._check_stop()
        self._path_dirty = False
        before = self._capture()
        if before is None:
            return False, grid, False

        # 매 이동 전에 격자선만 가볍게 다시 검출해서 셀 중심 좌표를 갱신한다.
        fresh = self._stable_grid(before)
        if fresh is not None:
            grid = fresh

        cx, cy = grid.cell_center(to[0], to[1])
        self._check_stop()
        clicked_at = time.time()
        screen = self.window.click_client(cx, cy, self.cfg.move_duration)
        if screen is None:
            self.log(f"[클릭] 실패: 창이 유효하지 않거나 좌표가 범위를 벗어났습니다 ({cx},{cy})")
            return False, grid, False
        self.log(f"[클릭] {direction}: {frm} -> {to} | 클라이언트({cx},{cy}) "
                 f"-> 화면({screen[0]},{screen[1]})")

        time.sleep(self.cfg.click_settle_sec)

        confirmed = 0
        deadline = time.time() + self.cfg.move_timeout_sec
        # 클릭 직전 사진의 셀 요약. 스크롤 판정에 쓰는데 매 폴링마다 똑같이 다시
        # 계산하고 있었다(실측 5.2ms). 격자가 바뀌면 그때만 다시 만든다.
        before_sig = None
        before_sig_grid = None

        # 걸음수는 이동에 성공할 때마다 정확히 1 줄어든다.
        # 다만 화면에 반영되기까지 0.53~2.10초가 걸리는데(실측 22건) 이동
        # 하나가 1.85초라, **직전 이동의 감소를 이번 이동의 성공으로 착각**할 수
        # 있다. 실제로 그랬다: 움직이지도 않았는데 성공으로 치고 (2,1) 과 (2,0)
        # 을 일곱 번 오갔다.
        #
        # 그래서 '클릭 직전 값'과 견주지 않고 **지금까지 본 최솟값**과 견준다.
        # 걸음수는 줄기만 하므로, 화면이 늦게 갱신될 때 보이는 값은 언제나 진짜
        # 값보다 크거나 같다. 그러니 최솟값보다 **더 작은** 값이 나왔다면 그건
        # 새로 줄어든 것, 곧 이번 이동이 성공한 것이다.
        if self.cfg.watch_counters:
            self._note_steps(counters.read(before))

        polls = 0
        still = 0
        retries = 0          # 화면이 클릭 전과 똑같은 채로 몇 번 연속인가
        # 확인 루프가 찍는 프레임을 (시각, 화면) 으로 조금 남긴다.
        # 도착 확인에 **움직임**을 쓰기 위해서다. 아래 _arrived_by_motion 참고.
        #
        # 몇 장이면 되는지는 정해져 있다. _arrived_by_motion 은 지금 화면과
        # motion_gap_sec 만큼 떨어진 **한 장**만 있으면 되므로, 그 간격을 덮을
        # 만큼만 들고 있으면 결과가 똑같다. 예전에는 60장이었는데, 709x1260
        # 프레임 한 장이 2.7MB 라 이동 한 번에 160MB 를 쥐고 있는 셈이었다.
        span = max(self.cfg.poll_interval_sec, 0.01)
        keep = min(24, max(4, int(self.cfg.motion_gap_sec / span) + 4))
        seen_frames: deque = deque(maxlen=keep)
        while time.time() < deadline:
            self._check_stop()
            after = self._capture()
            if after is None:
                time.sleep(self.cfg.poll_interval_sec)
                continue

            polls += 1
            if polls % self.cfg.toast_check_every == 0 and self._toast_visible(after):
                self._wait_toast_clear()
                return False, grid, False

            seen_frames.append((time.time(), after))

            light = self._stable_grid(after)
            use_grid = light or grid
            pos = self._track_player(after, use_grid)

            ok = False
            scrolled = False
            # **한 번만 뜨는 신호**인가. 걸음수 감소가 그렇다 — 최솟값을 갱신하고
            # 나면 다음 폴링부터는 다시 뜨지 않는다. 그런 신호는 confirm_repeat
            # (연속 확인 횟수) 를 적용하면 안 된다.
            #
            # 실측 회귀(299초): 실패로 적힌 이동 12건이 전부 이 경우였다.
            # 폴링 기록을 보면 #2 에서 ok=True 가 떴는데 #3 이 False 라
            # confirmed 가 0 으로 되돌아갔고, 그 뒤로는 걸음수가 다시 줄 리
            # 없으니 영영 확인되지 않았다. 그러고는 '클릭이 안 먹었다'며 같은
            # 칸을 다시 눌렀는데, 거기엔 이미 플레이어가 서 있어서 안내문이
            # 떴다. 안내문 12번 = 실패 12번으로 정확히 맞아떨어진다.
            conclusive = False
            if pos == to:
                ok = True
            elif self._steps_dropped(after):
                # **걸음수가 줄었으면 이동은 일어난 것이다.**
                #
                # 전진(오른쪽)은 판이 스크롤해서 플레이어가 화면상 같은 칸에
                # 남으므로 '목표 칸 도착'으로도 '움직인 칸'으로도 확인되지 않는다.
                # 실측: 실패한 이동 13건 중 11건이 오른쪽이었고 전부 2.5초
                # 제한을 다 썼다. 걸음수는 이동당 정확히 1 줄고 0.53~2.10초
                # 안에 화면에 반영된다(실측 22건, 전부 1씩).
                #
                # 무엇이 어떻게 바뀌었는지는 모르므로 남은 경로는 버린다.
                # 그래야 다음 클릭이 엉뚱한 칸으로 가지 않는다.
                #
                # **다만 '스크롤했다'로 치면 안 된다.** 판이 밀리는 것은
                # 1열에서 오른쪽을 눌렀을 때뿐이다(19장). 세로 이동까지 스크롤로
                # 세면 칩 추적기가 칩 자리를 한 열씩 더 밀어 버리고, 그러면
                # 없는 칩이 플레이어 자리로 밀려와 '먹었다'고 처리된다.
                #
                # 실측: '칩을 먹었다'고 판단한 10건 중 5건이 헛것이었고(상단
                # 보유량이 2.5초 동안 한 픽셀도 안 바뀌었다) 그 칸은 전부
                # 플레이어가 선 자리였다.
                ok = True
                conclusive = True
                scrolled = advances(direction, frm)
                self._path_dirty = True
            elif self._arrived_by_motion(seen_frames, use_grid, to):
                # 색 기반 빠른 추적은 가끔 말이 안 되는 칸을 낸다.
                # 실측: 실패한 이동 12건에서 (3,1)->(2,0), (4,1)->(4,4) 처럼
                # 순간이동하는 값을 봤고, 그 12건이 전부 2.5초 제한을 다 썼다
                # (150초 중 30초). 움직임은 그런 헛값을 내지 않는다.
                ok = True
            elif pos == frm:
                # 이전 칸에 남은 애니메이션 잔상이거나 아직 안 움직였다.
                # 어느 쪽이든 성공으로 처리하지 않는다.
                ok = False
            elif pos is None:
                ok = False

            if not ok:
                self._last_same = None
                self._last_moved = None
                # 게임판이 통째로 스크롤해서 플레이어가 화면상 같은 칸에
                # 남아 있는 경우를 확인한다.
                #
                # 예전에는 `pos in (frm, None)` 일 때만 봤다. 그런데 전진은
                # 스크롤 때문에 플레이어가 화면상 제자리에 남으므로 이 검사가
                # **전진을 확인하는 유일한 수단**이고, 빠른 추적이 헛값을 하나
                # 내면 그대로 건너뛰어 버렸다. 실측 298초에서 실패로 기록된
                # 이동 18건 중 13건이 사실은 성공한 이동이었고, 그 13건이
                # 제한시간 2.0~3.5초를 전부 소진했다(성공은 0.59초).
                # 그래서 못 맞힌 폴링에서는 **언제나** 스크롤을 확인한다.
                key = (use_grid.xs[0], use_grid.ys[0], use_grid.xs[-1], use_grid.ys[-1])
                if before_sig is None or before_sig_grid != key:
                    before_sig = self._cell_signature(before, use_grid)
                    before_sig_grid = key
                if self._scrolled_one_cell(before, after, use_grid, direction,
                                           before_sig):
                    ok = True
                    # **판이 밀렸다고 세는 조건은 한 가지뿐이다**(advances 참고).
                    # 화면 비교가 '한 칸 밀렸다'고 해도 1열에서 오른쪽을 누른
                    # 것이 아니면 둘 중 하나가 틀린 것이다. 그때 스크롤로 세면
                    # 칩 추적기가 칩 자리를 한 열 더 밀어 유령칩을 만든다.
                    # 움직인 것은 맞으니 성공으로 두되, 판이 어떻게 바뀌었는지
                    # 모르는 것으로 하고 남은 경로만 버린다.
                    scrolled = advances(direction, frm)
                    if not scrolled:
                        self._path_dirty = True

            # 이 폴링에서 무엇이 보였는지 통째로 남긴다. 확인이 왜 실패했는지는
            # 눈으로 볼 수 없어서, 남기지 않으면 매번 추측으로 고치게 된다.
            self.trace.write("poll", n=polls, dir=direction,
                             **{"from": list(frm)}, to=list(to),
                             pos=list(pos) if pos else None,
                             same=(None if self._last_same is None
                                   else round(self._last_same, 4)),
                             moved=(None if self._last_moved is None
                                    else round(self._last_moved, 4)),
                             ok=bool(ok), scrolled=bool(scrolled),
                             since=round(time.time() - clicked_at, 2))

            # **판이 그대로면 클릭이 먹지 않은 것이다.**
            #
            # 실측: 실패한 전진에서 '한 칸 밀림' 점수는 0.09~0.25 인데 '그대로'
            # 점수가 0.001~0.006 이었다. 판이 밀린 것도 아니고 아무것도 바뀌지
            # 않았다는 뜻이다. 성공한 전진은 '그대로' 점수가 0.07~0.17 이다.
            # 원본 픽셀로 보면 안 된다 — 디지몬 애니메이션 때문에 늘 조금씩
            # 다르다. 셀 요약(장애물 비율·평균 밝기)으로 봐야 갈린다.
            if (not ok and self._last_same is not None
                    and time.time() - clicked_at >= self.cfg.dead_click_wait_sec):
                if self._last_same < DEAD_CLICK_SAME:
                    still += 1
                    if still >= self.cfg.dead_click_polls:
                        # **먹지 않은 클릭은 그 자리에서 다시 누른다.**
                        #
                        # 아무 일도 일어나지 않았으므로 두 번 움직일 위험이 없다.
                        # 반면 판을 통째로 다시 인식하는 데는 인식 0.37초 +
                        # 움직임 촬영 0.45초가 든다. 다시 누르는 편이 훨씬 싸다.
                        #
                        # 실측: 목표가 강조칸(게임이 갈 수 있다고 표시한 칸)이고
                        # 바로 옆인데도 안 먹는 경우가 있었다. 논리가 아니라
                        # 클릭이 전달되지 않은 것이다.
                        # 다시 누르기 전에 안내문을 반드시 확인한다.
                        # 안내문이 떴다면 그 칸은 막힌 것이라, 또 누르면 안내문만
                        # 한 번 더 띄우고 2초를 더 기다리게 된다(실측: 다시 누른
                        # 12건 중 9건이 직후에 안내문이었다).
                        if self._toast_visible(after):
                            self._wait_toast_clear()
                            return False, grid, False
                        if retries < self.cfg.dead_click_retries:
                            retries += 1
                            still = 0
                            self.log(f"[클릭] 먹지 않았습니다. 다시 누릅니다 "
                                     f"({retries}/{self.cfg.dead_click_retries}) "
                                     f"{direction} {frm}->{to}")
                            self._check_stop()
                            self.window.click_client(cx, cy, self.cfg.move_duration)
                            clicked_at = time.time()
                            deadline = clicked_at + self.cfg.move_timeout_sec
                            time.sleep(self.cfg.click_settle_sec)
                            continue
                        self.log(f"[클릭] 판이 그대로입니다. 이 클릭은 먹지 "
                                 f"않았습니다 ({direction} {frm}->{to}).")
                        return False, grid, False
                else:
                    still = 0

            if ok:
                confirmed += 1
                if conclusive or confirmed >= self.cfg.confirm_repeat:
                    self.stats.moves += 1
                    return True, use_grid, scrolled
            else:
                confirmed = 0
            grid = use_grid
            time.sleep(self.cfg.poll_interval_sec)

        self.stats.failed_moves += 1
        self.log(f"[이동] 확인 실패: {frm} -> {to} ({direction}). 전체 재인식합니다.")
        if self.cfg.save_debug and self.last_frame is not None:
            self._save_debug("player_track_fail", self.last_frame, grid)
        return False, grid, False

    # ------------------------------------------------------------- 디버그
    def _save_debug(self, name: str, img: np.ndarray, grid: Grid | None = None,
                    scene: Scene | None = None, path=None, header: str = "") -> None:
        if not self.cfg.save_debug:
            return
        os.makedirs(DEBUG_DIR, exist_ok=True)
        overlay.save(os.path.join(DEBUG_DIR, f"{name}_raw.png"), img)
        if grid is not None:
            drawn = overlay.draw(img, grid, scene, path, header)
            overlay.save(os.path.join(DEBUG_DIR, f"{name}_overlay.png"), drawn)

    # --------------------------------------------------------------- 메인
    def run(self) -> None:
        # 여기서 stop_event 를 지우지 않는다. 지우면 '시작 전에 이미 눌린 정지'가
        # 없던 일이 돼 버린다. 새로 시작할 때는 엔진 인스턴스를 새로 만든다.
        self.stats = ExploreStats()

        # DPI 인식은 프로세스마다 한 번만 켜면 되고 여러 번 불러도 문제없다.
        # 부르는 쪽(launcher/gui)이 이미 켰겠지만 여기서도 켠다.
        #
        # 안 켜면 창 크기가 논리 픽셀로 보고돼 모든 좌표가 어긋난다
        # (실측: 709x1260 창이 567x1008 = 정확히 80% 로 보고됨). 그러면 격자를
        # 반쪽에서 찾고, 클릭이 엉뚱한 칸으로 가고, 없는 칩이 보이고, 좌우로
        # 왔다갔다 하다가 이동에 계속 실패한다. 그런데 **화면은 멀쩡해 보여서**
        # 원인을 찾기가 매우 어렵다. 실제로 이걸로 한참 헤맸다.
        enable_dpi_awareness()

        loaded = {k: len(v.images) for k, v in self.templates.items() if v.images}
        self.log(f"[템플릿] {loaded if loaded else '없음 (색 기반 인식으로 동작)'}")

        if not self.attach_window():
            return

        lost = 0
        try:
            while True:
                self._check_stop()

                # --- 1) 전체 화면 인식 --------------------------------
                img = self._capture()
                if img is None:
                    self.log("[캡처] 실패. 창이 닫혔을 수 있습니다.")
                    break

                if self._toast_visible(img):
                    self._wait_toast_clear()
                    continue

                # 남은 아이템 개수를 사이클마다 확인한다. 걸음수가 다 떨어졌으면
                # 더 움직일 수 없으므로 계속 클릭해 봐야 실패만 쌓인다.
                self._update_counts(img)
                if self._out_of_steps():
                    self.log("[개수] 걸음수를 다 썼습니다. 매크로를 멈춥니다.")
                    self.status("걸음수 소진")
                    break

                # 팝업이 떠 있으면 아래 화면은 클릭을 먹지 않는다. 먼저 닫는다.
                if self._close_popup_if_any(img):
                    continue

                grid = self._stable_grid(img, seed=True)
                if grid is None:
                    lost += 1
                    if lost == 1 or lost % self.cfg.max_lost_before_report == 0:
                        self.log(f"[인식] 5x5 게임판을 찾지 못했습니다 ({lost}회). "
                                 f"탐사 화면이 맞는지 확인하세요.")
                        self._save_debug("board_lost", img)
                    self._sleep(self.cfg.lost_retry_sec)
                    continue

                # 플레이어는 **움직임**으로 찾는 것이 가장 확실하다. 짧은 간격으로
                # 몇 장 더 찍어 어느 칸이 움직였는지 본다. 디지몬은 서 있을 때도
                # 제자리 애니메이션이 돌아가는 판 위의 유일한 움직이는 물체다.
                motion = self._motion_cell(img, grid)
                if self._board_animating:
                    # 판이 아직 스크롤 중이다. 이 프레임으로 계획을 세우면 안 된다.
                    # 실측: 애니메이션 중 프레임에서 칩이 7개로 잡혔고, 매크로가
                    # 그 유령 칩을 먹으러 왼쪽으로 갔다(150초에 되돌림 11회).
                    self._sleep(self.cfg.motion_gap_sec)
                    continue

                scene = analyze(img, grid, self.templates,
                                self.cfg.orange_goal_without_template,
                                motion_cell=motion)
                # 칩 획득 이펙트로 흩어진 칩을 목표로 삼지 않는다.
                self._ghost_suspect = 0
                self._confirm_goals(scene)
                if self._ghost_suspect and not self.chips.locked:
                    # 이펙트가 뜬 화면에서 **묶음을 새로 읽으려던 참**이다.
                    # 이때 읽으면 이펙트를 진짜 칩으로 잠가 버린다. 잠근 뒤로는
                    # 검출로 고칠 수도 없으니, 가라앉을 때까지 기다렸다 다시 본다.
                    self.log("[칩] 획득 이펙트가 보입니다. 화면이 가라앉은 뒤 "
                             "다시 읽습니다.")
                    self._sleep(self.cfg.ghost_settle_sec)
                    continue
                # 빠른 추적이 장애물/칩/아이템을 플레이어로 잡지 않게 넘겨준다.
                self._not_player = (
                    {(r, c) for r in range(N) for c in range(N)
                     if scene.cells[r][c] == Kind.OBSTACLE}
                    | {(d.row, d.col) for d in scene.goals}
                    | set(scene.item_kinds))

                # --- 2) 전체 경로 계산 --------------------------------
                plan = self._plan(scene)

                # 분석이 끝난 지금 다시 정지를 확인한다. 분석 도중에 정지를 눌렀다면
                # 여기서 빠져나가므로 이전 경로가 뒤늦게 실행되지 않는다.
                self._check_stop()

                self.trace.write(
                    "cycle", n=self.stats.cycles + 1,
                    player=[scene.player.row, scene.player.col] if scene.player else None,
                    player_note=scene.player.note if scene.player else "",
                    board=trace.board_rows(scene.cells),
                    chips=[[d.row, d.col] for d in scene.goals],
                    items={f"{r},{c}": k for (r, c), k in scene.item_kinds.items()},
                    highlights=[list(h) for h in scene.highlights],
                    counts={"steps": self.counts.steps, "break": self.counts.break_,
                            "dash": self.counts.dash},
                    break_cost=self._break_cost(),
                    plan=plan.kind.value, path=[list(c) for c in plan.path],
                    reason=plan.reason, notes=list(scene.notes))
                self.trace.frame(self.stats.cycles + 1, img)

                drawn = overlay.draw(img, grid, scene, plan.path,
                                     f"{plan.kind.value} | {' '.join(plan.moves)}")
                self.preview(drawn)
                self._save_debug("last", img, grid, scene, plan.path, plan.kind.value)

                if scene.player is None:
                    lost += 1
                    self.log("[인식] 플레이어를 찾지 못했습니다.")
                    self._save_debug("player_lost", img, grid, scene)
                    self._sleep(self.cfg.lost_retry_sec)
                    continue

                here = (scene.player.row, scene.player.col)
                # **말이 되는 자리인가.** 두 가지로 본다.
                #
                # 1) 플레이어는 0~1열에만 있다(19장). 2열로 잡혔다면 판이 밀리는
                #    도중을 찍은 것이다. 예전에는 그 자리를 곧이곧대로 믿고
                #    **왼쪽으로 되돌리는 이동**을 계획했다. 게임은 당연히
                #    거부했고, 실측 209.4초·233.2초에 2.1초씩 버렸다.
                #
                # 2) 직전 이동이 성공했다면 지금 어디 있어야 하는지 정확히 안다.
                #    실측 세 판(총 353사이클)에서 이 예측은 347번 맞았고,
                #    어긋난 여섯 번 중 넷이 **인식 쪽이 틀린 것**이었다
                #    (44.4초·233.2초는 한 행 위로, 207.2초·230.9초는 2열로).
                #
                # 한 번 더 보고도 같으면 그때는 인식을 믿는다. 게임이 우리가
                # 모르는 이유로 플레이어를 옮겼을 수도 있기 때문이다.
                odd = self._odd_position(here)
                if odd and self._recheck != here:
                    self._recheck = here
                    lost += 1
                    # 이 사이클은 **버린 것**이다. 바로 위에서 남긴 cycle 기록을
                    # 나중에 분석할 때 진짜 판단으로 세면 안 되므로 표시해 둔다.
                    self.trace.write("reject", at=list(here), why=odd)
                    self.log(f"[인식] 플레이어를 {here} 로 읽었는데 {odd}. "
                             f"다시 봅니다.")
                    self._sleep(self.cfg.lost_retry_sec)
                    continue
                if odd:
                    self.log(f"[인식] 두 번 봐도 {here} 입니다. 그대로 받아들입니다.")
                    self._expect = None
                self._recheck = None

                lost = 0
                self.stats.cycles += 1
                self.log(f"[인식] 판:\n{scene.summary()}")
                for note in scene.notes:
                    self.log(f"[인식] {note}")
                self.log(f"[경로] {plan.describe()}")

                if plan.kind == PlanKind.NONE or len(plan.path) < 2:
                    self.log("[경로] 진행할 경로가 없습니다. 잠시 뒤 다시 봅니다.")
                    self._sleep(self.cfg.lost_retry_sec)
                    continue

                # --- 갇혔는지 보고, 갇혔으면 부수거나 기다린다 --------
                # --- 돌진: 칩이 없으면 공짜로 세 칸 나아간다 ----------
                if self._dash_if_worth(scene):
                    continue

                if self._handle_blocked(scene, plan):
                    continue

                # --- 3~7) 한 칸씩 클릭하며 실제 도착을 확인 ------------
                current = plan.path[0]
                for i, nxt in enumerate(plan.path[1:]):
                    self._check_stop()
                    direction = plan.moves[i] if i < len(plan.moves) else "CLICK"

                    if scene.kind_at(nxt[0], nxt[1]) == Kind.OBSTACLE:
                        # **다음 칸이 장애물이면 이동이 아니라 파괴 클릭이다.**
                        #
                        # 실험으로 확인한 규칙(25장): 인접한 장애물을 클릭하면
                        # 부서진다. 플레이어는 들어가지 않고 걸음수도 들지 않으며,
                        # 부수기 아이템이 1 줄어든다.
                        #
                        # 그래서 경로 한가운데에 장애물이 있어도 괜찮다. 부수고
                        # 나면 판이 바뀌므로 남은 경로를 버리고 다시 본다.
                        cx, cy = grid.cell_center(nxt[0], nxt[1])
                        self._check_stop()
                        screen = self.window.click_client(cx, cy, self.cfg.move_duration)
                        self.stats.obstacles_broken += 1
                        self.trace.write("break", at=list(nxt),
                                         left=self.counts.break_)
                        self.log(f"[클릭] 장애물 파괴 시도 {nxt} | 클라이언트({cx},{cy}) -> "
                                 f"화면{screen}")
                        # **부서진 것을 눈으로 확인할 때까지 기다린다.**
                        #
                        # 예전에는 정해진 시간만 자고 끝냈다. 그런데 이 게임은
                        # 화면 반영이 느리다. 실측 176.3초: 파괴는 성공했는데
                        # 2.2초 뒤 재인식에서도 그 칸이 여전히 장애물로 보였고
                        # (하단 개수도 56 그대로였다), 그래서 같은 칸을 또 눌렀다.
                        # 그때는 이미 빈칸이라 그 클릭이 **이동**이 돼 버렸다
                        # (걸음수 1223 -> 1222, 부수기는 그대로).
                        #
                        # 그러니 시간이 아니라 **결과**를 기다린다. 이동에 이미
                        # 쓰고 있는 방식 그대로다(24장).
                        shot, broke = self._wait_broken(grid, nxt)
                        # 안내문이 떴다면 이 장애물은 부술 수 없는 것이다.
                        if not broke and shot is not None and self._toast_visible(shot):
                            self.break_fail_streak += 1
                            self.log(f"[장애물] 파괴가 통하지 않았습니다 "
                                     f"({self.break_fail_streak}/"
                                     f"{self.cfg.obstacle_break_max_failures}회 연속)")
                            # 판정을 먼저 확정한다. 안내문이 사라지길 기다리는
                            # 동안 정지가 들어오면 아래 코드가 실행되지 않아,
                            # 애써 쌓은 실패 횟수가 없던 일이 되기 때문이다.
                            if (self.break_fail_streak
                                    >= self.cfg.obstacle_break_max_failures):
                                self.break_disabled = True
                                self.log("[장애물] 이 게임에서는 장애물을 클릭해도 "
                                         "부서지지 않는 것으로 판단해 파괴 시도를 "
                                         "중단합니다. 이후에는 길이 열릴 때까지 "
                                         "기다립니다.")
                            self._wait_toast_clear()
                        # 여기서 기억해 둔 배치(_last_layout)를 지우고 싶어지는데,
                        # **지우면 안 된다.** 그 값은 매 사이클 새로 인식한 판에서
                        # 다시 계산하므로, 정말 부서졌다면 저절로 달라진다. 반대로
                        # 안내문 없이 조용히 안 부서지는 경우에 지워 버리면 갇힌
                        # 것을 영영 알아채지 못하고 같은 자리만 계속 두드린다.
                        break

                    _t0 = time.time()
                    ok, grid, scrolled = self._do_move(grid, current, nxt, direction)
                    self.trace.write("move", dir=direction, **{"from": list(current)},
                                     to=list(nxt), ok=bool(ok), scrolled=bool(scrolled),
                                     secs=round(time.time() - _t0, 2))
                    if not ok:
                        self._expect = None       # 어디 있는지 모른다
                        break     # 고속 경로 폐기 -> 바깥 루프에서 전체 재인식
                    current = nxt
                    # **이동에 성공했으니 플레이어가 어디 있는지 정확히 안다.**
                    # 전진(스크롤)이면 판이 밀려 화면상 제자리에 남는다(19장).
                    self._expect = (current[0], PLAYER_MAX_COL) if scrolled else nxt
                    self.break_fail_streak = 0   # 길이 열렸다면 다시 시도해 볼 만하다
                    if self._path_dirty and not scrolled:
                        # 이동은 했는데 무슨 일인지 확실치 않다. 남은 경로만 버린다.
                        self.log("[경로] 이동은 확인했지만 판이 어떻게 바뀌었는지 "
                                 "확실치 않아 다시 인식합니다.")
                        break
                    if scrolled:
                        self._scrolls_since += 1
                        # 판이 통째로 밀렸다. 남은 경로의 칸 번호는 이제 딴 곳을
                        # 가리키므로 그대로 클릭하면 '이동할 수 없습니다'가 뜬다.
                        self.log("[경로] 게임판이 스크롤해서 남은 경로를 버리고 "
                                 "다시 인식합니다.")
                        break

                # --- 8) 경로가 끝나면 다시 전체 화면 인식 -------------
                self._sleep(self.cfg.cycle_pause_sec)

        except Stopped:
            self.log("[정지] 요청을 받아 즉시 중단했습니다. 대기 중이던 클릭도 취소합니다.")
        except Exception as e:
            self.log(f"[오류] {type(e).__name__}: {e}")
            if self.last_frame is not None:
                self._save_debug("error", self.last_frame)
            raise
        finally:
            s = self.stats
            self.log(f"[종료] 사이클 {s.cycles} / 이동 {s.moves} / 실패 {s.failed_moves} "
                     f"/ 안내 {s.blocked_toasts} / 장애물 클릭 {s.obstacles_broken} "
                     f"/ 초록버튼 {s.green_button_uses}")
            self.trace.close()
            self.status("정지됨")
