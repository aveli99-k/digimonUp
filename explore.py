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
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

import overlay
from board import Grid, N, detect_board
from emulator_window import EmulatorWindow, capture_client, enumerate_candidates
from pathfind import PlanKind, plan_route
from recognize import (Kind, Scene, TemplateSet, analyze, find_blocked_toast,
                       find_green_button, find_top_tab, hsv_of, load_templates,
                       mask_highlight, mask_obstacle, track_player_fast, _frac)

from paths import DEBUG_DIR


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

    # 이동
    click_settle_sec: float = 0.12     # 클릭 직후 최소 대기
    move_timeout_sec: float = 2.2      # 한 칸 이동 확인 최대 대기
    poll_interval_sec: float = 0.06    # 이동 확인 폴링 주기
    confirm_repeat: int = 2            # 예상 칸을 몇 번 연속 확인해야 성공으로 볼지
    move_duration: float = 0.05        # 마우스 이동 시간

    # 사이클
    cycle_pause_sec: float = 0.25      # 전체 재인식 사이의 쉬는 시간
    lost_retry_sec: float = 0.8        # 인식 실패 시 재시도 간격
    max_lost_before_report: int = 5

    # 이동 불가 안내
    toast_min: float = 0.65
    toast_clear_repeat: int = 2

    # 목적지 = 주황칩(필수 아이템)
    # 탐사에는 종착점이 없지만, 판에 나오는 주황칩은 반드시 먹어야 한다.
    # 그래서 주황 카드를 목적지로 취급해 1순위로 가져간다.
    orange_goal_without_template: bool = True

    # 장애물 파괴
    # 오른쪽이 막히면 장애물을 직접 클릭하거나, 우측 하단 초록색 버튼을 눌러
    # 부술 수 있다. 초록 버튼은 사용 횟수가 정해져 있으므로(실측: 30회)
    # 먼저 장애물 클릭을 시도하고, 그게 안 먹힐 때만 버튼을 쓴다.
    allow_obstacle_break: bool = True
    obstacle_break_max_failures: int = 2
    use_green_button: bool = True
    green_button_max_uses: int = 0      # 0 = 제한 없음
    blocked_wait_sec: float = 2.0

    # 디버그
    save_debug: bool = True


class Stopped(Exception):
    """정지 요청. 진행 중인 모든 동작을 즉시 접는다."""


@dataclass
class ExploreStats:
    cycles: int = 0
    moves: int = 0
    failed_moves: int = 0
    blocked_toasts: int = 0
    obstacles_broken: int = 0
    green_button_uses: int = 0


class ExploreEngine:
    """탐사 자동화 엔진.

    GUI 든 콘솔이든 콜백만 갈아끼우면 그대로 쓸 수 있게 분리했다.
    """

    def __init__(self, cfg: ExploreConfig | None = None,
                 log=print, status=lambda s: None, preview=lambda img: None):
        self.cfg = cfg or ExploreConfig()
        self.log = log
        self.status = status
        self.preview = preview
        self.stop_event = threading.Event()
        self.window: EmulatorWindow | None = None
        self.templates: dict[str, TemplateSet] = load_templates()
        self.stats = ExploreStats()
        self.last_frame: np.ndarray | None = None
        self.last_overlay: np.ndarray | None = None
        self.candidates_report: list[str] = []
        # 장애물 파괴 연속 실패 횟수. 한도를 넘으면 파괴 시도를 접는다.
        self.break_fail_streak = 0
        self.break_disabled = not self.cfg.allow_obstacle_break
        # 막힌 주머니에서 제자리를 맴도는 것을 잡아내기 위한 상태
        self._last_layout: tuple | None = None
        self.stuck_cycles = 0
        # 고정된 격자. 게임판 패널은 화면에서 움직이지 않고 내용만 스크롤한다.
        self.locked_grid: Grid | None = None
        self._locked_size: tuple[int, int] | None = None
        self._grid_votes: deque = deque(maxlen=9)

    # ---------------------------------------------------------------- 정지
    def stop(self) -> None:
        self.stop_event.set()

    def _check_stop(self) -> None:
        """정지가 걸렸으면 즉시 예외로 빠져나온다.

        클릭 직전, 분석 직후 등 모든 갈림길에서 호출한다. 덕분에 '분석 중에
        정지를 눌렀는데 분석이 끝난 뒤 이전 경로가 뒤늦게 실행되는' 일이 없다.
        """
        if self.stop_event.is_set():
            raise Stopped()

    # ------------------------------------------------------------ 창 고정
    def pick_window(self) -> EmulatorWindow | None:
        """후보 창을 모두 평가해서 조건을 만족하는 창 하나를 고정한다.

        창 제목만 믿지 않는다. 두 조건을 함께 본다.
          1) 상단에 고정된 게임 탭 이미지가 있는가 (템플릿이 있을 때만)
          2) 화면 안에 5x5 게임판 격자 테두리가 있는가
        """
        self.candidates_report = []
        cands = enumerate_candidates(min_size=self.cfg.window_min_size,
                                     title_hint=self.cfg.window_title_hint)
        if not cands:
            self.log("[창] 앱플레이어로 볼 만한 창을 하나도 찾지 못했습니다. "
                     "앱플레이어가 실행 중이고 최소화되어 있지 않은지 확인하세요.")
            if self.cfg.window_title_hint:
                self.log(f"[창] config.json 의 window_title_hint="
                         f"'{self.cfg.window_title_hint}' 때문에 걸러졌을 수 있습니다.")
            return None

        tab_tpl = self.templates["top_tab"] if self.cfg.require_top_tab else None
        if self.templates["top_tab"] and not self.cfg.require_top_tab:
            self.log("[창] require_top_tab=false 라 상단 탭 검사를 건너뜁니다.")
        best = None
        for cand in cands:
            img = None
            try:
                img = capture_client(cand.hwnd)
            except Exception as e:
                cand.reasons.append(f"캡처 실패({e})")

            if img is None:
                cand.reasons.append("캡처 실패")
                self.candidates_report.append(cand.describe())
                continue

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

            self.candidates_report.append(cand.describe())
            if cand.ok and (best is None or cand.score > best.score):
                best = cand

        for line in self.candidates_report:
            self.log("[창] " + line)

        if best is None:
            self.log(f"[창] 후보 {len(cands)}개를 모두 봤지만 5x5 게임판이 있는 창이 "
                     f"없습니다. 탐사 화면을 띄운 상태인지 확인하세요.")
            self.log("[창] 창은 뜨는데 계속 실패하면 tools/detect_windows.py 를 "
                     "실행해 어떤 창이 어떻게 보이는지 확인할 수 있습니다.")
            return None

        win = EmulatorWindow(best.hwnd, best.top_hwnd, best.title)
        self.log(f"[창] 고정: HWND=0x{best.hwnd:X} ({best.width}x{best.height}) "
                 f"'{best.title}' 격자={best.board_score:.2f} 탭={best.tab_score:.2f}")
        return win

    # ------------------------------------------------------------- 캡처
    def _capture(self) -> np.ndarray | None:
        if self.window is None or not self.window.is_valid():
            return None
        img = self.window.capture()
        if img is not None:
            self.last_frame = img
        return img

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
    def _stable_grid(self, img: np.ndarray) -> Grid | None:
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
            # 몇 px 흔들리는 것은 같은 배치로 묶는다.
            key = (grid.xs[0] // 8, grid.ys[0] // 8,
                   int(grid.cell_w) // 8, int(grid.cell_h) // 8)
            self._grid_votes.append((key, grid))

        if not self._grid_votes:
            return self.locked_grid

        counts = Counter(k for k, _ in self._grid_votes)
        best_key, votes = counts.most_common(1)[0]
        chosen = next(g for k, g in reversed(self._grid_votes) if k == best_key)

        if self.locked_grid is None:
            self.log(f"[격자] 고정: x={chosen.xs} y={chosen.ys} "
                     f"(신뢰도 {chosen.confidence:.2f}, {votes}/{len(self._grid_votes)}표)")
        elif (chosen.xs[0] // 8, chosen.ys[0] // 8) != (
                self.locked_grid.xs[0] // 8, self.locked_grid.ys[0] // 8):
            self.log(f"[격자] 다수결로 갱신: y0 {self.locked_grid.ys[0]} -> "
                     f"{chosen.ys[0]} ({votes}/{len(self._grid_votes)}표)")

        self.locked_grid = chosen
        return chosen

    # --------------------------------------------- 초록 버튼(장애물 부수기)
    def _press_green_button(self) -> bool:
        """우측 하단 초록색 버튼을 눌러 장애물을 부순다.

        오른쪽으로 갈 길이 없을 때만 쓴다. 사용 횟수가 정해져 있으므로
        (실측: 30회) 함부로 누르면 안 된다.
        """
        if not self.cfg.use_green_button:
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
        self.log(f"[초록버튼] 장애물 부수기 (누적 {self.stats.green_button_uses}회) "
                 f"| 클라이언트{pos} -> 화면{screen}")
        self._sleep(0.6)
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
        return track_player_fast(img, grid, highlights, hsv)

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
        before = self._capture()
        if before is None:
            return False, grid, False

        # 매 이동 전에 격자선만 가볍게 다시 검출해서 셀 중심 좌표를 갱신한다.
        fresh = self._stable_grid(before)
        if fresh is not None:
            grid = fresh

        cx, cy = grid.cell_center(to[0], to[1])
        self._check_stop()
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

        while time.time() < deadline:
            self._check_stop()
            after = self._capture()
            if after is None:
                time.sleep(self.cfg.poll_interval_sec)
                continue

            if self._toast_visible(after):
                self._wait_toast_clear()
                return False, grid, False

            light = self._stable_grid(after)
            use_grid = light or grid
            pos = self._track_player(after, use_grid)

            ok = False
            scrolled = False
            if pos == to:
                ok = True
            elif pos == frm:
                # 이전 칸에 남은 애니메이션 잔상이거나 아직 안 움직였다.
                # 어느 쪽이든 성공으로 처리하지 않는다.
                ok = False
            elif pos is None:
                ok = False

            if not ok and pos in (frm, None):
                # 게임판이 통째로 스크롤해서 플레이어가 화면상 같은 칸에
                # 남아 있는 경우를 확인한다.
                key = (use_grid.xs[0], use_grid.ys[0], use_grid.xs[-1], use_grid.ys[-1])
                if before_sig is None or before_sig_grid != key:
                    before_sig = self._cell_signature(before, use_grid)
                    before_sig_grid = key
                if self._scrolled_one_cell(before, after, use_grid, direction,
                                           before_sig):
                    ok = True
                    scrolled = True

            if ok:
                confirmed += 1
                if confirmed >= self.cfg.confirm_repeat:
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

        loaded = {k: len(v.images) for k, v in self.templates.items() if v.images}
        self.log(f"[템플릿] {loaded if loaded else '없음 (색 기반 인식으로 동작)'}")

        self.window = self.pick_window()
        if self.window is None:
            self.status("창을 찾지 못함")
            return
        self.status(f"실행 중 (HWND 0x{self.window.hwnd:X})")

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

                grid = self._stable_grid(img)
                if grid is None:
                    lost += 1
                    if lost == 1 or lost % self.cfg.max_lost_before_report == 0:
                        self.log(f"[인식] 5x5 게임판을 찾지 못했습니다 ({lost}회). "
                                 f"탐사 화면이 맞는지 확인하세요.")
                        self._save_debug("board_lost", img)
                    self._sleep(self.cfg.lost_retry_sec)
                    continue

                scene = analyze(img, grid, self.templates,
                                self.cfg.orange_goal_without_template)

                # --- 2) 전체 경로 계산 --------------------------------
                plan = plan_route(scene)

                # 분석이 끝난 지금 다시 정지를 확인한다. 분석 도중에 정지를 눌렀다면
                # 여기서 빠져나가므로 이전 경로가 뒤늦게 실행되지 않는다.
                self._check_stop()

                drawn = overlay.draw(img, grid, scene, plan.path,
                                     f"{plan.kind.value} | {' '.join(plan.moves)}")
                self.last_overlay = drawn
                self.preview(drawn)
                self._save_debug("last", img, grid, scene, plan.path, plan.kind.value)

                if scene.player is None:
                    lost += 1
                    self.log("[인식] 플레이어를 찾지 못했습니다.")
                    self._save_debug("player_lost", img, grid, scene)
                    self._sleep(self.cfg.lost_retry_sec)
                    continue

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

                # --- 막힌 주머니 감지 --------------------------------
                # 오른쪽으로 한 칸도 못 가는 상태에서 장애물 배치까지 그대로면,
                # 세로로 움직여도 새 지형이 들어오지 않는 '주머니'에 갇힌 것이다.
                # 이때 계속 움직이면 (1,0)->(1,1)->(2,1)->(1,1)->... 처럼 제자리를
                # 맴돌며 이동 횟수만 축낸다(실측: 25번을 움직였는데 판이 그대로였다).
                # 플레이어 칸은 움직일 때마다 바뀌므로 빼고, **장애물 배치만** 본다.
                # 새 지형이 들어왔다면 장애물 배치가 반드시 달라진다.
                layout = tuple(tuple(c == Kind.OBSTACLE for c in row)
                               for row in scene.cells)
                if layout == self._last_layout:
                    self.stuck_cycles += 1
                else:
                    self.stuck_cycles = 0
                self._last_layout = layout

                if self.stuck_cycles >= 3:
                    # 오른쪽이 막혔다. 초록 버튼으로 장애물을 부순다.
                    if self._press_green_button():
                        self.stuck_cycles = 0
                        self._last_layout = None
                        continue
                    self.log(f"[갇힘] {self.stuck_cycles}사이클째 장애물 배치가 "
                             f"그대로인데 초록 버튼도 쓸 수 없습니다. "
                             f"{self.cfg.blocked_wait_sec:g}초 기다립니다.")
                    self._sleep(self.cfg.blocked_wait_sec)
                    continue

                if plan.kind == PlanKind.BREAK_OBSTACLE and self.break_disabled:
                    # 장애물 직접 클릭이 안 먹히는 것을 확인했다면 초록 버튼을 쓴다.
                    if self._press_green_button():
                        self._last_layout = None
                        continue
                    self.log(f"[경로] 우회로도 없고 장애물도 부술 수 없습니다. "
                             f"{self.cfg.blocked_wait_sec:g}초 기다렸다가 다시 봅니다.")
                    self._sleep(self.cfg.blocked_wait_sec)
                    continue

                # --- 3~7) 한 칸씩 클릭하며 실제 도착을 확인 ------------
                current = plan.path[0]
                for i, nxt in enumerate(plan.path[1:]):
                    self._check_stop()
                    direction = plan.moves[i] if i < len(plan.moves) else "CLICK"

                    if (plan.kind == PlanKind.BREAK_OBSTACLE
                            and scene.kind_at(nxt[0], nxt[1]) == Kind.OBSTACLE):
                        # 마지막 한 칸은 이동이 아니라 장애물 파괴 클릭이다.
                        cx, cy = grid.cell_center(nxt[0], nxt[1])
                        self._check_stop()
                        screen = self.window.click_client(cx, cy, self.cfg.move_duration)
                        self.stats.obstacles_broken += 1
                        self.log(f"[클릭] 장애물 파괴 시도 {nxt} | 클라이언트({cx},{cy}) -> "
                                 f"화면{screen}")
                        self._sleep(0.45)
                        # 안내문이 떴다면 이 장애물은 부술 수 없는 것이다.
                        shot = self._capture()
                        if shot is not None and self._toast_visible(shot):
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
                        break

                    ok, grid, scrolled = self._do_move(grid, current, nxt, direction)
                    if not ok:
                        break     # 고속 경로 폐기 -> 바깥 루프에서 전체 재인식
                    current = nxt
                    self.break_fail_streak = 0   # 길이 열렸다면 다시 시도해 볼 만하다
                    if scrolled:
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
            self.status("정지됨")

    def _sleep(self, sec: float) -> None:
        """정지 요청에 즉시 반응하는 sleep."""
        if self.stop_event.wait(sec):
            raise Stopped()
