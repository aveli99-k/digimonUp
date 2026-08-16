"""3번 기능: 던전 (도전 반복).

한 문장 요약:
    "고정된 앱플레이어 창에서 '도전' 버튼을 찾아 누르고, 판이 끝나 뜬 팝업
     (이기면 보상창, 지면 실패창)을 **바깥 한 번**으로 닫은 뒤 다시 도전한다."

매 스캔마다 **셋을 함께 찾는다.**
    던전 화면   하단에 파란 '도전' 버튼이 있다    -> 누른다
    실패창     위쪽에 '실패...' 글자가 있다      -> 바깥을 한 번 눌러 닫는다
    보상창     가운데에 '보상' 글자가 있다       -> 바깥을 한 번 눌러 닫는다
전투 중에는 셋 다 안 보인다. 그때는 아무것도 하지 않고 기다린다.

한 판의 끝은 **둘 중 하나**다. 지면 실패창, 이기면 보상창이 뜬다. 어느 쪽이
뜰지 미리 알 수 없으므로 순서를 정해 두지 않는다. 셋을 다 찾아 놓고 **보이는
쪽을 바로 누른다.** 그래서 이겨도 져도 같은 코드가 그대로 돈다.

클릭 규칙은 셋 다 똑같이 하나다.
    한 번 누른다 -> recheck_sec 동안은 다시 누르지 않는다
                 -> 그 뒤에도 그대로면 **한 번 더** 누른다 (반복)

    이 '기다렸다 한 번씩'이 중요하다. 화면이 아직 안 바뀌었다고 연달아 누르면,
    팝업이 닫히는 순간의 클릭이 그 아래 던전 화면으로 전달돼 엉뚱한 것을 누른다.

왜 실패창과 보상창을 '바깥 클릭'으로 닫는가
    둘 다 닫기 버튼이 없다. 어두워진 바깥을 아무 데나 한 번 누르면 닫힌다
    (보상창에는 아예 '터치하여 닫기' 라고 쓰여 있다). 실측으로 둘 다 클라이언트
    (60, 60) 한 번에 닫혔고, 보상창은 닫으면서 스테이지가 83 -> 84 로 넘어갔다
    (보상이 정상 반영됐다는 뜻이다).

    **왼쪽 위 구석을 누르는 것이 중요하다.** 보상창 아래쪽에는 '포기' 버튼이
    같이 보인다. 거기를 잘못 누르면 던전 보상을 못 받는다("포기 시 던전 보상은
    획득할 수 없습니다"). 화면에서 그 버튼과 가장 먼 자리를 고른 것이다.

    실패창에는 '성장 가이드' 안내판이 함께 붙어 나오는데, 그것까지 한 덩어리로
    같이 닫힌다. 그래서 따로 다룰 필요가 없다.

왜 '실패...' 만 보는가 (윗줄은 보지 않는다)
    실패창의 첫 줄은 **그때그때 다르다.** 실측으로 '던전 실패...' 와
    '스테이지 실패...' 두 가지를 봤다. 처음에는 '던전 실패...' 를 통째로 템플릿
    으로 떴는데, 첫 줄이 '스테이지' 로 바뀐 화면에서 점수가 1.000 -> 0.790 으로
    떨어졌다. 기준 0.70 을 아슬아슬하게 넘긴 것이라, 첫 줄이 더 긴 이름으로
    바뀌면 그대로 놓쳤을 것이다.

    둘째 줄 '실패...' 만 잘라 쓰면 두 화면 모두 1.000 이고, 실패창이 아닌
    던전 화면에서는 0.376 이다. 바뀌지 않는 부분만 보는 쪽이 옳다.

왜 5x5 게임판으로 창을 고르지 않는가
    탐사(2번)는 화면에 게임판이 있는지로 창을 확정한다. 던전 화면에는 게임판이
    없으므로 그 검사를 그대로 쓰면 후보가 0개가 된다. 대신 **이 기능이 실제로
    찾는 세 가지 중 하나라도 보이는 창**을 고른다. 판정을 화면 내용으로 한다는
    원칙은 같고, 보는 대상만 다르다.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from digimonup.base.common import Stopped, is_stop_key_pressed, vk_of
from digimonup.base.imgio import imwrite
from digimonup.base.paths import DEBUG_DIR, DUNGEON_TEMPLATE_DIR
from digimonup.vision.recognize import TemplateSet, match_big
from digimonup.win.emulator_window import (EmulatorWindow, capture_client,
                                           enable_dpi_awareness, enumerate_candidates)


@dataclass
class DungeonConfig:
    # 창 검증
    window_title_hint: str = ""
    window_min_size: int = 200

    # 인식 기준
    # 실측 (709x1260 캡처)
    #   도전 버튼   있을 때 1.000 / 없을 때(실패창) 0.533
    #   실패 글자   있을 때 1.000 (첫 줄이 '던전'이든 '스테이지'든 같다)
    #               / 없을 때(던전 화면) 0.376
    #   보상 글자   있을 때 0.826~1.000 / 없을 때 0.433
    #               (뒤에서 반짝이는 이펙트 때문에 프레임마다 흔들린다.
    #                템플릿을 두 장 넣어 가장 잘 맞는 쪽을 쓴다. 0.826 은 팝업이
    #                떠오르는 중에 잡힌 실주행 값이다)
    # 기준은 그 사이에 둔다.
    challenge_min: float = 0.80
    fail_min: float = 0.70
    reward_min: float = 0.70

    # 각 표시가 나오는 세로 범위 (화면 높이 대비).
    # 실측: 실패 글자 0.13~0.19, 보상 글자 0.23~0.28, 도전 버튼 0.77~0.81.
    # 넉넉히 잡아도 이만큼이면 되고, 좁혀 두면 빨라지는 데다 화면 다른 곳의
    # 비슷한 무늬를 아예 안 본다(던전 화면에도 '획득 가능 보상' 글자가 있다).
    fail_band: tuple[float, float] = (0.02, 0.32)
    reward_band: tuple[float, float] = (0.12, 0.45)
    challenge_band: tuple[float, float] = (0.65, 0.95)

    # 팝업(실패창/보상창)을 닫으려고 누를 자리 (클라이언트 가로/세로 대비 비율).
    #
    # 픽셀이 아니라 비율로 두는 이유: 앱플레이어 창 크기는 사람마다 다르다.
    # 실측한 (60, 60) 은 709x1260 에서 (0.085, 0.048) 이다.
    #
    # **왼쪽 위 구석이어야 한다.** 보상창 아래쪽에 '포기' 버튼이 함께 보이는데,
    # 거기를 누르면 던전 보상을 못 받는다. 그 버튼에서 가장 먼 자리다.
    popup_close_point: tuple[float, float] = (0.085, 0.048)

    # 대기
    scan_interval_sec: float = 0.5      # 화면 확인 주기
    move_duration: float = 0.05         # 마우스 이동 시간
    start_delay_sec: float = 3.0        # 시작 버튼을 누르고 게임 창을 띄울 시간

    # 한 번 누른 뒤, 화면이 그대로일 때 **다시 한 번** 누르기까지 기다리는 시간.
    #
    # 이 값이 이 기능의 안전장치 전부다. 화면이 바뀌는 데 걸리는 시간보다 넉넉히
    # 길어야 한다. 실측: 실패창은 바깥을 누르면 0.5초 안에 닫혔다. 3초면 닫히는
    # 중에 한 번 더 누르는 일이 없다.
    recheck_sec: float = 3.0
    # 같은 대상을 이만큼 연속으로 눌렀는데도 그대로면 경고를 남긴다.
    # 멈추지는 않는다 — 도전 횟수에는 제한이 없으므로, 안 눌리는 이유는 대개
    # 클릭이 게임까지 닿지 않는 것(관리자 권한 등)이고 그건 곧 풀릴 수도 있다.
    max_attempts: int = 4

    # 아무것도 안 보이는 상태(전투 중)를 몇 초마다 로그로 알릴지
    idle_report_sec: float = 15.0

    # 중지 키 (창 포커스와 무관하게 어디서 눌러도 먹는다)
    stop_key: str = "F12"

    # 디버그: 도전을 누른 프레임과 실패를 잡은 프레임을 debug/dungeon/ 에 남긴다
    save_debug: bool = True


# 찾는 것들. (한글 이름, 최소 유사도 설정 키, 세로 범위 설정 키)
# 표로 두면 넷째 팝업이 생겨도 한 줄 추가와 템플릿 폴더 하나로 끝난다.
KINDS: dict[str, tuple[str, str, str]] = {
    "fail": ("실패창", "fail_min", "fail_band"),
    "reward": ("보상창", "reward_min", "reward_band"),
    "challenge": ("도전", "challenge_min", "challenge_band"),
}

# 이 둘은 팝업이다. 글자가 있는 자리가 아니라 **바깥**을 눌러 닫는다.
# 그리고 떠 있는 동안에는 아래 도전 버튼을 눌러도 팝업이 클릭을 먹으므로,
# 도전보다 먼저 본다(KINDS 의 순서가 곧 우선순위다).
POPUP_KINDS = ("fail", "reward")


@dataclass
class Target:
    """지금 눌러야 할 것 하나."""
    kind: str                  # KINDS 의 열쇠
    at: tuple[int, int]        # 누를 클라이언트 좌표
    score: float

    @property
    def name(self) -> str:
        return KINDS[self.kind][0]


@dataclass
class DungeonStats:
    """대상별 클릭 수. 처음 누른 것과 '그대로여서 다시 누른 것'을 따로 센다."""
    clicks: Counter = field(default_factory=Counter)
    reclicks: Counter = field(default_factory=Counter)

    def line(self) -> str:
        parts = [f"{KINDS[k][0]} {self.clicks[k]}" for k in KINDS]
        again = sum(self.reclicks.values())
        return " / ".join(parts) + f" / 재클릭 {again}"


def load_dungeon_templates() -> dict[str, TemplateSet]:
    return {kind: TemplateSet(kind, base_dir=DUNGEON_TEMPLATE_DIR) for kind in KINDS}


class DungeonEngine:
    """던전 자동화 엔진. GUI 든 콘솔이든 콜백만 갈아끼우면 그대로 쓴다."""

    def __init__(self, cfg: DungeonConfig | None = None,
                 log=print, status=lambda s: None, preview=lambda img: None):
        self.cfg = cfg or DungeonConfig()
        self.log = log
        self.status = status
        self.preview = preview
        self.stop_event = threading.Event()
        self.window: EmulatorWindow | None = None
        self.templates = load_dungeon_templates()
        self.stats = DungeonStats()
        self.last_frame: np.ndarray | None = None
        self.candidates_report: list[str] = []
        self._stop_vk = vk_of(self.cfg.stop_key) if self.cfg.stop_key else 0
        # 마지막 스캔의 (도전, 실패) 점수. 대기 로그에 찍는다.
        self._last_scores: dict[str, float] = {kind: 0.0 for kind in KINDS}

    # ---------------------------------------------------------------- 정지
    def stop(self) -> None:
        self.stop_event.set()

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped()
        if self._stop_vk and is_stop_key_pressed(self._stop_vk):
            self.stop_event.set()
            self.log(f"[정지] {self.cfg.stop_key} 키를 눌러 중단합니다.")
            raise Stopped()

    def _sleep(self, sec: float) -> None:
        """정지 요청에 즉시 반응하는 sleep."""
        if self.stop_event.wait(sec):
            raise Stopped()

    # ------------------------------------------------------------ 화면 읽기
    def _in_band(self, img: np.ndarray, band: tuple[float, float]):
        """화면에서 띠만 잘라 (조각, 세로 오프셋) 로 돌려준다."""
        h = img.shape[0]
        top = max(0, int(h * band[0]))
        bottom = min(h, int(h * band[1]))
        if bottom <= top:
            return None, 0
        return img[top:bottom], top

    def _find(self, img: np.ndarray, name: str, band: tuple[float, float]):
        """띠 안에서 템플릿을 찾는다. 반환: (점수, 중심(x, y) | None)

        중심은 **화면 전체 기준**이다(띠 오프셋을 더해 돌려준다).
        """
        tset = self.templates[name]
        if not tset or img is None or img.size == 0:
            return 0.0, None
        strip, top = self._in_band(img, band)
        if strip is None:
            return 0.0, None
        score, box, _ = match_big(strip, tset, scales=(0.85, 1.0, 1.15))
        if box is None:
            return float(score), None
        return float(score), ((box[0] + box[2]) // 2, top + (box[1] + box[3]) // 2)

    def _find_kind(self, img: np.ndarray, kind: str):
        _, _, band_key = KINDS[kind]
        return self._find(img, kind, getattr(self.cfg, band_key))

    def _min_of(self, kind: str) -> float:
        return getattr(self.cfg, KINDS[kind][1])

    def popup_close_point(self) -> tuple[int, int]:
        """팝업을 닫으려고 누를 자리 (클라이언트 좌표)."""
        w, h = self.window.client_size()
        return (int(w * self.cfg.popup_close_point[0]),
                int(h * self.cfg.popup_close_point[1]))

    def _look(self, img: np.ndarray) -> Target | None:
        """셋을 **다 찾아서**, 지금 눌러야 할 것 하나를 돌려준다.

        순서를 정해 두지 않는 이유: 한 판의 끝은 실패창일 수도 보상창일 수도
        있다. '실패를 기다렸다가 도전' 으로 짜면 이긴 판에서 영영 기다린다.

        팝업이 도전보다 먼저다. 점수가 높은 쪽이 아니다. 팝업은 모달이라,
        떠 있는 동안 아래 도전 버튼을 눌러 봐야 팝업이 클릭을 먹고 아무 일도
        일어나지 않기 때문이다. (1번 기능은 매칭/포기 중 유사도가 높은 쪽을
        따르는데, 그쪽은 둘 다 그냥 버튼이라 사정이 다르다.)
        """
        found = {kind: self._find_kind(img, kind) for kind in KINDS}
        self._last_scores = {kind: score for kind, (score, _) in found.items()}

        ch_score, ch_at = found["challenge"]
        ch_ok = ch_score >= self.cfg.challenge_min and ch_at is not None

        for kind in POPUP_KINDS:
            score, _ = found[kind]
            if score < self._min_of(kind):
                continue
            if ch_ok:
                self.log(f"[화면] {KINDS[kind][0]}과 도전 버튼이 동시에 "
                         f"잡혔습니다 ({KINDS[kind][0]} {score:.3f} / "
                         f"도전 {ch_score:.3f}). 팝업이 클릭을 먹으므로 "
                         f"{KINDS[kind][0]}을 먼저 닫습니다.")
            # 누를 곳은 글자가 있는 자리가 아니라 팝업 **바깥**이다.
            return Target(kind, self.popup_close_point(), score)

        if ch_ok:
            return Target("challenge", ch_at, ch_score)
        return None

    # ------------------------------------------------------------ 창 고정
    def pick_window(self) -> EmulatorWindow | None:
        """후보 창을 모두 평가해서 던전 화면이 보이는 창 하나를 고정한다.

        창 제목만 믿지 않는다. 도전 버튼이나 실패 글자 중 **하나라도** 실제로
        보이는 창만 통과시킨다.
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

        best = None
        best_score = 0.0
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

            scores = {kind: self._find_kind(img, kind)[0] for kind in KINDS}
            cand.board_score = max(scores.values())   # 표시용 (describe 가 찍는다)
            cand.ok = any(score >= self._min_of(kind)
                          for kind, score in scores.items())
            cand.reasons.append(" ".join(f"{KINDS[k][0]}={s:.2f}"
                                         for k, s in scores.items()))
            if not cand.ok:
                cand.reasons.append("셋 다 기준 미달 (" + " / ".join(
                    f"{KINDS[k][0]} {self._min_of(k):.2f}" for k in KINDS) + ")")

            self.candidates_report.append(cand.describe())
            if cand.ok and cand.board_score > best_score:
                best, best_score = cand, cand.board_score

        for line in self.candidates_report:
            self.log("[창] " + line)

        if best is None:
            self.log(f"[창] 후보 {len(cands)}개를 모두 봤지만 던전 화면이 보이는 창이 "
                     f"없습니다. 게임에서 던전 화면(도전 버튼이 보이는 화면)이나 "
                     f"실패창/보상창을 띄운 상태인지 확인하세요.")
            if not all(self.templates.values()):
                self.log("[창] templates/dungeon/ 의 템플릿이 비어 있습니다. "
                         "tools/capture_dungeon.py 로 먼저 찍어주세요.")
            return None

        win = EmulatorWindow(best.hwnd, best.top_hwnd, best.title)
        self.log(f"[창] 고정: HWND=0x{best.hwnd:X} ({best.width}x{best.height}) "
                 f"'{best.title}' 일치={best_score:.2f}")
        return win

    # ------------------------------------------------------------- 캡처
    def _capture(self, preview: bool = False) -> np.ndarray | None:
        """창을 캡처한다.

        미리보기는 부르는 쪽이 정한다. 창 고르기처럼 판단과 무관한 캡처까지
        GUI 로 보내면 큐가 그림으로 가득 차 화면이 굼떠진다.
        """
        if self.window is None or not self.window.is_valid():
            return None
        img = self.window.capture()
        if img is not None:
            self.last_frame = img
            if preview:
                self.preview(img)
        return img

    def _save_debug(self, name: str, img: np.ndarray | None) -> None:
        if not self.cfg.save_debug or img is None:
            return
        imwrite(os.path.join(DEBUG_DIR, "dungeon", f"{name}.png"), img)

    # --------------------------------------------------------------- 클릭
    def _click(self, target: Target, repeats: int) -> bool:
        """대상을 **한 번** 누른다. repeats 는 같은 대상을 다시 누르는 횟수다."""
        self._check_stop()
        if repeats == 0:
            self._save_debug(f"{target.kind}_seen", self.last_frame)

        got = self.window.click_client(target.at[0], target.at[1],
                                       self.cfg.move_duration)
        if got is None:
            self.log(f"[{target.name}] ({target.at[0]},{target.at[1]}) 클릭에 "
                     f"실패했습니다. 창이 닫혔거나 좌표가 화면 밖입니다.")
            return False

        if repeats == 0:
            self.stats.clicks[target.kind] += 1
            where = ("바깥 검은 영역" if target.kind in POPUP_KINDS else "버튼")
            self.log(f"[{target.name}] 감지 (유사도 {target.score:.3f}) -> "
                     f"{where} ({target.at[0]},{target.at[1]}) 1회 클릭 "
                     f"(누적 {self.stats.clicks[target.kind]}회)")
        else:
            self.stats.reclicks[target.kind] += 1
            self.log(f"[{target.name}] {self.cfg.recheck_sec:g}초가 지나도 "
                     f"그대로입니다. 한 번 더 누릅니다. ({repeats}번째 재시도)")
        return True

    def _warn_stuck(self, target: Target) -> None:
        """한도까지 눌렀는데도 화면이 그대로일 때. 멈추지는 않는다."""
        self.log(f"[{target.name}] {self.cfg.max_attempts}회를 눌렀는데도 화면이 "
                 f"그대로입니다. 클릭이 게임까지 닿지 않는 것일 수 있습니다 "
                 f"(게임이 관리자 권한이면 매크로도 관리자 권한으로 실행하세요). "
                 f"계속 시도합니다.")
        self._save_debug(f"{target.kind}_stuck", self.last_frame)

    # --------------------------------------------------------------- 메인
    def run(self) -> None:
        # stop_event 를 여기서 지우지 않는다. 지우면 '시작 전에 이미 눌린 정지'가
        # 없던 일이 된다. 새로 시작할 때는 엔진을 새로 만든다.
        self.stats = DungeonStats()
        enable_dpi_awareness()

        loaded = {k: len(v.images) for k, v in self.templates.items()}
        self.log(f"[템플릿] {loaded}")
        if not all(self.templates.values()):
            self.log("[템플릿] 던전 템플릿이 없습니다. 던전은 색만으로는 판단할 수 "
                     "없어 템플릿이 반드시 있어야 합니다. "
                     "tools/capture_dungeon.py 로 찍어주세요.")
            self.status("템플릿 없음")
            return

        if self.cfg.start_delay_sec > 0:
            self.log(f"{self.cfg.start_delay_sec:.0f}초 뒤 시작합니다. "
                     f"던전 화면을 띄워주세요...")
            if self.stop_event.wait(self.cfg.start_delay_sec):
                self.log("[정지] 시작 전에 중지되었습니다.")
                self.status("정지됨")
                return

        self.window = self.pick_window()
        if self.window is None:
            self.status("창을 찾지 못함")
            return
        self.status(f"실행 중 (HWND 0x{self.window.hwnd:X})")
        if self._stop_vk:
            self.log(f"[정지] 멈추려면 GUI 의 정지 버튼 또는 "
                     f"{self.cfg.stop_key} 키를 누르세요 (어느 창에서든 먹습니다).")

        # 마지막으로 누른 대상과 시각. 같은 것이 계속 보이면 recheck_sec 뒤에
        # 한 번씩만 더 누른다.
        pending: str | None = None
        clicked_at = 0.0
        repeats = 0

        idle_since = None
        last_idle_report = 0.0

        try:
            while True:
                self._check_stop()

                img = self._capture(preview=True)
                if img is None:
                    self.log("[캡처] 실패. 창이 닫혔을 수 있습니다.")
                    break

                target = self._look(img)
                now = time.time()

                if target is None:
                    # 셋 다 안 보인다 = 전투 중이거나 화면 전환 중. 기다린다.
                    # 직전에 누른 것은 먹혔다는 뜻이므로 재시도 상태를 지운다.
                    pending, repeats = None, 0
                    if idle_since is None:
                        idle_since = now
                    if now - last_idle_report > self.cfg.idle_report_sec:
                        seen = " / ".join(f"{KINDS[k][0]} {s:.2f}"
                                          for k, s in self._last_scores.items())
                        self.log(f"[대기] 아무것도 안 보입니다 "
                                 f"({now - idle_since:.0f}초째, {seen})")
                        last_idle_report = now
                    self._sleep(self.cfg.scan_interval_sec)
                    continue

                idle_since = None

                if target.kind == pending:
                    # 눌렀는데 아직 그대로다. 정해 둔 시간 전에는 다시 누르지
                    # 않는다. 화면이 바뀌는 중일 수 있기 때문이다.
                    if now - clicked_at < self.cfg.recheck_sec:
                        self._sleep(self.cfg.scan_interval_sec)
                        continue
                    repeats += 1
                    if repeats >= self.cfg.max_attempts:
                        self._warn_stuck(target)
                        repeats = 0
                else:
                    pending, repeats = target.kind, 0

                if not self._click(target, repeats):
                    break
                clicked_at = time.time()
                self._sleep(self.cfg.scan_interval_sec)

        except Stopped:
            self.log("[정지] 요청을 받아 즉시 중단했습니다. 대기 중이던 클릭도 취소합니다.")
        except Exception as e:
            self.log(f"[오류] {type(e).__name__}: {e}")
            self._save_debug("error", self.last_frame)
            raise
        finally:
            self.log(f"[종료] {self.stats.line()}")
            self.status("정지됨")
