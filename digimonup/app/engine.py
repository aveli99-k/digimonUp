"""창 하나를 고정해서 쓰는 기능들의 공통 뼈대 (탐사 2번 / 던전 3번).

두 기능은 **무엇을 보고 창을 고르는지**만 다르다.
    탐사   화면 안에 5x5 게임판 격자가 있는가
    던전   도전 버튼 / 실패창 / 보상창 중 하나라도 보이는가

그 판정 말고는 통째로 같다 — 후보 열거, 후보마다 캡처, 캡처 실패 처리, 평가
보고서 출력, 가장 좋은 후보 고르기, HWND 고정. 실제로 두 파일에 21줄이 글자까지
같은 채로 들어 있었다(pylint symilar 로 확인).

사본이 둘이면 한쪽만 고쳐진다. 이미 그런 일이 있었다 — `pickwin.py` 주석에
"한쪽만 손봐서, 오래된 쪽은 앱플레이어를 가려내지 않고 안내문에도 MuMuPlayer 만
적혀 있었다"고 남아 있다. 같은 실수를 엔진 쪽에서 반복할 이유가 없다.

물려받는 쪽이 채울 것은 셋뿐이다.
    _judge(img, cand)      이 창이 쓸 만한지 보고 cand.ok / 점수 / 사유를 채운다
    _no_match_help(n)      하나도 못 골랐을 때 사용자에게 할 말 (줄 목록)
    _picked_note(cand)     고정 로그 끝에 붙일 점수 표기
"""

from __future__ import annotations

import numpy as np

from digimonup.base.common import StoppableEngine
from digimonup.win.emulator_window import (EmulatorWindow, capture_client,
                                           enumerate_candidates)


class WindowedEngine(StoppableEngine):
    """앱플레이어 창 하나를 고정하고 그 창만 캡처/클릭하는 엔진."""

    def __init__(self, stop_key: str = "", log=print,
                 status=lambda s: None, preview=lambda img: None):
        super().__init__(stop_key, log)
        self.status = status
        self.preview = preview
        self.window: EmulatorWindow | None = None
        self.last_frame: np.ndarray | None = None
        self.candidates_report: list[str] = []

    # --------------------------------------------------------------- 캡처
    def _capture(self, preview: bool = False) -> np.ndarray | None:
        """고정된 창을 캡처한다.

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

    def attach_window(self) -> bool:
        """창을 골라 고정한다. 못 찾으면 False — 부르는 쪽은 그대로 끝낸다.

        run() 첫머리가 두 기능에서 같아서 여기로 모았다. pick_window 를 **속성으로**
        부르므로, GUI 가 HWND 표시를 위해 갈아끼운 것도 그대로 먹는다(gui._run_windowed).
        """
        self.window = self.pick_window()
        if self.window is None:
            self.status("창을 찾지 못함")
            return False
        self.status(f"실행 중 (HWND 0x{self.window.hwnd:X})")
        self.log_stop_hint()
        return True

    # ----------------------------------------------------------- 창 고르기
    def pick_window(self) -> EmulatorWindow | None:
        """후보 창을 모두 평가해서 조건을 만족하는 창 하나를 고정한다.

        **창 제목만 믿지 않는다.** 최종 판정은 화면 내용으로 한다(_judge).
        제목은 후보를 좁히는 힌트일 뿐이다 — 실측: 이 저장소 페이지를 띄운
        브라우저 창의 제목에 'MuMu' 가 들어 있어서 앱플레이어로 잡혔다.
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

        self._prepare_judging()

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

            self._judge(img, cand)
            self.candidates_report.append(cand.describe())
            # 점수가 같으면 아는 앱플레이어를 우대한다 (Candidate.score 참고).
            if cand.ok and (best is None or cand.score > best.score):
                best = cand

        for line in self.candidates_report:
            self.log("[창] " + line)

        if best is None:
            for line in self._no_match_help(len(cands)):
                self.log("[창] " + line)
            return None

        win = EmulatorWindow(best.hwnd, best.top_hwnd, best.title)
        self.log(f"[창] 고정: HWND=0x{best.hwnd:X} ({best.width}x{best.height}) "
                 f"'{best.title}' {self._picked_note(best)}")
        return win

    # ------------------------------------------------------- 물려받아 채울 것
    def _prepare_judging(self) -> None:
        """후보를 보기 전에 한 번. 알릴 것이 있으면 여기서 알린다."""

    def _judge(self, img: np.ndarray, cand) -> None:
        """이 창이 쓸 만한지 보고 cand.ok / 점수 / reasons 를 채운다."""
        raise NotImplementedError

    def _no_match_help(self, n_candidates: int) -> list[str]:
        """하나도 못 골랐을 때 할 말. 한 줄에 하나씩."""
        return [f"후보 {n_candidates}개를 모두 봤지만 쓸 만한 창이 없습니다."]

    def _picked_note(self, cand) -> str:
        """고정 로그 끝에 붙일 점수 표기."""
        return f"점수={cand.score:.2f}"
