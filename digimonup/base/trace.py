"""매크로가 **왜 그렇게 했는지**를 통째로 남긴다.

왜 필요한가
    "쓸데없이 부순다", "칩을 지나친다" 같은 보고를 받으면 그때마다 임시
    스크립트를 짜서 쟀다. 그러다 보니 기준이 그때그때 달랐고, 실제로 틀린
    계측으로 잘못된 결론을 여러 번 냈다(상단 보유량은 0.1K 단위 반올림이라
    칩 한 개로는 안 움직이는데 그걸 정답지로 썼다).

    그래서 **한 번 제대로 심어 둔다.** 매 사이클의 판·판단·클릭·결과를 한 줄씩
    남기면, 나중에 어떤 질문이 와도 같은 기록으로 답할 수 있다.

무엇이 남는가
    debug/trace/<시각>/log.jsonl   한 줄에 한 사건 (아래 record 참고)
    debug/trace/<시각>/NNNN.png    그 사이클의 화면 (frames=True 일 때)

읽는 법
    python tools\\analyze_trace.py                 가장 최근 기록을 분석
    python tools\\analyze_trace.py debug\\trace\\...  특정 기록을 분석

한 줄의 생김새 (cycle)
    {"t": 12.3, "kind": "cycle", "n": 7,
     "player": [1,1], "player_note": "움직임으로 확인",
     "board": ["EEXEE", ...], "chips": [[0,3]],
     "items": {"2,4": "steps"}, "highlights": [[0,1],[1,0]],
     "counts": {"steps": 1782, "break": 68, "dash": 0},
     "break_cost": 12.0, "plan": "목적지", "path": [[1,1],[1,2]],
     "reason": "...", "notes": [...]}

한 줄의 생김새 (move / break / dash)
    {"t": 12.9, "kind": "move", "dir": "RIGHT", "from": [1,1], "to": [1,2],
     "ok": true, "scrolled": true, "secs": 0.62}
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from digimonup.base import imgio
from digimonup.base.paths import DEBUG_DIR

TRACE_DIR = os.path.join(DEBUG_DIR, "trace")


class Tracer:
    """사건을 한 줄씩 파일에 적는다. 꺼져 있으면 아무것도 하지 않는다."""

    def __init__(self, enabled: bool = False, frames: bool = False,
                 max_frames: int = 400):
        self.enabled = enabled
        self.frames = frames
        self.max_frames = max_frames
        self.dir: str | None = None
        self._fh = None
        self._t0 = time.time()
        self._frames_saved = 0
        if not enabled:
            return
        stamp = time.strftime("%m%d_%H%M%S")
        self.dir = os.path.join(TRACE_DIR, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self._fh = open(os.path.join(self.dir, "log.jsonl"), "w", encoding="utf-8")

    # ------------------------------------------------------------- 쓰기
    def write(self, kind: str, **fields: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        rec = {"t": round(time.time() - self._t0, 2), "kind": kind}
        rec.update(fields)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()          # 중간에 멈춰도 남아 있어야 한다

    def frame(self, n: int, img) -> None:
        """그 사이클의 화면. 나중에 눈으로 확인할 때 쓴다."""
        if (not self.enabled or not self.frames or img is None
                or self.dir is None or self._frames_saved >= self.max_frames):
            return
        self._frames_saved += 1
        imgio.imwrite(os.path.join(self.dir, f"{n:04d}.png"), img)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def board_rows(cells) -> list[str]:
    """판을 한 줄에 한 행씩 글자로. E=빈칸 P=플레이어 G=칩 O=장애물 I=아이템."""
    return ["".join(k.name[0] for k in row) for row in cells]


def latest_dir() -> str | None:
    """가장 최근 기록 폴더."""
    if not os.path.isdir(TRACE_DIR):
        return None
    dirs = [os.path.join(TRACE_DIR, d) for d in os.listdir(TRACE_DIR)]
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "log.jsonl"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def load(path: str | None = None) -> list[dict]:
    """기록을 읽어 온다. path 가 폴더면 그 안의 log.jsonl 을 읽는다."""
    if path is None:
        path = latest_dir()
    if path is None:
        return []
    if os.path.isdir(path):
        path = os.path.join(path, "log.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
