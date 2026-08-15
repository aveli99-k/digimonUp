"""캡처 도구들이 함께 쓰는 '창 고르기'.

capture_explore.py 와 capture_counters.py 에 거의 같은 함수가 하나씩 있었다.
그런데 한쪽만 손봐서, 오래된 쪽은 앱플레이어를 가려내지 않고 안내문에도
MuMuPlayer 만 적혀 있었다(여러 앱플레이어를 지원하기 전 문구다).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator_window import enumerate_candidates  # noqa: E402


def pick_window():
    """앱플레이어 창 하나를 고른다. 여러 개면 번호를 물어본다.

    앱플레이어로 알아본 창이 있으면 그것들만 후보로 보여 준다. 모르는 창까지
    섞어 놓으면 목록이 길어져 고르기 어렵다.
    """
    cands = enumerate_candidates()
    if not cands:
        print("앱플레이어 창을 찾지 못했습니다. 먼저 실행하세요.")
        return None

    known = [c for c in cands if c.emulator]
    pool = known or cands
    if len(pool) == 1:
        return pool[0]

    print("\n창을 고르세요.")
    for i, c in enumerate(pool):
        print(f"  {i}) hwnd=0x{c.hwnd:X} {c.width}x{c.height} "
              f"[{c.emulator or '모름'}] '{c.title}'")
    try:
        return pool[int(input("번호: ").strip())]
    except (ValueError, IndexError):
        return None
