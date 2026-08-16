"""창 탐지 진단 도구.

"창을 찾지 못했습니다" 가 뜰 때 **왜** 못 찾는지 눈으로 확인하는 도구다.
매크로가 보는 것과 똑같은 후보 목록을 뽑아, 각 창의 클래스/제목/크기와
5x5 게임판 검출 신뢰도를 보여주고, 캡처 이미지를 debug/windows/ 에 저장한다.

실행:
    python tools\\detect_windows.py

읽는 법
    격자 0.45 이상  -> 이 창이 게임 화면이다. 매크로가 여기를 쓴다.
    격자 0.00       -> 캡처는 됐지만 게임판이 안 보인다.
                       (탐사 화면이 아니거나, 다른 창이거나, 캡처가 검은 화면)
    캡처 실패       -> 이 창은 PrintWindow 로 내용을 못 가져온다.

앱플레이어를 여러 개 띄웠는데 엉뚱한 창을 고른다면, 여기 나온 제목의 일부를
config.json 의 explore.window_title_hint 에 적으면 그 창만 쓴다.
"""

from __future__ import annotations

import _bootstrap  # 저장소 루트를 sys.path 에 넣는다. 맨 먼저 가져온다

import os

from digimonup.vision import board                                          # noqa: E402
from digimonup.base import imgio                                          # noqa: E402
from digimonup.win.emulator_window import (EMULATOR_PROFILES, capture_client,  # noqa: E402
                             enable_dpi_awareness, enumerate_candidates)
from digimonup.base.paths import DEBUG_DIR                           # noqa: E402

OUT_DIR = os.path.join(DEBUG_DIR, "windows")


def main() -> int:
    enable_dpi_awareness()

    print("=" * 66)
    print(" digimonUp - 창 탐지 진단")
    print("=" * 66)
    print(f"아는 앱플레이어: {', '.join(p.name for p in EMULATOR_PROFILES)}")
    print("(목록에 없어도 화면에 5x5 게임판만 보이면 동작합니다)\n")

    cands = enumerate_candidates()
    if not cands:
        print("후보 창이 하나도 없습니다.")
        print("  - 앱플레이어가 실행 중인가요?")
        print("  - 창이 최소화되어 있지는 않나요? (최소화된 창은 캡처가 안 됩니다)")
        return 1

    print(f"후보 {len(cands)}개를 찾았습니다. 각 창을 캡처해 게임판을 찾아봅니다.\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    best = None
    for i, cand in enumerate(cands):
        who = cand.emulator or "모름"
        print(f"[{i}] hwnd=0x{cand.hwnd:X}  {cand.width}x{cand.height}  [{who}]")
        print(f"    제목  : {cand.title!r}")
        print(f"    클래스: {cand.cls}")
        print(f"    판단  : {', '.join(cand.reasons)}")

        img = None
        try:
            img = capture_client(cand.hwnd)
        except Exception as e:
            print(f"    캡처  : 실패 ({e})")

        if img is None:
            print("    캡처  : 실패 (PrintWindow 로 내용을 못 가져옴)")
            print()
            continue

        grid = board.detect_board(img, min_confidence=0.0)
        score = grid.confidence if grid else 0.0
        mark = "<-- 게임 화면으로 보입니다" if score >= 0.45 else ""
        print(f"    격자  : {score:.2f} {mark}")
        if grid is not None:
            # 세부 점수. 격자가 아슬아슬할 때 **어느 근거가 모자란지** 알려 준다.
            print("    근거  : " + "  ".join(f"{k}={v}"
                                            for k, v in grid.detail.items()))

        out = os.path.join(OUT_DIR, f"win{i}_0x{cand.hwnd:X}.png")
        if imgio.imwrite(out, img):
            print(f"    저장  : {out}")
        print()

        if score >= 0.45 and (best is None or score > best[0]):
            best = (score, cand)

    print("-" * 66)
    if best:
        score, cand = best
        print(f"결론: hwnd=0x{cand.hwnd:X} '{cand.title}' 를 게임 창으로 쓰면 됩니다 "
              f"(격자 {score:.2f}).")
        print("      매크로도 같은 창을 고릅니다. 그냥 실행하시면 됩니다.")
    else:
        print("결론: 5x5 게임판이 보이는 창이 없습니다.")
        print("  1) 게임에서 **탐사 화면**을 띄운 상태인지 확인하세요.")
        print(f"  2) {OUT_DIR} 의 이미지를 열어보세요.")
        print("     - 검은 화면이면: 앱플레이어의 렌더링 방식 때문입니다.")
        print("       앱플레이어 설정에서 그래픽 렌더링을 DirectX <-> OpenGL 로")
        print("       바꾸면 캡처가 되는 경우가 많습니다.")
        print("     - 게임 화면이 제대로 보이는데 격자가 0 이면:")
        print("       config.json 의 explore.board_min 을 0.35 정도로 낮춰보세요.")
    return 0


if __name__ == "__main__":
    _bootstrap.run_main(main)
