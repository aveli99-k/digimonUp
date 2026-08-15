"""아이템 개수 읽기용 숫자 템플릿 만들기.

왼쪽 아래의 걸음수 / 부수기 / 돌진 개수를 읽으려면 0~9 숫자 모양이 필요하다.
이 도구는 지금 화면에 보이는 숫자를 그대로 잘라 templates/counters/ 에 저장한다.

    python tools\\capture_counters.py

쓰는 법
    화면에 보이는 세 숫자를 그대로 입력하면, 잘라낸 글자와 짝을 지어 저장한다.
    한 번에 0~9 가 다 나오지는 않으므로, 숫자가 바뀌었을 때 다시 실행하면
    없던 숫자만 채워진다. 이미 있는 숫자는 건드리지 않는다.

왜 이런 방식인가
    OCR 라이브러리를 쓰면 간단하지만 배포본이 EXE 하나여야 해서 넣을 수 없다.
    다행히 이 게임 글꼴은 고정이라 숫자 모양을 한 번만 모아 두면 계속 쓸 수 있다.

숫자 템플릿이 하나도 없어도 매크로는 그냥 돈다. 개수를 못 읽을 뿐이고, 그때는
예전처럼 '해 보고 안 되면 포기하는' 방식으로 움직인다.
"""

from __future__ import annotations

# 이 도구는 tools/ 안에 있지만 루트의 counters 등을 가져다 쓴다.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                    # noqa: E402

import counters                                               # noqa: E402
import imgio                                                  # noqa: E402
from emulator_window import (capture_client,                   # noqa: E402
                             enable_dpi_awareness)
from pickwin import pick_window                                # noqa: E402
from paths import BASE_DIR                                    # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "templates", "counters")



def main() -> int:
    enable_dpi_awareness()
    print("=" * 62)
    print(" digimonUp - 아이템 개수 숫자 템플릿 만들기")
    print("=" * 62)

    have = sorted(counters.load_digits(force=True).keys())
    print(f"이미 가진 숫자: {', '.join(have) if have else '없음'}")
    missing = [d for d in "0123456789" if d not in have]
    print(f"아직 없는 숫자: {', '.join(missing) if missing else '없음 (다 모았습니다)'}")
    if not missing:
        print("\n더 만들 것이 없습니다. 그래도 다시 만들려면 templates/counters 를 비우세요.")
        return 0

    cand = pick_window()
    if cand is None:
        return 1
    img = capture_client(cand.hwnd)
    if img is None:
        print("캡처에 실패했습니다.")
        return 1

    rows = counters.find_rows(img)
    if len(rows) < 3:
        print(f"왼쪽 아래 아이콘 3개를 찾지 못했습니다 (찾은 것 {len(rows)}개).")
        print("게임의 탐사 화면이 떠 있는지 확인하세요.")
        imgio.imwrite(os.path.join(BASE_DIR, "debug", "counters_fail.png"), img)
        print("현재 화면을 debug/counters_fail.png 에 저장했습니다.")
        return 1

    labels = ("걸음수(발자국)", "부수기(노란 발톱)", "돌진(초록)")
    print("\n화면 왼쪽 아래에 보이는 숫자를 그대로 입력하세요.")
    print("쉼표는 있어도 되고 없어도 됩니다. 모르면 그냥 Enter (건너뜀).\n")

    saved: dict[str, str] = {}
    for icon, label in zip(rows, labels):
        glyphs = counters.split_glyphs(counters._strip_of(img, icon))
        if not glyphs:
            print(f"  {label}: 숫자를 찾지 못해 건너뜁니다.")
            continue
        text = input(f"  {label} 는 몇 인가요? (글자 {len(glyphs)}개) : ").strip()
        text = text.replace(",", "").replace(" ", "")
        if not text:
            continue
        if not text.isdigit():
            print("    숫자만 입력해주세요. 건너뜁니다.")
            continue
        if len(text) != len(glyphs):
            print(f"    글자 수가 안 맞습니다 (화면에서 {len(glyphs)}개를 잘랐는데 "
                  f"{len(text)}자를 입력). 건너뜁니다.")
            continue
        os.makedirs(OUT_DIR, exist_ok=True)
        for glyph, ch in zip(glyphs, text):
            if ch in have or ch in saved:
                continue      # 이미 있는 숫자는 덮어쓰지 않는다
            path = os.path.join(OUT_DIR, f"{ch}.png")
            imgio.imwrite(path, cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR))
            saved[ch] = path

    if saved:
        print(f"\n새로 저장한 숫자: {', '.join(sorted(saved))}")
    else:
        print("\n새로 저장한 숫자가 없습니다.")

    still = [d for d in "0123456789"
             if d not in counters.load_digits(force=True)]
    if still:
        print(f"아직 없는 숫자: {', '.join(still)}")
        print("게임을 진행해서 그 숫자가 보일 때 다시 실행하면 채워집니다.")
        print("(다 모으지 않아도 됩니다. 모르는 숫자가 낀 줄만 '모름'으로 둡니다.)")
    else:
        print("0~9 를 모두 모았습니다.")

    now = counters.read(img)
    print(f"\n지금 읽어 보면: {now.describe()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
