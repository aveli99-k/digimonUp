"""버튼 템플릿 캡처 도구.

실행하면 5초 뒤 전체 화면을 캡처하고, 마우스로 버튼 영역을 드래그해서 잘라 저장한다.
  1) '매칭' 버튼이 보이는 상태에서 실행 -> matching.png 저장
  2) '포기' 버튼이 보이는 상태에서 실행 -> giveup.png 저장
"""

from __future__ import annotations

# 이 도구는 tools/ 안에 있지만 루트의 common / emulator_window 등을 가져다 쓴다.
# 실행 방식(python tools/x.py, 배치 파일, IDE)에 상관없이 import 가 되도록
# 루트를 sys.path 에 직접 넣는다. 다른 import 보다 먼저 와야 한다.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import time

import cv2

from common import BASE_DIR, enable_dpi_awareness, ensure_windows, grab_screen
from imgio import imwrite

TARGETS = {
    "1": ("매칭 버튼", "matching.png"),
    "2": ("포기 버튼", "giveup.png"),
}


def main() -> int:
    ensure_windows()
    enable_dpi_awareness()

    print("=" * 52)
    print(" digimonUp - 버튼 템플릿 캡처")
    print("=" * 52)
    print("  1) 매칭 버튼 캡처")
    print("  2) 포기 버튼 캡처")
    choice = input("\n번호 선택 (1/2): ").strip()
    if choice not in TARGETS:
        print("잘못된 선택입니다.")
        return 1

    label, filename = TARGETS[choice]
    print(f"\n[{label}] 가 화면에 보이도록 게임 창을 띄워주세요.")
    for i in range(5, 0, -1):
        print(f"  {i}초 뒤 화면을 캡처합니다...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    screen = grab_screen()
    print(f"화면 캡처 완료: {screen.shape[1]}x{screen.shape[0]}")
    print("\n창이 열리면 버튼 영역을 마우스로 드래그한 뒤 Enter 를 누르세요. (취소: c)")

    win = "Select button area - drag then press ENTER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 화면이 커도 창이 잘리지 않도록 축소해서 보여주고, 좌표는 원본 기준으로 되돌린다.
    max_w, max_h = 1280, 720
    scale = min(max_w / screen.shape[1], max_h / screen.shape[0], 1.0)
    view = cv2.resize(screen, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else screen
    cv2.resizeWindow(win, view.shape[1], view.shape[0])

    roi = cv2.selectROI(win, view, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("선택이 취소되었습니다.")
        return 1

    inv = 1.0 / scale
    x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
    crop = screen[y:y + h, x:x + w]

    out_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    if not imwrite(out_path, crop):
        print(f"저장 실패: {out_path}")
        return 1

    print(f"\n저장 완료: {out_path}  ({w}x{h})")
    print("저장된 이미지를 확인하려면 아무 키나 누르세요. (창은 3초 뒤 자동으로 닫힙니다)")
    cv2.imshow("saved template", crop)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
