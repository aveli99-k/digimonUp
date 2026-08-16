"""버튼 템플릿 캡처 도구.

실행하면 5초 뒤 전체 화면을 캡처하고, 마우스로 버튼 영역을 드래그해서 잘라 저장한다.
  1) '매칭' 버튼이 보이는 상태에서 실행 -> matching.png 저장
  2) '포기' 버튼이 보이는 상태에서 실행 -> giveup.png 저장
"""

from __future__ import annotations

import _bootstrap  # 저장소 루트를 sys.path 에 넣는다. 맨 먼저 가져온다

import os
import time


import cv2

import cropsave

from digimonup.base.common import BASE_DIR, enable_dpi_awareness, ensure_windows, grab_screen
from digimonup.base.imgio import imwrite

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

    # 축소해서 보여주고 좌표를 원본으로 되돌리는 일은 cropsave 가 한 벌로 맡는다.
    # 전체 모니터라 창을 더 크게(1280x720) 잡는다.
    crop = cropsave.crop_by_drag(screen, "Select button area - drag then press ENTER",
                                 max_w=1280, max_h=720)
    if crop is None:
        print("선택이 취소되었습니다.")
        return 1
    h, w = crop.shape[:2]

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
    _bootstrap.run_main(main)
