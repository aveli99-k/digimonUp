"""캡처 도구들이 함께 쓰는 '창에서 잘라 저장하기'.

pickwin.py 와 같은 이유로 뽑아냈다. capture_explore.py 의 본문은 저장 폴더와
목록만 빼면 통째로 같아서, 던전용을 새로 만들면 백 줄짜리 사본이 하나 더
생긴다. 사본이 생기면 한쪽만 고쳐지고 다른 쪽은 조용히 낡는다.

여기서 하는 일
    1) 앱플레이어 창을 고른다 (pickwin)
    2) 몇 초 세고 **클라이언트 영역**만 캡처한다
       - 전체 모니터가 아니라 창만 찍어야 매크로가 실제로 보는 픽셀과 같다
    3) 마우스로 영역을 드래그해 잘라 <폴더>/<이름>_NN.png 로 저장한다
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (저장소 루트를 sys.path 에 넣는다. 맨 먼저)

import os
import time


import cv2  # noqa: E402

from digimonup.base.imgio import imwrite  # noqa: E402
from digimonup.win.emulator_window import (capture_client,  # noqa: E402
                                           enable_dpi_awareness)
from pickwin import pick_window  # noqa: E402


def choose(title: str, targets: dict[str, tuple[str, str]]):
    """메뉴를 찍고 고르게 한다. 반환: (폴더 이름, 설명) 또는 None."""
    print("=" * 56)
    print(f" digimonUp - {title}")
    print("=" * 56)
    for key, (folder, desc) in targets.items():
        print(f"  {key}) {folder:<14} {desc}")
    choice = input("\n번호 선택: ").strip()
    if choice not in targets:
        print("잘못된 선택입니다.")
        return None
    return targets[choice]


def crop_by_drag(img, win_title: str = "Drag area then press ENTER",
                 max_w: int = 900, max_h: int = 900):
    """창을 띄워 마우스로 영역을 드래그하게 하고 그 부분을 잘라 준다.

    화면이 커도 창이 잘리지 않도록 **축소해서 보여주고, 좌표는 원본 기준으로
    되돌린다.** 이 되돌리기를 빠뜨리면 엉뚱한 데가 잘리는데 그림만 보고는
    알아채기 어렵다. 그래서 한 벌만 둔다(capture.py 와 여기 두 벌이었다).

    취소하거나 폭·높이가 0 이면 None.
    """
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    scale = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)
    view = (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else img)
    cv2.resizeWindow(win_title, view.shape[1], view.shape[0])
    roi = cv2.selectROI(win_title, view, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    inv = 1.0 / scale
    x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
    return img[y:y + h, x:x + w]


def capture_and_crop(desc: str, out_root: str, folder: str,
                     countdown: int = 5) -> int:
    """세고 -> 창을 캡처하고 -> 드래그한 영역을 저장한다. 종료 코드를 돌려준다."""
    cand = pick_window()
    if cand is None:
        return 1

    print(f"\n[{desc}] 가 보이도록 게임 화면을 맞춰주세요.")
    for i in range(countdown, 0, -1):
        print(f"  {i}초 뒤 캡처합니다...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    img = capture_client(cand.hwnd)
    if img is None:
        print("캡처에 실패했습니다.")
        return 1
    print(f"캡처 완료: {img.shape[1]}x{img.shape[0]} (클라이언트 영역)")
    print("\n창이 열리면 영역을 드래그한 뒤 Enter. (취소: c)")

    crop = crop_by_drag(img)
    if crop is None:
        print("선택이 취소되었습니다.")
        return 1
    h, w = crop.shape[:2]

    out_dir = os.path.join(out_root, folder)
    n = 1
    while os.path.exists(os.path.join(out_dir, f"{folder}_{n:02d}.png")):
        n += 1
    out_path = os.path.join(out_dir, f"{folder}_{n:02d}.png")

    if not imwrite(out_path, crop):
        print("저장 실패")
        return 1
    print(f"\n저장 완료: {out_path}  ({w}x{h})")
    print("같은 종류를 여러 장 찍어두면 인식률이 올라갑니다.")
    return 0


def run(title: str, targets: dict[str, tuple[str, str]], out_root: str) -> int:
    """메뉴 -> 캡처 -> 저장까지 한 번에. 캡처 도구의 main() 이 이걸 부른다."""
    enable_dpi_awareness()
    picked = choose(title, targets)
    if picked is None:
        return 1
    folder, desc = picked
    return capture_and_crop(desc, out_root, folder)
