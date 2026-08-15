"""탐사용 템플릿 캡처 도구.

기존 capture.py 는 **전체 모니터**를 캡처하지만, 이 도구는 고정된 MuMuPlayer 창의
**클라이언트 영역**만 캡처한다. 그래야 매크로가 실제로 보는 것과 같은 픽셀에서
템플릿을 잘라낼 수 있다.

실행:  python capture_explore.py
       (또는 capture_explore.bat 더블클릭)

저장 위치는 templates/explore/<종류>/ 이고, 같은 종류에 여러 장을 넣어도 된다.
매크로는 그 폴더의 PNG 를 전부 불러서 모두 시도한다.
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

from emulator_window import capture_client, enable_dpi_awareness, enumerate_candidates

from paths import EXPLORE_TEMPLATE_DIR as OUT_ROOT

TARGETS = {
    "1": ("player", "플레이어 (기본 자세). 여러 장 찍을수록 좋습니다"),
    "2": ("player_body", "플레이어 몸통 중앙 (머리와 발을 뺀 부분)"),
    "3": ("goal", "목적지 / 필수 아이템 카드"),
    "4": ("obstacle", "장애물 (피라미드)"),
    "5": ("item", "일반 아이템"),
    "6": ("top_tab", "상단에 고정된 게임 탭"),
    "7": ("blocked_toast", "'해당 위치로 이동할 수 없습니다' 안내문"),
}


def pick_window():
    cands = enumerate_candidates()
    if not cands:
        print("MuMuPlayer 창을 찾지 못했습니다. 에뮬레이터를 먼저 실행하세요.")
        return None
    if len(cands) == 1:
        return cands[0]
    print("\n여러 창이 있습니다. 번호를 고르세요.")
    for i, c in enumerate(cands):
        print(f"  {i}) hwnd=0x{c.hwnd:X} {c.width}x{c.height} '{c.title}'")
    try:
        return cands[int(input("번호: ").strip())]
    except (ValueError, IndexError):
        return None


def main() -> int:
    enable_dpi_awareness()

    print("=" * 56)
    print(" digimonUp - 탐사 템플릿 캡처")
    print("=" * 56)
    for key, (folder, desc) in TARGETS.items():
        print(f"  {key}) {folder:<14} {desc}")
    choice = input("\n번호 선택: ").strip()
    if choice not in TARGETS:
        print("잘못된 선택입니다.")
        return 1
    folder, desc = TARGETS[choice]

    cand = pick_window()
    if cand is None:
        return 1

    print(f"\n[{desc}] 가 보이도록 게임 화면을 맞춰주세요.")
    for i in range(5, 0, -1):
        print(f"  {i}초 뒤 캡처합니다...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    img = capture_client(cand.hwnd)
    if img is None:
        print("캡처에 실패했습니다.")
        return 1
    print(f"캡처 완료: {img.shape[1]}x{img.shape[0]} (클라이언트 영역)")
    print("\n창이 열리면 영역을 드래그한 뒤 Enter. (취소: c)")

    win = "Drag area then press ENTER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scale = min(900 / img.shape[1], 900 / img.shape[0], 1.0)
    view = (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else img)
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
    crop = img[y:y + h, x:x + w]

    out_dir = os.path.join(OUT_ROOT, folder)
    os.makedirs(out_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(out_dir, f"{folder}_{n:02d}.png")):
        n += 1
    out_path = os.path.join(out_dir, f"{folder}_{n:02d}.png")

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        print("저장 실패")
        return 1
    buf.tofile(out_path)
    print(f"\n저장 완료: {out_path}  ({w}x{h})")
    print("같은 종류를 여러 장 찍어두면 인식률이 올라갑니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
