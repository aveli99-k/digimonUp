"""인식 점검 도구 (클릭하지 않음).

현재 화면에서 '매칭'/'포기' 버튼이 얼마나 잘 인식되는지 유사도를 보여주고,
찾은 위치를 표시한 이미지를 check_result.png 로 저장한다.
"""

from __future__ import annotations

import os
import sys
import time

import cv2

from common import (
    BASE_DIR,
    enable_dpi_awareness,
    ensure_windows,
    find_template,
    grab_screen,
    imwrite_unicode,
    load_config,
    load_template,
)


def main() -> int:
    ensure_windows()
    enable_dpi_awareness()
    cfg = load_config()

    try:
        tpl_match = load_template(cfg["match_template"])
        tpl_giveup = load_template(cfg["giveup_template"])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1

    print("=" * 52)
    print(" digimonUp - 인식 점검 (클릭하지 않습니다)")
    print("=" * 52)
    print(f"  매칭 템플릿 : {tpl_match.shape[1]}x{tpl_match.shape[0]}")
    print(f"  포기 템플릿 : {tpl_giveup.shape[1]}x{tpl_giveup.shape[0]}")
    print(f"  기준 유사도 : {cfg['confidence']}")

    print("\n확인하려는 화면을 띄워주세요.")
    for i in range(5, 0, -1):
        print(f"  {i}초 뒤 캡처합니다...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    region = cfg.get("region")
    screen = grab_screen(region)
    print(f"캡처 완료: {screen.shape[1]}x{screen.shape[0]}\n")

    conf = float(cfg["confidence"])
    scales = cfg.get("scales")
    ms = bool(cfg.get("multi_scale"))

    annotated = screen.copy()
    for label, tpl, color in (
        ("매칭", tpl_match, (0, 255, 0)),
        ("포기", tpl_giveup, (0, 165, 255)),
    ):
        found, pt, score = find_template(screen, tpl, conf, ms, scales)
        mark = "찾음  " if found else "못찾음"
        print(f"  [{label}] {mark}  최고 유사도 {score:.3f}"
              + (f"  위치 {pt}" if pt else ""))
        if not found and score >= 0.6:
            print(f"         -> 아깝습니다. config.json 의 confidence 를 "
                  f"{max(0.6, round(score - 0.05, 2))} 로 낮춰보세요.")
        if found and pt:
            h, w = tpl.shape[:2]
            cv2.rectangle(annotated, (pt[0] - w // 2, pt[1] - h // 2),
                          (pt[0] + w // 2, pt[1] + h // 2), color, 3)
            cv2.circle(annotated, pt, 5, color, -1)

    out = os.path.join(BASE_DIR, "check_result.png")
    imwrite_unicode(out, annotated)
    print(f"\n표시된 이미지 저장: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
