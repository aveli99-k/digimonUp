"""매크로를 돌리면서 **판단 근거를 통째로 기록한다**.

왜 필요한가
    "칩을 무시했다", "길이 있는데 부쉈다" 같은 보고를 받아도, 그 순간의 화면이
    없으면 무엇이 잘못됐는지 알 수 없다. 실제로 이 저장소에서는 그렇게 지목된
    사례를 뒤져 보니 **매크로가 맞았던 적**도 있었고(진짜 칩이었다, 정말 갇혀
    있었다) **진짜 버그였던 적**도 있었다(걸음수 착각, 늦은 칩 판단).
    둘을 가르려면 그 순간의 사진과 판단이 함께 있어야 한다.

쓰는 법
    python tools\\record.py            # 기본 5분
    python tools\\record.py 600        # 10분

    이상한 동작을 보면 그때의 시각(초)을 기억해 두었다가 debug\\record\\ 에서
    그 무렵 파일을 열어 보면 된다. 파일 이름에 시각과 판단이 들어 있다.

무엇이 남는가
    debug\\record\\0012_31.4s_목적지_RIGHT.png   격자·인식·경로를 그린 그림
    debug\\record\\log.txt                       모든 판단을 시각과 함께
"""

from __future__ import annotations

import threading
import time

import _bootstrap  # 저장소 루트를 sys.path 에 넣는다. 맨 먼저 가져온다

import os
import sys

from digimonup.app import explore                       # noqa: E402
from digimonup.base import imgio                        # noqa: E402
from digimonup.base.paths import DEBUG_DIR              # noqa: E402
from digimonup.base.settings import load_explore_config  # noqa: E402
from digimonup.win.emulator_window import enable_dpi_awareness  # noqa: E402

OUT_DIR = os.path.join(DEBUG_DIR, "record")


def main() -> int:
    enable_dpi_awareness()
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = load_explore_config()
    cfg.save_debug = False          # 여기서 직접 남기므로 중복 저장을 끈다

    lines: list[str] = []
    t0 = time.time()

    def log(msg: str) -> None:
        lines.append(f"{time.time() - t0:7.1f}s  {msg}")

    eng = explore.ExploreEngine(cfg, log=log)

    # 사이클마다 오버레이(격자·인식·경로를 그린 그림)를 저장한다.
    #
    # **그림을 그리는 그 자리에서 가로챈다.** 예전에는 plan_route 를 가로채고
    # 엔진이 들고 있던 그림을 꺼내 썼는데, 그때 그것은 아직 **이전 사이클** 것이라
    # 저장된 그림이 한 박자 밀렸다. 그 밀린 그림을 근거로 판단하다가 실제로
    # 잘못된 결론을 냈다.
    # 게다가 지금은 경로를 두 번 세우므로(아이템 없이 한 번, 부수며 한 번)
    # plan_route 가 사이클당 두 번 불려 그림도 두 배로 쌓였다.
    n = [0]
    real_draw = explore.overlay.draw

    def draw_hook(img, grid, scene, path, header, *a, **kw):
        drawn = real_draw(img, grid, scene, path, header, *a, **kw)
        n[0] += 1
        log(f"[기록] #{n[0]:04d} {header} 경로={path}")
        for note in scene.notes:
            log(f"[기록]        {note}")
        safe = header.replace("/", "_").replace("\\", "_").replace("|", "_")[:40]
        name = f"{n[0]:04d}_{time.time() - t0:.1f}s_{safe.strip()}.png"
        imgio.imwrite(os.path.join(OUT_DIR, name), drawn)
        return drawn

    explore.overlay.draw = draw_hook

    print(f"{seconds:.0f}초 동안 기록합니다. 저장 위치: {OUT_DIR}")
    print("이상한 동작을 보면 그때가 몇 초쯤이었는지 기억해 두세요.\n")

    th = threading.Thread(target=eng.run, daemon=True)
    th.start()
    try:
        while th.is_alive() and time.time() - t0 < seconds:
            time.sleep(1.0)
            done = int(time.time() - t0)
            if done % 30 == 0:
                s = eng.stats
                print(f"  {done:4d}초  사이클 {s.cycles} / 이동 {s.moves} / "
                      f"실패 {s.failed_moves} / 안내문 {s.blocked_toasts}")
    except KeyboardInterrupt:
        print("\n중단합니다.")
    eng.stop()
    th.join(timeout=10)

    with open(os.path.join(OUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n그림 {n[0]}장과 log.txt 를 {OUT_DIR} 에 남겼습니다.")
    return 0


if __name__ == "__main__":
    _bootstrap.run_main(main)
