"""탐사용 템플릿 캡처 도구.

capture.py 는 **전체 모니터**를 캡처하지만, 이 도구는 고정된 앱플레이어 창의
**클라이언트 영역**만 캡처한다. 그래야 매크로가 실제로 보는 것과 같은 픽셀에서
템플릿을 잘라낼 수 있다.

실행:  python tools\\capture_explore.py

저장 위치는 templates/explore/<종류>/ 이고, 같은 종류에 여러 장을 넣어도 된다.
매크로는 그 폴더의 PNG 를 전부 불러서 모두 시도한다.

고르기·캡처·자르기·저장은 던전 캡처 도구와 똑같아서 cropsave 에 한 벌만 둔다.
"""

from __future__ import annotations

# 이 도구는 tools/ 안에 있지만 루트의 모듈을 가져다 쓴다. 실행 방식에 상관없이
# import 가 되도록 루트를 sys.path 에 직접 넣는다. 다른 import 보다 먼저 와야 한다.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cropsave  # noqa: E402

from digimonup.base.paths import EXPLORE_TEMPLATE_DIR as OUT_ROOT  # noqa: E402

TARGETS = {
    "1": ("player", "플레이어 (기본 자세). 여러 장 찍을수록 좋습니다"),
    "2": ("player_body", "플레이어 몸통 중앙 (머리와 발을 뺀 부분)"),
    "3": ("goal", "목적지 / 필수 아이템 카드"),
    "4": ("obstacle", "장애물 (피라미드)"),
    "5": ("item", "일반 아이템"),
    "6": ("top_tab", "상단에 고정된 게임 탭"),
    "7": ("blocked_toast", "'해당 위치로 이동할 수 없습니다' 안내문"),
}


def main() -> int:
    return cropsave.run("탐사 템플릿 캡처", TARGETS, OUT_ROOT)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
