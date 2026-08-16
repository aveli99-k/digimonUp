"""도구들이 저장소 루트를 import 할 수 있게 해 준다. **맨 먼저 가져온다.**

    import _bootstrap  # noqa: F401

tools/ 안의 스크립트는 `digimonup.*` 를 가져다 쓰는데, `python tools/x.py` 로
실행하면 sys.path 에 들어가는 것은 tools/ 뿐이라 루트가 안 보인다. 그래서
스크립트마다 아래 네 줄이 통째로 복사돼 있었다(10개 파일).

    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

사본이 늘면 뒤따르는 것들도 함께 늘어난다. 실제로 check.py 와 capture.py 는
이 네 줄 뒤에 `import os, sys` 를 한 번 더 적고 있었고, check.py 는 그 혼란
속에서 imwrite 를 아예 가져오지 않은 채 마지막 줄에서 부르고 있었다
(도구가 할 일을 다 하고 저장할 때 NameError 로 죽었다).

여기 한 벌만 두고 각 도구는 한 줄로 가져온다. 같은 경로를 두 번 넣지 않는다.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_main(main) -> None:
    """도구의 진입점. Ctrl+C 를 조용히 받아 준다.

        if __name__ == "__main__":
            _bootstrap.run_main(main)

    이 여섯 줄도 도구마다 복사돼 있었다. 도구는 손으로 돌리는 것이라 Ctrl+C 로
    끊는 일이 흔한데, 빠뜨린 도구만 파이썬 역추적이 통째로 쏟아진다.
    """
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
        sys.exit(130)
