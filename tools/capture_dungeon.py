"""던전용 템플릿 캡처 도구.

던전(3번)은 화면 세 가지만 알아보면 된다.

    challenge  하단의 파란 '도전' 버튼   -> 이걸 누른다
    fail       위쪽의 '실패...' 글자     -> 졌을 때. 바깥을 눌러 닫는다
    reward     가운데의 '보상' 글자      -> 이겼을 때. 바깥을 눌러 닫는다

fail 은 **'실패...' 줄만** 잘라야 한다. 그 윗줄은 '던전' / '스테이지' 처럼
그때그때 바뀌는 이름이라, 같이 넣으면 다른 던전에서 못 알아본다.

reward 는 뒤에서 반짝이는 이펙트 때문에 프레임마다 조금씩 다르다. **두세 장을
서로 다른 순간에 찍어두면** 가장 잘 맞는 것으로 판정해서 안정된다.

탐사와 달리 **색만으로는 판단할 수 없어서 템플릿이 반드시 있어야 한다.**
버튼도 글자도 화면 어디에나 있을 수 있는 평범한 UI 라, 색 분포로는 옆에 있는
보라색 '초기화' 버튼과 갈라낼 근거가 없다.

실행:  python tools\\capture_dungeon.py

저장 위치는 templates/dungeon/<종류>/ 이고, 같은 종류에 여러 장 넣어도 된다.
매크로는 그 폴더의 PNG 를 전부 불러서 모두 시도한다.
"""

from __future__ import annotations

import _bootstrap  # 저장소 루트를 sys.path 에 넣는다. 맨 먼저 가져온다


import cropsave  # noqa: E402

from digimonup.base.paths import DUNGEON_TEMPLATE_DIR as OUT_ROOT  # noqa: E402

TARGETS = {
    "1": ("challenge", "던전 화면 하단의 파란 '도전' 버튼"),
    "2": ("fail", "실패창의 '실패...' 줄만 (윗줄은 빼세요)"),
    "3": ("reward", "보상창 가운데의 '보상' 글자 (여러 장 찍으세요)"),
}


def main() -> int:
    return cropsave.run("던전 템플릿 캡처", TARGETS, OUT_ROOT)


if __name__ == "__main__":
    _bootstrap.run_main(main)
