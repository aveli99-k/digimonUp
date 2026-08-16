"""tools/ 의 도구들이 **import 만으로는 죽지 않는지** 확인한다.

도구는 게임을 띄워 놓고 손으로 돌리는 것이라 아무도 테스트하지 않았고, 그래서
조용히 썩었다. 실제로 있었던 일:

  - 10개 파일이 저장소 루트를 sys.path 에 넣는 네 줄을 각자 복사해 갖고 있었고,
    그중 둘은 그 뒤에 `import os, sys` 를 한 번 더 적고 있었다.
  - check.py 는 그 혼란 속에서 imwrite 를 아예 가져오지 않은 채 마지막 줄에서
    불렀다. 화면을 캡처하고 유사도를 다 계산한 **뒤에** NameError 로 죽었다.

여기서 잡히는 것은 import 시점의 사고뿐이다(경로 부트스트랩, 모듈 이름 오타).
check.py 처럼 함수 안에서 나는 NameError 는 `python -m ruff check --select F`
가 잡는다 — requirements-dev.txt 참고.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"
NAMES = sorted(p.stem for p in TOOLS.glob("*.py"))


@pytest.fixture(scope="module", autouse=True)
def tools_on_path():
    """도구를 `python tools/x.py` 로 돌릴 때와 같은 sys.path 를 만든다.

    그때는 tools/ 가 sys.path[0] 이라 `import cropsave` 같은 이웃 import 가
    된다. 저장소 루트는 각 도구가 _bootstrap 으로 직접 넣는다.
    """
    here = str(TOOLS)
    added = here not in sys.path
    if added:
        sys.path.insert(0, here)
    yield
    if added:
        sys.path.remove(here)


def test_도구_목록이_비어_있지_않다():
    assert NAMES, "tools/ 에서 파이썬 파일을 하나도 못 찾았습니다"
    assert "_bootstrap" in NAMES


@pytest.mark.parametrize("name", NAMES)
def test_도구가_import_만으로_죽지_않는다(name):
    mod = importlib.import_module(name)
    assert mod is not None


def test_모든_도구가_경로_부트스트랩을_한_벌만_쓴다():
    """복사본이 늘면 뒤따르는 것들도 함께 늘어난다. 한 벌로 유지한다."""
    culprits = []
    for path in TOOLS.glob("*.py"):
        if path.name == "_bootstrap.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "digimonup" not in text:
            continue           # 저장소 코드를 안 쓰는 도구는 부트스트랩도 불필요
        if "sys.path.insert" in text:
            culprits.append(path.name + " (직접 넣고 있습니다)")
        elif "import _bootstrap" not in text:
            culprits.append(path.name + " (_bootstrap 을 안 씁니다)")
    assert not culprits, culprits


def test_부트스트랩은_루트를_두_번_넣지_않는다():
    import _bootstrap

    before = list(sys.path)
    importlib.reload(_bootstrap)
    assert sys.path.count(_bootstrap.ROOT) == before.count(_bootstrap.ROOT)
    assert os.path.isdir(os.path.join(_bootstrap.ROOT, "digimonup"))
