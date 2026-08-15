"""버전 번호가 한 곳에만 있고 형식이 맞는지 확인한다."""
from __future__ import annotations

import re

import version


def test_버전_형식이_맞다():
    assert re.fullmatch(r"\d+\.\d+\.\d+", version.__version__), \
        f"X.Y.Z 형식이어야 합니다: {version.__version__}"


def test_한_줄_표기():
    assert version.version_line() == f"digimonUp v{version.__version__}"


def test_버전을_여러_곳에_적어_두지_않는다():
    """실측 교훈: 여러 곳에 적으면 릴리스 때 한두 군데가 빠져,
    사용자가 보는 번호와 실제로 도는 코드가 어긋난다."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    v = version.__version__
    culprits = []
    for path in root.glob("*.py"):
        if path.name == "version.py":
            continue
        if v in path.read_text(encoding="utf-8"):
            culprits.append(path.name)
    assert not culprits, f"버전을 직접 적은 파일이 있습니다: {culprits}"
