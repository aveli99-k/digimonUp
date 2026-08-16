"""실행 위치 기준 경로.

**EXE 하나만 있어도 돌아가고, 옆에 파일을 두면 그쪽을 쓴다.**

config.json 과 templates/ 는 두 곳에 있을 수 있다.

  1. EXE(또는 소스) 옆        <- 사용자가 직접 고친 것. 있으면 **이쪽이 우선**
  2. EXE 안에 넣어 둔 기본값  <- PyInstaller 가 sys._MEIPASS 에 풀어 놓는다

이렇게 한 이유:
  - 받는 사람은 EXE 파일 하나만 받아 더블클릭하면 바로 쓸 수 있어야 한다.
    (기본 설정과 템플릿이 EXE 안에 들어 있다)
  - 그러면서도 템플릿을 자기 화면에서 다시 찍거나 설정을 고치고 싶은 사람은
    EXE 옆에 config.json / templates 폴더를 두기만 하면 된다. 다시 빌드할 필요가 없다.

debug/ 와 로그처럼 **쓰는** 파일은 항상 EXE 옆(BASE_DIR)이다.
sys._MEIPASS 는 프로그램이 끝나면 지워지는 임시 폴더라 거기에 쓰면 남지 않는다.
"""

from __future__ import annotations

import os
import sys


# 이 파일에서 저장소 루트까지 올라가야 하는 깊이 (digimonup/base/paths.py).
# 파일을 옮기면 여기도 함께 고쳐야 config.json 과 templates/ 를 찾는다.
_DEPTH_FROM_ROOT = 3


def app_dir() -> str:
    """EXE(또는 소스)가 있는 폴더. 사용자가 파일을 두는 곳이자, 쓰기의 기준."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    here = os.path.abspath(__file__)
    for _ in range(_DEPTH_FROM_ROOT):
        here = os.path.dirname(here)
    return here


def bundle_dir() -> str:
    """EXE 안에 함께 넣어 둔 파일이 풀리는 폴더. 빌드본이 아니면 소스 폴더와 같다."""
    return getattr(sys, "_MEIPASS", app_dir())


BASE_DIR = app_dir()
BUNDLE_DIR = bundle_dir()


def resource(*parts: str) -> str:
    """읽기 전용 자원(설정/템플릿)의 실제 경로.

    EXE 옆에 있으면 그것을, 없으면 EXE 안에 넣어 둔 기본값을 돌려준다.
    """
    external = os.path.join(BASE_DIR, *parts)
    if os.path.exists(external):
        return external
    return os.path.join(BUNDLE_DIR, *parts)


TEMPLATE_DIR = resource("templates")
EXPLORE_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "explore")
DUNGEON_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "dungeon")
CONFIG_PATH = resource("config.json")

# 쓰는 폴더는 항상 EXE 옆이다 (임시 폴더에 쓰면 프로그램이 끝날 때 사라진다).
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
