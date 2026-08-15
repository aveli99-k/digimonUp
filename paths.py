"""실행 위치 기준 경로.

PyInstaller 로 단일 EXE 를 만들면 __file__ 은 임시 압축 해제 폴더(sys._MEIPASS)를
가리킨다. 하지만 templates/ 와 config.json 은 **EXE 옆에** 두고 사용자가 자유롭게
바꿀 수 있어야 하므로, 그 경우에는 EXE 가 있는 폴더를 기준으로 삼는다.
"""

from __future__ import annotations

import os
import sys


def app_dir() -> str:
    """설정과 템플릿을 찾을 기준 폴더."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_dir()
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
EXPLORE_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "explore")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
