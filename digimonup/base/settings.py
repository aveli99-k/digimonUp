"""config.json 을 읽어 각 기능의 설정 객체로 만들어 준다."""

from __future__ import annotations

import json

from digimonup.base.paths import CONFIG_PATH


def load_raw() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _fill(cfg, section_name: str):
    """config.json 의 한 절을 설정 객체에 덮어쓴다.

    없는 키는 설정 객체의 기본값을 그대로 둔다. 값은 기본값의 타입으로 맞춘다
    (JSON 의 리스트는 tuple 필드에 tuple 로 들어간다).
    """
    raw = load_raw()
    # 중지 키는 기능들이 함께 쓰므로 최상위 stop_key 를 기본으로 삼는다.
    # 각 절에 따로 적으면 그것이 우선한다.
    if raw.get("stop_key") and hasattr(cfg, "stop_key"):
        cfg.stop_key = str(raw["stop_key"])
    for key, value in raw.get(section_name, {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg


def load_explore_config():
    """config.json 의 "explore" 절을 ExploreConfig 로 바꾼다."""
    from digimonup.app.explore import ExploreConfig

    return _fill(ExploreConfig(), "explore")


def load_dungeon_config():
    """config.json 의 "dungeon" 절을 DungeonConfig 로 바꾼다."""
    from digimonup.app.dungeon import DungeonConfig

    return _fill(DungeonConfig(), "dungeon")
