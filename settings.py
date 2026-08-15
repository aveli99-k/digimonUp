"""config.json 을 읽어 각 기능의 설정 객체로 만들어 준다."""

from __future__ import annotations

import json

from paths import CONFIG_PATH


def load_raw() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_explore_config():
    """config.json 의 "explore" 절을 ExploreConfig 로 바꾼다.

    없는 키는 ExploreConfig 의 기본값을 그대로 쓴다.
    """
    from explore import ExploreConfig

    cfg = ExploreConfig()
    section = load_raw().get("explore", {})
    for key, value in section.items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg
