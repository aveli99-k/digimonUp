"""config.json 을 읽어 각 기능의 설정 객체로 만들어 준다."""

from __future__ import annotations

import json

from digimonup.base.paths import CONFIG_PATH


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
    from digimonup.app.explore import ExploreConfig

    cfg = ExploreConfig()
    raw = load_raw()
    # 중지 키는 두 기능이 함께 쓰므로 최상위 stop_key 를 기본으로 삼는다.
    # explore 절에 따로 적으면 그것이 우선한다.
    if raw.get("stop_key"):
        cfg.stop_key = str(raw["stop_key"])
    section = raw.get("explore", {})
    for key, value in section.items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg
