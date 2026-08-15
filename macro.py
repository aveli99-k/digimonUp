"""1번 기능(네트워크) 콘솔 실행용 진입점.

실제 동작은 network_macro.NetworkMacro 에 있다. GUI 와 콘솔이 같은 코드를 쓰도록
여기서는 로그 파일 기록만 붙여서 실행한다.

중지: F12 (config.json 의 stop_key) 또는 Ctrl+C,
      비상 정지는 마우스를 화면 왼쪽 위 모서리로 밀면 된다(PyAutoGUI failsafe).
"""

from __future__ import annotations

import sys
from datetime import datetime

from common import ensure_windows, load_config
from network_macro import NetworkMacro


def main() -> int:
    ensure_windows()
    cfg = load_config()

    log_fp = None
    if cfg.get("log_file"):
        log_fp = open(cfg["log_file"], "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        if log_fp:
            log_fp.write(line + "\n")
            log_fp.flush()

    macro = NetworkMacro(log=log)
    try:
        return macro.run()
    except KeyboardInterrupt:
        macro.stop()
        log("Ctrl+C 로 중단되었습니다.")
        return 0
    finally:
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    sys.exit(main())
