"""코드를 고치면 배경에서 전체 테스트를 돌린다 (PostToolUse 훅).

이 저장소의 회귀 테스트는 '같은 함정을 다시 밟지 않으려고' 남긴 것이라
(docs/INTERNALS.md), 고치자마자 돌려봐야 값어치가 있다. 다만 221개가
30초쯤 걸려서 편집마다 동기로 기다리면 작업이 끊긴다.

그래서 asyncRewake 로 배경 실행한다 — 통과하면 아무 일도 없고, 실패했을
때만 종료 코드 2 로 Claude 를 깨운다.

편집이 연달아 들어오면 30초짜리가 겹쳐 쌓이므로, 잠금 파일 하나로 한 번에
하나만 돌게 막는다.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# .claude/hooks/pytest_after_edit.py -> 저장소 뿌리
ROOT = Path(__file__).resolve().parents[2]

# 이 아래의 .py 를 고쳤을 때만 돌린다. README.md 나 config.json 편집으로
# 30초를 태울 이유가 없다.
WATCHED = ("digimonup", "tools", "tests")

LOCK = ROOT / ".pytest_cache" / "claude-hook.lock"
LOCK_STALE_SEC = 600  # 훅이 중간에 죽어 남은 잠금은 10분 뒤 무시한다
TAIL_LINES = 40


def edited_path(payload):
    tool_input = payload.get("tool_input") or {}
    response = payload.get("tool_response") or {}
    return tool_input.get("file_path") or response.get("filePath") or ""


def is_watched(path):
    if not path.endswith(".py"):
        return False
    try:
        rel = Path(path).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return False
    return rel.parts and rel.parts[0] in WATCHED


def acquire_lock():
    """이미 도는 중이면 False. 성공하면 True 이고 호출자가 풀어야 한다."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        age = time.time() - LOCK.stat().st_mtime
        if age > LOCK_STALE_SEC:
            LOCK.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        # 잠금을 못 만드는 상황이면 잠금 없이라도 돌린다.
        return True
    os.close(fd)
    return True


def tail(text):
    lines = text.strip().splitlines()
    return "\n".join(lines[-TAIL_LINES:])


def main():
    try:
        # Windows 파이썬은 stdin 을 로케일 인코딩으로 읽는다. 저장소 경로에
        # '준' 이 들어 있어 그대로 두면 디코딩이 깨지고 훅이 통과로 빠진다.
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        return 0

    if not is_watched(edited_path(payload)):
        return 0

    if not acquire_lock():
        # 앞선 편집이 돌린 테스트가 아직 진행 중이다. 그 결과가 곧 나온다.
        return 0

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as err:
        print(f"테스트를 실행하지 못했다: {err}")
        return 2
    finally:
        LOCK.unlink(missing_ok=True)

    if result.returncode == 0:
        return 0

    print("편집 후 자동 실행한 테스트가 실패했다 (python -m pytest tests -q)")
    print(tail(result.stdout or result.stderr))
    return 2


if __name__ == "__main__":
    sys.exit(main())
