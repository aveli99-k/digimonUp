"""Write/Edit 직후 고친 파이썬 파일의 구문만 확인한다 (PostToolUse 훅).

전체 테스트는 30초쯤 걸리므로 배경에서 돌린다(pytest_after_edit.py).
그런데 구문이 깨진 편집은 그 30초를 기다릴 이유가 없다 — 무조건 전부 실패한다.
그래서 문법만 즉시(수십 ms) 보고, 깨졌으면 종료 코드 2 로 바로 되돌린다.

바이트코드는 만들지 않는다. compile() 로 파싱만 하므로 __pycache__ 를
건드리지 않고, 어떤 파일에도 쓰지 않는다.
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def edited_path(payload):
    """훅 입력 JSON 에서 방금 고친 파일 경로를 꺼낸다.

    Write 는 tool_input.file_path, Edit 는 tool_response.filePath 로 오는
    경우가 있어 둘 다 본다.
    """
    tool_input = payload.get("tool_input") or {}
    response = payload.get("tool_response") or {}
    return tool_input.get("file_path") or response.get("filePath") or ""


def main():
    try:
        # sys.stdin 을 그대로 쓰면 안 된다. Windows 파이썬은 stdin 을 로케일
        # 인코딩(여기서는 cp949)으로 읽어서, 저장소 경로의 '준' 같은 글자에
        # UnicodeDecodeError 가 나고 훅이 통과로 빠져버린다.
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        # 훅 입력이 이상하면 조용히 통과시킨다. 검사기가 편집을 막아서는 안 된다.
        return 0

    path = edited_path(payload)
    if not path.endswith(".py") or not os.path.isfile(path):
        return 0

    try:
        with open(path, "rb") as f:
            source = f.read()
    except OSError:
        return 0

    try:
        # bytes 를 넘기면 파일 안의 coding 선언과 BOM 을 파이썬이 알아서 처리한다.
        compile(source, path, "exec")
    except SyntaxError as err:
        where = f"{os.path.basename(path)}:{err.lineno}"
        print(f"구문 오류 {where} — {err.msg}")
        if err.text:
            print(f"    {err.text.rstrip()}")
        return 2
    except ValueError as err:
        # 널 바이트 등 compile() 이 SyntaxError 로 감싸지 않는 경우.
        print(f"구문 오류 {os.path.basename(path)} — {err}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
