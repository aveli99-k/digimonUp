"""기록(trace)을 읽어 **낭비를 세어 준다**.

쓰는 법
    python tools\\analyze_trace.py                    가장 최근 기록
    python tools\\analyze_trace.py debug\\trace\\0816_1
    python tools\\analyze_trace.py --chips            칩 관련만 자세히

답하는 질문
    1. 칩을 놓쳤는가          화면에 있던 칩이 안 먹힌 채 왼쪽으로 사라졌는가
    2. 쓸데없이 부쉈는가      부수지 않고도 갈 수 있었는데 부쉈는가
    3. 헛걸음했는가           전진하지 않는 이동(위/아래/왼쪽)이 얼마나 되는가
    4. 자원을 어디에 썼는가   걸음수·부수기·돌진이 얼마나 줄었는가

판단 기준은 실측으로 확정한 게임 규칙이다(docs/INTERNALS.md 19·25·29장).
    - 플레이어는 0~1열에만 있고, 1열에서 오른쪽을 누르면 판이 한 열 밀린다
    - 전진 한 번에 2열에 있던 것이 플레이어 자리로 들어온다
    - 장애물은 클릭하면 부서진다 (부수기 1, 걸음수 0)
    - 돌진은 세 칸 전진 (돌진 1, 걸음수 0), 지나가는 칩도 먹는다
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digimonup.base import trace                      # noqa: E402

N = 5
ADVANCE_COL = 2
# trace.board_rows() 는 Kind 이름의 첫 글자를 쓴다. OBSTACLE -> "O".
OBSTACLE_CH = "O"


def _walkable(board, r, c):
    return 0 <= r < N and 0 <= c < N and board[r][c] != OBSTACLE_CH


def _reachable_rows(board, start_row):
    """0~1열만 걸어서 닿을 수 있는 행들 (장애물은 못 지나간다)."""
    seen = {(start_row, 0), (start_row, 1)}
    seen = set()
    stack = [(start_row, 1), (start_row, 0)]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or not _walkable(board, r, c):
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < N and 0 <= c + dc <= 1:
                stack.append((r + dr, c + dc))
    return {r for r, c in seen}


def analyze(rows: list[dict]) -> None:
    # **버린 사이클은 빼고 센다.** 플레이어 자리가 말이 안 되면 엔진은 그
    # 사이클을 통째로 버리고 다시 본다(reject 기록이 바로 뒤에 붙는다).
    # 그 판을 진짜 판단으로 세면, 엔진이 이미 걸러낸 것을 두고 "칩을 놓쳤다"고
    # 잘못 세게 된다. 실측 178.3초가 그런 판이었다 — 칩 획득 이펙트가 판을
    # 뒤덮어 플레이어를 (1,0) 으로 읽었고, 엔진은 그걸 버리고 1.6초 뒤에
    # (3,1) 로 바로잡았다.
    cycles = []
    for k, r in enumerate(rows):
        if r["kind"] != "cycle":
            continue
        if k + 1 < len(rows) and rows[k + 1]["kind"] == "reject":
            continue
        cycles.append(r)
    nrej = sum(1 for r in rows if r["kind"] == "reject")
    if nrej:
        print(f"(자리가 말이 안 돼 버린 사이클 {nrej}개는 빼고 셉니다)")
    moves = [r for r in rows if r["kind"] == "move"]
    breaks = [r for r in rows if r["kind"] == "break"]
    dashes = [r for r in rows if r["kind"] == "dash"]
    if not cycles:
        print("기록이 비어 있습니다.")
        return

    span = rows[-1]["t"] - rows[0]["t"]
    ok_moves = [m for m in moves if m["ok"]]
    advances = [m for m in ok_moves if m["scrolled"]]
    print(f"=== {span:.0f}초 / 사이클 {len(cycles)} / 이동 {len(ok_moves)}"
          f"(실패 {len(moves) - len(ok_moves)}) ===")
    print(f"  실제 전진 {len(advances)}회 + 돌진 {len(dashes)}회(={len(dashes) * 3}칸)"
          f"  -> 총 {len(advances) + len(dashes) * 3}칸")

    # --- 0) 실패로 적힌 이동이 정말 실패였는가 -------------------------
    # 확인이 안 됐다고 실패로 적어도, 다음 사이클의 판을 보면 실제로는 움직인
    # 경우가 있다. 그러면 제한시간(2~3.5초)을 통째로 버리고 남은 경로까지
    # 버리므로 손해가 크다. 판이 정말 한 열 밀렸는지로 가른다.
    fails = [m for m in moves if not m["ok"]]
    fake = []
    for m in fails:
        bef = max((c for c in cycles if c["t"] < m["t"]), key=lambda c: c["t"],
                  default=None)
        aft = min((c for c in cycles if c["t"] > m["t"]), key=lambda c: c["t"],
                  default=None)
        if not bef or not aft:
            continue
        shift = sum(bef["board"][r][c + 1] == aft["board"][r][c]
                    for r in range(N) for c in range(N - 1)) / float(N * (N - 1))
        stay = sum(bef["board"][r][c] == aft["board"][r][c]
                   for r in range(N) for c in range(N)) / float(N * N)
        if (m["dir"] == "RIGHT" and shift >= 0.85 and shift > stay) or            (m["dir"] != "RIGHT" and aft.get("player") == m["to"]):
            fake.append(m)
    lost_sec = sum(m.get("secs", 0.0) for m in fake)
    print("")
    print(f"[0] 실패로 적힌 이동 {len(fails)}건 중 "
          f"**사실은 성공 {len(fake)}건**"
          + (f" ({len(fake) / len(fails) * 100:.0f}%)" if fails else ""))
    print(f"    여기서 버린 시간 {lost_sec:.0f}초 "
          f"(전체 {span:.0f}초의 {lost_sec / span * 100:.0f}%)")
    for m in fake[:6]:
        print(f"      {m['t']:6.1f}s {m['dir']:5s} {m['from']}->{m['to']} "
              f"{m.get('secs', 0):.1f}초 버림")

    # --- 1) 칩을 놓쳤는가 ---------------------------------------------
    # 칩 하나를 **판에서 사라질 때까지 끝까지 따라간다.**
    #   전진 한 번에 칩의 열이 하나 줄어든다.
    #   열이 2일 때 전진하면 그 칩이 내 자리로 들어온다(같은 행이어야 한다).
    #   열이 1이나 0으로 내려오면 걸어가서 먹을 수도 있다.
    #   열이 0보다 작아지면 화면 밖으로 나간 것 = 놓친 것이다.
    #
    # 이렇게 해야 "전진 때는 못 먹었지만 그 뒤 걸어가서 먹은" 경우를 놓쳤다고
    # 잘못 세지 않는다.
    got, lost, pending = [], [], 0
    for cy in cycles:
        board, pl = cy["board"], cy.get("player")
        if not pl:
            continue
        after = [m for m in ok_moves if m["t"] > cy["t"]]
        for r, c0 in (cy.get("chips") or []):
            if c0 < ADVANCE_COL:
                continue                    # 이미 걸어서 닿는 자리. 따로 안 센다
            rows_ok = _reachable_rows(board, pl[0])
            col, done = c0, None
            for m in after:
                if m["scrolled"]:
                    if col == ADVANCE_COL and m["from"][0] == r:
                        done = "먹음"       # 전진하면서 먹었다
                        break
                    col -= 1
                    if col < 0:
                        done = "놓침"       # 화면 밖으로 나갔다
                        break
                elif col <= 1 and m["to"] == [r, col]:
                    done = "먹음"           # 걸어가서 먹었다
                    break
            if done is None:
                pending += 1                # 기록이 끝나 결판이 안 났다
            elif done == "먹음":
                got.append((r, c0))
            else:
                lost.append((round(cy["t"], 1), (r, c0), pl[0],
                             r in rows_ok, abs(r - pl[0])))
    total = len(got) + len(lost)
    blocked = [x for x in lost if not x[3]]
    avoidable = [x for x in lost if x[3]]
    print("")
    print(f"[1] 앞쪽(2열 이상) 칩 {total}번 관측 (판정 유보 {pending})")
    print(f"    먹음 {len(got)}회 / 놓침 {len(lost)}회")
    print(f"      - 장애물에 막혀 그 행에 못 감 {len(blocked)}회")
    print(f"      - **갈 수 있었는데 안 감 {len(avoidable)}회**"
          + (f"  ({len(avoidable) / total * 100:.0f}%)" if total else ""))
    for t, ch, pr, _, dist in avoidable[:8]:
        print(f"      {t:6.1f}s  칩{ch}, 나는 {pr}행 (그때 {dist}칸만 움직이면 됐다)")

    # --- 1.5) 유령칩 --------------------------------------------------
    # 한 사이클에 보였다가 **먹지도 않았는데** 다음 사이클에 사라진 칩.
    # 정체는 칩 획득 이펙트다 — 먹은 칩이 상단 보유량으로 날아가는 동안 판
    # 곳곳에 주황 아이콘이 흩어진다(178.3초 화면에서 여섯 칸에 떠 있었다).
    kept = gone = 0
    ghosts = []
    for a, b in zip(cycles, cycles[1:]):
        if not a.get("player") or not b.get("player"):
            continue
        between = [m for m in moves if a["t"] < m["t"] < b["t"]]
        if any(not m["ok"] for m in between):
            continue                      # 무슨 일이 있었는지 확실치 않다
        adv = sum(1 for m in between if m["scrolled"])
        now = {tuple(c) for c in (b.get("chips") or [])}
        for r, c in (a.get("chips") or []):
            nc = c - adv
            if nc < 0:
                continue                  # 화면 밖으로 나갔다
            if nc <= 1 and (r, nc) == tuple(b["player"]):
                continue                  # 걸어가서 먹은 자리
            if adv and nc == 1 and r == b["player"][0]:
                continue                  # 전진하며 먹었다
            if (r, nc) in now:
                kept += 1
            else:
                gone += 1
                ghosts.append((a["t"], (r, c), (r, nc), adv))
    tot = kept + gone
    print("")
    print(f"[1.5] 칩이 다음 사이클에도 그대로 {kept} / **사라짐(유령칩) {gone}**"
          + (f"  ({gone / tot * 100:.0f}%)" if tot else ""))
    for t, was, exp, adv in ghosts[:6]:
        print(f"      {t:6.1f}s  칩{was} 이 {exp} 에 없습니다 (그사이 전진 {adv})")


    # --- 2) 쓸데없이 부쉈는가 -----------------------------------------
    print(f"\n[2] 장애물 파괴 {len(breaks)}회")
    waste = []
    for b in breaks:
        cy = min((c for c in cycles if c["t"] <= b["t"]),
                 key=lambda c: b["t"] - c["t"], default=None)
        if not cy or not cy.get("player"):
            continue
        board = cy["board"]
        pr = cy["player"][0]
        rows_ok = _reachable_rows(board, pr)
        # 부수지 않고 전진할 수 있는 행이 있었는가.
        # **1열에 설 수 있어야** 오른쪽을 누를 수 있다. 예전에는 2열만 보고
        # 판정해서, 정작 발판인 1열이 장애물인 행까지 '갈 수 있었다'고 셌다.
        free = [r for r in rows_ok
                if _walkable(board, r, 1) and _walkable(board, r, ADVANCE_COL)]
        if free:
            cost = min(abs(r - pr) for r in free)
            waste.append((round(b["t"], 1), tuple(b["at"]), pr, sorted(free), cost))
    print(f"    부수지 않고도 갈 수 있었던 경우 {len(waste)}회")
    for t, at, pr, free, cost in waste[:8]:
        print(f"      {t:6.1f}s  {at} 를 부쉈지만 {pr}행에서 {free} 행으로 "
              f"{cost}칸만 가면 됐다")

    # --- 3) 헛걸음 -----------------------------------------------------
    dirs = collections.Counter(m["dir"] for m in ok_moves)
    vert = dirs["UP"] + dirs["DOWN"]
    print(f"\n[3] 이동 방향 {dict(dirs)}")
    if ok_moves:
        print(f"    전진하지 않는 이동 {vert + dirs['LEFT']}회"
              f" = {(vert + dirs['LEFT']) / len(ok_moves) * 100:.0f}%")

    # --- 4) 자원 -------------------------------------------------------
    got = [c["counts"] for c in cycles if c.get("counts")]
    def first_last(key):
        vals = [g[key] for g in got if g.get(key) is not None]
        return (vals[0], vals[-1]) if vals else (None, None)
    print("\n[4] 자원")
    for key, label in (("steps", "걸음수"), ("break", "부수기"), ("dash", "돌진")):
        a, b = first_last(key)
        if a is None:
            print(f"    {label}: 못 읽음")
        else:
            print(f"    {label}: {a} -> {b}  (차이 {a - b})")
    nread = sum(1 for c in cycles if (c.get("counts") or {}).get("steps") is None)
    print(f"    개수를 못 읽은 사이클 {nread}/{len(cycles)}")

    # --- 5) 판단 근거 --------------------------------------------------
    print("\n[5] 판단")
    print(f"    계획: {dict(collections.Counter(c['plan'] for c in cycles))}")
    notes = collections.Counter()
    for c in cycles:
        for n in c.get("notes") or []:
            notes[n.split(" ")[0]] += 1
    if notes:
        print(f"    인식 참고: {dict(notes.most_common(6))}")
    src = collections.Counter()
    for c in cycles:
        note = c.get("player_note") or ""
        src["움직임" if "움직임" in note else
            "전체템플릿" if "전체 템플릿" in note else
            "몸통" if "몸통" in note else
            "색덩어리" if "색 기반" in note else "기타"] += 1
    print(f"    플레이어 검출: {dict(src)}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else None
    rows = trace.load(path)
    if not rows:
        print("기록이 없습니다. config.json 의 explore.trace 를 true 로 두고 돌리세요.")
        return 1
    where = path or trace.latest_dir()
    print(f"기록: {where}\n")
    analyze(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
