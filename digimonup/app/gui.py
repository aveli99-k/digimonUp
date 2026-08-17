"""digimonUp 매크로 GUI.

더블클릭 한 번으로 실행되고, 시작할 때 기능 번호를 고른다.
    1) 네트워크  - 기존 매칭/포기 자동 클릭
    2) 탐사      - 5x5 게임판 경로 자동 이동
    3) 던전      - 도전/토벌하기 자동 클릭, 실패창 자동 닫기

GUI 가 보여주는 것
    - 시작 / 정지 버튼
    - 고정된 HWND
    - 프로그램 상태
    - 후보 MuMuPlayer 평가 결과
    - 플레이어와 게임판 인식 로그
    - 계산된 이동 경로
    - 실제 클릭 좌표
    - 디버그 오버레이 미리보기
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from datetime import datetime

import tkinter as tk
from tkinter import ttk

from digimonup.base.version import __version__, version_line
from digimonup.win.single_instance import GUI_TITLE

# 제목 표시줄. **앞부분은 GUI_TITLE 그대로여야 한다.** 두 번째 실행이 떠 있는
# 창을 찾을 때 이 앞부분으로 찾기 때문이다(single_instance.find_window).
WINDOW_TITLE = f"{GUI_TITLE}  v{__version__}"

MODES = [
    ("1", "네트워크", "매칭 버튼을 누르고, 포기 버튼이 뜨면 잠시 뒤 눌러 반복합니다."),
    ("2", "탐사", "5x5 게임판을 인식해 장애물을 피해 한 칸씩 이동합니다."),
    ("3", "던전", "도전·토벌하기 버튼과 실패창을 함께 찾아, 보이는 쪽을 한 번씩 눌러 반복합니다."),
]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(WINDOW_TITLE)
        root.geometry("1020x720")
        root.minsize(860, 600)
        self._set_icon(root)

        self.mode = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="대기 중")
        self.hwnd_var = tk.StringVar(value="-")
        self.msgq: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.engine = None
        self._preview_img = None      # PhotoImage 참조 유지용 (GC 방지)

        self._build()
        self.root.after(60, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _set_icon(root: tk.Tk) -> None:
        """제목 표시줄과 작업 표시줄 아이콘. 없거나 실패해도 그냥 넘어간다."""
        try:
            from digimonup.base.paths import resource
            path = resource("assets", "icon.ico")
            if os.path.exists(path):
                root.iconbitmap(path)
        except Exception:
            pass      # 아이콘은 없어도 동작에는 아무 지장이 없다

    # ------------------------------------------------------------- 화면 구성
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="기능 선택", font=("Malgun Gothic", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        for i, (num, name, _desc) in enumerate(MODES):
            ttk.Radiobutton(top, text=f"{num}) {name}", value=num,
                            variable=self.mode).grid(row=0, column=1 + i, sticky="w",
                                                     padx=(0, 14))

        self.desc_var = tk.StringVar()
        ttk.Label(top, textvariable=self.desc_var, foreground="#555").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 8))
        self.mode.trace_add("write", lambda *_: self._update_desc())
        self._update_desc()

        self.start_btn = ttk.Button(top, text="시작", width=12, command=self.start)
        self.start_btn.grid(row=0, column=5, rowspan=2, padx=4)
        self.stop_btn = ttk.Button(top, text="정지", width=12, command=self.stop,
                                   state="disabled")
        self.stop_btn.grid(row=0, column=6, rowspan=2, padx=4)

        info = ttk.Frame(self.root, padding=(10, 0))
        info.pack(fill="x")
        ttk.Label(info, text="상태:").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.status_var,
                  font=("Malgun Gothic", 10, "bold")).grid(row=0, column=1, sticky="w",
                                                           padx=(4, 24))
        ttk.Label(info, text="고정된 HWND:").grid(row=0, column=2, sticky="w")
        ttk.Label(info, textvariable=self.hwnd_var,
                  font=("Consolas", 10)).grid(row=0, column=3, sticky="w", padx=4)
        # 버전은 오른쪽 끝에 붙여 둔다. 문제를 알려 줄 때 어느 판인지 바로
        # 확인할 수 있어야 하고, 제목 표시줄은 창을 최대화하면 잘 안 보인다.
        info.columnconfigure(4, weight=1)
        ttk.Label(info, text=f"v{__version__}", foreground="#888").grid(
            row=0, column=4, sticky="e")

        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Labelframe(pane, text="로그 (창 평가 / 인식 / 경로 / 클릭 좌표)")
        self.text = tk.Text(left, wrap="none", font=("Consolas", 9), height=20)
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        pane.add(left, weight=3)

        right = ttk.Labelframe(pane, text="디버그 오버레이 미리보기")
        self.canvas = tk.Label(right, background="#202020")
        self.canvas.pack(fill="both", expand=True)
        pane.add(right, weight=2)

    def _update_desc(self) -> None:
        for num, _name, desc in MODES:
            if num == self.mode.get():
                self.desc_var.set(desc)

    # ----------------------------------------------------------- 로그/미리보기
    def log(self, msg: str) -> None:
        self.msgq.put(("log", msg))

    def status(self, msg: str) -> None:
        self.msgq.put(("status", msg))

    def preview(self, img) -> None:
        self.msgq.put(("preview", img))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.msgq.get_nowait()
                if kind == "log":
                    self.text.insert("end", f"[{datetime.now():%H:%M:%S}] {payload}\n")
                    self.text.see("end")
                    if float(self.text.index("end-1c").split(".")[0]) > 2000:
                        self.text.delete("1.0", "500.0")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "hwnd":
                    self.hwnd_var.set(payload)
                elif kind == "preview":
                    self._show_preview(payload)
                elif kind == "done":
                    self._on_worker_done()
        except queue.Empty:
            pass
        self.root.after(60, self._drain)

    def _show_preview(self, img) -> None:
        """OpenCV BGR 이미지를 Tk 라벨에 띄운다 (PIL 없이 PPM 으로 변환)."""
        try:
            import cv2
            h, w = img.shape[:2]
            box_w = max(120, self.canvas.winfo_width())
            box_h = max(120, self.canvas.winfo_height())
            k = min(box_w / w, box_h / h, 1.0)
            if k < 1.0:
                img = cv2.resize(img, (int(w * k), int(h * k)),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".ppm", img)
            if not ok:
                return
            self._preview_img = tk.PhotoImage(data=buf.tobytes())
            self.canvas.configure(image=self._preview_img)
        except Exception:
            pass

    # ------------------------------------------------------------- 시작/정지
    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.text.delete("1.0", "end")
        # 로그 첫 줄에 버전을 남긴다. 사용자가 로그를 그대로 보내 줄 때
        # 어느 판에서 난 일인지 바로 알 수 있어야 한다.
        self.log(version_line())
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status("시작 중...")
        mode = self.mode.get()
        self.worker = threading.Thread(target=self._run, args=(mode,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        """정지: 진행 중이던 분석 적용, 새 경로, 다음 칸 클릭, 팝업 클릭,
        자동 재시작을 모두 즉시 차단한다."""
        self.status("정지 요청됨")
        self.log("[정지] 버튼이 눌렸습니다. 모든 동작을 차단합니다.")
        if self.engine is not None:
            self.engine.stop()

    def _run(self, mode: str) -> None:
        try:
            if mode == "2":
                self._run_explore()
            elif mode == "3":
                self._run_dungeon()
            else:
                self._run_network()
        except Exception:
            self.log("[오류]\n" + traceback.format_exc())
        finally:
            self.msgq.put(("done", None))

    def _run_windowed(self, engine) -> None:
        """창을 고정해서 쓰는 엔진(탐사/던전)을 돌린다.

        고정된 HWND 를 GUI 에 표시하려고 pick_window 를 한 겹 감싼다.
        """
        self.engine = engine
        orig_pick = engine.pick_window

        def pick():
            win = orig_pick()
            self.msgq.put(("hwnd", f"0x{win.hwnd:X}" if win else "-"))
            return win

        engine.pick_window = pick
        engine.run()

    def _run_explore(self) -> None:
        from digimonup.app.explore import ExploreEngine
        from digimonup.base.settings import load_explore_config

        self._run_windowed(ExploreEngine(load_explore_config(), log=self.log,
                                         status=self.status, preview=self.preview))

    def _run_dungeon(self) -> None:
        from digimonup.app.dungeon import DungeonEngine
        from digimonup.base.settings import load_dungeon_config

        self._run_windowed(DungeonEngine(load_dungeon_config(), log=self.log,
                                         status=self.status, preview=self.preview))

    def _run_network(self) -> None:
        from digimonup.app import network_macro
        engine = network_macro.NetworkMacro(log=self.log, status=self.status)
        self.engine = engine
        engine.run()

    def _on_worker_done(self) -> None:
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_var.get() not in ("창을 찾지 못함",):
            self.status("대기 중")
        self.engine = None

    def _on_close(self) -> None:
        if self.engine is not None:
            self.engine.stop()
        self.root.after(200, self.root.destroy)


def main() -> int:
    from digimonup.win.emulator_window import enable_dpi_awareness
    enable_dpi_awareness()

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
