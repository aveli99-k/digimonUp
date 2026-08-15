"""digimonUp 매크로 GUI.

더블클릭 한 번으로 실행되고, 시작할 때 기능 번호를 고른다.
    1) 네트워크  - 기존 매칭/포기 자동 클릭
    2) 탐사      - 5x5 게임판 경로 자동 이동

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODES = [
    ("1", "네트워크", "매칭 버튼을 누르고, 포기 버튼이 뜨면 잠시 뒤 눌러 반복합니다."),
    ("2", "탐사", "5x5 게임판을 인식해 장애물을 피해 한 칸씩 이동합니다."),
]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("digimonUp 매크로")
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
            from paths import resource
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
        for i, (num, name, desc) in enumerate(MODES):
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
        for num, name, desc in MODES:
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
            else:
                self._run_network()
        except Exception:
            self.log("[오류]\n" + traceback.format_exc())
        finally:
            self.msgq.put(("done", None))

    def _run_explore(self) -> None:
        from explore import ExploreEngine
        from settings import load_explore_config

        engine = ExploreEngine(load_explore_config(), log=self.log,
                               status=self.status, preview=self.preview)
        self.engine = engine
        # 창이 고정되면 GUI 에 표시한다.
        orig_pick = engine.pick_window

        def pick():
            win = orig_pick()
            self.msgq.put(("hwnd", f"0x{win.hwnd:X}" if win else "-"))
            return win

        engine.pick_window = pick
        engine.run()

    def _run_network(self) -> None:
        import network_macro
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
    from emulator_window import enable_dpi_awareness
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
