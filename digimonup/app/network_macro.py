"""1번 기능: 네트워크 (자동 매칭/포기).

기존 macro.py 의 동작을 그대로 옮기되, GUI 에서도 쓸 수 있도록
  - print 대신 log 콜백
  - 전역 단축키 대신 stop 이벤트(단축키도 함께 지원)
로 바꿔 클래스로 감쌌다. macro.py 는 이 클래스를 쓰는 콘솔 진입점이다.

동작 요약 (자세한 규칙은 README 참고)
  매 스캔마다 '매칭'과 '포기'를 둘 다 찾는다.
    - 매칭이 보이면 -> 클릭
    - 포기가 보이면 -> 처음 본 시점부터 giveup_delay_sec 뒤 클릭
    - 둘 다 없으면 -> 화면 전환 중. 계속 감시
  포기 클릭 뒤 남는 잔상을 새 등장으로 오인하지 않도록,
  "포기가 안 보이고 + 매칭이 보인다"일 때만 타이머를 다시 걸 수 있다.
"""

from __future__ import annotations

import threading
import time

import pyautogui

from digimonup.vision import popup
from digimonup.base.common import (enable_dpi_awareness, find_template, grab_screen,
                    is_stop_key_pressed, load_button_templates, load_config, vk_of)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class NetworkMacro:
    def __init__(self, log=print, status=lambda s: None):
        self.log = log
        self.status = status
        self.stop_event = threading.Event()
        self.rounds = 0

    def stop(self) -> None:
        self.stop_event.set()

    # ------------------------------------------------------------------
    def _should_stop(self, stop_vk: int) -> bool:
        return self.stop_event.is_set() or is_stop_key_pressed(stop_vk)

    def _click_at(self, x, y, offset, click_scale, move_duration) -> tuple[int, int]:
        sx = int((x + offset[0]) * click_scale)
        sy = int((y + offset[1]) * click_scale)
        pyautogui.moveTo(sx, sy, duration=move_duration)
        pyautogui.click()
        return sx, sy

    # ------------------------------------------------------------------
    def run(self) -> int:
        enable_dpi_awareness()
        cfg = load_config()

        try:
            tpl_match, tpl_giveup = load_button_templates(cfg)
        except (FileNotFoundError, ValueError) as e:
            self.log("[준비 필요] " + str(e))
            self.log("capture.bat 을 실행해서 '매칭'/'포기' 버튼을 각각 캡처해주세요.")
            self.status("템플릿 없음")
            return 1

        region = cfg.get("region")
        offset = (region[0], region[1]) if region else (0, 0)

        click_scale = 1.0
        captured_w = grab_screen().shape[1]
        mouse_w = pyautogui.size().width
        if captured_w and captured_w != mouse_w:
            click_scale = mouse_w / captured_w
            self.log(f"좌표 보정 적용: 캡처 {captured_w}px -> 마우스 {mouse_w}px "
                     f"(x{click_scale:.3f})")

        conf = float(cfg["confidence"])
        delay = float(cfg["giveup_delay_sec"])
        interval = float(cfg["scan_interval_sec"])
        cooldown = float(cfg["click_cooldown_sec"])
        multi_scale = bool(cfg.get("multi_scale"))
        scales = cfg.get("scales")
        move_duration = float(cfg.get("move_duration", 0.12))
        tolerance = float(cfg.get("giveup_lost_tolerance_sec", 1.5))
        recheck = float(cfg.get("giveup_recheck_sec", 0.4))
        max_attempts = int(cfg.get("giveup_max_attempts", 4))
        verify_sec = float(cfg.get("giveup_verify_sec", 3.0))
        max_wait = max(float(cfg.get("giveup_max_wait_sec", 7.0)), delay)
        stop_vk = vk_of(cfg.get("stop_key", "F12"))
        # 팝업을 닫고 기다리는 시간. 실측(던전): 바깥을 누르면 0.5초 안에 닫힌다.
        popup_settle = float(cfg.get("popup_settle_sec", 0.8))

        lock_match: dict = {}
        lock_giveup: dict = {}

        self.log("=" * 46)
        self.log("네트워크 매크로 시작 (매칭/포기 동시 탐색)")
        self.log(f"  정확도 기준 {conf} / 포기 대기 {delay:g}초 (최대 {max_wait:g}초)")
        self.log(f"  중지: 정지 버튼 또는 {cfg.get('stop_key', 'F12')} 키")
        self.log("=" * 46)
        self.status("실행 중 (네트워크)")

        start_delay = float(cfg.get("start_delay_sec", 3.0))
        if start_delay > 0:
            self.log(f"{start_delay:.0f}초 뒤 시작합니다. 게임 창을 띄워주세요...")
            if self.stop_event.wait(start_delay):
                self.log("시작 전에 중지되었습니다.")
                return 0

        giveup_first_seen = None
        giveup_last_seen = 0.0
        giveup_armed = True
        giveup_clicked_at = 0.0
        giveup_attempts = 0
        next_click_at = 0.0
        last_idle_report = 0.0
        idle_since = None

        try:
            while True:
                if self._should_stop(stop_vk):
                    self.log("중지되었습니다.")
                    break

                now = time.time()
                sleep_for = interval
                screen = grab_screen(region)

                # 판이 끝나면 실패창/보상창이 올라오고, 그동안 아래 화면은
                # 클릭을 먹지 않는다. 모르고 계속 누르면 헛클릭만 쌓인다.
                # 규칙은 던전(3번)에서 검증된 것을 그대로 쓴다(popup 참고).
                # 여기서는 모니터 전체를 캡처하므로 띠를 쓰지 않고 전체를 본다.
                found = popup.find(screen, use_band=False)
                if found:
                    kind, pscore, box = found
                    px, py = popup.close_point_for_box(
                        box, screen.shape[1], screen.shape[0])
                    self.log(f"[팝업] {popup.name_of(kind)}이(가) 떠 있습니다 "
                             f"({pscore:.2f}). 바깥을 눌러 닫습니다.")
                    self._click_at(px, py, offset, click_scale, move_duration)
                    if self.stop_event.wait(popup_settle):
                        break
                    continue

                m_found, m_pt, m_score = find_template(
                    screen, tpl_match, conf, multi_scale, scales, lock_match)
                g_found, _, g_score = find_template(
                    screen, tpl_giveup, conf, multi_scale, scales, lock_giveup)

                # 두 버튼은 같은 UI 라 표시 배율이 같다. 확정된 배율을 공유해 스캔을 줄인다.
                if "scale" in lock_match and "scale" not in lock_giveup:
                    lock_giveup["scale"] = lock_match["scale"]
                elif "scale" in lock_giveup and "scale" not in lock_match:
                    lock_match["scale"] = lock_giveup["scale"]

                if m_found and g_found:
                    self.log(f"두 버튼이 동시에 잡혔습니다 (매칭 {m_score:.3f} / "
                             f"포기 {g_score:.3f}). 유사도가 높은 쪽을 따릅니다.")
                    if m_score >= g_score:
                        g_found = False
                    else:
                        m_found = False

                if not giveup_armed:
                    if (not g_found) and m_found:
                        giveup_armed = True
                        giveup_attempts = 0
                    elif g_found and (now - giveup_clicked_at) >= verify_sec:
                        self.log(f"'포기'를 눌렀는데 {verify_sec:g}초 뒤에도 그대로입니다. "
                                 f"다시 누릅니다. ({giveup_attempts + 1}번째)")
                        if giveup_attempts >= max_attempts:
                            self.log(f"[주의] 포기 클릭이 {giveup_attempts}회 연속 "
                                     f"안 먹히고 있습니다. 게임 창이 맨 앞에 있는지, "
                                     f"관리자 권한이 필요한지 확인하세요.")
                        self.rounds = max(0, self.rounds - 1)
                        giveup_armed = True
                        giveup_first_seen = now - delay
                        giveup_last_seen = now

                waited = (now - giveup_first_seen) if giveup_first_seen is not None else 0.0
                timer_due = (giveup_first_seen is not None
                             and (waited >= max_wait
                                  or (waited >= delay and now >= next_click_at)))

                if g_found and not giveup_armed and not timer_due:
                    giveup_first_seen = None
                    idle_since = None

                elif not timer_due and g_found:
                    idle_since = None
                    if giveup_first_seen is None or (now - giveup_last_seen) > tolerance:
                        giveup_first_seen = now
                        self.log(f"'포기' 버튼 확인 (유사도 {g_score:.3f}) -> "
                                 f"{delay:g}초 뒤 클릭")
                    giveup_last_seen = now
                    remaining = delay - (now - giveup_first_seen)
                    if 0 < remaining < interval:
                        sleep_for = max(remaining, 0.02)

                elif timer_due:
                    idle_since = None
                    attempt = 0
                    while True:
                        if self._should_stop(stop_vk):
                            break
                        confirm = grab_screen(region)
                        _, g_cpt, g_cs = find_template(
                            confirm, tpl_giveup, 0.0, multi_scale, scales, lock_giveup)
                        _, m_cpt, m_cs = find_template(
                            confirm, tpl_match, 0.0, multi_scale, scales, lock_match)

                        if attempt == 0:
                            if g_cpt is None and m_cpt is None:
                                self.log("[주의] 클릭할 대상을 찾지 못했습니다.")
                                break
                            if m_cpt is not None and (g_cpt is None or m_cs > g_cs):
                                self.log(f"'매칭'({m_cs:.3f})이 '포기'({g_cs:.3f})보다 "
                                         f"정확도가 높아 매칭을 클릭합니다.")
                                sx, sy = self._click_at(m_cpt[0], m_cpt[1], offset,
                                                        click_scale, move_duration)
                                self.log(f"  클릭 좌표 ({sx},{sy})")
                                break
                            sx, sy = self._click_at(g_cpt[0], g_cpt[1], offset,
                                                    click_scale, move_duration)
                            self.log(f"'포기' 버튼 클릭 (대기 {waited:.1f}초, "
                                     f"유사도 {g_cs:.3f}) 좌표 ({sx},{sy})")
                            self.rounds += 1
                            self.log(f"1회 완료 (누적 {self.rounds}회)")
                            giveup_armed = False
                            giveup_clicked_at = time.time()
                        else:
                            if g_cpt is None or g_cs < conf:
                                break
                            sx, sy = self._click_at(g_cpt[0], g_cpt[1], offset,
                                                    click_scale, move_duration)
                            self.log(f"'포기'가 아직 남아 있습니다 (유사도 {g_cs:.3f}) "
                                     f"-> 즉시 재클릭 ({attempt}/{max_attempts}) "
                                     f"좌표 ({sx},{sy})")
                            giveup_clicked_at = time.time()

                        attempt += 1
                        if attempt > max_attempts:
                            self.log(f"[주의] 포기 버튼을 {max_attempts}회 더 눌렀는데도 "
                                     f"그대로입니다.")
                            break
                        if self.stop_event.wait(recheck):
                            break

                    giveup_attempts += attempt
                    giveup_first_seen = None
                    next_click_at = time.time() + cooldown

                elif m_found:
                    giveup_first_seen = None
                    idle_since = None
                    if now >= next_click_at:
                        sx, sy = self._click_at(m_pt[0], m_pt[1], offset,
                                                click_scale, move_duration)
                        self.log(f"'매칭' 버튼 발견 (유사도 {m_score:.3f}) -> 클릭 "
                                 f"좌표 ({sx},{sy})")
                        next_click_at = time.time() + cooldown

                else:
                    if giveup_first_seen is not None:
                        if (now - giveup_last_seen) > tolerance:
                            self.log(f"'포기'가 안 잡히지만 대기는 계속합니다 "
                                     f"({waited:.1f}/{delay:g}초)")
                            giveup_last_seen = now
                        remaining = delay - waited
                        if 0 < remaining < interval:
                            sleep_for = max(remaining, 0.02)
                    if idle_since is None:
                        idle_since = now
                    if now - last_idle_report > 15:
                        idle_for = now - idle_since
                        if idle_for >= 60:
                            self.log(f"[주의] {idle_for/60:.0f}분째 두 버튼 모두 "
                                     f"안 보입니다. (매칭 {m_score:.2f} / "
                                     f"포기 {g_score:.2f})")
                        else:
                            self.log(f"대기 중... (매칭 {m_score:.2f} / "
                                     f"포기 {g_score:.2f})")
                        last_idle_report = now

                if self.stop_event.wait(sleep_for):
                    self.log("중지되었습니다.")
                    break

        except pyautogui.FailSafeException:
            self.log("비상 정지(마우스 모서리)로 중단되었습니다.")
        finally:
            self.log(f"네트워크 매크로 종료. 총 {self.rounds}회 수행.")
            self.status("정지됨")
        return 0
