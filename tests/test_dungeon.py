"""던전 엔진 동작 테스트.

여기서 지키려는 것
  - 도전·실패창·보상창을 **매 스캔마다 다** 찾는다. 한 판의 끝은 져서 실패창일
    수도, 이겨서 보상창일 수도 있다. 순서를 정해 기다리면 한쪽에서 멈춘다.
  - 무엇이든 **한 번만** 누르고, recheck_sec 안에는 다시 누르지 않는다.
    화면이 바뀌는 중에 또 누르면 그 클릭이 아래 화면으로 새어 나간다.
  - recheck_sec 이 지나도 그대로면 **한 번씩** 더 누른다.
  - 팝업(실패창/보상창)은 도전보다 먼저다. 모달이라 클릭을 먹기 때문이다.
  - 팝업을 닫는 클릭은 **왼쪽 위 구석**이다. 보상창 아래의 '포기' 버튼을 누르면
    던전 보상을 못 받는다.
  - 정지를 누르면 뒤늦은 클릭이 절대 나가지 않는다.

인식 판정은 합성이 아니라 **실제 캡처**(tests/fixtures/dungeon_*.png)로 본다.
버튼과 글자는 색 분포가 아니라 생김새로 가려내는 것이라, 합성 화면으로는
기준값이 현실에서 갈리는지 확인할 수 없다.
"""

from __future__ import annotations

import cv2
import pytest

from digimonup.base.common import Stopped
from digimonup.app.dungeon import (KINDS, DungeonConfig, DungeonEngine,
                                   load_dungeon_templates)

FIXTURES = {
    "fail": "tests/fixtures/dungeon_fail.png",              # '던전 실패...'
    "fail_stage": "tests/fixtures/dungeon_fail_stage.png",  # '스테이지 실패...'
    # 보상창은 이펙트가 돌아서 프레임마다 다르다. 이 픽스처는 **템플릿으로 쓰지
    # 않은** 프레임이라, 처음 보는 화면에서도 잡히는지 확인하는 셈이 된다.
    "reward": "tests/fixtures/dungeon_reward.png",
    "entry": "tests/fixtures/dungeon_entry.png",
}


@pytest.fixture(scope="module")
def shots():
    got = {name: cv2.imread(path) for name, path in FIXTURES.items()}
    missing = [name for name, img in got.items() if img is None]
    assert not missing, f"던전 픽스처를 읽지 못했습니다: {missing}"
    got["blank"] = got["entry"] * 0        # 전투 중 = 아무것도 안 보이는 화면
    return got


class FakeWindow:
    """캡처 프레임을 대본대로 돌려주고, 클릭은 기록만 하는 가짜 창."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.clicks: list[tuple[int, int]] = []
        self.hwnd = 0x1234
        self.top_hwnd = 0x1230
        self.i = 0

    def is_valid(self):
        return True

    def client_size(self):
        return self.frames[0].shape[1], self.frames[0].shape[0]

    def capture(self):
        img = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return img

    def focus(self, retries=3):
        return True

    def click_client(self, x, y, move_duration=0.0, require_focus=True):
        self.clicks.append((int(x), int(y)))
        return int(x) + 1000, int(y) + 50


def _engine(frames, **cfg_kw):
    defaults = dict(save_debug=False, start_delay_sec=0.0, scan_interval_sec=0.0,
                    move_duration=0.0, recheck_sec=0.2)
    defaults.update(cfg_kw)
    eng = DungeonEngine(DungeonConfig(**defaults), log=lambda *_: None)
    eng.window = FakeWindow(frames)
    eng.pick_window = lambda: eng.window
    # 이 테스트는 저장소에 든 진짜 던전 템플릿으로 판정한다.
    eng.templates = load_dungeon_templates()
    return eng


def _run_until(eng, clicks: int):
    """클릭이 정해진 수만큼 나가면 멈추게 하고 run() 을 돌린다."""
    orig = eng.window.click_client

    def counting(*a, **kw):
        got = orig(*a, **kw)
        if len(eng.window.clicks) >= clicks:
            eng.stop()
        return got

    eng.window.click_client = counting
    eng.run()


# ------------------------------------------------------------- 인식 자체
def test_실제_캡처에서_셋이_서로_확실히_갈린다(shots):
    """각 화면에서 제 것만 기준을 넘고, 나머지는 넘지 않아야 한다."""
    eng = _engine([shots["entry"]])
    screens = {"challenge": "entry", "fail": "fail", "reward": "reward"}

    for kind, own in screens.items():
        on = eng._find_kind(shots[own], kind)[0]
        assert on >= eng._min_of(kind), f"{kind} 화면인데 {on:.3f}"
        for other in screens.values():
            if other == own:
                continue
            off = eng._find_kind(shots[other], kind)[0]
            assert off < eng._min_of(kind), (
                f"{other} 화면이 {kind} 로 잡혔다 {off:.3f}")
            # 기준을 조금 흔들어도 뒤집히지 않을 만큼 벌어져 있어야 한다.
            assert on - off > 0.2, f"{kind}: {own} {on:.3f} vs {other} {off:.3f}"


def test_도전_버튼을_실제_버튼_자리에서_찾는다(shots):
    eng = _engine([shots["entry"]])
    _, at = eng._find_kind(shots["entry"], "challenge")
    assert at is not None
    h = shots["entry"].shape[0]
    assert 0.70 * h < at[1] < 0.85 * h, "엉뚱한 곳을 누르려 합니다"


def test_실패창_첫_줄이_바뀌어도_알아본다(shots):
    """첫 줄은 던전 이름이라 바뀐다. 실측: '던전 실패...' 와 '스테이지 실패...'.

    '실패...' 줄만 보기 때문에 둘 다 같은 점수로 잡혀야 한다. 첫 줄까지 템플릿에
    넣었을 때는 '스테이지' 판에서 0.790 까지 떨어졌었다.
    """
    eng = _engine([shots["fail"]])
    dungeon_score = eng._find_kind(shots["fail"], "fail")[0]
    stage_score = eng._find_kind(shots["fail_stage"], "fail")[0]

    assert stage_score >= eng.cfg.fail_min, f"'스테이지 실패...' 를 놓쳤다 {stage_score:.3f}"
    assert abs(dungeon_score - stage_score) < 0.05, (
        f"첫 줄 때문에 점수가 갈렸다: 던전 {dungeon_score:.3f} / "
        f"스테이지 {stage_score:.3f}")


def test_보상창은_이펙트가_돌아도_알아본다(shots):
    """이 픽스처는 템플릿으로 쓰지 않은 프레임이다. 그래도 넘어야 한다."""
    eng = _engine([shots["reward"]])
    score = eng._find_kind(shots["reward"], "reward")[0]
    assert score >= eng.cfg.reward_min, f"처음 보는 보상 프레임을 놓쳤다 {score:.3f}"


# --------------------------------------------------------- 다 찾아 놓기
def test_한_스캔에서_셋_다_찾는다(shots):
    """어느 화면이 와도 같은 스캔이 알아본다. 순서를 기다리지 않는다."""
    eng = _engine([shots["entry"]])

    assert eng._look(shots["entry"]).kind == "challenge"
    assert eng._look(shots["fail"]).kind == "fail"
    assert eng._look(shots["fail_stage"]).kind == "fail"
    assert eng._look(shots["reward"]).kind == "reward"
    assert eng._look(shots["blank"]) is None


@pytest.mark.parametrize("popup", ["fail", "reward"])
def test_팝업이_도전을_이긴다(shots, monkeypatch, popup):
    """둘 다 잡히면 유사도와 무관하게 팝업이 먼저다. 모달이 클릭을 먹는다."""
    eng = _engine([shots["entry"]])
    # 도전 점수를 팝업보다 높게 만들어도 팝업을 골라야 한다.
    fake = {"challenge": (0.99, (350, 990)), popup: (0.75, (350, 200))}
    monkeypatch.setattr(eng, "_find_kind",
                        lambda img, kind: fake.get(kind, (0.0, None)))

    target = eng._look(shots["entry"])
    assert target.kind == popup
    assert target.at == eng.popup_close_point(), "팝업은 글자가 아니라 바깥을 눌러야 합니다"


def test_실패창이_없어도_바로_도전한다(shots):
    """이긴 판에는 실패창이 안 뜬다. 기다리지 않고 도전해야 한다."""
    eng = _engine([shots["entry"]] * 4 + [shots["blank"]] * 50)
    _run_until(eng, clicks=1)

    assert eng.stats.clicks["challenge"] == 1
    assert eng.stats.clicks["fail"] == 0


# ------------------------------------------------------- 1회 클릭 규칙
@pytest.mark.parametrize("screen,kind", [("entry", "challenge"),
                                         ("fail", "fail"),
                                         ("reward", "reward")])
def test_화면이_바뀌면_한_번만_누른다(shots, screen, kind):
    """누르고 화면이 바뀌면 그걸로 끝. 재클릭이 없어야 한다."""
    eng = _engine([shots[screen]] + [shots["blank"]] * 60, recheck_sec=99)
    _run_until(eng, clicks=1)

    assert len(eng.window.clicks) == 1
    assert eng.stats.clicks[kind] == 1
    assert sum(eng.stats.reclicks.values()) == 0


def test_recheck_전에는_같은_것을_다시_누르지_않는다(shots):
    """화면이 아직 그대로여도 recheck_sec 안에는 절대 다시 누르지 않는다."""
    eng = _engine([shots["fail"]] * 200, recheck_sec=99)

    scans = [0]
    orig_look = eng._look

    def counting_look(img):
        scans[0] += 1
        if scans[0] >= 12:      # 여러 번 스캔해도 클릭은 한 번뿐이어야 한다
            eng.stop()
        return orig_look(img)

    eng._look = counting_look
    eng.run()

    assert len(eng.window.clicks) == 1, f"{scans[0]}회 스캔에 클릭 {len(eng.window.clicks)}회"


def test_recheck_가_지나도_그대로면_한_번씩_더_누른다(shots):
    """계속 보상창만 보이는 상황. 한 번에 한 번씩만 더 눌러야 한다."""
    eng = _engine([shots["reward"]] * 400, recheck_sec=0.05)
    _run_until(eng, clicks=3)

    assert eng.window.clicks == [eng.popup_close_point()] * 3
    assert eng.stats.clicks["reward"] == 1, "같은 보상창은 한 번만 세야 합니다"
    assert eng.stats.reclicks["reward"] == 2


def test_한도를_넘겨도_멈추지_않는다(shots):
    """도전 횟수에 제한이 없으므로 스스로 끝내지 않는다. 경고만 남긴다."""
    logs: list[str] = []
    eng = _engine([shots["entry"]] * 400, recheck_sec=0.02, max_attempts=2)
    eng.log = logs.append
    _run_until(eng, clicks=5)

    assert len(eng.window.clicks) == 5, "한도를 넘겼다고 클릭을 멈추면 안 됩니다"
    assert any("계속 시도합니다" in line for line in logs)


# ------------------------------------------------------------ 클릭 자리
@pytest.mark.parametrize("screen", ["fail", "reward"])
def test_팝업을_닫는_클릭은_왼쪽_위_구석이다(shots, screen):
    """보상창 아래쪽의 '포기' 버튼을 누르면 던전 보상을 못 받는다."""
    eng = _engine([shots[screen]] + [shots["blank"]] * 50, recheck_sec=99)
    _run_until(eng, clicks=1)

    expect = eng.popup_close_point()
    assert eng.window.clicks == [expect]

    w, h = eng.window.client_size()
    assert expect[0] < 0.25 * w and expect[1] < 0.25 * h, (
        f"팝업 닫기 클릭이 구석에서 너무 멉니다: {expect}")


# ---------------------------------------------------------- 정지 처리
@pytest.mark.parametrize("screen", ["entry", "fail", "reward"])
def test_정지_요청_후에는_클릭이_나가지_않는다(shots, screen):
    eng = _engine([shots[screen]] * 50)
    eng.stop()
    eng.run()          # Stopped 를 안에서 잡고 조용히 끝나야 한다
    assert eng.window.clicks == []


def test_클릭_직전에_정지하면_클릭이_취소된다(shots):
    eng = _engine([shots["entry"]] * 50)
    target = eng._look(shots["entry"])
    eng.stop()
    with pytest.raises(Stopped):
        eng._click(target, repeats=0)
    assert eng.window.clicks == []


# ------------------------------------------------------------ 한 바퀴
def test_져서_실패창이_뜬_판을_돌린다(shots):
    """실패 -> 바깥 클릭 -> 던전 화면 -> 도전 클릭."""
    frames = ([shots["fail"]]            # 1) 실패창 -> 바깥 클릭
              + [shots["entry"]] * 3     # 2) 닫혔다 -> 도전 클릭
              + [shots["blank"]] * 60)
    eng = _engine(frames, recheck_sec=99)
    _run_until(eng, clicks=2)

    h = eng.window.client_size()[1]
    assert eng.window.clicks[0] == eng.popup_close_point(), "먼저 실패창을 닫아야 합니다"
    assert eng.window.clicks[1][1] > 0.7 * h, "그다음 도전 버튼을 눌러야 합니다"
    assert eng.stats.clicks["fail"] == 1 and eng.stats.clicks["challenge"] == 1
    assert sum(eng.stats.reclicks.values()) == 0


def test_이겨서_보상창이_뜬_판을_돌린다(shots):
    """보상 -> 바깥 클릭 -> 던전 화면 -> 도전 클릭. 실패창은 끝내 안 뜬다."""
    frames = ([shots["reward"]]          # 1) 보상창 -> 바깥 클릭
              + [shots["entry"]] * 3     # 2) 닫혔다 -> 도전 클릭
              + [shots["blank"]] * 60)
    eng = _engine(frames, recheck_sec=99)
    _run_until(eng, clicks=2)

    h = eng.window.client_size()[1]
    assert eng.window.clicks[0] == eng.popup_close_point(), "먼저 보상창을 닫아야 합니다"
    assert eng.window.clicks[1][1] > 0.7 * h, "그다음 도전 버튼을 눌러야 합니다"
    assert eng.stats.clicks["reward"] == 1 and eng.stats.clicks["challenge"] == 1
    assert eng.stats.clicks["fail"] == 0


def test_이긴_판과_진_판을_연달아_돌린다(shots):
    """도전 -> 전투 -> 보상 -> 도전 -> 전투 -> 실패 -> 도전."""
    frames = ([shots["entry"]]           # 도전
              + [shots["blank"]] * 2
              + [shots["reward"]]        # 이겼다 -> 보상창 닫기
              + [shots["entry"]] * 2     # 도전
              + [shots["blank"]] * 2
              + [shots["fail_stage"]]    # 졌다 -> 실패창 닫기
              + [shots["entry"]] * 2     # 도전
              + [shots["blank"]] * 60)
    eng = _engine(frames, recheck_sec=99)
    _run_until(eng, clicks=5)

    assert eng.stats.clicks["challenge"] == 3
    assert eng.stats.clicks["reward"] == 1
    assert eng.stats.clicks["fail"] == 1
    assert sum(eng.stats.reclicks.values()) == 0


# ------------------------------------------------------------ 설정 읽기
def test_config_json_의_던전_절이_그대로_읽힌다():
    from digimonup.base.settings import load_dungeon_config

    cfg = load_dungeon_config()
    assert isinstance(cfg.popup_close_point, tuple) and len(cfg.popup_close_point) == 2
    for kind, (_, min_key, band_key) in KINDS.items():
        band = getattr(cfg, band_key)
        assert isinstance(band, tuple) and len(band) == 2, kind
        assert 0.0 <= band[0] < band[1] <= 1.0, kind
        assert 0.0 < getattr(cfg, min_key) <= 1.0, kind
    assert cfg.recheck_sec > 0 and cfg.max_attempts > 0


def test_템플릿이_세_종류_다_들어_있다():
    """던전은 색으로는 판단할 수 없어 템플릿이 반드시 있어야 한다."""
    tpl = load_dungeon_templates()
    assert set(tpl) == set(KINDS)
    for kind, tset in tpl.items():
        assert tset.images, f"templates/dungeon/{kind}/ 가 비어 있습니다"
