"""앱 아이콘 만들기.

이 매크로가 하는 일(5x5 게임판에서 장애물을 피해 오른쪽으로 나아가기)을 그대로
그린 오리지널 아이콘을 만든다. 게임의 공식 로고는 쓰지 않는다. 공개 저장소와
배포용 EXE 에 남의 저작물을 넣으면 안 되기 때문이다.

    python tools\\make_icon.py                 <- 기본 아이콘을 그려서 만든다
    python tools\\make_icon.py 내로고.png       <- 직접 준비한 이미지로 만든다

만들어지는 것 (assets/ 폴더)
    icon.ico    EXE 와 창 아이콘용 (16/32/48/64/128/256 다 들어 있는 멀티 아이콘)
    icon.png    256x256. README·favicon 등 웹에서 쓸 때

작은 크기에서는 격자선이 뭉개지므로, 32px 이하는 요소를 줄여 단순하게 그린다.
"""

from __future__ import annotations

import _bootstrap  # 저장소 루트를 sys.path 에 넣는다. 맨 먼저 가져온다

import os
import sys

from PIL import Image, ImageDraw  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets")
SIZES = (16, 32, 48, 64, 128, 256)

# 게임 화면에서 실측한 색을 그대로 쓴다 (README 의 HSV 표 참고).
BG_DARK = (14, 22, 46)
CELL_LIT = (58, 140, 235)
CHIP = (255, 150, 32)


def _rounded(size: int, radius_ratio: float = 0.22):
    """배경이 될 둥근 사각형."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = max(2, int(size * radius_ratio))
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG_DARK)
    return img, d


# 굵을수록 작은 크기에서 잘 버틴다. 앞에 있는 것부터 찾아 쓴다.
FONT_CANDIDATES = (
    "ariblk.ttf",      # Arial Black
    "impact.ttf",
    "seguibl.ttf",     # Segoe UI Black
    "arialbd.ttf",
    "segoeuib.ttf",
)


def _load_font(px: int):
    """설치된 굵은 폰트를 크기 px 로 불러온다. 없으면 PIL 기본 폰트."""
    from PIL import ImageFont
    for name in FONT_CANDIDATES:
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
    return ImageFont.load_default()


def _fit_text(d, text: str, box_w: float, box_h: float):
    """box 안에 꽉 차게 들어가는 폰트와 그 실제 크기를 찾는다."""
    px = int(box_h * 1.4)
    while px > 6:
        font = _load_font(px)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        if (x1 - x0) <= box_w and (y1 - y0) <= box_h:
            return font, (x1 - x0), (y1 - y0), x0, y0
        px -= 1
    font = _load_font(8)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    return font, (x1 - x0), (y1 - y0), x0, y0


def draw_icon(size: int) -> Image.Image:
    """한 변이 size 인 아이콘 한 장.

    'DU'(digimonUp) 이니셜을 쓴다. 앞 글자는 흰색, 'U'는 주황으로 두어
    이름의 'Up' 이 눈에 들어오게 했다. 게임 로고를 쓰지 않고 이 앱의 머리글자만
    쓰므로 남의 저작물 문제가 없다.

    16px 처럼 아주 작을 때는 두 글자가 뭉개지므로 'D' 한 글자만 크게 그린다.
    """
    img, d = _rounded(size)
    pad = size * 0.14
    # 24px 미만에서는 밑줄을 넣으면 글자가 눌려 알아볼 수 없다. 글자만 크게 쓴다.
    with_bar = size >= 24
    text = "DU" if size >= 32 else "D"

    box_h = (size - pad * 2) * (0.78 if with_bar else 1.0)
    font, tw, th, ox, oy = _fit_text(d, text, size - pad * 2, box_h)
    tx = (size - tw) / 2 - ox
    ty = (size - th) / 2 - oy - (size * 0.04 if with_bar else 0)

    if text == "DU":
        # 글자마다 색을 달리하려면 따로 그려야 한다.
        dw = d.textlength("D", font=font)
        d.text((tx, ty), "D", font=font, fill=(255, 255, 255))
        d.text((tx + dw, ty), "U", font=font, fill=CHIP)
    else:
        d.text((tx, ty), text, font=font, fill=(255, 255, 255))

    if with_bar:
        # 'Up' 을 나타내는 밑줄. 디지털 느낌을 주려고 칸을 띄워 그린다.
        bar_y = size - pad * 0.95
        bar_h = max(2, int(size * 0.055))
        seg = (size - pad * 2) / 7
        for i in range(4):
            x = pad + i * seg * 1.75
            d.rounded_rectangle([x, bar_y, x + seg * 1.35, bar_y + bar_h],
                                radius=bar_h / 2, fill=CELL_LIT if i < 3 else CHIP)
    return img


def from_source(path: str, size: int) -> Image.Image:
    """직접 준비한 이미지를 정사각형으로 맞춰 아이콘 한 장으로 만든다."""
    src = Image.open(path).convert("RGBA")
    side = max(src.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(src, ((side - src.width) // 2, (side - src.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src and not os.path.exists(src):
        print(f"이미지를 찾을 수 없습니다: {src}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = [from_source(src, s) if src else draw_icon(s) for s in SIZES]

    ico = os.path.join(OUT_DIR, "icon.ico")
    frames[-1].save(ico, format="ICO",
                    sizes=[(s, s) for s in SIZES])
    png = os.path.join(OUT_DIR, "icon.png")
    frames[-1].save(png, format="PNG")

    print(f"만들었습니다 ({'직접 준비한 이미지' if src else '기본 디자인'})")
    print(f"  {ico}   ({', '.join(f'{s}x{s}' for s in SIZES)})")
    print(f"  {png}   256x256")
    print("\n적용하려면 다시 빌드하세요:  scripts\\build_exe.bat")
    return 0


if __name__ == "__main__":
    _bootstrap.run_main(main)
