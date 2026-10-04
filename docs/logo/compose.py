"""Compose the PBA Autonomous Workflow logo set from the Blender render.

Every design is written twice from ONE layout:
  *_live.svg     -- the name is live <text> (editable type in Illustrator; needs
                    URW Gothic installed -- the OTFs are shipped in fonts/)
  *_outlined.svg -- the same glyphs as vector paths (identical on any machine)
PNG and PDF exports are rasterised/converted from the outlined SVG with cairo,
so the exports match the editable files exactly.
"""

from __future__ import annotations

import base64
import io
import os
import sys

import cairosvg
import numpy as np
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FONT_DIR = "/usr/share/fonts/opentype/urw-base35"
FONTS = {"demi": ("URWGothic-Demi", f"{FONT_DIR}/URWGothic-Demi.otf"),
         "book": ("URWGothic-Book", f"{FONT_DIR}/URWGothic-Book.otf")}

BLUE, BLUE_DARKMODE = "#0B2D8F", "#6F9BFF"
INK, INK_DARKMODE = "#1C2230", "#F2F5FA"
GREY, GREY_DARKMODE = "#4A5363", "#A7B2C3"
GOLD = "#E3A21C"
TAGLINE = ("Closed-loop orchestration for", "Prussian blue analogue synthesis")


class Face:
    def __init__(self, key):
        self.ps, path = FONTS[key]
        self.tt = TTFont(path)
        self.gs = self.tt.getGlyphSet()
        self.upem = self.tt["head"].unitsPerEm
        blob = hb.Blob.from_file_path(path)
        self.hbfont = hb.Font(hb.Face(blob))
        self.names = self.tt.getGlyphOrder()

    def shape(self, text):
        buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        hb.shape(self.hbfont, buf, {"kern": True, "liga": True})
        return buf.glyph_infos, buf.glyph_positions

    def width(self, text, size):
        _, pos = self.shape(text)
        return sum(p.x_advance for p in pos) * size / self.upem

    def outline(self, text, size, x, y):
        """SVG path data for `text` with its baseline at (x, y)."""
        s = size / self.upem
        infos, pos = self.shape(text)
        pen = SVGPathPen(self.gs)
        cx = 0
        for inf, p in zip(infos, pos):
            name = self.names[inf.codepoint]
            t = TransformPen(pen, (s, 0, 0, -s, x + (cx + p.x_offset) * s, y - p.y_offset * s))
            self.gs[name].draw(t)
            cx += p.x_advance
        return pen.getCommands()


F = {k: Face(k) for k in FONTS}


def crystal_png():
    """Crop the render to its content and return (base64, w/h aspect)."""
    im = Image.open(SRC).convert("RGBA")
    a = np.array(im)[..., 3]
    ys, xs = np.nonzero(a > 6)
    pad = int(0.02 * im.width)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
    im = im.crop(box)
    im.save(os.path.join(OUT, "crystal.png"))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), im.width / im.height


B64, ASPECT = crystal_png()


def image(x, y, h):
    w = h * ASPECT
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{B64}"/>'), w


def text(face, s, size, x, y, fill, live, tracking=0.0):
    f = F[face]
    if live:
        ls = f' letter-spacing="{tracking:.2f}"' if tracking else ""
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{f.ps}" font-size="{size:.1f}"'
                f' fill="{fill}"{ls}>{s}</text>')
    return f'<path d="{f.outline(s, size, x, y)}" fill="{fill}"/>'


def fit(face, s, size, maxw):
    w = F[face].width(s, size)
    return size if w <= maxw else size * maxw / w


def svg(w, h, body, bg=None):
    defs = ""
    if bg:
        defs = ('<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
                f'<stop offset="0" stop-color="{bg[0]}"/><stop offset="1" stop-color="{bg[1]}"/>'
                '</linearGradient></defs>'
                f'<rect width="{w}" height="{h}" fill="url(#bg)"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{defs}{body}</svg>')


def social(live):
    """1280x640 GitHub social card; all content inside an 80 px safe border."""
    W, H, M = 1280, 640, 80
    img, iw = image(M - 10, M, H - 2 * M)
    x0 = M + iw + 30
    maxw = W - M - x0
    s_pba = fit("demi", "PBA", 170, maxw)
    s_aw = fit("book", "Autonomous Workflow", 60, maxw)
    s_tag = fit("book", TAGLINE[1], 26, maxw)
    y = 268
    parts = [img,
             text("demi", "PBA", s_pba, x0 - 4, y, BLUE, live),
             text("book", "Autonomous Workflow", s_aw, x0, y + 72, INK, live),
             f'<rect x="{x0:.1f}" y="{y + 108:.1f}" width="96" height="5" rx="2.5" fill="{GOLD}"/>',
             text("book", TAGLINE[0], s_tag, x0, y + 166, GREY, live),
             text("book", TAGLINE[1], s_tag, x0, y + 166 + s_tag * 1.35, GREY, live)]
    return svg(W, H, "".join(parts), bg=("#FFFFFF", "#E8EEF9"))


def banner(live, dark=False):
    """1600x400 README header, transparent background (light or dark mode)."""
    W, H = 1600, 400
    img, iw = image(10, 10, H - 20)
    x0 = 10 + iw + 40
    maxw = W - 40 - x0
    blue, ink, grey = (BLUE_DARKMODE, INK_DARKMODE, GREY_DARKMODE) if dark else (BLUE, INK, GREY)
    s = fit("demi", "PBA Autonomous Workflow", 118, maxw)       # size for the full line
    w_pba = F["demi"].width("PBA ", s)
    s_tag = fit("book", " ".join(TAGLINE), 34, maxw)
    y = 222
    parts = [img,
             text("demi", "PBA", s, x0, y, blue, live),
             text("book", "Autonomous Workflow", s, x0 + w_pba, y, ink, live),
             f'<rect x="{x0 + 2:.1f}" y="{y + 30:.1f}" width="110" height="6" rx="3" fill="{GOLD}"/>',
             text("book", " ".join(TAGLINE), s_tag, x0, y + 92, grey, live)]
    return svg(W, H, "".join(parts))


def black_bg(w, h, cx, cy, r):
    """Near-black vertical gradient plus a soft Prussian-blue glow behind the crystal."""
    return ('<defs>'
            '<linearGradient id="kbg" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#0B0F19"/><stop offset="1" stop-color="#000000"/>'
            '</linearGradient>'
            f'<radialGradient id="glow" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" gradientUnits="userSpaceOnUse">'
            '<stop offset="0" stop-color="#1A3FB8" stop-opacity="0.55"/>'
            '<stop offset="0.55" stop-color="#0B2D8F" stop-opacity="0.18"/>'
            '<stop offset="1" stop-color="#0B2D8F" stop-opacity="0"/>'
            '</radialGradient></defs>'
            f'<rect width="{w}" height="{h}" fill="url(#kbg)"/>'
            f'<rect width="{w}" height="{h}" fill="url(#glow)"/>')


def social_black(live):
    W, H, M = 1280, 640, 80
    img, iw = image(M - 10, M, H - 2 * M)
    x0 = M + iw + 30
    maxw = W - M - x0
    s_pba = fit("demi", "PBA", 170, maxw)
    s_aw = fit("book", "Autonomous Workflow", 60, maxw)
    s_tag = fit("book", TAGLINE[1], 26, maxw)
    y = 268
    parts = [black_bg(W, H, M - 10 + iw / 2, H / 2, iw * 0.62), img,
             text("demi", "PBA", s_pba, x0 - 4, y, BLUE_DARKMODE, live),
             text("book", "Autonomous Workflow", s_aw, x0, y + 72, INK_DARKMODE, live),
             f'<rect x="{x0:.1f}" y="{y + 108:.1f}" width="96" height="5" rx="2.5" fill="{GOLD}"/>',
             text("book", TAGLINE[0], s_tag, x0, y + 166, GREY_DARKMODE, live),
             text("book", TAGLINE[1], s_tag, x0, y + 166 + s_tag * 1.35, GREY_DARKMODE, live)]
    return svg(W, H, "".join(parts))


def banner_black(live):
    """1600x400 README header on a solid black panel: reads the same in GitHub
    light and dark themes, unlike a transparent banner."""
    W, H = 1600, 400
    img, iw = image(30, 18, H - 36)
    x0 = 30 + iw + 40
    maxw = W - 50 - x0
    s = fit("demi", "PBA Autonomous Workflow", 118, maxw)
    w_pba = F["demi"].width("PBA ", s)
    s_tag = fit("book", " ".join(TAGLINE), 34, maxw)
    y = 222
    parts = [black_bg(W, H, 30 + iw / 2, H / 2, iw * 0.66), img,
             text("demi", "PBA", s, x0, y, BLUE_DARKMODE, live),
             text("book", "Autonomous Workflow", s, x0 + w_pba, y, INK_DARKMODE, live),
             f'<rect x="{x0 + 2:.1f}" y="{y + 30:.1f}" width="110" height="6" rx="3" fill="{GOLD}"/>',
             text("book", " ".join(TAGLINE), s_tag, x0, y + 92, GREY_DARKMODE, live)]
    return svg(W, H, "".join(parts))


def icon_black():
    W = 1024
    img, iw = image((W - 1024 * ASPECT * 0.86) / 2, W * 0.07, W * 0.86)
    return svg(W, W, black_bg(W, W, W / 2, W / 2, W * 0.55) + img)


def icon():
    W = 1024
    img, iw = image(0, 0, W)
    x = (W - iw) / 2
    img = img.replace('x="0.0"', f'x="{x:.1f}"', 1)
    return svg(W, W, img)


designs = {"social_card": social, "banner_light": lambda live: banner(live, False),
           "banner_dark": lambda live: banner(live, True),
           "social_card_black": social_black, "banner_black": banner_black,
           "icon_black": lambda live: icon_black()}
for name, fn in designs.items():
    for live in (True, False):
        tag = "live" if live else "outlined"
        open(os.path.join(OUT, f"{name}_{tag}.svg"), "w").write(fn(live))
    out_svg = os.path.join(OUT, f"{name}_outlined.svg")
    cairosvg.svg2png(url=out_svg, write_to=os.path.join(OUT, f"{name}.png"), output_width=None)
    cairosvg.svg2pdf(url=out_svg, write_to=os.path.join(OUT, f"{name}.pdf"))
open(os.path.join(OUT, "icon.svg"), "w").write(icon())
cairosvg.svg2png(url=os.path.join(OUT, "icon.svg"), write_to=os.path.join(OUT, "icon.png"))
cairosvg.svg2png(url=os.path.join(OUT, "icon.svg"), write_to=os.path.join(OUT, "icon_256.png"),
                 output_width=256, output_height=256)
print("written:", sorted(os.listdir(OUT)))
