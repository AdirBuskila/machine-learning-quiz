# -*- coding: utf-8 -*-
"""Figure-cropping helper for connecting exam figures to questions.

Usage:
  PYTHONUTF8=1 python tools/crop_helper.py pages <pdf>
      -> prints page count.
  PYTHONUTF8=1 python tools/crop_helper.py grid <pdf> <page1based> <out.png>
      -> renders the page at 150 DPI with a labeled 0..10 (=0.0..1.0) grid overlay,
         so a viewer can read off a figure's bounding box as fractions.
  PYTHONUTF8=1 python tools/crop_helper.py crop <pdf> <page1based> <x0> <y0> <x1> <y1> <out.png>
      -> crops the fraction box [x0,y0,x1,y1] (each 0..1) of the page at 200 DPI,
         downscales to <=1200px wide, saves optimized PNG.

Coordinates: (0,0) = top-left, (1,1) = bottom-right of the page.
"""
import sys, pathlib
import fitz
from PIL import Image, ImageDraw, ImageFont

MAX_W = 1200


def _page(pdf, pg1):
    doc = fitz.open(pdf)
    p = doc[pg1 - 1]
    return doc, p


def cmd_pages(pdf):
    doc = fitz.open(pdf)
    print(len(doc))
    doc.close()


def cmd_grid(pdf, pg1, out):
    doc, page = _page(pdf, int(pg1))
    pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i in range(11):
        x = int(W * i / 10)
        y = int(H * i / 10)
        d.line([(x, 0), (x, H)], fill=(255, 0, 0), width=1)
        d.line([(0, y), (W, y)], fill=(255, 0, 0), width=1)
        d.text((x + 2, 2), str(i), fill=(255, 0, 0), font=font)
        d.text((2, y + 2), str(i), fill=(0, 0, 255), font=font)
    img.save(out)
    print(f"grid saved {out}  ({W}x{H}) — red=X (top), blue=Y (left), labels 0..10 = 0.0..1.0")
    doc.close()


def cmd_crop(pdf, pg1, x0, y0, x1, y1, out):
    doc, page = _page(pdf, int(pg1))
    r = page.rect
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                     r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if img.width > MAX_W:
        h = round(img.height * MAX_W / img.width)
        img = img.resize((MAX_W, h), Image.LANCZOS)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG", optimize=True)
    kb = pathlib.Path(out).stat().st_size / 1024
    print(f"crop saved {out}  ({img.width}x{img.height}, {kb:.0f} KB)")
    doc.close()


def cmd_resolve(stem):
    """Print the absolute PDF path for a manifest stem (handles zip-extracted files)."""
    import json, re
    HERE = pathlib.Path(__file__).parent
    man = {r["stem"]: r for r in json.loads((HERE / "raw" / "manifest.json").read_text(encoding="utf-8"))}
    SRC = pathlib.Path(r"C:\Users\Adir\Desktop\BSC\שנה ב\סמסטר ג\למידת מכונה")
    ZIPX = HERE / "raw" / "zip_extracted"
    rec = man[stem]
    if rec["origin"] == "zip":
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", rec["rel"][len("ZIP/"):])
        print(str(ZIPX / safe))
    else:
        print(str(SRC / rec["rel"]))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(1)
    if a[0] == "resolve":
        cmd_resolve(a[1])
    elif a[0] == "pages":
        cmd_pages(a[1])
    elif a[0] == "grid":
        cmd_grid(a[1], a[2], a[3])
    elif a[0] == "crop":
        cmd_crop(a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    else:
        print("unknown cmd", a[0]); sys.exit(1)
