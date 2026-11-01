# -*- coding: utf-8 -*-
"""Widgets for the unsup course videos (U01-U08): scatter plots with clusters and centroids, a dendrogram,
a pixel grid with a kernel window, generic tables, bar charts, token chips and a weighted number line.
Everything is computed by the caller (the video script asserts its numbers); these functions only draw.

Conventions: an element with class `h` starts invisible; ids are prefixed by the widget's pfx. All classes are
prefixed `u-` (every widget css loads into every video of the app). Fonts: Segoe UI / Consolas / Cambria Math only.
"""
import html as _html

K_COLORS = ["#2dd4bf", "#fb7185", "#fbbf24", "#a78bfa"]
GRAY = "#8b90b8"


def esc(s):
    return _html.escape(str(s), quote=True)


def num(v, nd=2):
    """3.0 -> '3', 2.5 -> '2.5', 3.6056 -> '3.61'"""
    v = round(float(v), nd)
    return str(int(v)) if v == int(v) else f"{v:.{nd}f}".rstrip("0").rstrip(".")


def _h(hidden):
    return ' class="h"' if hidden else ""


# ----------------------------------------------------------------------------- 2-D plot
class Plot:
    """an SVG coordinate plane. pt/cen/seg/txt add marks; begin(id)/end() wrap marks in a group shown/hidden together."""

    def __init__(self, pfx, xr, yr, w=820, h=600, pad=56, grid=1, labels=True, scale=1.0):
        self.pfx, self.xr, self.yr, self.w, self.h, self.pad, self.scale = pfx, xr, yr, w, h, pad, scale
        self.out = []
        x0, x1 = xr
        y0, y1 = yr
        g = []
        if grid:
            x = x0
            while x <= x1 + 1e-9:
                g.append(f'<line x1="{self.X(x):.1f}" y1="{self.Y(y0):.1f}" x2="{self.X(x):.1f}" y2="{self.Y(y1):.1f}" class="u-gl"/>')
                if labels:
                    g.append(f'<text x="{self.X(x):.1f}" y="{self.Y(y0) + 34:.1f}" class="u-tick" text-anchor="middle">{num(x)}</text>')
                x += grid
            y = y0
            while y <= y1 + 1e-9:
                g.append(f'<line x1="{self.X(x0):.1f}" y1="{self.Y(y):.1f}" x2="{self.X(x1):.1f}" y2="{self.Y(y):.1f}" class="u-gl"/>')
                if labels:
                    g.append(f'<text x="{self.X(x0) - 16:.1f}" y="{self.Y(y) + 9:.1f}" class="u-tick" text-anchor="end">{num(y)}</text>')
                y += grid
        self.out += g

    def X(self, x):
        x0, x1 = self.xr
        return self.pad + (x - x0) / (x1 - x0) * (self.w - 2 * self.pad)

    def Y(self, y):
        y0, y1 = self.yr
        return self.h - self.pad - (y - y0) / (y1 - y0) * (self.h - 2 * self.pad)

    def begin(self, id_, hidden=True):
        self.out.append(f'<g id="{id_}"{_h(hidden)}>')

    def end(self):
        self.out.append("</g>")

    def pt(self, x, y, k=None, id_=None, hidden=False, r=13, label=None, ring=False):
        fill = GRAY if k is None else K_COLORS[k]
        i = f' id="{id_}"' if id_ else ""
        cls = ' class="h"' if hidden else ""
        lab = (f'<text x="{self.X(x) + 18:.1f}" y="{self.Y(y) - 14:.1f}" class="u-plab">{esc(label)}</text>'
               if label else "")
        rr = f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r + 9}" class="u-ring"/>' if ring else ""
        self.out.append(f'<g{i}{cls}>{rr}<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{fill}" '
                        f'class="u-pt"/>{lab}</g>')

    def cen(self, x, y, k, id_=None, hidden=True, label=None, s=20, side="r"):
        cx, cy = self.X(x), self.Y(y)
        i = f' id="{id_}"' if id_ else ""
        dx, anc = (26, "start") if side == "r" else (-26, "end")
        lab = (f'<text x="{cx + dx:.1f}" y="{cy + 10:.1f}" class="u-clab" fill="{K_COLORS[k]}" text-anchor="{anc}">'
               f'{esc(label)}</text>' if label else "")
        self.out.append(f'<g{i}{_h(hidden)}><path d="M{cx - s},{cy} L{cx},{cy - s} L{cx + s},{cy} L{cx},{cy + s} Z" '
                        f'fill="{K_COLORS[k]}" class="u-cen"/>{lab}</g>')

    def seg(self, a, b, id_=None, hidden=True, k=None, dash=False, label=None, lab_dx=0, lab_dy=-12, width=4):
        col = GRAY if k is None else K_COLORS[k]
        i = f' id="{id_}"' if id_ else ""
        d = ' stroke-dasharray="10 9"' if dash else ""
        x1, y1, x2, y2 = self.X(a[0]), self.Y(a[1]), self.X(b[0]), self.Y(b[1])
        lab = ""
        if label:
            lab = (f'<text x="{(x1 + x2) / 2 + lab_dx:.1f}" y="{(y1 + y2) / 2 + lab_dy:.1f}" class="u-slab" text-anchor="middle" '
                   f'fill="{col}">{esc(label)}</text>')
        self.out.append(f'<g{i}{_h(hidden)}><line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{col}" stroke-width="{width}"{d}/>{lab}</g>')

    def txt(self, x, y, s, id_=None, hidden=False, cls="u-plab", anchor="middle"):
        i = f' id="{id_}"' if id_ else ""
        c = f"{cls} h" if hidden else cls
        self.out.append(f'<text{i} x="{self.X(x):.1f}" y="{self.Y(y):.1f}" class="{c}" text-anchor="{anchor}">{esc(s)}</text>')

    def svg(self):
        return (f'<svg class="u-plot" id="{self.pfx}" viewBox="0 0 {self.w} {self.h}" width="{self.w * self.scale:.0f}" '
                f'height="{self.h * self.scale:.0f}" direction="ltr">{"".join(self.out)}</svg>')


# ----------------------------------------------------------------------------- number line (1-D data, weights)
def numline(pfx, xs, lo, hi, layers, w=900, h=230, scale=1.0, ticks=None):
    """1-D points on a line. layers: [(id, hidden, [(x, k, radius, label_or_None)], threshold_or_None)].
    k = colour index (None = gray); a threshold draws a dashed vertical line (a decision stump)."""
    pad = 50
    X = lambda x: pad + (x - lo) / (hi - lo) * (w - 2 * pad)
    y = h * 0.55
    o = [f'<svg class="u-plot" id="{pfx}" viewBox="0 0 {w} {h}" width="{w * scale:.0f}" height="{h * scale:.0f}" direction="ltr">',
         f'<line x1="{pad - 20}" y1="{y}" x2="{w - pad + 20}" y2="{y}" class="u-axis"/>']
    for t in (ticks if ticks is not None else xs):
        o.append(f'<text x="{X(t):.1f}" y="{y + 62}" class="u-tick" text-anchor="middle">{num(t)}</text>')
    for lid, hidden, pts, thr in layers:
        o.append(f'<g id="{lid}"{_h(hidden)}>')
        if thr is not None:
            o.append(f'<line x1="{X(thr):.1f}" y1="{y - 95}" x2="{X(thr):.1f}" y2="{y + 30}" class="u-thr"/>')
        for x, k, r, lab in pts:
            fill = GRAY if k is None else K_COLORS[k]
            o.append(f'<circle cx="{X(x):.1f}" cy="{y}" r="{r:.1f}" fill="{fill}" class="u-pt"/>')
            if lab:
                o.append(f'<text x="{X(x):.1f}" y="{y - max(r, 12) - 12:.1f}" class="u-plab" text-anchor="middle">{esc(lab)}</text>')
        o.append("</g>")
    o.append("</svg>")
    return "".join(o)


# ----------------------------------------------------------------------------- dendrogram
def dendro(pfx, labels, merges, order, ymax, w=860, h=520, scale=1.0, cut=None):
    """labels: leaf names by original index; merges: scipy-style [(i, j, height)] where new clusters get ids n, n+1, …;
    order: leaf order left to right. Ids: {pfx}-m{k} the k-th merge, {pfx}-cut a dashed cut line at height `cut`."""
    n = len(labels)
    pad_l, pad_r, pad_t, pad_b = 90, 40, 30, 70
    xs = {leaf: pad_l + (i + 0.5) * (w - pad_l - pad_r) / n for i, leaf in enumerate(order)}
    Y = lambda v: h - pad_b - v / ymax * (h - pad_t - pad_b)
    top = {i: 0.0 for i in range(n)}
    o = [f'<svg class="u-plot" id="{pfx}" viewBox="0 0 {w} {h}" width="{w * scale:.0f}" height="{h * scale:.0f}" direction="ltr">',
         f'<line x1="{pad_l - 20}" y1="{Y(0)}" x2="{pad_l - 20}" y2="{Y(ymax)}" class="u-axis"/>']
    step = 1 if ymax <= 8 else 2 if ymax <= 16 else 5
    v = 0
    while v <= ymax + 1e-9:
        o.append(f'<text x="{pad_l - 32}" y="{Y(v) + 9:.1f}" class="u-tick" text-anchor="end">{num(v)}</text>')
        v += step
    for leaf in range(n):
        o.append(f'<text x="{xs[leaf]:.1f}" y="{h - pad_b + 44}" class="u-leaf">{esc(labels[leaf])}</text>')
    for k, (a, b, d) in enumerate(merges):
        xa, xb = xs[a], xs[b]
        ya, yb, yd = Y(top[a]), Y(top[b]), Y(d)
        o.append(f'<path id="{pfx}-m{k}" class="u-dg h" d="M{xa:.1f},{ya:.1f} L{xa:.1f},{yd:.1f} L{xb:.1f},{yd:.1f} '
                 f'L{xb:.1f},{yb:.1f}"/>')
        new = n + k
        xs[new] = (xa + xb) / 2
        top[new] = d
    if cut is not None:
        o.append(f'<line id="{pfx}-cut" class="u-cut h" x1="{pad_l - 30}" y1="{Y(cut):.1f}" x2="{w - 10}" y2="{Y(cut):.1f}"/>')
    o.append("</svg>")
    return "".join(o)


# ----------------------------------------------------------------------------- pixel grid
def pixgrid(pfx, M, cell=74, shade=True, hidden_vals=False, fmt=None, wins=(), hls=(), font=30, binary=False):
    """a matrix as a grid. shade: background = gray level (0..255; binary: 0 black, 1 white). Ids: {pfx}-{r}-{c}-v the
    value (hidden if hidden_vals), {pfx}-hl-{r}-{c} a cell ring, {pfx}-w-{r}-{c} a 3x3 window centred at (r, c)."""
    fmt = fmt or num
    R, C = len(M), len(M[0])
    o = [f'<div class="u-gridwrap" id="{pfx}" style="width:{C * cell}px;height:{R * cell}px">'
         f'<div class="u-grid" style="grid-template-columns:repeat({C},{cell}px);grid-auto-rows:{cell}px;font-size:{font}px">']
    for r in range(R):
        for c in range(C):
            v = M[r][c]
            if shade:
                g = 255 * v if binary else max(0, min(255, round(v)))
                ink = "#0a0c18" if g > 140 else "#eaedff"
                st = f' style="background:rgb({g},{g},{g});color:{ink}"'
            else:
                st = ""
            o.append(f'<div class="u-cell"{st}><span id="{pfx}-{r}-{c}-v"{_h(hidden_vals)}>{fmt(v)}</span></div>')
    o.append("</div>")
    for (r, c) in hls:
        o.append(f'<i id="{pfx}-hl-{r}-{c}" class="u-hl h" style="left:{c * cell}px;top:{r * cell}px;width:{cell}px;height:{cell}px"></i>')
    for (r, c) in wins:
        o.append(f'<i id="{pfx}-w-{r}-{c}" class="u-win h" style="left:{(c - 1) * cell}px;top:{(r - 1) * cell}px;'
                 f'width:{3 * cell}px;height:{3 * cell}px"></i>')
    o.append("</div>")
    return "".join(o)


# ----------------------------------------------------------------------------- tables
def table(pfx, head, rows, hide=(), cls="", hl=(), rowhead=True):
    """head: list of header html; rows: list of lists of cell html. hide: {(r, c)} cells whose content starts hidden
    (id {pfx}-{r}-{c}); hl: {(r, c)} cells with a highlight ring {pfx}-hl-{r}-{c}. LTR."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = []
    for r, row in enumerate(rows):
        tds = []
        for c, v in enumerate(row):
            ring = f'<i id="{pfx}-hl-{r}-{c}" class="u-thl h"></i>' if (r, c) in hl else ""
            tag = "th" if (rowhead and c == 0) else "td"
            tds.append(f'<{tag}>{ring}<span id="{pfx}-{r}-{c}"{_h((r, c) in hide)}>{v}</span></{tag}>')
        body.append(f'<tr>{"".join(tds)}</tr>')
    thead = f"<thead><tr>{th}</tr></thead>" if head else ""
    return f'<table class="u-tab {cls}" id="{pfx}" dir="ltr">{thead}<tbody>{"".join(body)}</tbody></table>'


def all_cells(pfx, rows, cols, r0=0, c0=0):
    return [f"{pfx}-{r}-{c}" for r in range(r0, rows) for c in range(c0, cols)]


# ----------------------------------------------------------------------------- bar / line charts
def bars(pfx, values, labels, vmax, w=760, h=420, scale=1.0, fmt=None, hidden=True, k=0, val_hidden=None, ylabel=None):
    """vertical bars. Ids: {pfx}-b{i} bar (+ its value label). hidden: bars start invisible."""
    fmt = fmt or num
    n = len(values)
    pad_l, pad_b, pad_t = 30, 60, 50
    bw = (w - pad_l * 2) / n
    Y = lambda v: h - pad_b - v / vmax * (h - pad_b - pad_t)
    o = [f'<svg class="u-plot" id="{pfx}" viewBox="0 0 {w} {h}" width="{w * scale:.0f}" height="{h * scale:.0f}" direction="ltr">',
         f'<line x1="{pad_l - 10}" y1="{Y(0)}" x2="{w - pad_l + 10}" y2="{Y(0)}" class="u-axis"/>']
    for i, (v, lab) in enumerate(zip(values, labels)):
        x = pad_l + i * bw + bw * 0.18
        bwi = bw * 0.64
        o.append(f'<text x="{x + bwi / 2:.1f}" y="{h - pad_b + 40}" class="u-tick" text-anchor="middle">{esc(lab)}</text>')
        o.append(f'<g id="{pfx}-b{i}"{_h(hidden)}><rect x="{x:.1f}" y="{Y(v):.1f}" width="{bwi:.1f}" height="{Y(0) - Y(v):.1f}" '
                 f'rx="6" fill="{K_COLORS[k]}" class="u-bar"/>'
                 f'<text x="{x + bwi / 2:.1f}" y="{Y(v) - 12:.1f}" class="u-bval" text-anchor="middle">{esc(fmt(v))}</text></g>')
    o.append("</svg>")
    return "".join(o)


def curve(pfx, xs, ys, xr, yr, w=820, h=470, scale=1.0, ystep=None, hl=None, labels=True):
    """a polyline revealed point by point: {pfx}-p{i} point i (with its value) and the segment into it."""
    p = Plot(pfx, xr, yr, w=w, h=h, grid=0, pad=80)
    o = p.out
    x0, x1 = xr
    y0, y1 = yr
    o.append(f'<line x1="{p.X(x0):.1f}" y1="{p.Y(y0):.1f}" x2="{p.X(x1):.1f}" y2="{p.Y(y0):.1f}" class="u-axis"/>')
    o.append(f'<line x1="{p.X(x0):.1f}" y1="{p.Y(y0):.1f}" x2="{p.X(x0):.1f}" y2="{p.Y(y1):.1f}" class="u-axis"/>')
    for x in xs:
        o.append(f'<text x="{p.X(x):.1f}" y="{p.Y(y0) + 36:.1f}" class="u-tick" text-anchor="middle">{num(x)}</text>')
    if ystep:
        v = y0
        while v <= y1 + 1e-9:
            o.append(f'<text x="{p.X(x0) - 12:.1f}" y="{p.Y(v) + 9:.1f}" class="u-tick" text-anchor="end">{num(v)}</text>')
            v += ystep
    for i, (x, y) in enumerate(zip(xs, ys)):
        seg = ""
        if i:
            seg = (f'<line x1="{p.X(xs[i - 1]):.1f}" y1="{p.Y(ys[i - 1]):.1f}" x2="{p.X(x):.1f}" y2="{p.Y(y):.1f}" '
                   f'class="u-cline"/>')
        lab = f'<text x="{p.X(x) + 10:.1f}" y="{p.Y(y) - 24:.1f}" class="u-bval" text-anchor="start">{num(y)}</text>' if labels else ""
        o.append(f'<g id="{pfx}-p{i}" class="h">{seg}<circle cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="10" '
                 f'fill="{K_COLORS[0]}" class="u-pt"/>{lab}</g>')
    if hl is not None:
        x, y = xs[hl], ys[hl]
        o.append(f'<circle id="{pfx}-hl" class="u-ring h" cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="26"/>')
    return p.svg()


# ----------------------------------------------------------------------------- tokens
def chips(pfx, toks, hidden=False, strike=(), cls=""):
    """tokens as chips, RTL order for Hebrew. Ids: {pfx}-{i} chip, {pfx}-x{i} a strike-through over chip i."""
    o = [f'<div class="u-chips {cls}" id="{pfx}">']
    for i, t in enumerate(toks):
        s = f'<i id="{pfx}-x{i}" class="u-strike h"></i>' if i in strike else ""
        o.append(f'<span class="u-chip{" h" if hidden else ""}" id="{pfx}-{i}">{esc(t)}{s}</span>')
    o.append("</div>")
    return "".join(o)


def ltr(s):
    """an LTR island inside RTL text (a formula written in plain text)"""
    return f'<span style="direction:ltr;unicode-bidi:isolate;display:inline-block">{s}</span>'


def stack(*parts, gap=26, cls=""):
    return f'<div class="u-stack {cls}" style="gap:{gap}px">{"".join(parts)}</div>'


def row(*parts, gap=40, cls=""):
    return f'<div class="u-row {cls}" style="gap:{gap}px">{"".join(parts)}</div>'


def big(id_, html_, hidden=True, size=46):
    return f'<div id="{id_}" class="u-big{" h" if hidden else ""}" style="font-size:{size}px">{html_}</div>'


def note(id_, html_, hidden=True, size=32):
    """a short RTL line on the stage"""
    return f'<div id="{id_}" class="u-note{" h" if hidden else ""}" style="font-size:{size}px">{html_}</div>'
