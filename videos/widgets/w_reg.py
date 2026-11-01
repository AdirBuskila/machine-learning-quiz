# -*- coding: utf-8 -*-
"""Regression & neural-network widgets for the reg videos (R01..R09). Every CSS class is prefixed rg- (all widgets/*.css
load into every video of the app). A widget only draws what it is given: every number is computed and asserted in the
script (numpy, exact Fractions where the course uses round numbers); nothing here decides a fact.

  Plot(pfx, xr, yr, ...)          2-D axes in data units: .dot .seg .poly .fn .text .arrow .band .svg ; ids {pfx}-<id>
  fit_plot(...)                   points + a fitted line + vertical residual segments with their values (R01, R02)
  table(pfx, head, rows, ...)     LTR numeric table; cell (r, c) id {pfx}-<r>-<c> (hidden cells listed in `hidden`),
                                  row highlight {pfx}-m<r>
  net(pfx, layers, edges, ...)    a neuron / a small fully connected network (SVG): node {pfx}-n-<key}, value tag
                                  {pfx}-v-<key>, edge highlight {pfx}-he-<a>-<b>, node highlight {pfx}-hn-<key>
  numline(pfx, versions, ...)     net values of examples on a number line, threshold at 0; one layer per version {pfx}-g<k>
  card / chip / eq / stack / row / swap / qpanel / opts      small layout helpers
  num_say / dec_say               numbers as spoken Hebrew (for say=)
  check_say(video)                asserts that no phrase's spoken text contains a Latin letter
"""
import html as _html
import math
import re
from fractions import Fraction as Fr

import stage as st

LATIN = re.compile(r"[A-Za-z]")


def esc(s):
    return _html.escape(str(s), quote=True)


def _h(hidden):
    return " h" if hidden else ""


# ---------------------------------------------------------------- spoken numbers (feminine, as numbers are read)
_U = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
_TEEN = ["עשר", "אחת עשרה", "שתים עשרה", "שלוש עשרה", "ארבע עשרה", "חמש עשרה", "שש עשרה", "שבע עשרה", "שמונה עשרה",
         "תשע עשרה"]
_TENS = {2: "עשרים", 3: "שלושים", 4: "ארבעים", 5: "חמישים", 6: "שישים", 7: "שבעים", 8: "שמונים", 9: "תשעים"}
_HUN = {1: "מאה", 2: "מאתיים", 3: "שלוש מאות", 4: "ארבע מאות", 5: "חמש מאות", 6: "שש מאות", 7: "שבע מאות",
        8: "שמונה מאות", 9: "תשע מאות"}
_THO = {1: "אלף", 2: "אלפיים", 3: "שלושת אלפים", 4: "ארבעת אלפים", 5: "חמשת אלפים", 6: "ששת אלפים", 7: "שבעת אלפים",
        8: "שמונת אלפים", 9: "תשעת אלפים"}


def num_say(n):
    """integer -> spoken Hebrew (feminine), up to 9999; negative -> 'מינוס …'"""
    n = int(n)
    if n < 0:
        return "מינוס " + num_say(-n)
    if n < 10:
        return _U[n]
    parts = []
    t, r = divmod(n, 1000)
    if t:
        parts.append(_THO[t])
    h, r = divmod(r, 100)
    if h:
        parts.append(_HUN[h])
    if 10 <= r < 20:
        parts.append(_TEEN[r - 10])
    else:
        d, u = divmod(r, 10)
        if d:
            parts.append(_TENS[d])
        if u:
            parts.append(_U[u])
    if len(parts) > 1:
        parts[-1] = "ו" + parts[-1]
    return " ".join(parts)


def dec_say(x, places=None):
    """a decimal as written: 0.75 -> 'אפס נקודה שבעים וחמש', 0.05 -> 'אפס נקודה אפס חמש', -1.2 -> 'מינוס אחת נקודה שתיים'.
    x may be a string (exactly as printed) or a number (formatted with `places`, trailing zeros dropped)."""
    s = x if isinstance(x, str) else (f"{x:.{places}f}" if places is not None else repr(float(x)))
    s = s.replace("−", "-")
    if "." in s and not isinstance(x, str):
        s = s.rstrip("0").rstrip(".")
    neg = s.startswith("-")
    s = s.lstrip("-")
    ip, _, fp = s.partition(".")
    out = num_say(int(ip))
    if fp:
        if len(fp) <= 2 and not fp.startswith("0"):
            out += " נקודה " + num_say(int(fp))
        else:
            out += " נקודה " + " ".join(_U[int(c)] for c in fp)
    return ("מינוס " if neg else "") + out


def fmt(x, places=2):
    """a number for the screen: trailing zeros dropped, a real minus sign"""
    if isinstance(x, Fr) and x.denominator == 1:
        x = x.numerator
    if isinstance(x, int):
        s = str(x)
    else:
        s = f"{float(x):.{places}f}".rstrip("0").rstrip(".")
        if s in ("-0", ""):
            s = "0"
    return s.replace("-", "−")


def check_say(video):
    """the voice misreads Latin letters: every spoken phrase must be Hebrew, digits and punctuation"""
    bad = []
    for S in video.scenes:
        for p in S.phrases:
            if LATIN.search(p.say):
                bad.append(f"[{S.key}] {p.say}")
    assert not bad, "Latin in spoken text:\n  " + "\n  ".join(bad)


# ---------------------------------------------------------------- small layout pieces
def ltr(s):
    """wrap runs of Latin text (code / English terms, with their digits and symbols) in an LTR isolate"""
    return re.sub(r"[A-Za-z][A-Za-z0-9_²\-+·.,()=/ ]*[A-Za-z0-9²)]|[A-Za-z]",
                  lambda m: f'<span class="rg-en">{m.group(0)}</span>', s)


def card(id_, kind, html_, hidden=True):
    c = st.card(id_, kind, html_)
    return c if hidden else c.replace(' h"', '"', 1)


def chip(id_, html_, kind="", hidden=True):
    return f'<span id="{id_}" class="rg-chip {kind}{_h(hidden)}">{html_}</span>'


def eq(id_, tex_html, size=40, hidden=True, cls=""):
    return f'<div id="{id_}" class="rg-eq {cls}{_h(hidden)}" style="font-size:{size}px">{tex_html}</div>'


def stack(*items, gap=22, cls=""):
    return f'<div class="rg-stack {cls}" style="gap:{gap}px">{"".join(items)}</div>'


def row(*items, gap=28, cls=""):
    return f'<div class="rg-row {cls}" style="gap:{gap}px">{"".join(items)}</div>'


def swap(pfx, items, height, first_visible=True, width=None):
    """several layers in one slot: {pfx}-<k>; layer 0 visible unless first_visible=False"""
    w = f"width:{width}px;" if width else ""
    layers = "".join(f'<div id="{pfx}-{k}" class="{_h(k > 0 or not first_visible).strip()}">{it}</div>'
                     for k, it in enumerate(items))
    return f'<div class="swap rg-swap" id="{pfx}" style="--swap-h:{height}px;{w}">{layers}</div>'


def opts(q, short=None):
    """bank options in bank order as (index, html); short: {index: a shorter display text that keeps the meaning}"""
    short = short or {}
    return [(k, ltr(esc(short.get(k, o)))) for k, o in enumerate(q["options"])]


def opt_index(q, needle):
    hits = [k for k, o in enumerate(q["options"]) if needle in o]
    assert len(hits) == 1, (q["id"], needle, hits)
    return hits[0]


def qpanel(qid, stem_html, opts_list, notes=None, size=None, pfx="q"):
    """exam side panel: id badge + stem + options (no letters: the app shuffles them)"""
    style = f' style="font-size:{size}px"' if size else ""
    return (f'<div class="rg-qp"{style}><div class="qstem"><span class="qid">{esc(qid)}</span>{stem_html}</div>'
            + st.options(pfx, opts_list, notes=notes) + "</div>")


# ---------------------------------------------------------------- tables
def table(pfx, head, rows, hidden=(), size=32, hl_rows=True, cls=""):
    """LTR table. head: [html]; rows: [[html]]. hidden: a set of (r, c) cells shown later (id {pfx}-<r>-<c>);
    a whole hidden column: pass (None, c). Row highlight {pfx}-m<r> (hidden)."""
    hid = set(hidden)
    cols = {c for r, c in hid if r is None}
    th = "".join(f"<th>{h}</th>" for h in head)
    body = []
    for r, cells in enumerate(rows):
        tds = []
        for c, v in enumerate(cells):
            hide = (r, c) in hid or c in cols
            mk = f'<i id="{pfx}-m{r}" class="rg-mk h"></i>' if (c == 0 and hl_rows) else ""
            tds.append(f'<td>{mk}<span id="{pfx}-{r}-{c}" class="rg-cv{_h(hide)}">{v}</span></td>')
        body.append(f'<tr>{"".join(tds)}</tr>')
    return (f'<table class="rg-tab {cls}" id="{pfx}" dir="ltr" style="font-size:{size}px">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>')


# ---------------------------------------------------------------- 2-D plot
class Plot:
    """axes in data units. Every drawn piece may carry an id ({pfx}-<id>) and start hidden."""

    def __init__(self, pfx, xr, yr, w=900, h=560, pad=(78, 26, 30, 62), xticks=(), yticks=(), xlabel="", ylabel="",
                 grid=True, x0axis=False, y0axis=False):
        self.pfx, self.xr, self.yr, self.w, self.h = pfx, xr, yr, w, h
        self.L, self.R, self.T, self.B = pad
        self.parts = []
        x0, x1 = self.X(xr[0]), self.X(xr[1])
        y0, y1 = self.Y(yr[0]), self.Y(yr[1])
        for t in yticks:
            if grid:
                self.parts.append(f'<line x1="{x0:.1f}" y1="{self.Y(t):.1f}" x2="{x1:.1f}" y2="{self.Y(t):.1f}" class="rg-grid"/>')
            self.parts.append(f'<text x="{x0 - 12:.1f}" y="{self.Y(t) + 8:.1f}" class="rg-tl e">{fmt(t)}</text>')
        for t in xticks:
            if grid:
                self.parts.append(f'<line x1="{self.X(t):.1f}" y1="{y0:.1f}" x2="{self.X(t):.1f}" y2="{y1:.1f}" class="rg-grid"/>')
            self.parts.append(f'<text x="{self.X(t):.1f}" y="{y0 + 34:.1f}" class="rg-tl">{fmt(t)}</text>')
        ax_y = self.Y(0) if (x0axis and yr[0] <= 0 <= yr[1]) else y0
        ax_x = self.X(0) if (y0axis and xr[0] <= 0 <= xr[1]) else x0
        self.parts.append(f'<line x1="{x0:.1f}" y1="{ax_y:.1f}" x2="{x1:.1f}" y2="{ax_y:.1f}" class="rg-ax"/>')
        self.parts.append(f'<line x1="{ax_x:.1f}" y1="{y0:.1f}" x2="{ax_x:.1f}" y2="{y1:.1f}" class="rg-ax"/>')
        if xlabel:
            self.parts.append(f'<text x="{x1:.1f}" y="{ax_y - 14:.1f}" class="rg-axl" text-anchor="end">{xlabel}</text>')
        if ylabel:
            self.parts.append(f'<text x="{ax_x + 12:.1f}" y="{y1 + 4:.1f}" class="rg-axl" text-anchor="start">{ylabel}</text>')

    def X(self, x):
        return self.L + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.w - self.L - self.R)

    def Y(self, y):
        return self.h - self.B - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.h - self.T - self.B)

    def _id(self, id_, hidden):
        return (f' id="{self.pfx}-{id_}"' if id_ else ""), _h(hidden)

    def dot(self, x, y, kind="k-a", id_=None, hidden=False, r=11):
        i, h = self._id(id_, hidden)
        self.parts.append(f'<circle{i} cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" class="rg-dot {kind}{h}"/>')

    def ring(self, x, y, id_=None, hidden=True, r=20):
        i, h = self._id(id_, hidden)
        self.parts.append(f'<circle{i} cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" class="rg-ring{h}"/>')

    def seg(self, x1, y1, x2, y2, kind="k-ink", id_=None, hidden=False, width=3, dash=False):
        i, h = self._id(id_, hidden)
        d = " dash" if dash else ""
        self.parts.append(f'<line{i} x1="{self.X(x1):.1f}" y1="{self.Y(y1):.1f}" x2="{self.X(x2):.1f}" y2="{self.Y(y2):.1f}" '
                          f'class="rg-ln {kind}{d}{h}" stroke-width="{width}"/>')

    def poly(self, pts, kind="k-ink", id_=None, hidden=False, width=4, dash=False):
        i, h = self._id(id_, hidden)
        d = " dash" if dash else ""
        p = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in pts)
        self.parts.append(f'<polyline{i} points="{p}" class="rg-poly {kind}{d}{h}" stroke-width="{width}"/>')

    def fn(self, f, a, b, n=160, **kw):
        """plot y = f(x) on [a, b], clipped to the y range"""
        pts = []
        for k in range(n + 1):
            x = a + (b - a) * k / n
            y = f(x)
            if self.yr[0] - 1e-9 <= y <= self.yr[1] + 1e-9:
                pts.append((x, y))
        self.poly(pts, **kw)

    def arrow(self, x1, y1, x2, y2, kind="k-b", id_=None, hidden=True, width=4):
        i, h = self._id(id_, hidden)
        self.parts.append(f'<line{i} x1="{self.X(x1):.1f}" y1="{self.Y(y1):.1f}" x2="{self.X(x2):.1f}" y2="{self.Y(y2):.1f}" '
                          f'class="rg-ln {kind}{h}" stroke-width="{width}" marker-end="url(#{self.pfx}-ah-{kind})"/>')

    def text(self, x, y, s, kind="k-ink", id_=None, hidden=False, size=28, anchor="middle", dx=0, dy=0, bold=True):
        i, h = self._id(id_, hidden)
        b = " b" if bold else ""
        self.parts.append(f'<text{i} x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" class="rg-tx {kind}{b}{h}" '
                          f'font-size="{size}" text-anchor="{anchor}">{s}</text>')

    def band(self, x1, x2, kind="k-a", id_=None, hidden=True):
        i, h = self._id(id_, hidden)
        self.parts.append(f'<rect{i} x="{self.X(x1):.1f}" y="{self.Y(self.yr[1]):.1f}" width="{self.X(x2) - self.X(x1):.1f}" '
                          f'height="{self.Y(self.yr[0]) - self.Y(self.yr[1]):.1f}" class="rg-band {kind}{h}"/>')

    def rect(self, x1, y1, x2, y2, kind="k-a", id_=None, hidden=True):
        i, h = self._id(id_, hidden)
        xa, xb = sorted((self.X(x1), self.X(x2)))
        ya, yb = sorted((self.Y(y1), self.Y(y2)))
        self.parts.append(f'<rect{i} x="{xa:.1f}" y="{ya:.1f}" width="{xb - xa:.1f}" height="{yb - ya:.1f}" class="rg-sq {kind}{h}"/>')

    def svg(self, scale=1.0, id_=None):
        marks = "".join(f'<marker id="{self.pfx}-ah-{k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
                        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" class="rg-mk-{k}"/></marker>'
                        for k in ("k-a", "k-b", "k-c", "k-ok", "k-bad", "k-ink"))
        i = f' id="{id_}"' if id_ else f' id="{self.pfx}"'
        return (f'<svg class="rg-svg"{i} viewBox="0 0 {self.w} {self.h}" width="{self.w * scale:.0f}" '
                f'height="{self.h * scale:.0f}" direction="ltr"><defs>{marks}</defs>{"".join(self.parts)}</svg>')


def fit_plot(pfx, xs, ys, line, xr, yr, xticks, yticks, res_hidden=True, line_hidden=True, labels=True,
             xlabel="x", ylabel="y", w=900, h=560, kind_line="k-a", errs=None, mean=None):
    """points (x, y), the line ŷ = b + m·x ({pfx}-ln), a residual segment per point ({pfx}-r<i>) with its value
    ŷ − y ({pfx}-rl<i>); mean: draw the horizontal line y = mean ({pfx}-mn) with its own segments ({pfx}-s<i>, {pfx}-sl<i>)"""
    P = Plot(pfx, xr, yr, w=w, h=h, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel)
    b, m = line
    if mean is not None:
        P.seg(xr[0], mean, xr[1], mean, kind="k-b", id_="mn", hidden=True, width=4, dash=True)
        for i, (x, y) in enumerate(zip(xs, ys)):
            P.seg(x, y, x, mean, kind="k-b", id_=f"s{i}", hidden=True, width=5)
            d = y - mean
            P.text(x, (y + mean) / 2, ("+" if d > 0 else "") + fmt(d) if d else "", kind="k-b", id_=f"sl{i}", hidden=True, size=26,
                   anchor="end", dx=-14, dy=9)
    P.seg(xr[0], b + m * xr[0], xr[1], b + m * xr[1], kind=kind_line, id_="ln", hidden=line_hidden, width=4)
    for i, (x, y) in enumerate(zip(xs, ys)):
        yh = b + m * x
        P.seg(x, y, x, yh, kind="k-bad", id_=f"r{i}", hidden=res_hidden, width=5)
        if labels:
            e = errs[i] if errs else yh - y
            right = (yh - y > 0) if m >= 0 else (yh - y < 0)        # the side of the segment away from the line
            P.text(x, (y + yh) / 2, ("+" if e > 0 else "") + fmt(e) if e else "", kind="k-bad", id_=f"rl{i}", hidden=res_hidden,
                   size=26, anchor="start" if right else "end", dx=14 if right else -14, dy=9)
    for i, (x, y) in enumerate(zip(xs, ys)):
        P.dot(x, y, kind="k-ink", id_=f"p{i}", hidden=False)
    return P


# ---------------------------------------------------------------- neuron / network diagram
def net(pfx, layers, edges, w=980, h=560, r=40, labels=None, values=None, hidden_values=True, node_cls=None,
        edge_label_t=0.5, scale=1.0, bias_side=None):
    """layers: [[(key, label)]] left to right; edges: [(a, b, label or '')]; labels drawn on the edge at
    fraction edge_label_t from a. values: {key: text} small value tags under/over nodes ({pfx}-v-<key>, hidden).
    node_cls: {key: extra class} (e.g. 'bias' for the constant-1 node, 'sum' for Σ, 'out')."""
    node_cls = node_cls or {}
    pos = {}
    n = len(layers)
    for li, L in enumerate(layers):
        x = 70 + li * (w - 140) / max(1, n - 1)
        m = len(L)
        for k, (key, _) in enumerate(L):
            y = h / 2 + (k - (m - 1) / 2) * min(150, (h - 120) / max(1, m - 1) if m > 1 else 150)
            pos[key] = (x, y)
    o = [f'<svg class="rg-svg rg-net" id="{pfx}" viewBox="0 0 {w} {h}" width="{w * scale:.0f}" height="{h * scale:.0f}" direction="ltr">',
         f'<defs><marker id="{pfx}-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="#c9cdf0"/></marker>'
         f'<marker id="{pfx}-ahh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="#2dd4bf"/></marker></defs>']
    base, hi, labs = [], [], []
    for a, b, lab in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        d = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        sx, sy, ex, ey = x1 + ux * r, y1 + uy * r, x2 - ux * (r + 4), y2 - uy * (r + 4)
        base.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="rg-edge" marker-end="url(#{pfx}-ah)"/>')
        hi.append(f'<line id="{pfx}-he-{a}-{b}" x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="rg-edge-hi h" '
                  f'marker-end="url(#{pfx}-ahh)"/>')
        if lab:
            t = edge_label_t
            lx, ly = sx + (ex - sx) * t, sy + (ey - sy) * t
            wlab = 14 * len(re.sub(r"<[^>]+>", "", lab)) + 18
            labs.append(f'<g id="{pfx}-el-{a}-{b}"><rect x="{lx - wlab / 2:.1f}" y="{ly - 20:.1f}" width="{wlab:.1f}" height="38" rx="9" '
                        f'class="rg-elbg"/><text x="{lx:.1f}" y="{ly + 9:.1f}" class="rg-elab">{lab}</text></g>')
    nodes = []
    for key, (x, y) in pos.items():
        lab = next(l for L in layers for k, l in L if k == key)
        cls = node_cls.get(key, "")
        nodes.append(f'<g id="{pfx}-n-{key}"><circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="rg-node {cls}"/>'
                     f'<text x="{x:.1f}" y="{y + 10:.1f}" class="rg-nlab {cls}">{lab}</text></g>'
                     f'<circle id="{pfx}-hn-{key}" cx="{x:.1f}" cy="{y:.1f}" r="{r + 8}" class="rg-node-hi h"/>')
        if values and key in values:
            vy = y - r - 16
            vals = values[key] if isinstance(values[key], (list, tuple)) else [values[key]]
            for k, v in enumerate(vals):          # several value tags in one place: {pfx}-v-<key>-<k> (or {pfx}-v-<key>)
                vid = f"{pfx}-v-{key}" if len(vals) == 1 else f"{pfx}-v-{key}-{k}"
                nodes.append(f'<text id="{vid}" x="{x:.1f}" y="{vy:.1f}" class="rg-val{_h(hidden_values)}">{v}</text>')
    o += base + hi + labs + nodes + ["</svg>"]
    return "".join(o)


# ---------------------------------------------------------------- number line of net values
def numline(pfx, versions, lo, hi_, ticks, w=980, h=250, labels=None, thr_label="0"):
    """versions: [[(net, true_positive: bool)]] — layer k = {pfx}-g<k> (layer 0 visible). A dot right of 0 is predicted
    positive (filled teal), left of 0 negative (filled rose); a ring marks a truly positive example."""
    L, R = 60, 40
    X = lambda v: L + (v - lo) / (hi_ - lo) * (w - L - R)
    y = 120
    o = [f'<svg class="rg-svg" id="{pfx}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" direction="ltr">',
         f'<line x1="{L}" y1="{y}" x2="{w - R}" y2="{y}" class="rg-ax"/>']
    for t in ticks:
        o.append(f'<line x1="{X(t):.1f}" y1="{y - 10}" x2="{X(t):.1f}" y2="{y + 10}" class="rg-ax"/>'
                 f'<text x="{X(t):.1f}" y="{y + 48}" class="rg-tl">{fmt(t)}</text>')
    o.append(f'<line x1="{X(0):.1f}" y1="{y - 80}" x2="{X(0):.1f}" y2="{y + 20}" class="rg-thr"/>')
    for k, V in enumerate(versions):
        g = [f'<g id="{pfx}-g{k}" class="{"" if k == 0 else "h"}">']
        for i, (v, pos) in enumerate(V):
            kind = "k-ok" if v >= 0 else "k-bad"
            yy = y - 34 - (i % 2) * 34
            g.append(f'<circle cx="{X(v):.1f}" cy="{yy}" r="13" class="rg-dot {kind}"/>')
            if pos:
                g.append(f'<circle cx="{X(v):.1f}" cy="{yy}" r="21" class="rg-pring"/>')
        g.append("</g>")
        o += g
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- spoken English terms (used inside say= strings)
SAY = {
    "MSE": "אֶם אֶס אִי", "MAE": "אֶם אֵיי אִי", "RMSE": "אָר אֶם אֶס אִי", "SSE": "אֶס אֶס אִי", "SAE": "אֶס אֵיי אִי",
    "SST": "אֶס אֶס טִי", "R2": "אָר בריבוע", "GD": "גְּרַדְיֶאנְט דִּיסֶנְט", "SGD": "אֶס גִ'י דִּי", "SVM": "אֶס וִי אֶם",
    "batch": "בֶּאץ'", "epoch": "אִיפּוֹק", "bias": "בַּיְיאַס", "sign": "סַיין", "heaviside": "הֶבִיסַייד",
    "sigmoid": "סִיגְמוֹאִיד", "overfitting": "אוֹבֶרְפִיטִינְג", "underfitting": "אַנְדֶרְפִיטִינְג", "dropout": "דְּרוֹפְּאָאוּט",
    "FN": "אֶף אֶן", "FP": "אֶף פִּי", "target": "טַארְגֶט", "early": "אֶרְלִי סְטוֹפִּינְג",
}
