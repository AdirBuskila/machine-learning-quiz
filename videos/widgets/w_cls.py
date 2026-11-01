# -*- coding: utf-8 -*-
"""Widgets and checked arithmetic for the classifier-course videos (C01..C09: KNN, decision trees, Naive Bayes).
Every class is prefixed cls- (all widget css files load into every video of the app). A widget only draws what it is
given; every number a script shows is computed by the helpers below and asserted in the script (and, where it
applies, cross-checked against scikit-learn / scipy).

  P(sub, say=..., do=..., hold=...)   vid.P whose spoken text has its decimals read in Hebrew words ("0.918" ->
                                      "אפס נקודה תשע אחת שמונה") and which refuses any Latin letter left in it
  Plot(pfx, xr, yr, ...)              2-D axes in data units; ids "<pfx>-<id>" (see the methods)
  table(pfx, head, rows, ...)         an LTR table. cell (i, j): value {pfx}-v{i}-{j}, highlight {pfx}-h{i}-{j},
                                      dim {pfx}-d{i}-{j}; hidden columns `hide_cols`; rows(...)/cells(...) give id lists
  Tree(pfx, w, h)                     a decision tree drawn box by box: .node(id, x, y, html) -> {pfx}-n-{id},
                                      .edge(a, b, label) -> {pfx}-e-{a}-{b}; .html()
  dots(counts)                        class composition as coloured dots (● Yes / ● No …)
  chip / eq / stem / note             small text helpers
"""
import html as _html
import math
import re

import vid

LATIN = re.compile(r"[A-Za-z]")
KINDS = ("a", "b", "c", "d", "ink", "bad", "ok", "mut")


def esc(s):
    return _html.escape(str(s), quote=True)


def _h(hidden):
    return "h" if hidden else ""


def fmt(v, nd=3):
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- spoken numbers
DIG = {"0": "אפס", "1": "אחת", "2": "שתיים", "3": "שלוש", "4": "ארבע", "5": "חמש", "6": "שש", "7": "שבע",
       "8": "שמונה", "9": "תשע"}


def dec_he(m):
    whole, frac = m.group(1), m.group(2)
    w = "אפס" if whole == "0" else whole
    return f"{w} נקודה " + " ".join(DIG[c] for c in frac)


def say_numbers(s):
    """0.918 -> אפס נקודה תשע אחת שמונה ; 24.5 -> 24 נקודה חמש (integers are read well as digits)"""
    return re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", dec_he, s)


class P(vid.P):
    @property
    def say(self):
        s = say_numbers(vid.P.say.fget(self))
        if LATIN.search(s):
            raise ValueError(f"Latin letters in the spoken text: {s}\n  -> give P(..., say=...)")
        return s


# ---------------------------------------------------------------- arithmetic (asserted in the scripts)
def l1(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))


def l2(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def linf(p, q):
    return max(abs(a - b) for a, b in zip(p, q))


def cos_sim(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))


def neighbours(train, q, k, metric=l2):
    """indexes of the k nearest training points (sorted), refusing a tie at the k-th place"""
    d = sorted((metric(p, q), i) for i, p in enumerate(train))
    if k < len(d):
        assert d[k][0] - d[k - 1][0] > 1e-9, f"tie at the k-th neighbour: {d[:k + 1]}"
    return [i for _, i in d[:k]], d


def H(counts, base=2):
    """entropy of a class-count list (zeros allowed)"""
    n = sum(counts)
    return -sum(c / n * math.log(c / n, base) for c in counts if c) + 0.0


def cond_H(groups, base=2):
    """H(Y|F) for groups = [[count per class] per value of F]: the size-weighted average of the branch entropies"""
    n = sum(sum(g) for g in groups)
    return sum(sum(g) / n * H(g, base) for g in groups)


def counts(rows, col, cls_col, value, classes):
    sub = [r for r in rows if r[col] == value]
    return [sum(1 for r in sub if r[cls_col] == c) for c in classes]


# ---------------------------------------------------------------- 2-D plot
class Plot:
    def __init__(self, pfx, xr, yr, w=640, h=520, xticks=(), yticks=(), xname="x", yname="y", grid=True, pad=None,
                 tick_fmt=None):
        self.pfx, self.xr, self.yr, self.w, self.h = pfx, xr, yr, w, h
        self.pl, self.pr, self.pt, self.pb = pad or (64, 26, 24, 60)
        self.xticks, self.yticks, self.xname, self.yname, self.grid = xticks, yticks, xname, yname, grid
        self.tick_fmt = tick_fmt or fmt
        self.layers = {"under": [], "mid": [], "pts": [], "top": []}

    def X(self, v):
        return self.pl + (v - self.xr[0]) * (self.w - self.pl - self.pr) / (self.xr[1] - self.xr[0])

    def Y(self, v):
        return self.pt + (self.yr[1] - v) * (self.h - self.pt - self.pb) / (self.yr[1] - self.yr[0])

    def _g(self, layer, id_, body, hidden):
        self.layers[layer].append(f'<g id="{self.pfx}-{id_}" class="{_h(hidden)}">{body}</g>')
        return f"{self.pfx}-{id_}"

    def pts(self, id_, P_, kinds="ink", hidden=False, r=13, each=False, shapes=None, labels=None, lab_dx=0, lab_dy=0):
        """kinds: one kind or one per point; shapes 'o' circle / 's' square / 't' triangle per point;
        labels per point (drawn above-right); each=True gives point i the id {id_}-{i}"""
        out = []
        for i, (x, y) in enumerate(P_):
            k = kinds if isinstance(kinds, str) else kinds[i]
            sh = (shapes or "o" * len(P_))[i]
            cx, cy = self.X(x), self.Y(y)
            if sh == "s":
                m = f'<rect class="cls-dot k-{k}" x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" height="{2 * r}" rx="3"/>'
            elif sh == "t":
                m = (f'<path class="cls-dot k-{k}" d="M{cx:.1f},{cy - r * 1.25:.1f} L{cx + r * 1.15:.1f},{cy + r * .8:.1f} '
                     f'L{cx - r * 1.15:.1f},{cy + r * .8:.1f} Z"/>')
            else:
                m = f'<circle class="cls-dot k-{k}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>'
            if labels and labels[i]:
                m += (f'<text class="cls-pl" x="{cx + r + 4 + lab_dx:.1f}" y="{cy - r - 2 + lab_dy:.1f}">'
                      f'{esc(labels[i])}</text>')
            out.append(f'<g id="{self.pfx}-{id_}-{i}">{m}</g>' if each else m)
        return self._g("pts", id_, "".join(out), hidden)

    def ring(self, id_, x, y, r=24, kind="ok", hidden=True):
        return self._g("top", id_, f'<circle class="cls-ring k-{kind}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>',
                       hidden)

    def circle(self, id_, x, y, rad, kind="a", hidden=True):
        rx = abs(self.X(x + rad) - self.X(x))
        ry = abs(self.Y(y + rad) - self.Y(y))
        return self._g("under", id_, f'<ellipse class="cls-circ k-{kind}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" '
                                     f'rx="{rx:.1f}" ry="{ry:.1f}"/>', hidden)

    def segs(self, id_, S, kind="a", hidden=True, cls="cls-seg"):
        body = "".join(f'<line class="{cls} k-{kind}" x1="{self.X(p[0]):.1f}" y1="{self.Y(p[1]):.1f}" '
                       f'x2="{self.X(q[0]):.1f}" y2="{self.Y(q[1]):.1f}"/>' for p, q in S)
        return self._g("under", id_, body, hidden)

    def path(self, id_, pts, kind="a", hidden=True, cls="cls-line", dots=False):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        body = f'<path class="{cls} k-{kind}" d="{d}"/>'
        if dots:
            body += "".join(f'<circle class="cls-dot k-{kind}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="7"/>'
                            for x, y in pts)
        return self._g("mid", id_, body, hidden)

    def arrow(self, id_, p, q, kind="a", hidden=True):
        x1, y1, x2, y2 = self.X(p[0]), self.Y(p[1]), self.X(q[0]), self.Y(q[1])
        a = math.atan2(y2 - y1, x2 - x1)
        L = 18
        h1 = (x2 - L * math.cos(a - 0.42), y2 - L * math.sin(a - 0.42))
        h2 = (x2 - L * math.cos(a + 0.42), y2 - L * math.sin(a + 0.42))
        body = (f'<line class="cls-arw k-{kind}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
                f'<path class="cls-arh k-{kind}" d="M{x2:.1f},{y2:.1f} L{h1[0]:.1f},{h1[1]:.1f} L{h2[0]:.1f},{h2[1]:.1f} Z"/>')
        return self._g("mid", id_, body, hidden)

    def vline(self, id_, x, kind="b", hidden=True):
        return self._g("under", id_, f'<line class="cls-vl k-{kind}" x1="{self.X(x):.1f}" y1="{self.Y(self.yr[1]):.1f}" '
                                     f'x2="{self.X(x):.1f}" y2="{self.Y(self.yr[0]):.1f}"/>', hidden)

    def text(self, id_, x, y, s, kind="ink", hidden=True, anchor="middle", dx=0, dy=0, size=26):
        return self._g("top", id_, f'<text class="cls-tx k-{kind}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" '
                                   f'text-anchor="{anchor}" font-size="{size}">{esc(s)}</text>', hidden)

    def svg(self, hidden=False):
        o = []
        x0, x1 = self.X(self.xr[0]), self.X(self.xr[1])
        y0, y1 = self.Y(self.yr[0]), self.Y(self.yr[1])
        for t in self.yticks:
            if self.grid:
                o.append(f'<line class="cls-grid" x1="{x0:.1f}" x2="{x1:.1f}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            o.append(f'<text class="cls-tl y" x="{x0 - 12:.1f}" y="{self.Y(t) + 9:.1f}" text-anchor="end">'
                     f'{self.tick_fmt(t)}</text>')
        for t in self.xticks:
            if self.grid:
                o.append(f'<line class="cls-grid" x1="{self.X(t):.1f}" x2="{self.X(t):.1f}" y1="{y1:.1f}" y2="{y0:.1f}"/>')
            o.append(f'<text class="cls-tl" x="{self.X(t):.1f}" y="{y0 + 38:.1f}">{self.tick_fmt(t)}</text>')
        o.append(f'<line class="cls-ax" x1="{x0:.1f}" x2="{x1:.1f}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
        if self.yticks:
            o.append(f'<line class="cls-ax" x1="{x0:.1f}" x2="{x0:.1f}" y1="{y1:.1f}" y2="{y0:.1f}"/>')
        if self.xname:
            o.append(f'<text class="cls-an" x="{x1:.1f}" y="{y0 - 12:.1f}" text-anchor="end">{esc(self.xname)}</text>')
        if self.yname and self.yticks:
            o.append(f'<text class="cls-an" x="{x0 + 12:.1f}" y="{y1 + 26:.1f}">{esc(self.yname)}</text>')
        body = self.layers["under"] + self.layers["mid"] + self.layers["pts"] + self.layers["top"]
        return (f'<svg class="cls-svg {_h(hidden)}" id="{self.pfx}" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" direction="ltr">{"".join(o)}{"".join(body)}</svg>')


# ---------------------------------------------------------------- table
def table(pfx, head, rows, hide_cols=(), hide_rows=False, size=30, rtl=False, hidden=False, head_ids=False):
    """cell (i, j): <td> with a highlight {pfx}-h{i}-{j} and a dim {pfx}-d{i}-{j} (both hidden) and the value
    {pfx}-v{i}-{j} (hidden when j in hide_cols, or when hide_rows). head cell j: {pfx}-th{j} when head_ids
    (hidden when j in hide_cols)."""
    ths = []
    for j, h in enumerate(head):
        idt = f' id="{pfx}-th{j}"' if head_ids else ""
        cls = ' class="h"' if (j in hide_cols and head_ids) else ""
        ths.append(f"<th><span{idt}{cls}>{h}</span></th>")
    body = []
    for i, row in enumerate(rows):
        tds = []
        for j, c in enumerate(row):
            hid = "h" if (j in hide_cols or hide_rows) else ""
            tds.append(f'<td><i id="{pfx}-h{i}-{j}" class="cls-hi h"></i><i id="{pfx}-d{i}-{j}" class="cls-dim h"></i>'
                       f'<span id="{pfx}-v{i}-{j}" class="cls-v {hid}">{c}</span></td>')
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (f'<table class="cls-tab {_h(hidden)}" id="{pfx}" dir="{"rtl" if rtl else "ltr"}" style="font-size:{size}px">'
            f'<thead><tr>{"".join(ths)}</tr></thead><tbody>{"".join(body)}</tbody></table>')


def cells(pfx, kind, rows_, cols_):
    """ids of a block of cells: kind 'h' (highlight), 'd' (dim) or 'v' (value)"""
    return [f"{pfx}-{kind}{i}-{j}" for i in rows_ for j in cols_]


# ---------------------------------------------------------------- decision tree
class Tree:
    """absolutely positioned node boxes (HTML, so $TeX$ works) over an SVG of edges. Coordinates are box centres."""
    def __init__(self, pfx, w=1000, h=640):
        self.pfx, self.w, self.h = pfx, w, h
        self.nodes, self.edges, self.pos = [], [], {}

    def node(self, id_, x, y, html_, kind="", hidden=True, width=None):
        self.pos[id_] = (x, y)
        st = f"left:{x}px;top:{y}px" + (f";width:{width}px" if width else "")
        self.nodes.append(f'<div id="{self.pfx}-n-{id_}" class="cls-node {kind} {_h(hidden)}" style="{st}">'
                          f'<i id="{self.pfx}-k-{id_}" class="cls-nhl h"></i>{html_}</div>')
        return f"{self.pfx}-n-{id_}"

    def edge(self, a, b, label="", hidden=True, dy=(60, 60)):
        (x1, y1), (x2, y2) = self.pos[a], self.pos[b]
        y1, y2 = y1 + dy[0], y2 - dy[1]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        side = -1 if x2 < x1 else 1
        lab = (f'<text class="cls-el" x="{mx + side * 18:.0f}" y="{my + 8:.0f}" '
               f'text-anchor="{"end" if side < 0 else "start"}">{esc(label)}</text>') if label else ""
        self.edges.append(f'<g id="{self.pfx}-e-{a}-{b}" class="{_h(hidden)}"><line class="cls-ed" x1="{x1}" y1="{y1}" '
                          f'x2="{x2}" y2="{y2}"/>{lab}</g>')
        return f"{self.pfx}-e-{a}-{b}"

    def html(self, hidden=False):
        return (f'<div class="cls-tree {_h(hidden)}" id="{self.pfx}" style="width:{self.w}px;height:{self.h}px">'
                f'<svg width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" direction="ltr">'
                f'{"".join(self.edges)}</svg>{"".join(self.nodes)}</div>')


def dots(cnt, kinds=("a", "b", "c")):
    """class composition: cnt = [count of class 0, count of class 1, ...] as coloured dots"""
    return '<span class="cls-dots">' + "".join(f'<i class="k-{kinds[c]}"></i>' * n for c, n in enumerate(cnt)) + "</span>"


# ---------------------------------------------------------------- small text helpers
def chip(id_, html_, kind="ok", hidden=True):
    return f'<div id="{id_}" class="cls-chip {kind} {_h(hidden)}">{html_}</div>'


def eq(id_, tex, size=40, hidden=True):
    return f'<div id="{id_}" class="cls-eq {_h(hidden)}" style="font-size:{size}px">{tex}</div>'


def stem(qid, html_):
    return f'<div class="qstem cls-stem"><span class="qid">{esc(qid)}</span>{html_}</div>'


def note(html_, id_=None, hidden=False):
    i = f' id="{id_}"' if id_ else ""
    return f'<div{i} class="cls-note {_h(hidden)}">{html_}</div>'


def col(*parts, gap=18):
    return f'<div class="cls-col" style="gap:{gap}px">{"".join(parts)}</div>'


def row(*parts, gap=26):
    return f'<div class="cls-row" style="gap:{gap}px">{"".join(parts)}</div>'


def opts(pfx, opts_, notes=None, size=27):
    """stage.options in the course's compact style (long exam options)"""
    import stage as st
    return f'<div class="cls-opts" style="--ofs:{size}px">{st.options(pfx, opts_, notes=notes)}</div>'
