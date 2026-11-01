# -*- coding: utf-8 -*-
"""Evaluation-course widgets (videos E01..E10). Every class/id helper is prefixed ev- so the other courses' widgets
(all widgets/*.css load into every video of the app) are untouched. A widget only draws what it is given: every
number is computed in the script (exact fractions here, cross-checked against scikit-learn) and asserted there.

  metrics(tp, fn, fp, tn)          exact Fractions: acc, err, prec, rec, f1, and the second class's prec_n / rec_n / f1_n,
                                   macro_p, macro_r, micro (= acc). sk_check() re-derives all of them with sklearn.
  cmatrix(pfx, tp, fn, fp, tn, …)  2x2 confusion matrix, rows = truth, columns = prediction (the course convention).
                                   {pfx}-h-<tp|fn|fp|tn> cell highlight; {pfx}-v-<cell> the value (hidden when vals=False)
  Plot(pfx, xr, yr, …)             2-D axes in data units: .pts .poly .line .arrow .text .band .svg; ids {pfx}-<id>
  folds(pfx, k, n)                 K-fold strips: round r = {pfx}-r{r}
  holdout(pfx, n, n_val)           a single split strip
  table / eq / chip / note / stem / bidi / opts      small helpers
  num_he / dec_he / pct_he         numbers as spoken Hebrew words (for say=)
  check_say(video)                 asserts that no phrase's spoken text contains a Latin letter
"""
import html as _html
import re
from fractions import Fraction as Fr

LATIN = re.compile(r"[A-Za-z]")


def esc(s):
    return _html.escape(str(s), quote=True)


def _h(hidden):
    return " h" if hidden else ""


# ---------------------------------------------------------------- spoken numbers
_UF = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
_UM = ["אפס", "אחד", "שניים", "שלושה", "ארבעה", "חמישה", "שישה", "שבעה", "שמונה", "תשעה"]
_TF = ["עשר", "אחת עשרה", "שתים עשרה", "שלוש עשרה", "ארבע עשרה", "חמש עשרה", "שש עשרה", "שבע עשרה", "שמונה עשרה",
       "תשע עשרה"]
_TM = ["עשרה", "אחד עשר", "שנים עשר", "שלושה עשר", "ארבעה עשר", "חמישה עשר", "שישה עשר", "שבעה עשר", "שמונה עשר",
       "תשעה עשר"]
_TENS = {2: "עשרים", 3: "שלושים", 4: "ארבעים", 5: "חמישים", 6: "שישים", 7: "שבעים", 8: "שמונים", 9: "תשעים"}
_HUN = {1: "מאה", 2: "מאתיים", 3: "שלוש מאות", 4: "ארבע מאות", 5: "חמש מאות", 6: "שש מאות", 7: "שבע מאות",
        8: "שמונה מאות", 9: "תשע מאות"}
_THO = {1: "אלף", 2: "אלפיים", 3: "שלושת אלפים", 4: "ארבעת אלפים", 5: "חמשת אלפים", 6: "ששת אלפים", 7: "שבעת אלפים",
        8: "שמונת אלפים", 9: "תשעת אלפים", 10: "עשרת אלפים"}


def _parts(n, g):
    """components of 0 < n < 11000, without the joining ו"""
    out = []
    t, n = divmod(n, 1000)
    if t:
        out.append(_THO[t])
    h, n = divmod(n, 100)
    if h:
        out.append(_HUN[h])
    if 10 <= n < 20:
        out.append((_TF if g == "f" else _TM)[n - 10])
    else:
        d, u = divmod(n, 10)
        if d:
            out.append(_TENS[d])
        if u:
            out.append((_UF if g == "f" else _UM)[u])
    return out


def num_he(n, g="f"):
    """an integer 0..10999 in words. g='f' (counting, or a feminine noun) or 'm' (a masculine noun)"""
    n = int(n)
    assert 0 <= n < 11000, n
    if n == 0:
        return "אפס"
    p = _parts(n, g)
    if len(p) > 1:
        p[-1] = "ו" + p[-1]
    return " ".join(p)


def dec_he(s):
    """a decimal given as a string ('0.75', '76.6', '0.05'): the fraction part is read as a number unless it starts
    with 0, then digit by digit"""
    s = str(s)
    if "." not in s:
        return num_he(int(s))
    a, b = s.split(".")
    frac = " ".join(_UF[int(c)] for c in b) if b.startswith("0") else num_he(int(b))
    return f"{num_he(int(a))} נקודה {frac}"


def pct_he(s):
    return dec_he(s) + " אחוז"


def fr_pct(x, nd=2):
    """a Fraction as a percent rounded half-up to nd decimals, as a string without trailing zeros"""
    from decimal import Decimal, ROUND_HALF_UP
    v = (Decimal(x.numerator) / Decimal(x.denominator) * 100).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{v:f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fr_dec(x, nd=2):
    from decimal import Decimal, ROUND_HALF_UP
    v = (Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{v:f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


# ---------------------------------------------------------------- metrics
def metrics(tp, fn, fp, tn):
    n = tp + fn + fp + tn
    q = lambda a, b: Fr(a, b) if b else Fr(0)
    m = {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
         "acc": q(tp + tn, n), "err": q(fn + fp, n),
         "prec": q(tp, tp + fp), "rec": q(tp, tp + fn),
         "prec_n": q(tn, tn + fn), "rec_n": q(tn, tn + fp)}
    hm = lambda p, r: q(2 * p * r, 1) / (p + r) if p + r else Fr(0)
    m["f1"] = hm(m["prec"], m["rec"])
    m["f1_n"] = hm(m["prec_n"], m["rec_n"])
    m["macro_p"] = (m["prec"] + m["prec_n"]) / 2
    m["macro_r"] = (m["rec"] + m["rec_n"]) / 2
    m["micro"] = m["acc"]
    return m


def sk_check(M):
    """re-derive every metric of M with scikit-learn on label vectors that realise the matrix"""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    tp, fn, fp, tn = M["tp"], M["fn"], M["fp"], M["tn"]
    y = np.r_[np.ones(tp + fn), np.zeros(fp + tn)].astype(int)
    p = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)].astype(int)
    kw = {"zero_division": 0}
    got = {"acc": accuracy_score(y, p), "prec": precision_score(y, p, **kw), "rec": recall_score(y, p, **kw),
           "f1": f1_score(y, p, **kw), "prec_n": precision_score(y, p, pos_label=0, **kw),
           "rec_n": recall_score(y, p, pos_label=0, **kw), "f1_n": f1_score(y, p, pos_label=0, **kw),
           "macro_p": precision_score(y, p, average="macro", **kw), "macro_r": recall_score(y, p, average="macro", **kw),
           "micro": precision_score(y, p, average="micro", **kw)}
    assert abs(recall_score(y, p, average="micro") - got["acc"]) < 1e-12
    for k, v in got.items():
        assert abs(float(M[k]) - v) < 1e-12, (k, float(M[k]), v)
    return True


# ---------------------------------------------------------------- confusion matrix
def cmatrix(pfx, tp, fn, fp, tn, pos="חיובי", neg="שלילי", size=34, hidden=False, vals=True, tags=True):
    """rows = the truth, columns = the prediction. vals=False: the values start hidden ({pfx}-v-<cell> shows one)"""
    def cell(tag, v, cls):
        t = f'<span class="ev-cmt">{tag.upper()}</span>' if tags else ""
        return (f'<td class="ev-cm {cls}"><i id="{pfx}-h-{tag}" class="ev-cmh h"></i>{t}'
                f'<b id="{pfx}-v-{tag}" class="{"" if vals else "h"}">{v}</b></td>')
    return (f'<table class="ev-cmx{_h(hidden)}" id="{pfx}" style="font-size:{size}px"><thead><tr><th></th>'
            f'<th>חזוי {pos}</th><th>חזוי {neg}</th></tr></thead><tbody>'
            f'<tr><th>באמת {pos}</th>{cell("tp", tp, "ok")}{cell("fn", fn, "bad")}</tr>'
            f'<tr><th>באמת {neg}</th>{cell("fp", fp, "bad")}{cell("tn", tn, "ok")}</tr></tbody></table>')


# ---------------------------------------------------------------- K-fold / hold-out strips
def folds(pfx, k, n, w=860, hidden_rows=True, label=True):
    size, rem = divmod(n, k)
    assert rem == 0, "equal folds only"
    bw = w / k
    rows = []
    for r in range(k):
        cells = []
        for j in range(k):
            val = j == r
            txt = (("תיקוף" if val else "אימון") + f" · {size}") if label else ""
            cells.append(f'<div class="ev-fold {"val" if val else "trn"}" style="width:{bw - 8:.0f}px">{txt}</div>')
        rows.append(f'<div id="{pfx}-r{r}" class="ev-frow{_h(hidden_rows)}">'
                    f'<span class="ev-fr">סבב {r + 1}</span>{"".join(cells)}</div>')
    return f'<div class="ev-folds" id="{pfx}">{"".join(rows)}</div>'


def holdout(pfx, n, n_val, w=860, hidden=False):
    wt = w * (n - n_val) / n
    return (f'<div id="{pfx}" class="ev-frow{_h(hidden)}"><span class="ev-fr">פעם אחת</span>'
            f'<div class="ev-fold trn" style="width:{wt - 8:.0f}px">אימון · {n - n_val}</div>'
            f'<div class="ev-fold val" style="width:{w - wt - 8:.0f}px">תיקוף · {n_val}</div></div>')


# ---------------------------------------------------------------- 2-D plot
class Plot:
    def __init__(self, pfx, xr, yr, w=640, h=480, xticks=(), yticks=(), xname="", yname="", grid=True, pad=None,
                 xlabels=None):
        self.pfx, self.xr, self.yr, self.w, self.h = pfx, xr, yr, w, h
        self.pl, self.pr, self.pt, self.pb = pad or (70, 30, 30, 62)
        self.xticks, self.yticks, self.xname, self.yname, self.grid = xticks, yticks, xname, yname, grid
        self.xlabels = xlabels or {}
        self.layers = {"under": [], "mid": [], "pts": [], "top": []}

    def X(self, v):
        return self.pl + (v - self.xr[0]) * (self.w - self.pl - self.pr) / (self.xr[1] - self.xr[0])

    def Y(self, v):
        return self.pt + (self.yr[1] - v) * (self.h - self.pt - self.pb) / (self.yr[1] - self.yr[0])

    def _g(self, layer, id_, body, hidden):
        self.layers[layer].append(f'<g id="{self.pfx}-{id_}" class="{"h" if hidden else ""}">{body}</g>')
        return f"{self.pfx}-{id_}"

    def pts(self, id_, P, kind="ink", hidden=False, r=11, each=False, labels=None):
        out = []
        for i, (x, y) in enumerate(P):
            k = kind if isinstance(kind, str) else kind[i]
            m = f'<circle class="ev-dot k-{k}" cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}"/>'
            if labels and labels[i]:
                m += (f'<text class="ev-pl" x="{self.X(x):.1f}" y="{self.Y(y) - r - 10:.1f}" '
                      f'text-anchor="middle">{esc(labels[i])}</text>')
            out.append(f'<g id="{self.pfx}-{id_}-{i}" class="{"h" if each else ""}">{m}</g>' if each else m)
        return self._g("pts", id_, "".join(out), hidden)

    def poly(self, id_, P, kind="a", hidden=True, width=4.5, dash=False, layer="mid"):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(P))
        cls = "ev-poly" + (" dash" if dash else "")
        return self._g(layer, id_, f'<path class="{cls} k-{kind}" d="{d}" style="stroke-width:{width}"/>', hidden)

    def line(self, id_, x0, y0, x1, y1, kind="mut", hidden=True, width=3, dash=False):
        cls = "ev-poly" + (" dash" if dash else "")
        return self._g("under", id_, f'<line class="{cls} k-{kind}" x1="{self.X(x0):.1f}" y1="{self.Y(y0):.1f}" '
                                     f'x2="{self.X(x1):.1f}" y2="{self.Y(y1):.1f}" style="stroke-width:{width}"/>',
                       hidden)

    def arrow(self, id_, x0, y0, x1, y1, kind="b", hidden=True, width=6):
        mk = f"{self.pfx}-{id_}-m"
        return self._g("top", id_, f'<defs><marker id="{mk}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" '
                                   f'markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="ev-mk k-{kind}"/>'
                                   f'</marker></defs><line class="ev-poly k-{kind}" x1="{self.X(x0):.1f}" '
                                   f'y1="{self.Y(y0):.1f}" x2="{self.X(x1):.1f}" y2="{self.Y(y1):.1f}" '
                                   f'style="stroke-width:{width}" marker-end="url(#{mk})"/>', hidden)

    def band(self, id_, x0, x1, kind="a", hidden=True):
        return self._g("under", id_, f'<rect class="ev-band k-{kind}" x="{self.X(x0):.1f}" y="{self.pt}" '
                                     f'width="{self.X(x1) - self.X(x0):.1f}" height="{self.h - self.pt - self.pb}"/>',
                       hidden)

    def text(self, id_, x, y, s, kind="ink", hidden=True, anchor="middle", dx=0, dy=0, size=26):
        return self._g("top", id_, f'<text class="ev-tx k-{kind}" x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" '
                                   f'text-anchor="{anchor}" font-size="{size}">{esc(s)}</text>', hidden)

    def svg(self, hidden=False):
        o = []
        x0, x1 = self.X(self.xr[0]), self.X(self.xr[1])
        y0, y1 = self.Y(self.yr[0]), self.Y(self.yr[1])
        for t in self.yticks:
            if self.grid:
                o.append(f'<line class="ev-grid" x1="{x0:.1f}" x2="{x1:.1f}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            o.append(f'<text class="ev-tl" x="{x0 - 12:.1f}" y="{self.Y(t) + 9:.1f}" text-anchor="end">{_fmt(t)}</text>')
        for t in self.xticks:
            if self.grid:
                o.append(f'<line class="ev-grid" x1="{self.X(t):.1f}" x2="{self.X(t):.1f}" y1="{y1:.1f}" y2="{y0:.1f}"/>')
            o.append(f'<text class="ev-tl" x="{self.X(t):.1f}" y="{y0 + 38:.1f}">{esc(self.xlabels.get(t, _fmt(t)))}</text>')
        o.append(f'<line class="ev-ax" x1="{x0:.1f}" x2="{x1:.1f}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
        if self.yticks:
            o.append(f'<line class="ev-ax" x1="{x0:.1f}" x2="{x0:.1f}" y1="{y1:.1f}" y2="{y0:.1f}"/>')
        if self.xname:
            o.append(f'<text class="ev-an" x="{x1:.1f}" y="{y0 - 14:.1f}" text-anchor="end">{esc(self.xname)}</text>')
        if self.yname:
            o.append(f'<text class="ev-an" x="{x0 + 14:.1f}" y="{y1 + 4:.1f}">{esc(self.yname)}</text>')
        body = self.layers["under"] + self.layers["mid"] + self.layers["pts"] + self.layers["top"]
        return (f'<svg class="ev-svg{_h(hidden)}" id="{self.pfx}" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" direction="ltr">{"".join(o)}{"".join(body)}</svg>')


def _fmt(v):
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- text helpers
def bidi(s):
    """isolate Latin runs (metric names, numbers with %) inside Hebrew text so the RTL line keeps its order"""
    return re.sub(r"(?<![&#A-Za-z])([A-Za-z][A-Za-z0-9\-_ .²]*[A-Za-z0-9²]|[A-Za-z])(?![A-Za-z0-9]*;)",
                  r'<span class="ev-l">\1</span>', s)


def opts(Q):
    """the bank question's options, in source order, as (correctIndex-style int id, html)"""
    return [(i, bidi(_html.escape(o, quote=False))) for i, o in enumerate(Q["options"])]


def stem(qid, html_):
    return f'<div class="qstem ev-stem"><span class="qid">{esc(qid)}</span>{html_}</div>'


def eq(id_, tex, size=40, hidden=True):
    return f'<div id="{id_}" class="ev-eq{_h(hidden)}" style="font-size:{size}px">{tex}</div>'


def chip(id_, html_, kind="ok", hidden=True):
    return f'<div id="{id_}" class="ev-chip {kind}{_h(hidden)}">{html_}</div>'


def note(html_, id_=None, hidden=False):
    i = f' id="{id_}"' if id_ else ""
    return f'<div{i} class="ev-note{_h(hidden)}">{html_}</div>'


def table(pfx, head, rows, hide_rows=True, size=30, row_ids=None):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f'<tr id="{pfx}-r{(row_ids or range(len(rows)))[i]}" class="{"h" if hide_rows else ""}">'
                   + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for i, row in enumerate(rows))
    return (f'<table class="ev-tab" id="{pfx}" style="font-size:{size}px"><thead><tr>{th}</tr></thead>'
            f'<tbody>{body}</tbody></table>')


def col(*parts, gap=18):
    return f'<div class="ev-col" style="gap:{gap}px">{"".join(parts)}</div>'


def row(*parts, gap=26):
    return f'<div class="ev-row" style="gap:{gap}px">{"".join(parts)}</div>'


# ---------------------------------------------------------------- checks
def check_say(video):
    """the voice misreads Latin letters: every spoken phrase must be Hebrew only"""
    bad = []
    for S in video.scenes:
        for p in S.phrases:
            s = p.say
            if LATIN.search(s):
                bad.append(f"[{S.key}] {s}")
    assert not bad, "Latin in spoken text:\n  " + "\n  ".join(bad)
    return video
