# -*- coding: utf-8 -*-
"""build_learn.py — build the Learn-section data from the per-subject exam briefs.

Reads the Hebrew briefs in docs/briefs/ (one file per chapter, copied from the
Obsidian vault "Practice By Subject"), renders each to static HTML, and emits:

    learn.js  ->  window.LEARN = [{id, title, topic, sub, html}, ...]

Static + offline: LaTeX is rendered to native MathML at BUILD time via KaTeX
(tools/render_math.js), so the page ships no math library, no CSS and no fonts.

Run:  PYTHONUTF8=1 python tools/build_learn.py
"""
import json, os, re, subprocess, sys
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRIEFS = os.path.join(ROOT, "docs", "briefs")
OUT_JS = os.path.join(ROOT, "learn.js")
RENDERER = os.path.join(HERE, "render_math.js")

# Chapter order == the study order the source Index prescribes (gaps first, then
# exam weight), which is also the briefs' own file numbering. `topic` must match a
# key in app.js TOPICS so the chapter can deep-link into a drill of that topic.
MANIFEST = [
    ("01 - Decision Trees.md",              "trees",       "trees",       False),
    ("01a - ID3 - תיאוריה למבחן.md",        "trees_id3",   "trees",       True),
    ("02 - הערכת מודל.md",                  "evaluation",  "evaluation",  False),
    ("03 - KNN.md",                         "knn",         "knn",         False),
    ("04 - רשתות נוירונים.md",              "neural_nets", "neural_nets", False),
    ("05 - רגרסיה ו-Gradient Descent.md",   "regression",  "regression",  False),
    ("06 - מבוא ומושגי יסוד.md",            "intro",       "intro",       False),
    ("07 - נאיב בייס.md",                   "naive_bayes", "naive_bayes", False),
    ("08 - אשכול.md",                       "clustering",  "clustering",  False),
    ("09 - ניתוח טקסט.md",                  "text",        "text",        False),
    ("10 - עיבוד תמונה.md",                 "image",       "image",       False),
    ("11 - SVM.md",                         "svm",         "svm",         False),
    ("12 - Flow.md",                        "flow",        "flow",        False),
    ("13 - למידת אנסמבל.md",                "ensemble",    "ensemble",    False),
]

# Wikilink targets that live in the vault but are NOT shipped; they render as muted
# text instead of dangling links.
FILE_TO_ID = {fn[:-3]: cid for fn, cid, _, _ in MANIFEST}

QID = r"(?:\d{2}[ABS]-[ABC]|SAMP-[123]|PRAC-[NX])-Q\d+"

# Blockquote callout vocabulary used by the briefs, richest-signal first.
CALLOUTS = [("🪤", "trap"), ("🚨", "alarm"), ("⚠️", "warn"), ("⚠", "warn"),
            ("🔑", "key"), ("💡", "tip"), ("✅", "yes"), ("❌", "no"), ("🚩", "flag")]


# ---------------------------------------------------------------- math

def split_code_fences(text):
    """Yield (is_code, chunk) so $…$ inside ``` blocks is never treated as math."""
    parts, buf, in_code = [], [], False
    for ln in text.split("\n"):
        if ln.lstrip().startswith("```"):
            buf.append(ln)
            if in_code:
                parts.append((True, "\n".join(buf))); buf = []
            else:
                parts.append((False, "\n".join(buf[:-1]))); buf = [ln]
            in_code = not in_code
            continue
        buf.append(ln)
    parts.append((in_code, "\n".join(buf)))
    return [(c, t) for c, t in parts if t]


DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def extract_math(text, store):
    """Replace math with inert placeholders markdown will pass through untouched."""
    def take(tex, display):
        store.append({"tex": tex.strip(), "display": display})
        return f"@@MATH{len(store) - 1}@@"

    out = []
    for is_code, chunk in split_code_fences(text):
        if is_code:
            out.append(chunk); continue
        chunk = DISPLAY_RE.sub(lambda m: take(m.group(1), True), chunk)
        chunk = INLINE_RE.sub(lambda m: take(m.group(1), False), chunk)
        out.append(chunk)
    return "\n".join(out)


def render_math(items):
    if not items:
        return []
    proc = subprocess.run([node_bin(), RENDERER], input=json.dumps({"items": items}),
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        sys.exit(f"render_math.js failed:\n{proc.stderr}")
    res = json.loads(proc.stdout)
    for e in res["errors"]:
        print(f"  MATH FAIL [{e['i']}] {e['tex'][:60]} -> {e['err']}")
    return res["html"]


def node_bin():
    return "node.exe" if os.name == "nt" else "node"


# ---------------------------------------------------------------- markdown prep

def wikilinks(md_text, titles):
    """[[target]] / [[target|label]] -> in-app chapter link, or muted text.

    An unlabelled link shows the chapter's title ('KNN (K-Nearest Neighbors)')
    rather than the vault filename ('03 - KNN')."""
    def sub(m):
        body = m.group(1).replace("\\|", "|")
        target, _, label = body.partition("|")
        target, label = target.strip(), label.strip()
        cid = FILE_TO_ID.get(target)
        if cid:
            return f'<a class="xref" data-chapter="{cid}">{label or titles.get(cid, target)}</a>'
        return f'<span class="xref-dead">{label or target}</span>'
    return re.sub(r"\[\[([^\]]+)\]\]", sub, md_text)


_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")


def normalize_lists(md):
    """python-markdown needs a blank line before a list that follows a paragraph."""
    out = []
    for ln in md.split("\n"):
        if _LIST_RE.match(ln):
            prev = out[-1] if out else ""
            p = prev.strip()
            if p and not _LIST_RE.match(prev) and not p.startswith("#") and not p.startswith("|"):
                out.append("")
        out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------- html post-pass

def linkify_qids(html):
    """`23S-A-Q3` -> a button that peeks at that question. All refs are backticked."""
    return re.sub(r"<code>(" + QID + r")</code>",
                  r'<button type="button" class="qref" data-q="\1">\1</button>', html)


def wrap_tables(html):
    """Wide Hebrew tables scroll inside their own box instead of stretching the page.

    The wrapper carries the column count as --cols, so CSS can give a 6-column table
    a scrollable minimum width while a 2-column one still just fits."""
    def sub(m):
        head = m.group(0)
        cols = max([len(re.findall(r"<t[hd][\s>]", row))
                    for row in re.findall(r"<tr>(.*?)</tr>", head, re.S)] or [3])
        return f'<div class="tbl-wrap" style="--cols:{cols}"><table>{m.group(1)}</table></div>'
    return re.sub(r"<table>(.*?)</table>", sub, html, flags=re.S)


def tag_callouts(html):
    """Give each blockquote a class from its leading emoji so CSS can colour it."""
    def sub(m):
        head = m.group(1)[:60]
        for emoji, cls in CALLOUTS:
            if emoji in head:
                return f'<blockquote class="cal cal-{cls}">{m.group(1)}'
        return f'<blockquote class="cal">{m.group(1)}'
    return re.sub(r"<blockquote>(.{0,80})", lambda m: sub(m), html, flags=re.S)


def task_lists(html):
    """python-markdown has no GFM task lists; render the 176 checklist items."""
    html = re.sub(r"<li>\s*\[ \]\s*", '<li class="task"><i class="box"></i>', html)
    return re.sub(r"<li>\s*\[[xX]\]\s*", '<li class="task done"><i class="box"></i>', html)


DEAD_SEG = re.compile(
    r'\s*(<strong>[^<]*</strong>\s*:?\s*)?<span class="xref-dead">[^<]*</span>\s*')


def meta_paragraph(html):
    """Style the leading '39 שאלות · …' block.

    The briefs write it as 2-3 consecutive source lines, which Markdown folds into
    one paragraph — so restore the line breaks, then drop any '·'-segment that is
    only a pointer to a doc we don't ship ('Index: [[00 - Index]]')."""
    m = re.search(r"<p>(.*?)</p>", html, re.S)
    if not m:
        return html
    lines = []
    for line in m.group(1).split("\n"):
        # Classify each '·'-segment: a live one, or a label whose link we dropped.
        kept = []
        for s in line.split("·"):
            hit = DEAD_SEG.fullmatch(s)
            if not hit:
                kept.append(("live", s.strip()))
            elif hit.group(1):
                kept.append(("label", hit.group(1).strip()))
        # 'נלווה ל: [[00 - Index]] · [[03 - KNN]]' keeps the label, because live links
        # still follow it; 'Index: [[00 - Index]]' alone loses label and all.
        while kept and kept[-1][0] == "label":
            kept.pop()
        out, pending = [], None
        for kind, text in kept:
            if kind == "label":
                pending = text
            elif text:
                out.append(f"{pending} {text}" if pending else text)
                pending = None
        if out:
            lines.append(" · ".join(out))
    return html[:m.start()] + '<p class="brief-meta">' + "<br>".join(lines) + "</p>" + html[m.end():]


def inject_math(html, rendered):
    def sub(m):
        i = int(m.group(1))
        frag = rendered[i]
        if frag is None:
            return f'<code class="math-fail">{m.group(0)}</code>'
        cls = "math-block" if "display=\"block\"" in frag else "math-inline"
        return f'<span class="{cls}">{frag}</span>'
    return re.sub(r"@@MATH(\d+)@@", sub, html)


# ---------------------------------------------------------------- main

def read_brief(fname):
    """-> (title, body) with the '# 03 — KNN' heading stripped off the body."""
    path = os.path.join(BRIEFS, fname)
    if not os.path.exists(path):
        sys.exit(f"missing brief: {path}")
    raw = open(path, encoding="utf-8").read()
    m = re.search(r"^#\s+(.+)$", raw, re.M)
    title = re.sub(r"^\d+[a-z]?\s*[—–-]\s*", "", m.group(1).strip()).strip() if m else fname
    body = re.sub(r"^#\s+.+$", "", raw, count=1, flags=re.M).strip()
    # the rule directly under the title only separated it from the meta line
    body = re.sub(r"^\s*---\s*$", "", body, count=1, flags=re.M).strip()
    return title, body


def main():
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    store = []
    staged = []

    briefs = [(fname, cid, topic, sub, *read_brief(fname))
              for fname, cid, topic, sub in MANIFEST]
    titles = {cid: title for _, cid, _, _, title, _ in briefs}

    for fname, cid, topic, sub, title, body in briefs:
        prepared = normalize_lists(wikilinks(extract_math(body, store), titles))
        staged.append({"id": cid, "title": title, "topic": topic, "sub": sub,
                       "md": prepared})

    print(f"formulas queued: {len(store)}")
    rendered = render_math(store)
    ok = sum(1 for r in rendered if r)
    print(f"formulas rendered: {ok}/{len(store)}")

    learn = []
    for ch in staged:
        md.reset()
        html = md.convert(ch["md"])
        html = meta_paragraph(html)
        html = wrap_tables(html)
        html = tag_callouts(html)
        html = task_lists(html)
        html = linkify_qids(html)
        html = inject_math(html, rendered)
        learn.append({"id": ch["id"], "title": ch["title"], "topic": ch["topic"],
                      "sub": ch["sub"], "html": html})

    qrefs = sum(len(re.findall(r'class="qref"', c["html"])) for c in learn)
    meta = {"generated": "auto", "chapters": len(learn),
            "formulas": ok, "qrefs": qrefs}
    payload = ("/* AUTO-GENERATED by tools/build_learn.py — do not edit by hand. */\n"
               "window.LEARN = " + json.dumps(learn, ensure_ascii=False) + ";\n"
               "window.LEARN_META = " + json.dumps(meta, ensure_ascii=False) + ";\n")
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write(payload)

    print(f"chapters   : {len(learn)} -> {[c['id'] for c in learn]}")
    print(f"question refs linked: {qrefs}")
    print(f"wrote      : {OUT_JS}  ({os.path.getsize(OUT_JS) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
