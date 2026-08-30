# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Hebrew/RTL machine-learning exam trainer. The shipped site is **100% static** — `index.html`
+ `styles.css` + `app.js` + `questions.js` + `learn.js` + `images/`. No bundler, no framework,
no runtime deps; it runs from `file://` and from GitHub Pages (`.nojekyll` is committed).
Everything under `tools/` is an offline data pipeline that *produces* `questions.js` and
`learn.js` — it is not part of the deployed app.

## Commands

Run the site: open `index.html` in a browser. Data arrives as globals (`window.QUESTIONS`,
`window.LEARN`), so there is nothing to serve.

```bash
node tools/smoke.js       # data integrity + shuffle/scoring invariant + js≡json sync
node tools/verify_dom.js  # jsdom: boots index.html, starts a session, answers a question
```

These two are the real test suite and need no browser. `tools/verify_ui.js`,
`verify_theme.js`, and `verify_learn.js` drive headless Chrome via puppeteer-core with a
**hardcoded** `C:\Program Files\Google\Chrome\Application\chrome.exe`; they write screenshots
to `tools/raw/`. `verify_learn.js` is the mobile guard: it walks all 14 briefs at 390px and
fails if any chapter overflows horizontally.

Node deps live in `tools/`, but **`tools/package.json` and `package-lock.json` are gitignored**,
so a fresh clone has no manifest to install from — name them explicitly:

```bash
cd tools && npm install katex jsdom puppeteer-core
```

`katex` is required to rebuild `learn.js`; the other two only run the tests.

Rebuild the data (only needed after editing `tools/raw/*.json` or `docs/briefs/*.md`):

```bash
PYTHONUTF8=1 python tools/validate.py         # lint tools/raw/*.json before building
PYTHONUTF8=1 python tools/build_questions.py  # tools/raw/*.json -> questions.js + questions.json + build_report.md
PYTHONUTF8=1 python tools/build_learn.py      # docs/briefs/*.md -> learn.js
```

`build_learn.py` shells out to `tools/render_math.js` (KaTeX, from `tools/node_modules`),
so Node must be on PATH to rebuild the learn section — but only at build time; see below.

`extract_exams.py` / `mark_answers.py` / `crop_helper.py` are one-time source-mining scripts.
They read from an absolute path outside the repo
(`C:\Users\Adir\Desktop\BSC\שנה ב\סמסטר ג\למידת מכונה`) and need `pdfplumber`, `PyMuPDF`,
`python-docx`, `Pillow`. You almost never need to rerun them.

`PYTHONUTF8=1` matters: the Windows console is cp1255 and the scripts print ASCII only —
Hebrew goes to UTF-8 files.

## Data flow

```
exam PDFs/DOCX  --extract_exams.py-->  tools/raw/txt/*.txt
                --mark_answers.py -->  tools/raw/txt/*.marked.txt   (**bold** = the keyed answer)
        (structured by hand/agent)  ->  tools/raw/<EXAMCODE>.json   <- THE SOURCE OF TRUTH
                --build_questions.py->  questions.js + questions.json
docs/briefs/*.md      --build_learn.py->  learn.js   (via render_math.js for LaTeX)
```

**Never hand-edit `questions.js`, `questions.json`, or `learn.js`** — all three carry generated
content and are overwritten. Fix `tools/raw/<CODE>.json` (or `docs/briefs/*.md`) and rebuild.

`docs/briefs/` holds 14 Hebrew per-subject exam briefs copied from the Obsidian vault
(`…/busi-notes/Year-2-Sem-C/Machine-Learning/Practice By Subject`). Filenames are kept
verbatim so the vault can be re-copied over the folder; `MANIFEST` in `build_learn.py` maps
each filename to its chapter id, topic key, and position. Chapter order is the study order
the vault's Index prescribes (gaps first, then exam weight), which is the files' own
numbering. `trees` has two chapters — the overview and the ID3 theory companion, which is
flagged `sub` and renders indented in the TOC.

Exam codes: `YY[A|B|S]-[A|B|C]` (year, semester A/B/summer, מועד א/ב/ג), plus `SAMP-1..3`
(sample papers) and `PRAC-N` / `PRAC-X` (practice banks). `build_questions.py` derives
`id`, `year`, `source`, `sourceLabel`, `topicLabel`, `dedupKey`, `lockOrder`, and `image` from
the code and the raw fields; raw files only carry
`num, topic, question, options, correctIndex, official, explanation, hasImage`.

Topics are a closed set of 13 keys, declared in four places that must stay in sync:
`TOPIC_LABEL` in `build_questions.py`, `TOPICS` in `validate.py`, `TOPICS` in `app.js`
(the app list additionally has `all` first), and the `topic` column of `MANIFEST` in
`build_learn.py`. Each learn chapter carries its topic key so it can deep-link into a drill
of that topic; `verify_dom.js` fails if a chapter names a topic no question uses.

## Invariants that tests enforce (and why they exist)

- **`correctIndex` is stored by content, never by position.** `app.js` Fisher–Yates-shuffles
  the display order on every render, so "always tap א" is defeated even though form-0 papers
  put the correct option first in the source. `smoke.js` simulates 5000 shuffles and asserts
  the click→original-index mapping never drifts.
- **`lockOrder`** is set by `build_questions.py` when an option cites its siblings by printed
  letter ("תשובות א ו-ג נכונות"). Those questions render in source order — shuffling would make
  the cross-reference point at whatever landed in those slots.
- **`id` must be unique.** `app.js` keys the exam answer map by `q.id`; duplicate ids silently
  overwrote each other's answers and mis-scored whole exams (fixed in 2950f21). `smoke.js`
  asserts this.
- **`questions.js` ≡ `questions.json`.** Only the `.js` is loaded by the browser; the `.json`
  is for reuse. `smoke.js` fails if they drift, which is the tell that someone edited a
  generated file by hand.
- **Duplicates are kept, not merged** ("policy A"). Whole-exam mode must replay a past paper in
  full, so `build_questions.py` keeps every copy and stamps a `dedupKey`; `app.js`'s
  `dedupePool()` de-duplicates only the topic/random practice pools at runtime.
- **`official: true` means a real solutions file keyed the answer.** Form-0 papers and the
  `PRAC-*` banks are `false` and render a "תשובה לא רשמית" badge. It was previously true on
  every record, which made the "official only" filter a no-op — keep it meaningful.
- **Every `image` path and every `img:<path>` option must exist on disk** (`verify_dom.js`).
  Figures live in `images/exams/<CODE>-Q<num>.png`; a question flagged `hasImage` with no
  matching crop is dropped at build time and reported as `image-missing`.
- **Math is rendered to MathML at build time, never at runtime.** KaTeX runs in
  `render_math.js` with `output:"mathml"`, so the site ships no math library, no CSS and no
  web fonts and still works from `file://`. Every `<math>` gets `dir="ltr"` — the page is
  `dir="rtl"` and unmarked math renders scrambled. `verify_dom.js` asserts both.
- **Every question id cited in a brief must resolve.** The briefs reference all 363 ids in
  backticks; the builder turns `<code>23S-A-Q3</code>` into a `.qref` button that peeks at the
  real question, and `verify_dom.js` fails on a dangling one (it would open an empty panel).

## app.js conventions

Single-file vanilla JS, no modules. Screens are `<section class="screen">` toggled by `show()`;
`S` is session state, `P` is persisted progress (`localStorage` key `mlq_progress_v1`; theme is
`mlq_theme`). Practice mode gives immediate feedback and refills endlessly; exam mode is timed,
caches per-question shuffles in `S._views` so back-navigation is stable, and grades on submit.
`startTopicPractice(topic)` is the single entry point into a topic drill — used by
`index.html?practice=<topicKey>`, the per-chapter "תרגל נושא זה" button, and the peek panel.

The learn section (`initLearn`) renders one brief at a time into `#learnContent` and delegates
clicks on it: `.qref` opens the question peek, `.xref` jumps to another chapter, `.learn-drill`
starts a drill. `markScrollable()` re-flags overflowing tables/formulas after every render and
on resize, because a clipped formula with no edge shadow reads as broken rather than scrollable.

CSS is token-driven: dark is the `:root` default, `:root[data-theme="light"]` is the manual
toggle, and the same light palette is repeated under
`@media (prefers-color-scheme: light) { :root:not([data-theme]) }`. Change a color in **all
three** blocks. Class names and `--var` names are load-bearing — `app.js` and the verify scripts
query them. `index.html` cache-busts assets with `?v=N`; bump it when shipping a CSS/JS change.

Hebrew is the UI language; option text with no Hebrew characters is force-rendered `dir="ltr"`
so the RTL page doesn't reorder code/output answers.

## Stale artifacts to be aware of

- `docs/ASK_ADIR.md` says **322 questions**; the current bank is **363**
  (`tools/build_report.md` is regenerated and accurate).
- `docs/build_plan.md` describes the original design and states form-0 ⇒ `official=true`;
  the shipped rule is narrower (solutions file only). Prefer the invariants above.
- `docs/learn_source.md` is the **retired** source of the old concept-summary chapters. The
  learn section is now built from `docs/briefs/` instead; the file is kept only for
  reference and nothing reads it.
