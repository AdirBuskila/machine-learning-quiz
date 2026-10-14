# Guided courses — "צריך עזרה?"

Four ground-up courses on the home screen, modelled on the databases app's courses. Each is a level map →
short lesson → exam-style quiz (pass at 80%) → summary, with a hand-off to the real exam questions of its
topics at the end. They group the 13 bank topics so there are only four:

| course id | title | bank topics | level ids |
|---|---|---|---|
| `cls` 🌳 | צריך עזרה ב-KNN, עצים ובייס? | knn, trees, naive_bayes | C01… |
| `eval` 📊 | צריך עזרה בהערכת מודל? | intro, evaluation | E01… |
| `reg` 🧠 | צריך עזרה ברגרסיה ורשתות נוירונים? | regression, neural_nets, svm | R01… |
| `unsup` 🧩 | צריך עזרה באשכול, טקסט ותמונה? | clustering, text, image, ensemble, flow | U01… |

Files: `courses.js` (registry), `course-ui.js` (screens), `courses/<course>/<ID>.js` (one level each).
`app.js` runs the quiz with `S.mode = "course"`; course answers never touch the bank stats
(`mlq_progress_v1`), each course has its own key `mlq_crs_<id>_v1`.
Deep links: `index.html?course=cls`, `?course=cls&level=C03`, `…&play=1`.

```bash
node tools/course_check.js --sync     # lint all courses + rewrite the level <script> block in index.html
node tools/course_check.js --dump cls C03   # questions with the keyed answer, for review
node tools/course_flow.js all light mobile  # every level end-to-end in headless Chrome (screens → tools/raw/courseflow)
```

## Content rules

- **Grounded in the past exams, nothing invented.** Every lesson teaches what the exam questions ask,
  the way they ask it, and cites them by id (`23S-A-Q3`) — ids become buttons that open the real
  question. Sources: `docs/briefs/*.md` (per-topic exam analyses) and the bank (`questions.json`).
  No concept that no exam question touches.
- **From the ground up.** Assume the student knows nothing: define each term before using it, one idea
  at a time, a worked example before the questions. Levels go from easy to exam-sized.
- **Every computed answer is computed.** Numbers in lessons, options and explanations are checked by
  running the arithmetic (node), never estimated.
- Each level mixes original practice questions (`{id, question, options, answer, explanation, recap}`)
  with real exam questions pulled from the bank verbatim (`{ref: "24S-C-Q9", recap}`) — at least one
  per level. The last level of a course is `boss: true`, a mixed exam-sized final.

## Level format

```js
/* <course> course · level N — <topic> */
ML_LEVEL("cls", {
  id: "C02", order: 2, icon: "🎯", title: "…", goal: "one sentence: what you can do after this level",
  boss: false,
  lesson: { html: `…` },
  questions: [
    { id: "C02-Q1", question: "…", options: ["…", "…", "…", "…"], answer: 0,
      recap: "one-line rule, shown in the summary when this was answered wrong",
      explanation: "why the answer is right AND why each tempting distractor is wrong" },
    { ref: "22B-A-Q6", recap: "…", explanation: "optional — overrides the bank explanation" },
  ],
});
```

Optional on an original question: `code` (LTR monospace block), `image` (path under `images/`),
`table: {head: [...], rows: [[...]], caption?}` (a data table shown above the options).

Text in `question` / `options` / `explanation` / `recap` / table cells: plain text with `**bold**`,
`` `code` `` and `\n`. Lesson `html`: `<p> <h3> <ul>/<li> <table> <b> <strong> <code>` plus the boxes
`<div class="lesson-rule">🔑 …</div>` (key rule), `<div class="lesson-trap">⚠️ …</div>` (exam trap),
`<div class="lesson-ex">…</div>` (worked example), `<div class="lesson-formula">…</div>` (a centred LTR formula).
No LaTeX — write math in Unicode: `√ Σ ² ³ · × ≤ ≥ ≈ ∞ ‖x‖ log₂ ⁄ − x̄ ŷ θ α η σ`.
The page is RTL: a non-Hebrew run inside Hebrew text (`(2, 5)`, `|7−2| = 5`) is isolated as LTR
automatically, but two separate formulas joined only by a comma become one run — separate them with a
Hebrew word (`ו-`, `ואילו`), not just ", ".
Options are shuffled: never write "תשובות א ו-ג", and avoid "כל התשובות נכונות" unless the exam itself does that.
Never put an exam id in a question stem (the peek button would reveal the answer); cite in explanations.
