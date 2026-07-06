# Build plan — Machine-Learning MCQ quiz (mirror of data-science-quiz)

**Goal:** a Hebrew-RTL, mobile-first, light/dark MCQ exam trainer for the *Machine Learning
(למידת מכונה)* course, replicating the proven `data-science-quiz` app end-to-end: a fully
static site + a Python/Node build pipeline that turns past exams into a canonical
`questions.json`, plus a "learn" mode built from the course summaries.

Source materials: `C:\Users\Adir\Desktop\BSC\שנה ב\סמסטר ג\למידת מכונה`
Target repo: `C:\Users\Adir\Desktop\Coding\Dev\machine-learning-quiz` (sibling to the DS app).

## Reference app — what we replicate (reverse-engineered)

**Runtime (100% static, no build step, works from `file://` and GitHub Pages):**
`index.html` + `styles.css` + `app.js` + `questions.js` (`window.QUESTIONS`) + `learn.js`
(`window.LEARN`) + `images/`. Features: RTL Hebrew; manual light/dark theme (localStorage,
system default); *practice* mode (immediate feedback + explanation) and *whole-exam* mode
(timed, scored, per-topic breakdown, answer review); topic filter; "official-only" and
"mistakes-only" filters; **Fisher–Yates option shuffle at render time**; keyboard shortcuts
(1–8, Enter/←); localStorage progress; learn mode with chapter TOC + curated figures +
lightbox.

**Question schema** (`questions.json`, one object per question):
```
{ id, examCode, year, topic, topicLabel, question, options[], correctIndex,
  official, explanation, source, sourceLabel, hasImage, image?, code? }
```
- `correctIndex` indexes into `options`; the app **shuffles display order every render**, so
  the correct answer is stored *by content position*, never "always first."
- An option may be plain text or `"img:<path>"` (image-option).
- `image` (e.g. `images/exams/26B-A-Q7.png`) attaches a figure to a question.
- `official=true` ⇒ answer came from an authoritative key (solutions file **or** a form-0
  exam); `false` ⇒ derived from course material (shown with a "תשובה לא רשמית" warning).

**Pipeline (`tools/`):**
1. `extract_exams.py` — dump raw text from every exam/solutions PDF with **both** pdfplumber
   and PyMuPDF (fitz); Hebrew RTL mangles differently per engine, so we keep both and pick
   the cleaner. DOCX via python-docx. Emits `tools/raw/*.txt` + `manifest.json` (pages, image
   counts per page, char counts, previews).
2. Per-exam structuring → `tools/raw/<CODE>.json` (MCQs reconstructed from the raw text).
3. `build_questions.py` — validate, attach cropped figures, dedup, **merge form-0 ↔
   non-form-0 siblings**, emit `questions.js` / `questions.json` + `build_report.md`.
4. `build_learn.py` — course-summary markdown → `learn.js` + downscaled `images/`.
5. `validate.py` + `smoke.js` + `verify_*.js` — data & UI sanity checks.

## ML-specific rules

### Form 0 vs "לא טופס 0" (correct-answer resolution) — the critical axis
- **Explicitly form-0** (filename/content says `טופס 0` / `form_0`, e.g.
  `ML 2026b moedA. form0.pdf`, `למידת מכונה קיץ 2025 טופס 0.pdf`,
  `ML-2024c_moed_a_form_0.pdf`): correct answer = the **first printed option** →
  `correctIndex = 0` in the extracted order, `official = true`.
- **Has a solutions/פתרונות sibling:** resolve `correctIndex` from the solutions key,
  `official = true`. Cross-check any assumed answer against it and flag mismatches.
- **Neither** (non-form-0, no key, no form-0 sibling): **do NOT guess.** Exclude from the
  shipped bank; add to `ASK_ADIR.md`.
- **2024 sitting pairs** (normal + "לא טופס 0" = same questions, shuffled): prefer the
  key-bearing version as the answer source; treat the sibling as a duplicate → **merge**,
  don't double-count.
- Answer is always stored **by content**; the app shuffles order at render, so "always tap A"
  is defeated even if the source data is form-0-ordered.

### Figures (Phase 2) — unattended handling
The user is asleep, so interactive confirmation is deferred. Figure-dependent questions
(confusion matrix, decision tree, scatter/regression plot, K-means diagram, formula image)
are **not** injected into the live bank. Instead each is cropped (best-effort) to
`images/exams/` **staging** and listed in `ASK_ADIR.md` with its source file + Q number for a
fast morning review. Never fabricate a figure or an answer.

### De-duplication
Same questions recur across years and across form-0 ↔ non-form-0 siblings. Dedup by
normalized (question + option-set); keep one canonical copy (official > derived, then form-0
> keyed > derived, then earlier year).

### Topic taxonomy (keys → Hebrew labels)
```
intro        מבוא ומושגי יסוד
regression    רגרסיה לינארית ו-Gradient Descent
knn           KNN
trees         עצי החלטה
naive_bayes   נאיב בייס
neural_nets   רשתות נוירונים ו-Backpropagation
ensemble      למידת אנסמבל (Ensemble)
svm           SVM
clustering    אשכול (K-Means / היררכי)
evaluation    הערכת מודל (Confusion Matrix, K-Fold, נרמול)
image         עיבוד תמונה
text          ניתוח טקסט (NLP)
flow          MLflow / Flow
```

## Learn mode (Phase 5)
Chapter content built from the richest summaries (`הרצאות + סיכומים/סיכומים/2025`, `2026`, and
the per-topic PDFs under `עד 2025/`), plus the two formula sheets (`דף נוסחאות.pdf`,
`דף נוסחאות מורחב לזכאים.pdf`). One section per topic, key formulas surfaced, curated figures
with captions + lightbox — mirroring the DS learn section.

## Execution order (autonomous)
0. ✅ Study reference app; write this plan.
1. Extract raw text for the core exam set + solutions + practice sets (+ ZIP uniques);
   write `docs/exam_inventory.md`.
2. Structure each exam → `tools/raw/<CODE>.json` (parallel agents), resolving answers per the
   form-0/solutions rules; flag figure-dependent & low-confidence to `ASK_ADIR.md`.
3. Adversarially verify resolved answers against the solutions files.
4. `build_questions.py` → `questions.json/js` + report (dedup + sibling-merge).
5. Build the app (`index.html`, `app.js`, `learn.js`, `styles.css`) adapted from DS.
6. Learn mode → `learn.js` + images.
7. `validate.py` + `smoke.js` + `verify_*.js`; report counts per topic/exam, form-0 vs
   non-form-0 merges, and the full `ASK_ADIR.md` list.

## Non-negotiables
- **Never fabricate** a question, option, answer, or figure.
- Ship only answer-verified questions; everything uncertain → `ASK_ADIR.md`.
- Be honest about extraction confidence in the inventory and the final report.
