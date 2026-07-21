# -*- coding: utf-8 -*-
"""Merge per-exam raw JSON (tools/raw/<CODE>.json) into the final ML question bank.

Outputs: ../questions.js  (window.QUESTIONS = [...])   loaded by the app
         ../questions.json (same data, reuse)
         build_report.md   (counts, exclusions, dedup)

Rules:
- Auto-discover tools/raw/<CODE>.json (skip manifest / helper files).
- Exclude questions flagged hasImage (unless a cropped figure exists), blank option,
  <2 options, bad correctIndex, empty question.
- De-duplicate AGGRESSIVELY by normalized (question + option-set). Keep ONE canonical
  copy, preferring real exams over samples over practice, then recent years, then form-0.
"""
import json, re, pathlib, collections

TOOLS = pathlib.Path(__file__).parent
RAW = TOOLS / "raw"
OUT = TOOLS.parent

TOPIC_LABEL = {
    "intro": "מבוא ומושגי יסוד",
    "regression": "רגרסיה ו-Gradient Descent",
    "knn": "KNN",
    "trees": "עצי החלטה",
    "naive_bayes": "נאיב בייס",
    "neural_nets": "רשתות נוירונים",
    "ensemble": "למידת אנסמבל",
    "svm": "SVM",
    "clustering": "אשכול (K-Means / היררכי)",
    "evaluation": "הערכת מודל",
    "image": "עיבוד תמונה",
    "text": "ניתוח טקסט (NLP)",
    "flow": "MLflow / Flow",
}
VALID_TOPICS = set(TOPIC_LABEL)

HE = {"A": "א׳", "B": "ב׳", "C": "ג׳"}
SEM = {"A": "סמסטר א׳", "B": "סמסטר ב׳", "S": "קיץ"}
PRAC_LABEL = {
    "PRAC-N": "תרגול — ניתוח טקסט ועיבוד תמונה",
    "PRAC-X": "תרגול — שאלות לדוגמה",
}
SAMP_LABEL = {
    "SAMP-1": "מבחן לדוגמה (1)",
    "SAMP-2": "מבחן לדוגמא עם פתרונות",
    "SAMP-3": "מבחן לדוגמה (3)",
}


def exam_label(code):
    if code in PRAC_LABEL:
        return PRAC_LABEL[code]
    if code in SAMP_LABEL:
        return SAMP_LABEL[code]
    m = re.match(r"^(\d{2})([ABS])-([ABC])$", code)
    if m:
        yy, s, md = m.groups()
        return f"20{yy} {SEM[s]} מועד {HE[md]}"
    return code


def source_of(code):
    if code.startswith("PRAC"):
        return "practice"
    return "exam"           # real exams + sample exams both count as 'exam'


def year_of(code):
    m = re.match(r"^(\d{2})", code)
    return int("20" + m.group(1)) if m else None


def priority(code):
    """Lower = more canonical (kept on dedup)."""
    if code.startswith("PRAC"):
        grp = 2
    elif code.startswith("SAMP"):
        grp = 1
    else:
        grp = 0
    yr = year_of(code) or 0
    return (grp, -yr, code)   # recent real exams win


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def find_image(code, num):
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        if (OUT / "images" / "exams" / f"{code}-Q{num}.{ext}").exists():
            return f"images/exams/{code}-Q{num}.{ext}"
    return None


def valid(q):
    opts = q.get("options", [])
    if len(opts) < 2:
        return False, "few-options"
    if any(not str(o).strip() for o in opts):
        return False, "blank-option"
    ci = q.get("correctIndex")
    if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
        return False, "bad-correctIndex"
    if not str(q.get("question", "")).strip():
        return False, "empty-question"
    if q.get("topic") not in VALID_TOPICS:
        return False, "bad-topic"
    return True, None


def raw_files():
    skip = {"manifest.json", "figs_index.json"}
    for p in sorted(RAW.glob("*.json")):
        if p.name in skip or p.name.startswith("_"):
            continue
        yield p


def main():
    raw_items = []
    per_file = collections.Counter()
    for p in raw_files():
        code = p.stem
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"SKIP {p.name}: parse error {e}")
            continue
        if not isinstance(data, list):
            continue
        for q in data:
            q["examCode"] = code
            q["source"] = source_of(code)
            q["sourceLabel"] = exam_label(code)
            q["year"] = year_of(code)
            q["id"] = f"{code}-Q{q.get('num')}"
            raw_items.append(q)
            per_file[code] += 1

    excl = collections.Counter()
    kept = []
    for q in raw_items:
        ok, why = valid(q)
        if not ok:
            excl[why] += 1
            continue
        if q.get("hasImage"):
            img = find_image(q["examCode"], q.get("num"))
            if not img:
                excl["image-missing"] += 1
                continue
            q["image"] = img
            q["hasImage"] = False
        kept.append(q)

    # aggressive content dedup: keep the most-canonical copy
    kept.sort(key=lambda q: priority(q["examCode"]))
    seen = {}
    final = []
    dup = 0
    for q in kept:
        key = norm(q["question"]) + " || " + "|".join(sorted(norm(o) for o in q["options"]))
        if key in seen:
            dup += 1
            continue
        seen[key] = q["id"]
        final.append(q)

    for q in final:
        q["topicLabel"] = TOPIC_LABEL.get(q["topic"], q["topic"])
        q.pop("num", None)
    final.sort(key=lambda q: (q["topic"], priority(q["examCode"]), q["id"]))

    payload = json.dumps(final, ensure_ascii=False, indent=1)
    (OUT / "questions.json").write_text(payload, encoding="utf-8")
    (OUT / "questions.js").write_text("window.QUESTIONS = " + payload + ";\n", encoding="utf-8")

    by_topic = collections.Counter(q["topic"] for q in final)
    by_exam = collections.Counter(q["examCode"] for q in final)
    by_origin = collections.Counter(q["source"] for q in final)
    lines = ["# Build report — ML question bank", "",
             f"- Raw items read: **{len(raw_items)}** from {len(per_file)} files",
             f"- Excluded: **{sum(excl.values())}**  ({dict(excl)})",
             f"- Duplicates merged (content): **{dup}**",
             f"- **Final questions: {len(final)}**", "",
             "## By topic", ""]
    for t, n in by_topic.most_common():
        lines.append(f"- {TOPIC_LABEL.get(t, t)} (`{t}`): {n}")
    lines += ["", "## By exam", ""]
    for c, n in sorted(by_exam.items(), key=lambda kv: priority(kv[0])):
        lines.append(f"- {exam_label(c)} (`{c}`): {n}")
    lines += ["", "## By origin",
              f"- exam: {by_origin['exam']} · practice: {by_origin['practice']}", ""]
    (TOOLS / "build_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"final={len(final)} excluded={sum(excl.values())} {dict(excl)} dup={dup}")
    print("by_topic:", dict(by_topic))
    print("by_exam:", dict(by_exam))


if __name__ == "__main__":
    main()
