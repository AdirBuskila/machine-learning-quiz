# -*- coding: utf-8 -*-
"""Validate the per-exam raw JSON files (tools/raw/<CODE>.json) the agents produced."""
import json, pathlib, collections

RAW = pathlib.Path(__file__).parent / "raw"
TOPICS = {"intro", "regression", "knn", "trees", "naive_bayes", "neural_nets",
          "ensemble", "svm", "clustering", "evaluation", "image", "text", "flow"}

total = usable = 0
by_topic = collections.Counter()
by_official = collections.Counter()
problems = []

files = [p for p in sorted(RAW.glob("*.json"))
         if p.name not in ("manifest.json", "figs_index.json") and not p.name.startswith("_")]
for p in files:
    code = p.stem
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        problems.append(f"{code}: JSON parse error: {e}")
        continue
    nimg = 0
    for q in data:
        total += 1
        n = q.get("num")
        opts = q.get("options", [])
        if not (2 <= len(opts) <= 6):
            problems.append(f"{code} Q{n}: {len(opts)} options")
        ci = q.get("correctIndex")
        if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
            problems.append(f"{code} Q{n}: bad correctIndex {ci}")
        if q.get("topic") not in TOPICS:
            problems.append(f"{code} Q{n}: bad topic {q.get('topic')}")
        if not str(q.get("question", "")).strip():
            problems.append(f"{code} Q{n}: empty question")
        if any(not str(o).strip() for o in opts):
            problems.append(f"{code} Q{n}: blank option")
        if q.get("hasImage"):
            nimg += 1
        else:
            usable += 1
            by_topic[q.get("topic")] += 1
            by_official["official" if q.get("official") else "derived"] += 1
    print(f"{code}: {len(data)} questions, {nimg} image-flagged, {len(data) - nimg} usable")

print(f"\nTOTAL: {total} raw, {usable} usable (non-image)")
print("by topic (usable):", dict(by_topic))
print("by source (usable):", dict(by_official))
print(f"\nPROBLEMS ({len(problems)}):")
for x in problems:
    print("  -", x)
