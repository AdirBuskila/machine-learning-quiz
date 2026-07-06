# -*- coding: utf-8 -*-
"""Re-dump each PDF preserving BOLD / colored runs inline, so the correct option
in a solutions file (marked bold, and sometimes colored) is visible to the
structuring step. Plain get_text() drops formatting; get_text("dict") keeps span
flags/font/color.

Marking:  **bold text**   and   ⟪colored text⟫   (both if bold+colored).
Output:   tools/raw/txt/<stem>.marked.txt   (one per PDF in the manifest)

Run:  PYTHONUTF8=1 python tools/mark_answers.py
"""
import json, pathlib, fitz

HERE = pathlib.Path(__file__).parent
RAW = HERE / "raw"
TXT = RAW / "txt"
SRC = pathlib.Path(r"C:\Users\Adir\Desktop\BSC\שנה ב\סמסטר ג\למידת מכונה")
ZIPX = RAW / "zip_extracted"


def is_bold(sp):
    f = sp["font"].lower()
    return bool(sp["flags"] & 16) or any(k in f for k in ("bold", "black", "heavy", "semibold"))


def is_colored(sp):
    return sp.get("color", 0) not in (0,)  # 0 == black


def marked_text(path):
    out = []
    doc = fitz.open(str(path))
    for i, page in enumerate(doc, 1):
        out.append(f"\n----PAGE {i}----\n")
        dd = page.get_text("dict")
        for b in dd["blocks"]:
            for ln in b.get("lines", []):
                buf = []
                for sp in ln["spans"]:
                    t = sp["text"]
                    if not t:
                        continue
                    b_, c_ = is_bold(sp), is_colored(sp)
                    if b_ and c_:
                        buf.append(f"**⟪{t}⟫**")
                    elif b_:
                        buf.append(f"**{t}**")
                    elif c_:
                        buf.append(f"⟪{t}⟫")
                    else:
                        buf.append(t)
                line = "".join(buf).rstrip()
                if line.strip():
                    out.append(line)
        out.append("")
    doc.close()
    return "\n".join(out)


def resolve_path(rec):
    if rec["origin"] == "zip":
        # rel looks like "ZIP/<orig name>"; our extractor wrote safe names into ZIPX
        import re
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", rec["rel"][len("ZIP/"):])
        return ZIPX / safe
    return SRC / rec["rel"]


def main():
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    done = 0
    for rec in manifest:
        if rec["fmt"] != "pdf":
            continue
        p = resolve_path(rec)
        if not p.exists():
            print(f"[{rec['stem']}] MISSING {p}")
            continue
        try:
            mt = marked_text(p)
            (TXT / f"{rec['stem']}.marked.txt").write_text(mt, encoding="utf-8")
            done += 1
            nb = mt.count("**") // 2
            print(f"[{rec['stem']}] marked chars={len(mt)} bold_runs~{nb}")
        except Exception as e:
            print(f"[{rec['stem']}] ERROR {e.__class__.__name__}: {e}")
    print(f"\nDONE. {done} marked files.")


if __name__ == "__main__":
    main()
