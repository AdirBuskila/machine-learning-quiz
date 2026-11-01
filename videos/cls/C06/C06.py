# -*- coding: utf-8 -*-
"""C06 · עצים: עומק, גיזום ומה ID3 לא מניח — video script (build: tools/video/vid.py C06).

Source: courses/cls/C06.js (lesson + goal: spot overfitting in a graph or a table, the purpose of pruning, when a
leaf is made, why scaling and independence do not matter to trees).
Exam anchor: 26B-A-Q4 (train vs validation f1 against max_features), which no guided quiz uses.
Not used on purpose: the C06 quiz tables (depths 1..8 with the minimum at 4; the flat 0.40 table), the merge /
split-condition / Dana-Ron-Gil / ID3-false / disadvantage items, and its refs 23S-A-Q6, 26B-A-Q2, 24S-C-Q2; their
twins 24S-C-Q20, 24B-A-Q20, 24B-C-Q16, 22B-A-Q13, 22B-C-Q9, 22B-C-Q13, 22B-C-Q18, SAMP-1-Q6, SAMP-2-Q5 / SAMP-3-Q5,
CLAUDE-26-Q8; 23B-B-Q12 (the C06-Q2 scenario). No card repeats a quiz option's wording; the scaling and duplicate-
feature facts are shown on a threshold and on a copied column, not stated as rules.
Worked examples (new): an error table for depths 1..9 (validation minimum at depth 5, train minimum at depth 9); a
post-pruning check on 10 validation rows (split: 6 right, leaf: 7 right -> cut); a threshold split that keeps the
same partition after min-max, z-score and centring; a copied column that gets exactly the same information gain.
"""
import math

import numpy as np

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- depth vs error (data; the reading is verified)
DEPTH = list(range(1, 10))
TRAIN = [0.34, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.01, 0.00]
VALID = [0.36, 0.28, 0.22, 0.19, 0.18, 0.20, 0.23, 0.26, 0.29]
best_v = DEPTH[VALID.index(min(VALID))]
best_t = DEPTH[TRAIN.index(min(TRAIN))]
assert best_v == 5 and best_t == 9 and min(VALID) == 0.18
assert VALID[8] - VALID[4] > 0.1                                          # depth 9: far worse than depth 5
assert all(a > b for a, b in zip(TRAIN, TRAIN[1:]))                        # train keeps falling
assert all(VALID[i] < VALID[i - 1] for i in range(1, 5)) and all(VALID[i] > VALID[i - 1] for i in range(5, 9))
GAP = [round(v - t, 2) for t, v in zip(TRAIN, VALID)]
assert GAP[:5] == [0.02, 0.03, 0.04, 0.07, 0.10] and GAP[-1] == 0.29     # the gap opens

# ---------------------------------------------------------------- post-pruning check (verified)
# 10 validation rows reach the node: (true class, branch the split sends it to); the split's leaves say A | B
VAL = [("A", "L"), ("A", "L"), ("A", "L"), ("A", "R"), ("A", "R"), ("A", "L"), ("B", "R"), ("B", "L"), ("A", "R"), ("B", "R")]
SPLIT_PRED = {"L": "A", "R": "B"}
right_split = sum(1 for c, b in VAL if SPLIT_PRED[b] == c)
right_leaf = sum(1 for c, b in VAL if c == "A")                            # the node as one leaf: its majority, A
assert (right_split, right_leaf) == (6, 7) and right_leaf >= right_split   # the branch does not help -> prune

# ---------------------------------------------------------------- scaling keeps the partition (verified)
XS = np.array([2.0, 5.0, 9.0, 14.0])
LAB = ["A", "A", "B", "B"]
THR = 7.0
part = lambda x, t: tuple(x <= t)
scalings = {
    "minmax": lambda x: (x - XS.min()) / (XS.max() - XS.min()),
    "z": lambda x: (x - XS.mean()) / XS.std(ddof=1),
    "centre": lambda x: x - XS.mean(),
}
for f in scalings.values():                    # every one is increasing: the same threshold, moved, splits the same way
    assert part(f(XS), f(np.array(THR))) == part(XS, THR) == (True, True, False, False)
MM = scalings["minmax"](XS)
assert [w.fmt(v, 2) for v in MM] == ["0", "0.25", "0.58", "1"] and w.fmt(scalings["minmax"](np.array(THR)), 2) == "0.42"

# ---------------------------------------------------------------- a copied column (verified)
ROWS = [(1, 1), (1, 1), (1, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 1)]    # (feature, class)
g = [w.counts(ROWS, 0, 1, v, (1, 0)) for v in (0, 1)]
COPY = [(r[0], r[0], r[1]) for r in ROWS]
g2 = [w.counts(COPY, 1, 2, v, (1, 0)) for v in (0, 1)]
assert g == g2 and w.cond_H(g) == w.cond_H(g2)                             # the copy is never better: same IG
# after splitting on the feature, the copy is constant inside each branch: splitting on it changes nothing
for v in (0, 1):
    sub = [r for r in COPY if r[0] == v]
    assert len({r[1] for r in sub}) == 1

# ---------------------------------------------------------------- exam 26B-A-Q4
Q = BANK["26B-A-Q4"]
assert Q["correctIndex"] == 0 and "max_features" in Q["question"] and "העליון מתייחס ל-train" in Q["question"]
OPTS = list(enumerate(Q["options"]))
# read off the figure (694x509): x = 1 at px 90, x = 20 at px 652; f1 1.00 at px 37, 0.75 at px 472
PX = lambda x: 90 + (x - 1) * (652 - 90) / 19
PY = lambda f: 37 + (1.0 - f) * (472 - 37) / 0.25
FIG = st.raster("fig", Q["image"], 600, (694, 509),
                rings=[("t10", PX(10), PY(0.968), 20), ("v10", PX(10), PY(0.922), 20), ("t20", PX(20), PY(1.0), 20)])

# ---------------------------------------------------------------- figures
E = w.Plot("er", (0.5, 9.5), (0, 0.4), w=760, h=500, xticks=DEPTH, yticks=[0, 0.1, 0.2, 0.3, 0.4], xname="עומק",
           yname="שגיאה")
E.path("tr", list(zip(DEPTH, TRAIN)), kind="a", dots=True, hidden=False)
E.path("va", list(zip(DEPTH, VALID)), kind="b", dots=True, hidden=False)
E.text("ltr", 7.2, 0.03, "אימון", kind="a", hidden=False, anchor="middle", dy=-22)
E.text("lva", 9.2, 0.29, "תיקוף", kind="b", hidden=False, anchor="end", dy=-16)
E.vline("v5", 5, kind="ok")
E.ring("r5", 5, 0.18, r=18, kind="ok")
E.vline("v9", 9, kind="bad")
EPLOT = E.svg()

SC = w.table("sc", ["", "x", "", "", "", "", "סף"],
             [["מקורי", "2", "5", "9", "14", "", "7"], ["min-max", "0", "0.25", "0.58", "1", "", "0.42"]],
             size=32, hide_cols=())

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד לזהות מתי עץ החלטה מתחיל לשנן, ומה עושים עם זה."),
    P("נקרא גרף של אימון מול תיקוף, נראה מה עושה גיזום, ונבדוק מה עץ בכלל לא צריך."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">C06 · מסווגים</div><h1>עצים: עומק, גיזום ומה ID3 לא מניח</h1>'
         '<p>עומק מול overfitting, בחירה לפי קו התיקוף, גיזום מראש ובדיעבד, וסילום ותלות בין מאפיינים, שלא משנים לעץ כלום.</p>',
   layout="title")

depth = Scene("depth", "עומק ושגיאה", [
    P("אימנו עצים בעומקים אחת עד תשע. הקו הכחול הוא השגיאה על סט האימון. הצהוב, על סט התיקוף.",
      say="אימנו עצים בעומקים אחת עד תשע. הקו הכחול הוא השגיאה על סט האימון. הצהוב, על סט התיקוף."),
    P("סט תיקוף הוא דוגמאות שהעץ לא ראה באימון. עליהן בודקים אם הוא באמת למד.", do=show("d-0")),
    P("ככל שהעץ עמוק יותר, שגיאת האימון רק יורדת, עד אפס בעומק תשע. עץ עמוק יכול לשנן כל שורה.", do=show("d-1")),
    P("אבל שגיאת התיקוף יורדת רק עד עומק חמש, ומשם עולה.", do=show("er-v5", "er-r5", "d-2")),
    P("מחמש והלאה הקווים נפרדים: האימון משתפר, והתיקוף מתקלקל. זה **overfitting**, העץ לומד גם את הרעש.",
      do=show("d-3"),
      say="מחמש והלאה הקווים נפרדים: האימון משתפר, והתיקוף מתקלקל. זה אוֹבֶרְפִיטִינְג, העץ לומד גם את הרעש."),
    P("ובעומק אחת: אימון 0.34 ותיקוף 0.36. כמעט אין פער, אבל שתי השגיאות גבוהות. העץ פשוט מדי: **underfitting**.",
      say="ובעומק אחת: אימון אפס נקודה שלוש ארבע, ותיקוף אפס נקודה שלוש שש. כמעט אין פער, אבל שתי השגיאות גבוהות. העץ פשוט מדי: אַנְדֶרְפִיטִינְג."),
    P("איזה עומק לבחור? את חמש, הנקודה הנמוכה על הקו הצהוב. עומק תשע הוא הנמוך על הכחול, ובתיקוף הוא גרוע בהרבה מעומק חמש.",
      do=show("er-v9", "d-t"), hold=0.4),
], stage=f'<div class="cls-row">{EPLOT}</div>',
   panel=st.card("d-0", "plain", "תיקוף = דוגמאות שהעץ <b>לא ראה</b>")
         + st.card("d-1", "ex", "אימון: יורד עד $0$ בעומק $9$")
         + st.card("d-2", "ex", "תיקוף: מינימום $0.18$ בעומק $5$")
         + st.card("d-3", "rule", "🔑 הקווים נפרדים ← overfitting")
         + st.card("d-t", "trap", "⚠️ בוחרים לפי הקו של התיקוף, לא של האימון"))

prune = Scene("prune", "גיזום", [
    P("איך עוצרים עץ מלשנן? יש שתי דרכים, ושתיהן נקראות גיזום."),
    P("גיזום מוקדם: עוצרים תוך כדי הבנייה. למשל עומק מרבי, או מינימום דוגמאות בצומת.", do=show("p-1")),
    P("גיזום בדיעבד: בונים את העץ המלא, ואז בודקים כל פיצול על סט התיקוף.", do=show("p-2")),
    P("לדוגמה: עשר דוגמאות תיקוף מגיעות לצומת מסוים. עם הפיצול שלו, שש מהן מסווגות נכון.", do=show("p-3")),
    P("ואם נחליף את הצומת בעלה אחד, עם מחלקת הרוב שלו? שבע נכונות.", do=show("p-4")),
    P("שש מול שבע: חותכים את הפיצול, והצומת הופך לעלה.", do=show("p-5"), hold=0.4),
], stage=w.col(st.card("p-1", "ex", "<b>מוקדם</b>: עוצרים בזמן הבנייה ($\\text{max\\_depth}$, מינימום דוגמאות)"),
               st.card("p-2", "ex", "<b>בדיעבד</b>: עץ מלא, ואז בדיקה של כל פיצול על התיקוף"),
               w.chip("p-3", "עם הפיצול: 6 מתוך 10 נכונות", "warn"),
               w.chip("p-4", "עלה אחד במקומו: 7 מתוך 10", "ok"),
               w.chip("p-5", "← חותכים את הפיצול", "bad"), gap=22),
   panel="")

need = Scene("need", "מה עץ לא צריך", [
    P("ועכשיו שני דברים שעץ לא צריך. הראשון: סילום."),
    P("עץ שואל על מאפיין אחד בכל פעם: האם איקס קטן או שווה לסף. כאן הסף שבע, ושני הערכים הקטנים הולכים שמאלה.",
      do=show("sc"),
      say="עץ שואל על מאפיין אחד בכל פעם: האם איקס קטן או שווה לסף. כאן הסף שבע, ושני הערכים הקטנים הולכים שמאלה."),
    P("אחרי מין-מקס, הערכים זזים, וגם הסף זז, ל-0.42. אבל אותם שני ערכים עדיין מתחתיו.", do=show("n-1"),
      say="אחרי מין-מקס, הערכים זזים, וגם הסף זז, לאפס נקודה ארבע שתיים. אבל אותם שני ערכים עדיין מתחתיו."),
    P("כל סילום ששומר על הסדר נותן אותה חלוקה בדיוק. ואם החלוקה זהה, כל חישובי האנטרופיה זהים.", do=show("n-2")),
    P("השני: תלות בין מאפיינים. נעתיק עמודה פעמיים. לעותק יש בדיוק אותו רווח מידע כמו למקור.", do=show("n-3")),
    P("אחרי שהעץ פיצל לפי המקור, העותק קבוע בתוך כל ענף, ואין לו מה לפצל. לכל היותר ייבחר במקום המקור, אף פעם לא בנוסף לו.",
      do=show("n-4"), hold=0.4),
], stage=w.col(SC, st.card("n-1", "ex", "סף $7$ ← $0.42$: שמאלה עדיין $2$ ו-$5$"),
               st.card("n-3", "ex", "עמודה מועתקת: אותו $IG$ בדיוק"), gap=26),
   panel=st.card("n-2", "rule", "🔑 עץ רואה רק את הסדר בתוך מאפיין")
         + st.card("n-4", "rule", "🔑 העותק קבוע בתוך כל ענף: אין לו מה לפצל"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: כיוונו את max_features של עץ. הגרף: ציון f1 על האימון בכחול, למעלה, ועל התיקוף בכתום.",
      say="שאלה ממבחן: כיוונו את מַקְס פִיצֶ'רְס של עץ. הגרף: ציון אֶף-וַאן על האימון בכחול, למעלה, ועל התיקוף בכתום."),
    P("שימו לב: כאן ציר ה-y הוא ציון, גבוה זה טוב. ובציר ה-x לא עומק, אבל זו בדיוק אותה קריאה.", do=show("x-1"),
      say="שימו לב: כאן ציר הוַואי הוא ציון, גבוה זה טוב. ובציר האִקְס לא עומק, אבל זו בדיוק אותה קריאה."),
    P("עד בערך עשר, שני הקווים עולים יחד. מעשר והלאה הכחול ממשיך לטפס לכיוון אחת, והכתום נתקע ואפילו יורד.",
      do=show("fig-ring-t10", "fig-ring-v10")),
    P("פער שנפתח, אימון משתפר ותיקוף לא: overfitting מעשר והלאה. זו התשובה.", do=show("q-ok-0", "x-2"),
      say="פער שנפתח, אימון משתפר ותיקוף לא: אוֹבֶרְפִיטִינְג מעשר והלאה. זו התשובה."),
    P("לקבוע עשרים כדי להגיע למאה אחוז? מאה אחוז הוא על האימון, הקו הכחול. בתיקוף עשרים דווקא לא טוב.",
      do=show("fig-ring-t20", "q-x-1", "q-n-1")),
    P("underfitting מעשר והלאה הפוך: שם המודל מורכב מדי, לא פשוט מדי.", do=show("q-x-2", "q-n-2"),
      say="אַנְדֶרְפִיטִינְג מעשר והלאה הפוך: שם המודל מורכב מדי, לא פשוט מדי."),
    P("ומכיוון שיש תשובה נכונה, גם ״שאר התשובות שגויות״ נופלת.", do=show("q-x-3", "q-n-3")),
    P("המלכודת: לבחור לפי הקו העליון, כי הוא נראה הכי טוב. תמיד מחפשים איפה התיקוף מפסיק להשתפר.", do=show("x-t"),
      hold=0.6),
], stage=w.col(FIG, w.chip("x-1", "ציון, לא שגיאה: למעלה = טוב", "info"),
               w.chip("x-2", "≥ 10: הכחול עולה, הכתום נתקע ← overfitting", "ok"), gap=18),
   panel=w.stem(Q["id"], "עצים עם ערכי max_features שונים. למעלה: f1 על train, למטה: על validation. מה המסקנה?")
         + w.opts("q", OPTS, notes={1: "זה האימון", 2: "הפוך", 3: "יש נכונה"}, size=25)
         + st.card("x-t", "trap", "⚠️ הקו העליון הוא האימון: לא בוחרים לפיו"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: בטבלה שלנו, מעומק חמש האימון המשיך להשתפר והתיקוף התקלקל. ככה מזהים שינון.", do=show("r-1")),
    P("את העומק, או כל פרמטר אחר, בוחרים לפי הנקודה הטובה ביותר על קו התיקוף.", do=show("r-2")),
    P("ובדוגמת הגיזום: שש נכונות עם הפיצול, שבע בלעדיו, ולכן הוא נחתך.", do=show("r-3")),
    P("ועץ רואה רק את הסדר בתוך כל מאפיין, ועותק של עמודה לא מוסיף לו פיצול. נתראה ברמה הבאה.", do=show("r-4"),
      hold=0.6),
], stage=w.col(w.eq("r-eq", r"$\text{depth}^* = \arg\min_d\ \text{err}_{\text{val}}(d)$", 56, hidden=False), gap=30),
   panel=st.card("r-1", "rule", "🔑 פער שנפתח ← overfitting")
         + st.card("r-2", "rule", "🔑 בוחרים לפי התיקוף")
         + st.card("r-3", "ex", "גיזום: $6$ מול $7$ ← הפיצול נחתך")
         + st.card("r-4", "trap", "⚠️ עץ משווה ערך של מאפיין אחד לסף, ותו לא"))

VIDEO = Video("C06", "cls", "עצים: עומק, גיזום ומה ID3 לא מניח", [title, depth, prune, need, exam, recap],
              bank={"26B-A-Q4": 0})
