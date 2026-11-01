# -*- coding: utf-8 -*-
"""C05 · ID3: בחירת הפיצול ובניית העץ — video script (build: tools/video/vid.py C05).

Source: courses/cls/C05.js (lesson + goal: pick the node's feature from a table by conditional entropy, know when a
branch becomes a leaf, read IG / mutual information).
Exam anchor: PRAC-X-Q6 (when is a leaf created? every listed case -> "all correct"), which no guided quiz uses.
Not used on purpose: the C05 quiz tables (weather / wind / holiday, the 6-row F/G with H(G|E) = 0.7, the X1/X2
probabilities, the MI and IG = 0 statements) and its refs 23B-B-Q2 / PRAC-X-Q38; 22B-A-Q15 (its X2 column with Y is
the F/G table of the quiz ref 23B-B-Q2 and the boss ref 24B-C-Q20, so solving it would solve them); SAMP-2-Q19 and
its twins (the C05-Q5 scenario). Mutual information is only named as another name of the number computed on screen;
no card states a decision rule from a quiz option.
Worked example (new, entropies cross-checked with scipy, the tree with scikit-learn): 10 rows (hot, weekend, diet ->
bought ice cream). H(Y|hot) = 0.485 < H(Y|diet) = 0.6 < H(Y|weekend) = 0.880 -> root "hot"; the "not hot" branch is
pure; on the 5 "hot" rows H(Y|diet) = 0 < H(Y|weekend) = 0.8 -> split by diet, two pure leaves.
"""
import math

import numpy as np
from scipy.stats import entropy as sp_entropy
from sklearn.tree import DecisionTreeClassifier

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- the table (verified)
# (hot, weekend, diet, bought): 1 = כן, 0 = לא
ROWS = [(1, 0, 0, 1), (0, 1, 0, 0), (1, 1, 1, 0), (0, 0, 1, 0), (1, 1, 0, 1),
        (0, 1, 1, 0), (1, 1, 1, 0), (0, 0, 0, 0), (1, 1, 0, 1), (0, 1, 0, 0)]
CL = (1, 0)                                           # class order on screen: כן, לא
FEAT = ["חם", "סופ״ש", "דיאטה"]
groups = lambda rows, j: [w.counts(rows, j, 3, v, CL) for v in (0, 1)]
HF = [w.cond_H(groups(ROWS, j)) for j in range(3)]
assert [f"{h:.3f}" for h in HF] == ["0.485", "0.880", "0.600"]
assert groups(ROWS, 0) == [[0, 5], [3, 2]] and groups(ROWS, 2) == [[3, 3], [0, 4]] and groups(ROWS, 1) == [[1, 2], [2, 5]]
assert f"{w.H([3, 2]):.3f}" == "0.971" and math.isclose(w.H([3, 2]), sp_entropy([3, 2], base=2))
assert f"{w.H([1, 2]):.3f}" == "0.918" and f"{w.H([2, 5]):.3f}" == "0.863"
assert min(range(3), key=lambda j: HF[j]) == 0
HOT = [r for r in ROWS if r[0] == 1]
assert len(HOT) == 5 and [w.counts(HOT, 3, 3, c, CL)[CL.index(c)] for c in CL] == [3, 2]
H2 = [w.cond_H(groups(HOT, j)) for j in (1, 2)]
assert f"{H2[0]:.1f}" == "0.8" and H2[1] == 0 and groups(HOT, 2) == [[3, 0], [0, 2]] and groups(HOT, 1) == [[1, 0], [2, 2]]
HY = w.H([3, 7])
assert f"{HY - HF[0]:.3f}" == "0.396"                 # IG of the root (= mutual information of hot and the class)
# scikit-learn's entropy tree grows the same tree: root "hot", then "diet", 3 leaves, all pure
X = np.array([r[:3] for r in ROWS])
y = np.array([r[3] for r in ROWS])
dt = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
assert dt.tree_.feature[0] == 0 and dt.get_n_leaves() == 3 and dt.score(X, y) == 1.0
assert 2 in set(dt.tree_.feature[dt.tree_.feature >= 0])

# ---------------------------------------------------------------- exam PRAC-X-Q6
Q = BANK["PRAC-X-Q6"]
assert Q["correctIndex"] == 2 and Q["options"][2] == "כל התשובות נכונות"
OPTS = [(i, Q["options"][i]) for i in (0, 1, 3, 2)]      # "all correct" last, as the app shows it
# the identical-vectors case: two rows with the same features and different classes cannot be split by any feature
DUP = [(1, 0, 1, 1), (1, 0, 1, 0)]
assert all(w.cond_H(groups(DUP, j)) == w.H([1, 1]) == 1 for j in range(3))

# ---------------------------------------------------------------- figures
yn = lambda v: "כן" if v else "לא"
TAB = w.table("tb", FEAT + ["קנה גלידה"], [[yn(v) for v in r] for r in ROWS], size=26, rtl=True)

TR = w.Tree("tr", w=860, h=600)
TR.node("r", 430, 60, f'<span class="nm">חם?</span>{w.dots([3, 7], ("a", "b"))}<span class="hv">H(Y|חם) = 0.485</span>',
        kind="q")
TR.node("n", 190, 290, f'<span class="nm">לא חם · 5</span>{w.dots([0, 5], ("a", "b"))}<span class="hv">עלה: לא</span>',
        kind="leafb")
TR.node("h", 640, 290, f'<span class="nm">חם · 5 ← דיאטה?</span>{w.dots([3, 2], ("a", "b"))}<span class="hv">H = 0.971</span>',
        kind="q")
TR.node("h0", 490, 520, f'<span class="nm">בלי דיאטה · 3</span>{w.dots([3, 0], ("a", "b"))}<span class="hv">עלה: כן</span>',
        kind="leaf")
TR.node("h1", 780, 520, f'<span class="nm">דיאטה · 2</span>{w.dots([0, 2], ("a", "b"))}<span class="hv">עלה: לא</span>',
        kind="leafb")
TR.edge("r", "n", "לא")
TR.edge("r", "h", "כן")
TR.edge("h", "h0", "לא")
TR.edge("h", "h1", "כן")
LEGEND = ('<div class="cls-note"><span class="cls-dots"><i class="k-a"></i></span> קנה &nbsp;·&nbsp; '
          '<span class="cls-dots"><i class="k-b"></i></span> לא קנה</div>')
TREE = TR.html() + LEGEND

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבנה עץ החלטה שלם עם ID3, מטבלה."),
    P("נבחר מאפיין לשורש, נחליט איזה ענף הופך לעלה, ונמשיך עד הסוף."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">C05 · מסווגים</div><h1>ID3: בחירת הפיצול ובניית העץ</h1>'
         '<p>בכל צומת: אם הוא טהור, עלה. אחרת, המאפיין עם האנטרופיה המותנית הנמוכה ביותר, ואותו תהליך בכל ענף.</p>',
   layout="title")

loop = Scene("loop", "הלולאה", [
    P("ID3 בונה את העץ מלמעלה למטה, ובכל צומת חוזר על אותו צעד.", do=show("l-0")),
    P("אם כל הדוגמאות בצומת מאותה מחלקה, הצומת הופך לעלה עם המחלקה הזו.", do=show("l-1")),
    P("אחרת, מחשבים לכל מאפיין את האנטרופיה המותנית, ובוחרים את **הנמוכה ביותר**.", do=show("l-2")),
    P("האנטרופיה לפני הפיצול זהה לכל המאפיינים. לכן אנטרופיה מותנית נמוכה ביותר, ורווח מידע גבוה ביותר, הם אותה בחירה.",
      do=show("l-3")),
    P("ואז מפצלים, ומפעילים את אותו צעד על כל ענף, רק עם השורות שהגיעו אליו.", do=show("l-4"), hold=0.4),
], stage=w.col(st.card("l-1", "ex", "1 · צומת טהור ← <b>עלה</b>"),
               st.card("l-2", "ex", "2 · אחרת: המאפיין עם $H(Y|F)$ <b>הנמוך</b>"),
               st.card("l-4", "ex", "3 · מפצלים, וחוזרים על כל ענף עם השורות שלו"), gap=24),
   panel=w.chip("l-0", "מלמעלה למטה, צומת אחרי צומת", "info")
         + st.card("l-3", "rule", "🔑 $IG = H(Y) - H(Y|F)$: הנמוך ב-$H(Y|F)$ = הגבוה ב-$IG$"))

root = Scene("root", "השורש", [
    P("הטבלה: עשרה ימים. שלושה מאפיינים: האם חם, האם סופ״ש, והאם אני בדיאטה. והמחלקה: האם קניתי גלידה."),
    P("מאפיין ״חם״: בחמשת הימים הלא חמים לא קניתי אף פעם. ענף טהור, אנטרופיה אפס."),
    P("בחמשת הימים החמים: שלוש קניות ושתיים בלי. אנטרופיה 0.971, ובמשקל חצי, יוצא 0.485.", do=show("h-1"),
      say="בחמשת הימים החמים: שלוש קניות ושתיים בלי. אנטרופיה 0.971, ובמשקל חצי, יוצא 0.485."),
    P("מאפיין ״דיאטה״: בארבעת ימי הדיאטה אף קנייה, טהור. בשישה האחרים שלוש מול שלוש, אנטרופיה אחת. במשקל: 0.6.",
      do=show("h-3"),
      say="מאפיין דיאטה: בארבעת ימי הדיאטה אף קנייה, טהור. בשישה האחרים שלוש מול שלוש, אנטרופיה אחת. במשקל: 0.6."),
    P("ומאפיין ״סופ״ש״: אין בו אף ענף טהור, ויוצא 0.88. הכי גרוע.", do=show("h-4"),
      say="ומאפיין סופ״ש: אין בו אף ענף טהור, ויוצא 0.88. הכי גרוע."),
    P("הנמוך הוא ״חם״, 0.485. הוא השורש.", do=show("h-r"),
      say="הנמוך הוא חם, 0.485. הוא השורש."),
    P("ושימו לב: ענף טהור אחד לא מספיק כדי לנצח. גם בדיאטה יש ענף טהור, אבל הענף השני שלה מעורבב לגמרי ושוקל יותר.",
      do=show("h-t"), hold=0.4),
], stage=w.col(TAB, gap=10),
   panel=st.card("h-1", "plain", r"חם: $\tfrac{5}{10}\cdot 0 + \tfrac{5}{10}\cdot 0.971 \approx 0.485$")
         + st.card("h-3", "plain", r"דיאטה: $\tfrac{4}{10}\cdot 0 + \tfrac{6}{10}\cdot 1 = 0.6$")
         + st.card("h-4", "plain", r"סופ״ש: $\approx 0.880$")
         + st.card("h-r", "rule", "🔑 השורש: ״חם״, האנטרופיה המותנית הנמוכה")
         + st.card("h-t", "trap", "⚠️ ענף טהור אחד לא מכריע: מה שקובע הוא הממוצע במשקל"))

grow = Scene("grow", "ממשיכים לגדל", [
    P("נבנה את העץ. בשורש: חם?", do=show("tr-n-r")),
    P("ענף ״לא חם״ טהור, כל החמישה בלי קנייה. הוא הופך לעלה ״לא״.", do=show("tr-e-r-n", "tr-n-n")),
    P("ענף ״חם״ מעורבב: שלוש מול שתיים. אז מפעילים עליו את אותו צעד, רק על חמש השורות שלו.", do=show("tr-e-r-h", "tr-n-h")),
    P("על חמש השורות האלה: דיאטה נותנת אנטרופיה מותנית אפס, וסופ״ש נותן 0.8. בוחרים דיאטה.", do=show("g-1"),
      say="על חמש השורות האלה: דיאטה נותנת אנטרופיה מותנית אפס, וסופ״ש נותן 0.8. בוחרים דיאטה."),
    P("בלי דיאטה: שלוש קניות, עלה ״כן״. עם דיאטה: שתיים בלי, עלה ״לא״. כל העלים טהורים, והעץ גמור.",
      do=show("tr-e-h-h0", "tr-e-h-h1", "tr-n-h0", "tr-n-h1")),
    P("שימו לב איך נבחר כל צומת: בשורש ״חם״, כי הוא היה הכי טוב שם. בענף ״חם״ דיאטה, לפי חמש השורות שלו בלבד.",
      do=show("g-2")),
    P("ואחרי שבחר, ID3 לא חזר לבדוק אם שורש אחר היה נותן עץ אחר. בחירה כזו, צעד אחרי צעד, נקראת **חמדנית**.",
      say="ואחרי שבחר, אַיי דִּי שָׁלוֹשׁ לא חזר לבדוק אם שורש אחר היה נותן עץ אחר. בחירה כזו, צעד אחרי צעד, נקראת חמדנית."),
    P("ועוד שם שכדאי להכיר: רווח המידע של מאפיין ביחס למחלקה נקרא גם mutual information. כאן, של ״חם״, בערך 0.396.",
      do=show("g-3"),
      say="ועוד שם שכדאי להכיר: רווח המידע של מאפיין ביחס למחלקה נקרא גם מְיוּצ'וּאַל אִינְפוֹרְמֵיישֶׁן. כאן, של חם, בערך 0.396.",
      hold=0.4),
], stage=TREE,
   panel=st.card("g-1", "plain", r"בענף ״חם״: דיאטה $0$, סופ״ש $0.8$")
         + st.card("g-2", "ex", "שורש: ״חם״ · בענף ״חם״: דיאטה")
         + st.card("g-3", "plain", r"רווח המידע של ״חם״: $0.881 - 0.485 \approx 0.396$"))

leaf = Scene("leaf", "מתי עוצרים", [
    P("בעץ שלנו כל עלה יצא טהור. אבל לפעמים צריך לעצור לפני כן."),
    P("נניח טבלה אחרת, עם שני ימים בדיוק באותם ערכים: חם, סופ״ש ובלי דיאטה. ביום אחד קניתי, ובשני לא.", do=show("e-1")),
    P("כל מאפיין שולח את שניהם לאותו ענף. אין שאלה שתפריד ביניהם, ולכן הצומת נשאר עלה: לפי הרוב, ובתיקו בחירה שרירותית.", do=show("e-2")),
    P("ועוד עצירה, בעץ שלנו: אילו קבענו מראש עומק מרבי אחד, הענף ״חם״ היה נעצר כעלה ״כן״, שלוש מול שתיים.", do=show("e-3"),
      hold=0.4),
], stage=w.col(st.card("e-1", "ex", "חם, סופ״ש, בלי דיאטה ← <b>קנה</b><br>חם, סופ״ש, בלי דיאטה ← <b>לא קנה</b>"),
               st.card("e-2", "ex", "אותו וקטור, מחלקות שונות: אף מאפיין לא מפריד"),
               st.card("e-3", "ex", "עומק מרבי $1$: הענף ״חם״ ← עלה ״כן״ ($3$ מול $2$)"), gap=30),
   panel="")

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: מתי נוצר עלה בעץ החלטה?"),
    P("כשלכל הדוגמאות בצומת אותה מחלקה. זה מה שראינו בכל העלים של העץ שלנו.", do=show("q-ok-0")),
    P("כשהעץ הגיע לעומק המרבי שנקבע מראש. גם אם הצומת מעורבב, אסור לפצל אותו יותר.", do=show("q-ok-1")),
    P("וכשכל הערכים של המאפיינים שנותרו זהים, כמו שני הימים הזהים. אין שאלה שתפריד.", do=show("q-ok-3")),
    P("שלוש הטענות נכונות, ולכן התשובה היא ״כל התשובות נכונות״.", do=show("q-ok-2")),
    P("המלכודת: לסמן רק את הצומת הטהור, כי זה המקרה המוכר. אבל עלה נוצר גם כשעוצרים בגלל מגבלה, או בגלל נתונים זהים.",
      do=show("x-t"), hold=0.6),
], stage=w.col(w.stem(Q["id"], "מתי נוצר עלה בעץ החלטה?"), w.opts("q", OPTS, size=28),
               st.card("x-t", "trap", "⚠️ לא רק צומת טהור: גם עומק מרבי, וגם וקטורים זהים"), gap=20),
   panel=st.card("x-1", "ex", "טהור ← עלה").replace(' h"', '"')
         + st.card("x-2", "ex", "עומק מרבי ← עלה").replace(' h"', '"')
         + st.card("x-3", "ex", "אותם ערכים ← עלה").replace(' h"', '"'))

recap = Scene("recap", "סיכום", [
    P("לסיכום: ID3 בונה מלמעלה למטה. צומת טהור הוא עלה.", do=show("r-1"),
      say="לסיכום: אַיי דִּי שָׁלוֹשׁ בונה מלמעלה למטה. צומת טהור הוא עלה."),
    P("אחרת בוחרים את המאפיין עם האנטרופיה המותנית הנמוכה, כלומר רווח המידע הגבוה.", do=show("r-2")),
    P("כל ענף ממשיך לבד, רק עם השורות שלו. והבחירה חמדנית.", do=show("r-3")),
    P("ועוצרים גם בעומק מרבי, או כשנשארו וקטורים זהים. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=w.col(w.eq("r-eq", r"$F^* = \arg\min_F\ H(Y \mid F)$", 60, hidden=False), gap=30),
   panel=st.card("r-1", "rule", "🔑 טהור ← עלה")
         + st.card("r-2", "rule", "🔑 $H(Y|F)$ נמוך ← $IG$ גבוה")
         + st.card("r-3", "rule", "🔑 כל ענף עם השורות שלו")
         + st.card("r-4", "trap", "⚠️ עלה גם בעצירה מראש ובנתונים זהים"))

VIDEO = Video("C05", "cls", "ID3: בחירת הפיצול ובניית העץ", [title, loop, root, grow, leaf, exam, recap],
              bank={"PRAC-X-Q6": 2})
