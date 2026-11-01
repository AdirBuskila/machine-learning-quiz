# -*- coding: utf-8 -*-
"""U02 · מה נכון על K-Means — video script (build: tools/video/vid.py U02).

Source: courses/unsup/U02.js (lesson + goal). Exam anchor: 23B-B-Q20 (preprocessing before k-means: scaling, missing
values, categorical values -> "all are correct"), which no guided quiz uses.
Avoided on purpose: the quiz items of U02 (23B-B-Q8 with its kmeans++ claim, PRAC-X-Q23, SAMP-1-Q14 and the originals:
the age/income customer, the "city" feature, the boundary between (1,1) and (5,1), the circle), their twins
(23B-A-Q8 / 24S-C-Q15, PRAC-X-Q4, SAMP-2-Q9 / SAMP-3-Q9 / PRAC-X-Q36) and kmeans++ altogether. The video does not
say "run several random starts and keep the best" (U02-Q2), does not mention seeds, never says a centroid cannot lie
on the boundary (U02-Q6), and never says "convert categories to numbers" (U02-Q4); it shows the rules on new data:
twelve points where one random start ends in a local optimum (sum of squared distances 82) and another in the natural
split (12); a rating/price product whose assignment flips after min-max scaling; and the boundary between the
centroids (1,4) and (5,2), the line y = 2x - 3, with (4,5) on it and (3,4) on c1's side.
Every run, distance and line below is computed and asserted here.
"""
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- local optimum (scene "local")
G = [(1,2), (2,1), (2,3), (3,2), (7,2), (8,1), (8,3), (9,2), (4,7), (5,6), (5,8), (6,7)]
X = np.array(G, float)


def lloyd(C):
    C = np.array(C, float)
    steps = 0
    while True:
        d = ((X[:, None] - C[None]) ** 2).sum(-1)
        assert all((np.isclose(r, r.min())).sum() == 1 for r in d), "no ties"
        a = d.argmin(1)
        new = np.array([X[a == k].mean(0) for k in range(len(C))])
        steps += 1
        if np.allclose(new, C):
            nxt = ((X[:, None] - new[None]) ** 2).sum(-1).argmin(1)
            assert (nxt == a).all()                      # converged: one more pass changes nothing
            return new, a, float(((X - new[a]) ** 2).sum()), steps
        C = new


BAD0, GOOD0 = [(3,2), (4,7), (5,8)], [(2,3), (8,1), (5,6)]
assert all(p in G for p in BAD0 + GOOD0)                 # starts are data points, as a random pick would be
CB, AB, WB, NB = lloyd(BAD0)
CG, AG, WG, NG = lloyd(GOOD0)
assert NB == NG == 2                                      # one move, then a pass that changes nothing
assert CB.tolist() == [[5, 2], [4.5, 6.5], [5.5, 7.5]] and WB == 82
assert CG.tolist() == [[2, 2], [8, 2], [5, 7]] and WG == 12
assert set(AB[:8]) == {0}                                # the two bottom groups share one centroid
assert len(set(AB[8:])) == 2                             # the top group is cut in two


# ---------------------------------------------------------------- scaling (scene "scale"): (rating 1..5, price 100..500)
PR, PA, PB = np.array([4.5, 300]), np.array([4.6, 380]), np.array([1.2, 310])
d = lambda u, v: float(np.sqrt(((u - v) ** 2).sum()))
assert num(d(PR, PA)) == "80" and num(d(PR, PB)) == "10.53" and d(PR, PB) < d(PR, PA)
LO, HI = np.array([1, 100]), np.array([5, 500])
mm = lambda v: (v - LO) / (HI - LO)
assert np.allclose(mm(PR), [0.875, 0.5]) and np.allclose(mm(PA), [0.9, 0.7]) and np.allclose(mm(PB), [0.05, 0.525])
assert num(d(mm(PR), mm(PA))) == "0.2" and num(d(mm(PR), mm(PB))) == "0.83" and d(mm(PR), mm(PA)) < d(mm(PR), mm(PB))
assert num(abs(PR - PB)[0]) == "3.3" and num(abs(PR - PB)[1]) == "10" and num(abs(PR - PA)[1]) == "80"

# ---------------------------------------------------------------- boundary (scene "bound")
M1, M2 = np.array([1, 4]), np.array([5, 2])
MID = (M1 + M2) / 2
assert MID.tolist() == [3, 3]
line = lambda x: 2 * x - 3                                 # perpendicular bisector of M1M2
seg_dir = M2 - M1
assert np.dot(seg_dir, np.array([1, 2])) == 0             # line direction (1,2) is perpendicular to the segment
for x in (2, 3, 4):
    p = np.array([x, line(x)])
    assert np.isclose(d(p, M1), d(p, M2))
assert ((np.array([4, 5]) - M1) ** 2).sum() == 10 == ((np.array([4, 5]) - M2) ** 2).sum()
R = np.array([3, 4])
assert R[1] > line(R[0]) and d(R, M1) == 2 and num(d(R, M2)) == "2.83"

# ---------------------------------------------------------------- exam 23B-B-Q20
Q = BANK["23B-B-Q20"]
assert Q["correctIndex"] == 3 and "סילום" in Q["options"][0] and "חסרים" in Q["options"][1]
assert "קטגוריאליים" in Q["options"][2] and Q["options"][3] == "כל התשובות נכונות"
OPTS = [(0, "מומלץ לבצע סילום"), (1, "ערכים חסרים: למלא אותם, או לוותר על המאפיין או הדוגמה"),
        (2, "ערכים קטגוריאליים: להחליף בערכים מספריים"), (3, "כל התשובות נכונות")]


# ---------------------------------------------------------------- widgets
def plot_with_start(pfx, C0, C, A):
    p = W.Plot(pfx, (0,10), (0,9), w=500, h=470, labels=False)
    for x, y in X:
        p.pt(x, y, r=11)
    p.begin(f"{pfx}-s")
    for k, (x, y) in enumerate(C0):
        p.cen(x, y, k, hidden=False, s=16)
    p.end()
    p.begin(f"{pfx}-a")
    for (x, y), k in zip(X, A):
        p.pt(x, y, k=int(k), r=11)
    for k, (x, y) in enumerate(C):
        p.cen(x, y, k, hidden=False, s=16)
    p.end()
    return p.svg()


LOCAL = W.row(
    W.stack(plot_with_start("lb", BAD0, CB, AB), W.note("lb-w", f"סכום ריבועי המרחקים: <b>{num(WB)}</b>"), gap=6),
    W.stack(plot_with_start("lg", GOOD0, CG, AG), W.note("lg-w", f"סכום ריבועי המרחקים: <b>{num(WG)}</b>"), gap=6),
    gap=30)


def bound_plot():
    p = W.Plot("bp", (0,7), (0,7), w=640, h=620)
    p.begin("bp-l")
    p.seg((1.5,0), (5,7), k=None, hidden=False, width=5)
    p.end()
    p.seg(tuple(M1), tuple(M2), id_="bp-m", dash=True)
    p.cen(*M1, 0, hidden=False, label="c1", side="l")
    p.cen(*M2, 1, hidden=False, label="c2")
    p.pt(3, 3, id_="bp-mid", hidden=True, label="(3,3)", r=10)
    p.pt(4, 5, id_="bp-q", hidden=True, k=2, label="(4,5)", r=11)
    p.seg((4,5), tuple(M1), id_="bp-q1", k=2, dash=True, label="√10", lab_dy=-14)
    p.seg((4,5), tuple(M2), id_="bp-q2", k=2, dash=True, label="√10", lab_dx=34, lab_dy=0)
    p.pt(3, 4, id_="bp-r", hidden=True, k=3, label="(3, 4)", r=11)
    p.seg((3,4), tuple(M1), id_="bp-r1", k=3, dash=True, label="2", lab_dy=-14)
    p.seg((3,4), tuple(M2), id_="bp-r2", k=3, dash=True, label="2.83", lab_dx=30, lab_dy=4)
    return p.svg()


ST = W.table("sc", ["", "דירוג", "מחיר", "מרחק"],
             [["מוצר חדש", "4.5", "300", ""], ["$A$", "4.6", "380", "80"], ["$B$", "1.2", "310", "10.53"]],
             hide={(1,3), (2,3)}, hl={(2,3)})
ST2 = W.table("sd", ["", "דירוג", "מחיר", "מרחק"],
              [["מוצר חדש", "0.875", "0.5", ""], ["$A$", "0.9", "0.7", "0.2"], ["$B$", "0.05", "0.525", "0.83"]],
              hide={(r, c) for r in range(3) for c in range(1,4)}, hl={(1,3)})

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבין מה K-Means מבטיח, ומה הוא לא מבטיח."),
    P("נראה ריצה שנתקעת, נראה למה חייבים סילום, ואיפה עובר הקו בין שני אשכולות."),
    P("ובסוף שאלה ממבחן על הכנת הנתונים.", hold=0.4),
], stage='<div class="kicker">U02 · אשכול, טקסט ותמונה</div><h1>מה נכון על K-Means</h1>'
         '<p>לזהות טענות נכונות ושגויות על K-Means, להתמודד עם אופטימום מקומי, להבין למה צריך סילום, ואיך נראה גבול ההחלטה.</p>',
   layout="title")

local = Scene("local", "אופטימום מקומי", [
    P("שתים עשרה נקודות, ובעין רואים שלוש קבוצות ברורות. מריצים K-Means עם $K$ שווה 3.",
      say="שתים עשרה נקודות, ובעין רואים שלוש קבוצות ברורות. מריצים קֵיי מינס עם קֵיי שווה שלוש."),
    P("משמאל, ההגרלה בחרה שלושה מרכזים התחלתיים: אחד למטה, ושניים צמודים למעלה.", do=show("lb-s")),
    P("מימין, הגרלה אחרת: מרכז אחד בכל קבוצה.", do=show("lg-s")),
    P("שתי הריצות מתכנסות אחרי שתי איטרציות. מימין יצאו שלוש הקבוצות הטבעיות.",
      do=hide("lb-s", "lg-s") + show("lb-a", "lg-a")),
    P("ומשמאל? שתי הקבוצות התחתונות חולקות מרכז אחד, והקבוצה העליונה נחתכה לשניים.", hold=0.3),
    P("נמדוד את זה: סכום ריבועי המרחקים של כל נקודה מהמרכז שלה. משמאל 82, ומימין רק 12.",
      do=show("lb-w", "lg-w")),
    P("כל צעד של האלגוריתם רק מקטין את הסכום הזה, או משאיר אותו. לכן הוא תמיד נעצר."),
    P("אבל משמאל הוא נעצר במקום גרוע: עוד איטרציה לא תזיז אף נקודה, אז הוא לא ייצא משם לבד.", do=show("l-c1")),
    P("זה אופטימום מקומי: אין צעד קטן שמשפר, למרות שקיימת חלוקה הרבה יותר טובה.", hold=0.3),
    P("ההבדל בין שתי הריצות היה רק ההגרלה בהתחלה. והסכום הוא מה שמראה לנו איזו מהן הצליחה יותר: 12 מול 82.",
      do=show("l-c2"), hold=0.4),
], stage=LOCAL,
   panel=st.card("l-c1", "trap", "⚠️ התכנס ≠ הגיע לחלוקה הטובה ביותר")
         + st.card("l-c2", "rule", "🔑 משווים ריצות לפי סכום ריבועי המרחקים"))

scale = Scene("scale", "סילום", [
    P("עכשיו מוצרים עם שני מאפיינים: דירוג בין 1 ל-5, ומחיר בין 100 ל-500 שקל."),
    P("מוצר חדש: דירוג 4.5, מחיר 300. מרכז $A$ דומה לו בדירוג, ומרכז $B$ דומה לו במחיר.", do=show("sc")),
    P("ההפרש מ-$A$ במחיר הוא 80 שקל. ההפרש מ-$B$ בדירוג הוא 3.3, ובמחיר רק 10."),
    P("המרחק האוקלידי ל-$A$ יוצא 80, ול-$B$ בערך 10.53. המוצר הולך ל-$B$.", do=show("sc-1-3", "sc-2-3", "sc-hl-2-3")),
    P("כלומר מוצר מצוין שובץ עם מוצר גרוע, רק כי המחיר נמדד במספרים גדולים.", hold=0.3),
    P("הפתרון: סילום. מביאים כל מאפיין לטווח 0 עד 1: מחסירים את המינימום, ומחלקים ברוחב הטווח.", do=show("s-f")),
    P("אחרי הסילום המרחק ל-$A$ הוא 0.2, ול-$B$ בערך 0.83. עכשיו המוצר הולך ל-$A$.",
      do=show(*W.all_cells("sd", 3, 4, 0, 1), "sd-hl-1-3")),
    P("שום מאפיין לא אמור לנצח רק בגלל היחידות שלו. כל אלגוריתם שמבוסס מרחק צריך את זה.", do=show("s-c"), hold=0.4),
], stage=W.stack(ST, W.big("s-f", "$x' = \\dfrac{x - \\min}{\\max - \\min}$", size=40), ST2, gap=34),
   panel=st.card("s-c", "rule", "🔑 בלי סילום, המאפיין עם המספרים הגדולים קובע את המרחק"))

bound = Scene("bound", "גבול ההחלטה", [
    P("ועכשיו הקו שמפריד בין שני אשכולות. נניח שהמרכזים התכנסו ל-(1,4) ול-(5,2).",
      say="ועכשיו הקו שמפריד בין שני אשכולות. נניח שהמרכזים התכנסו לאחת, ארבע ולחמש, שתיים."),
    P("כל נקודה הולכת למרכז הקרוב. אז הגבול הוא כל הנקודות שבאותו מרחק משניהם.", do=show("bp-m")),
    P("אמצע הקטע, (3,3), הוא נקודה כזו. וכל הנקודות האלה יושבות על ישר אחד, שפונה בזווית ישרה לקטע.",
      say="אמצע הקטע, שלוש, שלוש, הוא נקודה כזו. וכל הנקודות האלה יושבות על ישר אחד, שפונה בזווית ישרה לקטע.",
      do=show("bp-mid", "bp-l")),
    P("כאן הוא הישר $y = 2x - 3$.", say="כאן הוא הישר וַואי שווה שתיים אִקְס פחות שלוש.", do=show("b-eq")),
    P("נבדוק נקודה עליו, (4,5): המרחק לכל אחד מהמרכזים הוא שורש 10. בדיוק תיקו.",
      say="נבדוק נקודה עליו, ארבע, חמש: המרחק לכל אחד מהמרכזים הוא שורש עשר. בדיוק תיקו.",
      do=show("bp-q", "bp-q1", "bp-q2")),
    P("ונקודה מעל הקו, כמו (3,4)? היא במרחק 2 מ-$c_1$ ו-2.83 מ-$c_2$, אז היא בצד של $c_1$.",
      say="ונקודה מעל הקו, כמו שלוש, ארבע? היא במרחק שתיים מסִי אחת ושתיים נקודה שמונים ושלוש מסִי שתיים, אז היא בצד של סִי אחת.",
      do=show("bp-r", "bp-r1", "bp-r2", "b-c"), hold=0.4),
], stage=bound_plot(),
   panel=W.big("b-eq", "$y = 2x - 3$", size=46)
         + st.card("b-c0", "ex", "גבול = נקודות במרחק שווה משני המרכזים").replace(' h"', '"')
         + st.card("b-c", "rule", "🔑 צד הקו קובע את האשכול"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: לפני K-Means על נתוני לקוחות, איזה עיבוד מקדים צריך?"),
    P("סילום כבר ראינו: בלי סילום המאפיין עם המספרים הגדולים שולט במרחק. זו טענה נכונה.", do=show("q-n-0", "x-1")),
    P("ערך חסר: אי אפשר לחשב מרחק כשבאחד הצירים יש חור. ממלאים אותו, או מוותרים על המאפיין או על הדוגמה.",
      do=show("q-n-1", "x-2")),
    P("ערך קטגוריאלי, כמו שם של מחלקה: כספים פחות שיווק? אין לזה מובן. גם זו טענה נכונה.", do=show("q-n-2", "x-3")),
    P("שלוש הטענות נכונות, וכל אחת לבדה חלקית. לכן התשובה היא כל התשובות נכונות.", do=show("q-ok-3"), hold=0.3),
    P("המלכודת: לסמן את הסילום כי הוא הכי מוכר, ולעצור. כשכל טענה עומדת בבדיקה, בוחרים את האפשרות הכוללת.",
      do=show("x-t"), hold=0.4),
], stage=W.stack(st.card("x-1", "ex", "סילום: אף מאפיין לא שולט בגלל היחידות"),
                 st.card("x-2", "ex", "חסר: אין מרחק בלי ערך בכל ציר"),
                 st.card("x-3", "ex", "קטגוריאלי: אין חיסור בין מחרוזות"),
                 st.card("x-t", "trap", "⚠️ כל הטענות נכונות ← האפשרות הכוללת"), gap=24),
   panel=f'<div class="qstem"><span class="qid">{Q["id"]}</span>בונים clustering מבוסס k-means. '
         'איזה עיבוד מוקדם יש לבצע?</div>'
         + st.options("q", OPTS, notes={0: "נכון", 1: "נכון", 2: "נכון"}))

recap = Scene("recap", "סיכום", [
    P("לסיכום: K-Means תמיד מתכנס, אבל לאן, זה תלוי בהתחלה. הוא יכול להיתקע באופטימום מקומי.",
      do=show("r-1", "mb-a", "mg-a", "mb-w", "mg-w")),
    P("את הריצות משווים לפי סכום ריבועי המרחקים: כאן 12 מול 82.", do=show("r-2")),
    P("לפני הריצה בודקים שלושה דברים: סקאלות שונות, ערכים חסרים, וערכים שאינם מספרים.", do=show("r-3")),
    P("והגבול בין שני מרכזים: כל הנקודות שבמרחק שווה משניהם. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=LOCAL.replace('id="lb', 'id="mb').replace('id="lg', 'id="mg'),
   panel=st.card("r-1", "trap", "⚠️ התכנסות ≠ החלוקה הטובה ביותר")
         + st.card("r-2", "rule", "🔑 משווים ריצות לפי הסכום")
         + st.card("r-3", "rule", "🔑 סקאלות · חסרים · ערכים לא מספריים")
         + st.card("r-4", "rule", "🔑 גבול = מרחק שווה משני המרכזים"))

VIDEO = Video("U02", "unsup", "מה נכון על K-Means", [title, local, scale, bound, exam, recap], bank={"23B-B-Q20": 3})
