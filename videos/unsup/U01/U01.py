# -*- coding: utf-8 -*-
"""U01 · K-Means צעד אחר צעד — video script (build: tools/video/vid.py U01).

Source: courses/unsup/U01.js (lesson + goal). Exam part: an ORIGINAL "בסגנון מבחן" question (bank={}): the update step
run backwards (a centroid moved to (4,3), two of its three points are known, find the third: (5,4)). SAMP-1-Q7 was
dropped: it is the source twin of the boss item U08-Q1 (both "fv_b -> fv_a" under Chebyshev).
Avoided on purpose: the quiz items of U01 (25S-A-Q20, 25B-A-Q13 and the originals: the (3,5)/(4,5)/(2,1) assignments,
the 1-D run 1,2,4,9,10, the update (1,2),(3,4),(5,9)), the boss items (24S-A-Q14, U08-Q1's Chebyshev triple, U08-Q2's
four-point run) and the lesson's own numbers. Worked examples use new data: the point P=(4,3) against three centroids
(a Chebyshev tie), and a six-point run whose bad start (two centroids in one corner) is repaired in two iterations.
Every distance, assignment and mean below is computed and asserted here.
"""
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- distances (scene "dist")
PT = np.array([4, 3])
CEN = {"A": np.array([1, 1]), "B": np.array([6, 7]), "C": np.array([7, 2])}
DIFF = {k: np.abs(PT - v) for k, v in CEN.items()}
MAN = {k: int(d.sum()) for k, d in DIFF.items()}
EUC = {k: float(np.sqrt((d ** 2).sum())) for k, d in DIFF.items()}
CHE = {k: int(d.max()) for k, d in DIFF.items()}
assert [tuple(DIFF[k]) for k in "ABC"] == [(3,2), (2,4), (3,1)]
assert MAN == {"A": 5, "B": 6, "C": 4} and min(MAN, key=MAN.get) == "C"
assert [num(EUC[k]) for k in "ABC"] == ["3.61", "4.47", "3.16"] and min(EUC, key=EUC.get) == "C"
assert CHE == {"A": 3, "B": 4, "C": 3}                  # tie between A and C; B is strictly farther than both
assert CHE["B"] > CHE["A"] == CHE["C"]

# ---------------------------------------------------------------- the run (scene "run"), Euclidean
X = np.array([(1,1), (2,3), (3,2), (9,4), (10,6), (11,5)], float)
C0 = np.array([(1,1), (2,3)], float)                  # "random" start: two points of the same corner


def lloyd(X, C):
    hist = []
    while True:
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        a = d.argmin(1)
        assert all(sorted(r)[0] < sorted(r)[1] for r in d), "no ties in this run"
        new = np.array([X[a == k].mean(0) for k in range(len(C))])
        hist.append((C.copy(), d, a.copy(), new.copy()))
        if np.allclose(new, C):
            return hist
        C = new


H = lloyd(X, C0)
assert len(H) == 3                                       # two moves, then a third pass changes nothing
(C_0, D1, A1, C_1), (_, D2, A2, C_2), (_, _, A3, C_3) = H
assert A1.tolist() == [0, 1, 1, 1, 1, 1]
assert num(D1[2][0]) == "2.24" and num(D1[2][1]) == "1.41"          # (3,2): sqrt5 from c1, sqrt2 from c2
assert C_1.tolist() == [[1, 1], [7, 4]]
assert X[A1 == 1].sum(0).tolist() == [35, 20]                       # 35/5 = 7, 20/5 = 4
assert A2.tolist() == [0, 0, 0, 1, 1, 1]
assert num(D2[1][0]) == "2.24" and num(D2[1][1]) == "5.1"           # (2,3): sqrt5 from c1=(1,1), sqrt26 from c2=(7,4)
assert C_2.tolist() == [[2, 2], [10, 5]] and (A3 == A2).all() and np.allclose(C_3, C_2)

# ---------------------------------------------------------------- exam: original, בסגנון מבחן (update step backwards)
XC, XA, XB = np.array([4, 3]), np.array([2, 2]), np.array([5, 3])
XP = 3 * XC - XA - XB
assert XP.tolist() == [5, 4] and np.array_equal((XA + XB + XP) / 3, XC)
assert ((XA + XB) / 2).tolist() == [3.5, 2.5]                     # distractor: the mean of the two known points
assert (3 * XC).tolist() == [12, 9]                               # distractor: the sum, nothing subtracted
OPTS = [(0, W.ltr("(5, 4)")), (1, W.ltr("(4, 3)")), (2, W.ltr("(3.5, 2.5)")), (3, W.ltr("(12, 9)"))]

# ---------------------------------------------------------------- widgets
def idea_plot():
    p = W.Plot("ip", (0,12), (0,8), w=900, h=600)
    for x, y in X:
        p.pt(x, y)
    p.begin("ip-c")
    for k, (x, y) in enumerate(C_0):
        p.cen(x, y, k, hidden=False, label=f"c{k + 1}")
    p.end()
    return p.svg()


def dist_plot():
    p = W.Plot("dp", (0,8), (0,8), w=660, h=620)
    for k, name in enumerate("ABC"):
        p.seg(PT, CEN[name], id_=f"dp-s{name}", k=k, dash=True)
    for k, name in enumerate("ABC"):
        p.cen(*CEN[name], k, hidden=False, label=name)
    p.pt(*PT, label="P (4,3)", r=14)
    return p.svg()


def run_plot(f="rp", final=False):
    p = W.Plot(f, (0,12), (0,8), w=900, h=600)
    for i, (x, y) in enumerate(X):
        p.pt(x, y, label=f"({num(x)},{num(y)})")
    for it, A in ((1, A1), (2, A2)):
        p.begin(f"{f}-a{it}", hidden=not (final and it == 2))
        for (x, y), k in zip(X, A):
            p.pt(x, y, k=int(k))
        p.end()
    for it, C in ((0, C_0), (1, C_1), (2, C_2)):
        p.begin(f"{f}-c{it}", hidden=not (final and it == 2))
        for k, (x, y) in enumerate(C):
            p.cen(x, y, k, hidden=False, label=f"c{k + 1}", side="l" if k == 0 else "r")
        p.end()
    return p.svg()


DT = W.table("dt", ["", "הפרשים", "מנהטן", "אוקלידי", "צ׳בישב"],
             [[n, f"{DIFF[n][0]}, {DIFF[n][1]}", str(MAN[n]), "≈" + num(EUC[n]), str(CHE[n])] for n in "ABC"],
             hide={(r, c) for r in range(3) for c in range(1,5)}, hl={(2,2), (2,3), (0,4), (2,4)}, cls="sm")
col = lambda c: [f"dt-{r}-{c}" for r in range(3)]


# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נריץ K-Means ביד, צעד אחר צעד."),
    P("נראה איך מחליטים לאיזה מרכז נקודה שייכת, מה עושים כשיש תיקו, ואיך המרכזים זזים."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">U01 · אשכול, טקסט ותמונה</div><h1>K-Means צעד אחר צעד</h1>'
         '<p>לשייך וקטור למרכז הקרוב, לזהות תיקו ולענות עליו נכון, ולחשב את המרכזים החדשים אחרי איטרציה.</p>',
   layout="title")

idea = Scene("idea", "הרעיון", [
    P("יש לנו שש נקודות, ולאף אחת מהן אין תווית."),
    P("אף אחד לא אומר מה התשובה הנכונה. זו למידה לא מונחית: מחפשים קבוצות לבד."),
    P("את מספר הקבוצות, $K$, קובעים מראש. כאן $K$ שווה 2.",
      say="את מספר הקבוצות, קֵיי, קובעים מראש. כאן קֵיי שווה שתיים."),
    P("כל קבוצה מיוצגת בנקודה אחת, המרכז שלה. נסמן אותו ביהלום.", do=show("ip-c")),
    P("בהתחלה המרכזים נבחרים באקראי. כאן נפלו במקרה שני מרכזים באותה פינה.", hold=0.3),
    P("ואז שני שלבים. שיוך: כל נקודה הולכת למרכז הקרוב אליה ביותר.", do=show("i-s1")),
    P("עדכון: כל מרכז עובר לממוצע של הנקודות שהגיעו אליו.", do=show("i-s2")),
    P("וחוזרים על שניהם, עד שאף נקודה לא מחליפה קבוצה. זו ההתכנסות.", do=show("i-s3"), hold=0.4),
], stage=idea_plot(),
   panel=st.card("i-s1", "ex", "<b>שיוך:</b> כל נקודה למרכז <b>הקרוב</b> ביותר")
         + st.card("i-s2", "ex", "<b>עדכון:</b> כל מרכז ל<b>ממוצע</b> הנקודות שלו")
         + st.card("i-s3", "rule", "🔑 חוזרים עד שהשיוך לא משתנה"))

dist = Scene("dist", "מה זה קרוב", [
    P("אבל מה זה הקרוב ביותר? השאלה תמיד אומרת באיזה מרחק להשתמש."),
    P("הנקודה $P$ נמצאת ב-(4,3), ויש שלושה מרכזים.", say="הנקודה פִּי נמצאת בארבע, שלוש, ויש שלושה מרכזים.",
      do=show("dp-sA", "dp-sB", "dp-sC")),
    P("בכל המדדים מתחילים מההפרש בכל ציר. ל-$A$ ההפרשים הם 3 ו-2.", do=show("dt-0-1")),
    P("ל-$B$ ההפרשים 2 ו-4, ול-$C$ ההפרשים 3 ו-1.", do=show("dt-1-1", "dt-2-1")),
    P("מנהטן מחבר את ההפרשים: 5, 6 ו-4.", do=show(*col(2))),
    P("אוקלידי מעלה בריבוע, מחבר ולוקח שורש: בערך 3.61, 4.47 ו-3.16.", do=show(*col(3))),
    P("בשני המדדים האלה $C$ הכי קרוב, ו-$P$ הולכת אליו.", do=show("dt-hl-2-2", "dt-hl-2-3")),
    P("צ׳בישב לוקח רק את ההפרש הגדול מבין השניים: 3, 4 ו-3.", do=show(*col(4))),
    P("ופה יש תיקו: $A$ ו-$C$ שניהם במרחק 3.", do=show("dt-hl-0-4", "dt-hl-2-4"), hold=0.3),
    P("בתיקו אי אפשר לדעת לאן $P$ תלך. אולי ל-$A$, ואולי ל-$C$.", do=show("d-c1")),
    P("מה כן ודאי? רק ש-$B$ בחוץ: הוא רחוק יותר משניהם, ולכן $P$ לא תגיע אליו.", do=show("d-c2"), hold=0.4),
], stage=dist_plot(),
   panel=DT + st.card("d-c1", "trap", "⚠️ תיקו במינימום: אי אפשר לטעון לאן היא <b>תלך</b>")
         + st.card("d-c2", "rule", "🔑 ודאי רק מה שנפסל: המרכז הרחוק משניהם"))

run = Scene("run", "ריצה", [
    P("עכשיו נריץ את האלגוריתם על שש הנקודות, עם מרחק אוקלידי."),
    P("המרכזים ההתחלתיים: $c_1$ ב-(1,1), ו-$c_2$ ב-(2,3).", do=show("rp-c0"),
      say="המרכזים ההתחלתיים: סִי אחת באחת, אחת, וסִי שתיים בשתיים, שלוש."),
    P("איטרציה ראשונה, שיוך. הנקודה (3,2) רחוקה 2.24 מ-$c_1$, ורק 1.41 מ-$c_2$.",
      say="איטרציה ראשונה, שיוך. הנקודה שלוש, שתיים רחוקה 2.24 מסִי אחת, ורק 1.41 מסִי שתיים."),
    P("ככה בודקים כל נקודה: ל-$c_1$ הגיעה רק נקודה אחת, וכל החמש האחרות הלכו ל-$c_2$.", do=show("rp-a1")),
    P("עדכון: $c_1$ נשאר במקום, כי הממוצע של נקודה אחת הוא הנקודה עצמה."),
    P("ו-$c_2$ עובר לממוצע של חמש הנקודות שלו, בכל ציר בנפרד.", do=show("r-m")),
    P("בציר הראשון 35 חלקי 5, כלומר 7. בשני 20 חלקי 5, כלומר 4.", do=show("r-m1", "r-m2")),
    P("המרכז קפץ ל-(7,4).", say="המרכז קפץ לשבע, ארבע.", do=hide("rp-c0") + show("rp-c1"), hold=0.3),
    P("איטרציה שנייה. עכשיו (2,3) רחוקה 2.24 מ-$c_1$, ו-5.1 מ-$c_2$ החדש.",
      say="איטרציה שנייה. עכשיו הנקודה שתיים, שלוש רחוקה 2.24 מסִי אחת, וחמש נקודה אחת מסִי שתיים החדש."),
    P("לכן היא, וגם (3,2), עוברות קבוצה.", say="לכן היא, וגם הנקודה שלוש, שתיים, עוברות קבוצה.",
      do=hide("rp-a1") + show("rp-a2")),
    P("עדכון: $c_1$ זז ל-(2,2), ו-$c_2$ ל-(10,5).", do=hide("rp-c1") + show("rp-c2"),
      say="עדכון: סִי אחת זז לשתיים, שתיים, וסִי שתיים לעשר, חמש."),
    P("איטרציה שלישית: אף נקודה לא מחליפה קבוצה, ולכן גם המרכזים לא זזים. התכנסנו.", do=show("r-c1"), hold=0.3),
    P("שימו לב: הנקודות עצמן לא זזו אף פעם. רק הצבע שלהן התחלף. מי שזז הם המרכזים.", do=show("r-c2"), hold=0.4),
], stage=run_plot(),
   panel=st.card("r-m", "ex", "המרכז החדש של $c_2$ = ממוצע 5 נקודות"
                 f'<div id="r-m1" class="h">ציר ראשון: {W.ltr("(2+3+9+10+11) ⁄ 5 = <b>7</b>")}</div>'
                 f'<div id="r-m2" class="h">ציר שני: {W.ltr("(3+2+4+6+5) ⁄ 5 = <b>4</b>")}</div>')
         + st.card("r-c1", "rule", "🔑 שיוך לא השתנה ← התכנסות")
         + st.card("r-c2", "trap", "⚠️ הנקודות לא זזות; רק המרכזים"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: אחרי שלב העדכון, המרכז של אשכול זז ל-(4,3). לאשכול שויכו שלוש נקודות.",
      say="שאלה בסגנון מבחן: אחרי שלב העדכון, המרכז של אשכול זז לארבע, שלוש. לאשכול שויכו שלוש נקודות."),
    P("שתיים מהן ידועות, (2,2) ו-(5,3). מה הנקודה השלישית?",
      say="שתיים מהן ידועות, שתיים, שתיים וחמש, שלוש. מה הנקודה השלישית?", do=show("x-v")),
    P("המרכז הוא ממוצע, אז סכום שלוש הנקודות הוא שלוש פעמים המרכז: (12,9).",
      say="המרכז הוא ממוצע, אז סכום שלוש הנקודות הוא שלוש פעמים המרכז: שתים עשרה, תשע.", do=show("x-1")),
    P("מורידים את שתי הידועות, בכל ציר בנפרד: 12 פחות 2 פחות 5 זה 5, ו-9 פחות 2 פחות 3 זה 4.",
      say="מורידים את שתי הידועות, בכל ציר בנפרד: שתים עשרה פחות שתיים פחות חמש זה חמש, ותשע פחות שתיים פחות שלוש זה ארבע.",
      do=show("x-2", "q-ok-0")),
    P("בדיקה: הממוצע של (2,2), (5,3) ו-(5,4) הוא בדיוק (4,3).",
      say="בדיקה: הממוצע של שתיים, שתיים, חמש, שלוש וחמש, ארבע הוא בדיוק ארבע, שלוש.", do=show("x-3")),
    P("(4,3) הוא המרכז עצמו, לא נקודה של האשכול.", say="ארבע, שלוש הוא המרכז עצמו, לא נקודה של האשכול.",
      do=show("q-x-1", "q-n-1")),
    P("(3.5,2.5) הוא הממוצע של שתי הידועות בלבד. ו-(12,9) הוא הסכום, בלי שהורדנו כלום.",
      say="שלוש וחצי, שתיים וחצי הוא הממוצע של שתי הידועות בלבד. ושתים עשרה, תשע הוא הסכום, בלי שהורדנו כלום.",
      do=show("q-x-2", "q-n-2", "q-x-3", "q-n-3")),
    P("המלכודת: לעצור באמצע הדרך. קודם הסכום, אחר כך החיסור, ובסוף בדיקה שהממוצע יוצא נכון.",
      do=show("x-t"), hold=0.4),
], stage=W.stack(W.note("x-v", "מרכז " + W.ltr("(4, 3)") + " · ידועות " + W.ltr("(2, 2), (5, 3)"), size=34),
                 W.big("x-1", "$3 \\cdot (4, 3) = (12, 9)$", size=42),
                 W.big("x-2", "$(12 - 2 - 5,\\ 9 - 2 - 3) = (5, 4)$", size=42),
                 W.big("x-3", "$\\tfrac{1}{3}\\big((2,2) + (5,3) + (5,4)\\big) = (4, 3)$", size=40),
                 st.card("x-t", "trap", "⚠️ סכום ← חיסור ← בדיקה"), gap=26),
   panel='<div class="qstem"><span class="qid">בסגנון מבחן</span>אחרי עדכון, המרכז של אשכול הוא (4,3). '
         'שויכו אליו שלוש נקודות: (2,2), (5,3) ונקודה נוספת. מהי הנקודה הנוספת?</div>'
         + st.options("q", OPTS, notes={1: "המרכז", 2: "רק שתיים", 3: "בלי חיסור"}))

recap = Scene("recap", "סיכום", [
    P("לסיכום: מרכזים אקראיים, שיוך לקרוב, מרכז לממוצע, וחוזרים עד שאין שינוי.", do=show("s-1")),
    P("המדד קובע מי קרוב: מנהטן מחבר, אוקלידי לוקח שורש של סכום הריבועים, וצ׳בישב לוקח את הגדול.", do=show("s-2")),
    P("בתיקו לא יודעים לאן הנקודה תלך. ודאי רק שהיא לא תלך למרכז שרחוק משניהם.", do=show("s-3")),
    P("והמרכז החדש הוא ממוצע, בכל ציר בנפרד. נתראה ברמה הבאה.", do=show("s-4"), hold=0.6),
], stage=run_plot("rq", final=True),
   panel=st.card("s-1", "rule", "🔑 שיוך ← עדכון ← שוב, עד התכנסות")
         + st.card("s-2", "rule", "🔑 מנהטן Σ · אוקלידי √Σ² · צ׳בישב max")
         + st.card("s-3", "trap", "⚠️ תיקו: ודאי רק מה שנפסל")
         + st.card("s-4", "rule", "🔑 מרכז חדש = ממוצע לכל ציר"))

VIDEO = Video("U01", "unsup", "K-Means צעד אחר צעד", [title, idea, dist, run, exam, recap], bank={})
