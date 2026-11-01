# -*- coding: utf-8 -*-
"""C04 · אנטרופיה ורווח מידע — video script (build: tools/video/vid.py C04).

Source: courses/cls/C04.js (lesson + goal: entropy in log2 by hand, the weighted conditional entropy, Information
Gain; the base trap and the sign trap).
Exam anchor: PRAC-X-Q10 (IG(class | x2) on a 4-row table, and what it means), which no guided quiz uses.
Not used on purpose: the C04 quiz nodes (4/4, 6/2, 5/5, 9/1, 7/3, 10/0), p = 0.6 / 0.4 in base 10, the 10-row F
split, the 90/10 X2, the H(Y) = 0.881 perfect separator, and its refs SAMP-2-Q22 (with twins SAMP-3-Q22, PRAC-X-Q29)
and 25S-A-Q2; the F/G table of 23B-A-Q2 / 23B-B-Q2 / 24B-A-Q4 / 24B-C-Q20 (quiz + boss refs). The video never states
the "perfect separator -> IG = H(Y)" rule as a card; the exam scene only computes 1 - 0.
Worked examples (new, entropies cross-checked with scipy): a node with 3 Yes / 5 No -> 0.954 (0.287 in base 10);
split by "time": morning 0/3 (pure), evening 3/2 (0.971) -> H(Y|F) = 5/8 * 0.971 = 0.607, IG ~ 0.35; the unweighted
trap gives 0.485.
"""
import math

from scipy.stats import entropy as sp_entropy

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- entropy of a node (verified)
NODE = [3, 5]                                         # Yes, No
HY = w.H(NODE)
assert math.isclose(HY, sp_entropy(NODE, base=2)) and f"{HY:.3f}" == "0.954"
assert f"{math.log2(3 / 8):.3f}" == "-1.415" and f"{math.log2(5 / 8):.3f}" == "-0.678"
assert f"{3 / 8 * math.log2(3 / 8):.3f}" == "-0.531" and f"{5 / 8 * math.log2(5 / 8):.3f}" == "-0.424"
assert w.H([0, 4]) == 0 and w.H([3, 3]) == 1
H10 = w.H(NODE, 10)
assert f"{H10:.3f}" == "0.287" and math.isclose(H10, HY * math.log10(2))
assert math.isclose(math.log2(0.375), math.log10(0.375) / math.log10(2))

# ---------------------------------------------------------------- conditional entropy + IG (verified)
ROWS = [("בוקר", "No"), ("ערב", "Yes"), ("בוקר", "No"), ("ערב", "Yes"),
        ("ערב", "No"), ("בוקר", "No"), ("ערב", "Yes"), ("ערב", "No")]
CL = ("Yes", "No")
assert [sum(1 for r in ROWS if r[1] == c) for c in CL] == NODE
MORN = w.counts(ROWS, 0, 1, "בוקר", CL)
EVE = w.counts(ROWS, 0, 1, "ערב", CL)
assert MORN == [0, 3] and EVE == [3, 2]
HM, HE = w.H(MORN), w.H(EVE)
assert HM == 0 and f"{HE:.3f}" == "0.971" and math.isclose(HE, sp_entropy(EVE, base=2))
HYF = w.cond_H([MORN, EVE])
assert math.isclose(HYF, 3 / 8 * HM + 5 / 8 * HE) and f"{HYF:.3f}" == "0.607"
IG = HY - HYF
assert f"{IG:.2f}" == "0.35"
assert f"{(HM + HE) / 2:.3f}" == "0.485"               # the unweighted trap

# ---------------------------------------------------------------- exam PRAC-X-Q10
Q = BANK["PRAC-X-Q10"]
T = [(1, 1, 1, 1, "Pos"), (1, 1, 0, 0, "Pos"), (0, 0, 1, 0, "Neg"), (0, 0, 0, 0, "Neg")]
CLS = ("Pos", "Neg")
HC = w.H([sum(1 for r in T if r[4] == c) for c in CLS])
G = [w.counts(T, 1, 4, v, CLS) for v in (1, 0)]
assert HC == 1 and G == [[2, 0], [0, 2]] and w.cond_H(G) == 0
assert HC - w.cond_H(G) == 1 and Q["correctIndex"] == 3 and Q["options"][3].startswith("1,")
# no split of this table by any single feature gives IG = 0.5 (the 0.5 options are not a value of anything here)
igs = [HC - w.cond_H([w.counts(T, j, 4, v, CLS) for v in (0, 1)]) for j in range(4)]
assert all(abs(g - 0.5) > 0.15 for g in igs)
OPTS = list(enumerate(Q["options"]))

# ---------------------------------------------------------------- figures
CUR = w.Plot("cu", (0, 1), (0, 1.1), w=700, h=470, xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0.5, 1],
             xname="", yname="H")
xs = [i / 200 for i in range(201)]
CUR.path("curve", [(x, w.H([x, 1 - x]) if 0 < x < 1 else 0) for x in xs], kind="a", hidden=False)
CUR.pts("p0", [(0, 0), (1, 0)], "ok", r=11, hidden=True)
CUR.pts("p5", [(0.5, 1)], "b", r=11, hidden=True)
CUR.pts("p38", [(3 / 8, HY)], "c", r=11, hidden=True)
CPLOT = CUR.svg()

TR = w.Tree("tr", w=900, h=420)
TR.node("r", 450, 60, f'<span class="nm">כל השורות</span>{w.dots(NODE, ("a", "b"))}<span class="hv">H = 0.954</span>',
        kind="q", hidden=False)
TR.node("m", 210, 330, f'<span class="nm">בוקר · 3</span>{w.dots(MORN, ("a", "b"))}<span class="hv">H = 0</span>', kind="leaf")
TR.node("e", 690, 330, f'<span class="nm">ערב · 5</span>{w.dots(EVE, ("a", "b"))}<span class="hv">H = 0.971</span>')
TR.edge("r", "m", "בוקר")
TR.edge("r", "e", "ערב")
LEGEND = ('<div class="cls-note" style="direction:ltr"><span class="cls-dots"><i class="k-a"></i></span> Yes &nbsp;·&nbsp; '
          '<span class="cls-dots"><i class="k-b"></i></span> No</div>')
TREE = TR.html() + LEGEND

XT = w.table("xt", ["x1", "x2", "x3", "x4", "class"], [list(map(str, r)) for r in T], size=30)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נמדוד כמה צומת בעץ החלטה מעורבב: אנטרופיה."),
    P("נחשב אותה ביד בבסיס שתיים, נחשב ממוצע נכון אחרי פיצול, ונגיע לרווח המידע."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">C04 · מסווגים</div><h1>אנטרופיה ורווח מידע</h1>'
         '<p>אנטרופיה של צומת, אנטרופיה מותנית אחרי פיצול כממוצע במשקל, ו-Information Gain: כמה הערבוב ירד.</p>',
   layout="title")

idea = Scene("idea", "אנטרופיה", [
    P("עץ החלטה מפצל דוגמאות לפי מאפיין. כדי לבחור מאפיין, צריך למדוד כמה כל צומת **מעורבב**."),
    P("המדד הוא אנטרופיה. בשתי מחלקות, $p$ הוא החלק של אחת מהן, ואנטרופיה היא הגובה של העקומה הזו.",
      say="המדד הוא אנטרופיה. בשתי מחלקות, פִּי הוא החלק של אחת מהן, ואנטרופיה היא הגובה של העקומה הזו."),
    P("צומת **טהור**, כל הדוגמאות מאותה מחלקה: אנטרופיה אפס. אין שום ספק.", do=show("cu-p0", "i-1")),
    P("חצי-חצי: אנטרופיה אחת, הערבוב הכי גדול שאפשר בשתי מחלקות.", do=show("cu-p5", "i-2")),
    P("ובאמצע, למשל שלוש מתוך שמונה: משהו בין אפס לאחת. עכשיו נחשב אותו בדיוק.", do=show("cu-p38", "i-3"), hold=0.4),
], stage=f'<div class="cls-row">{CPLOT}</div>',
   panel=w.eq("i-f", r"$H = -\sum_j p_j \log_2 p_j$", 42, hidden=False)
         + st.card("i-1", "ex", "טהור ← $H = 0$")
         + st.card("i-2", "ex", "חצי-חצי ← $H = 1$")
         + st.card("i-3", "ex", "$3$ מתוך $8$ ← בין $0$ ל-$1$"))

calc = Scene("calc", "חישוב ביד", [
    P("בצומת שלוש דוגמאות Yes וחמש No. החלקים: שלוש שמיניות וחמש שמיניות.", do=show("c-1"),
      say="בצומת שלוש דוגמאות יֶס וחמש נוֹ. החלקים: שלוש שמיניות וחמש שמיניות."),
    P("לוג בבסיס שתיים של שלוש שמיניות הוא בערך מינוס 1.415. כפול שלוש שמיניות: מינוס 0.531.", do=show("c-2"),
      say="לוג בבסיס שתיים של שלוש שמיניות הוא בערך מינוס 1.415. כפול שלוש שמיניות: מינוס 0.531."),
    P("לוג של חמש שמיניות הוא בערך מינוס 0.678. כפול חמש שמיניות: מינוס 0.424.", do=show("c-3"),
      say="לוג של חמש שמיניות הוא בערך מינוס 0.678. כפול חמש שמיניות: מינוס 0.424."),
    P("מחברים ומוסיפים את המינוס שלפני הסכום: האנטרופיה בערך 0.954.", do=show("c-4"),
      say="מחברים ומוסיפים את המינוס שלפני הסכום: האנטרופיה בערך 0.954."),
    P("למה יש מינוס בנוסחה? כי לוג של מספר קטן מאחת הוא שלילי. המינוס הופך את הכל לחיובי.", do=show("c-r")),
    P("אז אנטרופיה אף פעם לא שלילית. תשובה עם מינוס היא מי ששכח אותו."),
    P("ואיך מחשבים לוג בבסיס שתיים במחשבון? לוג של המספר, חלקי לוג של שתיים, באותו בסיס למעלה ולמטה.", do=show("c-5"),
      hold=0.3),
    P("ואם האפשרויות כותבות בסיס עשר, מחשבים בבסיס עשר. כאן זה היה יוצא 0.287 במקום 0.954.", do=show("c-t"),
      say="ואם האפשרויות כותבות בסיס עשר, מחשבים בבסיס עשר. כאן זה היה יוצא 0.287 במקום 0.954.", hold=0.4),
], stage=w.col(
    w.eq("c-1", r"$p_{\text{Yes}} = \tfrac{3}{8},\qquad p_{\text{No}} = \tfrac{5}{8}$", 40),
    w.eq("c-2", r"$\tfrac{3}{8}\log_2\tfrac{3}{8} \approx \tfrac{3}{8}\cdot(-1.415) \approx -0.531$", 38),
    w.eq("c-3", r"$\tfrac{5}{8}\log_2\tfrac{5}{8} \approx \tfrac{5}{8}\cdot(-0.678) \approx -0.424$", 38),
    w.eq("c-4", r"$H = -(-0.531 - 0.424) \approx 0.954$", 42),
    w.eq("c-5", r"$\log_2 x = \dfrac{\log x}{\log 2}$", 38), gap=24),
   panel=st.card("c-r", "rule", "🔑 אנטרופיה אף פעם לא שלילית")
         + st.card("c-t", "trap", "⚠️ הבסיס כתוב באפשרויות: בבסיס $10$ יוצא $0.287$"))

cond = Scene("cond", "אנטרופיה מותנית", [
    P("עכשיו נפצל את אותן שמונה שורות לפי המאפיין ״שעה״: בוקר או ערב.", do=show("tr-e-r-m", "tr-e-r-e")),
    P("בבוקר שלוש שורות, כולן No. צומת טהור, אנטרופיה אפס.", do=show("tr-n-m"),
      say="בבוקר שלוש שורות, כולן נוֹ. צומת טהור, אנטרופיה אפס."),
    P("בערב חמש שורות: שלוש Yes ושתיים No. האנטרופיה שם בערך 0.971.", do=show("tr-n-e"),
      say="בערב חמש שורות: שלוש יֶס ושתיים נוֹ. האנטרופיה שם בערך 0.971."),
    P("האנטרופיה המותנית היא ממוצע **במשקל**: כל ענף שוקל לפי החלק של השורות שהגיעו אליו.", do=show("d-1")),
    P("שלוש שמיניות כפול אפס, ועוד חמש שמיניות כפול 0.971. יוצא בערך 0.607.", do=show("d-2"),
      say="שלוש שמיניות כפול אפס, ועוד חמש שמיניות כפול 0.971. יוצא בערך 0.607."),
    P("ורווח המידע הוא כמה הערבוב ירד: 0.954 לפני, פחות 0.607 אחרי. בערך 0.35.", do=show("d-3"),
      say="ורווח המידע הוא כמה הערבוב ירד: 0.954 לפני, פחות 0.607 אחרי. בערך 0.35."),
    P("המלכודת: לחשב ממוצע רגיל של אפס ו-0.971. יוצא 0.485, כאילו לענף של שלוש שורות יש אותו משקל כמו לענף של חמש.",
      do=show("d-t"),
      say="המלכודת: לחשב ממוצע רגיל של אפס ושל אפס נקודה תשע שבע אחת. יוצא אפס נקודה ארבע שמונה חמש, כאילו לענף של שלוש שורות יש אותו משקל כמו לענף של חמש.",
      hold=0.4),
], stage=w.col(TREE, w.eq("d-2", r"$H(Y|F) = \tfrac{3}{8}\cdot 0 + \tfrac{5}{8}\cdot 0.971 \approx 0.607$", 36), gap=20),
   panel=st.card("d-1", "rule", "🔑 $H(Y|F)$ = ממוצע במשקל גודל הענף")
         + w.eq("d-3", r"$IG = H(Y) - H(Y|F) \approx 0.35$", 36)
         + st.card("d-t", "trap", "⚠️ בלי משקלים: $0.485$, טעות"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: ארבע שורות, ארבעה מאפיינים ומחלקה. מהו רווח המידע של המחלקה בהינתן איקס שתיים, ומה המשמעות?",
      do=show("xt")),
    P("לפני הפיצול: שני Pos ושני Neg. חצי-חצי, אנטרופיה אחת.", do=show("x-1"),
      say="לפני הפיצול: שני פּוֹז ושני נֶג. חצי-חצי, אנטרופיה אחת."),
    P("עכשיו מפצלים לפי איקס שתיים. כשאיקס שתיים שווה אחת, שתי השורות הן Pos. צומת טהור.",
      do=show(*w.cells("xt", "h", (0, 1), (1, 4))),
      say="עכשיו מפצלים לפי איקס שתיים. כשאיקס שתיים שווה אחת, שתי השורות הן פּוֹז. צומת טהור."),
    P("וכשאיקס שתיים שווה אפס, שתי השורות הן Neg. גם טהור.", do=show(*w.cells("xt", "h", (2, 3), (1, 4))),
      say="וכשאיקס שתיים שווה אפס, שתי השורות הן נֶג. גם טהור."),
    P("שני הענפים באנטרופיה אפס, אז גם הממוצע במשקל אפס. רווח המידע: אחת פחות אפס, כלומר אחת.", do=show("x-2", "q-ok-3")),
    P("כל הערבוב נעלם: איקס שתיים מספר לנו את המחלקה בדיוק, ולכן נשאיר אותו."),
    P("שתי האפשרויות עם 0.5 נופלות על המספר: חישבנו אחת, לא חצי.", do=show("q-x-0", "q-x-2", "q-n-0", "q-n-2"),
      say="שתי האפשרויות עם חצי נופלות על המספר: חישבנו אחת, לא חצי."),
    P("והאפשרות עם אפס נופלת גם היא על המספר: חישבנו אחת, לא אפס.",
      do=show("q-x-1", "q-n-1"), hold=0.6),
], stage=w.col(XT, w.eq("x-1", r"$H(\text{class}) = H(\tfrac12, \tfrac12) = 1$", 38),
               w.eq("x-2", r"$IG = 1 - \left(\tfrac{2}{4}\cdot 0 + \tfrac{2}{4}\cdot 0\right) = 1$", 38), gap=26),
   panel=w.stem(Q["id"], "מהו $IG(\\text{class} \\mid x_2)$ (בסיס 2), ומה המשמעות שלו?")
         + w.opts("q", OPTS, notes={0: "אין $0.5$", 1: "זה $1$", 2: "אין $0.5$"}, size=24))

recap = Scene("recap", "סיכום", [
    P("לסיכום: אנטרופיה מודדת ערבוב. טהור זה אפס, חצי-חצי זה אחת.", do=show("r-1")),
    P("מחשבים מינוס סכום של $p$ כפול לוג בבסיס שתיים, וזה אף פעם לא שלילי.", do=show("r-2"),
      say="מחשבים מינוס סכום של פִּי כפול לוג בבסיס שתיים, וזה אף פעם לא שלילי."),
    P("אחרי פיצול: אנטרופיה לכל ענף, ממוצע במשקל גודל הענף.", do=show("r-3")),
    P("ורווח המידע הוא האנטרופיה לפני, פחות האנטרופיה המותנית. ותציצו בבסיס שבאפשרויות. נתראה ברמה הבאה.",
      do=show("r-4"), hold=0.6),
], stage=w.col(w.eq("r-eq", r"$IG = H(Y) - \sum_v \dfrac{n_v}{n}\, H(Y \mid F{=}v)$", 54, hidden=False), gap=30),
   panel=st.card("r-1", "rule", "🔑 טהור $0$ · חצי-חצי $1$")
         + st.card("r-2", "rule", "🔑 $H = -\\sum p\\log_2 p \\ge 0$")
         + st.card("r-3", "rule", "🔑 משקל לכל ענף: $n_v / n$")
         + st.card("r-4", "trap", "⚠️ בסיס הלוג: לפי האפשרויות"))

VIDEO = Video("C04", "cls", "אנטרופיה ורווח מידע", [title, idea, calc, cond, exam, recap],
              bank={"PRAC-X-Q10": 3})
