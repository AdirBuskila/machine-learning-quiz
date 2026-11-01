# -*- coding: utf-8 -*-
"""R06 · קווריאנס ומתאם פירסון — video script (build: tools/video/vid.py R06).

Source: courses/reg/R06.js (goal: compute covariance with n − 1, Pearson's r, and decide correctly whether to filter a
feature against the threshold the question defines). Exam anchor: 22B-C-Q20 (which methods can serve feature selection:
all three, the first being "filter one of two features whose Pearson r is too high"), used by no guided quiz.
Not used on purpose: 24B-C-Q2 (its r = 123.36/√(8.68·2608.053) ≈ 0.8199 makes two options true: "> 0.81 -> filter one"
and "> 0.83 -> filter none"), PRAC-X-Q20 (same four vectors as the quiz ref PRAC-X-Q3, so its covariance would answer
it), SAMP-1-Q16 (keys "attr2" by comparing a correlation with a covariance). Avoided: every R06 quiz item (the
2/4/6–1/3/8 table, the 4/12/64 and 9/30/144 matrices, r = 0.9 with threshold 0.8, the four vectors (10,12)…(40,44)),
the boss matrix 25/18/16 and the quiz refs 24S-A-Q3, 24S-C-Q3. R06-Q4 is not answered by the video: no sentence
"linear regression is affected by dependent features", "filter one, never both", and the threshold scene ends at
"dependency high / not high / low", without a filtering decision. Own data: X₁ = (1, 2, 3, 4, 5), X₂ = (2, 1, 4, 3, 5):
cov = 8/4 = 2, var₁ = var₂ = 2.5, r = 0.8 (numpy's cov / corrcoef agree); my own thresholds 0.75 / 0.85 / 0.82.
"""
from fractions import Fraction as Fr
import math

import numpy as np

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
X1, X2 = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
N = len(X1)
M1, M2 = Fr(sum(X1), N), Fr(sum(X2), N)
D1 = [x - M1 for x in X1]
D2 = [x - M2 for x in X2]
PR = [a * b for a, b in zip(D1, D2)]
assert (M1, M2) == (3, 3) and D1 == [-2, -1, 0, 1, 2] and D2 == [-1, -2, 1, 0, 2] and PR == [2, 2, 0, 0, 4]
COV = Fr(sum(PR), N - 1)
VAR1 = Fr(sum(d * d for d in D1), N - 1)
VAR2 = Fr(sum(d * d for d in D2), N - 1)
assert sum(PR) == 8 and COV == 2 and Fr(sum(PR), N) == Fr(8, 5) and VAR1 == VAR2 == Fr(5, 2)
R = COV / Fr(5, 2)                                     # √(2.5·2.5) = 2.5
assert R == Fr(4, 5) and math.isclose(float(COV) / math.sqrt(float(VAR1 * VAR2)), 0.8)
assert np.isclose(np.cov(X1, X2)[0, 1], 2) and np.isclose(np.corrcoef(X1, X2)[0, 1], 0.8)
# my thresholds: high if r > 0.75 -> filter one; high if r > 0.85 -> not high -> keep both; low if r < 0.82 -> low
TH = [("high", Fr(75, 100)), ("high", Fr(85, 100)), ("low", Fr(82, 100))]
dep_high = [R > t if kind == "high" else not (R < t) for kind, t in TH]
assert dep_high == [True, False, False]
# duplicate features: x₂ = x₁ makes many weight pairs give the same predictions (3·x₁ = 1·x₁ + 2·x₂ = 5·x₁ − 2·x₂)
assert all(3 * x == 1 * x + 2 * x == 5 * x - 2 * x for x in X1)

# exam 22B-C-Q20
Q = BANK["22B-C-Q20"]
ALL = w.opt_index(Q, "כל התשובות")
PEA = w.opt_index(Q, "פירסון")
MI = w.opt_index(Q, "mutual information")
VAR = w.opt_index(Q, "השונות")
assert Q["correctIndex"] == ALL == 3 and sorted([ALL, PEA, MI, VAR]) == [0, 1, 2, 3]
OPTS = w.opts(Q)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נמדוד אם שני מאפיינים מספרים את אותו הסיפור."),
    P("נחשב קווריאנס, נחשב את מקדם המתאם של פירסון, ונחליט אם לסנן מאפיין."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">R06 · רגרסיה ורשתות נוירונים</div><h1>קווריאנס ומתאם פירסון</h1>'
         '<p>לחשב קווריאנס (חלוקה ב-<span class="rg-en">n − 1</span>) ומקדם מתאם, ולהחליט נכון אם לסנן מאפיין.</p>',
   layout="title")

why = Scene("why", "למה לסנן", [
    P("שטח של דירה ומספר החדרים בה: שני מאפיינים, אבל כמעט אותו מידע. כשאחד גדל, גם השני."),
    P("למה זה מפריע לרגרסיה לינארית? נניח שהמאפיין השני הוא בדיוק עותק של הראשון.", do=show("y-0")),
    P("אז 3 כפול $x_1$, או $x_1$ ועוד 2 כפול $x_2$, או 5 כפול $x_1$ פחות 2 כפול $x_2$: כולם נותנים אותו חיזוי בדיוק.",
      do=show("y-1", "y-2", "y-3"),
      say="אז שלוש כפול אִקְס אחת, או אִקְס אחת ועוד שתיים כפול אִקְס שתיים, או חמש כפול אִקְס אחת פחות שתיים כפול אִקְס שתיים: "
          "כולם נותנים אותו חיזוי בדיוק."),
    P("המשקולות כבר לא אומרות כלום על כל מאפיין: אין דרך לדעת למי מהשניים לייחס את ההשפעה.", do=show("y-k")),
    P("לכן מחפשים זוגות כאלה עוד לפני האימון. איך מודדים כמה שני מאפיינים זזים יחד? זה הנושא של הסרטון.", do=show("y-t"),
      hold=0.3),
], stage=w.stack(w.eq("y-0", "$x_2 = x_1$", size=46),
                 w.eq("y-1", "$\\hat{y} = 3x_1$", size=44), w.eq("y-2", "$\\hat{y} = x_1 + 2x_2$", size=44),
                 w.eq("y-3", "$\\hat{y} = 5x_1 - 2x_2$", size=44), gap=26),
   panel=w.card("y-k", "rule", "🔑 מאפיינים תלויים ← משקולות לא יציבות")
         + w.card("y-t", "ex", "בודקים תלות בין מאפיינים לפני האימון"))

SC = w.Plot("sc", (0, 6), (0, 6), w=560, h=520, xticks=[1, 2, 3, 4, 5], yticks=[1, 2, 3, 4, 5], xlabel="X₁", ylabel="X₂",
            pad=(60, 20, 26, 56))
SC.rect(3, 3, 6, 6, kind="k-ok", id_="q1", hidden=True)
SC.rect(0, 0, 3, 3, kind="k-ok", id_="q3", hidden=True)
SC.seg(3, 0, 3, 6, kind="k-b", id_="mx", hidden=True, width=3, dash=True)
SC.seg(0, 3, 6, 3, kind="k-b", id_="my", hidden=True, width=3, dash=True)
for i, (a, b) in enumerate(zip(X1, X2)):
    SC.dot(a, b, kind="k-ink", id_=f"p{i}", r=12)
CT = w.table("ct", ["$X_1$", "$X_2$", "$x_1 - \\overline{x}_1$", "$x_2 - \\overline{x}_2$", "מכפלה"],
             [[a, b, w.fmt(d1), w.fmt(d2), w.fmt(p)] for a, b, d1, d2, p in zip(X1, X2, D1, D2, PR)]
             + [["", "", "", "סכום", "8"]], hidden={(None, 2), (None, 3), (None, 4)}, size=25, cls="sum")
cov = Scene("cov", "קווריאנס", [
    P("**קווריאנס** מודד אם שני מאפיינים זזים יחד. הנה חמש דוגמאות.", do=show("ct-m0")),
    P("הממוצע של שני המאפיינים הוא 3. מכל ערך מחסירים את הממוצע שלו.",
      do=hide("ct-m0") + show("sc-mx", "sc-my", *[f"ct-{r}-{c}" for r in range(5) for c in (2, 3)]),
      say="הממוצע של שני המאפיינים הוא שלוש. מכל ערך מחסירים את הממוצע שלו."),
    P("ומכפילים את שני ההפרשים. אם שניהם מעל הממוצע, או שניהם מתחתיו, המכפלה חיובית.",
      do=show("sc-q1", "sc-q3", *[f"ct-{r}-4" for r in range(5)])),
    P("המכפלות: 2, 2, 0, 0, 4. הסכום: 8.", do=show("ct-5-3", "ct-5-4"),
      say="המכפלות: שתיים, שתיים, אפס, אפס, ארבע. הסכום: שמונה."),
    P("ומחלקים ב-$n$ פחות 1, כלומר ב-4. הקווריאנס: 2.", do=show("v-1"),
      say="ומחלקים באֶן פחות אחת, כלומר בארבע. הקווריאנס: שתיים.", hold=0.3),
    P("חיובי: עולים יחד. שלילי: כשאחד עולה, השני יורד.", do=show("v-2")),
    P("חלוקה ב-5 במקום ב-4 נותנת 1.6, ועצירה בסכום נותנת 8. שני שלבי ביניים, ולא התשובה.", do=show("v-t"),
      say="חלוקה בחמש במקום בארבע נותנת אחת נקודה שש, ועצירה בסכום נותנת שמונה. שני שלבי ביניים, ולא התשובה."),
], stage=w.row(SC.svg(scale=0.72), CT, gap=18),
   panel=w.card("v-1", "rule", "🔑 $\\text{cov} = \\frac{8}{n - 1} = \\frac{8}{4} = 2$")
         + w.card("v-2", "ex", "חיובי ← עולים יחד · שלילי ← הפוכים")
         + w.card("v-t", "trap", "⚠️ $\\frac{8}{5} = 1.6$ חלוקה ב-$n$ · $8$ בלי חלוקה"))

MAT = w.table("mx", ["", "$X_1$", "$X_2$"], [["$X_1$", "2.5", "2"], ["$X_2$", "2", "2.5"]], size=40, hl_rows=False)
pear = Scene("pear", "מקדם המתאם", [
    P("הקווריאנס תלוי ביחידות: שטח במטרים או בסנטימטרים ייתן מספרים שונים. לכן מנרמלים."),
    P("**השונות** של מאפיין היא הקווריאנס שלו עם עצמו. כאן, לשני המאפיינים, 2.5.", do=show("e-v"),
      say="השונות של מאפיין היא הקווריאנס שלו עם עצמו. כאן, לשני המאפיינים, שתיים נקודה חמש."),
    P("ב**מטריצת קווריאנס** השונויות על האלכסון, והקווריאנס מחוץ לו.", do=show("e-m")),
    P("מקדם המתאם של פירסון: הקווריאנס, חלקי השורש של מכפלת השונויות.", do=show("e-f")),
    P("2, חלקי שורש של 2.5 כפול 2.5. השורש הוא 2.5, ולכן $r$ שווה 0.8.", do=show("e-r"),
      say="שתיים, חלקי שורש של שתיים נקודה חמש כפול שתיים נקודה חמש. השורש הוא שתיים נקודה חמש, ולכן אָר שווה אפס נקודה שמונה.",
      hold=0.3),
    P("$r$ תמיד בין מינוס 1 ל-1. קרוב ל-1: תלות חיובית חזקה.", do=show("e-k"),
      say="אָר תמיד בין מינוס אחת לאחת. קרוב לאחת: תלות חיובית חזקה."),
], stage=w.stack(MAT, w.eq("e-f", "$r = \\dfrac{\\text{cov}(X_1, X_2)}{\\sqrt{\\text{var}_1 \\cdot \\text{var}_2}}$", size=46),
                 w.eq("e-r", "$r = \\dfrac{2}{\\sqrt{2.5 \\cdot 2.5}} = \\dfrac{2}{2.5} = 0.8$", size=44), gap=30),
   panel=w.card("e-v", "ex", "$\\text{var}(X) = \\text{cov}(X, X)$")
         + w.card("e-m", "ex", "אלכסון: שונויות · מחוץ לאלכסון: קווריאנס")
         + w.card("e-k", "rule", "🔑 $-1 \\le r \\le 1$"))

dec = Scene("dec", "להחליט אם לסנן", [
    P("עכשיו ההחלטה. את $r$ מחשבים פעם אחת, אבל הסף מגיע מהגדרה, וכל שאלה מגדירה אותו אחרת.",
      say="עכשיו ההחלטה. את אָר מחשבים פעם אחת, אבל הסף מגיע מהגדרה, וכל שאלה מגדירה אותו אחרת."),
    P("אם תלות גבוהה מוגדרת מעל 0.75: 0.8 מעליו, והתלות גבוהה.", do=show("d-0"),
      say="אם תלות גבוהה מוגדרת מעל אפס נקודה שבעים וחמש: אפס נקודה שמונה מעליו, והתלות גבוהה."),
    P("אם תלות גבוהה מוגדרת מעל 0.85: 0.8 לא מגיע, והתלות לא נחשבת גבוהה.", do=show("d-1"),
      say="אם תלות גבוהה מוגדרת מעל אפס נקודה שמונים וחמש: אפס נקודה שמונה לא מגיע, והתלות לא נחשבת גבוהה."),
    P("ואם תלות **נמוכה** מוגדרת מתחת ל-0.82: 0.8 מתחת, והתלות נמוכה.", do=show("d-2"),
      say="ואם תלות נמוכה מוגדרת מתחת לאפס נקודה שמונים ושתיים: אפס נקודה שמונה מתחת, והתלות נמוכה.", hold=0.3),
    P("אותו $r$, שלוש הגדרות, ושתי מסקנות שונות. קוראים בכל פעם את הסף המדויק, ואת כיוון ההשוואה.", do=show("d-k"),
      say="אותו אָר, שלוש הגדרות, ושתי מסקנות שונות. קוראים בכל פעם את הסף המדויק, ואת כיוון ההשוואה."),
], stage=w.stack(*[f'<div id="d-{i}" class="h rg-case"><span class="rg-cq">{t}</span><span class="rg-chip {k}">{v}</span></div>'
                   for i, t, v, k in [(0, "גבוהה אם $r > 0.75$", "גבוהה", "warn"),
                                      (1, "גבוהה אם $r > 0.85$", "לא גבוהה", "ok"),
                                      (2, "נמוכה אם $r < 0.82$", "נמוכה", "ok")]],
                 w.eq("d-r", "$r = 0.8$", size=50, hidden=False), gap=22),
   panel=w.card("d-k", "rule", "🔑 $r$ פעם אחת, ואז כל הגדרה מול הסף שלה"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו שיטה יכולה לשמש לבחירת מאפיינים?",
      say="שאלה ממבחן: איזו שיטה יכולה לשמש לבחירת מאפיינים, פִיצֶ'ר סֶלֶקְשֶׁן?"),
    P("הראשונה: לסנן אחד משני מאפיינים שמקדם פירסון ביניהם גבוה מדי. ראינו למה זוג כזה מבלבל את המשקולות. נכונה.",
      do=show(f"q-n-{PEA}")),
    P("השנייה: להשאיר מאפיין שיש לו מידע משותף גבוה עם המחלקה. מאפיין שקשור חזק למה שחוזים, שווה לשמור. נכונה.",
      do=show(f"q-n-{MI}")),
    P("השלישית: לסנן מאפיין שהשונות שלו כמעט אפס. מאפיין כמעט קבוע לא מבדיל בין דוגמאות. גם נכונה.",
      do=show(f"q-n-{VAR}"), hold=0.3),
    P("שלוש השיטות תקפות, ולכן התשובה: כל התשובות נכונות.", do=show(f"q-ok-{ALL}")),
    P("שימו לב להבדל: קשר חזק **בין שני מאפיינים** הוא כפילות. קשר חזק **בין מאפיין למחלקה** הוא בדיוק המידע שרוצים.",
      do=show("x-k"), hold=0.5),
], stage=w.qpanel(Q["id"], Q["question"], OPTS, notes={PEA: "נכון", MI: "נכון", VAR: "נכון"}, size=30),
   panel=w.card("x-k", "rule", "🔑 מאפיין↔מאפיין: כפילות<br>מאפיין↔מחלקה: מידע שימושי"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: קווריאנס הוא סכום מכפלות ההפרשים מהממוצע, חלקי $n$ פחות 1.", do=show("r-1"),
      say="לסיכום: קווריאנס הוא סכום מכפלות ההפרשים מהממוצע, חלקי אֶן פחות אחת."),
    P("מקדם המתאם: הקווריאנס חלקי שורש מכפלת השונויות. הוא תמיד בין מינוס 1 ל-1.", do=show("r-2"),
      say="מקדם המתאם: הקווריאנס חלקי שורש מכפלת השונויות. הוא תמיד בין מינוס אחת לאחת."),
    P("ובהחלטה: כל הגדרה מול הסף שלה, ובכיוון ההשוואה שלה. נתראה ברמה הבאה.", do=show("r-3"), hold=0.6),
], stage=w.eq("r-eq", "$r = \\dfrac{\\sum (x_1 - \\overline{x}_1)(x_2 - \\overline{x}_2) / (n - 1)}{\\sqrt{\\text{var}_1 \\cdot \\text{var}_2}}$",
              size=44, hidden=False, cls="box"),
   panel=w.card("r-1", "rule", "🔑 קווריאנס: חלוקה ב-$n - 1$")
         + w.card("r-2", "rule", "🔑 $r = \\text{cov} / \\sqrt{\\text{var}_1 \\cdot \\text{var}_2}$")
         + w.card("r-3", "rule", "🔑 כל הגדרה: הסף שלה והכיוון שלה"))

VIDEO = Video("R06", "reg", "קווריאנס ומתאם פירסון", [title, why, cov, pear, dec, exam, recap], bank={"22B-C-Q20": ALL})
w.check_say(VIDEO)
