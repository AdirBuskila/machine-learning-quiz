# -*- coding: utf-8 -*-
"""E06 · איזה מדד מתאים? — video script (build: tools/video/vid.py E06).

Source: courses/eval/E06.js (goal: match a metric to the problem type, compute MAE, know when low / high is good,
and see when accuracy misleads). Exam anchor: 25S-A-Q13 (why a confusion matrix fits KNN but not K-Means), which no
guided quiz uses and has no twin among the quiz items.
Not used on purpose: the E06 quiz scenarios (MAE of 20/30/40 vs 22/27/40, the four regression students, 1000
transactions with 10 frauds, "which metric fits clustering", "which is not a regression metric", "why F1") and the
refs 25B-A-Q20, 23S-A-Q18, 25B-A-Q3, plus their twins (26S-A-Q8, 24B-A-Q16, SAMP-1-Q22, CLAUDE-26-Q10 — the
"who built the better model" family; 25S-A-Q3 — the F1 question; 26B-A-Q14 and friends — "not a regression metric").
Worked examples: delivery times 12/15/9/20 vs predictions 10/16/9/23 (MAE 1.5, MSE 3.5, RMSE ≈ 1.87; signed mean
-0.5), and 300 sky photos with 6 meteors (a lazy "no meteor" model vs a real one). All asserted, sklearn-checked.
"""
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

Fr = w.Fr
# ---------------------------------------------------------------- regression metrics (verified)
Y = np.array([12, 15, 9, 20])
YH = np.array([10, 16, 9, 23])
E = Y - YH
assert E.tolist() == [2, -1, 0, -3] and E.mean() == -0.5                     # signed errors cancel
assert mean_absolute_error(Y, YH) == 1.5 and mean_squared_error(Y, YH) == 3.5
RMSE = math.sqrt(3.5)
assert f"{RMSE:.2f}" == "1.87" and RMSE >= 1.5

# ---------------------------------------------------------------- rare class (verified)
LAZY = w.metrics(1, 5, 0, 294)
REAL = w.metrics(5, 1, 10, 284)
for X in (LAZY, REAL):
    w.sk_check(X)
    assert X["n"] == 300 and X["tp"] + X["fn"] == 6
assert LAZY["acc"] == Fr(59, 60) and LAZY["rec"] == Fr(1, 6) and w.fr_pct(LAZY["acc"], 1) == "98.3"   # finds 1 of 6
assert w.fr_dec(LAZY["f1"]) == "0.29" and w.fr_pct(LAZY["rec"], 1) == "16.7"
assert w.fr_pct(REAL["acc"], 1) == "96.3" and REAL["acc"] < LAZY["acc"]
assert w.fr_pct(REAL["rec"], 1) == "83.3" and REAL["prec"] == Fr(1, 3) and w.fr_dec(REAL["f1"]) == "0.48"

# ---------------------------------------------------------------- exam 25S-A-Q13
Q = BANK["25S-A-Q13"]
assert Q["correctIndex"] == 0 and "K-Means" in Q["question"] and "KNN" in Q["question"]
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
L = lambda t: f"<span class='ev-l'>{t}</span>"
KINDS = w.table("mk", ["סוג הבעיה", "מה משווים", "מדדים"],
                [["<b>סיווג</b>", "מחלקה חזויה מול מחלקה אמיתית", L("Accuracy, Precision, Recall, F1")],
                 ["<b>רגרסיה</b>", "מספר חזוי מול מספר אמיתי", L("MAE, MSE, RMSE")],
                 ["<b>אשכול</b>", "אין תוויות לחזות", "מדדי אשכול משלו (נלמד בהמשך)"]], size=27)
REG = w.table("rg", ["אמת $y$", "חיזוי $\\hat{y}$", "$y - \\hat{y}$", "$|y - \\hat{y}|$", "$(y - \\hat{y})^2$"],
              [[L(v) for v in (a, b, a - b, abs(a - b), (a - b) ** 2)] for a, b in zip(Y, YH)], size=28)
CL = w.cmatrix("cl", 1, 5, 0, 294, pos="מטאור", neg="ריק", size=24, tags=False)
CR = w.cmatrix("cr", 5, 1, 10, 284, pos="מטאור", neg="ריק", size=24, tags=False)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבחר את המדד הנכון לכל סוג בעיה, ונחשב את מדדי השגיאה של רגרסיה."),
    P("ונראה מתי דיוק גבוה הוא בעצם מלכודת.", hold=0.4),
], stage='<div class="kicker">E06 · הערכת מודל</div><h1>איזה מדד מתאים?</h1>'
         '<p>סיווג, רגרסיה או אשכול: לכל אחד המדדים שלו. מתי נמוך טוב, מתי גבוה טוב, ומתי accuracy מטעה.</p>',
   layout="title")

kinds = Scene("kinds", "מדד לכל סוג", [
    P("מדד בודק את המודל מול האמת. לכן הוא תלוי במה שהמודל מוציא.", do=show("mk")),
    P("מסווג מוציא **מחלקה**. אפשר לספור פגיעות וטעויות: מטריצת בלבול, דיוק, Precision, Recall ו-F1.",
      do=show("mk-r0"),
      say="מסווג מוציא מחלקה. אפשר לספור פגיעות וטעויות: מטריצת בלבול, דיוק, פְּרֶסִיזְ'ן, רִיקוֹל ואֶף וַואן."),
    P("מודל רגרסיה מוציא **מספר**. כמעט אף פעם הוא לא יפגע בדיוק, אז מודדים **כמה רחוק** הוא היה.",
      do=show("mk-r1")),
    P("ובאשכול אין בכלל תוויות לחזות. יש לו מדדים משלו, שבודקים כמה כל קבוצה אחידה.", do=show("mk-r2")),
    P("לכן מדד של סוג אחד לא מתאים לסוג אחר. זה הבסיס של כל שאלה בנושא.", do=show("k-k"), hold=0.4),
], stage=w.col(KINDS),
   panel=st.card("k-k", "rule", "🔑 קודם מזהים את <b>סוג הבעיה</b>, ורק אז בוחרים מדד"))

reg = Scene("reg", "מדדי רגרסיה", [
    P("מודל חוזה זמני משלוח בדקות. ארבעה משלוחים: האמת, והחיזוי.", do=show("rg", "rg-r0", "rg-r1", "rg-r2", "rg-r3")),
    P("השגיאות: שתיים, מינוס אחת, אפס, ומינוס שלוש. מה שמעניין אותנו הוא **גודל** הטעות, לא הכיוון שלה."),
    P("לכן MAE לוקח **ערך מוחלט**: שתיים, אחת, אפס ושלוש. הממוצע: אחת וחצי דקות.", do=show("g-2"),
      say="לכן אֶם אֵיי אִי לוקח ערך מוחלט: שתיים, אחת, אפס ושלוש. הממוצע: אחת וחצי דקות."),
    P("MSE מעלה בריבוע: ארבע, אחת, אפס ותשע. הממוצע שלוש וחצי.", do=show("g-3"),
      say="אֶם אֶס אִי מעלה בריבוע: ארבע, אחת, אפס ותשע. הממוצע שלוש וחצי."),
    P("ו-RMSE הוא השורש שלו: בערך 1.87. חזרנו ליחידות של דקות, אבל הטעות הגדולה קיבלה משקל רב יותר.",
      do=show("g-4"),
      say="ואָר אֶם אֶס אִי הוא השורש שלו: בערך אחת נקודה שמונים ושבע. חזרנו ליחידות של דקות, אבל הטעות הגדולה קיבלה משקל רב יותר."),
    P("שלושתם מודדים **טעות**: נמוך יותר זה טוב יותר. ו-MAE לעולם לא שלילי.", do=show("g-k"),
      say="שלושתם מודדים טעות: נמוך יותר זה טוב יותר. ואֶם אֵיי אִי לעולם לא שלילי."),
    P("R בריבוע הפוך: הוא מודד כמה מהשונות המודל מסביר, ושם **גבוה** יותר זה טוב יותר.", do=show("g-r"),
      hold=0.4, say="אָר בריבוע הפוך: הוא מודד כמה מהשונות המודל מסביר, ושם גבוה יותר זה טוב יותר."),
], stage=w.col(REG, gap=24),
   panel=st.card("g-2", "ex", "$\\text{MAE} = \\tfrac{2 + 1 + 0 + 3}{4} = 1.5$")
         + st.card("g-3", "ex", "$\\text{MSE} = \\tfrac{4 + 1 + 0 + 9}{4} = 3.5$")
         + st.card("g-4", "ex", "$\\text{RMSE} = \\sqrt{3.5} \\approx 1.87$")
         + st.card("g-k", "rule", "🔑 <span class='ev-l'>MAE, MSE, RMSE</span>: נמוך = טוב")
         + st.card("g-r", "rule", "🔑 <span class='ev-l'>R²</span>: גבוה = טוב"))

rare = Scene("rare", "כשהדיוק מטעה", [
    P("עכשיו סיווג, עם מחלקה נדירה: 300 תמונות שמיים, ורק ב-6 מהן יש מטאור.",
      say="עכשיו סיווג, עם מחלקה נדירה: שלוש מאות תמונות שמיים, ורק בשש מהן יש מטאור."),
    P("מודל חלש כמעט תמיד אומר: אין מטאור. הוא צודק ב-295 תמונות. דיוק של בערך 98.3 אחוז!", do=show("cl", "u-1"),
      say="מודל חלש כמעט תמיד אומר: אין מטאור. הוא צודק במאתיים תשעים וחמש תמונות. דיוק של בערך תשעים ושמונה נקודה שלוש אחוז!"),
    P("אבל הוא מצא רק מטאור אחד מתוך שישה. ה-Recall שלו בערך 16.7 אחוז, וה-F1 בערך 0.29.", do=show("cl-h-fn", "u-2"),
      say="אבל הוא מצא רק מטאור אחד מתוך שישה. הרִיקוֹל שלו בערך שש עשרה נקודה שבע אחוז, והאֶף וַואן בערך אפס נקודה עשרים ותשע."),
    P("מודל אמיתי מוצא 5 מתוך 6 המטאורים, במחיר 10 אזעקות שווא. הדיוק שלו נמוך יותר: בערך 96.3 אחוז.",
      do=show("cr", "u-3"),
      say="מודל אמיתי מוצא חמישה מתוך שישה המטאורים, במחיר עשר אזעקות שווא. הדיוק שלו נמוך יותר: בערך תשעים ושש נקודה שלוש אחוז."),
    P("אבל ה-Recall שלו 83 אחוז, וה-F1 בערך 0.48. לפי הדיוק, החלש ״מנצח״. לפי המחלקה החשובה, ההפך.",
      do=show("cr-h-tp", "u-4"),
      say="אבל הרִיקוֹל שלו שמונים ושלושה אחוז, והאֶף וַואן בערך אפס נקודה ארבעים ושמונה. לפי הדיוק, החלש מנצח. לפי המחלקה החשובה, ההפך."),
    P("כשמחלקה אחת נדירה, הדיוק מתנפח מכל השליליים הקלים. לכן בודקים גם כמה מהמקרים הנדירים נתפסו.", do=show("u-k"),
      hold=0.4),
], stage=w.row(w.col(CR, w.chip("u-3", "אמיתי: דיוק ≈ 96.3%", "ok"), w.chip("u-4", "<span class='ev-l'>Recall ≈ 83% · F1 ≈ 0.48</span>", "ok"), gap=10),
               w.col(CL, w.chip("u-1", "חלש: דיוק ≈ 98.3%", "warn"), w.chip("u-2", "<span class='ev-l'>Recall ≈ 17% · F1 ≈ 0.29</span>", "bad"), gap=10),
               gap=24),
   panel=st.card("u-k", "rule", "🔑 בעשרות ההחלטות הקלות הדיוק לא רואה את המקרים הנדירים: סופרים גם כמה מהם נתפסו"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: למה מטריצת בלבול לא רלוונטית ל-K-Means, אבל חיונית ל-KNN?",
      say="שאלה ממבחן: למה מטריצת בלבול לא רלוונטית לקֵיי מִינְס, אבל חיונית לקֵיי אֶן אֶן?"),
    P("מטריצת בלבול שמה מחלקה אמיתית מול מחלקה חזויה. היא קיימת רק כשהמודל **חוזה מחלקה**.", do=show("x-1")),
    P("KNN הוא מסווג: לכל דוגמה הוא מחזיר מחלקה. K-Means הוא אשכול: הוא מחלק לקבוצות, בלי לחזות תווית.",
      do=show("x-2", "q-ok-0"),
      say="קֵיי אֶן אֶן הוא מסווג: לכל דוגמה הוא מחזיר מחלקה. קֵיי מִינְס הוא אשכול: הוא מחלק לקבוצות, בלי לחזות תווית."),
    P("״רק לאלגוריתמים לינאריים״? KNN עצמו לא לינארי, ומטריצת בלבול מתאימה לו מצוין.", do=show("q-x-1", "q-n-1"),
      say="רק לאלגוריתמים לינאריים? קֵיי אֶן אֶן עצמו לא לינארי, ומטריצת בלבול מתאימה לו מצוין."),
    P("״נתונים מספריים מול בדידים״? שניהם עובדים על מרחקים בין מאפיינים מספריים. וזה בכלל לא קשור למדד.",
      do=show("q-x-2", "q-n-2")),
    P("ו״רק לרגרסיה״ הפוך לגמרי: ברגרסיה החיזוי הוא מספר, ואין מה לשים במשבצות.", do=show("q-x-3", "q-n-3"), hold=0.6),
], stage=w.col(st.card("x-1", "rule", "🔑 מטריצת בלבול = מחלקה אמיתית × מחלקה חזויה"),
               st.card("x-2", "ex", "<span class='ev-l'>KNN</span> ← מחלקה לכל דוגמה ✓<br>"
                                    "<span class='ev-l'>K-Means</span> ← קבוצות, בלי תווית ✗"), gap=26),
   panel=w.stem(Q["id"], "מדוע מטריצת בלבול אינה רלוונטית ל-<span class='ev-l'>K-Means</span>, אך חיונית ל-"
                         "<span class='ev-l'>KNN</span>?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={1: "KNN לא לינארי", 2: "לא קשור", 3: "הפוך"})
         + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: מזהים את סוג הבעיה, ובוחרים את המדדים שלה.", do=show("r-1")),
    P("ברגרסיה: MAE, MSE ו-RMSE הם טעות, ונמוך טוב. R בריבוע: גבוה טוב.", do=show("r-2"),
      say="ברגרסיה: אֶם אֵיי אִי, אֶם אֶס אִי ואָר אֶם אֶס אִי הם טעות, ונמוך טוב. אָר בריבוע: גבוה טוב."),
    P("MAE הוא ממוצע של ערכים מוחלטים, ולכן לעולם לא שלילי.", do=show("r-3"),
      say="אֶם אֵיי אִי הוא ממוצע של ערכים מוחלטים, ולכן לעולם לא שלילי."),
    P("ובמחלקה נדירה, הדיוק מטעה. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(KINDS.replace('id="mk', 'id="rk').replace(' class="h"', "")),
   panel=st.card("r-1", "rule", "🔑 סוג הבעיה קובע את המדד")
         + st.card("r-2", "rule", "🔑 שגיאה: נמוך טוב · <span class='ev-l'>R²</span>: גבוה טוב")
         + st.card("r-3", "rule", "🔑 $\\text{MAE} = \\tfrac{1}{n}\\sum |y - \\hat{y}| \\ge 0$")
         + st.card("r-4", "trap", "⚠️ מחלקה נדירה: דיוק גבוה ≠ מודל טוב"))

VIDEO = w.check_say(Video("E06", "eval", "איזה מדד מתאים?", [title, kinds, reg, rare, exam, recap],
                          bank={"25S-A-Q13": 0}))
