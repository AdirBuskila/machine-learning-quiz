# -*- coding: utf-8 -*-
"""E03 · מטריצת בלבול ו-accuracy — video script (build: tools/video/vid.py E03).

Source: courses/eval/E03.js (goal: turn a story into TP/FN/FP/TN, read a given matrix correctly, compute accuracy and
error rate). Exam anchor: PRAC-X-Q17 (a given matrix 400/500/300/100, rows = expected), which no guided quiz uses. Note: the
quiz ref SAMP-1-Q3 uses the same four numbers (400 / 300 / 100 / 500) in another arrangement and asks for precision
(0.8 there); this video's matrix and question differ (TP 400, FN 500, FP 300, TN 100; precision 4/7), and the
video never states SAMP-1-Q3's answer.
Not used on purpose: every E03 quiz scenario (fire sensor 50/44 + 15, the matrix 30/20/10/40, dogs 80/120 with 60/30,
35/15/35/15 and the coin, the medical test 400/120 with 96/40, 25S-A-Q7), the lesson's apples (200/60/48/20) and the
90/10/30/50 family (23B-A-Q10 and its ratio twins 23B-B-Q10, 23S-A-Q1, 24B-A-Q9).
Worked example: a bottle-inspection camera, 150 bottles, 30 cracked: TP 24, FN 6, FP 9, TN 111.
Every number is an exact fraction (w_eval.metrics) and is cross-checked against scikit-learn.
"""
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w
from w_eval import num_he as N, dec_he as D, pct_he as PC

# ---------------------------------------------------------------- worked example (verified)
TOT, POS, TP, FP = 150, 30, 24, 9
FN, NEG = POS - TP, TOT - POS
TN = NEG - FP
M = w.metrics(TP, FN, FP, TN)
w.sk_check(M)
assert (FN, NEG, TN) == (6, 120, 111) and TP + FN + FP + TN == TOT
assert M["acc"] == w.Fr(9, 10) and M["err"] == w.Fr(1, 10)
assert TOT - TP == 126 and 126 != TN                    # the wrong "everything minus TP"

# ---------------------------------------------------------------- exam PRAC-X-Q17 (verified)
Q = BANK["PRAC-X-Q17"]
assert Q["correctIndex"] == 2 and "400" in Q["question"] and "500" in Q["question"] and "300" in Q["question"]
X = w.metrics(400, 500, 300, 100)
w.sk_check(X)
assert X["n"] == 1300
assert w.fr_dec(X["acc"]) == "0.38" and w.fr_dec(X["err"]) == "0.62"     # the two numbers swap names in the options
assert X["tn"] == 100 and X["tp"] == 400                                   # "TN = 400" reads the TP cell
assert w.fr_dec(X["prec"]) == "0.57" and X["prec"] == w.Fr(4, 7)
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
CM = w.cmatrix("cm", TP, FN, FP, TN, pos="סדוק", neg="תקין", vals=False, hidden=True)
CM2 = w.cmatrix("ca", TP, FN, FP, TN, pos="סדוק", neg="תקין", size=30)
CMX = w.cmatrix("cx", 400, 500, 300, 100, pos="Positive", neg="Negative", size=28)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד לבנות מטריצת בלבול מתוך סיפור, ולקרוא מטריצה נתונה בלי להתבלבל."),
    P("נחשב ממנה את הדיוק ואת שיעור השגיאה, ונפתור שאלה ממבחן שבנויה כולה על המלכודות האלה.", hold=0.4),
], stage='<div class="kicker">E03 · הערכת מודל</div><h1>מטריצת בלבול ו-accuracy</h1>'
         '<p>מסיפור מילולי לארבעה מספרים: TP, FN, FP, TN. ואחר כך accuracy ו-error rate.</p>', layout="title")

four = Scene("four", "ארבע תוצאות", [
    P("מצלמה בפס ייצור בודקת בקבוקים, ומסמנת בקבוק סדוק. בדקו מאה חמישים בקבוקים, ושלושים מהם באמת סדוקים.",
      do=show("f-story")),
    P("המחלקה שאנחנו **מחפשים**, סדוק, נקראת החיובית. בקבוק תקין הוא השלילי.", do=show("cm")),
    P("בשורות שמים את **האמת**, ובעמודות את מה שהמודל **חזה**. כך זה במאגר כולו.", do=show("f-rule")),
    P("המצלמה סימנה 24 מהבקבוקים הסדוקים. סדוק שסומן כסדוק: חיובי אמיתי.", do=show("cm-v-tp", "cm-h-tp"),
      say="המצלמה סימנה עשרים וארבעה מהבקבוקים הסדוקים. סדוק שסומן כסדוק: חיובי אמיתי."),
    P("את שאר הסדוקים היא פספסה: 30 פחות 24, כלומר 6. זה שלילי כוזב.",
      do=hide("cm-h-tp") + show("cm-v-fn", "cm-h-fn"),
      say="את שאר הסדוקים היא פספסה: שלושים פחות עשרים וארבעה, כלומר שישה. זה שלילי כוזב."),
    P("היא גם סימנה בטעות 9 בקבוקים תקינים. תקין שסומן כסדוק: חיובי כוזב, אזעקת שווא.",
      do=hide("cm-h-fn") + show("cm-v-fp", "cm-h-fp"),
      say="היא גם סימנה בטעות תשעה בקבוקים תקינים. תקין שסומן כסדוק: חיובי כוזב, אזעקת שווא."),
    P("ושאר התקינים עברו כתקינים: שלילי אמיתי.", do=hide("cm-h-fp") + show("cm-h-tn")),
    P("השם בנוי משתי מילים: האות השנייה היא מה שהמודל **אמר**, והאות הראשונה היא אם הוא **צדק**.",
      do=hide("cm-h-tn") + show("f-name"), hold=0.4),
], stage=w.col(CM, w.note("150 בקבוקים · 30 מהם סדוקים", "f-story", hidden=True)),
   panel=st.card("f-rule", "rule", "🔑 שורה = <b>האמת</b> · עמודה = <b>החיזוי</b>")
         + st.card("f-name", "ex", "<span class='ev-l'>True / False</span>: האם המודל צדק<br>"
                                    "<span class='ev-l'>Positive / Negative</span>: מה המודל אמר"))

story = Scene("story", "מסיפור למספרים", [
    P("כמה שליליים אמיתיים יש? קודם סופרים את כל התקינים: 150 פחות 30, כלומר 120.", do=show("s-1"),
      say="כמה שליליים אמיתיים יש? קודם סופרים את כל התקינים: מאה חמישים פחות שלושים, כלומר מאה עשרים."),
    P("מהם מורידים את 9 אזעקות השווא, ונשארים 111.", do=show("s-2", "cm-v-tn"),
      say="מהם מורידים את תשע אזעקות השווא, ונשארים מאה אחד עשר."),
    P("״שאר התקינים״ פירושו מה שנשאר **מאותה מחלקה**. לא 150 פחות 24, שזה 126.", do=show("s-t"),
      say="שאר התקינים פירושו מה שנשאר מאותה מחלקה. לא מאה חמישים פחות עשרים וארבע, שזה מאה עשרים ושש."),
    P("ובדיקה אחרונה: 24 ועוד 6 ועוד 9 ועוד 111 זה בדיוק 150. כל בקבוק נספר פעם אחת.", do=show("s-3"),
      say="ובדיקה אחרונה: עשרים וארבע ועוד שש ועוד תשע ועוד מאה אחד עשר זה בדיוק מאה חמישים. כל בקבוק נספר פעם אחת.",
      hold=0.4),
], stage=w.col(CM),
   panel=st.card("s-1", "ex", "תקינים: $150 - 30 = 120$")
         + st.card("s-2", "ex", "$TN = 120 - 9 = 111$")
         + st.card("s-t", "trap", "⚠️ <span class='ev-l'>TN</span> ≠ הכול פחות <span class='ev-l'>TP</span>")
         + st.card("s-3", "rule", "🔑 $24 + 6 + 9 + 111 = 150$ ✓"))

acc = Scene("acc", "דיוק ושגיאה", [
    P("**Accuracy**, דיוק, שואל: מכל החיזויים, כמה נכונים? הנכונים הם האלכסון.", do=show("ca-h-tp", "ca-h-tn"),
      say="אַקְיוּרַסִי, דיוק, שואל: מכל החיזויים, כמה נכונים? הנכונים הם האלכסון."),
    P("24 ועוד 111 זה 135, מתוך 150 בקבוקים. כלומר 0.9, תשעים אחוז.", do=show("a-1"),
      say="עשרים וארבע ועוד מאה אחד עשר זה מאה שלושים וחמש, מתוך מאה חמישים בקבוקים. כלומר אפס נקודה תשע, תשעים אחוז."),
    P("**Error rate**, שיעור השגיאה, הוא החלק שהמודל טעה בו: המשבצות האדומות.",
      do=hide("ca-h-tp", "ca-h-tn") + show("ca-h-fn", "ca-h-fp"),
      say="אֶרוֹר רֵייט, שיעור השגיאה, הוא החלק שהמודל טעה בו: המשבצות האדומות."),
    P("6 ועוד 9 זה 15, מתוך 150: 0.1. ותמיד אחת פחות הדיוק.", do=show("a-2"),
      say="שש ועוד תשע זה חמש עשרה, מתוך מאה חמישים: אפס נקודה אחת. ותמיד, אחת פחות הדיוק."),
    P("שני המספרים משלימים זה את זה ל-1. שאלות מבחן אוהבות להחליף ביניהם את השמות.", do=show("a-t"), hold=0.4,
      say="שני המספרים משלימים זה את זה לאחת. שאלות מבחן אוהבות להחליף ביניהם את השמות."),
], stage=w.col(CM2,
               w.eq("a-1", "$\\text{Accuracy} = \\dfrac{TP + TN}{n} = \\dfrac{24 + 111}{150} = 0.9$", 36),
               w.eq("a-2", "$\\text{Error} = \\dfrac{FN + FP}{n} = \\dfrac{6 + 9}{150} = 0.1$", 36)),
   panel=st.card("a-k", "rule", "🔑 <span class='ev-l'>Accuracy</span> = האלכסון חלקי הכול")
         + st.card("a-t", "trap", "⚠️ <span class='ev-l'>Error rate = 1 − Accuracy</span>: שני מספרים, שני שמות"))
acc.phrases[0].do += show("a-k")

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: נתונה מטריצה, שורות הן הערך המצופה ועמודות הן החיזוי. איזו טענה נכונה, או הקרובה ביותר?"),
    P("קודם התאים. בשורה העליונה, החיוביים האמיתיים: 400 נחזו חיובי, ו-500 נחזו שלילי.", do=show("cx-h-tp", "cx-h-fn"),
      say="קודם התאים. בשורה העליונה, החיוביים האמיתיים: ארבע מאות נחזו חיובי, וחמש מאות נחזו שלילי."),
    P("בשורה התחתונה, השליליים: 300 אזעקות שווא, ו-100 שליליים אמיתיים.",
      do=hide("cx-h-tp", "cx-h-fn") + show("cx-h-fp", "cx-h-tn"),
      say="בשורה התחתונה, השליליים: שלוש מאות אזעקות שווא, ומאה שליליים אמיתיים."),
    P("אז שליליים אמיתיים יש 100, לא 400. ה-400 הוא התא של החיוביים האמיתיים.", do=show("q-x-1", "q-n-1"),
      say="אז שליליים אמיתיים יש מאה, לא ארבע מאות. הארבע מאות הוא התא של החיוביים האמיתיים."),
    P("הדיוק: 400 ועוד 100, מתוך 1300. זה בערך 0.38.", do=hide("cx-h-fp") + show("cx-h-tp", "x-a"),
      say="הדיוק: ארבע מאות ועוד מאה, מתוך אלף שלוש מאות. זה בערך אפס נקודה שלושים ושמונה."),
    P("ושיעור השגיאה הוא המשלים: בערך 0.62. עכשיו הסתכלו על התשובות.", do=show("x-e"),
      say="ושיעור השגיאה הוא המשלים: בערך אפס נקודה שישים ושתיים. עכשיו הסתכלו על התשובות."),
    P("שתיהן מצמידות מספר נכון לשם הלא נכון: 0.38 הוא הדיוק, ו-0.62 הוא השגיאה.",
      do=show("q-x-0", "q-n-0", "q-x-3", "q-n-3"),
      say="שתיהן מצמידות מספר נכון לשם הלא נכון: אפס נקודה שלושים ושמונה הוא הדיוק, ואפס נקודה שישים ושתיים הוא השגיאה."),
    P("נשארה הטענה על Precision: מכל מה שהמודל סימן חיובי, כמה באמת חיובי? 400 מתוך 700, בערך 0.57.",
      do=hide("cx-h-tn") + show("cx-h-fp", "x-p", "q-ok-2"),
      say="נשארה הטענה על הפְּרֶסִיזְ'ן: מכל מה שהמודל סימן חיובי, כמה באמת חיובי? ארבע מאות מתוך שבע מאות, בערך אפס נקודה חמישים ושבע."),
    P("את המדד הזה נפרק לעומק בשלב הבא. המלכודת כאן: מספרים אמיתיים, עם שמות מוחלפים.", do=show("x-t"), hold=0.6),
], stage=w.col(CMX,
               w.eq("x-a", "$\\text{Accuracy} = \\dfrac{400 + 100}{1300} \\approx 0.38$", 32),
               w.eq("x-e", "$\\text{Error} = \\dfrac{500 + 300}{1300} \\approx 0.62$", 32),
               w.eq("x-p", "$\\text{Precision} = \\dfrac{400}{400 + 300} \\approx 0.57$", 32), gap=18),
   panel=w.stem(Q["id"], "מטריצת בלבול (שורות = ערך מצופה, עמודות = חיזוי). איזו טענה נכונה?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={0: "זה הדיוק", 1: "זה TP", 3: "זו השגיאה"})
         + "</div>" + st.card("x-t", "trap", "⚠️ חשבו את <b>כל</b> המדדים, ואז התאימו שם למספר"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: שורה היא האמת, עמודה היא החיזוי.", do=show("r-1")),
    P("מסיפור: הפספוסים הם החיוביים פחות מה שנתפס, והשליליים האמיתיים הם השליליים פחות אזעקות השווא.", do=show("r-2")),
    P("ובודקים שהסכום שווה למספר הדוגמאות.", do=show("r-3")),
    P("הדיוק הוא האלכסון חלקי הכול, והשגיאה היא אחת פחות הדיוק. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(w.cmatrix("cr", "TP", "FN", "FP", "TN", tags=False)),
   panel=st.card("r-1", "rule", "🔑 שורה = אמת · עמודה = חיזוי")
         + st.card("r-2", "rule", "🔑 $FN = $ חיוביים $- TP$ · $TN = $ שליליים $- FP$")
         + st.card("r-3", "rule", "🔑 $TP + FN + FP + TN = n$")
         + st.card("r-4", "rule", "🔑 $\\text{Acc} = \\dfrac{TP + TN}{n}$ · $\\text{Err} = 1 - \\text{Acc}$"))

VIDEO = w.check_say(Video("E03", "eval", "מטריצת בלבול ו-accuracy", [title, four, story, acc, exam, recap],
                          bank={"PRAC-X-Q17": 2}))
