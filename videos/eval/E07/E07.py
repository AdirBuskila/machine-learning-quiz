# -*- coding: utf-8 -*-
"""E07 · Hold-out מול K-fold — video script (build: tools/video/vid.py E07).

Source: courses/eval/E07.js (goal: choose hold-out or k-fold by the amount of data, compute fold size and number of
trainings, recognise leave-one-out). Exam anchor: SAMP-2-Q6 ("which goal is a goal of cross-validation"), which no
guided quiz uses and is not a twin of any quiz item (its twin SAMP-3-Q6 is not used by a quiz either).
Not used on purpose: the E07 quiz scenarios (500 examples with K = 10, K = 8, 120 examples and three models, 5 million
examples, the 40-example leave-one-out, "the drawback of k-fold") and the refs 22B-A-Q16, 25B-A-Q12, 22B-C-Q4, the
boss items 23B-B-Q19 and E10's 900 / K = 6; also every "few data -> k-fold" exam question (24B-A-Q2, 24B-C-Q3,
24S-A-Q4, 26B-A-Q17, 26S-A-Q18) and "when hold-out" (23S-A-Q13, 22B-C-Q16, SAMP-2-Q21, PRAC-X-Q24), which mirror
quiz items. The choice rule is taught on new cases (3 million photos, 90 patients).
Worked example: 240 examples, K = 4 (sklearn KFold asserts 60 / 180 and that every example is validated once),
fold scores 0.80 / 0.85 / 0.75 / 0.80 -> 0.80, and leave-one-out on 6 examples.
"""
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

# ---------------------------------------------------------------- K-fold demo (verified)
N, K = 240, 4
SPL = list(KFold(n_splits=K).split(np.zeros(N)))
assert all(len(v) == 60 and len(t) == 180 for t, v in SPL)
VAL = np.concatenate([v for _, v in SPL])
assert sorted(VAL.tolist()) == list(range(N))                               # each example validated exactly once
TRN = np.concatenate([t for t, _ in SPL])
assert all(c == K - 1 for c in np.bincount(TRN, minlength=N))              # and trained on K-1 = 3 times
SC = [0.80, 0.85, 0.75, 0.80]
assert abs(sum(SC) / K - 0.80) < 1e-12
LOO = list(LeaveOneOut().split(np.zeros(6)))
assert len(LOO) == 6 and all(len(t) == 5 and len(v) == 1 for t, v in LOO)
assert 180 + 60 == N

# ---------------------------------------------------------------- exam SAMP-2-Q6
Q = BANK["SAMP-2-Q6"]
assert Q["correctIndex"] == 0 and "להעריך את הביצועים החיזויים" in Q["options"][0]
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
HO = w.holdout("ho", N, 60)
KF = w.folds("kf", K, N)
SCH = "".join(w.chip(f"s{r}", f"{SC[r]:.2f}", "info") for r in range(K))
DOTS = ('<div class="ev-row" style="gap:18px">' + "".join(
    f'<span id="lo-{i}" class="ev-node a h" style="font-size:28px;padding:10px 20px">{i + 1}</span>' for i in range(6))
    + "</div>")

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נשווה שתי דרכים להעריך מודל: hold-out ו-K-fold.",
      say="בסרטון הזה נשווה שתי דרכים להעריך מודל: הוֹלְד אַאוּט וקֵיי פוֹלְד."),
    P("נחשב כמה דוגמאות יש בכל חלק, כמה אימונים צריך, ונדע מתי לבחור כל אחת.", hold=0.4),
], stage='<div class="kicker">E07 · הערכת מודל</div><h1>Hold-out מול K-fold</h1>'
         '<p>גודל fold, מספר אימונים, leave-one-out, ומתי כל שיטה מתאימה.</p>', layout="title")

hold = Scene("hold", "Hold-out", [
    P("כדי להעריך מודל צריך דוגמאות שהוא **לא** התאמן עליהן. יש לנו 240 דוגמאות מתויגות.",
      say="כדי להעריך מודל צריך דוגמאות שהוא לא התאמן עליהן. יש לנו מאתיים ארבעים דוגמאות מתויגות."),
    P("הדרך הפשוטה: hold-out. מפרישים **פעם אחת** 60 דוגמאות לתיקוף, ומאמנים על 180.", do=show("ho"),
      say="הדרך הפשוטה: הוֹלְד אַאוּט. מפרישים פעם אחת שישים דוגמאות לתיקוף, ומאמנים על מאה שמונים."),
    P("כאן יש הרצה אחת בלבד של האימון.", do=show("h-1")),
    P("החיסרון: 60 הדוגמאות האלה אף פעם לא משמשות לאימון. והציון תלוי במזל: אילו דוגמאות נפלו בצד.",
      do=show("h-2"),
      say="החיסרון: שישים הדוגמאות האלה אף פעם לא משמשות לאימון. והציון תלוי במזל: אילו דוגמאות נפלו בצד.", hold=0.4),
], stage=w.col(HO, w.note("240 דוגמאות")),
   panel=st.card("h-1", "ex", "הרצה <b>אחת</b> של האימון")
         + st.card("h-2", "trap", "⚠️ חלק מהנתונים לא מתאמן, וציון של חלוקה אחת"))

kfold = Scene("kfold", "K-fold", [
    P("ב-K-fold מחלקים את 240 הדוגמאות ל-K חלקים שווים. נבחר K שווה ארבע: כל חלק, fold, הוא 60 דוגמאות.",
      do=show("kf-r0", "k-1"),
      say="בקֵיי פוֹלְד מחלקים את מאתיים ארבעים הדוגמאות לקֵיי חלקים שווים. נבחר קֵיי שווה ארבע: כל חלק, פוֹלְד, הוא שישים דוגמאות."),
    P("בסבב הראשון ה-fold הראשון הוא התיקוף, ושלושת האחרים, 180 דוגמאות, הם האימון.", do=show("k-2"),
      say="בסבב הראשון הפוֹלְד הראשון הוא התיקוף, ושלושת האחרים, מאה שמונים דוגמאות, הם האימון."),
    P("בכל סבב חלק אחר עובר לתיקוף. ארבעה סבבים, ארבעה אימונים.", do=show("kf-r1", "kf-r2", "kf-r3")),
    P("כל דוגמה נמצאת בתיקוף **פעם אחת בדיוק**, ובאימון בשלושת הסבבים האחרים.", do=show("k-3")),
    P("כל סבב נותן ציון. ממצעים את ארבעתם: הציון הסופי, 0.8.", do=show("s0", "s1", "s2", "s3", "k-4"),
      say="כל סבב נותן ציון. ממצעים את ארבעתם: הציון הסופי, אפס נקודה שמונה."),
    P("ארבעה סבבים הם ארבע הרצות של האימון. בתמורה, כל הנתונים עבדו גם באימון וגם בתיקוף.",
      do=show("k-5"), hold=0.4),
], stage=w.col(KF, f'<div class="ev-row" style="gap:16px">{SCH}</div>', gap=22),
   panel=st.card("k-1", "ex", "$\\text{fold} = \\dfrac{240}{4} = 60$")
         + st.card("k-2", "ex", "אימון בכל סבב: $(4 - 1) \\cdot 60 = 180$")
         + st.card("k-3", "rule", "🔑 כל דוגמה: תיקוף <b>פעם אחת</b>, אימון $K - 1$ פעמים")
         + st.card("k-4", "plain", "$\\tfrac{1}{4}(0.80 + 0.85 + 0.75 + 0.80) = 0.80$")
         + st.card("k-5", "plain", "ארבעה סבבים = ארבע הרצות"))

loo = Scene("loo", "Leave-One-Out", [
    P("ומה אם K שווה למספר הדוגמאות? נראה על דוגמה קטנה של שש דוגמאות.", do=show(*[f"lo-{i}" for i in range(6)]),
      say="ומה אם קֵיי שווה למספר הדוגמאות? נראה על דוגמה קטנה של שש דוגמאות."),
    P("כל fold הוא דוגמה **אחת**. בכל סבב מאמנים על חמש, ובודקים את זו שנשארה בחוץ.", do=show("l-1"),
      say="כל פוֹלְד הוא דוגמה אחת. בכל סבב מאמנים על חמש, ובודקים את זו שנשארה בחוץ."),
    P("שש דוגמאות, שישה אימונים. זה Leave-One-Out: מקרה פרטי של K-fold.", do=show("l-2"),
      say="שש דוגמאות, שישה אימונים. זה לִיב וַואן אַאוּט: מקרה פרטי של קֵיי פוֹלְד.", hold=0.4),
], stage=w.col(DOTS, w.note("K = 6 = מספר הדוגמאות"), gap=24),
   panel=st.card("l-1", "ex", "סבב: אימון על $6 - 1 = 5$, בדיקה על 1")
         + st.card("l-2", "rule", "🔑 <span class='ev-l'>K = N</span> ← <span class='ev-l'>Leave-One-Out</span>, "
                                  "<span class='ev-l'>N</span> אימונים"))
choose = Scene("choose", "מתי מה", [
    P("מתי שווה לשלם על ההרצות הנוספות? כשכל דוגמה יקרה.", do=show("c-q")),
    P("נניח תשעים מטופלים בלבד. אי אפשר לוותר על אף אחד מהם.", do=show("c-2")),
    P("K-fold נותן לכל מטופל לשמש גם לאימון וגם לתיקוף, וממצע כמה חלוקות במקום להמר על אחת.", do=show("c-2b"),
      say="קֵיי פוֹלְד נותן לכל מטופל לשמש גם לאימון וגם לתיקוף, וממצע כמה חלוקות במקום להמר על אחת."),
    P("ובשתי השיטות, קבוצת הבדיקה לא נוגעים בה. היא נשארת להערכה הסופית.", do=show("c-t"), hold=0.4),
], stage=w.col(st.card("c-q", "plain", "<b>כמה שווה כל דוגמה?</b>"),
               st.card("c-2", "ex", "90 מטופלים<br><b>מעט</b> נתונים"),
               w.chip("c-2b", "K-fold", "warn"), gap=26),
   panel=st.card("c-t", "trap", "⚠️ ה-<span class='ev-l'>test</span> לא משמש לתיקוף, באף שיטה"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו מטרה היא מטרה של cross-validation?",
      say="שאלה ממבחן: איזו מטרה היא מטרה של קְרוֹס וָלִידֵיישְׁן?"),
    P("ראינו מה הוא עושה: מאמן ובודק כמה פעמים, ומעריך עד כמה המודל טוב על דוגמאות שלא ראה.", do=show("x-1")),
    P("לכן הוא נועד להעריך את ביצועי החיזוי של המודלים. זו הטענה הנכונה.", do=show("q-ok-0")),
    P("הטענה על נתוני **הבדיקה** מדברת על שלב אחר. אבל כל הסבבים רצים בתוך נתוני האימון.", do=show("x-2")),
    P("הבדיקה היא שלב נפרד, שמגיע אחרי. cross-validation לא נוגע בה, ולכן הטענה הזו נופלת.",
      do=show("q-x-1", "q-n-1"),
      say="הבדיקה היא שלב נפרד, שמגיע אחרי. קְרוֹס וָלִידֵיישְׁן לא נוגע בה, ולכן הטענה הזו נופלת."),
    P("ומכאן ״גם וגם״ נופלת, כי הטענה על נתוני הבדיקה שגויה. ו״אף אחת״ נופלת, כי הטענה על הערכת הביצועים נכונה.",
      do=show("q-x-2", "q-n-2", "q-x-3", "q-n-3"), hold=0.6),
], stage=w.col(st.card("x-1", "rule", "🔑 <span class='ev-l'>CV</span> = הערכה של ביצועי חיזוי, מתוך נתוני האימון"),
               '<div id="x-2" class="h">' + w.row(
                   '<div class="ev-fold trn" style="width:560px">אימון: כאן רצים כל הסבבים</div>',
                   '<div class="ev-fold tst" style="width:220px">בדיקה: בסוף</div>', gap=10) + "</div>", gap=30),
   panel=w.stem(Q["id"], "איזו מטרה מהבאות מהווה מטרה עבור <span class='ev-l'>cross-validation</span>?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={1: "לא נוגע בבדיקה", 2: "אחת שגויה",
                                                                    3: "יש נכונה"}) + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: hold-out מפריש פעם אחת: הרצה אחת, וחלק מהנתונים לא מתאמן.", do=show("r-1"),
      say="לסיכום: הוֹלְד אַאוּט מפריש פעם אחת: הרצה אחת, וחלק מהנתונים לא מתאמן."),
    P("K-fold: גודל fold הוא מספר הדוגמאות חלקי K. כל דוגמה בתיקוף פעם אחת, ובאימון K פחות אחת פעמים.",
      do=show("r-2"),
      say="קֵיי פוֹלְד: גודל פוֹלְד הוא מספר הדוגמאות חלקי קֵיי. כל דוגמה בתיקוף פעם אחת, ובאימון קֵיי פחות אחת פעמים."),
    P("K שווה למספר הדוגמאות: Leave-One-Out.", do=show("r-3"), say="קֵיי שווה למספר הדוגמאות: לִיב וַואן אַאוּט."),
    P("וכשכל דוגמה יקרה, K-fold נותן לכל אחת לעבוד גם באימון וגם בתיקוף. נתראה בשלב הבא.", do=show("r-4"), hold=0.6,
      say="וכשכל דוגמה יקרה, קֵיי פוֹלְד נותן לכל אחת לעבוד גם באימון וגם בתיקוף. נתראה בשלב הבא."),
], stage=w.col(KF.replace('id="kf', 'id="rf').replace(' h"', '"')),
   panel=st.card("r-1", "rule", "🔑 <span class='ev-l'>hold-out</span>: חלוקה אחת")
         + st.card("r-2", "rule", "🔑 $\\text{fold} = \\dfrac{N}{K}$ · אימון $= (K - 1)\\cdot \\text{fold}$")
         + st.card("r-3", "rule", "🔑 <span class='ev-l'>K = N</span> ← <span class='ev-l'>Leave-One-Out</span>")
         + st.card("r-4", "trap", "⚠️ <span class='ev-l'>test</span> לא משמש לתיקוף"))

VIDEO = w.check_say(Video("E07", "eval", "Hold-out מול K-fold", [title, hold, kfold, loo, choose, exam, recap],
                          bank={"SAMP-2-Q6": 0}))
