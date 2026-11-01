# -*- coding: utf-8 -*-
"""E02 · אימון, תיקוף ו-overfitting — video script (build: tools/video/vid.py E02).

Source: courses/eval/E02.js (goal: the role of train / validation / test, what a hyper-parameter is, and recognising
overfitting and underfitting with their treatment). Exam anchor: 24S-C-Q1 ("what is hyperparameter tuning"), which no
guided quiz uses and has no twin among the quiz items.
Not used on purpose: the E02 quiz scenarios (99% / 62%, 58% / 55%, k of KNN, max_depth 3/5/8/12, early stopping,
kg / pound features, the linear classifier) and the quiz refs 24S-A-Q7, 22B-A-Q18, SAMP-1-Q10; the rule that
treats overfitting is shown only inside the polynomial demo (lower the degree / raise the degree), never as a list.
Worked example: polynomial regression of degree 1..7 on 8 training and 7 validation points (fixed data, numpy
polyfit); every error below is computed here and the U shape is asserted.
"""
import numpy as np

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

# ---------------------------------------------------------------- the polynomial demo (verified)
f = lambda x: 1 + 8 * x - 8 * x ** 2
XT = np.array([0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0])
XV = np.array([0.07, 0.21, 0.36, 0.5, 0.64, 0.79, 0.93])
YT = f(XT) + np.array([0.25, -0.3, 0.2, -0.25, 0.3, -0.2, 0.25, -0.3])
YV = f(XV) + np.array([-0.2, 0.25, -0.15, 0.2, -0.25, 0.15, -0.2])
DEG = list(range(1, 8))
FIT = {d: np.polyfit(XT, YT, d) for d in DEG}
TR = {d: float(np.mean((np.polyval(FIT[d], XT) - YT) ** 2)) for d in DEG}
VA = {d: float(np.mean((np.polyval(FIT[d], XV) - YV) ** 2)) for d in DEG}
assert all(TR[d + 1] <= TR[d] + 1e-12 for d in DEG[:-1])          # training error never goes up with the degree
assert TR[7] < 1e-12                                              # degree 7 passes through all 8 training points
assert min(VA, key=VA.get) == 2                                   # validation picks degree 2
assert VA[1] > 0.4 and TR[1] > 0.6                                # degree 1: bad on both -> underfitting
assert VA[7] > 0.7 and VA[7] > 10 * VA[2]                         # degree 7: perfect on train, bad on validation
R2 = lambda v: f"{v:.2f}"
assert (R2(TR[1]), R2(VA[1]), R2(TR[2]), R2(VA[2]), R2(TR[7]), R2(VA[7])) == ("0.66", "0.47", "0.06", "0.04", "0.00", "0.71")

# ---------------------------------------------------------------- exam 24S-C-Q1
Q = BANK["24S-C-Q1"]
assert Q["correctIndex"] == 0 and "הבלתי נלמדים" in Q["options"][0]
OPTS = w.opts(Q)

# ---------------------------------------------------------------- figures
XS = np.linspace(0, 1, 121)
PF = w.Plot("pf", (0, 1), (-0.4, 3.6), w=760, h=470, xticks=(0, 0.5, 1), yticks=(0, 1, 2, 3), xname="x", yname="y")
PF.pts("tr", list(zip(XT, YT)), "a")
PF.pts("va", list(zip(XV, YV)), "b", hidden=True)
for d, k in ((1, "c"), (2, "ok"), (7, "bad")):
    PF.poly(f"d{d}", list(zip(XS, np.polyval(FIT[d], XS))), k)
FITSVG = PF.svg()

PC = w.Plot("pc", (0.6, 7.4), (0, 0.86), w=760, h=470, xticks=DEG, yticks=(0, 0.2, 0.4, 0.6, 0.8),
            xname="", yname="")
PC.band("bu", 0.6, 1.4, "c")
PC.band("bo", 5.6, 7.4, "bad")
PC.poly("ctr", [(d, TR[d]) for d in DEG], "a", width=5)
PC.pts("ptr", [(d, TR[d]) for d in DEG], "a", hidden=True, r=8)
PC.poly("cva", [(d, VA[d]) for d in DEG], "b", width=5)
PC.pts("pva", [(d, VA[d]) for d in DEG], "b", hidden=True, r=8)
PC.pts("best", [(2, VA[2])], "ok", hidden=True, r=15)
PC.text("lbu", 1, 0.76, "חסר", "c", size=24)
PC.text("lbo", 6.5, 0.76, "יתר", "bad", size=24)
CURSVG = PC.svg()

LEG = ('<div class="ev-row" style="gap:30px;font-size:26px">'
       '<span id="lg-a" class="h" style="color:#38bdf8">● אימון</span>'
       '<span id="lg-b" class="h" style="color:#fbbf24">● תיקוף</span></div>')
SPLIT = ('<div class="ev-frow" id="sp">'
         '<div id="sp-tr" class="ev-fold trn h" style="width:500px">אימון · 600</div>'
         '<div id="sp-va" class="ev-fold val h" style="width:170px">תיקוף · 200</div>'
         '<div id="sp-te" class="ev-fold tst h" style="width:170px">בדיקה · 200</div></div>')

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבין למה מחלקים את הנתונים לשלוש קבוצות, ומה זה היפר-פרמטר."),
    P("ונראה בגרף אמיתי איך נראים overfitting ו-underfitting, ואיך בוחרים בין שניהם.", hold=0.4,
      say="ונראה בגרף אמיתי איך נראים אוֹבֶרְפִיטִינְג ואַנְדֶרְפִיטִינְג, ואיך בוחרים בין שניהם."),
], stage='<div class="kicker">E02 · הערכת מודל</div><h1>אימון, תיקוף ו-overfitting</h1>'
         '<p>למה משמשת כל קבוצה, מה ההבדל בין פרמטר להיפר-פרמטר, ואיך מזהים התאמת יתר והתאמת חסר.</p>',
   layout="title")

sets = Scene("sets", "שלוש קבוצות", [
    P("מודל טוב הוא לא מודל שזוכר את הדוגמאות שלו. הוא מודל שעובד טוב על דוגמאות **חדשות**. לזה קוראים הכללה."),
    P("לכן מחלקים, למשל, אלף דוגמאות לשלוש קבוצות.", do=show("sp-tr", "sp-va", "sp-te")),
    P("על קבוצת **האימון** האלגוריתם לומד. רק עליה.", do=show("s-1")),
    P("קבוצת **התיקוף** משמשת לבחירות: איזו הגדרה של המודל עדיפה, ואיזה מודל לקחת.", do=show("s-2")),
    P("וקבוצת **הבדיקה** נשמרת לסוף: ציון אחד, סופי, על המודל שכבר נבחר.", do=show("s-3")),
    P("אם בוחרים לפי הבדיקה, היא כבר לא ״חדשה״, והציון שלה יוצא אופטימי מדי.", do=show("s-t"), hold=0.4),
], stage=w.col(SPLIT, w.note("1000 דוגמאות")),
   panel=st.card("s-1", "ex", "<b>אימון</b>: עליה לומדים")
         + st.card("s-2", "ex", "<b>תיקוף</b>: עליה בוחרים")
         + st.card("s-3", "ex", "<b>בדיקה</b>: ציון סופי, פעם אחת")
         + st.card("s-t", "trap", "⚠️ בחירה לפי הבדיקה ← ציון מנופח"))

fits = Scene("fits", "שלושה מודלים", [
    P("נראה את זה על דוגמה. הנקודות הכחולות הן האימון: שמונה נקודות על גבעה, עם קצת רעש."),
    P("נתאים להן פולינום. המקדמים שלו נלמדים מהנתונים: הם ה**פרמטרים**.", do=show("h-1")),
    P("אבל את **מעלת** הפולינום בוחרים אנחנו, לפני האימון. זה **היפר-פרמטר**.", do=show("h-2")),
    P("מעלה אחת: קו ישר. הוא פשוט מדי, ולא תופס את הגבעה בכלל.", do=show("pf-d1")),
    P("מעלה שתיים: פרבולה. היא עוברת באמצע הנקודות, ומתעלמת מהרעש.", do=show("pf-d2")),
    P("מעלה שבע: עוברת **בדיוק** דרך כל שמונה הנקודות. נראה מושלם.", do=show("pf-d7")),
    P("עכשיו נוסיף את נקודות התיקוף, הכתומות, שהמודלים לא ראו.", do=show("pf-va", "lg-a", "lg-b")),
    P("העקומה של מעלה שבע מתפתלת בין הנקודות, ומפספסת את הכתומות. היא למדה גם את הרעש.", do=show("h-3"), hold=0.4),
], stage=w.col(FITSVG, LEG, gap=10),
   panel=st.card("h-1", "plain", "<b>פרמטר</b>: נלמד מהנתונים (המקדמים)")
         + st.card("h-2", "rule", "🔑 <b>היפר-פרמטר</b>: נקבע לפני האימון (המעלה)")
         + st.card("h-3", "trap", "⚠️ מושלם על האימון ≠ טוב על דוגמאות חדשות"))

curves = Scene("curves", "עקומות השגיאה", [
    P("נמדוד את זה. לכל מעלה, מאחת עד שבע, מחשבים את השגיאה על האימון ועל התיקוף."),
    P("שגיאת האימון, בכחול, רק יורדת. ככל שהמודל מורכב יותר, הוא מתאים יותר את עצמו לאימון.", do=show("pc-ptr", "pc-ctr")),
    P("במעלה שבע היא אפס.", do=show("c-1")),
    P("שגיאת התיקוף, בכתום, מספרת סיפור אחר: היא יורדת, ואז עולה.", do=show("pc-pva", "pc-cva")),
    P("במעלה אחת שתיהן גבוהות: אפס נקודה שישים ושש באימון. המודל נכשל אפילו על מה שראה. זה underfitting.",
      do=show("pc-bu", "pc-lbu", "c-2"),
      say="במעלה אחת שתיהן גבוהות: אפס נקודה שישים ושש באימון. המודל נכשל אפילו על מה שראה. זה אַנְדֶרְפִיטִינְג."),
    P("במעלה שבע האימון מושלם, אבל התיקוף קופץ לאפס נקודה שבעים ואחת. זה overfitting.",
      do=show("pc-bo", "pc-lbo", "c-3"),
      say="במעלה שבע האימון מושלם, אבל התיקוף קופץ לאפס נקודה שבעים ואחת. זה אוֹבֶרְפִיטִינְג."),
    P("הבחירה: המעלה עם שגיאת התיקוף הנמוכה ביותר. כאן מעלה שתיים, עם אפס נקודה אפס ארבע.", do=show("pc-best", "c-4")),
    P("בדוגמה שלנו: ממעלה אחת עולים במעלה, ממעלה שבע יורדים. ורק בסוף, מעלה שתיים נבחנת פעם אחת על הבדיקה.",
      do=show("c-5"), hold=0.4),
], stage=w.col(CURSVG, w.note("לרוחב: מעלת הפולינום · לגובה: שגיאה ממוצעת בריבוע"), LEG.replace('id="lg-a" class="h"', 'id="lg-c"').replace('id="lg-b" class="h"', 'id="lg-d"'),
               gap=10),
   panel=st.card("c-1", "plain", "אימון: $0.66 \\to 0.06 \\to \\dots \\to 0$")
         + st.card("c-2", "ex", "<b>חסר</b>: גרוע גם באימון וגם בתיקוף")
         + st.card("c-3", "ex", "<b>יתר</b>: מצוין באימון, גרוע בתיקוף")
         + st.card("c-4", "rule", "🔑 בוחרים לפי <b>מינימום</b> שגיאת התיקוף ← מעלה 2")
         + st.card("c-5", "plain", "ואחר כך: ציון סופי אחד על הבדיקה"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: מהו כוונון היפר-פרמטרים? בדיוק מה שעשינו עכשיו עם מעלת הפולינום."),
    P("את המעלה לא למדנו מהנתונים. קבענו אותה, ניסינו כמה ערכים, ובחרנו לפי התיקוף.", do=show("x-1")),
    P("לכן כוונון היפר-פרמטרים הוא כוונון של הפרמטרים ש**אינם נלמדים**. זו התשובה.", do=show("q-ok-0")),
    P("״למידת הפרמטרים בצורה מהירה״ מתארת אימון: שם לומדים את המקדמים, ולא כוונון.", do=show("q-x-1", "q-n-1")),
    P("״הפרמטרים של המישור המפריד״ הם בדיוק מה שמודל לומד באימון. ה״היפר״ לא קשור למילה מישור.",
      do=show("q-x-2", "q-n-2")),
    P("והאפשרות על אילון מאסק היא בדיחה: קיבוץ נתונים הוא אשכול, לא כוונון.", do=show("q-x-3", "q-n-3"), hold=0.6),
], stage=w.col(CURSVG.replace('id="pc', 'id="px').replace('class="h"', 'class=""').replace(' h"', '"'),
               st.card("x-1", "rule", "🔑 מעלה = היפר-פרמטר: <b>נבחרה לפי התיקוף</b>, לא נלמדה"), gap=20),
   panel=w.stem(Q["id"], "מהו <span class='ev-l'>Hyperparameter Tuning</span>?")
         + '<div class="ev-opts">' + st.options("q", OPTS, notes={1: "זה אימון", 2: "נלמד באימון", 3: "זה אשכול"})
         + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: לומדים על האימון, בוחרים על התיקוף, ומקבלים ציון אחד על הבדיקה.", do=show("r-1")),
    P("פרמטרים נלמדים מהנתונים. היפר-פרמטרים קובעים לפני האימון.", do=show("r-2")),
    P("גרוע גם באימון: התאמת חסר. מצוין באימון וגרוע בתיקוף: התאמת יתר.", do=show("r-3")),
    P("והגרף אומר לאן לזוז: לנקודה שבה שגיאת התיקוף הכי נמוכה. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(SPLIT.replace('id="sp', 'id="rs').replace(" h\"", "\"")),
   panel=st.card("r-1", "rule", "🔑 אימון ← לומדים · תיקוף ← בוחרים · בדיקה ← ציון סופי")
         + st.card("r-2", "rule", "🔑 פרמטר = נלמד · היפר-פרמטר = נקבע מראש")
         + st.card("r-3", "rule", "🔑 חסר: גרוע בשניהם · יתר: פער גדול")
         + st.card("r-4", "trap", "⚠️ שגיאת אימון נמוכה לבדה לא אומרת כלום"))

VIDEO = w.check_say(Video("E02", "eval", "אימון, תיקוף ו-overfitting", [title, sets, fits, curves, exam, recap],
                          bank={"24S-C-Q1": 0}))
