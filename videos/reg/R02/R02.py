# -*- coding: utf-8 -*-
"""R02 · מדדי רגרסיה ו-R² — video script (build: tools/video/vid.py R02).

Source: courses/reg/R02.js (goal: spot the metric that is not a regression metric, decide who built the better model,
compute R² = 1 − SSE/SST). Exam part: an original "בסגנון מבחן" question (bank={}): SST = 50, SSE = 60 -> R² = −0.2,
the model is worse than predicting the mean. The real "which metric is not a regression metric" questions (26B-A-Q14,
25B-A-Q18, 22B-A-Q9, 22B-C-Q11, PRAC-X-Q31, SAMP-2-Q16) all eliminate a metric that a quiz/boss item uses
(Precision, purity, F1, NMI, Mean Chebyshev Error), so none is solved here.
Avoided on purpose: the R02 quiz items (Precision / purity as the odd one out, SST = 200 & SSE = 50, the 2/4/6/8 table,
the four students 0.62/0.71/0.88/0.55, "why not MAE"), the boss R09-Q2/R09-Q3 and the quiz refs 23B-A-Q14, 25S-A-Q12,
26B-A-Q5 with their twins (24S-A-Q2, 24B-A-Q16, 26S-A-Q8, CLAUDE-26-Q10). The family scene names only accuracy, recall
and WSSE outside regression (no precision, F1, purity, NMI, information gain, conditional entropy, Mean Chebyshev Error).
Own data: y = 3, 5, 6, 7, 9 against ŷ = 2x (SST 20, SSE 4, R² 0.8); three models on one test set (n = 10, SST = 200)
in both R² and RMSE.
"""
from fractions import Fraction as Fr
import math

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
XS, YS = [1, 2, 3, 4, 5], [3, 5, 6, 7, 9]
YBAR = Fr(sum(YS), len(YS))
assert YBAR == 6
DEV = [y - YBAR for y in YS]
assert DEV == [-3, -1, 0, 1, 3]
SST = sum(d * d for d in DEV)
YH = [2 * x for x in XS]
E = [yh - y for yh, y in zip(YH, YS)]
assert E == [-1, -1, 0, 1, 1]
SSE = sum(e * e for e in E)
R2 = 1 - Fr(SSE, SST)
assert SST == 20 and SSE == 4 and R2 == Fr(4, 5)
assert Fr(SSE, SST) == Fr(1, 5)          # the "forgot 1 −" value

# three models on one test set: n = 10, SST = 200. R² = 1 − SSE/SST, RMSE = sqrt(SSE/n)
N, SST2 = 10, 200
MODELS = {"א": Fr(64, 100), "ב": Fr(82, 100), "ג": Fr(46, 100)}
RMSE = {k: math.sqrt((1 - r) * SST2 / N) for k, r in MODELS.items()}
RMSE_S = {k: f"{v:.2f}" for k, v in RMSE.items()}
assert RMSE_S == {"א": "2.68", "ב": "1.90", "ג": "3.29"}
best_r2 = max(MODELS, key=MODELS.get)
best_rmse = min(RMSE, key=RMSE.get)
assert best_r2 == best_rmse == "ב"
# same data -> same order: higher R² <=> lower RMSE
assert sorted(MODELS, key=MODELS.get) == sorted(RMSE, key=RMSE.get, reverse=True)

# exam part (original): SST = 50, SSE = 60
QSST, QSSE = 50, 60
assert 1 - Fr(QSSE, QSST) == Fr(-1, 5) and Fr(QSSE, QSST) == Fr(6, 5)
QOPTS = [(0, "$R^2 = -0.2$: המודל גרוע יותר מחיזוי קבוע של הממוצע"),
         (1, "$R^2 = 1.2$: המודל טוב יותר ממודל מושלם"),
         (2, "$R^2 = 0.2$, כי $R^2$ תמיד בין 0 ל-1"),
         (3, "לא ייתכן: <span class=\"rg-en\">SSE</span> לעולם אינו גדול מ-<span class=\"rg-en\">SST</span>")]
OK, RATIO, ABSV, IMPOS = 0, 1, 2, 3

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד לזהות איזה מדד שייך לרגרסיה, ואיזה לא."),
    P("נחשב את המדד R², ונכריע מי בנה מודל טוב יותר.", say=f"נחשב את המדד {SAY['R2']}, ונכריע מי בנה מודל טוב יותר."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">R02 · רגרסיה ורשתות נוירונים</div><h1>מדדי רגרסיה ו-<span class="rg-en">R²</span></h1>'
         '<p>מה מודד מדד של רגרסיה, מי בנה מודל טוב יותר, ואיך מחשבים '
         '<span class="rg-en">R² = 1 − SSE/SST</span>.</p>', layout="title")

FAM = [
    ("f-reg", "רגרסיה", "MSE · RMSE · MAE · SAE · SSE · R²", "ok"),
    ("f-cls", "סיווג", "accuracy · recall", "info"),
    ("f-clu", "אשכול", "WSSE", "warn"),
]
FAM_HTML = "".join(f'<div id="{i}" class="h rg-fam"><span class="rg-chip {k}">{t}</span>'
                   f'<span class="rg-en rg-fl">{m}</span></div>' for i, t, m, k in FAM)

fam = Scene("fam", "מדד של מי?", [
    P("איך יודעים אם מדד שייך לרגרסיה? שואלים מה הוא משווה."),
    P("מדד של רגרסיה משווה **מספר חזוי** ל**מספר אמיתי**: כמה רחוק החיזוי מהערך.", do=show("f-reg", "f-k"),
      say=f"מדד של רגרסיה משווה מספר חזוי למספר אמיתי: כמה רחוק החיזוי מהערך. {SAY['MSE']}, {SAY['RMSE']}, {SAY['MAE']}, "
          f"{SAY['SAE']}, {SAY['SSE']}, ו{SAY['R2']}: כולם בנויים מההפרש הזה."),
    P("דיוק וריקול סופרים כמה **תוויות** נחזו נכון. אין שם הפרש בין מספרים: אלה מדדי סיווג.", do=show("f-cls"),
      say="דיוק ורִיקוֹל סופרים כמה תוויות נחזו נכון. אין שם הפרש בין מספרים: אלה מדדי סיווג."),
    P("WSSE מודד כמה נקודות קרובות למרכז הקבוצה שלהן. אין שם ערך מטרה בכלל: זה אשכול, אף שהשם דומה ל-SSE.",
      do=show("f-clu", "f-t"),
      say=f"דַּאבְּלְיוּ אֶס אֶס אִי מודד כמה נקודות קרובות למרכז הקבוצה שלהן. אין שם ערך מטרה בכלל: זה אשכול, אף שהשם דומה ל{SAY['SSE']}.",
      hold=0.3),
], stage=f'<div class="rg-stack" style="gap:20px">{FAM_HTML}</div>',
   panel=w.card("f-k", "rule", "🔑 רגרסיה: הפרש בין מספר חזוי למספר אמיתי")
         + w.card("f-t", "trap", "⚠️ שם דומה לא אומר אותו נושא"))

FP = w.fit_plot("p", XS, YS, (0, 2), xr=(0, 6), yr=(0, 12), xticks=[1, 2, 3, 4, 5], yticks=[0, 2, 4, 6, 8, 10, 12],
                w=760, h=560, mean=6)
r2 = Scene("r2", "המדד R²", [
    P("נניח שה-SSE של מודל יצא 4. זה טוב? אי אפשר לדעת בלי נקודת השוואה.",
      say=f"נניח שה{SAY['SSE']} של מודל יצא ארבע. זה טוב? אי אפשר לדעת בלי נקודת השוואה."),
    P("נקודת ההשוואה היא המודל הכי פשוט שיש: לחזות תמיד את הממוצע. כאן הממוצע הוא 6.", do=show("p-mn", "c-mean"),
      say="נקודת ההשוואה היא המודל הכי פשוט שיש: לחזות תמיד את הממוצע. כאן הממוצע הוא שש."),
    P("ההפרשים מהממוצע: מינוס 3, מינוס 1, אפס, פלוס 1, פלוס 3.",
      do=show(*[f"p-s{i}" for i in range(5)], *[f"p-sl{i}" for i in range(5)]),
      say="ההפרשים מהממוצע: מינוס שלוש, מינוס אחת, אפס, פלוס אחת, פלוס שלוש."),
    P("בריבוע ובסכום: 9 ועוד 1 ועוד 0 ועוד 1 ועוד 9, כלומר 20. זה SST.", do=show("c-sst"),
      say=f"בריבוע ובסכום: תשע ועוד אחת ועוד אפס ועוד אחת ועוד תשע, כלומר עשרים. זה {SAY['SST']}.", hold=0.3),
    P("עכשיו המודל שלנו, $\\hat{y} = 2x$. השגיאות: מינוס 1, מינוס 1, אפס, פלוס 1, פלוס 1.",
      do=hide(*[f"p-s{i}" for i in range(5)], *[f"p-sl{i}" for i in range(5)])
      + show("p-ln", *[f"p-r{i}" for i in range(5)], *[f"p-rl{i}" for i in range(5)]),
      say="עכשיו המודל שלנו, וַואי כובע שווה שתיים אִקְס. השגיאות: מינוס אחת, מינוס אחת, אפס, פלוס אחת, פלוס אחת."),
    P("בריבוע ובסכום: 4. זה ה-SSE.", do=show("c-sse"), say=f"בריבוע ובסכום: ארבע. זה ה{SAY['SSE']}."),
    P("R² הוא 1 פחות היחס ביניהם: 1 פחות 4 חלקי 20, כלומר 0.8.", do=show("c-r2"),
      say=f"{SAY['R2']} הוא אחת פחות היחס ביניהם: אחת פחות ארבע חלקי עשרים, כלומר אפס נקודה שמונה.", hold=0.3),
    P("R² שווה 1: חיזוי מושלם. אפס: לא טוב יותר מהממוצע. וככל שהוא **גבוה** יותר, המודל טוב יותר.", do=show("c-k"),
      say=f"{SAY['R2']} שווה אחת: חיזוי מושלם. אפס: לא טוב יותר מהממוצע. וככל שהוא גבוה יותר, המודל טוב יותר."),
    P("בלי ה-1 פחות, יוצא 0.2: החלק של השגיאה שנשאר. זו טעות נפוצה.", do=show("c-t"),
      say="בלי האחת פחות, יוצא אפס נקודה שתיים: החלק של השגיאה שנשאר. זו טעות נפוצה."),
], stage=FP.svg(),
   panel=w.card("c-mean", "plain", "חיזוי הממוצע: $\\bar{y} = 6$")
         + w.card("c-sst", "ex", "$\\text{SST} = \\sum (y - \\bar{y})^2 = 20$")
         + w.card("c-sse", "ex", "$\\text{SSE} = \\sum (\\hat{y} - y)^2 = 4$")
         + w.card("c-r2", "rule", "🔑 $R^2 = 1 - \\frac{\\text{SSE}}{\\text{SST}} = 1 - \\frac{4}{20} = 0.8$")
         + w.card("c-k", "plain", "$R^2 = 1$ מושלם · $R^2 = 0$ כמו הממוצע")
         + w.card("c-t", "trap", "⚠️ בלי ה״אחת פחות״ יוצא 0.2: טעות"))

TAB = w.table("m", ["מודל", "$R^2$", "$\\text{RMSE}$"],
              [[k, w.fmt(float(MODELS[k])), RMSE_S[k]] for k in MODELS], hidden={(None, 1), (None, 2)}, size=40)
win = Scene("win", "מי ניצח?", [
    P("שלושה מודלים נבדקו על אותם נתוני בדיקה, בשני מדדים."),
    P("לפי R², הגבוה מנצח: מודל ב׳, עם 0.82.", do=show(*[f"m-{r}-1" for r in range(3)], "m-m1", "w-1"),
      say=f"לפי {SAY['R2']}, הגבוה מנצח: מודל בית, עם אפס נקודה שמונים ושתיים."),
    P("לפי RMSE, שהוא מדד שגיאה, **הנמוך** מנצח: שוב מודל ב׳, עם 1.9.", do=show(*[f"m-{r}-2" for r in range(3)], "w-2"),
      say=f"לפי {SAY['RMSE']}, שהוא מדד שגיאה, הנמוך מנצח: שוב מודל בית, עם אחת נקודה תשע."),
    P("זה לא מקרי: על אותם נתונים, שגיאה קטנה יותר פירושה R² גבוה יותר.",
      say=f"זה לא מקרי: על אותם נתונים, שגיאה קטנה יותר פירושה {SAY['R2']} גבוה יותר.", hold=0.3),
    P("אז הכלל: במדדי שגיאה הנמוך עדיף. ב-R² ובדיוק, הגבוה עדיף.", do=show("w-k"),
      say=f"אז הכלל: במדדי שגיאה הנמוך עדיף. ב{SAY['R2']} ובדיוק, הגבוה עדיף."),
], stage=TAB,
   panel=w.card("w-1", "ex", "$R^2$: הגבוה ← מודל ב׳")
         + w.card("w-2", "ex", "<span class=\"rg-en\">RMSE</span>: הנמוך ← מודל ב׳")
         + w.card("w-k", "rule", "🔑 מדדי שגיאה (<span class=\"rg-en\">MSE, RMSE, MAE, SSE</span>): נמוך עדיף<br>"
                  "$R^2$, דיוק: גבוה עדיף"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: על נתוני הבדיקה SST שווה 50, וה-SSE של המודל 60. מה נכון?",
      say=f"שאלה בסגנון מבחן: על נתוני הבדיקה {SAY['SST']} שווה חמישים, וה{SAY['SSE']} של המודל שישים. מה נכון?"),
    P("R² שווה 1 פחות 60 חלקי 50, כלומר 1 פחות 1.2: מינוס 0.2. שלילי!", do=show("x-1"),
      say=f"{SAY['R2']} שווה אחת פחות שישים חלקי חמישים, כלומר אחת פחות אחת נקודה שתיים: מינוס אפס נקודה שתיים. שלילי!"),
    P("SSE גדול מ-SST אומר שהמודל טועה יותר מהחיזוי הקבוע של הממוצע. זו התשובה.", do=show("x-2", f"q-ok-{OK}"),
      say=f"{SAY['SSE']} גדול מ{SAY['SST']} אומר שהמודל טועה יותר מהחיזוי הקבוע של הממוצע. זו התשובה.", hold=0.3),
    P("1.2 הוא היחס עצמו, בלי ה״אחת פחות״. ו-R² לעולם לא עובר את 1.", do=show(f"q-x-{RATIO}", f"q-n-{RATIO}"),
      say=f"אחת נקודה שתיים הוא היחס עצמו, בלי האחת פחות. ו{SAY['R2']} לעולם לא עובר את אחת."),
    P("0.2 מוחק את המינוס: R² חסום מלמעלה ב-1, אבל לא מלמטה.", do=show(f"q-x-{ABSV}", f"q-n-{ABSV}"),
      say=f"אפס נקודה שתיים מוחק את המינוס: {SAY['R2']} חסום מלמעלה באחת, אבל לא מלמטה."),
    P("ו״לא ייתכן״ נופלת: על נתונים חדשים, מודל גרוע יכול בהחלט לטעות יותר מהממוצע.",
      do=show(f"q-x-{IMPOS}", f"q-n-{IMPOS}"), hold=0.5),
], stage=w.qpanel("בסגנון מבחן", "מודל רגרסיה נבדק על נתוני בדיקה: <span class=\"rg-en\">SST = 50</span>, "
                  "<span class=\"rg-en\">SSE = 60</span>. מה נכון?", QOPTS,
                  notes={RATIO: "בלי ״אחת פחות״", ABSV: "המינוס נמחק", IMPOS: "ייתכן"}),
   panel=w.eq("x-1", "$R^2 = 1 - \\frac{60}{50} = -0.2$", size=46, cls="box")
         + w.card("x-2", "rule", "🔑 $R^2 < 0$ ⇔ גרוע מחיזוי הממוצע"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: מדד רגרסיה משווה מספר חזוי למספר אמיתי. מדד שסופר תוויות או מודד קבוצות שייך לנושא אחר.",
      do=show("r-1")),
    P("R² שווה 1 פחות SSE חלקי SST, ואסור לשכוח את ה-1 פחות.", do=show("r-2"),
      say=f"{SAY['R2']} שווה אחת פחות {SAY['SSE']} חלקי {SAY['SST']}, ואסור לשכוח את האחת פחות."),
    P("ובהשוואת מודלים: מדד שגיאה, הנמוך ניצח. R², הגבוה ניצח. נתראה ברמה הבאה.", do=show("r-3"), hold=0.6,
      say=f"ובהשוואת מודלים: מדד שגיאה, הנמוך ניצח. {SAY['R2']}, הגבוה ניצח. נתראה ברמה הבאה."),
], stage=w.eq("r-eq", "$R^2 = 1 - \\dfrac{\\text{SSE}}{\\text{SST}}$", size=64, hidden=False),
   panel=w.card("r-1", "rule", "🔑 רגרסיה: <span class=\"rg-en\">MSE · RMSE · MAE · SAE · SSE · R²</span>")
         + w.card("r-2", "trap", "⚠️ לא לשכוח את ה״אחת פחות״")
         + w.card("r-3", "rule", "🔑 שגיאה: נמוך עדיף · $R^2$: גבוה עדיף"))

VIDEO = Video("R02", "reg", "מדדי רגרסיה ו-R²", [title, fam, r2, win, exam, recap], bank={})
w.check_say(VIDEO)
