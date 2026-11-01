# -*- coding: utf-8 -*-
"""R01 · רגרסיה: חיזוי ושגיאה — video script (build: tools/video/vid.py R01).

Source: courses/reg/R01.js (goal: recognise a regression task, compute ŷ of a linear model, compute MAE / MSE / RMSE
without mixing them up). Exam part: an original "בסגנון מבחן" question (bank={}): MSE = 400 (thousands of ₪) -> RMSE = 20
thousand ₪. The real candidates were not usable: 26S-A-Q6 twins the quiz ref 25S-A-Q17 (MSE punishes large errors),
22B-A-Q19 is another course's video anchor.
Avoided on purpose: every R01 quiz item (flight delay, ŷ = 4 + 2x₁ − x₂, the 10/8/5 table and its RMSE, the two models
with errors 2,2,2,2 vs 0,0,0,8, logistic regression, the parabola), the quiz refs 25S-A-Q17 / PRAC-X-Q27 and their twins
(26S-A-Q6, SAMP-1-Q25). No outlier / "MSE punishes large errors" demo (that is 25S-A-Q17's key and R01-Q5's scenario);
the second property shown is the unit of each metric. Own data: the coffee model ŷ = 20 + 6x₁ − 0.5x₂, the line
ŷ = 2 + 2x on four points.
"""
from fractions import Fraction as Fr
import math

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
# the coffee model: cups = 20 + 6·(lectures) − 0.5·(temperature)
W0, W1, W2 = 20, 6, Fr(-1, 2)
X1, X2 = 5, 24
YH = W0 + W1 * X1 + W2 * X2
assert YH == 38 and W1 * X1 == 30 and W2 * X2 == -12
assert W1 * X1 + W2 * X2 == 18          # forgetting w0
assert W0 + W1 * X1 - W2 * X2 == 62     # losing the minus

# four examples, the line ŷ = 2 + 2x
XS, YS = [1, 2, 3, 4], [5, 5, 10, 8]
B, M = 2, 2
YHS = [B + M * x for x in XS]
E = [yh - y for yh, y in zip(YHS, YS)]
assert YHS == [4, 6, 8, 10] and E == [-1, 1, -2, 2]
assert sum(E) == 0                      # the plain mean is 0: the errors cancel
SAE = sum(abs(e) for e in E)
SSE = sum(e * e for e in E)
MAE, MSE = Fr(SAE, 4), Fr(SSE, 4)
RMSE = math.sqrt(MSE)
assert SAE == 6 and MAE == Fr(3, 2) and SSE == 10 and MSE == Fr(5, 2)
assert round(RMSE, 2) == 1.58 and round(math.sqrt(SSE), 2) == 3.16

# units: MAE and RMSE are in the target's unit, MSE in its square. The exam part: MSE = 400 -> RMSE = 20
QMSE = 400
assert math.isqrt(QMSE) ** 2 == QMSE and math.isqrt(QMSE) == 20 and QMSE / 2 == 200
QOPTS = [(0, "20 אלפי ₪"), (1, "400 אלפי ₪"), (2, "20, ביחידות של (אלפי ₪) בריבוע"), (3, "200 אלפי ₪")]
OK, NOROOT, SQU, HALF = 0, 1, 2, 3

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נכיר את משימת הרגרסיה: מודל שחוזה מספר."),
    P("נחשב חיזוי של מודל לינארי, ונמדוד כמה הוא טועה, בשלוש דרכים שונות."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">R01 · רגרסיה ורשתות נוירונים</div><h1>רגרסיה: חיזוי ושגיאה</h1>'
         '<p>לזהות משימת רגרסיה, לחשב חיזוי של מודל לינארי, ולחשב <span class="rg-en">MAE</span>, '
         '<span class="rg-en">MSE</span> ו-<span class="rg-en">RMSE</span> בלי להתבלבל ביניהם.</p>', layout="title")

CASES = [
    ("c1", "כמה כוסות קפה יימכרו מחר בקפיטריה?", "מספר", "רגרסיה", "ok"),
    ("c2", "האם הלקוח יחזור לקנות שוב?", "כן / לא", "סיווג", "info"),
    ("c4", "באיזה ז׳אנר הסרט: קומדיה, דרמה או אימה?", "קטגוריה", "סיווג", "info"),
]
CASE_ROWS = "".join(
    f'<div id="{i}" class="h rg-case"><span class="rg-cq">{q}</span><span class="rg-mut">{t}</span>'
    f'<span class="rg-chip {k}">{v}</span></div>' for i, q, t, v, k in CASES)

idea = Scene("idea", "רגרסיה או סיווג", [
    P("רגרסיה היא משימה שבה המודל חוזה **מספר**: מחיר, זמן, כמות."),
    P("את מה שחוזים קוראים **ערך המטרה**, ואת הנתונים שמהם חוזים קוראים **מאפיינים**.", do=show("i-def"),
      say=f"את מה שחוזים קוראים ערך המטרה, או {SAY['target']}, ואת הנתונים שמהם חוזים קוראים מאפיינים."),
    P("כמה כוסות קפה יימכרו מחר? התשובה היא מספר, ולכן זו רגרסיה.", do=show("c1")),
    P("האם הלקוח יחזור לקנות? התשובה היא כן או לא, כלומר קטגוריה. זה סיווג.", do=show("c2")),
    P("ז׳אנר של סרט: קומדיה, דרמה או אימה. קטגוריה, ולכן סיווג.", do=show("c4"), hold=0.3),
    P("אז שואלים שאלה אחת: ערך המטרה הוא מספר רציף, או קטגוריה?", do=show("i-r")),
], stage=f'<div class="rg-stack" style="gap:20px">{CASE_ROWS}</div>',
   panel=w.card("i-def", "plain", "<b>ערך המטרה</b> (<span class=\"rg-en\">target</span>, $y$): מה שחוזים<br>"
                "<b>מאפיינים</b> (<span class=\"rg-en\">features</span>, $x_1, x_2, \\dots$): ממה שחוזים")
         + w.card("i-r", "rule", "🔑 ערך מטרה מספרי ← <b>רגרסיה</b><br>ערך מטרה קטגוריאלי ← <b>סיווג</b>"))

model = Scene("model", "המודל הלינארי", [
    P("רגרסיה לינארית חוזה בעזרת **סכום משוקלל** של המאפיינים.", do=show("m-eq")),
    P("$\\hat{y}$ הוא החיזוי. $y$ בלי כובע הוא הערך האמיתי.", do=show("m-c1"),
      say="וַואי כובע הוא החיזוי. וַואי בלי כובע הוא הערך האמיתי."),
    P("כל מאפיין מוכפל ב**משקולת** שלו, ו-$w_0$ הוא **האיבר החופשי**: החיזוי כשכל המאפיינים אפס.",
      do=show("m-c2", "m-c3"),
      say="כל מאפיין מוכפל במשקולת שלו, ודַאבְּלְיוּ אפס הוא האיבר החופשי: החיזוי כשכל המאפיינים אפס.", hold=0.3),
    P("דוגמה: כוסות קפה, לפי מספר ההרצאות בקמפוס והטמפרטורה. נציב חמש הרצאות ועשרים וארבע מעלות.",
      do=show("m-ex", "m-s1")),
    P("עשרים, ועוד שש כפול חמש שזה שלושים, ועוד מינוס חצי כפול עשרים וארבע, שזה מינוס שתים עשרה.", do=show("m-s2")),
    P("החיזוי: **38** כוסות.", do=show("m-s3"), say="החיזוי: שלושים ושמונה כוסות.", hold=0.3),
    P("בלי האיבר החופשי היה יוצא 18, ובלי המינוס של המשקולת השנייה, 62. שתי טעויות נפוצות.", do=show("m-t"),
      say="בלי האיבר החופשי היה יוצא שמונה עשרה, ובלי המינוס של המשקולת השנייה, שישים ושתיים. שתי טעויות נפוצות."),
], stage=w.stack(
    w.eq("m-eq", "$\\hat{y} = w_0 + w_1 x_1 + w_2 x_2$", size=56),
    w.eq("m-ex", "$\\hat{y} = 20 + 6 x_1 - 0.5 x_2$", size=48, cls="box"),
    w.eq("m-s1", "$x_1 = 5,\\quad x_2 = 24$", size=40),
    w.eq("m-s2", "$\\hat{y} = 20 + 30 - 12$", size=44),
    w.eq("m-s3", "$\\hat{y} = 38$", size=56, cls="rg-ok"), gap=24),
   panel=w.card("m-c1", "plain", "$\\hat{y}$: החיזוי &nbsp;·&nbsp; $y$: הערך האמיתי")
         + w.card("m-c2", "ex", "$w_1, w_2$: המשקולות, כמה כל מאפיין משפיע")
         + w.card("m-c3", "ex", "$w_0$: האיבר החופשי (<span class=\"rg-en\">bias</span>)")
         + w.card("m-t", "trap", "⚠️ בלי $w_0$: 18 &nbsp;·&nbsp; בלי המינוס: 62"))

FP = w.fit_plot("f", XS, YS, (B, M), xr=(0, 5), yr=(0, 12), xticks=[1, 2, 3, 4], yticks=[0, 2, 4, 6, 8, 10, 12],
                w=760, h=560)
TAB = w.table("t", ["$x$", "$y$", "$\\hat{y}$", "$\\hat{y} - y$"],
              [[x, y, yh, ("+" if e > 0 else "") + w.fmt(e)] for x, y, yh, e in zip(XS, YS, YHS, E)],
              hidden={(None, 2), (None, 3)}, size=34)

errors = Scene("err", "השגיאה", [
    P("עם מאפיין אחד, החיזוי הוא **קו ישר**. הנה המודל $\\hat{y} = 2 + 2x$, וארבע דוגמאות.", do=show("f-ln"),
      say="עם מאפיין אחד, החיזוי הוא קו ישר. הנה המודל וַואי כובע שווה שתיים ועוד שתיים אִקְס, וארבע דוגמאות."),
    P("השגיאה היא החיזוי פחות האמיתי. בדוגמה הראשונה: 4 פחות 5, כלומר מינוס 1.",
      do=show("e-def", "f-r0", "f-rl0", "t-0-2", "t-0-3", "t-m0"),
      say="השגיאה היא החיזוי פחות הערך האמיתי. בדוגמה הראשונה: ארבע פחות חמש, כלומר מינוס אחת."),
    P("בשנייה: 6 מול 5, שגיאה פלוס 1.", do=hide("t-m0") + show("f-r1", "f-rl1", "t-1-2", "t-1-3", "t-m1"),
      say="בשנייה: שש מול חמש, שגיאה פלוס אחת."),
    P("בשלישית: 8 מול 10, מינוס 2. וברביעית: 10 מול 8, פלוס 2.",
      do=hide("t-m1") + show("f-r2", "f-rl2", "t-2-2", "t-2-3", "f-r3", "f-rl3", "t-3-2", "t-3-3"),
      say="בשלישית: שמונה מול עשר, מינוס שתיים. וברביעית: עשר מול שמונה, פלוס שתיים.", hold=0.3),
    P("איך מסכמים ארבע שגיאות למספר אחד? הרעיון הראשון: ממוצע."),
    P("אבל מינוס 1, ועוד 1, מינוס 2 ועוד 2, זה **אפס**: כאילו המודל מושלם. השגיאות **מתקזזות**.", do=show("e-t"),
      say="אבל מינוס אחת, ועוד אחת, מינוס שתיים ועוד שתיים, זה אפס: כאילו המודל מושלם. השגיאות מתקזזות."),
    P("לכן מוחקים את הסימן: או בערך מוחלט, או בריבוע.", do=show("e-r"), hold=0.3),
], stage=FP.svg(),
   panel=w.stack(TAB, w.card("e-def", "plain", "השגיאה: $e = \\hat{y} - y$"),
                 w.card("e-t", "trap", "⚠️ ממוצע עם הסימן: $({-1} + 1 - 2 + 2) / 4 = 0$"),
                 w.card("e-r", "rule", "🔑 ערך מוחלט $|e|$, או ריבוע $e^2$"), gap=16))

MT = w.table("mt", ["$e$", "$|e|$", "$e^2$"],
             [[w.fmt(e), abs(e), e * e] for e in E] + [["סכום", SAE, SSE]],
             hidden={(None, 1), (None, 2)}, size=36, cls="sum")
metrics = Scene("met", "שלושה מדדים", [
    P("בערך מוחלט השגיאות הן 1, 1, 2, 2. הסכום: 6.", do=show(*[f"mt-{r}-1" for r in range(5)]),
      say="בערך מוחלט השגיאות הן אחת, אחת, שתיים, שתיים. הסכום: שש."),
    P("מחלקים במספר הדוגמאות, 4, ומקבלים את ה-MAE: 1.5.", do=show("k-mae"),
      say=f"מחלקים במספר הדוגמאות, ארבע, ומקבלים את ה{SAY['MAE']}: אחת נקודה חמש."),
    P("בריבוע השגיאות הן 1, 1, 4, 4. הסכום: 10.", do=show(*[f"mt-{r}-2" for r in range(5)]),
      say="בריבוע השגיאות הן אחת, אחת, ארבע, ארבע. הסכום: עשר."),
    P("חלקי 4: 2.5. זה ה-MSE, השגיאה הריבועית הממוצעת.", do=show("k-mse"),
      say=f"חלקי ארבע: שתיים נקודה חמש. זה ה{SAY['MSE']}, השגיאה הריבועית הממוצעת."),
    P("ו-RMSE הוא השורש שלו: שורש של 2.5, בערך 1.58.", do=show("k-rmse"),
      say=f"וה{SAY['RMSE']} הוא השורש שלו: שורש של שתיים נקודה חמש, בערך אחת נקודה חמישים ושמונה.", hold=0.3),
    P("ה-RMSE הוא שורש של **הממוצע**. שורש של הסכום עשר נותן 3.16, וזה כבר מספר אחר.", do=show("k-ord", "k-t"),
      say=f"ה{SAY['RMSE']} הוא שורש של הממוצע. שורש של הסכום עשר נותן שלוש נקודה שש עשרה, וזה כבר מספר אחר."),
], stage=MT,
   panel=w.card("k-mae", "ex", "$\\text{MAE} = 6 / 4 = 1.5$")
         + w.card("k-mse", "ex", "$\\text{MSE} = 10 / 4 = 2.5$")
         + w.card("k-rmse", "ex", "$\\text{RMSE} = \\sqrt{2.5} \\approx 1.58$")
         + w.card("k-ord", "rule", "🔑 <span class=\"rg-en\">RMSE</span>: שורש של ה-<span class=\"rg-en\">MSE</span>")
         + w.card("k-t", "trap", "⚠️ $\\sqrt{10} \\approx 3.16$: שכחו לחלק"))

UT = w.table("u", ["מדד", "ערך", "יחידה"], [["<span class=\"rg-en\">MAE</span>", "1.5", "כוסות"],
                                            ["<span class=\"rg-en\">MSE</span>", "2.5", "כוסות²"],
                                            ["<span class=\"rg-en\">RMSE</span>", "1.58", "כוסות"]],
             hidden={(None, 2)}, size=38)
units = Scene("units", "היחידות", [
    P("עוד הבדל בין המדדים: היחידות. נניח שהמודל חוזה כוסות קפה, ולכן כל שגיאה נמדדת בכוסות."),
    P("ה-MAE הוא ממוצע של שגיאות, ולכן גם הוא בכוסות: בממוצע, טעות של כוס וחצי.", do=show("u-0-2"),
      say=f"ה{SAY['MAE']} הוא ממוצע של שגיאות, ולכן גם הוא בכוסות: בממוצע, טעות של כוס וחצי."),
    P("ה-MSE מעלה כל שגיאה בריבוע, ולכן 2.5 הוא בכוסות **בריבוע**: יחידה שקשה לפרש.", do=show("u-1-2"),
      say=f"ה{SAY['MSE']} מעלה כל שגיאה בריבוע, ולכן שתיים נקודה חמש הוא בכוסות בריבוע: יחידה שקשה לפרש."),
    P("השורש מחזיר ליחידה המקורית: RMSE של 1.58 **כוסות**. זה הצד החזק של ה-RMSE.", do=show("u-2-2", "u-k"),
      say=f"השורש מחזיר ליחידה המקורית: {SAY['RMSE']} של אחת נקודה חמישים ושמונה כוסות. זה הצד החזק של ה{SAY['RMSE']}.",
      hold=0.3),
], stage=UT,
   panel=w.card("u-k", "rule", "🔑 שורש של יחידה בריבוע ← היחידה המקורית"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: מודל חוזה מחיר דירה, באלפי שקלים. על נתוני הבדיקה התקבל MSE של 400. מה ה-RMSE?",
      say=f"שאלה בסגנון מבחן: מודל חוזה מחיר דירה, באלפי שקלים. על נתוני הבדיקה התקבל {SAY['MSE']} של ארבע מאות. מה ה{SAY['RMSE']}?"),
    P("RMSE הוא השורש של ה-MSE: שורש של 400 הוא 20.", do=show("x-1"),
      say=f"ה{SAY['RMSE']} הוא השורש של ה{SAY['MSE']}: שורש של ארבע מאות הוא עשרים."),
    P("וה-MSE היה באלפי שקלים בריבוע, אז השורש מחזיר לאלפי שקלים: 20 אלף שקל. זו התשובה.", do=show("x-2", f"q-ok-{OK}"),
      say=f"וה{SAY['MSE']} היה באלפי שקלים בריבוע, אז השורש מחזיר לאלפי שקלים: עשרים אלף שקל. זו התשובה.", hold=0.3),
    P("400 הוא ה-MSE עצמו, בלי השורש.", do=show(f"q-x-{NOROOT}", f"q-n-{NOROOT}"),
      say=f"ארבע מאות הוא ה{SAY['MSE']} עצמו, בלי השורש."),
    P("20 ביחידות בריבוע: המספר נכון, אבל השורש כבר הוריד את הריבוע מהיחידה.", do=show(f"q-x-{SQU}", f"q-n-{SQU}"),
      say="עשרים ביחידות בריבוע: המספר נכון, אבל השורש כבר הוריד את הריבוע מהיחידה."),
    P("ו-200 הוא חצי של 400, לא שורש.", do=show(f"q-x-{HALF}", f"q-n-{HALF}"),
      say="ומאתיים הוא חצי של ארבע מאות, לא שורש.", hold=0.5),
], stage=w.qpanel("בסגנון מבחן", "מודל רגרסיה חוזה מחיר דירה <b>באלפי ₪</b>. על נתוני הבדיקה התקבל "
                  "<span class=\"rg-en\">MSE = 400</span>. מה ה-<span class=\"rg-en\">RMSE</span>?", QOPTS,
                  notes={NOROOT: "בלי שורש", SQU: "יחידה שגויה", HALF: "חצי, לא שורש"}),
   panel=w.eq("x-1", "$\\text{RMSE} = \\sqrt{400} = 20$", size=44, cls="box")
         + w.card("x-2", "ex", "(אלפי ₪)² ← שורש ← אלפי ₪"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: ערך מטרה מספרי, רגרסיה. קטגוריה, סיווג.", do=show("r-1")),
    P("החיזוי הוא סכום משוקלל, ולא שוכחים את האיבר החופשי.", do=show("r-2")),
    P("שגיאה היא החיזוי פחות הערך האמיתי, ובלי ערך מוחלט או ריבוע השגיאות מתקזזות.", do=show("r-3")),
    P("ממוצע של ערך מוחלט, ממוצע של ריבוע, ושורש שמחזיר ליחידה המקורית. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=w.stack(w.eq("r-eq", "$\\hat{y} = w_0 + w_1 x_1 + w_2 x_2$", size=54, hidden=False),
                 w.eq("r-m", "<span class=\"rg-en\">MAE</span> $= \\frac{1}{n}\\sum |e|$ &nbsp;&nbsp; "
                      "<span class=\"rg-en\">MSE</span> $= \\frac{1}{n}\\sum e^2$ &nbsp;&nbsp; "
                      "<span class=\"rg-en\">RMSE</span> $= \\sqrt{\\text{MSE}}$", size=36, hidden=False), gap=40),
   panel=w.card("r-1", "rule", "🔑 מספר ← רגרסיה · קטגוריה ← סיווג")
         + w.card("r-2", "rule", "🔑 $\\hat{y} = w_0 + \\sum w_i x_i$, כולל $w_0$")
         + w.card("r-3", "trap", "⚠️ שגיאות עם סימן מתקזזות")
         + w.card("r-4", "rule", "🔑 <span class=\"rg-en\">MAE, RMSE</span>: ביחידות של ה-<span class=\"rg-en\">target</span>"))

VIDEO = Video("R01", "reg", "רגרסיה: חיזוי ושגיאה", [title, idea, model, errors, metrics, units, exam, recap],
              bank={})
w.check_say(VIDEO)
