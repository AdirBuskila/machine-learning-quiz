# -*- coding: utf-8 -*-
"""R08 · רשתות: שכבות, משקולות ואימון — video script (build: tools/video/vid.py R08).

Source: courses/reg/R08.js (goal: count the weights of a network including w₀, explain why a non-linear activation is
needed, and answer the training questions: backprop, perceptron vs logistic, Dropout). Exam anchor: 22B-C-Q6 (a sign
network whose four input weights are 0: which explanation cannot explain the model -> "a learning rate too small to
converge"), used by no guided quiz; shown as the original raster.
Not used on purpose: SAMP-1-Q15 (its key "K = 5 -> 135 weights" does not match the architecture: K -> 5 -> 5 has
5K + 35 = 60 weights for K = 5). Avoided: every R08 quiz item (6 -> 4 -> 1, 10 -> 3 -> 2, f(net) = 3·net + 1, the circle,
w = (0, 2, −1) with x = (1, 1), 50 vs 5 layers, Dropout's definition as an option), the boss 3 -> 4 -> 4 -> 2, the quiz refs
23S-A-Q16, SAMP-2-Q20, 25B-A-Q1 and their twins (24B-A-Q18, 24B-C-Q4, 24S-A-Q16, SAMP-3-Q20, 22B-A-Q5, 23B-A-Q13,
23B-B-Q13, 25B-A-Q2, 25S-A-Q11, 22B-A-Q4, 22B-C-Q3, 25S-A-Q14). Shown as demos only, never stated as rules: the linear
"network" collapsing to 5x + 4 (no "linear activation = a single perceptron", no "the activation adds non-linearity"),
the logistic update table (no "Δw ∝ input", no "logistic always updates / perceptron only on mistakes"), Dropout's
mechanism (no purpose, no L2 comparison, no "deeper is not always better").
Own data: a 2 -> 3 -> 1 network (13 weights, 9 without w₀); a linear "network" h₁ = 2x + 1, h₂ = −x + 3,
out = 3h₁ + h₂ − 2 = 5x + 4 (sympy), and the same with sigmoid hidden units (a curve); a logistic update with
w = (0.5, −1, 1), x = (2, 3), y = 1, η = 0.5.
"""
from fractions import Fraction as Fr
import math

import sympy as sp

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
LAYERS = [2, 3, 1]
edges = sum(a * b for a, b in zip(LAYERS, LAYERS[1:]))
w0s = sum(LAYERS[1:])
assert (edges, w0s, edges + w0s) == (9, 4, 13)
assert LAYERS[0] * LAYERS[1] + LAYERS[1] == 9 and LAYERS[1] * LAYERS[2] + LAYERS[2] == 4

x = sp.symbols("x")
h1, h2 = 2 * x + 1, -x + 3
lin = sp.expand(3 * h1 + h2 - 2)
assert lin == 5 * x + 4
s = lambda t: 1 / (1 + sp.exp(-t))
curv = 3 * s(h1) + s(h2) - 2
assert sp.diff(curv, x, 2).subs(x, 0).evalf() != 0                     # not a line
fcurv = sp.lambdify(x, curv, "math")

# logistic update (the course's rule): δ = (y − ŷ)·ŷ·(1 − ŷ), Δwᵢ = η·δ·xᵢ with x₀ = 1
LW, LX, LY, ETA = [0.5, -1.0, 1.0], [1, 2, 3], 1, 0.5
lnet = sum(a * b for a, b in zip(LW, LX))
yh = 1 / (1 + math.exp(-lnet))
delta = (LY - yh) * yh * (1 - yh)
dw = [ETA * delta * xi for xi in LX]
neww = [a + b for a, b in zip(LW, dw)]
assert lnet == 1.5 and round(yh, 4) == 0.8176 and round(delta, 4) == 0.0272
assert [round(v, 4) for v in dw] == [0.0136, 0.0272, 0.0408]
assert [round(v, 4) for v in neww] == [0.5136, -0.9728, 1.0408]
assert math.isclose(dw[2] / dw[0], 3) and math.isclose(dw[1] / dw[0], 2)      # proportional to the inputs 1, 2, 3
assert (1 if lnet > 0 else -1) == LY                                          # sign would be right: (y − ŷ) = 0, no update

# exam 22B-C-Q6: sign everywhere, input weights 0, hidden w₀ = 0.5, output weights 1, 1 and w₀ = 1
sign = lambda t: 1 if t > 0 else -1
for i1 in (-3, 0, 2.5):
    for i2 in (-1, 4):
        hh = [sign(0.5 + 0 * i1 + 0 * i2), sign(0.5 + 0 * i1 + 0 * i2)]
        assert hh == [1, 1] and sign(1 + hh[0] + hh[1]) == 1 and 1 + hh[0] + hh[1] == 3
Q = BANK["22B-C-Q6"]
OK = w.opt_index(Q, "קבוע למידה קטן")
NOREL = w.opt_index(Q, "אין תלות")
CONST = w.opt_index(Q, "target value קבוע")
VAR0 = w.opt_index(Q, "variance")
assert Q["correctIndex"] == OK == 2 and sorted([OK, NOREL, CONST, VAR0]) == [0, 1, 2, 3]
OPTS = w.opts(Q)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נחבר נוירונים לרשת."),
    P("נספור את המשקולות שלה, נראה מה קורה לרשת בלי אקטיבציה, ונחשב עדכון אחד של אימון."),
    P("ובסוף נפתור שאלה אמיתית ממבחן, עם איור של רשת.", hold=0.4),
], stage='<div class="kicker">R08 · רגרסיה ורשתות נוירונים</div><h1>רשתות: שכבות, משקולות ואימון</h1>'
         '<p>לספור משקולות כולל <span class="rg-en">w₀</span>, לראות רשת בלי אקטיבציה, ולחשב עדכון של אימון.</p>',
   layout="title")

NN = w.net("nw", [[("i1", "x₁"), ("i2", "x₂")], [("h1", "h₁"), ("h2", "h₂"), ("h3", "h₃")], [("o", "ŷ")]],
           [(a, b, "") for a in ("i1", "i2") for b in ("h1", "h2", "h3")] + [(b, "o", "") for b in ("h1", "h2", "h3")],
           w=760, h=540, node_cls={"o": "out"}, values={k: "+w₀" for k in ("h1", "h2", "h3", "o")})
IH = [f"nw-he-{a}-{b}" for a in ("i1", "i2") for b in ("h1", "h2", "h3")]
HO = [f"nw-he-{b}-o" for b in ("h1", "h2", "h3")]
count = Scene("count", "ספירת משקולות", [
    P("רשת נוירונים היא נוירונים מסודרים ב**שכבות**: קלט, שכבה חבויה, ופלט. הקלט רק מעביר את ערכי המאפיינים.",
      do=show("k-l")),
    P("ברשת מחוברת במלואה, כל נוירון מקבל קשת מכל הנוירונים בשכבה שלפניו.",
      say="ברשת מחוברת במלואה, פוּלִי קוֹנֶקְטֶד, כל נוירון מקבל קשת מכל הנוירונים בשכבה שלפניו.", hold=0.3),
    P("כמה משקולות? כל קשת היא משקולת. מהקלט לשכבה החבויה: 2 כפול 3, שש קשתות.", do=show(*IH),
      say="כמה משקולות? כל קשת היא משקולת. מהקלט לשכבה החבויה: שתיים כפול שלוש, שש קשתות."),
    P("ולכל נוירון חבוי יש גם $w_0$ משלו: עוד 3. ביחד 9.", do=show("nw-v-h1", "nw-v-h2", "nw-v-h3", "k-1"),
      say="ולכל נוירון חבוי יש גם דַאבְּלְיוּ אפס משלו: עוד שלוש. ביחד תשע."),
    P("לנוירון הפלט: 3 קשתות, ועוד $w_0$ אחד. ארבע.", do=hide(*IH) + show(*HO, "nw-v-o", "k-2"),
      say="לנוירון הפלט: שלוש קשתות, ועוד דַאבְּלְיוּ אפס אחד. ארבע."),
    P("סך הכול **13**. בלי ה-$w_0$ יוצא 9, וזו הטעות שהמבחנים מחכים לה. ולנוירוני הקלט אין משקולות בכלל.",
      do=show("k-3", "k-t"),
      say="סך הכול שלוש עשרה. בלי הדַאבְּלְיוּ אפס יוצא תשע, וזו הטעות שהמבחנים מחכים לה. ולנוירוני הקלט אין משקולות בכלל."),
], stage=NN,
   panel=w.card("k-l", "plain", "קלט ← חבויה ← פלט")
         + w.card("k-1", "ex", "קלט←חבויה: $2 \\cdot 3 + 3 = 9$")
         + w.card("k-2", "ex", "חבויה←פלט: $3 \\cdot 1 + 1 = 4$")
         + w.card("k-3", "rule", "🔑 סה״כ $13$ (בלי $w_0$: $9$)")
         + w.card("k-t", "trap", "⚠️ שכבת הקלט: בלי משקולות"))

PL = w.Plot("lc", (-4, 3), (-8, 20), w=620, h=470, xticks=[-4, -2, 0, 2], yticks=[-5, 0, 5, 10, 15, 20], xlabel="x",
            ylabel="ŷ", pad=(70, 20, 24, 56))
PL.fn(lambda v: 5 * v + 4, -4, 3, kind="k-b", id_="lin", hidden=True, width=5)
PL.fn(fcurv, -4, 3, kind="k-ok", id_="cur", hidden=True, width=5)
lin_s = Scene("lin", "רשת בלי אקטיבציה", [
    P("למה צריך פונקציית אקטיבציה? נבנה רשת קטנה, שבה האקטיבציה החבויה לא עושה כלום: מעבירה את net כמו שהוא.",
      say="למה צריך פונקציית אקטיבציה? נבנה רשת קטנה, שבה האקטיבציה החבויה לא עושה כלום: מעבירה את הנֶט כמו שהוא."),
    P("נוירון חבוי אחד מחשב $2x + 1$, והשני $-x + 3$. הפלט: 3 כפול הראשון, ועוד השני, פחות 2.", do=show("l-1"),
      say="נוירון חבוי אחד מחשב שתיים אִקְס ועוד אחת, והשני מינוס אִקְס ועוד שלוש. הפלט: שלוש כפול הראשון, ועוד השני, פחות שתיים."),
    P("נפתח סוגריים: $6x + 3 - x + 3 - 2$. יוצא $5x + 4$: **קו ישר אחד**.", do=show("l-2", "lc-lin"),
      say="נפתח סוגריים: שש אִקְס ועוד שלוש, פחות אִקְס ועוד שלוש, פחות שתיים. יוצא חמש אִקְס ועוד ארבע: קו ישר אחד."),
    P("ועכשיו אותה רשת בדיוק, עם סיגמואיד בנוירונים החבויים. הפעם הפלט כבר מתעקם.", do=show("lc-cur", "l-3"),
      say=f"ועכשיו אותה רשת בדיוק, עם {SAY['sigmoid']} בנוירונים החבויים. הפעם הפלט כבר מתעקם.", hold=0.3),
], stage=PL.svg(),
   panel=w.eq("l-1", "$\\hat{y} = 3(2x + 1) + (-x + 3) - 2$", size=36, cls="box")
         + w.eq("l-2", "$\\hat{y} = 5x + 4$", size=44)
         + w.card("l-3", "ex", "עם $\\sigma$: $\\hat{y} = 3\\sigma(2x + 1) + \\sigma(-x + 3) - 2$"))

UPD = w.table("up", ["", "$x_i$", "$\\Delta w_i = \\eta \\delta x_i$", "$w_i$ חדש"],
              [["$w_0$", "1", "0.0136", "0.5136"], ["$w_1$", "2", "0.0272", "−0.9728"], ["$w_2$", "3", "0.0408", "1.0408"]],
              hidden={(None, 2), (None, 3)}, size=32)
train = Scene("train", "עדכון באימון", [
    P("באימון, הטעות שמחושבת בפלט מחלחלת אחורה אל הקשתות. זה נקרא בֶּק פְּרוֹפָּגֵיישֶׁן.",
      say="באימון, הטעות שמחושבת בפלט מחלחלת אחורה, אל הקשתות. זה נקרא בֶּק פְּרוֹפָּגֵיישֶׁן."),
    P("נראה את זה בנוירון אחד עם סיגמואיד, כמו ברגרסיה לוגיסטית: $w = (0.5, -1, 1)$, $x = (2, 3)$, והמחלקה 1.",
      do=show("t-0"),
      say=f"נראה את זה בנוירון אחד עם {SAY['sigmoid']}, כמו ברגרסיה לוגיסטית: דַאבְּלְיוּ שווה אפס נקודה חמש, מינוס אחת, ואחת. "
          "אִקְס שווה שתיים ושלוש, והמחלקה אחת."),
    P("net שווה 1.5, והסיגמואיד נותנת 0.82. הטעות דלתא: האמת פחות החיזוי, כפול החיזוי, כפול אחת פחות החיזוי. בערך 0.027.",
      do=show("t-1", "t-2"),
      say=f"הנֶט שווה אחת נקודה חמש, וה{SAY['sigmoid']} נותנת אפס נקודה שמונים ושתיים. הטעות, דלתא: האמת פחות החיזוי, כפול החיזוי, "
          "כפול אחת פחות החיזוי. בערך אפס נקודה אפס שתיים שבע."),
    P("כל משקולת זזה באטה, כפול דלתא, כפול **הקלט שלה**. עם אטה חצי, התיקונים בטבלה.", do=show(*[f"up-{r}-2" for r in range(3)]),
      say="כל משקולת זזה באֵטָה, כפול דלתא, כפול הקלט שלה. עם אֵטָה חצי, התיקונים בטבלה."),
    P("המשקולות החדשות: 0.5136, מינוס 0.9728 ו-1.0408. שימו לב שגם $w_0$ זזה, דרך הקלט הפיקטיבי 1.",
      do=show(*[f"up-{r}-3" for r in range(3)]),
      say="המשקולות החדשות: אפס נקודה חמש אחת שלוש שש, מינוס אפס נקודה תשע שבע שתיים שמונה, ואחת נקודה אפס ארבע אפס שמונה. "
          "שימו לב שגם דַאבְּלְיוּ אפס זזה, דרך הקלט הפיקטיבי אחת.", hold=0.3),
], stage=w.stack(w.eq("t-0", "$w = (0.5, {-1}, 1), \\quad (x_0, x_1, x_2) = (1, 2, 3), \\quad y = 1$", size=34),
                 w.eq("t-1", "$\\text{net} = 1.5, \\quad \\hat{y} = \\sigma(1.5) \\approx 0.82$", size=36),
                 w.eq("t-2", "$\\delta = (y - \\hat{y}) \\, \\hat{y} \\, (1 - \\hat{y}) \\approx 0.027$", size=36), UPD, gap=18),
   panel=w.card("t-f", "rule", "🔑 $\\Delta w_i = \\eta \\cdot \\delta \\cdot x_i$, עם $x_0 = 1$", hidden=False))

DROP = w.net("dr", [[("i1", "x₁"), ("i2", "x₂"), ("i3", "x₃")], [("a", ""), ("b", ""), ("c", ""), ("d", "")], [("o", "ŷ")]],
             [(a, b, "") for a in ("i1", "i2", "i3") for b in ("a", "b", "c", "d")] + [(b, "o", "") for b in ("a", "b", "c", "d")],
             w=640, h=440, node_cls={"o": "out"}, values={"b": "✕", "d": "✕"})
over = Scene("over", "Dropout", [
    P("ומה קורה בשיטת Dropout? כך נראה צעד אימון אחד: מתוך ארבעת הנוירונים החבויים, שניים כבויים.",
      do=show("dr-v-b", "dr-v-d", "o-1"),
      say=f"ומה קורה בשיטת {SAY['dropout']}? כך נראה צעד אימון אחד: מתוך ארבעת הנוירונים החבויים, שניים כבויים."),
    P("בצעד הבא מגרילים מחדש אילו נוירונים לכבות. ובזמן החיזוי, אחרי האימון, כולם פעילים.", hold=0.3),
], stage=DROP,
   panel=w.card("o-1", "ex", "צעד אימון אחד: 2 מתוך 4 נוירונים חבויים כבויים"))

IMG = st.raster("fig", "images/exams/22B-C-Q6.png", 600, (952, 572),
                rings=[("a", 385, 145, 42), ("b", 343, 222, 42), ("c", 318, 330, 42), ("d", 372, 390, 42),
                       ("h1", 525, 230, 48), ("h2", 527, 345, 48), ("o", 820, 275, 52)])
exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: רשת עם sign בכל הנוירונים, שני קלטים, שני נוירונים חבויים ונוירון פלט. זה המודל שהתקבל.",
      say=f"שאלה ממבחן: רשת עם {SAY['sign']} בכל הנוירונים, שני קלטים, שני נוירונים חבויים, ונוירון פלט. זה המודל שהתקבל."),
    P("ארבע המשקולות מהקלטים לשכבה החבויה הן **אפס**.", do=show("fig-ring-a", "fig-ring-b", "fig-ring-c", "fig-ring-d")),
    P("לכן ה-net של כל נוירון חבוי הוא רק ה-$w_0$ שלו, 0.5. ו-sign של 0.5 היא פלוס 1, לכל קלט שהוא.",
      do=hide("fig-ring-a", "fig-ring-b", "fig-ring-c", "fig-ring-d") + show("fig-ring-h1", "fig-ring-h2", "e-1"),
      say=f"לכן הנֶט של כל נוירון חבוי הוא רק הדַאבְּלְיוּ אפס שלו, חצי. ו{SAY['sign']} של חצי היא פלוס אחת, לכל קלט שהוא."),
    P("בפלט: 1, ועוד 1, ועוד 1. net שווה 3, והפלט פלוס 1. המודל **תמיד** חוזה חיובי, ומתעלם מהקלט.",
      do=hide("fig-ring-h1", "fig-ring-h2") + show("fig-ring-o", "e-2"),
      say="בפלט: אחת, ועוד אחת, ועוד אחת. הנֶט שווה שלוש, והפלט פלוס אחת. המודל תמיד חוזה חיובי, ומתעלם מהקלט.", hold=0.3),
    P("איזה הסבר **לא יכול** להסביר מודל כזה? אין קשר בין המאפיינים למחלקה: אז אין מה ללמוד מהם. מסביר.",
      do=hide("fig-ring-o") + show(f"q-n-{NOREL}")),
    P("מחלקה קבועה בכל הדוגמאות: אז חיזוי קבוע הוא בדיוק הנכון. מסביר. ומאפיינים עם שונות אפס לא מבדילים בין דוגמאות. מסביר.",
      do=show(f"q-n-{CONST}", f"q-n-{VAR0}")),
    P("וקבוע למידה קטן מדי? הוא משאיר את המשקולות קרוב לאתחול. אבל אתחול אקראי לא נותן אפס בדיוק בארבע קשתות.",
      do=show("e-3")),
    P("ואם האתחול היה אפס, אז ה-$w_0$ זזו והקלט לא. בשני המקרים הקלט פשוט לא עזר לחיזוי, ואימון איטי לא מסביר את זה. זו התשובה.",
      do=show(f"q-ok-{OK}"), hold=0.5,
      say="ואם האתחול היה אפס, אז הדַאבְּלְיוּ אפס זזו, והקלט לא. בשני המקרים הקלט פשוט לא עזר לחיזוי, ואימון איטי לא מסביר את זה. "
          "זו התשובה."),
], stage=w.qpanel(Q["id"], "רשת עם <span class=\"rg-en\">sign</span> בכל הנוירונים: שני קלטים, שני נוירונים חבויים ונוירון "
                  "פלט. איזו תשובה <b>לא יכולה</b> להסביר את המודל שנוצר (באיור)?", OPTS,
                  notes={NOREL: "מסביר", CONST: "מסביר", VAR0: "מסביר"}, size=30),
   panel=IMG
         + w.card("e-1", "ex", "חבויים: $\\text{sign}(0.5) = +1$ לכל קלט")
         + w.card("e-2", "ex", "פלט: $\\text{sign}(1 + 1 + 1) = +1$ תמיד")
         + w.card("e-3", "trap", "⚠️ אפס <b>בדיוק</b> בכל קשתות הקלט: הקלט לא תרם"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: לכל נוירון חבוי ולכל נוירון פלט, קשת מכל נוירון בשכבה הקודמת, ועוד $w_0$. לקלט אין משקולות.",
      do=show("r-1"), say="לסיכום: לכל נוירון חבוי ולכל נוירון פלט, קשת מכל נוירון בשכבה הקודמת, ועוד דַאבְּלְיוּ אפס. לקלט אין משקולות."),
    P("ראינו רשת בלי אקטיבציה שהתכווצה לקו $5x + 4$, ואת אותה רשת מתעקמת עם סיגמואיד.", do=show("r-2"),
      say="ראינו רשת בלי אקטיבציה שהתכווצה לקו חמש אִקְס ועוד ארבע, ואת אותה רשת מתעקמת עם סיגמואיד."),
    P("ובאימון, כל משקולת זזה באטה כפול דלתא כפול הקלט שלה, וגם $w_0$ דרך הקלט 1. נתראה ברמה הבאה.", do=show("r-3"),
      hold=0.6, say="ובאימון, כל משקולת זזה באֵטָה כפול דלתא כפול הקלט שלה, וגם דַאבְּלְיוּ אפס, דרך הקלט אחת. נתראה ברמה הבאה."),
], stage=w.eq("r-eq", "$\\#w = \\sum_{\\ell} n_\\ell \\cdot (n_{\\ell - 1} + 1)$", size=52, hidden=False, cls="box"),
   panel=w.card("r-1", "rule", "🔑 כל שכבה (לא הקלט): נוירונים × (קודמת + 1)")
         + w.card("r-2", "ex", "$3(2x + 1) + (-x + 3) - 2 = 5x + 4$")
         + w.card("r-3", "rule", "🔑 $\\Delta w_i = \\eta \\cdot \\delta \\cdot x_i$"))

VIDEO = Video("R08", "reg", "רשתות: שכבות, משקולות ואימון", [title, count, lin_s, train, over, exam, recap],
              bank={"22B-C-Q6": OK})
w.check_say(VIDEO)
