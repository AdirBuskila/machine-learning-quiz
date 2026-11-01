# -*- coding: utf-8 -*-
"""R07 · הנוירון הבודד ו-SVM — video script (build: tools/video/vid.py R07).

Source: courses/reg/R07.js (goal: compute a perceptron's net including the bias, turn it into the output of sign,
heaviside or sigmoid, and solve the bias and SVM questions with the same computation). Exam part: an original
"בסגנון מבחן" question (bank={}) in the format of 22B-C-Q2 with new numbers: sigmoid neuron, w = (−0.5, 0.3, 0.2, 0.4),
x = (1, 3, −2) -> net = −0.4, σ ≈ 0.40 -> not interesting. 22B-C-Q2 itself is not used (net = −0.2, the boss R09-Q6's
value), nor 22B-A-Q3 (net 0.3 = R07-Q3's), 23S-A-Q20 (a heaviside twin of R09-Q6), 24B-A-Q3 (its key 0.9 does not match
the printed w and x, which give 1.1).
Avoided on purpose: every R07 quiz item (w = (0.2, 0.5, −0.4, 0.1) with x = (2, 3, −1); net = 0.3 for sigmoid; the
train set with w = (−2.5, 1, −1, 1); the plane 2x₁ − 4x₂ + 1 = 0 with w₀ -> ±10; w = (1, −2), b = 3; the λ table
0/1.2/0/0/0.7/0), the quiz refs 24S-A-Q12, 23B-A-Q6, 23S-A-Q15 and their twins (24S-A-Q11, 23B-B-Q6, 24B-C-Q8,
SAMP-1-Q13), and the lesson's example. The bias scene only shows every net moving by the same amount (no FN / FP
conclusion: that is 23S-A-Q15 / R07-Q5 / SAMP-1-Q13's key). Own data: w = (−1, 0.5, 2, −0.25) with x = (2, 1, 4) (net 1)
and x = (0, 0.5, 2) (net −0.5); seven net values shifted by +1.5; SVM w = (3, −1), b = −2; eight λᵢ with three positive.
"""
from fractions import Fraction as Fr
import math

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
W = [Fr(-1), Fr(1, 2), Fr(2), Fr(-1, 4)]
XA, XB = [2, 1, 4], [0, Fr(1, 2), 2]
net = lambda wv, x: wv[0] * 1 + sum(a * b for a, b in zip(wv[1:], x))
sign = lambda t: 1 if t > 0 else -1
heav = lambda t: 1 if t >= 0 else 0
sig = lambda t: 1 / (1 + math.exp(-float(t)))
NA, NB = net(W, XA), net(W, XB)
assert NA == 1 and NB == Fr(-1, 2)
assert (sign(NA), heav(NA), round(sig(NA), 2)) == (1, 1, 0.73)
assert (sign(NB), heav(NB), round(sig(NB), 2)) == (-1, 0, 0.38)
assert len(W) == len(XA) + 1                                      # w is one longer than x: the bias, x₀ = 1
assert net(W, XA) - W[0] == 2                                     # forgetting w₀ would give 2
# the three activations agree on the class: positive side <=> net > 0 <=> σ > 0.5
for t in (NA, NB, Fr(3, 10), Fr(-2, 10)):
    assert (sign(t) == 1) == (heav(t) == 1) == (sig(t) > 0.5) or t == 0

# bias shift: seven net values with their true class; raising w₀ by 1.5 adds 1.5 to every net
NETS = [(Fr(-22, 10), False), (Fr(-14, 10), True), (Fr(-8, 10), False), (Fr(-6, 10), True), (Fr(3, 10), False),
        (Fr(9, 10), True), (Fr(18, 10), True)]
SHIFT = Fr(3, 2)
fn = lambda L: sum(1 for v, pos in L if pos and v < 0)
fp = lambda L: sum(1 for v, pos in L if not pos and v >= 0)
NETS2 = [(v + SHIFT, pos) for v, pos in NETS]
assert (fn(NETS), fp(NETS)) == (2, 1) and (fn(NETS2), fp(NETS2)) == (0, 2)
assert sum(v >= 0 for v, _ in NETS2) > sum(v >= 0 for v, _ in NETS)   # more predicted positive

# SVM
SW, SB = (3, -1), -2
svm = lambda p: SW[0] * p[0] + SW[1] * p[1] + SB
assert svm((1, 2)) == -1 and svm((2, 1)) == 3
LAMS = [0, Fr(4, 10), 0, 0, Fr(9, 10), 0, Fr(5, 10), 0]
assert sum(1 for l in LAMS if l > 0) == 3 and len(LAMS) - 3 == 5

# exam part (original, the format of 22B-C-Q2): sigmoid, w = (−0.5, 0.3, 0.2, 0.4), x = (1, 3, −2)
QW, QX = [Fr(-5, 10), Fr(3, 10), Fr(2, 10), Fr(4, 10)], [1, 3, -2]
QN = net(QW, QX)
assert QN == Fr(-4, 10) and round(sig(QN), 2) == 0.40 and round(sig(-QN), 2) == 0.60
assert net(QW, QX) - QW[0] == Fr(1, 10)                                   # without w₀
assert QW[0] + QW[1] * 1 + QW[2] * 3 + QW[3] * 2 == Fr(12, 10)            # the minus of x₃ lost
QOPTS = [(0, "<span class=\"rg-en\">net = −0.4</span>, ולכן הסרטון לא מעניין"),
         (1, "<span class=\"rg-en\">net = 0.1</span>, ולכן הסרטון מעניין"),
         (2, "<span class=\"rg-en\">net = 1.2</span>, ולכן הסרטון מעניין"),
         (3, "<span class=\"rg-en\">net = −0.4</span>, הפלט בערך 0.6, ולכן הסרטון מעניין")]
OK, NOW0, LOSTM, SIGW = 0, 1, 2, 3

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נכיר את אבן הבניין של רשתות נוירונים: נוירון בודד."),
    P("נחשב את הקלט שלו, נתרגם אותו לפלט של שלוש פונקציות אקטיבציה, ונראה שגם מודל אֶס וִי אֶם עובד באותו חישוב.",
      say=f"נחשב את הקלט שלו, נתרגם אותו לפלט של שלוש פונקציות אקטיבציה, ונראה שגם מודל {SAY['SVM']} עובד באותו חישוב."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">R07 · רגרסיה ורשתות נוירונים</div><h1>הנוירון הבודד ו-<span class="rg-en">SVM</span></h1>'
         '<p>לחשב <span class="rg-en">net</span> כולל ה-<span class="rg-en">bias</span>, ולתרגם אותו לפלט של '
         '<span class="rg-en">sign</span>, <span class="rg-en">heaviside</span> או <span class="rg-en">sigmoid</span>.</p>',
   layout="title")

NET = w.net("nn", [[("b", "1"), ("x1", "x₁"), ("x2", "x₂"), ("x3", "x₃")], [("s", "Σ")], [("o", "f")]],
            [("b", "s", "−1"), ("x1", "s", "0.5"), ("x2", "s", "2"), ("x3", "s", "−0.25"), ("s", "o", "")],
            w=820, h=560, node_cls={"b": "bias", "s": "sum", "o": "out"}, edge_label_t=0.42,
            values={"x1": ["2", "0"], "x2": ["1", "0.5"], "x3": ["4", "2"], "s": ["net = 1", "net = −0.5"]})
neuron = Scene("neuron", "הנוירון", [
    P("נוירון מלאכותי, או פרספטרון, מקבל וקטור מאפיינים.", say="נוירון מלאכותי, או פֶּרְסֶפְּטְרוֹן, מקבל וקטור מאפיינים."),
    P("כל מאפיין מוכפל במשקולת שלו, והכול מתחבר לסכום אחד, שנקרא net.", do=show("nn-hn-s"),
      say="כל מאפיין מוכפל במשקולת שלו, והכול מתחבר לסכום אחד, שנקרא נֶט."),
    P("ועוד איבר אחד: $w_0$, ה-bias. הוא מוכפל במאפיין פיקטיבי שתמיד שווה 1.", do=hide("nn-hn-s") + show("nn-hn-b", "u-b"),
      say=f"ועוד איבר אחד: דַאבְּלְיוּ אפס, ה{SAY['bias']}. הוא מוכפל במאפיין פיקטיבי שתמיד שווה אחת."),
    P("לכן וקטור המשקולות ארוך באחד מוקטור המאפיינים. זה לא באג, ואם לא נאמר אחרת, $w_0$ הוא הערך הראשון.",
      do=show("u-l"), say="לכן וקטור המשקולות ארוך באחד מוקטור המאפיינים. זה לא באג, ואם לא נאמר אחרת, דַאבְּלְיוּ אפס הוא הערך הראשון."),
    P("את net מעבירים דרך **פונקציית אקטיבציה**, $f$, וזה הפלט.", do=hide("nn-hn-b") + show("nn-hn-o", "u-f"),
      say="את הנֶט מעבירים דרך פונקציית אקטיבציה, אֶף, וזה הפלט.", hold=0.3),
    P("דוגמה: $w = (-1, 0.5, 2, -0.25)$, ולקוח עם $x = (2, 1, 4)$.", do=hide("nn-hn-o") + show("nn-v-x1-0", "nn-v-x2-0", "nn-v-x3-0"),
      say="דוגמה: דַאבְּלְיוּ שווה מינוס אחת, אפס נקודה חמש, שתיים, ומינוס אפס נקודה עשרים וחמש. והקלט: שתיים, אחת, ארבע."),
    P("מינוס 1 כפול 1, ועוד חצי כפול 2, ועוד 2 כפול 1, ועוד מינוס רבע כפול 4.", do=show("u-c"),
      say="מינוס אחת כפול אחת, ועוד חצי כפול שתיים, ועוד שתיים כפול אחת, ועוד מינוס רבע כפול ארבע."),
    P("כלומר מינוס 1 ועוד 1 ועוד 2 פחות 1: net שווה **1**.", do=show("nn-v-s-0"),
      say="כלומר מינוס אחת, ועוד אחת, ועוד שתיים, פחות אחת: הנֶט שווה אחת."),
    P("בלי ה-bias היה יוצא 2. אסור לשכוח אותו.", do=show("u-t"), say=f"בלי ה{SAY['bias']} היה יוצא שתיים. אסור לשכוח אותו."),
], stage=NET,
   panel=w.card("u-b", "ex", "$x_0 = 1$, ולכן $w_0 \\cdot 1$ תמיד בסכום")
         + w.card("u-l", "plain", "$w$ ארוך ב-1 מ-$x$: הערך הראשון הוא $w_0$")
         + w.card("u-f", "rule", "🔑 $\\text{net} = w_0 + \\sum w_i x_i, \\quad \\hat{y} = f(\\text{net})$")
         + w.card("u-c", "ex", "$\\text{net} = -1 + 1 + 2 - 1 = 1$")
         + w.card("u-t", "trap", "⚠️ בלי $w_0$: $2$"))

SG = w.Plot("sg", (-5, 5), (0, 1), w=620, h=420, xticks=[-4, -2, 0, 2, 4], yticks=[0, 0.5, 1], xlabel="net", ylabel="σ",
            pad=(70, 20, 24, 56))
SG.seg(-5, 0.5, 5, 0.5, kind="k-b", width=2, dash=True)
SG.fn(sig, -5, 5, kind="k-a", width=5)
SG.dot(1, sig(1), kind="k-ok", id_="a", hidden=True, r=12)
SG.text(1, sig(1), "0.73", kind="k-ok", id_="at", hidden=True, size=26, anchor="start", dx=16, dy=24)
SG.dot(-0.5, sig(-0.5), kind="k-bad", id_="b", hidden=True, r=12)
SG.text(-0.5, sig(-0.5), "0.38", kind="k-bad", id_="bt", hidden=True, size=26, anchor="end", dx=-16, dy=-8)
ACT = w.table("ac", ["", "$\\text{net} = 1$", "$\\text{net} = -0.5$"],
              [["<span class=\"rg-en\">sign</span>", "+1", "−1"], ["<span class=\"rg-en\">heaviside</span>", "1", "0"],
               ["<span class=\"rg-en\">sigmoid</span>", "0.73", "0.38"]],
              hidden={(None, 1), (None, 2)}, size=30)
act = Scene("act", "שלוש אקטיבציות", [
    P("שלוש פונקציות אקטיבציה במבחן. sign מחזירה פלוס 1 ל-net חיובי, ומינוס 1 לשלילי.", do=show("ac-0-1"),
      say=f"שלוש פונקציות אקטיבציה במבחן. {SAY['sign']} מחזירה פלוס אחת לנֶט חיובי, ומינוס אחת לשלילי."),
    P("heaviside מחזירה 1 ל-net אפס ומעלה, ו**אפס** לשלילי.", do=show("ac-1-1"),
      say=f"{SAY['heaviside']} מחזירה אחת לנֶט אפס ומעלה, ואפס לשלילי."),
    P("ו-sigmoid מחזירה מספר בין 0 ל-1. עבור net שווה 1, יוצא בערך 0.73.", do=show("ac-2-1", "sg-a", "sg-at"),
      say=f"ו{SAY['sigmoid']} מחזירה מספר בין אפס לאחת. עבור נֶט שווה אחת, יוצא בערך אפס נקודה שבעים ושלוש.", hold=0.3),
    P("לקוח שני: $x = (0, 0.5, 2)$. הפעם net יוצא מינוס חצי.", do=show("ac-0-2", "ac-1-2", "ac-2-2", "sg-b", "sg-bt"),
      say="לקוח שני: אפס, אפס נקודה חמש, שתיים. הפעם הנֶט יוצא מינוס חצי."),
    P("sign נותנת מינוס 1. heaviside נותנת 0, ולעולם לא מינוס 1. ו-sigmoid נותנת 0.38.", do=show("v-t"),
      say=f"{SAY['sign']} נותנת מינוס אחת. {SAY['heaviside']} נותנת אפס, ולעולם לא מינוס אחת. ו{SAY['sigmoid']} נותנת אפס נקודה שלושים ושמונה."),
    P("ושימו לב: בשלושתן הגבול הוא net שווה אפס. ב-sigmoid זה בדיוק הפלט 0.5.", do=show("v-k"),
      say=f"ושימו לב: בשלושתן הגבול הוא נֶט שווה אפס. ב{SAY['sigmoid']} זה בדיוק הפלט חצי."),
    P("כלומר: הסימן של net קובע את המחלקה. את סוג הפונקציה צריך רק כששואלים מה **הפלט** המדויק.",
      say="כלומר: הסימן של הנֶט קובע את המחלקה. את סוג הפונקציה צריך רק כששואלים מה הפלט המדויק."),
], stage=w.stack(SG.svg(), ACT, gap=18),
   panel=w.card("v-t", "trap", "⚠️ <span class=\"rg-en\">heaviside</span>: רק $0$ או $1$, אף פעם לא $-1$")
         + w.card("v-k", "rule", "🔑 גבול ההחלטה: $\\text{net} = 0$ ($\\sigma = 0.5$) בשלושתן"))

NL = w.numline("nl", [[(v, False) for v, _ in NETS], [(v, False) for v, _ in NETS2]], -2.5, 3.5, [-2, -1, 0, 1, 2, 3], w=1000, h=230)
CROSS = sum(1 for (a, _), (b, _) in zip(NETS, NETS2) if a < 0 <= b)
assert CROSS == 3
bias = Scene("bias", "הזזת ה-bias", [
    P("שבע דוגמאות, כל אחת לפי ה-net שלה. מימין לאפס: חזוי חיובי. משמאל: חזוי שלילי.",
      say="שבע דוגמאות, כל אחת לפי הנֶט שלה. מימין לאפס: חזוי חיובי. משמאל: חזוי שלילי."),
    P("עכשיו נגדיל את $w_0$ ב-1.5. הוא נמצא ב-net של **כל** דוגמה, ולכן כל הנקודות זזות ימינה באותו מרחק.",
      do=hide("nl-g0") + show("nl-g1", "b-0"),
      say="עכשיו נגדיל את דַאבְּלְיוּ אפס באחת נקודה חמש. הוא נמצא בנֶט של כל דוגמה, ולכן כל הנקודות זזות ימינה, באותו מרחק."),
    P("שלוש נקודות עברו את האפס, וצבען התחלף. שאר הנקודות זזו בדיוק באותו מרחק, בלי שום שינוי ביניהן.", do=show("b-k"),
      hold=0.3),
], stage=w.stack(NL, '<div class="rg-row" style="gap:40px;font-size:28px;direction:rtl">'
                     '<span class="rg-ok">● חזוי חיובי</span><span class="rg-bad">● חזוי שלילי</span></div>',
                 gap=10),
   panel=w.card("b-0", "plain", "מגדילים את $w_0$ ב-1.5: כל <span class=\"rg-en\">net</span> עולה ב-1.5")
         + w.card("b-k", "ex", "🔑 $w_0$ מזיז את כל ה-<span class=\"rg-en\">net</span> יחד"))

LT = w.table("lm", ["דוגמה", *[str(i + 1) for i in range(8)]], [["$\\lambda_i$", *[w.fmt(float(l)) for l in LAMS]]],
             size=30, hl_rows=False)
svm_s = Scene("svm", "SVM", [
    P("גם SVM לומד מישור מפריד, ומסווג לפי הסימן של $w \\cdot x + b$. כאן $b$ הוא ה-bias.", do=show("m-f"),
      say=f"גם {SAY['SVM']} לומד מישור מפריד, ומסווג לפי הסימן של דַאבְּלְיוּ כפול אִקְס ועוד בִּי. כאן בִּי הוא ה{SAY['bias']}."),
    P("נניח $w = (3, -1)$ ו-$b = -2$. הנקודה $(1, 2)$: 3 פחות 2 פחות 2, מינוס 1. קטגוריה מינוס 1.", do=show("m-1"),
      say="נניח דַאבְּלְיוּ שווה שלוש ומינוס אחת, ובִּי מינוס שתיים. הנקודה אחת, שתיים: שלוש, פחות שתיים, פחות שתיים, מינוס אחת. "
          "קטגוריה מינוס אחת."),
    P("והנקודה $(2, 1)$: 6 פחות 1 פחות 2, פלוס 3. קטגוריה 1.", do=show("m-2"),
      say="והנקודה שתיים, אחת: שש, פחות אחת, פחות שתיים, פלוס שלוש. קטגוריה אחת.",
      hold=0.3),
    P("ועוד מושג: **וקטורי תמיכה**. אלה הדוגמאות שנוגעות בשוליים ו\"מחזיקות\" את המישור.", do=show("lm")),
    P("מזהים אותן לפי מכפיל לגראנז׳: דוגמה עם $\\lambda_i$ חיובי היא וקטור תמיכה. כאן: שלוש.", do=show("m-k"),
      say="מזהים אותן לפי מכפיל לַגְרַאנְז': דוגמה עם למדא חיובי היא וקטור תמיכה. כאן: שלוש."),
    P("לחמש האחרות למדא אפס, והן לא משפיעות על המישור בכלל."),
], stage=w.stack(w.eq("m-f", "$\\text{sign}(w \\cdot x + b)$", size=52, cls="box"),
                 w.eq("m-1", "$(1, 2): \\ 3 - 2 - 2 = -1 \\ \\Rightarrow \\ -1$", size=40),
                 w.eq("m-2", "$(2, 1): \\ 6 - 1 - 2 = 3 \\ \\Rightarrow \\ +1$", size=40),
                 f'<div id="lm" class="h">{LT}</div>', gap=26),
   panel=w.card("m-k", "rule", "🔑 וקטור תמיכה ⇔ $\\lambda_i > 0$"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: נוירון עם sigmoid, $w = (-0.5, 0.3, 0.2, 0.4)$, וסרטון חדש $x = (1, 3, -2)$. מעניין, או לא?",
      say=f"שאלה בסגנון מבחן: נוירון עם {SAY['sigmoid']}. דַאבְּלְיוּ: מינוס אפס נקודה חמש, אפס נקודה שלוש, אפס נקודה שתיים, "
          "ואפס נקודה ארבע. וסרטון חדש: אחת, שלוש, מינוס שתיים. מעניין, או לא?"),
    P("net: מינוס 0.5, ועוד 0.3 כפול 1, ועוד 0.2 כפול 3, ועוד 0.4 כפול מינוס 2. כלומר מינוס 0.5 ועוד 0.3 ועוד 0.6 פחות 0.8.",
      do=show("x-1"),
      say="נֶט: מינוס אפס נקודה חמש, ועוד אפס נקודה שלוש כפול אחת, ועוד אפס נקודה שתיים כפול שלוש, ועוד אפס נקודה ארבע כפול מינוס שתיים. "
          "כלומר מינוס אפס נקודה חמש, ועוד אפס נקודה שלוש, ועוד אפס נקודה שש, פחות אפס נקודה שמונה."),
    P("net שווה מינוס 0.4. שלילי, ולכן הסיגמואיד נותנת פחות מחצי, בערך 0.40: לא מעניין.", do=show("x-2", f"q-ok-{OK}"),
      say=f"הנֶט שווה מינוס אפס נקודה ארבע. שלילי, ולכן ה{SAY['sigmoid']} נותנת פחות מחצי, בערך אפס נקודה ארבעים: לא מעניין.",
      hold=0.3),
    P("0.1 מתקבל בלי $w_0$, ו-1.2 מתקבל אם מאבדים את המינוס של המאפיין השלישי.",
      do=show(f"q-x-{NOW0}", f"q-n-{NOW0}", f"q-x-{LOSTM}", f"q-n-{LOSTM}"),
      say="אפס נקודה אחת מתקבל בלי דַאבְּלְיוּ אפס, ואחת נקודה שתיים מתקבל אם מאבדים את המינוס של המאפיין השלישי."),
    P("והאפשרות עם 0.6 הופכת את הסימן: 0.6 הוא הפלט עבור net חיובי של 0.4, לא שלילי.",
      do=show(f"q-x-{SIGW}", f"q-n-{SIGW}"),
      say="והאפשרות עם אפס נקודה שש הופכת את הסימן: אפס נקודה שש הוא הפלט עבור נֶט חיובי של אפס נקודה ארבע, לא שלילי.", hold=0.5),
], stage=w.qpanel("בסגנון מבחן", "נוירון עם <span class=\"rg-en\">sigmoid</span>: $w = ({-0.5}, 0.3, 0.2, 0.4)$ (הערך הראשון "
                  "הוא $w_0$), סרטון חדש $x = (1, 3, {-2})$. מה ה-<span class=\"rg-en\">net</span>, ואיך נסווג?", QOPTS,
                  notes={NOW0: "בלי $w_0$", LOSTM: "מינוס אבד", SIGW: "סימן הפוך"}),
   panel=w.eq("x-1", "${-0.5} + 0.3 + 0.6 - 0.8$", size=44, cls="box")
         + w.eq("x-2", "$\\text{net} = -0.4, \\quad \\sigma \\approx 0.40 < 0.5$", size=38))

recap = Scene("recap", "סיכום", [
    P("לסיכום: net הוא $w_0$ ועוד סכום המשקולות כפול המאפיינים, ו-$w_0$ הוא הערך הראשון ב-$w$.", do=show("r-1"),
      say="לסיכום: הנֶט הוא דַאבְּלְיוּ אפס, ועוד סכום המשקולות כפול המאפיינים. ודַאבְּלְיוּ אפס הוא הערך הראשון בוקטור."),
    P("הסימן של net קובע את המחלקה, בשלוש הפונקציות. heaviside מחזירה אפס או אחת בלבד.", do=show("r-2"),
      say=f"הסימן של הנֶט קובע את המחלקה, בשלוש הפונקציות. {SAY['heaviside']} מחזירה אפס או אחת בלבד."),
    P("הגדלת $w_0$ מזיזה את כל ה-net למעלה. ו-SVM: הסימן של $w \\cdot x + b$, ווקטורי תמיכה הם למדא חיובי. נתראה ברמה הבאה.",
      do=show("r-3"), hold=0.6,
      say=f"הגדלת דַאבְּלְיוּ אפס מזיזה את כל הנֶט למעלה. ו{SAY['SVM']}: הסימן של דַאבְּלְיוּ כפול אִקְס ועוד בִּי, ווקטורי תמיכה הם למדא חיובי. "
          "נתראה ברמה הבאה."),
], stage=w.eq("r-eq", "$\\hat{y} = f(w_0 + w_1 x_1 + \\dots + w_n x_n)$", size=50, hidden=False, cls="box"),
   panel=w.card("r-1", "rule", "🔑 $x_0 = 1$, ו-$w_0$ ראשון ב-$w$")
         + w.card("r-2", "rule", "🔑 $\\text{net} > 0$ ← חיובי · <span class=\"rg-en\">heaviside</span> ∈ {0, 1}")
         + w.card("r-3", "rule", "🔑 $w_0$ ↑ ← יותר חיוביים · וקטור תמיכה: $\\lambda_i > 0$"))

VIDEO = Video("R07", "reg", "הנוירון הבודד ו-SVM", [title, neuron, act, bias, svm_s, exam, recap], bank={})
w.check_say(VIDEO)
