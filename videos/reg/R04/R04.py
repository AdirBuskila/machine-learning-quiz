# -*- coding: utf-8 -*-
"""R04 · התכנסות: α, קמירות ו-SGD — video script (build: tools/video/vid.py R04).

Source: courses/reg/R04.js (goal: what a small or large learning rate does, why linear regression always reaches the
global minimum, batch vs SGD). Exam anchor: 26S-A-Q7 (which claim about the learning rate is right), used by no guided
quiz. Avoided on purpose: the R04 quiz items (J = w² from w = 5 with α = 0.25; w = 2 with α = 1.5 jumping to −4; 500
examples per epoch; J₃ = J/2; zero gradient; two initialisations) and the quiz refs 25B-A-Q6, 26B-A-Q13, 22B-A-Q10
with their twins (24B-A-Q12, 23S-A-Q12, SAMP-1-Q18, 24B-C-Q18, CLAUDE-26-Q11, PRAC-X-Q12); the lesson's J = w² table
is not reused. Own runs: J(w) = (w − 3)² from w = 8 with α = 0.15 / 0.02 / 1.1 (distance to the minimum is multiplied
by 1 − 2α each step: 0.7, 0.96, −1.2); one run down the bowl, whose slope 2(w − 3) is zero only at w = 3; one run on a
two-valley curve f(w) = w⁴/4 − 2w² + w/2 that stops in the higher valley; 6 examples for batch vs SGD. No "two starts
reach the same weights" (R04-Q6), no "small α -> guaranteed optimum" (22B-A-Q10) and no "zero gradient -> global
minimum because convex" (R04-Q5) sentence: the bowl and the two-valley curve are shown, not summarised as a rule.
The J₂ = n·J family is left out (quiz/boss territory).
"""
from fractions import Fraction as Fr

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
J = lambda v: (v - 3) ** 2
dJ = lambda v: 2 * (v - 3)


def run(w0, a, k):
    out = [Fr(w0)]
    for _ in range(k):
        out.append(out[-1] - a * dJ(out[-1]))
    return out


RA = run(8, Fr(15, 100), 3)
RB = run(8, Fr(2, 100), 3)
RC = run(8, Fr(11, 10), 3)
assert RA == [8, Fr(13, 2), Fr(109, 20), Fr(943, 200)]                 # 8, 6.5, 5.45, 4.715
assert [round(float(v), 2) for v in RB] == [8, 7.8, 7.61, 7.42]
assert RC == [8, -3, Fr(51, 5), Fr(-141, 25)]                           # 8, −3, 10.2, −5.64
for r, a in ((RA, Fr(15, 100)), (RB, Fr(2, 100)), (RC, Fr(11, 10))):     # distance × (1 − 2α) every step
    assert all(r[i + 1] - 3 == (1 - 2 * a) * (r[i] - 3) for i in range(3))
assert all(J(RA[i + 1]) < J(RA[i]) for i in range(3)) and all(J(RC[i + 1]) > J(RC[i]) for i in range(3))
assert 1 - 2 * Fr(15, 100) == Fr(7, 10) and 1 - 2 * Fr(2, 100) == Fr(96, 100) and 1 - 2 * Fr(11, 10) == Fr(-6, 5)
# α = 0.02 needs many more steps than α = 0.15 to get within 0.01 of the minimum
steps = lambda a: next(k for k in range(1, 10000) if abs(run(8, a, k)[-1] - 3) < Fr(1, 100))
assert steps(Fr(15, 100)) == 18 and steps(Fr(2, 100)) == 153

# the bowl from two starts (α = 0.15): both end at the bottom w = 3
BR = run(8, Fr(15, 100), 60)
assert abs(BR[-1] - 3) < Fr(1, 10 ** 6) and dJ(3) == 0
assert all(dJ(Fr(k, 10)) != 0 for k in range(-100, 200) if k != 30)        # the slope is 0 only at w = 3

# two valleys: f(w) = w⁴/4 − 2w² + w/2, f'(w) = w³ − 4w + 1/2; α = 0.05
f = lambda v: v ** 4 / 4 - 2 * v ** 2 + v / 2
df = lambda v: v ** 3 - 4 * v + 0.5


def run2(v, k=400, a=0.05):
    out = [v]
    for _ in range(k):
        out.append(out[-1] - a * df(out[-1]))
    return out


NL, NR = run2(-0.4), run2(0.6)
assert abs(df(NL[-1])) < 1e-9 and abs(df(NR[-1])) < 1e-9
assert round(NL[-1], 2) == -2.06 and round(NR[-1], 2) == 1.93
assert f(NL[-1]) < f(NR[-1]) - 1                    # the right valley is only a local minimum

# batch vs SGD on 6 examples
NEX = 6
assert (1, NEX) == (1, 6)

# exam 26S-A-Q7
Q = BANK["26S-A-Q7"]
OK = w.opt_index(Q, "התכנסות איטית")
FIT = w.opt_index(Q, "overfitting")
ACC = w.opt_index(Q, "מדויקת יותר")
INIT = w.opt_index(Q, "ערך האתחול")
assert Q["correctIndex"] == OK == 1 and sorted([OK, FIT, ACC, INIT]) == [0, 1, 2, 3]
OPTS = w.opts(Q)

fm = lambda v: w.fmt(float(v), 3)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("ברמה הקודמת עשינו צעד עדכון אחד. בסרטון הזה נחזור עליו שוב ושוב, ונבדוק אם מגיעים ליעד."),
    P("נראה מה עושה קבוע למידה קטן או גדול, איך נראית פונקציה קמורה, ומה ההבדל בין שתי שיטות עדכון."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">R04 · רגרסיה ורשתות נוירונים</div><h1>התכנסות: <span class="rg-en">α</span>, קמירות ו-<span class="rg-en">SGD</span></h1>'
         '<p>קבוע למידה קטן או גדול, קערה מול עקומה עם כמה עמקים, וכמה דוגמאות נכנסות לכל צעד.</p>', layout="title")

PA = w.Plot("a", (-7, 13), (0, 110), w=820, h=560, xticks=[-5, 0, 3, 5, 10], yticks=[0, 25, 50, 75, 100],
            xlabel="w", ylabel="J")
PA.fn(J, -7, 13, kind="k-mut", width=4)
for nm, R, kind in (("ra", RA, "k-ok"), ("rb", RB, "k-a"), ("rc", RC, "k-bad")):
    PA.poly([(float(v), float(J(v))) for v in R], kind=kind, id_=f"{nm}-l", hidden=True, width=3, dash=True)
    for i, v in enumerate(R):
        PA.dot(float(v), float(J(v)), kind=kind, id_=f"{nm}-{i}", hidden=True, r=11 if i else 13)
RUNS = w.table("rt", ["$\\alpha$", "$w$", "", "", ""],
               [["0.15", *[fm(v) for v in RA]], ["0.02", *[w.fmt(float(v), 2) for v in RB]], ["1.1", *[fm(v) for v in RC]]],
               hidden={(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}, size=30)
alpha = Scene("alpha", "קבוע הלמידה", [
    P("פונקציה לניסוי: $J(w) = (w - 3)^2$. המינימום ב-$w = 3$, והשיפוע הוא $2(w - 3)$.",
      say="פונקציה לניסוי: ג'יי של דַאבְּלְיוּ שווה דַאבְּלְיוּ פחות שלוש, בריבוע. המינימום בדַאבְּלְיוּ שווה שלוש, והשיפוע הוא פעמיים, "
          "בסוגריים דַאבְּלְיוּ פחות שלוש."),
    P("מתחילים ב-$w = 8$. עם אלפא 0.15 הצעדים: 6.5, 5.45, 4.715. מתקרבים יפה.",
      do=show("a-ra-0", "a-ra-1", "a-ra-2", "a-ra-3", "a-ra-l", "rt-0-2", "rt-0-3", "rt-0-4"),
      say="מתחילים בדַאבְּלְיוּ שווה שמונה. עם אלפא אפס נקודה חמש עשרה, הצעדים: שש נקודה חמש, חמש נקודה ארבעים וחמש, "
          "ארבע נקודה שבע אחת חמש. מתקרבים יפה."),
    P("עם אלפא 0.02: 7.8, 7.61, 7.42. אותו כיוון, אבל לאט מאוד.",
      do=show("a-rb-0", "a-rb-1", "a-rb-2", "a-rb-3", "a-rb-l", "rt-1-2", "rt-1-3", "rt-1-4"),
      say="עם אלפא אפס נקודה אפס שתיים: שבע נקודה שמונה, שבע נקודה שישים ואחת, שבע נקודה ארבעים ושתיים. אותו כיוון, אבל לאט מאוד."),
    P("עם אלפא 1.1: מינוס 3, ואז 10.2, ואז מינוס 5.64. כל צעד עובר את המינימום ונוחת רחוק יותר.",
      do=show("a-rc-0", "a-rc-1", "a-rc-2", "a-rc-3", "a-rc-l", "rt-2-2", "rt-2-3", "rt-2-4"),
      say="עם אלפא אחת נקודה אחת: מינוס שלוש, ואז עשר נקודה שתיים, ואז מינוס חמש נקודה שישים וארבע. "
          "כל צעד עובר את המינימום, ונוחת רחוק יותר.", hold=0.3),
    P("למה? בכל צעד המרחק מהמינימום מוכפל ב-1 פחות פעמיים אלפא.", do=show("a-k"),
      say="למה? בכל צעד, המרחק מהמינימום מוכפל באחת פחות פעמיים אלפא."),
    P("0.7 מקטין אותו מהר. 0.96 מקטין בקושי. ומינוס 1.2 מגדיל אותו, ומחליף צד בכל צעד.", do=show("a-k2"),
      say="אפס נקודה שבע מקטין אותו מהר. אפס נקודה תשעים ושש מקטין בקושי. ומינוס אחת נקודה שתיים מגדיל אותו, ומחליף צד בכל צעד."),
    P("כדי להגיע למרחק של מאית מהמינימום: 18 צעדים עם 0.15, ו-153 צעדים עם 0.02.", do=show("a-n"),
      say="כדי להגיע למרחק של אחד חלקי מאה מהמינימום: שמונה עשר צעדים עם אפס נקודה חמש עשרה, ומאה חמישים ושלושה צעדים עם אפס נקודה אפס שתיים."),
], stage=PA.svg(),
   panel=RUNS + w.card("a-k", "rule", "🔑 $w - 3 \\ \\leftarrow\\ (1 - 2\\alpha)(w - 3)$")
         + w.card("a-k2", "ex", "$0.7$ מתכנס · $0.96$ איטי · ${-1.2}$ מתבדר")
         + w.card("a-n", "plain", "עד מרחק $0.01$: 18 צעדים מול 153"))

PB = w.Plot("b", (-3, 9), (0, 30), w=480, h=430, xticks=[-2, 0, 3, 6, 8], yticks=[0, 10, 20, 30], xlabel="w", ylabel="J",
            pad=(60, 20, 26, 56))
PB.fn(J, -3, 9, kind="k-a", width=4)
pts = [(float(v), float(J(v))) for v in BR[:8]]
PB.poly(pts, kind="k-b", id_="r-p", hidden=True, width=3, dash=True)
for i, (x, y) in enumerate(pts):
    PB.dot(x, y, kind="k-b", id_=f"r-{i}", hidden=True, r=9)
PB.dot(3, 0, kind="k-ink", id_="bot", hidden=True, r=12)
PN = w.Plot("n", (-3, 3), (-6, 4), w=480, h=430, xticks=[-2, 0, 2], yticks=[-6, -3, 0, 3], xlabel="w", ylabel="J",
            pad=(60, 20, 26, 56))
PN.fn(f, -3, 3, kind="k-c", width=4)
pts = [(v, f(v)) for v in NR[:1] + NR[1:40:6] + NR[-1:]]
PN.poly(pts, kind="k-b", id_="r-p", hidden=True, width=3, dash=True)
PN.dot(pts[0][0], pts[0][1], kind="k-b", id_="r-s", hidden=True, r=10)
PN.dot(pts[-1][0], pts[-1][1], kind="k-b", id_="r-e", hidden=True, r=13)
PN.dot(NL[-1], f(NL[-1]), kind="k-ok", id_="l-e", hidden=True, r=13)
convex = Scene("convex", "קמירות", [
    P("איפה הצעדים נעצרים? כשהשיפוע אפס. אז אלפא כפול השיפוע הוא אפס, ו-$w$ כבר לא זז.", do=show("c-1"),
      say="איפה הצעדים נעצרים? כשהשיפוע אפס. אז אלפא כפול השיפוע הוא אפס, ודַאבְּלְיוּ כבר לא זז."),
    P("בקערה שלנו השיפוע הוא $2(w - 3)$, והוא מתאפס בנקודה אחת בלבד: $w = 3$, התחתית.",
      do=show("b-r-p", *[f"b-r-{i}" for i in range(8)], "b-bot"),
      say="בקערה שלנו השיפוע הוא פעמיים, בסוגריים דַאבְּלְיוּ פחות שלוש, והוא מתאפס בנקודה אחת בלבד: דַאבְּלְיוּ שווה שלוש, התחתית."),
    P("פונקציית המחיר של רגרסיה לינארית בנויה כמו קערה כזו: היא **קמורה**, בלי עמקים נוספים.", do=show("c-2"), hold=0.3),
    P("עקומה שאינה קמורה, כמו ברשתות נוירונים, יכולה להכיל כמה עמקים. כאן מתחילים ב-0.6, והצעדים נעצרים בעמק הימני.",
      do=show("n-r-s", "n-r-p", "n-r-e"),
      say="עקומה שאינה קמורה, כמו ברשתות נוירונים, יכולה להכיל כמה עמקים. כאן מתחילים באפס נקודה שש, והצעדים נעצרים בעמק הימני."),
    P("גם שם השיפוע אפס, אבל העמק השמאלי נמוך יותר. בעקומה כזו, נקודה שטוחה היא לא בהכרח הנמוכה ביותר.",
      do=show("n-l-e", "c-4")),
], stage=w.row(PB.svg(), PN.svg(), gap=30),
   panel=w.card("c-1", "ex", "גרדיאנט $0$ ← הצעד $0$")
         + w.card("c-2", "rule", "🔑 קמורה: נקודה אחת בלבד עם שיפוע $0$")
         + w.card("c-4", "trap", "⚠️ כמה עמקים: שיפוע $0$ גם בעמק הגבוה"))

ex6 = "".join(f'<span class="rg-ex">{i + 1}</span>' for i in range(NEX))
sgd6 = "".join(f'<span class="rg-ex">{i + 1}</span><span id="g-u{i}" class="rg-up h">עדכון</span>' for i in range(NEX))
batch = Scene("batch", "Batch מול SGD", [
    P("עוד שאלה: כמה דוגמאות משתתפות בכל צעד?"),
    P("בשיטת באץ', כל צעד מחושב מכל הדוגמאות יחד: הסכום שחישבנו ברמה הקודמת.", do=show("g-b"),
      say=f"בשיטת {SAY['batch']}, כל צעד מחושב מכל הדוגמאות יחד: הסכום שחישבנו ברמה הקודמת."),
    P("עם 6 דוגמאות: עוברים על כולן, ורק אז עדכון אחד.", do=show("g-bu"),
      say="עם שש דוגמאות: עוברים על כולן, ורק אז, עדכון אחד."),
    P("בשיטה הסטוכסטית, SGD, מעדכנים אחרי **כל דוגמה בודדת**: 6 עדכונים במעבר אחד.",
      do=show("g-s", *[f"g-u{i}" for i in range(NEX)]),
      say=f"בשיטה הסטוכסטית, {SAY['SGD']}, מעדכנים אחרי כל דוגמה בודדת: שישה עדכונים במעבר אחד."),
    P("הצעדים שלה מהירים, אבל רועשים: כל אחד נקבע מדוגמה אחת בלבד.", do=show("g-k")),
    P("מעבר אחד על כל הנתונים נקרא epoch. בבאץ' הוא גם איטרציה אחת.", do=show("g-e"),
      say=f"מעבר אחד על כל הנתונים נקרא {SAY['epoch']}. ב{SAY['batch']} הוא גם איטרציה אחת.",
      hold=0.3),
], stage=w.stack(
    f'<div id="g-b" class="h rg-lane"><span class="rg-lt">Batch</span>{ex6}<span id="g-bu" class="rg-up h">עדכון</span></div>',
    f'<div id="g-s" class="h rg-lane"><span class="rg-lt">SGD</span>{sgd6}</div>', gap=50),
   panel=w.card("g-k", "ex", "6 דוגמאות במעבר: <span class=\"rg-en\">Batch</span> ← צעד 1 · <span class=\"rg-en\">SGD</span> ← 6 צעדים")
         + w.card("g-e", "plain", "<span class=\"rg-en\">epoch</span> = מעבר אחד על כל הנתונים"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו טענה נכונה לגבי קבוע הלמידה?"),
    P("ראינו את זה: אלפא 0.02 זחל לאט, ואלפא 1.1 קפץ מצד לצד והתרחק. זו הטענה הנכונה.", do=show(f"q-ok-{OK}"),
      say="ראינו את זה: אלפא אפס נקודה אפס שתיים זחל לאט, ואלפא אחת נקודה אחת קפץ מצד לצד, והתרחק. זו הטענה הנכונה.",
      hold=0.3),
    P("הטענה על התאמת יתר וחסר נופלת: אלפא קובע איך צועדים אל המינימום, לא כמה המודל מורכב.",
      do=show(f"q-x-{FIT}", f"q-n-{FIT}")),
    P("״גדול יותר, מדויק יותר״ נופלת: ראינו אלפא גדול שלא מגיע למינימום בכלל.", do=show(f"q-x-{ACC}", f"q-n-{ACC}")),
    P("ו״משפיע רק על האתחול״ נופלת: אלפא נמצא בנוסחה של **כל** צעד, לא רק בהתחלה.",
      do=show(f"q-x-{INIT}", f"q-n-{INIT}"), hold=0.5),
], stage=w.qpanel(Q["id"], Q["question"], OPTS,
                  notes={FIT: "זה לא תפקידו", ACC: "1.1 התבדר", INIT: "בכל צעד"}),
   panel=w.eq("x-f", "$w \\leftarrow w - \\alpha \\cdot \\nabla J$", size=48, hidden=False, cls="box"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: אלפא קטן מתקדם לאט. אלפא גדול מדי עובר את המינימום, ויכול להתבדר.", do=show("r-1")),
    P("בנקודת קיצון הגרדיאנט אפס, והצעדים נעצרים.", do=show("r-2")),
    P("פונקציית מחיר קמורה היא קערה: יש בה רק נקודה אחת עם שיפוע אפס.", do=show("r-3")),
    P("ובמעבר על שש דוגמאות, באץ' עשה צעד אחד ו-SGD שישה. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6,
      say=f"ובמעבר על שש דוגמאות, {SAY['batch']} עשה צעד אחד, ו{SAY['SGD']} שישה. נתראה ברמה הבאה."),
], stage=w.eq("r-eq", "$w \\leftarrow w - \\alpha \\cdot \\nabla J$", size=56, hidden=False, cls="box"),
   panel=w.card("r-1", "rule", "🔑 $\\alpha$ קטן ← איטי · $\\alpha$ גדול ← קפיצות והתבדרות")
         + w.card("r-2", "rule", "🔑 בקיצון: גרדיאנט $= 0$")
         + w.card("r-3", "rule", "🔑 קמורה: נקודת שיפוע אפס אחת")
         + w.card("r-4", "ex", "6 דוגמאות: <span class=\"rg-en\">Batch</span> 1 צעד · <span class=\"rg-en\">SGD</span> 6 צעדים"))

VIDEO = Video("R04", "reg", "התכנסות: α, קמירות ו-SGD", [title, alpha, convex, batch, exam, recap],
              bank={"26S-A-Q7": OK})
w.check_say(VIDEO)
