# -*- coding: utf-8 -*-
"""R09 · מבחן מסכם — boss walkthrough video (build: tools/video/vid.py R09).

Source: courses/reg/R09.js (boss level: exam-size questions from the whole course). Three exam questions that no guided
quiz uses: 24B-A-Q15 (the gradient is 0 at an extremum), SAMP-1-Q11 (Hessian: second derivatives, too costly -> GD),
PRAC-N-Q2 (pixel counts: the white area is x₂ exactly -> w₁ = 0, w₂ = 1, for the right reason).
Avoided on purpose: the boss quiz items (batch step from w = (1, 1), SST = 80 with SSE = 20, the electricity metric,
the 25/18/16 matrix, λ = 50, heaviside with w = (−0.3, 0.2, 0.5, 0.4), 3 -> 4 -> 4 -> 2) and the boss refs 24B-C-Q18,
23B-A-Q1, SAMP-1-Q13 with their twins; 23S-A-Q20 (a heaviside twin of R09-Q6) and SAMP-2-Q11 (an SVM twin of R07-Q6)
are not used; 26S-A-Q4 is left out because its point (convex -> converges to the optimum) is the quiz ref 22B-A-Q10's.
The q1 elimination of the "convex" option says only that convexity is not given and not needed for a zero gradient at an
extremum (no "convex -> global minimum" sentence: R04-Q5's key). The exam anchors of R01..R08 are not repeated.
Twins of the anchors (SAMP-2-Q2, SAMP-3-Q2, PRAC-X-Q8) are not quiz items either.
"""
from fractions import Fraction as Fr

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
# 1. 24B-A-Q15: J(w) = (w − 3)² has J'(3) = 0 at its minimum; f(w) = −(w − 1)² has f'(1) = 0 at its maximum;
#    the Hessian (second derivative) at the minimum is 2, not 0
dJ = lambda v: 2 * (v - 3)
assert dJ(3) == 0 and (lambda v: -2 * (v - 1))(1) == 0
assert 2 != 0
Q1 = BANK["24B-A-Q15"]
A1 = w.opt_index(Q1, "בנקודת הקיצון הוא תמיד 0")
B1 = w.opt_index(Q1, "צעדים המנוגדים")
C1 = w.opt_index(Q1, "הסיאן")
D1 = w.opt_index(Q1, "convex")
assert Q1["correctIndex"] == A1 == 0 and sorted([A1, B1, C1, D1]) == [0, 1, 2, 3]

# 2. SAMP-1-Q11 (Hessian): with d weights it has d·d second derivatives
assert 1000 * 1000 == 10 ** 6
QH = BANK["SAMP-1-Q11"]
AH = w.opt_index(QH, "יקרה מדי")
BH, CH, DH = w.opt_index(QH, "מסדר ראשון"), w.opt_index(QH, "כדי לבצע gradient descent"), w.opt_index(QH, "prototypes")
assert QH["correctIndex"] == AH == 2 and sorted([AH, BH, CH, DH]) == [0, 1, 2, 3]

# 3. PRAC-N-Q2: 50×50 binary image, x1 = black count, x2 = white count, x1 + x2 = 2500; the white area is x2
TOT, WHITE, BLACK = 2500, 1000, 1500
assert WHITE + BLACK == TOT == 50 * 50
assert 0 * BLACK + 1 * WHITE == WHITE and 0.6 * BLACK + 0.4 * WHITE == 1300
# with x1 = 2500 − x2: w1·x1 + w2·x2 = 2500·w1 + (w2 − w1)·x2 equals x2 for every image only for w1 = 0, w2 = 1
for a, b in ((0, 1), (0.6, 0.4), (-1, 0), (0.5, 1.5)):
    ok = all(abs(a * (TOT - x2) + b * x2 - x2) < 1e-9 for x2 in (0, 700, 1000, 2500))
    assert ok == ((a, b) == (0, 1))
Q2 = BANK["PRAC-N-Q2"]
A2 = w.opt_index(Q2, "באופן בלעדי")
B2 = w.opt_index(Q2, "תמיד מאפס")
C2, D2 = w.opt_index(Q2, "אינווריאנטיות"), w.opt_index(Q2, "מנורמלות")
assert Q2["correctIndex"] == A2 == 3 and sorted([A2, B2, C2, D2]) == [0, 1, 2, 3]

SHORT1 = {k: o.replace(" יש לבחור תשובה אחת בלבד (הנכונה ביותר):", "") for k, o in enumerate(Q1["options"])}

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("זה הסרטון המסכם של הקורס: שלוש שאלות אמיתיות, ממבחנים ומתרגול."),
    P("גרדיאנט, מטריצת הסיאן, ומשקולות של מודל לינארי. בכל שאלה בודקים גם את המספר, וגם את הנימוק של כל אפשרות.",
      say="גרדיאנט, מטריצת הֶסְיָאן, ומשקולות של מודל לינארי. בכל שאלה בודקים גם את המספר, וגם את הנימוק של כל אפשרות.",
      hold=0.4),
], stage='<div class="kicker">R09 · רגרסיה ורשתות נוירונים</div><h1>מבחן מסכם</h1>'
         '<p>שלוש שאלות מכל הקורס, עם המלכודת של כל אחת.</p>', layout="title")

q1 = Scene("q1", "שאלה 1: גרדיאנט", [
    P("שאלה ראשונה: כדי למצוא את המינימום של פונקציית המחיר, מחפשים גרדיאנט אפס. האם הטענה נכונה, ומדוע?"),
    P("נזכר: הגרדיאנט הוא השיפוע. בתחתית הקערה, ובפסגה של גבעה, השיפוע שטוח, כלומר אפס.", do=show("a-1")),
    P("לכן בכל נקודת קיצון של פונקציה גזירה, הגרדיאנט שווה אפס. הטענה נכונה, ומהסיבה הזו.", do=show(f"q-ok-{A1}"),
      hold=0.3),
    P("״לא נכון, כי צועדים נגד הגרדיאנט״: זה לא סותר. צועדים נגד השיפוע, ונעצרים בדיוק כשהוא אפס.",
      do=show(f"q-x-{B1}", f"q-n-{B1}")),
    P("״צריך מטריצת הסיאן ששווה לאפס״: ההסיאן הוא נגזרות שניות. במינימום של $(w - 3)^2$ הוא 2, לא אפס.",
      do=show(f"q-x-{C1}", f"q-n-{C1}", "a-2"),
      say="צריך מטריצת הֶסְיָאן ששווה לאפס: ההסיאן הוא הנגזרות השניות. במינימום של דַאבְּלְיוּ פחות שלוש, בריבוע, הוא שתיים, לא אפס."),
    P("״נכון, כי הפונקציה קמורה״: הטענה עוסקת בכל פונקציה גזירה. קמירות לא נתונה, ולא צריך אותה לגרדיאנט אפס בקיצון.",
      do=show(f"q-x-{D1}", f"q-n-{D1}", "a-3")),
    P("כשבודקים אפשרות, בודקים גם את המילה ״כי״.", hold=0.4),
], stage=w.qpanel(Q1["id"], "טענה: כדי למצוא את פונקציית המחיר המזערית, מחפשים גרדיאנט אפסי. האם הטענה נכונה? ומדוע?",
                  w.opts(Q1, short=SHORT1), notes={B1: "לא סותר", C1: "שגוי", D1: "נימוק שגוי"}, size=28),
   panel=w.card("a-1", "rule", "🔑 בקיצון: גרדיאנט $= 0$")
         + w.card("a-2", "ex", "$J = (w - 3)^2$: $J'(3) = 0$, $J''(3) = 2$")
         + w.card("a-3", "trap", "⚠️ נימוק שלא נתון בשאלה ← האפשרות נופלת"))

SHORT_H = {k: o.replace("מטריצת Hessian היא ", "") for k, o in enumerate(QH["options"])}
qh = Scene("qh", "שאלה 2: Hessian", [
    P("שאלה שנייה, באותו נושא: מהי מטריצת הסיאן, וכיצד משתמשים בה ברגרסיה לינארית?",
      say="שאלה שנייה, באותו נושא: מהי מטריצת הֶסְיָאן, וכיצד משתמשים בה ברגרסיה לינארית?"),
    P("הגרדיאנט הוא הנגזרות הראשונות: שיפוע לכל משקולת. ההסיאן הוא הנגזרות **השניות**: לכל זוג משקולות, איך השיפוע משתנה.",
      do=show("h-1"),
      say="הגרדיאנט הוא הנגזרות הראשונות: שיפוע לכל משקולת. ההֶסְיָאן הוא הנגזרות השניות: לכל זוג משקולות, איך השיפוע משתנה."),
    P("עם אלף משקולות, אלה מיליון נגזרות שניות, ושיטה שמשתמשת בהסיאן צריכה גם להפוך את המטריצה הזו. זה יקר מאוד.",
      do=show("h-2"),
      say="עם אלף משקולות, אלה מיליון נגזרות שניות, ושיטה שמשתמשת בהֶסְיָאן צריכה גם להפוך את המטריצה הזו. זה יקר מאוד."),
    P("לכן משתמשים בירידה במורד הגרדיאנט, שצריכה רק את הנגזרות הראשונות. זו התשובה.", do=show(f"qh-ok-{AH}"), hold=0.3),
    P("״נגזרות מסדר ראשון״ נופלת: זו ההגדרה של הגרדיאנט. ו״משתמשים בה כדי לבצע את הירידה״ נופלת: הצעד משתמש רק בגרדיאנט.",
      do=show(f"qh-x-{BH}", f"qh-n-{BH}", f"qh-x-{CH}", f"qh-n-{CH}")),
    P("ומטריצת מרחקים בין דוגמאות למרכזים שייכת לאשכול, לא לנגזרות.", do=show(f"qh-x-{DH}", f"qh-n-{DH}"), hold=0.4),
], stage=w.qpanel(QH["id"], "מהי מטריצת <span class=\"rg-en\">Hessian</span>, וכיצד משתמשים בה ברגרסיה לינארית?",
                  w.opts(QH, short=SHORT_H), notes={BH: "זה הגרדיאנט", CH: "רק גרדיאנט", DH: "אשכול"}, size=26, pfx="qh"),
   panel=w.card("h-1", "ex", "גרדיאנט: $d$ נגזרות ראשונות · הסיאן: $d \\times d$ נגזרות שניות")
         + w.card("h-2", "rule", "🔑 $d = 1000$ ← $10^6$ נגזרות שניות, ועוד היפוך מטריצה"))

SHORT2 = {k: o.replace("משום ש", "כי ").replace("מכיוון ש", "כי ") for k, o in enumerate(Q2["options"])}
q2 = Scene("q2", "שאלה 3: משקולות", [
    P("שאלה שלישית: תמונות שחור-לבן, 50 על 50. $x_1$ סופר פיקסלים שחורים, $x_2$ לבנים. המודל $y = w_1 x_1 + w_2 x_2$ חוזה את שטח הלבן.",
      say="שאלה שלישית: תמונות שחור-לבן, חמישים על חמישים. אִקְס אחת סופר פיקסלים שחורים, אִקְס שתיים לבנים. "
          "המודל: דַאבְּלְיוּ אחת אִקְס אחת, ועוד דַאבְּלְיוּ שתיים אִקְס שתיים, חוזה את שטח הלבן."),
    P("שטח הלבן הוא **בדיוק** מספר הפיקסלים הלבנים. אז $w_1 = 0$ ו-$w_2 = 1$ נותנים חיזוי מושלם, לכל תמונה.",
      do=show("c-1"),
      say="שטח הלבן הוא בדיוק מספר הפיקסלים הלבנים. אז דַאבְּלְיוּ אחת אפס, ודַאבְּלְיוּ שתיים אחת, נותנים חיזוי מושלם, לכל תמונה."),
    P("נבדוק את 0.6 ו-0.4 על התמונה מהשאלה: 0.6 כפול 1500, ועוד 0.4 כפול 1000, זה 1300. אבל שטח הלבן הוא 1000.",
      do=show("c-2", f"q2-x-{C2}", f"q2-n-{C2}", f"q2-x-{D2}", f"q2-n-{D2}"),
      say="נבדוק את אפס נקודה שש ואפס נקודה ארבע על התמונה מהשאלה: אפס נקודה שש כפול אלף וחמש מאות, ועוד אפס נקודה ארבע כפול אלף, "
          "זה אלף שלוש מאות. אבל שטח הלבן הוא אלף.", hold=0.3),
    P("נשארו שתי אפשרויות עם 0 ו-1. באחת הנימוק: רגרסיה לינארית **תמיד** מאפסת את משקולת הרקע. אין כלל כזה.",
      do=show(f"q2-x-{B2}", f"q2-n-{B2}"),
      say="נשארו שתי אפשרויות עם אפס ואחת. באחת הנימוק: רגרסיה לינארית תמיד מאפסת את משקולת הרקע. אין כלל כזה."),
    P("הנימוק הנכון: שטח הלבן נקבע רק על ידי ספירת הלבנים, ולכן לשחורים אין מה לתרום. זו התשובה.",
      do=show(f"q2-ok-{A2}", "c-k"), hold=0.5),
], stage=w.qpanel(Q2["id"], "תמונה בינארית $50 \\times 50$: $x_1$ = מספר הפיקסלים השחורים, $x_2$ = מספר הלבנים. "
                  "המודל $y = w_1 x_1 + w_2 x_2$ חוזה את שטח הלבן. מה המשקולות במודל מושלם?",
                  w.opts(Q2, short=SHORT2), notes={B2: "נימוק שגוי", C2: "1300 ≠ 1000", D2: "1300 ≠ 1000"}, size=26, pfx="q2"),
   panel=w.eq("c-1", "$y = 0 \\cdot x_1 + 1 \\cdot x_2$", size=44, cls="box")
         + w.eq("c-2", "$0.6 \\cdot 1500 + 0.4 \\cdot 1000 = 1300$", size=36)
         + w.card("c-k", "rule", "🔑 מספרים נכונים + נימוק נכון"))

recap = Scene("recap", "סיכום", [
    P("לסיכום הקורס: בכל שאלה מחשבים פעם אחת, ובודקים גם את המספר וגם את המסקנה של כל אפשרות."),
    P("רוב המסיחים הם שלב ביניים, סימן הפוך, או $w_0$ שנשכח.", do=show("r-1"),
      say="רוב המסיחים הם שלב ביניים, סימן הפוך, או דַאבְּלְיוּ אפס שנשכח."),
    P("ואפשרות עם מסקנה נכונה ונימוק שגוי היא עדיין שגויה. בהצלחה במבחן!", do=show("r-2"), hold=0.6),
], stage=w.eq("r-eq", "$\\text{net} = w_0 + \\sum w_i x_i$", size=60, hidden=False, cls="box"),
   panel=w.card("r-1", "trap", "⚠️ שלב ביניים · סימן הפוך · $w_0$ שנשכח")
         + w.card("r-2", "rule", "🔑 בודקים מספר, מסקנה, ונימוק"))

VIDEO = Video("R09", "reg", "מבחן מסכם", [title, q1, qh, q2, recap],
              bank={"24B-A-Q15": A1, "SAMP-1-Q11": AH, "PRAC-N-Q2": A2})
w.check_say(VIDEO)
