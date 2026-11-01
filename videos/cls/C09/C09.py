# -*- coding: utf-8 -*-
"""C09 · מבחן מסכם: KNN, עצים ובייס — boss walkthrough (build: tools/video/vid.py C09).

Source: courses/cls/C09.js (boss level: exam-sized questions from the three topics, telling apart what is true of
KNN, of trees and of Naive Bayes).
Exam anchors (none is used by any guided quiz or boss):
  24S-C-Q9   what KNN is for numeric features: classification and regression (shown on one neighbourhood)
  25B-A-Q10  decision trees learn non-linear models (and three wrong statements)
  24S-C-Q6   Naive Bayes for continuous features: a generative algorithm assuming normal distributions
Not used on purpose: the boss refs 23B-B-Q9 (and its twin 23B-A-Q9), 24B-C-Q20 (and the F/G twins), 22B-C-Q1, and the
boss's own items (weighted 0.5 / 2, the W/E cities, the churn table, the frequency/device table, the scaling and
independence statements, K large vs deep tree). The independence distractor of 25B-A-Q10 is eliminated through how
post-pruning decides (validation accuracy), without stating which algorithm assumes independence (C09-Q6 asks that).
"""
import math
from fractions import Fraction as F

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- 24S-C-Q9 (verified)
Q1 = BANK["24S-C-Q9"]
NBR = [(1.0, "A", 10), (1.5, "A", 14), (2.5, "B", 18)]     # the 3 nearest: (distance, label, price)
labels = [l for _, l, _ in NBR]
assert max(set(labels), key=labels.count) == "A" and labels.count("A") == 2
assert sum(v for _, _, v in NBR) / 3 == 14
assert Q1["correctIndex"] == 0 and "לסיווג" in Q1["options"][0] and "ולחיזוי" in Q1["options"][0]
OPTS1 = list(enumerate(Q1["options"]))

# ---------------------------------------------------------------- 25B-A-Q10
Q2 = BANK["25B-A-Q10"]
assert Q2["correctIndex"] == 1 and "הפרדה לינארית" in Q2["options"][1] and "post-pruning" in Q2["options"][0]
OPTS2 = list(enumerate(Q2["options"]))
# a depth-2 tree computes XOR, which no single line separates (checked on the four corners)
XOR = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
tree = lambda a, b: (1 if b else 0) if a == 0 else (0 if b else 1)
assert all(tree(a, b) == y for (a, b), y in XOR.items())
# no line w1*a + w2*b + c > 0 matches XOR (grid search over a fine grid of weights)
grid = [i / 4 for i in range(-12, 13)]
assert not any(all((w1 * a + w2 * b + c > 0) == bool(y) for (a, b), y in XOR.items())
               for w1 in grid for w2 in grid for c in grid)

# ---------------------------------------------------------------- 24S-C-Q6
Q3 = BANK["24S-C-Q6"]
assert Q3["correctIndex"] == 3 and "גנרטיבי" in Q3["options"][3] and "נורמליות" in Q3["options"][3]
OPTS3 = list(enumerate(Q3["options"]))

# ---------------------------------------------------------------- figures
VT = w.table("vt", ["מרחק", "תווית", "מחיר"], [["1", "A", "10"], ["1.5", "A", "14"], ["2.5", "B", "18"]],
             size=32, rtl=True)

XP = w.Plot("xp", (-0.5, 1.5), (-0.5, 1.5), w=430, h=400, xticks=[0, 1], yticks=[0, 1], xname="", yname="")
XP.pts("p1", [(0, 1), (1, 0)], "b", r=16, shapes="tt")
XP.pts("p0", [(0, 0), (1, 1)], "a", r=16)
XPLOT = XP.svg()

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("זה הסרטון האחרון בקורס: שלוש שאלות מבחן, אחת מכל נושא."),
    P("מה KNN עושה, עצי החלטה, ונאיב בייס על מאפיינים רציפים.",
      say="מה קֵיי אֶן אֶן עושה, עצי החלטה, ונָאִיב בֵּייס על מאפיינים רציפים."),
    P("בכל אחת נראה איך מסיח לוקח עובדה נכונה ממקום אחר, ומדביק אותה למקום הלא נכון.", hold=0.4),
], stage='<div class="kicker">C09 · מסווגים · מבחן מסכם</div><h1>מבחן מסכם: KNN, עצים ובייס</h1>'
         '<p>שלוש שאלות אמיתיות ממבחנים, פתורות צעד אחרי צעד, וכל מסיח נופל עם הסיבה שלו.</p>', layout="title")

q1 = Scene("q1", "שאלה 1 · KNN", [
    P("שאלה ראשונה: מהו KNN עבור מאפיינים מספריים?",
      say="שאלה ראשונה: מהו קֵיי אֶן אֶן עבור מאפיינים מספריים?"),
    P("נבדוק על שכונה אחת. שלושת השכנים הקרובים של שאילתה, ולכל אחד גם תווית וגם מחיר.", do=show("vt")),
    P("אם השאלה היא איזו תווית, השכנים מצביעים: שניים A מול אחד B. זה סיווג.", do=show("a-1"),
      say="אם השאלה היא איזו תווית, השכנים מצביעים: שתיים אֵיי מול אחת בִּי. זה סיווג."),
    P("ואם השאלה היא כמה, ממצעים את המחירים: עשר, ארבע עשרה ושמונה עשרה, ממוצע ארבע עשרה. זו רגרסיה.", do=show("a-2")),
    P("אותם שכנים, אותו מנגנון, ושני סוגי תשובה. אז KNN עושה גם סיווג וגם חיזוי של ערכים מספריים.", do=show("q1-ok-0"),
      say="אותם שכנים, אותו מנגנון, ושני סוגי תשובה. אז קֵיי אֶן אֶן עושה גם סיווג וגם חיזוי של ערכים מספריים."),
    P("לכן נופלות שתי האפשרויות של ״בלבד״: כל אחת מהן זורקת חצי ממה שראינו עכשיו.", do=show("q1-x-1", "q1-x-2", "q1-n-1", "q1-n-2")),
    P("ורשת נוירונים? ב-KNN אין שכבות ואין משקלים. הוא שומר את הטבלה ומודד מרחקים.", do=show("q1-x-3", "q1-n-3"),
      say="ורשת נוירונים? בקֵיי אֶן אֶן אין שכבות ואין משקלים. הוא שומר את הטבלה ומודד מרחקים."),
    P("המלכודת: המילה ״סיווג״ מופיעה בכל מקום ליד KNN, ולכן קל לשכוח את הרגרסיה.", do=show("a-t"),
      say="המלכודת: המילה סיווג מופיעה בכל מקום ליד קֵיי אֶן אֶן, ולכן קל לשכוח את הרגרסיה.", hold=0.5),
], stage=w.col(VT, w.chip("a-1", "תווית: $2$ מול $1$ ← A (סיווג)", "ok"),
               w.chip("a-2", "מחיר: $(10+14+18)/3 = 14$ (רגרסיה)", "info"), gap=24),
   panel=w.stem(Q1["id"], "מהו KNN עבור מאפיינים מספריים?")
         + w.opts("q1", OPTS1, notes={1: "גם סיווג", 2: "גם רגרסיה", 3: "אין שכבות"}, size=27)
         + st.card("a-t", "trap", "⚠️ לא רק סיווג"))

q2 = Scene("q2", "שאלה 2 · עצים", [
    P("שאלה שנייה: איזו טענה על בניית עץ החלטה נכונה?"),
    P("נתחיל מהנכונה: עץ יכול ללמוד מודלים מורכבים, שהמחלקות בהם לא מופרדות על ידי קו ישר.", do=show("q2-ok-1")),
    P("דוגמה: ארבע פינות של ריבוע. שתי פינות אלכסוניות כחולות, ושתי האחרות צהובות. אף קו ישר לא מפריד ביניהן.",
      do=show("b-1")),
    P("ועץ בעומק שתיים כן: קודם שואל על הציר האופקי, ואז בכל ענף על הציר האנכי. ארבעה עלים, כולם צודקים.", do=show("b-2")),
    P("הטענה על אי-תלות וגיזום: גיזום בדיעבד מחליט לפי הדיוק על סט התיקוף. שום הנחת אי-תלות לא משתתפת בהחלטה.",
      do=show("q2-x-0", "q2-n-0")),
    P("הטענה שאי אפשר למדוד איכות בגלל הנטייה ל-overfitting הפוכה: בגלל הנטייה הזו בדיוק, מודדים על נתונים שהמודל לא ראה.",
      do=show("q2-x-3", "q2-n-3"),
      say="הטענה שאי אפשר למדוד איכות בגלל הנטייה לאוֹבֶרְפִיטִינְג הפוכה: בגלל הנטייה הזו בדיוק, מודדים על נתונים שהמודל לא ראה."),
    P("ו״כל הטענות שגויות״ נופלת, כי מצאנו טענה נכונה.", do=show("q2-x-2", "q2-n-2"), hold=0.5),
], stage=w.col(w.stem(Q2["id"], "איזו טענה נכונה בקשר לבניית עץ החלטה?"),
               w.opts("q2", OPTS2, notes={0: "לפי תיקוף", 2: "יש נכונה", 3: "הפוך"}, size=23), gap=14),
   panel=f'<div id="b-1" class="h">{XPLOT}</div>'
         + st.card("b-2", "ex", "עץ בעומק $2$: שאלה על כל ציר ← $4$ עלים טהורים"))

q3 = Scene("q3", "שאלה 3 · נאיב בייס", [
    P("שאלה שלישית: מהו נאיב בייס עבור מאפיינים רציפים?"),
    P("עם מאפיין רציף אי אפשר לספור ערכים, כמו בטבלאות שלנו. במקום זה מניחים שבכל מחלקה, המאפיין מתפלג נורמלית.",
      do=show("c-1")),
    P("זו הנחה של הגרסה הגאוסיאנית בלבד, זו שמיועדת למאפיינים רציפים."),
    P("לכל מחלקה לומדים ממוצע וסטיית תקן, ומהעקומה מקבלים את ה-likelihood.", do=show("c-2"),
      say="לכל מחלקה לומדים ממוצע וסטיית תקן, ומהעקומה מקבלים את הלַייקְלִיהוּד."),
    P("וכמו תמיד בנאיב בייס, המודל מתאר איך הנתונים של כל מחלקה מתפלגים. מודל כזה הוא גנרטיבי. זו התשובה.",
      do=show("q3-ok-3")),
    P("״דיסקרימינטיבי״ נופל: מודל דיסקרימינטיבי לומד רק את הגבול בין המחלקות, לא את ההתפלגות שלהן.",
      do=show("q3-x-2", "q3-n-2")),
    P("״הקטנת ממדים״ נופלת: נאיב בייס מסווג, ולא מקטין את מספר המאפיינים.", do=show("q3-x-0", "q3-n-0")),
    P("ו״בסיסי וילדותי״ הוא משחק מילים על השם. ״נאיבי״ מתאר הנחה מתמטית שראינו, לא את רמת האלגוריתם.",
      do=show("q3-x-1", "q3-n-1"), hold=0.5),
], stage=w.col(w.stem(Q3["id"], "מהו אלגוריתם Naive Bayes עבור מאפיינים רציפים?"),
               w.opts("q3", OPTS3, notes={0: "מסווג", 1: "השם", 2: "הפוך"}, size=26), gap=16),
   panel=st.card("c-1", "ex", "רציף: בכל מחלקה, התפלגות נורמלית")
         + w.eq("c-2", r"$P(x \mid Y) = \mathcal{N}(x;\ \mu_Y, \sigma_Y)$", 36))

recap = Scene("recap", "סיכום", [
    P("לסיכום הקורס כולו: ב-KNN, מרחק במטריקה שנתנו, K שכנים, ובמשוקלל משקל אחת חלקי המרחק.", do=show("r-1"),
      say="לסיכום הקורס כולו: בקֵיי אֶן אֶן, מרחק במטריקה שנתנו, קֵיי שכנים, ובמשוקלל משקל אחת חלקי המרחק."),
    P("בעצים, אנטרופיה מותנית במשקל, הנמוכה מנצחת, ובוחרים עומק לפי התיקוף.", do=show("r-2")),
    P("ובנאיב בייס, prior כפול likelihoods, בתוך כל מחלקה.", do=show("r-3"),
      say="ובנָאִיב בֵּייס, פְּרַיְיאוֹר כפול לַייקְלִיהוּדִים, בתוך כל מחלקה."),
    P("ובכל שאלה: בדקו לאיזה אלגוריתם שייכת כל עובדה, לפני שאתם מסמנים. בהצלחה במבחן!", do=show("r-4"), hold=0.8),
], stage=w.col(w.eq("r-a", r"$\hat{y} = \dfrac{\sum w_i y_i}{\sum w_i}$", 44, hidden=False),
               w.eq("r-b", r"$H(Y \mid F) = \sum_v \tfrac{n_v}{n} H(Y \mid F{=}v)$", 40, hidden=False),
               w.eq("r-c", r"$P(Y)\prod_j P(x_j \mid Y)$", 44, hidden=False), gap=34),
   panel=st.card("r-1", "rule", "🔑 KNN: מטריקה, $K$, ממוצע או רוב")
         + st.card("r-2", "rule", "🔑 עץ: $H(Y|F)$ נמוך, עומק לפי התיקוף")
         + st.card("r-3", "rule", "🔑 בייס: prior × likelihoods")
         + st.card("r-4", "trap", "⚠️ עובדה נכונה, אלגוריתם לא נכון"))

VIDEO = Video("C09", "cls", "מבחן מסכם: KNN, עצים ובייס", [title, q1, q2, q3, recap],
              bank={"24S-C-Q9": 0, "25B-A-Q10": 1, "24S-C-Q6": 3})
