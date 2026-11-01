# -*- coding: utf-8 -*-
"""C08 · נאיב בייס: ההנחה הנאיבית — video script (build: tools/video/vid.py C08).

Source: courses/cls/C08.js (lesson + goal: what "naive" means, what happens when features depend on each other, why
smoothing is needed, why Naive Bayes is generative).
Exam anchor: 24B-C-Q17 (a disadvantage of Naive Bayes: it does not work well when features are highly dependent),
which no guided quiz uses.
Not used on purpose: the C08 quiz items (what is naive, independence formulas, NB vs ID3 under dependence, 0.8·0.7·0,
the reason for smoothing, which model can generate, features added later, 500 word counts) and its refs 23B-A-Q4,
24B-A-Q19, 26S-A-Q3; their twins 22B-A-Q14 / 23B-A-Q17 / 23B-B-Q17, 25S-A-Q9, 24S-A-Q18, 25S-A-Q4 / SAMP-1-Q24 /
CLAUDE-26-Q3, 24S-A-Q8, 25B-A-Q5, PRAC-N-Q1; 23B-B-Q18 (doubtful key, see the report). The facts are shown on
numbers (a copied feature, a zero count, a sampled example), and no card repeats a quiz option.
Worked examples (new, cross-checked with scikit-learn's CategoricalNB): x and z give A 0.09 vs B 0.075; a copy of z
turns it into A 0.027 vs B 0.0375. A spam word never seen in 10 spam mails: 0.9 · 0.6 · 0 = 0; with add-one
smoothing 10/12 · 7/12 · 1/12 ~ 0.041.
"""
import math
import random

import numpy as np
from sklearn.naive_bayes import CategoricalNB

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- a copied feature (verified)
PR = {"A": 0.5, "B": 0.5}
LX = {"A": 0.6, "B": 0.3}
LZ = {"A": 0.3, "B": 0.5}
S2 = {c: PR[c] * LX[c] * LZ[c] for c in PR}
S3 = {c: S2[c] * LZ[c] for c in PR}                       # z counted twice
assert {c: round(v, 4) for c, v in S2.items()} == {"A": 0.09, "B": 0.075} and S2["A"] > S2["B"]
assert {c: round(v, 4) for c, v in S3.items()} == {"A": 0.027, "B": 0.0375} and S3["B"] > S3["A"]
# the same with scikit-learn on a table built to have exactly these frequencies (10 rows per class)
rows = []
for c, nx, nz in (("A", 6, 3), ("B", 3, 5)):
    for i in range(10):
        rows.append((int(i < nx), int(i >= 10 - nz), c))
for c in "AB":
    sub = [r for r in rows if r[2] == c]
    assert sum(r[0] for r in sub) / 10 == LX[c] and sum(r[1] for r in sub) / 10 == LZ[c]
X2 = np.array([[r[0], r[1]] for r in rows])
X3 = np.array([[r[0], r[1], r[1]] for r in rows])
yy = np.array([r[2] for r in rows])
nb2 = CategoricalNB(alpha=1e-10, force_alpha=True).fit(X2, yy)
nb3 = CategoricalNB(alpha=1e-10, force_alpha=True).fit(X3, yy)
assert nb2.predict([[1, 1]])[0] == "A" and nb3.predict([[1, 1, 1]])[0] == "B"

# ---------------------------------------------------------------- smoothing (verified)
# the six dogs of the C07 animals table (2 small, 3 long-haired) and a colour no dog had: ginger
N_DOG = 6
CNT = {"size": (2, 2), "hair": (3, 2), "colour": (0, 3)}   # (dogs with the query's value, number of values)
raw = [CNT[k][0] / N_DOG for k in ("size", "hair", "colour")]
assert raw == [2 / 6, 3 / 6, 0.0] and math.prod(raw) == 0
lap = [(CNT[k][0] + 1) / (N_DOG + CNT[k][1]) for k in ("size", "hair", "colour")]   # add-one
assert lap == [3 / 8, 4 / 8, 1 / 9] and f"{math.prod(lap):.3f}" == "0.021"

# ---------------------------------------------------------------- generating an example (verified)
rnd = random.Random(4)
draw_class = "A" if rnd.random() < PR["A"] else "B"
draw_x = int(rnd.random() < LX[draw_class])
draw_z = int(rnd.random() < LZ[draw_class])
assert (draw_class, draw_x, draw_z) == ("A", 1, 0)

# ---------------------------------------------------------------- exam 24B-C-Q17
Q = BANK["24B-C-Q17"]
assert Q["correctIndex"] == 2 and "תלות גבוהה" in Q["options"][2]
OPTS = list(enumerate(Q["options"]))

# ---------------------------------------------------------------- figures
DUP = w.table("dp", ["", "prior", "$P(x \\mid \\cdot)$", "$P(z \\mid \\cdot)$", "ציון", "$P(z \\mid \\cdot)$", "ציון"],
              [["A", "0.5", "0.6", "0.3", "0.09", "0.3", "0.027"], ["B", "0.5", "0.3", "0.5", "0.075", "0.5", "0.0375"]],
              hide_cols=(4, 5, 6), size=30, head_ids=True)
SM = w.table("sm", ["מאפיין", "מתוך 6 כלבים", "בלי החלקה", "עם החלקה"],
             [["קטן", "2", r"$\tfrac{2}{6}$", r"$\tfrac{3}{8}$"], ["ארוך", "3", r"$\tfrac{3}{6}$", r"$\tfrac{4}{8}$"],
              ["ג׳ינג׳י", "0", "0", r"$\tfrac{1}{9}$"]],
             hide_cols=(2, 3), size=32, rtl=True, head_ids=True)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נבין מה נאיבי בנאיב בייס, ומה זה עולה לנו."),
    P("נראה מה קורה כשמאפיינים תלויים, למה צריך החלקה, ולמה קוראים לו מודל גנרטיבי."),
    P("ובסוף נפתור שאלה אמיתית ממבחן.", hold=0.4),
], stage='<div class="kicker">C08 · מסווגים</div><h1>נאיב בייס: ההנחה הנאיבית</h1>'
         '<p>מכפלת ה-likelihoods מותרת רק כשהמאפיינים בלתי תלויים בהינתן המחלקה. מה קורה כשזה לא נכון, ומה קורה כשאחד מהם אפס.</p>',
   layout="title")

naive = Scene("naive", "מה נאיבי", [
    P("ברמה הקודמת הכפלנו likelihood של כל מאפיין בנפרד. מתי מותר לעשות את זה?", do=show("n-1"),
      say="ברמה הקודמת הכפלנו לַייקְלִיהוּד של כל מאפיין בנפרד. מתי מותר לעשות את זה?"),
    P("ההסתברות של שני מאפיינים יחד שווה למכפלה רק אם הם בלתי תלויים, בתוך כל מחלקה.", do=show("n-2")),
    P("בלתי תלויים פירושו: לדעת את המאפיין הראשון לא משנה כלום לגבי השני.", do=show("n-3")),
    P("במציאות זה כמעט אף פעם לא מדויק. מי שמניח את זה בכל זאת, נאיבי. ומכאן השם.", do=show("n-r")),
    P("ומה מרוויחים? כל מאפיין נלמד לבד, מספירה פשוטה. מהיר וקל, גם עם הרבה מאפיינים.", hold=0.4),
], stage=w.col(w.eq("n-1", r"$P(x, z \mid Y) \;\overset{?}{=}\; P(x \mid Y)\cdot P(z \mid Y)$", 46),
               w.eq("n-2", r"$\text{only if } x \perp z \mid Y$", 40),
               w.eq("n-3", r"$P(z \mid x, Y) = P(z \mid Y)$", 40), gap=34),
   panel=st.card("n-r", "rule", "🔑 המכפלה מותרת רק אם בתוך המחלקה, מאפיין אחד לא מספר כלום על האחר"))

dep = Scene("dep", "מאפיין כפול", [
    P("מה קורה כשההנחה נשברת? שתי מחלקות, prior חצי וחצי, ושני מאפיינים, איקס וזד.", do=show("dp"),
      say="מה קורה כשההנחה נשברת? שתי מחלקות, פְּרַיְיאוֹר חצי וחצי, ושני מאפיינים, אִקְס וזֶד."),
    P("A: חצי, כפול 0.6, כפול 0.3. יוצא 0.09. B: חצי, כפול 0.3, כפול 0.5. יוצא 0.075. מנצחת A.",
      do=show("dp-th4", *w.cells("dp", "v", (0, 1), (4,))) + show("dp-h0-4"),
      say="אֵיי: חצי, כפול 0.6, כפול 0.3. יוצא 0.09. בִּי: חצי, כפול 0.3, כפול 0.5. יוצא 0.075. מנצחת אֵיי."),
    P("עכשיו נוסיף עמודה שהיא עותק מדויק של זד. אין בה שום מידע חדש. זה מקרה קיצוני של תלות.",
      do=show("dp-th5", *w.cells("dp", "v", (0, 1), (5,))),
      say="עכשיו נוסיף עמודה שהיא עותק מדויק של זֶד. אין בה שום מידע חדש. זה מקרה קיצוני של תלות."),
    P("אבל נאיב בייס מכפיל בה שוב. A יורדת ל-0.027, B ל-0.0375. ועכשיו מנצחת B.",
      do=hide("dp-h0-4") + show("dp-th6", *w.cells("dp", "v", (0, 1), (6,))) + show("dp-h1-6"),
      say="אבל נָאִיב בֵּייס מכפיל בה שוב. אֵיי יורדת לאפס נקודה אפס שתיים שבע, ובִּי לאפס נקודה אפס שלוש שבע חמש. ועכשיו מנצחת בִּי."),
    P("העדות של זד נספרה פעמיים, והכריעה בגלל זה. כשהמאפיינים תלויים, אותה עדות נספרת יותר מפעם אחת.", do=show("d-r"),
      say="העדות של זֶד נספרה פעמיים, והכריעה בגלל זה. כשהמאפיינים תלויים, אותה עדות נספרת יותר מפעם אחת."),
    P("זה המחיר של ההנחה הנאיבית: ככל שהמאפיינים תלויים יותר, הציונים מעוותים יותר.", hold=0.4),
], stage=w.col(DUP, w.note("שלוש העמודות האחרונות: אחרי הוספת עותק של $z$"), gap=24),
   panel=st.card("d-r", "rule", "🔑 תלות ← אותה עדות נספרת פעמיים"))

smooth = Scene("smooth", "החלקה", [
    P("בעיה שנייה: אפסים. נחזור לששת הכלבים מטבלת החיות, ונוסיף מאפיין: צבע. ג׳ינג׳י, שחור או לבן.", do=show("sm")),
    P("אף כלב באימון לא היה ג׳ינג׳י. ועכשיו הגיעה חיה קטנה, ארוכת שיער וג׳ינג׳ית."),
    P("בלי החלקה, ה-likelihoods של כלב הם שליש, חצי, ואפס.", do=show("sm-th2", *w.cells("sm", "v", range(3), (2,))),
      say="בלי החלקה, הלַייקְלִיהוּדִים של כלב הם שליש, חצי, ואפס."),
    P("המכפלה של כלב היא אפס. ערך אחד שלא נראה מחק את שני המאפיינים האחרים.", do=show("s-1", "sm-h2-2")),
    P("החלקה מוסיפה אחת לכל ספירה, ולמכנה מוסיפים את מספר הערכים: שתיים לגודל ולשיער, שלוש לצבע.", do=show("s-2")),
    P("עכשיו: 3/8, 4/8 ו-1/9. אף אחד מהם לא אפס.",
      do=hide("sm-h2-2") + show("sm-th3", *w.cells("sm", "v", range(3), (3,))),
      say="עכשיו: שלוש חלקי שמונה, ארבע חלקי שמונה, ואחת חלקי תשע. אף אחד מהם לא אפס."),
    P("המכפלה יוצאת בערך 0.021. קטנה, אבל לא אפס, והגודל והשיער עדיין משפיעים.", do=show("s-3"),
      say="המכפלה יוצאת בערך אפס נקודה אפס שתיים אחת. קטנה, אבל לא אפס, והגודל והשיער עדיין משפיעים.", hold=0.4),
], stage=w.col(SM, w.eq("s-1", r"$\tfrac{2}{6}\cdot\tfrac{3}{6}\cdot 0 = 0$", 40),
               w.eq("s-2", r"$\hat{P}(v \mid \text{dog}) = \dfrac{\text{count} + 1}{6 + \#\text{values}}$", 36),
               w.eq("s-3", r"$\tfrac{3}{8}\cdot\tfrac{4}{8}\cdot\tfrac{1}{9} \approx 0.021$", 38), gap=20),
   panel=st.card("s-r", "trap", "⚠️ ג׳ינג׳י לא נראה בכלבים ← ציון הכלב אפס").replace(' h"', '"'))

gen = Scene("gen", "גנרטיבי", [
    P("ודבר אחרון: נאיב בייס לומד איך נראות הדוגמאות של כל מחלקה. טבלה של $P(X|Y)$ לכל מאפיין.",
      say="ודבר אחרון: נָאִיב בֵּייס לומד איך נראות הדוגמאות של כל מחלקה. טבלה של פִּי של אִקְס בהינתן וַואי לכל מאפיין.",
      do=show("g-1")),
    P("ועם הטבלה הזו אפשר גם לייצר דוגמה. מגרילים מחלקה לפי ה-prior: יצאה A.", do=show("g-2"),
      say="ועם הטבלה הזו אפשר גם לייצר דוגמה. מגרילים מחלקה לפי הפְּרַיְיאוֹר: יצאה אֵיי."),
    P("ואז כל מאפיין לפי ה-likelihood שלו בתוך A: איקס יצא אחת, וזד יצא אפס. הנה דוגמה חדשה של A.", do=show("g-3"),
      say="ואז כל מאפיין לפי הלַייקְלִיהוּד שלו בתוך אֵיי: אִקְס יצא אחת, וזֶד יצא אפס. הנה דוגמה חדשה של אֵיי."),
    P("מודל שלומד איך הנתונים נוצרים נקרא **גנרטיבי**. מודל שלומד רק את הגבול בין המחלקות נקרא דיסקרימינטיבי.",
      do=show("g-r"), hold=0.4),
], stage=w.col(st.card("g-1", "ex", "נלמד: $P(Y)$, ו-$P(x_j \\mid Y)$ לכל מאפיין"),
               w.chip("g-2", "1 · מגרילים מחלקה לפי ה-prior: יצאה A", "info"),
               w.chip("g-3", "2 · מגרילים כל מאפיין בתוך A: יצא $x=1,\\ z=0$", "ok"), gap=26),
   panel=st.card("g-r", "rule", "🔑 גנרטיבי = לומד איך כל מחלקה נראית"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו מהתשובות היא חיסרון של נאיב בייס?"),
    P("ראינו עכשיו: עמודה תלויה אחת הפכה את ההחלטה. ההנחה הנאיבית נשברת, ואותה עדות נספרת פעמיים. זו התשובה.",
      do=show("q-ok-2")),
    P("נטייה ל-overfitting? לנאיב בייס יש רק הסתברות אחת לכל ערך ולכל מחלקה. זה מודל פשוט, לא מודל שמשנן.",
      do=show("q-x-0", "q-n-0"),
      say="נטייה לאוֹבֶרְפִיטִינְג? לנָאִיב בֵּייס יש רק הסתברות אחת לכל ערך ולכל מחלקה. זה מודל פשוט, לא מודל שמשנן."),
    P("סילום? אין כאן מרחק שמערבב סולמות של מאפיינים שונים. כל מאפיין נספר לבד, בתוך כל מחלקה.",
      do=show("q-x-1", "q-n-1")),
    P("ומאפיינים קטגוריאליים? זה בדיוק מה שנאיב בייס הבסיסי סופר. כל הטבלאות שראינו היו כאלה.",
      do=show("q-x-3", "q-n-3"),
      say="ומאפיינים קטגוריאליים? זה בדיוק מה שנָאִיב בֵּייס הבסיסי סופר. כל הטבלאות שראינו היו כאלה."),
    P("המלכודת: overfitting נשמע כמו חיסרון כללי של כל מודל. אבל הוא החיסרון של עץ עמוק, לא של מודל שסופר.",
      do=show("x-t"),
      say="המלכודת: אוֹבֶרְפִיטִינְג נשמע כמו חיסרון כללי של כל מודל. אבל הוא החיסרון של עץ עמוק, לא של מודל שסופר.",
      hold=0.6),
], stage=w.col(w.stem(Q["id"], "איזו תשובה מהווה חיסרון של מודל Naive Bayes?"),
               w.opts("q", OPTS, notes={0: "מודל פשוט", 1: "אין מרחק", 3: "סופר קטגוריות"}, size=28), gap=20),
   panel=st.card("x-1", "ex", "עמודה תלויה הפכה את ההחלטה").replace(' h"', '"')
         + st.card("x-t", "trap", "⚠️ overfitting: החיסרון של עץ עמוק"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: את ה-likelihoods מכפילים רק בזכות הנחה, שבתוך המחלקה מאפיין אחד לא מספר כלום על האחר.", do=show("r-1"),
      say="לסיכום: את הלַייקְלִיהוּדִים מכפילים רק בזכות הנחה, שבתוך המחלקה מאפיין אחד לא מספר כלום על האחר."),
    P("כשהם תלויים, אותה עדות נספרת שוב ושוב, וזה יכול להפוך החלטה.", do=show("r-2")),
    P("ערך שלא נראה במחלקה איפס לה את הציון, וההחלקה הוסיפה אחת לכל ספירה.", do=show("r-3")),
    P("ונאיב בייס לומד איך כל מחלקה נראית, ולכן הוא גנרטיבי. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=w.col(w.eq("r-eq", r"$P(Y)\prod_j P(x_j \mid Y)$", 64, hidden=False), gap=30),
   panel=st.card("r-1", "rule", "🔑 מכפלה ← הנחת אי-תלות בתוך המחלקה")
         + st.card("r-2", "rule", "🔑 תלות ← עדות כפולה")
         + st.card("r-3", "rule", "🔑 החלקה: $+1$ לכל ספירה")
         + st.card("r-4", "rule", "🔑 גנרטיבי: לומד את $P(X \\mid Y)$"))

VIDEO = Video("C08", "cls", "נאיב בייס: ההנחה הנאיבית", [title, naive, dep, smooth, gen, exam, recap],
              bank={"24B-C-Q17": 2})
