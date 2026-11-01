# -*- coding: utf-8 -*-
"""C07 · נאיב בייס: חישוב מטבלה — video script (build: tools/video/vid.py C07).

Source: courses/cls/C07.js (lesson + goal: prior and likelihood from a table, multiply into the posterior and
classify; the traps of the denominator, the direction of the bar and the missing prior).
Exam part: an original "בסגנון מבחן" question (bank={}): three classes, two words, priors 0.5 / 0.3 / 0.2. No fresh
bank question fits: every computational Naive Bayes item is a quiz/boss ref (23S-A-Q14, 26B-A-Q16, 25B-A-Q9,
22B-C-Q1) or a content twin of one (SAMP-1-Q8 and 26S-A-Q13 repeat the 23S-A-Q14 template and answer; CLAUDE-26-Q13
is the C07-Q4 spam/ham scenario); 24S-C-Q14 has a doubtful key (see the report).
Not used on purpose: the C07 quiz data (the red/blue/green table, 1 of 3 vs 2 of 6, spam 0.3 / 0.05 / 0.8,
0.4 / 0.3 / 0.5, Y/N 0.6·0.5·0.4 vs 0.4·0.3·0.9, 0.75 / 0.6) and the boss table C09-Q4 (frequency / device, which
gives 0.167 vs 0.1). The bar-direction trap is explained without numbers.
Worked example (new, cross-checked with scikit-learn's CategoricalNB, alpha ~ 0): 10 animals (size, hair) -> cat or
dog; query (small, long): cat 4/10 · 3/4 · 2/4 = 0.15, dog 6/10 · 2/6 · 3/6 = 0.1 -> cat, although the prior favours
dog; normalised 0.6 / 0.4.
"""
import math

import numpy as np
from sklearn.naive_bayes import CategoricalNB

from vid import Scene, Video, show, hide, BANK
import stage as st
import w_cls as w
from w_cls import P

# ---------------------------------------------------------------- the table (verified)
ROWS = [("קטן", "ארוך", "חתול"), ("גדול", "ארוך", "כלב"), ("קטן", "ארוך", "חתול"), ("קטן", "קצר", "כלב"),
        ("גדול", "קצר", "כלב"), ("קטן", "קצר", "חתול"), ("גדול", "ארוך", "כלב"), ("קטן", "ארוך", "כלב"),
        ("גדול", "קצר", "חתול"), ("גדול", "קצר", "כלב")]
QX = ("קטן", "ארוך")
CLS = ("חתול", "כלב")
n = len(ROWS)
size = {c: sum(1 for r in ROWS if r[2] == c) for c in CLS}
assert size == {"חתול": 4, "כלב": 6}
prior = {c: size[c] / n for c in CLS}
lik = {c: [sum(1 for r in ROWS if r[2] == c and r[j] == QX[j]) / size[c] for j in (0, 1)] for c in CLS}
assert lik["חתול"] == [3 / 4, 2 / 4] and lik["כלב"] == [2 / 6, 3 / 6]
score = {c: prior[c] * lik[c][0] * lik[c][1] for c in CLS}
assert math.isclose(score["חתול"], 0.15) and math.isclose(score["כלב"], 0.1)
assert prior["כלב"] > prior["חתול"] and score["חתול"] > score["כלב"]       # the likelihood overturns the prior
tot = sum(score.values())
assert math.isclose(score["חתול"] / tot, 0.6) and math.isclose(score["כלב"] / tot, 0.4)
# the traps
assert sum(1 for r in ROWS if r[0] == "קטן" and r[2] == "חתול") / n == 0.3          # / table size: wrong
small = [r for r in ROWS if r[0] == "קטן"]
assert len(small) == 5 and sum(1 for r in small if r[2] == "חתול") / len(small) == 0.6   # P(cat | small): posterior
# scikit-learn agrees (alpha ~ 0 = plain counts)
enc = {"קטן": 0, "גדול": 1, "ארוך": 0, "קצר": 1}
X = np.array([[enc[r[0]], enc[r[1]]] for r in ROWS])
y = np.array([CLS.index(r[2]) for r in ROWS])
nb = CategoricalNB(alpha=1e-10, force_alpha=True).fit(X, y)
pp = nb.predict_proba([[enc[QX[0]], enc[QX[1]]]])[0]
assert np.allclose(pp, [0.6, 0.4], atol=1e-6) and nb.predict([[0, 0]])[0] == 0

# ---------------------------------------------------------------- the exam-style question (verified)
PRI = {"ספורט": 0.5, "פוליטיקה": 0.3, "טכנולוגיה": 0.2}
L1 = {"ספורט": 0.1, "פוליטיקה": 0.2, "טכנולוגיה": 0.2}
L2 = {"ספורט": 0.6, "פוליטיקה": 0.7, "טכנולוגיה": 0.8}
POST = {c: PRI[c] * L1[c] * L2[c] for c in PRI}
LIKO = {c: L1[c] * L2[c] for c in PRI}
SUMS = {c: PRI[c] * (L1[c] + L2[c]) for c in PRI}
assert {c: round(v, 3) for c, v in POST.items()} == {"ספורט": 0.03, "פוליטיקה": 0.042, "טכנולוגיה": 0.032}
assert max(POST, key=POST.get) == "פוליטיקה"
assert max(LIKO, key=LIKO.get) == "טכנולוגיה" and round(LIKO["טכנולוגיה"], 2) == 0.16
assert max(PRI, key=PRI.get) == "ספורט"
assert max(SUMS, key=SUMS.get) == "ספורט" and round(SUMS["ספורט"], 2) == 0.35
OPTS = [("a", "פוליטיקה, עם ציון $0.042$"),
        ("b", "טכנולוגיה, עם ציון $0.16$"),
        ("c", "ספורט, כי ה-prior שלה הגבוה ביותר"),
        ("d", "ספורט, עם ציון $0.35$")]

# ---------------------------------------------------------------- figures
TAB = w.table("tb", ["גודל", "שיער", "חיה"], [list(r) for r in ROWS], size=26, rtl=True)
cat_rows = [i for i, r in enumerate(ROWS) if r[2] == "חתול"]
dog_rows = [i for i, r in enumerate(ROWS) if r[2] == "כלב"]
cat_small = [i for i in cat_rows if ROWS[i][0] == "קטן"]
cat_long = [i for i in cat_rows if ROWS[i][1] == "ארוך"]
dog_small = [i for i in dog_rows if ROWS[i][0] == "קטן"]
dog_long = [i for i in dog_rows if ROWS[i][1] == "ארוך"]
assert (len(cat_small), len(cat_long), len(dog_small), len(dog_long)) == (3, 2, 2, 3)

NEWS = w.table("nw", ["", "ספורט", "פוליטיקה", "טכנולוגיה"],
               [["prior", "0.5", "0.3", "0.2"], ["$P(w_1 \\mid \\cdot)$", "0.1", "0.2", "0.2"],
                ["$P(w_2 \\mid \\cdot)$", "0.6", "0.7", "0.8"],
                ["ציון", "0.03", "0.042", "0.032"]], size=30, rtl=True)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נסווג עם נאיב בייס מתוך טבלה, ביד."),
    P("נספור prior ו-likelihood, נכפיל, ונשווה. ונכיר את שלוש המלכודות שהמבחן אוהב."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">C07 · מסווגים</div><h1>נאיב בייס: חישוב מטבלה</h1>'
         '<p>prior כפול likelihood לכל מחלקה, והגבוה מנצח. והמכנה של likelihood הוא גודל המחלקה.</p>',
   layout="title")

terms = Scene("terms", "שלושה מונחים", [
    P("נאיב בייס שואל, לכל מחלקה, כמה סביר שהדוגמה החדשה שייכת אליה. שלושה מונחים, והמבחן מבלבל ביניהם בכוונה."),
    P("prior: כמה המחלקה שכיחה באימון, לפני שראינו את הדוגמה.", do=show("t-1")),
    P("likelihood: בתוך המחלקה, כמה שכיח הערך שקיבלנו. זה $P(X|Y)$, המחלקה **אחרי** הקו.", do=show("t-2"),
      say="לַייקְלִיהוּד: בתוך המחלקה, כמה שכיח הערך שקיבלנו. זה פִּי של אִקְס בהינתן וַואי, המחלקה אחרי הקו."),
    P("posterior: מה שמשווים בסוף. $P(Y|X)$, המחלקה **לפני** הקו. הוא פרופורציונלי ל-likelihood כפול prior.",
      do=show("t-3", "t-f"),
      say="פּוֹסְטֵירְיוֹר: מה שמשווים בסוף. פִּי של וַואי בהינתן אִקְס, המחלקה לפני הקו. הוא פרופורציונלי ללַייקְלִיהוּד כפול פְּרַיְיאוֹר."),
    P("ועם כמה מאפיינים, ה-likelihood הוא מכפלה. המכפלה מותרת רק בזכות הנחת אי-תלות בהינתן המחלקה, וזה החלק הנאיבי.",
      do=show("t-4"), hold=0.4,
      say="ועם כמה מאפיינים, הלַייקְלִיהוּד הוא מכפלה. המכפלה מותרת רק בזכות הנחת אי-תלות בהינתן המחלקה, וזה החלק הנאיבי."),
], stage=w.col(st.card("t-1", "ex", "<b>prior</b> &nbsp; $P(Y)$: שכיחות המחלקה"),
               st.card("t-2", "ex", "<b>likelihood</b> &nbsp; $P(X \\mid Y)$: הערך, <b>בתוך</b> המחלקה"),
               st.card("t-3", "ex", "<b>posterior</b> &nbsp; $P(Y \\mid X)$: מה שמשווים"), gap=22),
   panel=w.eq("t-f", r"$P(Y \mid X) \;\propto\; P(X \mid Y)\cdot P(Y)$", 40)
         + w.eq("t-4", r"$P(x_1, x_2 \mid Y) = P(x_1 \mid Y)\cdot P(x_2 \mid Y)$", 32))

count = Scene("count", "סופרים", [
    P("הטבלה: עשר חיות, גודל ושיער, ומה החיה. הגיעה חיה חדשה: קטנה, עם שיער ארוך.", do=show("c-q")),
    P("prior: ארבעה חתולים ושישה כלבים. ארבע עשיריות ושש עשיריות.", do=show("c-1") + show(*w.cells("tb", "h", cat_rows, (2,)))),
    P("likelihood של חתול: סופרים **רק בתוך ארבעת החתולים**. שלושה מהם קטנים, ושניים ארוכי שיער.",
      do=hide(*w.cells("tb", "h", cat_rows, (2,))) + show(*w.cells("tb", "h", cat_small, (0,)), *w.cells("tb", "h", cat_long, (1,)),
                                                          *w.cells("tb", "d", dog_rows, (0, 1, 2)), "c-2"),
      say="לַייקְלִיהוּד של חתול: סופרים רק בתוך ארבעת החתולים. שלושה מהם קטנים, ושניים ארוכי שיער."),
    P("כלומר שלושה רבעים, ושני רבעים.",
      say="כלומר שלושה רבעים, ושני רבעים."),
    P("ושל כלב: בתוך ששת הכלבים, שניים קטנים ושלושה ארוכי שיער. שתי שישיות ושלוש שישיות.",
      do=hide(*w.cells("tb", "h", cat_small, (0,)), *w.cells("tb", "h", cat_long, (1,)), *w.cells("tb", "d", dog_rows, (0, 1, 2)))
      + show(*w.cells("tb", "h", dog_small, (0,)), *w.cells("tb", "h", dog_long, (1,)), *w.cells("tb", "d", cat_rows, (0, 1, 2)), "c-3")),
    P("המכנה של likelihood הוא תמיד **גודל המחלקה**, לא כל הטבלה.", do=show("c-r"),
      say="המכנה של לַייקְלִיהוּד הוא תמיד גודל המחלקה, לא כל הטבלה.", hold=0.4),
], stage=w.col(TAB, gap=10),
   panel=st.card("c-q", "plain", "דוגמה חדשה: <b>קטן, ארוך</b>")
         + st.card("c-1", "ex", "prior: חתול $\\tfrac{4}{10}$ · כלב $\\tfrac{6}{10}$")
         + st.card("c-2", "ex", "חתול: $\\tfrac{3}{4}$ קטן · $\\tfrac{2}{4}$ ארוך")
         + st.card("c-3", "ex", "כלב: $\\tfrac{2}{6}$ קטן · $\\tfrac{3}{6}$ ארוך")
         + st.card("c-r", "rule", "🔑 המכנה של likelihood = גודל המחלקה"))

post = Scene("post", "מכפילים ומשווים", [
    P("עכשיו מכפילים, לכל מחלקה: prior כפול כל ה-likelihoods שלה.", do=show("p-0"),
      say="עכשיו מכפילים, לכל מחלקה: פְּרַיְיאוֹר כפול כל הלַייקְלִיהוּדִים שלה."),
    P("חתול: ארבע עשיריות, כפול שלושה רבעים, כפול חצי. יוצא 0.15.", do=show("p-1"),
      say="חתול: ארבע עשיריות, כפול שלושה רבעים, כפול חצי. יוצא 0.15."),
    P("כלב: שש עשיריות, כפול שליש, כפול חצי. יוצא 0.1.", do=show("p-2"),
      say="כלב: שש עשיריות, כפול שליש, כפול חצי. יוצא 0.1."),
    P("0.15 גדול מ-0.1, ולכן חתול. ושימו לב: לפי ה-prior לבד, הכלבים שכיחים יותר. ה-likelihood הפך את ההחלטה.",
      do=show("p-3"),
      say="אפס נקודה אחת חמש גדול מאפס נקודה אחת, ולכן חתול. ושימו לב: לפי הפְּרַיְיאוֹר לבד, הכלבים שכיחים יותר. הלַייקְלִיהוּד הפך את ההחלטה."),
    P("אפשר גם לנרמל: נחלק את שני הציונים בסכום שלהם, ונקבל 0.6 ו-0.4. החתול עדיין מנצח.", do=show("p-4"),
      say="אפשר גם לנרמל: נחלק את שני הציונים בסכום שלהם, ונקבל אפס נקודה שש ואפס נקודה ארבע. החתול עדיין מנצח.", hold=0.4),
], stage=w.col(
    w.eq("p-0", r"$\text{score}(Y) = P(Y)\cdot P(x_1 \mid Y)\cdot P(x_2 \mid Y)$", 36),
    st.card("p-1", "plain", r"<span style='font-size:38px'>חתול: $\tfrac{4}{10}\cdot\tfrac{3}{4}\cdot\tfrac{2}{4} = 0.15$</span>"),
    st.card("p-2", "plain", r"<span style='font-size:38px'>כלב: $\tfrac{6}{10}\cdot\tfrac{2}{6}\cdot\tfrac{3}{6} = 0.1$</span>"),
    w.eq("p-4", r"$\dfrac{0.15}{0.25} = 0.6 \qquad \dfrac{0.1}{0.25} = 0.4$", 38), gap=26),
   panel=st.card("p-3", "ex", "חתול מנצח, אף שה-prior של כלב גבוה יותר")
)

traps = Scene("traps", "המלכודות", [
    P("שלוש מלכודות. הראשונה: לחלק בכל הטבלה. שלושה חתולים קטנים מתוך עשר זה 0.3, וזה לא likelihood.", do=show("m-1"),
      say="שלוש מלכודות. הראשונה: לחלק בכל הטבלה. שלושה חתולים קטנים מתוך עשר זה 0.3, וזה לא לַייקְלִיהוּד."),
    P("השנייה: לספור בתוך הערך במקום בתוך המחלקה. מתוך חמש החיות הקטנות, שלוש הן חתולים.", do=show("m-2")),
    P("אבל שלוש חמישיות זה הסיכוי של חתול **בהינתן** קטן."),
    P("כלומר זה כבר posterior של מאפיין אחד, ולא likelihood. אם הוא מופיע באפשרויות, הוא שם כדי לבלבל.",
      say="כלומר זה כבר פּוֹסְטֵירְיוֹר של מאפיין אחד, ולא לַייקְלִיהוּד. אם הוא מופיע באפשרויות, הוא שם כדי לבלבל."),
    P("והשלישית: כיוון הקו. כשנותנים הסתברות שהמחלקה כתובה בה לפני הקו, היא כבר posterior. לא מכפילים אותה שוב ב-prior.",
      do=show("m-3"),
      say="והשלישית: כיוון הקו. כשנותנים הסתברות שהמחלקה כתובה בה לפני הקו, היא כבר פּוֹסְטֵירְיוֹר. לא מכפילים אותה שוב בפְּרַיְיאוֹר."),
    P("ואם נתנו prior רק של מחלקה אחת מתוך שתיים, השני הוא אחת פחות הראשון.", do=show("m-4"),
      say="ואם נתנו פְּרַיְיאוֹר רק של מחלקה אחת מתוך שתיים, השני הוא אחת פחות הראשון."),
    P("ומכפילים כל likelihood ב-prior **של אותה מחלקה**, לא של השנייה.",
      say="ומכפילים כל לַייקְלִיהוּד בפְּרַיְיאוֹר של אותה מחלקה, לא של השנייה.", hold=0.4),
], stage=w.col(st.card("m-1", "trap", "⚠️ $\\tfrac{3}{10}$: חילקו בכל הטבלה, לא בגודל המחלקה"),
               st.card("m-2", "trap", "⚠️ $\\tfrac{3}{5}$ = הסיכוי לחתול <b>בהינתן</b> קטן: זה כבר posterior"),
               st.card("m-3", "trap", "⚠️ $P(Y \\mid X)$ נתון? לא מכפילים ב-prior"),
               st.card("m-4", "trap", "⚠️ prior חסר = $1 -$ השני, ובזוג הנכון"), gap=22),
   panel=w.eq("m-f", r"$P(Y \mid X) \;\propto\; P(X \mid Y)\cdot P(Y)$", 40, hidden=False))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: מסווגים כתבה לאחת משלוש מחלקות. בכתבה יש שתי מילים, $w_1$ ו-$w_2$.",
      say="שאלה בסגנון מבחן: מסווגים כתבה לאחת משלוש מחלקות. בכתבה יש שתי מילים, דַּבֶּלְיוּ אחת ודַּבֶּלְיוּ שתיים."),
    P("נתונים ה-priors וה-likelihoods של שתי המילים בכל מחלקה. לאיזו מחלקה תסווג הכתבה?", do=show("nw"),
      say="נתונים הפְּרַיְיאוֹרים והלַייקְלִיהוּדים של שתי המילים בכל מחלקה. לאיזו מחלקה תסווג הכתבה?"),
    P("ספורט: חצי, כפול 0.1, כפול 0.6. יוצא 0.03.", do=show("e-1"),
      say="ספורט: חצי, כפול 0.1, כפול 0.6. יוצא 0.03."),
    P("פוליטיקה: 0.3, כפול 0.2, כפול 0.7. יוצא 0.042.", do=show("e-2"),
      say="פוליטיקה: 0.3, כפול 0.2, כפול 0.7. יוצא 0.042."),
    P("טכנולוגיה: 0.2, כפול 0.2, כפול 0.8. יוצא 0.032. הגבוה הוא פוליטיקה.", do=show("e-3", "q-ok-a"),
      say="טכנולוגיה: 0.2, כפול 0.2, כפול 0.8. יוצא 0.032. הגבוה הוא פוליטיקה."),
    P("0.16 זה טכנולוגיה בלי prior: רק מכפלת ה-likelihoods. בלי prior זה לא posterior.", do=show("q-x-b", "q-n-b"),
      say="אפס נקודה אחת שש זה טכנולוגיה בלי פְּרַיְיאוֹר: רק מכפלת הלַייקְלִיהוּדים. בלי פְּרַיְיאוֹר זה לא פּוֹסְטֵירְיוֹר."),
    P("ספורט בגלל ה-prior? ה-prior הוא רק גורם אחד במכפלה, וכאן ה-likelihood של המילה הראשונה מוריד אותו.", do=show("q-x-c", "q-n-c"),
      say="ספורט בגלל הפְּרַיְיאוֹר? הפְּרַיְיאוֹר הוא רק גורם אחד במכפלה, וכאן הלַייקְלִיהוּד של המילה הראשונה מוריד אותו."),
    P("ו-0.35 יוצא מחיבור ה-likelihoods במקום הכפלה: חצי כפול 0.7. נאיב בייס מכפיל.", do=show("q-x-d", "q-n-d"),
      say="ואפס נקודה שלוש חמש יוצא מחיבור הלַייקְלִיהוּדים במקום הכפלה: חצי כפול 0.7. נָאִיב בֵּייס מכפיל."),
    P("המלכודת כאן: כל אחת מהאפשרויות השגויות מנצחת לפי חישוב חלקי. רק המכפלה המלאה עונה נכון.", do=show("e-t"), hold=0.6),
], stage=w.col(NEWS, w.eq("e-1", r"$0.5\cdot 0.1\cdot 0.6 = 0.03$", 32), w.eq("e-2", r"$0.3\cdot 0.2\cdot 0.7 = 0.042$", 32),
               w.eq("e-3", r"$0.2\cdot 0.2\cdot 0.8 = 0.032$", 32), gap=14),
   panel=w.stem("בסגנון מבחן", "כתבה עם המילים $w_1$ ו-$w_2$. לפי נאיב בייס (בלי $P(X)$), לאיזו מחלקה תסווג, ובאיזה ציון?")
         + w.opts("q", OPTS, notes={"b": "בלי prior", "c": "prior לבד", "d": "חיבור"}, size=26)
         + st.card("e-t", "trap", "⚠️ כל מסיח = חישוב חלקי שמנצח"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: prior הוא גודל המחלקה חלקי כל הטבלה.", do=show("r-1"),
      say="לסיכום: פְּרַיְיאוֹר הוא גודל המחלקה חלקי כל הטבלה."),
    P("likelihood סופרים בתוך המחלקה, וחלקי גודל המחלקה. ועם כמה מאפיינים, מכפילים.", do=show("r-2"),
      say="לַייקְלִיהוּד סופרים בתוך המחלקה, וחלקי גודל המחלקה. ועם כמה מאפיינים, מכפילים."),
    P("הציון: prior כפול ה-likelihoods, והגבוה מנצח.", do=show("r-3"),
      say="הציון: פְּרַיְיאוֹר כפול הלַייקְלִיהוּדים, והגבוה מנצח."),
    P("ותבדקו את כיוון הקו: מחלקה לפני הקו זה כבר posterior. נתראה ברמה הבאה.", do=show("r-4"),
      say="ותבדקו את כיוון הקו: מחלקה לפני הקו זה כבר פּוֹסְטֵירְיוֹר. נתראה ברמה הבאה.", hold=0.6),
], stage=w.col(w.eq("r-eq", r"$\hat{y} = \arg\max_Y\ P(Y)\prod_j P(x_j \mid Y)$", 56, hidden=False), gap=30),
   panel=st.card("r-1", "rule", "🔑 prior = גודל המחלקה ⁄ כל הטבלה")
         + st.card("r-2", "rule", "🔑 likelihood: בתוך המחלקה")
         + st.card("r-3", "rule", "🔑 prior × likelihoods")
         + st.card("r-4", "trap", "⚠️ $P(Y \\mid X)$ כבר posterior"))

VIDEO = Video("C07", "cls", "נאיב בייס: חישוב מטבלה", [title, terms, count, post, traps, exam, recap], bank={})
