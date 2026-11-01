# -*- coding: utf-8 -*-
"""U07 · אנסמבל ועיבוד מקדים — video script (build: tools/video/vid.py U07).

Source: courses/unsup/U07.js (lesson + goal). Exam anchors: 25B-A-Q17 (which function scales: MinMax) and 25B-A-Q8
(which reason is NOT a good reason to filter features: none, all three are good), neither used by a guided quiz.
Avoided on purpose: the quiz items of U07 (SAMP-2-Q1, 24S-A-Q20, PRAC-X-Q16 and the originals: "which is bagging",
the mislabeled example's weight, "describe boosting", MinMax of 20 in [10, 50], correlation vs variance, the outlier
2,4,6,8,10 / 14 with thresholds 2 vs 3, the AND table) and their twins (SAMP-3-Q1, 24S-C-Q12, 25B-A-Q11, PRAC-X-Q5).
Random Forest is not named, "parallel / in series" descriptions are not used, AdaBoost runs on CLEAN data (no
mislabeled point, no noise-sensitivity claim), and the outlier check uses one threshold with two new values.
New data: bagging on nine points (x = 2 mislabeled): three bootstrap samples (numpy seed 3) -> three best stumps ->
votes at x = 2.2 (one tree fooled, the majority not). AdaBoost on six clean points + + - - + -: three weighted stumps
(2.5, 5.5, 4.5) whose weighted vote is right on all six. Flow: MinMax of 4, 10, 13, 16; z of 17 and of 3 against
4, 6, 9, 13 (sample sd, n - 1) with threshold 2.
Every number is computed and asserted here.
"""
import math
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

X = np.arange(1,10)
Y = np.array([-1, 1, -1, -1, 1, 1, 1, 1, 1])                # x = 2 is labelled + by mistake
NOISE = 1                                                    # index of x = 2


def stumps(xs):
    u = np.unique(xs)
    return [((a + b) / 2, s) for a, b in zip(u, u[1:]) for s in (1, -1)]


def pred(t, s, x):
    return np.where(x > t, s, -s)


def best_stump(xs, ys, w):
    res = sorted((float(w[pred(t, s, xs) != ys].sum()), t, s) for t, s in stumps(xs))
    assert res[0][0] < res[1][0] - 1e-12, "a unique best stump"
    return res[0]


# ---------------------------------------------------------------- bagging (scene "bag")
rng = np.random.default_rng(3)
BAGS = []
for _ in range(3):
    idx = rng.integers(0, 9, 9)
    e, t, s = best_stump(X[idx], Y[idx], np.ones(9))
    BAGS.append((sorted((idx + 1).tolist()), t, s, int(pred(t, s, np.array([2.2]))[0])))
assert [b[0] for b in BAGS] == [[1, 1, 2, 2, 3, 6, 8, 8, 8], [1, 2, 3, 3, 4, 5, 6, 7, 7], [1, 2, 4, 4, 4, 5, 5, 6, 8]]
assert [(b[1], b[2], b[3]) for b in BAGS] == [(1.5, 1, 1), (4.5, 1, -1), (4.5, 1, -1)]
assert all(len(set(b[0])) < 9 for b in BAGS)                # drawn with replacement: repeats, and missing points
assert sorted(b[3] for b in BAGS) == [-1, -1, 1] and sum(b[3] for b in BAGS) < 0     # majority: negative
assert BAGS[0][0].count(2) == 2                              # tree 1 drew the mislabeled point twice

# ---------------------------------------------------------------- AdaBoost (scene "boost"): clean data, no noise
XB = np.arange(1, 7)
YB = np.array([1, 1, -1, -1, 1, -1])                         # + + - - + - : no single threshold separates it
assert all((pred(t, s_, XB) != YB).any() for t, s_ in stumps(XB))
WB, ROUNDS, FB = np.ones(6) / 6, [], np.zeros(6)
for _ in range(3):
    e, t, s_ = best_stump(XB, YB, WB)
    a = 0.5 * math.log((1 - e) / e)
    p = pred(t, s_, XB)
    FB = FB + a * p
    ROUNDS.append((WB.copy(), t, s_, e, a, list(np.flatnonzero(p != YB) + 1)))
    WB = WB * np.exp(-a * YB * p)
    WB /= WB.sum()
assert [(r[1], r[2]) for r in ROUNDS] == [(2.5, -1), (5.5, -1), (4.5, 1)]
assert [r[5] for r in ROUNDS] == [[5], [3, 4], [1, 2, 6]]
assert np.isclose(ROUNDS[2][3], 3 / 16)                       # round 3: three light points, 3/16 together          # who each stump gets wrong
assert np.allclose(ROUNDS[1][0], [0.1, 0.1, 0.1, 0.1, 0.5, 0.1])            # after round 1: x = 5 holds half
assert np.allclose(ROUNDS[2][0], [1 / 16, 1 / 16, 1 / 4, 1 / 4, 5 / 16, 1 / 16])
assert np.isclose(ROUNDS[1][3], 0.2)                          # round 2's stump: x = 3 and 4 cost 0.1 + 0.1
assert (np.sign(FB) == YB).all()                              # the weighted vote of the three stumps is right on all six
assert all(r[4] > 0 for r in ROUNDS)

# ---------------------------------------------------------------- flow (scene "flow")
MM = [4, 10, 13, 16]
MMS = [(v - min(MM)) / (max(MM) - min(MM)) for v in MM]
assert MMS == [0, 0.5, 0.75, 1]
ZD = [4, 6, 9, 13]
MU = sum(ZD) / len(ZD)
SS = sum((v - MU) ** 2 for v in ZD)
SD = math.sqrt(SS / (len(ZD) - 1))
Z17 = (17 - MU) / SD
assert MU == 8 and SS == 46 and num(SD) == "3.92" and num(Z17) == "2.3" and 2 < Z17 < 3
SDP = math.sqrt(SS / len(ZD))
assert num(SDP) == "3.39" and num((17 - MU) / SDP) == "2.65"
Z3 = (3 - MU) / SD
assert num(Z3) == "-1.28" and abs(Z3) < 2 < Z17

# ---------------------------------------------------------------- exams
Q1 = BANK["25B-A-Q17"]
assert Q1["correctIndex"] == 3 and "MinMax" in Q1["options"][3] and "LinearRegression" in Q1["options"][0]
assert "kMeans" in Q1["options"][1] and "PreCision" in Q1["options"][2]
OPT1 = [(0, "פונקציית LinearRegression"), (1, "פונקציית kMeans"), (2, "פונקציית PreCision"), (3, "פונקציית MinMax")]
Q2 = BANK["25B-A-Q8"]
assert Q2["correctIndex"] == 2 and "אינה מהווה" in Q2["question"]
assert "תלות גבוהה" in Q2["options"][0] and "סיבוכיות" in Q2["options"][1] and "להעלות את הדיוק" in Q2["options"][3]
OPT2 = [(0, "תלות גבוהה בין מאפיינים: סינון ממש הכרחי לאלגוריתמים מסוימים"),
        (1, "סינון מאפיינים מוריד מהסיבוכיות של למידת המודל"),
        (2, "כל התשובות אינן נכונות (כלומר, כל הסיבות בהן כן טובות)"),
        (3, "סינון מאפיינים שאינם תורמים מידע עלול אף להעלות את הדיוק")]

# ---------------------------------------------------------------- widgets
COL = lambda y: 1 if y > 0 else 0
rad = lambda w: 8 + 40 * math.sqrt(w)
BASE = [(x, COL(y), rad(1 / 9), None) for x, y in zip(X, Y)]
lab = lambda w: {0.1: "1/10", 0.5: "1/2", 0.25: "1/4", 0.1667: "1/6"}.get(round(float(w), 4))
layer = lambda W_: [(int(x), COL(y), rad(w), lab(w)) for x, y, w in zip(XB, YB, W_)]
NL = W.numline("ab", XB, 0.5, 6.5, [("ab-0", True, layer(ROUNDS[0][0]), None), ("ab-1", True, layer(ROUNDS[1][0]), None),
                                    ("ab-2", True, layer(ROUNDS[2][0]), None),
                                    ("ab-f", True, [(int(x), COL(y), 18, "✓") for x, y in zip(XB, YB)], None),
                                    ("ab-t1", True, [], ROUNDS[0][1]), ("ab-t2", True, [], ROUNDS[1][1]),
                                    ("ab-t3", True, [], ROUNDS[2][1])], w=900, h=260)
NLB = W.numline("nb", X, 0.5, 9.5, [("nb-0", False, BASE, None), ("nb-q", True, [(2.2, None, 12, "2.2")], None)],
                w=940, h=230)
vote = lambda v: '<b style="color:#fb7185">+</b>' if v > 0 else '<b style="color:#2dd4bf">−</b>'
BT = W.table("bt", ["עץ", "מדגם (עם חזרות)", "סף", "קול ל-2.2"],
             [[str(i + 1), " ".join(map(str, b[0])), num(b[1]), vote(b[3])] for i, b in enumerate(BAGS)],
             cls="sm", hide={(r, c) for r in range(3) for c in (1, 2, 3)}, hl={(0,1)})

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נשלב הרבה מודלים חלשים לתחזית אחת: bagging מול boosting."),
    P("נראה איך AdaBoost מעדכן משקלים בין עץ לעץ, ונעבור על עיבוד מקדים: סילום וחריגים.",
      say="נראה איך אָדָה בּוּסְט מעדכן משקלים בין עץ לעץ, ונעבור על עיבוד מקדים: סילום וחריגים."),
    P("ובסוף שתי שאלות ממבחן.", hold=0.4),
], stage='<div class="kicker">U07 · אשכול, טקסט ותמונה</div><h1>אנסמבל ועיבוד מקדים</h1>'
         '<p>להבחין בין bagging ל-boosting, לנמק למה AdaBoost רגיש לרעש, ולענות על שאלות הסילום, בחירת המאפיינים והחריגים.</p>',
   layout="title")

bag = Scene("bag", "Bagging", [
    P("תשע נקודות על ישר. השליליות משמאל, החיוביות מימין, ונקודה אחת, 2, סומנה חיובית בטעות.",
      say="תשע נקודות על ישר. השליליות משמאל, החיוביות מימין, ונקודה אחת, שתיים, סומנה חיובית בטעות."),
    P("המודל הפנימי הוא עץ החלטה הכי פשוט: סף אחד. מימין לסף חיובי, ומשמאל שלילי."),
    P("Bagging: מגרילים מדגמים חדשים מתוך הנתונים, עם חזרות. חלק מהנקודות נבחרות פעמיים, וחלק לא בכלל.",
      say="בֶּגִינְג: מגרילים מדגמים חדשים מתוך הנתונים, עם חזרות. חלק מהנקודות נבחרות פעמיים, וחלק לא בכלל.",
      do=show(*[f"bt-{r}-1" for r in range(3)])),
    P("כל עץ רואה רק את המדגם שלו, ובוחר לפיו את הסף.", do=show(*[f"bt-{r}-2" for r in range(3)])),
    P("העץ הראשון הגריל את הנקודה הרועשת פעמיים, ולכן שם את הסף ב-1.5.",
      say="העץ הראשון הגריל את הנקודה הרועשת פעמיים, ולכן שם את הסף באחת וחצי.", do=show("bt-hl-0-1")),
    P("עכשיו נקודה חדשה, 2.2. העץ הראשון אומר חיובי, ושני האחרים אומרים שלילי.",
      say="עכשיו נקודה חדשה, שתיים נקודה שתיים. העץ הראשון אומר חיובי, ושני האחרים אומרים שלילי.",
      do=show("nb-q", *[f"bt-{r}-3" for r in range(3)])),
    P("הצבעת רוב: שלילי. עץ אחד נפל ברעש, והרוב תיקן אותו.", do=show("b-c1")),
    P("ככל שיש יותר מדגמים ויותר קולות, לנקודה רועשת אחת יש פחות כוח על ההכרעה.", do=show("b-c2"), hold=0.4),
], stage=W.stack(NLB, BT, gap=30),
   panel=st.card("b-c1", "ex", "קולות: + , − , − ← <b>שלילי</b>")
         + st.card("b-c2", "rule", "🔑 מדגם מוגרל לכל עץ, והכרעה ברוב קולות"))

boost = Scene("boost", "AdaBoost", [
    P("ועכשיו AdaBoost, על נתונים נקיים: שש נקודות, פלוס, פלוס, מינוס, מינוס, פלוס, מינוס.",
      say="ועכשיו אָדָה בּוּסְט, על נתונים נקיים: שש נקודות, פלוס, פלוס, מינוס, מינוס, פלוס, מינוס.", do=show("ab-0")),
    P("אין סף אחד שמפריד אותן: הנקודה 5 חיובית, והיא תקועה בין שליליות.",
      say="אין סף אחד שמפריד אותן: הנקודה חמש חיובית, והיא תקועה בין שליליות."),
    P("לכל נקודה יש משקל, ובהתחלה לכולן שישית. גודל העיגול הוא המשקל."),
    P("עץ ראשון: הסף הטוב ביותר הוא 2.5. הוא טועה רק בנקודה 5.",
      say="עץ ראשון: הסף הטוב ביותר הוא שתיים וחצי. הוא טועה רק בנקודה חמש.", do=show("ab-t1")),
    P("עכשיו המשקלים מתעדכנים: לנקודה 5 יש חצי מכל המשקל, ולכל אחת מהאחרות עשירית.",
      say="עכשיו המשקלים מתעדכנים: לנקודה חמש יש חצי מכל המשקל, ולכל אחת מהאחרות עשירית.",
      do=hide("ab-0") + show("ab-1")),
    P("העץ השני נבחר לפי המשקלים החדשים. הסף 5.5 טועה ב-3 וב-4, אבל ביחד הן שוות רק חמישית.",
      say="העץ השני נבחר לפי המשקלים החדשים. הסף חמש וחצי טועה בשלוש ובארבע, אבל ביחד הן שוות רק חמישית.",
      do=hide("ab-t1") + show("ab-t2")),
    P("שוב עדכון: 3 ו-4 מקבלות רבע כל אחת. העץ השלישי, סף 4.5, טועה ב-1, 2 ו-6: נקודות קלות, ביחד פחות מחמישית.",
      say="שוב עדכון: שלוש וארבע מקבלות רבע כל אחת. העץ השלישי, סף ארבע וחצי, טועה באחת, בשתיים ובשש: נקודות קלות, ביחד פחות מחמישית.",
      do=hide("ab-1", "ab-t2") + show("ab-2", "ab-t3")),
    P("כל עץ מקבל קול לפי הדיוק שלו, ומחברים את הקולות. שלושה ספים פשוטים, וכל שש הנקודות מסווגות נכון.",
      do=hide("ab-2", "ab-t3") + show("ab-f", "o-c1")),
    P("ההבדל מ-bagging: שם כל עץ הגריל מדגם לבד. כאן כל עץ תלוי במשקלים שהעצים שלפניו השאירו.",
      say="ההבדל מבֶּגִינְג: שם כל עץ הגריל מדגם לבד. כאן כל עץ תלוי במשקלים שהעצים שלפניו השאירו.",
      do=show("o-c2"), hold=0.4),
], stage=W.stack(NL, W.note("o-l", "<span class='m'>גודל העיגול = המשקל · אדום = חיובי · ירוק = שלילי</span>",
                            hidden=False, size=28), gap=20),
   panel=st.card("o-c1", "ex", "3 עצים, קול משוקלל ← 6 מתוך 6 נכונות")
         + st.card("o-c2", "rule", "🔑 AdaBoost: משקלים לנקודות, קול משוקלל לכל עץ"))

flow = Scene("flow", "עיבוד מקדים", [
    P("ועכשיו עיבוד מקדים. סילום MinMax מכווץ כל מאפיין לטווח 0 עד 1.",
      say="ועכשיו עיבוד מקדים. סילום מִין מַקְס מכווץ כל מאפיין לטווח אפס עד אחת.", do=show("f-mm")),
    P("הערכים 4, 10, 13 ו-16: מחסירים את המינימום 4, ומחלקים ברוחב הטווח, 12.",
      say="הערכים ארבע, עשר, שלוש עשרה ושש עשרה: מחסירים את המינימום ארבע, ומחלקים ברוחב הטווח, שתים עשרה."),
    P("יוצא 0, חצי, שלושה רבעים ו-1.", say="יוצא אפס, חצי, שלושה רבעים ואחת.", do=show("f-mm2")),
    P("חריגים: בודקים כמה סטיות תקן ערך רחוק מהממוצע. ערכי המאפיין 4, 6, 9 ו-13, והממוצע 8.",
      say="חריגים: בודקים כמה סטיות תקן ערך רחוק מהממוצע. ערכי המאפיין ארבע, שש, תשע ושלוש עשרה, והממוצע שמונה.",
      do=show("f-z")),
    P("סטיית תקן מדגמית: סכום ריבועי הסטיות 46, חלקי 3, ושורש. בערך 3.92.",
      say="סטיית תקן מדגמית: סכום ריבועי הסטיות ארבעים ושש, חלקי שלוש, ושורש. בערך שלוש נקודה תשעים ושתיים.",
      do=show("f-z2")),
    P("שימו לב: מחלקים ב-$n$ פחות 1, לא ב-$n$.", say="שימו לב: מחלקים באֶן פחות אחת, לא באֶן.", do=show("f-c1")),
    P("ערך חדש, 17: הוא רחוק 9 מהממוצע, כלומר בערך 2.3 סטיות תקן.",
      say="ערך חדש, שבע עשרה: הוא רחוק תשע מהממוצע, כלומר בערך שתיים נקודה שלוש סטיות תקן.", do=show("f-z3")),
    P("נניח שהסף הוא 2 סטיות תקן: 17 עובר אותו, ולכן חריג. ערך 3 רחוק רק 1.28 סטיות תקן, ונשאר.",
      say="נניח שהסף הוא שתי סטיות תקן: שבע עשרה עובר אותו, ולכן חריג. ערך שלוש רחוק רק אחת נקודה עשרים ושמונה סטיות תקן, ונשאר.",
      do=show("f-z4", "f-c2"), hold=0.4),
], stage=W.stack(W.big("f-mm", "$x' = \\dfrac{x - \\min}{\\max - \\min}$", size=42),
                 W.note("f-mm2", W.ltr("4, 10, 13, 16 &nbsp;⟶&nbsp; 0, 0.5, 0.75, 1"), size=34),
                 W.big("f-z", "$z = \\dfrac{x - \\mu}{\\sigma}$", size=42),
                 W.note("f-z2", "σ = √(46 ⁄ 3) ≈ 3.92", size=34),
                 W.note("f-z3", W.ltr("z(17) = 9 ⁄ 3.92 ≈ <b>2.3</b>"), size=34),
                 W.note("f-z4", W.ltr("z(3) = −5 ⁄ 3.92 ≈ <b>−1.28</b>"), size=34), gap=14),
   panel=st.card("f-c1", "trap", "⚠️ סטיית תקן מדגמית: חלוקה ב-" + W.ltr("n − 1"))
         + st.card("f-c2", "rule", "🔑 משווים את |z| לסף שהשאלה קבעה"))

exam1 = Scene("exam1", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו פונקציה משמשת לסילום?"),
    P("MinMax, בדיוק מה שעשינו עכשיו: כל מאפיין לטווח 0 עד 1.",
      say="מִין מַקְס, בדיוק מה שעשינו עכשיו: כל מאפיין לטווח אפס עד אחת.", do=show("p-ok-3")),
    P("רגרסיה לינארית ו-K-Means הם אלגוריתמי למידה, לא הכנה של הנתונים. ו-precision הוא מדד הערכה.",
      say="רגרסיה לינארית וקֵיי מִינְס הם אלגוריתמי למידה, לא הכנה של הנתונים. ופְּרֵיסִיזְיָה הוא מדד הערכה.",
      do=show("p-x-0", "p-n-0", "p-x-1", "p-n-1", "p-x-2", "p-n-2"), hold=0.3),
], stage=W.stack(W.big("x-mm", "$x' = \\dfrac{x - \\min}{\\max - \\min}$", size=46, hidden=False), gap=10),
   panel=f'<div class="qstem"><span class="qid">{Q1["id"]}</span>איזו פונקציה משמשת לסילום?</div>'
         + st.options("p", OPT1, notes={0: "אלגוריתם", 1: "אלגוריתם", 2: "מדד"}))

exam2 = Scene("exam2", "שאלה ממבחן", [
    P("ועוד שאלה: איזו סיבה אינה סיבה טובה לסנן מאפיינים? כאן בודקים כל סיבה לגופה."),
    P("תלות גבוהה: למשל נאיב בייס מניח שהמאפיינים בלתי תלויים, ומאפיין משוכפל נספר אצלו פעמיים. סיבה טובה.",
      do=show("q-n-0")),
    P("פחות מאפיינים: פחות חישוב ומודל פשוט יותר. גם זו סיבה טובה.", do=show("q-n-1")),
    P("ומאפיין שלא תורם מידע הוא רעש: למשל ב-KNN הוא מעוות את המרחק. הסרתו יכולה דווקא להעלות דיוק. סיבה טובה.",
      say="ומאפיין שלא תורם מידע הוא רעש: למשל בקֵיי אֶן אֶן הוא מעוות את המרחק. הסרתו יכולה דווקא להעלות דיוק. סיבה טובה.",
      do=show("q-n-3")),
    P("שלוש הסיבות טובות, אז אין אף סיבה שאינה טובה. התשובה: כל התשובות אינן נכונות.", do=show("q-ok-2")),
    P("המלכודת היא השלילה בשאלה: מחפשים את הסיבה הגרועה, ולא את הטובה. כשכולן טובות, בוחרים באפשרות שאומרת את זה.",
      do=show("x-t"), hold=0.4),
], stage=W.stack(st.card("x-t", "trap", "⚠️ השאלה שואלת מה <b>אינו</b> סיבה טובה"), gap=10),
   panel=f'<div class="qstem"><span class="qid">{Q2["id"]}</span>איזו סיבה <b>אינה</b> סיבה טובה לסינון '
         'מאפיינים מה-dataset?</div>'
         + '<div class="u-qsm">' + st.options("q", OPT2, notes={0: "טובה", 1: "טובה", 3: "טובה"}) + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: ב-bagging כל עץ מקבל מדגם מוגרל משלו, ובסוף סופרים קולות.",
      say="לסיכום: בבֶּגִינְג כל עץ מקבל מדגם מוגרל משלו, ובסוף סופרים קולות.", do=show("r-1")),
    P("ב-AdaBoost לכל נקודה יש משקל. נקודה שעץ טעה בה מקבלת יותר משקל, והעץ הבא נבחר לפי המשקלים.",
      say="באָדָה בּוּסְט לכל נקודה יש משקל. נקודה שעץ טעה בה מקבלת יותר משקל, והעץ הבא נבחר לפי המשקלים.",
      do=show("r-2")),
    P("וכל עץ מצביע לפי הדיוק שלו. ככה כמה ספים פשוטים פותרים דפוס שסף אחד לא פותר.", do=show("r-3")),
    P("סילום הוא MinMax, וחריג נבדק בסטיות תקן מדגמיות מול הסף שנקבע. נתראה ברמה הבאה.",
      say="סילום הוא מִין מַקְס, וחריג נבדק בסטיות תקן מדגמיות מול הסף שנקבע. נתראה ברמה הבאה.", do=show("r-4"),
      hold=0.6),
], stage=NL.replace('id="ab', 'id="ac').replace('id="ac-f" class="h"', 'id="ac-f"'),
   panel=st.card("r-1", "rule", "🔑 Bagging: מדגם מוגרל לכל עץ, רוב קולות")
         + st.card("r-2", "rule", "🔑 AdaBoost: משקלים שמתעדכנים אחרי כל עץ")
         + st.card("r-3", "ex", "קול משוקלל לפי דיוק")
         + st.card("r-4", "rule", "🔑 MinMax · z עם " + W.ltr("n − 1")))

VIDEO = Video("U07", "unsup", "אנסמבל ועיבוד מקדים", [title, bag, boost, flow, exam1, exam2, recap],
              bank={"25B-A-Q17": 3, "25B-A-Q8": 2})
