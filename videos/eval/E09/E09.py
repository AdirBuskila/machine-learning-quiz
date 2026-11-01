# -*- coding: utf-8 -*-
"""E09 · בחירת מאפיינים ו-PCA — video script (build: tools/video/vid.py E09).

Source: courses/eval/E09.js (goal: which feature to filter and which to keep, correlation vs causation, and PCA:
variance, orthogonality and the processing order). Exam anchors: SAMP-1-Q12 (which function is not used for feature
selection) and PRAC-X-Q21 (PCA on identical constant images), neither used by a guided quiz nor a twin of a quiz item.
Not used on purpose: the E09 quiz scenarios (PCA weights 0.45 / 0.25, the processing-order question, a near-constant
feature, ice cream and drownings, 50 -> 5 components, the eigenvector check of [[4, 1], [2, 3]], "feature selection
vs PCA") and the refs 22B-A-Q20, 26B-A-Q19, 24S-A-Q13, plus their twins 22B-C-Q20, 26S-A-Q20, CLAUDE-26-Q20,
22B-A-Q7 / 22B-C-Q7, 25S-A-Q16 and SAMP-2-Q13 (high vs low variance), which mirror quiz items.
Worked examples: an apartments table (rooms vs area r ≈ 0.997, a constant "city" column), fans sold vs sunburns
(r ≈ 0.998, both driven by summer), and PCA on 8 standardized 2-D points (sklearn PCA: 90.3% / 9.7%, axes at 45°).
"""
import numpy as np
from sklearn.decomposition import PCA

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_eval as w

# ---------------------------------------------------------------- feature selection table (verified)
ROOMS = np.array([2, 3, 3, 4, 5, 5])
AREA = np.array([55, 72, 75, 95, 118, 121])
CITY = ["חיפה"] * 6
EXP = ["לא", "לא", "לא", "כן", "כן", "כן"]                     # the label: price above 2M?
r_ra = np.corrcoef(ROOMS, AREA)[0, 1]
assert f"{r_ra:.3f}" == "0.997"
assert len(set(CITY)) == 1                                       # variance 0: no information at all
# the label separates by area (and by rooms): area > 90 <-> "כן"
assert all((a > 90) == (e == "כן") for a, e in zip(AREA, EXP))

# ---------------------------------------------------------------- correlation, not causation (verified)
MONTHS = ["מרץ", "אפר׳", "מאי", "יוני", "יולי", "אוג׳"]
FANS = np.array([20, 35, 60, 90, 85, 40])
BURN = np.array([3, 6, 11, 17, 15, 7])
r_fb = np.corrcoef(FANS, BURN)[0, 1]
assert f"{r_fb:.3f}" == "0.998"

# ---------------------------------------------------------------- PCA (verified)
X = np.array([[1, 2], [2, 1.5], [3, 3.5], [4, 3], [5, 5.5], [6, 4.5], [2.5, 3.5], [4.5, 3.5]])
Zs = (X - X.mean(0)) / X.std(0, ddof=1)
pca = PCA().fit(Zs)
EV = pca.explained_variance_ratio_
assert [f"{100 * e:.1f}" for e in EV] == ["90.3", "9.7"]
PC1, PC2 = pca.components_
assert abs(abs(PC1[0]) - 2 ** -0.5) < 1e-9 and abs(abs(PC1[1]) - 2 ** -0.5) < 1e-9 and PC1[0] * PC1[1] > 0
assert abs(PC1 @ PC2) < 1e-12                                    # perpendicular
C = np.cov(Zs.T)
lam = np.linalg.eigvalsh(C)[::-1]
assert np.allclose(C @ PC1, lam[0] * PC1)                        # C·v = λ·v
assert abs(EV.sum() - 1) < 1e-12

# ---------------------------------------------------------------- exams
Q1 = BANK["SAMP-1-Q12"]
assert Q1["correctIndex"] == 2 and "t-distribution standardization" in Q1["options"][2]
Q2 = BANK["PRAC-X-Q21"]
assert Q2["correctIndex"] == 3 and "5x5" in Q2["question"]
IMG = np.full((40, 25), 100.0)                                   # 40 identical 5x5 images, every pixel = 100
CI = np.cov(IMG.T)
assert CI.shape == (25, 25) and np.allclose(CI, 0)               # the covariance matrix is all zeros
assert np.allclose(np.linalg.eigvalsh(CI), 0)                    # every direction has variance 0: no maximum
assert np.allclose(IMG - IMG.mean(0), 0)                         # centring removes the 100
O1, O2 = w.opts(Q1), w.opts(Q2)

# ---------------------------------------------------------------- figures
L = lambda t: f"<span class='ev-l'>{t}</span>"
FT = w.table("ft", ["חדרים", "שטח (מ״ר)", "עיר", "מעל 2 מיליון?"],
             [[L(r), L(a), c, e] for r, a, c, e in zip(ROOMS, AREA, CITY, EXP)], hide_rows=False, size=27)
PL = w.Plot("cp", (-0.3, 5.3), (0, 100), w=760, h=330, xticks=range(6), yticks=(0, 50, 100),
            xlabels=dict(enumerate(MONTHS)), xname="", yname="")
PL.poly("f", [(i, v) for i, v in enumerate(FANS)], "a", width=5)
PL.poly("b", [(i, 100 * v / BURN.max()) for i, v in enumerate(BURN)], "c", width=5, dash=True)
CPSVG = PL.svg()
LEG = ('<div class="ev-row" style="gap:30px;font-size:26px"><span style="color:#38bdf8">— מאווררים שנמכרו</span>'
       '<span style="color:#f472b6">- - כוויות שמש (מנורמל)</span></div>')

PP = w.Plot("pp", (-2.2, 2.2), (-2.2, 2.2), w=520, h=520, xticks=(-2, -1, 0, 1, 2), yticks=(-2, -1, 0, 1, 2),
            xname="z₁", yname="z₂", pad=(56, 24, 24, 56))
PP.pts("x", Zs.tolist(), "ink", r=10)
PP.arrow("v1", 0, 0, 1.6, 1.6, "b", width=7)                     # PC1 is the rising diagonal (asserted above)
PP.arrow("v2", 0, 0, -0.75, 0.75, "c", width=6)                  # PC2 perpendicular to it
PP.text("t1", 1.75, 1.25, "PC1", "b", size=26)
PP.text("t2", -1.05, 1.0, "PC2", "c", size=26)
PCSVG = PP.svg()

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד אילו מאפיינים לזרוק ואילו לשמור, ונבדיל בין קורלציה לסיבתיות."),
    P("ואז נכיר את PCA: איך הוא בונה מאפיינים חדשים, ולמה הוא מחפש שונות גבוהה.", hold=0.4,
      say="ואז נכיר את פִּי סִי אֵיי: איך הוא בונה מאפיינים חדשים, ולמה הוא מחפש שונות גבוהה."),
], stage='<div class="kicker">E09 · הערכת מודל</div><h1>בחירת מאפיינים ו-PCA</h1>'
         '<p>מה לסנן ומה לשמור, קורלציה מול סיבתיות, ו-PCA: שונות, רכיבים מאונכים וסדר העיבוד.</p>', layout="title")

fs = Scene("fs", "מה לסנן", [
    P("בחירת מאפיינים: משאירים **חלק** מהעמודות המקוריות, כמו שהן, וזורקים את השאר. פחות עמודות, מודל פשוט יותר.",
      do=show("ft")),
    P("כאן רוצים לחזות אם דירה עולה מעל שני מיליון. שלושה מאפיינים: חדרים, שטח ועיר."),
    P("עמודת העיר זהה בכל השורות. השונות שלה אפס: היא לא יכולה להבדיל בין שום שתי דירות. מסננים.", do=show("f-1")),
    P("חדרים ושטח עולים יחד: המתאם ביניהם בערך 0.997. הם אומרים כמעט אותו דבר, אז מספיק אחד מהם.", do=show("f-2"),
      say="חדרים ושטח עולים יחד: המתאם ביניהם בערך אפס נקודה תשע מאות תשעים ושבע. הם אומרים כמעט אותו דבר, אז מספיק אחד מהם."),
    P("ואת השטח שומרים: הקשר שלו **לתווית** חזק. מעל תשעים מטר, תמיד כן.", do=show("f-3")),
    P("שימו לב להבדל: קשר חזק **בין שני מאפיינים** הוא כפילות. קשר חזק **עם התווית** הוא בדיוק מה שמחפשים.",
      do=show("f-k"), hold=0.4),
], stage=w.col(FT),
   panel=st.card("f-1", "ex", "עיר: שונות 0 ← <b>מסננים</b>")
         + st.card("f-2", "ex", "חדרים ↔ שטח: $r \\approx 0.997$ ← <b>אחד מיותר</b>")
         + st.card("f-3", "ex", "שטח ↔ תווית: חזק ← <b>שומרים</b>")
         + st.card("f-k", "rule", "🔑 מאפיין↔מאפיין חזק: כפילות · מאפיין↔תווית חזק: זהב"))

corr = Scene("corr", "קורלציה וסיבתיות", [
    P("לאורך חצי שנה: כמה מאווררים נמכרו, וכמה כוויות שמש טופלו. שני הגרפים עולים ויורדים יחד.", do=show("cp-f", "cp-b")),
    P("המתאם ביניהם בערך 0.998. זו **קורלציה** חזקה מאוד.", do=show("c-1"),
      say="המתאם ביניהם בערך אפס נקודה תשע מאות תשעים ושמונה. זו קורלציה חזקה מאוד."),
    P("אבל מאווררים לא גורמים לכוויות, וכוויות לא גורמות לקנות מאוורר. שניהם נובעים מסיבה שלישית: חום הקיץ.",
      do=show("c-2")),
    P("גרפים שנעים יחד מראים קשר. הם לא אומרים מי גורם למי.", do=show("c-k"), hold=0.4),
], stage=w.col(CPSVG, LEG, gap=14),
   panel=st.card("c-1", "ex", "$r \\approx 0.998$")
         + st.card("c-2", "plain", "הסיבה המשותפת: <b>הקיץ</b>")
         + st.card("c-k", "rule", "🔑 קורלציה ≠ סיבתיות"))

pca = Scene("pca", "PCA", [
    P("עכשיו PCA. הוא לא בוחר עמודות קיימות. הוא **בונה** עמודות חדשות, משילוב של כל הישנות.",
      say="עכשיו פִּי סִי אֵיי. הוא לא בוחר עמודות קיימות. הוא בונה עמודות חדשות, משילוב של כל הישנות."),
    P("שלב ראשון: מנרמלים כל מאפיין, כדי שמאפיין עם יחידות גדולות לא ישתלט על השונות. אלה שמונה נקודות מנורמלות.",
      do=show("a-1")),
    P("עכשיו מחפשים את הכיוון שבו הנקודות **הכי מפוזרות**. זה הרכיב הראשון: האלכסון.", do=show("pp-v1", "pp-t1")),
    P("לאורכו נמצאים תשעים נקודה שלוש אחוז מכל השונות. שם המידע.", do=show("a-2")),
    P("הרכיב השני **מאונך** לראשון, ולוקח את מה שנשאר: תשע נקודה שבע אחוז.", do=show("pp-v2", "pp-t2", "a-3")),
    P("העמודה החדשה הראשונה היא צירוף לינארי: חצי שורש שתיים כפול הראשון, ועוד חצי שורש שתיים כפול השני.", do=show("a-4")),
    P("אם נשמור רק את הרכיב הראשון, ירדנו משתי עמודות לעמודה אחת, ושמרנו מעל תשעים אחוז מהשונות.", do=show("a-5")),
    P("ושימו לב: לא השתמשנו בתווית אף פעם. PCA לא מונחה. ואחריו, מאמנים את המודל על העמודות החדשות.",
      say="ושימו לב: לא השתמשנו בתווית אף פעם. פִּי סִי אֵיי לא מונחה. ואחריו, מאמנים את המודל על העמודות החדשות.",
      do=show("a-k"), hold=0.4),
], stage=w.col(PCSVG),
   panel=st.card("a-1", "plain", "קודם מנרמלים כל מאפיין")
         + st.card("a-2", "ex", "<span class='ev-l'>PC1</span>: 90.3% מהשונות")
         + st.card("a-3", "ex", "<span class='ev-l'>PC2</span> ⟂ <span class='ev-l'>PC1</span>: 9.7%")
         + st.card("a-4", "plain", "$\\text{PC}_1 = \\tfrac{1}{\\sqrt{2}} z_1 + \\tfrac{1}{\\sqrt{2}} z_2$")
         + st.card("a-5", "ex", "רכיב אחד ← שומרים 90.3%")
         + st.card("a-k", "rule", "🔑 <span class='ev-l'>PCA</span> בונה עמודות חדשות, בכיווני השונות הגבוהה"))

exam1 = Scene("exam1", "שאלה ממבחן", [
    P("שאלה ממבחן: איזו פונקציה **אינה** בשימוש לבחירת מאפיינים?"),
    P("שונות: ראינו אותה בעמודת העיר. שונות אפס, מסננים. היא כן בשימוש.", do=show("p-x-0", "p-n-0")),
    P("אנטרופיה מותנית מודדת כמה מאפיין מספר על המחלקה. מאפיין שלא מספר כלום, מסננים. כן בשימוש.",
      do=show("p-x-1", "p-n-1")),
    P("ומקדם המתאם של פירסון: ראינו אותו בחדרים מול שטח. כן בשימוש.", do=show("p-x-3", "p-n-3")),
    P("נשארה הסטנדרטיזציה: היא **משנה את הסקאלה** של מאפיין, ולא מחליטה אם לשמור אותו. זה סילום. זו התשובה.",
      do=show("p-ok-2"), hold=0.5),
], stage=w.col(st.card("y-1", "rule", "🔑 בחירת מאפיינים = <b>להחליט מה נשאר</b><br>סילום = לשנות את הסקאלה"), gap=20),
   panel=w.stem(Q1["id"], "איזו פונקציה <b>אינה</b> בשימוש עבור <span class='ev-l'>feature selection</span>?")
         + '<div class="ev-opts">' + st.options("p", O1, notes={0: "שונות 0 ← סינון", 1: "מידע על המחלקה",
                                                                  3: "כפילות"}) + "</div>")
exam1.phrases[0].do += show("y-1")

exam2 = Scene("exam2", "עוד שאלה", [
    P("ועוד שאלה: מאגר שכולו תמונות זהות, חמש על חמש, וכל פיקסל בגוון מאה. מורידים עם PCA ל-ממד אחד.",
      say="ועוד שאלה: מאגר שכולו תמונות זהות, חמש על חמש, וכל פיקסל בגוון מאה. מורידים עם פִּי סִי אֵיי לממד אחד."),
    P("כל פיקסל הוא מאפיין, וכל מאפיין זהה בכל התמונות. אז השונות של כל אחד אפס, וגם השונות המשותפת.", do=show("z-1")),
    P("מטריצת השונות המשותפת כולה אפסים. וכל כיוון שנבחר, הפיזור לאורכו אפס.", do=show("z-2")),
    P("אין כיוון ״הכי מפוזר״, ולכן הרכיב הראשון לא מוגדר. זו התשובה.", do=show("z-3", "z-ok-3")),
    P("הרכיב לא ״מכיל את הערך מאה״: PCA מחסיר את הממוצע, ומה שנשאר הוא אפסים. רכיב הוא כיוון, לא בהירות.",
      do=show("z-x-0", "z-n-0"),
      say="הרכיב לא מכיל את הערך מאה: פִּי סִי אֵיי מחסיר את הממוצע, ומה שנשאר הוא אפסים. רכיב הוא כיוון, לא בהירות."),
    P("אלפיים וחמש מאות בכל תא, או מטריצת יחידה, דורשים שונות שאינה אפס. וכאן אין שום פיזור.",
      do=show("z-x-1", "z-n-1", "z-x-2", "z-n-2"), hold=0.6),
], stage=w.col(w.stem(Q2["id"], "מאגר של תמונות זהות 5×5, כל פיקסל = 100. <span class='ev-l'>PCA</span> לממד 1. "
                                      "מה מטריצת השונות המשותפת, ומה הרכיב הראשון?"),
               w.eq("z-1", "$\\text{Var}(x_j) = 0, \\quad \\text{Cov}(x_i, x_j) = 0$", 34),
               w.eq("z-2", "$C = \\mathbf{0}_{25 \\times 25}$", 40),
               st.card("z-3", "rule", "🔑 אין שונות ← אין כיוון של שונות מקסימלית"), gap=26),
   panel='<div class="ev-opts sm">' + st.options("z", O2, notes={0: "הממוצע מוחסר", 1: "אין פיזור", 2: "אין פיזור"})
         + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: שונות אפס, מסננים. שני מאפיינים כמעט זהים, מסננים אחד. קשר חזק לתווית, שומרים.", do=show("r-1")),
    P("קורלציה לא אומרת מי גורם למי.", do=show("r-2")),
    P("PCA בונה עמודות חדשות, מאונכות, בכיווני השונות הגבוהה. בלי תוויות.", do=show("r-3"),
      say="פִּי סִי אֵיי בונה עמודות חדשות, מאונכות, בכיווני השונות הגבוהה. בלי תוויות."),
    P("ובלי נרמול, היחידות הגדולות קובעות את הכיוון במקום הנתונים. נתראה בשלב הבא.", do=show("r-4"), hold=0.6),
], stage=w.col(PCSVG.replace('id="pp', 'id="pq').replace('class="h"', 'class=""')),
   panel=st.card("r-1", "rule", "🔑 שונות 0 ← סינון · כפילות ← סינון אחד · קשר לתווית ← שמירה")
         + st.card("r-2", "rule", "🔑 קורלציה ≠ סיבתיות")
         + st.card("r-3", "rule", "🔑 <span class='ev-l'>PCA</span>: רכיבים חדשים, מאונכים, שונות גבוהה")
         + st.card("r-4", "trap", "⚠️ מנרמלים <b>לפני</b> שמחשבים שונות"))

VIDEO = w.check_say(Video("E09", "eval", "בחירת מאפיינים ו-PCA", [title, fs, corr, pca, exam1, exam2, recap],
                          bank={"SAMP-1-Q12": 2, "PRAC-X-Q21": 3}))
