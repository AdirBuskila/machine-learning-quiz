# -*- coding: utf-8 -*-
"""R05 · רגולריזציה ו-underfitting — video script (build: tools/video/vid.py R05).

Source: courses/reg/R05.js (goal: what λ does to the weights, what fixes underfitting and what makes it worse, and the
explanation that cannot explain a strange model). Exam part: an original "בסגנון מבחן" question (bank={}): w_A = (4, 0)
vs w_B = (2, 2) with λ = 0.1 -> L2 penalises A more (1.6 vs 0.8), L1 penalises both alike (0.4).
Not used on purpose: 24B-C-Q11 and its twin 24S-A-Q15 (their distractors are R05-Q3's options), 26S-A-Q5 (R05-Q2's key),
24B-C-Q1 (doubtful key), the "can't explain" family (22B-A-Q1, 23B-B-Q1, 24B-A-Q1, CLAUDE-26-Q18 = R05-Q5/Q6).
Avoided: every R05 quiz item (MSE = 2 with w = (2, −3) and λ = 0.1; "λ very large -> underfitting"; the underfitting /
overfitting remedy lists; ŷ = 0·x₁ + 0·x₂ + 5 and ŷ = 500·x₁ + 900·x₂), the boss R09-Q5 and the quiz refs 24S-C-Q10,
25S-A-Q6, 24S-A-Q9. No remedy is named for underfitting (the video shows what an underfitting model looks like:
the best line through a parabola, MSE 12 on its own training data), no sentence "large λ -> underfitting" or
"regularisation = penalty term against overfitting": the λ sweep shows the numbers only.
Own data: MSE = 1.2, w = (4, −2), λ = 0.05; a ridge sweep on six points (λ = 0, 1, 10, 100) solved in closed form with
numpy and cross-checked with scikit-learn's Ridge when it is installed; y = (x − 3)² on x = 0..6.
"""
from fractions import Fraction as Fr

import numpy as np

from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_reg as w
from w_reg import SAY

# ---------------------------------------------------------------- numbers (checked)
MSE0, W1, W2, LAM = Fr(6, 5), 4, -2, Fr(1, 20)
PEN2 = LAM * (W1 ** 2 + W2 ** 2)
PEN1 = LAM * (abs(W1) + abs(W2))
assert PEN2 == 1 and MSE0 + PEN2 == Fr(11, 5) and PEN1 == Fr(3, 10) and MSE0 + PEN1 == Fr(3, 2)

# ridge sweep: minimise (1/n)·Σ(ŷ − y)² + λ·(w1² + w2²), the intercept is not penalised
X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]], dtype=float)
Y = np.array([3, 4, 8, 8, 12, 12], dtype=float)
N = len(Y)
Xc, yc = X - X.mean(0), Y - Y.mean()
LAMS = [0, 1, 10, 100]
SWEEP = []
for lam in LAMS:
    wv = np.linalg.solve(Xc.T @ Xc / N + lam * np.eye(2), Xc.T @ yc / N)
    b = Y.mean() - X.mean(0) @ wv
    mse = float(np.mean((X @ wv + b - Y) ** 2))
    # the gradient of the regularised cost is zero there (it is the minimum)
    g = 2 / N * Xc.T @ (Xc @ wv - yc) + 2 * lam * wv
    assert np.allclose(g, 0)
    SWEEP.append((lam, wv, b, mse))
norms = [float(np.linalg.norm(s[1])) for s in SWEEP]
mses = [s[3] for s in SWEEP]
assert all(norms[i + 1] < norms[i] for i in range(3)) and all(mses[i + 1] > mses[i] for i in range(3))
assert norms[-1] < 0.1 and mses[0] < 0.1 and mses[-1] > 10
try:                                    # cross-check with scikit-learn (Ridge minimises Σe² + α‖w‖², α = λ·n)
    from sklearn.linear_model import Ridge
    for lam, wv, b, _ in SWEEP[1:]:
        r = Ridge(alpha=lam * N).fit(X, Y)
        assert np.allclose(r.coef_, wv) and np.isclose(r.intercept_, b)
except ImportError:
    pass
SW = [(lam, f"{wv[0]:.2f}", f"{wv[1]:.2f}", f"{mse:.2f}") for lam, wv, b, mse in SWEEP]
assert SW[0][1:] == ("1.23", "0.90", "0.07") and SW[-1][1:] == ("0.05", "0.05", "10.95")
VAR_Y = float(np.mean((Y - Y.mean()) ** 2))
assert round(VAR_Y, 2) == 12.14 and mses[-1] < VAR_Y           # λ = 100: almost the "always the mean" model

# the narrated sweep values
assert all(float(v) < 0.4 for v in SWEEP[2][1]) and round(SWEEP[2][3], 1) == 5.2

# what underfitting looks like: y = (x − 3)², the best straight line is ŷ = 4 with training MSE 12
UX = np.arange(7, dtype=float)
UY = (UX - 3) ** 2
coef, *_ = np.linalg.lstsq(np.c_[np.ones(7), UX], UY, rcond=None)
assert np.allclose(coef, [4, 0]) and np.isclose(np.mean((coef[0] + coef[1] * UX - UY) ** 2), 12)

# exam part (original): L1 vs L2 on w_A = (4, 0) and w_B = (2, 2), λ = 0.1
QL = Fr(1, 10)
L2 = lambda v: QL * sum(x * x for x in v)
L1 = lambda v: QL * sum(abs(x) for x in v)
WA, WB = (4, 0), (2, 2)
assert (L2(WA), L2(WB)) == (Fr(8, 5), Fr(4, 5)) and L1(WA) == L1(WB) == Fr(2, 5)
QOPTS = [(0, "<span class=\"rg-en\">L2</span> מעניש את $w_A$ יותר, ו-<span class=\"rg-en\">L1</span> מעניש את שניהם באותה מידה"),
         (1, "<span class=\"rg-en\">L2</span> מעניש את שניהם באותה מידה"),
         (2, "<span class=\"rg-en\">L1</span> מעניש את $w_A$ יותר"),
         (3, "שני העונשים זהים לשני הווקטורים, כי סכום המשקולות זהה")]
OK, L2SAME, L1MORE, ALLSAME = 0, 1, 2, 3

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נראה מה עושה רגולריזציה למשקולות, ומתי מודל פשוט מדי."),
    P("נראה איך נראה מודל שפשוט מדי בשביל הנתונים."),
    P("ובסוף נפתור שאלה בסגנון מבחן.", hold=0.4),
], stage='<div class="kicker">R05 · רגרסיה ורשתות נוירונים</div><h1>רגולריזציה ו-<span class="rg-en">underfitting</span></h1>'
         '<p>מה <span class="rg-en">λ</span> עושה למשקולות, ואיך נראה מודל פשוט מדי.</p>',
   layout="title")

two = Scene("two", "שתי בעיות הפוכות", [
    P("שתי בעיות הפוכות. התאמת יתר: מודל מסובך מדי, ששינן את נתוני האימון ונכשל על דוגמאות חדשות.", do=show("t-o"),
      say=f"שתי בעיות הפוכות. התאמת יתר, {SAY['overfitting']}: מודל מסובך מדי, ששינן את נתוני האימון, ונכשל על דוגמאות חדשות."),
    P("התאמת חסר: מודל פשוט מדי, שנכשל **כבר על נתוני האימון**.", do=show("t-u"),
      say=f"התאמת חסר, {SAY['underfitting']}: מודל פשוט מדי, שנכשל כבר על נתוני האימון."),
    P("לכן הכלים הפוכים: מודל מסובך מדי צריך פישוט, ומודל פשוט מדי צריך יותר יכולת.", do=show("t-k"), hold=0.3),
], stage=w.stack(
    w.card("t-o", "trap", "<b>התאמת יתר</b> (<span class=\"rg-en\">overfitting</span>): מצוין על האימון, גרוע על חדשות"),
    w.card("t-u", "trap", "<b>התאמת חסר</b> (<span class=\"rg-en\">underfitting</span>): גרוע כבר על האימון"), gap=30),
   panel=w.card("t-k", "rule", "🔑 יתר ← לפשט · חסר ← להוסיף יכולת"))

reg = Scene("reg", "האיבר הנוסף", [
    P("בלי מגבלה, מודל יכול לנפח משקולות כדי להתאים בדיוק לכל נקודת אימון."),
    P("ב**רגולריזציה** מוסיפים לפונקציית המחיר איבר שני, שגדל כשהמשקולות גדלות.", do=show("g-eq")),
    P("בגרסה L2 העונש הוא למדא כפול סכום ריבועי המשקולות. למדא הוא **קבוע הרגולריזציה**.", do=show("g-l2"),
      say="בגרסת אֶל שתיים, העונש הוא למדא כפול סכום ריבועי המשקולות. למדא הוא קבוע הרגולריזציה."),
    P("דוגמה: MSE של 1.2, משקולות 4 ומינוס 2, ולמדא 0.05.", do=show("g-ex"),
      say=f"דוגמה: {SAY['MSE']} של אחת נקודה שתיים, משקולות ארבע ומינוס שתיים, ולמדא אפס נקודה אפס חמש."),
    P("העונש: 0.05 כפול 16 ועוד 4, כלומר 1. המחיר הכולל: 2.2.", do=show("g-c"),
      say="העונש: אפס נקודה אפס חמש, כפול שש עשרה ועוד ארבע, כלומר אחת. המחיר הכולל: שתיים נקודה שתיים."),
    P("בגרסה L1 לוקחים ערכים מוחלטים: 0.05 כפול 6, כלומר 0.3.", do=show("g-l1"),
      say="בגרסת אֶל אחת לוקחים ערכים מוחלטים: אפס נקודה אפס חמש, כפול שש, כלומר אפס נקודה שלוש.", hold=0.3),
    P("ושימו לב: למדא הוא לא אלפא. אלפא קובע את גודל הצעד, ולמדא קובע כמה מענישים משקולות.", do=show("g-t")),
], stage=w.stack(
    w.eq("g-eq", "$J_{reg}(w) = \\text{MSE} + \\lambda \\sum_j w_j^2$", size=52, cls="box"),
    w.eq("g-ex", "$\\text{MSE} = 1.2, \\quad w = (4, {-2}), \\quad \\lambda = 0.05$", size=40),
    w.eq("g-c", "$J_{reg} = 1.2 + 0.05 \\cdot (16 + 4) = 2.2$", size=42),
    w.eq("g-l1", "$\\text{L1}: \\ 0.05 \\cdot (4 + 2) = 0.3$", size=38), gap=28),
   panel=w.card("g-l2", "ex", "<span class=\"rg-en\">L2</span>: $\\lambda \\sum w_j^2$ &nbsp;·&nbsp; <span class=\"rg-en\">L1</span>: $\\lambda \\sum |w_j|$")
         + w.card("g-t", "trap", "⚠️ $\\alpha$ = גודל הצעד · $\\lambda$ = כמה מענישים"))

bar = lambda v, k=150: f'<span class="rg-bar" style="width:{max(4, round(float(v) * k))}px"></span>'
SWT = w.table("sw", ["$\\lambda$", "$w_1$", "$w_2$", "MSE (אימון)"],
              [[str(lam), f"{bar(a)} {a}", f"{bar(b)} {b}", m] for lam, a, b, m in SW],
              hidden={(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}, size=32)
sweep = Scene("sweep", "מה למדא עושה", [
    P("נאמן אותו מודל, עם שני מאפיינים ועל אותן שש דוגמאות, ונגדיל את למדא בכל פעם.", do=show("sw-m0")),
    P("למדא אפס: בלי האיבר הנוסף. המשקולות בערך 1.2 ו-0.9, והשגיאה על האימון כמעט אפס.",
      say="למדא אפס: בלי האיבר הנוסף. המשקולות בערך אחת נקודה שתיים, ואפס נקודה תשע, והשגיאה על האימון כמעט אפס."),
    P("למדא 1: המשקולות קטנות יותר, והשגיאה עלתה מעט.", do=hide("sw-m0") + show("sw-1-1", "sw-1-2", "sw-1-3", "sw-m1"),
      say="למדא אחת: המשקולות קטנות יותר, והשגיאה עלתה מעט."),
    P("למדא 10: המשקולות ירדו מתחת ל-0.4, והשגיאה קפצה ל-5.2.", do=hide("sw-m1") + show("sw-2-1", "sw-2-2", "sw-2-3", "sw-m2"),
      say="למדא עשר: המשקולות ירדו מתחת לאפס נקודה ארבע, והשגיאה קפצה לחמש נקודה שתיים."),
    P("למדא 100: המשקולות כמעט אפס, והשגיאה על האימון כמעט כמו של מודל שתמיד חוזה את הממוצע.",
      do=hide("sw-m2") + show("sw-3-1", "sw-3-2", "sw-3-3", "sw-m3", "s-u"),
      say="למדא מאה: המשקולות כמעט אפס, והשגיאה על האימון כמעט כמו של מודל שתמיד חוזה את הממוצע.", hold=0.3),
    P("ככה נראית המשיכה של למדא: בכל שורה המשקולות קטנות, והשגיאה על האימון גדלה.", do=show("s-k")),
], stage=SWT,
   panel=w.card("s-u", "ex", "$\\lambda = 100$: שגיאה $10.95$ · חיזוי הממוצע: $12.14$")
         + w.card("s-k", "rule", "🔑 $\\lambda$ עולה ← המשקולות יורדות ← שגיאת האימון עולה"))

UP = w.fit_plot("uf", list(UX), list(UY), (4, 0), xr=(-0.5, 6.5), yr=(0, 10), xticks=[0, 1, 2, 3, 4, 5, 6],
                yticks=[0, 2, 4, 6, 8, 10], w=760, h=520, labels=False)
under = Scene("under", "מודל פשוט מדי", [
    P("ואיך נראה מודל פשוט מדי? הנה שבע נקודות שיוצרות קשת."),
    P("זה הקו הישר **הטוב ביותר** עבורן: $\\hat{y} = 4$. ה-MSE שלו על נתוני האימון עצמם: 12.",
      do=show("uf-ln", *[f"uf-r{i}" for i in range(7)], "v-1"),
      say=f"זה הקו הישר הטוב ביותר עבורן: וַואי כובע שווה ארבע. ה{SAY['MSE']} שלו על נתוני האימון עצמם: שתים עשרה."),
    P("אף קו ישר לא יעשה טוב יותר: קו פשוט לא מסוגל לבטא קשת. זו התאמת חסר, כבר באימון.", do=show("v-2"),
      say=f"אף קו ישר לא יעשה טוב יותר: קו פשוט לא מסוגל לבטא קשת. זו התאמת חסר, {SAY['underfitting']}, כבר באימון."),
    P("מה שחסר כאן הוא יכולת: המודל צריך לבטא יותר, לא פחות.", do=show("v-k"), hold=0.3),
], stage=UP.svg(),
   panel=w.card("v-1", "ex", "הקו הטוב ביותר: $\\hat{y} = 4$, $\\text{MSE} = 12$")
         + w.card("v-2", "trap", "⚠️ שגיאה גבוהה כבר על נתוני האימון")
         + w.card("v-k", "rule", "🔑 פשוט מדי ← חסרה יכולת"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: שני וקטורי משקולות, $w_A = (4, 0)$ ו-$w_B = (2, 2)$, ולמדא 0.1. איך מעניש כל סוג רגולריזציה?",
      say="שאלה בסגנון מבחן: שני וקטורי משקולות: ארבע ואפס, ושתיים ושתיים. ולמדא אפס נקודה אחת. איך מעניש כל סוג רגולריזציה?"),
    P("L2, סכום ריבועים: ל-A, 0.1 כפול 16, כלומר 1.6. ל-B, 0.1 כפול 8, כלומר 0.8.", do=show("x-1"),
      say="אֶל שתיים, סכום ריבועים: לוקטור הראשון, אפס נקודה אחת כפול שש עשרה, כלומר אחת נקודה שש. לשני, אפס נקודה אחת כפול שמונה, "
          "כלומר אפס נקודה שמונה."),
    P("L1, סכום ערכים מוחלטים: ל-A, 0.1 כפול 4. ל-B, 0.1 כפול 4. שניהם 0.4.", do=show("x-2"),
      say="אֶל אחת, סכום ערכים מוחלטים: לראשון, אפס נקודה אחת כפול ארבע. לשני, אפס נקודה אחת כפול ארבע. שניהם אפס נקודה ארבע."),
    P("כלומר L2 מעניש יותר את המשקולת הבודדת הגדולה, ו-L1 לא מבחין ביניהם. זו התשובה.", do=show(f"q-ok-{OK}"),
      say="כלומר אֶל שתיים מעניש יותר את המשקולת הבודדת הגדולה, ואֶל אחת לא מבחין ביניהם. זו התשובה.", hold=0.3),
    P("״L2 שווה לשניהם״ ו״L1 מעניש את A יותר״ מחליפות בין שני הסוגים.",
      do=show(f"q-x-{L2SAME}", f"q-n-{L2SAME}", f"q-x-{L1MORE}", f"q-n-{L1MORE}"),
      say="האפשרות שאֶל שתיים שווה לשניהם, והאפשרות שאֶל אחת מעניש את הראשון יותר, מחליפות בין שני הסוגים."),
    P("והאפשרות ששני העונשים זהים נכונה רק ל-L1: בריבוע, 16 גדול מ-4 ועוד 4.", do=show(f"q-x-{ALLSAME}", f"q-n-{ALLSAME}"),
      say="והאפשרות ששני העונשים זהים נכונה רק לאֶל אחת: בריבוע, שש עשרה גדול מארבע ועוד ארבע.", hold=0.5),
], stage=w.qpanel("בסגנון מבחן", "שני וקטורי משקולות: $w_A = (4, 0)$, $w_B = (2, 2)$, וקבוע רגולריזציה $\\lambda = 0.1$. "
                  "איזו טענה נכונה?", QOPTS, notes={L2SAME: "זה L1", L1MORE: "זה L2", ALLSAME: "רק ב-L1"}),
   panel=w.eq("x-1", "$\\text{L2}: \\ 1.6 \\ \\text{vs} \\ 0.8$", size=42, cls="box")
         + w.eq("x-2", "$\\text{L1}: \\ 0.4 \\ \\text{vs} \\ 0.4$", size=42, cls="box"))

recap = Scene("recap", "סיכום", [
    P("לסיכום: ברגולריזציה מוסיפים למחיר איבר: למדא כפול סכום ריבועי המשקולות.", do=show("r-1")),
    P("ככל שלמדא גדל, המשקולות קטנות והשגיאה על האימון גדלה. ולמדא הוא לא אלפא.", do=show("r-2")),
    P("ומודל פשוט מדי נכשל כבר על נתוני האימון: חסרה לו יכולת. נתראה ברמה הבאה.", do=show("r-3"), hold=0.6),
], stage=w.eq("r-eq", "$J_{reg}(w) = \\text{MSE} + \\lambda \\sum_j w_j^2$", size=54, hidden=False, cls="box"),
   panel=w.card("r-1", "rule", "🔑 <span class=\"rg-en\">L2</span>: $\\lambda \\sum w_j^2$, <span class=\"rg-en\">L1</span>: $\\lambda \\sum |w_j|$")
         + w.card("r-2", "rule", "🔑 $\\lambda$ עולה ← משקולות יורדות")
         + w.card("r-3", "rule", "🔑 פשוט מדי ← שגיאה גבוהה כבר באימון"))

VIDEO = Video("R05", "reg", "רגולריזציה ו-underfitting", [title, two, reg, sweep, under, exam, recap],
              bank={})
w.check_say(VIDEO)
