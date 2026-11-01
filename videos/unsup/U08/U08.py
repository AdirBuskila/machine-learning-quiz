# -*- coding: utf-8 -*-
"""U08 · מבחן מסכם (boss) — video script (build: tools/video/vid.py U08).

Source: courses/unsup/U08.js (checklist lesson + goal). A walkthrough of three questions from three of the course's
topics, none used by any guided quiz (vid.py refuses quiz ids):
  original   clustering, "בסגנון מבחן": single-linkage merges of 2, 4, 9, 10, 17 and a cut at height 3 (3 clusters).
             23S-A-Q8 was dropped: refuting its R2/RMSE/f1 distractors lets a student eliminate U03-Q5's options.
  PRAC-N-Q10 text: what IDF stands for (the word's discriminating power)
  PRAC-N-Q6  image: the normalized histogram of [10 20; 30 40] with 4 bins ([0.25]*4, and which meaning is right)
Avoided on purpose: every U08 quiz item (the Chebyshev triple, the four-point run, 24S-A-Q14, the two-cluster WSSE, the
true-statement question with WSSE/hierarchical/purity/tf-idf, the two-document TF-IDF, PRAC-X-Q2, the blue cell
histogram, the upscaled histogram, 24S-C-Q12) and their twins (SAMP-1-Q7, PRAC-X-Q28, PRAC-N-Q4, 24S-A-Q20). No metric
families are listed, purity is never named, nothing is said about when hierarchical clustering fits, and the histogram
question is refuted without the image-size argument of U08-Q7.
Every number is computed and asserted here.
"""
import math
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- q1: original, בסגנון מבחן (dendrogram cut)
HV = [2, 4, 9, 10, 17]
cl, ids, MERGES, nxt = [[v] for v in HV], list(range(5)), [], 5
while len(cl) > 1:
    i, j = min(((i, j) for i in range(len(cl)) for j in range(i + 1, len(cl))),
               key=lambda ij: min(abs(a - b) for a in cl[ij[0]] for b in cl[ij[1]]))
    MERGES.append((ids[i], ids[j], min(abs(a - b) for a in cl[i] for b in cl[j])))
    cl[i], ids[i] = cl[i] + cl[j], nxt
    nxt += 1
    del cl[j], ids[j]
assert [m[2] for m in MERGES] == [1, 2, 5, 7]
nclusters = lambda h: len(HV) - sum(1 for m in MERGES if m[2] < h)
assert nclusters(3) == 3 and nclusters(6) == 2 and nclusters(1.5) == 4 and nclusters(0.5) == 5
OPT1 = [(0, "3"), (1, "2"), (2, "4"), (3, "5")]

# ---------------------------------------------------------------- PRAC-N-Q10
Q2 = BANK["PRAC-N-Q10"]
assert Q2["correctIndex"] == 1 and "כושר ההבחנה" in Q2["options"][1] and "Stop Word" in Q2["options"][0]
assert "השכיחה ביותר" in Q2["options"][2] and "אורך המסמך הממוצע" in Q2["options"][3]
OPT2 = [(0, "ההסתברות המותנית למצוא את המילה, בהינתן שהיא stop word"),
        (1, "מדד לכושר ההבחנה של המילה: מפחית מחשיבותן של מילים שבמסמכים רבים"),
        (2, "התדירות היחסית של המילה במסמך, מול המילה השכיחה ביותר בו"),
        (3, "אורך המסמך הממוצע חלקי מספר ההופעות במסמך הנוכחי")]
N = 64
IDFS = {df: math.log2(N / df) for df in (2, 8, 64)}
assert IDFS == {2: 5.0, 8: 3.0, 64: 0.0}
assert all(IDFS[a] > IDFS[b] for a, b in ((2,8), (8,64)))       # more documents -> lower weight

# ---------------------------------------------------------------- PRAC-N-Q6
Q3 = BANK["PRAC-N-Q6"]
IMG = np.array([[10, 20], [30, 40]])
BINS = [(0,10), (10,20), (20,30), (30,40)]                     # (a, b]
assert "[10 20] [30 40]" in Q3["question"] and "(0,10]" in Q3["question"] and "(30,40]" in Q3["question"]
COUNTS = [int(((IMG > a) & (IMG <= b)).sum()) for a, b in BINS]
NORM = [c / IMG.size for c in COUNTS]
assert COUNTS == [1, 1, 1, 1] and NORM == [0.25] * 4
assert (IMG.flatten() / 100).tolist() == [0.1, 0.2, 0.3, 0.4]     # option 1's numbers: pixel values / 100
assert Q3["correctIndex"] == 3 and Q3["options"][0].startswith("[1, 1, 1, 1]")
assert Q3["options"][1].startswith("[0.1, 0.2, 0.3, 0.4]") and "Cells" in Q3["options"][2]
assert Q3["options"][2].startswith("[0.25") and Q3["options"][3].startswith("[0.25") and "לתא נפרד" in Q3["options"][3]
OPT3 = [(0, W.ltr("[1, 1, 1, 1]") + ": ספירה אבסולוטית בכל תא, שמאפשרת ל-KNN להשוות תמונות בגדלים שונים"),
        (1, W.ltr("[0.1, 0.2, 0.3, 0.4]") + ": הסתברות מותנית של Naive Bayes, בהינתן שהתמונה כהה"),
        (2, W.ltr("[0.25, 0.25, 0.25, 0.25]") + ": המיקום המרחבי האחיד של הפיקסלים על רשת ה-Cells"),
        (3, W.ltr("[0.25, 0.25, 0.25, 0.25]") + ": התפלגות אחידה: כל פיקסל בתא נפרד לפי עוצמתו")]

# ---------------------------------------------------------------- widgets
DG = W.dendro("dg", [str(v) for v in HV], MERGES, order=[0, 1, 2, 3, 4], ymax=8, w=700, h=430, cut=3)
IT = W.table("it", ["df", "N ⁄ df", "IDF"], [[str(df), num(N / df), num(v)] for df, v in IDFS.items()],
             hide={(r, c) for r in range(3) for c in (1,2)}, hl={(2,2)})
HG = W.pixgrid("hg", IMG.tolist(), cell=96, font=30)
HB = W.bars("hb", NORM, ["(0,10]", "(10,20]", "(20,30]", "(30,40]"], 0.4, w=600, h=320, hidden=True,
            fmt=lambda v: num(v))

QSM = lambda html: '<div class="u-qsm">' + html + "</div>"

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("זה המבחן המסכם של הקורס. נפתור שלוש שאלות, משלושה נושאים: אשכול, טקסט ותמונה."),
    P("בכל שאלה נפסול כל אפשרות שגויה, עם הסיבה המדויקת."),
    P("ובכל אפשרות עם נימוק, בודקים גם את המספר וגם את הנימוק.", hold=0.4),
], stage='<div class="kicker">U08 · אשכול, טקסט ותמונה</div><h1>מבחן מסכם</h1>'
         '<p>לפתור בתנאי מבחן שאלות מכל חמשת הנושאים: אשכול, טקסט, תמונה, אנסמבל ועיבוד מקדים.</p>', layout="title")

q1 = Scene("q1", "שאלה 1 · אשכול", [
    P("שאלה ראשונה, בסגנון מבחן: חמש נקודות על ישר, 2, 4, 9, 10 ו-17. אשכול היררכי, והמרחק בין אשכולות הוא בין הנקודות הקרובות.",
      say="שאלה ראשונה, בסגנון מבחן: חמש נקודות על ישר, שתיים, ארבע, תשע, עשר ושבע עשרה. אשכול היררכי, והמרחק בין אשכולות הוא בין הנקודות הקרובות."),
    P("כמה אשכולות יש אם חותכים את העץ בגובה 3?", say="כמה אשכולות יש אם חותכים את העץ בגובה שלוש?"),
    P("בונים את העץ: קודם 9 ו-10, במרחק 1. אחר כך 2 ו-4, במרחק 2.",
      say="בונים את העץ: קודם תשע ועשר, במרחק אחת. אחר כך שתיים וארבע, במרחק שתיים.", do=show("dg-m0", "dg-m1")),
    P("אחר כך שני הזוגות, מ-4 עד 9: מרחק 5. ולבסוף 17, במרחק 7 מ-10.",
      say="אחר כך שני הזוגות, מארבע עד תשע: מרחק חמש. ולבסוף שבע עשרה, במרחק שבע מעשר.", do=show("dg-m2", "dg-m3")),
    P("החיתוך בגובה 3 משאיר רק את שני האיחודים שמתחתיו. חמש נקודות פחות שני איחודים: שלושה אשכולות.",
      say="החיתוך בגובה שלוש משאיר רק את שני האיחודים שמתחתיו. חמש נקודות פחות שני איחודים: שלושה אשכולות.",
      do=show("dg-cut", "qa-ok-0")),
    P("2 היה נכון לחיתוך מעל 5. ו-4 יוצא מי שספר רק את האיחוד הראשון.",
      say="שתיים היה נכון לחיתוך מעל חמש. וארבע יוצא מי שספר רק את האיחוד הראשון.",
      do=show("qa-x-1", "qa-n-1", "qa-x-2", "qa-n-2")),
    P("ו-5 מתעלם מהעץ כולו: כאילו אף נקודה לא אוחדה.", say="וחמש מתעלם מהעץ כולו: כאילו אף נקודה לא אוחדה.",
      do=show("qa-x-3", "qa-n-3"), hold=0.4),
], stage=DG,
   panel='<div class="qstem"><span class="qid">בסגנון מבחן</span>הנקודות 2, 4, 9, 10, 17. אשכול היררכי '
         '(מרחק בין הנקודות הקרובות). כמה אשכולות בחיתוך בגובה 3?</div>'
         + st.options("qa", OPT1, notes={1: "חיתוך מעל 5", 2: "איחוד אחד", 3: "בלי איחודים"}))

q2 = Scene("q2", "שאלה 2 · טקסט", [
    P("שאלה שנייה: מה מייצג רכיב ה-IDF במשקל TF-IDF?"),
    P("נחזור לנוסחה: לוג בבסיס 2 של מספר המסמכים חלקי מספר המסמכים שבהם המילה מופיעה.",
      say="נחזור לנוסחה: לוג בבסיס שתיים של מספר המסמכים חלקי מספר המסמכים שבהם המילה מופיעה.", do=show("f2")),
    P("בקורפוס של 64 מסמכים: מילה בשני מסמכים מקבלת 5, בשמונה מסמכים 3, ובכל ה-64 מקבלת 0.",
      say="בקורפוס של שישים וארבעה מסמכים: מילה בשני מסמכים מקבלת חמש, בשמונה מסמכים שלוש, ובכל השישים וארבעה מקבלת אפס.",
      do=show(*W.all_cells("it", 3, 3, 0, 1), "it-hl-2-2")),
    P("ככל שהמילה מופיעה ביותר מסמכים, היא עוזרת פחות להבדיל ביניהם, והמשקל שלה יורד. זו התשובה.",
      do=show("qb-ok-1")),
    P("הסתברות מותנית בהינתן stop word? אין בנוסחה שום הסתברות, ואין בה שום תנאי על stop words.",
      say="הסתברות מותנית בהינתן סְטוֹפּ ווֹרְד? אין בנוסחה שום הסתברות, ואין בה שום תנאי על סְטוֹפּ ווֹרְדְס.",
      do=show("qb-x-0", "qb-n-0")),
    P("תדירות בתוך מסמך אחד היא כבר צד ה-TF. ה-IDF מסתכל על כל הקורפוס.",
      say="תדירות בתוך מסמך אחד היא כבר הצד של הטִי אֶף. האַיי דִּי אֶף מסתכל על כל הקורפוס.", do=show("qb-x-2", "qb-n-2")),
    P("ואורך מסמך ממוצע לא מופיע בנוסחה בכלל. רק שני מספרים: כמה מסמכים, ובכמה מהם המילה.",
      do=show("qb-x-3", "qb-n-3"), hold=0.4),
], stage=W.stack(W.big("f2", "$IDF = \\log_2 \\dfrac{N}{df}$", size=50), IT, gap=40),
   panel=f'<div class="qstem"><span class="qid">{Q2["id"]}</span>מה מייצג רכיב ה-IDF עבור מילת מפתח?</div>'
         + QSM(st.options("qb", OPT2, notes={0: "אין הסתברות", 2: "זה TF", 3: "אין אורך"})))

q3 = Scene("q3", "שאלה 3 · תמונה", [
    P("שאלה שלישית: תמונה של שני פיקסלים על שני פיקסלים: 10, 20, 30 ו-40.", do=show("hg")),
    P("היסטוגרמה מנורמלת עם ארבעה תאים: עד 10, עד 20, עד 30 ועד 40, כל תא כולל את הקצה העליון שלו."),
    P("10 נופל בתא הראשון, 20 בשני, 30 בשלישי ו-40 ברביעי. פיקסל אחד בכל תא.", do=show("hc")),
    P("מנורמלת: כל ספירה חלקי ארבעה פיקסלים. רבע, רבע, רבע ורבע.", do=show("hb-b0", "hb-b1", "hb-b2", "hb-b3")),
    P("שתי אפשרויות נותנות בדיוק את המספרים האלה. מה שמכריע ביניהן הוא הנימוק.", do=show("x-t")),
    P("הנכונה: התפלגות אחידה, כי כל פיקסל נפל לתא אחר לפי העוצמה שלו.", do=show("qc-ok-3")),
    P("והשנייה מדברת על מיקום מרחבי ברשת של תאים. אבל היסטוגרמה סופרת עוצמות, ולא יודעת איפה הפיקסל ישב.",
      do=show("qc-x-2", "qc-n-2")),
    P("1, 1, 1, 1 היא הספירה לפני הנרמול. השאלה ביקשה מנורמלת.",
      say="אחת, אחת, אחת, אחת היא הספירה לפני הנרמול. השאלה ביקשה מנורמלת.", do=show("qc-x-0", "qc-n-0")),
    P("ו-0.1, 0.2, 0.3, 0.4 הם ערכי הפיקסלים חלקי 100, לא ספירה. ונאיב בייס לא קשור כאן בכלל.",
      say="ואפס נקודה אחת, אפס נקודה שתיים, אפס נקודה שלוש, אפס נקודה ארבע הם ערכי הפיקסלים חלקי מאה, לא ספירה. ונאיב בייס לא קשור כאן בכלל.",
      do=show("qc-x-1", "qc-n-1"), hold=0.4),
], stage=W.stack(W.row(HG, HB, gap=50), W.note("hc", "ספירה: " + W.ltr("[1, 1, 1, 1]") + " · חלקי 4", size=34),
                 st.card("x-t", "trap", "⚠️ אותם מספרים, נימוק אחר: בודקים גם את המשמעות"), gap=30),
   panel=f'<div class="qstem"><span class="qid">{Q3["id"]}</span>תמונה 2×2: [10 20] [30 40]. היסטוגרמה '
         'מנורמלת עם 4 תאים. מה הווקטור ומה משמעותו?</div>'
         + QSM(st.options("qc", OPT3, notes={0: "לא מנורמל", 1: "ערכים ⁄ 100", 2: "אין מיקום"})))

recap = Scene("recap", "סיכום", [
    P("לסיכום המבחן: בעץ היררכי, חיתוך בגובה משאיר רק את האיחודים שמתחתיו.", do=show("r-1")),
    P("IDF מסתכל על כל הקורפוס: ככל שהמילה במסמכים רבים יותר, המשקל שלה נמוך יותר.",
      say="אַיי דִּי אֶף מסתכל על כל הקורפוס: ככל שהמילה במסמכים רבים יותר, המשקל שלה נמוך יותר.", do=show("r-2")),
    P("היסטוגרמה מנורמלת היא ספירה חלקי מספר הפיקסלים.", do=show("r-3")),
    P("ובכל אפשרות עם נימוק: גם המספר וגם הנימוק חייבים להיות נכונים. בהצלחה במבחן!", do=show("r-4"), hold=0.6),
], stage=W.stack(st.card("r-1", "rule", "🔑 אשכולות = נקודות פחות איחודים מתחת לחיתוך"),
                 st.card("r-2", "rule", "🔑 IDF = כושר הבחנה: יותר מסמכים ← פחות משקל"),
                 st.card("r-3", "rule", "🔑 מנורמלת = ספירה ⁄ פיקסלים"),
                 st.card("r-4", "trap", "⚠️ מספר נכון + נימוק שגוי = תשובה שגויה"), gap=24),
   layout="full")

VIDEO = Video("U08", "unsup", "מבחן מסכם", [title, q1, q2, q3, recap],
              bank={"PRAC-N-Q10": 1, "PRAC-N-Q6": 3})
