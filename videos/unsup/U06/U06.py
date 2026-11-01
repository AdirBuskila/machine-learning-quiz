# -*- coding: utf-8 -*-
"""U06 · קרנלים והיסטוגרמות — video script (build: tools/video/vid.py U06).

Source: courses/unsup/U06.js (lesson + goal). Exam part: an ORIGINAL "בסגנון מבחן" question (bank={}): a kernel with
a single 1 in its top-left corner shifts the image one pixel down and to the right (zero padding). 23B-A-Q3 was dropped:
refuting its distractors would answer U06-Q3; the remaining image bank items are quiz twins or used by U08.
Avoided on purpose: the quiz items of U06 (24B-C-Q15, 25B-A-Q4, 26B-A-Q7 and the originals: the 90/0/30 neighbourhood,
the vertical 1/3 kernel, the 1/9 kernel, the "+" detector, the 64x48 shift, the 2x3 normalized histogram, 16 cells x 8
bins, the Red threshold), their twins (24B-A-Q5, 23B-B-Q3, 22B-A-Q17, 22B-C-Q17, 26S-A-Q12, CLAUDE-26-Q12, PRAC-N-Q3,
PRAC-N-Q5) and the boss items (PRAC-X-Q28 / U08-Q6, U08-Q7). No horizontal, vertical or 3x3 mean kernel is shown;
the video does not say "blur / sharpen", "illegal kernel", "cannot know without the image", "spatial location" or
"cells x bins".
New data: a 5x5 image (multiples of 30) under a diagonal 1/3 kernel (feature map 100 170 130 / 30 160 160 / 120 60 120);
a "\\" detector (1 on the diagonal, -1 elsewhere) scoring 3, 2 and -3 on three binary patches; a 3x4 image with
histogram [6,3,0,3], normalized [0.5,0.25,0,0.25], and its mirror image with the same histogram but other cell
histograms. Every value is computed and asserted here (numpy).
"""
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- convolution (scene "conv")
IMG = np.array([[30, 60, 90, 150, 240], [0, 120, 210, 180, 90], [60, 30, 150, 240, 120],
                [90, 180, 60, 210, 30], [150, 240, 120, 90, 0]])
KD = np.eye(3) / 3


def conv_valid(M, K):
    return np.array([[(M[r - 1:r + 2, c - 1:c + 2] * K).sum() for c in range(1, M.shape[1] - 1)]
                     for r in range(1, M.shape[0] - 1)])


FM = conv_valid(IMG, KD)
assert np.allclose(FM, [[100, 170, 130], [30, 160, 160], [120, 60, 120]])
assert IMG[0, 0] + IMG[1, 1] + IMG[2, 2] == 300 and IMG[1, 1] + IMG[2, 2] + IMG[3, 3] == 480
assert np.allclose(np.rot90(KD, 2), KD)                    # symmetric: convolution and correlation agree here
assert KD.sum() == 1 and (KD >= 0).all()

# ---------------------------------------------------------------- shape detector (scene "shape")
KS = np.where(np.eye(3) == 1, 1, -1)
PA = np.eye(3, dtype=int)                                   # exactly the diagonal
PB = np.eye(3, dtype=int)
PB[0, 2] = 1                                                # the diagonal plus one stray white pixel
PC = np.ones((3,3), dtype=int)                             # all white
score = lambda p: int((p * KS).sum())
assert (score(PA), score(PB), score(PC)) == (3, 2, -3)
best = max(score(np.array(b).reshape(3,3)) for b in np.ndindex(*(2,) * 9))
assert best == 3 == int((KS == 1).sum())                    # the best score = the number of +1 cells

# ---------------------------------------------------------------- histograms (scene "hist")
H1 = np.array([[20, 30, 200, 230], [40, 100, 210, 10], [50, 90, 120, 60]])
H2 = H1[:, ::-1]                                            # the mirror image
BINS = [(0,63), (64,127), (128,191), (192,255)]
hist = lambda M: [int(((M >= a) & (M <= b)).sum()) for a, b in BINS]
assert hist(H1) == [6, 3, 0, 3] == hist(H2) and sum(hist(H1)) == H1.size == 12
NORM = [c / H1.size for c in hist(H1)]
assert NORM == [0.5, 0.25, 0, 0.25] and sum(NORM) == 1
CELL1 = hist(H1[:, :2]) + hist(H1[:, 2:])
CELL2 = hist(H2[:, :2]) + hist(H2[:, 2:])
assert CELL1 == [4, 2, 0, 0, 2, 1, 0, 3] and CELL2 == [2, 1, 0, 3, 4, 2, 0, 0] and len(CELL1) == 2 * 4
assert CELL1 != CELL2

# ---------------------------------------------------------------- exam: original, בסגנון מבחן (shift kernel)
KI = np.zeros((3, 3))
KI[0, 0] = 1
pad = np.pad(IMG, 1)
FI = np.array([[(pad[r:r + 3, c:c + 3] * KI).sum() for c in range(5)] for r in range(5)])
SHIFT = np.zeros_like(IMG)
SHIFT[1:, 1:] = IMG[:-1, :-1]
assert (FI == SHIFT).all()                                  # out[r][c] = in[r-1][c-1]: the picture moves down-right
assert (FI[0] == 0).all() and (FI[:, 0] == 0).all()        # the first row and column come from the zero padding
UPLEFT = np.zeros_like(IMG)
UPLEFT[:-1, :-1] = IMG[1:, 1:]
assert not (FI == UPLEFT).all() and not (FI == IMG).all()
AVG2 = np.array([[(pad[r:r + 3, c:c + 3] * np.array([[.5, 0, 0], [0, .5, 0], [0, 0, 0]])).sum() for c in range(5)]
                 for r in range(5)])
assert not np.allclose(FI, AVG2)
OPTS = [(0, "התמונה זזה פיקסל אחד למטה וימינה; השורה והעמודה הראשונות מתמלאות באפסים"),
        (1, "התמונה זזה פיקסל אחד למעלה ושמאלה"),
        (2, "כל פיקסל הופך לממוצע שלו ושל השכן מלמעלה משמאל"),
        (3, "ה-feature map זהה לתמונה, כי יש בקרנל רק 1 אחד")]

# ---------------------------------------------------------------- widgets
fr = lambda v: "1/3" if abs(v - 1 / 3) < 1e-9 else num(v)
IMGG = W.pixgrid("im", IMG.tolist(), cell=78, font=24, wins=[(1,1), (2,2)], hls=[(0,0), (1,1), (2,2), (3,3)])
KERG = W.pixgrid("kd", KD.tolist(), cell=72, shade=False, fmt=fr, font=24)
OUTG = W.pixgrid("fm", FM.tolist(), cell=78, font=24, hidden_vals=True)
CONV = W.row(W.stack(W.note("cl1", "תמונה", hidden=False, size=28), IMGG, gap=8),
             W.stack(W.note("cl2", "קרנל", hidden=False, size=28), KERG, gap=8),
             W.stack(W.note("cl3", "feature map", hidden=False, size=28), OUTG, gap=8), gap=36)

KSG = W.pixgrid("ks", KS.tolist(), cell=66, shade=False, font=26)
patch = lambda pfx, M: W.pixgrid(pfx, M.tolist(), cell=56, binary=True, font=22)
SHAPE = W.stack(W.row(W.stack(W.note("sl0", "קרנל", hidden=False, size=28), KSG, gap=8), gap=10),
                W.row(*[W.stack(patch(f"sp{i}", M), W.note(f"ss{i}", f"ציון: <b>{W.ltr(str(s).replace('-', '−'))}</b>", size=30), gap=8)
                        for i, (M, s) in enumerate([(PA, 3), (PB, 2), (PC, -3)])], gap=60), gap=34)

HG1 = W.pixgrid("h1", H1.tolist(), cell=54, font=20)
HG2 = W.pixgrid("h2", H2.tolist(), cell=54, font=20)
HB = W.bars("hb", hist(H1), ["0–63", "64–127", "128–191", "192–255"], 7, w=480, h=320, hidden=True,
            fmt=lambda v: str(int(v)))

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד לקרוא קרנל ולחשב פיקסל אחרי קונבולוציה."),
    P("ואחר כך נספור פיקסלים: היסטוגרמה רגילה, מנורמלת, והיסטוגרמה לפי תאים."),
    P("ובסוף שאלה ממבחן על קרנל אחד מיוחד.", hold=0.4),
], stage='<div class="kicker">U06 · אשכול, טקסט ותמונה</div><h1>קרנלים והיסטוגרמות</h1>'
         '<p>לקרוא קרנל ולדעת מה הוא עושה, לחשב פיקסל אחרי קונבולוציה, ולחשב היסטוגרמה רגילה, מנורמלת ו-Cell Histogram.</p>',
   layout="title")

conv = Scene("conv", "קונבולוציה", [
    P("תמונה בגווני אפור היא מטריצה: כל פיקסל הוא מספר בין 0, שחור, ל-255, לבן.", do=show("c-c0")),
    P("בתמונת RGB יש שלוש מטריצות כאלה, אחת לכל ערוץ צבע.",
      say="בתמונה צבעונית, אַר גִ'י בִּי, יש שלוש מטריצות כאלה, אחת לכל ערוץ צבע."),
    P("קרנל הוא מטריצה קטנה. כאן יש שליש על האלכסון, ואפס בכל השאר."),
    P("מניחים אותו על פיקסל ושכניו, כופלים כל תא בתא שמתחתיו, ומחברים.", do=show("im-w-1-1")),
    P("רק האלכסון נכנס: 30, 120 ו-150. סכום 300, כפול שליש, כלומר 100.",
      say="רק האלכסון נכנס: שלושים, מאה עשרים ומאה חמישים. סכום שלוש מאות, כפול שליש, כלומר מאה.",
      do=show("im-hl-0-0", "im-hl-1-1", "im-hl-2-2", "fm-0-0-v")),
    P("מזיזים את הקרנל צעד. במרכז התמונה: 120, 150 ו-210. סכום 480, ולכן 160.",
      say="מזיזים את הקרנל צעד. במרכז התמונה: מאה עשרים, מאה חמישים ומאתיים עשר. סכום ארבע מאות שמונים, ולכן מאה שישים.",
      do=hide("im-w-1-1", "im-hl-0-0") + show("im-w-2-2", "im-hl-3-3", "fm-1-1-v")),
    P("ככה עוברים על כל התמונה, והתוצאה נקראת feature map.",
      do=hide("im-w-2-2", "im-hl-1-1", "im-hl-2-2", "im-hl-3-3") + show(*[f"fm-{r}-{c}-v" for r in range(3) for c in range(3)])),
    P("ומה הקרנל הזה עושה? לא צריך תמונה בשביל לדעת. סופרים איפה הערכים שאינם 0.", do=show("c-c1")),
    P("שלושה תאים, על האלכסון, כל אחד שליש: ממוצע של הפיקסל עם השכן מלמעלה משמאל, ועם השכן מלמטה מימין.",
      do=show("c-c2")),
    P("וכל הערכים חיוביים וסכומם 1: כל פיקסל חדש הוא ממוצע של שלושה פיקסלים מהמקור.",
      say="וכל הערכים חיוביים וסכומם אחת: כל פיקסל חדש הוא ממוצע של שלושה פיקסלים מהמקור.", do=show("c-c3"), hold=0.4),
], stage=CONV,
   panel=st.card("c-c0", "ex", "גווני אפור: 0 שחור · 255 לבן")
         + st.card("c-c1", "rule", "🔑 מה הקרנל עושה? סופרים את התאים שאינם 0")
         + st.card("c-c2", "ex", "שליש על האלכסון ← ממוצע לאורך האלכסון")
         + st.card("c-c3", "trap", "🔑 חיובי וסכום 1 ← ממוצע"))

shape = Scene("shape", "גלאי צורה", [
    P("ועכשיו קרנל עם ערכים שליליים: 1 על האלכסון, ומינוס 1 בכל השאר.",
      say="ועכשיו קרנל עם ערכים שליליים: אחת על האלכסון, ומינוס אחת בכל השאר."),
    P("נפעיל אותו על תמונה בשחור ולבן, שבה כל פיקסל הוא 0 או 1.",
      say="נפעיל אותו על תמונה בשחור ולבן, שבה כל פיקסל הוא אפס או אחת."),
    P("כשהפיקסלים הלבנים יושבים בדיוק על האלכסון: שלוש פעמים 1, והציון 3.",
      say="כשהפיקסלים הלבנים יושבים בדיוק על האלכסון: שלוש פעמים אחת, והציון שלוש.", do=show("ss0")),
    P("פיקסל לבן אחד מחוץ לאלכסון נופל על מינוס 1, ומוריד את הציון ל-2.",
      say="פיקסל לבן אחד מחוץ לאלכסון נופל על מינוס אחת, ומוריד את הציון לשתיים.", do=show("ss1")),
    P("ותמונה לבנה לגמרי? שלוש פעמים פלוס, שש פעמים מינוס: מינוס 3.",
      say="ותמונה לבנה לגמרי? שלוש פעמים פלוס, שש פעמים מינוס: מינוס שלוש.", do=show("ss2")),
    P("אז הקרנל מחפש קו אלכסוני. הציון הכי גבוה מתקבל כשיש לבן בדיוק בתאים של ה-1, ושחור בכל השאר.",
      do=show("s-c1")),
    P("ולכן הציון המקסימלי הוא מספר התאים עם 1. כאן שלושה.", do=show("s-c2"), hold=0.4),
], stage=SHAPE,
   panel=st.card("s-c1", "ex", "שליליים בקרנל ← הוא מחפש צורה")
         + st.card("s-c2", "rule", "🔑 ציון מקסימלי = מספר התאים עם 1"))

histo = Scene("hist", "היסטוגרמות", [
    P("היסטוגרמה סופרת כמה פיקסלים נופלים בכל טווח עוצמה. כל טווח נקרא תא, bin.",
      say="היסטוגרמה סופרת כמה פיקסלים נופלים בכל טווח עוצמה. כל טווח נקרא תא, בִּין."),
    P("בתמונה הזו 12 פיקסלים, וארבעה bins ברוחב 64. שישה כהים, שלושה בינוניים, אף אחד בתא השלישי, ושלושה בהירים.",
      say="בתמונה הזו שנים עשר פיקסלים, וארבעה בִּינְס ברוחב שישים וארבע. שישה כהים, שלושה בינוניים, אף אחד בתא השלישי, ושלושה בהירים.",
      do=show("hb-b0", "hb-b1", "hb-b2", "hb-b3")),
    P("הסכום תמיד שווה למספר הפיקסלים בתמונה, כאן 12.", do=show("h-c1")),
    P("בהיסטוגרמה מנורמלת מחלקים כל ספירה במספר הפיקסלים: חצי, רבע, 0 ורבע. הסכום 1.",
      say="בהיסטוגרמה מנורמלת מחלקים כל ספירה במספר הפיקסלים: חצי, רבע, אפס ורבע. הסכום אחת.", do=show("h-n")),
    P("ככה אפשר להשוות תמונות בגדלים שונים, בלי שהגודל ישפיע.", do=show("h-c2")),
    P("עכשיו נשקף את התמונה, כמו במראה. אותם שנים עשר ערכים, ולכן ההיסטוגרמה זהה לגמרי.", do=show("h2w", "h-cl")),
    P("נחלק כל תמונה לחצי שמאלי ולחצי ימני, ונספור כל חצי לבד: Cell Histogram.", do=show("h-v1")),
    P("שני חצאים, כל אחד עם ארבעה bins: וקטור באורך 8. ובתמונה המשוקפת שני החצאים מתחלפים.",
      say="שני חצאים, כל אחד עם ארבעה בִּינְס: וקטור באורך שמונה. ובתמונה המשוקפת שני החצאים מתחלפים.",
      do=show("h-v2", "h-c3"), hold=0.4),
], stage=W.stack(W.row(W.stack(HG1, gap=4), W.stack(HG2.replace('class="u-gridwrap"', 'class="u-gridwrap"'), gap=4)
                       .replace('<div class="u-stack', '<div id="h2w" class="h u-stack', 1), HB, gap=36),
                 W.note("h-n", "מנורמלת: " + W.ltr("[0.5, 0.25, 0, 0.25]"), size=32),
                 W.note("h-v1", "לפי חצאים, מקור: " + W.ltr("[4, 2, 0, 0 | 2, 1, 0, 3]"), size=30),
                 W.note("h-v2", "לפי חצאים, משוקף: " + W.ltr("[2, 1, 0, 3 | 4, 2, 0, 0]"), size=30), gap=22),
   panel=st.card("h-c1", "rule", "🔑 סכום רגילה = מספר הפיקסלים")
         + st.card("h-c2", "ex", "מנורמלת: ספירה ⁄ פיקסלים, סכום 1")
         + st.card("h-cl", "trap", "⚠️ תמונה ומראה שלה: אותה היסטוגרמה")
         + st.card("h-c3", "rule", "🔑 כאן: 2 חצאים, 4 bins בכל אחד ← 8 מספרים"))

exam = Scene("exam", "שאלה בסגנון מבחן", [
    P("שאלה בסגנון מבחן: קרנל עם 1 בפינה השמאלית העליונה, ו-0 בכל השאר. מה הוא עושה לתמונה, עם ריפוד של אפסים?",
      say="שאלה בסגנון מבחן: קרנל עם אחת בפינה השמאלית העליונה, ואפס בכל השאר. מה הוא עושה לתמונה, עם ריפוד של אפסים?"),
    P("סופרים תאים שאינם 0: רק אחד, הפינה. אז כל פיקסל חדש לוקח את ערך השכן שמעליו ומשמאלו.",
      say="סופרים תאים שאינם אפס: רק אחד, הפינה. אז כל פיקסל חדש לוקח את ערך השכן שמעליו ומשמאלו.", do=show("x-k")),
    P("הפעלנו אותו על התמונה שלנו: ה-30 שהיה בפינה עבר שורה אחת למטה ועמודה אחת ימינה. וכך כל פיקסל.",
      say="הפעלנו אותו על התמונה שלנו: השלושים שהיה בפינה עבר שורה אחת למטה ועמודה אחת ימינה. וכך כל פיקסל.",
      do=show("x-i", "qq-ok-0")),
    P("השורה והעמודה הראשונות מקבלות את האפסים של הריפוד."),
    P("למעלה ושמאלה? זה הכיוון של השכן שממנו לוקחים, לא הכיוון שאליו התמונה זזה.", do=show("qq-x-1", "qq-n-1")),
    P("ממוצע? יש בקרנל רק ערך אחד, והוא 1. אין מה למצע.", say="ממוצע? יש בקרנל רק ערך אחד, והוא אחת. אין מה למצע.",
      do=show("qq-x-2", "qq-n-2")),
    P("וזהות? בזהות ה-1 יושב במרכז. כאן הוא בפינה, והמיקום שלו הוא כל ההבדל.",
      say="וזהות? בזהות האחת יושבת במרכז. כאן היא בפינה, והמיקום שלה הוא כל ההבדל.", do=show("qq-x-3", "qq-n-3", "x-t"),
      hold=0.4),
], stage=W.stack(W.row(W.stack(W.note("xl", "קרנל", hidden=False, size=28),
                               W.pixgrid("x-kg", KI.tolist(), cell=70, shade=False, font=26), gap=8),
                       W.stack(W.note("xl2", "feature map", hidden=False, size=28),
                               W.pixgrid("x-fi", FI.tolist(), cell=62, font=20), gap=8), gap=50),
                 W.note("x-k", "תא אחד שאינו 0: הפינה השמאלית העליונה", size=32),
                 W.note("x-i", "כל ערך זז שורה למטה ועמודה ימינה", size=32),
                 st.card("x-t", "trap", "⚠️ לא רק כמה תאים, אלא גם איפה הם"), gap=22),
   panel='<div class="qstem"><span class="qid">בסגנון מבחן</span>הקרנל 1 0 0 / 0 0 0 / 0 0 0, עם ריפוד אפסים. '
         'מה יהיה ה-feature map?</div>'
         + '<div class="u-qsm">' + st.options("qq", OPTS, notes={1: "הכיוון הפוך", 2: "ערך אחד", 3: "לא במרכז"}) + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: קונבולוציה כופלת את הקרנל בסביבה של כל פיקסל, ומחברת.", do=show("r-1")),
    P("מה הקרנל עושה? סופרים את התאים שאינם 0, ובודקים איפה הם. חיובי וסכום 1 זה ממוצע.",
      say="מה הקרנל עושה? סופרים את התאים שאינם אפס, ובודקים איפה הם. חיובי וסכום אחת זה ממוצע.", do=show("r-2")),
    P("ערכים שליליים מחפשים צורה, והציון המקסימלי הוא מספר התאים עם 1.",
      say="ערכים שליליים מחפשים צורה, והציון המקסימלי הוא מספר התאים עם אחת.", do=show("r-3")),
    P("היסטוגרמה רגילה מסתכמת במספר הפיקסלים, ומנורמלת ב-1. וחלוקה לחצאים הבדילה בין תמונה למראה שלה. נתראה ברמה הבאה.",
      say="היסטוגרמה רגילה מסתכמת במספר הפיקסלים, ומנורמלת באחת. וחלוקה לחצאים הבדילה בין תמונה למראה שלה. נתראה ברמה הבאה.",
      do=show("r-4"), hold=0.6),
], stage=CONV.replace('id="im', 'id="jm').replace('id="kd', 'id="jd').replace('id="fm', 'id="jf')
          .replace('id="cl', 'id="jl').replace('<span id="jf', '<span data-x="1" id="jf').replace(' class="h">', '>'),
   panel=st.card("r-1", "rule", "🔑 כפל תא בתא, וחיבור")
         + st.card("r-2", "rule", "🔑 סופרים תאים שאינם 0")
         + st.card("r-3", "rule", "🔑 שליליים: צורה; מקסימום = מספר ה-1")
         + st.card("r-4", "trap", "⚠️ רגילה Σ = פיקסלים · מנורמלת Σ = 1"))

VIDEO = Video("U06", "unsup", "קרנלים והיסטוגרמות", [title, conv, shape, histo, exam, recap], bank={})
