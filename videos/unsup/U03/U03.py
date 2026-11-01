# -*- coding: utf-8 -*-
"""U03 · WSSE, מרפק ואשכול היררכי — video script (build: tools/video/vid.py U03).

Source: courses/unsup/U03.js (lesson + goal). Exam anchor: 25B-A-Q15 (PCA: using only the second component keeps 30%),
which no guided quiz uses.
Avoided on purpose (wording too): "WSSE always decreases with K" (U08-Q4) is stated only as a property of the best
partition per K; "hierarchical when the number of clusters is unknown / expensive on big data" (U03-Q6, U08-Q4, SAMP-2-Q23)
is not said, the dendrogram is simply cut at two heights. Also avoided: the quiz items of U03 (PRAC-X-Q32, 26B-A-Q8 and the originals: centre 5 with 3,4,5,8; centre (2,3)
with (1,3),(2,5),(4,4); the WSSE table 500/260/120/…; the metric question with purity; the "number of clusters unknown"
question; PCA weights 0.5/0.25/0.15/0.1) and the boss items (U08-Q3's two-cluster WSSE, U08-Q4). The video does not
list clustering metrics (that would answer U03-Q5) and keeps its elbow at K = 4, not 3.
New data: a WSSE of 16 around the centre (3,2); twelve points in four groups whose best WSSE per K is 238, 127, 55, 16,
13, 10 (elbow at 4); a single-linkage dendrogram of 1, 3, 8, 9, 20. All computed and asserted here.
"""
import itertools
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- WSSE of one cluster (scene "wsse")
CEN = np.array([3, 2])
PTS = np.array([(2,1), (3,5), (4,0)])
assert PTS.mean(0).tolist() == CEN.tolist()               # the centre is the cluster's mean, as after an update step
SQ = ((PTS - CEN) ** 2).sum(1)
assert SQ.tolist() == [2, 9, 5] and SQ.sum() == 16
ROOT = np.sqrt(SQ)
assert num(ROOT.sum()) == "6.65"                          # summing distances (with roots) gives another number

# ---------------------------------------------------------------- elbow (scene "elbow")
GC = [(2,2), (3,8), (9,3), (8,9)]
OFF = [(0,1), (-1, -1), (1,0)]
X = np.array([(a + u, b + v) for a, b in GC for u, v in OFF], float)


def lloyd(C):
    C = np.array(C, float)
    for _ in range(100):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[a == k].mean(0) if (a == k).any() else C[k] for k in range(len(C))])
        if np.allclose(new, C):
            break
        C = new
    return float(((X - C[a]) ** 2).sum())


WSSE = [min(lloyd(X[list(c)]) for c in itertools.combinations(range(len(X)), K)) for K in range(1,7)]
assert [round(w, 6) for w in WSSE] == [238, 127, 55, 16, 13, 10]
DROPS = [WSSE[i] - WSSE[i + 1] for i in range(5)]
assert [round(x) for x in DROPS] == [111, 72, 39, 3, 3]
assert all(WSSE[i] >= WSSE[i + 1] for i in range(5))
ELBOW = 1 + max(range(1,5), key=lambda i: DROPS[i - 1] / DROPS[i])           # the drop collapses most after K
assert ELBOW == 4
assert lloyd(X) == 0                                       # K = 12: every point is its own centre
try:                                                       # cross-check with scikit-learn when it is installed
    from sklearn.cluster import KMeans
    for K in range(1,7):
        assert abs(KMeans(K, n_init=50, random_state=0).fit(X).inertia_ - WSSE[K - 1]) < 1e-6
except ImportError:
    pass

# ---------------------------------------------------------------- hierarchical (scene "hier"), single linkage
V = [1, 3, 8, 9, 20]
clusters = [[v] for v in V]
ids = list(range(len(V)))
MERGES = []
nxt = len(V)
while len(clusters) > 1:
    best = min(((i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))),
               key=lambda ij: min(abs(a - b) for a in clusters[ij[0]] for b in clusters[ij[1]]))
    i, j = best
    dist = min(abs(a - b) for a in clusters[i] for b in clusters[j])
    MERGES.append((ids[i], ids[j], dist))
    clusters[i] = clusters[i] + clusters[j]
    ids[i] = nxt
    nxt += 1
    del clusters[j], ids[j]
assert MERGES == [(2, 3, 1), (0, 1, 2), (6, 5, 5), (7, 4, 11)]
try:
    from scipy.cluster.hierarchy import linkage
    Z = linkage(np.array(V, float)[:, None], "single")
    assert [(int(a), int(b), float(c)) for a, b, c, _ in Z] == [(2, 3, 1.0), (0, 1, 2.0), (5, 6, 5.0), (4, 7, 11.0)]
except ImportError:
    pass
# the same merge order under complete / average linkage, so the picture does not depend on the linkage choice
for agg in (max, lambda ds: sum(ds) / len(ds)):
    cl = [[v] for v in V]
    order = []
    while len(cl) > 1:
        i, j = min(((i, j) for i in range(len(cl)) for j in range(i + 1, len(cl))),
                   key=lambda ij: agg([abs(a - b) for a in cl[ij[0]] for b in cl[ij[1]]]))
        order.append(sorted(cl[i] + cl[j]))
        cl[i] = cl[i] + cl[j]
        del cl[j]
    assert order == [[8, 9], [1, 3], [1, 3, 8, 9], [1, 3, 8, 9, 20]]
# cut at height 4 -> {1,3}, {8,9}, {20}; cut at 8 -> {1,3,8,9}, {20}
assert [m[2] for m in MERGES if m[2] < 4] == [1, 2] and [m[2] for m in MERGES if m[2] < 8] == [1, 2, 5]

# ---------------------------------------------------------------- exam 25B-A-Q15 (PCA)
Q = BANK["25B-A-Q15"]
W1, W2 = 0.6, 0.3
assert "0.6" in Q["question"] and "0.3" in Q["question"] and "השני" in Q["question"]
assert Q["correctIndex"] == 0 and "30%" in Q["options"][0] and "90%" in Q["options"][1]
assert "72%" in Q["options"][2] and "12%" in Q["options"][3]
assert round(W2 * 100) == 30 and round((W1 + W2) * 100) == 90
assert round((1 - W1 - W2) * 100) == 10                    # everything after the second component together: 10%
OPTS = [(0, "30%: חלק משמעותי נשמר, אך הרוב הולך לאיבוד"),
        (1, "90%: הרוב הגדול של המידע נשמר"),
        (2, "72%: עדיף לבחור ברכיב החמישי או השישי"),
        (3, "12%: הרכיבים אינם קומבינציה לינארית של המקוריים")]


# ---------------------------------------------------------------- widgets
def wsse_plot():
    p = W.Plot("wp", (0,6), (0,6), w=560, h=560)
    for i, (x, y) in enumerate(PTS):
        p.seg(tuple(CEN), (x, y), id_=f"wp-s{i}", k=1, dash=True)
    for i, (x, y) in enumerate(PTS):
        p.pt(x, y, k=0, label=f"({x},{y})")
    p.cen(*CEN, 1, hidden=False, label="(3,2)")
    return p.svg()


WT = W.table("wt", ["נקודה", "הפרשים", "ריבוע המרחק"],
             [["(2,1)", "1, 1", "1 + 1 = 2"], ["(3,5)", "0, 3", "0 + 9 = 9"], ["(4,0)", "1, 2", "1 + 4 = 5"],
              ["", "", "<b>WSSE = 16</b>"]],
             hide={(r, c) for r in range(4) for c in (1,2)})

ELB = W.curve("ec", list(range(1,7)), WSSE, (0.5,6.5), (0,260), w=760, h=470, ystep=50, hl=3)


def dendro_fig(pfx):
    return W.dendro(pfx, [str(v) for v in V], MERGES, order=[0, 1, 2, 3, 4], ymax=12, w=760, h=470, cut=4)


PCA = W.bars("pb", [0.6, 0.3, 0.1], ["PC1", "PC2", "שאר הרכיבים"], 0.7, w=600, h=380, hidden=False)

# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("בסרטון הזה נלמד לשפוט חלוקה לאשכולות: כמה היא צפופה, וכמה אשכולות כדאי לבקש."),
    P("נחשב WSSE, נמצא מרפק בגרף, ונבנה עץ של אשכול היררכי."),
    P("ובסוף שאלה ממבחן על PCA.", hold=0.4),
], stage='<div class="kicker">U03 · אשכול, טקסט ותמונה</div><h1>WSSE, מרפק ואשכול היררכי</h1>'
         '<p>לחשב WSSE, לבחור K בשיטת המרפק, ולדעת מתי אשכול היררכי ומה שומר PCA.</p>', layout="title")

wsse = Scene("wsse", "WSSE", [
    P("איך מודדים אם אשכול צפוף? WSSE: סכום ריבועי המרחקים של כל נקודה מהמרכז של האשכול שלה."),
    P("כאן אשכול עם מרכז ב-(3,2), ושלוש נקודות.", say="כאן אשכול עם מרכז בשלוש, שתיים, ושלוש נקודות.",
      do=show("wp-s0", "wp-s1", "wp-s2")),
    P("הנקודה (2,1): ההפרשים 1 ו-1. ריבוע המרחק הוא 1 ועוד 1, כלומר 2.",
      say="הנקודה שתיים, אחת: ההפרשים אחת ואחת. ריבוע המרחק הוא אחת ועוד אחת, כלומר שתיים.",
      do=show("wt-0-1", "wt-0-2")),
    P("שימו לב: לא מוציאים שורש. ריבוע המרחק הוא פשוט סכום ריבועי ההפרשים.", do=show("w-c1")),
    P("הנקודה (3,5) תורמת 9, והנקודה (4,0) תורמת 1 ועוד 4, כלומר 5.",
      say="הנקודה שלוש, חמש תורמת תשע, והנקודה ארבע, אפס תורמת אחת ועוד ארבע, כלומר חמש.",
      do=show("wt-1-1", "wt-1-2", "wt-2-1", "wt-2-2")),
    P("הסכום: 2 ועוד 9 ועוד 5, כלומר WSSE של 16.", do=show("wt-3-2")),
    P("מי שמחבר את המרחקים עצמם, עם שורש, יקבל בערך 6.65. זה מספר אחר, וזה לא WSSE.", do=show("w-c2")),
    P("ואם יש כמה אשכולות, מחברים את התרומות של כולם. ככל שהסכום קטן יותר, הנקודות צמודות יותר למרכזים.",
      hold=0.4),
], stage=wsse_plot(),
   panel=WT + st.card("w-c1", "rule", "🔑 ריבוע המרחק: Σ הפרש², בלי שורש")
         + st.card("w-c2", "trap", "⚠️ סכום מרחקים עם שורש ≈ 6.65 ≠ WSSE"))

elbow = Scene("elbow", "שיטת המרפק", [
    P("כמה אשכולות לבקש? מריצים K-Means שוב ושוב, עם $K$ אחר בכל פעם, ורושמים את ה-WSSE.",
      say="כמה אשכולות לבקש? מריצים קֵיי מינס שוב ושוב, עם קֵיי אחר בכל פעם, ורושמים את הדַּאבְּלְיוּ אֶס אֶס אִי."),
    P("על שתים עשרה נקודות יצא: עם אשכול אחד 238, עם שניים 127, ועם שלושה 55.", do=show("ec-p0", "ec-p1", "ec-p2")),
    P("עם ארבעה 16, ואז 13, ו-10.", do=show("ec-p3", "ec-p4", "ec-p5")),
    P("כאן כל ערך קטן מהקודם, וזה לא מקרי: בחלוקה הטובה ביותר לכל מספר מרכזים, עוד מרכז לא יכול להזיק לסכום."),
    P("ואם יש מרכז לכל נקודה, הסכום הוא 0. לכן לחפש את המספר הנמוך ביותר זה חסר טעם.", do=show("e-c1")),
    P("מסתכלים על הירידות: 111, אחר כך 72, ואז 39. כל אשכול נוסף עוד משפר הרבה.", do=show("e-d")),
    P("ואחרי ארבעה הירידה צונחת ל-3, ושוב 3. אשכול חמישי כמעט לא עוזר.", hold=0.3),
    P("זה המרפק: הנקודה שאחריה השיפור נעשה קטן. כאן $K$ שווה 4.",
      say="זה המרפק: הנקודה שאחריה השיפור נעשה קטן. כאן קֵיי שווה ארבע.", do=show("ec-hl", "e-c2"), hold=0.4),
], stage=W.stack(ELB, W.note("e-d", "ירידות: 111 · 72 · 39 · <b>3</b> · 3", size=34), gap=10),
   panel=st.card("e-c1", "trap", "⚠️ לבחור את הסכום הנמוך = מרכז לכל נקודה")
         + st.card("e-c2", "rule", "🔑 מרפק = אחריו השיפור קטן ← K = 4"))

hier = Scene("hier", "אשכול היררכי", [
    P("ועכשיו גישה אחרת לגמרי: אשכול היררכי, שבונה עץ של איחודים."),
    P("חמש נקודות על ישר: 1, 3, 8, 9 ו-20. בהתחלה כל נקודה היא אשכול לבד."),
    P("בכל צעד מאחדים את שני האשכולות הקרובים ביותר. כאן נמדוד לפי הנקודות הקרובות שבהם."),
    P("הכי קרובים: 8 ו-9, במרחק 1. מאחדים, ומציירים חיבור בגובה 1.", do=show("dg-m0")),
    P("אחר כך 1 ו-3, במרחק 2.", do=show("dg-m1")),
    P("אחר כך שני הזוגות: מ-3 עד 8 יש 5.", do=show("dg-m2")),
    P("ולבסוף 20 מצטרף, במרחק 11 מ-9.", do=show("dg-m3")),
    P("העץ הזה נקרא דנדרוגרמה. עכשיו חותכים אותו בגובה שרוצים.", do=show("h-c1")),
    P("חיתוך בגובה 4 נותן שלושה אשכולות: 1 ו-3, 8 ו-9, ו-20 לבד.", do=show("dg-cut")),
    P("וחיתוך בגובה 8 נותן רק שניים: 1, 3, 8 ו-9 ביחד, ו-20 לבד. אותו עץ, וכל גובה הוא חלוקה אחרת.",
      say="וחיתוך בגובה שמונה נותן רק שניים: אחת, שלוש, שמונה ותשע ביחד, ועשרים לבד. אותו עץ, וכל גובה הוא חלוקה אחרת.",
      do=show("h-c2", "h-c3"), hold=0.4),
], stage=dendro_fig("dg"),
   panel=st.card("h-c1", "ex", "כל צעד: מאחדים את שני האשכולות הקרובים")
         + st.card("h-c2", "ex", "גובה 4 ← 3 אשכולות · גובה 8 ← 2")
         + st.card("h-c3", "rule", "🔑 העץ אחד; גובה החיתוך קובע את החלוקה"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן על PCA. PCA מחליף את המאפיינים ברכיבים חדשים, ממוינים לפי כמה מהשונות כל אחד שומר."),
    P("הרכיב הראשון קיבל משקל 0.6, והשני 0.3. בוחרים להשתמש ברכיב **השני**. מה המשמעות?",
      say="הרכיב הראשון קיבל משקל אפס נקודה שש, והשני אפס נקודה שלוש. בוחרים להשתמש ברכיב השני. מה המשמעות?"),
    P("שומרים בדיוק את המשקל של הרכיבים שלוקחים. לקחנו רק את השני, אז 0.3, כלומר 30%.",
      say="שומרים בדיוק את המשקל של הרכיבים שלוקחים. לקחנו רק את השני, אז אפס נקודה שלוש, כלומר שלושים אחוז.",
      do=show("q-ok-0")),
    P("30% זה חלק לא מבוטל, אבל רוב השונות הלכה, וזה בדיוק מה שהאפשרות הזאת אומרת.",
      say="שלושים אחוז זה חלק לא מבוטל, אבל רוב השונות הלכה, וזה בדיוק מה שהאפשרות הזאת אומרת."),
    P("90% זה הראשון והשני ביחד. זו המלכודת: לא קראו שלוקחים רק את השני.",
      say="תשעים אחוז זה הראשון והשני ביחד. זו המלכודת: לא קראו שלוקחים רק את השני.",
      do=show("q-x-1", "q-n-1", "x-t")),
    P("72% לא יוצא מהנתונים. והרכיבים ממוינים: החמישי והשישי שומרים פחות, ביחד לכל היותר 10%.",
      say="שבעים ושתיים אחוז לא יוצא מהנתונים. והרכיבים ממוינים: החמישי והשישי שומרים פחות, ביחד לכל היותר עשרה אחוז.",
      do=show("q-x-2", "q-n-2", "pb-b2")),
    P("12% לא יוצא מהנתונים, והנימוק שגוי: כל רכיב של PCA הוא כן קומבינציה לינארית של המאפיינים המקוריים.",
      say="שנים עשר אחוז לא יוצא מהנתונים, והנימוק שגוי: כל רכיב של פִּי סִי אֵיי הוא כן קומבינציה לינארית של המאפיינים המקוריים.",
      do=show("q-x-3", "q-n-3"), hold=0.4),
], stage=W.stack(PCA, st.card("x-t", "trap", "⚠️ מחברים רק את הרכיבים שבאמת לוקחים"), gap=30),
   panel=f'<div class="qstem"><span class="qid">{Q["id"]}</span>PCA: לרכיב הראשון משקל 0.6, לשני 0.3. '
         'משתמשים ברכיב <b>השני</b>. מה המשמעות?</div>'
         + st.options("q", OPTS, notes={1: "0.6 + 0.3", 2: "ממוינים", 3: "כן לינאריים"}))

recap = Scene("recap", "סיכום", [
    P("לסיכום: WSSE הוא סכום ריבועי המרחקים מהמרכזים, בלי שורש.", do=show("r-1")),
    P("לא בוחרים את הסכום הקטן ביותר, אלא את המרפק: הנקודה שאחריה השיפור קטן.", do=show("r-2")),
    P("אשכול היררכי בונה עץ של איחודים, וכל חיתוך שלו הוא חלוקה.", do=show("r-3")),
    P("וב-PCA מחברים רק את המשקלים של הרכיבים שלוקחים. נתראה ברמה הבאה.",
      say="ובפִּי סִי אֵיי מחברים רק את המשקלים של הרכיבים שלוקחים. נתראה ברמה הבאה.", do=show("r-4"), hold=0.6),
], stage=dendro_fig("dh").replace('class="u-dg h"', 'class="u-dg"'),
   panel=st.card("r-1", "rule", "🔑 WSSE = Σ ריבועי מרחקים")
         + st.card("r-2", "rule", "🔑 מרפק: אחריו השיפור קטן")
         + st.card("r-3", "rule", "🔑 היררכי: עץ איחודים; חותכים בגובה")
         + st.card("r-4", "trap", "⚠️ PCA: רק הרכיבים שנבחרו"))

VIDEO = Video("U03", "unsup", "WSSE, מרפק ואשכול היררכי", [title, wsse, elbow, hier, exam, recap],
              bank={"25B-A-Q15": 0})
