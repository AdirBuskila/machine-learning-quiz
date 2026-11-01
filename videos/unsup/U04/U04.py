# -*- coding: utf-8 -*-
"""U04 · מטקסט לווקטור — video script (build: tools/video/vid.py U04).

Source: courses/unsup/U04.js (lesson + goal). Exam anchor: 26B-A-Q15 (why cosine similarity instead of Euclidean
distance between document vectors), which no guided quiz uses.
Avoided on purpose: the quiz items of U04 (PRAC-X-Q11, 22B-A-Q12 and the originals: "the cat sat on the mat",
"the cat saw the dog", the 3 x 7 matrix question and its "a row per document, a column per word" wording, the
cat/dog Hebrew pair, the stop-word list של/את/על, the מכונית/רכב pair and the "represent a word by its context"
wording, the three-document cosine-vs-Euclid ranking A=(2,1,0), B=(4,2,0), D=(3,0,1)), the twins of 22B-A-Q12
(22B-C-Q12, 23B-A-Q16, 23B-B-Q16) and the boss item U08-Q4. The lesson's dog/man example is not reused either.
New data: an 8-token sentence with a 7-word vocabulary; a 4-document corpus whose Bag of Words table is 4 x 6 (checked
against scikit-learn's CountVectorizer when it is installed); two orders of document 3 with the same row; co-occurrence
vectors where coffee and tea score cosine 0.8 although they never share a sentence; and ONE pair of documents,
(1,2) and (3,5), whose cosine (0.997) is computed step by step next to their Euclidean distance (3.61).
Every count and score is computed and asserted here.
"""
import numpy as np
from vid import P, Scene, Video, show, hide, BANK
import stage as st
import w_unsup as W
from w_unsup import num

# ---------------------------------------------------------------- tokens (scene "tok")
SENT = "הוא אכל פיצה כי הוא היה רעב מאוד"
TOKS = SENT.split()
VOC = list(dict.fromkeys(TOKS))
assert len(TOKS) == 8 and len(VOC) == 7 and TOKS.count("הוא") == 2
STOP = {"הוא", "כי", "היה"}
STRIKE = {i for i, t in enumerate(TOKS) if t in STOP}
assert STRIKE == {0, 3, 4, 5}
KEPT = [t for t in TOKS if t not in STOP]
assert KEPT == ["אכל", "פיצה", "רעב", "מאוד"]

# ---------------------------------------------------------------- Bag of Words (scene "bow")
DOCS = ["דני שותה קפה", "רון שותה תה", "קפה חם קפה", "תה חם"]
VOCAB = sorted({w for d in DOCS for w in d.split()})
assert VOCAB == ["דני", "חם", "קפה", "רון", "שותה", "תה"]
BOW = [[d.split().count(w) for w in VOCAB] for d in DOCS]
assert BOW == [[1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 0, 0], [0, 1, 0, 0, 0, 1]]
try:
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    M = cv.fit_transform(DOCS).toarray()
    assert list(cv.get_feature_names_out()) == VOCAB and M.tolist() == BOW and M.shape == (4,6)
except ImportError:
    pass
ALT = "קפה קפה חם"                                      # another order of document 3: the same row
assert [ALT.split().count(w) for w in VOCAB] == BOW[2] and ALT != DOCS[2]

# ---------------------------------------------------------------- meaning from context (scene "ctx")
CTX_SENTS = ["שתיתי כוס קפה חם", "שתיתי כוס תה חם", "הזמנתי קפה עם חלב", "הזמנתי תה עם לימון",
             "האוטובוס נסע לאט", "האוטובוס היה מלא"]
TARGETS = ["קפה", "תה", "האוטובוס"]
CSTOP = {"עם"}
CWORDS = list(dict.fromkeys(w for s in CTX_SENTS for w in s.split() if w not in TARGETS and w not in CSTOP))
assert CWORDS == ["שתיתי", "כוס", "חם", "הזמנתי", "חלב", "לימון", "נסע", "לאט", "היה", "מלא"]


def ctx(t):
    v = np.zeros(len(CWORDS))
    for s in CTX_SENTS:
        ws = s.split()
        if t in ws:
            for w in ws:
                if w in CWORDS:
                    v[CWORDS.index(w)] += 1
    return v


cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
CV = {t: ctx(t) for t in TARGETS}
assert not any("קפה" in s.split() and "תה" in s.split() for s in CTX_SENTS)      # never in the same sentence
assert num(cos(CV["קפה"], CV["תה"])) == "0.8" and cos(CV["קפה"], CV["האוטובוס"]) == 0
assert int(CV["קפה"] @ CV["תה"]) == 4 and int(CV["קפה"].sum()) == 5 == int(CV["תה"].sum())
SHARED = [w for w in CWORDS if CV["קפה"][CWORDS.index(w)] and CV["תה"][CWORDS.index(w)]]
assert SHARED == ["שתיתי", "כוס", "חם", "הזמנתי"]

# ---------------------------------------------------------------- cosine vs Euclid (scene "cos"): [כדור, שחקן, בחירות]
DS, DL, DT = np.array([1, 1, 0]), np.array([4, 3, 0]), np.array([0, 0, 2])
eu = lambda a, b: float(np.linalg.norm(a - b))
assert num(eu(DS, DL)) == "3.61" and num(eu(DS, DT)) == "2.45" and eu(DS, DT) < eu(DS, DL)
assert num(cos(DS, DL)) == "0.99" and cos(DS, DT) == 0
assert DS @ DL == 7 and num(np.linalg.norm(DS)) == "1.41" and np.linalg.norm(DL) == 5

# ---------------------------------------------------------------- exam 26B-A-Q15
Q = BANK["26B-A-Q15"]
assert Q["correctIndex"] == 0 and "אורך המסמך" in Q["options"][0] and "רציפים" in Q["options"][1]
assert "מילים עוקבות" in Q["options"][2] and "[0,1]" in Q["options"][3]
OPTS = [(0, "לנטרל את השפעת אורך המסמך: אוקלידי מציג פער בין מסמכים דומים רק בגלל אורך"),
        (1, "אוקלידי רגיש רק לערכים רציפים, ואי אפשר לחשב אותו על וקטורי תדירויות"),
        (2, "קוסינוס מחשב את המרחק הקצר ביותר בין מילים עוקבות בפסקה"),
        (3, "קוסינוס משנה לינארית את ערכי התכונות לטווח [0,1]")]

# ---------------------------------------------------------------- widgets
TOKCH = W.chips("tk", TOKS, hidden=True, strike=STRIKE)
BOWT = W.table("bw", [""] + VOCAB, [[f"מסמך {i + 1}"] + [str(v) for v in r] for i, r in enumerate(BOW)],
               hide={(r, c) for r in range(4) for c in range(1,7)}, cls="sm rtl", hl={(2,3)})
CTXT = W.table("cx", [""] + CWORDS[:6] + ["…"],
               [[t] + [num(v) for v in CV[t][:6]] + ["0" if t != "האוטובוס" else "4"] for t in TARGETS[:2]],
               cls="sm rtl", hide={(r, c) for r in range(2) for c in range(1,8)},
               hl={(r, c) for r in range(2) for c in (1, 2, 3, 4)})
assert CV["קפה"][6:].sum() == 0 == CV["תה"][6:].sum() and CV["האוטובוס"][6:].sum() == 4


DA, DB = np.array([1, 2]), np.array([3, 5])                  # [קפה, חלב]: a short and a longer coffee text
assert DA @ DB == 13 and num(np.linalg.norm(DA)) == "2.24" and num(np.linalg.norm(DB)) == "5.83"
assert num(cos(DA, DB), 3) == "0.997" and num(np.linalg.norm(DA - DB)) == "3.61"
assert DA.sum() == 3 and DB.sum() == 8


def cos_plot(pfx, scale=1.0):
    p = W.Plot(pfx, (0, 4), (0, 6), w=480, h=620, scale=scale)
    p.seg((0, 0), (1, 2), id_=f"{pfx}-a", k=0, hidden=False, label="a", lab_dx=-24, lab_dy=0, width=6)
    p.seg((0, 0), (3, 5), id_=f"{pfx}-b", k=1, hidden=False, label="b", lab_dx=26, lab_dy=10, width=6)
    p.seg((1, 2), (3, 5), id_=f"{pfx}-d", k=2, dash=True, label="3.61", lab_dx=-34, lab_dy=-4)
    p.txt(0.15, 5.6, "חלב", cls="u-plab", anchor="start")
    p.txt(3.9, 0.25, "קפה", cls="u-plab", anchor="end")
    return p.svg()


CF = W.stack(W.big("c-f", "$\\cos(a,b) = \\dfrac{a \\cdot b}{\\lVert a \\rVert \\, \\lVert b \\rVert}$", size=40),
             W.note("c-1", W.ltr("a · b = 1·3 + 2·5 = 13"), size=32),
             W.note("c-2", W.ltr("‖a‖ = √5 ≈ 2.24 &nbsp; ‖b‖ = √34 ≈ 5.83"), size=32),
             W.note("c-3", W.ltr("cos = 13 ⁄ (2.24 · 5.83) ≈ <b>0.997</b>"), size=34), gap=18)


# ---------------------------------------------------------------- scenes
title = Scene("title", "", [
    P("מודל למידה מקבל מספרים, לא מילים. בסרטון הזה נהפוך טקסט לווקטור."),
    P("נפרק משפט לטוקנים, נבנה Bag of Words, ונראה מה הייצוג הזה מאבד."),
    P("ובסוף שאלה ממבחן: למה משווים מסמכים בקוסינוס.", hold=0.4),
], stage='<div class="kicker">U04 · אשכול, טקסט ותמונה</div><h1>מטקסט לווקטור</h1>'
         '<p>להבין איך מסמך הופך לווקטור מספרים, מה הוא מאבד, ומה באמת תופס משמעות של מילה.</p>', layout="title")

tok = Scene("tok", "טוקנים", [
    P("כל מסמך, משפט או ביקורת או כתבה, מתחיל כרצף של תווים.", do=show("t-s")),
    P("השלב הראשון: Tokenization, פירוק הרצף ליחידות בדידות. כאן נפרק לפי רווחים.",
      do=show(*[f"tk-{i}" for i in range(8)])),
    P("יצאו שמונה טוקנים. כל מופע נספר: המילה הוא מופיעה פעמיים, אז היא שני טוקנים.", do=show("t-n1")),
    P("אבל באוצר המילים, רשימת המילים השונות, היא נכנסת פעם אחת. לכן יש שבע מילים שונות.",
      do=show("t-n2", "t-c1")),
    P("ומילים כמו הוא, כי והיה מופיעות כמעט בכל טקסט, ולא אומרות כלום על הנושא. הן נקראות stop words.",
      do=show(*[f"tk-x{i}" for i in sorted(STRIKE)], "t-c2")),
    P("בדרך כלל מסננים אותן. כאן נשארים ארבעה טוקנים, והם אלה שבאמת מספרים על המשפט.", do=show("t-n3"), hold=0.4),
], stage=W.stack(W.note("t-s", f"״{SENT}״", size=40), TOKCH,
                 W.note("t-n1", "8 טוקנים", size=34), W.note("t-n2", "7 מילים שונות באוצר", size=34),
                 W.note("t-n3", "בלי stop words: 4 טוקנים", size=34), gap=22),
   panel=st.card("t-c1", "rule", "🔑 טוקן = כל מופע · אוצר מילים = המילים השונות")
         + st.card("t-c2", "ex", "stop words: מילים נפוצות בלי מידע על הנושא"))

bow = Scene("bow", "Bag of Words", [
    P("עכשיו קורפוס, אוסף של ארבעה מסמכים קצרים.", do=show("b-docs")),
    P("נאסוף את כל המילים השונות בקורפוס: יש שש. כל אחת מקבלת עמודה בטבלה.", do=show("bw")),
    P("במסמך הראשון: דני, שותה, קפה. כותבים 1 בעמודות שלהן, ו-0 בשאר.", do=show(*W.all_cells("bw", 1, 7, 0, 1))),
    P("השני: רון, שותה, תה.", do=show(*W.all_cells("bw", 2, 7, 1, 1))),
    P("בשלישי המילה קפה מופיעה פעמיים, ולכן בעמודה שלה כתוב 2.",
      do=show(*W.all_cells("bw", 3, 7, 2, 1), "bw-hl-2-3")),
    P("והרביעי: תה חם. זה Bag of Words, ובספריות של פייתון הכלי שבונה אותו נקרא Count Vectorizer.",
      say="והרביעי: תה חם. זה בֶּג אוֹף ווֹרְדְס, ובספריות של פייתון הכלי שבונה אותו נקרא קַאוּנְט וֶקְטוֹרַייזֶר.",
      do=show(*W.all_cells("bw", 4, 7, 3, 1), "b-c1")),
    P("והמחיר? נסו לשחזר את המסמך השלישי מהשורה שלו בלבד.", do=show("b-p")),
    P("אולי זה היה קפה חם קפה, ואולי קפה קפה חם. לשני הסדרים יש בדיוק אותה שורה.", do=show("b-p2")),
    P("השורה יודעת כמה פעמים כל מילה הופיעה, ולא באיזה מקום. הסדר נשפך מהשקית.", do=show("b-c2"), hold=0.4),
], stage=W.stack(W.note("b-docs", "<span class='m'>1:</span> דני שותה קפה &nbsp; · &nbsp; <span class='m'>2:</span> רון שותה תה"
                        " &nbsp; · &nbsp; <span class='m'>3:</span> קפה חם קפה &nbsp; · &nbsp; <span class='m'>4:</span> תה חם", size=32),
                 BOWT,
                 W.note("b-p", "שורה 3 ← איזה משפט?", size=34),
                 W.note("b-p2", "״קפה חם קפה״ &nbsp; או &nbsp; ״קפה קפה חם״? <b>אותה שורה</b>", size=34), gap=30),
   panel=st.card("b-c1", "rule", "🔑 Count Vectorizer: מקורפוס לטבלת ספירות")
         + st.card("b-c2", "trap", "⚠️ Bag of Words סופר, ולא זוכר סדר"))

ctxs = Scene("ctx", "משמעות מהקשר", [
    P("וקטור כזה מייצג את המילה עצמה. אבל המילים קפה ותה הן שתי עמודות נפרדות, בלי שום קשר ביניהן."),
    P("ננסה משהו אחר: נבדוק עם אילו מילים כל מילה נפגשת במשפטים, ונספור את הפגישות.", do=show("cx-s")),
    P("קפה מופיעה ליד שתיתי, כוס, חם, הזמנתי וחלב.", do=show(*W.all_cells("cx", 1, 8, 0, 1))),
    P("ותה מופיעה ליד שתיתי, כוס, חם, הזמנתי ולימון.", do=show(*W.all_cells("cx", 2, 8, 1, 1))),
    P("ארבע מתוך חמש השכנות משותפות. הקוסינוס בין שני הווקטורים יוצא 0.8, גבוה.",
      say="ארבע מתוך חמש השכנות משותפות. הקוסינוס בין שני הווקטורים יוצא אפס נקודה שמונה, גבוה.",
      do=show(*[f"cx-hl-{r}-{c}" for r in range(2) for c in (1, 2, 3, 4)], "x-c1")),
    P("ושימו לב: קפה ותה אף פעם לא הופיעו באותו משפט. הדמיון בא רק מהסביבה שלהן.", do=show("x-c2")),
    P("ומילה כמו האוטובוס, שגרה בסביבה אחרת לגמרי, מקבלת קוסינוס 0 מול קפה.", do=show("x-c3"), hold=0.4),
], stage=W.stack(W.note("cx-s", "שתיתי כוס קפה חם · שתיתי כוס תה חם · הזמנתי קפה עם חלב<br>"
                        "הזמנתי תה עם לימון · האוטובוס נסע לאט · האוטובוס היה מלא", size=30),
                 CTXT, gap=40),
   panel=st.card("x-c1", "ex", "קפה ⟷ תה: קוסינוס 0.8")
         + st.card("x-c2", "rule", "🔑 אותן שכנויות ← וקטורים דומים")
         + st.card("x-c3", "ex", "קפה ⟷ האוטובוס: קוסינוס 0"))

cosine = Scene("cos", "קוסינוס", [
    P("ואיך משווים שני מסמכים? ניקח רק שתי עמודות, קפה וחלב, כדי שאפשר יהיה לצייר."),
    P("מסמך a קצר: קפה פעם אחת, חלב פעמיים. מסמך b ארוך יותר: קפה שלוש פעמים, חלב חמש.",
      say="מסמך אֵיי קצר: קפה פעם אחת, חלב פעמיים. מסמך בִּי ארוך יותר: קפה שלוש פעמים, חלב חמש."),
    P("כל מסמך הוא חץ. המרחק האוקלידי הוא הקו בין קצות החצים: שורש 13, בערך 3.61.",
      say="כל מסמך הוא חץ. המרחק האוקלידי הוא הקו בין קצות החצים: שורש שלוש עשרה, בערך שלוש נקודה שישים ואחת.",
      do=show("cp-d")),
    P("קוסינוס שואל שאלה אחרת: מה הזווית בין החצים.", do=show("c-f")),
    P("המכפלה: 1 כפול 3, ועוד 2 כפול 5, כלומר 13.",
      say="המכפלה: אחת כפול שלוש, ועוד שתיים כפול חמש, כלומר שלוש עשרה.", do=show("c-1")),
    P("האורכים: שורש 5 ושורש 34, בערך 2.24 ו-5.83.",
      say="האורכים: שורש חמש ושורש שלושים וארבע, בערך שתיים נקודה עשרים וארבע וחמש נקודה שמונים ושלוש.",
      do=show("c-2")),
    P("ולכן 13 חלקי המכפלה של האורכים: בערך 0.997. החצים כמעט מקבילים.",
      say="ולכן שלוש עשרה חלקי, בסוגריים, שתיים נקודה עשרים וארבע כפול חמש נקודה שמונים ושלוש: בערך אפס נקודה תשע תשע שבע. החצים כמעט מקבילים.",
      do=show("c-3")),
    P("כלומר יחס המילים כמעט זהה. רוב הפער של 3.61 נובע מכך ש-b פשוט ארוך יותר.",
      say="כלומר יחס המילים כמעט זהה. רוב הפער של שלוש נקודה שישים ואחת נובע מכך שבִּי פשוט ארוך יותר.",
      do=show("c-c1"), hold=0.4),
], stage=W.row(cos_plot("cp"), CF, gap=40),
   panel=st.card("c-c1", "trap", "⚠️ קוסינוס רואה כיוון; אוקלידי רואה גם אורך"))

exam = Scene("exam", "שאלה ממבחן", [
    P("שאלה ממבחן: למה משווים וקטורים של מסמכים בקוסינוס, ולא במרחק אוקלידי?"),
    P("ראינו את זה: יחס המילים כמעט זהה, ובכל זאת אוקלידי מודד פער, בגלל האורך. זו האפשרות הנכונה.",
      do=show("q-ok-0")),
    P("האם אוקלידי עובד רק על ערכים רציפים? לא. לפני רגע חישבנו אותו על ספירות, ויצא 3.61.",
      do=show("q-x-1", "q-n-1")),
    P("מילים עוקבות בפסקה? קוסינוס מקבל רק שני וקטורים של ספירות. אין בהם בכלל סדר של מילים.",
      do=show("q-x-2", "q-n-2")),
    P("ושינוי התכונות לטווח 0 עד 1? קוסינוס לא משנה אף ערך בווקטור. הוא מחזיר מספר אחד, הזווית.",
      do=show("q-x-3", "q-n-3")),
    P("המלכודת היא השם: מרחק נשמע כמו המדד הטבעי. אבל בטקסט, האורך מטעה.", do=show("x-t"), hold=0.4),
], stage=W.row(cos_plot("cq").replace(' class="h"', ''),
                 st.card("x-t", "trap", "⚠️ בטקסט, אורך המסמך מטעה את אוקלידי"), gap=40),
   panel=f'<div class="qstem"><span class="qid">{Q["id"]}</span>מדוע משתמשים בדמיון קוסינוס במקום '
         'במרחק אוקלידי בין וקטורים של מסמכים?</div>'
         + '<div class="u-qsm">' + st.options("q", OPTS, notes={1: "חישבנו", 2: "אין סדר", 3: "לא משנה ערכים"}) + "</div>")

recap = Scene("recap", "סיכום", [
    P("לסיכום: Tokenization מפרק לטוקנים, ואוצר המילים הוא המילים השונות.", do=show("r-1")),
    P("stop words הן מילים נפוצות בלי מידע על הנושא, ובדרך כלל מסננים אותן.", do=show("r-2")),
    P("Bag of Words סופר כמה פעמים כל מילה הופיעה, ולא שומר איפה.", do=show("r-3")),
    P("מילים עם אותן שכנויות מקבלות וקטורים דומים, וקוסינוס מודד כיוון ולא אורך. נתראה ברמה הבאה.",
      do=show("r-4"), hold=0.6),
], stage=BOWT.replace('id="bw', 'id="bx').replace(' class="h"', ''),
   panel=st.card("r-1", "rule", "🔑 טוקן = מופע; אוצר = מילים שונות")
         + st.card("r-2", "rule", "🔑 stop words: נפוצות, בלי נושא")
         + st.card("r-3", "trap", "⚠️ Bag of Words: ספירה, בלי סדר")
         + st.card("r-4", "rule", "🔑 שכנויות דומות · קוסינוס = כיוון"))

VIDEO = Video("U04", "unsup", "מטקסט לווקטור", [title, tok, bow, ctxs, cosine, exam, recap], bank={"26B-A-Q15": 0})
